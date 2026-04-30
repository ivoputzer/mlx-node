#define NAPI_VERSION 8
#include <node_api.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <libgen.h>

// --- Swift Bridge Externs ---
extern void bridge_metal_load(void);
extern void bridge_model_load(const char *path, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_model_free(void *ptr);

extern void bridge_generate_stream(void *model_ptr, const int32_t *prompt_tokens, int32_t prompt_length, const char *config_json, void *context, void (*callback)(void *, const int32_t *, int32_t, bool, bool, const char *));
extern void bridge_generate_abort(void *model_ptr);
extern char* bridge_metrics(void);

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

typedef struct {
  void* swift_ptr;
  void (*free_func)(void*);
} ResourceHandle;

typedef struct {
  napi_env env;
  napi_deferred deferred;
  napi_threadsafe_function threadsafe_fn;
} ModelLoadContext;

typedef struct {
  bool is_success;
  void *model_ptr;
  char *error_message;
} ModelLoadEventData;

typedef struct {
  int32_t token_count;
  bool is_done;
  bool is_error;
  size_t payload_len;
  int32_t *tokens;
  const char *payload;
} StreamEventData;

// ============================================================================
// UNIVERSAL RESOURCE MANAGEMENT
// ============================================================================

static void FreeResource(ResourceHandle* handle) {
  if (handle != NULL && handle->swift_ptr != NULL) {
    handle->free_func(handle->swift_ptr);
    handle->swift_ptr = NULL; // Safe tombstone prevents double-free
  }
}

static void FinalizeResource(napi_env env, void* finalize_data, void* finalize_hint) {
  ResourceHandle* handle = (ResourceHandle*)finalize_data;
  FreeResource(handle);
  free(handle); // Free the C struct
}

static void FinalizeModelLoadContext(napi_env env, void* finalize_data, void* finalize_hint) {
  if (finalize_data != NULL) free(finalize_data);
}

// ============================================================================
// MODEL LOADING PIPELINE
// ============================================================================

static void ResolveModelLoadOnMainThread(napi_env env, napi_value js_callback, void *context, void *data) {
  ModelLoadContext *load_ctx = (ModelLoadContext *)context;
  ModelLoadEventData *event_data = (ModelLoadEventData *)data;

  if (env != NULL && load_ctx != NULL) {
    if (event_data->is_success) {
      ResourceHandle* handle = malloc(sizeof(ResourceHandle));
      handle->swift_ptr = event_data->model_ptr;
      handle->free_func = bridge_model_free; // Assign the Swift function pointer!

      napi_value js_handle;
      napi_create_external(env, handle, FinalizeResource, NULL, &js_handle);
      napi_resolve_deferred(env, load_ctx->deferred, js_handle);
    } else {
      napi_value err_code, err_msg, js_error;
      napi_create_string_utf8(env, "MLX_LOAD_ERR", NAPI_AUTO_LENGTH, &err_code);
      napi_create_string_utf8(env, event_data->error_message ? event_data->error_message : "Unknown loading error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &js_error);
      napi_reject_deferred(env, load_ctx->deferred, js_error);
    }
  }

  if (event_data->error_message) free(event_data->error_message);
  free(event_data);
}

static void OnModelLoadCompleted(void *context, bool success, void *model_ptr, const char *error_msg) {
  ModelLoadContext *load_ctx = (ModelLoadContext *)context;

  ModelLoadEventData *event_data = malloc(sizeof(ModelLoadEventData));
  event_data->is_success = success;
  event_data->model_ptr = model_ptr;
  event_data->error_message = error_msg ? strdup(error_msg) : NULL;

  if (napi_call_threadsafe_function(load_ctx->threadsafe_fn, event_data, napi_tsfn_nonblocking) != napi_ok) {
    if (event_data->error_message) free(event_data->error_message);
    free(event_data);
  }

  napi_release_threadsafe_function(load_ctx->threadsafe_fn, napi_tsfn_release);
}

// ============================================================================
// TEXT GENERATION STREAM PIPELINE
// ============================================================================

// Main Thread Emitter (Fixme: `void *context` isn't used anymore)
static void EmitStreamEventOnMainThread(napi_env env, napi_value js_callback, void *context, void *data) {
  StreamEventData *event_data = (StreamEventData *)data;

  if (env != NULL && js_callback != NULL) {
    napi_value argv[4], global, js_null;
    napi_get_global(env, &global);
    napi_get_null(env, &js_null);

    if (event_data->is_error) {
      napi_value err_code, err_msg;
      napi_create_string_utf8(env, "MLX_STREAM_ERR", 14, &err_code);
      napi_create_string_utf8(env, event_data->payload != NULL ? event_data->payload : "Unknown stream error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &argv[0]);
      argv[1] = js_null;
      napi_get_boolean(env, true, &argv[2]);
      argv[3] = js_null;
    } else {
      argv[0] = js_null;
      if (event_data->token_count > 0 && event_data->tokens != NULL) {
        void* array_data;
        napi_value arraybuffer;
        napi_create_arraybuffer(env, event_data->token_count * sizeof(int32_t), &array_data, &arraybuffer);
        memcpy(array_data, event_data->tokens, event_data->token_count * sizeof(int32_t));
        napi_create_typedarray(env, napi_int32_array, event_data->token_count, arraybuffer, 0, &argv[1]);
      } else {
        argv[1] = js_null;
      }
      napi_get_boolean(env, event_data->is_done, &argv[2]);
      if (event_data->payload != NULL) {
        napi_create_string_utf8(env, event_data->payload, event_data->payload_len, &argv[3]);
      } else {
        argv[3] = js_null;
      }
    }
    napi_call_function(env, global, js_callback, 4, argv, NULL);
  }
  free(event_data);
}

static void OnStreamEventReceived(void *context, const int32_t *tokens, int32_t count, bool is_done, bool is_error, const char *payload) {
  napi_threadsafe_function tsfn = (napi_threadsafe_function)context; // <-- MAGIC

  size_t struct_size = sizeof(StreamEventData);
  size_t tokens_size = count > 0 ? count * sizeof(int32_t) : 0;
  size_t payload_len = payload ? strlen(payload) : 0;
  size_t payload_bytes = payload ? payload_len + 1 : 0;

  void *ptr = malloc(struct_size + tokens_size + payload_bytes);
  if (!ptr) return;

  StreamEventData *event_data = (StreamEventData *)ptr;
  event_data->token_count = count;
  event_data->is_done = is_done;
  event_data->is_error = is_error;
  event_data->payload_len = payload_len;

  if (tokens_size > 0 && tokens != NULL) {
    event_data->tokens = (int32_t *)((char *)ptr + struct_size);
    memcpy(event_data->tokens, tokens, tokens_size);
  } else {
    event_data->tokens = NULL;
  }

  if (payload_bytes > 0) {
    event_data->payload = (char *)ptr + struct_size + tokens_size;
    memcpy((void *)event_data->payload, payload, payload_bytes);
  } else {
    event_data->payload = NULL;
  }

  // Pass tsfn directly. If V8 has torn down, this safely returns != napi_ok
  if (napi_call_threadsafe_function(tsfn, event_data, napi_tsfn_nonblocking) != napi_ok) {
    free(event_data);
  }

  if (is_done) {
    napi_release_threadsafe_function(tsfn, napi_tsfn_release);
  }
}

// ============================================================================
// JAVASCRIPT API EXPORTS
// ============================================================================

napi_value Export_LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  size_t path_len;
  napi_get_value_string_utf8(env, args[0], NULL, 0, &path_len);
  char* path_string = (char*)malloc(path_len + 1);
  napi_get_value_string_utf8(env, args[0], path_string, path_len + 1, &path_len);

  ModelLoadContext *load_ctx = malloc(sizeof(ModelLoadContext));
  load_ctx->env = env;

  napi_value promise, resource_name;
  napi_create_promise(env, &load_ctx->deferred, &promise);
  napi_create_string_utf8(env, "MLXLoadModel", NAPI_AUTO_LENGTH, &resource_name);

  napi_create_threadsafe_function(env, NULL, NULL, resource_name, 0, 1, load_ctx, FinalizeModelLoadContext, load_ctx, ResolveModelLoadOnMainThread, &load_ctx->threadsafe_fn);

  bridge_model_load(path_string, load_ctx, OnModelLoadCompleted);
  free(path_string);
  return promise;
}

// Replaces both Export_FreeModel and Export_UnloadModel
napi_value Export_Free(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  if (argc < 1) {
    napi_value js_result; napi_get_boolean(env, false, &js_result); return js_result;
  }

  ResourceHandle* handle;
  napi_status status = napi_get_value_external(env, args[0], (void**)&handle);

  if (status != napi_ok) {
    napi_value js_result; napi_get_boolean(env, false, &js_result); return js_result;
  }

  if (handle != NULL && handle->swift_ptr != NULL) {
    FreeResource(handle);
    napi_value js_result; napi_get_boolean(env, true, &js_result); return js_result;
  }

  napi_value js_result; napi_get_boolean(env, false, &js_result); return js_result;
}

napi_value Export_AbortGeneration(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  if (argc < 1) {
    napi_throw_type_error(env, "MLX_ERR", "Model handle argument is required"); return NULL;
  }

  ResourceHandle* handle;
  napi_status status = napi_get_value_external(env, args[0], (void**)&handle);

  if (status != napi_ok) {
    napi_throw_type_error(env, "MLX_ERR", "Argument must be a valid model handle"); return NULL;
  }

  if (handle != NULL && handle->swift_ptr != NULL) {
    bridge_generate_abort(handle->swift_ptr);
  }

  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_GenerateStream(napi_env env, napi_callback_info info) {
  size_t argc = 4; napi_value args[4];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  ResourceHandle* handle;
  napi_status status = napi_get_value_external(env, args[0], (void**)&handle);

  if (status != napi_ok || handle == NULL || handle->swift_ptr == NULL) {
    napi_throw_type_error(env, "MLX_ERR", "Model is already unloaded or invalid"); return NULL;
  }

  napi_typedarray_type type; size_t length; void* data; size_t byte_offset;
  napi_get_typedarray_info(env, args[1], &type, &length, &data, NULL, &byte_offset);

  int32_t* prompt_tokens = (int32_t*)((char*)data + byte_offset);

  size_t json_len;
  napi_get_value_string_utf8(env, args[2], NULL, 0, &json_len);
  char* config_json = (char*)malloc(json_len + 1);
  napi_get_value_string_utf8(env, args[2], config_json, json_len + 1, &json_len);

  napi_value js_callback = args[3];

  napi_value resource_name;
  napi_create_string_utf8(env, "MLXStreamGeneration", NAPI_AUTO_LENGTH, &resource_name);

  // We don't need a struct context anymore, just pass NULLs
  napi_threadsafe_function tsfn;
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, EmitStreamEventOnMainThread, &tsfn);

  // Pass tsfn natively to Swift!
  bridge_generate_stream(handle->swift_ptr, prompt_tokens, (int32_t)length, config_json, (void*)tsfn, OnStreamEventReceived);

  free(config_json);
  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_Metrics(napi_env env, napi_callback_info info) {
  char *json_str = bridge_metrics();
  napi_value result;
  if (json_str == NULL) {
    napi_create_string_utf8(env, "{}", NAPI_AUTO_LENGTH, &result);
  } else {
    napi_create_string_utf8(env, json_str, NAPI_AUTO_LENGTH, &result);
    free(json_str);
  }
  return result;
}

// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

napi_value init(napi_env env, napi_value exports) {
  char old_cwd[1024];
  if (getcwd(old_cwd, sizeof(old_cwd)) != NULL) {
    Dl_info info;
    if (dladdr((void*)init, &info)) {
      char *path_copy = strdup(info.dli_fname);
      chdir(dirname(path_copy));
      bridge_metal_load();
      chdir(old_cwd);
      free(path_copy);
    }
  }

  napi_property_descriptor desc[] = {
      {"load", NULL, Export_LoadModel, NULL, NULL, NULL, napi_default, NULL},
      {"free", NULL, Export_Free, NULL, NULL, NULL, napi_default, NULL}, // Renamed to free
      {"stream", NULL, Export_GenerateStream, NULL, NULL, NULL, napi_default, NULL},
      {"abort", NULL, Export_AbortGeneration, NULL, NULL, NULL, napi_default, NULL},
      {"metrics", NULL, Export_Metrics, NULL, NULL, NULL, napi_default, NULL}
  };

  napi_define_properties(env, exports, 5, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
