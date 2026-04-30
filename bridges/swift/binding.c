#define NAPI_VERSION 8
#include <node_api.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <libgen.h>

// ============================================================================
// EXTERNAL SWIFT FUNCTIONS
// ============================================================================

extern void bridge_metal_load(void);
extern char* bridge_metrics(void);

// Model API
extern void bridge_model_load(const char *path, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_model_free(void *ptr);

// Generation API
extern void bridge_generate_stream(void *model_ptr, const int32_t *prompt_tokens, int32_t prompt_length, const char *config_json, void *context, void (*callback)(void *, const int32_t *, int32_t, bool, bool, const char *));
extern void bridge_generate_abort(void *model_ptr);

// ============================================================================
// TYPE DEFINITIONS (DATA STRUCTURES)
// ============================================================================

// The universal wrapper for any Swift/C++ object exposed to JS
typedef struct {
  void* native_ptr;
  void (*destructor)(void*);
} NativeResource;

// State retained while loading a model asynchronously
typedef struct {
  napi_env env;
  napi_deferred deferred;
  napi_threadsafe_function threadsafe_fn;
} ModelLoadContext;

// Payload sent from Swift to V8 when a model finishes loading
typedef struct {
  bool is_success;
  void *native_ptr;
  char *error_message;
} ModelLoadResult;

// Payload sent from Swift to V8 for every generation step
typedef struct {
  int32_t token_count;
  bool is_done;
  bool is_error;
  size_t payload_len;
  int32_t *tokens;
  const char *payload;
} StreamPayload;

// ============================================================================
// MEMORY MANAGEMENT & GARBAGE COLLECTION
// ============================================================================

// Called manually by JS (resource.dispose) or automatically by GC
static void DestroyNativeResource(NativeResource* resource) {
  if (resource != NULL && resource->native_ptr != NULL) {
    resource->destructor(resource->native_ptr);
    resource->native_ptr = NULL; // Safe tombstone prevents double-free
  }
}

// V8 hook for when a NativeResource object is garbage collected
static void GC_FinalizeNativeResource(napi_env env, void* finalize_data, void* finalize_hint) {
  (void)env; (void)finalize_hint; // Silence unused warnings
  NativeResource* resource = (NativeResource*)finalize_data;
  DestroyNativeResource(resource);
  free(resource);
}

// Universal freer for simple malloc blocks attached to Threadsafe Functions
static void GC_FinalizeMemoryBlock(napi_env env, void* finalize_data, void* finalize_hint) {
  (void)env; (void)finalize_hint;
  if (finalize_data != NULL) free(finalize_data);
}

// ============================================================================
// DOMAIN: MODEL LOADING
// ============================================================================

// 2. Executes on V8 Main Thread to resolve the JS Promise
static void V8_OnModelLoadResolved(napi_env env, napi_value js_callback, void *context, void *data) {
  (void)js_callback; // We resolve a promise, no callback needed
  ModelLoadContext *ctx = (ModelLoadContext *)context;
  ModelLoadResult *result = (ModelLoadResult *)data;

  if (env != NULL && ctx != NULL) {
    if (result->is_success) {
      NativeResource* resource = malloc(sizeof(NativeResource));
      resource->native_ptr = result->native_ptr;
      resource->destructor = bridge_model_free; // Assign Swift destructor

      napi_value js_resource;
      napi_create_external(env, resource, GC_FinalizeNativeResource, NULL, &js_resource);
      napi_resolve_deferred(env, ctx->deferred, js_resource);
    } else {
      napi_value err_code, err_msg, js_error;
      napi_create_string_utf8(env, "MLX_LOAD_ERR", NAPI_AUTO_LENGTH, &err_code);
      napi_create_string_utf8(env, result->error_message ? result->error_message : "Unknown error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &js_error);
      napi_reject_deferred(env, ctx->deferred, js_error);
    }
  }

  if (result->error_message) free(result->error_message);
  free(result);
}

// 1. Called by Swift on a background thread when loading finishes
static void Swift_OnModelLoadCompleted(void *context, bool success, void *native_ptr, const char *error_msg) {
  ModelLoadContext *ctx = (ModelLoadContext *)context;

  ModelLoadResult *result = malloc(sizeof(ModelLoadResult));
  result->is_success = success;
  result->native_ptr = native_ptr;
  result->error_message = error_msg ? strdup(error_msg) : NULL;

  if (napi_call_threadsafe_function(ctx->threadsafe_fn, result, napi_tsfn_nonblocking) != napi_ok) {
    if (result->error_message) free(result->error_message);
    free(result);
  }

  napi_release_threadsafe_function(ctx->threadsafe_fn, napi_tsfn_release);
}

// ============================================================================
// DOMAIN: TEXT GENERATION (STREAMING)
// ============================================================================

// 2. Executes on V8 Main Thread to fire the JS stream callback
static void V8_OnStreamEvent(napi_env env, napi_value js_callback, void *context, void *data) {
  (void)context; // Explicitly silence the unused context warning
  StreamPayload *payload = (StreamPayload *)data;

  if (env != NULL && js_callback != NULL) {
    napi_value argv[4], global, js_null;
    napi_get_global(env, &global);
    napi_get_null(env, &js_null);

    if (payload->is_error) {
      napi_value err_code, err_msg;
      napi_create_string_utf8(env, "MLX_STREAM_ERR", 14, &err_code);
      napi_create_string_utf8(env, payload->payload != NULL ? payload->payload : "Stream error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &argv[0]);
      argv[1] = js_null;
      napi_get_boolean(env, true, &argv[2]);
      argv[3] = js_null;
    } else {
      argv[0] = js_null;
      if (payload->token_count > 0 && payload->tokens != NULL) {
        void* array_data;
        napi_value arraybuffer;
        napi_create_arraybuffer(env, payload->token_count * sizeof(int32_t), &array_data, &arraybuffer);
        memcpy(array_data, payload->tokens, payload->token_count * sizeof(int32_t));
        napi_create_typedarray(env, napi_int32_array, payload->token_count, arraybuffer, 0, &argv[1]);
      } else {
        argv[1] = js_null;
      }
      napi_get_boolean(env, payload->is_done, &argv[2]);
      if (payload->payload != NULL) {
        napi_create_string_utf8(env, payload->payload, payload->payload_len, &argv[3]);
      } else {
        argv[3] = js_null;
      }
    }
    napi_call_function(env, global, js_callback, 4, argv, NULL);
  }
  free(payload);
}

// 1. Called by Swift on a background thread when tokens are generated
static void Swift_OnStreamEvent(void *context, const int32_t *tokens, int32_t count, bool is_done, bool is_error, const char *json_payload) {
  napi_threadsafe_function tsfn = (napi_threadsafe_function)context;

  size_t struct_size = sizeof(StreamPayload);
  size_t tokens_size = count > 0 ? count * sizeof(int32_t) : 0;
  size_t payload_len = json_payload ? strlen(json_payload) : 0;
  size_t payload_bytes = json_payload ? payload_len + 1 : 0;

  void *ptr = malloc(struct_size + tokens_size + payload_bytes);
  if (!ptr) return;

  StreamPayload *payload = (StreamPayload *)ptr;
  payload->token_count = count;
  payload->is_done = is_done;
  payload->is_error = is_error;
  payload->payload_len = payload_len;

  if (tokens_size > 0 && tokens != NULL) {
    payload->tokens = (int32_t *)((char *)ptr + struct_size);
    memcpy(payload->tokens, tokens, tokens_size);
  } else {
    payload->tokens = NULL;
  }

  if (payload_bytes > 0) {
    payload->payload = (char *)ptr + struct_size + tokens_size;
    memcpy((void *)payload->payload, json_payload, payload_bytes);
  } else {
    payload->payload = NULL;
  }

  if (napi_call_threadsafe_function(tsfn, payload, napi_tsfn_nonblocking) != napi_ok) {
    free(payload);
  }

  if (is_done) {
    napi_release_threadsafe_function(tsfn, napi_tsfn_release);
  }
}

// ============================================================================
// N-API EXPORTS (JS BOUNDARY)
// ============================================================================

napi_value Export_ResourceFree(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  if (argc > 0 && napi_get_value_external(env, args[0], (void**)&resource) == napi_ok && resource != NULL) {
    DestroyNativeResource(resource);
    napi_value js_true; napi_get_boolean(env, true, &js_true); return js_true;
  }

  napi_value js_false; napi_get_boolean(env, false, &js_false); return js_false;
}

napi_value Export_ModelLoad(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  size_t path_len;
  napi_get_value_string_utf8(env, args[0], NULL, 0, &path_len);
  char* path_string = (char*)malloc(path_len + 1);
  napi_get_value_string_utf8(env, args[0], path_string, path_len + 1, &path_len);

  ModelLoadContext *ctx = malloc(sizeof(ModelLoadContext));
  ctx->env = env;

  napi_value promise, resource_name;
  napi_create_promise(env, &ctx->deferred, &promise);
  napi_create_string_utf8(env, "MLXModelLoad", NAPI_AUTO_LENGTH, &resource_name);

  // GC_FinalizeMemoryBlock ensures `ctx` is freed if V8 kills the threadsafe function
  napi_create_threadsafe_function(env, NULL, NULL, resource_name, 0, 1, ctx, GC_FinalizeMemoryBlock, ctx, V8_OnModelLoadResolved, &ctx->threadsafe_fn);

  bridge_model_load(path_string, ctx, Swift_OnModelLoadCompleted);
  free(path_string);
  return promise;
}

napi_value Export_ModelGenerate(napi_env env, napi_callback_info info) {
  size_t argc = 4; napi_value args[4];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  if (napi_get_value_external(env, args[0], (void**)&resource) != napi_ok || resource == NULL || resource->native_ptr == NULL) {
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
  napi_create_string_utf8(env, "MLXModelStream", NAPI_AUTO_LENGTH, &resource_name);

  napi_threadsafe_function tsfn;
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnStreamEvent, &tsfn);

  bridge_generate_stream(resource->native_ptr, prompt_tokens, (int32_t)length, config_json, (void*)tsfn, Swift_OnStreamEvent);

  free(config_json);
  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_ModelAbort(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  if (napi_get_value_external(env, args[0], (void**)&resource) != napi_ok) {
    napi_throw_type_error(env, "MLX_ERR", "Argument must be a valid resource handle"); return NULL;
  }

  if (resource != NULL && resource->native_ptr != NULL) {
    bridge_generate_abort(resource->native_ptr);
  }

  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_SystemMetrics(napi_env env, napi_callback_info info) {
  (void)info; // Silence unused warning
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

  // Explicit mappings. JS wrapper handles mapping these to nice object methods.
  napi_property_descriptor desc[] = {
      {"resourceFree", NULL, Export_ResourceFree, NULL, NULL, NULL, napi_default, NULL},
      {"modelLoad", NULL, Export_ModelLoad, NULL, NULL, NULL, napi_default, NULL},
      {"modelGenerate", NULL, Export_ModelGenerate, NULL, NULL, NULL, napi_default, NULL},
      {"modelAbort", NULL, Export_ModelAbort, NULL, NULL, NULL, napi_default, NULL},
      {"systemMetrics", NULL, Export_SystemMetrics, NULL, NULL, NULL, napi_default, NULL}
  };

  napi_define_properties(env, exports, 5, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
