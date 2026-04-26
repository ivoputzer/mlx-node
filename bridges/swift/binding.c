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
extern void bridge_model_load(const char *path, void *context, void (*callback)(void *, bool, int32_t, const char *));
extern int32_t bridge_model_unload(int32_t model_id);
extern void bridge_generate_abort(int32_t model_id);
extern void bridge_generate_stream(int32_t model_id, const int32_t *prompt_tokens, int32_t prompt_length, const char *config_json, void *context, void (*callback)(void *, const int32_t *, int32_t, bool, bool, const char *));
extern char* bridge_metrics(void);


// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

typedef struct {
  napi_env env;
  napi_deferred deferred;
  napi_threadsafe_function threadsafe_fn;
} ModelLoadContext;

typedef struct {
  napi_env env;
  napi_threadsafe_function threadsafe_fn;
} GenerationStreamContext;

typedef struct {
  bool is_success;
  int32_t model_id;
  char *error_message;
} ModelLoadEventData;

typedef struct {
  int32_t *tokens;
  int32_t token_count;
  bool is_done;
  bool is_error;
  char *payload;
} StreamEventData;


// ============================================================================
// V8 GARBAGE COLLECTION FINALIZERS
// ============================================================================

static void FinalizeModelLoadContext(napi_env env, void* finalize_data, void* finalize_hint) {
  if (finalize_data != NULL) free(finalize_data);
}

static void FinalizeGenerationStreamContext(napi_env env, void* finalize_data, void* finalize_hint) {
  if (finalize_data != NULL) free(finalize_data);
}


// ============================================================================
// MODEL LOADING PIPELINE
// ============================================================================

// 2. Executes on the V8 Main Thread to resolve the JavaScript Promise
static void ResolveModelLoadOnMainThread(napi_env env, napi_value js_callback, void *context, void *data) {
  ModelLoadContext *load_ctx = (ModelLoadContext *)context;
  ModelLoadEventData *event_data = (ModelLoadEventData *)data;

  if (env != NULL && load_ctx != NULL) {
    if (event_data->is_success) {
      napi_value js_model_id;
      napi_create_int32(env, event_data->model_id, &js_model_id);
      napi_resolve_deferred(env, load_ctx->deferred, js_model_id);
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

// 1. Called by Swift from a background thread when loading finishes
static void OnModelLoadCompleted(void *context, bool success, int32_t model_id, const char *error_msg) {
  ModelLoadContext *load_ctx = (ModelLoadContext *)context;

  ModelLoadEventData *event_data = malloc(sizeof(ModelLoadEventData));
  event_data->is_success = success;
  event_data->model_id = model_id;
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

// 2. Executes on the V8 Main Thread to fire the JavaScript callback
static void EmitStreamEventOnMainThread(napi_env env, napi_value js_callback, void *context, void *data) {
  StreamEventData *event_data = (StreamEventData *)data;

  if (env != NULL && js_callback != NULL) {
    napi_value argv[4], global;
    napi_get_global(env, &global);

    if (event_data->is_error) {
      napi_value err_code, err_msg;
      napi_create_string_utf8(env, "MLX_STREAM_ERR", NAPI_AUTO_LENGTH, &err_code);
      napi_create_string_utf8(env, event_data->payload ? event_data->payload : "Unknown stream error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &argv[0]);
      napi_get_null(env, &argv[1]);
      napi_get_boolean(env, true, &argv[2]);
      napi_get_null(env, &argv[3]);
    } else {
      napi_get_null(env, &argv[0]); // No Error

      if (event_data->token_count > 0 && event_data->tokens != NULL) {
        void* array_data;
        napi_value arraybuffer;
        napi_create_arraybuffer(env, event_data->token_count * sizeof(int32_t), &array_data, &arraybuffer);
        memcpy(array_data, event_data->tokens, event_data->token_count * sizeof(int32_t));
        napi_create_typedarray(env, napi_int32_array, event_data->token_count, arraybuffer, 0, &argv[1]);
      } else {
        napi_get_null(env, &argv[1]); // No Tokens
      }

      napi_get_boolean(env, event_data->is_done, &argv[2]);

      if (event_data->payload != NULL) {
        napi_create_string_utf8(env, event_data->payload, NAPI_AUTO_LENGTH, &argv[3]);
      } else {
        napi_get_null(env, &argv[3]); // No Payload/Stats
      }
    }

    napi_call_function(env, global, js_callback, 4, argv, NULL);
  }

  if (event_data->tokens) free(event_data->tokens);
  if (event_data->payload) free(event_data->payload);
  free(event_data);
}

// 1. Called by Swift from a background thread when tokens/stats are yielded
static void OnStreamEventReceived(void *context, const int32_t *tokens, int32_t count, bool is_done, bool is_error, const char *payload) {
  GenerationStreamContext *stream_ctx = (GenerationStreamContext *)context;

  StreamEventData *event_data = malloc(sizeof(StreamEventData));
  event_data->token_count = count;
  event_data->is_done = is_done;
  event_data->is_error = is_error;
  event_data->payload = payload ? strdup(payload) : NULL;

  if (count > 0 && tokens != NULL) {
    event_data->tokens = malloc(count * sizeof(int32_t));
    memcpy(event_data->tokens, tokens, count * sizeof(int32_t));
  } else {
    event_data->tokens = NULL;
  }

  if (napi_call_threadsafe_function(stream_ctx->threadsafe_fn, event_data, napi_tsfn_nonblocking) != napi_ok) {
    if (event_data->tokens) free(event_data->tokens);
    if (event_data->payload) free(event_data->payload);
    free(event_data);
  }

  if (is_done) {
    napi_release_threadsafe_function(stream_ctx->threadsafe_fn, napi_tsfn_release);
  }
}


// ============================================================================
// JAVASCRIPT API EXPORTS
// ============================================================================

napi_value Export_LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  // Dynamic allocation for unbounded file paths
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

  free(path_string); // Swift copies this synchronously, safe to free immediately
  return promise;
}

napi_value Export_UnloadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);

  int32_t success_flag = bridge_model_unload(model_id);

  napi_value js_result;
  napi_get_boolean(env, success_flag == 1, &js_result);
  return js_result;
}

napi_value Export_AbortGeneration(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);

  bridge_generate_abort(model_id);

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_GenerateStream(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);

  napi_typedarray_type type;
  size_t length;
  void* data;
  napi_value arraybuffer;
  size_t byte_offset;
  napi_get_typedarray_info(env, args[1], &type, &length, &data, &arraybuffer, &byte_offset);

  int32_t* prompt_tokens = (int32_t*)((char*)data + byte_offset);
  int32_t prompt_length = (int32_t)length;

  // Dynamic allocation for unbounded configuration payloads
  size_t json_len;
  napi_get_value_string_utf8(env, args[2], NULL, 0, &json_len);
  char* config_json = (char*)malloc(json_len + 1);
  napi_get_value_string_utf8(env, args[2], config_json, json_len + 1, &json_len);

  napi_value js_callback = args[3];

  GenerationStreamContext *stream_ctx = malloc(sizeof(GenerationStreamContext));
  stream_ctx->env = env;

  napi_value resource_name;
  napi_create_string_utf8(env, "MLXStreamGeneration", NAPI_AUTO_LENGTH, &resource_name);

  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, stream_ctx, FinalizeGenerationStreamContext, stream_ctx, EmitStreamEventOnMainThread, &stream_ctx->threadsafe_fn);

  bridge_generate_stream(model_id, prompt_tokens, prompt_length, config_json, stream_ctx, OnStreamEventReceived);

  free(config_json); // Swift copies this synchronously, safe to free immediately

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
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
  // Execute macOS Metal initialization in the correct framework directory
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
      {"unload", NULL, Export_UnloadModel, NULL, NULL, NULL, napi_default, NULL},
      {"stream", NULL, Export_GenerateStream, NULL, NULL, NULL, napi_default, NULL},
      {"abort", NULL, Export_AbortGeneration, NULL, NULL, NULL, napi_default, NULL},
      {"metrics", NULL, Export_Metrics, NULL, NULL, NULL, napi_default, NULL}
  };

  napi_define_properties(env, exports, 5, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
