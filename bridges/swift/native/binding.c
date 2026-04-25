#define NAPI_VERSION 8
#include <node_api.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <libgen.h>

extern void mlx_swift_init_metal(void);
extern void mlx_swift_load_model(const char *path, void *context, void (*callback)(void *, bool, int32_t, const char *));
extern int32_t mlx_swift_unload_model(int32_t model_id);
extern void mlx_swift_cancel_generate(int32_t model_id);
extern void mlx_swift_generate_stream(int32_t model_id, const int32_t *prompt_tokens, int32_t prompt_length, const char *config_json, void *context, void (*callback)(void *, const int32_t *, int32_t, bool, bool, const char *));

// Methods following the new bridge_ extern convention
extern char* bridge_metrics(void);

// --- Structs ---
typedef struct {
  napi_env env;
  napi_deferred deferred;
  napi_threadsafe_function tsfn;
} PromiseContext;

typedef struct {
  napi_env env;
  napi_threadsafe_function tsfn;
} StreamContext;

typedef struct {
  bool success;
  int32_t model_id;
  char *payload;
} PromiseResult;

typedef struct {
  int32_t *tokens;
  int32_t count;
  bool is_done;
  bool is_error;
  char *payload;
} StreamResult;

// --- Cleanup Callbacks (Called by V8 Garbage Collector) ---
static void FinalizePromiseContext(napi_env env, void* finalize_data, void* finalize_hint) {
  free(finalize_data);
}

static void FinalizeStreamContext(napi_env env, void* finalize_data, void* finalize_hint) {
  free(finalize_data);
}

// --- Load Callbacks ---
static void CallJsPromise(napi_env env, napi_value js_cb, void *context, void *data) {
  PromiseContext *ctx = (PromiseContext *)context;
  PromiseResult *result = (PromiseResult *)data;
  if (env != NULL) {
    if (result->success) {
      napi_value js_result;
      napi_create_int32(env, result->model_id, &js_result);
      napi_resolve_deferred(env, ctx->deferred, js_result);
    } else {
      napi_value err_code, err_msg, error;
      napi_create_string_utf8(env, "MLX_ERR", NAPI_AUTO_LENGTH, &err_code);
      napi_create_string_utf8(env, result->payload ? result->payload : "Unknown error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &error);
      napi_reject_deferred(env, ctx->deferred, error);
    }
  }
  if (result->payload) free(result->payload);
  free(result);
}

static void SwiftLoadPromiseCallback(void *context, bool success, int32_t model_id, const char *error_msg) {
  PromiseContext *ctx = (PromiseContext *)context;
  PromiseResult *result = malloc(sizeof(PromiseResult));
  result->success = success;
  result->model_id = model_id;
  result->payload = error_msg ? strdup(error_msg) : NULL;

  if (napi_call_threadsafe_function(ctx->tsfn, result, napi_tsfn_nonblocking) != napi_ok) {
    if (result->payload) free(result->payload);
    free(result);
  }
  napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
}

// --- Stream Callbacks ---
static void CallJsStream(napi_env env, napi_value js_cb, void *context, void *data) {
  StreamResult *result = (StreamResult *)data;

  if (env != NULL && js_cb != NULL) {
    napi_value argv[4], global;
    napi_get_global(env, &global);

    if (result->is_error) {
      napi_value err_code, err_msg;
      napi_create_string_utf8(env, "MLX_ERR", NAPI_AUTO_LENGTH, &err_code);
      napi_create_string_utf8(env, result->payload ? result->payload : "Unknown error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &argv[0]);
      napi_get_null(env, &argv[1]);
      napi_get_boolean(env, true, &argv[2]);
      napi_get_null(env, &argv[3]);
    } else {
      napi_get_null(env, &argv[0]);

      if (result->count > 0 && result->tokens != NULL) {
        void* array_data;
        napi_value arraybuffer;
        napi_create_arraybuffer(env, result->count * sizeof(int32_t), &array_data, &arraybuffer);
        memcpy(array_data, result->tokens, result->count * sizeof(int32_t));
        napi_create_typedarray(env, napi_int32_array, result->count, arraybuffer, 0, &argv[1]);
      } else {
        napi_get_null(env, &argv[1]);
      }

      napi_get_boolean(env, result->is_done, &argv[2]);

      if (result->payload != NULL) {
        napi_create_string_utf8(env, result->payload, NAPI_AUTO_LENGTH, &argv[3]);
      } else {
        napi_get_null(env, &argv[3]);
      }
    }
    napi_call_function(env, global, js_cb, 4, argv, NULL);
  }

  // Free data memory
  if (result->tokens) free(result->tokens);
  if (result->payload) free(result->payload);
  free(result);
}

static void SwiftStreamCallback(void *context, const int32_t *tokens, int32_t count, bool is_done, bool is_error, const char *payload) {
  StreamContext *ctx = (StreamContext *)context;
  StreamResult *result = malloc(sizeof(StreamResult));

  result->count = count;
  result->is_done = is_done;
  result->is_error = is_error;
  result->payload = payload ? strdup(payload) : NULL;

  if (count > 0 && tokens != NULL) {
    result->tokens = malloc(count * sizeof(int32_t));
    memcpy(result->tokens, tokens, count * sizeof(int32_t));
  } else {
    result->tokens = NULL;
  }

  // CRITICAL FIX: If queue is closed (e.g. JS aborted abruptly), free memory immediately!
  if (napi_call_threadsafe_function(ctx->tsfn, result, napi_tsfn_nonblocking) != napi_ok) {
    if (result->tokens) free(result->tokens);
    if (result->payload) free(result->payload);
    free(result);
  }

  if (is_done) {
    napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
  }
}

// Metrics Binding (Synchronous)
napi_value Metrics(napi_env env, napi_callback_info info) {
  char *json_str = bridge_metrics();
  napi_value result;

  if (json_str == NULL) {
    napi_create_string_utf8(env, "{}", NAPI_AUTO_LENGTH, &result);
  } else {
    napi_create_string_utf8(env, json_str, NAPI_AUTO_LENGTH, &result);
    free(json_str); // Prevent memory leak!
  }

  return result;
}

// --- API EXPORTS ---
napi_value LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char path[1024];
  size_t result_len;
  napi_get_value_string_utf8(env, args[0], path, sizeof(path), &result_len);

  PromiseContext *ctx = malloc(sizeof(PromiseContext));
  ctx->env = env;
  napi_value promise, resource_name;
  napi_create_promise(env, &ctx->deferred, &promise);
  napi_create_string_utf8(env, "MLXLoad", NAPI_AUTO_LENGTH, &resource_name);

  // Notice the FinalizePromiseContext callback here!
  napi_create_threadsafe_function(env, NULL, NULL, resource_name, 0, 1, NULL, FinalizePromiseContext, ctx, CallJsPromise, &ctx->tsfn);

  mlx_swift_load_model(path, ctx, SwiftLoadPromiseCallback);
  return promise;
}

// UnloadModel & CancelGenerate identical to last iteration...
napi_value UnloadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);
  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);
  int32_t success = mlx_swift_unload_model(model_id);
  napi_value js_result;
  napi_get_boolean(env, success == 1, &js_result);
  return js_result;
}

napi_value CancelGenerate(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);
  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);
  mlx_swift_cancel_generate(model_id);
  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value GenerateStream(napi_env env, napi_callback_info info) {
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

  // DYNAMICALLY ALLOCATE CONFIG_JSON TO SUPPORT ANY LENGTH
  size_t str_len;
  napi_get_value_string_utf8(env, args[2], NULL, 0, &str_len); // pass NULL first to get the exact length of the JSON string
  char* config_json = (char*)malloc(str_len + 1); // malloc the exact size + 1 (for the \0 null terminator)
  napi_get_value_string_utf8(env, args[2], config_json, str_len + 1, &str_len);

  napi_value js_callback = args[3];

  StreamContext *ctx = malloc(sizeof(StreamContext));
  ctx->env = env;
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXStream", NAPI_AUTO_LENGTH, &resource_name);

  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, FinalizeStreamContext, ctx, CallJsStream, &ctx->tsfn);

  mlx_swift_generate_stream(model_id, prompt_tokens, prompt_length, config_json, ctx, SwiftStreamCallback);

  free(config_json); // Swift has already copied it synchronously via String(cString:)

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value init(napi_env env, napi_value exports) {
  // [CWD Metal init hook from before...]
  char old_cwd[1024];
  if (getcwd(old_cwd, sizeof(old_cwd)) != NULL) {
    Dl_info info;
    if (dladdr((void*)init, &info)) {
      char *path_copy = strdup(info.dli_fname);
      chdir(dirname(path_copy));
      mlx_swift_init_metal();
      chdir(old_cwd);
      free(path_copy);
    }
  }

  napi_property_descriptor desc[] = {
      {"load", NULL, LoadModel, NULL, NULL, NULL, napi_default, NULL},
      {"unload", NULL, UnloadModel, NULL, NULL, NULL, napi_default, NULL},
      {"stream", NULL, GenerateStream, NULL, NULL, NULL, napi_default, NULL},
      {"abort", NULL, CancelGenerate, NULL, NULL, NULL, napi_default, NULL},
      {"metrics", NULL, Metrics, NULL, NULL, NULL, napi_default, NULL}
  };
  napi_define_properties(env, exports, 4, desc);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
