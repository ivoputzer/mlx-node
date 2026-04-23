#define NAPI_VERSION 8
#include <node_api.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <libgen.h>

// Declare the Swift function
extern void mlx_swift_init_metal(void);

typedef struct
{
  napi_env env;
  napi_deferred deferred;
  napi_threadsafe_function tsfn;
} PromiseContext;

typedef struct
{
  napi_env env;
  napi_threadsafe_function tsfn;
} StreamContext;

typedef struct
{
  bool success;
  int32_t model_id;
  char *payload;
} PromiseResult;

typedef struct
{
  char *chunk;
  bool is_done;
  char *error_msg;
} StreamResult;

typedef void (*LoadPromiseCallback)(void *context, bool success, int32_t model_id, const char *error_msg);
typedef void (*GeneratePromiseCallback)(void *context, bool success, const char *payload);
typedef void (*StreamCallback)(void *context, const char *chunk, bool is_done, const char *error_msg);

void mlx_swift_load_model(const char *path, void *context, LoadPromiseCallback callback);
int32_t mlx_swift_unload_model(int32_t model_id);
void mlx_swift_generate(int32_t model_id, const char *prompt, const char *config_json, void *context, GeneratePromiseCallback callback);
void mlx_swift_generate_stream(int32_t model_id, const char *prompt, const char *config_json, void *context, StreamCallback callback);

// --- Callbacks (Omitted standard TSFN boilerplate for brevity, same as previous) ---
static void CallJsPromise(napi_env env, napi_value js_cb, void *context, void *data)
{
  PromiseContext *ctx = (PromiseContext *)context;
  PromiseResult *result = (PromiseResult *)data;
  if (env != NULL)
  {
    if (result->success)
    {
      napi_value js_result;
      if (result->payload)
        napi_create_string_utf8(env, result->payload, NAPI_AUTO_LENGTH, &js_result);
      else
        napi_create_int32(env, result->model_id, &js_result);
      napi_resolve_deferred(env, ctx->deferred, js_result);
    }
    else
    {
      napi_value err_code, err_msg, error;
      napi_create_string_utf8(env, "MLX_ERR", NAPI_AUTO_LENGTH, &err_code);
      napi_create_string_utf8(env, result->payload ? result->payload : "Unknown error", NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &error);
      napi_reject_deferred(env, ctx->deferred, error);
    }
  }
  if (result->payload)
    free(result->payload);
  free(result);
  free(ctx);
}

static void SwiftLoadPromiseCallback(void *context, bool success, int32_t model_id, const char *error_msg)
{
  PromiseContext *ctx = (PromiseContext *)context;
  PromiseResult *result = malloc(sizeof(PromiseResult));
  result->success = success;
  result->model_id = model_id;
  result->payload = error_msg ? strdup(error_msg) : NULL;
  napi_call_threadsafe_function(ctx->tsfn, result, napi_tsfn_nonblocking);
  napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
}

static void SwiftGeneratePromiseCallback(void *context, bool success, const char *payload)
{
  PromiseContext *ctx = (PromiseContext *)context;
  PromiseResult *result = malloc(sizeof(PromiseResult));
  result->success = success;
  result->payload = payload ? strdup(payload) : NULL;
  napi_call_threadsafe_function(ctx->tsfn, result, napi_tsfn_nonblocking);
  napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
}

static void CallJsStream(napi_env env, napi_value js_cb, void *context, void *data)
{
  StreamContext *ctx = (StreamContext *)context;
  StreamResult *result = (StreamResult *)data;
  if (env != NULL && js_cb != NULL)
  {
    napi_value argv[3], global;
    napi_get_global(env, &global);
    if (result->error_msg)
    {
      napi_create_string_utf8(env, result->error_msg, NAPI_AUTO_LENGTH, &argv[0]);
      napi_get_null(env, &argv[1]);
      napi_get_boolean(env, true, &argv[2]);
    }
    else if (result->is_done)
    {
      napi_get_null(env, &argv[0]);
      napi_get_null(env, &argv[1]);
      napi_get_boolean(env, true, &argv[2]);
    }
    else
    {
      napi_get_null(env, &argv[0]);
      napi_create_string_utf8(env, result->chunk ? result->chunk : "", NAPI_AUTO_LENGTH, &argv[1]);
      napi_get_boolean(env, false, &argv[2]);
    }
    napi_call_function(env, global, js_cb, 3, argv, NULL);
  }
  if (result->chunk)
    free(result->chunk);
  if (result->error_msg)
    free(result->error_msg);
  free(result);
  if (data && (((StreamResult *)data)->is_done || ((StreamResult *)data)->error_msg))
    free(ctx);
}

static void SwiftStreamCallback(void *context, const char *chunk, bool is_done, const char *error_msg)
{
  StreamContext *ctx = (StreamContext *)context;
  StreamResult *result = malloc(sizeof(StreamResult));
  result->chunk = chunk ? strdup(chunk) : NULL;
  result->is_done = is_done;
  result->error_msg = error_msg ? strdup(error_msg) : NULL;
  napi_call_threadsafe_function(ctx->tsfn, result, napi_tsfn_nonblocking);
  if (is_done || error_msg != NULL)
    napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
}

// --- API EXPORTS ---
napi_value LoadModel(napi_env env, napi_callback_info info)
{
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char path[1024];
  size_t result;
  napi_get_value_string_utf8(env, args[0], path, sizeof(path), &result);

  PromiseContext *ctx = malloc(sizeof(PromiseContext));
  ctx->env = env;
  napi_value promise;
  napi_create_promise(env, &ctx->deferred, &promise);
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXLoadModel", NAPI_AUTO_LENGTH, &resource_name);
  napi_create_threadsafe_function(env, NULL, NULL, resource_name, 0, 1, NULL, NULL, ctx, CallJsPromise, &ctx->tsfn);

  mlx_swift_load_model(path, ctx, SwiftLoadPromiseCallback);
  return promise;
}

napi_value UnloadModel(napi_env env, napi_callback_info info)
{
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

napi_value Generate(napi_env env, napi_callback_info info)
{
  size_t argc = 3;
  napi_value args[3];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);

  char prompt[32768];
  size_t result;
  napi_get_value_string_utf8(env, args[1], prompt, sizeof(prompt), &result);

  char config_json[4096];
  napi_get_value_string_utf8(env, args[2], config_json, sizeof(config_json), &result);

  PromiseContext *ctx = malloc(sizeof(PromiseContext));
  ctx->env = env;
  napi_value promise;
  napi_create_promise(env, &ctx->deferred, &promise);
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXGenerate", NAPI_AUTO_LENGTH, &resource_name);
  napi_create_threadsafe_function(env, NULL, NULL, resource_name, 0, 1, NULL, NULL, ctx, CallJsPromise, &ctx->tsfn);

  mlx_swift_generate(model_id, prompt, config_json, ctx, SwiftGeneratePromiseCallback);
  return promise;
}

napi_value GenerateStream(napi_env env, napi_callback_info info)
{
  size_t argc = 4;
  napi_value args[4];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t model_id;
  napi_get_value_int32(env, args[0], &model_id);

  char prompt[32768];
  size_t result;
  napi_get_value_string_utf8(env, args[1], prompt, sizeof(prompt), &result);

  char config_json[4096];
  napi_get_value_string_utf8(env, args[2], config_json, sizeof(config_json), &result);

  napi_value js_callback = args[3];

  StreamContext *ctx = malloc(sizeof(StreamContext));
  ctx->env = env;
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXStream", NAPI_AUTO_LENGTH, &resource_name);
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, ctx, CallJsStream, &ctx->tsfn);

  mlx_swift_generate_stream(model_id, prompt, config_json, ctx, SwiftStreamCallback);
  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value init(napi_env env, napi_value exports)
{
  // --- SYNCHRONOUS CWD HOOK ---
  char old_cwd[1024];
  if (getcwd(old_cwd, sizeof(old_cwd)) != NULL) {
    Dl_info info;
    // dladdr gives us the absolute path to this mlx_swift.node binary
    if (dladdr((void*)init, &info)) {
      char *path_copy = strdup(info.dli_fname);

      // 1. Jump to the package directory where default.metallib lives
      chdir(dirname(path_copy));

      // 2. Force MLX to boot up. It uses CWD fallback, finds the file, and caches it.
      mlx_swift_init_metal();

      // 3. Jump immediately back to the user's project CWD
      chdir(old_cwd);
      free(path_copy);
    }
  }
  // --- END HOOK ---

  napi_property_descriptor desc[] = {
      {"loadModel", NULL, LoadModel, NULL, NULL, NULL, napi_default, NULL},
      {"unloadModel", NULL, UnloadModel, NULL, NULL, NULL, napi_default, NULL},
      {"generate", NULL, Generate, NULL, NULL, NULL, napi_default, NULL},
      {"generateStream", NULL, GenerateStream, NULL, NULL, NULL, napi_default, NULL}};
  napi_define_properties(env, exports, 4, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
