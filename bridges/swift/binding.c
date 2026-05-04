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
extern void bridge_metal_clear_cache(void);
extern char* bridge_metal_metrics(void);

extern void bridge_model_load(const char *path, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_model_free(void *ptr);

extern void* bridge_cache_create(void* model_ptr, const char* config_json);
extern void bridge_cache_free(void* ptr);
extern void* bridge_cache_clone(void* ptr);
extern void bridge_cache_save(void* ptr, const char* path, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_cache_load(const char* path, void *context, void (*callback)(void *, bool, void *, const char *));
extern int32_t bridge_cache_trim(void* ptr, int32_t num_tokens);

extern void* bridge_cache_slice(void* ptr, int32_t start, int32_t end);
extern char* bridge_cache_debug(void* ptr);

extern void* bridge_model_generate_task(void* model_ptr, void* cache_ptr, const int32_t* prompt_tokens, int32_t prompt_length, const char* config_json, void* context, void (*callback)(void*, const int32_t*, int32_t, bool, bool, const char*));
extern void* bridge_model_evaluate_task(void* model_ptr, void* cache_ptr, const int32_t* prompt_tokens, int32_t prompt_length, const char* config_json, void* context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_model_abort_task(void* ptr);
extern void bridge_model_free_task(void* ptr);


// ============================================================================
// TYPE DEFINITIONS (DATA STRUCTURES)
// ============================================================================

// The universal wrapper for any Swift/C++ object exposed to JS
typedef struct {
  void* native_ptr;
  void (*destructor)(void*);
} NativeResource;

// Universal async payload
typedef struct {
  bool is_success;
  void *native_ptr;
  char *string_data; // <-- Generic: Holds Error Msg OR JSON Payload
  void (*destructor)(void*);
} AsyncPayload;

typedef struct {
  napi_threadsafe_function tsfn;
  void (*destructor)(void*);
} AsyncContext;

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
// UNIVERSAL ASYNC PIPELINE (For Disk I/O & Heavy Operations)
// ============================================================================
static void V8_OnAsyncComplete(napi_env env, napi_value js_callback, void *context, void *data) {
  AsyncPayload *payload = (AsyncPayload *)data;
  napi_value argv[2], global, js_null;
  napi_get_global(env, &global);
  napi_get_null(env, &js_null);

  if (payload->is_success) {
    argv[0] = js_null; // err = null

    if (payload->native_ptr) {
      // Path 1: Returns a Resource (e.g. LoadModel, LoadCache)
      NativeResource* resource = malloc(sizeof(NativeResource));
      resource->native_ptr = payload->native_ptr;
      resource->destructor = payload->destructor;
      napi_create_external(env, resource, GC_FinalizeNativeResource, NULL, &argv[1]);
    } else if (payload->string_data) {
      // Path 2: Returns a JSON String (e.g. EvaluateTask)
      napi_create_string_utf8(env, payload->string_data, NAPI_AUTO_LENGTH, &argv[1]);
    } else {
      // Path 3: Returns Void (e.g. SaveCache)
      argv[1] = js_null;
    }
  } else {
    // Error Path
    napi_value err_code, err_msg;
    napi_create_string_utf8(env, "MLX_ERR", NAPI_AUTO_LENGTH, &err_code);
    napi_create_string_utf8(env, payload->string_data ? payload->string_data : "Unknown error", NAPI_AUTO_LENGTH, &err_msg);
    napi_create_error(env, err_code, err_msg, &argv[0]);
    argv[1] = js_null;
  }

  napi_call_function(env, global, js_callback, 2, argv, NULL);
  if (payload->string_data) free(payload->string_data);
  free(payload);
}

static void Swift_OnAsyncComplete(void *context, bool success, void *native_ptr, const char *string_data) {
  AsyncContext *ctx = (AsyncContext *)context;
  AsyncPayload *payload = malloc(sizeof(AsyncPayload));
  payload->is_success = success;
  payload->native_ptr = native_ptr;
  payload->string_data = string_data ? strdup(string_data) : NULL;
  payload->destructor = ctx->destructor;

  if (napi_call_threadsafe_function(ctx->tsfn, payload, napi_tsfn_nonblocking) != napi_ok) {
    if (payload->string_data) free(payload->string_data);
    free(payload);
  }
  napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
  free(ctx);
}

// ============================================================================
// DOMAIN: MODEL LOADING
// ============================================================================

// 2. Executes on V8 Main Thread to fire the JS callback: callback(err, ref)
static void V8_OnModelLoadCallback(napi_env env, napi_value js_callback, void *context, void *data) {
  (void)context;
  AsyncPayload *payload = (AsyncPayload *)data;

  if (env != NULL && js_callback != NULL) {
    napi_value argv[2], global, js_null;
    napi_get_global(env, &global);
    napi_get_null(env, &js_null);

    if (payload->is_success) {
      NativeResource* resource = malloc(sizeof(NativeResource));
      resource->native_ptr = payload->native_ptr;
      resource->destructor = bridge_model_free;

      napi_value js_resource;
      napi_create_external(env, resource, GC_FinalizeNativeResource, NULL, &js_resource);

      argv[0] = js_null; // err = null
      argv[1] = js_resource; // ref = object
    } else {
      napi_value err_code, err_msg;
      napi_create_string_utf8(env, "MLX_LOAD_ERR", NAPI_AUTO_LENGTH, &err_code);
      const char* error_str = payload->string_data ? payload->string_data : "Error loading model";
      napi_create_string_utf8(env, error_str, NAPI_AUTO_LENGTH, &err_msg);
      napi_create_error(env, err_code, err_msg, &argv[0]);
      argv[1] = js_null;
    }

    napi_call_function(env, global, js_callback, 2, argv, NULL);
  }

  if (payload->string_data) free(payload->string_data);
  free(payload);
}

// 1. Called by Swift on a background thread when loading finishes
static void Swift_OnModelLoadCompleted(void *context, bool success, void *native_ptr, const char *error_msg) {
  napi_threadsafe_function tsfn = (napi_threadsafe_function)context;

  AsyncPayload *payload = malloc(sizeof(AsyncPayload));
  payload->is_success = success;
  payload->native_ptr = native_ptr;
  payload->string_data = error_msg ? strdup(error_msg) : NULL;

  if (napi_call_threadsafe_function(tsfn, payload, napi_tsfn_nonblocking) != napi_ok) {
    if (payload->string_data) free(payload->string_data);
    free(payload);
  }

  napi_release_threadsafe_function(tsfn, napi_tsfn_release);
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
  size_t argc = 2; napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  size_t path_len;
  napi_get_value_string_utf8(env, args[0], NULL, 0, &path_len);
  char* path_string = (char*)malloc(path_len + 1);
  napi_get_value_string_utf8(env, args[0], path_string, path_len + 1, &path_len);

  napi_value js_callback = args[1];
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXModelLoad", NAPI_AUTO_LENGTH, &resource_name);

  AsyncContext* ctx = malloc(sizeof(AsyncContext));
  ctx->destructor = bridge_model_free;
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnAsyncComplete, &ctx->tsfn);
  bridge_model_load(path_string, ctx, Swift_OnAsyncComplete);

  free(path_string);

  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_CacheCreate(napi_env env, napi_callback_info info) {
  size_t argc = 2; napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* model_res;
  napi_get_value_external(env, args[0], (void**)&model_res);

  size_t json_len;
  napi_get_value_string_utf8(env, args[1], NULL, 0, &json_len);
  char* config_json = (char*)malloc(json_len + 1);
  napi_get_value_string_utf8(env, args[1], config_json, json_len + 1, &json_len);

  void* cache_ptr = bridge_cache_create(model_res->native_ptr, config_json);
  free(config_json);

  NativeResource* resource = malloc(sizeof(NativeResource));
  resource->native_ptr = cache_ptr;
  resource->destructor = bridge_cache_free;

  napi_value js_resource;
  napi_create_external(env, resource, GC_FinalizeNativeResource, NULL, &js_resource);
  return js_resource;
}

napi_value Export_CacheClone(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* orig_res;
  if (napi_get_value_external(env, args[0], (void**)&orig_res) != napi_ok || !orig_res || !orig_res->native_ptr) {
    napi_throw_type_error(env, "MLX_ERR", "Invalid cache pointer"); return NULL;
  }

  void* cloned_ptr = bridge_cache_clone(orig_res->native_ptr);
  NativeResource* new_res = malloc(sizeof(NativeResource));
  new_res->native_ptr = cloned_ptr;
  new_res->destructor = bridge_cache_free;

  napi_value js_resource;
  napi_create_external(env, new_res, GC_FinalizeNativeResource, NULL, &js_resource);
  return js_resource;
}

napi_value Export_CacheSave(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* cache_res;
  napi_get_value_external(env, args[0], (void**)&cache_res);

  size_t path_len;
  napi_get_value_string_utf8(env, args[1], NULL, 0, &path_len);
  char* path_string = (char*)malloc(path_len + 1);
  napi_get_value_string_utf8(env, args[1], path_string, path_len + 1, &path_len);

  napi_value js_callback = args[2];
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXCacheSave", NAPI_AUTO_LENGTH, &resource_name);

  AsyncContext* ctx = (AsyncContext*)malloc(sizeof(AsyncContext));
  ctx->destructor = NULL; // Save doesn't return a new object to be wrapped

  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnAsyncComplete, &ctx->tsfn);
  bridge_cache_save(cache_res->native_ptr, path_string, ctx, Swift_OnAsyncComplete);

  free(path_string);

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_CacheLoad(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  size_t path_len;
  napi_get_value_string_utf8(env, args[0], NULL, 0, &path_len);
  char* path_string = (char*)malloc(path_len + 1);
  napi_get_value_string_utf8(env, args[0], path_string, path_len + 1, &path_len);

  napi_value js_callback = args[1];
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXCacheLoad", NAPI_AUTO_LENGTH, &resource_name);

  AsyncContext* ctx = (AsyncContext*)malloc(sizeof(AsyncContext));
  ctx->destructor = bridge_cache_free;

  napi_create_threadsafe_function(env,js_callback,NULL,resource_name,0,1,NULL,NULL,NULL,V8_OnAsyncComplete,&ctx->tsfn);
  bridge_cache_load(path_string, ctx, Swift_OnAsyncComplete);
  free(path_string);

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_CacheTrim(napi_env env, napi_callback_info info) {
  size_t argc = 2; napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  napi_get_value_external(env, args[0], (void**)&resource);

  int32_t num_tokens;
  napi_get_value_int32(env, args[1], &num_tokens);

  int32_t actual_trimmed = bridge_cache_trim(resource->native_ptr, num_tokens);

  napi_value result;
  napi_create_int32(env, actual_trimmed, &result);
  return result;
}

napi_value Export_CacheSlice(napi_env env, napi_callback_info info) {
  size_t argc = 3; napi_value args[3];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* orig_res;
  napi_get_value_external(env, args[0], (void**)&orig_res);

  int32_t start, end;
  napi_get_value_int32(env, args[1], &start);
  napi_get_value_int32(env, args[2], &end);

  void* sliced_ptr = bridge_cache_slice(orig_res->native_ptr, start, end);
  NativeResource* new_res = malloc(sizeof(NativeResource));
  new_res->native_ptr = sliced_ptr;
  new_res->destructor = bridge_cache_free;

  napi_value js_resource;
  napi_create_external(env, new_res, GC_FinalizeNativeResource, NULL, &js_resource);
  return js_resource;
}

napi_value Export_CacheDebug(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  napi_get_value_external(env, args[0], (void**)&resource);

  char* json_str = bridge_cache_debug(resource->native_ptr);
  napi_value result;
  napi_create_string_utf8(env, json_str, NAPI_AUTO_LENGTH, &result);
  free(json_str);
  return result;
}

napi_value Export_ModelGenerateTask(napi_env env, napi_callback_info info) {
  size_t argc = 5; napi_value args[5]; // model, cache, tokens, json, callback
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* model_res;
  if (napi_get_value_external(env, args[0], (void**)&model_res) != napi_ok || !model_res || !model_res->native_ptr) {
    napi_throw_type_error(env, "MLX_ERR", "Model is already unloaded or invalid"); return NULL;
  }

  void* cache_ptr = NULL;
  napi_valuetype cache_type;
  napi_typeof(env, args[1], &cache_type);
  if (cache_type == napi_external) {
    NativeResource* cache_res;
    napi_get_value_external(env, args[1], (void**)&cache_res);
    if (cache_res) cache_ptr = cache_res->native_ptr;
  }

  napi_typedarray_type type; size_t length; void* data; size_t byte_offset;
  napi_get_typedarray_info(env, args[2], &type, &length, &data, NULL, &byte_offset);
  int32_t* prompt_tokens = (int32_t*)((char*)data + byte_offset);

  size_t json_len;
  napi_get_value_string_utf8(env, args[3], NULL, 0, &json_len);
  char* config_json = (char*)malloc(json_len + 1);
  napi_get_value_string_utf8(env, args[3], config_json, json_len + 1, &json_len);

  napi_value js_callback = args[4];
  napi_value resource_name;
  napi_create_string_utf8(env, "MLXModelStream", NAPI_AUTO_LENGTH, &resource_name);

  napi_threadsafe_function tsfn;
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnStreamEvent, &tsfn);

  void* stream_ptr = bridge_model_generate_task(model_res->native_ptr, cache_ptr, prompt_tokens, (int32_t)length, config_json, (void*)tsfn, Swift_OnStreamEvent);
  free(config_json);

  NativeResource* stream_res = malloc(sizeof(NativeResource));
  stream_res->native_ptr = stream_ptr;
  stream_res->destructor = bridge_model_free_task;

  napi_value js_stream_res;
  napi_create_external(env, stream_res, GC_FinalizeNativeResource, NULL, &js_stream_res);
  return js_stream_res;
}

napi_value Export_ModelEvaluateTask(napi_env env, napi_callback_info info) {
  size_t argc = 5; napi_value args[5]; // model, cache, tokens, json, callback
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  // 1. SAFELY extract Model Pointer
  NativeResource* model_res;
  if (napi_get_value_external(env, args[0], (void**)&model_res) != napi_ok || !model_res || !model_res->native_ptr) {
    napi_throw_type_error(env, "MLX_ERR", "Model is already unloaded or invalid"); return NULL;
  }

  // 2. SAFELY extract OPTIONAL Cache Pointer
  void* cache_ptr = NULL;
  napi_valuetype cache_type;
  napi_typeof(env, args[1], &cache_type);
  if (cache_type == napi_external) {
    NativeResource* cache_res;
    napi_get_value_external(env, args[1], (void**)&cache_res);
    if (cache_res) cache_ptr = cache_res->native_ptr;
  }

  // 3. Extract Tokens
  napi_typedarray_type type; size_t length; void* data; size_t byte_offset;
  napi_get_typedarray_info(env, args[2], &type, &length, &data, NULL, &byte_offset);
  int32_t* prompt_tokens = (int32_t*)((char*)data + byte_offset);

  // 4. Extract Config JSON
  size_t json_len;
  napi_get_value_string_utf8(env, args[3], NULL, 0, &json_len);
  char* config_json = (char*)malloc(json_len + 1);
  napi_get_value_string_utf8(env, args[3], config_json, json_len + 1, &json_len);

  // 5. Create Async Pipeline
  napi_value resource_name; napi_create_string_utf8(env, "MLXEvaluateTask", NAPI_AUTO_LENGTH, &resource_name);

  AsyncContext* ctx = malloc(sizeof(AsyncContext));
  ctx->destructor = NULL;
  napi_create_threadsafe_function(env, args[4], NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnAsyncComplete, &ctx->tsfn);

  // 6. Execute Swift Function
  void* task_ptr = bridge_model_evaluate_task(model_res->native_ptr, cache_ptr, prompt_tokens, (int32_t)length, config_json, ctx, Swift_OnAsyncComplete);
  free(config_json);

  // 7. Return Task Handle
  NativeResource* task_res = malloc(sizeof(NativeResource));
  task_res->native_ptr = task_ptr;
  task_res->destructor = bridge_model_free_task;

  napi_value js_task_res;
  napi_create_external(env, task_res, GC_FinalizeNativeResource, NULL, &js_task_res);
  return js_task_res;
}

napi_value Export_ModelAbort(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  if (napi_get_value_external(env, args[0], (void**)&resource) != napi_ok) {
    napi_throw_type_error(env, "MLX_ERR", "Argument must be a valid resource handle"); return NULL;
  }

  if (resource != NULL && resource->native_ptr != NULL) {
    bridge_model_abort_task(resource->native_ptr);
  }

  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_ModelAbortTask(napi_env env, napi_callback_info info) {
  size_t argc = 1; napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  NativeResource* resource;
  if (napi_get_value_external(env, args[0], (void**)&resource) == napi_ok && resource && resource->native_ptr) {
    bridge_model_abort_task(resource->native_ptr);
  }
  napi_value undefined; napi_get_undefined(env, &undefined); return undefined;
}

napi_value Export_SystemMetrics(napi_env env, napi_callback_info info) {
  (void)info; // Silence unused warning
  char *json_str = bridge_metal_metrics();
  napi_value result;
  if (json_str == NULL) {
    napi_create_string_utf8(env, "{}", NAPI_AUTO_LENGTH, &result);
  } else {
    napi_create_string_utf8(env, json_str, NAPI_AUTO_LENGTH, &result);
    free(json_str);
  }
  return result;
}

napi_value Export_SystemClearCache(napi_env env, napi_callback_info info) {
  bridge_metal_clear_cache();
  napi_value undefined; napi_get_undefined(env, &undefined);
  return undefined;
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
      {"freeResource", NULL, Export_ResourceFree, NULL, NULL, NULL, napi_default, NULL},

      {"createCache", NULL, Export_CacheCreate, NULL, NULL, NULL, napi_default, NULL},
      {"loadCache", NULL, Export_CacheLoad, NULL, NULL, NULL, napi_default, NULL},
      {"saveCache", NULL, Export_CacheSave, NULL, NULL, NULL, napi_default, NULL},
      {"cloneCache", NULL, Export_CacheClone, NULL, NULL, NULL, napi_default, NULL},
      {"trimCache", NULL, Export_CacheTrim, NULL, NULL, NULL, napi_default, NULL},
      {"sliceCache", NULL, Export_CacheSlice, NULL, NULL, NULL, napi_default, NULL},
      {"debugCache", NULL, Export_CacheDebug, NULL, NULL, NULL, napi_default, NULL},

      {"loadModel", NULL, Export_ModelLoad, NULL, NULL, NULL, napi_default, NULL},
      {"generateTask", NULL, Export_ModelGenerateTask, NULL, NULL, NULL, napi_default, NULL},
      {"evaluateTask", NULL, Export_ModelEvaluateTask, NULL, NULL, NULL, napi_default, NULL},
      {"abortTask", NULL, Export_ModelAbortTask, NULL, NULL, NULL, napi_default, NULL},

      {"systemMetrics", NULL, Export_SystemMetrics, NULL, NULL, NULL, napi_default, NULL},
      {"systemClearCache", NULL, Export_SystemClearCache, NULL, NULL, NULL, napi_default, NULL},
  };

  napi_define_properties(env, exports, 13, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
