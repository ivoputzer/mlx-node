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
// 1. SWIFT EXTERNS (THE NATIVE BRIDGE)
// ============================================================================

extern void bridge_metal_load(void);
extern void bridge_metal_clear_cache(void);
extern char *bridge_metal_metrics(void);

extern void bridge_model_load(const char *path, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_model_free(void *ptr);

extern void *bridge_cache_create(void *model_ptr, const char *config_json);
extern void bridge_cache_free(void *ptr);
extern void *bridge_cache_clone(void *ptr);
extern void *bridge_cache_slice(void *ptr, int32_t start, int32_t end);
extern void bridge_cache_save(void *ptr, const char *path, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_cache_load(const char *path, void *context, void (*callback)(void *, bool, void *, const char *));
extern int32_t bridge_cache_trim(void *ptr, int32_t num_tokens);
extern char *bridge_cache_debug(void *ptr);

extern void *bridge_model_generate_task(void *model_ptr, void *cache_ptr, const int32_t *prompt_tokens, int32_t prompt_length, const char *config_json, void *context, void (*callback)(void *, const int32_t *, int32_t, const int32_t *, const float *, int32_t, bool, bool, const char *));
extern void *bridge_model_batch_task(void *model_ptr, void *cache_ptr, const int32_t *flat_tokens, int32_t max_len, int32_t batch_size, const char *config_json, void *context, void (*callback)(void *, const int32_t *, int32_t, const int32_t *, const float *, int32_t, bool, bool, const char *));
extern void *bridge_model_evaluate_task(void *model_ptr, void *cache_ptr, const int32_t *prompt_tokens, int32_t prompt_length, const char *config_json, void *context, void (*callback)(void *, bool, void *, const char *));
extern void bridge_model_abort_task(void *ptr);
extern void bridge_model_free_task(void *ptr);

// ============================================================================
// 2. DATA STRUCTURES
// ============================================================================

typedef struct
{
  void *native_ptr;
  void (*destructor)(void *);
} NativeResource;

typedef struct
{
  bool is_success;
  void *native_ptr;
  char *string_data;
  void (*destructor)(void *);
} AsyncPayload;

typedef struct
{
  napi_threadsafe_function tsfn;
  void (*destructor)(void *);
} AsyncContext;

typedef struct
{
  int32_t token_count;
  bool is_done;
  bool is_error;
  size_t payload_len;
  int32_t *tokens;
  int32_t *top_tokens;
  float *top_probs;
  int32_t top_k;
  const char *payload;
} StreamPayload;

// ============================================================================
// 3. MEMORY MANAGEMENT & GARBAGE COLLECTION
// ============================================================================

static void DestroyNativeResource(NativeResource *resource)
{
  if (resource != NULL && resource->native_ptr != NULL)
  {
    resource->destructor(resource->native_ptr);
    resource->native_ptr = NULL; // Safe tombstone prevents double-free
  }
}

static void GC_FinalizeNativeResource(napi_env env, void *finalize_data, void *finalize_hint)
{
  (void)env;
  (void)finalize_hint;
  NativeResource *resource = (NativeResource *)finalize_data;
  DestroyNativeResource(resource);
  free(resource);
}

// ============================================================================
// 4. N-API UTILITIES (DRY HELPERS)
// ============================================================================

// Wrap a Raw C Pointer into a JS External object with GC hooks
static napi_value NAPI_WrapPointer(napi_env env, void *native_ptr, void (*destructor)(void *))
{
  if (!native_ptr)
    return NULL;
  NativeResource *resource = malloc(sizeof(NativeResource));
  resource->native_ptr = native_ptr;
  resource->destructor = destructor;
  napi_value js_resource;
  napi_create_external(env, resource, GC_FinalizeNativeResource, NULL, &js_resource);
  return js_resource;
}

// Extract the Raw C Pointer from a JS External
static void *NAPI_ExtractPointer(napi_env env, napi_value js_external)
{
  napi_valuetype type;
  napi_typeof(env, js_external, &type);
  if (type != napi_external)
    return NULL;

  NativeResource *resource;
  if (napi_get_value_external(env, js_external, (void **)&resource) == napi_ok && resource != NULL)
  {
    return resource->native_ptr;
  }
  return NULL;
}

// Extract a JS String into a malloc'd C string (Caller must free())
static char *NAPI_ExtractString(napi_env env, napi_value js_string)
{
  size_t len;
  if (napi_get_value_string_utf8(env, js_string, NULL, 0, &len) != napi_ok)
    return NULL;
  char *result = (char *)malloc(len + 1);
  napi_get_value_string_utf8(env, js_string, result, len + 1, &len);
  return result;
}

// Extract a JS Int32Array into a C pointer
static int32_t *NAPI_ExtractInt32Array(napi_env env, napi_value js_array, int32_t *out_length)
{
  napi_typedarray_type type;
  size_t length;
  void *data;
  size_t byte_offset;
  if (napi_get_typedarray_info(env, js_array, &type, &length, &data, NULL, &byte_offset) == napi_ok)
  {
    if (out_length)
      *out_length = (int32_t)length;
    return (int32_t *)((char *)data + byte_offset);
  }
  return NULL;
}

// Fetch arguments uniformly
static bool NAPI_GetArgs(napi_env env, napi_callback_info info, size_t expected_count, napi_value *out_args)
{
  size_t argc = expected_count;
  return napi_get_cb_info(env, info, &argc, out_args, NULL, NULL) == napi_ok && argc >= expected_count;
}

// Convert C String to JS String
static napi_value NAPI_CreateString(napi_env env, const char *str)
{
  napi_value result;
  napi_create_string_utf8(env, str ? str : "", NAPI_AUTO_LENGTH, &result);
  return result;
}

// ============================================================================
// 5. ASYNC PIPELINE (DISPATCH & CALLBACKS)
// ============================================================================

static void V8_OnAsyncComplete(napi_env env, napi_value js_callback, void *context, void *data)
{
  AsyncPayload *payload = (AsyncPayload *)data;
  napi_value argv[2], global, js_null;
  napi_get_global(env, &global);
  napi_get_null(env, &js_null);

  if (payload->is_success)
  {
    argv[0] = js_null;
    if (payload->native_ptr)
    {
      argv[1] = NAPI_WrapPointer(env, payload->native_ptr, payload->destructor);
    }
    else if (payload->string_data)
    {
      argv[1] = NAPI_CreateString(env, payload->string_data);
    }
    else
    {
      argv[1] = js_null;
    }
  }
  else
  {
    napi_value err_code = NAPI_CreateString(env, "MLX_ERR");
    napi_value err_msg = NAPI_CreateString(env, payload->string_data ? payload->string_data : "Unknown error");
    napi_create_error(env, err_code, err_msg, &argv[0]);
    argv[1] = js_null;
  }

  napi_call_function(env, global, js_callback, 2, argv, NULL);
  if (payload->string_data)
    free(payload->string_data);
  free(payload);
}

static void Swift_OnAsyncComplete(void *context, bool success, void *native_ptr, const char *string_data)
{
  AsyncContext *ctx = (AsyncContext *)context;
  AsyncPayload *payload = malloc(sizeof(AsyncPayload));
  payload->is_success = success;
  payload->native_ptr = native_ptr;
  payload->string_data = string_data ? strdup(string_data) : NULL;
  payload->destructor = ctx->destructor;

  if (napi_call_threadsafe_function(ctx->tsfn, payload, napi_tsfn_nonblocking) != napi_ok)
  {
    if (payload->string_data)
      free(payload->string_data);
    free(payload);
  }
  napi_release_threadsafe_function(ctx->tsfn, napi_tsfn_release);
  free(ctx);
}

static AsyncContext *SetupAsyncPipeline(napi_env env, napi_value js_callback, const char *name, void (*destructor)(void *))
{
  AsyncContext *ctx = malloc(sizeof(AsyncContext));
  ctx->destructor = destructor;
  napi_value resource_name = NAPI_CreateString(env, name);
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnAsyncComplete, &ctx->tsfn);
  return ctx;
}

// ============================================================================
// 6. STREAM PIPELINE (GENERATION TASKS)
// ============================================================================

// V8 Garbage Collection Hook: Triggered when V8 destroys the External ArrayBuffer
static void GC_FinalizeStreamPayload(napi_env env, void *finalize_data, void *finalize_hint)
{
  size_t total_size = (size_t)finalize_hint;
  free(finalize_data);

  // Tell V8 we released the memory!
  int64_t adjusted_memory;
  napi_adjust_external_memory(env, -(int64_t)total_size, &adjusted_memory);
}

static void V8_OnStreamEvent(napi_env env, napi_value js_callback, void *context, void *data)
{
  (void)context;
  StreamPayload *payload = (StreamPayload *)data;
  bool v8_owns_memory = false; // Track if V8 took ownership of the pointer

  if (env != NULL && js_callback != NULL)
  {
    napi_value argv[7], global, js_null;
    napi_get_global(env, &global);
    napi_get_null(env, &js_null);

    if (payload->is_error)
    {
      napi_value err_code = NAPI_CreateString(env, "MLX_STREAM_ERR");
      napi_value err_msg = NAPI_CreateString(env, payload->payload ? payload->payload : "Stream error");
      napi_create_error(env, err_code, err_msg, &argv[0]);
      argv[1] = js_null; // tokens
      argv[2] = js_null; // topTokens
      argv[3] = js_null; // topProbs
      napi_create_int32(env, 0, &argv[4]); // topK
      napi_get_boolean(env, true, &argv[5]); // done
      argv[6] = js_null; // json
    }
    else
    {
      argv[0] = js_null;

      if (payload->token_count > 0 && payload->tokens != NULL)
      {
        // 1. Calculate the exact total size of our malloc block
        size_t struct_size = sizeof(StreamPayload);
        size_t tokens_size = payload->token_count * sizeof(int32_t);
        size_t top_tokens_size = payload->top_tokens ? (payload->token_count * payload->top_k * sizeof(int32_t)) : 0;
        size_t top_probs_size = payload->top_probs ? (payload->token_count * payload->top_k * sizeof(float)) : 0;
        size_t payload_bytes = payload->payload ? payload->payload_len + 1 : 0;
        size_t total_size = struct_size + tokens_size + top_tokens_size + top_probs_size + payload_bytes;

        // 2. Zero-Copy: Hand the entire C-pointer directly to V8
        napi_value v8_buffer;
        napi_create_external_arraybuffer(env, payload, total_size, GC_FinalizeStreamPayload, (void*)total_size, &v8_buffer);
        v8_owns_memory = true; // V8 is now responsible for free() via GC!

        // Tell V8 we just took up `total_size` bytes of native memory!
        int64_t adjusted_memory;
        napi_adjust_external_memory(env, (int64_t)total_size, &adjusted_memory);

        // 3. Create TypedArray Views mapping EXACTLY to our C-memory offsets
        size_t tokens_offset = (char *)payload->tokens - (char *)payload;
        napi_create_typedarray(env, napi_int32_array, payload->token_count, v8_buffer, tokens_offset, &argv[1]);

        if (payload->top_k > 0 && payload->top_tokens != NULL)
        {
          size_t top_tokens_offset = (char *)payload->top_tokens - (char *)payload;
          napi_create_typedarray(env, napi_int32_array, payload->token_count * payload->top_k, v8_buffer, top_tokens_offset, &argv[2]);

          size_t top_probs_offset = (char *)payload->top_probs - (char *)payload;
          napi_create_typedarray(env, napi_float32_array, payload->token_count * payload->top_k, v8_buffer, top_probs_offset, &argv[3]);
        }
        else
        {
          argv[2] = js_null;
          argv[3] = js_null;
        }
      }
      else
      {
        argv[1] = js_null;
        argv[2] = js_null;
        argv[3] = js_null;
      }

      napi_create_int32(env, payload->top_k, &argv[4]);
      napi_get_boolean(env, payload->is_done, &argv[5]);
      argv[6] = payload->payload ? NAPI_CreateString(env, payload->payload) : js_null;
    }
    napi_call_function(env, global, js_callback, 7, argv, NULL);
  }

  // If V8 didn't take ownership (e.g., error state or no tokens), free manually to prevent leaks
  if (!v8_owns_memory)
  {
    free(payload);
  }
}

static void Swift_OnStreamEvent(void *context, const int32_t *tokens, int32_t count, const int32_t *top_tokens, const float *top_probs, int32_t top_k, bool is_done, bool is_error, const char *json_payload)
{
  napi_threadsafe_function tsfn = (napi_threadsafe_function)context;

  size_t struct_size = sizeof(StreamPayload);
  size_t tokens_size = count > 0 ? count * sizeof(int32_t) : 0;
  size_t top_tokens_size = top_tokens ? (count * top_k * sizeof(int32_t)) : 0;
  size_t top_probs_size = top_probs ? (count * top_k * sizeof(float)) : 0;
  size_t payload_len = json_payload ? strlen(json_payload) : 0;
  size_t payload_bytes = json_payload ? payload_len + 1 : 0;

  void *ptr = malloc(struct_size + tokens_size + top_tokens_size + top_probs_size + payload_bytes);
  if (!ptr) return;

  StreamPayload *payload = (StreamPayload *)ptr;
  payload->token_count = count;
  payload->top_k = top_k;
  payload->is_done = is_done;
  payload->is_error = is_error;
  payload->payload_len = payload_len;

  char *cursor = (char *)ptr + struct_size;

  if (tokens_size > 0) {
    payload->tokens = (int32_t *)cursor;
    memcpy(payload->tokens, tokens, tokens_size);
    cursor += tokens_size;
  } else { payload->tokens = NULL; }

  if (top_tokens_size > 0) {
    payload->top_tokens = (int32_t *)cursor;
    memcpy(payload->top_tokens, top_tokens, top_tokens_size);
    cursor += top_tokens_size;

    payload->top_probs = (float *)cursor;
    memcpy(payload->top_probs, top_probs, top_probs_size);
    cursor += top_probs_size;
  } else {
    payload->top_tokens = NULL;
    payload->top_probs = NULL;
  }

  if (payload_bytes > 0)
  {
    payload->payload = cursor;
    memcpy((void *)payload->payload, json_payload, payload_bytes);
  }
  else
  {
    payload->payload = NULL;
  }

  if (napi_call_threadsafe_function(tsfn, payload, napi_tsfn_nonblocking) != napi_ok)
    free(payload);

  if (is_done)
    napi_release_threadsafe_function(tsfn, napi_tsfn_release);
}

static napi_threadsafe_function SetupStreamPipeline(napi_env env, napi_value js_callback, const char *name)
{
  napi_threadsafe_function tsfn;
  napi_value resource_name = NAPI_CreateString(env, name);
  napi_create_threadsafe_function(env, js_callback, NULL, resource_name, 0, 1, NULL, NULL, NULL, V8_OnStreamEvent, &tsfn);
  return tsfn;
}

// ============================================================================
// 7. EXPORTED JS FUNCTIONS
// ============================================================================

napi_value Export_ResourceFree(napi_env env, napi_callback_info info)
{
  napi_value args[1];
  if (!NAPI_GetArgs(env, info, 1, args))
    return NULL;

  NativeResource *resource;
  if (napi_get_value_external(env, args[0], (void **)&resource) == napi_ok && resource != NULL)
  {
    DestroyNativeResource(resource);
    napi_value js_true;
    napi_get_boolean(env, true, &js_true);
    return js_true;
  }
  napi_value js_false;
  napi_get_boolean(env, false, &js_false);
  return js_false;
}

napi_value Export_ModelLoad(napi_env env, napi_callback_info info)
{
  napi_value args[2];
  if (!NAPI_GetArgs(env, info, 2, args))
    return NULL;

  char *path = NAPI_ExtractString(env, args[0]);
  AsyncContext *ctx = SetupAsyncPipeline(env, args[1], "MLXModelLoad", bridge_model_free);

  bridge_model_load(path, ctx, Swift_OnAsyncComplete);
  free(path);

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_CacheCreate(napi_env env, napi_callback_info info)
{
  napi_value args[2];
  if (!NAPI_GetArgs(env, info, 2, args))
    return NULL;

  void *model_ptr = NAPI_ExtractPointer(env, args[0]);
  char *config_json = NAPI_ExtractString(env, args[1]);

  void *cache_ptr = bridge_cache_create(model_ptr, config_json);
  free(config_json);

  return NAPI_WrapPointer(env, cache_ptr, bridge_cache_free);
}

napi_value Export_CacheClone(napi_env env, napi_callback_info info)
{
  napi_value args[1];
  if (!NAPI_GetArgs(env, info, 1, args))
    return NULL;

  void *orig_ptr = NAPI_ExtractPointer(env, args[0]);
  if (!orig_ptr)
    return NULL;

  void *cloned_ptr = bridge_cache_clone(orig_ptr);
  return NAPI_WrapPointer(env, cloned_ptr, bridge_cache_free);
}

napi_value Export_CacheSlice(napi_env env, napi_callback_info info)
{
  napi_value args[3];
  if (!NAPI_GetArgs(env, info, 3, args))
    return NULL;

  void *orig_ptr = NAPI_ExtractPointer(env, args[0]);
  int32_t start, end;
  napi_get_value_int32(env, args[1], &start);
  napi_get_value_int32(env, args[2], &end);

  void *sliced_ptr = bridge_cache_slice(orig_ptr, start, end);
  return NAPI_WrapPointer(env, sliced_ptr, bridge_cache_free);
}

napi_value Export_CacheSave(napi_env env, napi_callback_info info)
{
  napi_value args[3];
  if (!NAPI_GetArgs(env, info, 3, args))
    return NULL;

  void *cache_ptr = NAPI_ExtractPointer(env, args[0]);
  char *path = NAPI_ExtractString(env, args[1]);
  AsyncContext *ctx = SetupAsyncPipeline(env, args[2], "MLXCacheSave", NULL);

  bridge_cache_save(cache_ptr, path, ctx, Swift_OnAsyncComplete);
  free(path);

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_CacheLoad(napi_env env, napi_callback_info info)
{
  napi_value args[2];
  if (!NAPI_GetArgs(env, info, 2, args))
    return NULL;

  char *path = NAPI_ExtractString(env, args[0]);
  AsyncContext *ctx = SetupAsyncPipeline(env, args[1], "MLXCacheLoad", bridge_cache_free);

  bridge_cache_load(path, ctx, Swift_OnAsyncComplete);
  free(path);

  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_CacheTrim(napi_env env, napi_callback_info info)
{
  napi_value args[2];
  if (!NAPI_GetArgs(env, info, 2, args))
    return NULL;

  void *cache_ptr = NAPI_ExtractPointer(env, args[0]);
  int32_t num_tokens;
  napi_get_value_int32(env, args[1], &num_tokens);

  int32_t actual_trimmed = bridge_cache_trim(cache_ptr, num_tokens);
  napi_value result;
  napi_create_int32(env, actual_trimmed, &result);
  return result;
}

napi_value Export_CacheDebug(napi_env env, napi_callback_info info)
{
  napi_value args[1];
  if (!NAPI_GetArgs(env, info, 1, args))
    return NULL;

  void *cache_ptr = NAPI_ExtractPointer(env, args[0]);
  char *json_str = bridge_cache_debug(cache_ptr);
  napi_value result = NAPI_CreateString(env, json_str);
  free(json_str);
  return result;
}

napi_value Export_ModelGenerateTask(napi_env env, napi_callback_info info)
{
  napi_value args[5];
  if (!NAPI_GetArgs(env, info, 5, args))
    return NULL;

  void *model_ptr = NAPI_ExtractPointer(env, args[0]);
  void *cache_ptr = NAPI_ExtractPointer(env, args[1]);

  int32_t length;
  int32_t *prompt_tokens = NAPI_ExtractInt32Array(env, args[2], &length);
  char *config_json = NAPI_ExtractString(env, args[3]);
  napi_threadsafe_function tsfn = SetupStreamPipeline(env, args[4], "MLXModelStream");

  void *task_ptr = bridge_model_generate_task(model_ptr, cache_ptr, prompt_tokens, length, config_json, (void *)tsfn, Swift_OnStreamEvent);
  free(config_json);

  return NAPI_WrapPointer(env, task_ptr, bridge_model_free_task);
}

napi_value Export_ModelBatchTask(napi_env env, napi_callback_info info)
{
  napi_value args[7];
  if (!NAPI_GetArgs(env, info, 7, args))
    return NULL;

  void *model_ptr = NAPI_ExtractPointer(env, args[0]);
  void *cache_ptr = NAPI_ExtractPointer(env, args[1]);

  int32_t total_length;
  int32_t *flat_tokens = NAPI_ExtractInt32Array(env, args[2], &total_length);

  int32_t max_len, batch_size;
  napi_get_value_int32(env, args[3], &max_len);
  napi_get_value_int32(env, args[4], &batch_size);

  char *config_json = NAPI_ExtractString(env, args[5]);
  napi_threadsafe_function tsfn = SetupStreamPipeline(env, args[6], "MLXModelBatchStream");

  void *task_ptr = bridge_model_batch_task(model_ptr, cache_ptr, flat_tokens, max_len, batch_size, config_json, (void *)tsfn, Swift_OnStreamEvent);
  free(config_json);

  return NAPI_WrapPointer(env, task_ptr, bridge_model_free_task);
}

napi_value Export_ModelEvaluateTask(napi_env env, napi_callback_info info)
{
  napi_value args[5];
  if (!NAPI_GetArgs(env, info, 5, args))
    return NULL;

  void *model_ptr = NAPI_ExtractPointer(env, args[0]);
  void *cache_ptr = NAPI_ExtractPointer(env, args[1]);

  int32_t length;
  int32_t *prompt_tokens = NAPI_ExtractInt32Array(env, args[2], &length);
  char *config_json = NAPI_ExtractString(env, args[3]);

  AsyncContext *ctx = SetupAsyncPipeline(env, args[4], "MLXEvaluateTask", NULL);
  void *task_ptr = bridge_model_evaluate_task(model_ptr, cache_ptr, prompt_tokens, length, config_json, ctx, Swift_OnAsyncComplete);
  free(config_json);

  return NAPI_WrapPointer(env, task_ptr, bridge_model_free_task);
}

napi_value Export_TaskAbort(napi_env env, napi_callback_info info)
{
  napi_value args[1];
  if (NAPI_GetArgs(env, info, 1, args))
  {
    void *task_ptr = NAPI_ExtractPointer(env, args[0]);
    if (task_ptr)
      bridge_model_abort_task(task_ptr);
  }
  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

napi_value Export_SystemMetrics(napi_env env, napi_callback_info info)
{
  (void)info;
  char *json_str = bridge_metal_metrics();
  napi_value result = NAPI_CreateString(env, json_str ? json_str : "{}");
  if (json_str)
    free(json_str);
  return result;
}

napi_value Export_SystemClearCache(napi_env env, napi_callback_info info)
{
  (void)info;
  bridge_metal_clear_cache();
  napi_value undefined;
  napi_get_undefined(env, &undefined);
  return undefined;
}

// ============================================================================
// 8. MODULE INITIALIZATION
// ============================================================================

napi_value init(napi_env env, napi_value exports)
{
  char old_cwd[1024];
  if (getcwd(old_cwd, sizeof(old_cwd)) != NULL)
  {
    Dl_info info;
    if (dladdr((void *)init, &info))
    {
      char *path_copy = strdup(info.dli_fname);
      chdir(dirname(path_copy));
      bridge_metal_load();
      chdir(old_cwd);
      free(path_copy);
    }
  }

  napi_value package_version;
  napi_create_string_utf8(env, NPM_PACKAGE_VERSION, NAPI_AUTO_LENGTH, &package_version);

  napi_property_descriptor desc[] = {
      {"version", NULL, NULL, NULL, NULL, package_version, napi_enumerable, NULL},
      {"freeResource", NULL, Export_ResourceFree, NULL, NULL, NULL, napi_default, NULL},

      {"createCache", NULL, Export_CacheCreate, NULL, NULL, NULL, napi_default, NULL},
      {"loadCache", NULL, Export_CacheLoad, NULL, NULL, NULL, napi_default, NULL},
      {"saveCache", NULL, Export_CacheSave, NULL, NULL, NULL, napi_default, NULL},
      {"cloneCache", NULL, Export_CacheClone, NULL, NULL, NULL, napi_default, NULL},
      {"sliceCache", NULL, Export_CacheSlice, NULL, NULL, NULL, napi_default, NULL},
      {"trimCache", NULL, Export_CacheTrim, NULL, NULL, NULL, napi_default, NULL},
      {"debugCache", NULL, Export_CacheDebug, NULL, NULL, NULL, napi_default, NULL},

      {"loadModel", NULL, Export_ModelLoad, NULL, NULL, NULL, napi_default, NULL},
      {"generateTask", NULL, Export_ModelGenerateTask, NULL, NULL, NULL, napi_default, NULL},
      {"batchTask", NULL, Export_ModelBatchTask, NULL, NULL, NULL, napi_default, NULL},
      {"evaluateTask", NULL, Export_ModelEvaluateTask, NULL, NULL, NULL, napi_default, NULL},
      {"abortTask", NULL, Export_TaskAbort, NULL, NULL, NULL, napi_default, NULL},

      {"systemMetrics", NULL, Export_SystemMetrics, NULL, NULL, NULL, napi_default, NULL},
      {"systemClearCache", NULL, Export_SystemClearCache, NULL, NULL, NULL, napi_default, NULL},
  };

  napi_define_properties(env, exports, 16, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
