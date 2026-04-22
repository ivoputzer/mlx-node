#define NAPI_VERSION 8
#include <node_api.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declarations of our Swift functions
int32_t mlx_swift_load_model(const char *modelName);
char *mlx_swift_generate(const char *prompt);

napi_value LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char model_name[256];
  size_t result;
  napi_get_value_string_utf8(env, args[0], model_name, 256, &result);

  int32_t success = mlx_swift_load_model(model_name);

  napi_value js_result;
  napi_get_boolean(env, success == 1, &js_result);
  return js_result;
}

napi_value Generate(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  char prompt[1024];
  size_t result;
  napi_get_value_string_utf8(env, args[0], prompt, 1024, &result);

  char *output = mlx_swift_generate(prompt);

  napi_value js_result;
  napi_create_string_utf8(env, output, NAPI_AUTO_LENGTH, &js_result);

  free(output); // Clean up the strdup from Swift
  return js_result;
}

napi_value init(napi_env env, napi_value exports) {
  napi_property_descriptor desc[] = {
      {"loadModel", NULL, LoadModel, NULL, NULL, NULL, napi_default, NULL},
      {"generate", NULL, Generate, NULL, NULL, NULL, napi_default, NULL}};
  napi_define_properties(env, exports, 2, desc);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
