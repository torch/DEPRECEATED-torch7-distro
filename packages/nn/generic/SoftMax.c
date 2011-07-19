#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMax.c"
#else

static int nn_(SoftMax_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  real inputMax = THTensor_(max)(input);
  real sum = 0;

  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,                         \
                   real z = THExpMinusApprox(inputMax - *input_data); \
                   *output_data = z;                                  \
                   sum += z;)

  THTensor_(mul)(output, 1/sum);

  return 1;
}

static int nn_(SoftMax_backward)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  real sum;

  sum = THTensor_(dot)(gradOutput, output);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,     \
                   *gradInput_data = *output_data * (*gradOutput_data - sum);)
  return 1;
}

static const struct luaL_Reg nn_(SoftMax__) [] = {
  {"SoftMax_forward", nn_(SoftMax_forward)},
  {"SoftMax_backward", nn_(SoftMax_backward)},
  {NULL, NULL}
};

static void nn_(SoftMax_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SoftMax__), "nn");
  lua_pop(L,1);
}

#endif
