#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSoftMax.c"
#else

static int nn_(LogSoftMax_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  real maxInput = THTensor_(max)(input);
	accreal logsum = 0;

  TH_TENSOR_APPLY(real, input,
                  logsum += THExpMinusApprox(maxInput - *input_data););
  logsum = maxInput + log(logsum);

  THTensor_(resizeAs)(output, input);
  
  TH_TENSOR_APPLY2(real, output, real, input,
                   *output_data = *input_data - logsum;)

  return 1;
}

static int nn_(LogSoftMax_backward)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  real sum = THTensor_(sum)(gradOutput);

  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,     \
                   *gradInput_data = *gradOutput_data - exp(*output_data)*sum;);

  return 1;
}

static const struct luaL_Reg nn_(LogSoftMax__) [] = {
  {"LogSoftMax_forward", nn_(LogSoftMax_forward)},
  {"LogSoftMax_backward", nn_(LogSoftMax_backward)},
  {NULL, NULL}
};

void nn_(LogSoftMax_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(LogSoftMax__), "nn");
  lua_pop(L,1);
}

#endif
