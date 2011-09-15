#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sqrt.c"
#else


static int nn_(Sqrt_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  input = THTensor_(newContiguous)(input);

  THTensor_(resizeAs)(gradInput, input);
  real *gradInput_data = THTensor_(data)(gradInput);
  real *output_data = THTensor_(data)(output);
  real *gradOutput_data = THTensor_(data)(gradOutput);

  long nelem = THTensor_(nElement)(input);

  long i;
  for (i = 0; i < nelem; i++)
    gradInput_data[i] = 0.5 * (gradOutput_data[i] / output_data[i]);

  THTensor_(free)(input);
  return 1;
}

static const struct luaL_Reg nn_(Sqrt__) [] = {
  {"Sqrt_backward", nn_(Sqrt_backward)},
  {NULL, NULL}
};

static void nn_(Sqrt_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Sqrt__), "nn");
  lua_pop(L,1);
}

#endif
