#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

static int nn_(Square_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  
  input = THTensor_(newContiguous)(input);

  THTensor_(resizeAs)(output, input);

  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  long nelem = THTensor_(nElement)(input);

  long i;
  for (i = 0; i < nelem; i++)
    output_data[i] = input_data[i] * input_data[i];

  THTensor_(free)(input);
  return 1;
}

static int nn_(Square_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  real *gradInput_data = THTensor_(data)(gradInput);
  real *input_data = THTensor_(data)(input);
  real *gradOutput_data = THTensor_(data)(gradOutput);

  long nelem = THTensor_(nElement)(input);

  long i;
  for (i = 0; i < nelem; i++)
    gradInput_data[i] = 2 * gradOutput_data[i] * input_data[i];

  return 1;
}

static const struct luaL_Reg nn_(Square__) [] = {
  {"Square_forward", nn_(Square_forward)},
  {"Square_backward", nn_(Square_backward)},
  {NULL, NULL}
};

static void nn_(Square_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Square__), "nn");
  lua_pop(L,1);
}

#endif
