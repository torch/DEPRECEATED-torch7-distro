#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SquareOmp.c"
#else

static int nnOmp_(Square_updateOutputOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  setompnthread(L,1,"nThread");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,		\
		     *output_data = (*input_data) * (*input_data););
  }
  else
  {
    real* output_data = THTensor_(data)(output);
    real* input_data  = THTensor_(data)(input);
    long i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(input); i++)
      output_data[i] = input_data[i]*input_data[i];
  }
  return 1;
}

static int nnOmp_(Square_updateGradInputOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  setompnthread(L,1,"nThread");
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);

  if (input->nDimension == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,	\
		     *gradInput_data  = (*gradOutput_data) * (*input_data););
  }
  else
  {
    real* gradOutput_data = THTensor_(data)(gradOutput);
    real* gradInput_data  = THTensor_(data)(gradInput);
    real* input_data  = THTensor_(data)(input);
    long i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(gradInput); i++)
      gradInput_data[i] = 2.0 * gradOutput_data[i] * input_data[i];
  }
  return 1;
}

static const struct luaL_Reg nnOmp_(Square__) [] = {
  {"Square_updateOutputOmp", nnOmp_(Square_updateOutputOmp)},
  {"Square_updateGradInputOmp", nnOmp_(Square_updateGradInputOmp)},
  {NULL, NULL}
};

static void nnOmp_(Square_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"nn");
  luaL_register(L, NULL, nnOmp_(Square__));
  lua_pop(L,1);
}

#endif
