#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardTanhOmp.c"
#else

static int nnOmp_(HardTanh_forwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,	    \
		     if(*input_data < -1)	    \
		       *output_data = -1;		\
		     else if(*input_data <= 1)		\
		       *output_data = *input_data;	\
		     else				\
		       *output_data = 1;);
  }
  else
  {
    real* output_data = THTensor_(data)(output);
    real* input_data  = THTensor_(data)(input);
    long k;
#pragma omp parallel for private(k)
    for (k = 0; k < input->size[0]; k++)
    {
      real* ptr_output = output_data + k*input->stride[0];
      real* ptr_input  = input_data  + k*input->stride[0];
      long i;
      for (i = 0; i < input->stride[0]; i++)
      {
	if(ptr_input[i] < -1)
	  ptr_output[i] = -1;
	else if (ptr_input[i] <= 1)
	  ptr_output[i] = ptr_input[i];
	else
	  ptr_output[i] = 1;
      }
    }
  }
  
  return 1;
}

static int nnOmp_(HardTanh_backwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);

  if (input->nDimension == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,	\
		     if(*input_data < -1 || *input_data > 1)		\
		       *gradInput_data = 0;                             \
		     else						\
		       *gradInput_data = *gradOutput_data;);
  }
  else
  {
    real* gradOutput_data = THTensor_(data)(gradOutput);
    real* gradInput_data  = THTensor_(data)(gradInput);
    real* input_data      = THTensor_(data)(input);
    long k;
#pragma omp parallel for private(k)
    for (k = 0; k < input->size[0]; k++)
    {
      real* ptr_gradOutput = gradOutput_data + k*input->stride[0];
      real* ptr_gradInput  = gradInput_data  + k*input->stride[0];
      real* ptr_input      = input_data      + k*input->stride[0];
      long i;
      for (i = 0; i < input->stride[0]; i++)
      {
	if(ptr_input[i] < -1 || ptr_input[i] > 1)
	  ptr_gradInput[i] = 0;
	else
	  ptr_gradInput[i] = ptr_gradOutput[i];
      }
    }
  }
  return 1;
}

static const struct luaL_Reg nnOmp_(HardTanh__) [] = {
  {"HardTanh_forwardOmp", nnOmp_(HardTanh_forwardOmp)},
  {"HardTanh_backwardOmp", nnOmp_(HardTanh_backwardOmp)},
  {NULL, NULL}
};

static void nnOmp_(HardTanh_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"nn");
  luaL_register(L, NULL, nnOmp_(HardTanh__));
  lua_pop(L,1);
}

#endif
