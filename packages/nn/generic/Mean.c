#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Mean.c"
#else

static int nn_(Mean_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THLongStorage *dim;
  long i;

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  dim = THLongStorage_newWithSize(input->nDimension);
  for(i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THTensor_(resize)(output, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, output, real, input, dimension,
                       real sum = 0;
                       for(i = 0; i < input_size; i++)
                         sum += input_data[i*input_stride];
                       *output_data = sum;)

  THTensor_(mul)(output, 1./((real)input->size[dimension]));
  THTensor_(select)(output, dimension, 0);

  return 1;
}

static int nn_(Mean_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THLongStorage *dim, *str;
  int i, j;

  THTensor_(resizeAs)(gradInput, gradOutput);
  THTensor_(copy)(gradInput, gradOutput);
  THTensor_(mul)(gradInput, 1./((real)input->size[dimension]));

  dim = THLongStorage_newWithSize(gradOutput->nDimension+1);
  str = THLongStorage_newWithSize(gradOutput->nDimension+1);
  for(i = 0, j =  0; j < gradOutput->nDimension+1; j++)
  {
    if(j == dimension)
    {
      dim->data[j] = input->size[dimension];
      str->data[j] = 0;
      continue;
    }

    dim->data[j] = gradOutput->size[i];
    str->data[j] = gradOutput->stride[i];
    i++;
  }

  THTensor_(resize)(gradInput, dim, str);
  THLongStorage_free(dim);
  THLongStorage_free(str);

  return 1;
}

static const struct luaL_Reg nn_(Mean__) [] = {
  {"Mean_forward", nn_(Mean_forward)},
  {"Mean_backward", nn_(Mean_backward)},
  {NULL, NULL}
};

static void nn_(Mean_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Mean__), "nn");
  lua_pop(L,1);
}

#endif
