#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_Sum_id = NULL;

static int nn_Sum_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  long *dim;
  int i;

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  dim = THAlloc(sizeof(long)*input->nDimension);
  for(i = 0; i < input->nDimension; i++) 
    dim[i] = input->size[i];
  dim[dimension] = 1;
  THTensor_resize(output, input->nDimension, dim);
  THFree(dim);

  TH_TENSOR_DIM_APPLY2(double, output, double, input, dimension,
                       double sum = 0;
                       for(i = 0; i < input_size; i++)
                         sum += input_p[i*input_stride];
                       *output_p = sum;)

  THTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int nn_Sum_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  long *dim, *str;
  int i, j;

  THTensor_resizeAs(gradInput, gradOutput);
  THTensor_copy(gradInput, gradOutput);

  dim = THAlloc(sizeof(long)*(gradOutput->nDimension+1));
  str = THAlloc(sizeof(long)*(gradOutput->nDimension+1));
  for(i = 0, j =  0; j < gradOutput->nDimension+1; j++)
  {
    if(j == dimension)
    {
      dim[j] = input->size[dimension];
      str[j] = 0;
      continue;
    }

    dim[j] = gradOutput->size[i];
    str[j] = gradOutput->stride[i];
    i++;
  }
  THTensor_setStorage(gradInput, gradInput->storage, gradInput->storageOffset, gradInput->nDimension+1, dim, str);
  THFree(dim);
  THFree(str);

  return 1;
}

static const struct luaL_Reg nn_Sum__ [] = {
  {"forward", nn_Sum_forward},
  {"backward", nn_Sum_backward},
  {NULL, NULL}
};

void nn_Sum_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_Sum_id = luaT_newmetatable(L, "nn.Sum", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_Sum__);
  lua_pop(L, 1);
}
