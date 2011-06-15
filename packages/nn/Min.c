#include "luaT.h"
#include "TH.h"

static const void* torch_Tensor_id = NULL;
static const void* nn_Min_id = NULL;

static int nn_Min_forward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor_id);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

  long *dim;
  int i;

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  dim = THAlloc(sizeof(long)*input->nDimension);
  for(i = 0; i < input->nDimension; i++)
    dim[i] = input->size[i];
  dim[dimension] = 1;
  THTensor_resize(output, input->nDimension, dim);
  THTensor_resize(indices, input->nDimension, dim);
  THFree(dim);

  TH_TENSOR_DIM_APPLY3(double, output, double, input, double, indices, dimension,
                       int theIndex = 0;
                       double theMin = input_p[0];
                       for(i = 1; i < input_size; i++)
                       {
                         if(input_p[i*input_stride] < theMin)
                         {
                           theIndex = i;
                           theMin = input_p[i*input_stride];
                         }
                       }
                       *indices_p = theIndex+1;
                       *output_p = theMin;)

  THTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int nn_Min_backward(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor_id);
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *gradInput  = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);

  THTensor *gradOutputPlusOneDim;
  long *dim, *str;
  int i, j;

  THTensor_resizeAs(gradInput, input);
  THTensor_zero(gradInput);

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
  gradOutputPlusOneDim = THTensor_newWithStorage(gradOutput->storage, gradOutput->storageOffset, gradOutput->nDimension+1, dim, str);
  THFree(dim);
  THFree(str);

  TH_TENSOR_DIM_APPLY3(double, gradInput, double, gradOutputPlusOneDim, double, indices, dimension,
                       gradInput_p[ ((long)(*indices_p)-1)*gradInput_stride ] = *gradOutputPlusOneDim_p;)
 
  THTensor_free(gradOutputPlusOneDim);

  return 1;
}

static const struct luaL_Reg nn_Min__ [] = {
  {"forward", nn_Min_forward},
  {"backward", nn_Min_backward},
  {NULL, NULL}
};

void nn_Min_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_Min_id = luaT_newmetatable(L, "nn.Min", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_Min__);
  lua_pop(L, 1);
}
