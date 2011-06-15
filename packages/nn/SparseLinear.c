#include "luaT.h"
#include "TH.h"

static const void * torch_Tensor_id = NULL;
static const void * nn_SparseLinear_id = NULL;
static int nn_SparseLinear_forward(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor_id);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  long dim = weight->size[1]; /* number of weights.. */

  THTensor_copy(output, bias);
  for(i = 0; i < input->size[0]; i++)
  {
    long offset = (long)(*(THTensor_dataPtr2d(input, i, 0)))-1;
    
    if(offset >= 0 && offset < dim) /* make sure indices are in bounds.. */
    {
      double val = *(THTensor_dataPtr2d(input, i,1));          
      THBlas_add(output->size[0], 
		 val, 
		 THTensor_dataPtr2d(weight, 0, offset), 
		 1, 
		 THTensor_dataPtr(output), 
		 1);
    }
    else
      luaL_error(L, "index out of bound");
  }

  return 1;
}


static int nn_SparseLinear_backward(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor_id);
  THTensor * gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor_id);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor_id);
  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor_id);
  THTensor * lastInput = luaT_getfieldcheckudata(L, 1, "lastInput", torch_Tensor_id);
  double weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");
  THTensor * gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
  long dim = weight->size[1]; /* number of weights.. */

  for(i = 0; i < input->size[0]; i++)
  {
    long offset = (long)(*(THTensor_dataPtr2d(input, i, 0)))-1;

    if(offset >= 0 && offset < dim) /* make sure indices are in bounds.. */
    {
	double val = *(THTensor_dataPtr2d(input, i,1));
	THBlas_scale(gradOutput->size[0], 
		     0, 
		     THTensor_dataPtr2d(gradWeight, 0, offset), 
		     1); /* zero */
	THBlas_add(gradOutput->size[0], 
		   val, 
		   THTensor_dataPtr(gradOutput), 
		   1, 
		   THTensor_dataPtr2d(gradWeight, 0, offset), 
		   1);
    }
    else
      luaL_error(L, "index out of bound");
  }
  
  THTensor_addTensor(gradBias, 1, gradOutput); 
  
  if(weightDecay != 0)
    THTensor_addTensor(gradWeight, weightDecay, weight);
  
  THTensor_resizeAs(lastInput, input);
  THTensor_copy(lastInput, input);
  
  return 1;
}

int nn_SparseLinear_updateParameters(lua_State *L)
{
  long i;
  double learningRate = luaL_checknumber(L, 2);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor_id);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor_id);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor_id);
  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor_id);
  THTensor * lastInput = luaT_getfieldcheckudata(L, 1, "lastInput", torch_Tensor_id);
  double weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");
  
  long dim = weight->size[1]; /* number of weights.. */
  THTensor_addTensor(bias, -learningRate, gradBias);

  for(i = 0; i < lastInput->size[0]; i++) 
  {
    long offset = (long)(*(THTensor_dataPtr2d(lastInput, i, 0)))-1;
    
    if(offset >= 0 && offset < dim) /* make sure indices are in bounds.. */
    {
      THBlas_add(bias->size[0], 
		 -learningRate, 
		 THTensor_dataPtr2d(gradWeight, 0, offset), 
		 1, 
		 THTensor_dataPtr2d(weight, 0, offset), 
		 1);
    }
    else
      luaL_error(L, "index out of bound");
  }
  return 0;
}

static const struct luaL_Reg nn_SparseLinear__ [] = {
  {"forward", nn_SparseLinear_forward},
  {"backward", nn_SparseLinear_backward},
  {"updateParameters", nn_SparseLinear_updateParameters},
  {NULL, NULL}
};

void nn_SparseLinear_init(lua_State *L)
{
  torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
  nn_SparseLinear_id = luaT_newmetatable(L, "nn.SparseLinear", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, nn_SparseLinear__);
  lua_pop(L, 1);
}
