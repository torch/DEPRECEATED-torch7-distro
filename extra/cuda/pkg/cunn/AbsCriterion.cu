#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

struct abs_functor
{
  abs_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      float z = x-y;
      return z >= 0 ? z : -z;
    }
};


static int cunn_AbsCriterion_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  float sum;

  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (float) 0, thrust::plus<float>(), abs_functor());

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(input);
  THCudaTensor_free(target);
 
  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct abs_updateGradInput_functor
{
  const float norm;

  abs_updateGradInput_functor(float norm_) : norm(norm_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      return (x - y) >= 0 ? norm : -norm;
    }
};

static int cunn_AbsCriterion_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  long size = THCudaTensor_nElement(input);
  float norm = (sizeAverage ? 1./size : 1.);

  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);

  THCudaTensor_resizeAs(gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, abs_updateGradInput_functor(norm));

  THCudaTensor_free(input);
  THCudaTensor_free(target);
  return 1;
}

static const struct luaL_Reg cunn_AbsCriterion__ [] = {
  {"AbsCriterion_updateOutput", cunn_AbsCriterion_updateOutput},
  {"AbsCriterion_updateGradInput", cunn_AbsCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunn_AbsCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_AbsCriterion__, "nn");
  lua_pop(L,1);
}
