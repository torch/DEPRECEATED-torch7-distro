require 'torch'
require 'lab'
require 'nn'
require 'libopenmp'

local function nn_enable(tensorType)
   tensorType.nn.SpatialConvolution_forward_ = tensorType.nn.SpatialConvolution_forward
   tensorType.nn.SpatialConvolution_forward  = tensorType.nn.SpatialConvolution_forwardOmp
end
local function nn_disable(tensorType)
   tensorType.nn.SpatialConvolution_forward  = tensorType.nn.SpatialConvolution_forward_
   tensorType.nn.SpatialConvolution_forward_ = tensorType.nn.SpatialConvolution_forwardOmp
end

function openmp.enable ()
   lab.conv2_ = lab.conv2
   lab.conv2 = lab.conv2omp

   nn_enable(torch.DoubleTensor)
   nn_enable(torch.FloatTensor)
end

function openmp.disable ()
   lab.conv2 = lab.conv2_
   lab.conv2_ = lab.conv2omp

   nn_disable(torch.DoubleTensor)
   nn_disable(torch.FloatTensor)
end

openmp.enable()