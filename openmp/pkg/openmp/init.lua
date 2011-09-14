require 'torch'
require 'lab'
require 'nn'
require 'paths'
paths.require 'libopenmp'

local function spatialConvolution_enable(tensorType)
   tensorType.nn.SpatialConvolution_forward_ = tensorType.nn.SpatialConvolution_forward
   tensorType.nn.SpatialConvolution_forward  = tensorType.nn.SpatialConvolution_forwardOmp
   tensorType.nn.SpatialConvolution_backward_ = tensorType.nn.SpatialConvolution_backward
   tensorType.nn.SpatialConvolution_backward  = tensorType.nn.SpatialConvolution_backwardOmp
   tensorType.nn.SpatialConvolution_accGradParameters_ = tensorType.nn.SpatialConvolution_accGradParameters
   tensorType.nn.SpatialConvolution_accGradParameters  = tensorType.nn.SpatialConvolution_accGradParametersOmp
end
local function spatialConvolution_disable(tensorType)
   tensorType.nn.SpatialConvolution_forward  = tensorType.nn.SpatialConvolution_forward_
   tensorType.nn.SpatialConvolution_forward_ = tensorType.nn.SpatialConvolution_forwardOmp
   tensorType.nn.SpatialConvolution_backward = tensorType.nn.SpatialConvolution_backward_
   tensorType.nn.SpatialConvolution_backward_  = tensorType.nn.SpatialConvolution_backwardOmp
   tensorType.nn.SpatialConvolution_accGradParameters = tensorType.nn.SpatialConvolution_accGradParameters_
   tensorType.nn.SpatialConvolution_accGradParameters_  = tensorType.nn.SpatialConvolution_accGradParametersOmp
end
local function spatialConvolutionMap_enable(tensorType)
   tensorType.nn.SpatialConvolutionMap_forward_ = tensorType.nn.SpatialConvolutionMap_forward
   tensorType.nn.SpatialConvolutionMap_forward  = tensorType.nn.SpatialConvolutionMap_forwardOmp
   tensorType.nn.SpatialConvolutionMap_backward_ = tensorType.nn.SpatialConvolutionMap_backward
   tensorType.nn.SpatialConvolutionMap_backward  = tensorType.nn.SpatialConvolutionMap_backwardOmp
   tensorType.nn.SpatialConvolutionMap_accGradParameters_ = tensorType.nn.SpatialConvolutionMap_accGradParameters
   tensorType.nn.SpatialConvolutionMap_accGradParameters  = tensorType.nn.SpatialConvolutionMap_accGradParametersOmp
end
local function spatialConvolutionMap_disable(tensorType)
   tensorType.nn.SpatialConvolutionMap_forward  = tensorType.nn.SpatialConvolutionMap_forward_
   tensorType.nn.SpatialConvolutionMap_forward_ = tensorType.nn.SpatialConvolutionMap_forwardOmp
   tensorType.nn.SpatialConvolutionMap_backward = tensorType.nn.SpatialConvolutionMap_backward_
   tensorType.nn.SpatialConvolutionMap_backward_  = tensorType.nn.SpatialConvolutionMap_backwardOmp
   tensorType.nn.SpatialConvolutionMap_accGradParameters = tensorType.nn.SpatialConvolutionMap_accGradParameters_
   tensorType.nn.SpatialConvolutionMap_accGradParameters_  = tensorType.nn.SpatialConvolutionMap_accGradParametersOmp
end
local function spatialMaxPooling_enable(tensorType)
   tensorType.nn.SpatialMaxPooling_forward_ = tensorType.nn.SpatialMaxPooling_forward
   tensorType.nn.SpatialMaxPooling_forward  = tensorType.nn.SpatialMaxPooling_forwardOmp
   tensorType.nn.SpatialMaxPooling_backward_ = tensorType.nn.SpatialMaxPooling_backward
   tensorType.nn.SpatialMaxPooling_backward  = tensorType.nn.SpatialMaxPooling_backwardOmp
end
local function spatialMaxPooling_disable(tensorType)
   tensorType.nn.SpatialMaxPooling_forward  = tensorType.nn.SpatialMaxPooling_forward_
   tensorType.nn.SpatialMaxPooling_forward_ = tensorType.nn.SpatialMaxPooling_forwardOmp
   tensorType.nn.SpatialMaxPooling_backward = tensorType.nn.SpatialMaxPooling_backward_
   tensorType.nn.SpatialMaxPooling_backward_  = tensorType.nn.SpatialMaxPooling_backwardOmp
end
local function spatialSubSampling_enable(tensorType)
   tensorType.nn.SpatialSubSampling_forward_ = tensorType.nn.SpatialSubSampling_forward
   tensorType.nn.SpatialSubSampling_forward  = tensorType.nn.SpatialSubSampling_forwardOmp
   tensorType.nn.SpatialSubSampling_backward_ = tensorType.nn.SpatialSubSampling_backward
   tensorType.nn.SpatialSubSampling_backward  = tensorType.nn.SpatialSubSampling_backwardOmp
   tensorType.nn.SpatialSubSampling_accGradParameters_ = tensorType.nn.SpatialSubSampling_accGradParameters
   tensorType.nn.SpatialSubSampling_accGradParameters  = tensorType.nn.SpatialSubSampling_accGradParametersOmp
end
local function spatialSubSampling_disable(tensorType)
   tensorType.nn.SpatialSubSampling_forward  = tensorType.nn.SpatialSubSampling_forward_
   tensorType.nn.SpatialSubSampling_forward_ = tensorType.nn.SpatialSubSampling_forwardOmp
   tensorType.nn.SpatialSubSampling_backward = tensorType.nn.SpatialSubSampling_backward_
   tensorType.nn.SpatialSubSampling_backward_  = tensorType.nn.SpatialSubSampling_backwardOmp
   tensorType.nn.SpatialSubSampling_accGradParameters = tensorType.nn.SpatialSubSampling_accGradParameters_
   tensorType.nn.SpatialSubSampling_accGradParameters_  = tensorType.nn.SpatialSubSampling_accGradParametersOmp
end
local function hardTanh_enable(tensorType)
   tensorType.nn.HardTanh_forward_ = tensorType.nn.HardTanh_forward
   tensorType.nn.HardTanh_forward  = tensorType.nn.HardTanh_forwardOmp
   tensorType.nn.HardTanh_backward_ = tensorType.nn.HardTanh_backward
   tensorType.nn.HardTanh_backward  = tensorType.nn.HardTanh_backwardOmp
end
local function hardTanh_disable(tensorType)
   tensorType.nn.HardTanh_forward  = tensorType.nn.HardTanh_forward_
   tensorType.nn.HardTanh_forward_ = tensorType.nn.HardTanh_forwardOmp
   tensorType.nn.HardTanh_backward = tensorType.nn.HardTanh_backward_
   tensorType.nn.HardTanh_backward_  = tensorType.nn.HardTanh_backwardOmp
end
local function tanh_enable(tensorType)
   tensorType.nn.Tanh_forward_ = tensorType.nn.Tanh_forward
   tensorType.nn.Tanh_forward  = tensorType.nn.Tanh_forwardOmp
   tensorType.nn.Tanh_backward_ = tensorType.nn.Tanh_backward
   tensorType.nn.Tanh_backward  = tensorType.nn.Tanh_backwardOmp
end
local function tanh_disable(tensorType)
   tensorType.nn.Tanh_forward  = tensorType.nn.Tanh_forward_
   tensorType.nn.Tanh_forward_ = tensorType.nn.Tanh_forwardOmp
   tensorType.nn.Tanh_backward = tensorType.nn.Tanh_backward_
   tensorType.nn.Tanh_backward_  = tensorType.nn.Tanh_backwardOmp
end

local function nn_enable(tensorType)
   spatialConvolution_enable(tensorType)
   spatialConvolutionMap_enable(tensorType)
   spatialMaxPooling_enable(tensorType)
   spatialSubSampling_enable(tensorType)
   hardTanh_enable(tensorType)
   tanh_enable(tensorType)
end
local function nn_disable(tensorType)
   spatialConvolution_disable(tensorType)
   spatialConvolutionMap_disable(tensorType)
   spatialMaxPooling_disable(tensorType)
   spatialSubSampling_disable(tensorType)
   hardTanh_disable(tensorType)
   tanh_disable(tensorType)
end

function openmp.enable ()
   lab.conv2_ = lab.conv2
   lab.conv2 = lab.conv2omp

   nn_enable(torch.DoubleTensor)
   nn_enable(torch.FloatTensor)

   openmp.enabled = true
end

function openmp.disable ()
   lab.conv2 = lab.conv2_
   lab.conv2_ = lab.conv2omp

   nn_disable(torch.DoubleTensor)
   nn_disable(torch.FloatTensor)

   openmp.enabled = false
end

openmp.enabled = false
openmp.enable()

torch.include('openmp', 'test.lua')
