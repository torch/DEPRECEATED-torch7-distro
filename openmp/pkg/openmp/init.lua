require 'torch'
require 'nn'
require 'paths'
paths.require 'libopenmp'

local function spatialConvolution_enable(tensorType)
   tensorType.nn.SpatialConvolution_updateOutput_ = tensorType.nn.SpatialConvolution_updateOutput
   tensorType.nn.SpatialConvolution_updateOutput  = tensorType.nn.SpatialConvolution_updateOutputOmp
   tensorType.nn.SpatialConvolution_updateGradInput_ = tensorType.nn.SpatialConvolution_updateGradInput
   tensorType.nn.SpatialConvolution_updateGradInput  = tensorType.nn.SpatialConvolution_updateGradInputOmp
   tensorType.nn.SpatialConvolution_accGradParameters_ = tensorType.nn.SpatialConvolution_accGradParameters
   tensorType.nn.SpatialConvolution_accGradParameters  = tensorType.nn.SpatialConvolution_accGradParametersOmp
end
local function spatialConvolution_disable(tensorType)
   tensorType.nn.SpatialConvolution_updateOutput  = tensorType.nn.SpatialConvolution_updateOutput_
   tensorType.nn.SpatialConvolution_updateOutput_ = tensorType.nn.SpatialConvolution_updateOutputOmp
   tensorType.nn.SpatialConvolution_updateGradInput = tensorType.nn.SpatialConvolution_updateGradInput_
   tensorType.nn.SpatialConvolution_updateGradInput_  = tensorType.nn.SpatialConvolution_updateGradInputOmp
   tensorType.nn.SpatialConvolution_accGradParameters = tensorType.nn.SpatialConvolution_accGradParameters_
   tensorType.nn.SpatialConvolution_accGradParameters_  = tensorType.nn.SpatialConvolution_accGradParametersOmp
end
local function spatialConvolutionMap_enable(tensorType)
   tensorType.nn.SpatialConvolutionMap_updateOutput_ = tensorType.nn.SpatialConvolutionMap_updateOutput
   tensorType.nn.SpatialConvolutionMap_updateOutput  = tensorType.nn.SpatialConvolutionMap_updateOutputOmp
   tensorType.nn.SpatialConvolutionMap_updateGradInput_ = tensorType.nn.SpatialConvolutionMap_updateGradInput
   tensorType.nn.SpatialConvolutionMap_updateGradInput  = tensorType.nn.SpatialConvolutionMap_updateGradInputOmp
   tensorType.nn.SpatialConvolutionMap_accGradParameters_ = tensorType.nn.SpatialConvolutionMap_accGradParameters
   tensorType.nn.SpatialConvolutionMap_accGradParameters  = tensorType.nn.SpatialConvolutionMap_accGradParametersOmp
end
local function spatialConvolutionMap_disable(tensorType)
   tensorType.nn.SpatialConvolutionMap_updateOutput  = tensorType.nn.SpatialConvolutionMap_updateOutput_
   tensorType.nn.SpatialConvolutionMap_updateOutput_ = tensorType.nn.SpatialConvolutionMap_updateOutputOmp
   tensorType.nn.SpatialConvolutionMap_updateGradInput = tensorType.nn.SpatialConvolutionMap_updateGradInput_
   tensorType.nn.SpatialConvolutionMap_updateGradInput_  = tensorType.nn.SpatialConvolutionMap_updateGradInputOmp
   tensorType.nn.SpatialConvolutionMap_accGradParameters = tensorType.nn.SpatialConvolutionMap_accGradParameters_
   tensorType.nn.SpatialConvolutionMap_accGradParameters_  = tensorType.nn.SpatialConvolutionMap_accGradParametersOmp
end
local function spatialMaxPooling_enable(tensorType)
   tensorType.nn.SpatialMaxPooling_updateOutput_ = tensorType.nn.SpatialMaxPooling_updateOutput
   tensorType.nn.SpatialMaxPooling_updateOutput  = tensorType.nn.SpatialMaxPooling_updateOutputOmp
   tensorType.nn.SpatialMaxPooling_updateGradInput_ = tensorType.nn.SpatialMaxPooling_updateGradInput
   tensorType.nn.SpatialMaxPooling_updateGradInput  = tensorType.nn.SpatialMaxPooling_updateGradInputOmp
end
local function spatialMaxPooling_disable(tensorType)
   tensorType.nn.SpatialMaxPooling_updateOutput  = tensorType.nn.SpatialMaxPooling_updateOutput_
   tensorType.nn.SpatialMaxPooling_updateOutput_ = tensorType.nn.SpatialMaxPooling_updateOutputOmp
   tensorType.nn.SpatialMaxPooling_updateGradInput = tensorType.nn.SpatialMaxPooling_updateGradInput_
   tensorType.nn.SpatialMaxPooling_updateGradInput_  = tensorType.nn.SpatialMaxPooling_updateGradInputOmp
end
local function spatialSubSampling_enable(tensorType)
   tensorType.nn.SpatialSubSampling_updateOutput_ = tensorType.nn.SpatialSubSampling_updateOutput
   tensorType.nn.SpatialSubSampling_updateOutput  = tensorType.nn.SpatialSubSampling_updateOutputOmp
   tensorType.nn.SpatialSubSampling_updateGradInput_ = tensorType.nn.SpatialSubSampling_updateGradInput
   tensorType.nn.SpatialSubSampling_updateGradInput  = tensorType.nn.SpatialSubSampling_updateGradInputOmp
   tensorType.nn.SpatialSubSampling_accGradParameters_ = tensorType.nn.SpatialSubSampling_accGradParameters
   tensorType.nn.SpatialSubSampling_accGradParameters  = tensorType.nn.SpatialSubSampling_accGradParametersOmp
end
local function spatialSubSampling_disable(tensorType)
   tensorType.nn.SpatialSubSampling_updateOutput  = tensorType.nn.SpatialSubSampling_updateOutput_
   tensorType.nn.SpatialSubSampling_updateOutput_ = tensorType.nn.SpatialSubSampling_updateOutputOmp
   tensorType.nn.SpatialSubSampling_updateGradInput = tensorType.nn.SpatialSubSampling_updateGradInput_
   tensorType.nn.SpatialSubSampling_updateGradInput_  = tensorType.nn.SpatialSubSampling_updateGradInputOmp
   tensorType.nn.SpatialSubSampling_accGradParameters = tensorType.nn.SpatialSubSampling_accGradParameters_
   tensorType.nn.SpatialSubSampling_accGradParameters_  = tensorType.nn.SpatialSubSampling_accGradParametersOmp
end
local function hardTanh_enable(tensorType)
   tensorType.nn.HardTanh_updateOutput_ = tensorType.nn.HardTanh_updateOutput
   tensorType.nn.HardTanh_updateOutput  = tensorType.nn.HardTanh_updateOutputOmp
   tensorType.nn.HardTanh_updateGradInput_ = tensorType.nn.HardTanh_updateGradInput
   tensorType.nn.HardTanh_updateGradInput  = tensorType.nn.HardTanh_updateGradInputOmp
end
local function hardTanh_disable(tensorType)
   tensorType.nn.HardTanh_updateOutput  = tensorType.nn.HardTanh_updateOutput_
   tensorType.nn.HardTanh_updateOutput_ = tensorType.nn.HardTanh_updateOutputOmp
   tensorType.nn.HardTanh_updateGradInput = tensorType.nn.HardTanh_updateGradInput_
   tensorType.nn.HardTanh_updateGradInput_  = tensorType.nn.HardTanh_updateGradInputOmp
end
local function tanh_enable(tensorType)
   tensorType.nn.Tanh_updateOutput_ = tensorType.nn.Tanh_updateOutput
   tensorType.nn.Tanh_updateOutput  = tensorType.nn.Tanh_updateOutputOmp
   tensorType.nn.Tanh_updateGradInput_ = tensorType.nn.Tanh_updateGradInput
   tensorType.nn.Tanh_updateGradInput  = tensorType.nn.Tanh_updateGradInputOmp
end
local function tanh_disable(tensorType)
   tensorType.nn.Tanh_updateOutput  = tensorType.nn.Tanh_updateOutput_
   tensorType.nn.Tanh_updateOutput_ = tensorType.nn.Tanh_updateOutputOmp
   tensorType.nn.Tanh_updateGradInput = tensorType.nn.Tanh_updateGradInput_
   tensorType.nn.Tanh_updateGradInput_  = tensorType.nn.Tanh_updateGradInputOmp
end
local function sqrt_enable(tensorType)
   tensorType.nn.Sqrt_updateOutput_ = tensorType.nn.Sqrt_updateOutput
   tensorType.nn.Sqrt_updateOutput  = tensorType.nn.Sqrt_updateOutputOmp
   tensorType.nn.Sqrt_updateGradInput_ = tensorType.nn.Sqrt_updateGradInput
   tensorType.nn.Sqrt_updateGradInput  = tensorType.nn.Sqrt_updateGradInputOmp
end
local function sqrt_disable(tensorType)
   tensorType.nn.Sqrt_updateOutput  = tensorType.nn.Sqrt_updateOutput_
   tensorType.nn.Sqrt_updateOutput_ = tensorType.nn.Sqrt_updateOutputOmp
   tensorType.nn.Sqrt_updateGradInput = tensorType.nn.Sqrt_updateGradInput_
   tensorType.nn.Sqrt_updateGradInput_  = tensorType.nn.Sqrt_updateGradInputOmp
end

local function square_enable(tensorType)
   tensorType.nn.Square_updateOutput_ = tensorType.nn.Square_updateOutput
   tensorType.nn.Square_updateOutput  = tensorType.nn.Square_updateOutputOmp
   tensorType.nn.Square_updateGradInput_ = tensorType.nn.Square_updateGradInput
   tensorType.nn.Square_updateGradInput  = tensorType.nn.Square_updateGradInputOmp
end
local function square_disable(tensorType)
   tensorType.nn.Square_updateOutput  = tensorType.nn.Square_updateOutput_
   tensorType.nn.Square_updateOutput_ = tensorType.nn.Square_updateOutputOmp
   tensorType.nn.Square_updateGradInput = tensorType.nn.Square_updateGradInput_
   tensorType.nn.Square_updateGradInput_  = tensorType.nn.Square_updateGradInputOmp
end

local function nn_enable(tensorType)
   spatialConvolution_enable(tensorType)
   spatialConvolutionMap_enable(tensorType)
   spatialMaxPooling_enable(tensorType)
   spatialSubSampling_enable(tensorType)
   hardTanh_enable(tensorType)
   tanh_enable(tensorType)
   sqrt_enable(tensorType)
   square_enable(tensorType)
end
local function nn_disable(tensorType)
   spatialConvolution_disable(tensorType)
   spatialConvolutionMap_disable(tensorType)
   spatialMaxPooling_disable(tensorType)
   spatialSubSampling_disable(tensorType)
   hardTanh_disable(tensorType)
   tanh_disable(tensorType)
   sqrt_disable(tensorType)
   square_disable(tensorType)
end

function openmp.enable ()
   torch.conv2_ = torch.conv2
   torch.conv2 = torch.conv2omp

   nn_enable(torch.DoubleTensor)
   nn_enable(torch.FloatTensor)

   openmp.enabled = true
end

function openmp.disable ()
   torch.conv2 = torch.conv2_
   torch.conv2_ = torch.conv2omp

   nn_disable(torch.DoubleTensor)
   nn_disable(torch.FloatTensor)

   openmp.enabled = false
end

openmp.enabled = false
openmp.enable()

torch.include('openmp', 'test.lua')
