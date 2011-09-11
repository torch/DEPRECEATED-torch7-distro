require 'torch'
require 'cunn'

local cunntest = {}
local precision = 1e-2
local nloop = 1
local times = {}

function cunntest.SpatialConvolution_forward()
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,2)
   local sj = math.random(1,2)
   local outi = math.random(1,256)
   local outj = math.random(1,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialConvolution.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]', 
                               from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm

   torch.setdefaulttensortype('torch.FloatTensor')
   local input = lab.randn(from,inj,ini)
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   torch.setdefaulttensortype('torch.CudaTensor')
   input = torch.Tensor(from,inj,ini):copy(input)
   local gconv = nn.SpatialConvolution(from,to,ki,kj,si,sj)
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   tm.gpu = a:time().real

   torch.setdefaulttensortype('torch.FloatTensor')
   local error = torch.Tensor(to,outi,outj):copy(rescuda)
   error = (error - groundtruth):abs()
   mytester:assertlt(error:max(), precision, 'error on state (forward) ')
end

function cunntest.SpatialConvolution_backward()
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,13)
   local kj = math.random(3,13)
   local si = 1 --math.random(1,2)
   local sj = 1 --math.random(1,2)
   local outi = math.random(8,256)
   local outj = math.random(8,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialConvolution.backward %dx%dx%d o %dx%d -> %dx%dx%d', 
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   torch.setdefaulttensortype('torch.FloatTensor')
   local input = lab.randn(from,inj,ini)
   local gradOutput = lab.randn(to,outj,outi)
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   torch.setdefaulttensortype('torch.CudaTensor')
   input = torch.Tensor(from,inj,ini):copy(input)
   local gradOutput = torch.Tensor(to,outj,outi):copy(gradOutput)
   local gconv = nn.SpatialConvolution(from,to,ki,kj,si,sj)
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   tm.gpu = a:time().real

   torch.setdefaulttensortype('torch.FloatTensor')
   local error = torch.Tensor(from,ini,inj):copy(rescuda)
   error = (error - groundgrad)
   local werror = torch.Tensor(to,from,ki,kj):copy(weightcuda)
   werror = (werror - groundweight)
   local berror = torch.Tensor(to):copy(biascuda)
   berror = (berror - groundbias)

   mytester:assertlt(error:abs():max(), precision, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision, 'error on bias (backward) ')
end

function cunntest.SpatialSubSampling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.forward %dx%dx%d o %dx%d -> %dx%dx%d', 
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   torch.setdefaulttensortype('torch.FloatTensor')
   local input = lab.randn(from,inj,ini)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   torch.setdefaulttensortype('torch.CudaTensor')
   input = torch.Tensor(from,inj,ini):copy(input)
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   tm.gpu = a:time().real

   torch.setdefaulttensortype('torch.FloatTensor')
   local error = torch.Tensor(to,outi,outj):copy(rescuda)
   error = (error - groundtruth)
   mytester:assertlt(error:abs():max(), precision, 'error on state (forward) ')
end

function cunntest.SpatialSubSampling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.backward %dx%dx%d o %dx%d -> %dx%dx%d', 
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   torch.setdefaulttensortype('torch.FloatTensor')
   local input = lab.randn(from,inj,ini)
   local gradOutput = lab.randn(to,outj,outi)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   torch.setdefaulttensortype('torch.CudaTensor')
   input = torch.Tensor(from,inj,ini):copy(input)
   local gradOutput = torch.Tensor(to,outj,outi):copy(gradOutput)
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   tm.gpu = a:time().real

   torch.setdefaulttensortype('torch.FloatTensor')
   local error = torch.Tensor(from,ini,inj):copy(rescuda)
   error = (error - groundgrad)
   local werror = torch.Tensor(to):copy(weightcuda)
   werror = (werror - groundweight)
   local berror = torch.Tensor(to):copy(biascuda)
   berror = (berror - groundbias)

   mytester:assertlt(error:abs():max(), precision, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision, 'error on bias (backward) ')
end

function nn.testcuda()
   math.randomseed(os.time())
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(cunntest)
   mytester:run()
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.cpu / tm.gpu))
   end
end

nn.testcuda()
