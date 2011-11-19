require 'torch'
require 'cunn'

local cunntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}
local cunntestx = {}

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
   error = (error - groundtruth)
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
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
   sconv:accGradParameters(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
      sconv:accGradParameters(input, gradOutput)
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
   gconv:accGradParameters(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
      gconv:accGradParameters(input, gradOutput)
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

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialSubSampling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,64)
   local outj = math.random(32,64)
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
   sconv:accGradParameters(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
      sconv:accGradParameters(input, gradOutput)
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
   gconv:accGradParameters(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
      gconv:accGradParameters(input, gradOutput)
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

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.mse()
   local size = math.random(3000,5000)
   torch.setdefaulttensortype('torch.FloatTensor')
   local input = lab.randn(size,1,1)
   local target = lab.randn(size)
   local mod = nn.MSECriterion()

   local tm = {}
   local title = string.format('MSECriterion %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input,target)
   local fgin = mod:backward(input,target):clone()
   tm.cpu = a:time().real

   torch.setdefaulttensortype('torch.CudaTensor')
   local cinput = torch.Tensor(size):copy(input)
   local ctarget = torch.Tensor(size):copy(target)
   local cmod = nn.MSECriterion()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   tm.gpu = a:time().real

   local tm2 = {}
   local title = string.format('MSECriterion2 %d ',size)
   times[title] = tm2
   tm2.cpu = tm.cpu
   torch.setdefaulttensortype('torch.CudaTensor')
   local cinput2 = torch.Tensor(size):copy(input)
   local ctarget2 = torch.Tensor(size):copy(target)
   local cmod2 = nn.MSECriterion()
   a:reset()
   local cout2 = cinput2.nn.MSECriterion_updateOutput2(cmod,cinput2,ctarget2)
   local cgin2 = cinput2.nn.MSECriterion_updateGradInput2(cmod,cinput2,ctarget2)
   tm2.gpu = a:time().real

   torch.setdefaulttensortype('torch.FloatTensor')

   mytester:assertlt(math.abs(fout-cout), precision_forward, 'error  on output')
   local fcgin = torch.Tensor():resizeAs(fgin):copy(cgin)
   local gerr = fcgin - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')

   mytester:assertlt(math.abs(fout-cout2), precision_forward, 'error  on output - 2')
   local fcgin2 = torch.Tensor():resizeAs(fgin):copy(cgin2)
   local gerr2 = fcgin2 - fgin
   mytester:assertlt(gerr2:abs():max(), precision_forward, 'error  on gradInput -2')

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
