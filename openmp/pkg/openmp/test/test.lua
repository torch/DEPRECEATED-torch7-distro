require 'torch'
require 'nn'

local omptest = {}
local precision = 1e-5

function omptest.openmp()
   openmp.disable()
   mytester:assert(not openmp.enabled,'openmp disable ')

   openmp.enable()
   mytester:assert(openmp.enabled,'openmp enable ')

   local nth = math.random(2,10)
   openmp.setNumThreads(nth)
   mytester:asserteq(openmp.getNumThreads(), nth, 'openmp get/setNumThreads ')

   openmp.setNumThreads(openmp.getDefaultNumThreads())
   mytester:asserteq(openmp.getNumThreads(), openmp.getDefaultNumThreads(), 'openmp get/setDefaultNumThreads ')

   mytester:assertgt(openmp.getDefaultNumThreads(), 1, 'openmp running on multi-core ')
end

function omptest.SpatialConvolutionJacobianBatch()

   -- batch
   
   --verbose = true
   local batch = math.random(5,10)
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(4,8)
   local outj = math.random(4,8)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = torch.Tensor(batch,from,inj,ini):zero()

   --print(input:nElement())
   --print(module.weight:nElement())

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')
   
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'batch error on weight [direct update] ')
   
   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'batch error on bias [direct update] ')
end

function omptest.SpatialConvolutionJacobian()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')
   
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err , precision, 'error on weight [direct update] ')
   
   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err , precision, 'error on bias [direct update] ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function omptest.SpatialSubSamplingJacobian()
   local from = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')
   
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function omptest.SpatialSubSamplingJacobianBatch()
   local batch = math.random(2,5)
   local from = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(5,10)
   local outj = math.random(5,10)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   local input = torch.Tensor(batch,from, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'batch error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'batch error on weight ')
   
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'batch error on bias ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function omptest.TanhJacobian()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.Tanh()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.testIO(module, input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function omptest.HardTanhJacobian()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.HardTanh()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.testIO(module, input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

local function modtester(module,input,params)
   params = params or {}
   local tensordmax= function(t1,t2)
			local d = t1-t2
			return d:abs():max()
		     end
   local precision = 1e-8

   openmp.disable()
   local outseq = module:forward(input):clone()
   local goutseq = lab.rand(outseq:size())
   module:zeroGradParameters()
   local ginseq = module:backward(input,goutseq):clone()
   module:accGradParameters(input,goutseq)
   local gparseq = {}
   for i=1,#params do
      gparseq[i] = module[params[i]]:clone()
   end

   openmp.enable()
   local outomp = module:forward(input):clone()
   local goutomp = goutseq:clone()
   module:zeroGradParameters()
   local ginomp = module:backward(input,goutomp):clone()
   module:accGradParameters(input,goutomp)
   local gparomp = {}
   for i=1,#params do
      gparomp[i] = module[params[i]]:clone()
   end

   mytester:assertne(outseq,outomp, ' same output tensor')
   mytester:assertne(goutseq,goutomp, ' same output gradient tensor')
   mytester:assertne(ginseq,ginomp, ' same input gradient tensor')

   mytester:assertlt(tensordmax(outseq,outomp), precision, ' forward output')
   mytester:assertlt(tensordmax(goutseq,goutomp), precision, 'output gradient')
   mytester:assertlt(tensordmax(ginseq,ginomp), precision, ' input gradient')

   for i=1,#params do
      mytester:assertne(gparseq[i],gparomp[i], ' same ' .. params[i] .. ' tensor')
      mytester:assertlt(tensordmax(gparseq[i],gparomp[i]), precision, ' input gradient')
   end
end

function omptest.SpatialConvolutionCompare()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = lab.rand(from, inj, ini)
   
   modtester(module,input,{'gradWeight','gradBias'})
end

function omptest.SpatialSubSamplingCompare()
   local from = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   
   modtester(module,input,{'gradWeight','gradBias'})
end

function omptest.SpatialConvolutionBatchCompare()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = lab.randn(from,inj,ini)

   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end

function omptest.SpatialSubSamplingBatchCompare()
   local from = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialSubSampling(from, ki, kj, si, sj)
   local input = lab.randn(from,inj,ini)--torch.Tensor(from, inj, ini):zero()
   batchcompare(module,input, {'weight','bias','gradWeight','gradBias'})
end


function omptest.TanhCompare()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.Tanh()
   
   modtester(module,input)
end


function omptest.HardTanhCompare()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.HardTanh()
   modtester(module,input)
end

function openmp.test()
   -- randomize stuff
   math.randomseed(os.time())
   
   jac = nn.Jacobian
   mytester = torch.Tester()
   mytester:add(omptest)
   mytester:run()
end
