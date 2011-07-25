require 'torch'

local mytester = torch.Tester()
local jac

local precision = 1e-5

local nntest = {}

function nntest.Add()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Add(ini*inj*ink)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.CMul()
   local ini = math.random(5,15)
   local inj = math.random(5,15)
   local ink = math.random(5,15)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.CMul(ini*inj*ink)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Exp()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Exp()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.HardTanh()
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

function nntest.Abs()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.Abs()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.testIO(module, input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Threshold()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Threshold(random.uniform(-2,2),random.uniform(-2,2))

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.HardShrink()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.HardShrink()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Power()
   local in1 = lab.rand(10,20)
   local module = nn.Power(2)
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local pw = random.uniform()*math.random(1,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Power(pw)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Square()
   local in1 = lab.rand(10,20)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Square()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Sqrt()
   local in1 = lab.rand(10,20)
   local module = nn.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Linear()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local input = torch.Tensor(ini):zero()
   local module = nn.Linear(ini,inj)

   -- 1D
   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on weight ')

   -- 2D
   local nframe = math.random(50,70)
   local input = torch.Tensor(nframe, ini):zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on weight ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.LogSigmoid()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.LogSigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.LogSoftmax()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.LogSoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.TemporalLogSoftmax()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local input = torch.Tensor(ini,inj):zero()
   local module = nn.TemporalLogSoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Max()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Max(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Min()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Min(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mean()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Mean(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mul()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mul(ini*inj*ink)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on bias ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function nntest.Sigmoid()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Sigmoid()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softmax()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.SoftMax()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SoftPlus()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.SoftPlus()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialConvolution()
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
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialConvolutionMap()
   local from = math.random(1,10)
   local fanin = math.random(1, from)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(10,20)
   local outj = math.random(10,20)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local module = nn.SpatialConvolutionMap(nn.tables.random(from, to, fanin), ki, kj, si, sj)
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

function nntest.SpatialSubSampling()
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

function nntest.Sum()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Sum(1)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Tanh()
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

function nntest.TemporalConvolution()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local si = math.random(1,4)
   local outi = math.random(10,20)
   local ini = (outi-1)*si+ki
   local module = nn.TemporalConvolution(from, to, ki,si)
   local input = torch.Tensor(ini, from):zero()
   
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

function nntest.TemporalSubSampling()
   local from = math.random(1,10)
   local ki = math.random(1,10)
   local si = math.random(1,4)
   local outi = math.random(10,20)
   local ini = (outi-1)*si+ki
   local module = nn.TemporalSubSampling(from, ki, si)
   local input = torch.Tensor(ini, from):zero()
   
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



mytester:add(nntest)
--mytester:add(test_SpatialConvolution)
--mytester:add(test_AbsCriterion)

if not nn then
   require 'nn'
   jac = nn.Jacobian
   mytester:run()
else
   jac = nn.Jacobian
   function nn.test()
      mytester:run()
   end
end