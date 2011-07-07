
local mytester = torch.Tester()
local jac = nn.jacobian

local precision = 1e-5

local nntest = {}

function nntest.Add()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Add(ini*inj*ink)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')
   local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
   mytester:assert_lt(err,precision, 'error on bias ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.CMul()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.CMul(ini*inj*ink)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
   mytester:assert_lt(err,precision, 'error on weight ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Exp()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Exp()

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.HardTanh()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.HardTanh()
   
   local err = jac.test_jac(module, input)
   mytester:assert_lt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.test_io(module, input, 0.1, 2)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Linear()
   local ini = math.random(50,70)
   local inj = math.random(50,70)
   local input = torch.Tensor(ini):zero()
   local module = nn.Linear(ini,inj)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
   mytester:assert_lt(err,precision, 'error on weight ')

   local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
   mytester:assert_lt(err,precision, 'error on weight ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.LogSigmoid()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.LogSigmoid()

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.LogSoftmax()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.LogSoftMax()

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Max()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Max(1)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Min()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Min(1)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mean()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj*ink):zero()
   local module = nn.Mean(1)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Mul()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Mul(ini*inj*ink)

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')
   local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
   mytester:assert_lt(err,precision, 'error on bias ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end


function nntest.Sigmoid()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.Sigmoid()

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Softmax()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.SoftMax()

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SoftPlus()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   local module = nn.SoftPlus()

   local err = jac.test_jac(module,input)
   mytester:assert_lt(err,precision, 'error on state ')

   local ferr,berr = jac.test_io(module,input)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
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
   
   local err = jac.test_jac(module, input)
   mytester:assert_lt(err, precision, 'error on state ')
   
   local err = jac.test_jac_param(module, input, module.weight, module.gradWeight)
   mytester:assert_lt(err , precision, 'error on weight ')
   
   local err = jac.test_jac_param(module, input, module.bias, module.gradBias)
   mytester:assert_lt(err , precision, 'error on bias ')
   
   local ferr, berr = jac.test_io(module, input)
   mytester:assert_eq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Tanh()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()
   
   local module = nn.Tanh()
   
   local err = jac.test_jac(module, input)
   mytester:assert_lt(err, precision ,  'error on state ')
   
   local ferr, berr = jac.test_io(module, input, 0.1, 2)
   mytester:assert_eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assert_eq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

mytester:add(nntest)
--mytester:add(test_SpatialConvolution)
--mytester:add(test_AbsCriterion)

function nn.test()
   mytester:run()
end
