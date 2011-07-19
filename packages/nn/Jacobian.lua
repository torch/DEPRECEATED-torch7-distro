nn.Jacobian = {}

function nn.Jacobian.backward (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   module:forward(input)
   local dout = module.output.new():resizeAs(module.output)
   -- 1D view
   local sdout = module.output.new(dout:storage(),1,dout:nElement())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor(param:nElement(),dout:nElement()):zero()

   for i=1,sdout:nElement() do
      dout:zero()
      sdout[i] = 1
      module:zeroGradParameters()
      local din = module:backward(input, dout)
      if doparam == 1 then
	 jacobian:select(2,i):copy(dparam)
      else
	 jacobian:select(2,i):copy(din)
      end
   end
   return jacobian
end

function nn.Jacobian.forward(module, input, param)
   param = param or input
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   local tst = param:storage()
   local sin = param.new(tst,1,tst:size())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())
   
   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))
   
   for i=1,sin:nElement() do      
      sin[i] = sin[i] - small
      outa:copy(module:forward(input))
      sin[i] = sin[i] + 2*small
      outb:copy(module:forward(input))
      sin[i] = sin[i] - small

      outb:add(-1,outa):div(2*small)
      jacobian:select(1,i):copy(outb)
   end
   return jacobian
end

function nn.Jacobian.testJacobian (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(lab.rand(input:nElement()):mul(inrange):add(minval))
   local jac_fprop = nn.Jacobian.forward(module,input)
   local jac_bprop = nn.Jacobian.backward(module,input)
   local error = jac_fprop:dist(jac_bprop,2)
   return error
end

function nn.Jacobian.testJacobianParameters (module, input, param, dparam, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(lab.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(lab.rand(param:nElement()):mul(inrange):add(minval))
   local jac_bprop = nn.Jacobian.backward(module, input, param, dparam)
   local jac_fprop = nn.Jacobian.forward(module, input, param)
   local error = jac_fprop - jac_bprop
   return error:abs():max()
end

function nn.Jacobian.testIO(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local go = module.output:clone():copy(lab.rand(module.output:nElement()):mul(inrange):add(minval))
   module:backward(input,go)

   local fo = module.output:clone()
   local bo = module.gradInput:clone()

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   m:backward(input,go)
   -- cleanup
   os.remove('tmp.bin')

   local fo2 = m.output:clone()
   local bo2 = m.gradInput:clone()

   local errf = fo - fo2
   local errb = bo - bo2
   return errf:abs():max(), errb:abs():max()
end
