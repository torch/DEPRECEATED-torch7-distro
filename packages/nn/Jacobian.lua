nn.jacobian = {}

function nn.jacobian.get_jac_bprop (module, input, param, dparam)
   local doparam = 0
   if param then
      doparam = 1
   end
   param = param or input
   -- output deriv
   local dout = torch.Tensor():resizeAs(module:forward(input))
   -- 1D view
   local sdout = torch.Tensor(dout:storage(),1,dout:nElement())
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

function nn.jacobian.get_jac_fprop(module, input, param)
   param = param or input
   -- perturbation amount
   local small = 1e-6
   -- 1D view of input
   local tst = param:storage()
   local sin = torch.Tensor(tst,1,tst:size())
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

function nn.jacobian.test_jac (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(lab.rand(input:nElement()):mul(inrange):add(minval))
   local jac_fprop = nn.jacobian.get_jac_fprop(module,input)
   local jac_bprop = nn.jacobian.get_jac_bprop(module,input)
   local error = jac_fprop:dist(jac_bprop,2)
   return error
end

function nn.jacobian.test_jac_param (module, input, param, dparam, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(lab.rand(input:nElement()):mul(inrange):add(minval))
   param:copy(lab.rand(param:nElement()):mul(inrange):add(minval))
   jac_bprop = nn.jacobian.get_jac_bprop(module, input, param, dparam)
   jac_fprop = nn.jacobian.get_jac_fprop(module, input, param)
   local error = jac_fprop - jac_bprop
   return error:abs():max()
end

function nn.jacobian.test_io(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local go = torch.Tensor():resizeAs(module.output):copy(lab.rand(module.output:nElement()):mul(inrange):add(minval))
   module:backward(input,go)

   local fo = torch.Tensor():resizeAs(module.output):copy(module.output)
   local bo = torch.Tensor():resizeAs(module.gradInput):copy(module.gradInput)

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   m:backward(input,go)
   -- cleanup
   os.execute('rm tmp.bin')

   local fo2 = torch.Tensor():resizeAs(m.output):copy(m.output)
   local bo2 = torch.Tensor():resizeAs(m.gradInput):copy(m.gradInput)

   local errf = fo - fo2
   local errb = bo - bo2
   return errf:abs():max(), errb:abs():max()
end
