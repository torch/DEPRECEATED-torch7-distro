local SparseLinear, parent = torch.class('nn.SparseLinear', 'nn.Module')

function SparseLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.lastInput = torch.Tensor()
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)

   self:reset()
end


function SparseLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   -- we do this so the initialization is exactly
   -- the same than in previous torch versions
   for i=1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
                                        return random.uniform(-stdv, stdv)
                                     end)
      self.bias[i] = random.uniform(-stdv, stdv) * 0.000001
   end
end

function SparseLinear:zeroGradParameters()
   --self.gradWeight:zero()
   self.gradBias:zero()
end

function SparseLinear:write(file)
   parent.write(self, file)
   file:writeDouble(self.weightDecay)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function SparseLinear:read(file)
   parent.read(self, file)
   self.weightDecay = file:readDouble()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()

   self.lastInput = torch.Tensor()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self.output:resizeAs(self.bias)
end
