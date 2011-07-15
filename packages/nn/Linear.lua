local Linear, parent = torch.class('nn.Linear', 'nn.Module')

function Linear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weightDecay = 0
  
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   self:reset()
end


function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end

   -- we do this so the initialization is exactly
   -- the same than in previous torch versions
   for i=1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
                                        return random.uniform(-stdv, stdv)
                                     end)
      self.bias[i] = random.uniform(-stdv, stdv)
   end
end

function Linear:forward(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      local bias = torch.Tensor(self.bias:storage(), 1,
                                nframe, 0,
                                nunit, 1)

      self.output:resize(nframe, nunit)
      self.output:copy(bias)
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Linear:backward(input, gradOutput)
   if input:dim() == 1 then
      self.gradWeight:addr(1, gradOutput, input)
      self.gradBias:add(gradOutput)
      
      if self.weightDecay ~= 0 then
         self.gradWeight:add(self.weightDecay, self.weight)
      end
      
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      self.gradInput:addmv(1, self.weight:t(), gradOutput)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      local gradBias = torch.Tensor(self.gradBias:storage(), 1,
                                    nframe, 0,
                                    nunit, 1)

      self.gradWeight:addmm(1, gradOutput:t(), input)
      gradBias:add(gradOutput)

      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      self.gradInput:addmm(1, gradOutput, self.weight)
   end

   return self.gradInput
end

function Linear:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function Linear:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
   self.bias:add(-learningRate, self.gradBias)
end

function Linear:write(file)
   parent.write(self, file)
   file:writeDouble(self.weightDecay)
   file:writeObject(self.weight)
   file:writeObject(self.bias)
   file:writeObject(self.gradWeight)
   file:writeObject(self.gradBias)
end

function Linear:read(file)
   parent.read(self, file)
   self.weightDecay = file:readDouble()
   self.weight = file:readObject()
   self.bias = file:readObject()
   self.gradWeight = file:readObject()
   self.gradBias = file:readObject()
end
