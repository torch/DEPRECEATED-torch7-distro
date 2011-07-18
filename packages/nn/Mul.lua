local Mul, parent = torch.class('nn.Mul', 'nn.Module')

function Mul:__init(inputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(1)
   self.gradWeight = torch.Tensor(1)
   
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end

 
function Mul:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   self.weight[1] = random.uniform(-stdv, stdv);
end

function Mul:forward(input)
   self.output:copy(input);
   self.output:mul(self.weight[1]);
   return self.output 
end

function Mul:backward(input, gradOutput) 
   self.gradWeight[1] =   self.gradWeight[1] + input:dot(gradOutput);
   self.gradInput:zero()
   self.gradInput:add(self.weight[1], gradOutput)
   return self.gradInput
end

function Mul:zeroGradParameters()
   self.gradWeight:zero()
end

function Mul:updateParameters(learningRate)
   self.weight:add(-learningRate, self.gradWeight)
end

function Mul:write(file)
   parent.write(self, file)
   file:writeObject(self.weight)
   file:writeObject(self.gradWeight)
end

function Mul:read(file)
   parent.read(self, file) 
   self.weight = file:readObject()
   self.gradWeight = file:readObject()
end
