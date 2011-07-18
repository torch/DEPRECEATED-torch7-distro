local Add, parent = torch.class('nn.Add', 'nn.Module')

function Add:__init(inputSize,scalar)
   parent.__init(self)
  
   local size=inputSize
   if scalar then size=1; end
   self.bias = torch.Tensor(size)
   self.gradBias = torch.Tensor(size)
     
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(inputSize) 

   self:reset()
end

 
function Add:reset(stdv)
   if stdv then 
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.bias:size(1))
   end

   for i=1,self.bias:size(1) do
	   self.bias[i] = random.uniform(-stdv, stdv)
   end
end

function Add:forward(input)
   self.output:copy(input);
   if self.gradBias:size(1)==1 then
     self.output:add(self.bias[1]);
   else
     self.output:add(self.bias);
   end
   return self.output
end 

function Add:backward(input, gradOutput)
   if self.gradBias:size(1)==1 then
	self.gradBias[1]=self.gradBias[1]+ gradOutput:sum();
   else
   	self.gradBias:add(gradOutput)
   end
   self.gradInput:copy(gradOutput) 
   return self.gradInput
end

function Add:zeroGradParameters()
   self.gradBias:zero()
end

function Add:updateParameters(learningRate)
   self.bias:add(-learningRate, self.gradBias)
end

function Add:write(file)
   parent.write(self, file)
   file:writeObject(self.bias)
   file:writeObject(self.gradBias)
end

function Add:read(file)
   parent.read(self, file) 
   self.bias = file:readObject()
   self.gradBias = file:readObject()
end
