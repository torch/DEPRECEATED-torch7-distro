local Criterion = torch.class('nn.Criterion')

function Criterion:__init()
   self.gradInput = torch.Tensor()
   self.output = 0
end

function Criterion:forward(input, target)
end

function Criterion:backward(input, target)
end

function Criterion:write(file)
   file:writeObject(self.gradInput)
   file:writeDouble(self.output) 
end

function Criterion:read(file)
   self.gradInput = file:readObject()
   self.output = file:readDouble()
end
