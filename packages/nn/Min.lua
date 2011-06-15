local Min, parent = torch.class('nn.Min', 'nn.Module')

function Min:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.indices = torch.Tensor()
end

function Min:forward(input)
   return input.nn.Min_forward(self, input)
end

function Min:backward(input, gradOutput)
   return input.nn.Min_backward(self, input, gradOutput)
end

function Min:write(file)
   parent.write(self, file)
   file:writeInt(self.dimension)
   file:writeObject(self.indices)
end

function Min:read(file)
   parent.read(self, file)
   self.dimension = file:readInt()
   self.indices = file:readObject()
end
