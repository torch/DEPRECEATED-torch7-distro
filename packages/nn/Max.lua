local Max, parent = torch.class('nn.Max', 'nn.Module')

function Max:__init(dimension)
   parent.__init(self)
   dimension = dimension or 2
   self.dimension = dimension
   self.indices = torch.Tensor()
end

function Max:write(file)
   parent.write(self, file)
   file:writeInt(self.dimension)
   file:writeObject(self.indices)
end

function Max:read(file)
   parent.read(self, file)
   self.dimension = file:readInt()
   self.indices = file:readObject()
end
