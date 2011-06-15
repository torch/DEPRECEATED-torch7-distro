local Sum, parent = torch.class('nn.Sum', 'nn.Module')

function Sum:__init(dimension)
   parent.__init(self)
   dimension = dimension or 2
   self.dimension = dimension
end

function Sum:write(file)
   parent.write(self, file)
   file:writeInt(self.dimension)
end

function Sum:read(file)
   parent.read(self, file)
   self.dimension = file:readInt()
end
