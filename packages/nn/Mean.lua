local Mean, parent = torch.class('nn.Mean', 'nn.Module')

function Mean:__init(dimension)
   parent.__init(self)
   dimension = dimension or 2
   self.dimension = dimension
end

function Mean:write(file)
   parent.write(self, file)
   file:writeInt(self.dimension)
end

function Mean:read(file)
   parent.read(self, file)
   self.dimension = file:readInt()
end
