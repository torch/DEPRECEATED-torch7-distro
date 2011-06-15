local SoftMax, parent = torch.class('nn.SoftMax', 'nn.Module')

function SoftMax:__init()
   parent.__init(self)
   self.shift = 0
   self.computeShift = true
end

function SoftMax:write(file)
   parent.write(self, file)
   file:writeDouble(self.shift)
   file:writeBool(self.computeShift)
end

function SoftMax:read(file)
   parent.read(self, file)
   self.shift = file:readDouble()
   self.computeShift = file:readBool()
end
