local LogSigmoid, parent = torch.class('nn.LogSigmoid', 'nn.Module')

function LogSigmoid:__init()
   parent.__init(self)
   self.buffer = torch.Tensor()
end

function LogSigmoid:write(file)
   parent.write(self, file)
   file:writeObject(self.buffer)
end

function LogSigmoid:read(file)
   parent.read(self, file)
   self.buffer = file:readObject()
end
