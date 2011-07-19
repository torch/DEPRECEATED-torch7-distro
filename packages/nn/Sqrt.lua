local Sqrt, parent = torch.class('nn.Sqrt','nn.Module')

function Sqrt:__init(args)
   parent.__init(self)
end

function Sqrt:forward(input)
   self.output:resizeAs(input):copy(input)
   self.output:sqrt()
   return self.output
end

function Sqrt:backward(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   self.gradInput:cdiv(self.output):mul(0.5)
   return self.gradInput
end
   

function Sqrt:write(file)
   parent.write(self,file)
end

function Sqrt:read(file)
   parent.read(self,file)
end
