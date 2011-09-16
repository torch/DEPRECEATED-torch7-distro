local Power, parent = torch.class('nn.Power','nn.Module')

function Power:__init(p)
   parent.__init(self)
   self.pow = p
   if not p then
      error('nn.Power(power)')
   end
end

function Power:forward(input)
   self.output:resizeAs(input):copy(input)
   self.output:pow(self.pow)
   return self.output
end

function Power:backward(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   self.gradInput:cmul(self.output):cdiv(input):mul(self.pow)
   return self.gradInput
end
   

function Power:write(file)
   parent.write(self,file)
   file:writeDouble(self.pow)
end

function Power:read(file)
   parent.read(self,file)
   self.pow = file:readDouble()
end
