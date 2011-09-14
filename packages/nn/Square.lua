local Square, parent = torch.class('nn.Square','nn.Module')

function Square:__init(args)
   parent.__init(self)
end

function Square:forward(input)
   self.output:resizeAs(input):copy(input)
   self.output:cmul(input)
   return self.output
end

function Square:backward(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   self.gradInput:cmul(input):mul(2)
   return self.gradInput
end
   

function Square:write(file)
   parent.write(self,file)
end

function Square:read(file)
   parent.read(self,file)
end
