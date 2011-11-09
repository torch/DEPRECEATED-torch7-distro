local SoftMin, parent = torch.class('nn.SoftMin', 'nn.Module')

function SoftMin:forward(input)
   self.mininput = self.mininput or input.new()
   self.mininput:resizeAs(input):copy(input):mul(-1)
   return input.nn.SoftMax_forward(self, self.mininput)
end

function SoftMin:backward(input, gradOutput)
   self.mininput = self.mininput or input.new()
   self.mininput:resizeAs(input):copy(input):mul(-1)
   self.gradInput = input.nn.SoftMax_backward(self, self.mininput, gradOutput)
   self.gradInput:mul(-1)
   return self.gradInput
end
