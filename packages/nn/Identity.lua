local Identity, parent = torch.class('nn.Identity', 'nn.Module')

function Identity:forward(input)
   local currentOutput = input
   self.output = currentOutput
   return self.output
end


function Identity:backward(input, gradOutput)
   local gradInput = gradOutput
   self.gradInput = gradOutput
   return self.gradInput
end
