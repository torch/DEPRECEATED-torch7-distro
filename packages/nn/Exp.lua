local Exp = torch.class('nn.Exp', 'nn.Module')

function Exp:forward(input)
   return input.nn.Exp_forward(self, input)
end

function Exp:updateGradInput(input, gradOutput)
   return input.nn.Exp_updateGradInput(self, input, gradOutput)
end
