local Sqrt, parent = torch.class('nn.Sqrt','nn.Module')

function Sqrt:__init(args)
   parent.__init(self)
end

function Sqrt:forward(input)
   return input.nn.Sqrt_forward(self,input)
end

function Sqrt:backward(input, gradOutput)
   return input.nn.Sqrt_backward(self,input,gradOutput)
end
