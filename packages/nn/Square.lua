local Square, parent = torch.class('nn.Square','nn.Module')

function Square:__init(args)
   parent.__init(self)
end

function Square:forward(input)
   return input.nn.Square_forward(self, input)
end

function Square:backward(input, gradOutput)
   return input.nn.Square_backward(self, input, gradOutput)
end
