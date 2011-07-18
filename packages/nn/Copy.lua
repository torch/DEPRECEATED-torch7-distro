local Copy, parent = torch.class('nn.Copy', 'nn.Module')

function Copy:__init(intype, outtype)
   intype = intype or torch.getmetatable(torch.Tensor.__typename)
   outtype = outtype or torch.getmetatable(torch.Tensor.__typename)

   parent.__init(self)
   self.gradInput = torch.getmetatable(intype).new()
   self.output = torch.getmetatable(outtype).new()
end

function Copy:forward(input)
   self.output:resize(input:size()):copy(input)
   return self.output
end

function Copy:backward(input, gradOutput)
   self.gradInput:resize(gradOutput:size()):copy(gradOutput)
   return self.gradInput
end
