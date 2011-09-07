local Sum, parent = torch.class('nn.Sum', 'nn.Module')

function Sum:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
end

function Sum:forward(input)
   input.lab.sum_(self.output, input, self.dimension)
   self.output = self.output:select(self.dimension, 1)
   return self.output
end

function Sum:backward(input, gradOutput)
   local size = input:size()
   local stride = input:stride()
   stride[self.dimension] = 0

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:resize(size, stride)

   return self.gradInput
end
