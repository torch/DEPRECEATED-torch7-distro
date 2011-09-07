local Mean, parent = torch.class('nn.Mean', 'nn.Module')

function Mean:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
end

function Mean:forward(input)
   input.lab.mean_(self.output, input, self.dimension)
   self.output = self.output:select(self.dimension, 1)
   return self.output
end

function Mean:backward(input, gradOutput)
   local size = input:size()
   local stride = input:stride()
   stride[self.dimension] = 0

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:mul(1/input:size(self.dimension))
   self.gradInput:resize(size, stride)

   return self.gradInput
end
