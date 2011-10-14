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
   local size = gradOutput:size():totable()
   local stride = gradOutput:stride():totable()
   table.insert(size, self.dimension, input:size(self.dimension))
   table.insert(stride, self.dimension, 0)

   self.gradInput:set(gradOutput:storage(),
                      1,
                      torch.LongStorage(size),
                      torch.LongStorage(stride))
                      
   return self.gradInput
end
