local LookupTable, parent = torch.class('nn.LookupTable', 'nn.Module')

LookupTable.__version = 2

function LookupTable:__init(nIndex, ...)
   parent.__init(self)

   if select('#', ...) == 1 and type(select(1, ...)) ~= "number" then
      local size = select(1, ...)
      self.size = torch.LongStorage(#size + 1)
      for i=1,#size do
         self.size[i+1] = size[i]
      end
   else
      self.size = torch.LongStorage(select('#', ...)+1)
      for i=1,select('#',...) do
         self.size[i+1] = select(i, ...)
      end
   end

   self.size[1] = nIndex
   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size)
   self.currentInputs = {}

   self:reset()
end

function LookupTable:reset(stdv)
   stdv = stdv or 1
   self.weight:apply(function()
                        return random.normal(0, stdv)
                     end)
end

function LookupTable:forward(input)
   local nIndex = input:size(1)
   self.size[1] = nIndex
   self.output:resize(self.size)

   for i=1,nIndex do
      self.output:select(1, i):copy(self.weight:select(1, input[i]))
   end

   return self.output
end

function LookupTable:zeroGradParameters()
   for i=1,#self.currentInputs do
      self.gradWeight:select(1, currentInput[i]):zero()
      self.currentInputs[i] = nil
   end
end

function LookupTable:accGradParameters(input, gradOutput, scale)
   table.insert(self.currentInputs, input.new(input:size()):copy(input))
   self.gradWeight:select(1, currentInput[i]):add(scale, gradOutput:select(1, i))
end

function LookupTable:accUpdateGradParameters(input, gradOutput, lr)
   self.weight:select(1, currentInput[i]):add(-lr, gradOutput:select(1, i))
end

function LookupTable:updateParameters(learningRate)
   for i=1,#self.currentInputs do
      local currentInput = self.currentInputs[i]
      local currentGradWeight = self.currentGradWeights[i]
      for i=1,currentInput:size(1) do
         self.weight:select(1, currentInput[i]):add(-learningRate, self.gradWeight:select(1, currentInput[i]))
      end
   end
end
