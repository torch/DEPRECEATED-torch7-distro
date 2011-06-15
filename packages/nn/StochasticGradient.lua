require "lab"

local StochasticGradient = torch.class('nn.StochasticGradient')

function StochasticGradient:__init(module, criterion)
   self.learningRate = 0.01
   self.learningRateDecay = 0
   self.maxIteration = 25
   self.shuffleIndices = true
   self.module = module
   self.criterion = criterion
end

function StochasticGradient:train(dataset)
   local iteration = 1
   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion

   local shuffledIndices = lab.randperm(dataset:size())
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   end

   print("# StochasticGradient: training")

   while true do
      local currentError = 0
      for t = 1,dataset:size() do
         local example = dataset[shuffledIndices[t]]
         local input = example[1]
         local target = example[2]

         currentError = currentError + criterion:forward(module:forward(input), target)

         module:zeroGradParameters()
         module:backward(input, criterion:backward(module.output, target))
         module:updateParameters(currentLearningRate)

         if self.hookExample then
            self.hookExample(self, example)
         end
      end

      if self.hookIteration then
         self.hookIteration(self, iteration)
      end

      currentError = currentError / dataset:size()
      print("# current error = " .. currentError)
      iteration = iteration + 1
      currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
      if self.maxIteration > 0 and iteration > self.maxIteration then
         print("# StochasticGradient: you have reached the maximum number of iterations")
         break
      end
   end
end

function StochasticGradient:write(file)
   file:writeDouble(self.learningRate)
   file:writeDouble(self.learningRateDecay)
   file:writeInt(self.maxIteration)
   file:writeBool(self.shuffleIndices)
   file:writeObject(self.module)
   file:writeObject(self.criterion)
end

function StochasticGradient:read(file)
   self.learningRate = file:readDouble()
   self.learningRateDecay = file:readDouble()
   self.maxIteration = file:readInt()
   self.shuffleIndices = file:readBool()
   self.module = file:readObject()
   self.criterion = file:readObject()
end
