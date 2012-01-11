local Sequential, parent = torch.class('nn.Sequential', 'nn.Module')

function Sequential:__init()
   self.modules = {}
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:size()
   return #self.modules
end

function Sequential:get(index)
   return self.modules[index]
end

function Sequential:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do 
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end 
   self.output = currentOutput
   return currentOutput
end

function Sequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function Sequential:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end

function Sequential:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function Sequential:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Sequential:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Sequential:parameters()
   local function tinsert(from, to)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(from[i],to)
         end
      else
         table.insert(from,to)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      tinsert(mw,w)
      tinsert(mgw,gw)
   end
   return w,gw
end

