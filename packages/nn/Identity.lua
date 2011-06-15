local Identity, parent = torch.class('nn.Identity', 'nn.Module')

function Identity:__init()
   parent.__init(self)
end


function Identity:forward(input)
   local currentOutput=input;
   self.output= currentOutput
   return self.output
end


function Identity:backward(input, gradOutput)
   local gradInput=gradOutput;   
   self.gradInput=gradOutput 
   return self.gradInput
end

function Identity:zeroGradParameters()
end

function Identity:updateParameters(learningRate)
end

function Identity:write(file)
   parent.write(self, file)
end

function Identity:read(file)
   parent.read(self, file)
end
