local Euclidean, parent = torch.class('nn.Euclidean', 'nn.Module')

function Euclidean:__init(inputSize,outputSize)
   parent.__init(self)
  
   self.weight = torch.Tensor(inputSize,outputSize) 
   self.gradWeight = torch.Tensor(inputSize,outputSize)
   
   -- state
   self.gradInput:resize(inputSize)
   self.output:resize(outputSize)

   self:reset()
end
  
function Euclidean:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   for i=1,self.weight:size(2) do
      self.weight:select(2, i):apply(function()
                                        return random.uniform(-stdv, stdv)
                                     end)
   end
end

function Euclidean:forward(input) 
   self.output:zero()
   for i=1,self.weight:size(2) do
	self.output[i]=input:dist(self.weight:select(2,i))
   end 
   return self.output
end

function Euclidean:updateGradInput(input, gradOutput)
  if self.gradInput then
     self.gradInput:zero()
     for i=1,self.weight:size(2) do
        self.gradInput:add(2*gradOutput[i],input);
        self.gradInput:add(-2*gradOutput[i],self.weight:select(2,i));
     end
     return self.gradInput
  end
end

function Euclidean:accGradParameters(input, gradOutput, scale)
   for i=1,self.weight:size(2) do
      local gW=self.gradWeight:select(2,i) 
      gW:add(2*gradOutput[i]*scale,self.weight:select(2,i));
      gW:add(-2*gradOutput[i]*scale,input);
   end
end
