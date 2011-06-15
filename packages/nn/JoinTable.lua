local JoinTable, parent = torch.class('nn.JoinTable', 'nn.Module')

function JoinTable:__init(dimension)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.dimension = dimension
   self.gradInput = {}
end 

function JoinTable:forward(input) 
   for i=1,#input do
      local currentOutput = input[i]
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[self.dimension] = self.size[self.dimension] 
				     + currentOutput:size(self.dimension)
      end 
   end
   self.output:resize(self.size)
    
   local offset = 1  
   for i=1,#input do
      local currentOutput = input[i]
      self.output:narrow(self.dimension, offset, 
			 currentOutput:size(self.dimension)):copy(currentOutput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.output

end


function JoinTable:backward(input, gradOutput)
   for i=1,#input do 
	  if self.gradInput[i]==nil then
		self.gradInput[i]=torch.Tensor();
	  end
	  self.gradInput[i]:resizeAs(input[i])
   end

   local offset = 1
   for i=1,#input do
      local currentOutput = input[i] 
      local currentGradInput = gradOutput:narrow(self.dimension, offset, 
					  currentOutput:size(self.dimension))
      self.gradInput[i]:copy(currentGradInput)
      offset = offset + currentOutput:size(self.dimension)
   end
   return self.gradInput
end

function JoinTable:zeroGradParameters()
end

function JoinTable:updateParameters(learningRate)
end

function JoinTable:write(file)
   parent.write(self, file)
   file:writeObject(self.size)
   file:writeInt(self.dimension)
end

function JoinTable:read(file, version)
   parent.read(self, file)
   if version > 0 then
      self.size = file:readObject()
   else
      local size = file:readObject()
      self.size = torch.LongStorage(size:size())
      self.size:copy(size)
   end
   self.dimension = file:readInt()
end
