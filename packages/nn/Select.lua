local Select, parent = torch.class('nn.Select', 'nn.Module')

function Select:__init(dimension,index)
   parent.__init(self)
   self.dimension=dimension
   self.index=index 
end

function Select:forward(input)
   local output=input:select(self.dimension,self.index);
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function Select:backward(input, gradOutput)
   self.gradInput:resizeAs(input)  
   self.gradInput:zero();
   self.gradInput:select(self.dimension,self.index):copy(gradOutput) 
   return self.gradInput
end 

function Select:write(file) 
   parent.write(self, file)
   file:writeInt(self.dimension)
   file:writeLong(self.index)
end

function Select:read(file, version)
   parent.read(self, file)
   self.dimension = file:readInt()
   if version > 0 then
      self.index = file:readLong()
   else
      self.index = file:readInt()
   end
end
