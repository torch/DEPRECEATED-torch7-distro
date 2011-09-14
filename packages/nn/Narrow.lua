local Narrow, parent = torch.class('nn.Narrow', 'nn.Module')

function Narrow:__init(dimension,offset,length)
   parent.__init(self)
   self.dimension=dimension
   self.index=offset
   self.length=length or 1
   if not dimension or not offset then
      error('nn.Narrow(dimension, offset, length)')
   end
end

function Narrow:forward(input)
   local output=input:narrow(self.dimension,self.index,self.length);
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function Narrow:backward(input, gradOutput)
   self.gradInput:resizeAs(input)  
   self.gradInput:zero();
   self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
   return self.gradInput
end 

function Narrow:write(file) 
   parent.write(self, file)
   file:writeInt(self.dimension)
   file:writeLong(self.index)
   file:writeLong(self.length)
end

function Narrow:read(file, version)
   parent.read(self, file)
   self.dimension = file:readInt()
   self.index = file:readLong()
   self.length = file:readLong()
end
