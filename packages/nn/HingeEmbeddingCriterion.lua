local HingeEmbeddingCriterion, parent = 
	torch.class('nn.HingeEmbeddingCriterion', 'nn.Module')

function HingeEmbeddingCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1 
   self.margin = margin 
   self.gradInput = torch.Tensor(1)
end 
 
function HingeEmbeddingCriterion:forward(input,y)
   self.output=input[1]
   if y==-1 then
	 self.output = math.max(0,self.margin - self.output);
   end
   return self.output
end

function HingeEmbeddingCriterion:backward(input, y)
  self.gradInput[1]=y
  local dist = input[1]
  if y == -1 and  dist > self.margin then
     self.gradInput[1]=0;
  end
  return self.gradInput 
end


function HingeEmbeddingCriterion:write(file)
   parent.write(self, file)
   file:writeDouble(self.margin)
end

function HingeEmbeddingCriterion:read(file)
   parent.read(self, file)
   self.margin = file:readDouble()
end



