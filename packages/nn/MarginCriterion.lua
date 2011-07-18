local MarginCriterion, parent = 
	torch.class('nn.MarginCriterion', 'nn.Module')

function MarginCriterion:__init(margin)
   parent.__init(self)
   margin=margin or 1   
   self.margin = margin 
   self.gradInput = torch.Tensor(1)
end 
 
function MarginCriterion:forward(input,y)
   self.output=math.max(0, self.margin- y* input[1])
   return self.output
end

function MarginCriterion:backward(input, y)
  if (y*input[1])<self.margin then
     self.gradInput[1]=-y		
  else
     self.gradInput[1]=0;
  end
  return self.gradInput 
end


function MarginCriterion:write(file)
   parent.write(self, file)
   file:writeDouble(self.margin)
end

function MarginCriterion:read(file)
   parent.read(self, file)
   self.margin = file:readDouble()
end



