local Threshold, parent = torch.class('nn.Threshold','nn.Module')

function Threshold:__init(th,v)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.Threshold(threshold, value)')
   end
end

function Threshold:forward(input)
   input.nn.Threshold_forward(self, input)
   return self.output
end

function Threshold:backward(input, gradOutput)
   input.nn.Threshold_backward(self, input, gradOutput)
   return self.gradInput
end
