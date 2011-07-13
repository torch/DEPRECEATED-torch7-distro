function torch.CudaTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
end
