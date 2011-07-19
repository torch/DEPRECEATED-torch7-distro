local Replicate, parent = torch.class('nn.Replicate','nn.Module')

function Replicate:__init(nf)
   parent.__init(self)
   self.nfeatures = nf
end

function Replicate:forward(input)
   local sz = torch.LongStorage(input:dim()+1)
   sz[1] = self.nfeatures
   for i = 1,input:dim() do
      sz[i+1] = input:size(i)
   end
   local st = torch.LongStorage(input:dim()+1)
   st[1] = 0
   for i = 1,input:dim() do
      st[i+1] = input:stride(i)
   end
   self.output = torch.Tensor(input:storage(),input:storageOffset(),sz,st)
   return self.output
end

function Replicate:backward(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   for k = 1,gradOutput:size(1) do
      self.gradInput:add(gradOutput[k])
   end
   return self.gradInput
end

function Replicate:write(file)
   parent.write(self,file)
   file:writeInt(self.nfeatures)
end

function Replicate:read(file)
   parent.read(self,file)
   self.nfeatures = file:readInt()
end
