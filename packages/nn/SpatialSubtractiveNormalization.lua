local SpatialSubtractiveNormalization, parent = torch.class('nn.SpatialSubtractiveNormalization','nn.Module')

function SpatialSubtractiveNormalization:__init(nInputPlane, kernel)
   parent.__init(self)

   -- get args
   self.nInputPlane = nInputPlane or 1
   self.kernel = kernel or torch.Tensor(9,9):fill(1)

   -- check args
   if self.kernel:nDimension() ~= 2 then
      error('<SpatialSubtractiveNormalization> averaging kernel must be 2D')
   end
   if (self.kernel:size(1) % 2) == 0 or (self.kernel:size(2) % 2) == 0 then
      error('<SpatialSubtractiveNormalization> averaging kernel must have ODD dimensions')
   end

   -- normalize kernel
   self.kernel:div(self.kernel:sum() * self.nInputPlane)

   -- padding values
   local padW = math.floor(self.kernel:size(2)/2)
   local padH = math.floor(self.kernel:size(1)/2)

   -- create convolutional mean extractor
   self.meanestimator = nn.Sequential()
   self.meanestimator:add(nn.SpatialZeroPadding(padW, padW, padH, padH))
   self.meanestimator:add(nn.SpatialConvolutionMap(nn.tables.oneToOne(self.nInputPlane),
                                                   self.kernel:size(2), self.kernel:size(1)))
   self.meanestimator:add(nn.Sum(1))
   self.meanestimator:add(nn.Replicate(self.nInputPlane))

   -- set kernel and bias
   for i = 1,self.nInputPlane do 
      self.meanestimator.modules[2].weight[i] = self.kernel
   end
   self.meanestimator.modules[2].bias:zero()

   -- other operation
   self.subtractor = nn.CSubTable()
   self.divider = nn.CDivTable()

   -- coefficient array, to adjust side effects
   self.coef = torch.Tensor(1,1,1)
end

function SpatialSubtractiveNormalization:forward(input)
   -- compute side coefficients
   if (input:size(3) ~= self.coef:size(2)) or (input:size(2) ~= self.coef:size(1)) then
      local ones = input.new():resizeAs(input):fill(1)
      self.coef = self.meanestimator:forward(ones)
      self.coef = self.coef:clone()
   end

   -- compute mean
   self.localsums = self.meanestimator:forward(input)
   self.adjustedsums = self.divider:forward{self.localsums, self.coef}
   self.output = self.subtractor:forward{input, self.adjustedsums}

   -- done
   return self.output
end

function SpatialSubtractiveNormalization:backward(input, gradOutput)
   -- resize grad
   self.gradInput:resizeAs(input):zero()

   -- backprop through all modules
   local gradsub = self.subtractor:backward({input, self.adjustedsums}, gradOutput)
   local graddiv = self.divider:backward({self.localsums, self.coef}, gradsub[2])
   self.gradInput:add(self.meanestimator:backward(input, graddiv[1]))
   self.gradInput:add(gradsub[1])

   -- done
   return self.gradInput
end

function SpatialSubtractiveNormalization:type(type)
   parent.type(self,type)
   self.meanestimator:type(type)
   self.divider:type(type)
   self.subtractor:type(type)
   return self
end
