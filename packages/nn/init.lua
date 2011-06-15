require('torch')
require('random')
require('libnn')

torch.include('nn', 'Module.lua')
 
torch.include('nn', 'Concat.lua')
torch.include('nn', 'Parallel.lua')  
torch.include('nn', 'Sequential.lua')

torch.include('nn', 'Linear.lua')
torch.include('nn', 'SparseLinear.lua')
torch.include('nn', 'Reshape.lua')  
torch.include('nn', 'Select.lua')
  
torch.include('nn', 'Min.lua')
torch.include('nn', 'Max.lua')
torch.include('nn', 'Mean.lua')
torch.include('nn', 'Sum.lua')


torch.include('nn', 'CMul.lua')  
torch.include('nn', 'Mul.lua')  
torch.include('nn', 'Add.lua')  

torch.include('nn', 'Euclidean.lua')  
torch.include('nn', 'PairwiseDistance.lua')  
torch.include('nn', 'CosineDistance.lua')  
torch.include('nn', 'DotProduct.lua')  

torch.include('nn', 'Exp.lua')
torch.include('nn', 'HardTanh.lua')
torch.include('nn', 'LogSigmoid.lua')
torch.include('nn', 'LogSoftMax.lua')
torch.include('nn', 'Sigmoid.lua')
torch.include('nn', 'SoftMax.lua')
torch.include('nn', 'SoftPlus.lua')
torch.include('nn', 'Tanh.lua')

torch.include('nn', 'LookupTable.lua')
torch.include('nn', 'SpatialConvolution.lua')
torch.include('nn', 'SpatialSubSampling.lua')
torch.include('nn', 'TemporalConvolution.lua')
torch.include('nn', 'TemporalSubSampling.lua')

torch.include('nn', 'ParallelTable.lua')  
torch.include('nn', 'ConcatTable.lua')  
torch.include('nn', 'SplitTable.lua')  
torch.include('nn', 'JoinTable.lua')  
torch.include('nn', 'CriterionTable.lua')
torch.include('nn', 'Identity.lua')  

torch.include('nn', 'Criterion.lua')
torch.include('nn', 'MSECriterion.lua')
torch.include('nn', 'MarginCriterion.lua')
torch.include('nn', 'AbsCriterion.lua')
torch.include('nn', 'ClassNLLCriterion.lua')
torch.include('nn', 'MultiCriterion.lua')
torch.include('nn', 'L1HingeEmbeddingCriterion.lua')
torch.include('nn', 'HingeEmbeddingCriterion.lua')
torch.include('nn', 'CosineEmbeddingCriterion.lua')
torch.include('nn', 'MarginRankingCriterion.lua')


torch.include('nn', 'StochasticGradient.lua')
