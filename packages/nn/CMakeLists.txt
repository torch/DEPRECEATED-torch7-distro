SET(src init.c HardTanh.c Exp.c LogSoftMax.c LogSigmoid.c Sigmoid.c SoftPlus.c Tanh.c SoftMax.c Mean.c Max.c Min.c Sum.c SpatialConvolution.c SpatialSubSampling.c TemporalConvolution.c TemporalSubSampling.c MSECriterion.c AbsCriterion.c SparseLinear.c)
SET(luasrc init.lua Linear.lua SparseLinear.lua CMul.lua Mul.lua Add.lua Euclidean.lua DotProduct.lua CosineDistance.lua PairwiseDistance.lua Sequential.lua CriterionTable.lua ParallelTable.lua ConcatTable.lua SplitTable.lua JoinTable.lua Identity.lua HardTanh.lua Exp.lua Module.lua LogSoftMax.lua LogSigmoid.lua Sigmoid.lua SoftPlus.lua Reshape.lua Select.lua Tanh.lua Concat.lua Parallel.lua SoftMax.lua Mean.lua Max.lua Min.lua Sum.lua SpatialConvolution.lua SpatialSubSampling.lua TemporalConvolution.lua TemporalSubSampling.lua Criterion.lua MSECriterion.lua AbsCriterion.lua ClassNLLCriterion.lua MultiCriterion.lua L1HingeEmbeddingCriterion.lua HingeEmbeddingCriterion.lua MarginCriterion.lua StochasticGradient.lua LookupTable.lua CosineEmbeddingCriterion.lua MarginRankingCriterion.lua)

ADD_TORCH_PACKAGE(nn "${src}" "${luasrc}" "Machine Learning")

TARGET_LINK_LIBRARIES(nn luaT TH)
