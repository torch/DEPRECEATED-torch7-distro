SET(src init.c)

SET(luasrc init.lua Linear.lua SparseLinear.lua Copy.lua CMul.lua
           Mul.lua Add.lua Euclidean.lua DotProduct.lua CosineDistance.lua
           CAddTable.lua CMulTable.lua CSubTable.lua CDivTable.lua
           PairwiseDistance.lua Sequential.lua CriterionTable.lua
           ParallelTable.lua ConcatTable.lua SplitTable.lua JoinTable.lua
           Identity.lua HardTanh.lua HardShrink.lua Exp.lua Module.lua
           LogSoftMax.lua TemporalLogSoftMax.lua LogSigmoid.lua Sigmoid.lua
           SoftPlus.lua Reshape.lua Select.lua Narrow.lua Abs.lua
           Tanh.lua Concat.lua Parallel.lua Replicate.lua
           Power.lua Sqrt.lua Square.lua Threshold.lua
           SoftMax.lua Mean.lua Max.lua Min.lua Sum.lua SpatialConvolution.lua
           SpatialSubSampling.lua TemporalConvolution.lua TemporalSubSampling.lua
           Criterion.lua MSECriterion.lua AbsCriterion.lua ClassNLLCriterion.lua
           TemporalClassNLLCriterion.lua MultiCriterion.lua
           L1HingeEmbeddingCriterion.lua HingeEmbeddingCriterion.lua
           MarginCriterion.lua StochasticGradient.lua LookupTable.lua
           CosineEmbeddingCriterion.lua MarginRankingCriterion.lua Jacobian.lua
           test/test.lua)

ADD_TORCH_PACKAGE(nn "${src}" "${luasrc}" "Machine Learning")

TARGET_LINK_LIBRARIES(nn luaT TH)
