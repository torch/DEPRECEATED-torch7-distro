#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalLogSoftMax.c"
#else

static int nn_(TemporalLogSoftMax_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  long t, d;

  if(input->nDimension != 2)
    luaL_error(L, "input: invalid number of dimension (expected 2)");
  
  THTensor_(resizeAs)(output, input);
  
  for(t = 0; t < input->size[0]; t++)  
  {
    accreal logsum = 0;
    real maxInput = -THInf;

    for(d = 0; d < input->size[1]; d++)
      maxInput = THMax(maxInput, THTensor_fastGet2d(input, t, d));

    for(d = 0; d < input->size[1]; d++)
      logsum += THExpMinusApprox(maxInput-THTensor_fastGet2d(input, t, d));
    logsum = maxInput + log(logsum);
    
    for(d = 0; d < input->size[1]; d++)
      THTensor_fastSet2d(output, t, d, THTensor_fastGet2d(input, t, d)-logsum);
  }

  return 1;
}

static int nn_(TemporalLogSoftMax_backward)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  long t, d;

  THTensor_(resizeAs)(gradInput, output);
  for(t = 0; t < output->size[0]; t++)
  {
    accreal sum = 0;
    for(d = 0; d < output->size[1]; d++)
      sum += THTensor_fastGet2d(gradOutput, t, d);

    for(d = 0; d < output->size[1]; d++)
      THTensor_fastSet2d(gradInput, t, d, 
                         THTensor_fastGet2d(gradOutput, t, d) - exp(THTensor_fastGet2d(output, t, d))*sum);
  }

  return 1;
}

static const struct luaL_Reg nn_(TemporalLogSoftMax__) [] = {
  {"TemporalLogSoftMax_forward", nn_(TemporalLogSoftMax_forward)},
  {"TemporalLogSoftMax_backward", nn_(TemporalLogSoftMax_backward)},
  {NULL, NULL}
};

void nn_(TemporalLogSoftMax_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(TemporalLogSoftMax__), "nn");
  lua_pop(L,1);
}

#endif
