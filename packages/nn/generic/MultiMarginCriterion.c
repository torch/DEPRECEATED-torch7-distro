#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiMarginCriterion.c"
#else

static int nn_(MultiMarginCriterion_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  long target = luaL_checklong(L, 3)-1;  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  real sum, input_target, *input_data;
  long input_size, i;
  
  THArgCheck(input->nDimension == 1, 2, "vector expected");
  THArgCheck((target >= 0) && (target < input->size[0]), 3, "target out of range");

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  input_size = input->size[0];
  input_target = input_data[target];

  sum = 0;
  for(i = 0; i < input_size; i++)
  {
    real z = 1 - input_target + input_data[i];
    if(i == target)
      continue;
    
    if(z > 0)
      sum += z;
  }

  if(sizeAverage)
    sum /= input_size;

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  THTensor_(free)(input);
  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(MultiMarginCriterion_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  long target = luaL_checklong(L, 3)-1;
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));
  real *input_data;
  real input_target;
  real *gradInput_data;
  long input_size = input->size[0];
  long i;
  real gradInput_target = 0;

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  input_target = input_data[target];

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);
  
  for(i = 0; i < input_size; i++)
  {
    real z = 1 - input_target + input_data[i];
    if(i == target)
      continue;
    
    if(z > 0)
    {
      gradInput_target -= 1;
      gradInput_data[i] = 1;
    }
    else
      gradInput_data[i] = 0;
  }
  gradInput_data[target] = gradInput_target;

  if(sizeAverage)
    THTensor_(mul)(gradInput, 1./((real)input_size));

  THTensor_(free)(input);  
  return 1;
}

static const struct luaL_Reg nn_(MultiMarginCriterion__) [] = {
  {"MultiMarginCriterion_forward", nn_(MultiMarginCriterion_forward)},
  {"MultiMarginCriterion_backward", nn_(MultiMarginCriterion_backward)},
  {NULL, NULL}
};

static void nn_(MultiMarginCriterion_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(MultiMarginCriterion__), "nn");
  lua_pop(L,1);
}

#endif
