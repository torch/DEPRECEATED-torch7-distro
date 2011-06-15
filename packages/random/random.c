#include "luaT.h"
#include "TH.h"

static int random_seed(lua_State *L)
{
  lua_pushnumber(L, THRandom_seed());
  return 1;
}

static int random_manualSeed(lua_State *L)
{
  long seed = luaL_checklong(L, 1);
  THRandom_manualSeed(seed);
  return 0;
}

static int random_initialSeed(lua_State *L)
{
  lua_pushnumber(L, THRandom_initialSeed());
  return 1;
}

static int random_random(lua_State *L)
{
  lua_pushnumber(L, THRandom_random());
  return 1;
}

static int random_uniform(lua_State *L)
{
  double a = luaL_optnumber(L, 1, 0);
  double b = luaL_optnumber(L, 2, 1);
  lua_pushnumber(L, THRandom_uniform(a, b));
  return 1;
}

static int random_normal(lua_State *L)
{
  double a = luaL_optnumber(L, 1, 0);
  double b = luaL_optnumber(L, 2, 1);
  luaL_argcheck(L, b > 0, 2, "positive number required");
  lua_pushnumber(L, THRandom_normal(a, b));
  return 1;
}

static int random_exponential(lua_State *L)
{
  double lambda = luaL_checknumber(L, 1);
  lua_pushnumber(L, THRandom_exponential(lambda));
  return 1;
}

static int random_cauchy(lua_State *L)
{
  double median = luaL_optnumber(L, 1, 0);
  double sigma  = luaL_optnumber(L, 2, 1);
  lua_pushnumber(L, THRandom_cauchy(median, sigma));
  return 1;
}

static int random_logNormal(lua_State *L)
{
  double mean = luaL_checknumber(L, 1);
  double stdv = luaL_checknumber(L, 2);
  luaL_argcheck(L, stdv > 0, 2, "positive number required");
  lua_pushnumber(L, THRandom_logNormal(mean, stdv));
  return 1;
}

static int random_geometric(lua_State *L)
{
  double p = luaL_checknumber(L, 1);
  luaL_argcheck(L, p > 0 && p < 1, 2, "must be > 0 and < 1");
  lua_pushnumber(L, THRandom_geometric(p));
  return 1;
}

static int random_bernoulli(lua_State *L)
{
  double p = luaL_optnumber(L, 1, 0.5);
  luaL_argcheck(L, p > 0 && p < 1, 2, "must be > 0 and < 1");
  lua_pushnumber(L, THRandom_bernoulli(p));
  return 1;
}

static const struct luaL_Reg random__ [] = {
  {"seed", random_seed},
  {"manualSeed", random_manualSeed},
  {"initialSeed", random_initialSeed},
  {"random", random_random},
  {"uniform", random_uniform},
  {"normal", random_normal},
  {"exponential", random_exponential},
  {"cauchy", random_cauchy},
  {"logNormal", random_logNormal},
  {"geometric", random_geometric},
  {"bernoulli", random_bernoulli},
  {NULL, NULL}
};

DLL_EXPORT int luaopen_librandom(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "random");
  luaL_register(L, NULL, random__);
  return 1;
}
