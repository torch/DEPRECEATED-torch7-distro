local ffi = require 'ffi'
require "paths"

-- torch is global for now [due to getmetatable()]
torch = {}

-- load TH and setup error handlers
local TH = ffi.load('/Users/ronan/usr7/lib/libTH.dylib')
torch.TH = TH

ffi.cdef([[
void THSetErrorHandler( void (*torchErrorHandlerFunction)(const char *msg) );
void THSetArgErrorHandler( void (*torchArgErrorHandlerFunction)(int argNumber, const char *msg) );
]])

TH.THSetErrorHandler(function(str)
                        error(str)
                     end)

TH.THSetArgErrorHandler(function(argnumber, str)
                           if str then
                              error(string.format("invalid argument %d: %s", argnumber, ffi.string(str)))
                           else
                              error(string.format("invalid argument %d", argnumber))
                           end
                        end)

-- adapt usual global functions to torch7 objects
local luatostring = tostring
function tostring(arg)
   local flag, func = pcall(function(arg) return arg.__tostring end, arg)
   if flag and func then
      return func(arg)
   end
   return luatostring(arg)
end

local luatype = type
function type(arg)
   local flag, type = pcall(function(arg) return arg.__typename end, arg)
   if flag and type then
      return type
   end
   return luatype(arg)
end

torch.typename = type -- backward compatibility... keep it or not?

function torch.getmetatable(str)
   local module, name = str:match('([^%.]+)%.(.+)')   
   local rtbl = _G[module][name]
   if rtbl then 
      return getmetatable(rtbl)
   end
end

function include(file, env)
   if env then
      local filename = paths.thisfile(file, 3)
      local f = io.open(filename)
      local txt = f:read('*all')
      f:close()
      local code, err = loadstring(txt, filename)
      if not code then
         error(err)
      end
      setfenv(code, env)
      code()      
   else
      paths.dofile(file, 3)
   end
end

function torch.class(tname, parenttname)

   local function constructor(...)
      local self = {}
      torch.setmetatable(self, tname)
      if self.__init then
         self:__init(...)
      end
      return self
   end
   
   local function factory()
      local self = {}
      torch.setmetatable(self, tname)
      return self
   end

   local mt = torch.newmetatable(tname, parenttname, constructor, nil, factory)
   local mpt
   if parenttname then
      mpt = torch.getmetatable(parenttname)
   end
   return mt, mpt
end

function torch.setdefaulttensortype(typename)
   assert(type(typename) == 'string', 'string expected')
   if torch.getconstructortable(typename) then
      torch.Tensor = torch.getconstructortable(typename)
      torch.Storage = torch.getconstructortable(torch.typename(torch.Tensor(1):storage()))
   else
      error(string.format("<%s> is not a string describing a torch object", typename))
   end
end

local function includetemplate(file, env)
   env = env or _G
   local filename = paths.thisfile(file, 3)
   local f = io.open(filename)
   local txt = f:read('*all')
   f:close()
   local types = {char='Char', short='Short', int='Int', long='Long', float='Float', double='Double'}
   types['unsigned char'] = 'Byte'
   for real,Real in pairs(types) do
      local txt = txt:gsub('([%p%s])real([%p%s])', '%1' .. real .. '%2')
      txt = txt:gsub('THStorage', 'TH' .. Real .. 'Storage')
      txt = txt:gsub('THTensor', 'TH' .. Real .. 'Tensor')
      txt = txt:gsub('([%p%s])Storage([%p%s])', '%1' .. Real .. 'Storage' .. '%2')
      txt = txt:gsub('([%p%s])Tensor([%p%s])', '%1' .. Real .. 'Tensor' .. '%2')
      local code, err = loadstring(txt, filename)
      if not code then
         error(err)
      end
      setfenv(code, env)
      code()
   end   
end

--torch.setdefaulttensortype('torch.DoubleTensor')

local env = {ffi=ffi, torch=torch, TH=TH}
setmetatable(env, {__index=_G})

includetemplate('Storage.lua', env)
includetemplate('StorageCopy.lua', env)
includetemplate('Tensor.lua', env)
includetemplate('TensorCopy.lua', env)
include('print.lua')
include('TensorMath.lua', env)

--include('File.lua')
--include('CmdLine.lua')
--include('Tester.lua')
--include('test.lua')

return torch
