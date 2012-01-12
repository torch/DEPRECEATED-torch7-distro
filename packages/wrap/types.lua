wrap.argtypes = {}

wrap.argtypes.Tensor = {

   helpname = function(arg)
               return "Tensor"
            end,

   declare = function(arg)
              return string.format("THTensor *arg%d = NULL;", arg.i)
           end,

   init = function(arg)
          end,
   
   check = function(arg, idx)
            return string.format("luaT_isudata(L, %d, torch_(Tensor_id))", idx)
         end,

   read = function(arg, idx)
             return string.format("arg%d = luaT_toudata(L, %d, torch_(Tensor_id));", arg.i, idx)
          end,
   
   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
                local txt = {}
                if arg.default and arg.returned then
                   table.insert(txt, string.format('if(arg%d)', arg.i))
                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                   table.insert(txt, 'else')
                   table.insert(txt, string.format('arg%d = THTensor_(new)();', arg.i))
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                elseif arg.default then
                   error('a tensor cannot be optional if not returned')
                elseif arg.returned then
                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                end
                return table.concat(txt, '\n')
             end,

   postcall = function(arg)
                 local txt = {}
                 if arg.creturned then
                    -- this next line is actually debatable
                    table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                 end
                 return table.concat(txt, '\n')
              end
}

wrap.argtypes.IndexTensor = {

   helpname = function(arg)
               return "LongTensor"
            end,

   declare = function(arg)
              return string.format("THLongTensor *arg%d = NULL;", arg.i)
           end,
   
   init = function(arg)
             return ''
          end,

   check = function(arg, idx)
            return string.format("luaT_isudata(L, %d, torch_LongTensor_id)", idx)
         end,

   read = function(arg, idx)
             local txt = {}
             table.insert(txt, string.format("arg%d = luaT_toudata(L, %d, torch_LongTensor_id);", arg.i, idx))
             table.insert(txt, string.format("THLongTensor_add(arg%d, -1);", arg.i));
             return table.concat(txt, '\n')
          end,
   
   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
                local txt = {}
                if arg.default and arg.returned then
                   table.insert(txt, string.format('if(arg%d)', arg.i))
                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                   table.insert(txt, 'else')
                   table.insert(txt, string.format('arg%d = THTensor_(new)();', arg.i))
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                elseif arg.default then
                   error('a tensor cannot be optional if not returned')
                elseif arg.returned then
                   table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                   table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                end
                return table.concat(txt, '\n')
             end,

   postcall = function(arg)
                 local txt = {}
                 if arg.creturned or arg.returned then
                    table.insert(txt, string.format("THLongTensor_add(arg%d, 1);", arg.i));
                 end
                 if arg.creturned then
                    -- this next line is actually debatable
                    table.insert(txt, string.format('THTensor_(retain)(arg%d);', arg.i))
                    table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_(Tensor_id));', arg.i))
                 end
                 return table.concat(txt, '\n')
              end
}

for _,typename in ipairs({"ByteTensor", "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
                          "FloatTensor", "DoubleTensor"}) do

   wrap.argtypes[typename] = {

      helpname = function(arg)
                    return typename
                 end,
      
      declare = function(arg)
                   return string.format("TH%s *arg%d = NULL;", typename, arg.i)
                end,
      
      init = function(arg)
                return ''
             end,

      check = function(arg, idx)
                 return string.format("luaT_isudata(L, %d, torch_%s_id)", idx, typename)
              end,

      read = function(arg, idx)
                return string.format("arg%d = luaT_toudata(L, %d, torch_%s_id);", arg.i, idx, typename)
             end,
      
      carg = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,

      creturn = function(arg, idx)
                   return string.format('arg%d', arg.i)
             end,
      
      precall = function(arg)
                   local txt = {}
                   if arg.default and arg.returned then
                      table.insert(txt, string.format('if(arg%d)', arg.i))
                      table.insert(txt, string.format('TH%s_retain(arg%d);', typename, arg.i))
                      table.insert(txt, 'else')
                      table.insert(txt, string.format('arg%d = TH%s_new();', arg.i, typename))
                      table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_%s_id);', arg.i, typename))
                   elseif arg.default then
                      error('a tensor cannot be optional if not returned')
                   elseif arg.returned then
                      table.insert(txt, string.format('TH%s_retain(arg%d);', typename, arg.i))
                      table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_%s_id);', arg.i, typename))
                   end
                   return table.concat(txt, '\n')
                end,

      postcall = function(arg)
                    local txt = {}
                    if arg.creturned then
                       -- this next line is actually debatable
                       table.insert(txt, string.format('TH%s_retain(arg%d);', typename, arg.i))
                       table.insert(txt, string.format('luaT_pushudata(L, arg%d, torch_%s_id);', arg.i, typename))
                    end
                    return table.concat(txt, '\n')
                 end
   }
end

local function interpretdefaultvalue(arg)
   local default = arg.default
   if type(default) == 'boolean' then
      if default then
         return '1'
      else
         return '0'
      end
   elseif type(default) == 'number' then
      return tostring(default)
   elseif type(default) == 'string' then
      return default
   elseif type(default) == 'function' then
      default = default(arg)
      assert(type(default) == 'string', 'a default function must return a string')
      return default
   elseif type(default) == 'nil' then
      return nil
   else
      error('unknown default type value')
   end   
end

wrap.argtypes.index = {

   helpname = function(arg)
               return "index"
            end,

   declare = function(arg)
                -- if it is a number we initialize here
                local default = tonumber(interpretdefaultvalue(arg)) or 1
                return string.format("long arg%d = %d;", arg.i, tonumber(default)-1)
           end,

   init = function(arg)
             -- otherwise do it here
             if arg.default then
                local default = interpretdefaultvalue(arg)
                if not tonumber(default) then
                   return string.format("arg%d = %s-1;", arg.i, default)
                end
             end
          end,

   check = function(arg, idx)
              return string.format("lua_isnumber(L, %d)", idx)
           end,

   read = function(arg, idx)
             return string.format("arg%d = (long)lua_tonumber(L, %d)-1;", arg.i, idx)
          end,

   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,

   precall = function(arg)
                if arg.returned then
                   return string.format('lua_pushnumber(L, (lua_Number)arg%d+1);', arg.i)
                end
             end,

   postcall = function(arg)
                 if arg.creturned then
                    return string.format('lua_pushnumber(L, (lua_Number)arg%d+1);', arg.i)
                 end
              end
}

wrap.argtypes.byte = {

   helpname = function(arg)
               return "byte"
            end,

   declare = function(arg)
                -- if it is a number we initialize here
                local default = tonumber(interpretdefaultvalue(arg)) or 0
                return string.format("unsigned char arg%d = %d;", arg.i, tonumber(default))
           end,

   init = function(arg)
             -- otherwise do it here
             if arg.default then
                local default = interpretdefaultvalue(arg)
                if not tonumber(default) then
                   return string.format("arg%d = %s;", arg.i, default)
                end
             end
          end,

   check = function(arg, idx)
              return string.format("lua_isnumber(L, %d)", idx)
           end,

   read = function(arg, idx)
             return string.format("arg%d = (unsigned char)lua_tonumber(L, %d);", arg.i, idx)
          end,

   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,

   precall = function(arg)
                if arg.returned then
                   return string.format('lua_pushnumber(L, (lua_Number)arg%d);', arg.i)
                end
             end,

   postcall = function(arg)
                 if arg.creturned then
                    return string.format('lua_pushnumber(L, (lua_Number)arg%d);', arg.i)
                 end
              end
}

for _,typename in ipairs({"real", "char", "short", "int", "long", "float", "double"}) do
   wrap.argtypes[typename] = {

      helpname = function(arg)
                    return typename
                 end,

      declare = function(arg)
                   -- if it is a number we initialize here
                   local default = tonumber(interpretdefaultvalue(arg)) or 0
                   return string.format("%s arg%d = %d;", typename, arg.i, tonumber(default))
                end,

      init = function(arg)
                -- otherwise do it here
                if arg.default then
                   local default = interpretdefaultvalue(arg)
                   if not tonumber(default) then
                      return string.format("arg%d = %s;", arg.i, default)
                   end
                end
             end,
      
      check = function(arg, idx)
                 return string.format("lua_isnumber(L, %d)", idx)
              end,
      
      read = function(arg, idx)
                return string.format("arg%d = (%s)lua_tonumber(L, %d);", arg.i, typename, idx)
             end,
      
      carg = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,

      creturn = function(arg, idx)
                   return string.format('arg%d', arg.i)
                end,
      
      precall = function(arg)
                   if arg.returned then
                      return string.format('lua_pushnumber(L, (lua_Number)arg%d);', arg.i)
                   end
                end,
      
      postcall = function(arg)
                    if arg.creturned then
                       return string.format('lua_pushnumber(L, (lua_Number)arg%d);', arg.i)
                    end
                 end
   }
end

wrap.argtypes.boolean = {

   helpname = function(arg)
                 return "boolean"
              end,

   declare = function(arg)
                -- if it is a number we initialize here
                local default = tonumber(interpretdefaultvalue(arg)) or 0
                return string.format("int arg%d = %d;", arg.i, tonumber(default))
             end,

   init = function(arg)
             -- otherwise do it here
             if arg.default then
                local default = interpretdefaultvalue(arg)
                if not tonumber(default) then
                   return string.format("arg%d = %s;", arg.i, default)
                end
             end
          end,
             
   check = function(arg, idx)
              return string.format("lua_isboolean(L, %d)", idx)
           end,

   read = function(arg, idx)
             return string.format("arg%d = lua_toboolean(L, %d);", arg.i, idx)
          end,

   carg = function(arg, idx)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg, idx)
                return string.format('arg%d', arg.i)
             end,

   precall = function(arg)
                if arg.returned then
                   return string.format('lua_pushboolean(L, arg%d);', arg.i)
                end
             end,

   postcall = function(arg)
                 if arg.creturned then
                    return string.format('lua_pushboolean(L, arg%d);', arg.i)
                 end
              end
}
