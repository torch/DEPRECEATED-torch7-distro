wrap.argtypes = {}

wrap.argtypes.Tensor = {

   helpname = function(arg)
               return "Tensor"
            end,

   declare = function(arg)
              return string.format("THTensor *arg%d = NULL;", arg.i)
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

for _,typename in ipairs({"ByteTensor", "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
                          "FloatTensor", "TensorDouble"}) do

   wrap.argtypes[typename] = {

      helpname = function(arg)
                    return typename
                 end,
      
      declare = function(arg)
                   return string.format("TH%s *arg%d = NULL;", typename, arg.i)
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

wrap.argtypes.integer = {

   helpname = function(arg)
               return "integer"
            end,

   declare = function(arg)
              return string.format("long arg%d = %d;", arg.i, arg.default or 0)
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

for _,typename in ipairs({"real", "char", "short", "int", "long", "float", "double"}) do
   wrap.argtypes[typename] = {

      helpname = function(arg)
                    return typename
                 end,

      declare = function(arg)
                 return string.format("%s arg%d = %d;", typename, arg.i, arg.default or 0)
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
              local default = 0
              if arg.default then
                 default = 1
              end
              return string.format("int arg%d = %d;", arg.i, default or 0)
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
