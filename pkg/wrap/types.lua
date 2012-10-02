wrap.argtypes = {}

wrap.argtypes.IndexTensor = {

   helpname = function(arg)
               return "LongTensor"
            end,

   declare = function(arg)
                return string.format("local arg%d", arg.i)
           end,
   
   check = function(arg, idx)
              return string.format('type(arg[%d]) == "torch.LongTensor"', idx)
           end,

   read = function(arg, idx)
             local txt = {}
             table.insert(txt, string.format("arg%d = arg[%d]", arg.i, idx))
             table.insert(txt, string.format("arg%d:add(-1)", arg.i, arg.i));
             return table.concat(txt, '\n')
          end,

   init = function(arg)
             return string.format('arg%d = torch.LongTensor()', arg.i)
          end,
   
   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
             end,

   postcall = function(arg)
                 if arg.creturned or arg.returned then
                    return string.format("arg%d:add(1)", arg.i)
                 end
              end
}

for _,typename in ipairs({"ByteTensor", "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
                          "FloatTensor", "DoubleTensor"}) do

   wrap.argtypes[typename] = {

      helpname = function(arg)
                    if arg.dim then
                       return string.format('%s~%dD', typename, arg.dim)
                    else
                       return typename
                    end
                 end,
      
      declare = function(arg)
                   return string.format("local arg%d", arg.i)
                end,
      
      check = function(arg, idx)
                 if arg.dim then
                    return string.format('type(arg[%d]) == "torch.%s" and arg[%d].__nDimension == %d', idx, typename, idx, arg.dim)
                 else
                    return string.format('type(arg[%d]) == "torch.%s"', idx, typename)
                 end
              end,

      read = function(arg, idx)
                return string.format('arg%d = arg[%d]', arg.i, idx)
             end,
      
      init = function(arg)
                if type(arg.default) == 'boolean' then
                   return string.format('arg%d = torch.%s()', arg.i, typename)
                elseif type(arg.default) == 'number' then
                   return string.format('arg%d = %s', arg.i, arg.args[arg.default]:carg())
                else
                   error('unknown default tensor type value')
                end
             end,

      carg = function(arg)
                return string.format('arg%d', arg.i)
             end,

      creturn = function(arg)
                   return string.format('arg%d', arg.i)
             end,
      
      precall = function(arg)
                end,

      postcall = function(arg)
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
                return string.format("local arg%d = %d", arg.i, tonumber(default)-1)
           end,

   check = function(arg, idx)
              return string.format("type(arg[%d]) == 'number'", idx)
           end,

   read = function(arg, idx)
             return string.format('arg%d = arg[%d]-1', arg.i, idx)
          end,

   init = function(arg)
             -- otherwise do it here
             if arg.default then
                local default = interpretdefaultvalue(arg)
                if not tonumber(default) then
                   return string.format("arg%d = %s-1", arg.i, default)
                end
             end
          end,

   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,

   precall = function(arg)
             end,

   postcall = function(arg)
              end
}

wrap.argtypes.number = {
   
   helpname = function(arg)
                 return 'number'
              end,

   declare = function(arg)
                -- if it is a number we initialize here
                local default = tonumber(interpretdefaultvalue(arg)) or 0
                return string.format("local arg%d = %d", arg.i, tonumber(default))
             end,

   check = function(arg, idx)
              return string.format("type(arg[%d]) == 'number'", idx)
           end,
   
   read = function(arg, idx)
             return string.format("arg%d = arg[%d]", arg.i, idx)
          end,
   
   init = function(arg)
             -- otherwise do it here
             if arg.default then
                local default = interpretdefaultvalue(arg)
                if not tonumber(default) then
                   return string.format("arg%d = %s", arg.i, default)
                end
             end
          end,
   
   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,
   
   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,
   
   precall = function(arg)
             end,
   
   postcall = function(arg)
              end
}

wrap.argtypes.boolean = {

   helpname = function(arg)
                 return "boolean"
              end,

   declare = function(arg)
                -- if it is a number we initialize here
                local default = tonumber(interpretdefaultvalue(arg)) or 0
                return string.format("local arg%d = %d", arg.i, tonumber(default))
             end,

   check = function(arg, idx)
              return string.format("type(arg[%d]) == 'boolean'", idx)
           end,

   read = function(arg, idx)
             return string.format("arg%d = arg[%d]", arg.i, idx)
          end,

   init = function(arg)
             -- otherwise do it here
             if arg.default then
                local default = interpretdefaultvalue(arg)
                if not tonumber(default) then
                   return string.format("arg%d = %s", arg.i, default)
                end
             end
          end,

   carg = function(arg)
             return string.format('arg%d', arg.i)
          end,

   creturn = function(arg)
                return string.format('arg%d', arg.i)
             end,

   precall = function(arg)
             end,

   postcall = function(arg)
              end
}
