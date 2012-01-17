
local function sizestr(x)
   local strt = {}
   if x:nDimension() == 0 then
      table.insert(strt, _G.torch.typename(x):match('torch%.(.+)') .. ' - empty')
   else
      table.insert(strt, _G.torch.typename(x):match('torch%.(.+)') .. ' - size: ')
      for i=1,x:nDimension() do
         table.insert(strt, x:size(i))
         if i ~= x:nDimension() then
            table.insert(strt, 'x')
         end
      end
   end
   return table.concat(strt)
end

-- k : name of variable
-- m : max length
local function printvar(key,val,m)
   local name = '[' .. tostring(key) .. ']'
   --io.write(name)
   name = name .. string.rep(' ',m-name:len()+2)
   local tp = type(val)
   if tp == 'userdata' then
      tp = torch.typename(val)
      if tp:find('torch.*Tensor') then
	 tp = sizestr(val)
      end
      if tp:find('torch.*Storage') then
	 tp = sizestr(val)
      end
   elseif tp == 'table' then
      tp = tp .. ' - size: ' .. #val
   elseif tp == 'string' then
      local tostr = val:gsub('\n','\\n')
      if #tostr>40 then
	 tostr = tostr:sub(1,40)
      end
      tp = tp .. ' : "' .. tostr .. '"'
   else
      tp = tostring(val)
   end
   return name .. ' = ' .. tp
end

local function getmaxlen(vars)
   local m = 0
   if type(vars) ~= 'table' then return tostring(vars):len() end
   for k,v in pairs(vars) do
      local s = tostring(k)
      if s:len() > m then
         m = s:len()
      end
   end
   return m
end

-- who:
-- a simple function that prints all the symbols defined by the user
-- very much like Matlab's who function
function who()
   local m = getmaxlen(_G)
   local p = _G._preloaded_
   local function printsymb(sys)
      for k,v in pairs(_G) do
         if (sys and p[k]) or (not sys and not p[k]) then
	    print(printvar(k,_G[k],m))
         end
      end
   end
   print('== System Variables ==')
   printsymb(true)
   print('== User Variables ==')
   printsymb(false)
   print('==')
end

print_old=print
_G._preloaded_ = {}
for k,v in pairs(_G) do
   _G._preloaded_[k] = true
end

-- print:
-- a smarter print for Lua, the default Lua print is quite terse
-- this new print is much more verbose, automatically recursing through
-- lua tables, and objects.
function print(obj,...)
   local m = getmaxlen(obj)
   if _G.type(obj) == 'table' then
      local mt = _G.getmetatable(obj)
      if mt and mt.__tostring__ then
         _G.io.write(mt.__tostring__(obj))
      else
         local tos = _G.tostring(obj)
         local obj_w_usage = false
         if tos and not _G.string.find(tos,'table: ') then
            if obj.usage and _G.type(obj.usage) == 'string' then
               _G.io.write(obj.usage)
               _G.io.write('\n\nFIELDS:\n')
               obj_w_usage = true
            else
               _G.io.write(tos .. ':\n')
            end
         end
         _G.io.write('{')
         local idx = 1
	 local tab = ''
	 local newline = ''
         for k,v in pairs(obj) do
	    local line = printvar(k,v,m)
	    _G.io.write(newline .. tab .. line)
	    if idx == 1 then
	       tab = ' '
	       newline = '\n'
	    end
	    idx = idx + 1
         end
         _G.io.write('}')
         if obj_w_usage then
            _G.io.write('')
         end
      end
   else
      _G.io.write(_G.tostring(obj))
   end
   if _G.select('#',...) > 0 then
      _G.io.write('    ')
      print(...)
   else
      _G.io.write('\n')
   end
end
