
-- welcome message
print 'Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU'

-- custom prompt
_PROMPT  = 't7> '
_PROMPT2 = '. > '

-- helper
local function sizestr(x)
   local strt = {}
   if _G.torch.typename(x):find('torch.*Storage') then
      return _G.torch.typename(x):match('torch%.(.+)') .. ' - size: ' .. x:size()
   end
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
      tp = torch.typename(val) or ''
      if tp:find('torch.*Tensor') then
         tp = sizestr(val)
      elseif tp:find('torch.*Storage') then
         tp = sizestr(val)
      else
         tp = tostring(val)
      end
   elseif tp == 'table' then
      tp = tp .. ' - size: ' .. #val
   elseif tp == 'string' then
      local tostr = val:gsub('\n','\\n')
      if #tostr>40 then
         tostr = tostr:sub(1,40) .. '...'
      end
      tp = tp .. ' : "' .. tostr .. '"'
   else
      tp = tostring(val)
   end
   return name .. ' = ' .. tp
end

-- helper
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

-- exit:
-- a simple function to exit Torch :-)
function exit()
   os.exit()
end

_G._preloaded_ = {}
for k,v in pairs(_G) do
   _G._preloaded_[k] = true
end

-- a function to colorize output:
local function colorize(object)
   -- Colors:
   local c = {none = '\27[0m',
             black = '\27[0;30m',
             red = '\27[0;31m',
             green = '\27[0;32m',
             yellow = '\27[0;33m',
             blue = '\27[0;34m',
             magenta = '\27[0;35m',
             cyan = '\27[0;36m',
             white = '\27[0;37m',
             Black = '\27[1;30m',
             Red = '\27[1;31m',
             Green = '\27[1;32m',
             Yellow = '\27[1;33m',
             Blue = '\27[1;34m',
             Magenta = '\27[1;35m',
             Cyan = '\27[1;36m',
             White = '\27[1;37m',
             _black = '\27[40m',
             _red = '\27[41m',
             _green = '\27[42m',
             _yellow = '\27[43m',
             _blue = '\27[44m',
             _magenta = '\27[45m',
             _cyan = '\27[46m',
             _white = '\27[47m'}

   -- Apply:
   local apply
   if torch.isatty(io.stdout) then
      apply = function(color, txt)
         return c[color] .. txt .. c.none
      end
   else
      apply = function(color, txt)
         return txt
      end
   end

   -- Type?
   if object == nil then
      return apply('Black', 'nil')
   elseif type(object) == 'number' then
      return apply('cyan', tostring(object))
   elseif type(object) == 'boolean' then
      return apply('blue', tostring(object))
   elseif type(object) == 'string' then
      return apply('yellow', object)
   elseif type(object) == 'function' then
      return apply('magenta', tostring(object))
   elseif type(object) == 'userdata' or type(object) == 'cdata' then
      local tp = torch.typename(object) or ''
      if tp:find('torch.*Tensor') then
         tp = sizestr(object)
      elseif tp:find('torch.*Storage') then
         tp = sizestr(object)
      else
         tp = tostring(object)
      end
      if tp ~= '' then
         return apply('red', tp)
      else
         return apply('red', tostring(object))
      end
   elseif type(object) == 'table' then
      return apply('green', tostring(object))
   else
      return apply('black', tostring(object))
   end
end

-- This is a new recursive, colored print.
local ndepth = 4
local print_old=print
local function print_new(...)
   local function printrecursive(obj,depth)
      local depth = depth or 0
      local tab = depth*4
      local line = function(s) for i=1,tab do io.write(' ') end print_old(s) end
      local mt = getmetatable(obj)
      if mt and mt.__tostring and torch.typename(obj) == nil then
         print_old(tostring(obj))
      else
         if torch.typename(obj) then
            line(tostring(obj):gsub('\n','\n' .. string.rep(' ',tab)))
         end
         line('{')
         tab = tab+2
         for k,v in pairs(obj) do
            if type(v) == 'table' then
               if depth >= (ndepth-1) or next(v) == nil then
                  line(tostring(k) .. ' : ' .. colorize(v))
               else
                  line(tostring(k) .. ' : ') printrecursive(v,depth+1)
               end
            else
               line(tostring(k) .. ' : ' .. colorize(v))
            end
         end
         tab = tab-2
         line('}')
      end
   end
   for i = 1,select('#',...) do
      local obj = select(i,...)
      if type(obj) ~= 'table' then
         if type(obj) == 'userdata' or type(obj) == 'cdata' then
            print_old(obj)
         else
            io.write(colorize(obj) .. '\t')
            if i == select('#',...) then
               print_old()
            end
         end
      else 
         printrecursive(obj) 
      end
   end
   if select('#',...) == 0 then
      print_old()
   end
end

function setprintlevel(n)
  if n == nil or n < 0 then
    error('expected number [0,+)')
  end
  n = math.floor(n)
  ndepth = n
  if ndepth == 0 then
    print = print_old
  else
    print = print_new
  end
end
setprintlevel(5)

-- table():
-- ok, this is slightly out of context, but that function
-- should really exist in Lua. It creates a new table, and then
-- imports all the table methods into it, so that you can do things
-- like:
-- t = table()
-- t:insert(1)
-- t:insert(2)
-- print(t)
-- > {1,2}
local function newtable()
   local t = {}
   for k,v in pairs(table) do
      t[k] = v
   end
   return t
end
setmetatable(table, {__call=newtable})

-- import:
-- this function is a python-like loader, it requires a module,
-- and then imports all its symbols globally
function import(package, forced)
   require(package)
   if _G[package] then
      _G._torchimport = _G._torchimport or {}
      _G._torchimport[package] = _G[package]
   end
   for k,v in pairs(_G[package]) do
      if not _G[k] or forced then
         _G[k] = v
      end
   end
end


-- install module:
-- this function builds and install a specified module
function install(path)
   path = paths.concat(paths.cwd(), path)
   print('--> installing module ' .. path)
   os.execute('mkdir ' .. paths.concat(path,'build') .. '; '
           .. 'cd ' .. paths.concat(path,'build') .. '; '
        .. 'cmake .. -DCMAKE_INSTALL_PREFIX=' .. paths.install_prefix .. '; '
        .. 'make install; cd .. ; rm -r build')
   print('--> module installed')
end

function loaddefaultlibs(loadwithimport)
   if loadwithimport == nil then loadwithimport = false end
   if not loadwithimport then
      -- preload basic libraries
      require 'torch'
      require 'gnuplot'
      require 'dok'
   else
      import 'torch'
      import 'gnuplot'
      import 'dok'
   end
end

loaddefaultlibs(loadwithimport)

-- setup local paths (for LuarRocks and Torch-pkg)
local localinstalldir = paths.concat(os.getenv('HOME'),'.torch','usr')
if paths.dirp(localinstalldir) then
   package.path = paths.concat(localinstalldir,'share','torch','lua','?','init.lua') .. ';' .. package.path
   package.path = paths.concat(localinstalldir,'share','torch','lua','?.lua') .. ';' ..  package.path
   package.cpath = paths.concat(localinstalldir,'lib','torch','lua','?.so') .. ';' .. package.cpath
   package.cpath = paths.concat(localinstalldir,'lib','torch','lua','?.dylib') .. ';' .. package.cpath
   package.cpath = paths.concat(localinstalldir,'lib','torch','?.so') .. ';' .. package.cpath
   package.cpath = paths.concat(localinstalldir,'lib','torch','?.dylib') .. ';' .. package.cpath
end
local localinstalldir = paths.concat(os.getenv('HOME'),'.luarocks')
if paths.dirp(localinstalldir) then
   package.path = paths.concat(localinstalldir,'share','lua','5.1','?','init.lua') .. ';' .. package.path
   package.path = paths.concat(localinstalldir,'share','lua','5.1','?.lua') .. ';' ..  package.path
   package.cpath = paths.concat(localinstalldir,'lib','lua','5.1','?.so') .. ';' .. package.cpath
   package.cpath = paths.concat(localinstalldir,'lib','lua','5.1','?.dylib') .. ';' .. package.cpath
end

