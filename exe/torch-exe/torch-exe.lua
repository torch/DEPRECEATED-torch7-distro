-- use paths lib
require 'paths'

-- help/args
help = 
[=[Torch7 Shell

Usage: torch [options] [script [args]]

General options:
  -e string        execute string
  -l lib           require lib
  -i               enter interactive mode after executing script [false]
  -m|-import       import torch and gnuplot packages into global namespace
  -v|-version      show version information [false]
  -h|-help         this help [false]

Qt options:
  -nographics|-ng  disable all the graphical capabilities [false]
  -ide             enable IDE (graphical console) [false]
  -onethread       run lua in the main thread (might be safer) [false] ]=]

-- default lua: qlua
lua = 'torch-qlua'

-- preload torch environment
env = ' -e "' .. "require 'torch-env'" .. '" '

-- by default, be interactive
interactive = true

-- parse some arguments
for i,a in ipairs(arg) do
   --  no graphics mode?
   if a == '-nographics' or a == '-ng' then
      lua = 'torch-lua'
      arg[i] = ''
   end
   -- help?
   if a == '-help' or a == '--help' or a == '-h' then
      print(help)
      os.exit()
   end
   -- version?
   if a == '-v' or a == '-version' then
      print('Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU')
      os.execute(paths.concat(paths.install_bin,lua) .. ' -v')
      os.exit()
   end
   -- use import
   if a == '-m' or a == '-import' then
      env = ' -e "loadwithimport=true"' .. env
      -- we don't pass this to qlua
      arg[i] = ' '
   end
   -- autostart interactive sessions if no user script:
   if a:find('%.lua$') and paths.filep(a) then
      interactive = false
      -- do not consider further arguments
      break
   end
end

-- interactive?
if interactive then
   env = env .. ' -i '
end

-- re-pack arguments
for i,a in ipairs(arg) do
   if (a:find('[^-=+.%w]')) then
      arg[i] = '"' .. string.gsub(arg[i],'[$`"\\]','\\%0') .. '"'
   end
end
args = table.concat(arg, ' ')

-- test qlua existence
if lua == 'torch-qlua' and not paths.filep(paths.concat(paths.install_bin,lua))
then
   print('Unable to find torch-qlua (disabling graphics)')
   print('Fix this by installing Qt4 and rebuilding Torch7')
   lua = 'torch-lua'
elseif os.getenv('DISPLAY') == '' or os.getenv('DISPLAY') == nil then
   print('Unable to connect X11 server (disabling graphics)')
   lua = 'torch-lua'
end

-- messages
if interactive then
   if lua == 'torch-qlua' then
       print('Try the IDE: torch -ide')
   end
   print('Type help() for more info')
end

-- finally execute main thread, with proper options
os.execute(paths.concat(paths.install_bin,lua) .. env .. args)
