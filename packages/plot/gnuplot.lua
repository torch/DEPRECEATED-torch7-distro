
require 'paths'

local _gptable = {}
_gptable.current = nil
_gptable.term = nil
_gptable.exe = nil

local function getexec()
   if not _gptable.exe then
      error('gnuplot executable is not set')
   end
   return _gptable.exe
end

local function findos()
   if paths.dirp('C:\\') then
      return 'windows'
   else
      local ff = io.popen('uname -a','r')
      local s = ff:read('*all')
      ff:close()
      if s:match('Darwin') then
	 return 'mac'
      elseif s:match('Linux') then
	 return 'linux'
      else
	 --error('I don\'t know your operating system')
	 return '?'
      end
   end
end

local function gnuplothasterm(term)
   if not _gptable.exe then
      return false--error('gnuplot exe is not found, can not chcek terminal')
   end
   local tfni = os.tmpname()
   local tfno = os.tmpname()
   local fi = io.open(tfni,'w')
   fi:write('set terminal\n\n')
   fi:close()
   os.execute(getexec() .. ' < ' .. tfni .. ' 2> ' .. tfno)
   local tf = io.open(tfno,'r')
   local s = tf:read('*l')
   while s do
      if s:match('^.*%s+  '.. term .. ' ') then
	 return true
      end
      s = tf:read('*l')
   end
   return false
end

local function findgnuplotexe()
   local os = findos()
   if os == 'windows' then
      return 'gnuplot.exe' -- I don't know how to find executables in Windows
   else
      local ff = io.popen('which gnuplot','r')
      local s=ff:read('*l')
      ff:close()
      if s and s:len() > 0 and s:match('gnuplot') then
	 return s
      else
	 return nil
      end
   end
end

local function findgnuplot()
   local exe = findgnuplotexe()
   local os = findos()
   if not exe then
      return nil--error('I could not find gnuplot exe')
   end
   _gptable.exe = exe

   if os == 'windows' and gnuplothasterm('windows') then
      _gptable.term = 'windows'
   elseif os == 'linux' and gnuplothasterm('wxt') then
      _gptable.term = 'wxt'
   elseif os == 'linux' and gnuplothasterm('x11') then
      _gptable.term = 'x11'
   elseif os == 'mac' and gnuplothasterm('aqua') then
      _gptable.term = 'aqua'
   elseif os == 'mac' and gnuplothasterm('x11') then
      _gptable.term = 'x11'
   else
      return nil--error('can not find terminal')
   end
end

function plot.setgnuplotexe(exe)
   if paths.filep(exe) then
      _gptable.exe = exe
      print('You have manually set the gnuplot exe, run plot.setgnuplotterminal("terminal-name") to set term type')
   else
      error(exe .. ' does not exist')
   end
end

function plot.setgnuplotterminal(term)
   if gnuplothasterm(term) then
      _gptable.term = term
   else
      error('gnuplot does not seem to have this term')
   end
end

local function getCurrentPlot()
   if _gptable.current == nil then
      plot.figure()
   end
   return _gptable[_gptable.current]
end

local function writeToCurrent(str)
   local _gp = getCurrentPlot()
   _gp:writeString(str .. '\n')
   _gp:synchronize()
end

-- t is the arguments for one plot at a time
local function getvars(t)
   local legend = nil
   local x = nil
   local y = nil
   local format = nil

   local function istensor(v)
      return type(v) == 'userdata' and torch.typename(v):sub(-6) == 'Tensor'
   end

   local function isstring(v)
      return type(v) == 'string'
   end

   if #t == 0 then
      error('empty argument list')
   end

   if #t >= 1 then
      if isstring(t[1]) then
	 legend = t[1]
      elseif istensor(t[1]) then
	 x = t[1]
      else
	 error('expecting [string,] tensor [,tensor] [,string]')
      end
   end
   if #t >= 2 then
      if x and isstring(t[2]) then
	 format = t[2]
      elseif x and istensor(t[2]) then
	 y = t[2]
      elseif legend and istensor(t[2]) then
	 x = t[2]
      else
	 error('expecting [string,] tensor [,tensor] [,string]')
      end
   end
   if #t >= 3 then
      if legend and x and istensor(t[3]) then
	 y = t[3]
      elseif legend and x and isstring(t[3]) then
	 format = t[3]
      elseif x and y and isstring(t[3]) then
	 format = t[3]
      else
	 error('expecting [string,] tensor [,tensor] [,string]')
      end
   end
   if #t == 4 then
      if legend and x and y and isstring(t[4]) then
	 format = t[4]
      else
	 error('expecting [string,] tensor [,tensor] [,string]')
      end
   end
   legend = legend or ''
   format = format or ''
   if not x then
      error('expecting [string,] tensor [,tensor] [,string]')
   end
   if not y then
      y = x
      x = lab.range(1,y:size(1))
   end
   if x:nDimension() ~= 1 or y:nDimension() ~= 1 then
      error('x and y are expected to be vectors x = ' .. x:nDimension() .. 'D y = ' .. y:nDimension() .. 'D')
   end
   return legend,x,y,format
end

local function gnuplot_string(legend,x,y,format)
   local hstr = 'plot '
   local dstr = ''
   local coef
   local function gformat(f)
      if f ~= '~' and f:find('~') or f:find('acsplines') then
         coef = f:gsub('~',''):gsub('acsplines','')
         coef = tonumber(coef)
         f = 'acsplines'
      end
      if f == ''  or f == '' then return ''
      elseif f == '+'  or f == 'points' then return 'with points'
      elseif f == '.' or f == 'dots' then return 'with dots'
      elseif f == '-' or f == 'lines' then return 'with lines'
      elseif f == '+-' or f == 'linespoints' then return 'with linespoints' 
      elseif f == '|' or f == 'boxes' then return 'with boxes'
      elseif f == '~' or f == 'csplines' then return 'smooth csplines'
      elseif f == 'acsplines' then return 'smooth acsplines'
      end
      error("format string accepted: '.' or '-' or '+' or '+-' or '~' or '~ COEF'")
   end
   for i=1,#legend do
      if i > 1 then hstr = hstr .. ' , ' end
      hstr = hstr .. " '-' title '" .. legend[i] .. "' " .. gformat(format[i])
   end
   hstr = hstr .. '\n'
   for i=1,#legend do
      for j=1,x[i]:size(1) do
         if coef then
            dstr = dstr .. string.format('%g %g %g\n',x[i][j],y[i][j],coef)
         else
            dstr = dstr .. string.format('%g %g\n',x[i][j],y[i][j])
         end
      end
      dstr = string.format('%se\n',dstr)
   end
   return hstr,dstr
end

function plot.closeall()
   for i,v in pairs(_gptable) do
      if type(i) ==  number and torch.typename(v) == 'torch.PipeFile' then
	 v:close()
	 v=nil
      end
   end
   _gptable = {}
   findgnuplot()
   _gptable.current = nil
end

function plot.epsfigure(fname)
   local n = #_gptable+1
   _gptable[n] = torch.PipeFile(getexec() .. ' -persist ','w')
   _gptable.current = n
   writeToCurrent('set term postscript eps enhanced color')
   writeToCurrent('set output \''.. fname .. '\'')
end

function plot.pngfigure(fname)
   local n = #_gptable+1
   _gptable[n] = torch.PipeFile(getexec() .. ' -persist ','w')
   _gptable.current = n
   writeToCurrent('set term png')
   writeToCurrent('set output \''.. fname .. '\'')
end

function plot.print(fname)
   local suffix = fname:match('.+%.(.+)')
   local term = nil
   if suffix == 'eps' then term = 'postscript eps enhanced color'
   elseif suffix == 'png' then term = 'png'
   else error('only eps and png for print')
   end
   writeToCurrent('set term ' .. term)
   writeToCurrent('set output \''.. fname .. '\'')
   writeToCurrent('refresh')
   writeToCurrent('set term ' .. _gptable.term .. ' ' .. _gptable.current .. '\n')   
end

function plot.figure(n)
   local nfigures = #_gptable
   if not n or _gptable[n] == nil then -- we want new figure
      n = n or #_gptable+1
      _gptable[n] = torch.PipeFile(getexec() .. ' -persist ','w')
   end
   _gptable.current = n
   writeToCurrent('set term ' .. _gptable.term .. ' ' .. n .. '\n')
   return n
end

function plot.plotflush(n)
   if not n then n = _gptable.current end
   if not n then return end
   if _gptable[n] == nil then return end
   local _gp = _gptable[n]
   _gp:writeString('unset output\n')
   _gp:synchronize()
end

function plot.gnuplot(legend,x,y,format)
   local hdr,data = gnuplot_string(legend,x,y,format)
   --writeToCurrent('set pointsize 2')
   writeToCurrent(hdr)
   writeToCurrent(data)
end

function plot.xlabel(label)
   local _gp = getCurrentPlot()
   writeToCurrent('set xlabel "' .. label .. '"')
   writeToCurrent('refresh')
end
function plot.ylabel(label)
   local _gp = getCurrentPlot()
   writeToCurrent('set ylabel "' .. label .. '"')
   writeToCurrent('refresh')
end
function plot.title(label)
   local _gp = getCurrentPlot()
   writeToCurrent('set title "' .. label .. '"')
   writeToCurrent('refresh')
end
function plot.grid(toggle)
   if not toggle then
      print('toggle expects 1 for grid on, 0 for grid off')
   end
   local _gp = getCurrentPlot()
   if toggle == 1 then
      writeToCurrent('set grid')
      writeToCurrent('refresh')
   elseif toggle == 0 then
      writeToCurrent('unset grid')
      writeToCurrent('refresh')
   else
      print('toggle expects 1 for grid on, 0 for grid off')
   end
end
function plot.movelegend(hloc,vloc)
   if hloc ~= 'left' and hloc ~= 'right' and hloc ~= 'center' then
      error('horizontal location is unknown : plot.movelegend expects 2 strings as location {left|right|center}{bottom|top|middle}')
   end
   if vloc ~= 'bottom' and vloc ~= 'top' and vloc ~= 'middle' then
      error('horizontal location is unknown : plot.movelegend expects 2 strings as location {left|right|center}{bottom|top|middle}')
   end
   writeToCurrent('set key ' .. hloc .. ' ' .. vloc)
   writeToCurrent('refresh')
end
function plot.gnuplotraw(str)
   writeToCurrent(str)
end

-- plot(x)
-- plot(x,'.'), plot(x,'.-')
-- plot(x,y,'.'), plot(x,y,'.-')
-- plot({x1,y1,'.'},{x2,y2,'.-'})
-- plot({{x1,y1,'.'},{x2,y2,'.-'}})
function plot.plot(...)
   local arg = {...}
   if select('#',...) == 0 then
      error('no inputs, expecting at least a vector')
   end

   local formats = {}
   local xdata = {}
   local ydata = {}
   local legends = {}

   if type(arg[1]) == "table" then
      if type(arg[1][1]) == "table" then
         arg = arg[1]
      end
      for i,v in ipairs(arg) do
         local l,x,y,f = getvars(v)
         legends[#legends+1] = l
         formats[#formats+1] = f
         xdata[#xdata+1] = x
         ydata[#ydata+1] = y
      end
   else
      local l,x,y,f = getvars(arg)
      legends[#legends+1] = l
      formats[#formats+1] = f
      xdata[#xdata+1] = x
      ydata[#ydata+1] = y
   end
   plot.gnuplot(legends,xdata,ydata,formats)
end

-- bar(y)
-- bar(x,y)
function plot.bar(...)
   local arg = {...}
   local nargs = {}
   for i = 1,select('#',...) do
      table.insert(nargs,arg[i])
   end
   table.insert(nargs, '|')
   plot.plot(nargs)
end

-- complete function: compute hist and display it
function plot.hist(tensor,bins,min,max)
   local h = lab.histc(tensor,bins,min,max)
   local x_axis = torch.Tensor(#h)
   for i = 1,#h do
      x_axis[i] = h[i].val
   end
   plot.bar(x_axis, h.raw)
   return h
end


findgnuplot()
