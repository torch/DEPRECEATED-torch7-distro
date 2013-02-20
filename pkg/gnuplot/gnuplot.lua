
require 'paths'

local _gptable = {}
_gptable.current = nil
_gptable.defaultterm = nil
_gptable.exe = nil
_gptable.hasrefresh = true

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
      if s and s:match('Darwin') then
         return 'mac'
      elseif s and s:match('Linux') then
         return 'linux'
      elseif s and s:match('FreeBSD') then
         return 'freebsd'
      else
         --error('I don\'t know your operating system')
         return '?'
      end
   end
end

local function getfigure(n)
   local n = n
   if not n or n == nil then
      n = #_gptable+1
   end
   if _gptable[n] == nil then
      _gptable[n] = {}
      if _gptable.defaultterm == nil then
         error('Gnuplot terminal is not set')
      end
      _gptable[n].term = _gptable.defaultterm
      _gptable[n].pipe = torch.PipeFile(getexec() .. ' -persist > /dev/null 2>&1 ','w')
   end
   _gptable.current = n
   return _gptable[n]
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
   os.execute(getexec() .. ' < ' .. tfni .. ' > ' .. tfno .. ' 2>&1 ')
   os.remove(tfni)
   local tf = io.open(tfno,'r')
   local s = tf:read('*l')
   while s do
      if s:match('^.*%s+  '.. term .. ' ') then
	 tf:close()
	 os.remove(tfno)
         return true
      end
      s = tf:read('*l')
   end
   tf:close()
   os.remove(tfno)
   return false
end

local function findgnuplotversion(exe)
   local ff = io.popen(exe .. '  --version','r')
   local ss = ff:read('*l')
   ff:close()
   local v,vv = ss:match('(%d).(%d)')
   v=tonumber(v)
   vv=tonumber(vv)
   return v,vv
end

local function findgnuplotexe()
   local os = findos()
   if os == 'windows' then
      return 'gnuplot.exe' -- I don't know how to find executables in Windows
   else
      _gptable.hasrefresh = true
      local ff = io.popen('which gnuplot','r')
      local s=ff:read('*l')
      ff:close()
      if s and s:len() > 0 and s:match('gnuplot') then
         local v,vv = findgnuplotversion(s)
         if  v < 4 then
            error('gnuplot version 4 is required')
         end
         if vv < 4 then
            -- try to find gnuplot44
            if os == 'linux' and paths.filep('/usr/bin/gnuplot44') then
               local ss = '/usr/bin/gnuplot44'
               v,vv = findgnuplotversion(ss)
               if v == 4 and vv == 4 then
                  return ss
               end
            end
            _gptable.hasrefresh = false
            print('Gnuplot package working with reduced functionality.')
            print('Please install gnuplot version >= 4.4.')
         end
         return s
      else
         return nil
      end
   end
end

local function getgnuplotdefaultterm(os)
   if os == 'windows' and gnuplothasterm('windows') then
      return  'windows'
   elseif os == 'linux' and gnuplothasterm('wxt') then
      return  'wxt'
   elseif os == 'linux' and gnuplothasterm('x11') then
      return  'x11'
   elseif os == 'freebsd' and gnuplothasterm('wxt') then
      return  'wxt'
   elseif os == 'freebsd' and gnuplothasterm('x11') then
      return  'x11'
   elseif os == 'mac' and gnuplothasterm('aqua') then
      return  'aqua'
   elseif os == 'mac' and gnuplothasterm('x11') then
      return  'x11'
   else
      print('Can not find any of the default terminals for ' .. os .. ' you can manually set terminal by gnuplot.setterm("terminal-name")')
      return nil
   end
end

local function findgnuplot()
   local exe = findgnuplotexe()
   local os = findos()
   if not exe then
      return nil--error('I could not find gnuplot exe')
   end
   _gptable.exe = exe
   _gptable.defaultterm = getgnuplotdefaultterm(os)
end


function gnuplot.setgnuplotexe(exe)
   local oldexe = _gptable.exe

   if not paths.filep(exe) then
      error(exe .. ' does not exist')
   end

   _gptable.exe = exe
   local v,vv = findgnuplotversion(exe)
   if v < 4 then error('gnuplot version 4 is required') end
   if vv < 4 then 
      _gptable.hasrefresh = false
      print('Some functionality like adding title, labels, ... will be disabled, it is better to install gnuplot version 4.4')
   else
      _gptable.hasrefresh = true
   end
   
   local os = findos()
   local term = getgnuplotdefaultterm(os)
   if term == nil then
      print('You have manually set the gnuplot exe and I can not find default terminals, run gnuplot.setterm("terminal-name") to set term type')
   end
end

function gnuplot.setterm(term)
   if gnuplothasterm(term) then
      _gptable.defaultterm = term
   else
      error('gnuplot does not seem to have this term')
   end
end

local function getCurrentPlot()
   if _gptable.current == nil then
      gnuplot.figure()
   end
   return _gptable[_gptable.current]
end

local function writeToPlot(gp,str)
   local pipe = gp.pipe
   pipe:writeString(str .. '\n\n\n')
   pipe:synchronize()
end
local function refreshPlot(gp)
   if gp.fname then
      writeToPlot(gp,'set output "' .. gp.fname .. '"')
   end
   writeToPlot(gp,'refresh')
   if gp.fname then
      writeToPlot(gp,'unset output')
   end
end
local function writeToCurrent(str)
   writeToPlot(getCurrentPlot(),str)
end
local function refreshCurrent()
   refreshPlot(getCurrentPlot())
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
      if x:dim() == 2 and x:size(2) == 2 then
         y = x:select(2,2)
         x = x:select(2,1)
      elseif x:dim() == 2 and x:size(2) == 4 then
         y = torch.Tensor(x:size(1),2)
         xx= torch.Tensor(x:size(1),2)
         y:select(2,1):copy(x:select(2,2))
         y:select(2,2):copy(x:select(2,4))
         xx:select(2,1):copy(x:select(2,1))
         xx:select(2,2):copy(x:select(2,3))
         x = xx
      else
         y = x
         x = torch.range(1,y:size(1))
      end
   end
   if x:dim() ~= y:dim() or x:nDimension() > 2 or y:nDimension() > 2 then
      error('x and y dims are wrong :  x = ' .. x:nDimension() .. 'D y = ' .. y:nDimension() .. 'D')
   end
   --print(x:size(),y:size())
   return legend,x,y,format
end

-- t is the arguments for one plot at a time
local function getsplotvars(t)
   local legend = nil
   local x = nil
   local y = nil
   local z = nil

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
         error('expecting [string,] tensor [,tensor] [,tensor]')
      end
   end
   if #t >= 2 and #t <= 4 then
      if x and istensor(t[2]) and istensor(t[3]) then
         y = t[2]
         z = t[3]
      elseif legend and istensor(t[2]) and istensor(t[3]) and istensor(t[4]) then
         x = t[2]
         y = t[3]
         z = t[4]
      elseif legend and istensor(t[2]) then
         x = t[2]
      else
         error('expecting [string,] tensor [,tensor] [,tensor]')
      end
   elseif #t > 4 then
      error('expecting [string,] tensor [,tensor] [,tensor]')
   end
   legend = legend or ''
   if not x then
      error('expecting [string,] tensor [,tensor] [,tensor]')
   end
   if not z then
      z = x
      x = torch.Tensor(z:size())
      y = torch.Tensor(z:size())
      for i=1,x:size(1) do x:select(1,i):fill(i) end
      for i=1,y:size(2) do y:select(2,i):fill(i) end
   end
   if x:nDimension() ~= 2 or y:nDimension() ~= 2 or z:nDimension() ~= 2 then
      error('x and y and z are expected to be matrices x = ' .. x:nDimension() .. 'D y = ' .. y:nDimension() .. 'D z = '.. z:nDimension() .. 'D' )
   end
   return legend,x,y,z
end

local function getimagescvars(t)
   local palette  = nil
   local x = nil

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
      if istensor(t[1]) then
         x = t[1]
      else
         error('expecting tensor [,string]')
      end
   end
   if #t == 2 then
      if x and isstring(t[2]) then
         palette = t[2]
      else
         error('expecting tensor [,string]' )
      end
   elseif #t > 2 then
      error('expecting tensor [,string]')
   end
   legend = legend or ''
   if not x then
      error('expecting tensor [,string]')
   end
   if not palette then
      palette = 'gray'
   end
   if x:nDimension() ~= 2 then
      error('x is expected to be matrices x = ' .. x:nDimension() .. 'D')
   end
   return x,palette
end

local function gnuplot_string(legend,x,y,format)
   local hstr = 'plot '
   local dstr = {''}
   local coef = {}
   local vecplot = {}
   local function gformat(f,i)
      if f ~= '~' and f:find('~') or f:find('acsplines') then
         coef[i] = f:gsub('~',''):gsub('acsplines','')
         coef[i] = tonumber(coef[i])
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
      elseif f == 'V' or f == 'v' or f == 'vectors' then vecplot[i]=true;return 'with vectors'
      end
      error("format string accepted: '.' or '-' or '+' or '+-' or '~' or '~ COEF'")
   end
   for i=1,#legend do
      if i > 1 then hstr = hstr .. ' , ' end
      hstr = hstr .. " '-' title '" .. legend[i] .. "' " .. gformat(format[i],i)
   end
   hstr = hstr .. '\n'
   for i=1,#legend do
      local xi = x[i]
      local yi = y[i]
      for j=1,xi:size(1) do
         if coef[i] then
            --print(i,coef)
            table.insert(dstr,string.format('%g %g %g\n',xi[j],yi[j],coef[i]))
         elseif vecplot[i] then
            --print(xi,yi)
            table.insert(dstr,string.format('%g %g %g %g\n',xi[j][1],yi[j][1],xi[j][2],yi[j][2]))
         else
            table.insert(dstr,string.format('%g %g\n',xi[j],yi[j]))
         end
      end
      table.insert(dstr,'e\n')
   end
   return hstr,table.concat(dstr)
end
local function gnu_splot_string(legend,x,y,z)
   local hstr = string.format('%s\n','set contour base')
   hstr = string.format('%s%s\n',hstr,'set cntrparam bspline\n')
   hstr = string.format('%s%s\n',hstr,'set cntrparam levels auto\n')
   hstr = string.format('%s%s\n',hstr,'set style data lines\n')
   hstr = string.format('%s%s\n',hstr,'set hidden3d\n')

   hstr = hstr .. 'splot '
   local dstr = {''}
   local coef
   for i=1,#legend do
      if i > 1 then hstr = hstr .. ' , ' end
      hstr = hstr .. " '-'title '" .. legend[i] .. "' " .. 'with lines'
   end
   hstr = hstr .. '\n'
   for i=1,#legend do
      local xi = x[i]
      local yi = y[i]
      local zi = z[i]
      for j=1,xi:size(1) do
         local xij = xi[j]
         local yij = yi[j]
         local zij = zi[j]
         for k=1,xi:size(2) do
            table.insert(dstr, string.format('%g %g %g\n',xij[k],yij[k],zij[k]))
         end
         table.insert(dstr,'\n')
      end
      table.insert(dstr,'e\n')
   end
   return hstr,table.concat(dstr)
end

local function gnu_imagesc_string(x,palette)
   local hstr = string.format('%s\n','set view map')
   hstr = string.format('%s%s %s\n',hstr,'set palette',palette)
   hstr = string.format('%s%s\n',hstr,'set style data linespoints')
   hstr = string.format('%s%s%g%s\n',hstr,"set xrange [ -0.5 : ",x:size(2)-0.5,"] noreverse nowriteback")
   hstr = string.format('%s%s%g%s\n',hstr,"set yrange [ -0.5 : ",x:size(1)-0.5,"] reverse nowriteback")
   hstr = string.format('%s%s\n',hstr,"splot '-' matrix with image")
   local dstr = {''}
   for i=1,x:size(1) do
      local xi = x[i];
      for j=1,x:size(2) do
         table.insert(dstr,string.format('%g ',xi[j]))
      end
      table.insert(dstr, string.format('\n'))
   end
   table.insert(dstr,string.format('e\ne\n'))
   return hstr,table.concat(dstr)
end

function gnuplot.close(n)
   if not n then
      n = _gptable.current
   end
   local gp = _gptable[n]
   if gp == nil then return end
   if type(n) ==  'number' and torch.typename(gp.pipe) == 'torch.PipeFile' then
      _gptable.current = nil
      gnuplot.plotflush(n)
      writeToPlot(gp, 'quit')
      
      -- pipefile:close is buggy in TH
      --gp.pipe:close()
      gp.pipe=nil
      gp = nil
      _gptable[n] = nil
   end
   collectgarbage()
end

function gnuplot.closeall()
   for i,v in pairs(_gptable) do
      gnuplot.close(i)
   end
   _gptable = {}
   collectgarbage()
   findgnuplot()
   _gptable.current = nil
end

local function filefigure(fname,term,n)
   if not _gptable.hasrefresh then
      print('Plotting to files is disabled in gnuplot 4.2, install gnuplot 4.4')
   end
   local gp = getfigure(n)
   gp.fname = fname
   gp.term = term
   writeToCurrent('set term '.. gp.term)
   --writeToCurrent('set output \'' .. gp.fname .. '\'')
end
function gnuplot.epsfigure(fname,n)
   filefigure(fname,'postscript eps enhanced color',n)
   return _gptable.current
end

function gnuplot.svgfigure(fname,n)
   filefigure(fname,'svg',n)
   return _gptable.current
end

function gnuplot.pngfigure(fname,n)
   filefigure(fname,'png',n)
   return _gptable.current
end

function gnuplot.figprint(fname)
   local suffix = fname:match('.+%.(.+)')
   local term = nil
   if suffix == 'eps' then
      term = 'postscript eps enhanced color'
   elseif suffix == 'png' then
      term = 'png'
   else
      error('only eps and png for figprint')
   end
   writeToCurrent('set term ' .. term)
   writeToCurrent('set output \''.. fname .. '\'')
   refreshCurrent()
   writeToCurrent('unset output')
   writeToCurrent('set term ' .. _gptable[_gptable.current].term .. ' ' .. _gptable.current .. '\n')
end

function gnuplot.figure(n)
   local gp = getfigure(n)
   writeToCurrent('set term ' .. _gptable[_gptable.current].term .. ' ' .. _gptable.current .. '\n')
   writeToCurrent('raise')
   return _gptable.current
end

function gnuplot.plotflush(n)
   if not n then
      n = _gptable.current
   end
   if not n or _gptable[n] == nil then
      print('no figure ' ..  tostring(n))
      return
   end
   local gp = _gptable[n]
   --xprint(gp)
   refreshPlot(gp)
   -- if gp.fname then
   --    writeToPlot(gp,'set output "' .. gp.fname .. '"')
   --    writeToPlot(gp,'refresh')
   --    writeToPlot(gp,'unset output')
   -- end
end

local function gnulplot(legend,x,y,format)
   local hdr,data = gnuplot_string(legend,x,y,format)
   writeToCurrent(hdr)
   writeToCurrent(data)
end
local function gnusplot(legend,x,y,z)
   local hdr,data = gnu_splot_string(legend,x,y,z)
   writeToCurrent(hdr)
   writeToCurrent(data)
end
local function gnuimagesc(x,palette)
   local hdr,data = gnu_imagesc_string(x,palette)
   writeToCurrent(hdr)
   writeToCurrent(data)
end

function gnuplot.xlabel(label)
   if not _gptable.hasrefresh then
      print('gnuplot.xlabel disabled')
      return
   end
   writeToCurrent('set xlabel "' .. label .. '"')
   refreshCurrent()
end
function gnuplot.ylabel(label)
   if not _gptable.hasrefresh then
      print('gnuplot.ylabel disabled')
      return
   end
   writeToCurrent('set ylabel "' .. label .. '"')
   refreshCurrent()
end
function gnuplot.zlabel(label)
   if not _gptable.hasrefresh then
      print('gnuplot.zlabel disabled')
      return
   end
   writeToCurrent('set zlabel "' .. label .. '"')
   refreshCurrent()
end
function gnuplot.title(label)
   if not _gptable.hasrefresh then
      print('gnuplot.title disabled')
      return
   end
   writeToCurrent('set title "' .. label .. '"')
   refreshCurrent()
end
function gnuplot.grid(toggle)
   if not _gptable.hasrefresh then
      print('gnuplot.grid disabled')
      return
   end
   if toggle then
      writeToCurrent('set grid')
      refreshCurrent()
   else
      writeToCurrent('unset grid')
      refreshCurrent()
   end
end
function gnuplot.movelegend(hloc,vloc)
   if not _gptable.hasrefresh then
      print('gnuplot.movelegend disabled')
      return
   end
   if hloc ~= 'left' and hloc ~= 'right' and hloc ~= 'center' then
      error('horizontal location is unknown : plot.movelegend expects 2 strings as location {left|right|center}{bottom|top|middle}')
   end
   if vloc ~= 'bottom' and vloc ~= 'top' and vloc ~= 'middle' then
      error('horizontal location is unknown : plot.movelegend expects 2 strings as location {left|right|center}{bottom|top|middle}')
   end
   writeToCurrent('set key ' .. hloc .. ' ' .. vloc)
   refreshCurrent()
end

function gnuplot.axis(axis)
   if not _gptable.hasrefresh then
      print('gnuplot.axis disabled')
      return
   end
   if axis == 'auto' then
      writeToCurrent('set size nosquare')
      writeToCurrent('set autoscale')
      refreshCurrent()
   elseif axis == 'image' or axis == 'equal' then
      writeToCurrent('set size ratio -1')
      refreshCurrent()
   elseif axis == 'fill' then
      writeToCurrent('set size ratio 1,1')
      refreshCurrent()
   elseif type(axis) == 'table' then
      if #axis ~= 4 then print('axis should have 4 componets {xmin,xmax,ymin,ymax}'); return end
      writeToCurrent('set xrange [' .. axis[1] .. ':' .. axis[2] .. ']')
      writeToCurrent('set yrange [' .. axis[3] .. ':' .. axis[4] .. ']')
      refreshCurrent()
   end
end

function gnuplot.raw(str)
   writeToCurrent(str)
end

-- plot(x)
-- plot(x,'.'), plot(x,'.-')
-- plot(x,y,'.'), plot(x,y,'.-')
-- plot({x1,y1,'.'},{x2,y2,'.-'})
-- plot({{x1,y1,'.'},{x2,y2,'.-'}})
function gnuplot.plot(...)
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
   gnulplot(legends,xdata,ydata,formats)
end

-- splot(z)
-- splot({x1,y1,z1},{x2,y2,z2})
-- splot({'name1',x1,y1,z1},{'name2',x2,y2,z2})
function gnuplot.splot(...)
   local arg = {...}
   if select('#',...) == 0 then
      error('no inputs, expecting at least a matrix')
   end

   local xdata = {}
   local ydata = {}
   local zdata = {}
   local legends = {}

   if type(arg[1]) == "table" then
      if type(arg[1][1]) == "table" then
         arg = arg[1]
      end
      for i,v in ipairs(arg) do
         local l,x,y,z = getsplotvars(v)
         legends[#legends+1] = l
         xdata[#xdata+1] = x
         ydata[#ydata+1] = y
         zdata[#zdata+1] = z
      end
   else
      local l,x,y,z = getsplotvars(arg)
      legends[#legends+1] = l
      xdata[#xdata+1] = x
      ydata[#ydata+1] = y
      zdata[#zdata+1] = z
   end
   gnusplot(legends,xdata,ydata,zdata)
end

-- imagesc(x) -- x 2D tensor [0 .. 1]
function gnuplot.imagesc(...)
   local arg = {...}
   if select('#',...) == 0 then
      error('no inputs, expecting at least a matrix')
   end
   gnuimagesc(getimagescvars(arg))
end

-- bar(y)
-- bar(x,y)
function gnuplot.bar(...)
   local arg = {...}
   local nargs = {}
   for i = 1,select('#',...) do
      table.insert(nargs,arg[i])
   end
   table.insert(nargs, '|')
   gnuplot.plot(nargs)
end

-- complete function: compute hist and display it
function gnuplot.hist(tensor,bins,min,max)
   local h = gnuplot.histc(tensor,bins,min,max)
   local x_axis = torch.Tensor(#h)
   for i = 1,#h do
      x_axis[i] = h[i].val
   end
   gnuplot.bar(x_axis, h.raw)
   return h
end


findgnuplot()
