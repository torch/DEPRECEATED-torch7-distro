
require 'paths'

local torchstyle = [[
##### Modified version of the sample given in
##### http://www.guidolin.net/blog/files/2010/03/gnuplot


set macro

#####  Color Palette by Color Scheme Designer
#####  Palette URL: http://colorschemedesigner.com/#3K40zsOsOK-K-


   blue_000 = "#A9BDE6" # = rgb(169,189,230)
   blue_025 = "#7297E6" # = rgb(114,151,230)
   blue_050 = "#1D4599" # = rgb(29,69,153)
   blue_075 = "#2F3F60" # = rgb(47,63,96)
   blue_100 = "#031A49" # = rgb(3,26,73)

   green_000 = "#A6EBB5" # = rgb(166,235,181)
   green_025 = "#67EB84" # = rgb(103,235,132)
   green_050 = "#11AD34" # = rgb(17,173,52)
   green_075 = "#2F6C3D" # = rgb(47,108,61)
   green_100 = "#025214" # = rgb(2,82,20)

   red_000 = "#F9B7B0" # = rgb(249,183,176)
   red_025 = "#F97A6D" # = rgb(249,122,109)
   red_050 = "#E62B17" # = rgb(230,43,23)
   red_075 = "#8F463F" # = rgb(143,70,63)
   red_100 = "#6D0D03" # = rgb(109,13,3)

   brown_000 = "#F9E0B0" # = rgb(249,224,176)
   brown_025 = "#F9C96D" # = rgb(249,201,109)
   brown_050 = "#E69F17" # = rgb(230,159,23)
   brown_075 = "#8F743F" # = rgb(143,116,63)
   brown_100 = "#6D4903" # = rgb(109,73,3)

   grid_color = "#d5e0c9"
   text_color = "#222222"

   my_font = "SVBasic Manual, 12"
   my_export_sz = "1024,768"

   my_line_width = "2"
   my_axis_width = "1"
   my_ps = "1"
   my_font_size = "14"

# must convert font fo svg and ps
# set term svg  size @my_export_sz fname my_font fsize my_font_size enhanced dynamic rounded
# set term png  size @my_export_sz large font my_font
# set term jpeg size @my_export_sz large font my_font
# set term wxt enhanced font my_font

set style data linespoints
set style function lines
set pointsize my_ps

set style line 1  linecolor rgbcolor blue_050  linewidth @my_line_width pt 7
set style line 2  linecolor rgbcolor green_050 linewidth @my_line_width pt 7
set style line 3  linecolor rgbcolor red_050   linewidth @my_line_width pt 7
set style line 4  linecolor rgbcolor brown_050 linewidth @my_line_width pt 7
set style line 5  linecolor rgbcolor blue_025  linewidth @my_line_width pt 5
set style line 6  linecolor rgbcolor green_025 linewidth @my_line_width pt 5
set style line 7  linecolor rgbcolor red_025   linewidth @my_line_width pt 5
set style line 8  linecolor rgbcolor brown_025 linewidth @my_line_width pt 5
set style line 9  linecolor rgbcolor blue_075  linewidth @my_line_width pt 9
set style line 10 linecolor rgbcolor green_075 linewidth @my_line_width pt 9
set style line 11 linecolor rgbcolor red_075   linewidth @my_line_width pt 9
set style line 12 linecolor rgbcolor brown_075 linewidth @my_line_width pt 9
set style line 13 linecolor rgbcolor blue_100  linewidth @my_line_width pt 13
set style line 14 linecolor rgbcolor green_100 linewidth @my_line_width pt 13
set style line 15 linecolor rgbcolor red_100   linewidth @my_line_width pt 13
set style line 16 linecolor rgbcolor brown_100 linewidth @my_line_width pt 13
set style line 17 linecolor rgbcolor "#224499" linewidth @my_line_width pt 11

## plot 1,2,3,4,5,6,7,8,9
set style increment user
set style arrow 1 filled

## used for bar chart borders
## set style fill solid 0.5

set size noratio
set samples 300

set border 31 lw @my_axis_width lc rgb text_color
]]

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
   local s = paths.uname()
   if s and s:match("Windows") then
      return 'windows'
   elseif s and s:match('Darwin') then
      return 'mac'
   elseif s and s:match('Linux') then
      return 'linux'
   elseif s and s:match('FreeBSD') then
      return 'freebsd'
   else
      return '?'
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
      local silent = '> /dev/null 2>&1'
      if paths.is_win() then
         silent = '> nul 2>&1'
      end
      _gptable[n].term = _gptable.defaultterm
      _gptable[n].pipe = torch.PipeFile(getexec() .. ' -persist ' .. silent,'w')
   end
   _gptable.current = n
   if not paths.filep(paths.concat(paths.home,'.gnuplot')) then
      _gptable[n].pipe:writeString(torchstyle .. '\n\n\n')
      _gptable[n].pipe:synchronize()
   end
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
   local o = findos()
   local s = paths.findprogram('gnuplot')
   if type(s) == 'string' and s:match(' ') then
      s = '"' .. s .. '"'
   end
   _gptable.hasrefresh = true
   do -- preserve indentation to minimize merging issues
      if s and s:len() > 0 and s:match('gnuplot') then
         local v,vv = findgnuplotversion(s)
         if  v < 4 then
            error('gnuplot version 4 is required')
         end
         if vv < 4 then
            -- try to find gnuplot44
            if o == 'linux' and paths.filep('/usr/bin/gnuplot44') then
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
   if os == 'windows' then
      return  'windows'
   elseif os == 'linux' and gnuplothasterm('wxt') then
      return  'wxt'
   elseif os == 'linux' and gnuplothasterm('qt') then
      return  'qt'
   elseif os == 'linux' and gnuplothasterm('x11') then
      return  'x11'
   elseif os == 'freebsd' and gnuplothasterm('wxt') then
      return  'wxt'
   elseif os == 'freebsd' and gnuplothasterm('qt') then
      return  'qt'
   elseif os == 'freebsd' and gnuplothasterm('x11') then
      return  'x11'
   elseif os == 'mac' and gnuplothasterm('aqua') then
      return  'aqua'
   elseif os == 'mac' and gnuplothasterm('wxt') then
      return  'wxt'
   elseif os == 'mac' and gnuplothasterm('qt') then
      return  'qt'
   elseif os == 'mac' and gnuplothasterm('x11') then
      return  'x11'
   else
      print('Can not find any of the default terminals for ' .. os .. '. ' ..
            'You can manually set terminal by gnuplot.setterm("terminal-name")')
      return nil
   end
end

local function findgnuplot()
   local exe = findgnuplotexe()
   local os = findos()
   if not exe then
      return nil
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
      elseif x:dim() == 2 and x:size(2) == 4 and format == 'v' then
         y = torch.Tensor(x:size(1),2)
         xx= torch.Tensor(x:size(1),2)
         y:select(2,1):copy(x:select(2,2))
         y:select(2,2):copy(x:select(2,4))
         xx:select(2,1):copy(x:select(2,1))
         xx:select(2,2):copy(x:select(2,3))
         x = xx
      elseif x:dim() == 2 and x:size(2) > 1 then
         y = x[{ {}, {2,-1} }]
         x = x:select(2,1)
      else
         y = x
         x = torch.range(1,y:size(1))
      end
   end
   if x:dim() ~= 1 and x:dim() ~= 2 then
      error('x and y dims are wrong :  x = ' .. x:nDimension() .. 'D y = ' .. y:nDimension() .. 'D')
   end
   if y:size(1) ~= x:size(1) then
      error('x and y dims are wrong :  x = ' .. x:nDimension() .. 'D y = ' .. y:nDimension() .. 'D')
   end
   -- if x:dim() ~= y:dim() or x:nDimension() > 2 or y:nDimension() > 2 then
   --    error('x and y dims are wrong :  x = ' .. x:nDimension() .. 'D y = ' .. y:nDimension() .. 'D')
   -- end
   -- print(x:size(),y:size())
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
      else return 'with ' .. f
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
         elseif yi:dim() == 1 then
            table.insert(dstr,string.format('%g %g\n',xi[j],yi[j]))
         else
            table.insert(dstr,string.format(string.rep('%g ',1+yi:size(2)) .. '\n',xi[j],unpack(yi[j]:clone():storage():totable())))
         end
      end
      collectgarbage()
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
   local term = gnuplothasterm('pngcairo') and 'pngcairo' or 'png'
   filefigure(fname,term,n)
   return _gptable.current
end

function gnuplot.pdffigure(fname,n)
   local haspdf = gnuplothasterm('pdf') or gnuplothasterm('pdfcairo')
   if not haspdf then
     error('your installation of gnuplot does not have pdf support enabled')
   end
   local term = nil
   if gnuplothasterm('pdfcairo') then
      term = 'pdfcairo enhanced color'
   else
      term = 'pdf enhanced color'
   end
   filefigure(fname,term,n)
   return _gptable.current
end

function gnuplot.figprint(fname)
   local suffix = fname:match('.+%.(.+)')
   local term = nil
   local haspdf = gnuplothasterm('pdf') or gnuplothasterm('pdfcairo')
   if suffix == 'eps' then
      term = 'postscript eps enhanced color'
   elseif suffix == 'png' then
      term = gnuplothasterm('pngcairo') and 'pngcairo' or 'png'
      term = term .. ' size "1024,768"'
   elseif suffix == 'pdf' and haspdf then
      if not haspdf then
          error('your installation of gnuplot does not have pdf support enabled')
      end
      if gnuplothasterm('pdfcairo') then
          term = 'pdfcairo'
      else
          term = 'pdf'
      end
      term = term .. ' enhanced color'
   else
      local errmsg = 'only eps and png'
      if haspdf then
          errmsg = errmsg .. ' and pdf'
      end
      error(errmsg ' for figprint')
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
