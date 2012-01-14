
require 'qtcore'
require 'qtgui'
require 'qtsvg'

if qt and qt.qApp and qt.qApp:runsWithoutGraphics() then
   --print("qlua: not loading module qtwidget (running with -nographics)")
   print('qlua: qtwidget window functions will not be usable (running with -nographics)')
   --return
end

qt.require 'libqtwidget'
if (package.loaded['torch']) then 
   require('libqttorch') 
end


-- More handy variants of some functions

local oldcurrentpoint = qt.QtLuaPainter.currentpoint
function qt.QtLuaPainter:currentpoint()
   local p = oldcurrentpoint(self):totable()
   return p.x, p.y
end

local oldsetpoint = qt.QtLuaPainter.setpoint
function qt.QtLuaPainter:setpoint(x,y)
   oldsetpoint(self, qt.QPointF{x=x,y=y})
end


-- Additional functions for qtluapainter


function qt.QtLuaPainter:currentfontsize()
   local f = self:currentfont():totable()
   return f.size
end

function qt.QtLuaPainter:setfontsize(sz)
   local f = self:currentfont():totable()
   f.size = sz;
   f.pixelSize = nil;
   f.pointSize = nil;
   self:setfont(qt.QFont(f))
end

function qt.QtLuaPainter:currentdash()
   local p = self:currentpen():totable();
   return p.dashPattern, p.dashOffset
end

function qt.QtLuaPainter:setdash(size,offset)
   local p = self:currentpen():totable();
   if (type(size) == 'number') then
      size = { size, size }
   end
   if (type(size) == 'table' and #size == 0) then
      size = nil
   end
   if (type(size) == 'table' and #size % 2 == 1) then
      size[#size+1] = size[#size]
   end
   if (size) then
      p.style = "CustomDashLine"
      p.dashPattern = size
      p.dashOffset = offset
   else
      p.style = "SolidLine"
      p.dashPattern = nil
      p.dashOffset = nil
   end
   self:setpen(qt.QPen(p))
end

function qt.QtLuaPainter:currentcolor()
   local s = self:currentbrush():totable().color:totable()
   return s.r, s.g, s.b, s.a
end

function qt.QtLuaPainter:setcolor(...)
   local b = self:currentbrush():totable()
   local p = self:currentpen():totable()
   local c = ...
   if (qt.type(c) ~= "QColor") then
      c = qt.QColor(...)
   end
   b.color = c
   p.color = c
   p.brush = nil
   self:setbrush(qt.QBrush(b))
   self:setpen(qt.QPen(p))
end

function qt.QtLuaPainter:currentlinewidth()
   local p = self:currentpen():totable()
   return p.width
end

function qt.QtLuaPainter:setlinewidth(w)
   local p = self:currentpen():totable()
   p.width = w
   self:setpen(qt.QPen(p))
end

function qt.QtLuaPainter:setpattern(p,x,y)
   local image = nil;
   if (qt.type(p) == "QImage") then image = p else image = p:image() end
   local b = self:currentbrush():totable()
   local p = self:currentpen():totable()
   b.style = "TexturePattern"
   b.texture = image
   if (x and y and x ~= 0 and y ~= 0) then
      b.transform = qt.QTransform():translated(-x,-y)
   end
   local qb = qt.QBrush(b)
   p.brush = qb
   p.color = nil
   local qp = qt.QPen(p)
   self:setbrush(qb)
   self:setpen(qp)
end

function qt.QtLuaPainter:write(...)
   local i = self:image()
   i:save(...)
end

function qt.QtLuaPainter:currentsize()
   local s = self:size():totable()
   return s.width, s.height
end



-- Convenience functions for opening windows
-- or drawing into files and other images.

local G = _G
local qt = qt
local type = type
local pcall = pcall
local setmetatable = setmetatable

module('qtwidget')

local painterFunctions = {
   -- c functions
   "arc", "arcn", "arcto", "charpath", "clip", "close", "closepath", "concat",
   "currentangleunit", "currentbackground", "currentbrush", 
   "currentbrushorigin", "currentclip", "currentfont", "currenthints",
   "currentmatrix", "currentmode", "currentpath", "currentpen", "currentpoint",
   "currentstylesheet", "curveto", "depth", "device", "eoclip", "eofill",
   "fill", "gbegin", "gend", "grestore", "gsave", "height", "image", 
   "initclip", "initgraphics", "initmatrix", "lineto", "moveto", "newpath",
   "painter", "pixmap", "printer", "rcurveto", "rect", "rectangle", "refresh",
   "rlineto", "rmoveto", "rotate", "scale", "setangleunit", "setbackground",
   "setbrush", "setbrushorigin", "setclip", "setfont", "sethints",
   "setmatrix", "setmode", "setpath", "setpen", "setpoint", "setstylesheet",
   "show", "showpage", "size", "stringrect", "stringwidth",
   "stroke", "translate", "widget", "width",
   -- lua functions
   "currentfontsize", "setfontsize", 
   "currentdash", "setdash",
   "currentcolor", "setcolor",
   "currentlinewidth", "setlinewidth",
   "setpattern", "write", "currentsize"
}

local function declareRelayFunctions(class)
   for i=1,#painterFunctions do
      local f = painterFunctions[i];
      class[f] = function(self,...) return self.port[f](self.port,...) end
   end
end


-- windows

windowClass = {}
windowClass.__index = windowClass
declareRelayFunctions(windowClass)

function windowClass:valid()
   if qt.isa(self.widget,'QWidget') 
      and self.widget.visible then
      return 1 
   else
      return false
   end
end

function windowClass:resize(w,h)
   self.widget.size = qt.QSize{width=w, height=h}
end

function windowClass:onResize(f)
   qt.disconnect(self.timer, 'timeout()')
   if (type(f) == 'function') then
      qt.connect(self.timer, 'timeout()',
                 function() pcall(f, self.width, self.height) end )
   end
end

function windowClass:close()
   pcall(function() qt.disconnect(self.timer, 'timeout()') end)
   pcall(function() self.timer:deleteLater() end)
   pcall(function() self.port:close() end)
   pcall(function() self.widget:deleteLater() end)
end

function newwindow(w,h,title)
   local self = {}
   setmetatable(self, windowClass)
   self.widget = qt.QWidget()
   self.widget.size = qt.QSize{width=w,height=h}
   self.widget.windowTitle = title or "QLua Graphics"
   self.widget.focusPolicy = 'WheelFocus'
   self.listener = qt.QtLuaListener(self.widget)
   self.timer = qt.QTimer()
   self.timer.singleShot = true
   qt.connect(self.listener, 'sigResize(int,int)',
              function(w,h) 
                 pcall(function() 
                          self.width = w; 
                          self.height = h;
                          self.timer:start(0) 
                       end) end)
   qt.connect(self.listener, 'sigClose()',
              function()
                 pcall(function() self:close() end)
              end )
   self.port = qt.QtLuaPainter(self.widget)
   self.width, self.height = self.port:currentsize()
   self.depth = self.port.depth
   self.widget:show();
   self.widget:raise();
   return self
end



-- images

imageClass = {}
imageClass.__index = imageClass
declareRelayFunctions(imageClass)

function imageClass:valid()
   return true;
end

function newimage(...)
   local self = {}
   setmetatable(self, imageClass)
   local firstarg = ...
   if (G.package.loaded['torch'] and G.package.loaded['libqttorch'] and
       G.torch.typename(firstarg) == "torch.Tensor") then
      self.port = qt.QtLuaPainter(qt.QImage.fromTensor(firstarg))
   else
      self.port = qt.QtLuaPainter(...)
   end
   self.width, self.height = self.port:currentsize()
   self.depth = self.port.depth
   return self
end


-- printer

printerClass = {}
printerClass.__index = printerClass
declareRelayFunctions(printerClass)

function printerClass:valid()
   return true;
end

function newprint(w,h,printername)
   local self = {}
   setmetatable(self, printerClass)
   self.printer = qt.QtLuaPrinter()
   self.printer.printerName = printername or "";
   self.printer.paperSize = qt.QSizeF{width=w,height=h}
   if (printername ~= nil or self.printer:setup()) then
      self.port = qt.QtLuaPainter(self.printer)
      self.width, self.height = self.port:currentsize()
      self.depth = self.port.depth
      return self
   else
      return nil
   end
end

function newpdf(w,h,filename)
   local self = {}
   setmetatable(self, printerClass)
   self.printer = qt.QtLuaPrinter()
   self.printer.outputFileName = filename
   self.printer.outputFormat = 'PdfFormat'
   self.printer.paperSize = qt.QSizeF{width=w,height=h}
   self.printer.pageSize = 'Custom'
   self.printer.fullPage = true;
   self.port = qt.QtLuaPainter(self.printer)
   self.width, self.height = self.port:currentsize()
   self.depth = self.port.depth
   return self
end

function newps(w,h,filename)
   local self = {}
   setmetatable(self, printerClass)
   self.printer = qt.QtLuaPrinter()
   self.printer.outputFileName = filename
   self.printer.outputFormat = 'PostScriptFormat'
   self.printer.paperSize = qt.QSizeF{width=w,height=h}
   self.printer.pageSize = 'Custom'
   self.port = qt.QtLuaPainter(self.printer)
   self.width, self.height = self.port:currentsize()
   self.depth = self.port.depth
   return self
end

function newsvg(w,h,filename)
   local self = {}
   setmetatable(self, printerClass)
   self.svg = qt.QtLuaSvgGenerator(filename)
   self.svg.size = qt.QSize{width=w,height=h}
   self.svg.title = "QtLua SVG Document"
   self.port = qt.QtLuaPainter(self.svg)
   self.width, self.height = self.port:currentsize()
   self.depth = self.port.depth
   return self
end

