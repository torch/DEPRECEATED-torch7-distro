
-- A little demo of the capabilities of package uiloader.


require 'paths'
require 'qtuiloader'
require 'qtwidget'
require 'qt'


-- search file test.ui
local testui = paths.thisfile('test.ui')

-- demo proper
function demo()

   local widget = qtuiloader.load(testui)
   local painter = qt.QtLuaPainter(widget.frame)

   local function min(a,b)
      if (a < b) then return a else return b end
   end

   local function draw(w)
      painter:newpath()
      painter:moveto(-100,-100); 
      painter:curveto(-100,100,100,-100,100,100); 
      painter:closepath()
      if widget.checkBoxFC.checked then
         painter:setcolor("red"); 
         painter:fill(false)
      end
      if widget.checkBoxSC.checked then
         painter:setcolor("blue"); 
         painter:setlinewidth(5); 
         painter:stroke(false)
      end
      if widget.checkBoxST.checked then
         painter:moveto(-70,-80)
         painter:setcolor("black")
         painter:setfont(qt.QFont{serif=true,italic=true,size=16})
         painter:show(widget.lineEdit.text)
      end
      if widget.checkBoxFR.checked then
         painter:newpath()
         painter:rectangle(-50,-50,100,100)
         painter:setcolor(1,1,0,.5)
         painter:fill()
      end
   end

   local zrot = 0

   local function paint()
      local xrot = widget.verticalSlider.value
      painter:gbegin()
      painter:showpage()
      painter:gsave()
      painter:translate(painter.width/2, painter.height/2)
      local s = min(painter.width,painter.height)
      painter:scale(s/250, s/250)
      painter:concat(qt.QTransform():rotated(xrot,'XAxis'))
      painter:rotate(zrot)
      draw()
      painter:grestore()
      painter:gend()
   end

   local timer = qt.QTimer()
   timer.interval = 40
   timer.singleShot = true
   qt.connect(timer,'timeout()',
              function() 
                 if qt.isa(widget, "QWidget") and widget.visible then
                    zrot = ( zrot + widget.horizontalSlider.value ) % 360
                    paint()
                    timer:start()
                 end
              end )
   
   local listener = qt.QtLuaListener(widget)
   qt.connect(listener,'sigClose()',
              function()
                 timer:stop()
                 timer:deleteLater()
                 widget:deleteLater()
              end )
   qt.connect(listener,'sigShow(bool)',
              function(b) 
                 if b then timer:start() end 
              end )
   
   widget.windowTitle = "QtUiLoader demo"
   widget:show()

   return { widget=widget, 
            painter=painter, 
            listener=listener,
            timer=timer }
end



stuff = demo()

