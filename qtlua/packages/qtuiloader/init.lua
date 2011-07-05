
require 'qtcore'

if qt and qt.qApp and qt.qApp:runsWithoutGraphics() then
   print("qlua: not loading module qtuiloader (running with -nographics)")
   return
end

qt.require 'libqtuiloader'

local qt = qt

module('qtuiloader')

local theloader = nil

function loader()
   if (not theloader or not theloader:tobool()) then
      theloader = qt.QUiLoader()
   end
   return theloader;
end

local loaderFunctions = {
   "load", "availableWidgets", "createWidget",
   "createLayout", "createAction", "createActionGroup" }

for i = 1,#loaderFunctions do
   local f = loaderFunctions[i]
   _M[f] = function(...) 
              local uiloader = loader()
              return uiloader[f](uiloader,...)
           end
end
