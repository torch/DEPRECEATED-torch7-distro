
require 'qt'
require 'qtcore'

if qt and qt.qApp and qt.qApp:runsWithoutGraphics() then
   print("qlua: not loading module qtgui (running with -nographics)")
   return
end

qt.require 'libqtgui'


