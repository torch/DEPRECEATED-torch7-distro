
require 'qtuiloader'

local G = _G
local dofile = dofile
local error = error
local ipairs = ipairs
local loadstring = loadstring
local pairs = pairs
local paths = paths
local pcall = pcall
local print = print
local qt = qt
local string = string
local tonumber = tonumber
local tostring = tostring
local type = type

module('qtide.prefs')


local uiPreferences = paths.thisfile("prefs.ui")

local function getFontForLabel(label)
   local f = qt.QFontDialog.getFont(label.font, label)
   if qt.isa(f,'QFont') then
      label.font = f
   end
end

local function readSettingsNumber(a,k)
   local v = a:readSettings(k)
   if qt.type(v) then v = v:tostring() end
   return tonumber(v)
end

local function readSettingsBoolean(a,k)
   local v = a:readSettings(k)
   if qt.type(v) then v = v:tostring() end
   if v == "true" then return true end
   if v == "false" then return false end
   return nil
end

function createPreferencesDialog()
   if not paths.filep(uiPreferences) then
      error("Unable to locate file 'prefs.ui'")
   end
   local v = nil
   local d = {}
   local a = qt.qApp
   -- create dialog
   d.dialog = G.qtuiloader.load(uiPreferences)
   -- cache subwidgets
   d.labelFontConsole = d.dialog.labelFontConsole
   d.btnFontConsole = d.dialog.btnFontConsole
   d.spnConsoleLines = d.dialog.spnConsoleLines
   d.spnHistorySize = d.dialog.spnHistorySize
   d.spnConsoleSizeW = d.dialog.spnConsoleSizeW
   d.spnConsoleSizeH = d.dialog.spnConsoleSizeH
   d.spnInputSizeW = d.dialog.spnInputSizeW
   d.spnInputSizeH = d.dialog.spnInputSizeH
   d.spnConsoleTabSize = d.dialog.spnConsoleTabSize
   d.chkConsoleTabExpand = d.dialog.chkConsoleTabExpand
   d.labelFontEditor = d.dialog.labelFontEditor
   d.btnFontEditor = d.dialog.btnFontEditor
   d.spnEditorSizeW = d.dialog.spnEditorSizeW
   d.spnEditorSizeH = d.dialog.spnEditorSizeH
   d.spnEditorTabSize = d.dialog.spnEditorTabSize
   d.chkEditorTabExpand = d.dialog.chkEditorTabExpand
   d.comboMode = d.dialog.comboMode
   d.spnBaseIndent = d.dialog.spnBaseIndent
   d.comboFormat = d.dialog.comboFormat
   d.labelFormat = d.dialog.labelFormat
   d.chkBold = d.dialog.chkBold
   d.chkItalic = d.dialog.chkItalic
   d.btnForeground = d.dialog.btnForeground
   d.btnBackground = d.dialog.btnBackground
   -- set window title
   d.dialog.windowTitle = a.applicationName:tostring() .. " Preferences"
   -- connect font buttons
   qt.connect(d.btnFontEditor,'clicked()',
              function() getFontForLabel(d.labelFontEditor) end, true)
   qt.connect(d.btnFontConsole,'clicked()',
              function() getFontForLabel(d.labelFontConsole) end, true)
   -- connect color buttons
   -- default font
   local f = qt.QFont{typewriter=true, fixedPitch=true, family="monospace"}
   if qt.isa(qt.qLuaSdiMain,'QLuaSdiMain') then
      f = qt.QFont(qt.qLuaSdiMain:consoleWidget().font:info())
   end
   -- read editor/font
   v = a:readSettings("editor/font")
   if not qt.isa(v,"QFont") then v = f end
   d.labelFontEditor.font = v
   -- read editor/size
   v = a:readSettings("editor/size")
   if qt.isa(v,"QSize") then
      local t = v:totable()
      d.spnEditorSizeW.value = t.width
      d.spnEditorSizeH.value = t.height
   end
   -- read editor/tabSize
   v = readSettingsNumber(a,"editor/tabSize")
   if type(v) == "number" and v > 0 and v <= 16 then
      d.spnEditorTabSize.value = v
   end
   -- read editor/tabExpand
   v = readSettingsBoolean(a,"editor/tabExpand")
   if type(v) == "boolean" then
      d.chkEditorTabExpand.checked = v
   end
   -- read console/font
   v = a:readSettings("console/font")
   if not qt.isa(v,"QFont") then v = f end
   d.labelFontConsole.font = v
   -- read console/consoleLines
   v = readSettingsNumber(a,"console/consoleLines")
   if type(v) == "number" and v > 0 and v <= 99999 then
      d.spnConsoleLines.value = v
   end
   -- read console/historySize
   v = readSettingsNumber(a,"console/historySize")
   if type(v) == "number" and v > 0 and v <= 9999 then
      d.spnHistorySize.value = v
   end
   -- read console/editorSize
   v = a:readSettings("console/editorSize")
   if qt.isa(v,"QSize") then
      local t = v:totable()
      d.spnInputSizeW.value = t.width
      d.spnInputSizeH.value = t.height
   end
   -- read console/consoleSize
   v = a:readSettings("console/consoleSize")
   if qt.isa(v,"QSize") then
      local t = v:totable()
      d.spnConsoleSizeW.value = t.width
      d.spnConsoleSizeH.value = t.height
   end
   -- read console/tabSize
   v = readSettingsNumber(a,"editor/tabSize")
   if type(v) == "number" and v > 0 and v <= 16 then
      d.spnEditorTabSize.value = v
   end
   -- read console/tabExpand
   v = readSettingsBoolean(a,"editor/tabExpand")
   if type(v) == "boolean" then
      d.chkEditorTabExpand.checked = v
   end
   -- formats (not yet implemented)
   d.dialog.tabFormats:deleteLater() -- to avoid confusion
   -- return
   return d
end


function savePreferences(d)
   local a = qt.qApp
   local f,w,h,ts,te,cl,hs
   local ide = qt.QLuaIde()
   -- find windows
   local windows = {}
   for i,n in pairs(ide:windowNames():totable()) do
      windows[i] = qt[n:tostring()]
   end
   -- save editor/font
   f = d.labelFontEditor.font
   a:writeSettings("editor/font", f)
   -- save editor/size
   w = d.spnEditorSizeW.value
   h = d.spnEditorSizeH.value
   a:writeSettings("editor/size", qt.QSize{width=w,height=h})
   -- save editor/tabSize
   ts = d.spnEditorTabSize.value
   a:writeSettings("editor/tabSize", ts)
   -- save editor/tabExpand
   te = d.chkEditorTabExpand.checked
   a:writeSettings("editor/tabExpand", te)
   -- update editors
   for _,w in pairs(windows) do
      if qt.isa(w, 'QLuaEditor') then
         local e = w:widget()
         e.font = f
         e.tabSize = ts
         e.tabExpand = te
      end
   end
   -- save console/font
   f = d.labelFontConsole.font
   a:writeSettings("console/font", f)
   -- save console/consoleSize
   w = d.spnConsoleSizeW.value
   h = d.spnConsoleSizeH.value
   a:writeSettings("console/consoleSize", qt.QSize{width=w,height=h})
   -- save console/editorSize
   w = d.spnInputSizeW.value
   h = d.spnInputSizeH.value
   a:writeSettings("console/editorSize", qt.QSize{width=w,height=h})
   -- save console/tabSize
   ts = d.spnConsoleTabSize.value
   a:writeSettings("console/tabSize", ts)
   -- save console/tabExpand
   te = d.chkConsoleTabExpand.checked
   a:writeSettings("console/tabExpand", te)
   -- save console/consoleLines
   cl = d.spnConsoleLines.value
   a:writeSettings("console/consoleLines", cl)
   -- save console/historySize
   hs = d.spnHistorySize.value
   a:writeSettings("console/historySize", cl)
   -- update console
   for _,w in pairs(windows) do
      if qt.isa(w, 'QLuaSdiMain') then
         local e = w:editorWidget()
         local c = w:consoleWidget()
         e.font = f
         e.tabSize = ts
         c.font = f
         c.tabSize = ts
         c.tabExpand = te
         w.consoleLines = cl
         w.historySize = hs
      end
   end
   
end


