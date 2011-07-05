
require 'paths'
require 'qtcore'
require 'qtgui'

if qt and qt.qApp and qt.qApp:runsWithoutGraphics() then
   print("qlua: not loading module qtide (running with -nographics)")
   return
end

qt.require 'libqtide'

local G = _G
local error = error
local dofile = dofile
local loadstring = loadstring
local print = print
local qt = qt
local string = string
local tonumber = tonumber
local tostring = tostring
local type = type
local paths = paths
local pcall = pcall

module('qtide')

-- Startup --

local qluaide = qt.QLuaIde()

local function realmode(mode)
   mode = mode or qt.qApp:readSettings("ide/mode")
   if qluaide.mdiDefault then
      return mode or 'mdi'
   else
      return mode or 'sdi'
   end
end

function setup(mode)
   mode = realmode(mode)
   if mode == "mdi" then     -- subwindows within a big window
      local mdi = qluaide:createMdiMain()
      mdi.tabMode = false
      mdi.clientClass = "QWidget"
      mdi:adoptAll()
   elseif mode == "tab" then     -- groups all editors in tabs
      local mdi = qluaide:createMdiMain()
      mdi.tabMode = true
      mdi.clientClass = "QLuaEditor"
      mdi:adoptAll()
   elseif mode == "tab2" then    -- groups all editors + console in tabs
      local mdi = qluaide:createMdiMain()
      mdi.tabMode = true
      mdi.clientClass = "QLuaMainWindow"
      mdi:adoptAll()
   else                            -- all windows separate
      if mode ~= "sdi" then
         print("Warning: The recognized ide styles are: sdi, mdi, tab")
         mode = 'sdi'
      end
      local mdi = qt.qLuaMdiMain
      if mdi then
         mdi:hide()
         mdi.tabMode = false
         mdi.clientClass = "-none"
         mdi:adoptAll()
         mdi:deleteLater()
      end
   end
   qt.qApp:writeSettings("ide/mode", mode)
end

function start(mode)
   setup(mode)
   if not qt.qLuaSdiMain then
      qluaide:createSdiMain()
      qluaide.editOnError = true
   end
end



-- Editor --

function editor(s)
   local e = qluaide:editor(s or "")
   if e == nil and type(s) == "string" then
      error(string.format("Unable to read file '%s'", s))
   end
   return e
end


function doeditor(e)
   -- validate parameter
   if not qt.isa(e, 'QLuaEditor*') then 
      error(string.format("QLuaEditor expected, got %s.", s)); 
   end
   -- retrieve text
   local n = "qt." .. tostring(e.objectName)
   local currentfile = nil;
   if e.fileName:tobool() then
      currentfile = e.fileName:tostring()
   end
   if (currentfile and not e.windowModified) then
      dofile(currentfile)
   else
      -- load data from editor
      local chunk, m = loadstring(e:widget().plainText:tostring(), n)
      if not chunk then
         print(m)
         -- error while parsing the data
         local _,_,l,m = string.find(m,"^%[string.*%]:(%d+): +(.*)")
         if l and m and qluaide.editOnError then 
            e:widget():showLine(tonumber(l)) 
            e:showStatusMessage(m)
         end
      else
         -- execution starts
         chunk()
      end
   end
end


-- Inspector --

function inspector(...)
   error("Function qtide.inspector is not yet working")
end





-- Browser --

function browser(url)
   return qluaide:browser(url or "about:/")
end


-- Help --

local function locate_help_files()
   local appname = qt.qApp.applicationName:tostring()
   local index1 = paths.concat(paths.install_html, appname:lower(), "index.html")
   local index2 = paths.concat(paths.install_html,"index.html")
   if index1 and paths.filep(index1) then
      return qt.QUrl(index1)
   elseif index2 and paths.filep(index2) then
      return qt.QUrl(index2)
   else
      return qt.QUrl("http://torch5.sourceforge.net/manual/index.html")
   end
end


helpbrowser = nil
helpurl = locate_help_files()

function help()
   local appname = qt.qApp.applicationName:tostring()
   if not helpurl then
      error("The html help files are not installed.")
   end
   if not qt.isa(helpbrowser, "QWidget") then
      helpbrowser = qluaide:browser()
   end
   helpbrowser.baseTitle =  appname .. " Help Browser"
   helpbrowser.homeUrl = helpurl;
   helpbrowser.url = helpurl
   helpbrowser:raise()
   return helpbrowser
end

qt.disconnect(qluaide, 'helpRequested(QWidget*)')
qt.connect(qluaide,'helpRequested(QWidget*)', 
           function(main) 
              local success, message = pcall(help)
              if not success and type(message) == "string" then
                 qluaide:messageBox("QLua Warning", message)
              end
           end)


-- Preferences --


function preferences()
   G.require 'qtide.prefs'
   local d = prefs.createPreferencesDialog()
   if d and d.dialog:exec() > 0 then
      prefs.savePreferences(d)
   end
end

qt.disconnect(qluaide,'prefsRequested(QWidget*)')
qt.connect(qluaide,'prefsRequested(QWidget*)',
           function(main) 
              local success, message = pcall(preferences)
              if not success and type(message) == "string" then
                 qluaide:messageBox("QLua Warning", message)
              end
           end)




