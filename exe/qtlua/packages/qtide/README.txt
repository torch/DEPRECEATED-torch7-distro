

This directory contains the qlua user 
interface as a loadable lua module.
Here is a brief description of the various files:


qtide.h            
   General header


qtide.qrc          
   General resource file


qtide.cpp          
   Lua bindings


qluatextedit.h
qluatextedit.cpp

   Class QLuaTextEdit is a text editor widget derived 
   from QPlainTextEdit with support for line numbers, 
   find dialog, replace dialog, and printing.

   Class QLuaTextEditMode represents customizable parts of 
   the editor for various languages. Modes can be changed by passing 
   a factory to function QLuaTextEdit::setEditorMode().

   Class QLuaTextEditModeFactory is used to construct QLuaTextEditMode 
   objects. All factories are chained to easily find the 
   most appropriate mode.


qluamode.h
qluamode.cpp

   Class QLuaMode is an abstract class derived from QLuaTextEditMode 
   that facilitates the implementation of editor modes with 
   syntax highlighting, matches, autoindent, completion, etc.


qluamode_txt.cpp
qluamode_hlp.cpp
qluamode_lua.cpp
qluamode_c.cpp

   These files define subclasses of QLuaMode for text files,
   torch help files, and lua source files.


qluamainwindow.h
qluamainwindow.cpp

   Class QLuaMainWindow is derived from QMainWindow 
   and implements common features of main windows,
   particularly to create QActions and to facilitate
   the MDI mode.


qluaeditor.h
qluaeditor.cpp

   Class QLuaEditor is a subclass of QLuaMainWindow
   implementing a complete text editor based on QLuaTextEdit


qluasdimain.h
qluasdimain.cpp

   Class QLuaconsoleWidget is a widget derived from QLuaTextEdit
   that mirrors the output of the Lua console.
  
   Class QLuaSdiMain is a subclass of QLuaMainWindow containing
   a QLuaconsoleWidget and a small editor for entering commands.
   This class also implements the command history support.
 

qluamdimain.h
qluamdimain.cpp

   Class QLuaMdiMain is a main window that captures the other 
   main windows as subwindows in a mdi setup. This is rather
   complicated as it involves copying the subwindow menubars
   into the mdi window and displacing the subwindow status bars
   into the mdi window as well.  


qluaide.h
qluaide.cpp

   Class QLuaIde is an invisible object that coordinates
   all the windows of the qlua interface.  It maintains 
   a list of active windows, keeps a list of recent files, 
   opens editors when a lua error occurs, etc.


qluabrowser.h
qluabrowser.cpp

   Class QLuaBrowser is a minimal web browser based on qtwebkit
   that is convenient for browsing the torch documentation. 
   When qtwebkit is not available, the class invokes a
   real browser instead.


prefs.lua
prefs.ui

   The implementation of the preference dialog.
