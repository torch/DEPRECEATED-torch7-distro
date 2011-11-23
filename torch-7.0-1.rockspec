
package = "torch"
version = "7.0-1"

source = {
   url = package.."-"..version..".tgz"
}

description = {
   summary = "Torch7 provides a Matlab-like environment for Lua.",
   detailed = [[
         Torch7 provides a Matlab-like environment for state-of-the-art machine
         learning algorithms. It is easy to use and provides a very efficient
         implementation, thanks to an easy and fast scripting language (Lua) and a
         underlying C implementation.
   ]],
   homepage = "https://github.com/andresy/torch",
   license = "MIT/X11" 
}

dependencies = {
   "lua >= 5.1"
}

build = {
   type = "cmake",
   variables = {
      LUAROCKS_PREFIX = "$(PREFIX)"
   }
}
