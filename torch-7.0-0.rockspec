package = "torch"
version = "7.0-0"

source = {
  url = "https://github.com/andresy/torch.git"
}

description = {
  summary = "Torch 7",
  detailed = [[
Torch 7: a Matlab-like numeric framework for Lua.  
]],
  homepage = "http://www.torch.ch",
  license = "MIT/X11"
}

dependencies = {
  "lua == 5.1"
}

build = {
  type = "cmake",
  variables = {LUA_DIR="$(LUADIR)", LIB_DIR="$(LIBDIR)"}
}

