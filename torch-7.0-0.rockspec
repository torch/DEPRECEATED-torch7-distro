package = "torch"
version = "7.0-0"

source = {
  url = "https://github.com/andresy/torch.git"
}

description = {
  summary = "Torch",
  detailed = [[
Torch7 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. 
It is easy to use and provides a very efficient implementation, thanks 
to an easy and fast scripting language (Lua) and a underlying C 
implementation.
  ]],
  homepage = "http://www.torch.ch",
  license = "MIT/X11"
}

dependencies = {
  "lua >= 5.1"
}

build = {
  type = "cmake",
  variables = {LUAROCKS_PREFIX="$(PREFIX)"}
}

