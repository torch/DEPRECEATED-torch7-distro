package = "torch"
version = "7.0.1-0"

source = {
  url = "git://github.com/andresy/torch.git",
  tag = '7.0.1-0'
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
  license = "BSD"
}

dependencies = {
  "lua >= 5.1"
}

build = {
  type = "cmake",
  variables = {
    LUAROCKS_PREFIX="$(PREFIX)",
    WITH_LUA_JIT="$(WITH_LUA_JIT)"
  }
}

