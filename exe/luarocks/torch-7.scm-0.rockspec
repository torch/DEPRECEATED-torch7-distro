package = "torch"

version = "7.scm-0"

source = {
   url = "http://www.torch.ch/license"
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
   -- If you depend on other rocks, add them here
}

build = {
   type = "builtin",
   modules = {}
   -- We'll start here.
}
