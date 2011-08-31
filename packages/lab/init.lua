require "torch"
require "liblab"

local oldtorchsetdefaulttensortype = torch.setdefaulttensortype

function torch.setdefaulttensortype(typename)
   oldtorchsetdefaulttensortype(typename)
   lab.setdefaulttensortype(typename)
end

lab.setdefaulttensortype(torch.getdefaulttensortype())

torch.include('lab','hist.lua')
