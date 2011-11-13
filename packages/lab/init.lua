require "torch"
require "liblab"
require "random"

local oldtorchsetdefaulttensortype = torch.setdefaulttensortype

function torch.setdefaulttensortype(typename)
   oldtorchsetdefaulttensortype(typename)
   lab.setdefaulttensortype(typename)
end

function lab.manualSeed(seed)
   random.manualSeed(seed)
end

lab.setdefaulttensortype(torch.getdefaulttensortype())

torch.include('lab','hist.lua')
torch.include('lab','test.lua')