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

for _,tensortype in ipairs({'ByteTensor',
                      'CharTensor',
                      'ShortTensor',
                      'IntTensor',
                      'LongTensor',
                      'FloatTensor',
                      'DoubleTensor'}) do

   for _,func in ipairs({'add',
                         'mul',
                         'div',
                         'cmul',
                         'cdiv',
                         'addcmul',
                         'addcdiv',
                         'log',
                         'log1p',
                         'exp',
                         'cos',
                         'acos',
                         'cosh',
                         'sin',
                         'asin',
                         'sinh',
                         'tan',
                         'atan',
                         'tanh',
                         'pow',
                         'sqrt',
                         'ceil',
                         'floor',
                         'abs'
                      }) do

      local labfunc = torch[tensortype].lab[func]
      torch[tensortype][func] = function(self, ...)
                             return labfunc(self, self, ...)
                          end      
   end

   for _,func in ipairs({'addmv',
                         'addmm',
                         'addr'}) do
      
      local labfunc = torch[tensortype].lab[func]
      torch[tensortype][func] = function(self, next1, next2, ...)
                                   if type(next1) == 'number' and type(next2) == 'number' then
                                      return labfunc(self, next1, self, next2, ...)
                                   elseif type(next1) == 'number' then
                                      return labfunc(self, self, next1, next2, ...)                                      
                                   else
                                      return labfunc(self, self, next1, next2, ...)
                                   end
                          end      
   end

   for _,func in ipairs({'zero',
                         'fill',
                         'dot',
                         'minall',
                         'maxall',
                         'sumall',                         
                         'numel',
                         'max',
                         'min',
                         'sum',
                         'prod',
                         'cumsum',
                         'cumprod',
                         'trace',
                         'cross',
                         'zeros',
                         'ones',
                         'diag',
                         'eye',
                         'range',
                         'randperm',
                         'reshape',
                         'sort',
                         'tril',
                         'triu',
                         '_histc',
                         'cat',
                         'mean',
                         'std',
                         'var',
                         'norm',
                         'dist',
                         'meanall',
                         'varall',
                         'stdall',
                         'linspace',
                         'logspace',
                         'rand',
                         'randn'}) do

      torch[tensortype][func] = torch[tensortype].lab[func]
   end
end
