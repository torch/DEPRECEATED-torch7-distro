local Tester = torch.class('torch.Tester')

function Tester:__init()
   self.errors = {}
   self.tests = {}
   self.testnames = {}
   self.curtestname = ''
end


function Tester:assert_sub (condition, message)
   if not condition then
      local ss = debug.traceback('tester',2)
      --print(ss)
      ss = ss:match('[^\n]+\n[^\n]+\n([^\n]+\n[^\n]+)\n')
      self.errors[#self.errors+1] = self.curtestname .. '\n' .. message .. '\n' .. ss .. '\n'
   end
end
function Tester:assert (condition, message)
   self:assert_sub(condition,string.format('%s\n%s  condition=%s',message,' BOOL violation ', tostring(condition)))
end
function Tester:assert_lt (val, condition, message)
   self:assert_sub(val<condition,string.format('%s\n%s  val=%s, condition=%s',message,' LT(<) violation ', tostring(val), tostring(condition)))
end
function Tester:assert_gt (val, condition, message)
   self:assert_sub(val>condition,string.format('%s\n%s  val=%s, condition=%s',message,' LT(<) violation ', tostring(val), tostring(condition)))
end
function Tester:assert_le (val, condition, message)
   self:assert_sub(val<=condition,string.format('%s\n%s  val=%s, condition=%s',message,' LT(<) violation ', tostring(val), tostring(condition)))
end
function Tester:assert_ge (val, condition, message)
   self:assert_sub(val>=condition,string.format('%s\n%s  val=%s, condition=%s',message,' LT(<) violation ', tostring(val), tostring(condition)))
end
function Tester:assert_eq (val, condition, message)
   self:assert_sub(val==condition,string.format('%s\n%s  val=%s, condition=%s',message,' LT(<) violation ', tostring(val), tostring(condition)))
end

function Tester:pcall(f)
   local nerr = #self.errors
   local res = f()
--    local stat, result = pcall(f)
--    if not stat then
--       result = result .. debug.traceback()
--    end
--    return stat, result, stat and (nerr == #self.errors)
   return true, res, nerr == #self.errors
end

function Tester:report()
   print('Completed ' .. #self.tests .. ' tests with ' .. #self.errors .. ' errors')
   print()
   print(string.rep('-',80))
   for i,v in ipairs(self.errors) do
      print(v)
      print(string.rep('-',80))
   end
end

function Tester:run()
   print('Running ' .. #self.tests .. ' tests')
   io.write(string.rep('_',#self.tests).. '\r')
   for i,v in ipairs(self.tests) do
      io.write(string.format('|'))
      self.curtestname = self.testnames[i]
      
      local stat, message, pass = self:pcall(v)
      
      if pass then
	 io.write(string.format('\b_'))
      else
	 io.write(string.format('\b*'))
      end
      
      if not stat then
	 print()
	 print('Function call failed: Test No ' .. i .. ' ' .. self.testnames[i])
	 print(message)
      end
   end
   print()
   print()
   self:report()
end

function Tester:add(f,name)
   name = name or 'unknown'
   if type(f) == "table" then
      for i,v in pairs(f) do
	 self:add(v,i)
      end
   elseif type(f) == "function" then
      self.tests[#self.tests+1] = f
      self.testnames[#self.tests] = name
   else
      error('Tester:add(f) expects a function or a table of functions')
   end
end
