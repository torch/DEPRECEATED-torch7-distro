wrap = {}

dofile(debug.getinfo(1).source:gsub('init%.lua$', 'types.lua'):gsub('^@', ''))

local function bit(p)
   return 2 ^ (p - 1)  -- 1-based indexing                                                          
end

local function hasbit(x, p)
   return x % (p + p) >= p
end

local function beautify(txt)
   local indent = 0
   for i=1,#txt do
      if txt[i]:match('}') then
         indent = indent - 2
      end
      if indent > 0 then
         txt[i] = string.rep(' ', indent) .. txt[i]
      end
      if txt[i]:match('{') then
         indent = indent + 2
      end
   end
end

local function writeheaders(txt, args, argoffset)
   local helpargs = {}
   local cargs = {}
   local argcreturned
   argoffset = argoffset or 0

   for i,arg in ipairs(args) do
      arg.i = i+argoffset
      assert(wrap.argtypes[arg.name], 'unknown type ' .. arg.name)
      table.insert(txt, wrap.argtypes[arg.name].declare(arg))
      local helpname = wrap.argtypes[arg.name].helpname(arg)
      if arg.returned then
         name = string.format('*%s*', helpname)
      end
      if arg.default then
         table.insert(helpargs, string.format('[%s]', helpname))
      elseif not arg.creturned then
         table.insert(helpargs, helpname)
      end
      if arg.creturned then
         if argcreturned then
            error('A C function can only return one argument!')
         end
         if arg.default then
            error('Obviously, an "argument" returned by a C function cannot have a default value')
         end
         if arg.returned then
            error('Options "returned" and "creturned" are incompatible')
         end
         argcreturned = arg
      else
         table.insert(cargs, wrap.argtypes[arg.name].carg(arg))
      end
   end
   return helpargs, cargs, argcreturned
end

local function writechecks(txt, args, argset)
   local multiargset = argset
   argset = argset or 1

   local nopt = 0
   for i,arg in ipairs(args) do
      if arg.default then
         nopt = nopt + 1
      end
   end

   for variant=0,math.pow(2, nopt)-1 do
      local opt = 0
      local currentargs = {}
      for i,arg in ipairs(args) do
         if arg.default then
            opt = opt + 1
            if hasbit(variant, bit(opt)) then
               table.insert(currentargs, arg)
            end
         elseif not arg.creturned then
            table.insert(currentargs, arg)
         end
      end

      if variant == 0 and argset == 1 then
         table.insert(txt, string.format('if(narg == %d', #currentargs))
      else
         table.insert(txt, string.format('else if(narg == %d', #currentargs))
      end

      for stackidx, arg in ipairs(currentargs) do
         table.insert(txt, string.format("&& %s", wrap.argtypes[arg.name].check(arg, stackidx)))
      end
      table.insert(txt, ')')
      table.insert(txt, '{')

      if multiargset then
         table.insert(txt, string.format('argset = %d;', argset))
      end

      for stackidx, arg in ipairs(currentargs) do
         table.insert(txt, wrap.argtypes[arg.name].read(arg, stackidx))
      end

      table.insert(txt, '}')

   end
end

local function writecall(txt, args, cfuncname, cargs, argcreturned)
   for _,arg in ipairs(args) do
      local precall = wrap.argtypes[arg.name].precall(arg)
      if not precall or not precall:match('^%s*$') then
         table.insert(txt, precall)
      end
   end

   if argcreturned then
      table.insert(txt, string.format('%s = %s(%s);', wrap.argtypes[argcreturned.name].creturn(argcreturned), argcreturned.i, cfuncname, table.concat(cargs, ',')))
   else
      table.insert(txt, string.format('%s(%s);', cfuncname, table.concat(cargs, ',')))
   end

   for _,arg in ipairs(args) do
      local postcall = wrap.argtypes[arg.name].postcall(arg)
      if not postcall or not postcall:match('^%s*$') then
         table.insert(txt, postcall)
      end
   end

   local nret = 0
   if argcreturned then
      nret = nret + 1
   end
   for _,arg in ipairs(args) do
      if arg.returned then
         nret = nret + 1
      end
   end
   table.insert(txt, string.format('return %d;', nret))
end

function wrap.cinterface(luafuncname, ...)
   local txt = {}
   local varargs = {...}

   assert(#varargs > 0 and #varargs % 2 == 0, 'must provide both the C function name and the corresponding arguments')

   table.insert(txt, string.format("static int %s(lua_State *L)", luafuncname))
   table.insert(txt, "{")
   table.insert(txt, "int narg = lua_gettop(L);")

   if #varargs == 2 then
      local cfuncname = varargs[1]
      local args = varargs[2]
      
      local helpargs, cargs, argcreturned = writeheaders(txt, args)
      writechecks(txt, args)
      
      table.insert(txt, 'else')
      table.insert(txt, string.format('luaL_error(L, "expected arguments: %s");', table.concat(helpargs, ' ')))

      writecall(txt, args, cfuncname, cargs, argcreturned)
   else
      local allcfuncname = {}
      local allargs = {}
      local allhelpargs = {}
      local allcargs = {}
      local allargcreturned = {}

      table.insert(txt, "int argset = 0;")

      for k=1,#varargs/2 do
         allcfuncname[k] = varargs[(k-1)*2+1]
         allargs[k] = varargs[(k-1)*2+2]
      end

      local argoffset = 0
      for k=1,#varargs/2 do
         allhelpargs[k], allcargs[k], allargcreturned[k] = writeheaders(txt, allargs[k], argoffset)
         argoffset = argoffset + #allargs[k]
      end

      for k=1,#varargs/2 do
         writechecks(txt, allargs[k], k)
      end

      table.insert(txt, 'else')
      local allconcathelpargs = {}
      for k=1,#varargs/2 do
         table.insert(allconcathelpargs, table.concat(allhelpargs[k], ' '))
      end
      table.insert(txt, string.format('luaL_error(L, "expected arguments: %s");', table.concat(allconcathelpargs, ' | ')))

      for k=1,#varargs/2 do
         if k == 1 then
            table.insert(txt, string.format('if(argset == %d)', k))
         else
            table.insert(txt, string.format('else if(argset == %d)', k))
         end
         table.insert(txt, '{')
         writecall(txt, allargs[k], allcfuncname[k], allcargs[k], allargcreturned[k])
         table.insert(txt, '}')
      end

      table.insert(txt, 'return 0;')
   end

   table.insert(txt, '}')
   table.insert(txt, '')
   
   --   beautify(txt)
   print(table.concat(txt, '\n'))
   
end

