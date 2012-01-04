
dok.inline = {}

paths.install_dok = paths.concat(paths.install_html, '..', 'dok')
paths.install_dokmedia = paths.concat(paths.install_html, '..', 'dokmedia')

dok.colors = {
   none = '\27[0m',
   black = '\27[0;30m',
   red = '\27[0;31m',
   green = '\27[0;32m',
   yellow = '\27[0;33m',
   blue = '\27[0;34m',
   magenta = '\27[0;35m',
   cyan = '\27[0;36m',
   white = '\27[0;37m',
   Black = '\27[1;30m',
   Red = '\27[1;31m',
   Green = '\27[1;32m',
   Yellow = '\27[1;33m',
   Blue = '\27[1;34m',
   Magenta = '\27[1;35m',
   Cyan = '\27[1;36m',
   White = '\27[1;37m',
   _black = '\27[40m',
   _red = '\27[41m',
   _green = '\27[42m',
   _yellow = '\27[43m',
   _blue = '\27[44m',
   _magenta = '\27[45m',
   _cyan = '\27[46m',
   _white = '\27[47m'
}
local c = dok.colors

local style = {
   banner = '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++',
   list = c.blue .. '> ' .. c.none,
   title = c.Magenta,
   pre = c.cyan,
   em = c.Black,
   img = c.red,
   link = c.red,
   code = c.green,
   none = c.none
}

local function uncleanText(txt)
   txt = txt:gsub('&#39;', "'")
   txt = txt:gsub('&#42;', '%*')
   txt = txt:gsub('&#43;', '%+')
   txt = txt:gsub('&lt;', '<')
   txt = txt:gsub('&gt;', '>')
   return txt
end

local function string2symbol(str)
   local ok, res = pcall(loadstring('local t = ' .. str .. '; return t'))
   if not ok then
      ok, res = pcall(loadstring('local t = _torchimport.' .. str .. '; return t'))
   end
   return res
end

local function maxcols(str, cols)
   cols = cols or 70
   local res = ''
   local k = 1
   local color = false
   for i = 1,#str do
      res = res .. str:sub(i,i)
      if str:sub(i,i) == '\27' then
         color = true
      elseif str:sub(i,i) == 'm' then
         color = false
      end
      if k == cols then
         if str:sub(i,i) == ' ' then
            res = res .. '\n'
            k = 1
         end
      elseif not color then
         k = k + 1
      end
      if str:sub(i,i) == '\n' then
         k = 1
      end
   end
   return res
end

function dok.stylize(html, package)
   local styled = html
   -- (0) useless white space
   styled = styled:gsub('^%s+','')
   -- (1) function title
   styled = style.banner .. '\n' .. styled
   styled = styled:gsub('<a.-name="(.-)">.-</a>%s*', function(title) return style.title .. title:upper() .. style.none .. '\n' end)
   -- (2) lists
   styled = styled:gsub('<ul>(.+)</ul>', function(list) 
                                            return list:gsub('<li>%s*(.-)%s*</li>%s*', style.list .. '%1\n')
                                         end)
   -- (3) code
   styled = styled:gsub('%s*<code>%s*(.-)%s*</code>%s*', style.code .. ' %1 ' .. style.none)
   -- (4) pre
   styled = styled:gsub('<pre.->(.-)</pre>', style.pre .. '%1' .. style.none)
   -- (5) formatting
   styled = styled:gsub('<em>(.-)</em>', style.em .. '%1' .. style.none)
   -- (6) links
   styled = styled:gsub('<a.->(.-)</a>', style.none .. '%1' .. style.none)
   -- (7) images
   styled = styled:gsub('<img.-src="(.-)".->%s*', 
                         style.img .. 'image: file://' 
                         .. paths.concat(paths.install_dokmedia,package,'%1')
                         .. style.none .. '\n')
   -- (-) paragraphs
   styled = styled:gsub('<p>', '')
   -- (-) special chars
   styled = uncleanText(styled)
   -- (-) max columns
   styled = maxcols(styled)
   -- (-) conclude
   styled = styled:gsub('%s*$','')
   styled = styled .. '\n' .. style.banner
   return styled
end

function dok.html2funcs(html, package)
   local funcs = {}
   local next = html:gfind('<div class="level%d">(.-)</div>')
   for body in next do
      local func = body:gfind('<a name="' .. package .. '%.(.-)">.-</a>')()
      if func then
         funcs[func] = dok.stylize(body, package)
      end
   end
   return funcs
end

function dok.refresh()
   for package in paths.files(paths.install_dok) do
      if package ~= '.' and package ~= '..' and _G[package] then
         local dir = paths.concat(paths.install_dok, package)
         for file in paths.files(dir) do
            if file ~= '.' and file ~= '..' then
               local path = paths.concat(dir, file)
               local f = io.open(path)
               if f then
                  local content = f:read('*all')
                  local html = dok.dok2html(content)
                  local funcs = dok.html2funcs(html, package)
                  local pkg = _G[package]
                  if type(pkg) ~= 'table' then -- unsafe import, use protected import
                     pkg = _G._torchimport[package]
                  end
                  -- level 0: the package itself
                  dok.inline[pkg] = funcs['dok'] or funcs['reference.dok'] or funcs['overview.dok']
                  -- next levels
                  for key,symb in pairs(pkg) do
                     -- level 1: global functions and objects
                     local entry = (key):lower()
                     if funcs[entry] then
                        dok.inline[string2symbol(package .. '.' .. key)] = funcs[entry]
                     end
                     -- level 2: objects' methods
                     if type(pkg[key]) == 'table' then
                        local entries = {}
                        for k,v in pairs(pkg[key]) do
                           entries[k] = v
                        end
                        local mt = getmetatable(pkg[key]) or {}
                        for k,v in pairs(mt) do
                           entries[k] = v
                        end
                        for subkey,subsymb in pairs(entries) do
                           local entry = (key .. '.' .. subkey):lower()
                           if funcs[entry] then
                              dok.inline[string2symbol(package .. '.' .. key .. '.' .. subkey)] = funcs[entry]
                           end
                        end
                     end
                  end
               end
            end
         end
      end
   end
end

function dok.help(symbol)
   -- no symbol? global help
   if not symbol then
      print('help(symbol): get help on a specific symbol \n'
            .. 'or checkout the complete help:\n'
            .. style.link .. paths.concat(paths.install_html,'index.html')
            .. style.none)
      return
   end
   -- always refresh (takes time, but insures that 
   -- we generate help for all packages loaded)
   dok.refresh()
   local inline = dok.inline[symbol]
   if inline then
      print(inline)
   else
      print('undocumented symbol')
   end
end

help = dok.help
