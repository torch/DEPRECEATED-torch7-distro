--------------------------------------------------------------------------------
-- search in help
-- that file defines all the tools and goodies to allow search
--------------------------------------------------------------------------------
dok.entries = {}

paths.install_dok = paths.concat(paths.install_html, '..', 'dok')
paths.install_dokmedia = paths.concat(paths.install_html, '..', 'dokmedia')

local function html2entries(html, package, file)
   local funcs = {}
   local next = html:gfind('<div.->\n<h%d><a.->%s+(.-)%s+</a></h%d><a.-></a>\n<a name="(.-)"></a>\n(.-)</div>')
   for title,link,body in next do
      link = package .. '/' .. file:gsub('.txt','.html') .. '#' .. link
      body = body:gsub('<img.->','')
      table.insert(dok.entries, {title, link, body})
   end
end

function dok.gensearch()
   for package in paths.files(paths.install_dok) do
      if package ~= '.' and package ~= '..' then
         local dir = paths.concat(paths.install_dok, package)
         for file in paths.files(dir) do
            if file ~= '.' and file ~= '..' then
               local path = paths.concat(dir, file)
               local f = io.open(path)
               if f then
                  local content = f:read('*all')
                  local html = dok.dok2html(content)
                  html2entries(html, package, file)
               end
            end
         end
      end
   end
end

function dok.installsearch()
   dok.gensearch()
   local vars = {}
   for i,entry in ipairs(dok.entries) do
      table.insert(vars, 's[' .. (i-1) .. '] = "' 
                   .. table.concat(entry, '^'):gsub('"','\\"'):gsub('\n',' ') .. '";')
   end
   local array = table.concat(vars, '\n')
   local f = paths.concat(paths.install_html, 'jse_form.js')
   local js = io.open(f):read('*all')
   js = js:gsub('// SEARCH_ARRAY //', array)
   local w = io.open(f,'w')
   w:write(js)
   w:close()
end
