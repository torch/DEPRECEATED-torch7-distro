print('now in dokindex')
for k,v in ipairs(arg) do
   print(k,v)
end

local doktemplate = arg[1]
local dokcurrentindex = arg[2]
local dokindex = arg[3]
local package = arg[4]
local section = arg[5]
local title = arg[6]
local rank = arg[7]

-- "${TORCH_DOK_HTML_TEMPLATE}"
-- "${TORCH_BINARY_DIR}/dokindex.txt"
-- "${TORCH_BINARY_DIR}/dok/index.txt"
-- "${package}"
-- "${section}"
-- "${title}"
-- "${rank}"

-- find out rank/package rank
local ranksec = tonumber(rank:match('^(%d+)%.')) or 111111
local rankpkg = tonumber(rank:match('(%d+)$')) or 111111

-- create sections
sections = {}

-- add new (given) section
sections[section] = sections[section] or {rank=ranksec, packages={}}
sections[section].packages[package] = {title=title, rank=rankpkg}
sections[section].rank = math.min(sections[section].rank, ranksec)

-- add existing sections
local f = io.open(dokcurrentindex)
if f then
   dofile(dokcurrentindex)
   f:close()
end

-- write all the stuff on disk so we can reload it easily
local f = io.open(dokcurrentindex, 'w')
for secname, section in pairs(sections) do
   f:write(string.format('sections["%s"] = {rank=%d, packages={}}\n', secname, section.rank))
   for pkgname, package in pairs(section.packages) do
      f:write(string.format('sections["%s"].packages["%s"] = {title="%s", rank=%d}\n', secname, pkgname, package.title, package.rank))
   end
end
f:close()

-- sort sections
local sortedsections = {}
for k,v in pairs(sections) do
   table.insert(sortedsections, {secname=k, rank=v.rank, packages=v.packages})
end
table.sort(sortedsections, function(sa, sb)
                              return sa.rank < sb.rank
                           end)

-- sort packages (for each section)
local txt = {}
for _,section in ipairs(sortedsections) do
   table.insert(txt, string.format('<h2>%s</h2>', section.secname))
   table.insert(txt, '<ul>\n')
   local sortedpackages = {}
   for k,v in pairs(section.packages) do
      table.insert(sortedpackages, {pkgname=k, title=v.title, rank=v.rank})      
   end
   table.sort(sortedpackages, function(sa, sb)
                                 return sa.rank < sb.rank
                              end)

   for _,package in ipairs(sortedpackages) do
      table.insert(txt, string.format('  <li><a href="%s/index.html">%s</a></li>', package.pkgname, package.title))
   end
   table.insert(txt, '</ul>')
end

-- write the html
local templatehtml = io.open(doktemplate):read('*all')

templatehtml = templatehtml:gsub('%%CONTENTS%%', table.concat(txt, '\n'))
templatehtml = templatehtml:gsub('%%TITLE%%', 'Torch7 Documentation')
templatehtml = templatehtml:gsub('%%TOC%%', '')

local f = io.open(dokindex, 'w')
f:write(templatehtml)
f:close()
