-- print('now in dokindex')
-- for k,v in ipairs(arg) do
--    print(k,v)
-- end

local dokutils = arg[1]
local doktemplate = arg[2]
local dokcurrentindex = arg[3]
local dokindex = arg[4]
local htmlindex = arg[5]
local package = arg[6]
local section = arg[7]
local title = arg[8]
local rank = arg[9]

dofile(dokutils)

-- find out rank/package rank
local ranksec = tonumber(rank:match('^(%d+)%.')) or 111111
local rankpkg = tonumber(rank:match('(%d+)$')) or 111111

-- create sections
sections = {}

-- add existing sections
local f = io.open(dokcurrentindex)
if f then
   dofile(dokcurrentindex)
   f:close()
end

-- add new (given) section
sections[section] = sections[section] or {rank=ranksec, packages={}}
sections[section].packages[package] = {title=title, rank=rankpkg}
sections[section].rank = math.min(sections[section].rank, ranksec)

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
local txtdok = {}
for _,section in ipairs(sortedsections) do
   table.insert(txtdok, string.format('===== %s =====', section.secname))
   local sortedpackages = {}
   for k,v in pairs(section.packages) do
      table.insert(sortedpackages, {pkgname=k, title=v.title, rank=v.rank})      
   end
   table.sort(sortedpackages, function(sa, sb)
                                 return sa.rank < sb.rank
                              end)

   for _,package in ipairs(sortedpackages) do
      table.insert(txtdok, string.format('  * [[.:%s:|%s]]', package.pkgname, package.title))
   end
end

-- write the dok
table.insert(txtdok, '\n')
txtdok = table.concat(txtdok, '\n')
local f = io.open(dokindex, 'w')
f:write(txtdok)
f:close()

-- write the html
local templatehtml = io.open(doktemplate):read('*all')
txthtml = dok.dok2html(txtdok)
templatehtml = templatehtml:gsub('%%CONTENTS%%', txthtml)
templatehtml = templatehtml:gsub('%%TITLE%%', 'Torch7 Documentation')
templatehtml = templatehtml:gsub('%%TOC%%', '')
local f = io.open(htmlindex, 'w')
f:write(templatehtml)
f:close()
