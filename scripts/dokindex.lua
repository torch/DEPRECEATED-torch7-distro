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

local templatehtml = io.open(doktemplate):read('*all')


local ranksec = tonumber(rank:match('^(%d+)%.')) or 111111
local rankpkg = tonumber(rank:match('(%d+)$')) or 111111

sections = {}
sections[section] = sections[section] or {rank=ranksec, packages={}}
sections[section].packages[package] = {title=title, rank=rankpkg}
sections[section].rank = math.min(sections[section].rank, ranksec)

local f = io.open(dokcurrentindex)
if f then
   dofile(dokcurrentindex)
   f:close()
end

local f = io.open(dokcurrentindex, 'w')
for secname, section in pairs(sections) do
   f:write(string.format('sections["%s"] = {rank=%d, packages={}}\n', secname, section.rank))
   for pkgname, package in pairs(section.packages) do
      f:write(string.format('sections["%s"].packages["%s"] = {title="%s", rank=%d}\n', secname, pkgname, package.title, package.rank))
   end
end
f:close()

