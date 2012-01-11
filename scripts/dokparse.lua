-- print('now in dokparse')
-- for k,v in ipairs(arg) do
--    print(k,v)
-- end

local dokutils = arg[1]
local doktemplate = arg[2]
local src = arg[3]
local title = arg[4]
local rootdir = arg[5]
local dokdst = arg[6]
local htmldst = arg[7]

dofile(dokutils)

local txt = io.open(src):read('*all')

local sections = dok.parseSection(txt)

local toc = {}
local function addtocsubsections(toc, section)
   table.insert(toc, string.format('<ul>'))
   for k,v in pairs(section.subsections) do
      table.insert(toc, string.format('<li><a href="#%s">%s</a></li>', dok.link2wikilink(v.title), v.title))
      if v.subsections and #v.subsections > 0 then
         addtocsubsections(toc, v)
      end
   end
   table.insert(toc, string.format('</ul>'))
end
addtocsubsections(toc, sections)
toc = table.concat(toc, '\n')

local navhome = '<a href="' .. rootdir .. '/index.html">Torch7 Documentation</a>'
navhome = navhome .. ' > <a href="index.html">' .. title .. '</a>'

local templatehtml = io.open(doktemplate):read('*all')
local txthtml = dok.dok2html(txt)
templatehtml = templatehtml:gsub('%%CONTENTS%%', txthtml)
--local title = src:gsub('^.*/', ''):gsub('%..-$', '')
templatehtml = templatehtml:gsub('%%TITLE%%', title)
templatehtml = templatehtml:gsub('%%NAVLINE%%',navhome)
templatehtml = templatehtml:gsub('%%TOC%%', toc)
templatehtml = templatehtml:gsub('%%LASTMODIFIED%%','Generated at ' .. os.date())
io.open(htmldst, 'w'):write(templatehtml)

io.open(dokdst, 'w'):write(txt)


-- all in html?