print('now in dokparse')
for k,v in ipairs(arg) do
   print(k,v)
end

local dokutils = arg[1]
local doktemplate = arg[2]
local src = arg[3]
local dokdst = arg[4]
local htmldst = arg[5]

dofile(dokutils)

local txt = io.open(src):read('*all')

local sections = dok.parseSection(txt)
for k,v in pairs(sections) do
   print(k,v)
end
for k,v in pairs(sections.subsections) do
   print(k,v.title)
end

local templatehtml = io.open(doktemplate):read('*all')
local txthtml = dok.dok2html(txt)
templatehtml = templatehtml:gsub('%%CONTENTS%%', txthtml)
io.open(htmldst, 'w'):write(templatehtml)

io.open(dokdst, 'w'):write(txt)


-- all in html?
