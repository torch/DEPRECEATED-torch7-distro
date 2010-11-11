if #arg < 6 or #arg > 8 then
   error("Usage: lua buildHelpIndex.lua <main input index.hlp file> <main output index.hlp file> <section output file> <package name> <package index.hlp file> <package index.hlp install file full path> [<package section> [<package rank>]]")
end

inHlpFile = arg[1]
outHlpFile = arg[2]
sectionsFile = arg[3]

packageName = arg[4]
packageIndexFile = arg[5]
packageInstallIndexFile = arg[6]

packageSection = arg[7] or "Misc"
packageRank = arg[8] or 10

--- read the title in the package help file (first line)
local f = io.open(packageIndexFile, 'r')
if not f then
   error('Cannot open help file <' .. packageIndexFile .. '> for reading')
end
local packageTitle = f:read('*line')
f:close()

--- create sections, and start with what we got just now
local sections = {}
sections[packageSection] = {}
sections[packageSection][packageName] = '   * [' .. packageName .. '] ' .. '[[' .. packageInstallIndexFile .. '][' .. packageTitle .. ']]'

--- read previously seen sections
local f = io.open(sectionsFile)
while f do
   local section_ = f:read('*line')
   local package_ = f:read('*line')
   local title_ = f:read('*line')
   if not title_ or not package_ or not section_ then
      f:close()
      break
   end
   if not sections[section_] then
      sections[section_] = {}
   end
   sections[section_][package_] = title_
end

--- open the "in" hlp file
f = io.open(inHlpFile)
local txt = ""
if f then
   txt = f:read('*all')
   f:close()
end
txt = txt .. '\n'

--- allright, now we can write the "out" hlp file
f = io.open(outHlpFile, 'w')
f:write(txt)
local sortedSections = {}
for section,_ in pairs(sections) do
   table.insert(sortedSections, section)
end
table.sort(sortedSections)
for _,section in ipairs(sortedSections) do
   local subsections = sections[section]
   f:write('\n---+ ' .. section .. '\n\n')
   local sortedPackages = {}
   for package,_ in pairs(subsections) do
      table.insert(sortedPackages, package)
   end
   table.sort(sortedPackages)
   for _,package in pairs(sortedPackages) do
      title = subsections[package]
      f:write(title .. '\n')
   end
end
f:close()

--- and we finally write the sections file
f = io.open(sectionsFile, 'w')
for section,subsections in pairs(sections) do
   for package,title in pairs(subsections) do
      f:write(section .. '\n')
      f:write(package .. '\n')
      f:write(title .. '\n')
   end
end
f:close()
