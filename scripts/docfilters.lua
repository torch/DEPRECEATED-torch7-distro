function filterSubSections(txt)
   if txt == '' then
      return ''
   else
      return '<div id="subsections">\n<h3>Subsections</h3>\n' .. txt .. '\n</div>'
   end
end
