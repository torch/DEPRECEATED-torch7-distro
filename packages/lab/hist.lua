
-- rudimentary histogram diplay on the command line.
local function histc__tostring(h, barHeight)
   barHeight = barHeight or 10
   local m =  h.max.nb + h.max.nb * 0.1
   local tl = torch.Tensor(#h):fill(0)
   local incr = (m/barHeight)
   local top = '+-+'
   local bar = '| |'
   local blank = '   '
   local str = 'nsamples|'
   str = str .. string.format('  minbin: %d val %2.2f maxbin: %d val %2.2f\n',
                          h.min.nb,h.min.val,
                          h.max.nb,h.max.val)
   str = str .. '--------+\n'
   for i = 1,barHeight do
      -- y axis
      if i%1==0 then
         str = str .. string.format('%7d |',m)
      end
      for j = 1,#h do
         if tl[j] == 1 then
            str = str .. bar
         elseif h[j].nb > m then
            tl[j] = 1
            str = str .. top
         else
            str = str .. blank
         end
      end
      str = str .. '\n'
      m = m - incr
   end
   -- x axis
   str = str .. string.format('--------+-^-')
   for j = 1,#h,2 do
      str = str .. string.format('----^-')
   end
   str = str .. '\ncenters '
   for j = 1,#h,2 do
      if h[j].val < 0 then
         str = str .. '-'
      else
         str = str .. ' '
      end
      str = str .. string.format('%2.2f ',math.abs(h[j].val))
   end
   if #h%2==0 then
      if h[#h].val < 0 then
         str = str .. '-'
      else
         str = str .. ' '
      end
      str = str .. string.format('%2.2f ',math.abs(h[#h].val))
   end
   return str
end

-- a simple function that computes the histogram of a tensor
function lab.histc(...)
   -- get args
   local args = {...}
   local tensor = args[1] or error('usage: lab.histc (tensor [, nBins] [, min] [, max]')
   local bins = args[2] or 10
   local min = args[3] or tensor:min()
   local max = args[4] or tensor:max()

   -- compute histogram
   local hist = lab.zeros(bins)
   local ten = torch.Tensor(tensor:nElement()):copy(tensor)
   ten:add(-min):div(max-min):mul(bins - 1e-6):floor():add(1)
   ten:apply(function (x)
                -- need to treat edge cases if we allow arbitrary
                -- min and max args.
                if x < 1 then
                   hist[1] = hist[1] + 1
                elseif x > bins then
                   hist[bins] = hist[bins] + 1
                else
                   hist[x] = hist[x] + 1
                end
                return x
             end)

   -- cleanup hist
   local cleanhist = {}
   cleanhist.raw = hist
   local _,mx = lab.max(cleanhist.raw)
   local _,mn = lab.min(cleanhist.raw)
   cleanhist.bins = bins
   cleanhist.binwidth = (max-min)/bins
   for i = 1,bins do
      cleanhist[i] = {}
      cleanhist[i].val = min + (i-0.5)*cleanhist.binwidth
      cleanhist[i].nb = hist[i]
   end
   cleanhist.max = cleanhist[mx[1]]
   cleanhist.min = cleanhist[mn[1]]

   -- print function
   setmetatable(cleanhist, {__tostring=histc__tostring})
   return cleanhist
end

-- complete function: compute hist and display it
function lab.hist(tensor,bins,min,max)
   local h = lab.histc(tensor,bins,min,max)
   local x_axis = torch.Tensor(#h)
   for i = 1,#h do
      x_axis[i] = h[i].val
   end
   lab.bar(x_axis, h.raw)
   return h
end
