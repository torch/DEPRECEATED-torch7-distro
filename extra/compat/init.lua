package.preload.lab = function()
                         require 'torch'
                         print("***** WARNING: the 'lab' package is deprecated. Please use 'torch' package instead *****")
                         lab = torch
                         return torch
                      end

package.preload.random = function()
                         require 'torch'
                         print("***** WARNING: the 'random' package is deprecated. Please use 'torch' package instead *****")
                         random = torch
                         return torch
                      end
