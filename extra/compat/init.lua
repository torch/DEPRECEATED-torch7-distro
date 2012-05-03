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

package.preload.openmp = function()
                         require 'torch'
                         print("***** WARNING: the 'openmp' package is deprecated. Now fully integrated in torch *****")
                         openmp = torch
                         return torch
                      end
