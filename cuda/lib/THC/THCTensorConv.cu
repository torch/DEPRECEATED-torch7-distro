#include "THCTensorConv.h"
#include "THCGeneral.h"

/*
  Base xcorr2 routine: 3D input, 3D output, 4D kernel
  All chunks of data should be contiguous
 */
__global__ void validXCorr2ptr(float *input, float *kernel, float *output,
                               long input_n, long input_h, long input_w, 
                               long kernel_n, long kernel_h, long kernel_w,
                               long stride_h, long stride_w)
{
  // output dimensions
  long output_h = (input_h - kernel_h) / stride_h + 1;
  long output_w = (input_w - kernel_w) / stride_w + 1;
  long output_n = kernel_n / input_n;

  // pre-load B_Y pixels from B_Y*filtersPerThread filters
  //__shared__ float shFilters[B_Y*channelCache][B_Y*filtersPerThread];

  // pre-load B_Y pixels from B_X*imgsPerThread images
  //__shared__ float shImages[B_Y*channelCache][B_X*imgsPerThread];

  // generate offsets according to block/thread ids
  long yy_start = threadIdx.y * 4; // 4 lines per thread
  long yy_end = yy_start + 4; if (yy_end > output_h) yy_end = output_h;
  long xx_start = 0;
  long xx_end = output_w;

  // convolution loop
  long oo, ii, xx, yy;
  for(oo = 0; oo < output_n; oo++) {
    for(ii = 0; ii < input_n; ii++) {
      for(yy = yy_start; yy < yy_end; yy++) {
        for(xx = xx_start; xx < xx_end; xx++) {
          // Dot product in two dimensions... (between input image and the mask)
          float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
          float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
          float *kernel_p = kernel + oo*output_n + ii;
          float sum = 0;
          long kx, ky;
          for(ky = 0; ky < kernel_h; ky++) {
            for(kx = 0; kx < kernel_w; kx++) {
              sum += input_p[kx]*kernel_p[kx];
            }
            input_p += input_w; /* next input line */
            kernel_p += kernel_w; /* next mask line */
          }
          *output_p += sum;
        }
      }
    }
  }
}

/*
  3D input, 4D kernel, 3D output
  matrix vector product like
  y <- Ax + beta*y
*/
TH_API void THCudaTensor_conv2Dmv(THCudaTensor *output, float beta, THCudaTensor *input,
                                  THCudaTensor *kernel, long srow, long scol, const char *type)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;

  THArgCheck(output->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(kernel->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(type[0] == 'v' || type[0] == 'f', 7, "type of convolution can 'v' or 'f'");
  THArgCheck(type[1] == 'c' || type[1] == 'x', 7, "type of convolution can 'x' or 'c'");

  input = THCudaTensor_newContiguous(input);
  kernel = THCudaTensor_newContiguous(kernel);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2,
              "conv2Dmv : Input image is smaller than kernel");

  if (*type == 'f') {
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;
  } else { // valid
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  long nelem = THCudaTensor_nElement(output);
  THCudaTensor_resize3d(output, nOutputPlane, nOutputRows, nOutputCols);

  if (beta == 0 || nelem != THCudaTensor_nElement(output)) {
    THCudaTensor_zero(output);
  } else if (beta != 1) {
    THCudaTensor_mul(output, beta);
  }

  float *input_data = THCudaTensor_data(input);
  float *weight_data = THCudaTensor_data(kernel);
  float *output_data = THCudaTensor_data(output);

  // auto compute nb of blocks and threads
  dim3 blocks(nOutputPlane);
  dim3 threads(1, nOutputRows / 4);
  if ((nOutputRows % 4) != 0) threads.y++;

  // convolution: input with kernel, 4 modes (full, valid, xcorr2 or conv2)
  if (type[0] == 'f') {

    if (type[1] == 'x') {
      THError("full xcorr2 not implemented yet");
    } else {
      THError("full conv2 not implemented yet");
    }

  } else { // 'v'

    if (type[1] == 'x') {
      validXCorr2ptr <<<blocks, threads>>> (input_data, weight_data, output_data,
                                            nInputPlane, nInputRows, nInputCols,
                                            nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                            srow, scol);
    } else {
      THError("valid conv2 not implemented yet");
    }

  }

  // clean up
  THCudaTensor_free(input);
  THCudaTensor_free(kernel);
}

/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
*/
TH_API void THCudaTensor_conv2Dger(THCudaTensor *output, float beta, THCudaTensor *input,
                                   THCudaTensor *kernel, long srow, long scol, const char *type)
{

}

/*
  3D input, 3D kernel, 4D output
  like rank1 update
  A <- xx' + beta*A
  for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for
  calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
*/
TH_API void THCudaTensor_conv2DRevger(THCudaTensor *output, float beta, THCudaTensor *input,
                                      THCudaTensor *kernel, long srow, long scol)
{

}
