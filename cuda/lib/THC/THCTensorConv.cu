#include "THCTensorConv.h"
#include "THCGeneral.h"

/*
 * Description:
 *   This code provides convolutions and xcorrelations that are API compatible with
 *   the ones in THLabConv.
 *
 * History:
 *   July 22, 2011, 8:38PM   -  Clement Farabet  -  All Valid/Full/XCORR/CONV implemented
 *   July 22, 2011, 4:00PM   -  Clement Farabet  -  Rewrote for loop to insure memory coalescing
 *   July 21, 2011, 11:21PM  -  Clement Farabet  -  Creation, based conv2d routine
 */

#define CUDA_SHARED_MEM_SIZE (4*1024-32) // this is given by nVidia: max shared mem per block
#define CUDA_ORDER_FOR_COALESCING        // this reorders memory reads to enable coalescing

/*
 * Description:
 *   base conv2D routine: 3D input, 3D output, 4D kernel
 *
 *   - all chunks of data should be contiguous
 *   - the swapkernel flag can be used to generate a conv2 instead of xcorr2
 *   - the templated kernel size is useful to generate code that's 2x faster
 *     but can be set to 0 to allow arbitrary kernel sizes
 */
template <bool swapkernel, int T_kernel_h, int T_kernel_w>
  __global__ void conv2generic(float *input, float *kernel, float *output,
                               int input_n, int input_h, int input_w,
                               int kernel_n, int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int patch_h, int patch_w)
{
  // output dimensions
  int output_h = (input_h - kernel_h) / stride_h + 1;
  int output_w = (input_w - kernel_w) / stride_w + 1;

  // xcorr or conv
  int koffset = swapkernel ? kernel_w*kernel_h-1 : 0;

  // generate offsets according to block/thread ids
#ifdef CUDA_ORDER_FOR_COALESCING
  int xx_start = threadIdx.x; // * patch_w;
  int xx_end = output_w;
  int xx_step = blockDim.x;
#else
  int xx_start = threadIdx.x * patch_w;
  int xx_end = xx_start + patch_w; if (xx_end > output_w) xx_end = output_w;
  int xx_step = 1;
#endif
  int yy_start = threadIdx.y * patch_h;
  int yy_end = yy_start + patch_h; if (yy_end > output_h) yy_end = output_h;
  int yy_step = 1;
  int oo_start = blockIdx.x;
  int oo_end = oo_start+1;
  int ii_start = 0;
  int ii_end = input_n;

  // iterators
  int oo, ii, xx, yy, kx, ky, kk;

  // do the kernels fit in shared mem ?
  if (input_n*kernel_w*kernel_h <= CUDA_SHARED_MEM_SIZE) {

    // put the kernel in shared memory
    __shared__ float shared_kernel[CUDA_SHARED_MEM_SIZE];

    // first thread of each block does the copy
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      for (kk = 0; kk < kernel_w*kernel_h*input_n; kk++) {
        shared_kernel[kk] = kernel[input_n*kernel_w*kernel_h*blockIdx.x + kk];
      }
    }

    // sync threads
    __syncthreads();

    // templated kernel size
    if ((T_kernel_w > 0) && (T_kernel_h > 0)) {
      // unrolled convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for(ii = ii_start; ii < ii_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
              float *kernel_p = shared_kernel + ii * kernel_w * kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    } else {
      // default convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for(ii = ii_start; ii < ii_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
              float *kernel_p = shared_kernel + ii * kernel_w * kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
                for(ky = 0; ky < kernel_h; ky++) {
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
                for(ky = 0; ky < kernel_h; ky++) {
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    }

  } else { // not enough shared mem for kernels, simply stream them

    // convolution loop
    for(oo = oo_start; oo < oo_end; oo++) {
      for(ii = ii_start; ii < ii_end; ii++) {
        for(yy = yy_start; yy < yy_end; yy+=yy_step) {
          for(xx = xx_start; xx < xx_end; xx+=xx_step) {
            // Dot product in two dimensions... (between input image and the mask)
            float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
            float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
            float *kernel_p = kernel + (oo * input_n + ii) * kernel_w * kernel_h + koffset;
            float sum = 0;
            for(ky = 0; ky < kernel_h; ky++) {
              for(kx = 0; kx < kernel_w; kx++) {
                if (swapkernel) sum += input_p[kx]*(*kernel_p--);
                else sum += input_p[kx]*(*kernel_p++);
              }
              input_p += input_w;
            }
            *output_p += sum;
          }
        }
      }
    }
  }
}

/*
 * Description:
 *   base conv2D routine with reversed stride: 3D input, 4D output, 3D kernel
 *   this is useful for computing gradients with respect to kernels, where:
 *   input=input, kernel=gradOutput, output=gradWeight
 *
 *   - all chunks of data should be contiguous
 *   - the swapkernel flag can be used to generate a conv2 instead of xcorr2
 */
__global__ void conv2genericrev(float *input, float *kernel, float *output,
                                int input_n, int input_h, int input_w,
                                int kernel_n, int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int patch_h, int patch_w)
{
  // output dimensions
  int output_h = input_h - (kernel_h - 1) * stride_h;
  int output_w = input_w - (kernel_w - 1) * stride_w;

  // generate offsets according to block/thread ids
#ifdef CUDA_ORDER_FOR_COALESCING
  int xx_start = threadIdx.x; // * patch_w;
  int xx_end = output_w;
  int xx_step = blockDim.x;
#else
  int xx_start = threadIdx.x * patch_w;
  int xx_end = xx_start + patch_w; if (xx_end > output_w) xx_end = output_w;
  int xx_step = 1;
#endif
  int yy_start = threadIdx.y * patch_h;
  int yy_end = yy_start + patch_h; if (yy_end > output_h) yy_end = output_h;
  int yy_step = 1;

  int kk_start = blockIdx.x;
  int kk_end = kk_start+1;

  int ii_start = blockIdx.y;
  int ii_end = ii_start+1;

  // iterators
  int kk, ii, xx, yy, kx, ky;

  // convolution loop
  for(kk = kk_start; kk < kk_end; kk++) {
    for(ii = ii_start; ii < ii_end; ii++) {
      for(yy = yy_start; yy < yy_end; yy+=yy_step) {
        for(xx = xx_start; xx < xx_end; xx+=xx_step) {
          // Dot product in two dimensions... (between input image and kernel)
          float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
          float *kernel_p = kernel + kk*kernel_w*kernel_h;
          float *output_p = output + (kk * input_n + ii)*output_h*output_w + yy*output_w + xx;
          float sum = 0;
          for(ky = 0; ky < kernel_h; ky++) {
            for(kx = 0; kx < kernel_w; kx++) {
              sum += input_p[kx]*(*kernel_p++);
            }
            input_p += input_w;
          }
          *output_p += sum;
        }
      }
    }
  }
}

/*
 * API-compatible with THRealTensor_conv2Dmv
 * 3D input, 4D kernel, 3D output
 * matrix vector product like: y <- Ax + beta*y
 */
TH_API void THCudaTensor_conv2Dmv(THCudaTensor *output, float beta, THCudaTensor *input,
                                  THCudaTensor *kernel, long srow, long scol, const char *type)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;

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
    // output dims
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;

    // create a zero-padded input
    long nInputRowsPadded = (nOutputRows - 1) * srow + nKernelRows;
    long nInputColsPadded = (nOutputCols - 1) * scol + nKernelCols;
    THCudaTensor *inputP = THCudaTensor_newWithSize3d(nInputPlane,
                                                      nInputRowsPadded,
                                                      nInputColsPadded);
    THCudaTensor_zero(inputP);

    THCudaTensor *centered = THCudaTensor_new();
    THCudaTensor_narrow(centered, inputP, 2, nKernelCols-1, nInputCols);
    THCudaTensor_narrow(centered, NULL, 1, nKernelRows-1, nInputRows);
    THCudaTensor_copy(centered, input);
    THCudaTensor_free(centered);

    // remap input to newly created tensor
    THCudaTensor_free(input);
    input = inputP;
    nInputRows = nInputRowsPadded;
    nInputCols = nInputColsPadded;

  } else { // 'v'
    // output dims
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

  // auto compute nb of blocks and threads:
  // this might look a bit arbitrary, but it's not. We want to be in a sweet spot,
  // where most of the time we'll have 16x16 threads per output map, that is 256 threads
  // dealing with each map, and one block per map.
  // example: if we have 16 output maps, each being 256x256, we'll have 16 blocks
  // with 256 threads, each thread processing a 16x16 suboutput.
  // call me if it's not clear ;-)
  int patch_w = (int)(pow(2, ceil(log2((float)nOutputCols))) / 16);
  if (patch_w < 2) patch_w = 2;
  else if (patch_w > 32) patch_w = 32;
  int patch_h = patch_w;
  dim3 blocks(nOutputPlane);
  dim3 threads(nOutputCols / patch_w, nOutputRows / patch_h);
  if ((nOutputRows % patch_h) != 0) threads.y++;
  if ((nOutputCols % patch_w) != 0) threads.x++;

  // convolution: xcorr2 or conv2
  if (type[1] == 'x') {
    if ((nKernelCols == 3) && (nKernelRows == 3))
      conv2generic <false, 3, 3> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                        nInputPlane, nInputRows, nInputCols,
                                                        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                        srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 5) && (nKernelRows == 5))
      conv2generic <false, 5, 5> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                        nInputPlane, nInputRows, nInputCols,
                                                        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                        srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 7) && (nKernelRows == 7))
      conv2generic <false, 7, 7> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                        nInputPlane, nInputRows, nInputCols,
                                                        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                        srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 9) && (nKernelRows == 9))
      conv2generic <false, 9, 9> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                        nInputPlane, nInputRows, nInputCols,
                                                        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                        srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 11) && (nKernelRows == 11))
      conv2generic <false, 11, 11> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                          nInputPlane, nInputRows, nInputCols,
                                                          nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                          srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 13) && (nKernelRows == 13))
      conv2generic <false, 13, 13> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                          nInputPlane, nInputRows, nInputCols,
                                                          nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                          srow, scol, patch_h, patch_w);
    else
      conv2generic <false, 0 , 0> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                         nInputPlane, nInputRows, nInputCols,
                                                         nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                         srow, scol, patch_h, patch_w);
  } else { // 'c'
    if ((nKernelCols == 3) && (nKernelRows == 3))
      conv2generic <true, 3, 3> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                       nInputPlane, nInputRows, nInputCols,
                                                       nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                       srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 5) && (nKernelRows == 5))
      conv2generic <true, 5, 5> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                       nInputPlane, nInputRows, nInputCols,
                                                       nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                       srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 7) && (nKernelRows == 7))
      conv2generic <true, 7, 7> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                       nInputPlane, nInputRows, nInputCols,
                                                       nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                       srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 9) && (nKernelRows == 9))
      conv2generic <true, 9, 9> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                       nInputPlane, nInputRows, nInputCols,
                                                       nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                       srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 11) && (nKernelRows == 11))
      conv2generic <true, 11, 11> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                         nInputPlane, nInputRows, nInputCols,
                                                         nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                         srow, scol, patch_h, patch_w);
    else if ((nKernelCols == 13) && (nKernelRows == 13))
      conv2generic <true, 13, 13> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                         nInputPlane, nInputRows, nInputCols,
                                                         nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                         srow, scol, patch_h, patch_w);
    else
      conv2generic <true, 0 , 0> <<<blocks, threads>>> (input_data, weight_data, output_data,
                                                        nInputPlane, nInputRows, nInputCols,
                                                        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,
                                                        srow, scol, patch_h, patch_w);
  }

  // sync
  cudaThreadSynchronize();

  // clean up
  THCudaTensor_free(input);
  THCudaTensor_free(kernel);

  // check potential errors
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
}

/*
 * API-compatible with THRealTensor_conv2DRevger
 * 3D input, 3D kernel, 4D output
 * like rank1 update
 * A <- xx' + beta*A
 * for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for
 * calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
 */
TH_API void THCudaTensor_conv2DRevger(THCudaTensor *output, float beta, THCudaTensor *input,
                                      THCudaTensor *kernel, long srow, long scol)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputRows, nOutputCols;

  THArgCheck(input->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(kernel->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THCudaTensor_newContiguous(input);
  kernel = THCudaTensor_newContiguous(kernel);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2,
             "conv2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  long nelem = THCudaTensor_nElement(output);
  THCudaTensor_resize4d(output, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THCudaTensor_nElement(output)) {
    THCudaTensor_zero(output);
  } else if (beta != 1) {
    THCudaTensor_mul(output, beta);
  }

  float *input_data = THCudaTensor_data(input);
  float *kernel_data = THCudaTensor_data(kernel);
  float *output_data = THCudaTensor_data(output);

  // auto compute nb of blocks and threads:
  // this might look a bit arbitrary, but it's not. We want to be in a sweet spot,
  // where most of the time we'll have 16x16 threads per output map, that is 256 threads
  // dealing with each map, and one block per map.
  // example: if we have 16 output maps, each being 256x256, we'll have 16 blocks
  // with 256 threads, each thread processing a 16x16 suboutput.
  // call me if it's not clear ;-)
  int patch_w = (int)(pow(2, ceil(log2((float)nOutputCols))) / 16);
  if (patch_w < 2) patch_w = 2;
  else if (patch_w > 32) patch_w = 32;
  int patch_h = patch_w;
  dim3 blocks(nKernelPlane, nInputPlane);
  dim3 threads(nOutputCols / patch_w, nOutputRows / patch_h);
  if ((nOutputRows % patch_h) != 0) threads.y++;
  if ((nOutputCols % patch_w) != 0) threads.x++;

  // compute rev conv
  conv2genericrev <<<blocks, threads>>> (input_data, kernel_data, output_data,
                                         nInputPlane, nInputRows, nInputCols,
                                         nKernelPlane, nKernelRows, nKernelCols,
                                         srow, scol, patch_h, patch_w);

  // clean up
  THCudaTensor_free(input);
  THCudaTensor_free(kernel);
}
