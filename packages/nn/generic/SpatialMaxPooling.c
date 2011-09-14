#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

static int nn_(SpatialMaxPooling_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[2] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  // sizes
  long nslices = input->size[0];
  long iheight = input->size[1];
  long iwidth = input->size[2];
  long oheight = (iheight - kH) / dH + 1;
  long owidth = (iwidth - kW) / dW + 1;

  // get contiguous input
  input = THTensor_(newContiguous)(input);

  // resize output
  THTensor_(resize3d)(output, nslices, oheight, owidth);

  // indices will contain i,j locatyions for each output point
  THTensor_(resize4d)(indices, 2, nslices, oheight, owidth);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *indices_data = THTensor_(data)(indices);

  // compute max pooling for each input slice
  long k;
  for (k = 0; k < nslices; k++) {
    // pointers to slices
    real *input_p = input_data + k*iwidth*iheight;
    real *output_p = output_data + k*owidth*oheight;
    real *indx_p = indices_data + k*owidth*oheight;
    real *indy_p = indices_data + (k+nslices)*owidth*oheight;

    // loop over output
    int i,j;
    for(i = 0; i < oheight; i++) {
      for(j = 0; j < owidth; j++) {
        // local pointers
        real *ip = input_p + i*iwidth*dH + j*dW;
        real *op = output_p + i*owidth + j;
        real *indxp = indx_p + i*owidth + j;
        real *indyp = indy_p + i*owidth + j;

        // compute local max:
	long maxindex = -1;
	real maxval = -THInf;
	long tcntr = 0;
        int x,y;
        for(y = 0; y < kH; y++) {
          for(x = 0; x < kW; x++) {
            real val = *(ip + y*iwidth + x);
            if (val > maxval) {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

        // set output to local max
        *op = maxval;

        // store location of max (x,y)
        *indxp = (int)(maxindex / dW)+1;
        *indyp = (maxindex % dW) +1;
      }
    }
  }

  // cleanup
  THTensor_(free)(input);

  return 1;
}

static int nn_(SpatialMaxPooling_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradOutputPlane, *gradInputPlane, *unfoldedGradInputPlane, *gradLocalInput;
  int k,i,j;

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  gradInputPlane = THTensor_(new)();
  gradOutputPlane = THTensor_(new)();
  gradLocalInput = THTensor_(new)();
  unfoldedGradInputPlane = THTensor_(new)();

  for (k = 0; k < input->size[0]; k++)
  {
    /* get input and output plane */
    THTensor_(select)(gradOutputPlane, gradOutput, 0, k);
    THTensor_(select)(gradInputPlane, gradInput, 0, k);

    /* Unfold input to get each local window */
    THTensor_(unfold)(unfoldedGradInputPlane, gradInputPlane, 0, kH, dH);
    THTensor_(unfold)(unfoldedGradInputPlane, NULL,           1, kW, dW);

    /* Calculate max points */
    for(i = 0; i < gradOutputPlane->size[0]; i++) {
      for(j = 0; j < gradOutputPlane->size[1]; j++) {
	THTensor_(select)(gradLocalInput, unfoldedGradInputPlane,0,i);
	THTensor_(select)(gradLocalInput, NULL,                  0,j);
	long maxi = THTensor_(get4d)(indices,0,k,i,j)-1;
	long maxj = THTensor_(get4d)(indices,1,k,i,j)-1;
	double gi = THTensor_(get2d)(gradLocalInput,maxi,maxj)+THTensor_(get2d)(gradOutputPlane,i,j);
	THTensor_(set2d)(gradLocalInput,maxi,maxj,gi);
      }
    }
  }

  /* Cleanup */
  THTensor_(free)(gradInputPlane);
  THTensor_(free)(gradOutputPlane);
  THTensor_(free)(unfoldedGradInputPlane);
  THTensor_(free)(gradLocalInput);

  return 1;
}

static const struct luaL_Reg nn_(SpatialMaxPooling__) [] = {
  {"SpatialMaxPooling_forward", nn_(SpatialMaxPooling_forward)},
  {"SpatialMaxPooling_backward", nn_(SpatialMaxPooling_backward)},
  {NULL, NULL}
};

static void nn_(SpatialMaxPooling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
