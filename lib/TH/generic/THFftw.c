#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THFftw.c"
#else

/* 1-D FFT */
void THFftw_(fft)(real *r, real *x, long n)
{
#ifdef USE_FFTW
#if defined(TH_REAL_IS_DOUBLE)
  fftw_complex *out = (fftw_complex*)r;
  fftw_plan p = fftw_plan_dft_r2c_1d(n, x, out, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
#else
  fftwf_complex *out = (fftwf_complex*)r;
  fftwf_plan p = fftwf_plan_dft_r2c_1d(n, x, out, FFTW_ESTIMATE);
  fftwf_execute(p);
  fftwf_destroy_plan(p);
#endif
#else
  THError("fft : FFTW Library was not found in compile time\n");
#endif
}
/* 1-D IFFT */
void THFftw_(ifft)(real *r, real *x, long n)
{
#ifdef USE_FFTW
#if defined(TH_REAL_IS_DOUBLE)
  fftw_complex *in = (fftw_complex*)x;
  fftw_plan p = fftw_plan_dft_c2r_1d(n, in, r, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
#else
  fftwf_complex *in = (fftwf_complex*)x;
  fftwf_plan p = fftwf_plan_dft_c2r_1d(n, in, r, FFTW_ESTIMATE);
  fftwf_execute(p);
  fftwf_destroy_plan(p);
#endif
#else
  THError("ifft : FFTW Library was not found in compile time\n");
#endif
}

#endif
