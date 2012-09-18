Fs = 1000;
T = 1/Fs; 
L = 1000;
t = torch.range(0,L-1)*T; 

-- 0.7 at 50Hz and 1 and 120Hz
x = torch.sin(t*2*math.pi*50)*0.7 + torch.sin(t*2*math.pi*120);

-- add noise
y = x + torch.randn(t:nElement())*2;
gnuplot.figure()
gnuplot.plot(t[{ {1,50} }]*Fs, y[{ {1,50} }],'-')
gnuplot.title('Original Signal')

-- do FFT
NFFT=1024
Y = torch.fft(y,NFFT)/L;
f = torch.linspace(0,1,NFFT/2+1)*Fs/2;

-- Torch7 do not have imaginary representation
Yabs = torch.cmul(Y,Y)
Yabs = torch.sum( Yabs,2)
Yabs = torch.squeeze( Yabs:sqrt() )
gnuplot.figure()
gnuplot.plot(f,Yabs[{ {1,NFFT/2+1} }]*2,'-')
gnuplot.title('FFT')

yy = torch.ifft( Y )
gnuplot.figure()
gnuplot.plot({'Original Signal',y[{ {1,50} }],'+'},{'IFFT',yy[{ {1,50} }]*L,'-'})
gnuplot.title('Inverse FFT and Original Signal')

