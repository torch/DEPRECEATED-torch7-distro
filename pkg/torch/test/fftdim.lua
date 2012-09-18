Fs = 1000;
T = 1/Fs; 
L = 1000;
t = torch.range(0,L-1)*T; 

gnuplot.closeall()

-- 0.7 at 50Hz and 1 and 120Hz
x = torch.sin(t*2*math.pi*50)*0.7 + torch.sin(t*2*math.pi*120);

-- add noise
nsig = 3
y = torch.zeros(nsig,t:nElement())
for i=1,nsig do
   y[i] = x + torch.randn(t:nElement())*2;
end
gnuplot.figure(1)
plt = {}
for i=1,nsig do
   plt[i] = {'signal ' .. tostring(i),t[{ {1,50} }]*Fs, y[{ i , {1,50} }],'-'}
end
gnuplot.plot(plt)
gnuplot.title('Original Signals')

-- do FFT
NFFT=1024
Y = torch.fft(y,NFFT)/L;
f = torch.linspace(0,1,NFFT/2+1)*Fs/2;

-- Torch7 do not have imaginary representation
Yabs = torch.cmul(Y,Y)
Yabs = torch.sum( Yabs,Y:dim())
Yabs = torch.squeeze( Yabs:sqrt() )
plt = {}
for i=1,nsig do
   plt[i] = {'fft ' .. tostring(i),f,Yabs[{ i , {1,NFFT/2+1} }]*2,'-'}
end
gnuplot.figure(2)
gnuplot.plot(plt)
gnuplot.title('FFTs')

yy = torch.ifft( Y )
plt = {}
for i=1,nsig*2,2 do
   plt[i]   = {'Original Signal ' .. i , y[{ (i-1)/2+1 , {1,50} }],'+'}
   plt[i+1] = {'IFFT ' .. i , yy[{ (i-1)/2+1 , {1,50} }]*L,'-'}
end
gnuplot.figure(3)
gnuplot.plot(plt)
gnuplot.title('Inverse FFT and Original Signal')

