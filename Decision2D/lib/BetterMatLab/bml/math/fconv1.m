function y = fconv1(x, h)
% FCONV1 Vectorized fast convolution on the first dimension
%
% y = FCONV1(x, h) convolves x and h, column by column
%
% x: input array
% h: input array
% 
% See also: CONV, FCONV

% FCONV Coded by: Stephen G. McGovern, 2003-2004.
% 2013 Yul Kang: Added buffering to make it faster
% 2016 Yul Kang: Allowed column-by-column convolution

persistent h0 H

if isvector(x) && isvector(h)
    y = fconv(x, h);
    return;
end

Ly=size(x,1)+size(h,1)-1;  % 
Ly2=pow2(nextpow2(Ly));    % Find smallest power of 2 that is > Ly
X=fft(x, Ly2);		   % Fast Fourier transform

if ~isequal(h0, h)
    h0 = h;
    H = fft(h, Ly2);	           % Fast Fourier transform
end

if isequal(size(X), size(H))
    Y=X.*H;
else
    Y=bsxfun(@times, X, H);
end
y=real(ifft(Y, Ly2));      % Inverse fast Fourier transform
y=y(1:1:Ly,:);               % Take just the first N elements
% y=y/max(abs(y));           % Normalize the output