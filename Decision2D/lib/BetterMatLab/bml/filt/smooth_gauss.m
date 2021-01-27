function v = smooth_gauss(v, sigma)
% v = smooth_gauss(v, sigma)
%
% Works along the first dimension.
%
% sigma: in the unit of elements = sigma_in_t / dt_per_element

if sigma == 0
    % Skip smoothing.
    return;
end

f = filt_gauss(sigma);
if isvector(v)
    v = conv(v,f,'same');
elseif ismatrix(v)
    v = conv2(v,f(:),'same');
else
    v = convn(v,f(:),'same');
end
