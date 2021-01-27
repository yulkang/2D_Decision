function v = conv_t_back(v, flt)
% v = conv_t_back(v, flt)
%
% Convolves v in the reverse time direction.
% Gives more weight to v in the earlier time, so that the 
% convolved area always sum to sum(flt), even when the filter is truncated at 0.
%
% See also: bml.math.conv_t

% 2016 (c) Yul Kang. hk2699 at columbia dot edu.

assert(isvector(flt)); % Otherwise, the results will be wrong!
flt    = flt(:);
% sumflt = sum(flt);
% cumflt = cumsum(flt);
% wt     = nan0(sumflt ./ cumflt);
siz    = size(v);
% 
% if isvector(v)
%     v  = v(:) .* wt(:);
% elseif ismatrix(v)
%     v  = bsxfun(@times, v, wt);
% end

v = flipud(conv_t(flipud(v), flt));
v = reshape(v, siz);