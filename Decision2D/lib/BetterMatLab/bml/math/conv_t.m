function v = conv_t(a, b, varargin)
% CONV_T  Convolves two time series starting from t=0
%
% Gives the first part with the same length as the 1st argument.
%
% v = conv_t(a, vec)
%
% a: a vector or an array. 
%    if an array, convolution works on the first dimension.
% b: a vector or an array.
%    If an array,
%    - dimensions other than the first should match with a's.
%    - convolution is done column-by-column.
%
% OPTION:
% 'to_use_fconv', false
% 'len', size(a,1) % result's length.
% 'ix0', 1 % ignore ix0-1 elements at the beginning.
%
% Note: when convolving back in time 
%       (to find original distribution from a delayed data),
%       use conv_t_back to appropriately account for 
%       truncation of the filter for the early data.
% 
% See also: conv_t_back

% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'to_use_fconv', []
    'len', size(a,1)
    'ix0', 1
    });

len = S.len;
ix0 = S.ix0;

if isempty(S.to_use_fconv)
    S.to_use_fconv = ~isvector(b);
end

if isvector(a)
    if S.to_use_fconv
        v = fconv1(a,b);
    else
        v = conv(a,b);
    end
    v = v(1:length(a));
    
else % if ismatrix(a)
    siz = size(a);
    
    nCol = prod(siz(2:end));
    a = reshape(a, size(a,1), []);
    
    if isvector(b)
        b = b(:);
    else
        b = reshape(b, size(b,1), []);
    end
    v = zeros(size(a,1) + size(b,1) - 1, nCol);
    
    if S.to_use_fconv
        v = fconv1(a, b);
    else
        for ii = 1:size(a,2)
            v(:,ii) = conv(a(:,ii), b);
        end
    end
    
    v = v(ix0 - 1 + (1:len),:);
    
    if ~isequal(size(v), siz)
        v = reshape(v, siz);
    end
% else
%     error('a should be either a vector or a matrix!');
end
