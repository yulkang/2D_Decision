function G = adummyvar(v,ignorenan_flag)

if nargin<2
    ignorenan_flag = 0;
end

if any(isnan(v)) && ignorenan_flag==0
    error('hay nans')
end
u = nanunique(v);

G = zeros(length(v),length(u));

for i=1:length(u)
    G(:,i) = v==u(i);
end



