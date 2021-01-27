function src = nan0(src)
% Replaces NaNs with zero. 
% res = nan0(src)
src(isnan(src)) = 0;
end