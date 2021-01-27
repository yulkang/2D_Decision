function xr = redondear(x,ndecimales)
% xr = redondear(x,ndecimales)

xr = round(x*10^ndecimales)/10^ndecimales;

% xr(isnan(x)) = nan; % added, new