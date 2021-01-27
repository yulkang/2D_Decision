function XLIM = same_xlim(ax_handle)
%function XLIM = same_xlim(ax_handle)
%pasarle un listado de "ax handles" 

a = get(ax_handle,'xlim');
a = cat(1,a{:});
XLIM = [min(a(:,1)) max(a(:,2))];
set(ax_handle,'xlim',XLIM)