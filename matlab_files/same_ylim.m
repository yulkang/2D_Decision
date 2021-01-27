function YLIM = same_ylim(ax_handle)
%function XLIM = same_xlim(ax_handle)
%pasarle un listado de "ax handles" 

a = get(ax_handle,'ylim');
a = cat(1,a{:});
YLIM = [min(a(:,1)) max(a(:,2))];
set(ax_handle,'ylim',YLIM)