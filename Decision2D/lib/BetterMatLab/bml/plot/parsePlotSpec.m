function cc = parsePlotSpec(c)
% PARSEPLOTSPEC Parse plot specification.
%
% cc = parsePlotSpec(c)
%
% c: Cell array in the form of any one of the following.
%
% Format                            : Example
% 
% {'Linespec'}                      : {'k.-'}
% {'Colorspec'}                     : {'k'}, {[0 0.2 0.5]}
% {'Linespec', 'name1', opt1, ...}  : {'k.-', 'LineWidth', 0.5}
% {'Linespec', 'Colorspec', ...}    : {'.-', [0 0.2 0.5], 'LineWidth', 0.5}
% {'name1', opt1, ...}              : {'LineWidth', 0.5}
%
% Example:
%   function examplePlot(x,y,varargin)
%       cc = parsePlotSpec(varargin)
%       plot(x, y, cc{:});
% 
% See also ISCOLORSPEC.

if ischar(c{1})
    cc = c(1);
elseif iscell(c{1})
    if ischar(c{1}{1}) %  && isnumeric(c{1}{2})
        if isColorSpec(c{1}{2})
            cc = {c{1}{1}, 'color', c{1}{2}, c{1}{3:end}};
        else
            cc = {c{1}{1}, c{1}{2:end}};
        end
    else
        cc = c{1};
    end
elseif isColorSpec(c{1})
    cc = {'color', c{1}};
else
    error('parsePlotSpec:WrongFormat', ...
          'Wrong format. See help parsePlotSpec.');
end
end