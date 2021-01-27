function [h, tag] = fig_tag(tag, varargin)
% FIG_TAG  Same as figure() except fig_tag uses a string tag instead of h.
%
% [h, tag] = fig_tag(tag)
% : Gives the handle of the figure with the tag.
%
% fig_tag(tag, 'Property1', property1, ...)
% : Focus and sets the figure's properties
%
% tag can be either a string or a cell array of strings.
%
% See also obj_tag, axes_tag
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

if iscell(tag)
    if verLessThan('matlab', '8.4')
        h = zeros(size(tag));
    else
        h = gobjects(size(tag));
    end
    for ii = 1:numel(tag)
        h(ii) = fig_tag(tag{ii}, varargin{:});
    end
    h = reshape(h, size(tag));
    return;
end

tag = strrep(tag, '_', '-'); % safe_name(tag);

C = varargin2C(varargin, {'Name', tag, 'NumberTitle', 'off'});

h = obj_tag('figure', tag, C{:});