function varargout = axlim(kind, varargin)
% axlim allows parametric choice of x, y, or zlim
% varargout = axlim(kind='x'|'y'|'z', varargin)
%
% See also: xlim ylim zlim
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

if ~ischar(kind)
    ax = kind;
    kind = varargin{1};
    
    if isscalar(ax)
        varargin = [{ax}, hVec(varargin(2:end))];
    else
        [varargout{1:nargout}] = ...
            arrayfun(@(a) bml.plot.axlim(a, kind, varargin{2:end}), ...
                ax, 'UniformOutput', false);
        return;
    end
end

switch kind
    case 'x'
        [varargout{1:nargout}] = xlim(varargin{:});
    case 'y'
        [varargout{1:nargout}] = ylim(varargin{:});
    case 'z'
        [varargout{1:nargout}] = zlim(varargin{:});
    otherwise
        error('Not implemented kind: %s\n', kind);
end
end