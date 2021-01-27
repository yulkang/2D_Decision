function M = hsv2rev(varargin)
% hsv2rev Same as hsv2 but reversed.

M = flipud(hsv2(varargin{:}));
