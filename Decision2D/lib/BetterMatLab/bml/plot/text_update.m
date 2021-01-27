function h_text = text_update(x, y, str, varargin)
% h_text = text_update(x, y, str, varargin)
h_text = findobj(gca, 'Type', 'Text', 'Position', [x, y, 0]);
if isvalidhandle(h_text)
    set(h_text, 'String', str, varargin{:});
else
    h_text = text(x, y, str, varargin{:});
end
end