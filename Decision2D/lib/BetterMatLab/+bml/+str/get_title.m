function txt = get_title(txt, varargin)
% s = get_title(s, varargin)

C2 = varargin2C2(varargin, {
    '_', '-'
    '^', '.'
    '{', '\{'
    '}', '\}'
    });

txt = bml.str.wrap_text(strrep_cell(txt, C2));