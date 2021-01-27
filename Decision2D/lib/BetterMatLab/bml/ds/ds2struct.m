function S = ds2struct(S, varargin)
% ds2struct  Convert dataset, object, or struct into a struct.
%
% S = ds2struct(S)
%
% See also: dataset, PsyLib
%
% 2013 (c) Yul Kang. See help PsyLib for the license.

SS = varargin2S(varargin, {
    'cellfields', false % Make a scalar struct with fields of column cell vectors.
    });

if isa(S, 'dataset')
    C = dataset2cell(S);
    if SS.cellfields
        S = struct;
        for ii = 1:size(C,2)
            S.(C{1,ii}) = C(2:end,ii);
        end
    else
        S = cell2struct(C(2:end,:), C(1,:), 2);
    end
elseif isobject(S)
    S = copyFields(struct, S);
elseif ~isstruct(S)
    error('Give only dataset, struct, or object!');
end