function ds = cell2ds2(C, varargin)
% Similar to cell2ds but accepts name-pair arguments.
%
% function ds = cell2ds2(C, ...)
%
% OPTIONS
% -------
% 'get_colname', true
% 'get_rowname', false
% 'matcols', {}
%
% Give get_rowname == 2 to have a column of the rowname.

S = varargin2S(varargin, {
    'get_colname', true
    'get_rowname', false
    'matcols', {}
    });

if S.get_rowname == 2
    C = [C(:,1), C];
end

if S.get_colname && S.get_rowname
    ds = dataset([{C(2:end,2:end)}, C(1,2:end)], 'ObsNames', C(2:end,1)');
    
elseif S.get_colname && ~S.get_rowname
    ds = dataset([{C(2:end,:)}, C(1,:)]);
    
elseif ~S.get_colname && S.get_rowname
    ds = dataset([{C(:,2:end)}, csprintf('Var%d', 1:(size(C,2)-1))], ...
        'ObsNames', C(:,1)');
    
else
    ds = dataset([{C}, csprintf('Var%d', 1:size(C,2))]);   
end

if ~isempty(S.matcols)
    ds = ds_colfun(ds, @cell2mat2, S.matcols);
end