function [ds, added_cols] = ds_set(ds, ix, varargin)
% Assign data to columns. Convenient when adding column or using scalar value.
% 
% ds = ds_set(ds, ix, 'col1', val1, 'col2', val2, ...)
% ds = ds_set(ds, ix, dataset_or_struct)
% ds = ds_set(ds, ix, dataset_or_struct, 'only', {'col1', 'col2', ...})
% ds = ds_set(ds, ix, dataset_or_struct, 'except', {'col1', 'col2', ...})
% ds = ds_set(ds, ix, cell_array_of_columns) % Fill from the first columns
% ds = ds_set(ds, ix, cell_array_of_columns, 'only', logical_ix) % Fill designated columns
% ds = ds_set(ds, ix, cell_array_of_columns, 'only', numerical_ix)
% ds = ds_set(ds, ix, cell_array_of_columns, 'only', {'col1', 'col2', ...}) % One can augment columns by giving new column names
% [ds, added_cols] = ds_set(...)
%
% ix: logical or numerical index for the row(s), or a function handle.
% added_cols: Columns that did not exist before adding.
%
% If values for a column have variable lengths, give a cell array.
%
% EXAMPLE:
% >> ds = dataset;
% >> ds.rep = {'aa', 'bb', 'aa'}'
% ds = 
%     rep     
%     'aa'    
%     'bb'    
%     'aa'    
% 
% >> ds = ds_set(ds, ':', 'succT', false)
% ds = 
%     rep         succT
%     'aa'        false
%     'bb'        false
%     'aa'        false
%
% >> ds = ds_set(ds, 2:3, 'str', {'abc', 'd'})
% ds = 
%     rep         succT         str
%     'aa'        false         []
%     'bb'        false         'abc'
%     'aa'        false         'd'
%
% See also: dsSet, ds_pack_vars

if isempty(varargin), added_cols = {}; return; end

if isempty(ix), return; end
if isa(ix, 'function_handle')
    ix = ix(ds);
end

if isequal(ds, []), ds = dataset; end

if nargout >= 2
    cols_orig = ds.Properties.VarNames;
end

if iscell(varargin{1}) % Cell array of columns
    C = varargin{1};
    colnames = ds.Properties.VarNames;
    
    if length(varargin) < 2
        colnames = colnames(1:length(C));
    else
        if isnumeric(varargin{3})
            ccolnames = colnames(varargin{3});
        elseif islogical(varargin{3})
            ccolnames = colnames(varargin{3});
        elseif iscellstr(varargin{3})
            ccolnames = varargin{3};
        end
        
        switch varargin{2}
            case 'only'
                % keep cols
                colnames = ccolnames;
                
            case 'except'
                colnames = setdiff(colnames, ccolnames, 'stable');
                
            otherwise
                error('The fourth argument should be ''only'', ''except'', or omitted!');
        end
    end
    
    for icol = 1:length(colnames)
        ccol = colnames{icol};
        w    = size(C{icol}, 2);
        
        ds.(ccol)(ix,1:w) = C{icol};
    end
    
elseif ~ischar(varargin{1})
    S = varargin{1};
    f = fieldnames(S)';
    
    if length(varargin) >= 2
        switch varargin{2}
            case 'only'
                f = varargin{3};
                
            case 'except'
                f = setdiff(f, varargin{3}, 'stable');
        end
    end
    
    if isstruct(S)
        % String fields in struct is char, while those in dataset is cell.
        % To prevent complexity, just convert to dataset.
        % ds_set_cell uses a different approach.
        
        try
            S = dataset(S);
        catch
            for cf = f
                if ischar(S.(cf{1}))
                    S.(cf{1}) = {S.(cf{1})};
                end
            end
        end
        
    elseif isa(S, 'dataset')
        % When S is a dataset, remove 'Properties' from fieldnames.
        f = setdiff(f, {'Properties'}, 'stable');
    elseif exist('istable', 'file') && istable(S)
        f = setdiff(f, {'Properties', 'Row', 'Variables'}, 'stable');
    end
    
    if ischar(ix) && isequal(ix, ':')
        for c_f = f
            ds.(c_f{1})(:,1:size(S.(c_f{1}),2)) = S.(c_f{1});
        end
    elseif isscalar(ix) || nnz(ix) == 1
        for c_f = f
            if isscalar(S.(c_f{1}))
                try
                    % Scalars are saved as a column vector.
                    ds.(c_f{1})(ix,1) = S.(c_f{1});
                catch 
                    try
                        % If previously saved as a cell, continue so.
                        ds.(c_f{1})(ix,1) = {S.(c_f{1})};
                    catch err
                        warning('Error processing %s\n', c_f{1});
                        rethrow(err);
                    end
                end
            else
                try
                    % Try saving as a cell
                    ds.(c_f{1})(ix,1) = {S.(c_f{1})};
                catch
                    % Some legacy entries are saved as row vectors.
                    ds.(c_f{1})(ix,1:size(S.(c_f{1}),2)) = S.(c_f{1});
                end
            end
        end
    else
        for c_f = f
            try
                ds.(c_f{1})(ix,1:size(S.(c_f{1}),2)) = S.(c_f{1});
            catch err
                if size(S.(c_f{1}), 1) == 1
                    n_ix = bml.matrix.n_ix(ix, length(ds));
                    try
                        ds.(c_f{1})(ix,1:size(S.(c_f{1}),2)) = ...
                            repmat(S.(c_f{1}), [n_ix, 1]);
                    catch err
                        warning('Error processing %s\n', c_f{1});
                        rethrow(err);
                    end
                else
                    warning('Error processing %s\n', c_f{1});
                    rethrow(err);
                end
            end
        end
    end
else
    if ischar(ix) && isequal(ix, ':')
        for ii = 1:2:length(varargin)
            ds.(varargin{ii})(:,1:size(varargin{ii+1},2)) = varargin{ii+1};
        end
    elseif isscalar(ix) || nnz(ix) == 1
        for ii = 1:2:length(varargin)
            if isscalar(varargin{ii+1})
                try
                    % Scalars are saved as a column vector.
                    ds.(varargin{ii})(ix,1) = varargin{ii+1};
                catch 
                    try
                        % If previously saved as a cell, continue so.
                        ds.(varargin{ii})(ix,1) = varargin(ii+1);
                    catch err
                        warning('Error processing %s\n', varargin{ii});
                        rethrow(err);
                    end                    
                end
            else
                try
                    % Try saving as a cell
                    ds.(varargin{ii})(ix,1) = varargin(ii+1);
                catch
                    try
                        % Some legacy entries are saved as row vectors.
                        ds.(varargin{ii})(ix,1:size(varargin{ii+1},2)) = varargin{ii+1};
                    catch err
                        warning('Error processing %s\n', varargin{ii});
                        rethrow(err);
                    end
                end
            end
        end        
    else
        for ii = 1:2:length(varargin)
            try
                ds.(varargin{ii})(ix,1:size(varargin{ii+1},2)) = varargin{ii+1};
            catch err
                warning('Error processing %s\n', varargin{ii});
                rethrow(err);
            end
        end
    end
end

if nargout >= 2
    added_cols = setdiff(ds.Properties.VarNames, cols_orig, 'stable');
end