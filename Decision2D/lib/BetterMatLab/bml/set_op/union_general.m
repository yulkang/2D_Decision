function varargout = union_general(a, b, varargin)
% union that works with all types, including cell arrays (use 'stable' mode).
% Note that only 'stable' mode is allowed for the types that doesn't work with 
% union(). Also, concatenation must be defined between the two inputs.
%
% EXAMPLE:
% >> union_general({4,5}, {2,3,4}, 'stable')
% ans = 
%     [4]    [5]    [2]    [3]
%
% >> union_general({4,5,6;7,8,9},{1,2,3;4,5,6}, 'stable', 'rows')
% Warning: The 'rows' input is not supported for cell array inputs. 
% > In cell/union>cellunionR2012a (line 204)
%   In cell/union (line 134)
%   In union_general (line 11) 
% ans = 
%     [4]    [5]    [6]
%     [7]    [8]    [9]
%     [1]    [2]    [3]
%
% See also: union, setdiff_general
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

if ~(iscell(a) && ismember('rows', varargin))    
    try
        % TODO: remove warning when given cell arrays with 'rows' mode.
        [varargout{1:nargout}] = union(a, b, varargin{:});
        return;
    catch
    end
end

is_stable = any(strcmp('stable', varargin));
is_rows   = any(strcmp('rows', varargin));
try
    assert(is_stable, ...
        'Only ''stable'' mode is allowed for non-numeric, non-string inputs!');
catch err
    warning(err_msg(err));
    varargin{end + 1} = 'stable';
    is_stable = true;
end
% TODO: giving second and third outputs is possible.
assert(nargout <= 1, ...
    'Second and third outputs are not implemented yet!');

[~, common_b] = intersect_ix_general(a, b, is_rows);

% Part that depends on the kind of set operation
if is_rows
    varargout{1} = [a; b(~common_b, :)];
else
    if ~isrow(a)
        a = a(:); % To match union()'s behavior.
    end

    a(end + (1:nnz(~common_b))) = b(~common_b);

    varargout{1} = a;
end
end