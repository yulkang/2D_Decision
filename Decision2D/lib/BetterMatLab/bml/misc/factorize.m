function [res n] = factorize(rep, varargin)
% FACTORIZE Combine factors into a cell array.
%
% [res n] = factorize({[cell]Array1, [cell]Array2, ...})
%
% OPTIONS
% -------
% 'firstFast', false
%
% EXAMPLE
% -------
%
% factorize({{1 2}, [10 20 30]})
% ans = 
%     [1]    [10]
%     [1]    [20]
%     [1]    [30]
%     [2]    [10]
%     [2]    [20]
%     [2]    [30]
% 
% [r n] = factorize({{2 1}, {10 20 30}, {5}, {6}})
% r = 
%     [2]    [10]    [5]    [6]
%     [2]    [20]    [5]    [6]
%     [2]    [30]    [5]    [6]
%     [1]    [10]    [5]    [6]
%     [1]    [20]    [5]    [6]
%     [1]    [30]    [5]    [6]
% n = 
%     6
%
% See also: factorDS.
    
S = varargin2S(varargin, {
    'firstFast', false
    });

if S.firstFast
    rep = fliplr(rep);
end

nV   = length(rep);
res  = cell( prod( cellfun(@numel, rep) ), nV );
iRes = 0;
history = zeros(1, nV);

for iRep = 1:length(rep)
    if ~iscell(rep{iRep})
        rep{iRep} = num2cell(rep{iRep});
    end
end

combine(1);
n = size(res,1);

if S.firstFast
    res = fliplr(res);
end

    function combine(depth)
        if depth <= nV
            
            for ii = 1:numel(rep{depth})
                history(depth) = ii;
                
                combine(depth+1);
            end
        
        else
            iRes = iRes + 1;
            
            for ii = 1:nV
                res{iRes, ii} = rep{ii}{history(ii)};
            end
        end
    end
end