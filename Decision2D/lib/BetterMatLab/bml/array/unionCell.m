function c1 = unionCell(c1, c2)
% Slow but robust union. Order is always stable. Both cells must be vectors.
%
% unionCell(c1, c2)

%% Trivial if c1 or c2 is empty
if isempty(c2), return; end
if isempty(c1), c1 = c2; return; end

%% Check input
assert(isvector(c1) && isvector(c2), 'Both inputs must be vectors!');

%% Pre-assign space
l1 = length(c1);
l2 = length(c2);

c1{l1 + l2} = [];

%% Merge
for ii = 1:l2
    match_found = false;
    
    for jj = 1:l1
        if isequal(c1{jj}, c2{ii})
            match_found = true;
            break; 
        end
    end
    
    if ~match_found
        l1 = l1 + 1;
        c1(l1) = c2(ii);
    end
end
c1 = c1(1:l1);