function S_fun = func2str_S(S_fun)

funs = fieldnames(S_fun)';

for f = funs
    if ischar(S_fun.(f{1})), continue; end
    S_fun.(f{1}) = func2str(S_fun.(f{1}));
end