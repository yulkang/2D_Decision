function C_fun = func2str_C(C_fun)
% C_fun = func2str_C(C_fun)

for ii = 1:length(C_fun)
    if ischar(C_fun{ii}), continue; end
    C_fun{ii} = func2str(C_fun{ii});
end