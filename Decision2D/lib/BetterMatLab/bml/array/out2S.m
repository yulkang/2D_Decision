function res = out2S(fun, fieldNames)
% OUT2S  Packs multiple outputs into one output struct.
% Give '' as field names to ignore the corresponding output.
%
% res = out2S(fun, fieldNames)

n = length(fieldNames);
[resCell{1:n}] = fun();

res = struct;
for ii = 1:n
    if ~isempty(fieldNames{ii})
        res.(fieldNames{ii}) = resCell{ii};
    end
end
