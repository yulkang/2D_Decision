function ix = strfind_end(src, ptn)
% ix = strfind_end(src, ptn)
ix = strfind(src, ptn) + length(ptn);
if ~isempty(ix)
    ix = ix(end);
end