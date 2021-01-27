function res = safe_name(res, preserve_chr, replace_with)
% SAFE_NAME - replace non-alphanumeric characters with '_'
%
% res = safe_name(res)
% res = safe_name(res, [preserve_chr = ''])
% res = safe_name(res, preserve_chr, [replace_with = '_'])
%
% preserve_chr : Characters to preserve in addition to alphanumeric.
% replace_with : A character to replace the non-preserved characters with.
%
% In case names starting with '_' or a number is prohibited, use
% a common prefix, e.g., safe_name(['c_' res])
%
% EXAMPLE:
% >> safe_name('a+b= 3"')
% ans =
% a_b__3_
%
% >> safe_name('a+b= 3"', '+')
% ans =
% a+b__3_
% 
% >> safe_name('a+b= 3"', '+', '.')
% ans =
% a+b..3.
%
% 2013 (c) Yul Kang, hk2699 at columbia dot edu.
    
if ~exist('preserve_chr', 'var'), preserve_chr = ''; end
if ~exist('replace_with', 'var'), replace_with = '_'; end

if iscell(res)
    res = cellfun(@(s) safe_name(s, preserve_chr, replace_with), res, ...
        'UniformOutput', false);
else
    ix = bsxEq(res(:), ['a':'z', 'A':'Z', '0':'9', preserve_chr(:)']);
    res(~ix) = replace_with;
    
    if ~isempty(res) && any(res(1) == ['_', '0':'9'])
        res = ['x' res];
    end
end