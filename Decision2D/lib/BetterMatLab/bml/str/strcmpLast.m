function res = strcmpLast(a, b)
% Compare last parts of two strings.
%
% res = strcmpLast(a, b)
lenA = length(a);
lenB = length(b);

if lenA < lenB
    res = strcmp(a, b((lenB-lenA+1):lenB));
else
    res = strcmp(a((lenA-lenB+1):lenA), b);
end