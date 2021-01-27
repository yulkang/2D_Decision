function [l, m, c, r] = parse_linespec(s)
% [line, marker, color, rest] = parse_linespec(s)

c_l  = {'--', ':', '-.'};
ix_l = strfinds(s, c_l, 'first');
if ~isempty(ix_l)
    assert(length(ix_l) == 1, 'Give only one line style in LineSpec!');
    
    l = c_l{ix_l};
    
    loc = strfind(s, c_l{ix_l});
    s(loc + (0:(length(c_l{ix_l})-1))) = '';
%     s = s([1:(ix_l-1), (ix_l+1+length(l)):end]);
else
    tf_l = (s=='-');
    assert(sum(tf_l)<=1, 'Give only one line style in LineSpec!');
    
    l    = s(tf_l);
    s    = s(~tf_l);
end

tf_m = bsxEq(s, '+o*.xsd^v><ph');
m    = s(tf_m);
s    = s(~tf_m);

tf_c = bsxEq(s, 'rgbcmykw');
c    = s(tf_c);
r    = s(~tf_c);