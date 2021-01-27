function p = betacdf_ms(x, m, s)
% p = betacdf_ms(x, m, s)
%
% m, s: mean and sd of the distribution

v = s.^2;
m1 = m .* (1 - m) ./ v - 1;
b1 = m .* m1;
b2 = (1 - m) .* m1;
p = betacdf(x, b1, b2);
end