function [y,IA,IC] = nanunique(x)

inds = not(isnan(x));
[y,IA,IC] = unique(x(inds));
