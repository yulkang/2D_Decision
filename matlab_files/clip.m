 function  y = clip( x, lo, hi )
 %y = clip(x, lo, hi )
 %clip values in x (scalar, vector or matrix) to be between lo and hi
 %(either scalars or same size as x)
 
 y = (x .* [x<=hi])  +  (hi .* [x>hi]);
 y = (y .* [x>=lo])  +  (lo .* [x<lo]);