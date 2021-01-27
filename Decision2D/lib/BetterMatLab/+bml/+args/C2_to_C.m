function C = C2_to_C(C2)
% C = C2_to_C(C2)
%
% C2: n x 2 cell array
% C: 1 x (n*2) cell array. Useful for providing as a comma-separated lists.
C = hVec(C2');