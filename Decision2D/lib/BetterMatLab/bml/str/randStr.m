function s = randStr(n, r)
% s = randStr(n=7, r)

if nargin == 0
    n = 7;
end

if isscalar(n)
    n = [1, n];
end
if nargin >= 2
    if isnumeric(r)
        assert(all((n > 0) & (n < 1)), 'Provide numbers in (0, 1)!');
        s = n;
    elseif isa(r, 'RandStream')
        s = rand(r, [1, n]);
    else
        error('2nd argument must be either random numbers or a RandStream!');
    end
else
    s = rand(n);
end

s = char( ...
    (s <  0.5) .* (floor(s * 2 * 26) + 'a') ...
  + (s >= 0.5) .* (floor((s - 0.5) * 2 * 26) + 'A'));