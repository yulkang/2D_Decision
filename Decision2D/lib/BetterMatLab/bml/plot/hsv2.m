function map = hsv2(m, s, v)
% First 60% of hsv, flipped so that it goes from blue (small) to red (big).
%
% map = hsv2(m, s, v)
% 
% m: Scalar. Number of colors.
% s, v: Scalar. Saturation and Value (brightness), in [0, 1].
%       Value is adjusted automatically so that green is not too bright.
%
% 2014 (c) Yul Kang. hk2699 at columbia dot edu.

if nargin < 1, m = size(get(gcf,'colormap'),1); end
if nargin < 2, s = 1; end
if nargin < 3, v = 1; end

st = 0.6;
en = -0.45;
steep = 3;
h1 = cos(linspace(acos(st), acos(en), m))';
h2 = (1 - (1 - abs(h1)).^steep) .* sign(h1);
h3 = (h2 - h2(end)) ./ (h2(1) - h2(end));
h4 = min(max(h3, 0), 1);
h = h4 * 0.6; % Steeper in the middle, around green.

% plot(h1);
% plot(h2);
% plot(h3);
% plot(h4);
% plot(h);
% disp('Testing colormap');

% fig_tag('colormap');
% % xLim = xlim;
% % x = linspace(xLim(1), xLim(2), m);
% plot(h, 'k-'); 
% ylim([0 1]); 
% % h = linspace(0.6, 0, m)'; % (0:m-1)'/max(m,1);
if isempty(h)
  map = [];
else
  map = hsv2rgb([h, zeros(m,1) + s, zeros(m,1) + v]);
  map = hsv2rgb([h, zeros(m,1) + s, v - map(:,2)/4]); % Green is too bright
end