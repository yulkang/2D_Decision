%rainbow_colors    [C] = rainbow_colors(n)
%
% returns a set of n colors following a
% stretch of the rainbow.

% (c) 2004 CK Machens & CD Brody

function [C] = rainbow_colors(nclasses,varargin)

colorType = 0; %default
for i=1:length(varargin)
    if ischar(varargin{i}) && strcmp(lower(varargin{i}),'colortype')
        colorType = varargin{i+1};
    end
end

if colorType == 0
    rainbowcolormap = hsv(256);
    rainbowcolormap = ...
        rainbowcolormap([250:256 1:40 50:110 135:155],:);
elseif colorType == 1
    rainbowcolormap = gray(256);
    rainbowcolormap = rainbowcolormap(1:230,:);
elseif colorType == 2
    rainbowcolormap = jet(256);
elseif colorType == 3
    rainbowcolormap = copper(256);
elseif colorType == 4
    rainbowcolormap = hsv(256);    
elseif colorType == 5
    rainbowcolormap = cbrewer('div','RdYlGn',nclasses);
elseif colorType == 6
%     rainbowcolormap = cbrewer('qual','Set1',nclasses);    
    rainbowcolormap = cbrewer('qual','Dark2',nclasses);    
elseif colorType == 7
    rainbowcolormap = colors_az(nclasses);    
elseif colorType == 8
    rainbowcolormap = movshon_colors(nclasses);
end


rainbowcolormap = rainbowcolormap(end:-1:1,:);
cmap = rainbowcolormap;

C = zeros(nclasses, 3);

for i=1:nclasses
    g = ((i-1)/(nclasses-1))*(size(cmap,1)-1) + 1;
    g = round(g);
    C(i,:) = cmap(g,:);
end;
