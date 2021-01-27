function h = ghandles(varargin)
% Graphics handle placeholders compatible across MATLAB versions.

global glVerLessThanMatlab84

% defined in startup.m verLessThan('matlab', '8.4')
% verLessThan causes infinite recursion in parpool dependency analysis
if glVerLessThanMatlab84 
    h = zeros(varargin{:});
else
    h = gobjects(varargin{:});
end