function inst = varargin2fields(inst, vararginCell, suppressError)
% USAGE: inst = varargin2fields(inst, vararginCell, suppressError=true)
	if nargin < 2 || isempty(vararginCell), return; end
    if nargin < 3, suppressError = true; end
    
    % Enforce behavior consistent with varargin2S and varargin2C
    vararginCell = varargin2C(vararginCell);

    for iArgin = 1:2:numel(vararginCell)
        try
            inst.(vararginCell{iArgin}) = vararginCell{iArgin+1};
        catch
            if ~suppressError, rethrow(lasterror); end
        end
    end
end