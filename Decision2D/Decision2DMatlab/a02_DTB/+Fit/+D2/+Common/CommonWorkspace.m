classdef CommonWorkspace ...
        < Fit.Common.CommonWorkspace ...
    % Fit.D2.Common.CommonWorkspace
    
    % 2016 YK wrote the initial version.

%% Init
methods
    function W = CommonWorkspace(varargin)        
        W.set_Data;
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% 2D specific
methods
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D2.Common.DataChRtPdf; end
        obj_or_name = W.enforce_class('Fit.D2.Common.DataChRtPdf', obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end
end
end