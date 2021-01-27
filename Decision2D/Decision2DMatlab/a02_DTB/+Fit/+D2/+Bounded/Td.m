classdef Td < Fit.D2.Common.CommonWorkspace
    % Decision time.
    
    % 2015 YK wrote the initial version.
properties (Transient)
    td_pdfs
end
methods (Static)
    function W = create(varargin)
        W = Fit.D2.Bounded.TdSer(varargin{:});
    end
end
end