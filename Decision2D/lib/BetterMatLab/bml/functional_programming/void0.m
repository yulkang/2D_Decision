function out = void0(f, out)
% Same as void() except the default output is 0. Useful in PlotFcns. - YK
%
% out = void(f, [out=1])
% 
% This is a helper function to enable more complex anonymous functions.
%
% While some functions don't return anything, it can be useful to return
% something after calling them. This function calls the input function and
% then returns either 1 (the default) or the second input, which can be any
% value of the user's choosing. Consider the following anonymous function:
%
% plot_it = @() {figure(1), ...
%                clf(), ...
%                plot(randn(1, 50), '.'), ...
%                axis('off')};
%
% plot_it() % This throws an error.
% 
% Clearly, this should open a figure, clear it, plot some points, and turn
% off the axis. However, the axis('off') command doesn't return anything,
% and something is necessary for that position of the cell array, so MATLAB
% throws an error. However, if we just pass the function to the "void"
% command, as below, it works just fine, returning a 1 that we simply don't
% use but that fills the cell array as MATLAB requires.
%
% plot_it = @() {figure(1), ...
%                clf(), ...
%                plot(randn(1, 50), '.'), ...
%                void(@() axis('off'))};
%
% plot_it() % This works.
%
% Tucker McClure
% Copyright 2013 The MathWorks, Inc.

% Copyright (c) 2013, The MathWorks, Inc.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the The MathWorks, Inc. nor the names 
%       of its contributors may be used to endorse or promote products derived 
%       from this software without specific prior written permission.
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.


    if nargin == 1, out = 0; end
    f();
    
end
