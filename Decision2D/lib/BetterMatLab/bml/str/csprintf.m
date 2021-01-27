function res = csprintf(fmt, varargin)
% CSPRINTF sprintf that repeats over cell array or array input.
%
% res = CSPRINTF(fmt, arg1, arg2, ...)
%
%   res: Cell array
%   arg: Either cell array or array. Single-element array will be expanded.
%
%   See also CFPRINTF, SPRINTF.
%
% 2013 (c) Yul Kang

    nArgin = length(varargin);
    lenArg = cellfun(@numel, varargin);
    lenMax = max( lenArg );
    
    if any(lenArg==0)
        if lenMax > 0
            error('At least one arg is empty, while another is not!');
        else
            res = {''};
            return;
        end
        
    else    
        for ii = nArgin:-1:1
            if iscell(varargin{ii})
                arg{ii} = varargin{ii};
            else
                arg{ii} = num2cell(varargin{ii});
            end

            arg{ii} = rep2fit(arg{ii}(:)', [1 lenMax]);
        end

        res = cellfun( @(varargin)sprintf(fmt, varargin{:}), arg{:}, ...
                       'UniformOutput', false);
    end
end