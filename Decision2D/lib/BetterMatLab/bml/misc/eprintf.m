function eprintf(varargin)
% eprintf(variable1, variable2, ...)
%
% Pretty-prints expressions.
%
% EXAMPLE:
% >> a.a = 2; a.b = 3; eprintf('a.a', 'a.a+a.b')
%
%     a.a = 2
% a.a+a.b = 5
%
% See also: vprintf

if ~any(cellfun(@ischar, varargin)), error('eprintf: only give string expressions!'); end

for ii = 1:length(varargin)
    fprintf('%s = ', varargin{ii});
    
    val = evalin('caller', varargin{ii});
    
    if isnumeric(val)
        if size(val,1) == 1
            fprintf('%1.3f ', val);
            fprintf('\n');
        else
            fprintf('\n');
            disp(val);
        end
        
    elseif ischar(val)
        fprintf('''%s''\n', val);
        
    else
        fprintf('\n');
        disp(val);
    end
end