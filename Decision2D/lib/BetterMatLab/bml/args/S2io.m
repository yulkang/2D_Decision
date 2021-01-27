function S = S2io(S, f, out, in, varargin)
% Input and output from a struct's fields
%
% S = S2io(S, f, out, in, ...) % Ignored if in = out = {}. In that case, S = f(S).
%
% OPTIONS
% -------
% 'use_varargout', true
%
% Give {} for out or in to specify S itself as the input or output.
%
% Otherwise,
% S = S2io(S, f, {'out1', out2'}, {'in1', 'in2', 'in3'}) calls
% [S.out1, S.out2] = f(S.in1, S.in2, S.in3)
%
% S = S2io(S, f, {'out1', out2'}, {'in1', 'in2', 'in3'}, 'use_varargout', false) calls
% {S.out1, S.out2} = f(S.in1, S.in2, S.in3),
% where f() is expected to give a cell vector of length 2.
%
% 2015 (c) Yul Kang. hk2699 at cumc dot columbia dot edu.

opt = varargin2S(varargin, {
    'use_varargout', true % Ignored if in = out = {}. In that case, S = f(S).
    });

if isempty(in) && isempty(out) 
    % Fastest. Call-by-reference.
    S = f(S);
    
elseif isempty(in)
    % output
    nout = length(out);
    if opt.use_varargout
        [outs{1:nout}] = f(S);
    else
        outs = f(S);
    end
    for ii = 1:nout
        S.(out{ii}) = outs{ii};
    end    
    
elseif isempty(out)
    % input
    nin  = length(in);
    ins  = cell(1, nin);
    for ii = 1:nin
        ins{ii} = S.(in{ii});
    end
    S = f(ins{:});

else
    % Most reusable.
    % input
    nin  = length(in);
    ins  = cell(1, nin);
    for ii = 1:nin
        if ~isempty(in{ii})
            ins{ii} = S.(in{ii});
        end
    end

    % output
    nout = length(out);
    if opt.use_varargout
        [outs{1:nout}] = f(ins{:});
    else
        outs = f(ins{:});
    end
    for ii = 1:nout
        if ~isempty(out{ii})
            S.(out{ii}) = outs{ii};
        end
    end
end