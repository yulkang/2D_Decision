classdef PackageInheriter < matlab.mixin.Copyable
    % W = PackageInheriter.enforce_class(default_class, [obj_or_suffix, args])
    %
    % 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.
    methods (Static)
        function W = enforce_class(default_class, W, args)
            % W = enforce_class(default_class, obj_or_suffix, args)
            %
            % Give empty W to use default.
            %
            % (1) When W is an object, just check the type.
            % (2) When W is char, 
            %     (2.1) When default_class.create exists and W is empty
            %           : W = default_class.create(args{:})
            %     (2.2) Otherwise, instantiate default_class with suffix W.
            %           : W = [default_class, W](args{:});
            if nargin < 2
                W = '';
            end
            if nargin < 3
                args = {};
            else
                % Allows the matrix format: {'arg1', arg1; ...}
                args = varargin2C(args);
            end
            if ischar(W)
                if isempty(W) && is_direct_method(default_class, 'create')
                    % When the default_class is also a template class,
                    % it may directly define create() method to 
                    % create its subclass instead of itself.
                    %
                    % i.e., TemplateClass.create(...) is just a delegate for
                    % DefaultClassConstructor(...)
                    W = feval([default_class '.create'], args{:});
                else
                    W = feval([default_class, W], args{:});
                end
            else
                % Evaluate if a function handle (as from pkg2S)
                if isa(W, 'function_handle')
                    W = W();
                end
                
                % If an object is given, just check its type.
                assert(isa(W, default_class));
            end
        end
    end
end