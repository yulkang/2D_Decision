classdef FitParam < matlab.mixin.Copyable
    % FitParam
    %
    % 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.
properties
    name = '';
    th  = []; % current value
    th0 = [];
    th_lb = [];
    th_ub = [];
    th_grad = []; % scalar gradient
    
    th_samp = []; % Samples - catered from, e.g., FitFlow
end
properties (Dependent)
    th_fix
end
methods
    function Prm = FitParam(name, th0, th_lb, th_ub)
        if nargin == 0, return; end
        
        % Batch instantiation
        if iscell(name)
            n = numel(name);
            for ii = n:-1:1
                Prm(ii) = FitParam(name{ii}{:});
            end
            Prm = reshape(Prm, size(name));
            return;
        end
        
        % Single instantiation
        assert(ischar(name) && isnumeric(th0));
        Prm.name = name;
        Prm.th  = th0;
        Prm.th0 = th0;
        
        siz = Prm.get_size();
        
        if nargin < 3, th_lb = -inf(siz); end
        if nargin < 4, th_ub =  inf(siz); end
        
        Prm.th_lb = th_lb;
        Prm.th_ub = th_ub;
        Prm.th_grad = zeros(siz);
    end
    function Prm = merge(Prm, Prm2)
        for ii = 1:numel(Prm2)
            names = Prm.get_names;
            ix = strcmp(Prm2(ii).name, names);
            if ~any(ix)
                Prm(end+1) = Prm2(ii); %#ok<AGROW>
            else
                Prm(ix) = Prm2(ii);
            end
        end
    end
    function aPrm = get_param(Prm, name)
        [ix, exists] = find_param(Prm, name);
        if exists
            aPrm = Prm(ix);
        else
            aPrm = [];
        end
    end
    function [ix, exists] = find_param(Prm, name)
        if isempty_(Prm)
            exists = false;
            ix = 1;
        else
            ix = strcmp(name, Prm.get_names());
            if ~any(ix)
                exists = false;
                ix = numel(Prm) + 1;
            else
                assert(nnz(ix) == 1, 'Non-unique names exist!');
                exists = true;
            end
        end
    end
    function Prm = add_param(Prm, name, varargin)
        assert(nargout > 0, 'Provide output argument to apply modification!');
        ix = find_param(Prm, name);
        Prm(ix) = FitParam(name, varargin{:});
    end
    function Prm = set_(Prm, name, prop, v)
        assert(nargout > 0, 'Provide output argument to apply modification!');
        ix = find_param(Prm, name);
        Prm(ix).(prop) = v;
    end
    function v = get_(Prm, name, prop)
        if nargin < 3, prop = 'th'; end
        [ix, exists] = find_param(Prm, name);
        assert(exists, '%s does not exist!', name);
        v = Prm(ix).(prop);
    end
    function Prm = add_params(Prm, args)
        n = numel(args);
        for ii = 1:n
            Prm = add_param(Prm, args{ii}{:});
        end
    end
    function Prm = remove_params(Prm, names)
        assert(nargout > 0, 'Provide output argument to apply modification!');
        if nargin < 2
            Prm = remove_params_all(Prm);
        end
        if ischar(names)
            names = {names}; 
        end
        Prm = Prm(~ismember(Prm.get_names, names));
        if isempty(Prm)
            Prm = FitParam; % Empty
        end
    end
    function Prm = remove_params_all(~)
        Prm = FitParam;
    end
    function names = get_names(Prm)
        if isempty_(Prm)
            names = {};
            return;
        end        
        n = numel(Prm);
        names = cell(size(Prm));
        for ii = 1:n
            names{ii} = Prm(ii).name;
        end
    end
    function names_attach(Prm, prefix, postfix)
        if nargin < 2, prefix = ''; end
        if nargin < 3, postfix = ''; end
        n = numel(Prm);
        for ii = 1:n
            Prm(ii).name = [prefix, Prm(ii).name, postfix];
        end
    end
    function names_strrep(Prm, src, dst)
        n = numel(Prm);
        for ii = 1:n
            Prm(ii).name = strrep(Prm(ii).name, src, dst);
        end
    end
    function th = get_struct(Prm, prop)
        % th = get_struct(Prm, prop = 'th')
        % prop: 'th', 'th0', 'th_lb', 'th_ub'
        
        if nargin < 2, prop = 'th'; end
        
        th = struct;
        if isempty_(Prm), return; end
        n = numel(Prm);
        
        for ii = 1:n
            th.(Prm(ii).name) = Prm(ii).(prop);
        end
    end
    function Prm = set_struct(Prm, S, prop)
        if nargin < 3, prop = 'th'; end
        
        dst_names = Prm.get_names();
        src_names = fieldnames(S);
        
        for src_name = src_names(:)'
            ix = find(strcmp(src_name{1}, dst_names));
            if isempty(ix)
                continue;
            end
            assert(nnz(ix) == 1, 'Non-unique names exist!');
            Prm(ix).(prop) = S.(src_name{1});
        end
    end
    function th = get_vec(Prm, prop)
        % th = get_vec(Prm, prop = 'th'|struct)
        % prop: 'th', 'th0', 'th_lb', 'th_ub'
        
        if nargin < 2, prop = 'th'; end
%         if ischar(prop)
%             S = Prm.get_struct(prop);
%         end
        
        if isempty_(Prm), th = []; return; end
        
        numels = Prm.get_numel;
        th = zeros(1, sum(numels));
        
        for ii = 1:numel(Prm)
            loc = sum(numels(1:(ii-1))) + (1:numels(ii));
            th(loc) = Prm(ii).(prop)(:)';
        end
    end
    function [Prm, numels] = set_vec(Prm, th, prop)
        if nargin < 3, prop = 'th'; end
        
        numels = Prm.get_numel;
        
        for ii = 1:numel(Prm)
            loc = sum(numels(1:(ii-1))) + (1:numels(ii));
            Prm(ii).(prop) = reshape(th(loc), Prm(ii).get_size);
        end
    end
    function [Prm, numels] = set_mat(Prm, th, prop)
        if nargin < 3, prop = 'th_samp'; end
        
        numels = Prm.get_numel;
        n_samp = size(th, 1);
        
        for ii = 1:numel(Prm)
            loc = sum(numels(1:(ii-1))) + (1:numels(ii));
            Prm(ii).(prop) = reshape(th(:,loc), [n_samp,Prm(ii).get_size]);
        end
    end
    function v = get_numel(Prm, S)
        if nargin < 2
            n = numel(Prm);
            v = zeros(size(Prm));
            for ii = 1:n
                v(ii) = numel(Prm(ii).th);
            end
        else
            fs = fieldnames(S)';
            n  = numel(fs);
            v  = zeros(1,n);
            for ii = 1:n
                v(ii) = numel(S.(fs{ii}));
            end
        end
    end
    function th = get_size(Prm)
        th = size(Prm.th);
    end
    function set.th0(Prm, th)
        assert(isequal(size(th), Prm.get_size));
        Prm.th0 = th;
    end
    function set.th_lb(Prm, th)
        assert(isequal(size(th), Prm.get_size));
        Prm.th_lb = th;
    end
    function set.th_ub(Prm, th)
        assert(isequal(size(th), Prm.get_size));
        Prm.th_ub = th;
    end
    function v = get.th_fix(Prm)
        v = Prm.th_lb == Prm.th_ub;
    end
    function set.th_fix(Prm, v)
        if v
            Prm.th_lb = Prm.th0;
            Prm.th_ub = Prm.th0;
        end
    end
    function v = isempty_(Prm)
        v = isscalar(Prm) && isempty(Prm.name);
    end
    function disp(Param)
        for ii = 1:numel(Param)
            fprintf('%10s %10.2g <- %10.2g [%10.2g, %10.2g]\n', ...
                Param(ii).name, Param(ii).th, Param(ii).th0, Param(ii).th_lb, Param(ii).th_ub);
        end
    end
end
end