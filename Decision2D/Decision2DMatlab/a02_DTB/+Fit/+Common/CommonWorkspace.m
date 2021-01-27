classdef CommonWorkspace ...
        < FitWorkspace ...
        & EvAxis.EvidenceAxisInheritable ...
        & TimeAxis.TimeInheritable ...
        & PackageInheriter ...
        & bml.oop.Class2Kind ...
        & bml_local.oop.PropFileNameTree
    % Fit.Common.CommonWorkspace
    
    % 2015-2016 YK wrote the initial version.

%% === Fitting ===
properties
    fit_args = {};
end
methods
    function varargout = fit(W, varargin)
        % [Fl, res] = fit(Fl, ...)
        %
        % OPTIONS:
        % 'optim_fun', @FminconReduce.fmincon
        % 'args', {}
        % 'opts', {}
        % 'outs', {}
        % 'Params', []
  
        C = varargin2C(varargin, W.fit_args);
        [varargout{1:nargout}] = W.fit@FitWorkspace(C{:});
    end
end
%% === CommonWorkspace & Data ===
properties (Transient)
    is_initialized = false;
end 
methods
    function W = CommonWorkspace(varargin)
        W.add_deep_copy({'Fl'});
        W.set_Data;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        S = varargin2S(varargin);
        if ~isfield(S, 'tr_min') && isfield(S, 'subj') ...
                && ~isempty(strfind(S.subj, 'FR'))
            S.tr_min = 1; % Monkey data is already filtered
        end
        C = S2C(S);
        
        W.init@FitWorkspace(C{:});
        
        if ~W.skip_loading_data
            if W.to_init
                W.load_data;
            else
                W.set_path;
            end
        end
        W.is_initialized = true;
    end
    function [files, Ws] = batch(W0, varargin)
        if iscell(varargin{1})
            Ss = varargin{1};
            n = numel(Ss);
        else
            [Ss, n] = W0.get_Ss(varargin{:});
        end
        if nargout >= 1
            files = cell(n, 1);
        end
        if nargout >= 2
            Ws = cell(n, 1);
        end
        for ii = 1:n
            S = Ss{ii};
%             S0 = W0.get_S0_file;
%             if isfield(S, 'subj_parad')
%                 S0 = rmfield(S0, {'subj', 'parad'}); 
%             end
%             S = varargin2S(S, S0);
            C = S2C(S);
            
            W = feval(class(W0), C{:});
            if nargout >= 2
                Ws{ii} = W;
            end
            
%             varargin2props(W, C, true);
            file = [W.get_file, '.mat'];
            if nargout >= 1
                files{ii} = file;
            end
            
            is_fitted = exist(file, 'file');
            if W.skip_existing_mat && is_fitted
                fprintf('Skipping existing file: %s\n', file);
            elseif ~W.to_fit
                fprintf('Not fitting file (to_fit = false): %s\n', file);
            else
                fprintf('Starting %s at %s\n', file, datestr(now, 30));
                W.init(C{:});
                W.main;
                is_fitted = true;
            end
            
            if W.to_plot && is_fitted
                try
                    W.load_mat;
                catch err
                    warning(err_msg(err));
                    continue;
                end
                if ~isempty(W.Fl) && all(~isnan(W.Fl.res.out.x))
                    W.plot_and_save_all;
                end
            end
        end
    end
    function [Ss, n, S_batch] = get_Ss(~, varargin)
        S_batch = varargin2S(varargin, {
            'parad', {'RT', 'sh'}
%             'subj_parad', Data.Consts.subj_parad_all
            });
        [Ss, n] = factorizeS(S_batch);
        Ss = num2cell(Ss(:));
    end
    function plot_and_save_all(W)
        % Use to_plot and skip_existing_fig
        warning('Implement in subclasses!');
    end
    function set_Data(W, Data)
        if nargin < 2, Data = Fit.Common.DataChRtPdf; end
        W.set_Data@FitWorkspace(Data); % Data is always maintained as the root's.
        
        % Match Time to Data.Time.
        W.Data.set_Time(W.get_Time); % Time is updated to match the workspace's.
        
        if W.Data.loaded
            W.customize_th_for_Data;
        end
    end
    function file = save_mat(W)
        file = [W.get_file, '.mat'];
        Fl = W.get_Fl;
        res = Fl.res;
        S0_file = W.get_S0_file;
        L = packStruct(W, Fl, res, S0_file); %#ok<NASGU>
        mkdir2(fileparts(file));
        save(file, '-struct', 'L');
        fprintf('Saved to %s\n', file);
    end
    function success = load_mat(W)
        success = false;
        
        file = [W.get_file, '.mat'];
        L = load(file, 'res');
        W.Fl = [];
        Fl = W.get_Fl;
        Fl.res = L.res;
        if ~Fl.is_valid_res
            return;
        else
            Fl.res2W;
            success = true;
        end
    end
    function W = load_mat_file(W0, file)
        L = load(file);
        fprintf('Loaded %sfrom %s\n', ...
            csprintf('%s, ', fieldnames(L)), ...
            file);
        if isfield(L, 'W')
            if strcmp(class(L.W), class(W0))
                W = L.W;
                return;
            else
                warning('class(W) from file = %s ~= class(W0)', ...
                    class(L.W), class(W0));
            end
        end
        W = feval(class(W0));
        if isfield(L, 'S0_file')
            S0_file = L.S0_file;
        else
            warning('No S0_file found in the file! Using W0''s.');
            S0_file = W0.S0_file;
        end
        C = S2C(S0_file);
        W.init(C{:});
        
        Fl = W.get_Fl;
        Fl.res = L.res;
        Fl.res2W;
    end
end
%% === Batchable ===
properties
    skip_existing_mat = true;
    skip_existing_fig = true;
    to_fit = true;
    to_plot = true;
end    
%% === EvTime ===
methods
    %% Always use root's Time and EvAxis.
    function set_Time(W, Time)
        root = W.get_Data_source;
        root.set_Time@TimeAxis.TimeInheritable(Time);
    end
    function Time = get_Time(W)
        root = W.get_Data_source;
        Time = root.get_Time@TimeAxis.TimeInheritable;
    end
    function set_EvAxis(W, EvAxis)
        root = W.get_Data_source;
        root.set_EvAxis@EvAxis.EvidenceAxisInheritable(EvAxis);
    end
    function EvAxis = get_EvAxis(W)
        root = W.get_Data_source;
        EvAxis = root.get_EvAxis@EvAxis.EvidenceAxisInheritable;
    end
    function set_root(W, new_root)
        % When the W itself becomes a root,
        % set its Time & EvAxis to the previous root's Time & EvAxis.
        
        prev_root = W.get_root;
        W.set_root@FitWorkspace(new_root);
        if W.is_root % Equivalent to W == new_root
            W.set_Time(prev_root.get_Data);
            W.set_EvAxis(prev_root.get_EvAxis);
        end
    end
    function src = get_Data_source(W)
        % Defaults to the root. 
        % Modify, e.g., to self, in subclasses if necessary.
        % Then set_root should be changed as well.
        src = W.get_root;
    end
end
%% === LoclaDimRel ===
properties
    dim_rel_W_ = 1;
end
properties (Dependent)
    dim_rel_W
    dim_irr_W
end    
methods
    %% For local 1D view of the data
    function v = get.dim_rel_W(W)
        v = W.get_dim_rel_W;
    end
    function v = get_dim_rel_W(W)
        v = W.dim_rel_W_;
    end
    function set.dim_rel_W(W, v)
        W.set_dim_rel_W(v);
    end
    function set_dim_rel_W(W, v)
        assert(isscalar(v) && any(v == [1 2]));
        W.dim_rel_W_ = v;
    end
    function v = get.dim_irr_W(W)
        v = W.get_dim_irr_W;
    end
    function v = get_dim_irr_W(W)
        v = 3 - W.get_dim_rel_W;
    end
    function set.dim_irr_W(W, v)
        W.set_dim_irr_W(v);
    end
    function set_dim_irr_W(W, v)
        assert(isscalar(v) && any(v == [1 2]));
        W.dim_rel_W_ = 3 - v;
    end
    function varargout = call_Data_method(W, meth, args)
        % varargout = call_Data_method(W, meth, args)
        assert(ischar(meth));
        assert(~exist('args', 'var') || iscell(args));
        
        % Localize Data
        dim_rel_Data = W.Data.get_dim_rel;
        W.Data.set_dim_rel(W.get_dim_rel_W_);
        
        % Call the method
        [varargout{1:nargout}] = W.Data.(meth)(args{:});
        
        % Restore Data
        W.Data.set_dim_rel(dim_rel_Data);
    end
end
%% === DataFilter ===
properties
    %% Time-saver
    
    % to_init 
    % : whether to perform time-consuming initialization, 
    %   e.g., data loading
    to_init = true;
    
    % Skip loading data, as when it is a child of another workspace.
    skip_loading_data = false;
        
    %% Trial filters
    dif_rel_incl = 'all';
    dif_irr_incl = 'all';
    cond_rel_incl = 'all';
    cond_irr_incl = 'all';
    accu_rel_incl = 'all';
    accu_irr_incl = 'all';    
    
    % dif_rel_irr_incl:
    % {[dif_rel1, dif_irr1], ...}
    % Overrides dif_rel_incl and dif_irr_incl if nonempty
    dif_rel_irr_incl = {}; 
end    
properties (Dependent)
    n_run_excl % # runs excluded in the data
    
    n_dim_task
    
    subj_parad % To specify both jointly
    subj
    parad
    task
    
    tr_min
    
    to_excl_outlier_runs
end
properties
    % tr_incl_prct: uses (tr_incl_prct(1), tr_incl_prct(2)]
    % (exclusive st, inclusive en) to make [0 100] mean all.
    tr_incl_prct = [0 100];
    
    % tr_incl: selects ismember(Data.ds0.i_all_Tr(filt_spec), tr_incl)
    tr_incl = ':';
    
    % tr_incl_name: describes tr_incl. Should set manually.
    tr_incl_name = '';
    
    rt_incl_ =  []; % [0 100];
    rt_incl_unit = 'prct';
end
properties (Dependent)
    tr_incl_prct_name % [] if [0 100] to preserve older names
    
    rt_incl_ms
    rt_incl_prct
    rt_incl
end    
% properties (Access = protected)
%     subj_
%     parad_
%     task_
% end
%% Naming file / Loading data / Importing parameters
properties (Dependent)
    dif_rel_incl_name
    dif_irr_incl_name
    cond_irr_incl_name
    accu_irr_incl_name
end
methods
    function v = get_file_fields0(~)
        v = {
            'subj',                 'sbj'
            'parad',                'prd'
            'task',                 'tsk'
            'n_dim_task',           'dtk'
            'dim_rel_W',            'dmr'
            'dif_rel_incl_name',    'dfr'
            'dif_irr_incl_name',    'dfi'
            'cond_irr_incl_name',   'cdi'
            'accu_irr_incl_name',   'aci'
            'rt_incl',              'rt'
            'tr_incl_prct_name',    'trp'
            'tr_incl_name',         'trn'
            'tr_min',               'trm'
            'to_excl_outlier_runs', 'eor'
%             'n_run_excl',           'nre'
            };
    end
    function load_data(W)
        W.set_path;
        
        if W.Data.loaded ...
                && (isempty(W.Data.ix_run_to_excl) ...
                    == ~W.Data.to_excl_outlier_runs)
            fprintf('%s already loaded - skipping loading.\n', ...
                W.Data.get_path);
        else
            W.Data.load_data;
            fprintf('Done.\n');
        end
        W.filt_data(true);
    end
    
    function set_path(W)
        W.Data.set_path({'subj', W.subj, 'parad', W.parad, 'task', W.task});        
    end
    function set.subj_parad(W, subj_parad)
        W.subj = subj_parad{1};
        W.parad = subj_parad{2};
    end
    function set.subj(W, subj)
        W.Data.set_path({'subj', subj});
    end
    function set.parad(W, parad)
        W.Data.set_path({'parad', parad});
    end
    function set.task(W, task)
        W.Data.set_path({'task', task});
    end
    function v = get.subj(W)
        v = W.Data.subj;
    end
    function v = get.parad(W)
        v = W.Data.parad;
    end
    function v = get.task(W)
        v = W.Data.task;
    end
    
    function v = get.tr_min(W)
        v = W.Data.tr_min;
    end
    function set.tr_min(W, v)
        W.Data.tr_min = v;
    end
    
    function v = get.dif_rel_incl_name(W)
        try
            v = W.dif_rel_incl;
            if isequal(v, 'all')
                v = '';
            end
        catch
            keyboard;
        end
    end
    function set.dif_rel_incl_name(W, v)
        if isequal(v, '')
            v = 'all';
        end
        W.dif_rel_incl = v;
    end
    function v = get.dif_irr_incl_name(W)
        v = W.dif_irr_incl;
        if isequal(v, 'all')
            v = '';
        end
    end
    function set.dif_irr_incl_name(W, v)
        if isequal(v, '')
            v = 'all';
        end
        W.dif_irr_incl = v;
    end
    function v = get.cond_irr_incl_name(W)
        v = W.cond_irr_incl;
        if isequal(v, 'all')
            v = '';
        end
    end
    function set.cond_irr_incl_name(W, v)
        if isequal(v, '')
            v = 'all';
        end
        W.cond_irr_incl = v;
    end
    function v = get.accu_irr_incl_name(W)
        v = W.accu_irr_incl;
        if isequal(v, 'all')
            v = '';
        end
    end
    function set.accu_irr_incl_name(W, v)
        if isequal(v, '')
            v = 'all';
        end
        W.accu_irr_incl = v;
    end    
    
%     function v = get.n_run_excl(W)
%         v = numel(W.Data.ix_run_to_excl);
%     end
%     function set.n_run_excl(W, v)
%         fprintf('set.n_run_excl is not defined! (ignore if didn''t try to)\n');
%     end
    
    function set.to_excl_outlier_runs(W, v)
        W.Data.to_excl_outlier_runs = v;
    end
    function v = get.to_excl_outlier_runs(W)
        v = W.Data.to_excl_outlier_runs;
    end
end
%% Filtering trials
methods 
    function filt_data(W, force_filt)
        if ~exist('force_filt', 'var'), force_filt = false; end

        filt_bef = W.Data.get_dat_filt;
        
        W.Data.set_filt_spec(W.get_filt_spec);
        
        filt_aft = W.Data.get_dat_filt;
        
        if force_filt || ~isequal(filt_bef, filt_aft)
            W.Data.filt_ds;
        end
    end
    function filt_spec = get_filt_spec(W)
        dim_rel = W.get_dim_rel_W;
        dim_irr = W.get_dim_irr_W;
        
        Dat = W.Data;
        
        % tr_min
        tr_wi_task = W.calc_tr_wi_task(Dat.ds0.task, W.task);
        incl_by_tr_min = tr_wi_task >= W.tr_min;
        
        filt_spec = ...
              bsxEq(Dat.ds0.task, W.task) ...
            & incl_by_tr_min;
        
        n_dim = Data.Consts.n_dim;
        for dim = n_dim:-1:1
            [~,~,Dat.ds0.dCond(:,dim)] = unique(Dat.ds0.cond(:,dim));
        end
        for filt1 = {
                Dat.ds0.adCond(:, dim_rel), 'dif_rel_incl'
                Dat.ds0.adCond(:, dim_irr), 'dif_irr_incl'
                Dat.accu0_all_dim(:, dim_rel), 'accu_rel_incl'
                Dat.accu0_all_dim(:, dim_irr), 'accu_irr_incl'
                Dat.ds0.dCond(:, dim_rel),  'cond_rel_incl'
                Dat.ds0.dCond(:, dim_irr),  'cond_irr_incl'
                }'
            [val, prop] = deal(filt1{:});
            if ~(ischar(W.(prop)) && isequal(W.(prop), 'all'))
                filt_spec = filt_spec & bsxEq(val, W.(prop));
            end
        end
        
        % rt_incl
        if ~isempty(W.rt_incl)
            switch W.rt_incl_unit
                case 'ms'
                    filt_spec = filt_spec ...
                        & ((W.rt_incl_ms(1) / 1000) <= Dat.ds0.RT) ...
                        & (Dat.ds0.RT <= (W.rt_incl_ms(2) / 1000));

                    W.set_max_t(W.rt_incl_ms(2) / 1e3);

                case 'prct'
                    rt = Dat.ds0.RT(filt_spec);
                    rt_incl = prctile(rt, W.rt_incl_prct);

                    filt_spec = filt_spec ...
                        & (rt_incl(1) <= Dat.ds0.RT) ...
                        & (Dat.ds0.RT <= rt_incl(2));

                    W.set_max_t(rt_incl(2));
            end
        end
        
        % tr_incl_prct: uses (tr_incl_prct(1), tr_incl_prct(2)]
        % (exclusive st, inclusive en) to make [0 100] mean all.
        if ~isequal(W.tr_incl_prct, [0, 100])
            cum_in_filt = cumsum(filt_spec) .* filt_spec;
            n_in_filt = max(cum_in_filt);
            tr_incl_tr = n_in_filt * W.tr_incl_prct / 100;
            filt_spec = filt_spec ...
                & (cum_in_filt > tr_incl_tr(1)) ...
                & (cum_in_filt <= tr_incl_tr(2));
        end
        
        % tr_incl : uses i_all_Tr
        if ~(ischar(W.tr_incl) && isequal(W.tr_incl, ':'))
            incl = ismember(W.Data.ds0.i_all_Tr, W.tr_incl);
            filt_spec1 = filt_spec & incl;
            filt_spec = filt_spec1;
        end
    end
    function v = get.tr_incl_prct_name(W)
        if isequal(W.tr_incl_prct, [0 100])
            v = [];
        else
            v = W.tr_incl_prct;
        end
    end
    function set.tr_incl_prct_name(W, v)
        if isempty(v)
            W.tr_incl_prct = [0 100];
        else
            assert(numel(v) == 2);
            assert(isnumeric(v));
            W.tr_incl_prct = v;
        end
    end
    function rt_incl = get.rt_incl(W)
        rt_incl = W.rt_incl_;
    end
    function set.rt_incl(W, rt_incl)
        assert(isnumeric(rt_incl));
        if isempty(rt_incl)
            W.rt_incl_ = [];
            return;
        else
            assert(numel(rt_incl) == 2);
            assert(rt_incl(1) <= rt_incl(2));
            W.rt_incl_ = rt_incl;
            
            if rt_incl(2) <= 100
                W.rt_incl_unit = 'prct';
            else
                W.rt_incl_unit = 'ms';
            end
        end
    end
    function set.rt_incl_prct(W, rt_incl_prct)
        if isempty(rt_incl_prct) && ~isempty(W.rt_incl_)
            % Don't do anything.
            % rt_incl can be set to zero only by setting rt_incl = [];
            % not by rt_incl_ms = []; or rt_incl_prct = [];
            % to prevent overwriting saved results from the old version.
        else
            assert(numel(rt_incl_prct) == 2);
            assert(rt_incl_prct(1) <= rt_incl_prct(2));
            W.rt_incl_unit = 'prct';
            W.rt_incl_ = rt_incl_prct;
        end
    end
    function rt_incl_prct = get.rt_incl_prct(W)
        if isempty(W.rt_incl_)
            rt_incl_prct = [];
        else
            switch W.rt_incl_unit
                case 'prct'
                    rt_incl_prct = W.rt_incl_;
                case 'ms'
                    rt = W.Data.get_RT;
                    rt_incl_prct = invprctile(rt, W.rt_incl_ ./ 1e3);
                otherwise
                    rt_incl_prct = nan(1,2);
            end
        end
    end
    function set.rt_incl_ms(W, rt_incl_ms)
        if isempty(rt_incl_ms) && ~isempty(W.rt_incl_)
            % Don't do anything.
            % rt_incl can be set to zero only by setting rt_incl = [];
            % not by rt_incl_ms = []; or rt_incl_prct = [];
            % to prevent overwriting saved results from the old version.
        else
            assert(numel(rt_incl_ms) == 2);
            assert(rt_incl_ms(1) <= rt_incl_ms(2));
            W.rt_incl_unit = 'ms';
            W.rt_incl_ = rt_incl_ms;
        end
    end
    function rt_incl_ms = get.rt_incl_ms(W)
        if isempty(W.rt_incl_)
            rt_incl_ms = [];
        else
            switch W.rt_incl_unit
                case 'ms'
                    rt_incl_ms = W.rt_incl_;
                case 'prct'
                    rt = W.Data.get_RT;
                    rt_incl_ms = prctile(rt, W.rt_incl_) * 1000;
                otherwise
                    warning('rt_incl_unit=%s is not supported!\n', ...
                        W.rt_incl_unit);
                    rt_incl_ms = nan(1,2);
            end
        end
    end
    function tr_wi_task = calc_tr_wi_task(~, task, tasks)
        % tr_wi_task = calc_tr_wi_task(~, task, tasks)
        %
        % EXAMPLE:
        % W.calc_tr_wi_task(('ABCAAACCBBBB')', 'AB')
        % ans =
        %      1
        %      1
        %      0
        %      2
        %      3
        %      4
        %      0
        %      0
        %      2
        %      3
        %      4
        %      5
        
        task_eq = bsxfun(@eq, task, tasks(:)');
        tr_wi_task = sum(cumsum(task_eq) .* task_eq, 2);
    end
    function S = get_S_filt_data(W)
        S = copyprops(struct, W, 'props', {
            'subj'
            'parad'
            'task'
            'dim_rel_W'
            'dif_rel_incl'
            'dif_irr_incl'
            'accu_rel_incl'
            'accu_irr_incl'
            'cond_rel_incl'
            'cond_irr_incl'
            'tr_min'
            });
    end
end
%% Other Data attributes
methods 
    function dCond = get_dCond(W)
        cond = W.Data.get_cond0;
        for dim = W.Data.get_n_dim:-1:1
            conds = unique(cond(:, dim));
            dCond(:, dim) = bsxFind(cond(:, dim), conds);
        end
        dCond = dCond(W.Data.get_dat_filt, :);
    end
    function cond = get_cond_2D(W)
        cond = W.Data.get_cond;
    end
    function cond = get_ch_2D(W)
        cond = W.Data.get_ch;
    end
end
%% Utilities
methods
    function ch = get_ch_dim_rel(W)
        ch = W.get_ch_dim(W.get_dim_rel_W);
    end
    function ch = get_ch_dim_irr(W)
        ch = W.get_ch_dim(W.get_dim_irr_W);
    end
    function ch = get_ch_dim(W, dim)
        ch = W.Data.get_ch;
        ch = ch(:, dim) == 2;
    end
    function ch = get_ch(W)
        ch = W.Data.get_ch == 2;
        ch = ch(:, W.get_dim_rel_W);
        assert(islogical(ch) && iscolumn(ch));
        
%         W.ch = ch(:, W.get_dim_rel_W);
%         assert(islogical(W.ch) && iscolumn(W.ch));
    end    
    function v = get.n_dim_task(W)
        if strcmp(W.task, 'A')
            v = 2;
        else
            v = 1;
        end
    end
    function set.n_dim_task(W, v)
        if v == 2
            W.task = 'A';
        elseif v == 1
            if W.get_dim_rel_W == 1
                W.task = 'H';
            else
                W.task = 'V';
            end
        else
            error('v=%d is not allowed!', v);
        end
    end
end
end