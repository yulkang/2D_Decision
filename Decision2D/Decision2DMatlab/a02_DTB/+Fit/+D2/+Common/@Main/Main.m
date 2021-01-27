classdef Main ...
        < Fit.Common.Main ...
        & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.Main
    
    % 2015 YK wrote the initial version.
    
%% Settings - Fit
properties
    % to_use_easiest_only
    % : If +1, calculate cost from the easiest conditions only.
    %   If 0, calculate cost from all conditions.
    %   If -1, calculate cost from non-easiest conditions only.
    %
    % : 1 means the easiest conditions, negative means 'except for'
    %   Note: The convention is opposite to dif_rel_incl, etc.,
    %   where 1 is the hardest condition.
    %
    % : 0 or [] is to use all.
    % : Vector means include all elements.
    %   A nonscalar vector input must contain 
    %   all-positive or all-negative elements.
    to_use_easiest_only = 1;
    
    % to_use_easiest_only to use during fitting
    to_use_easiest_only_for_fit = 1;
    
    % to_use_easiest_only to use during model comparison
    to_use_easiest_only_for_comparison = -1;
    
    % to_include_last_frame
    % : If true, include last frame in calculating cost
    %   If not true, set predicted RT at last frame to zero and normalize
    %   so that p(predicted_RT) sums to 1 within each condition.
    to_include_last_frame = false;
    
    % to_exclude_bins_wo_trials
    % default: [] for parametric fits, 
    %          10 for nonparametric fits (should be set
    %          in the respective constructor: see Td2Tnd.Main)
    % : empty: don't exclude any bin.
    % : scalar: Remove if the number of trials within the training
    %           bin is no more than that.
    to_exclude_bins_wo_trials = [];
    
    fix_kappa = [];
end
properties (Dependent)
    % 0 if to_exclude_bins_wo_trials is empty,
    % otherwise the same as to_exclude_bins_wo_trials
    to_exclude_bins_wo_trials_thres
end
%% Properties - common
properties (SetAccess=protected)
    Miss
end
%% Properties - Settings    
properties
    to_use_history = true;
    
%     to_plot = true;
    to_save_plot = true;
    to_plot_kind_ = 'all';    
end
properties (Dependent)
    to_plot_kind
end
%% Properties - Intermediate variables
properties (Transient)
    W_now % so that when batch is stopped, W can be accessed.    
end
properties (Dependent)
    % cond_ch_to_exclude(cond1, cond2, ch1, ch2)
    % : set in init()
    % : depends on to_exclude_bins_wo_trials
    cond_ch_to_exclude % = []; 
end
%% Init
methods
    function set_Miss(W, obj_or_name)
        default_obj = Fit.Common.Miss;
        if nargin < 2, obj_or_name = W.miss_kind; end
        W.Miss = W.enforce_class(class(default_obj), obj_or_name);
        W.set_sub_from_props({'Miss'});
    end
end
%% Main - Template
methods
    function W = Main(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.Common.CommonWorkspace(varargin{:});
    end
    function v = get.cond_ch_to_exclude(W)
        v = W.get_cond_ch_to_exclude;
    end
    function n_tr_cond_ch = get_n_tr_cond_ch(W)
        data = W.get_data_pdf;
        n_tr_cond_ch = permute(sum(data, 1), [2, 3, 4, 5, 1]);        
    end
    function n_tr_cond_ch_dim = get_n_tr_cond_ch_dim(W, dim)
        n_tr_cond_ch = W.get_n_tr_cond_ch;
        if dim == 1
            n_tr_cond_ch_dim = sums(n_tr_cond_ch, [2, 4], true);
        else
            assert(dim == 2, 'dim must be 1 or 2!');
            n_tr_cond_ch_dim = sums(n_tr_cond_ch, [1, 3], true);
        end
    end
    function v = get_cond_ch_to_exclude(W)
        % Set cond_ch_to_exclude based on to_exclude_bins_wo_trials
        thres = W.to_exclude_bins_wo_trials_thres;
            
        to_incl_cond = W.get_to_incl_cond_training;
        to_incl_cond_ch = repmat(to_incl_cond, [1, 1, 2, 2]);

        n_tr_cond_ch = W.get_n_tr_cond_ch;

        n_cond_ch_training = n_tr_cond_ch .* to_incl_cond_ch;
        cond_ch1_to_exclude = all(all(n_cond_ch_training <= thres, 2), 4);
        cond_ch2_to_exclude = all(all(n_cond_ch_training <= thres, 1), 3);

        v = bsxfun(@or, cond_ch1_to_exclude, cond_ch2_to_exclude);        
    end
    function v = get.to_exclude_bins_wo_trials_thres(W)
        if isempty(W.to_exclude_bins_wo_trials)
            v = 0;
        else
            v = W.to_exclude_bins_wo_trials;
        end
    end
    function [Fls, ress] = batch(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'model_kind', 'RT'
            });
        Ss = bml.args.factorizeS(S_batch);        
        [Fls, ress] = W0.batch_Ss(Ss);
    end
    function [Fls, ress] = batch_Ss(W0, Ss)
        if isstruct(Ss), Ss = num2cell(Ss); end
        n = numel(Ss);
        
        Fls = cell(n, 1);
        ress = cell(n, 1);
        
        % Data stored in case there is error
        Ws = cell(n, 1); 
        errs = cell(n, 1);
        n_err = 0;
        
        for ii = 1:n
            S = Ss{ii};
            C = S2C(S);
            
            W = W0.create(C{:});
            W0.W_now = W;
            Ws{ii} = W;
            
            try
                [Fls{ii}, ress{ii}] = W.main;
            catch err
                warning(err_msg(err));
                errs{ii} = err;
                n_err = n_err + 1;
            end
        end     
        
        if n_err > 0
            mkdir2(fullfile(Data.Consts.data_root, 'temp'));
            file = fullfile(Data.Consts.data_root, 'temp', ...
                sprintf('errs_Fit2_Common_Main_%s', datestr(now, 30)));
            save(file);
            fprintf('%d errors occured: errs and Fls saved to %s\n', ...
                n_err, file);
        end
    end
    function [files, Ss] = batch_files(W0, varargin)
        S_batch = varargin2S(varargin);
        [Ss, n] = factorizeS(S_batch);
        files = cell(n, 1);
        W = feval(class(W0));
        for ii = 1:n
            S = Ss(ii);
            varargin2props(W, S);
            files{ii} = W.get_file;
        end
    end
    function [Fl, res] = main(W)
        file = W.get_file;
        if W.skip_existing_mat && exist([file, '.mat'], 'file')
            fprintf('Skipping existing fit: %s\n', [file '.mat']);
            if nargout > 0 || (W.to_save_plot && ~W.skip_existing_fig)
                L = load([file '.mat']);
                Fl = W.get_Fl;
                Fl.res = L.res;
                res = L.res;
            else
                Fl = [];
                res = [];
            end
        else
            [Fl, res] = W.fit;
            W.save_mat;
            if W.to_save_plot
                W.plot_and_save_all;
            end
        end
    end
    function save_mat(W)
        if isempty(W.Fl)
            warning('W.Fl is empty! Skipping saving.');
            return;
        elseif isequal(W.Fl.res, struct)
            warning('W.Fl.res is empty! Skipping saving.');
            return;
        end
        
        Fl = W.Fl;
        res = Fl.res;
        L = packStruct(W, Fl, res); %#ok<NASGU>
        
        file = [W.get_file '.mat'];
        mkdir2(fileparts(file));
        
        save(file, '-struct', 'L');
        fprintf('Saved to %s\n', file);        
    end
    function varargout = get_cost_validation(W)
        % [cost, cost_sep] = get_cost_validation(W)
        % : Use to_use_easiest_only_for_comparison for to_use_easiest_only
        to_use_easiest_only0 = W.to_use_easiest_only;
        W.to_use_easiest_only = W.to_use_easiest_only_for_comparison;
        [varargout{1:nargout}] = W.get_cost;
        W.to_use_easiest_only = to_use_easiest_only0;
    end
    function cond_ch_to_incl = get_cond_ch_to_incl(W, to_use_easiest_only)
        if nargin < 2
            to_use_easiest_only = W.to_use_easiest_only;
        end
        cond_incl = W.get_to_incl_cond(to_use_easiest_only);
        cond_ch_excl_n_tr = W.cond_ch_to_exclude;
        cond_ch_to_incl = bsxfun(@and, cond_incl, ~cond_ch_excl_n_tr);
    end
    function cond_ch_to_incl = get_cond_ch_to_include_train(W)
        cond_ch_to_incl = ...
            W.get_cond_ch_to_incl(W.to_use_easiest_only_for_fit);
    end
    function cond_ch_to_incl = get_cond_ch_to_include_valid(W)
        cond_ch_to_incl = ...
            W.get_cond_ch_to_incl(W.to_use_easiest_only_for_comparison);
    end
    function [cost, cost_sep] = calc_cost(W)
        % [cost, cost_sep] = calc_cost(W)
        pred = W.get_pred_pdf;
        data = W.get_data_pdf;
%         pred = W.Data.get_RT_pred_pdf;
%         data = W.Data.get_RT_data_pdf;
        
        if ~W.to_include_last_frame
            % Normalize pred after removing the last frame
            pred = W.set_last_frame_0_and_normalize(pred);
        end
        
        % Reshape into a (bin, cond) matrix for nll_bin.
        siz0 = size(pred);
        n_conds = siz0([2 3]);
        siz  = [prod(siz0([1 4 5])), prod(n_conds)];
        
        pred = reshape(permute(pred, [1 4 5 2 3]), siz);
        data = reshape(permute(data, [1 4 5 2 3]), siz);
        
        % cost_sep(t x ch1 x ch2, cond1 x cond2)
        [~, cost_sep] = bml.stat.nll_bin( ...
            pred, data, ...
            'normalize', true);
        
        % cost_sep(t, cond1, cond2, ch1, ch2)
        cost_sep = permute(reshape(cost_sep, ...
            siz0([1, 4, 5, 2, 3])), [1, 4, 5, 2, 3]);

        cond_ch_to_incl = W.get_cond_ch_to_incl;
        cost = sum(vVec(bsxfun(@times, cost_sep, ...
            permute(cond_ch_to_incl, [5, 1, 2, 3, 4]))));

        if any(isnan(pred(:)))
            warning('any(isnan(pred(:)))');
            keyboard;
        end
    end
    function to_incl = get_to_incl_cond_validation(W)
        to_incl = W.get_to_incl_cond( ...
            W.to_use_easiest_only_for_comparison);
    end
    function to_incl = get_to_incl_cond_training(W)
        to_incl = W.get_to_incl_cond( ...
            W.to_use_easiest_only_for_fit);
    end
    function to_incl = get_to_incl_cond(W, to_use_easiest_only)
        % to_incl = get_to_incl_cond(W, to_use_easiest_only)
        %
        % to_incl(cond1, cond2)
        
        if nargin < 2
            to_use_easiest_only = W.to_use_easiest_only;
%             to_use_easiest_only = W.to_use_easiest_only_for_fit;
        end
        
        data = W.get_data_pdf;
        siz0 = size(data);
        n_conds = siz0([2 3]);
        
        % to_use_easiest_only
        % : 1 means the easiest conditions, negative means 'except for'
        %   Note: The convention is opposite to dif_rel_incl, etc.,
        %   where 1 is the hardest condition.
        %
        % : 0 or [] is to use all.
        % : Vector means include all elements.
        %   A nonscalar vector input must contain 
        %   all-positive or all-negative elements.
        if isempty(to_use_easiest_only) ...
            || isequal(to_use_easiest_only, 0)
            to_incl = true(n_conds);
        else
            dif_incl = to_use_easiest_only;
            if ~(all(dif_incl > 0) || all(dif_incl < 0))
                disp('to_use_easiest_only:');
                disp(dif_incl);
                error(['All elements of to_use_easiest_only must have' ...
                       'the same sign']);
            end
            
            to_excl = all(dif_incl < 0);
            if to_excl
                dif_incl = -dif_incl;
            end

            n_dim = 2;
            dif_incls = cell(1, n_dim);

            for dim = 1:n_dim
                conds = uniquetol(W.Data.cond(:,dim));

                % smallest conds = hardest condition becomes 1 in d_cond
                [~,~,d_cond] = unique(abs(conds));

                % Make 1 the easiest condition
                difs = max(d_cond) + 1 - d_cond;
                dif_incls{dim} = ismember(difs, dif_incl);
            end

            to_incl = false(n_conds);
            to_incl(dif_incls{1}, :) = true;
            to_incl(:, dif_incls{2}) = true;
            if to_excl
                to_incl = ~to_incl;
            end
        end
%         disp('Included conditions');
%         disp(to_incl);        
    end
    function pred = get_pred_pdf(W)
        pred = W.Data.get_RT_pred_pdf;
    end
    function data = get_data_pdf(W)
        data = W.Data.get_RT_data_pdf;
    end
    function p = set_last_frame_0_and_normalize(~, p)
        % p(frame, cond1, cond2, ch1, ch2)
        p(end,:,:,:,:) = 0;
        p = bsxfun(@rdivide, p, sums(p, [1, 4, 5]));
    end
    function [Fl, res] = fit(W, varargin)
        % [Fl, res] = fit(W, varargin)
        %
        % A template for fitting functions.
        % See also: FitFlow.fit_grid
        Fl = W.get_Fl;
        
        S = varargin2S(varargin, {
            'opts', {}
            });
        S.opts = varargin2S(S.opts, {
            'UseParallel', 'always'
            'FiniteDifferenceType', 'central'
            });
        C = S2C(S);
        
        to_use_easiest_only0 = W.to_use_easiest_only;
        W.to_use_easiest_only = W.to_use_easiest_only_for_fit;
        res = Fl.fit(C{:});
        W.to_use_easiest_only = to_use_easiest_only0;
    end
    function Fl = get_Fl(W)
        Fl = W.get_Fl@Fit.D2.Common.CommonWorkspace;        
        Fl.plot_opt.to_plot = W.to_plot;        
    end
end
%% Batch - RT
properties (Constant)
    models_RT = {'Ser', 'Par', 'Exv', 'Trg', 'InhDrift', 'InhFree'}; % , 'InhFixFano'};
    models_RT_Inh = {'Ser', 'Par', 'InhDrift', 'InhFree'}; % , 'InhFixFano'};
    models_RT_long = varargin2S({
        'Ser', 'Serial'
        'Par', 'Parallel'
        'Exv', 'Exhaustive'
        'Trg', 'Targetwise'
        'InhDrift', sprintf('Signal\nSuppression')
        'InhFree', 'Flexible'
        });
end
methods
    %% Collapsing bound w/o changing sigmaSq w/ interaction
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Quad_Ixn(W0, varargin)
        S = varargin2S(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'QuadPreDrift'
            'fix_irr_ixn', false
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Quad_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Quad_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Quad_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Quad_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound w/o changing sigmaSq w/ interaction
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn(W0, varargin)
        S = varargin2S(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'LinearMinPreDrift'
            'fix_irr_ixn', false
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Linear_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Linear_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound and const sigmaSq w/ interaction
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(W0, varargin)
        S = varargin2S(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'Const'
            'fix_irr_ixn', false
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound w/o changing sigmaSq
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Const(W0, varargin)
        S = varargin2C(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'Const'
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Const(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Const(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound and changing sigmaSq
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Linear(W0, varargin)
        S = varargin2C(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'LinearMinPreDrift'
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Linear(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Linear(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Const bound, const sigmaSq, ixn
    function S = get_S_batch_fit_RT_Inh_Const_Const_Ixn(W0, varargin)
        S = varargin2C(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'LinearMinPreDrift'
            });
    end
    function batch_fit_RT_Inh_Const_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_Const_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_Const_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_Const_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% General
    function S_batch = get_S_batch_RT(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'parad', 'RT'
            'model', W0.models_RT
            });
    end
    function batch_plot_RT(W0, varargin)
        S_batch = W0.get_S_batch_RT(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_RT(C{:});
            W0.W_now = W;
            
            file = W.get_file;
            L = load(file);
            
            L.Fl.res2W;
            W = L.Fl.W;
            bml.oop.varargin2props(W, C, true);
            W.Fl = L.Fl;
            
            W.plot_and_save_all;
        end
    end
    function varargout = batch_fit_RT(W0, varargin)
        S_batch = W0.get_S_batch_RT(varargin{:});
        Ss = bml.args.factorizeS(S_batch);
        
        [varargout{1:nargout}] = W0.batch_Ss(Ss);
    end
    function Ls = batch_load_RT(W0, varargin)
        % Ls(file): Struct containing:
        % .Fl
        % .res
        
        S_batch = W0.get_S_batch_RT(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        for ii = n:-1:1
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_RT(C{:});
            file = W.get_file;
            fprintf('Loading (%d/%d): %s\n', ii, n, file);
            L1 = load([file, '.mat']);
            Ls(ii) = copyFields(S, L1, {'Fl', 'res'});
        end
    end    
    function W = create(W0, varargin)
        S = varargin2S(varargin, {
            'model_kind', 'RT' % 'RT'|'short'
            });
        C = S2C(S);
        switch S.model_kind
            case 'RT'
                W = W0.create_RT(C{:});
            case 'short'
                W = W0.create_sh(C{:});
        end
    end
end
%% Constraining Tnd
methods
    constrain_tnd_from_data(W)
end
%% Batch - Short
properties (Constant)
    models_sh = {'Ser', 'Par', 'InhDrift', 'InhFree', 'DurFree'};
end
methods
    function imgather_sh(W0, varargin)
        
    end
    function S_batch = get_S_batch_sh(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_sh
            'parad', 'sh'
            'model', W0.models_sh
            });
    end
    function batch_plot_sh(W0, varargin)
        S_batch = W0.get_S_batch_sh(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_sh(C{:});
            
            file = W.get_file;
            L = load(file);
            
            L.Fl.res2W;
            W = L.Fl.W;
            bml.oop.varargin2props(W, C, true);
            W.Fl = L.Fl;
            
            W.plot_and_save_all;
        end
    end
    function batch_fit_sh(W0, varargin)
        S_batch = W0.get_S_batch_sh(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_sh(C{:});
            W0.W_now = W;
            
            W.main;
        end
    end
    function W = create_sh(W0, varargin)
        W = W0.create_RT(varargin{:});
        
        S = varargin2S(varargin, {
            'model', 'Ser'
            'bound', 'Const'
            'fix_irr_ixn', true
            'fix_miss', false
            'fix_sigma', true
            'fix_fano', true
            'fano_max', 1
            ... Short paradigm specific
            'p_dim1_1st', 0.5
            'fix_p_dim1_1st', false
            'buffer_dur_sec', 0.12 - 4/75;
            'fix_drift_t_st', true
            'b_mean0', 0.3
            });        
        
        % Collapse bound early for prd=sh
        logit_b_mean0 = logit(S.b_mean0/5);
        b_asym0 = 0.1;
        if isfield(W.th0, 'Dtb__Bound1__b_logitmean')
            W.th0.Dtb__Bound1__b_logitmean = logit_b_mean0;
            W.th.Dtb__Bound1__b_logitmean = logit_b_mean0;
            W.th0.Dtb__Bound2__b_logitmean = logit_b_mean0;
            W.th.Dtb__Bound2__b_logitmean = logit_b_mean0;

            W.th0.Dtb__Bound1__b_asym = b_asym0;
            W.th.Dtb__Bound1__b_asym = b_asym0;
            W.th0.Dtb__Bound2__b_asym = b_asym0;
            W.th.Dtb__Bound2__b_asym = b_asym0;
            
        elseif isfield(W.th0, 'Dtb__Dtb1__Bound__b_logitmean')
            W.th0.Dtb__Dtb1__Bound__b_logitmean = logit_b_mean0;
            W.th.Dtb__Dtb1__Bound__b_logitmean = logit_b_mean0;
            W.th0.Dtb__Dtb2__Bound__b_logitmean = logit_b_mean0;
            W.th.Dtb__Dtb2__Bound__b_logitmean = logit_b_mean0;

            W.th0.Dtb__Dtb1__Bound__b_asym = b_asym0;
            W.th.Dtb__Dtb1__Bound__b_asym = b_asym0;
            W.th0.Dtb__Dtb2__Bound__b_asym = b_asym0;
            W.th.Dtb__Dtb2__Bound__b_asym = b_asym0;
        end        
        
        % Set UB of Tnd mean and variance based on minimum RTs
        W.constrain_tnd_from_data;
        
        % Fixing starting time of the decline of the drift
        if S.fix_drift_t_st
            if isa(W, 'Fit.D2.Inh.MainBatch')
                nams = {
                    'Dtb__Drift1__log10_t_st'
                    'Dtb__Drift2__log10_t_st'
                    }';
            else
                nams = {
                    'Dtb__Dtb1__Drift__log10_t_st'
                    'Dtb__Dtb2__Drift__log10_t_st'
                    }';
            end
            for nam = nams(:)'
                W.th.(nam{1}) = log10(S.buffer_dur_sec);
                W.fix_to_th_(nam{1});
            end
        end
    end
end
%% Table
methods
    function [ds_txt, file, Ls] = tabulate_files(W0, files)
        %%
        n = numel(files);
        Ls = cell(n, 1);
        for ii = 1:n
            file = files{ii};
            if ~exist(file, 'file')
                warning('%s is absent! skipping..\n', file);
                continue;
            end
            L = load(file, 'res', 'W');
            fprintf('Loaded %s\n', file);
            
            % Compute fval & BIC based on the validation set
            W = L.W;
            L.res.fval = W.get_cost_validation;
            
            
            %
            Ls{ii} = L;
        end
        
        %%
        Ss = cell(n, 1);
        for ii = 1:n
            Ss{ii} = Ls{ii}.W.get_S0_file;
        end
        
        %%
        txts = cell(n, 1);
        for ii = 1:numel(files)
            L = Ls{ii};
            if isempty(L)
                continue; 
            end
            
            txt = struct;
            txt = W0.tabulate_metainfo(txt, L);
            txt.file = files{ii};

            txts{ii} = txt;
        end
        ds_txt = bml.ds.from_Ss(txts);
        ds_txt = cell2mat2_ds(ds_txt);
        
        %% Best models
        S0 = Ss{1};
        ic = 'BIC';
        subjs = unique(ds_txt.subj);
        parads = unique(ds_txt.parad);
        
        i_res = 0;
        ds_best = dataset;
        ix_best = [];
        
        for i_subj = 1:numel(subjs)
            subj = subjs{i_subj};
            for i_parad = 1:numel(parads)
                parad = parads{i_parad};
                
                incl = find(strcmp(ds_txt.subj, subj) ...
                    & strcmp(ds_txt.parad, parad));
                [min_ic, min_ix] = min(ds_txt.(ic)(incl));
                min_ix = incl(min_ix);
                
                ds_txt.(['delta_' ic])(incl,1) = ds_txt.(ic)(incl) - min_ic;
                
                i_res = i_res + 1;
                ds_best(i_res,:) = ds_txt(min_ix, :);
                ix_best(i_res) = min_ix; %#ok<AGROW>
            end
        end
        n_res = i_res;
        
        %% Add params to the best models
        txt_best = cell(n_res, 1);
        for i_res = 1:n_res
            ix_best1 = ix_best(i_res);
            txt = struct;
            
            L = Ls{ix_best1};
            res = L.res;
            th_names = fieldnames(res.th)';
            for jj = 1:numel(th_names)
                th_name = th_names{jj};
                
                txt = W0.tabulate_param(txt, th_name, L);
            end
            txt_best{i_res} = txt;
        end
        ds_best_param = bml.ds.from_Ss(txt_best);
        ds_best = [ds_best, ds_best_param];
        
        %%
        S0_fields = fieldnames(Ss{1})';
        for f = S0_fields
            S0.(f{1}) = unique(ds_txt.(f{1}));
        end
        file = W0.get_file_from_S0(S0, {
            'tbl', 'all'
            });
        
        export(ds_txt, 'File', [file '.csv'], 'Delimiter', ';');
        fprintf('Exported fit results to %s.csv\n', file);
        
        %%
        file = W0.get_file_from_S0(S0, {
            'tbl', 'best'
            });
        
        export(ds_best, 'File', [file '.csv'], 'Delimiter', ';');
        fprintf('Exported best fit results to %s.csv\n', file);        
    end
    function txt = tabulate_metainfo(~, txt, L)
        
        S0_file = L.W.get_S0_file;
        txt = copyFields(txt, S0_file);
        
        res = L.res;
        txt.NParam = res.k;
        txt.fval = res.fval;
        txt.BIC = res.bic;
    end
    function txt = tabulate_param(~, txt, th_name, L)
        res = L.res;
        th = res.th.(th_name);
        se = res.se.(th_name);
        if isscalar(th)
            txt.(th_name) = sprintf('%1.3g +- %1.3g', ...
                th, se);
        else
            for ii = 1:numel(th)
                th_name1 = sprintf('%s_%d', th_name, ii);
                txt.(th_name1) = sprintf('%1.3g +- %1.3g', ...
                    th(ii), se(ii));
            end
        end
    end
end
%% Plot - Goodness of Fit
methods
    function axs = imgather_RT_wi_subj(W0, varargin)
        %
        S = W0.get_S_batch_RT(varargin{:});
        S.plot = {
            {
                'plt', 'ch'
                'dX', 1
            }
            {
                'plt', 'ch'
                'dX', 2
            }
            {
                'plt', 'rt'
                'dX', 1
            }
            {
                'plt', 'rt'
                'dX', 2
            }
            };
        n_subj = numel(S.subj);
        n_model = numel(S.model);
        n_plot = numel(S.plot);
        
        %
        for i_subj = n_subj:-1:1
            subj = S.subj{i_subj};
            
            fig_tag(subj);
            clf;
            
            for i_model = n_model:-1:1
                model = S.model{i_model};
                
                W = W0.create_RT('subj', subj, 'model', model);
                
                for i_plot = n_plot:-1:1
                    S_plot = varargin2S(S.plot{i_plot});
                    C_plot = varargin2C( ...
                        copyFields(struct, S_plot, {
                            'plt', 'dX'
                        }));
                    file = [W.get_file(C_plot), '.fig'];
                
                    ax1 = subplotRC(n_model, n_plot, i_model, i_plot);
                    ax1 = bml.plot.openfig_to_axes(file, ax1);
                
                    ax(i_model, i_plot) = ax1;
                end
            end
            axs{i_subj} = ax;
        end
        
        %
        for i_subj = n_subj:-1:1
            subj = S.subj{i_subj};            
            fig_tag(subj);
            ax = axs{i_subj};
            
            for i_model = n_model:-1:1
                model = S.model{i_model};
                
                for i_plot = n_plot:-1:1
                    S_plot = varargin2S(S.plot{i_plot});
                    ax1 = ax(i_model, i_plot);
                    
                    title(ax1, '');

                    if i_plot == 1
                        col_title = W0.models_RT_long.(model);
                        if i_model == n_model
                            ylab = sprintf('%s\nP_{right}', col_title);
                        else
                            ylab = sprintf('%s\n _{ }', col_title);
                        end
                    else
                        if i_model == n_model
                            switch S_plot.plt
                                case 'ch'
                                    if S_plot.dX == 1
                                        ylab = 'P_{right}';
                                    else
                                        ylab = 'P_{blue}';
                                    end
                                case 'rt'
                                    if S_plot.dX == 1
                                        ylab = 'RT (s)';
                                    else
                                        ylab = ' ';
                                    end
                            end
                        else
                            ylab = '';
                        end
                    end
                    ylabel(ax1, ylab);
                    
                    if mod(i_plot, 2) == 0
                        set(ax1, 'YTickLabel', {''});
                    end
                    
                    xticks = cellstr(get(ax1, 'XTickLabel'));
                    if i_model == n_model
                        xticks(2:2:end) = {''};
                    else
                        xticks(:) = {''};
                        xlabel(ax1, '');
                    end
                    set(ax1, 'XTickLabel', xticks);
                    
                    h = bml.plot.children2struct(ax1);
                    set(h.marker, 'MarkerSize', 4);
                    set(h.marker, 'LineWidth', 0.25);
                    set(ax1, 'FontSize', 9);
                end
            end
            
            bml.plot.position_subplots(ax, ...
                'margin_top', 0.02, ...
                'margin_left', 0.125, ...
                'margin_bottom', 0.06, ...
                'btw_row', 0.015, ...
                'btw_col', [0.05, 0.075, 0.05]);
            
            %
            file = W0.get_file({
                'sbj', subj
                'mdl', S.model
                'plt', 'ch_rt'
                });
            bml.plot.savefigs(file, ...
                'PaperPosition', [0, 0, ...
                    Fig.Consts.width_column2_cm, ...
                    n_model * 3.5], ...
                'ext', {'.fig', '.png', '.tif'}); % [600, n_subj * 400]);
        end
    end
    function plot_gof_wi_subj(W0, Ls, varargin)
        S = varargin2S(varargin, {
            'gof', 'bic'
            'gof_label', ''
            });
        
        subjs_all = {Ls.subj};
        subjs = unique(subjs_all);
        n_subj = numel(subjs);
        for i_subj = 1:n_subj
            subj = subjs{i_subj};
    
            clf;
            W0.plot_gof_wi_subj_unit(Ls, subj, varargin{:});
            
            file = W0.get_file({
                'sbj', subj
                'plt', 'gof'
                'gof', S.gof});
            Fig.savefigs_column2(file, n_model * 3.5);
        end
    end
    function imgather_gof(W0, Ls, varargin)
        if iscell(Ls)
            Ls0 = Ls;
            clear Ls
            n = numel(Ls0);
            for f = {'subj', 'parad', 'model', 'Fl', 'res'}
                for ii = n:-1:1
                    Ls(ii).(f{1}) = Ls0{ii}.(f{1});
                end
            end
        end
        
        S = varargin2S(varargin, {
            'gof', 'bic'
            'gof_label', ''
            });
        
        subjs_all = {Ls.subj};
        subjs = unique(subjs_all);
        n_subj = numel(subjs);
        
        clf;
        for i_subj = n_subj:-1:1
            subj = subjs{i_subj};
    
            ax1 = subplotRC(1, n_subj, 1, i_subj);
            [~, dgof{i_subj}] = ...
                W0.plot_gof_wi_subj_unit(Ls, subj, varargin{:});
            
            ax(1, i_subj) = ax1;
        end
        
        bml.plot.position_subplots(ax, ...
            'margin_top', 0.18, ...
            'margin_left', 0.08, ...
            'margin_right', 0.04, ...
            'margin_bottom', 0.22);
        
        for i_subj = 1:n_subj
            ax1 = ax(1, i_subj);
            
            set(ax1, 'FontSize', 9, 'TickLength', [0.015, 0.01]);
            
            if i_subj > 1
                set(ax1, 'YTick', []);
            end
            if i_subj ~= round((n_subj + 1) / 2)
                xlabel(ax1, '');
            end
            
            dgof_sorted = sort(dgof{i_subj});
            
            bnd = 5;
            
            xlim_small_max = min( ...
                ceil(dgof_sorted(end-bnd) / 10) * 12, ...
                dgof_sorted(end) - 1);
            xlim_large_min = max( ...
                floor(dgof_sorted(end) / 100) * 50, ...
                dgof_sorted(end-1) + 1);

            [h, hch] = bml.plot.break_axis(ax1, 'x', ...
                xlim_small_max, xlim_large_min);
            
            if i_subj == round((n_subj + 1) / 2)
                hlabel = hch.label;
                hlabelpos = get(hlabel, 'Position');
                hlabelylim = get(h(3), 'YLim');
                hlabelpos(2) = ...
                    (hlabelpos(2) - hlabelylim(1)) * 1.07 ...
                    + hlabelylim(1);
                set(hlabel, 'Position', hlabelpos);
%                 str_label = get(hlabel, 'String');
%                 delete(hlabel);
%                 xlabel(h(3), sprintf('\n%s', str_label));
            end
            
            xtick = get(h(1), 'XTick');
            set(h(1), 'XTick', xtick([1, end]));
%             set(h(1), 'XTick', xtick([1, max(end - 1, 2)]));
            
            xtick = get(h(2), 'XTick');
            set(h(2), 'XTick', xtick([1, end]));
%             set(h(2), 'XTick', xtick([min(2, end - 1), end]));
        end
        
        file = W0.get_file({
            'sbj', subjs
            'plt', 'gof'
            'gof', S.gof});
        Fig.savefigs_column2(file, 5);
    end
    function [h, dgof] = plot_gof_wi_subj_unit(W0, Ls, subj, varargin)
        S = varargin2S(varargin, {
            'gof', 'bic'
            'gof_label', ''
            });        
        if isempty(S.gof_label)
            S.gof_label = upper(S.gof);
        end
        
        ax1 = gca;
        
        subjs_all = {Ls.subj};
        incl = strcmp(subj, subjs_all);
        Ls1 = Ls(incl);

        models = {Ls1.model};

        n_models = numel(Ls1);
        gof = zeros(n_models, 1);
        for i_model = 1:n_models
            L = Ls1(i_model);
            gof(i_model) = L.res.(S.gof);
        end

        dgof = gof - min(gof);
        h = barh(1:n_models, dgof, 'k');
        set(ax1, 'YTickLabel', models);
        set(ax1, 'YDir', 'reverse');
        
        xlabel(['\Delta ' S.gof_label]);
        bml.plot.beautify;

        title(sprintf('Subject %s\n', subj(1)));        
    end
end
%% PlotFun
methods
    function add_plotfun(W, Fl)
        W.add_plotfun@Fit.D2.Common.CommonWorkspace(Fl);
        Fit.D2.Common.Plot.PlotFuns.add_plotfun(Fl);
    end
end
%% Comparison plot
methods
    function plot_cost_distrib_all_diff_models(W0, ...
            batch_args, model_args)
        
        if ~exist('batch_args', 'var')
            batch_args = {};
        end
        S_batch = varargin2S(batch_args, {
            'subj', {'DX', 'MA', 'VL'}
            });
        [Ss, n_batch] = bml.args.factorizeS(S_batch);
        
        if ~exist('model_args', 'var')
            model_args = {
                W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn('model', 'Ser')
                W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn('model', 'Par')
                };
        end
        n_model = numel(model_args);
        
        for i_batch = 1:n_batch
            Ls = cell(1, n_model);
            for i_model = 1:n_model
                S = Ss(i_batch);
                S_model = varargin2S(model_args{i_model});
                C = varargin2C(S, S_model);
                
                W = W0.create_RT(C{:});
                Ls{i_model} = load(W.get_file);
                
                if i_model >= 2
                    W = Ls{1}.Fl.W;
                    W2 = Ls{i_model}.Fl.W;
                    
                    clf;
                    W.plot_cost_distrib_all_diff(W2);
                    file = W.get_file({
                        'plt', 'cstdf'
                        'mdl2', S_model.model
                        });
                    savefigs(file, 'size', [900, 900]);
                end
            end
        end
    end
end
%% Plot
methods
    function plot_and_save_all(W, varargin)
        S0 = varargin2S(varargin, {
            'conds_oversample_factor', 1 % 10
            'subdir', class(W)
            'remove_fields', {}
            'add_fields', {}
            });
        
        if is_in_parallel
            warning('Cannot plot when in parallel pool!');
            return;
        end
        
        opt_file = {S0.remove_fields, S0.subdir};
        
        for dimOnX = 1:2
            W.Data.set_conds_oversample_factor( ...
                1, ...
                dimOnX);
        end

        is_estimated = false;
        
        is_to_plot = @(v) any(ismember(vVec(v), W.to_plot_kind)) ...
            || isequal(W.to_plot_kind, 'all');
        
        % Plots without oversample
        kinds = {
                'plotfuns',  'plotfuns', {'size', [1200, 800]}
                'rt_distrib_all', 'rtdst', {'size', [800, 800]}
                };
        for kind = kinds'
            [kind_long, kind_short, savefigs_args] = deal(kind{:});
            
            if ~is_to_plot({kind_long, kind_short})
                continue;
            end
            
            file = W.get_file([
                S0.add_fields; {
                'plt', kind_short}], opt_file{:});

            if exist([file, '.fig'], 'file') && W.skip_existing_fig
                fprintf('Skipping existing figure: %s\n', [file, '.fig']);
            else
                if ~is_estimated
                    W.Fl.res2W;
                    is_estimated = true;
                end
                
                try
                    clf;
                    W.(['plot_' kind_long]);
                    savefigs(file, savefigs_args{:});
                catch err
                    warning(err_msg(err));
                end
            end
        end
        
        % mean/var plots
        [Ss, n] = bml.args.factorizeS(varargin2S({
            'dim_on_x', {1, 2}
            'yfun', {'mean', 'var'}
            }));
        
        for ii = 1:n
            S = varargin2S(Ss(ii), S0);
            C = S2C(S);
            
            clf;
            W.plot_rt_vs_rt(C{:});
            
            file = W.get_file([
                S0.add_fields; {
                'dimx', S.dim_on_x
                'yfun', S.yfun}], ...
                opt_file{:});
            
            savefigs(file);
        end
        
        % Plots with oversample
        kinds = {
            'ch',        'ch'
            'rt',        'rt'
            ... 'rt_stdev',  'rtsd'
            ... 'rt_skew',   'rtsk'
            };
        
        if is_to_plot(kinds(:,1:2))
            for dimOnX = 1:2
                W.Data.set_conds_oversample_factor( ...
                    S.conds_oversample_factor, ...
                    dimOnX);
                W.pred;

                for kind = kinds'

                    [kind_long, kind_short] = deal(kind{:});

                    if ~is_to_plot({kind_long, kind_short})
                        continue;
                    end

                    file = W.get_file([
                        S0.add_fields; {
                        'plt', kind_short
                        'dX', dimOnX}], ...
                        opt_file{:});
                    if exist([file, '.fig'], 'file') && W.skip_existing_fig
                        fprintf('Skipping existing figure: %s\n', ...
                            [file, '.fig']);
                    else
                        try
                            clf;
                            W.(['plot_' kind_long])('dimOnX', dimOnX);
                            title(sprintf('%s-dimOnX=%d', ...
                                strrep(kind_long, '_', '-'), ...
                                dimOnX));
                            savefigs(file);
                        catch err
                            warning(err_msg(err));
                        end
                    end
                end
            end
        end        
    end
    function plot_plotfuns(W)
        W.get_Fl;
        W.Fl.runPlotFcns;
    end
    function varargout = plot_ch(W, varargin)
        Pl = DtbPlot.PlotCh2D;
        [varargout{1:nargout}] = Pl.plot_W_pred_data(W, varargin{:});
    end
    function varargout = plot_rt(W, varargin)
        Pl = DtbPlot.PlotRt2D;
        [varargout{1:nargout}] = Pl.plot_W_pred_data(W, varargin{:});
    end
    function varargout = plot_rt_wrong(W, varargin)
        C = varargin2C(varargin, {
            'accuOnlyAxis', [2, 0]
            });
        Pl = DtbPlot.PlotRt2D;
        [varargout{1:nargout}] = Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_rt_distrib_all(W, varargin)
        C = varargin2C(varargin, {
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_rt_distrib_cum_all(W, varargin)
        C = varargin2C(varargin, {
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data_cum(W, C{:});
    end
    function varargout = plot_rt_distrib_all_data(W, varargin)
        C = varargin2C(varargin, {
            'src', {'data'}
            'use_bias', false
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_rt_distrib_all_pred(W, varargin)
        % [hd, hp, Pl_d, Pl_p] = plot_rt_distrib_all_pred(W, ...)
        C = varargin2C(varargin, {
            'src', {'pred'}
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_cost_distrib_all(W, varargin)
        C = varargin2C(varargin, {
            'src', 'cost'
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W(W, C{:});
    end
    function varargout = plot_cost_distrib_all_diff(W, W2, varargin)
        C = varargin2C(varargin, {
            'src', 'cost_dif'
            'W2', W2
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W(W, C{:});
    end
    function varargout = plot_cost_distrib_all_diff_cum(W, W2, varargin)
        C = varargin2C(varargin, {
            'src', 'cost_dif_cum'
            'W2', W2
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W(W, C{:});
    end
    function varargout = plot_rt_d1_vs_d2(W, varargin)
        S = varargin2S(varargin, {
            'abs_cond', true
            'fun', 'mean'
            'yfun', [] % @mean
            'efun', [] % @sem
            });
        switch S.fun
            case 'mean'
                S.yfun = @mean;
                S.efun = @sem;
            case 'var'
                S.yfun = @var;
                S.efun = @sev;
            otherwise
                error('Unknown fun=%s\n', S.fun);
        end
        
        %%
        cond = W.Data.ds.cond;
        rt = W.Data.ds.RT;
        
        %%
        if S.abs_cond
            cond = abs(cond);
        end
        n_dim = size(cond, 2);
        n_tr = size(cond, 1);
        d_cond = zeros(n_tr, n_dim);
        for dim = n_dim:-1:1
            [~,~,d_cond(:,dim)] = unique(cond(:,dim));
        end
        
        %%
        y = accumarray(d_cond, rt, [], S.yfun, nan);
        e = accumarray(d_cond, rt, [], S.efun, nan);
        
        %%
%         y = bsxfun(@minus, y, y(:,1));
        
        %%
        n_line = size(y, 2);
        colors = hsv2rev(n_line);
        
        for i_line = 1:n_line
            errorbar(y(:,end), y(:,i_line), e(:,i_line), ...
                'o-', 'Color', colors(i_line,:));
            hold on;
        end
        hold off;
        axis equal
        grid on;
        bml.plot.beautify;
        
        xlabel(S.fun);
        ylabel(S.fun);
        title(W.get_title);
    end
    function [h_pred, h_data] = plot_rt_vs_rt(W, varargin)
        S = varargin2S(varargin, {
            'dim_on_x', 1
            'yfun', 'mean' % 'mean'|'var'
            'accu_only', true % f
            'to_plot_last_frame', false
            'ch_incl', []
            'p_data', []
            'p_pred', []
            'to_use_to_excl', true
            });
        
        if isempty(S.p_data)
            rt_data_pdf = W.Data.RT_data_pdf;
        else
            rt_data_pdf = S.p_data;
        end
        if isempty(S.p_pred)
            rt_pred_pdf = W.Data.RT_pred_pdf;
        else
            rt_pred_pdf = S.p_pred;
        end
        
        if S.to_use_to_excl
            % Exclude those that are excluded from the fit
            to_excl = W.cond_ch_to_exclude;

            % Stretch on time axis
            to_excl = repmat( ...
                permute(to_excl, [5, 1, 2, 3, 4]), ...
                [W.nt, 1, 1, 1, 1]);

            rt_data_pdf(to_excl) = 0;
            rt_pred_pdf(to_excl) = 0;
        end
        
        [h_pred,~,~,y1,e1] = W.plot_rt_vs_rt_unit(rt_pred_pdf, varargin{:}, ...
            'style', 'pred');
        hold on;
        
        [h_data,~,~,y2,e2] = W.plot_rt_vs_rt_unit(rt_data_pdf, varargin{:}, ...
            'style', 'data');
        cellfun(@(h) set(h, 'LineStyle', 'none'), h_data);
        hold on;
        
        axis equal;
        
        ye = [y1; y2-e2; y2+e2];
        xy = [ye(:,1), ye(:,end)];
        
%         xy = bml.plot.get_all_xy(gca);
        x_min = min(xy(:,1));
        x_max = max(xy(:,1));
        x_mid = (x_min + x_max) / 2;
        x_range = x_max - x_min;
        
        y_min = min(xy(:,2));
        y_max = max(xy(:,2));
        y_mid = (y_min + y_max) / 2;
        y_range = y_max - y_min;
        
        xy_range = max(x_range, y_range);
        
        x_lim = [x_mid - xy_range * 0.6, x_mid + xy_range * 0.6];
        xlim(x_lim);
        y_lim = [y_mid - xy_range * 0.6, y_mid + xy_range * 0.6];
        ylim(y_lim);
        
        bml.plot.beautify_tick(gca, 'X');
        bml.plot.beautify_tick(gca, 'Y');

        hold off;
        
        bml.plot.beautify;
    end
    function [h, hxe, hye, y, e] = plot_rt_vs_rt_unit(W, p, varargin)
        S = varargin2S(varargin, {
            'dim_on_x', 1
            'group_y', {0, 1:2} % 1:2} % 1:3}
            % accu_only defaults to false because conditions & ch
            % are already filtered by cond_ch_to_exclude and
            % to_exclude_bins_wo_trials
            'accu_only', true % f
            'yfun', 'mean' % 'mean'|'var'
            'style', 'data' % 'data'|'pred'
            'to_plot_last_frame', false
            'ch_incl', [] % if nonempty, shows only [ch1, ch2]
            'to_plot', true
            });

        n_ch = 2;
        if S.dim_on_x == 2
            p = permute(p, [1 3 2 5 4]);
        end
            
        if ~W.to_include_last_frame && strcmp(S.style, 'pred')
            p = W.set_last_frame_0_and_normalize(p);
        end        
        incl_t = true(size(p,1), 1);
        if ~S.to_plot_last_frame
            incl_t(end) = false;
        end
        p = p(incl_t, :,:,:,:);
        
        n_cond1 = size(p, 2);
        n_cond2 = size(p, 3);
        
        [~, ~, cond_plot_x] = unique(abs((1:n_cond1) - ((1 + n_cond1) / 2)));
        [~, ~, cond_plot_y] = unique(abs((1:n_cond2) - ((1 + n_cond2) / 2)));

        group_y = S.group_y;
        n_group_y = length(group_y);        
        n_cond_plot_x = max(cond_plot_x);

        if S.accu_only
            for cond1 = 1:n_cond1
                for cond2 = 1:n_cond2
                    for ch1 = 1:n_ch
                        for ch2 = 1:n_ch
                            if sign(ch1 - 1.5) == ...
                                    -sign(cond1 - (1 + n_cond1) / 2) ...
                                    || sign(ch2 - 1.5) ...
                                        == -sign(cond2 - (1 + n_cond2) / 2)
                                p(:,cond1,cond2,ch1,ch2) = 0;
                            end
                        end
                    end
                end
            end
        end
        if ~isempty(S.ch_incl)
            p(:,:,:, setdiff(1:2,S.ch_incl(1)), :) = 0;
            p(:,:,:, :, setdiff(1:2,S.ch_incl(2))) = 0;
        end
        
        y = zeros(n_cond_plot_x, n_group_y);
        e = zeros(n_cond_plot_x, n_group_y);
        t = vVec(W.t(incl_t));
        
        n_sep = max(cond_plot_y);
        for ii = 1:numel(group_y)
            group_y1 = group_y{ii};
            is_neg = group_y1 <= 0;
            group_y1(is_neg) = n_sep + group_y1(is_neg);
            group_y{ii} = group_y1;
        end
       
        % Pooling probability
        for cond_x = 1:n_cond_plot_x
            for i_group = 1:n_group_y
                cond_x_incl = cond_plot_x == cond_x;
                cond_y_incl = bsxEq(cond_plot_y(:), group_y{i_group}(:)');
                
                
                cond_y_incl = find(cond_y_incl);

                p1 = p(:, cond_x_incl, cond_y_incl, :, :); % Use this

                p1(isnan(p1)) = 0;
                p1 = sums(p1, 2:5, true);
%                 p1 = sums(p(:, cond_x_incl, cond_y_incl, :, :), 2:5, true);
                
                switch S.yfun
                    case 'mean'
                        [e1, y1] = bml.stat.sem_distrib(p1, t);
                        
                    case 'var'
                        [e1, y1] = bml.stat.sev_distrib(p1, t);
                end
                
                e(cond_x, i_group) = e1;
                y(cond_x, i_group) = y1;
                n_tr(cond_x, i_group) = sum(p1(:));
            end
        end
                
        if ~S.to_plot
            h = {};
            hxe = {};
            hye = {};
            return;
        end
        for group = n_group_y:-1:2
            [h1, hxe1, hye1] = ...
                bml.plot.errorbar_wo_tick2( ...
                    y(:,1), y(:,group), ...
                    e(:,1), [], ...
                    e(:,group), []);
            hold on;
                
            switch S.style
                case 'data'
                    h0 = h1;
                    h1 = bml.plot.color_points(h0, [], 'colors', @hsv2rev);
                    delete(h0);
                    hold on;
                    
                case 'pred'
                    set(h1, 'Marker', 'none'); % 'x'); % 
                    set(hxe1, 'LineStyle', 'none');
                    set(hye1, 'LineStyle', 'none');
            end
            h{group - 1} = h1;
            hxe{group - 1} = hxe1;
            hye{group - 1} = hye1;
        end
        hold off;
        
        dim_names = {'M', 'C'};
        dim_name = dim_names{3 - S.dim_on_x};
        
        switch S.yfun
            case 'mean'
                xlabel(sprintf('RT_{easy %s} (s)', dim_name));
                ylabel(sprintf('RT_{hard %s} (s)', dim_name));
            case 'var'
                xlabel(sprintf('Var RT_{easy %s} (s^2)', dim_name));
                ylabel(sprintf('Var RT_{hard %s} (s^2)', dim_name));
            otherwise
                error('Unknown yfun=%d\n', S.yfun);
        end
        
        switch S.style
            case 'data'
                y_hard_minus_easy_easiest = y(end,2) - y(end,1);
                bml.plot.crossLine('NE', y_hard_minus_easy_easiest, ...
                    {'--', 0.7 + zeros(1,3)})
        end
        axis equal;
    end
    function varargout = plot_rt_mean_vs_var(W, varargin)
        S = varargin2S(varargin, {
            'abs_cond', true
            'xfun', @mean
            'yfun', @var
            'efun', @sev
            });
        
        %%
        cond = W.Data.ds.cond;
        rt = W.Data.ds.RT;
        
        %%
        if S.abs_cond
            cond = abs(cond);
        end
        n_dim = size(cond, 2);
        n_tr = size(cond, 1);
        d_cond = zeros(n_tr, n_dim);
        for dim = n_dim:-1:1
            [~,~,d_cond(:,dim)] = unique(cond(:,dim));
        end
        
        %%
        x = accumarray(d_cond, rt, [], S.xfun, nan);
        y = accumarray(d_cond, rt, [], S.yfun, nan);
        e = accumarray(d_cond, rt, [], S.efun, nan);
        
        %%
%         y = bsxfun(@minus, y, y(:,1));
        
        %%
        n_line = size(y, 2);
        colors = hsv2rev(n_line);
        
        for i_line = 1:n_line
            errorbar(x(:,i_line), y(:,i_line), e(:,i_line), ...
                'o-', 'Color', colors(i_line,:));
            hold on;
        end
        hold off;
        axis equal
        
    end
    function set.to_plot_kind(W, v)
        if ischar(v) || ~strcmp(v, 'all')
            v = {v};
        else
            assert(iscell(v) && all(cellfun(@ischar, v(:))));
        end
        W.to_plot_kind_ = v;
    end
    function v = get.to_plot_kind(W)
        v = W.to_plot_kind_;
    end
end
%% Bias
properties
    cond_bias
end
methods
    function b = get.cond_bias(W)
        % b(dim) : bias. Used in plotting, etc.
        b = W.get_cond_bias;
    end
    function b = get_cond_bias(~)
        warning('Implement in subclasses!');
        b = {0, 0};
    end
end
%% File
methods
    function v = get_file_fields0(W)
        v = union_general( ...
                W.get_file_fields0@Fit.Common.Main, ...
                W.get_file_fields0@Fit.D2.Common.CommonWorkspace, ...
            'stable', 'rows');            
        v = union_general(v, ...
            {
            'to_use_easiest_only_for_fit', 'ef'
            'to_use_easiest_only_for_comparison', 'ec'
            'to_include_last_frame', 'lf'
            'to_exclude_bins_wo_trials', 'eb'
            'fix_kappa', 'fk'
            }, 'stable', 'rows');            
    end
end
end