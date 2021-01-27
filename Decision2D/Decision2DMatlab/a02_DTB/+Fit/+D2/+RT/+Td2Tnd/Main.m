classdef Main < Fit.D2.Common.Main
    % Fits Tnd and predict the rest.
    %
    % Fit.D2.Td2Tnd.Main
    
    % 2016 (c) Yul Kang. hk2699 at columbia dot edu.    

%% Settings - Fit
properties
    td_kind = 'Ser'; % 'Ser'|'Par'
    
    to_fold_cond = [];
    
    max_sd_mu_ratio = []; % sd/mu of td. if [], set to 1.
    ub_mu_margin = []; % empty (1) or scalar, compared to data
    ub_sd_margin = []; % empty (1) or scalar, compared to data
    
%     % to_use_easiest_only: Inherited from Fit.D2.Common.Main
%     % : If +1, calculate cost from the easiest conditions only.
%     %   If 0, calculate cost from all conditions.
%     %   If -1, calculate cost from non-easiest conditions only.
%     to_use_easiest_only = 1;
    
%     to_fit = true;
end
%% Intermediate variables
properties
    mean_td = {}; % {dim}(cond)
    std_td = []; % {dim}(cond)
    cost_td = []; % (cond1, cond2, ch1, ch2)
    
    Td
    Tnd
%     Miss
end
properties (Dependent)
    cond_ch_to_exclude_jt
    ub_mu_margin_val
    ub_sd_margin_val
end
%% Batch
methods
    function S_batch = get_S_batch(~, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            ... 'td_kind', {'Ser', 'Par'}
            });
        % Make td_kind vary first.
        if ~isfield(S_batch, 'td_kind')
            S_batch.td_kind = {'Ser', 'Par'};
        end
    end
    function batch(W0, varargin)
        varargin2props(W0, varargin, true);
        
        S = varargin2S(varargin, {
            'MaxIter', 1e4 % 1e4 for real | 0 for testing
            'MaxFunEvals', 1e6 %
            'UseParallel', 'always' % 'always'|'never'
            });            
        
        S_batch = W0.get_S_batch(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        disp('Ss:');
        disp(struct2dataset(Ss(:)));
        
        for ii = 1:n
            C = S2C(Ss(ii));
            W = feval(class(W0), C{:});
            W0.W_now = W;
            W.main( ...
                'MaxIter', S.MaxIter, ...
                'MaxFunEvals', S.MaxFunEvals, ...
                'UseParallel', S.UseParallel);
        end
%         W0.batch_postprocess;
    end
    function batch_postprocess(W0)
        W0.imgather;
        W0.compare_models;
    end
    function batch_plot(W0, varargin)
%         % Component plots: if not drawn already
%         S_batch = W0.get_S_batch(varargin{:}, 'to_plot', false);
%         [Ss, n] = bml.args.factorizeS(S_batch);
%         for ii = 1:n
%             C = S2C(Ss(ii));
%             W = feval(class(W0), C{:});
%             file = [W.get_file, '.mat'];
%             
%             if ~exist(file, 'file')
%                 warning('File absent: %s\n', file);
%                 continue;
%             end
%             
%             L = load(file);
%             L.Fl.res2W;
%             L.Fl.W.plot_and_save_all;
%         end

        % Imgather
        varargin2props(W0, varargin, true);
        W0.imgather(varargin{:});
    end
end
%% Compare models
methods
    function compare_models(W0, varargin)
        S0 = varargin2S(varargin, {
            'force_to_use_easiest_only_for_comparison', []
            });
        
        %% Get file names
        C = varargin2C(varargin, ...
            rmfield(W0.S0_file, {'subj', 'td_kind'}));
        varargin2props(W0, C, true);
        
        S_batch = W0.get_S_batch(C{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        files = cell(n, 1);
        
        W = feval(class(W0));
        W.init;
        W0.W_now = W;
        for ii = 1:n
            C = S2C(Ss(ii));
            varargin2props(W, C, true);

            file = W.get_file;
            files{ii} = file;
        end
        
        %% Load files
        is_absent = false(n, 1);
        for ii = 1:n
            file1 = [files{ii}, '.mat'];
            if ~exist(file1, 'file')
                warning('%s does not exist!\n', file1);
                is_absent(ii) = true;
                continue;
            end
        end
        if any(is_absent)
            fprintf('Files absent:\n');
            for ii = hVec(find(is_absent))
                disp(files{ii});
            end
%             return;
        end
        
        Ls = cell(n, 1);
        for ii = 1:n
            file1 = [files{ii}, '.mat'];
            if ~is_absent(ii)
                Ls{ii} = load(file1);
                fprintf('Loaded %s\n', file1);
            end
        end
        
%         Ls = cell2mat(Ls);
%         ress = [Ls.res];
        
        %%
        n_subj = numel(S_batch.subj);
        n_td = numel(S_batch.td_kind);
        subjs = {Ss.subj};
        models = {Ss.td_kind};
        cost = nan(n_subj, n_td);
        
        for i_subj = n_subj:-1:1
            for i_td = n_td:-1:1
                ix = strcmp(S_batch.subj{i_subj}, subjs) ...
                    & strcmp(S_batch.td_kind{i_td}, models);
                
                if isempty(Ls{ix})
                    continue;
                end
                    
                W = Ls{ix}.W;
                to_use_easiest_only0 = W.to_use_easiest_only;
                if ~isempty(S0.force_to_use_easiest_only_for_comparison)
                    W.to_use_easiest_only = ...
                        S0.force_to_use_easiest_only_for_comparison;
                else
                    W.to_use_easiest_only = ...
                        W0.to_use_easiest_only_for_comparison;
                end
                cost1 = W.get_cost;
                W.to_use_easiest_only = to_use_easiest_only0;
                
                cost(i_subj, i_td) = cost1;
            end
        end
        dcost = cost(:,2) - cost(:,1);
        
        %%
        C = {
            'sbj', sprintf('%s-%s', S_batch.subj{1}, S_batch.subj{end})
            'td', S_batch.td_kind
            'comp', 'fval'
            };
        if ~isempty(S0.force_to_use_easiest_only_for_comparison)
            C = varargin2C({
                'ec', ...
                    S0.force_to_use_easiest_only_for_comparison
                }, C);
        end
        file_comp = W0.get_file(C);
        
        %%
        ds_cost = cell2ds2([
            {'subj'}, S_batch.td_kind(:)', {'dcost', 'BayesFactor'}
            S_batch.subj(:), num2cell([cost, dcost, exp(dcost)])
            ]);
        disp(ds_cost);
        export(ds_cost, 'File', [file_comp, '.csv'], 'Delimiter', ',');
        fprintf('Saved to %s.csv\n', file_comp);
        
        save([file_comp, '.mat'], ...
            'S_batch', 'cost', 'ds_cost', 'models', 'subjs');
        fprintf('Saved to %s.mat\n', file_comp);
    end
end
%% Main
methods
    function main(W, varargin)
        file = W.get_file;
        if W.skip_existing_mat && ...
                exist([file, '.mat'], 'file')
            fprintf('Skipping existing fit: %s.mat\n', file);
            return;
        end
        if W.to_fit
            fprintf('Start fitting: %s.mat\n', file);
            
            C = varargin2C(varargin, {
                'MaxIter', 1e4 % 1e4|0
                'MaxFunEvals', 1e6
                'UseParallel', 'always' % 'always'|'never'
                });
            W.fit('opts', C);
        end
        if W.to_plot && ~is_in_parallel
            try
                W.plot_and_save_all;
            catch err
                warning(err_msg(err));
            end
        end
    end
end
%% Fit
methods
    function W = Main(varargin)
        W.to_exclude_bins_wo_trials = 10;
        W.set_Miss;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.Common.Main(varargin{:});
%         bml.oop.varargin2props(W, varargin, true);

        %%
        S = varargin2S(varargin, {
            'mu0_tnd_proportion', 1/3
            'var0_tnd_proportion', 1/3
            });

        if ~W.fix_miss
            W.Miss.th0.logit_miss = logit(1e-3);
            W.Miss.th.logit_miss = logit(1e-3);
            W.Miss.th_lb.logit_miss = logit(1e-4);
        end
        
        switch W.td_kind
            case 'Ser'
                W.Td = Fit.D2.Bounded.TdSer;
            case 'Par'
                W.Td = Fit.D2.Bounded.TdPar;
            otherwise
                error('Unknown td_kind=%s\n', W.td_kind);
        end
        W.add_children_props({'Td'});
        
        W.Tnd = Fit.D2.Common.Tnd;
        W.Tnd.disper_kind = 'std';
        W.Tnd.init_params0;
        W.add_children_props({'Tnd'});
        
        %% Find sd & mean of the data
        n_dim = 2;
        n_ch = 2;
        
        [mu_data, sd_data, min_mu_data, min_sd_data] = ...
            W.get_mu_sd_cond_ch;
        
        %% Specify Tnd
        if isempty(W.max_sd_mu_ratio)
            max_sd_mu_ratio = 1;
        else
            max_sd_mu_ratio = W.max_sd_mu_ratio;
        end
        
        ub_mu_margin = W.ub_mu_margin_val;
        ub_sd_margin = W.ub_sd_margin_val;
        
        % Maximum mu and sd tnd cannot be bigger than 
        % minimum mu and sd data
        mu0_tnd = min_mu_data * S.mu0_tnd_proportion;
        mu_lb_tnd = min(0.12, mu0_tnd / 5);
        mu_ub_tnd = max(min_mu_data * ub_mu_margin, mu0_tnd);
%         mu_lb_tnd = max(mu0_tnd / 5, 0.12);
%         mu_ub_tnd = min(mu0_tnd * 2, 5);
        
        sd0_tnd = sqrt(min_sd_data.^2 * S.var0_tnd_proportion);
        sd_lb_tnd = min(0.02, sd0_tnd / 5);
        sd_ub_tnd = max(min_sd_data * ub_mu_margin, sd0_tnd);
        
%         sd_lb_tnd = max(sqrt(sd0_tnd.^2 / 5), 0.02);
%         sd_ub_tnd = max(sqrt(sd0_tnd.^2 / 5), 0.02);
        
%         disper0_tnd = W.Tnd.calc_disper(sd0_tnd, mu0_tnd);
%         disper_lb_tnd = W.Tnd.calc_disper( ...
%             max(sqrt(sd0_tnd.^2 / 3), 0.02), mu0_tnd);
%         disper_ub_tnd = W.Tnd.calc_disper( ...
%             max(sqrt(sd0_tnd.^2 / 3), 0.02), mu0_tnd);

        for ch1 = 1:n_ch
            for ch2 = 1:n_ch
                mu = sprintf('mu_%d_%d', ch1, ch2);
                sd = sprintf('disper_%d_%d', ch1, ch2);
%                 disper = sprintf('disper_%d_%d', ch1, ch2);
                
                W.Tnd.th0.(mu) = mu0_tnd;
                W.Tnd.th.(mu) = mu0_tnd;
                W.Tnd.th_lb.(mu) = mu_lb_tnd;
                W.Tnd.th_ub.(mu) = mu_ub_tnd;
                
                W.Tnd.th0.(sd) = sd0_tnd;
                W.Tnd.th.(sd) = sd0_tnd;
                W.Tnd.th_lb.(sd) = sd_lb_tnd;
                W.Tnd.th_ub.(sd) = sd_ub_tnd;
                
                if ~isnan(max_sd_mu_ratio)
                    W.Tnd.add_constraints({
                        {'A', {sd, mu}, {[1, -max_sd_mu_ratio], -0.001}}
                        });
                end
                
%                 W.th0.(disper) = disper0_tnd;
            end
        end
        
        %% Specify Td
        % Add a param per 1D condition and choice

        mu_tnds = W.Tnd.get_mus;
        sd_tnds = W.Tnd.get_sds;
        
        for dim = 1:n_dim
            n_cond = W.Data.nConds(dim);
            
            for cond = 1:n_cond
                for ch = 1:n_ch
                    mu = W.get_th_name('mu', dim, cond, ch);
                    sd = W.get_th_name('sd', dim, cond, ch);
                    
                    switch dim
                        case 1
                            mu_tnds1 = mean(mu_tnds, 2);
                            sd_tnds1 = sqrt(mean(sd_tnds.^2, 2));
                        case 2
                            mu_tnds1 = mean(mu_tnds, 1);
                            sd_tnds1 = sqrt(mean(sd_tnds.^2, 1));
                    end
                    
                    mu_data1 = mu_data{dim}(cond, ch);
                    sd_data1 = sd_data{dim}(cond, ch);
            
                    mu_tnd = mu_tnds1(ch);
                    sd_tnd = sd_tnds1(ch);
                    
                    switch W.td_kind
                        case 'Ser'
                            mu0 = (mu_data1 - mu_tnd) / 2;
                            sd0 = sqrt((sd_data1.^2 - sd_tnd.^2) / 2);
                        case 'Par'
                            mu0 = mu_data1 - mu_tnd;
                            sd0 = sqrt(sd_data1.^2 - sd_tnd.^2);
                    end
                    
                    min_mu = nanmin(0.05, mu0 / 5);
                    min_sd = nanmin(0.01, sd0 / 5);
                    
                    % UB is when mu or sd in the data
                    % is totally explained by Td.
                    W.add_params({
                        {mu, mu0, min_mu, mu_data1 * ub_mu_margin}
                        {sd, sd0, min_sd, sd_data1 * ub_sd_margin}
                        });
                    if ~isnan(max_sd_mu_ratio)
                        W.add_constraints({
                            {'A', {sd, mu}, {[1, -max_sd_mu_ratio], -0.001}}
                            });
                    end
                    
                    %% Fix params if the condition/ch is excluded
                    % Use cond_ch_to_exclude_jt instead of cond_ch_to_exclude
                    % Therefore, if no single condition-choice combination
                    % has more than 10 trial across the entire
                    % irr_cond and irr_ch, the combination is excluded
                    % because we cannot set the initial value.
                    if ~isempty(W.to_exclude_bins_wo_trials)
                        if dim == 1
                            cond_ch_excluded = ...
                                squeeze(all(all( ...
                                    W.cond_ch_to_exclude_jt, 2), 4));
                        else
                            cond_ch_excluded = ...
                                squeeze(all(all( ...
                                    W.cond_ch_to_exclude_jt, 1), 3));
                        end

                        if cond_ch_excluded(cond, ch)
                            W.th0.(mu) = 3; % W.th_ub.(mu);
                            W.th0.(sd) = 2; % W.th0.(mu) * 0.9;
                            W.fix_to_th0_(mu);
                            W.fix_to_th0_(sd);
                        elseif isnan(mu0) || isnan(sd0)
                            error('NaN in included condition-choice!');
                        end
                    end
                end
            end
        end
    end
    function [mu_data, sd_data, min_mu_data, min_sd_data] = ...
            get_mu_sd_cond_ch(W)
        % mu_data{dim_rel}(cond_rel, ch_rel)
        % : minimum mean across cond_irr and ch_irr,
        %   after excluding cond_ch with too few trials
        %   (cond_ch_to_exclude_jt).
        %
        % sd_data: same for sd.
        
        t = W.t(:);
        RT_data_pdf = W.Data.RT_data_pdf;
        
        % mu_data(cond1, cond2, ch1, ch2)
        mu_data0 = permute(bml.stat.mean_distrib(RT_data_pdf, t), ...
            [2, 3, 4, 5, 1]);
        % sd_data(cond1, cond2, ch1, ch2)
        sd_data0 = permute(bml.stat.std_distrib(RT_data_pdf, t), ...
            [2, 3, 4, 5, 1]);     
        
        to_exclude = W.cond_ch_to_exclude_jt;
        
        mu_data0(to_exclude) = nan;
        sd_data0(to_exclude) = nan;
        
%         cond_ch_to_incl = W.get_cond_ch_to_include_train;
%         mu_data0(~cond_ch_to_incl) = nan;
%         sd_data0(~cond_ch_to_incl) = nan;
        
        mu_data{1} = squeeze(nanmin(nanmin(mu_data0, [], 2), [], 4));
        sd_data{1} = squeeze(nanmin(nanmin(sd_data0, [], 2), [], 4));
        
        mu_data{2} = squeeze(nanmin(nanmin(mu_data0, [], 1), [], 3));
        sd_data{2} = squeeze(nanmin(nanmin(sd_data0, [], 1), [], 3));
        
        min_mu_data = nanmin(cell2vec(mu_data));
        min_sd_data = nanmin(cell2vec(sd_data));        
    end
    function v = get_th_name(~, th, dim, cond, ch)
        v = sprintf('%s_d%d_co%d_ch%d', th, dim, cond, ch);
    end
    function v = get.cond_ch_to_exclude_jt(W)
        n_tr_cond_ch = permute(sum(W.Data.RT_data_pdf), ...
            [2, 3, 4, 5, 1]);
        
        v00 = ~(W.get_cond_ch_to_include_train ...
              | W.get_cond_ch_to_include_valid);
        
        v = v00 | ...
            (n_tr_cond_ch <= W.to_exclude_bins_wo_trials_thres);
    end
    function v = get.ub_mu_margin_val(W)
        if isempty(W.ub_mu_margin)
            v = 1;
        else
            v = W.ub_mu_margin;
        end
    end
    function v = get.ub_sd_margin_val(W)
        if isempty(W.ub_sd_margin)
            v = 1;
        else
            v = W.ub_sd_margin;
        end
    end
    function to_incl = any_cond_ch_on_other_dim(~, to_incl)
        v1 = any(any(to_incl, 2), 4);
        v2 = any(any(to_incl, 1), 3);
        to_incl = bsxfun(@and, v1, v2);
    end
    function pred(W)
        n_ch = 2;
        n_dim = 2;
        t = W.t(:);
        
        for dim = n_dim:-1:1
            n_cond = W.Data.nConds(dim);
            
            for cond = n_cond:-1:1
                for ch = n_ch:-1:1
                    mu = W.th.(W.get_th_name('mu', dim, cond, ch));
                    sd = W.th.(W.get_th_name('sd', dim, cond, ch));
                    
                    td_pred_pdfs{dim}(:, cond, ch) = ...
                        bml.distrib.gampdf_ms(t, mu, sd, 1);
                end
            end
        end
        
        td_pred_pdf = W.Td.get_Td_pdf(td_pred_pdfs);
        W.Data.Td_pred_pdf = td_pred_pdf;
        W.Data.RT_pred_pdf = W.Tnd.Td2RT(td_pred_pdf);
        
        % Match choice frequency with data
        W.Data.RT_pred_pdf = nan0(bsxfun(@rdivide, ...
            W.Data.RT_pred_pdf, sum(W.Data.RT_pred_pdf)));
        W.Data.RT_pred_pdf = bsxfun(@times, ...
            W.Data.RT_pred_pdf, sum(W.Data.RT_data_pdf));

        % Add miss
        if ~isempty(W.Miss)
            W.Miss.pred;
        end
        
        % Enforce sum to 1
        W.Data.RT_pred_pdf = W.Data.RT_pred_pdf ...
            ./ sum(W.Data.RT_pred_pdf(:));
    end
    function pred = get_pred_pdf(W)
        pred = W.get_pred_pdf@Fit.D2.Common.Main;
        if ~isempty(W.to_fold_cond) && W.to_fold_cond
            pred = W.fold_cond(pred);
        end
    end
    function data = get_data_pdf(W)
        data = W.get_data_pdf@Fit.D2.Common.Main;
        if ~isempty(W.to_fold_cond) && W.to_fold_cond
            data = W.fold_cond(data);
        end
    end
    function p = fold_cond(~, p)
        import bml.array.flips
        p = (p ...
            + flips(p, [2, 4]) ...
            + flips(p, [3, 5]) ...
            + flips(p, [2, 3, 4, 5])) ./ 4;
    end
    function mu = get_mu_td(W)
        n_dim = 2;
        n_ch = 2;
        for dim = n_dim:-1:1
            n_cond = W.Data.nConds(dim);
            
            for cond = n_cond:-1:1
                for ch = n_ch:-1:1
                    mu(cond,ch,dim) = ...
                        W.th.(W.get_th_name('mu', dim, cond, ch));
                end
            end
        end
    end
    function sd = get_sd_td(W)
        n_dim = 2;
        n_ch = 2;
        for dim = n_dim:-1:1
            n_cond = W.Data.nConds(dim);
            
            for cond = n_cond:-1:1
                for ch = n_ch:-1:1
                    sd(cond,ch,dim) = ...
                        W.th.(W.get_th_name('mu', dim, cond, ch));
                end
            end
        end
    end
    function add_plotfun(W, Fl, varargin)
        W.add_plotfun@Fit.D2.Common.Main(Fl, varargin{:});
        
        Fl.add_plotfun({
            @(Fl) @(varargin) void( ...
                @() Fl.W.plot_rt('dim_on_x', 1), false);
            @(Fl) @(varargin) void( ...
                @() Fl.W.plot_rt('dim_on_x', 2), false);
            @(Fl) @(varargin) void( ...
                @() Fl.W.plot_rt('dim_on_x', 1, 'yfun', 'var'), false);
            @(Fl) @(varargin) void( ...
                @() Fl.W.plot_rt('dim_on_x', 2, 'yfun', 'var'), false);
            });
    end
    function [Fl, res] = fit(W, varargin)
        S = varargin2S(varargin, {
            'opts', {}
            });
        S.opts = varargin2C(S.opts, {
            'UseParallel', 'always'
            });
        C = S2C(S);
        
        [Fl, res] = W.fit@Fit.D2.Common.Main(C{:});
        
        W.save_mat(Fl);
    end
end
%% Plot
methods
    function plot_and_save_all(W)
        % mean/var plots
        [Ss, n] = bml.args.factorizeS(varargin2S({
            'dim_on_x', {1, 2}
            'yfun', {'mean', 'var'}
            }));
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            clf;
            W.plot_rt_vs_rt(C{:});
            
            file = W.get_file({
                'dimx', S.dim_on_x
                'yfun', S.yfun});
            
            savefigs(file);
        end

        % mean/var with ch_incl
        [Ss, n] = bml.args.factorizeS(varargin2S({
            'dim_on_x', {1, 2}
            'yfun', {'mean', 'var'}
            'accu_only', true
            'ch_incl', {[1,1], [1,2], [2,1], [2,2]}
            }));
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            clf;
            W.plot_rt_vs_rt(C{:});
            
            file = W.get_file({
                'dimx', S.dim_on_x
                'yfun', S.yfun
                'aco', S.accu_only
                'ch', S.ch_incl
                });
            
            savefigs(file);
        end
        
        % plotfuns
        clf;
        W.plot_plotfuns;
        file = W.get_file({
            'plt', 'plotfuns'
            });
        savefigs(file, 'size', [1200 900]);
    end
    function ax = imgather(W0, varargin)
        for yfun = {'mean', 'var'} % {'var'} % 
            for bat = {
                    [], []
                    }'
                [aco, chi] = deal(bat{:});
                
                C = varargin2C(varargin, {
                    'accu_only', aco, ...
                    'ch_incl', chi
                    });
                
                W0.imgather_unit(yfun{1}, C{:});
            end
        end            
    end
    function imgather_unit(W0, yfun, varargin)
        % yfun: 'mean' or 'var'
        
        S = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'accu_only', []
            'ch_incl', []
            });
        
        tds = {'Ser', 'Par'};
        n_dim = 2;
        subjs = S.subj;
        n_subj = numel(subjs);
        
        %% Imgather
        fig_tag(yfun);
        clf;
        ax = gobjects(n_dim, n_subj);

        for i_subj = 1:n_subj
            subj = subjs{i_subj};

            for dim_on_x = 1:n_dim
                ax1 = subplotRC( ...
                    n_dim, n_subj, dim_on_x, i_subj);

                all_files_exist = true;
                for td = tds

                    W = bml.oop.varargin2props(W0, {
                        'subj', subj
                        'td_kind', td{1}
                        });
                    file = [W.get_file({
                        'msf', 0
                        'dimx', dim_on_x
                        'yfun', yfun
                        'aco', S.accu_only
                        'ch', S.ch_incl
                        }), '.fig'];

                    if ~exist(file, 'file')
                        warning('File absent: %s\n', file);
                        all_files_exist = false;
                        keyboard;
                        continue;
                    end
                    
                    [ax1, hs] = bml.plot.openfig_to_axes(file, ax1);

                    data = hs.src.marker;
                    set(data, 'LineWidth', 0.25, 'MarkerSize', 4);
                    pred = findobj(hs.src.nonsegment, ...
                        'LineStyle', '-');
                    set(pred, 'LineWidth', 1);
                end
                ax(dim_on_x, i_subj) = ax1;
                
                if ~all_files_exist
                    continue;
                end
                
                pred = findobj(hs.src.nonsegment, 'LineStyle', '-');
                others = setdiff(hs.src.children, pred);
                delete(others);

                set(pred, 'LineStyle', '--');
            end
        end

        %% Beautify
        set(ax, 'FontSize', 9, 'TickLength', [0.02, 0.025]);

        for i_subj = 1:n_subj
            subj = subjs{i_subj};

            for dim_on_x = 1:n_dim
                ax1 = ax(dim_on_x, i_subj);

                if i_subj > 1
                    ylabel(ax1, '');
                    xlabel(ax1, '');
                end
                if dim_on_x == 1
                    title(ax1, sprintf('Subject %s\n ', subj(2))); % (1)));
                end
                
                S_fig = figure2struct(ax1);
                ix_diag = cellfun(@(c) max(abs(c - 0.75)) < 0.01, ...
                    get(S_fig.line, 'Color'));
                lines = S_fig.line(~ix_diag);
%                 lines = setdiff(lines, S_fig.segment);
                
                h_data = S_fig.marker;
                incl_data = cellfun(@length, get(h_data, 'XData')) == 1;
                h_data = h_data(incl_data);
                
                x_data = cell2vec(get(h_data, 'XData'));
                y_data = cell2vec(get(h_data, 'YData'));
                
                seg = S_fig.segment(1:(end-1));
                seg_x = cell2mat(get(seg, 'XData'));
                seg_y = cell2mat(get(seg, 'YData'));
                seg_incl = ismember(seg_x(:,1), x_data) ...
                         | ismember(seg_y(:,1), y_data);
                lines = setdiff(lines, seg(~seg_incl));
                
                xs = cell2vec(get(lines, 'XData'));
                ys = cell2vec(get(lines, 'YData'));

                min_x = min(xs);
                max_x = max(xs);
                min_y = min(ys);
                max_y = max(ys);
                dif_x = max_x - min_x;
                dif_y = max_y - min_y;
                dif = max([dif_x, dif_y]);
                margin = 1.15;
                margin2 = 1.3;
                
                min_xlim = max(max_x - dif_x * margin); % , -0.1);
                max_xlim = min_xlim + dif * margin;
                min_ylim = max(max_y - dif_y * margin); % , -0.1);
                max_ylim = min_ylim + dif * margin;
                
                xlim(ax1, [min_xlim, max_xlim]);
                ylim(ax1, [min_ylim, max_ylim]);
                
                bml.plot.beautify_tick(ax1, 'X', 'tick', {
                    0:1:5
                    0:0.5:5
                    0:0.2:5
                    0:0.1:5
                    });
                bml.plot.beautify_tick(ax1, 'Y', 'tick', {
                    0:1:5
                    0:0.5:5
                    0:0.2:5
                    0:0.1:5
                    });
            end
        end

        bml.plot.position_subplots(ax, ...
            'margin', [0.13, 0.17, 0.0, 0.15], ...
            'btw_row', 0.2, ...
            'btw_col', 0.06);
        
        file = W0.get_file({
            'sbj', S.subj
            'td', tds
            'dimx', 1:n_dim
            'yfun', yfun
            'aco', S.accu_only
            'ch', S.ch_incl
            });
        bml.plot.savefigs(file, ...
            'PaperPosition', [0, 0, Fig.Consts.width_column1_cm, ...
                              7.5], ...
            'ext', {'.fig', '.png', '.tif'});
    end
end
%% Save
methods
    function save_mat(W, Fl)
        if nargin < 2
            Fl = W.Fl;
        end
        
        file = [W.get_file, '.mat'];
        mkdir2(fileparts(file));
        
        res = Fl.res;
        L = packStruct(W, Fl, res); %#ok<NASGU>
        save(file, '-struct', 'L');
        fprintf('Saved to %s\n', file);
    end
    function fs = get_file_fields0(W)
        fs = union_general( ...
            W.get_file_fields0@Fit.D2.Common.Main, ...
            {
            'td_kind', 'td'
            'to_fold_cond', 'fc'
            'max_sd_mu_ratio', 'smr'
            'ub_mu_margin', 'um'
            'ub_sd_margin', 'us'
            }, 'stable', 'rows');
    end
end
end