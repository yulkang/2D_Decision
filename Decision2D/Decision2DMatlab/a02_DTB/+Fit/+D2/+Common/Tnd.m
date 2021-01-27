classdef Tnd < Fit.Common.Tnd
    % Fit.D2.Common.Tnd
    %
    % Doesn't modify init_bef_fit, pred, or get_cost.
    % Not called from FitFlow directly.
    
    % 2015 YK wrote the initial version.
    
%% Settings
properties
    % distrib : inherited from Fit.Common.Tnd
    
    % n_Tnd
    % : 1, 3, or 4.
    % : Defaults to 4 to address concern that differences in costs are
    %   due to different Tnd distributions.
    n_Tnd = 4; 

    % td_dep_tnd
    % '': none
    % 'm': mu_per_td only
    td_dep_tnd = '';
end
%% Results
properties    
    log_pdf_tnd = [];
    pdf_tnd = [];
    RT_obs = [];
    RT_pred = [];
end
%% Init
methods
    function W = Tnd(varargin)
    %     W.reset_on_save('WTnd', Fit.Common.Tnd);
    %     W.empty_on_save({'pdf_tnd', 'RT_pred', 'RT_obs'});

        bml.oop.varargin2props(W, varargin);
    %     varargin2fields(W, varargin{:});
        W.init_params0;
    end
    function init_params0(W)
        % FitWsTnd for calculation
        W.remove_params_all;
        W.remove_constraints_all;
        switch W.n_Tnd
            case 1
                W.add_params0('mu');
                W.add_params0('disper')
            case 3
                W.add_params0('mu');
                W.add_params({
                    {'mu_UmD', 0, -0.2, 0.2}
                    {'mu_RmL', 0, -0.2, 0.2}
                    });
                W.add_params0('disper');
                W.add_constraints({
                    {'A', {'mu', 'mu_UmD'}, {-[1, -0.5], -0.15}} % mu - 0.5mu_UmD >= 0.2
                    {'A', {'mu', 'mu_UmD'}, {-[1,  0.5], -0.15}} % mu + 0.5mu_UmD >= 0.2
                    {'A', {'mu', 'mu_RmL'}, {-[1, -0.5], -0.15}} % mu - 0.5mu_RmL >= 0.2
                    {'A', {'mu', 'mu_RmL'}, {-[1,  0.5], -0.15}} % mu + 0.5mu_RmL >= 0.2
                    });
            case 4
                for ii = 1:2
                    for jj = 1:2
                        W.add_params0('mu', {ii, jj});
                        W.add_params0('disper', {ii, jj});
                    end
                end
        end

        switch W.td_dep_tnd
            case {'', 'n'}
                % Do nothing
            case 'm'
                % tnd_switch = switch_per_td * td + switch_baseline
                W.add_params({
                    {'switch_per_td', 0.1, 0, 1}
                    {'switch_baseline', 0.1, 0, 1}
                    });
            case 'md'
                % tnd_switch = switch_per_td * td + switch_baseline
                W.add_params({
                    {'switch_per_td', 0.3, 0, 1}
                    {'switch_baseline', 0.02, 0.01, 1}
                    {'switch_disper', -1.5, -3, 0}
                    });
            case 'b'
                % betamean switch time
                W.add_params({
                    {'switch_min', 0.01, 0.01, 0.01}
                    {'switch_max', 0.5, 0, 1}
                    {'switch_t_mean', 0.3, 0, 1}
                    {'switch_t_sd', 0.02, 0, 0.5}
                    {'switch_disper', -1.5, -3, 0}
                    });
        end
    end
end
%% pdf_tnd
methods
    function pdf_tnd = get_pdf_tnd(W, ch1, ch2)
        % pdf_tnd = get_pdf_tnd(W, [ch1, ch2])
        %
        % pdf_tnd: nt x 2 x 2
        n_ch = 2;
        if ~exist('ch1', 'var'), ch1 = 1:n_ch; end
        if ~exist('ch2', 'var'), ch2 = 1:n_ch; end

        mus = W.get_mus;
        sds = W.get_sds;

        mus = mus(ch1, ch2);
        sds = sds(ch1, ch2);

        W.pdf_tnd = W.get_pdf_tnd_with_mu_sd(mus, sds);
        if nargout >= 1
            pdf_tnd = W.pdf_tnd;
        end
    end
    function pdf_tnd = get_pdf_tnd_with_mu_sd(W, mus, sds)
        assert(isequal(size(mus), size(sds)));
        assert(ismatrix(mus));
        siz = num2cell(size(mus));
        pdf_tnd = zeros(W.get_nt, siz{:});
        Tnd_unit = W.get_Tnd_unit;
        for ii = 1:siz{1}
            for jj = 1:siz{2}
                Tnd_unit.mu = mus(ii,jj);
                Tnd_unit.set_sd(sds(ii,jj));
                pdf_tnd(:,ii,jj) = Tnd_unit.get_pdf_tnd;

                if any(isnan(pdf_tnd(:,ii,jj)))
                    keyboard;
                end
            end
        end
    end
end
%% log_pdf_tnd
methods
    function log_pdf_tnd = get_log_pdf_tnd(W)
        % log_pdf_tnd = get_log_pdf_tnd(W)
        %
        % log_pdf_tnd: nt x 2 x 2
        mus = W.get_mus;
        sds = W.get_sds;

        W.log_pdf_tnd = W.get_log_pdf_tnd_with_mu_sd(mus, sds);
        if nargout >= 1
            log_pdf_tnd = W.log_pdf_tnd;
        end
    end
    function log_pdf_tnd = get_log_pdf_tnd_with_mu_sd(W, mus, sds)
        assert(isequal(size(mus), size(sds)));
        assert(ismatrix(mus));
        siz = num2cell(size(mus));
        log_pdf_tnd = zeros(W.get_nt, siz{:});
        Tnd_unit = W.get_Tnd_unit;
        for ii = 1:2
            for jj = 1:2
                Tnd_unit.mu = mus(ii,jj);
                Tnd_unit.set_sd(sds(ii,jj));
                log_pdf_tnd(:,ii,jj) = Tnd_unit.get_log_pdf_tnd;
            end
        end
    end
end
%% Conversion between Td and RT
methods
    function pdf_RT = Td2RT(W, pdf_Td)
        % pdf_RT = Td2RT(W, pdf_Td)

        if ismember(W.td_dep_tnd, {'', 'n'})
            pdf_RT = W.conv_w_Tnd(pdf_Td);
        else
            pdf_RT = W.conv_w_Td_dep_Tnd(pdf_Td);
        end
    end
    function pdf_Td = RT2Td(W, pdf_RT)
        if ismember(W.td_dep_tnd, {'', 'n'})
            pdf_Td = W.conv_w_Tnd(pdf_RT, [], @conv_t_back);
        else
            error('Not implemented yet!');            
        end
    end
    function p_Td_given_RT = get_p_Td_given_RT(W, RT_ix)
        % p_Td_given_RT: nt x nCond_rel x 1 x 2 x 2
        persistent prev_th prev_pdf_tnd_

        curr_th = W.th;
        if isempty(prev_th) || ~isequal(curr_th, prev_th)
            pdf_tnd_ = W.get_pdf_tnd; % nt x 2 x 2
        else
            pdf_tnd_ = prev_pdf_tnd_;
        end
        p_Td_given_RT = [pdf_tnd_((RT_ix:-1:1), :, :);
                   zeros(W.nt - RT_ix, 2, 2)];
        p_Td_given_RT = permute(p_Td_given_RT, [1 4 5 2 3]);

        prev_th = curr_th;
        prev_pdf_tnd_ = pdf_tnd_;
    end
end
%% Internal - when td_dep_tnd = true
methods
    function pdf_RT = conv_w_Td_dep_Tnd(W, pdf_Td)
        % pdf_RT = conv_w_Td_dep_Tnd(W, pdf_Td)
        %
        % pdf_Td(t, cond1, cond2, ch1, ch2)
        % pdf_RT(t, cond1, cond2, ch1, ch2)
        
        %%
        if nargin < 2
            pdf_Td = W.Data.Td_pred_pdf;
        end
        
        % p_rt_given_td(tnd, td)
        p_rt_given_td = W.get_p_rt_given_td;
        
        siz_pdf_RT = size(pdf_Td);
        pdf_Td = reshape(pdf_Td, 1, siz_pdf_RT(1), []);
        pdf_RT = sum(bsxfun(@times, pdf_Td, p_rt_given_td), 2);
        pdf_RT = reshape(pdf_RT, siz_pdf_RT);
        
        if nargout == 0
            W.Data.RT_pred_pdf = pdf_RT;
        end
    end
    function plot_p_rt_given_td(W)
        p = W.get_p_rt_given_td;
        t = W.t;
        imagesc(t, t, p);
        axis xy;
        xlabel('Td (s)');
        ylabel('RT (s)');
        h = crossLine('NE', 0, 'w--');
        uistack(h, 'top');
        
        y = mean_distrib(p, t(:));
        hold on;
        plot(t, y, 'w-');
        hold off;
    end
    function p = get_p_rt_given_td(W)
        % p = get_p_rt_given_td(W)
        %
        % p(rt, td)
        p = W.get_p_tnd_given_td;
        n = size(p, 1);
        assert(size(p, 2) == n);
        
        for ii = 1:n
            p(ii:n, ii) = p(1:(n+1-ii), ii);
            p(1:(ii-1), ii) = 0;
        end
        
        sum0 = sum(p) == 0;
        p = bsxfun(@rdivide, p, sum(p));
        p(:, sum0) = 1 / n;        
    end
    function p = get_p_tnd_given_td(W)
        % p_tnd_given_td(tnd, td)
        
        %%
        t = W.t;
        
        switch W.td_dep_tnd
            case 'm'
                % mus(1, td)
                mus = W.th.switch_baseline + ...
                    W.th.switch_per_td * t;

                % sds(1, td)
                sds = W.get_sd(mus);
                
            case 'md'
                % mus(1, td)
                mus = W.th.switch_baseline + ...
                    W.th.switch_per_td * t;

                % sds(1, td)
                sds = W.get_sd(mus, W.th.switch_disper);
                
            case 'b'
                t_rel = t ./ max(t);
                t_add_rel = bml.distrib.betacdf_ms(t_rel, ...
                    W.th.switch_t_mean, W.th.switch_t_sd);
                mus = W.th.switch_min ...
                    + t_add_rel .* (W.th.switch_max - W.th.switch_min);
                sds = W.get_sd(mus, W.th.switch_disper);
                
            case 'n'
                % Do nothing
                
            otherwise
                error('td_dep_tnd=%s is not supported!\n', W.td_dep_tnd);
        end
        
        %%
        p = W.get_pdf_tnd_with_mu_sd(mus(:), sds(:));
    end
end
%% Internal - when td_dep_tnd = false
methods
    function pdf_res = conv_w_Tnd(W, pdf_src, pdf_tnd, f_conv)
        % pdf_RT = conv_Td_w_tnd(W, pdf_src, pdf_tnd, [f_conv = @conv_t])
        %
        % density outside the range is added to the end (similar to sign rule),
        % to make sure all density is considered.
        if ~exist('pdf_tnd', 'var') || isempty(pdf_tnd)
            pdf_tnd = W.get_pdf_tnd;
        end
        if ~exist('f_conv', 'var')
            f_conv = [];
        end

        nt = size(pdf_src, 1);
        assert(nt == size(pdf_tnd, 1));

        siz_src = size(pdf_src);
        siz_cond = siz_src(2:(end - 2));
        siz_ch = sizes(pdf_tnd, [2 3]);
        assert(isequal(siz_ch, siz_src([end-1, end])));

        n_cond_all = prod(siz_cond);
        n_ch_all = prod(siz_ch);

        pdf_src = reshape(pdf_src, [nt, n_cond_all, n_ch_all]);
        pdf_res = zeros(nt, n_cond_all, n_ch_all);
        pdf_tnd = reshape(pdf_tnd, nt, n_ch_all);

        for ch = 1:n_ch_all
            if isempty(f_conv)
                pdf_res(:, :, ch) = ...
                    bml.math.conv_t(pdf_src(:, :, ch), pdf_tnd(:, ch));
            else
                pdf_res(:, :, ch) = f_conv(pdf_src(:, :, ch), pdf_tnd(:, ch));
            end
        end
        pdf_res = reshape(pdf_res, siz_src);

        sum_src = reshape(sum(pdf_src), [1, siz_src(2:end)]);
        sum_res = sum(pdf_res);
        pdf_res(end, :, :, :, :) = max( ...
            pdf_res(end, :, :, :, :) + (sum_src - sum_res), ...
            0);

        try
            assert_isequal_within(sum(pdf_res), sum_src, ...
                1e-2, ... % 1e-10, ...
                'relative_tol', false);
        catch err
            warning(err_msg(err));
        end
            
    %     
    %     W.RT_pred = pdf_RT; % Shouldn't have side effect
    end
    function mus = get_mus(W)
        switch W.n_Tnd
            case 1
                mus = repmat(W.get_('mu'), [2 2]);
            case 3
                mu0 = W.get_('mu');
                mu_UmD = W.get_('mu_UmD');
                mu_RmL = W.get_('mu_RmL');

                % 1st dim is motion, L (1) vs R (2)
                % 2nd dim is color,  D (1) vs U (2) % This is reversed
                mus(1,1) = mu0 - mu_UmD / 2 - mu_RmL / 2;
                mus(1,2) = mu0 + mu_UmD / 2 - mu_RmL / 2;
                mus(2,1) = mu0 - mu_UmD / 2 + mu_RmL / 2;
                mus(2,2) = mu0 + mu_UmD / 2 + mu_RmL / 2;
            case 4
                mus = zeros(2,2);
                for ii = 1:2
                    for jj = 1:2
                        mus(ii,jj) = W.get_(str_con('mu', ii, jj));
                    end
                end
            otherwise
                error('Illegal n_Tnd=%d\n', W.n_Tnd);
        end
    end
    function mus = get_mus_as_on_expr_display(W)
        mus = W.get_mus;
        mus = W.permute_as_on_expr_display(mus);
    end
    function sds = get_sds(W)
        if ismember('disper', W.get_names) 
            disper = W.get_('disper');
            sds = W.get_sd(W.get_mus, disper);
        elseif W.n_Tnd == 4
            mu = zeros(2,2);
            disper = zeros(2,2);
            for ii = 1:2
                for jj = 1:2
                    mu(ii,jj) = W.get_(str_con('mu', ii, jj));
                    disper(ii,jj) = W.get_(str_con('disper', ii, jj));
                end
            end
            sds = W.get_sd(mu, disper);
        else
            sds = zeros(2,2);
        end
    end
    function sds = get_sds_as_on_expr_display(W)
        sds = W.get_sds;
        sds = W.permute_as_on_expr_display(sds);
    end
    function mus_or_sds = permute_as_on_expr_display(~, mus_or_sds)
        mus_or_sds = [
            mus_or_sds(1,2), mus_or_sds(2,2)
            mus_or_sds(1,1), mus_or_sds(2,1)
            ];
    end
    function Tnd_unit = get_Tnd_unit(W)
        Tnd_unit = Fit.Common.Tnd('t', W.t, 'distrib', W.distrib);
    end
end
%% Plot
methods
    function [h, x, y] = plot(W, varargin)
        x = W.t;
        y = W.get_pdf_tnd;
        for ii = 1:2
            for jj = 1:2
                % 1st dim is motion, R vs L
                % 2nd dim is color,  U vs D
                subplotRC(2,2,jj,ii);
                plot(x, y(:,ii,jj), varargin{:});
            end
        end
    end
    function [h, res] = plot_overlaid(W, varargin)
        S = varargin2S(varargin, {
            'ind', {1:2, 1:2}
            'plot_args', {}
            'draw_legend', true
            });

        x = W.t;
        y = W.get_pdf_tnd;
        if ~iscell(S.ind)
            S.ind = {S.ind};
        end
        nt = size(y, 1);
        y = reshape(y(:, S.ind{:}), nt, []);
        h = plot(x(:), y, S.plot_args{:});

        if isequal(S.ind, {1:2, 1:2}) && S.draw_legend
            legend(h, {'LD', 'RD', 'LU', 'RU'});
        end

        res = packStruct(h, x, y, S);
    end
end
methods (Static)
    function WTnd = demo
        WTnd = Fit.D2.Common.Tnd;
        WTnd.plot;
    end
end
end