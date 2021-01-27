classdef Miss < Fit.Common.CommonWorkspace
    % Fit.Common.Miss
    %
    % Gets Td_pdf (1D) and RT_pdf (1D), and adds miss to Td_pdf
    
    % 2015 YK wrote the initial version.
properties
    min_RT = -inf;
    max_RT = inf;
end
properties (Dependent)
    fix_miss
end
%% Core
methods
    function W = Miss
        W.set_kind_general('Miss');
        W.add_params0;
    end
    function add_params0(W)
        W.add_params({
            {'logit_miss', logit(1e-3), logit(1e-4), logit(0.15)}
%             {'logit_miss', logit(0.01), logit(1e-4), logit(0.15)}
%             {'miss', 0.01, 1e-4, 0.15}
            });
    end
    function pred(W)
        W.set_pred_pdf(W.add_miss(W.get_pred_pdf));
    end
    function lapse_pdf = get_lapse_pdf(W, pred_pdf)
        % lapse_pdf = get_lapse_pdf(W, Td_pdf)
        %
        % Sums to 1.
        % uniform distribution between min_RT and max_RT.
        
        if nargin < 2
            pred_pdf = W.get_pred_pdf; 
        end
        
        if (W.min_RT == -inf) && (W.max_RT == inf)
            % Fill in the whole time
            lapse_pdf = ones(size(pred_pdf, 1), 1);
        else
            assert(W.nt == size(pred_pdf, 1));
            t = W.t(:);
            lapse_pdf = double((t >= W.min_RT) & (t <= W.max_RT));
        end
        
        lapse_pdf = lapse_pdf / sum(lapse_pdf);
    end
    function [pred_pdf, lapse_pdf] = add_miss(W, pred_pdf, miss)
        % [Td_pdf, lapse_pdf] = add_miss(W, Td_pdf)
        %
        % Td_pdf : [time x condition x choice]
        
        if nargin < 2
            pred_pdf = W.get_pred_pdf;
        end
        if nargin < 3
            if isfield(W.th, 'miss')
                miss = W.th.miss;
            else
                miss = invLogit(W.th.logit_miss);
            end
        end

        lapse_pdf = W.get_lapse_pdf(pred_pdf);
            
        % Sum within each condition across time and choice
        n_cond_ch = prod(sizes(pred_pdf, ...
            [W.Data.dim_pdf.cond, W.Data.dim_pdf.ch]));
        
        miss_total = sums(pred_pdf * miss);
        pred_pdf = pred_pdf * (1 - miss);
        pred_pdf = bsxfun(@plus, ...
            pred_pdf, lapse_pdf * miss_total / n_cond_ch);
    end
    function set.fix_miss(W, v)
        if v
            W.th.logit_miss = W.th_lb.logit_miss;
            W.fix_to_th_('logit_miss');
        else
            W.remove_params({'logit_miss'});
            W.add_params0;
        end        
    end
    function v = get.fix_miss(W)
        v = W.th_fix.logit_miss;
    end
end
%% Data interface
methods
    function pred_pdf = get_pred_pdf(W)
        pred_pdf = W.Data.get_RT_pred_pdf;
    end
    function set_pred_pdf(W, pred_pdf)
        W.Data.set_RT_pred_pdf(pred_pdf);
    end
    function ch_dim = get_ch_dim(W)
        ch_dim = W.Data.dim_pdf.ch;
    end
end
%% Test
methods
    function dif = test_add_miss(W)
        pred_pdf = W.get_pred_pdf;
        sum_bef = sum(pred_pdf(:));
        pred_pdf = W.add_miss(pred_pdf);
        sum_aft = sum(pred_pdf(:));
        
        dif = sum_aft - sum_bef;
        fprintf('Miss: sum(pred_pdf) bef - aft = %1.3g\n', dif);
        asssert(abs(dif) < 1e-5);
    end
end
end