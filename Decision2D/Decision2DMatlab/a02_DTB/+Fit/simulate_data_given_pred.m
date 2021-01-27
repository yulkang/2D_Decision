function [data, ch_new, rt_vec_new] = ...
    simulate_data_given_pred(pred, dCond, t, varargin)
% data = simulate_data_given_pred(pred, varargin)
%
% INPUT:
% pred(t, cond1, cond2, ch1, ch2)
% dCond(tr, dim) = dCond % 1 to nCond(dim)
%
% OUTPUT:
% data(t, cond1, cond2, ch1, ch2)

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

S = varargin2S(varargin, {
    'seed', 1
    'ch', []
    'keep_ch', true
    });

%%
rng(S.seed);

n_cond1 = size(pred, 2);
n_cond2 = size(pred, 3);
n_ch = 2;
n_tr = size(dCond, 1);
nt = numel(t);
t_ch = (1:(nt * n_ch^2))';

data = zeros(size(pred));
ch = S.ch;

ch_new = zeros(n_tr, n_ch);
rt_vec_new = zeros(n_tr, 1);

if S.keep_ch
    % To keep condition & choice frequencies in the bootstrap
    ch_new = ch;
    for dCond1 = 1:n_cond1
        for dCond2 = 1:n_cond2            
            for ch1 = 1:n_ch
                for ch2 = 1:n_ch
                    tr_incl = (dCond(:,1) == dCond1) ...
                            & (dCond(:,2) == dCond2) ...
                            & (ch(:,1) == ch1) ...
                            & (ch(:,2) == ch2);
                    n_tr_incl = nnz(tr_incl);

                    if n_tr_incl > 0
                        rt_pdf1 = pred(:, dCond1, dCond2, ch1, ch2);
                        rt_vec_new(tr_incl) = randsample( ...
                            t, n_tr_incl, true, rt_pdf1);
                    end
                end
            end
        end
    end
else
    % To keep only condition frequencies in the bootstrap
    for dCond1 = 1:n_cond1
        for dCond2 = 1:n_cond2            
            tr_incl = (dCond(:,1) == dCond1) ...
                    & (dCond(:,2) == dCond2);
            n_tr_incl = nnz(tr_incl);

            if n_tr_incl > 0
                rt_pdf1 = vVec(pred(:, dCond1, dCond2, :, :));
                rt_ch_vec = ...
                    randsample(t_ch, n_tr_incl, true, rt_pdf1);

                [rt_ix, ch1, ch2] = ind2sub([nt, n_ch, n_ch], rt_ch_vec);
                rt_vec_new(tr_incl) = t(rt_ix);
                ch_new(tr_incl, :) = [ch1, ch2];

                RT_data_pdf1 = accumarray([rt_ix, ch1, ch2], 1, ...
                    [nt, n_ch, n_ch], @sum);

                data(:, dCond1, dCond2, :, :) = ...
                    reshape(RT_data_pdf1, [nt, 1, 1, n_ch, n_ch]);
            end            
        end
    end
end