function [nll, nll_sep] = nll_bin(pred_pmf, obs_pmf, varargin)
% negative log likelihood from binned data and prediction.
%
% nll = nll_bin(pred_pmf, obs_pmf, ...)
%
% pred_pmf(bin,cond) : predicted probability that a trial belongs to k-th bin.
% obs_pmf(bin,cond)  : observed number of trials that belong to k-th bin.
% nll     : negative log likelihood of the prediction.
%
% OPTIONS:
% 'normalize': If true (default), adds a very small number (eps) preserving the sum.

% Yul Kang 2016. hk2699 at columbia dot edu.

S = varargin2S(varargin, {
    'normalize', true
    });

% pred_pmf = pred_pmf(:);
% obs_pmf  = obs_pmf(:);

if S.normalize
    % Make sure pred is positive and sums to 1 within each condition
    pred_pmf = max(pred_pmf, 0) + eps;
    pred_pmf = bsxfun(@rdivide, pred_pmf, sum(pred_pmf, 1));
    
% %     Make sure pred is positive, while preserving its sum.
%     pred_pmf = max(pred_pmf, 0);
%     sum_pred = sum(pred_pmf) + eps;
%     pred_pmf = bsxfun(@times, pred_pmf + eps, ...
%         sum_pred ./ (sum_pred + eps .* size(pred_pmf,1)));
%
%         (pred_pmf + eps) ...
%             .* (sum_pred ./ (sum_pred + eps .* numel(pred_pmf)));
end

n = sum(obs_pmf);

nll_sep = gammaln(obs_pmf + 1) - obs_pmf .* log(pred_pmf);
nll = sum(vVec(-gammaln(n + 1) + sum(nll_sep)));
