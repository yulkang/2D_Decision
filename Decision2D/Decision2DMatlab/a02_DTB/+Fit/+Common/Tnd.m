classdef Tnd < Fit.Common.CommonWorkspace
    % Nondecision time.
    
    % 2015 YK wrote the initial version.
properties
    distrib = 'invgauss' % 'halfnorm'; % 'delta' | 'norm' | 'halfnorm' | 'gamma'
    disper_kind = 'log10_fano'; % 'std', 'cv', 'log10_cv', 'log10_fano'
    
    mu = 0.2;
    disper = 0.1; % (sig^2)/mu < 1
end
methods
function W = Tnd(varargin)
    W.set_kind_general('Tnd');
    W.init_params0(varargin{:});
end
function init_params0(W, varargin)
    varargin2fields(W, varargin);
    W.add_params0('mu');
    W.add_params0('disper');
end
function pred(W)
    W.Data.set_RT_pred_pdf(W.Td2RT(W.Data.get_Td_pred_pdf));
end
function RT_pred_pdf = Td2RT(W, Td_pred_pdf)
    error('Modify in subclasses!');
end
%% Parameters
function add_params0(W, kind, postfix)
    % add_params0(W, kind = 'mu'|'disper', postfix='')
    % postfix: char, numeric, or cell array of the two.
    if nargin < 3, postfix = ''; end
    if ~iscell(postfix), postfix = {postfix}; end
    assert(ismember(kind, {'mu', 'disper'}));
    name = str_con(kind, postfix{:});
    
    switch kind
        case 'mu'
            W.add_params({{name, 0.3, 0.1, 5}}); % 0.75}});
            
        case 'disper'
            switch W.distrib
                case 'delta'
                    W.remove_params({name});
                otherwise
                    mu0 = W.th0.(str_con('mu', postfix{:}));
                    sd0 = 0.1; % 0.02;
                    sd_lb = 0.01; % 0.02;
                    
                    if strcmp(W.distrib, 'gamma')
                        log10_fano_ub = log10(0.995);
                    else
                        log10_fano_ub = log10(1.5);
                    end
                    
                    disper0 = W.calc_disper(sd0, mu0);
                    disper_lb = W.calc_disper(sd_lb, mu0);
                    
                    sd_ub = W.get_sd(mu0, log10_fano_ub, ...
                        'disper_kind', 'log10_fano');
                    disper_ub = W.calc_disper(sd_ub, mu0);
                    
                    W.add_params({{name, disper0, disper_lb, disper_ub}});
%                     W.add_params({{name, -2, -3, -0.2}});
            end
    end
end
function sd = get_sd(W, mu, disper, varargin)
    % sd = get_sd(W, mu=W.mu, disper=W.disper, varargin)
    
    if nargin < 2 || isempty(mu), mu = W.mu; end
    if nargin < 3 || isempty(disper), disper = W.disper; end
    S = varargin2S(varargin, {
        'disper_kind', W.disper_kind
        });
    switch S.disper_kind
    case 'log10_fano'
        sd = sqrt(mu .* 10.^disper);
    case 'log10_cv'
        sd = mu .* 10.^disper;
    case 'std'
        sd = disper;
    case 'cv'
        sd = mu .* disper;
    end
end
function set_sd(W, sd, mu)
    if ~exist('mu', 'var'), mu = W.mu; end
    W.disper = W.calc_disper(sd, mu);
end
function disper = calc_disper(W, sd, mu)
    if ~exist('mu', 'var'), mu = W.mu; end
    switch W.disper_kind
    case 'log10_fano'
        disper = log10(sd.^2 ./ mu);
    case 'log10_cv'
        disper = log10(sd ./ mu);
    case 'std'
        disper = sd;
    case 'cv'
        disper = sd ./ mu;
    end
    W.disper = disper;
end
function cv = get_cv(W, disper, mu)
    % cv = get_cv(W, disper, [mu])
    switch W.disper_kind
        case 'log10_fano'
            cv = sqrt(10.^disper);
        case 'log10_cv'
            cv = 10.^disper;
        case 'cv'
            cv = disper;
        case 'std'
            if exist('mu', 'var') && ~isempty(mu)
                cv = disper ./ mu;
            else
                error('Cannot convert std to cv!');
            end
    end
end
%% Preferences
function set_distrib(W, distrib)
    assert(ismember(distrib, {'delta', 'norm', 'halfnorm', 'gamma'}));
    W.distrib = distrib;
end
function set_disper_kind(W, disper_kind)
    assert(ismember(disper_kind, {'std', 'cv', 'log10_cv'}));
    W.disper_kind = disper_kind;
end
%% pdf_tnd
function pdf_tnd = get_pdf_tnd(W)
    % pdf_tnd: nt x 1
    nt = length(W.t);
    pdf_tnd = zeros(nt, 1);
    
    sd = W.get_sd;
    
    try
        switch W.distrib
            case 'delta'
                pdf_tnd = delta_fun(W.mu, W.t(:));
            case 'gamma'
                pdf_tnd = gampdf_ms(W.t(:), W.mu, sd, 1);
            case 'norm'
                pdf_tnd = normpdf(W.t(:), W.mu, sd, 1);
            case 'halfnorm'
                pdf_tnd = bml.distrib.halfnormpdf(W.t(:), W.mu, sd, 1);
            case 'invgauss'
                pdf_tnd = bml.distrib.invgausspdf_ms(W.t(:), W.mu, sd, 1);
            otherwise
                error('Unknown distrib!');
        end
    catch err
        warning(err_msg(err));
        keyboard;
    end
    % Normalize so that each column sums to 1.
    pdf_tnd = pdf_tnd / sum(pdf_tnd);
end
%% log_pdf_tnd
function pdf_tnd = get_log_pdf_tnd(W)
    % log_pdf_tnd: nt x 1
    nt = length(W.t);
    pdf_tnd = zeros(nt, 1);
    
    sd = W.get_sd;
    
    try
        switch W.distrib
            case 'delta'
                pdf_tnd = log(max(delta_fun(W.mu, W.t(:)), eps));
            case 'gamma'
                pdf_tnd = bml.distrib.loggampdf_ms(W.t(:), W.mu, sd, 1) ...
                    + log(W.dt);
            case 'norm'
                pdf_tnd = bml.distrib.lognormpdf(W.t(:), W.mu, sd, 1) ...
                    + log(W.dt);
            case 'halfnorm'
                pdf_tnd = bml.distrib.loghalfnormpdf(W.t(:), W.mu, sd, 1) ...
                    + log(W.dt);
            otherwise
                error('Unknown distrib!');
        end
    catch err
        warning(err_msg(err));
        keyboard;
    end
    % Normalize so that each column sums to 1.
    pdf_tnd = bml.math.normalize_log_p(pdf_tnd);
end
%% Plot
function [h, x, y] = plot(W, varargin)
    x = W.t;
    y = W.get_pdf_tnd;
    h = plot(x, y, varargin{:});
    
    xlim(W.t([1,end]));
    bml.plot.beautify;
end
%% Simulation
function [p, t_sim] = sim_data(W, varargin)
    S = varargin2S(varargin, {
        'n_sim', 1000
        });
    
    p = W.get_pdf_tnd;
    t = W.t;
    nt = length(t);
    dt = (t(end) - t(1)) / (nt - 1);
    t_sim = randsample(numel(p), S.n_sim, true, p(:));
end
end
methods (Static)
function WTnd = demo
    WTnd = Fit.Common.Tnd;
    WTnd.Params2W;
    WTnd.plot;
end
function Fl = demo_fit
    %%
    Fl = FitFlow;
end
end
end