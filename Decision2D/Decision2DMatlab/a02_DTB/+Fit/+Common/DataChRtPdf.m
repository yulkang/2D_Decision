classdef DataChRtPdf ...
        < FitData ...
        & TimeAxis.TimeInheritable
    % Fit.Common.DataChRtPdf
    %
    % Uses conds_oversample_factor to smooth conds

    % 2021 Yul Kang. hk2699 at caa dot columbia dot edu.
    
%% === DataChRtPdf0 ===
properties
    subj = Data.Consts.subjs_RT{1};
    parad = 'RT';
    fun_bias_cond = {};

    to_excl_outlier_runs = 0;
    
    ix_run_to_excl = []; % a vector of i_all_Run to exclude.
    
    to_determine_accu_from_bias_ch = true;
    
    tr_min = Data.Consts.n_tr_initial_skip + 1;
end
properties (Dependent)
    n_tr
    n_tr0
    
%     % to_excl_outlier_runs: preset controlling all other settings.
%     to_excl_outlier_runs

    accu_all_dim
    accu0_all_dim
    
    ch_all_dim
    ch0_all_dim
    
    conds0_wo_oversample_all_dim
    conds_wo_oversample_all_dim
    
    answer_all_dim
    answer0_all_dim    
    
    accu0_all_dim_bias_ch % (tr0, dim) % calculated from ds0 based on bias_ch
    accu_all_dim_bias_ch % (tr, dim)
    
    RT_data_pdf
    Td_pred_pdf
    RT_pred_pdf
    
    cond % (tr, dim)
    dCond % (tr, dim)
    adCond % (tr, dim) % Defined with logistic regression (use multinomial?)
    accu % (tr, dim)
    answer % (tr, dim)
    ch % (tr, dim)
    rt % (tr, 1)
    
    nConds
    conds
    aConds
end
properties (Transient) 
    RT_data_pdf_ % nt x nConds x ch or nt x nConds1 x nConds2 x ch1 x ch2
    Td_pred_pdf_ 
    RT_pred_pdf_ 
end
properties
    RT_data_pdf_perm % non-transient to be kept through fitting
end
properties % (SetAccess = protected)
    hash_RT_data_pdf
end
properties (Dependent)
    dim_pdf
    dim_pdf_rel % Named dimensions for results from get_RT_pdf_rel, etc.
end
% properties (Dependent)
%     cond % nTr x nDim matrix
%     ch % ch(tr,dim) is one of 1, 2, nan.
%     answer % answer(tr,dim) is the correct answer
%     RT % nTr x 1 vector.    
%     
%     aCond % abs(cond)
%     dCond % [~,~,dCond(:,dim)] = uniquetol_wrap(cond(:,dim))
%     adCond % [~,~,adCond(:,dim)] = uniquetol_wrap(abs(cond(:,dim)))
%     accu % accu(tr,X) is one of 1, 2, nan.
%     nConds % 1 x nDim vector.
%     naConds % 1 x nDim vector.    
%     conds % {conds1, conds2}, where conds{Dim} = uniquetol_wrap(cond(:,dim))
%     aConds % {aConds1, aConds2}, where aConds{Dim} = uniquetol_wrap(aCond(:,dim))
% end
properties
    n_dim_task = 2;
    dim_rel_ = 1;
end
properties (Dependent) % Determined by n_dim_task and dim_rel
    task
    dim_rel % always scalar
    dim_irr
    dim_used % 1:2 if n_dim_task = 2; dim_rel if n_dim_task = 1
end
methods
    function Dat = DataChRtPdf(varargin)
        Dat.loaded = false;
        if nargin > 0
            Dat.init(varargin{:});
        end
    end
    function init(Dat, varargin)
        bml.oop.varargin2props(Dat, varargin, true);
    end
%% ds0
function set_ds0(Dat, ds0)
    Dat.set_ds0@FitData(ds0);
    Dat.loaded = true; % To prevent infinite recursion
    Dat.prepare_ds0;
    Dat.reset_RT_data_pdf; % Initialize so that it is recalculated
end
function prepare_ds0(Dat)
%     filt_prev = Dat.get_dat_filt;
%     Dat.set_filt_spec(true(Dat.get_n_tr0, 1));

    Dat.ds0.cond = [Dat.ds0.condM, Dat.ds0.condC];
    Dat.ds0.ch = [Dat.ds0.subjM, Dat.ds0.subjC] == 2;
    Dat.ds0.answer = [Dat.ds0.corrM, Dat.ds0.corrC] == 2;
    Dat.ds0.accu = Dat.ds0.ch == Dat.ds0.answer;
    
    for dim = Data.Consts.n_dim:-1:1
        [~,~,Dat.ds0.adCond(:, dim)] = unique(abs(Dat.ds0.cond(:, dim)));
    end
    
%     for f = {
%             'cond', 'ch', 'accu', 'dCond', 'adCond', 'answer'
%             }
%         Dat.ds0.(f{1}) = Dat.(['get_' f{1}])();
%     end
%     Dat.ds0.ch = Dat.ds0.ch == 2;
%     Dat.ds0.answer = Dat.ds0.answer == 2;

    Dat.ds0.n_dim_task = (Dat.ds0.task == 'A') + 1; % QUIRK
    
%     Dat.set_filt_spec(filt_prev);
end
%%
function v = get_path(Dat)
    if isempty(Dat.path)
        Dat.path = Data.DataLocator.sTr( ...
            'subj', Dat.subj, ...
            'parad', Dat.parad, ...
            'task', Dat.task);
        Dat.path = Dat.path{1};
    end
    v = Dat.path;
end
function set_filt_spec(Dat, varargin)
    Dat.set_filt_spec@FitData(varargin{:});
    Dat.reset_RT_data_pdf; % Initialize so that it is recalculated
end
function set_path(Dat, path0, task, dim_rel)
    % set_path(Dat, path, task, dim_rel)
    if nargin < 2
        path0 = {};
    end
    if nargin >= 3 && ~isempty(task)
        Dat.task = task;
    end
%     if nargin < 3
%         task = Dat.get_default_task;
%         warning(['Task is not given and automatically set to ''%s''.\n' ...
%                  'Use this syntax only for testing purposes.'], task);
%     end

    if iscell(path0) || isstruct(path0)
        path0 = varargin2C(path0, {
            'subj',  Dat.subj
            'parad', Dat.parad
            'task',  Dat.task
            });

        [path, S] = Data.DataLocator.sTr(path0{:});
        if iscell(path) % When multiple paths were expected
            path = path{1};
        end
        Dat.subj = S.subj;
        Dat.parad = S.parad;
        Dat.task = S.task;
    else
        assert(ischar(path0));
        path = path0;
    end
    assert(ischar(path));
    Dat.set_path@FitData(path);
    
    if nargin >= 4
        Dat.dim_rel = dim_rel;
    end
end
function task = get_default_task(~)
    task = Data.Consts.tasks{1,1};
end
function v = get.task(Dat)
    v = Data.Consts.tasks{Dat.n_dim_task, Dat.dim_rel};
end
function set.task(Dat, v)
    [Dat.n_dim_task, dim_rel_new] = find( ...
        strcmp(Data.Consts.tasks, v), 1, 'first');
    if Dat.n_dim_task == 1
        Dat.dim_rel = dim_rel_new;
    end
end
function load_data(Dat)
    Dat.load_data@FitData();
    Dat.ix_run_to_excl = [];
end
function dat_filt = get_dat_filt(Dat)
    % Filter by relevant dim & task
    
    dat_filt = Dat.get_dat_filt@FitData;
    
    if isempty(dat_filt) % When there is no trial, skip the rest
        return;
        
    else
        dat_filt_nan = ...
            all(isnan(Dat.get_cond0), 2) | ...
            all(isnan(Dat.get_ch0), 2) | ...
            all(isnan(Dat.get_answer0), 2) | ...
            all(isnan(Dat.get_RT0), 2);

        assert(~isempty(Dat.task), 'Set Dat.task before use!');
        dat_filt_task = bsxEq(Dat.ds0.task, Dat.task);

        assert(isnumeric(dat_filt));
        
        dat_filt = dat_filt( ...
            bsxEq(dat_filt, find(~dat_filt_nan)));
        dat_filt = dat_filt( ...
            bsxEq(dat_filt, find(dat_filt_task)));

        % tr_min
        tf_task = bsxfun(@eq, Dat.ds0.task(:), Dat.task(:)');
        ix_in_task = ...
            sum(cumsum(tf_task) .* tf_task, 2);
        ix_in_task = ix_in_task(dat_filt);

        dat_filt = dat_filt(ix_in_task >= Dat.tr_min);
        
        % exclude outlier runs
        if Dat.to_excl_outlier_runs
            ix_run = Dat.ds0_.i_all_Run(dat_filt);
            incl = ~ismember(ix_run, Dat.ix_run_to_excl);
            dat_filt = dat_filt(incl);
        end
    end
end
%% Transfer between FitData objects
function import_FitData(Dat_dst, Dat_src)
    % import_FitData(Dat_dst, Dat_src)
    Dat_dst.subj = Dat_src.subj;
    Dat_dst.task = Dat_src.task;
    Dat_dst.parad = Dat_src.parad;
    Dat_dst.import_FitData@FitData(Dat_src);
end
%% RT and TD data and pred pdf
function v = get.RT_pred_pdf(Dat)
    v = Dat.get_RT_pred_pdf;
end
function set.RT_pred_pdf(Dat, v)
    Dat.set_RT_pred_pdf(v);
end
function v = get.Td_pred_pdf(Dat)
    v = Dat.get_Td_pred_pdf;
end
function set.Td_pred_pdf(Dat, v)
    Dat.set_Td_pred_pdf(v);
end
function v = get.RT_data_pdf(Dat)
    v = Dat.get_RT_data_pdf;
end
function set.RT_data_pdf(Dat, v)
    Dat.set_RT_data_pdf(v);
end

function v = get_RT_pred_pdf(Dat)
    v = Dat.RT_pred_pdf_;
end
function v = get_Td_pred_pdf(Dat)
    v = Dat.Td_pred_pdf_;
end
function set_RT_pred_pdf(Dat, v)
    assert(isempty(v) || isequal(size(v), Dat.get_size_RT_Td_pdf));
    Dat.RT_pred_pdf_ = v;
end
function set_Td_pred_pdf(Dat, v)
    assert(isempty(v) || isequal(size(v), Dat.get_size_RT_Td_pdf));
    Dat.Td_pred_pdf_ = v;
end
function siz = get_size_RT_Td_pdf(Dat)
    error('Modify in subclasses!');
end
function v = get_RT_data_pdf(Dat)
    if ~isempty(Dat.RT_data_pdf_perm) 
        Dat.RT_data_pdf_ = Dat.RT_data_pdf_perm;
        
    elseif isempty(Dat.RT_data_pdf_)
        Dat.RT_data_pdf_ = Dat.calc_RT_data_pdf;
        
%         if any(isnan(ch(:)))
%             warning('DataChRtPDf0:ChNaNto1', 'Converted NaN choices to 1!');
%             ch(isnan(ch)) = 1;
%         end
%         Dat.RT_data_pdf_ = accumarray( ...
%             [RT_ix, Dat.get_dCond, ch], ...
%             1, ...
%             [Dat.nt, Dat.get_nConds, 2 + zeros(1, size(ch,2))], ...
%             @sum);
    end
    if nargout > 0
        v = Dat.RT_data_pdf_;
    end
end
function v = calc_RT_data_pdf(Dat, varargin)
    S = varargin2S(varargin, {
        'RT_ix', Dat.get_RT_ix
        'ch', Dat.get_ch
        'dCond', Dat.get_dCond
        'nConds', Dat.get_nConds
        'nt', Dat.nt
        });
    
    if any(isnan(S.ch(:)))
        warning('DataChRtPDf0:ChNaNto1', 'Converted NaN choices to 1!');
        S.ch(isnan(S.ch)) = 1;
    end
    v = accumarray( ...
        [S.RT_ix, S.dCond, S.ch], ...
        1, ...
        [S.nt, S.nConds, 2 + zeros(1, size(S.ch,2))], ...
        @sum);
end
function set_RT_data_pdf(Dat, v)
    Dat.RT_data_pdf_ = v;
end
function reset_RT_data_pdf(Dat)
    Dat.RT_data_pdf_ = [];
    Dat.hash_RT_data_pdf = [];
end
function refresh_RT_data_pdf(Dat)
    Dat.reset_RT_data_pdf;
    Dat.get_RT_data_pdf;
end
%% Conversion of RT between sec and fr
function v = get_RT_ix(Dat)
    v = Dat.convert_RT_sec2fr_ix(Dat.get_RT);
end
function fr = convert_RT_sec2fr_ix(Dat, RT_sec)
    fr = Dat.Time.convert_sec2fr_ix(RT_sec);
end
%% n_tr
function v = get.n_tr(Dat)
    if ~Dat.is_loaded
        try
            Dat.load_data;
        catch err
            warning(err_msg(err));
            warning('Load data before use!');
            v = [];
            return;
        end
    end
    v = size(Dat.ds, 1);
end
function v = get.n_tr0(Dat)
    if ~Dat.is_loaded
        try
            Dat.load_data;
        catch err
            warning(err_msg(err));
            warning('Load data before use!');
            v = [];
            return;
        end
    end
    v = size(Dat.ds0, 1);
end
%% Conditions
function v = get_aCond(Dat)
    v = abs(Dat.get_cond);
end
function v = get_dCond(Dat)
    cond = Dat.get_cond;
    conds = Dat.get_conds;
    
    nDim = size(cond,2);
    for ii = nDim:-1:1
        v(:, ii) = bsxFind(cond(:,ii), conds{ii}); % For oversampled conds
%         [~,~,v(:,ii)] = uniquetol_wrap(cond(:,ii));
    end
end
function v = get.adCond(Dat)
    v = Dat.get_adCond;
end
function v = get_adCond(Dat)
    % Same as get_dCond except for getting cond and conds
    cond = Dat.get_aCond; 
    conds = Dat.get_aConds;
    
    nDim = size(cond,2);
    for ii = nDim:-1:1
        v(:, ii) = bsxFind(cond(:,ii), conds{ii}); % For oversampled conds
%         [~,~,v(:,ii)] = uniquetol_wrap(cond(:,ii));
    end
end
function v = get.conds(Dat)
    v = Dat.get_conds;
end
function conds_rel = get_conds_rel(Dat)
    dim_rel = Dat.get_dim_rel;
    assert(isscalar(dim_rel));
    conds = Dat.get_conds;
    conds_rel = conds{dim_rel};
end
function v = get.aConds(Dat)
    v = Dat.get_aConds;
end
function v = get_aConds(Dat)
    v = Dat.get_conds;
    nDim = length(v);
    for ii = nDim:-1:1
        v{ii} = uniquetol_wrap(abs(v{ii}));
    end
end
function v = get.nConds(Dat)
    v = Dat.get_nConds;
end
function v = get_nConds(Dat)
    conds = Dat.get_conds;
    for iDim = numel(conds):-1:1
        v(iDim) = length(conds{iDim});
    end
end
function v = get_naConds(Dat)
    aConds = Dat.get_aConds;
    for iDim = numel(conds):-1:1
        v(iDim) = length(aConds);
    end
end

function v = get.ch(Dat)
    v = Dat.get_ch;
end
function set.ch(Dat, v)
    Dat.set_ch(v);
end

function v = get.rt(Dat)
    v = Dat.get_rt;
end
function v = get_rt(Dat)
    v = Dat.get_ds('RT');
%     v = Dat.ds.RT
end
function set.rt(Dat, v)
    Dat.set_rt(v);
end
function v = set_rt(Dat, v)
    Dat.set_ds('RT', v);
end

function v = get.accu(Dat)
    v = Dat.get_accu;
end
function v = get.answer(Dat)
    v = Dat.get_answer;
end
function v = get.cond(Dat)
    v = Dat.get_cond;
end
function v = get.dCond(Dat)
    v = Dat.get_dCond;
end

function v = get.accu0_all_dim(Dat)
    if Dat.to_determine_accu_from_bias_ch
        v = Dat.accu0_all_dim_bias_ch;
    else
        v = Dat.ch0_all_dim == Dat.answer0_all_dim;
    end
end
function v = get.accu_all_dim(Dat)
    v = Dat.accu0_all_dim(Dat.get_dat_filt, :);
end
function v = get.ch0_all_dim(Dat)
    v = [Dat.ds0(:, 'subjM'), Dat.ds0(:, 'subjC')];
end
function v = get.ch_all_dim(Dat)
    v = Dat.ch0_all_dim(Dat.get_dat_filt, :);
end
function v = get.answer0_all_dim(Dat)
    v = [Dat.ds0(:, 'corrM'), Dat.ds0(:, 'corrC')];
end
function v = get.answer_all_dim(Dat)
    v = Dat.answer0_all_dim(Dat.get_dat_filt, :);
end

function v = get.accu0_all_dim_bias_ch(Dat)
    v(:, Dat.dim_rel) = Dat.calc_accu_from_bias_ch( ...
        Dat.ds0.ch, Dat.ds0.cond, Dat.dim_rel, Dat.dim_irr);

    if Dat.n_dim_task == 2
        v(:, Dat.dim_irr) = Dat.calc_accu_from_bias_ch( ...
            Dat.ds0.ch, Dat.ds0.cond, Dat.dim_irr, Dat.dim_rel);
    else
        v(:, Dat.dim_irr) = true(size(Dat.ds0,1), 1);
    end
end
function v = get.accu_all_dim_bias_ch(Dat)
    v = Dat.accu0_all_dim_bias_ch(Dat.get_dat_filt, :);
end
function [accu, bias] = calc_accu_from_bias_ch(~, ch, cond, dim_ch, dim_sep)
    [~,~,sep] = unique(cond(:, dim_sep), 'rows');
    n_sep = max(sep);
    accu = true(size(ch, 1), 1);
    x = cond(:, dim_ch);
    y = ch(:, dim_ch) == 1;
    for i_sep = n_sep:-1:1
        incl = sep == i_sep;
        b = glmfit(x(incl), y(incl), 'binomial');
        bias1 = -b(1) / b(2);
        x_bias = x(incl) - bias1;
        accu(incl) = sign(x_bias) ~= -sign(y(incl) - 0.5);
        
        bias(i_sep) = bias1;
    end
end
function v = get_accu(Dat)
    if Dat.to_determine_accu_from_bias_ch
        v = Dat.accu_all_dim_bias_ch;
    else
        v = Dat.get_ch == Dat.get_answer;
    end
end
function set.dim_rel(Dat, v)
%     assert(isscalar(v));
    if isempty(Dat.W)
        Dat.dim_rel_ = v;
    else
        Dat.W.dim_rel_W = v;
    end
end
function v = get.dim_rel(Dat)
    if isempty(Dat.W)
        v = Dat.dim_rel_;
    else
        v = Dat.W.dim_rel_W;
    end
end
function v = get.dim_irr(Dat)
    v = setdiff(1:Data.Consts.n_dim, Dat.dim_rel);
end
function v = get.dim_used(Dat)
    if Dat.n_dim_task == 1
        v = Dat.dim_rel;
    else
        v = 1:Data.Consts.n_dim;
    end
end

function set_fun_bias_cond(Dat, v)
    assert(iscell(v));
    assert(numel(v) == 2);
    assert(all(cellfun(@(c) isempty(c) || isa(c, 'function_handle'), v)));
    Dat.fun_bias_cond = v;
end
function v = get_fun_bias_cond(Dat)
    v = Dat.fun_bias_cond;
end
function S = get.dim_pdf(Dat)
    S = Dat.get_dim_pdf; % Can be modified in subclasses
end
function S = get.dim_pdf_rel(Dat)
    S = Dat.get_dim_pdf_rel; % Can be modified in subclasses
end
end
%% Demo
methods
    function load_demo(Dat)
        Dat.set_path;
        Dat.load_data;
    end
end
%% Internal
methods (Static)
% function Dat = loadobj(Dat)
%     Dat.loadobj@FitData;
%     
%     warning('Call Dat.load_data and adapt_ds');
% %     Dat.load_data;
% %     Dat.adapt_ds;
% end
end
%% Empty methods - modify in subclasses
methods
    %% from ds
    function res = get_cond(Dat)
        error('Modify in subclasses!');
    end
    function res = get_ch(Dat)
        error('Modify in subclasses!');
    end
    function res = get_answer(Dat)
        error('Modify in subclasses!');
    end
    function res = get_RT(Dat)
        error('Modify in subclasses!');
    end
    
    % from ds0, for filtering
    function res = get_cond0(Dat)
        error('Modify in subclasses!');
    end
    function res = get_ch0(Dat)
        error('Modify in subclasses!');
    end
    function res = get_answer0(Dat)
        error('Modify in subclasses!');
    end
    function res = get_RT0(Dat)
        error('Modify in subclasses!');
    end
    
    % Others
    function res = get_dim_pdf(Dat)
        error('Modify in subclasses!');
    end
    function res = get_dim_pdf_rel(Dat)
        error('Modify in subclasses!');
    end
end
%% === Conds_oversample_factor related ===
properties (Access = protected)
    % length(conds_oversampled) ...
    % = (length(conds) - 1) * conds_oversample_factor(dim) + 1
    %
    % For example, conds = [1 2] with factor = 2 becomes conds = [1 1.5 2].
    %
    % Set it to ~10 to smooth predictions.
    conds_oversample_factor = [1 1]; 
end
properties (Dependent)
    conds_wo_oversample
    is_pred_done % Check if prediction is done, e.g., after oversampling.
end
methods
    function v = get.is_pred_done(Dat)
        v = Dat.get_is_pred_done;
    end
    function v = get_is_pred_done(Dat)
        try
            nConds = Dat.get_nConds;
            v = isequal(nConds, bml.array.sizes( ...
                    Dat.RT_pred_pdf, ...
                    Dat.dim_pdf.cond));
        catch err
            warning(err_msg(err));
            warning('Error in get_is_pred_done: returning True');
            v = true;
        end
    end
end
methods
    function v = get_conds(Dat)
        v = Dat.get_conds_wo_oversample;
        n_dim = length(v);
        
        for i_dim = 1:n_dim
            fac = Dat.get_conds_oversample_factor(i_dim);
            if fac ~= 1
                v0 = v{i_dim};
                n_cond0 = length(v0);
                n_cond1 = (n_cond0 - 1) * fac + 1;
                v1 = zeros(n_cond1, 1);
                
                for i_cond = 1:(n_cond0 - 1)
                    i_st = (i_cond - 1) * fac + 1;
                    i_en = i_cond * fac + 1;
                    v1(i_st:i_en) = linspace(v0(i_cond), v0(i_cond+1), ...
                        fac + 1);
                end
                v{i_dim} = v1;
            end
        end
    end
    function v = get.conds_wo_oversample(Dat)
        v = Dat.get_conds_wo_oversample;
    end
    function v = get_conds_wo_oversample(Dat)
        cond = Dat.get_cond;
        nDim = size(cond,2);
        for ii = nDim:-1:1
            v{ii} = uniquetol_wrap(cond(:,ii));
        end
    end
    function conds = get.conds0_wo_oversample_all_dim(Dat)
        n_dim = Data.Consts.n_dim;
        conds = cell(1, n_dim);
        for dim = n_dim:-1:1
            nam = Data.Consts.dimNames{dim};
            cond1 = Dat.ds0.(['cond' nam]);
            conds{dim} = unique(cond1);
        end
    end
    function conds = get.conds_wo_oversample_all_dim(Dat)
        n_dim = Data.Consts.n_dim;
        conds = cell(1, n_dim);
        for dim = n_dim:-1:1
            nam = Data.Consts.dimNames{dim};
            cond1 = Dat.ds.(['cond' nam]);
            conds{dim} = unique(cond1);
        end
    end
    function v = set_conds_oversample_factor(Dat, v, dim)
        assert(isnumeric(v));
        if ~exist('dim', 'var')
            assert(numel(v) == 2, ...
                ['Specify dim to oversample or ' ...
                 'give a 2-vector for oversample_factor: ' ...
                 'Usually you want to oversample only one dimension ' ...
                 '(for smooth curves) and not the other dimension, ' ...
                 'to have one curve per condition.']);
            Dat.conds_oversample_factor = v;
        else
            Dat.conds_oversample_factor(dim) = v;
        end
        assert(all(v) >= 1);
        assert(all(v == floor(v)));
    end
    function v = get_conds_oversample_factor(Dat, dim)
        if ~exist('dim', 'var')
            v = Dat.conds_oversample_factor;
        else
            v = Dat.conds_oversample_factor(dim);
        end
    end
end
%% Saving
methods
    function Dat = saveobj(Dat)
        Dat = Dat.saveobj@FitData;
            
        % Skip resetting RT_data_pdf_perm if invoked within serialize()
        % as in parallel fitting.
        dbst = dbstack;
        if ~any(strcmp('serialize', {dbst.name}))
            Dat.RT_data_pdf_perm = [];
        end
    end
end
end