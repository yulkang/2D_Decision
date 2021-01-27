classdef Main < Fit.Common.CommonWorkspace
    % Fit.Common.Main
    
    % 2016 YK wrote the initial version.
    
%% Properties - Settings    
properties (Dependent)
    drift_kind
    bound_kind
    sigmaSq_kind
    tnd_kind
    n_tnd
    miss_kind
    fix_miss
    
    drift_short
    bound_short
    sigmaSq_short
    tnd_short
end
%% Naming / Model setting
properties
    model = ''; % Description of the settings. Used in file names
    import_th = '';
end
methods
    function fs = get_file_fields0(W)
        fs = W.get_file_fields0@Fit.Common.CommonWorkspace;
        fs = union_general(fs, ...
            {
                'model', 'mdl'
                'drift_short', 'dft'
                'bound_short', 'bnd'
                'sigmaSq_short', 'ssq'
                'tnd_short', 'tnd'
                'n_tnd', 'ntnd'
                'miss_kind', 'msk'
                'fix_miss',         'msf'
                'import_th',            'ith'
%                 'fix_drift_bias_1', 'db1f'
%                 'fix_drift_bias_2', 'db2f'
%                 'fix_bias_irr_1',   'bif1'
%                 'fix_bias_abs_irr_1', 'baif1'
%                 'fix_bias_irr_2',   'bif2'
%                 'fix_bias_abs_irr_2', 'baif2'
            }, 'stable', 'rows');
    end
    
    function v = get.drift_kind(W)
        try
            v = W.get_drift_kind;
        catch err
            warning(err_msg(err));
            v = '';
        end            
    end
    function set.drift_kind(W, v)
        W.set_drift_kind(v);
    end

    function v = get.bound_kind(W)
        try
            v = W.get_bound_kind;
        catch err
            warning(err_msg(err));
            v = '';
        end
    end
    function set.bound_kind(W, v)
        W.set_bound_kind(v);
    end

    function v = get.sigmaSq_kind(W)
        try
            v = W.get_sigmaSq_kind;
        catch err
            warning(err_msg(err));
            v = '';
        end
    end
    function set.sigmaSq_kind(W, v)
        W.set_sigmaSq_kind(v);
    end

    function v = get.tnd_kind(W)
        try
            v = W.get_tnd_distrib;
        catch err
            warning(err_msg(err));
            v = '';
        end
    end
    function set.tnd_kind(W, v)
        W.set_tnd_distrib(v);
    end

    function v = get.n_tnd(W)
        try
            v = W.get_n_tnd;
        catch err
            warning(err_msg(err));
            v = '';
        end            
    end
    function set.n_tnd(W, v)
        W.set_n_tnd(v);
    end
    
    function v = get.miss_kind(W)
        if isfield(W.children, 'Miss')
            v = W.Miss.kind;
        else
            v = '';
        end
    end
    function set.miss_kind(W, v)
        if isfield(W.children, 'Miss')
            W.set_Miss(v);
        else
            warning('Miss is not a child of %s: cannot set miss_kind=%s\n', ...
                class(W), v);
        end
    end
    
    function set.fix_miss(W, v)
        W.Miss.fix_miss = v;
    end
    function v = get.fix_miss(W)
        try
            v = W.Miss.fix_miss;
        catch err
            warning(err_msg(err));
            v = '';
        end
    end
end
properties (Constant)
    drift_orig_short = {
        'Const',        'C'
        'Exp',          'E'
        'Power',        'P'
        'IrrSep',       'S'
        'IxnHistory',   'IH'
        'BetaMean',     'M'
        'IrrSepExp',    'SE'
        };
    bound_orig_short = {
        'Const',        'C'
        'BetaCdf',      'B'
        'BetaMean',     'M'
        'BetaMeanAsym', 'A'
        'BetaMeanAsymDec', 'AD'
        'BMA2',         'A2'
        'CosBasis',     'O'
        };
    sigmaSq_orig_short = {
        'QuadPreDrift', 'QPrD'
        'LinearMin',    'LM'
        'LinearMinPreDrift', 'LMPrD'
        'Linear',       'L'
        'LinearPreDrift', 'LPrD'
        'Const',        'C'
        'PowerIxn',     'PI'
        };
    tnd_orig_short = {
        'gamma',        'g'
        'halfnorm',     'h'
        'normal',       'n'
        'invgauss',     'i'
        };
end
methods
    function v = get.drift_short(W)
        v = W.get_prop_short('drift');
    end
    function set.drift_short(W, v)
        v0 = W.get_prop_orig('drift', v);
        if isprop(W, 'Dtb')
            W.Dtb.set_Drift(v0);
        else
            warning('No property Dtb exists for %s\n', class(W));
        end
    end
    function v = get.bound_short(W)
        v = W.get_prop_short('bound');
    end
    function set.bound_short(W, v)
        v0 = W.get_prop_orig('bound', v);
        if isprop(W, 'Dtb')
            W.Dtb.set_Bound(v0);
        else
            warning('No property Dtb exists for %s\n', class(W));
        end
    end
    function v = get.sigmaSq_short(W)
        v = W.get_prop_short('sigmaSq');
    end
    function set.sigmaSq_short(W, v)
        v0 = W.get_prop_orig('sigmaSq', v);
        if isprop(W, 'Dtb')
            W.Dtb.set_SigmaSq(v0);
        else
            warning('No property Dtb exists for %s\n', class(W));
        end
    end    
    function v = get.tnd_short(W)
        v = W.get_prop_short('tnd');
    end
    function set.tnd_short(W, v)
        v0 = W.get_prop_orig('tnd', v);
        W.set_tnd_distrib(v0);
    end    
    function kind_short = get_prop_short(W, prop)
        kind = W.([prop '_kind']);
        orig_short = W.([prop '_orig_short']);
        ix = find(strcmp(kind, orig_short(:,1)));
        if isempty(ix)
            warning('Cannot shorten drift=%s\n', kind);
            kind_short = kind;
        else
            kind_short = orig_short{ix,2};
        end
    end
    function kind_orig = get_prop_orig(W, prop, kind_short)
        orig_short = W.([prop '_orig_short']);
        ix = find(strcmp(kind_short, orig_short(:,2)));
        if isempty(ix)
            warning('Cannot find a short form of drift=%s\n', kind_short);
            kind_orig = kind_short;
        else
            kind_orig = orig_short{ix,1};
        end
    end
end
%% fit_unit - deprecated
% %% Modify in subclasses
% methods
%     function S_fit_unit = get_S_fit_unit_default(~)
%         S_fit_unit = varargin2S({
%             'subj', Data.Consts.subjs_RT{1}
%             'parad', 'RT'
%             'task', 'A'
%             ...
%             'fit_args', {}
%             });
%     end    
%     function [Fl, S_fit_unit, res] = get_Fl(W, varargin)
%         S_fit_unit = W.get_S_fit_unit(varargin{:});
%         res = struct;
%         Fl = W.get_Fl@Fit.Common.CommonWorkspace;
%     end
% end
% %% Common
% methods
%     function [Fl, S, res] = fit_unit(W, varargin)
%         % [Fl, S, res] = fit_unit(W, varargin)
%         
%         S = W.get_S_fit_unit(varargin{:});
%         res = struct;
%         
%         %% Get Fl
%         C = S2C(S);
%         [Fl, S, res_Fl] = W.get_Fl(C{:});
%         res = copyFields(res, res_Fl);
%         
%         %% Test
%         cost = Fl.get_cost;
%         disp(cost);
%         Fl.runPlotFcns;
%         
%         %% Fit
%         Fl.fit(S.fit_args{:});
%         
%         %% Results
%         disp(Fl.res);
%         disp('res.th');
%         disp(Fl.res.th);
%         disp('res.se');
%         disp(Fl.res.se);
%     end
%     function S_fit_unit = get_S_fit_unit(W, varargin)
%         S_fit_unit = varargin2S(varargin, W.get_S_fit_unit_default);
%     end
% end
end