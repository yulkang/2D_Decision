classdef FitHistory < matlab.mixin.Copyable
% FitHistory
%
% 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.    
properties
    n_iter     = 0;
    max_iter   = 1000;
    th_names   = {};
    history    = dataset;
    
    disp_iter  = false;
    
    t_last_iter = nan;
    t_last_iter_duration = nan;
    t_fit_started = nan;
    t_fit_duration = nan;

    Params
    
    logged_props = {'n_iter'};
    logged_w = {};
end
methods
function H = FitHistory(varargin)
    H = varargin2fields(H, varargin, false);
end
function init_bef_fit(H, Params)
    if nargin < 2 || isempty(Params)
        Params = H.Params;
    end
    assert(isa(Params, 'FitParams'));
    
    % Since Params is set in iterate to get names, etc., 
    % it needs to be copied.
    H.Params = Params; % .deep_copy_Params; % deep_copy_Params is buggy
    H.n_iter  = 0;
    H.th_names = hVec(H.Params.get_names_recursive);
    
%     th0 = H.Params.get_struct_recursive;
    
    H.history = dataset;
    for column = [H.logged_props(:)', {'fval', 't_clock_last_iter', 't_elapsed'}]
        H.history.(column{1}) = nan(H.max_iter, 1);
    end
    for th = H.th_names
        H.history.(th{1}) = cell(H.max_iter, 1); % nan(H.max_iter, numel(th0.(th{1})));
    end
    H.t_fit_started = tic;
    H.t_last_iter = tic;
end
function stop = iterate(H, x, optimValues, state)
    stop = false;
    H.n_iter = H.n_iter + 1;
    if isnan(H.t_last_iter)
        warning('H.t_last_iter not stored! t_last_iter_duration is set to 0.');
        H.t_last_iter_duration = 0;
    else
        H.t_last_iter_duration = toc(H.t_last_iter);
    end
    H.t_last_iter = tic;
    
    H.history.fval(H.n_iter,1) = optimValues.fval;
    for prop = H.logged_props(:)'
        H.history.(prop{1})(H.n_iter,1) = H.(prop{1});
    end
    
    if isnan(H.t_fit_started)
        warning('H.t_fit_started not stored! t_fit_duration is set to 0.');
        H.t_fit_duration = 0;
    else
        H.t_fit_duration = toc(H.t_fit_started);
    end
    H.history.t_elapsed(H.n_iter,1) = H.t_fit_duration;
    H.history.t_clock_last_iter(H.n_iter, 1) = now;
        
    H.Params.set_vec_recursive(x);
    S = H.Params.get_struct_recursive;
    for th = H.th_names
        % Convert to row vector
        v = S.(th{1})(:)';
        if ~iscell(H.history.(th{1}))
            H.history.(th{1}) = row2cell(H.history.(th{1}));
        end 
        H.history.(th{1}){H.n_iter, 1} = v; % (H.n_iter, 1:length(v)) = v;
    end
    
    if H.disp_iter
        stop = stop || H.display_th(x, optimValues);
    end
end
function history = finish(H, varargin)
    S = varargin2S(varargin, {
        'record_t_finish', true
        });
    if S.record_t_finish
        H.t_fit_duration = toc(H.t_fit_started);
    end
    H.history = H.history(1:H.n_iter,:);
    
    % Enforce matrix form
    for col = H.history.Properties.VarNames(:)'
        H.history.(col{1}) = cell2mat2(H.history.(col{1}));
    end
    
    if nargout >= 1, history = H.history; end
end

%% Add-hoc addition
function add_history(H, name, val)
    if isempty(val), return; end
    val = val(:)';
    len = length(val);
    H.history.(name)(H.n_iter, 1:len) = val;
end

%% Display
function f = get_disp_fun(H)
    f = @(varargin) H.display_th(varargin{:});    
end
function stop = display_th(H, x, optimValues, state)
    stop = false;
    
    fprintf('Iter %d (fval=%1.5g)', optimValues.iteration, optimValues.fval);
%         disp(x); % Workaround MATLAB's bug
%         cfprintf(' %s=%1.5g', th_names, x); % alternative
    
    th_names = H.Params.get_names_recursive;
    for ii = 1:length(th_names)
        fprintf('  %s = %1.5g', th_names{ii}, x(ii));
    end
    fprintf(' (%1.3fs/%1.0fs total)', ...
        H.t_last_iter_duration, H.t_fit_duration);
    Fl.t_last_iter = tic;
    fprintf('\n');
end
end
end