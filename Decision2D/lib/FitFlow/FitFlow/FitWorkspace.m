classdef FitWorkspace < FitParamsForcibleSoft & PartialSaveTree & LinkProps % & matlab.unittest.TestCase
% 2013-2015 (c) Yul Kang.  yul dot kang dot on at gmail dot com.

%% Settings: prediction and costs
properties
    pred_fun = @(W) nan; % Simple predictions. Called from W.pred.
    cost_fun = @(W) nan; % For simple costs. Called from W.calc_cost.
    grad_fun = @(W) nan(1, length(W.get_vec));
    hess_fun = [];
    
    to_use_nested_fit = false; % For use with fitflow.NestedFit.
end
%% Internal variables
properties (Dependent)
    Data
end
properties (Access=private)
    % Cannot access without invoking set_Data and get_Data.
    Data_
end
properties (Transient)
    Fl
end
%% Methods
methods
function W = FitWorkspace(varargin)
    W.hess_fun = @(W) use(length(W.get_vec), @(n_th) nan(n_th, n_th));
    W.add_deep_copy({'Data_'});
    
    if nargin > 0
        W.init(varargin{:});
    end
end
function init(W, varargin)
    bml.oop.varargin2props(W, varargin, true);
    W.init_children(varargin{:});
end
function init_children(W, varargin)
    for child_name = fieldnames(W.children)'
        W.init_child(child_name{1}, varargin{:});
    end
end
function init_child(W, child_name, varargin)
    W.children.(child_name).init(varargin{:});
end
function [Fl, res] = fit(W, varargin)
    % [Fl, res] = fit(W, varargin)
    %
    % A template for fitting functions.
    % See also: FitFlow.fit_grid
    Fl = W.get_Fl;
    W.pred; % For the initial runPlotFcns
    res = Fl.fit(varargin{:});
end
function [Fl, res] = fit_grid(W, varargin)
    % A template for fitting functions.
    % See also: FitFlow.fit_grid
    Fl = W.get_Fl;
    res = Fl.fit_grid(varargin{:});
end
function Fl = get_Fl(W, new_Fl_instance, varargin)
    S = varargin2S(varargin, {
        'add_plotfun', true
        });
    
    if ~isempty(W.Fl)
        Fl = W.Fl;
        return;
    end
    if nargin >= 2
        Fl = new_Fl_instance;
    else
        Fl = FitFlow;
    end
    W.Fl = Fl;
%     Fl.set_W0(W); % .deep_copy);
    Fl.set_W(W);
%     try
%         Fl.W0.init_W0;
%         Fl.init_bef_fit;
%     catch err
%         warning(err_msg(err));
%     end

    if S.add_plotfun
        W.add_plotfun(Fl);
    end
end
function add_plotfun(W, Fl, varargin)
    % Fl is required to prevent endless recursion with get_Fl.
    W.add_plotfun_optimplotfval(Fl, varargin{:});
    W.add_plotfun_optimplotx(Fl, varargin{:});
end
function add_plotfun_optimplotfval(W, Fl)
    Fl.add_plotfun({
        @optimplotfval
        });
end
function add_plotfun_optimplotgrad(W, Fl, varargin)
    C = varargin2C({
        'src', 'grad'
        }, varargin);
    W.add_plotfun_optimplotx(Fl, C{:});
end
function add_plotfun_optimplotx(W, Fl, varargin)
    S = varargin2S(varargin, {
        'param_per_optimplotx', 5
        'src', 'x'
        });
    
    if nargin < 2 || isempty(Fl)
        Fl = W.get_Fl;
    end
    
    names_scalar = Fl.W.th_names_scalar;
    n = numel(names_scalar);
    for ii = 1:ceil(n / S.param_per_optimplotx)
        st = (ii - 1) * S.param_per_optimplotx + 1;
        en = min(st - 1 + S.param_per_optimplotx, n);
        
        switch S.src
            case 'x'
                Fl.add_plotfun({
                    str2func(sprintf( ...
                        '@(Fl,x,v,s) optimplotx(Fl,x,v,s,''ix'',%d:%d)', ...
                            st, en))
                    });
            case 'grad'
                Fl.add_plotfun({
                    str2func(sprintf( ...
                        '@(Fl,x,v,s) optimplotx(Fl,x,v,s,''ix'',%d:%d,''src'',''grad'')', ...
                            st, en))
                    });
        end
    end
    
    names_nonscalar = Fl.W.th_names_nonscalar;
    if ~isempty(names_nonscalar)
        for name = names_nonscalar(:)'
            switch S.src
                case 'x'
                    Fl.add_plotfun({
                        str2func(sprintf( ...
                            '@(Fl,x,v,s) optimplotx_vec(Fl,''%s'',x,v,s)', ...
                            name{1}))
                        });
                case 'grad'
                    Fl.add_plotfun({
                        str2func(sprintf( ...
                            '@(Fl,x,v,s) optimplotx_vec(Fl,''%s'',x,v,s,''src'',''grad'')', ...
                            name{1}))
                        });
            end
        end
    end
end
function [Fl, c] = test_Fl(W)
    Fl = W.get_Fl;
    c = Fl.get_cost(Fl.W.th_vec);
    disp(c);
end
end

%% Data sharing - always use root's.
methods
    function Dat = get.Data(W)
        Dat = W.get_Data;
    end
    function set.Data(W, Dat)
        W.set_Data(Dat);
    end
end
methods (Sealed)
    function Dat = get_Data_(W)
        % Called from DeepCopyable.deep_copy. No side effect.
        % Especially, doesn't get root's data.
        Dat = W.Data_;
    end
    function set_Data_(W, Dat)
        % Called from DeepCopyable.deep_copy. No side effect.
        % Especially, doesn't set root's data.
        W.Data_ = Dat;
    end
end
methods
    function Dat = get_Data(W)
        % get_Data can be extended in subclasses, but Data_ can be retrieved
        % only by using FitWorkspace.get_Data.
        src = W.get_Data_source;
        Dat = src.get_Data_;
    end
    function set_Data(W, Dat)
        % set_Data can be extended in subclasses, but Data_ can be changed
        % only by using FitWorkspace.set_Data.
        src = W.get_Data_source;
        src.set_Data_(Dat);
        Dat.W = src;
    end
    function set_root(W, new_root)
        % When the W itself becomes a root,
        % set its Data to the previous root's Data.

        prev_root = W.get_Data_source;
        W.set_root@FitParams(new_root);
        
        if W.is_root
            W.set_Data(prev_root.get_Data);
%             disp('Self is root!'); % DEBUG
        end
        
%         % DEBUG
%         disp([ ...
%             sprintf('%d : ', W.Data.get_Time == new_root.get_Time), ...
%             sprintf('%d : ', W.Data.get_Time == new_root.Data.get_Time), ...
%             sprintf('%d : ', new_root.get_Time == new_root.Data.get_Time), ...
%             class(W), ' : ', ...
%             class(prev_root), ' <- ', ...
%             class(new_root)]);
    end
    function src = get_Data_source(W)
        % Defaults to the root. 
        % Modify, e.g., to self, in subclasses if necessary.
        % Then set_root should be changed as well.
        src = W.get_root;
    end
end
%% Data - etc
methods
    function n = get_n_tr(W)
        if isa(W.Data, 'FitData')
            n = W.Data.get_n_tr;
        else
            n = nan;
        end
    end
    function n = get_n_tr0(W)
        if isa(W.Data, 'FitData')
            n = W.Data.get_n_tr0;
        else
            n = nan;
        end
    end    
end
%% Params and other fields - obsolete. Use VisitorToTree methods.
methods
% function Params2W(W, fields)
%     % Copies parameters to direct properties. Use optionally.
%     
%     th = W.get_struct('th'); % Not struct_all - only direct properties.
%     if nargin < 2 || (~iscell(fields) && isempty(fields))
%         fields = fieldnames(th); 
%     end
%     copyFields(W, th, fields, false, false);    
% end
% function Params2W_recursive(W)
%     % Copies parameters to direct properties recursively. Use optionally.
%     % If a workspace uses this,
%     % all parameters must also be a direct property.
%     % Not called from FitFlow automatically.
%     W.Params2W;
%     for sub = fieldnames(W.sub)'
%         W.sub.(sub{1}).Params2W_recursive;
%     end
% end
end

%% Subworkspace management
methods
    function set_sub_from_props(W, props)
        % May use VisitableTree.add_children_props later, save the time checking
        if nargin < 2, props = {}; end
        if ischar(props), props = {props}; end
        props = props(:);
        assert(all(cellfun(@ischar, props)));
        for prop = props'
            assert(isempty(W.(prop{1})) || isa(W.(prop{1}), 'FitParams'));
    %         W.add_child(W.(prop{1}), prop{1});
    %         
    %         % To keep W.(prop) == W.get_child(prop) after deep_copy
    %         W.add_deep_copy(prop{1});
        end
        W.add_children_props(props);
    end
    function remove_child(W, child)
        W.remove_child@FitParamsForcibleSoft(child);
        W.Data.W = W; % Recover link
    end
end
%% Deprecated - init_W0
methods
function customize_th_for_Data(W, varargin)
    % Ignored if not implemented
end

function init_W0(W, props, varargin)
    % init_W0(W, props, varargin)
    %
    % Initialization that does not vary with the values of th0, lb, or ub
    % (which might be set by grid).
    %
    % Customizes parameters and subworkspaces.
    % Do add parameters and subworkspaces on construction, not here,
    % so that W works consistently without init_W0.
    %
    % Parameters and subworkspaces are used by FitFlow to determine the parameters
    % to feed optimizers such as fmincon.
    %
    % init_W0 is NOT called automatically by FitFlow, but perhaps called by
    % higher-level workspaces' init_W0, which in turn is called separate from
    % FitFlow.
    %
    % It is good to allow constructors to work without arguments,
    % e.g., to allow calling regular methods like static methods.
    % 
    % To use a workspace without using its default sets of parameters,
    % remove the parameters using W.remove_params({..}) or just by setting
    % W.fixed.(param_name) = true.
    %
    % %% Modify in subclasses %%    
    
    % Placeholder. Copies properties into subworkspaces.
    if nargin < 2, props = {}; end
    W.set_sub_from_props(props);
    
    % Template + Chain-of-responsibility.
    W.init_W0_bef_subs(varargin{:});
    
    % Call init_W0 of subworkspaces
    %
    % init_W0_subs : Initialize subworkspaces.
    % Since this might be needed either before or after init_W0 of
    % the parent workspace, no template (that specifies order) is provided.
    W.init_W0_subs; 
    
    % Template + Chain-of-responsibility.
    W.init_W0_aft_subs(varargin);
end
% FIXIT: Consider mergining into init_W0, or call from init_W0
function init_W0_bef_subs(W, varargin)
end
function init_W0_aft_subs(W, varargin)
end
function init_W0_subs(W) % , subs)
%     if nargin < 2, subs = fieldnames(W.sub)'; end

    for child = W.get_children
        child{1}.init_W0;
    end
end

%% FitFlow Interface - limit to get_cost Automatically called if a top-level workspace
function init_bef_fit(W)
    % W = init_bef_fit(W, varargin)
    % Initialize W after 
    %
    % FitFlow.fit calls this once before the first iteration,
    % on the top workspace only.
    %
    % %% Modify this in subclasses %%    
end
function pred(W)
    % pred(W, varargin)
    % Called from get_cost.
    % Doesn't involve c, unlike calc_cost.
    % - In case prediction is time-consuming, separating pred from get_cost
    % might be a good idea.
    % - In case prediction is fast, skip implementing pred,
    % so that the intermediate state need not be 
    % stored within W.
    %
    % %% Modify this in subclasses %%    
    
    W.pred_fun(W);
end
function [cost, grad, hess, cost_sep] = calc_cost(W)
    % [cost, grad, hess] = calc_cost(W, varargin)
    % Usually calculating cost is much faster than
    % prediction. 
    % Also, predicted state might be used in other classes
    % that uses other cost functions.
    % Therefore, it might be good to separate
    % pred from calc_cost.
    %
    % %% Modify this in subclasses %%    
    if nargout >= 4
        [cost, cost_sep] = W.cost_fun(W);
    else
        cost = W.cost_fun(W);
    end
    
    if nargout >= 2
        grad = W.grad_fun(W);
    end
    if nargout >= 3
        hess = W.hess_fun(W);
    end
end
function varargout = get_cost(W, varargin)
    % [cost, grad, hess, cost_sep] = get_cost(W, varargin)
    %
    % cost, grad, hess: as in fmincon.
    % cost_sep: element-wise cost.
    %
    % Follows Hollywood Principle:
    % does not call FitFlow methods. FitFlow sets parameters if needed.
    % Called from FitFlow.fit on every iteration after pred.
    W.pred;
    [varargout{1:nargout}] = W.calc_cost;
end
end
methods
    function W = test_nonscalar_param(W0)
        %%
        W = feval(class(W0));
        W.add_params({
            {'vec', 1:5, zeros(1,5), 10 + zeros(1,5)}
            {'vec_mixed', 1:5, [0 0 3 4 5], [10 10 3 4 5]}
            {'vec_fixed', 1:5, 1:5, 1:5}
            {'scalar1', 2, 0, 10}
            {'scalar_fixed', 3, 3, 3}
            {'scalar2', 5, 3, 6}
            });
        disp(W);
        
        %%
        Fl = W.get_Fl;
        disp(Fl);
        
        %%
        W.cost_fun = @(W) ...
              sum((W.th.vec - (6:-1:2)).^4) ...
            + sum((W.th.vec_mixed(1:2) - [2 8]).^2) ...
            + (W.th.scalar1 - 8) .^ 2 ...
            + (W.th.scalar2 - 8) .^ 2;
        disp(W.get_cost);
        
        %%
        Fl = W.get_Fl;
        Fl.fit;
        disp(Fl.res.th.vec);
    end
end
end