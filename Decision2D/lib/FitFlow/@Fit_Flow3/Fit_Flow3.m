classdef Fit_Flow3 < matlab.mixin.Copyable
    % - Makes Fit_Flow2 compatible with parallel fitting.
    % - Simplifies functions - they get W only.
    %
    % The motivation is to make input-output relationship
    % flexible and transparent in the script,
    % facilitating combination and reuse of existing functions.
    % 
    % The method add_fun() allows arbitrary input-output relationship,
    % while the property W provides the common workspace.
    %
    % Caveats:
    % - To save disk space, 
    %   set dat_path and dat_filt to use common data across different models.
    %
    % - To ensure replicability,
    %   - Use Fl.W0. Fl.W is erased by Fl.saveobj.
    %   - Ensure Fl.W = Fl.W0 before running a fit.
    %
    % 2015 (c) Yul Kang. hk2699 at columbia dot edu.
    
    properties
        W     = struct; % workspace. Deleted by saveobj().
        W0    = struct; % initial values needed to reconstruct W with Fl.run().
        
        %% Parameters
        th    = struct; % struct of thetas
        th0   = struct; % struct of initial guesses
        th_lb = struct; % struct of lower bounds
        th_ub = struct; % struct of upper bounds
        th_nested = {}; % fields of W that are copied to res.nested, and copied back to W on res2W, after copying W0 before copying th.
        
        %% Data
        dat  = dataset; % dataset of data.
        dat_path = '';  % path to find the data. If specified, save/load from/to the path 
                        % as a mat file containing 'dat', rather than saving the data with Fl.
        dat_filt = ':'; % load only the specified portion of the data from dat_path. Defaults to all.
        col_map  = []; % col_map.(column_name_in_Fl) = column_name_in_dat_file
        
        %% Optimization
        % current scalar cost.
        cost = inf; 
        
        % constraints: See help fmincon_cond
        constr = {};
        
        % res.(solver)
        % : Last fit result from the solver.
        %   Fields: Copy of the properties - fit_arg, fit_opt, fit_res
        res  = struct; 
        
        % handle_mode
        % : If true, assigns W back to Fl.W on every evaluation (see cost_fun()).
        %   Useful in plotting but may be incompatible with parallel processing.
        handle_mode = false; 
        plot_opt = varargin2S({}, {
            'per_iter', 1 % Plot on every iteration
            });
        
        % id: Prevent confusion between multiple Fl 
        % without using Fl (handle) explicitly during fitting
        % which adds overhead during parallel processing.
        % Used in identification in non-handle functions, e.g., history
        id = ''; 
        
        % fit_arg.(solver)
        % : Struct of default positional arguments for the solver.
        %   Field order determines the arguments' order.
        fit_arg = varargin2S({}, {
            'fmincon', @(Fl) varargin2S({
                'fun', Fl.cost_fun
                'x0',  Fl.th0_vec(~Fl.th_fix_vec)
                'A',   []
                'b',   []
                'Aeq', []
                'beq', []
                'lb',  Fl.th_lb_vec(~Fl.th_fix_vec)
                'ub',  Fl.th_ub_vec(~Fl.th_fix_vec)
                'nonlcon', []
                'options', {}
                });
            'fminsearchbnd', @(Fl) varargin2S({
                'fun', Fl.cost_fun
                'x0',  Fl.th0_vec(~Fl.th_fix_vec)
                'lb',  Fl.th_lb_vec(~Fl.th_fix_vec)
                'ub',  Fl.th_ub_vec(~Fl.th_fix_vec)
                'options', {}
                });
            'etc_', @(Fl) varargin2S({
                'fun', Fl.cost_fun
                'x0',  Fl.th0_vec(~Fl.th_fix_vec)
                });
            });
        
        % fit_opt.(solver)
        % : Struct of default options for the solver.
        fit_opt = varargin2S({}, {
            'fmincon', @(Fl) varargin2S({
                'PlotFcns',  Fl.plotfun
                'OutputFcn', Fl.outfun % includes Fl.OutputFcn, history, etc
                });
            'fminsearchbnd', @(Fl) varargin2S({
                'PlotFcns',  Fl.plotfun
                'OutputFcn', Fl.outfun % includes Fl.OutputFcn, history, etc
                });
            'etc_',    @(Fl) varargin2S({}, {});
            });
        
        % fit_out.(solver)
        % : Cell array of output names from the solver.
        fit_out = varargin2S({}, {
            'fmincon',      {'x', 'fval', 'exitflag', 'output', 'lambda', 'grad', 'hessian'}
            'fminsearchbnd',{'x', 'fval', 'exitflag', 'output'}
            'MultiStart',   {'x', 'fval', 'exitflag', 'output', 'solutions'}
            'etc_',         {'x', 'fval'}
            });
        
        %% History
        n_iter              = 0;
        max_iter            = 1000;
        save_history        = true;
        disp_cost           = false;
        
        %% PlotFcns, OutputFcns
        % Cell array of plot functions, evaluated in sequence.
        % Use only @(Fl) @(x,fval,state) ... and inside it, Fl-related values only,
        % for recovery after load.
        PlotFcns = {
            };
        
        % Cell array of output functions, evaluated in sequence
        % Use only @(Fl) @(x,fval,state) ... and inside it, Fl-related values only,
        % for recovery after load.
        OutputFcns = {
            };
        
        %% Grid
        grid_spec = {}; % each field corresponds to a parameter
        grid_opt  = varargin2S({
            'restrict', true
            'parallel', false
            });
        
        %% Functions
        % fun
        % : Use only @(W) and W-related variables (no outside variables) 
        %   to enable recovery after load.
        fun  = struct;
        fun_out = struct;
%         nargin_fun = struct; % Unused in Fit_Flow3
        fun_opt = struct; % fun_opt.(opt).(fun)
        fun_iter   = {}; % Functions to include in the interation.
        
        %% Subflows
        % Subflows has their workspace in W.(sub).
        % At the beginning of every iteration, parameters listed in 
        % sub_th.(sub) is copied to W.(sub).
        %
        % On the main th, the name appears as th.(['name_' sub]).
        % (Make sure to reserve the names!)
        % On the subflow workspace, it is just W.(sub).(name).
        %
        % sub2main(W, sub, vars) and main2sub(W, sub, vars) copies 
        % variables between the main workspace and the sub workspace,
        % attaching and detaching '_sub' as appropriate.
        sub_th = struct;
        
        %% Debug
        debug_mode = false;
        dbstop_if = varargin2S({}, {
            'naninf', false
            });
        debug = varargin2S({}, { % Goes to calc_cost_opt_C
            'cost_inf2realmax', true
            'cost_nan2realmax', true
            'th_imag2realmax', true
            });        
        
        % Access is not private so that subclasses can modify it in the constructor.
        VERSION
        VERSION_DESCRIPTION
    end
    
    properties (Dependent)
        th_vec % vector of thetas
        th0_vec
        th_ub_vec
        th_lb_vec
        th_names
        
        th_fix_vec % = (th_lb_vec == th_ub_vec). Vector of fixed thetas. excluded at calc_cost.
        th_fix % = struct built on th_fix_vec.
        
        funs
        fun_names
        fun_init
        
        n_th
        n_dat
        
        verFitFlow
    end
    
    properties (Transient)
        % Functions
        f_record_history    = [];
        f_calc_cost_iter    = [];
        f_calc_cost_init    = [];
        
        % For use between submethods of grid_fit 
        res_all = {};
    end
    
    %% ===== Regular methods =====
    methods
        %% ----- Create -----
        function Fl = Fit_Flow3
            Fl.VERSION = 3;
            Fl.VERSION_DESCRIPTION = 'Functions get W only.';
            if isempty(Fl.id)
                Fl.id = randStr(7);
            end
        end
        
        function add_fun_init(Fl, funs)
            % Shorthand for add_fun(Fl, {{... 'iter', false}, ...})
            %
            % add_fun_init(Fl, funs)
            
            for ii = 1:size(funs, 1)
                add_fun(Fl, funs{ii}{:}, 'iter', false);
            end
        end
        
        function add_fun(Fl, fun_name, fun_out, fun, varargin)
            % add_fun(Fl, 'fun_name', {'out_1', 'out_2', ...}, fun, 'opt1', opt1, ...)
            % add_fun(Fl, {
            %   {'fun_name', {'out_1', 'out_2', ...}, fun, 'opt1', opt1, ...}
            %   {...}
            %   ...
            %   }, [{common_options}])
            %
            % If iter = true (default), the function is run on every
            % iteration during fitting.
            % If iter = false, the function can still be run manually
            % by run(Fl, {'function_name'}).
            %
            % Options, defaults
            % -----------------
            % 'inp',  {}   % give W as the input argument. To give fields, specify their names.
            % 'iter', true % If false, not run during fitting.
            % 
            % 'sub',  ''   % If nonempty, outputs are saved to W.(sub).(out1) rather than W.(out1).
            % 'use_varargout', false
            %
            % Functions
            % ---------
            % Each function has one of the following formats:
            %   W = fun(W)            % Saves memory, but harder to make as a one-liner
            %   out = fun(W) % out can be {out_1, ...}. Easy to write one-liner
            %   [out_1, ...] = fun(W.(in_1), ...) % Can reuse existing functions. Set 'use_varargout' as true.
            % where W is the workspace struct, Fl.W. 
            % Leave out empty to use W = fun(W) form.
            %
            % All inputs are optional.
            % The output is a cell array. Each element will be set to W.(out_k) 
            % after execution of the function.
            %
            % Options
            % -------
            % 'inp',  {} % List of input fields. Omit to give W itself. New in Fit_Flow3, to allow direct incorporation of existing functions
            % 'iter', true
            % 'use_varargout', false
            % 'sub',  ''
            % 'in_type', ''  % 'W': fields of W or W itself; 'Fl': properties of Fl or Fl itself (handle mode only)
            % 'out_type', '' % 'W': fields of W or W itself; 'Fl': properties of Fl or Fl itself (handle mode only)            
            
            if iscell(fun_name)
                if nargin < 3, fun_out = {}; end
                
                for ii = 1:size(fun_name, 1)
                    add_fun(Fl, fun_name{ii}{:}, fun_out{:});
                end
                
            elseif ischar(fun_name)
                % Default fun_opt
                S = varargin2S(varargin, {
                    'inp',  {} % List of input fields. Omit to give W itself. New in Fit_Flow3, to allow direct incorporation of existing functions
                    'iter', true
                    'use_varargout', false
                    'sub',  ''
                    'in_type', ''  % 'W': fields of W or W itself; 'Fl': properties of Fl or Fl itself (handle mode only) - will be used in FitFlow5.
                    'out_type', '' % 'W': fields of W or W itself; 'Fl': properties of Fl or Fl itself (handle mode only) - will be used in FitFlow5.
                    });
               
                % fun_name -> fun_name_sub
                if ~isempty(S.sub)
                    fun_name = [fun_name, '_' S.sub];
                end
                
                % Fill in properties: fun, fun_out, fun_opt, fun_iter
                Fl.fun.(fun_name)        = fun;
                Fl.fun_out.(fun_name)    = fun_out;
%                 Fl.nargin_fun.(fun_name) = nargin(fun); % unused in Fit_Flow3
                
                for opt = fieldnames(S)'
                    Fl.fun_opt.(opt{1}).(fun_name) = S.(opt{1});
                end
                
                if S.iter
                    Fl.fun_iter = union(Fl.fun_iter, {fun_name}, 'stable');
                end
            else
                error('The first argument must be either cell or char!');
            end
        end
        
        function show_fun(Fl)
            fs = Fl.fun_names;
            n  = length(fs);
            
            for ii = 1:n
                f = fs{ii};
                
                out = Fl.fun_out.(f);
                if ischar(Fl.fun.(f))
                    fun = Fl.fun.(f);
                else
                    fun = func2str(Fl.fun.(f));
                end
                
                if isempty(out), out = {'W'}; end
                out_str = str_bridge(', ', out);
                
                fprintf('%15s: [%15s] = %s', f, out_str, indentStr(fun, 'indent', 37));
            end
        end
        
        function add_th_S(Fl, guesses, lbs, ubs, opts)
            % add_th_S(Fl, guesses, lbs, ubs, opts)
            
            fs = fieldnames(guesses)';
            n  = length(fs);
            
            if nargin < 4, opts = struct; end
            
            for ii = 1:n
                f = fs{ii};
                
                th0 = guesses.(f);
                
                lb  = lbs.(f);
                ub  = ubs.(f);
                try opt = opts.(f); catch, opt = {}; end
                
                Fl.add_th(f, th0, lb, ub, opt{:});
            end
        end
        
        function add_th(Fl, th_name, th0, th_lb, th_ub, varargin)
            % add_th(Fl, 'th_name', th0, th_lb, th_ub, ['sub', '', 'fixed', false, ...])
            % add_th(Fl, {{'th_name', th0, th_lb, th_ub}, {...}, ...}, {common_options})
            %
            % Each row specifies one parameter. 
            % th0, th_lb, th_ub are scalar numericals.
            % When either th_lb or th_ub are omitted, they are set to
            % -inf and inf, respectively.
            
            optDefault = {
                    'sub', ''
                    'fix', false
                    };
            
            if iscell(th_name)
                if nargin < 3, th0 = {}; end
                C = varargin2C(th0, optDefault);
                
                for ii = 1:size(th_name, 1)
                    add_th(Fl, th_name{ii}{:}, C{:});
                end
                
            elseif ischar(th_name)
                S = varargin2S(varargin, optDefault);
                
                if ~isempty(S.sub)
                    if isfield(Fl.sub_th, S.sub)
                        Fl.sub_th.(S.sub) = union(Fl.sub_th.(S.sub)(:)', {th_name}, 'stable');
                    else
                        Fl.sub_th.(S.sub) = {th_name};
                    end
                    th_name = [th_name, '_' S.sub];
                end
                
                Fl.th.(th_name)    = th0;
                Fl.th0.(th_name)   = th0;
                
                if nargin >= 4
                    Fl.th_lb.(th_name) = th_lb;
                else
                    Fl.th_lb.(th_name) = -inf(size(th0));
                end
                if nargin >= 5
                    Fl.th_ub.(th_name) = th_ub;
                else
                    Fl.th_ub.(th_name) = inf(size(th0));
                end
            else
                error('The first argument must be either a string or a cell array of strings!');
            end
        end
        
        function add_constr(Fl, constr, varargin)
            % A wrapper for fmincon_cond.
            %
            % add_constr(Fl, constr, varargin)
            %
            % Give the second input for fmincon_cond (conds).
            % Parameter names will be filled in from Fl.th_names.
            % Results are saved in Fl.constr.
            %
            % TODO:
            % Also can give a function of W to specify conditions.
            % The function is evaluated at the time of initialization.
            %
            % EXAMPLE:
            % constr: {{'A',  {'B_exp_y', 'B'},       [1, -1], -0.01}}
            %
            % OPTIONS
            % -------
            % 'sub', ''
            
            S = varargin2S(varargin, {
                'sub', ''
                });
            
            % Convert variable names if sub is specified.
            if ~isempty(S.sub)
                for ii = 1:length(constr)
                    constr{ii}{2} = cellfun(@(s) [s '_' S.sub], constr{ii}{2}, ...
                        'UniformOutput', false);
                end
            end
            
            % Add constraint
            Fl.constr = unionCell(Fl.constr, constr);
        end
        
        %% ----- Edit -----
        function remove_th(Fl, th_name, varargin)
            % Remove parameter(s).
            % 
            % remove_th(Fl, th_name, ...)
            %
            % th_name : char or cell.
            
            S = varargin2S(varargin, {
                'sub', ''
                });
            
            if iscell(th_name)
                for ii = 1:numel(th_name)
                    remove_th(Fl, th_name{ii}, varargin{:});
                end
            else
                %% Reverse what's done in add_th
                if ~isempty(S.sub)
                    Fl.sub_th.(S.sub) = setdiff(Fl.sub_th.(S.sub), th_name, 'stable');
                    th_name = [th_name, '_' S.sub];
                end
                
                try Fl.th    = rmfield(Fl.th, th_name);     catch, end %  err, warning(err_msg(err)); end
                try Fl.th0   = rmfield(Fl.th0, th_name);    catch, end %  err, warning(err_msg(err)); end
                try Fl.th_lb = rmfield(Fl.th_lb, th_name);  catch, end %  err, warning(err_msg(err)); end
                try Fl.th_ub = rmfield(Fl.th_ub, th_name);  catch, end %  err, warning(err_msg(err)); end
            end
        end
        
        function remove_fun(Fl, fun_name, throwError)
            % Remove function(s).
            %
            % remove_fun(Fl, fun_name, throwError = false)
            % remove_fun(Fl, '_all', ...) % Remove all
            %
            % fun_name : char or cell.
            
            if nargin < 3, throwError = false; end
            if ischar(fun_name) && strcmp(fun_name, '_all')
                fun_name = Fl.fun_names;
            end
            
            try 
                Fl.fun      = rmfield(Fl.fun,      fun_name);
            catch err
                if throwError, rethrow(err); end
            end
            try
                Fl.fun_out  = rmfield(Fl.fun_out,  fun_name);
            catch err
                if throwError, rethrow(err); end
            end
            try
                Fl.fun_iter = setdiff(Fl.fun_iter, fun_name, 'stable');
            catch err
                if throwError, rethrow(err); end
            end
            for f = fieldnames(Fl.fun_opt)'
                try
                    Fl.fun_opt.(f{1})  = rmfield(Fl.fun_opt.(f{1}),  fun_name);
                catch err
                    if throwError, rethrow(err); end
                end            
            end
        end
        
        function reorder_fun(Fl, funs, rel, fun)
            % reorder_fun(Fl, funs, rel, fun)
            %
            % See also fieldOrder
            
            Fl.fun = fieldOrder(Fl.fun, funs, rel, fun);
        end
        
        function merge(Fl, Fl2, compo, sub)
            % merge(Fl, Fl2, compo = {'W', 'th', 'fun', 'constr'}, sub = '');
            
            if nargin < 4 || isempty(sub)
                sub = '';
            else
                error('Not implemented yet from th!');
            end
            
            if nargin < 3 || isempty(compo)
                compo = {'W', 'th', 'fun', 'constr'};
            end
            
            if ismember('W', compo)
                if isempty(sub)
                    Fl.W = copyFields(Fl.W, Fl2.W);
                else
                    if ~isfield(Fl.W, sub), Fl.W.sub = struct; end
                    Fl.W.(sub) = copyFields(Fl.W.(sub), Fl2.W);
                end
            end
            if ismember('th', compo)
                Fl.th       = copyFields(Fl.th,  Fl2.th);
                Fl.th0      = copyFields(Fl.th0, Fl2.th0);
                Fl.th_lb    = copyFields(Fl.th_lb, Fl2.th_lb);
                Fl.th_ub    = copyFields(Fl.th_ub, Fl2.th_ub);
            end
            if ismember('fun', compo)
                Fl.fun      = copyFields(Fl.fun, Fl2.fun);
                Fl.fun_out  = copyFields(Fl.fun_out, Fl2.fun_out);
                Fl.fun_iter = union(Fl.fun_iter, Fl2.fun_iter, 'stable');
                for f = fieldnames(Fl.fun_opt)'
                    Fl.fun_opt.(f{1}) = copyField(Fl.fun_opt.(f{1}), Fl2.fun_opt.(f{1}));
                end
            end
            if ismember('constr', compo)
                Fl.constr = unionCell(Fl.constr, Fl2.constr);
            end
        end
                
        %% Evaluate
        function f = cost_fun(Fl, op)
            % Gives a function handle that can be fed to fmincon, etc.
            %
            % f = cost_fun(Fl, op='iter'|'init')
            
            % Initialize fun - workaround MATLAB's bug in parallel processing,
            % where it invokes saveobj without subsequently invoking loadobj.
            Fl = loadobj(Fl);
            
            if nargin < 2, op = 'iter'; end
            assert(any(strcmp(op, {'iter', 'init'})), 'op must be either iter or init!');
            
            prop_fun = ['fun_' op];
            
            if Fl.handle_mode
                % Save results to Fl.W and Fl.cost.
                % Useful in plotting but may be incompatible with parallel processing.
                f = @calc_cost_handle;
            else
                % For backward compatibility
                Fl.W.sub_th = Fl.sub_th;
                
                % Gives values only - W, fun
                W = Fl.W; % DEBUG
                
                f = @(c_th) Fit_Flow3.calc_cost( ...
                    Fit_Flow3.fill_vec(c_th, ~Fl.th_fix_vec, Fl.th0_vec), ...
                    Fl.th_names, W, Fl.fun, ...
                    Fl.fun_out, Fl.fun_opt, Fl.(prop_fun), ...
                    Fl.calc_cost_opt_C);
            end
            
            function [cost, W] = calc_cost_handle(c_th)
                % Save results to Fl.W and Fl.cost
                Fl.th_vec(~Fl.th_fix_vec) = c_th;
                
                [cost, W] = Fit_Flow3.calc_cost( ...
                    Fit_Flow3.fill_vec(c_th, ~Fl.th_fix_vec, Fl.th0_vec), ...
                    Fl.th_names, Fl.W, Fl.fun, ...
                    Fl.fun_out, Fl.fun_opt, Fl.(prop_fun), ...
                    Fl.calc_cost_opt_C);
                
                Fl.W = W;
                Fl.cost = cost;
            end
        end
        
        function run_init(Fl)
            % Runs init functions            
            % and gets and changes th0, th_lb, th_ub, constr afterwards.
           
            run(Fl, Fl.fun_init);
        end
        
        function run_init_pre(Fl)
            % Copy th0, th_lb, th_ub, constr into W so that they can be referenced.
            for fs = {'th0', 'th_lb', 'th_ub', 'constr'}
                Fl.W.(fs{1}) = Fl.(fs{1});
                
                if ~strcmp(fs{1}, 'constr')
                    for subs = fieldnames(Fl.sub_th)'
                        for ths = Fl.sub_th.(subs{1})(:)'
                            Fl.W.(subs{1}).(fs{1}).(ths{1}) = Fl.W.(fs{1}).([ths{1} '_' subs{1}]);
                        end
                    end
                end
            end
        end
        
        function run_init_post(Fl)
            % Change th
            for fs = {'th0', 'th_lb', 'th_ub'}
                Fl.W.(fs{1}) = Fl.(fs{1});
                
                for subs = fieldnames(Fl.sub_th)'
                    for ths = Fl.sub_th.(subs{1})(:)'
                        Fl.W.(fs{1}).([ths{1} '_' subs{1}]) = Fl.W.(subs{1}).(fs{1}).(ths{1});
                    end
                end
            end
            
            %  Allows for setting one guess based on another guess.
            Fl.th0 = copyFields(Fl.th0, Fl.W.th0);
            Fl.th_lb = copyFields(Fl.th_lb, Fl.W.th_lb);
            Fl.th_ub = copyFields(Fl.th_ub, Fl.W.th_ub);
            
            % Change constr
            if isfield(Fl.W, 'constr')
                Fl.add_constr(Fl.W.constr);
            end
        end
        
        function run_iter_pre(Fl)
            if Fl.dbstop_if.naninf
                dbstop if naninf
            end
            
            Fl.W.sub_th = Fl.sub_th;
        end
        
        function run_iter_post(Fl)
            if Fl.dbstop_if.naninf
                dbclear if naninf
            end
        end
        
        function [cost, W] = run_iter(Fl)
            % [cost, W] = run_iter(Fl)
            %
            % Runs Fl.fun_iter

            [cost, W] = Fl.run(Fl.fun_iter);
        end
        
        function [cost, W] = run(Fl, names)
            if nargin < 2
                Fl.run_init;
                [cost, W] = Fl.run_iter;
            else
                % Initialize fun - workaround MATLAB's bug in parallel processing,
                % where it invokes saveobj without subsequently invoking loadobj.
                loadobj(Fl);
                
                if nargin < 2
                    names = Fl.fun_names; 
                elseif ischar(names)
                    names = {names};
                end
                
                f_init = ismember(names, Fl.fun_init);
                if any(f_init)
                    Fl.run_init_pre;
                    
                    [~, Fl.W] = Fit_Flow3.calc_cost(Fl.th_vec, Fl.th_names, Fl.W, ...
                        Fl.fun, Fl.fun_out, Fl.fun_opt, names(f_init), ...
                        varargin2C({'calc_cost', false}, Fl.calc_cost_opt_C));
    
                    Fl.run_init_post;
                end
                
                f_iter = ismember(names, Fl.fun_iter);
                if any(f_iter)
                    Fl.run_iter_pre;
                    
                    [Fl.cost, Fl.W] = Fit_Flow3.calc_cost(Fl.th_vec, Fl.th_names, Fl.W, ...
                        Fl.fun, Fl.fun_out, Fl.fun_opt, names(f_iter), Fl.calc_cost_opt_C);
    
                    Fl.run_iter_post;
                end
                
                if nargout >= 1, cost = Fl.cost; end
                if nargout >= 2, W    = Fl.W;    end                
            end           
        end

        function res2W(Fl, res)
            % Given res or from Fl.res, set th and run to construct W.
            %
            % res2W(Fl, res)
            %
            % res : if a struct, replaces Fl.res
            %       if a number, replaces Fl.th with Fl.res.res_all{res}.th
            %       (after grid fitting)
            
            if nargin >= 2 && ~isempty(res)
                if isstruct(res)
                    Fl.res = res;
                elseif isnumeric(res)
                    Fl.res = Fl.res.res_all{res}.th;
                end
            end
            Fl.W = Fl.W0; % may cause a bug in older implementation % DEBUG
            try Fl.W = copyFields(Fl.W, Fl.res.nested, Fl.th_nested); catch, end
            Fl.th = Fl.res.th;
            Fl.run;
        end
        
        function [stop, h] = runPlotFcns(Fl, varargin)
            S = varargin2S(varargin, {
                'cla', false
                'optimValues', {}
                'f', {}
                'state', 'done'
                'catchError', true
                });
            
            if isempty(S.f)
                S.f = Fl.plotfun;
            end
            n = length(S.f);
            
            th_vec = Fl.th_vec(~Fl.th_fix_vec);
            
            nf = length(S.f);
            nR = ceil(sqrt(nf));
            nC = ceil(nf / nR);
            
            S.optimValues = varargin2S(S.optimValues, {
                'funcCount', Fl.n_iter * length(th_vec)
                'fval',      Fl.cost
                'iteration', Fl.n_iter
                'procedure', []
                });
           
            h = ghandles(1,n);
            
            stop = false;
            for ii = 1:n
                h(ii) = subplot(nR, nC, ii);
                if S.cla, cla; end
                
                try
                    stop = stop || S.f{ii}(th_vec, S.optimValues, S.state);
                catch err
                    if S.catchError
                        warning(err_msg(err));
                    else
                        rethrow(err);
                    end
                end
            end
        end
        
        function stop = runOutputFcns(Fl)
            f = Fl.outfun;
            
            th_vec = Fl.th_vec(~Fl.th_fix_vec);
            
            optimValues = varargin2S({}, {
                'funcCount', Fl.n_iter * length(th_vec)
                'fval',      Fl.cost
                'iteration', Fl.n_iter
                'procedure', []
                });
            
            stop = f(th_vec, optimValues, 'iter');            
        end
        
        %% Optimization interface
        function [res, W] = fit(Fl, optim_fun, args, opts, outs)
            % [res, W] = fit(Fl, optim_fun, args, opts, outs)
            
            %% optim_fun
            if nargin < 2 || isempty(optim_fun), optim_fun = @fmincon; end
            if isa(optim_fun, 'char')
                optim_nam = optim_fun;
                optim_fun = evalin('caller', ['@' optim_nam]);
            elseif isa(optim_fun, 'function_handle')
                optim_nam = char(optim_fun);                
            else
                error('optim_fun must be either a function name or a function handle!');
            end
            
            %% Initialize Fl
            % Initialize fun - workaround MATLAB's bug in parallel processing,
            % where it invokes saveobj without subsequently invoking loadobj.
            Fl = loadobj(Fl);
            if ~isempty(Fl.W0)
                Fl.W = Fl.W0; % May cause a bug in older implementations. % DEBUG
            end
            Fl.th = Fl.th0;
            Fl.run_init;
            Fl.run_iter; % Run once to avoid errors in plotting, etc., due to absent variables in W.
            
            %% Prepare arguments for optim_fun
            if nargin < 3 || isempty(args), args = {}; end
            if nargin < 4 || isempty(opts), opts = {}; end
            
            % Arguments - get from Fl.fit_arg.(optim_fun)
            try
                args = varargin2S(args, Fl.fit_arg.(optim_nam)(Fl));
            catch
                args = varargin2S(args, Fl.fit_arg.etc_(Fl));
            end
            
            % Options
            try
                opts = varargin2S(opts, Fl.fit_opt.(optim_nam)(Fl));
            catch
                opts = varargin2S(opts, Fl.fit_opt.etc_(Fl));
            end
            
            % Constraints
            if ~isempty(Fl.constr)
                C_constr = fmincon_cond(Fl);
                args = varargin2S({
                    'A',        C_constr{1}
                    'b',        C_constr{2}
                    'Aeq',      C_constr{3}
                    'beq',      C_constr{4}
                    'nonlcon',  C_constr{5}
                    }, args);
            end
            
            % Include in arguments only if nonempty
            if isfield(args, 'options')
                if isempty(args.options), args.options = {}; end
                args.options = varargin2S(opts, args.options);
            elseif ~isempty(opts)
                args.options = opts;
            end
            
            %% Prepare output
            if nargin < 5 || isempty(outs)
                try
                    outs = Fl.fit_out.(optim_nam);
                catch
                    outs = Fl.fit_out.etc_;
                end
            end
            
            n_outs = length(outs);
            C_args = struct2cell(args);
            
            % history
            if Fl.save_history
                Fl.f_record_history = Fl.record_history;
                Fl.f_record_history([],[],'init'); % Initialize
            end
            
            %% Run optimization
            st = tic;
            fprintf('Fitting Fl.id=%s began at %s\n', Fl.id, datestr(now, 'yyyymmddTHHMMSS'));
            
            Fl.n_iter = 0;
            [c_outs{1:n_outs}] = optim_fun(C_args{:});
            
            el = toc(st);
            fprintf('Fitting Fl.id=%s finished at %s\n', Fl.id, datestr(now, 'yyyymmddTHHMMSS'));
            fprintf('Fitting Fl.id=%s took %1.3f seconds.\n', Fl.id, el);
            
            %% Store in res
            res.optim_fun_name = optim_nam;
            res.out  = cell2struct(c_outs(:), outs(:), 1);
            res.arg  = args;
            res.th0  = Fl.th0;
            res.arg.th0 = Fl.th0;
            res.opt  = opts;
            res.tSt  = st;
            res.tEl  = el;
            res.tEn  = st + el;
            res.nested = copyFields(struct, Fl.W, Fl.th_nested);
            
            % Truncate history
            if Fl.save_history
                res.history = Fl.f_record_history([],[],'retrieve');
                
                Fl.f_record_history([],[],'delete');
            end
            
            % Postprocess
            res = Fl.fit_postprocess(res);
            
            % Output
%             Fl.f_calc_cost_iter = Fl.cost_fun('iter');
%             [Fl.cost, Fl.W] = feval(Fl.f_calc_cost_iter, res.out.x);
            Fl.res = res;
            Fl.res2W;
            
            if nargout >= 2, W = Fl.W; end
        end
        
        function res = fit_postprocess(Fl, res)
            % res = fit_postprocess(Fl, [res])
            
            if nargin < 2 || isempty(res)
                res = Fl.res;
            end
            
            % Postprocesses
            try
                res.fval = res.out.fval;
            catch
                res.fval = nan;
            end
            try
                res.th = Fl.vec2S(hVec(Fit_Flow3.fill_vec(res.out.x, ~Fl.th_fix_vec, Fl.th0_vec)));
            catch
                res.th = Fl.vec2S(nan(1, Fl.n_th));
            end
            try
                res.out.se = hVec(diag(inv(res.out.hessian)));
                assert(length(res.out.se) == nnz(~Fl.th_fix_vec));
            catch
                res.out.se = nan(1, nnz(~Fl.th_fix_vec));
            end           
            res.se = Fl.vec2S(Fit_Flow3.fill_vec(res.out.se, ~Fl.th_fix_vec, zeros(1, Fl.n_th)));
            
            % Prepare to calculate information criteria
            k = Fl.n_th;
            n = Fl.n_dat;
            NLL = res.fval;
            
            % Count the number of fixed parameters and subtract from k
            n_fixed = nnz(Fl.th_fix_vec);
            k = k - n_fixed;
            res.k = k;
            res.n = n;
            res.n_fixed = n_fixed;
            
            % Information criteria
            res.bic = 2 * NLL + k * log(n);
            res.aic = 2 * k + 2 * NLL;
            
            % Correction for finite model size (Burnham & Anderson, 2002; Cavanaugh 1997) 
            res.aic_c = res.aic + 2 * k * (k+1) / (n - k - 1); 
            
            % Convert functions to strings
            res = Fit_Flow3.res_func2str(res);
            
            % Put res back into Fl
            Fl.res = res;
        end
        
        %% Grid
        function [res, res_all] = fit_grid(Fl, spec, fit_opt, grid_opt)
            % fit_grid : all-in-one function.
            % TODO: make this use grid_setup/run/gather.
            %
            % spec: empty: 
            %       {'var1', val1, ...}               : Use all combinations
            %       {{'var1', var2, ...}, {[x0_1_1, lb1_1, ub1_1], ...; [x0_2_1, lb2_1, ub2_1], ...}, ...}    : Use given combinations
            %
            % val : vector: evaluate within [val(1), val(2)], with an initial value of (val(1)+val(2))/2, then [val(2), val(3)], ...
            %       scalar: equivalent to giving linspace(lb, ub, val)
            %       cell  : evaluate within [val{1}(2), val{1}(3)], with an initial value of val{1}(1), then [val{2}(1), val{2}(2)], ...
            
            % Parse spec - get spec_nam, nspec, comb, ncomb
            if isempty(spec)
                spec = arg2C([Fl.th_names(:), repmat({1}, [length(Fl.th_vec), 1])]);
            end
            
            if iscell(spec{1})
                spec_nam   = spec{1};
                comb       = spec{2};
                ncomb      = size(comb, 1);
            else
                spec_nam   = spec(1:2:end);
                spec_range = spec(2:2:end);
                nspec      = length(spec_nam);
            
                for ispec = 1:nspec
                    spec_range{ispec} = parse_spec(spec_nam{ispec}, spec_range{ispec});
                end
                
                [comb, ncomb] = factorize(spec_range);
            end
            
            function cspec = parse_spec(nam, cspec)
                % Coerce into a cell form
                if isnumeric(cspec)
                    if isscalar(cspec)
                        cspec = parse_spec(nam, ...
                            linspace(Fl.th_lb.(nam), Fl.th_ub.(nam), cspec + 1));
                    else
                        vec   = cspec;
                        nvec  = length(vec) - 1;
                        cspec = cell(1, nvec);
                        for kk = 1:nvec
                            cspec{kk} = [(vec(kk) + vec(kk+1))/2, vec(kk), vec(kk+1)];
                        end
                    end
                end
            end
            
            % Parse opt
            grid_opt = varargin2S(grid_opt, {
                'restrict', true % Set to false to check global convergence.
                'parallel', false
                });
            
            if ncomb == 1, grid_opt.parallel = false; end
            
            % Fit
            res_all  = cell(1,ncomb);

            if grid_opt.parallel
                parfor ii = 1:ncomb
                    res_all{ii} = fit_grid_cell(Fl, comb(ii,:), spec_nam, fit_opt, grid_opt);
                end
            else
                for ii = 1:ncomb
                    res_all{ii} = fit_grid_cell(Fl, comb(ii,:), spec_nam, fit_opt, grid_opt);
                end
            end
            
            % Find minimum
            fval_min = inf;
            for ii = 1:ncomb
                if res_all{ii}.out.fval < fval_min
                    res = res_all{ii};
                    fval_min = res_all{ii}.out.fval;
                end
            end
            
            % Output
            res.grid = packStruct(res_all, grid_opt, spec_nam, comb, ncomb);
            
            Fl.res = res;
            Fl.res2W;
        end
        
        function res = fit_grid_cell(Fl, comb, spec_nam, fit_opt, grid_opt)    
            % Called from fit_grid.
            %
            % comb: a cell vector of combinations.
            %       each cell contains [th0, th_lb, th_ub].
            %       th_lb and th_ub are used only when grid_opt.restrict = true.
            % spec_nam: comb{k} specifies th0.(spec_nam{k}), th_lb.(spec_nam{k}), and th_ub.(spec_nam{k}).
            % fit_opt: cell vector fed to Fit_Flow3.fit().
            % grid_opt: if grid_opt.restrict = true, restricts ranges.
            
            grid_opt = varargin2S(grid_opt);
            cFl = copy(Fl);
            
            nspec = length(spec_nam);
            
            for jj = 1:nspec
                cFl.th0.(spec_nam{jj})   = comb{jj}(1);

                if grid_opt.restrict
                    cFl.th_lb.(spec_nam{jj}) = comb{jj}(2);
                    cFl.th_ub.(spec_nam{jj}) = comb{jj}(3);
                end
            end

            fit_opt = varargin2C(fit_opt);
            res = fit(cFl, fit_opt{:});
        end
        
        %% Grid - split (gives res.grid different from fit_grid!)
        function [res, res_all, ds_grid] = grid_fit(Fl, spec, fit_opt, grid_opt)
            % [res, res_all, ds_grid] = grid_fit(Fl, spec, fit_opt, grid_opt)
            %
            % See also: grid_setup, grid_eval, grid_unit, grid_gather
            
            if nargin < 3, fit_opt = {}; end
            if nargin < 4, grid_opt = {}; end
            [grid_spec, grid_opt] = grid_setup(Fl, spec, fit_opt, grid_opt);
            ds_grid = Fit_Flow3.grid_view(grid_spec) %#ok<NOPRT>
            res_all = grid_eval(Fl, grid_spec, grid_opt);
            [res, res_all] = grid_gather(Fl, res_all, grid_spec, grid_opt);
            
            Fl.res = res;
            Fl.res2W;
        end
        
        function [grid_spec, grid_opt] = grid_setup(Fl, spec, fit_opt, grid_opt)
            % grid_setup : wrapper for grid_setup_static
            %
            % grid_setup : grid gives specifications to run fits
            %
            % [grid_spec, grid_opt] = grid_setup(Fl, spec, fit_opt, grid_opt)
            %
            % spec: empty: 
            %       {'var1', val1, ...}               : Use all combinations
            %       {{'var1', 'var2', ...}, {[x0_1_1, lb1_1, ub1_1], ...; [x0_2_1, lb2_1, ub2_1], ...}, ...}    : Use given combinations
            % 
%             % Considering struct spec... % TODO
%             % spec: empty % use default th0, th_lb, th_ub.
%             %       struct: spec.(var1) = val1, ... % Use all combinations among variables.
%             %       cell array of struct: % Use given combinations
            %
            % val : vector: evaluate within [val(1), val(2)], with an initial value of (val(1)+val(2))/2, then [val(2), val(3)], ...
            %       scalar: equivalent to giving linspace(lb, ub, val)
            %       cell  : evaluate within [val{1}(2), val{1}(3)], with an initial value of val{1}(1), then [val{2}(1), val{2}(2)], ...
            %
            % grid_spec{k}: a struct with fields of th0, th_lb, th_ub, fit_opt.
            % grid_opt    : a struct.
            % .restrict   : restrict th_lb and th_ub around th0.
            % .parallel   : use parfor
            
            if nargin < 2 || isempty(spec)
                % If no spec is given, use existing th0, th_lb, th_ub.

%                 % Considering struct spec... % TODO
%                 ths = Fl.th_names(:)';
%                 
%                 for ii = 1:length(ths)
%                     spec{1}{ii,1} = 
%                 end

                spec = arg2C([Fl.th_names(:), repmat({1}, [length(Fl.th_vec), 1])]);
            end
            if nargin < 3, fit_opt = {}; end
            if nargin < 4, grid_opt = {}; end
            grid_opt = varargin2S(grid_opt, Fl.grid_opt);
            
            [grid_spec, grid_opt] = Fit_Flow3.grid_setup_static(spec, fit_opt, grid_opt);
            
            if ~isempty(Fl.grid_spec)
                disp('Adding to existing grid_spec. Set Fl.grid_spec = {}; before grid_setup to avoid accumulation.');
            end
            Fl.grid_spec = [Fl.grid_spec; grid_spec];
        end
        
        function res_all = grid_eval(Fl, grid_spec, grid_opt)
            % res_all = grid_eval(Fl, grid_spec, grid_opt)
            
            if nargin < 2 || isempty(grid_spec)
                grid_spec = Fl.grid_spec;
            end
            if nargin < 3 || isempty(grid_opt)
                grid_opt  = Fl.grid_opt;
            end
            ncomb = length(grid_spec);
            
            res_all = cell(1,ncomb);
            
            st = tic;
            fprintf('Fitting grid in Fl.id=%s began at %s\n', Fl.id, datestr(now, 'yyyymmddTHHMMSS'));
            Fl.res.tSt = st;
            
            switch grid_opt.parallel
                case 0 % regular for
                    for ii = 1:ncomb
                        res_all{ii} = grid_unit(Fl, grid_spec{ii}, grid_opt);
                    end
                case 1 % parfor
                    parfor ii = 1:ncomb
                        res_all{ii} = grid_unit(Fl, grid_spec{ii}, grid_opt);
                    end
                case 2 % parfeval
                    for ii = 1:ncomb
                        res_all{ii} = parfeval(@grid_unit, 1, Fl, grid_spec{ii}, grid_opt);
                    end
            end
            
            el = toc(st);
            en = st + el;
            fprintf('Fitting grid in Fl.id=%s finished at %s\n', Fl.id, datestr(now, 'yyyymmddTHHMMSS'));
            fprintf('Fitting grid in Fl.id=%s took %1.3f seconds.\n', Fl.id, el);
            Fl.res.tSt = st; % In case this is overwritten, write again.
            Fl.res.tEl = el;
            Fl.res.tEn = en;
        end
        
        function [res, cFl] = grid_unit(Fl, grid_spec, grid_opt)
            % [res, cFl] = grid_unit(Fl, grid_spec, grid_opt)
            %
            % See also: grid_setup, grid_submit, grid_gather
            
            if isnumeric(grid_spec)
                grid_spec = Fl.grid_spec{grid_spec};
            end
            if nargin < 3 || isempty(grid_opt)
                grid_opt = Fl.grid_opt;
            end
            
            cFl = loadobj(copy(Fl));
            
            if isfield(grid_spec, 'th0')
                cFl.th0   = varargin2S(grid_spec.th0, cFl.th0, true); % copyFields(cFl.th0,   grid_spec.th0);
            end
            
            if grid_opt.restrict
                if isfield(grid_spec, 'th_lb')
                    cFl.th_lb = varargin2S(grid_spec.th_lb, cFl.th_lb, true); % copyFields(cFl.th_lb, grid_spec.th_lb);
                end
                if isfield(grid_spec, 'th_ub')
                    cFl.th_ub = varargin2S(grid_spec.th_ub, cFl.th_ub, true); % copyFields(cFl.th_ub, grid_spec.th_ub);
                end
            end
            
            res = fit(cFl, grid_spec.fit_opt{:});
        end
        
        function [res, res_all] = grid_gather(Fl, res_all, grid_spec, grid_opt)
            % Wrapper for grid_gather_static
            %
            % [res, res_all] = grid_gather(Fl, [res_all, grid_spec, grid_opt])
            
            if nargin < 2 || isempty(res_all),   res_all = Fl.res_all; end
            if nargin < 3 || isempty(grid_spec), grid_spec = Fl.grid_spec; end
            if nargin < 4 || isempty(grid_opt),  grid_opt  = Fl.grid_opt; end

            [res, res_all] = Fit_Flow3.grid_gather_static(res_all, grid_spec, grid_opt);
            
            % TODO: update time stamps (res.tSt, tEl, tEn)
            
            Fl.res_all = res_all;
            Fl.res = res;
            Fl.res2W;
        end
        
        %% Global optimization
        function res = fit_global(Fl, g_fun, g_opt, run_arg, loc_opt, loc_fun, loc_arg)
            % res = fit_global(Fl, g_fun, g_opt, run_arg, loc_opt, loc_fun=@fmincon, loc_arg)
            
            if nargin < 2, g_fun = 'MultiStart'; end
            if nargin < 3, g_opt = {}; end
            if nargin < 4, run_arg = {}; end
            if nargin < 5, loc_opt = {}; end
            if nargin < 6, loc_fun = @fmincon; end
            if nargin < 7, loc_arg = {}; end
            
            C_constr = fmincon_cond(Fl);
            loc_opt = varargin2C(loc_opt, {
                'Display',      'iter'
                'Algorithm',    'interior-point'
                'FinDiffType',  'central'
                'UseParallel',  'always'
                }, loc_opt);
            loc_optim = optimoptions(loc_fun, loc_opt{:});
            
            loc_arg = varargin2C(loc_arg, {
                'x0',           Fl.th0_vec(~Fl.th_fix_vec)
                'lb',           Fl.th_lb_vec(~Fl.th_fix_vec)
                'ub',           Fl.th_ub_vec(~Fl.th_fix_vec)
                'objective',    Fl.cost_fun
                'Aineq',        C_constr{1}
                'bineq',        C_constr{2}
                'Aeq',          C_constr{3}
                'beq',          C_constr{4}
                'nonlcon',      C_constr{5}
                'options',      loc_optim
                });
            problem = createOptimProblem(char(loc_fun), loc_arg{:});
                
            g_opt = varargin2C(g_opt, {
                'UseParallel',  true
                });
            
            switch g_fun
                case 'MultiStart'
                    if isempty(run_arg)
                        run_arg = {200}; % Number of runs
                    end
                    
                    g_obj = MultiStart(g_opt{:});
                    
                    % Fit and store results
                    res.optim_fun_name = g_fun;
                    res.out = out2S( ...
                        @() run(g_obj, problem, run_arg{:}), ...
                        Fl.fun_out.MultiStart);
                    res.arg = {g_fun, g_opt, run_arg, loc_opt, loc_fun, loc_arg};
                    
                    res = copyFields(res, packStruct(...
                        loc_optim, problem, g_opt, g_obj));
                    
                otherwise 
                    error('Not implemented yet!');
            end
        end
        
        function C = fmincon_cond(Fl)
            constr = Fl.constr;
            W = Fl.W;
            
            for ii = 1:length(constr)
                if ischar(constr{ii}), constr{ii} = str2func(constr{ii}); end
                if isa(constr{ii}, 'function_handle')
                    constr{ii} = constr{ii}(W);
                end
            end
            
            C = fmincon_cond(Fl.th_names(~Fl.th_fix_vec), constr);
        end
        
        %% Output/plotting functions
        function f = outfun(Fl)
            % (1) Evaluate functions in Fl.OutputFcns 
            % (2) Also evaluate Fl.record_history
            
            cOutputFcns = Fl.OutputFcns;
            if Fl.save_history
                cOutputFcns = [{@Fl.f_record_history}, cOutputFcns(:)'];
            end
            
            f = @c_outfun;
            
            function stop = c_outfun(x, optimValues, state)
                stop = false;
                
                for ii = 1:length(cOutputFcns)
                    stop = stop || cOutputFcns{ii}(x, optimValues, state);
                end
            end
        end
        
        function f = plotfun(Fl, hMode)
            % f = plotfun(Fl, hMode)
            if nargin < 2 || isempty(hMode)
                hMode = Fl.handle_mode;
            end
            
            if hMode
                f = cell(size(Fl.PlotFcns));

                for ii = 1:length(f)
                    f{ii} = Fl.PlotFcns{ii}(Fl);
                end
            else % e.g., when UseParallel = 'always'
                n = numel(Fl.PlotFcns);
                if n > 0
                    f = [repmat({@(x,v,s) void(@() plot(nan), 0)}, [1, n-1]), ...
                         {@(x,v,s) plotfun_par(Fl,x,v,s)}];
                else
                    f = {};
                end
            end
            
            function stop = plotfun_par(Fl, x, v, s)
                % When Fl itself is not updated on every iteration,
                % as when UseParallel = 'always',
                % force update using x, 
                % then bypass fmincon's PlotFcns - just use runPlotFcns.
                
                stop = false;
                
                if Fl.plot_opt.per_iter ~= 0
                    if mod(v.iteration, Fl.plot_opt.per_iter) == 0
                        Fl.th_vec(~Fl.th_fix_vec) = x;
                        Fl.run_iter; % Set W in appropriate state
        %                 assert(Fl.cost == v.fval, 'Discrepancy in cost!'); % DEBUG

        %                 fig_tag('Optimization PlotFcns');
        %                 disp('plotfun_par'); % Seems not to work..
        %                 keyboard; % DEBUG
                        Fl.runPlotFcns('f', Fl.plotfun(true), 'optimValues', v, 'state', s);
                        drawnow;
                    end
                end
            end
        end
        
        function f = dispfun(Fl)
            f = @c_outfun;
            th_names = Fl.th_names(~Fl.th_fix_vec);
            
            function stop = c_outfun(x, optimValues, state)
                fprintf('Iter %4d (fval=%1.5g)', optimValues.iteration, optimValues.fval);
                disp(x); % Workaround MATLAB's bug
%                 for ii = 1:length(th_names)
%                     fprintf(' %s=%1.5g', th_names{ii}, x(ii));
%                 end
%                 cfprintf(' %s=%1.5g', th_names, x);
                stop = false;
            end
        end
        
        function f = record_history(Fl)
            % Gives Fl.f_record_history. Used in outfun.
            
            th_names = Fl.th_names(~Fl.th_fix_vec);
            max_iter = Fl.max_iter;
            n_th     = nnz(~Fl.th_fix_vec);
            
            % id: Prevent confusion between multiple Fl 
            % without using Fl (handle) explicitly during fitting
            % which adds overhead during parallel processing.
            f = @(x,v,s) f_rec(x,v,s,Fl.id);
            
            function stop = f_rec(x, optimValues, state, id)
                persistent history
                
                % Flag
                stop = false;

                switch state
                    case 'init'
                        % Initialize
                        history.(id) = mat2dataset(zeros(max_iter,n_th+1), 'VarNames', [th_names(:)', {'fval'}]);

                    case 'iter'
                        % Record
                        citer = optimValues.iteration + 1;
                        
                        for ii = 1:n_th
                            history.(id).(th_names{ii})(citer, 1) = x(ii);
                        end
                        history.(id).fval(citer,1) = optimValues.fval;
                        
                    case 'done'
                        % Truncate
                        history.(id) = history.(id)(1:min(optimValues.iteration, end), :);

                    case 'retrieve'
                        % Return
                        stop = history.(id);
                        
                    case 'delete'
                        history = rmfield(history, id);
                        
                    case 'deleteAll'
                        history = struct;
                end
            end
        end
        
        function f = optimplotfval(~)
            f = @f_optimplotfval;
            
            function stop = f_optimplotfval(~,optimValues,state,varargin)
                persistent v
                
                switch state
                    case 'init'
                        % Initialize
                        citer = 0;
                        v = [];

                    case 'iter'
                        % Record
                        citer = optimValues.iteration + 1;
                        v(citer) = optimValues.fval;
                        
                    case 'done'
                        citer = length(v);
                        % Do nothing
                end                
                plot(1:length(v), v, 'kd', 'MarkerFaceColor', 'm');
                if citer > 0
                    title(sprintf('Current Function Value: %1.1f', v(citer)));
                end
                xlabel('Iteration');
                ylabel('Function value');
                stop = false;
            end
        end
        
        function f = optimplotx(Fl)
            names = Fl.th_names(~Fl.th_fix_vec);
            n     = length(names);
            ub    = Fl.th_ub_vec(~Fl.th_fix_vec);
            lb    = Fl.th_lb_vec(~Fl.th_fix_vec);
            
            f = @f_optimplotx;
            
            function stop = f_optimplotx(x,optimValues,state,varargin)
                
%                 persistent ht
                
                % Show normalized plot
                x_plot = (x - lb) ./ (ub - lb);
                
                barh(x_plot, 'FaceColor', [0 1 1], 'EdgeColor', 'none');
                
%                 stop = optimplotx(x_plot,optimValues,state,varargin{:});
%                 xlabel(''); % Remove the xlabel that obscures variable names in <R2014b.
                
                hLabels = findobj(gca, 'Type', 'text');
                if isempty(hLabels) || numel(hLabels) ~= 3 * n
                    for ii = 1:n
                        text(0.5, ii, sprintf('%1.2g', lb(ii)), ...
                            'HorizontalAlignment', 'center');
                        text(1, ii, sprintf('%1.2g', ub(ii)), ...
                            'HorizontalAlignment', 'right');
                        text(0, ii, sprintf('%1.3g', x(ii)), ...
                            'HorizontalAlignment', 'left');
                    end
                else
                    for ii = 1:n
                        set(hLabels(ii*3-2), 'Position', [0.5, ii, 0], ...
                            'String', sprintf('%1.2g', lb(ii)), ...
                            'HorizontalAlignment', 'center');
                        set(hLabels(ii*3-1), 'Position', [1, ii, 0], ...
                            'String', sprintf('%1.2g', ub(ii)), ...
                            'HorizontalAlignment', 'right');
                        set(hLabels(ii*3-0), 'Position', [0, ii, 0], ...
                            'String', sprintf('%1.3g', x(ii)), ...
                            'HorizontalAlignment', 'left');
                    end
                end

%                 labels = cell(4,n);
%                 for ii = 1:n
%                     labels{ii} = [
%                         strrep(names{ii}, '_', '-'), ': ', ... % '\newline', ...
%                         sprintf('%1.3g', x(ii)), ' ', ... % '\newline', ...
%                         sprintf('(%1.2g - %1.2g)', lb(ii), ub(ii))];
%                 end  
                
                set(gca, 'YTick', 1:n, 'YTickLabel', strrep(names, '_', '-'), 'YDir', 'reverse');
                xlim([0 1]);
                ylim([0 n+1]);
                
%                 if verLessThan('matlab', '8.4')
%                     try delete(ht); catch, end
%                     ht = format_ticks(gca, labels);
%                 else
%                     set(gca, 'XTickLabel', labels);
%                 end
%                 ylim([0 1]);

                stop = false;
            end
        end
        
        function v = optimValues(Fl, varargin)
            % v = optimValues(Fl, varargin)
            v = varargin2S(varargin, {
                'iteration', Fl.n_iter
                'fval', Fl.cost
                'searchdirection', []
                });
        end
        
        %% Get/Set
        function v = get.th_vec(Fl)
            v = S2vec(Fl, Fl.th);
        end
        
        function v = get.th0_vec(Fl)
            v = S2vec(Fl, Fl.th0);
        end
        
        function v = get.th_ub_vec(Fl)
            v = S2vec(Fl, Fl.th_ub);
        end
        
        function v = get.th_lb_vec(Fl)
            v = S2vec(Fl, Fl.th_lb);
        end
        
        function v = get.th_fix_vec(Fl)
            v = Fl.th_lb_vec == Fl.th_ub_vec;
        end
        
        function S = get.th_fix(Fl)
            S = vec2S(Fl, Fl.th_fix_vec);
        end
        
        function set.th_fix_vec(Fl, v)
            Fl.th_lb_vec(v) = Fl.th0_vec(v);
            Fl.th_ub_vec(v) = Fl.th0_vec(v);
        end
        
        function set.th_fix(Fl, v)
            Fl.th_fix_vec = S2vec(Fl, v);
        end
        
        function set.th_vec(Fl, v)
            Fl.th = vec2S(Fl, v);
        end
        
        function set.th0_vec(Fl, v)
            Fl.th0 = vec2S(Fl, v);
        end
        
        function set.th_lb_vec(Fl, v)
            Fl.th_lb = vec2S(Fl, v);
        end
        
        function set.th_ub_vec(Fl, v)
            Fl.th_ub = vec2S(Fl, v);
        end
        
        function S = vec2S(Fl, v)
            S = cell2struct(num2cell(v(:)), fieldnames(Fl.th), 1);
        end
        
        function v = S2vec(~, S)
            v = cell2mat(struct2cell(S))';
        end
        
        function v = get.th_names(Fl)
            v = fieldnames(Fl.th)';
        end
        
        function v = get.funs(Fl)
            v = struct2cell(Fl.fun)';
        end
        
        function v = get.fun_names(Fl)
            v = fieldnames(Fl.fun)';
        end
        
        function v = get.fun_init(Fl)
            v = setdiff(Fl.fun_names, Fl.fun_iter, 'stable');
        end
        
        function v = get.fun_iter(Fl)
            v = intersect(Fl.fun_names, Fl.fun_iter, 'stable');
        end
        
        function f = get.f_calc_cost_iter(Fl)
            if isempty(Fl.f_calc_cost_iter)
                f = Fl.cost_fun('iter');
            else
                f = Fl.f_calc_cost_iter;
            end
        end
        
        function f = get.f_calc_cost_init(Fl)
            if isempty(Fl.f_calc_cost_init)
                f = Fl.cost_fun('init');
            else
                f = Fl.f_calc_cost_init;
            end
        end
        
        function f = get.f_record_history(Fl)
%             if isempty(Fl.f_record_history)
%                 f = Fl.record_history;
%             else
                f = Fl.f_record_history;
%             end
        end
        
        function v = get.n_th(Fl)
            v = length(Fl.th_names);
        end
        
        function v = get.n_dat(Fl)
            if ~isempty(Fl.dat)
                v = length(Fl.dat);
            elseif isnumeric(Fl.dat_filt) || islogical(Fl.dat_filt)
                v = nnz(Fl.dat_filt);
            else
                v = nan;
            end
        end
        
        function set.sub_th(Fl, v)
            Fl.sub_th = v;
            Fl.W.sub_th = v; %#ok<MCSUP>
        end
        
%         function v = get.cost(Fl)
%             % Forward compatibility
%             if Fl.VERSION >= 6
%                 v = Fl.get_cost; % Too time-consuming
%                 Fl.cost = v;
%             else
%                 v = Fl.cost;
%             end
%         end
        
        function v = get.verFitFlow(Fl)
            v = Fl.VERSION;
        end
        
        %% Subflows
        function v = Wsub(Fl, f, sub)
            % val = Wsub(Fl, fieldName, sub)
            
            if nargin < 3 || isempty(sub)
                v = Fl.W.(f);
            else
                v = Fl.W.(sub).(f);
            end
        end
        
        function subs = subflows(Fl)
            % subs = subflows(Fl)
            subs = fieldnames(Fl.sub_th)';
        end
        
        function W = sub2main0(Fl, subs, vars, prop)
            % sub2main0(Fl, subs, vars, prop)
            if nargin < 2 || isempty(subs)
                subs = fieldnames(Fl.sub_th)';
            elseif ischar(subs)
                subs = {subs};
            end
            if nargin < 3 || isempty(vars)
                vars = Fl.sub_th.(subs{1});
            end
            if nargin < 4 || isempty(prop)
                prop = 'W';
            end
            
            for sub = subs(:)'
                Fl.(prop) = Fit_Flow3.main2sub(Fl.(prop), sub{1}, vars);
            end
            
            if nargout > 0, W = Fl.(prop); end
        end
        
        function W = main2sub0(Fl, subs, vars, prop)
            % main2sub(Fl, sub, vars, prop)
            if nargin < 2 || isempty(subs)
                subs = fieldnames(Fl.sub_th)';
            elseif ischar(subs)
                subs = {subs};
            end
            if nargin < 3 || isempty(vars)
                vars = Fl.sub_th.(subs{1});
            end
            if nargin < 4 || isempty(prop)
                prop = 'W';
            end
            
            for sub = subs(:)'
                Fl.(prop) = Fit_Flow3.sub2main(Fl.(prop), sub{1}, vars);
            end
            
            if nargout > 0, W = Fl.W; end
        end
        
        %% Internal
        function C = calc_cost_opt_C(Fl)
            % Make this agree with opt in calc_cost.
            
            C = varargin2C(Fl.debug);
        end
        
        %% Save
        function v = saveobj(Fl) % prepare_save(Fl) % 
%             v = Fl;
            v = copy(Fl);
            
            if ~isempty(v.dat_path)
                % Do not modify data file from Fl.
                
%                 if exist(v.dat_path, 'file')
%                     warning('%s already exists - skip saving. To update, delete existing file manually.', ...
%                         v.dat_path);
%                 else
%                     if ~exist(fileparts(v.dat_path), 'dir')
%                         mkdir(fileparts(v.dat_path));
%                     end
%                     
%                     dat = v.dat; %#ok<NASGU>
%                     save(v.dat_path, 'dat');
%                 end
                
                % Erase dat
                v.dat = dataset;
            end
            
            v.W = struct;
            
            v.PlotFcns = func2str_C(v.PlotFcns);
            v.OutputFcns = func2str_C(v.OutputFcns);
            
            v.fun = func2str_S(v.fun);
        end
        
        function dat = loadDatFl(Fl)
            if ~isempty(Fl.dat_path) && isempty(Fl.dat)
                Fl.dat = Fit_Flow3.loadDat(Fl.dat_path, Fl.dat_filt, Fl.col_map);
            end
            if nargout > 0
                dat = Fl.dat;
            end
        end
        
        function Fl = loadobj(Fl)
            Fl.loadDatFl;
            
            %% Fill in th_fix for backward compatibility
            if ~isprop(Fl,'th_fix') || isempty(Fl.th_fix) ...
                    || ~isempty(setdiff(fieldnames(Fl.th_fix), fieldnames(Fl.th)))
                ths = fieldnames(Fl.th);
                nTh = length(ths);
                Fl.th_fix = cell2struct(num2cell(false(nTh,1)),ths,1);
            end
            
            Flnew = Fit_Flow3;
            Fl = copyFields(Fl, Flnew, {'fit_arg', 'fit_opt'});
            
            %% Restore fun
            C = cellfun(@str2func, struct2cell(Fl.fun), 'UniformOutput', false, ...
                    'ErrorHandler', @(err, arg) arg);
            if ~isempty(C)
                Fl.fun = cell2struct(C, fieldnames(Fl.fun));
            else
                Fl.fun = struct;
            end
            
            %% Restore PlotFcns
            Fl.PlotFcns = cellfun(@str2func, Fl.PlotFcns, 'UniformOutput', false, ...
                'ErrorHandler', @(err, arg) arg);
            
            %% Restore OutputFcns
            Fl.OutputFcns = cellfun(@str2func, Fl.OutputFcns, 'UniformOutput', false, ...
                'ErrorHandler', @(err, arg) arg);
            
            %% Restore W
            % If Fl.W is emptied and Fl.W0 has something, replace W with W0.
            if ~isempty(Fl.W0) && (isempty(Fl.W) || isequal(Fl.W, struct))
                Fl.W = Fl.W0;
            end
        end
    end
    
    %% ===== Static methods =====
    methods (Static)
        %% Fitting
        function [cost, W] = calc_cost(th_vec, th_names, W, fun, fun_out, fun_opt, fun_names, opt)
            % [cost, W] = calc_cost(th_vec, th_names, W, fun, fun_out, fun_opt, fun_names, opt)
            
            if nargin < 8
                opt = {};
            end
            opt = varargin2S(opt, {
                'cost_inf2realmax', true
                'cost_nan2realmax', true
                'th_imag2realmax',  true
                'calc_cost',        true
                'fun_aft_every_step',  [] % Give a function handle, e.g., Fl.plotfun
                'pause_every_step', false
                });
            
            % Copy current function value
            nth = length(th_vec);
            for ith = 1:nth
                W.(th_names{ith}) = th_vec(ith);
            end
            
            % Copy subflow variables, if any
            if isfield(W, 'sub_th')
                for csub = fieldnames(W.sub_th)'
                    W = Fit_Flow3.main2sub(W, csub{1}, W.sub_th.(csub{1}));
                end
            end
            
            % Which functions to run
            if nargin < 7 || isempty(fun_names)
                fun_names = fieldnames(fun)';
            end
            
            % Run functions   
%             tic; % DEBUG
%             tSt = datestr(now, 'HHMMSS.fff'); % DEBUG
%             disp(tSt); % DEBUG
            for ccfun = fun_names
                cfun = ccfun{1};
                
                if ischar(fun.(cfun)), fun.(cfun) = str2func(fun.(cfun)); end
                
%                 fprintf('-- %s: %s\n', tSt, cfun); % DEBUG
%                 disp(W);
%                 fprintf('== %s: %s\n', tSt, cfun); % DEBUG
                
                try
                    csub = fun_opt.sub.(cfun);
                catch
                    csub = ''; % For backward compatibility
                end
%                 try
                    if isempty(csub)
                        W = S2io(W, fun.(cfun), fun_out.(cfun), fun_opt.inp.(cfun), ...
                            'use_varargout', fun_opt.use_varargout.(cfun));
                    else
                        W.(csub) = S2io(W.(csub), fun.(cfun), fun_out.(cfun), fun_opt.inp.(cfun), ...
                            'use_varargout', fun_opt.use_varargout.(cfun));
                    end
%                 catch err
%                     fprintf('Error evaluating fun.%s: %s\n', cfun, func2str(fun.(cfun)));
%                     rethrow(err);
%                 end
            end
            
%             toc; % DEBUG
            
            % Options
            if opt.calc_cost
                try
                    % Cost postprocessing
                    if opt.cost_inf2realmax && isinf(W.cost)
                        cost = realmax;
                    elseif opt.cost_nan2realmax && isnan(W.cost)
                        cost = realmax;
                    elseif opt.th_imag2realmax && any(~isreal(th_vec))
                        cost = realmax;
                    else
                        cost = W.cost;
                    end

                    if ~isfinite(cost) || any(~isfinite(th_vec))
                        warning('cost or at least one of the parameters is not finite!');
                        eprintf cost
                        eprintf th_vec
                        keyboard;
                    end
                catch err
                    warning(err_msg(err));
                    cost = nan;
                end
            else
                cost = nan;
            end
            
            if ~isempty(opt.fun_aft_every_step)
                opt.fun_aft_every_step();
            end
            
            if opt.pause_every_step
                input('Press ENTER to continue:', 's');
            end
        end
        
        function grid_spec = grid_factorize(grid_specs)
            % grid_spec = grid_factorize(grid_specs)
            
            n = length(grid_specs);
            
            grid_spec = grid_specs{1};
            for mm = 2:n
                grid_spec = grid_combine(grid_spec, grid_specs{mm});
            end
            
            function g = grid_combine(g1, g2)
                if isempty(g1), g = g2; return; end
                if isempty(g2), g = g1; return; end
                
                n1 = length(g1);
                n2 = length(g2);
                g  = cell(n1*n2,1);
                
                kk = 0;
                for ii = 1:length(g1)
                    for jj = 1:length(g2)
                        kk = kk + 1;
                        
                        for f = {'th0', 'th_lb', 'th_ub'}
                            if ~isfield(g1{ii}, f{1}), g1{ii}.(f{1}) = struct; end
                            if ~isfield(g2{jj}, f{1}), g2{jj}.(f{1}) = struct; end
                            
                            g{kk}.(f{1}) = copyFields(g1{ii}.(f{1}), g2{jj}.(f{1}));
                        end
                        
                        for f = {'fit_opt'}
                            if ~isfield(g1{ii}, f{1}), g1{ii}.(f{1}) = {}; end
                            if ~isfield(g2{jj}, f{1}), g2{jj}.(f{1}) = {}; end
                            
                            g{kk}.(f{1}) = varargin2C(copyFields( ...
                                varargin2S(g1{ii}.(f{1})), ...
                                varargin2S(g2{jj}.(f{1}))));
                        end
                    end
                end
            end
        end
        
        function ds = grid_view(grid_spec)
            % ds = grid_view(grid_spec)
            
            ds = dataset;
            
            for ii = length(grid_spec):-1:1
                for jj = {'th0', 'th_lb', 'th_ub'}
                    for kk = fieldnames(grid_spec{ii}.(jj{1}))'
                        col = str_con(kk{1}, jj{1});
                        ds = ds_set(ds, ii, col, grid_spec{ii}.(jj{1}).(kk{1}));
                    end
                end
            end
            
            [~,ix] = sort(ds.Properties.VarNames);
            ds = ds(:,ix);
        end
        
        function [grid_spec, grid_opt] = grid_setup_static(spec, fit_opt, grid_opt, Fl)
            % grid_setup : wrapper for grid_setup_static
            %
            % grid_setup : grid gives specifications to run fits
            %
            % [grid_spec, grid_opt] = grid_setup(Fl, spec, fit_opt, grid_opt, [Fl])
            %
            % spec: 
            %   {} % Use default th0, th_lb, th_ub
            %
            %   {'var1', val1, ...} % Use all combinations
            %
            %   {'var1', 'var2', ..., 'varK'
            %     [x0_var1_1, lb_var1_1, ub_var1_1], [x0_var2_1, ...], ..., [x0_varK_1, ...]
            %     [x0_var1_2, lb_var1_2, ub_var1_2], ...
            %     ...
            %     [x0_var1_NCOMB, lb_var1_NCOMB, ub_var1_NCOMB], ...
            %   }} % Use given combinations. Allowed only when number of variables > 1.
            %      % Use {'var1', {[x0_1, lb_1, ub_1], ...}} in case of one variable.
            % 
            % val : vector: evaluate within [val(1), val(2)], with an initial value of (val(1)+val(2))/2, then [val(2), val(3)], ...
            %       scalar: equivalent to giving linspace(lb, ub, val)
            %       cell  : evaluate within [val{1}(2), val{1}(3)], with an initial value of val{1}(1), then [val{2}(1), val{2}(2)], ...
            %       cell with a scalar numeric: fix value to val{1}(1). Equivalent to repmat(val{1}(1), [1, 3]).
            %       give NaN to use the value of Fl.th0, th_lb, th_ub.
            %
            % grid_spec{k}: a struct with fields of th0, th_lb, th_ub, fit_opt.
            % grid_opt    : a struct.
            % .restrict   : restrict th_lb and th_ub around th0.
            % .parallel   : use parfor
            %
            % See also: grid_factorize
            
            if nargin < 2, fit_opt = {}; end
            if nargin < 3, grid_opt = {}; end
            grid_opt = varargin2S(grid_opt, {
                'restrict', true
                'parallel', false
                });
            
            % Parse spec
            if ~isempty(spec) && iscell(spec) && isstruct(spec{1})
                grid_spec = spec;
                ncomb = length(grid_spec);
            else
                % Parse spec - get spec_nam, nspec, comb, ncomb
                if size(spec,1) > 1 && ~ischar(spec{2,1})
                    % First row contains variable names
                    spec_nam   = spec(1,:);
                    comb       = spec(2:end,:);
                    ncomb      = size(comb, 1);
                else
                    % Name-value pair
                    spec_nam   = spec(1:2:end);
                    spec_range = spec(2:2:end);
                    nspec      = length(spec_nam);

                    for ispec = 1:nspec
                        spec_range{ispec} = parse_spec(spec_nam{ispec}, spec_range{ispec});
                    end

                    [comb, ncomb] = factorize(spec_range);
                end
                
                % Output
                grid_spec = cell(ncomb,1);
                for ii = 1:ncomb
                    nspec = length(spec_nam);
                    for jj = 1:nspec
                        cspec = comb{ii,jj};

                        nam = spec_nam{jj};
                        if isnan(cspec(1)) % Clamp to th0
                            grid_spec{ii}.th0.(nam) = Fl.th0.(nam);
                        else
                            grid_spec{ii}.th0.(nam) = cspec(1);
                        end
                        if length(cspec) == 1
                            % Fix to one value.
                            grid_spec{ii}.th_lb.(nam) = grid_spec{ii}.th0.(nam);
                            grid_spec{ii}.th_ub.(nam) = grid_spec{ii}.th0.(nam);
                        else % Give NaN for lb and ub to preserve its original value
                            if length(cspec) >= 2 && ~isnan(cspec(2)), grid_spec{ii}.th_lb.(nam) = cspec(2); end
                            if length(cspec) >= 3 && ~isnan(cspec(3)), grid_spec{ii}.th_ub.(nam) = cspec(3); end
                        end
                    end
                end
            end
            if ncomb == 1, grid_opt.parallel = false; end
            
            % Modify fit_opt
            for ii = 1:ncomb
                if grid_opt.parallel
                    if length(fit_opt) < 3
                        fit_opt{3} = {};
                    end
                    fit_opt{3} = varargin2C(fit_opt{3}, {
                        'PlotFcns', {} % Cannot plot if parallel
                        });
                end
                grid_spec{ii}.fit_opt = fit_opt;
            end
            
            function cspec = parse_spec(nam, cspec)
                % Coerce into a cell form
                if isnumeric(cspec)
                    if isscalar(cspec)
                        cspec = parse_spec(nam, ...
                            linspace(Fl.th_lb.(nam), Fl.th_ub.(nam), cspec + 1));
                    else
                        vec   = cspec;
                        nvec  = length(vec) - 1;
                        cspec = cell(1, nvec);
                        for kk = 1:nvec
                            cspec{kk} = [(vec(kk) + vec(kk+1))/2, vec(kk), vec(kk+1)];
                        end
                    end
                end
            end
            
%             grid_spec = packStruct(comb, spec_nam, grid_opt, fit_opt); % Old format. Deprecated.
        end

        function [res, res_all] = grid_gather_static(res_all, grid_spec, grid_opt)
            % [res, res_all] = grid_gather_static(res_all, grid_spec, grid_opt)
            
            % Fetch output if a job
            nres = numel(res_all);
            finished = true(1,nres);
            for ii = 1:nres
                if ~isstruct(res_all{ii})
                    if strcmp(res_all{ii}.State, 'finished')
                        res_all{ii} = fetchOutputs(res_all{ii});
                    else
                        finished(ii) = false;
                    end
                end
            end
            
            % Find minimum
            res      = struct;
            res_all  = res_all(finished);
            
            if any(finished)
                fval_min = inf;
                for ii = 1:nres
                    % Find minimum
                    if res_all{ii}.out.fval < fval_min
                        res = res_all{ii};
                        fval_min = res_all{ii}.out.fval;
                    end
                end
                % Output
                res.grid = packStruct(res_all, grid_spec, grid_opt);
            else
                res = struct;
            end
        end
        
        %% Subflows (static)
        function W = sub2main(W, sub, vars)
            % sub2main(W, sub, vars)
            for v = vars(:)'
                W.([v{1}, '_' sub]) = W.(sub).(v{1});
            end
        end
        
        function W = main2sub(W, sub, vars)
            % main2sub(W, sub, vars)
            for v = vars(:)'
                W.(sub).(v{1}) = W.([v{1}, '_' sub]);
            end
        end
        
        %% Load
        function res = res_func2str(res)
            try res.arg.fun = func2str(res.arg.fun); catch, end
            try res.arg.options.PlotFcns = cellfun(@func2str, ...
                    res.arg.options.PlotFcns, 'UniformOutput', false); catch, end
            
            try res.arg.options.OutputFcn = ...
                    func2str(res.arg.options.OutputFcn); catch, end
            try res.opt = res.arg.options; catch, end
            
            try
                if isfield(res, 'grid') && ~isempty(res.grid) && isfield(res.grid, 'res_all') && ~isempty(res.grid.res_all)
                    res.grid.res_all = cellfun(@Fit_flow3.res_func2str, res.grid.res_all, ...
                        'UniformOutput', false, 'ErrorHandler', @(varargin) []);
                end
            catch
            end
        end
        
        function resave(f)
            if iscell(f)
                for ii = 1:numel(f)
                    Fit_Flow3.resave(f{ii});
                end
            else
                L = load(f); %#ok<NASGU>
                save(f, '-struct', 'L');
                fprintf('Fit_Flow3.resave : Resaved %s\n', f);
            end
        end
        
        function dat = loadDat(dat_path, dat_filt, col_map)
            try
                load(dat_path, 'dat');
            catch err
                warning(err_msg(err));
            end
            if exist('dat', 'var')
                fprintf('Data loaded from %s\n', dat_path);
                n = length(dat); %#ok<NODEF>

                if isempty(dat_filt)
                    dat_filt = true(n, 1);
                else
                    if isa(dat_filt, 'function_handle')
                        dat_filt = dat_filt(dat);
                    end
                end
                dat = dat(dat_filt,:);
                n_filt = length(dat);
                fprintf('%d/%d rows are used.\n', n_filt, n);

                if ~isempty(col_map)
                    cols = fieldnames(col_map)';

                    for ccol = cols
                        dst  = ccol{1};
                        src  = col_map.(ccol{1});
                        fprintf('Column %s -> %s\n', src, dst);
                        dat.(dst) = dat.(src);
                    end
                    
                    % Remove other columns
                    dat = dat(:, cols);
                end
            else
                warning('Neither dat nor dat_path is empty! Keeping dat and skipping loading...');
            end
        end
        
        %% Small utilities
        function vOld = fill_vec(vNew, ixNew, vOld)
            % vOld = fill_vec(vNew, ixNew, vOld)
            vOld(ixNew) = vNew;
        end
        
        %% Declarations - compare, collect
        ds = compare(Fls_or_files, varargin);
        ds = collect(flt, varargin);
    end
end

