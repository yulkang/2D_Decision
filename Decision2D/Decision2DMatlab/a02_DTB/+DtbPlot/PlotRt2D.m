classdef PlotRt2D < DtbPlot.PlotPdf2D
    % DtbPlot.PlotRt2D
    %
    % With Fit.D2.Common.Main and its subclasses, 
    % try using plot_W_pred_data and plot_W:
    %
    % [hd, hp, Pl_d, Pl_p] = plot_W_pred_data(W, varargin)
    %
    % : Draws data over pred.
    %   See also plot_W.
    %
    % [h, Pl] = plot_W(W, varargin)
    %
    % OPTIONS
    % -------
    % 'src', 'data' % 'data' | 'pred'
    % 'dimOnX', 1
    % 'foldX', false
    % 'foldSep', true
    % 'oversample_factor', 10
    % 'plArgs', {}
    % 'plotArgs', {}
    %
    %
    % Pl = PlotRt2D(cPdf, plArgs, plotArgs)
    %
    % cPdf(t, cond1, cond2, ch1, ch2) : probability mass
    % plArgs: properties of Pl
    % plotArgs: name-value pair arguments of plot()
    %
    % plArgs
    % ------
    % dimOnX = 1; % Dimension to plot on x. 1 or 2
    % conds = {[],[]}; % conds{dim}: vector of conditions.
    % 
    % logAxis = [false, false]; % [logX, logY]
    % 
    % % conds_tick{dim}: vector of conditions to show on tick / legend
    % conds_tick = {[], []}; 
    % 
    % foldAxis = [true, true]; % [foldX, foldSep]
    % accuOnlyAxis = [true, true]; % [accuOnlyX, accuOnlySep]
    % colors = []; % set as a function handle
    % 
    % x_bias = []; % if nonempty, break predictions
    % 
    % plotArgs = {};
    % dt = 0.01;
    % 
    % plotNow = false; % Set on construction to plot right away
    %
    % See also: DtbPlot.PlotCh2D, DtbPlot.PlotPdf2D
    
    % 2015 YK wrote the initial version.
        
properties (Dependent)
    y_fun
end
properties
    y_fun_ = 'mean';
end    
methods
    function Pl = PlotRt2D(varargin)
        % Pl = PlotRt2D(pdf, varargin)
        % pdf(t, cond1, cond2, ch1, ch2) : probability mass

        Pl = Pl@DtbPlot.PlotPdf2D(varargin{:});
    end
    function y = get_y(Pl) 
        % p0: (t, condX, condSep, chX, chSep) 
        % y: (dimOnX, dimSep, chX)
        p0 = Pl.pdf_permuted;
        
        % Filter accu
        for axis = 1:2
            p0 = Pl.filter_p_by_accu(p0, axis);
        end
        
        % Fold axes
        p0 = Pl.fold_p(p0);
        
        % Group conditions
        for axis = 1:2
            p0 = Pl.group_p(p0, axis);
        end
        
        % Sum across sep choices
        p = sums(p0, 5);
        
        % Summarize
        switch Pl.y_fun
            case 'mean'
                y = mean_distrib(p, Pl.t(:), 1);
            case 'var'
                y = std_distrib(p, Pl.t(:), 1).^2;
            case 'stdev'
                y = std_distrib(p, Pl.t(:), 1);
            case 'skew'
                y = skew_distrib(p, Pl.t(:), 1);
        end
        
        % Skip y that are filtered out
        is_out = permute(sum(y,1) == 0, [2, 3, 4, 1]);
        y(is_out) = nan;

        % y: (dimOnX, dimSep, chX) <- (t, dimOnX, dimSep, chX)
        y = permute(y, [2, 3, 4, 1]);
    end  
    function v = get.y_fun(Pl)
        v = Pl.y_fun_;
    end
    function set.y_fun(Pl, y_fun)
        Pl.set_y_fun(y_fun);
    end
    function set_y_fun(Pl, y_fun)
        assert(ischar(y_fun) && ismember(y_fun, {'mean', 'var', 'stdev', 'skew'}));
        Pl.y_fun_ = y_fun;
    end
    function h = plot(Pl, varargin)
        S = varargin2S(varargin, {
            'src', 'pred' % 'pred'|'data'
            });
        
        y = Pl.get_y;
        x = Pl.get_x;
        ax = Pl.ax;
        
        n_sep = size(y, 2);
        n_ch = size(y, 3);
        colors = Pl.get_colors(n_sep);
        
        h = ghandles(n_sep, n_ch);
        
        if strcmp(S.src, 'data') && all(Pl.biasDim{Pl.dimOnX} == 0)
            % If bias == 0, average across choices
            ix0 = Pl.condsDim{Pl.dimOnX} == 0;
            y(ix0, :, :) = repmat(mean(y(ix0, :, :), 3), ...
                [1, 1, size(y, 3)]);
        end            
        
        S2 = rmfield(S, 'src');
        C2 = varargin2C(S2);
        for i_sep = 1:n_sep
            color = colors(i_sep, :);
            for i_ch = 1:n_ch
                C = bml.plot.varargin2plot(...
                        bml.plot.varargin2plot(C2, {
                            'Color', color
                            'MarkerFaceColor', color
                            }), Pl.plotArgs);
                h(i_sep, i_ch) = plot(ax, x, y(:,i_sep,i_ch), C{:});
                hold(ax, 'on');
            end
        end
        hold(ax, 'off');
        
        bml.plot.lim_margin('h', ax, 'axis', 'y', 'direction', 'pos');

        [x_tick, x_ticklabel] = Pl.get_x_tick;
        if ~isempty(x_tick)
            set(ax, ...
                'XTick', x_tick, ...
                'XTickLabel', csprintf('%g', x_ticklabel));
        end

        if Pl.foldAxis(1)
            if ~isempty(x_tick)
                xlim(ax, [x_tick(1), ...
                    x_tick(1) + (max(x_tick) - min(x_tick)) * 1.05]);
            end
        else
            max_abs_x = max(abs(Pl.x));
            xlim(ax, [-1.05, 1.05] .* max_abs_x);
        end

        bml.plot.beautify_lim('ax', ax);
        bml.plot.beautify(ax);
        bml.plot.beautify_tick(ax, 'y');
        
        switch Pl.y_fun
            case 'mean'
                ylabel(ax, 'RT (s)');
            otherwise
                ylabel(ax, [Pl.y_fun, ' RT (s)']);
        end     
    end
end 
%% PlotFcns
methods (Static)
    function stop = outputfun(x, v, s, varargin)
        stop = false;
        Pl = DtbPlot.PlotRt2D(varargin{:});
        Pl.plot;
    end
end
end % classdef