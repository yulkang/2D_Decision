classdef PlotCh2D < DtbPlot.PlotPdf2D
    % DtbPlot.PlotCh2D
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
    % Pl = PlotCh2D(cPdf, plArgs, plotArgs)
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
    % See also: DtbPlot.PlotRt2D, DtbPlot.PlotPdf2D
    
    % 2015 YK wrote the initial version.

methods
function Pl = PlotCh2D(varargin)
    % Pl = PlotCh2D(pdf, varargin)
    % pdf(t, cond1, cond2, ch1, ch2) : probability mass

    Pl = Pl@DtbPlot.PlotPdf2D(varargin{:});
end
function y = get_y(Pl)
    % y(condX, condSep)

    % p0: (t, condX, condSep, chX, chSep) 
    p0 = Pl.pdf_permuted;
        
    % Filter accu on dimSep only. 
    % (Cannot filter accu on dimX - then the line would be always flat.)
    for axis = 2
        p0 = Pl.filter_p_by_accu(p0, axis);
    end

    % Fold axes
    p0 = Pl.fold_p(p0);
    
    % Group conditions
    for axis = 1:2
        p0 = Pl.group_p(p0, axis);
    end

    % Marginalize over time and chSep
    p = sums(p0, [1, 5]);
    p = permute(p, [2, 3, 4, 1]);

    % Proportion of choice
    ySum = sum(p, 3);
    y1 = p(:,:,2);
    y = y1 ./ ySum;
end
function h = plot(Pl, varargin)
    S = varargin2S(varargin, {
        'src', 'pred' % 'pred'|'data'
        });
        
    y = Pl.get_y;
    x = Pl.get_x;
    ax = Pl.ax;
    
    n_sep = size(y, 2);
    colors = Pl.get_colors(n_sep);
    
    S2 = rmfield(S, 'src');
    C2 = varargin2C(S2);
    h = ghandles(1, n_sep);
    for i_sep = 1:n_sep
        color = colors(i_sep, :);
        C = bml.plot.varargin2plot(...
                bml.plot.varargin2plot(C2, {
                    'Color', color
                    'MarkerFaceColor', color
                    }), Pl.plotArgs);
        h(i_sep) = plot(ax, x, y(:, i_sep), C{:});
        hold(ax, 'on');
    end
    hold(ax, 'off');
    
    if Pl.foldDim(Pl.dimOnX)
        ylim(ax, [0.475 1]);
        set(ax, ...
            'YTick', [0.5 0.75 1]);
    else
        ylim(ax, [-0.05, 1]);
        set(ax, ...
            'YTick', 0:0.25:1, ...
            'YTickLabel', {'0', '', '0.5', '', '1'});
    end
    
    [x_tick, x_ticklabel] = Pl.get_x_tick;
    if ~isempty(x_tick)
        set(ax, ...
            'XTick', x_tick, ...
            'XTickLabel', csprintf('%g', x_ticklabel));
    end
        
    if Pl.foldDim(Pl.dimOnX)
        if ~isempty(x_tick)
            xlim(ax, [x_tick(1), ...
                x_tick(1) + (max(x_tick) - min(x_tick)) * 1.05]);
        end
    else
        max_abs_x = max(abs(Pl.x));
        xlim(ax, [-1.05, 1.05] .* max_abs_x);
    end
    
    bml.plot.beautify(ax);
    
    switch Pl.dimOnX
        case 1
            ylabel(ax, 'P_{right}');
            
        case 2
            ylabel(ax, 'P_{blue}');
    end
end
end
end