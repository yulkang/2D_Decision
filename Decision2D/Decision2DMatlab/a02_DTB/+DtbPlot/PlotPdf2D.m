classdef PlotPdf2D < matlab.mixin.Copyable
    % DtbPlot.PlotPdf2D
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
    % 'plot_kind', 'Ch' % 'Ch'|'Rt'    
    %
    %
    % For any array(t, cond1, cond2, ch1, ch2), use:
    %
    % Pl = PlotPdf2D(cPdf, plArgs, plotArgs)
    %
    % cPdf(t, cond1, cond2, ch1, ch2) : probability mass
    % plArgs: properties of Pl
    % plotArgs: name-value pair arguments of plot()
    %
    % plArgs
    % ------
    % dimOnX = 1; % Dimension to plot on x. 1 or 2
    % condsDim = {[],[]}; % condsDim{dim}: vector of conditions.
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
    % plotArgs = {}; % Fed to plot()
    % dt = 0.01;
    % 
    % % plotNow: Set true on construction to plot right away.
    % %          If false, use h = Pl.plot to plot.
    % plotNow = false; 
    %
    % See also: DtbPlot.PlotCh2D, DtbPlot.PlotRt2D
    
    % 2015 YK wrote the initial version.
        
%% Properties - Settings - Required
properties (Dependent)
    pdf % pdf(t, cond1, cond2, ch1, ch2) : probability mass
end
properties (Hidden)
    pdf_ = [];
end
%% Properties - Settings - Optional
properties (Dependent)
    ax % handle to the axis to draw on.
    conds % conds{dim}: same as condsDim. Kept for compatibility.
    condsDim % condsDim{dim} :  vector of conditions.
    condsAxis % condsAxis{axis}
    
    % groupAxis{axis}
    % : empty: skip grouping.
    % : scalar: number of groups.
    % : vector: (cond) = i_group. Set 0 to skip the condition.
    % : Applies after filtering accu and folding.
    groupAxis 
    
    x_bias % (1, condSep): if nonempty, break predictions
    sep_bias % (1, condX): determines how to filter accurate/not
end
properties
    dimOnX = 1; % Dimension to plot on x. 1 or 2
    logAxis = [false, false]; % [logX, logY]    
    
    % conds_tick{dim}: vector of conditions to show on tick / legend
    conds_tick = {[], []}; 
    
    foldAxis = [false, true]; % [foldX, foldSep]
    
    % accuOnlyAxis: 0 for both, 1 for accurate only, 2 for wrong only
    % [accuOnlyX, accuOnlySep]
    accuOnlyAxis = [true, true]; 
    colors = []; % set as a function handle
    
    biasDim_ = {[], []};
    use_bias = true;

    plotArgs = {}; % Fed to plot()
    dt = 0.01;
    
    % plotNow: Set true on construction to plot right away.
    %          If false, use h = Pl.plot to plot.
    plotNow = false; 
end
%% Intermediate variables
properties (Dependent)
    n_cond
end
properties (Hidden)
    ax_ = [];
    
    condsDim_ = {[],[]};
    groupAxis_ = {[], []};    
    
    n_cond_ = [];
end
%% Properties - Intermediate variables
properties (Dependent)
    x
    x_tick
    x_ticklabel
    
    y
    t % Determined by size(pdf,1) and dt
    nt % Determined by size(pdf,1)
    dimSep % dimension that separates curves.
    foldDim % foldAxis([dimOnX, dimSep])
    accuOnlyDim % accuOnlyAxis([dimOnX, dimSep])
    sepConds
    
    dimAxis % (axis) = dim corresponds to that axis (1=x, 2=sep)
    
    biasDim % {(bias1_in_cond2), (bias2_in_cond1)}
    biasAxis % {(biasX_in_condSep), (biasSep_in_condX)}
    
    pdf_permuted % (t, dimOnX, dimSep, chX, chSep)
end
%% FitFlow interface - OutputFun
methods (Static)
    function stop = outputfun(x, v, s, varargin)
        stop = false;
        Pl = DtbPlot.PlotCh2D(varargin{:});
        Pl.plot;
    end
end
%% FitWorkspace interface
methods
    function [hd, hp, Pl_d, Pl_p] = plot_W_pred_data_cum(Pl0, W, varargin)
        % [hd, hp, Pl_d, Pl_p] = plot_W_pred_data_cum(Pl, W, varargin)
        %
        % Draws data over pred. (cumsum across time)
        % See also plot_W.
        
        S = varargin2S(varargin, {
            'src', {'pred_cum', 'data_cum'}
            });
        
        if ismember('pred_cum', S.src)
            C = varargin2C({'src', 'pred_cum'}, varargin);
            [hp, Pl_p] = Pl0.plot_W(W, C{:});

            bml.plot.hold(Pl_p.ax, 'on');
        end
        if ismember('data_cum', S.src)
            C = varargin2C({
                'src', 'data_cum'
                'yfun', 'raw'
                }, varargin);
            [hd, Pl_d] = Pl0.plot_W(W, C{:});

            bml.plot.hold(Pl_d.ax, 'off');
        end
    end
    function [hd, hp, Pl_d, Pl_p] = plot_W_pred_data(Pl0, W, varargin)
        % [hd, hp, Pl_d, Pl_p] = plot_W_pred_data(Pl, W, varargin)
        %
        % Draws data over pred.
        % See also plot_W.
        
        S = varargin2S(varargin, {
            'src', {'pred', 'data'}
            });
        
        if ismember('pred', S.src)
            C = varargin2C({'src', 'pred'}, varargin);
            [hp, Pl_p] = Pl0.plot_W(W, C{:});

            bml.plot.hold(Pl_p.ax, 'on');
        end
        if ismember('data', S.src)
            C = varargin2C({'src', 'data'}, varargin);
            [hd, Pl_d] = Pl0.plot_W(W, C{:});

            bml.plot.hold(Pl_d.ax, 'off');
        end
    end
    function [h, Pl] = plot_W(Pl0, W, varargin)
        % [h, Pl] = plot_W(Pl, W, varargin)
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
        % 'plot_kind', 'Ch' % 'Ch'|'Rt'
        
        S = varargin2S(varargin, {
            'src', 'data' % 'data'|'pred'|'cost'
            'oversample_factor', 1 % 10
            'plotArgs', {}
            'dimOnX', 1
            });
        n_dim = 2;
        S.dimSep = n_dim + 1 - S.dimOnX;
        
        p = Pl0.get_p_from_W(W, S.src, S.dimOnX, S.oversample_factor, S);
        plotArgs = bml.plot.varargin2plot(S.plotArgs, ...
            Pl0.get_style_by_src(S.src));
        
        conds_tick = W.Data.get_conds_wo_oversample;
        conds_tick{S.dimOnX} = ...
            conds_tick{S.dimOnX}([1, round(end/2) ,end]);
        
        plArgsS = varargin2S(S, varargin2S(Pl0.get_plArgs_by_src(S.src), {
            'dt', W.get_dt
            'condsDim', W.Data.get_conds
            'conds_tick', conds_tick
            'use_bias', true
            'x_bias', W.cond_bias{S.dimOnX}
            'sep_bias', W.cond_bias{S.dimSep}
            }));
        plArgsS = bml.struct.orderfields(plArgsS, ...
            fieldnames(S), 'first');
        plArgs = varargin2C(plArgsS);
        
        Pl = feval(class(Pl0), p, plArgs, plotArgs);        
        
        if isempty(p)
            warning('p is empty!');
            h = gobjects;
        else        
            h = Pl.plot;
        end
    end
    function [h, Pl] = plot_p(Pl0, p, varargin)
        % [h, Pl] = plot_W(Pl, p, varargin)
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
        % 'plot_kind', 'Ch' % 'Ch'|'Rt'
        
        S = varargin2S(varargin, {
            'src', 'data' % 'data'|'pred'|'cost'
            'oversample_factor', 1 % 10
            'plotArgs', {}
            'dimOnX', 1
            });
        n_dim = 2;
        S.dimSep = n_dim + 1 - S.dimOnX;
        
        plotArgs = bml.plot.varargin2plot(S.plotArgs, ...
            Pl0.get_style_by_src(S.src));
        
        conds_dim = repmat( ...
            {[-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1]}, ...
            [1, 2]);
        conds_tick = conds_dim;
        conds_tick{S.dimOnX} = ...
            [-1, -0.5, 0, 0.5, 1];
        
        plArgsS = varargin2S(S, varargin2S(Pl0.get_plArgs_by_src(S.src), {
            'dt', 1/75
            'condsDim', conds_dim
            'conds_tick', conds_tick
            'use_bias', true
            'x_bias', 0
            'sep_bias', 0
            }));
        plArgsS = bml.struct.orderfields(plArgsS, ...
            fieldnames(S), 'first');
        plArgs = varargin2C(plArgsS);
        
        Pl = feval(class(Pl0), p, plArgs, plotArgs);        
        
        if isempty(p)
            warning('p is empty!');
            h = gobjects;
        else        
            h = Pl.plot('src', S.src);
        end
    end
    function p = get_p_from_W(~, W, src, dimOnX, oversample_factor, S)
        n_dim = 2;
        dimSep = n_dim + 1 - dimOnX;
        switch src
            case 'data'
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
%                 W.Data.refresh_RT_data_pdf;
                
                p = W.Data.get_RT_data_pdf;
                
            case 'pred'
                W.Data.set_conds_oversample_factor( ...
                    oversample_factor, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                if ~W.Data.is_pred_done
                    W.pred;
                end
                
                p = W.Data.get_RT_pred_pdf;
                
            case 'data_cum'
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                W.Data.refresh_RT_data_pdf;
                
                p0 = W.Data.get_RT_data_pdf;
                p = cumsum(p0);
                
            case 'pred_cum'
                W.Data.set_conds_oversample_factor( ...
                    oversample_factor, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                if ~W.Data.is_pred_done
                    W.pred;
                end
                
                p = cumsum(W.Data.get_RT_pred_pdf);
                
            case 'cost'
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                W.Data.refresh_RT_data_pdf;
                
                [~, p] = W.get_cost;
                
            case 'cost_dif'
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                W.Data.refresh_RT_data_pdf;
                
                [~, p1] = W.get_cost;
                
                W = S.W2;
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                W.Data.refresh_RT_data_pdf;
                
                [~, p2] = W.get_cost;
                
                p = p1 - p2;
                
            case 'cost_dif_cum'
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                W.Data.refresh_RT_data_pdf;
                
                [~, p1] = W.get_cost;
                
                W = S.W2;
                W.Data.set_conds_oversample_factor(1, dimOnX);
                W.Data.set_conds_oversample_factor(1, dimSep);
                W.Data.refresh_RT_data_pdf;
                
                [~, p2] = W.get_cost;
                
                p = p1 - p2;        
                p = cumsum(p); % cumsum across time
        end
    end
    function plArgs = get_plArgs_by_src(~, src)
        plArgs = {};
    end
    function plotArgs = get_style_by_src(~, src)
        switch src
            case 'data'
                plotArgs = Fig.style_data;
                
            case {'pred', 'cost'}
                plotArgs = Fig.style_pred;
        end
    end
end
%% Init
methods
    function Pl = PlotPdf2D(cPdf, plArgs, plotArgs)
        % Pl = PlotPdf2D(cPdf, plArgs, plotArgs)
        %
        % cPdf(t, cond1, cond2, ch1, ch2) : probability mass
        % plArgs: properties of Pl
        % plotArgs: arguments of plot()
        %
        % OPTIONS:
        % dimOnX = 1; % Dimension to plot on x. 1 or 2
        % condsDim = {[],[]}; % condsDim{dim}: vector of conditions.
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
        %           % If false, use h = Pl.plot to plot.

        if nargin < 1, cPdf = []; end
        if nargin < 2, plArgs = {}; end
        if nargin < 3, plotArgs = {}; end

        if ~isempty(cPdf)
            Pl.set_pdf(cPdf);
        end
        bml.oop.varargin2props(Pl, plArgs, true);
        
        Pl.plotArgs = plotArgs;
        if Pl.plotNow || (nargout == 0 && ~isempty(cPdf))
            Pl.plot(plotArgs{:});
        end
    end
end
%% Plot
methods
    function varargout = plot(Pl, varargin) %#ok<STOUT>
        error('Modify in subclass!');
        
%         % x : (dimOnX, dimSep)
%         % y : (dimOnX, dimSep)
%         plotArgs = varargin2C(varargin, Pl.plotArgs);
% 
%         x = Pl.x;
%         y = Pl.y;
% 
%         x_tick = Pl.x_tick;
% 
%         if Pl.logAxis(1) && x(2) < x_tick(2)
%             y(2,:) = y(1,:);
%         end
% 
%         if isempty(Pl.colors)
%             if size(y, 2) == 1
%                 colors = [0 0 0];
%             else
%                 colors = @hsv2rev;
%             end
%         else
%             colors = Pl.colors;
%         end 
% 
%         [varargout{1:nargout}] = plotsep(x, y, colors, plotArgs);
    end
end
%% p
methods
    function p = filter_p_by_accu(Pl, p, axis1)
        if axis1 == 2
            p = permute(p, [1 3 2 5 4]);
        end

        n_axis = 2;
        axis2 = n_axis + 1 - axis1;
        conds1 = Pl.condsAxis{axis1};
        conds2 = Pl.condsAxis{axis2};
        n_cond1 = length(conds1);
        n_cond2 = length(conds2);        
        n_ch = 2;
        
        [i_cond1_all, i_cond2_all, ch1_all] = ...
            ndgrid(1:n_cond1, 1:n_cond2, 1:n_ch);
        [is_wrong_all, is0_all] = ...
            Pl.is_wrong(axis1, i_cond1_all, i_cond2_all, ch1_all);
        
        for i_cond2 = 1:n_cond2
            for i_cond1 = 1:n_cond1
                for ch1 = 1:n_ch
                    is_wrong1 = is_wrong_all(i_cond1, i_cond2, ch1);
                    is0 = is0_all(i_cond1, i_cond2, ch1);
                    
                    switch Pl.accuOnlyAxis(axis1)                        
                        case 1 % correct only
                            if ~is0 && is_wrong1
                                p(:, i_cond1, i_cond2, ch1, :) = 0;
                            end
                        case 2 % wrong only
                            if ~is0 && ~is_wrong1
                                p(:, i_cond1, i_cond2, ch1, :) = 0;
                            end
                    end
                end
            end
        end        
        if axis1 == 2
            p = permute(p, [1 3 2 5 4]);
        end
    end
    function [is_wrong1, is0] = is_wrong(Pl, axis1, i_cond1, i_cond2, ch1)
        conds1 = Pl.condsAxis{axis1}(i_cond1);
        
        % Not sure if correct
        axis2 = 3 - axis1;
        bias1 = Pl.biasAxis{axis2}(i_cond2);
        
        if ~Pl.use_bias
            bias1 = zeros(size(bias1));
        end
        
        cond_bias = conds1 - bias1;
        
        is0 = cond_bias == 0;
        is_wrong1 = sign(cond_bias) ~= sign(ch1 - 1.5);
    end
    function p = fold_p(Pl, p0, foldAxis)
        % Input/Output: p(t, dimOnX, dimSep, chX, chSep)
        % axis12: [foldX, foldSep]
        
        if ~exist('p0', 'var')
            p0 = Pl.pdf_permuted;
        end
        if ~exist('foldAxis', 'var')
            foldAxis = Pl.foldAxis;
        end
        if ~any(foldAxis)
            p = p0;
            return;
        end
        
        % Before folding, flip decisions to so that ch=2 is correct.
        for axis1 = find(foldAxis(:)')
            p0 = Pl.flip_ch2_to_accu(p0, axis1);
        end
        
        % Then fold.
        p = p0;
        for axis1 = find(foldAxis(:)')
            p = Pl.fold_p_cond(p, axis1);
        end
    end
    function p0 = flip_ch2_to_accu(Pl, p0, axis1)
        if axis1 == 2
            p0 = permute(p0, [1 3 2 5 4]);
        end
        
        n_cond1 = size(p0, 2);
        n_cond2 = size(p0, 3);
        
        for i_cond2 = 1:n_cond2
            for i_cond1 = 1:n_cond1
                for ch1 = 2 % if ch1 = 2 is wrong, flip.
                    [is_wrong1, is0] = Pl.is_wrong(axis1, ...
                        i_cond1, i_cond2, ch1);
                    
                    if is_wrong1 && ~is0
                        p0(:, i_cond1, i_cond2, :, :) = ...
                            flip(p0(:, i_cond1, i_cond2, :, :), 4);
                    end
                end
            end
        end         
        
        if axis1 == 2
            p0 = permute(p0, [1 3 2 5 4]);
        end
    end
    function p = fold_p_cond(~, p0, axis1)
        if axis1 == 2
            p0 = permute(p0, [1 3 2 5 4]);
        end
        
        % Fold condition.
        n = size(p0, 2);
        n_half1 = floor(n / 2);
        n_half = round(n / 2);
        n_half2 = n - n_half1 + 1;
        
        % Fold, but retain both choices.
        % On the folded axis, choice 2 is the correct choice.
        p = (p0(:, n_half1:-1:1, :, :, :) ...
           + p0(:, n_half2:end,  :, :, :)) ./ 2;

        % Leave the middle condition unchanged.
        if mod(n, 2) == 1
            p = cat(2, p0(:, n_half, :, :, :), p);
        end
        
        if axis1 == 2
            p = permute(p, [1 3 2 5 4]);
        end
    end
    function p = group_p(Pl, p0, axis, varargin)
        % Input
        % : p(t, dimOnX, dimSep, chX, chSep)
        % : axis: 1 if dimX, 2 if dimSep.
        %
        % Options
        % 'op', 'mean'; % 'mean'|'sum'.
        % 
        % Output
        % : if axis=1, p(t, groupX, :, :, :)
        %   if axis=2, p(t, :, groupSep, :, :)
        
        S = varargin2S(varargin, {
            'op', 'mean' % 'mean'|'sum'
            'group', []
            });
        
        if axis == 2
            p0 = permute(p0, [1 3 2 5 4]);
        end
        
        if isempty(S.group)
            group = Pl.get_groupAxis(axis, size(p0, 2));
        else
            group = S.group;
        end
        n_group = max(group);
        
        siz_p = size(p0);
        siz_p(2) = n_group;
        p = zeros(siz_p);
        
        for group1 = 1:n_group
            incl = group == group1;
            
            switch S.op
                case 'mean'
                    p(:,group1,:,:,:) = mean(p0(:,incl,:,:,:), 2);
                case 'sum'
                    p(:,group1,:,:,:) = sum(p0(:,incl,:,:,:), 2);
                otherwise
                    error('Unknown op=%s\n', S.op);
            end
        end
        
        if axis == 2
            p = permute(p, [1 3 2 5 4]);
        end        
    end
    function p = get.pdf_permuted(Pl)
        if Pl.dimOnX == 2
            p = permute(Pl.pdf, [1 3 2 5 4]);
        else
            p = Pl.pdf;
        end
    end
    function set.pdf(Pl, pdf)
        Pl.set_pdf(pdf);
    end
    function set_pdf(Pl, pdf)
        Pl.pdf_ = pdf;
        
        v = size(pdf);
        Pl.n_cond = v([2 3]);
    end
    function v = get.pdf(Pl)
        v = Pl.pdf_;
    end
end
%% x
% NOTE: grouping x conditions are not supported yet.
methods
    function x = get_x(Pl)
        % x(dimOnX, 1)
        %
        % x = get_x(Pl)
        
        dim = Pl.dimOnX;
        x = Pl.condsDim{dim}(:);
        if Pl.foldDim(dim)
            x = x + flipud(-x);
            x = x(round(end / 2):end);
            
            if Pl.logAxis(1) % [dimOnX, dimSep]
                Ax = bml.plot.LogAxis;
                x = Ax.convert_v(x, Pl.get_x_tick0);
            end
        end
    end
    function [x, xticklabel] = get_x_tick(Pl, dim)
        if nargin < 2, dim = Pl.dimOnX; end
        x = Pl.get_x_tick0(dim);
        if Pl.logAxis(1) % [dimOnX, dimSep]
            Ax = bml.plot.LogAxis;
            x = Ax.convert_tick(x);
        end
        if nargout >= 2
            xticklabel = Pl.get_x_ticklabel(dim);
        end
    end
    function x = get_x_tick0(Pl, dim)
        if nargin < 2, dim = Pl.dimOnX; end
        x = Pl.conds_tick{dim};
        if Pl.foldDim(dim)
            x = uniquetol_wrap(abs(x));
        end
    end
    function xticklabel = get_x_ticklabel(Pl, dim)
        if nargin < 2, dim = Pl.dimOnX; end

        xticklabel = Pl.conds_tick{dim};
        if Pl.foldDim(dim)
            xticklabel = uniquetol_wrap(abs(xticklabel));
        end

        is_motion_coh = any(Pl.get_x_tick0 == 0.128);
        if is_motion_coh
            xticklabel = round(xticklabel * 1000) / 10;
        end
    end
end
%% sep : level for each curve. % Under construction
methods
%     function v = get_sep(Pl)
%         v0 = Pl.condsAxis{2};
%     end
%     function v = get_sep_label(Pl)
%     end
end
%% y
methods
    function y = get_y(Pl) %#ok<STOUT>
        error('Implement in subclasses!');
    end
end
%% Decoration
methods
    function beautify_x_axis(Pl)
        ax = Pl.ax;
        switch S.dimOnX
            case 1 % Motion
                xtick = get(ax, 'XTick');
                set(ax, 'XTickLabel', xtick * 100);
                xlabel(ax, 'Motion strength (%)');
                
            case 2 % Color
                xlabel(ax, 'Log odds blue (logit)');
        end
    end
    function legend(Pl, fmt, varargin)
        if nargin < 2, fmt = '%1.2f'; end
        legend(csprintf(fmt, Pl.sepConds), varargin{:});
    end
    function colors = get_colors(Pl, n)
        if isempty(Pl.colors)
            if n == 1
                colors = [0 0 0];
            else
                colors = hsv2rev(n);
            end
        elseif isa(Pl.colors, 'function_handle')
            colors = Pl.colors(n);
        else
            assert(isnumeric(Pl.colors));
            colors = Pl.colors(1:n, :);
        end
    end
    function set.colors(Pl, v)
        Pl.colors = v;
    end
    function v = get.ax(Pl)
        v = Pl.get_ax;
    end
    function v = get_ax(Pl)
        if isempty(Pl.ax_)
            v = gca;
        else
            v = Pl.ax_;
        end        
    end
    function set.ax(Pl, v)
        Pl.ax_ = v;
    end
end
%% Get/Set - dim
methods
    function set.dimOnX(Pl, d)
        assert(isscalar(d) && ((d == 1)  || (d == 2)));
        Pl.dimOnX = d;
    end
    function set.foldAxis(Pl, v)
        assert(all(islogical(v) | isnumeric(v)) ...
                && isequal(size(v), [1 2]));
        Pl.foldAxis = v;
    end
    function v = get.accuOnlyDim(Pl)
        v([Pl.dimOnX, Pl.dimSep]) = Pl.accuOnlyAxis;
    end
    function set.accuOnlyDim(Pl, v)
        Pl.accuOnlyAxis([Pl.dimOnX, Pl.dimSep]) = v;
    end
    function vs = get.biasDim(Pl)
        vs = Pl.get_biasDim;
    end
    function vs = get_biasDim(Pl)
        n_dim = 2;
        vs = Pl.biasDim_;
        for dim = 1:n_dim
            v1 = vs{dim};
            
            dim2 = n_dim + 1 - dim;
            n_conds = length(Pl.condsDim{dim2});
            if isempty(v1)
                v1 = zeros(1, n_conds);
            elseif isscalar(v1)
                v1 = zeros(1, n_conds) + v1;
            end
            vs{dim} = v1;
        end
    end
    function set.biasDim(Pl, v)
        Pl.biasDim_ = v;
    end
    function v = get.biasAxis(Pl)
        v([Pl.dimOnX, Pl.dimSep]) = Pl.biasDim;
    end
    function set.biasAxis(Pl, v)
        assert(iscell(v) && numel(v) == 2);
        Pl.biasDim([Pl.dimOnX, Pl.dimSep]) = v;
    end
end
%% Get/Set - x, y
methods
    function v = get_fold_x(Pl)
        v = Pl.foldAxis(1);
    end
    function v = get_fold_sep(Pl)
        v = Pl.foldAxis(2);
    end
    function set.foldDim(Pl, v)
        Pl.foldAxis([Pl.dimOnX, Pl.dimSep]) = v;
    end
    function v = get.foldDim(Pl)
        v = Pl.foldAxis([Pl.dimOnX, Pl.dimSep]);
    end
    function v = get.n_cond(Pl)
        v = Pl.get_n_cond;
    end
    function v = get_n_cond(Pl)
        if isempty(Pl.n_cond_)
            v = size(Pl.pdf_);
            if prod(v) == 0
                Pl.n_cond_ = [0 0];
            else
                Pl.n_cond_ = v(2:3);
            end
        end
        v = Pl.n_cond_;
    end
    function set.n_cond(Pl, v)
        Pl.n_cond_ = v;
    end
    function x = get.x(Pl)
        x = get_x(Pl);
    end
    function x = get.x_tick(Pl)
        x = get_x_tick(Pl);
    end
    function x = get.x_ticklabel(Pl)
        x = get_x_ticklabel(Pl);
    end
    function y = get.y(Pl)
        y = get_y(Pl);
    end
    function sepConds = get.sepConds(Pl)
        sepConds = Pl.condsDim{Pl.dimSep};
        if Pl.foldDim(Pl.dimSep)
            sepConds = uniquetol(abs(sepConds));
        end 
    end    
    
    function v = get.conds(Pl)
        v = Pl.get_conds;
    end
    function v = get_conds(Pl)
        warning('Deprecated - use condsDim instead!');
        v = Pl.condsDim;
    end
    function set.conds(Pl, v)
        Pl.set_conds(v);
    end
    function set_conds(Pl, v)
        warning('Deprecated - use condsDim instead!');
        Pl.condsDim = v;
    end
    
    function v = get.condsDim(Pl)
        v = Pl.get_condsDim;
    end
    function v = get_condsDim(Pl)
        v = Pl.condsDim_;
        
        n_dim = 2;
        for i_dim = 1:n_dim
            if isempty(v{i_dim})
                nCondHalf = floor(Pl.n_cond(i_dim) / 2);
                if mod(Pl.n_cond(i_dim), 2) == 1
                    v{i_dim} = -nCondHalf:nCondHalf;
                else
                    v{i_dim} = [-nCondHalf:-1, 1:nCondHalf];
                end
%             else
%                 assert(length(v{i_dim}) == Pl.n_cond(i_dim));
%                 v{i_dim} = v{i_dim};
            end
        end
    end
    function set.condsDim(Pl, v)
        Pl.condsDim_ = v;
    end
    
    function v = get.condsAxis(Pl)
        v = Pl.condsDim([Pl.dimOnX, Pl.dimSep]);
    end
    function set.condsAxis(Pl, v)
        Pl.condsDim([Pl.dimOnX, Pl.dimSep]) = v;
    end
    
    function v = get_groupAxis(Pl, axis, n)
        v = Pl.groupAxis_{axis};

        if ~exist('n', 'var')
            n = length(Pl.condsAxis{axis});
        end
        
        if isempty(v)
            v = 1:n;
        elseif isscalar(v)
            v0 = v;
            v = 1:n;
            v = ceil(v ./ n .* v0);
        end
    end
    function v = get.groupAxis(Pl)
        v = Pl.groupAxis_;
    end
    function set.groupAxis(Pl, v)
        Pl.groupAxis_ = v;
    end
end
%% Get/Set - bias
methods
    function bias = get.x_bias(Pl)
        bias = Pl.get_x_bias;
    end
    function bias = get_x_bias(Pl)
        % x_bias(condSep)
        bias = get_bias_dim(Pl, Pl.dimOnX);
    end
    function set.x_bias(Pl, v)
        Pl.set_x_bias(v);
    end
    function set_x_bias(Pl, v)
        Pl.biasDim{Pl.dimOnX} = v;
    end
    function bias = get.sep_bias(Pl)
        % sep_bias(condX)
        bias = get_bias_dim(Pl. Pl.dimSep);
    end
    function set.sep_bias(Pl, v)
        Pl.biasDim{Pl.dimSep} = v;
    end
    function bias = get_bias_dim(Pl, dim)
        bias = Pl.biasDim{dim};
%         bias0 = Pl.biasDim{dim};
%         n_cond = size(Pl.pdf, dim + 1);
%         if isempty(bias0)
%             bias = zeros(n_cond, 1);
%         elseif isscalar(bias0)
%             bias = zeros(n_cond, 1) + bias0;
%         else
%             assert(length(bias0) == n_cond);
%             bias = bias0(:);
%         end
    end
end
%% AccuOnly
methods
    function p = get_pdf_accyOnly_dim(Pl, dim)
        if dim == 2
            p = permute(Pl.pdf, [1 3 2 5 4]);
        else
            p = Pl.pdf;
        end
        
    end
end
%% t
methods
    function t = get.t(Pl)
        t = (0:(Pl.nt - 1)) * Pl.dt;
    end
    function nt = get.nt(Pl)
        nt = size(Pl.pdf, 1);
    end
    function dimSep = get.dimSep(Pl)
        dimSep = 3 - Pl.dimOnX;
    end
    function set.dimSep(Pl, v)
        Pl.dimOnX = 3 - v;
    end
    function v = get.dimAxis(Pl)
        v = [Pl.dimOnX, Pl.dimSep];
    end
    function set.dimAxis(Pl, v)
        assert(numel(v) == 2);
        assert(isnumeric(v));
        Pl.dimOnX = v(1);
        Pl.dimSep = v(2);
    end
end % methods
end % classdef