classdef PlotRtDistribAll2D < DtbPlot.PlotRtDistrib2D
% Plot rt distributions in an array
    
% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

%% Properties - Options
properties
    ch_incl = {':', ':'};
    
    styles_ch = {
        {'-'}, {'-'}
        {'-'}, {'-'}
        };
    
    y_sign_ch = [
        -1, -1
        1, 1
        ];
end
%% FitWorkspace interface
methods
    function plArgs = get_plArgs_by_src(~, src)
        switch src
            case 'pred'
                plArgs = {
                    'styles_ch', repmat({
                        {'-', 'Color', 'r'}
                        {'-', 'Color', 'b'}
                        }', [2, 1]);
                    'smoothTime', 0
                    };
            case 'data'
                plArgs = {
                    'styles_ch', repmat({
                        {'-', 'Color', bml.plot.color_lines('o')}
                        {'-', 'Color', bml.plot.color_lines('c')}
                        }', [2, 1]);
                    };
            case 'cost'
                plArgs = {
                    'styles_ch', repmat({
                        {'-', 'Color', 'r'}
                        {'-', 'Color', 'b'}
                        }', [2, 1]);
                    'y_scale_kind', 'rel_global'
                    };
            case {'cost_dif', 'cost_dif_cum'}
                f = @bml.plot.color_lines;
                plArgs = {
                    'styles_ch', {
                        {'-', 'Color', f('o')}, {'-', 'Color', f('c')}
                        {'-', 'Color', 'r'},    {'-', 'Color', 'b'}
                        }
                    'y_sign_ch', ones(2,2)
                    'y_scale_kind', 'rel_global'
                    };
            otherwise
                warning('Unknown src=%s\n', src);
        end
    end
    function plotArgs = get_style_by_src(~, src)
        plotArgs = {'LineWidth', 2};
    end
end
%% Main
methods
    function Pl = PlotRtDistribAll2D(varargin)
        % Pl = PlotRtDistribAll2D(cPdf, plArgs, plotArgs)
        %
        % cPdf(t, cond1, cond2, ch1, ch2) : probability mass
        % plArgs: properties of Pl
        % plotArgs: arguments of plot()
        
        if nargin >= 2
            varargin{2} = varargin2C(varargin{2}, {
                'foldAxis', [1 1]
                'accuOnlyAxis', [0 0]
                'oversample_factor', 1
                'condX_incl', ':'
                'condSep_incl', ':'
                });
        end
        
        Pl = Pl@DtbPlot.PlotRtDistrib2D(varargin{:});
    end
    function p0 = pool_p0(~, p0)
        % Skip pooling, unlike PlotRtDistrib2D.
    end
    function h_ax = plot(Pl, varargin)
        % h_ax = plot(Pl, varargin)
        if ~isempty(varargin)
            Pl.plotArgs = bml.plot.varargin2plot(varargin, Pl.plotArgs);
        end
        
        y = Pl.get_y;
        n_col = size(y, 2);
        n_row = size(y, 3);
        
        h_ax = Pl.get_ax(n_row, n_col);
        
        % On unfolded axes, ch=2 points up.
        % On folded axes, correct choice points up.
        n_chCol = size(y, 4);
        n_chRow = size(y, 5);
        
        for row = 1:n_row
            for col = 1:n_col
                ax1 = h_ax(row, col);
                
                if ~isempty(Pl.yAxisLocation)
                    if ~verLessThan('matlab', 'R2016a')
                        yyaxis(ax1, Pl.yAxisLocation);
                    end
                end
                
                condCol = col;
                condRow = n_row + 1 - row;
                
                switch Pl.y_scale_kind
                    case 'rel'
                        y_max = permute( ...
                            max(abs(y(:, condCol, condRow, :, :)), [], 1), ...
                            [4, 5, 1, 2, 3]);
                        y_max = y_max ./ max(y_max(:));
                        
                    case 'rel_global'
                        y_max = max(abs(y(:)));
                end
                
                ch_incl_row = 1:n_chRow;
                ch_incl_row = ch_incl_row(Pl.ch_incl{1});
                ch_incl_col = 1:n_chCol;
                ch_incl_col = ch_incl_col(Pl.ch_incl{2});
                
                for chRow = ch_incl_row(:)'
                    for chCol = ch_incl_col(:)'
                        plotArgs = bml.plot.varargin2plot( ...
                            Pl.styles_ch{chCol, chRow}, ...
                            Pl.plotArgs);

                        y0 = y(:, condCol, condRow, chCol, chRow);
                        y_sign = Pl.y_sign_ch(chCol, chRow);
                        switch Pl.y_scale_kind
                            case 'abs'
                                y_scale = Pl.y_scale_factor;
                                y1 = y0;
                                
                            case 'rel'
                                y_scale = y_max(chCol, chRow) ...
                                    .* Pl.y_scale_factor;
                                y1 = y0 ./ max(abs(y0));
                                
                            case 'rel_global'
                                y_scale = Pl.y_scale_factor;
                                y1 = y0 ./ y_max;
                        end
                        y1 = y1 .* y_sign .* y_scale;
                        
                        % Truncate the last element to remove the
                        % sign rule part that is just for checking the sum.
                        plot(ax1, ...
                            Pl.x, ... (1:(end-1)), ...
                            y1, ... (1:(end-1)), ...
                            plotArgs{:});
                        
                        hold(ax1, 'on');
                    end
                end
                
                xlim(ax1, [0, Pl.t(end)]);
                ylim(ax1, [-1, 1]);
                hold(ax1, Pl.to_hold);
                
                bml.plot.beautify(ax1);
                bml.plot.crossLine(ax1, 'h', 0, {'-', 0.7 + [0 0 0]});
                
                if row < n_row || col > 1
                    set(ax1, 'XTickLabel', [], 'YTickLabel', []);
                end
            end
        end
    end
    function ax = get_ax(Pl, n_row, n_col)
        if nargin == 1 || (n_row == 1 && n_col == 1)
            ax = Pl.get_ax@DtbPlot.PlotPdf2D;
        else
            if isequal(size(Pl.ax_), [n_row, n_col]) ...
                    && all(isvalidhandle(Pl.ax_(:)))
                ax = Pl.ax_;
            else
                ax = subplotRCs(n_row, n_col);
                Pl.ax_ = ax; % cache
            end
        end
    end
end
end