classdef PlotFuns
    % Fit.D2.Common.Plot.PlotFuns
    %
    % 2015 YK wrote the initial version.
methods (Static)
    function add_plotfun(Fl)
%         n_th_free = nnz(~Fl.W.th_fix_vec);
%         n_th_per_plot = 8;
%         for ii = 1:n_th_per_plot:n_th_free
%             ix = ii:min(ii + n_th_per_plot - 1, n_th_free);
%             
%             Fl.add_plotfun({
%                 @(Fl) @(x,v,s) Fl.optimplotx(x,v,s, 'ix', ix);
%                 @(Fl) @(x,v,s) void0( @() axis('off'))
%                 });
%         end
        
        Fl.add_plotfun({
%             @(Fl) @optimplotfval
            @(Fl) @(x,v,s) ...
                Fit.D2.Common.Plot.PlotFuns.history_sum_p_pred(Fl, x, v, s)
            @(Fl) @(x,v,s) void0( @() ...
                use(DtbPlot.PlotCh2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1)))
            @(Fl) @(x,v,s) void0( @() ...
                use(DtbPlot.PlotRt2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1)))
            @(Fl) @(x,v,s) void0( @() ...
                use(DtbPlot.PlotRt2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'y_fun', 'var')))
            @(Fl) @(x,v,s) void0( @() ...
                use(DtbPlot.PlotCh2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'dimOnX', 2)))
            @(Fl) @(x,v,s) void0( @() ...
                use(DtbPlot.PlotRt2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'dimOnX', 2)))
            @(Fl) @(x,v,s) void0( @() ...
                use(DtbPlot.PlotRt2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'dimOnX', 2, ...
                    'y_fun', 'var')))
            @(Fl) @(x,v,s) void0(@() {...
                use(DtbPlot.PlotRtDistribAll2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'condX_incl', 1, 'condSep_incl', 1))
                title(sprintf('M=%d, C=%d', 1, 1))
                    })
            @(Fl) @(x,v,s) void0(@() {...
                use(DtbPlot.PlotRtDistribAll2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'condX_incl', 1, 'condSep_incl', 0))
                title(sprintf('M=%d, C=%d', 1, 0))
                    })
            @(Fl) @(x,v,s) void0(@() {...
                use(DtbPlot.PlotRtDistribAll2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'condX_incl', 0, 'condSep_incl', 1))
                title(sprintf('M=%d, C=%d', 0, 1))
                    })
            @(Fl) @(x,v,s) void0(@() {...
                use(DtbPlot.PlotRtDistribAll2D, ...
                @(Pl) Pl.plot_W_pred_data(Fl.W, 'oversample_factor', 1, ...
                    'condX_incl', 0, 'condSep_incl', 0))
                title(sprintf('M=%d, C=%d', 0, 0))
                    })
            });        
        try
            Fl.add_plotfun({
                @(Fl) @(x,v,s) ...
                    void0(@() Fl.W.plot_rt_vs_rt( ...
                        'dim_on_x', 1, 'yfun', 'mean'))
                @(Fl) @(x,v,s) ...
                    void0(@() Fl.W.plot_rt_vs_rt( ...
                        'dim_on_x', 2, 'yfun', 'mean'))
                @(Fl) @(x,v,s) ...
                    void0(@() Fl.W.plot_rt_vs_rt( ...
                        'dim_on_x', 1, 'yfun', 'var'))
                @(Fl) @(x,v,s) ...
                    void0(@() Fl.W.plot_rt_vs_rt( ...
                        'dim_on_x', 2, 'yfun', 'var'))
                });            
        catch
        end
        
        if isprop(Fl.W, 'Dtb') && isprop(Fl.W.Dtb, 'Dtb1')
            try
                Fl.add_plotfun({
                    @(Fl) @(x,v,s) void0(@() {
                        void0(@() Fl.W.Dtb.Dtb1.Drift.plot_drift_cond_t)
                        void0(@() hold('on'))
                        void0(@() Fl.W.Dtb.Dtb1.Bound.plot)
                        void0(@() hold('off'))
                        })
                    @(Fl) @(x,v,s) ...
                        void0(@() Fl.W.Dtb.Dtb1.Drift.plot_drift_by_cond)
                    @(Fl) @(x,v,s) void0(@() {
                        void0(@() Fl.W.Dtb.Dtb2.Drift.plot_drift_cond_t)
                        void0(@() hold('on'))
                        void0(@() Fl.W.Dtb.Dtb2.Bound.plot)
                        void0(@() hold('off'))
                        })
                    @(Fl) @(x,v,s) ...
                        void0(@() Fl.W.Dtb.Dtb2.Drift.plot_drift_by_cond)
                    });
            catch
            end
        elseif isprop(Fl.W, 'Dtb') ...
                && isprop(Fl.W.Dtb, 'Bound1') ...
                && isprop(Fl.W.Dtb, 'Bound2')
            try
                Fl.add_plotfun({
                    @(Fl) @(x,v,s) void0(@() {
                        void0(@() Fl.W.Dtb.Drift1.plot_drift_cond_t)
                        void0(@() hold('on'))
                        void0(@() Fl.W.Dtb.Bound1.plot)
                        void0(@() hold('off'))
                        })
                    @(Fl) @(x,v,s) ...
                        void0(@() Fl.W.Dtb.Drift1.plot_drift_by_cond)
                    @(Fl) @(x,v,s) void0(@() {
                        void0(@() Fl.W.Dtb.Drift2.plot_drift_cond_t)
                        void0(@() hold('on'))
                        void0(@() Fl.W.Dtb.Bound2.plot)
                        void0(@() hold('off'))
                        })
                    @(Fl) @(x,v,s) ...
                        void0(@() Fl.W.Dtb.Drift2.plot_drift_by_cond)
                    });
            catch
            end
        elseif isprop(Fl.W, 'Dtb') ...
                && isprop(Fl.W.Dtb, 'Bound') ...
                && isprop(Fl.W.Dtb.Bound, 'Bound1') ...
                && isprop(Fl.W.Dtb.Bound, 'Bound2')
            try
                Fl.add_plotfun({
                    @(Fl) @(x,v,s) void0(@() {
                        void0(@() Fl.W.Dtb.Drift.Drift1.plot_drift_cond_t)
                        void0(@() hold('on'))
                        void0(@() Fl.W.Dtb.Bound.Bound1.plot)
                        void0(@() hold('off'))
                        })
                    @(Fl) @(x,v,s) ...
                        void0(@() Fl.W.Dtb.Drift.Drift1.plot_drift_by_cond)
                    @(Fl) @(x,v,s) void0(@() {
                        void0(@() Fl.W.Dtb.Drift.Drift2.plot_drift_cond_t)
                        void0(@() hold('on'))
                        void0(@() Fl.W.Dtb.Bound.Bound2.plot)
                        void0(@() hold('off'))
                        })
                    @(Fl) @(x,v,s) ...
                        void0(@() Fl.W.Dtb.Drift.Drift2.plot_drift_by_cond)
                    });
            catch
            end
        end
    end
end
%% Modules
methods (Static)
    function stop = history_sum_p_pred(Fl, x, v, s, varargin)
        if ~exist('x', 'var')
            x = [];
        end
        if isempty(v)
            v = struct;
        end
        v = varargin2S(v, {
            'iteration', nan
            });
        if ~exist('s', 'var') || isempty(s)
            s = 'user';
        end
        S = varargin2S(varargin, {
            'add_history', ismember(s, {'init', 'iter'})
            });
        
        sum_RT_pred = sums(Fl.W.Data.get_RT_pred_pdf);
        dsum_RT_pred = sum_RT_pred ...
            - prod(sizes(Fl.W.Data.get_RT_pred_pdf, ...
                         Fl.W.Data.dim_pdf.cond));
        if S.add_history
            H = Fl.History;
            H.add_history('sum_RT_pred', sum_RT_pred);
            H.add_history('dsum_RT_pred', dsum_RT_pred);
        end
        
        stop = ...
            Fit.D2.Common.Plot.PlotFuns.history_plotyy(Fl, ...
                'dsum_RT_pred', ...
                x, v, s);
            
%         xlabel(sprintf('iter: %d, sum: %1.3f, cost: %1.2f', ...
%             v.iteration, sum_history(end), Fl.cost));
    end
    function [stop, y1, y2] = history_plotyy(Fl, column_names, x, v, s)
        % [stop, y1, y2] = history_plotyy(Fl, ...
        %                  column_names = name|{name1, name2}, x, v, s)
        if ischar(column_names)
            column_names = {column_names, 'fval'};
        else
            assert(iscell(column_names) && numel(column_names) == 2);
            assert(all(cellfun(@ischar, column_names)));
        end
        
        if ~exist('x', 'var')
            x = [];
        end
        v = varargin2S(v, {
            'iteration', nan
            });
        if ~exist('s', 'var') || isempty(s)
            s = 'done';
        end        
        
        stop = false;
        
        H = Fl.History;
        iter = 1:H.n_iter;

        y1 = cell2mat2(H.history.(column_names{1})(iter));
        y2 = cell2mat2(H.history.(column_names{2})(iter));
        
        ax = plotyy(iter, y1, iter, y2, @(x,y) plot(x,y,'o-'));
        for ii = 1:2
            ylabel(ax(ii), strrep(column_names{ii}, '_', '-'));
            axis(ax(ii), 'tight');
            ticks = get(ax(ii), 'YTick');
            set(ax(ii), 'YTickLabel', csprintf('%1.5g', ticks));
        end
        
        txt = sprintf('iter: %d\n%s: %1.2e\n%s: %1.2f', ...
            v.iteration, ...
            strrep(column_names{1}(1:min(4,end)), '_', '-'), y1(end), ...
            strrep(column_names{2}(1:min(4,end)), '_', '-'), y2(end));
        
        h_txt = findobj(ax, 'Type', 'Text');
        bml.plot.text_align(txt, 'h_txt', h_txt);
%         xlabel(txt);
        grid on;
    end
    function [stop, ys] = history_w_legends(Fl, column_names, x, v, s, varargin)
        % [stop, ys] = history_w_cost(Fl, column_names, x, v, s, ...)
        assert(iscell(column_names));
        assert(~isempty(column_names));
        assert(all(cellfun(@ischar, column_names)));
        
        S = varargin2S(varargin, {
            'normalize', true
            'add_legend', []
            });
        
        stop = false;
        
        H = Fl.History;
        iter = 1:H.n_iter;
        
        if isempty(S.add_legend) && H.n_iter == 1
            S.add_legend = true;
        end
        
        n_col = numel(column_names);
        ys = cell(n_col, 1);
        for i_col = 1:n_col
            y = cell2mat2(H.history.(column_names{i_col})(iter));
        
            if S.normalize
                y_plot = bsxfun(@minus, y, min(y)) ...
                               ./ (max(y) - min(y));
            else
                y_plot = y;
            end
            
            plot(iter, y_plot, 'o-');
            hold on;
        end
        hold off;
        
%         if S.add_legend
            legend(strrep(column_names, '_', '-'), ...
                'Location', 'Best', 'box', 'off');
%         end
        xlabel('iter');
        grid on;
        
        if nargout >= 2
            ys = H.history(iter, column_names);
        end
    end
end
end