classdef PlotRtDistrib2D < DtbPlot.PlotPdf2D
    % DtbPlot.PlotRtDistrib2D

    % 2015 YK wrote the initial version.
        
%% Properties - Options
properties
    smoothTime = 0.1; % in seconds.
    yfun = 'raw'; % 'raw' or 'cumsum'
    pool_kind = 'mean'; % 'mean' or 'sum'
    
    % y_scale_kind
    % : 'abs': factor * raw_value
    % : 'rel': factor * raw_value / max(abs(raw_value))
    % : 'rel_global': factor * raw_value ...
    %                 / max(abs(raw_value_across_all_conditions))
    %
    % Note
    % : Scale only applies during plotting. Not to get_y.
    %   Use get_y_plot to get y scaled for plot.
    y_scale_kind = 'rel'; 
    y_scale_factor = 1;
    to_hold = 'off'
    yAxisLocation = ''; % ''|'left'|'right'
    
    condX_incl = 1; % numerical indices or ':'
    condSep_incl = 1; % numerical indices or ':'
    chX_incl = ':'; % numerical indices (1 and/or 2) or ':'
    chSep_incl = ':'; % numerical indices (1 and/or 2) or ':'
end
%% FitWorkspace interface
methods
    function varargout = plot_W_pred_data(Pl, W, varargin)
        C = varargin2C(varargin, {
            'oversample_factor', 1
            });
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data@DtbPlot.PlotPdf2D(W, C{:});
    end
    function varargout = plot_W(Pl, W, varargin)
        C = varargin2C(varargin, {
            'oversample_factor', 1
            });
        [varargout{1:nargout}] = ...
            Pl.plot_W@DtbPlot.PlotPdf2D(W, C{:});
    end
end
%% Main
methods
    function Pl = PlotRtDistrib2D(varargin)
        % Pl = PlotRtDistrib2D(cPdf, plArgs, plotArgs)
        %
        % cPdf(t, cond1, cond2, ch1, ch2) : probability mass
        % plArgs: properties of Pl
        % plotArgs: arguments of plot()
        %
        % plot(Pl, cond1, cond2, ch1, ch2, varargin)
        
        if nargin >= 2
            varargin{2} = varargin2C(varargin{2}, {
                'accuOnlyAxis', [0 0]
                'oversample_factor', 1
                });
        end
        
        Pl = Pl@DtbPlot.PlotPdf2D(varargin{:});
    end
    function y = get_y(Pl, condX, condSep, chX, chSep)
        % p0: (t, :, :, :, :)
        % y: yfun(p0(t, condX, condSep, chX, chSep))
        p0 = Pl.pdf_permuted;

        % Filter accu
        for axis = 1:2
            p0 = Pl.filter_p_by_accu(p0, axis);
        end

        % Fold axes
        p0 = Pl.fold_p(p0);
        
        % Group p0
        for axis = 1:2
            p0 = Pl.group_p(p0, axis);
        end

        % Leave only the relavant conditions and choices
        if ~exist('condX', 'var'), condX = Pl.condX_incl; end
        if ~exist('condSep', 'var'), condSep = Pl.condSep_incl; end
        if ~exist('chX', 'var'), chX = Pl.chX_incl; end
        if ~exist('chSep', 'var'), chSep = Pl.chSep_incl; end
        
        condX   = bml.indsub.ix2py(condX,   size(p0, 2));
        condSep = bml.indsub.ix2py(condSep, size(p0, 3));
        chX     = bml.indsub.ix2py(chX,     size(p0, 4));
        chSep   = bml.indsub.ix2py(chSep,   size(p0, 5));
        
        % Pool
        % : y(t, ...) <- (t, dimOnX, dimSep, chX, chSep)
        y = Pl.pool_p0(p0(:, condX, condSep, chX, chSep));
        
        % Apply yfun
        switch Pl.yfun
            case 'raw'
                t = Pl.t;
                dt = t(2) - t(1);
                sigma_bin = Pl.smoothTime / dt;
                y = smooth_gauss(y, sigma_bin);

            case 'cumsum'
                y = cumsum(y);

            otherwise
                error('Unknown yfun=%s!\n', Pl.yfun);
        end
    end
    function y = get_y_plot(Pl, varargin)
        y = Pl.get_y(varargin{:});
        switch Pl.y_scale_kind
            case 'abs'
                y = y .* Pl.y_scale_factor;
            case 'rel'
                y = bsxfun(@rdivide, y, max(max(max(abs(y),1),4),5)) ...
                    .* Pl.y_scale_factor;
            case 'rel_global'
                y = bsxfun(@rdivide, y, max(abs(y(:)))) ...
                    .* Pl.y_scale_factor;
        end
    end
    function x = get_x(Pl)
        x = Pl.t;
    end
    function y = pool_p0(Pl, p0)
        switch Pl.pool_kind
            case 'mean'
                y = bml.math.means(p0, 2:5);
            case 'sum'
                y = bml.math.sums(p0, 2:5);
        end
    end
    function h = plot(Pl, varargin)
        % h = plot(Pl, varargin)

        % Interpret input
        plotArgs = bml.plot.varargin2plot(Pl.plotArgs);
        
        % Get x and y
        y = Pl.get_y;
        x = Pl.get_x;

        % Plot
        h = plot(x, y, plotArgs{:});
    end
    function set_smoothTime(Pl, v)
        assert(isscalar(v));
        Pl.v = v;
    end
    function set_yfun(Pl, v)
        assert(ismember(v, {'raw', 'cumsum'}));
        Pl.yfun = v;
    end
end
methods (Static)
    function stop = outputfun(x, v, s, varargin)
        stop = false;
        Pl = DtbPlot.PlotRtDistrib2D(varargin{:});
        Pl.plot;
    end
end
end