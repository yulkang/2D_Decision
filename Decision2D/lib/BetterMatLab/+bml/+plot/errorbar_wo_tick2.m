function [h, hxe, hye] = errorbar_wo_tick2(x, y, xl, xu, yl, yu, plot_args, tick_args, varargin)
% [h, hxe, hye] = errorbar_wo_tick2(x, y, xle, xue, yle, yue, plot_args, tick_args, ...)
% [h, hxe, hye] = errorbar_wo_tick2(x, y, xe, [], ye, [], plot_args, tick_args, ...)
% : Errorbars along both x and y axes.
%
% [h, hxe] = errorbar_wo_tick2(x, y, xe, [], [], [], plot_args, tick_args, ...)
% : Errorbar along x-axis only.
%
% See also: bml.plot.errorbar_wo_tick

% 2015-2016 Yul Kang. hk2699 at columbia dot edu.

if nargin < 4 || isempty(xu), xu = xl; xl = -xu; end
if nargin < 5 || isempty(yl), yl = 0; end
if nargin < 6 || isempty(yu), yu = yl; yl = -yu; end
if nargin < 7, plot_args = {}; end
if nargin < 8, tick_args = {}; end

plot_args = varargin2S(varargin2plot(plot_args, {
    'Marker', 'o'
    'MarkerSize', 8
    'LineStyle', 'none'
    'LineWidth', 2
    'Color', 'k'
    'MarkerEdgeColor', 'w'
    }));

S = varargin2S(varargin, {
    'ax', gca
    });

if ~isfield(plot_args, 'MarkerFaceColor')
    plot_args.MarkerFaceColor = plot_args.Color;
end
plot_args = varargin2C(plot_args);

tick_args = varargin2plot(tick_args, ...
    varargin2C(rmfield(varargin2S(plot_args), {
            'Marker', 'LineWidth', 'LineStyle'}), {
        'Marker', 'none'
        'LineStyle', '-'
        'LineWidth', 0.5
        'Color', 'k'
        }));
    
h = plot(S.ax, x, y, plot_args{:});
hold on;

hxe = plot(S.ax, ...
    [x(:) - abs(xl(:)), x(:) + xu(:)]', ...
    [y(:), y(:)]', ...
    tick_args{:});
hold on;

hye = plot(S.ax, ...
    [x(:), x(:)]', ...
    [y(:) - abs(yl(:)), y(:) + yu(:)]', ...
    tick_args{:});
hold off;