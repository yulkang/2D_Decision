classdef DataLocator < matlab.mixin.Copyable
% Data.DataLocator
    
% 2015 YK wrote the initial version.
methods (Static)
function [files, S] = pred(varargin)
    % [files, S] = pred(...)
    % 
    % format: Data/class/subdir/model2D_parad_subjext
    %
    %     'class', 'PredMetaInh'
    %     'model2D', 'ser'
    %     'parad', 'RT'
    %     'subj', 'S1'
    %     'postfix', ''
    %     'ext', '.mat'
    S = varargin2S(varargin, {
        'class', 'PredMetaInh'
        'subdir', ''
        'model2D', 'ser'
        'parad', 'RT'
        'subj', 'S1'
        'task', 'A'
        'postfix', ''
        'ext', '.mat'
        });
    files = Data.DataLocator.factorize('../../data/%s/%s_%s_%s%s%s', ...
        fullfile(S.class, S.subdir), S.model2D, S.parad, S.subj, S.postfix, S.ext);
end
function [files, S, Ss, n] = sTr(varargin)
    % [files, S, Ss, n] = sTr(varargin)    
    % 
    % format: ../Data/subdir/parad_SubjPostfixExt
    %
    %     'subdir', ''
    %     'parad', 'RT'
    %     'subj', 'S1'
    %     'postfix', ''
    %     'ext', '.mat'
    S = varargin2S(varargin, {
        'subdir', ''
        'parad', 'RT'
        'subj', 'S1'
        'task', 'A'
        'postfix', ''
        'ext', '.mat'
        });    
    
    [Ss, n] = bml.args.factorizeS(S);
    
    files = arrayfun(@(S) fullfile('../../data/sTr', S.subdir, ...
        sprintf('%s_%s%s%s', S.parad, S.subj, S.postfix, S.ext)), ...
        Ss, 'UniformOutput', false);
end

%% Common utility
function s = factorize(fmt, varargin)
    % s = factorize(fmt, [char_or_cell_array1, ...])
    [c, n] = bml.args.factorize(varargin);
    s = cell(n, 1);
    for ii = 1:n
        s{ii} = sprintf(fmt, c{ii,:});
    end
end
end
end