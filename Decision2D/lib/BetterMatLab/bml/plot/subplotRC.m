function varargout = subplotRC(nR, nC, r, c, varargin)
% SUBPLOTRC  Specifies subplot by row and column rather than an index.
%
% h = subplotRC(nR, nC, r, c)
% subplotRC(nR, nC, 'clear')
% [pos_vec, units] = subplotRC(nR, nC, r, c, 'pos_only', true)
%
% See also: SUBPLOT.
%
% 2013-2014 (c) Yul Kang. hk2699 at columbia dot edu.

S = varargin2S(varargin, { ...
    'replace_with', []
    'pos_only',     false % Give position only without leaving the axes itself.
    'align',        false % Does not remove existing axis
    });

if S.align
    C = {'align'};
else
    C = {};
end

if S.pos_only
    %% Positioning only
    h = subplotRC(nR, nC, r, c, 'align', true);
    varargout{1} = get(h, 'Position');
    varargout{2} = get(h, 'Units');
    delete(h);
else
    if ischar(r) && strcmp(r, 'clear')
        %% Clear mode
        for ii = 1:nR*nC
            subplot(nR, nC, ii, C{:});
            cla;
        end
        
    else
        %% Regular call
        % Check UserData.subplotRC to see if the same call has been made
        fig = gcf;
        u   = getappdata(fig, 'subplotRC');
        
        h   = [];
        arg = {nR, nC, r, c};
        
        % Examine previous calls
        nu = length(u);
        to_remove = false(1, nu);
        
        u_id = nan;
        
        for ii = 1:nu
            if isequal(arg, u(ii).arg)
                try
                    if isvalidhandle(u(ii).h)
                        h = u(ii).h;
                        if nargout == 0
                            axes(h); %#ok<LAXES>
                        end
                        u_id = ii;
                        break;
                    else
                        to_remove(ii) = true;
                    end
                catch err
                    warning(err_msg(err));
                end
                % Remove invalid entry
                to_remove(ii) = true;
            end
        end
        if ~isnan(u_id)
            u_id = u_id - sum(to_remove(1:u_id));
        end
        u(to_remove) = [];
        
        % Call subplot if not found already
        if isempty(h) || ~isvalidhandle(h)
            h = subplot(nR, nC, subplot_ix(nR, nC, r, c), C{:});
            
            if ~isfield(u, 'subplotRC')
                l = 0;
            else
                l = length(u);
            end
            
            u_id = l+1;
            u(u_id).arg = {nR, nC, r, c};
            u(u_id).h   = h;
        end
        
        % Replace handles if requested
        if ~isempty(S.replace_with)
            % Copy into the current figure if not already in the same figure
            if ~isequal(get(S.replace_with(1), 'Parent'), gcf)
                newh = copyobj(S.replace_with, gcf);
            else
                newh = S.replace_with;
            end
            
            % Match the size
            for ii = 1:numel(S.replace_with)
                pos = get(h(ii), 'Position');
                set(newh(ii), 'Position', pos);
            end
            
            % Replace remembered handles
            delete(h);
            u(u_id).h = newh;
            h = newh;
        end
        
        % Set userdata
        setappdata(fig, 'subplotRC', u);
        
        % Output
        if nargout > 0
            varargout = {h};
        end
    end
end
end

function ix = subplot_ix(nR, nC, r, c)
ix = nC*(r-1)+c;
end