classdef TdPar < Fit.D2.Bounded.Td
    % Fit.D2.Bounded.TdPar
    %
    % Given two Td_pdfs, give a merged parallel Td_pdf.
    
    % 2015 YK wrote the initial version.
methods
    function W = TdPar
        W.set_Data;
    end
end
methods (Static)
    function [Td_pdf, Td_last_pdf] = get_Td_pdf(Td_pdfs)
        % [Td_pdf, Td_last_pdf] = get_Td_pdf(W, Td_pdfs)
        %
        % Td_pdfs{dim}(t, cond, ch)
        % Td_pdf(t, cond1, cond2, ch1, ch2)
        % Td_last_pdf(t, dim, cond1, cond2, ch1, ch2)
        
        assert(iscell(Td_pdfs) && numel(Td_pdfs) == 2);
        assert(size(Td_pdfs{1},1) == size(Td_pdfs{2},1));
        assert(size(Td_pdfs{1},3) == 2 && size(Td_pdfs{2},3) == 2);
        assert(ndims(Td_pdfs{1}) == 3 && ndims(Td_pdfs{2}) == 3);

        W.td_pdfs = Td_pdfs; % Cache

        %% Td_pdf
        for i_dim = 2:-1:1
            n_cond(i_dim) = size(Td_pdfs{i_dim}, 2);
        end
        for cond1 = n_cond(1):-1:1
            for cond2 = n_cond(2):-1:1
                for ch1 = 2:-1:1
                    for ch2 = 2:-1:1      
                        p1 = Td_pdfs{1}(:,cond1,ch1);
                        p2 = Td_pdfs{2}(:,cond2,ch2);
                        cum_p1 = cumsum(p1);
                        cum_p2 = cumsum(p2);
                        p_max = p1 .* cum_p2 + p2 .* cum_p1 - p1 .* p2;
                        
%                         [p_max, p_last] = ...
%                            bml.stat.max_distrib([
%                                Td_pdfs{1}(:,cond1,ch1), ...
%                                Td_pdfs{2}(:,cond2,ch2)
%                                ]);
                               
                        Td_pdf(:,cond1,cond2,ch1,ch2) = p_max;
                        
                        if nargout >= 2
                            error('Not supported!');
                            
                            Td_last_pdf(:,:,cond1,cond2,ch1,ch2) = p_last;
                        end
                    end
                end
            end
        end
        
        min_p_Td = min(Td_pdf(:));
        max_p_Td = max(Td_pdf(:));
        
        if min_p_Td < 0 || max_p_Td > 1
            error('min_p_Td < 0 || max_p_Td > 1');
        end

        %% Normalize within each condition.
        % Since we observe only RT <= t_max,
        % we are effectively conditionalizing the RTs.
        % (Though it didn't make much difference with or without this.)
        Td_pdf = nan0(bsxfun(@rdivide, Td_pdf, sums(Td_pdf, [1, 4, 5])));
        
        min_p_Td = min(Td_pdf(:));
        max_p_Td = max(Td_pdf(:));
        
        if min_p_Td < 0 || max_p_Td > 1
            disp(min_p_Td);
            disp(max_p_Td);
            
            keyboard;
        end        
    end
end
end