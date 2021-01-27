classdef TdSer < Fit.D2.Bounded.Td
    % Fit.D2.Bounded.TdSer
    %
    % Given two Td_pdfs, give a merged serial Td_pdf.
    
    % 2015 YK wrote the initial version.
methods
    function W = TdSer
        W.set_Data;
    end
end
methods
    function Td_pdf = get_Td_pdf(W, Td_pdfs)
        % Td_pdf = get_Td_pdf(W, Td_pdfs)
        %
        % Td_pdfs{dim}(t, cond, ch)
        % Td_pdf(t, cond1, cond2, ch1, ch2)
        
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
                        Td_pdf(:,cond1,cond2,ch1,ch2) = ...
                            bml.math.conv_t(Td_pdfs{1}(:,cond1,ch1), ...
                                   Td_pdfs{2}(:,cond2,ch2));
                    end
                end
            end
        end
        
        % Normalize within each condition.
        % Since we observe only RT <= t_max,
        % we are effectively conditionalizing the RTs.
        % (Though it didn't make much difference with or without this.)
        Td_pdf = nan0(bsxfun(@rdivide, Td_pdf, sums(Td_pdf, [1, 4, 5])));
    end
end
end