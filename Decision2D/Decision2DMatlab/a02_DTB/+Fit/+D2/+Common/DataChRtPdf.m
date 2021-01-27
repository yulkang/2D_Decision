classdef DataChRtPdf < Fit.Common.DataChRtPdf
    % Fit.D2.Common.DataChRtPdf
    
    % 2015 YK wrote the initial version.
methods
    function Dat = DataChRtPdf(varargin)
        % See also: Fit.D2.Common.DataChRtPdf
        Dat = Dat@Fit.Common.DataChRtPdf(varargin{:});

        Dat.n_dim_task = 2;
        Dat.set_fun_bias_cond;
    end
    function set_ds0(Dat, ds0)
        % Omit ds0 to set a default dataset.
        if ~exist('ds0', 'var')
            ds0 = struct2dataset(varargin2S({
                'condM', zeros(0,1)
                'condC', zeros(0,1)
                'subjM', zeros(0,1)
                'subjC', zeros(0,1)
                'corrM', zeros(0,1)
                'corrC', zeros(0,1)
                'RT', zeros(0,1)
                'task', char(zeros(0,1))
                }));
        end
        Dat.set_ds0@Fit.Common.DataChRtPdf(ds0);
    end
end
%% Fields from selected rows
methods
    function v = get_cond(Dat)
        v = [Dat.get_ds('condM'), Dat.get_ds('condC')];
    end
    function v = get_ch(Dat)
        v = [Dat.get_ds('subjM'), Dat.get_ds('subjC')];
    end
    function set_cond(Dat, v)
        Dat.set_ds('condM', v(:,1));
        Dat.set_ds('condC', v(:,2));
    end
    function v = set_ch(Dat, v)
        Dat.set_ds('subjM', v(:,1));
        Dat.set_ds('subjC', v(:,2));
    end
    function answer = get_answer(Dat)
        answer = [Dat.ds.corrM, Dat.ds.corrC];

        fun_bias_cond = Dat.get_fun_bias_cond;
        if ~isempty(fun_bias_cond)
            cond = Dat.get_cond;
            for i_dim = 1:2
                if isempty(fun_bias_cond{i_dim}), continue; end

                cond(:, i_dim) = fun_bias_cond{i_dim}(cond(:, i_dim));
                answer(cond(:, i_dim) > 0, i_dim) = 2;
                answer(cond(:, i_dim) < 0, i_dim) = 1;
                % When cond == 0, answer is left as given by Dat.ds.corrX
            end
        end
    end
    function v = get_RT(Dat)
        v = Dat.get_ds('RT');
    end
end
%% File
methods
    function task = get_default_task(~)
        task = Data.Consts.tasks{2,1};
    end
end
%% Bias
methods
    function set_fun_bias_cond(Dat, v, dim)
        if nargin < 2
            v = {[], []};
        else
            if nargin < 3
                dim = [1 2];
            end
            if ~iscell(v)
                v = {v};
            end
            v0 = Dat.get_fun_bias_cond;
            v0(dim) = v;
            v = v0;
        end
        Dat.set_fun_bias_cond@Fit.Common.DataChRtPdf(v);
    end
    function v = get_fun_bias_cond(Dat, dim)
        v = Dat.get_fun_bias_cond@Fit.Common.DataChRtPdf;
        if isempty(v)
            v = {[], []};
        end
        if nargin < 2
            dim = [1 2];
        end
        assert(isnumeric(dim) && ~isempty(dim) && numel(dim) <= 2);
        assert(all(dim <= 2) && all(dim >= 1));
        if isscalar(dim)
            v = v{dim};
        else
            v = v(dim);
        end
    end
end
%% Fields from all rows
methods
    function v = get_cond0(Dat)
        v = [Dat.ds0.condM, Dat.ds0.condC];
    end
    function v = get_dCond0(Dat)
        v = Dat.get_cond0;
        for dim = 1:size(v, 2)
            [~,~,v(:, dim)] = unique(v(:, dim));
        end
    end
    function v = get_ch0(Dat)
        v = [Dat.ds0.subjM, Dat.ds0.subjC];
    end
    function v = get_answer0(Dat)
        v = [Dat.ds0.corrM, Dat.ds0.corrC];
    end
    function v = get_RT0(Dat)
        v = Dat.ds0.RT;
    end
end
%% rel
methods
    function v = get_accu_rel(Dat)
        v = Dat.get_accu;
        v = v(:,Dat.get_dim_rel);
    end
    function v = get_cond_rel(Dat)
        v = Dat.get_cond;
        v = v(:,Dat.get_dim_rel);
    end
    function v = get_conds_rel(Dat)
        assert(isscalar(Dat.get_dim_rel));
        v = Dat.get_conds;
        v = v{Dat.get_dim_rel};
    end
    function v = get_cond_irr(Dat)
        v = Dat.get_cond;
        v = v(:,Dat.get_dim_irr);
    end
    function v = get_adCond_rel(Dat)
        v = Dat.get_adCond;
        v = v(:,Dat.get_dim_rel);
    end
    function v = get_adCond_irr(Dat)
        v = Dat.get_adCond;
        v = v(:,Dat.get_dim_irr);
    end
    function v = get_nadCond_irr(Dat)
        v = Dat.get_adCond;
        v = max(v);
    end
    function v = get_ch_rel(Dat)
        v = Dat.get_ch;
        v = v(:,Dat.get_dim_rel);
    end
    function siz = get_size_RT_Td_pdf(Dat)
        siz = [Dat.Time.nt, Dat.get_nConds, 2, 2];
    end
end
%% Dimension indices
methods
    function v = get_dim_irr(Dat)
        v = 3 - Dat.get_dim_rel;
    end
    function S = get_dim_pdf(~)
        S = varargin2S({
            't', 1
            'cond', [2 3]
            'cond1', 2
            'cond2', 3
            'ch', [4 5]
            'ch1', 4
            'ch2', 5
            });
    end
    function S = get_dim_pdf_rel(~)
        S = varargin2S({
            't', 1
            'cond', [2 3]
            'cond_rel', 2
            'cond_irr', 3
            'ch', [4 5]
            'ch_rel', 4
            'ch_irr', 5
            });
    end
end
%% Demo
methods (Static)
    function Dat = demo 
        Dat = Fit.D2.Common.DataChRtPdf;
        Dat.set_path({}, 'A');
        Dat.load_data;
        disp(Dat);
    end     
end
%% Fold to get 1D representation
methods (Static)
    function p = fold_pdf_rel_accu(p)
        % Instantiate so that we can use dependent variables
        Dat = Fit.D2.Common.DataChRtPdf;
        n_cond_rel = size(p, 2);
        conds_rel1 = 1:floor(n_cond_rel / 2);
        % Flip ch_rel so that it becomes accu_rel.
        p(:,conds_rel1,:,:,:) = flip(p(:,conds_rel1,:,:,:), ...
            Dat.dim_pdf_rel.ch_rel);
        p = Dat.fold_pdf(p, Dat.dim_pdf_rel.cond_rel);
    end
    function p = fold_pdf_irr(p)
        % Assumes that 3th & 5th dims are irr_cond and irr_ch.
        import Fit.D2.Common.DataChRtPdf
        p = DataChRtPdf.fold_pdf(p, 3);
        p = DataChRtPdf.fold_pdf(p, 5);
    end
    function p = fold_pdf_rel(p)
        % Assumes that 2nd & 4th dims are rel_cond and rel_ch.
        import Fit.D2.Common.DataChRtPdf
        p = DataChRtPdf.fold_pdf(p, 2);
        p = DataChRtPdf.fold_pdf(p, 4);
    end
    function p = fold_pdf(p, d)
        % Folds p on d-th dimension, such that the center comes first.
        % p = fold_pdf(p, d)
        %
        % Example:
        % magic(3)
        % ans =
        %      8     1     6
        %      3     5     7
        %      4     9     2
        % 
        % W.Data.fold_pdf(magic(3),1)
        % ans =
        %      3     5     7
        %      6     5     4
     
        n = size(p, d);
        n_half = ceil(n / 2);
        nd = ndims(p);
        c0 = repmat({':'}, [1, nd]);
        
        for ii1 = 1:n_half
            ii2 = n + 1 - ii1;
            c1 = c0;
            c2 = c0;
            c1{d} = ii1;
            c2{d} = ii2;
            p(c1{:}) = (p(c1{:}) + p(c2{:})) ./ 2;
        end
        
        c_res = c0;
        c_res{d} = n_half:-1:1; % So that the center comes first.
        p = p(c_res{:});
    end
end
%% Bin conditions
methods
    function pdf_rel = bin_pdf_irr(pdf_rel, bins)
        import Fit.D2.Common.DataChRtPdf

        pdf_rel = DataChRtPdf.bin_pdf(pdf_rel, 3, bins);
        if to_fold_ch_irr
            pdf_rel = DataChRtPdf.bin_pdf(pdf_rel, 5, [1 1]);
        end
    end
    function dst = bin_pdf(src, dim, bins)
        % dst = bin_pdf(src, dim, bins)
        % bins: natural numbers of bin numbers.
        assert(length(bins) == size(src, dim));
        n_bin = max(bins);
        nd = ndims(src);
        for i_bin = n_bin:-1:1
            sub = repmat({':'}, [1, nd]);
            sub_src = sub;
            sub_src{dim} = find(bins == i_bin);
            sub_res{dim} = i_bin;
            dst(sub_res{:}) = sum(src(sub_src{:}), dim);
        end
    end
end
end