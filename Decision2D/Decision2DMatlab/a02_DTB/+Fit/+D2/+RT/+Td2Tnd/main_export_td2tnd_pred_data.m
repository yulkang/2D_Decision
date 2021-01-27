function main_export_td2tnd_pred_data(varargin)

clear;
init_path;

%%
for to_use_easiest_only = [0]
    for subj_parad = Data.Consts.subj_parad_td2tnd'
        [subj, parad] = deal(subj_parad{:});

        for td_kind1 = {'Ser', 'Par'}
            td_kind = td_kind1{1};

            W = Fit.D2.RT.Td2Tnd.Main;
            C = varargin2C({
                'subj', subj
                'parad', parad
                'td_kind', td_kind
%                 'to_exclude_bins_wo_trials', 10
                'to_use_easiest_only', -to_use_easiest_only
                'to_use_easiest_only_for_fit', to_use_easiest_only
                'to_use_easiest_only_for_comparison', -to_use_easiest_only
%                 'fix_miss', true
%                 'ub_mu_margin', 2
%                 'ub_sd_margin', 2
%                 'max_sd_mu_ratio', nan
                }, varargin);
            W.init(C{:});

            %%
            file0 = W.get_file;
            file = [file0, '.mat'];
            L = load(file);
            fprintf('Loaded %s\n', file);

            %%
            Fl = L.Fl;
            Fl.res2W;

            %%
            W = Fl.W;
            pred = W.Data.RT_pred_pdf;
            data = W.Data.RT_data_pdf;
            t = W.t;
            conds = W.Data.conds;
            cond_ch_to_incl_train = W.get_cond_ch_to_include_train;
            cond_ch_to_incl_test = W.get_cond_ch_to_include_valid;
            W.to_use_easiest_only = 0;
            cond_ch_to_incl = W.get_cond_ch_to_incl;

            %%
            L2 = packStruct(subj, parad, td_kind, pred, data, t, conds, ...
                cond_ch_to_incl, ...
                cond_ch_to_incl_train, cond_ch_to_incl_test);
            [pth, nam] = fileparts(file);
            file_out = fullfile(pth, ['export=pred_data+', nam, '.mat']);
            save(file_out, '-v7', '-struct', 'L2');
            fprintf('Exported pred and data to %s\n', file_out);
        end    
    end
end
