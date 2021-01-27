clear
init_path;

%%
file_model_comp = 'model_comparison_delta_logl_predictions.mat';
files_model_recovery = {
    'model_comparison_delta_logl_predictions_groundtruth_serial.mat'
    'model_comparison_delta_logl_predictions_groundtruth_parallel.mat'
};

%%
L = load('../../data/Data_2D_Py/a0_dtb/RT_dtb_simple/orig_mat/model_comparison_delta_logl_predictions.mat');
save('../../data/Data_2D_Py/a0_dtb/RT_dtb_simple/orig_mat/model_comparison_delta_logl_predictions.mat', '-v7', '-struct', 'L');

%%
models = {'serial', 'parallel'};
readme = [
    'positive values are support for the parallel model. ' ...
    'One datapoint per subject.'];

for i_model_sim = 1:2
    model_sim = models{i_model_sim};
    delta_logl_predictions = struct;
    
    for i_model_fit = 1:2
        model_fit = models{i_model_fit};
        for subj = 1:3
            delta_logl_predictions.yuls_exp(subj, i_model_fit) = ...
                get_logl(model_sim, model_fit, subj, 0);
        end
        for subj = 6:13
            delta_logl_predictions.annes_mono(subj - 5, i_model_fit) = ...
                get_logl(model_sim, model_fit, subj, 0);
            
            delta_logl_predictions.annes_bi(subj - 5, i_model_fit) = ...
                get_logl(model_sim, model_fit, subj, 1);
        end
    end
    
    for f = {'yuls_exp', 'annes_mono', 'annes_bi'}
        delta_logl_predictions.(f{1}) = ...
            delta_logl_predictions.(f{1})(:, 2) ...
            - delta_logl_predictions.(f{1})(:, 1);
        
        fprintf('ground truth: %s, delta_log_predictions.(%s):\n', ...
            model_sim, f{1});
        disp(delta_logl_predictions.(f{1}));
    end
    
    file_out = fullfile( ...
        '../../data/DDM_recovery_2D', ...
        sprintf( ...
            'model_comparison_delta_logl_predictions_groundtruth_%s', ...
            model_sim ...
        ));
    save(file_out, '-v7', 'delta_logl_predictions', 'readme');
    fprintf('Saved to %s\n', file_out);
end


function logl = get_logl(model_sim, model_fit, subj, bi)
    fname = @(model_sim, model_fit, subj, bi) sprintf( ...
            ['../../data/model_fit_RT_sim_data/' ...
             'like_not_pred_groundtruth_%s/fit_%s_d%d_s%d_b%d.mat'], ...
            model_sim, model_fit, (subj > 3) + 1, subj, bi);
    file_in = fname(model_sim, model_fit, subj, bi);
    L = load(file_in);
    fprintf('Loaded %s\n', file_in);
    logl = L.logl_pred;
end
