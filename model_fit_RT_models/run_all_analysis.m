function run_all_analysis()
% function run_all_analysis()
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

addpath(genpath('../matlab_files'))

if 0
    % run_prep_data(); % preps the data
    run_fit_2D(); % run this in cluster with runcode.sh; then manually move files to folder 'from_fits'
end

overwrite = 1;
run_eval_best(overwrite); % evals best fits and saves more outputs to 'full_dist' folder
run_calc_fine(overwrite); % calculates finer model predictions
redo_calc = 1;
run_calc_like_not_pred(redo_calc); % calcs and plots likelihood comparison

% figures for paper
% dataset (1/2) | bimanual? (0/1) | use unimanual fits? (0,1)
run_fig2(1,0,0); % eye RT
run_fig2(2,0,0); % unimanual RT
run_fig2(2,1,1); % bimanual RT - using unimanual fits
run_fig2(2,1,0); % bimanual RT - using bimanual fits

run_fig2_per_suj(); 

end