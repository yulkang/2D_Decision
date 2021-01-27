function main()
% Runs all analyses run on MATLAB

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

clear;
init_path;

%% Convert data
main_combine_data_RT_VD;

%% -- 2D RT - Ser vs Par, model-free Figure and Table
% Real & simulated data

% Fits for Fig 2 SFigs 2-5 - takes days to run
main_fig2supp2_5;
