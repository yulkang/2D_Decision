function simulate_td2tnd(varargin)
% Simulate data for model-free comparison

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

%% Get RT_pred_pdf from Ser and Par
S = varargin2S(varargin, {
    'subj', Data.Consts.subjs_RT(1)
    'parad', 'RT'
    'td_kind', 'Ser'
    'to_use_easiest_only', 0
    'to_exclude_bins_wo_trials', 10
    'ub_mu_margin', 2
    'ub_sd_margin', 2
    'max_sd_mu_ratio', nan
    'fix_miss', false
    'seed', 1
    });
C = S2C(S);

W = Fit.D2.RT.Td2Tnd.Main(C{:});

%%
file = [W.get_file, '.mat'];
assert(exist(file, 'file') == 2, 'File not found: %s\n', file);

L = load(file);
fprintf('Loaded fit from %s\n', file);

Fl = L.Fl;
Fl.res2W;
W = Fl.W;

RT_pred_pdf = W.Data.RT_pred_pdf;

dCond = W.Data.get_dCond;
ch = W.Data.get_ch;
t = W.t;

%% Simulate data from RT_pred_pdf by sampling
[~, ch_new, rt_vec_new] = Fit.simulate_data_given_pred( ...
    RT_pred_pdf, dCond, t, 'seed', S.seed, 'ch', ch);

%%
subj_name = sprintf('%s_td%s_seed%d_ef%d', ...
    W.Data.subj, S.td_kind, S.seed, S.to_use_easiest_only);
subj0 = W.Data.subj;
file0 = W.Data.get_path;

W.Data.path = '';
W.Data.subj = subj_name;
file = W.Data.get_path;
W.Data.subj = subj0;
W.Data.path = file0;

%%
ds00 = W.Data.ds0;
W.Data.ch = ch_new;
W.Data.rt = rt_vec_new;

L = struct;
L.dat = W.Data.ds0;
L.dat = L.dat(:, {
    'task', 'succT', 'i_all_Run', ... 'i_all_Tr', 'i_Tr', ...
    'subjM', 'subjC', 'condM', 'condC', 'corrM', 'corrC', 'RT'
    }); %#ok<STRNU>

if exist(file, 'file')
    warning('File already exists: %s\n', file);
    return;
end

save(file, '-struct', 'L');
fprintf('Saved to %s\n', file);
W.Data.ds0 = ds00;
