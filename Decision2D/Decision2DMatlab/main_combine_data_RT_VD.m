function main_combine_data_RT_VD()
% Combine data_RT with data_VD into data_RT_VD.mat

% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

clear;
init_path;

%%
L1 = load('../../data/RT_task/data_RT.mat');
disp(L1);

L2 = load('../../data/Var_Dur/data_VD.mat');
disp(L2);

%% Combine files back into data_RT_VD.mat
L20 = struct;
n1 = size(L1.RT, 1);
for f0 = fieldnames(L2)'  % data_VD also has t_RDK_dur
    f = f0{1};
    
    if ~isfield(L1, f)
        L1.(f) = zeros(n1, size(L2.(f), 2)) + nan;
    end
    L20.(f) = [L1.(f); L2.(f)];
end
disp(L20);

%% Convert L20's fields to YK's format
L20.coh_color = sign(L20.coh_color) .* round(abs(L20.coh_color), 4);

L21 = struct;
n_incl = size(L20.RT, 1);

L21.RT = L20.RT;
L21.ch = [L20.choice_motion, L20.choice_color];
L21.cond = [L20.coh_motion, L20.coh_color];
L21.accu = [L20.corr_motion, L20.corr_color];
L21.dim_rel = true(n_incl, 2);
L21.en = zeros(n_incl, 0, 2);
L21.i_all_Run = ones(n_incl, 1);
L21.n_dim_task = zeros(n_incl, 1) + 2;
L21.RTs(L20.color_responded_first ~= true, :) = [
    L20.RT1(L20.color_responded_first ~= true, 1), ... % motion RT is shorter
    L20.RT(L20.color_responded_first ~= true, 1) % color RT is longer
    ];
L21.RTs(L20.color_responded_first == true, :) = [
    L20.RT(L20.color_responded_first == true, 1), ... % motion RT is longer
    L20.RT1(L20.color_responded_first == true, 1) % color RT is shorter
    ];
L21.bimanual = L20.bimanual;
L21.t_RDK_dur = L20.t_RDK_dur;
L21.task = repmat('A', [n_incl, 1]);
L21.to_excl = false(n_incl, 1);
L21.id_subj = L20.group;
L21.id_parad = L20.dataset;
L21.subjs = csprintf('S%d', unique(L20.group));
L21.parads = {'RT', 'unibimanual', 'VD'};

file_out = '../../data/orig/data_RT_VD.mat';
if ~exist(fileparts(file_out), 'dir')
    mkdir(fileparts(file_out));
end
save(file_out, '-v7', '-struct', 'L21');
fprintf('Saved to %s\n', file_out);

disp(L21);

%% Separate each subj & parad for RT-only fits
ds = struct2dataset(rmfield(L21, {'subjs', 'parads'}));
incl = ismember(ds.id_parad, [ ...
    find(strcmp(L21.parads, 'RT')), ...
    find(strcmp(L21.parads, 'unibimanual'))]);
ds1 = ds(incl, :);

ds1.succT = ~ds1.to_excl;
ds1.ch = ds1.ch + 1;
ds1.subjM = ds1.ch(:, 1);
ds1.subjC = ds1.ch(:, 2);
ds1.condM = ds1.cond(:, 1);
ds1.condC = ds1.cond(:, 2);
ds1.corrM = ds1.accu(:, 1);
ds1.corrC = ds1.accu(:, 2);
ds1.RT = ds1.RT;

%%
pth_out = '../../data/sTr';
if ~exist(pth_out, 'dir')
    mkdir(pth_out);
end

[id_subj_parads, ~, ix] = unique([ds1.id_subj, ds1.id_parad], 'rows');
tabulate(ix);

%%
for ii = unique(ix)'
    incl1 = ix == ii;
    dat1 = ds1(incl1, :);
    id_subj_parad = id_subj_parads(ii, :);
    subj = L21.subjs{id_subj_parad(1)};
    parad = L21.parads{id_subj_parad(2)};    
    
    assert(all(id_subj_parad(1) == dat1.id_subj));
    assert(all(id_subj_parad(2) == dat1.id_parad));
    parads_manual = {'unimanual', 'bimanual'};
    if strcmp(parad, 'unibimanual')
        for jj = 0:1
            dat = dat1(dat1.bimanual == jj, :);
            parad1 = parads_manual{jj + 1};
            
            file = fullfile(pth_out, sprintf('%s_%s.mat', parad1, subj));
            save(file, '-v7', 'dat');
            fprintf('Saved to %s\n', file);
        end
    else
        dat = dat1;
        file = fullfile(pth_out, sprintf('%s_%s.mat', parad, subj));
        save(file, '-v7', 'dat');
        fprintf('Saved to %s\n', file);
    end
end