% 2021 Yul Kang. hk2699 at caa dot columbia dot edu.

used_fields = {
    'RT', ... 
    'i_all_Run', ...
    't_RDK_dur', 'task', ...
    ...
    'condM', 'condC', 'subjM', 'subjC', ...
    'corrM', 'corrC', 'succT'
    };

for subj = {'S1', 'S2', 'S3'}
    file = Data.DataLocator.sTr('parad', 'RT', 'subj', subj{1});
    file = file{1};
    L = load(file);
    L.dat = L.dat(:, used_fields);
    save(file, '-struct', 'L');
    fprintf('Updated %s\n', file);
end