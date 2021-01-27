classdef Consts    
    % Common constants.
    
    % 2021 Yul Kang. hk2699 at caa dot columbia dot edu.
  
properties (Constant)
    subj_parad_td2tnd = {
        'S1', 'RT'
        'S2', 'RT'
        'S3', 'RT'
        'S6', 'unimanual'
        'S7', 'unimanual'
        'S8', 'unimanual'
        'S9', 'unimanual'
        'S10', 'unimanual'
        'S11', 'unimanual'
        'S12', 'unimanual'
        'S13', 'unimanual'
        'S6', 'bimanual'
        'S7', 'bimanual'
        'S8', 'bimanual'
        'S9', 'bimanual'
        'S10', 'bimanual'
        'S11', 'bimanual'
        'S12', 'bimanual'
        'S13', 'bimanual'
    }
    subj_parad_RT = {
        {'S1', 'RT'}
        {'S2', 'RT'}
        {'S3', 'RT'}
        }';
    subjs_RT = {'S1', 'S2', 'S3'};
    
    n_tr_initial_skip = 0;
    
    tasks = {'H', 'V'; 'A', 'A'};
    dimNames = {'M', 'C'};    
    dimNames_long = {'Motion', 'Color'};
    n_dim = 2;
    
    data_root = '../../data';
    
    %%
    dt_frame = 1/75;
end
methods (Static)
    function subjs = get_subjs_parad(parad)
        subjs = Data.Consts.subj_parad_td2tnd( ...
            strcmp(Data.Consts.subj_parad_td2tnd(:, 2), parad) ...
            , 1);
    end
end
end