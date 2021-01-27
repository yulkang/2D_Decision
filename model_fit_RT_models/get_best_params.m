function [v_theta,combs] = get_best_params(serial_flag)

%%

load('../data/RT_task/data_RT','bimanual','dataset','group');


%% all unique combs

combs = unique([dataset,group,bimanual],'rows');

% serial and parallel
nn = size(combs,1);
% combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];

%%
v_theta = [];

for i=1:size(combs,1)
    % for i=1:size(combs,1)
%     serial_flag = combs(i,4);
    
    %% filename
    if serial_flag
        str = 'fit_serial';
    else
        str = 'fit_parallel';
    end
    
    
    filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),'.mat'];
    
    
    %%
    
    fullname = fullfile('./from_fits/',filename);
    
    
    aux = load(fullname,'theta');
    %     theta = tg;
    %     nlogl = wrapper_2D(theta,coh_motion(I),coh_color(I), choice_motion(I),choice_color(I),RT(I),...
    %         corr_motion(I),corr_color(I),plot_flag);
    
    
    theta = aux.theta;
    v_theta = [v_theta;theta];
    
    
end





