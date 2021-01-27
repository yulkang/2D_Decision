function run_eval_best(overwrite)
% function run_eval_best(overwrite)
%
% 07/2020 Ariel Zylberberg wrote it (ariel.zylberberg@rochester.edu)

if nargin==0
    overwrite = 0;
end

addpath(genpath('../matlab_files'));

%%

load('../data/RT_task/data_RT','RT','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group');


%% all unique combs
combs = unique([dataset,group,bimanual],'rows');
% combs = unique([1,1,0],'rows');

% serial and parallel
nn = size(combs,1);
combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];

%% eval

plot_flag = false;
% pars = struct('USfunc','Logistic');

file_extensions = {'','_allcoh'};

for j=1:length(file_extensions)
    for i=1:size(combs,1)
        % for i=1:size(combs,1)
        
        idx = dataset==combs(i,1) & group==combs(i,2) & bimanual==combs(i,3) & ~isnan(RT);
        serial_flag = combs(i,4);
        
        % like Yul's
        % I = I & (abs(coh_motion)==max(abs(coh_motion(I))) | abs(coh_color)==max(abs(coh_color(I))));
        
        if serial_flag
            str = 'fit_serial';
        else
            str = 'fit_parallel';
        end
        
        filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),file_extensions{j},'.mat'];
        % filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),'_fullcoh.mat'];
        
        fullname = fullfile('./from_fits/',filename);
        
        savefilename = fullfile('./full_dist/',filename);
        
        if isfile(fullname) && (overwrite || ~isfile(savefilename))
            load(fullname)
            
            [nlogl,Pmotion,Pcolor,E_RT_correct,S_RT_correct,pPred,dist] = ...
                wrapper_2D(theta,coh_motion(idx),coh_color(idx), choice_motion(idx),choice_color(idx),RT(idx),...
                corr_motion(idx),corr_color(idx),serial_flag, plot_flag);
            

            save(savefilename,'dist','idx','pPred','Pmotion','Pcolor');
            
            % plot
            
            
            
        end
        
    end
end

%%



end




