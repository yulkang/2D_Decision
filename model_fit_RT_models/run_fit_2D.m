function run_fit_2D()
% function run_fit_2D()
%
% 07/2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

fit_type = 1; % use for most figures (except for some of uni-/bimanual)
% 0 = just use the highest coh
% 1 = use all coherence levels

addpath(genpath('../matlab_files'));

%%
load('../data/RT_task/data_RT','RT','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group');

%% all unique combs
combs = unique([dataset,group,bimanual],'rows');

% serial and parallel
nn = size(combs,1);
combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];

% combs = combs(combs(:,1)==1,:); % just refit yul
% combs = [1,1,0,1;1,1,0,0];


%% fit to a subset of trials

% kappa, ndt_mu, ndt_sigma, B0, a, d, coh0
TL = [1,  0.1, .001 ,0.5 , -2, -2,-0.1];
TH = [40, 1, .08 ,5   , 10 ,4,0.1];
% TG = [15, 0.8, .02 ,1   , 0.5 ,0,0];

ind = [1,4,5,6,7,1,4,5,6,7,2,3];
tl = TL(ind);
th = TH(ind);
% tg = TG(ind);
tg = (tl + th)/2;

plot_flag = false;
% pars = struct('USfunc','Logistic');

isLocalComputer = 1;
if ~isLocalComputer
    parpool(18);
end

% overwrite = false; 
overwrite = true;


rng(223123,'twister');


parfor i=1:size(combs,1)
    % for i=1:size(combs,1)
    tic
    clc
    I = dataset==combs(i,1) & group==combs(i,2) & bimanual==combs(i,3) & ~isnan(RT);
    serial_flag = combs(i,4);
    
    if fit_type==0
        % like Yul's: just use the highest coh
        I = I & (abs(coh_motion)==max(abs(coh_motion(I))) | abs(coh_color)==max(abs(coh_color(I))));
        filename_ext = '';
    else
        filename_ext = '_allcoh';
    end
        
    %% filename
    if serial_flag
        str = 'fit_serial';
    else
        str = 'fit_parallel';
    end
    
    filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),filename_ext,'.mat'];
    
    %%
    
    if overwrite || ~isfile(filename)

        %     theta = tg;
        %     nlogl = wrapper_2D(theta,coh_motion(I),coh_color(I), choice_motion(I),choice_color(I),RT(I),...
        %         corr_motion(I),corr_color(I),plot_flag);


        fn_fit = @(theta) (wrapper_2D(theta,coh_motion(I),coh_color(I), choice_motion(I),choice_color(I),RT(I),...
            corr_motion(I),corr_color(I),serial_flag, plot_flag));
        % fn_fit = @(theta) (wrapper_etb_extrema(theta,rt,coh,choice,c,pars,plot_flag));

        options = optimset('Display','final','TolFun',.1,'TolX',0.02,'FunValCheck','on');% added TolX
        ptl = tl;
        pth = th;
        [theta,fval,exitflag,output] = bads(@(theta) fn_fit(theta),tg,tl,th,ptl,pth,options);
    
    
        %%
        struct_to_save = struct('theta',theta,'fval',fval);
        save_parallel(filename,struct_to_save);
        
    end
    toc
end



end
