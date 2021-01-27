function run_do_fit(redo_fit)
% function run_do_fit(redo_fit)
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

addpath('../model_fit_RT_models/');
datadir = '../data/RT_task/';
dat = load(fullfile(datadir,'data_RT'),'RT','RT1','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group','color_responded_first');


%% all unique combs
filt = dat.dataset==2 & dat.bimanual==1;
combs = unique([dat.dataset(filt),dat.group(filt),dat.bimanual(filt)],'rows');

fit_type = 3;
fixed_isi_flag = 0;

if fixed_isi_flag
    savefilename = 'fits_fix';
else
    savefilename = 'fits';
end


if redo_fit
    
    for i=1:size(combs,1)
        
        idx = dat.dataset==combs(i,1) & dat.group==combs(i,2) & dat.bimanual==combs(i,3);
        uni_coh_color = nanunique(dat.coh_color(idx));
        uni_coh_motion = nanunique(dat.coh_motion(idx));
        
        
        %%
        
        filename = ['fit_serial_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3))];
        load(fullfile('..','model_fit_RT_models','from_fits',filename),'theta');
        load(fullfile('..','model_fit_RT_models','full_dist',filename),'Pmotion','Pcolor');
        
        %ndt_m = theta(end-1);
        
        
        %%
        
        
        % isi, p start w color, non-dec time
        
        if fixed_isi_flag
            tl = [50, 0.05, 0.1];
            th = [50, 0.95, 0.9];
            tg = [50, 0.5,0.4];
        else
            tl = [0.05, 0.05, 0.1];
            th = [9, 0.95, 0.9];
            tg = [0.2, 0.5,0.4];
        end
        
        fn_fit = @(theta) (wrapper_eval_params_simulations(theta, Pcolor, Pmotion, dat, idx, uni_coh_motion, uni_coh_color, fit_type));
        
        %         if 1
        options = optimset('Display','final','TolFun',.1,'FunValCheck','on');
        ptl = tl;
        pth = th;
        [theta,fval,exitflag,output] = bads(@(theta) fn_fit(theta),tg,tl,th,ptl,pth,options);
        
        
        % eval best
        
        [~, y_mm(:,i), y_mc(:,i), y_cm(:,i), y_cc(:,i)] = fn_fit(theta);
        
        v_theta(i,:) = theta;
        
    end
    
    save(savefilename,'v_theta','y_mm','y_mc','y_cm','y_cc');
    
else
    load(savefilename);
end

%% eval best

for i=1:size(combs,1)
    
    idx = dat.dataset==combs(i,1) & dat.group==combs(i,2) & dat.bimanual==combs(i,3);
    uni_coh_color = nanunique(dat.coh_color(idx));
    uni_coh_motion = nanunique(dat.coh_motion(idx));
    
    %%
    filename = ['fit_serial_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3))];
    load(fullfile('..','model_fit_RT_models','from_fits',filename),'theta');
    load(fullfile('..','model_fit_RT_models','full_dist',filename),'Pmotion','Pcolor');
    
    %%
    wrapper_eval_params_simulations(v_theta(i,:), Pcolor, Pmotion, dat, idx, uni_coh_motion, uni_coh_color,fit_type);
    
end




end
