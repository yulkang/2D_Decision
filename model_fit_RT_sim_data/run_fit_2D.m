function run_fit_2D(fit_type, groundtruth_serial_flag)

addpath('../model_fit_RT_models/');

% fit_type: 0 [pred based on highest coh], 1 [fit with all trials], 2 [ignore first ~200 trials];

if nargin==0
    fit_type = 0;% Just fit when one of the coh is highest
end

addpath(genpath('../matlab_files'));

%%

if groundtruth_serial_flag
    aux = load('simdata_RT_yul_anne_serial','RT','coh_motion','coh_color','corr_motion','corr_color',...
         'choice_color','choice_motion','bimanual','dataset','group');
else
    aux = load('simdata_RT_yul_anne_parallel','RT','coh_motion','coh_color','corr_motion','corr_color',...
        'choice_color','choice_motion','bimanual','dataset','group');
end

I = ~isnan(aux.RT);
dataset = aux.dataset(I); % otherwise, conflict with matlab 2020
RT = aux.RT(I);
coh_motion = aux.coh_motion(I);
coh_color = aux.coh_color(I);
corr_motion = aux.corr_motion(I);
corr_color = aux.corr_color(I);
choice_color = aux.choice_color(I);
choice_motion = aux.choice_motion(I);
bimanual = aux.bimanual(I);
group = aux.group(I);

%% all unique combs
combs = unique([dataset,group,bimanual],'rows');
% combs = unique([1,1,0],'rows');

% serial and parallel
nn = size(combs,1);
combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];

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

parfor i=1:size(combs,1)
    % for i=1:size(combs,1)
    
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
    
    if groundtruth_serial_flag
        filename = fullfile('from_fits_groundtruth_serial',filename);
    else
        filename = fullfile('from_fits_groundtruth_parallel',filename);
    end
    
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
    
end




end