% in prep

addpath(genpath('../matlab_files'));

%%

dat = load('../data/RT_task/data_RT','RT','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group');
struct2vars(dat);

%% all unique combs
combs = unique([dataset,group,bimanual],'rows');
% combs = unique([1,1,0],'rows');

%% set seed
rng(223124,'twister');

%% fit to a subset of trials

plot_flag = false;
% pars = struct('USfunc','Logistic');
overwrite = 1;
% file_extensions = '_allcoh';
file_extensions = '';

for i_serial=0:1
    
    % prep
    f = fieldnames(dat);
    ntr = length(RT);
    for i=1:length(f)
        m.(f{i}) = nan(ntr,1);
    end
    
    for i=1:size(combs,1)
        
        disp(num2str(i));
        % for i=1:size(combs,1)
        
        idx = dataset==combs(i,1) & group==combs(i,2) & bimanual==combs(i,3) & ~isnan(RT);
        serial_flag = i_serial;
        
        % like Yul's
        % I = I & (abs(coh_motion)==max(abs(coh_motion(I))) | abs(coh_color)==max(abs(coh_color(I))));
        
        if serial_flag
            str = 'fit_serial';
        else
            str = 'fit_parallel';
        end
        
        filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),file_extensions,'.mat'];
        % filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),'_fullcoh.mat'];
        
        fullname = fullfile('../model_fit_RT_models/from_fits/',filename);
        aux = load(fullname);
        
        ndt_m = aux.theta(end-1);
        ndt_s = aux.theta(end);
        
        fullname = fullfile('../model_fit_RT_models/full_dist/',filename);
        
        %             save(savefilename,'dist','Pmotion','Pcolor','idx');
        d = load(fullname,'dist','idx','pPred','Pmotion','Pcolor');
        
        
        uni_motion = nanunique(coh_motion(idx));
        uni_color = nanunique(coh_color(idx));
        
        pdf_motion = [d.Pmotion.lo.pdf_t, d.Pmotion.up.pdf_t];
        pdf_color = [d.Pcolor.lo.pdf_t, d.Pcolor.up.pdf_t];
        tt = [d.Pmotion.t, d.Pmotion.t];
        ch = [zeros(size(d.Pmotion.t)),ones(size(d.Pmotion.t))];
        
        ind = 1:length(tt);
        
        for im=1:length(uni_motion)
            for ic = 1:length(uni_color)
                I = idx & coh_motion==uni_motion(im) & coh_color==uni_color(ic);
                ind_motion = randsample(ind,sum(I),true,pdf_motion(im,:));
                ind_color = randsample(ind,sum(I),true,pdf_color(ic,:));
                ch_motion = ch(ind_motion)';
                ch_color = ch(ind_color)';
                DT_motion = tt(ind_motion)';
                DT_color = tt(ind_color)';
                
                non_dec_t = ndt_m + randn(sum(I),1)*ndt_s;
                RT_serial = DT_motion + DT_color + non_dec_t;
                RT_parallel = max([DT_motion,DT_color],[],2) + non_dec_t;
                
                % save
                if serial_flag
                    m.RT(I) = RT_serial;
                else
                    m.RT(I) = RT_parallel;
                end
                m.choice_motion(idx & I) = ch_motion;
                m.choice_color(idx & I) = ch_color;
            end
        end
        
        m.coh_motion(idx) = coh_motion(idx);
        m.coh_color(idx) = coh_color(idx);
        m.bimanual(idx) = bimanual(idx);
        m.dataset(idx) = dataset(idx);
        m.group(idx) = group(idx);
        
        
        % plot
        
    end
    
    %accuracy
    m.corr_motion = (m.choice_motion==1 & m.coh_motion>0) | (m.choice_motion==0 & m.coh_motion<0);
    m.corr_motion(m.coh_motion==0) = rand(sum(m.coh_motion==0),1)>0.5;
    
    m.corr_color = (m.choice_color==1 & m.coh_color>0) | (m.choice_color==0 & m.coh_color<0);
    m.corr_color(m.coh_color==0) = rand(sum(m.coh_color==0),1)>0.5;
    
    if i_serial==0
        filename = 'simdata_RT_yul_anne_parallel';
    else
        filename = 'simdata_RT_yul_anne_serial';
    end
    save(filename,'-struct','m');
    
    
    % save separatelly the serial and teh parallel
end

%%








