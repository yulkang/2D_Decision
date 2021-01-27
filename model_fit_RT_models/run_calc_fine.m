function run_calc_fine(overwrite)

if nargin==0
    overwrite = 0;
end

addpath(genpath('../matlab_files'));

%%

load('../data/RT_task/data_RT','RT','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group');


%% all unique combs
combs = unique([dataset,group,bimanual],'rows');

% serial and parallel
nn = size(combs,1);
combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];

%% fit to a subset of trials

plot_flag = false;
% pars = struct('USfunc','Logistic');

file_extensions = {'','_allcoh'};
% file_extensions = {''};

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
        
        savefilename = fullfile('./finer_coh/',filename);
        
        if isfile(fullname) && (overwrite || ~isfile(savefilename))
            load(fullname)
            
            %             wrapper_2D(theta,coh_motion(idx),coh_color(idx), choice_motion(idx),choice_color(idx),RT(idx),...
            %                 corr_motion(idx),corr_color(idx),serial_flag, plot_flag);
            
            
            
            % solve with more drifts
            coh_motion_fine = linspace(-max(abs(coh_motion(idx))),max(abs(coh_motion(idx))),51);
            coh_color_fine = linspace(-max(abs(coh_color(idx))),max(abs(coh_color(idx))),51);
            u_coh_motion = unique(coh_motion(idx));
            u_coh_color = unique(coh_color(idx));
            
            coh_m = cartesian_product(coh_motion_fine,u_coh_color);
            n = size(coh_m,1);
            [nlogl,Pmotion_fine,~,E_RT_motion_correct,S_RT_motion_correct] = ...
                wrapper_2D(theta,coh_m(:,1),coh_m(:,2), nan(n,1),nan(n,1),nan(n,1),...
                nan(n,1),nan(n,1),serial_flag,  0 );
            
            coh_c = cartesian_product(u_coh_motion, coh_color_fine);
            n = size(coh_c,1);
            [nlogl,~,Pcolor_fine,E_RT_color_correct,S_RT_color_correct] = ...
                wrapper_2D(theta,coh_c(:,1),coh_c(:,2), nan(n,1),nan(n,1),nan(n,1),...
                nan(n,1),nan(n,1),serial_flag,  0 );
            
            %%
            figure();
            set(gcf, 'units','normalized', 'Position',[.1 .1 .4 .7]);
            
            subplot(2,1,1);
            
            u = unique(abs(coh_m(:,2)));
            nn = length(u);
            
            colores = cbrewer('qual','Dark2',nn);
            
            plot(coh_motion_fine,Pmotion_fine.up.p,'k')
            hold all
            [tt,xx,ss] = curva_media_hierarch(choice_motion,coh_motion,abs(coh_color),idx,0);
            for k=1:length(u)
                terrorbar(tt,xx(:,k),ss(:,k),'color',colores(k,:),'linestyle','none','marker','o','markersize',8);
                hold all
            end
            set(gca,'xlim',[min(tt),max(tt)])

            subplot(2,1,2);
            
            [tt,xx,ss] = curva_media_hierarch(E_RT_motion_correct,coh_m(:,1),abs(coh_m(:,2)),[],0);
            for k=1:length(u)
                plot(tt,xx(:,k),'color',colores(k,:));
                hold on
            end
            [tt,xx,ss] = curva_media_hierarch(RT,coh_motion,abs(coh_color),idx,0);
            for k=1:length(u)
                terrorbar(tt,xx(:,k),ss(:,k),'color',colores(k,:),'linestyle','none','marker','o','markersize',8);
                hold all
            end
            
            %         subplot(3,1,3);
            %         u = unique(abs(coh(:,2)));
            %         nn = length(u);
            %         [tt,xx,ss] = curva_media_hierarch(S_RT_correct,coh(:,1),abs(coh(:,2)),[],0);
            %         for k=1:length(u)
            %             plot(tt,xx(:,k),'color',colores(k,:));
            %             hold on
            %         end
            %
            %         for k=1:length(u)
            %             [tt,xx,ss] = curva_desvio(RT,coh_motion,I & abs(coh_color)==u(k),0);
            %             plot(tt,xx,'color',colores(k,:),'linestyle','none','marker','o','markersize',8);
            %             hold all
            %         end
            
            
            format_figure(gcf);
            set(gca,'xlim',[min(tt),max(tt)])
            %set(children,'xlim',[min(tt),max(tt)])
            drawnow
            saveas(gcf, fullfile('./finer_coh/',[filename '.pdf'])); %export_fig('-pdf',fullfile('./finer_coh/',filename));
            
            close all
            
            %% save
            p_motion_fine = Pmotion_fine.up.p;
            p_color_fine = Pcolor_fine.up.p;
            
            save(savefilename,'coh_motion_fine','coh_color_fine','p_motion_fine','coh_color_fine','p_color_fine',...
                'E_RT_motion_correct','E_RT_color_correct','u_coh_color','u_coh_motion','idx','coh_m','coh_c',...
                'S_RT_motion_correct','S_RT_color_correct','Pmotion_fine','Pcolor_fine');
            
            
        end
        
    end
end

%%
end
