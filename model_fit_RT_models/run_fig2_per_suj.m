function run_fig2_per_suj()
% function run_fig2_per_suj()
%
% 07/2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

addpath(genpath('../matlab_files'))

%%

load('../data/RT_task/data_RT','RT','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group','SignedColorStrengthLogodds');

% plot color in log scale
% coh_color = SignedColorStrengthLogodds;
coh_color = fix(coh_color*1e6)/1e6;

% idx for correct trials (and all 0% coherence trials) for RT plots
IDX_DATA_CORR = (corr_color | coh_color == 0) & (corr_motion | coh_motion == 0);


%% all unique combs

all_combs = unique([dataset,group,bimanual],'rows');

uni_group_eye = unique(group(dataset==1));
uni_group_hand = unique(group(dataset==2));

%%

fit_type = 0;
if fit_type == 0
    savefilename = 'fig_RT_per_subject/fig_RT_PER_SUJ.pdf';
else
    savefilename = 'fig_RT_per_subject/fig_RT_PER_SUJ_allcoh.pdf';
end

show_delta_log_in_title = 0;

system(['rm ',savefilename]); % because I'll append to it

for iDataset=0:1
    plot_bimanual = iDataset;
    if iDataset==1
        always_use_monomaual_fits = 1;
    else
        always_use_monomaual_fits = 0;
    end
    
    
    
    uni_cond = unique([all_combs(:,1),all_combs(:,2)],'rows');
    for icond=1:size(uni_cond,1)
        
        uni_serial = [0,1];
        for i_serial = 1:length(uni_serial)
            
            is_serial = uni_serial(i_serial);
            
            combs = all_combs;
            % serial and parallel
            nn = size(combs,1);
            combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];
            
            I = combs(:,1)==uni_cond(icond,1) & ...     % 1: Yul; 2: Anne
                combs(:,2)==uni_cond(icond,2) & ...
                combs(:,3)==plot_bimanual & ...     % 0: monomanual; 1: bimanual
                combs(:,4)==is_serial;          % 0: parallel; 1: serial
            
            existe_y_es_unico = sum(I)==1;
            if existe_y_es_unico
                if i_serial==1
                    p = publish_plot(3,2);
                    set(gcf,'Position',[409  138  577  652]);
                    p.displace_ax([1,2,3,4],-0.06,2);
                end
                
                combs = combs(I,:);
                
                
                switch fit_type
                    case 0
                        extension = '';
                    case 1
                        extension = '_allcoh';
                end
                
                %%
                IDX = zeros(size(coh_color));
                p_motion_fine = [];
                p_color_fine = [];
                clear mRT_motion mRT_color
                
                serial_flag = combs(4);
                
                % like Yul's
                % I = I & (abs(coh_motion)==max(abs(coh_motion(I))) | abs(coh_color)==max(abs(coh_color(I))));
                
                title_string = '';
                if combs(1)==1
                    %title_string = [title_string,'Dataset: Yul, S',num2str(combs(2))];
                    suj_num = find(uni_group_eye==combs(2));
                    title_string = ['Eye-RT, S',num2str(suj_num)];
                else
                    %title_string = [title_string,'Dataset: Anne, S',num2str(combs(2))];
                    suj_num = find(uni_group_hand==combs(2))+5;
                    if combs(3)==0
                        title_string = ['Unimanual, S',num2str(suj_num)];
                    else
                        title_string = ['Bimanual, S',num2str(suj_num)];
                    end
                end
                
                if serial_flag
                    str = 'fit_serial';
                else
                    str = 'fit_parallel';
                end
                
                if always_use_monomaual_fits
                    filename = [str,'_d',num2str(combs(1)),'_s',num2str(combs(2)),'_b0',extension,'.mat'];
                else
                    filename = [str,'_d',num2str(combs(1)),'_s',num2str(combs(2)),'_b',num2str(combs(3)),extension,'.mat'];
                end
                
                %             filename = [str,'_d',num2str(combs(1)),'_s',num2str(combs(2)),'_b',num2str(combs(3)),extension,'.mat'];
                
                if (show_delta_log_in_title)
                    delta_log = 0;
                    ff = ['d',num2str(combs(1)),'_s',num2str(combs(2)),'_b',num2str(combs(3)),extension,'.mat'];
                    ff = fullfile('./delta_logl/',ff);
                    aux = load(ff,'delta_logl_for_serial');
                    title_string = {title_string, ['\Delta logl for parallel: ', num2str(-1*aux.delta_logl_for_serial)]};
                end
                
                
                fullname = fullfile('./finer_coh/',filename);
                
                d = load(fullname);
                IDX = d.idx;
                
                % convert the color scale to logodds
%                 pblue = (d.coh_color_fine+1)/2;
%                 d.coh_color_fine = log(pblue./(1-pblue));
                
                 d.coh_m = fix(d.coh_m*1e6)/1e6; % needs some rounding to make unique work - annoying
                 d.u_coh_color = fix(d.u_coh_color*1e6)/1e6;
                
                [uc,mRT_color] = curva_media_hierarch(d.E_RT_color_correct,d.coh_c(:,2),abs(d.coh_c(:,1)),[],0);
                [um,mRT_motion] = curva_media_hierarch(d.E_RT_motion_correct,d.coh_m(:,1),abs(d.coh_m(:,2)),[],0);
                
                p_motion_fine = [p_motion_fine; d.p_motion_fine'];
                p_color_fine = [p_color_fine; d.p_color_fine'];
                
                
                
                %%
                
                
                msize = 6;
                
                %             colores1 = rainbow_colors(size(mRT_color,2),'colorType',2);
                %             colores2 = rainbow_colors(size(mRT_motion,2),'colorType',3);
                addpath('../colormaps/YK/');
                n = size(mRT_color,2);
                colores1 = winter2(n);
                colores1 = colores1(end:-1:1,:);
                n = size(mRT_motion,2);
                colores2 = cool2(n);
                colores2 = colores2(end:-1:1,:);
                
                p.current_ax(1);
                if is_serial==1
                    %                 n = size(mRT_motion,2);
                    % colores = cbrewer('qual','Dark2',n);
                    % colores = colors_az(n);
                    % colores = cbrewer('seq','BuPu',n+4);
                    % colores = colores(5:end,:);
                    
                    %                 plot(d.coh_motion_fine,nanmean(p_motion_fine,1),'k');
                    plot(d.coh_motion_fine,nanmean(p_motion_fine,1),'LineStyle','-','color',colores1(end,:));
                    [tt,xx,ss] = curva_media_hierarch(choice_motion,coh_motion,abs(coh_color),IDX,0);
                    for i=1:n
                        hold all
                        handle_color(i) = plot(tt,xx(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markerfacecolor',colores1(i,:),...
                            'markersize',msize);
                    end
                    if fit_type==0 && ~always_use_monomaual_fits
                        plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                            'markeredgecolor',colores1(n,:),'markersize',msize);
                    end
                    
                else
                    plot(d.coh_motion_fine,nanmean(p_motion_fine,1),'LineStyle','--','color',colores1(end,:));
                    hold on
                end
                
                p.current_ax(2);
                if is_serial==1
                    n = size(mRT_color,2);
                    %                 plot(d.coh_color_fine,nanmean(p_color_fine,1),'k');
                    plot(d.coh_color_fine,nanmean(p_color_fine,1),'LineStyle','-','color',colores2(end,:));
                    [tt,xx,ss] = curva_media_hierarch(choice_color,coh_color,abs(coh_motion),IDX,0);
                    for i=1:n
                        hold all
                        %     terrorbar(tt,xx(:,i),ss(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor',colores2(i,:),...
                        %         'markersize',msize);
                        handle_motion(i) = plot(tt,xx(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor',colores2(i,:),...
                            'markersize',msize);
                    end
                    if fit_type==0 && ~always_use_monomaual_fits
                        plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                            'markeredgecolor',colores2(n,:),'markersize',msize);
                    end
                else
                    plot(d.coh_color_fine,nanmean(p_color_fine,1),'--','color',colores2(end,:));
                    hold on
                end
                
                if is_serial==0
                    p.current_ax(5);
                else
                    p.current_ax(3);
                end
                n = size(mRT_motion,2);
                [tt,xx,ss] = curva_media_hierarch(RT,coh_motion,abs(coh_color),IDX & IDX_DATA_CORR,0);
                for i=1:n
                    plot(d.coh_motion_fine,nanmean(mRT_motion(:,i,:),3),'color',colores1(i,:))
                    hold all
                    %     terrorbar(tt,xx(:,i),ss(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markersize',msize,...
                    %         'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
                    plot(tt,xx(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markersize',msize,...
                        'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
                end
                if fit_type==0 && ~always_use_monomaual_fits
                    %                 plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                    %                     'markeredgecolor','k','markersize',msize);
                    plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                        'markeredgecolor',colores1(i,:),'markersize',msize);
                    for i=1:size(xx,2)
                        plot(tt(1),xx(1,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                            'markeredgecolor',colores1(i,:),'markersize',msize);
                        plot(tt(end),xx(end,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                            'markeredgecolor',colores1(i,:),'markersize',msize);
                    end
                end
                
                
                p.next();
                n = size(mRT_color,2);
                [tt,xx,ss] = curva_media_hierarch(RT,coh_color,abs(coh_motion),IDX & IDX_DATA_CORR,0);
                for i=1:n
                    plot(d.coh_color_fine,nanmean(mRT_color(:,i,:),3),'color',colores2(i,:))
                    hold all
                    %     terrorbar(tt,xx(:,i),ss(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markersize',msize,...
                    %         'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
                    plot(tt,xx(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markersize',msize,...
                        'markerfacecolor',colores2(i,:),'markeredgecolor',colores2(i,:));
                end
                if fit_type==0 && ~always_use_monomaual_fits
                    %                 plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                    %                     'markeredgecolor','k','markersize',msize);
                    plot(tt,xx(:,n),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor','w',...
                        'markeredgecolor',colores2(i,:),'markersize',msize);
                    for i=1:size(xx,2)
                        plot(tt(1),xx(1,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                            'markeredgecolor',colores2(i,:),'markersize',msize);
                        plot(tt(end),xx(end,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                            'markeredgecolor',colores2(i,:),'markersize',msize);
                    end
                end
                
            end
        end
        
        if existe_y_es_unico
            
            
            if 0
                set(p.h_ax([1,3,5]),'xlim',[min(d.u_coh_motion),max(d.u_coh_motion)]);
                set(p.h_ax([2,4,6]),'xlim',[min(d.u_coh_color),max(d.u_coh_color)]);
                
                set(p.h_ax([1:4]),'xticklabel',[]);
                
                p.current_ax(1);
                %         hl(1) = legend_n(unique(abs(d.u_coh_color)),'hline',handle_color,'title','|color coh|');
                ylabel({'Proportion','putative choice'})
                
                p.current_ax(2);
                %         hl(2) = legend_n(unique(abs(d.u_coh_motion)),'hline',handle_motion,'title','|motion coh|');
                % ylabel('Proportion "blue" choices')
                
                p.current_ax(3);
                ylabel({'Response time (s)', 'SERIAL model'})
                
                p.current_ax(5);
                xlabel('Motion coherence')
                ylabel({'Response time (s)', 'PARALLEL model'})
                
                p.current_ax(6);
                % ylabel('Response time (s)')
                xlabel('Color coherence')
                
                
                same_ylim(p.h_ax([1,2]));
                same_ylim(p.h_ax([3,4]));
                same_ylim(p.h_ax([5,6]));
                
                set(p.h_ax([2,4,6]),'yticklabel',[],'ycolor','none');
                set(p.h_ax([1,2,3,4]),'xcolor','none');
                
                
                set(p.h_ax,'tickdir','out');
                
                p.format('FontSize',13,'LineWidthPlot',1,'LineWidthAxes',.5);
                %         set(hl,'FontSize',10,'Location','SouthEast','Box','off');
                
                
                nontouching_spines(p.h_ax,'ticklength',0.02);
                
            else
                
                %             set(p.h_ax([1,3,5]),'xlim',[nanmin(coh_motion),nanmax(coh_motion)]);
                %             set(p.h_ax([2,4,6]),'xlim',[nanmin(coh_color),nanmax(coh_color)]);
                set(p.h_ax([1,3,5]),'xlim',[min(d.u_coh_motion),max(d.u_coh_motion)]);
                set(p.h_ax([2,4,6]),'xlim',[min(d.u_coh_color),max(d.u_coh_color)]);
                
                % set(p.h_ax([1:4]),'xticklabel',[]);
                set(p.h_ax([3:4]),'xticklabel',[]);
                
                
                
                p.current_ax(1);
                u = unique(abs(d.u_coh_color));
                u = redondear(u,3);
                hl(1) = legend(handle_color,strjust(num2str(unique(abs(d.u_coh_color))),'left'));
                set(get(hl(1),'Title'),'String','Color strength')
                %hl(1) = legend_n(u,'hline',handle_color,'title','Color strength');
                
                p.current_ax(2);
                hl(2) = legend(handle_motion,strjust(num2str(unique(abs(d.u_coh_motion))),'left'));
                set(get(hl(2),'Title'),'String','Motion strength')
                %hl(2) = legend_n(unique(abs(d.u_coh_motion)),'hline',handle_motion,'title','Motion strength');
                % ylabel('Proportion "blue" choices')
                
                
                p.current_ax(1);
                xlabel('Motion strength (coh.)')
                ylabel('Proportion ''rightward'' choices');
                
                p.current_ax(2);
                xlabel('Color strength (coh.)')
                ylabel('Proportion ''blue'' choices');
                
                p.current_ax(3);
                ylabel({'Response time (s)', '(Serial model)'})
                
                p.current_ax(5);
                xlabel('Motion strength (coh.)')
                ylabel({'Response time (s)', '(Parallel model)'})
                
                p.current_ax(6);
                xlabel('Color strength (coh.)')
                
                
                
                %             set(p.h_ax(1:2:end),'xlim',xli1);
                %             set(p.h_ax(2:2:end),'xlim',xli2);
                
                % set(p.h_ax([2,4,6]),'yticklabel',[],'ycolor','none');
                % set(p.h_ax([1,2,3,4]),'xcolor','none');
                
                same_ylim(p.h_ax([1,2]));
                same_ylim(p.h_ax([3,4]));
                same_ylim(p.h_ax([5,6]));
                
                set(p.h_ax,'tickdir','out');
                
                p.format('FontSize',13,'LineWidthPlot',1,'LineWidthAxes',.5);
                
                set(hl,'FontSize',10,'Location','SouthEast','Box','off');
                pos = get(hl(2),'position');
                pos(1) = pos(1)+0.05;
                set(hl(2),'position',pos);
                
                set(p.h_ax,'tickdir','out');
                
            end
            %% texto
            ht=annotation('textbox',[0,0.9,1,0.1]);
            set(ht,'String',title_string,'verticalalignment','middle','horizontalalignment','center','linestyle','none','FontSize',16);
            
            %%
            if fit_type==0
                export_fig('-pdf',[savefilename '_S' num2str(suj_num)],'-append');
            else
                export_fig('-pdf',['fig_RT_per_subject/fig_RT_PER_SUJ_allcoh' '_S' num2str(suj_num) '.pdf'],'-append');
            end
            
        end
        
    end
end

end



