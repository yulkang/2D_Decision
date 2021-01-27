function run_calc_fine_and_plot(do_recalc_fine)
% function run_calc_fine_and_plot(do_recalc_fine)
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

%%
do_fig_per_subject = 0;

addpath('../model_fit_RT_models/')
addpath(genpath('../matlab_files/'));

dat = load('../data/RT_task/data_RT.mat','RT','RT1','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group','color_responded_first');

filt = dat.dataset==2 & dat.bimanual==1;

%%

files = {'fits','fits_fix'}; %
% files = {'fits'}; %

%do_recalc_fine = 1;
if do_recalc_fine
    
    for k=1:length(files)
        %% all unique combs
        
        combs = unique([dat.dataset(filt),dat.group(filt),dat.bimanual(filt)],'rows');
        
        %%
        
        load(files{k},'v_theta');
        
        %%
        
        for i=1:size(combs,1)
            
            idx = dat.dataset==combs(i,1) & dat.group==combs(i,2) & dat.bimanual==combs(i,3);
            uni_coh_color = nanunique(dat.coh_color(idx));
            uni_coh_motion = nanunique(dat.coh_motion(idx));
            
            
            %%
            
            filename = ['fit_serial_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3))];
            
            load(fullfile('..','model_fit_RT_models','full_dist',filename),'Pmotion','Pcolor');
            load(fullfile('..','model_fit_RT_models','finer_coh',filename),'Pmotion_fine','Pcolor_fine','coh_color_fine','coh_motion_fine');
            
            theta = v_theta(i,:);
            ndt_m = theta(3);
            
            prior = ones(size(uni_coh_motion));
            prior([length(prior)+1]/2) = 2; % zero has higher prior
            prior = prior/sum(prior);
            
            Nsamples = 10000;
            [mean_dec_time_motion, mean_dec_time_color,winner] = calc_Mean_Dec_Time_with_switches_sim(Pmotion_fine, Pcolor, theta(1),theta(2),Nsamples);
            y_mm(i,:,k) = mean_dec_time_motion*prior + ndt_m;
            y_cm(i,:,k) = mean_dec_time_color*prior + ndt_m;
            
            [mean_dec_time_motion, mean_dec_time_color,winner] = calc_Mean_Dec_Time_with_switches_sim(Pmotion, Pcolor_fine, theta(1),theta(2),Nsamples);
            y_mc(i,:,k) = mean_dec_time_motion'*prior + ndt_m;
            y_cc(i,:,k) = mean_dec_time_color'*prior + ndt_m;
            
            
            
        end
        
    end
    save fine y_mm y_cm y_mc y_cc coh_color_fine coh_motion_fine
else
    load fine
end


%% plot, fig for paper
p = publish_plot(2,2);
set(gcf,'Position',[286  227  574  446])
% set(gcf,'Position',[286  227  650  446]);
p.displace_ax([1,2],-0.07,2);
p.displace_ax([1,3],0.07,1);
ha(1) = annotation('textbox',[0,0.15,0.1,0.2],'string','Color reported first');
ha(2) = annotation('textbox',[0,0.55,0.1,0.2],'string','Motion reported first');


set(ha,'linestyle','none','fontsize',14,'horizontalalignment','center');


%
dat.corr = (dat.corr_motion | dat.coh_motion  == 0) & (dat.corr_color | dat.coh_color == 0);

%
idx = filt; % & dat.corr;
p.next();
plot(coh_motion_fine, squeeze(nanmean(y_mm)))
hold all
[tt,xx]=curva_media_hierarch(dat.RT1,dat.coh_motion,dat.group,idx & dat.color_responded_first==0,0);
terrorbar(tt,nanmean(xx,2),stderror(xx'),'color','k','linestyle','none','marker','.')
hold off
ylabel('RT first response (s)')
set(gca, 'YLim', [.5 2]);
title('A', 'FontSize', 24, 'FontWeight', 'normal'); 
set(get(gca,'title'),'unit','norm','Position',[.08 .91]);

p.next();
plot(coh_color_fine, squeeze(nanmean(y_mc)))
hold all
% [tt,xx,ss] = curva_media(dat.RT1,dat.coh_color, idx & dat.color_responded_first==0,3);
[tt,xx]=curva_media_hierarch(dat.RT1,dat.coh_color,dat.group,idx & dat.color_responded_first==0,0);
terrorbar(tt,nanmean(xx,2),stderror(xx'),'color','k','linestyle','none','marker','.')
set(gca, 'YLim', [.5 2]);
title('B', 'FontSize', 24, 'FontWeight', 'normal'); 
set(get(gca,'title'),'unit','norm','Position',[.08 .91]);
hold off

p.next();
plot(coh_motion_fine, squeeze(nanmean(y_cm)))
hold all
[tt,xx]=curva_media_hierarch(dat.RT1,dat.coh_motion,dat.group,idx & dat.color_responded_first==1,0);
terrorbar(tt,nanmean(xx,2),stderror(xx'),'color','k','linestyle','none','marker','.')
hold off
xlabel('Motion strength (coh.)')
ylabel('RT first response (s)')
set(gca, 'YLim', [.5 2]);
title('C', 'FontSize', 24, 'FontWeight', 'normal'); 
set(get(gca,'title'),'unit','norm','Position',[.08 .91]);
%legend('Single-switch model','Multi-switch model');

p.next();
plot(coh_color_fine, squeeze(nanmean(y_cc)))
hold all
% [tt,xx,ss] = curva_media(dat.RT1,dat.coh_color,idx & dat.color_responded_first==1,3);
[tt,xx]=curva_media_hierarch(dat.RT1,dat.coh_color,dat.group,idx & dat.color_responded_first==1,0);
terrorbar(tt,nanmean(xx,2),stderror(xx'),'color','k','linestyle','none','marker','.')
hold off
xlabel('Color strength (coh.)')
set(gca, 'YLim', [.5 2]);
title('D', 'FontSize', 24, 'FontWeight', 'normal'); 
set(get(gca,'title'),'unit','norm','Position',[.08 .91]);
% same_ylim(p.h_ax)
% p.format();

drawnow

p.format('FontSize',14,'LineWidthPlot',1.0,'MarkerSize',16,'LineWidthPlot',1);
p.unlabel_center_plots();


same_ylim(p.h_ax);
maxi = max(dat.coh_motion(idx));
set(p.h_ax([1,3]),'xlim',[-maxi,maxi])
maxi = max(dat.coh_color(idx));
set(p.h_ax([2,4]),'xlim',[-maxi,maxi])


% set(p.h_ax([2,4]),'ycolor','none')
% nontouching_spines(p.h_ax)

export_fig('-pdf','fig_switching_for_paper');




%% fig for paper, per suj
if do_fig_per_subject
nsuj = size(y_mm,1);
!rm fig_switching_for_paper_PER_SUJ.pdf
uni_group = unique(dat.group(filt));
f=load('fits.mat');
for isuj=1:nsuj

    p = publish_plot(2,2);
    set(gcf,'Position',[286  227  574  446])
    % set(gcf,'Position',[286  227  650  446]);
    p.displace_ax([1,2],-0.07,2);
    p.displace_ax([1,3],0.07,1);
    ha(1) = annotation('textbox',[0,0.15,0.1,0.2],'string','Color reported first');
    ha(2) = annotation('textbox',[0,0.55,0.1,0.2],'string','Motion reported first');

    set(ha,'linestyle','none','fontsize',14,'horizontalalignment','center');

    %
    idx = filt;
    p.next();
    plot(coh_motion_fine, squeeze(y_mm(isuj,:,:)))
    hold all
    [tt,xx,ss]=curva_media(dat.RT1,dat.coh_motion,dat.group==uni_group(isuj) & idx & dat.color_responded_first==0,0);
    terrorbar(tt,xx,ss,'color','k','linestyle','none','marker','.')
    hold off
    ylabel('RT first response (s)')

    p.next();
    plot(coh_color_fine, squeeze(y_mc(isuj,:,:)));
    hold all
    [tt,xx,ss]=curva_media(dat.RT1,dat.coh_color,dat.group==uni_group(isuj) & idx & dat.color_responded_first==0,0);
    terrorbar(tt,xx,ss,'color','k','linestyle','none','marker','.')
    hold off

    p.next();
    plot(coh_motion_fine, squeeze(y_cm(isuj,:,:)));
    hold all
    [tt,xx,ss] = curva_media(dat.RT1,dat.coh_motion,dat.group==uni_group(isuj) & idx & dat.color_responded_first==1,0);
    terrorbar(tt,xx,ss,'color','k','linestyle','none','marker','.')
    hold off
    xlabel('Motion strength (coh.)')
    ylabel('RT first response (s)')

    p.next();
    plot(coh_color_fine, squeeze(y_cc(isuj,:,:)))
    hold all
    % [tt,xx,ss] = curva_media(dat.RT1,dat.coh_color,idx & dat.color_responded_first==1,3);
    [tt,xx]=curva_media(dat.RT1,dat.coh_color,dat.group==uni_group(isuj) & idx & dat.color_responded_first==1,0);
    terrorbar(tt,xx,ss,'color','k','linestyle','none','marker','.')
    hold off
    xlabel('Color strength (coh.)')
    % same_ylim(p.h_ax)
    % p.format();

    drawnow

    p.format('FontSize',14,'LineWidthPlot',1.0,'MarkerSize',16,'LineWidthPlot',1);
    p.unlabel_center_plots();

    
    s = sprintf('isi [s]: %2.2f, p_{motion-1st}:%2.2f, t_{ndf}[s]:%2.2f',  ....
        f.v_theta(isuj,1)*1.5,1-f.v_theta(isuj,2),f.v_theta(isuj,3));
    
    %ha = annotate_top_of_figure('Best fit parameters:',0.07,15);
    %ha = annotate_top_of_figure(s,0.14,15);

    same_ylim(p.h_ax);
    maxi = max(dat.coh_motion(idx));
    set(p.h_ax([1,3]),'xlim',[-maxi,maxi])
    maxi = max(dat.coh_color(idx));
    set(p.h_ax([2,4]),'xlim',[-maxi,maxi])

    %export_fig('-pdf','fig_switching_for_paper_PER_SUJ','-append');


end
end

%% plot of isi's

load fits
p = publish_plot(1,1);
set(gcf,'Position',[637  269  254  253]);
isi = 1.5*v_theta(:,1);
n = size(v_theta,1);
ind = isi<6;
% color = [242,120,31]/255;
color = [0 0.4470 0.7410];
plot(isi(ind),find(ind),'o','markerfacecolor',color,'markeredgecolor','none','markersize',10)
hold all
plot(isi(~ind),find(~ind),'o','markerfacecolor',.6*[1,1,1],'markeredgecolor','none','markersize',10)

if 0
    for i=1:n
        if ind(i)==1
            text(isi(i),i,['  ',num2str(redondear(isi(i),2))],'horizontalalignment','left','color',color);
        else
            text(isi(i),i,['  ',num2str(redondear(isi(i),2))],'horizontalalignment','left','color',0.6*[1,1,1]);
        end
    end
end

ylim([0,size(v_theta,1)]);
xlim([0.1,10]);
m = mean(isi(ind));
se = stderror(isi(ind));
hold all
plot([m-se,m+se],[0,0],'LineWidth',6,'color',color);
% plot([m,m],[0,0.5],'color',color,'LineWidth',3);

set(gca,'xscale','log','xtick',[0.1,1,10],'xticklabel',{'0.1','1','10'},'XminorTick','on',...
    'ytick',1:length(ind))
%arrow([m,0.8],[m,0],'Length',10,'facecolor',color,'edgecolor',color,'linewidth',2);
% text(m-0.45,0.4,[num2str(redondear(m,2)),' s'],'color',color);
text(m-0.25,0.4,[num2str(redondear(m,2))],'color',color,'horizontalalignment','center');
xlabel('Mean time between switches (s)')
ylabel('Participant');


p.format('FontSize',14);

%%
load fits
p = publish_plot(1,1);

set(gcf,'Position',[576  274  339  340]);
isi = 1.5*v_theta(:,1);
n = size(v_theta,1);
ind = isi<6;
% color = [242,120,31]/255;
color = [0 0.4470 0.7410];
plot(find(ind),isi(ind),'o','markerfacecolor',color,'markeredgecolor','none','markersize',10)
hold all
plot(find(~ind),isi(~ind),'o','markerfacecolor',.6*[1,1,1],'markeredgecolor','none','markersize',10)


xlim([0,size(v_theta,1)]);
ylim([0.1,10]);
m = mean(isi(ind));
se = stderror(isi(ind));
hold all
plot([0,0],[m-se,m+se],'LineWidth',6,'color',color);
% plot([m,m],[0,0.5],'color',color,'LineWidth',3);

set(gca,'yscale','log','ytick',[0.1,1,10],'yticklabel',{'0.1','1','10'},'YminorTick','on',...
    'xtick',1:length(ind),'tickdir','out')
%arrow([0.8,m],[0,m],'Length',10,'facecolor',color,'edgecolor',color,'linewidth',2);
% text(m-0.45,0.4,[num2str(redondear(m,2)),' s'],'color',color);
text(0.8,m-0.15,[num2str(redondear(m,2))],'color',color,'horizontalalignment','center');

ylabel('Mean time between switches (s)')
xlabel('Participant');


p.format('FontSize',17)
p.shrink(1,0.9,0.9)
p.displace_ax(1,0.1,1)
p.displace_ax(1,0.1,2)






end
