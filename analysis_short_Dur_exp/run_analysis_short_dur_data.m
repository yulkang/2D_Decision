function run_analysis_short_dur_data()
% function run_analysis_short_dur_data()
% 
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

addpath(genpath('../matlab_files/'));

%%

load ../data/Short_Dur/all.mat
m = load('../data/Short_Dur/me.mat');


%%
% convert logodds to coh_color
%pblue = exp(coh_color)./(1+exp(coh_color));
%coh_color = 2*pblue-1; % overwrite
            
%% rescale the motion and color energy to the max

% colores = cbrewer('qual','Paired',10);
% colores1 = colores([4,8,4,8],:);
% colores2 = colores([4,8,4,8],:);

colores = cbrewer('qual','Dark2',3);
colores = colores([2,1],:);
colores1 = [colores;colores];
colores2 = colores1;

p = publish_plot(2,1);
set(gcf,'Position',[541  155  316  508]);
p.next();
dt = 1/75; % is this true???

[~,xx,ss] = curva_media(m.motion_res,choice_motion,task=='A',0);
ind = ~isnan(xx(1,:));
tt = [1:size(xx,2)] * dt - dt;
% double task
[~,hl(1)] = niceBars2(tt(ind),xx(1,ind),ss(1,ind),colores1(1,:));
hold all
[~,hl(2)] = niceBars2(tt(ind),xx(2,ind),ss(2,ind),colores1(3,:));

% single task
[~,xx,ss] = curva_media(m.motion_res,choice_motion,task=='H',0);
ind = ~isnan(xx(1,:));
% tt = [1:size(xx,2)] * dt;
[~,hl(3)] = niceBars2(tt(ind),xx(1,ind),ss(1,ind),colores1(2,:));
hold all
[~,hl(4)] = niceBars2(tt(ind),xx(2,ind),ss(2,ind),colores1(4,:));


% L = legend(hl,'\leftarrow 2D','\rightarrow 2D','\leftarrow 1D','\rightarrow 1D');
% set(get(L,'title'),'String','choice & task');

set(hl,'LineWidth',1.5);
set(hl([3,4]),'LineStyle','-');

p.next();

[~,xx,ss] = curva_media(m.color_res,choice_color,task=='A',0);
ind = ~isnan(xx(1,:));
tt = [1:size(xx,2)] * dt - dt;
[~,hl(1)] = niceBars2(tt(ind),xx(1,ind),ss(1,ind),colores2(1,:));
hold all
[~,hl(2)] = niceBars2(tt(ind),xx(2,ind),ss(2,ind),colores2(3,:));
pxlim

[~,xx,ss] = curva_media(m.color_res,choice_color,task=='V',0);
ind = ~isnan(xx(1,:));
% tt = [1:size(xx,2)] * dt;
[~,hl(3)] = niceBars2(tt(ind),xx(1,ind),ss(1,ind),colores2(2,:));
hold all
[~,hl(4)] = niceBars2(tt(ind),xx(2,ind),ss(2,ind),colores2(4,:));
pxlim


set(hl,'LineWidth',1.5);
set(hl([3,4]),'LineStyle','-');

set(p.h_ax,'xlim',[0,0.35]);

p.current_ax(1);
ylabel('Excess motion (a.u.)')

p.current_ax(2);
xlabel('Time from stimulus onset (s)')
ylabel('Excess color energy (a.u.)')

p.format('FontSize',14)
% set(L,'FontSize',11,'location','northeast')
% set(L2,'FontSize',11,'location','northeast')

set(p.h_ax(1),'xticklabel',[]);
symmetric_y(p.h_ax);
drawnow
p.displace_ax(1,-0.05,2)

export_fig('-pdf', 'fig_for_paper_me','-nocrop','-m5', '-q101');

%% choice

p = publish_plot(2,2);
p.displace_ax([1,2],-.05,2);
% p = publish_plot(2,1);
% set(gcf,'Position',[541  155  316  508]);


colores = cbrewer('qual','Dark2',3);
colores = colores(1:2,:);

lw = 1;

p.next();
[tt1,xx1,ss1] = curva_media(choice_motion,coh_motion,task=='H',0);
[tt2,xx2,ss2] = curva_media(choice_motion,coh_motion,task=='A',0);
terrorbar(tt1,xx1,ss1,'color',colores(1,:),'marker','.','linestyle','none','markersize',20);
hold all
terrorbar(tt2,xx2,ss2,'color',colores(2,:),'marker','.','linestyle','none','markersize',20);

x = [linspace(-0.5,0.5,101)];
I = task=='H';
b = glmfit(coh_motion(I), [choice_motion(I)], 'binomial', 'link', 'logit');
yfit = glmval(b, x, 'logit');
plot(x,yfit,'color',colores(1,:),'linewidth',lw);

I = task=='A';
b = glmfit(coh_motion(I), [choice_motion(I)], 'binomial', 'link', 'logit');
yfit = glmval(b, x, 'logit');
plot(x,yfit,'color',colores(2,:),'linewidth',lw);
xlim([-0.6,0.6])

% xlabel('Motion coherence')
ylabel({'Proportion',' ''rightward'' choices'})
set(gca,'xticklabel','');
% set(gca,'xcolor','none');

p.next();

[tt1,xx1,ss1] = curva_media(choice_color,coh_color,task=='V',0);
[tt2,xx2,ss2] = curva_media(choice_color,coh_color,task=='A',0);
terrorbar(tt1,xx1,ss1,'color',colores(1,:),'marker','.','linestyle','none','markersize',20);
hold all
terrorbar(tt2,xx2,ss2,'color',colores(2,:),'marker','.','linestyle','none','markersize',20);



x = [linspace(-nanmax(coh_color),nanmax(coh_color),101)];
I = task=='V';
b = glmfit(coh_color(I), [choice_color(I)], 'binomial', 'link', 'logit');
yfit = glmval(b, x, 'logit');
lw = 2;
plot(x,yfit,'color',colores(1,:),'linewidth',lw);

I = task=='A';
b = glmfit(coh_color(I), [choice_color(I)], 'binomial', 'link', 'logit');
yfit = glmval(b, x, 'logit');
plot(x,yfit,'color',colores(2,:),'linewidth',lw);

% xlabel('Color coherence')
ylabel({'Proportion',' ''blue'' choices'})
set(gca,'xticklabel','');

% hl = legend('single task','double task');
hl(1) = text(0.3,0.6,'single task','color',colores(1,:));
hl(2) = text(0.3,0.45,'double task','color',colores(2,:));
% set(gca,'ycolor','none','xcolor','none');


%% RTs
p.next();
[tt1,xx1,ss1] = curva_media(RT,coh_motion,task=='H',0);
[tt2,xx2,ss2] = curva_media(RT,coh_motion,task=='A',0);
% [tt1,xx1,ss1] = curva_media(RT,coh_motion,task=='A' & abs(coh_color)==min(abs(coh_color)),0);
% [tt2,xx2,ss2] = curva_media(RT,coh_motion,task=='A' & abs(coh_color)==max(abs(coh_color)),0);
terrorbar(tt1,xx1,ss1,'color',colores(1,:),'marker','.','linestyle','-','markersize',20,'linewidth',lw);
hold all
terrorbar(tt2,xx2,ss2,'color',colores(2,:),'marker','.','linestyle','-','markersize',20,'linewidth',lw);


% xlim([-0.6,0.6])

xlabel('Motion strength (coh.)')
ylabel('Response time (s)')

p.next();

[tt1,xx1,ss1] = curva_media(RT,coh_color,task=='V',0);
[tt2,xx2,ss2] = curva_media(RT,coh_color,task=='A',0);
terrorbar(tt1,xx1,ss1,'color',colores(1,:),'marker','.','linestyle','-','markersize',20,'linewidth',lw);
hold all
terrorbar(tt2,xx2,ss2,'color',colores(2,:),'marker','.','linestyle','-','markersize',20,'linewidth',lw);

%

xlabel('Color strength (coh.)')

ylabel('Response time (s)')


p.current_ax(1);
title('Motion')

p.current_ax(2);
title('Color')

same_xlim(p.h_ax([1,3]));
set(p.h_ax([2,4]),'xlim',[-0.9,0.9],'xtick',[-0.5,0,0.5]);
set(p.h_ax,'tickdir','out');
p.format('FontSize',14,'LineWidthAxes',1);

export_fig('-pdf', 'fig_for_paper','-nocrop','-m5', '-q101');

end
