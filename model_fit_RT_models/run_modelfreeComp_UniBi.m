%% Suppl Fig 8 - Model-free comparison of uni- vs. bimanual task
clear
close all
clc

addpath(genpath('../matlab_files'));
load('../data/RT_task/data_RT.mat');

IDs = [6:13];

trials = dataset == 2; % select hand data only


% get absolute coherence levels
uMotCoh = uniquetol(abs(coh_motion(trials))); % unsigned motion coherence levels
uColCoh = uniquetol(abs(coh_color(trials))); % unsigned color coherence levels
sMotCoh = unique(coh_motion(trials));
sColCoh = unique(coh_color(trials));

% group into weak vs. strong coherence
Motion_grouped = [uMotCoh(1:3) uMotCoh(4:6)];
Color_grouped = [uColCoh(1:3) uColCoh(4:6)];
    

% Only include correct trials (or 0% coherence) in RT analyses
correct = (corr_motion | coh_motion == 0) & (corr_color | coh_color == 0);


%% get results for each individual participant
for i = 1:length(IDs)
    
    subjID = IDs(i);
    
    %% get results from uni-/bimanual task
    for k = 1:2
        if k == 1 % unimanual
            trialIdx = trials & group == IDs(i) & bimanual == 0 & ~isnan(RT); % ignore miss trials
        elseif k == 2 % bimanual
            trialIdx = trials & group == IDs(i) & bimanual == 1 & ~isnan(RT) & RT1 ~= RT; % ignore miss trials
        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %--------------------- Choice Performance ------------------------%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MOTION: % "right" choices       
        % fit psychometric curve for each color coherence level
        for c = 1:length(uColCoh)
            trialIdx_choice = trialIdx & ismember(abs(coh_color), uColCoh(c));
            MotionChoice(c,:,i,k) = glmfit(coh_motion(trialIdx_choice),[choice_motion(trialIdx_choice) == 1 ones(sum(trialIdx_choice),1)],'binomial','logit');            
        end
        
        % COLOR: % "blue" choices
        % fit psychometric curve for each motion coherence level
        for m = 1:length(uMotCoh)
            trialIdx_choice = trialIdx & ismember(abs(coh_motion), uMotCoh(m));
            ColorChoice(m,:,i,k) = glmfit(coh_color(trialIdx_choice),[choice_color(trialIdx_choice) == 1 ones(sum(trialIdx_choice),1)],'binomial','logit');            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %----------------------------- RTs -------------------------------%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MOTION: RTs for each motion x color coherence
        for m = 1:length(uMotCoh)
            for c = 1:size(Color_grouped,2)
                trialIdx_rt = trialIdx & correct & abs(coh_motion) == uMotCoh(m) & ismember(abs(coh_color),Color_grouped(:,c));
                RTmotion(i,m,c,k) = mean(RT(trialIdx_rt));
            end
        end
        
        % COLOR: RTs for each motion x color coherence
        for c = 1:length(uColCoh)
            for m = 1:size(Motion_grouped,2)
                trialIdx_rt = trialIdx & correct & abs(coh_color) == uColCoh(c) & ismember(abs(coh_motion),Motion_grouped(:,m));
                RTcolor(i,c,m,k) = mean(RT(trialIdx_rt));
            end
        end
        
    end
end

%% set up figure properties
set(0,'DefaultAxesBox', 'off',...
    'DefaultAxesFontSize',16,...
    'DefaultFigureUnits', 'normalized', ...
    'DefaultFigurePosition', [0.1, 0.1, 0.65, 0.95]);

color_uni = [0, 0.4470, 0.7410];
color_bi = [0.6350, 0.0780, 0.1840];

MarkerSize = 10;


%% Plot results
% choice sensitivity as function of current x other dimension
figure(1);
subplot(2,2,1); axis square; 

hold all
h1 = errorbar(1:length(uMotCoh), mean(ColorChoice(:,2,:,1),3),std(ColorChoice(:,2,:,1),[],3)/sqrt(size(ColorChoice,3)),'o','Color',color_uni,'LineWidth',2,'MarkerFaceColor',color_uni, 'MarkerSize', MarkerSize); % unimanual
h2 = errorbar(1:length(uMotCoh), mean(ColorChoice(:,2,:,2),3),std(ColorChoice(:,2,:,2),[],3)/sqrt(size(ColorChoice,3)),'o','Color',color_bi,'LineWidth',2,'MarkerFaceColor',color_bi, 'MarkerSize', MarkerSize); % bimanual
% placeholder plots for legends
h3 = plot(-5,-5,'o','Color',color_uni, 'MarkerFaceColor',color_uni,'LineWidth',2, 'MarkerSize', MarkerSize);
h4 = plot(-5,-5,'o','Color',color_bi, 'MarkerFaceColor',color_bi,'LineWidth',2, 'MarkerSize', MarkerSize);
set(gca,'Xlim', [-.05 length(uMotCoh)+1], 'XTick', 1:length(uMotCoh), 'XTickLabels', {},'Ylim', [7 17],'TickDir','out');
ylab = ylabel('Color sensitivity (\beta)'); 
leg = legend([h3 h4], 'Unimanual','Bimanual', 'Location', 'SouthWest', 'box','off');
title(leg,'Task version', 'FontSize',16);
%xlim=get(gca,'XLim'); ylim=get(gca,'YLim');
%text(xlim(1)+(30*xlim(1)),ylim(2)+ylim(2)/25,'A', 'FontSize',24,'FontWeight','bold');

subplot(2,2,2); axis square; hold all
h1 = errorbar(1:length(uColCoh), mean(MotionChoice(:,2,:,1),3),std(MotionChoice(:,2,:,1),[],3)/sqrt(size(MotionChoice,3)),'o','Color',color_uni,'LineWidth',2, 'MarkerFaceColor',color_uni, 'MarkerSize', MarkerSize); % unimanual
h2 = errorbar(1:length(uColCoh), mean(MotionChoice(:,2,:,2),3),std(MotionChoice(:,2,:,2),[],3)/sqrt(size(MotionChoice,3)),'o','Color',color_bi,'LineWidth',2, 'MarkerFaceColor',color_bi, 'MarkerSize', MarkerSize); % bimanual
set(gca,'Xlim', [-.05 length(uColCoh)+1], 'XTick', 1:length(uColCoh), 'XTickLabels', {},'Ylim', [12.5 37.5],'TickDir','out');
ylab = ylabel('Motion sensitivity (\beta)'); 
%xlim=get(gca,'XLim'); ylim=get(gca,'YLim');
%text(xlim(1)+(30*xlim(1)),ylim(2)+ylim(2)/25,'B', 'FontSize',24,'FontWeight','bold');
pos = get(gca,'Position');
set(gca,'Position',[pos(1)-.03 pos(2:4)]);


% RTs for current & other dimension
subplot(2,2,3); axis square; hold all
for c = 1:size(Color_grouped,2)
    if c == 1
    errorbar(1:length(uMotCoh), mean(RTmotion(:,:,c,1),1),std(RTmotion(:,:,c,1),[],1)/sqrt(size(RTmotion,1)),'o','Color',color_uni,'LineWidth',2, 'MarkerFaceColor',[1 1 1], 'MarkerSize', MarkerSize); % unimanual
    errorbar(1:length(uMotCoh), mean(RTmotion(:,:,c,2),1),std(RTmotion(:,:,c,2),[],1)/sqrt(size(RTmotion,1)),'o','Color',color_bi,'LineWidth',2, 'MarkerFaceColor',[1 1 1], 'MarkerSize', MarkerSize); % bimanual
    else
    errorbar(1:length(uMotCoh), mean(RTmotion(:,:,c,1),1),std(RTmotion(:,:,c,1),[],1)/sqrt(size(RTmotion,1)),'o','Color',color_uni,'LineWidth',2, 'MarkerFaceColor',color_uni, 'MarkerSize', MarkerSize); % unimanual
    errorbar(1:length(uMotCoh), mean(RTmotion(:,:,c,2),1),std(RTmotion(:,:,c,2),[],1)/sqrt(size(RTmotion,1)),'o','Color',color_bi,'LineWidth',2, 'MarkerFaceColor',color_bi, 'MarkerSize', MarkerSize); % bimanual    
    end
end
set(gca,'Xlim', [-.05 length(uMotCoh)+1], 'XTick', 1:length(uMotCoh), 'XTickLabels', round(100*uMotCoh)/100,'XTickLabelRotation',45, 'Ylim', [.4 2.6], 'YTick', .5:.5:2.5,'TickDir','out');
xlab = xlabel('Motion strength (|coh|)');
ylab = ylabel('RT (s)'); 
% placeholder plots for legends
h5 = plot(-5,-5,'o','Color',[0.1 0.1 0.1], 'MarkerFaceColor',[1 1 1],'LineWidth',2, 'MarkerSize', MarkerSize);
h6 = plot(-5,-5,'o','Color',[0.1 0.1 0.1], 'MarkerFaceColor',[0.1 0.1 0.1],'LineWidth',2, 'MarkerSize', MarkerSize);
pos = get(gca,'Position');
set(gca,'Position',[pos(1) pos(2)+.08 pos(3:4)]);

leg = legend([h5 h6], 'Weak','Strong', 'Location', 'SouthWest','box','off');
title(leg,'             Other dimension','FontSize',16);
legpos = get(leg, 'Position');
set(leg, 'Position', [legpos(1)-.04 legpos(2:4)]);
subplot(2,2,4); axis square; hold all
for m = 1:size(Motion_grouped,2)
    if m == 1
    h1 = errorbar(1:length(uColCoh), mean(RTcolor(:,:,m,1),1),std(RTcolor(:,:,m,1),[],1)/sqrt(size(RTcolor,1)),'o','Color',color_uni,'LineWidth',2, 'MarkerFaceColor',[1 1 1], 'MarkerSize', MarkerSize); % unimanual
    h2 = errorbar(1:length(uColCoh), mean(RTcolor(:,:,m,2),1),std(RTcolor(:,:,m,2),[],1)/sqrt(size(RTcolor,1)),'o','Color',color_bi,'LineWidth',2, 'MarkerFaceColor',[1 1 1], 'MarkerSize', MarkerSize); % bimanual
    else
    h1 = errorbar(1:length(uColCoh), mean(RTcolor(:,:,m,1),1),std(RTcolor(:,:,m,1),[],1)/sqrt(size(RTcolor,1)),'o','Color',color_uni,'LineWidth',2, 'MarkerFaceColor',color_uni, 'MarkerSize', MarkerSize); % unimanual
    h2 = errorbar(1:length(uColCoh), mean(RTcolor(:,:,m,2),1),std(RTcolor(:,:,m,2),[],1)/sqrt(size(RTcolor,1)),'o','Color',color_bi,'LineWidth',2, 'MarkerFaceColor',color_bi, 'MarkerSize', MarkerSize); % bimanual    
    end
end
set(gca,'Xlim', [-.05 length(uColCoh)+1], 'XTick', 1:length(uColCoh), 'XTickLabels', round(100*uColCoh)/100,'XTickLabelRotation',45, 'Ylim', [.4 2.6], 'YTick', .5:.5:2.5,'TickDir','out');
xlab = xlabel('Color strength (|coh|)'); 
ylab = ylabel('RT (s)'); 
pos = get(gca,'Position');
set(gca,'Position',[pos(1)-.03 pos(2)+.08 pos(3:4)]);

export_fig('-pdf', 'Fig5Supp2','-nocrop','-m5', '-q101', '-transparent');

%saveas(figure(1), 'Fig5Supp2.pdf');

