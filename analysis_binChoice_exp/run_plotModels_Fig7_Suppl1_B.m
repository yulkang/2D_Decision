%% plot mean RTs from data + serial/parallel gamma model + DDM
% PREDICTIONS FROM GAMMA MODEL ARE CREATED IN Fig7_Suppl_A.m
% PREDICTIONS FROM DDM ARE CREATED IN Fig7.m

clear all
close all
clc


IDs = [7, 12]; %{'S07', 'S12'};

% load data
load('../data/RT_task/data_RT_binChoice.mat')
% DATA VARIABLES
% D.coh1 = sCoh1 (signed coherence of stimulus 1 = on left side)
% D.coh2 = sCoh2 (signed coherence of stimulus 2 = on right side)
% D.rt = RT in sec (only correct trials, all others = nan)
% D.choice = choice (0/1 for down/up and different/same, respectively)
% D.correct = 0/1 = error/correct
% D.cohCond = categorical variable indicating coherence condition (1-3 = 1D; 5-10 = 2D)
% D.numstim = number of stimuli (1 vs. 2)

uCoh = unique(abs(data.coh1)); % get unique unsigned coherence levels


% load RT gamma model predictions
load(['results_RTmodel.mat']);


%% plot results
set(0,'DefaultAxesBox', 'off',...
    'DefaultAxesFontSize',20,...
    'DefaultFigureUnits', 'normalized', ...
    'DefaultFigurePosition', [0.1, 0.1, .75, .75]);

LineWidth = 2;
MarkerSize = 20;


for subj = 1:length(IDs)
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % create new data structure D
    % include only correct 2D trials from a given participant
    trialIDs = data.subjID == IDs(subj) & data.numstim == 2 & data.correct;
    Variables = fieldnames(data);
    for i = 1:numel(Variables)
        Field = Variables{i};
        D.(Field) = data.(Field)(trialIDs);
    end
    
    
    % get mean for each condition from data and DDM model
    % only get 2D trial conditions (cond >= 5)
    RT_mean = []; RT_se = [];
    for cond = 5:max(D.cohCond)
        trialIDs = D.cohCond == cond;
        
        % get RTs
        RT_mean = [RT_mean; nanmean(D.rt(trialIDs))];
        RT_se = [RT_se; nanstd(D.rt(trialIDs))/sqrt(sum(trialIDs))];
    end
    
    % get mean RTs from serial/parallel gamma model
    RTser = []; RTpar = [];
    for c1 = 1:length(uCoh)
        for c2 = 1:length(uCoh)
            if c1 <= c2
                RTser = [RTser; RTsum_2D_means{subj}(c1,c2)];
                RTpar = [RTpar; RTmax_2D_means{subj}(c1,c2)];
            end
        end
    end
    
    
    %% plot RTs
    figure(1); subplot(1,2,subj);
    axis square; hold all;
    title({['S' num2str(IDs(subj))],''},'FontSize',24,'FontWeight', 'bold');
    
    % plot data
    h1 = bar(1:length(RT_mean),RT_mean,0.7,'FaceColor',[1 1 1],'EdgeColor',[0 0 0],'LineWidth',0.5);
    
    % plot models
    for i = 1:length(RT_mean)
        
        % serial
        h2 = plot([i-.34, i+.349],[RTser(i)/1000, RTser(i)/1000],'-','LineWidth',2.5,'Color','r');
        
        % parallel
        h3 = plot([i-.34, i+.349],[RTpar(i)/1000, RTpar(i)/1000],'-','LineWidth',2.5,'Color','b');
        
    end
    
    % plot errorbar for data on top
    errorbar(1:length(RT_mean),RT_mean,RT_se,'LineStyle', 'None','Color',[0 0 0],'CapSize',0,'LineWidth',0.5);
    
    
    % define axes and labels
    xlabel({'','Coherence condition'}); ylabel('RT (s)');
    set(gca,'Xlim', [0 length(RT_mean)+1], 'XTick', [1:length(RT_mean)], 'XTickLabel',{'LL', 'LM','LH','MM','MH','HH'},'Ylim', [.3 2.3],'YTick', [.5:.5:2],'tickdir', 'out');
    legend([h1 h2 h3], {'Data', 'Serial', 'Parallel'}, 'Location', 'NorthEast', 'box','off','Interpreter','latex');

end




