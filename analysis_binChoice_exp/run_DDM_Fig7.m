%% Fig. 7 - fit DDM for up/down (1D) and same/different (2D) choice-RT 
clear all
close all
clc


IDs = [7, 12]; %{'S07', 'S12'};

load('../data/RT_task/data_RT_binChoice.mat')
% DATA VARIABLES
% D.coh1 = sCoh1 (signed coherence of stimulus 1 = on left side)
% D.coh2 = sCoh2 (signed coherence of stimulus 2 = on right side)
% D.rt = RT in sec (only correct trials, all others = nan)
% D.choice = choice (0/1 for down/up and different/same, respectively)
% D.correct = 0/1 = error/correct
% D.cohCond = categorical variable indicating coherence condition (1-3 = 1D; 5-10 = 2D)
% D.numstim = number of stimuli (1 vs. 2)

% get unique coherence levels
sCoh = unique(data.coh1); % signed coherences
uCoh = unique(abs(data.coh1)); % unsigned coherences


%% fit model for each subject
for subj = 1:length(IDs)

    % create new data structure D
    % (contains only relevant data for given participant)
    trialIDs = data.subjID == IDs(subj);
    Variables = fieldnames(data);
    for i = 1:numel(Variables)
        Field = Variables{i};
        D.(Field) = data.(Field)(trialIDs);
    end
    D.incl_rt = D.correct; % include only correct trials for RT likelihood
    
    % calculate std of RT for each coherence combination for RT likelihood calculation    
    for j = 1:length(sCoh)
        for k = 1:length(sCoh)
            tr = D.incl_rt & D.coh1 == sCoh(j) & D.coh2 == sCoh(k); % select relevant trial conditions
            if length(tr)>1
                D.rt_sd(tr) = nanstd(D.rt(tr));
            end
        end
    end
    
    
    % define starting value and lower/upper bound of each parameter
    theta = [0.80 0.80 0.01 12.0 0.35 0.35]; % initial value
    theta_lo = [0 0 -1  0.1 0.0100 0.0100]; % lower bound
    theta_hi = [2 2  1 20.0 5.0079 5.0079]; % upper bound
    %-- 6 MODEL PARAMETERS: --%
    % B1=theta(1); % bound for 1D
    % B2=theta(2); % bound for 2D
    % coh0=theta(3); % coherence bias
    % kappa=theta(4); % kappa/drift rate
    % tnd1=theta(5); % non-decision time for 1D
    % tnd2=theta(6); % non-decision time for 2D
    
    
    % optimise parameters for serial & parallel model
    [theta_opt_sum{subj}, fval_sum(subj)] = fmincon(@(theta) dtb_cost_means(theta,D,'sum'),theta,[],[],[],[],theta_lo,theta_hi,[]);
    [theta_opt_max{subj}, fval_max(subj)] = fmincon(@(theta) dtb_cost_means(theta,D,'max'),theta,[],[],[],[],theta_lo,theta_hi,[]);

    % run optimised serial & parallel model
    [nlog_sum, Rfit_sum{subj}]=dtb_cost_means(theta_opt_sum{subj},D,'sum');
    [nlog_max, Rfit_max{subj}]=dtb_cost_means(theta_opt_max{subj},D,'max');
    
    % compute BICs and BF
    BIC_sum(subj) = 2*fval_sum(subj)+length(theta)*log(length(D.subjID));
    BIC_max(subj) = 2*fval_max(subj)+length(theta)*log(length(D.subjID));
    % save BF for each subject
    BF(subj) = exp((BIC_max(subj)-BIC_sum(subj))/2);
        
    
    %% plot results
    set(0,'DefaultAxesBox', 'off',...
        'DefaultAxesFontSize',14,...
        'DefaultFigureUnits', 'normalized', ...
        'DefaultFigurePosition', [0.3, 0.1, .6, .8]);
    LineWidth = 2;
    MarkerSize = 20;    

    % get mean for each condition
    for cond = 1:max(D.cohCond) % cohCond is categorical variable indicating each unique coherence combination
        tr = D.cohCond == cond;
        
        % get choices
        Corr_mean(cond) = nanmean(D.correct(tr)); % data mean
        Corr_se(cond) = sqrt(Corr_mean(cond)*(1-Corr_mean(cond))/sum(tr));  % data SE
        Corr_pred_sum(cond) = mean(Rfit_sum{subj}.pcorrect(Rfit_sum{subj}.cohCond == cond)); % serial DDM prediction (average of negative and positive choices)
        Corr_pred_max(cond) = mean(Rfit_max{subj}.pcorrect(Rfit_max{subj}.cohCond == cond)); % parallel DDM prediction (average of negative and positive choices)
        
        % get RTs
        RT_mean(cond) = nanmean(D.rt(tr & D.incl_rt)); % data mean
        RT_se(cond) = nanstd(D.rt(tr & D.incl_rt))/sqrt(sum(tr & D.incl_rt)); % data SE
        RT_pred_sum(cond) = mean(Rfit_sum{subj}.rt(Rfit_sum{subj}.cohCond == cond)); % serial DDM prediction (average of negative and positive choices)
        RT_pred_max(cond) = mean(Rfit_max{subj}.rt(Rfit_max{subj}.cohCond == cond)); % serial DDM prediction (average of negative and positive choices)
    end
    
    
    %% plot choices
    figure(1);
    subplot(2,2,subj); hold on
    title({['S' num2str(IDs(subj))],''});
    
    % plot data
    h1 = bar(1:max(D.cohCond),Corr_mean,0.7,'FaceColor',[1 1 1],'EdgeColor',[0 0 0],'LineWidth',0.5);
    
    % plot serial model prediction
    for i = 1:max(D.cohCond)
        plot([i-.34, i+.349],[Corr_pred_sum(i), Corr_pred_sum(i)],'-','LineWidth',2.5,'Color','r'); hold on
    end
    % plot serial model prediction
    for i = 1:max(D.cohCond)
        plot([i-.34, i+.349],[Corr_pred_max(i), Corr_pred_max(i)],'-','LineWidth',2.5,'Color','b'); hold on
    end
    
    % plot errorbar for data on top
    errorbar(1:max(D.cohCond),Corr_mean,Corr_se,'LineStyle', 'None','Color',[0 0 0],'CapSize',0,'LineWidth',0.5);
    
    % figure axes and labels
    set(gca,'Xlim', [0 max(D.cohCond)+1], 'XTick', unique(D.cohCond), 'XTickLabel',[],'Ylim', [.45 1.01]);  
    if subj == 1
        ylabel('P(correct)');
        set(gca,'YTick', [.5:.1:1]);
    else
        set(gca,'YTick', [], 'YTickLabel', []);
    end
    axisBreakMark(3.8,.45); % break x-axis between 1D and 2D conditions
    
    
    %% plot RTs
    subplot(2,2,2+subj); hold on
    
    % plot data
    h1 = bar(1:max(D.cohCond),RT_mean,0.7,'FaceColor',[1 1 1],'EdgeColor',[0 0 0],'LineWidth',0.5);
    
    % plot serial model
    for i = 1:max(D.cohCond)
        h2 = plot([i-.34, i+.349],[RT_pred_sum(i), RT_pred_sum(i)],'-','LineWidth',2.5,'Color','r'); hold on
    end
    % plot parallel model
    for i = 1:max(D.cohCond)
        h3 = plot([i-.34, i+.349],[RT_pred_max(i), RT_pred_max(i)],'-','LineWidth',2.5,'Color','b'); hold on
    end
    
    % plot errorbar for data on top
    errorbar(1:max(D.cohCond),RT_mean,RT_se,'LineStyle', 'None','Color',[0 0 0],'CapSize',0,'LineWidth',0.5);
    
    % figure axes and labels
    xlabel({'','Coherence condition'});
    set(gca,'Xlim', [0 max(D.cohCond)+1], 'XTick', unique(D.cohCond), 'XTickLabel', {'L','M','H','LL', 'LM','LH','MM','MH','HH'},'Ylim', [.3 2.3]);
    if subj == 1
        ylabel('RT (s)');
        set(gca,'YTick', [.5:.5:2]);
        legend([h1 h2 h3], {'Data', 'Serial', 'Parallel'}, 'Location', 'NorthEast', 'box','off');
    else
        set(gca,'YTick', [], 'YTickLabel', []);
    end
    axisBreakMark(3.8,.3);
    
    
end

%% save optimized model predictions
save(['results_DDM.mat'], 'Rfit_sum','Rfit_max','theta_opt_sum','theta_opt_max');

% show log10 BF (positive values are support for serial model)
logBF = log10(BF)


%% function to fit DDM
function [nlogl Rfit]=dtb_cost_means(theta,D,model)

B1=theta(1); % bound for 1D
B2=theta(2); % bound for 2D
coh0=theta(3); % coherence bias
kappa=theta(4); % kappa/drift rate
tnd1=theta(5); % non-decision time 1D
tnd2=theta(6); % non-decision time 2D

% calculate drift based on kappa, coherence, and coh bias
drift1 = kappa*(D.coh1 + coh0); % 1D 
drift21 = kappa*(D.coh1 + coh0); % 2D - stim 1
drift22 = kappa*(D.coh2 + coh0); % 2D - stim 2

% calculate probability of "up" judgments based on drift and bound
p1up = (1+exp(-2*drift1*B1)).^(-1); % 1D
p21up = (1+exp(-2*drift21*B2)).^(-1); % 2D - stim 1
p22up = (1+exp(-2*drift22*B2)).^(-1); % 2D - stim 2

% for 2D: calculate prob. of "same" choices based on P("Up") for each stimulus 
% (take into account double errors => "same" choice for 2 "down" judgments)
psame = (p21up.*p22up) + (1-p21up).*(1-p22up); 

% calculate predicted RTs
rt1 = tnd1 + (B1./drift1).*tanh(drift1*B1); % 1D: RT = non-decision time + decision time
dt21 = (B2./(drift21)).*tanh(drift21*B2); % 2D - decision time for stimulus 1
dt22 = (B2./(drift22)).*tanh(drift22*B2); % 2D - decision time for stimulus 2
if contains(model,'sum')
    rt2 = tnd2 + dt21 + dt22; % 2D serial decision time = non-decision time + SUM of 2 decision times
elseif contains(model,'max')
    rt2 = tnd2 + max([dt21,dt22],[],2); % 2D parallel decision time = non-decision time + MAX of 2 decision times
end

% compute log likelihoods based on actual choices and RTs
y1 = D.incl_rt.*( 0.5 * ((D.rt - rt1)./D.rt_sd).^2 + log(sqrt(2*pi) * D.rt_sd)); % 1D RTs
y1 = y1 - D.choice.*log(p1up) - (1-D.choice).*log(1-p1up); % 1D choices
y2 = D.incl_rt.*(0.5 * ((D.rt - rt2)./D.rt_sd).^2 + log(sqrt(2*pi) * D.rt_sd)); % 2D RTs
y2 = y2 - D.choice.*log(psame) -(1-D.choice).*log(1-psame); % 2D choices
% obtain negative log likelihood across all 1D and 2D trials
nlogl = nansum(y1(D.numstim == 1,:)) + nansum(y2(D.numstim == 2,:));


% save model predictions in structure Rfit
s = D.numstim == 1; % 1D trial idx

Rfit.pchoice(s,1) = p1up(s); % pred. probability of "Up" judgments in 1D task
Rfit.pchoice(~s,1) = psame(~s); % pred. probability of "Same" judgments in 2D task

Rfit.rt(s,1) = rt1(s); % predicted 1D RTs
Rfit.rt(~s,1) = rt2(~s); % predicted 2D RTs

% predicted prob. of accurate choices based on signed coh & probability of positive choice in a given trial
% 1D accuracy
w = sign(D.coh1(s));
Rfit.pcorrect(s,1) = (w>0).*Rfit.pchoice(s) + (w<0).*(1-Rfit.pchoice(s));
% 2D accuracy
w=sign(D.coh1(~s)).*sign(D.coh2(~s)); % signed "similarity" (negative = different; positive = same)
Rfit.pcorrect(~s,1)=(w>0).*Rfit.pchoice(~s) + (w<0).*(1-Rfit.pchoice(~s));

% categorical variable indicating coherence combination condition (1-3 = 1D; 5-10 = 2D)
Rfit.cohCond = D.cohCond; 


end

function hH = axisBreakMark(x,y)
% makes a crude x-axis breaker. Centered at x,y. returns handles to the H from which this is constructed.
hH = text(x,y,'H');
set(hH,'FontAngle','italic');
set(hH,'erasemode','xor');
end

