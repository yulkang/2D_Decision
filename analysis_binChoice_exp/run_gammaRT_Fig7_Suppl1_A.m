%% fit RTs - gamma distribution for each (abs) coherence level
clear
close all
clc


IDs = [7, 12];

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
uCoh = unique(abs(data.coh1)); % unsigned coherences
cohStr = {'L','M','H'}; % strings for low/medium/high coherence levels


%% fit model for each subject
for subj = 1:length(IDs)
    
    
    % create new data structure D
    % include only correct 2D trials from a given participant
    trialIDs = data.subjID == IDs(subj) & data.numstim == 2 & data.correct;
    Variables = fieldnames(data);
    for i = 1:numel(Variables)
        Field = Variables{i};
        D.(Field) = data.(Field)(trialIDs);
    end
    
    % get RTs in ms
    D.rt = D.rt*1000;

        
    %% MODEL OPTIMIZATION
    opts = optimoptions('fmincon','GradObj','off');
    opts.TolX = 1.e-4;
        
    % fit a/b parameters for 3 gamma distributions - 1 for each abs coherence
    a = [500 300 200]; % a parameter for gamma distribution (governing mean)
    b = [300 240 200].^2/1000; % b parameter for gamma distribution (governing sd)
    bnd_a = [50 1500]; % lower/upper bound for a
    bnd_b = [20 1000].^2/1000; % lower/upper bound for b
    
    % additional gamma distribution for non-decision time
    tnd_a = 500;
    tnd_b = 100.^2/1000;
    tnd_bnd_a = [20 1000]; %lower/upper bound for tnd gamma parameter a
    tnd_bnd_b = [20 1000].^2/1000; %lower/upper bound for tnd gamma parameter b
    
    % initialize all parameters and their bounds
    fitParams = [a'; b'; tnd_a; tnd_b];
    LB = [bnd_a(1) bnd_a(1) bnd_a(1) bnd_b(1) bnd_b(1) bnd_b(1) tnd_bnd_a(1) tnd_bnd_b(1)]';
    UB = [bnd_a(2) bnd_a(2) bnd_a(2) bnd_b(2) bnd_b(2) bnd_b(2) tnd_bnd_a(2) tnd_bnd_b(2)]';
    

    %%%%%%%%%%%%%%%%%
    % fit RTsum model
    [xMin_sum(:,subj),fval_sum(subj),~] = fmincon(@(fitParams) model2D_RT(fitParams, D, 'sum'), fitParams, [], [], [], [], LB, UB,[], opts);
    % run optimized sum model
    [~, RTsum_2D_means{subj},RTsum_distr{subj}] = model2D_RT(xMin_sum(:,subj), D, 'sum');
    
    %%%%%%%%%%%%%%%%%
    % fit RTmax model
    [xMin_max(:,subj),fval_max(subj),~] = fmincon(@(fitParams) model2D_RT(fitParams, D, 'max'), fitParams, [], [], [], [], LB, UB,[], opts);
    % run optimized max model
    [~, RTmax_2D_means{subj},RTmax_distr{subj}]=model2D_RT(xMin_max(:,subj), D, 'max');

    
    % compute BICs and BF
    BIC_sum(subj) = 2*fval_sum(subj)+length(fitParams)*log(length(D.subjID));
    BIC_max(subj) = 2*fval_max(subj)+length(fitParams)*log(length(D.subjID));
    % save BF for each subject
    BF(subj) = exp((BIC_max(subj)-BIC_sum(subj))/2);
    
    
    %% plot results   
    set(0,'DefaultAxesBox', 'off',...
        'DefaultAxesFontSize',18,...
        'DefaultFigureUnits', 'normalized', ...
        'DefaultFigurePosition', [0.1, 0.1, .3, 1]);
    
    figure(subj); hold all
    plotNum = 1;
    for c1 = 1:length(uCoh)
        for c2 = 1:length(uCoh)
            if c1 <= c2
                
                trialIDs = abs(D.coh1) == uCoh(c1) & abs(D.coh2) == uCoh(c2) & ~isnan(D.rt) & D.correct;
                
                subplot(3,2,plotNum); 
                box off; hold all
                set(gca, 'tickdir', 'out')
                
                % plot serial and parallel model cdf's
                h_serial = plot(cumsum(RTsum_distr{subj}(:,c1,c2)),'r','LineWidth',2);
                h_parallel = plot(cumsum(RTmax_distr{subj}(:,c1,c2)),'b','LineWidth',2);
                
                % plot data cdf
                [f,xi,U] = ksdensity(D.rt(trialIDs),'Bandwidth',100,'Function','cdf'); % get data cdf
                h_data = plot(xi, f, 'k:', 'LineWidth', 2);
                
                if plotNum == 1
                    title({['S' num2str(IDs(subj))],''},'FontSize',24,'FontWeight', 'bold');
                    legend([h_data h_serial h_parallel], {'Data','Serial','Parallel'},'Location','SouthEast','box','off','TextColor',[0,0,0]);
                end
                
                set(gca,'XLim',[0 4500], 'XTick', [0:1000:4000]);
                if plotNum > 4
                    set(gca, 'XTickLabel', {'0','1','2','3','4'});
                    xlabel('RT (s)');
                else
                    set(gca,'XTickLabel', []);
                end
                set(gca,'YLim',[0 1.02], 'YTick',[0:.5:1]);
                if plotNum == 1 || plotNum == 3 || plotNum == 5
                    ylabel('Cumul. prob.');
                     set(gca,'YTickLabel',[0 .5 1]);
                else
                    set(gca,'YTickLabel',[]);
                end

                % add coherence labels in each subplot
                xlim=get(gca,'XLim'); ylim=get(gca,'YLim');
                lab = text(.01*xlim(2),ylim(1)+0.95,[cohStr{c1} cohStr{c2}], 'FontSize',18,'FontWeight','bold');
                
                plotNum = plotNum+1;
                
            end
            
        end
    end    

end

% save mean predicted RTs from gamma model for plotting purposes 
% (see Fig7_Suppl1_B.m)
% also save xMin_sum and xMin_max = best-fitting parameters for each subject
save('results_RTmodel.mat','xMin_sum','xMin_max','RTsum_2D_means','RTmax_2D_means');
    
% show log10 BF (positive values are support for serial model)
logBF = log10(BF)


%% Function running RT model
function [nlogl,RT_mean,RT_distr] = model2D_RT(fitParams, D, model)
%MODEL2D_RT 
% For each coherence level, fit RT distribution with gamma function 
% Either sum or max rule

% fitParams 
% 1-3 = a1-a3
% 4-6 = b1-b3
% 7-8 = a and b for tnd distribution 

% get 3 coherence levels
coh = unique(abs(D.coh1));

% gamma b parameters * 1000
fitParams(4:6)=fitParams(4:6)*1000;
fitParams(8)=fitParams(8)*1000;

% transform into k/theta parameters of gamma distributions
k=(fitParams(1:3).^2)./fitParams(4:6);
theta=fitParams(4:6)./fitParams(1:3);

k_tnd = fitParams(7)^2/fitParams(8);
theta_tnd=fitParams(8)/fitParams(7);


t=1:5000; % time points

%% create gamma distribution for each coherence level, with 2 parameters (a & b) each
for c = 1:length(coh)
    g(c,:) = gampdf(t,k(c),theta(c));
    gcdf(c,:) = gamcdf(t,k(c),theta(c));
end

% tnd distribution
tnd_distr = gampdf(t,k_tnd,theta_tnd);
tnd_distr = tnd_distr/sum(tnd_distr); % normalize
tnd_mean = (1:length(tnd_distr))*tnd_distr';


%% get distribution for each coh1 x coh2 combination under sum vs. max rule
for c1 = 1:length(coh)
    for c2 = 1:length(coh)
        % sum rule = convolution of 2 pdfs
        RTsum(:,c1,c2) = conv(conv(g(c1,:),g(c2,:)),tnd_distr)';
        RTsum(:,c1,c2)=RTsum(:,c1,c2)/sum(RTsum(:,c1,c2)); % normalize
        RTsum_mean(c1,c2)=(1:length(RTsum(:,c1,c2)))*RTsum(:,c1,c2);
        
        % max rule = diff of pointwise product of cdfs
        RTmax(:,c1,c2) = conv(diff(gcdf(c1,:).*gcdf(c2,:)),tnd_distr)';
        RTmax(:,c1,c2)=RTmax(:,c1,c2)/sum(RTmax(:,c1,c2)); % normalize
        RTmax_mean(c1,c2)=(1:length(RTmax(:,c1,c2)))*RTmax(:,c1,c2);
    end
end

% save RT distribution under sum/max model as output
if contains(model,'sum')
    RT_distr = RTsum;
elseif contains(model,'max')
    RT_distr = RTmax;
end


%% get log likelihood for RT of each trial under sum vs. max rule
for tr = 1:length(D.rt)
    
    % find 2 coherence levels of given trial
    coh1 = find(abs(D.coh1(tr)) == coh);
    coh2 = find(abs(D.coh2(tr)) == coh);
    
    negloglik_sum(tr) = -1*log(RTsum(round(D.rt(tr)),coh1,coh2));
    negloglik_max(tr) = -1*log(RTmax(round(D.rt(tr)),coh1,coh2));
    
end

% sum of log likelihood
if contains(model, 'sum')
    nlogl = sum(negloglik_sum);
    RT_mean=RTsum_mean;
elseif contains(model, 'max')
    nlogl = sum(negloglik_max);
    RT_mean=RTmax_mean;
end

end