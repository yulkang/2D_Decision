%% Simulate RT data under serial/parallel model for each participant from binary-choice task
% Resulting log10(BF) values are averaged across simulations for each subject 
% Results are shown in Fig2-Suppl3
% This takes a while to run, depending on number of simulations 
% (and results will vary slightly every time the code is run due to random drawing in simulation)...

clear
close all
clc

simN = 20; % specify number of simulations to run per participant, set to lower value to run faster
sim = 'serial'; % simulate data under 'serial' | 'parallel' model

IDs = [7, 12]; 

% load results from gamma RT model to use best-fitting parameters for each subject for simulation
load('results_RTmodel.mat');

%% load actual data set to know trial numbers per condition
% (only counting correct trials)
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



%%
for subj = 1:length(IDs)
    
    % get trial numbers per condition
    % (only includes correct trials for given participant)
    for c1 = 1:length(uCoh)
        for c2 = 1:length(uCoh)
            if c2 >= c1 % ignore order (i.e., low-high = high-low)
                trialNum(c1,c2) = sum(data.subjID == IDs(subj) & data.numstim == 2 & data.correct & abs(data.coh1)==uCoh(c1) & abs(data.coh2)==uCoh(c2));
            end
        end
    end

    
    %% run simulations
    for s = 1:simN
        
        disp(['******* Subj ' num2str(subj) ', SIMULATION #' num2str(s) ' (' sim ') *******']);
        
        % run RTsum model simulation
        if contains(sim,'serial')
            [~,simData]=model2D_RTsim(xMin_sum(:,subj), data, 'sum', 'sim', 1, 'trialNum', trialNum); % use best fitting parameters from serial model
        elseif contains(sim,'parallel')
            [~,simData]=model2D_RTsim(xMin_max(:,subj), data, 'max', 'sim', 1, 'trialNum', trialNum); % use best fitting parameters from parallel model
        end
        
        %% fit both models
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
        [xMin_sum(:,subj),fval_sum(subj),~] = fmincon(@(fitParams) model2D_RTsim(fitParams, simData, 'sum'), fitParams, [], [], [], [], LB, UB,[], opts);
%         % run optimized sum model
%         [~, RTsum_2D_means{subj},RTsum_distr{subj}] = model2D_RTsim(xMin_sum(:,subj), simData, 'sum');
        
        %%%%%%%%%%%%%%%%%
        % fit RTmax model
        [xMin_max(:,subj),fval_max(subj),~] = fmincon(@(fitParams) model2D_RTsim(fitParams, simData, 'max'), fitParams, [], [], [], [], LB, UB,[], opts);
%         % run optimized max model
%         [~, RTmax_2D_means{subj},RTmax_distr{subj}]=model2D_RTsim(xMin_max(:,subj), simData, 'max');
        
        % compute BICs and BF
        BIC_sum(subj) = 2*fval_sum(subj)+length(fitParams)*log(sum(trialNum(:)));
        BIC_max(subj) = 2*fval_max(subj)+length(fitParams)*log(sum(trialNum(:)));
        
        % save BF for each subject and each simulation
        % NEGATIVE values represent evidence for SERIAL model
        BF(subj,s) = exp((BIC_sum(subj)-BIC_max(subj))/2); 
        
    end
    
end

%% show average log10(BF) for each subject
% NEGATIVE values represent evidence for SERIAL model
disp('');disp('');
disp('***********');
for id = 1:length(IDs)
    disp(['mean log10(BF): ID' num2str(IDs(id)) ' = ' num2str(mean(log10(BF(id,:))))]);
end
disp('(negative values = support for serial)');

% save log10(BF) values as .mat file
save(['BFsim_' sim '.mat'], 'BF');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function simulating/running gamma RT model 
% (same as in run_gammaRT_Fig7_Suppl1_A, but with optional parameter to
% simulate data instead of fitting model)
function [nlogl,simData,RT_mean,RT_distr] = model2D_RTsim(fitParams, D, model, varargin)
%MODEL2D_RT
% For each coherence level, fit RT distribution with gamma function
% Either sum or max rule
% if 'sim' argument is set to 1, will simulate data instead of fitting model

% set up optional input arguments and provide default values
p = inputParser; % Create instance of inputParser class.
addParameter(p,'sim', 0, @(x) (x == 0 | x == 1)); % simulate data (1) or not (0) - default = 0
addParameter(p,'trialNum', [], @(x) (sum(x(:)) > 0)); % number of trials to run (only relevant for simulation)
parse(p,varargin{:}); % Call the parse method of the object to read and validate each argument in the schema

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

% save RT distributions and means under sum/max model as output
if contains(model,'sum')
    RT_distr = RTsum;
    RT_mean=RTsum_mean;
elseif contains(model,'max')
    RT_distr = RTmax;
    RT_mean=RTmax_mean;
end


simData = [];
nlogl = [];


%% simulate RT of single trial by drawing from sum vs. max distribution
if p.Results.sim
    simData.coh1 = [];
    simData.coh2 = [];
    simData.rt = [];
    for c1 = 1:length(coh)
        for c2 = 1:length(coh)
            if c2 >= c1 % e.g., only consider low-medium, but ignore medium-low
                for tr = 1:p.Results.trialNum(c1,c2)
                    if contains(model, 'sum')
                        RT = find(cumsum(RTsum(:,c1,c2)) > rand,1,'first');
                    elseif contains(model, 'max')
                        RT = find(cumsum(RTmax(:,c1,c2)) > rand,1,'first');
                    end
                    
                    simData.coh1 = [simData.coh1; coh(c1)];
                    simData.coh2 = [simData.coh2; coh(c2)];
                    simData.rt = [simData.rt; RT];
                    
                end
            end
        end
    end
end

%% get log likelihood for RT of each trial under sum vs. max rule
% if no simulation, just fitting
if ~p.Results.sim
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
end

