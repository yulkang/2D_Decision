function run_do_stats()
    
load ../data/Short_Dur/all.mat
m = load('../data/Short_Dur/me.mat');

%%
% convert logodds to coh_color
pblue = exp(coh_color)./(1+exp(coh_color));
coh_color = 2*pblue-1; % overwrite
            
%% do stats

I = ismember(task,{'A','V'});
dep = choice_color(I);
dummytask = adummyvar(task(I));
dummygroup = adummyvar(group(I));
indep = {'coh_color',coh_color(I),'ones',dummygroup,'is_double',ismember(task(I),'V'),...
    'interact',bsxfun(@times,coh_color(I),dummytask(:,1))};
testSignificance.vars = [1,2,3,4];
[beta,idx,stats,x,LRT_color] = f_regression(dep,[],indep,testSignificance);


I = ismember(task,{'A','H'});
dep = choice_motion(I);
dummytask = adummyvar(task(I));
dummygroup = adummyvar(group(I));
indep = {'coh_motion',coh_motion(I),'ones',dummygroup,'is_double',ismember(task(I),'H'),...
    'interact',bsxfun(@times,coh_motion(I),dummytask(:,1))};
testSignificance.vars = [1,2,3,4];
[beta,idx,stats,x,LRT_motion] = f_regression(dep,[],indep,testSignificance);

% stats in paper:
LRT_motion(4).bic
LRT_color(4).bic

end


