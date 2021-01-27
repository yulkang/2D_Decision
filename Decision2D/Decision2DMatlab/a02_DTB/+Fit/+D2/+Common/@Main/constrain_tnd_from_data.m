function constrain_tnd_from_data(W)        
% Set UB of Tnd mean and variance based on minimum RTs

W.Tnd.disper_kind = 'std';
W.Tnd.init_params0;

%
p_data = W.Data.RT_data_pdf;
n_tr_thres = 10;

n_data = squeeze( ...
    sum(p_data));
excl = n_data < n_tr_thres;

[sem_data, mean_data] = sem_distrib(p_data, W.t(:));
sem_data = squeeze(sem_data);
mean_data = squeeze(mean_data);

mean_data(excl) = nan;
sem_data(excl) = nan;

%
[sev_data, var_data] = bml.stat.sev_distrib(p_data, W.t(:));
sev_data = squeeze(sev_data);
var_data = squeeze(var_data);

sev_data(excl) = nan;
var_data(excl) = nan;


ub_mean_data = mean_data + sem_data * 2;
ub_var_data = var_data + sev_data * 2;

mu_ub_tnd = squeeze( ...
    nanmin(nanmin(ub_mean_data,[],1),[],2));
sd_ub_tnd = sqrt(squeeze( ...
    nanmin(nanmin(ub_var_data,[],1),[],2)));

%%
n_ch = 2;
for ch1 = 1:n_ch
    for ch2 = 1:n_ch
        mu = sprintf('mu_%d_%d', ch1, ch2);
        sd = sprintf('disper_%d_%d', ch1, ch2);
        
        mu_ub1 = mu_ub_tnd(ch1, ch2);

        W.Tnd.th_ub.(mu) = mu_ub1;
        W.Tnd.th0.(mu) = mu_ub1 / 3;
        W.Tnd.th.(mu) = mu_ub1 / 3;
        W.Tnd.th_lb.(mu) = min(mu_ub1 / 6, W.Tnd.th_lb.(mu));

        sd_ub1 = sd_ub_tnd(ch1, ch2);

        W.Tnd.th_ub.(sd) = sd_ub_tnd(ch1, ch2);
        W.Tnd.th0.(sd) = sqrt(sd_ub1.^2 / 3);
        W.Tnd.th.(sd) = sqrt(sd_ub1.^2 / 3);
        W.Tnd.th_lb.(sd) = min(sqrt(sd_ub1.^2 / 6), W.Tnd.th_lb.(sd));
    end
end
end