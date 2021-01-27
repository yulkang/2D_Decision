function P = get_P_for_one_dim(theta,USfunc,coh)
% function P = get_P_for_one_dim(theta,USfunc,coh)
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

kappa  = theta(1);
B0     = theta(2);
a      = theta(3);
d      = theta(4);
coh0   = theta(5);
y0a    = 0;

%%
notabs_flag = false;

%%

dt = 0.0005;
t  = 0:dt:10;

%% bounds
[Bup,Blo] = expand_bounds(t,B0,a,d,USfunc);

%%

y  = linspace(min(Blo)-0.3,max(Bup)+0.3,1500)';

y0a = clip(y0a,Blo(1),Bup(1));

y0 = zeros(size(y));
y0(findclose(y,y0a)) = 1;
y0 = y0/sum(y0);


%%
% prior = Rtable(coh)/sum(Rtable(coh));

%%
drift = kappa * unique(coh + coh0);
P = dtb_fp_cc_vec(drift,t,Bup,Blo,y,y0,notabs_flag);


end