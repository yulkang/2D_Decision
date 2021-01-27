function [t,Y,S,SIZ,ugroup] = curva_media_hierarch(DEP,INDEP,group,filt,plotflag)
%%function [t,Y,S,SIZ,ugroup] = curva_media_hierarch(DEP,INDEP,group,filt,plotflag)

if isempty(filt)
    filt = true(size(group));
end

ugroup = unique(group(filt));
ngroup = length(ugroup);
t = unique(INDEP(filt));
nt = length(t);
if isvector(DEP)
    Y = nan(nt,ngroup);
    S = nan(nt,ngroup);
    SIZ = nan(nt,ngroup);
else
    m = numel(DEP)/nt;
    Y = nan(nt,m,ngroup);
    S = nan(nt,m,ngroup);
    SIZ = nan(nt,m,ngroup);
end

hfig = randi(1000);
for i=1:ngroup
    pflag = plotflag;
    if plotflag>0 && plotflag<=2
        figure()
    elseif plotflag==3
        figure(hfig);%same fig
        pflag = 1;
    else
        pflag = 0;
    end
        
    [tt,yy,ss,siz] = curva_media(DEP,INDEP,filt & group==ugroup(i),pflag);
    if pflag~=0
        hold all
    end
    inds = ismember(t,tt);
    if isvector(yy)
        Y(inds,i) = yy;
        S(inds,i) = ss;
        SIZ(inds,i) = siz;
    else
        Y(inds,:,i) = yy;
        S(inds,:,i) = ss;
        SIZ(inds,:,i) = siz;
    end
end