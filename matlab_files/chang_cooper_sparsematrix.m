function M = chang_cooper_sparsematrix(drift,nd,ny,dy,dt,var)
% function M = chang_cooper_sparsematrix(drift,nd,ny,dy,dt,var)

if nargin<6 || isempty(var)
    var = 1;
end

% half the variance of the momentary evidence
D = 0.5 * var;
if isscalar(var) %make it a vector
    D = D*ones(size(drift));
end

aux = nan(nd*ny,3);
for idrift = 1:nd
    a = -1*drift(idrift);
    w = dy*a/D(idrift);
    %if drift is too close to zero consider it zero to avoid problems with
    %singular matrix
    if abs(a)<1e-9
        delta = 0.5;
    else
        delta = 1/w - 1/(exp(w)-1);
    end
    m1 = D(idrift)/dy-delta*a;
    m2 = -1*(1/dy*2*D(idrift)+a-2*delta*a) - dy/dt;
    m3 = (1-delta)*a+D(idrift)/dy;
    
    inds = [1:ny]+ny*(idrift-1);
    aux(inds,:) = -dt/dy * repmat([m1,m2,m3],ny,1);
end
M = spdiags(aux,-1:1,ny*nd,ny*nd);
