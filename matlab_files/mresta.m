function P = mresta(M,x)
%function mresta(M,x)
%le resta a M un repmat de x

dim = find(size(M)==length(x));
if dim==1
    P = M-repmat(x,1,size(M,2));
else
    P = M-repmat(x,size(M,1),1);
end