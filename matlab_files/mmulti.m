function P = mmulti(M,x)
%function mmulti(M,x)
%hace un repmat de x y lo multiplica con M punto a punto

dim = find(size(M)==length(x));
if dim==1
    P = M.*repmat(x,1,size(M,2));
else
    P = M.*repmat(x,size(M,1),1);
end