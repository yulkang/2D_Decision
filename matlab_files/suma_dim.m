function y = suma_dim(x,dim)
%function y = suma_dim(x,dim)

for i=1:length(dim)
    x=nansum(x,dim(i));
end
y=squeeze(x);

