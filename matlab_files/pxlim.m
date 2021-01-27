function pxlim(ylevel)
if nargin==0
    ylevel=0;
end
hold on
plot(xlim,[ylevel ylevel],'k:')