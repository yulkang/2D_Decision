function SE = stderror(x)
%function SE = stderr(x)

SE = nanstd(x,[],1)'./sqrt(sum(~isnan(x(:,1))));
