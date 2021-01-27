% AIC & BIC criteria
function varargout=aicbic(logl,numParams,numObs)
% function varargout=aicbic(logl,numParams,numObs)

% n=length(err);
% nlogmse=n*log(mean(err.^2));
out.aic = -2*logl + 2*numParams;
out.bic = -2*logl + numParams*log(numObs);

if nargout==1
    varargout={out};
else
    varargout={out.aic,out.bic};
end