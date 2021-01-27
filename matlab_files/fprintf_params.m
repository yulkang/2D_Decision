function fprintf_params(var_names,err,theta)
% function fprintf_params(var_names,err,theta)

str = 'err=%.3f ';
for i=1:length(var_names)
    str = [str, var_names{i},'=%2.2f '];
end
str = [str, ' \n'];
v = [err, theta(:)'];
% 'err=%.3f kappa=%.2f ndt_mu=%.2f ndt_s=%.2f coh0=%.2f y0=%.2f \n'

fprintf(str, v);

end