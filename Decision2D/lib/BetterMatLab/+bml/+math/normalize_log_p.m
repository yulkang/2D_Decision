function v = normalize_log_p(v)
% make exp(v) sum to 1.
% 
% v = normalize_log_p(v)
v = bsxfun(@minus, v, max(v));
v = nan0( ...
    bsxfun(@minus, v, log(nansum(exp(v)))));
        
return;

%% Test
v0 = rand(1,5);
v1 = v0 ./ sum(v0);
v2 = exp(bml.math.normalize_log_p(log(v0)));

disp(v1);
disp([sum(v1) - 1, sum(v2) - 1]);