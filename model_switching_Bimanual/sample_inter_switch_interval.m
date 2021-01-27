function R = sample_inter_switch_interval(isi_params, dim,type)

if nargin<3 || isempty(type)
    type = 1;
end
switch type
    case 1
        R = exprnd(isi_params,[dim]);
    case 2 % max of two exp
        R = max(exprnd(isi_params,[dim]),exprnd(isi_params,[dim]));
    case 3 % min + exp sample
        R = exprnd(isi_params(2),[dim]) + isi_params(1);
end

end