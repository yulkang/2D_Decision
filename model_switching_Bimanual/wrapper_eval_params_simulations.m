function [err, y_mm, y_mc, y_cm, y_cc,motion_was_first] = wrapper_eval_params_simulations(theta, Pcolor, Pmotion, dat, idx, uni_coh_motion, uni_coh_color,fit_type)
% function [err, y_mm, y_mc, y_cm, y_cc,motion_was_first] = wrapper_eval_params_simulations(theta, Pcolor, Pmotion, dat, idx, uni_coh_motion, uni_coh_color,fit_type)
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

isi = theta(1);
p_start_with_color = theta(2);
ndt_m = theta(3);

isi_params = isi;
[mean_dec_time_motion, mean_dec_time_color,motion_was_first] = ...
    calc_Mean_Dec_Time_with_switches_sim(Pmotion, Pcolor, [isi_params], p_start_with_color);


prior = ones(length(Pcolor.drift),1);
prior([length(prior)+1]/2) = 2; % zero has higher prior
prior = prior/sum(prior);

do_plot = 1;
if do_plot
    
    p = publish_plot(2,2,'hfig',1);
    
    p.next();
    y_mm = mean_dec_time_motion*prior + ndt_m;
    plot(uni_coh_motion, y_mm);
    hold all
    [tt,xx,ss] = curva_media(dat.RT1,dat.coh_motion,idx & dat.color_responded_first==0,3);
    pp = normpdf(xx,y_mm,ss);
    pp(pp<eps)=eps;
    L(1,1) = sum(log(pp));
    R(1,1) = sum((xx-y_mm).^2);
    hold off
    
    p.next();
    y_mc = mean_dec_time_motion'*prior + ndt_m;
    plot(uni_coh_color, y_mc)
    hold all
    [tt,xx,ss] = curva_media(dat.RT1,dat.coh_color,idx & dat.color_responded_first==0,3);
    pp = normpdf(xx,y_mc,ss);
    R(1,2) = sum((xx-y_mc).^2);
    pp(pp<eps)=eps;
    L(1,2) = sum(log(pp));
    hold off
    
    p.next();
    y_cm = mean_dec_time_color*prior + ndt_m;
    plot(uni_coh_motion, y_cm)
    hold all
    [tt,xx,ss] = curva_media(dat.RT1,dat.coh_motion,idx & dat.color_responded_first==1,3);
    pp = normpdf(xx,y_cm,ss);
    pp(pp<eps)=eps;
    L(2,1) = sum(log(pp));
    R(2,1) = sum((xx-y_cm).^2);
    hold off
    
    p.next();
    y_cc = mean_dec_time_color'*prior + ndt_m;
    plot(uni_coh_color, y_cc)
    hold all
    [tt,xx,ss] = curva_media(dat.RT1,dat.coh_color,idx & dat.color_responded_first==1,3);
    pp = normpdf(xx,y_cc,ss);
    pp(pp<eps)=eps;
    L(2,2) = sum(log(pp));
    R(2,2) = sum((xx-y_cc).^2);
    hold off
    
    same_ylim(p.h_ax)
    p.format();
    
    drawnow
    
end

%% calc err
% calculo el likelihood con los errores de los datos
L(isinf(L) | isnan(L)) = 20*log(eps);
if fit_type==1
    err = -sum(L(:));
elseif fit_type==2
    err = -1*(L(1,1)+L(2,2));
elseif fit_type==3
    err = sum(R(:));
elseif fit_type==4
    err = sum(R(1,1)+R(2,2));    
end


%% 

var_names = {'isi','p_start_with_color','ndt_m'};
fprintf_params(var_names,err,theta);

%%

end