function [nlogl,Pmotion,Pcolor,E_RT_correct,S_RT_correct,pPred,dist] = wrapper_2D(theta,coh_motion,coh_color, choice_motion,choice_color,RT,corr_motion,corr_color,serial_flag, do_plot)
% function [nlogl,Pmotion,Pcolor,E_RT_correct,S_RT_correct,pPred,dist] = wrapper_2D(theta,coh_motion,coh_color, choice_motion,choice_color,RT,corr_motion,corr_color,serial_flag, do_plot)
% 
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

% theta = [10,1,0.1,0.5,0, 10,1,0.1,0.5,0, 0.5, 0.05]; 

% motion parameters
mo.kappa  = theta(1); % kappa/drift
mo.B0     = theta(2); % bound
mo.a      = theta(3); % collapsing bound
mo.d      = theta(4); % 
mo.coh0   = theta(5); % coherence bias

% color parameters
c.kappa  = theta(6);
c.B0     = theta(7);
c.a      = theta(8);
c.d      = theta(9);
c.coh0   = theta(10);

ndt_m  = theta(11);
ndt_s  = theta(12);

if length(theta)>12
    p_guess = theta(13);
else
    p_guess = 0;
end


%% compute 1D marginals
USfunc = 'Logistic';
theta_motion = [mo.kappa,mo.B0,mo.a,mo.d,mo.coh0];
Pmotion = get_P_for_one_dim(theta_motion, USfunc, coh_motion);

theta_color = [c.kappa,c.B0,c.a,c.d,c.coh0];
Pcolor = get_P_for_one_dim(theta_color, USfunc, coh_color);


%% convolve (or max) the decision times

u_coh_motion_all = unique(coh_motion); 
u_coh_color_all = unique(coh_color);
uni_coh = unique([coh_motion,coh_color],'rows');
n = size(uni_coh,1);
str = {'lo','up'};
dec_t = zeros(n,length(Pmotion.t*2-1),2,2);
for l = 1:2 % choice_motion
    for m = 1:2 % choice_color
        for k = 1:n %coherence pair
            cc = uni_coh(k,:);
            i = u_coh_motion_all==cc(1);
            j = u_coh_color_all==cc(2);
            chm = str{l};
            chc = str{m};
            
            % for serial
            % serial_flag = 0; % read it from elsewhere
            if serial_flag
                imax1 = find(cumsum(Pmotion.(chm).pdf_t(i,:))>0.99999 * sum(Pmotion.(chm).pdf_t(i,:)),1);
                imax2 = find(cumsum(Pcolor.(chc).pdf_t(j,:))>0.99999 * sum(Pcolor.(chc).pdf_t(j,:)) ,1);

                aux = conv(Pmotion.(chm).pdf_t(i,1:imax1),Pcolor.(chc).pdf_t(j,1:imax2));
                dec_t(k,1:length(aux),l,m) = aux;
                
            else % parallel
                p1 = Pmotion.(chm).pdf_t(i,:);
                p2 = Pcolor.(chc).pdf_t(j,:);
                c1 = [0,cumsum(p1(1:end-1))];
                c2 = [0,cumsum(p2(1:end-1))];

                aux = p1.*c2 + p2.*c1 + p1.*p2; % distribution of the maximum: either p1 finished and the other one finished before, or ... 
                % or both finished together
                dec_t(k,1:length(aux),l,m) = aux;
                
            end
        end
    end
end

%% just keep what's needed from dec_t - for speed
aux = suma_dim(dec_t,[3,4]);
a = bsxfun(@times,cumsum(aux,2),1./sum(aux,2));
[~,B] = max(a>0.99999,[],2);
imax = max(B);
dec_t = dec_t(:,1:imax,:,:); 

%% compute likelihoods

dt = Pmotion.t(2)-Pmotion.t(1);
t = [0:(size(dec_t,2)-1)]*dt;

% convolve with distribution of non-dec times
pd = makedist('Normal','mu',ndt_m,'sigma',ndt_s);   
pd_trunc = truncate(pd,0,inf);
ndt = pd_trunc.pdf(t)*dt;
imax = find(cumsum(ndt)>0.999999,1);
ndt = ndt(1:imax);
        
pPred = nan(size(choice_motion));

for l = 1:2 % choice_motion
    for m = 1:2 % choice_color
        
        resp_t = conv2(1,ndt,dec_t(:,:,l,m));
        
        for k = 1:n
            
            cc = uni_coh(k,:);

            % get the trials for this condition
            I = find(coh_motion == cc(1) & coh_color == cc(2) & ...
                choice_motion==(l-1) & choice_color==(m-1));

            % now judge the prob
            rt_step = ceil(RT(I)/dt);

            J = rt_step<=size(resp_t,2); % out of bounds
            pPred(I(J)) = resp_t(k,rt_step(J));
            
        end
        
        % if plot, compute the expectation of RT
        if do_plot || nargout>1
            
            t = dt*[0:size(dec_t,2)-1];
            
            
            if mo.coh0>0
                K1 = (l==1 & uni_coh(:,1)<0) | (l==2 & uni_coh(:,1)>=0);
            else
                K1 = (l==1 & uni_coh(:,1)<=0) | (l==2 & uni_coh(:,1)>0);
            end
            
            if c.coh0>0
                K2 = (m==1 & uni_coh(:,2)<0) | (m==2 & uni_coh(:,2)>=0);
            else
                K2 = (m==1 & uni_coh(:,2)<=0) | (m==2 & uni_coh(:,2)>0);
            end
            
%             K = (l==1 & uni_coh(:,1)<0 | l==2 & uni_coh(:,1)>=0) & ...
%                 (m==1 & uni_coh(:,2)<0 | m==2 & uni_coh(:,2)>=0);
            
            K = K1 & K2;
            
            dec_t_norm(K,:) = bsxfun(@times,dec_t(K,:,l,m),1./sum(dec_t(K,:,l,m),2));
            
            %(:,l,m) = E_dec_t + ndt_m + switch_timecost_vs_dec_time*dec_t*t';
            

        end
        
    end
end


%clip

if p_guess==0
    pPred(pPred<eps | isnan(pPred)) = eps;
else
    % guess rate - uniform over the range of times
    n_alternatives = 4;
    range_RT = [nanmin(RT),nanmax(RT)];
    pQuess = dt/(diff(range_RT))* 1/n_alternatives;
    pPred = (1-p_guess)*pPred + p_guess * pQuess;
end

nlogl = -sum(log(pPred));

%% print
if length(theta)==12
    var_names = {'k','B','a','d','coh0','k','B','a','d','coh0','m','s'};
elseif length(theta)==13
    var_names = {'k','B','a','d','coh0','k','B','a','d','coh0','m','s','p_guess'};
end
fprintf_params(var_names,nlogl,theta);

%%
if do_plot || nargout>1
    
    E_dec_t = dec_t_norm*t';
    aux = (t-E_dec_t).^2;
    S_dec_t = sqrt(sum(aux.*dec_t_norm,2));
    %E_dec_t = S_dec_t;
    
    E_RT_correct = E_dec_t + ndt_m;
    %     E_RT_correct = 2*E_dec_t;
    
    S_RT_correct = sqrt(S_dec_t.^2 + ndt_s.^2);
    
    % the full dist, for saving
    aux = conv2(1,ndt,dec_t_norm);
    dist.resp_t_norm_correct = aux(:,1:length(t));
    dist.uni_coh = uni_coh;
    dist.t = t;
    
end

if do_plot
    
    figure(1);
    set(gcf,'Position',[504   75  496  723]);
    
    subplot(4,1,1)
    cla
    [t,Y] = curva_media_hierarch(choice_motion,coh_motion,abs(coh_color),[],0);
    rplot(t,Y);
    hold all
    plot(unique(coh_motion),Pmotion.up.p,'k','LineWidth',2)
    ylabel('P motion'); 
    
    subplot(4,1,2)
    cla
    [t,Y] = curva_media_hierarch(choice_color,coh_color,abs(coh_motion),[],0);
    rplot(t,Y);
    hold all
    plot(unique(coh_color),Pcolor.up.p,'k','LineWidth',2)
    ylabel('P color'); 
    
    subplot(4,1,3)
    cla
    K = (corr_color | coh_color == 0) & (corr_motion | coh_motion == 0); %corr_motion==1 & corr_color == 1;
    curva_media(RT,coh_motion,abs(coh_color)==max(abs(coh_color)) & K,3);
    hold all
    curva_media(RT,coh_motion,abs(coh_color)==min(abs(coh_color)) & K,3);
    
    % not I get the model fits
    [tt1,xx1,ss1] = curva_media(E_RT_correct,uni_coh(:,1),abs(uni_coh(:,2))==max(abs(uni_coh(:,2))),0);
    plot(tt1,xx1,'b');
    hold all
    [tt2,xx2,ss2] = curva_media(E_RT_correct,uni_coh(:,1),abs(uni_coh(:,2))==min(abs(uni_coh(:,2))),0);
    plot(tt2,xx2,'r');
    
    ylabel('RT (s)')
    xlabel('Motion coh');
    %     [tt1,xx1,ss1] = curva_media(E_dec_t,uni_coh(:,1),abs(uni_coh(:,2))==max(abs(uni_coh(:,2))),0);
    %     plot(tt1,xx1,'b');
    %     hold all
    %     [tt2,xx2,ss2] = curva_media(E_dec_t,uni_coh(:,1),abs(uni_coh(:,2))==min(abs(uni_coh(:,2))),0);
    %     plot(tt2,xx2,'r');
    %
    %     hold all
    %     plot(tt1,xx2-xx1);
    
    
    subplot(4,1,4);
    cla
    curva_desvio(RT,coh_motion,abs(coh_color)==max(abs(coh_color)) & K, 2);
    hold all
    curva_desvio(RT,coh_motion,abs(coh_color)==min(abs(coh_color)) & K, 2);
    
    
    % not I get the model fits
    [tt1,xx1,ss1] = curva_media(S_RT_correct,uni_coh(:,1),abs(uni_coh(:,2))==max(abs(uni_coh(:,2))),0);
    plot(tt1,xx1,'b');
    hold all
    [tt2,xx2,ss2] = curva_media(S_RT_correct,uni_coh(:,1),abs(uni_coh(:,2))==min(abs(uni_coh(:,2))),0);
    plot(tt2,xx2,'r');
    
    ylabel('RT (s)')
    xlabel('Motion coh');
    
    
    format_figure(gcf);
    drawnow
    
end

%%


end


