function [mean_dec_time_motion_first, mean_dec_time_color_first,winner] = ...
                calc_Mean_Dec_Time_with_switches_sim(Pmotion, Pcolor, isi_params, p_start_with_color,Nsamples)

if nargin<5 || isempty(Nsamples)
    Nsamples = 1000;
end

t = Pmotion.t;

% isi = .2; %inter switch interval
% p_start_with_color = 0.5; % prob of starting with color

ndrifts_motion = length(Pmotion.drift);
ndrifts_color = length(Pcolor.drift);
mean_dec_time_motion_first = nan(ndrifts_motion,ndrifts_color);
mean_dec_time_color_first = nan(ndrifts_motion,ndrifts_color);
for i=1:ndrifts_motion
    for j=1:ndrifts_color
        
        % sample dec times for each dimension; then sample the number of
        % switches for those decision times
        
        
        
        dist = Pmotion.up.pdf_t(i,:) + Pmotion.lo.pdf_t(i,:);
        K = 1:length(dist);
        dt_motion = t(randsample(K,Nsamples,true,dist))';
        
        dist = Pcolor.up.pdf_t(j,:) + Pcolor.lo.pdf_t(j,:);
        K = 1:length(dist);
        dt_color = t(randsample(K,Nsamples,true,dist))';
        
        start_with_color = rand(Nsamples,1)<p_start_with_color;
        
        %end_with_color = start_with_color;
%         max_switches = ceil(25/isi);
        max_switches = 100;
        type_switch = 2;
        isi_samples = sample_inter_switch_interval(isi_params, [Nsamples, max_switches],type_switch);
        
        dts = [dt_motion, dt_color];
        dt_remaining = dts;
        ind = ones(Nsamples,1);
        ind(start_with_color) = 2;
        finished = false(Nsamples,1);
        nswitches = nan(Nsamples,1);
        winner = nan(Nsamples,1);
        for k=1:max_switches
            dt_remaining(ind==1,1) = dt_remaining(ind==1,1) - isi_samples(ind==1,k); % motion
            dt_remaining(ind==2,2) = dt_remaining(ind==2,2) - isi_samples(ind==2,k); % color
            
            ind = 3 - ind; % switch for the next iteration
            
            I = dt_remaining(:,1)<0 & ~finished;
            J = dt_remaining(:,2)<0 & ~finished;
            nswitches(I | J) = k-1;
            finished(I | J) = 1;
            winner(I) = 1; % motion
            winner(J) = 2; % color
            
        end
        

        % now calc the decision time based on the number of switches before
        % finishing
        dec_t = nan(Nsamples,1);
        for k=1:Nsamples
            if ~isnan(winner(k))
                dec_t(k) = dts(k,winner(k));
                if nswitches(k)>0
                    I = nswitches(k):-2:1;
                    dec_t(k) = dec_t(k) + sum(isi_samples(k,I));
                end
            end
        end
        
        mean_dec_time_motion_first(i,j) = nanmean(dec_t(winner==1));
        mean_dec_time_color_first(i,j) = nanmean(dec_t(winner==2));
        
        %         for itr=1:Nsamples
        %             icont = 0;
        %             while
        %                 isi_samples(itr,
        %
        %             end
        %         end
        
        
        
        
        %         I = ~finished;
        %         while sum(I)>0
        %
        %             isi_samples(I)
        %
        %         end
        
        
        
        %         pother = Pother.up.pdf_t(j,:) + Pother.lo.pdf_t(j,:);
        %         [tt,xx] = curva_suma(pother,fase,[],0);
        %         cxx1 = cumsum(xx);
        %         cxx2 = cumsum([0;xx(1:end-1)]); % pad one zero, if it started with color
        %         dtdist_if_started_with_other = [];
        %         dtdist_if_started_with_same = [];
        %         for k=1:floor(length(t)/fase_length)
        %             tind = fase == k;
        %             pp = P_resp_first.up.pdf_t(i,tind)';
        %
        %             pp = pp*(1-cxx1(k));
        %             dtdist_if_started_with_other = [dtdist_if_started_with_other, [pp; zeros(fase_length,1)]];
        %
        %             pp = pp*(1-cxx2(k));
        %             dtdist_if_started_with_same = [dtdist_if_started_with_same, [pp; zeros(fase_length,1)]];
        %         end
        %
        %         dtdist = dtdist_if_started_with_same(:)*p_start_with_resp_first + ...
        %             dtdist_if_started_with_other(:)*(1-p_start_with_resp_first);
        %
        %         tt = dt*[[1:length(dtdist)]-1];
        %
        %         mean_dec_time(i,j) = (tt*dtdist)/sum(dtdist);
    end
end


end
