% 07/2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

%%

load ../data/RT_task/data_RT.mat


%% Bigger model, more interactions
clear delta_aic delta_bic
for idataset=1:3
    for itask = 1:2
        switch idataset % different datasets
            case 1
                I = dataset==1 & bimanual==0;
            case 2
                I = dataset==2 & bimanual==0;
            case 3
                I = dataset==2 & bimanual==1;
        end
        
        testSignificance.vars = [1,2,3,4];
        
        dummy_group = adummyvar(group(I));
        
        if itask==1
            depvar = choice_motion(I);
            dummy_coh = adummyvar(coh_motion(I));
            dummy_coh_other = adummyvar(coh_color(I));
            coh = coh_motion(I);
            coh_other = coh_color(I);
        else
            depvar = choice_color(I);
            dummy_coh = adummyvar(coh_color(I));
            dummy_coh_other = adummyvar(coh_motion(I));
            coh = coh_color(I);
            coh_other = coh_motion(I);
        end
        
        dummy_coh_other_group = [];
        for kk=1:size(dummy_group,2)
            dummy_coh_other_group = [dummy_coh_other_group, ...
                bsxfun(@times,dummy_coh_other, dummy_group(:,kk)==1)];
        end
        
        indepvar = {'coh_main',bsxfun(@times,dummy_group,coh),...
            'coh_other',bsxfun(@times,dummy_group,abs(coh_other)),...
            'coh_interact',bsxfun(@times,dummy_group,coh.*abs(coh_other)),...
            'ones',dummy_group};
        
        [beta,idx,stats,x,LRT] = f_regression(depvar,[],indepvar,testSignificance);
        
        for ivar=1:length(testSignificance.vars)
            delta_bic(idataset,itask,ivar) = LRT(ivar).bic.full_minus_restricted;
            delta_aic(idataset,itask,ivar) = LRT(ivar).aic.full_minus_restricted;
            pvals(idataset,itask,ivar) = LRT(ivar).p;
        end
        
    end
    
end



%% separate models for each subject
clear delta_aic delta_bic
for i=1:3
    switch i % different datasets
        case 1
            J = dataset==1 & bimanual==0;
        case 2
            J = dataset==2 & bimanual==0;
        case 3
            J = dataset==2 & bimanual==1;
    end
    uni_group = unique(group(J));
    
    for j=1:length(uni_group)
        
        I = J & group==uni_group(j);
        
        testSignificance.vars = [1,2,3,4];
        
        dummy_group = adummyvar(group(I));
        
        for itask=1:2
            if itask==1
                depvar = choice_motion(I);
                dummy_coh = adummyvar(coh_motion(I));
                dummy_coh_other = adummyvar(coh_color(I));
                coh = coh_motion(I);
                coh_other = coh_color(I);
            else
                depvar = choice_color(I);
                dummy_coh = adummyvar(coh_color(I));
                dummy_coh_other = adummyvar(coh_motion(I));
                coh = coh_color(I);
                coh_other = coh_motion(I);
            end
            
            
            indepvar = {'coh_main',coh,...
                'coh_other',abs(coh_other),...
                'coh_interact',coh.*abs(coh_other),...
                'ones',dummy_group};
            
            [beta,idx,stats,x,LRT] = f_regression(depvar,[],indepvar,testSignificance);
            
            delta_aic{i}(j,itask) = LRT(3).aic.full_minus_restricted;
            delta_bic{i}(j,itask) = LRT(3).bic.full_minus_restricted;
            
            %         for k=1:length(testSignificance.vars)
            %             delta_bic(i,itask,k) = LRT(k).bic.full_minus_restricted;
            %             delta_aic(i,itask,k) = LRT(k).aic.full_minus_restricted;
            %             pvals(i,itask,k) = LRT(k).p;
            %         end
        end
        
    end
    
end

% sum over subjects
[sum(delta_bic{1}),sum(delta_bic{2})]
[sum(delta_aic{1}),sum(delta_aic{2})]

%%



