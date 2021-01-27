
%%

for groundtruth_is_serial = 0:1
    
    if groundtruth_is_serial
        load('simdata_RT_yul_anne_serial','RT','coh_motion','coh_color','corr_motion','corr_color',...
            'choice_color','choice_motion','bimanual','dataset','group');
    else
        load('simdata_RT_yul_anne_parallel','RT','coh_motion','coh_color','corr_motion','corr_color',...
            'choice_color','choice_motion','bimanual','dataset','group');
    end
    
    I = ~isnan(dataset);
    combs = unique([dataset(I),group(I),bimanual(I)],'rows');
    
    
    %% all unique combs
    redo_calc = 1;
    if redo_calc
        for serial_flag = 0:1
            % serial and parallel
            %     nn = size(combs,1);
            %     combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];
            
            %% fit to a subset of trials
            
            
            plot_flag = true;
            % pars = struct('USfunc','Logistic');
            
            for i=1:size(combs,1)
                % for i=1:size(combs,1)
                
                I = dataset==combs(i,1) & group==combs(i,2) & bimanual==combs(i,3) & ~isnan(RT);
                % serial_flag = combs(i,4);
                
                
                % like Yul's: just use the highest coh
                I_fit = (abs(coh_motion(I))==max(abs(coh_motion(I))) | abs(coh_color(I))==max(abs(coh_color(I))));
                I_pred = ~I_fit;
                
                
                %% filename
                if serial_flag
                    str = 'fit_serial';
                else
                    str = 'fit_parallel';
                end
                
                filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),'.mat'];
                
                
                %%
                if groundtruth_is_serial
                    fullname = fullfile('./from_fits_groundtruth_serial/',filename);
                else
                    fullname = fullfile('./from_fits_groundtruth_parallel/',filename);
                end
                
                aux = load(fullname,'theta');
                %     theta = tg;
                %     nlogl = wrapper_2D(theta,coh_motion(I),coh_color(I), choice_motion(I),choice_color(I),RT(I),...
                %         corr_motion(I),corr_color(I),plot_flag);
                
                
                theta = aux.theta;
                
                [nlogl,~,~,~,~,pPred] = wrapper_2D(theta,coh_motion(I),coh_color(I), choice_motion(I),choice_color(I),RT(I),...
                    corr_motion(I),corr_color(I),serial_flag, plot_flag);
                
                logl_fit = sum(log(pPred(I_fit)));
                logl_pred = sum(log(pPred(I_pred)));
                
                %% save
                if groundtruth_is_serial
                    savefilename = fullfile('./like_not_pred_groundtruth_serial',filename);
                else
                    savefilename = fullfile('./like_not_pred_groundtruth_parallel',filename);
                end
                save(savefilename,'logl_fit','logl_pred');
                
            end
        end
    end
    %% analyze
    
    
    % select combs
    
    %% all unique combs
    % combs = unique([dataset,group,bimanual],'rows');
    
    % serial and parallel
    % nn = size(combs,1);
    % combs = [[combs;combs], [ones(nn,1);zeros(nn,1)]];
    
    
    % I = combs(:,1)==2 & ...     % 1: Yul; 2: Anne
    %     combs(:,3)==0 & ...     % 0: monomanual; 1: bimanual
    %     combs(:,4)==1;          % 0: parallel; 1: serial
    
    I = ismember(combs(:,3),[0,1]);     % 0: monomanual; 1: bimanual
    
    combs = combs(I,:);
    
    clear v_logl_pred v_logl_all
    for j=1:2
        for i=1:size(combs,1)
            % for i=1:size(combs,1)
            
            %         I = dataset==combs(i,1) & group==combs(i,2) & bimanual==combs(i,3) & ~isnan(RT);
            %
            %
            %         % like Yul's: just use the highest coh
            %         I_fit = (abs(coh_motion(I))==max(abs(coh_motion(I))) | abs(coh_color(I))==max(abs(coh_color(I))));
            %         I_pred = ~I_fit;
            
            
            %% filename
            if j==2
                str = 'fit_serial';
            else
                str = 'fit_parallel';
            end
            
            filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),'.mat'];
            
            %%
            if groundtruth_is_serial
                fullname = fullfile('./like_not_pred_groundtruth_serial/',filename);
            else
                fullname = fullfile('./like_not_pred_groundtruth_parallel/',filename);
            end
            aux = load(fullname);
            v_logl_pred(i,j) = aux.logl_pred;
            v_logl_all(i,j) = aux.logl_pred + aux.logl_fit;
        end
    end
    
    %%
    p = publish_plot(1,1);
    % set(gcf,'Position',[343  264  565  283])
    set(gcf,'Position',[424  121  484  426])
    
    y = diff(-1*v_logl_pred,[],2);
    y = bsxfun(@times,y,adummyvar(combs(:,1)));
    h = barh( y ,'stacked');
    set(h,'barwidth',0.5);
    symmetric_x(gca)
    hold all
    plot([0,0],ylim,'k','LineWidth',2)
    hold all
    plot([-10,-10],ylim,'k--')
    
    plot([10,10],ylim,'k--')
    
    str = {};
    for i=1:size(combs,1)
        str{i} = ['S',num2str(combs(i,2))];
    end
    set(gca,'ytick',1:size(v_logl_pred,1),'yticklabel',str)
    xlabel('Difference in log-likelihood of the predictions')
    ylabel('Participants')
    
    set(gca,'tickdir','out');
    
    xlim([-350,350])
    format_figure(gcf,'FontSize',16);
    
    %%
    %% save for Yul
    
    model_comp.readme = 'positive values are support for the parallel model. One datapoint per subject.';
    vv = -diff(v_logl_pred,[],2);
    I = combs(:,1)==1;
    model_comp.delta_logl_predictions.yuls_exp = vv(I,:);
    I = combs(:,1)==2 & combs(:,3)==0;
    model_comp.delta_logl_predictions.annes_mono = vv(I,:);
    I = combs(:,1)==2 & combs(:,3)==1;
    model_comp.delta_logl_predictions.annes_bi = vv(I,:);
    
    if groundtruth_is_serial
        savefilename = 'model_comparison_delta_logl_predictions_groundtruth_serial';
    else
        savefilename = 'model_comparison_delta_logl_predictions_groundtruth_parallel';
    end
    
    save(savefilename,'-struct','model_comp');
    
    
end
