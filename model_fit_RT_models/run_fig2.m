function run_fig2(dataset_num,bimanual_flag,always_use_monomanual_fits)

% function run_fig2(dataset_num,bimanual_flag,always_use_monomanual_fits)
%
% INPUTS:
% dataset_num:      1-Eye-RT, 2-Hand-RT
% bimanual_flag:    [0,1], only relevant for Anne's dataset
% always_use_monomanual_fits: [0,1], 
%
% OUPUTS: 
% Figure 2 of the paper
% 
% 07/2020 Ariel Zylberberg wrote it (ariel.zylberberg@rochester.edu)
% 

%%

fit_type = 0; % fit pred, or all coh

addpath(genpath('../matlab_files'));

%%

datadir = '../data/RT_task/';
load(fullfile(datadir,'data_RT'),'RT','coh_motion','coh_color','corr_motion','corr_color',...
    'choice_color','choice_motion','bimanual','dataset','group','SignedColorStrengthLogodds');

% plot color in log scale - only for yul, because of the need to combine Ss
% with different coh, that are proportional only in Logodds scale
I = dataset==1;
coh_color(I) = SignedColorStrengthLogodds(I);

IDX_DATA = dataset == dataset_num & bimanual == bimanual_flag & ~isnan(RT); % NEW: index for data points (not fits)
% idx for correct trials (and all 0% coherence trials) for RT plots
IDX_DATA_CORR = IDX_DATA & (corr_color | coh_color == 0) & (corr_motion | coh_motion == 0);

%%

switch fit_type
    case 0
        extension = '';
    case 1
        extension = '_allcoh';
    case 2
        extension = ''; % used for the mono pred with bi data
end


show_data_pred = 1;

show_errorbars_flag = 0;
do_average_per_suj_first = 0;


if always_use_monomanual_fits==1
    show_data_fit = 0;
else
    show_data_fit = 1;
end


if dataset_num==1 % for eye data
    savefilename = 'fig_paper_eyeRT';
else % for manual task
    if bimanual_flag==0
        savefilename = 'fig_paper_monoRT'; % unimanual
    else
        if always_use_monomanual_fits
            savefilename = 'fig_paper_biRT'; % bimanual data (unimanual fits)
        else
            savefilename = 'fig_paper_biRT_fitted'; % bimanual data (bimanual fits)
        end
    end
end

do_save = 1;

%% all unique combinations

all_combs = unique([dataset,group,bimanual],'rows');


%%
p = publish_plot(3,2);
% set(gcf,'Position',[395  196  703  526]);
set(gcf,'Position',[409  138  577  652]);
p.displace_ax([1,2,3,4],-0.06,2);
p.displace_ax([1,2],0.03,2);
% p.displace_ax([1,2],-0.06,2);

uni_serial = [0,1];
for i_serial = 1:length(uni_serial)
    
    serial_flag = uni_serial(i_serial);
    
    combs = all_combs;
    
    I = combs(:,1)==dataset_num & ...     % 1: eye; 2: hand
        combs(:,3)==bimanual_flag;     % 0: monomanual; 1: bimanual
    
    
    combs = combs(I,:);
    
    
    %%
    IDX = zeros(size(coh_color));
    p_motion_fine = [];
    p_color_fine = [];
    clear mRT_motion mRT_color count
    for i=1:size(combs,1)
        
        if serial_flag
            str = 'fit_serial';
        else
            str = 'fit_parallel';
        end
        
        if bimanual_flag == 0 %always_use_monomanual_fits
            filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b0',extension,'.mat']; % eye data / unimanual fits
        else
            filename = [str,'_d',num2str(combs(i,1)),'_s',num2str(combs(i,2)),'_b',num2str(combs(i,3)),extension,'.mat']; % bimanual fits
        end
        
        fullname = fullfile('./finer_coh/',filename);
        
        d = load(fullname);
                 
        IDX = IDX | d.idx;
        count(i) = sum(d.idx);
        d.coh_m = fix(d.coh_m*1e6)/1e6; % needs some rounding to make unique work - annoying
        
        % convert the color scale to logodds - only for Yul
        if dataset_num==1
            pblue = (d.coh_color_fine+1)/2;
            d.coh_color_fine = log(pblue./(1-pblue));
        end

        % get subject data
        [~,mRT_color(:,:,i)] = curva_media_hierarch(d.E_RT_color_correct,d.coh_c(:,2),abs(d.coh_c(:,1)),[],0);
        try
            [~,mRT_motion(:,:,i)] = curva_media_hierarch(d.E_RT_motion_correct,d.coh_m(:,1),abs(d.coh_m(:,2)),[],0);
        catch
            aa
        end
        
        p_motion_fine = [p_motion_fine; d.p_motion_fine'];
        p_color_fine = [p_color_fine; d.p_color_fine'];
        
        % normalize the cohs to the max for each subjectg
        do_normalize = 1;
        if do_normalize && all(combs(:,1)==1) % Yul's dataset (eye data)
            norm_m = max(d.coh_motion_fine);
            norm_c = max(d.coh_color_fine);
            %disp(norm_m)
            
            d.coh_motion_fine = d.coh_motion_fine/norm_m;
            d.coh_color_fine = d.coh_color_fine/norm_c;
            
            if i_serial==1 % to not do it twice
                
%                 coh_motion(d.idx) = coh_motion(d.idx)/norm_m;
%                 coh_color(d.idx) = coh_color(d.idx)/norm_c;
                 coh_motion(d.idx) = coh_motion(d.idx)/max(coh_motion(d.idx));
                 coh_color(d.idx) = coh_color(d.idx)/max(coh_color(d.idx));
                 
            end
        end
        
    end
    
    % do weighted (by trial) averages over subjects
    if ~do_average_per_suj_first
        nsuj = length(count);
        count = reshape(count,[1,size(count)]);
        mRT_color = nsuj * bsxfun(@times,mRT_color,count) / sum(count);
        mRT_motion = nsuj *bsxfun(@times,mRT_motion,count) / sum(count);
    
        p_motion_fine = nsuj * bsxfun(@times,p_motion_fine,squeeze(count))/sum(count);
        p_color_fine = nsuj * bsxfun(@times,p_color_fine,squeeze(count))/sum(count);
    end
    
    
    xli1 = [nanmin(coh_motion(IDX)),nanmax(coh_motion(IDX))];
    xli2 = [nanmin(coh_color(IDX)),nanmax(coh_color(IDX))];
    
    
    
    
    %%
    
    msize = 6;
    
    n = size(mRT_motion,2);
    
    
    colors_flag = 3;
    switch colors_flag
        case 1
            colores2 = bone(n+1);
            colores2 = colores2(1:end-1,:);
            colores2 = colores2(end:-1:1,:);
            colores1 = rainbow_colors(n,'colorType',3);
        case 2
            colores1 = rainbow_colors(n,'colorType',2);
            colores2 = rainbow_colors(n,'colorType',3);
        case 3
            addpath('../colormaps/YK/');
            colores1 = winter2(n);
            colores1 = colores1(end:-1:1,:);
            colores2 = cool2(n);
            colores2 = colores2(end:-1:1,:);
            
    end
    
    p.current_ax(1);
    if serial_flag==1
        
        % plot model predictions
        plot(d.coh_motion_fine,nanmean(p_motion_fine,1),'LineStyle','-','color',colores1(end,:));
        
        
        if do_average_per_suj_first
            tt = unique(coh_motion(IDX_DATA));
            conditions = [coh_motion,abs(coh_color),group];
            [Mean,~,uni_conditions,~,~] = average_per_condition(choice_motion, ...
                conditions,'filter',IDX_DATA);
            xx = average_per_condition(Mean,uni_conditions(:,1:2));
            xx = reshape(xx,n,length(xx)/n)';
        else
            [tt,xx,ss] = curva_media_hierarch(choice_motion,coh_motion,abs(coh_color),IDX_DATA,0);
        end
        
        
        if show_data_pred
            for i=1:n
                
                hold all
                handle_color(i) = plot(tt,xx(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markerfacecolor',colores1(i,:),...
                    'markersize',msize);
            end
        end
        if fit_type==0 && show_data_fit
            
            plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                'markeredgecolor',colores1(n,:),'markersize',msize);
            
        end
        
    else
        plot(d.coh_motion_fine,nanmean(p_motion_fine,1),'LineStyle','--','color',colores1(end,:));
        hold on
    end
    
    p.current_ax(2);
    if serial_flag==1
        n = size(mRT_color,2);
        plot(d.coh_color_fine,nanmean(p_color_fine,1),'LineStyle','-','color',colores2(end,:));
        
        if do_average_per_suj_first
            tt = unique(coh_color(IDX_DATA));
            conditions = [coh_color,abs(coh_motion),group];
            [Mean,~,uni_conditions,~,~] = average_per_condition(choice_color, ...
                conditions,'filter',IDX_DATA);
            xx = average_per_condition(Mean,uni_conditions(:,1:2));
            xx = reshape(xx,n,length(xx)/n)';
        else
            [tt,xx,ss] = curva_media_hierarch(choice_color,coh_color,abs(coh_motion),IDX_DATA,0);
        end
        
        if show_data_pred
            for i=1:n
                hold all
                %     terrorbar(tt,xx(:,i),ss(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor',colores2(i,:),...
                %         'markersize',msize);
                handle_motion(i) = plot(tt,xx(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor',colores2(i,:),...
                    'markersize',msize);
            end
        end
        if fit_type==0 && show_data_fit
            plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                'markeredgecolor',colores2(n,:),'markersize',msize);
        end
    else
        plot(d.coh_color_fine,nanmean(p_color_fine,1),'LineStyle','--','color',colores2(end,:));
        hold on
    end
    
    if serial_flag==0
        p.current_ax(5);
    else
        p.current_ax(3);
    end
    n = size(mRT_motion,2);
    

    
    [tt,xx,ss] = curva_media_hierarch(RT,coh_motion,abs(coh_color),IDX_DATA_CORR,0);
    % test

    if do_average_per_suj_first
        tt = unique(coh_motion(IDX_DATA));
        conditions = [coh_motion,abs(coh_color),group];
        [Mean,Stdev,uni_conditions,tr_per_cond,idx_cond] = average_per_condition(RT, conditions,'filter',IDX_DATA_CORR);
        [xx,ss,uni_conditions, tr_per_cond] = average_per_condition(Mean,uni_conditions(:,1:2));

        xx = reshape(xx,n,length(xx)/n)';
        ss = ss./sqrt(tr_per_cond);
        ss = reshape(ss,n,length(ss)/n)';
    end
    % end test
    
    for i=1:n
        plot(d.coh_motion_fine,nanmean(mRT_motion(:,i,:),3),'color',colores1(i,:))
        
        hold all
        %     terrorbar(tt,xx(:,i),ss(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markersize',msize,...
        %         'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
        if show_data_pred
            if show_errorbars_flag && (i==1 || i==n)
                terrorbar(tt,xx(:,i),ss(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markersize',msize,...
                    'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
            else
                plot(tt,xx(:,i),'color',colores1(i,:),'linestyle','none','marker','o','markersize',msize,...
                    'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
            end
        end
    end
    if fit_type==0 && show_data_fit
        plot(tt,xx(:,n),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
            'markeredgecolor',colores1(i,:),'markersize',msize);
        for i=1:size(xx,2)
            plot(tt(1),xx(1,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                'markeredgecolor',colores1(i,:),'markersize',msize);
            plot(tt(end),xx(end,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                'markeredgecolor',colores1(i,:),'markersize',msize);
        end
    end
    
    
    p.next();
    n = size(mRT_color,2);
    [tt,xx,ss] = curva_media_hierarch(RT,coh_color,abs(coh_motion),IDX_DATA_CORR,0);
    
    if do_average_per_suj_first
        tt = unique(coh_color(IDX_DATA));
        conditions = [coh_color,abs(coh_motion),group];
        [Mean,Stdev,uni_conditions,tr_per_cond,idx_cond] = average_per_condition(RT, conditions,'filter',IDX_DATA_CORR);
        [xx,ss,uni_conditions,tr_per_cond] = average_per_condition(Mean,uni_conditions(:,1:2));

        xx = reshape(xx,n,length(xx)/n)';
        ss = ss./sqrt(tr_per_cond);
        ss = reshape(ss,n,length(ss)/n)';
    end
    
    
    for i=1:n
        plot(d.coh_color_fine,nanmean(mRT_color(:,i,:),3),'color',colores2(i,:))
        hold all
        %     terrorbar(tt,xx(:,i),ss(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markersize',msize,...
        %         'markerfacecolor',colores1(i,:),'markeredgecolor',colores1(i,:));
        if show_data_pred
            if show_errorbars_flag && (i==1 || i==n)
                terrorbar(tt,xx(:,i),ss(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markersize',msize,...
                    'markerfacecolor',colores2(i,:),'markeredgecolor',colores2(i,:));
            else
                plot(tt,xx(:,i),'color',colores2(i,:),'linestyle','none','marker','o','markersize',msize,...
                    'markerfacecolor',colores2(i,:),'markeredgecolor',colores2(i,:));
            end
        end
    end
    
    hold all
    if fit_type==0 && show_data_fit
%         plot(tt,xx(:,n),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor','w',...
%             'markeredgecolor','k','markersize',msize);
        plot(tt,xx(:,n),'color',colores2(i,:),'linestyle','none','marker','o','markerfacecolor','w',...
            'markeredgecolor',colores2(i,:),'markersize',msize);
        for i=1:size(xx,2)
            plot(tt(1),xx(1,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                'markeredgecolor',colores2(i,:),'markersize',msize);
            plot(tt(end),xx(end,i),'color','w','linestyle','none','marker','o','markerfacecolor','w',...
                'markeredgecolor',colores2(i,:),'markersize',msize);
        end
    end
    
end

set(p.h_ax([1,3,5]),'xlim',[nanmin(coh_motion),nanmax(coh_motion)]);
set(p.h_ax([2,4,6]),'xlim',[nanmin(coh_color),nanmax(coh_color)]);

% set(p.h_ax([1:4]),'xticklabel',[]);
set(p.h_ax([3:4]),'xticklabel',[]);


if show_data_pred
    p.current_ax(1);
    if dataset_num==1
        % for Yul's dataset, the other dims values are not unique, and thus
        % I use 1-5 labels
        hl(1) = legend(handle_color,{'low','','','','high'});
        set(get(hl(1),'Title'),'String','Color strength')
    else
        hl(1) = legend(handle_color,strjust(num2str(unique(abs(d.u_coh_color))),'left'));
        set(get(hl(1),'Title'),'String','Color strength')
        %hl(1) = legend_n(unique(abs(d.u_coh_color)),'hline',handle_color,'title','Color strength');
    end
    
    
    p.current_ax(2);
    if dataset_num==1
        % for Yul's dataset, the other dims values are not unique, and thus
        % I use 1-5 labels
        hl(2) = legend(handle_motion,{'low','','','','high'});
        set(get(hl(2),'Title'),'String','Motion strength')
    else
        hl(2) = legend(handle_motion,strjust(num2str(unique(abs(d.u_coh_motion))),'left'));
        set(get(hl(2),'Title'),'String','Motion strength')
        %hl(2) = legend_n(unique(abs(d.u_coh_motion)),'hline',handle_motion,'title','Motion strength');
    end
    % ylabel('Proportion "blue" choices')
end

p.current_ax(1);
if do_normalize && all(combs(:,1)==1)
    xlabel('Motion strength (norm.)')
else
    xlabel('Motion strength (coh.)')
end
ylabel('Proportion ''rightward'' choices');

p.current_ax(2);
if do_normalize && all(combs(:,1)==1)
    xlabel('Color strength (norm.)')
else
    xlabel('Color strength (coh.)')
end
ylabel('Proportion ''blue'' choices');

p.current_ax(3);
ylabel({'Response time (s)', '(Serial model)'})

p.current_ax(5);
if do_normalize && all(combs(:,1)==1)
    xlabel('Motion strength (norm.)')
else
    xlabel('Motion strength (coh.)')
end
ylabel({'Response time (s)', '(Parallel model)'})

p.current_ax(6);
if do_normalize && all(combs(:,1)==1)
    xlabel('Color strength (norm.)')
else
    xlabel('Color strength (coh.)')
end


same_ylim(p.h_ax([1,2]));
same_ylim(p.h_ax([3,4]));
same_ylim(p.h_ax([5,6]));

set(p.h_ax(1:2:end),'xlim',xli1);
set(p.h_ax(2:2:end),'xlim',xli2);


set(p.h_ax,'tickdir','out');

p.format('FontSize',13,'LineWidthPlot',1,'LineWidthAxes',.5);
if show_data_pred
    set(hl,'FontSize',10,'Location','SouthEast','Box','off');
    pos = get(hl(2),'position');
    pos(1) = pos(1)+0.025;
    set(hl(2),'position',pos);
end

set(p.h_ax,'tickdir','out');

%% save

if do_save
   
    export_fig('-pdf', savefilename,'-nocrop','-m5', '-q101');
    close all;
end

end