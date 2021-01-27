classdef publish_plot < handle
    % figure wrapper - includes many functions that help prettify matlab
    % figures (e.g., moving/reshaping plots, formatting, ...)
    % 07/2014 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)
    
    properties
        h_ax
        h_fig
        figure_label
        active_axis
        data
        legend
        text
        savename
        savename_fig
        savedir
        invert_colors = 0;
    end
    
    methods
        function obj = publish_plot(n_filas,n_col,varargin)
            
            if nargin==0
                n_filas = [];
            end
            
            hfig = [];
            for i=1:length(varargin)
                if isequal(varargin{i},'hfig')
                    hfig = varargin{i+1};
                end
            end
            
            if ~isempty(hfig)
                obj.h_fig = figure(hfig);
            else
                obj.h_fig = figure;
            end
            
            if ~isempty(n_filas)
                cont = 0;
                for i = 1:n_filas
                    for j = 1:n_col
                        cont = cont + 1;
                        obj.h_ax(cont) = subplot(n_filas,n_col,cont);
                    end
                end
                obj.data.n_filas = n_filas;
                obj.data.n_col = n_col;
            end
            
            set(gcf,'color','w')
            
            % figure size for some common shapes
            if obj.data.n_filas==3 && obj.data.n_col==1
                set(gcf,'Position',[505   56  365  688]);
            elseif obj.data.n_filas==4 && obj.data.n_col==1                
                set(gcf,'Position',[495   56  411  749]);
            elseif obj.data.n_filas==2 && obj.data.n_col==1
                set(gcf,'Position',[474   87  401  576]);
            elseif obj.data.n_filas==1 && obj.data.n_col==5
                set(gcf,'Position',[150   372  1170   177]);
            elseif obj.data.n_filas==1 && obj.data.n_col==2
                set(gcf,'Position',[302  352  743  263]);
            elseif obj.data.n_filas==1 && obj.data.n_col==3
                set(gcf,'Position',[297  335  751  208])
            end
            
        end
        
        function combine(obj,hid)
            
            for i=1:length(hid)
                pos = get(obj.h_ax(hid(i)),'Position');
                Position(i,:) = pos;
            end
            w = max(Position(:,1) + Position(:,3));
            h = max(Position(:,2) + Position(:,4));
            new_pos(1) = min(Position(:,1));
            new_pos(2) = min(Position(:,2));
            new_pos(3) = w-new_pos(1);
            new_pos(4) = h-new_pos(2);
            delete(obj.h_ax(hid));
            obj.h_ax(hid) = nan;
            obj.new_axes('Position',new_pos);
            
        end
        
        function clean_hax(obj)
            obj.h_ax(isnan(obj.h_ax)) = [];
        end
        
        function [idx,I1,I2] = center_plots(obj)
            % returns indices of central plots
            I1 = 1:(obj.data.n_col*(obj.data.n_filas-1));
            
            I2 = 1:length(obj.h_ax);
            I2(1:obj.data.n_col:end) = [];
            
            idx = intersect(I1,I2);
            
        end
        
        
        function unlabel_center_plots(obj,varargin)
            
            tickmarks = true;
            for i=1:length(varargin)
                if isequal(varargin{i},'tickmarks')
                    tickmarks = varargin{i+1};
                end
            end
            
            I1 = 1:(obj.data.n_col*(obj.data.n_filas-1));
            set(obj.h_ax(I1),'xticklabel','');
            for i=1:length(I1)
                set(get(obj.h_ax(I1(i)),'xlabel'),'string','');
            end
            
            I2 = 1:length(obj.h_ax);
            I2(1:obj.data.n_col:end) = [];
            set(obj.h_ax(I2),'yticklabel','');
            for i=1:length(I2)
                set(get(obj.h_ax(I2(i)),'ylabel'),'string','');
            end
            
            if tickmarks==0
                h = obj.h_ax(intersect(I1,I2));
                set(h,'xtick',[],'ytick',[]);
            end
            
        end
        
        function load_from_fig(obj,figure_filename)
            h_fig_old = obj.h_fig;
            h_fig   = open(figure_filename);
            children = get(h_fig,'children');
            h_ax    = children(end:-1:1);
            obj.h_fig = h_fig;
            obj.h_ax  = h_ax;
            set(obj.h_fig,'color','w')
            close(h_fig_old)
            
        end
        
        function set_fig_size(obj,varargin)
            
            xSize = 21.6; ySize = 27.9;%tama�o de la figure
            for i = 1:length(varargin)
                if isequal(varargin{i},'xSize')
                    xSize = varargin{i+1};
                elseif isequal(varargin{i},'ySize')
                    ySize = varargin{i+1};
                end
            end
            
            % Tama�o y posici�n de la figura
            set(obj.h_fig,'PaperUnits','centimeters')
            
            xLeft = (21.6-xSize)/2; yTop = (27.9-ySize)/2;
            set(gcf,'PaperPosition',[xLeft yTop xSize ySize])
            
            %Para lo que se ve en pantalla:
            set(gcf,'Position',[200    -100   700*xSize/21.6  888*ySize/27.9])
            
        end
        
        function new_axes_in_rect(obj,rect,num_axes,separation,dim)
            
            if nargin<5
                dim = 1; % single row
            end
            
            if nargin<4 || isempty(separation)
                separation = 0.04;
            end
            
            if dim==1
                width_rect = rect(3);
                width_each_plot = 1/num_axes*(width_rect - (num_axes-1)*separation);

                for i=1:num_axes
                    xini = (i-1)*(width_each_plot+separation) + rect(1);
                    rect_axis = [xini,rect(2),width_each_plot,rect(4)];
                    obj.new_axes('position',rect_axis);
                end
            else
                height_rect = rect(4);
                height_each_plot = 1/num_axes*(height_rect - (num_axes-1)*separation);

                for i=1:num_axes
                    yini = (i-1)*(height_each_plot+separation) + rect(2);
                    rect_axis = [rect(1),yini,rect(3),height_each_plot];
                    obj.new_axes('position',rect_axis);
                end
                
            end
        end
        
        function h = add_text(obj,h_ax,string,halign,valign)
            % halign:left,center,right
            % valign:middle,top,botton
            obj.current_ax(h_ax);
            xli = xlim;
            yli = ylim;
            switch halign
                case 'left'
                    x = xli(1);
                case 'center'
                    x = mean(xli);
                case 'right'
                    x = xli(2);
            end
            switch valign
                case 'bottom'
                    y = yli(1);
                case 'middle'
                    y = mean(yli);
                case 'top'
                    y = yli(2);
            end
            
            h = text(x,y,string,'horizontalalignment',halign,'verticalalignment',valign);
            
        end
        
        function new_axes(obj,varargin)
            position = [];
            for i = 1:length(varargin)
                if isequal(varargin{i},'position') || isequal(varargin{i},'Position')
                    position = varargin{i+1};
                end
            end
            if isempty(position)
                obj.h_ax(end+1) = axes();
            else
                obj.h_ax(end+1) = axes('position',position);
            end
        end
        
        function resize_vertical(obj,ax_num,vert_ratio,vert_separation)
            % obj.resize_vertical([1,2],[1,3])
            
            % vertical separation
            if nargin<4
                vert_separation = 0.13;
            end
            
            % get positions
            pos = [];
            for i=1:length(ax_num)
                pos = [pos; get(obj.h_ax(ax_num(i)),'Position')];
            end
            % resort
            [~,ind] = sort(pos(:,2));
            pos = pos(ind,:);
            ax_num = ax_num(ind);
            
            % new vert sizes
            V = pos(:,4);
            newV = V(:).*vert_ratio(:)/sum(vert_ratio);
            newV = newV * sum(V)/sum(newV);
            
            % new positions
            for i=1:length(ax_num)
                posit(i,:) = [pos(i,1),pos(i,2),pos(i,3),newV(i)];
                if i>1
                    posit(i,2) = posit(i-1,2)+posit(i-1,4)+vert_separation;
                end
            end
            % go
            for i=1:length(ax_num)
                set(obj.h_ax(ax_num(i)),'Position',posit(i,:));
            end
        end
        
        
        
        function resize_horizontal(obj,ax_num,hori_ratio,hori_separation)
            % obj.resize_horizontal([1,2],[1,3])
            
            % vertical separation
            if nargin<4
                hori_separation = 0.05;
            end
            
            n = length(ax_num);
            if length(hori_separation)==1
                hori_separation = ones(1,n)*hori_separation;
            end
            
            % get positions
            pos = [];
            for i=1:length(ax_num)
                pos = [pos; get(obj.h_ax(ax_num(i)),'Position')];
            end
            % resort
            [~,ind] = sort(pos(:,1));
            pos = pos(ind,:);
            ax_num = ax_num(ind);
            
            % new vert sizes
            V = pos(:,3);
            newV = V(:).*hori_ratio(:)/sum(hori_ratio);
            newV = newV * sum(V)/sum(newV);
            
            
            w = pos(end,1)+pos(end,3)-pos(1,1) - sum(hori_separation);
            wnew = sum(newV);
            newV = newV*w/wnew;
            
            % new positions
            for i=1:length(ax_num)
                posit(i,:) = [pos(i,1),pos(i,2),newV(i),pos(i,4)];
                if i>1
                    posit(i,1) = posit(i-1,1)+posit(i-1,3)+hori_separation(i-1);
                end
            end
            % go
            for i=1:length(ax_num)
                set(obj.h_ax(ax_num(i)),'Position',posit(i,:));
            end
        end
        
        function bracket_on_top(obj,ax_num,text,varargin)
            % example: p = publish_plot(2,2);p.shrink(1:4,0.9);p.bracket_on_top([1,2],'bla','height',0.1);
            
            font_size = 13;
            hparam = 0.05; % fraction of the plots height
            spacing_param = 0.04;
            for i=1:length(varargin)
                if isequal(varargin{i},'font_size')
                    font_size = varargin{i+1};
                elseif isequal(varargin{i},'height')
                    hparam = varargin{i+1};
                elseif isequal(varargin{i},'spacing')
                    spacing_param = varargin{i+1};
                end
            end
            
            for i=1:length(ax_num)
                pos(i,:) = get(obj.h_ax(ax_num(i)),'Position');
            end
            
            left = min(pos(:,1));
            right = max(pos(:,1)+pos(:,3));
            top = max(pos(:,2)+pos(:,4));
            
            h = hparam * max(pos(:,4)); % make it a fraction of the height
            spacing = spacing_param * max(pos(:,4)); % make it a fraction of the height
            
            X = [left,left];
            Y = clip([top+spacing,top+spacing+h],0,1);
            annotation('line',X,Y);
            
            X = [left,right];
            Y = clip([top+spacing+h,top+spacing+h],0,1);
            annotation('line',X,Y);
            
            X = [right,right];
            Y = clip([top+spacing,top+spacing+h],0,1);
            annotation('line',X,Y,'linewidth',0.5);
            
            center = (left + right)/2;
            w = 0.01 * length(text); % should find a better way
            posit = [center-w/2,top+spacing,w, 2*h];
            annotation('rectangle',posit,'FaceColor','w');
            ha = annotation('textbox',posit,'string',text);
            set(ha,'EdgeColor','w','horizontalalignment','center','verticalalignment','middle',...
                'FontSize',font_size,'fontangle','italic');
            
        end
        
        
        function align(obj)
            % align top, bottom, left, right
            % to do
        end
        function distribute(obj)
            % to do
        end
        
        function shrink(obj,ax_num,shrink_coeff_w,shrink_coeff_h,centerx_flag,centery_flag)
            % p.shrink([1,2],1,0.9,0,0);
            
            if nargin<4 || isempty(shrink_coeff_h)
                shrink_coeff_h = shrink_coeff_w;
            end
            if nargin<5 || isempty(centerx_flag)
                centerx_flag = 0;
            end
            if nargin<6 || isempty(centery_flag)
                centery_flag = 0;
            end
            
            for i = 1:length(ax_num)
                hax = obj.h_ax(ax_num(i));
                pos = get(hax,'position');
                w = pos(3)*shrink_coeff_w;
                h = pos(4)*shrink_coeff_h;
                if centerx_flag
                    dw = pos(3)*(1-shrink_coeff_w)/2;
                else
                    dw = 0;
                end
                if centery_flag
                    dh = pos(4)*(1-shrink_coeff_h)/2;
                else
                    dh = 0;
                end
                pos = [pos(1)+dw, pos(2)+dh, w, h];
                set(hax,'Position',pos);
            end
        end
        
        function copy_ax_to_another_ax(obj,ax_handle_from_num,ax_handle_to)
            
            if nargin<3 || isempty(ax_handle_to)
                hfig = figure();
                ax_handle_to = gca;
            end
            
            I = ax_handle_from_num;
            copyobj(allchild(obj.h_ax(I)),ax_handle_to);
            props = {'xlim','ylim','xtick','ytick','xticklabel','yticklabel',...
                'xscale','yscale','FontSize','color'}; % annoying solution
            for i=1:length(props)
                set(ax_handle_to,props{i},get(obj.h_ax(I),props{i}));
            end
            
            
        end
        
        function draggable(obj,h)
            
            right = 29;
            left  = 28;
            up    = 30;
            down  = 31;
            DELTA = [0.05 0.01];
            delta = DELTA(1);
            
            key_d = 100; % letra d: duplica axes
            key_n = 110;
            key_w = 119;
            
            var      = 'position';
            %             var      = 'size';
            char  = getkey(1);
            position = get(h,'position');
            figure(obj.h_fig)
            h_borde = [];
            while not(char==27)
                if char==key_n % next axes
                    obj.next();
                    h = gca;
                    position = get(h,'position');
                end
                if char==key_d
                    % duplicates axes
                    obj.new_axes('position',position);
                    h = obj.h_ax(end);
                    position = get(h,'position');
                end
                if char==key_w
                    ind = find(DELTA == delta);
                    if ind==length(DELTA)
                        delta = DELTA(1);
                    else
                        delta = DELTA(ind+1);
                    end
                end
                
                switch char
                    case 115 %"s" for size
                        var = 'size';
                        char  = getkey(1);
                    case 112 % "p" for position
                        var = 'position';
                        char  = getkey(1);
                end
                if isequal(var,'position')
                    switch char
                        case right
                            position(1) = position(1) + delta;
                        case up
                            position(2) = position(2) + delta;
                        case left
                            position(1) = position(1) - delta;
                        case down
                            position(2) = position(2) - delta;
                    end
                elseif isequal(var,'size')
                    switch char
                        case right
                            position(3) = position(3) + delta;
                        case up
                            position(4) = position(4) + delta;
                        case left
                            position(3) = position(3) - delta;
                        case down
                            position(4) = position(4) - delta;
                    end
                end
                
                set(h,'position',position)
                delete(h_borde);
                h_borde = borde_en_axis('ax',h);
                position = get(h,'position');
                
                char  = getkey(1);
                figure(obj.h_fig)
            end
            delete(h_borde);
            
        end
        
        function move(obj,haxes,where,delta)
            for i=1:length(haxes)
                pos = get(haxes(i),'Position');
                switch where
                    case 'up'
                        pos(2) = pos(2)+delta;
                    case 'down'
                        pos(2) = pos(2)-delta;
                    case 'right'
                        pos(1) = pos(1)+delta;
                    case 'left'
                        pos(1) = pos(1)-delta;
                end
                set(haxes(i),'Position',pos);
            end
            
        end
        
        function draggable_print(obj)
            for i=1:length(obj.h_ax)
                pos = get(obj.h_ax(i),'Position');
                str= ['set(p.h_ax(',num2str(i),'),''Position'',[',num2str(pos),'])'];
                disp(str)
            end
        end
        
        function copy_from_ax(obj,h_from,h_to,closefig_flag)
            % function copy_from_ax(obj,h_from,h_to)
            if nargin<3 || isempty(h_to)
                h_to = obj.h_ax(obj.active_axis);
            end
            
            if round(h_to)==h_to % is integer
                h_to = obj.h_ax(h_to);
            end
            
            if nargin<4 || isempty(closefig_flag)
                closefig_flag = 0;
            end
            
            copyobj(allchild(h_from),h_to);
            
            %             copyobj(h_from,h_to);
            props = {'xlim','ylim','xlabel','ylabel','yaxislocation','xscale','yscale','xtick','xticklabel'};
            for i=1:length(props)
                if isempty(strfind(version,'R2014b')) && (isequal(props{i},'xlabel') || isequal(props{i},'ylabel'))
                    gg = get(get(h_from,props{i}),'String');
                    set(get(h_to,props{i}),'String',gg);
                else
                    prop = get(h_from,props{i});
                    set(h_to,props{i},prop);
                end
            end
            
            % set active figure to the copied one
            figure(obj.h_fig);
            obj.current_ax(h_to);
            
            %             xli = get(h_from,'xlim');
            %             yli = get(h_from,'ylim');
            %             set(h_to,'xlim',xli);
            %             set(h_to,'ylim',yli);
            
            if closefig_flag
                close(get(h_from,'parent'));
            end
            
        end
        
        function saveas(obj,varargin)
            % save as .fig
            
            filename_fig = 'f.fig';
            if length(varargin)==1
                filename_fig = varargin{1};
            else
                for i = 1:length(varargin)
                    if isequal(varargin{i},'filename')
                        filename_fig = varargin{i+1};
                    end
                end
            end
            
            obj.data.filename_fig = filename_fig;
            saveas(obj.h_fig,filename_fig);
            obj.savename_fig = filename_fig;
            
            
        end
        
        
        function plot(obj,i,varargin)
            set(gcf,'CurrentAxes',obj.h_ax(i))
            plot(varargin{:},'LineWidth',0.5)
            set(obj.h_ax(i),'FontSize',6.8,'LineWidth',0.335)
            set(gca,'box','off')
            %obj.data.data(i) = varargin;
            %obj.data.fun(i) = 'plot';
        end
        
        function set_active(obj,varargin)
            obj.current_ax(obj,varargin{2:end});
        end
        
        function current_ax(obj,varargin)
            if length(varargin)==1
                i = varargin{1};
                if mod(i,1)==0
                    set(obj.h_fig,'CurrentAxes',obj.h_ax(i))
                else
                    set(obj.h_fig,'CurrentAxes',i)
                end
                %                 obj.active_axis = obj.h_ax(i);
                obj.active_axis = i;
            elseif length(varargin)==2
                i = varargin{1}; j = varargin{2};
                z = obj.data.n_col*(i-1)+j;
                set(obj.h_fig,'CurrentAxes',obj.h_ax(z))
                %                 obj.active_axis = obj.h_ax(z);
                obj.active_axis = z;
            else
                disp('error en input')
            end
        end
        
        function delete_ax(obj,h)
            if mod(h,1)==0
                ind = h;
                h   = obj.h_ax(ind);
            else
                ind = find(obj.h_ax==h);
            end
            delete(h); % deletes the axes
            obj.h_ax(ind) = [];
            
        end
        
        function next(obj)
            if isempty(obj.active_axis)
                set(obj.h_fig,'CurrentAxes',obj.h_ax(1))
                obj.active_axis = 1;
            else
                obj.active_axis = obj.active_axis+1;
                if obj.active_axis>length(obj.h_ax)
                    obj.active_axis = 1;
                end
                set(obj.h_fig,'CurrentAxes',obj.h_ax(obj.active_axis))
            end
        end
        
        function save(obj,varargin)
            %p.save('ancho',7,'filename','blabla','renderer','painters','resolution',150)
            filename = 'aaa';
            dire = pwd;
            renderer = 'painters';
            font_size = 10;
            color = 'rgb';
            %             resolution = num2str(150);
            %             ancho = 21.6;
            set_ancho = false;
            for i = 1:length(varargin)
                switch varargin{i}
                    case 'filename'
                        filename = varargin{i+1};
                    case 'ancho'
                        ancho_cm  = varargin{i+1};
                        set_ancho = true;
                    case 'renderer'
                        renderer = varargin{i+1};
                    case 'fontsize'
                        font_size = varargin{i+1};
                    case 'color'
                        color = varargin{i+1};
                        %                     case 'resolution'
                        %                         resolution = num2str(varargin{i+1});
                end
            end
            
            % trying to remove annoying white spaces
            for i=1:length(obj.h_ax)
                set(obj.h_ax(i),'LooseInset',get(obj.h_ax(i),'TightInset'));
            end
            
            exportfig(obj.h_fig,filename,'Format','eps','Preview','none','Width',ancho_cm,'Renderer',renderer,'Resolution',300,'FontMode','fixed',...
                'FontSize',font_size,'LineMode','fixed','LineWidth',0.5,'Color',color);
            
            % NOT WORKING
            
            
%             if set_ancho
%                 set_figure_size(ancho);
%             end
%             
%             obj.savename = filename;
%             obj.savedir = dire;
%             old_dir = pwd;
%             cd(dire)
%             %figName = [filename,'.eps'];
%             figName = [filename,'.pdf'];
%             % eval(['print -depsc2 -tiff -',renderer,' -r',resolution,' ',figName])
%             eval(['print -dpdf -',renderer,' ',figName])
%             cd(old_dir)
        end
        
        function font_size(obj,axnum,fontsize)
            set(findall(obj.h_ax(axnum),'-property','FontSize'),'FontSize',fontsize)
        end
        
        function format(obj,varargin)
            %FontSize = 6.8;
            %LineWidthAxes = 0.335;
            FontSize = 18;
            LineWidthAxes = 0.5;
            LineWidthPlot = [];
            MarkerSize = [];
            interpreter = '';
            for i=1:length(varargin)
                if isequal(varargin{i},'presentation')
                    FontSize = 25;
                    LineWidthAxes = 2;
                    LineWidthPlot = 2.5;
                elseif isequal(varargin{i},'invert_colors') && obj.invert_colors==0
                    obj.invert_colors = 1;
                elseif isequal(varargin{i},'FontSize')
                    FontSize = varargin{i+1};
                elseif isequal(varargin{i},'LineWidthPlot')
                    LineWidthPlot = varargin{i+1};
                elseif isequal(varargin{i},'LineWidthAxes')
                    LineWidthAxes = varargin{i+1};
                elseif isequal(varargin{i},'MarkerSize')
                    MarkerSize = varargin{i+1};
                elseif isequal(lower(varargin{i}),'interpreter')
                    interpreter = varargin{i+1};
                end
            end
            
            if not(isempty(LineWidthPlot))
                for j=1:length(obj.h_ax)
                    ch = get(obj.h_ax(j),'Children');
                    ch = findall(ch,'Type','Line');
                    set(ch,'LineWidth',LineWidthPlot)
                end
            end
            
            if not(isempty(MarkerSize))
                for j=1:length(obj.h_ax)
                    ch = get(obj.h_ax(j),'Children');
                    ch = findall(ch,'Type','Line');
                    set(ch,'MarkerSize',MarkerSize)
                end
            end
            

            all_text = findall(obj.h_fig,'Type','text');
            if ~isempty(interpreter)
                set(all_text,'FontSize',FontSize,'Interpreter',interpreter);
            else
                set(all_text,'FontSize',FontSize);
            end
            if isfield(obj.text,'number_plot') && ~isempty(obj.text.number_plot)
                set(obj.text.number_plot,'FontSize',FontSize);
            end
            for i=1:length(obj.h_ax)
                set(obj.h_ax(i),'FontSize',FontSize,'LineWidth',LineWidthAxes,'box','off')
            end
            
            set(obj.h_ax,'color','none'); % so it removes the bk color
            
            
            if obj.invert_colors==1
                a = findall(obj.h_fig);
                w = findobj(a,'Color','w');
                b = findobj(a,'Color','k');
                set(w,'Color','k');
                set(b,'Color','w');
                
                for j=1:length(obj.h_ax)
                    set(obj.h_ax(j),'Ycolor','w')
                    set(obj.h_ax(j),'Xcolor','w')
                    set(obj.h_ax(j),'Color','none') % Para fondo
                    %transparente
                end
                
                set(obj.h_fig,'Color','none') % Para fondo transparente
                set(obj.h_fig,'InvertHardcopy','off')
            end
            
        end
        
        function number_the_plots(obj,action)
            if nargin==1
                action = 'show';
            end
            if isequal(action,'show')
                obj.text.number_plot = [];
                for i = 1:length(obj.h_ax)
                    obj.current_ax(i);
                    xli = xlim;
                    yli = ylim;
                    obj.text.number_plot(i) = text(sum(xli)/2,sum(yli)/2,num2str(i));
                end
            elseif isequal(action,'hide')
                delete(obj.text.number_plot);
            end
        end
        
        function h = letter_the_plots(obj,action,order)
            if nargin<2 || isempty(action)
                action = 'show';
                order = 1:length(obj.h_ax);
            end
            
            h = [];
            if isequal(action,'show')
                Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
                obj.text.letter_plot = [];
                for i = 1:length(order)
                    obj.current_ax(order(i));
                    posit = get(gca,'position');
                    delta = posit(3)/20;
                    x = posit(1) - delta;
                    y = posit(2) + posit(4) + delta;
                    w = 0.05;
                    obj.text.number_plot(i) = annotation('textbox',[x,y,w,w],'string',Alphabet(i),'FontWeight',...
                        'bold','LineStyle','none');
                    
                    % obj.text.number_plot(i) = text(sum(xli)/2,sum(yli)/2,num2str(i));
                end
                h = obj.text.number_plot;
            elseif isequal(action,'hide')
                delete(obj.text.number_plot);
            end
        end
        
        
        function shadow_plot(obj,i,varargin)
            set(gcf,'CurrentAxes',obj.h_ax(i))
            [errorPatch,dataLine] = niceBars(varargin{:});
            set(obj.h_ax(i),'FontSize',6.8,'LineWidth',0.335)
            set(gca,'box','off')
            obj.data(i).data = varargin;
            obj.data(i).fun = 'niceBars';
        end
        
        function rainbow_plot(obj,i,x,y,varargin)
            set(gcf,'CurrentAxes',obj.h_ax(i))
            rplot(x,y,varargin{:});
            set(obj.h_ax(i),'FontSize',6.8,'LineWidth',0.335)
            set(gca,'box','off')
            obj.data(i).data = {x,y,varargin{:}};
            obj.data(i).fun = 'rainbow_plot';
        end
        
        function legend_save(obj)
            fid = fopen([obj.savedir,'/',['legend_',obj.savename,'.txt']],'w');
            fprintf(fid,'%s\n','\begin{figure}[t]');
            fprintf(fid,'%s\n','\begin{center}');
            fprintf(fid,'%s\n',['\includegraphics{',obj.savename,'.eps}']);
            fprintf(fid,'%s\n',['\caption{',obj.legend,'}']);
            fprintf(fid,'%s\n',['\label{',obj.savename,'}']);
            fprintf(fid,'%s\n','\end{center}');
            fprintf(fid,'%s\n','\end{figure}');
            fclose(fid);
        end
        
        function set_figure_size(obj,xSize,filename)
            % function set_figure_size(hfig,xSize,filename_optional)
            % sets figure size, for printing. xSize in centimeters
            
            hfig = obj.h_fig;
            dosave = true;
            if nargin<3 || isempty(filename)
                dosave = false;
            end
            
            % MEDIDAS LETTER: xSize = 21.6; ySize = 27.9;
            pos = get(hfig,'Position');
            YoverX = pos(4)/pos(3);
            
            set(hfig,'PaperUnits','centimeters','PaperPosition',[1 1 xSize YoverX*xSize])
            
            if (dosave)
                saveas(hfig,filename,'epsc');
            end
            
        end
        
        
        function displace_ax(obj,h_ax,delta_pos,dim)
            
            if all(mod(h_ax,1)==0)
                h_ax = obj.h_ax(h_ax);
            end
            
            % hack to deal with matlab's annoyance
            H = setdiff(obj.h_ax,h_ax);
            for i=1:length(H)
                pos = get(H(i),'Position');
                newpos = pos;
                set(H(i),'Position',newpos);% same
            end
            
            % go
            for i=1:length(h_ax)
                pos = get(h_ax(i),'Position');
                newpos = pos;
                newpos(dim) = pos(dim) + delta_pos;
                set(h_ax(i),'Position',newpos);
            end
            
        end
        
        
        
        function displace_obj(obj,h_obj_to_displace,delta_pos,dim)
            % displaces any obj that has a Position property
            
            for i=1:length(h_obj_to_displace)
                pos = get(h_obj_to_displace(i),'Position');
                newpos = pos;
                newpos(dim) = pos(dim) + delta_pos;
                set(h_obj_to_displace(i),'Position',newpos);
            end
            
        end
        
        
        
        function hax = handle(obj,i,j)
            ind = obj.data.n_col*(i-1)+j;
            hax = obj.h_ax(ind);
            
        end
        
        
        function plot_for_presentation
        end
        
        function save_for_presentation
        end
    end
end
