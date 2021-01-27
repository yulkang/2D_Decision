function [X,T,STD_ERR,siz] = curva_media(DEP,INDEP,filt,plotflag)

%%function [t,y,STD_ERR]=curva_media(DEP,INDEP,filt,plotflag)
% Toma una variable dependiente (por ejemplo el tiempo de respuesta) y una
% varaible independiente (por ejemplo el SOA) y genera una curva del valor
% medio. Toma ademas un filtro (por ejemplo si uno quiere solo para las
% correctas) y como cuarta opcion si plotear o no la curva media.
% Devuelve X (todos los valores de la variable indpendiente) e Y (los
% valores medios de la variable dependiente en funcion de X).
X = unique(INDEP);

if not(max(size(INDEP))==size(DEP,1)) && max(size(INDEP))==size(DEP,2)
    DEP = DEP'; %trials en las filas
end

if nargin > 2 % A filter has been called
   if not(isempty(filt))
      DEP=DEP(filt,:);INDEP=INDEP(filt);
      X=unique(INDEP); %por si el filtro deja afuera algunos valores de X
   end
end

for i=1:length(X)
   T(i,:)=nanmean(DEP(INDEP==X(i),:),1);
end


if all(ismember(DEP(~isnan(DEP)),[0,1]))
    % prop errores
    for i=1:length(X)
       siz(i,:)=sum(not(isnan(DEP(INDEP==X(i),:))),1);
    end
    STD_ERR = sqrt(T.*(1-T)./siz);
else
    % gauss errors
    for i=1:length(X)
       STD(i,:) = nanstd(DEP(INDEP==X(i),:),[],1);
       siz(i,:) = sum(not(isnan(DEP(INDEP==X(i),:))),1);
    end
    STD_ERR = STD./sqrt(siz);
end


if nargin > 3 % A reference to the plot
   if plotflag>0
       if isvector(T)
           if plotflag==1
                plot(X,T,'.-');
           elseif plotflag==2
               %errorbar(X,T,STD_ERR,'.-');
               
%                co = get(gca,'ColorOrder');
%                coi = mod ( get(gca,'ColorOrderIndex') , size(co,1) );
               
               terrorbar(X,T,STD_ERR,'.-');
%                set(gca,'ColorOrderIndex',coi+1);
               hold all
%                errorbar_noh(X,T,STD_ERR);
            elseif plotflag==3 % just the error bars
                terrorbar(X,T,STD_ERR,'.','LineStyle','none');
           end
       else
           if plotflag==1
               plot(T','.-');
           elseif plotflag==2
               %
               hold all
               colores = rainbow_colors(size(T,1));
               for i=1:size(T,1)
                   inds = ~isnan(T(i,:));
                   tt = 1:size(T,2);
                   niceBars(tt(inds), T(i,inds), STD_ERR(i,inds), colores(i,:), 0.5);
               end
           end
       end
   end
end