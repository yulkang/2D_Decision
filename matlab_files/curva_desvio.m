%%[X,T,siz]=curva_desvio(DEP,INDEP,filt,plotflag)
% Toma una variable dependiente (por ejemplo el tiempo de respuesta) y una
% varaible independiente (por ejemplo el SOA) y genera una curva del desvio. 
% Toma ademas un filtro (por ejemplo si uno quiere solo para las
% correctas) y como cuarta opcion si plotear o no la curva media.
% Los dos ultimos parametros son opcionales, pero para plotear (4
% parametro) hay que llamarla con filtro = 0.
% Devuelve X (todos los valores de la variable indpendiente) e Y (los
% valores medios de la variable dependiente en funcion de X).



function [X,T,siz]=curva_desvio(DEP,INDEP,filt,plotflag)
X=unique(INDEP);

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
   T(i,:)=nanstd(DEP(INDEP==X(i),:),[],1);
   siz(i,:)=sum(not(isnan(DEP(INDEP==X(i),:))),1);
end

if nargin > 3 % A reference to the plot
   if plotflag==1
       plot(X,T,'.-');
   elseif plotflag==2
       plot(X,T,'.');
   end
end