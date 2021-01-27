function f = findclose(v,val)
%function f = findclose(v,val)
%devuelve la posici�n del valor de v m�s cercano a val
if length(val)==1
    f = find(abs(v-val) == min(abs(v-val)));
else
    f = nan(size(val));
    for i=1:length(val)
        if not(isnan(val(i)))
%             f(i) = find(abs(v-val(i)) == min(abs(v-val(i))));
            f(i) = find(abs(v-val(i)) == min(abs(v-val(i))),1);
        end
    end
end


