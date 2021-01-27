[theta,combs] = get_best_params(1);


rows = {'1','2','3','6','7','8','9','10',...
    '11','12','13',...
    '6','7','8','9','10',...
    '11','12','13'};
params = redondear(theta,3);
params(:,1:end-1) = redondear(params(:,1:end-1),2);
% T.Properties.RowNames = rows;
Task = table({'Eye RT','','','Monomanual','','','','','','','','Bimanual','','','','','','',''}',rows',...
    'VariableNames',{'Task','Subject ID'});

columns = {'\kappa_m','B_m','a_m','d_m','c0_m','\kappa_c','B_c','a_c','d_c','c0_c','\mu_{nd}','\sigma_{nd}'};
T = array2table(params,'VariableNames',columns);

T = [Task, T];


table2latex(T,'table_serial_model');

%       LastName = {'Sanchez';'Johnson';'Li';'Diaz';'Brown'};             %
%       Age = [38;43;38;40;49];                                           %
%       Smoker = logical([1;0;1;0;1]);                                    %
%       Height = [71;69;64;67;64];                                        %
%       Weight = [176;163;131;133;119];                                   %
%       T = table(Age,Smoker,Height,Weight);                              %
%       T.Properties.RowNames = LastName;                                 %
%       table2latex(T);