addpath(genpath('../matlab_files'))

load('fits','v_theta');
rows = {'6','7','8','9','10','11','12','13'};
columns = {'Inter switch interval (s)','p_{motion 1st}','\mu_{nd}'};
v_theta(:,1) = v_theta(:,1)*1.5;
params = redondear(v_theta,2);
T = table(params(:,1),1-params(:,2),params(:,3),'VariableNames',columns);
T.Properties.RowNames = rows;
table2latex(T,'table_switching');

save('Table3.mat', 'T')

%       LastName = {'Sanchez';'Johnson';'Li';'Diaz';'Brown'};             %
%       Age = [38;43;38;40;49];                                           %
%       Smoker = logical([1;0;1;0;1]);                                    %
%       Height = [71;69;64;67;64];                                        %
%       Weight = [176;163;131;133;119];                                   %
%       T = table(Age,Smoker,Height,Weight);                              %
%       T.Properties.RowNames = LastName;                                 %
%       table2latex(T);