classdef PropFileNameTree ...
        < bml.oop.PropFileName ...
        & bml.oop.VisitableTreeProps
% Combines all decendents' file_fields and S0_file.
methods
    function S0_file = get_S0_file(W)
        S0_file = W.get_S0_file@bml.oop.PropFileName;
        for child_name = fieldnames(W.children)'
            child = W.children.(child_name{1});
            S0_file = copyFields(S0_file, child.S0_file);
        end
    end
    function fs = get_file_fields(W)
        fs = W.get_file_fields0;
        for child_name = fieldnames(W.children)'
            child = W.children.(child_name{1});
            fs = union_general(fs, child.file_fields, ...
                'stable', 'rows');
        end
    end
    function C = get_file_fields0(~)
        C = cell(0,2); % Placeholder for {name_orig, name_short}
    end
end
end