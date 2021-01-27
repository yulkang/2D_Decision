classdef VisitableTreeProps < VisitableTree
    % Offers dependent properties as useful shortcuts.
    % Having the names on VisitableTree would make it incompatible with classes
    % that need those names for other uses.
    %
    % 2016 (c) Yul Kang. hk2699 at columbia dot edu.
    properties (Dependent)
        root
        parent
        children
    end
    methods
        function v = get.root(Tree)
            v = Tree.get_root;
        end
        function v = get.parent(Tree)
            v = Tree.get_parent;
        end
        function v = get.children(Tree)
            [~,~,v] = Tree.get_children;
        end
        function set.root(Tree, v)
            Tree.set_root(v);
        end
        function set.parent(Tree, v)
            Tree.set_parent(v);
        end
    end
end