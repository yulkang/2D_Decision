classdef VisitableTree < DeepCopyable
    % A tree visitable by VisitorToTree.
    %
    % USAGE:
    %   Tree = VisitableTree(name, [parent])
    %   Tree.add_children({child_tree1, child_tree2, ...})
    % 
    % Get methods:
    %   get_root/parent/children
    %   get_child(name)
    %
    % Set methods:
    %   add_children(struct_with_children_as_fields)
    %   add_child(child_tree, [name, auto_add_name=true])
    %
    % Test with [Tree, name_struct] = VisitableTree.test
    %
    % See also: VisitorToTree
    properties (SetAccess = protected)
        root_ = [];
        parent_ = [];
        children_ = struct;
        
        % name_ : a nonempty string.
        % Before adding a parent, a tree is the root of itself.
        name_ = 'root'; 
    end
    methods
        %% Common interface
        function Tree = VisitableTree(varargin)
            % Tree = VisitableTree(name, [parent])
            %
            % Before adding a parent, a tree is the root of itself.
            Tree.set_root_(Tree); 
            Tree.add_deep_copy({'root_', 'parent_', 'children_'}); % New syntax
            if nargin >= 1
                Tree.init_tree(varargin{:});
            else
%                 warning('Name unset for a VisitableTree node! Defaulting to ''root''...');
            end
        end
        function init_tree(Tree, name, parent)
            % init_tree(Tree, name, [parent])
            Tree.set_name_(name);
            if nargin >= 3
                Tree.set_parent(parent);
            end
        end
        function add_children_props(Tree, props)
            % Add properties as children
            %
            % add_children_props(Tree, children)
            if nargin < 2, props = {}; end
            if ischar(props), props = {props}; end
            props = props(:);
            assert(all(cellfun(@ischar, props)));
            for prop = props'
                prop1 = Tree.(prop{1});
                
                Tree.add_child(prop1, prop{1});
                Tree.add_deep_copy(prop{1});
            end         
        end
        function varargin2children_props(Tree, varargin)
            S = varargin2S(varargin);
            varargin2props(Tree, S);
            Tree.add_children_props(fieldnames(S));
        end
        function add_children(Tree, children)
            % add_children(Tree, obj_struct)
            if isa(children, 'VisitableTree')
                children = {children};
            end
            if iscell(children)
                for child = children(:)'
                    Tree.add_child(child{1});
                end
            elseif isstruct(children)
                for child_name = fieldnames(children)'
                    child = children.(child_name{1});
                    Tree.add_child(child, child_name{1});
                end
            else
                error('children must be a VisitableTree, cell, or struct!');
            end
        end
        function tf = has_children(Tree)
            tf = ~isempty(fieldnames(Tree.children_));
        end
        function tf = is_root(Tree)
            tf = Tree == Tree.get_root;
        end
        function [obj, name] = get_root(Tree)
            % [obj, name] = get_root(Tree)
            obj = Tree.root_;
            if isempty(obj)
                obj = Tree;
            end
            name = obj.get_name;
        end
        function [obj, name] = get_parent(Tree)
            % [obj, name] = get_parent(Tree)
            obj = Tree.parent_;
            if isempty(obj)
                name = '';
            else
                name = obj.get_name;
            end
        end
        function [objs, names, obj_struct] = get_children(Tree)
            % [objs, names, obj_struct] = get_children(Tree)
            obj_struct = Tree.children_;
            
            assert(isscalar(obj_struct)); % To ensure that objs is 1 x fields
            
            % permute so that size(objs) == [1, fields]
            objs = struct2cell(Tree.children_)'; 
            names = fieldnames(Tree.children_)';
        end
        function obj = get_child(Tree, name)
            obj = Tree.children_.(name);
        end
        function name = get_name(Tree)
            name = Tree.name_;
        end
        %% Elementary operation: add/remove child/parent
        % Since there can be multiple children but only one parent,
        % call set/remove_parent from add/remove_child/ren.
        function add_child(Tree, child, name, varargin)
            % add_child(Tree, child, name, ...)
            S = varargin2S(varargin, {
                'auto_add_name', true
                });
            
            if nargin < 3
                name = '';
            end

            if iscell(child)
                for ii = 1:numel(child)
                    Tree.add_child( ...
                        child{ii}, ...
                        sprintf('c%d_%s', name, ii), ...
                        varargin{:});
                end
                
            elseif isstruct(child)
                fs = fieldnames(child);
                for ii = 1:numel(fs)
                    name1 = fs{ii};
                    child1 = child.(name1);
                    
                    Tree.add_child( ...
                        child1, ...
                        sprintf('s%d_%s', name1, ii), ...
                        varargin{:});
                end
                
            elseif ~isscalar(child)
                for ii = 1:numel(child)
                    Tree.add_child( ...
                        child(ii), ...
                        sprintf('m%d_%s', name, ii), ...
                        varargin{:});
                end
                
            else % scalar child.
                assert(isa(child, 'VisitableTree'));
                
                if isempty(name)
                    name = child.get_name;
                elseif S.auto_add_name
                    child.set_name_(name);
                else
                    assert(strcmp(name, child.get_name));
                end
                Tree.children_.(name) = child;
                child.set_parent(Tree);
            end
        end
        function add_child_unit(Tree, child, name, varargin)
        end
        function remove_child(Tree, child_or_name)
            % remove_child(Tree, child_name|child_obj)
            child = Tree.parse_child(child_or_name);
            child.remove_parent;
        end
        function remove_all_children(Tree)
            children = Tree.get_children;
            for child = children(:)'
                Tree.remove_child(child{1});
            end
        end
        function set_parent(Tree, parent)
            % Set parent and and update descendents' root.
            % set_parent(Tree, parent)
            if isempty(parent)
                Tree.remove_parent;
            else
                Tree.set_parent_(parent);
                Tree.update_root(parent.get_root);
            end
        end
        function remove_parent(Tree)
            % Remove parent and update descendents' root.
            % remove_parent(Tree, update_root=true)
            parent = Tree.get_parent;
            Tree.set_parent_([]);
            
            if ~isempty(parent)
                % Remove child of the parent
                parent.remove_child_(Tree);
            end
            
            % The Tree is the root of itself.                    
            Tree.update_root(Tree);
        end
        %% Internal
        function update_root(Tree, root)
            % Update self & descendents' root.
            Tree.set_root(root);
            children = Tree.get_children;
            if ~isempty(children)
                for child = children(:)'
                    child{1}.update_root(root);
                end
            end
        end
        function child = parse_child(Tree, child_or_name)
            if isa(child_or_name, 'VisitableTree')
                child = child_or_name;
            elseif ischar(child_or_name)
                child = Tree.get_child(child_or_name);
            end
        end
        function set_root(Tree, root)
            % Extensible in subclasses.
            Tree.set_root_(root);
        end
    end
    methods (Sealed)
        %% Low-level set methods without side effects
        function set_root_(Tree, root)
            assert(isempty(root) || isa(root, 'VisitableTree'));
            Tree.root_ = root;
        end
        function set_name_(Tree, name)
            assert(is_valid_variable_name(name));
            Tree.name_ = name;
        end
        function set_children_(Tree, children)
            assert(isstruct(children));
            Tree.children_ = children;
        end
        function set_parent_(Tree, parent)
            % Internal. Set parent_ without updating root or other side effects.
            assert(isempty(parent) || isa(parent, 'VisitableTree'));
            Tree.parent_ = parent;            
        end
        function remove_child_(Tree, child)
            % Internal. Remove child without updating root or other side effects.
            assert(isa(child, 'VisitableTree'));
            Tree.children_ = rmfield(Tree.children_, child.get_name);
        end
    end
    %% Saving-related methods.
    methods
        function Tree = obj2struct(Tree0)
            Tree = copy(Tree0);
            Tree.empty_links_;
            Tree = bml.oop.copyprops(struct, Tree, ...
                'props_to_skip', {'root'});
        end
        function empty_links_(Tree)
            Tree.set_root_([]);
            Tree.set_parent_([]);
            Tree.empty_children_;
        end
        function empty_children_(Tree)
            % Set all children fields as empty, as needed when saving as struct.
            % Also attempts to set same-named properties as empty if present.
            children = Tree.children_;
            names = fieldnames(children);
            for name = names(:)'
                if isprop(Tree, name{1}) ...
                        && Tree.(name{1}) == Tree.children_.(name{1})
                    Tree.(name{1}) = [];
                end
                Tree.children_.(name{1}) = [];
            end
        end
    end
end