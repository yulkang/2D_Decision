classdef PropFileName < matlab.mixin.Copyable
% bml.oop.PropFileName
%
% Properties:
%
% file_fields = {
%   'prop_name', 'name_on_filename'
%   'prop_name', '' % empty to keep original
%   };
% file_mult = {
%   'prop_name', multiply_from_prop2filename
%
% 2016 (c) Yul Kang. hk2699 at columbia dot edu.
properties % (Access = private)
    file_fields_ = {
        };    
    file_mult_ = {
        };
    
    S0_file_ = struct; % Fields in addition to properties
    S_file_ = []; % Overrides S_file if not empty
end
properties (Dependent)
    % file_fields = {
    %   'prop_name', 'name_on_filename'
    %   'prop_name', '' % empty to keep original
    %   };
    file_fields
    
    % file_mult = {
    %   'prop_name', multiply_from_prop2filename
    file_mult
    
    S0_file
    S_file
    
    file_name
end
properties
    root_data_dir = '../../Data';
    file_name_ = '';
end
%% Names from multiple files
methods
    function [files, names, S_files, S0_files] = get_files_from_S0s(PFile0, ...
            S0s, remove_fields)
        % [files, names, S_files, S0_files] = get_files_from_S0s(PFile0, ...
        %     S0s, remove_fields)
        
        n = numel(S0s);
        siz = size(S0s);
        files = cell(siz);
        names = cell(siz);
        S_files = cell(siz);
        S0_files = cell(siz);
        
        if ~exist('remove_fields', 'var')
            remove_fields = {};
        end
        if ~iscell(S0s), S0s = num2cell(S0s); end
        
        for ii = 1:n
            PFile = feval(class(PFile0));
            C = S2C(S0s{ii});
            bml.oop.varargin2props(PFile, C, true);
            
            [S_files{ii}, S0_files{ii}] = ...
                PFile.get_S_file({}, remove_fields);
            files{ii} = PFile.get_file({}, remove_fields);
            names{ii} = PFile.get_file_name({}, remove_fields);
        end
    end
    function [file, name, S0_file, files, names, S_files, S0_files] = ...
            get_file_compare_S0s(PFile, S0s, remove_fields)
        % [file, name, S0_file, files, names, S_files, S0_files] = ...
        %     get_file_compare_S0s(PFile, S0s, remove_fields)        
        
        if ~exist('remove_fields', 'var')
            remove_fields = {};
        end
        if ~iscell(S0s), S0s = num2cell(S0s); end
        
        [files,names,S_files,S0_files] = PFile.get_files_from_S0s(S0s, remove_fields);
        fields = fieldnames(S0_files{1})';
        
        S0_file = PFile.get_S0_file;
        n = numel(S0s);
        for f = fields
            vs = cell(1, n);
            for ii = 1:n
                if isfield(S0s{ii}, f{1})
                    vs{ii} = S0s{ii}.(f{1});
                elseif isfield(S0s{ii}, f{1})
                    vs{ii} = S0_files{ii}.(f{1});
                else
                    vs{ii} = [];
                end
            end
            if all(cellfun(@(v) isequal(v, vs{1}), vs(2:end)))
                S0_file.(f{1}) = vs{1};
            else
                S0_file.(f{1}) = bml.matrix.unique_general(vs);
            end
        end
        
        S0_file = rmfield(S0_file, remove_fields);
        [file, name] = PFile.get_file_from_S0(S0_file, remove_fields);
    end
end
%% Files
properties
    subdir_ = '';
end
properties (Dependent)
    subdir
end
methods
    function [file, name] = get_file_from_S0(PFile, S0, add_fields, remove_fields)
        % get_file_from_S0(PFile, S0, add_fields, remove_fields)
        if ~exist('add_fields', 'var'), add_fields = struct; end
        if ~exist('remove_fields', 'var'), remove_fields = {}; end
        
        add_fields = varargin2S( ...
            add_fields, ...
            PFile.convert_to_S_file(S0));
        
        [file, name] = PFile.get_file(add_fields, remove_fields);
    end
    function [file, name] = get_file(PFile, add_fields, remove_fields, subdir)
        % [file, name] = get_file(PFile, add_fields, remove_fields, subdir)
        if ~exist('add_fields', 'var'), add_fields = struct; end
        if ~exist('remove_fields', 'var'), remove_fields = {}; end
        if ~exist('subdir', 'var'), subdir = PFile.subdir; end
        
        name = PFile.get_file_name(add_fields, remove_fields);
        file = fullfile(PFile.root_data_dir, subdir, name);
    end
    function d = get_dir(PFile)
        d = fullfile(PFile.root_data_dir, PFile.subdir);
    end
    function subdir = get.subdir(PFile)
        subdir = PFile.get_subdir;
    end
    function subdir = get_subdir(PFile)
        if isempty(PFile.subdir_)
            subdir = class(PFile);
        else
            subdir = PFile.subdir_;
        end
    end
    function set.subdir(PFile, subdir)
        PFile.set_subdir(subdir);
    end
    function set_subdir(PFile, subdir)
        PFile.subdir_ = subdir;
    end
    function name = get_file_name(PFile, add_fields, remove_fields)
        if ~exist('add_fields', 'var'), add_fields = struct; end
        if ~exist('remove_fields', 'var'), remove_fields = {}; end
        
        S2s = bml.str.Serializer;
        
        if ~isempty(PFile.file_name_)
            name = PFile.file_name_;
            
            if ~isequal(add_fields, struct) || ~isempty(remove_fields)
                S_file = S2s.convert(name);
                S_file = varargin2S(add_fields, S_file);
                S_file = rmfield(S_file, remove_fields);
                name = S2s.convert(S_file);
            end
            return;
        end
        
        S_file = PFile.get_S_file(add_fields, remove_fields);
        name = S2s.convert(S_file);
        
        % Prevent erroneous behavior regarding extensions.
        name = strrep(name, '.', '^'); 
    end
    function set.file_name(PFile, name)
        PFile.set_file_name(name);
    end
    function set_file_name(PFile, name)
        PFile.file_name_ = name;
    end
    function S_file = get.S_file(PFile)
        S_file = PFile.get_S_file;
    end
    function [S_file, S0_file] = get_S_file(PFile, add_fields, remove_fields)
        if ~exist('add_fields', 'var'), add_fields = struct; end
        if ~exist('remove_fields', 'var'), remove_fields = {}; end
        
%         if bml.matrix.is_cc(PFile.file_fields)
%             file_fields0 = bml.matrix.cc2cmat(PFile.file_fields);
%             file_fields = file_fields0(:, [1 2]);
%             file_mult = file_fields0(:, [1 3]);
%         else
            file_fields = PFile.file_fields;
%             file_mult = PFile.file_mult;
%         end

        S0_file = PFile.get_S0_file;
        
        if ~isempty(PFile.S_file_)
            S_file = PFile.S_file_;
        else
            if isempty(file_fields)
                S_file = struct;
            else
                [~, ia1] = setdiff(file_fields(:,1), remove_fields(:), 'stable');
                [~, ia2] = setdiff(file_fields(:,2), remove_fields(:), 'stable');
                ia = intersect(ia1, ia2, 'stable');
                file_fields = file_fields(ia, :);

                S2s = bml.str.Serializer;
                S_file = S2s.field_strrep(S0_file, file_fields);

    %             if ~isempty(file_mult)
    %                 [~, ia] = setdiff(file_mult(:,1), remove_fields(:), 'stable');
    %                 file_mult = file_mult(ia, :);
    %             end
    %         
    %             [S_file, S0_file] = bml.str.Serializer.convert_to_S_file(PFile, ...
    %                 file_fields, ...
    %                 'mult', file_mult);
            end
        end
        
        S_file = varargin2S(add_fields, S_file);
    end
    function S0_file = get.S0_file(PFile)
        S0_file = PFile.get_S0_file;
    end
    function S0_file = get_S0_file(PFile)
        S0_file = struct;
        fs = PFile.file_fields;
        
        n = size(fs, 1);
        S0_file_ = PFile.S0_file_;
        for ii = 1:n
            fs1 = fs{ii,1};
            if isprop(PFile, fs1)
                S0_file.(fs1) = PFile.(fs1);
            elseif isfield(S0_file_, fs1)
                S0_file.(fs1) = S0_file_.(fs1);
            else
                % Skip.
            end
        end
%         S0_file = copyprops(struct, PFile, 'props', fs(:,1));
    end
    function S0_file = convert_from_S_file(PFile, S_file)
        % TODO: Make consistent with get_S_file regarding cc and cmat
        S0_file = bml.str.Serializer.convert_from_S_file(S_file, ...
            PFile.file_fields, 'mult', PFile.file_mult);
    end
    function [S_file, S] = convert_to_S_file(PFile, S0_file, varargin)
        % S_file = convert_to_S_file(PFile, S0_file, varargin)
        %
        % 'file_fields', PFile.file_fields
        % 'mult', PFile.file_mult
        % 'leave_unmatched', true % false
        %
        % TODO: Make consistent with get_S_file regarding cc and cmat
        S = varargin2S(varargin, {
            'file_fields', {}
            'mult', PFile.file_mult
            'leave_unmatched', true % false
            });
        S.file_fields = bml.args.S2C2(varargin2S( ...
            varargin2S(S.file_fields), ...
            varargin2S(PFile.file_fields) ...
            ));
        
        if ~isscalar(S0_file)
            S_file = arrayfun(@(S1) bml.str.Serializer.convert_to_S_file( ...
                S1, S.file_fields, 'mult', S.mult), S0_file);
        else
            S_file = bml.str.Serializer.convert_to_S_file(S0_file, ...
                S.file_fields, 'mult', S.mult);
        end
        
        if S.leave_unmatched
            for ii = 1:numel(S0_file)
                fs = setdiff(fieldnames(S0_file), ...
                             S.file_fields(:,1), 'stable');
                for f1 = fs(:)'
                    S_file(ii).(f1{1}) = S0_file(ii).(f1{1});
                end
            end
        end
    end
end
%% Get file name of a batch from a table or a dataset
methods
    function [file, S_files, S0_files] = get_file_batch(PFile, S_batch, varargin)
        % [file, S_files, S0_files] = get_file_batch(PFile, S_batch, varargin)
        S = varargin2S(varargin, {
            'add_from_columns', {}
            'add_fields', {}
            });
        
        fs = PFile.get_file_fields;
        fs = [fs(:,1)', S.add_from_columns(:)'];
        
        for f = fs
            c = S_batch.(f{1});
            if isempty(c)
                continue;
            end
            if iscell(c) && all(cellfun(@ischar, c))
                strs = c;
            elseif iscell(c)
                strs = cellfun(@bml.str.Serializer.char, c, ...
                    'UniformOutput', false);
            elseif ischar(c) && ~isa(S_batch, 'table') && ~isa(S_batch, 'dataset')
                strs = {c};
            else
                strs = arrayfun(@bml.str.Serializer.char, c, ...
                    'UniformOutput', false);
            end
                
            S0_files.(f{1}) = unique(strs);
        end
        S0_files = varargin2S(S.add_fields, S0_files);
        S_files = varargin2S(S.add_fields, PFile.convert_to_S_file(S0_files));
        
        file = fullfile(PFile.root_data_dir, class(PFile), ...
            bml.str.Serializer.convert(S_files));
    end
end
%% Get file name given custom fields to copy from another object
methods
    function [file, S_file, S0_file] = get_file_from_obj(PFile, ...
            obj, file_fields)
        if ~exist('file_fields', 'var')
            file_fields = PFile.get_file_fields;
        end
        if ~exist('obj', 'var')
            obj = PFile;
        end
        S0_file = bml.oop.copyprops(struct, PFile.get_S0_file, ...
            'props', file_fields(:,1), ...
            'hide_error', true);
        S0_file = bml.oop.copyprops(S0_file, obj, ...
            'props', file_fields(:,1), ...
            'hide_error', true);
        S_file = PFile.convert_to_S_file(S0_file, ...
            'file_fields', file_fields);
        file = fullfile(PFile.root_data_dir, class(PFile), ...
            bml.str.Serializer.convert(S_file));
    end
end
%% Figures
methods
    function txt = get_title(W, args)
        % txt = get_title(W, args)
        if ~exist('args', 'var')
            args = struct;
        end
        S_title = W.get_S_file(args);
        txt = bml.str.Serializer.convert(S_title);
        txt = bml.str.get_title(txt);
    end
    function [axs, files, titles] = imgather(W0, row_args, col_args, page_args, add_args, varargin)
        % [axs, files, titles] = imgather(W0, row_args, col_args, page_args, add_args, ...)
        %
        % INPUT:
        % row_args, col_args, page_args
        % : Name-value arguments to be combined along
        %   rows, columns, and pages. 
        %   Set as {} to have only one row/column/page.
        %
        % add_args
        % : input to W.get_file(add_args).
        %   Fields that are not properties but on the list.
        %
        % When there are conflicts, priority is given to
        % the row over column over page.
        %
        % OUTPUT:
        % axs{page}(row, col)
        % : handle of the subplot.
        %   When there are multiple pages, only the last page is kept.
        %
        % files{page}
        % : file to save the page.
        %
        % titles{page}
        % : struct containing page, row, and column titles
        %
        % OPTIONS:
        % ... % 'title_subplot'
        % ... % if true, gives full title to each subplot
        % ... % if false, gives row/column/page title
        % 'title_subplot', false
        % ...
        % 'savefigs', true
        % 'savefigs_args', {}
        % ...
        % 'to_clf', true % Set false to retrieve ax of multiple pages
        %
        % EXAMPLE:
        % imgather(W0, {
        %     'subj', {'S1', 'S2'}
        %     }, {
        %     't0', {'st', 'en'}
        %     }, {
        %     'parad', {'RT', 'VD'}
        %     'truncate_st_msec', {500, 700}
        %     }, 'title_subplot', false);
        %
        % : subj along rows (S1 and S2), 
        %   t0 along columns (st and en),
        %   parad and truncate_st_msec along pages 
        %   (RT-500, RT-700, VD-500, and VD-700),
        %   with the option title_subplot=false.
        
        if nargin < 2, row_args = {}; end
        if nargin < 3, col_args = {}; end
        if nargin < 4, page_args = {}; end
        if nargin < 5, add_args = {}; end

        [Ss_page, n_page] = factorizeC(page_args);
        Ss_page_file = W0.convert_to_S_file(Ss_page);
        
        axs = cell(n_page, 1);
        files = cell(n_page, 1);
        titles = cell(n_page, 1);
        for page = 1:n_page
            page_args = varargin2C(Ss_page_file(page));
            
            [axs{page}, files{page}, titles{page}] = ...
                W0.imgather_page(row_args, col_args, page_args, add_args, ...
                    varargin{:});
        end
    end
    function [ax, file, titles] = imgather_page(W0, row_args, col_args, page_args, add_args, varargin)
        if nargin < 2, row_args = {}; end
        if nargin < 3, col_args = {}; end
        if nargin < 4, page_args = {}; end
        if nargin < 5, add_args = {}; end
        
        opt = varargin2S(varargin, {
            'clear_title', true % Clear existing title.
            ...
            ... % 'title_subplot'
            ... % if true, gives full title to each subplot
            ... % if false, gives row/column/page title
            'title_subplot', false
            'to_gltitle', true
            ...
            'savefigs', true
            'savefigs_args', {}
            ...
            'to_clf', true % Set false to retrieve ax of multiple pages
            });        
        
        [Ss_row, n_row] = factorizeC(row_args);
        [Ss_col, n_col] = factorizeC(col_args);
        S_page = varargin2S(page_args);
        
        Ss_row_file = W0.convert_to_S_file(Ss_row);
        Ss_col_file = W0.convert_to_S_file(Ss_col);
        
        % A single page
        S_page_file = W0.convert_to_S_file(S_page);
        
        ax = ghandles(n_row, n_col);
        titles.row = cell(n_row, 1);
        titles.col = cell(n_col, 1);
        titles.page = '';
        
        S2s = bml.str.Serializer;
        
        if opt.to_clf
            clf;            
        else
            figure;
        end
        for row = 1:n_row
            for col = 1:n_col
                ax1 = subplotRC(n_row, n_col, row, col);

                % row overrides col overrides page.
                S_row = Ss_row(row);
                S_col = Ss_col(col);

                S_row_file = Ss_row_file(row);
                S_col_file = Ss_col_file(col);

                titles.row{row} = S2s.convert(S_row_file);
                titles.col{col} = S2s.convert(S_col_file);
                titles.page = S2s.convert(S_page_file);

                W = feval(class(W0));
                S = varargin2S( ...
                        varargin2S( ...
                            S_row, ...
                            S_col), ...
                        S_page);
                C = S2C(S);

                W = varargin2fields(W, C);

                C1 = W0.convert_to_S_file(S);
                file_args = varargin2C(C1, add_args);
                file = [W.get_file(file_args), '.fig'];

                ax1 = openfig_to_axes(file, ax1);

                if opt.clear_title
                    title(ax1, '');
                end
                if opt.title_subplot
                    title(W.get_title(C));
                end

                ax(row,col) = ax1;
            end
        end

        if ~opt.title_subplot && opt.to_gltitle
            f_title = @(s) strrep(s, '_', '-');

            gltitle(ax, 'row', f_title(titles.row));
            gltitle(ax, 'col', f_title(titles.col));
            gltitle(ax, 'all', bml.str.wrap_text( ...
                f_title(titles.page{page})));
        end

        if opt.savefigs
            S_file = varargin2S({
                'page', {S_page_file}
                'row', {S2s.Ss2s(Ss_row_file)}
                'col', {S2s.Ss2s(Ss_col_file)}
                'add', {add_args}
                });
            name = S2s.convert(S_file);
            file = fullfile(PFile.root_data_dir, class(W), name);
            savefigs(file, opt.savefigs_args{:});
        end
    end
end
%% Add fields that are not properties
methods
    function add_file_fields(W, name_orig, name_short, value)
        % add_file_fields(W, name_orig, name_short, value)
        % add_file_fields(W, {name_orig1, name_short1, value1; ...})
        % 
        % Give an empty name_short to keep name_orig as is in file names.
        %
        % This method is not named set_file_fields, 
        % becasue it does not replace file_fields with the given ones.
        
        if iscell(name_orig)
            % {name_orig1, name_short1, value1; ...}
            assert(size(name_orig, 2) == 3);
            n = size(name_orig, 1);
            for ii = 1:n
                W.add_file_fields(name_orig{ii,:});
            end
            return;
        else
            assert(ischar(name_orig));
            assert(ischar(name_short) || isempty(name_short));
        end
        
        fs = W.file_fields;
        [~, locb] = ismember(name_orig, fs(:,1));
        
        if locb == 0
            % Then add to file_fields_, the one not specified directly
            % by get_file_fields
            fs_ = W.file_fields_;
            n1 = size(fs_, 1) + 1;
            W.file_fields_{n1, 1} = name_orig;
            W.file_fields_{n1, 2} = name_short;
        end

        % Assign value if given
        if nargin >= 4
            if isprop(W, name_orig)
                if ~isequal(W.(name_orig), value)
                    try
                        varargin2props(W, {name_orig, value});
                    catch err
                        warning(err_msg(err));
                    end
                end
            else
                W.S0_file_.(name_orig) = value;
            end
        end
    end
    function copy_file_fields(W, src, file_fields)
        % USAGE:
        % copy_file_fields(W, PFile_src)
        % copy_file_fields(W, PFile_src, {name_orig1; name_orig2; ...})
        % copy_file_fields(W, src, file_fields)
        % - src: struct or object
        % - file_fields: {name_orig1, name_short1; ...}
        
        if nargin < 3
            assert((isstruct(src) && isfield(src, 'file_fields')) ...
                || isprop(src, 'file_fields'));
            file_fields = src.file_fields;
        else
            assert(iscell(file_fields));
            
            if size(file_fields, 2) == 1
                assert((isstruct(src) && isfield(src, 'file_fields')) ...
                    || isprop(src, 'file_fields'));
                file_fields0 = src.file_fields;
                if isempty(file_fields0)
                    file_fields = cell(0,2);
                else
                    [~,~,ib] = intersect( ...
                        file_fields(:,1), ...
                        file_fields0(:,1), ...
                        'stable'); % Follows the order of file_fields
                    
                    file_fields = file_fields0(ib,:);
                end
            else
                assert(size(file_fields, 2) == 2);
            end
        end
        fs = file_fields(:,1);
        n = size(fs, 1);
        S0_file = src.S0_file;
        for ii = 1:n
            W.add_file_fields( ...
                file_fields{ii,1}, file_fields{ii,2}, ...
                S0_file.(file_fields{ii,1}));
        end
    end
end
%% Properties
methods
    function v = get.file_fields(W)
        v = W.get_file_fields_merged_;
    end
    function fs = get_file_fields_merged_(W)
        fs0 = W.get_file_fields;
        fs_ = W.file_fields_;
        
        if isempty(fs_)
            % Nothing to add to fs0
            fs = fs0;
        else
            % When there are overlapping fields in fs and fs_, 
            % fs, the one returned by get_file_fields, is prioritized.
            [~, ia] = setdiff(fs_(:,1), fs0(:,1), 'stable');

            fs = [fs0; fs_(ia, :)];
        end
    end
    function v = get_file_fields(~)
        v = cell(0,2);
    end
    
    function set.file_fields(W, v)
        W.set_file_fields(v);
    end
    function set_file_fields(W, v)
        % set_file_fields(W, file_fields)
        %
        % USAGE:
        % file_fields = {name_orig1, name_short1; ...}
        % file_fields = {name_orig1, name_short1, value1; ...}
        n = size(v, 1);
        for ii = 1:n
            W.add_file_fields(v{ii,:});
        end
    end

    function v = get.file_mult(W)
        v = W.get_file_mult;
    end
    function set.file_mult(W, v)
        W.set_file_mult(v);
    end

    function v = get_file_mult(W)
        v = W.file_mult_;
    end
    function set_file_mult(W, v)
        W.file_mult_ = v;
    end
end
end