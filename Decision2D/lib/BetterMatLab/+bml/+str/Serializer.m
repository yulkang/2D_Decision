classdef Serializer < matlab.mixin.Copyable
    % Recursively serialize struct, cell array, string, numerical arrays,
    % or a mix of them in an rsync-compatible way.
    %
    % EXAMPLE:
    % >> bml.str.Serializer.convert(varargin2S({'a', 1:2, 'bc', {'d', 'e', 2, varargin2S({'A', 1, 'b', 2, 'c', {'Z', [1 3 4], 'D'}})}}))
    % ans =
    % a=[1,2]+bc={d}{e}{2}{A=1+b=2+c={Z}{[1,3,4]}{D}}    
    %
    % EXAMPLE2:
    %     
    % >> s = 'parad=sh+task=A+subj=DX+SigmaSq=Const+ms=100.2+d=A_bc';
    % >> S2s.convert(s)
    % 
    % ans = 
    %       parad: 'sh'
    %        task: 'A'
    %        subj: 'DX'
    %     SigmaSq: 'Const'
    %          ms: 100.2000
    %           d: 'A_bc'
    % 
    % >> S2s.convert(S2s.convert(s))
    % ans =
    % parad=sh+task=A+subj=DX+SigmaSq=Const+ms=100.2+d=A_bc
    % 
    % >> strcmp(S2s.convert(S2s.convert(s)), s)
    % ans =
    %      1
    %
    % See also: v2str
    %
    % 2015-2016 (c) Yul Kang. hk2699 at columbia dot edu.
properties
    sep_fields = '+'; % '___' % ; and & doesn't work with rsync
    sep_field_val = '='; % '__'
    sep_val = ','; % '_'
    st_mat = '[';
    en_mat = ']';
    st_cell = '{';
    en_cell = '}';
    sep_cell = ',';
    
    % replace_pair{k,1} in value is replaced with {k,2} in string,
    % and vice versa.
    replace_pair = {
        '.', '^'
        };
    
    skip_fields_with_error = true;
    skip_empty = true;
end
%% User Interface - conversion
methods (Static)
    function res = convert(S, varargin)
        if ischar(S)
            if any(S == '=')
                res = bml.str.Serializer.struct(S, varargin{:});
            else
                S2s = bml.str.Serializer(varargin{:});
                res = S2s.str2value(S);
            end
        else
            res = bml.str.Serializer.char(S, varargin{:});
        end
    end
    function str = char(S, varargin)
        S2s = bml.str.Serializer(varargin{:});
        str = S2s.value2str(S);
    end
    function S = struct(S, varargin)
        S2s = bml.str.Serializer(varargin{:});
        S = S2s.str2struct(S);
    end
    function [s, S] = Ss2s(Ss, varargin)
        fs = fieldnames(Ss);
        S = struct;
        for f = fs(:)'
            vs = bml.matrix.unique_general({Ss.(f{1})});
            if isscalar(vs)
                vs = vs{1};
            end
            S.(f{1}) = vs;
        end
        s = bml.str.Serializer.convert(S);
    end
end
%% User Interface - field name shortener/recoverer
methods (Static)
    function [S, S0] = convert_to_S_file(obj, C, varargin)
        % [S, S0] = convert_to_S_file(obj, C, varargin)
        %
        % S0 : struct with the original fields and values
        % S  : struct with shortened field names and multiplied values.
        % C{i_pair, [orig, short]}: same as file_fields of ProfFileName.
        opt = varargin2S(varargin, {
            'mult', {}
            });
        
        if isempty(C)
            S0 = struct;
        else
            S0 = copyprops(struct, obj, 'props', C(:,1));
        end
        S = S0;
        
        % Since field_multiply uses the original field name,
        % do multiply before strrep.
        S = bml.str.Serializer.field_multiply(S, opt.mult);
        S = bml.str.Serializer.field_strrep(S, C);
    end
    function S0 = convert_from_S_file(S, C, varargin)
        % S0 = convert_from_S_file(S, C, varargin)
        %
        % S  : struct with shortened field names and multiplied values.
        % S0 : struct with the original fields and values
        % C{i_pair, [orig, short]}: same as file_fields of ProfFileName.
        opt = varargin2S(varargin, {
            'mult', {}
            });
        S0 = S;
        % Since field_multiply uses the original field name,
        % do multiply after strrep.
        S0 = bml.str.Serializer.field_strrep(S0, C(:,[2 1]));
        S0 = bml.str.Serializer.field_multiply(S0, opt.mult, true);
    end
    function S = field_multiply(S, mult, to_inv)
        if isempty(mult)
            return;
        end
        if nargin < 3
            to_inv = false;
        end
        assert(isstruct(S) || isobject(S));
        assert(iscell(mult));
        assert(size(mult, 2) == 2);
        assert(all(cellfun(@ischar, mult(:,1))));
        assert(all(cellfun(@isnumeric, mult(:,2))));
        
        n = size(mult, 1);
        for ii = 1:n
            f = mult{ii, 1};
            m = mult{ii, 2};
            if to_inv
                m = 1 / m;
            end
            if isfield(S, f)
                if iscell(S.(f))
                    for jj = 1:numel(S.(f))
                        S.(f){jj} = S.(f){jj} .* m;
                        if m > 1
                            S.(f){jj} = round(S.(f){jj});
                        end
                    end
                else
                    S.(f) = S.(f) .* m;
                    if m > 1
                        S.(f) = round(S.(f));
                    end
                end
            end
        end
    end
    function S = field_strrep(S0, C)
        % S0: original struct (S0.(src1) ...)
        % C : {src1, dst1; src2, dst2; ...}
        % S : converted struct (S.(dst1) = S0.(src1) ...)
        if isempty(C)
            S = S0;
            return;
        end
        assert(isstruct(S0) || isobject(S0));
        assert(iscell(C));
        assert(size(C, 2) == 2);
        assert(all(cellfun(@ischar, C(:))));
        
        S = struct;
        for ii = 1:size(C, 1)
            src = C{ii, 1};
            dst = C{ii, 2};
            
            if isempty(dst) % keep the original name
                if isfield(S0, src)
                    S.(src) = S0.(src);
                end
            elseif isempty(src) % keep the destination name
                if isfield(S0, dst)
                    S.(dst) = S0.(dst);
                end
            else
                if isfield(S0, src)
                    S.(dst) = S0.(src);
                end
            end
        end
    end
end
%% User Interface - file
methods (Static)
    function [ds, files] = ls2ds(filt, ls_args)
        % [ds, files] = ls2ds(filt, ls_args)
        S2s = bml.str.Serializer;
        if ~exist('ls_args', 'var')
            ls_args = {};
        end
        ls_args = varargin2C(ls_args);
        
        if ischar(filt)
            files = S2s.ls(filt, ls_args{:});
        else
            files = filt;
        end
        n = numel(files);
        
        for ii = n:-1:1
            [~, nam] = fileparts(files{ii});
            Ss{ii} = S2s.convert(nam);
        end
        if n > 0
            ds = bml.ds.from_Ss(Ss);
        else
            ds = dataset;
        end
    end
    function [files, desc, info] = ls(filt, varargin)
        % [files, desc, info] = ls(filt, varargin)
        %
        % filt: filter string or a cell array of file paths.
        %
        % OPTIONS
        % -------
        % 'allof', []
        % 'anyof', []
        % 'noneof', []
        % 'notallof', []
        % 'props', {} % properties of Serializer
        %
        % Example:
        % >> S2s.ls('Data/*.mat', {'allof', {'subj=EX', 'dim=1,2'}})
        % >> S2s.ls('Data/*.mat', {'allof', ...
        %           varargin2S({'subj', 'EX', 'dim', 1:2})
        
        if ~exist('filt', 'var')
            filt = pwd;
        end
        opt = varargin2S(varargin, {
            'allof', []
            'anyof', []
            'noneof', []
            'notallof', []
            'props', {} % properties of Serializer
            'fullpath', true
            });
        is_filt_list = iscell(filt);

        S2s = bml.str.Serializer(opt.props{:});
        
        for f = {'allof', 'anyof', 'noneof', 'notallof'}
            v = opt.(f{1});
            if iscell(v) && size(v, 2) == 2 ...
                    && ((size(v, 1) > 1) ...
                     || any(~cellfun(@ischar, v(:,2))))
                v = varargin2S(v);
            end
            if isstruct(v)
                v = S2s.struct2cellstr(v);
            end
            opt.(f{1}) = v;
        end
        
        if ~is_filt_list
            info = vVec(dir(filt));
            files = vVec({info.name});
        else
            files = filt;
        end
        n = numel(files);
        
        desc = S2s.get_desc(files);
%         [~,~,~,~,desc] = S2s.fileparts(files);
        
        incl = true(size(files));
        
        if ~isequal(opt.allof, [])
            incl = incl ...
                 & cellfun(@(s) ...
                     all(ismember(opt.allof, s)), desc);
        end
        if ~isequal(opt.anyof, [])
            incl = incl ...
                 & cellfun(@(s) ...
                     any(ismember(opt.anyof, s)), desc);
        end
        if ~isequal(opt.noneof, [])
            incl = incl ...
                 & cellfun(@(s) ...
                     ~any(ismember(opt.noneof, s)), desc);
        end
        if ~isequal(opt.notallof, [])
            incl = incl ...
                 & cellfun(@(s) ...
                     ~all(ismember(opt.notallof, s)), desc);
        end
        
        files = files(incl);
        desc = desc(incl);
        if ~is_filt_list
            info = info(incl);
        end
        
        if opt.fullpath && ~is_filt_list
            pth = bml.file.filt2dir(filt);
            files = fullfile(pth, files);
        end
    end
    function desc = get_desc(file)
        % desc = get_desc(file)
        %
        % desc: cell array of descriptor strings
        %
        % See also fileparts.
        if iscell(file)
            desc = cellfun(@bml.str.Serializer.get_desc, file, ...
                'UniformOutput', false);
            return;
        end
        
        [~, nam] = fileparts(file);
        desc = strsep_cell(nam, '+');
    end
    function strrep_in_dir(d)
        d = fullfile(d, '*.*');
        
        files = bml.file.strrep_filename('___', '+', ...
            'filt', d);
        
        if ~isempty(files)
            bml.file.strrep_filename('__', '=', ...
                'files', files, 'confirm', false);
        end
%         bml.file.strrep_filename('==', '+', 'filt', d);
    end
    function files = strrep(files)
        files = strrep_cell(files, {
            '___', '+'
            '__', '='
            }, [], 'wholeStringOnly', false);
    end
end
%% Direct manipulation of strings
methods
    function s = str_con(S2s, s1, s2)
        s = str_bridge(S2s.sep_fields, s1, s2);
    end
end
%% Init
methods
    function S2s = Struct2Str(varargin)
        S2s = varargin2props(S2s, varargin{:});
    end
end
%% Internal - objects to str
methods
    function cstr = struct2cellstr(S2s, S)
        % cstr = struct2cellstr(S2s, S)
        %
        % EXAMPLE
        % -------
        % >> cstr = S2s.struct2cellstr(varargin2S({'a', 1, 'c', 'abc'}))
        % cstr = 
        %     'a=1'
        %     'c=abc'
        % 
        % >> S2s.struct2cellstr(cstr)
        % ans = 
        %     'a=1'
        %     'c=abc'    
        %
        % >> cstr = S2s.struct2cellstr({varargin2S({'a', 1, 'c', 'abc'}), ...
        %           varargin2S({'a', 10, 'c', 'ABC'})}); cstr2 = [cstr{:}]
        % cstr2 = 
        %     'a=1'      'a=10' 
        %     'c=abc'    'c=ABC'        
        
        if iscell(S)
            if ischar(S{1})
                cstr = S;
                return;
            else
                assert(isstruct(S{1}));
                cstr = cellfun(@S2s.struct2cellstr, S, ...
                    'UniformOutput', false);
                return;
            end
        end
        fs = fieldnames(S);
        n = numel(fs);
        cstr = cell(n, 1);
        for ii = 1:n
            cstr{ii} = S2s.field2str(fs{ii}, S.(fs{ii}));
        end
    end
    function str_v = value2str(S2s, v)
        if iscell(v)
            str_vs = cellfun(@S2s.value2str, v, 'UniformOutput', false);
            str_v = [S2s.st_cell, ...
                     str_bridge(S2s.sep_cell, str_vs{:}), ...
                     S2s.en_cell];
            
        elseif isstruct(v)
            str_v = S2s.struct2str(v);
            
        elseif ischar(v)
            str_v = v;
            
        else
            assert(isnumeric(v) || islogical(v));
            if isscalar(v)
                str_v = sprintf('%g', v);
            else
                str_vs = arrayfun(@(v) sprintf('%g', v), v, ...
                    'UniformOutput', false);
                str_v = [
                    S2s.st_mat, ...
                    str_bridge(S2s.sep_val, str_vs{:}), ...
                    S2s.en_mat
                    ];
            end
        end
        str_v = strrep_cell(str_v, S2s.replace_pair);
    end
    function str = struct2str(S2s, S)        
%         if isequal(S2s.fields, [])
%             S2s.fields = fieldnames(S);
%         end
%         if ~S2s.include_fields
%             S2s.fields = setdiff(fieldnames(S), S2s.fields, 'stable');
%         end
%         nf = numel(S2s.fields);

        fields = fieldnames(S);
        nf = numel(fields);

        str = '';
        for ii = 1:nf
            f = fields{ii};
            v = S.(f);
            
            str_f = S2s.field2str(f, v);
            str = str_bridge(S2s.sep_fields, str, str_f);            
        end
    end
    function str_f = field2str(S2s, f, v)
        if (isempty(v) || isequal(v, struct)) && S2s.skip_empty
            str_f = '';
            return;
        end
        str_f = f;
        
        try
            str_v = S2s.value2str(v);
            str_f = [str_f, S2s.sep_field_val, str_v];
            
        catch err
            if S2s.skip_fields_with_error
                str_f = '';
            else
                rethrow(err);
            end
        end
    end
end
%% Internal - manipulate str
methods
    function [S_file, pth, ext, name, desc] = fileparts(S2s, s)
        % [S_file, pth, ext, name, desc] = fileparts(S2s, s)
        % desc: cell array of descriptor strings
        %
        % To get desc only, use get_desc instead (much faster).
        %
        % See also: get_desc
        if iscell(s)
            n = numel(s);
            siz = size(s);
            S_file = cell(siz);
            pth = repmat({''}, siz);
            ext = pth;
            name = pth;
            desc = pth;
            for ii = 1:n
                try
                    [S_file{ii}, pth{ii}, ext{ii}, name{ii}, desc{ii}] = ...
                        S2s.fileparts(s{ii});
                catch err
                    warning(err_msg(err));
                    
                    continue;
                end
            end
%             [S_file, pth, ext, name, desc] = cellfun(@S2s.fileparts, s, ...
%                 'UniformOutput', false);
            return;
        end
        
        [pth, name, ext] = fileparts(s);
        S_file = S2s.convert(name);
        if nargout >= 5
            desc = S2s.strsep(name);
        end
    end
    function compo = strsep(S2s, s)
        if iscell(s)
            compo = cellfun(@S2s.strsep, s, 'UniformOutput', false);
            return;
        end
        
        compo = strsep2C(s, S2s.sep_fields);
    end
    function file = fullfile(S2s, pth, S_file, ext)
        if ~exist('ext', 'var')
            ext = '';
        end
        file = fullfile(pth, [S2s.convert(S_file), ext]);
    end
end
%% Internal - str to objects
methods
    function S = str2struct(S2s, s)
        S = struct;
        while any(s == S2s.sep_field_val)
            % Field name
            ix = find(s == S2s.sep_field_val, 1, 'first');
            f = s(1:(ix - 1));
            
            % Field value
            ix2 = find(s == '+', 1, 'first');
            if isempty(ix2)
                ix2 = length(s) + 1;
            end
            v = s((ix+1):(ix2 - 1));
            v = S2s.str2value(v);
            
            % Assign field
            S.(f) = v;
            
            % Go to the next field
            s = s((ix2+1):end);
        end
    end
    function v = str2value(S2s, s)
        if isempty(s)
            v = [];
        elseif all(ismember(strrep_cell(s, {
                'NaN', '0'
                'Inf', '0'
                }), ['-', '.', '0':'9', ',']))
            v = eval(['[', s, ']']);
        elseif s(1) == '[' && s(end) == ']'
            s = strrep(s, '^', '.');
            if all(ismember(s(2:(end-1)), ['-', '.', '0':'9', ',']))
                v = eval(s);
            else
                error('Not implemented yet!');
            end
        elseif all(ismember(s, ['0':'9', '^']))
            s = strrep(s, '^', '.');
            v = eval(s);
        elseif all(ismember(s, ['_', '0':'9', 'a':'z', 'A':'Z', '^']))
            s = strrep_cell(s, S2s.replace_pair(:, [2 1]));
            v = s;
        elseif s(1) == '{' && s(end) == '}'
            if any(s == '[')
                s = ['],', s(2:(end-1)), ',['];
                s = strrep(s, '],[', '];[');
                ix_sep = find(s == ';');
                if any(s(ix_sep - 1) ~= ']') ...
                        || any(s(ix_sep + 1) ~= '[')
                	error(['Currently works only when all elements ' ...
                           'of the cell array are vectors, ' ...
                           'e.g., {[],[],[]}']);
                end
                
                s = strrep(s, ',', ';');
                ss = strsep_cell(s, ';');
                ss = ss(2:(end-1));
                n_sep = numel(ss);
                v = cell(1, n_sep);
                for ii = 1:n_sep
                    v{ii} = S2s.str2value(ss{ii});
                end
                
            else
%             if all(ismember(s(2:(end-1)), ['_', '0':'9', 'a':'z', 'A':'Z', ',']))
                s = [',', s(2:(end-1)), ','];
                ix_sep = find(s == ',');
                n_sep = length(ix_sep) - 1;
                v = cell(1, n_sep);
                for ii = 1:n_sep
                    v{ii} = S2s.str2value( ...
                        s((ix_sep(ii) + 1):(ix_sep(ii + 1) - 1)));
                end
            end
%             else
%                 error('Not implemented yet!');
%             end
        else
            error('Not implemented yet!');
        end 
    end
end
end