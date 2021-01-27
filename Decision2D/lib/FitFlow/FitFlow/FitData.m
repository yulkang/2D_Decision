classdef FitData < PartialSave
% Loads/saves data into struct/obj with flexible mapping of names.
%
% 2015 (c) Yul Kang. yul dot kang dot on at gmail dot com.
    
properties
    path = '';
    
    file2dat = struct; % file2dat.(field_name_in_file) = 'file_name_in_loaded_obj'
    field_included = {}; % if nonempty, overrides field_excluded.
    field_excluded = {}; % exclude fields on load & save.
    
    % Internal use only. Whether to save and load the whole dataset (true),
    % or each column separately as a variable (false - preferred).
    use_dataset = false;
    
    % dataset_name: used as a fallback even when use_dataset is false.
    dataset_name = 'dat';
    
    n_data = 0;
    
    % dat_filt_spec
    % : If empty, ds = ds0.
    % : If a function_handle, ds = ds0(fun(Dat),:)
    %   - Always use a single input 'Dat', which refers to the whole object.
    %     FitData converts function_handle to char on save.
    % : If a logical or numerical vector, ds = ds0(vec,:)
    %
    % Note that dat_filt_spec of [] is interpreted as choosing all, 
    % while an empty output from dat_filt_spec set as a function handle 
    % is interpreted as choosing none.
    dat_filt_spec = ''; 
    
    % On saving, if dat_filt_spec is a function_handle,
    % it is converted to string and stored in dat_filt_desc,
    % and dat_filt_spec gets the logical vector instead.
    dat_filt_desc = ''; 

    loaded = false;
    filtered = false;
end
properties (Dependent)
    dat_filt
    dat2file % inverse mapping of file2dat
    
    is_loaded
end
properties (Dependent)
    ds
    ds0
end
properties (Transient)
    ds_ = dataset;
    ds0_ = dataset;
end
properties
    W % Retain a link to FitWorkspace
end
methods
%% Construct
function Dat = FitData(varargin)
    Dat = varargin2fields(Dat, varargin, false);
end
%% Filter
function set_filt_spec(Dat, filt_spec)
    % set_filt_spec(Dat, filt_spec)
    %
    % filt_spec: either a index vector or a function handle that gets Dat.
    % To use the dataset, use a function of the form @(Dat) Dat.ds0.(field)
    %
    % Note that dat_filt_spec of [] is interpreted as choosing all, 
    % while an empty output from dat_filt_spec set as a function handle 
    % is interpreted as choosing none.
    %
    % Call filt_ds to actually apply filt_spec
    if isa(filt_spec, 'function_handle')
        assert(nargin(filt_spec) == 1);
        Dat.dat_filt_spec = filt_spec;
        % Not checked right away!
    else
        if isempty(filt_spec)
            % Set as empty.
        elseif isnumeric(filt_spec)
            assert(isvector(filt_spec) && ...
                min(filt_spec) > 0 && max(filt_spec) <= Dat.get_n_tr0);
        elseif islogical(filt_spec)
            assert(isvector(filt_spec) && length(filt_spec) == Dat.get_n_tr0);
        elseif ischar(filt_spec) && isequal(filt_spec, ':')
            % This is fine, too.
        else
            error('filt_spec must be a function_handle, a vector, or empty!');
        end
        Dat.dat_filt_spec = filt_spec;
    end
    Dat.filtered = false;
    % filt_ds is called automatically on get.ds
end
function filt = get.dat_filt(Dat)
    % Always returns a numeric vector.
    filt = Dat.get_dat_filt;
end
function filt = get_dat_filt(Dat)
    % Always returns a numeric vector.
    % called by get.dat_filt. Modifiable by subclasses.
    filt_spec = Dat.dat_filt_spec;
    
    if isa(filt_spec, 'function_handle')
        try
            filt = filt_spec(Dat);
        catch err
            disp(filt_spec);
            rethrow(err);
        end
    elseif isempty(filt_spec) ...
            || (ischar(filt_spec) && isequal(filt_spec, ':'))
        filt = true(Dat.get_n_tr0, 1);
        % Note that dat_filt_spec of [] is interpreted as choosing all, 
        % while an empty output from dat_filt_spec set as a function handle 
        % is interpreted as choosing none.
    else
        filt = filt_spec;
    end
    filt = Dat.unify_filt_class(filt);
end
function filt = unify_filt_class(Dat, filt)
    filt = Dat.convert_filt_to_numeric(filt);
    
    % logical index is less flexible
    % (e.g., cannot sample with replacement.)
%     filt = Dat.convert_filt_to_logical(filt);
end
function filt = get_dat_filt_logical(Dat)
    filt = Dat.convert_filt_to_logical(Dat.get_dat_filt);
end
function filt = get_dat_filt_numeric(Dat)
    filt = Dat.convert_filt_to_numeric(Dat.get_dat_filt);
end
function filt = convert_filt_to_logical(Dat, filt)
    % Postprocess if numeric, whether it is a result of a function call 
    % or it is the value of dat_filt_spec itself.
    if isnumeric(filt)
        filt = ix2tf([Dat.get_n_tr0, 1], filt);
    elseif islogical(filt) && isscalar(filt)
        filt = repmat(filt, [Dat.get_n_tr0, 1]);
    elseif ischar(filt) && isequal(filt, ':')
        filt = true(Dat.get_n_tr0, 1);
    end
    assert(islogical(filt));
    assert(isvector(filt));
    assert(length(filt) == Dat.get_n_tr0);    
end
function filt = convert_filt_to_numeric(Dat, filt)
    if islogical(filt)
        filt = find(filt);
    elseif ischar(filt) && isequal(filt, ':')
        filt = (1:Dat.get_n_tr0)';
    end
    assert(isnumeric(filt));
    assert(isvector(filt));
    if ~isempty(filt)
        % DEBUG
%         disp(max(filt));
%         disp(min(filt));
%         disp(length(filt));
%         disp(Dat.get_n_tr0);
%         disp(Dat.loaded);
%         disp(length(Dat.ds));
%         disp(length(Dat.ds0));
%         plot(filt);
        
        assert(max(filt) <= Dat.get_n_tr0);
        assert(min(filt) >= 1);
        assert(all(floor(filt) == filt));
    end
end
function filt_ds(Dat)
    filt = Dat.get_dat_filt;
%     if isempty(filt) % = filt on set_ds0
%         Dat.ds = Dat.ds0;
%     else
        Dat.ds = Dat.get_ds0(filt,':');
%     end
%     if ~is_in_parallel
%         fprintf('Filtering %d/%d trials\n', Dat.get_n_tr, Dat.get_n_tr0);
%     end
    Dat.filtered = true;
end
function v = get.ds(Dat)
    if ~Dat.is_loaded
        Dat.load_data;
    end
    if ~Dat.filtered
        Dat.filt_ds;
    end
    v = Dat.get_ds;
end
function ds = get_ds(Dat, varargin)
    % ds = get_ds(Dat, varargin)
    % Gives Dat.ds(varargin{:})
    % Load & filter before getting the value, if not done already.
    if ~Dat.is_loaded
        Dat.load_data;
    end
    if Dat.is_loaded && ~Dat.filtered
        Dat.filt_ds;
    end
    if length(varargin) == 1
        ds = Dat.ds_.(varargin{1});
    elseif length(varargin) == 2
        ds = Dat.ds_(varargin{:});
    else
        assert(isempty(varargin));
        ds = Dat.ds_;
    end
end
function set.ds(Dat, v)
    Dat.set_ds(v);
end
function set_ds(Dat, varargin)
    if nargin == 2
        Dat.ds_ = varargin{1};
    else
        Dat.ds_ = ds_set(Dat.ds_, ':', varargin{:});
        
        ix = Dat.dat_filt;
        Dat.ds0_ = ds_set(Dat.ds0_, ix, varargin{:});
    end
end
function ds = get_ds_field(Dat, field)
    ds = Dat.get_ds;
    ds = ds.(field);
end
function v = get.ds0(Dat)
    v = Dat.get_ds0;
end
function ds0 = get_ds0(Dat, varargin)
    if ~Dat.is_loaded
        Dat.load_data;
    end
    if length(varargin) == 1
        ds0 = Dat.ds0_.(varargin{1});
    elseif length(varargin) == 2
        ds0 = Dat.ds0_(varargin{:});
    else
        assert(isempty(varargin));
        ds0 = Dat.ds0_;
    end
end
function ds0 = get_ds0_field(Dat, field)
    ds0 = Dat.get_ds0;
    ds0 = ds0.(field);
end
function set.ds0(Dat, v)
    Dat.ds0_ = v;
end
function n = get_n_tr(Dat)
    n = nnz(Dat.get_dat_filt);
end
function n = get_n_tr0(Dat)
    n = length(Dat.get_ds0);
end
%% Transfer between FitData objects
function import_FitData(Dat_dst, Dat_src)
    % import_FitData(Dat_dst, Dat_src)
    Dat_dst.set_path(Dat_dst.get_path);
    Dat_dst.set_ds0(Dat_src.ds0);
    Dat_dst.set_filt_spec(Dat_src.dat_filt_spec);
    Dat_dst.filt_ds;
end
%% Load/Save
function set_path(Dat, path_)
    assert(ischar(path_));
    if ~strcmp(Dat.get_path, path_)
        Dat.path = path_;
        Dat.loaded = false; 
        % Does not load automatically because it will take time.
    end
end
function path_ = get_path(Dat)
    % Not doing much here, but subclasses might infer path from
    % other information.
    path_ = Dat.path;
end
function load_data(Dat, field_excluded, field_included)
    % load_data(Dat, field_excluded, field_included)
    % field_included overrides field_excluded if specified and nonempty.
    %
    % Skips loading when loaded == true.
    % (set to false on set_path to different path)
    % Set Dat.loaded = false to reload. 
    %
    % Not called automatically on load.
    % To load automatically, implement loadobj and call load_obj.
    
    pth = Dat.get_path;
    
    if Dat.is_loaded
        fprintf('FitData.load_data: Loaded already. Skipping loading %s\n', ...
            pth);
        return; 
    end
    
    if nargin < 2, field_excluded = Dat.field_excluded; end
    if nargin < 3, field_included = Dat.field_included; end
    
    if isempty(pth), return; end
    if strcmpStart('../Data', pth)
        % Backward compatibility
        pth = strrep(pth, '../Data', '../../Data');
    end
    
    fields_exist = who('-file', pth);
    if isscalar(fields_exist) && strcmp(fields_exist{1}, Dat.dataset_name)
        use_dataset = true;
    else
        use_dataset = Dat.use_dataset;
    end
    
    if ~is_in_parallel
        fprintf('Loading %s ... ', pth);
    end
    if use_dataset
        ds0 = load(pth, Dat.dataset_name);
        ds0 = ds0.(Dat.dataset_name);
        if isempty(field_included)
            fields_exist = ds0.Properties.VarNames;
            field_included = setdiff(fields_exist, field_excluded, 'stable');
        end
        ds0 = ds0(:, field_included);        
    else
        if isempty(field_included)
            fields_exist = who('-file', pth);
            field_included = setdiff(fields_exist, field_excluded, 'stable');
        end
        ds0 = load(pth, field_included{:});
        ds0 = struct2dataset(ds0);
    end
    if ~is_in_parallel
        fprintf('done.\n');
    end
    Dat.set_ds0(ds0);
    Dat.loaded = true;
    Dat.filtered = false;
end
function set_ds0(Dat, ds0)
    % Map fields
    field_src = fieldnames(Dat.file2dat);
    if ~isempty(field_src)
        field_dst = struct2cell(Dat.file2dat);
        ds0.Properties.VarNames = strrep_cell(ds0.Properties.VarNames, ...
            field_src, field_dst);
    end
    
    Dat.ds0 = ds0;
end
function Dat = saveobj(Dat)
    % On saving, if dat_filt_spec is a function_handle,
    % it is converted to string and stored in dat_filt_desc,
    % and dat_filt_spec gets the logical vector instead.
    if ~isempty(Dat.dat_filt_spec) && ...
            isa(Dat.dat_filt_spec, 'function_handle')
        Dat.dat_filt_desc = func2str(Dat.dat_filt_spec);
        Dat.dat_filt_spec = Dat.get_dat_filt;
    end    
    Dat.loaded = false;
end
function save_data(Dat, field_excluded, field_included, file_out)
    % save_data(Dat, field_excluded, field_included)
    % field_included overrides field_excluded if specified and nonempty.
    %
    % Not called automatically on save.
    % To load automatically, implement saveobj and call save_obj.
    if nargin < 2, field_excluded = Dat.field_excluded; end
    if nargin < 3, field_included = Dat.field_included; end
    
    % Set back to orig
    Dat.ds0 = ds_set(Dat.get_ds0, Dat.dat_filt, Dat.get_ds);
    
    % Filter fields
    if isempty(field_included)
        field_included = setdiff(Dat.ds.Properties.VarNames, ...
            field_excluded, 'stable');
    end
    ds = Dat.get_ds0(':', field_included);
    
    % Map fields
    mapping = Dat.dat2file;
    field_src = fieldnames(mapping);
    if ~isempty(field_src)
        field_dst = struct2cell(mapping);
        ds.Properties.VarNames = strrep_cell(ds.Properties.VarNames, ...
            field_src, field_dst);
    end
    
    % Dataset or not
    if Dat.use_dataset
        L.(Dat.dataset_name) = ds;
    else
        L = dataset2struct(ds, 'AsScalar', true);
    end
    
    % Save
    if ~exist('file_out', 'var')
        file_out = Dat.get_path;
        [~,~,ext] = fileparts(file_out);
        if ~isequal(ext, '.mat'), file_out = [file_out, '.mat']; end
        if exist(file_out, 'file')
            if ~inputYN_def( ...
                    sprintf('%s exists already! Overwrite', file_out), ...
                    false);
                return;
            end
        end
    end
    mkdir2(fileparts(file_out));
    save(file_out, '-struct', 'L');    
    
    column_names = ds.Properties.VarNames;
    
    fprintf('Saved to %s\n:', file_out);
    fprintf(' %s', column_names{:});
    fprintf('\n');
end
function tf = get.is_loaded(Dat)
    tf = Dat.loaded && ~isempty(Dat.ds0_);
end
function set.is_loaded(Dat, tf)
    Dat.loaded = tf;
end
%% dat2file
function dat2file = get.dat2file(Dat)
    dat2file = Dat.get_dat2file;
end
function dat2file = get_dat2file(Dat, file2dat)
    if nargin < 2
        file2dat = Dat.file2dat;
    end
    if isequal(file2dat, struct)
        dat2file = struct;
    else
        dat2file = cell2struct(fieldnames(file2dat), struct2cell(file2dat));
    end
end
end
methods (Static)
function Dat = loadobj(Dat)
    % Recover filt
    if ~isempty(Dat.dat_filt_spec) && ischar(Dat.dat_filt_spec)
        Dat.dat_filt_spec = str2func(Dat.dat_filt_spec);
    end
    
%     % Results in duplicate loading if Data handle is not shared properly.
%     Dat.load_data;
    
%     disp('Loading'); % DEBUG
    
    % Not calling load_data at this point.
    % In case there are multiple references to Dat,
    % it seems that loadobj is called multiple times.
%     warning('Call load_data to load actual data!');
end    
end
end