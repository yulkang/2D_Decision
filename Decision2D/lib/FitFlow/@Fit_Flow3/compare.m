function ds = compare(modelDirs, varargin)
% ds = compare(Fls_or_files, varargin)

S = varargin2S(varargin, {
    'by', 'bic' % fval, aic, aic_c, or bic (default)
    'varname', 'Fl'
    'FilterSpec', 'Data/run_fit_1D'
    });

if nargin == 0 || isempty(modelDirs) || ischar(modelDirs)
    if nargin >= 1 && ischar(modelDirs) && ~isempty(modelDirs)
        flt = modelDirs;
    elseif exist(S.FilterSpec, 'dir')
        flt = S.FilterSpec;
    else
        flt = '';
    end
    if isempty(flt)
        modelDirs = uipickfiles;
    else
        modelDirs = uipickfiles('FilterSpec', flt);
    end
    if isequal(modelDirs, 0)
        ds = [];
        return;
    end
end
assert(iscell(modelDirs), 'First argument must be a cell array of directory names!');
n_model = numel(modelDirs);

ds = dataset;
ds.id = {};

[~, models] = filepartsAll(modelDirs);

%% Load results
for ii = 1:n_model
    modelDir = modelDirs{ii};
    model    = models{ii};
    
    d   = rdir(modelDir, @(d) strcmpLast(d.name, '.mat'));
    d   = {d.name};
    [~,ids] = filepartsAll(d);
    n_id = length(ids);
    
    ds.id = union(ds.id, ids, 'stable');
    
    for jj = 1:n_id
        cid = ids{jj};
        ix = find(strcmp(cid, ds.id));
        
        if isempty(ix)
            ix = length(ds) + 1;
            ds.id{ix,1} = cid;
        end
        
        L = load(d{jj}, S.varname);

        Fl = L.(S.varname);
        Fl.fit_postprocess;

        ds.(model)(ix,1) = Fl.res.(S.by);
    end
end

%% Clean up missing values
for ii = 1:n_model
    modelDir  = modelDirs{ii};
    model     = models{ii};
    
    ds.(model)(ds.(model) == 0) = nan;
end

%% Find the best model
M = ds2mat(ds(:,2:end));
[best_ic, best_ix] = nanmin(M, [], 2);

ds.best_ic = best_ic;
ds.best_model = vVec(models(best_ix));

