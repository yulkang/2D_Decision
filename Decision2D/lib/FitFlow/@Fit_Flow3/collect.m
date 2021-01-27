function ds = collect(flt, varargin)
S = varargin2S(varargin, {
    'varname',      'Fl'
    });

if nargin < 1 || isempty(flt)
    fs = uipickfiles('FilterSpec', 'Data/run_fit_1D');
else
    fs = uipickfiles('FilterSpec', flt);
end
if isequal(fs, 0)
    ds = [];
    return;
end

files = rdir(fs{1}, @(d) ~d.isdir && strcmpLast(d.name, '.mat'));
files = {files.name};
[~,nams] = filepartsAll(files);
n = length(files);

model = fileparts(files{1});
[~,model] = fileparts(model);

ds = dataset;

for ii = n:-1:1
    L  = load(files{ii}, S.varname);
    Fl = L.(S.varname);
    Fl.fit_postprocess;
    
    ds.model{ii,1} = model;
    ds.id{ii,1}   = nams{ii};
    ds.fval(ii,1) = Fl.res.fval;
    
    for jj = 1:Fl.n_th
        c_th = Fl.th_names{jj};
        
        ds.(c_th)(ii,1) = Fl.res.th.(c_th);
        ds.(['se_' c_th])(ii,1) = Fl.res.se.(c_th);
    end
end