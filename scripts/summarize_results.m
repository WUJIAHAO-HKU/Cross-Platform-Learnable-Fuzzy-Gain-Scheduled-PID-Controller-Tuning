function T = summarize_results(varargin)
%SUMMARIZE_RESULTS Aggregate baseline/hybrid result JSON & MAT into a table.
% Usage:
%   T = summarize_results();
%   T = summarize_results('Dir','results/baseline');
% Options:
%   'Dir'      : directory to scan (recursive=false)
%   'Limit'    : max files (default 200)
%   'SortBy'   : column to sort (default 'timestamp')
%   'Descending' : logical (default true)
%
% Extracted columns (if present):
%   method, tag, seed, rmse_final_quick, energy_est, peak_tau, StopTime, horizon
%
% Missing fields become NaN or '' accordingly.

p = inputParser;
addParameter(p,'Dir','results/baseline',@ischar);
addParameter(p,'Limit',200,@(x)isnumeric(x)&&x>0);
addParameter(p,'SortBy','timestamp',@ischar);
addParameter(p,'Descending',true,@islogical);
parse(p,varargin{:});
opt = p.Results;

if ~exist(opt.Dir,'dir')
    error('Directory %s not found', opt.Dir);
end

files_json = dir(fullfile(opt.Dir,'*.json'));
files_mat  = dir(fullfile(opt.Dir,'*.mat'));
files = [files_json; files_mat];

if isempty(files)
    warning('No result files in %s', opt.Dir); T = table(); return; end

% Limit
files = files(1:min(numel(files), opt.Limit));

rows = [];
for i=1:numel(files)
    f = files(i);
    path = fullfile(f.folder,f.name);
    try
        if endsWith(f.name,'.json')
            txt = fileread(path);
            S = jsondecode(txt);
        else
            L = load(path,'results');
            S = L.results;
        end
    catch ME
        fprintf('[summarize_results] skip %s (%s)\n', f.name, ME.message);
        continue;
    end

    row = struct();
    row.file = string(f.name);
    row.method = getStr(S,'method');
    row.tag = getStr(S,'tag');
    row.seed = getNum(S,'seed');
    row.rmse_quick = getNum(S,'rmse_final_quick');
    row.energy_est = getNum(S,'energy_est');
    row.peak_tau = getNum(S,'peak_tau');
    row.StopTime = getNum(S,'StopTime');
    row.horizon = getNum(S,'horizon');
    row.timestamp = parseTime(getStr(S,'timestamp'));
    rows = [rows; row]; %#ok<AGROW>
end

if isempty(rows)
    T = table(); return; end

T = struct2table(rows);

% Sort
if any(strcmp(opt.SortBy, T.Properties.VariableNames))
    [~,idx] = sort(T.(opt.SortBy));
    if opt.Descending, idx = flipud(idx); end
    T = T(idx,:);
end

% Display summary stats
if ~isempty(T)
    fprintf('\nSummary (%d rows)\n', height(T));
    methods = unique(T.method);
    for m = 1:numel(methods)
        k = T.method == methods(m);
        fprintf('  %s: n=%d rmse_quick(mean)=%.4g energy(mean)=%.4g\n', methods(m), sum(k), mean(T.rmse_quick(k),'omitnan'), mean(T.energy_est(k),'omitnan'));
    end
end

end

function v = getStr(S, f)
if isfield(S,f) && ischar(S.(f))
    v = string(S.(f));
elseif isfield(S,f) && isstring(S.(f))
    v = S.(f);
else
    v = string('');
end
end
function v = getNum(S,f)
if isfield(S,f) && isnumeric(S.(f)) && ~isempty(S.(f))
    v = double(S.(f)(1));
else
    v = NaN;
end
end
function t = parseTime(str)
try
    t = datetime(str,'InputFormat','yyyy-MM-dd HH:mm:ss');
catch
    try
        t = datetime(str,'InputFormat','yyyy-mm-dd HH:MM:SS');
    catch
        t = NaT;
    end
end
end