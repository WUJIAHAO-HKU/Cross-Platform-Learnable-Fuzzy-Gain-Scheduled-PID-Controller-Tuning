function results = run_baseline_capture(varargin)
% RUN_BASELINE_CAPTURE  Execute baseline simulation, compute metrics, save results.
% Usage:
%   results = run_baseline_capture();
%   results = run_baseline_capture('Model','FrankaPanda_JournalSim_Practical','StopTime',10,'Tag','Baseline_v2');
%
% Parameters (Name-Value):
%   'Model'     : Simulink model name (default: 'FrankaPanda_JournalSim_Practical')
%   'StopTime'  : Simulation stop time in seconds (default: 10)
%   'Tag'       : Result tag string (default: auto timestamp)
%   'OutDir'    : Output directory for MAT + JSON (default: ./results/baseline)
%   'Seed'      : RNG seed (default: 42)
%   'SaveJSON'  : true/false (default: true)
%   'SaveMAT'   : true/false (default: true)
%
% Expected signals (To Workspace / logged):
%   q, q_ref, u, rmse_q, energy_u, err_norm, int_e2, tau (optional), t (Clock)
%
% Requirements:
%   - Model must compile & run
%   - All required signals accessible via Simulink.SimulationOutput
%
% Returns struct 'results' with fields:
%   tag, timestamp, StopTime, rmse_final, energy_final, peak_tau, settling_time,
%   overshoot_per_joint, steady_state_error, joint_rmse_vector, config
%
% NOTE: Adjust signal names below if they differ in your model.

p = inputParser;
addParameter(p,'Model','FrankaPanda_JournalSim_Practical',@ischar);
addParameter(p,'StopTime',10,@(x)isnumeric(x)&&x>0);
addParameter(p,'Tag','',@ischar);
addParameter(p,'OutDir','results/baseline',@ischar);
addParameter(p,'Seed',42,@(x)isnumeric(x)&&isscalar(x));
addParameter(p,'SaveJSON',true,@islogical);
addParameter(p,'SaveMAT',true,@islogical);
% Unified benchmarking additions
addParameter(p,'Method','PD',@ischar);        % 'PD','MPC','RL','Hybrid'
addParameter(p,'Gains',struct(),@(x) isstruct(x)||isempty(x));
addParameter(p,'Horizon',[],@(x) isempty(x)||(isnumeric(x)&&isscalar(x)&&x>0));
parse(p,varargin{:});
opt = p.Results;

if isempty(opt.Tag)
    opt.Tag = datestr(now,'yyyymmdd_HHMMSS');
end

rng(opt.Seed);
if ~bdIsLoaded(opt.Model)
    load_system(opt.Model);
end
set_param(opt.Model,'StopTime',num2str(opt.StopTime));

% Run simulation
simOut = sim(opt.Model,'ReturnWorkspaceOutputs','on');

% Helper to fetch from simOut robustly
fetch = @(name,default) ( ...
    (isprop(simOut,name) && ~isempty(simOut.(name))) *1 ); %#ok<NASGU>

% Extract time series (assuming timeseries or array output)
getTS = @(field) localGetTimeseries(simOut,field);

% Primary retrieval
q      = getTS('q');
q_ref  = getTS('q_ref');
u      = getTS('u');

% Alias fallbacks for q
if isempty(q)

% Additional attempt: reconstruct q_ref from performance_metrics if it stores both actual & reference vectors
if isempty(q_ref) && isprop(simOut,'performance_metrics')
    pm = simOut.performance_metrics;
    if isstruct(pm)
        cands = fieldnames(pm);
        bestScore = inf; bestTS = [];
        for ic = 1:numel(cands)
            ts_try = localStructToTimeseries(pm.(cands{ic}));
            if isempty(ts_try), continue; end
            if size(ts_try.Data,2) == 7 && exist('q','var') && ~isempty(q) && numel(ts_try.Data)==numel(q.Data)
                % Score: final difference vs q larger than small threshold (avoid picking q itself)
                diffFinal = norm(ts_try.Data(end,:) - q.Data(end,:));
                if diffFinal > 1e-6 && diffFinal < bestScore
                    bestScore = diffFinal; bestTS = ts_try; %#ok<NASGU>
                end
            end
        end
        if exist('bestTS','var') && ~isempty(bestTS)
            q_ref = bestTS; 
        end
    end
end
    q_aliases = {'q_log','joint_positions','joint_pos','joint_position'};
    for ia = 1:numel(q_aliases)
        q = getTS(q_aliases{ia});
        if ~isempty(q), break; end
    end
end

% Alias fallbacks for q_ref (reference trajectory)
if isempty(q_ref)
    qref_aliases = {'q_ref_ts','reference_signal','reference_traj','qref','q_ref_log'};
    for ia = 1:numel(qref_aliases)
        q_ref = getTS(qref_aliases{ia});
        if ~isempty(q_ref), break; end
    end
end

% Alias fallbacks for u (torque)
if isempty(u)
    u_aliases = {'u_log','u_log1','outu_log1','control_torques','torque','tau'}; % added outu_log1
    for ia = 1:numel(u_aliases)
        u = getTS(u_aliases{ia});
        if ~isempty(u), break; end
    end
end

% If still empty, try to derive from struct fields (e.g., control_torques / joint_positions inside SimulationOutput)
if isempty(q) || isempty(q_ref) || isempty(u)
    % Examine all properties that are structs and try converting
    props = properties(simOut);
    for ip = 1:numel(props)
        if ~isempty(q) && ~isempty(q_ref) && ~isempty(u), break; end
        pName = props{ip};
        val = simOut.(pName);
        if isstruct(val)
            % Try keys inside struct
            fns = fieldnames(val);
            for kf = 1:numel(fns)
                sub = val.(fns{kf});

if isempty(q_ref)
    % Provide diagnostic listing of available candidates
    diagMsg = listAvailableCandidates(simOut);
    warning(['Reference signal (q_ref) not found. Add To Workspace for reference trajectory named q_ref ', ...
        'or q_ref_ts. Available time-series like signals: ', diagMsg]);
end
                ts_try = localStructToTimeseries(sub);
                if isempty(ts_try), continue; end
                if isempty(q) && size(ts_try.Data,2) == 7
                    q = ts_try; continue; end
                if isempty(u) && size(ts_try.Data,2) == 7
                    u = ts_try; continue; end
                % Heuristic: reference often similar dimension
                if isempty(q_ref) && size(ts_try.Data,2) == 7
                    % Additional check: first sample maybe equals initial q
                    q_ref = ts_try; continue; end
            end
        end
    end
end

if isempty(q)
    struct_fallbacks = {'joint_positions','q_log'}; % legacy minimal
    for sf = 1:numel(struct_fallbacks)
        if isprop(simOut, struct_fallbacks{sf})
            val = simOut.(struct_fallbacks{sf});
            ts_try = localStructToTimeseries(val);
            if ~isempty(ts_try), q = ts_try; break; end
        end
    end
end
if isempty(u)
    struct_fallbacks_u = {'control_torques','u_struct'};
    for sf = 1:numel(struct_fallbacks_u)
        if isprop(simOut, struct_fallbacks_u{sf})
            val = simOut.(struct_fallbacks_u{sf});
            ts_try = localStructToTimeseries(val);
            if ~isempty(ts_try), u = ts_try; break; end
        end
    end
end
% Base workspace numeric fallback (user may have To Workspace saved as plain array)
if isempty(u)
    bw_names = {'u','u_log1','outu_log1','tau'};
    for ib=1:numel(bw_names)
        try
            if evalin('base',sprintf('exist(''%s'',''var'')',bw_names{ib}))
                raw = evalin('base',bw_names{ib});
                if isnumeric(raw) && ~isempty(raw)
                    % Construct synthetic time vector (uniform) using StopTime
                    N = size(raw,1);
                    if N>1
                        t_raw = linspace(0,opt.StopTime,N)';
                    else
                        t_raw = 0; % single sample
                    end
                    try
                        u = timeseries(raw,t_raw);
                        fprintf('[run_baseline_capture] Adopted base workspace numeric %s (%d samples) as torque signal.\n',bw_names{ib},N);
                        break;
                    catch
                    end
                end
            end
        catch
        end
    end
end
rmse_q = getTS('rmse_q');
energy_u = getTS('energy_u');
err_norm = getTS('err_norm');
int_e2   = getTS('int_e2');
if isprop(simOut,'tau')
    tau = getTS('tau');
else
    tau = u; % fallback
end

% Basic time vector
if ~isempty(q)
    t = q.Time;
else
    error(['Signal q missing or empty. Tried names: q, q_log, joint_positions, joint_pos, joint_position. ', ...
        'Please add a To Workspace block (name q, format timeseries) or rename existing signal.']);
end

% Joint-wise RMSE computation
if ~isempty(q) && ~isempty(q_ref)
    err = q.Data - q_ref.Data;
    joint_rmse = sqrt(mean(err.^2,1));
else
    joint_rmse = [];
end

% Final scalar metrics
rmse_final   = lastScalar(rmse_q);
energy_final = lastScalar(energy_u);

% Peak torque
if ~isempty(tau)
    peak_tau = max(abs(tau.Data),[], 'all');
else
    peak_tau = NaN;
end

% Overshoot & settling (per joint) relative to final reference value
[overshoot_per_joint, settling_time, steady_state_error] = localStepMetrics(t, q, q_ref, 0.02, 0.02);

% --- rmse_quick instrumentation (minimal) ---
rmse_final_quick = NaN; rmse_quick_ts = [];
if ~isempty(q) && ~isempty(q_ref)
    try
        err_mat = q.Data - q_ref.Data;
        if ndims(err_mat)==3 % flatten if needed
            err_mat = squeeze(err_mat);
        end
        rmse_series = sqrt(mean(err_mat.^2,2));
        rmse_quick_ts = timeseries(rmse_series, t);
        rmse_final_quick = rmse_series(end);
    catch ME
        warning('rmse_quick computation failed: %s', ME.message);
    end
end

% --- energy estimate (if not provided) ---
energy_est = energy_final; % fallback
% If u missing but tau exists, use tau for energy estimate
if ( (isempty(u) || isempty(u.Data)) && exist('tau','var') && ~isempty(tau) )
    u = tau; % adopt tau as control signal proxy
end
if (isnan(energy_est) || isempty(energy_est)) && ~isempty(u)
    try
        udata = u.Data; if ndims(udata)==3, udata = squeeze(udata); end
        if any(isnan(udata),'all')
            udata_clean = udata; udata_clean(isnan(udata_clean)) = 0; % treat NaN as zero contribution
        else
            udata_clean = udata;
        end
        % If sample counts mismatch, interpolate to q timeline length
        if numel(t) ~= size(udata_clean,1)
            t_u = linspace(t(1), t(end), size(udata_clean,1))';
            try
                udata_interp = interp1(t_u, udata_clean, t, 'previous', 'extrap');
            catch
                % fallback simple repeat
                idx = min(round(linspace(1,size(udata_clean,1),numel(t)))', size(udata_clean,1));
                udata_interp = udata_clean(idx,:);
            end
        else
            udata_interp = udata_clean;
        end
        energy_est = trapz(t, sum(udata_interp.^2,2));
        if size(udata,1) < 5
            fprintf(['[run_baseline_capture] NOTE: torque samples very few (%d). Energy approximated by interpolation. ', ...
                     'Add To Workspace block on torque (timeseries, name u_log1) for accurate energy.\n'], size(udata,1));
        end
    catch ME
        warning('energy_est computation failed: %s', ME.message);
        energy_est = NaN;
    end
elseif isnan(energy_est) || isempty(energy_est)
    fprintf('[run_baseline_capture] NOTE: No torque signal (u) found; energy_est remains NaN. Add To Workspace block named u or u_log1 (timeseries).\n');
end

results = struct();
% Unified benchmark header
results.method = opt.Method;
results.gains = opt.Gains;
results.horizon = opt.Horizon;
results.seed = opt.Seed;
% Legacy / extended metrics
results.tag = opt.Tag;
results.timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS');
results.StopTime = opt.StopTime;
results.rmse_final = rmse_final;              % legacy (from rmse_q signal if present)
results.rmse_final_quick = rmse_final_quick;  % new minimal RMSE over entire run
results.energy_final = energy_final;
results.energy_est = energy_est;              % derived energy (sum u^2 dt)
results.peak_tau = peak_tau;
results.settling_time = settling_time;
results.overshoot_per_joint = overshoot_per_joint;
results.steady_state_error = steady_state_error;
results.joint_rmse_vector = joint_rmse;
results.config = opt;
% Store raw signals for downstream analysis
sigStruct = struct();
sigStruct.q = q; %#ok<STRNU>
sigStruct.q_ref = q_ref;
sigStruct.u = u;
sigStruct.rmse_q = rmse_q;
if ~isempty(rmse_quick_ts); sigStruct.rmse_quick = rmse_quick_ts; end
sigStruct.energy_u = energy_u;
sigStruct.err_norm = err_norm;
sigStruct.int_e2 = int_e2;
if exist('tau','var'); sigStruct.tau = tau; end
results.signals = sigStruct;

% Ensure output directory
if ~exist(opt.OutDir,'dir')
    mkdir(opt.OutDir);
end
matPath = fullfile(opt.OutDir, [opt.Tag '.mat']);
jsonPath = fullfile(opt.OutDir, [opt.Tag '.json']);

if opt.SaveMAT
    save(matPath,'results');
end
if opt.SaveJSON
    try
        jsonFriendly = results;
        % Remove function handles / non-serializable objects if any
        jsonFriendly = stripNonSerializable(jsonFriendly);
        jsonText = jsonencode(jsonFriendly);
        fid = fopen(jsonPath,'w'); fwrite(fid,jsonText,'char'); fclose(fid);
    catch ME
        warning('JSON save failed: %s', ME.message);
    end
end

fprintf('[run_baseline_capture] Saved results to %s\n', opt.OutDir);

end

function ts = localGetTimeseries(simOut, field)
% Try multiple strategies to retrieve a signal as timeseries:
% 1. Direct property (timeseries)
% 2. Direct property (numeric array) + tout
% 3. logsout element
% 4. Case-insensitive match over properties
    ts = [];
    % 1 & 2 direct property
    if isprop(simOut, field)
        val = simOut.(field);
        ts = convertValue(val, simOut);
        if ~isempty(ts); return; end
    end
    % 4 case-insensitive property scan
    props = properties(simOut);
    ci = strcmpi(props, field);
    if any(ci)
        val = simOut.(props{find(ci,1)});
        ts = convertValue(val, simOut);
        if ~isempty(ts); return; end
    end
    % 3 logsout search
    if isprop(simOut,'logsout') && ~isempty(simOut.logsout)
        try
            el = simOut.logsout.get(field);
            if ~isempty(el)
                val = el.Values;
                ts = convertValue(val, simOut);
                if ~isempty(ts); return; end
            end
        catch %#ok<CTCH>
        end
        % try case-insensitive in logsout
        try
            names = arrayfun(@(x) x.Name, simOut.logsout, 'UniformOutput', false);
            ci2 = strcmpi(names, field);
            if any(ci2)
                el = simOut.logsout(ci2);
                val = el.Values;
                ts = convertValue(val, simOut);
                if ~isempty(ts); return; end
            end
        catch %#ok<CTCH>
        end
    end
end

function ts = convertValue(val, simOut)
    ts = [];
    if isa(val,'timeseries')
        ts = val; return;
    end
    if isnumeric(val)
        % Attempt to pair with tout or simulation time inside structure
        t = [];
        if isprop(simOut,'tout') && ~isempty(simOut.tout)
            t = simOut.tout;
        elseif isprop(simOut,'time') && ~isempty(simOut.time)
            t = simOut.time;
        elseif size(val,1) > 1
            t = (0:size(val,1)-1)'; % fallback index time
        end
        if ~isempty(t)
            % If val is (N x channels) or (N x 1 x channels)
            data = val;
            if ndims(data)==3 && size(data,2)==1
                data = squeeze(data);
            end
            try
                ts = timeseries(data, t);
            catch
                ts = [];
            end
        end
        return;
    end
    % logsout element type
    if isstruct(val) && isfield(val,'Time') && isfield(val,'Data')
        try
            ts = timeseries(val.Data, val.Time);
            return;
        catch
        end
    end
end

function ts = localStructToTimeseries(s)
    ts = [];
    if isa(s,'timeseries'); ts = s; return; end
    if ~isstruct(s); return; end
    % Expect fields Time / Data (common pattern) OR t / y
    candTime = {};
    if isfield(s,'Time'), candTime{end+1} = s.Time; end %#ok<AGROW>
    if isfield(s,'t'), candTime{end+1} = s.t; end %#ok<AGROW>
    candData = {};
    if isfield(s,'Data'), candData{end+1} = s.Data; end %#ok<AGROW>
    if isfield(s,'y'), candData{end+1} = s.y; end %#ok<AGROW>
    if isempty(candTime) || isempty(candData); return; end
    T = candTime{1}; D = candData{1};
    if ~isvector(T) || isempty(D); return; end
    try
        ts = timeseries(D, T);
    catch
        ts = [];
    end
end

function S = stripNonSerializable(S)
    if isstruct(S)
        f = fieldnames(S);
        for i=1:numel(f)
            val = S.(f{i});
            if isa(val,'function_handle') || isa(val,'Simulink.SimulationMetadata')
                S.(f{i}) = []; continue; end
            if isstruct(val)
                S.(f{i}) = stripNonSerializable(val);
            elseif iscell(val)
                for c=1:numel(val)
                    if isa(val{c},'function_handle')
                        val{c} = [];
                    elseif isstruct(val{c})
                        val{c} = stripNonSerializable(val{c});
                    end
                end
                S.(f{i}) = val;
            end
        end
    end
end

function msg = listAvailableCandidates(simOut)
    msg = '';
    try
        props = properties(simOut);
        names = {};
        for i=1:numel(props)
            val = simOut.(props{i});
            if isa(val,'timeseries')
                sz = size(val.Data);
                names{end+1} = sprintf('%s(timeseries %s)',props{i},mat2str(sz)); %#ok<AGROW>
            elseif isstruct(val)
                fns = fieldnames(val);
                for k=1:numel(fns)
                    sub = val.(fns{k});
                    if isa(sub,'timeseries')
                        sz = size(sub.Data);
                        names{end+1} = sprintf('%s.%s(timeseries %s)',props{i},fns{k},mat2str(sz)); %#ok<AGROW>
                    end
                end
            end
        end
        msg = strjoin(names, ', ');
    catch
        msg = 'unavailable';
    end
end

function v = lastScalar(ts)
    if isempty(ts)
        v = NaN; return; end
    try
        d = ts.Data;
        v = d(end);
    catch
        v = NaN;
    end
end

function [overshoot, settling_time, ss_error] = localStepMetrics(t, q, q_ref, settleTol, ssTol)
    if isempty(q) || isempty(q_ref)
        overshoot = []; settling_time = NaN; ss_error = []; return; end
    data = q.Data; ref = q_ref.Data;
    final_ref = squeeze(ref(end,:,:));
    if ndims(data)==3
        data2 = squeeze(data); ref2 = squeeze(ref);
    else
        data2 = data; ref2 = ref;
    end
    nJ = size(data2,2);
    overshoot = zeros(1,nJ);
    settling_time = NaN; % global (max across joints)
    ss_error = zeros(1,nJ);
    st_times = zeros(1,nJ);
    for j=1:nJ
        y = data2(:,j); rj = ref2(:,j); r_final = rj(end);
        if abs(r_final) < 1e-9
            ref_mag = max(abs(rj));
            if ref_mag < 1e-9
                overshoot(j) = 0; ss_error(j)=0; st_times(j)=t(end); continue; end
            r_final = rj(end); % fallback
        end
        overshoot(j) = (max(y)-r_final)/max(abs(r_final),1e-9);
        ss_error(j) = (y(end)-r_final);
        band = settleTol * max(1,abs(r_final));
        idx = find(abs(y - r_final) <= band, 1, 'first');
        if ~isempty(idx)
            % ensure it stays inside band thereafter
            inside = all(abs(y(idx:end)-r_final) <= band);
            if inside
                st_times(j) = t(idx);
            else
                st_times(j) = t(end);
            end
        else
            st_times(j) = t(end);
        end
    end
    settling_time = max(st_times);
end
