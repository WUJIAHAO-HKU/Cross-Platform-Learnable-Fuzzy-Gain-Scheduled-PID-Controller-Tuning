function report = check_logging_setup(varargin)
% CHECK_LOGGING_SETUP  Diagnose whether key signals are logged at sufficient fidelity.
% Usage:
%   check_logging_setup();
%   report = check_logging_setup('Model','FrankaPanda_JournalSim_Practical');
%
% What it checks:
%   1. q_ref exists (timeseries) and length >= 10
%   2. torque signal (u / u_log1 / outu_log1 / tau) is timeseries with sample count comparable to q
%   3. Warns if only a tiny numeric array (<= 5 samples) is present (energy_est will be rough)
%   4. Suggests exact To Workspace settings for improved energy accuracy
%
% Returns a struct report with fields:
%   has_q_ref, has_torque_ts, q_len, u_len, fidelity_ratio, recommendations (cell array)
%
% This does NOT modify the model; it's a passive diagnostic.

p = inputParser;
addParameter(p,'Model','FrankaPanda_JournalSim_Practical',@ischar);
addParameter(p,'Results',[],@(x) isempty(x) || isstruct(x)); % optionally pass results struct from run_baseline_capture to avoid re-run
parse(p,varargin{:});
opt = p.Results;

report = struct('has_q_ref',false,'has_torque_ts',false,'q_len',0,'u_len',0,'fidelity_ratio',NaN,'recommendations',{{}});
recs = {};

% Acquire signals either from provided results or workspace variables
if ~isempty(opt.Results)
    R = opt.Results;
    if isfield(R,'signals')
        sigs = R.signals;
    else
        sigs = struct();
    end
else
    % Fallback: try grabbing variables from base workspace (after a sim)
    baseVars = {'q_ref','q','u_log1','u','tau'};
    sigs = struct();
    for i=1:numel(baseVars)
        if evalin('base',sprintf('exist(''%s'',''var'')',baseVars{i}))
            sigs.(baseVars{i}) = evalin('base',baseVars{i});
        end
    end
end

% Helper to standardize
normalizeTS = @(v) (isa(v,'timeseries'));

% q_ref
if isfield(sigs,'q_ref') && normalizeTS(sigs.q_ref)
    report.has_q_ref = true;
    report.q_len = numel(sigs.q_ref.Time);
else
    recs{end+1} = 'Add To Workspace (Variable: q_ref, Format: timeseries) to reference trajectory branch.'; %#ok<AGROW>
end

% torque candidates ordered by priority
candTorque = {'u_log1','u','tau','outu_log1'};
u_ts = [];
for k=1:numel(candTorque)
    if isfield(sigs,candTorque{k}) && normalizeTS(sigs.(candTorque{k}))
        u_ts = sigs.(candTorque{k});
        break;
    end
end
if ~isempty(u_ts)
    report.has_torque_ts = true;
    report.u_len = numel(u_ts.Time);
else
    % maybe numeric short array
    for k=1:numel(candTorque)
        if isfield(sigs,candTorque{k}) && isnumeric(sigs.(candTorque{k}))
            report.u_len = size(sigs.(candTorque{k}),1);
            recs{end+1} = sprintf('Torque signal %s is numeric array (len=%d). Use To Workspace timeseries for accurate energy.',candTorque{k},report.u_len); %#ok<AGROW>
            break;
        end
    end
    if report.u_len==0
        recs{end+1} = 'Add To Workspace on torque output (Variable: u_log1, Format: timeseries).'; %#ok<AGROW>
    end
end

if report.has_q_ref && report.has_torque_ts
    report.fidelity_ratio = report.u_len / max(report.q_len,1);
    if report.fidelity_ratio < 0.9
        recs{end+1} = sprintf('Torque sampling undersampled (ratio=%.2f). Ensure same sample time / no decimation.',report.fidelity_ratio); %#ok<AGROW>
    end
end

% Additional targeted advice
if report.q_len < 10 && report.has_q_ref
    recs{end+1} = 'Reference trajectory too short (<10 samples); confirm StopTime and sample time.'; %#ok<AGROW>
end
if report.u_len > 0 && report.u_len < 10
    recs{end+1} = 'Torque samples extremely few; energy will be rough. Log as high-resolution timeseries.'; %#ok<AGROW>
end

report.recommendations = recs;

fprintf('--- Logging Diagnostic ---\n');
fprintf('q_ref: %s (len=%d)\n', tf(report.has_q_ref), report.q_len);
fprintf('torque ts: %s (len=%d)\n', tf(report.has_torque_ts), report.u_len);
if ~isnan(report.fidelity_ratio)
    fprintf('fidelity ratio (u vs q): %.3f\n', report.fidelity_ratio);
end
if isempty(recs)
    fprintf('No recommendations: logging setup looks good.\n');
else
    fprintf('Recommendations:\n');
    for i=1:numel(recs)
        fprintf('  - %s\n', recs{i});
    end
end

end

function s = tf(b)
if b, s='YES'; else, s='NO'; end
end