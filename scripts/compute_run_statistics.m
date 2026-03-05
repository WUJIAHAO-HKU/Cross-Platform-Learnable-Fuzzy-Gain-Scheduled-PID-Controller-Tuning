function stats = compute_run_statistics(simOutOrPath, varargin)
% COMPUTE_RUN_STATISTICS  Post-process simulation outputs for advanced metrics.
%
% Usage:
%   stats = compute_run_statistics(simOut);
%   stats = compute_run_statistics('results/baseline/Baseline_v1.mat');
%
% Optional Name-Value:
%   'Window'      : smoothing window for moving averages (default 51 samples)
%   'Quantiles'   : vector of quantiles for error distribution (default [0.5 0.9 0.95 0.99])
%   'SavePath'    : if provided, save JSON summary
%
% Outputs struct stats fields:
%   .rmse_final, .energy_final, .overshoot_per_joint, .settling_time,
%   .steady_state_error, .joint_rmse_vector, .error_quantiles,
%   .max_velocity, .mean_abs_torque, .control_energy_density
%
% Expects signals: q, q_ref, u, (optional) rmse_q, energy_u, err_norm

p = inputParser;
addParameter(p,'Window',51,@(x)isnumeric(x)&&isscalar(x));
addParameter(p,'Quantiles',[0.5 0.9 0.95 0.99],@(x)isnumeric(x));
addParameter(p,'SavePath','',@ischar);
parse(p,varargin{:});
opt = p.Results;

if ischar(simOutOrPath) || isstring(simOutOrPath)
    S = load(simOutOrPath,'results');
    if isfield(S,'results'); simOut = S.results; else; error('File missing results struct'); end
else
    simOut = simOutOrPath;
end

% Extract timeseries or arrays
q   = fetchTS(simOut,'q');
qR  = fetchTS(simOut,'q_ref');
u   = fetchTS(simOut,'u');
rmse_ts = fetchTS(simOut,'rmse_q');
energy_ts = fetchTS(simOut,'energy_u');

% If missing, attempt nested signals struct (results.signals.*)
if (isempty(q) || isempty(qR)) && isstruct(simOut) && isfield(simOut,'signals')
    nested = simOut.signals;
    if isempty(q);  q  = fetchTS(nested,'q'); end
    if isempty(qR); qR = fetchTS(nested,'q_ref'); end
    if isempty(u);  u  = fetchTS(nested,'u'); end
    if isempty(rmse_ts); rmse_ts = fetchTS(nested,'rmse_q'); end
    if isempty(energy_ts); energy_ts = fetchTS(nested,'energy_u'); end
end

if isempty(q) || isempty(qR)
    error('q or q_ref missing');
end

[t, qData] = toArray(q);
[~, qRData] = toArray(qR);
if ~isempty(u)
    [~, uData] = toArray(u); else; uData = []; end

err = qRData - qData; % N x J
rmse_vec = sqrt(mean(err.^2,1));
rmse_final = nan;
if ~isempty(rmse_ts)
    rmse_final = rmse_ts.Data(end);
else
    rmse_final = sqrt(mean(err(:).^2));
end

energy_final = nan;
if ~isempty(energy_ts)
    energy_final = energy_ts.Data(end); end

% Overshoot / settling reuse baseline logic (copy in minimal form)
[overshoot_per_joint, settling_time, steady_state_error] = stepMetrics(t,qData,qRData,0.02);

% Error distribution quantiles
abs_err = abs(err(:));
error_quantiles = quantile(abs_err, opt.Quantiles);

% Control stats
if ~isempty(uData)
    mean_abs_torque = mean(abs(uData),'all');
    control_energy_density = energy_final / t(end);
else
    mean_abs_torque = NaN; control_energy_density = NaN;
end

% Max joint velocity (finite diff)
qd_est = diff(qData)./diff(t);
max_velocity = max(abs(qd_est),[],1);

stats = struct();
stats.rmse_final = rmse_final;
stats.energy_final = energy_final;
stats.overshoot_per_joint = overshoot_per_joint;
stats.settling_time = settling_time;
stats.steady_state_error = steady_state_error;
stats.joint_rmse_vector = rmse_vec;
stats.error_quantiles = table(opt.Quantiles(:), error_quantiles(:), 'VariableNames',{'q','abs_err'});
stats.max_velocity = max_velocity;
stats.mean_abs_torque = mean_abs_torque;
stats.control_energy_density = control_energy_density;

if ~isempty(opt.SavePath)
    summary = stats; summary.error_quantiles = table2struct(summary.error_quantiles);
    try
        jsonText = jsonencode(summary);
        fid = fopen(opt.SavePath,'w'); fwrite(fid,jsonText,'char'); fclose(fid);
    catch ME
        warning('Could not save JSON summary: %s', ME.message);
    end
end

end

function ts = fetchTS(simOut, field)
    if isstruct(simOut) && isfield(simOut, field)
        val = simOut.(field); else; ts = []; return; end
    if isa(val,'timeseries')
        ts = val; return;
    else
        ts = [];
    end
end

function [t, data] = toArray(ts)
    t = ts.Time; data = ts.Data;
    if ndims(data)==3; data = squeeze(data); end
end

function [overshoot, settling_time, ss_error] = stepMetrics(t,y,yref, tol)
    if nargin<4; tol=0.02; end
    [N,J] = size(y);
    overshoot = zeros(1,J); ss_error = zeros(1,J); settle_each = zeros(1,J);
    for j=1:J
        yj = y(:,j); rj = yref(:,j); r_final = rj(end);
        if abs(r_final) < 1e-9
            r_final = rj(end);
        end
        overshoot(j) = (max(yj)-r_final)/max(abs(r_final),1e-9);
        ss_error(j) = yj(end)-r_final;
        band = tol*max(1,abs(r_final));
        idx = find(abs(yj-r_final)<=band,1,'first');
        if isempty(idx); settle_each(j)=t(end); else
            if all(abs(yj(idx:end)-r_final)<=band)
                settle_each(j)=t(idx); else; settle_each(j)=t(end); end
        end
    end
    settling_time = max(settle_each);
end
