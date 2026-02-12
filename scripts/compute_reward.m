function R = compute_reward(obs, action, info, cfg)
%COMPUTE_REWARD  Minimal reward shaping placeholder.
% R = compute_reward(obs, action, info, cfg)
%
% Inputs:
%   obs    : current observation vector (state abstraction output)
%   action : action vector (normalized or physical)
%   info   : struct with optional fields:
%              .q          current joint positions
%              .q_ref      reference joint positions
%              .u          torque command
%              .tracking_err (precomputed vector) overrides q - q_ref
%              .stability_margin (optional scalar)
%              .safety_violation (boolean)
%   cfg    : struct with tuning weights (fields optional):
%              .w_track, .w_energy, .w_smooth, .w_stability, .w_safety
%
% Output:
%   R : struct with fields
%        .r_total   scalar reward
%        .r_track   tracking component (negative squared error)
%        .r_energy  energy penalty
%        .r_smooth  action variation penalty
%        .r_stability stability bonus/penalty
%        .r_safety  safety penalty
%
% Default weights chosen conservatively.

if nargin < 4 || isempty(cfg), cfg = struct(); end
w_track = getfielddef(cfg,'w_track',1.0); %#ok<GFLDDEF>
w_energy = getfielddef(cfg,'w_energy',0.001);
w_smooth = getfielddef(cfg,'w_smooth',0.01);
w_stab   = getfielddef(cfg,'w_stability',0.2);
w_safety = getfielddef(cfg,'w_safety',5.0);

% Tracking error
if isfield(info,'tracking_err') && ~isempty(info.tracking_err)
    e = info.tracking_err;
elseif all(isfield(info,{'q','q_ref'})) && ~isempty(info.q) && ~isempty(info.q_ref)
    e = info.q - info.q_ref;
else
    e = zeros(size(action)); % fallback
end
r_track = -sum(e.^2); % negative L2^2

% Energy penalty (use u if present else action)
if isfield(info,'u') && ~isempty(info.u)
    u_use = info.u;
else
    u_use = action;
end
r_energy = -sum(u_use.^2);

% Smoothness penalty needs previous action (optional)
persistent prev_action
if isempty(prev_action) || numel(prev_action) ~= numel(action)
    prev_action = zeros(size(action));
end
r_smooth = -sum((action - prev_action).^2);
prev_action = action;

% Stability margin (positive is good)
if isfield(info,'stability_margin') && ~isempty(info.stability_margin)
    r_stability = +info.stability_margin; % encourage larger margin
else
    r_stability = 0;
end

% Safety violation
if isfield(info,'safety_violation') && info.safety_violation
    r_safety = -1; % base penalty
else
    r_safety = 0;
end

% Weighted sum
r_total = w_track*r_track + w_energy*r_energy + w_smooth*r_smooth + ...
          w_stab*r_stability + w_safety*r_safety;

R = struct();
R.r_total = r_total;
R.r_track = w_track*r_track;
R.r_energy = w_energy*r_energy;
R.r_smooth = w_smooth*r_smooth;
R.r_stability = w_stab*r_stability;
R.r_safety = w_safety*r_safety;
R.e_norm = norm(e);

end

function v = getfielddef(s, name, default)
if isfield(s,name) && ~isempty(s.(name))
    v = s.(name); else, v = default; end
end