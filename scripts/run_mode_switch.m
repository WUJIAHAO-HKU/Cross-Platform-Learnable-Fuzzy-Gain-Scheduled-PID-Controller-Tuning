function result = run_mode_switch(varargin)
% RUN_MODE_SWITCH Quick entry to run baseline capture for different control modes.
% Usage:
%   run_mode_switch('Mode','PD');
%   run_mode_switch('Mode','MPC','Horizon',20);
%   run_mode_switch('Mode','Hybrid','Seed',123,'StopTime',15);
%
% Parameters:
%   Mode     : 'PD' | 'MPC' | 'RL' | 'Hybrid'
%   Model    : Simulink model name (default FrankaPanda_JournalSim_Practical)
%   StopTime : Simulation stop time (s)
%   Horizon  : MPC horizon (optional)
%   Seed     : RNG seed
%   Gains    : struct of gains (e.g., PD gains) for logging
%   Tag      : custom tag (defaults auto)
%
% This script sets necessary model workspace variables / masks to select
% mode, then calls run_baseline_capture with unified result structure.
%
% NOTE: You must implement inside the Simulink model a mode selection
% variable, e.g., CONTROL_MODE taking integer codes:
%   0 = PD, 1 = MPC, 2 = RL, 3 = Hybrid
% And ensure the model uses that variable to route signals.
%
% If such variable does not yet exist, adapt code below accordingly.

p = inputParser;
addParameter(p,'Mode','PD',@ischar);
addParameter(p,'Model','FrankaPanda_JournalSim_Practical',@ischar);
addParameter(p,'StopTime',10,@(x)isnumeric(x)&&x>0);
addParameter(p,'Horizon',[],@(x) isempty(x)||(isnumeric(x)&&isscalar(x)&&x>0));
addParameter(p,'Seed',42,@(x)isnumeric(x)&&isscalar(x));
addParameter(p,'Gains',struct(),@(x) isstruct(x)||isempty(x));
addParameter(p,'Tag','',@ischar);
parse(p,varargin{:});
opt = p.Results;

modeMap = struct('PD',0,'MPC',1,'RL',2,'Hybrid',3);
if ~isfield(modeMap,opt.Mode)
    error('Unknown Mode %s', opt.Mode);
end
modeCode = modeMap.(opt.Mode);

% Load model if needed
if ~bdIsLoaded(opt.Model)
    load_system(opt.Model);
end

% Push variables to base workspace (simplest integration path)
assignin('base','CONTROL_MODE',modeCode);
if ~isempty(opt.Horizon)
    assignin('base','MPC_HORIZON',opt.Horizon);
end
if ~isempty(fieldnames(opt.Gains))
    assignin('base','LOGGED_GAINS',opt.Gains);
end

% Auto tag if absent
if isempty(opt.Tag)
    opt.Tag = sprintf('%s_%s', opt.Mode, datestr(now,'HHMMSS'));
end

% Call capture
result = run_baseline_capture('Model',opt.Model,'StopTime',opt.StopTime, ...
    'Tag',opt.Tag,'Seed',opt.Seed,'Method',opt.Mode,'Horizon',opt.Horizon, ...
    'Gains',opt.Gains);

% Convenience print
fprintf('[run_mode_switch] Mode=%s rmse_quick=%.4g energy_est=%.4g savedTag=%s\n', ...
    opt.Mode, result.rmse_final_quick, result.energy_est, result.tag);

end