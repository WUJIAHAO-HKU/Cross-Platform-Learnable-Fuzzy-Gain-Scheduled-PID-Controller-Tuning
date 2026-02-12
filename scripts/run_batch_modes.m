function table_out = run_batch_modes(varargin)
% RUN_BATCH_MODES Run a small batch across modes PD/MPC/RL/Hybrid.
% Usage:
%   run_batch_modes();
%   run_batch_modes('StopTime',8,'Repeats',2);
%
% Returns a MATLAB table with unified result metrics.

p = inputParser;
addParameter(p,'Modes',{'PD','MPC','RL','Hybrid'}); % choose subset if desired
addParameter(p,'StopTime',6,@(x)isnumeric(x)&&x>0);
addParameter(p,'Repeats',1,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'Seed',42,@(x)isnumeric(x)&&isscalar(x));
addParameter(p,'Model','FrankaPanda_JournalSim_Practical',@ischar);
parse(p,varargin{:});
opt = p.Results;

rows = [];
for r=1:opt.Repeats
    baseSeed = opt.Seed + (r-1)*1000;
    for m=1:numel(opt.Modes)
        mode = opt.Modes{m};
        fprintf('\n[run_batch_modes] Repeat %d Mode %s\n', r, mode);
        res = run_mode_switch('Mode',mode,'StopTime',opt.StopTime, ...
            'Seed',baseSeed,'Model',opt.Model);
        row = struct();
        row.mode = string(mode);
        row.seed = baseSeed;
        row.rmse_quick = res.rmse_final_quick;
        row.energy_est = res.energy_est;
        row.tag = string(res.tag);
        rows = [rows; row]; %#ok<AGROW>
    end
end

table_out = struct2table(rows);

% Save aggregate
outDir = 'results/batch'; if ~exist(outDir,'dir'), mkdir(outDir); end
fname = fullfile(outDir, sprintf('batch_%s.mat', datestr(now,'yyyymmdd_HHMMSS')));
save(fname,'table_out');

try
    writetable(table_out, fullfile(outDir,'latest_batch.csv'));
catch ME
    warning('Failed to write CSV: %s', ME.message);
end

fprintf('\n[run_batch_modes] Completed. Rows=%d Saved=%s\n', height(table_out), fname);
end