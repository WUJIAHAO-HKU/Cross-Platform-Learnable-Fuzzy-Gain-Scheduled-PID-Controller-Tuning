function traj = prepare_quintic_qref(modelName, q0, qT, T, dt)
% PREPARE_QUINTIC_QREF  生成五次多项式关节参考并放入 base workspace 供模型作为 qref 输入
%
% 用法示例：
%   % 1) 基本用法（自动从模型读 StopTime/FixedStep，7 关节零到小幅目标）
%   prepare_quintic_qref('FrankaPanda_JournalSim_Practical');
%
%   % 2) 自定义目标、时长与步长
%   q0 = zeros(1,7);
%   qT = [0.5,-0.3,0.2, 0.1, -0.1, 0.15, 0];
%   prepare_quintic_qref('FrankaPanda_JournalSim_Practical', q0, qT, 5.0, 0.001);
%
% 该函数会在 base workspace 放入：
%   qref_ts    : timeseries，Data 维度 [N x 7]，time [N x 1]
%   qref_array : 数组 [N x 7]（便于脚本使用）
%   t_qref     : 时间向量 [N x 1]
%
% 注意：From Workspace 直接读 timeseries 时通常输出 [1x7] 行向量，
% 若你的控制栈需要 [7x1] 列向量，可在模型里在 From Workspace 后接一个 Reshape，设 OutputDimensions=[7 1]。

if nargin < 1 || isempty(modelName)
    modelName = 'FrankaPanda_JournalSim_Practical';
end

% 读取模型仿真时间与步长
try
    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end
    if nargin < 4 || isempty(T)
        StopTime = get_param(modelName,'StopTime');
        if strcmpi(StopTime,'auto') || isempty(StopTime)
            T = 5.0;
        else
            T = str2double(StopTime);
            if ~isfinite(T) || isnan(T), T = 5.0; end
        end
    end
    if nargin < 5 || isempty(dt)
        FixedStep = get_param(modelName,'FixedStep');
        dt = str2double(FixedStep);
        if ~isfinite(dt) || isnan(dt) || dt<=0
            dt = 0.001; % fallback
        end
    end
catch
    if nargin < 4 || isempty(T),  T  = 5.0;  end
    if nargin < 5 || isempty(dt), dt = 0.001; end
end

% 关节维数与边界目标（默认 7 关节）
if nargin < 2 || isempty(q0)
    q0 = zeros(1,7);
end
if nargin < 3 || isempty(qT)
    qT = [0.5, -0.3, 0.2, 0.1, -0.1, 0.15, 0.0];
    if numel(q0) ~= numel(qT)
        qT = zeros(size(q0));
        qT(1:min(7,numel(q0))) = [0.5, -0.3, 0.2, 0.1, -0.1, 0.15, 0.0];
        qT = qT(1:numel(q0));
    end
end

% 生成轨迹
traj = generate_quintic_joint_trajectory(q0, qT, T, dt);

% timeseries（Data: [N x 7]）
qref_ts = timeseries(traj.q, traj.t);

% 推送到 base workspace
assignin('base','qref_ts', qref_ts);
assignin('base','qref_array', traj.q);
assignin('base','t_qref', traj.t);

fprintf('[prepare_quintic_qref] 轨迹已生成: N=%d, dt=%.4g, T=%.3f\n', numel(traj.t), dt, T);
fprintf('[prepare_quintic_qref] 已写入 base: qref_ts (timeseries), qref_array (Nx7), t_qref (Nx1)\n');
fprintf('[prepare_quintic_qref] 在模型中使用建议：From Workspace -> Reshape([7 1]) -> qref 输入\n');

end


