function plot_reward_components(logStruct)
%PLOT_REWARD_COMPONENTS Plot reward components over training.
% logStruct: array of structs with fields (any subset):
%   step, average_reward, critic_loss, actor_loss
% If HighLevelRL.learning_curve is passed, it should already contain these fields.

if isempty(logStruct)
    warning('Empty logStruct'); return; end

% Convert to table-like vectors
steps = [logStruct.step]';
critic = getFieldOr(logStruct,'critic_loss');
actor  = getFieldOr(logStruct,'actor_loss');
avgR   = getFieldOr(logStruct,'average_reward');

figure('Name','Learning Curve','Position',[120 120 900 600]);
subplot(3,1,1);
plot(steps, avgR,'LineWidth',1.4);
ylabel('Avg Reward'); grid on;
subplot(3,1,2);
plot(steps, critic,'LineWidth',1.2); ylabel('Critic Loss'); grid on;
subplot(3,1,3);
plot(steps, actor,'LineWidth',1.2); ylabel('Actor Loss'); xlabel('Step'); grid on;
end

function v = getFieldOr(S,f)
try
    v = [S.(f)]';
    if isempty(v), v = nan(numel(S),1); end
catch
    v = nan(numel(S),1);
end
end