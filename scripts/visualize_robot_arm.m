function visualize_robot_arm(q, varargin)
% VISUALIZE_ROBOT_ARM Simple kinematic visualization placeholder for 7-DOF arm.
%   visualize_robot_arm(q)
%   visualize_robot_arm(q, 'DT', 0.02, 'LinkLengths', [0.333 0.316 0.0825 0.384 0.088 0.107 0.1])
%
% Inputs:
%   q : N×7 joint angles (radians)
%
% Name-Value:
%   'DT'          : frame pause time (s)
%   'LinkLengths' : 1×7 approximate link length chain (used for simple straight-line segments)
%   'ShowTrail'   : true/false show end-effector trail
%   'TrailLength' : number of past points for trail
%   'Figure'      : figure handle reuse
%
% NOTE: Placeholder only. Replace with full Franka DH or URDF-based forward kinematics later.

p = inputParser;
addParameter(p,'DT',0.02,@(x)isnumeric(x)&&isscalar(x));
addParameter(p,'LinkLengths',[0.333 0.316 0.0825 0.384 0.088 0.107 0.1],@(x)isnumeric(x));
addParameter(p,'ShowTrail',true,@islogical);
addParameter(p,'TrailLength',200,@(x)isnumeric(x));
addParameter(p,'Figure',[],@(x)ishandle(x)||isempty(x));
parse(p,varargin{:});
opt = p.Results;

[N, dof] = size(q);
if dof ~=7
    warning('Expected 7 joints, got %d. Proceeding.', dof);
end
L = opt.LinkLengths(:)';
if numel(L) < dof
    L = [L, repmat(L(end),1,dof-numel(L))];
end

if isempty(opt.Figure) || ~isvalid(opt.Figure)
    fig = figure('Name','RobotArm Visualization');
else
    fig = opt.Figure;
    figure(fig);
end
clf(fig);
ax = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); view(135,25);
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');

hLinks = gobjects(dof,1);
for i=1:dof
    hLinks(i) = plot3(ax,[0 0],[0 0],[0 0],'LineWidth',2);
end
if opt.ShowTrail
    hTrail = plot3(ax,NaN,NaN,NaN,'r.','MarkerSize',6);
else
    hTrail = []; %#ok<NASGU>
end

trailX = []; trailY = []; trailZ = [];

for k=1:N
    qk = q(k,:);
    % Very naive kinematics: planar-ish cumulative rotations mapping to 3D helix style (placeholder!)
    pts = zeros(dof+1,3);
    R = eye(3);
    dir = [0;0;1];
    for j=1:dof
        % Alternate rotation axes for rough spatial spread
        switch mod(j,3)
            case 1, axisRot = [0;0;1];
            case 2, axisRot = [0;1;0];
            otherwise, axisRot = [1;0;0];
        end
        R = localAxisAngle(R, axisRot, qk(j));
        pts(j+1,:) = pts(j,:) + (R*dir*L(j))';
    end
    % Update plots
    for j=1:dof
        set(hLinks(j),'XData',pts(j:j+1,1),'YData',pts(j:j+1,2),'ZData',pts(j:j+1,3));
    end
    if opt.ShowTrail
        trailX = [trailX pts(end,1)]; %#ok<AGROW>
        trailY = [trailY pts(end,2)]; %#ok<AGROW>
        trailZ = [trailZ pts(end,3)]; %#ok<AGROW>
        if numel(trailX) > opt.TrailLength
            trailX = trailX(end-opt.TrailLength+1:end);
            trailY = trailY(end-opt.TrailLength+1:end);
            trailZ = trailZ(end-opt.TrailLength+1:end);
        end
        set(hTrail,'XData',trailX,'YData',trailY,'ZData',trailZ);
    end
    axis(ax,[-1 1 -1 1 0 1.2]);
    drawnow;
    pause(opt.DT);
end

end

function Rnew = localAxisAngle(R, axis, theta)
    axis = axis / max(1e-12, norm(axis));
    K = [    0     -axis(3)  axis(2);
          axis(3)     0     -axis(1);
         -axis(2)  axis(1)     0    ];
    Rrot = eye(3) + sin(theta)*K + (1-cos(theta))*(K*K);
    Rnew = R*Rrot;
end
