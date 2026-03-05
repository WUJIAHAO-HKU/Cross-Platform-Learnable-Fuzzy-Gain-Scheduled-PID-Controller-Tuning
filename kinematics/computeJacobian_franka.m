function J = computeJacobian_franka(q)
% COMPUTEJACOBIAN_FRANKA  Approximate geometric Jacobian (6x7) for Franka Panda.
%   J = computeJacobian_franka(q)
% Returns spatial Jacobian mapping joint velocities to end-effector twist:
%   [vx; vy; vz; wx; wy; wz]
%
% Uses the same approximate kinematic chain as forwardKinematics_franka.
%
% Method:
%   - Accumulate transforms for each joint origin
%   - Assume all joints are revolute (z-axis of each joint frame)
%   - For joint i: angular part = z_i; linear part = z_i x (p_end - p_i)
%
% NOTE: For real precision tasks, replace with validated kinematics.

q = q(:);
if numel(q)~=7; error('Expected 7x1 joint vector'); end

alpha = [0, -pi/2,  pi/2,  pi/2, -pi/2,  pi/2,  pi/2];
a     = [0,     0,     0, 0.0825, -0.0825, 0, 0.088];
d     = [0.333, 0, 0.316, 0, 0.384, 0, 0.107];

% Precompute transforms
T = eye(4);
origins = zeros(3,8);
Zaxes   = zeros(3,8);
origins(:,1) = [0;0;0];
Zaxes(:,1) = [0;0;1];
Ts = cell(1,8); Ts{1} = T;
for i=1:7
    A = dhTransform(alpha(i), a(i), d(i), q(i));
    T = T * A;
    Ts{i+1} = T;
    origins(:,i+1) = T(1:3,4);
    Zaxes(:,i+1)   = T(1:3,3); % z axis of current frame
end
p_end = origins(:,end);

J = zeros(6,7);
for i=1:7
    p_i = origins(:,i);
    z_i = Zaxes(:,i);
    J(1:3,i) = cross(z_i, p_end - p_i); % linear velocity component
    J(4:6,i) = z_i;                     % angular velocity component
end

end

function A = dhTransform(alpha, a, d, theta)
    ca = cos(alpha); sa = sin(alpha);
    ct = cos(theta); st = sin(theta);
    A = [ ct, -st, 0, a;
          st*ca, ct*ca, -sa, -d*sa;
          st*sa, ct*sa,  ca,  d*ca;
          0, 0, 0, 1];
end
