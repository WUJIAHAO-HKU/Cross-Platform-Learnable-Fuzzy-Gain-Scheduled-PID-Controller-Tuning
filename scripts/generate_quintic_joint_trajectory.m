function traj = generate_quintic_joint_trajectory(q0, qT, T, dt, v0, vT, a0, aT)
% GENERATE_QUINTIC_JOINT_TRAJECTORY Create multi-joint 5th-order polynomial trajectories.
%   traj = generate_quintic_joint_trajectory(q0, qT, T, dt)
%   traj = generate_quintic_joint_trajectory(q0, qT, T, dt, v0, vT, a0, aT)
%
% Inputs:
%   q0  : 1×n or n×1 initial joint positions
%   qT  : 1×n target joint positions
%   T   : total motion duration (seconds)
%   dt  : timestep for sampling (seconds)
%   v0, vT (optional) : initial/final velocities (default 0)
%   a0, aT (optional) : initial/final accelerations (default 0)
%
% Output struct traj:
%   .t      : time vector (column)
%   .q      : N×n joint positions
%   .qd     : N×n joint velocities
%   .qdd    : N×n joint accelerations
%   .coeffs : 6×n polynomial coefficients (a0..a5 per column)
%
% Quintic form per joint:
%   q(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
%
% Constraints at t=0 and t=T: position, velocity, acceleration
%
% Example:
%   q0 = zeros(1,7); qT = [0.5,-0.3,0.2,0,0,0,0];
%   traj = generate_quintic_joint_trajectory(q0,qT,5,0.001);
%   plot(traj.t, traj.q); legend('q1','q2','q3','q4','q5','q6','q7')
%
% Author: Auto-generated template

if nargin < 5 || isempty(v0); v0 = 0; end
if nargin < 6 || isempty(vT); vT = 0; end
if nargin < 7 || isempty(a0); a0 = 0; end
if nargin < 8 || isempty(aT); aT = 0; end

q0 = q0(:)';
qT = qT(:)';
nJ = numel(q0);
if numel(qT) ~= nJ
    error('q0 and qT must have same length.');
end

% Allow scalar velocities/accelerations broadcast
if isscalar(v0); v0 = repmat(v0,1,nJ); else; v0 = v0(:)'; end
if isscalar(vT); vT = repmat(vT,1,nJ); else; vT = vT(:)'; end
if isscalar(a0); a0 = repmat(a0,1,nJ); else; a0 = a0(:)'; end
if isscalar(aT); aT = repmat(aT,1,nJ); else; aT = aT(:)'; end

% Time vector
N = floor(T/dt)+1;
t = linspace(0,T,N)';

% Preallocate
q   = zeros(N,nJ);
qd  = zeros(N,nJ);
qdd = zeros(N,nJ);
coeffs = zeros(6,nJ);

for j=1:nJ
    % Boundary conditions
    p0 = q0(j); v0j = v0(j); a0j = a0(j);
    pT = qT(j); vTj = vT(j); aTj = aT(j);

    % Solve coefficients: a0..a5
    % Known:
    a0c = p0;
    a1c = v0j;
    a2c = a0j/2;
    % Remaining system for a3,a4,a5 using conditions at t=T
    TT = T; TT2 = TT^2; TT3 = TT^3; TT4 = TT^4; TT5 = TT^5;
    M = [ TT3    TT4     TT5;
          3*TT2  4*TT3   5*TT4;
          6*TT   12*TT2  20*TT3];
    b = [ pT - (a0c + a1c*TT + a2c*TT2);
          vTj - (a1c + 2*a2c*TT);
          aTj - (2*a2c) ];
    x = M\b; % a3,a4,a5
    a3c = x(1); a4c = x(2); a5c = x(3);
    coeffs(:,j) = [a0c;a1c;a2c;a3c;a4c;a5c];

    % Sample
    tt = t;
    q(:,j)   = a0c + a1c*tt + a2c*tt.^2 + a3c*tt.^3 + a4c*tt.^4 + a5c*tt.^5;
    qd(:,j)  = a1c + 2*a2c*tt + 3*a3c*tt.^2 + 4*a4c*tt.^3 + 5*a5c*tt.^4;
    qdd(:,j) = 2*a2c + 6*a3c*tt + 12*a4c*tt.^2 + 20*a5c*tt.^3;
end

traj = struct('t',t,'q',q,'qd',qd,'qdd',qdd,'coeffs',coeffs,'T',T,'dt',dt);
end
