function report = verify_fk_with_urdf(numTests, tol_pos, tol_rot)
% VERIFY_FK_WITH_URDF  Optional cross-check of hand-written FK vs URDF (if toolbox available).
%   report = verify_fk_with_urdf(numTests, tol_pos, tol_rot)
%
% Inputs:
%   numTests : number of random joint configurations (default 20)
%   tol_pos  : position tolerance (m) (default 2e-3)
%   tol_rot  : rotation tolerance (angle radians) (default 5e-3)
%
% Output struct report:
%   .available      : toolbox & model available flag
%   .numTests       : number performed
%   .max_pos_error  : maximum position norm difference
%   .max_rot_error  : maximum rotation angle difference
%   .pass           : boolean all within tolerance
%   .details        : per-test errors
%
% Requirements:
%   - Robotics System Toolbox (optional)
%   - panda_arm.urdf accessible on path (user supplied)
%
% If not available, returns report.available = false.

if nargin < 1 || isempty(numTests); numTests = 20; end
if nargin < 2 || isempty(tol_pos); tol_pos = 2e-3; end
if nargin < 3 || isempty(tol_rot); tol_rot = 5e-3; end

report = struct('available',false,'numTests',0,'max_pos_error',NaN,'max_rot_error',NaN,'pass',false,'details',[]);

hasRST = licenseCheck('Robotics_System_Toolbox');
if ~hasRST
    fprintf('[verify_fk_with_urdf] Robotics System Toolbox not available. Skipping.\n');
    return;
end

try
    robot = importrobot('panda_arm.urdf'); %#ok<UNRCH>
catch ME
    fprintf('[verify_fk_with_urdf] URDF import failed: %s\n', ME.message);
    return;
end

report.available = true;
report.numTests = numTests;

errors = zeros(numTests,2);
for k=1:numTests
    % Joint limits (approx)
    q = [
        uniform(-2.9, 2.9);
        uniform(-1.76, 1.76);
        uniform(-2.9, 2.9);
        uniform(-3.07, -0.07);
        uniform(-2.9, 2.9);
        uniform(-0.01, 3.75);
        uniform(-2.9, 2.9)];
    [T_fk, ~, ~] = forwardKinematics_franka(q);
    % URDF forward
    % base: base_link, ee: panda_link8 or tool0 depending on URDF variant
    try
        T_urdf = getTransform(robot,q','panda_link8');
    catch
        try
            T_urdf = getTransform(robot,q','tool0');
        catch
            warning('Could not resolve end-effector link');
            continue;
        end
    end
    p_fk = T_fk(1:3,4); p_urdf = T_urdf(1:3,4);
    R_fk = T_fk(1:3,1:3); R_urdf = T_urdf(1:3,1:3);
    pos_err = norm(p_fk - p_urdf);
    dR = R_fk' * R_urdf; ang_err = rotationAngle(dR);
    errors(k,:) = [pos_err, ang_err];
end

report.max_pos_error = max(errors(:,1));
report.max_rot_error = max(errors(:,2));
report.pass = report.max_pos_error <= tol_pos && report.max_rot_error <= tol_rot;
report.details = errors;

fprintf('[verify_fk_with_urdf] Max pos err=%.4g m, max rot err=%.4g rad, pass=%d\n', ...
    report.max_pos_error, report.max_rot_error, report.pass);

end

function r = uniform(a,b)
    r = a + (b-a)*rand();
end

function ang = rotationAngle(R)
    tr = trace(R);
    ang = acos( min(1,max(-1,(tr-1)/2)) );
end

function ok = licenseCheck(id)
    try
        v = ver; names = {v.Name};
        ok = any(contains(lower(names),'robotics')); %#ok<NASGU>
    catch
        ok = false;
    end
end
