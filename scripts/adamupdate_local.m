function state = adamupdate_local(state, grads, lr, beta1, beta2, eps)
%ADAMUPDATE_LOCAL  Local Adam optimizer (renamed to avoid toolbox name clash).
% Usage:
%   state = adamupdate_local(state, grads, lr)
%   state = adamupdate_local(state, grads, lr, beta1, beta2, eps)
%
% state: struct with at least field .params (cell or struct of numeric arrays)
% grads: same container type as state.params
% lr   : learning rate (scalar)
% beta1, beta2 (optional): momentum coefficients
% eps  : numerical stability constant
%
% Returns updated state with fields:
%   .params (updated parameters)
%   .m (first moments)
%   .v (second moments)
%   .t (time step)

if nargin < 6, eps = 1e-8; end
if nargin < 5, beta2 = 0.999; end
if nargin < 4, beta1 = 0.9; end

if ~isfield(state,'t') || isempty(state.t), state.t = 0; end
state.t = state.t + 1;

% Allocate moments if first time
if ~isfield(state,'m') || isempty(state.m)
    state.m = zeroLike(grads);
end
if ~isfield(state,'v') || isempty(state.v)
    state.v = zeroLike(grads);
end

[state.m, state.v] = updateMoments(state.m, state.v, grads, beta1, beta2);

mhat = scaleContainer(state.m, 1/(1 - beta1^state.t));
vhat = scaleContainer(state.v, 1/(1 - beta2^state.t));

state.params = adamStep(state.params, mhat, vhat, grads, lr, eps);
end

function z = zeroLike(x)
if iscell(x)
    z = cell(size(x));
    for i=1:numel(x)
        if isnumeric(x{i})
            z{i} = zeros(size(x{i}), 'like', x{i});
        else
            z{i} = [];
        end
    end
elseif isstruct(x)
    f = fieldnames(x); z = struct();
    for i=1:numel(f)
        xi = x.(f{i});
        if isnumeric(xi)
            z.(f{i}) = zeros(size(xi), 'like', xi);
        else
            z.(f{i}) = [];
        end
    end
else
    error('Unsupported grads container type');
end
end

function [mNew,vNew] = updateMoments(m,v,g,b1,b2)
if iscell(g)
    mNew = m; vNew = v;
    for i=1:numel(g)
        if isnumeric(g{i})
            mNew{i} = b1*m{i} + (1-b1)*g{i};
            vNew{i} = b2*v{i} + (1-b2)*(g{i}.^2);
        end
    end
elseif isstruct(g)
    mNew = m; vNew = v; f = fieldnames(g);
    for i=1:numel(f)
        gi = g.(f{i});
        if isnumeric(gi)
            mNew.(f{i}) = b1*m.(f{i}) + (1-b1)*gi;
            vNew.(f{i}) = b2*v.(f{i}) + (1-b2)*(gi.^2);
        end
    end
else
    error('Unsupported grad container');
end
end

function y = scaleContainer(x, s)
if iscell(x)
    y = x; for i=1:numel(x), if isnumeric(x{i}), y{i} = x{i}*s; end; end
elseif isstruct(x)
    y = x; f=fieldnames(x); for i=1:numel(f), if isnumeric(x.(f{i})), y.(f{i}) = x.(f{i})*s; end; end
else
    error('Unsupported type');
end
end

function paramsNew = adamStep(params, m, v, g, lr, eps)
if iscell(params)
    paramsNew = params;
    for i=1:numel(params)
        if isnumeric(params{i}) && isnumeric(m{i}) && isnumeric(v{i})
            paramsNew{i} = params{i} - lr * m{i} ./ (sqrt(v{i}) + eps);
        end
    end
elseif isstruct(params)
    paramsNew = params; f = fieldnames(params);
    for i=1:numel(f)
        if isnumeric(params.(f{i}))
            paramsNew.(f{i}) = params.(f{i}) - lr * m.(f{i}) ./ (sqrt(v.(f{i})) + eps);
        end
    end
else
    error('Unsupported params container');
end
end
