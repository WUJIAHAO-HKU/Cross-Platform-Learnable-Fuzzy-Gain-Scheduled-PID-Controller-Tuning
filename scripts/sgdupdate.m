function newParams = sgdupdate(params, grads, lr)
%SGDUPDATE  Minimal SGD parameter update supporting cell arrays or struct arrays.
%   newParams = sgdupdate(params, grads, lr)
%   - params: dlnetwork Weights cell array or struct with numeric fields
%   - grads : same structure/shape as params
%   - lr    : scalar learning rate
%
% If a gradient entry is empty or non-numeric, it's skipped.
%
% NOTE: This is intentionally simple; later you can replace with Adam.

if nargin < 3, error('sgdupdate requires params, grads, lr'); end
if isempty(params), newParams = params; return; end

if iscell(params)
    newParams = params;
    for i=1:numel(params)
        g = grads{i}; p = params{i};
        if isnumeric(p) && isnumeric(g) && ~isempty(g)
            newParams{i} = p - lr * g;
        end
    end
elseif isstruct(params)
    newParams = params;
    fields = fieldnames(params);
    for i=1:numel(fields)
        f = fields{i};
        p = params.(f); g = grads.(f);
        if isnumeric(p) && isnumeric(g) && ~isempty(g)
            newParams.(f) = p - lr * g;
        end
    end
else
    error('Unsupported params type: %s', class(params));
end
end