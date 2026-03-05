function state = adamupdate(state, grads, lr, beta1, beta2, eps)
% Wrapper maintained only to avoid name clash errors when legacy calls exist.
% Redirects to adamupdate_local (preferred). Signatures kept identical.
if nargin < 6, eps = 1e-8; end
if nargin < 5, beta2 = 0.999; end
if nargin < 4, beta1 = 0.9; end
state = adamupdate_local(state, grads, lr, beta1, beta2, eps);
end