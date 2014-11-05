function [ y ] = softmax(x, dim)
% y = softmax(x, [dim])
% Compute softmax maxtrix along dimision dim. Default dim is 1.

if notDefined('dim')
    dim = 1;
end

maxX = max(x, [], dim);
tmp = exp(bsxfun(@minus, x, maxX));
y = bsxfun(@rdivide, tmp, sum(tmp, sum));

