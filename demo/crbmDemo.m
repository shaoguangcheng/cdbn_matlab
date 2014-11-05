% test crbm
% -------------------------------------------
% This implementation is based on 'Unsupervised Learning of Hierarchical Representations
% with Convolutional Deep Belief Networks' by Honglak Lee. 
% -------------------------------------------
% By shaoguangcheng. From Xi'an, China
% Email : chengshaoguang1291@126.com
if ispc
    addpath('..\', '..\data', '..\util');
    dataPath = '..\data\mnistSmall';
end

if isunix
    addpath('../', '../data', '../util');
    dataPath = '../data/mnistSmall';
end


load(dataPath);
trainData = reshape(trainData', [28,28,10000]);
trainData = trainData(:,:,1:20);
dataSize = [28,28];

arch = struct('dataSize', dataSize, ...
        'nFeatureMapVis', 1, ...
		'nFeatureMapHid', 9, ...
        'kernelSize', [7 7], ...
        'poolingScale', [2 2], ...
        'inputType', 'binary');

arch.opt = {'nEpoch', 500, ...
			 'learningRate', .03, ...
			 'displayInterval',10, ...
			 'sparsity', .003, ...
			 'lambda1', 5, ...
             'isUseGPU', 0, ...
             'isSaveModel', 1};
         
m = crbm(arch);
m = m.train(trainData);

[m, hidSample] = m.inference(trainData(:,:,7));
m  = m.crbmFeedForward(trainData);

figure(1);
[r,c,n]=size(m.W);
W = reshape(m.W,r*c,n);
visWeights(W,1);
title('Convolution Kernel');

figure(2);
imagesc(trainData(:,:,7)); colormap gray; axis image; axis off
title(sprintf('Original Sample'));

figure(3);
[r,c,n]=size(hidSample);
hidSample = reshape(hidSample,r*c,n);
visWeights(hidSample);
title('Feature Maps')

figure(4);
[r,c,n] = size(m.outputPooling(:,:,7,:));
visWeights(reshape(m.outputPooling(:,:,7,:),r*c,n)); colormap gray
title(sprintf('Pooling output'))
drawnow
