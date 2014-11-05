% test cdbn
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
trainData = trainData(:,:,1:100);
dataSize1 = [28,28];
dataSize2 = [10,10];

arch1 = struct('dataSize', dataSize1, ...
        'nFeatureMapVis', 1, ...
		'nFeatureMapHid', 9, ...
        'kernelSize', [7 7], ...
        'poolingScale', [2 2], ...
        'inputType', 'binary');

arch1.opt = {'nEpoch', 10, ...
			 'learningRate', .05, ...
			 'displayInterval',2, ...
			 'sparsity', .02, ...
			 'lambda1', 5, ...
             'isUseGPU', 0};
         
arch2 = struct('dataSize', dataSize2, ...
        'nFeatureMapVis', 9, ...
		'nFeatureMapHid', 20, ...
        'kernelSize', [3 3], ...
        'poolingScale', [2 2], ...
        'inputType', 'binary');

arch2.opt = {'nEpoch', 10, ...
			 'learningRate', .05, ...
			 'displayInterval',2, ...
			 'sparsity', .02, ...
			 'lambda1', 5, ...
             'isUseGPU', 0};
m = cdbn([arch1, arch2]);
m = m.train(trainData);

feature1 = m.getUnpoolingFeature(trainData,1);
feature2 = m.getUnpoolingFeature(trainData,2);

m = m.cdbnFeedForward(trainData);

figure(1);
[r,c,n]=size(m.model{1}.W);
W = reshape(m.model{1}.W,r*c,n);
visWeights(W,1);
title('Convolution Kernel of First Layer');

figure(2);
[r,c,n]=size(m.model{2}.W);
W = reshape(m.model{2}.W,r*c,n);
visWeights(W,1);
title('Convolution Kernel of Second Layer');

figure(3);
[r,c,n]=size(feature1);
feature1 = reshape(feature1,r*c,n);
visWeights(feature1);
title('Feature Maps of First Layer')

figure(4);
[r,c,n]=size(feature2);
feature2 = reshape(feature2,r*c,n);
visWeights(feature2);
title('Feature Maps of second Layer')

figure(5);
[r,c,n] = size(m.output(:,:,1,:));
visWeights(reshape(m.output(:,:,1,:),r*c,n)); colormap gray
title(sprintf('Pooling output'))
drawnow