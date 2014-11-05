classdef cdbn
    % Convolution Deep Belif Network (CDBN)
    % -------------------------------------------
    % This implementation is based on 'Unsupervised Learning of Hierarchical Representations
    % with Convolutional Deep Belief Networks' by Honglak Lee. 
    % -------------------------------------------
    % By shaoguangcheng. From Xi'an, China.
    % Email : chengshaoguang1291@126.com
    
    properties
        className = 'cdbn';
        
        nLayer;        % total number of layers in CDBN
        model;         % all layer paramters and structures
        output;        % output of cdbn
        outputFolder;
        isSaveModel = true;
    end
    
    methods
        function self = cdbn(netStructure)
            % ---------------------------
            % create cdbn
            % ---------------------------
           self.nLayer = numel(netStructure);
           self.model = cell(1, self.nLayer);
           
           % initialize all layers in cdbn
           self.model{1} = crbm(netStructure(1));
           if self.model{1}.nFeatureMapVis ~= 1
              error('First layer in cdbn must only have single feature map'); 
           end
           
           if self.nLayer > 1
               for i = 2 : self.nLayer
                  self.model{i} = crbm(netStructure(i)); 
               end
           end
           
            self.outputFolder = sprintf('%s%s%s','..', filesep, 'log');
            if ~exist(self.outputFolder, 'dir')
                mkdir(self.outputFolder);
            end
           
        end
        
        function self = train(self, data)
            % ----------------------
            % train cdbn model
            % ----------------------
             self.model{1} = self.model{1}.train(data);
             self.model{1} = self.model{1}.crbmFeedForward(data);            
            if self.nLayer > 1
                for i = 2 : self.nLayer
                    self.model{i} = self.model{i}.train(self.model{i-1}.outputPooling);
                    self.model{i} = self.model{i}.crbmFeedForward(self.model{i-1}.outputPooling);
                end
            end
            
            if self.isSaveModel
                self.save;
            end
        end
        
        function self = cdbnFeedForward(self, data)
           % ---------------------------
           % cal output of cdbn given input data
           % ---------------------------
           m = self.model{1}.crbmFeedForward(data);
           self.output = m.outputPooling;
           if self.nLayer > 1
               for i = 2 : self.nLayer
                   m = self.model{i}.crbmFeedForward(m.outputPooling);
               end
               self.output = m.outputPooling;
           end
        end
        
        function feature = getUnpoolingFeature(self, data, layer)
            % -----------------------------
            % extracting unpooling features of layer for data
            % -----------------------------
            if layer == 1
               [m, feature] = self.model{1}.inference(data); 
            end
            
            if layer > 1
                for i = 1 : layer - 1
  %                  data = self.model{i}.trimDataForPooling(data);
                    m = self.model{i}.crbmFeedForward(data);
                    data = m.outputPooling;
                end
                 data = self.model{layer}.trimDataForPooling(data);
                [m, feature] = self.model{layer}.inference(data);
            end
        end
        
        function [] = save(self)
            % ----------------------------
            % save cdbn
            % ----------------------------
            fmt = sprintf('%s%s%s_cdbn_model.mat',self.outputFolder, ...
                filesep, datestr(clock, 'yyyy_mm_dd_HH_MM_SS')); 
            
            cdbnModel = cell(1, self.nLayer);
            for i  = 1 : self.nLayer
               cdbnModel{i}.W = self.model{i}.W;
               cdbnModel{i}.visBias = self.model{i}.visBias;
               cdbnModel{i}.hidBias = self.model{i}.hidBias;
            end
            
            save(fmt, 'cdbnModel');
        end
        
    end
    
end

