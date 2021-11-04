%script for null_layer
  
    %% dataset
    %choose the dataset
    dataset='mnist';
    portion1=0.1; %XTrain
    portion2=0.5; %XValidation
    path2= ''; %Set up the path
    
    switch dataset
        case 'mnist'
            %load dataset
            XTrain=datastore(path2,'IncludeSubfolders',true,'LabelSource','foldernames');
            [XTrain,XValidation,XResto] = splitEachLabel(XTrain,portion1,portion2);
            XTest=datastore('','IncludeSubfolders',true,'LabelSource','foldernames');
            YTest=XTest.Labels;
            %resize for match with AlexNet
            XTrain=augmentedImageDatastore([224 224 1], XTrain);
            XValidation=augmentedImageDatastore([224 224 1], XValidation);
            XTest=augmentedImageDatastore([224 224 1], XTest);
        case 'cifar10'
            %load dataset
            XTrain=datastore(path2,'IncludeSubfolders',true,'LabelSource','foldernames');
            [XTrain,XValidation,XResto] = splitEachLabel(XTrain,portion1,portion2);
            XTest=datastore('','IncludeSubfolders',true,'LabelSource','foldernames');
            YTest=XTest.Labels;
            %resize for match with AlexNet
            XTrain=augmentedImageDatastore([224 224 1], XTrain);
            XValidation=augmentedImageDatastore([224 224 1], XValidation);
            XTest=augmentedImageDatastore([224 224 1], XTest);
        case 'flower'
            %load dataset
            XTrain=datastore(path2,'IncludeSubfolders',true,'LabelSource','foldernames');
            [XTrain,XValidation,XTest] = splitEachLabel(XTrain,portion1,portion2);
            YTest=XTest.Labels;
            XTrain=augmentedImageDatastore([263 320 3], XTrain);
            XValidation=augmentedImageDatastore([263 320 3], XValidation);
            XTest=augmentedImageDatastore([263 320 3], XTest);
        case 'fruit'
            %load dataset
            XTrain=datastore(path2,'IncludeSubfolders',true,'LabelSource','foldernames');
            [XTrain,XValidation,XTest] = splitEachLabel(XTrain,portion1,portion2);
            YTest=XTest.Labels;
    end
    
    numClasses = numel(categories(YTest));
    
    %% options
    maxEpochs = 10;
    miniBatchSize = 32;
    numObservations = numel(XTrain.Files);
    numIterationsPerEpoch = floor(numObservations / miniBatchSize);
    
    options = trainingOptions('sgdm', ...
        'MaxEpochs',maxEpochs, ...
        'ValidationData',XValidation, ...
        'ValidationFrequency',numIterationsPerEpoch, ...
        'MiniBatchSize',miniBatchSize,...
        'Verbose',false,...
        'InitialLearnRate',0.0001,...
        'Momentum',0.8,...
        'VerboseFrequency',miniBatchSize,...
        'Verbose',0,...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
 
    %%
    for s=5:5
        %options for activation function
        switch s
            case 1
                ativa=tanhLayer 
                ativa.Name='tanh';
            case 2
                ativa=leakyReluLayer 
                ativa.Name='leaky';
            case 3
                ativa=clippedReluLayer(5) 
                ativa.Name='clipped';
            case 4
                ativa=eluLayer
                ativa.Name='elu'; 
            case 5
                ativa=reluLayer 
                ativa.Name='relu';
        end
        
        
        %% null-layer
        for I=1:3
            I
            
            %list  of  inicializations
            inicializa={'glorot' 'he' 'narrow-normal'};
            ini=char(inicializa(I))
            inicializa2={'zeros'};
            ini2=char(inicializa2(1))
            
            %put the layers here
            layers = [];
                      
            
            switch dataset
                case 'mnist'
                    layers(1)=imageInputLayer([224 224 1],"Name","data");
                case 'cifar10'
                    layers(1)=imageInputLayer([224 224 3],"Name","data");
                case 'flower'
                    layers(1)=imageInputLayer([263 320 3],"Name","data");
                case 'fruit'
                    layers(1)=imageInputLayer([100 100 3],"Name","data");
            end
            
            % training network
            for r=1:5
                r
                [net,info] = trainNetwork(XTrain,layers,options);
                
                %Networks
                NET(r)=net;
                INFO(r)=info;
                YPred = classify(net, XTest);
                accuracy = sum(YTest == YPred)/numel(YTest);
                ACCURACY(r)=accuracy;
            end
            
            % save
            path=pwd; 
            mkdir(ativa.Name); 
            texto1=['\net_layer1_zero_' char(inicializa(I)) '_mnist'];
            texto2=['\info_layer1_zero_' char(inicializa(I)) '_mnist'];
            texto3=['\Accuracy_layer1_zero_' char(inicializa(I)) '_mnist'];
            
            save (strcat(path,"\",ativa.Name,texto1),'NET')
            save (strcat(path,"\",ativa.Name,texto2),'INFO')
            save (strcat(path,"\",ativa.Name,texto3),'ACCURACY')
            clearvars -except YTest dataset maxEpochs miniBatchSize numClasses numIterationsPerEpoch numObservations options XTest XTrain XValidation ativa path        
        end
        
        
        %% traditional
        for I=1:3
            I
            
            %list  of  inicializations
            inicializa={'glorot' 'he' 'narrow-normal'};
            ini=char(inicializa(I))
            
            %put the layers here
            layers = [];
            
            switch dataset
                case 'mnist'
                    layers(1)=imageInputLayer([224 224 1],"Name","data");
                case 'cifar10'
                    layers(1)=imageInputLayer([224 224 3],"Name","data");
                case 'flower'
                    layers(1)=imageInputLayer([263 320 3],"Name","data");
                case 'fruit'
                    layers(1)=imageInputLayer([100 100 3],"Name","data");
            end
            
            % training network
            for r=1:5
                r
                [net,info] = trainNetwork(XTrain,layers,options);
                
                %Networks
                NET(r)=net;
                INFO(r)=info;
                YPred = classify(net, XTest);
                accuracy = sum(YTest == YPred)/numel(YTest);
                ACCURACY(r)=accuracy;
            end
            
            % save
            texto1=['\net_' char(inicializa(I)) '_mnist'];
            texto2=['\info_' char(inicializa(I)) '_mnist'];
            texto3=['\Accuracy_' char(inicializa(I)) '_mnist'];
            
            save (strcat(path,"\",ativa.Name,texto1),'NET')
            save (strcat(path,"\",ativa.Name,texto2),'INFO')
            save (strcat(path,"\",ativa.Name,texto3),'ACCURACY')
            clearvars -except YTest dataset maxEpochs miniBatchSize numClasses numIterationsPerEpoch numObservations options XTest XTrain XValidation ativa path
        end
        
    end

