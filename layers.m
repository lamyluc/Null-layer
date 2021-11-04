%% CNN3

%null-layer
layers =[imageInputLayer([32 32 3],"Name","imageinput")
                convolution2dLayer([7 7],16,"Name","c1",'Padding',[0 0 0 0],'WeightsInitializer',ini2,'BiasInitializer','ones')
                averagePooling2dLayer([2 2],"Name","s2","Stride",[2 2],'Padding',[0 0 0 0])
                ativa
                convolution2dLayer([7 7],32,"Name","c3",'Padding',[0 0 0 0],'WeightsInitializer',ini)
                averagePooling2dLayer([2 2],"Name","s4","Padding","same","Stride",[2 2])
                convolution2dLayer([7 7],128,"Name","c5","Padding","same",'WeightsInitializer',ini)
                fullyConnectedLayer(numClasses,"Name","f6",'WeightsInitializer',ini)
                ativa
                softmaxLayer("Name","softmax")
                classificationLayer("Name","classoutput")];

%traditional
 layers =[imageInputLayer([32 32 3],"Name","imageinput")
                convolution2dLayer([7 7],16,"Name","c1",'Padding',[0 0 0 0],'WeightsInitializer',ini,'BiasInitializer','ones')
                averagePooling2dLayer([2 2],"Name","s2","Stride",[2 2],'Padding',[0 0 0 0])
                ativa
                convolution2dLayer([7 7],32,"Name","c3",'Padding',[0 0 0 0],'WeightsInitializer',ini)
                averagePooling2dLayer([2 2],"Name","s4","Padding","same","Stride",[2 2])
                convolution2dLayer([7 7],128,"Name","c5","Padding","same",'WeightsInitializer',ini)
                fullyConnectedLayer(numClasses,"Name","f6",'WeightsInitializer',ini)
                ativa
                softmaxLayer("Name","softmax")
                classificationLayer("Name","classoutput")];


%% AlexNet

%null-layer
 layers = [
                imageInputLayer([227 227 3],"Name","data")
                convolution2dLayer([11 11],96,"Name","conv1","Stride",[4 4],'WeightsInitializer',ini2,'BiasInitializer','ones')
                ativa
                crossChannelNormalizationLayer(5,"Name","norm1","K",1)
                maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
                groupedConvolution2dLayer([5 5],128,2,"Name","conv2","Padding",[2 2 2 2],'WeightsInitializer',ini)
                ativa
                crossChannelNormalizationLayer(5,"Name","norm2","K",1)
                maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
                convolution2dLayer([3 3],384,"Name","conv3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                groupedConvolution2dLayer([3 3],192,2,"Name","conv4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                groupedConvolution2dLayer([3 3],128,2,"Name","conv5","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
                fullyConnectedLayer(4096,"Name","fc6",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop6")
                fullyConnectedLayer(4096,"Name","fc7",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop7")
                fullyConnectedLayer(1000,"Name","fc8",'WeightsInitializer',ini)
                softmaxLayer("Name","prob")
                classificationLayer("Name","output")];
            
 %traditional
 layers = [
                imageInputLayer([227 227 3],"Name","data")
                convolution2dLayer([11 11],96,"Name","conv1","Stride",[4 4],'WeightsInitializer',ini)
                ativa
                crossChannelNormalizationLayer(5,"Name","norm1","K",1)
                maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
                groupedConvolution2dLayer([5 5],128,2,"Name","conv2","Padding",[2 2 2 2],'WeightsInitializer',ini)
                ativa
                crossChannelNormalizationLayer(5,"Name","norm2","K",1)
                maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
                convolution2dLayer([3 3],384,"Name","conv3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                groupedConvolution2dLayer([3 3],192,2,"Name","conv4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                groupedConvolution2dLayer([3 3],128,2,"Name","conv5","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
                fullyConnectedLayer(4096,"Name","fc6",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop6")
                fullyConnectedLayer(4096,"Name","fc7",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop7")
                fullyConnectedLayer(1000,"Name","fc8",'WeightsInitializer',ini)
                softmaxLayer("Name","prob")
                classificationLayer("Name","output")];

%% VGG19

%null-layer
layers = [
                imageInputLayer([224 224 3],"Name","input")
                convolution2dLayer([3 3],64,"Name","conv1_1","Padding",[1 1 1 1],'WeightsInitializer',ini2,'BiasInitializer','ones')
                ativa
                convolution2dLayer([3 3],64,"Name","conv1_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
                convolution2dLayer([3 3],128,"Name","conv2_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],128,"Name","conv2_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
                convolution2dLayer([3 3],256,"Name","conv3_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],256,"Name","conv3_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],256,"Name","conv3_3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],256,"Name","conv3_4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
                convolution2dLayer([3 3],512,"Name","conv4_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv4_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv4_3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv4_4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
                convolution2dLayer([3 3],512,"Name","conv5_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv5_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv5_3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv5_4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
                fullyConnectedLayer(4096,"Name","fc6",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop6")
                fullyConnectedLayer(4096,"Name","fc7",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop7")
                fullyConnectedLayer(numClasses,"Name","fc8",'WeightsInitializer',ini)
                softmaxLayer("Name","prob")
                classificationLayer("Name","output")];

%traditional
layers = [
                imageInputLayer([224 224 3],"Name","input")
                convolution2dLayer([3 3],64,"Name","conv1_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],64,"Name","conv1_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
                convolution2dLayer([3 3],128,"Name","conv2_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],128,"Name","conv2_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
                convolution2dLayer([3 3],256,"Name","conv3_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],256,"Name","conv3_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],256,"Name","conv3_3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],256,"Name","conv3_4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
                convolution2dLayer([3 3],512,"Name","conv4_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv4_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv4_3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv4_4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
                convolution2dLayer([3 3],512,"Name","conv5_1","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv5_2","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv5_3","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                convolution2dLayer([3 3],512,"Name","conv5_4","Padding",[1 1 1 1],'WeightsInitializer',ini)
                ativa
                maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
                fullyConnectedLayer(4096,"Name","fc6",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop6")
                fullyConnectedLayer(4096,"Name","fc7",'WeightsInitializer',ini)
                ativa
                dropoutLayer(0.5,"Name","drop7")
                fullyConnectedLayer(numClasses,"Name","fc8",'WeightsInitializer',ini)
                softmaxLayer("Name","prob")
                classificationLayer("Name","output")];