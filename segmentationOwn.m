%%********* Load and Prepare Data *********%%

% Set the paths for images and labels
dataSetDir = fullfile('C:\Users\LLR User\Downloads\CV\CW\data_for_moodle\data_for_moodle'); % Main data directory
imageDir = fullfile(dataSetDir, 'images_256'); % Directory containing images
labelDir = fullfile(dataSetDir, 'labels_256'); % Directory containing labels

% Load images by creating image datastore
imgDS = imageDatastore(imageDir);

% Ground truth
classNames = ["flower", "background"]; % Define class names
labelIDs = {1, [2; 3; 4]}; % Set corresponding class label IDs; assigns 1(flower) to "flower" and 2(leaves), 3(background) & 4(sky) to "background"
%labelIDs2 = [1, 3]; % Set corresponding class label IDs; assigns 1(flower) to "flower" and 3(background) to "background"

% Load pixels by creating pixel label datastore
pxlDS = pixelLabelDatastore(labelDir, classNames, labelIDs); 

% Create a map to store label filenames
labelMap = containers.Map(); 

% Populate the map with label base names
for i = 1:numel(pxlDS.Files) % Loop through each label filename
   [~, lblName, ~] = fileparts(pxlDS.Files{i}); % Extract only base (label) filename 
   labelMap(lblName) = pxlDS.Files{i}; % Store the full path of the file
end

% Initialize storage for matched images and labels
imageWithLabels = {}; % For images
existingLabels = {}; % For labels

% Find matching images and label names
for i = 1:numel(imgDS.Files) % Loop through each image filename
   [~, imgName, ~] = fileparts(imgDS.Files{i}); % Extract only base (image) filename
   if isKey(labelMap, imgName) % Check if the key exists in the map
       imageWithLabels{end+1} = imgDS.Files{i}; % Add matching image file path
       existingLabels{end+1} = labelMap(imgName); % Add corresponding label file path
   end
end

% Update the datastores
imgDS.Files = imageWithLabels; % Update with matched images
pxlDS = pixelLabelDatastore(existingLabels, classNames, labelIDs); % Create new datastore for matched labels

%%********* Analyze Dataset Statistics *********%%
% Get pixel count for each class
pxlCountTbl = countEachLabel(pxlDS); 

% Get class frequencies
frequency = pxlCountTbl.PixelCount/sum(pxlCountTbl.PixelCount);
classWeights = 1 ./frequency; % Calculate weights

% Display the pixel counts
figure;
bar(1:numel(classNames),frequency)
xticks(1:numel(classNames))  
xticklabels(pxlCountTbl.Name)
xtickangle(45)
ylabel('Frequency')
title('Pixel Count per Class')

%% Building Own Network for Segmentation of the Images

% Partition data into training, validation and test sets
[imgDS_Train, imgDS_Val, imgDS_Test, pxlDS_Train, pxlDS_Val, pxlDS_Test] = partitionData(imgDS, pxlDS, 0.6, 0.2);

% Define custom model architecture
imageSize = [256, 256, 3]; % Input image size
numClasses = numel(classNames); % Number of classes

% Build a custom architecture
layers = [
    % Level 1: First downsampling block
    imageInputLayer(imageSize) % Define input layer with sepcified image size
    convolution2dLayer(3, 32, 'Padding', 'same') % Convolutional layer with 32 filters of 3x3
    batchNormalizationLayer % Batch normalization to normalize activations
    reluLayer % ReLU activation for inroducing non-linerity
    maxPooling2dLayer(2, 'Stride', 2) % To downsample the feature map by a factor of 2

    % Level 2: Second downsampling block
    convolution2dLayer(3, 64, 'Padding', 'same') % Convolutional layer with 64 filters of 3x3
    batchNormalizationLayer % Batch normalization to normalize activations
    reluLayer % ReLU activation for inroducing non-linerity
    maxPooling2dLayer(2, 'Stride', 2) % To downsample the feature map by a factor of 2

    % Level 3: Third downsampling block
    convolution2dLayer(3, 128, 'Padding', 'same') % Convolutional layer with 128 filters of 3x3
    batchNormalizationLayer % Batch normalization to normalize activations
    reluLayer % ReLU activation for inroducing non-linerity
    maxPooling2dLayer(2, 'Stride', 2) % To downsample the feature map by a factor of 2

    % Level 4: First upsampling block
    transposedConv2dLayer(3, 128, 'Stride', 2, 'Cropping', 'same') % Transposed convolution to usample by a fator of 2
    batchNormalizationLayer % Batch normalization to normalize activations
    reluLayer % ReLU activation for inroducing non-linerity

    % Level 5: Second upsampling block
    transposedConv2dLayer(3, 64, 'Stride', 2, 'Cropping', 'same') % Transposed convolution to usample by a fator of 2
    batchNormalizationLayer % Batch normalization to normalize activations
    reluLayer % ReLU activation for inroducing non-linerity

    % Output block
    transposedConv2dLayer(3, 32, 'Stride', 2, 'Cropping', 'same') % Transposed convolution to usample by a fator of 2
    convolution2dLayer(1, numClasses, 'Padding', 'same') % Final convolutional layer
    softmaxLayer %To convert predictions to probabilities
    pixelClassificationLayer('Name', 'PixelClass',... % To classify each pixel
    'Classes', pxlCountTbl.Name, 'ClassWeights', classWeights); % Use class weights
    ];

lgraph = layerGraph(layers); % Convert layers into a layer graph for DAG network creation

% Data augmentation
augmenter = imageDataAugmenter('RandRotation', [-10, 10], 'RandXTranslation', [-10, 10], 'RandYTranslation', [-10, 10]);

% Create training and validation datastores
ds_imgTrain = pixelLabelImageDatastore(imgDS_Train, pxlDS_Train, 'DataAugmentation', augmenter); % Create augmented training datastore
ds_imgVal = pixelLabelImageDatastore(imgDS_Val, pxlDS_Val); % Create validation datastore

% Training Options
options = trainingOptions("adam",... % Use Adam optimizer
    ValidationData=ds_imgVal,... % Set validation data
    MaxEpochs=20,...  % Number of epochs
    MiniBatchSize=8,... % Batch size 
    Shuffle="every-epoch",... % Shuffle data for every epoch
    ValidationFrequency=10,... % Validate every 10 iterations
    VerboseFrequency=10,... % Print progress after every 10 iterations
    Plots="training-progress",... % Plot training progress
    OutputNetwork="best-validation-loss"); % Save best model

% Train the custom model
net = trainNetwork(ds_imgTrain, lgraph, options);

% Save the trained network to a .mat file
save('segmentownnet.mat', 'net')

% Evaluate the network on the test set
ds_imgTest = pixelLabelImageDatastore(imgDS_Test, pxlDS_Test); % Create test datastore
predictedSeg = semanticseg(ds_imgTest, net, "MiniBatchSize", 10); % Predict segmentation on test set

% Evaluate and display segmentation metrics
segMetrics = evaluateSemanticSegmentation(predictedSeg, pxlDS_Test); % Evaluate metrics
disp(segMetrics.DataSetMetrics); % Display dataset metrics
disp(segMetrics.ClassMetrics); % Display class metrics

% Confusion matrix
confMx = confusionchart(segMetrics.ConfusionMatrix.Variables,...
    classNames, Normalization="row-normalized"); % Display normalized confusion matrix
confMx.Title = "Normalized Confusion Matrix (%)"; % Add title to the confusion matrix

% Calculate mean IoU
meanIoU = segMetrics.ImageMetrics.MeanIoU;
fprintf('Mean IoU: %.2f\n', meanIoU); % Print the mean IoU

% Plot IoU Values
figure; 
histogram(meanIoU); % Plot histogram of IoU values
xlabel('IoU');
ylabel('Frequency');

% Test for a random image
numFilesTest = numel(imgDS_Test.Files);
idxRand = randi(numFilesTest); % Get a random index
randImgTest = readimage(imgDS_Test, idxRand); % Read the random test image
randPredictedSeg = semanticseg(randImgTest, net); % Get the predicted segmentation for the random image

% Define a colormap for overlay
cmap = [1 1 0; 1 0 0]; % Yellow for "flower" and red for "background"
overlay = labeloverlay(randImgTest, randPredictedSeg, 'Colormap', cmap, 'Transparency', 0.4); % Create the overlay

% Display random segmented image
figure;
subplot(1, 2, 1); % Create first subplot
imshow(randImgTest); % Show original image
title('Original Image');

subplot(1, 2, 2); % Create second subplot
imshow(overlay); % Show segmented image
title('Segmented Overlay');

% Function to data into training, validation and test sets
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds, pxds, trainRatio, valRatio)
    idx = randperm(numel(imds.Files)); % Shuffle indices
    numTrainSet = round(trainRatio * numel(imds.Files)); % Number of training samples
    numValSet = round(valRatio * numel(imds.Files)); % Number of validation samples
    
    idxTrain = idx(1:numTrainSet); % Training indices
    idxVal = idx(numTrainSet+1:numTrainSet+numValSet); % Validation indices
    idxTest = idx(numTrainSet+numValSet+1:end); % Test indices
    
    imdsTrain = subset(imds, idxTrain); % Training images
    pxdsTrain = subset(pxds, idxTrain); % Training labels
    
    imdsVal = subset(imds, idxVal); % Validation images
    pxdsVal = subset(pxds, idxVal); % Validation labels
    
    imdsTest = subset(imds, idxTest); % Test images
    pxdsTest = subset(pxds, idxTest); % Test labels
end
