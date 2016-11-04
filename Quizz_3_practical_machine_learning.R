## Q1. 

# The the cell segmentation data loaded from the AppliedPredictiveModeling package:

library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

# The data subseted to a training set and testing set based on the Case variable in the data set.

inTrain <- createDataPartition(y=segmentationOriginal$Case, p=0.7, list = FALSE)
training <- segmentationOriginal[inTrain, ]
testing <- segmentationOriginal[-inTrain, ]
dim(training); dim(testing)

# The seed set to 125 and a CART model fit with the rpart method using all predictor variables and default caret settings.

set.seed(125)
modelFit <- train(Case ~ ., method = "rpart", data = training)

# if there is an error msg like: "Error in requireNamespaceQuietStop("e1071") : package e1071 is required", do the following and run the codes again:
# install.packages('e1071', dependencies=TRUE)

print(modelFit$finalModel)

# The final model prediction for cases with the following variable values:
# a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2
# b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100
# c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100
# d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2

suppressMessages(library(rattle))
library(rpart.plot)
fancyRpartPlot(modelFit$finalModel)

# Based on the decision tree, the model prediction is PS > WS > PS > Not possible to predict


## Q2.
# The bias is larger and variance smaller. Under leave one out cross-validation, k is equal to sample.

## Q3.

# Loadded dataset

library(pgmm)
data("olive")
olive = olive[,-1]

summary(olive)

# Fit a classification tree where Area is the outcome variable
inTrain <- createDataPartition(y=olive$Area, p=0.5, list = FALSE)
training <- olive[inTrain, ]
testing <- olive[-inTrain, ]
dim(training)
dim(testing)

fitModel2 <- train(Area ~ ., method = "rpart", data = training)

# Predict the value of area for the following data frame using the tree command with all defaults
# newdata = as.data.frame(t(colMeans(olive)))

predict(fitModel2, newdata = as.data.frame(t(colMeans(olive))))

# Q4.

# Load the dataset

library(ElemStatLearn)
data("SAheart")
set.seed(8484)
train <- sample(1:dim(SAheart), size = dim(SAheart)[1]/2, replace = F)
trainSA <- SAheart[train,]
testSA <- SAheart[-train,]

# Set the seed to 13234 and fit a logistic regression model (method="glm", be sure to specify family="binomial") with Coronary Heart Disease (chd) as the outcome and age at onset, current alcohol consumption, obesity levels, cumulative tabacco, type-A behavior, and low density lipoprotein cholesterol as predictors. 
set.seed(13234)
modelFitReg <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, data = trainSA, method = "glm", family = "binomial")

# Calculate the misclassification rate for your model using this function and a prediction on the "response" scale: 
# missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass <- function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass(trainSA$chd, predict(modelFitReg, trainSA))

missClass(testSA$chd, predict(modelFitReg, testSA))


## Q5.

# Load the dataser
library(ElemStatLearn)
data("vowel.train")
data("vowel.test")

# Set the variable y to be a factor variable in both the training and test set.
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

# set the seed to 33833.
set.seed(33833)

# Fit a random forest predictor relating the factor variable y to the remaining variables.
library(randomForest)
ModelFitRF <- randomForest(y ~ ., data = vowel.train)

# Calculate the variable importance using the varImp function in the caret package
library(caret)
order(varImp(ModelFitRF), decreasing = TRUE)


