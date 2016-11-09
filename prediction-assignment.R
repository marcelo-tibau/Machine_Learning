# Prediction Assignment - Practical Machine Learning
## The goal of your project is to predict the manner in which they did the exercise.
## This is the "classe" variable in the training set. 
## You may use any of the other variables to predict with.
## You should create a report describing how you built your model, how you used cross validation,
## what you think the expected out of sample error is, and why you made the choices you did.
## You will also use your prediction model to predict 20 different test cases.

# Codes to load the libraries to be used:
library("caret")
library("gbm")
library("rpart")
library("rpart.plot")
library("RColorBrewer")
library("rattle")
library("randomForest")

# Codes to download and read the train and test sets:
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", dest="pml-training.csv", mode="wb")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", dest="pml-testing.csv", mode="wb")

dataTrain <- read.csv("pml-training.csv")
dataTest <- read.csv("pml-testing.csv")
head(dataTrain)
head(dataTest)

# As we can see above, there's some NA data in both datasets. I intend to proceed a cleaning of the dataset, but first I will partion the training set into two and check possibles the NearZeroVariance Variables.  

# Codes to partioning the training dataset into two: 70% for training and 30% for testing.

set.seed(13563)
inTrain <- createDataPartition(y=dataTrain$classe, p=0.7, list=FALSE)
training <- dataTrain[inTrain, ]
testing <- dataTrain[-inTrain, ]
dim(training)
dim(testing)

# Code to cheack possibles NearZeroVariance Variables.
NZV_check <- nearZeroVar(training, saveMetrics = TRUE)
NZV_check

# Codes to reset both training and testing sets without NZV:
training <- training[, NZV_check$nzv==FALSE]
dim(training)

NZV_check2 <- nearZeroVar(testing, saveMetrics = TRUE)
testing <- testing[, NZV_check2$nzv==FALSE]
dim(testing)

# Theres a second transformation worth doing, which is removing the ID variable (the first column) so that it won't interfer with ML Algorithms:
training <- training[c(-1)]
testing <- testing[c(-1)]

# Codes to clean the variables with more than 60% NAs.The threshold was chosen based at the 60% method, which I considered the most appropriate Training Threshold method for this particular dataset.
# To explain what's going on: first, I will create a temporary subset to iterate in loop, then I will check for NA in every column in the dataset, select the columns under the 60% method and remove them.For closers, I will set it back to the proper dataset and remove the temp dataset.

# Cleaning the training dataset:
temp_train <- training
for (i in 1:length(training)) {
  if(sum(is.na(training[, i]))/nrow(training)>=0.6) {
    for (j in 1:length(temp_train)) {
      if(length(grep(names(training[i]), names(temp_train)[j]))==1){
        temp_train <- temp_train[, -j]
      }
    }
  }
}
dim(temp_train)  
  
training <- temp_train
rm(temp_train)

# Cleaning the testing dataset:
temp_test <- testing
for (i in 1:length(testing)) {
  if(sum(is.na(testing[, i]))/nrow(testing)>=0.6) {
    for (j in 1:length(temp_test)) {
      if(length(grep(names(testing[i]), names(temp_test)[j]))==1){
        temp_test <- temp_test[, -j]
      }
    }
  }
}
dim(temp_test)  

testing <- temp_test
rm(temp_test)


# I choose to build 3 models. One model, using a random forest ("rf"), the other, using decision trees and a third, using boosted trees ("gbm").
# Then, I intend to cross validate it predicting the outcomes and checking the accuracy of each model at the testing set.

# Codes to build the models:


modelFitRF <- train(roll_dumbbell~pitch_dumbbell, data = dataTrain, method = "rf", prox = TRUE)

modelFitGBM <- train(classe~yaw_arm, data = dataTrain, method = "gbm", verbose = FALSE)

