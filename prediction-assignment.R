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

# Code to cheack possibles Near Zero Variance Variables.
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


# I chose to build 3 models. One model, using a random forest ("rf"), the other, using decision trees and a third, using boosted trees ("gbm").
# Then, I intend to cross validate it predicting the outcomes and checking the accuracy of each model at the testing set.

# Codes to build the models:

# Random Forest Algorithm
set.seed(13563)
modelFitRF <- randomForest(classe~., data = training)

# Cross validating the model:
predictFitRF <- predict(modelFitRF, testing, type = "class")

# To check the accuracy:
accuracy_FitRF <- confusionMatrix(predictFitRF, testing$classe)
accuracy_FitRF

# The accuracy of the Random Forest model is 0.9983, a very good one. To facilitate the visualization, I intend to plot it.
plot(modelFitRF, main = "Random Forest Algorithm")

plot(accuracy_FitRF$table, col = accuracy_FitRF$byClass, main = paste("Random Forest Algorithm Accuracy =", round(accuracy_FitRF$overall['Accuracy'], 4)))


# Decision Tree Algorithm
set.seed(13563)
modelFitDT <- rpart(classe ~., data = training, method = "class")
fancyRpartPlot(modelFitDT)

# Cross validating the model:
predictFitDT <- predict(modelFitDT, testing, type = "class")

# To check the accuracy:
accuracy_FitDT <- confusionMatrix(predictFitDT, testing$classe)  
accuracy_FitDT

# The model accuracy rate is 0.8731. Not a bad one, but less then Random Forest's.Again, in order to facilitate the visualization, a plot is needed.
plot(accuracy_FitDT$table, col = accuracy_FitDT$byClass, main = paste("Decision Tree Algorithm Accuracy =", round(accuracy_FitDT$overall['Accuracy'], 4)))


# Boosted Trees Algorithm
library(plyr)
set.seed(13563)

# I usually don't use the trainControl function at the caret package because one of its uses is allow to perform a variety of cross validation.
# As the Confusion Matrix and the predict function allow us to do the same, I usually don't see the point to trainControl the model. However, in this case
# the model fit is taking to long, so I used it to cut it short.

FitControlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

modelFitGBM <- train(classe~., data = training, method = "gbm", trControl = FitControlGBM, verbose = FALSE)

FinalmodelFitGBM <- modelFitGBM$finalModel

# Cross validating the model:
predictFitGBM <- predict(modelFitGBM, newdata = testing)

# To check the accuracy:
accuracy_FitGBM <- confusionMatrix(predictFitGBM, testing$classe)
accuracy_FitGBM

# The accuracy of the model is rated at 0.9966. Although, comparing to Random Forest's 0.9983 it's not the best model. 
# Once again, a plot to facilitate the visualization.
plot(modelFitGBM, ylim = c(0.9, 1))

## Using the prediction model to predict 20 different test cases
# Random Forests gave an accuracy of 99.89%, this means that this model is more accurate then the Decision Trees or GBM models.
# The expected out-of-sample error is 0.11% (100-99.89).
prediction_results <- predict(modelFitRF, testing, type = "class")
prediction_results

# To generate a text file with predictions to submit for assignment:
file_to_assignment <- function(x){
  n=length(x)
  for (i in 1:n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}

file_to_assignment(prediction_results)


## Codes to answer the Course Project Prediction Quizz

# code for submission

dataTest <- dataTest[,colSums(is.na(dataTest)) == 0]

colnames(dataTest)

dataTest <- dataTest[c(-60)]

dataTest["classe"] <- NA 

dataTest <- dataTest[c(-1)]

feature_set <- colnames(training)
newdata <- dataTest

x <- newdata 
x <- x[feature_set[feature_set!='classe']]

answers <- predict(modelFitRF, newdata=x)

