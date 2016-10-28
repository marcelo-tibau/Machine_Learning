## Coursera: Practical Machine Learning

# Q.1 <-  commands to create non-overlapping training and test sets with about 50% of the observations assigned to each:
# A.1 <- The following codes:

library(caret)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)

adData = data.frame(diagnosis, predictors)
summary(adData)

testIndex = createDataPartition(diagnosis, p = 0.50, list = FALSE)
training = adData[-testIndex, ]
testing = adData[testIndex, ]



# Q.2 <- Make a plot of the outcome (CompressiveStrength) versus the index of the samples. Color by each of the variables in the data set:

data("concrete")
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training2 = mixtures[inTrain, ]
testing2 = mixtures[-inTrain, ]

library(Hmisc)
x_names = colnames(concrete)[1:8]
featurePlot(x=training2[, x_names], y=training2$CompressiveStrength, plot = "pairs")

# There was no relation between the outcome and other variables.Another shot is necessary. 
index_q2 = seq_along(1:nrow(training2))
ggplot(data = training2, aes(x=index_q2, y=CompressiveStrength)) + geom_point() + theme_bw() 

ggplot(data = training2, aes(x=index_q2, y=CompressiveStrength)) + geom_jitter(col="#31ead2") + theme_bw() 

# Tryout with a step-like pattern with 4 categories using the cut2() function in the Hmisc package to turn continuous covariates into factors:

CompressiveStrength_2 = cut2(training2$CompressiveStrength, g=4)

summary(CompressiveStrength_2)

ggplot(data = training2, aes(y=index_q2, x=CompressiveStrength_2)) + geom_boxplot() + geom_jitter(col="#df9005") + theme_bw()

## A.2 <- # There is a step-like pattern in the plot of outcome versus index in the training set that isn't explained by any of the predictor variables so there may be a variable missing.



## Q.3 <- Make a histogram and confirm the SuperPlasticizer variable is skewed.

data("concrete")
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training2 = mixtures[inTrain, ]
testing2 = mixtures[-inTrain, ]

par(mfrow=c(2,1))

ggplot(data = training2, aes(x=Superplasticizer)) + geom_histogram(col="#23b31b") + theme_bw()

t2 <- log(training2)
ggplot(data = t2, aes(x=Superplasticizer)) + geom_histogram(col="#d8852c") + theme_bw()

library(ggplot2)
library(grid)
library(gridExtra)

p1 = ggplot(data = training2, aes(x=Superplasticizer)) + geom_histogram(col="#23b31b") + theme_bw()
p2 = ggplot(data = t2, aes(x=Superplasticizer)) + geom_histogram(col="#d8852c") + theme_bw()
grid.arrange(p1, p2, ncol = 2, main = "Two Plots")

## A.3 <- There are values of zero so when you take the log() transform those values will be -Inf.



### Q.4 <- Find all the predictor variables in the training set that begin with IL. Perform principal components on these variables with the preProcess() function from the caret package. Calculate the number of principal components needed to capture 90% of the variance. 

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)

data(AlzheimerDisease)
adData2 = data.frame(diagnosis, predictors)

inTrain2 = createDataPartition(adData2$diagnosis, p=3/4)[[1]]
training3 = adData2[inTrain2, ]
testing3 = adData2[-inTrain2, ]

# To find all the predictor variables in the training set that begin with IL:

predictor_variable_IL = training3[, grep('^IL', x=names(training3))]
  
# To perform principal components on these variables with the preProcess() function from the caret package and to calculate the number of principal components needed to capture 90% of the variance:

principal_components = preProcess(predictor_variable_IL, method = 'pca', thresh = 0.9, outcome = training3$diagnosis)

principal_components$rotation

## A.4 <- 9



## Q.5 <- Create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis. Build two predictive models, one using the predictors as they are and one using PCA with principal components explaining 80% of the variance in the predictors. Use method="glm" in the train function.
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p=3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# To create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis:
set.seed(3433)
IL = grep("^IL", colnames(training), value = TRUE)
predictors_IL = predictors[, IL]
new_data = data.frame(diagnosis, predictors_IL)

inTrain3 = createDataPartition(new_data$diagnosis, p=3/4)[[1]]
training = new_data[inTrain3, ]
testing = new_data[-inTrain3, ]

# To build a model using the predictors as they are and to check the accuracy:

modelFit1 = train(diagnosis~., method = "glm", data = training)
prediction1 = predict(modelFit1, newdata = testing)

# if there is an error msg like: "Error in requireNamespaceQuietStop("e1071") : package e1071 is required", do the following and run the codes again:

#install.packages('e1071', dependencies=TRUE)

check_point1 = confusionMatrix(prediction1, testing$diagnosis)
print(check_point1)


# To build a model using PCA with principal components explaining 80% of the variance in the predictors and to check the accuracy:

modelFit2 = train(diagnosis ~ .,
                  method = "glm",
                  preProcess = "pca",
                  data = training,
                  trControl = trainControl(preProcOptions=list(thresh=0.8)))

check_point2 = confusionMatrix(testing$diagnosis, predict(modelFit2, testing))
print(check_point2)

# To read only both accuracies:
acc_check_point1 = check_point1$overall[1]  
acc_check_point2 = check_point2$overall[1]
acc_check_point1
acc_check_point2

## A.5 <- NON-PCA Accuracy = 0.6463415 / PCA Accuracy = 0.7195122 
