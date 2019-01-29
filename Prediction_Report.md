---
title: "Prediction Report"
author: "Lei Cheng"
date: "January 23, 2019"
output: 
  html_document:
    keep_md: true
---
#Summary:

Six participants were asked to perform one set of 10 repetitions of barbell lifts correctly and incorrectly in 5 different ways. Data were collected from the accelerometers on the belt, forearm, arm, and dumbell. In this project, we will use the data to predict the **manner** in which participants did the exercise. 


```r
library(tidyverse)
library(caret)
library(parallel)
library(doParallel)
library(randomForest)
library(rattle)
```

##Loading Data

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile = "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile = "pml-testing.csv")
training<-read_csv("pml-training.csv")
testing<-read_csv("pml-testing.csv")
```

##Data Cleaning and Splitting

The data set includes summary statistics with variable names prefixed with "kurtosis_", "skewness_", "max_", "min_", "amplitude_", "var_", "avg_", and "stddev_". These columns contain very high proportions of missing values. I dropped the columns with mostly NA's and used only the raw sensor data. I also dropped the first few columns that contain information on the participants and the time windows. To conduct cross validation, I splitted the *pml-training.csv* file into the training set and the validation set.


```r
training<-training%>%select_if(~!any(is.na(.)))
testing<-testing%>%select_if(~!any(is.na(.)))

training<-select(training,roll_belt:classe)
testing<-select(testing,roll_belt:magnet_forearm_z)

set.seed(12345)
inTrain<-createDataPartition(y=training$classe,p=0.7,list=F)
training<-training[inTrain,]
validation<-training[-inTrain,]
results<-validation$classe
validation<-validation%>%select(-classe)
```

##Configure Parallel Processing and *trainControl* Object

Parallel processing was configured on the computer in order to improve the performance of random forest. I also chose the k-fold cross validation with k=5 for the resampling method in the *trainControl* object.

```r
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv", number = 5,
                           allowParallel = TRUE)
```

##Model Training and Cross Validation
I used four different algorithms (trees, boosting, bagging and random forest) and compared their performance. Random forest and bagging yield the optimal accuracy of over 99% in the prediction on the validation set. I expect the out of sample error rate to be above 80%.

###Trees
Prediction with classification trees only yields an accuracy of around 50%, which means this is not a good model for this problem.

```r
fit_trees<-train(classe~.,method="rpart",data = training)
fit_trees
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03753433  0.4874586  0.32594283
##   0.05987862  0.3792904  0.14956447
##   0.11585800  0.3343457  0.07852292
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03753433.
```

```r
fancyRpartPlot(fit_trees$finalModel)
```

![](Prediction_Report_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

```r
pred<-predict(fit_trees,validation)
confMat<-confusionMatrix(pred,as.factor(results))
confMat$overall[1]
```

```
##  Accuracy 
## 0.4939173
```

```r
confMat$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1074  319  345  319  100
##          B   11  276   20  128   91
##          C   67  207  354  259  205
##          D    0    0    0    0    0
##          E    9    0    0    0  326
```

###Boosting
Boosting yields quite an improvement from classification trees. With a tree depth of 3 and 150 iterations, we can get an accuracy of over 97%. 

```r
fit_boosting<-train(classe~.,method="gbm", data = training, verbose=F)
fit_boosting
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7529172  0.6866471
##   1                  100      0.8186652  0.7704449
##   1                  150      0.8503636  0.8106193
##   2                   50      0.8533877  0.8141765
##   2                  100      0.9046523  0.8792865
##   2                  150      0.9291123  0.9102711
##   3                   50      0.8958363  0.8680848
##   3                  100      0.9398347  0.9238429
##   3                  150      0.9592428  0.9484179
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
plot(fit_boosting)
```

![](Prediction_Report_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
pred<-predict(fit_boosting,validation)
confMat<-confusionMatrix(pred,as.factor(results))
confMat$overall[1]
```

```
##  Accuracy 
## 0.9746959
```

```r
confMat$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1151   20    0    0    2
##          B    7  776   16    1    6
##          C    2    6  694   19    9
##          D    1    0    9  684    4
##          E    0    0    0    2  701
```


###Bagging
Bagging classification trees with 25 bootstrap replications yields an accuracy of over 99%.  

```r
fit_bag <- train(classe~., method="treebag",data=training,trControl = fitControl)
fit_bag
```

```
## Bagged CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10989, 10990, 10989, 10990 
## Resampling results:
## 
##   Accuracy   Kappa   
##   0.9842759  0.980109
```

```r
pred<-predict(fit_bag,validation)
confMat<-confusionMatrix(pred,as.factor(results))
confMat$overall[1]
```

```
## Accuracy 
##        1
```

```r
confMat$table 
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1161    0    0    0    0
##          B    0  802    0    0    0
##          C    0    0  719    0    0
##          D    0    0    0  706    0
##          E    0    0    0    0  722
```

###Random Forest
Random Forest yields an optimal accuracy of over 99% with the number of variables at each split mtry=2 and 500 trees.

```r
set.seed(12345)
fit_rf <- train(classe~., method="rf",data=training,trControl = fitControl)
fit_rf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10991, 10989, 10990, 10988 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9911194  0.9887655
##   27    0.9901000  0.9874765
##   52    0.9844943  0.9803840
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
fit_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.72%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    2    2    0    1 0.001280082
## B   16 2637    5    0    0 0.007900677
## C    0   26 2365    5    0 0.012938230
## D    0    0   34 2216    2 0.015985790
## E    0    0    0    6 2519 0.002376238
```

```r
plot(fit_rf)
```

![](Prediction_Report_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

```r
plot(fit_rf$finalModel)
```

![](Prediction_Report_files/figure-html/unnamed-chunk-8-2.png)<!-- -->

```r
pred<-predict(fit_rf,validation)
confMat<-confusionMatrix(pred,as.factor(results))
confMat$overall[1]
```

```
## Accuracy 
##        1
```

```r
confMat$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1161    0    0    0    0
##          B    0  802    0    0    0
##          C    0    0  719    0    0
##          D    0    0    0  706    0
##          E    0    0    0    0  722
```

##De-register parallel processing cluster

```r
stopCluster(cluster)
registerDoSEQ()
```

##Conclusion and Prediction
Upon comparing the accuracy of the four models, I decided that both bagging and Random Forest would be a good model for this problem. When tested on the test set, they yield the same results in the predictions.

```r
testPredRF<-predict(fit_rf,testing)
testPredRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
testPredBag<-predict(fit_bag,testing)
testPredBag
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

