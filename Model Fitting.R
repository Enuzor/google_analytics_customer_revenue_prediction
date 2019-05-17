library(plyr)
library(reshape2)
library(lubridate)
library(tidyverse)
library(gridExtra)
library(xgboost)
library(Matrix)
library(caret)
library(glmnet)
library(randomForest)

library(parallel)
library(doParallel)
library(cloudml)

train <- read_csv("/Users/peterfagan/Desktop/training_set.csv")
test <- read_csv("/Users/peterfagan/Desktop/testing_set.csv")

train <- as.data.frame(train)
test <- as.data.frame(test)
train$fullVisitorId <- NULL


### Splitting up datasets for analyisis ###

#Note in report we need to be able to evaluate performance.
#Therefore we require target variable values for test set.

#Splitting into training and test sets (smaller samples for quick code)
N_sample = round(nrow(train)*0.01)
itrain_sample = sample(1:N_sample,size=round(N_sample*0.7),replace=FALSE)
x_train_sample <- train[itrain_sample,c(-1,-2)]
x_test_sample <- train[-itrain_sample,c(-1,-2)]
target_train_sample <- train[itrain_sample,1]
target_test_sample <- train[-itrain_sample,1]
ret_train_sample <- as.factor(make.names(train[itrain_sample,2]))
ret_test_sample <- as.factor(make.names(train[-itrain_sample,2]))


#Full samples
N = nrow(train)
itrain = sample(1:N,size=round(N*0.7),replace=FALSE)
x_train <- train[itrain,c(-1,-2)]
x_test <- train[-itrain,c(-1,-2)]
target_train <- train[itrain,1]
target_test <- train[-itrain,1]
ret_train <- as.factor(make.names(train[itrain,2]))
ret_test <- as.factor(make.names(train[-itrain,2]))


x = data.matrix(x_train_sample)
y = target_train_sample
y_ret = ret_train_sample
x_t = data.matrix(x_test_sample)

### Feature Analysis ###

#Variable Importance (Interpretability)
param <- list(max.depth = 5, eta = 0.1,  objective="reg:linear",subsample=0.9)
xgb_mod <- xgboost(param, data = x, label =y,nround = 10)
pred_xgb <- predict(xgb_mod, x_t)
xgb_imp <- xgb.importance(feature_names = colnames(x),model = xgb_mod)
xgb.plot.importance(xgb_imp)
xgb.sel <- xgb_imp$Feature

#Variable Correlations
cor_mat <- round(cor(x[,-18]),2)
high_cor <- findCorrelation(cor_mat,cutoff=0.8)
colnames(x)[high_cor]
melted_cor_mat <- melt(cor_mat)
ggplot(data = melted_cor_mat, aes(x=Var1, y=Var2, fill=value)) + geom_tile() + theme(axis.text.x = element_text(angle = 60, hjust = 1))



### HyperParameter Tuning (Regression) ###

#XGBoost (Regression)
#Grid search to tune parameters.
#For the case of brevity of run time it has been minimised here.

xgb_trcontrol = trainControl(
  method = "cv",
  number = 10,  
  verboseIter = FALSE
)

xgbGrid <- expand.grid(nrounds = c(20,60,100),
                       max_depth = 8,
                       eta = c(0.01,0.02,0.05,0.1),
                       gamma=0,
                       min_child_weight = 1,
                       colsample_bytree = 0.8,
                       subsample = 1
)

cluster <- makeCluster(detectCores() - 1) #leave 1 core for Operating System
registerDoParallel(cluster)

set.seed(150)
xgb_model = caret::train(
  x, y,  
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  metric="RMSE",
  allowParallel = TRUE
)

stopCluster(cluster)

xgb_pred <- predict(xgb_model,x_t)
(RMSE_xgb <- sqrt(mean((xgb_pred-target_test_sample)^2)))

plot(xgb_model)
xgb_model$bestTune

#Neural Network (Regression)
#Grid search to tune parameters.
#For the case of brevity of run time it has been mimimised here.

nnet_trcontrol <- trainControl(
  method = 'cv', 
  number = 10, 
  verboseIter = FALSE
)

nnetGrid <- expand.grid(
  size = c(5,10,15,20),
  decay = c(0.01,0.05,0.1,0.2)
)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(150)
nnet_model = caret::train(
  x, y,  
  trControl = nnet_trcontrol,
  tuneGrid = nnetGrid,
  method = "nnet",
  metric="RMSE",
  preProcess=c("scale","center"),
  allowParallel = TRUE
)


stopCluster(cluster)

nnet_pred <- predict(nnet_model,x_t)
(RMSE_nnet <- sqrt(mean((nnet_pred-target_test_sample)^2)))

plot(nnet_model)
nnet_model$bestTune


#Lasso Regression

glmnet_trcontrol <- trainControl(
  method = 'cv', 
  number = 10, 
  verboseIter = FALSE
)

glmnetGrid <- expand.grid(
  alpha =  seq(0,1,0.1),
  lambda = seq(0.001,0.1,by = 0.001)
  )

registerDoSEQ()
set.seed(150)
glmnet_model <- train(
  x, y, 
  method = "glmnet", 
  trControl = glmnet_trcontrol ,
  preProcess=c("scale","center"),
  tuneGrid = glmnetGrid)

pred_glmnet <- predict(glmnet_model, newx = x_t, s = "lambda.min")
(RMSE_glm <- sqrt(mean((target_test_sample-pred_glmnet)^2)))

glmnet_model$bestTune



### Model Comparison ###
results <- resamples(list(xgboost=xgb_model, NeuralNetwork=nnet_model,Lasso =glmnet_model ))
results$values
summary(results)
bwplot(results)

rmse_xgb = rmse_nnet = rmse_glmnet = numeric(10)
folds = cut(1:30000,30,labels = FALSE)
random_folds = sample(folds,10000)

for(i in 1:30){
  itrain = which(random_folds==i)
  
  x_train <- data.matrix(train[itrain,c(-1,-2)])
  x_test <- data.matrix(train[-itrain,c(-1,-2)])
  y_train <- train[itrain,1]
  y_test <- train[-itrain,1]
  
  
  cluster <- makeCluster(detectCores() - 1) #leave 1 core for Operating System
  registerDoParallel(cluster)
   
   
  xgb_model = caret::train(
    x_train, y_train,  
    method = "xgbTree",
    metric="RMSE",
    allowParallel = TRUE
  )
   
  stopCluster(cluster)
   
  xgb_pred <- predict(xgb_model,x_test)
  rmse_xgb[i] <- sqrt(mean((xgb_pred-y_test)^2))


  cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
  registerDoParallel(cluster)
   
  nnet_model = caret::train(
    x_train, y_train,  
    method = "nnet",
    metric="RMSE",
    preProcess=c("scale","center")
  )
   
  stopCluster(cluster)
   
  nnet_pred <- predict(nnet_model,x_test)
  rmse_nnet[i] <- sqrt(mean((nnet_pred - y_test)^2))


  registerDoSEQ()
  glmnet_model <- train(
    x_train, y_train, 
    method = "glmnet"
    )
  
  pred_glmnet <- predict(glmnet_model, newx = x_test)
  rmse_glmnet[i] <- sqrt(mean((y_test-pred_glmnet)^2))
  
  
}

boxplot(rmse_xgb,rmse_nnet,rmse_glmnet,
        col = c("green","white","orange"),
        names=c("XGBoost","Neural Network","Lasso"),
        main="Boxplots of RMSE for Independent samples",
        ylab = "RMSE")

#Normality Assumption
shapiro.test(rmse_xgb) #Fails normaility test
shapiro.test(rmse_nnet) #Fails normality test
shapiro.test(rmse_glmnet)

#Assumption of homogeneity of variances
bartlett.test(dat$RMSE,dat$Model) #passes implies unequal variances 

rmses <- c(rmse_xgb,rmse_nnet,rmse_glmnet)
groups <- as.factor(c(replicate(10,"rmse_xgb"), replicate(10,"rmse_nnet"), replicate(10,"rmse_glmnet")))
dat <- as.data.frame(cbind(groups,rmses))
colnames(dat)=c("Model","RMSE")
anova <- aov(RMSE~Model,data=dat)
summary(anova) #Not significant at 0.05 level

t.test(rmse_nnet,rmse_glmnet,paired = TRUE)
t.test(rmse_nnet,rmse_xgb,paired = TRUE)
t.test(rmse_xgb, rmse_glmnet,pared=TRUE)



### Hyperparameter tuning (classification) ###

#Next we aim to improve our above predictions through additionally predicting the
#likelihood of a customer returning. This will be treated as a binary classification,
#problem from which I will extract class probabilities

#Neural Network (Classification)
nnetcl_trcontrol <- trainControl(
  method = 'cv', 
  number = 10, 
  verboseIter = FALSE,
  classProbs=TRUE,
  summaryFunction=twoClassSummary
)

nnetclGrid <- expand.grid(
  size = 10,
  decay = 0.01
)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

nnetcl_model <- caret::train(
  x, y_ret,  
  trControl = nnetcl_trcontrol,
  method = "nnet",
  metric="ROC",
  preProcess=c("scale","center"),
  allowParallel = TRUE
)


stopCluster(cluster)

#Determining the classification accuracy
nnetcl_pred <- predict(nnetcl_model,x_t,type="prob")
tb = table(max.col(nnetcl_pred),ret_test_sample)
(acc_nnetcl = sum(diag(tb))/sum(tb))

#Lasso for Classification
glmcl_mod <- cv.glmnet(x, y_ret, alpha = 1, family="binomial",type.measure = "auc", nfolds = 10)
pred_glmcl <- predict(glmcl_mod, newx = x_t, s = "lambda.min",type="class")
(tb = table(pred_glmcl,ret_test_sample))
(acc_glmcl = sum(diag(tb))/sum(tb))

pred_glmcl <- predict(glmcl_mod, newx = x_t, s = "lambda.min",type="response")



### Final Model & Prediction ###
nnet_trcontrol <- trainControl(
  method = 'cv', 
  number = 10, 
  verboseIter = FALSE
)

nnetGrid <- expand.grid(
  size = c(5,10,15,20),
  decay = c(0.01,0.05,0.1,0.2)
)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(150)
nnet_model = caret::train(
  x, y,  
  trControl = nnet_trcontrol,
  tuneGrid = nnetGrid,
  method = "nnet",
  metric="RMSE",
  preProcess=c("scale","center"),
  allowParallel = TRUE
)


stopCluster(cluster)

nnet_pred <- predict(nnet_model,x_t)


final_prediction <- pred_glmcl*nnet_pred
final_rmse <- sqrt(mean((target_test_sample - final_prediction)^2))




### Kaggle Competition submission (Run as google cloudml job) ###
x = data.matrix(x_train)
y = target_train
y_ret = ret_train
x_t = data.matrix(x_test)

nnet_trcontrol <- trainControl(
  method = 'cv', 
  number = 10, 
  verboseIter = FALSE
)

nnetGrid <- expand.grid(
  size = c(5,10,15,20,25,30),
  decay = c(0.01,0.05,0.1,0.2)
)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(150)
nnet_model = caret::train(
  x, y,  
  trControl = nnet_trcontrol,
  tuneGrid = nnetGrid,
  method = "nnet",
  metric="RMSE",
  preProcess=c("scale","center"),
  allowParallel = TRUE
)


stopCluster(cluster)

nnet_pred <- predict(nnet_model,x_t)

glmcl_mod <- cv.glmnet(x, y_ret, alpha = 1, family="binomial",type.measure = "auc", nfolds = 10)
pred_glmcl <- predict(glm_mod, newx = x_t, s = "lambda.min",type="class")


final_prediction = nnet_pred*pred_glmcl