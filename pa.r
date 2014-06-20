library(caret)
library(doMC)
registerDoMC(cores = 8)

data<-read.csv(file="pml-training.csv",header=TRUE,sep=",")

datatest<-read.csv(file="pml-testing.csv",header=TRUE,sep=",")

set.seed(15151)

inTrain = createDataPartition(data$classe, p = 3/4)[[1]]
training = data[ inTrain,]
validation = data[-inTrain,]

#######################################
#pre-process data
col_names<-c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt",
			"gyros_belt_x", "gyros_belt_y", "gyros_belt_z",
			"accel_belt_x",	"accel_belt_y", "accel_belt_z",
			"magnet_belt_x", "magnet_belt_y", "magnet_belt_z",
			"roll_arm",	"pitch_arm", "yaw_arm",	"total_accel_arm",
			"gyros_arm_x", "gyros_arm_y", "gyros_arm_z",
			"accel_arm_x", "accel_arm_y", "accel_arm_z",
			"magnet_arm_x", "magnet_arm_y",	"magnet_arm_z",
			"roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell",
			"gyros_dumbbell_x",	"gyros_dumbbell_y",	"gyros_dumbbell_z",
			"accel_dumbbell_x",	"accel_dumbbell_y",	"accel_dumbbell_z",
			"magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z",
			"roll_forearm","pitch_forearm","yaw_forearm", "total_accel_forearm",
			"gyros_forearm_x","gyros_forearm_y","gyros_forearm_z",
			"accel_forearm_x","accel_forearm_y","accel_forearm_z",
			"magnet_forearm_x","magnet_forearm_y","magnet_forearm_z",
			"classe")
test_col_names<-c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt",
			"gyros_belt_x", "gyros_belt_y", "gyros_belt_z",
			"accel_belt_x",	"accel_belt_y", "accel_belt_z",
			"magnet_belt_x", "magnet_belt_y", "magnet_belt_z",
			"roll_arm",	"pitch_arm", "yaw_arm",	"total_accel_arm",
			"gyros_arm_x", "gyros_arm_y", "gyros_arm_z",
			"accel_arm_x", "accel_arm_y", "accel_arm_z",
			"magnet_arm_x", "magnet_arm_y",	"magnet_arm_z",
			"roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell",
			"gyros_dumbbell_x",	"gyros_dumbbell_y",	"gyros_dumbbell_z",
			"accel_dumbbell_x",	"accel_dumbbell_y",	"accel_dumbbell_z",
			"magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z",
			"roll_forearm","pitch_forearm","yaw_forearm", "total_accel_forearm",
			"gyros_forearm_x","gyros_forearm_y","gyros_forearm_z",
			"accel_forearm_x","accel_forearm_y","accel_forearm_z",
			"magnet_forearm_x","magnet_forearm_y","magnet_forearm_z")

refinedTrain = training[col_names]
refinedValidation = validation[col_names]
refinedTest = datatest[test_col_names]

#######################################
#PCA
#0.80 -> 12 PCA
#0.90 -> 18 PCA
preProc<-preProcess(refinedTrain[-53],method="pca",thresh=.99)
pca_train<-predict(preProc,refinedTrain[-53])
pca_validation<-predict(preProc,refinedValidation[-53])
pca_test<-predict(preProc,refinedTest)

#pca_train<-refinedTrain[-53]
#pca_validation<-refinedValidation[-53]
#pca_test<-refinedTest

#######################################
#Neural Network
#nnet
#0.8pca maxit=100 2535/4904 
#0.9pca maxit=100 2855/4904
#0.8pca maxit=1000 2834/4904 
#0.9pca maxit=1000 ?/4904
#fit<-train(refinedTrain$classe~.,data=pca_train,method='nnet',maxit=1000,trace=T,linout=1)

#multinom
#0.8pca 2511/4904
#0.9pca 2544/4904
#fit<-train(refinedTrain$classe~.,data=pca_train,method='multinom',hidden=5,threshold=0.01,trace=T,linout=1)


#######################################
#Random forest
#pca0.8 4727/4904
#> fit_rf$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry) 
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 3.76%
#Confusion matrix:
#     A    B    C    D    E class.error
#A 4089   21   48   20    7  0.02293907
#B   72 2691   64   14    7  0.05512640
#C   20   40 2468   34    5  0.03856642
#D   17   10   99 2277    9  0.05597015
#E    3   24   18   22 2639  0.02475979
#
#pca0.9 
#> fit_rf$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry) 
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 2.62%
#Confusion matrix:
#     A    B    C    D    E class.error
#A 4137   13   25    5    5  0.01146953
#B   64 2740   40    0    4  0.03792135
#C   11   41 2492   21    2  0.02921698
#D    8    5   93 2300    6  0.04643449
#E    3   12   17   10 2664  0.01552106
#> 
#pca0.95
#> fit_rf$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry, trace = TRUE)
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 2.35%
#Confusion matrix:
#     A    B    C    D    E class.error
#A 4149    9   22    3    2 0.008602151
#B   52 2752   39    0    5 0.033707865
#C    2   34 2507   20    4 0.023373588
#D    3    5   92 2306    6 0.043946932
#E    1   13   20   14 2658 0.017738359
#>
#pca0.8 tc_repeat=3
##
#> fit_rf$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry, tr = ..1)
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 2.41%
#Confusion matrix:
#     A    B    C    D    E class.error
#A 4154   10   17    2    2 0.007407407
#B   50 2752   42    0    4 0.033707865
#C    3   37 2506   21    0 0.023763148
#D    4    5   99 2301    3 0.046019900
#E    1   15   21   19 2650 0.020694752
#>
#pca0.9 tc_repeat=10
#> fit_rf$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry, tr = ..1)
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 2.52%
#Confusion matrix:
#     A    B    C    D    E class.error
#A 4140   10   21    9    5  0.01075269
#B   57 2743   45    0    3  0.03686798
#C   12   34 2498   21    2  0.02687963
#D    4    5   94 2303    6  0.04519071
#E    1   12   15   15 2663  0.01589061
#>
#pca0.99
#> fit_rf$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry)
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 1.92%
#Confusion matrix:
#     A    B    C    D    E class.error
#A 4175    5    1    2    2 0.002389486
#B   59 2764   24    0    1 0.029494382
#C    2   36 2517   11    1 0.019477990
#D    1    0   90 2315    6 0.040215589
#E    2    6   20   13 2665 0.015151515
#>

#NO PCA
#> fit$finalModel
#
#Call:
# randomForest(x = x, y = y, mtry = param$mtry)
#               Type of random forest: classification
#                     Number of trees: 500
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 0.7%
#Confusion matrix:
#     A    B    C    D    E  class.error
#A 4183    1    0    0    1 0.0004778973
#B   21 2818    9    0    0 0.0105337079
#C    0   20 2545    2    0 0.0085703155
#D    0    0   41 2368    3 0.0182421227
#E    0    0    1    4 2701 0.0018477458
#>


#fit_ctrl<-trainControl(method = "repeatedcv", repeats = 10)
#fit_rf<-train(refinedTrain$classe~.,method="rf",data=pca_train,tr=fit_ctrl)

fit_rf<-train(refinedTrain$classe~.,method="rf",data=pca_train)

fit<-fit_rf
#parallel random forest

res<-predict(fit,pca_validation)
sum(res==refinedValidation$classe)
length(res==refinedValidation$classe)

#######################################
#predict on test
res_test<-predict(fit_rf,pca_test)

confusionMatrix(res,refinedValidation$classe)

#######################################
#write out result
answers<-res_test

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
