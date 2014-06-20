Practical Machine Learning WriteUp
===============================

Author: Zhang Tong <lovewilliam@gmail.com>

REPO: https://github.com/lovewilliam/predmachlearn-002  

1.How you built your model
---------------------------
##1.1 Preprocess data##
open the ``pml-training.csv'' to examine raw data.

extract header names using following shell command

```sh
sh ~ # head -n1 pml-training.csv
"","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp",...,"classe"
```

- noticed the 1st column contains index which is useless, so we omit it.
- noticed the 2rd column contains username which may be useful, but I think to grade human activity, there must be some rules that work on all people not just one specific person. So, just omit it to test. If it doesn't work, then we will check it later.
- noticed there are some time stamp columns which I think is obviously useless to determine whether the activity is good or not. Just ignore these columns.
- noticed there are lots of **no** in column `new_window` so it is not suitable for prediction. `num_window` is also obviously have nothing to do with `classe` for ABCDE are distributed evenly on `num_window` range.
- columns contain **NaN** and **empty** values will not contribute to the model.
- row with col `new_window==yes` only count up to 406 items which is a small number compared to all item count 19622. I treat these rows as special rows and negeleted them.

the final columns are 
```
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
```

The column names for test data set are `col_names` without `classe`.

##1.2 Split Data##
Split data into train and validation partition for we already have test data.

##1.3 Reduce Dimension##
To make the training process runs faster, it's better to reduce dimension of the data using PCA. As I see there may be some redundant columns such as `total_accel*` that may be the abs value of sum of vector `accel*`. So the matrix may be not be a full rank matrix.

There are 53 columns in preprocesses data. That's way too much I think. Using PCA to achieve thresh=0.99 we got 36 components. About 32% of the data are removed.


##1.4 Train Model##

As I observed, to determine whether an action is good or not which follows a decision-tree like pattern.

>1.Stand in front of a loaded barbell.  
>2.While keeping the back as straight as possible, bend your knees, bend forward and grasp the bar using a medium (shoulder width) overhand grip. This will be the starting position of the exercise. Tip: If it is difficult to hold on to the bar with this grip, alternate your grip or use wrist straps.  
>3.While holding the bar, start the lift by pushing with your legs while simultaneously getting your torso to the upright position as you breathe out. In the upright position, stick your chest out and contract the back by bringing the shoulder blades back. Think of how the soldiers in the military look when they are in standing in attention.  
>4.Go back to the starting position by bending at the knees while simultaneously leaning the torso forward at the waist while keeping the back straight. When the weights on the bar touch the floor you are back at the starting position and ready to perform another repetition.
Perform the amount of repetitions prescribed in the program.  
>  
>From: http://www.bodybuilding.com/exercises/detail/view/name/barbell-deadlift


So I choose `randomForest` to get the decision tree.

##1.5 Speed up training speed using Multi-core##

I tested a small part of training data first to get better know of the model. I found that the training speed is very slow. About 1 hour on my CPU using 1 core. I decided to speed up using multi-core by using library doMC.

After using doMC on 4 cores, time used by train on full-sized `traing_data` reduced to about 17 min.

##1.6 Result##
When using PCA=0.99 and randomForest we got 1.92% error rate.

```
> fit_rf$finalModel

Call:
 randomForest(x = x, y = y, mtry = param$mtry)
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 2

        OOB estimate of  error rate: 1.92%
Confusion matrix:
     A    B    C    D    E class.error
A 4175    5    1    2    2 0.002389486
B   59 2764   24    0    1 0.029494382
C    2   36 2517   11    1 0.019477990
D    1    0   90 2315    6 0.040215589
E    2    6   20   13 2665 0.015151515
>
```

NO PCA, use only randomForest: we got 0.7% error rate.
```
> fit$finalModel

Call:
 randomForest(x = x, y = y, mtry = param$mtry)
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 2

        OOB estimate of  error rate: 0.7%
Confusion matrix:
     A    B    C    D    E  class.error
A 4183    1    0    0    1 0.0004778973
B   21 2818    9    0    0 0.0105337079
C    0   20 2545    2    0 0.0085703155
D    0    0   41 2368    3 0.0182421227
E    0    0    1    4 2701 0.0018477458
>
```

<img width=500 src="http://lovewilliam.github.io/predmachlearn-002/var_imp.png" />

I think that is good enough for prediction.

2.How you used cross validation 
-------------------------------
I partitioned original training data as following: 3/4 training, 1/4 validation.

I predicted outcome using validation data using trained model. And I checked error rate of the validation data set. I think the error rate is in an acceptable range. So I just predicted outcome on test dataset. And got:

```
B A B A A E D B A A B C B A E E A B B B
```

3.What you think the expected out of sample error is  
----------------------------------------------------
I think the obvious error is misclassification.

without PCA I got this on validation data
```
> confusionMatrix(res,refinedValidation$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1395    3    0    0    0
         B    0  945    6    0    0
         C    0    1  849   15    0
         D    0    0    0  788    1
         E    0    0    0    1  900

Overall Statistics
                                         
               Accuracy : 0.9945         
                 95% CI : (0.992, 0.9964)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.993          
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9958   0.9930   0.9801   0.9989
Specificity            0.9991   0.9985   0.9960   0.9998   0.9998
Pos Pred Value         0.9979   0.9937   0.9815   0.9987   0.9989
Neg Pred Value         1.0000   0.9990   0.9985   0.9961   0.9998
Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
Detection Rate         0.2845   0.1927   0.1731   0.1607   0.1835
Detection Prevalence   0.2851   0.1939   0.1764   0.1609   0.1837
Balanced Accuracy      0.9996   0.9971   0.9945   0.9899   0.9993
> 
```

which means that
B is possible to be mis-classified as A(more) and C(less)
C is possible to be mis-classified as B
D is possible to be mis-classified as C(more) and E
E is possible to be mis-classified as D

In summary, the predictor tends to give higher grade on validation data.
