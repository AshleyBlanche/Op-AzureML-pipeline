# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data from direct marketing of a Portuguese banking institution. The classification target is to predict if the client will undertake a bank term deposit.

The best performing model was the HyperDrive model which had an accuracy of 0.918. In contrast, for the AutoML model the accuracy was 0.916 and the algorithm used was VotingEnsemble.

## Scikit-learn Pipeline

*Parameter Sampling*

I chose discrete values for parameters C and max_iter respectively.

C is for Regularisation and max_iter refers to the maximum number of iterations.

RandomParameterSampling was utilised as it is fast and supports early stopping of low-performance runs. If an extensive budget was available GridParameterSampling would have been ideal to use as it allows for extensive searches over the search space. BayesianParameterSampling can also be used to explore the hyperparameter space.

*Early Stopping Policy*

Early stopping policy has been utilised to automatically terminate poorly performing runs. The BanditPolicy has been utilised with an evaluation interval and a slack factor.

Runs that are not within the slack factor of the evaluation interval of the best performing run will be terminated. As such with respect to the policy, best performing runs continue until completion, hence the reason for its choice and use.

## AutoML

The following has been parameters have been used for Auto ML:

*experiment_timeout_minutes*

This is the exit criterion to define how long, the runing period should be in mins. To prevent timout the default 30 mins was utilised.

*task='classification'*

This paramter defines the classification for this task.

*primary_metric='accuracy'*

Accuracy is the metric that has been utilised.

*enable_onnx_compatible_models*

Open Neural Network Exchange (ONNX)has ben used. ONNX, inference, or model scoring, is the phase where the deployed model is used for prediction, most commonly on production data and is utlised for representing machine learning models. 

*n_cross_validations*

Cross validations is due to the same number of folds (number of subsets). As one cross-validation could result in overfit, two folds have been utilised to conduct cross-validation.


## Pipeline comparison

The difference in accuracy between the two models is relatively small. 

The HyperDrive model performed better in terms of accuracy, however overall the AutoML model is better due to its AUC_weighted metric and overall is a better fit as a model for the imbalance in the data. If more time was utilised for AutoML, the model results overall would be btter. AutoML as a model is beneficial as all of the necessary calculations, trainings, validations, etc. are all automated without the need for further input from myself. This is the main difference with the Scikit-learn Logistic Regression pipeline, in which adjustments, changes, etc. are needed manually by myself and come to a final model after many trials & errors.

## Future work

Imbalance is a common problem in classification problems for ML. Imbalanced data negatively impacts the model's accuracy due to it's ease for the model to be accurate  by predicting the main class, which can lead to the accuracy for the minority class catgorically failing. Due to this a s metric such as accuracy can cause issues when checking the accuracy of your model and can be misleading.

The main mitigations for imbalanced data are:

Changing the metric (e.g. AUC_weighted which s utilised fo balanced data)
Changing the algorithm used
Random Over-Sampling of minority class
Random Under-Sampling of majority class
Imbalanced-learn package

High data imbalance would be a focus for future execution, which would lead to improvement of the model.

Cross-validation could be another improvement as it is the process of taking subsets of training data and training the model on each subset. The greater the number of cross validations, the higher the accuracy. This could lead to greater training timescales which is proportional to cost so there must be a balance between the two factors.
