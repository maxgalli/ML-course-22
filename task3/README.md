# Task 3

## Task Description

In this task, you will make decisions on food taste similarity based on images and human judgements.

We provide you with a dataset of images of 10.000 dishes.

We also provide you with a set of triplets (A, B, C) representing human annotations: the human annotator judged that the taste of dish A is more similar to the taste of dish B than to the taste of dish C.

You task is to predict for unseen triplets (A, B, C) whether dish A is more similar in taste to B or C.

## Data Description

In the handout for this project, you will find the the following files:

- ```food.zip``` - archive of the dish images
- ```train_triplets.txt``` - contains the training triplets. The entries of each triplet denote file names. For example, the triplet "00723 00478 02630" denotes that the dish in image "00723.jpg" is more similar in taste to the dish in image "00478.jpg" than to the dish in image "02630.jpg" according to a human annotator.
- ```test_triplets.txt``` - the triplets you should make predictions for
- ```sample.txt``` - a sample submission file


Your task is the following: for each triplet (A, B, C) in test_triplets.txt you should predict 0 or 1 as follows:
1 if the dish in image A.jpg is closer in taste to the dish in image B.jpg than to the dish in C.jpg
0 if the dish in image A.jpg is closer in taste to the dish in image C.jpg than to the dish in B.jpg