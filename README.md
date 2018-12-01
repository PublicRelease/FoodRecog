# FoodRecog

This project contains our codes for experiments on two datasets, i.e. VireoFood-172 and Food101 datasets. All codes are in Python3 language with networks implemented using Pytorch packages. They are tested on Ubuntu 16.04 with Cuda 8.0.

The codes used for each dataset are organized in the respective folders, which has three categories:
1. TextProcessing: that processes the ingredient text of the dataset to produce the data as input to our ingredient channel.
2. Models: that includes the models trained sequentially to obtain our final model. 
3. ModelTest: that includes the codes for testing the performance of the models in the "Models" folder.

Details of the codes are illustrated in the readme file available under each dataset folder. 

Due to space limit, we provide a toy dataset in */VireoFood-172/data. It contains the first 32 examples in all of our input files. We hope they help the readers to know the data format of the inputs and easily apply our codes to their own problems. 
