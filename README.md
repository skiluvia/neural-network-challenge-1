# Module 18 Challenge neural-network-challenge-1

## Overview
This company specializes in student loan refinancing. We are looking into finding a way to predict whether a borrower will repay their loan, since it can provide a more accurate interest rate for the borrower. This is a model to predict student loan repayment.

We have data in a CSV file that contains information about previous student loan recipients. We will be using machine learning and neural networks, you decide to use the features . We will use features in the provided dataset to create a model that will predict the likelihood that an applicant will repay their student loans. This CSV file contains information about these students, such as their credit ranking.

## Preparing the data for use on a neural network model
First, thing is to import all the necessary libraries and dependencies. Then we will load the provided data from `student-loans.csv` into a DataFrame. In viewing the data We find there are `12` rows. All columns are of `float64` data type except the last column which is of `float` data type. If we look at the data, we can also see that the last column is the target variable, which indicates whether the loan was repaid. We will use the `sklearn` library to scale the features and the target variables. 

We also see that there are two groups (`value_count`) as represented below:
```
credit_ranking
1    855
0    744
```

We will put our featured into `X` dataframe and `y` dataframe for the target variable. 

We will then split this data into training and testing datasets. We will use the `train_test_split` method to split the data into training and testing datasets. We will also use the `StandardScaler` to scale the features and target variables.

## Compile and Evaluate a Model Using a Neural Network
This is where **deep neaural network** starts by defining our features, creating the Sequential model instance, and defining the model's layers. 

We will use the `relu` activation function for all the hidden layers and the `sigmoid` activation function for the output layer. We will also use the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

## Loan Repayment prediction success by using your neural network model
If we display the sequential model summary, we get below. Please note that `additional hidden layer` with 3 nodes was added
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 8)                 96        
                                                                 
 dense_1 (Dense)             (None, 5)                 45        
                                                                 
 dense_2 (Dense)             (None, 3)                 18        
                                                                 
 dense_3 (Dense)             (None, 1)                 4         
                                                                 
=================================================================
Total params: 163 (652.00 Byte)
Trainable params: 163 (652.00 Byte)
Non-trainable params: 0 (0.00 Byte)
```
A layer was added to see if accuracy will improve. We will then compile the model, fit the model, and evaluate the model using the test data to determine the model's loss and accuracy.

One noticable thing was by recompiling and fitting the model, the accuracy improved from `0.46` to `0.83`. This is a good sign that the model is learning and improving. Also can see that the loss is decreasing as the model is learning. Below is an output of the model fitting

```
Epoch 1/150
38/38 [==============================] - 1s 2ms/step - loss: 0.7306 - accuracy: 0.4646
Epoch 2/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6995 - accuracy: 0.4879
Epoch 3/150
38/38 [==============================] - 0s 1ms/step - loss: 0.6830 - accuracy: 0.5563
Epoch 4/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6710 - accuracy: 0.6314
Epoch 5/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6601 - accuracy: 0.6597
Epoch 6/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6500 - accuracy: 0.6839
Epoch 7/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6398 - accuracy: 0.7064
Epoch 8/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6310 - accuracy: 0.7156
Epoch 9/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6228 - accuracy: 0.7231
Epoch 10/150
38/38 [==============================] - 0s 3ms/step - loss: 0.6161 - accuracy: 0.7331
Epoch 11/150
38/38 [==============================] - 0s 2ms/step - loss: 0.6100 - accuracy: 0.7356
Epoch 12/150
38/38 [==============================] - 0s 4ms/step - loss: 0.6043 - accuracy: 0.7356
Epoch 13/150
38/38 [==============================] - 0s 1ms/step - loss: 0.5996 - accuracy: 0.7348
...
Epoch 149/150
38/38 [==============================] - 0s 2ms/step - loss: 0.4388 - accuracy: 0.8032
Epoch 150/150
38/38 [==============================] - 0s 1ms/step - loss: 0.4388 - accuracy: 0.8023
```

## Recommendation system for student loans
When we avaluate the model, we get the following output
```
13/13 - 0s - loss: 0.5271 - accuracy: 0.7450 - 50ms/epoch - 4ms/step
Loss: 0.5270754098892212, Accuracy: 0.7450000047683716
```
The model's accuracy is `0.7450` which is fairly good. This means that the model is able to accurately predict whether a student will repay their loan. This means that the model is not learning well. We can conclude that the model is a fairly good model for predicting student loan repayment.

Also in looking at the classification report suggest the same observation
```
              precision    recall  f1-score   support

           0       0.71      0.78      0.74       188
           1       0.78      0.72      0.75       212

    accuracy                           0.74       400
   macro avg       0.75      0.75      0.74       400
weighted avg       0.75      0.74      0.75       400
```

## Conclusion

As for recommendation system for student loans, we can use use a combination of `content-based` and `collaborative-filtering` to recommend loans to students. While **Content-Based filtering** is based on specific of items themselves such as interest rates, repayment terms, loan forgiveness programs, lender reputation, and eligibility criteria. Content-based filtering can ensure that recommendations align with specific criteria or preferences provided by the student, where as **Collaborative filtering** is based on the interactions of users with items, preferences and behavior of similar users. For student loans, this could involve analyzing the choices made by students with similar academic backgrounds, financial situations, career aspirations, etc. Collaborative filtering can provide personalized recommendations by leveraging the wisdom of the crowd.

By combining these two techniques, you can leverage the strengths of both approaches:

Collaborative filtering can provide personalized recommendations based on similarities between students.
Content-based filtering can ensure that recommended loan options meet specific criteria or preferences provided by the student.

In the real word, the challenge would be other factors not included into the data. These factors could be **economic recession or pandemic. These factors could affect the student's ability to repay the loan.**