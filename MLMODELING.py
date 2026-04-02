from dataengineering import train_data , test_data , TEST_REAL_GPA_VALUES ,REAL_GPA_VALUES
import numpy as np 
import pandas as pd 
import torch as pt 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics  import mean_squared_error , mean_absolute_error ,r2_score 

# CONVERTING THE DATA INTO MATRIX FORAMT 

ml_train = train_data.to_numpy()
ml_test = test_data.to_numpy() 
target_vector = TEST_REAL_GPA_VALUES.values 
target_train = REAL_GPA_VALUES.values 




# Creating  a Linear Regression Model 
linear_model = LinearRegression() 
linear_model.fit(ml_train,target_train)  # training the model 

# Making The Model precit 

model_predictions = linear_model.predict(ml_test) # making the model predict the GPA based on the train data set 

# Now evaluating the performance of the model 
rmse = np.sqrt(mean_squared_error(target_vector ,model_predictions))

mae = mean_absolute_error(target_vector , model_predictions )

r2 = r2_score(target_vector , model_predictions )
#print(f"Root Mean Squared Error {rmse}" )
#print(f"Mean Absolte Error {mae} ")
#print("R2: ", r2 ) # one could say that the model explains 50% of the existing variability  in the target_varaible 

# The coefficiens tell us  on how each feature influences GPA 
# for name , coef in zip(train_data.columns , linear_model.coef_):
   # print(name,coef )

print(rmse , mae , r2 )