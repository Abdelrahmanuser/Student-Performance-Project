from MLMODELING import target_vector , model_predictions
import numpy as np 

mean_GPA = np.mean(target_vector)

# Now I want to calcualte the Residual Sum of Squares which is 
# Summation (y_real - y_predicted) ^2 -  the square happens for every single data 

sumOfResidaualSquares  = np.sum(((target_vector - model_predictions )**2) )

#print(f"The sum of residual Squares Aka the total Error of the model: {sumOfResidaualSquares}")

# Next Is that we are going to calculate the Base line total error - we assume that the base line strategy 
# for a model trying to reduce its error is to predict the mean for all predictions so we see what will be the error
# in that case 

total_baseline_error = np.sum(((target_vector - mean_GPA )**2))

#print(f"This will be the total Error of the model if it simply predicted The mean for every single Value\n{total_baseline_error}")

print(f"R^2 is equal to{sumOfResidaualSquares/total_baseline_error}")
# Now as you can see 
# The total Error for the model ( Residual sum of squares) is 107.5k ~ GPA points all over 8 million samples
# Now the total Error for the Baseline Model - the mean ( every prediction is mean is)  ~ 211.1K over 8 million samples 

# so basically error per student is like 0.013 GPA Points for the trained model
# and error per student for the dumb model is 0.026375 GPA points for the dumb model 

# R^2 which is the coefficient of determinnance basiicaly captures - the amount by which 
# the model is able to correctly catch the wind - meaning 
# do the weights and bias values that we have result to predictions that move in the right direction ( up down ) when 
# a feature value is moved up or down - and does the prediction value move by the right amount 

# Formula for R^2 = 1 - (Residaul_sum_of_squares/ total_baseline_error)
# 
# So In essece the reason why we divide the total error of the trained model 
# by the error of the baseline model is that we investigate the relative performance of the model
# basically for the dumb model is is really retarded it does not even try to capture any patterns
# so in essence we are saying out of the total number of errors of the dumb models 
# what percent of these errors were not commited by the smart model val_smart / val_dumb 
# then you get a percent - now we know this percent was not commited by the smart model 
# meaning that the model was able to capture the pattern aka the weights and bias combo was able to result 
# in a prediction that shifts to the right direct by the right amount when feature values go up or down 
# Then 1 is the total variation aka - every single possible change in values 
# so we subtract 1- the value we got - aka the ratio to know what percent of the variation or the amount of variation 
# that was not modeled or predicted by the model - to be percise explained 