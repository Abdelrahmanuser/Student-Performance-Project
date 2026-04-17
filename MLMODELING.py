import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score 
from dataengineering import load_data , preprocessing
from sklearn.linear_model import Ridge , Lasso 
from sklearn.ensemble import RandomForestRegressor 
import joblib 
# a method to evaluate all models 

def evaluate(model_name , y_true , y_pred ):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred ))
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true , y_pred)
    print(f"\n{model_name }")
    print(f" RMSE : {rmse:.4f}")
    print(f" MAE : {mae:.4f}")
    print(f"r2 : {r2:.4f}")
    return {"model":model_name , "RMSE":rmse , "MAE": mae , "R2":r2 }


# Testing out Another Additional Models 
def train_allModels(X_train , y_train , X_val , y_val):
    models =  {
       "Linear Regression" : LinearRegression() ,
       "Ridge" : Ridge(alpha=1.0) ,
        "Lasso" : Lasso(alpha = 0.001) ,
        # I guess my device was not all that powerful will have to reduce number of trees and cpu cores that 
        # are running at the same time 
        "Random Forest" : RandomForestRegressor(n_estimators = 100,  random_state = 42 , n_jobs = 4 )
    }
    results = [] 
    for name , model in models.items() :
        print(f"This Model: {name} is currently in Training")
        model.fit(X_train,y_train)
        predictions = model.predict(X_val)
        training_results = evaluate(name , y_val , predictions ) # the method we defined above
        results.append(training_results)
    return results 

def save_model_and_scaler(LinearRegressionModel , Scaler ):
    joblib.dump(LinearRegressionModel,"linearregression.pkl") 
    joblib.dump(Scaler,"SCALER.pkl")

if __name__ == "__main__":
    train , val , test = load_data() 
    
    X_train , X_val , scaler , X_test , y_train , y_val , y_test  = preprocessing(train, val , test)
    #results = train_allModels(X_train , y_train , X_val , y_val )
    best_model = LinearRegression() 
    best_model.fit(X_train,y_train)
    joblib.dump(best_model , "linearregression.pkl")
    joblib.dump(scaler,"SCALER.pkl")

    