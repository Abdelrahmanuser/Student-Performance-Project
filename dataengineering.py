import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 

#____ Loading All the three data sets into Pandas Data Frames
def load_data() :
    train = pd.read_csv("data/train.csv")
    val = pd.read_csv("data/validation.csv")
    test = pd.read_csv("data/test.csv")
    return train  , val , test 

# Encoding a version of the data frame while keeping the original one as is  

def encode(df) :
    df = df.copy() 
    df["Gender"] = df["Gender"].map({"Male":0, "Female":1}) 
    df["SchoolType"] = df["SchoolType"].map({"Public":0,"Private":1})
    df["ParentalEducation"] = df["ParentalEducation"].map(
        {"<HS":0,"HS":1 ,"SomeCollege":2,"Bachelors+":3}
        )
    df = pd.get_dummies(df,columns= ["Race","Locale"])
    return df 

# Defining the Features That the model will be trained on 

FEATURES = [
    "Gender", "SES_Quartile", "ParentalEducation", "SchoolType",
    "AttendanceRate", "StudyHours", "InternetAccess", "Extracurricular",
    "ParentSupport", "Romantic", "FreeTime", "GoOut",
    "Race_Asian", "Race_Black", "Race_Hispanic", "Race_Other",
    "Race_Two-or-more", "Race_White",
    "Locale_City", "Locale_Rural", "Locale_Suburban", "Locale_Town"
]
SCALE_FEATURES = ["AttendanceRate", "StudyHours", "FreeTime", "GoOut"]

TARGET = "GPA"
 
def preprocessing(train_raw , val_raw , test_raw) :
    train = encode(train_raw)
    val = encode(val_raw)
    test = encode(test_raw)

    # Seperating The fetures and the target for each data set 

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val,   y_val   = val[FEATURES] , val[TARGET]
    X_test,  y_test  = test[FEATURES] , test[TARGET]
    scaler = StandardScaler() 
    X_train[SCALE_FEATURES] = scaler.fit_transform(X_train[SCALE_FEATURES])
    # we are basically calculating std and mean using the train data set 
    X_val[SCALE_FEATURES] = scaler.transform(X_val[SCALE_FEATURES])
    X_test[SCALE_FEATURES] = scaler.transform(X_test[SCALE_FEATURES])
    X_train_sample = X_train.sample(500000,random_state = 42 )
    y_train_sample = y_train.loc[X_train_sample.index] 
    # the above line ensures that we are getting the correct y values that 
    # match the randomly sampled x values 
    return X_train_sample, X_val , scaler , X_test , y_train_sample, y_val , y_test 

if __name__ == "__main__" :
    train_raw , val_raw , test_raw  = load_data()
    print(train_raw[["StudyHours","GPA"]].corr())
    X_train , X_val , scaler , X_test , y_train , y_val , y_test  = preprocessing(train_raw , val_raw , test_raw)
    print(f" The dimensions of the Train data set is: {X_train.shape}")
    print(f"We should have 8 million rows since this is y_train{y_train.shape}")
    print(f" The dimensions of the test data set is: {X_test.shape}")
    print(f" The dimensions of the validation data set is: {X_val.shape}")
    print("X_train_sample first index:", X_train.index[0])
    print("y_train_sample first index:", y_train.index[0])



