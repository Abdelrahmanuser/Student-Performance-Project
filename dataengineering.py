import numpy as np 
import pandas as pd 


# The Train.csv has around  8 Million 774 Rows 
#  a sample of 100 00 thousand rows to examnie the data
student_data = pd.read_csv("train.csv")


# Feature Engineering and Data Prep 

# checking for any duplicated rows 
# print(student_data.duplicated().sum()) - This Returned 0 means all rows are unique 

# Binary Encoding  / Label Encoding 

student_data["Gender"] = student_data["Gender"].map(
    {
        "Male": 0 ,
        "Female": 1 
    })

student_data["SchoolType"] = student_data["SchoolType"].map(
    {
        "Public": 0 , 
        "Private": 1 
    })



# Performing Ordinal Incoding on Parental Education 

# From Lowest Academic Degree till the highet 
student_data["ParentalEducation"] = student_data["ParentalEducation"].map(
    {
        "<HS": 0 ,
        "HS" : 1 , 
        "SomeCollege" : 2 , 
        "Bachelors+":3 
    })

# Performing One hot Encoding For Race and locale 

student_data = pd.get_dummies(student_data, columns = ["Race","Locale"]) 

# Verifying Feature Types 


#1 Confirming the encoding - by checking variable dtypes - ensuring the required variables are int , float , Bool 
#print(student_data.dtypes) 

#2 Data Set Shape and NAN  
    # (8000774, 29) Data Frame Shape 
    # There is no NAN values found in the data set ( expected from a synthetic data set )

#print(student_data.shape)
#print(student_data.isna().sum())

#  Step 2 Defining x and y variables 
# These will be the variables used  will leave out any grade related variable so that it does not influence the predictions 
  # X variables 
    # Gender
    # Race 
    # SES
    # PARENTAL EDUCATION 
    # LOCALE 
    # ATTENDANCE RATE 
    #  Study Hours
    #   Internet access 
    #   Extra curricular 
    #   Parent SUpport 
    #  Romantic 
    #  Freetime 
    #  Goout 
    # SchoolType
# The y varaibale is  GPA 

# Slicing the data we need 


train_data = student_data.loc[:,["Gender","SES_Quartile",
 "ParentalEducation","SchoolType","AttendanceRate","StudyHours","InternetAccess","Extracurricular","ParentSupport","Romantic","FreeTime",
 "GoOut","Race_Asian","Race_Black","Race_Hispanic","Race_Other","Race_Two-or-more"
 ,"Race_White","Locale_City","Locale_Rural","Locale_Suburban","Locale_Town"]
 ]


REAL_GPA_VALUES= student_data.loc[:,"GPA"]

# Below is a dictionary of dictionaries  ( transoformable variables )
t_variables = {
    "GoOut" : 
    {
        "MEAN" : train_data["GoOut"].mean() ,
        "STD" :  train_data["GoOut"].std() , 
    } ,
    "FreeTime" : 
    {
        
      "MEAN":  train_data["FreeTime"].mean(), 
       "STD": train_data["FreeTime"].std()
    }, 
    "StudyHours" :  
    {
        "MEAN" :  train_data["StudyHours"].mean() ,
        "STD" :train_data["StudyHours"].std() ,
    }
    
}



 

#   STEP 3 FEATURE SCALING USING Z-SCORE SCALING
     # Variables that need feature Scaling  Goout , FreeTime   , Study Hours  , 

train_data["GoOut"] = (train_data["GoOut"] -t_variables["GoOut"]["MEAN"] ) /t_variables["GoOut"]["STD"]  
train_data["FreeTime"] = (train_data["FreeTime"] -t_variables["FreeTime"]["MEAN"] ) /t_variables["FreeTime"]["STD"]  
train_data["StudyHours"] = (train_data["StudyHours"] - t_variables["StudyHours"]["MEAN"] ) /t_variables["StudyHours"]["STD"]  
#_______________________________________________- PROCESSING THE TEST.CSV
student_data_test = pd.read_csv("test.csv")

student_data_test["Gender"] = student_data_test["Gender"].map(
    {
        "Male": 0 ,
        "Female": 1 
    })

student_data_test["SchoolType"] = student_data_test["SchoolType"].map(
    {
        "Public": 0 , 
        "Private": 1 
    })



# Performing Ordinal Incoding on Parental Education 

# From Lowest Academic Degree till the highet 
student_data_test["ParentalEducation"] = student_data_test["ParentalEducation"].map(
    {
        "<HS": 0 ,
        "HS" : 1 , 
        "SomeCollege" : 2 , 
        "Bachelors+":3 
    })

# Performing One hot Encoding For Race and locale 

student_data_test = pd.get_dummies(student_data_test, columns = ["Race","Locale"]) 



test_data = student_data_test.loc[:,["Gender","SES_Quartile",
 "ParentalEducation","SchoolType","AttendanceRate","StudyHours","InternetAccess","Extracurricular","ParentSupport","Romantic","FreeTime",
 "GoOut","Race_Asian","Race_Black","Race_Hispanic","Race_Other","Race_Two-or-more"
 ,"Race_White","Locale_City","Locale_Rural","Locale_Suburban","Locale_Town"]
 ]


TEST_REAL_GPA_VALUES= student_data_test.loc[:,"GPA"]


#   STEP 3 FEATURE SCALING USING Z-SCORE SCALING
     # Variables that need feature Scaling  Goout , FreeTime   , Study Hours  ,  
     # For the test_data we use the train mean value and std because in the real world the model will be tested on data 
     # whcih it does not have statistics for 

test_data["GoOut"] = (test_data["GoOut"] -t_variables["GoOut"]["MEAN"] ) /t_variables["GoOut"]["STD"]  
test_data["FreeTime"] = (test_data["FreeTime"] -t_variables["FreeTime"]["MEAN"] ) /t_variables["FreeTime"]["STD"]  
test_data["StudyHours"] = (test_data["StudyHours"] - t_variables["StudyHours"]["MEAN"] ) /t_variables["StudyHours"]["STD"]  