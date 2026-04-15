import streamlit as st 
import joblib 
import numpy as np 

linearRegressionModel = joblib.load("models/linearregression.pkl")
scaler = joblib.load("models/SCALER.pkl")
st.title(" Student GPA Predictor ")
st.write("Please Enter the Student Information below to predict their GPA")

 #__Handling the user inputs 

gender = st.selectbox("Gender",["Male","Female"])
ses = st.slider("SES Quartile",1,4,2)
parental_edu = st.selectbox("Parental Education" , ["<HS", "HS", "SomeCollege", "Bachelors+"])
school_type = st.selectbox("School Type", ["Public", "Private"])
attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
study_hours = st.slider("Study Hours per Week", 0, 40, 10)
internet = st.selectbox("Internet Access", [0, 1])
extracurricular = st.selectbox("Extracurricular Activities", [0, 1])
parent_support = st.slider("Parent Support", 1, 5, 3)
romantic = st.selectbox("Romantic Relationship", [0, 1])
freetime = st.slider("Free Time", 1, 5, 3)
goout = st.slider("Goes Out", 1, 5, 3)
race = st.selectbox("Race", ["Asian", "Black", "Hispanic", "Other", "Two-or-more", "White"])
locale = st.selectbox("Locale", ["City", "Rural", "Suburban", "Town"])


# encoding the values that we received 

gender_enc = 0 if gender == "Male" else 1
school_enc = 0 if school_type == "Public" else 1
edu_map = {"<HS": 0, "HS": 1, "SomeCollege": 2, "Bachelors+": 3}
edu_enc = edu_map[parental_edu]

race_options = ["Asian", "Black", "Hispanic", "Other", "Two-or-more", "White"]
locale_options = ["City", "Rural", "Suburban", "Town"]
race_enc = [1 if race == r else 0 for r in race_options ] # check the value if it equals one of these replace it with one else then put 0 for every other race 
locale_enc = [1 if locale == l else 0 for l in locale_options ]

## Building a Feature vector to pass into the Model 
base_features = [gender_enc , locale_enc , ses, edu_enc , attendance , study_hours , internet , extracurricular , 
                 parent_support , romantic , freetime , goout ]

full_features = base_features + race_enc + locale_enc 
input_array = np.array(full_features).reshape(1,-1)

# Sclaing the features that needed scaling 

index_of_features_that_need_scaling = [11 ,5,10 ]

input_array[:,index_of_features_that_need_scaling] = scaler.transform(input_array[:,index_of_features_that_need_scaling])
# all rows of these columns

#predition time 

if st.button("Predict GPA"):
    prediction = linearRegressionModel.predict(input_array)
    st.sucess(f"The predicted GPA is: {prediction:.2f}")
