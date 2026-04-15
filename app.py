import streamlit as st 
import joblib 
import numpy as np 
import pandas as pd 

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

# Scale before building input array
to_scale = pd.DataFrame([[attendance , study_hours , freetime , goout]], columns=["AttendanceRate", "StudyHours", "FreeTime", "GoOut"])
scaled = scaler.transform(to_scale)[0] # flattening the (1,4) numpy array into a 1d array 
attendance_s, study_hours_s, freetime_s, goout_s = scaled[0], scaled[1], scaled[2], scaled[3] # Basically unpacking the 1d vector that scaler object returned 
 
## Building a Feature vector to pass into the Model 
base_features = [gender_enc , ses, edu_enc, school_enc , attendance_s , study_hours_s , internet , extracurricular , 
                 parent_support , romantic , freetime_s , goout_s ]

full_features = base_features + race_enc + locale_enc 
input_array = np.array(full_features).reshape(1,-1)




if st.button("Predict GPA"):
    prediction = linearRegressionModel.predict(input_array)
    st.success(f"The predicted GPA is: {prediction[0]:.2f}")
    print("input Array")
    print(input_array)
