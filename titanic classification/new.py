import pandas as pd

# Load the dataset
data = pd.read_csv('titanic.csv')

# Preprocess the data
# Example: Filling missing values and encoding categorical features
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'])

# Select features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = data[features]
y = data['Survived']
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)




#####streamlit code ###############################3333



import streamlit as st
import numpy as np
import pandas as pd 

st.title('Titanic Survival Prediction')

st.sidebar.header('User Input Features')

def user_input_features():
    Pclass = st.sidebar.selectbox('Pclass', (1, 2, 3))
    Sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    Age = st.sidebar.text_input('Age', 0, 80, 29)
    # if Age==text_input:
    #     Age="Age"
    # else:
    #     Age="invalid"

    SibSp = st.sidebar.slider('SibSp', 0, 8, 0)
    Parch = st.sidebar.slider('Parch', 0, 6, 0)
    Fare = st.sidebar.slider('Fare', 0, 513, 32)
    Embarked = st.sidebar.selectbox('Embarked', ('C', 'Q', 'S'))

    data = {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode and preprocess the user input features
input_df['Sex'] = input_df['Sex'].map({'male': 0, 'female': 1})
input_df = pd.get_dummies(input_df, columns=['Embarked'])
input_df = input_df.reindex(columns=features, fill_value=0)

# # Display user input features
# st.subheader('User Input features')
# st.write(input_df)

# Predict survival
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Survived' if prediction[0] == 1 else 'Did not survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)