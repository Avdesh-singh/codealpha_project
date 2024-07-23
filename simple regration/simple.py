import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title of the app
st.title('Simple Linear Regression App')

# Upload dataset
uploaded_file = st.file_uploader("salary_data.csv", type=["csv"])

if uploaded_file is not None:
    # Read the data into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("Dataset:")
    # st.write(data)

    # Selecting features and target
    st.write("Select the feature and target columns")
    columns = data.columns.tolist()
    feature = st.selectbox("Select feature column", columns)
    target = st.selectbox("Select target column", columns)

    if feature and target:
        X = data[[feature]]
        y = data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the target for the test set
        y_pred = model.predict(X_test)

        # Calculate the mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # Display the results
        # st.write("Mean Squared Error:", mse)
        st.write("Model Coefficients:", model.coef_)
        st.write("Model Intercept:", model.intercept_)

        # Input for prediction
        st.write("Make a prediction")
        user_input = st.number_input(f"Enter {feature} value")

        if user_input:
            prediction = model.predict([[user_input]])
            st.write(f"Predicted {target} value:", prediction[0])

# Run the app with: streamlit run app.py