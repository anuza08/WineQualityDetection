import streamlit as st
import numpy as np
import pickle

# Load the trained model
filename = 'wine_quality_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to predict wine quality
def predict_wine_quality(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction[0]

# Main function for Streamlit app
def main():
    st.title("Wine Quality Prediction")
    st.sidebar.title("Input Features")

    # Set background color for main content area
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f2f6
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input fields for user to input wine characteristics
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.6, 16.0, 8.31, step=0.1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.12, 1.58, 0.52, step=0.01)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.28, step=0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.9, 15.5, 2.5, step=0.1)
    chlorides = st.sidebar.slider("Chlorides", 0.012, 0.611, 0.087, step=0.001)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1.0, 72.0, 15.0, step=1.0)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6.0, 289.0, 46.0, step=1.0)
    density = st.sidebar.slider("Density", 0.990, 1.004, 0.996, step=0.001)
    pH = st.sidebar.slider("pH", 2.74, 4.01, 3.31, step=0.01)
    sulphates = st.sidebar.slider("Sulphates", 0.33, 2.0, 0.68, step=0.01)
    alcohol = st.sidebar.slider("Alcohol", 8.4, 14.9, 10.4, step=0.1)

    # Format input data as numpy array
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                            pH, sulphates, alcohol]])

    if st.sidebar.button("Predict"):
        prediction = predict_wine_quality(input_data)
        if prediction == 'bad':
            st.error("The predicted wine quality is: Bad Quality Wine")
        elif prediction == 'moderate':
            st.warning("The predicted wine quality is: Moderate Quality Wine")
        elif prediction == 'best quality':
            st.success("The predicted wine quality is: Best Quality Wine")
        else:
            st.info("The predicted wine quality is not classified as bad, moderate, or best quality.")

# Run the main function
if __name__ == "__main__":
    main()
