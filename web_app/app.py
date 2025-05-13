import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Load model and scaler
try:
    model = joblib.load('./models/rf_model.joblib')
    scaler = joblib.load('./models/scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please run train_model.py first.")
    st.stop()

# Feature names (same as breast cancer dataset)
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
    'worst fractal dimension'
]

# Streamlit app
st.title("Breast Cancer Prediction System")
st.markdown("Enter the 30 features below to predict whether a tumor is **benign** or **malignant**. All values must be non-negative numbers.")

# Create a form for input
with st.form(key="prediction_form"):
    st.header("Input Features")
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    features = []
    
    # Split features between two columns for better layout
    for i, name in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:
            value = st.number_input(
                label=name.capitalize(),
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key=name
            )
            features.append(value)
    
    # Submit button
    submit_button = st.form_submit_button("Predict")

# Process prediction
if submit_button:
    try:
        # Validate inputs
        if any(np.isnan(f) for f in features):
            st.error("All fields must be filled with valid numbers.")
        else:
            # Convert features to numpy array and scale
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Display results
            st.success("Prediction completed!")
            st.subheader("Results")
            result = "Benign" if prediction == 1 else "Malignant"
            confidence = probabilities[prediction] * 100
            
            st.write(f"**Prediction**: {result}")
            st.write(f"**Confidence**: {confidence:.2f}%")
            st.write(f"**Probability (Malignant)**: {probabilities[0] * 100:.2f}%")
            st.write(f"**Probability (Benign)**: {probabilities[1] * 100:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add a note about feature ranges
with st.expander("Help: About the Features"):
    st.markdown("""
    The 30 features correspond to measurements from a breast cancer biopsy. Examples include:
    - **Mean radius**: Average size of the tumor (typical range: 6.9–28.1).
    - **Mean texture**: Variation in gray-scale intensities (typical range: 9.7–39.3).
    - **Mean perimeter**: Tumor perimeter (typical range: 43.8–188.5).
    
    Use realistic values based on medical data. For testing, you can use approximate means from the dataset.
    """)