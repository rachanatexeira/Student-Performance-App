import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# from tensorflow.tools.docs.doc_controls import header

# from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

st.title('Higher Education Student Performance')
st.write('Upload a CSV file containing student data to predict and analyze their performance. The app uses key factors such as Gender, Class Attendance, and other relevant features to provide insights into how these variables influence student success. Simply upload your file, and the app will process the data and give you performance predictions based on the model’s analysis.')
st.write('Please note: Model is prepopulated with sample csv. ')
# Load the pre-trained model
model_svm_path = "model_svm.pkl"  # Replace with your saved model path
scaler_minmax_path = "scaler_minmax.pkl"
with open(model_svm_path, "rb") as model_file:
    model = pickle.load(model_file)
with open(scaler_minmax_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

uploaded_file = st.file_uploader('Choose a file: ')
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin-1', header=0)
else:
    df = pd.read_csv('student_data.csv', encoding='latin-1', header=0)

st.header('Student Performance')

st.write('Uploaded Dataset')
st.dataframe(df)
st.write(df.describe())

if df.isnull().sum().sum() > 0:
    st.error("Dataset contains missing values. Please clean the data and try again.")
else:  # Ensure all columns are numeric
    X = df.astype(float)  # Convert all columns to float for scaler compatibility

    # Scale the data
    scaled_data = scaler.transform(X)

    # Make predictions
    predictions = model.predict(scaled_data)
    prediction_labels = {0: "Low", 1: "Average", 2: "High"}
    df["Prediction"] = predictions
    df["Prediction_Label"] = df["Prediction"].map(prediction_labels)

    # Display predictions
    st.header("Predictions:")
    st.dataframe(df['Prediction_Label'])

    # Visualization
    st.header("Performance Visualization")
    # Count the number of students in each category (Low, Medium, High)
    performance_counts = df["Prediction_Label"].value_counts()

    # Show the counts in the Streamlit app
    st.write("Number of students in each performance category:")
    st.write(performance_counts)

    # Plot a bar chart of the performance categories
    st.header("Performance Level Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    performance_counts.plot(kind='bar', alpha=0.7, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Performance Level')
    ax.set_ylabel('Number of Students')
    ax.set_title('Number of Students in Each Performance Level')
    st.pyplot(fig)

    # Map Gender to Labels (1 = Female, 2 = Male)
    df["Gender_Label"] = df["GENDER"].map({1: "Female", 2: "Male"})

    # Group by Gender and Performance Level, then count
    gender_performance_counts = df.groupby(['Gender_Label', 'Prediction_Label']).size().unstack().fillna(0)

    # Show the counts in the Streamlit app
    st.write("Number of male and female students in each performance category:")
    st.write(gender_performance_counts)

    # Plot a stacked bar chart of the performance categories by gender
    st.header("Performance Level Distribution by Gender")
    gender_performance_counts.plot(kind='bar', stacked=True, alpha=0.7, color=['skyblue', 'lightcoral', 'red'],
                                   figsize=(8, 6))
    plt.xlabel('Gender')
    plt.ylabel('Number of Students')
    plt.title('Number of Students in Each Performance Level (by Gender)')
    st.pyplot(plt)

# Add a sidebar for additional options
st.sidebar.header("About")
st.sidebar.write("This app predicts student performance based on key factors, including Gender and Class Attendance, which have the highest influence on academic outcomes. Additionally, being actively engaged in classes and regular reading habits appear to positively impact performance. The presence of siblings also shows some influence, while having a job seems to negatively affect performance. By analyzing these factors, the app provides insights into what contributes most to student success.")
