import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- STEP 1: LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_and_prepare_data():
    # Loading the dataset as per your Step 2
    df = pd.read_csv('StudentPerformance.csv')
    
    # Handling missing values (Step 3)
    df['Study Hours per Week'] = df['Study Hours per Week'].fillna(df['Study Hours per Week'].median())
    df['Attendance Rate'] = df['Attendance Rate'].fillna(df['Attendance Rate'].median())
    df['Previous Grades'] = df['Previous Grades'].fillna(df['Previous Grades'].median())
    df.dropna(subset=['Passed'], inplace=True)
    
    # Data Encoding (Step 4)[cite: 1]
    df['Passed'] = df['Passed'].map({'Yes': 1, 'No': 0})
    
    # Feature and Target Selection (Step 5)[cite: 1]
    X = df[['Study Hours per Week', 'Attendance Rate', 'Previous Grades']]
    y = df['Passed']
    
    # Train-Test Split (Step 6)[cite: 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and training the model (Step 7)[cite: 1]
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    
    # Calculate accuracy for the sidebar (Step 9)[cite: 1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, X.columns, acc

model, feature_names, accuracy = load_and_prepare_data()

# --- STEP 2: SIDEBAR METRICS ---
st.sidebar.title("📊 Model Metrics")
st.sidebar.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
st.sidebar.write("The accuracy is calculated on a 20% test split.")

# --- STEP 3: USER INTERFACE ---
st.title("🎓 Student Performance Predictor")
st.write("Enter the student details to predict their success.")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input("Study Hours/Week", min_value=0.0, max_value=168.0, value=20.0)
with col2:
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
with col3:
    prev_grades = st.number_input("Previous Grades", min_value=0, max_value=100, value=75)

# --- STEP 4: PREDICTION ---
if st.button("Predict Result"):
    # Formatting input for prediction
    input_data = [[study_hours, attendance, prev_grades]]
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("✅ **Result: Passed**")
    else:
        st.error("❌ **Result: Failed**")

# --- STEP 5: VISUALIZATION ---
st.divider()
st.subheader("🌳 Decision Tree Visualization")
st.write("This tree illustrates the logical thresholds used for prediction.")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, 
          feature_names=list(feature_names), 
          class_names=['Failed', 'Passed'], 
          filled=True, 
          rounded=True, 
          max_depth=3) # Depth limit for visual clarity[cite: 1]
st.pyplot(fig)
