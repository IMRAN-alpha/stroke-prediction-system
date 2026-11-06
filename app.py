import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / 'healthcare-dataset-stroke-data.csv'

st.set_page_config(page_title='Stroke Prediction | EDA Dashboard', page_icon='🧠', layout='wide')
st.markdown('''<style>
.stApp{background:#f4f7fb}.block-container{padding-top:1.5rem}
[data-testid="stMetric"]{background:white;padding:16px;border:1px solid #e3eaf2;border-radius:12px}
.hero{padding:24px;border-radius:16px;background:linear-gradient(120deg,#12304a,#176b75);color:white;margin-bottom:18px}
.hero h1{color:white;margin:0}.hero p{color:#d8e8ee;margin:8px 0 0}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['stroke_label'] = df['stroke'].map({0:'No Stroke', 1:'Stroke'})
    return df

st.markdown('''<div class="hero"><h1>🧠 Stroke Prediction — Data Exploration Dashboard</h1>
<p>Python for Data Science Internship · Exploratory Data Analysis & Data Visualization · MD IMRAN</p></div>''', unsafe_allow_html=True)

if not DATA_FILE.exists():
    st.error('Dataset file not found. Keep healthcare-dataset-stroke-data.csv beside app.py.')
    st.stop()

df = load_data()

with st.sidebar:
    st.header('Dashboard Controls')
    st.caption('Explore the supplied healthcare dataset.')
    gender_filter = st.multiselect('Gender', sorted(df['gender'].dropna().unique()), default=sorted(df['gender'].dropna().unique()))
    residence_filter = st.multiselect('Residence', sorted(df['Residence_type'].dropna().unique()), default=sorted(df['Residence_type'].dropna().unique()))
    stroke_filter = st.multiselect('Target class', ['No Stroke','Stroke'], default=['No Stroke','Stroke'])
    st.divider()
    st.markdown('**Developer**  \nMD IMRAN')
    st.markdown('[LinkedIn](https://www.linkedin.com/in/imran-alpha)')
    st.caption('Educational data-analysis demonstration. Not a diagnostic tool.')

filtered = df[df['gender'].isin(gender_filter) & df['Residence_type'].isin(residence_filter) & df['stroke_label'].isin(stroke_filter)]

c1,c2,c3,c4 = st.columns(4)
c1.metric('Total Records', f'{len(df):,}')
c2.metric('Features', '12')
c3.metric('Stroke Records', f'{int(df.stroke.sum()):,}')
c4.metric('Missing BMI Values', f'{int(df.bmi.isna().sum()):,}')

st.info('The filters below affect the visual analysis. Dataset overview metrics above describe the complete original dataset.')

tab1,tab2,tab3,tab4 = st.tabs(['Dataset Overview','Target Analysis','Health & Lifestyle','Model Results'])
with tab1:
    st.subheader('Dataset preview')
    st.dataframe(filtered.drop(columns=['stroke_label']).head(20), use_container_width=True)
    st.subheader('Numerical summary')
    st.dataframe(filtered[['age','avg_glucose_level','bmi']].describe().round(2), use_container_width=True)
    
# ==============================
# AI STROKE PREDICTION FEATURE
# ==============================

st.divider()
st.header("🧠 AI Stroke Risk Prediction")
st.write("Enter patient information for an educational ML prediction.")

@st.cache_resource
def get_stroke_model():
    X = df[NUMERIC + CATEGORICAL]
    y = df["stroke"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC),
        ("cat", categorical_pipeline, CATEGORICAL)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model


NUMERIC = [
    "age", "hypertension", "heart_disease",
    "avg_glucose_level", "bmi"
]

CATEGORICAL = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status"
]

model = get_stroke_model()

with st.form("stroke_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        married = st.selectbox("Ever Married", ["Yes", "No"])

    with col2:
        glucose = st.number_input("Average Glucose Level", 1.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
        work = st.selectbox("Work Type", [
            "Private", "Self-employed", "Govt_job",
            "children", "Never_worked"
        ])
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        smoking = st.selectbox("Smoking Status", [
            "formerly smoked", "never smoked",
            "smokes", "Unknown"
        ])

    submitted = st.form_submit_button(
        "Predict Stroke Risk",
        use_container_width=True
    )

if submitted:
    patient = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "gender": gender,
        "ever_married": married,
        "work_type": work,
        "Residence_type": residence,
        "smoking_status": smoking
    }])

    probability = model.predict_proba(patient)[0][
        list(model.classes_).index(1)
    ]

    prediction = model.predict(patient)[0]

    st.subheader("Prediction Result")

    st.metric("Model Estimated Probability", f"{probability * 100:.2f}%")

    if prediction == 1:
        st.warning("Model classified this input in the positive class.")
    else:
        st.success("Model classified this input in the negative class.")

    st.caption(
        "Educational machine-learning output only. "
        "This is not a medical diagnosis or validated clinical risk assessment."
)
        
    a,b=st.columns(2)
    with a:
        fig,ax=plt.subplots(); ax.hist(filtered['age'].dropna(),bins=25); ax.set_title('Age Distribution'); ax.set_xlabel('Age'); ax.set_ylabel('Number of Records'); st.pyplot(fig); plt.close(fig)
    with b:
        counts=filtered['Residence_type'].value_counts()
        fig,ax=plt.subplots(); counts.plot(kind='bar',ax=ax); ax.set_title('Residence Type Distribution'); ax.set_xlabel('Residence'); ax.set_ylabel('Records'); st.pyplot(fig); plt.close(fig)
with tab2:
    st.subheader('Stroke vs non-stroke class distribution')
    counts=filtered['stroke_label'].value_counts().reindex(['No Stroke','Stroke']).fillna(0)
    a,b=st.columns(2)
    with a:
        fig,ax=plt.subplots(); counts.plot(kind='bar',ax=ax); ax.set_title('Target Class Count'); ax.set_xlabel('Target'); ax.set_ylabel('Records'); st.pyplot(fig); plt.close(fig)
    with b:
        fig,ax=plt.subplots(); counts.plot(kind='pie',autopct='%1.1f%%',ax=ax); ax.set_ylabel(''); ax.set_title('Target Class Share'); st.pyplot(fig); plt.close(fig)
    st.caption(f"In the full dataset, stroke-positive records are {df.stroke.mean()*100:.2f}% of all records. The target is imbalanced.")
with tab3:
    st.subheader('Health and lifestyle patterns')
    a,b=st.columns(2)
    with a:
        c=filtered['smoking_status'].value_counts()
        fig,ax=plt.subplots(); c.plot(kind='bar',ax=ax); ax.set_title('Smoking Status Distribution'); ax.set_xlabel('Smoking Status'); ax.set_ylabel('Records'); plt.xticks(rotation=25,ha='right'); st.pyplot(fig); plt.close(fig)
    with b:
        c=filtered.groupby('stroke_label')['avg_glucose_level'].mean()
        fig,ax=plt.subplots(); c.plot(kind='bar',ax=ax); ax.set_title('Average Glucose by Target Class'); ax.set_xlabel('Target'); ax.set_ylabel('Average Glucose Level'); st.pyplot(fig); plt.close(fig)
    st.write('BMI missing values:', int(df.bmi.isna().sum()), 'of', len(df), 'records')
with tab4:
    st.subheader('Model comparison — values reported in the internship presentation')
    st.caption('These are presentation-reported results, displayed for explanation; this dashboard does not run the trained model.')
    m1,m2,m3=st.columns(3)
    m1.metric('Logistic Regression Accuracy','94.59%')
    m2.metric('Average Precision','0.23')
    m3.metric('Class-1 Recall','0.46')
    st.warning('Accuracy alone can be misleading with an imbalanced target. Interpret recall and Average Precision alongside accuracy. This is an educational prototype, not a medical diagnosis or clinical decision tool.')

st.divider()
st.caption('Libraries used in the project presentation: Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn')
st.caption('Dataset: healthcare-dataset-stroke-data.csv · 5,110 rows · 12 columns')
