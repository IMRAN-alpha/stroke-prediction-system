
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# ==========================================
# APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Stroke Prediction | AI Healthcare",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "healthcare-dataset-stroke-data.csv"


# ==========================================
# FUTURISTIC UI DESIGN
# ==========================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #102c46 0%, transparent 35%),
        radial-gradient(circle at 90% 80%, #073b46 0%, transparent 35%),
        #071321;
    color: #e8f4ff;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* GLOBAL TEXT */

html, body, [class*="css"] {
    color: #e8f4ff;
}

h1, h2, h3, h4 {
    color: #eaf7ff !important;
}

p, label, span {
    color: #c1d5e8;
}

/* HERO SECTION */

.hero {
    position: relative;
    overflow: hidden;
    padding: 35px 30px;
    border: 1px solid rgba(0, 220, 255, 0.35);
    border-radius: 22px;
    background: linear-gradient(
        120deg,
        rgba(12, 39, 64, 0.96),
        rgba(5, 74, 88, 0.85)
    );
    box-shadow: 0 0 30px rgba(0, 200, 255, 0.12);
    margin-bottom: 25px;
    animation: fadeUp 0.8s ease-out;
}

.hero h1 {
    font-size: clamp(25px, 4vw, 42px);
    font-weight: 800;
    letter-spacing: 1px;
    color: #ffffff !important;
    margin-bottom: 8px;
}

.hero p {
    color: #a9e9f5 !important;
    font-size: 15px;
}

.hero-tag {
    display: inline-block;
    color: #52eaff;
    background: rgba(0, 220, 255, 0.10);
    border: 1px solid rgba(0, 220, 255, 0.35);
    border-radius: 30px;
    padding: 6px 14px;
    font-size: 12px;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(17, 43, 67, 0.96),
        rgba(9, 29, 48, 0.96)
    );
    border: 1px solid rgba(0, 220, 255, 0.23);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 22px rgba(0, 0, 0, 0.20);
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border-color: #00d9ff;
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #8eb8d2 !important;
}

[data-testid="stMetricValue"] {
    color: #55eaff !important;
    font-weight: 800;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #00d9ff;
    background: linear-gradient(110deg, #087d9b, #07516e);
    color: white !important;
    font-weight: 700;
    padding: 12px 18px;
    transition: all 0.3s ease;
    box-shadow: 0 0 12px rgba(0, 217, 255, 0.12);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(110deg, #00a9c9, #087d9b);
    border-color: #8af3ff;
    box-shadow: 0 0 22px rgba(0, 217, 255, 0.35);
    transform: translateY(-2px);
}

.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: scale(0.98);
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(9, 29, 48, 0.75);
    border: 1px solid rgba(0, 217, 255, 0.16);
    border-radius: 14px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #a8c5d8;
    padding: 10px 15px;
    transition: all 0.25s ease;
}

.stTabs [aria-selected="true"] {
    background: #075c78 !important;
    color: white !important;
    border-bottom: 2px solid #00e5ff !important;
}

/* INPUT FIELDS */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: #10283d !important;
    color: white !important;
    border-color: rgba(0, 217, 255, 0.25) !important;
    border-radius: 10px !important;
}

[data-baseweb="popover"] {
    background: #10283d !important;
}

/* FORM CONTAINER */

[data-testid="stForm"] {
    background: rgba(12, 35, 55, 0.85);
    border: 1px solid rgba(0, 217, 255, 0.24);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.16);
}

/* DATAFRAMES */

[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 217, 255, 0.20);
    border-radius: 12px;
    overflow: hidden;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2034, #071321);
    border-right: 1px solid rgba(0, 217, 255, 0.20);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #55eaff !important;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* CUSTOM SECTION */

.section-heading {
    color: #5cecff;
    font-size: 21px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
}

.prediction-panel {
    background: linear-gradient(
        135deg,
        rgba(10, 51, 72, 0.95),
        rgba(12, 31, 53, 0.95)
    );
    border: 1px solid rgba(0, 217, 255, 0.32);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 0 25px rgba(0, 217, 255, 0.08);
    animation: fadeUp 0.6s ease-out;
}

.footer {
    text-align: center;
    padding: 22px;
    border-top: 1px solid rgba(0, 217, 255, 0.20);
    color: #83a9c1;
    font-size: 13px;
}

/* ANIMATIONS */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.10);
    }
    50% {
        box-shadow: 0 0 22px rgba(0, 217, 255, 0.23);
    }
}

.hero-tag {
    animation: glow 3s infinite ease-in-out;
}

/* REDUCED MOTION ACCESSIBILITY */

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* MOBILE RESPONSIVENESS */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 24px 18px;
    }

    .hero h1 {
        font-size: 25px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px;
        font-size: 12px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["stroke_label"] = df["stroke"].map({
        0: "No Stroke",
        1: "Stroke"
    })
    return df


if not DATA_FILE.exists():
    st.error(
        "Dataset file not found. Keep "
        "healthcare-dataset-stroke-data.csv beside app.py."
    )
    st.stop()

df = load_data()


# ==========================================
# HERO HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">ARTIFICIAL INTELLIGENCE • HEALTHCARE</div>
    <h1>🧠 STROKE PREDICTION SYSTEM</h1>
    <p>
        Intelligent Data Exploration & Machine Learning Dashboard
    </p>
    <p>
        Python for Data Science Internship · Data Visualization · MD IMRAN
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ⚡ AI CONTROL CENTER")
    st.caption("Dashboard configuration")

    gender_filter = st.multiselect(
        "Gender",
        sorted(df["gender"].dropna().unique()),
        default=sorted(df["gender"].dropna().unique())
    )

    residence_filter = st.multiselect(
        "Residence",
        sorted(df["Residence_type"].dropna().unique()),
        default=sorted(df["Residence_type"].dropna().unique())
    )

    stroke_filter = st.multiselect(
        "Target Class",
        ["No Stroke", "Stroke"],
        default=["No Stroke", "Stroke"]
    )

    st.divider()

    st.markdown("### 👨‍💻 DEVELOPER")
    st.write("MD IMRAN")

    st.markdown(
        "[🔗 Connect on LinkedIn](https://www.linkedin.com/in/imran-alpha)"
    )

    st.caption(
        "Educational demonstration only. "
        "Not a medical diagnostic system."
    )


# ==========================================
# FILTER DATA
# ==========================================

filtered = df[
    df["gender"].isin(gender_filter)
    & df["Residence_type"].isin(residence_filter)
    & df["stroke_label"].isin(stroke_filter)
]


# ==========================================
# TOP METRICS
# ==========================================

st.markdown(
    '<div class="section-heading">◈ SYSTEM OVERVIEW</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("TOTAL RECORDS", f"{len(df):,}")
c2.metric("TOTAL FEATURES", "12")
c3.metric("STROKE RECORDS", f"{int(df.stroke.sum()):,}")
c4.metric("MISSING BMI", f"{int(df.bmi.isna().sum()):,}")

st.info(
    "Dashboard filters affect visual analysis. "
    "Overview metrics describe the complete original dataset."
)


# ==========================================
# EDA TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "🎯 Target Analysis",
    "❤️ Health & Lifestyle",
    "🤖 Model Results"
])


# ==========================================
# TAB 1: DATASET OVERVIEW
# ==========================================

with tab1:

    st.markdown(
        '<div class="section-heading">DATASET EXPLORATION</div>',
        unsafe_allow_html=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered.drop(columns=["stroke_label"]).head(20),
        use_container_width=True
    )

    st.subheader("Numerical Summary")

    st.dataframe(
        filtered[
            ["age", "avg_glucose_level", "bmi"]
        ].describe().round(2),
        use_container_width=True
    )

    a, b = st.columns(2)

    with a:

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#10283d")
        ax.set_facecolor("#10283d")

        ax.hist(
            filtered["age"].dropna(),
            bins=25,
            edgecolor="#071321"
        )

        ax.set_title("Age Distribution", color="white")
        ax.set_xlabel("Age", color="white")
        ax.set_ylabel("Number of Records", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#42627c")

        st.pyplot(fig)
        plt.close(fig)

    with b:

        counts = filtered["Residence_type"].value_counts()

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#10283d")
        ax.set_facecolor("#10283d")

        counts.plot(kind="bar", ax=ax)

        ax.set_title("Residence Type Distribution", color="white")
        ax.set_xlabel("Residence", color="white")
        ax.set_ylabel("Records", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#42627c")

        st.pyplot(fig)
        plt.close(fig)


# ==========================================
# TAB 2: TARGET ANALYSIS
# ==========================================

with tab2:

    st.markdown(
        '<div class="section-heading">TARGET CLASS ANALYSIS</div>',
        unsafe_allow_html=True
    )

    counts = (
        filtered["stroke_label"]
        .value_counts()
        .reindex(["No Stroke", "Stroke"])
        .fillna(0)
    )

    a, b = st.columns(2)

    with a:

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#10283d")
        ax.set_facecolor("#10283d")

        counts.plot(kind="bar", ax=ax)

        ax.set_title("Target Class Count", color="white")
        ax.set_xlabel("Target", color="white")
        ax.set_ylabel("Records", color="white")
        ax.tick_params(colors="white")

        st.pyplot(fig)
        plt.close(fig)

    with b:

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#10283d")
        ax.set_facecolor("#10283d")

        counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")
        ax.set_title("Target Class Share", color="white")

        st.pyplot(fig)
        plt.close(fig)

    st.info(
        f"In the full dataset, stroke-positive records are "
        f"{df.stroke.mean() * 100:.2f}% of all records. "
        "The target is imbalanced."
    )


# ==========================================
# TAB 3: HEALTH & LIFESTYLE
# ==========================================

with tab3:

    st.markdown(
        '<div class="section-heading">HEALTH & LIFESTYLE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        c = filtered["smoking_status"].value_counts()

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#10283d")
        ax.set_facecolor("#10283d")

        c.plot(kind="bar", ax=ax)

        ax.set_title("Smoking Status Distribution", color="white")
        ax.set_xlabel("Smoking Status", color="white")
        ax.set_ylabel("Records", color="white")
        ax.tick_params(colors="white", rotation=25)

        st.pyplot(fig)
        plt.close(fig)

    with b:

        c = filtered.groupby(
            "stroke_label"
        )["avg_glucose_level"].mean()

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("#10283d")
        ax.set_facecolor("#10283d")

        c.plot(kind="bar", ax=ax)

        ax.set_title("Average Glucose by Target Class", color="white")
        ax.set_xlabel("Target", color="white")
        ax.set_ylabel("Average Glucose Level", color="white")
        ax.tick_params(colors="white")

        st.pyplot(fig)
        plt.close(fig)

    st.write(
        "BMI missing values:",
        int(df.bmi.isna().sum()),
        "of",
        len(df),
        "records"
    )


# ==========================================
# TAB 4: MODEL RESULTS
# ==========================================

with tab4:

    st.markdown(
        '<div class="section-heading">MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Values reported in the internship presentation. "
        "These are separate from the live demo model."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Logistic Regression Accuracy", "94.59%")
    m2.metric("Average Precision", "0.23")
    m3.metric("Class-1 Recall", "0.46")

    st.warning(
        "Accuracy alone can be misleading with an imbalanced target. "
        "Interpret recall and Average Precision alongside accuracy. "
        "These reported results are not a clinical validation."
    )


# ==========================================
# AI STROKE PREDICTION
# ==========================================

st.divider()

st.markdown("""
<div class="prediction-panel">
    <div class="hero-tag">AI PREDICTION MODULE</div>
    <h2>🧠 Stroke Prediction</h2>
    <p>
        Enter the requested information to see the output
        of the educational machine-learning demo model.
    </p>
</div>
""", unsafe_allow_html=True)


NUMERIC = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

CATEGORICAL = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]


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


model = get_stroke_model()

# ═══════════════════════════
# TAB 2 — ABOUT STROKE
# ═══════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>🧠 What is a Stroke?</div>
            <div style='font-size:14px; color:#444; line-height:1.8;'>
                A stroke occurs when blood flow to part of the brain is blocked or a blood vessel bursts.
                Brain cells begin to die within minutes without oxygen. It is a medical emergency
                requiring immediate attention.
            </div>
        </div>

        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>⚡ Warning Signs (FAST)</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                <b style='color:#e05c1a;'>F</b> — Face drooping on one side<br>
                <b style='color:#e05c1a;'>A</b> — Arm weakness or numbness<br>
                <b style='color:#e05c1a;'>S</b> — Speech difficulty or slurred<br>
                <b style='color:#e05c1a;'>T</b> — Time to call emergency services
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>📊 Global Statistics</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                🌍 13 million strokes occur globally each year<br>
                💔 5.5 million deaths annually from stroke<br>
                ♿ Leading cause of long-term disability<br>
                ⏰ Every 40 seconds someone has a stroke in the US<br>
                🎯 80% of strokes are preventable
            </div>
        </div>

        <div class='stroke-info'>
            <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>🔬 Types of Stroke</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                <b>Ischemic (87%)</b> — Blood clot blocks artery<br>
                <b>Hemorrhagic (13%)</b> — Blood vessel ruptures<br>
                <b>TIA</b> — Temporary blockage (mini-stroke)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='stroke-info'>
        <div style='font-size:18px; font-weight:700; color:#e05c1a; margin-bottom:12px;'>🤖 About This Model</div>
        <div style='font-size:14px; color:#444; line-height:1.8;'>
            This system uses <b>Logistic Regression</b> — a well-established machine learning algorithm
            chosen for its high interpretability in healthcare settings. The model was trained on
            5,110 real patient records from Kaggle, with SMOTE applied to handle class imbalance.
            Preprocessing included BMI imputation, one-hot encoding, and StandardScaler normalization.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════
# TAB 3 — RISK FACTORS
# ═══════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='font-size:16px; font-weight:700; color:#e05c1a; margin-bottom:14px;'>⚠️ High Risk Factors</div>
        """, unsafe_allow_html=True)

        factors_high = [
            ("Hypertension", "High blood pressure damages arteries increasing stroke risk significantly."),
            ("High Glucose Level", "Blood sugar above 140 mg/dL increases stroke risk — diabetes is a major factor."),
            ("Old Age (60+)", "Risk doubles every 10 years after age 55."),
            ("Heart Disease", "Irregular heartbeat and other heart conditions increase clot formation."),
            ("Smoking", "Smoking doubles stroke risk by damaging blood vessels and increasing clot formation."),
            ("High BMI", "Obesity (BMI > 30) leads to hypertension and diabetes — both stroke risk factors."),
        ]

        for title, desc in factors_high:
            st.markdown(f"""
            <div class='risk-card risk-high'>
                <span class='risk-tag tag-high'>HIGH RISK</span>
                <div style='font-size:14px; font-weight:600; color:#1a1a1a;'>{title}</div>
                <div style='font-size:13px; color:#666; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='font-size:16px; font-weight:700; color:#16a34a; margin-bottom:14px;'>✅ Protective Factors</div>
        """, unsafe_allow_html=True)

        factors_low = [
            ("Normal Blood Pressure", "Keeping BP below 120/80 mmHg significantly reduces stroke risk."),
            ("Healthy Glucose Level", "Maintaining blood sugar below 100 mg/dL protects brain vessels."),
            ("Young Age", "People under 40 have significantly lower stroke risk."),
            ("Never Smoked", "Non-smokers have half the stroke risk of smokers."),
            ("Healthy BMI (18.5-24.9)", "Maintaining healthy weight reduces hypertension and diabetes risk."),
            ("Active Lifestyle", "Regular exercise reduces stroke risk by up to 27%."),
        ]

        for title, desc in factors_low:
            st.markdown(f"""
            <div class='risk-card risk-low'>
                <span class='risk-tag tag-low'>PROTECTIVE</span>
                <div style='font-size:14px; font-weight:600; color:#1a1a1a;'>{title}</div>
                <div style='font-size:13px; color:#666; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# PREDICTION FORM
# ==========================================

with st.form("stroke_prediction_form"):

    st.markdown("### ⚙️ PATIENT INFORMATION")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1]
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1]
        )

        married = st.selectbox(
            "Ever Married",
            ["Yes", "No"]
        )

    with col2:

        glucose = st.number_input(
            "Average Glucose Level",
            min_value=1.0,
            max_value=300.0,
            value=100.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=25.0
        )

        work = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )

        residence = st.selectbox(
            "Residence",
            ["Urban", "Rural"]
        )

        smoking = st.selectbox(
            "Smoking Status",
            [
                "formerly smoked",
                "never smoked",
                "smokes",
                "Unknown"
            ]
        )

    submitted = st.form_submit_button(
        "⚡ RUN AI PREDICTION",
        use_container_width=True
    )


# ==========================================
# PREDICTION OUTPUT
# ==========================================

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

    probability_percent = probability * 100

    # Risk Indicator (Illustrative thresholds)

    if probability_percent < 30:
        risk_label = "LOWER PREDICTED PROBABILITY"
        risk_color = "#22c55e"
        risk_icon = "🟢"

    elif probability_percent < 60:
        risk_label = "MODERATE PREDICTED PROBABILITY"
        risk_color = "#eab308"
        risk_icon = "🟡"

    else:
        risk_label = "HIGHER PREDICTED PROBABILITY"
        risk_color = "#ef4444"
        risk_icon = "🔴"

    st.markdown("### ◈ PREDICTION OUTPUT")

    # Probability
    st.metric(
        "Model Estimated Probability",
        f"{probability_percent:.2f}%"
    )

    # Risk Indicator
    st.markdown(
        f"### {risk_icon} {risk_label}"
    )

    # Probability Progress Bar
    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    # Model Classification
    if prediction == 1:
        st.warning(
            "The demo model classified this input "
            "in the positive class."
        )
    else:
        st.success(
            "The demo model classified this input "
            "in the negative class."
        )

    st.caption(
        "The colour categories are illustrative probability "
        "bands, not clinically validated risk levels. "
        "This educational model is not a medical diagnosis."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>🧠 STROKE PREDICTION SYSTEM</strong>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "Python · Pandas · Matplotlib · Scikit-learn · Streamlit"
)

st.markdown(
    "Data Exploration • Visualization • Machine Learning"
)

st.markdown("Developed by **MD IMRAN**")

st.link_button(
    "🔗 Connect with me on LinkedIn",
    "https://www.linkedin.com/in/imran-alpha"
)
