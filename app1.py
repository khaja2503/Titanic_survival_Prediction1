import streamlit as st
import pandas as pd
import joblib
import base64
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="ðŸš¢",
    layout="wide"
)


# ---------------------------------------------------------
# Background image
# ---------------------------------------------------------

def set_background(image_file):
    image_path = Path(__file__).resolve().parent / image_file

    if not image_path.exists():
        st.warning("Background image not found. Using a default dark theme instead.")
        css = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #071827, #102a3d 45%, #1e3d5a);
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #f5f7fb;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        return

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    css = f"""
        <style>

    .stApp {{
        background-image:
        linear-gradient(
            rgba(6, 18, 32, 0.72),
            rgba(12, 28, 46, 0.68)
        ),
        url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f5f7fb;
    }}

    .main-title {{
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 0.5px;
        color: #f8f9fb;
        text-shadow: 0 3px 18px rgba(10, 20, 32, 0.55);
        margin-top: 10px;
        margin-bottom: 8px;
    }}

    .subtitle {{
        font-size: 20px;
        text-align: center;
        color: #dfe9f7;
        font-weight: 500;
        letter-spacing: 0.4px;
        margin-bottom: 30px;
        opacity: 0.96;
    }}

    .prediction-box {{
        padding: 26px 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(13, 32, 51, 0.9), rgba(21, 49, 72, 0.88));
        border: 1px solid rgba(164, 196, 255, 0.32);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        text-align: center;
        margin-top: 25px;
        color: #f2f7ff;
    }}

    .stSubheader {{
        color: #edf3ff !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        text-shadow: 0 2px 14px rgba(0, 0, 0, 0.35);
    }}

    div[data-testid="stHorizontalBlock"] > div {{
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 14px;
        backdrop-filter: blur(2px);
    }}

    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(128, 155, 189, 0.4) !important;
        box-shadow: inset 0 1px 2px rgba(16, 25, 42, 0.06);
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #0b2a4a, #1a4d75);
        color: #ffffff;
        border: 1px solid rgba(165, 208, 255, 0.5);
        border-radius: 12px;
        font-weight: 700;
        padding: 0.7rem 1.4rem;
        box-shadow: 0 8px 22px rgba(15, 47, 72, 0.35);
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #123d5f, #2367a8);
        color: #f6fbff;
    }}

    .element-container .stMarkdown h3,
    .element-container .stMarkdown p,
    .element-container .stMarkdown div {{
        color: #edf3ff;
    }}

    </style>
    """

    st.markdown(
        css,
        unsafe_allow_html=True
    )


# Set background image
set_background("Images/Titanic image.jpeg")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "titanic_best_model.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
DATA_PATH = ROOT_DIR / "Data" / "Titanic-Dataset.csv"


def train_model():
    df = pd.read_csv(DATA_PATH)

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = df["Survived"]

    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource
def load_model():
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        model_path = MODEL_DIR / metadata["model_path"]
        model_type = metadata["model_type"]
        if model_type == "sklearn_pipeline":
            return model_type, joblib.load(model_path), None, metadata["model_name"]

        if model_type in {"keras_ann", "keras_cnn"}:
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path, compile=False)
            preprocessor = joblib.load(MODEL_DIR / metadata["preprocessor_path"])
            return model_type, model, preprocessor, metadata["model_name"]

        raise ValueError(f"Unsupported model type: {model_type}")

    for model_path in (MODEL_PATH, ROOT_DIR / "titanic_best_model.pkl"):
        try:
            model = joblib.load(model_path)
            if model is not None:
                return "sklearn_pipeline", model, None, "Random Forest"
        except Exception:
            continue

    st.warning("No usable saved model was found. Rebuilding it from the dataset...")
    return "sklearn_pipeline", train_model(), None, "Random Forest"


model_type, model, preprocessor, model_name = load_model()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Titanic Survival Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Survival Prediction</div>',
    unsafe_allow_html=True
)
st.caption(f"Selected model: {model_name}")

if MODEL_COMPARISON_PATH.exists():
    with st.expander("Model comparison"):
        comparison = pd.read_csv(MODEL_COMPARISON_PATH)
        st.dataframe(comparison, hide_index=True, width="stretch")


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.subheader("Passenger Information")


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Column 1
# ---------------------------------------------------------

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )


# ---------------------------------------------------------
# Column 2
# ---------------------------------------------------------

with col2:

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "Predict Survival",
    use_container_width=True
):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })


    if model_type.startswith("keras_"):
        transformed_input = preprocessor.transform(input_data)
        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()
        if model_type == "keras_cnn":
            transformed_input = transformed_input.reshape(
                transformed_input.shape[0], transformed_input.shape[1], 1
            )
        survival_probability = float(
            model.predict(transformed_input, verbose=0).reshape(-1)[0]
        )
        prediction = int(survival_probability >= 0.5)
        death_probability = 1.0 - survival_probability
    else:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        death_probability = probability[0]
        survival_probability = probability[1]


    # -----------------------------------------------------
    # Display prediction
    # -----------------------------------------------------

    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.success(
            "Passenger is predicted to SURVIVE."
        )

    else:

        st.error(
            "Passenger is predicted NOT TO SURVIVE."
        )


    # Display probabilities
    st.write(
        f"Survival Probability: "
        f"{survival_probability:.2%}"
    )

    st.write(
        f"Not Survival Probability: "
        f"{death_probability:.2%}"
    )


    # Display progress bar
    st.progress(
        float(survival_probability)
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Titanic Survival Prediction | "
    "Machine Learning + Streamlit"
)