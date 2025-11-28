# webapp/app.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import joblib
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
import spacy
from collections import Counter
from utils.constants import CATEGORY_MAPPING  # import category mapping


# Add src to the path so we can import preprocess_text
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from preprocess import preprocess_text  # Import the preprocess_text function


# function to load NLP models and sentiment analyzer
@st.cache_resource
def get_nlp_models():
    try:
        nltk.data.find("vader_lexicon") # check if vader_lexicon is already downloaded
    except LookupError:
        nltk.download("vader_lexicon", quiet=True) # download if not found

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Load the small English model from spaCy
    analyzer = SentimentIntensityAnalyzer() # Initialize VADER sentiment analyzer

    return nlp, analyzer

# function to load the trained XGBoost model
@st.cache_resource
def get_xgb_model():
    # 1. Create a Path object to your saved model (relative to repo root)
    model_dir = Path(__file__).resolve().parents[1] / "model"
    model_path = model_dir / "review_classifier.pkl"

    # 2. Load the model using joblib
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None

    try:
        best_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

    return {"best_model": best_model}





# ---------------------------
# POS TAGS YOU SELECTED DURING TRAINING
# ---------------------------
POS_WHITELIST = {"VERB", "NOUN", "ADV", "ADJ", "PRON", "DET"}

# ---------------------------
# POS COUNTING
# ---------------------------
def pos_counts(text, nlp):
    """
    Count POS tags in a text using the whitelist.
    Must pass in `nlp` so the web app can load the model once.
    """
    doc = nlp(text)
    return Counter(token.pos_ for token in doc if token.pos_ in POS_WHITELIST)

def add_pos_features(df, nlp):
    """
    Add POS count columns to a DF.
    """
    pos_series = df["cleaned_text"].apply(lambda txt: pos_counts(txt, nlp))
    pos_df = pd.DataFrame(list(pos_series)).fillna(0)
    pos_df.index = df.index
    return pd.concat([df, pos_df], axis=1)

# ---------------------------
# MAIN FEATURE EXTRACTION FUNCTION
# ---------------------------
def extract_features(text, include_pos=False):
    """
    Extract all features needed for inference.
    This mirrors your training feature process.
    """

    cleaned_text = preprocess_text(text)
    orig_text = str(text)  # keep original text for uppercase ratio

    # Load NLP + sentiment analyzer (loaded ONCE per request)
    nlp, analyzer = get_nlp_models()

    df = pd.DataFrame([{
        "cleaned_text": cleaned_text,
        "char_length": len(cleaned_text),
        "word_count": len(cleaned_text.split()),
        "punctuation_ct": sum(1 for c in cleaned_text if c in string.punctuation),
        "avg_word_len": sum(len(w) for w in cleaned_text.split()) / max(1, len(cleaned_text.split())),
        "uppercase_ratio": sum(1 for w in orig_text.split() if w.isupper()) / max(1, len(orig_text.split()))
    }])

    if include_pos:
        df = add_pos_features(df, nlp)

    # Category features
    # important_categories = [
    #     "category_Kindle_Store_5", "category_Clothing_Shoes_and_Jewelry_5",
    #     "category_Pet_Supplies_5", "category_Books_5", "category_Movies_and_TV_5"
    # ]
    # if categories is None:
    #     categories = {}
    # for cat in important_categories:
    #     df[cat] = categories.get(cat, 0)

    return df

# ---------------------------
# PREPARE FEATURES FOR PREDICTION  
# ---------------------------

def prepare_features_for_prediction(text, category="unknown"):
    """
    Prepares a single review's features for prediction.
    Ensures all expected columns exist and are in the correct order.
    """

    # 1️ Decide if POS features were used in your final model
    include_pos = True  

    # 2️ Call extract_features to create a DataFrame
    df = extract_features(text, include_pos=include_pos)

    # 3️ Load the feature names your model expects
    # Update path to your saved feature_names.json
    with open("../model/feature_names.json", "r") as f:
        feature_data = json.load(f)
    
    # 4️ Initialize all category columns to 0
    category_columns = [col for col in feature_data if col.startswith("category_")]
    for col in category_columns:
        df[col] = 0
    
    # 5️ Set the appropriate category column to 1 based on the input category
    category_col_name = f"category_{category}_5"
    if category_col_name in category_columns:
        df[category_col_name] = 1
    # If the input category is unknown or not in training, all remain 0

    # 6️⃣ Ensure all expected features are present
    for feat in feature_data:
        if feat not in df.columns:
            df[feat] = 0.0  # Add missing features with default value

    # 7️⃣ Return features in the exact order the model expects
    df = df[feature_data]

    return df

def xgb_predict(text, model, category="unknown"):
    """
    Makes a prediction using a trained XGBoost model.
    
    Args:
        text (str): Review text
        model: Trained XGBoost model
        category (str): Category name
        rating (float): Review rating
    
    Returns:
        label (str): "Human" or "AI"
        confidence (float): Probability of predicted class
        probabilities (list): Probabilities for all classes [class 0, class 1]
    """
    if model is None:
        return "Error no model", 0.0, [0.0, 0.0]

    # 1️⃣ Prepare features for prediction
    features = prepare_features_for_prediction(text, category=category)
    if features is None:
        return "Error no features", 0.0, [0.0, 0.0]
    
    # 2️⃣ Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction]
    
    # 3️⃣ Convert prediction to label
    label = "Human" if prediction == 1 else "AI"
    
    return label, confidence, probabilities.tolist()











# Build the UI

# app.py

def main():
    # Configure the page
    st.set_page_config(
        page_title="Amazon Review Analyzer",
        page_icon="🤖"
    )

    # Title and description
    st.title("🤖 Amazon Review Analyzer")
    st.markdown("Determine whether a product review was written by a human or generated by AI.")

    # Load model with a loading spinner
    with st.spinner("Loading XGBoost model..."):
        model_dict = get_xgb_model()

    if model_dict is None:
        st.error("Failed to load model. Please check if model files exist.")
        return

    st.success("XGBoost model loaded successfully!")

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Enter Review Text:**")
        
        # Create a text area for review input
        input_review = st.text_area(
            "Review text:",
            height=200,
            label_visibility="collapsed"
        )

      
        # Create a select box for product category
        category_options = list(CATEGORY_MAPPING.keys())
        category = st.selectbox(
            "Product Category:",
            category_options
        )   

        # Analyze button
        analyze_button = st.button("Analyze Review", type="primary")

    # Results Section
    with col2:
        st.write("**Analysis Results:**")

        # Only run analysis if button is clicked and there's input
        if analyze_button and input_review.strip():
            with st.spinner("Analyzing with XGBoost model..."):
                try:
                    # Map user-friendly category to dataset category
                    dataset_category = CATEGORY_MAPPING[category]

                    # Make prediction with xgb_predict()
                    model = model_dict.get("best_model")
                    label, confidence, probabilities = xgb_predict(
                        input_review,
                        model,
                        category=dataset_category
                    )

                    # Display the prediction with appropriate styling
                    if label == "AI":
                        st.error(f"🤖 Prediction: **{label}**")
                    else:
                        st.success(f"👤 Prediction: **{label}**")

                    # Display confidence score
                    confidence_pct = round(confidence * 100, 2)
                    st.metric("Confidence", f"{confidence_pct}%")

                    # Feature Analysis Expander
                    with st.expander("Feature Analysis"):
                        include_pos = True  # Match your model's feature set

                        # Get the features using extract_features
                        features_df = extract_features(input_review, include_pos=include_pos)

                        st.write("**Extracted Features:**")

                        # Display features in two columns
                        col1_feat, col2_feat = st.columns(2)

                        # Separate basic features from POS features
                        basic_features = {
                            "char_length": features_df.get("char_length", [0])[0],
                            "word_count": features_df.get("word_count", [0])[0],
                            "punctuation_ct": features_df.get("punctuation_ct", [0])[0],
                            "avg_word_len": features_df.get("avg_word_len", [0])[0],
                            "uppercase_ratio": features_df.get("uppercase_ratio", [0])[0]
                        }

                        pos_features = {col: features_df.get(col, [0])[0] for col in POS_WHITELIST if col in features_df.columns}

                        with col1_feat:
                            st.write("**Basic Features:**")
                            for feat_name, feat_value in basic_features.items():
                                st.write(f"- {feat_name}: {feat_value:.2f}")

                        with col2_feat:
                            st.write("**POS Features:**")
                            for feat_name, feat_value in pos_features.items():
                                st.write(f"- {feat_name}: {int(feat_value)}")

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)

        elif analyze_button and not input_review.strip():
            st.warning("Please enter a review to analyze!")


if __name__ == "__main__":
    main()