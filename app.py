import streamlit as st
import numpy as np
import json
import os
from PIL import Image
from src.features import extract_all_features
import joblib


def load_saved_model(model_dir='output/rf'):
    """Load saved sklearn model and label encoder from output directory."""
    results_path = os.path.join(model_dir, 'results.json')
    model_file = os.path.join(model_dir, 'model.joblib')
    encoder_file = os.path.join(model_dir, 'label_encoder.joblib')

    metadata = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            metadata = json.load(f)

    model = None
    label_encoder = None
    if os.path.exists(model_file):
        model = joblib.load(model_file)
    if os.path.exists(encoder_file):
        label_encoder = joblib.load(encoder_file)

    return model, label_encoder, metadata


def process_image(image):
    """
    Process uploaded image and extract features (hand-crafted).
    """
    image_array = np.array(image)
    features = extract_all_features(image_array)
    return features


def main():
    st.title("Scanner Identification System")
    st.write("Upload a scanned document to identify the source scanner model.")

    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # convert multi-frame or palette images to RGB for feature extraction
        try:
            if getattr(image, 'n_frames', 1) > 1:
                image = image.convert('RGB')
            elif image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception:
            image = image.convert('RGB')

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Identify Scanner'):
            with st.spinner('Analyzing image...'):
                features = process_image(image)

                # Load saved model and encoder
                model, label_encoder, metadata = load_saved_model('output/rf')

                if model is None or label_encoder is None:
                    st.error('Trained model or label encoder not found in output/rf. Run training first.')
                    return

                # Prepare feature vector for sklearn model (flatten dict -> vector)
                def flatten_features(feat_dict):
                    # deterministic ordering: sort keys
                    vec = []
                    for k in sorted(feat_dict.keys()):
                        v = feat_dict[k]
                        if isinstance(v, (list, tuple, np.ndarray)):
                            vec.extend(list(v))
                        else:
                            vec.append(v)
                    return np.asarray(vec).reshape(1, -1)

                X = flatten_features(features)

                # Predict
                if not hasattr(model, 'predict'):
                    st.error('Loaded object is not a scikit-learn estimator (missing predict).')
                    return

                # Get probabilities if available
                probs = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(X)[0]
                    except Exception:
                        probs = None

                # Get predicted encoded label (model's predict returns encoded integer)
                try:
                    pred_enc = model.predict(X)[0]
                except Exception as e:
                    st.error(f'Prediction failed: {e}')
                    return

                # Map encoded class to human label using label encoder
                try:
                    pred_label = label_encoder.inverse_transform([int(pred_enc)])[0]
                except Exception:
                    # Fallback: if label_encoder not matching, show raw value
                    pred_label = str(pred_enc)

                st.subheader('Results')
                st.success(f'Predicted Scanner Model: {pred_label}')

                st.subheader('Confidence Scores')
                st.subheader('Confidence Scores')
                if probs is not None:
                    # Resolve class order from model.classes_ if available
                    try:
                        model_classes = getattr(model, 'classes_', None)
                        if model_classes is not None:
                            # convert encoded classes to human labels
                            try:
                                human_classes = label_encoder.inverse_transform([int(c) for c in model_classes])
                            except Exception:
                                human_classes = list(label_encoder.classes_)
                        else:
                            human_classes = list(label_encoder.classes_)
                    except Exception:
                        human_classes = list(label_encoder.classes_)

                    for cls, p in zip(human_classes, probs):
                        st.write(f'{cls}: {p*100:.2f}%')
                        st.progress(float(p))
                else:
                    st.write('Probability scores not available for this model. Showing predicted label only.')


if __name__ == '__main__':
    main()