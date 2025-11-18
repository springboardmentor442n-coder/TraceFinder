import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="TraceFinder – Scanner ID", layout="centered")

# --------------------------------------------------------
# 1. REBUILD FUNCTIONAL MODEL (IMPORTANT)
# --------------------------------------------------------
inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', name="last_conv")(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(8, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.load_weights("scanner_cnn_model.h5")
st.success("Model Loaded Successfully")


# --------------------------------------------------------
# 2. Grad-CAM function
# --------------------------------------------------------
def generate_gradcam(img_array, layer_name="last_conv"):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))

    cam = np.zeros(conv_out.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_out[0,:,:,i]

    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    return cam, int(pred_idx), float(preds[0][pred_idx])


# --------------------------------------------------------
# 3. Streamlit UI
# --------------------------------------------------------
st.title("🔍 TraceFinder – Forensic Scanner Identification")
uploaded_file = st.file_uploader("Upload Scanned Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img_np, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img_np, (224,224)) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)

    # Prediction + Grad-CAM
    cam, class_idx, conf = generate_gradcam(input_tensor)

    classes = ["Canon", "Epson", "HP", "Brand4", "Brand5", "Brand6", "Brand7", "Brand8"]

    st.subheader("🔎 Prediction Result")
    st.write(f"*Scanner Brand:* {classes[class_idx]}")
    st.write(f"*Confidence:* {conf*100:.2f}%")

    # Grad-CAM Overlay
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam_resized*255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    st.subheader("🔥 Grad-CAM (Scanner Trace Highlight)")
    st.image(overlay, use_column_width=True)

    # Log prediction
    log_text = f"{classes[class_idx]}, {conf:.4f}\n"
    with open("prediction_log.txt", "a") as f:
        f.write(log_text)

    st.download_button(
        "Download Prediction Log",
        data=log_text,
        file_name="prediction.txt"
    )