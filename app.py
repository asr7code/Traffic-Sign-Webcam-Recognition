import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 🚦 Streamlit page config
st.set_page_config(page_title="Traffic Sign Voice Alert", layout="centered")

# 🧠 Label map
CLASS_LABELS = {
    0: "Speed limit 20 km per hour", 1: "Speed limit 30 km per hour", 2: "Speed limit 50 km per hour",
    3: "Speed limit 60 km per hour", 4: "Speed limit 70 km per hour", 5: "Speed limit 80 km per hour",
    6: "End of speed limit 80 km per hour", 7: "Speed limit 100 km per hour", 8: "Speed limit 120 km per hour",
    9: "No passing", 10: "No passing for vehicles over 3.5 tons", 11: "Right of way at next intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left", 20: "Dangerous curve to the right",
    21: "Double curve", 22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing", 29: "Bicycles crossing",
    30: "Beware of ice or snow", 31: "Wild animals crossing", 32: "End of all speed and passing limits",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing by vehicles over 3.5 tons"
}

# 📦 Load model
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# 📂 Preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# 🔊 Speak
def auto_speak_js(label_text):
    safe_label = label_text.replace('"', '\\"')
    js_code = f"""
        <script>
        var msg = new SpeechSynthesisUtterance("Caution! {safe_label}");
        msg.lang = 'en-US';
        msg.rate = 0.9;
        speechSynthesis.speak(msg);
        </script>
    """
    st.components.v1.html(js_code)

# 🎥 Webcam transformer
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = cv2.resize(img, (64, 64))
        processed_img = processed_img.astype("float32") / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)

        prediction = model.predict(processed_img)
        class_idx = np.argmax(prediction)
        label = CLASS_LABELS.get(class_idx, f"Class {class_idx}")
        confidence = prediction[0][class_idx] * 100

        cv2.putText(img, f"{label} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Trigger voice alert
        st.session_state['latest_prediction'] = label

        return img

# 🌟 UI
st.title("🚘 Traffic Sign Voice Alert System")
st.write("Choose an image or use your webcam to recognize traffic signs and get voice alerts.")

# 📤 Option 1: File upload
uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_np = np.array(image)[..., ::-1]  # Convert RGB to BGR

    processed_image = preprocess_image(image_np)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100
    predicted_label = CLASS_LABELS.get(predicted_index, f"Class {predicted_index}")

    st.markdown(f"### 🧠 Predicted Sign: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")
    auto_speak_js(predicted_label)

# 📷 Option 2: Webcam
st.write("Or use your **webcam** to capture traffic signs in real-time:")
ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

# 🗣️ Say prediction from webcam
if 'latest_prediction' in st.session_state:
    auto_speak_js(st.session_state['latest_prediction'])

