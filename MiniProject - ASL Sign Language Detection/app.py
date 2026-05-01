import os
import sys
import time
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify, request
import cv2
import joblib
import mediapipe as mp

# Import feature extraction functions from the new engine
from engine import algo1, algo2

app = Flask(__name__)

# Global state
state = {
    "current_prediction": "None",
    "current_confidence": "0.0",
    "current_word": "",
    "active_algorithm": 2  # Default to Algorithm 2
}

# Models dictionary to store both algorithms
models = {
    1: {"model": None, "scaler": None, "engine": algo1},
    2: {"model": None, "scaler": None, "engine": algo2}
}

def load_models():
    base_path = os.path.dirname(__file__)
    
    # Load Algorithm 1
    try:
        models[1]["model"] = joblib.load(os.path.join(base_path, 'models', 'svm_model_1.pkl'))
        models[1]["scaler"] = joblib.load(os.path.join(base_path, 'models', 'scaler_1.pkl'))
        print("Algorithm 1 models loaded.")
    except Exception as e:
        print(f"Error loading Algorithm 1: {e}")

    # Load Algorithm 2
    try:
        models[2]["model"] = joblib.load(os.path.join(base_path, 'models', 'svm_model_2.pkl'))
        models[2]["scaler"] = joblib.load(os.path.join(base_path, 'models', 'scaler_2.pkl'))
        print("Algorithm 2 models loaded.")
    except Exception as e:
        print(f"Error loading Algorithm 2: {e}")

load_models()

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.buffer = deque(maxlen=10)
        self.last_letter = ""
        self.last_added_time = 0
        self.COOLDOWN = 1.2
        self.STABILITY_THRESHOLD = 7

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global state
        
        success, frame = self.video.read()
        if not success:
            return None
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        algo_id = state["active_algorithm"]
        current_algo = models[algo_id]
        
        if current_algo["model"] is not None and current_algo["scaler"] is not None:
            result = hands.process(rgb)
            
            display_letter = "None"
            confidence = 0.0

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Use the active algorithm's feature extraction
                    features = current_algo["engine"].extract_features(hand_landmarks)
                    features_scaled = current_algo["scaler"].transform([features])

                    prediction = current_algo["model"].predict(features_scaled)[0]
                    
                    if hasattr(current_algo["model"], "predict_proba"):
                        proba = current_algo["model"].predict_proba(features_scaled)[0]
                        confidence = max(proba)
                    else:
                        confidence = 1.0

                    self.buffer.append(prediction)
                    
                    most_common, count = Counter(self.buffer).most_common(1)[0]
                    display_letter = most_common
                    
                    state["current_prediction"] = display_letter
                    state["current_confidence"] = str(confidence)

                    current_time = time.time()
                    if count >= self.STABILITY_THRESHOLD:
                        if most_common != self.last_letter and (current_time - self.last_added_time) > self.COOLDOWN:
                            if most_common == "space":
                                state["current_word"] += " "
                            elif most_common == "del":
                                state["current_word"] = state["current_word"][:-1]
                            else:
                                state["current_word"] += most_common

                            self.last_letter = most_common
                            self.last_added_time = current_time
            else:
                state["current_prediction"] = "None"
                state["current_confidence"] = "0.0"

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html', active_algo=state["active_algorithm"])

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/prediction')
def get_prediction():
    return jsonify({
        "prediction": state["current_prediction"],
        "confidence": state["current_confidence"],
        "word": state["current_word"],
        "active_algorithm": state["active_algorithm"]
    })

@app.route('/api/switch_algorithm', methods=['POST'])
def switch_algorithm():
    global state
    data = request.json
    new_algo = data.get('algorithm')
    if new_algo in [1, 2]:
        state["active_algorithm"] = new_algo
        return jsonify({"success": True, "active_algorithm": new_algo})
    return jsonify({"success": False, "error": "Invalid algorithm ID"}), 400

@app.route('/api/clear_word', methods=['POST'])
def clear_word():
    global state
    state["current_word"] = ""
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

