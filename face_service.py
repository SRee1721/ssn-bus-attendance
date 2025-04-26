# server_face_stream.py

from flask import Flask, request, Response
import cv2
import numpy as np
from threading import Lock

app = Flask(__name__)

frame_lock = Lock()
latest_frame = None

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    global latest_frame
    if 'frame' not in request.files:
        return 'No frame uploaded', 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with frame_lock:
        latest_frame = frame

    return 'Frame received', 200

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Face Recognition Stream (Server)</title>
    </head>
    <body>
        <h1>Live Video Stream from Raspberry Pi</h1>
        <img src="/video_feed" width="640">
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
