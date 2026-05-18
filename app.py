from flask import Flask, render_template, Response, jsonify
import cv2

from detection.face_mesh import FaceMeshDetector
from detection.state import driver_state

app = Flask(__name__)

camera = cv2.VideoCapture(0)

detector = FaceMeshDetector()


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = detector.detect_mesh(frame)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/driver_data')
def driver_data():
    return jsonify(driver_state)


if __name__ == "__main__":
    app.run(debug=True)