import cv2
import mediapipe as mp

from detection.ear import calculate_ear
from detection.mar import calculate_mar
from detection.alarm import play_alarm, stop_alarm
from detection.state import driver_state


class FaceMeshDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # THRESHOLDS
        self.EAR_THRESHOLD = 0.20
        self.MAR_THRESHOLD = 0.75

        self.CLOSED_EYES_FRAMES = 15

        self.frame_counter = 0

    def detect_mesh(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        h, w, _ = frame.shape

        status = "ACTIVE"

        fatigue_score = 10

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark

                # =========================
                # LEFT EYE
                # =========================
                left_eye_indices = [33, 160, 158, 133, 153, 144]

                # RIGHT EYE
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                left_eye = []
                right_eye = []

                for idx in left_eye_indices:

                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)

                    left_eye.append((x, y))

                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                for idx in right_eye_indices:

                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)

                    right_eye.append((x, y))

                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # =========================
                # EAR
                # =========================
                left_ear = calculate_ear(left_eye)

                right_ear = calculate_ear(right_eye)

                ear = (left_ear + right_ear) / 2

                # =========================
                # MOUTH
                # =========================
                mouth_indices = [61, 81, 13, 311, 291, 402, 14, 178]

                mouth = []

                for idx in mouth_indices:

                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)

                    mouth.append((x, y))

                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                # =========================
                # MAR
                # =========================
                mar = calculate_mar(mouth)

                # =========================
                # DISPLAY VALUES
                # =========================
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

                # =========================
                # DROWSINESS
                # =========================
                if ear < self.EAR_THRESHOLD:

                    self.frame_counter += 1

                    fatigue_score += 5 * self.frame_counter

                    status = "DROWSY"

                    if self.frame_counter >= self.CLOSED_EYES_FRAMES:

                        cv2.putText(
                            frame,
                            "DROWSINESS DETECTED!",
                            (100, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

                        play_alarm()

                else:

                    self.frame_counter = 0

                    stop_alarm()

                # =========================
                # YAWNING
                # =========================
                if mar > self.MAR_THRESHOLD:

                    status = "YAWNING"

                    fatigue_score += 25

                    cv2.putText(
                        frame,
                        "YAWNING DETECTED!",
                        (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3
                    )

                # LIMIT SCORE
                fatigue_score = min(fatigue_score, 100)

                # DISPLAY SCORE
                cv2.putText(
                    frame,
                    f"Fatigue: {fatigue_score}%",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # DISPLAY STATUS
                cv2.putText(
                    frame,
                    f"Status: {status}",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # UPDATE GLOBAL STATE
                driver_state["status"] = status
                driver_state["fatigue_score"] = fatigue_score

        return frame