import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time
import pygame
# Load the pre-trained drowsiness detection model
model_drowsiness = load_model("cnnCat2.h5")
model_yawn = load_model("crazy4.keras")
labels_drowsiness = ['drowsy', 'awake']

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

total_frames = 0
drowsy_frames=0
start_time = time.time()
pygame.mixer.init()
pygame.mixer.music.load('alert.mp3')

# Initialize the webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while True:
        flag=0
        total_frames += 1
        ret, frame = cap.read()

        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image using MediaPipe Face Detection
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Adjust the region of interest to include more of the mouth area
                roi_y = y + int(h / 2)
                roi_x = x
                roi_height = int(h / 2)  # Reduce the height to focus more on the mouth area
                roi_width = w

                # Calculate the new width (decrease width for a rectangle around the center)
                new_width = int(roi_width * 0.5)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Extract region from under the nose to chin
                nose_to_chin_region = gray_frame[roi_y:roi_y + roi_height,
                                      roi_x + int((roi_width - new_width) / 2):roi_x + int(
                                          (roi_width - new_width) / 2) + new_width]
                nose_to_chin_region = cv2.resize(nose_to_chin_region, (24, 24))
                nose_to_chin_region = np.expand_dims(nose_to_chin_region, axis=0)
                nose_to_chin_region = nose_to_chin_region / 255.0  # Normalize pixel values to [0, 1]

                # Make predictions using your drowsiness detection model
                prediction = model_yawn.predict(nose_to_chin_region)

                # Get the predicted label
                if prediction < 0.055:
                    predicted_label = 'no yawn'
                else:
                    predicted_label = 'yawn'
                    if flag==0:
                        drowsy_frames+=1
                        flag=1

                # Display the predicted label on the nose region
                cv2.putText(frame, predicted_label, (int(roi_x + (roi_width - new_width) / 2), roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add the code you provided here
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                # See where the user's head is tilting
                if x < -10 or x > 20:
                    text = "Nodding"
                    if flag==0:
                        drowsy_frames+=1
                        flag=1
                else:
                    text = "No Nodding"

                # Add the text on the image
                image=cv2.flip(image, 1)
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Continue with the rest of the code for drowsiness detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(image)
        faces = results.multi_face_landmarks

        if faces:
            for landmarks in faces:
                left_eye_landmarks = [landmarks.landmark[i] for i in range(159, 145, -1)]
                right_eye_landmarks = [landmarks.landmark[i] for i in range(386, 380, -1)]

                left_eye_x = [int(landmark.x * img_w) for landmark in left_eye_landmarks]
                left_eye_y = [int(landmark.y * img_h) for landmark in left_eye_landmarks]

                right_eye_x = [int(landmark.x * img_w) for landmark in right_eye_landmarks]
                right_eye_y = [int(landmark.y * img_h) for landmark in right_eye_landmarks]

                radius = 12
                left_eye_region = gray[min(left_eye_y) - radius:max(left_eye_y) + radius, min(left_eye_x) - radius:max(left_eye_x) + radius]
                left_eye_region = cv2.resize(left_eye_region, (24, 24))
                left_eye_region = np.uint8(left_eye_region)

                right_eye_region = gray[min(right_eye_y) - radius:max(right_eye_y) + radius, min(right_eye_x) - radius:max(right_eye_x) + radius]
                right_eye_region = cv2.resize(right_eye_region, (24, 24))
                right_eye_region = np.uint8(right_eye_region)

                prediction1 = model_drowsiness.predict(np.expand_dims(left_eye_region, axis=0))
                prediction2 = model_drowsiness.predict(np.expand_dims(right_eye_region, axis=0))

                if labels_drowsiness[prediction1.argmax()] == 'drowsy' and labels_drowsiness[prediction2.argmax()] == 'drowsy':
                    predicted_label_drowsiness = 'Drowsy'
                    if flag==0:
                        drowsy_frames+=1
                        flag=1
                else:
                    predicted_label_drowsiness = 'Non-Drowsy'

                cv2.putText(image, f"Drowsiness: {predicted_label_drowsiness}", (left_eye_x[0], left_eye_y[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if time.time() - start_time >= 3:
            perclos = (drowsy_frames / float(total_frames)) * 100
            start_time = time.time()
            total_frames = 0
            drowsy_frames = 0
            if perclos>=50:
                pygame.mixer.music.play()
        cv2.imshow("Face Detection and Drowsiness Detection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()