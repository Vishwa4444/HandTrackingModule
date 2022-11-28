# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     roi = frame[290:2895,250:1400]
#
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("gray_Roi", gray_roi)
#
#
#     _, threshold = cv2.threshold(gray_roi,5,255,cv2.THRESH_BINARY_INV)
#     cv2.imshow("Roi", roi)
#     cv2.imshow("Threshold", threshold)
#     key = cv2.waitKey(30)
#     if ret == 27:
#         break
#
# cv2.destroyAllWindows()
# """
# Demonstration of the GazeTracking library.
# Check the README.md for complete documentation.
# """
#
# import cv2
# import gaze_tracking
#
# gaze = gaze_tracking()
# webcam = cv2.VideoCapture(0)
#
# while True:
#     # We get a new frame from the webcam
#     _, frame = webcam.read()
#
#     # We send this frame to GazeTracking to analyze it
#     gaze.refresh(frame)
#
#     frame = gaze.annotated_frame()
#     text = ""
#
#     if gaze.is_blinking():
#         text = "Blinking"
#     elif gaze.is_right():
#         text = "Looking right"
#     elif gaze.is_left():
#         text = "Looking left"
#     elif gaze.is_center():
#         text = "Looking center"
#
#     cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
#
#     left_pupil = gaze.pupil_left_coords()
#     right_pupil = gaze.pupil_right_coords()
#     cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#     cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#
#     cv2.imshow("Demo", frame)
#
#     if cv2.waitKey(1) == 27:
#         break
#
# webcam.release()
# cv2.destroyAllWindows()

import cv2 as cv
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[362,382,381,380,373,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_tracking_confidence=0.5,min_detection_confidence=0.5) as face_mesh:    #478 landmarks new version
   while True:
     ret, frame = cap.read()
     if not ret:
         break
     frame = cv.flip(frame,1)
     rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
     img_h,img_w = frame.shape[:2]
     results = face_mesh.process(rgb_frame)
     if results.multi_face_landmarks:
         #print(results.multi_face_landmarks)  # will get landmarks x,y,z

        # print(results.multi_face_landmarks[0].landmark)  #normalized values from 0 to 1 (get width and heigth and Multi with x and we get pixels cordn

        #[print(p.x,p.y) for p in results.multi_face_landmarks[0].landmark]
        mesh_points =  np.array([np.multiply([p.x, p.y], [img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        #print(mesh_points.shape)   gives two points (468, 2)

        # cv.polylines(frame,[mesh_points[LEFT_IRIS]],True,(0,255,0),1,cv.LINE_AA)
        # cv.polylines(frame,[mesh_points[RIGHT_IRIS]],True,(0,255,0),1,cv.LINE_AA)   #eyes detect
        (l_cx,l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx,l_cy],dtype=np.int32)
        center_right = np.array([r_cx,r_cy],dtype=np.int32)
        cv.circle(frame,center_left,int(l_radius),(255,0,255),1,cv.LINE_AA)
        cv.circle(frame,center_right,int(r_radius),(255,0,255),1,cv.LINE_AA)



     cv.imshow('img',frame)


     key = cv.waitKey(1)
     if key == ord('q') :    # converts char into unicode
         break
cap.release()
cv.destroyAllWindows()





