import cv2 as cv
import mediapipe as mp
import numpy as np
import math
mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYE =[362,382,381,380,373,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_IRIS=[474,475,476,477]
LEFT_IRIS=[469,470,471,472]

L_H_LEFT = [33]  #right eye right most landmark
L_H_RIGHT = [133] #right eye left most landmark
R_H_LEFT = [362]  #left eye right most landmark
R_H_RIGHT = [263] #left eye left most landmark

def euclidean_distance(point1,point2):
    x1,y1 = point1.ravel()
    x2,y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
def irisPosition(iris_center,right_point,left_point):
    center_to_right_dist = euclidean_distance(iris_center,right_point)
    total_distance = euclidean_distance(right_point,left_point)
    ratio = center_to_right_dist/total_distance
    irisPosition=""
    if ratio<=0.42:
        irisPosition="right"
    elif ratio>0.42 and ratio <=0.57:
        irisPosition="center"
    else:
        irisPosition="left"
    return irisPosition,ratio




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
        cv.circle(frame,center_right,int(l_radius),(255,0,255),1,cv.LINE_AA)
        cv.circle(frame,mesh_points[R_H_LEFT][0],3,(255,100,255),-1,cv.LINE_AA)
        cv.circle(frame,mesh_points[R_H_RIGHT][0],3,(0,255,255),-1,cv.LINE_AA)


        irisPos,ratio = irisPosition(center_right,mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
        #print(irisPos)  #Gives position left,right,center
        cv.putText(frame,f"Iris pos:{irisPos} {ratio:.2f}",( 30,30),cv.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv.LINE_AA,)



     cv.imshow('img',frame)


     key = cv.waitKey(1)
     if key == ord('q') :    # converts char into unicode
         break
cap.release()
cv.destroyAllWindows()





