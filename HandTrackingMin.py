import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)   #1 is webcam number


mpHands = mp.solutions.hands
hands=mpHands.Hands()   #ctrl click Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #line 10 object only uses RGB IMG so onverting here
    results = hands.process(imgRGB)          #process the frame and gives results
    # print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:    #to know how many hands

            for id , lm in enumerate(handLms.landmark):    #id is index no.
                #print(id,lm)
                h , w , c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                #if id==5:
                cv2.circle(img,(cx,cy),15,(250,0,250),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)    #img here is original not RGB


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

                                 #10 AND 70 ARE POSITIONS, 3 IS SCALE..... 3 THICKNESS
    cv2.putText(img , str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


