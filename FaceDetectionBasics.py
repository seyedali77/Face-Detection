import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("videos/1.mp4")
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()




pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)
    if results.detections:
        for id, detections in enumerate(results.detections):
            #mpDraw.draw_detection(img, detections)
            # print(id, detections)
            # print(detections.score)
            # print(detections.location_data.relative_bounding_box)
            bboxC = detections.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox, (255, 0, 0), 3 )
            cv2.putText(img, f'{int(detections.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)




    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                3,(0, 255, 0), 3)

    img_resized = cv2.resize(img, (800, 800))
    cv2.imshow("Image", img_resized)
    cv2.waitKey(10)
