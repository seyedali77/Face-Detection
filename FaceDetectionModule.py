import cv2  # Import OpenCV library for image processing
import mediapipe as mp  # Import Mediapipe for face detection
import time  # Import time module for calculating FPS

class FaceDetector():
    def __init__(self, minDetectionCon=0.5, modelSelection=0):
        # Initialize face detection parameters
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection

        # Load Mediapipe face detection module
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon, self.modelSelection)

    def findFaces(self, img, draw=True):
        # Convert image to RGB format (Mediapipe requires RGB)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image to detect faces
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []  # List to store bounding boxes and confidence scores

        if self.results.detections:
            for id, detections in enumerate(self.results.detections):
                # Get bounding box coordinates
                bboxC = detections.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detections.score])

                # Draw fancy bounding box and confidence score
                img = self.fancyDraw(img, bbox)
                cv2.putText(img, f'{int(detections.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 2)

        return img, bboxs  # Return processed image and bounding boxes

    def fancyDraw(self, img, bbox, l=30, rt=1, t=5):
        # Extract bounding box coordinates
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Draw bounding box rectangle
        cv2.rectangle(img, bbox, (255, 0, 0), rt)

        # Draw corner lines for a stylish bounding box
        # Top left
        cv2.line(img, (x, y), (x + l, y), (255, 0, 0), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 0), t)
        # Top right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 0), t)
        # Bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 0), t)
        # Bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 0), t)

        return img  # Return image with fancy bounding box

def main():
    cap = cv2.VideoCapture("videos/1.mp4")  # Load video file
    pTime = 0  # Initialize previous time for FPS calculation
    detector = FaceDetector()  # Create FaceDetector object

    while True:
        success, img = cap.read()  # Read frame from video
        img, bboxs = detector.findFaces(img)  # Detect faces in frame

        # Calculate frames per second (FPS)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the image
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        # Resize and display image
        img_resized = cv2.resize(img, (800, 800))
        cv2.imshow("Image", img_resized)
        cv2.waitKey(15)  # Delay for smooth video playback

if __name__ == "__main__":
    main()  # Run the main function
