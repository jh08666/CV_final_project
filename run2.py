import cv2
import time
import math

cap = cv2.VideoCapture(0)
counter = 5000
fps_time = 0
turnon = True 

while (turnon):
    success, img = cap.read()
    if ((cv2.waitKey(1) == ord('q')) | (cv2.waitKey(1) == 27)):
        turnon = False
    if (counter>29):
        cv2.imwrite('.\\image\\'+str(counter-29)+'.jpg',img)
    counter += 1
    fps = 1.0 / (time.time() - fps_time)
    FPS = str(math.floor(fps))
    cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Webcam", img)
    fps_time = time.time()
    print(counter-30)

print('Total image: '+str(counter-30))    
cap.release()
cv2.destroyAllWindows()