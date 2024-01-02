import cv2
import keras
import time
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

#--Load Model--#
model = keras.models.load_model('D:\\tf-pose-estimation-master\\PRJ11210-main\\models\\Model_4types_DR_LSTM.h5')

if __name__ == '__main__': 
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    cap = cv2.VideoCapture(0)
    turnon = True
    fps_time = 0
    putWord = ""

    while(turnon):
        #--Webcam Capture and OpenPose Estimator--#
        ret, image = cap.read()
        humans = e.inference(image, resize_to_default=True, upsample_size=2.0)
        img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #--Double Reaction Prediction--#
        Predict = np.reshape(flat[0:36], (1,36,1))
        Predict_Result = model.predict_classes(Predict)
        Predict_Result = int(Predict_Result)

        #--Double Reaction String--#
        if(Predict_Result==0):
            putWord = "LeftHand"
        elif(Predict_Result==1):
            putWord = "RightHand"
        elif(Predict_Result==2):
            putWord = "Stand"
        else:
            putWord = "Normal"

        #--Show Frame--#
        cv2.rectangle(img, (0, 0), (240, 60), (204, 255, 255), -1)
        cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img,"Double Reaction: " + putWord, (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) 
        cv2.imshow("DR_test_v2", img)
        fps_time = time.time()

        #--Stop--#
        if cv2.waitKey(1) == ord('q'):
            turnon = False
    
    cap.release()
    cv2.destroyAllWindows()
