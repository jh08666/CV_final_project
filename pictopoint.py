import cv2
import numpy as np 
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

if __name__ == '__main__': 
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))

    f = open('D:\\tf-pose-estimation-master\\PRJ11210-main\\lefthand.txt', "w")
    print('Start!')
    for i in range(4971, 5109):
        image = cv2.imread('D:\\tf-pose-estimation-master\\PRJ11210-main\\image\\'+str(i)+'.jpg')
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        w = np.reshape(flat[0:36], (18,2))
        for j in range(0, 18):
            x = w[j, 0]
            y = w[j, 1]
            if (x > 0):
                x = int(x)
            if (y > 0):
                y = int(y)    
            f.write(str(x)+','+str(y)+',')
        f.write('\n')
    print('Finished!')
    f.close()

    # f = open('.\\pictopoint\\Data\\normal.txt', "w")
    # print('Start!')
    # for i in range(1, 6001):
    #     image = cv2.imread('.\\pictopoint\\images\\normal\\'+str(i)+'.jpg')
    #     humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
    #     img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    #     w = np.reshape(flat[0:36], (18,2))
    #     for j in range(0, 18):
    #         x = w[j, 0]
    #         y = w[j, 1]
    #         if (x > 0):
    #             x = int(x)
    #         if (y > 0):
    #             y = int(y)    
    #         f.write(str(x)+','+str(y)+',')
    #     f.write('\n')
    # print('Finished!')
    # f.close()

    # f = open('.\\pictopoint\\Data\\righthand.txt', "w")
    # print('Start!')
    # for i in range(1, 6001):
    #     image = cv2.imread('.\\pictopoint\\images\\righthand\\'+str(i)+'.jpg')
    #     humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
    #     img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    #     w = np.reshape(flat[0:36], (18,2))
    #     for j in range(0, 18):
    #         x = w[j, 0]
    #         y = w[j, 1]
    #         if (x > 0):
    #             x = int(x)
    #         if (y > 0):
    #             y = int(y)    
    #         f.write(str(x)+','+str(y)+',')
    #     f.write('\n')
    # print('Finished!')
    # f.close()

    # f = open('.\\pictopoint\\Data\\stand.txt', "w")
    # print('Start!')
    # for i in range(1, 6001):
    #     image = cv2.imread('.\\pictopoint\\images\\stand\\'+str(i)+'.jpg')
    #     humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
    #     img, flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    #     w = np.reshape(flat[0:36], (18,2))
    #     for j in range(0, 18):
    #         x = w[j, 0]
    #         y = w[j, 1]
    #         if (x > 0):
    #             x = int(x)
    #         if (y > 0):
    #             y = int(y)    
    #         f.write(str(x)+','+str(y)+',')
    #     f.write('\n')
    # print('Finished!')
    # f.close()
