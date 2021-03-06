import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import glob as glob
import cv2
import random
import os
import skimage.data
import skimage.transform
import numpy as np
import sys
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from gtts import gTTS
from pygame import mixer

#path to your project folder
PATH='./project/'

import sys
#insert path to models/research
sys.path.insert(0,'./models/research')
#insert path to models/research/object_detection
sys.path.insert(1,'./models/research/object_detection')
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous'


MODEL_PATH = os.path.join(PATH, MODEL_NAME)
PATH_TO_CKPT = os.path.join(MODEL_PATH,'inference_graph/frozen_inference_graph.pb')
print(PATH_TO_CKPT)
PATH_TO_LABELS = os.path.join(PATH+'traffic-sign-detection/', 'gtsdb.pbtxt')

NUM_CLASSES = 43

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (20, 20)
g=0
detected=[]
def detect():
    global g
    global detected
    vidcap = cv2.VideoCapture('video1.avi')
    success,frame_image = vidcap.read()
    count1 = 0
    if not os.path.exists('test1'):
        os.makedirs('test1')
    with detection_graph.as_default():
        # with tf.Session(graph=detection_graph) as sess:
        sess=tf.Session(graph=detection_graph)
        while success:
            if count1%10==0:
                g=0
                image = Image.fromarray(frame_image)
                (im_width, im_height) = image.size
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=0)
                final_score = np.squeeze(scores)    
                count = 0
                for i in range(100):
                    if scores is None or final_score[i] > 0.5:
                            count = count + 1
                print(count)
                for i in range(count):
                    ymin = int((boxes[0][i][0]*im_height))
                    xmin = int((boxes[0][i][1]*im_width))
                    ymax = int((boxes[0][i][2]*im_height))
                    xmax = int((boxes[0][i][3]*im_width))
                    xmin-=5
                    ymin-=5
                    xmax+=5
                    ymax+=5
                    detected.append(xmin)
                    detected.append(ymin)
                    detected.append(xmax)
                    detected.append(ymax)
                    print(xmin, xmax, ymin, ymax)
                    if xmin==0:
                        continue
                    img2 = image.crop((xmin,ymin,xmax,ymax))
                    img3=img2.resize((100,100),PIL.Image.ANTIALIAS)
                    img3.save(PATH+'results_imgs/'+str(g)+'.jpg','JPEG')

                    g=g+1
                print(detected)
                image1= load_image_into_numpy_array(image)
                for i in range(0,len(detected),4):
                    print(i)
                    cv2.rectangle(image1,(detected[i],detected[i+1]),(detected[i+2],detected[i+3]),(0,255,0),2)
                    # i+=3
                h=count1/10
                cv2.imwrite("video/%dout.jpg"%h,image1)
                if count>0:
                    test()
                detected=[]

            cv2.imwrite("test1/%d.jpg" % count1, frame_image)      
            success,frame_image = vidcap.read()
            count1+=1

def load_data(data_dir):
    labels = []
    images = []

    file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    
    for f in file_names:
        images.append(skimage.data.imread(f))
            # labels.append(int(d))
    return images
  
recent=np.array([])
def test():
    global PATH
    global recent
    sign_names = ['Speed limit 20','Speed limit 30','Speed limit 50','Speed limit 60','Speed limit 70','Speed limit 80','End of speed limit 80','Speed limit 100','Speed limit 120','No passing','No passing for vehicles over 3 metric tons','Right-of-way at the next intersection','Priority road','Yield','Stop','No vehicles','Vehicles over 3 metric tons prohibited','No entry','General caution','Dangerous curve to the left','Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right','Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow','Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead','Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory','End of no passing','End of no passing by vehicles over 3 metric tons']
    images_test  = load_data(PATH+'results_imgs/')
   
    images64 = [skimage.transform.resize(image, (64,64)) for image in images_test]

    for image in images64[:5]:
        print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))    

    X = np.array(images64)
    json_file = open(PATH+'model_german_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(PATH+"model_german_1.h5")
    print("Loaded model from disk")
    # model.summary()

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    sample_images = [X[i] for i in range(len(X))]
    X_sample = np.array(sample_images)
    prediction = model.predict(X_sample)

    predicted_categories = np.argmax(prediction, axis=1)
    print(predicted_categories)
    count=0
    np.array_equal(predicted_categories,recent)

    import scipy.misc
    for i,img,j in zip(predicted_categories,sample_images,range(len(sample_images))):
        directory=PATH+'resans/'+str(sign_names[i])
        if not os.path.exists(directory):
            os.makedirs(directory)
        scipy.misc.imsave(directory+'/%d.jpg'%j, img)
    if not np.array_equal(predicted_categories,recent):
        recent=predicted_categories
        for category in predicted_categories:
            cat=str(category)
            os.system('mpg321 audio/'+cat+'.mp3')
            count+=1
    os.system('rm results_imgs/*')

detect()
