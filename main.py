import numpy as np
import cv2

image_path = 'street.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

min_confidence = 0.4 #the minimum confidence needed for program to recognise an object

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa ", "train", "tvmonitor"]

np.random.seed(543210) #to always get same random result when generating coloured boxes
colours = np.random.uniform(0, 255, size = (len(classes), 3)) #generate value from 0 255, size is lengh of classes list, generate 3 values for each (RGB)

#loading pre-trained model, feeding data into it for prediction

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

#image = cv2.imread(image_path)
cam = cv2.VideoCapture(0)

while True:
    _, image = cam.read()
    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130) #can change 0.007 and 130 to get better results, maybe play around(increase/decrease) with them

    net.setInput(blob)
    detected_objects = net.forward()

    print(detected_objects[0][0][0])

    #making box around any objects that are detected
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {confidence*100: .2f}%"

            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colours[class_index], 3)
            cv2.putText(image, prediction_text, (upper_left_x,
                        upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours[class_index], 2)
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(5)

cv2.destroyAllWindows()
cam.release()