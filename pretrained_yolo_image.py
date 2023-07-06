import numpy as np
import cv2

img_to_detect = cv2.imread('images/testing/IMG_5549.jpg')
img_height = img_to_detect.shape[0]
img_width = img_to_detect.shape[1]

img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (608, 608), swapRB=True, crop=False)

class_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

red = np.array([255, 0, 0], dtype=np.uint8)
green = np.array([0, 255, 0], dtype=np.uint8)
blue = np.array([0, 0, 255], dtype=np.uint8)
cyan = np.array([0, 255, 255], dtype=np.uint8)
yellow = np.array([255, 255, 0], dtype=np.uint8)

class_colors = np.array([red, green, blue, cyan, yellow])
class_colors = np.tile(class_colors, (16, 1))

print(f'Class colors have the shape: {class_colors.shape}')

yolo_model = cv2.dnn.readNetFromDarknet('model/yolov3-spp.cfg', 'model/yolov3-spp.weights')
yolo_layers = yolo_model.getLayerNames()
yolo_output_layer = [
    yolo_layers[yolo_layer- 1]
    for yolo_layer in yolo_model.getUnconnectedOutLayers()
]

yolo_model.setInput(img_blob)
obj_detection_layers = yolo_model.forward(yolo_output_layer)

for odl in obj_detection_layers:
    for object_detection in odl:
        all_scores = object_detection[5:]
        predicted_class_id = np.argmax(all_scores)
        prediction_confidence = all_scores[predicted_class_id]

        if prediction_confidence > 0.2:
            predicted_class_label = class_labels[predicted_class_id]
            bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
            (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype('int')
            start_x_pt = int(box_center_x_pt - (box_width/2))
            start_y_pt =int(box_center_y_pt - (box_height/2))
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height

            box_color = class_colors[predicted_class_id]
            box_color = [int(c) for c in box_color]

            predicted_class_label = '{}: {:.2f}%'.format(predicted_class_label, prediction_confidence * 100)
            print(f'Predicted object is: {predicted_class_label}')

            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 2)
            cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_DUPLEX, 1.5, box_color, 2)


cv2.imshow('Detection Output', img_to_detect)
cv2.waitKey(0)
print('hello world')