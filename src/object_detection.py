import cv2 as cv
import time
import numpy as np

def object_detection(video, network, layers_names_output)
    f = 0
    # Defining variable for counting total time
    # At the end we will show time spent for processing all frames
    t = 0
    # Defining loop for catching frames
    while True:
        # Capturing frame-by-frame
        print('frame: ', f)
        ret, frame = video.read()
        if not ret:
            break

        if w is None or h is None:
            # Slicing from tuple only first two elements
            h, w = frame.shape[:2]


        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)

        print('blob')

        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()
        print('time ', end-start)
        # Increasing counters for frames and total time
        f += 1
        t += end - start

        # Showing spent time for single current frame
        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min,
                                           int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():

                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Preparing colour for current bounding box
                # and converting from numpy array to list
                colour_box_current = colours[class_numbers[i]].tolist()


                # Drawing bounding box on the original current frame
                cv.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                       confidences[i])

                # Putting text with label and confidence on the original image
                cv.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

        if writer is None:
            # Constructing code of the codec
            # to be used in the function VideoWriter
            fourcc = cv.VideoWriter_fourcc(*'mp4v')

            writer = cv.VideoWriter('sample_data', fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        print('frame: ', f)
        # Write processed current frame to the file
        cv.imshow(frame)
        writer.write(frame)



if __name__ == '__main__':
    video = cv.VideoCapture("segmentation_result.mp4")
    # that we will use to write processed frames
    writer = None
    # Preparing variables for spatial dimensions of the frames
    h, w = None, None
    with open('coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]
    network = cv.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
    layers_names_all = network.getLayerNames()
    layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    probability_minimum = 0.5
    # Setting threshold for filtering weak bounding boxes
    # with non-maximum suppression
    threshold = 0.3
    # Generating colours for representing every detected object
    # with function randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    yolo_dection_result = object_detection(video, network, layers_names_output)
    # Printing final results
    print('Total number of frames', f)
    print('Total amount of time {:.5f} seconds'.format(t))
    print('FPS:', round((f / t), 1))

    # Releasing video reader and writer
    video.release()
    writer.release()
