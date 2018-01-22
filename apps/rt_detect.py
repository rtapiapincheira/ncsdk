from mvnc import mvncapi as mvnc

import cv2
import numpy
import sys.argv


def setup_mvnc():
    print('Setting MVNC up ...')
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
    print('    OK')


def get_devices():
    print('Getting devices ...')
    _devices = mvnc.EnumerateDevices()
    if len(_devices) == 0:
        print('No devices found')
        quit()
    print('    OK')
    return _devices


def get_graph(_device, _model_file):
    print('Loading model ...')
    with open(_model_file, mode='rb') as f:
        blob = f.read()
    print('    OK')
    print('Allocating graph object ...')
    g = device.AllocateGraph(blob)
    print('    OK')
    return g


def choose_device(_devices):
    print('Opening device NCS #0 ...')
    d = mvnc.Device(_devices[0])
    d.OpenDevice()
    print('    OK')
    return d


def release(_graph, _device):
    print('Releasing graph & device ...')
    _graph.DeallocateGraph()
    _device.CloseDevice()
    print('    OK')


# Interpret the output from a single inference of TinyYolo (GetResult)
# and filter out objects/boxes with low probabilities.
def filter_objects(inference_result, input_image_width, input_image_height):
    # the raw number of floats returned from the inference (GetResult())
    num_inference_results = len(inference_result)

    # the 20 classes this network was trained on
    network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # only keep boxes with probabilities greater than this
    probability_threshold = 0.07

    num_classifications = len(network_classifications)  # should be 20
    grid_size = 7  # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell = 2  # the number of boxes returned for each grid cell

    # grid_size is 7 (grid is 7x7)
    # num classifications is 20
    # boxes per grid cell is 2
    all_probabilities = numpy.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    # classification_probabilities  contains a probability for each classification for
    # each 64x64 pixel square of the grid.  The source image contains
    # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
    classification_probabilities = numpy.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    # The probability scale factor for each box
    box_prob_scale_factor = numpy.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    # get the boxes from the results and adjust to be pixel units
    all_boxes = numpy.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

    # adjust the probabilities with the scaling factor
    for box_index in range(boxes_per_grid_cell):  # loop over boxes
        for class_index in range(num_classifications):  # loop over classifications
            all_probabilities[:, :, box_index, class_index] = numpy.multiply(classification_probabilities[:, :, class_index], box_prob_scale_factor[:, :, box_index])

    probability_threshold_mask = numpy.array(all_probabilities >= probability_threshold, dtype='bool')
    box_threshold_mask = numpy.nonzero(probability_threshold_mask)
    boxes_above_threshold = all_boxes[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]
    classifications_for_boxes_above = numpy.argmax(all_probabilities, axis=3)[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]
    probabilities_above_threshold = all_probabilities[probability_threshold_mask]

    # sort the boxes from highest probability to lowest and then
    # sort the probabilities and classifications to match
    argsort = numpy.array(numpy.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold = probabilities_above_threshold[argsort]

    # get mask for boxes that seem to be the same object
    duplicate_box_mask = get_duplicate_box_mask(boxes_above_threshold)

    # update the boxes, probabilities and classifications removing duplicates.
    boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        classes_boxes_and_probs.append(
            [network_classifications[classifications_for_boxes_above[i]], boxes_above_threshold[i][0],
             boxes_above_threshold[i][1], boxes_above_threshold[i][2], boxes_above_threshold[i][3],
             probabilities_above_threshold[i]])

    return classes_boxes_and_probs


# creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
# that should be considered the same object.  This is determined by how similar the boxes are
# based on the intersection-over-union metric.
# box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
def get_duplicate_box_mask(box_list):
    # The intersection-over-union threshold to use when determining duplicates.
    # objects/boxes found that are over this threshold will be
    # considered the same object
    max_iou = 0.35

    box_mask = numpy.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                box_mask[j] = 0.0

    filter_iou_mask = numpy.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask


def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):
    # number of boxes per grid cell
    boxes_per_cell = 2

    # setup some offset values to map boxes to pixels
    # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
    box_offset = numpy.transpose(numpy.reshape(numpy.array([numpy.arange(grid_size)] * (grid_size * 2)), (boxes_per_cell, grid_size, grid_size)), (1, 2, 0))

    # adjust the box center
    box_list[:, :, :, 0] += box_offset
    box_list[:, :, :, 1] += numpy.transpose(box_offset, (1, 0, 2))
    box_list[:, :, :, 0:2] = box_list[:, :, :, 0:2] / (grid_size * 1.0)

    # adjust the lengths and widths
    box_list[:, :, :, 2] = numpy.multiply(box_list[:, :, :, 2], box_list[:, :, :, 2])
    box_list[:, :, :, 3] = numpy.multiply(box_list[:, :, :, 3], box_list[:, :, :, 3])

    # scale the boxes to the image size in pixels
    box_list[:, :, :, 0] *= image_width
    box_list[:, :, :, 1] *= image_height
    box_list[:, :, :, 2] *= image_width
    box_list[:, :, :, 3] *= image_height


def get_intersection_over_union(box_1, box_2):
    # one diminsion of the intersecting box
    intersection_dim_1 = min(box_1[0] + 0.5 * box_1[2], box_2[0] + 0.5 * box_2[2]) - max(box_1[0] - 0.5 * box_1[2], box_2[0] - 0.5 * box_2[2])

    # the other dimension of the intersecting box
    intersection_dim_2 = min(box_1[1] + 0.5 * box_1[3], box_2[1] + 0.5 * box_2[3]) - max(box_1[1] - 0.5 * box_1[3], box_2[1] - 0.5 * box_2[3])

    if intersection_dim_1 < 0 or intersection_dim_2 < 0:
        # no intersection area
        intersection_area = 0
    else:
        # intersection area is product of intersection dimensions
        intersection_area = intersection_dim_1 * intersection_dim_2

    # calculate the union area which is the area of each box added
    # and then we need to subtract out the intersection area since
    # it is counted twice (by definition it is in each box)
    union_area = box_1[2] * box_1[3] + box_2[2] * box_2[3] - intersection_area;

    # now we can return the intersection over union
    return intersection_area / union_area


def display_objects_in_gui(source_image, filtered_objects, pause_time=0):
    # copy image so we can draw on it.
    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    print('Found this many objects in the image: ' + str(len(filtered_objects)))
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3]) // 2
        half_height = int(filtered_objects[obj_index][4]) // 2

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        print('box at index ' + str(obj_index) + ' is... left: ' + str(box_left) + ', top: ' + str(box_top) + ', right: ' + str(box_right) + ', bottom: ' + str(box_bottom))

        # draw the rectangle on the image.  This is hopefully around the object
        box_color = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70)  # greyish green background for text
        label_text_color = (255, 255, 255)  # white text
        cv2.rectangle(display_image, (box_left, box_top - 20), (box_right, box_top), label_background_color, -1)
        cv2.putText(display_image, filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5], (box_left + 5, box_top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    window_name = 'TinyYolo (hit key to exit)'
    cv2.imshow(window_name, display_image)
    return cv2.waitKey(pause_time) & 0xFF == ord('q')


print('Running [modified] NCS Caffe TinyYolo example')

setup_mvnc()

devices = get_devices()

device = choose_device(devices)

graph, dim = get_graph(device, '../ncappzoo/caffe/TinyYolo/graph'), (448, 448)

cap = cv2.VideoCapture('/home/rene/Desktop/Videos/video1.mp4')

pause_time = int(sys.argv[2]) if len(sys.argv) >= 3 else 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    (w, h) = frame.shape

    if w != h:
        if w < h:
            b = (h - w) / 2
            frame = cv2.copyMakeBorder(frame, 0, 0, b, b, cv2.BORDER_CONSTANT, 0)
        else:
            b = (w - h) / 2
            frame = cv2.copyMakeBorder(frame, b, b, 0, 0, cv2.BORDER_CONSTANT, 0)

    # Read image from file, resize it to network width and height
    # save a copy in img_cv for display, then convert to float32, normalize (divide by 255),
    # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
    input_image = cv2.resize(frame, dim, cv2.INTER_LINEAR)

    display_image = input_image
    input_image = input_image.astype(numpy.float32)
    input_image = numpy.divide(input_image, 255.0)

    # Load tensor and get result.  This executes the inference on the NCS
    graph.LoadTensor(input_image.astype(numpy.float16), 'user object')
    output, userobj = graph.GetResult()

    # filter out all the objects/boxes that don't meet thresholds
    filtered_objs = filter_objects(output.astype(numpy.float32), input_image.shape[1], input_image.shape[0])  # fc27 instead of fc12 for yolo_small

    # display the filtered objects/boxes in a GUI window
    if display_objects_in_gui(display_image, filtered_objs, pause_time):
        break

# When everything done, release the capture
print('Releasing camera resources ...')
cap.release()
cv2.destroyAllWindows()
print('    OK')

release(graph, device)

print('Finished')

