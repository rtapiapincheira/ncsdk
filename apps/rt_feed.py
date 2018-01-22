from mvnc import mvncapi as mvnc

import cv2
import numpy


def get_file_image(_image):
    return cv2.imread(_image)


def get_labels_file(_filename):
    return numpy.loadtxt(_filename, str, delimiter='\t')


def setup_mvnc():
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)


def get_devices():
    _devices = mvnc.EnumerateDevices()
    if len(_devices) == 0:
        print('No devices found')
        quit()
    return _devices


def get_graph(_device, _modelfile):
    with open(_modelfile, mode='rb') as f:
        blob = f.read()
    return device.AllocateGraph(blob)


def choose_device(_devices):
    d = mvnc.Device(_devices[0])
    d.OpenDevice()
    return d


def get_means_file(_filename):
    return numpy.load(_filename).mean(1).mean(1)  # loading the mean file


def prepare_image(_image, _ilsvrc_mean, _dim):
    ri = cv2.resize(_image, _dim)
    fi = ri.astype(numpy.float32)
    fi[:, :, 0] = (fi[:, :, 0] - _ilsvrc_mean[0])
    fi[:, :, 1] = (fi[:, :, 1] - _ilsvrc_mean[1])
    fi[:, :, 2] = (fi[:, :, 2] - _ilsvrc_mean[2])
    return fi


def release(_graph, _device):
    _graph.DeallocateGraph()
    _device.CloseDevice()


def load_tensor_and_get_result(_graph, _image):
    _graph.LoadTensor(_image.astype(numpy.float16), 'user object')
    (_output, _) = _graph.GetResult()
    return _output, _output.argsort()[::-1][6]


# ----------------------------- End of function definitions -----------------------------

labels = get_labels_file('../examples/data/ilsvrc12/synset_words.txt')

setup_mvnc()

devices = get_devices()

device = choose_device(devices)

graph = get_graph(device, '../examples/caffe/AlexNet/graph')

ilsvrc_mean = get_means_file('../examples/data/ilsvrc12/ilsvrc_2012_mean.npy')

# original = get_file_image('../examples/data/images/nps_electric_guitar.png')

# Camera #0
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()

    print(frame.shape)

    img = prepare_image(frame, ilsvrc_mean, (227, 227))
    output, order = load_tensor_and_get_result(graph, img)

    print('\n------- predictions --------')
    for i in range(0, 5):
        print('prediction ' + str(i) + ' (probability ' + str(output[order[i]] * 100) + '%) is ' + labels[
            order[i]] + '  label index is: ' + str(order[i]))
    print('\n----------------------------')

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

release(graph, device)
