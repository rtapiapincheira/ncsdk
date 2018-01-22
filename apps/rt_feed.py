from mvnc import mvncapi as mvnc

import cv2
import numpy


def get_file_image(_image):
    print('Loading file image:' + _image)
    r = cv2.imread(_image)
    print('    OK')
    return r


def get_labels_file(_filename):
    print('Loading labels file:' + _filename)
    result = numpy.loadtxt(_filename, str, delimiter='\t')
    print('    OK')
    return result


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


def get_graph(_device, _modelfile):
    print('Loading model ...')
    with open(_modelfile, mode='rb') as f:
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


def get_means_file(_filename):
    print('Loading means file ...')
    mf = numpy.load(_filename).mean(1).mean(1)  # loading the mean file
    print('    OK')
    return mf


def prepare_image(_image, _ilsvrc_mean, _dim):
    ri = cv2.resize(_image, _dim)
    fi = ri.astype(numpy.float32)
    fi[:, :, 0] = (fi[:, :, 0] - _ilsvrc_mean[0])
    fi[:, :, 1] = (fi[:, :, 1] - _ilsvrc_mean[1])
    fi[:, :, 2] = (fi[:, :, 2] - _ilsvrc_mean[2])
    return fi


def release(_graph, _device):
    print('Releasing graph & device ...')
    _graph.DeallocateGraph()
    _device.CloseDevice()
    print('    OK')


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

cap = cv2.VideoCapture('/home/rene/Desktop/Videos/video1.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if False:
        img = prepare_image(frame, ilsvrc_mean, (227, 227))
        output, order = load_tensor_and_get_result(graph, img)

        print('\n------- predictions --------')
        for i in range(0, 0):
            oi = order[i]
            text = '(probability ' + str(output[oi] * 100) + '%) is ' + labels[oi] + '  label index is: ' + str(oi)
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            # print('prediction ' + str(i) + )

        print('\n----------------------------')

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

release(graph, device)
