from ctypes import *


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("../../resources/darknet.so", RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)
set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect(net_main, meta_main, alt_names, image, thresh, hier_thresh, nms):
    num = c_int(0)
    pnum = pointer(num)
    im, arr = array_to_image(image)
    predict_image(net_main, im)
    dets = get_network_boxes(net_main, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    do_nms_sort(dets, num, meta_main.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta_main.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if alt_names is None:
                    name_tag = meta_main.names[i]
                else:
                    name_tag = alt_names[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


def load_network(config_path, weight_path, meta_path, names_path):
    net_main = load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)
    meta_main = load_meta(meta_path.encode("ascii"))
    with open(names_path, encoding='latin1') as names_file:
        names_list = names_file.read().strip().split("\n")
        alt_names = [x.strip() for x in names_list]
    return net_main, meta_main, alt_names
