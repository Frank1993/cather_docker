import os.path
import numpy as np
import random as rd
from datetime import datetime
from _sparse_data_parser import _parse


def parse_sparse_line(line, label_dims, data_dims, label_dtype= 'i'):
    """Parse one line of sparse data to ndarray for data and label

    Parameters
    ----------
    line: the string contains sparse data
    label_dims: label dims
    data_dims: data dims
    label_dtype: label type, 'f' or 'i'

    Returns
    -------
    (label, data)

    """
    assert label_dtype == 'i' or label_dtype == 'f'
    assert len(line) > 0
    label_dim = _get_product(label_dims)
    data_dim = _get_product(data_dims)
    assert label_dim > 0 and data_dim > 0
    label_dtype_is_int = 1 if label_dtype == 'i' else 0

    label, data= _parse(line, label_dim, data_dim, label_dtype_is_int)
    label = label.reshape(label_dims)
    data = data.reshape(data_dims)
    return label, data 

def _get_product(dims):
    if len(dims) == 1:
        return dims[0]

    res = 1
    for dim in dims:
        res *= dim
    return res

def test_parse_sparse_line():
    _test_int_label()
    _test_float_label()
    _test_correctly_parse()
    _test_performance()

def _test_int_label():
    line = "0:1 2:1 20:1.1";
    label_dim = 1
    data_dim = 25
    sample = parse_sparse_line(line, (label_dim,), (data_dim,), 'i')
    label = sample[0]
    data = sample[1]
    assert len(label) == label_dim
    assert len(data) == data_dim
    assert label[0] == 1
    assert data[1] == 1
    assert np.isclose(data[19], 1.1)
    print("finished test int label")

def _test_float_label():
    line = "0:1.1 2:1 20:1.1";
    label_dim = 4
    data_dim = 25
    sample = parse_sparse_line(line, (label_dim,), (data_dim,), 'f')
    label = sample[0]
    data = sample[1]
    assert len(label) == label_dim
    assert len(data) == data_dim
    assert np.isclose(label[0], 1.1)
    assert np.isclose(label[2], 1)
    assert np.isclose(data[16], 1.1)
    print("finished test float label")

def _test_performance():
    batch_size = 128
    label_dim = 1
    data_dim = 16014
    samples, lines = _genereate_test_samples(label_dim, data_dim, batch_size)
     
    res = []
    start_time = datetime.now()
    for line in lines:
        sample = parse_sparse_line(line, (label_dim, ), (471, 34, 1), 'i')
        res.append(sample)
    seconds = (datetime.now() - start_time).total_seconds()
    print("finished performance test: %f seconds per batch, %f per sample" %(seconds, seconds / batch_size))

def _genereate_test_samples(label_dim, data_dim, batch_size):
    i = 0
    samples = []
    sparse_percent = 0.125
    # generating test samples
    while i < batch_size:
        label = np.zeros((label_dim, ))
        data = np.zeros((data_dim, ))
        j = 0
        while j < len(label):
            label[j] = rd.randint(0, 34)
            j += 1
        j = 0
        while j < len(data):
            p = rd.random()
            if p < sparse_percent / 5:
                data[j] = rd.random() * 34
            elif p < sparse_percent:
                data[j] = rd.randint(1,10)
            j += 1
        samples.append((label, data))
        i += 1

    assert len(samples) == batch_size

    # encoding to lines
    lines = []
    for sample in samples:
        label = sample[0]
        data = sample[1]
        assert len(label) == label_dim
        assert len(data) == data_dim
        line = ''
        line += (str(0) + ':' + str(label[0]))
        i = 1
        while i < label_dim:
            line += ' ' + (str(i) + ':' + str(label[i]))
            i += 1
        i = 0
        while i < data_dim:
            if (data[i] > 1e-10):
                line += ' ' + (str(i + label_dim) + ':' + str(data[i]))
            i += 1
        lines.append(line)
    assert len(samples) == batch_size
    assert len(lines) == batch_size
    return samples, lines

def _test_correctly_parse():
    label_dim = 10
    data_dim = 10000
    batch_size = 10
    samples, lines = _genereate_test_samples(label_dim, data_dim, batch_size)
    res = []
    for line in lines:
        sample = parse_sparse_line(line, (label_dim, ), (data_dim, ), 'i')
        res.append(sample)
    
    for i in range(len(res)):
        assert np.isclose(samples[i][0], res[i][0]).all()
        assert np.isclose(samples[i][1], res[i][1]).all()
    
    print("finished test correctly parse")
