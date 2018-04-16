Build and install:
    python setup.py build
    python setup.py install

Run test(optional):
    python -c "from sparse_data_parser import test_parse_sparse_line; test_parse_sparse_line()"

Usage:
    from sparse_data_parser import parse_sparse_line
    parse_sparse_line(line, label_dims, data_dims, label_dtype= 'i')

Example:
    line = "A SPARSE STRING"    # like "0:1 5:10 8:12.2"
    label_dims = (13,)
    data_dims = (471,34,1)
    label_dtype = 'i'
    sample = parse_sparse_line(line, label_dims, data_dims, 'i')

FAQ:
If you encounter "error: Unable to find vcvarsall.bat" on windows, please set environment variable for VC tools:
   For Visual Studio 2013 (VS12): SET VS90COMNTOOLS=%VS120COMNTOOLS%
   For Visual Studio 2015 (VS14): SET VS90COMNTOOLS=%VS140COMNTOOLS