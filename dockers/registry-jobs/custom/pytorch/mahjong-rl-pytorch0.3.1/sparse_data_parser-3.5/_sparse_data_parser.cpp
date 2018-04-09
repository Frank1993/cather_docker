#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <Python.h>
#include <numpy/arrayobject.h>

class SyntaxError : public std::runtime_error {
public:
    SyntaxError(std::string const &msg)
        : std::runtime_error(msg + " in sparse data parser.")
    {
    }
};

static PyObject* doParse(const char* line, npy_intp labelDim, npy_intp dataDim, int labelDtypeIsInt)
{
    int labelType = (labelDtypeIsInt == 1) ? NPY_INT32 : NPY_FLOAT32;
    
    PyArrayObject *outLabel = (PyArrayObject *)PyArray_ZEROS(1, &labelDim, labelType, 0);
    PyArrayObject *outData = (PyArrayObject *)PyArray_ZEROS(1, &dataDim, NPY_FLOAT32, 0);

    if (NULL == outLabel || NULL == outData)
    {
        throw std::bad_alloc();
    }

    void* pOutLabel = (void*)outLabel->data;
    npy_float32 * pOutData = (npy_float32 *)outData->data;

    npy_float32 value;
    int index;

    int i = 0;
    int j = 0;
    int lineLength = strlen(line);
    char number[100];
    while(i < lineLength)
    {
        // get index
        j = 0;
        while (i + j < lineLength && line[i + j] != ':') ++j;
        memcpy(number, line + i, j);
        number[j] = '\0';
        index = atoi(number);
        i = i + j + 1;

        // get value
        j = 0;
        while (i + j < lineLength && line[i + j] != ' ') ++j;
        memcpy(number, line + i, j);
        number[j] = '\0';
        value = atof(number);
        i = i + j + 1;

        if (index < 0 || index > labelDim + dataDim)
        {
            std::cout << "invalid index:" << index << std::endl;
            throw SyntaxError(std::string("invalid index."));
        }

        if (index < labelDim)
        {
            if (labelDtypeIsInt == 1)
            {
                ((npy_int32*) pOutLabel)[index] = (npy_int32)value;
            }
            else
            {
                ((npy_float32 *)pOutLabel)[index] = value;
            }
        }
        else
        {
            pOutData[index - labelDim] = value;
        }
    }

    PyObject * result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, (PyObject*)outLabel);
    PyTuple_SetItem(result, 1, (PyObject*)outData);
    return result;
}


static PyObject *_parse(PyObject *self, PyObject *args)
{
    const char* line;
    int labelDim;
    int  dataDim;
    int labelDtypeIsInt;

    if (!PyArg_ParseTuple(args, "siii", &line, &labelDim, &dataDim, &labelDtypeIsInt))
    {
        return NULL;
    }

    try
    {
        return doParse(line, labelDim, dataDim, labelDtypeIsInt);
    }
    catch (std::exception const& e)
    {
        std::cout << e.what() << std::endl;
        std::string msg("error in sparse data parser: ");
        msg += e.what();
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        return 0;
    }
}

static PyMethodDef SparseDataParserMethods[] = {
    { "_parse", _parse, METH_VARARGS, "Parse one line of sparse data" },
    { NULL, NULL, 0, NULL }     /* Sentinel - marks the end of this structure */
};


#define PYTHON_3

#ifdef PYTHON_3

static struct PyModuleDef CallModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_sparse_data_parser",
    "doc of sparse data parser",
    -1,
    SparseDataParserMethods,
    NULL,
    NULL,
    NULL,
    NULL
     };

extern "C" {
    PyMODINIT_FUNC
        PyInit__sparse_data_parser(void)
    {
        try
        {
            auto module = PyModule_Create(&CallModuleDef);
            import_array();
            return module;
        }
        catch (std::exception const& e)
        {
            std::cout << e.what() << std::endl;
            std::string msg("error in PyInit__sparse_data_parser: ");
            msg += e.what();
            PyErr_SetString(PyExc_RuntimeError, msg.c_str());
            return 0;
        }
    }
}


#else

extern "C" {
    PyMODINIT_FUNC
        init_sparse_data_parser(void)
    {
        (void)Py_InitModule("_sparse_data_parser", SparseDataParserMethods);
        import_array();
    }
}

#endif
