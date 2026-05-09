/*
 * CPython C extension: pointer-chase loop for RAM latency measurement.
 *
 * Follows a linked-list-style permutation array for `iters` hops,
 * defeating hardware prefetchers so we measure true random-access latency.
 *
 * Built as warpt.stress._pointer_chase_ext by setuptools (see setup.py).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>

static int64_t chase(const int64_t* arr, int64_t iters) {
    int64_t idx = 0;
    for (int64_t i = 0; i < iters; ++i) {
        idx = arr[idx];
    }
    return idx;
}

static PyObject* py_pointer_chase(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long data_ptr;
    long long iters;

    if (!PyArg_ParseTuple(args, "KL", &data_ptr, &iters))
        return NULL;

    int64_t result = chase(reinterpret_cast<const int64_t*>(data_ptr), iters);
    return PyLong_FromLongLong(result);
}

static PyMethodDef methods[] = {
    {"pointer_chase", py_pointer_chase, METH_VARARGS,
     "pointer_chase(data_ptr, iters) -> int\n\n"
     "Follow a permutation array for `iters` hops and return the final index."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pointer_chase_ext",
    "C++ pointer-chase backend for RAM latency measurement.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__pointer_chase_ext(void) {
    return PyModule_Create(&moduledef);
}
