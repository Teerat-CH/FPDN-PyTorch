#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

std::pair<double,double> two_sum(double x, double y) {
    double s = x + y;
    double w = s - x;
    double v = s - w;
    double a = y - w;
    double b = v - x;
    double e = a - b;
    return {s, e};
}

py::array_t<double> kahan_dot(py::array_t<double> X, py::array_t<double> W) {
    auto bufX = X.unchecked<2>();
    auto bufW = W.unchecked<2>();

    ssize_t batch_size = bufX.shape(0);
    ssize_t input_dim  = bufX.shape(1);
    ssize_t output_dim = bufW.shape(1);

    auto result = py::array_t<double>({batch_size, output_dim});
    auto c      = py::array_t<double>({batch_size, output_dim});

    auto bufRes = result.mutable_unchecked<2>();
    auto bufC   = c.mutable_unchecked<2>();

    for (ssize_t i=0; i<input_dim; i++) {
        for (ssize_t b=0; b<batch_size; b++) {
            for (ssize_t o=0; o<output_dim; o++) {
                double prod = bufX(b,i) * bufW(i,o);
                auto [temp_sum, c_new] = two_sum(prod, bufC(b,o));
                auto [new_result, result_err] = two_sum(bufRes(b,o), temp_sum);
                bufRes(b,o) = new_result;
                bufC(b,o)   = c_new + result_err;
            }
        }
    }
    return result;
}

PYBIND11_MODULE(kahandot, m) {
    m.def("kahan_dot", &kahan_dot);
}