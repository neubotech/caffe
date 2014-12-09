#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/col2row1.hpp"


namespace caffe {

template <typename Dtype>
__global__ void col2row1_gpu_kernel(const int n, const Dtype* data_in,
    const int w, const int h, const int m, const int n_g, const int g_in,
    Dtype* data_out) {
  CUDA_KERNEL_LOOP(index, n) {

    int w_in = index % w;
    int w_index = index / w;
    int h_in = w_index % h;
    int h_index = w_index / h;
    // int g_in = h_index % n_g;
    // int g_index = h_index /n_g;
    int m_in = h_index;

    Dtype* data_out_ptr = data_out + w_in + h_in * w + m_in * w * h;//+ g_in * m * w * h;
    *data_out_ptr = data_in[w_in + h_in * w + g_in * w * h + m_in * w * h * n_g];
    // *data_out_ptr = data_in[index];
  }
}

template <typename Dtype>
void col2row1_gpu(const Dtype* data_in, const int w,
    const int h, const int m, const int n_g, const int g_in,
    Dtype* data_out) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = w * h * m;// * n_g;
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2row1_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
       num_kernels, data_in,
       w, h, m, n_g, g_in, data_out);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2row1_gpu<float>(const float* data_in, const int w,
    const int h, const int m, const int n_g, const int g_in,
    float* data_out);
template void col2row1_gpu<double>(const double* data_in, const int w,
    const int h, const int m, const int n_g, const int g_in,
    double* data_out);

template <typename Dtype>
__global__ void row2col1_gpu_kernel(const int n, const Dtype* data_in,
    const int w, const int h, const int m, //const int n_g,
    const int ofst, int topi, Dtype* data_out) {
  CUDA_KERNEL_LOOP(index, n) {

    int w_in = index % w;
    int w_index = index / w;
    int h_in = w_index % h;
    int h_index = w_index / h;
    int m_in = h_index;

    Dtype* data_out_ptr = data_out + topi*(m_in + ofst) + w_in + h_in * w;

    *data_out_ptr = data_in[index];
  }
}

template <typename Dtype>
void row2col1_gpu(const Dtype* data_in, const int w,
    const int h, const int m, //const int n_g,
    const int ofst, int topi, Dtype* data_out) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = w * h * m;
  // NOLINT_NEXT_LINE(whitespace/operators)
  row2col1_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
       num_kernels, data_in,
       w, h, m, ofst, topi, data_out);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void row2col1_gpu<float>(const float* data_in, const int w,
    const int h, const int m, //const int n_g,
    const int ofst, int topi, float* data_out);
template void row2col1_gpu<double>(const double* data_in, const int w,
    const int h, const int m, //const int n_g,
    const int ofst, int topi, double* data_out);

}  // namespace caffe
