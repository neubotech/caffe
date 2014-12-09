#ifndef _CAFFE_UTIL_COL2ROW1_HPP_
#define _CAFFE_UTIL_COL2ROW1_HPP_

namespace caffe {



template <typename Dtype>
void col2row1_gpu(const Dtype* data_in, const int w,
    const int h, const int m, const int n_g, const int g_in, Dtype* data_out);

template <typename Dtype>
void row2col1_gpu(const Dtype* data_in, const int w,
    const int h, const int m, //const int n_g,
    const int ofst, int topi, Dtype* data_out);

// template <typename Dtype>
// void row2col_gpu(const Dtype* data_in, const int w,
//     const int h, const int m, Dtype* data_out);

}  // namespace caffe

#endif  // CAFFE_UTIL_COL2ROW_HPP_
