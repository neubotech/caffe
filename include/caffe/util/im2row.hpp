#ifndef _CAFFE_UTIL_IM2ROW_HPP_
#define _CAFFE_UTIL_IM2ROW_HPP_

namespace caffe {


template <typename Dtype>
void im2row_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);


}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_