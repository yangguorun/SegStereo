#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 *  * @brief Abstract base class that factors out the BLAS code common to
 *   *        ConvolutionLayer and DeconvolutionLayer.
 *    */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int num_;
  int channels_;
  int pad_h_, pad_w_;
  int hole_h_,hole_w_;
  int height_, width_;
  int group_;
  bool group_compute_in_parallel_;
  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  bool is_1x1_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    im2col_cpu(data, 1, conv_in_channels_, conv_in_height_, conv_in_width_,
      kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, 
      1, 1, col_buff);
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    col2im_cpu(col_buff, 1, conv_in_channels_, conv_in_height_, conv_in_width_,
      kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, 
      1, 1, data);
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    im2col_gpu(data, 1, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, 
        1, 1, col_buff);
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    col2im_gpu(col_buff, 1, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, 
        1, 1, data);
  }
#endif

  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int conv_in_height_;
  int conv_in_width_;
  int kernel_dim_;
  int weight_offset_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
