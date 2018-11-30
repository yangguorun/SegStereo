#ifndef CAFFE_BN_LAYER_HPP_
#define CAFFE_BN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BNLayer : public Layer<Dtype> {
public:
  explicit BNLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const  vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {return "BN";}
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,  const vector<Blob<Dtype>*>& bottom);

  // spatial mean & variance
  Blob<Dtype> spatial_statistic_;
  // batch mean & variance
  Blob<Dtype> batch_statistic_;
  // buffer blob
  Blob<Dtype> buffer_blob_;
  // x_norm_ is large and will keep after forward, so release it and recompute x_norm_
  // in backward in gpu mode. 
  Blob<Dtype> mean_, var_, x_norm_, x_std_;
  int count_;
  // May use for-loop instead

  // x_sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> spatial_sum_multiplier_, batch_sum_multiplier_;

  // dimension
  int axis_, num_, channels_, hw_;
  // eps
  Dtype var_eps_;
  // decay factor
  Dtype decay_;
  // whether use history mean&var in test
  bool use_history_in_test_;
  // use and freeze history mean&var in train
  bool frozen_;
};

}  // namespace caffe

#endif  // CAFFE_BN_LAYER_HPP_
