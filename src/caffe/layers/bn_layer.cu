#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/kernels/dnn_bn.hpp"

#define THREAD_BLOCK_SIZE 256
namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* norm_data = NULL;
  if(bottom[0] == top[0]) {
    norm_data = x_norm_.mutable_gpu_data();
  }
  bn_forward<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, hw_, channels_, 
                   (Dtype)1/(num_*hw_), var_eps_, decay_,
                   (!frozen_ && this->phase_ == TRAIN), frozen_ || (use_history_in_test_ && this->phase_ == TEST), 
                   bottom[0]->gpu_data(), 
                   this->mean_.mutable_gpu_data(), this->blobs_[2]->mutable_gpu_data(), 
                   this->var_.mutable_gpu_data(), this->blobs_[3]->mutable_gpu_data(),
                   top[0]->mutable_gpu_data(), norm_data,
                   this->x_std_.mutable_gpu_data(), 
                   this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0] || this->param_propagate_down_[1] || propagate_down[0]) {
    if(this->frozen_) {
      bn_backward_frozen<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, hw_, channels_,
                    top[0]->gpu_diff(), this->blobs_[3]->gpu_data(), this->var_eps_, this->blobs_[0]->gpu_data(), 
                    bottom[0]->mutable_gpu_diff()); 
      CUDA_POST_KERNEL_CHECK;
      return;
    }
    Dtype* norm_data = NULL;
    if (bottom[0] == top[0]) {
      norm_data = x_norm_.mutable_gpu_data();
    }
    bn_backward<Dtype><<<channels_, THREAD_BLOCK_SIZE>>>(num_, hw_, channels_,
                    top[0]->gpu_diff(), norm_data, bottom[0]->gpu_data(), this->mean_.gpu_data(), 
                    this->x_std_.gpu_data(), bottom[0]->mutable_gpu_diff(), 
                    this->blobs_[0]->mutable_gpu_diff(), this->blobs_[1]->mutable_gpu_diff(),
                    this->blobs_[0]->gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);
}  // namespace caffe
