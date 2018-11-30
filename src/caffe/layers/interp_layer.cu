#include <algorithm>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/interp_layer.hpp"

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if(interp_type_ == "bilinear"){
    caffe_gpu_interp2<Dtype,false>(num_ * channels_,
      bottom[0]->gpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
      top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
  }
  else{
    caffe_gpu_nninterp2<Dtype,false>(num_ * channels_,
      bottom[0]->gpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
      top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
  }
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {return;}
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  if (interp_type_ == "bilinear"){
    caffe_gpu_interp2_backward<Dtype,false>(num_ * channels_,
      bottom[0]->mutable_gpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
      top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
  }
  else{
  	caffe_gpu_nninterp2_backward<Dtype,false>(num_ * channels_,
      bottom[0]->mutable_gpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
      top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(InterpLayer);


}  // namespace caffe
