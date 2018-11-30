#ifndef CAFFE_DNN_BN_HPP_
#define CAFFE_DNN_BN_HPP_
#define THREAD_BLOCK_SIZE 256

template <typename Dtype>
__global__ void bn_forward(const int num, const int map_size, const int channels,
    Dtype stat_ratio, Dtype stat_eps, Dtype decay, bool save_stat, bool use_history,
    const Dtype* in, Dtype* mean, Dtype* history_mean, Dtype* var, Dtype* history_var,
    Dtype* out, Dtype* x_norm, Dtype* x_std, const Dtype* scale,const Dtype* shift) {

  __shared__ Dtype buffer[THREAD_BLOCK_SIZE];

  // compute mean
  buffer[threadIdx.x] = 0;
  if(!use_history) {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * (map_size * (channels - 1)) + i + blockIdx.x * map_size;
      buffer[threadIdx.x] += in[location];
    }
    __syncthreads();
    for(int i = blockDim.x / 2; i > 0; i >>= 1) {
      if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      buffer[0] = buffer[0] * stat_ratio;
      mean[blockIdx.x] = buffer[0];
      if(save_stat) history_mean[blockIdx.x] += decay * (buffer[0] - history_mean[blockIdx.x]);
    }
  }
  else if(threadIdx.x == 0) {
    buffer[0] = history_mean[blockIdx.x];
    mean[blockIdx.x] = buffer[0];
  }
  __syncthreads();

  // compute var
  buffer[threadIdx.x] = 0;
  if(!use_history) {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * (map_size * (channels - 1)) + i + blockIdx.x * map_size;
      out[location] = in[location] - mean[blockIdx.x];
      buffer[threadIdx.x] += out[location] * out[location];
    }
    __syncthreads();
    for(int i = blockDim.x/2; i > 0; i >>= 1) {
      if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      buffer[0] = buffer[0] * stat_ratio;
      var[blockIdx.x] = buffer[0];
      if(save_stat) history_var[blockIdx.x] += decay * (buffer[0] - history_var[blockIdx.x]);
    }
  }
  else {
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * (map_size * (channels - 1)) + i + blockIdx.x * map_size;
      out[location] = in[location] - mean[blockIdx.x];
    }
    if(threadIdx.x == 0) {
      buffer[0] = history_var[blockIdx.x];
      var[blockIdx.x] = buffer[0];
    }
  }
  __syncthreads();

  // compute output 
  const Dtype temp = sqrt(var[blockIdx.x] + stat_eps);
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * (map_size * (channels - 1)) + i + blockIdx.x * map_size;
    if (x_norm) x_norm[location] = out[location] / temp;
    out[location] = out[location] / temp * scale[blockIdx.x] + shift[blockIdx.x];
  }

  if(threadIdx.x == 0)
    x_std[blockIdx.x] = temp;
}


template<typename Dtype> 
__global__ void bn_backward_frozen(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* var, const Dtype stat_eps, const Dtype* scale, Dtype* out) {
  Dtype std = sqrt(var[blockIdx.x] + stat_eps);
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    out[location] = in[location] * scale[blockIdx.x] / std;
  }
}

template<typename Dtype> 
__global__ void bn_backward(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, const Dtype* bottom_data, const Dtype* mean, 
    const Dtype* x_std, Dtype* out, Dtype* scale_diff, Dtype* shift_diff, const Dtype* scale_data) {

  if (x_norm == NULL) {
    // compute x_norm with mean and std
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * (map_size * (channels - 1)) + i + blockIdx.x * map_size;
      out[location] = (bottom_data[location] - mean[blockIdx.x]) / x_std[blockIdx.x];
    }
    x_norm = out;
  }

  __shared__ Dtype buffer_scale_diff[THREAD_BLOCK_SIZE];
  __shared__ Dtype buffer_shift_diff[THREAD_BLOCK_SIZE];
  buffer_scale_diff[threadIdx.x] = 0;
  buffer_shift_diff[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
    buffer_shift_diff[threadIdx.x] += in[location];
  }
  __syncthreads();
  for(int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(threadIdx.x < i) buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i];
    if(threadIdx.x < i) buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i];
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    scale_diff[blockIdx.x] += buffer_scale_diff[0];
    shift_diff[blockIdx.x] += buffer_shift_diff[0];
  }
  __syncthreads();
  Dtype s_data_v = scale_data[blockIdx.x], x_std_v = x_std[blockIdx.x];
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    out[location] = s_data_v * (in[location] - (x_norm[location] *
          buffer_scale_diff[0] + buffer_shift_diff[0]) / (num * map_size)) / x_std_v;
  }
}

template <typename Dtype>
__global__ void bn_forward_mean_before_allreduce(const int num, const int map_size, const int channels,
    Dtype stat_ratio, const Dtype* in, Dtype* mean) {
  __shared__ Dtype buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    buffer[threadIdx.x] += in[location];
  }   
  __syncthreads();
  for(int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i]; 
    __syncthreads();
  }   
  if(threadIdx.x == 0) {
    buffer[0] = buffer[0] * stat_ratio;
    mean[blockIdx.x] = buffer[0];
  }   
}

template <typename Dtype>
__global__ void bn_forward_var_before_allreduce(const int num, const int map_size, const int channels,
    Dtype stat_ratio, const Dtype* in, const Dtype* mean, Dtype* var, Dtype* out) {

  __shared__ Dtype buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    out[location] = in[location] - mean[blockIdx.x];
    //buffer[threadIdx.x] += pow(out[location], (Dtype)2);
    buffer[threadIdx.x] += out[location] * out[location];
  }
  __syncthreads();
  for(int i = blockDim.x/2; i > 0; i >>= 1) {
    if(threadIdx.x < i) buffer[threadIdx.x] += buffer[threadIdx.x + i];
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    buffer[0] = buffer[0] * stat_ratio;
    var[blockIdx.x] = buffer[0];
  }
}

template <typename Dtype>
__global__ void bn_forward_after_allreduce(const int num, const int map_size, const int channels,
    Dtype stat_eps, Dtype decay, Dtype* out, 
    const Dtype* mean, Dtype* history_mean, const Dtype* var, Dtype* history_var,
    Dtype* x_norm, Dtype* x_std, const Dtype* scale, const Dtype* shift) {
  //Dtype temp = pow(var[blockIdx.x] + stat_eps, (Dtype)0.5);
  Dtype temp = sqrt(var[blockIdx.x] + stat_eps);
  Dtype scale_value = scale[blockIdx.x], shift_value = shift[blockIdx.x];
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    if (x_norm) x_norm[location] = out[location] / temp;
    out[location] = out[location] / temp * scale_value + shift_value;
  }
  if(threadIdx.x == 0) {
    history_mean[blockIdx.x] += decay * (mean[blockIdx.x] - history_mean[blockIdx.x]);
    history_var[blockIdx.x] += decay * (var[blockIdx.x] - history_var[blockIdx.x]);
    x_std[blockIdx.x] = temp;
  }
}

template <typename Dtype>
__global__ void bn_backward_before_allreduce(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, const Dtype* bottom_data, const Dtype* mean, 
    const Dtype* x_std, Dtype* out, Dtype* local_scale_diff, Dtype* local_shift_diff, 
    Dtype* scale_diff, Dtype* shift_diff) {
  if (x_norm == NULL) {
    // compute x_norm with mean and std
    for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
      int location = i / map_size * (map_size * (channels - 1)) + i + blockIdx.x * map_size;
      out[location] = (bottom_data[location] - mean[blockIdx.x]) / x_std[blockIdx.x];
    }
    x_norm = out;
  }

  __shared__ Dtype buffer_scale_diff[THREAD_BLOCK_SIZE];
  __shared__ Dtype buffer_shift_diff[THREAD_BLOCK_SIZE];
  buffer_scale_diff[threadIdx.x] = 0;
  buffer_shift_diff[threadIdx.x] = 0;
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
    buffer_shift_diff[threadIdx.x] += in[location];
  }
  __syncthreads();
  for(int i = blockDim.x / 2; i > 0; i >>= 1) {
    if(threadIdx.x < i) {
      buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i]; 
      buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i]; 
    }
    __syncthreads();
  }
  if(threadIdx.x == 0) {
    local_scale_diff[blockIdx.x] = buffer_scale_diff[0];
    local_shift_diff[blockIdx.x] = buffer_shift_diff[0];
    scale_diff[blockIdx.x] += buffer_scale_diff[0];
    shift_diff[blockIdx.x] += buffer_shift_diff[0];
  }
}

template <typename Dtype>
__global__ void bn_backward_after_allreduce(const int num, const int map_size, const int channels,
    const Dtype* in, const Dtype* x_norm, const Dtype* local_scale_diff, const Dtype* local_shift_diff, 
    const Dtype* scale_data, const Dtype* x_std, Dtype* out, const int num_thread) {
  for(int i = threadIdx.x; i < num * map_size; i += blockDim.x) {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    out[location] = scale_data[blockIdx.x] * (in[location] - (x_norm[location] * local_scale_diff[blockIdx.x] 
                       + local_shift_diff[blockIdx.x]) / (num * map_size * num_thread)) / x_std[blockIdx.x];
  }
}

#endif
