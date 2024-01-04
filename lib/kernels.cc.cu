// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include <cuComplex.h> // CUDA complex arithmetic
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/reverse.h>
#include <thrust/iterator/zip_iterator.h>

#include "jax_stsc_ops.h"
#include "kernel_helpers.h"
#include "kernels.h"

namespace jax_stsc_ops {

namespace {

template <typename T>
struct EMAOp {
    __host__ __device__ thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& a, const thrust::tuple<T, T>& b) const {
        T x_a = thrust::get<0>(a);
        T f_a = thrust::get<1>(a);
        T x_b = thrust::get<0>(b);
        T f_b = thrust::get<1>(b);
        T x_out = f_b * x_a + x_b;
        T f_out = f_a * f_b;
        return thrust::make_tuple(x_out, f_out);
    }
};

template <>
struct EMAOp<cuComplex> {
    __host__ __device__ thrust::tuple<cuComplex, cuComplex> operator()(const thrust::tuple<cuComplex, cuComplex>& a, const thrust::tuple<cuComplex, cuComplex>& b) const {
        cuComplex x_a = thrust::get<0>(a);
        cuComplex f_a = thrust::get<1>(a);
        cuComplex x_b = thrust::get<0>(b);
        cuComplex f_b = thrust::get<1>(b);

        cuComplex x_out = cuCaddf(cuCmulf(f_b, x_a), x_b);
        cuComplex f_out = cuCmulf(f_a, f_b);
        return thrust::make_tuple(x_out, f_out);
    }
};

template <>
struct EMAOp<cuDoubleComplex> {
    __host__ __device__ thrust::tuple<cuDoubleComplex, cuDoubleComplex> operator()(const thrust::tuple<cuDoubleComplex, cuDoubleComplex>& a, const thrust::tuple<cuDoubleComplex, cuDoubleComplex>& b) const {
        cuDoubleComplex x_a = thrust::get<0>(a);
        cuDoubleComplex f_a = thrust::get<1>(a);
        cuDoubleComplex x_b = thrust::get<0>(b);
        cuDoubleComplex f_b = thrust::get<1>(b);

        cuDoubleComplex x_out = cuCadd(cuCmul(f_b, x_a), x_b);
        cuDoubleComplex f_out = cuCmul(f_a, f_b);
        return thrust::make_tuple(x_out, f_out);
    }
};

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_cumulative_ema(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len) {
  const CumulativeEmaDescriptor &d = *UnpackDescriptor<CumulativeEmaDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;
  const bool reverse = d.reverse;

  const T *values = reinterpret_cast<const T *>(buffers[0]);
  const T *factors = reinterpret_cast<const T *>(buffers[1]);
  T *output = reinterpret_cast<T *>(buffers[2]);

  // std::cout << "values" << std::endl;
  // T* host_values = new T[size];
  // cudaMemcpy(host_values, values, size * sizeof(T), cudaMemcpyDeviceToHost);
  // for (size_t i = 0; i < size; ++i) {
  //   std::cout << host_values[i] << std::endl;
  // }
  // delete[] host_values;
  // std::cout << "factors" << std::endl;
  // T* host_factors = new T[size];
  // cudaMemcpy(host_factors, factors, size * sizeof(T), cudaMemcpyDeviceToHost);
  // for (size_t i = 0; i < size; ++i) {
  //   std::cout << host_factors[i] << std::endl;
  // }
  // delete[] host_factors;

  thrust::device_vector<T> tmp(size);

  auto values_iter = thrust::device_ptr<const T>(values);
  auto factors_iter = thrust::device_ptr<const T>(factors);
  auto output_iter = thrust::device_ptr<T>(output);

  auto in_iter = thrust::make_zip_iterator(thrust::make_tuple(values_iter, factors_iter));
  auto out_iter = thrust::make_zip_iterator(thrust::make_tuple(output_iter, tmp.begin()));

  auto exec_policy = thrust::cuda::par.on(stream);
  if (reverse) {
    thrust::inclusive_scan(
      exec_policy,
      thrust::make_reverse_iterator(in_iter + size),
      thrust::make_reverse_iterator(in_iter),
      thrust::make_reverse_iterator(out_iter + size),
      EMAOp<T>()
    );
  } else {
    thrust::inclusive_scan(
      exec_policy,
      in_iter,
      in_iter + size,
      out_iter,
      EMAOp<T>()
    );
  }

  ThrowIfError(cudaGetLastError());
}

template <typename T>
inline void apply_segment_cumulative_ema(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len) {
  const CumulativeEmaDescriptor &d = *UnpackDescriptor<CumulativeEmaDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;
  const bool reverse = d.reverse;

  const T *values = reinterpret_cast<const T *>(buffers[0]);
  const T *factors = reinterpret_cast<const T *>(buffers[1]);
  const int *segment_ids = reinterpret_cast<const int *>(buffers[2]);
  T *output = reinterpret_cast<T *>(buffers[3]);

  thrust::device_vector<T> tmp(size);

  auto values_iter = thrust::device_ptr<const T>(values);
  auto factors_iter = thrust::device_ptr<const T>(factors);
  auto segment_ids_iter = thrust::device_ptr<const int>(segment_ids);
  auto output_iter = thrust::device_ptr<T>(output);

  auto in_iter = thrust::make_zip_iterator(thrust::make_tuple(values_iter, factors_iter));
  auto out_iter = thrust::make_zip_iterator(thrust::make_tuple(output_iter, tmp.begin()));

  thrust::equal_to<int> binary_pred;
  auto exec_policy = thrust::cuda::par.on(stream);
  if (reverse) {
    thrust::inclusive_scan_by_key(
      exec_policy,
      thrust::make_reverse_iterator(segment_ids_iter + size),
      thrust::make_reverse_iterator(segment_ids_iter),
      thrust::make_reverse_iterator(in_iter + size),
      thrust::make_reverse_iterator(out_iter + size),
      binary_pred,
      EMAOp<T>()
    );
  } else {
    thrust::inclusive_scan_by_key(
      exec_policy,
      segment_ids_iter,
      segment_ids_iter + size,
      in_iter,
      out_iter,
      binary_pred,
      EMAOp<T>()
    );
  }

  ThrowIfError(cudaGetLastError());
}

// struct ravel_functor {
//   int size;

//   ravel_functor(int size) : size(size) {}

//   __host__ __device__
//   int operator()(const thrust::tuple<int, int>& indices) const {
//       int i0 = thrust::get<0>(indices);
//       int i1 = thrust::get<1>(indices);
//       return i0 * size + i1;
//   }
// };

// struct cycle_functor{
//   int period;
//   cycle_functor(int period): period(period) {}
//   __host__ __device__
//   int operator()(const int n) const {
//     return n % period;
//   }
// };

// template<typename Iterator>
// thrust::permutation_iterator<Iterator, thrust::transform_iterator<cycle_functor, thrust::counting_iterator<int>>> make_cycle_iterator(Iterator iterator, int period)
// {
//   thrust::counting_iterator<int> counter(0);
//   typedef thrust::transform_iterator<cycle_functor, thrust::counting_iterator<int>> index_iterator;
//   index_iterator indices_iter(counter, cycle_functor(period));
//   return thrust::permutation_iterator<Iterator, index_iterator>(iterator, indices_iter);
// }

// struct repeat_functor{
//   int repeats;
//   repeat_functor(int repeats): repeats(repeats) {}
//   __host__ __device__
//   int operator()(const int n) const {
//     return n / repeats;
//   }
// };

// template<typename Iterator>
// thrust::permutation_iterator<Iterator, thrust::transform_iterator<repeat_functor, thrust::counting_iterator<int>>> make_repeat_iterator(Iterator iterator, int repeats)
// {
//   thrust::counting_iterator<int> counter(0);
//   typedef thrust::transform_iterator<repeat_functor, thrust::counting_iterator<int>> index_iterator;
//   index_iterator indices_iter(counter, repeat_functor(repeats));
//   return thrust::permutation_iterator<Iterator, index_iterator>(iterator, indices_iter);
// }

// template <typename T>
// inline void apply_tiled_segment_cumulative_ema(cudaStream_t stream, void **buffers,
//                                          const char *opaque, std::size_t opaque_len) {
//   const TiledCumulativeEmaDescriptor &d = *UnpackDescriptor<TiledCumulativeEmaDescriptor>(opaque, opaque_len);
//   const std::int64_t repeats = d.repeats;
//   const std::int64_t size = d.size;
//   const bool reverse = d.reverse;
//   const std::int64_t total_size = repeats * size;

//   const T *values = reinterpret_cast<const T *>(buffers[0]);
//   const T *factors = reinterpret_cast<const T *>(buffers[1]);
//   const int *segment_ids = reinterpret_cast<const int *>(buffers[2]);
//   T *output = reinterpret_cast<T *>(buffers[3]);

//   thrust::device_vector<T> tmp(total_size);

//   auto values_iter = thrust::device_ptr<const T>(values);
//   auto factors_iter = thrust::device_ptr<const T>(factors);
//   auto segment_ids_iter = thrust::device_ptr<const int>(segment_ids);
//   auto output_iter = thrust::device_ptr<T>(output);

//   auto in_iter = thrust::make_zip_iterator(thrust::make_tuple(values_iter, factors_iter));
//   auto out_iter = thrust::make_zip_iterator(thrust::make_tuple(output_iter, tmp.begin()));

//   auto tiled_segment_ids_iter = thrust::make_transform_iterator(
//     thrust::make_zip_iterator(
//       thrust::make_tuple(
//         make_cycle_iterator(segment_ids_iter, size),
//         make_repeat_iterator(thrust::counting_iterator<int>(0), size)
//       )
//     ),
//     ravel_functor(repeats)
//   );

//   thrust::equal_to<int> binary_pred;
//   auto exec_policy = thrust::cuda::par.on(stream);
//   if (reverse) {
//     thrust::inclusive_scan_by_key(
//       exec_policy,
//       thrust::make_reverse_iterator(tiled_segment_ids_iter + total_size),
//       thrust::make_reverse_iterator(tiled_segment_ids_iter),
//       thrust::make_reverse_iterator(in_iter + total_size),
//       thrust::make_reverse_iterator(out_iter + total_size),
//       binary_pred,
//       EMAOp<T>()
//     );
//   } else {
//     thrust::inclusive_scan_by_key(
//       exec_policy,
//       tiled_segment_ids_iter,
//       tiled_segment_ids_iter + total_size,
//       in_iter,
//       out_iter,
//       binary_pred,
//       EMAOp<T>()
//     );
//   }

//   ThrowIfError(cudaGetLastError());
// }
template <typename T>
inline void apply_tiled_segment_cumulative_ema(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len) {
  const TiledCumulativeEmaDescriptor &d = *UnpackDescriptor<TiledCumulativeEmaDescriptor>(opaque, opaque_len);
  const std::int64_t repeats = d.repeats;
  const std::int64_t size = d.size;
  const bool reverse = d.reverse;
  const std::int64_t total_size = repeats * size;

  const T *values = reinterpret_cast<const T *>(buffers[0]);
  const T *factors = reinterpret_cast<const T *>(buffers[1]);
  const int *segment_ids = reinterpret_cast<const int *>(buffers[2]);
  T *output = reinterpret_cast<T *>(buffers[3]);

  thrust::device_vector<T> tmp(total_size);

  auto values_iter = thrust::device_ptr<const T>(values);
  auto factors_iter = thrust::device_ptr<const T>(factors);
  auto segment_ids_iter = thrust::device_ptr<const int>(segment_ids);
  auto output_iter = thrust::device_ptr<T>(output);

  auto in_iter = thrust::make_zip_iterator(thrust::make_tuple(values_iter, factors_iter));
  auto out_iter = thrust::make_zip_iterator(thrust::make_tuple(output_iter, tmp.begin()));

  thrust::equal_to<int> binary_pred;

  cudaStream_t streams[repeats];

  // Create streams and launch operations in each stream
  for (int i = 0; i < repeats; ++i) {
    cudaStreamCreate(&streams[i]);
    auto exec_policy = thrust::cuda::par.on(streams[i]);
    if (reverse) {
      thrust::inclusive_scan_by_key(
        exec_policy,
        thrust::make_reverse_iterator(segment_ids_iter + size),
        thrust::make_reverse_iterator(segment_ids_iter),
        thrust::make_reverse_iterator(in_iter + size),
        thrust::make_reverse_iterator(out_iter + size),
        binary_pred,
        EMAOp<T>()
      );
      in_iter -= size;
      out_iter -= size;
    } else {
      thrust::inclusive_scan_by_key(
        exec_policy,
        segment_ids_iter,
        segment_ids_iter + size,
        in_iter,
        out_iter,
        binary_pred,
        EMAOp<T>()
      );
      in_iter += size;
      out_iter += size;
    }
  }

  // Synchronize all streams
  for (int i = 0; i < repeats; ++i) {
      cudaStreamSynchronize(streams[i]);
  }

  // Destroy streams
  for (int i = 0; i < repeats; ++i) {
      cudaStreamDestroy(streams[i]);
  }

  ThrowIfError(cudaGetLastError());
}

template <typename T>
__global__ void _segment_cumulative_ema_kernel(
    const T *values,
    const T *factors,
    const int *splits,
    T *output,
    const std::int64_t num_channels,
    const std::int64_t num_events,
    const std::int64_t num_segments,
    const bool reverse
) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int s = blockIdx.y * blockDim.y + threadIdx.y;

  if (c < num_channels && s < num_segments) {
    int start = (s == 0) ? 0 : splits[s - 1];
    int stop = splits[s];

    T acc = T(0);
    if (reverse) {
      for (int e = stop - 1; e >= start; --e) {
        acc *= factors[c * num_events + e];
        acc += values[c * num_events + e];
        output[c * num_events + e] = acc;
      }
    } else {
      for (int e = start; e < stop; ++e) {
        acc *= factors[c * num_events + e];
        acc += values[c * num_events + e];
        output[c * num_events + e] = acc;
      }
    }
  }
}

template <typename T>
inline void apply_serial_segment_cumulative_ema(cudaStream_t stream, void **buffers,
                                         const char *opaque, std::size_t opaque_len) {
  const TiledSegmentCumulativeEmaDescriptor &d = *UnpackDescriptor<TiledSegmentCumulativeEmaDescriptor>(opaque, opaque_len);
  const std::int64_t num_channels = d.num_channels;
  const std::int64_t num_events = d.num_events;
  const std::int64_t num_segments = d.num_segments;
  const bool reverse = d.reverse;

  const T *values = reinterpret_cast<const T *>(buffers[0]);
  const T *factors = reinterpret_cast<const T *>(buffers[1]);
  const int *splits = reinterpret_cast<const int *>(buffers[2]);
  T *output = reinterpret_cast<T *>(buffers[3]);

  dim3 block_size(256, 1); // You can adjust this block size based on your hardware
  dim3 grid_size((num_channels + block_size.x - 1) / block_size.x, (num_segments + block_size.y - 1) / block_size.y);

  _segment_cumulative_ema_kernel<<<grid_size, block_size>>>(
      values, factors, splits, output, num_channels, num_events, num_segments, reverse
  );

  cudaDeviceSynchronize();
}

}  // namespace

void gpu_cumulative_ema_f32(cudaStream_t stream, void **buffers,
                            const char *opaque, std::size_t opaque_len) {
  apply_cumulative_ema<float>(stream, buffers, opaque, opaque_len);
}

void gpu_cumulative_ema_f64(cudaStream_t stream, void **buffers,
                            const char *opaque, std::size_t opaque_len) {
  apply_cumulative_ema<double>(stream, buffers, opaque, opaque_len);
}
void gpu_cumulative_ema_c64(cudaStream_t stream, void **buffers,
                            const char *opaque, std::size_t opaque_len) {
  apply_cumulative_ema<cuComplex>(stream, buffers, opaque, opaque_len);
}

void gpu_cumulative_ema_c128(cudaStream_t stream, void **buffers,
                            const char *opaque, std::size_t opaque_len) {
  apply_cumulative_ema<cuDoubleComplex>(stream, buffers, opaque, opaque_len);
}

void gpu_segment_cumulative_ema_f32(cudaStream_t stream, void **buffers,
                                    const char *opaque, std::size_t opaque_len) {
  apply_segment_cumulative_ema<float>(stream, buffers, opaque, opaque_len);
}

void gpu_segment_cumulative_ema_f64(cudaStream_t stream, void **buffers,
                                    const char *opaque, std::size_t opaque_len) {
  apply_segment_cumulative_ema<double>(stream, buffers, opaque, opaque_len);
}

void gpu_tiled_segment_cumulative_ema_f32(cudaStream_t stream, void **buffers,
                                    const char *opaque, std::size_t opaque_len) {
  apply_tiled_segment_cumulative_ema<float>(stream, buffers, opaque, opaque_len);
}

void gpu_tiled_segment_cumulative_ema_f64(cudaStream_t stream, void **buffers,
                                    const char *opaque, std::size_t opaque_len) {
  apply_tiled_segment_cumulative_ema<double>(stream, buffers, opaque, opaque_len);
}

void gpu_serial_segment_cumulative_ema_f32(cudaStream_t stream, void **buffers,
                                    const char *opaque, std::size_t opaque_len) {
  apply_serial_segment_cumulative_ema<float>(stream, buffers, opaque, opaque_len);
}

void gpu_serial_segment_cumulative_ema_f64(cudaStream_t stream, void **buffers,
                                    const char *opaque, std::size_t opaque_len) {
  apply_serial_segment_cumulative_ema<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace jax_stsc_ops
