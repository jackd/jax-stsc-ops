#ifndef _JAX_STSC_OPS_KERNELS_H_
#define _JAX_STSC_OPS_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace jax_stsc_ops {
struct CumulativeEmaDescriptor {
  std::int64_t size;
  bool reverse;
};
struct TiledCumulativeEmaDescriptor {
  std::int64_t repeats;
  std::int64_t size;
  bool reverse;
};
struct TiledSegmentCumulativeEmaDescriptor {
  std::int64_t num_channels;
  std::int64_t num_events;
  std::int64_t num_segments;
  bool reverse;
};

void gpu_cumulative_ema_f32(cudaStream_t stream, void** buffers,
                            const char* opaque, std::size_t opaque_len);
void gpu_cumulative_ema_f64(cudaStream_t stream, void** buffers,
                            const char* opaque, std::size_t opaque_len);
void gpu_cumulative_ema_c64(cudaStream_t stream, void** buffers,
                            const char* opaque, std::size_t opaque_len);
void gpu_cumulative_ema_c128(cudaStream_t stream, void** buffers,
                             const char* opaque, std::size_t opaque_len);
void gpu_segment_cumulative_ema_f32(cudaStream_t stream, void** buffers,
                                    const char* opaque, std::size_t opaque_len);
void gpu_segment_cumulative_ema_f64(cudaStream_t stream, void** buffers,
                                    const char* opaque, std::size_t opaque_len);
void gpu_tiled_segment_cumulative_ema_f32(cudaStream_t stream, void** buffers,
                                          const char* opaque, std::size_t opaque_len);
void gpu_tiled_segment_cumulative_ema_f64(cudaStream_t stream, void** buffers,
                                          const char* opaque, std::size_t opaque_len);
void gpu_serial_segment_cumulative_ema_f32(cudaStream_t stream, void** buffers,
                                           const char* opaque, std::size_t opaque_len);
void gpu_serial_segment_cumulative_ema_f64(cudaStream_t stream, void** buffers,
                                           const char* opaque, std::size_t opaque_len);
}  // namespace jax_stsc_ops

#endif