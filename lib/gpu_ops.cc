#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace jax_stsc_ops;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_cumulative_ema_f32"] = EncapsulateFunction(gpu_cumulative_ema_f32);
  dict["gpu_cumulative_ema_f64"] = EncapsulateFunction(gpu_cumulative_ema_f64);
  dict["gpu_cumulative_ema_c64"] = EncapsulateFunction(gpu_cumulative_ema_c64);
  dict["gpu_cumulative_ema_c128"] = EncapsulateFunction(gpu_cumulative_ema_c128);
  dict["gpu_segment_cumulative_ema_f32"] = EncapsulateFunction(gpu_segment_cumulative_ema_f32);
  dict["gpu_segment_cumulative_ema_f64"] = EncapsulateFunction(gpu_segment_cumulative_ema_f64);
  dict["gpu_tiled_segment_cumulative_ema_f32"] = EncapsulateFunction(gpu_tiled_segment_cumulative_ema_f32);
  dict["gpu_tiled_segment_cumulative_ema_f64"] = EncapsulateFunction(gpu_tiled_segment_cumulative_ema_f64);
  dict["gpu_serial_segment_cumulative_ema_f32"] = EncapsulateFunction(gpu_serial_segment_cumulative_ema_f32);
  dict["gpu_serial_segment_cumulative_ema_f64"] = EncapsulateFunction(gpu_serial_segment_cumulative_ema_f64);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_cumulative_ema_descriptor",
        [](std::int64_t size, bool reverse) { return PackDescriptor(CumulativeEmaDescriptor{size, reverse}); });
  m.def("build_tiled_cumulative_ema_descriptor",
        [](std::int64_t repeats, std::int64_t size, bool reverse) { return PackDescriptor(TiledCumulativeEmaDescriptor{repeats, size, reverse}); });
  m.def("build_tiled_segment_cumulative_ema_descriptor",
        [](std::int64_t num_channels, std::int64_t num_events, std::int64_t num_segments, bool reverse) { return PackDescriptor(TiledSegmentCumulativeEmaDescriptor{num_channels, num_events, num_segments, reverse}); });
}
}  // namespace
