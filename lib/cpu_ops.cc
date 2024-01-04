#include <iostream>
#include <limits>
#include <complex>

#include "jax_stsc_ops.h"
#include "pybind11_kernel_helpers.h"

using namespace jax_stsc_ops;

namespace {

void counting_argsort(void *out, const void **in) {
  // Parse the inputs
  const std::int64_t num_ids = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t num_segments = *reinterpret_cast<const std::int64_t *>(in[1]);
  const int *ids = reinterpret_cast<const int *>(in[2]);
  const int *splits = reinterpret_cast<const int *>(in[3]);

  int *result = reinterpret_cast<int *>(out);

  // create temporary counts array of size num_segments
  int *counts = new int[num_segments]();

  // implementation below
  for (int i = 0; i < num_segments; ++i) {
    counts[i] = 0;
  }
  int n = splits[num_segments];
  for (int i = 0; i < n; ++i) {
    int id = ids[i];
    result[splits[id] + counts[id]] = i;
    counts[id] += 1;
  }
  delete[] counts;  // Deallocate the memory allocated for counts
}

void throttled_sample(void *out_tuple, const void **in) {
  const std::int64_t grid_size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t batch_size = *reinterpret_cast<const std::int64_t *>(in[1]);
  const std::int64_t size_in = *reinterpret_cast<const std::int64_t *>(in[2]);
  const std::int64_t size_out = *reinterpret_cast<const std::int64_t *>(in[3]);
  const std::int64_t sample_rate = *reinterpret_cast<const std::int64_t *>(in[4]);
  const float min_dt = *reinterpret_cast<const float *>(in[5]);

  const int *pixel_ids = reinterpret_cast<const int *>(in[6]);
  const float *times = reinterpret_cast<const float *>(in[7]);
  const int *batch_splits = reinterpret_cast<const int *>(in[8]);

  // The output is stored as a list of pointers since we have multiple outputs
  void **out = reinterpret_cast<void **>(out_tuple);
  int *sample_ids  = reinterpret_cast<int *>(out[0]);
  int *batch_splits_out = reinterpret_cast<int *>(out[1]);

  // create temporary counts / earliest_possible
  int *counts = new int[grid_size]();
  float *earliest_possible = new float[grid_size]();

  float neg_infinity = -std::numeric_limits<float>::infinity();
  int count = 0;
  batch_splits_out[0] = 0;
  for (auto b = 0; b < batch_size; ++b) {
    // reset counts / earliest_possible
    for (auto i = 0; i < grid_size; ++i) {
      counts[i] = 0;
      earliest_possible[i] = neg_infinity;
    }
    int e_start = batch_splits[b];
    int e_end = batch_splits[b+1];
    for (auto e = e_start; e < e_end; ++e) {
      int s = pixel_ids[e];
      int t = times[e];
      counts[s] += 1;
      if (counts[s] >= sample_rate && t >= earliest_possible[s]) {
        counts[s] = 0;
        earliest_possible[s] = t + min_dt;
        sample_ids[count] = e;
        count += 1;
      }
    }
    batch_splits_out[b + 1] = count;
  }
  // fill meaningless values
  for (auto e = count; e < size_out; ++e) {
    sample_ids[e] = size_in;
  }

  // Deallocate the memory allocated for temporary arrays
  delete[] counts;
  delete[] earliest_possible;
}

void get_stationary_predecessor_ids(void *out, const void **in) {
  const std::int64_t grid_size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t batch_size = *reinterpret_cast<const std::int64_t *>(in[1]);
  const std::int64_t num_events = *reinterpret_cast<const std::int64_t *>(in[2]);
  const std::int64_t kernel_size = *reinterpret_cast<const std::int64_t *>(in[3]);

  const int *pixel_ids = reinterpret_cast<const int *>(in[4]);
  const int *batch_splits = reinterpret_cast<const int *>(in[5]);
  const int *kernel_offsets = reinterpret_cast<const int *>(in[6]);

  // output
  int *predecessor_ids = reinterpret_cast<int *>(out);

  // temp array
  int *last = new int[grid_size]();


  for (int b = 0; b < batch_size; ++b) {
    for (int p = 0; p < grid_size; ++p) {
        last[p] = num_events;
    }
    int i_start = batch_splits[b];
    int i_stop = batch_splits[b+1];
    for (int i = i_start; i < i_stop; ++i) {
      int pixel = pixel_ids[i];
      last[pixel] = i;
      for (auto k = 0; k < kernel_size; ++k) {
        predecessor_ids[i * kernel_size + k] = last[pixel + kernel_offsets[k]];
      }
    }
  }
  int ik_start = batch_splits[batch_size] * kernel_size;
  int ik_stop = num_events * kernel_size;
  for (int ik = ik_start; ik < ik_stop; ++ik) {
    predecessor_ids[ik] = num_events;
  }
  delete[] last;  // Deallocate the memory for temp array
}

void get_permuted_stationary_predecessor_ids(void *out, const void **in) {
  const std::int64_t grid_size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t batch_size = *reinterpret_cast<const std::int64_t *>(in[1]);
  const std::int64_t num_events = *reinterpret_cast<const std::int64_t *>(in[2]);
  const std::int64_t kernel_size = *reinterpret_cast<const std::int64_t *>(in[3]);

  const int *pixel_ids = reinterpret_cast<const int *>(in[4]);
  const int *batch_splits = reinterpret_cast<const int *>(in[5]);
  const int *kernel_offsets = reinterpret_cast<const int *>(in[6]);
  const int *perm_in = reinterpret_cast<const int *>(in[7]);
  const int *perm_out = reinterpret_cast<const int *>(in[8]);

  // output
  int *predecessor_ids = reinterpret_cast<int *>(out);

  // temp array
  int *last = new int[grid_size]();


  for (int b = 0; b < batch_size; ++b) {
    for (int p = 0; p < grid_size; ++p) {
        last[p] = num_events;
    }
    int i_start = batch_splits[b];
    int i_stop = batch_splits[b+1];
    for (int i = i_start; i < i_stop; ++i) {
      int pixel = pixel_ids[i];
      last[pixel] = perm_in[i];
      for (auto k = 0; k < kernel_size; ++k) {
        predecessor_ids[perm_out[i] * kernel_size + k] = last[pixel + kernel_offsets[k]];
      }
    }
  }
  int ik_start = batch_splits[batch_size] * kernel_size;
  int ik_stop = num_events * kernel_size;
  for (int ik = ik_start; ik < ik_stop; ++ik) {
    predecessor_ids[ik] = num_events;
  }
  delete[] last;  // Deallocate the memory for temp array
}

void get_successor_ids(void *out, const void **in) {
  const std::int64_t grid_size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t batch_size = *reinterpret_cast<const std::int64_t *>(in[1]);
  const std::int64_t num_events_in = *reinterpret_cast<const std::int64_t *>(in[2]);
  const std::int64_t num_events_out = *reinterpret_cast<const std::int64_t *>(in[3]);

  const int *pixel_ids_in = reinterpret_cast<const int *>(in[4]);
  const float *times_in = reinterpret_cast<const float *>(in[5]);
  const int *batch_splits_in = reinterpret_cast<const int *>(in[6]);
  const int *pixel_ids_out = reinterpret_cast<const int *>(in[7]);
  const float *times_out = reinterpret_cast<const float *>(in[8]);
  const int *batch_splits_out = reinterpret_cast<const int *>(in[9]);

  // output
  int *successor_ids = reinterpret_cast<int *>(out);

  // temp array
  int *last = new int[grid_size]();

  for (int b = 0; b < batch_size; ++b){
    int e_in_start = batch_splits_in[b];
    int e_in_end = batch_splits_in[b+1];
    int e_out_start = batch_splits_out[b];
    int e_out_end = batch_splits_out[b+1];
    for (int i = 0; i < grid_size; ++i) {
      last[i] = num_events_out;
    }
    int e_out = e_out_end - 1;
    for (int e_in = e_in_end - 1; e_in >= e_in_start; --e_in) {
      float t_in = times_in[e_in];
      while (e_out >= e_out_start && t_in <= times_out[e_out]) {
        last[pixel_ids_out[e_out]] = e_out;
        e_out -= 1;
      }
      successor_ids[e_in] = last[pixel_ids_in[e_in]];
    }
  }
  for (int e_in = batch_splits_in[batch_size]; e_in < num_events_in; ++e_in) {
    successor_ids[e_in] = num_events_out;
  }
  delete[] last;  // Deallocate the memory for temp array
}

void get_permuted_successor_ids(void *out, const void **in) {
  const std::int64_t grid_size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t batch_size = *reinterpret_cast<const std::int64_t *>(in[1]);
  const std::int64_t num_events_in = *reinterpret_cast<const std::int64_t *>(in[2]);
  const std::int64_t num_events_out = *reinterpret_cast<const std::int64_t *>(in[3]);

  const int *pixel_ids_in = reinterpret_cast<const int *>(in[4]);
  const float *times_in = reinterpret_cast<const float *>(in[5]);
  const int *batch_splits_in = reinterpret_cast<const int *>(in[6]);
  const int *perm_in = reinterpret_cast<const int *>(in[7]);
  const int *pixel_ids_out = reinterpret_cast<const int *>(in[8]);
  const float *times_out = reinterpret_cast<const float *>(in[9]);
  const int *batch_splits_out = reinterpret_cast<const int *>(in[10]);
  const int *perm_out = reinterpret_cast<const int *>(in[11]);

  // output
  int *successor_ids = reinterpret_cast<int *>(out);

  // temp array
  int *last = new int[grid_size]();

  for (int b = 0; b < batch_size; ++b){
    int e_in_start = batch_splits_in[b];
    int e_in_end = batch_splits_in[b+1];
    int e_out_start = batch_splits_out[b];
    int e_out_end = batch_splits_out[b+1];
    for (int i = 0; i < grid_size; ++i) {
      last[i] = num_events_out;
    }
    int e_out = e_out_end - 1;
    for (int e_in = e_in_end - 1; e_in >= e_in_start; --e_in) {
      float t_in = times_in[e_in];
      while (e_out >= e_out_start && t_in <= times_out[e_out]) {
        last[pixel_ids_out[e_out]] = perm_out[e_out];
        e_out -= 1;
      }
      successor_ids[perm_in[e_in]] = last[pixel_ids_in[e_in]];
    }
  }
  for (int e_in = batch_splits_in[batch_size]; e_in < num_events_in; ++e_in) {
    successor_ids[e_in] = num_events_out;
  }
  delete[] last;  // Deallocate the memory for temp array
}

template <typename T>
void cumulative_ema(void *out, const void **in) {
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const bool reverse = *reinterpret_cast<const bool *>(in[1]);
  const T *values = reinterpret_cast<const T *>(in[2]);
  const T *factors = reinterpret_cast<const T *>(in[3]);

  // output
  T *output = reinterpret_cast<T *>(out);

  T acc = 0.0;

  if (reverse) {
    for (int i = size - 1; i >= 0; --i) {
      acc *= factors[i];
      acc += values[i];
      output[i] = acc;
      // output[i] = values[i];
    }
  } else {
    for (int i = 0; i < size; ++i) {
      acc *= factors[i];
      acc += values[i];
      output[i] = acc;
      // output[i] = values[i];
    }
  }
}

void _segment_cumulative_ema(
  const std::int64_t size,
  const bool reverse,
  const float* values,
  const float* factors,
  const int *segment_ids,
  float *output)
{
  if (size == 0) return;

  if (reverse) {
    output[size - 1] = values[size - 1];
    int segment = segment_ids[size - 1];
    for (int i = size - 2; i >= 0; --i) {
      int next_segment = segment_ids[i];
      if (segment == next_segment) {
        output[i] = output[i+1] * factors[i] + values[i];
      } else {
        output[i] = values[i];
      }
      segment = next_segment;
    }
  } else {
    output[0] = values[0];
    int segment = segment_ids[0];
    for (int i = 1; i < size; ++i) {
      int next_segment = segment_ids[i];
      if (segment == next_segment) {
        output[i] = output[i-1] * factors[i] + values[i];
      } else {
        output[i] = values[i];
      }
      segment = next_segment;
    }
  }
}

void segment_cumulative_ema(void *out, const void **in) {
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const bool reverse = *reinterpret_cast<const bool *>(in[1]);
  const float *values = reinterpret_cast<const float *>(in[2]);
  const float *factors = reinterpret_cast<const float *>(in[3]);
  const int *segment_ids = reinterpret_cast<const int *>(in[4]);

  // output
  float *output = reinterpret_cast<float *>(out);
  _segment_cumulative_ema(size, reverse, values, factors, segment_ids, output);
}

void tiled_segment_cumulative_ema(void *out, const void **in) {
  const std::int64_t repeats = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[1]);
  const bool reverse = *reinterpret_cast<const bool *>(in[2]);
  const float *values = reinterpret_cast<const float *>(in[3]);
  const float *factors = reinterpret_cast<const float *>(in[4]);
  const int *segment_ids = reinterpret_cast<const int *>(in[5]);

  // output
  float *output = reinterpret_cast<float *>(out);
  for (int r = 0; r < repeats; ++r) {
    _segment_cumulative_ema(
      size, reverse, values + r*size, factors + r*size, segment_ids, output + r*size);
  }
}

template <typename T>
void _serial_segment_cumulative_ema(
    const std::int64_t num_channels,
    const std::int64_t num_events,
    const std::int64_t num_segments,
    const bool reverse,
    const T *values,
    const T* factors,
    const int *splits,
    T* output
  ) {
  for (int c = 0; c < num_channels; ++c) {
    for (int s = 0; s < num_segments; ++s) {
      T acc = T(0);
      int start = splits[s];
      int stop = splits[s+1];
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
}

template <typename T>
void serial_segment_cumulative_ema(void *out, const void **in) {
  // unpack inputs
  const std::int64_t num_channels = *reinterpret_cast<const std::int64_t *>(in[0]);
  const std::int64_t num_events = *reinterpret_cast<const std::int64_t *>(in[1]);
  const std::int64_t num_segments = *reinterpret_cast<const std::int64_t *>(in[2]);
  const bool reverse = *reinterpret_cast<const bool *>(in[3]);
  const T *values = reinterpret_cast<const T *>(in[4]);
  const T *factors = reinterpret_cast<const T *>(in[5]);
  const int *splits = reinterpret_cast<const int *>(in[6]);

  // output
  float *output = reinterpret_cast<float *>(out);

  // _serial_segment_cumulative_ema(
  //   num_channels, num_events, num_segments, reverse, values, factors, splits, output);
  for (int c = 0; c < num_channels; ++c) {
    for (int s = 0; s < num_segments; ++s) {
      T acc = T(0);
      int start = splits[s];
      int stop = splits[s+1];
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
}


pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_counting_argsort"] = EncapsulateFunction(counting_argsort);
  dict["cpu_throttled_sample"] = EncapsulateFunction(throttled_sample);
  dict["cpu_get_stationary_predecessor_ids"] = EncapsulateFunction(get_stationary_predecessor_ids);
  dict["cpu_get_permuted_stationary_predecessor_ids"] = EncapsulateFunction(get_permuted_stationary_predecessor_ids);
  dict["cpu_get_successor_ids"] = EncapsulateFunction(get_successor_ids);
  dict["cpu_get_permuted_successor_ids"] = EncapsulateFunction(get_permuted_successor_ids);
  dict["cpu_cumulative_ema_f32"] = EncapsulateFunction(cumulative_ema<float>);
  dict["cpu_cumulative_ema_f64"] = EncapsulateFunction(cumulative_ema<double>);
  dict["cpu_cumulative_ema_c64"] = EncapsulateFunction(cumulative_ema<std::complex<float>>);
  dict["cpu_cumulative_ema_c128"] = EncapsulateFunction(cumulative_ema<std::complex<double>>);
  dict["cpu_segment_cumulative_ema"] = EncapsulateFunction(segment_cumulative_ema);
  dict["cpu_tiled_segment_cumulative_ema"] = EncapsulateFunction(tiled_segment_cumulative_ema);
  dict["cpu_serial_segment_cumulative_ema_f32"] = EncapsulateFunction(serial_segment_cumulative_ema<float>);
  dict["cpu_serial_segment_cumulative_ema_f64"] = EncapsulateFunction(serial_segment_cumulative_ema<double>);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
