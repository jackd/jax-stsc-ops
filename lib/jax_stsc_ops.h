// This header defines the actual algorithm for our op. It is reused in cpu_ops.cc and
// kernels.cc.cu to expose this as a XLA custom call. The details aren't too important
// except that directly implementing this algorithm as a higher-level JAX function
// probably wouldn't be very efficient. That being said, this is not meant as a
// particularly efficient or robust implementation. It's just here to demonstrate the
// infrastructure required to extend JAX.

#ifndef _JAX_STSC_OPS_H_
#define _JAX_STSC_OPS_H_

#include <cmath>

namespace jax_stsc_ops {

#ifdef __CUDACC__
#define JAX_STSC_OPS_INLINE_OR_DEVICE __host__ __device__
#else
#define JAX_STSC_OPS_INLINE_OR_DEVICE inline
#endif

}  // namespace JAX_STSC_OPS

#endif