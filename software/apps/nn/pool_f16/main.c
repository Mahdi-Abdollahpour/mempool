// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* pool_mean_f16p at CELERE's row-5 shape: 137 spans pooled in windows of 12.
 *
 * THE SHORT FINAL WINDOW IS THE POINT.  137 = 11*12 + 5, and the kernel divides
 * that last window by 12 all the same -- the model pools a zero-padded block,
 * so skipping the pad rows is only equivalent to reading them while the divisor
 * stays fixed.  lane = 3072 is a multiple of 2*BANKING_FACTOR and under
 * 2*BANKING_FACTOR*num_cores, so the last cores idle, as in the pipeline. */

#include <stdint.h>

#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_pool_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_pool_f16.h"

__fp16 pool_in[pool_NIN * pool_LANE]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 pool_out[pool_NOUT * pool_LANE]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    dma_memcpy_blocking(pool_in, l2_A,
                        (pool_NIN * pool_LANE) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  mempool_start_benchmark();
  pool_mean_f16p(pool_in, pool_out, pool_NIN, pool_NOUT, pool_WIN, pool_LANE,
                 core_id, num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(pool_out, l2_B, pool_NOUT * pool_LANE, 0.01f, 0);
  mempool_barrier(num_cores);
  return 0;
}
