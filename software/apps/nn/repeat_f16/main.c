// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Mahdi Abdollahpour, University of Bologna

#include <stdint.h>
#include <string.h>

#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_repeat_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_repeat_f16.h"

__fp16 buf[BUF_N]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);

  if (core_id == 0) {
    dma_memcpy_blocking(buf, l2_buf, BUF_N * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

#if BANKING_FACTOR == 8
  // a bias row over the rows of a GEMM output
  mempool_start_benchmark();
  repeat_f16p_local(buf + r1_SRC, buf + r1_DST, r1_N, r1_LEN, r1_REPS, core_id);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  // rows repeated
  mempool_start_benchmark();
  repeat_f16p_local(buf + r2_SRC, buf + r2_DST, r2_N, r2_LEN, r2_REPS, core_id);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  // a block of rows repeated
  mempool_start_benchmark();
  repeat_f16p_local(buf + r3_SRC, buf + r3_DST, r3_N, r3_LEN, r3_REPS, core_id);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  // small blocks repeated
  mempool_start_benchmark();
  repeat_f16p_local(buf + r4_SRC, buf + r4_DST, r4_N, r4_LEN, r4_REPS, core_id);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  // two rows over many rows
  mempool_start_benchmark();
  repeat_f16p_local(buf + r5_SRC, buf + r5_DST, r5_N, r5_LEN, r5_REPS, core_id);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();
#endif

  mempool_check_f16(buf, l2_gold, BUF_N, 0.01f, 1);
  mempool_barrier(num_cores);
  return 0;
}
