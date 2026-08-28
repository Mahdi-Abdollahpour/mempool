// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* transpose_f16p at CEViT's row-13 shape: K [1092, 32] -> K^T.
 *
 * TWO CALLS, BECAUSE THE KERNEL HAS TWO BRANCHES: it splits rows when
 * rows >= numThreads and columns otherwise.  At 256 cores the first call takes
 * one branch and the second the other, and since the second transposes the
 * first's output it is a round trip back to the input. */

#include <stdint.h>

#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_transpose_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_transpose_f16.h"

__fp16 matrix_a[matrix_M * matrix_N]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_b[matrix_M * matrix_N]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 matrix_c[matrix_M * matrix_N]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    dma_memcpy_blocking(matrix_a, l2_A,
                        (matrix_M * matrix_N) * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // [M, N] -> [N, M].  M = 1092 >= num_cores: the ROW-split branch.
  mempool_start_benchmark();
  transpose_f16p(matrix_a, matrix_b, matrix_M, matrix_N, core_id, num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(matrix_b, l2_B, matrix_M * matrix_N, 0.01f, 0);
  mempool_barrier(num_cores);

  // Back again: [N, M] -> [M, N].  N = 32 < num_cores, so this one takes the
  // COLUMN-split branch, and the result must be the input.
  mempool_start_benchmark();
  transpose_f16p(matrix_b, matrix_c, matrix_N, matrix_M, core_id, num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(matrix_c, l2_A, matrix_M * matrix_N, 0.01f, 0);
  mempool_barrier(num_cores);
  return 0;
}
