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

#include "data_celere_wterm_f16.h"

#include "baremetal/mempool_celere_wterm_f16.h"
#include "baremetal/mempool_checks.h"

#define Y_N (wterm_EL * wterm_PRB * wterm_C)
#define H_N (Y_N + wterm_EL * wterm_NSUM * wterm_C)

// [y | ys] as h in the app, y at the start of a row of banks
__fp16 h[H_N]
    __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  mempool_barrier_init(core_id);

  if (core_id == 0) {
    dma_memcpy_blocking(h, l2_h, H_N * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

#if BANKING_FACTOR == 8
  mempool_start_benchmark();
  celere_wterm_f16(0x3C00, h + Y_N, h, wterm_EL, wterm_PRB, wterm_WIN,
                   wterm_SPAN, wterm_C);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();
#endif

  mempool_check_f16(h, l2_gold, H_N, 0.01f, 1);
  mempool_barrier(num_cores);
  return 0;
}
