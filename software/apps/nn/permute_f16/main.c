// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* permute_f16p driven by CELERE's front-end unpack (template row 3): six
 * levels, which is PERMUTE_F16_MAX_LEVELS.
 *
 * THE LEVEL TABLE COMES FROM THE DATA HEADER, so the same table made the golden
 * and drives the call -- the test cannot pass by agreeing with itself.  The map
 * is a dense bijection at both ends, so a wrong stride has nowhere harmless to
 * land: it must overwrite an element another run owns. */

#include <stdint.h>

#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_permute_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_permute_f16.h"

__fp16 perm_src[array_N]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 perm_dst[array_N]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  /* The header carries the levels as int32 triples -- print_array writes flat
   * typed arrays -- so the struct is filled here.  It is a handful of loads on
   * every core, outside the timed region. */
  perm_level_t lv[perm_NLEV];
  for (uint32_t k = 0; k < perm_NLEV; k++) {
    lv[k].n = (uint32_t)l2_levels[3 * k];
    lv[k].src_step = l2_levels[3 * k + 1];
    lv[k].dst_step = l2_levels[3 * k + 2];
  }

  if (core_id == 0) {
    dma_memcpy_blocking(perm_src, l2_A, array_N * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  mempool_start_benchmark();
  permute_f16p(perm_src, perm_dst, perm_RUN, lv, perm_NLEV, core_id, num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(perm_dst, l2_B, array_N, 0.01f, 0);
  mempool_barrier(num_cores);
  return 0;
}
