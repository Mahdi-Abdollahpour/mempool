// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* patchify_f16p and unpatchify_f16p at CEViT's shape: a [3276, 14, 2] grid
 * <-> 1092 token rows of a 12 x 7 patch.
 *
 * Three checks: the whole grid against the golden, unpatchify back to it, and
 * THE SECOND LANE'S SLAB at tok0 = 546, f0 = 1632, whose 546 rows must equal
 * rows 546..1091 of the same golden.  The third is the one worth having: 546/4
 * is not an integer, so that slab does not start on a subcarrier block and the
 * kernel must decompose (c, tb, fb) from a token index its own slab does not
 * begin at.  With tok0 = f0 = 0 every offset is zero and nothing tests it. */

#include <stdint.h>

#include "builtins_v2.h"
#include "dma.h"
#include "encoding.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_patchify_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_patchify_f16.h"

#define GRID_ELEMS (grid_F * grid_T * grid_C)
#define TOK_ELEMS (patch_NTOK * patch_P)

/* Tokens one subcarrier block contributes: (Nt/Nw) symbol blocks x C channels.
 */
#define TOK_PER_FB ((grid_T / patch_W) * grid_C)
/* The second of two lanes, exactly as cevit_pipeline.py places it. */
#define LANE1_NTOK (patch_NTOK / 2)
#define LANE1_TOK0 (patch_NTOK / 2)
#define LANE1_F0 ((LANE1_TOK0 / TOK_PER_FB) * patch_H)
#define LANE1_SRC (LANE1_F0 * grid_T * grid_C)
#define LANE1_DST (LANE1_TOK0 * patch_P)

__fp16 grid[GRID_ELEMS]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 patched[TOK_ELEMS]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));
__fp16 regrid[GRID_ELEMS]
    __attribute__((aligned(sizeof(int32_t)), section(".l1_prio")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    dma_memcpy_blocking(grid, l2_A, GRID_ELEMS * sizeof(int16_t));
  }
  mempool_barrier(num_cores);

  // 1. The whole grid, one cluster owning every token.
  mempool_start_benchmark();
  patchify_f16p(grid, patched, 0, 0, patch_NTOK, grid_T, patch_H, patch_W,
                grid_C, core_id, num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(patched, l2_B, TOK_ELEMS, 0.01f, 0);
  mempool_barrier(num_cores);

  // 2. And back: the inverse map must reproduce the grid it came from.
  mempool_start_benchmark();
  unpatchify_f16p(patched, regrid, 0, 0, patch_NTOK, grid_T, patch_H, patch_W,
                  grid_C, core_id, num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(regrid, l2_A, GRID_ELEMS, 0.01f, 0);
  mempool_barrier(num_cores);

  // 3. The second lane's slab.  Clear its half of the destination first, or
  // check 1 would have left the right answer there and this would pass without
  // running.
  for (uint32_t i = LANE1_DST + core_id; i < TOK_ELEMS; i += num_cores) {
    patched[i] = (__fp16)0.0f;
  }
  mempool_barrier(num_cores);

  mempool_start_benchmark();
  patchify_f16p(grid + LANE1_SRC, patched + LANE1_DST, LANE1_TOK0, LANE1_F0,
                LANE1_NTOK, grid_T, patch_H, patch_W, grid_C, core_id,
                num_cores);
  mempool_barrier(num_cores);
  mempool_stop_benchmark();

  mempool_check_f16(patched + LANE1_DST, l2_B + LANE1_DST, LANE1_NTOK * patch_P,
                    0.01f, 0);
  mempool_barrier(num_cores);
  return 0;
}
