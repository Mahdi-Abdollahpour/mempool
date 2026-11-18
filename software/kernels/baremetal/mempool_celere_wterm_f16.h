// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Mahdi Abdollahpour, University of Bologna

#pragma once
#include <stdint.h>

#include "builtins_v2.h"


// Window term of the Doppler head, y[e][r] += a * ys[e][r / win][r % span]
// Needs y at the start of a row of banks for locality
// No barrier: the caller syncs.
// y is processed locally, and core based offsets lower bank traffics for ys
#if BANKING_FACTOR == 8

static inline __attribute__((always_inline)) void
celere_wterm_f16(uint32_t a, __fp16 *ys, __fp16 *in_y, uint32_t n_el,
                 uint32_t n_prb, uint32_t win, uint32_t span, uint32_t width) {

  const uint32_t core_id = mempool_get_core_id();
  const uint32_t len = n_el * n_prb * width;
  const uint32_t i0 = 2 * core_id * BANKING_FACTOR;
  if (i0 >= len) {
    return;
  }
  const uint32_t n_sum = (n_prb + win - 1) / win * span;
  const uint32_t n_step = (len - i0 + 2 * NUM_BANKS - 1) / (2 * NUM_BANKS);
  // the core's first slice
  const uint32_t m0 =
      (2 * (i0 / width) + 3 * (core_id % NUM_CORES_PER_TILE)) % n_step;

  // a step moves dr rows, x moves by whole windows
  const uint32_t dr = 2 * NUM_BANKS / width;
  const uint32_t win_x = span * width;
  const uint32_t dr_rw = dr % win;
  const uint32_t step_x = dr / win * win_x;
  const uint32_t np_rw = n_prb % win;
  const uint32_t np_g = n_prb % span;
  const int32_t next_el =
      (int32_t)((n_sum - n_prb / win * span - np_g) * width);

  uint32_t aa = (a << 16U) | a;
  v2h x01, x23, x45, x67, x89, xab, xcd, xef;
  v2h y0, y1, y2, y3;
  uint32_t i = i0 + m0 * 2 * NUM_BANKS;
  uint32_t i_end = len;
  // from the first slice to the end, then the slices before it
  for (uint32_t pass = 0; pass < 2; pass++) {
   
    const uint32_t row = i / width;
    const uint32_t e = row / n_prb;
    const uint32_t r = row - e * n_prb;
    uint32_t rw = r % win, g = r % span;
    __fp16 *px = ys + (e * n_sum + r / win * span + g) * width + i % width;
    __fp16 *py = in_y + i;
    // where the next element and the pass end in y
    __fp16 *py_el = in_y + (e + 1) * n_prb * width;
    __fp16 *const py_end = in_y + i_end;
    while (py < py_end) {
   
      asm volatile(
          "1:"
          "lw %[x23], 4(%[px]);"
          "lw %[x45], 8(%[px]);"
          "lw %[x67], 12(%[px]);"
          "lw %[x89], 16(%[px]);"
          // y words 1 to 4
          "lw %[y0], 4(%[py]);"
          "lw %[y1], 8(%[py]);"
          "lw %[y2], 12(%[py]);"
          "lw %[y3], 16(%[py]);"
          // the last load steps x to the next slice
          "lw %[xab], 20(%[px]);"
          "lw %[xcd], 24(%[px]);"
          "lw %[xef], 28(%[px]);"
          "p.lw %[x01], %[dx](%[px]!);"
          //
          "add %[rw], %[rw], %[dr_rw];"
          "bltu %[rw], %[win], 2f;"
          "sub %[rw], %[rw], %[win];"
          "add %[px], %[px], %[wx];"
          "2:"
          "vfmac.h %[y0], %[x23], %[aa];"
          "vfmac.h %[y1], %[x45], %[aa];"
          "vfmac.h %[y2], %[x67], %[aa];"
          "vfmac.h %[y3], %[x89], %[aa];"
          "sw %[y0], 4(%[py]);"
          "sw %[y1], 8(%[py]);"
          "sw %[y2], 12(%[py]);"
          "sw %[y3], 16(%[py]);"
          // y words 5 to 7 and 0, the last store steps y to the next slice
          "lw %[y0], 20(%[py]);"
          "lw %[y1], 24(%[py]);"
          "lw %[y2], 28(%[py]);"
          "lw %[y3], 0(%[py]);"
          "vfmac.h %[y0], %[xab], %[aa];"
          "vfmac.h %[y1], %[xcd], %[aa];"
          "vfmac.h %[y2], %[xef], %[aa];"
          "vfmac.h %[y3], %[x01], %[aa];"
          "sw %[y0], 20(%[py]);"
          "sw %[y1], 24(%[py]);"
          "sw %[y2], 28(%[py]);"
          "p.sw %[y3], %[dy](%[py]!);"
          "bgeu %[py], %[el], 3f;"
          "bltu %[py], %[end], 1b;"
          "3:"
          : [x01] "=&r"(x01), [x23] "=&r"(x23), [x45] "=&r"(x45),
            [x67] "=&r"(x67), [x89] "=&r"(x89), [xab] "=&r"(xab),
            [xcd] "=&r"(xcd), [xef] "=&r"(xef), [y0] "=&r"(y0), [y1] "=&r"(y1),
            [y2] "=&r"(y2), [y3] "=&r"(y3), [px] "+&r"(px), [py] "+&r"(py),
            [rw] "+&r"(rw)
          : [dx] "r"(step_x * sizeof(__fp16)), [wx] "r"(win_x * sizeof(__fp16)),
            [dy] "r"(2 * NUM_BANKS * sizeof(__fp16)), [dr_rw] "r"(dr_rw),
            [win] "r"(win), [aa] "r"(aa), [el] "r"(py_el), [end] "r"(py_end)
          : "memory");
      // the next element
      while (py >= py_el) {
        py_el += n_prb * width;
        px += next_el;
        if (rw < np_rw) {
          rw += win;
          px -= win_x;
        }
        rw -= np_rw;
        if (g < np_g) {
          g += span;
          px += win_x;
        }
        g -= np_g;
      }
    }
    i = i0;
    i_end = i0 + m0 * 2 * NUM_BANKS;
  }
  return;
}

#endif
