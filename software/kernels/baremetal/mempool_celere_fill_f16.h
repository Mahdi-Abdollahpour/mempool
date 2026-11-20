// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Mahdi Abdollahpour, University of Bologna

#pragma once
#include <stdint.h>

#include "builtins_v2.h"



// Bias fills of a GEMM output, before its TE job. No barrier inside
#if BANKING_FACTOR == 8

// dst[i] = row[i % width], or 0 if row is NULL
// width and the offset of dst in a bank row: multiples of 2 * BANKING_FACTOR
static inline void celere_fill_rows_f16(__fp16 *dst, const __fp16 *row,
                                        uint32_t width, uint32_t n_rows) {

  const uint32_t core_id = mempool_get_core_id();
  const uint32_t step = 2 * NUM_BANKS;
  const uint32_t len = n_rows * width;
  // first slice in the core's banks
  const uint32_t off = ((uint32_t)dst / sizeof(__fp16)) % step;
  const uint32_t e0 = (2 * BANKING_FACTOR * core_id + step - off) % step;
  // slice columns repeat every n_per slices
  uint32_t a = step, b = width;
  while (b) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  const uint32_t n_per = width / a;

  for (uint32_t k = 0; k < n_per; k++) {
    uint32_t e = e0 + k * step;
    if (e >= len) {
      break;
    }
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
    if (row) {
      const uint32_t *r = (const uint32_t *)(row + e % width);
      w0 = r[0];
      w1 = r[1];
      w2 = r[2];
      w3 = r[3];
      w4 = r[4];
      w5 = r[5];
      w6 = r[6];
      w7 = r[7];
    }
    for (; e < len; e += n_per * step) {
      uint32_t *d = (uint32_t *)(dst + e);
      d[0] = w0;
      d[1] = w1;
      d[2] = w2;
      d[3] = w3;
      d[4] = w4;
      d[5] = w5;
      d[6] = w6;
      d[7] = w7;
    }
  }
  return;
}

// Window term, y[e][r] = ys[e][r / win][r % span], y at the start of a bank row
// y is processed locally, and core based offsets lower bank traffics for ys
static inline __attribute__((always_inline)) void
celere_fill_wterm_f16(__fp16 *ys, __fp16 *in_y, uint32_t n_el, uint32_t n_prb,
                      uint32_t win, uint32_t span, uint32_t width) {

  const uint32_t core_id = mempool_get_core_id();
  const uint32_t len = n_el * n_prb * width;
  const uint32_t i0 = 2 * core_id * BANKING_FACTOR;
  if (i0 >= len) {
    return;
  }
  const uint32_t n_sum = (n_prb + win - 1) / win * span;
  const uint32_t n_step = (len - i0 + 2 * NUM_BANKS - 1) / (2 * NUM_BANKS);
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

  v2h x01, x23, x45, x67, x89, xab, xcd, xef;
  uint32_t i = i0 + m0 * 2 * NUM_BANKS;
  uint32_t i_end = len;
  // first slice to the end, then the slices before it
  for (uint32_t pass = 0; pass < 2; pass++) {

    const uint32_t row = i / width;
    const uint32_t e = row / n_prb;
    const uint32_t r = row - e * n_prb;
    uint32_t rw = r % win, g = r % span;
    __fp16 *px = ys + (e * n_sum + r / win * span + g) * width + i % width;
    __fp16 *py = in_y + i;
    __fp16 *py_el = in_y + (e + 1) * n_prb * width;
    __fp16 *const py_end = in_y + i_end;
    while (py < py_end) {

      asm volatile(
          "1:"
          "lw %[x23], 4(%[px]);"
          "lw %[x45], 8(%[px]);"
          "lw %[x67], 12(%[px]);"
          "lw %[x89], 16(%[px]);"
          "lw %[xab], 20(%[px]);"
          "lw %[xcd], 24(%[px]);"
          "lw %[xef], 28(%[px]);"
          "p.lw %[x01], %[dx](%[px]!);"
          "add %[rw], %[rw], %[dr_rw];"
          "bltu %[rw], %[win], 2f;"
          "sub %[rw], %[rw], %[win];"
          "add %[px], %[px], %[wx];"
          "2:"
          "sw %[x23], 4(%[py]);"
          "sw %[x45], 8(%[py]);"
          "sw %[x67], 12(%[py]);"
          "sw %[x89], 16(%[py]);"
          "sw %[xab], 20(%[py]);"
          "sw %[xcd], 24(%[py]);"
          "sw %[xef], 28(%[py]);"
          "p.sw %[x01], %[dy](%[py]!);"
          "bgeu %[py], %[el], 3f;"
          "bltu %[py], %[end], 1b;"
          "3:"
          : [x01] "=&r"(x01), [x23] "=&r"(x23), [x45] "=&r"(x45),
            [x67] "=&r"(x67), [x89] "=&r"(x89), [xab] "=&r"(xab),
            [xcd] "=&r"(xcd), [xef] "=&r"(xef), [px] "+&r"(px), [py] "+&r"(py),
            [rw] "+&r"(rw)
          : [dx] "r"(step_x * sizeof(__fp16)), [wx] "r"(win_x * sizeof(__fp16)),
            [dy] "r"(2 * NUM_BANKS * sizeof(__fp16)), [dr_rw] "r"(dr_rw),
            [win] "r"(win), [el] "r"(py_el), [end] "r"(py_end)
          : "memory");
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
