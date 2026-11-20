// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Mahdi Abdollahpour, University of Bologna

#pragma once
#include <stdint.h>

#include "builtins_v2.h"

#if BANKING_FACTOR == 8

// Repeats each block of src reps times in a row: src holds n blocks of len
// values, dst holds n * reps blocks of the same length, and dst block j is src block j / reps.
// Stores are local: each core writes only the slices of dst in its own banks,
// src is loaded again only on a new block. 
// Needs:
// - src word aligned, apart from dst
// - dst at a multiple of 2 * BANKING_FACTOR in a bank row, so a slice stays in
//   one core's banks
// - len a multiple of 2 * BANKING_FACTOR
// - all cores call it, each with its core_id
// In tf: tf.repeat(x, reps, axis=a), n the product of the dims of x up to a,
// len: the product of the the rest of the dims; for x[2][3][4][5][16], axis=2: n = 2 * 3 * 4, len = 5 * 16
static inline void repeat_f16p_local(const __fp16 *__restrict__ src,
                                     __fp16 *__restrict__ dst, uint32_t n,
                                     uint32_t len, uint32_t reps,
                                     uint32_t core_id) {

  const uint32_t step = 2 * NUM_BANKS;
  const uint32_t sb = reps * len;
  const uint32_t total = n * sb;
  const uint32_t off = ((uint32_t)dst / sizeof(__fp16)) % step;
  uint32_t e = (2 * BANKING_FACTOR * core_id + step - off) % step;
  if (e >= total) {
    return;
  }
  // phases: n_per = len / gcd(step, len), step a power of two: halve while even
  uint32_t n_per = len;
  for (uint32_t t = 1; t < step && !(n_per & 1); t <<= 1) {
    n_per >>= 1;
  }
  const uint32_t dd = n_per * step;
  // src of e: (e / sb) * len + e % len
  uint32_t i = e % len, q = e / sb, r = e % sb;
  const uint32_t di = step % len, dq = step / sb, dr = step % sb;
  const uint32_t ddq = dd / sb, ddr = dd % sb;
  const uint32_t dw = dd / 2, lw = len / 2, qw = ddq * lw;
  uint32_t *const end = (uint32_t *)(dst + total);
  uint32_t w0, w1, w2, w3, w4, w5, w6, w7;

  for (uint32_t k = 0; k < n_per && e < total; k++) {
    const uint32_t *s = (const uint32_t *)(src + q * len + i);
    uint32_t *d = (uint32_t *)(dst + e);
    if (n == 1) {
      // one block: stores only
      w0 = s[0];
      w1 = s[1];
      w2 = s[2];
      w3 = s[3];
      w4 = s[4];
      w5 = s[5];
      w6 = s[6];
      w7 = s[7];
      for (; d < end; d += dw) {
        d[0] = w0;
        d[1] = w1;
        d[2] = w2;
        d[3] = w3;
        d[4] = w4;
        d[5] = w5;
        d[6] = w6;
        d[7] = w7;
      }
    } else if (ddq == 0) {
      // a new block every few slices
      uint32_t rr = r;
      w0 = s[0];
      w1 = s[1];
      w2 = s[2];
      w3 = s[3];
      w4 = s[4];
      w5 = s[5];
      w6 = s[6];
      w7 = s[7];
      for (;;) {
        d[0] = w0;
        d[1] = w1;
        d[2] = w2;
        d[3] = w3;
        d[4] = w4;
        d[5] = w5;
        d[6] = w6;
        d[7] = w7;
        d += dw;
        if (d >= end) {
          break;
        }
        rr += ddr;
        if (rr >= sb) {
          rr -= sb;
          s += lw;
          w0 = s[0];
          w1 = s[1];
          w2 = s[2];
          w3 = s[3];
          w4 = s[4];
          w5 = s[5];
          w6 = s[6];
          w7 = s[7];
        }
      }
    } else {
      // a new block every slice
      uint32_t rr = r;
      asm volatile(
          "1:"
          "lw %[w1], 4(%[s]);"
          "lw %[w2], 8(%[s]);"
          "lw %[w3], 12(%[s]);"
          "lw %[w4], 16(%[s]);"
          "lw %[w5], 20(%[s]);"
          "lw %[w6], 24(%[s]);"
          "lw %[w7], 28(%[s]);"
          "p.lw %[w0], %[qs](%[s]!);"
          "add %[rr], %[rr], %[ddr];"
          "bltu %[rr], %[sb], 2f;"
          "sub %[rr], %[rr], %[sb];"
          "add %[s], %[s], %[ls];"
          "2:"
          "sw %[w1], 4(%[d]);"
          "sw %[w2], 8(%[d]);"
          "sw %[w3], 12(%[d]);"
          "sw %[w4], 16(%[d]);"
          "sw %[w5], 20(%[d]);"
          "sw %[w6], 24(%[d]);"
          "sw %[w7], 28(%[d]);"
          "p.sw %[w0], %[ds](%[d]!);"
          "bltu %[d], %[end], 1b;"
          : [w0] "=&r"(w0), [w1] "=&r"(w1), [w2] "=&r"(w2), [w3] "=&r"(w3),
            [w4] "=&r"(w4), [w5] "=&r"(w5), [w6] "=&r"(w6), [w7] "=&r"(w7),
            [s] "+&r"(s), [d] "+&r"(d), [rr] "+&r"(rr)
          : [qs] "r"(qw * sizeof(uint32_t)), [ls] "r"(lw * sizeof(uint32_t)),
            [ds] "r"(dw * sizeof(uint32_t)), [ddr] "r"(ddr), [sb] "r"(sb),
            [end] "r"(end)
          : "memory");
    }
    e += step;
    i += di;
    if (i >= len) {
      i -= len;
    }
    q += dq;
    r += dr;
    if (r >= sb) {
      r -= sb;
      q++;
    }
  }
}

#endif
