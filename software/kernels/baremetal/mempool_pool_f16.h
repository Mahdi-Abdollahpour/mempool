// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "builtins_v2.h"

/* Mean-pool a strided sub-range of one axis:
 *
 *     out[w, j] = (1/W) * SUM_{s in window w} in[s, j]      j in [0, lane)
 *
 * THIS IS NOT A PERMUTATION AND IS NOT WRITTEN LIKE ONE.  It reduces the WINDOW
 * axis only; every one of the `lane` elements beside it is independent AND
 * CONTIGUOUS -- in CELERE `lane` is i_3 * i_4 = 16 * 384 = 6144 halfwords -- so the
 * reduction is a straight vector accumulate, `vfadd.h` over v2h, two elements per
 * instruction, exactly as the other element-wise kernels here run.  Nothing is
 * gathered and no address is strided by less than a whole row.
 *
 * THE DIVISOR IS ALWAYS `win`, INCLUDING FOR A SHORT FINAL WINDOW.  The model pools
 * a zero-PADDED block, so the last window is (1/W) * SUM over however many live rows
 * it has -- 5 of 12 at 273 PRB.  That fixed divisor is exactly what lets this kernel
 * SKIP the pad rows rather than read zeros: the two are the same number.  A masked
 * mean -- dividing by the live count -- would be a different function, and the
 * template flags it as a variant this is deliberately not.
 *
 * Cores split the LANE axis on the BANKING_FACTOR interleave the other element-wise
 * kernels use, so each core's 2*BANKING_FACTOR run sits in its own banks.  The
 * window axis is walked by every core, which is what keeps the accumulator in
 * registers: one pass over the input, no scratch.
 *
 * `n_in` is the live row count, not the padded one.  Ending without a barrier is
 * the convention here; the caller rendezvouses.
 */

/* Accumulate one v2h row into the four running sums.  `p` walks the window. */
#define POOL_F16VEC_UNROLLED4_ACC                                              \
  {                                                                            \
    v2h x0 = *(v2h *)&p[0];                                                    \
    v2h x1 = *(v2h *)&p[2];                                                    \
    v2h x2 = *(v2h *)&p[4];                                                    \
    v2h x3 = *(v2h *)&p[6];                                                    \
    asm volatile("vfadd.h %[a0], %[a0], %[x0];"                                \
                 "vfadd.h %[a1], %[a1], %[x1];"                                \
                 "vfadd.h %[a2], %[a2], %[x2];"                                \
                 "vfadd.h %[a3], %[a3], %[x3];"                                \
                 : [a0] "+&r"(a0), [a1] "+&r"(a1), [a2] "+&r"(a2),             \
                   [a3] "+&r"(a3)                                              \
                 : [x0] "r"(x0), [x1] "r"(x1), [x2] "r"(x2), [x3] "r"(x3));    \
  }

void pool_mean_f16p(const __fp16 *__restrict__ src, __fp16 *__restrict__ dst,
                    uint32_t n_in, uint32_t n_out, uint32_t win, uint32_t lane,
                    uint32_t core_id, uint32_t numThreads) {

  // 1/win, broadcast into both halves once -- the same idiom layernorm uses for
  // its 1/N.  The short final window divides by this too; see the header note.
  float invW = 1.0f / (float)win;
  v2h vInvW;
  asm volatile("vfcpka.h.s %[vInvW], %[invW], %[invW];"
               : [vInvW] "+&r"(vInvW)
               : [invW] "r"(invW));

  for (uint32_t w = 0; w < n_out; w++) {
    uint32_t s0 = w * win;
    uint32_t s1 = s0 + win;
    if (s1 > n_in) {
      s1 = n_in; // short final window: fewer rows, same divisor
    }
    const __fp16 *base = src + (uint32_t)s0 * lane;
    __fp16 *out = dst + (uint32_t)w * lane;

    for (uint32_t i = 2 * BANKING_FACTOR * core_id; i < lane;
         i += 2 * BANKING_FACTOR * numThreads) {
      /* 2*BANKING_FACTOR elements per core per pass, in blocks of 8 -- the whole
       * run, not half of it. */
      for (uint32_t b = 0; b < BANKING_FACTOR / 4; b++) {
        uint32_t o = i + b * 8;
        v2h a0 = (v2h)0.0f, a1 = (v2h)0.0f, a2 = (v2h)0.0f, a3 = (v2h)0.0f;
        const __fp16 *p = base + o;
        for (uint32_t s = s0; s < s1; s++) {
          POOL_F16VEC_UNROLLED4_ACC;
          p += lane;
        }
        asm volatile("vfmul.h %[a0], %[a0], %[m];"
                     "vfmul.h %[a1], %[a1], %[m];"
                     "vfmul.h %[a2], %[a2], %[m];"
                     "vfmul.h %[a3], %[a3], %[m];"
                     : [a0] "+&r"(a0), [a1] "+&r"(a1), [a2] "+&r"(a2),
                       [a3] "+&r"(a3)
                     : [m] "r"(vInvW));
        (*(v2h *)&out[o + 0]) = a0;
        (*(v2h *)&out[o + 2]) = a1;
        (*(v2h *)&out[o + 4]) = a2;
        (*(v2h *)&out[o + 6]) = a3;
      }
    }
  }
}
