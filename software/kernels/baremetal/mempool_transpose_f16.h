// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "builtins_v2.h"

/* Out-of-place transpose of an fp16 matrix: dst[c][r] = src[r][c].
 *
 * A PERMUTATION BELONGS ON THE CORES, NOT ON THE DMA.  The iDMA moves whole TCDM
 * lines at one line per cycle, and a line here is tcdm_width = cores * banking * 4
 * bytes -- 8 KiB or 16 KiB.  A transposing descriptor has to walk one ELEMENT per
 * repetition, so it would spend a whole line-cycle to move two bytes: transposing
 * [1092, 32] costs 34,944 cycles that way.  The cores address the TCDM by word and
 * there are hundreds of them, so the same work is a few dozen iterations each.
 *
 * ONE END OF THE COPY IS ALWAYS CONTIGUOUS AND THE OTHER ALWAYS STRIDED -- that is
 * what a transpose IS -- so both pointers are walked by the memory ops themselves,
 * p.lhu/p.sh with post-increment, and the loop body carries no address arithmetic at
 * all.  The eight loads issue before the first store so the load-use latency of one
 * is covered by the next, the same reason mempool_cmatmul_f16.h interleaves its
 * p.lw with the FMAs.
 *
 * THE `!` IS LOAD-BEARING.  `p.lhu %[v], %[s](%[p]!)` is the post-increment encoding;
 * written without it the assembler takes the register-offset form that does NOT write
 * the address back, the pointer never moves, and the "+&r" constraint hides it from
 * the compiler.  (transpose_K in software/apps/nn/multihead_f16/main.c omits it and is
 * broken for exactly this reason -- do not copy it.)
 *
 * The two halves differ only in which axis is split, and the split is chosen so
 * every core gets work whichever way the matrix is shaped.  Neither ends in a
 * barrier: the caller rendezvouses, like every other kernel here.
 */

/* Eight elements: load the run, then store it.  `ls` and `ds` are the post-increment
 * strides in BYTES -- one of the two is always 2, the contiguous end. */
#define TRANSPOSE_F16_UNROLLED8_LOOP                                           \
  {                                                                            \
    asm volatile("p.lhu %[v0], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v1], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v2], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v3], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v4], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v5], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v6], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v7], %[ls](%[s]!) \n\t"                              \
                 "p.sh  %[v0], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v1], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v2], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v3], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v4], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v5], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v6], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v7], %[ds](%[d]!) \n\t"                              \
                 : [v0] "=&r"(v0), [v1] "=&r"(v1), [v2] "=&r"(v2),             \
                   [v3] "=&r"(v3), [v4] "=&r"(v4), [v5] "=&r"(v5),             \
                   [v6] "=&r"(v6), [v7] "=&r"(v7), [s] "+&r"(s), [d] "+&r"(d)  \
                 : [ls] "r"(ls), [ds] "r"(ds)                                  \
                 : "memory");                                                  \
  }

/* One element, for the tail `rows`/`cols` leaves when it is not a multiple of 8. */
#define TRANSPOSE_F16_SINGLE                                                   \
  {                                                                            \
    asm volatile("p.lhu %[v0], %[ls](%[s]!) \n\t"                              \
                 "p.sh  %[v0], %[ds](%[d]!) \n\t"                              \
                 : [v0] "=&r"(v0), [s] "+&r"(s), [d] "+&r"(d)                  \
                 : [ls] "r"(ls), [ds] "r"(ds)                                  \
                 : "memory");                                                  \
  }

void transpose_f16p(const __fp16 *__restrict__ src, __fp16 *__restrict__ dst,
                    uint32_t rows, uint32_t cols, uint32_t core_id,
                    uint32_t numThreads) {

  uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
  const uint32_t elem_b = (uint32_t)sizeof(__fp16);

  if (rows >= numThreads) {
    // A core owns whole ROWS: contiguous reads, writes strided by `rows`.
    const uint32_t ls = elem_b, ds = rows * elem_b;
    for (uint32_t r = core_id; r < rows; r += numThreads) {
      const __fp16 *s = src + r * cols;
      __fp16 *d = dst + r;
      for (uint32_t c = cols >> 3; c > 0; c--) {
        TRANSPOSE_F16_UNROLLED8_LOOP;
      }
      for (uint32_t c = cols & 7u; c > 0; c--) {
        TRANSPOSE_F16_SINGLE;
      }
    }
  } else {
    // Fewer rows than cores; splitting them would idle most of the cluster, so
    // split the COLUMNS instead -- strided reads, contiguous writes.
    const uint32_t ls = cols * elem_b, ds = elem_b;
    for (uint32_t c = core_id; c < cols; c += numThreads) {
      const __fp16 *s = src + c;
      __fp16 *d = dst + c * rows;
      for (uint32_t r = rows >> 3; r > 0; r--) {
        TRANSPOSE_F16_UNROLLED8_LOOP;
      }
      for (uint32_t r = rows & 7u; r > 0; r--) {
        TRANSPOSE_F16_SINGLE;
      }
    }
  }
}
