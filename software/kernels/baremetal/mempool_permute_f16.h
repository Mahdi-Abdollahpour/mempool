// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "builtins_v2.h"

/* A LEVELLED STRIDED COPY: one kernel for every index re-map whose innermost axis
 * is a contiguous run.
 *
 *     for l3 in [0, n3): for l2 in [0, n2): for l1 in [0, n1):
 *         copy `run` contiguous elements
 *         src += that level's src_step, dst += its dst_step
 *
 * WHY ONE KERNEL AND NOT ONE PER ROW.  CELERE's in-stage permutations differ only in
 * their extents and strides -- the head-gather, the local fold and the per-PRB
 * re-pack are the same loop with different numbers -- so they are DATA, not code.
 * The generator emits a level table per call site and this walks it.
 *
 * WHY A KERNEL AND NOT A DESCRIPTOR.  These runs are 48 to 128 halfwords, 96 to
 * 256 bytes.  The iDMA emits one burst per repetition and the TCDM back-end issues
 * one LINE per cycle, where a line is cores * banking * 4 bytes -- 8 KiB or 16 KiB --
 * so a 256 B run would spend a whole line-cycle to move 1/32 of a line.  The cores
 * address the TCDM by word and there are hundreds of them.
 *
 * THE RUN CARRIES THE COST, so the loop scaffolding barely matters: at run = 128 the
 * four levels' counters and carries are under a tenth of the work.  That is why this
 * can afford to be general where mempool_patchify_f16.h, whose run is Nw = 7, could
 * not.
 *
 * The innermost run is the same p.lhu/p.sh post-increment block
 * mempool_transpose_f16.h uses -- see its header for why the `!` is load-bearing.
 * Both pointers here are CONTIGUOUS within a run, so both strides are one halfword
 * and the level table carries every jump.
 *
 * Cores take a CONTIGUOUS CHUNK of the flattened level space, so the walk is pure
 * increment and carry: the divisions that decompose a chunk's first index are paid
 * once, on entry.  No barrier at the end; the caller rendezvouses.
 */

/* SIX, because the front-end unpack needs six.  Row 3 composes three
 * factorizations -- the antenna plane into g_v x g_h tiles, the tile into its own
 * two axes, and the subcarrier axis into spans of span_prbs PRBs -- and a 2-D tile
 * index over a 2-D plane cannot be one uniform stride, so each half of each pair is
 * its own level: (pilot-in-PRB run) x kh x kv x prb x bh x bv x span.  Every other
 * row here uses three or four and leaves the rest as single passes, which cost one
 * compare each. */
#define PERMUTE_F16_MAX_LEVELS 6

/* One level of the walk.  `n` is its extent, and the two steps are how far the
 * source and destination move when it advances -- in ELEMENTS, not bytes, and
 * signed, because a level that unfolds an axis can walk backwards. */
typedef struct {
  uint32_t n;
  int32_t src_step;
  int32_t dst_step;
} perm_level_t;

/* Eight elements of a run: load them, then store them, so one load's use latency is
 * covered by the next.  Both ends contiguous, so both post-increments are 2 bytes. */
#define PERMUTE_F16_UNROLLED8_LOOP                                             \
  {                                                                            \
    asm volatile("p.lhu %[v0], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v1], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v2], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v3], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v4], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v5], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v6], 2(%[s]!) \n\t"                                  \
                 "p.lhu %[v7], 2(%[s]!) \n\t"                                  \
                 "p.sh  %[v0], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v1], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v2], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v3], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v4], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v5], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v6], 2(%[d]!) \n\t"                                  \
                 "p.sh  %[v7], 2(%[d]!) \n\t"                                  \
                 : [v0] "=&r"(v0), [v1] "=&r"(v1), [v2] "=&r"(v2),             \
                   [v3] "=&r"(v3), [v4] "=&r"(v4), [v5] "=&r"(v5),             \
                   [v6] "=&r"(v6), [v7] "=&r"(v7), [s] "+&r"(s), [d] "+&r"(d)  \
                 : : "memory");                                                \
  }

#define PERMUTE_F16_SINGLE                                                     \
  {                                                                            \
    asm volatile("p.lhu %[v0], 2(%[s]!) \n\t"                                  \
                 "p.sh  %[v0], 2(%[d]!) \n\t"                                  \
                 : [v0] "=&r"(v0), [s] "+&r"(s), [d] "+&r"(d)                  \
                 : : "memory");                                                \
  }

void permute_f16p(const __fp16 *__restrict__ src, __fp16 *__restrict__ dst,
                  uint32_t run, const perm_level_t *lv, uint32_t nlv,
                  uint32_t core_id, uint32_t numThreads) {

  uint32_t v0, v1, v2, v3, v4, v5, v6, v7;

  /* The level table, hoisted into locals: it is read once per call, not once per
   * run, and a level the caller did not fill is a single pass at zero stride. */
  uint32_t n[PERMUTE_F16_MAX_LEVELS];
  int32_t ss[PERMUTE_F16_MAX_LEVELS], ds[PERMUTE_F16_MAX_LEVELS];
  uint32_t total = 1u;
  for (uint32_t k = 0; k < PERMUTE_F16_MAX_LEVELS; k++) {
    if (k < nlv) {
      n[k] = lv[k].n ? lv[k].n : 1u;
      ss[k] = lv[k].src_step;
      ds[k] = lv[k].dst_step;
    } else {
      n[k] = 1u;
      ss[k] = 0;
      ds[k] = 0;
    }
    total *= n[k];
  }

  /* This core's contiguous chunk of the flattened space, and the only divisions in
   * the kernel: the loop below increments and carries from here. */
  uint32_t u = (core_id * total) / numThreads;
  uint32_t u1 = ((core_id + 1u) * total) / numThreads;
  if (u >= u1) {
    return;
  }

  /* Decompose the chunk's first index, level 0 fastest, and place both pointers. */
  uint32_t idx[PERMUTE_F16_MAX_LEVELS];
  int32_t so = 0, dof = 0;
  uint32_t rem = u;
  for (uint32_t k = 0; k < PERMUTE_F16_MAX_LEVELS; k++) {
    idx[k] = rem % n[k];
    rem /= n[k];
    so += (int32_t)idx[k] * ss[k];
    dof += (int32_t)idx[k] * ds[k];
  }

  const __fp16 *s = src + so;
  __fp16 *d = dst + dof;
  const uint32_t body = run >> 3, tail = run & 7u;

  for (;;) {
    /* One run.  Both pointers end it advanced by `run`, which is why the carry
     * below subtracts it back out before applying the level's own step. */
    for (uint32_t c = body; c > 0; c--) {
      PERMUTE_F16_UNROLLED8_LOOP;
    }
    for (uint32_t c = tail; c > 0; c--) {
      PERMUTE_F16_SINGLE;
    }
    if (++u >= u1) {
      break;
    }
    s -= run;
    d -= run;
    /* Carry: advance level 0, and roll into the next level whenever one wraps. */
    for (uint32_t k = 0; k < PERMUTE_F16_MAX_LEVELS; k++) {
      s += ss[k];
      d += ds[k];
      if (++idx[k] < n[k]) {
        break;
      }
      s -= (int32_t)n[k] * ss[k];
      d -= (int32_t)n[k] * ds[k];
      idx[k] = 0;
    }
  }
}
