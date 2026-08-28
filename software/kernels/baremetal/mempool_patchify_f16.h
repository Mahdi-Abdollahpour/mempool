// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "builtins_v2.h"

/* Patchify and its inverse, for a [Nf, Nt, C] resource grid.
 *
 *     [Nf, Nt, C]  ->  [Nf/Nh * Nt/Nw * C, Nh*Nw]
 *
 * A TOKEN is one (subcarrier block, symbol block, channel) triple and its row is
 * that block's Nh x Nw window of one channel, read subcarrier-major:
 *
 *     token = (fb * (Nt/Nw) + tb) * C + c        fb in [0, Nf/Nh), tb in [0, Nt/Nw)
 *     p     = (f - fb*Nh) * Nw + (t - tb*Nw)
 *     dst[token][p] = src[f][t][c]
 *
 * THE CHANNEL AXIS IS WHY THIS IS NOT A DMA JOB.  It is interleaved in the source
 * and is part of the token index in the destination, so the innermost run of any
 * descriptor is a single 2-byte element at a 2*C-byte stride -- one TCDM line-cycle
 * per element.  On the cores it is an ordinary strided gather.
 *
 * `tok0` is the first GLOBAL token this cluster owns and `f0` the first global
 * subcarrier of the slab it holds; `src` points at that slab, not at the grid.  The
 * two are separate because a lane's token slab need not start on a subcarrier
 * block -- see the per-lane slab premise in cevit_pipeline.py.
 *
 * ---- how it walks -----------------------------------------------------------
 *
 * THE TOKEN SIDE IS ONE CONTIGUOUS RUN.  A work unit is a (token, patch row) pair
 * and the units are numbered u = j*Nh + a, so consecutive units write consecutive
 * Nw-element rows: over a whole chunk of units the token-major pointer only ever
 * post-increments by one halfword and is never recomputed.
 *
 * THE GRID SIDE WALKS ITSELF TOO, with the two-stride trick mempool_cmatmul_f16.h
 * uses for its %[N] / %[N31] pair.  Within a patch row the grid steps by C; the LAST
 * element of the row is given the wrap stride (Nt - Nw + 1) * C instead, which lands
 * the pointer exactly on the next patch row's first element.  So an entire token --
 * Nh rows -- costs one address computation, not Nh of them.
 *
 * AND SO THERE IS NO DIVISION IN THE BODY.  A core takes a CONTIGUOUS CHUNK of the
 * unit axis rather than an interleaved stride of it, which lets (fb, tb, c, a)
 * increment and carry from one unit to the next.  The five divisions that decompose
 * the chunk's first unit are paid once, on entry.  Occupancy is unaffected: there are
 * n_tok * Nh units, thousands of them, against at most 1024 cores.
 *
 * THE `!` IS LOAD-BEARING.  `p.lhu %[v], %[s](%[p]!)` is the post-increment encoding;
 * without it the assembler takes the register-offset form that does NOT write the
 * address back, the pointer never moves, and the "+&r" constraint hides it from the
 * compiler.
 *
 * Neither kernel ends in a barrier: the caller rendezvouses, like every other kernel
 * here.
 */

/* Four elements of a patch row: load the run, then store it, so the load-use latency
 * of one is covered by the next.  `ls` and `ds` are post-increment strides in BYTES;
 * one of the two is always 2 -- the token-major end, which is contiguous. */
#define PATCH_F16_UNROLLED4_LOOP                                               \
  {                                                                            \
    asm volatile("p.lhu %[v0], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v1], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v2], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v3], %[ls](%[s]!) \n\t"                              \
                 "p.sh  %[v0], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v1], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v2], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v3], %[ds](%[d]!) \n\t"                              \
                 : [v0] "=&r"(v0), [v1] "=&r"(v1), [v2] "=&r"(v2),             \
                   [v3] "=&r"(v3), [s] "+&r"(s), [d] "+&r"(d)                  \
                 : [ls] "r"(ls), [ds] "r"(ds)                                  \
                 : "memory");                                                  \
  }

/* Two elements and one element at the in-row strides.  The tail is written
 * STRAIGHT-LINE rather than as a second loop: a patch row is Nw elements (7 in the
 * CEViT template), so the remainder is most of it and a loop's addi/bnez would be a
 * third of the tail's instructions. */
#define PATCH_F16_UNROLLED2_LOOP                                               \
  {                                                                            \
    asm volatile("p.lhu %[v0], %[ls](%[s]!) \n\t"                              \
                 "p.lhu %[v1], %[ls](%[s]!) \n\t"                              \
                 "p.sh  %[v0], %[ds](%[d]!) \n\t"                              \
                 "p.sh  %[v1], %[ds](%[d]!) \n\t"                              \
                 : [v0] "=&r"(v0), [v1] "=&r"(v1), [s] "+&r"(s), [d] "+&r"(d)  \
                 : [ls] "r"(ls), [ds] "r"(ds)                                  \
                 : "memory");                                                  \
  }

#define PATCH_F16_SINGLE                                                       \
  {                                                                            \
    asm volatile("p.lhu %[v0], %[ls](%[s]!) \n\t"                              \
                 "p.sh  %[v0], %[ds](%[d]!) \n\t"                              \
                 : [v0] "=&r"(v0), [s] "+&r"(s), [d] "+&r"(d)                  \
                 : [ls] "r"(ls), [ds] "r"(ds)                                  \
                 : "memory");                                                  \
  }

/* The LAST element of a patch row, at the WRAP strides: the grid side advances by
 * (Nt - Nw + 1) * C instead of C, which is what makes the next iteration start on the
 * next patch row with no pointer fixup between them. */
#define PATCH_F16_WRAP                                                         \
  {                                                                            \
    asm volatile("p.lhu %[v0], %[lw](%[s]!) \n\t"                              \
                 "p.sh  %[v0], %[dw](%[d]!) \n\t"                              \
                 : [v0] "=&r"(v0), [s] "+&r"(s), [d] "+&r"(d)                  \
                 : [lw] "r"(lw), [dw] "r"(dw)                                  \
                 : "memory");                                                  \
  }

/* One patch row: Nw elements, the last of them carrying the wrap stride. */
#define PATCH_F16_ROW                                                          \
  {                                                                            \
    uint32_t _k = nw - 1u;                                                     \
    for (; _k >= 4u; _k -= 4u) {                                               \
      PATCH_F16_UNROLLED4_LOOP;                                                \
    }                                                                          \
    if (_k & 2u) {                                                             \
      PATCH_F16_UNROLLED2_LOOP;                                                \
    }                                                                          \
    if (_k & 1u) {                                                             \
      PATCH_F16_SINGLE;                                                        \
    }                                                                          \
    PATCH_F16_WRAP;                                                            \
  }

void patchify_f16p(const __fp16 *__restrict__ src, __fp16 *__restrict__ dst,
                   uint32_t tok0, uint32_t f0, uint32_t n_tok, uint32_t nt,
                   uint32_t nh, uint32_t nw, uint32_t nc, uint32_t core_id,
                   uint32_t numThreads) {

  const uint32_t elem_b = (uint32_t)sizeof(__fp16);
  uint32_t tbc = nt / nw; // symbol blocks per subcarrier block
  uint32_t units = n_tok * nh;

  // The grid side is strided, the token side contiguous.
  const uint32_t ls = nc * elem_b, ds = elem_b;
  const uint32_t lw = (nt - nw + 1u) * nc * elem_b, dw = elem_b;
  uint32_t v0, v1, v2, v3;

  // This core's CONTIGUOUS chunk of the unit axis, and the only divisions in the
  // kernel: everything below increments and carries from here.
  uint32_t u = (core_id * units) / numThreads;
  uint32_t u1 = ((core_id + 1u) * units) / numThreads;

  uint32_t j = u / nh, a = u - j * nh;
  uint32_t tok = tok0 + j;
  uint32_t c = tok % nc, r = tok / nc;
  uint32_t tb = r % tbc, fb = r / tbc;

  __fp16 *d = dst + u * nw;
  while (u < u1) {
    // Where row `a` of this token starts in the slab.  Paid once per token, not
    // once per row -- PATCH_F16_ROW's wrap stride walks `s` to the next row.
    const __fp16 *s = src + (((fb * nh + a) - f0) * nt + tb * nw) * nc + c;
    uint32_t run = nh - a; // rows of this token left, clipped to the chunk
    if (run > u1 - u) {
      run = u1 - u;
    }
    u += run;
    a += run;
    for (; run > 0u; run--) {
      PATCH_F16_ROW;
    }
    if (a == nh) { // token finished: step (c, tb, fb)
      a = 0;
      if (++c == nc) {
        c = 0;
        if (++tb == tbc) {
          tb = 0;
          fb++;
        }
      }
    }
  }
}

/* The inverse: src[f][t][c] = dst[token][p], same indexing and the same walk
 * throughout, with the load and store roles swapped -- the token-major side is the
 * contiguous one to READ from and the grid the strided one to write.  `grid` is this
 * cluster's slab of the resource grid and `patched` the token-major tensor. */
void unpatchify_f16p(const __fp16 *__restrict__ patched,
                     __fp16 *__restrict__ grid, uint32_t tok0, uint32_t f0,
                     uint32_t n_tok, uint32_t nt, uint32_t nh, uint32_t nw,
                     uint32_t nc, uint32_t core_id, uint32_t numThreads) {

  const uint32_t elem_b = (uint32_t)sizeof(__fp16);
  uint32_t tbc = nt / nw;
  uint32_t units = n_tok * nh;

  const uint32_t ls = elem_b, ds = nc * elem_b;
  const uint32_t lw = elem_b, dw = (nt - nw + 1u) * nc * elem_b;
  uint32_t v0, v1, v2, v3;

  uint32_t u = (core_id * units) / numThreads;
  uint32_t u1 = ((core_id + 1u) * units) / numThreads;

  uint32_t j = u / nh, a = u - j * nh;
  uint32_t tok = tok0 + j;
  uint32_t c = tok % nc, r = tok / nc;
  uint32_t tb = r % tbc, fb = r / tbc;

  const __fp16 *s = patched + u * nw;
  while (u < u1) {
    __fp16 *d = grid + (((fb * nh + a) - f0) * nt + tb * nw) * nc + c;
    uint32_t run = nh - a;
    if (run > u1 - u) {
      run = u1 - u;
    }
    u += run;
    a += run;
    for (; run > 0u; run--) {
      PATCH_F16_ROW;
    }
    if (a == nh) {
      a = 0;
      if (++c == nc) {
        c = 0;
        if (++tb == tbc) {
          tb = 0;
          fb++;
        }
      }
    }
  }
}
