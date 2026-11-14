// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Mahdi Abdollahpour, University of Bologna

#pragma once
#include <stdint.h>

#include "builtins_v2.h"


// The arguments
typedef struct {
  const __fp16 *span; // address of the q|k|v matrix: X_BUF + A{1,2}_QKV
  uint32_t span_line; // step to the next line: 137 * 768
  uint32_t span_step; // step to the next span: 768
  const __fp16 *sum;  // address of the summary rows: span + 274 * 768
  uint32_t sum_line;  // step to the next line: 12 * 768
  uint32_t sum_step;  // step to the next window: 768
  const __fp16 *lpr;  // address of the local probes: span + 322 * 768 in 2.4
  uint32_t lpr_line;  // step to the next line: 12 * 768 in 2.4, 0 in 1.4
  uint32_t lpr_step;  // step to the next window: 768 in 2.4, 0 in 1.4
  const __fp16 *bias; // address of G, then delta: W_BUF + A{1,2}_W_LB
  __fp16 *a_span;     // out, span weights: X_BUF + A{1,2}_ASPAN
  __fp16 *a_hub;      // out, hub weights: X_BUF + A{1,2}_AHUB
  uint32_t n_lines;   // lines: 2
  uint32_t n_heads;   // heads: 2
  uint32_t n_hub;     // hubs that query: 2 in 1.4 (S, L), 1 in 2.4 (S)
  uint32_t n_span;    // spans in a line: 137
  uint32_t win;       // spans in a window: 12
  uint32_t d_model;   // model width: 256; k starts here in a row
  uint32_t d_head;    // head width: 128
} lhsm_args_t;


#define LHSM_LOG2E 0x3FB8AA3B // 1.44269504
#define LHSM_TMIN 0xC2C80000  // -100: lower t, 2^t is far below any weight
#define LHSM_C0 0x3F7FFB58    // 0.99992895
#define LHSM_C1 0x3F317A8D    // 0.69327623
#define LHSM_C2 0x3E786D32    // 0.24260405
#define LHSM_C3 0x3D61A4AB    // 0.05508868

#define LHSM_DEAD (-1.0e30f)



// one hub query
static inline void lhsm_hub_weights(const __fp16 *q, const __fp16 *k,
                                    int32_t step, const __fp16 *ks,
                                    const __fp16 *kl, uint32_t d,
                                    uint32_t n_live, float gs, float g1,
                                    float g2, uint16_t *o) {
  // after span 11: back to span 0, one word on
  int32_t back = 4 - 11 * step;
  // where q ends
  const __fp16 *const end = q + d;
  // start at biases
  float s0 = gs;
  float s1 = (n_live > 1) ? gs : LHSM_DEAD;
  float s2 = (n_live > 2) ? gs : LHSM_DEAD;
  float s3 = (n_live > 3) ? gs : LHSM_DEAD;
  float s4 = (n_live > 4) ? gs : LHSM_DEAD;
  float s5 = (n_live > 5) ? gs : LHSM_DEAD;
  float s6 = (n_live > 6) ? gs : LHSM_DEAD;
  float s7 = (n_live > 7) ? gs : LHSM_DEAD;
  float s8 = (n_live > 8) ? gs : LHSM_DEAD;
  float s9 = (n_live > 9) ? gs : LHSM_DEAD;
  float s10 = (n_live > 10) ? gs : LHSM_DEAD;
  float s11 = (n_live > 11) ? gs : LHSM_DEAD;
  float s12 = g1, s13 = g2;


  register uint16_t *o5 asm("x5") = o;
 
  asm volatile( // logits and softmax
      "addi sp, sp, -16;"
      "sw x5, 0(sp);" // o, parked
      ".balign 32;"
      "1:"
      "p.lw x5, 4(%[q]!);"        // q
      "p.lw x7, %[step](%[k]!);"  // span 0
      "p.lw x28, %[step](%[k]!);" // span 1
      "p.lw x29, %[step](%[k]!);" // span 2
      "p.lw x30, %[step](%[k]!);" // span 3
      "p.lw x31, %[step](%[k]!);" // span 4
      "vfdotpex.s.h %[s0], x5, x7;"
      "p.lw x7, %[step](%[k]!);" // span 5
      "vfdotpex.s.h %[s1], x5, x28;"
      "p.lw x28, %[step](%[k]!);" // span 6
      "vfdotpex.s.h %[s2], x5, x29;"
      "p.lw x29, %[step](%[k]!);" // span 7
      "vfdotpex.s.h %[s3], x5, x30;"
      "p.lw x30, %[step](%[k]!);" // span 8
      "vfdotpex.s.h %[s4], x5, x31;"
      "p.lw x31, %[step](%[k]!);" // span 9
      "vfdotpex.s.h %[s5], x5, x7;"
      "p.lw x7, %[step](%[k]!);" // span 10
      "vfdotpex.s.h %[s6], x5, x28;"
      "p.lw x28, %[back](%[k]!);" // span 11, then back to span 0
      "vfdotpex.s.h %[s7], x5, x29;"
      "p.lw x29, 4(%[ks]!);" // S
      "vfdotpex.s.h %[s8], x5, x30;"
      "p.lw x30, 4(%[kl]!);" // L
      "vfdotpex.s.h %[s9], x5, x31;"
      "vfdotpex.s.h %[s10], x5, x7;"
      "vfdotpex.s.h %[s11], x5, x28;"
      "vfdotpex.s.h %[s12], x5, x29;"
      "vfdotpex.s.h %[s13], x5, x30;"
      "bne %[q], %[end], 1b;"

      // lets already do the softmax (14 logits)
      // t = max((x - max) log2(e), -100), n = round(t), f = t - n, |f| <= 1/2
      // exp(x - max) = 2^n (((c3 f + c2) f + c1) f + c0), w = exp / sum of 14
      "fmax.s %[q], %[s0], %[s1];"
      "fmax.s %[k], %[s2], %[s3];"
      "fmax.s %[ks], %[s4], %[s5];"
      "fmax.s %[kl], %[s6], %[s7];"
      "fmax.s %[q], %[q], %[s8];"
      "fmax.s %[k], %[k], %[s9];"
      "fmax.s %[ks], %[ks], %[s10];"
      "fmax.s %[kl], %[kl], %[s11];"
      "fmax.s %[q], %[q], %[s12];"
      "fmax.s %[k], %[k], %[s13];"
      "li x5, %[log2e];" // log2(e)
      "fmax.s %[q], %[q], %[k];"
      "fmax.s %[ks], %[ks], %[kl];"
      "li x6, %[tmin];"           // -100
      "li x7, %[c0];"             // the cubic of 2^f: c0 to c3
      "fmax.s %[q], %[q], %[ks];" // the max
      "li x28, %[c1];"
      "li x29, %[c2];"
      "li x30, %[c3];"
      "fmul.s x31, %[q], x5;"          // max . log2(e)
      "fmsub.s %[s0], %[s0], x5, x31;" // t = x log2(e) - max log2(e).  [0,1,2]
      "fmsub.s %[s1], %[s1], x5, x31;"
      "fmsub.s %[s2], %[s2], x5, x31;"
      "fmax.s %[s0], %[s0], x6;" // t, not below -100
      "fmax.s %[s1], %[s1], x6;"
      "fmax.s %[s2], %[s2], x6;"
      "fcvt.w.s %[q], %[s0], rne;" // n = round(t)
      "fcvt.w.s %[ks], %[s1], rne;"
      "fcvt.w.s %[step], %[s2], rne;"
      "fcvt.s.w %[k], %[q];" // n, as fp32
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s0], %[s0], %[k];" // f = t - n, in [-0.5, 0.5]
      "fsub.s %[s1], %[s1], %[kl];"
      "fsub.s %[s2], %[s2], %[back];"
      "fmadd.s %[k], %[s0], x30, x29;" // c3 f + c2
      "fmadd.s %[kl], %[s1], x30, x29;"
      "fmadd.s %[back], %[s2], x30, x29;"
      "fmadd.s %[k], %[k], %[s0], x28;" // (c3 f + c2) f + c1
      "fmadd.s %[kl], %[kl], %[s1], x28;"
      "fmadd.s %[back], %[back], %[s2], x28;"
      "fmadd.s %[k], %[k], %[s0], x7;" // (..) f + c0: 2^f
      "fmadd.s %[kl], %[kl], %[s1], x7;"
      "fmadd.s %[back], %[back], %[s2], x7;"
      "slli %[q], %[q], 23;" // n, into the exponent field
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s0], %[k], %[q];" // exp(x - max) = 2^f 2^n
      "add %[s1], %[kl], %[ks];"
      "add %[s2], %[back], %[step];"
      "fmsub.s %[s3], %[s3], x5, x31;"   // [3,4,5]
      "fmsub.s %[s4], %[s4], x5, x31;"
      "fmsub.s %[s5], %[s5], x5, x31;"
      "fmax.s %[s3], %[s3], x6;"
      "fmax.s %[s4], %[s4], x6;"
      "fmax.s %[s5], %[s5], x6;"
      "fcvt.w.s %[q], %[s3], rne;"
      "fcvt.w.s %[ks], %[s4], rne;"
      "fcvt.w.s %[step], %[s5], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s3], %[s3], %[k];"
      "fsub.s %[s4], %[s4], %[kl];"
      "fsub.s %[s5], %[s5], %[back];"
      "fmadd.s %[k], %[s3], x30, x29;"
      "fmadd.s %[kl], %[s4], x30, x29;"
      "fmadd.s %[back], %[s5], x30, x29;"
      "fmadd.s %[k], %[k], %[s3], x28;"
      "fmadd.s %[kl], %[kl], %[s4], x28;"
      "fmadd.s %[back], %[back], %[s5], x28;"
      "fmadd.s %[k], %[k], %[s3], x7;"
      "fmadd.s %[kl], %[kl], %[s4], x7;"
      "fmadd.s %[back], %[back], %[s5], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s3], %[k], %[q];"
      "add %[s4], %[kl], %[ks];"
      "add %[s5], %[back], %[step];"
      "fmsub.s %[s6], %[s6], x5, x31;" // [6,7,8]
      "fmsub.s %[s7], %[s7], x5, x31;"
      "fmsub.s %[s8], %[s8], x5, x31;"
      "fmax.s %[s6], %[s6], x6;"
      "fmax.s %[s7], %[s7], x6;"
      "fmax.s %[s8], %[s8], x6;"
      "fcvt.w.s %[q], %[s6], rne;"
      "fcvt.w.s %[ks], %[s7], rne;"
      "fcvt.w.s %[step], %[s8], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s6], %[s6], %[k];"
      "fsub.s %[s7], %[s7], %[kl];"
      "fsub.s %[s8], %[s8], %[back];"
      "fmadd.s %[k], %[s6], x30, x29;"
      "fmadd.s %[kl], %[s7], x30, x29;"
      "fmadd.s %[back], %[s8], x30, x29;"
      "fmadd.s %[k], %[k], %[s6], x28;"
      "fmadd.s %[kl], %[kl], %[s7], x28;"
      "fmadd.s %[back], %[back], %[s8], x28;"
      "fmadd.s %[k], %[k], %[s6], x7;"
      "fmadd.s %[kl], %[kl], %[s7], x7;"
      "fmadd.s %[back], %[back], %[s8], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s6], %[k], %[q];"
      "add %[s7], %[kl], %[ks];"
      "add %[s8], %[back], %[step];"
      "fmsub.s %[s9], %[s9], x5, x31;"        // [9,10,11]
      "fmsub.s %[s10], %[s10], x5, x31;"
      "fmsub.s %[s11], %[s11], x5, x31;"
      "fmax.s %[s9], %[s9], x6;"
      "fmax.s %[s10], %[s10], x6;"
      "fmax.s %[s11], %[s11], x6;"
      "fcvt.w.s %[q], %[s9], rne;"
      "fcvt.w.s %[ks], %[s10], rne;"
      "fcvt.w.s %[step], %[s11], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s9], %[s9], %[k];"
      "fsub.s %[s10], %[s10], %[kl];"
      "fsub.s %[s11], %[s11], %[back];"
      "fmadd.s %[k], %[s9], x30, x29;"
      "fmadd.s %[kl], %[s10], x30, x29;"
      "fmadd.s %[back], %[s11], x30, x29;"
      "fmadd.s %[k], %[k], %[s9], x28;"
      "fmadd.s %[kl], %[kl], %[s10], x28;"
      "fmadd.s %[back], %[back], %[s11], x28;"
      "fmadd.s %[k], %[k], %[s9], x7;"
      "fmadd.s %[kl], %[kl], %[s10], x7;"
      "fmadd.s %[back], %[back], %[s11], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s9], %[k], %[q];"
      "add %[s10], %[kl], %[ks];"
      "add %[s11], %[back], %[step];"
      "fmsub.s %[s12], %[s12], x5, x31;"  // [12,13]
      "fmsub.s %[s13], %[s13], x5, x31;"
      "fmax.s %[s12], %[s12], x6;"
      "fmax.s %[s13], %[s13], x6;"
      "fcvt.w.s %[q], %[s12], rne;"
      "fcvt.w.s %[ks], %[s13], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fsub.s %[s12], %[s12], %[k];"
      "fsub.s %[s13], %[s13], %[kl];"
      "fmadd.s %[k], %[s12], x30, x29;"
      "fmadd.s %[kl], %[s13], x30, x29;"
      "fmadd.s %[k], %[k], %[s12], x28;"
      "fmadd.s %[kl], %[kl], %[s13], x28;"
      "fmadd.s %[k], %[k], %[s12], x7;"
      "fmadd.s %[kl], %[kl], %[s13], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "add %[s12], %[k], %[q];"
      "add %[s13], %[kl], %[ks];"
      "lw x31, 0(sp);" // o, back
      "addi sp, sp, 16;"
      "fadd.s x5, %[s0], %[s1];"
      "fadd.s x6, %[s2], %[s3];"
      "fadd.s x7, %[s4], %[s5];"
      "fadd.s x28, %[s6], %[s7];"
      "fadd.s x5, x5, %[s8];"
      "fadd.s x6, x6, %[s9];"
      "fadd.s x7, x7, %[s10];"
      "fadd.s x28, x28, %[s11];"
      "fadd.s x5, x5, %[s12];"
      "fadd.s x6, x6, %[s13];"
      "li x29, 0x3F800000;" // 1.0
      "fadd.s x5, x5, x6;"
      "fadd.s x7, x7, x28;"
      "fadd.s x5, x5, x7;"        // sum
      "fdiv.s x29, x29, x5;"      // r = 1 / sum
      "fmul.s %[s0], %[s0], x29;" // w 
      "fmul.s %[s1], %[s1], x29;"
      "fmul.s %[s2], %[s2], x29;"
      "fmul.s %[s3], %[s3], x29;"
      "fmul.s %[s4], %[s4], x29;"
      "fmul.s %[s5], %[s5], x29;"
      "fmul.s %[s6], %[s6], x29;"
      "fmul.s %[s7], %[s7], x29;"
      "fmul.s %[s8], %[s8], x29;"
      "fmul.s %[s9], %[s9], x29;"
      "fmul.s %[s10], %[s10], x29;"
      "fmul.s %[s11], %[s11], x29;"
      "fmul.s %[s12], %[s12], x29;"
      "fmul.s %[s13], %[s13], x29;"
      "vfcpka.h.s x5, %[s0], %[s1];" // w0, w1: w0 in the low half
      "vfcpka.h.s x6, %[s2], %[s3];"
      "vfcpka.h.s x7, %[s4], %[s5];"
      "sw x5, 0(x31);"                 // store as 7 words
      "vfcpka.h.s x5, %[s6], %[s7];"
      "sw x6, 4(x31);"
      "vfcpka.h.s x6, %[s8], %[s9];"
      "sw x7, 8(x31);"
      "vfcpka.h.s x7, %[s10], %[s11];"
      "sw x5, 12(x31);"
      "vfcpka.h.s x5, %[s12], %[s13];"
      "sw x6, 16(x31);"
      "sw x7, 20(x31);"
      "sw x5, 24(x31);"
      : [s0] "+&r"(s0), [s1] "+&r"(s1), [s2] "+&r"(s2), [s3] "+&r"(s3),
        [s4] "+&r"(s4), [s5] "+&r"(s5), [s6] "+&r"(s6), [s7] "+&r"(s7),
        [s8] "+&r"(s8), [s9] "+&r"(s9), [s10] "+&r"(s10), [s11] "+&r"(s11),
        [s12] "+&r"(s12), [s13] "+&r"(s13), [q] "+&r"(q), [k] "+&r"(k),
        [ks] "+&r"(ks), [kl] "+&r"(kl), [step] "+&r"(step), [back] "+&r"(back),
        [o] "+&r"(o5)
      : [end] "r"(end), [log2e] "i"(LHSM_LOG2E), [tmin] "i"(LHSM_TMIN),
        [c0] "i"(LHSM_C0), [c1] "i"(LHSM_C1), [c2] "i"(LHSM_C2),
        [c3] "i"(LHSM_C3)
      : "x6", "x7", "x28", "x29", "x30", "x31", "memory");
}

// 4 span queries
static inline void lhsm_span_weights(const __fp16 *p, int32_t dm, int32_t step,
                                     const __fp16 *ks, const __fp16 *kl,
                                     uint32_t d, float g0, float g1, float g2,
                                     uint32_t skip, __fp16 *o) {
  // from the k of a row to the q of the next
  int32_t nx = step - dm;
  // after the k of row 3: back to the q of row 0, one word on
  int32_t back = 4 - 3 * step - dm;
  // where the q of row 0 ends
  const __fp16 *end = p + d;
  // the 12 sums, fp32, start at their biases: row 0 k, S, L, then row 1, 2, 3
  float s0 = g0, s1 = g1, s2 = g2, s3 = g0, s4 = g1, s5 = g2;
  float s6 = g0, s7 = g1, s8 = g2, s9 = g0, s10 = g1, s11 = g2;
  // 23 instructions a loop step, 12 vfdotpex.s.h: x5, x6 hold S and L, x7 and
  // x28 to x31 the q and k of the rows, in turn
  asm volatile(
      "1:"
      "p.lw x5, 4(%[ks]!);"     // S
      "p.lw x6, 4(%[kl]!);"     // L
      "p.lw x7, %[dm](%[p]!);"  // q, row 0
      "p.lw x28, %[nx](%[p]!);" // k, row 0
      "p.lw x29, %[dm](%[p]!);" // q, row 1
      "p.lw x30, %[nx](%[p]!);" // k, row 1
      "p.lw x31, %[dm](%[p]!);" // q, row 2
      "vfdotpex.s.h %[s1], x7, x5;"
      "vfdotpex.s.h %[s2], x7, x6;"
      "vfdotpex.s.h %[s0], x7, x28;"
      "p.lw x7, %[nx](%[p]!);"  // k, row 2
      "p.lw x28, %[dm](%[p]!);" // q, row 3
      "vfdotpex.s.h %[s4], x29, x5;"
      "vfdotpex.s.h %[s5], x29, x6;"
      "vfdotpex.s.h %[s3], x29, x30;"
      "p.lw x29, %[back](%[p]!);" // k, row 3, then back to row 0
      "vfdotpex.s.h %[s7], x31, x5;"
      "vfdotpex.s.h %[s8], x31, x6;"
      "vfdotpex.s.h %[s6], x31, x7;"
      "vfdotpex.s.h %[s10], x28, x5;"
      "vfdotpex.s.h %[s11], x28, x6;"
      "vfdotpex.s.h %[s9], x28, x29;"
      "bne %[p], %[end], 1b;"

      // softmax, a row at a time: the max of 3, then 3 exps
      // t = max((x - max) log2(e), -100), n = round(t), f = t - n
      // exp(x - max) = 2^n (((c3 f + c2) f + c1) f + c0)
      "fmax.s x31, %[s0], %[s1];" // max, row 0
      "fmax.s x16, %[s3], %[s4];" // max, row 1
      "li x5, %[log2e];"
      "fmax.s x31, x31, %[s2];"
      "fmax.s x16, x16, %[s5];"
      "li x6, %[tmin];"
      "fmul.s x31, x31, x5;" // max log2(e), row 0
      "fmul.s x16, x16, x5;" // max log2(e), row 1
      "li x7, %[c0];"
      "li x28, %[c1];"
      "li x29, %[c2];"
      "li x30, %[c3];"
      "fmsub.s %[s0], %[s0], x5, x31;" // t = x log2(e) - max log2(e)
      "fmsub.s %[s1], %[s1], x5, x31;"
      "fmsub.s %[s2], %[s2], x5, x31;"
      "fmax.s x31, %[s6], %[s7];" // max, row 2
      "fmax.s %[s0], %[s0], x6;"  // t, not below -100
      "fmax.s %[s1], %[s1], x6;"
      "fmax.s %[s2], %[s2], x6;"
      "fmax.s x31, x31, %[s8];"
      "fcvt.w.s %[p], %[s0], rne;" // n = round(t)
      "fcvt.w.s %[ks], %[s1], rne;"
      "fcvt.w.s %[kl], %[s2], rne;"
      "fmul.s x31, x31, x5;" // max log2(e), row 2
      "fcvt.s.w %[dm], %[p];"
      "fcvt.s.w %[nx], %[ks];"
      "fcvt.s.w %[back], %[kl];"
      "fsub.s %[s0], %[s0], %[dm];" // f = t - n
      "fsub.s %[s1], %[s1], %[nx];"
      "fsub.s %[s2], %[s2], %[back];"
      "fmadd.s %[dm], %[s0], x30, x29;" // 2^f, the cubic
      "fmadd.s %[nx], %[s1], x30, x29;"
      "fmadd.s %[back], %[s2], x30, x29;"
      "fmadd.s %[dm], %[dm], %[s0], x28;"
      "fmadd.s %[nx], %[nx], %[s1], x28;"
      "fmadd.s %[back], %[back], %[s2], x28;"
      "fmadd.s %[dm], %[dm], %[s0], x7;"
      "fmadd.s %[nx], %[nx], %[s1], x7;"
      "fmadd.s %[back], %[back], %[s2], x7;"
      "slli %[p], %[p], 23;" // n, into the exponent field
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s0], %[dm], %[p];" // exp(x - max) = 2^f 2^n
      "add %[s1], %[nx], %[ks];"
      "add %[s2], %[back], %[kl];"
      "fmsub.s %[s3], %[s3], x5, x16;" // row 1
      "fmsub.s %[s4], %[s4], x5, x16;"
      "fmsub.s %[s5], %[s5], x5, x16;"
      "fmax.s x16, %[s9], %[s10];" // max, row 3
      "fmax.s %[s3], %[s3], x6;"
      "fmax.s %[s4], %[s4], x6;"
      "fmax.s %[s5], %[s5], x6;"
      "fmax.s x16, x16, %[s11];"
      "fcvt.w.s %[p], %[s3], rne;"
      "fcvt.w.s %[ks], %[s4], rne;"
      "fcvt.w.s %[kl], %[s5], rne;"
      "fmul.s x16, x16, x5;" // max log2(e), row 3
      "fcvt.s.w %[dm], %[p];"
      "fcvt.s.w %[nx], %[ks];"
      "fcvt.s.w %[back], %[kl];"
      "fsub.s %[s3], %[s3], %[dm];"
      "fsub.s %[s4], %[s4], %[nx];"
      "fsub.s %[s5], %[s5], %[back];"
      "fmadd.s %[dm], %[s3], x30, x29;"
      "fmadd.s %[nx], %[s4], x30, x29;"
      "fmadd.s %[back], %[s5], x30, x29;"
      "fmadd.s %[dm], %[dm], %[s3], x28;"
      "fmadd.s %[nx], %[nx], %[s4], x28;"
      "fmadd.s %[back], %[back], %[s5], x28;"
      "fmadd.s %[dm], %[dm], %[s3], x7;"
      "fmadd.s %[nx], %[nx], %[s4], x7;"
      "fmadd.s %[back], %[back], %[s5], x7;"
      "slli %[p], %[p], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s3], %[dm], %[p];"
      "add %[s4], %[nx], %[ks];"
      "add %[s5], %[back], %[kl];"
      "fmsub.s %[s6], %[s6], x5, x31;" // row 2
      "fmsub.s %[s7], %[s7], x5, x31;"
      "fmsub.s %[s8], %[s8], x5, x31;"
      "fmax.s %[s6], %[s6], x6;"
      "fmax.s %[s7], %[s7], x6;"
      "fmax.s %[s8], %[s8], x6;"
      "fcvt.w.s %[p], %[s6], rne;"
      "fcvt.w.s %[ks], %[s7], rne;"
      "fcvt.w.s %[kl], %[s8], rne;"
      "fcvt.s.w %[dm], %[p];"
      "fcvt.s.w %[nx], %[ks];"
      "fcvt.s.w %[back], %[kl];"
      "fsub.s %[s6], %[s6], %[dm];"
      "fsub.s %[s7], %[s7], %[nx];"
      "fsub.s %[s8], %[s8], %[back];"
      "fmadd.s %[dm], %[s6], x30, x29;"
      "fmadd.s %[nx], %[s7], x30, x29;"
      "fmadd.s %[back], %[s8], x30, x29;"
      "fmadd.s %[dm], %[dm], %[s6], x28;"
      "fmadd.s %[nx], %[nx], %[s7], x28;"
      "fmadd.s %[back], %[back], %[s8], x28;"
      "fmadd.s %[dm], %[dm], %[s6], x7;"
      "fmadd.s %[nx], %[nx], %[s7], x7;"
      "fmadd.s %[back], %[back], %[s8], x7;"
      "slli %[p], %[p], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s6], %[dm], %[p];"
      "add %[s7], %[nx], %[ks];"
      "add %[s8], %[back], %[kl];"
      "fmsub.s %[s9], %[s9], x5, x16;" // row 3
      "fmsub.s %[s10], %[s10], x5, x16;"
      "fmsub.s %[s11], %[s11], x5, x16;"
      "fmax.s %[s9], %[s9], x6;"
      "fmax.s %[s10], %[s10], x6;"
      "fmax.s %[s11], %[s11], x6;"
      "fcvt.w.s %[p], %[s9], rne;"
      "fcvt.w.s %[ks], %[s10], rne;"
      "fcvt.w.s %[kl], %[s11], rne;"
      "fcvt.s.w %[dm], %[p];"
      "fcvt.s.w %[nx], %[ks];"
      "fcvt.s.w %[back], %[kl];"
      "fsub.s %[s9], %[s9], %[dm];"
      "fsub.s %[s10], %[s10], %[nx];"
      "fsub.s %[s11], %[s11], %[back];"
      "fmadd.s %[dm], %[s9], x30, x29;"
      "fmadd.s %[nx], %[s10], x30, x29;"
      "fmadd.s %[back], %[s11], x30, x29;"
      "fmadd.s %[dm], %[dm], %[s9], x28;"
      "fmadd.s %[nx], %[nx], %[s10], x28;"
      "fmadd.s %[back], %[back], %[s11], x28;"
      "fmadd.s %[dm], %[dm], %[s9], x7;"
      "fmadd.s %[nx], %[nx], %[s10], x7;"
      "fmadd.s %[back], %[back], %[s11], x7;"
      "slli %[p], %[p], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s9], %[dm], %[p];"
      "add %[s10], %[nx], %[ks];"
      "add %[s11], %[back], %[kl];"
      // w = exp / sum of its row, with one fdiv.s for the 4 rows:
      // R = 1 / (S0 S1 S2 S3), 1 / S0 = S1 (S2 S3 R), 1 / S2 = S3 (S0 S1 R)
      "fadd.s %[p], %[s0], %[s1];"   // S0, the sum of row 0
      "fadd.s %[ks], %[s3], %[s4];"  // S1
      "fadd.s %[kl], %[s6], %[s7];"  // S2
      "fadd.s %[dm], %[s9], %[s10];" // S3
      "fadd.s %[p], %[p], %[s2];"
      "fadd.s %[ks], %[ks], %[s5];"
      "fadd.s %[kl], %[kl], %[s8];"
      "fadd.s %[dm], %[dm], %[s11];"
      "fmul.s %[nx], %[p], %[ks];"    // S0 S1
      "fmul.s %[back], %[kl], %[dm];" // S2 S3
      "li x5, 0x3F800000;"            // 1.0
      "fmul.s x6, %[nx], %[back];"    // S0 S1 S2 S3
      "fdiv.s x5, x5, x6;"            // R
      "fmul.s %[nx], %[nx], x5;"      // S0 S1 R = 1 / (S2 S3)
      "fmul.s %[back], %[back], x5;"  // S2 S3 R = 1 / (S0 S1)
      "fmul.s x7, %[ks], %[back];"    // 1 / S0
      "fmul.s x28, %[p], %[back];"    // 1 / S1
      "fmul.s x29, %[dm], %[nx];"     // 1 / S2
      "fmul.s x30, %[kl], %[nx];"     // 1 / S3
      "fmul.s %[s0], %[s0], x7;"      // w = exp / sum
      "fmul.s %[s1], %[s1], x7;"
      "fmul.s %[s2], %[s2], x7;"
      "fmul.s %[s3], %[s3], x28;"
      "fmul.s %[s4], %[s4], x28;"
      "fmul.s %[s5], %[s5], x28;"
      "fmul.s %[s6], %[s6], x29;"
      "fmul.s %[s7], %[s7], x29;"
      "fmul.s %[s8], %[s8], x29;"
      "fmul.s %[s9], %[s9], x30;"
      "fmul.s %[s10], %[s10], x30;"
      "fmul.s %[s11], %[s11], x30;"
      : [s0] "+&r"(s0), [s1] "+&r"(s1), [s2] "+&r"(s2), [s3] "+&r"(s3),
        [s4] "+&r"(s4), [s5] "+&r"(s5), [s6] "+&r"(s6), [s7] "+&r"(s7),
        [s8] "+&r"(s8), [s9] "+&r"(s9), [s10] "+&r"(s10), [s11] "+&r"(s11),
        [p] "+&r"(p), [ks] "+&r"(ks), [kl] "+&r"(kl), [dm] "+&r"(dm),
        [nx] "+&r"(nx), [back] "+&r"(back)
      : [end] "r"(end), [log2e] "i"(LHSM_LOG2E), [tmin] "i"(LHSM_TMIN),
        [c0] "i"(LHSM_C0), [c1] "i"(LHSM_C1), [c2] "i"(LHSM_C2),
        [c3] "i"(LHSM_C3)
      : "x5", "x6", "x7", "x16", "x28", "x29", "x30", "x31", "memory");
  // rows skip to 3, as fp16, 3 a row
  if (skip == 0) {
    o[0] = (__fp16)s0;
    o[1] = (__fp16)s1;
    o[2] = (__fp16)s2;
    o += 3;
  }
  if (skip <= 1) {
    o[0] = (__fp16)s3;
    o[1] = (__fp16)s4;
    o[2] = (__fp16)s5;
    o += 3;
  }
  if (skip <= 2) {
    o[0] = (__fp16)s6;
    o[1] = (__fp16)s7;
    o[2] = (__fp16)s8;
    o += 3;
  }
  o[0] = (__fp16)s9;
  o[1] = (__fp16)s10;
  o[2] = (__fp16)s11;
}


static inline void lhsm_span_batch(const lhsm_args_t *a, uint32_t b,
                                   uint32_t n_win) {
  // heads (2), window (12), head width (128)
  const uint32_t n_heads = a->n_heads, win = a->win, d = a->d_head;
  // t = (l * n_heads + h) * n_win + w
  const uint32_t t = b / 3;
  // part: rows 4 part to 4 part + 3 of the window
  const uint32_t part = b - 3 * t;
  // u = l * n_heads + h
  const uint32_t u = t / n_win;
  // w: the window, 0 to 11
  const uint32_t w = t - u * n_win;
  // l: the line, 0 or 1
  const uint32_t l = u / n_heads;
  // h: the head, 0 or 1
  const uint32_t h = u - l * n_heads;
  // live spans of the window: 12, 5 in the last one
  const uint32_t n_live = (w == n_win - 1) ? a->n_span - w * win : win;
  // the first row this batch stores
  const uint32_t i0 = 4 * part;
  // part 2 of the short window: no rows
  if (i0 >= n_live) {
    return;
  }
  // the first of the 4 rows computed: i0, or earlier when fewer are left
  const uint32_t r0 = (i0 + 4 <= n_live) ? i0 : n_live - 4;
  // where q and k of head h start in a row
  const uint32_t qo = h * d, ko = a->d_model + h * d;
  // span row r0 of window w of line l
  const __fp16 *row =
      a->span + l * a->span_line + (w * win + r0) * a->span_step;
  // the summary row S_w of line l
  const __fp16 *srow = a->sum + l * a->sum_line + w * a->sum_step;
  // the local-probe row L_w (constant in 1.4)
  const __fp16 *lrow = a->lpr + l * a->lpr_line + w * a->lpr_step;
  // G[h][0][0..2]: the biases of a span query
  const __fp16 *g = a->bias + h * (1 + a->n_hub) * 3;
  // a_span row of span i0: r = u * n_span + i, 3 slots a row
  __fp16 *o = a->a_span + 3 * (u * a->n_span + w * win + i0);
  // the 4 rows against their own k, S_w and L_w (biases G[h][0][0..2]); the
  // batch's own rows, i0 - r0 to 3, to o
  lhsm_span_weights(row + qo, (int32_t)(2 * a->d_model),
                    (int32_t)(2 * a->span_step), srow + ko, lrow + ko, d,
                    (float)g[0], (float)g[1], (float)g[2], i0 - r0, o);
}

static inline void lhsm_hub_row(const lhsm_args_t *a, uint32_t r,
                                uint32_t n_win) {
  const uint32_t n_hub = a->n_hub, n_heads = a->n_heads, win = a->win;
  const uint32_t d = a->d_head;
  // t = (l * n_heads + h) * n_win + w
  uint32_t t = r / n_hub;
  // eta: the hub that queries, 0 S, 1 L
  const uint32_t eta = r - t * n_hub;
  // u = l * n_heads + h
  uint32_t u = t / n_win;
  // w: the window, 0 to 11
  const uint32_t w = t - u * n_win;
  // l: the line, 0 or 1
  const uint32_t l = u / n_heads;
  // h: the head, 0 or 1
  const uint32_t h = u - l * n_heads;
  // live spans: 12, 5 in the last window, which is padded to 12
  const uint32_t n_live = (w == n_win - 1) ? a->n_span - w * win : win;
  // where q and k of head h start in a row
  const uint32_t qo = h * d, ko = a->d_model + h * d;
  // the summary row S_w of line l
  const __fp16 *srow = a->sum + l * a->sum_line + w * a->sum_step;
  // the local-probe row L_w (constant in 1.4)
  const __fp16 *lrow = a->lpr + l * a->lpr_line + w * a->lpr_step;
  // the first span row of window w
  const __fp16 *span0 = a->span + l * a->span_line + w * win * a->span_step;
  // G[h][1 + eta]: the 3 biases of this hub query
  const __fp16 *g = a->bias + (h * (1 + n_hub) + 1 + eta) * 3;
  // the bias of the span keys
  float g_span = (float)g[0];
  // in the short last window, plus delta[h], stored after all of G
  if (n_live < win) {
    g_span += (float)a->bias[n_heads * (1 + n_hub) * 3 + h];
  }

  lhsm_hub_weights((eta ? lrow : srow) + qo, span0 + ko,
                   (int32_t)(2 * a->span_step), srow + ko, lrow + ko, d, n_live,
                   g_span, (float)g[1], (float)g[2],
                   (uint16_t *)(a->a_hub + r * (win + 2)));
}

// ==== the kernel ==========================================================

void local_hub_softmax_f16(const lhsm_args_t *a, uint32_t core_id,
                           uint32_t num_cores) {
  // windows a line, the short one too: 12
  const uint32_t n_win = (a->n_span + a->win - 1) / a->win;
  // hub rows: 96 in 1.4 (S, L), 48 in 2.4 (S)
  const uint32_t n_hr = a->n_lines * a->n_heads * n_win * a->n_hub;
  // span batches, 3 a window: 2 lines x 2 heads x 12 x 3 = 144, 4 empty
  const uint32_t n_sb = a->n_lines * a->n_heads * n_win * 3;
  // more cores than hub rows, as in the app
  if (num_cores > n_hr) {
    // the first n_hr cores take one hub row each, row = core id
    if (core_id < n_hr) {
      lhsm_hub_row(a, core_id, n_win);
    } else {
      // the others (160 in 1.4, 208 in 2.4) take the span batches round
      // robin: one each in the app, or none
      for (uint32_t b = core_id - n_hr; b < n_sb; b += num_cores - n_hr) {
        lhsm_span_batch(a, b, n_win);
      }
    }
  } else {
    // fewer cores, or as many as hub rows: every hub row round robin,
    for (uint32_t r = core_id; r < n_hr; r += num_cores) {
      lhsm_hub_row(a, r, n_win);
    }
    // then every span batch round robin
    for (uint32_t b = core_id; b < n_sb; b += num_cores) {
      lhsm_span_batch(a, b, n_win);
    }
  }
}
