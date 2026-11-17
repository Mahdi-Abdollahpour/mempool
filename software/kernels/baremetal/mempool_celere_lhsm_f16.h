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
  uint32_t lpr_copy;  // step to the next group's copy: 1024 in 1.4, 0 in 2.4
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
// 2^f, |f| <= 1/2, is the cubic c0 + c1 f + c2 f^2 + c3 f^3 of least relative
// error (minimax: 7.5e-5). The asm takes it over c3, as ((f + A) f + B) f + C
// with A = c2 / c3, B = c1 / c3, C = c0 / c3; the 1 / c3 cancels in the sum
#define LHSM_A 0x408CB763 // 4.39738607
#define LHSM_B 0x41490C64 // 12.5655251
#define LHSM_C 0x4190FDD5 // 18.1239414

#define LHSM_DEAD (-1.0e30f)

// LHSM_PIPELINED 1 (default): a hub row's logits in two software-pipelined
// passes of 7, each key loaded a step before its use; 0: one pass over the 14
#ifndef LHSM_PIPELINED
#define LHSM_PIPELINED 1
#endif

// fp16 to fp32 in one instruction: MemPool's LLVM turns the C cast into a
// library call
static inline float lhsm_f32(const __fp16 *h) {
  const uint32_t b = *(const uint16_t *)h;
  float f;
  asm volatile("fcvt.s.h %[f], %[b];" : [f] "=r"(f) : [b] "r"(b));
  return f;
}



// one hub query: its 14 logits, then their softmax, stored at o as 7 fp16 pairs
static inline void lhsm_hub_weights(const __fp16 *q, const __fp16 *k,
                                    int32_t step, const __fp16 *ks,
                                    const __fp16 *kl, uint32_t d,
                                    uint32_t n_live, float gs, float g1,
                                    float g2, uint16_t *o) {
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

#if LHSM_PIPELINED
  // two passes of 7 logits, spans 0 to 6, then spans 7 to 11, S and L: each
  // key has its own register, loaded a step before its use, and the next word
  // of q its own too
  // after span 6: back to span 0, one word on (pass A)
  int32_t back_a = 4 - 6 * step;
  // after span 11: back to span 7, one word on (pass B)
  int32_t back_b = 4 - 4 * step;
  // span 7, where pass B starts
  const __fp16 *k7 = k + 7 * (step / 2);
  // the rounds stop at the last word but one, two steps finish a pass (d: a
  // multiple of 4, at least 8)
  const __fp16 *const end = q + d - 2;
  // two words of q; p walks q
  uint32_t qa, qb;
  const __fp16 *p = q;

  asm volatile(                   // pass A: spans 0 to 6
      "p.lw %[qa], 4(%[p]!);"     // q, word 0
      "p.lw x5, %[step](%[k]!);"  // span 0
      "p.lw x6, %[step](%[k]!);"  // span 1
      "p.lw x7, %[step](%[k]!);"  // span 2
      "p.lw x28, %[step](%[k]!);" // span 3
      "p.lw x29, %[step](%[k]!);" // span 4
      "p.lw x30, %[step](%[k]!);" // span 5
      "p.lw x31, %[back](%[k]!);" // span 6, then back to span 0, one word on
      ".balign 32;"
      "1:"                    // two steps a round
      "p.lw %[qb], 4(%[p]!);" // next word of q
      "vfdotpex.s.h %[s0], %[qa], x5;"
      "p.lw x5, %[step](%[k]!);" // span 0, next word
      "vfdotpex.s.h %[s1], %[qa], x6;"
      "p.lw x6, %[step](%[k]!);"
      "vfdotpex.s.h %[s2], %[qa], x7;"
      "p.lw x7, %[step](%[k]!);"
      "vfdotpex.s.h %[s3], %[qa], x28;"
      "p.lw x28, %[step](%[k]!);"
      "vfdotpex.s.h %[s4], %[qa], x29;"
      "p.lw x29, %[step](%[k]!);"
      "vfdotpex.s.h %[s5], %[qa], x30;"
      "p.lw x30, %[step](%[k]!);"
      "vfdotpex.s.h %[s6], %[qa], x31;"
      "p.lw x31, %[back](%[k]!);"
      "p.lw %[qa], 4(%[p]!);" // the word after
      "vfdotpex.s.h %[s0], %[qb], x5;"
      "p.lw x5, %[step](%[k]!);"
      "vfdotpex.s.h %[s1], %[qb], x6;"
      "p.lw x6, %[step](%[k]!);"
      "vfdotpex.s.h %[s2], %[qb], x7;"
      "p.lw x7, %[step](%[k]!);"
      "vfdotpex.s.h %[s3], %[qb], x28;"
      "p.lw x28, %[step](%[k]!);"
      "vfdotpex.s.h %[s4], %[qb], x29;"
      "p.lw x29, %[step](%[k]!);"
      "vfdotpex.s.h %[s5], %[qb], x30;"
      "p.lw x30, %[step](%[k]!);"
      "vfdotpex.s.h %[s6], %[qb], x31;"
      "p.lw x31, %[back](%[k]!);"
      "bne %[p], %[end], 1b;"
      // the last two steps
      "p.lw %[qb], 4(%[p]!);"
      "vfdotpex.s.h %[s0], %[qa], x5;"
      "p.lw x5, %[step](%[k]!);"
      "vfdotpex.s.h %[s1], %[qa], x6;"
      "p.lw x6, %[step](%[k]!);"
      "vfdotpex.s.h %[s2], %[qa], x7;"
      "p.lw x7, %[step](%[k]!);"
      "vfdotpex.s.h %[s3], %[qa], x28;"
      "p.lw x28, %[step](%[k]!);"
      "vfdotpex.s.h %[s4], %[qa], x29;"
      "p.lw x29, %[step](%[k]!);"
      "vfdotpex.s.h %[s5], %[qa], x30;"
      "p.lw x30, %[step](%[k]!);"
      "vfdotpex.s.h %[s6], %[qa], x31;"
      "p.lw x31, %[step](%[k]!);"
      "vfdotpex.s.h %[s0], %[qb], x5;"
      "vfdotpex.s.h %[s1], %[qb], x6;"
      "vfdotpex.s.h %[s2], %[qb], x7;"
      "vfdotpex.s.h %[s3], %[qb], x28;"
      "vfdotpex.s.h %[s4], %[qb], x29;"
      "vfdotpex.s.h %[s5], %[qb], x30;"
      "vfdotpex.s.h %[s6], %[qb], x31;"
      : [s0] "+&r"(s0), [s1] "+&r"(s1), [s2] "+&r"(s2), [s3] "+&r"(s3),
        [s4] "+&r"(s4), [s5] "+&r"(s5), [s6] "+&r"(s6), [p] "+&r"(p),
        [k] "+&r"(k), [qa] "=&r"(qa), [qb] "=&r"(qb)
      : [step] "r"(step), [back] "r"(back_a), [end] "r"(end)
      : "x5", "x6", "x7", "x28", "x29", "x30", "x31", "memory");

  p = q;
  asm volatile(                    // pass B: spans 7 to 11, S and L
      "p.lw %[qa], 4(%[p]!);"      // q, word 0
      "p.lw x5, %[step](%[k7]!);"  // span 7
      "p.lw x6, %[step](%[k7]!);"  // span 8
      "p.lw x7, %[step](%[k7]!);"  // span 9
      "p.lw x28, %[step](%[k7]!);" // span 10
      "p.lw x29, %[back](%[k7]!);" // span 11, then back to span 7, one word on
      "p.lw x30, 4(%[ks]!);"       // S
      "p.lw x31, 4(%[kl]!);"       // L
      ".balign 32;"
      "1:"                    // two steps a round
      "p.lw %[qb], 4(%[p]!);" // next word of q
      "vfdotpex.s.h %[s7], %[qa], x5;"
      "p.lw x5, %[step](%[k7]!);" // span 7, next word
      "vfdotpex.s.h %[s8], %[qa], x6;"
      "p.lw x6, %[step](%[k7]!);"
      "vfdotpex.s.h %[s9], %[qa], x7;"
      "p.lw x7, %[step](%[k7]!);"
      "vfdotpex.s.h %[s10], %[qa], x28;"
      "p.lw x28, %[step](%[k7]!);"
      "vfdotpex.s.h %[s11], %[qa], x29;"
      "p.lw x29, %[back](%[k7]!);"
      "vfdotpex.s.h %[s12], %[qa], x30;"
      "p.lw x30, 4(%[ks]!);"
      "vfdotpex.s.h %[s13], %[qa], x31;"
      "p.lw x31, 4(%[kl]!);"
      "p.lw %[qa], 4(%[p]!);" // the word after
      "vfdotpex.s.h %[s7], %[qb], x5;"
      "p.lw x5, %[step](%[k7]!);"
      "vfdotpex.s.h %[s8], %[qb], x6;"
      "p.lw x6, %[step](%[k7]!);"
      "vfdotpex.s.h %[s9], %[qb], x7;"
      "p.lw x7, %[step](%[k7]!);"
      "vfdotpex.s.h %[s10], %[qb], x28;"
      "p.lw x28, %[step](%[k7]!);"
      "vfdotpex.s.h %[s11], %[qb], x29;"
      "p.lw x29, %[back](%[k7]!);"
      "vfdotpex.s.h %[s12], %[qb], x30;"
      "p.lw x30, 4(%[ks]!);"
      "vfdotpex.s.h %[s13], %[qb], x31;"
      "p.lw x31, 4(%[kl]!);"
      "bne %[p], %[end], 1b;"
      // the last two steps
      "p.lw %[qb], 4(%[p]!);"
      "vfdotpex.s.h %[s7], %[qa], x5;"
      "p.lw x5, %[step](%[k7]!);"
      "vfdotpex.s.h %[s8], %[qa], x6;"
      "p.lw x6, %[step](%[k7]!);"
      "vfdotpex.s.h %[s9], %[qa], x7;"
      "p.lw x7, %[step](%[k7]!);"
      "vfdotpex.s.h %[s10], %[qa], x28;"
      "p.lw x28, %[step](%[k7]!);"
      "vfdotpex.s.h %[s11], %[qa], x29;"
      "p.lw x29, %[step](%[k7]!);"
      "vfdotpex.s.h %[s12], %[qa], x30;"
      "p.lw x30, 4(%[ks]!);"
      "vfdotpex.s.h %[s13], %[qa], x31;"
      "p.lw x31, 4(%[kl]!);"
      "vfdotpex.s.h %[s7], %[qb], x5;"
      "vfdotpex.s.h %[s8], %[qb], x6;"
      "vfdotpex.s.h %[s9], %[qb], x7;"
      "vfdotpex.s.h %[s10], %[qb], x28;"
      "vfdotpex.s.h %[s11], %[qb], x29;"
      "vfdotpex.s.h %[s12], %[qb], x30;"
      "vfdotpex.s.h %[s13], %[qb], x31;"
      : [s7] "+&r"(s7), [s8] "+&r"(s8), [s9] "+&r"(s9), [s10] "+&r"(s10),
        [s11] "+&r"(s11), [s12] "+&r"(s12), [s13] "+&r"(s13), [p] "+&r"(p),
        [k7] "+&r"(k7), [ks] "+&r"(ks), [kl] "+&r"(kl), [qa] "=&r"(qa),
        [qb] "=&r"(qb)
      : [step] "r"(step), [back] "r"(back_b), [end] "r"(end)
      : "x5", "x6", "x7", "x28", "x29", "x30", "x31", "memory");
#else
  // one pass over the 14 logits: 6 registers for the loaded words, each filled
  // two or three times a step
  // after span 11: back to span 0, one word on
  int32_t back = 4 - 11 * step;
  // where q ends
  const __fp16 *const end = q + d;

  asm volatile( // logits
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
      : [s0] "+&r"(s0), [s1] "+&r"(s1), [s2] "+&r"(s2), [s3] "+&r"(s3),
        [s4] "+&r"(s4), [s5] "+&r"(s5), [s6] "+&r"(s6), [s7] "+&r"(s7),
        [s8] "+&r"(s8), [s9] "+&r"(s9), [s10] "+&r"(s10), [s11] "+&r"(s11),
        [s12] "+&r"(s12), [s13] "+&r"(s13), [q] "+&r"(q), [k] "+&r"(k),
        [ks] "+&r"(ks), [kl] "+&r"(kl)
      : [step] "r"(step), [back] "r"(back), [end] "r"(end)
      : "x5", "x7", "x28", "x29", "x30", "x31", "memory");
#endif

  
  uint32_t u0, u1, u2, u3, u4, u5;
  register uint16_t *o5 asm("x5") = o;
  asm volatile( // softmax (14 logits)
      "addi sp, sp, -16;"
      "sw x5, 0(sp);" // o, parked
      // t = max((x - max) log2(e), -100), n = round(t), f = t - n, |f| <= 1/2
      // exp(x - max) / c3 = 2^n (((f + A) f + B) f + C), w = exp / sum of 14
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
      "li x7, %[cc];"             // the cubic: A, B, C
      "fmax.s %[q], %[q], %[ks];" // the max
      "li x28, %[cb];"
      "li x29, %[ca];"
      "fmul.s x31, %[q], x5;" // max . log2(e)
      "mv %[k], x31;"
      "mv %[kl], x31;"
      "mv %[back], x31;"
      "fmsub.s %[k], %[s0], x5, %[k];" // t = x log2(e) - max log2(e).  [0,1,2]
      "fmsub.s %[kl], %[s1], x5, %[kl];"
      "fmsub.s %[back], %[s2], x5, %[back];"
      "fmax.s %[s0], %[k], x6;" // t, not below -100
      "fmax.s %[s1], %[kl], x6;"
      "fmax.s %[s2], %[back], x6;"
      "fcvt.w.s %[q], %[s0], rne;" // n = round(t)
      "fcvt.w.s %[ks], %[s1], rne;"
      "fcvt.w.s %[step], %[s2], rne;"
      "fcvt.s.w %[k], %[q];" // n, as fp32
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s0], %[s0], %[k];" // f = t - n, in [-0.5, 0.5]
      "fsub.s %[s1], %[s1], %[kl];"
      "fsub.s %[s2], %[s2], %[back];"
      "fadd.s %[k], %[s0], x29;" // f + A
      "fadd.s %[kl], %[s1], x29;"
      "fadd.s %[back], %[s2], x29;"
      "fmul.s %[k], %[k], %[s0];" // (f + A) f + B
      "fmul.s %[kl], %[kl], %[s1];"
      "fmul.s %[back], %[back], %[s2];"
      "fadd.s %[k], %[k], x28;"
      "fadd.s %[kl], %[kl], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[k], %[k], %[s0];" // (..) f + C: 2^f / c3
      "fmul.s %[kl], %[kl], %[s1];"
      "fmul.s %[back], %[back], %[s2];"
      "fadd.s %[k], %[k], x7;"
      "fadd.s %[kl], %[kl], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[q], %[q], 23;" // n, into the exponent field
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s0], %[k], %[q];" // exp(x - max) / c3 = 2^n (2^f / c3)
      "add %[s1], %[kl], %[ks];"
      "add %[s2], %[back], %[step];"
      "mv %[k], x31;" // [3,4,5]
      "mv %[kl], x31;"
      "mv %[back], x31;"
      "fmsub.s %[k], %[s3], x5, %[k];"
      "fmsub.s %[kl], %[s4], x5, %[kl];"
      "fmsub.s %[back], %[s5], x5, %[back];"
      "fmax.s %[s3], %[k], x6;"
      "fmax.s %[s4], %[kl], x6;"
      "fmax.s %[s5], %[back], x6;"
      "fcvt.w.s %[q], %[s3], rne;"
      "fcvt.w.s %[ks], %[s4], rne;"
      "fcvt.w.s %[step], %[s5], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s3], %[s3], %[k];"
      "fsub.s %[s4], %[s4], %[kl];"
      "fsub.s %[s5], %[s5], %[back];"
      "fadd.s %[k], %[s3], x29;"
      "fadd.s %[kl], %[s4], x29;"
      "fadd.s %[back], %[s5], x29;"
      "fmul.s %[k], %[k], %[s3];"
      "fmul.s %[kl], %[kl], %[s4];"
      "fmul.s %[back], %[back], %[s5];"
      "fadd.s %[k], %[k], x28;"
      "fadd.s %[kl], %[kl], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[k], %[k], %[s3];"
      "fmul.s %[kl], %[kl], %[s4];"
      "fmul.s %[back], %[back], %[s5];"
      "fadd.s %[k], %[k], x7;"
      "fadd.s %[kl], %[kl], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s3], %[k], %[q];"
      "add %[s4], %[kl], %[ks];"
      "add %[s5], %[back], %[step];"
      "mv %[k], x31;" // [6,7,8]
      "mv %[kl], x31;"
      "mv %[back], x31;"
      "fmsub.s %[k], %[s6], x5, %[k];"
      "fmsub.s %[kl], %[s7], x5, %[kl];"
      "fmsub.s %[back], %[s8], x5, %[back];"
      "fmax.s %[s6], %[k], x6;"
      "fmax.s %[s7], %[kl], x6;"
      "fmax.s %[s8], %[back], x6;"
      "fcvt.w.s %[q], %[s6], rne;"
      "fcvt.w.s %[ks], %[s7], rne;"
      "fcvt.w.s %[step], %[s8], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s6], %[s6], %[k];"
      "fsub.s %[s7], %[s7], %[kl];"
      "fsub.s %[s8], %[s8], %[back];"
      "fadd.s %[k], %[s6], x29;"
      "fadd.s %[kl], %[s7], x29;"
      "fadd.s %[back], %[s8], x29;"
      "fmul.s %[k], %[k], %[s6];"
      "fmul.s %[kl], %[kl], %[s7];"
      "fmul.s %[back], %[back], %[s8];"
      "fadd.s %[k], %[k], x28;"
      "fadd.s %[kl], %[kl], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[k], %[k], %[s6];"
      "fmul.s %[kl], %[kl], %[s7];"
      "fmul.s %[back], %[back], %[s8];"
      "fadd.s %[k], %[k], x7;"
      "fadd.s %[kl], %[kl], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s6], %[k], %[q];"
      "add %[s7], %[kl], %[ks];"
      "add %[s8], %[back], %[step];"
      "mv %[k], x31;" // [9,10,11]
      "mv %[kl], x31;"
      "mv %[back], x31;"
      "fmsub.s %[k], %[s9], x5, %[k];"
      "fmsub.s %[kl], %[s10], x5, %[kl];"
      "fmsub.s %[back], %[s11], x5, %[back];"
      "fmax.s %[s9], %[k], x6;"
      "fmax.s %[s10], %[kl], x6;"
      "fmax.s %[s11], %[back], x6;"
      "fcvt.w.s %[q], %[s9], rne;"
      "fcvt.w.s %[ks], %[s10], rne;"
      "fcvt.w.s %[step], %[s11], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fcvt.s.w %[back], %[step];"
      "fsub.s %[s9], %[s9], %[k];"
      "fsub.s %[s10], %[s10], %[kl];"
      "fsub.s %[s11], %[s11], %[back];"
      "fadd.s %[k], %[s9], x29;"
      "fadd.s %[kl], %[s10], x29;"
      "fadd.s %[back], %[s11], x29;"
      "fmul.s %[k], %[k], %[s9];"
      "fmul.s %[kl], %[kl], %[s10];"
      "fmul.s %[back], %[back], %[s11];"
      "fadd.s %[k], %[k], x28;"
      "fadd.s %[kl], %[kl], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[k], %[k], %[s9];"
      "fmul.s %[kl], %[kl], %[s10];"
      "fmul.s %[back], %[back], %[s11];"
      "fadd.s %[k], %[k], x7;"
      "fadd.s %[kl], %[kl], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[q], %[q], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[step], %[step], 23;"
      "add %[s9], %[k], %[q];"
      "add %[s10], %[kl], %[ks];"
      "add %[s11], %[back], %[step];"
      "mv %[k], x31;" // [12,13]
      "mv %[kl], x31;"
      "fmsub.s %[k], %[s12], x5, %[k];"
      "fmsub.s %[kl], %[s13], x5, %[kl];"
      "fmax.s %[s12], %[k], x6;"
      "fmax.s %[s13], %[kl], x6;"
      "fcvt.w.s %[q], %[s12], rne;"
      "fcvt.w.s %[ks], %[s13], rne;"
      "fcvt.s.w %[k], %[q];"
      "fcvt.s.w %[kl], %[ks];"
      "fsub.s %[s12], %[s12], %[k];"
      "fsub.s %[s13], %[s13], %[kl];"
      "fadd.s %[k], %[s12], x29;"
      "fadd.s %[kl], %[s13], x29;"
      "fmul.s %[k], %[k], %[s12];"
      "fmul.s %[kl], %[kl], %[s13];"
      "fadd.s %[k], %[k], x28;"
      "fadd.s %[kl], %[kl], x28;"
      "fmul.s %[k], %[k], %[s12];"
      "fmul.s %[kl], %[kl], %[s13];"
      "fadd.s %[k], %[k], x7;"
      "fadd.s %[kl], %[kl], x7;"
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
        [s12] "+&r"(s12), [s13] "+&r"(s13), [q] "=&r"(u0), [k] "=&r"(u1),
        [ks] "=&r"(u2), [kl] "=&r"(u3), [step] "=&r"(u4), [back] "=&r"(u5),
        [o] "+&r"(o5)
      : [log2e] "i"(LHSM_LOG2E), [tmin] "i"(LHSM_TMIN), [ca] "i"(LHSM_A),
        [cb] "i"(LHSM_B), [cc] "i"(LHSM_C)
      : "x6", "x7", "x28", "x29", "x31", "memory");
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
      // exp(x - max) / c3 = 2^n (((f + A) f + B) f + C)
      "fmax.s x31, %[s0], %[s1];" // max, row 0
      "fmax.s x16, %[s3], %[s4];" // max, row 1
      "li x5, %[log2e];"
      "fmax.s x31, x31, %[s2];"
      "fmax.s x16, x16, %[s5];"
      "li x6, %[tmin];"
      "fmul.s x31, x31, x5;" // max log2(e), row 0
      "fmul.s x16, x16, x5;" // max log2(e), row 1
      "li x7, %[cc];"
      "li x28, %[cb];"
      "li x29, %[ca];"
      "mv %[dm], x31;"
      "mv %[nx], x31;"
      "mv %[back], x31;"
      "fmsub.s %[dm], %[s0], x5, %[dm];" // t = x log2(e) - max log2(e)
      "fmsub.s %[nx], %[s1], x5, %[nx];"
      "fmsub.s %[back], %[s2], x5, %[back];"
      "fmax.s x31, %[s6], %[s7];" // max, row 2
      "fmax.s %[s0], %[dm], x6;"  // t, not below -100
      "fmax.s %[s1], %[nx], x6;"
      "fmax.s %[s2], %[back], x6;"
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
      "fadd.s %[dm], %[s0], x29;" // 2^f / c3, the cubic
      "fadd.s %[nx], %[s1], x29;"
      "fadd.s %[back], %[s2], x29;"
      "fmul.s %[dm], %[dm], %[s0];"
      "fmul.s %[nx], %[nx], %[s1];"
      "fmul.s %[back], %[back], %[s2];"
      "fadd.s %[dm], %[dm], x28;"
      "fadd.s %[nx], %[nx], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[dm], %[dm], %[s0];"
      "fmul.s %[nx], %[nx], %[s1];"
      "fmul.s %[back], %[back], %[s2];"
      "fadd.s %[dm], %[dm], x7;"
      "fadd.s %[nx], %[nx], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[p], %[p], 23;" // n, into the exponent field
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s0], %[dm], %[p];" // exp(x - max) / c3 = 2^n (2^f / c3)
      "add %[s1], %[nx], %[ks];"
      "add %[s2], %[back], %[kl];"
      "mv %[dm], x16;" // row 1
      "mv %[nx], x16;"
      "mv %[back], x16;"
      "fmsub.s %[dm], %[s3], x5, %[dm];"
      "fmsub.s %[nx], %[s4], x5, %[nx];"
      "fmsub.s %[back], %[s5], x5, %[back];"
      "fmax.s x16, %[s9], %[s10];" // max, row 3
      "fmax.s %[s3], %[dm], x6;"
      "fmax.s %[s4], %[nx], x6;"
      "fmax.s %[s5], %[back], x6;"
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
      "fadd.s %[dm], %[s3], x29;"
      "fadd.s %[nx], %[s4], x29;"
      "fadd.s %[back], %[s5], x29;"
      "fmul.s %[dm], %[dm], %[s3];"
      "fmul.s %[nx], %[nx], %[s4];"
      "fmul.s %[back], %[back], %[s5];"
      "fadd.s %[dm], %[dm], x28;"
      "fadd.s %[nx], %[nx], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[dm], %[dm], %[s3];"
      "fmul.s %[nx], %[nx], %[s4];"
      "fmul.s %[back], %[back], %[s5];"
      "fadd.s %[dm], %[dm], x7;"
      "fadd.s %[nx], %[nx], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[p], %[p], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s3], %[dm], %[p];"
      "add %[s4], %[nx], %[ks];"
      "add %[s5], %[back], %[kl];"
      "mv %[dm], x31;" // row 2
      "mv %[nx], x31;"
      "mv %[back], x31;"
      "fmsub.s %[dm], %[s6], x5, %[dm];"
      "fmsub.s %[nx], %[s7], x5, %[nx];"
      "fmsub.s %[back], %[s8], x5, %[back];"
      "fmax.s %[s6], %[dm], x6;"
      "fmax.s %[s7], %[nx], x6;"
      "fmax.s %[s8], %[back], x6;"
      "fcvt.w.s %[p], %[s6], rne;"
      "fcvt.w.s %[ks], %[s7], rne;"
      "fcvt.w.s %[kl], %[s8], rne;"
      "fcvt.s.w %[dm], %[p];"
      "fcvt.s.w %[nx], %[ks];"
      "fcvt.s.w %[back], %[kl];"
      "fsub.s %[s6], %[s6], %[dm];"
      "fsub.s %[s7], %[s7], %[nx];"
      "fsub.s %[s8], %[s8], %[back];"
      "fadd.s %[dm], %[s6], x29;"
      "fadd.s %[nx], %[s7], x29;"
      "fadd.s %[back], %[s8], x29;"
      "fmul.s %[dm], %[dm], %[s6];"
      "fmul.s %[nx], %[nx], %[s7];"
      "fmul.s %[back], %[back], %[s8];"
      "fadd.s %[dm], %[dm], x28;"
      "fadd.s %[nx], %[nx], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[dm], %[dm], %[s6];"
      "fmul.s %[nx], %[nx], %[s7];"
      "fmul.s %[back], %[back], %[s8];"
      "fadd.s %[dm], %[dm], x7;"
      "fadd.s %[nx], %[nx], x7;"
      "fadd.s %[back], %[back], x7;"
      "slli %[p], %[p], 23;"
      "slli %[ks], %[ks], 23;"
      "slli %[kl], %[kl], 23;"
      "add %[s6], %[dm], %[p];"
      "add %[s7], %[nx], %[ks];"
      "add %[s8], %[back], %[kl];"
      "mv %[dm], x16;" // row 3
      "mv %[nx], x16;"
      "mv %[back], x16;"
      "fmsub.s %[dm], %[s9], x5, %[dm];"
      "fmsub.s %[nx], %[s10], x5, %[nx];"
      "fmsub.s %[back], %[s11], x5, %[back];"
      "fmax.s %[s9], %[dm], x6;"
      "fmax.s %[s10], %[nx], x6;"
      "fmax.s %[s11], %[back], x6;"
      "fcvt.w.s %[p], %[s9], rne;"
      "fcvt.w.s %[ks], %[s10], rne;"
      "fcvt.w.s %[kl], %[s11], rne;"
      "fcvt.s.w %[dm], %[p];"
      "fcvt.s.w %[nx], %[ks];"
      "fcvt.s.w %[back], %[kl];"
      "fsub.s %[s9], %[s9], %[dm];"
      "fsub.s %[s10], %[s10], %[nx];"
      "fsub.s %[s11], %[s11], %[back];"
      "fadd.s %[dm], %[s9], x29;"
      "fadd.s %[nx], %[s10], x29;"
      "fadd.s %[back], %[s11], x29;"
      "fmul.s %[dm], %[dm], %[s9];"
      "fmul.s %[nx], %[nx], %[s10];"
      "fmul.s %[back], %[back], %[s11];"
      "fadd.s %[dm], %[dm], x28;"
      "fadd.s %[nx], %[nx], x28;"
      "fadd.s %[back], %[back], x28;"
      "fmul.s %[dm], %[dm], %[s9];"
      "fmul.s %[nx], %[nx], %[s10];"
      "fmul.s %[back], %[back], %[s11];"
      "fadd.s %[dm], %[dm], x7;"
      "fadd.s %[nx], %[nx], x7;"
      "fadd.s %[back], %[back], x7;"
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
      "fcvt.h.s %[s0], %[s0];" // fp16, in the low half
      "fcvt.h.s %[s1], %[s1];"
      "fcvt.h.s %[s2], %[s2];"
      "fcvt.h.s %[s3], %[s3];"
      "fcvt.h.s %[s4], %[s4];"
      "fcvt.h.s %[s5], %[s5];"
      "fcvt.h.s %[s6], %[s6];"
      "fcvt.h.s %[s7], %[s7];"
      "fcvt.h.s %[s8], %[s8];"
      "fcvt.h.s %[s9], %[s9];"
      "fcvt.h.s %[s10], %[s10];"
      "fcvt.h.s %[s11], %[s11];"
      : [s0] "+&r"(s0), [s1] "+&r"(s1), [s2] "+&r"(s2), [s3] "+&r"(s3),
        [s4] "+&r"(s4), [s5] "+&r"(s5), [s6] "+&r"(s6), [s7] "+&r"(s7),
        [s8] "+&r"(s8), [s9] "+&r"(s9), [s10] "+&r"(s10), [s11] "+&r"(s11),
        [p] "+&r"(p), [ks] "+&r"(ks), [kl] "+&r"(kl), [dm] "+&r"(dm),
        [nx] "+&r"(nx), [back] "+&r"(back)
      : [end] "r"(end), [log2e] "i"(LHSM_LOG2E), [tmin] "i"(LHSM_TMIN),
        [ca] "i"(LHSM_A), [cb] "i"(LHSM_B), [cc] "i"(LHSM_C)
      : "x5", "x6", "x7", "x16", "x28", "x29", "x30", "x31", "memory");
  // rows skip to 3, as fp16, 3 a row
  if (skip == 0) {
    o[0] = *(__fp16 *)&s0;
    o[1] = *(__fp16 *)&s1;
    o[2] = *(__fp16 *)&s2;
    o += 3;
  }
  if (skip <= 1) {
    o[0] = *(__fp16 *)&s3;
    o[1] = *(__fp16 *)&s4;
    o[2] = *(__fp16 *)&s5;
    o += 3;
  }
  if (skip <= 2) {
    o[0] = *(__fp16 *)&s6;
    o[1] = *(__fp16 *)&s7;
    o[2] = *(__fp16 *)&s8;
    o += 3;
  }
  o[0] = *(__fp16 *)&s9;
  o[1] = *(__fp16 *)&s10;
  o[2] = *(__fp16 *)&s11;
}


static inline __attribute__((always_inline)) void
lhsm_span_batch(const lhsm_args_t *a, const __fp16 *lpr, uint32_t b,
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
  const __fp16 *lrow = lpr + l * a->lpr_line + w * a->lpr_step;
  // G[h][0][0..2]: the biases of a span query
  const __fp16 *g = a->bias + h * (1 + a->n_hub) * 3;
  // a_span row of span i0: r = u * n_span + i, 3 slots a row
  __fp16 *o = a->a_span + 3 * (u * a->n_span + w * win + i0);
  // the 4 rows against their own k, S_w and L_w (biases G[h][0][0..2]); the
  // batch's own rows, i0 - r0 to 3, to o
  lhsm_span_weights(row + qo, (int32_t)(2 * a->d_model),
                    (int32_t)(2 * a->span_step), srow + ko, lrow + ko, d,
                    lhsm_f32(g), lhsm_f32(g + 1), lhsm_f32(g + 2), i0 - r0, o);
}

static inline __attribute__((always_inline)) void
lhsm_hub_row(const lhsm_args_t *a, const __fp16 *lpr, uint32_t r,
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
  const __fp16 *lrow = lpr + l * a->lpr_line + w * a->lpr_step;
  // the first span row of window w
  const __fp16 *span0 = a->span + l * a->span_line + w * win * a->span_step;
  // G[h][1 + eta]: the 3 biases of this hub query
  const __fp16 *g = a->bias + (h * (1 + n_hub) + 1 + eta) * 3;
  // the bias of the span keys
  float g_span = lhsm_f32(g);
  // in the short last window, plus delta[h], stored after all of G
  if (n_live < win) {
    g_span += lhsm_f32(a->bias + n_heads * (1 + n_hub) * 3 + h);
  }

  lhsm_hub_weights((eta ? lrow : srow) + qo, span0 + ko,
                   (int32_t)(2 * a->span_step), srow + ko, lrow + ko, d, n_live,
                   g_span, lhsm_f32(g + 1), lhsm_f32(g + 2),
                   (uint16_t *)(a->a_hub + r * (win + 2)));
}

// ==== the kernel ==========================================================

// Always inlined: with a static const a, as in the apps, every field is a
// constant in the code, and nothing is loaded from the struct (in L2)
static inline __attribute__((always_inline)) void
celere_lhsm_f16(const lhsm_args_t *a, uint32_t core_id,
                uint32_t num_cores) {
  // the local probes of this core's group: in 1.4 every core reads the same
  // row, so the app keeps one copy of it in the banks of each group
  const __fp16 *lpr = a->lpr + (core_id / NUM_CORES_PER_GROUP) * a->lpr_copy;
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
      lhsm_hub_row(a, lpr, core_id, n_win);
    } else {
      // the others (160 in 1.4, 208 in 2.4) take the span batches roundrobin
      for (uint32_t b = core_id - n_hr; b < n_sb; b += num_cores - n_hr) {
        lhsm_span_batch(a, lpr, b, n_win);
      }
    }
  } else {
    // fewer cores, or as many as hub rows: every hub row round robin,
    for (uint32_t r = core_id; r < n_hr; r += num_cores) {
      lhsm_hub_row(a, lpr, r, n_win);
    }
    // then every span batch round robin
    for (uint32_t b = core_id; b < n_sb; b += num_cores) {
      lhsm_span_batch(a, lpr, b, n_win);
    }
  }
}
