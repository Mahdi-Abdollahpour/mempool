// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Mahdi Abdollahpour, University of Bologna

#include <stdint.h>

#include "builtins_v2.h"
#include "dma.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "data_local_hub_softmax_f16.h"

#include "baremetal/mempool_checks.h"
#include "baremetal/mempool_local_hub_softmax_f16.h" // lhsm_args_t and the kernel

#ifndef LHSM_TOL
#define LHSM_TOL (0.001f)
#endif
#ifndef LHSM_VERBOSE
#define LHSM_VERBOSE 0
#endif

#define QKV_N (lhsm_ROWS * lhsm_DQKV)
#define LHSM_STR_(x) #x
#define LHSM_STR(x) LHSM_STR_(x)


static __fp16 qkv[QKV_N] __attribute__((aligned(64), section(".l1")));
static __fp16 probe[lhsm_DQKV] __attribute__((aligned(64), section(".l1")));
static __fp16 bias1[lhsm_BIAS1_N] __attribute__((aligned(64), section(".l1")));
static __fp16 bias2[lhsm_BIAS2_N] __attribute__((aligned(64), section(".l1")));
static __fp16 a_span1[lhsm_ASPAN_N]
    __attribute__((aligned(64), section(".l1")));
static __fp16 a_hub1[lhsm_AHUB1_N] __attribute__((aligned(64), section(".l1")));
static __fp16 a_span2[lhsm_ASPAN_N]
    __attribute__((aligned(64), section(".l1")));
static __fp16 a_hub2[lhsm_AHUB2_N] __attribute__((aligned(64), section(".l1")));

static const lhsm_args_t call1 = {
    .span = qkv,
    .span_line = lhsm_SPANS * lhsm_DQKV,
    .span_step = lhsm_DQKV,
    .sum = qkv + lhsm_ROW_S * lhsm_DQKV,
    .sum_line = lhsm_NWIN * lhsm_DQKV,
    .sum_step = lhsm_DQKV,
    .lpr = probe,
    .lpr_line = 0u,
    .lpr_step = 0u,
    .bias = bias1,
    .a_span = a_span1,
    .a_hub = a_hub1,
    .n_lines = lhsm_LINES,
    .n_heads = lhsm_HEADS,
    .n_hub = 2u,
    .n_span = lhsm_SPANS,
    .win = lhsm_WIN,
    .d_model = lhsm_DMODEL,
    .d_head = lhsm_DHEAD,
};
static const lhsm_args_t call2 = {
    .span = qkv,
    .span_line = lhsm_SPANS * lhsm_DQKV,
    .span_step = lhsm_DQKV,
    .sum = qkv + lhsm_ROW_S * lhsm_DQKV,
    .sum_line = lhsm_NWIN * lhsm_DQKV,
    .sum_step = lhsm_DQKV,
    .lpr = qkv + lhsm_ROW_LL * lhsm_DQKV,
    .lpr_line = lhsm_NWIN * lhsm_DQKV,
    .lpr_step = lhsm_DQKV,
    .bias = bias2,
    .a_span = a_span2,
    .a_hub = a_hub2,
    .n_lines = lhsm_LINES,
    .n_heads = lhsm_HEADS,
    .n_hub = 1u,
    .n_span = lhsm_SPANS,
    .win = lhsm_WIN,
    .d_model = lhsm_DMODEL,
    .d_head = lhsm_DHEAD,
};

int main(void) {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  mempool_barrier_init(core_id);

  /* the inputs from L2 to L1 (not timed) */
  if (core_id == 0u) {
    dma_memcpy_blocking(qkv, l2_qkv, QKV_N * sizeof(__fp16));
    dma_memcpy_blocking(probe, l2_probe, lhsm_DQKV * sizeof(__fp16));
    dma_memcpy_blocking(bias1, l2_bias1, lhsm_BIAS1_N * sizeof(__fp16));
    dma_memcpy_blocking(bias2, l2_bias2, lhsm_BIAS2_N * sizeof(__fp16));
  }
  mempool_barrier(num_cores);

  for (uint32_t call = 1u; call <= 2u; call++) {
    for (uint32_t run = 1u; run <= 2u; run++) {
      const mempool_timer_t t0 = mempool_get_timer();
      mempool_start_benchmark();
      local_hub_softmax_f16(call == 1u ? &call1 : &call2, core_id, num_cores);
      mempool_barrier(num_cores);
      mempool_stop_benchmark();
      const mempool_timer_t t1 = mempool_get_timer();
      if (core_id == 0u) {
        printf("[LHSM] call %u run %u: %u cycles (kernel and barrier)\n", call,
               run, t1 - t0);
      }
      mempool_barrier(num_cores);
    }
  }


  mempool_check_f16(a_span1, l2_aspan1, lhsm_ASPAN_N, 0.01f, 1);
  mempool_check_f16(a_hub1, l2_ahub1, lhsm_AHUB1_N,  0.01f, 1);
  mempool_check_f16(a_span2, l2_aspan2, lhsm_ASPAN_N,  0.01f, 1);
  mempool_check_f16(a_hub2, l2_ahub2, lhsm_AHUB2_N,  0.01f, 1);
  mempool_barrier(num_cores);
  return 0;
}
