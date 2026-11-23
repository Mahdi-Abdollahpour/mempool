// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


// CEViT channel estimator on multi-cluster TensorPool, compute-then-transfer pipeline:
//
//   stage 1  front end and Q|K|V projection    CE_S1_LANES lanes
//   stage 2  attention core                    CE_S2_GROUPS x CE_S2_QSPLIT lanes
//   stage 3  FFN and output projection         CE_S3_LANES lanes
//

#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "builtins_v2.h"
#include "baremetal/mempool_redmule_f16.h"
#include "baremetal/mempool_matmul_f16.h"
#include "baremetal/mempool_softmax_f16.h"
#include "baremetal/mempool_layernorm_f16.h"
#include "baremetal/mempool_gelu_f16.h"
#include "baremetal/mempool_relu_f16.h"
#include "baremetal/mempool_axpy_f16.h"
#include "baremetal/mempool_transpose_f16.h"
#include "baremetal/mempool_patchify_f16.h"

// ==== knobs ===================================================================

// model: the received grid and the patches
#ifndef CE_N_PRB
#define CE_N_PRB        273u   // physical resource blocks
#endif
#ifndef CE_SC_PER_PRB
#define CE_SC_PER_PRB   12u    // subcarriers per PRB
#endif
#ifndef CE_N_SYM
#define CE_N_SYM        14u    // OFDM symbols
#endif
#ifndef CE_PATCH_SC
#define CE_PATCH_SC     12u    // subcarriers in a patch
#endif
#ifndef CE_PATCH_SYM
#define CE_PATCH_SYM    7u     // symbols in a patch
#endif

// model: the encoder
#ifndef CE_D_MODEL
#define CE_D_MODEL      128u   // token width
#endif
#ifndef CE_HEADS
#define CE_HEADS        4u     // attention heads
#endif
#ifndef CE_FFN
#define CE_FFN          256u   // width of the two hidden FFN layers
#endif

// mapping: clusters per stage
#ifndef CE_S1_LANES
#define CE_S1_LANES     2u     // front end, split over the tokens
#endif
#ifndef CE_S2_GROUPS
#define CE_S2_GROUPS    2u     // attention core: head groups
#endif
#ifndef CE_S2_QSPLIT
#define CE_S2_QSPLIT    3u     // attention core: query slabs of each head group
#endif
#ifndef CE_S3_LANES
#define CE_S3_LANES     2u     // FFN and output projection, split over the tokens
#endif
#ifndef CE_S1_PAD
#define CE_S1_PAD       0u     // idle clusters after stage 1
#endif
#ifndef CE_S2_PAD
#define CE_S2_PAD       0u     // idle clusters after stage 2
#endif

// LPDDR
#ifndef CE_LPDDR_SOUTH
#define CE_LPDDR_SOUTH  4u     // south LPDDR nodes: model input, attention residual
#endif
#ifndef CE_LPDDR_NORTH
#define CE_LPDDR_NORTH  4u     // north LPDDR nodes: the estimate
#endif
#ifndef CE_RES_PIECES
#define CE_RES_PIECES   2u     // pieces the attention residual is cut into in LPDDR
#endif
#ifndef CE_RES_SLOTS
#define CE_RES_SLOTS    4u     // items the residual rotates over in LPDDR
#endif

// run
#ifndef CE_N_ITER
#define CE_N_ITER       1u     // items streamed through the pipeline
#endif
#ifndef CE_BENCH
#define CE_BENCH        0      // 1: time every kernel and every copy
#endif

// ==== integer helpers (compile time) ==========================================
#define CE_CDIV(a, b)     (((a) + (b) - 1u) / (b))   // a/b, rounding up
#define CE_RUP(a, b)      (CE_CDIV(a, b) * (b))      // a rounded up to a multiple of b
#define CE_MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CE_MAX3(a, b, c)  CE_MAX(CE_MAX(a, b), c)
#define CE_LINE           32u   // one DMA line, in fp16 elements
#define CE_BYTES          2u    // fp16
// GEMM rows: RedMulE splits them over its engines, so they are padded to a multiple
#define CE_M(rows)        CE_RUP(rows, NUM_REDMULE_TILES)

// ==== the network built from the knobs ========================================
// the received grid and its tokens
#define CE_CH           2u                                     // real and imaginary part
#define CE_NF           (CE_N_PRB * CE_SC_PER_PRB)             // subcarriers
#define CE_TOK_BLK      (CE_N_SYM / CE_PATCH_SYM * CE_CH)      // tokens of one block of CE_PATCH_SC subcarriers
#define CE_TOKENS       (CE_NF / CE_PATCH_SC * CE_TOK_BLK)     // tokens
#define CE_PATCH        (CE_PATCH_SC * CE_PATCH_SYM)           // token length at the input
#define CE_D_HEAD       (CE_D_MODEL / CE_HEADS)                // head width

// what one lane holds
#define TT1             (CE_TOKENS / CE_S1_LANES)              // tokens of a stage-1 lane
#define S1_QSPLIT       CE_S1_LANES                            // stage 1 holds all heads in one group
#define S1_HPL          CE_HEADS                               // heads of a stage-1 lane
#define S2_LANES        (CE_S2_GROUPS * CE_S2_QSPLIT)
#define S2_QSPLIT       CE_S2_QSPLIT
#define S2_HPL          (CE_HEADS / CE_S2_GROUPS)              // heads of a stage-2 lane
#define TQ              (CE_TOKENS / CE_S2_QSPLIT)             // queries of a stage-2 lane
#define S3_LANES        CE_S3_LANES
#define TT              (CE_TOKENS / CE_S3_LANES)              // tokens of a stage-3 lane

// subcarrier blocks the tokens of lane l touch, n tokens a lane: from the block of its first
// token to the block of its last
#define CE_BLKS(l, n)   (((l) * (n) + (n) - 1u) / CE_TOK_BLK - (l) * (n) / CE_TOK_BLK + 1u)
// grid slab of a lane: its blocks of [CE_PATCH_SC, CE_N_SYM, CE_CH], sized for the lane with the
// most (up to 4 lanes). When the token split falls inside a block, both lanes hold that block.
#define CE_SLAB(lanes, n)                                                                   \
    (CE_MAX(CE_MAX(CE_BLKS(0u, n), (lanes) > 1u ? CE_BLKS(1u, n) : 0u),                     \
            CE_MAX((lanes) > 2u ? CE_BLKS(2u, n) : 0u, (lanes) > 3u ? CE_BLKS(3u, n) : 0u)) \
     * CE_PATCH_SC * CE_N_SYM * CE_CH)
#define S1_SLAB         CE_SLAB(CE_S1_LANES, TT1)
#define S3_SLAB         CE_SLAB(CE_S3_LANES, TT)

// ==== L1 maps, in fp16 elements ==============================================
// stage 1, X: the grid slab off LPDDR
#define S1_X_PATCH      0u
// stage 1, Y
#define S1_Y_PATCHED    0u                                                        // patches
#define S1_Y_EMB        CE_RUP(S1_Y_PATCHED + CE_M(TT1) * CE_PATCH, CE_LINE)       // embedding, then + positions
#define S1_Y_RES        CE_RUP(S1_Y_EMB + CE_M(TT1) * CE_D_MODEL, CE_LINE)         // layernorm: attention input and residual
#define S1_Y_QKV        CE_RUP(S1_Y_RES + CE_M(TT1) * CE_D_MODEL, CE_LINE)         // Q|K|V
#define S1_Y_END        (S1_Y_QKV + CE_M(TT1) * 3u * CE_D_MODEL)
// stage 1, W
#define S1_W_EMB        0u                                                        // patch embedding
#define S1_W_POS        CE_RUP(S1_W_EMB + CE_PATCH * CE_D_MODEL, CE_LINE)          // positional encoding of this lane's tokens
#define S1_W_QKV        CE_RUP(S1_W_POS + S1_ADD, CE_LINE)                        // Q|K|V projection
#define S1_W_END        (S1_W_QKV + CE_D_MODEL * 3u * CE_D_MODEL)
#define S1_ADD          CE_RUP(TT1 * CE_D_MODEL, 8u)                              // positional encoding, axpy length

// stage 2, X: what boundary 1 -> 2 lands, then its second copy (S2_XPAR_E further)
#define S2_X_Q          0u                                                        // Q, one head after the other
#define S2_X_K          CE_RUP(S2_X_Q + TQ * CE_D_HEAD * S2_HPL, CE_LINE)          // K
#define S2_X_V          CE_RUP(S2_X_K + CE_TOKENS * CE_D_HEAD * S2_HPL, CE_LINE)   // V
#define S2_X_END        (S2_X_V + CE_TOKENS * CE_D_HEAD * S2_HPL)
#define S2_XPAR_E       CE_RUP(S2_X_END, CE_LINE)
// stage 2, Y
#define S2_Y_SCR        0u                                                        // attention weights
#define S2_Y_KT         CE_RUP(S2_Y_SCR + CE_M(TQ) * CE_TOKENS, CE_LINE)           // K^T
#define S2_Y_SCORES     CE_RUP(S2_Y_KT + S2_HPL * CE_D_HEAD * CE_TOKENS, CE_LINE)  // scores Q K^T
#define S2_Y_ATTNH      CE_RUP(S2_Y_SCORES + CE_M(TQ) * CE_TOKENS, CE_LINE)        // attention output, one head after the other
#define S2_Y_END        (S2_Y_ATTNH + (S2_HPL - 1u) * S2_Q_HEAD + CE_M(TQ) * CE_D_HEAD)
// one head of Q and of A*V, and one head of K, V and K^T
#define S2_Q_HEAD       (TQ * CE_D_HEAD)
#define S2_KV_HEAD      (CE_TOKENS * CE_D_HEAD)

// stage 3, X: the attention residual off LPDDR; the FFN and output GEMMs read the buffer
// base too, as placeholders
#define S3_X_RES        0u
#define S3_X_END        (CE_M(TT) * CE_MAX(CE_FFN, CE_D_MODEL))
// stage 3, Y: the FFN scratch, then what boundary 2 -> 3 lands, and its
// second copy at the end (S3_YPAR_E further)
#define S3_Y_SCR        0u
#define S3_Y_FFNR       CE_RUP(S3_Y_SCR + CE_M(TT) * CE_MAX(CE_FFN, CE_D_MODEL), CE_LINE)  // attention output + residual: FFN input
#define S3_Y_OUT        CE_RUP(S3_Y_FFNR + S3_FFNR_LEN, CE_LINE)                          // output projection
#define S3_Y_UNP        CE_RUP(S3_Y_OUT + CE_M(TT) * CE_PATCH, CE_LINE)                   // the estimate on the grid slab
#define S3_Y_END        (S3_Y_UNP + S3_SLAB)
#define S3_FFNR_LEN     CE_MAX(S3_ADD, CE_RUP(TT, 2u) * CE_D_MODEL)
#define S3_YPAR_E       (CE_RUP(S3_Y_END, CE_LINE) - S3_Y_FFNR)
// stage 3, W
#define S3_W_FF1        0u                                                        // FFN up
#define S3_W_FF2        CE_RUP(S3_W_FF1 + CE_D_MODEL * CE_FFN, CE_LINE)            // FFN hidden
#define S3_W_FF3        CE_RUP(S3_W_FF2 + CE_FFN * CE_FFN, CE_LINE)                // FFN down
#define S3_W_OUT        CE_RUP(S3_W_FF3 + CE_FFN * CE_D_MODEL, CE_LINE)            // output projection
#define S3_W_END        (S3_W_OUT + CE_D_MODEL * CE_PATCH)
#define S3_ADD          CE_RUP(TT * CE_D_MODEL, 8u)                               // skip sums, axpy length
#define S3_GELU         CE_RUP(TT * CE_FFN, 8u)                                   // GELU length

// the three buffers, each the largest claim of a stage, second copies included
#define X_ELEMS  CE_MAX3(CE_RUP(S1_SLAB, CE_LINE), S2_XPAR_E + S2_X_END, CE_RUP(S3_X_END, CE_LINE))
#define W_ELEMS  CE_MAX(CE_RUP(S1_W_END, CE_LINE), CE_RUP(S3_W_END, CE_LINE))
#define Y_ELEMS  CE_MAX3(CE_RUP(S1_Y_END, CE_LINE), CE_RUP(S2_Y_END, CE_LINE),               \
                         CE_RUP(S3_Y_END, CE_LINE) + S3_FFNR_LEN)

// ==== hand-offs and LPDDR legs, in bytes ======================================
#define DH_B            (CE_D_HEAD * CE_BYTES)
#define DM_B            (CE_D_MODEL * CE_BYTES)
// boundary 1 -> 2: Q by query slab, K and V whole to every lane of a head group
#define QKV_Y           (S1_Y_QKV * CE_BYTES)        // the projection, [TT1, 3 * d_model]
#define QKV_ROW_B       (3u * DM_B)                  // one row of it, the source stride
#define QKV_Q_SLAB      (TQ * DH_B)                  // one head of Q at the far end
#define QKV_KV_SLAB     (CE_TOKENS * DH_B)           // one head of K or V there: every token
#define QKV_Q_X         (S2_X_Q * CE_BYTES)
#define QKV_K_X         (S2_X_K * CE_BYTES)
#define QKV_V_X         (S2_X_V * CE_BYTES)
// boundary 2 -> 3: A*V, merged over the heads by the destination stride
#define HEAD_SRC        (TQ * DH_B)                  // stride between two heads of A*V in the producer
#define ATTNH_Y         (S2_Y_ATTNH * CE_BYTES)
#define FFN_RESID_Y     (S3_Y_FFNR * CE_BYTES)
// second copies of the landing zones
#define S2_XPAR         (S2_XPAR_E * CE_BYTES)
#define S3_YPAR         (S3_YPAR_E * CE_BYTES)
// LPDDR: the input on the south nodes, then the residual
#define LP_RES_NODES    (CE_RES_PIECES < CE_LPDDR_SOUTH ? CE_RES_PIECES : CE_LPDDR_SOUTH)
#define S1_RES_PIECES   (CE_RES_PIECES / CE_S1_LANES)                       // pieces a stage-1 lane writes
#define S3_RES_PIECES   (CE_RES_PIECES / CE_S3_LANES)                       // pieces a stage-3 lane reads
#define LP_RES_B        (TT * CE_D_MODEL * CE_BYTES / S3_RES_PIECES)        // one piece
#define LP_RES_OFF      CE_RUP(CE_CDIV(CE_S1_LANES, CE_LPDDR_SOUTH) * S1_SLAB * CE_BYTES, 64u)

// ==== placement ===============================================================
// stage s owns the clusters base_s .. base_s + lanes_s - 1; a pad leaves clusters idle
#define N_ITER          CE_N_ITER
#define N_STAGES        3
#define S1_BASE_CID     0u
#define S2_BASE_CID     (S1_BASE_CID + CE_S1_LANES + CE_S1_PAD)
#define S3_BASE_CID     (S2_BASE_CID + S2_LANES + CE_S2_PAD)
#define N_CLUSTERS      (S3_BASE_CID + CE_S3_LANES)      // clusters the pipeline occupies

// ==== checks ==================================================================
_Static_assert(ARCH_NUM_CLUSTER_X == 4, "this pipeline is laid out for a 4-wide grid");
_Static_assert(ARCH_NUM_CLUSTER_Y == 3, "this pipeline is laid out for a 3-tall grid");
_Static_assert(N_CLUSTERS <= ARCH_NUM_CLUSTER, "the pipeline needs more clusters than this arch has");
_Static_assert(CE_S1_LANES <= ARCH_NUM_CLUSTER_X
               && S3_BASE_CID / ARCH_NUM_CLUSTER_X == ARCH_NUM_CLUSTER_Y - 1u
               && (N_CLUSTERS - 1u) / ARCH_NUM_CLUSTER_X == ARCH_NUM_CLUSTER_Y - 1u,
               "stage 1 must sit in the south row and stage 3 in the north row, beside their LPDDR nodes (adjust CE_S1_PAD, CE_S2_PAD)");
_Static_assert(CE_NF % CE_PATCH_SC == 0u && CE_N_SYM % CE_PATCH_SYM == 0u,
               "the grid does not divide into whole patches");
_Static_assert(CE_D_MODEL % CE_HEADS == 0u && CE_D_HEAD % CE_LINE == 0u,
               "a head must be whole DMA lines: the hand-offs move one run per token and head");
_Static_assert(CE_TOKENS % 4u == 0u, "the softmax walks the keys four at a time");
_Static_assert(CE_TOKENS % CE_S1_LANES == 0u && CE_TOKENS % CE_S2_QSPLIT == 0u
               && CE_TOKENS % CE_S3_LANES == 0u && CE_HEADS % CE_S2_GROUPS == 0u,
               "the lanes must divide the tokens and the heads");
_Static_assert(CE_RES_PIECES % CE_S1_LANES == 0u && CE_RES_PIECES % CE_S3_LANES == 0u
               && LP_RES_B * S3_RES_PIECES == TT * CE_D_MODEL * CE_BYTES,
               "a lane of stage 1 or 3 must move whole pieces of the residual");
_Static_assert(CE_RES_SLOTS >= 3u,
               "the residual is read two steps after it is written, while the next one is written: 3 slots at least");
_Static_assert(LP_RES_OFF + CE_CDIV(CE_RES_PIECES, LP_RES_NODES) * CE_RES_SLOTS * LP_RES_B <= ARCH_HBM_NODE_ADDR_SPACE
               && CE_CDIV(CE_S3_LANES, CE_LPDDR_NORTH) * S3_SLAB * CE_BYTES <= ARCH_HBM_NODE_ADDR_SPACE,
               "the data of one LPDDR node do not fit its address window");

// CE_REGION { ... } marks a bench region, timed when CE_BENCH=1
#if CE_BENCH
#define CE_REGION  mempool_start_benchmark(); \
                   for (int ce_r_ = 1; ce_r_; ce_r_ = (mempool_stop_benchmark(), 0))
#else
#define CE_REGION
#endif

// ==== placement tables ========================================================

// Stages are numbered from 1; stage 0 is an idle cluster
enum { STAGE_1 = 1, STAGE_2, STAGE_3 };

// stage and lane of cluster c
#define CID_IN(c, base, lanes)  ((c) >= (base) && (c) < (base) + (lanes))
#define CID_STAGE_OF(c)  (1u * CID_IN(c, S1_BASE_CID, CE_S1_LANES)                \
                        + 2u * CID_IN(c, S2_BASE_CID, S2_LANES)                   \
                        + 3u * CID_IN(c, S3_BASE_CID, CE_S3_LANES))
#define CID_LANE_OF(c)   (((c) - S1_BASE_CID) * CID_IN(c, S1_BASE_CID, CE_S1_LANES)   \
                        + ((c) - S2_BASE_CID) * CID_IN(c, S2_BASE_CID, S2_LANES)      \
                        + ((c) - S3_BASE_CID) * CID_IN(c, S3_BASE_CID, CE_S3_LANES))

// F(c) for the first N_CLUSTERS clusters: the tables hold one entry per occupied cluster
#define CID_LIST_3(F)   F(0u), F(1u), F(2u)
#define CID_LIST_4(F)   CID_LIST_3(F), F(3u)
#define CID_LIST_5(F)   CID_LIST_4(F), F(4u)
#define CID_LIST_6(F)   CID_LIST_5(F), F(5u)
#define CID_LIST_7(F)   CID_LIST_6(F), F(6u)
#define CID_LIST_8(F)   CID_LIST_7(F), F(7u)
#define CID_LIST_9(F)   CID_LIST_8(F), F(8u)
#define CID_LIST_10(F)  CID_LIST_9(F), F(9u)
#define CID_LIST_11(F)  CID_LIST_10(F), F(10u)
#define CID_LIST_12(F)  CID_LIST_11(F), F(11u)
#if N_CLUSTERS == 3u
#define CID_LIST        CID_LIST_3
#elif N_CLUSTERS == 4u
#define CID_LIST        CID_LIST_4
#elif N_CLUSTERS == 5u
#define CID_LIST        CID_LIST_5
#elif N_CLUSTERS == 6u
#define CID_LIST        CID_LIST_6
#elif N_CLUSTERS == 7u
#define CID_LIST        CID_LIST_7
#elif N_CLUSTERS == 8u
#define CID_LIST        CID_LIST_8
#elif N_CLUSTERS == 9u
#define CID_LIST        CID_LIST_9
#elif N_CLUSTERS == 10u
#define CID_LIST        CID_LIST_10
#elif N_CLUSTERS == 11u
#define CID_LIST        CID_LIST_11
#elif N_CLUSTERS == 12u
#define CID_LIST        CID_LIST_12
#else
#error "the pipeline occupies more clusters than the 4 x 3 grid has"
#endif

static const uint8_t CID_STAGE[N_CLUSTERS] = { CID_LIST(CID_STAGE_OF) };
static const uint8_t CID_LANE [N_CLUSTERS] = { CID_LIST(CID_LANE_OF) };
// first cluster of each stage, and one past the last
static const uint8_t STAGE_BASE [N_STAGES + 2] = { 0, S1_BASE_CID, S2_BASE_CID, S3_BASE_CID, N_CLUSTERS };

// how far the second copy of a stage's landing zone is, in bytes: stage 2 in X, stage 3
// in Y, where the skip sum adds the residual to the attention output
static const uint32_t XPAR[N_STAGES + 2] = { 0u, 0u, S2_XPAR, 0u, 0u };
static const uint32_t YPAR[N_STAGES + 2] = { 0u, 0u, 0u, S3_YPAR, 0u };

// ==== LPDDR legs ==============================================================
// bycol = 1: the node in the cluster's own column. bycol = 0: the tensor is cut into
// pieces; a cluster moves `pieces` of them from lane * pieces, piece j at node j % nodes,
// share j / nodes, so two stages of any width name the same bytes. `slots`: copies that
// rotate with the item, for a tensor read on a later step than it is written.
typedef struct {
    uint8_t  edge;        // 0 = south, 1 = north
    uint8_t  buf;         // 0 = X_BUF (inbound), 1 = Y_BUF (outbound)
    uint8_t  nodes;       // LPDDR nodes this leg spreads over
    uint8_t  bycol;       // 1 = own column, 0 = by piece index
    uint8_t  pieces;      // pieces this cluster moves
    uint8_t  slots;       // item rotation; 1 = none
    uint32_t lpddr_off;   // bytes into the node, before piece and slot
    uint32_t l1_off;      // bytes into the buffer
    uint32_t bytes;       // of one piece
} lpddr_t;

static const lpddr_t LPDDR_IN[2] = {
    // stage 1: its grid slab of the model input
    { 0, 0, CE_LPDDR_SOUTH, 1, 1, 1, 0u, S1_X_PATCH * CE_BYTES, S1_SLAB * CE_BYTES },
    // stage 3: the attention residual
    { 0, 0, LP_RES_NODES, 0, S3_RES_PIECES, CE_RES_SLOTS, LP_RES_OFF, S3_X_RES * CE_BYTES, LP_RES_B },
};
static const uint8_t LPDDR_IN_FIRST[N_STAGES + 2] = { 0, 0, 1, 1, 2 };

static const lpddr_t LPDDR_OUT[2] = {
    // stage 1: the attention residual, which skips stage 2
    { 0, 1, LP_RES_NODES, 0, S1_RES_PIECES, CE_RES_SLOTS, LP_RES_OFF, S1_Y_RES * CE_BYTES, LP_RES_B },
    // stage 3: the estimate, back on the grid
    { 1, 1, CE_LPDDR_NORTH, 1, 1, 1, 0u, S3_Y_UNP * CE_BYTES, S3_SLAB * CE_BYTES },
};
static const uint8_t LPDDR_OUT_FIRST[N_STAGES + 2] = { 0, 0, 1, 1, 2 };

// ==== copies ==================================================================

// blocking 2D DMA: reps runs of chunk bytes
static inline void dma_2d_strided(uint64_t dst, uint64_t src, uint32_t chunk,
                                  uint32_t dst_stride, uint32_t src_stride,
                                  uint32_t reps)
{
    CE_REGION {
        flex_dma_async_2d(dst, src, chunk, dst_stride, src_stride, reps);
        flex_dma_async_wait_all();
    }
}

static inline void dma_2d(uint64_t dst, uint64_t src, uint32_t bytes)
{
    dma_2d_strided(dst, src, bytes, bytes, bytes, 1);
}

// what one cluster owes another is where their two ranges of the same axis overlap
static inline uint32_t lo_of(uint32_t a, uint32_t b) { return a > b ? a : b; }
static inline uint32_t hi_of(uint32_t a, uint32_t b) { return a < b ? a : b; }

// boundary 1 -> 2: the source column picks Q, K or V and the head.
// A stage-2 lane scores only its own queries but against every key, so Q goes once and
// K and V to every lane of the head group. K arrives as K, stage 2 transposes it.
static __attribute__((noinline)) void push_qkv(uint32_t lane, uint32_t x_base,
                                               uint64_t src, uint32_t par)
{
    uint32_t base = STAGE_BASE[STAGE_2];
    // the consumer computes this item on the copy of the same parity
    x_base += par * S2_XPAR;
    // this producer: its first head and its token slab
    uint32_t hp = (lane / S1_QSPLIT) * S1_HPL, qp = lane % S1_QSPLIT;
    // where Q, K and V land on the consumer
    static const uint32_t DST[3] = { QKV_Q_X, QKV_K_X, QKV_V_X };

    for (uint32_t k = 0; k < S2_LANES; k++) {
        uint32_t hk = (k / S2_QSPLIT) * S2_HPL, qk = k % S2_QSPLIT;
        uint32_t h0 = lo_of(hp, hk), h1 = hi_of(hp + S1_HPL, hk + S2_HPL);
        for (uint32_t s = 0; s < 3u; s++) {
            // stripe 0 is Q; 1 and 2 are K and V
            uint32_t q  = (s == 0u);
            uint32_t lo = q ? lo_of(qp * TT1, qk * TQ) : qp * TT1;
            uint32_t hi = q ? hi_of(qp * TT1 + TT1, qk * TQ + TQ) : qp * TT1 + TT1;
            if (lo >= hi)
                continue;
            uint32_t slab = q ? QKV_Q_SLAB : QKV_KV_SLAB;
            uint32_t dst  = lo - (q ? qk * TQ : 0u);
            for (uint32_t h = h0; h < h1; h++)
                dma_2d_strided((uint64_t)remote_cid(base + k,
                                   x_base + DST[s] + (h - hk) * slab + dst * DH_B),
                               src + (uint64_t)((lo - qp * TT1) * QKV_ROW_B
                                   + QKV_Y + s * DM_B + h * DH_B),
                               DH_B, DH_B, QKV_ROW_B, hi - lo);
        }
    }
}

// boundary 2 -> 3: one run per token and head at the head's column of a
// d_model row is the transpose back to tokens and the merge of the heads
static __attribute__((noinline)) void push_attn(uint32_t lane, uint32_t y_base,
                                                uint64_t src, uint32_t par)
{
    uint32_t base = STAGE_BASE[STAGE_3];
    // the landing zone is in the consumer's Y, so is its second copy
    y_base += par * S3_YPAR;
    uint32_t h = lane / S2_QSPLIT, q = lane % S2_QSPLIT;

    for (uint32_t k = 0; k < S3_LANES; k++) {
        uint32_t lo = lo_of(q * TQ, k * TT);
        uint32_t hi = hi_of(q * TQ + TQ, k * TT + TT);
        if (lo >= hi)
            continue;
        for (uint32_t e = 0; e < S2_HPL; e++)
            dma_2d_strided((uint64_t)remote_cid(base + k,
                               y_base + FFN_RESID_Y + (lo - k * TT) * DM_B
                               + (h * S2_HPL + e) * DH_B),
                           src + (uint64_t)(ATTNH_Y + e * HEAD_SRC
                               + (lo - q * TQ) * DH_B),
                           DH_B, DM_B, DH_B, hi - lo);
    }
}

static inline uint64_t lpddr_addr(uint32_t edge, uint32_t node, uint32_t off)
{
    return edge ? hbm_north(node, off) : hbm_south(node, off);
}

// the LPDDR legs of a stage in one direction; `item` picks the slot of a rotating leg
static __attribute__((noinline)) void lpddr_move(uint32_t stage, uint32_t lane,
                                                 uint32_t dir, uint32_t item,
                                                 uint64_t x_base, uint64_t y_base,
                                                 uint32_t par)
{
    const lpddr_t *tab   = dir ? LPDDR_OUT : LPDDR_IN;
    const uint8_t *first = dir ? LPDDR_OUT_FIRST : LPDDR_IN_FIRST;
    uint32_t col = get_pos(flex_get_cluster_id()).x;
    x_base += (uint64_t)par * XPAR[stage];
    y_base += (uint64_t)par * YPAR[stage];
    for (uint32_t e = first[stage]; e < first[stage + 1u]; e++) {
        const lpddr_t *t = &tab[e];
        uint32_t slot = t->slots > 1u ? item % t->slots : 0u;
        for (uint32_t p = 0; p < t->pieces; p++) {
            uint32_t j     = lane * t->pieces + p;
            uint32_t node  = t->bycol ? col : j % t->nodes;
            uint32_t share = t->bycol ? 0u  : j / t->nodes;
            uint64_t mem   = lpddr_addr(t->edge, node,
                                        t->lpddr_off
                                        + (share * t->slots + slot) * t->bytes);
            uint64_t l1    = (t->buf ? y_base : x_base) + t->l1_off
                             + (uint64_t)p * t->bytes;
            if (dir) dma_2d(mem, l1, t->bytes);
            else     dma_2d(l1, mem, t->bytes);
        }
    }
}

// ==== buffers =================================================================
// X_BUF: what arrives from outside the cluster, except the A*V hand-off, which lands in
// Y_BUF. W_BUF: the weights. Y_BUF: every kernel's output and the source of every copy out.
static __fp16 X_BUF[X_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")));
static __fp16 W_BUF[W_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1"), used));
static __fp16 Y_BUF[Y_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")));

// ==== stages ==================================================================

// stage 1: embedding, layernorm and Q|K|V of this lane's tokens, all heads
static __attribute__((noinline)) void compute_stage_1(uint32_t lane, uint32_t par)
{
    (void)par;
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    // patchify: grid slab to tokens
    CE_REGION {
        patchify_f16p(X_BUF + S1_X_PATCH, Y_BUF + S1_Y_PATCHED, lane * TT1, (lane * TT1 / CE_TOK_BLK) * CE_PATCH_SC, TT1,
            CE_N_SYM, CE_PATCH_SC, CE_PATCH_SYM, CE_CH, cid, nt);
        flex_intra_cluster_sync();
    }
    // patch embedding
    CE_REGION {
        redmule_synch_parallel(Y_BUF + S1_Y_PATCHED, W_BUF + S1_W_EMB, Y_BUF + S1_Y_EMB, CE_M(TT1), CE_PATCH, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    // + positional encoding
    CE_REGION {
        axpy_f16vecp_local_unrolled4(0x3C00u, W_BUF + S1_W_POS, Y_BUF + S1_Y_EMB, S1_ADD);
    }
    // layernorm
    CE_REGION {
        layernorm_parallel_2x4_f16vec(Y_BUF + S1_Y_EMB, Y_BUF + S1_Y_RES, CE_RUP(TT1, 2u), CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    // Q|K|V projection, all heads
    CE_REGION {
        redmule_synch_parallel(Y_BUF + S1_Y_RES, W_BUF + S1_W_QKV, Y_BUF + S1_Y_QKV, CE_M(TT1), CE_D_MODEL, 3u * CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
}

// stage 2: attention of this lane's queries, for its heads one at a time
static __attribute__((noinline)) void compute_stage_2(uint32_t lane, uint32_t par)
{
    (void)lane;
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    // K^T of each head
    for (uint32_t h = 0; h < S2_HPL; h++) {
        CE_REGION {
            transpose_f16p(X_BUF + S2_X_K + h * S2_KV_HEAD + par * S2_XPAR_E, Y_BUF + S2_Y_KT + h * S2_KV_HEAD, CE_TOKENS, CE_D_HEAD, cid, nt);
            flex_intra_cluster_sync();
        }
    }
    for (uint32_t h = 0; h < S2_HPL; h++) {
        // scores Q K^T
        CE_REGION {
            redmule_synch_parallel(X_BUF + S2_X_Q + h * S2_Q_HEAD + par * S2_XPAR_E, Y_BUF + S2_Y_KT + h * S2_KV_HEAD, Y_BUF + S2_Y_SCORES, CE_M(TQ), CE_D_HEAD, CE_TOKENS, GEMM, false, 0);
            flex_intra_cluster_sync();
        }
        // softmax over the keys
        CE_REGION {
            softmax_parallel_2x4_f16vec(Y_BUF + S2_Y_SCORES, Y_BUF + S2_Y_SCR, CE_RUP(TQ, 2u), CE_TOKENS, cid, nt);
            flex_intra_cluster_sync();
        }
        // attention output: weights x V
        CE_REGION {
            redmule_synch_parallel(Y_BUF + S2_Y_SCR, X_BUF + S2_X_V + h * S2_KV_HEAD + par * S2_XPAR_E, Y_BUF + S2_Y_ATTNH + h * S2_Q_HEAD, CE_M(TQ), CE_TOKENS, CE_D_HEAD, GEMM, false, 0);
            flex_intra_cluster_sync();
        }
    }
}

// stage 3: FFN block and output projection of this lane's tokens
static __attribute__((noinline)) void compute_stage_3(uint32_t lane, uint32_t par)
{
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    // skip sum: attention output + residual
    CE_REGION {
        axpy_f16vecp_local_unrolled4(0x3C00u, X_BUF + S3_X_RES, Y_BUF + S3_Y_FFNR + par * S3_YPAR_E, S3_ADD);
    }
    // layernorm
    CE_REGION {
        layernorm_parallel_2x4_f16vec(Y_BUF + S3_Y_FFNR + par * S3_YPAR_E, Y_BUF + S3_Y_SCR, CE_RUP(TT, 2u), CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    // FFN: up, GELU, hidden, GELU, down
    CE_REGION {
        redmule_synch_parallel(X_BUF, W_BUF + S3_W_FF1, Y_BUF + S3_Y_SCR, CE_M(TT), CE_D_MODEL, CE_FFN, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {
        gelu_f16(Y_BUF + S3_Y_SCR, S3_GELU, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {
        redmule_synch_parallel(X_BUF, W_BUF + S3_W_FF2, Y_BUF + S3_Y_SCR, CE_M(TT), CE_FFN, CE_FFN, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {
        gelu_f16(Y_BUF + S3_Y_SCR, S3_GELU, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {
        redmule_synch_parallel(X_BUF, W_BUF + S3_W_FF3, Y_BUF + S3_Y_SCR, CE_M(TT), CE_FFN, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    // skip sum: FFN output + its input
    CE_REGION {
        axpy_f16vecp_local_unrolled4(0x3C00u, Y_BUF + S3_Y_FFNR + par * S3_YPAR_E, Y_BUF + S3_Y_SCR, S3_ADD);
    }
    // output projection to patches
    CE_REGION {
        redmule_synch_parallel(X_BUF, W_BUF + S3_W_OUT, Y_BUF + S3_Y_OUT, CE_M(TT), CE_D_MODEL, CE_PATCH, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    // unpatchify: tokens back to the grid slab
    CE_REGION {
        unpatchify_f16p(Y_BUF + S3_Y_OUT, Y_BUF + S3_Y_UNP, lane * TT, (lane * TT / CE_TOK_BLK) * CE_PATCH_SC, TT,
            CE_N_SYM, CE_PATCH_SC, CE_PATCH_SYM, CE_CH, cid, nt);
        flex_intra_cluster_sync();
    }
}

int main()
{
    uint32_t cid     = flex_get_cluster_id();
    uint32_t eoc_val = 0;
    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : 0u;
    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;
    // stage s works on item m at step m + s - 1
    uint32_t phase   = stage - 1u;
    // buffer addresses, the same in every cluster: a producer writes a peer's L1 with its own
    const uint32_t x_off = (uint32_t)(uintptr_t)X_BUF;
    const uint32_t y_off = (uint32_t)(uintptr_t)Y_BUF;

    flex_barrier_xy_init();
    flex_global_barrier_xy();
    // fp16 1.0 everywhere before timing, so no NaN or subnormal slows the cores down
    for (uint32_t i = mempool_get_core_id(); i < X_ELEMS; i += mempool_get_core_count()) X_BUF[i] = (__fp16)1.0f;
    for (uint32_t i = mempool_get_core_id(); i < Y_ELEMS; i += mempool_get_core_count()) Y_BUF[i] = (__fp16)1.0f;
    for (uint32_t i = mempool_get_core_id(); i < W_ELEMS; i += mempool_get_core_count()) W_BUF[i] = (__fp16)1.0f;
    flex_intra_cluster_sync();
    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();
    flex_global_barrier_xy();

    // Each step: the LPDDR read, compute, then the copies out, and one barrier closing the
    // step. A producer may push the next item while its consumer still computes, so it
    // writes the copy of its own item's parity, never the one being read.
    for (uint32_t step = 0; step < N_ITER + N_STAGES - 1; step++) {
        uint32_t active = (cid < N_CLUSTERS) && (stage != 0u) && (phase <= step)
                          && (step - phase < N_ITER);
        uint32_t item   = step - phase;
        uint32_t par    = item & 1u;

        if (active) {
            if (LPDDR_IN_FIRST[stage] != LPDDR_IN_FIRST[stage + 1u]) {
                // the cores wait for this item's input off LPDDR
                if (flex_is_dm_core())
                    lpddr_move(stage, lane, 0u, item,
                               (uint64_t)(uintptr_t)X_BUF,
                               (uint64_t)(uintptr_t)Y_BUF, par);
                flex_intra_cluster_sync();
            }
            switch (stage) {
                case STAGE_1: compute_stage_1(lane, par); break;
                case STAGE_2: compute_stage_2(lane, par); break;
                case STAGE_3: compute_stage_3(lane, par); break;
                default: break;
            }
            // the LPDDR legs out, then the boundary this stage feeds
            if (flex_is_dm_core()) {
                lpddr_move(stage, lane, 1u, item, (uint64_t)(uintptr_t)X_BUF,
                           (uint64_t)(uintptr_t)Y_BUF, par);
                if (stage == STAGE_1)
                    push_qkv (lane, x_off, (uint64_t)(uintptr_t)Y_BUF, par);
                else if (stage == STAGE_2)
                    push_attn(lane, y_off, (uint64_t)(uintptr_t)Y_BUF, par);
            }
        }
        flex_global_barrier_xy();
    }

    if (flex_get_core_id() == 0 && cid == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
