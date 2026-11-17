// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


/* CELERE-Spectral channel estimator on multi-cluster TensorPool:
 *
 *   stage 1  front end                 2 lanes of 8 lines
 *   stage 2  attention stage 1          8 lanes of 2 lines
 *   stage 3  attention stage 2          8 lanes of 2 lines
 *   stage 4  FFN and patch decoder      8 lanes of 2 lines
 *   stage 5  Doppler head               4 lanes of 16 elements
 */

#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "builtins_v2.h"
#include "baremetal/mempool_redmule_f16.h"
#include "baremetal/mempool_matmul_f16.h"
#include "baremetal/mempool_softmax_f16.h"
#include "baremetal/mempool_layernorm_f16.h"
#include "baremetal/mempool_gelu_f16.h"
#include "baremetal/mempool_axpy_f16.h"
#include "baremetal/mempool_pool_f16.h"
#include "baremetal/mempool_permute_f16.h"

/* ==== knobs, sizes, L1 maps and checks =======================================
 *
 * Every size, offset and table entry of the app is a name computed here from a
 * few knobs. Set a knob on the build line:
 *
 *     make celere_spectral APP_DEFINES="-DCE_N_ITER=6 -DCE_BENCH=1"
 *
 * Units: element counts and offsets are fp16 elements; names ending in _B are
 * bytes; names ending in _M are GEMM rows, padded to a multiple of the 16 RedMulE
 * engines. "Per lane" means one cluster's share of a stage. Row ids (F.4, 1.8a,
 * H.4, ...) are the rows of the operation table (in docs).
 * Numbers in comments are the values for the default knobs. */

/* ==== knobs: each one a define with a default, which the build line can override */

/* model: the received grid */
#ifndef CE_N_PRB
#define CE_N_PRB        273u   // physical resource blocks
#endif
#ifndef CE_SC_PER_PRB
#define CE_SC_PER_PRB   12u    // subcarriers per PRB
#endif
#ifndef CE_N_SYM
#define CE_N_SYM        14u    // OFDM symbols in the estimated slot
#endif
#ifndef CE_PILOT_SYMS
#define CE_PILOT_SYMS   2u     // DMRS symbols in the input
#endif
/* the array: dual-polarized 4 x 8 UPA, 64 receive elements */
#ifndef CE_N_POL
#define CE_N_POL        2u     // polarizations
#endif
#ifndef CE_UPA_V
#define CE_UPA_V        4u     // element rows
#endif
#ifndef CE_UPA_H
#define CE_UPA_H        8u     // element columns
#endif
/* model: tokens */
#ifndef CE_TILE_V
#define CE_TILE_V       2u     // elements per tile, vertical
#endif
#ifndef CE_TILE_H
#define CE_TILE_H       2u     // elements per tile, horizontal
#endif
#ifndef CE_SPAN_PRBS
#define CE_SPAN_PRBS    2u     // PRBs in one frequency span
#endif
#ifndef CE_WIN_SPANS
#define CE_WIN_SPANS    12u    // spans in one window
#endif
/* model: widths */
#ifndef CE_D_MODEL
#define CE_D_MODEL      256u   // token width d, a multiple of CE_LINE, so offsets counted in token rows sit on DMA lines
#endif
#ifndef CE_HEADS
#define CE_HEADS        2u     // attention heads
#endif
#ifndef CE_FFN
#define CE_FFN          512u   // hidden width of the FFN
#endif
#ifndef CE_C_HID
#define CE_C_HID        64u    // hidden width of the Doppler head
#endif

/* mapping: clusters per stage
 * Stage 1: front end | 2: attention stage 1 | 3: attention stage 2 | 4: FFN + decoder |
 * 5: Doppler head. Stages 1 to 4 split the 16 lattice lines, stage 5 the 64
 * elements; each lane count divides 16. */
#ifndef CE_S1_LANES
#define CE_S1_LANES     2u     // front end
#endif
#ifndef CE_ENC_LANES
#define CE_ENC_LANES    8u     // stages 2 and 3 each
#endif
#ifndef CE_S4_LANES
#define CE_S4_LANES     8u     // FFN and decoder
#endif
#ifndef CE_S5_LANES
#define CE_S5_LANES     4u     // Doppler head
#endif
#ifndef CE_S5_TILES
#define CE_S5_TILES     4u     // tiles of the head's output layer (H.4)
#endif
#ifndef CE_S5_DBUF
#define CE_S5_DBUF      1      // 1: two output buffers, a drain runs under the next tile's GEMM;
                               // 0: one buffer, each GEMM waits for the drain before it
#endif
/* idle clusters after a stage, so that the stages that touch LPDDR start at a column */
#ifndef CE_S1_PAD
#define CE_S1_PAD       2u
#endif
#ifndef CE_S2_PAD
#define CE_S2_PAD       0u
#endif
#ifndef CE_S3_PAD
#define CE_S3_PAD       0u
#endif
#ifndef CE_S4_PAD
#define CE_S4_PAD       0u
#endif

/* run */
#ifndef CE_N_ITER
#define CE_N_ITER       1u     // items streamed through the pipeline
#endif
#ifndef CE_BENCH
#define CE_BENCH        0      // 1: time every row and every copy, see CE_REGION below
#endif

/* stand-ins: rows without a kernel are timed waits (for the moment being). The wait
 * assumes the cores reach 280 FLOP per cycle (1 TFLOP/s at a PE utilization of 0.28,
 * at 1 GHz). */
#ifndef CE_PE_FLOP_PER_CYCLE
#define CE_PE_FLOP_PER_CYCLE 280u
#endif
#ifndef CE_LHSM_STANDIN
#define CE_LHSM_STANDIN 0      // 1: celere_lhsm_f16 is a timed wait too
#endif

#ifndef CE_LHSM_FLOP_PER_LOGIT
#define CE_LHSM_FLOP_PER_LOGIT 10u   // stand-in only: bias, max, exp, sum, divide
#endif

/* LPDDR nodes on each side, as hbm_chan_placement = [0,4,0,4] (west, north, east,
 * south) in the arch config */
#ifndef CE_LPDDR_SOUTH
#define CE_LPDDR_SOUTH  4u     // model input, summary patches
#endif
#ifndef CE_LPDDR_NORTH
#define CE_LPDDR_NORTH  4u     // the estimate
#endif

/* ==== integer helpers (compile time) ====================================== */
#define CE_CDIV(a, b)     (((a) + (b) - 1u) / (b))   // a/b, rounding up
#define CE_RUP(a, b)      (CE_CDIV(a, b) * (b))      // a rounded up to a multiple of b
#define CE_MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CE_MAX3(a, b, c)  CE_MAX(CE_MAX(a, b), c)
#define CE_LINE           32u   // one 64 B DMA line, in fp16 elements
#define CE_BYTES          2u    // fp16
#define CE_BANK_ROW       (NUM_BANKS * 4u / CE_BYTES)   // 4,096: one word in every bank, 8 KiB, group 0's banks first; X_BUF and W_BUF start on one
#define CE_GRP_COPY       (CE_BANK_ROW / NUM_GROUPS)    // 1,024: from one group's banks to the next, 2 KiB

/* ==== the network: what the knobs imply ==================================== */

/* frequency: spans and windows of one line */
#define CE_F            (CE_N_PRB * CE_SC_PER_PRB)                // 3276 subcarriers
#define CE_SPAN_SC      (CE_SPAN_PRBS * CE_SC_PER_PRB)            // 24 subcarriers a span
#define CE_N_SPAN       CE_CDIV(CE_F, CE_SPAN_SC)                 // 137 spans
#define CE_FP           (CE_N_SPAN * CE_SPAN_SC)                  // 3288 subcarriers, padded to whole spans
#define CE_N_WIN        CE_CDIV(CE_N_SPAN, CE_WIN_SPANS)          // 12 windows
#define CE_LAST         (CE_N_SPAN - (CE_N_WIN - 1u) * CE_WIN_SPANS)   // 5 spans in the last
#define CE_CROP         (CE_SPAN_PRBS * CE_N_SPAN - CE_N_PRB)     // 1 PRB cropped at the end: how many padding PRBs the last span has.
#define CE_NRS          (CE_N_SPAN + CE_N_WIN)                    // 149 rows a line: spans, summaries
#define CE_LG           (CE_N_SPAN + CE_N_WIN)                    // 149 keys of the global pass: spans, global probes
#define CE_LW           (CE_WIN_SPANS + 2u)                       // 14 slots of a local window
#define CE_LG_PAD       CE_RUP(CE_LG, 4u)                         // 152: CE_LG padded to a multiple of 4 (softmax)

/* space: elements, tiles, lines */
#define CE_N_EL         (CE_N_POL * CE_UPA_V * CE_UPA_H)          // 64 elements
#define CE_T_V          (CE_UPA_V / CE_TILE_V)                    // 2 tile rows
#define CE_T_H          (CE_UPA_H / CE_TILE_H)                    // 4 tile columns
#define CE_TILE_EL      (CE_TILE_V * CE_TILE_H)                   // 4 elements a tile
#define CE_NL           (CE_N_EL / CE_TILE_EL)                    // 16 lattice lines
/* Element order of the input: a = (p, t_h, e_h, t_v, e_v). The head input and the
 * estimate go tile by tile instead, a = (p, t_h, t_v, e_h, e_v) (D.9), so any group
 * of lines is one contiguous block of elements. The step of each index of the input,
 * in elements: */
#define CE_ST_EV        1u                                        // e_v
#define CE_ST_TV        CE_TILE_V                                 // 2: t_v
#define CE_ST_EH        CE_UPA_V                                  // 4: e_h
#define CE_ST_TH        (CE_UPA_V * CE_TILE_H)                    // 8: t_h
/* A lattice line is one tile (p, t_h, t_v) and holds its 4 elements (e_h, e_v). */

/* widths */
#define CE_RE           (CE_PILOT_SYMS * 2u)                      // 4 values a subcarrier: 2 pilot symbols x (re, im)
#define CE_RUN          (CE_SC_PER_PRB * CE_RE)                   // 48 values: one PRB of one element (C_up)
#define CE_C            (CE_SPAN_PRBS * CE_TILE_EL * CE_RUN)      // // 384 values a patch: 2 PRBs x 4 elements x 48; a multiple of CE_LINE, so offsets counted in patch rows sit on DMA lines
_Static_assert(CE_C % CE_LINE == 0u, "a patch row is not whole 64 B lines"); // so every offset counted in patch rows (S1_WIN, S4_WIN) is on a DMA line
#define CE_PER_EL       (CE_FP * CE_RE)                           // 13,152 values: the input of one element
#define CE_D_HEAD       (CE_D_MODEL / CE_HEADS)                   // 128: width of a head
#define CE_D_QKV        (3u * CE_D_MODEL)                         // 768: q|k|v of a row
#define CE_C_OUT        (CE_SC_PER_PRB * CE_N_SYM * 2u)           // 336 values: the estimate of a PRB, 12 x 14 x (re, im)
#define CE_NSUM_HR      (CE_SPAN_PRBS * CE_N_WIN)                 // 24 summary rows of the head an element: 12 windows x 2

/* whole tensors that cross a stage or chip boundary, all lanes, in elements */
#define CE_IN_ALL       (CE_N_EL * CE_PER_EL)                     // 841,728 model input
#define CE_WIN_ALL      (CE_NL * CE_N_WIN * CE_C)                 // 73,728 summary patches
#define CE_ZR_ALL       (CE_NL * CE_N_SPAN * CE_D_MODEL)          // 561,152 span tokens
#define CE_ZS_ALL       (CE_NL * CE_N_WIN * CE_D_MODEL)           // 49,152 summary tokens (or one probe family)
#define CE_HPRB_ALL     (CE_N_EL * CE_N_PRB * CE_RUN)             // 838,656 head input, PRB rows
#define CE_HSUM_ALL     (CE_N_EL * CE_NSUM_HR * CE_RUN)           // 73,728 head input, summary rows
#define CE_OUT_ALL      (CE_N_EL * CE_N_PRB * CE_C_OUT)           // 5,870,592 estimate

/* parameters, in elements (dense: matrix + bias) */
#define CE_W_EMB        (CE_C * CE_D_MODEL)                       // 98,304 W_e
#define CE_W_TT         (2u * CE_D_MODEL)                         // 512: the token-type rows b_e + t_R, b_e + t_S
#define CE_W_LN         (2u * CE_D_MODEL)                         // 512: gamma, beta
#define CE_W_QKV        (CE_D_MODEL * CE_D_QKV)                   // 196,608: W_qkv, no bias
#define CE_W_WO         (CE_D_MODEL * CE_D_MODEL + CE_D_MODEL)    // 65,792: W_o, b_o
#define CE_W_PQKV       (2u * CE_D_QKV)                           // 1,536: q|k|v of the two probe rows (1.cp)
#define CE_W_PE         (2u * CE_D_MODEL)                         // 512: the probe embeddings e_g, e_l (1.cp)
#define CE_W_ROPE1      (2u * CE_HEADS * CE_N_SPAN * (CE_D_HEAD / 2u))   // 35,072: cos, sin of the span rows (1.cr)
#define CE_W_ROPE2      (2u * CE_HEADS * CE_LG * (CE_D_HEAD / 2u))       // 38,144: cos, sin of spans, global probes (2.cr)
#define CE_W_LB1        (CE_HEADS * 3u * 3u + CE_HEADS)           // 20: local bias G [h][3][3], delta [h] (1.cl)
#define CE_W_LB2        (CE_HEADS * 2u * 3u + CE_HEADS)           // 14: local bias G [h][2][3], delta [h] (2.cl)
#define CE_W_FF1        (CE_D_MODEL * CE_FFN + CE_FFN)            // 131,584: W_1, b_1
#define CE_W_FF2        (CE_FFN * CE_D_MODEL + CE_D_MODEL)        // 131,328: W_2, b_2
#define CE_W_DEC        (CE_D_MODEL * CE_C + CE_C)                // 98,688: W_dec, b_dec
#define CE_W_HX         (CE_RUN * CE_C_HID + CE_C_HID)            // 3,136 W_x, b_x
#define CE_W_HS         (CE_RUN * CE_C_HID)                       // 3,072 W_s
#define CE_W_HO         (CE_C_HID * CE_C_OUT + CE_C_OUT)          // 21,840 W_h, b_h

/* ==== stage maps: each stage's share, its X_BUF map and its W_BUF map ======
 *
 * The layout rules:
 *   - X_BUF: the landing zone of the hand-off first (stages 2 to 5; it has a twin
 *     X_ELEMS further), then the stage's regions in row order. Every
 *     row writes a region of its own, so a row reads exactly what the row before
 *     it wrote, and nothing is overwritten while it is still needed.
 *   - every region is a whole number of 64 B DMA lines: CE_RUP(n, CE_LINE);
 *   - a GEMM's M is a multiple of the 16 RedMulE engines (the _M names), and its
 *     regions are sized with the padded M;
 *   - W_BUF holds every row's parameters, one region each, resident.
 *
 * Inside a stage the lattice rows are ordered by family, then line:
 *   [ spans R: line, span | summaries S: line, window | global probes Lg | local probes Ll ]
 * so every family of every line is contiguous, and each hand-off is a few plain
 * copies (one per family). */

/* == stage 1: front end (block F) == */
#define S1_LINES        (CE_NL / CE_S1_LANES)                     // 8 lines a lane
#define S1_IN_N         (CE_IN_ALL / CE_S1_LANES)                 // 420,864: the input of the lane's 32 elements
#define S1_XR_ROWS      (S1_LINES * CE_N_SPAN)                    // 1,096 span patches
#define S1_XS_ROWS      (S1_LINES * CE_N_WIN)                     // 96 summary patches
#define S1_TOK          (S1_XR_ROWS + S1_XS_ROWS)                 // 1,192 tokens: span and summary patches
#define S1_TOK_M        CE_RUP(S1_TOK, NUM_REDMULE_TILES)         // 1,200: GEMM rows of the embedding (F.4)
#define S1_WIN_N        (S1_XS_ROWS * CE_C)                       // 36,864: elements of the summary patches
#define S1_TT_N         (S1_TOK * CE_D_MODEL)                     // 305,152: elements of all tokens (F.5)
/* X_BUF: [ IN | X: span patches, summary patches | Z: span tokens, summary tokens ]
 * Each name is the start of a region, in fp16 elements into X_BUF. */
#define S1_IN           0u                                        // start of the model input of this lane (F.1)
#define S1_X            (S1_IN + CE_RUP(S1_IN_N, CE_LINE))        // start of the patch matrix [span | summary] (F.2, F.3)
#define S1_WIN          (S1_X + S1_XR_ROWS * CE_C)                // start of the summary patches, right after the span patches (F.3)
#define S1_Z            (S1_X + CE_RUP(S1_TOK_M * CE_C, CE_LINE)) // start of the tokens [span | summary] (F.4, F.5)
#define S1_ZS           (S1_Z + S1_XR_ROWS * CE_D_MODEL)          // start of the summary tokens, right after the span tokens
#define S1_END          (S1_Z + CE_RUP(S1_TOK_M * CE_D_MODEL, CE_LINE))   // end of the stage 1 map
/* W_BUF: each name is the start of a parameter region, in fp16 elements into W_BUF */
#define S1_W_EMB        0u                                        // start of the embedding matrix W_e (F.4)
#define S1_W_TT         (S1_W_EMB + CE_RUP(CE_W_EMB, CE_LINE))    // start of the two token-type rows (F.5)
#define S1_W_END        (S1_W_TT + CE_RUP(CE_W_TT, CE_LINE))      // end of the stage 1 weights

/* == stages 2 and 3: attention stages 1 and 2 == */
#define E_LINES         (CE_NL / CE_ENC_LANES)                    // 2 lines a lane
#define E_PAIRS         (E_LINES * CE_HEADS)                      // 4 (line, head) pairs
#define E_TE_PAIR       (NUM_REDMULE_TILES / E_PAIRS)             // 4 TEs a pair in the global pass
#define E_R_ROWS        (E_LINES * CE_N_SPAN)                     // 274 span rows
#define E_S_ROWS        (E_LINES * CE_N_WIN)                      // 24 rows of one family: S, Lg or Ll
#define E_RS_ROWS       (E_R_ROWS + E_S_ROWS)                     // 298: span and summary rows
#define E_ALL_ROWS      (E_RS_ROWS + 2u * E_S_ROWS)               // 346: all rows, probes too
/* where each row family of a lane starts, in rows; the lattice Z and the q|k|v
 * matrix QKV share this row order */
#define E_ROW_S         E_R_ROWS                                  // 274: first summary row, after the 2 x 137 span rows
#define E_ROW_LG        E_RS_ROWS                                 // 298: first global-probe row, after the 24 summaries
#define E_ROW_LL        (E_RS_ROWS + E_S_ROWS)                    // 322: first local-probe row, after the 24 global probes
#define E_RS_N          (E_RS_ROWS * CE_D_MODEL)                  // 76,288: elements of the span and summary rows, what arrives in stage 2 and leaves stage 3 (2.13)
#define E_ALL_N         (E_ALL_ROWS * CE_D_MODEL)                 // 88,576: elements of the whole lattice of a lane, probes included (1.13)
#define E_KP            CE_LG_PAD                                 // 152 keys of a pair in the global pass: 137 spans, 12 global probes, 3 pads with bias -inf (1.8a to 1.10a)
#define E_ASPAN_N       (E_PAIRS * CE_N_SPAN * 3u)                // 1,644 local weights of the span queries: 4 pairs x 137 spans x 3 keys (itself, S_w, L_w) (1.4, 1.11)
/* query rows of a pair in the global pass, padded to a multiple of its 4 TEs */
#define A1_QR           CE_RUP(CE_LG, E_TE_PAIR)                  // 152: 137 spans and 12 global probes, 38 rows a TE (1.8a)
#define A2_QR           CE_RUP(CE_N_SPAN, E_TE_PAIR)              // 140: 137 spans, 35 rows a TE; the global-probe queries are dead (2.8a)
/* hub queries of a window in the local pass */
#define A1_NHUB         2u                                        // S_w and L_w, both updated (1.4)
#define A2_NHUB         1u                                        // S_w only: nothing reads an update of L_w after this stage (2.4)
#define A1_AHUB_N       (E_PAIRS * CE_N_WIN * A1_NHUB * CE_LW)    // 1,344 local weights of the hub queries: 4 pairs x 12 windows x 2 hubs x 14 slots (1.4, 1.11)
#define A2_AHUB_N       (E_PAIRS * CE_N_WIN * A2_NHUB * CE_LW)    // 672: the same with one hub (2.4, 2.11)
/* live logits of the local pass in a lane: 3 for each span, 14 for each hub; they
 * set the time of the local-mix stand-in */
#define A1_LOC          (E_PAIRS * (CE_N_SPAN * 3u + CE_N_WIN * A1_NHUB * CE_LW))   // 2,988 = 4 x (411 + 336) (1.11)
#define A2_LOC          (E_PAIRS * (CE_N_SPAN * 3u + CE_N_WIN * A2_NHUB * CE_LW))   // 2,316 = 4 x (411 + 168) (2.11)
/* stand-ins: RoPE turns each (j, j + 64) pair of q and k with 3 FLOP an element;
 * the local mix takes 2 FLOP for each weight and value element */
#define A1_ROPE_WAIT    ((3u * 2u * E_PAIRS * CE_N_SPAN * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)        // 1,503 cycles
#define A2_ROPE_WAIT    ((3u * E_PAIRS * (CE_N_SPAN + CE_LG) * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)   // 1,568
#define A1_LAV_WAIT     ((2u * A1_LOC * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)                         // 2,731
#define A2_LAV_WAIT     ((2u * A2_LOC * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)                         // 2,117


/* stage 2 (attention stage 1). Each name is the start of a region, in fp16 elements.
 * X_BUF: [ Z | H | QKV | A_span | A_hub | Q_g | K_g^T | V_g | S_g | A_g | U_g | U | WO out ]
 * Z is the lattice of the lane: the span and summary rows arrive from stage 1, the
 * probe rows are set from the probe embeddings (1.13), and Z leaves updated in place
 * (1.14). The global-probe rows of Q_g, K_g^T and V_g are constants (1.cp), written
 * once before the run. */
#define A1_Z            0u                                        // the lattice [R | S | Lg | Ll], 346 x 256; the landing zone, with its twin at + X_ELEMS
#define A1_H_M          CE_RUP(E_RS_ROWS, NUM_REDMULE_TILES)      // 304: GEMM rows of the projection, 298 span and summary rows padded (1.2)
#define A1_WO_M         CE_RUP(E_ALL_ROWS, NUM_REDMULE_TILES)     // 352: GEMM rows of the output projection, all 346 rows padded (1.12)
#define A1_H            (A1_Z + CE_RUP(E_ALL_N, CE_LINE))         // layernorm output H [2*149 rup to 304][256]; the probes' LN is a constant (1.1, read by 1.2)
#define A1_QKV          (A1_H + CE_RUP(A1_H_M * CE_D_MODEL, CE_LINE))     // q|k|v [304][768], unrotated (1.2, read by 1.4, 1.6, 1.7)
#define A1_ASPAN        (A1_QKV + CE_RUP(A1_H_M * CE_D_QKV, CE_LINE))     // local weights of the span queries [pair][137][3] (1.4, read by 1.11)
#define A1_AHUB         (A1_ASPAN + CE_RUP(E_ASPAN_N, CE_LINE))           // local weights of the hub queries [pair][12][2][14] (1.4, read by 1.11)
#define A1_QG           (A1_AHUB + CE_RUP(A1_AHUB_N, CE_LINE))            // rotated queries Q_g [pair][152][128] (1.6, read by 1.8a)
#define A1_KT           (A1_QG + CE_RUP(E_PAIRS * A1_QR * CE_D_HEAD, CE_LINE))   // rotated keys K_g^T [pair][128][152] (1.6, read by 1.8a)
#define A1_VG           (A1_KT + CE_RUP(E_PAIRS * CE_D_HEAD * E_KP, CE_LINE))    // values V_g [pair][152][128] (1.7, read by 1.10a)
#define A1_SG           (A1_VG + CE_RUP(E_PAIRS * E_KP * CE_D_HEAD, CE_LINE))    // global logits S_g [pair][152][152], bias added in place (1.8a, 1.9a)
#define A1_AG           (A1_SG + CE_RUP(E_PAIRS * A1_QR * E_KP, CE_LINE))        // global weights A_g [pair][152][152] (1.9a, read by 1.10a)
#define A1_UG           (A1_AG + CE_RUP(E_PAIRS * A1_QR * E_KP, CE_LINE))        // global output U_g [pair][152][128] (1.10a, read by 1.11)
#define A1_U            (A1_UG + CE_RUP(E_PAIRS * A1_QR * CE_D_HEAD, CE_LINE))   // attention output U, heads merged [352][256] (1.11, read by 1.12)
#define A1_WOUT         (A1_U + CE_RUP(A1_WO_M * CE_D_MODEL, CE_LINE))           // output projection [352][256] (1.12, added into Z by 1.13)
#define A1_END          (A1_WOUT + CE_RUP(A1_WO_M * CE_D_MODEL, CE_LINE))        // end of the stage 2 map
/* W_BUF: each name is the start of a parameter region, in fp16 elements */
/* W_BUF: [ LN | W_qkv | probe q|k|v | probe embeddings | cos, sin | B_g | G, delta | W_o | Ll copies ] */
#define A1_W_LN         0u                                        // layernorm gamma, beta (1.1)
#define A1_W_QKV        (A1_W_LN + CE_RUP(CE_W_LN, CE_LINE))      // W_qkv [256][768] (1.2)
#define A1_W_PQKV       (A1_W_QKV + CE_RUP(CE_W_QKV, CE_LINE))    // constant q|k|v of the probes: the Lg row, then the Ll row (1.cp)
#define A1_W_PE         (A1_W_PQKV + CE_RUP(CE_W_PQKV, CE_LINE))  // probe embeddings: e_g, then e_l, copied into Z by 1.13 (1.cp)
#define A1_W_ROPE       (A1_W_PE + CE_RUP(CE_W_PE, CE_LINE))      // cos, sin of the span rows; not read while the rotation is a stand-in (1.cr)
#define A1_W_GB         (A1_W_ROPE + CE_RUP(CE_W_ROPE1, CE_LINE)) // global bias [h][152][152], the same for every line, pad keys at -inf (1.cg, read by 1.9a)
#define A1_W_LB         (A1_W_GB + CE_RUP(CE_HEADS * A1_QR * E_KP, CE_LINE))     // local bias G [h][3][3], delta [h] (1.cl, read by 1.4)
#define A1_W_WO         (A1_W_LB + CE_RUP(CE_W_LB1, CE_LINE))     // W_o, b_o (1.12)
#define A1_W_LLC        CE_RUP(A1_W_WO + CE_RUP(CE_W_WO, CE_LINE), CE_BANK_ROW)   // the Ll row once per group, CE_GRP_COPY apart, on a row of banks: 1.4 reads the copy in its own group (1.cp)
#define A1_W_END        (A1_W_LLC + CE_BANK_ROW)                  // end of the stage 2 weights

/* stage 3 (attention stage 2). Each name is the start of a region, in fp16 elements.
 * X_BUF: [ Z | H | QKV | A_span | A_hub | Q_g | K_g^T | V_g | S_g | A_g | U_g | U | WO out ]
 * The regions of stage 2, in the same order. Attention stage 1 updated the probes,
 * so Z arrives whole, the layernorm and the projection take all 346 rows, and the
 * probes are keys and values only: their queries are dead. Only the span and
 * summary rows are updated, and only they leave (2.14). */
#define A2_Z            0u                                        // the lattice [R | S | Lg | Ll], 346 x 256; the landing zone, with its twin at + X_ELEMS
#define A2_H_M          CE_RUP(E_ALL_ROWS, NUM_REDMULE_TILES)     // 352: GEMM rows of the projection, all 346 rows padded (2.2)
#define A2_WO_M         CE_RUP(E_RS_ROWS, NUM_REDMULE_TILES)      // 304: GEMM rows of the output projection, 298 span and summary rows padded (2.12)
#define A2_H            (A2_Z + CE_RUP(E_ALL_N, CE_LINE))         // layernorm output H: 346 rows of 256, sized 352 for the TEs (2.1, read by 2.2)
#define A2_QKV          (A2_H + CE_RUP(A2_H_M * CE_D_MODEL, CE_LINE))     // q|k|v of all 346 rows, unrotated, sized [352][768] (2.2, read by 2.4, 2.6, 2.7)
#define A2_ASPAN        (A2_QKV + CE_RUP(A2_H_M * CE_D_QKV, CE_LINE))     // local weights of the span queries [pair][137][3] (2.4, read by 2.11)
#define A2_AHUB         (A2_ASPAN + CE_RUP(E_ASPAN_N, CE_LINE))           // local weights of the one hub query S_w [pair][12][14] (2.4, read by 2.11)
#define A2_QG           (A2_AHUB + CE_RUP(A2_AHUB_N, CE_LINE))            // rotated span queries Q_g [pair][140][128], 137 real (2.6, read by 2.8a)
#define A2_KT           (A2_QG + CE_RUP(E_PAIRS * A2_QR * CE_D_HEAD, CE_LINE))   // rotated keys K_g^T [pair][128][152]: spans, then global probes (2.6, read by 2.8a)
#define A2_VG           (A2_KT + CE_RUP(E_PAIRS * CE_D_HEAD * E_KP, CE_LINE))    // values V_g [pair][152][128]: spans, then global probes (2.7, read by 2.10a)
#define A2_SG           (A2_VG + CE_RUP(E_PAIRS * E_KP * CE_D_HEAD, CE_LINE))    // global logits S_g [pair][140][152], bias added in place (2.8a, 2.9a)
#define A2_AG           (A2_SG + CE_RUP(E_PAIRS * A2_QR * E_KP, CE_LINE))        // global weights A_g [pair][140][152] (2.9a, read by 2.10a)
#define A2_UG           (A2_AG + CE_RUP(E_PAIRS * A2_QR * E_KP, CE_LINE))        // global output U_g [pair][140][128], spans only (2.10a, read by 2.11)
#define A2_U            (A2_UG + CE_RUP(E_PAIRS * A2_QR * CE_D_HEAD, CE_LINE))   // attention output U, heads merged: 298 span and summary rows, sized 304 (2.11, read by 2.12)
#define A2_WOUT         (A2_U + CE_RUP(A2_WO_M * CE_D_MODEL, CE_LINE))           // output projection, 298 rows sized [304][256] (2.12, added into Z by 2.13)
#define A2_END          (A2_WOUT + CE_RUP(A2_WO_M * CE_D_MODEL, CE_LINE))        // end of the stage 3 map
/* W_BUF: [ LN | W_qkv | cos, sin | B_g | G, delta | W_o ] */
#define A2_W_LN         0u                                        // layernorm gamma, beta (2.1)
#define A2_W_QKV        (A2_W_LN + CE_RUP(CE_W_LN, CE_LINE))      // W_qkv [256][768] (2.2)
#define A2_W_ROPE       (A2_W_QKV + CE_RUP(CE_W_QKV, CE_LINE))    // cos, sin of the span and global-probe rows; not read while the rotation is a stand-in (2.cr)
#define A2_W_GB         (A2_W_ROPE + CE_RUP(CE_W_ROPE2, CE_LINE)) // global bias [h][140][152], the same for every line, pad keys at -inf (2.cg, read by 2.9a)
#define A2_W_LB         (A2_W_GB + CE_RUP(CE_HEADS * A2_QR * E_KP, CE_LINE))     // local bias G [h][2][3], delta [h] (2.cl, read by 2.4)
#define A2_W_WO         (A2_W_LB + CE_RUP(CE_W_LB2, CE_LINE))     // W_o, b_o (2.12)
#define A2_W_END        (A2_W_WO + CE_RUP(CE_W_WO, CE_LINE))      // end of the stage 3 weights

/* == stage 4: FFN and patch decoder (block D) ==
 * Each name is the start of a region, in elements.
 * X_BUF: [ Z | BASE: X, S | H | F1 | F2 | D | head input: PRB rows, summary rows ]
 * Z (spans and summaries) arrives from stage 3. The FFN and the decoder run on all
 * 298 rows in one launch each, then D.9 cuts the decoded patches into per-PRB rows
 * of each element, the input of the head (D.10). */
#define S4_LINES        (CE_NL / CE_S4_LANES)                     // 2 lines a lane
#define S4_EL           (CE_N_EL / CE_S4_LANES)                   // 8 elements a lane: 2 lines x 4 elements a tile
#define S4_R_ROWS       (S4_LINES * CE_N_SPAN)                    // 274 span rows
#define S4_ROWS         (S4_LINES * CE_NRS)                       // 298 span and summary rows
#define S4_M            CE_RUP(S4_ROWS, NUM_REDMULE_TILES)        // 304: GEMM rows, the 298 rows padded (D.2, D.4, D.7)
#define S4_Z_N          (S4_ROWS * CE_D_MODEL)                    // 76,288: elements of the lattice that arrives from stage 3
// #define S4_XB_N         (CE_IN_ALL / CE_S4_LANES)                 // 105,216: the lane's input share from LPDDR (D.6)
#define S4_XB_N         (S4_R_ROWS * CE_C)                          // 105,216: X, the lane's span patches from LPDDR, 274 x 384 (D.6)
#define S4_WIN_N        (CE_WIN_ALL / CE_S4_LANES)                // 9,216: the lane's summary patches from LPDDR (D.6)
#define S4_BASE_N       (S4_ROWS * CE_C)                          // 114,432: the decoder residual [X | S], 298 x 384 (D.6, D.8)
#define S4_HPRB_N       (S4_EL * CE_N_PRB * CE_RUN)               // 104,832: head input, PRB rows, 8 x 273 x 48 (D.9)
#define S4_HSUM_N       (S4_EL * CE_NSUM_HR * CE_RUN)             // 9,216: head input, summary rows, 8 x 24 x 48 (D.9)
#define S4_Z            0u                                        // the lattice [R | S], 298 x 256; the landing zone, with its twin at + X_ELEMS
#define S4_XB           (S4_Z + CE_RUP(S4_Z_N, CE_LINE))          // the decoder residual BASE = [X | S] starts here, X first (D.6)
#define S4_WIN          (S4_XB + S4_XB_N)                         // S, right after X, so BASE is one tensor (D.6)
#define S4_H            (S4_WIN + CE_RUP(S4_WIN_N, CE_LINE))      // layernorm output H: 298 rows of 256, sized 304 (D.1, read by D.2)
#define S4_F1           (S4_H + CE_RUP(S4_M * CE_D_MODEL, CE_LINE))       // FFN hidden [304][512], GELU in place (D.2, D.3, read by D.4)
#define S4_F2           (S4_F1 + CE_RUP(S4_M * CE_FFN, CE_LINE))          // FFN output [304][256], + Z in place (D.4, D.5, read by D.7)
#define S4_D            (S4_F2 + CE_RUP(S4_M * CE_D_MODEL, CE_LINE))      // decoded patches [304][384], + BASE in place (D.7, D.8, read by D.9)
#define S4_BLK          (S4_D + CE_RUP(S4_M * CE_C, CE_LINE))             // head input, PRB rows [element][273][48] (D.9, sent by D.10)
#define S4_BLK_S        (S4_BLK + S4_HPRB_N)                              // head input, summary rows [element][24][48], right after (D.9, sent by D.10)
#define S4_END          (S4_BLK_S + CE_RUP(S4_HSUM_N, CE_LINE))           // end of the stage 4 map
/* W_BUF: [ LN | W_1 | W_2 | W_dec ]; each name is the start of a parameter region */
#define S4_W_LN         0u                                        // layernorm gamma, beta (D.1)
#define S4_W_FF1        (S4_W_LN + CE_RUP(CE_W_LN, CE_LINE))      // W_1, b_1 (D.2, D.3)
#define S4_W_FF2        (S4_W_FF1 + CE_RUP(CE_W_FF1, CE_LINE))    // W_2, b_2 (D.4)
#define S4_W_DEC        (S4_W_FF2 + CE_RUP(CE_W_FF2, CE_LINE))    // W_dec, b_dec (D.7)
#define S4_W_END        (S4_W_DEC + CE_RUP(CE_W_DEC, CE_LINE))    // end of the stage 4 weights

/* == stage 5: Doppler head (block H) ==
 * Each name is the start of a region, in elements.
 * X_BUF: [ x_in | h | OUT[0] | OUT[1] ]
 * x_in, the head input [PRB rows | summary rows][48], arrives from stage 4. H.1 to
 * H.3 write h only, H.4 writes OUT one tile at a time, and only the drain to LPDDR
 * (H.5) reads OUT, so an item's last drain can run under the next item's H.1 to
 * H.3. With two OUT buffers this is the largest map, and it sets the size of X_BUF. */
#define S5_EL           (CE_N_EL / CE_S5_LANES)                   // 16 elements a lane
#define S5_XR_M         (S5_EL * CE_N_PRB)                        // 4,368 PRB rows: 16 elements x 273 PRBs, 273 rows a TE (H.1)
#define S5_XS_M         (S5_EL * CE_NSUM_HR)                      // 384 summary rows: 16 elements x 24, 24 rows a TE (H.1)
#define S5_X_N          ((S5_XR_M + S5_XS_M) * CE_RUN)            // 228,096: elements of the head input that arrives from stage 4
#define S5_HX_N         (S5_XR_M * CE_C_HID)                      // 279,552: elements of h for the PRB rows, 4,368 x 64 (H.2, H.3)
#define S5_H_N          ((S5_XR_M + S5_XS_M) * CE_C_HID)          // 304,128: elements of h, PRB and summary rows
#define S5_Y_ROWS       (S5_XR_M / CE_S5_TILES)                   // 1,092 PRB rows a tile of the output layer: 4 elements (H.4)
#define S5_Y_M          CE_RUP(S5_Y_ROWS, NUM_REDMULE_TILES)      // 1,104: GEMM rows of a tile, the 1,092 padded (H.4)
#define S5_TILE_X       (S5_Y_ROWS * CE_C_HID)                    // 69,888: step from one tile's rows of h to the next (H.4)
#define S5_OUT_N        CE_RUP(S5_Y_M * CE_C_OUT, CE_LINE)        // 370,944: one OUT buffer, a tile's estimate sized [1,104][336] (H.4)
#define S5_NBUF         (CE_S5_DBUF ? 2u : 1u)                    // OUT buffers: 2 lets a drain run under the next tile's GEMM
#define S5_TILE_B       (S5_Y_ROWS * CE_C_OUT * CE_BYTES)         // 733,824 B: the real rows of a tile, what one drain sends to LPDDR (H.5)
#define S5_X            0u                                        // the head input x_in; the landing zone, with its twin at + X_ELEMS
#define S5_H            (S5_X + CE_RUP(S5_X_N, CE_LINE))          // hidden layer h [PRB rows | summary rows][64] (H.1 to H.3, read by H.4)
#define S5_OUT          (S5_H + CE_RUP(S5_H_N, CE_LINE))          // 532,224: OUT[0], with OUT[1] right after (H.4, read by H.5)
#define S5_END          (S5_OUT + S5_NBUF * S5_OUT_N)             // 1,274,112: end of the stage 5 map
/* W_BUF: [ W_x | W_s | W_h ]; each name is the start of a parameter region */
#define S5_W_HX         0u                                        // W_x [48][64], b_x: hidden layer of the PRB rows (H.1, b_x in H.2)
#define S5_W_HS         (S5_W_HX + CE_RUP(CE_W_HX, CE_LINE))      // W_s [48][64]: hidden layer of the summary rows, the window term (H.1)
#define S5_W_HO         (S5_W_HS + CE_RUP(CE_W_HS, CE_LINE))      // W_h [64][336], b_h: the output layer (H.4)
#define S5_W_END        (S5_W_HO + CE_RUP(CE_W_HO, CE_LINE))      // end of the stage 5 weights
/* ==== the two buffers ========================================================
 * X_BUF holds the largest stage map, its twin the largest landing zone, W_BUF
 * the largest weight map. */
enum { CE_X_ELEMS = CE_RUP(CE_MAX3(CE_MAX(S1_END, A1_END), CE_MAX(A2_END, S4_END), S5_END), CE_LINE) };
enum { CE_X_PAR_ELEMS = CE_RUP(CE_MAX3(E_ALL_N, S4_Z_N, S5_X_N), CE_LINE) };
enum { CE_W_ELEMS = CE_RUP(CE_MAX3(CE_MAX(S1_W_END, A1_W_END), CE_MAX(A2_W_END, S4_W_END), S5_W_END),
                           CE_LINE) };
/* ==== LPDDR ===================================================================
* Offsets inside one node. South node j: [ model input | summary patches ]; stage 1
* reads the input, writes its span patches over it and the summary patches next to
* it, and stage 4 reads both back for the residual sum. North node j: the
* estimate of the head lane in column j, tile after tile. */
#define LP_IN_OFF       0u                                       // start of the model input in a south node, and of the span patches written over it
#define LP_IN_B         (CE_IN_ALL * CE_BYTES / CE_LPDDR_SOUTH)  // a quarter of the model input, 16 elements (F.1); then the span patches of 4 lines, written over it (F.6, read by D.6)
#define LP_WIN_OFF      CE_RUP(LP_IN_OFF + LP_IN_B, 64u)         // start of the summary patches, right after the input, on a 64 B line
#define LP_WIN_B        (CE_WIN_ALL * CE_BYTES / CE_LPDDR_SOUTH) // 36,864 B: the summary patches of 4 lines; stage 1 writes them (F.6), stage 4 reads them (D.6)
#define LP_SOUTH_B      (LP_WIN_OFF + LP_WIN_B)                  // 457,728 B: all a south node holds
#define LP_OUT_B        (CE_OUT_ALL * CE_BYTES / CE_LPDDR_NORTH) // 2,935,296 B: the estimate of one head lane, 16 elements, in its north node
#define LPDDR_OUTPUT_OFFSET 0u                                   // start of the estimate in a north node
#define TILE_OUT_BYTES  S5_TILE_B                                // 733,824 B: one tile of the estimate, one drain; tile t lands at LPDDR_OUTPUT_OFFSET + t x TILE_OUT_BYTES (H.5)

#define CE_NODES(lanes, edge)    ((lanes) <= (edge) ? (edge) / (lanes) : 1u)   // nodes per lane
#define CE_SHARERS(lanes, edge)  ((lanes) <= (edge) ? 1u : (lanes) / (edge))   // lanes per node


/* ==== placement: a stage starts after the previous stage's lanes and idle ones */
#define N_STAGES        5
#define S1_BASE_CID     0u
#define S2_BASE_CID     (S1_BASE_CID + CE_S1_LANES + CE_S1_PAD)
#define S3_BASE_CID     (S2_BASE_CID + CE_ENC_LANES + CE_S2_PAD)
#define S4_BASE_CID     (S3_BASE_CID + CE_ENC_LANES + CE_S3_PAD)
#define S5_BASE_CID     (S4_BASE_CID + CE_S4_LANES + CE_S4_PAD)
#define N_CLUSTERS      (S5_BASE_CID + CE_S5_LANES)               // cids the pipeline occupies
#define S1_NODES        CE_NODES(CE_S1_LANES, CE_LPDDR_SOUTH)     // 2 nodes a lane
#define S1_SHARERS      CE_SHARERS(CE_S1_LANES, CE_LPDDR_SOUTH)
#define S4_NODES        CE_NODES(CE_S4_LANES, CE_LPDDR_SOUTH)
#define S4_SHARERS      CE_SHARERS(CE_S4_LANES, CE_LPDDR_SOUTH)   // 2 lanes a node
#define S5_NODES        CE_NODES(CE_S5_LANES, CE_LPDDR_NORTH)
#define S5_SHARERS      CE_SHARERS(CE_S5_LANES, CE_LPDDR_NORTH)

/* ==== checks =================================================================
 * A knob value that cannot work stops the build */

/* the model */
_Static_assert(CE_UPA_V % CE_TILE_V == 0u && CE_UPA_H % CE_TILE_H == 0u, "tiles must cover the array");
_Static_assert(CE_D_MODEL % CE_HEADS == 0u && CE_D_HEAD % 2u == 0u, "d_model must split into heads of even width");
_Static_assert(CE_CROP < CE_SPAN_PRBS, "the last span holds at least one PRB");

/* every stage splits the lines (stage 5: the elements) */
_Static_assert(CE_NL % CE_S1_LANES == 0u && S1_LINES % CE_T_V == 0u,
               "stage 1: a lane holds whole (p, t_h) columns of tiles");
_Static_assert(CE_NL % CE_ENC_LANES == 0u, "encoder lanes must divide the lines");
_Static_assert(CE_NL % CE_S4_LANES == 0u, "stage 4 lanes must divide the lines");
_Static_assert(CE_N_EL % CE_S5_LANES == 0u, "stage 5 lanes must divide the elements");
/* one lane count divides the other, so every share is whole */
_Static_assert(CE_S1_LANES % CE_ENC_LANES == 0u || CE_ENC_LANES % CE_S1_LANES == 0u, "1 -> 2: lane counts");
_Static_assert(CE_ENC_LANES % CE_S4_LANES == 0u || CE_S4_LANES % CE_ENC_LANES == 0u, "3 -> 4: lane counts");
_Static_assert(CE_S4_LANES % CE_S5_LANES == 0u || CE_S5_LANES % CE_S4_LANES == 0u, "4 -> 5: lane counts");

/* the global pass: each (line, head) pair gets whole TEs */
_Static_assert(NUM_REDMULE_TILES % E_PAIRS == 0u, "the 16 TEs must split evenly over the (line, head) pairs");
_Static_assert((E_PAIRS * A1_QR) % NUM_REDMULE_TILES == 0u && (E_PAIRS * A2_QR) % NUM_REDMULE_TILES == 0u,
               "global pass: M must be a multiple of the 16 TEs");
_Static_assert(A1_QR >= CE_LG && A2_QR >= CE_N_SPAN && E_KP >= CE_LG, "global pass: padded sizes too small");

/* kernel shapes */
_Static_assert(E_RS_ROWS % 2u == 0u && E_ALL_ROWS % 2u == 0u && S4_ROWS % 2u == 0u, "layernorm: M must be even");
_Static_assert(CE_D_MODEL % 4u == 0u && E_KP % 4u == 0u, "layernorm, softmax: row length a multiple of 4");
_Static_assert((E_PAIRS * A1_QR) % 2u == 0u && (E_PAIRS * A2_QR) % 2u == 0u, "softmax: M must be even");
_Static_assert(S1_TT_N % 8u == 0u && E_ALL_N % 8u == 0u && E_RS_N % 8u == 0u && S4_Z_N % 8u == 0u
               && S4_BASE_N % 8u == 0u && (S4_ROWS * CE_FFN) % 8u == 0u && S5_HX_N % 8u == 0u
               && (CE_HEADS * A1_QR * E_KP) % 8u == 0u && (CE_HEADS * A2_QR * E_KP) % 8u == 0u,
               "axpy and gelu: length a multiple of 8");
_Static_assert(S1_TOK_M < 65536u && S5_XR_M < 65536u, "RedMulE: M, N, P below 65,536");
_Static_assert(CE_D_QKV <= CE_GRP_COPY, "1.4: a copy of the Ll row must stay in the banks of its group");

/* the landing zones and what stays put */
_Static_assert(CE_D_MODEL % CE_LINE == 0u, "a token row is not whole 64 B lines");
_Static_assert(E_RS_N % CE_LINE == 0u && S4_Z_N % CE_LINE == 0u
               && S5_X_N % CE_LINE == 0u, "inbound shares are not whole DMA lines"); //TODO

/* stage 5 */
_Static_assert(S5_XR_M % CE_S5_TILES == 0u, "stage 5: the tiles must split the PRB rows evenly");

/* L1: X_BUF (with its twin, rounded up to its 8 KiB alignment) and W_BUF fit */
#define CE_L1_BUDGET_B  ((uint32_t)NUM_BANKS * L1_BANK_SIZE - (uint32_t)NUM_CORES * STACK_SIZE \
                         - (2u * NUM_BANKS * 4u + 2u * NUM_BANKS + 64u))
_Static_assert(CE_RUP((CE_X_ELEMS + CE_X_PAR_ELEMS) * CE_BYTES, NUM_BANKS * 4u)
                   + CE_W_ELEMS * CE_BYTES <= CE_L1_BUDGET_B,
               "X_BUF + W_BUF do not fit one cluster's L1 (with CE_S5_DBUF=1, try more CE_S5_TILES)");

/* LPDDR placement */
_Static_assert(CE_LPDDR_SOUTH <= ARCH_NUM_CLUSTER_X && CE_LPDDR_NORTH <= ARCH_NUM_CLUSTER_X,
               "one LPDDR node per column at most");
// _Static_assert(CE_S1_LANES % CE_LPDDR_SOUTH == 0u || CE_LPDDR_SOUTH % CE_S1_LANES == 0u,
//                "stage 1 lanes and south nodes must divide");
_Static_assert(CE_LPDDR_SOUTH % CE_S1_LANES == 0u, "stage 1: every lane has its own south nodes");
_Static_assert(S1_XR_ROWS * CE_C == S1_IN_N, "stage 1: the span patches must fill the input they are written over (F.6)");

_Static_assert(CE_S4_LANES % CE_LPDDR_SOUTH == 0u || CE_LPDDR_SOUTH % CE_S4_LANES == 0u,
               "stage 4 lanes and south nodes must divide");
_Static_assert(CE_S5_LANES % CE_LPDDR_NORTH == 0u || CE_LPDDR_NORTH % CE_S5_LANES == 0u,
               "stage 5 lanes and north nodes must divide");
_Static_assert(S1_BASE_CID % ARCH_NUM_CLUSTER_X == 0u && S4_BASE_CID % ARCH_NUM_CLUSTER_X == 0u
               && S5_BASE_CID % ARCH_NUM_CLUSTER_X == 0u,
               "a stage that touches LPDDR must start at a column boundary (adjust the CE_Sx_PAD knobs)");

_Static_assert(S1_BASE_CID / ARCH_NUM_CLUSTER_X == (S1_BASE_CID + CE_S1_LANES - 1u) / ARCH_NUM_CLUSTER_X
               && S5_BASE_CID / ARCH_NUM_CLUSTER_X == (S5_BASE_CID + CE_S5_LANES - 1u) / ARCH_NUM_CLUSTER_X,
               "stages 1 and 5 own their nodes one per lane and must lie in one grid row");
_Static_assert(S5_BASE_CID / ARCH_NUM_CLUSTER_X == ARCH_NUM_CLUSTER_Y - 1u,
               "the self-draining head is not in the north row, so its drain is not one hop");
_Static_assert(LP_SOUTH_B <= ARCH_HBM_NODE_ADDR_SPACE && LP_OUT_B <= ARCH_HBM_NODE_ADDR_SPACE,
               "per-node LPDDR data does not fit one node"); // a node's address window is 8 MiB; past it the address lands in the next node
_Static_assert(CE_S5_LANES == CE_LPDDR_NORTH && CE_S5_TILES * TILE_OUT_BYTES == LP_OUT_B,
               "the head drains each lane into the north node of its column: one lane per north node");
_Static_assert(LPDDR_OUTPUT_OFFSET + CE_S5_TILES * TILE_OUT_BYTES <= ARCH_HBM_NODE_ADDR_SPACE,
               "the head's output does not fit one LPDDR node");
_Static_assert(ARCH_NUM_CLUSTER == 32u, "CID_STAGE and CID_LANE list 32 cids, one per cluster");
_Static_assert(N_CLUSTERS <= ARCH_NUM_CLUSTER, "the pipeline needs more clusters than this arch has");



#include "baremetal/mempool_celere_lhsm_f16.h"

/* CE_REGION { ... } marks a bench region: one row of a stage, or one copy.
 *
 * CE_BENCH=1 */

#if CE_BENCH
#define CE_REGION  mempool_start_benchmark(); \
                   for (int ce_r_ = 1; ce_r_; ce_r_ = (mempool_stop_benchmark(), 0))
#else
#define CE_REGION
#endif

/* ==== placement tables ======================================================
 * Stage s owns the consecutive clusters base_s .. base_s + lanes_s - 1,
 * and cluster c is its 'c - base_s' lane.
 * A cluster outside every stage is idle: stage 0, lane 0. */


 // 1 if cluster c is a lane of the stage that owns clusters base .. base + lanes - 1, else 0
#define CID_IN(c, base, lanes)  ((c) >= (base) && (c) < (base) + (lanes))

// Stage of cluster c (1 to 5), or 0 if c is idle
#define CID_STAGE_OF(c)  (1u * CID_IN(c, S1_BASE_CID, CE_S1_LANES)                     \
                        + 2u * CID_IN(c, S2_BASE_CID, CE_ENC_LANES)                    \
                        + 3u * CID_IN(c, S3_BASE_CID, CE_ENC_LANES)                    \
                        + 4u * CID_IN(c, S4_BASE_CID, CE_S4_LANES)                     \
                        + 5u * CID_IN(c, S5_BASE_CID, CE_S5_LANES))

// Lane of cluster c in its stage (0-based), or 0 if c is idle
#define CID_LANE_OF(c)   (((c) - S1_BASE_CID) * CID_IN(c, S1_BASE_CID, CE_S1_LANES)    \
                        + ((c) - S2_BASE_CID) * CID_IN(c, S2_BASE_CID, CE_ENC_LANES)   \
                        + ((c) - S3_BASE_CID) * CID_IN(c, S3_BASE_CID, CE_ENC_LANES)   \
                        + ((c) - S4_BASE_CID) * CID_IN(c, S4_BASE_CID, CE_S4_LANES)    \
                        + ((c) - S5_BASE_CID) * CID_IN(c, S5_BASE_CID, CE_S5_LANES))

// Stage (1 to 5, 0 = idle) of clusters 0 to 31, one grid row per line
#define CID_STAGE_LIST  CID_STAGE_OF(0u),  CID_STAGE_OF(1u),  CID_STAGE_OF(2u),  CID_STAGE_OF(3u),   \
                        CID_STAGE_OF(4u),  CID_STAGE_OF(5u),  CID_STAGE_OF(6u),  CID_STAGE_OF(7u),   \
                        CID_STAGE_OF(8u),  CID_STAGE_OF(9u),  CID_STAGE_OF(10u), CID_STAGE_OF(11u),  \
                        CID_STAGE_OF(12u), CID_STAGE_OF(13u), CID_STAGE_OF(14u), CID_STAGE_OF(15u),  \
                        CID_STAGE_OF(16u), CID_STAGE_OF(17u), CID_STAGE_OF(18u), CID_STAGE_OF(19u),  \
                        CID_STAGE_OF(20u), CID_STAGE_OF(21u), CID_STAGE_OF(22u), CID_STAGE_OF(23u),  \
                        CID_STAGE_OF(24u), CID_STAGE_OF(25u), CID_STAGE_OF(26u), CID_STAGE_OF(27u),  \
                        CID_STAGE_OF(28u), CID_STAGE_OF(29u), CID_STAGE_OF(30u), CID_STAGE_OF(31u)

// Lane of cluster c in its stage, from 0; an idle cluster also gets 0
#define CID_LANE_LIST   CID_LANE_OF(0u),  CID_LANE_OF(1u),  CID_LANE_OF(2u),  CID_LANE_OF(3u),   \
                        CID_LANE_OF(4u),  CID_LANE_OF(5u),  CID_LANE_OF(6u),  CID_LANE_OF(7u),   \
                        CID_LANE_OF(8u),  CID_LANE_OF(9u),  CID_LANE_OF(10u), CID_LANE_OF(11u),  \
                        CID_LANE_OF(12u), CID_LANE_OF(13u), CID_LANE_OF(14u), CID_LANE_OF(15u),  \
                        CID_LANE_OF(16u), CID_LANE_OF(17u), CID_LANE_OF(18u), CID_LANE_OF(19u),  \
                        CID_LANE_OF(20u), CID_LANE_OF(21u), CID_LANE_OF(22u), CID_LANE_OF(23u),  \
                        CID_LANE_OF(24u), CID_LANE_OF(25u), CID_LANE_OF(26u), CID_LANE_OF(27u),  \
                        CID_LANE_OF(28u), CID_LANE_OF(29u), CID_LANE_OF(30u), CID_LANE_OF(31u)

/* Stages are numbered from 1; every per-stage table has an unused entry 0. */
enum { STAGE_1 = 1, STAGE_2, STAGE_3, STAGE_4, STAGE_5 };

// store for runtime: stage and lane of each cluster, computed by the compiler; each cluster reads its own entry at start-up
static const uint8_t CID_STAGE[ARCH_NUM_CLUSTER] = { CID_STAGE_LIST };
static const uint8_t CID_LANE [ARCH_NUM_CLUSTER] = { CID_LANE_LIST };

/* stage -> its first cid, and the number of clusters that run it */
static const uint8_t STAGE_BASE [N_STAGES + 2] = { 0, S1_BASE_CID, S2_BASE_CID, S3_BASE_CID, S4_BASE_CID, S5_BASE_CID, N_CLUSTERS };
static const uint8_t STAGE_LANES[N_STAGES + 1] = { 0, CE_S1_LANES, CE_ENC_LANES, CE_ENC_LANES, CE_S4_LANES, CE_S5_LANES };

// Output tiles per stage (0 = idle). Stage 5 writes its tiles to LPDDR itself, so main() sends nothing for it.
static const uint8_t STAGE_TILES[N_STAGES + 1] = { 0, 1, 1, 1, 1, CE_S5_TILES };

/* 1: stages 2 and 3 send the lattice they updated in place in the landing zone,
 * so the source is this item's copy (+ par * XPAR_B).
 * 0: stages 1 and 4 send from their regions above the landing zone. Only the
 * landing zone has a twin, so these regions have one address for every item. */
static const uint8_t STAGE_SRC_PAR[N_STAGES + 1] = { 0, 0, 1, 1, 0, 0 };

/* ==== hand-offs ==============================================================
 * One entry per tensor sent to the next stage, in the order of the consumer's
 * landing zone, where the tensors land back to back. */
typedef struct {
    uint32_t bytes;    // the whole tensor, all lanes
    uint32_t src_off;  // start of this lane's share in the producer's X_BUF, in bytes
} xfer_t;
static const xfer_t STAGE_XFER[10] = {
    { CE_ZR_ALL * CE_BYTES,   S1_Z * CE_BYTES },                                   // stage 1: span tokens
    { CE_ZS_ALL * CE_BYTES,   S1_ZS * CE_BYTES },                                  // stage 1: summary tokens
    { CE_ZR_ALL * CE_BYTES,   A1_Z * CE_BYTES },                                   // stage 2: span rows
    { CE_ZS_ALL * CE_BYTES,   (A1_Z + E_ROW_S * CE_D_MODEL) * CE_BYTES },          // stage 2: summary rows
    { CE_ZS_ALL * CE_BYTES,   (A1_Z + E_ROW_LG * CE_D_MODEL) * CE_BYTES },         // stage 2: global probes
    { CE_ZS_ALL * CE_BYTES,   (A1_Z + E_ROW_LL * CE_D_MODEL) * CE_BYTES },         // stage 2: local probes
    { CE_ZR_ALL * CE_BYTES,   A2_Z * CE_BYTES },                                   // stage 3: span rows
    { CE_ZS_ALL * CE_BYTES,   (A2_Z + E_ROW_S * CE_D_MODEL) * CE_BYTES },          // stage 3: summary rows
    { CE_HPRB_ALL * CE_BYTES, S4_BLK * CE_BYTES },                                 // stage 4: head input, PRB rows
    { CE_HSUM_ALL * CE_BYTES, S4_BLK_S * CE_BYTES },                               // stage 4: head input, summary rows
};

/* stage s sends the entries STAGE_XFER_FIRST[s] to STAGE_XFER_FIRST[s + 1] - 1 */
static const uint8_t STAGE_XFER_FIRST[N_STAGES + 2] = { 0, 0, 2, 6, 8, 10, 10 };

/* The landing zones of stages 2 to 5 hold, family after family, what the
 * producers send. */
_Static_assert(E_ROW_S * CE_D_MODEL * CE_BYTES == CE_ZR_ALL * CE_BYTES / CE_ENC_LANES
               && E_ROW_LG * CE_D_MODEL * CE_BYTES == (CE_ZR_ALL + CE_ZS_ALL) * CE_BYTES / CE_ENC_LANES
               && E_ROW_LL * CE_D_MODEL * CE_BYTES == (CE_ZR_ALL + 2u * CE_ZS_ALL) * CE_BYTES / CE_ENC_LANES,
               "stages 2 and 3: the lattice rows are not where the hand-off puts them");
_Static_assert(S4_R_ROWS * CE_D_MODEL == CE_ZR_ALL / CE_S4_LANES && S4_Z_N == (CE_ZR_ALL + CE_ZS_ALL) / CE_S4_LANES,
               "stage 4: the lattice rows are not where the hand-off puts them");
_Static_assert(S5_XR_M * CE_RUN == CE_HPRB_ALL / CE_S5_LANES && S5_X_N == (CE_HPRB_ALL + CE_HSUM_ALL) / CE_S5_LANES,
               "stage 5: the head input is not where the hand-off puts it");
/* the smallest share the lane rules allow: one line of the lattice; the head input
 * of one tile, 4 elements */
_Static_assert((CE_ZR_ALL * CE_BYTES / CE_NL) % 64u == 0u && (CE_ZS_ALL * CE_BYTES / CE_NL) % 64u == 0u
               && (CE_HPRB_ALL * CE_BYTES / CE_NL) % 64u == 0u && (CE_HSUM_ALL * CE_BYTES / CE_NL) % 64u == 0u,
               "a share of a hand-off is not whole 64 B DMA lines");

/* ==== LPDDR transfers ========================================================
 * The copies between L1 and LPDDR, one entry per tensor (the head writes its own
 * output, in compute_stage_5). Stage 1 reads the model input from the south nodes
 * and writes the summary patches next to it; stage 4 reads both again, for the
 * decoder residual. A lane spans `nodes` nodes (stage 1: 2), or `lanes_per_node`
 * lanes share one node (stage 4: 2). The n-th copy of a lane (n < nodes):
 *
 *     node = (lane / lanes_per_node) * nodes + n
 *     off  = lpddr_off + (lane % lanes_per_node) * lane_step   (inside the node) */



/* Which side of the chip a transfer's LPDDR nodes are on. At the moment all transfers
 * in these tables use the south nodes, so NORTH is not used; the head writes to
 * the north nodes by itself, with hbm_north(). We keep NORTH for later mappings. */
#define LPDDR_EDGE_SOUTH  0u
#define LPDDR_EDGE_NORTH  1u


typedef struct {
    uint32_t lpddr_off;       // where the tensor starts inside a node, in bytes; the same in every node
    uint32_t l1_off;          // where the lane's share starts in X_BUF, in bytes; copy n lands at l1_off + n * chunk
    uint32_t lane_step;       // distance between the starts of lanes that share a node, in bytes; 0 if a lane has the node to itself
    uint32_t chunk;           // bytes in one run; with reps = 1, the whole copy from one node
    uint32_t lpddr_stride;    // step between runs in LPDDR; has no effect while reps = 1
    uint32_t l1_stride;       // step between runs in L1; has no effect while reps = 1
    uint32_t reps;            // runs in one copy; 1 today, so each copy is one plain block
    uint32_t edge;            // side of the chip: LPDDR_EDGE_SOUTH or LPDDR_EDGE_NORTH
    uint32_t nodes;           // nodes one lane reads or writes, one copy each (stage 1: 2)
    uint32_t lanes_per_node;  // lanes that share one node (stage 4: 2)
} lpddr_xfer_t;


/* One LPDDR leg/transfer: the node_b bytes of a tensor in one node, split between the
 * lanes that share the node, one descriptor each. */
#define LPDDR_LEG(lpddr_off, l1_off, node_b, edge, nodes, share)                 \
    { (lpddr_off), (l1_off), (share) > 1u ? (node_b) / (share) : 0u,             \
      (node_b) / (share), (node_b) / (share), (node_b) / (share), 1u, (edge),    \
      (nodes), (share) }

static const lpddr_xfer_t LPDDR_IN_XFER[2] = {
    // LPDDR_LEG(LP_IN_OFF,  S1_IN * CE_BYTES,  LP_IN_B,  LPDDR_EDGE_SOUTH, S1_NODES, S1_SHARERS),  // stage 1: model input (F.1) // moved to stage 1 explicitly
    LPDDR_LEG(LP_IN_OFF,  S4_XB * CE_BYTES,  LP_IN_B,  LPDDR_EDGE_SOUTH, S4_NODES, S4_SHARERS),  // stage 4: span patches, written by stage 1 over its input (D.6)
    LPDDR_LEG(LP_WIN_OFF, S4_WIN * CE_BYTES, LP_WIN_B, LPDDR_EDGE_SOUTH, S4_NODES, S4_SHARERS),  // stage 4: summary patches (D.6)
};

/* stage s reads the legs LPDDR_IN_FIRST[s] to LPDDR_IN_FIRST[s + 1] - 1, and writes
 * the legs LPDDR_OUT_FIRST[s] to LPDDR_OUT_FIRST[s + 1] - 1 */
// static const uint8_t LPDDR_IN_FIRST[N_STAGES + 2] = { 0, 0, 1, 1, 1, 3, 3 }; // stage 1 lpddr pull moved explicitly to stage1 processing
static const uint8_t LPDDR_IN_FIRST[N_STAGES + 2] = { 0, 0, 0, 0, 0, 2, 2 };

// static const lpddr_xfer_t LPDDR_OUT_XFER[1] = { //moved explicitly to stage1 processing
//     LPDDR_LEG(LP_WIN_OFF, S1_WIN * CE_BYTES, LP_WIN_B, LPDDR_EDGE_SOUTH, S1_NODES, S1_SHARERS),  // stage 1: summary patches (F.6)
// };
// static const uint8_t LPDDR_OUT_FIRST[N_STAGES + 2] = { 0, 0, 1, 1, 1, 1, 1 };

/* ==== copies =================================================================
 * Every blocking copy goes through here: the DM core starts a 2D DMA of reps runs
 * of chunk bytes and waits until it is done. With CE_BENCH=1 each copy is one
 * bench region. The head's drains are the exception: they do not wait
 * (compute_stage_5). */
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

/* push_to_next: run by the DM core of each producer cluster, for this cluster's
 * share only. lane is the cluster's position in its stage.
 * dst_base is the consumer's landing zone, the same local address in every cluster;
 * src_base is this cluster's X_BUF, + the item's parity for stages 2 and 3. */
static __attribute__((noinline)) void push_to_next(uint32_t stage, uint32_t lane,
                                   uint32_t dst_base, uint64_t src_base)
{
    uint32_t next     = stage + 1u;
    uint32_t lanes    = STAGE_LANES[stage];
    uint32_t lanes_nx = STAGE_LANES[next];
    uint32_t base_nx  = STAGE_BASE[next];
    /* The consumer's tensors lie next to each other in table order, so a running
     * sum (dst_off) reaches each one. */
    uint32_t dst_off  = 0u;

    for (uint32_t e = STAGE_XFER_FIRST[stage]; e < STAGE_XFER_FIRST[next]; e++) {
        const xfer_t *x = &STAGE_XFER[e];
        uint32_t own_src = x->bytes / lanes;      // what this cluster holds
        uint32_t own_dst = x->bytes / lanes_nx;   // what one consumer needs

        if (lanes_nx > lanes) {
            /* Fan-out: this lane's share covers r consecutive consumers. */
            uint32_t r = lanes_nx / lanes;
            for (uint32_t k = 0; k < r; k++)
                dma_2d((uint64_t)remote_cid(base_nx + lane * r + k, dst_base + dst_off),
                       src_base + (uint64_t)(x->src_off + k * own_dst), own_dst);
        } else if (lanes > lanes_nx) {
            /* Fan-in: r producers share one consumer; this lane lands in slot
             * lane % r of the consumer's tensor. */
            uint32_t r = lanes / lanes_nx;
            dma_2d((uint64_t)remote_cid(base_nx + lane / r, dst_base + dst_off + (lane % r) * own_src),
                   src_base + (uint64_t)x->src_off, own_src);
        } else {
            dma_2d((uint64_t)remote_cid(base_nx + lane, dst_base + dst_off),
                   src_base + (uint64_t)x->src_off, own_src);
        }
        dst_off += own_dst;
    }
}

/* Address of byte off in LPDDR node `node` (node j sits at column j) on one side. */
static inline uint64_t lpddr_node_addr(uint32_t edge, uint32_t node, uint32_t off)
{
    return edge == LPDDR_EDGE_NORTH ? hbm_north(node, off) : hbm_south(node, off);
}

static __attribute__((noinline)) void lpddr_pull(uint32_t stage, uint32_t lane, uint64_t dst_base)
{
    for (uint32_t e = LPDDR_IN_FIRST[stage]; e < LPDDR_IN_FIRST[stage + 1u]; e++) {
        const lpddr_xfer_t *x = &LPDDR_IN_XFER[e];
        uint32_t node = (lane / x->lanes_per_node) * x->nodes;
        uint32_t off  = x->lpddr_off + (lane % x->lanes_per_node) * x->lane_step;
        for (uint32_t n = 0; n < x->nodes; n++)
            dma_2d_strided(dst_base + (uint64_t)(x->l1_off + n * x->chunk),
                           lpddr_node_addr(x->edge, node + n, off),
                           x->chunk, x->l1_stride, x->lpddr_stride, x->reps);
    }
}

// static __attribute__((noinline)) void lpddr_push(uint32_t stage, uint32_t lane, uint64_t src_base)
// {
//     for (uint32_t e = LPDDR_OUT_FIRST[stage]; e < LPDDR_OUT_FIRST[stage + 1u]; e++) {
//         const lpddr_xfer_t *x = &LPDDR_OUT_XFER[e];
//         uint32_t node = (lane / x->lanes_per_node) * x->nodes;
//         uint32_t off  = x->lpddr_off + (lane % x->lanes_per_node) * x->lane_step;
//         for (uint32_t n = 0; n < x->nodes; n++)
//             dma_2d_strided(lpddr_node_addr(x->edge, node + n, off),
//                            src_base + (uint64_t)(x->l1_off + n * x->chunk),
//                            x->chunk, x->lpddr_stride, x->l1_stride, x->reps);
//     }
// }


/* ==== the two L1 buffers of a cluster ===================================== */
#define X_ELEMS     CE_X_ELEMS       // the largest stage map, in elements
#define W_ELEMS     CE_W_ELEMS       // the largest weight map, in elements
#define X_PAR_ELEMS CE_X_PAR_ELEMS   // the largest landing zone: the size of its twin
#define XPAR_B      (X_ELEMS * (uint32_t)sizeof(__fp16))   // bytes from a landing-zone region to its twin
// activations: the stage's map, then the twin of the landing zone, where odd items land
static __fp16 X_BUF[X_ELEMS + X_PAR_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")));
// parameters of every row, filled once, then only read
static __fp16 W_BUF[W_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1"), used));




// ==== small kernels of this app ===============================================
/* Starts G GEMMs at once, Y_g += X_g . W_g for g = 0 .. G-1, with 16 / G TEs each.
 * X, Y and W are the bases; product g uses
 *   X_g = X + g * (M/G) * N    [M/G][N], so the G blocks lie back to back in X
 *   Y_g = Y + g * (M/G) * P    [M/G][P], likewise in Y
 *   W_g = W + g * w_step       [N][P]
 * Until wait_redmule(): no barrier, no access to X, W, Y. */
static inline void redmule_grouped_start(__fp16 *X, __fp16 *W, __fp16 *Y,
                                         uint32_t M, uint32_t N, uint32_t P,
                                         uint32_t groups, uint32_t w_step)
{
    uint32_t te   = mempool_get_redmule_id();
    uint32_t n_te = mempool_get_redmule_count();
    if (te < n_te) {
        uint32_t m = M / n_te;                  // rows a TE
        uint32_t g = te / (n_te / groups);      // the product this TE works on
        unsigned int x = (unsigned int)(X + te * m * N);
        unsigned int y = (unsigned int)(Y + te * m * P);
        unsigned int w = (unsigned int)(W + g * w_step);
        hwpe_soft_clear();
        mempool_wait(10);
        redmule_cfg(x, w, y, (uint16_t)m, (uint16_t)N, (uint16_t)P, (uint16_t)0, GEMM, Float16);
        mempool_wait(10);
        hwpe_trigger_job();
    }
}


// redmule_grouped_start, then wait for the TEs and one barrier of all cores
static inline void redmule_grouped_parallel(__fp16 *X, __fp16 *W, __fp16 *Y,
                                            uint32_t M, uint32_t N, uint32_t P,
                                            uint32_t groups, uint32_t w_step)
{
    redmule_grouped_start(X, W, Y, M, N, P, groups, w_step);
    wait_redmule();
    mempool_barrier(mempool_get_core_count());
}

// sets the probe rows of the lattice to the probe embeddings e_g, e_l; all cores, no barrier inside
static inline void probe_reset(__fp16 *dst, const __fp16 *e, uint32_t cid, uint32_t nt)
{
    uint32_t *d = (uint32_t *)dst;
    const uint32_t *s = (const uint32_t *)e;
    const uint32_t row_w = CE_D_MODEL / 2u;     // words a row
    for (uint32_t i = cid; i < 2u * E_S_ROWS * row_w; i += nt)
        d[i] = s[(i / row_w) / E_S_ROWS * row_w + i % row_w];
}

/* ==== per-cluster compute: one function per stage ============================
 * Each function does this cluster's part of one item, op row by row; each row is one bench region. */

/* Stage 1, front end: patches, summary patches, embedding and
 * LPDDR transfers. */
static __attribute__((noinline)) void compute_stage_1(uint32_t par, uint32_t lane)
{
    (void)par;   // stage 1 reads from LPDDR: no twin (not used without warning)
    uint32_t cid = mempool_get_core_id();     // in-cluster core id
    uint32_t nt  = mempool_get_core_count();  // cores per cluster
    uint32_t node = lane * S1_NODES;           // the lane's first south node; it has S1_NODES, 4 lines each


    if (flex_is_dm_core())   // the model input, from each of the lane's south nodes (F.1 IN)
        for (uint32_t n = 0; n < S1_NODES; n++)
            dma_2d((uint64_t)(uintptr_t)(X_BUF + S1_IN) + n * LP_IN_B, hbm_south(node + n, LP_IN_OFF), LP_IN_B);
    flex_intra_cluster_sync();   // the other cores waits for the input


    CE_REGION {   // patching: one patch per span and 2x2 tile; x -> X [line][span][384] (F.2 PAT)
        /* x: the lane's input, elements in order (p, t_h, e_h, t_v, e_v), each [3288 subcarriers][2 pilots][re, im].
         * X: lines in (p, t_h, t_v) order; a patch is [PRB of the span][e_h][e_v][48], 48 = one PRB of one element. */
        static const perm_level_t lv[] = {
            { CE_TILE_V,          CE_ST_EV * CE_PER_EL,     CE_RUN },                           // e_v
            { CE_TILE_H,          CE_ST_EH * CE_PER_EL,     CE_TILE_V * CE_RUN },               // e_h
            { CE_SPAN_PRBS,       CE_RUN,                   CE_TILE_EL * CE_RUN },              // PRB of the span
            { CE_N_SPAN,          CE_SPAN_SC * CE_RE,       CE_C },                             // span
            { CE_T_V,             CE_ST_TV * CE_PER_EL,     CE_N_SPAN * CE_C },                 // t_v
            { S1_LINES / CE_T_V,  CE_ST_TH * CE_PER_EL,     CE_T_V * CE_N_SPAN * CE_C },        // (p, t_h)
        };
        permute_f16p(X_BUF + S1_IN, X_BUF + S1_X, CE_RUN, lv, 6u, cid, nt);
        flex_intra_cluster_sync();
    }

    CE_REGION {   // summary patch = mean of the window's spans; 32 cores per line, all lines at once (F.3 WIN)
        uint32_t per = nt / S1_LINES;   // 32 cores a line; the pool keeps 24 of them busy (384 / 16)
        uint32_t l   = cid / per;

        /* line l = cid / per, so each 32-core group pools one line; core cid % per of the group
         * takes 16 columns (from 384). Elements from X_BUF:
         *   src  X_BUF + S1_X   + l * 137 * 384   span patches of line l, [137 spans][384]
         *   dst  X_BUF + S1_WIN + l * 12  * 384   summary patches of line l, [12 windows][384]
         *   CE_N_SPAN     n_in:  137 span rows of the line
         *   CE_N_WIN      n_out: 12 windows
         *   CE_WIN_SPANS  win:   12 spans a window (also the divisor)
         *   CE_C          lane:  384 values a row; each column is averaged on its own
         *   cid % per, per   core id and core count inside the group
         * Window w averages rows 12w to 12w + 11 of src into row w of dst. */
        pool_mean_f16p(X_BUF + S1_X + l * CE_N_SPAN * CE_C, X_BUF + S1_WIN + l * CE_N_WIN * CE_C,
                    CE_N_SPAN, CE_N_WIN, CE_WIN_SPANS, CE_C, cid % per, per);
        flex_intra_cluster_sync();
    }


    /* F.6 under F.4: the TEs start the embedding without the DM core, which joins at
     * its closing barrier, so these copies cost no time. */
    if (flex_is_dm_core())   // span patches overide the input they came from, summary patches after it, into each south node (F.6 OUT)
        for (uint32_t n = 0; n < S1_NODES; n++) {
            dma_2d(hbm_south(node + n, LP_IN_OFF),  (uint64_t)(uintptr_t)(X_BUF + S1_X) + n * LP_IN_B,    LP_IN_B);
            dma_2d(hbm_south(node + n, LP_WIN_OFF), (uint64_t)(uintptr_t)(X_BUF + S1_WIN) + n * LP_WIN_B, LP_WIN_B);
        }


    CE_REGION {   // embedding: tokens Z = [X; S] . W_e, spans and summaries in one launch on the 16 TEs (F.4 EMB)
        /* TE t takes rows t * M/16 to (t + 1) * M/16 - 1 of X and Y and all of W; the cores
         * meet in a barrier inside the kernel. Offsets in fp16 elements from X_BUF, W_BUF:
         *   X_BUF + S1_X      X: [X; S], 1,096 span patches then 96 summary patches, rows of 384
         *   W_BUF + S1_W_EMB  W: W_e [384][256]
         *   X_BUF + S1_Z      Y: tokens [1,200][256], span tokens, then summary tokens from S1_ZS
         *   S1_TOK_M          M: 1,192 rows padded to 1,200, 75 per TE; the 8 pad rows are never read
         *   CE_C              N: 384, a patch
         *   CE_D_MODEL        P: 256, a token
         *   GEMM, false, 0    the operation; no column rotation between TEs */
        redmule_synch_parallel(X_BUF + S1_X, W_BUF + S1_W_EMB, X_BUF + S1_Z, S1_TOK_M, CE_C, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }

    CE_REGION {   // token type: + (b_e + t_R) on spans, + (b_e + t_S) on summaries, Y += X; X stands in for the broadcast bias rows (F.5 TT)
        axpy_f16vecp_local_unrolled4(0x3C00u, X_BUF + S1_X, X_BUF + S1_Z, S1_TT_N);
    }
}



/* Stage 2, attention stage 1 (1.1 to 1.13). The global pass runs on the TEs while
 * the local pass runs on the cores:
 *
 *     cores:  1.6 Q_g, K_g^T | 1.4 LHSM, 1.7 V_g | wait | 1.9a ...
 *     TEs:                   | 1.8a S_g = Q_g . K_g^T  |
 *
 * 1.6 writes Q_g, K_g^T out of place, so 1.4 still reads the unrotated q, k. Under
 * the TE job the cores touch none of its operands and cross no barrier until the
 * wait. The hand-off (1.14) runs in main(). */
static __attribute__((noinline)) void compute_stage_2(uint32_t par)
{
    uint32_t cid = mempool_get_core_id();      // in-cluster core id
    uint32_t nt  = mempool_get_core_count();   // cores per cluster
    __fp16 *z = X_BUF + par * X_ELEMS + A1_Z;  // this item's lattice
    /* 1.4 reads spans and summaries from the projection; the local probe is one
     * constant row (1.cp), so its steps are 0, and each group reads its own copy */
    static const lhsm_args_t lhsm = {
        .span = X_BUF + A1_QKV,                       .span_line = CE_N_SPAN * CE_D_QKV, .span_step = CE_D_QKV,
        .sum  = X_BUF + A1_QKV + E_ROW_S * CE_D_QKV,  .sum_line  = CE_N_WIN * CE_D_QKV,  .sum_step  = CE_D_QKV,
        .lpr  = W_BUF + A1_W_LLC,                     .lpr_line  = 0u,                   .lpr_step  = 0u,       .lpr_copy = CE_GRP_COPY,
        .bias = W_BUF + A1_W_LB,
        .a_span = X_BUF + A1_ASPAN, .a_hub = X_BUF + A1_AHUB,
        .n_lines = E_LINES, .n_heads = CE_HEADS, .n_hub = A1_NHUB,
        .n_span = CE_N_SPAN, .win = CE_WIN_SPANS, .d_model = CE_D_MODEL, .d_head = CE_D_HEAD,
    };
    CE_REGION {   // layernorm of the span and summary rows; Z -> H; the probes' LN is a constant (1.1 LN)
        layernorm_parallel_2x4_f16vec(z, X_BUF + A1_H, E_RS_ROWS, CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // projection H . W_qkv -> q|k|v [304][768] (1.2 QKV)
        redmule_synch_parallel(X_BUF + A1_H, W_BUF + A1_W_QKV, X_BUF + A1_QKV, A1_H_M, CE_D_MODEL, CE_D_QKV, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // q, k, v per head: a view, no code (1.3 SPLIT)
    }
    /* TODO: one kernel to replace the next two regions (the copy and the RoPE wait).
     *  in:   q|k|v [304][768] at X_BUF + A1_QKV; row l * 137 + i is span i of line l;
     *        head h: q in columns h*128 .. h*128+127, k in 256 + h*128 .. 256 + h*128+127
     *        cos|sin table [head][137][128] at W_BUF + A1_W_ROPE: per row 64 cos, then 64 sin
     *  do:   rotate each q and k vector of a span by its row of the table, pairs (x[j], x[j+64]):
     *        lo' = lo*cos - hi*sin, hi' = hi*cos + lo*sin, each product and sum rounded to fp16
     *  out:  Q_g   [pair][152][128] at X_BUF + A1_QG: row i = rotated q of span i (i < 137)
     *        K_g^T [pair][128][152] at X_BUF + A1_KT: column i = rotated k of span i (i < 137)
     *        pair = line * 2 + head; rows / columns 137-151 left as they are
     *  keep: q|k|v unchanged (1.4 reads the unrotated q, k); no barrier inside
     *  rope_gather_f16p (mempool_rope_f16.h) is written to this spec; it still needs its GVSoC run.
     * TODO (numerics only): Q_g and V_g rows 137-148 and K_g^T columns 137-148 (global-probe
     * q, k, v, the same every item), and rows / columns 149-151 (padding, must be 0), are
     * never written; write them once at start-up. */
    CE_REGION {   // span q -> Q_g [pair][152][128], span k -> K_g [pair][152][128], not transposed yet (1.6 ROPE)
        static const perm_level_t lq[] = {
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },                        // span
            { CE_HEADS,  CE_D_HEAD,             A1_QR * CE_D_HEAD },                // head
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * A1_QR * CE_D_HEAD },     // line
        };
        static const perm_level_t lk[] = {
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },                        // span
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },                 // head
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * E_KP * CE_D_HEAD },      // line
        };
        permute_f16p(X_BUF + A1_QKV, X_BUF + A1_QG, CE_D_HEAD, lq, 3u, cid, nt);
        permute_f16p(X_BUF + A1_QKV + CE_D_MODEL, X_BUF + A1_KT, CE_D_HEAD, lk, 3u, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // rotate q, k by position in Q_g, K_g^T; a timed wait, no RoPE kernel yet (1.6 ROPE)
        mempool_wait(A1_ROPE_WAIT);
        flex_intra_cluster_sync();   // Q_g, K_g^T complete before the TEs read them
    }
    CE_REGION {   // start S_g = Q_g . K_g^T, one product per pair on 4 TEs -> S_g [pair][152][152]; no wait, no barrier (1.8a GQK)
        redmule_grouped_start(X_BUF + A1_QG, X_BUF + A1_KT, X_BUF + A1_SG, E_PAIRS * A1_QR, CE_D_HEAD, E_KP,
                              E_PAIRS, CE_D_HEAD * E_KP);
    }
    CE_REGION {   // under the TEs: local logits on the unrotated q, k, + bias, softmax -> A_span, A_hub; no barrier (1.4 LHSM)
        celere_lhsm_f16(&lhsm, cid, nt);
    }
    CE_REGION {   // under the TEs: span v -> V_g [pair][152][128]; no barrier (1.7 VG)
        static const perm_level_t lv[] = {
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },                        // span
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },                 // head
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * E_KP * CE_D_HEAD },      // line
        };
        permute_f16p(X_BUF + A1_QKV + 2u * CE_D_MODEL, X_BUF + A1_VG, CE_D_HEAD, lv, 3u, cid, nt);
    }
    CE_REGION {   // the hosts wait for their TEs, then all cores meet (1.8a GQK)
        wait_redmule();
        flex_intra_cluster_sync();   // S_g, A_span, A_hub, V_g complete
    }
    CE_REGION {   // + global bias, the same for every line; pad keys get -inf (1.9a GSM, 1.cg)
        for (uint32_t l = 0; l < E_LINES; l++)
            axpy_f16vecp_local_unrolled4(0x3C00u, W_BUF + A1_W_GB, X_BUF + A1_SG + l * CE_HEADS * A1_QR * E_KP,
                                         CE_HEADS * A1_QR * E_KP);
    }
    CE_REGION {   // softmax over the 152 keys of each row, S_g -> A_g (1.9a GSM)
        softmax_parallel_2x4_f16vec(X_BUF + A1_SG, X_BUF + A1_AG, E_PAIRS * A1_QR, E_KP, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // U_g = A_g . V_g, one product per pair on 4 TEs -> U_g [pair][152][128] (1.10a GAV)
        redmule_grouped_parallel(X_BUF + A1_AG, X_BUF + A1_VG, X_BUF + A1_UG, E_PAIRS * A1_QR, E_KP, CE_D_HEAD,
                                 E_PAIRS, E_KP * CE_D_HEAD);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // local mix and head merge -> U [352][256]; a timed wait, no kernel yet (1.11 LAV)
        mempool_wait(A1_LAV_WAIT);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // output projection U . W_o, all 346 rows (1.12 WO)
        redmule_synch_parallel(X_BUF + A1_U, W_BUF + A1_W_WO, X_BUF + A1_WOUT, A1_WO_M, CE_D_MODEL, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // probe rows reset to e_g, e_l, then the residual Z += WO out (1.13 RES)
        probe_reset(z + E_ROW_LG * CE_D_MODEL, W_BUF + A1_W_PE, cid, nt);
        flex_intra_cluster_sync();
        axpy_f16vecp_local_unrolled4(0x3C00u, X_BUF + A1_WOUT, z, E_ALL_N);
    }
}





/* Stage 3, attention stage 2 (rows 2.1 to 2.13): the rows of stage 2 with the
 * second weight set, in the same order (2.4 and 2.7 under 2.8a). The probes arrive
 * with the lattice now; their queries are dead, the global probes are keys and
 * values, the local probes keys. */
static __attribute__((noinline)) void compute_stage_3(uint32_t par)
{
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    __fp16 *z = X_BUF + par * X_ELEMS + A2_Z;
    static const lhsm_args_t lhsm = {
        .span = X_BUF + A2_QKV,                        .span_line = CE_N_SPAN * CE_D_QKV, .span_step = CE_D_QKV,
        .sum  = X_BUF + A2_QKV + E_ROW_S * CE_D_QKV,   .sum_line  = CE_N_WIN * CE_D_QKV,  .sum_step  = CE_D_QKV,
        .lpr  = X_BUF + A2_QKV + E_ROW_LL * CE_D_QKV,  .lpr_line  = CE_N_WIN * CE_D_QKV,  .lpr_step  = CE_D_QKV,  .lpr_copy = 0u,
        .bias = W_BUF + A2_W_LB,
        .a_span = X_BUF + A2_ASPAN, .a_hub = X_BUF + A2_AHUB,
        .n_lines = E_LINES, .n_heads = CE_HEADS, .n_hub = A2_NHUB,
        .n_span = CE_N_SPAN, .win = CE_WIN_SPANS, .d_model = CE_D_MODEL, .d_head = CE_D_HEAD,
    };
    CE_REGION {   // layernorm over d, all 346 rows; X: the lattice, Y: H (2.1 LN)
        layernorm_parallel_2x4_f16vec(z, X_BUF + A2_H, E_ALL_ROWS, CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // projection H . W_qkv, all rows in one launch; the probe q columns are dead but cost no extra 32-row block (2.2 QKV)
        redmule_synch_parallel(X_BUF + A2_H, W_BUF + A2_W_QKV, X_BUF + A2_QKV, A2_H_M, CE_D_MODEL, CE_D_QKV, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // q, k, v per head: a view, no code (2.3 SPLIT)
    }
    CE_REGION {   // gather span q into Q_g, span and global-probe k into K_g^T, out of place (2.6 ROPE)
        static const perm_level_t lq[] = {
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },
            { CE_HEADS,  CE_D_HEAD,             A2_QR * CE_D_HEAD },
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * A2_QR * CE_D_HEAD },
        };
        static const perm_level_t lk[] = {       // the span rows of a pair
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * E_KP * CE_D_HEAD },
        };
        static const perm_level_t lg[] = {       // its 12 global-probe rows, after them
            { CE_N_WIN,  CE_D_QKV,              CE_D_HEAD },
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },
            { E_LINES,   CE_N_WIN * CE_D_QKV,   CE_HEADS * E_KP * CE_D_HEAD },
        };
        permute_f16p(X_BUF + A2_QKV, X_BUF + A2_QG, CE_D_HEAD, lq, 3u, cid, nt);
        permute_f16p(X_BUF + A2_QKV + CE_D_MODEL, X_BUF + A2_KT, CE_D_HEAD, lk, 3u, cid, nt);
        permute_f16p(X_BUF + A2_QKV + E_ROW_LG * CE_D_QKV + CE_D_MODEL, X_BUF + A2_KT + CE_N_SPAN * CE_D_HEAD,
                     CE_D_HEAD, lg, 3u, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // rotate span q, span and global-probe k; a timed wait, as there is no RoPE kernel yet (2.6 ROPE)
        mempool_wait(A2_ROPE_WAIT);
        flex_intra_cluster_sync();   // Q_g, K_g^T complete before the TEs read them
    }
    CE_REGION {   // start the global logits Q_g . K_g^T, span queries, one product per (line, head) on 4 TEs; no wait, no barrier (2.8a GQK)
        redmule_grouped_start(X_BUF + A2_QG, X_BUF + A2_KT, X_BUF + A2_SG, E_PAIRS * A2_QR, CE_D_HEAD, E_KP,
                              E_PAIRS, CE_D_HEAD * E_KP);
    }
    CE_REGION {   // local logits of the live pairs, + bias, softmax, while the TEs run; the summary is the only hub query; no barrier (2.4 LHSM)
        celere_lhsm_f16(&lhsm, cid, nt);
    }
    CE_REGION {   // gather v of spans and global probes into V_g, while the TEs run; no barrier (2.7 VG)
        static const perm_level_t lv[] = {
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * E_KP * CE_D_HEAD },
        };
        static const perm_level_t lg[] = {
            { CE_N_WIN,  CE_D_QKV,              CE_D_HEAD },
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },
            { E_LINES,   CE_N_WIN * CE_D_QKV,   CE_HEADS * E_KP * CE_D_HEAD },
        };
        permute_f16p(X_BUF + A2_QKV + 2u * CE_D_MODEL, X_BUF + A2_VG, CE_D_HEAD, lv, 3u, cid, nt);
        permute_f16p(X_BUF + A2_QKV + E_ROW_LG * CE_D_QKV + 2u * CE_D_MODEL, X_BUF + A2_VG + CE_N_SPAN * CE_D_HEAD,
                     CE_D_HEAD, lg, 3u, cid, nt);
    }
    CE_REGION {   // end of the global logits: the hosts wait for their TEs, then all cores meet (2.8a GQK)
        wait_redmule();
        flex_intra_cluster_sync();   // S_g, A_span, A_hub and V_g complete
    }
    CE_REGION {   // + global bias, Y += X (2.9a GSM, bias 2.cg)
        for (uint32_t l = 0; l < E_LINES; l++)
            axpy_f16vecp_local_unrolled4(0x3C00u, W_BUF + A2_W_GB, X_BUF + A2_SG + l * CE_HEADS * A2_QR * E_KP,
                                         CE_HEADS * A2_QR * E_KP);
    }
    CE_REGION {   // softmax over the 152 key columns (2.9a GSM)
        softmax_parallel_2x4_f16vec(X_BUF + A2_SG, X_BUF + A2_AG, E_PAIRS * A2_QR, E_KP, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // global mix A_g . V_g (2.10a GAV)
        redmule_grouped_parallel(X_BUF + A2_AG, X_BUF + A2_VG, X_BUF + A2_UG, E_PAIRS * A2_QR, E_KP, CE_D_HEAD,
                                 E_PAIRS, E_KP * CE_D_HEAD);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // local mix and head merge, spans and summaries; a timed wait, as there is no kernel yet (2.11 LAV)
        mempool_wait(A2_LAV_WAIT);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // output projection U . W_o, spans and summaries; the probe rows are dead (2.12 WO)
        redmule_synch_parallel(X_BUF + A2_U, W_BUF + A2_W_WO, X_BUF + A2_WOUT, A2_WO_M, CE_D_MODEL, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // residual Z += WO out, spans and summaries; Y: the lattice (2.13 RES)
        axpy_f16vecp_local_unrolled4(0x3C00u, X_BUF + A2_WOUT, z, E_RS_N);
    }
}

/* Stage 4, output path: FFN, patch decoder, decoder residual, per-PRB vectors
 * (rows D.1 to D.9). D.6 is its LPDDR read, D.10 its hand-off. */
static __attribute__((noinline)) void compute_stage_4(uint32_t par)
{
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    __fp16 *z = X_BUF + par * X_ELEMS + S4_Z;
    CE_REGION {   // layernorm over d; X: the lattice [span rows | summary rows], Y: H (D.1 LN)
        layernorm_parallel_2x4_f16vec(z, X_BUF + S4_H, S4_ROWS, CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // FFN up, H . W_1 (D.2 FF1)
        redmule_synch_parallel(X_BUF + S4_H, W_BUF + S4_W_FF1, X_BUF + S4_F1, S4_M, CE_D_MODEL, CE_FFN, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // GELU phi(. + b_1), in place (D.3 GELU)
        gelu_f16(X_BUF + S4_F1, S4_ROWS * CE_FFN, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // FFN down, . W_2 (D.4 FF2)
        redmule_synch_parallel(X_BUF + S4_F1, W_BUF + S4_W_FF2, X_BUF + S4_F2, S4_M, CE_FFN, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // residual FFN out += Z; X: the lattice (D.5 RES)
        axpy_f16vecp_local_unrolled4(0x3C00u, z, X_BUF + S4_F2, S4_Z_N);
    }
    CE_REGION {   // the decoder residual [X | S] from LPDDR, one tensor: a view, no code (D.6 BASE)
    }
    CE_REGION {   // patch decoder . W_dec, spans and summaries (D.7 DEC)
        redmule_synch_parallel(X_BUF + S4_F2, W_BUF + S4_W_DEC, X_BUF + S4_D, S4_M, CE_D_MODEL, CE_C, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // decoder residual + [X | S], Y += X; X: BASE (D.8 RES2)
        axpy_f16vecp_local_unrolled4(0x3C00u, X_BUF + S4_XB, X_BUF + S4_D, S4_BASE_N);
    }
    CE_REGION {   // per-PRB vectors of each element, 274 PRBs cropped to 273; Y: [element][PRB][48], [element][24][48] (D.9 BLK)
        /* a row is [PRB of the span][e_h][e_v][48]; the lane's elements go tile by tile,
         * element a' = (line, e_h, e_v) of the lane */
        static const perm_level_t lb[] = {       // span rows, whole spans
            { CE_TILE_V,          CE_RUN,                       CE_N_PRB * CE_RUN },                // e_v
            { CE_TILE_H,          CE_TILE_V * CE_RUN,           CE_TILE_V * CE_N_PRB * CE_RUN },    // e_h
            { CE_SPAN_PRBS,       CE_TILE_EL * CE_RUN,          CE_RUN },                           // PRB of the span
            { CE_N_SPAN - (CE_CROP ? 1u : 0u), CE_C,            CE_SPAN_PRBS * CE_RUN },            // span
            { S4_LINES,           CE_N_SPAN * CE_C,             CE_TILE_EL * CE_N_PRB * CE_RUN },   // line
        };
        static const perm_level_t ls[] = {       // summary rows: window row r = w * sp + position in the span
            { CE_TILE_V,          CE_RUN,                       CE_NSUM_HR * CE_RUN },
            { CE_TILE_H,          CE_TILE_V * CE_RUN,           CE_TILE_V * CE_NSUM_HR * CE_RUN },
            { CE_SPAN_PRBS,       CE_TILE_EL * CE_RUN,          CE_RUN },
            { CE_N_WIN,           CE_C,                         CE_SPAN_PRBS * CE_RUN },
            { S4_LINES,           CE_N_WIN * CE_C,              CE_TILE_EL * CE_NSUM_HR * CE_RUN },
        };
        permute_f16p(X_BUF + S4_D, X_BUF + S4_BLK, CE_RUN, lb, 5u, cid, nt);
#if CE_CROP
        static const perm_level_t lt[] = {       // the last span: its PRBs that are not cropped
            { CE_TILE_V,          CE_RUN,                       CE_N_PRB * CE_RUN },
            { CE_TILE_H,          CE_TILE_V * CE_RUN,           CE_TILE_V * CE_N_PRB * CE_RUN },
            { CE_SPAN_PRBS - CE_CROP, CE_TILE_EL * CE_RUN,      CE_RUN },
            { S4_LINES,           CE_N_SPAN * CE_C,             CE_TILE_EL * CE_N_PRB * CE_RUN },
        };
        permute_f16p(X_BUF + S4_D + (CE_N_SPAN - 1u) * CE_C, X_BUF + S4_BLK + (CE_N_SPAN - 1u) * CE_SPAN_PRBS * CE_RUN,
                     CE_RUN, lt, 4u, cid, nt);
#endif
        permute_f16p(X_BUF + S4_D + S4_R_ROWS * CE_C, X_BUF + S4_BLK_S, CE_RUN, ls, 5u, cid, nt);
        flex_intra_cluster_sync();
    }
}

/* Stage 5, Doppler head (rows H.1 to H.5).
 *
 * H.1 to H.3 write h only. The output layer H.4 runs in CE_S5_TILES tiles of
 * S5_Y_ROWS rows (4 elements each); tile t of item k goes to OUT[b], b the parity
 * of k * tiles + t (it runs on across items), and the DM core drains it to the
 * north node (H.5) without waiting.
 *
 * The rule: a GEMM never writes a buffer whose drain still runs. With two buffers,
 * GEMM t + 1 writes the buffer that drain t - 1 reads. So after GEMM t the DM core
 * first waits for all earlier drains (flex_dma_async_wait_all waits until the DMA
 * is idle, and the head's DMA has no other jobs), then starts drain t, and only
 * then the barrier lets GEMM t + 1 start. Waiting
 * after the start would wait for drain t too. Drain t runs under GEMM t + 1, and
 * the item's last drain under the next item's H.1 to H.3, covered by the wait
 * after its first GEMM. With one buffer (CE_S5_DBUF=0) the wait comes before each
 * GEMM. A late drain only makes the wait longer, never the result wrong. main()
 * waits once more after the last item, so the timer sees the last drain. */
static __attribute__((noinline)) void compute_stage_5(uint32_t par, uint32_t item)
{
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    uint32_t hbm_node = get_pos(flex_get_cluster_id()).x;   // the north node of this column
    __fp16 *x = X_BUF + par * X_ELEMS + S5_X;              // [PRB rows | summary rows][48]
    CE_REGION {   // hidden layer, PRB rows . W_x; Y: h, PRB rows (H.1 HXS)
        redmule_synch_parallel(x, W_BUF + S5_W_HX, X_BUF + S5_H, S5_XR_M, CE_RUN, CE_C_HID, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // hidden layer, summary rows . W_s; Y: h, summary rows (H.1 HXS)
        redmule_synch_parallel(x + S5_XR_M * CE_RUN, W_BUF + S5_W_HS, X_BUF + S5_H + S5_HX_N, S5_XS_M, CE_RUN, CE_C_HID,
                               GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // + window term and b_x; a stand-in, h += h, as the broadcast add has no kernel (H.2 HP)
        axpy_f16vecp_local_unrolled4(0x3C00u, X_BUF + S5_H, X_BUF + S5_H, S5_HX_N);
    }
    CE_REGION {   // GELU phi, in place (H.3 GELU)
        gelu_f16(X_BUF + S5_H, S5_HX_N, cid, nt);
        flex_intra_cluster_sync();
    }
    for (uint32_t t = 0; t < CE_S5_TILES; t++) {
        __fp16 *out = X_BUF + S5_OUT + (CE_S5_DBUF ? ((item * CE_S5_TILES + t) & 1u) : 0u) * S5_OUT_N;
#if !CE_S5_DBUF
        if (flex_is_dm_core()) {
            CE_REGION {   // wait until the drain that reads the one buffer is done (H.5 OUT)
                flex_dma_async_wait_all();
            }
        }
        flex_intra_cluster_sync();
#endif
        CE_REGION {   // output layer, one tile of h . W_h; Y: OUT (H.4 HO)
            redmule_synch_parallel(X_BUF + S5_H + t * S5_TILE_X, W_BUF + S5_W_HO, out, S5_Y_M, CE_C_HID, CE_C_OUT,
                                   GEMM, false, 0);
            flex_intra_cluster_sync();
        }
        if (flex_is_dm_core()) {
            CE_REGION {   // start this tile's drain to the north node; with two buffers, first wait for the drains before it (H.5 OUT)
#if CE_S5_DBUF
                flex_dma_async_wait_all();
#endif
                flex_dma_async_2d(hbm_north(hbm_node, LPDDR_OUTPUT_OFFSET + t * TILE_OUT_BYTES),
                                  (uint64_t)(uintptr_t)out, TILE_OUT_BYTES, TILE_OUT_BYTES, TILE_OUT_BYTES, 1u);
            }
        }
#if CE_S5_DBUF
        flex_intra_cluster_sync();   // GEMM t + 1 may write the other buffer now
#endif
    }
}

int main(void)
{
    uint32_t cid     = flex_get_cluster_id();
    uint32_t eoc_val = 0;
    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : 0u;
    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;
    /* phase: the step in which this stage first works, 0 for stage 1. An idle
     * cluster has stage 0, so its phase wraps to the largest uint32_t and it never
     * computes or copies; it still takes part in every global barrier. */
    uint32_t phase   = stage - 1u;
    const uint32_t buf_off = (uint32_t)(uintptr_t)X_BUF;

    flex_barrier_xy_init();
    flex_global_barrier_xy();
    /* Fill both buffers with fp16 1.0 before timing, so no NaN or subnormal slows
     * the arithmetic; this also stands for the constants written once at start
     * (1.cp, the global-probe rows of Q_g, K_g^T, V_g). Not timed. */
    for (uint32_t i = mempool_get_core_id(); i < X_ELEMS + X_PAR_ELEMS; i += mempool_get_core_count()) X_BUF[i] = (__fp16)1.0f;
    for (uint32_t i = mempool_get_core_id(); i < W_ELEMS; i += mempool_get_core_count()) W_BUF[i] = (__fp16)1.0f;
    flex_intra_cluster_sync();
    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();
    flex_global_barrier_xy();

    /* Compute then transfer, one global barrier per step. Nothing orders the
     * clusters inside a step, and the twin landing zone makes that safe: a producer
     * writes the copy of its item's parity, never the copy its consumer reads. So a
     * step costs the slowest stage's compute plus transfer. Only the head's last
     * drain of an item runs across a barrier, and only stage 5 waits for it. */
    for (uint32_t step = 0; step < CE_N_ITER + N_STAGES - 1; step++) {
        uint32_t active = (cid < N_CLUSTERS) && (phase <= step) && (step - phase < CE_N_ITER);
        uint32_t item   = step - phase;     // the item this stage works on in this step
        uint32_t par    = item & 1u;        // its parity: the copy of the landing zone
        if (active) {
            /* The stage's own LPDDR read for this item. It waits until the data are
             * in, so an LPDDR landing zone needs no twin. */
            if (flex_is_dm_core())
                lpddr_pull(stage, lane, (uint64_t)(uintptr_t)X_BUF);
            flex_intra_cluster_sync();   // worker cores wait for the LPDDR load
            switch (stage) {
            case STAGE_1: compute_stage_1(par, lane); break;
            case STAGE_2: compute_stage_2(par); break;
            case STAGE_3: compute_stage_3(par); break;
            case STAGE_4: compute_stage_4(par); break;
            case STAGE_5: compute_stage_5(par, item); break;
            default: break;
            }
            /* No barrier before this: the DM core leaves the switch with the rest of
             * its cluster and starts sending at once. A hand-off lands in the
             * consumer's landing zone, + XPAR_B for an odd item; the source is the
             * item's copy only for stages 2 and 3 (STAGE_SRC_PAR). */
            if (flex_is_dm_core()) {
                if (stage == STAGE_5 && item == CE_N_ITER - 1u) {
                    CE_REGION {   // the last item's last drain lands before the run ends (H.5 OUT)
                        flex_dma_async_wait_all();
                    }
                }
                uint64_t out = (uint64_t)(uintptr_t)X_BUF + (uint64_t)(par * STAGE_SRC_PAR[stage]) * XPAR_B;
                // lpddr_push(stage, lane, out); // disabled: only stage 1 uses it, moved to stage 1.
                if (stage < N_STAGES && STAGE_TILES[stage] == 1u)
                    push_to_next(stage, lane, buf_off + par * XPAR_B, out);
            }
        }
        flex_global_barrier_xy();   // one per step: compute and transfer done
    }

    if (flex_get_core_id() == 0 && cid == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}