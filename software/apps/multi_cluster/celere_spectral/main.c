// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


// CELERE-Spectral channel estimator on multi-cluster TensorPool:
//
//   stage 1  front end                 CE_S1_LANES lanes
//   stage 2  attention stage 1          CE_ENC_LANES lanes
//   stage 3  attention stage 2          CE_ENC_LANES lanes
//   stage 4  FFN and patch decoder      CE_S4_LANES lanes
//   stage 5  Doppler head               CE_S5_LANES lanes

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

// ==== knobs, sizes, L1 maps and checks =======================================
//
// Every size, offset and table entry of the app is a name computed here from a few knobs.
// Row ids (F.4, 1.8a, H.4, ...) are the rows of the OP table in:
// Mahdi Abdollahpour, PhD thesis, "EFFICIENT AND SCALABLE NEURAL BASEBAND PROCESSING FOR THE 5G NR UPLINK", 
// 2026, university of Bologna

// model: the received grid
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
// the array: a dual-polarized UPA
#ifndef CE_N_POL
#define CE_N_POL        2u     // polarizations
#endif
#ifndef CE_UPA_V
#define CE_UPA_V        4u     // UPA rows
#endif
#ifndef CE_UPA_H
#define CE_UPA_H        8u     // UPA columns
#endif
// model: tokens
#ifndef CE_TILE_V
#define CE_TILE_V       2u     // antenna elements per tile, vertical
#endif
#ifndef CE_TILE_H
#define CE_TILE_H       2u     // antenna elements per tile, horizontal
#endif
#ifndef CE_SPAN_PRBS
#define CE_SPAN_PRBS    2u     // PRBs in one frequency span
#endif
#ifndef CE_WIN_SPANS
#define CE_WIN_SPANS    12u    // spans in one window
#endif
// model: widths
#ifndef CE_D_MODEL
#define CE_D_MODEL      256u   // token width d, a multiple of CE_LINE
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

// mapping: clusters per stage
// Stage 1: front end | 2: attention stage 1 | 3: attention stage 2 | 4: FFN + decoder | 5: Doppler head. 
// Stages 1 to 4 parallelize on lattice lines (CE_NL), stage 5 the antenna elements (CE_N_EL) --> each lane count divides CE_NL.
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
#define CE_S5_TILES     4u     // tiles of the head's output layer (OP table H.4)
#endif
#ifndef CE_S5_DBUF
#define CE_S5_DBUF      1      // 1: double buffered last gemm/dma | 0: single buffer
#endif
#ifndef CE_S1_PAD
#define CE_S1_PAD       2u     // idle clusters after stage 1
#endif
#ifndef CE_S2_PAD
#define CE_S2_PAD       0u     // idle clusters after stage 2
#endif
#ifndef CE_S3_PAD
#define CE_S3_PAD       0u     // idle clusters after stage 3
#endif
#ifndef CE_S4_PAD
#define CE_S4_PAD       0u     // idle clusters after stage 4
#endif

#ifndef CE_N_ITER
#define CE_N_ITER       1u     // items streamed through the pipeline
#endif
#ifndef CE_BENCH
#define CE_BENCH        1      // 1: time every region
#endif

#ifndef CE_PE_FLOP_PER_CYCLE
#define CE_PE_FLOP_PER_CYCLE 280u   // waits for rows without a kernel
#endif
#ifndef CE_LHSM_STANDIN
#define CE_LHSM_STANDIN 0      // 1: timed wait (TODO: implemented, remove this)
#endif

#ifndef CE_LHSM_FLOP_PER_LOGIT
#define CE_LHSM_FLOP_PER_LOGIT 10u   // (TODO: remove)
#endif


#ifndef CE_LPDDR_SOUTH
#define CE_LPDDR_SOUTH  4u     // south LPDDR nodes: model input, summary patches
#endif
#ifndef CE_LPDDR_NORTH
#define CE_LPDDR_NORTH  4u     // north LPDDR nodes: the estimate
#endif

// ==== integer helpers (all used only at compile time) ======================================
#define CE_CDIV(a, b)     (((a) + (b) - 1u) / (b))   // a/b, rounding up
#define CE_RUP(a, b)      (CE_CDIV(a, b) * (b))      // a rounded up to a multiple of b
#define CE_MAX(a, b)      ((a) > (b) ? (a) : (b))
#define CE_MAX3(a, b, c)  CE_MAX(CE_MAX(a, b), c)
#define CE_LINE           32u   // one DMA line, in fp16 elements
#define CE_BYTES          2u    // fp16
#define CE_BANK_ROW       (NUM_BANKS * 4u / CE_BYTES)   // one word in every bank; X_BUF and W_BUF start on one
#define CE_GRP_COPY       (CE_BANK_ROW / NUM_GROUPS)    // from one group's banks to the next

// ==== the network built from the knobs ====================================

// frequency spans and windows of one line
#define CE_F            (CE_N_PRB * CE_SC_PER_PRB)                // subcarriers
#define CE_SPAN_SC      (CE_SPAN_PRBS * CE_SC_PER_PRB)            // subcarriers a span
#define CE_N_SPAN       CE_CDIV(CE_F, CE_SPAN_SC)                 // spans
#define CE_FP           (CE_N_SPAN * CE_SPAN_SC)                  // subcarriers, padded to whole spans
#define CE_N_WIN        CE_CDIV(CE_N_SPAN, CE_WIN_SPANS)          // windows
#define CE_LAST         (CE_N_SPAN - (CE_N_WIN - 1u) * CE_WIN_SPANS)   // spans in the last window
#define CE_CROP         (CE_SPAN_PRBS * CE_N_SPAN - CE_N_PRB)     // padding PRBs of the last span, cropped at the end
#define CE_NRS          (CE_N_SPAN + CE_N_WIN)                    // rows a line: spans, summaries
#define CE_LG           (CE_N_SPAN + CE_N_WIN)                    // keys of the global pass: spans, global probes
#define CE_LW           (CE_WIN_SPANS + 2u)                       // slots of a local window
#define CE_LG_PAD       CE_RUP(CE_LG, 4u)                         // CE_LG padded for the softmax

// antenna: elements, tiles, lines
#define CE_N_EL         (CE_N_POL * CE_UPA_V * CE_UPA_H)          // antenna elements
#define CE_T_V          (CE_UPA_V / CE_TILE_V)                    // tile rows
#define CE_T_H          (CE_UPA_H / CE_TILE_H)                    // tile columns
#define CE_TILE_EL      (CE_TILE_V * CE_TILE_H)                   // antenna elements a tile
#define CE_NL           (CE_N_EL / CE_TILE_EL)                    // lattice lines

// input strides, in antenna elements
#define CE_ST_EV        1u                                        // e_v
#define CE_ST_TV        CE_TILE_V                                 // t_v
#define CE_ST_EH        CE_UPA_V                                  // e_h
#define CE_ST_TH        (CE_UPA_V * CE_TILE_H)                    // t_h
// A lattice line is one tile (p, t_h, t_v) and holds its antenna elements (e_h, e_v).

// widths
#define CE_RE           (CE_PILOT_SYMS * 2u)                      // a subcarrier
#define CE_RUN          (CE_SC_PER_PRB * CE_RE)                   // one PRB of one antenna element (C_up)
#define CE_C            (CE_SPAN_PRBS * CE_TILE_EL * CE_RUN)      // a patch
_Static_assert(CE_C % CE_LINE == 0u, "a patch row is not whole 64 B lines");
#define CE_PER_EL       (CE_FP * CE_RE)                           // the input of one antenna element
#define CE_D_HEAD       (CE_D_MODEL / CE_HEADS)                   // width of a head
#define CE_D_QKV        (3u * CE_D_MODEL)                         // q|k|v of a row
#define CE_C_OUT        (CE_SC_PER_PRB * CE_N_SYM * 2u)           // the estimate of a PRB
#define CE_NSUM_HR      (CE_SPAN_PRBS * CE_N_WIN)                 // summary rows of the head an antenna element

// whole tensors that cross a stage or chip boundary, all lanes, in elements 
#define CE_IN_ALL       (CE_N_EL * CE_PER_EL)                     // model input
#define CE_WIN_ALL      (CE_NL * CE_N_WIN * CE_C)                 // summary patches
#define CE_ZR_ALL       (CE_NL * CE_N_SPAN * CE_D_MODEL)          // span tokens
#define CE_ZS_ALL       (CE_NL * CE_N_WIN * CE_D_MODEL)           // summary tokens (or one probe family)
#define CE_HPRB_ALL     (CE_N_EL * CE_N_PRB * CE_RUN)             // head input, PRB rows
#define CE_HSUM_ALL     (CE_N_EL * CE_NSUM_HR * CE_RUN)           // head input, summary rows
#define CE_OUT_ALL      (CE_N_EL * CE_N_PRB * CE_C_OUT)           // estimate

// parameters, in elements 
#define CE_W_EMB        (CE_C * CE_D_MODEL)                       // W_e
#define CE_W_TT         (2u * CE_D_MODEL)                         // the token-type rows
#define CE_W_LN         (2u * CE_D_MODEL)                         // gamma, beta
#define CE_W_QKV        (CE_D_MODEL * CE_D_QKV)                   // W_qkv, no bias
#define CE_W_WO         (CE_D_MODEL * CE_D_MODEL + CE_D_MODEL)    // W_o, b_o
#define CE_W_PQKV       (2u * CE_D_QKV)                           // q|k|v of the two probe rows (OP table 1.cp)
#define CE_W_PE         (2u * CE_D_MODEL)                         // the probe embeddings e_g, e_l (OP table 1.cp)
#define CE_W_ROPE1      (2u * CE_HEADS * CE_N_SPAN * (CE_D_HEAD / 2u))   // cos, sin of the span rows (OP table 1.cr)
#define CE_W_ROPE2      (2u * CE_HEADS * CE_LG * (CE_D_HEAD / 2u))       // cos, sin of spans, global probes (OP table 2.cr)
#define CE_W_LB1        (CE_HEADS * 3u * 3u + CE_HEADS)           // local bias G, delta (OP table 1.cl)
#define CE_W_LB2        (CE_HEADS * 2u * 3u + CE_HEADS)           // local bias G, delta (OP table 2.cl)
#define CE_W_FF1        (CE_D_MODEL * CE_FFN + CE_FFN)            // W_1, b_1
#define CE_W_FF2        (CE_FFN * CE_D_MODEL + CE_D_MODEL)        // W_2, b_2
#define CE_W_DEC        (CE_D_MODEL * CE_C + CE_C)                // W_dec, b_dec
#define CE_W_HX         (CE_RUN * CE_C_HID + CE_C_HID)            // W_x, b_x
#define CE_W_HS         (CE_RUN * CE_C_HID)                       // W_s
#define CE_W_HO         (CE_C_HID * CE_C_OUT + CE_C_OUT)          // W_h, b_h

// ==== stage maps===========================
//
// The layout rules:
//   - X_BUF: the landing zone of the hand-off first (stages 2 to 5; it has a twin
//     X_ELEMS further), then the stage's regions in row order. Every
//     row writes a region of its own, so a row reads exactly what the row before
//     it wrote, and nothing is overwritten while it is still needed.
//   - every region is a whole number of DMA lines: CE_RUP(n, CE_LINE);
//   - a GEMM's M is a multiple of the RedMulE engines 
//   - W_BUF holds every op row's parameters
//   - each name in a map is the start of a region, in fp16 elements
//
// Inside a stage the lattice rows are ordered by family, then line:
//   [ spans R: line, span | summaries S: line, window | global probes Lg | local probes Ll ]
// so every family of every line is contiguous, and each hand-off transfer is a few plain
// copies (one per family).

// == stage 1: front end (block F) ===============================================
#define S1_LINES        (CE_NL / CE_S1_LANES)                     // lines a lane
#define S1_IN_N         (CE_IN_ALL / CE_S1_LANES)                 // the input of the lane's antenna elements
#define S1_XR_ROWS      (S1_LINES * CE_N_SPAN)                    // span patches
#define S1_XS_ROWS      (S1_LINES * CE_N_WIN)                     // summary patches
#define S1_TOK          (S1_XR_ROWS + S1_XS_ROWS)                 // tokens: span and summary patches
#define S1_TOK_M        CE_RUP(S1_TOK, NUM_REDMULE_TILES)         // GEMM rows of the embedding (OP table F.4)
#define S1_WIN_N        (S1_XS_ROWS * CE_C)                       // elements of the summary patches
#define S1_TT_N         (S1_TOK * CE_D_MODEL)                     // elements of all tokens (OP table F.5)
// X_BUF: [ IN | X: span patches, summary patches | Z: span tokens, summary tokens ]
#define S1_IN           0u                                        // start of the model input of this lane (OP table F.1)
#define S1_X            (S1_IN + CE_RUP(S1_IN_N, CE_LINE))        // the patch matrix [span | summary] (OP table F.2, F.3)
#define S1_WIN          (S1_X + S1_XR_ROWS * CE_C)                // the summary patches, right after the span patches (OP table F.3)
#define S1_Z            (S1_X + CE_RUP(S1_TOK_M * CE_C, CE_LINE)) // the tokens [span | summary] (OP table F.4, F.5)
#define S1_ZS           (S1_Z + S1_XR_ROWS * CE_D_MODEL)          // the summary tokens, right after the span tokens
#define S1_END          (S1_Z + CE_RUP(S1_TOK_M * CE_D_MODEL, CE_LINE))   // end of the stage 1 map
// W_BUF: [ W_e | token-type rows ]
#define S1_W_EMB        0u                                        // start of the embedding matrix W_e (OP table F.4)
#define S1_W_TT         (S1_W_EMB + CE_RUP(CE_W_EMB, CE_LINE))    // the two token-type rows (OP table F.5)
#define S1_W_END        (S1_W_TT + CE_RUP(CE_W_TT, CE_LINE))      // end of the stage 1 weights

// == stages 2 and 3: attention stages 1 and 2 ===================================
#define E_LINES         (CE_NL / CE_ENC_LANES)                    // lines a lane
#define E_PAIRS         (E_LINES * CE_HEADS)                      // (line, head) pairs
#define E_TE_PAIR       (NUM_REDMULE_TILES / E_PAIRS)             // TEs a pair in the global pass
#define E_R_ROWS        (E_LINES * CE_N_SPAN)                     // span rows
#define E_S_ROWS        (E_LINES * CE_N_WIN)                      // rows of one family: S, Lg or Ll
#define E_RS_ROWS       (E_R_ROWS + E_S_ROWS)                     // span and summary rows
#define E_ALL_ROWS      (E_RS_ROWS + 2u * E_S_ROWS)               // all rows, probes too
// where each row family of a lane starts 
#define E_ROW_S         E_R_ROWS                                  // first summary row, after the span rows
#define E_ROW_LG        E_RS_ROWS                                 // first global-probe row, after the summaries
#define E_ROW_LL        (E_RS_ROWS + E_S_ROWS)                    // first local-probe row, after the global probes
#define E_RS_N          (E_RS_ROWS * CE_D_MODEL)                  // elements of the span and summary rows, what arrives in stage 2 and leaves stage 3 (OP table 2.13)
#define E_ALL_N         (E_ALL_ROWS * CE_D_MODEL)                 // elements of the whole lattice of a lane, probes included (OP table 1.13)
#define E_KP            CE_LG_PAD                                 // keys of a pair in the global pass: spans, global probes, pads with bias -inf (OP table 1.8a to 1.10a)
#define E_ASPAN_N       (E_PAIRS * CE_N_SPAN * 3u)                // local weights of the span queries (OP table 1.4, 1.11)
// query rows of a pair in the global pass, padded to a multiple of its TEs
#define A1_QR           CE_RUP(CE_LG, E_TE_PAIR)                  // spans and global probes (OP table 1.8a)
#define A2_QR           CE_RUP(CE_N_SPAN, E_TE_PAIR)              // spans; the global-probe queries are dead (OP table 2.8a)
// hub queries of a window in the local pass
#define A1_NHUB         2u                                        // S_w and L_w, both updated (OP table 1.4)
#define A2_NHUB         1u                                        // S_w only: nothing reads an update of L_w after this stage (OP table 2.4)
#define A1_AHUB_N       (E_PAIRS * CE_N_WIN * A1_NHUB * CE_LW)    // local weights of the hub queries (OP table 1.4, 1.11)
#define A2_AHUB_N       (E_PAIRS * CE_N_WIN * A2_NHUB * CE_LW)    // the same with one hub (OP table 2.4, 2.11)
// live logits of the local pass in a lane: span keys, hub slots
#define A1_LOC          (E_PAIRS * (CE_N_SPAN * 3u + CE_N_WIN * A1_NHUB * CE_LW))   // OP table 1.11
#define A2_LOC          (E_PAIRS * (CE_N_SPAN * 3u + CE_N_WIN * A2_NHUB * CE_LW))   // OP table 2.11
// stand-ins: RoPE
#define A1_ROPE_WAIT    ((3u * 2u * E_PAIRS * CE_N_SPAN * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)        // cycles
#define A2_ROPE_WAIT    ((3u * E_PAIRS * (CE_N_SPAN + CE_LG) * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)   // cycles
#define A1_LAV_WAIT     ((2u * A1_LOC * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)                         // cycles
#define A2_LAV_WAIT     ((2u * A2_LOC * CE_D_HEAD) / CE_PE_FLOP_PER_CYCLE)                         // cycles


// stage 2 (attention stage 1)
// X_BUF: [ Z | H | QKV | A_span | A_hub | Q_g | K_g^T | V_g | S_g | A_g | U_g | U | WO out ]
#define A1_Z            0u                                        // the lattice [R | S | Lg | Ll] [E_ALL_ROWS][CE_D_MODEL]; the landing zone, with its twin at + X_ELEMS
#define A1_H_M          CE_RUP(E_RS_ROWS, NUM_REDMULE_TILES)      // GEMM rows of the projection, the span and summary rows padded (OP table 1.2)
#define A1_WO_M         CE_RUP(E_ALL_ROWS, NUM_REDMULE_TILES)     // GEMM rows of the output projection, all rows padded (OP table 1.12)
#define A1_H            (A1_Z + CE_RUP(E_ALL_N, CE_LINE))         // layernorm output H [A1_H_M][CE_D_MODEL]; the probes' LN is a constant (OP table 1.1, read by 1.2)
#define A1_QKV          (A1_H + CE_RUP(A1_H_M * CE_D_MODEL, CE_LINE))     // q|k|v [A1_H_M][CE_D_QKV], unrotated (OP table 1.2, read by 1.4, 1.6, 1.7)
#define A1_ASPAN        (A1_QKV + CE_RUP(A1_H_M * CE_D_QKV, CE_LINE))     // local weights of the span queries [pair][CE_N_SPAN][3] (OP table 1.4, read by 1.11)
#define A1_AHUB         (A1_ASPAN + CE_RUP(E_ASPAN_N, CE_LINE))           // local weights of the hub queries [pair][CE_N_WIN][A1_NHUB][CE_LW] (OP table 1.4, read by 1.11)
#define A1_QG           (A1_AHUB + CE_RUP(A1_AHUB_N, CE_LINE))            // rotated queries Q_g [pair][A1_QR][CE_D_HEAD] (OP table 1.6, read by 1.8a)
#define A1_KT           (A1_QG + CE_RUP(E_PAIRS * A1_QR * CE_D_HEAD, CE_LINE))   // rotated keys K_g^T [pair][CE_D_HEAD][E_KP] (OP table 1.6, read by 1.8a)
#define A1_VG           (A1_KT + CE_RUP(E_PAIRS * CE_D_HEAD * E_KP, CE_LINE))    // values V_g [pair][E_KP][CE_D_HEAD] (OP table 1.7, read by 1.10a)
#define A1_SG           (A1_VG + CE_RUP(E_PAIRS * E_KP * CE_D_HEAD, CE_LINE))    // global logits S_g [pair][A1_QR][E_KP], bias added in place (OP table 1.8a, 1.9a)
#define A1_AG           (A1_SG + CE_RUP(E_PAIRS * A1_QR * E_KP, CE_LINE))        // global weights A_g [pair][A1_QR][E_KP] (OP table 1.9a, read by 1.10a)
#define A1_UG           (A1_AG + CE_RUP(E_PAIRS * A1_QR * E_KP, CE_LINE))        // global output U_g [pair][A1_QR][CE_D_HEAD] (OP table 1.10a, read by 1.11)
#define A1_U            (A1_UG + CE_RUP(E_PAIRS * A1_QR * CE_D_HEAD, CE_LINE))   // attention output U, heads merged [A1_WO_M][CE_D_MODEL] (OP table 1.11, read by 1.12)
#define A1_WOUT         (A1_U + CE_RUP(A1_WO_M * CE_D_MODEL, CE_LINE))           // output projection [A1_WO_M][CE_D_MODEL] (OP table 1.12, added into Z by 1.13)
#define A1_END          (A1_WOUT + CE_RUP(A1_WO_M * CE_D_MODEL, CE_LINE))        // end of the stage 2 map
// W_BUF: [ LN | W_qkv | probe q|k|v | probe embeddings | cos, sin | B_g | G, delta | W_o | Ll copies ]
#define A1_W_LN         0u                                        // layernorm gamma, beta (OP table 1.1)
#define A1_W_QKV        (A1_W_LN + CE_RUP(CE_W_LN, CE_LINE))      // W_qkv [CE_D_MODEL][CE_D_QKV] (OP table 1.2)
#define A1_W_PQKV       (A1_W_QKV + CE_RUP(CE_W_QKV, CE_LINE))    // constant q|k|v of the probes: the Lg row, then the Ll row (OP table 1.cp)
#define A1_W_PE         (A1_W_PQKV + CE_RUP(CE_W_PQKV, CE_LINE))  // probe embeddings: e_g, then e_l, copied into Z by OP table 1.13 (1.cp)
#define A1_W_ROPE       (A1_W_PE + CE_RUP(CE_W_PE, CE_LINE))      // cos, sin of the span rows; not read while the rotation is a stand-in (OP table 1.cr)
#define A1_W_GB         (A1_W_ROPE + CE_RUP(CE_W_ROPE1, CE_LINE)) // global bias [h][A1_QR][E_KP], the same for every line, pad keys at -inf (OP table 1.cg, read by 1.9a)
#define A1_W_LB         (A1_W_GB + CE_RUP(CE_HEADS * A1_QR * E_KP, CE_LINE))     // local bias G [h][1 + A1_NHUB][3], delta [h] (OP table 1.cl, read by 1.4)
#define A1_W_WO         (A1_W_LB + CE_RUP(CE_W_LB1, CE_LINE))     // W_o, b_o (OP table 1.12)
#define A1_W_LLC        CE_RUP(A1_W_WO + CE_RUP(CE_W_WO, CE_LINE), CE_BANK_ROW)   // the Ll row once per group, CE_GRP_COPY apart, on a row of banks: OP table 1.4 reads the copy in its own group (1.cp)
#define A1_W_END        (A1_W_LLC + CE_BANK_ROW)                  // end of the stage 2 weights

// stage 3 (attention stage 2)
// X_BUF: [ Z | H | QKV | A_span | A_hub | Q_g | K_g^T | V_g | S_g | A_g | U_g | U | WO out ]
#define A2_Z            0u                                        // the lattice [R | S | Lg | Ll] [E_ALL_ROWS][CE_D_MODEL]; the landing zone, with its twin at + X_ELEMS
#define A2_H_M          CE_RUP(E_ALL_ROWS, NUM_REDMULE_TILES)     // GEMM rows of the projection, all rows padded (OP table 2.2)
#define A2_WO_M         CE_RUP(E_RS_ROWS, NUM_REDMULE_TILES)      // GEMM rows of the output projection, the span and summary rows padded (OP table 2.12)
#define A2_H            (A2_Z + CE_RUP(E_ALL_N, CE_LINE))         // layernorm output H: all rows, sized [A2_H_M][CE_D_MODEL] for the TEs (OP table 2.1, read by 2.2)
#define A2_QKV          (A2_H + CE_RUP(A2_H_M * CE_D_MODEL, CE_LINE))     // q|k|v of all rows, unrotated, sized [A2_H_M][CE_D_QKV] (OP table 2.2, read by 2.4, 2.6, 2.7)
#define A2_ASPAN        (A2_QKV + CE_RUP(A2_H_M * CE_D_QKV, CE_LINE))     // local weights of the span queries [pair][CE_N_SPAN][3] (OP table 2.4, read by 2.11)
#define A2_AHUB         (A2_ASPAN + CE_RUP(E_ASPAN_N, CE_LINE))           // local weights of the one hub query S_w [pair][CE_N_WIN][CE_LW] (OP table 2.4, read by 2.11)
#define A2_QG           (A2_AHUB + CE_RUP(A2_AHUB_N, CE_LINE))            // rotated span queries Q_g [pair][A2_QR][CE_D_HEAD], CE_N_SPAN real (OP table 2.6, read by 2.8a)
#define A2_KT           (A2_QG + CE_RUP(E_PAIRS * A2_QR * CE_D_HEAD, CE_LINE))   // rotated keys K_g^T [pair][CE_D_HEAD][E_KP]: spans, then global probes (OP table 2.6, read by 2.8a)
#define A2_VG           (A2_KT + CE_RUP(E_PAIRS * CE_D_HEAD * E_KP, CE_LINE))    // values V_g [pair][E_KP][CE_D_HEAD]: spans, then global probes (OP table 2.7, read by 2.10a)
#define A2_SG           (A2_VG + CE_RUP(E_PAIRS * E_KP * CE_D_HEAD, CE_LINE))    // global logits S_g [pair][A2_QR][E_KP], bias added in place (OP table 2.8a, 2.9a)
#define A2_AG           (A2_SG + CE_RUP(E_PAIRS * A2_QR * E_KP, CE_LINE))        // global weights A_g [pair][A2_QR][E_KP] (OP table 2.9a, read by 2.10a)
#define A2_UG           (A2_AG + CE_RUP(E_PAIRS * A2_QR * E_KP, CE_LINE))        // global output U_g [pair][A2_QR][CE_D_HEAD], spans only (OP table 2.10a, read by 2.11)
#define A2_U            (A2_UG + CE_RUP(E_PAIRS * A2_QR * CE_D_HEAD, CE_LINE))   // attention output U, heads merged: the span and summary rows, sized A2_WO_M (OP table 2.11, read by 2.12)
#define A2_WOUT         (A2_U + CE_RUP(A2_WO_M * CE_D_MODEL, CE_LINE))           // output projection, sized [A2_WO_M][CE_D_MODEL] (OP table 2.12, added into Z by 2.13)
#define A2_END          (A2_WOUT + CE_RUP(A2_WO_M * CE_D_MODEL, CE_LINE))        // end of the stage 3 map
// W_BUF: [ LN | W_qkv | cos, sin | B_g | G, delta | W_o ]
#define A2_W_LN         0u                                        // layernorm gamma, beta (OP table 2.1)
#define A2_W_QKV        (A2_W_LN + CE_RUP(CE_W_LN, CE_LINE))      // W_qkv [CE_D_MODEL][CE_D_QKV] (OP table 2.2)
#define A2_W_ROPE       (A2_W_QKV + CE_RUP(CE_W_QKV, CE_LINE))    // cos, sin of the span and global-probe rows; not read while the rotation is a stand-in (OP table 2.cr)
#define A2_W_GB         (A2_W_ROPE + CE_RUP(CE_W_ROPE2, CE_LINE)) // global bias [h][A2_QR][E_KP], the same for every line, pad keys at -inf (OP table 2.cg, read by 2.9a)
#define A2_W_LB         (A2_W_GB + CE_RUP(CE_HEADS * A2_QR * E_KP, CE_LINE))     // local bias G [h][1 + A2_NHUB][3], delta [h] (OP table 2.cl, read by 2.4)
#define A2_W_WO         (A2_W_LB + CE_RUP(CE_W_LB2, CE_LINE))     // W_o, b_o (OP table 2.12)
#define A2_W_END        (A2_W_WO + CE_RUP(CE_W_WO, CE_LINE))      // end of the stage 3 weights

// == stage 4: FFN and patch decoder (block D) ===================================
// X_BUF: [ Z | BASE: X, S | H | F1 | F2 | D | head input: PRB rows, summary rows ]
#define S4_LINES        (CE_NL / CE_S4_LANES)                     // lines a lane
#define S4_EL           (CE_N_EL / CE_S4_LANES)                   // antenna elements a lane
#define S4_R_ROWS       (S4_LINES * CE_N_SPAN)                    // span rows
#define S4_ROWS         (S4_LINES * CE_NRS)                       // span and summary rows
#define S4_M            CE_RUP(S4_ROWS, NUM_REDMULE_TILES)        // GEMM rows, the rows padded (OP table D.2, D.4, D.7)
#define S4_Z_N          (S4_ROWS * CE_D_MODEL)                    // elements of the lattice that arrives from stage 3
#define S4_XB_N         (S4_R_ROWS * CE_C)                          // X, the lane's span patches from LPDDR, [S4_R_ROWS][CE_C] (OP table D.6)
#define S4_WIN_N        (CE_WIN_ALL / CE_S4_LANES)                // the lane's summary patches from LPDDR (OP table D.6)
#define S4_BASE_N       (S4_ROWS * CE_C)                          // the decoder residual [X | S] [S4_ROWS][CE_C] (OP table D.6, D.8)
#define S4_HPRB_N       (S4_EL * CE_N_PRB * CE_RUN)               // head input, PRB rows [S4_EL][CE_N_PRB][CE_RUN] (OP table D.9)
#define S4_HSUM_N       (S4_EL * CE_NSUM_HR * CE_RUN)             // head input, summary rows [S4_EL][CE_NSUM_HR][CE_RUN] (OP table D.9)
#define S4_Z            0u                                        // the lattice [R | S] [S4_ROWS][CE_D_MODEL]; the landing zone, with its twin at + X_ELEMS
#define S4_XB           (S4_Z + CE_RUP(S4_Z_N, CE_LINE))          // the decoder residual BASE = [X | S], X first (OP table D.6)
#define S4_WIN          (S4_XB + S4_XB_N)                         // S, right after X, so BASE is one tensor (OP table D.6)
#define S4_H            (S4_WIN + CE_RUP(S4_WIN_N, CE_LINE))      // layernorm output H [S4_M][CE_D_MODEL] (OP table D.1, read by D.2)
#define S4_F1           (S4_H + CE_RUP(S4_M * CE_D_MODEL, CE_LINE))       // FFN hidden [S4_M][CE_FFN], GELU in place (OP table D.2, D.3, read by D.4)
#define S4_F2           (S4_F1 + CE_RUP(S4_M * CE_FFN, CE_LINE))          // FFN output [S4_M][CE_D_MODEL], + Z in place (OP table D.4, D.5, read by D.7)
#define S4_D            (S4_F2 + CE_RUP(S4_M * CE_D_MODEL, CE_LINE))      // decoded patches [S4_M][CE_C], + BASE in place (OP table D.7, D.8, read by D.9)
#define S4_BLK          (S4_D + CE_RUP(S4_M * CE_C, CE_LINE))             // head input, PRB rows [antenna element][CE_N_PRB][CE_RUN] (OP table D.9, sent by D.10)
#define S4_BLK_S        (S4_BLK + S4_HPRB_N)                              // head input, summary rows [antenna element][CE_NSUM_HR][CE_RUN], right after (OP table D.9, sent by D.10)
#define S4_END          (S4_BLK_S + CE_RUP(S4_HSUM_N, CE_LINE))           // end of the stage 4 map
// W_BUF: [ LN | W_1 | W_2 | W_dec ]
#define S4_W_LN         0u                                        // layernorm gamma, beta (OP table D.1)
#define S4_W_FF1        (S4_W_LN + CE_RUP(CE_W_LN, CE_LINE))      // W_1, b_1 (OP table D.2, D.3)
#define S4_W_FF2        (S4_W_FF1 + CE_RUP(CE_W_FF1, CE_LINE))    // W_2, b_2 (OP table D.4)
#define S4_W_DEC        (S4_W_FF2 + CE_RUP(CE_W_FF2, CE_LINE))    // W_dec, b_dec (OP table D.7)
#define S4_W_END        (S4_W_DEC + CE_RUP(CE_W_DEC, CE_LINE))    // end of the stage 4 weights

// == stage 5: Doppler head (block H) ============================================
// X_BUF: [ x_in | h | OUT[0] | OUT[1] | go ]
#define S5_EL           (CE_N_EL / CE_S5_LANES)                   // antenna elements a lane
#define S5_XR_M         (S5_EL * CE_N_PRB)                        // PRB rows (OP table H.1)
#define S5_XS_M         (S5_EL * CE_NSUM_HR)                      // summary rows (OP table H.1)
#define S5_X_N          ((S5_XR_M + S5_XS_M) * CE_RUN)            // elements of the head input that arrives from stage 4
#define S5_HX_N         (S5_XR_M * CE_C_HID)                      // elements of h for the PRB rows [S5_XR_M][CE_C_HID] (OP table H.2, H.3)
#define S5_H_N          ((S5_XR_M + S5_XS_M) * CE_C_HID)          // elements of h, PRB and summary rows
#define S5_Y_ROWS       (S5_XR_M / CE_S5_TILES)                   // PRB rows a tile of the output layer (OP table H.4)
#define S5_Y_M          CE_RUP(S5_Y_ROWS, NUM_REDMULE_TILES)      // GEMM rows of a tile, S5_Y_ROWS padded (OP table H.4)
#define S5_TILE_X       (S5_Y_ROWS * CE_C_HID)                    // step from one tile's rows of h to the next (OP table H.4)
#define S5_OUT_N        CE_RUP(S5_Y_M * CE_C_OUT, CE_LINE)        // one OUT buffer, a tile's estimate sized [S5_Y_M][CE_C_OUT] (OP table H.4)
#define S5_NBUF         (CE_S5_DBUF ? 2u : 1u)                    // OUT buffers
#define S5_TILE_B       (S5_Y_ROWS * CE_C_OUT * CE_BYTES)         // the real rows of a tile, what one drain sends to LPDDR (OP table H.5)
#define S5_X            0u                                        // the head input x_in; the landing zone, with its twin at + X_ELEMS
#define S5_H            CE_RUP(S5_X + S5_X_N, CE_BANK_ROW)        // hidden layer h [PRB rows | summary rows][CE_C_HID], on a row of banks so H.2 and H.3 stay local (OP table H.1 to H.3, read by H.4)
#define S5_OUT          (S5_H + CE_RUP(S5_H_N, CE_LINE))          // OUT[0], with OUT[1] right after (OP table H.4, read by H.5)
#define S5_GO           CE_RUP(S5_OUT + S5_NBUF * S5_OUT_N, CE_BANK_ROW)    // two buffers: one go word a tile from the DM core, in the tile's first bank (OP table H.4)
#define S5_END          (CE_S5_DBUF ? S5_GO + CE_BANK_ROW : S5_OUT + S5_NBUF * S5_OUT_N)   // end of the stage 5 map
// W_BUF: [ W_x | W_s | W_h ]
#define S5_W_HX         0u                                        // W_x [CE_RUN][CE_C_HID], b_x: hidden layer of the PRB rows (OP table H.1, b_x into y^s)
#define S5_W_HS         (S5_W_HX + CE_RUP(CE_W_HX, CE_LINE))      // W_s [CE_RUN][CE_C_HID]: hidden layer of the summary rows, the window term (OP table H.1)
#define S5_W_HO         (S5_W_HS + CE_RUP(CE_W_HS, CE_LINE))      // W_h [CE_C_HID][CE_C_OUT], b_h: the output layer (OP table H.4)
#define S5_W_END        (S5_W_HO + CE_RUP(CE_W_HO, CE_LINE))      // end of the stage 5 weights
#define S5_B_X          (S5_W_HX + CE_RUN * CE_C_HID)             // b_x, after W_x: y^s starts from it (OP table H.c)
#define S5_B_H          (S5_W_HO + CE_C_HID * CE_C_OUT)           // b_h, after W_h: each OUT buffer starts from it (OP table H.c)
// ==== the two buffers ========================================================
enum { CE_X_ELEMS = CE_RUP(CE_MAX3(CE_MAX(S1_END, A1_END), CE_MAX(A2_END, S4_END), S5_END), CE_LINE) };
enum { CE_X_PAR_ELEMS = CE_RUP(CE_MAX3(E_ALL_N, S4_Z_N, S5_X_N), CE_LINE) };
enum { CE_W_ELEMS = CE_RUP(CE_MAX3(CE_MAX(S1_W_END, A1_W_END), CE_MAX(A2_W_END, S4_W_END), S5_W_END),
                           CE_LINE) };
// ==== LPDDR ===================================================================
// Offsets inside one node. South node j: [ model input | summary patches ] 
#define LP_IN_OFF       0u                                       // start of the model input in a south node, and of the span patches written over it
#define LP_IN_B         (CE_IN_ALL * CE_BYTES / CE_LPDDR_SOUTH)  // the node's share of the model input (OP table F.1); then the span patches of its lines, written over it (F.6, read by D.6)
#define LP_WIN_OFF      CE_RUP(LP_IN_OFF + LP_IN_B, 64u)         // the summary patches, right after the input, on a DMA line
#define LP_WIN_B        (CE_WIN_ALL * CE_BYTES / CE_LPDDR_SOUTH) // the summary patches of the node's lines; stage 1 writes them (OP table F.6), stage 4 reads them (D.6)
#define LP_SOUTH_B      (LP_WIN_OFF + LP_WIN_B)                  // all a south node holds
#define LP_OUT_B        (CE_OUT_ALL * CE_BYTES / CE_LPDDR_NORTH) // the estimate of one head lane, in its north node
#define LPDDR_OUTPUT_OFFSET 0u                                   // the estimate in a north node
#define TILE_OUT_BYTES  S5_TILE_B                                // one tile of the estimate, one drain (OP table H.5)

#define CE_NODES(lanes, edge)    ((lanes) <= (edge) ? (edge) / (lanes) : 1u)   // nodes per lane
#define CE_SHARERS(lanes, edge)  ((lanes) <= (edge) ? 1u : (lanes) / (edge))   // lanes per node


// ==== placement: a stage starts after the previous stage's lanes and idle ones
#define N_STAGES        5
#define S1_BASE_CID     0u
#define S2_BASE_CID     (S1_BASE_CID + CE_S1_LANES + CE_S1_PAD)
#define S3_BASE_CID     (S2_BASE_CID + CE_ENC_LANES + CE_S2_PAD)
#define S4_BASE_CID     (S3_BASE_CID + CE_ENC_LANES + CE_S3_PAD)
#define S5_BASE_CID     (S4_BASE_CID + CE_S4_LANES + CE_S4_PAD)
#define N_CLUSTERS      (S5_BASE_CID + CE_S5_LANES)               // cids the pipeline occupies
#define S1_NODES        CE_NODES(CE_S1_LANES, CE_LPDDR_SOUTH)     // nodes a lane
#define S1_SHARERS      CE_SHARERS(CE_S1_LANES, CE_LPDDR_SOUTH)
#define S4_NODES        CE_NODES(CE_S4_LANES, CE_LPDDR_SOUTH)
#define S4_SHARERS      CE_SHARERS(CE_S4_LANES, CE_LPDDR_SOUTH)   // lanes a node
#define S5_NODES        CE_NODES(CE_S5_LANES, CE_LPDDR_NORTH)
#define S5_SHARERS      CE_SHARERS(CE_S5_LANES, CE_LPDDR_NORTH)

// ==== checks =================================================================
// stop the build for invalid knobs

_Static_assert(CE_UPA_V % CE_TILE_V == 0u && CE_UPA_H % CE_TILE_H == 0u, "tiles must cover the array");
_Static_assert(CE_D_MODEL % CE_HEADS == 0u && CE_D_HEAD % 2u == 0u, "d_model must split into heads of even width");
_Static_assert(CE_CROP < CE_SPAN_PRBS, "the last span holds at least one PRB");

_Static_assert(CE_NL % CE_S1_LANES == 0u && S1_LINES % CE_T_V == 0u,
               "stage 1: a lane holds whole (p, t_h) columns of tiles");
_Static_assert(CE_NL % CE_ENC_LANES == 0u, "encoder lanes must divide the lines");
_Static_assert(CE_NL % CE_S4_LANES == 0u, "stage 4 lanes must divide the lines");
_Static_assert(CE_N_EL % CE_S5_LANES == 0u, "stage 5 lanes must divide the elements");

_Static_assert(CE_S1_LANES % CE_ENC_LANES == 0u || CE_ENC_LANES % CE_S1_LANES == 0u, "1 -> 2: lane counts");
_Static_assert(CE_ENC_LANES % CE_S4_LANES == 0u || CE_S4_LANES % CE_ENC_LANES == 0u, "3 -> 4: lane counts");
_Static_assert(CE_S4_LANES % CE_S5_LANES == 0u || CE_S5_LANES % CE_S4_LANES == 0u, "4 -> 5: lane counts");

_Static_assert(NUM_REDMULE_TILES % E_PAIRS == 0u, "the 16 TEs must split evenly over the (line, head) pairs");
_Static_assert((E_PAIRS * A1_QR) % NUM_REDMULE_TILES == 0u && (E_PAIRS * A2_QR) % NUM_REDMULE_TILES == 0u,
               "global pass: M must be a multiple of the 16 TEs");
_Static_assert(A1_QR >= CE_LG && A2_QR >= CE_N_SPAN && E_KP >= CE_LG, "global pass: padded sizes too small");

_Static_assert(E_RS_ROWS % 2u == 0u && E_ALL_ROWS % 2u == 0u && S4_ROWS % 2u == 0u, "layernorm: M must be even");
_Static_assert(CE_D_MODEL % 4u == 0u && E_KP % 4u == 0u, "layernorm, softmax: row length a multiple of 4");
_Static_assert((E_PAIRS * A1_QR) % 2u == 0u && (E_PAIRS * A2_QR) % 2u == 0u, "softmax: M must be even");
_Static_assert(S1_TT_N % 16u == 0u && E_ALL_N % 16u == 0u && E_RS_N % 16u == 0u && S4_Z_N % 16u == 0u
               && S4_BASE_N % 16u == 0u && (S4_ROWS * CE_FFN) % 16u == 0u && S5_HX_N % 16u == 0u
               && (CE_HEADS * A1_QR * E_KP) % 16u == 0u && (CE_HEADS * A2_QR * E_KP) % 16u == 0u,
               "axpy and gelu: length a multiple of 16, whole slices");
_Static_assert(BANKING_FACTOR == 8u, "axpy unrolled8 and the H.2 window term are built for 8 banks a core");
_Static_assert(CE_C_HID % 16u == 0u && CE_BANK_ROW % CE_C_HID == 0u
               && CE_NSUM_HR == CE_CDIV(CE_N_PRB, CE_WIN_SPANS * CE_SPAN_PRBS) * CE_SPAN_PRBS,
               "H.2: rows of h must be whole slices within a row of banks, with the summary rows the kernel counts");
_Static_assert(CE_C_OUT % 16u == 0u && S5_OUT % 16u == 0u && S5_OUT_N % 16u == 0u && (S5_H + S5_HX_N) % 16u == 0u
               && S5_B_X % 2u == 0u && S5_B_H % 2u == 0u, "H.1, H.4: the bias fills write whole slices");
_Static_assert(S1_TOK_M < 65536u && S5_XR_M < 65536u, "RedMulE: M, N, P below 65,536");
_Static_assert(CE_D_QKV <= CE_GRP_COPY, "1.4: a copy of the Ll row must stay in the banks of its group");

_Static_assert(CE_D_MODEL % CE_LINE == 0u, "a token row is not whole 64 B lines");
_Static_assert(E_RS_N % CE_LINE == 0u && S4_Z_N % CE_LINE == 0u
               && S5_X_N % CE_LINE == 0u, "inbound shares are not whole DMA lines"); //TODO

_Static_assert(S5_XR_M % CE_S5_TILES == 0u, "stage 5: the tiles must split the PRB rows evenly");

#define CE_L1_BUDGET_B  ((uint32_t)NUM_BANKS * L1_BANK_SIZE - (uint32_t)NUM_CORES * STACK_SIZE \
                         - (2u * NUM_BANKS * 4u + 2u * NUM_BANKS + 64u))
_Static_assert(CE_RUP((CE_X_ELEMS + CE_X_PAR_ELEMS) * CE_BYTES, NUM_BANKS * 4u)
                   + CE_W_ELEMS * CE_BYTES <= CE_L1_BUDGET_B,
               "X_BUF + W_BUF do not fit one cluster's L1 (with CE_S5_DBUF=1, try more CE_S5_TILES)");

_Static_assert(CE_LPDDR_SOUTH <= ARCH_NUM_CLUSTER_X && CE_LPDDR_NORTH <= ARCH_NUM_CLUSTER_X,
               "one LPDDR node per column at most");
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
               "per-node LPDDR data does not fit one node"); // a node's address window is ARCH_HBM_NODE_ADDR_SPACE; past it the address lands in the next node
_Static_assert(CE_S5_LANES == CE_LPDDR_NORTH && CE_S5_TILES * TILE_OUT_BYTES == LP_OUT_B,
               "the head drains each lane into the north node of its column: one lane per north node");
_Static_assert(LPDDR_OUTPUT_OFFSET + CE_S5_TILES * TILE_OUT_BYTES <= ARCH_HBM_NODE_ADDR_SPACE,
               "the head's output does not fit one LPDDR node");
_Static_assert(ARCH_NUM_CLUSTER == 32u, "CID_STAGE and CID_LANE list 32 cids, one per cluster");
_Static_assert(N_CLUSTERS <= ARCH_NUM_CLUSTER, "the pipeline needs more clusters than this arch has");



#include "baremetal/mempool_celere_lhsm_f16.h"
#include "baremetal/mempool_celere_wterm_f16.h"
#include "baremetal/mempool_celere_fill_rows_f16.h"

// CE_REGION { ... } marks a benchmarck region
// set CE_BENCH=1 

#if CE_BENCH
#define CE_REGION  mempool_start_benchmark(); \
                   for (int ce_r_ = 1; ce_r_; ce_r_ = (mempool_stop_benchmark(), 0))
#else
#define CE_REGION
#endif

// ==== placement tables ======================================================
// Stage s owns the consecutive clusters base_s .. base_s + lanes_s - 1,
// and cluster c is its 'c - base_s' lane.
// A cluster outside every stage is idle: stage 0, lane 0.


// cluster c is a lane of the stage at base
#define CID_IN(c, base, lanes)  ((c) >= (base) && (c) < (base) + (lanes))

// stage of cluster c
#define CID_STAGE_OF(c)  (1u * CID_IN(c, S1_BASE_CID, CE_S1_LANES)                     \
                        + 2u * CID_IN(c, S2_BASE_CID, CE_ENC_LANES)                    \
                        + 3u * CID_IN(c, S3_BASE_CID, CE_ENC_LANES)                    \
                        + 4u * CID_IN(c, S4_BASE_CID, CE_S4_LANES)                     \
                        + 5u * CID_IN(c, S5_BASE_CID, CE_S5_LANES))

// lane of cluster c in its stage
#define CID_LANE_OF(c)   (((c) - S1_BASE_CID) * CID_IN(c, S1_BASE_CID, CE_S1_LANES)    \
                        + ((c) - S2_BASE_CID) * CID_IN(c, S2_BASE_CID, CE_ENC_LANES)   \
                        + ((c) - S3_BASE_CID) * CID_IN(c, S3_BASE_CID, CE_ENC_LANES)   \
                        + ((c) - S4_BASE_CID) * CID_IN(c, S4_BASE_CID, CE_S4_LANES)    \
                        + ((c) - S5_BASE_CID) * CID_IN(c, S5_BASE_CID, CE_S5_LANES))

// stage of every cluster, one grid row per line
#define CID_STAGE_LIST  CID_STAGE_OF(0u),  CID_STAGE_OF(1u),  CID_STAGE_OF(2u),  CID_STAGE_OF(3u),   \
                        CID_STAGE_OF(4u),  CID_STAGE_OF(5u),  CID_STAGE_OF(6u),  CID_STAGE_OF(7u),   \
                        CID_STAGE_OF(8u),  CID_STAGE_OF(9u),  CID_STAGE_OF(10u), CID_STAGE_OF(11u),  \
                        CID_STAGE_OF(12u), CID_STAGE_OF(13u), CID_STAGE_OF(14u), CID_STAGE_OF(15u),  \
                        CID_STAGE_OF(16u), CID_STAGE_OF(17u), CID_STAGE_OF(18u), CID_STAGE_OF(19u),  \
                        CID_STAGE_OF(20u), CID_STAGE_OF(21u), CID_STAGE_OF(22u), CID_STAGE_OF(23u),  \
                        CID_STAGE_OF(24u), CID_STAGE_OF(25u), CID_STAGE_OF(26u), CID_STAGE_OF(27u),  \
                        CID_STAGE_OF(28u), CID_STAGE_OF(29u), CID_STAGE_OF(30u), CID_STAGE_OF(31u)

// lane of every cluster, one grid row per line
#define CID_LANE_LIST   CID_LANE_OF(0u),  CID_LANE_OF(1u),  CID_LANE_OF(2u),  CID_LANE_OF(3u),   \
                        CID_LANE_OF(4u),  CID_LANE_OF(5u),  CID_LANE_OF(6u),  CID_LANE_OF(7u),   \
                        CID_LANE_OF(8u),  CID_LANE_OF(9u),  CID_LANE_OF(10u), CID_LANE_OF(11u),  \
                        CID_LANE_OF(12u), CID_LANE_OF(13u), CID_LANE_OF(14u), CID_LANE_OF(15u),  \
                        CID_LANE_OF(16u), CID_LANE_OF(17u), CID_LANE_OF(18u), CID_LANE_OF(19u),  \
                        CID_LANE_OF(20u), CID_LANE_OF(21u), CID_LANE_OF(22u), CID_LANE_OF(23u),  \
                        CID_LANE_OF(24u), CID_LANE_OF(25u), CID_LANE_OF(26u), CID_LANE_OF(27u),  \
                        CID_LANE_OF(28u), CID_LANE_OF(29u), CID_LANE_OF(30u), CID_LANE_OF(31u)

// Stages are numbered from 1
enum { STAGE_1 = 1, STAGE_2, STAGE_3, STAGE_4, STAGE_5 };

// stage and lane of each cluster
static const uint8_t CID_STAGE[ARCH_NUM_CLUSTER] = { CID_STAGE_LIST };
static const uint8_t CID_LANE [ARCH_NUM_CLUSTER] = { CID_LANE_LIST };

// stage -> its first cid, and the number of clusters that run it
static const uint8_t STAGE_BASE [N_STAGES + 2] = { 0, S1_BASE_CID, S2_BASE_CID, S3_BASE_CID, S4_BASE_CID, S5_BASE_CID, N_CLUSTERS };
static const uint8_t STAGE_LANES[N_STAGES + 1] = { 0, CE_S1_LANES, CE_ENC_LANES, CE_ENC_LANES, CE_S4_LANES, CE_S5_LANES };

// Output tiles per stage (0 = idle). Stage 5 writes its tiles to LPDDR itself, so main() sends nothing for it.
static const uint8_t STAGE_TILES[N_STAGES + 1] = { 0, 1, 1, 1, 1, CE_S5_TILES };

// 1: send from this item's copy of the landing zone | 0: from the stage's own regions
static const uint8_t STAGE_SRC_PAR[N_STAGES + 1] = { 0, 0, 1, 1, 0, 0 };

// ==== hand-offs ==============================================================
// One entry per tensor sent to the next stage, in the order of the consumer's
// landing zone, where the tensors land back to back.
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

// stage s sends the entries STAGE_XFER_FIRST[s] to STAGE_XFER_FIRST[s + 1] - 1
static const uint8_t STAGE_XFER_FIRST[N_STAGES + 2] = { 0, 0, 2, 6, 8, 10, 10 };

// The landing zones of stages 2 to 5 hold, family after family, what the
// producers send.
_Static_assert(E_ROW_S * CE_D_MODEL * CE_BYTES == CE_ZR_ALL * CE_BYTES / CE_ENC_LANES
               && E_ROW_LG * CE_D_MODEL * CE_BYTES == (CE_ZR_ALL + CE_ZS_ALL) * CE_BYTES / CE_ENC_LANES
               && E_ROW_LL * CE_D_MODEL * CE_BYTES == (CE_ZR_ALL + 2u * CE_ZS_ALL) * CE_BYTES / CE_ENC_LANES,
               "stages 2 and 3: the lattice rows are not where the hand-off puts them");
_Static_assert(S4_R_ROWS * CE_D_MODEL == CE_ZR_ALL / CE_S4_LANES && S4_Z_N == (CE_ZR_ALL + CE_ZS_ALL) / CE_S4_LANES,
               "stage 4: the lattice rows are not where the hand-off puts them");
_Static_assert(S5_XR_M * CE_RUN == CE_HPRB_ALL / CE_S5_LANES && S5_X_N == (CE_HPRB_ALL + CE_HSUM_ALL) / CE_S5_LANES,
               "stage 5: the head input is not where the hand-off puts it");
// the smallest share the lane rules allow: one line of the lattice; the head input
// of one tile
_Static_assert((CE_ZR_ALL * CE_BYTES / CE_NL) % 64u == 0u && (CE_ZS_ALL * CE_BYTES / CE_NL) % 64u == 0u
               && (CE_HPRB_ALL * CE_BYTES / CE_NL) % 64u == 0u && (CE_HSUM_ALL * CE_BYTES / CE_NL) % 64u == 0u,
               "a share of a hand-off is not whole 64 B DMA lines");

// ==== LPDDR transfers ========================================================

#define LPDDR_EDGE_SOUTH  0u
#define LPDDR_EDGE_NORTH  1u


typedef struct {
    uint32_t lpddr_off;       // where the tensor starts inside a node, in bytes; the same in every node
    uint32_t l1_off;          // where the lane's share starts in X_BUF, in bytes
    uint32_t lane_step;       // distance between the starts of lanes that share a node, in bytes
    uint32_t chunk;           // bytes in one run
    uint32_t lpddr_stride;    // step between runs in LPDDR
    uint32_t l1_stride;       // step between runs in L1
    uint32_t reps;            // runs in one copy
    uint32_t edge;            // side of the chip: LPDDR_EDGE_SOUTH or LPDDR_EDGE_NORTH
    uint32_t nodes;           // nodes one lane reads or writes, one copy each
    uint32_t lanes_per_node;  // lanes that share one node
} lpddr_xfer_t;


// One LPDDR leg/transfer: the node_b bytes of a tensor in one node, split between the
// lanes that share the node, one descriptor each.
#define LPDDR_LEG(lpddr_off, l1_off, node_b, edge, nodes, share)                 \
    { (lpddr_off), (l1_off), (share) > 1u ? (node_b) / (share) : 0u,             \
      (node_b) / (share), (node_b) / (share), (node_b) / (share), 1u, (edge),    \
      (nodes), (share) }

static const lpddr_xfer_t LPDDR_IN_XFER[2] = {
    LPDDR_LEG(LP_IN_OFF,  S4_XB * CE_BYTES,  LP_IN_B,  LPDDR_EDGE_SOUTH, S4_NODES, S4_SHARERS),  // stage 4: span patches, written by stage 1 over its input (OP table D.6)
    LPDDR_LEG(LP_WIN_OFF, S4_WIN * CE_BYTES, LP_WIN_B, LPDDR_EDGE_SOUTH, S4_NODES, S4_SHARERS),  // stage 4: summary patches (OP table D.6)
};

// stage s reads the legs LPDDR_IN_FIRST[s] to LPDDR_IN_FIRST[s + 1] - 1
static const uint8_t LPDDR_IN_FIRST[N_STAGES + 2] = { 0, 0, 0, 0, 0, 2, 2 };

// ==== copies =================================================================
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

// push_to_next: run by the DM core of each producer cluster.
// dst_base: the consumer's landing zone
// src_base: this cluster's X_BUF, + the item's parity for stages 2 and 3
static __attribute__((noinline)) void push_to_next(uint32_t stage, uint32_t lane,
                                   uint32_t dst_base, uint64_t src_base)
{
    uint32_t next     = stage + 1u;
    uint32_t lanes    = STAGE_LANES[stage];
    uint32_t lanes_nx = STAGE_LANES[next];
    uint32_t base_nx  = STAGE_BASE[next];
    // The consumer's tensors lie next to each other in table order, so a running
    // acc (dst_off) reaches each one.
    uint32_t dst_off  = 0u;

    for (uint32_t e = STAGE_XFER_FIRST[stage]; e < STAGE_XFER_FIRST[next]; e++) {
        const xfer_t *x = &STAGE_XFER[e];
        uint32_t own_src = x->bytes / lanes;      // what this cluster holds
        uint32_t own_dst = x->bytes / lanes_nx;   // what one consumer needs

        if (lanes_nx > lanes) {
            // Fan-out: this lane's share covers r consecutive consumers.
            uint32_t r = lanes_nx / lanes;
            for (uint32_t k = 0; k < r; k++)
                dma_2d((uint64_t)remote_cid(base_nx + lane * r + k, dst_base + dst_off),
                       src_base + (uint64_t)(x->src_off + k * own_dst), own_dst);
        } else if (lanes > lanes_nx) {
            // Fan-in: r producers share one consumer; this lane lands in slot
            // lane % r of the consumer's tensor.
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

// Address of byte off in LPDDR node `node` (node j sits at column j) on one side.
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


// ==== the two L1 buffers of a cluster =====================================
#define X_ELEMS     CE_X_ELEMS       // the largest stage map, in elements
#define W_ELEMS     CE_W_ELEMS       // the largest weight map, in elements
#define X_PAR_ELEMS CE_X_PAR_ELEMS   // the largest landing zone: the size of its twin
#define XPAR_B      (X_ELEMS * (uint32_t)sizeof(__fp16))   // bytes from a landing-zone region to its twin
// the stage's map, then the twin of the landing zone, where odd items land
static __fp16 X_BUF[X_ELEMS + X_PAR_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")));
// parameters of every row, filled once, then only read
static __fp16 W_BUF[W_ELEMS] __attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1"), used));




// ==== utils ===============================================
// Starts G GEMMs at once
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


static inline void redmule_grouped_parallel(__fp16 *X, __fp16 *W, __fp16 *Y,
                                            uint32_t M, uint32_t N, uint32_t P,
                                            uint32_t groups, uint32_t w_step)
{
    redmule_grouped_start(X, W, Y, M, N, P, groups, w_step);
    wait_redmule();
    mempool_barrier(mempool_get_core_count());
}

static inline void probe_reset(__fp16 *dst, const __fp16 *e, uint32_t cid, uint32_t nt)
{
    uint32_t *d = (uint32_t *)dst;
    const uint32_t *s = (const uint32_t *)e;
    const uint32_t row_w = CE_D_MODEL / 2u;     // words a row
    for (uint32_t i = cid; i < 2u * E_S_ROWS * row_w; i += nt)
        d[i] = s[(i / row_w) / E_S_ROWS * row_w + i % row_w];
}

// ==== per-cluster compute: one function per stage ============================


// Stage 1, front end: patches, summary patches, embedding and LPDDR transfers.
static __attribute__((noinline)) void compute_stage_1(uint32_t par, uint32_t lane)
{
    (void)par;   // stage 1 reads from LPDDR: no twin
    uint32_t cid = mempool_get_core_id();     // in-cluster core id
    uint32_t nt  = mempool_get_core_count();  // cores per cluster
    uint32_t node = lane * S1_NODES;           // the lane's first south node, of its S1_NODES


    if (flex_is_dm_core())   // OP table F.1 IN
        for (uint32_t n = 0; n < S1_NODES; n++)
            dma_2d((uint64_t)(uintptr_t)(X_BUF + S1_IN) + n * LP_IN_B, hbm_south(node + n, LP_IN_OFF), LP_IN_B);
    flex_intra_cluster_sync();   // the other cores waits for the input


    CE_REGION {   // OP table F.2 PAT
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

    CE_REGION {   // OP table F.3 WIN
        uint32_t per = nt / S1_LINES;   // cores a line
        uint32_t l   = cid / per;

        pool_mean_f16p(X_BUF + S1_X + l * CE_N_SPAN * CE_C, X_BUF + S1_WIN + l * CE_N_WIN * CE_C,
                    CE_N_SPAN, CE_N_WIN, CE_WIN_SPANS, CE_C, cid % per, per);
        flex_intra_cluster_sync();
    }


    if (flex_is_dm_core())   // OP table F.6 OUT, under F.4
        for (uint32_t n = 0; n < S1_NODES; n++) {
            dma_2d(hbm_south(node + n, LP_IN_OFF),  (uint64_t)(uintptr_t)(X_BUF + S1_X) + n * LP_IN_B,    LP_IN_B);
            dma_2d(hbm_south(node + n, LP_WIN_OFF), (uint64_t)(uintptr_t)(X_BUF + S1_WIN) + n * LP_WIN_B, LP_WIN_B);
        }


    CE_REGION {   // OP table F.4 EMB
        redmule_synch_parallel(X_BUF + S1_X, W_BUF + S1_W_EMB, X_BUF + S1_Z, S1_TOK_M, CE_C, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }

    CE_REGION {   // OP table F.5 TT
        axpy_f16vecp_local_unrolled8(0x3C00u, X_BUF + S1_X, X_BUF + S1_Z, S1_TT_N);
        flex_intra_cluster_sync();
    }
}



// Stage 2, attention stage 1 (OP table rows 1.1 to 1.13)
static __attribute__((noinline)) void compute_stage_2(uint32_t par)
{
    uint32_t cid = mempool_get_core_id();      // in-cluster core id
    uint32_t nt  = mempool_get_core_count();   // cores per cluster
    __fp16 *z = X_BUF + par * X_ELEMS + A1_Z;  // this item's lattice
    // OP table 1.4
    static const lhsm_args_t lhsm = {
        .span = X_BUF + A1_QKV,                       .span_line = CE_N_SPAN * CE_D_QKV, .span_step = CE_D_QKV,
        .sum  = X_BUF + A1_QKV + E_ROW_S * CE_D_QKV,  .sum_line  = CE_N_WIN * CE_D_QKV,  .sum_step  = CE_D_QKV,
        .lpr  = W_BUF + A1_W_LLC,                     .lpr_line  = 0u,                   .lpr_step  = 0u,       .lpr_copy = CE_GRP_COPY,
        .bias = W_BUF + A1_W_LB,
        .a_span = X_BUF + A1_ASPAN, .a_hub = X_BUF + A1_AHUB,
        .n_lines = E_LINES, .n_heads = CE_HEADS, .n_hub = A1_NHUB,
        .n_span = CE_N_SPAN, .win = CE_WIN_SPANS, .d_model = CE_D_MODEL, .d_head = CE_D_HEAD,
    };
    CE_REGION {   // OP table 1.1 LN
        layernorm_parallel_2x4_f16vec(z, X_BUF + A1_H, E_RS_ROWS, CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.2 QKV
        redmule_synch_parallel(X_BUF + A1_H, W_BUF + A1_W_QKV, X_BUF + A1_QKV, A1_H_M, CE_D_MODEL, CE_D_QKV, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.3 SPLIT
    }
    // TODO

    CE_REGION {   // OP table 1.6 ROPE
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
    CE_REGION {   // OP table 1.6 ROPE, stand-in
        mempool_wait(A1_ROPE_WAIT);
        flex_intra_cluster_sync();   // Q_g, K_g^T complete before the TEs read them
    }
    CE_REGION {   // OP table 1.8a GQK
        redmule_grouped_start(X_BUF + A1_QG, X_BUF + A1_KT, X_BUF + A1_SG, E_PAIRS * A1_QR, CE_D_HEAD, E_KP,
                              E_PAIRS, CE_D_HEAD * E_KP);
    }
    CE_REGION {   // OP table 1.4 LHSM
        celere_lhsm_f16(&lhsm, cid, nt);
    }
    CE_REGION {   // OP table 1.7 VG
        static const perm_level_t lv[] = {
            { CE_N_SPAN, CE_D_QKV,              CE_D_HEAD },                        // span
            { CE_HEADS,  CE_D_HEAD,             E_KP * CE_D_HEAD },                 // head
            { E_LINES,   CE_N_SPAN * CE_D_QKV,  CE_HEADS * E_KP * CE_D_HEAD },      // line
        };
        permute_f16p(X_BUF + A1_QKV + 2u * CE_D_MODEL, X_BUF + A1_VG, CE_D_HEAD, lv, 3u, cid, nt);
    }
    CE_REGION {   // OP table 1.8a GQK
        wait_redmule();
        flex_intra_cluster_sync();   // S_g, A_span, A_hub, V_g complete
    }
    CE_REGION {   // OP table 1.9a GSM, 1.cg
        for (uint32_t l = 0; l < E_LINES; l++)
            axpy_f16vecp_local_unrolled8(0x3C00u, W_BUF + A1_W_GB, X_BUF + A1_SG + l * CE_HEADS * A1_QR * E_KP,
                                         CE_HEADS * A1_QR * E_KP);
        // one barrier for all lines: they write different rows and only read the bias
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.9a GSM
        softmax_parallel_2x4_f16vec(X_BUF + A1_SG, X_BUF + A1_AG, E_PAIRS * A1_QR, E_KP, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.10a GAV
        redmule_grouped_parallel(X_BUF + A1_AG, X_BUF + A1_VG, X_BUF + A1_UG, E_PAIRS * A1_QR, E_KP, CE_D_HEAD,
                                 E_PAIRS, E_KP * CE_D_HEAD);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.11 LAV, stand-in
        mempool_wait(A1_LAV_WAIT);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.12 WO
        redmule_synch_parallel(X_BUF + A1_U, W_BUF + A1_W_WO, X_BUF + A1_WOUT, A1_WO_M, CE_D_MODEL, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 1.13 RES
        probe_reset(z + E_ROW_LG * CE_D_MODEL, W_BUF + A1_W_PE, cid, nt);
        flex_intra_cluster_sync();
        axpy_f16vecp_local_unrolled8(0x3C00u, X_BUF + A1_WOUT, z, E_ALL_N);
        flex_intra_cluster_sync();
    }
}





// Stage 3, attention stage 2 (OP table rows 2.1 to 2.13)
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
    CE_REGION {   // OP table 2.1 LN
        layernorm_parallel_2x4_f16vec(z, X_BUF + A2_H, E_ALL_ROWS, CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.2 QKV
        redmule_synch_parallel(X_BUF + A2_H, W_BUF + A2_W_QKV, X_BUF + A2_QKV, A2_H_M, CE_D_MODEL, CE_D_QKV, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.3 SPLIT
    }
    CE_REGION {   // OP table 2.6 ROPE
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
        static const perm_level_t lg[] = {       // its global-probe rows, after them
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
    CE_REGION {   // OP table 2.6 ROPE, stand-in
        mempool_wait(A2_ROPE_WAIT);
        flex_intra_cluster_sync();   // Q_g, K_g^T complete before the TEs read them
    }
    CE_REGION {   // OP table 2.8a GQK
        redmule_grouped_start(X_BUF + A2_QG, X_BUF + A2_KT, X_BUF + A2_SG, E_PAIRS * A2_QR, CE_D_HEAD, E_KP,
                              E_PAIRS, CE_D_HEAD * E_KP);
    }
    CE_REGION {   // OP table 2.4 LHSM
        celere_lhsm_f16(&lhsm, cid, nt);
    }
    CE_REGION {   // OP table 2.7 VG
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
    CE_REGION {   // OP table 2.8a GQK
        wait_redmule();
        flex_intra_cluster_sync();   // S_g, A_span, A_hub and V_g complete
    }
    CE_REGION {   // OP table 2.9a GSM, 2.cg
        for (uint32_t l = 0; l < E_LINES; l++)
            axpy_f16vecp_local_unrolled8(0x3C00u, W_BUF + A2_W_GB, X_BUF + A2_SG + l * CE_HEADS * A2_QR * E_KP,
                                         CE_HEADS * A2_QR * E_KP);
        // one barrier for all lines: they write different rows and only read the bias
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.9a GSM
        softmax_parallel_2x4_f16vec(X_BUF + A2_SG, X_BUF + A2_AG, E_PAIRS * A2_QR, E_KP, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.10a GAV
        redmule_grouped_parallel(X_BUF + A2_AG, X_BUF + A2_VG, X_BUF + A2_UG, E_PAIRS * A2_QR, E_KP, CE_D_HEAD,
                                 E_PAIRS, E_KP * CE_D_HEAD);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.11 LAV, stand-in
        mempool_wait(A2_LAV_WAIT);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.12 WO
        redmule_synch_parallel(X_BUF + A2_U, W_BUF + A2_W_WO, X_BUF + A2_WOUT, A2_WO_M, CE_D_MODEL, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table 2.13 RES
        axpy_f16vecp_local_unrolled8(0x3C00u, X_BUF + A2_WOUT, z, E_RS_N);
        flex_intra_cluster_sync();
    }
}

// Stage 4, FFN and patch decoder (OP table rows D.1 to D.9)
static __attribute__((noinline)) void compute_stage_4(uint32_t par)
{
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    __fp16 *z = X_BUF + par * X_ELEMS + S4_Z;
    CE_REGION {   // OP table D.1 LN
        layernorm_parallel_2x4_f16vec(z, X_BUF + S4_H, S4_ROWS, CE_D_MODEL, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.2 FF1
        redmule_synch_parallel(X_BUF + S4_H, W_BUF + S4_W_FF1, X_BUF + S4_F1, S4_M, CE_D_MODEL, CE_FFN, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.3 GELU
        gelu_f16(X_BUF + S4_F1, S4_ROWS * CE_FFN, cid, nt);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.4 FF2
        redmule_synch_parallel(X_BUF + S4_F1, W_BUF + S4_W_FF2, X_BUF + S4_F2, S4_M, CE_FFN, CE_D_MODEL, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.5 RES
        axpy_f16vecp_local_unrolled8(0x3C00u, z, X_BUF + S4_F2, S4_Z_N);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.6 BASE
    }
    CE_REGION {   // OP table D.7 DEC
        redmule_synch_parallel(X_BUF + S4_F2, W_BUF + S4_W_DEC, X_BUF + S4_D, S4_M, CE_D_MODEL, CE_C, GEMM, false, 0);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.8 RES2
        axpy_f16vecp_local_unrolled8(0x3C00u, X_BUF + S4_XB, X_BUF + S4_D, S4_BASE_N);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table D.9 BLK
        static const perm_level_t lb[] = {       // span rows, whole spans
            { CE_TILE_V,          CE_RUN,                       CE_N_PRB * CE_RUN },                // e_v
            { CE_TILE_H,          CE_TILE_V * CE_RUN,           CE_TILE_V * CE_N_PRB * CE_RUN },    // e_h
            { CE_SPAN_PRBS,       CE_TILE_EL * CE_RUN,          CE_RUN },                           // PRB of the span
            { CE_N_SPAN - (CE_CROP ? 1u : 0u), CE_C,            CE_SPAN_PRBS * CE_RUN },            // span
            { S4_LINES,           CE_N_SPAN * CE_C,             CE_TILE_EL * CE_N_PRB * CE_RUN },   // line
        };
        static const perm_level_t ls[] = {       // summary rows
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

// Stage 5, Doppler head (OP table rows H.1 to H.5).
static __attribute__((noinline)) void compute_stage_5(uint32_t par, uint32_t item)
{
    uint32_t cid = mempool_get_core_id();
    uint32_t nt  = mempool_get_core_count();
    uint32_t hbm_node = get_pos(flex_get_cluster_id()).x;   // the north node of this column
    __fp16 *x = X_BUF + par * X_ELEMS + S5_X;              // [PRB rows | summary rows][CE_RUN]
    __fp16 *ys = X_BUF + S5_H + S5_HX_N;                   // summary rows of h, b_x already in
    CE_REGION {   // OP table H.1 HXS
        redmule_synch_parallel(x + S5_XR_M * CE_RUN, W_BUF + S5_W_HS, ys, S5_XS_M, CE_RUN, CE_C_HID, GEMM, false, 0);
    }
    CE_REGION {   // OP table H.2 HP
        celere_wterm_f16(ys, X_BUF + S5_H, S5_EL, CE_N_PRB, CE_WIN_SPANS * CE_SPAN_PRBS, CE_SPAN_PRBS, CE_C_HID);
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table H.1 HXS
        redmule_asynch_parallel(x, W_BUF + S5_W_HX, X_BUF + S5_H, S5_XR_M, CE_RUN, CE_C_HID, GEMM, false, 0);
    }
#if CE_S5_DBUF
    CE_REGION {   // OP table H.4 HO
        celere_fill_rows_f16(X_BUF + S5_OUT + ((item * CE_S5_TILES) & 1u) * S5_OUT_N, W_BUF + S5_B_H, CE_C_OUT, S5_Y_ROWS);
    }
#endif
    CE_REGION {   // OP table H.1 HXS
        wait_redmule();
        flex_intra_cluster_sync();
    }
    CE_REGION {   // OP table H.3 GELU
        gelu_f16(X_BUF + S5_H, S5_HX_N, cid, nt);
        flex_intra_cluster_sync();
    }

    for (uint32_t t = 0; t < CE_S5_TILES; t++) {

        // selects one of two buffers when double buffered (if the last drain works under the next item, then we need 'item' to choose the buffer)
        const uint32_t g = item * CE_S5_TILES + t;
        __fp16 *out = X_BUF + S5_OUT + (CE_S5_DBUF ? (g & 1u) : 0u) * S5_OUT_N;
#if !CE_S5_DBUF
        if (flex_is_dm_core()) {
            CE_REGION {   // OP table H.5 OUT
                flex_dma_async_wait_all();
            }
        }
        flex_intra_cluster_sync();
        CE_REGION {   // OP table H.4 HO
            celere_fill_rows_f16(out, W_BUF + S5_B_H, CE_C_OUT, S5_Y_ROWS);
            flex_intra_cluster_sync();
        }
#endif
        CE_REGION {   // OP table H.4 HO
            redmule_asynch_parallel(X_BUF + S5_H + t * S5_TILE_X, W_BUF + S5_W_HO, out, S5_Y_M, CE_C_HID, CE_C_OUT,
                                    GEMM, false, 0);
        }
#if CE_S5_DBUF
        volatile uint32_t *go = (volatile uint32_t *)(X_BUF + S5_GO);
        if (flex_is_dm_core()) {
            CE_REGION {   // OP table H.5 OUT
                flex_dma_async_wait_all();
                for (uint32_t c = 0; c < nt; c += NUM_CORES_PER_TILE)
                    go[c * BANKING_FACTOR] = g + 1u;
            }
        }
        CE_REGION {   // OP table H.4 HO
            if (t + 1u < CE_S5_TILES) {
                while (go[(cid & ~(NUM_CORES_PER_TILE - 1u)) * BANKING_FACTOR] != g + 1u)
                    mempool_wait(64);
                celere_fill_rows_f16(X_BUF + S5_OUT + ((g + 1u) & 1u) * S5_OUT_N, W_BUF + S5_B_H, CE_C_OUT, S5_Y_ROWS);
            } else {
                celere_fill_rows_f16(ys, W_BUF + S5_B_X, CE_C_HID, S5_XS_M);
            }
        }
#else
        if (t + 1u == CE_S5_TILES) {
            CE_REGION {   // OP table H.1 HXS
                celere_fill_rows_f16(ys, W_BUF + S5_B_X, CE_C_HID, S5_XS_M);
            }
        }
#endif
        CE_REGION {   // OP table H.4 HO
            wait_redmule();
            flex_intra_cluster_sync();
        }
        if (flex_is_dm_core()) {
            CE_REGION {   // OP table H.5 OUT
                flex_dma_async_2d(hbm_north(hbm_node, LPDDR_OUTPUT_OFFSET + t * TILE_OUT_BYTES),
                                  (uint64_t)(uintptr_t)out, TILE_OUT_BYTES, TILE_OUT_BYTES, TILE_OUT_BYTES, 1u);
            }
        }
    }
}

int main(void)
{
    uint32_t cid     = flex_get_cluster_id();
    uint32_t eoc_val = 0;
    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : 0u;
    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;

    uint32_t phase   = stage - 1u;   // first active step
    const uint32_t buf_off = (uint32_t)(uintptr_t)X_BUF;

    flex_barrier_xy_init();
    flex_global_barrier_xy();
    // Fill both buffers
    for (uint32_t i = mempool_get_core_id(); i < X_ELEMS + X_PAR_ELEMS; i += mempool_get_core_count()) X_BUF[i] = (__fp16)1.0f;
    for (uint32_t i = mempool_get_core_id(); i < W_ELEMS; i += mempool_get_core_count()) W_BUF[i] = (__fp16)1.0f;
    flex_intra_cluster_sync();
    if (stage == STAGE_5) {
        // b_x into y^s for the first item; later items get it under the last tile of the item before
        celere_fill_rows_f16(X_BUF + S5_H + S5_HX_N, W_BUF + S5_B_X, CE_C_HID, S5_XS_M);
        flex_intra_cluster_sync();
    }
    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();
    flex_global_barrier_xy();


    for (uint32_t step = 0; step < CE_N_ITER + N_STAGES - 1; step++) {
        uint32_t active = (cid < N_CLUSTERS) && (phase <= step) && (step - phase < CE_N_ITER);
        uint32_t item   = step - phase;     
        uint32_t par    = item & 1u;        // parity
        if (active) {

            if (flex_is_dm_core())
                lpddr_pull(stage, lane, (uint64_t)(uintptr_t)X_BUF);
            flex_intra_cluster_sync();   // wait for the LPDDR load
            switch (stage) {
            case STAGE_1: compute_stage_1(par, lane); break;
            case STAGE_2: compute_stage_2(par); break;
            case STAGE_3: compute_stage_3(par); break;
            case STAGE_4: compute_stage_4(par); break;
            case STAGE_5: compute_stage_5(par, item); break;
            default: break;
            }

            if (flex_is_dm_core()) {
                if (stage == STAGE_5 && item == CE_N_ITER - 1u) {
                    CE_REGION {   
                        flex_dma_async_wait_all();
                    }
                }
                uint64_t out = (uint64_t)(uintptr_t)X_BUF + (uint64_t)(par * STAGE_SRC_PAR[stage]) * XPAR_B;
                if (stage < N_STAGES && STAGE_TILES[stage] == 1u)
                    push_to_next(stage, lane, buf_off + par * XPAR_B, out);
            }
        }
        flex_global_barrier_xy();   // one per step
    }

    if (flex_get_core_id() == 0 && cid == 0) flex_timer_end();
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}