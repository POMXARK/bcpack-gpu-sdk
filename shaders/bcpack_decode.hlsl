// Clean-room Xbox Series BCPack (BC1) Compute Shader decoder (HLSL SM 6.6)
// v0.4.0 Zero-Barrier Wave32 Kernel:
// - 0 GroupMemoryBarrierWithGroupSync() calls inside the hot decode loop
// - Register-resident 64B stream window + WaveReadLaneAt byte reads (zero LDS stream traffic)
// - WaveReadLaneAt symbol handoff (eliminates 4,096 emit barriers per chunk)
// - 4-step branchless binary search + fused decode_and_advance (cuts LDS reads in half)
// - Reduced per-cycle wave votes (ballot_bad_end / ballot_bad_hdr only on chunk completion / 64 cycles)

struct ModelData {
    uint bf[10];     // low16 = base, high16 = freq
    uint rfrsn[10];  // bits 0..15 = rf, bits 16..20 = rs, bit 21 = rn
};

struct ChunkDesc {
    uint stream_offset;
    uint stream_len;
    uint blocks;
    uint out_block_offset;
};

StructuredBuffer<ModelData> g_Models    : register(t0);
StructuredBuffer<uint>      g_Window    : register(t1);
StructuredBuffer<uint>      g_QuadMasks : register(t2);
ByteAddressBuffer           g_Streams   : register(t3);
StructuredBuffer<ChunkDesc> g_Chunks    : register(t4);

RWStructuredBuffer<uint2>   g_OutBlocks : register(u0);
RWStructuredBuffer<uint4>   g_Status    : register(u1);

static const uint IDLE  = 0u;
static const uint HDR0  = 1u;
static const uint HDR1  = 2u;
static const uint HDR2  = 3u;
static const uint PROC  = 4u;
static const uint DRAIN = 5u;

static const uint RANS_L = 0x01000000u;

static const uint BC1_ALPHABET_BITS[12] = { 7u, 7u, 4u, 5u, 4u, 4u, 5u, 4u, 7u, 7u, 7u, 7u };

groupshared uint ring_ep[2][256];
groupshared uint lane_bf[10][24];
groupshared uint lane_rfrsn[10][24];

uint advance_state_slow(uint lm, uint x, uint sym) {
    uint seg = (sym == 0u) ? 0u : (firstbithigh(sym) + 1u);
    uint st = (seg == 0u) ? 0u : (1u << (seg - 1u));
    uint bf_s = lane_bf[seg][lm];
    uint base_s = bf_s & 0xFFFFu;
    uint fa = bf_s >> 16u;
    uint cfa = (base_s + fa * (sym - st)) & 0xFFFFu;
    uint f = (cfa == 0u) ? ((fa - 1u) & 0xFFFFu) : fa;
    uint c = (cfa == 0u) ? 1u : cfa;
    return f * (x >> 16u) + (x & 0xFFFFu) - c;
}

// 4-step binary search over [0, nseg) + fused rANS state advance (returns uint2(sym, next_p0))
uint2 decode_and_advance(uint lm, uint nseg, uint x) {
    uint cf = x & 0xFFFFu;
    uint lo = 0u;
    uint hi = nseg;

    uint mid = (lo + hi) >> 1u;
    if (cf >= (lane_bf[mid][lm] & 0xFFFFu)) lo = mid; else hi = mid;

    mid = (lo + hi) >> 1u;
    if (mid > lo && cf >= (lane_bf[mid][lm] & 0xFFFFu)) lo = mid; else hi = mid;

    mid = (lo + hi) >> 1u;
    if (mid > lo && cf >= (lane_bf[mid][lm] & 0xFFFFu)) lo = mid; else hi = mid;

    mid = (lo + hi) >> 1u;
    if (mid > lo && cf >= (lane_bf[mid][lm] & 0xFFFFu)) lo = mid;

    uint seg = lo;
    uint bf_s = lane_bf[seg][lm];
    uint base_s = bf_s & 0xFFFFu;
    uint r_s = lane_rfrsn[seg][lm];
    uint rf_s = r_s & 0xFFFFu;
    uint rs_s = (r_s >> 16u) & 31u;
    uint rn_s = (r_s >> 21u) & 1u;

    uint base_x = (1u << seg) >> 1u;
    uint dist = (cf - base_s + rn_s) & 0xFFFFu;
    uint steps = (((dist * rf_s) >> 15u) >> rs_s) & 0x1FFu;
    uint sym = (base_x + steps) & 0x1FFu;

    uint seg2 = (sym == 0u) ? 0u : (firstbithigh(sym) + 1u);
    uint bf_a = (seg2 == seg) ? bf_s : lane_bf[seg2][lm];
    uint st2 = (seg2 == 0u) ? 0u : (1u << (seg2 - 1u));
    uint base_a = bf_a & 0xFFFFu;
    uint fa = bf_a >> 16u;
    uint cfa = (base_a + fa * (sym - st2)) & 0xFFFFu;
    uint f = (cfa == 0u) ? ((fa - 1u) & 0xFFFFu) : fa;
    uint c = (cfa == 0u) ? 1u : cfa;
    uint next_p0 = f * (x >> 16u) + (x & 0xFFFFu) - c;
    return uint2(sym, next_p0);
}

bool is_legal_end_state(uint s) {
    return (s == RANS_L) || (s == (RANS_L >> 8u)) || (s == (RANS_L >> 16u));
}

uint back_distance(uint v) {
    v &= 0xFFu;
    return ((v & 1u) == 0u) ? (v >> 1u) : (255u - (v >> 1u));
}

int unzigzag(uint v) {
    int iv = int(v);
    return ((iv & 1) != 0) ? -((iv + 1) >> 1) : (iv >> 1);
}

[WaveSize(32)]
[numthreads(32, 1, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex) {
    uint sub = lane >> 4u;        // 0 = Chunk A (lanes 0..11), 1 = Chunk B (lanes 16..27)
    uint ll = lane & 15u;         // local lane index 0..15
    uint half_shift = sub * 16u;  // 0 or 16
    uint chunk_id = groupID.x * 2u + sub;

    ChunkDesc desc = g_Chunks[chunk_id];

    // Pre-fill 256-entry preset endpoint window for both chunks
    for (uint i = lane; i < 256u; i += 32u) {
        uint w = g_Window[i];
        ring_ep[0][i] = w;
        ring_ep[1][i] = w;
    }
    GroupMemoryBarrierWithGroupSync();

    bool half_active = (desc.blocks > 0u);
    bool lane_running = half_active && (ll < 12u);
    bool half_done_latch = !half_active;
    uint bit = (ll < 12u) ? (1u << ll) : 0u;
    uint ab = (ll < 12u) ? BC1_ALPHABET_BITS[ll] : 0u;
    uint lm = sub * 12u + min(ll, 11u);

    uint state = IDLE;
    uint ib = 0u;
    bool ib_full = false;
    uint sym = 0u;
    uint next_p0 = 0u;
    bool sym_full = false;
    uint slot_sym = 0u;
    uint in_rem = 0u;
    uint out_rem = 0u;
    uint p0 = 0u;
    uint p1 = 0u;
    bool hdr3 = false;
    bool flush = false;
    uint q = 0u;
    uint eab = 0u;
    uint nseg = 2u;
    bool done = false;
    bool go = true;
    bool exhausted = false;
    bool bad_hdr = false;
    uint final_state = 0u;

    uint slots = 0u;
    uint stream_pos = 0u;
    uint buf_base = 0xFFFFFFFFu;
    uint cache_word = 0u;
    uint block_idx = 0u;
    uint cycle = 0u;
    uint max_cycles = WaveActiveMax(half_active ? (64u * (12u * desc.blocks + desc.stream_len) + 1000u) : 1u);
    uint3 last_prog = uint3(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);

    while (cycle < max_cycles) {
        bool active_tick = !half_done_latch;

        // Register-resident 64-byte stream cache refill (0 barriers!)
        uint cur_abs = desc.stream_offset + stream_pos;
        if (active_tick && (buf_base == 0xFFFFFFFFu || (cur_abs + 8u) > (buf_base + 64u))) {
            buf_base = cur_abs & ~3u;
            cache_word = g_Streams.Load(buf_base + ll * 4u);
        }

        // ---------------- Phase 0: forward update, feed (max 8 bytes/clock per half), hand-off ----------------
        bool need_byte = active_tick && lane_running && !ib_full && (
            (state == HDR0 || state == HDR1 || state == HDR2) ||
            (state == PROC && in_rem > 0u)
        );
        bool blocked_out = active_tick && lane_running && sym_full && ((slots & bit) != 0u);
        bool want_byte = active_tick && lane_running && need_byte && !blocked_out && !exhausted;

        uint want_half = (WaveActiveBallot(want_byte).x >> half_shift) & 0xFFFu;
        uint rank = countbits(want_half & ((1u << ll) - 1u));
        bool in_window = want_byte && (rank < 8u);
        uint byte_idx = stream_pos + rank;
        bool valid = in_window && (byte_idx < desc.stream_len);
        if (in_window && !valid) {
            exhausted = true;
        }
        uint valid_half = (WaveActiveBallot(valid).x >> half_shift) & 0xFFFu;
        stream_pos += countbits(valid_half);

        // Unconditional uniform warp shuffle byte fetch (0 LDS reads, 0 barriers!)
        uint safe_rel = valid ? (desc.stream_offset + byte_idx - buf_base) : 0u;
        uint word = WaveReadLaneAt(cache_word, half_shift + ((safe_rel >> 2u) & 15u));
        uint fetched = valid ? ((word >> ((safe_rel & 3u) * 8u)) & 0xFFu) : 0u;

        bool stall = active_tick && lane_running && ((need_byte && !valid) || blocked_out);

        if (active_tick && lane_running && !stall) {
            if (state == PROC || state == DRAIN) {
                p0 = sym_full ? next_p0 : p1;
            }
            if (valid) {
                ib = fetched;
                ib_full = true;
            }
        }

        bool handoff = active_tick && lane_running && !stall && sym_full;
        if (handoff) {
            slot_sym = sym;
            sym_full = false;
        }
        slots |= (WaveActiveBallot(handoff).x >> half_shift) & 0xFFFu;

        // ---------------- Phase 1: header FSM, renorm, fused decode+advance ----------------
        if (active_tick && lane_running && !stall) {
            if (state == IDLE) {
                if (go) {
                    go = false;
                    state = HDR0;
                    out_rem = desc.blocks;
                    p0 = 0u;
                    p1 = 0u;
                }
            } else if (state == HDR0) {
                if (ib_full) {
                    uint b = ib;
                    ib_full = false;
                    in_rem = 0u;
                    state = HDR1;
                    q = b & 15u;
                    hdr3 = (b & 0x20u) != 0u;
                    flush = (b & 0x10u) != 0u;
                    int e = (int(ab) - int(b >> 6u)) & 15;
                    eab = (e < 10) ? uint(e) : 9u;
                    if (eab > 8u) {
                        bad_hdr = true;
                    }
                }
            } else if (state == HDR1) {
                if (ib_full) {
                    ib_full = false;
                    uint mid_id = eab * 16u + q;
                    nseg = eab + 2u;
                    ModelData m = g_Models[mid_id];
                    [unroll]
                    for (uint k = 0u; k < 10u; ++k) {
                        lane_bf[k][lm] = m.bf[k];
                        lane_rfrsn[k][lm] = m.rfrsn[k];
                    }
                    if (hdr3) {
                        in_rem = ib;
                        state = HDR2;
                    } else {
                        in_rem = ib + 1u;
                        state = PROC;
                    }
                }
            } else if (state == HDR2) {
                if (ib_full) {
                    ib_full = false;
                    in_rem = ((ib << 8u) | (in_rem & 0xFFu)) + 257u;
                    state = PROC;
                }
            } else if (state == PROC) {
                bool will = (p0 < RANS_L) && (in_rem > 0u) && ib_full;
                uint x = will ? (((p0 & 0xFFFFFFu) << 8u) | ib) : p0;
                bool nsf = (x > RANS_L) && (out_rem > 0u);
                if (nsf) {
                    out_rem -= 1u;
                    uint2 da = decode_and_advance(lm, nseg, x);
                    sym = da.x;
                    next_p0 = da.y;
                }
                p1 = x;
                sym_full = nsf;
                if (will) {
                    in_rem -= 1u;
                    ib_full = false;
                }
                if (in_rem == 0u) {
                    if (out_rem > 0u) {
                        if (x == RANS_L) {
                            state = HDR0;
                            if (flush) {
                                p1 = 0u;
                            }
                        } else {
                            state = DRAIN;
                        }
                    } else {
                        done = true;
                        state = IDLE;
                        final_state = nsf ? next_p0 : x;
                    }
                }
            } else { // DRAIN
                uint x = p0;
                bool nsf = (x > RANS_L) && (out_rem > 0u);
                if (nsf) {
                    out_rem -= 1u;
                    uint2 da = decode_and_advance(lm, nseg, x);
                    sym = da.x;
                    next_p0 = da.y;
                }
                p1 = x;
                sym_full = nsf;
                if (x <= RANS_L) {
                    if (out_rem > 0u) {
                        state = HDR0;
                        if (flush) {
                            p0 = 0u;
                            p1 = 0u;
                        }
                    } else {
                        state = IDLE;
                        done = true;
                        final_state = nsf ? next_p0 : x;
                    }
                }
            }
        }

        uint ballot_active = WaveActiveBallot(active_tick && lane_running && !(done && !sym_full && ((slots & bit) == 0u))).x;
        uint active_mask = (ballot_active >> half_shift) & 0xFFFu;
        bool half_emit = active_tick && (slots != 0u) && ((slots & active_mask) == active_mask);

        // ---------------- Lockstep symbol vector -> BC1 block via WaveReadLaneAt (0 barriers!) ----------------
        if (WaveActiveAnyTrue(half_emit)) {
            uint s0  = WaveReadLaneAt(slot_sym, half_shift + 0u);
            uint s2  = WaveReadLaneAt(slot_sym, half_shift + 2u);
            uint s3  = WaveReadLaneAt(slot_sym, half_shift + 3u);
            uint s4  = WaveReadLaneAt(slot_sym, half_shift + 4u);
            uint s5  = WaveReadLaneAt(slot_sym, half_shift + 5u);
            uint s6  = WaveReadLaneAt(slot_sym, half_shift + 6u);
            uint s7  = WaveReadLaneAt(slot_sym, half_shift + 7u);
            uint s8  = WaveReadLaneAt(slot_sym, half_shift + 8u);
            uint s9  = WaveReadLaneAt(slot_sym, half_shift + 9u);
            uint s10 = WaveReadLaneAt(slot_sym, half_shift + 10u);
            uint s11 = WaveReadLaneAt(slot_sym, half_shift + 11u);

            if (half_emit && ll == 0u && block_idx < desc.blocks) {
                uint d = back_distance(s0);
                uint pred = 0u;
                if (d > 0u) {
                    uint slot = uint((int(block_idx) - int(d)) & 255);
                    pred = ring_ep[sub][slot];
                }

                uint pr0 = (pred >> 11u) & 31u;
                uint pg0 = (pred >> 5u)  & 63u;
                uint pb0 = pred          & 31u;
                uint pr1 = (pred >> 27u) & 31u;
                uint pg1 = (pred >> 21u) & 63u;
                uint pb1 = (pred >> 16u) & 31u;

                uint r0 = uint(int(pr0) + unzigzag(s2)) & 31u;
                uint g0 = uint(int(pg0) + unzigzag(s3)) & 63u;
                uint b0 = uint(int(pb0) + unzigzag(s4)) & 31u;
                uint r1 = uint(int(pr1) + unzigzag(s5)) & 31u;
                uint g1 = uint(int(pg1) + unzigzag(s6)) & 63u;
                uint b1 = uint(int(pb1) + unzigzag(s7)) & 31u;

                uint c0 = (r0 << 11u) | (g0 << 5u) | b0;
                uint c1 = (r1 << 11u) | (g1 << 5u) | b1;
                uint ep = c0 | (c1 << 16u);
                ring_ep[sub][block_idx & 255u] = ep;

                uint idx_word = g_QuadMasks[s8 & 255u]
                              | g_QuadMasks[256u + (s9 & 255u)]
                              | g_QuadMasks[512u + (s10 & 255u)]
                              | g_QuadMasks[768u + (s11 & 255u)];

                g_OutBlocks[desc.out_block_offset + block_idx] = uint2(ep, idx_word);
            }

            if (half_emit) {
                block_idx += 1u;
                slots = 0u;
            }
        }

        cycle += 1u;

        // Per-half completion check (deferred error ballots only when a half finishes)
        uint ballot_rem = WaveActiveBallot(active_tick && lane_running && !(done && !sym_full)).x;
        uint rem_half = (ballot_rem >> half_shift) & 0xFFFu;
        bool just_done = active_tick && (rem_half == 0u);

        if (WaveActiveAnyTrue(just_done)) {
            uint ballot_bad_end = WaveActiveBallot(active_tick && lane_running && !is_legal_end_state(final_state)).x;
            uint ballot_bad_hdr = WaveActiveBallot(active_tick && bad_hdr).x;
            if (just_done) {
                uint err = (((ballot_bad_hdr >> half_shift) & 0xFFFu) != 0u)
                    ? 1u
                    : ((((ballot_bad_end >> half_shift) & 0xFFFu) != 0u) ? 2u : 0u);
                if (ll == 0u) {
                    g_Status[chunk_id] = uint4(err, stream_pos, cycle, block_idx);
                }
                half_done_latch = true;
            }
            if (WaveActiveAllTrue(half_done_latch)) {
                break;
            }
        }

        // Amortized bad-header & deadlock guard every 64 cycles
        if ((cycle & 63u) == 0u) {
            uint ballot_bad_hdr = WaveActiveBallot(active_tick && bad_hdr).x;
            uint st_val = (active_tick && lane_running) ? state : 0u;
            uint st_sum0 = WaveActiveSum((sub == 0u) ? st_val : 0u);
            uint st_sum1 = WaveActiveSum((sub == 1u) ? st_val : 0u);
            uint half_st_sum = (sub == 0u) ? st_sum0 : st_sum1;
            uint3 prog = uint3(stream_pos, block_idx, half_st_sum);
            bool half_stalled = active_tick && all(prog == last_prog);
            bool half_bad_hdr = active_tick && (((ballot_bad_hdr >> half_shift) & 0xFFFu) != 0u);
            if (half_bad_hdr || half_stalled) {
                if (ll == 0u) {
                    g_Status[chunk_id] = uint4(half_bad_hdr ? 1u : 3u, stream_pos, cycle, block_idx);
                }
                half_done_latch = true;
            }
            last_prog = prog;
            if (WaveActiveAllTrue(half_done_latch)) {
                break;
            }
        }
    }

    if (!half_done_latch && ll == 0u) {
        g_Status[chunk_id] = uint4(4u, stream_pos, cycle, block_idx);
    }
}
