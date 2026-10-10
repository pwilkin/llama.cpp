#pragma once
#include "common.cuh"

// 32 bf16 weights from 16 packed pairs with four 16-byte stores
__device__ __forceinline__ void mmw_st4(uint16_t * dst, const uint32_t (&p)[16]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) *(uint4 *) (dst + 8 * i) = make_uint4(p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]);
}

// ksigns_iq2xs[i] without the table: 7 sign bits plus an even-parity bit 7
__device__ __forceinline__ uint32_t mmw_ksigns(const uint32_t i) { return i | ((__popc(i) & 1u) << 7); }

// one MXFP4 block for the prefetch: the 16 qs bytes and the e8m0 scale, each in its own register
struct mmw_mxfp4_blk { uint4 qs; uint32_t e; };

// the blocks are 17 bytes, so qs is only byte-aligned; it is still one 16-byte load
__device__ __forceinline__ void mmw_mxfp4_fetch(const block_mxfp4 * x, mmw_mxfp4_blk & f) {
    memcpy(&f.qs, x->qs, sizeof(f.qs));
    f.e = x->e;
}

// 4 entries of a 16-entry byte table (entry i is byte i & 3 of t[i >> 2]): sel has each code's low 3 bits, s has bit 3 in bit 7. Codes 0..7 and 8..15 come from two v_perms; perm selectors 8..11 turn the bits 7 into byte masks.
__device__ __forceinline__ uint32_t mmw_lut16(const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3,
                                              const uint32_t sel, const uint32_t s) {
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel), hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = __builtin_amdgcn_perm(s << 8, s, 0x090B080Au);
    return (lo & ~m) | (hi & m);
}

// kvalues_mxfp4 + 12 for 4 codes (sel, s as mmw_lut16)
__device__ __forceinline__ uint32_t mmw_mxfp4_kv4(const uint32_t sel, const uint32_t s) {
    return mmw_lut16(0x0F0E0D0Cu, 0x18141210u, 0x090A0B0Cu, 0x00040608u, sel, s);
}

// the 32 weights of an MXFP4 block with e >= 252 (d * kv can overflow) as bf16 pairs: (d * kv) * 0.5f per weight as dequantize_mxfp4. The values are integers times a power of two (or inf/NaN), so their low 16 bits are zero and the high half is the RNE bf16.
__device__ __forceinline__ void mmw_dq_mxfp4_big(const mmw_mxfp4_blk & f, uint32_t (&p)[16]) {
    const float d = ggml_cuda_e8m0_to_fp32(f.e);
    const uint32_t qs[4] = {f.qs.x, f.qs.y, f.qs.z, f.qs.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint32_t u = mmw_mxfp4_kv4((qs[w] >> 4 * k) & 0x07070707u, qs[w] << (4 - 4 * k));
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = d * ((float) ((u >> 8 * j) & 0xFF) - 12.0f) * 0.5f;
            p[8 * k + 2 * w]     = __builtin_amdgcn_perm(__float_as_uint(v[1]), __float_as_uint(v[0]), 0x07060302u);
            p[8 * k + 2 * w + 1] = __builtin_amdgcn_perm(__float_as_uint(v[3]), __float_as_uint(v[2]), 0x07060302u);
        }
    }
}

// 16 bytes of an 18-byte Q2_0 block for half h: half 0 loads bytes 0..15 (d, qs 0..13), half 1 bytes 2..17 (qs 0..15). So one load per thread covers the block; half 1 gets d from its neighbour lane.
struct mmw_q20_half { uint4 b; };

// the blocks are 18 bytes, so the load is only 2-byte aligned
__device__ __forceinline__ void mmw_q20_fetch(const block_q2_0 * x, const int h, mmw_q20_half & f) {
    memcpy(&f.b, (const uint8_t *) x + 2 * h, sizeof(f.b));
}

// 4 two-bit codes (one byte) -> one per byte
__device__ __forceinline__ uint32_t mmw_spread4(const uint32_t c8) {
    const uint32_t t = (c8 | (c8 << 12)) & 0x000f000fu;
    return (t | (t << 6)) & 0x03030303u;
}

// x rounded to bf16 (RNE as mmw_f2bf), the bf16 in the high half
__device__ __forceinline__ uint32_t mmw_rb(const float x) { const uint32_t u = __float_as_uint(x); return u + 0x7fffu + ((u >> 16) & 1u); }

// a 4-entry bf16 table for v_perm from mmw_rb values: the low bytes in lo, the high bytes in hi
__device__ __forceinline__ void mmw_tab(const uint32_t r0, const uint32_t r1, const uint32_t r2, const uint32_t r3, uint32_t & lo, uint32_t & hi) {
    const uint32_t p01 = __builtin_amdgcn_perm(r1, r0, 0x07030602u), p23 = __builtin_amdgcn_perm(r3, r2, 0x07030602u);
    lo = __builtin_amdgcn_perm(p23, p01, 0x05040100u);
    hi = __builtin_amdgcn_perm(p23, p01, 0x07060302u);
}

// 4 bf16 weights from their codes (0..3, one per byte of c) and a mmw_tab table. For each 2 weights one v_perm makes the byte selector (code, code + 4) and one takes their bytes.
__device__ __forceinline__ uint2 mmw_tab4(const uint32_t lo, const uint32_t hi, const uint32_t c) {
    const uint32_t c4 = c | 0x04040404u;
    return make_uint2(__builtin_amdgcn_perm(hi, lo, __builtin_amdgcn_perm(c4, c, 0x05010400u)),
                      __builtin_amdgcn_perm(hi, lo, __builtin_amdgcn_perm(c4, c, 0x07030602u)));
}

// the 32 weights of half h of a Q2_0 block, (code - 1) * d as dequantize_q2_0; lanes 2k and 2k + 1 hold the two halves. For a finite d the 4 values are rounded to bf16 once and picked with v_perm; a non-finite d multiplies per weight (so NaN signs match).
__device__ __forceinline__ void mmw_dq_q20_half(const mmw_q20_half & f, const int h, uint16_t * dst) {
    // DPP quad_perm [0, 0, 2, 2]: every lane gets the first dword of the even lane of its pair, the one holding d
    const uint32_t d16 = __builtin_amdgcn_update_dpp(0u, f.b.x, 0xa0, 0xf, 0xf, false) & 0xffff;
    const uint2 qs = h ? make_uint2(f.b.z, f.b.w) : make_uint2(__builtin_amdgcn_alignbit(f.b.y, f.b.x, 16), __builtin_amdgcn_alignbit(f.b.z, f.b.y, 16));
    const float d = mmw_h2f((uint16_t) d16);
    uint32_t p[16];
    if ((d16 & 0x7c00u) == 0x7c00u) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t q = (w < 2 ? qs.x : qs.y) >> 16 * (w & 1);
#pragma unroll
            for (int j = 0; j < 4; ++j) p[4 * w + j] = mmw_pack2(((int) ((q >> (4 * j)) & 3) - 1) * d, ((int) ((q >> (4 * j + 2)) & 3) - 1) * d);
        }
    } else {
        uint32_t lo, hi;
        mmw_tab(mmw_rb(-d), mmw_rb(0.0f * d), mmw_rb(d), mmw_rb(2.0f * d), lo, hi);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t q = (w < 2 ? qs.x : qs.y) >> 16 * (w & 1);
            const uint2 a = mmw_tab4(lo, hi, mmw_spread4(q & 0xff)), b = mmw_tab4(lo, hi, mmw_spread4((q >> 8) & 0xff));
            p[4 * w] = a.x; p[4 * w + 1] = a.y; p[4 * w + 2] = b.x; p[4 * w + 3] = b.y;
        }
    }
    mmw_st4(dst, p);
}

// lo/hi tables of db * {8, 25, 43}, the IQ2 grid values (codes 0, 1, 2): the same fp32 products as the per-weight decode
__device__ __forceinline__ void mmw_iq2_tab(const float db, uint32_t & lo, uint32_t & hi) {
    mmw_tab(mmw_rb(db * 8.0f), mmw_rb(db * 25.0f), mmw_rb(db * 43.0f), 0u, lo, hi);
}

// the bf16 sign-bit XOR masks of 4 weights from their 4 sign bits: entry i of the LDS sign table
__device__ __forceinline__ uint2 mmw_sign_masks(const uint32_t i) {
    return make_uint2((i & 1u) << 15 | (i & 2u) << 30, (i & 4u) << 13 | (i & 8u) << 28);
}

// 8 bf16 weights (a: 0..3, b: 4..7) with their sign bits flipped by the sign byte sg. RNE is sign-symmetric, so this equals rounding d * grid * sign.
__device__ __forceinline__ uint4 mmw_sign8(const uint2 a, const uint2 b, const uint32_t sg) {
    return make_uint4(a.x ^ ((sg &  1u) << 15 | (sg &   2u) << 30), a.y ^ ((sg &  4u) << 13 | (sg &   8u) << 28),
                      b.x ^ ((sg & 16u) << 11 | (sg &  32u) << 26), b.y ^ ((sg & 64u) <<  9 | (sg & 128u) << 24));
}

// mmw_sign8 with the masks from the 16-entry LDS sign table st
__device__ __forceinline__ uint4 mmw_sign8(const uint2 a, const uint2 b, const uint2 * st, const uint32_t sg) {
    const uint2 m0 = st[sg & 15], m1 = st[sg >> 4];
    return make_uint4(a.x ^ m0.x, a.y ^ m0.y, b.x ^ m1.x, b.y ^ m1.y);
}

// two-part lo/hi tables (codes 0..3 in [0], 4..7 in [1]) of db * v[c] for the 8 grid values v of an IQ3 type. These are the same fp32 products as the per-weight decode.
__device__ __forceinline__ void mmw_iq3_tab(const float db, const float (&v)[8], uint32_t (&lo)[2], uint32_t (&hi)[2]) {
#pragma unroll
    for (int h = 0; h < 2; ++h) mmw_tab(mmw_rb(db * v[4 * h]), mmw_rb(db * v[4 * h + 1]), mmw_rb(db * v[4 * h + 2]), mmw_rb(db * v[4 * h + 3]), lo[h], hi[h]);
}

// 4 bf16 weights from their codes (0..7, one per byte of c) and a two-part table. The low and the high bytes come from separate v_perms (selector 12 gives a zero byte).
__device__ __forceinline__ uint2 mmw_tab4x(const uint32_t (&lo)[2], const uint32_t (&hi)[2], const uint32_t c) {
    auto pick = [&](const uint32_t sl, const uint32_t sh) {
        return __builtin_amdgcn_perm(lo[1], lo[0], __builtin_amdgcn_perm(0x0c0c0c0cu, c, sl)) |
               __builtin_amdgcn_perm(hi[1], hi[0], __builtin_amdgcn_perm(0x0c0c0c0cu, c, sh));
    };
    return make_uint2(pick(0x05010400u, 0x01050004u), pick(0x07030602u, 0x03070206u));
}

// LDS-codebook decoders for the grid-based quants: fetch() loads the thread's blocks of a super-block, one field per register (no early waits). The LDS grid (entry()) holds per-weight codes; decode<S>() rounds the few scaled grid values to bf16 once and picks them with v_perm.
template <int WTYPE> struct mmw_lb { static constexpr bool ok = false; using grid_t = uint32_t; static constexpr int N = 1; struct regs {}; };

// the dword of block sub (0..1) of a slice's pair with one v_perm, so the pick by lane cannot turn into a register index
__device__ __forceinline__ uint32_t mmw_pick(const uint2 v, const int sub) { return __builtin_amdgcn_perm(v.y, v.x, sub ? 0x07060504u : 0x03020100u); }

template <> struct mmw_lb<32 + GGML_TYPE_IQ3_S> {
    static constexpr bool ok = true; using grid_t = uint32_t; static constexpr int N = 512;
    // per slice: q 8 low grid index bytes, sg 4 sign bytes; qh the high index bits and sc the 4-bit scales of all blocks
    struct regs { uint2 q[4]; uint32_t sg[4]; uint2 qh; uint32_t sc, d; };
    static __device__ __forceinline__ grid_t entry(const int i) { return (iq3s_grid[i] >> 1) & 0x07070707u; }   // grid byte 1, 3, .., 15 -> code 0..7
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, regs & f) {
        const block_iq3_s * x = (const block_iq3_s *) row + (ks * 64) / QK_K;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            memcpy(&f.q[s], x->qs + 8 * (2 * s + sub), sizeof(f.q[s]));
            memcpy(&f.sg[s], x->signs + 4 * (2 * s + sub), sizeof(f.sg[s]));
        }
        memcpy(&f.qh, x->qh, sizeof(f.qh));
        memcpy(&f.sc, x->scales, sizeof(f.sc));
        f.d = *(const uint16_t *) &x->d;
    }
    template <int S> static __device__ __forceinline__ void decode(const grid_t * g, const uint2 * st, const regs & f, const int sub, uint16_t * dst) {
        const uint32_t qh = ((S < 2 ? f.qh.x : f.qh.y) >> (16 * (S & 1) + 8 * sub)) & 0xff;
        constexpr float v[8] = {1, 3, 5, 7, 9, 11, 13, 15};
        uint32_t lo[2], hi[2];
        mmw_iq3_tab(mmw_h2f((uint16_t) f.d) * (1 + 2 * (int) ((f.sc >> (8 * S + 4 * sub)) & 0xf)), v, lo, hi);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q = (il < 2 ? f.q[S].x : f.q[S].y) >> 16 * (il & 1);
            *(uint4 *) (dst + 8 * il) = mmw_sign8(mmw_tab4x(lo, hi, g[(q & 0xff) | ((qh << (8 - 2 * il)) & 256)]),
                mmw_tab4x(lo, hi, g[((q >> 8) & 0xff) | ((qh << (7 - 2 * il)) & 256)]), st, (f.sg[S] >> 8 * il) & 0xff);
        }
    }
};
template <> struct mmw_lb<32 + GGML_TYPE_IQ2_XXS> {
    static constexpr bool ok = true; using grid_t = uint64_t; static constexpr int N = 256;
    // per slice: q.x 4 grid indices, q.y the signs and the scale
    struct regs { uint2 q[4]; uint32_t d; };
    static __device__ __forceinline__ grid_t entry(const int i) { return (iq2xxs_grid[i] >> 4) & 0x0f0f0f0f0f0f0f0full; }   // grid byte >> 4 = code
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, regs & f) {
        const block_iq2_xxs * x = (const block_iq2_xxs *) row + (ks * 64) / QK_K;
#pragma unroll
        for (int s = 0; s < 4; ++s) memcpy(&f.q[s], x->qs + 4 * (2 * s + sub), sizeof(f.q[s]));
        f.d = *(const uint16_t *) &x->d;
    }
    template <int S> static __device__ __forceinline__ void decode(const grid_t * g, const uint2 * st, const regs & f, const int sub, uint16_t * dst) {
        GGML_UNUSED(st); GGML_UNUSED(sub);
        uint32_t lo, hi;
        mmw_iq2_tab(mmw_h2f((uint16_t) f.d) * (0.5f + (f.q[S].y >> 28)) * 0.25f, lo, hi);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint64_t c = g[(f.q[S].x >> 8 * il) & 0xff];
            *(uint4 *) (dst + 8 * il) = mmw_sign8(mmw_tab4(lo, hi, (uint32_t) c), mmw_tab4(lo, hi, (uint32_t) (c >> 32)),
                mmw_ksigns((f.q[S].y >> 7 * il) & 127));
        }
    }
};
template <> struct mmw_lb<32 + GGML_TYPE_IQ2_XS> {
    static constexpr bool ok = true; using grid_t = uint64_t; static constexpr int N = 512;
    // per slice: q 4 grid index + sign words; sc the scale bytes of all blocks
    struct regs { uint2 q[4]; uint2 sc; uint32_t d; };
    static __device__ __forceinline__ grid_t entry(const int i) { return (iq2xs_grid[i] >> 4) & 0x0f0f0f0f0f0f0f0full; }   // grid byte >> 4 = code
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, regs & f) {
        const block_iq2_xs * x = (const block_iq2_xs *) row + (ks * 64) / QK_K;
#pragma unroll
        for (int s = 0; s < 4; ++s) memcpy(&f.q[s], x->qs + 4 * (2 * s + sub), sizeof(f.q[s]));
        memcpy(&f.sc, x->scales, sizeof(f.sc));
        f.d = *(const uint16_t *) &x->d;
    }
    template <int S> static __device__ __forceinline__ void decode(const grid_t * g, const uint2 * st, const regs & f, const int sub, uint16_t * dst) {
        GGML_UNUSED(st);
        const uint32_t sc = (S < 2 ? f.sc.x : f.sc.y) >> (16 * (S & 1) + 8 * sub);
        uint32_t lo[2], hi[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) mmw_iq2_tab(mmw_h2f((uint16_t) f.d) * (0.5f + ((sc >> 4 * h) & 0xf)) * 0.25f, lo[h], hi[h]);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q = ((il < 2 ? f.q[S].x : f.q[S].y) >> 16 * (il & 1)) & 0xffff;
            const uint64_t c = g[q & 511];
            *(uint4 *) (dst + 8 * il) = mmw_sign8(mmw_tab4(lo[il / 2], hi[il / 2], (uint32_t) c), mmw_tab4(lo[il / 2], hi[il / 2], (uint32_t) (c >> 32)),
                mmw_ksigns(q >> 9));
        }
    }
};
// IQ2_S: the LDS table holds the 8 codes of an entry as nibbles (4 KB instead of 8 KB): low nibbles weights 0..3, high 4..7
template <> struct mmw_lb<32 + GGML_TYPE_IQ2_S> {
    static constexpr bool ok = true; using grid_t = uint32_t; static constexpr int N = 1024;
    // per slice: qs the low grid index bytes and sg the sign bytes of both blocks; qh the high index bits and sc the scale bytes of all blocks
    struct regs { uint2 qs[4], sg[4]; uint2 qh, sc; uint32_t d; };
    static __device__ __forceinline__ grid_t entry(const int i) {
        const uint64_t c = (iq2s_grid[i] >> 4) & 0x0f0f0f0f0f0f0f0full;   // grid byte >> 4 = code
        return (uint32_t) c | (uint32_t) (c >> 28);
    }
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, regs & f) {
        GGML_UNUSED(sub);
        const block_iq2_s * x = (const block_iq2_s *) row + (ks * 64) / QK_K;
        memcpy(f.qs, x->qs, sizeof(f.qs));
        memcpy(f.sg, x->qs + QK_K / 8, sizeof(f.sg));
        memcpy(&f.qh, x->qh, sizeof(f.qh));
        memcpy(&f.sc, x->scales, sizeof(f.sc));
        f.d = *(const uint16_t *) &x->d;
    }
    template <int S> static __device__ __forceinline__ void decode(const grid_t * g, const uint2 * st, const regs & f, const int sub, uint16_t * dst) {
        constexpr int sh = 16 * (S & 1);
        const uint32_t qh = ((S < 2 ? f.qh.x : f.qh.y) >> (sh + 8 * sub)) & 0xff, sc = (S < 2 ? f.sc.x : f.sc.y) >> (sh + 8 * sub);
        const uint32_t qs = mmw_pick(f.qs[S], sub), sg = mmw_pick(f.sg[S], sub);
        uint32_t lo[2], hi[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) mmw_iq2_tab(mmw_h2f((uint16_t) f.d) * (0.5f + ((sc >> 4 * h) & 0xf)) * 0.25f, lo[h], hi[h]);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t c = g[((qs >> 8 * il) & 0xff) | ((qh << (8 - 2 * il)) & 0x300)];
            *(uint4 *) (dst + 8 * il) = mmw_sign8(mmw_tab4(lo[il / 2], hi[il / 2], c & 0x0f0f0f0fu), mmw_tab4(lo[il / 2], hi[il / 2], (c >> 4) & 0x0f0f0f0fu),
                st, (sg >> 8 * il) & 0xff);
        }
    }
};
template <> struct mmw_lb<32 + GGML_TYPE_IQ3_XXS> {
    static constexpr bool ok = true; using grid_t = uint32_t; static constexpr int N = 256;
    // per slice: q 8 grid indices, a the signs and the scale
    struct regs { uint2 q[4]; uint32_t a[4]; uint32_t d; };
    static __device__ __forceinline__ grid_t entry(const int i) { return (iq3xxs_grid[i] >> 3) & 0x07070707u; }   // grid byte 4, 12, .., 52, 62 -> code 0..7
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int sub, regs & f) {
        const block_iq3_xxs * x = (const block_iq3_xxs *) row + (ks * 64) / QK_K;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            memcpy(&f.q[s], x->qs + 8 * (2 * s + sub), sizeof(f.q[s]));
            memcpy(&f.a[s], x->qs + QK_K / 4 + 4 * (2 * s + sub), sizeof(f.a[s]));
        }
        f.d = *(const uint16_t *) &x->d;
    }
    template <int S> static __device__ __forceinline__ void decode(const grid_t * g, const uint2 * st, const regs & f, const int sub, uint16_t * dst) {
        GGML_UNUSED(sub);
        constexpr float v[8] = {4, 12, 20, 28, 36, 44, 52, 62};
        uint32_t lo[2], hi[2];
        mmw_iq3_tab(mmw_h2f((uint16_t) f.d) * (0.5f + (f.a[S] >> 28)) * 0.5f, v, lo, hi);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t q = (il < 2 ? f.q[S].x : f.q[S].y) >> 16 * (il & 1);
            *(uint4 *) (dst + 8 * il) = mmw_sign8(mmw_tab4x(lo, hi, g[q & 0xff]), mmw_tab4x(lo, hi, g[(q >> 8) & 0xff]), st,
                mmw_ksigns((f.a[S] >> 7 * il) & 127));
        }
    }
};

// Register decoders for the other quants: a thread decodes one 32-weight half h of a row's 64-wide K slice (two threads per row). fetch() loads ns slices, one field per register (no early waits); decode<S>() writes slice S as bf16 with the fp32 math of dequantize_*.
template <int WTYPE> struct mmw_rq { static constexpr bool ok = false; static constexpr int ns = 1; struct regs {}; };

// the 32-weight-block quants: a thread's half of a slice is one block (B::load, B::decode), and NS slices are fetched at once. The last fetch of a row repeats its last slice instead of reading past the row (those registers are never decoded).
template <typename B, int NS> struct mmw_rq_blk {
    static constexpr bool ok = true; static constexpr int ns = NS;
    struct regs { typename B::slot s[NS]; };
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int nks, const int h, regs & f) {
#pragma unroll
        for (int i = 0; i < NS; ++i) B::load(row, min(ks + i, nks - 1), h, f.s[i]);
    }
    template <int S> static __device__ __forceinline__ void decode(const regs & f, const int h, uint16_t * row) {
        uint32_t p[16];
        B::decode(f.s[S], p);
        mmw_st4(row + 32 * h, p);
    }
};

// Q8_0: d * q as dequantize_q8_0
struct mmw_q8_0 {
    struct slot { uint4 q0, q1; uint32_t d; };
    // the blocks are 34 bytes, so qs is only 2-byte aligned; each half is still one 16-byte load
    static __device__ __forceinline__ void load(const uint8_t * row, const int k, const int h, slot & f) {
        const uint8_t * p = row + (size_t) k * 68 + 34 * h;
        memcpy(&f.q0, p + 2, sizeof(f.q0));
        memcpy(&f.q1, p + 18, sizeof(f.q1));
        memcpy(&f.d, p, sizeof(f.d));
    }
    // for a positive finite d, d * q is fma(q + 128, d, -128 * d): -128 * d is exact, so the rounding is the same and q = 0 gives +0. q + 128 is the byte with its top bit flipped, read by v_cvt_f32_ubyteN; other d keep the per-weight multiply.
    static __device__ __forceinline__ void decode(const slot & f, uint32_t (&p)[16]) {
        const uint32_t d16 = __builtin_amdgcn_perm(0u, f.d, 0x0c0c0100u);   // opaque: a & 0xffff can be hoisted next to the load
        const float d = mmw_h2f((uint16_t) d16);
        const uint32_t q[8] = {f.q0.x, f.q0.y, f.q0.z, f.q0.w, f.q1.x, f.q1.y, f.q1.z, f.q1.w};
        if (d16 - 1u < 0x7bffu) {
            const float md = -128.0f * d;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t u = q[w] ^ 0x80808080u;
                p[2 * w]     = mmw_pack2(fmaf((float) (u & 0xFFu), d, md), fmaf((float) ((u >> 8) & 0xFFu), d, md));
                p[2 * w + 1] = mmw_pack2(fmaf((float) ((u >> 16) & 0xFFu), d, md), fmaf((float) (u >> 24), d, md));
            }
        } else {
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                p[2 * w]     = mmw_pack2(d * (float) (int8_t) (q[w]),       d * (float) (int8_t) (q[w] >> 8));
                p[2 * w + 1] = mmw_pack2(d * (float) (int8_t) (q[w] >> 16), d * (float) (int8_t) (q[w] >> 24));
            }
        }
    }
};
template <> struct mmw_rq<1> : mmw_rq_blk<mmw_q8_0, 2> {};

// IQ4_NL: kv * d as fma(kv + 128, d, -128 * d) (-128 * d is exact, so the single rounding equals RN(kv * d) of dequantize_iq4_nl)
struct mmw_iq4_nl {
    struct slot { uint4 q; uint32_t d; };
    static __device__ __forceinline__ void load(const uint8_t * row, const int k, const int h, slot & f) {
        const uint8_t * p = row + (size_t) k * 36 + 18 * h;
        memcpy(&f.q, p + 2, sizeof(f.q));
        f.d = *(const uint16_t *) p;
    }
    // low nibbles are weights 0..15, high nibbles 16..31. The 16 values of the block are rounded to bf16 once. Their lo and hi bytes are two 16-entry v_perm tables, each read once per 4 weights.
    static __device__ __forceinline__ void decode(const slot & f, uint32_t (&p)[16]) {
        const float d = mmw_h2f((uint16_t) f.d), md = -128.0f * d;
        constexpr float kv[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};   // kv + 128
        uint32_t lo[4], hi[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) mmw_tab(mmw_rb(fmaf(kv[4 * t], d, md)), mmw_rb(fmaf(kv[4 * t + 1], d, md)), mmw_rb(fmaf(kv[4 * t + 2], d, md)),
                                            mmw_rb(fmaf(kv[4 * t + 3], d, md)), lo[t], hi[t]);
        const uint32_t q[4] = {f.q.x, f.q.y, f.q.z, f.q.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t sel = (q[w] >> 4 * k) & 0x07070707u, s = q[w] << (4 - 4 * k);
                const uint32_t l = mmw_lut16(lo[0], lo[1], lo[2], lo[3], sel, s), u = mmw_lut16(hi[0], hi[1], hi[2], hi[3], sel, s);
                p[8 * k + 2 * w]     = __builtin_amdgcn_perm(u, l, 0x05010400u);
                p[8 * k + 2 * w + 1] = __builtin_amdgcn_perm(u, l, 0x07030602u);
            }
        }
    }
};
template <> struct mmw_rq<0> : mmw_rq_blk<mmw_iq4_nl, 4> {};

// bits 0..3 of x to bit 0 of bytes 0..3
__device__ __forceinline__ uint32_t mmw_bits4(const uint32_t x) {
    uint32_t t = x & 15;
    t |= t << 7;
    t |= t << 14;
    return t & 0x01010101u;
}

// Q5_1: q * d + m as dequantize_q5_1
struct mmw_q5_1 {
    struct slot { uint2 dmh; uint4 qs; };   // dmh: d and m, qh
    // d, m and qh are 8-byte aligned, qs is 16-byte aligned only in odd blocks
    static __device__ __forceinline__ void load(const uint8_t * row, const int k, const int h, slot & f) {
        const uint8_t * p = row + (size_t) k * 48 + 24 * h;
        f.dmh = *(const uint2 *) p;
        memcpy(&f.qs, p + 8, sizeof(f.qs));
    }
    // low nibbles are weights 0..15, high nibbles 16..31, qh bit i is the fifth bit of weight i
    static __device__ __forceinline__ void decode(const slot & f, uint32_t (&p)[16]) {
        const float d = mmw_h2f((uint16_t) f.dmh.x), m = mmw_h2f((uint16_t) (f.dmh.x >> 16));
        const uint32_t qs[4] = {f.qs.x, f.qs.y, f.qs.z, f.qs.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t q = ((qs[j] >> 4 * k) & 0x0F0F0F0Fu) | mmw_bits4(f.dmh.y >> (16 * k + 4 * j)) << 4;
                float v[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) v[l] = d * (float) ((q >> 8 * l) & 0xFF) + m;
                p[8 * k + 2 * j] = mmw_pack2(v[0], v[1]); p[8 * k + 2 * j + 1] = mmw_pack2(v[2], v[3]);
            }
        }
    }
};
template <> struct mmw_rq<32 + GGML_TYPE_Q5_1> : mmw_rq_blk<mmw_q5_1, 4> {};

// MXFP4: one 17-byte block (mmw_mxfp4_fetch)
struct mmw_mxfp4 {
    using slot = mmw_mxfp4_blk;
    static __device__ __forceinline__ void load(const uint8_t * row, const int k, const int h, slot & f) {
        mmw_mxfp4_fetch((const block_mxfp4 *) row + 2 * k + h, f);
    }
    // for e < 252 every weight is (d / 2) * kv exactly (bf16 = the high half), so the block has only 8 magnitudes. Their lo bytes and their hi bytes without and with the sign bit (code 8 is +0) are v_perm tables; larger e keeps the per-weight path.
    static __device__ __forceinline__ void decode(const slot & f, uint32_t (&p)[16]) {
        if (f.e >= 252) {
            mmw_dq_mxfp4_big(f, p);
            return;
        }
        const float dh = ggml_cuda_e8m0_to_fp32(f.e) * 0.5f;
        uint32_t lo[2], hi[2];
        mmw_tab(0u, __float_as_uint(dh), __float_as_uint(dh * 2.0f), __float_as_uint(dh * 3.0f), lo[0], hi[0]);
        mmw_tab(__float_as_uint(dh * 4.0f), __float_as_uint(dh * 6.0f), __float_as_uint(dh * 8.0f), __float_as_uint(dh * 12.0f), lo[1], hi[1]);
        const uint32_t hn0 = hi[0] | 0x80808000u, hn1 = hi[1] | 0x80808080u;
        const uint32_t qs[4] = {f.qs.x, f.qs.y, f.qs.z, f.qs.w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t sel = (qs[w] >> 4 * k) & 0x07070707u;
                const uint32_t l = __builtin_amdgcn_perm(lo[1], lo[0], sel), u = mmw_lut16(hi[0], hi[1], hn0, hn1, sel, qs[w] << (4 - 4 * k));
                p[8 * k + 2 * w]     = __builtin_amdgcn_perm(u, l, 0x05010400u);
                p[8 * k + 2 * w + 1] = __builtin_amdgcn_perm(u, l, 0x07030602u);
            }
        }
    }
};
template <> struct mmw_rq<32 + GGML_TYPE_MXFP4> : mmw_rq_blk<mmw_mxfp4, 2> {};

// the narrow tile multiplies few tokens per weight slice, so it waits on the weight fetches more: Q8_0 and MXFP4 fetch 4 slices there
template <int WTYPE> struct mmw_rq_narrow : mmw_rq<WTYPE> {};
template <> struct mmw_rq_narrow<1> : mmw_rq_blk<mmw_q8_0, 4> {};
template <> struct mmw_rq_narrow<32 + GGML_TYPE_MXFP4> : mmw_rq_blk<mmw_mxfp4, 4> {};

// get_scale_min_k4 of sub-blocks 2s and 2s + 1 (s = slice) from the scale bytes 0..3 (m.y), 4..7 (m.z) and 8..11 (m.w)
__device__ __forceinline__ void mmw_k4_scales(const uint4 m, const int s, uint32_t (&sc)[2], uint32_t (&mn)[2]) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int sh = 16 * (s & 1) + 8 * k;
        if (s < 2) {
            sc[k] = (m.y >> sh) & 63;
            mn[k] = (m.z >> sh) & 63;
        } else {
            sc[k] = ((m.w >> sh) & 15) | ((m.y >> (sh + 6)) & 3) << 4;
            mn[k] = ((m.w >> (sh + 4)) & 15) | ((m.z >> (sh + 6)) & 3) << 4;
        }
    }
}

// Q4_K: half h of a slice is qs bytes 16h..16h+15 of its 32, both nibbles: weights 16h.. (sub-block 2s) and 32 + 16h.. (2s + 1). fetch() loads the header and the 16 qs bytes of all 4 slices of a super-block.
template <> struct mmw_rq<32 + GGML_TYPE_Q4_K> {
    static constexpr bool ok = true; static constexpr int ns = 4;
    struct regs { uint4 m, q[4]; };
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int nks, const int h, regs & f) {
        GGML_UNUSED(nks);
        const block_q4_K * x = (const block_q4_K *) row + ks / 4;
        f.m = *(const uint4 *) x;
#pragma unroll
        for (int s = 0; s < 4; ++s) f.q[s] = *(const uint4 *) (x->qs + 32 * s + 16 * h);
    }
    // d * sc * q - dmin * mn as dequantize_q4_K
    template <int S> static __device__ __forceinline__ void decode(const regs & f, const int h, uint16_t * row) {
        uint32_t sc[2], mn[2];
        mmw_k4_scales(f.m, S, sc, mn);
        const float d = mmw_h2f((uint16_t) f.m.x), dm = mmw_h2f((uint16_t) (f.m.x >> 16));
        const float d0 = d * sc[0], m0 = dm * mn[0], d1 = d * sc[1], m1 = dm * mn[1];
        const uint32_t q[4] = {f.q[S].x, f.q[S].y, f.q[S].z, f.q[S].w};
        uint32_t p[16];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            p[2 * w]         = mmw_pack2(d0 * (q[w] & 15) - m0, d0 * ((q[w] >> 8) & 15) - m0);
            p[2 * w + 1]     = mmw_pack2(d0 * ((q[w] >> 16) & 15) - m0, d0 * ((q[w] >> 24) & 15) - m0);
            p[8 + 2 * w]     = mmw_pack2(d1 * ((q[w] >> 4) & 15) - m1, d1 * ((q[w] >> 12) & 15) - m1);
            p[8 + 2 * w + 1] = mmw_pack2(d1 * ((q[w] >> 20) & 15) - m1, d1 * (q[w] >> 28) - m1);
        }
        *(uint4 *) (row + 16 * h)          = make_uint4(p[0], p[1], p[2], p[3]);
        *(uint4 *) (row + 16 * h + 8)      = make_uint4(p[4], p[5], p[6], p[7]);
        *(uint4 *) (row + 32 + 16 * h)     = make_uint4(p[8], p[9], p[10], p[11]);
        *(uint4 *) (row + 32 + 16 * h + 8) = make_uint4(p[12], p[13], p[14], p[15]);
    }
};

// Q5_K: the Q4_K split plus qh bytes 16h..16h+15 (bit 2s for the low nibbles, 2s + 1 for the high ones)
template <> struct mmw_rq<32 + GGML_TYPE_Q5_K> {
    static constexpr bool ok = true; static constexpr int ns = 4;
    struct regs { uint4 m, qh, q[4]; };
    // the blocks are 176 bytes, so every field is aligned for its load
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int nks, const int h, regs & f) {
        GGML_UNUSED(nks);
        const block_q5_K * x = (const block_q5_K *) row + ks / 4;
        f.m  = *(const uint4 *) x;
        f.qh = *(const uint4 *) (x->qh + 16 * h);
#pragma unroll
        for (int s = 0; s < 4; ++s) f.q[s] = *(const uint4 *) (x->qs + 32 * s + 16 * h);
    }
    // d * sc * q - dmin * mn as dequantize_q5_K
    template <int S> static __device__ __forceinline__ void decode(const regs & f, const int h, uint16_t * row) {
        uint32_t sc[2], mn[2];
        mmw_k4_scales(f.m, S, sc, mn);
        float d[2], dmn[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            d[k] = mmw_h2f((uint16_t) f.m.x) * (float) sc[k];
            dmn[k] = mmw_h2f((uint16_t) (f.m.x >> 16)) * (float) mn[k];
        }
        const uint32_t qs[4] = {f.q[S].x, f.q[S].y, f.q[S].z, f.q[S].w}, qh[4] = {f.qh.x, f.qh.y, f.qh.z, f.qh.w};
        uint32_t p[16];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t q = ((qs[w] >> 4 * k) & 0x0F0F0F0Fu) | ((qh[w] >> (2 * S + k)) & 0x01010101u) << 4;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = d[k] * (float) ((q >> 8 * j) & 0xFF) - dmn[k];
                p[8 * k + 2 * w] = mmw_pack2(v[0], v[1]); p[8 * k + 2 * w + 1] = mmw_pack2(v[2], v[3]);
            }
        }
        *(uint4 *) (row + 16 * h)          = make_uint4(p[0], p[1], p[2], p[3]);
        *(uint4 *) (row + 16 * h + 8)      = make_uint4(p[4], p[5], p[6], p[7]);
        *(uint4 *) (row + 32 + 16 * h)     = make_uint4(p[8], p[9], p[10], p[11]);
        *(uint4 *) (row + 32 + 16 * h + 8) = make_uint4(p[12], p[13], p[14], p[15]);
    }
};

// Q6_K: a 128-weight group is 2 slices; slice weights l and 32 + l (l < 32) use ql bytes l, 32 + l (low nibbles in slice 0, high in 1) and qh byte l. Half h takes weights 16h.. and 32 + 16h.. (3 x 16 bytes per group); fetch() loads both groups and the scales of a super-block.
template <> struct mmw_rq<32 + GGML_TYPE_Q6_K> {
    static constexpr bool ok = true; static constexpr int ns = 4;
    struct regs { uint4 qa[2], qb[2], qh[2], sc; uint32_t d; };
    // the blocks are 210 bytes, so the fields are only 2-byte aligned
    static __device__ __forceinline__ void fetch(const uint8_t * row, const int ks, const int nks, const int h, regs & f) {
        GGML_UNUSED(nks);
        const block_q6_K * x = (const block_q6_K *) row + ks / 4;
#pragma unroll
        for (int ip = 0; ip < 2; ++ip) {
            memcpy(&f.qa[ip], x->ql + 64 * ip + 16 * h, sizeof(f.qa[ip]));
            memcpy(&f.qb[ip], x->ql + 64 * ip + 32 + 16 * h, sizeof(f.qb[ip]));
            memcpy(&f.qh[ip], x->qh + 32 * ip + 16 * h, sizeof(f.qh[ip]));
        }
        memcpy(&f.sc, x->scales, sizeof(f.sc));
        f.d = *(const uint16_t *) &x->d;
    }
    // d * sc * (q - 32) as dequantize_q6_K; slice S is half S & 1 of group S / 2
    template <int S> static __device__ __forceinline__ void decode(const regs & f, const int h, uint16_t * row) {
        constexpr int g = S / 2, half = S & 1;
        const uint32_t scs[4] = {f.sc.x, f.sc.y, f.sc.z, f.sc.w}, scw = scs[S];
        const float d[2] = {mmw_h2f((uint16_t) f.d) * (float) (int8_t) (scw >> 8 * h),
                            mmw_h2f((uint16_t) f.d) * (float) (int8_t) (scw >> (16 + 8 * h))};
        const uint32_t qa[4] = {f.qa[g].x, f.qa[g].y, f.qa[g].z, f.qa[g].w}, qb[4] = {f.qb[g].x, f.qb[g].y, f.qb[g].z, f.qb[g].w},
                       qh[4] = {f.qh[g].x, f.qh[g].y, f.qh[g].z, f.qh[g].w};
        uint32_t p[16];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const uint32_t q = (((k ? qb[w] : qa[w]) >> 4 * half) & 0x0F0F0F0Fu) | (((qh[w] >> (4 * half + 2 * k)) & 0x03030303u) << 4);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = d[k] * (__uint_as_float(__builtin_amdgcn_perm(0x4B000000u, q, 0x07040400u | j)) - 8388640.0f);
                p[8 * k + 2 * w] = mmw_pack2(v[0], v[1]); p[8 * k + 2 * w + 1] = mmw_pack2(v[2], v[3]);
            }
        }
        *(uint4 *) (row + 16 * h)          = make_uint4(p[0], p[1], p[2], p[3]);
        *(uint4 *) (row + 16 * h + 8)      = make_uint4(p[4], p[5], p[6], p[7]);
        *(uint4 *) (row + 32 + 16 * h)     = make_uint4(p[8], p[9], p[10], p[11]);
        *(uint4 *) (row + 32 + 16 * h + 8) = make_uint4(p[12], p[13], p[14], p[15]);
    }
};

// bytes of one weight row for a decoder id: 0 = IQ4_NL, 1 = Q8_0, 32 + ggml_type = the other types
template <int WTYPE>
__host__ __device__ constexpr size_t mmw_row_bytes(int k) {
    if constexpr (WTYPE == 0) return (size_t)(k / 32) * 18;
    else if constexpr (WTYPE == 1) return (size_t)(k / 32) * 34;
    else {
        using traits = ggml_cuda_type_traits<(ggml_type)(WTYPE - 32)>;
        return (size_t)(k / traits::qk) * traits::bs;
    }
}
