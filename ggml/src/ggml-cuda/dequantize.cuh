#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q2_K * x = (const block_q2_K *) vx;
    const float2 dm = __half22float2(x[ib].dm);
    const float dall = dm.x;
    const float dmin = dm.y;

    auto dequantize_one = [&](const int idx) -> float {
        const int n = idx / 128;
        const int r = idx % 128;
        const int g = r / 32;
        const int l = r % 32;
        const int is = 8 * n + l / 16;

        const uint8_t q = x[ib].qs[32 * n + l];
        const uint8_t sc = x[ib].scales[is + 2 * g];
        const float d = dall * (sc & 0xF);
        const float m = dmin * (sc >> 4);

        return d * ((q >> (2 * g)) & 3) - m;
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q3_K * x = (const block_q3_K *) vx;
    const float d_all = x[ib].d;

    auto dequantize_one = [&](const int idx) -> float {
        const int n = idx / 128;
        const int r = idx % 128;
        const int j = r / 32;
        const int l = r % 32;

        const int is0 = l / 16;
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const uint8_t m = 1 << (4 * n + j);

        const int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        const float dl = d_all * (us - 32);
        const uint8_t q = x[ib].qs[32 * n + l];
        const uint8_t h = x[ib].hmask[l];
        const int8_t qv = ((q >> shift) & 3) - ((h & m) ? 0 : 4);

        return dl * qv;
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q4_K * x = (const block_q4_K *) vx;
    const float2 dm = __half22float2(x[ib].dm);
    const float dall = dm.x;
    const float dmin = dm.y;

    auto dequantize_one = [&](const int idx) -> float {
        const int il = idx / 64;
        const int in = idx % 64;
        const int is = 2 * il + (in >= 32 ? 1 : 0);
        const int qsi = 32 * il + (in & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        const float d = dall * sc;
        const float mn = dmin * m;
        const uint8_t q = x[ib].qs[qsi];
        const uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        return d * qv - mn;
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q5_K * x = (const block_q5_K *) vx;
    const float2 dm = __half22float2(x[ib].dm);
    const float dall = dm.x;
    const float dmin = dm.y;

    auto dequantize_one = [&](const int idx) -> float {
        const int il = idx / 64;
        const int in = idx % 64;
        const int is = 2 * il + (in >= 32 ? 1 : 0);
        const int ir = (in & 31) / 2;
        const int iq = in & 1;

        const uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        const uint8_t h = x[ib].qh[2 * ir + iq];
        const uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        const float d = dall * sc;
        const float mn = dmin * m;
        const uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        return (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q6_K * x = (const block_q6_K *) vx;
    const float d = x[ib].d;

    auto dequantize_one = [&](const int idx) -> float {
        const int ip = idx / 128;
        const int in = idx % 128;
        const int il = in & 31;
        const int ig = in / 32;
        const int is = 8 * ip + il / 16;

        const uint8_t ql0 = x[ib].ql[64 * ip + il];
        const uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        const uint8_t qh = x[ib].qh[32 * ip + il];
        const int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        if (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } else if (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } else if (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } else {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        return d * scale * ((int8_t) qv - 32);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const uint16_t * q2 = x[ib].qs + 4 * ib8;
        const uint8_t * aux8 = (const uint8_t *) q2;
        const uint8_t * grid = (const uint8_t *) (iq2xxs_grid + aux8[il]);
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float d = (float) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        return d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const uint16_t * q2 = x[ib].qs + 4 * ib8;
        const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
        const float d = (float) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        return d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq2_s * x = (const block_iq2_s *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        const uint8_t * grid = (const uint8_t *) (iq2s_grid + grid_id);
        const float d = (float) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        return d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const uint8_t * q3 = x[ib].qs + 8 * ib8;
        const uint16_t * gas = (const uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = (float) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        if (j < 4) {
            return d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        return d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq3_s * x = (const block_iq3_s *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const uint8_t * qs = x[ib].qs + 8 * ib8;
        const uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        const uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        const uint8_t * grid1 = (const uint8_t *) (iq3s_grid + grid1_id);
        const uint8_t * grid2 = (const uint8_t *) (iq3s_grid + grid2_id);
        const float d = (float) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        const uint8_t signs = x[ib].signs[4 * ib8 + il];

        if (j < 4) {
            return d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        return d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq1_s * x = (const block_iq1_s *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const float delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        const float d = (float) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        const uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        const uint32_t g = iq1s_grid_gpu[grid_id];
        const int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        return d * (qv + delta);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq1_m * x = (const block_iq1_m *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int il = r / 8;
        const int j = r % 8;

        const uint16_t * sc = (const uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        const int ib16 = 2 * ib8 + il / 2;
        const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        const uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        const float delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        const uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        const uint32_t g = iq1s_grid_gpu[grid_id];
        const int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        return d * (qv + delta);
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq4_nl * x = (const block_iq4_nl *) vx;
    const float d = (float) x[ib].d;

    auto dequantize_one = [&](const int idx) -> float {
        if (idx < 16) {
            return d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        return d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}

static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq4_xs * x = (const block_iq4_xs *) vx;

    auto dequantize_one = [&](const int idx) -> float {
        const int ib8 = idx / 32;
        const int r = idx % 32;
        const int byte_idx = (r < 16) ? r : (r - 16);
        const uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        const uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        const float d = (float) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        return d * kvalues_iq4nl[qv];
    };

    v.x = dequantize_one(iqs + 0);
    v.y = dequantize_one(iqs + 1);
}
