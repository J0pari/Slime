#include <cuda_fp16.h>
#include <mma.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <cfloat>
#ifdef _WIN32
#include <direct.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif
namespace wmma = nvcuda::wmma;

constexpr int G = 28;
constexpr int NC = G * G;
constexpr int CH = 20;
constexpr int CH_STATE = 19;
constexpr int CH_CLS = 10;
constexpr int CH_CLS_OFF = CH - CH_CLS;
constexpr int WM = 16, WN = 16, WK = 16;

constexpr int P_IN_RAW = 9 * CH;
constexpr int P_IN = 192;
constexpr int HID = 80;
constexpr int HID_PAD = 80;
constexpr int U_OUT_PAD = 32;

constexpr int BATCH = 16;
constexpr int POOL = BATCH * 10;
constexpr int STEPS = 20;
constexpr float FIRE = 0.5f;
constexpr float NOISE_STD = 0.02f;
constexpr float ALIVE_TH = 0.1f;

constexpr float LR0 = 1e-3f;
constexpr int LR_STEP1 = 30000;
constexpr int LR_STEP2 = 70000;
constexpr float LR_DECAY = 0.1f;
constexpr float B1 = 0.9f;
constexpr float B2 = 0.999f;
constexpr float EPS = 1e-8f;

constexpr int NTRAIN = 60000;
constexpr int NTEST = 10000;
constexpr int ITERS = 25000;
constexpr int LOG_EVERY = 100;

constexpr int WP_SZ = P_IN * HID_PAD;
constexpr int W1_SZ = HID_PAD * HID_PAD;
constexpr int W2_SZ = HID_PAD * U_OUT_PAD;
constexpr int BP_SZ = HID_PAD;
constexpr int B1_SZ = HID_PAD;
constexpr int B2_SZ = U_OUT_PAD;
constexpr int NTENSORS = 6;
constexpr int BLK = 256;

#define CK(call) do { cudaError_t e = (call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

__host__ float get_lr(int step) {
    float lr = LR0;
    if (step > LR_STEP2) lr *= LR_DECAY * LR_DECAY;
    else if (step > LR_STEP1) lr *= LR_DECAY;
    return lr;
}

static uint32_t rbe32(FILE* f) {
    uint8_t b[4]; fread(b,1,4,f);
    return (uint32_t)b[0]<<24|(uint32_t)b[1]<<16|(uint32_t)b[2]<<8|b[3];
}

static float* load_images(const char* p, int n) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    rbe32(f); rbe32(f); rbe32(f); rbe32(f);
    size_t tot=(size_t)n*NC;
    uint8_t* raw=(uint8_t*)malloc(tot);
    fread(raw,1,tot,f); fclose(f);
    float* h=(float*)malloc(tot*4);
    for(size_t i=0;i<tot;i++) h[i]=raw[i]/255.0f;
    free(raw);
    float* d; CK(cudaMalloc(&d,tot*4));
    CK(cudaMemcpy(d,h,tot*4,cudaMemcpyHostToDevice));
    free(h);
    fprintf(stderr,"Loaded %d images from %s\n",n,p);
    return d;
}

static uint8_t* load_labels(const char* p, int n, uint8_t** hout) {
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    rbe32(f); rbe32(f);
    uint8_t* h=(uint8_t*)malloc(n);
    fread(h,1,n,f); fclose(f);
    uint8_t* d; CK(cudaMalloc(&d,n));
    CK(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice));
    *hout=h;
    fprintf(stderr,"Loaded %d labels from %s\n",n,p);
    return d;
}

struct Weights {
    half *wp, *w1, *w2;
    float *fp_wp, *fp_w1, *fp_w2;
    float *bp, *b1, *b2;
    float *gp, *g1, *g2, *gbp, *gb1, *gb2;
    float *mp, *vp, *m1, *v1, *m2, *v2;
    float *mbp, *vbp, *mb1, *vb1, *mb2, *vb2;
    float *gnorms;
};

static Weights alloc_weights() {
    Weights w;
    CK(cudaMalloc(&w.wp, WP_SZ*2)); CK(cudaMalloc(&w.w1, W1_SZ*2)); CK(cudaMalloc(&w.w2, W2_SZ*2));
    CK(cudaMalloc(&w.fp_wp, WP_SZ*4)); CK(cudaMalloc(&w.fp_w1, W1_SZ*4)); CK(cudaMalloc(&w.fp_w2, W2_SZ*4));
    CK(cudaMalloc(&w.bp, BP_SZ*4)); CK(cudaMalloc(&w.b1, B1_SZ*4)); CK(cudaMalloc(&w.b2, B2_SZ*4));
    CK(cudaMalloc(&w.gp, WP_SZ*4)); CK(cudaMalloc(&w.g1, W1_SZ*4)); CK(cudaMalloc(&w.g2, W2_SZ*4));
    CK(cudaMalloc(&w.gbp, BP_SZ*4)); CK(cudaMalloc(&w.gb1, B1_SZ*4)); CK(cudaMalloc(&w.gb2, B2_SZ*4));
    CK(cudaMalloc(&w.mp, WP_SZ*4)); CK(cudaMalloc(&w.vp, WP_SZ*4));
    CK(cudaMalloc(&w.m1, W1_SZ*4)); CK(cudaMalloc(&w.v1, W1_SZ*4));
    CK(cudaMalloc(&w.m2, W2_SZ*4)); CK(cudaMalloc(&w.v2, W2_SZ*4));
    CK(cudaMalloc(&w.mbp, BP_SZ*4)); CK(cudaMalloc(&w.vbp, BP_SZ*4));
    CK(cudaMalloc(&w.mb1, B1_SZ*4)); CK(cudaMalloc(&w.vb1, B1_SZ*4));
    CK(cudaMalloc(&w.mb2, B2_SZ*4)); CK(cudaMalloc(&w.vb2, B2_SZ*4));
    CK(cudaMalloc(&w.gnorms, NTENSORS*4));
    CK(cudaMemset(w.mp,0,WP_SZ*4)); CK(cudaMemset(w.vp,0,WP_SZ*4));
    CK(cudaMemset(w.m1,0,W1_SZ*4)); CK(cudaMemset(w.v1,0,W1_SZ*4));
    CK(cudaMemset(w.m2,0,W2_SZ*4)); CK(cudaMemset(w.v2,0,W2_SZ*4));
    CK(cudaMemset(w.mbp,0,BP_SZ*4)); CK(cudaMemset(w.vbp,0,BP_SZ*4));
    CK(cudaMemset(w.mb1,0,B1_SZ*4)); CK(cudaMemset(w.vb1,0,B1_SZ*4));
    CK(cudaMemset(w.mb2,0,B2_SZ*4)); CK(cudaMemset(w.vb2,0,B2_SZ*4));
    return w;
}

static void save_weights(const Weights& w) {
    FILE* f = fopen("../weights.bin", "wb");
    if (!f) { fprintf(stderr, "Cannot save weights\n"); return; }
    float* h;
    h = (float*)malloc(WP_SZ * 4);
    CK(cudaMemcpy(h, w.fp_wp, WP_SZ*4, cudaMemcpyDeviceToHost)); fwrite(h, 4, WP_SZ, f);
    CK(cudaMemcpy(h, w.fp_w1, W1_SZ*4, cudaMemcpyDeviceToHost)); fwrite(h, 4, W1_SZ, f);
    CK(cudaMemcpy(h, w.fp_w2, W2_SZ*4, cudaMemcpyDeviceToHost)); fwrite(h, 4, W2_SZ, f);
    free(h);
    h = (float*)malloc(BP_SZ * 4);
    CK(cudaMemcpy(h, w.bp, BP_SZ*4, cudaMemcpyDeviceToHost)); fwrite(h, 4, BP_SZ, f);
    CK(cudaMemcpy(h, w.b1, B1_SZ*4, cudaMemcpyDeviceToHost)); fwrite(h, 4, B1_SZ, f);
    CK(cudaMemcpy(h, w.b2, B2_SZ*4, cudaMemcpyDeviceToHost)); fwrite(h, 4, B2_SZ, f);
    free(h);
    fclose(f);
    fprintf(stderr, "Weights saved to ../weights.bin\n");
}

__global__ void fp32_to_fp16_k(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static bool load_weights(Weights& w) {
    FILE* f = fopen("../weights.bin", "rb");
    if (!f) return false;
    float* h = (float*)malloc(WP_SZ * 4);
    fread(h, 4, WP_SZ, f); CK(cudaMemcpy(w.fp_wp, h, WP_SZ*4, cudaMemcpyHostToDevice));
    fread(h, 4, W1_SZ, f); CK(cudaMemcpy(w.fp_w1, h, W1_SZ*4, cudaMemcpyHostToDevice));
    fread(h, 4, W2_SZ, f); CK(cudaMemcpy(w.fp_w2, h, W2_SZ*4, cudaMemcpyHostToDevice));
    free(h);
    h = (float*)malloc(BP_SZ * 4);
    fread(h, 4, BP_SZ, f); CK(cudaMemcpy(w.bp, h, BP_SZ*4, cudaMemcpyHostToDevice));
    fread(h, 4, B1_SZ, f); CK(cudaMemcpy(w.b1, h, B1_SZ*4, cudaMemcpyHostToDevice));
    fread(h, 4, B2_SZ, f); CK(cudaMemcpy(w.b2, h, B2_SZ*4, cudaMemcpyHostToDevice));
    free(h);
    fclose(f);
    fp32_to_fp16_k<<<(WP_SZ+255)/256, 256>>>(w.fp_wp, w.wp, WP_SZ);
    fp32_to_fp16_k<<<(W1_SZ+255)/256, 256>>>(w.fp_w1, w.w1, W1_SZ);
    fp32_to_fp16_k<<<(W2_SZ+255)/256, 256>>>(w.fp_w2, w.w2, W2_SZ);
    CK(cudaDeviceSynchronize());
    fprintf(stderr, "Weights loaded from ../weights.bin\n");
    return true;
}

__global__ void init_weights_k(half* wp, half* w1, half* w2,
                                float* fp_wp, float* fp_w1, float* fp_w2,
                                float* bp, float* b1, float* b2, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState rng; curand_init(seed, i, 0, &rng);
    float lim_p = sqrtf(6.0f / (float)(P_IN_RAW + HID));
    if (i < WP_SZ) {
        int row = i / HID_PAD;
        float v = row < P_IN_RAW ? (curand_uniform(&rng)*2-1)*lim_p : 0.0f;
        fp_wp[i] = v;
        wp[i] = __float2half(v);
    }
    float lim_1 = sqrtf(6.0f / (float)(HID + HID));
    if (i < W1_SZ) {
        int row = i / HID_PAD;
        float v = row < HID ? (curand_uniform(&rng)*2-1)*lim_1 : 0.0f;
        fp_w1[i] = v;
        w1[i] = __float2half(v);
    }
    if (i < W2_SZ) { fp_w2[i] = 0.0f; w2[i] = __float2half(0.0f); }
    if (i < BP_SZ) bp[i] = 0.0f;
    if (i < B1_SZ) b1[i] = 0.0f;
    if (i < B2_SZ) b2[i] = 0.0f;
}

struct Tape {
    half* im2col;
    half* perc_out;
    half* l1_out;
    uint8_t* mask;
};

static Tape alloc_tape() {
    Tape t;
    size_t sbc = (size_t)STEPS * BATCH * NC;
    CK(cudaMalloc(&t.im2col, sbc * P_IN * 2));
    CK(cudaMalloc(&t.perc_out, sbc * HID_PAD * 2));
    CK(cudaMalloc(&t.l1_out, sbc * HID_PAD * 2));
    CK(cudaMalloc(&t.mask, sbc));
    return t;
}

__global__ void init_rng_k(curandStatePhilox4_32_10_t* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, (unsigned long long)i, 0, &states[i]);
}

__device__ void matmul_wmma(const half* A, const half* B, float* C, int M, int N, int K, int wid, int nw) {
    int tM = M/WM, tN = N/WN, tK = K/WK, tot = tM * tN;
    wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> fb;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> fc;
    for (int t = wid; t < tot; t += nw) {
        int r = (t/tN)*WM, c = (t%tN)*WN;
        wmma::fill_fragment(fc, 0.0f);
        for (int tk = 0; tk < tK; tk++) {
            wmma::load_matrix_sync(fa, A+r*K+tk*WK, K);
            wmma::load_matrix_sync(fb, B+tk*WK*N+c, N);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        wmma::store_matrix_sync(C+r*N+c, fc, N, wmma::mem_row_major);
    }
}

__device__ void matmul_A_BT(const half* A, const half* B, float* C, int M, int N, int K, int wid, int nw) {
    int tM = M/WM, tN = N/WN, tK = K/WK, tot = tM * tN;
    wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::col_major> fb;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> fc;
    for (int t = wid; t < tot; t += nw) {
        int r = (t/tN)*WM, c = (t%tN)*WN;
        wmma::fill_fragment(fc, 0.0f);
        for (int tk = 0; tk < tK; tk++) {
            wmma::load_matrix_sync(fa, A+r*K+tk*WK, K);
            wmma::load_matrix_sync(fb, B+c*K+tk*WK, K);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        wmma::store_matrix_sync(C+r*N+c, fc, N, wmma::mem_row_major);
    }
}

__global__ void im2col_k(const float* __restrict__ state, half* __restrict__ out, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    const float* s = state + bid * NC * CH;
    half* o = out + bid * NC * P_IN;
    for (int cell = threadIdx.x; cell < NC; cell += blockDim.x) {
        int cy = cell / G, cx = cell % G;
        half* dst = o + cell * P_IN;
        int idx = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = cy + dy, nx = cx + dx;
                bool valid = ny >= 0 && ny < G && nx >= 0 && nx < G;
                const float* src = s + (ny * G + nx) * CH;
                for (int c = 0; c < CH; c++) {
                    dst[idx++] = valid ? __float2half(src[c]) : __float2half(0.0f);
                }
            }
        }
        for (int i = P_IN_RAW; i < P_IN; i++) dst[i] = __float2half(0.0f);
    }
}

__global__ void matmul_layer_k(const half* __restrict__ A, const half* __restrict__ B,
                                float* __restrict__ C, int M_total, int N, int K) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int nw_total = (gridDim.x * blockDim.x) / 32;
    int tM = M_total/WM, tN = N/WN, tK = K/WK, tot = tM * tN;
    wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> fb;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> fc;
    for (int t = wid; t < tot; t += nw_total) {
        int r = (t/tN)*WM, c = (t%tN)*WN;
        wmma::fill_fragment(fc, 0.0f);
        for (int tk = 0; tk < tK; tk++) {
            wmma::load_matrix_sync(fa, A+r*K+tk*WK, K);
            wmma::load_matrix_sync(fb, B+tk*WK*N+c, N);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        wmma::store_matrix_sync(C+r*N+c, fc, N, wmma::mem_row_major);
    }
}

__global__ void bias_relu_save_k(float* __restrict__ data, const float* __restrict__ bias,
                                  half* __restrict__ save, int M, int width, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    float* d = data + (size_t)bid * M * width;
    half* s = save + (size_t)bid * M * width;
    for (int i = threadIdx.x; i < M * width; i += blockDim.x) {
        float v = d[i] + bias[i % width];
        s[i] = __float2half(v > 0.0f ? v : 0.0f);
    }
}

__global__ void bias_nosave_k(float* __restrict__ data, const float* __restrict__ bias,
                               int M, int width, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    float* d = data + (size_t)bid * M * width;
    for (int i = threadIdx.x; i < M * width; i += blockDim.x) {
        d[i] += bias[i % width];
    }
}

__global__ void stochastic_update_k(
    const float* __restrict__ in, float* __restrict__ out,
    const float* __restrict__ ds, uint8_t* __restrict__ mask_save,
    curandStatePhilox4_32_10_t* __restrict__ rngs,
    float fire_rate, int batch_size, bool save_mask
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    int tid = threadIdx.x;
    const float* si = in + (size_t)bid * NC * CH;
    float* so = out + (size_t)bid * NC * CH;
    const float* d = ds + (size_t)bid * NC * U_OUT_PAD;
    uint8_t* ms = save_mask ? mask_save + (size_t)bid * NC : nullptr;
    curandStatePhilox4_32_10_t rng = rngs[bid * blockDim.x + tid];
    for (int cell = tid; cell < NC; cell += blockDim.x) {
        float gray = si[cell * CH];
        bool alive = gray > ALIVE_TH;
        bool fires = (curand_uniform(&rng) <= fire_rate) && alive;
        uint8_t m = fires ? 1 : 0;
        if (ms) ms[cell] = m;
        so[cell * CH] = gray;
        float mf = (float)m;
        for (int c = 0; c < CH_STATE; c++) {
            float noise = curand_normal(&rng) * NOISE_STD;
            float delta = (d[cell * U_OUT_PAD + c] + noise) * mf;
            so[cell * CH + 1 + c] = si[cell * CH + 1 + c] + delta;
        }
    }
    rngs[bid * blockDim.x + tid] = rng;
}

static int mm_grid(int M_total, int N) {
    int tiles = (M_total/WM) * (N/WN);
    int warps_per_block = BLK / 32;
    return (tiles + warps_per_block - 1) / warps_per_block;
}

static void forward_one_step(
    Weights& w, Tape& tape, int step,
    float* state_in, float* state_out,
    curandStatePhilox4_32_10_t* rngs,
    float* matmul_scratch, half* fp16_scratch,
    int batch_size, bool save
) {
    int M = batch_size * NC;
    size_t base = (size_t)step * batch_size * NC;
    half* im = save ? tape.im2col + base * P_IN : fp16_scratch;
    im2col_k<<<batch_size, BLK>>>(state_in, im, batch_size);

    matmul_layer_k<<<mm_grid(M, HID_PAD), BLK>>>(im, w.wp, matmul_scratch, M, HID_PAD, P_IN);
    half* po = save ? tape.perc_out + base * HID_PAD : fp16_scratch;
    bias_relu_save_k<<<batch_size, BLK>>>(matmul_scratch, w.bp, po, NC, HID_PAD, batch_size);

    matmul_layer_k<<<mm_grid(M, HID_PAD), BLK>>>(po, w.w1, matmul_scratch, M, HID_PAD, HID_PAD);
    half* l1o = save ? tape.l1_out + base * HID_PAD : fp16_scratch;
    bias_relu_save_k<<<batch_size, BLK>>>(matmul_scratch, w.b1, l1o, NC, HID_PAD, batch_size);

    matmul_layer_k<<<mm_grid(M, U_OUT_PAD), BLK>>>(l1o, w.w2, matmul_scratch, M, U_OUT_PAD, HID_PAD);
    bias_nosave_k<<<batch_size, BLK>>>(matmul_scratch, w.b2, NC, U_OUT_PAD, batch_size);

    uint8_t* ms = save ? tape.mask + base : nullptr;
    stochastic_update_k<<<batch_size, BLK>>>(state_in, state_out, matmul_scratch, ms, rngs, FIRE, batch_size, save);
}

__global__ void loss_grad_k(
    const float* __restrict__ state,
    const uint8_t* __restrict__ labels,
    float* __restrict__ d_state,
    float* __restrict__ loss_out,
    int* __restrict__ correct_out,
    int* __restrict__ total_out,
    int batch_size
) {
    extern __shared__ float smem[];
    float tl = 0.0f;
    int tc = 0, tt = 0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size * NC; idx += gridDim.x * blockDim.x) {
        int b = idx / NC, cell = idx % NC;
        int sb = b * NC * CH + cell * CH;
        float gray = state[sb];
        bool alive = gray > ALIVE_TH;
        int pred = 0; float mx = -1e30f;
        for (int c = 0; c < CH_CLS; c++) {
            int si = sb + CH_CLS_OFF + c;
            float sv = state[si];
            float tgt = (alive && c == (int)labels[b]) ? 1.0f : 0.0f;
            float diff = (alive ? sv : 0.0f) - tgt;
            if (alive) tl += diff * diff * 0.5f;
            d_state[si] = alive ? diff / (float)batch_size : 0.0f;
            if (sv > mx) { mx = sv; pred = c; }
        }
        for (int c = 0; c < CH_CLS_OFF; c++) d_state[sb + c] = 0.0f;
        if (alive) { tt++; if (pred == (int)labels[b]) tc++; }
    }
    smem[threadIdx.x] = tl;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_out, smem[0] / (float)batch_size);
    atomicAdd(correct_out, tc);
    atomicAdd(total_out, tt);
}

__global__ void backward_stochastic_k(
    float* __restrict__ d_state, float* __restrict__ d_ds,
    const uint8_t* __restrict__ mask, int batch_size
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    float* dso = d_state + (size_t)bid * NC * CH;
    float* dd = d_ds + (size_t)bid * NC * U_OUT_PAD;
    const uint8_t* m = mask + (size_t)bid * NC;
    for (int cell = threadIdx.x; cell < NC; cell += blockDim.x) {
        float mf = (float)m[cell];
        for (int c = 0; c < CH_STATE; c++)
            dd[cell * U_OUT_PAD + c] = dso[cell * CH + 1 + c] * mf;
        for (int c = CH_STATE; c < U_OUT_PAD; c++)
            dd[cell * U_OUT_PAD + c] = 0.0f;
        dso[cell * CH] = 0.0f;
    }
}

__global__ void backward_bias_k(const float* __restrict__ d_out, float* __restrict__ d_bias,
                                 int M, int width, int batch_size) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    float s = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        const float* d = d_out + (size_t)b * M * width;
        for (int r = 0; r < M; r++) s += d[r * width + c];
    }
    atomicAdd(&d_bias[c], s);
}

__global__ void drelu_k(float* __restrict__ grad, const half* __restrict__ relu_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __half2float(relu_out[i]) <= 0.0f) grad[i] = 0.0f;
}

__global__ void col2im_k(const float* __restrict__ d_im2col, float* __restrict__ d_state, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    const float* dim = d_im2col + (size_t)bid * NC * P_IN;
    float* ds = d_state + (size_t)bid * NC * CH;
    for (int cell = threadIdx.x; cell < NC; cell += blockDim.x) {
        int cy = cell / G, cx = cell % G;
        for (int c = 1; c < CH; c++) {
            float acc = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = cy - dy, nx = cx - dx;
                    if (ny < 0 || ny >= G || nx < 0 || nx >= G) continue;
                    int src_cell = ny * G + nx;
                    int dir = (dy + 1) * 3 + (dx + 1);
                    acc += dim[src_cell * P_IN + dir * CH + c];
                }
            }
            ds[cell * CH + c] += acc;
        }
    }
}

__global__ void f32_to_f16_k(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

constexpr int GW_SPLITS = 256;

__global__ void matmul_grad_weight_k(const half* __restrict__ act, const half* __restrict__ dout_h,
                                      float* __restrict__ dW, int K, int N, int M_total) {
    extern __shared__ float smem_gw[];
    int split_id = blockIdx.x;
    int wid = threadIdx.x / 32, nw = blockDim.x / 32;
    int tK = K/WM, tN = N/WN, tot = tK * tN;
    int M_tiles = M_total / WK;
    int tiles_per_split = (M_tiles + GW_SPLITS - 1) / GW_SPLITS;
    int tk_start = split_id * tiles_per_split;
    int tk_end = tk_start + tiles_per_split;
    if (tk_end > M_tiles) tk_end = M_tiles;
    wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::col_major> fa;
    wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> fb;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> fc;
    float* warp_buf = smem_gw + wid * WM * WN;
    for (int t = wid; t < tot; t += nw) {
        int tr = t / tN, tc_i = t % tN;
        int r = tr*WM, c = tc_i*WN;
        wmma::fill_fragment(fc, 0.0f);
        for (int tk = tk_start; tk < tk_end; tk++) {
            wmma::load_matrix_sync(fa, act + tk*WK*K + r, K);
            wmma::load_matrix_sync(fb, dout_h + tk*WK*N + c, N);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        wmma::store_matrix_sync(warp_buf, fc, WN, wmma::mem_row_major);
        for (int i = threadIdx.x % 32; i < WM * WN; i += 32)
            atomicAdd(&dW[(r + i/WN)*N + (c + i%WN)], warp_buf[i]);
    }
}

__global__ void matmul_grad_input_k(const half* __restrict__ dout_h, const half* __restrict__ W,
                                     float* __restrict__ din, int M_total, int N, int K) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int nw_total = (gridDim.x * blockDim.x) / 32;
    matmul_A_BT(dout_h, W, din, M_total, N, K, wid, nw_total);
}

static void backward_one_step(
    Weights& w, Tape& tape, int step,
    float* d_state,
    float* d_ds, half* h_scratch, float* f_scratch, float* f_scratch2, float* d_im2col,
    int batch_size
) {
    size_t base = (size_t)step * batch_size * NC;
    int M = batch_size * NC;
    int nw = BLK / 32;
    size_t gw_smem = nw * WM * WN * sizeof(float);

    backward_stochastic_k<<<batch_size, BLK>>>(d_state, d_ds, tape.mask + base, batch_size);

    int n_ds = M * U_OUT_PAD;
    f32_to_f16_k<<<(n_ds+255)/256, 256>>>((const float*)d_ds, h_scratch, n_ds);

    backward_bias_k<<<(B2_SZ+255)/256, 256>>>(d_ds, w.gb2, NC, U_OUT_PAD, batch_size);
    matmul_grad_weight_k<<<GW_SPLITS, BLK, gw_smem>>>(tape.l1_out + base*HID_PAD, h_scratch, w.g2, HID_PAD, U_OUT_PAD, M);
    matmul_grad_input_k<<<mm_grid(M, HID_PAD), BLK>>>(h_scratch, w.w2, f_scratch, M, HID_PAD, U_OUT_PAD);

    int nh = M * HID_PAD;
    drelu_k<<<(nh+255)/256, 256>>>(f_scratch, tape.l1_out + base*HID_PAD, nh);

    f32_to_f16_k<<<(nh+255)/256, 256>>>(f_scratch, h_scratch, nh);

    backward_bias_k<<<(B1_SZ+255)/256, 256>>>(f_scratch, w.gb1, NC, HID_PAD, batch_size);
    matmul_grad_weight_k<<<GW_SPLITS, BLK, gw_smem>>>(tape.perc_out + base*HID_PAD, h_scratch, w.g1, HID_PAD, HID_PAD, M);
    matmul_grad_input_k<<<mm_grid(M, HID_PAD), BLK>>>(h_scratch, w.w1, f_scratch2, M, HID_PAD, HID_PAD);

    drelu_k<<<(nh+255)/256, 256>>>(f_scratch2, tape.perc_out + base*HID_PAD, nh);

    f32_to_f16_k<<<(nh+255)/256, 256>>>(f_scratch2, h_scratch, nh);

    backward_bias_k<<<(BP_SZ+255)/256, 256>>>(f_scratch2, w.gbp, NC, HID_PAD, batch_size);
    matmul_grad_weight_k<<<GW_SPLITS, BLK, gw_smem>>>(tape.im2col + base*P_IN, h_scratch, w.gp, P_IN, HID_PAD, M);
    matmul_grad_input_k<<<mm_grid(M, P_IN), BLK>>>(h_scratch, w.wp, d_im2col, M, P_IN, HID_PAD);

    col2im_k<<<batch_size, BLK>>>(d_im2col, d_state, batch_size);
}

__global__ void grad_norm_k(const float* __restrict__ g, float* __restrict__ norm_out, int n) {
    extern __shared__ float smem[];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = g[i]; s += v*v; }
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int s2 = blockDim.x/2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) smem[threadIdx.x] += smem[threadIdx.x + s2];
        __syncthreads();
    }
    if (threadIdx.x == 0) *norm_out = sqrtf(smem[0]);
}

__global__ void adam_fp16_k(half* __restrict__ w, float* __restrict__ fp_w,
                             float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v,
                             const float* __restrict__ norm, int n,
                             float lr, float b1t, float b2t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sc = 1.0f / (*norm + EPS);
    float gi = g[i] * sc;
    float mi = B1 * m[i] + (1-B1) * gi;
    float vi = B2 * v[i] + (1-B2) * gi * gi;
    m[i] = mi; v[i] = vi;
    float wf = fp_w[i];
    wf -= lr * (mi/(1-b1t)) / (sqrtf(vi/(1-b2t)) + EPS);
    fp_w[i] = wf;
    w[i] = __float2half(wf);
    g[i] = 0.0f;
}

__global__ void adam_fp32_k(float* __restrict__ w, float* __restrict__ g,
                             float* __restrict__ m, float* __restrict__ v,
                             const float* __restrict__ norm, int n,
                             float lr, float b1t, float b2t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sc = 1.0f / (*norm + EPS);
    float gi = g[i] * sc;
    float mi = B1 * m[i] + (1-B1) * gi;
    float vi = B2 * v[i] + (1-B2) * gi * gi;
    m[i] = mi; v[i] = vi;
    w[i] -= lr * (mi/(1-b1t)) / (sqrtf(vi/(1-b2t)) + EPS);
    g[i] = 0.0f;
}

static void optim_step(Weights& w, float lr, int t) {
    float b1t = powf(B1, (float)t);
    float b2t = powf(B2, (float)t);
    int thr = 256;
    grad_norm_k<<<1, thr, thr*4>>>(w.gp, w.gnorms+0, WP_SZ);
    grad_norm_k<<<1, thr, thr*4>>>(w.g1, w.gnorms+1, W1_SZ);
    grad_norm_k<<<1, thr, thr*4>>>(w.g2, w.gnorms+2, W2_SZ);
    grad_norm_k<<<1, thr, thr*4>>>(w.gbp, w.gnorms+3, BP_SZ);
    grad_norm_k<<<1, thr, thr*4>>>(w.gb1, w.gnorms+4, B1_SZ);
    grad_norm_k<<<1, thr, thr*4>>>(w.gb2, w.gnorms+5, B2_SZ);
    adam_fp16_k<<<(WP_SZ+thr-1)/thr, thr>>>(w.wp, w.fp_wp, w.gp, w.mp, w.vp, w.gnorms+0, WP_SZ, lr, b1t, b2t);
    adam_fp16_k<<<(W1_SZ+thr-1)/thr, thr>>>(w.w1, w.fp_w1, w.g1, w.m1, w.v1, w.gnorms+1, W1_SZ, lr, b1t, b2t);
    adam_fp16_k<<<(W2_SZ+thr-1)/thr, thr>>>(w.w2, w.fp_w2, w.g2, w.m2, w.v2, w.gnorms+2, W2_SZ, lr, b1t, b2t);
    adam_fp32_k<<<(BP_SZ+thr-1)/thr, thr>>>(w.bp, w.gbp, w.mbp, w.vbp, w.gnorms+3, BP_SZ, lr, b1t, b2t);
    adam_fp32_k<<<(B1_SZ+thr-1)/thr, thr>>>(w.b1, w.gb1, w.mb1, w.vb1, w.gnorms+4, B1_SZ, lr, b1t, b2t);
    adam_fp32_k<<<(B2_SZ+thr-1)/thr, thr>>>(w.b2, w.gb2, w.mb2, w.vb2, w.gnorms+5, B2_SZ, lr, b1t, b2t);
}

__global__ void init_state_k(float* __restrict__ state, const float* __restrict__ images,
                              const int* __restrict__ indices, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * NC * CH) return;
    int e = idx / (NC * CH), rem = idx % (NC * CH);
    int cell = rem / CH, ch = rem % CH;
    state[e * NC * CH + cell * CH + ch] = (ch == 0) ? images[indices[e] * NC + cell] : 0.0f;
}

struct Pool {
    float* states;
    int h_labels[POOL];
    int h_imgs[POOL];
};

static void init_pool(Pool& pool, const float* d_images, const uint8_t* h_labels) {
    CK(cudaMalloc(&pool.states, (size_t)POOL * NC * CH * 4));
    int h_idx[POOL];
    for (int i = 0; i < POOL; i++) {
        h_idx[i] = rand() % NTRAIN;
        pool.h_labels[i] = h_labels[h_idx[i]];
        pool.h_imgs[i] = h_idx[i];
    }
    int* d_idx; CK(cudaMalloc(&d_idx, POOL*4));
    CK(cudaMemcpy(d_idx, h_idx, POOL*4, cudaMemcpyHostToDevice));
    init_state_k<<<(POOL*NC*CH+255)/256, 256>>>(pool.states, d_images, d_idx, POOL);
    CK(cudaDeviceSynchronize());
    cudaFree(d_idx);
}

__global__ void copy_pool_batch_k(float* __restrict__ batch, const float* __restrict__ pool,
                                   const int* __restrict__ indices, int bs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * NC * CH) return;
    int b = idx / (NC * CH);
    batch[idx] = pool[(size_t)indices[b] * NC * CH + idx % (NC * CH)];
}

__global__ void commit_batch_k(float* __restrict__ pool, const float* __restrict__ batch,
                                const int* __restrict__ indices, int bs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * NC * CH) return;
    int b = idx / (NC * CH);
    pool[(size_t)indices[b] * NC * CH + idx % (NC * CH)] = batch[idx];
}

__global__ void fresh_init_one_k(float* __restrict__ batch, const float* __restrict__ images,
                                  int img_idx, int batch_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NC * CH) return;
    int cell = idx / CH, ch = idx % CH;
    batch[(size_t)batch_pos * NC * CH + idx] = (ch == 0) ? images[img_idx * NC + cell] : 0.0f;
}

__global__ void mutate_one_k(float* __restrict__ batch, const float* __restrict__ images,
                              int img_idx, int batch_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NC * CH) return;
    int cell = idx / CH, ch = idx % CH;
    size_t off = (size_t)batch_pos * NC * CH + idx;
    float new_gray = images[img_idx * NC + cell];
    if (ch == 0) batch[off] = new_gray;
    else if (new_gray <= ALIVE_TH) batch[off] = 0.0f;
}

static void sample_batch(Pool& pool, float* d_batch, uint8_t* h_batch_labels,
                          int* h_pool_idx, const float* d_images, const uint8_t* h_labels,
                          int* d_batch_idx, uint8_t* d_blabels) {
    int deck[POOL];
    for (int i = 0; i < POOL; i++) deck[i] = i;
    for (int i = 0; i < BATCH; i++) {
        int j = i + rand() % (POOL - i);
        int t = deck[i]; deck[i] = deck[j]; deck[j] = t;
        h_pool_idx[i] = deck[i];
    }
    for (int i = 0; i < BATCH; i++)
        h_batch_labels[i] = pool.h_labels[h_pool_idx[i]];

    int q_bs = BATCH / 4;
    for (int i = 0; i < q_bs; i++) {
        int new_img = rand() % NTRAIN;
        pool.h_labels[h_pool_idx[i]] = h_labels[new_img];
        pool.h_imgs[h_pool_idx[i]] = new_img;
        h_batch_labels[i] = h_labels[new_img];
    }
    for (int i = 0; i < q_bs; i++) {
        int pos = BATCH - 1 - i;
        int new_img = rand() % NTRAIN;
        pool.h_labels[h_pool_idx[pos]] = h_labels[new_img];
        pool.h_imgs[h_pool_idx[pos]] = new_img;
        h_batch_labels[pos] = h_labels[new_img];
    }

    CK(cudaMemcpyAsync(d_batch_idx, h_pool_idx, BATCH*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(d_blabels, h_batch_labels, BATCH, cudaMemcpyHostToDevice));
    copy_pool_batch_k<<<(BATCH*NC*CH+255)/256, 256>>>(d_batch, pool.states, d_batch_idx, BATCH);

    for (int i = 0; i < q_bs; i++)
        fresh_init_one_k<<<(NC*CH+255)/256, 256>>>(d_batch, d_images, pool.h_imgs[h_pool_idx[i]], i);
    for (int i = 0; i < q_bs; i++) {
        int pos = BATCH - 1 - i;
        mutate_one_k<<<(NC*CH+255)/256, 256>>>(d_batch, d_images, pool.h_imgs[h_pool_idx[pos]], pos);
    }
}

static void commit_batch(Pool& pool, const float* d_batch, int* d_batch_idx) {
    commit_batch_k<<<(BATCH*NC*CH+255)/256, 256>>>(pool.states, d_batch, d_batch_idx, BATCH);
}

constexpr int EVAL_STEPS = 200;
constexpr int EVAL_VIS_SAMPLES = 32;
constexpr int EVAL_VIS_STEPS = 64;

static void dump_eval_snapshots(Weights& w, Tape& tape,
                                float* buf_a, float* buf_b,
                                float* matmul_scratch, half* fp16_scratch,
                                curandStatePhilox4_32_10_t* rngs,
                                const float* test_images, const uint8_t* h_test_labels,
                                const char* ts) {
    float* h_state = (float*)malloc((size_t)EVAL_VIS_SAMPLES * NC * CH * 4);
    int* d_idx; CK(cudaMalloc(&d_idx, EVAL_VIS_SAMPLES * 4));

    int h_idx[EVAL_VIS_SAMPLES];
    uint8_t h_lbl[EVAL_VIS_SAMPLES];
    for (int i = 0; i < EVAL_VIS_SAMPLES; i++) {
        h_idx[i] = i;
        h_lbl[i] = h_test_labels[i];
    }
    CK(cudaMemcpy(d_idx, h_idx, EVAL_VIS_SAMPLES * 4, cudaMemcpyHostToDevice));

    int leftover = EVAL_VIS_SAMPLES % BATCH;

    char bin_path[256];
    snprintf(bin_path, sizeof(bin_path), "../telemetry/snapshot_%s.bin", ts);
    FILE* f = fopen(bin_path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", bin_path); free(h_state); cudaFree(d_idx); return; }

    int32_t header[3] = {EVAL_VIS_SAMPLES, EVAL_VIS_STEPS, NC * CH};
    fwrite(header, 4, 3, f);
    fwrite(h_lbl, 1, EVAL_VIS_SAMPLES, f);

    float* h_gray = (float*)malloc(EVAL_VIS_SAMPLES * NC * 4);
    for (int i = 0; i < EVAL_VIS_SAMPLES; i++) {
        float h_img[NC];
        CK(cudaMemcpy(h_img, test_images + (size_t)i * NC, NC * 4, cudaMemcpyDeviceToHost));
        memcpy(h_gray + i * NC, h_img, NC * 4);
    }
    fwrite(h_gray, 4, EVAL_VIS_SAMPLES * NC, f);
    free(h_gray);

    for (int batch_start = 0; batch_start < EVAL_VIS_SAMPLES; batch_start += BATCH) {
        int bs = (batch_start + BATCH <= EVAL_VIS_SAMPLES) ? BATCH : leftover;
        if (bs == 0) break;

        int b_idx[BATCH];
        for (int i = 0; i < bs; i++) b_idx[i] = batch_start + i;
        for (int i = bs; i < BATCH; i++) b_idx[i] = 0;
        CK(cudaMemcpy(d_idx, b_idx, BATCH * 4, cudaMemcpyHostToDevice));
        init_state_k<<<(BATCH*NC*CH+255)/256, 256>>>(buf_a, test_images, d_idx, BATCH);
        CK(cudaDeviceSynchronize());
        float* in = buf_a, *out = buf_b;

        for (int step = 0; step < EVAL_VIS_STEPS; step++) {
            forward_one_step(w, tape, step % STEPS, in, out, rngs, matmul_scratch, fp16_scratch, BATCH, false);
            float* t = in; in = out; out = t;

            CK(cudaMemcpy(h_state, in, (size_t)BATCH * NC * CH * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < bs; i++) {
                fwrite(h_state + (size_t)i * NC * CH, 4, NC * CH, f);
            }
        }
    }

    fclose(f);
    free(h_state);
    cudaFree(d_idx);
    fprintf(stderr, "  Snapshot dump -> %s (%d samples, %d steps)\n", bin_path, EVAL_VIS_SAMPLES, EVAL_VIS_STEPS);
}

static void eval_telemetry(Weights& w, Tape& tape,
                            float* buf_a, float* buf_b,
                            float* matmul_scratch, half* fp16_scratch,
                            curandStatePhilox4_32_10_t* rngs,
                            const float* test_images, const uint8_t* h_test_labels,
                            int num_samples, int iter,
                            FILE* f_csv, FILE* f_summary) {

    int bs = BATCH;
    int nbatch = num_samples / bs;
    float* h_eval = (float*)malloc((size_t)bs * NC * CH * 4);

    int alive_counts[EVAL_STEPS] = {0};
    int correct_counts[EVAL_STEPS] = {0};
    int agreement_counts[EVAL_STEPS] = {0};
    int sample_counts[EVAL_STEPS] = {0};

    int* d_eval_idx; CK(cudaMalloc(&d_eval_idx, bs*4));
    for (int eb = 0; eb < nbatch; eb++) {
        int h_idx[BATCH];
        uint8_t h_lbl[BATCH];
        for (int i = 0; i < bs; i++) {
            h_idx[i] = eb * bs + i;
            h_lbl[i] = h_test_labels[h_idx[i]];
        }
        CK(cudaMemcpy(d_eval_idx, h_idx, bs*4, cudaMemcpyHostToDevice));
        init_state_k<<<(bs*NC*CH+255)/256, 256>>>(buf_a, test_images, d_eval_idx, bs);
        CK(cudaDeviceSynchronize());
        float* in = buf_a, *out = buf_b;

        for (int step = 0; step < EVAL_STEPS; step++) {
            forward_one_step(w, tape, step % STEPS, in, out, rngs, matmul_scratch, fp16_scratch, bs, false);
            float* t = in; in = out; out = t;

            CK(cudaMemcpy(h_eval, in, (size_t)bs * NC * CH * 4, cudaMemcpyDeviceToHost));
            for (int b = 0; b < bs; b++) {
                int label = h_lbl[b];
                int s_alive = 0, s_corr = 0;
                int class_votes[CH_CLS] = {0};
                for (int cell = 0; cell < NC; cell++) {
                    float gray = h_eval[b * NC * CH + cell * CH];
                    if (gray <= ALIVE_TH) continue;
                    s_alive++;
                    int pred = 0; float mx = -1e30f;
                    for (int c = 0; c < CH_CLS; c++) {
                        float v = h_eval[b * NC * CH + cell * CH + CH_CLS_OFF + c];
                        if (v > mx) { mx = v; pred = c; }
                    }
                    if (pred == label) s_corr++;
                    class_votes[pred]++;
                }
                alive_counts[step] += s_alive;
                correct_counts[step] += s_corr;
                if (s_alive > 0) {
                    sample_counts[step]++;
                    int max_votes = 0;
                    for (int c = 0; c < CH_CLS; c++) if (class_votes[c] > max_votes) max_votes = class_votes[c];
                    int total_votes = s_alive;
                    if (max_votes == total_votes) agreement_counts[step]++;
                }
            }
        }
    }

    float top_acc = 0; int top_step = 0;
    float top_agr = 0; int top_agr_step = 0;
    for (int s = 0; s < EVAL_STEPS; s++) {
        float acc = alive_counts[s] > 0 ? 100.0f * correct_counts[s] / alive_counts[s] : 0.0f;
        float agr = sample_counts[s] > 0 ? 100.0f * agreement_counts[s] / sample_counts[s] : 0.0f;
        fprintf(f_csv, "%d,%d,%.4f,%.4f,%d,%d,%d,%d\n",
                iter, s+1, acc, agr, correct_counts[s], alive_counts[s], agreement_counts[s], sample_counts[s]);
        if (acc > top_acc) { top_acc = acc; top_step = s+1; }
        if (agr > top_agr) { top_agr = agr; top_agr_step = s+1; }
    }
    fflush(f_csv);

    float acc_200 = alive_counts[EVAL_STEPS-1] > 0 ? 100.0f * correct_counts[EVAL_STEPS-1] / alive_counts[EVAL_STEPS-1] : 0.0f;
    float agr_200 = sample_counts[EVAL_STEPS-1] > 0 ? 100.0f * agreement_counts[EVAL_STEPS-1] / sample_counts[EVAL_STEPS-1] : 0.0f;

    fprintf(f_summary, "iter: %d\n", iter);
    fprintf(f_summary, "num_samples: %d\n", num_samples);
    fprintf(f_summary, "top_accuracy: %.2f at %d\n", top_acc, top_step);
    fprintf(f_summary, "accuracy_at_200: %.2f\n", acc_200);
    fprintf(f_summary, "top_agreement: %.2f at %d\n", top_agr, top_agr_step);
    fprintf(f_summary, "agreement_at_200: %.2f\n", agr_200);
    fprintf(f_summary, "\n");
    fflush(f_summary);

    cudaFree(d_eval_idx);
    free(h_eval);
    fprintf(stderr, "  EVAL (%d samples): top_acc=%.2f%%@%d acc200=%.2f%% top_agr=%.2f%%@%d agr200=%.2f%%\n",
            num_samples, top_acc, top_step, acc_200, top_agr, top_agr_step, agr_200);
}

static void sha1(const uint8_t* msg, int len, uint8_t out[20]) {
    uint32_t h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
    uint64_t bits = (uint64_t)len * 8;
    int padded = ((len + 8) / 64 + 1) * 64;
    uint8_t* buf = (uint8_t*)calloc(padded, 1);
    memcpy(buf, msg, len);
    buf[len] = 0x80;
    for (int i = 0; i < 8; i++) buf[padded - 1 - i] = (uint8_t)(bits >> (i * 8));
    for (int blk = 0; blk < padded; blk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)buf[blk+i*4]<<24|(uint32_t)buf[blk+i*4+1]<<16|
                   (uint32_t)buf[blk+i*4+2]<<8|buf[blk+i*4+3];
        for (int i = 16; i < 80; i++) {
            uint32_t v = w[i-3]^w[i-8]^w[i-14]^w[i-16];
            w[i] = (v<<1)|(v>>31);
        }
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f,k;
            if (i<20)      { f=(b&c)|((~b)&d); k=0x5A827999; }
            else if (i<40) { f=b^c^d;           k=0x6ED9EBA1; }
            else if (i<60) { f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
            else           { f=b^c^d;           k=0xCA62C1D6; }
            uint32_t t = ((a<<5)|(a>>27)) + f + e + k + w[i];
            e=d; d=c; c=(b<<30)|(b>>2); b=a; a=t;
        }
        h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
    }
    free(buf);
    uint32_t hh[5] = {h0,h1,h2,h3,h4};
    for (int i = 0; i < 5; i++) {
        out[i*4]=(uint8_t)(hh[i]>>24); out[i*4+1]=(uint8_t)(hh[i]>>16);
        out[i*4+2]=(uint8_t)(hh[i]>>8); out[i*4+3]=(uint8_t)hh[i];
    }
}

static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void base64_encode(const uint8_t* in, int len, char* out) {
    int o = 0;
    for (int i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)in[i] << 16;
        if (i+1 < len) v |= (uint32_t)in[i+1] << 8;
        if (i+2 < len) v |= in[i+2];
        out[o++] = b64[(v>>18)&63];
        out[o++] = b64[(v>>12)&63];
        out[o++] = (i+1 < len) ? b64[(v>>6)&63] : '=';
        out[o++] = (i+2 < len) ? b64[v&63] : '=';
    }
    out[o] = 0;
}

static bool ws_handshake(SOCKET client) {
    char buf[4096];
    int n = recv(client, buf, sizeof(buf)-1, 0);
    if (n <= 0) return false;
    buf[n] = 0;
    char* key_start = strstr(buf, "Sec-WebSocket-Key: ");
    if (!key_start) return false;
    key_start += 19;
    char* key_end = strstr(key_start, "\r\n");
    if (!key_end) return false;
    char combined[256];
    int klen = (int)(key_end - key_start);
    memcpy(combined, key_start, klen);
    memcpy(combined + klen, "258EAFA5-E914-47DA-95CA-C5AB0DC85B11", 36);
    combined[klen + 36] = 0;
    uint8_t hash[20];
    sha1((uint8_t*)combined, klen + 36, hash);
    char accept[32];
    base64_encode(hash, 20, accept);
    char resp[512];
    int rlen = snprintf(resp, sizeof(resp),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n", accept);
    send(client, resp, rlen, 0);
    return true;
}

static int recv_exact(SOCKET s, char* buf, int len) {
    int got = 0;
    while (got < len) {
        int r = recv(s, buf + got, len - got, 0);
        if (r > 0) { got += r; continue; }
        if (r == 0) return -2;
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK) {
            if (got == 0) return -1;
            Sleep(1);
            continue;
        }
        return -2;
    }
    return got;
}

static int ws_recv(SOCKET s, uint8_t* out, int max_len) {
    uint8_t hdr[2];
    int r = recv_exact(s, (char*)hdr, 2);
    if (r == -1) return -1;
    if (r < 0) return 0;
    int len = hdr[1] & 0x7F;
    bool masked = (hdr[1] & 0x80) != 0;
    if (len == 126) {
        uint8_t ext[2];
        if (recv_exact(s, (char*)ext, 2) != 2) return 0;
        len = (ext[0] << 8) | ext[1];
    } else if (len == 127) {
        uint8_t ext[8];
        if (recv_exact(s, (char*)ext, 8) != 8) return 0;
        len = 0;
        for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
    }
    uint8_t mask[4] = {0,0,0,0};
    if (masked) {
        if (recv_exact(s, (char*)mask, 4) != 4) return 0;
    }
    if (len > max_len) return 0;
    if (recv_exact(s, (char*)out, len) != len) return 0;
    if (masked) for (int i = 0; i < len; i++) out[i] ^= mask[i % 4];
    return len;
}

static bool ws_send_bin(SOCKET s, const uint8_t* data, int len) {
    uint8_t hdr[10];
    int hlen = 0;
    hdr[hlen++] = 0x82;
    if (len < 126) {
        hdr[hlen++] = (uint8_t)len;
    } else if (len < 65536) {
        hdr[hlen++] = 126;
        hdr[hlen++] = (uint8_t)(len >> 8);
        hdr[hlen++] = (uint8_t)(len & 0xFF);
    } else {
        hdr[hlen++] = 127;
        for (int i = 7; i >= 0; i--) hdr[hlen++] = (uint8_t)((len >> (i*8)) & 0xFF);
    }
    if (send(s, (char*)hdr, hlen, 0) != hlen) return false;
    int sent = 0;
    while (sent < len) {
        int r = send(s, (char*)data + sent, len - sent, 0);
        if (r <= 0) return false;
        sent += r;
    }
    return true;
}

static void serve_inference(Weights& w, Tape& tape,
                       float* buf_a, float* buf_b,
                       float* matmul_scratch, half* fp16_scratch,
                       curandStatePhilox4_32_10_t* rngs) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);

    SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8765);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "bind failed\n"); return;
    }
    listen(srv, 1);
    fprintf(stderr, "\n=== INFERENCE SERVER on ws://localhost:8765 ===\n");

    float* h_state = (float*)malloc(NC * CH * 4);
    float* h_pixels = (float*)malloc(NC * 4);
    uint8_t* ws_buf = (uint8_t*)malloc(NC * CH * 4 + 16);
    int* d_idx;
    CK(cudaMalloc(&d_idx, 4));
    int zero = 0;
    CK(cudaMemcpy(d_idx, &zero, 4, cudaMemcpyHostToDevice));

    for (;;) {
        fprintf(stderr, "Waiting for connection...\n");
        SOCKET client = accept(srv, NULL, NULL);
        if (client == INVALID_SOCKET) continue;
        fprintf(stderr, "Client connected, handshaking...\n");
        if (!ws_handshake(client)) { closesocket(client); continue; }
        fprintf(stderr, "WebSocket established\n");

        u_long nonblock = 1;
        ioctlsocket(client, FIONBIO, &nonblock);

        bool have_image = false;
        float* in_buf = buf_a;
        float* out_buf = buf_b;
        int step = 0;

        for (;;) {
            int n = ws_recv(client, ws_buf, NC * 4 + 16);
            if (n == NC * 4) {
                float* new_px = (float*)ws_buf;
                if (have_image) {
                    CK(cudaMemcpy(h_state, in_buf, NC * CH * 4, cudaMemcpyDeviceToHost));
                    bool changed[NC];
                    for (int i = 0; i < NC; i++)
                        changed[i] = (new_px[i] != h_pixels[i]);
                    int labels[NC];
                    for (int i = 0; i < NC; i++) labels[i] = -1;
                    int queue[NC], qh, qt;
                    bool dirty_comp[NC];
                    int nlbl = 0;
                    for (int seed = 0; seed < NC; seed++) {
                        if (labels[seed] >= 0 || !(new_px[seed] > ALIVE_TH)) continue;
                        int lbl = nlbl++;
                        bool has_change = false;
                        labels[seed] = lbl;
                        queue[0] = seed; qh = 0; qt = 1;
                        while (qh < qt) {
                            int cur = queue[qh++];
                            if (changed[cur]) has_change = true;
                            int cx = cur % G, cy = cur / G;
                            for (int dy = -1; dy <= 1; dy++)
                                for (int dx = -1; dx <= 1; dx++) {
                                    if (dx == 0 && dy == 0) continue;
                                    int nx = cx + dx, ny = cy + dy;
                                    if (nx < 0 || nx >= G || ny < 0 || ny >= G) continue;
                                    int ni = ny * G + nx;
                                    if (labels[ni] < 0 && new_px[ni] > ALIVE_TH) {
                                        labels[ni] = lbl;
                                        queue[qt++] = ni;
                                    }
                                }
                        }
                        dirty_comp[lbl] = has_change;
                    }
                    for (int cell = 0; cell < NC; cell++) {
                        h_state[cell * CH] = new_px[cell];
                        if (labels[cell] >= 0 && dirty_comp[labels[cell]])
                            for (int c = 1; c < CH; c++) h_state[cell * CH + c] = 0.0f;
                        else if (!(new_px[cell] > ALIVE_TH))
                            for (int c = 1; c < CH; c++) h_state[cell * CH + c] = 0.0f;
                    }
                } else {
                    for (int cell = 0; cell < NC; cell++) {
                        h_state[cell * CH] = new_px[cell];
                        for (int c = 1; c < CH; c++) h_state[cell * CH + c] = 0.0f;
                    }
                }
                memcpy(h_pixels, new_px, NC * 4);
                CK(cudaMemcpy(in_buf, h_state, NC * CH * 4, cudaMemcpyHostToDevice));
                have_image = true;
                step = 0;
            } else if (n == -1) {
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK) break;
            } else if (n == 0) {
                break;
            }

            if (have_image) {
                forward_one_step(w, tape, step % STEPS, in_buf, out_buf, rngs,
                                 matmul_scratch, fp16_scratch, 1, false);
                float* t = in_buf; in_buf = out_buf; out_buf = t;
                step++;
                CK(cudaMemcpy(h_state, in_buf, NC * CH * 4, cudaMemcpyDeviceToHost));
                if (!ws_send_bin(client, (uint8_t*)h_state, NC * CH * 4)) break;
            } else {
                Sleep(10);
            }
        }
        fprintf(stderr, "Client disconnected\n");
        closesocket(client);
    }

    free(h_state); free(h_pixels); free(ws_buf);
    cudaFree(d_idx);
    closesocket(srv);
    WSACleanup();
}

int main(int argc, char** argv) {
    bool server_only = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--server") == 0) server_only = true;
    }
    srand((unsigned)time(NULL));
    FILE* f_csv = NULL; FILE* f_summary = NULL; FILE* f_train = NULL;
    char ts[64] = {0};
    uint8_t *h_train_labels = NULL, *h_test_labels = NULL;
    float *d_train_images = NULL, *d_test_images = NULL;
    uint8_t *d_train_labels = NULL, *d_test_labels = NULL;

    if (!server_only) {
#ifdef _WIN32
        _mkdir("../telemetry");
#else
        mkdir("../telemetry", 0755);
#endif
        time_t now = time(NULL);
        struct tm* t = localtime(&now);
        strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", t);
        char csv_path[256], summary_path[256], train_csv_path[256];
        snprintf(csv_path, sizeof(csv_path), "../telemetry/eval_%s.csv", ts);
        snprintf(summary_path, sizeof(summary_path), "../telemetry/summary_%s.txt", ts);
        snprintf(train_csv_path, sizeof(train_csv_path), "../telemetry/train_%s.csv", ts);
        f_csv = fopen(csv_path, "w");
        f_summary = fopen(summary_path, "w");
        f_train = fopen(train_csv_path, "w");
        if (!f_csv || !f_summary || !f_train) { fprintf(stderr, "Cannot open telemetry files\n"); return 1; }
        fprintf(f_csv, "iter,step,cell_acc,agreement,correct,alive,agreed_samples,total_samples\n");
        fprintf(f_train, "iter,loss,lr\n");
        fprintf(stderr, "Telemetry -> %s, %s, %s\n", csv_path, summary_path, train_csv_path);
    }
    if (!server_only) {
        fprintf(stderr, "Loading MNIST...\n");
        d_train_images = load_images("../data/vision/mnist/train-images-idx3-ubyte", NTRAIN);
        d_train_labels = load_labels("../data/vision/mnist/train-labels-idx1-ubyte", NTRAIN, &h_train_labels);
        d_test_images = load_images("../data/vision/mnist/t10k-images-idx3-ubyte", NTEST);
        d_test_labels = load_labels("../data/vision/mnist/t10k-labels-idx1-ubyte", NTEST, &h_test_labels);
    }

    Weights w = alloc_weights();
    int max_wsz = WP_SZ > W1_SZ ? WP_SZ : W1_SZ;
    max_wsz = max_wsz > W2_SZ ? max_wsz : W2_SZ;
    if (server_only) {
        if (!load_weights(w)) { fprintf(stderr, "No weights.bin found — train first\n"); return 1; }
    } else {
        init_weights_k<<<(max_wsz+255)/256, 256>>>(w.wp, w.w1, w.w2, w.fp_wp, w.fp_w1, w.fp_w2, w.bp, w.b1, w.b2, 42ULL);
        CK(cudaDeviceSynchronize());
    }

    Tape tape = alloc_tape();

    int bs = server_only ? 1 : BATCH;
    float* matmul_scratch; CK(cudaMalloc(&matmul_scratch, (size_t)bs * NC * HID_PAD * 4));
    half* fp16_scratch; CK(cudaMalloc(&fp16_scratch, (size_t)bs * NC * P_IN * 2));
    float* buf_a; CK(cudaMalloc(&buf_a, (size_t)bs * NC * CH * 4));
    float* buf_b; CK(cudaMalloc(&buf_b, (size_t)bs * NC * CH * 4));

    curandStatePhilox4_32_10_t* rngs;
    CK(cudaMalloc(&rngs, (size_t)bs * BLK * sizeof(curandStatePhilox4_32_10_t)));
    init_rng_k<<<(bs*BLK+255)/256, 256>>>(rngs, 12345ULL, bs*BLK);
    CK(cudaDeviceSynchronize());

    float* d_ds = NULL; half* h_scratch = NULL;
    float* f_scratch = NULL; float* f_scratch2 = NULL;
    float* d_im2col = NULL; float* d_state_grad = NULL;
    float* d_loss = NULL; int* d_correct = NULL; int* d_total = NULL;
    uint8_t* d_blabels = NULL; int* d_batch_idx = NULL;

    if (!server_only) {
        CK(cudaMalloc(&d_ds, (size_t)BATCH * NC * U_OUT_PAD * 4));
        CK(cudaMalloc(&h_scratch, (size_t)BATCH * NC * HID_PAD * 2));
        CK(cudaMalloc(&f_scratch, (size_t)BATCH * NC * HID_PAD * 4));
        CK(cudaMalloc(&f_scratch2, (size_t)BATCH * NC * HID_PAD * 4));
        CK(cudaMalloc(&d_im2col, (size_t)BATCH * NC * P_IN * 4));
        CK(cudaMalloc(&d_state_grad, (size_t)BATCH * NC * CH * 4));
        CK(cudaMalloc(&d_loss, 4));
        CK(cudaMalloc(&d_correct, 4));
        CK(cudaMalloc(&d_total, 4));
        CK(cudaMalloc(&d_blabels, BATCH));
        CK(cudaMalloc(&d_batch_idx, BATCH*4));
    }

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "GPU: %s (%zu MB)\n", prop.name, prop.totalGlobalMem >> 20);

    if (!server_only) {
        Pool pool; init_pool(pool, d_train_images, h_train_labels);
        fprintf(stderr, "Params: %d | batch=%d pool=%d steps=%d\n",
                WP_SZ+W1_SZ+W2_SZ+BP_SZ+B1_SZ+B2_SZ, BATCH, POOL, STEPS);

        float loss_ema = -1;
        clock_t wall_start = clock(), interval_start = wall_start;
        int h_pool_idx[BATCH]; uint8_t h_blabels[BATCH];
        int lb = (BATCH * NC + 255) / 256;
        if (lb > 256) lb = 256;

        for (int iter = 1; iter <= ITERS; iter++) {
            sample_batch(pool, buf_a, h_blabels, h_pool_idx, d_train_images, h_train_labels, d_batch_idx, d_blabels);
            float* in = buf_a, *out = buf_b;
            for (int s = 0; s < STEPS; s++) {
                forward_one_step(w, tape, s, in, out, rngs, matmul_scratch, fp16_scratch, BATCH, true);
                float* t = in; in = out; out = t;
            }
            float* final_state = in;
            CK(cudaMemsetAsync(d_loss, 0, 4));
            CK(cudaMemsetAsync(d_correct, 0, 4));
            CK(cudaMemsetAsync(d_total, 0, 4));
            CK(cudaMemsetAsync(d_state_grad, 0, (size_t)BATCH * NC * CH * 4));
            loss_grad_k<<<lb, 256, 256*4>>>(final_state, d_blabels, d_state_grad, d_loss, d_correct, d_total, BATCH);
            CK(cudaMemsetAsync(w.gp, 0, WP_SZ*4));
            CK(cudaMemsetAsync(w.g1, 0, W1_SZ*4));
            CK(cudaMemsetAsync(w.g2, 0, W2_SZ*4));
            CK(cudaMemsetAsync(w.gbp, 0, BP_SZ*4));
            CK(cudaMemsetAsync(w.gb1, 0, B1_SZ*4));
            CK(cudaMemsetAsync(w.gb2, 0, B2_SZ*4));
            for (int s = STEPS - 1; s >= 0; s--)
                backward_one_step(w, tape, s, d_state_grad, d_ds, h_scratch, f_scratch, f_scratch2, d_im2col, BATCH);
            optim_step(w, get_lr(iter), iter);
            commit_batch(pool, final_state, d_batch_idx);
            if (iter % LOG_EVERY == 0 || iter <= 5) {
                CK(cudaDeviceSynchronize());
                float h_loss;
                CK(cudaMemcpy(&h_loss, d_loss, 4, cudaMemcpyDeviceToHost));
                int h_corr, h_tot;
                CK(cudaMemcpy(&h_corr, d_correct, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&h_tot, d_total, 4, cudaMemcpyDeviceToHost));
                float h_gnorms[NTENSORS];
                CK(cudaMemcpy(h_gnorms, w.gnorms, NTENSORS*4, cudaMemcpyDeviceToHost));
                loss_ema = loss_ema < 0 ? h_loss : 0.95f * loss_ema + 0.05f * h_loss;
                fprintf(f_train, "%d,%.6f,%.1e\n", iter, h_loss, get_lr(iter));
                fflush(f_train);
                if (!isfinite(h_loss) || h_loss > 1e6f) {
                    fprintf(stderr, "\nFATAL: loss=%f iter %d\n", h_loss, iter); return 1;
                }
                clock_t now_c = clock();
                float el = (float)(now_c - wall_start) / CLOCKS_PER_SEC;
                float iv = (float)(now_c - interval_start) / CLOCKS_PER_SEC;
                float ips = iv > 0 ? (iter <= 5 ? iter : LOG_EVERY) / iv : 0;
                interval_start = now_c;
                float acc = h_tot > 0 ? 100.0f * h_corr / h_tot : 0.0f;
                fprintf(stderr, "iter %6d | loss %7.3f (ema %7.3f) | acc %5.1f%% (%d/%d) | lr %.1e | gnorm wp=%.2f w1=%.2f w2=%.2f | %.1f it/s | %.0fs\n",
                        iter, h_loss, loss_ema, acc, h_corr, h_tot, get_lr(iter), h_gnorms[0], h_gnorms[1], h_gnorms[2], ips, el);
            }
        }
        save_weights(w);
        fprintf(stderr, "\n=== Final eval (10k test samples, 200 steps) ===\n");
        eval_telemetry(w, tape, buf_a, buf_b, matmul_scratch, fp16_scratch, rngs,
                       d_test_images, h_test_labels, NTEST, ITERS, f_csv, f_summary);
        fprintf(stderr, "\n=== Dumping eval snapshots ===\n");
        dump_eval_snapshots(w, tape, buf_a, buf_b, matmul_scratch, fp16_scratch, rngs,
                           d_test_images, h_test_labels, ts);
        fclose(f_csv);
        fclose(f_summary);
        fclose(f_train);
    }

    fprintf(stderr, "\n=== Entering inference server mode ===\n");
    serve_inference(w, tape, buf_a, buf_b, matmul_scratch, fp16_scratch, rngs);
    return 0;
}
