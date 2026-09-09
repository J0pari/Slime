#pragma once
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

struct ServerWeights {
    half *wp, *w1, *w2;
    float *fp_wp, *fp_w1, *fp_w2;
    float *bp, *b1, *b2;
};

static ServerWeights alloc_server_weights() {
    ServerWeights sw;
    CK(cudaMalloc(&sw.wp, WP_SZ*2)); CK(cudaMalloc(&sw.w1, W1_SZ*2)); CK(cudaMalloc(&sw.w2, W2_SZ*2));
    CK(cudaMalloc(&sw.fp_wp, WP_SZ*4)); CK(cudaMalloc(&sw.fp_w1, W1_SZ*4)); CK(cudaMalloc(&sw.fp_w2, W2_SZ*4));
    CK(cudaMalloc(&sw.bp, BP_SZ*4)); CK(cudaMalloc(&sw.b1, B1_SZ*4)); CK(cudaMalloc(&sw.b2, B2_SZ*4));
    return sw;
}

__global__ void srv_fp32_to_fp16_k(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static bool server_load_weights(ServerWeights& sw, cudaStream_t stream) {
    FILE* f = fopen("../weights.bin", "rb");
    if (!f) return false;
    float* h = (float*)malloc(WP_SZ * 4);
    fread(h, 4, WP_SZ, f); CK(cudaMemcpyAsync(sw.fp_wp, h, WP_SZ*4, cudaMemcpyHostToDevice, stream));
    fread(h, 4, W1_SZ, f); CK(cudaMemcpyAsync(sw.fp_w1, h, W1_SZ*4, cudaMemcpyHostToDevice, stream));
    fread(h, 4, W2_SZ, f); CK(cudaMemcpyAsync(sw.fp_w2, h, W2_SZ*4, cudaMemcpyHostToDevice, stream));
    fread(h, 4, BP_SZ, f); CK(cudaMemcpyAsync(sw.bp, h, BP_SZ*4, cudaMemcpyHostToDevice, stream));
    fread(h, 4, B1_SZ, f); CK(cudaMemcpyAsync(sw.b1, h, B1_SZ*4, cudaMemcpyHostToDevice, stream));
    fread(h, 4, B2_SZ, f); CK(cudaMemcpyAsync(sw.b2, h, B2_SZ*4, cudaMemcpyHostToDevice, stream));
    free(h);
    fclose(f);
    srv_fp32_to_fp16_k<<<(WP_SZ+255)/256, 256, 0, stream>>>(sw.fp_wp, sw.wp, WP_SZ);
    srv_fp32_to_fp16_k<<<(W1_SZ+255)/256, 256, 0, stream>>>(sw.fp_w1, sw.w1, W1_SZ);
    srv_fp32_to_fp16_k<<<(W2_SZ+255)/256, 256, 0, stream>>>(sw.fp_w2, sw.w2, W2_SZ);
    CK(cudaStreamSynchronize(stream));
    return true;
}

static time_t get_weights_mtime() {
    struct _stat st;
    if (_stat("../weights.bin", &st) != 0) return 0;
    return st.st_mtime;
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

struct ServerContext {
    ServerWeights sw;
    float* buf_a;
    float* buf_b;
    float* matmul_scratch;
    half* fp16_scratch;
    curandStatePhilox4_32_10_t* rngs;
    Tape tape;
    cudaStream_t stream;
    bool weights_loaded;
    time_t weights_mtime;
};

static void server_forward(ServerContext& ctx, float* in, float* out, int step) {
    Weights w;
    w.wp = ctx.sw.wp; w.w1 = ctx.sw.w1; w.w2 = ctx.sw.w2;
    w.fp_wp = ctx.sw.fp_wp; w.fp_w1 = ctx.sw.fp_w1; w.fp_w2 = ctx.sw.fp_w2;
    w.bp = ctx.sw.bp; w.b1 = ctx.sw.b1; w.b2 = ctx.sw.b2;
    w.gp=w.g1=w.g2=w.gbp=w.gb1=w.gb2=nullptr;
    w.mp=w.vp=w.m1=w.v1=w.m2=w.v2=nullptr;
    w.mbp=w.vbp=w.mb1=w.vb1=w.mb2=w.vb2=nullptr;
    w.gnorms=nullptr;
    forward_one_step(w, ctx.tape, step % STEPS, in, out, ctx.rngs,
                     ctx.matmul_scratch, ctx.fp16_scratch, 1, false);
}

static void server_try_reload(ServerContext& ctx) {
    time_t mt = get_weights_mtime();
    if (mt == 0) return;
    if (mt != ctx.weights_mtime) {
        if (server_load_weights(ctx.sw, ctx.stream)) {
            ctx.weights_mtime = mt;
            ctx.weights_loaded = true;
            fprintf(stderr, "[server] Reloaded weights (mtime %lld)\n", (long long)mt);
        }
    }
}

static void serve_loop(ServerContext& ctx) {
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
        fprintf(stderr, "[server] bind failed\n"); return;
    }
    listen(srv, 1);
    fprintf(stderr, "\n=== INFERENCE SERVER on ws://localhost:8765 ===\n");

    float* h_state = (float*)malloc(NC * CH * 4);
    float* h_pixels = (float*)malloc(NC * 4);
    uint8_t* ws_buf = (uint8_t*)malloc(NC * CH * 4 + 16);

    for (;;) {
        server_try_reload(ctx);
        if (!ctx.weights_loaded) {
            fprintf(stderr, "[server] No weights.bin yet, waiting...\n");
            Sleep(5000);
            continue;
        }

        fprintf(stderr, "[server] Waiting for connection...\n");

        fd_set rset;
        FD_ZERO(&rset);
        FD_SET(srv, &rset);
        struct timeval tv;
        tv.tv_sec = 5; tv.tv_usec = 0;
        int sel = select(0, &rset, NULL, NULL, &tv);
        if (sel <= 0) continue;

        SOCKET client = accept(srv, NULL, NULL);
        if (client == INVALID_SOCKET) continue;
        fprintf(stderr, "[server] Client connected, handshaking...\n");
        if (!ws_handshake(client)) { closesocket(client); continue; }
        fprintf(stderr, "[server] WebSocket established\n");

        u_long nonblock = 1;
        ioctlsocket(client, FIONBIO, &nonblock);

        bool have_image = false;
        float* in_buf = ctx.buf_a;
        float* out_buf = ctx.buf_b;
        int step = 0;
        int reload_counter = 0;

        for (;;) {
            int n = ws_recv(client, ws_buf, NC * 4 + 16);
            if (n == NC * 4) {
                float* new_px = (float*)ws_buf;
                if (have_image) {
                    CK(cudaMemcpy(h_state, in_buf, NC * CH * 4, cudaMemcpyDeviceToHost));
                    for (int cell = 0; cell < NC; cell++) {
                        float og = h_pixels[cell], ng = new_px[cell];
                        h_state[cell * CH] = ng;
                        if ((ng > ALIVE_TH) && !(og > ALIVE_TH))
                            for (int c = 1; c < CH; c++) h_state[cell * CH + c] = 0.0f;
                        else if (!(ng > ALIVE_TH))
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
            } else if (n == -1) {
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK) break;
            } else if (n == 0) {
                break;
            }

            if (have_image) {
                server_forward(ctx, in_buf, out_buf, step);
                float* t = in_buf; in_buf = out_buf; out_buf = t;
                step++;
                CK(cudaMemcpy(h_state, in_buf, NC * CH * 4, cudaMemcpyDeviceToHost));
                if (!ws_send_bin(client, (uint8_t*)h_state, NC * CH * 4)) break;
                if (++reload_counter % 200 == 0) server_try_reload(ctx);
                Sleep(16);
            } else {
                Sleep(10);
            }
        }
        fprintf(stderr, "[server] Client disconnected\n");
        closesocket(client);
    }

    free(h_state); free(h_pixels); free(ws_buf);
    closesocket(srv);
    WSACleanup();
}

static unsigned __stdcall server_thread_func(void* arg) {
    (void)arg;

    ServerContext ctx = {};
    CK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    ctx.sw = alloc_server_weights();
    CK(cudaMalloc(&ctx.buf_a, (size_t)NC * CH * 4));
    CK(cudaMalloc(&ctx.buf_b, (size_t)NC * CH * 4));
    CK(cudaMalloc(&ctx.matmul_scratch, (size_t)NC * HID_PAD * 4));
    CK(cudaMalloc(&ctx.fp16_scratch, (size_t)NC * P_IN * 2));
    CK(cudaMalloc(&ctx.rngs, (size_t)BLK * sizeof(curandStatePhilox4_32_10_t)));
    init_rng_k<<<(BLK+255)/256, 256, 0, ctx.stream>>>(ctx.rngs, 99999ULL, BLK);
    CK(cudaStreamSynchronize(ctx.stream));

    ctx.tape = {};
    ctx.weights_loaded = false;
    ctx.weights_mtime = 0;

    serve_loop(ctx);
    return 0;
}

static void start_server_thread() {
    _beginthreadex(NULL, 0, server_thread_func, NULL, 0, NULL);
}
