#ifndef FIELD_VIEW_CUH
#define FIELD_VIEW_CUH

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include <cuda_runtime.h>


enum FieldChannel {
    
    CH_CA_0 = 0,
    CH_CA_1, CH_CA_2, CH_CA_3, CH_CA_4, CH_CA_5, CH_CA_6, CH_CA_7,
    CH_CA_8, CH_CA_9, CH_CA_10, CH_CA_11, CH_CA_12, CH_CA_13, CH_CA_14, CH_CA_15,

    
    CH_CHEM_CONCENTRATION,
    CH_CHEM_GRADIENT_X,
    CH_CHEM_GRADIENT_Y,
    CH_CHEM_LAPLACIAN,
    CH_CHEM_SOURCES,
    CH_CHEM_DECAY,

    
    CH_RESOURCE_DENSITY,
    CH_RESOURCE_NEXT,
    CH_FITNESS_LANDSCAPE,
    CH_RESOURCE_GRAD_X,
    CH_RESOURCE_GRAD_Y,

    
    CH_BEHAVIORAL_START,
    CH_BEHAVIORAL_END = CH_BEHAVIORAL_START + 31,

    
    CH_STRESS,              
    CH_DORMANCY,            
    CH_REACTIVATION,        
    CH_PHASE_INDICATOR,     

    
    CH_CA_OUTPUT_RECURRENCE,

    CH_COUNT
};


static_assert(CH_CA_15 < CH_CHEM_CONCENTRATION, "CA channels must be contiguous");
static_assert(CH_CHEM_DECAY < CH_RESOURCE_DENSITY, "Chemical channels must be contiguous");
static_assert(CH_RESOURCE_GRAD_Y < CH_BEHAVIORAL_START, "Resource channels must be contiguous");
static_assert(CH_BEHAVIORAL_END < CH_STRESS, "Behavioral channels must be contiguous");
static_assert(CH_COUNT <= 128, "Total channel count must fit in reasonable bounds");



struct FieldView {
    float* base;            
    int stride_x;           
    int stride_y;           
    int stride_channel;     
    int channel_offset;     
    int num_channels;       
    int grid_size;          

    
    
    

    
    __device__ __forceinline__ float read(int x, int y, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::read");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::read channel");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return 0.0f;
        }
        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        int idx = y * stride_y + x * stride_x + (channel_offset + channel) * stride_channel;
        float val = base[idx];

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(val, "FieldView::read value");
        #endif

        return val;
    }

    
    __device__ __forceinline__ float read_ldg(int x, int y, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::read_ldg");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::read_ldg channel");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return 0.0f;
        }
        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        int idx = y * stride_y + x * stride_x + (channel_offset + channel) * stride_channel;
        float val = ldg_float(&base[idx]);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(val, "FieldView::read_ldg value");
        #endif

        return val;
    }

    
    __device__ __forceinline__ void write(int x, int y, int channel, float value) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::write");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::write channel");
        VALIDATE_FINITE(value, "FieldView::write value");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return;
        }
        if (channel < 0 || channel >= num_channels) {
            return;
        }

        int idx = y * stride_y + x * stride_x + (channel_offset + channel) * stride_channel;
        base[idx] = value;
    }

    
    __device__ __forceinline__ void atomic_add(int x, int y, int channel, float value) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::atomic_add");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::atomic_add channel");
        VALIDATE_FINITE(value, "FieldView::atomic_add value");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return;
        }
        if (channel < 0 || channel >= num_channels) {
            return;
        }

        int idx = y * stride_y + x * stride_x + (channel_offset + channel) * stride_channel;
        atomicAdd(&base[idx], value);
    }

    
    __device__ __forceinline__ float* ptr_at(int x, int y, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::ptr_at");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::ptr_at channel");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return nullptr;
        }
        if (channel < 0 || channel >= num_channels) {
            return nullptr;
        }

        int idx = y * stride_y + x * stride_x + (channel_offset + channel) * stride_channel;
        return &base[idx];
    }

    
    __device__ __forceinline__ float* channel_base(int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::channel_base");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return nullptr;
        }

        return &base[(channel_offset + channel) * stride_channel];
    }

    
    
    

    
    __device__ __forceinline__ void load_3x3(
        float (&stencil)[3][3],
        int x, int y,
        int channel
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::load_3x3");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::load_3x3 channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            #pragma unroll
            for (int dy = 0; dy < 3; dy++) {
                #pragma unroll
                for (int dx = 0; dx < 3; dx++) {
                    stencil[dy][dx] = 0.0f;
                }
            }
            return;
        }

        const float* channel_ptr = channel_base(channel);
        Stencils::load_3x3(stencil, channel_ptr, x, y, grid_size, stride_x);
    }

    
    __device__ __forceinline__ float laplacian_at(int x, int y, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::laplacian_at");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::laplacian_at channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        const float* channel_ptr = channel_base(channel);
        float lap = Stencils::laplacian_at(channel_ptr, x, y, grid_size, stride_x);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(lap, "FieldView::laplacian_at result");
        #endif

        return lap;
    }

    
    __device__ __forceinline__ float gradient_x_at(int x, int y, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::gradient_x_at");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::gradient_x_at channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        const float* channel_ptr = channel_base(channel);
        float grad_x = Stencils::gradient_x_at(channel_ptr, x, y, grid_size, stride_x);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(grad_x, "FieldView::gradient_x_at result");
        #endif

        return grad_x;
    }

    
    __device__ __forceinline__ float gradient_y_at(int x, int y, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::gradient_y_at");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::gradient_y_at channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        const float* channel_ptr = channel_base(channel);
        float grad_y = Stencils::gradient_y_at(channel_ptr, x, y, grid_size, stride_x);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(grad_y, "FieldView::gradient_y_at result");
        #endif

        return grad_y;
    }

    
    __device__ __forceinline__ void gradients_at(
        float& grad_x,
        float& grad_y,
        int x, int y,
        int channel
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::gradients_at");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::gradients_at channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            grad_x = 0.0f;
            grad_y = 0.0f;
            return;
        }

        const float* channel_ptr = channel_base(channel);
        Stencils::gradients_at(grad_x, grad_y, channel_ptr, x, y, grid_size, stride_x);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(grad_x, "FieldView::gradients_at grad_x");
        VALIDATE_FINITE(grad_y, "FieldView::gradients_at grad_y");
        #endif
    }

    
    __device__ __forceinline__ void all_operators(
        float& grad_x,
        float& grad_y,
        float& lap,
        float& center,
        int x, int y,
        int channel
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::all_operators");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::all_operators channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            grad_x = 0.0f;
            grad_y = 0.0f;
            lap = 0.0f;
            center = 0.0f;
            return;
        }

        const float* channel_ptr = channel_base(channel);
        Stencils::all_operators(grad_x, grad_y, lap, center, channel_ptr, x, y, grid_size, stride_x);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(grad_x, "FieldView::all_operators grad_x");
        VALIDATE_FINITE(grad_y, "FieldView::all_operators grad_y");
        VALIDATE_FINITE(lap, "FieldView::all_operators lap");
        VALIDATE_FINITE(center, "FieldView::all_operators center");
        #endif
    }

    
    
    

    
    
    __device__ __forceinline__ float bilinear_read(float fx, float fy, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::bilinear_read channel");
        VALIDATE_FINITE(fx, "FieldView::bilinear_read fx");
        VALIDATE_FINITE(fy, "FieldView::bilinear_read fy");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        fx = clamp(fx, 0.0f, static_cast<float>(grid_size - 1));
        fy = clamp(fy, 0.0f, static_cast<float>(grid_size - 1));

        int x0 = static_cast<int>(fx);
        int y0 = static_cast<int>(fy);
        int x1 = min(x0 + 1, grid_size - 1);
        int y1 = min(y0 + 1, grid_size - 1);

        float tx = fx - static_cast<float>(x0);
        float ty = fy - static_cast<float>(y0);

        float tl = read_ldg(x0, y0, channel);
        float tr = read_ldg(x1, y0, channel);
        float bl = read_ldg(x0, y1, channel);
        float br = read_ldg(x1, y1, channel);

        float result = Interpolation::bilinear(tl, tr, bl, br, tx, ty);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(result, "FieldView::bilinear_read result");
        #endif

        return result;
    }

    
    
    __device__ __forceinline__ float3 bilinear_with_grad(float fx, float fy, int channel) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::bilinear_with_grad channel");
        VALIDATE_FINITE(fx, "FieldView::bilinear_with_grad fx");
        VALIDATE_FINITE(fy, "FieldView::bilinear_with_grad fy");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        fx = clamp(fx, 0.0f, static_cast<float>(grid_size - 1));
        fy = clamp(fy, 0.0f, static_cast<float>(grid_size - 1));

        int x0 = static_cast<int>(fx);
        int y0 = static_cast<int>(fy);
        int x1 = min(x0 + 1, grid_size - 1);
        int y1 = min(y0 + 1, grid_size - 1);

        float tx = fx - static_cast<float>(x0);
        float ty = fy - static_cast<float>(y0);

        float tl = read_ldg(x0, y0, channel);
        float tr = read_ldg(x1, y0, channel);
        float bl = read_ldg(x0, y1, channel);
        float br = read_ldg(x1, y1, channel);

        float3 result = Interpolation::bilinear_with_grad(tl, tr, bl, br, tx, ty);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(result.x, "FieldView::bilinear_with_grad value");
        VALIDATE_FINITE(result.y, "FieldView::bilinear_with_grad grad_x");
        VALIDATE_FINITE(result.z, "FieldView::bilinear_with_grad grad_y");
        #endif

        return result;
    }

    
    __device__ __forceinline__ float4 bilinear_weights(float fx, float fy) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(fx, "FieldView::bilinear_weights fx");
        VALIDATE_FINITE(fy, "FieldView::bilinear_weights fy");
        #endif

        fx = clamp(fx, 0.0f, static_cast<float>(grid_size - 1));
        fy = clamp(fy, 0.0f, static_cast<float>(grid_size - 1));

        int x0 = static_cast<int>(fx);
        int y0 = static_cast<int>(fy);

        float tx = fx - static_cast<float>(x0);
        float ty = fy - static_cast<float>(y0);

        return Interpolation::bilinear_weights(tx, ty);
    }

    
    
    

    
    __device__ __forceinline__ float sample_neighborhood_avg(
        int x, int y,
        int channel,
        int radius = 1
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::sample_neighborhood_avg");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::sample_neighborhood_avg channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        float sum = 0.0f;
        int count = 0;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                    sum += read_ldg(nx, ny, channel);
                    count++;
                }
            }
        }

        float result = (count > 0) ? sum / static_cast<float>(count) : 0.0f;

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(result, "FieldView::sample_neighborhood_avg result");
        #endif

        return result;
    }

    
    __device__ __forceinline__ float sample_neighborhood_max(
        int x, int y,
        int channel,
        int radius = 1
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::sample_neighborhood_max");
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::sample_neighborhood_max channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return 0.0f;
        }

        float max_val = -INFINITY;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                    max_val = fmaxf(max_val, read_ldg(nx, ny, channel));
                }
            }
        }

        float result = (max_val == -INFINITY) ? 0.0f : max_val;

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(result, "FieldView::sample_neighborhood_max result");
        #endif

        return result;
    }

    
    
    

    
    
    
    
    template<int TILE_DIM, int HALO, int BANK_OFFSET>
    __device__ __forceinline__ void load_tile_with_halo(
        float (&tile)[TILE_DIM + 2 * HALO + BANK_OFFSET][TILE_DIM + 2 * HALO + BANK_OFFSET],
        int channel
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::load_tile_with_halo channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return;
        }

        const float* channel_ptr = channel_base(channel);
        TiledSection2D<TILE_DIM, HALO, BANK_OFFSET>::load_with_halo(tile, channel_ptr, grid_size);
    }

    
    template<int TILE_DIM, int HALO, int BANK_OFFSET>
    __device__ __forceinline__ void store_from_tile(
        const float (&tile)[TILE_DIM + 2 * HALO + BANK_OFFSET][TILE_DIM + 2 * HALO + BANK_OFFSET],
        int channel
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_RANGE(channel, 0, num_channels - 1, "FieldView::store_from_tile channel");
        #endif

        if (channel < 0 || channel >= num_channels) {
            return;
        }

        float* channel_ptr = channel_base(channel);
        TiledSection2D<TILE_DIM, HALO, BANK_OFFSET>::store_from_tile(channel_ptr, tile, grid_size);
    }

    
    
    

    
    
    __device__ __forceinline__ float4 read_float4(int x, int y, int channel_start) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::read_float4");
        VALIDATE_RANGE(channel_start, 0, num_channels - 4, "FieldView::read_float4 channel_start");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (channel_start < 0 || channel_start + 3 >= num_channels) {
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        
        if (stride_channel == 1) {
            int idx = y * stride_y + x * stride_x + (channel_offset + channel_start) * stride_channel;
            return ldg_float4((const float4*)&base[idx]);
        } else {
            
            return make_float4(
                read_ldg(x, y, channel_start + 0),
                read_ldg(x, y, channel_start + 1),
                read_ldg(x, y, channel_start + 2),
                read_ldg(x, y, channel_start + 3)
            );
        }
    }

    
    __device__ __forceinline__ void write_float4(int x, int y, int channel_start, float4 values) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::write_float4");
        VALIDATE_RANGE(channel_start, 0, num_channels - 4, "FieldView::write_float4 channel_start");
        VALIDATE_FINITE(values.x, "FieldView::write_float4 value.x");
        VALIDATE_FINITE(values.y, "FieldView::write_float4 value.y");
        VALIDATE_FINITE(values.z, "FieldView::write_float4 value.z");
        VALIDATE_FINITE(values.w, "FieldView::write_float4 value.w");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return;
        }
        if (channel_start < 0 || channel_start + 3 >= num_channels) {
            return;
        }

        
        if (stride_channel == 1) {
            int idx = y * stride_y + x * stride_x + (channel_offset + channel_start) * stride_channel;
            ((float4*)base)[idx / 4] = values;
        } else {
            
            write(x, y, channel_start + 0, values.x);
            write(x, y, channel_start + 1, values.y);
            write(x, y, channel_start + 2, values.z);
            write(x, y, channel_start + 3, values.w);
        }
    }

    
    __device__ __forceinline__ void copy_all_channels(
        int src_x, int src_y,
        int dst_x, int dst_y
    ) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(src_x, src_y, grid_size, "FieldView::copy_all_channels src");
        VALIDATE_GRID_COORDINATES(dst_x, dst_y, grid_size, "FieldView::copy_all_channels dst");
        #endif

        if (src_x < 0 || src_x >= grid_size || src_y < 0 || src_y >= grid_size) {
            return;
        }
        if (dst_x < 0 || dst_x >= grid_size || dst_y < 0 || dst_y >= grid_size) {
            return;
        }

        
        int c = 0;
        if (stride_channel == 1) {
            for (; c + 3 < num_channels; c += 4) {
                float4 vals = read_float4(src_x, src_y, c);
                write_float4(dst_x, dst_y, c, vals);
            }
        }

        
        for (; c < num_channels; c++) {
            float val = read_ldg(src_x, src_y, c);
            write(dst_x, dst_y, c, val);
        }
    }

    
    __device__ __forceinline__ void clear_all_channels(int x, int y) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::clear_all_channels");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return;
        }

        
        int c = 0;
        if (stride_channel == 1) {
            for (; c + 3 < num_channels; c += 4) {
                write_float4(x, y, c, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }

        
        for (; c < num_channels; c++) {
            write(x, y, c, 0.0f);
        }
    }

    
    __device__ __forceinline__ float channel_norm_l2(int x, int y) const {
        #ifdef __CUDA_ARCH__
        VALIDATE_GRID_COORDINATES(x, y, grid_size, "FieldView::channel_norm_l2");
        #endif

        if (x < 0 || x >= grid_size || y < 0 || y >= grid_size) {
            return 0.0f;
        }

        float sum_sq = 0.0f;
        for (int c = 0; c < num_channels; c++) {
            float val = read_ldg(x, y, c);
            sum_sq = __fmaf_rn(val, val, sum_sq);
        }

        float result = sqrtf(sum_sq);

        #ifdef __CUDA_ARCH__
        VALIDATE_FINITE(result, "FieldView::channel_norm_l2 result");
        #endif

        return result;
    }
};





__device__ __host__ __forceinline__ FieldView make_ca_view(float* backing_memory, int grid_size) {
    FieldView view;
    view.base = backing_memory;
    view.stride_x = 1;
    view.stride_y = grid_size;
    view.stride_channel = grid_size * grid_size;
    view.channel_offset = CH_CA_0;
    view.num_channels = 16;
    view.grid_size = grid_size;
    return view;
}

__device__ __host__ __forceinline__ FieldView make_chemical_view(float* backing_memory, int grid_size) {
    FieldView view;
    view.base = backing_memory;
    view.stride_x = 1;
    view.stride_y = grid_size;
    view.stride_channel = grid_size * grid_size;
    view.channel_offset = CH_CHEM_CONCENTRATION;
    view.num_channels = 6;
    view.grid_size = grid_size;
    return view;
}

__device__ __host__ __forceinline__ FieldView make_resource_view(float* backing_memory, int grid_size) {
    FieldView view;
    view.base = backing_memory;
    view.stride_x = 1;
    view.stride_y = grid_size;
    view.stride_channel = grid_size * grid_size;
    view.channel_offset = CH_RESOURCE_DENSITY;
    view.num_channels = 5;
    view.grid_size = grid_size;
    return view;
}

__device__ __host__ __forceinline__ FieldView make_behavioral_view(float* backing_memory, int grid_size) {
    FieldView view;
    view.base = backing_memory;
    view.stride_x = 1;
    view.stride_y = grid_size;
    view.stride_channel = grid_size * grid_size;
    view.channel_offset = CH_BEHAVIORAL_START;
    view.num_channels = 32;
    view.grid_size = grid_size;
    return view;
}

__device__ __host__ __forceinline__ FieldView make_lifecycle_view(float* backing_memory, int grid_size) {
    FieldView view;
    view.base = backing_memory;
    view.stride_x = 1;
    view.stride_y = grid_size;
    view.stride_channel = grid_size * grid_size;
    view.channel_offset = CH_STRESS;
    view.num_channels = 4;
    view.grid_size = grid_size;
    return view;
}

__device__ __host__ __forceinline__ FieldView make_custom_view(
    float* backing_memory,
    int grid_size,
    int channel_offset,
    int num_channels,
    int stride_x = 1,
    int stride_y = -1,
    int stride_channel = -1
) {
    FieldView view;
    view.base = backing_memory;
    view.stride_x = stride_x;
    view.stride_y = (stride_y == -1) ? grid_size : stride_y;
    view.stride_channel = (stride_channel == -1) ? (grid_size * grid_size) : stride_channel;
    view.channel_offset = channel_offset;
    view.num_channels = num_channels;
    view.grid_size = grid_size;
    return view;
}

#endif
