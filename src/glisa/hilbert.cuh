/**
 * G-LISA: GPU-Native Multidimensional Learned Index
 * Hilbert Curve Implementation for better spatial locality
 * 
 * Hilbert curve preserves locality better than Z-order curve,
 * which is crucial for range queries and kNN.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace glisa {

/**
 * 2D Hilbert Curve encoding/decoding
 * Order n means the curve fills a 2^n x 2^n grid
 */
class HilbertCurve {
public:
    static constexpr int ORDER = 16;  // 2^16 x 2^16 = 65536 x 65536 grid
    static constexpr uint32_t MAX_COORD = (1u << ORDER) - 1;

    /**
     * Convert (x, y) coordinates to Hilbert index
     * Host and Device compatible
     */
    __host__ __device__
    static uint32_t encode(uint32_t x, uint32_t y) {
        uint32_t rx, ry, s, d = 0;
        for (s = (1u << ORDER) / 2; s > 0; s /= 2) {
            rx = (x & s) > 0 ? 1 : 0;
            ry = (y & s) > 0 ? 1 : 0;
            d += s * s * ((3 * rx) ^ ry);
            
            // 将坐标转换为子象限的局部坐标，避免 rotate 中 n - 1 - x 下溢
            x &= (s - 1);
            y &= (s - 1);
            
            rotate(s, &x, &y, rx, ry);
        }
        return d;
    }

    /**
     * Convert Hilbert index back to (x, y) coordinates
     * Host and Device compatible
     */
    __host__ __device__
    static void decode(uint32_t d, uint32_t* x, uint32_t* y) {
        uint32_t rx, ry, s, t = d;
        *x = *y = 0;
        for (s = 1; s < (1u << ORDER); s *= 2) {
            rx = 1 & (t / 2);
            ry = 1 & (t ^ rx);
            rotate(s, x, y, rx, ry);
            *x += s * rx;
            *y += s * ry;
            t /= 4;
        }
    }

    /**
     * Batch encode on GPU
     */
    static void batchEncode(const uint32_t* x, const uint32_t* y, 
                           uint32_t* indices, int n);

    /**
     * Batch decode on GPU  
     */
    static void batchDecode(const uint32_t* indices,
                           uint32_t* x, uint32_t* y, int n);

private:
    /**
     * Rotate/flip quadrant
     */
    __host__ __device__
    static void rotate(uint32_t n, uint32_t* x, uint32_t* y, 
                       uint32_t rx, uint32_t ry) {
        if (ry == 0) {
            if (rx == 1) {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }
            // Swap x and y
            uint32_t t = *x;
            *x = *y;
            *y = t;
        }
    }
};

// ============================================================
// Z-order curve for comparison (existing implementation)
// ============================================================
class ZOrderCurve {
public:
    static constexpr int ORDER = 16;
    static constexpr uint32_t MAX_COORD = (1u << ORDER) - 1;

    __host__ __device__
    static uint32_t encode(uint32_t x, uint32_t y) {
        uint32_t idx = 0;
        for (int i = 0; i < ORDER; i++) {
            idx |= ((((x >> i) & 1) << 1) | ((y >> i) & 1)) << (i << 1);
        }
        return idx;
    }

    __host__ __device__
    static void decode(uint32_t idx, uint32_t* x, uint32_t* y) {
        *x = *y = 0;
        for (int i = 0; i < ORDER; i++) {
            *x |= ((idx >> (i << 1 | 1)) & 1) << i;
            *y |= ((idx >> (i << 1)) & 1) << i;
        }
    }
};

// ============================================================
// GPU Kernels for batch operations
// ============================================================

__global__ void hilbertEncodeKernel(const uint32_t* x, const uint32_t* y,
                                     uint32_t* indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = HilbertCurve::encode(x[i], y[i]);
    }
}

__global__ void hilbertDecodeKernel(const uint32_t* indices,
                                     uint32_t* x, uint32_t* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        HilbertCurve::decode(indices[i], &x[i], &y[i]);
    }
}

__global__ void zorderEncodeKernel(const uint32_t* x, const uint32_t* y,
                                    uint32_t* indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = ZOrderCurve::encode(x[i], y[i]);
    }
}

// Inline implementations for batch operations
inline void HilbertCurve::batchEncode(const uint32_t* x, const uint32_t* y,
                                       uint32_t* indices, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    hilbertEncodeKernel<<<gridSize, blockSize>>>(x, y, indices, n);
    cudaDeviceSynchronize();
}

inline void HilbertCurve::batchDecode(const uint32_t* indices,
                                       uint32_t* x, uint32_t* y, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    hilbertDecodeKernel<<<gridSize, blockSize>>>(indices, x, y, n);
    cudaDeviceSynchronize();
}

} // namespace glisa
