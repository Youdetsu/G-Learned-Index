/**
 * G-LISA: GPU Kernels for query operations
 */

#pragma once
#include <cuda_runtime.h>
#include "glisa_index.cuh"
#include "hilbert.cuh"

namespace glisa {

#define GLISA_MIN(x, y) ((x) < (y) ? (x) : (y))
#define GLISA_MAX(x, y) ((x) > (y) ? (x) : (y))
#define SUB_EPS(x, eps) ((x) <= (eps) ? 0 : ((x) - (eps)))
#define ADD_EPS(x, eps, size) ((x) + (eps) + 2 >= (size) ? (size) : (x) + (eps) + 2)

/**
 * GPU Kernel: Batch point query
 * Each thread handles one query
 */
template<int Epsilon, CurveType Curve>
__global__ void batchQueryKernel(
    const uint32_t* query_x,
    const uint32_t* query_y,
    const Segment* segments,
    const int* levels_offsets,
    uint32_t first_key,
    int data_size,
    int num_queries,
    int* results_lo,
    int* results_hi
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    
    // Step 1: Encode query point to 1D key
    uint32_t key;
    if constexpr (Curve == CurveType::HILBERT) {
        key = HilbertCurve::encode(query_x[tid], query_y[tid]);
    } else {
        key = ZOrderCurve::encode(query_x[tid], query_y[tid]);
    }
    key = GLISA_MAX(key, first_key);
    
    // Step 2: Binary search in segments to find the right segment
    int left = 0;
    int right = levels_offsets[1] - 1;
    int seg_idx = 0;
    
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (segments[mid].key <= key) {
            left = mid + 1;
            seg_idx = mid;
        } else {
            right = mid - 1;
        }
    }
    
    // Step 3: Predict position using linear model
    int64_t pos = int64_t(segments[seg_idx].slope * (key - segments[seg_idx].key)) 
                  + segments[seg_idx].intercept;
    int next_intercept = segments[seg_idx + 1].intercept;
    pos = GLISA_MIN(pos, next_intercept);
    pos = pos > 0 ? pos : 0;
    
    // Step 4: Return search range
    results_lo[tid] = SUB_EPS(pos, Epsilon);
    results_hi[tid] = ADD_EPS(pos, Epsilon, data_size);
}

/**
 * GPU Kernel: Exact search within predicted range
 * This moves the "last mile" search to GPU
 */
__global__ void exactSearchKernel(
    const uint32_t* keys,
    const uint32_t* query_keys,
    const int* lo,
    const int* hi,
    int num_queries,
    int* results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    
    uint32_t target = query_keys[tid];
    int found = -1;
    
    // Linear search in the predicted range
    for (int i = lo[tid]; i < hi[tid]; i++) {
        if (keys[i] == target) {
            found = i;
            break;
        }
    }
    
    results[tid] = found;
}

/**
 * GPU Kernel: Range query - check if points are in rectangle
 * Used after getting candidate range from index
 */
__global__ void rangeCheckKernel(
    const uint32_t* data_x,
    const uint32_t* data_y,
    int start_idx,
    int end_idx,
    uint32_t x1, uint32_t y1,
    uint32_t x2, uint32_t y2,
    int* in_range,      // Output: 1 if in range, 0 otherwise
    int* count          // Atomic counter for results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start_idx + tid;
    
    if (idx >= end_idx) return;
    
    uint32_t x = data_x[idx];
    uint32_t y = data_y[idx];
    
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        in_range[tid] = 1;
        atomicAdd(count, 1);
    } else {
        in_range[tid] = 0;
    }
}

/**
 * GPU Kernel: kNN query - compute distances
 */
__global__ void knnDistanceKernel(
    const uint32_t* data_x,
    const uint32_t* data_y,
    uint32_t query_x,
    uint32_t query_y,
    int start_idx,
    int end_idx,
    float* distances
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start_idx + tid;
    
    if (idx >= end_idx) {
        distances[tid] = 1e30f;  // Large value for out-of-range
        return;
    }
    
    float dx = float(data_x[idx]) - float(query_x);
    float dy = float(data_y[idx]) - float(query_y);
    distances[tid] = dx * dx + dy * dy;  // Squared distance
}

// ============================================================
// Host wrapper functions
// ============================================================

template<int Epsilon>
void launchBatchQuery(
    const uint32_t* d_query_x,
    const uint32_t* d_query_y,
    const Segment* d_segments,
    const int* d_levels_offsets,
    uint32_t first_key,
    int data_size,
    int num_queries,
    int* d_results_lo,
    int* d_results_hi,
    CurveType curve,
    int blockSize = 256
) {
    int gridSize = (num_queries + blockSize - 1) / blockSize;
    
    if (curve == CurveType::HILBERT) {
        batchQueryKernel<Epsilon, CurveType::HILBERT><<<gridSize, blockSize>>>(
            d_query_x, d_query_y, d_segments, d_levels_offsets,
            first_key, data_size, num_queries, d_results_lo, d_results_hi
        );
    } else {
        batchQueryKernel<Epsilon, CurveType::ZORDER><<<gridSize, blockSize>>>(
            d_query_x, d_query_y, d_segments, d_levels_offsets,
            first_key, data_size, num_queries, d_results_lo, d_results_hi
        );
    }
}

} // namespace glisa
