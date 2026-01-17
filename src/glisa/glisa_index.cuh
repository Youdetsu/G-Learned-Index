/**
 * G-LISA: GPU-Native Multidimensional Learned Index
 * Main Index Structure with Unified Memory Support
 */

#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "hilbert.cuh"
#include "../../include/pgm/pgm_index.hpp"

namespace glisa {

// Space-filling curve type
enum class CurveType {
    ZORDER,
    HILBERT
};

// Memory mode
enum class MemoryMode {
    EXPLICIT,       // Traditional cudaMalloc + cudaMemcpy
    UNIFIED         // cudaMallocManaged with prefetching
};

/**
 * Configuration for G-LISA index
 */
struct GLISAConfig {
    CurveType curve = CurveType::HILBERT;
    MemoryMode memory = MemoryMode::UNIFIED;
    int epsilon = 128;          // PGM error bound
    int blockSize = 256;        // CUDA block size
    bool enablePrefetch = true; // Enable prefetching for UM
};

/**
 * Segment structure for GPU
 */
struct Segment {
    uint32_t key;
    float slope;
    int32_t intercept;
};

/**
 * G-LISA Index: GPU-accelerated Learned Index for Spatial Data
 */
template<int Epsilon = 128>
class GLISAIndex {
public:
    GLISAConfig config;
    
    // Host data
    std::vector<uint32_t> h_data_x;
    std::vector<uint32_t> h_data_y;
    std::vector<uint32_t> h_keys;      // 1D keys after curve encoding
    
    // Device data (Unified Memory or explicit)
    uint32_t* d_keys = nullptr;
    Segment* d_segments = nullptr;
    int* d_levels_offsets = nullptr;
    
    // Index metadata
    int data_size = 0;
    int segments_count = 0;
    int levels_count = 0;
    uint32_t first_key = 0;
    
    std::vector<Segment> h_segments;
    std::vector<int> h_levels_offsets;

public:
    GLISAIndex(const GLISAConfig& cfg = GLISAConfig()) : config(cfg) {}
    
    ~GLISAIndex() {
        freeDeviceMemory();
    }

    /**
     * Build index from 2D points
     */
    void build(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y) {
        data_size = x.size();
        h_data_x = x;
        h_data_y = y;
        h_keys.resize(data_size);
        
        std::cout << "[G-LISA] Building index with " << data_size << " points\n";
        std::cout << "[G-LISA] Curve type: " 
                  << (config.curve == CurveType::HILBERT ? "Hilbert" : "Z-order") << "\n";
        std::cout << "[G-LISA] Memory mode: "
                  << (config.memory == MemoryMode::UNIFIED ? "Unified" : "Explicit") << "\n";
        
        // Step 1: Encode 2D points to 1D keys using space-filling curve
        encodePoints();
        
        // Step 2: Sort keys (and maintain mapping)
        std::vector<size_t> indices(data_size);
        for (size_t i = 0; i < data_size; i++) indices[i] = i;
        std::sort(indices.begin(), indices.end(), 
                  [this](size_t a, size_t b) { return h_keys[a] < h_keys[b]; });
        
        std::vector<uint32_t> sorted_keys(data_size);
        std::vector<uint32_t> sorted_x(data_size), sorted_y(data_size);
        for (size_t i = 0; i < data_size; i++) {
            sorted_keys[i] = h_keys[indices[i]];
            sorted_x[i] = h_data_x[indices[i]];
            sorted_y[i] = h_data_y[indices[i]];
        }
        h_keys = sorted_keys;
        h_data_x = sorted_x;
        h_data_y = sorted_y;
        first_key = h_keys[0];
        
        // Step 3: Build PGM index on sorted keys
        buildPGMIndex();
        
        // Step 4: Allocate and transfer to GPU
        allocateDeviceMemory();
        
        std::cout << "[G-LISA] Index built: " << segments_count << " segments, "
                  << levels_count << " levels\n";
        
        // Debug: Print first segment (disabled)
        if (false && segments_count > 0) {
            std::cout << "[DEBUG] First segment: key=" << h_segments[0].key 
                      << ", slope=" << h_segments[0].slope 
                      << ", intercept=" << h_segments[0].intercept << "\n";
            std::cout << "[DEBUG] First key=" << first_key << ", data_size=" << data_size << "\n";
        }
    }

    /**
     * Batch point query on GPU
     * Returns predicted positions for each query
     */
    void batchQuery(const uint32_t* query_x, const uint32_t* query_y,
                    int* results_lo, int* results_hi, int num_queries);

    /**
     * Range query: find all points in rectangle [x1,x2] x [y1,y2]
     */
    void rangeQuery(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                    std::vector<std::pair<uint32_t, uint32_t>>& results);

private:
    void encodePoints() {
        if (config.curve == CurveType::HILBERT) {
            for (int i = 0; i < data_size; i++) {
                h_keys[i] = HilbertCurve::encode(h_data_x[i], h_data_y[i]);
            }
        } else {
            for (int i = 0; i < data_size; i++) {
                h_keys[i] = ZOrderCurve::encode(h_data_x[i], h_data_y[i]);
            }
        }
    }
    
    void buildPGMIndex() {
        pgm::PGMIndex<uint32_t, Epsilon> pgm_index(h_keys);
        
        segments_count = pgm_index.segments.size();
        levels_count = pgm_index.levels_offsets.size();
        
        h_segments.resize(segments_count);
        for (int i = 0; i < segments_count; i++) {
            h_segments[i].key = pgm_index.segments[i].key;
            h_segments[i].slope = pgm_index.segments[i].slope;
            h_segments[i].intercept = pgm_index.segments[i].intercept;
        }
        
        h_levels_offsets.resize(levels_count);
        for (int i = 0; i < levels_count; i++) {
            h_levels_offsets[i] = pgm_index.levels_offsets[i];
        }
    }
    
    void allocateDeviceMemory() {
        freeDeviceMemory();
        
        size_t keys_size = data_size * sizeof(uint32_t);
        size_t segments_size = segments_count * sizeof(Segment);
        size_t levels_size = levels_count * sizeof(int);
        
        if (config.memory == MemoryMode::UNIFIED) {
            // Unified Memory allocation
            cudaMallocManaged(&d_keys, keys_size);
            cudaMallocManaged(&d_segments, segments_size);
            cudaMallocManaged(&d_levels_offsets, levels_size);
            
            // Copy data
            memcpy(d_keys, h_keys.data(), keys_size);
            memcpy(d_segments, h_segments.data(), segments_size);
            memcpy(d_levels_offsets, h_levels_offsets.data(), levels_size);
            
            // Note: cudaMemPrefetchAsync API changed in CUDA 13
            // Prefetch disabled for now, UM will handle page migration automatically
            cudaDeviceSynchronize();
        } else {
            // Explicit memory allocation
            cudaMalloc(&d_keys, keys_size);
            cudaMalloc(&d_segments, segments_size);
            cudaMalloc(&d_levels_offsets, levels_size);
            
            cudaMemcpy(d_keys, h_keys.data(), keys_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_segments, h_segments.data(), segments_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_levels_offsets, h_levels_offsets.data(), levels_size, cudaMemcpyHostToDevice);
        }
    }
    
    void freeDeviceMemory() {
        if (d_keys) { cudaFree(d_keys); d_keys = nullptr; }
        if (d_segments) { cudaFree(d_segments); d_segments = nullptr; }
        if (d_levels_offsets) { cudaFree(d_levels_offsets); d_levels_offsets = nullptr; }
    }
};

} // namespace glisa
