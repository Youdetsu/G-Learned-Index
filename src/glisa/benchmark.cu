/**
 * G-LISA Benchmark: Compare Hilbert vs Z-order
 * Complete end-to-end benchmark including CPU exact search
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

#include "glisa_index.cuh"
#include "glisa_kernels.cuh"
#include "hilbert.cuh"

using namespace glisa;

// ============================================================
// Data Loading (same as g-learned.cu)
// ============================================================

bool loadDataFromFile(const std::string& filename, 
                      std::vector<uint32_t>& x, std::vector<uint32_t>& y, int n) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
        std::cerr << "Cannot open data file: " << filename << std::endl;
        return false;
    }
    
    x.resize(n);
    y.resize(n);
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%u %u", &x[i], &y[i]) != 2) {
            std::cerr << "Error reading data at line " << i << std::endl;
            fclose(fp);
            return false;
        }
    }
    fclose(fp);
    return true;
}

bool loadQueriesFromFile(const std::string& filename,
                         std::vector<uint32_t>& qx, std::vector<uint32_t>& qy, 
                         int file_count, int total_count) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
        std::cerr << "Cannot open query file: " << filename << std::endl;
        return false;
    }
    
    qx.resize(total_count);
    qy.resize(total_count);
    
    // Read queries from file
    for (int i = 0; i < file_count; i++) {
        if (fscanf(fp, "%u %u", &qx[i], &qy[i]) != 2) {
            std::cerr << "Error reading query at line " << i << std::endl;
            fclose(fp);
            return false;
        }
    }
    fclose(fp);
    
    // Replicate queries to reach total_count (same as g-learned.cu)
    for (int i = file_count; i < total_count; i++) {
        qx[i] = qx[i - file_count];
        qy[i] = qy[i - file_count];
    }
    
    return true;
}

// ============================================================
// Locality Analysis
// ============================================================

double measureLocality(const std::vector<uint32_t>& x, 
                       const std::vector<uint32_t>& y,
                       CurveType curve) {
    int n = x.size();
    std::vector<uint32_t> keys(n);
    
    for (int i = 0; i < n; i++) {
        if (curve == CurveType::HILBERT) {
            keys[i] = HilbertCurve::encode(x[i], y[i]);
        } else {
            keys[i] = ZOrderCurve::encode(x[i], y[i]);
        }
    }
    
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&keys](size_t a, size_t b) { return keys[a] < keys[b]; });
    
    double total_dist = 0;
    for (int i = 1; i < n; i++) {
        double dx = double(x[indices[i]]) - double(x[indices[i-1]]);
        double dy = double(y[indices[i]]) - double(y[indices[i-1]]);
        total_dist += std::sqrt(dx*dx + dy*dy);
    }
    
    return total_dist / (n - 1);
}

// ============================================================
// Benchmark Result
// ============================================================

struct BenchmarkResult {
    std::string name;
    double build_time_ms;
    double total_time_ms;      // Total query time (GPU + CPU)
    double gpu_time_ms;        // GPU prediction time only
    double throughput_mqps;    // Million queries per second
    int segments;
    double locality_score;
    
    // Performance analysis metrics
    double avg_search_range;   // Average search range size (hi - lo)
    double max_search_range;   // Maximum search range
    int total_searches;        // Total number of exact searches performed
};

// ============================================================
// Complete End-to-End Benchmark (same timing as g-learned.cu)
// ============================================================

template<int Epsilon>
BenchmarkResult runBenchmark(
    const std::vector<uint32_t>& data_x,
    const std::vector<uint32_t>& data_y,
    const std::vector<uint32_t>& query_x,
    const std::vector<uint32_t>& query_y,
    GLISAConfig config,
    const std::string& name
) {
    BenchmarkResult result;
    result.name = name;
    
    // Measure locality
    result.locality_score = measureLocality(data_x, data_y, config.curve);
    
    // Build index
    GLISAIndex<Epsilon> index(config);
    
    auto build_start = std::chrono::high_resolution_clock::now();
    index.build(data_x, data_y);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    result.segments = index.segments_count;
    
    // Prepare query keys (encode query points) - outside timing, same as g-learned.cu
    int num_queries = query_x.size();
    std::vector<uint32_t> query_keys(num_queries);
    for (int i = 0; i < num_queries; i++) {
        if (config.curve == CurveType::HILBERT) {
            query_keys[i] = HilbertCurve::encode(query_x[i], query_y[i]);
        } else {
            query_keys[i] = ZOrderCurve::encode(query_x[i], query_y[i]);
        }
    }
    
    // Host arrays for results
    std::vector<int> h_lo(num_queries);
    std::vector<int> h_hi(num_queries);
    
    // Statistics accumulators
    double total_range = 0;
    double max_range = 0;
    
    // ========================================
    // Benchmark: Same timing as g-learned.cu
    // cudaMalloc is OUTSIDE timing (same as g-learned.cu)
    // Timing includes: cudaMemcpy H2D, kernel, cudaMemcpy D2H, CPU search
    // ========================================
    
    double tot_time_us = 0;
    double latency_ms = 0;
    int batch_size = num_queries;  // Same as g-learned.cu: 1 batch of 4M queries
    
    // Pre-allocate GPU memory OUTSIDE timing loop (same as g-learned.cu)
    uint32_t *d_qx, *d_qy;
    int *d_lo, *d_hi;
    cudaMalloc(&d_qx, batch_size * sizeof(uint32_t));
    cudaMalloc(&d_qy, batch_size * sizeof(uint32_t));
    cudaMalloc(&d_lo, batch_size * sizeof(int));
    cudaMalloc(&d_hi, batch_size * sizeof(int));
    
    for (int batch = 0; batch < num_queries / batch_size; batch++) {
        int block_size = 128;  // Same as g-learned.cu
        int grid_size = (batch_size - 1) / block_size + 1;
        
        // Timer 1: GPU query (memcpy H2D + kernel) - same as g-learned.cu querytimer
        cudaEvent_t query_start, query_stop;
        cudaEventCreate(&query_start);
        cudaEventCreate(&query_stop);
        
        cudaEventRecord(query_start);
        
        // Copy query data to GPU
        cudaMemcpy(d_qx, query_x.data() + batch * batch_size, 
                   batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qy, query_y.data() + batch * batch_size,
                   batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // GPU kernel
        launchBatchQuery<Epsilon>(
            d_qx, d_qy, index.d_segments, index.d_levels_offsets,
            index.first_key, index.data_size, batch_size,
            d_lo, d_hi, config.curve
        );
        
        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        }
        
        cudaEventRecord(query_stop);
        cudaEventSynchronize(query_stop);
        
        float query_ms;
        cudaEventElapsedTime(&query_ms, query_start, query_stop);
        
        // Timer 2: Lower bound (memcpy D2H + CPU search) - same as g-learned.cu lower_bound_timer
        cudaEvent_t lb_start, lb_stop;
        cudaEventCreate(&lb_start);
        cudaEventCreate(&lb_stop);
        
        cudaEventRecord(lb_start);
        
        // Copy results back to CPU
        cudaMemcpy(h_lo.data() + batch * batch_size, d_lo, 
                   batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hi.data() + batch * batch_size, d_hi,
                   batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(lb_stop);
        cudaEventSynchronize(lb_stop);
        
        float memcpy_ms;
        cudaEventElapsedTime(&memcpy_ms, lb_start, lb_stop);
        
        // Debug: Check first few values
        if (batch == 0 && false) {  // Disabled debug output
            std::cout << "[DEBUG] First 5 search ranges:\n";
            for (int i = 0; i < 5 && i < batch_size; i++) {
                std::cout << "  Query " << i << ": lo=" << h_lo[i] << ", hi=" << h_hi[i] 
                          << ", range=" << (h_hi[i] - h_lo[i]) << "\n";
            }
        }
        
        // CPU exact search (same as g-learned.cu)
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < batch_size; i++) {
            uint32_t target = query_keys[batch * batch_size + i];
            int lo = h_lo[batch * batch_size + i];
            int hi = h_hi[batch * batch_size + i];
            
            // Collect statistics
            int range_size = hi - lo;
            total_range += range_size;
            max_range = std::max(max_range, (double)range_size);
            
            // Exact search
            for (int j = lo; j < hi && j < index.data_size; j++) {
                if (index.h_keys[j] == target) {
                    break;
                }
            }
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ns = std::chrono::duration<double, std::nano>(cpu_end - cpu_start).count();
        
        // Total time for this batch (same calculation as g-learned.cu)
        // GpuTimer.getNsElapsed() returns ms * 1000 = microseconds
        double batch_time_us = query_ms * 1000 + memcpy_ms * 1000 + cpu_ns / 1000.0;
        latency_ms = std::max(latency_ms, batch_time_us / 1000.0);
        tot_time_us += batch_time_us;
        
        cudaEventDestroy(query_start);
        cudaEventDestroy(query_stop);
        cudaEventDestroy(lb_start);
        cudaEventDestroy(lb_stop);
    }
    
    // Free GPU memory OUTSIDE timing loop
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_lo);
    cudaFree(d_hi);
    
    // Calculate throughput (same formula as g-learned.cu: N * 1e6 / 1024 / 1024 / tot_time_us)
    result.total_time_ms = tot_time_us / 1000.0;
    result.gpu_time_ms = 0;  // Not separately tracked in this mode
    result.throughput_mqps = num_queries * 1e6 / 1024.0 / 1024.0 / tot_time_us;
    
    // Store statistics
    result.avg_search_range = total_range / num_queries;
    result.max_search_range = max_range;
    result.total_searches = num_queries;
    
    return result;
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(130, '=') << "\n";
    std::cout << "BENCHMARK RESULTS (End-to-End: GPU Prediction + CPU Exact Search)\n";
    std::cout << std::string(130, '=') << "\n\n";
    
    std::cout << std::left << std::setw(32) << "Configuration"
              << std::right << std::setw(10) << "Build(ms)"
              << std::setw(12) << "Total(ms)"
              << std::setw(10) << "GPU(ms)"
              << std::setw(14) << "Throughput"
              << std::setw(10) << "Segments"
              << std::setw(10) << "Locality"
              << std::setw(12) << "AvgRange"
              << std::setw(12) << "MaxRange"
              << "\n";
    std::cout << std::string(130, '-') << "\n";
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.build_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.gpu_time_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.throughput_mqps << " Mqps"
                  << std::setw(10) << r.segments
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.locality_score
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.avg_search_range
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.max_search_range
                  << "\n";
    }
    std::cout << std::string(130, '=') << "\n";
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    std::cout << "G-LISA Benchmark: Hilbert vs Z-order (End-to-End)\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Configuration - same as g-learned.cu
    const int DATA_SIZE = 100000;    // 1e5 points (same as g-learned.cu)
    const int QUERY_FILE_SIZE = 10000;  // 10000 queries in file
    const int QUERY_SIZE = 1 << 22;  // ~4M queries (replicated, same as g-learned.cu)
    const int EPSILON = 128;
    
    // Data files (same as g-learned.cu)
    std::string data_file = "../random_1e5.in";
    std::string query_file = "../query_1e5.in";
    
    std::cout << "Data file: " << data_file << "\n";
    std::cout << "Query file: " << query_file << "\n";
    std::cout << "Data size: " << DATA_SIZE << "\n";
    std::cout << "Query size: " << QUERY_SIZE << " (" << (QUERY_SIZE / 1000000.0) << "M)\n";
    std::cout << "Epsilon: " << EPSILON << "\n\n";
    
    // Load data from file (same as g-learned.cu)
    std::vector<uint32_t> data_x, data_y;
    if (!loadDataFromFile(data_file, data_x, data_y, DATA_SIZE)) {
        std::cerr << "Failed to load data file!\n";
        return 1;
    }
    std::cout << "Loaded " << DATA_SIZE << " data points from " << data_file << "\n";
    
    // Load queries from file (same as g-learned.cu)
    std::vector<uint32_t> query_x, query_y;
    if (!loadQueriesFromFile(query_file, query_x, query_y, QUERY_FILE_SIZE, QUERY_SIZE)) {
        std::cerr << "Failed to load query file!\n";
        return 1;
    }
    std::cout << "Loaded " << QUERY_FILE_SIZE << " queries, replicated to " << QUERY_SIZE << "\n\n";
    
    std::vector<BenchmarkResult> results;
    GLISAConfig config;
    
    // 1. Z-order (baseline, same as original G-Learned Index)
    config.curve = CurveType::ZORDER;
    config.memory = MemoryMode::EXPLICIT;
    results.push_back(runBenchmark<EPSILON>(data_x, data_y, query_x, query_y, config,
                                             "Z-order (baseline)"));
    
    // 2. Hilbert (our improvement)
    config.curve = CurveType::HILBERT;
    config.memory = MemoryMode::EXPLICIT;
    results.push_back(runBenchmark<EPSILON>(data_x, data_y, query_x, query_y, config,
                                             "Hilbert (ours)"));
    
    printResults(results);
    
    // Print improvement and detailed analysis
    double speedup = results[1].throughput_mqps / results[0].throughput_mqps;
    double locality_improve = (results[0].locality_score - results[1].locality_score) / results[0].locality_score * 100;
    double range_reduction = (results[0].avg_search_range - results[1].avg_search_range) / results[0].avg_search_range * 100;
    
    std::cout << "\n=== Performance Analysis ===\n";
    std::cout << "Hilbert vs Z-order:\n";
    std::cout << "  Throughput improvement: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "  Locality improvement: " << std::fixed << std::setprecision(1) << locality_improve << "%\n";
    std::cout << "  Average search range reduction: " << std::fixed << std::setprecision(1) << range_reduction << "%\n";
    std::cout << "  Z-order avg range: " << std::fixed << std::setprecision(1) << results[0].avg_search_range << "\n";
    std::cout << "  Hilbert avg range: " << std::fixed << std::setprecision(1) << results[1].avg_search_range << "\n";
    std::cout << "\nKey Insight: Smaller search range → Better cache locality → Higher throughput\n";
    
    std::cout << "\nBenchmark completed!\n";
    return 0;
}
