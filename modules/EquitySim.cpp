module;

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <future>
#include <immintrin.h>

export module EquitySim;

// Results of the simulation
export struct SimResult {
  double starting_price;
  double final_mean;
  double max_path;
  double min_path;
  double volatility_ann;
  double drift_ann;
};

// SIMD-aligned storage for 4 doubles
struct alignas(32) Vec4 {
    double data[4];
    
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
};

// Fast scalar exp for array processing (more reliable than full SIMD exp)
inline void fast_exp_4(const double* input, double* output) {
    for (int i = 0; i < 4; ++i) {
        double x = input[i];
        // Clamp to avoid overflow
        if (x > 709.0) x = 709.0;
        if (x < -709.0) x = -709.0;
        
        // Range reduction
        double k = std::round(x * 1.4426950408889634);  // x / ln(2)
        double f = x - k * 0.6931471805599453;  // x - k*ln(2)
        
        // Polynomial approximation for exp(f) where f in [-0.5, 0.5]
        double f2 = f * f;
        double f4 = f2 * f2;
        double p = 1.0 + f * (1.0 + f * (0.5 + f * (0.16666666666666666 + 
               f * (0.041666666666666664 + f * (0.008333333333333333 + 
               f * 0.001388888888888889)))));
        
        // Scale by 2^k
        output[i] = std::ldexp(p, static_cast<int>(k));
    }
}

// Box-Muller transform for generating normal random numbers
// Generates 4 normal variates from 4 uniform variates
inline void box_muller_4(const double* u1, const double* u2, double* z1, double* z2) {
    const double two_pi = 2.0 * M_PI;
    
    for (int i = 0; i < 4; ++i) {
        // r = sqrt(-2 * log(u1))
        double log_u1 = std::log(std::max(u1[i], 1e-300));
        double r = std::sqrt(-2.0 * log_u1);
        
        // theta = 2 * pi * u2
        double theta = two_pi * u2[i];
        
        // z1 = r * cos(theta), z2 = r * sin(theta)
        z1[i] = r * std::cos(theta);
        z2[i] = r * std::sin(theta);
    }
}

export class EquityEngine {
private:
  double mu = 0.0;    // Annualized Drift
  double sigma = 0.0; // Annualized Volatility
  const double TRADING_DAYS = 252.0;

public:
  // Calibrate the model using historical closing prices
  void calibrate(const std::vector<double> &history) {
    if (history.size() < 2)
      return;

    std::vector<double> log_returns;
    for (size_t i = 1; i < history.size(); ++i) {
      log_returns.push_back(std::log(history[i] / history[i - 1]));
    }

    // Calculate Mean of log returns
    double sum = std::accumulate(log_returns.begin(), log_returns.end(), 0.0);
    double mean_return = sum / log_returns.size();

    // Calculate Variance/Stdev of log returns
    double sq_sum = std::inner_product(log_returns.begin(), log_returns.end(),
                                       log_returns.begin(), 0.0);
    double var = (sq_sum / log_returns.size()) - (mean_return * mean_return);
    double stdev = std::sqrt(var);

    // Annualize parameters
    mu = mean_return * TRADING_DAYS;
    sigma = stdev * std::sqrt(TRADING_DAYS);
  }

  SimResult run(double s0, int days_to_sim, int num_paths) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    double dt = 1.0 / TRADING_DAYS;
    std::vector<double> final_prices;
    final_prices.reserve(num_paths);

    for (int i = 0; i < num_paths; ++i) {
      double price = s0;
      for (int d = 0; d < days_to_sim; ++d) {
        double z = dist(gen);
        // Geometric Brownian Motion formula
        price *= std::exp((mu - 0.5 * sigma * sigma) * dt +
                          (sigma * std::sqrt(dt) * z));
      }
      final_prices.push_back(price);
    }

    auto [min_it, max_it] =
        std::minmax_element(final_prices.begin(), final_prices.end());
    double avg =
        std::accumulate(final_prices.begin(), final_prices.end(), 0.0) /
        num_paths;

    return {s0, avg, *max_it, *min_it, sigma, mu};
  }

  SimResult run_parallel(double s0, int days_to_sim, int total_paths) {
    // Determine how many threads the CPU can actually handle
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2; // Fallback

    int paths_per_thread = total_paths / num_threads;
    std::vector<std::future<std::vector<double>>> futures;

    // Launch "chunks" of work
    for (unsigned int t = 0; t < num_threads; ++t) {
        futures.push_back(std::async(std::launch::async, [this, s0, days_to_sim, paths_per_thread, t]() {
            // Each thread gets its own generator and unique seed
            std::mt19937 gen(std::random_device{}() + t); 
            std::normal_distribution<double> dist(0.0, 1.0);
            
            std::vector<double> local_results;
            double dt = 1.0 / 252.0;

            for (int i = 0; i < paths_per_thread; ++i) {
                double price = s0;
                for (int d = 0; d < days_to_sim; ++d) {
                    price *= std::exp((mu - 0.5 * sigma * sigma) * dt + (sigma * std::sqrt(dt) * dist(gen)));
                }
                local_results.push_back(price);
            }
            return local_results;
        }));
    }

    // Collect and merge results
    std::vector<double> all_prices;
    for (auto& f : futures) {
        auto chunk = f.get();
        all_prices.insert(all_prices.end(), chunk.begin(), chunk.end());
    }

    auto [min_it, max_it] = std::minmax_element(all_prices.begin(), all_prices.end());
    double avg = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();

    return { s0, avg, *max_it, *min_it, sigma, mu };
  }

  // SIMD-optimized version using AVX2
  SimResult run_simd(double s0, int days_to_sim, int num_paths) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    double dt = 1.0 / TRADING_DAYS;
    
    // Precompute drift term: (mu - 0.5 * sigma * sigma) * dt
    double drift_term = (mu - 0.5 * sigma * sigma) * dt;
    double sigma_sqrt_dt = sigma * std::sqrt(dt);
    
    std::vector<double> final_prices;
    final_prices.reserve(num_paths);
    
    // Process paths in chunks of 4 using SIMD-aligned arrays
    int simd_paths = num_paths / 4 * 4;
    alignas(32) double prices[4];
    alignas(32) double u1[4], u2[4];
    alignas(32) double z1[4], z2[4];
    alignas(32) double exp_input[4];
    alignas(32) double exp_output[4];
    
    for (int i = 0; i < simd_paths; i += 4) {
        // Initialize all 4 prices to s0
        prices[0] = prices[1] = prices[2] = prices[3] = s0;
        
        for (int d = 0; d < days_to_sim; ++d) {
            // Generate 8 uniform random numbers [0, 1)
            // Using Box-Muller, we get 8 normals from 8 uniforms
            for (int j = 0; j < 4; ++j) {
                u1[j] = dist(gen);
                u2[j] = dist(gen);
            }
            
            // Convert to normal using Box-Muller (generates 8 normals)
            box_muller_4(u1, u2, z1, z2);
            
            // Process first 4 normals (z1) for this day
            // GBM: price *= exp(drift + sigma * sqrt(dt) * z)
            for (int j = 0; j < 4; ++j) {
                exp_input[j] = drift_term + sigma_sqrt_dt * z1[j];
            }
            
            // Compute exp for all 4 paths
            fast_exp_4(exp_input, exp_output);
            
            // Multiply prices
            for (int j = 0; j < 4; ++j) {
                prices[j] *= exp_output[j];
            }
        }
        
        // Store results
        for (int j = 0; j < 4; ++j) {
            final_prices.push_back(prices[j]);
        }
    }
    
    // Handle remaining paths (scalar fallback)
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    for (int i = simd_paths; i < num_paths; ++i) {
        double price = s0;
        for (int d = 0; d < days_to_sim; ++d) {
            double z = normal_dist(gen);
            price *= std::exp(drift_term + sigma_sqrt_dt * z);
        }
        final_prices.push_back(price);
    }
    
    auto [min_it, max_it] = std::minmax_element(final_prices.begin(), final_prices.end());
    double avg = std::accumulate(final_prices.begin(), final_prices.end(), 0.0) / num_paths;
    
    return {s0, avg, *max_it, *min_it, sigma, mu};
  }

  void print_result(const SimResult &result) {

    std::cout << "------------------------------------------\n";
    std::cout << std::format("Estimated Annual Drift: {:.2f}%\n",
                             result.drift_ann * 100);
    std::cout << std::format("Estimated Volatility:   {:.2f}%\n",
                             result.volatility_ann * 100);
    std::cout << "------------------------------------------\n";
    std::cout << std::format("Starting Price:         ${:.2f}\n",
                             result.starting_price);
    std::cout << std::format("Expected Price (Mean):  ${:.2f}\n",
                             result.final_mean);
    std::cout << std::format("Worst Case Path:        ${:.2f}\n",
                             result.min_path);
    std::cout << std::format("Best Case Path:         ${:.2f}\n",
                             result.max_path);
  }
};
