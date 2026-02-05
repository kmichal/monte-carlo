module;

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

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