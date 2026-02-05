#include <iostream>
#include <CLI/CLI.hpp>
#include <string>
#include <chrono>
#include <cstdlib>


import MarketData;
import EquitySim;


int main(int argc, char** argv) {

    std::string token;
    std::string symbol;
    int lookback_days;
    int days_forward;
    int simulations;
    std::string start_date_str;
    std::string end_date_str;

    CLI::App app{"Monte Carlo Simulation Tool"};

    app.add_option("--tradier-token", token, "Tradier API token")->envname("TRADIER_API_TOKEN")->required();
    app.add_option("--symbol", symbol, "Ticker symbol")->required();
    app.add_option("--lookback-days", lookback_days, "Number of trading days to look back")->default_val(90);
    app.add_option("--days-forward", days_forward, "Number of trading days to look forward")->default_val(90);
    app.add_option("--simulations", simulations, "Number of simulations to run")->default_val(10000);

    CLI11_PARSE(app, argc, argv);

    auto now = std::chrono::system_clock::now();

    auto x_days_ago = now - std::chrono::days(lookback_days + (lookback_days / 5 * 2)); // Roughly account for weekends

    try {
        std::chrono::zoned_time eastern_now{"America/New_York", now};
        std::chrono::zoned_time eastern_start{"America/New_York", x_days_ago};

        end_date_str = std::format("{:%F}", eastern_now);
        start_date_str = std::format("{:%F}", eastern_start);

        std::cout << "Current Date (ET): " << end_date_str << std::endl;
        std::cout << "Start Date (-" << lookback_days << " trading days): " << start_date_str << std::endl;
    } 
    catch (const std::runtime_error& e) {
        std::cerr << "Time zone error: " << e.what() << std::endl;
    }


    TradierClient client(token);


    HistoryRequest req{
        .symbol = symbol,
        .start_date = start_date_str,
        .end_date = end_date_str
    };


    std::vector<double> history = {};

    auto result = client.get_market_history(req);
    
        if (result) {
        std::cout << "Retrieved " << result->size() << " candles for " << req.symbol << ":\n";
        
        for (const auto& candle : *result) {
            history.push_back(candle.close);

        }
    } else {
        std::cerr << "Failed to retrieve or parse data." << std::endl;
    }

    EquityEngine engine;
    engine.calibrate(history);

    double current_price = history.back();
    SimResult res = engine.run_parallel(current_price, days_forward, simulations);
    engine.print_result(res);

    return 0;

}