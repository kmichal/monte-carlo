#include <iostream>
#include <CLI/CLI.hpp>

import MarketData;


int main(int argc, char** argv) {

    std::string token;
    std::string symbol;
    int lookback_days;
    int days_forward;
    int simulations;

    CLI::App app{"Monte Carlo Simulation Tool"};

    app.add_option("--tradier-token", token, "Tradier API token")->envname("TRADIER_API_TOKEN")->required();
    app.add_option("--symbol", symbol, "Ticker symbol")->required();
    app.add_option("--lookback-days", lookback_days, "Number of trading days to look back")->default_val(90);
    app.add_option("--days-forward", days_forward, "Number of trading days to look forward")->default_val(90);
    app.add_option("--simulations", simulations, "Number of simulations to run")->default_val(10000);

    CLI11_PARSE(app, argc, argv);


    TradierClient client;

    return 0;

}