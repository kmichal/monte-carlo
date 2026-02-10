module; // Global module fragment
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <optional>
#include <iostream>


export module MarketData;

using json = nlohmann::json;


export struct HistoryRequest {
    std::string symbol;
    std::string start_date;
    std::string end_date;
    std::string interval = "daily";
};

export struct Candle {
    std::string date;
    double open;
    double high;
    double low;
    double close;
    long long volume;
};

void from_json(const json& j, Candle& c) {
    j.at("date").get_to(c.date);
    j.at("open").get_to(c.open);
    j.at("high").get_to(c.high);
    j.at("low").get_to(c.low);
    j.at("close").get_to(c.close);
    j.at("volume").get_to(c.volume);
}

export class TradierClient {
private:
    static constexpr const char* HISTORY_API_URL = "https://api.tradier.com/v1/markets/history?";

public:
    explicit TradierClient(std::string api_token) 
        : token_(std::move(api_token)) {
    }

    ~TradierClient() = default;

    // Now returns a vector of Candles instead of a string
    std::optional<std::vector<Candle>> get_market_history(const HistoryRequest& req) {
        
        std::optional<std::string> raw_json = perform_request(req);
        
        if (!raw_json) return std::nullopt;

        try {
            auto j = json::parse(*raw_json);
            
            // Navigate the specific Tradier structure: { "history": { "day": [...] } }
            // Note: We use .at() for safety; it throws if the key is missing.
            std::vector<Candle> candles = j.at("history").at("day").get<std::vector<Candle>>();
            
            return candles;
            
        } catch (const json::exception& e) {
            std::cerr << "JSON Parsing Error: " << e.what() << "\n";
            return std::nullopt;
        }
    }

private:
    std::string token_;

    // Helper to handle the raw CURL work (Moved to private helper for cleanliness)
    std::optional<std::string> perform_request(const HistoryRequest& req) {
        CURL* curl = curl_easy_init();
        if (!curl) return std::nullopt;

        std::string buffer;
        struct curl_slist* headers = nullptr;

        std::ostringstream url_ss;
        url_ss << HISTORY_API_URL
               << "symbol=" << req.symbol << "&interval=" << req.interval
               << "&start=" << req.start_date << "&end=" << req.end_date;
        
        curl_easy_setopt(curl, CURLOPT_URL, url_ss.str().c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

        headers = curl_slist_append(headers, "Accept: application/json");
        std::string auth = "Authorization: Bearer " + token_;
        headers = curl_slist_append(headers, auth.c_str());

        // Request fresh data and avoid 304 Not Modified responses
        headers = curl_slist_append(headers, "Cache-Control: no-cache");
        headers = curl_slist_append(headers, "Pragma: no-cache");
        headers = curl_slist_append(headers, "If-Modified-Since: 0");

        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CURLcode res = curl_easy_perform(curl);
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) return std::nullopt;
        return buffer;
    }

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t total_size = size * nmemb;
        ((std::string*)userp)->append((char*)contents, total_size);
        return total_size;
    }



};