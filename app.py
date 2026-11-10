
import os
import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"
CMC_API_KEY = os.getenv("CMC_API_KEY", "")

TIMEFRAMES = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h",
    "12h", "1d", "3d", "1w", "1M"
}


def get_json(url, params=None, headers=None):
    try:
        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=12
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


@app.route("/")
def home():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        "index.html"
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "app": "GANJ AI LIVE PRO",
        "version": "1.0"
    })


@app.route("/api/klines")
def klines():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    interval = request.args.get("interval", "4h")
    market = request.args.get("market", "futures").lower()
    limit = min(max(int(request.args.get("limit", 500)), 10), 1000)

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    if interval not in TIMEFRAMES:
        return jsonify({"error": "Invalid timeframe"}), 400

    base = BINANCE_FUTURES if market == "futures" else BINANCE_SPOT

    result = get_json(
        base + "/fapi/v1/klines"
        if market == "futures"
        else base + "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 502

    return jsonify(result)


@app.route("/api/ticker")
def ticker():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    market = request.args.get("market", "futures").lower()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    if market == "futures":
        url = BINANCE_FUTURES + "/fapi/v1/ticker/24hr"
    else:
        url = BINANCE_SPOT + "/api/v3/ticker/24hr"

    return jsonify(get_json(url, {"symbol": symbol}))


@app.route("/api/book")
def order_book():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    market = request.args.get("market", "futures").lower()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    if market == "futures":
        url = BINANCE_FUTURES + "/fapi/v1/depth"
    else:
        url = BINANCE_SPOT + "/api/v3/depth"

    return jsonify(get_json(url, {
        "symbol": symbol,
        "limit": 100
    }))


@app.route("/api/trades")
def trades():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    market = request.args.get("market", "futures").lower()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    url = (
        BINANCE_FUTURES + "/fapi/v1/aggTrades"
        if market == "futures"
        else BINANCE_SPOT + "/api/v3/aggTrades"
    )

    return jsonify(get_json(url, {
        "symbol": symbol,
        "limit": 100
    }))


@app.route("/api/open-interest")
def open_interest():
    symbol = request.args.get("symbol", "BTCUSDT").upper()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    return jsonify(get_json(
        BINANCE_FUTURES + "/fapi/v1/openInterest",
        {"symbol": symbol}
    ))


@app.route("/api/funding")
def funding():
    symbol = request.args.get("symbol", "BTCUSDT").upper()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    return jsonify(get_json(
        BINANCE_FUTURES + "/fapi/v1/premiumIndex",
        {"symbol": symbol}
    ))


@app.route("/api/long-short")
def long_short():
    symbol = request.args.get("symbol", "BTCUSDT").upper()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    return jsonify(get_json(
        BINANCE_FUTURES + "/futures/data/globalLongShortAccountRatio",
        {
            "symbol": symbol,
            "period": "4h",
            "limit": 30
        }
    ))


@app.route("/api/cmc")
def cmc():
    symbol = request.args.get("symbol", "").upper()

    if not symbol.isalnum():
        return jsonify({"error": "Invalid symbol"}), 400

    if not CMC_API_KEY:
        return jsonify({
            "available": False,
            "message": "CMC API key is not configured"
        })

    result = get_json(
        "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest",
        {"symbol": symbol},
        {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
