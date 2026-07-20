"""Analisi tecnica automatica + RAG sui libri di trading.

Dato uno o piu' ticker, scarica gli OHLCV dall'exchange (pubblico, senza API
key), calcola una batteria completa di indicatori, li interpreta e chiede al
RAG (i tuoi libri) cosa direbbero gli autori su questo scenario.

Uso:
  python signal_check.py BTC
  python signal_check.py BTC ETH SOL --tf 4h
  python signal_check.py BTC --tf 1d --no-llm       # solo dashboard + retrieval
  python signal_check.py BTC --exchange kraken --quote USD

NOTA: strumento informativo/di studio. Non e' un sistema di segnali ne'
consulenza finanziaria: la decisione operativa resta tua.
"""
import argparse
import os
import sys

import ccxt
import pandas as pd
import ta

import config
from query import retrieve, format_hits

CANDLES = 400  # abbastanza per EMA200 + Ichimoku


# ----------------------------------------------------------------- dati

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str) -> pd.DataFrame:
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLES)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


# ------------------------------------------------------------ indicatori

def compute_indicators(df: pd.DataFrame) -> dict:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    out = {}

    # --- Momentum ---
    out["rsi_14"] = ta.momentum.RSIIndicator(c, 14).rsi()
    st = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    out["stoch_k"], out["stoch_d"] = st.stoch(), st.stoch_signal()
    srsi = ta.momentum.StochRSIIndicator(c)
    out["stochrsi_k"], out["stochrsi_d"] = srsi.stochrsi_k() * 100, srsi.stochrsi_d() * 100
    out["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r()
    out["roc_12"] = ta.momentum.ROCIndicator(c, 12).roc()
    out["cci_20"] = ta.trend.CCIIndicator(h, l, c, 20).cci()

    # --- Trend ---
    macd = ta.trend.MACD(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = (
        macd.macd(), macd.macd_signal(), macd.macd_diff()
    )
    for n in (20, 50, 200):
        out[f"ema_{n}"] = ta.trend.EMAIndicator(c, n).ema_indicator()
        out[f"sma_{n}"] = ta.trend.SMAIndicator(c, n).sma_indicator()
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    out["adx"], out["di_plus"], out["di_minus"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    out["psar"] = ta.trend.PSARIndicator(h, l, c).psar()
    ichi = ta.trend.IchimokuIndicator(h, l)
    out["ichi_tenkan"] = ichi.ichimoku_conversion_line()
    out["ichi_kijun"] = ichi.ichimoku_base_line()
    out["ichi_span_a"] = ichi.ichimoku_a()
    out["ichi_span_b"] = ichi.ichimoku_b()

    # --- Volatilita' ---
    bb = ta.volatility.BollingerBands(c, 20, 2)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = (
        bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
    )
    out["bb_pct_b"] = bb.bollinger_pband()
    out["bb_width"] = bb.bollinger_wband()
    out["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    kc = ta.volatility.KeltnerChannel(h, l, c)
    out["kc_upper"], out["kc_lower"] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
    out["donchian_high"] = h.rolling(20).max()
    out["donchian_low"] = l.rolling(20).min()

    # --- Volume ---
    out["mfi_14"] = ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index()
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    out["vwap_14"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v, 14).volume_weighted_average_price()
    out["vol_sma_20"] = v.rolling(20).mean()
    out["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow()

    return {k: s.iloc[-1] for k, s in out.items()} | {
        "obv_slope_10": float(out["obv"].iloc[-1] - out["obv"].iloc[-11]),
        "macd_hist_prev": float(out["macd_hist"].iloc[-2]),
        "close": float(c.iloc[-1]),
        "volume": float(v.iloc[-1]),
        "change_1": float(c.pct_change(1).iloc[-1] * 100),
        "change_7": float(c.pct_change(7).iloc[-1] * 100),
        "change_30": float(c.pct_change(30).iloc[-1] * 100),
        "hi_52": float(c.rolling(min(len(c), 365)).max().iloc[-1]),
        "lo_52": float(c.rolling(min(len(c), 365)).min().iloc[-1]),
    }


# --------------------------------------------------------- interpretazione

def interpret(ind: dict) -> list[str]:
    """Traduce i numeri in stati sintetici (usati anche per il retrieval)."""
    s = []
    px = ind["close"]

    rsi = ind["rsi_14"]
    s.append(
        f"RSI(14) {rsi:.1f}: " + ("ipervenduto" if rsi < 30 else "ipercomprato" if rsi > 70 else "neutrale")
    )
    if ind["macd_hist"] > 0 > ind["macd_hist_prev"]:
        s.append("MACD: incrocio rialzista appena avvenuto")
    elif ind["macd_hist"] < 0 < ind["macd_hist_prev"]:
        s.append("MACD: incrocio ribassista appena avvenuto")
    else:
        s.append(f"MACD histogram {'positivo (momentum rialzista)' if ind['macd_hist'] > 0 else 'negativo (momentum ribassista)'}")

    above = [n for n in (20, 50, 200) if px > ind[f"ema_{n}"]]
    s.append(f"prezzo sopra EMA {above if above else 'nessuna'} (EMA200 {'sopra' if px < ind['ema_200'] else 'sotto'} il prezzo)")
    if ind["ema_50"] > ind["ema_200"]:
        s.append("EMA50 > EMA200 (assetto golden cross)")
    else:
        s.append("EMA50 < EMA200 (assetto death cross)")

    adx = ind["adx"]
    trend_dir = "rialzista" if ind["di_plus"] > ind["di_minus"] else "ribassista"
    s.append(f"ADX {adx:.1f}: trend {trend_dir} " + ("forte" if adx > 25 else "debole/laterale"))

    pb = ind["bb_pct_b"]
    s.append(f"Bollinger %B {pb:.2f}: " + ("sotto la banda inferiore" if pb < 0 else "sopra la banda superiore" if pb > 1 else "dentro le bande"))
    s.append(f"ATR(14) {ind['atr_14']:.4g} ({ind['atr_14'] / px * 100:.2f}% del prezzo)")

    if ind["stoch_k"] < 20:
        s.append(f"Stocastico {ind['stoch_k']:.0f}: ipervenduto")
    elif ind["stoch_k"] > 80:
        s.append(f"Stocastico {ind['stoch_k']:.0f}: ipercomprato")

    mfi = ind["mfi_14"]
    if mfi < 20 or mfi > 80:
        s.append(f"MFI {mfi:.0f}: " + ("ipervenduto (flussi in uscita esauriti?)" if mfi < 20 else "ipercomprato"))
    s.append(f"OBV in {'accumulazione' if ind['obv_slope_10'] > 0 else 'distribuzione'} (ultime 10 candele)")
    s.append(f"volume {'sopra' if ind['volume'] > ind['vol_sma_20'] else 'sotto'} la media 20")
    s.append(f"prezzo {'sopra' if px > ind['psar'] else 'sotto'} il Parabolic SAR")

    cloud_top = max(ind["ichi_span_a"], ind["ichi_span_b"])
    cloud_bot = min(ind["ichi_span_a"], ind["ichi_span_b"])
    s.append("Ichimoku: prezzo " + ("sopra la nuvola" if px > cloud_top else "sotto la nuvola" if px < cloud_bot else "dentro la nuvola"))

    s.append(f"distanza dal massimo di periodo {(px / ind['hi_52'] - 1) * 100:.1f}%, dal minimo {(px / ind['lo_52'] - 1) * 100:.1f}%")
    s.append(f"variazioni: {ind['change_1']:+.2f}% (1 candela), {ind['change_7']:+.2f}% (7), {ind['change_30']:+.2f}% (30)")
    return s


def dashboard(symbol: str, tf: str, ind: dict, states: list[str]) -> str:
    def fmt(x):
        return f"{x:,.2f}" if abs(x) >= 100 else f"{x:.6g}"
    lines = [
        f"=== {symbol} [{tf}] — close {fmt(ind['close'])} ===",
        "",
        f"Momentum : RSI {ind['rsi_14']:.1f} | Stoch {ind['stoch_k']:.0f}/{ind['stoch_d']:.0f} | StochRSI {ind['stochrsi_k']:.0f} | W%R {ind['williams_r']:.0f} | CCI {ind['cci_20']:.0f} | ROC {ind['roc_12']:.2f}",
        f"Trend    : MACD {ind['macd']:.4g} sig {ind['macd_signal']:.4g} hist {ind['macd_hist']:.4g} | ADX {ind['adx']:.1f} (+DI {ind['di_plus']:.1f} / -DI {ind['di_minus']:.1f})",
        f"Medie    : EMA20 {fmt(ind['ema_20'])} EMA50 {fmt(ind['ema_50'])} EMA200 {fmt(ind['ema_200'])} | SMA200 {fmt(ind['sma_200'])} | PSAR {fmt(ind['psar'])}",
        f"Ichimoku : tenkan {fmt(ind['ichi_tenkan'])} kijun {fmt(ind['ichi_kijun'])} | nuvola {fmt(min(ind['ichi_span_a'], ind['ichi_span_b']))}–{fmt(max(ind['ichi_span_a'], ind['ichi_span_b']))}",
        f"Volatil. : BB {fmt(ind['bb_lower'])}–{fmt(ind['bb_upper'])} (%B {ind['bb_pct_b']:.2f}, width {ind['bb_width']:.2f}) | ATR {fmt(ind['atr_14'])} | Donchian {fmt(ind['donchian_low'])}–{fmt(ind['donchian_high'])}",
        f"Volume   : MFI {ind['mfi_14']:.0f} | CMF {ind['cmf_20']:.3f} | VWAP14 {fmt(ind['vwap_14'])} | vol/media20 {ind['volume'] / ind['vol_sma_20']:.2f}x",
        "",
        "Lettura sintetica:",
        *[f"  - {x}" for x in states],
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------- RAG

def ask_books(symbol, tf, ind, states, k, use_llm):
    scenario = "; ".join(states)
    retrieval_q = (
        f"Setup e criteri operativi per questo scenario: {scenario}. "
        "Condizioni di ingresso, conferme, stop loss, position sizing, gestione del rischio."
    )
    hits = retrieve(retrieval_q, k)

    if not use_llm:
        print("\n--- Passaggi piu' pertinenti dai libri ---\n")
        print(format_hits(hits))
        return

    import anthropic

    context = format_hits(hits)
    prompt = (
        f"SCENARIO TECNICO {symbol} timeframe {tf}:\n"
        + "\n".join(f"- {x}" for x in states)
        + f"\n\nVALORI GREZZI: { {k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in ind.items()} }\n\n"
        f"ESTRATTI DAI LIBRI:\n\n{context}\n\n"
        "DOMANDA: In base ESCLUSIVAMENTE agli estratti, come inquadrerebbero gli autori "
        "questo scenario? Quali setup si avvicinano, quali conferme mancherebbero, dove "
        "collocherebbero stop e target (usa l'ATR se pertinente), e che criteri di position "
        "sizing suggerirebbero? Cita le fonti [file, pagina]. Concludi ricordando in una riga "
        "che non e' un consiglio finanziario."
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2500,
        system=(
            "Sei un analista tecnico esperto. Rispondi solo in base agli estratti forniti, "
            "citando sempre [nome_file, pagina]. Non inventare contenuti non presenti."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n--- Analisi basata sui tuoi libri ---\n")
    text = next((block.text for block in msg.content if block.type == "text"), None)
    print(text)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+", help="es: BTC ETH SOL, oppure BTC/USDT")
    ap.add_argument("--tf", default="4h", help="timeframe (1m 5m 15m 1h 4h 1d ...)")
    ap.add_argument("--quote", default="USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--k", type=int, default=8, help="chunk da recuperare")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    use_llm = not args.no_llm and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not args.no_llm and not use_llm:
        print("(ANTHROPIC_API_KEY non impostata: mostro dashboard + retrieval)\n")

    for t in args.tickers:
        symbol = t.upper() if "/" in t else f"{t.upper()}/{args.quote}"
        try:
            df = fetch_ohlcv(args.exchange, symbol, args.tf)
        except Exception as e:
            print(f"[errore] {symbol}: {e}")
            continue
        if len(df) < 60:
            print(f"[skip] {symbol}: dati insufficienti ({len(df)} candele)")
            continue

        ind = compute_indicators(df)
        states = interpret(ind)
        print("\n" + dashboard(symbol, args.tf, ind, states))
        ask_books(symbol, args.tf, ind, states, args.k, use_llm)
        print("\n" + "=" * 70)

    print("\nStrumento informativo: non costituisce consulenza finanziaria.")


if __name__ == "__main__":
    main()
