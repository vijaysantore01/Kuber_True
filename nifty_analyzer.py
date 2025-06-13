# nifty_analyzer.py
import os, sys, time, math
import datetime as dt
import zoneinfo
import pandas as pd
import logging
import pandas_ta as ta
from typing import Dict, Any, Tuple, List

# Import the TrueData API client instance
from truedata_api_client import true_data_client

# --- GLOBAL CONFIGURATION FOR ANALYZER ---
SYMBOL = "NIFTY"  # The underlying symbol for analysis (e.g., "NIFTY")
FIXED_ATR_LOW_THRESHOLD = 25.0  # Example threshold, can be made dynamic later


# --- ANALYTICAL HELPER FUNCTIONS ---

def calculate_pcr(df: pd.DataFrame) -> float:
    """Calculates Put-Call Ratio from filtered option chain DataFrame."""
    if df.empty:
        logging.warning("[calculate_pcr] Input DataFrame is empty. Returning 0.0.")
        return 0.0
    try:
        ce_sum = df.CE_OI.sum()
        pe_sum = df.PE_OI.sum()
        pcr = round(pe_sum / ce_sum, 2) if ce_sum > 0 else 0.0
        logging.debug(f"[calculate_pcr] CE_OI.sum: {ce_sum}, PE_OI.sum: {pe_sum}, PCR: {pcr}")
        return pcr
    except Exception as e:
        logging.error(f"[calculate_pcr] Error calculating PCR: {e}", exc_info=True)
        return 0.0


def calculate_option_skew(df: pd.DataFrame, spot: float) -> Tuple[float, float, float, int]:
    """Calculates option skew and returns ATM strike LTPs from filtered option chain DataFrame."""
    if df.empty:
        logging.warning("[calculate_option_skew] Input DataFrame is empty. Returning zeros.")
        return 0.0, 0.0, 0.0, 0
    try:
        # Find ATM strike closest to spot
        idx = (df.Strike - spot).abs().idxmin()
        r = df.loc[idx]  # Get the row for the ATM strike

        ce_ltp = r.get("CE_LTP", 0.0)
        pe_ltp = r.get("PE_LTP", 0.0)
        strike = r.get("Strike", 0)

        skew_val = round(ce_ltp - pe_ltp, 2)
        logging.debug(
            f"[calculate_option_skew] ATM Strike: {strike}, CE_LTP: {ce_ltp}, PE_LTP: {pe_ltp}, Skew: {skew_val}")
        return skew_val, float(ce_ltp), float(pe_ltp), int(strike)
    except Exception as e:
        logging.error(f"[calculate_option_skew] Error calculating option skew: {e}", exc_info=True)
        return 0.0, 0.0, 0.0, 0


def calculate_max_pain(df: pd.DataFrame) -> Tuple[int, float, List[int], List[float]]:
    """
    Calculates Max Pain price for options and the associated losses for plotting.
    Returns: max_pain_strike, max_pain_loss_val, list_of_strikes, list_of_losses.
    """
    if df.empty:
        logging.warning("[calculate_max_pain] Input DataFrame is empty. Returning 0, 0.0, [], [].")
        return 0, 0.0, [], []
    strikes = df.Strike.values
    if not strikes.size:
        logging.warning("[calculate_max_pain] No strike prices found in DataFrame. Returning 0, 0.0, [], [].")
        return 0, 0.0, [], []
    try:
        # Calculate total loss for both CE and PE writers at each strike price
        # This also needs to return the full losses array for plotting
        losses = [((s - strikes).clip(0) * df.PE_OI).sum() + ((strikes - s).clip(0) * df.CE_OI).sum() for s in strikes]

        if not losses:
            logging.warning("[calculate_max_pain] No losses calculated. Returning 0, 0.0, [], [].")
            return 0, 0.0, [], []

        # Max Pain is the strike price where option writers face minimum total loss
        min_loss_index = pd.Series(losses).idxmin()
        max_pain_strike = int(strikes[min_loss_index])
        max_pain_loss_val = float(losses[min_loss_index])

        logging.debug(
            f"[calculate_max_pain] Strikes: {strikes}, Losses: {losses}, Max Pain Index: {min_loss_index}, Max Pain Strike: {max_pain_strike}")

        # Return max pain strike, max pain loss, and the full losses data for plotting
        return max_pain_strike, max_pain_loss_val, list(strikes), [float(l) for l in losses]
    except Exception as e:
        logging.error(f"[calculate_max_pain] Error calculating Max Pain: {e}", exc_info=True)
        return 0, 0.0, [], []


def summarize_sentiment(df: pd.DataFrame) -> Tuple[str, int, int]:
    """Summarizes bullish/bearish sentiment based on OI bias from filtered option chain DataFrame."""
    if df.empty:
        logging.warning("[summarize_sentiment] Input DataFrame is empty. Returning Neutral.")
        return "⚖️ Neutral", 0, 0
    cnt = df.Bias.value_counts()
    bullish_count = cnt.get("Bullish", 0)
    bearish_count = cnt.get("Bearish", 0)

    if bullish_count > bearish_count:
        label = "📈 Bullish"
    elif bearish_count > bullish_count:
        label = "📉 Bearish"
    else:
        label = "⚖️ Neutral"

    logging.debug(f"[summarize_sentiment] Bullish: {bullish_count}, Bearish: {bearish_count}, Label: {label}")
    return label, bullish_count, bearish_count


def calculate_technical_indicators(df_bars: pd.DataFrame, interval_minutes: int) -> Dict[str, Any]:
    """
    Calculates various technical indicators from a DataFrame of bars.
    Returns individual scalar values and indicator history lists.
    """
    logging.debug(
        f"[calculate_technical_indicators] Input df_bars shape: {df_bars.shape if df_bars is not None else 'None'}")

    if df_bars is None or df_bars.empty:
        logging.warning("[calculate_technical_indicators] Input DataFrame is empty. Returning default indicators.")
        return {
            "atr": 0.0, "ma_15m": 0.0, "ma_50_ma": 0.0,
            "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
            "supertrend": 0.0, "adx": 0.0, "vwap": 0.0,
            "bb_lower": 0.0, "bb_middle": 0.0, "bb_upper": 0.0,
            "atr_bands_lower": 0.0, "atr_bands_upper": 0.0,
            "closes_history": [], "highs_history": [], "lows_history": [], "rsi_history": []
        }

    # Ensure required columns are present and numeric
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_bars.columns and pd.api.types.is_numeric_dtype(df_bars[col]) for col in required_cols):
        logging.error(
            f"[calculate_technical_indicators] Missing or non-numeric required columns: {required_cols}. Actual columns: {df_bars.columns.tolist()}")
        return {
            "atr": 0.0, "ma_15m": 0.0, "ma_50_ma": 0.0,
            "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
            "supertrend": 0.0, "adx": 0.0, "vwap": 0.0,
            "bb_lower": 0.0, "bb_middle": 0.0, "bb_upper": 0.0,
            "atr_bands_lower": 0.0, "atr_bands_upper": 0.0,
            "closes_history": [], "highs_history": [], "lows_history": [], "rsi_history": []
        }

    # Ensure enough data points for calculations (adjust lengths as needed)
    # ATR, RSI (14), BB (20), ADX (14), ST (10). Max of these is 20 for basic setup.
    min_bars_for_indicators = max(14, 20, 30)  # A buffer is good
    if len(df_bars) < min_bars_for_indicators:
        logging.warning(
            f"[calculate_technical_indicators] Not enough data ({len(df_bars)} bars) for full indicator calculation (min {min_bars_for_indicators} needed).")
        # Ensure 'close', 'high', 'low' are still in df_bars for history lists before returning.
        return {
            "atr": 0.0, "ma_15m": 0.0, "ma_50_ma": 0.0,
            "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
            "supertrend": 0.0, "adx": 0.0, "vwap": 0.0,
            "bb_lower": 0.0, "bb_middle": 0.0, "bb_upper": 0.0,
            "atr_bands_lower": 0.0, "atr_bands_upper": 0.0,
            "closes_history": df_bars['close'].to_list() if 'close' in df_bars.columns else [],
            "highs_history": df_bars['high'].to_list() if 'high' in df_bars.columns else [],
            "lows_history": df_bars['low'].to_list() if 'low' in df_bars.columns else [],
            "rsi_history": []
        }

    # Calculate indicators using pandas_ta (appends columns directly)
    # NEW: Set 'time' column as DataFrame index for pandas_ta methods that require it
    df_processed_ta = df_bars.copy()
    if 'time' in df_processed_ta.columns and pd.api.types.is_datetime64_any_dtype(df_processed_ta['time']):
        df_processed_ta = df_processed_ta.set_index('time', drop=False)
        logging.debug("[calculate_technical_indicators] 'time' column set as DataFrame index.")
    else:
        logging.warning(
            "[calculate_technical_indicators] 'time' column not suitable for index (missing or not datetime). VWAP/other time-based indicators might fail.")

    # Calculate indicators using pandas_ta (appends columns directly)
    df_processed_ta.ta.atr(append=True)  # ATR (default 14)
    df_processed_ta.ta.sma(close='close', length=3, append=True)  # SMA_3 (for 15m MA from 5m bars)
    df_processed_ta.ta.sma(close='close', length=50, append=True)  # SMA_50
    df_processed_ta.ta.rsi(append=True)  # RSI (default 14)
    df_processed_ta.ta.macd(append=True)  # MACD (default 12,26,9)
    df_processed_ta.ta.stoch(append=True)  # Stochastic (default 14,3,3)

    # NEW INDICATORS
    df_processed_ta.ta.supertrend(high='high', low='low', close='close', append=True, length=10,
                                  multiplier=3)  # SuperTrend (default 10,3)
    df_processed_ta.ta.adx(high='high', low='low', close='close', append=True, length=14)  # ADX (default 14)
    # VWAP: requires high, low, close, volume. Appends 'VWAP'
    # VWAP requires DatetimeIndex for grouping by anchor.
    # The check for pd.api.types.is_datetime64_any_dtype(df_processed_ta.index) handles this.
    if pd.api.types.is_datetime64_any_dtype(df_processed_ta.index):
        df_processed_ta.ta.vwap(high='high', low='low', close='close', volume='volume', append=True)
    else:
        logging.warning(
            "[calculate_technical_indicators] VWAP not calculated: Index is not DatetimeIndex or is not suitable. Current index type: %s",
            type(df_processed_ta.index))
        df_processed_ta['VWAP'] = pd.NA

    df_processed_ta.ta.bbands(close='close', append=True, length=20, std=2)  # Bollinger Bands (default 20,2)

    # ATR Bands are often calculated manually using price +/- multiple of ATR.
    # We will use ATRr_14 (from df_processed_ta.ta.atr()) to calculate these bands.
    atr_val_temp = df_processed_ta['ATRr_14'].iloc[-1] if 'ATRr_14' in df_processed_ta.columns and not df_processed_ta[
        'ATRr_14'].empty and pd.notna(df_processed_ta['ATRr_14'].iloc[-1]) else 0.0
    last_close = df_processed_ta['close'].iloc[-1] if 'close' in df_processed_ta.columns and not df_processed_ta[
        'close'].empty and pd.notna(df_processed_ta['close'].iloc[-1]) else 0.0

    if atr_val_temp > 0 and last_close > 0:
        atr_bands_lower_val = last_close - (atr_val_temp * 1.5)  # Example: 1.5 times ATR below close
        atr_bands_upper_val = last_close + (atr_val_temp * 1.5)  # Example: 1.5 times ATR above close
    else:
        atr_bands_lower_val = 0.0
        atr_bands_upper_val = 0.0

    # Drop any rows with NaN values resulting from indicator calculations
    # Log shape before and after dropna
    logging.debug(f"[calculate_technical_indicators] df_processed_ta shape before dropna: {df_processed_ta.shape}")
    # df_processed_ta = df_processed_ta.dropna() # REMOVED: This was too aggressive, removing all data.
    logging.debug(f"[calculate_technical_indicators] df_processed_ta shape after dropna: {df_processed_ta.shape}")

    if df_processed_ta.empty:
        logging.warning(
            "[calculate_technical_indicators] DataFrame became empty after dropping NaNs. Returning default indicators.")
        return {
            "atr": 0.0, "ma_15m": 0.0, "ma_50_ma": 0.0,
            "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
            "supertrend": 0.0, "adx": 0.0, "vwap": 0.0,
            "bb_lower": 0.0, "bb_middle": 0.0, "bb_upper": 0.0,
            "atr_bands_lower": 0.0, "atr_bands_upper": 0.0,
            "closes_history": df_bars['close'].to_list() if 'close' in df_bars.columns else [],
            "highs_history": df_bars['high'].to_list() if 'high' in df_bars.columns else [],
            "lows_history": df_bars['low'].to_list() if 'low' in df_bars.columns else [],
            "rsi_history": []
        }

    # Determine trend direction from 50-period MA
    trend_direction = "Neutral"
    if 'SMA_50' in df_processed_ta.columns and not df_processed_ta['SMA_50'].empty and pd.notna(
            df_processed_ta['SMA_50'].iloc[-1]):
        if df_processed_ta['close'].iloc[-1] > df_processed_ta['SMA_50'].iloc[-1]:
            trend_direction = "Bullish"
        elif df_processed_ta['close'].iloc[-1] < df_processed_ta['SMA_50'].iloc[-1]:
            trend_direction = "Bearish"
    else:
        logging.warning("SMA_50 column not found or empty for trend direction. Remains Neutral.")

    # Extract latest values (ensure they are scalars using .item() and defensive checks)
    latest_values = {
        "atr": df_processed_ta['ATRr_14'].iloc[-1].item() if 'ATRr_14' in df_processed_ta.columns and not
        df_processed_ta['ATRr_14'].empty and pd.notna(df_processed_ta['ATRr_14'].iloc[-1]) else 0.0,
        "ma_15m": df_processed_ta['SMA_3'].iloc[-1].item() if 'SMA_3' in df_processed_ta.columns and not
        df_processed_ta['SMA_3'].empty and pd.notna(df_processed_ta['SMA_3'].iloc[-1]) else 0.0,
        "ma_50_ma": df_processed_ta['SMA_50'].iloc[-1].item() if 'SMA_50' in df_processed_ta.columns and not
        df_processed_ta['SMA_50'].empty and pd.notna(df_processed_ta['SMA_50'].iloc[-1]) else 0.0,
        "rsi": df_processed_ta['RSI_14'].iloc[-1].item() if 'RSI_14' in df_processed_ta.columns and not df_processed_ta[
            'RSI_14'].empty and pd.notna(df_processed_ta['RSI_14'].iloc[-1]) else 0.0,
        "macd_line": df_processed_ta['MACD_12_26_9'].iloc[
            -1].item() if 'MACD_12_26_9' in df_processed_ta.columns and not df_processed_ta[
            'MACD_12_26_9'].empty and pd.notna(df_processed_ta['MACD_12_26_9'].iloc[-1]) else 0.0,
        "macd_signal": df_processed_ta['MACDs_12_26_9'].iloc[
            -1].item() if 'MACDs_12_26_9' in df_processed_ta.columns and not df_processed_ta[
            'MACDs_12_26_9'].empty and pd.notna(df_processed_ta['MACDs_12_26_9'].iloc[-1]) else 0.0,
        "macd_hist": df_processed_ta['MACDh_12_26_9'].iloc[
            -1].item() if 'MACDh_12_26_9' in df_processed_ta.columns and not df_processed_ta[
            'MACDh_12_26_9'].empty and pd.notna(df_processed_ta['MACDh_12_26_9'].iloc[-1]) else 0.0,
        "stoch_k": df_processed_ta['STOCHk_14_3_3'].iloc[
            -1].item() if 'STOCHk_14_3_3' in df_processed_ta.columns and not df_processed_ta[
            'STOCHk_14_3_3'].empty and pd.notna(df_processed_ta['STOCHk_14_3_3'].iloc[-1]) else 0.0,
        "stoch_d": df_processed_ta['STOCHd_14_3_3'].iloc[
            -1].item() if 'STOCHd_14_3_3' in df_processed_ta.columns and not df_processed_ta[
            'STOCHd_14_3_3'].empty and pd.notna(df_processed_ta['STOCHd_14_3_3'].iloc[-1]) else 0.0,
        "trend_direction": trend_direction,
        "closes_history": df_processed_ta['close'].to_list(),
        "highs_history": df_processed_ta['high'].to_list(),
        "lows_history": df_processed_ta['low'].to_list(),
        "rsi_history": df_processed_ta['RSI_14'].to_list() if 'RSI_14' in df_processed_ta.columns and not
        df_processed_ta['RSI_14'].empty else [],
        # NEW INDICATORS
        "supertrend": df_processed_ta['SUPERT_10_3.0'].iloc[
            -1].item() if 'SUPERT_10_3.0' in df_processed_ta.columns and not df_processed_ta[
            'SUPERT_10_3.0'].empty and pd.notna(df_processed_ta['SUPERT_10_3.0'].iloc[-1]) else 0.0,
        "adx": df_processed_ta['ADX_14'].iloc[-1].item() if 'ADX_14' in df_processed_ta.columns and not df_processed_ta[
            'ADX_14'].empty and pd.notna(df_processed_ta['ADX_14'].iloc[-1]) else 0.0,
        "vwap": df_processed_ta['VWAP'].iloc[-1].item() if 'VWAP' in df_processed_ta.columns and not df_processed_ta[
            'VWAP'].empty and pd.notna(df_processed_ta['VWAP'].iloc[-1]) else 0.0,
        "bb_lower": df_processed_ta['BBL_20_2.0'].iloc[-1].item() if 'BBL_20_2.0' in df_processed_ta.columns and not
        df_processed_ta['BBL_20_2.0'].empty and pd.notna(df_processed_ta['BBL_20_2.0'].iloc[-1]) else 0.0,
        "bb_middle": df_processed_ta['BBM_20_2.0'].iloc[-1].item() if 'BBM_20_2.0' in df_processed_ta.columns and not
        df_processed_ta['BBM_20_2.0'].empty and pd.notna(df_processed_ta['BBM_20_2.0'].iloc[-1]) else 0.0,
        "bb_upper": df_processed_ta['BBU_20_2.0'].iloc[-1].item() if 'BBU_20_2.0' in df_processed_ta.columns and not
        df_processed_ta['BBU_20_2.0'].empty and pd.notna(df_processed_ta['BBU_20_2.0'].iloc[-1]) else 0.0,
        "atr_bands_lower": atr_bands_lower_val,  # Use calculated value
        "atr_bands_upper": atr_bands_upper_val  # Use calculated value
    }
    return latest_values


def filter_strikes_near_spot(df: pd.DataFrame, spot: float, window: int = 5, step: int = 50) -> Tuple[
    pd.DataFrame, int]:
    """Filters option chain DataFrame to strikes near the spot price."""
    if df.empty:
        logging.warning("[filter_strikes_near_spot] Input DataFrame is empty. Returning empty and 0.")
        return pd.DataFrame(), 0

    # Ensure 'Strike' column exists and is numeric
    if 'Strike' not in df.columns or not pd.api.types.is_numeric_dtype(df['Strike']):
        logging.error("[filter_strikes_near_spot] 'Strike' column missing or not numeric. Cannot filter strikes.")
        return pd.DataFrame(), 0

    atm_strike_candidates = df.Strike.values
    if not atm_strike_candidates.size:
        logging.warning("[filter_strikes_near_spot] No strike prices found. Returning empty and 0.")
        return pd.DataFrame(), 0

    # Calculate ATM (At The Money) strike
    atm = int(round(spot / step) * step)
    if atm not in atm_strike_candidates:
        atm = min(atm_strike_candidates, key=lambda x: abs(x - spot))  # Find nearest if exact ATM not present

    # Filter strikes within the specified window around ATM
    low_strike_val, high_strike_val = atm - window * step, atm + window * step
    result_df = df[df.Strike.between(low_strike_val, high_strike_val)].copy()  # Use new variable names

    logging.debug(
        f"[filter_strikes_near_spot] Filtered {len(result_df)} strikes between {low_strike_val} and {high_strike_val}. ATM: {atm}")
    return result_df, int(atm)


def _extract_strike_metrics(df_filtered_oi: pd.DataFrame, strike_price: int) -> Dict[str, Any]:
    """
    Helper to extract specific OI/LTP/ChgOI metrics for a given strike price.
    Returns a dictionary of metrics for that strike, or 0.0 defaults if not found.
    """
    strike_row = df_filtered_oi[df_filtered_oi['Strike'] == strike_price]
    if not strike_row.empty:
        data = strike_row.iloc[0]  # Get the first (and likely only) matching row
        return {
            'strike': strike_price,
            'CE_OI': data.get('CE_OI', 0.0),
            'PE_OI': data.get('PE_OI', 0.0),
            'CE_ChgOI': data.get('CE_ChgOI', 0.0),
            'PE_ChgOI': data.get('PE_ChgOI', 0.0),
            'CE_LTP': data.get('CE_LTP', 0.0),
            'PE_LTP': data.get('PE_LTP', 0.0)
        }
    return {'strike': strike_price, 'CE_OI': 0.0, 'PE_OI': 0.0, 'CE_ChgOI': 0.0, 'PE_ChgOI': 0.0, 'CE_LTP': 0.0,
            'PE_LTP': 0.0}


def detect_divergence(closes: list, indicator_values: list, lookback_bars: int = 5) -> int:
    """
    Detects bullish or bearish divergence based on recent price and indicator trends.
    Returns: +1 for bullish, -1 for bearish, 0 for no clear divergence.
    """
    if len(closes) < lookback_bars or len(indicator_values) < lookback_bars:
        logging.debug(
            f"[detect_divergence] Not enough data: closes={len(closes)}, indicator={len(indicator_values)}, lookback={lookback_bars}")
        return 0

    recent_closes = closes[-lookback_bars:]
    recent_indicator = indicator_values[-lookback_bars:]

    # Bearish Divergence: Price higher highs, Indicator lower highs
    if recent_closes[-1] > recent_closes[0] and recent_indicator[-1] < recent_indicator[0]:
        price_trend_up = all(
            recent_closes[i] <= recent_closes[i + 1] for i in range(lookback_bars - 1))  # Price generally rising
        indicator_trend_down = all(recent_indicator[i] >= recent_indicator[i + 1] for i in
                                   range(lookback_bars - 1))  # Indicator generally falling
        if price_trend_up and indicator_trend_down:
            logging.debug("[detect_divergence] Bearish Divergence Detected: Price HH, Indicator LH")
            return -1

    # Bullish Divergence: Price lower lows, Indicator higher lows
    if recent_closes[-1] < recent_closes[0] and recent_indicator[-1] > recent_indicator[0]:
        price_trend_down = all(
            recent_closes[i] >= recent_closes[i + 1] for i in range(lookback_bars - 1))  # Price generally falling
        indicator_trend_up = all(recent_indicator[i] <= recent_indicator[i + 1] for i in
                                 range(lookback_bars - 1))  # Indicator generally rising
        if price_trend_down and indicator_trend_up:
            logging.debug("[detect_divergence] Bullish Divergence Detected: Price LL, Indicator HL")
            return 1

    logging.debug("[detect_divergence] No clear divergence detected.")
    return 0

def generate_final_suggestion(
        pcr: float, skew_val: float, spot: float, mp: int, atr_val: float, ma_15m: float, ATR_LOW_THRESHOLD: float,
        rsi_val: float, macd_vals: Tuple[float, float, float], stoch_vals: Tuple[float, float],
        daily_ma_val: float, volume_strength_score: int, divergence_score: int
) -> Tuple[str, int, int, int, int, int]:  # Added returns for individual scores
    """
    Generates a final trade suggestion based on a scoring system,
    now incorporating adaptive ATR, trend filter, and momentum indicators.
    Returns: suggestion, final_score, oi_score, price_score, momentum_score, additional_score
    """
    logging.info(f"[generate_final_suggestion] Called with pcr={pcr}, skew_val={skew_val}, spot={spot}, mp={mp}, "
                 f"atr_val={atr_val}, ma_15m={ma_15m}, ATR_LOW_THRESHOLD={ATR_LOW_THRESHOLD}, "
                 f"rsi_val={rsi_val}, macd_vals={macd_vals}, stoch_vals={stoch_vals}, "
                 f"daily_ma_val={daily_ma_val}, volume_strength_score={volume_strength_score}, "
                 f"divergence_score={divergence_score}")

    oi_score = 0
    if pcr < 0.90:
        oi_score -= 1
        logging.debug("[generate_final_suggestion] oi_score -- PCR < 0.90")
    elif pcr > 1.10:
        oi_score += 1
        logging.debug("[generate_final_suggestion] oi_score ++ PCR > 1.10")
    if skew_val < -3:
        oi_score -= 1
        logging.debug("[generate_final_suggestion] oi_score -- Skew < -3")
    elif skew_val > +3:
        oi_score += 1
        logging.debug("[generate_final_suggestion] oi_score ++ Skew > 3")
    if spot > mp + 20:  # Spot significantly above Max Pain (bearish bias from unwinding calls)
        oi_score -= 1
        logging.debug("[generate_final_suggestion] oi_score -- spot > mp+20")
    elif spot < mp - 20:  # Spot significantly below Max Pain (bullish bias from unwinding puts)
        oi_score += 1
        logging.debug("[generate_final_suggestion] oi_score ++ spot < mp-20")
    logging.debug(f"[generate_final_suggestion] oi_score: {oi_score}")

    price_score = 0
    if ma_15m != 0.0:  # Only score if 15m MA was calculated
        if spot < ma_15m:
            price_score -= 1  # Bearish if spot below 15m MA
            logging.debug("[generate_final_suggestion] price_score -- spot < ma_15m")
        elif spot > ma_15m:
            price_score += 1  # Bullish if spot above 15m MA
            logging.debug("[generate_final_suggestion] price_score ++ spot > ma_15m")
    logging.debug(f"[generate_final_suggestion] price_score: {price_score}")

    # --- Momentum Indicator Scoring ---
    momentum_score = 0
    if rsi_val != 0.0:  # Only score if RSI is valid
        if rsi_val > 60:
            momentum_score -= 1
            logging.debug("[generate_final_suggestion] momentum_score -- rsi_val > 60")
        elif rsi_val < 40:
            momentum_score += 1
            logging.debug("[generate_final_suggestion] momentum_score ++ rsi_val < 40")

    macd_line, signal_line, _ = macd_vals
    if macd_line != 0.0 or signal_line != 0.0:  # Only score if MACD values are valid
        if macd_line > signal_line:
            momentum_score += 1
            logging.debug("[generate_final_suggestion] momentum_score ++ macd_line > signal_line")
        elif macd_line < signal_line:
            momentum_score -= 1
            logging.debug("[generate_final_suggestion] momentum_score -- macd_line < signal_line")

    stoch_k, stoch_d = stoch_vals
    if stoch_k != 0.0 or stoch_d != 0.0:  # Only score if Stochastic values are valid
        if stoch_k > stoch_d:
            momentum_score += 1
            logging.debug("[generate_final_suggestion] momentum_score ++ stoch_k > stoch_d")
        elif stoch_k < stoch_d:
            momentum_score -= 1
            logging.debug("[generate_final_suggestion] momentum_score -- stoch_k < stoch_d")
    logging.debug(f"[generate_final_suggestion] momentum_score: {momentum_score}")

    # --- Additional Scoring Components ---
    additional_score = 0
    if daily_ma_val != 0.0:  # Only score if daily MA was successfully calculated
        if spot > daily_ma_val:
            additional_score += 1  # Bullish bias from daily MA
            logging.debug("[generate_final_suggestion] additional_score ++ spot > daily_ma_val")
        elif spot < daily_ma_val:
            additional_score -= 1  # Bearish bias from daily MA
            logging.debug("[generate_final_suggestion] additional_score -- spot < daily_ma_val")

    additional_score += volume_strength_score  # Will be 0 unless implemented
    logging.debug(f"[generate_final_suggestion] additional_score += volume_strength_score: {volume_strength_score}")
    additional_score += divergence_score  # Will be 0 unless implemented
    logging.debug(f"[generate_final_suggestion] additional_score += divergence_score: {divergence_score}")
    logging.debug(f"[generate_final_suggestion] additional_score: {additional_score}")

    # Final score is the sum of all component scores
    final_score = oi_score + price_score + momentum_score + additional_score
    logging.info(f"[generate_final_suggestion] Raw Final Score (pre-ATR filter): {final_score}")

    # Apply ATR filter logic from old bot (ATR will be 0.0 if not enough data)
    # If ATR is very low, it implies low volatility/thin market, so we might give a neutral suggestion.
    # The scoring logic remains the same for raw scores.
    if atr_val > ATR_LOW_THRESHOLD:
        logging.info("[generate_final_suggestion] ATR is above threshold. Considering full score.")
    else:
        logging.info(
            f"[generate_final_suggestion] ATR ({atr_val:.2f}) is below threshold ({ATR_LOW_THRESHOLD:.2f}). Suggestion might be more neutral due to low volatility.")
        # Optionally, you could penalize the score or force neutral here if ATR is too low.
        # For now, it just logs, and the score is already computed.

    # Decision logic based on final_score
    if final_score <= -2:
        suggestion = "BUY PUT"
        logging.info("[generate_final_suggestion] Suggestion = BUY PUT")
    elif final_score >= 2:
        suggestion = "BUY CALL"
        logging.info("[generate_final_suggestion] Suggestion = BUY CALL")
    else:
        suggestion = "Stay Neutral"
        logging.info("[generate_final_suggestion] Suggestion = Stay Neutral")

    logging.info(
        f"[generate_final_suggestion] Return: suggestion={suggestion}, final_score={final_score}, oi_score={oi_score}, price_score={price_score}, momentum_score={momentum_score}, additional_score={additional_score}")
    return suggestion, final_score, oi_score, price_score, momentum_score, additional_score


# # In nifty_analyzer.py, please ensure this function is present.
# # It should be defined at the top level, alongside other helper functions like calculate_pcr, detect_divergence, etc.
#
def final_suggestion_extended(score: int, skip_price: bool) -> Tuple[str, int]:
    """
    Determines the final trade suggestion (BUY CALL, BUY PUT, Stay Neutral)
    and calculates a confidence percentage based on the overall score.
    This function is called by dashboard_app.py directly.
    """
    # Max theoretical score for all factors contributing is 9 (3 for OI + 1 for Price + 3 for Momentum + 2 for Additional)
    max_theoretical_score_sum = 9

    # Calculate confidence as a percentage of the absolute score relative to max theoretical sum.
    # Ensure score_clamped is within expected range before calculating confidence.
    score_clamped = max(-max_theoretical_score_sum, min(score, max_theoretical_score_sum))

    # Avoid ZeroDivisionError if max_theoretical_score_sum is 0 (should not happen with current scoring)
    conf = int((abs(score_clamped) / max_theoretical_score_sum) * 100) if max_theoretical_score_sum > 0 else 0

    # Decision logic based on the overall score from generate_final_suggestion
    # The thresholds (2 and -2) are defined in generate_final_suggestion
    if score >= 2:
        return "BUY CALL", conf
    elif score <= -2:
        return "BUY PUT", conf
    else:
        return "Stay Neutral", conf

# In nifty_analyzer.py, replace the entire get_nifty_analysis method

# In nifty_analyzer.py, replace the entire get_nifty_analysis method

# In nifty_analyzer.py, please ensure this entire function is present.

def get_nifty_analysis() -> Dict[str, Any]:
    """
    Main function to fetch all required NIFTY data using TrueDataAPIClient
    and perform comprehensive analysis for the dashboard.
    """
    logging.info("[get_nifty_analysis] Starting comprehensive NIFTY analysis...")

    # 1. Get Live Spot Price
    # Uses TD_live client from truedata_api_client.py
    spot_price = true_data_client.get_live_spot_price("NIFTY")  # Assuming 'NIFTY' is the symbol for indices
    if spot_price == 0.0:
        logging.error("[get_nifty_analysis] Failed to get live spot price from TrueData. Aborting analysis.")
        return {}  # Return empty dict if critical data is missing

    # 2. Get Nearest Expiry Date
    # Uses TrueDataAPIClient.get_symbol_expiry_list
    current_expiry_dates = true_data_client.get_symbol_expiry_list(SYMBOL)
    current_expiry_str = None
    current_expiry_dt = None

    if current_expiry_dates:
        # TrueData expiry format is typically YYYY-MM-DD.
        # Need to find the nearest future expiry.
        today = dt.date.today()
        future_expiries = []
        for exp_str in current_expiry_dates:
            try:
                exp_dt = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()  # Parse TrueData's format YYYY-MM-DD
                if exp_dt >= today:
                    future_expiries.append(exp_dt)
            except ValueError:
                # Log a warning for each unparseable date but don't abort immediately
                logging.warning(f"[get_nifty_analysis] Could not parse expiry date string from TrueData: {exp_str}")
                continue  # Skip to next expiry date

        if future_expiries:
            future_expiries.sort()  # Sort to find the nearest
            current_expiry_dt = future_expiries[0]
            current_expiry_str = current_expiry_dt.strftime("%d-%b-%Y")  # Format for dashboard display
            logging.info(f"[get_nifty_analysis] Nearest expiry from TrueData: {current_expiry_str}")
        else:
            logging.error("[get_nifty_analysis] No future expiry dates found from TrueData. Aborting analysis.")
            return {}
    else:
        logging.error("[get_nifty_analysis] Failed to get expiry dates from TrueData. Aborting analysis.")
        return {}

    # 3. Get Live Option Chain Data
    # Uses TD_live client from truedata_api_client.py
    df_option_chain = true_data_client.get_live_option_chain_data(SYMBOL, current_expiry_dt)
    if df_option_chain.empty:
        logging.error("[get_nifty_analysis] Failed to get live option chain from TrueData. Aborting analysis.")
        return {}

    # 4. Filter Strikes Near Spot (Using existing logic)
    # The 'step' for Nifty strikes is typically 50
    NIFTY_STRIKE_STEP = 50
    df_filtered_oi, atm_strike = filter_strikes_near_spot(df_option_chain, spot_price, step=NIFTY_STRIKE_STEP)
    if df_filtered_oi.empty:
        logging.error("[get_nifty_analysis] Filtered OI data is empty. Aborting analysis.")
        return {}

    # NEW: Calculate 'Bias' column for option chain sentiment
    # This needs to be done AFTER the filtering, so PE_ChgOI and CE_ChgOI are relevant
    df_filtered_oi["Bias"] = df_filtered_oi.apply(
        lambda r: "Bullish" if r.PE_ChgOI > r.CE_ChgOI
        else ("Bearish" if r.CE_ChgOI > r.PE_ChgOI else "Neutral"),
        axis=1
    )

    # 5. Calculate OI Metrics (PCR, Skew, Max Pain, Sentiment)
    pcr = calculate_pcr(df_filtered_oi)
    skew_val, ce_ltp, pe_ltp, _ = calculate_option_skew(df_filtered_oi, spot_price)
    max_pain_strike, max_pain_loss, max_pain_strikes_list, max_pain_losses_list = calculate_max_pain(df_filtered_oi)
    oi_bias_label, oi_bull_strikes, oi_bear_strikes = summarize_sentiment(df_filtered_oi)

    # --- NEW: Extract OTM/ITM Option Chain Metrics ---
    otm_itm_metrics = {}

    # Calculate target strikes for OTM/ITM based on ATM and step
    # Calls: ITM below ATM, OTM above ATM
    # Puts: ITM above ATM, OTM below ATM

    # Call Strikes
    otm_itm_metrics['ATM'] = _extract_strike_metrics(df_filtered_oi, atm_strike)
    otm_itm_metrics['1 OTM Call'] = _extract_strike_metrics(df_filtered_oi, atm_strike + 1 * NIFTY_STRIKE_STEP)
    otm_itm_metrics['2 OTM Call'] = _extract_strike_metrics(df_filtered_oi, atm_strike + 2 * NIFTY_STRIKE_STEP)
    otm_itm_metrics['1 ITM Call'] = _extract_strike_metrics(df_filtered_oi, atm_strike - 1 * NIFTY_STRIKE_STEP)
    otm_itm_metrics['2 ITM Call'] = _extract_strike_metrics(df_filtered_oi, atm_strike - 2 * NIFTY_STRIKE_STEP)

    # Put Strikes (using corresponding Call OTM/ITM strikes, but labeled as Put OTM/ITM relative to spot)
    otm_itm_metrics['1 OTM Put'] = _extract_strike_metrics(df_filtered_oi, atm_strike - 1 * NIFTY_STRIKE_STEP)
    otm_itm_metrics['2 OTM Put'] = _extract_strike_metrics(df_filtered_oi, atm_strike - 2 * NIFTY_STRIKE_STEP)
    otm_itm_metrics['1 ITM Put'] = _extract_strike_metrics(df_filtered_oi, atm_strike + 1 * NIFTY_STRIKE_STEP)
    otm_itm_metrics['2 ITM Put'] = _extract_strike_metrics(df_filtered_oi, atm_strike + 2 * NIFTY_STRIKE_STEP)

    # 6. Get Historical Daily Bars (for Daily MA)
    # Uses TD_hist client from truedata_api_client.py
    # Use "NIFTY 50" as symbol for historical EOD data, as per docs/Postman.
    df_daily_bars = true_data_client.get_historical_daily_bars("NIFTY 50", days_back=90)
    daily_ma_val = 0.0
    if not df_daily_bars.empty and len(df_daily_bars) >= 50:
        daily_ma_val = df_daily_bars['close'].iloc[-50:].mean().item()  # Ensure scalar
        prev_close = df_daily_bars['close'].iloc[-1].item()  # Latest close as prev_close
    else:
        logging.warning("[get_nifty_analysis] Insufficient daily bars for Daily MA calculation. Daily MA will be 0.0.")
        prev_close = 0.0  # Default prev_close if no data

    # 7. Get Historical Intraday Bars (for Technical Indicators)
    # Uses TD_hist client from truedata_api_client.py
    # Use "NIFTY 50" as symbol for historical intraday data, as per docs/Postman.
    df_intraday_bars = true_data_client.get_historical_intraday_bars("NIFTY 50", interval_minutes=5,
                                                                     days_back=7)  # 7 days of 5-min bars
    tech_indicators = calculate_technical_indicators(df_intraday_bars, interval_minutes=5)

    # 8. Calculate Divergence Score (needs history from technical indicators)
    divergence_score = detect_divergence(tech_indicators['closes_history'], tech_indicators['rsi_history'])

    # 9. Placeholder for Volume Strength Score (requires specific implementation)
    volume_strength_score = 0  # Implement this if you have a rule for it

    # 10. Generate Final Suggestion and Individual Scores
    suggestion, final_score, oi_score_raw, price_score_raw, momentum_score_raw, additional_score_raw = generate_final_suggestion(
        pcr=pcr, skew_val=skew_val, spot=spot_price, mp=max_pain_strike,
        atr_val=tech_indicators['atr'], ma_15m=tech_indicators['ma_15m'], ATR_LOW_THRESHOLD=FIXED_ATR_LOW_THRESHOLD,
        rsi_val=tech_indicators['rsi'],
        macd_vals=(tech_indicators['macd_line'], tech_indicators['macd_signal'], tech_indicators['macd_hist']),
        stoch_vals=(tech_indicators['stoch_k'], tech_indicators['stoch_d']),
        daily_ma_val=daily_ma_val, volume_strength_score=volume_strength_score, divergence_score=divergence_score
    )

    # 11. Compile all data for the dashboard
    analysis_data = {
        "timestamp": dt.datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST"),
        "spot_price": spot_price,
        "expiry": current_expiry_str,
        "pcr": pcr,
        "skew": skew_val,
        "ce_ltp": ce_ltp,
        "pe_ltp": pe_ltp,
        "max_pain": max_pain_strike,
        "max_pain_loss": max_pain_loss,
        "max_pain_strikes_list": max_pain_strikes_list,  # For Max Pain Graph
        "max_pain_losses_list": max_pain_losses_list,  # For Max Pain Graph
        "oi_bias_label": oi_bias_label,
        "oi_bull_strikes": oi_bull_strikes,
        "oi_bear_strikes": oi_bear_strikes,
        "daily_ma": daily_ma_val,
        "vol_strength_score": volume_strength_score,
        "divergence_score": divergence_score,

        # Technical Indicator values
        "atr": tech_indicators['atr'],
        "atr_threshold": FIXED_ATR_LOW_THRESHOLD,  # Currently fixed
        "ma_15m": tech_indicators['ma_15m'],
        "ma_50_5m": tech_indicators['ma_50_ma'],  # Renamed for clarity on 5m data
        "trend_direction": tech_indicators['trend_direction'],
        "rsi": tech_indicators['rsi'],
        "macd_line": tech_indicators['macd_line'],
        "macd_signal": tech_indicators['macd_signal'],
        "macd_hist": tech_indicators['macd_hist'],
        "stoch_k": tech_indicators['stoch_k'],
        "stoch_d": tech_indicators['stoch_d'],
        "supertrend": tech_indicators['supertrend'],  # NEW
        "adx": tech_indicators['adx'],  # NEW
        "vwap": tech_indicators['vwap'],  # NEW
        "bb_lower": tech_indicators['bb_lower'],  # NEW
        "bb_middle": tech_indicators['bb_middle'],  # NEW
        "bb_upper": tech_indicators['bb_upper'],  # NEW
        "atr_bands_lower": tech_indicators['atr_bands_lower'],  # NEW
        "atr_bands_upper": tech_indicators['atr_bands_upper'],  # NEW

        # Decision Scores
        "final_suggestion": suggestion,
        "final_score": final_score,  # This is the raw final score from generate_final_suggestion
        "oi_score_raw": oi_score_raw,
        "price_score_raw": price_score_raw,
        "momentum_score_raw": momentum_score_raw,
        "additional_score_raw": additional_score_raw,

        # Data for OI Chart (DataFrame itself)
        "df_filtered_oi_data": df_filtered_oi,

        # NEW: OTM/ITM Option Chain Metrics
        "otm_itm_metrics": otm_itm_metrics,  # Add the prepared OTM/ITM data

        # Placeholders for future / derived values needed by dashboard
        "trade_confidence": 0,  # Will be set by dashboard_app from final_suggestion_extended
        "market_sentiment": "Neutral",  # Will be set by dashboard_app from final_score
        "premium_at_signal": 0.0,  # Will be set by dashboard_app
        "active_trade": None,  # From Streamlit session state
        "prev_close": prev_close  # From daily bars in this function
    }
    logging.info("[get_nifty_analysis] Comprehensive NIFTY analysis completed.")
    return analysis_data