import streamlit as st
import time
import pandas as pd
import logging
import os
import datetime as dt
import zoneinfo  # Ensure zoneinfo is imported
from typing import Union
from pandas.api.types import is_scalar

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="NIFTY Bot Real-time Dashboard")

# --- Laxmi Kuber Mantra ---
st.markdown(
    "<h1 style='text-align: center; color: #DAA520; font-family: \"Georgia\", serif; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>ॐ श्रीं ह्रीं क्लीं श्रीं क्लीं वित्तेश्वराय नमः॥</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #555555; font-size: 14px; margin-top: -10px;'>Om Shreem Hreem Kleem Shreem Kleem Vitteshvaraya Namah॥</p>",
    unsafe_allow_html=True)
st.markdown("---")

# --- Initial Logging Setup ---
if 'logging_setup_complete' not in st.session_state:
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Set to logging.DEBUG in nifty_analyzer.py for detailed bot logs
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)
        st.session_state.logging_setup_complete = True
    except Exception as e:
        print(f"WARNING: Could not configure Streamlit logging handlers due to: {e}", file=sys.stderr)

# --- Import Core Analysis Logic from nifty_analyzer.py ---
try:
    # We now import only what's necessary for the UI from nifty_analyzer.py
    from nifty_analyzer import get_nifty_analysis
    from nifty_analyzer import generate_final_suggestion  # Needed for its return types
    from nifty_analyzer import final_suggestion_extended  # Needed for confidence calculation
    from nifty_analyzer import SYMBOL  # Global symbol constant (if needed for display)
    from nifty_analyzer import FIXED_ATR_LOW_THRESHOLD  # Global ATR threshold (if needed for display)

    # These state variables are now managed within Streamlit's session state.
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'active_trade' not in st.session_state:  # This will be managed by future trade execution logic
        st.session_state.active_trade = None  # Placeholder for now

except ImportError as e:
    st.error(
        f"Error importing bot logic: {e}. Please ensure 'nifty_analyzer.py' and 'truedata_api_client.py' are in the same directory and all dependencies are installed.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during initial Streamlit setup: {e}")
    st.stop()

# --- Streamlit UI Controls ---
REFRESH_INTERVAL_SECONDS = st.sidebar.slider("Auto-Refresh Interval (seconds)", 5, 60, 15)
st.sidebar.markdown("---")
# Clock placeholder in the sidebar
clock_placeholder = st.sidebar.empty()


# Function to run the bot logic and update the dashboard
def update_dashboard():
    try:
        # Update the clock
        current_time_ist = dt.datetime.now(dt.timezone.utc).astimezone(zoneinfo.ZoneInfo("Asia/Kolkata")).strftime(
            "%H:%M:%S IST")
        clock_placeholder.markdown(f"**Current Time:**\n\n### {current_time_ist}")

        logging.info("[update_dashboard] Starting dashboard data refresh...")

        # Call the main analysis function to get all computed data
        analysis_data = get_nifty_analysis()

        if not analysis_data:
            st.warning("Failed to fetch or analyze NIFTY data. Displaying last known data or defaults.")
            logging.warning("[update_dashboard] Analysis data is empty. Rerunning.")
            time.sleep(REFRESH_INTERVAL_SECONDS)
            st.rerun()
            return

        # --- History Adding Logic ---
        # Only add to history if it's a BUY CALL or BUY PUT suggestion
        if analysis_data['final_suggestion'] in ["BUY CALL", "BUY PUT"]:
            st.session_state.analysis_history.append(analysis_data)
            if len(st.session_state.analysis_history) > 10:  # Keep last 10 entries for history
                st.session_state.analysis_history.pop(0)

        # --- Streamlit UI Layout (Single Page View) ---

        # Row 1: Spot Price, Final Suggestion, Overall Confidence
        col_main_spot, col_main_suggestion, col_main_confidence = st.columns([1, 2, 1])
        with col_main_spot:
            st.metric(label="Current Spot Price", value=f"₹{analysis_data['spot_price']:.2f}")

        with col_main_suggestion:
            suggestion_color = 'green' if 'BUY CALL' in analysis_data['final_suggestion'] else \
                'red' if 'BUY PUT' in analysis_data['final_suggestion'] else 'orange'
            suggestion_icon = "📈" if 'BUY CALL' in analysis_data['final_suggestion'] else \
                "📉" if 'BUY PUT' in analysis_data['final_suggestion'] else "⚖️"
            st.markdown(
                f"<div style='background-color:#F0F2F6; padding: 10px; border-radius: 5px; text-align: center; font-family: \"Verdana\", Arial, sans-serif;'>"
                f"<h3 style='color:{suggestion_color}; font-size:28px; margin-bottom: 0px;'>{suggestion_icon} {analysis_data['final_suggestion']}</h3>"
                f"<p style='color:grey; font-size:15px; margin-top: 5px;'>Overall Market Sentiment: <b>{analysis_data['market_sentiment']}</b></p>"
                f"</div>", unsafe_allow_html=True
            )

        with col_main_confidence:
            # final_suggestion_extended calculates confidence based on final_score.
            # We call it here directly for confidence calculation as it's a simple helper
            _sugg, _conf = final_suggestion_extended(analysis_data['final_score'], analysis_data['atr'] == 0.0)
            st.metric(label="Overall Confidence", value=f"{_conf}%")
            # NEW: Display Supporting Indicators Score
            if 'num_supporting_indicators' in analysis_data and 'total_indicators_evaluated' in analysis_data:
                st.markdown(
                    f"**Support Score:** {analysis_data['num_supporting_indicators']}/"
                    f"{analysis_data['total_indicators_evaluated']}", unsafe_allow_html=True
                )

        st.markdown("---")  # Separator

        # Row 2: Active Trade Status & PnL (placeholders for now, requires trade execution logic)
        col_trade_status, col_pnl_display = st.columns([2, 1])
        if st.session_state.active_trade:  # Using st.session_state for active_trade
            # This logic needs to be fully integrated with a live trading system's PnL tracking
            # It's currently a placeholder based on previous code.
            with col_trade_status:
                st.markdown("<h3>Active Trade: <span style='color:green;'>Active (Placeholder)</span></h3>",
                            unsafe_allow_html=True)
            with col_pnl_display:
                st.metric(label="Live PnL (Demo)", value="₹100.00", delta="5.00%")  # Dummy PnL for demo
        else:
            with col_trade_status:
                st.markdown("<h3>Active Trade: <span style='color:orange;'>None</span></h3>", unsafe_allow_html=True)
            with col_pnl_display:
                st.empty()  # Clear PnL metrics if no active trade

        st.markdown("---")  # Separator

        # Row 3: Score Breakdown
        st.subheader("Decision Score Breakdown")
        col_oi_score, col_price_score, col_momentum_score, col_additional_score = st.columns(4)

        def get_score_color(score):
            if score > 0: return "green"
            if score < 0: return "red"
            return "orange"

        with col_oi_score:
            score_color = get_score_color(analysis_data['oi_score_raw'])
            st.markdown(
                f"**OI Score:** <span style='color:{score_color}; font-size:18px;'>{analysis_data['oi_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>PCR, Skew, Max Pain</i></small>", unsafe_allow_html=True)

        with col_price_score:
            score_color = get_score_color(analysis_data['price_score_raw'])
            st.markdown(
                f"**Price Score:** <span style='color:{score_color}; font-size:18px;'>{analysis_data['price_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>15m MA Comparison</i></small>", unsafe_allow_html=True)

        with col_momentum_score:
            score_color = get_score_color(analysis_data['momentum_score_raw'])
            st.markdown(
                f"**Momentum Score:** <span style='color:{score_color}; font-size:18px;'>{analysis_data['momentum_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>RSI, MACD, Stoch</i></small>", unsafe_allow_html=True)

        with col_additional_score:
            score_color = get_score_color(analysis_data['additional_score_raw'])
            st.markdown(
                f"**Additional Score:** <span style='color:{score_color}; font-size:18px;'>{analysis_data['additional_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>Daily MA, Vol, Div</i></small>", unsafe_allow_html=True)

        st.markdown("---")  # Separator

        # Row 4: Detailed Metrics Tables (in expanders for cleanliness)
        with st.expander("📊 Detailed Market Metrics"):
            col_detail_oi, col_detail_price, col_detail_indicators = st.columns(3)

            with col_detail_oi:
                oi_data_display = {
                    "Metric": ["PCR", "Skew", "Max Pain", "OI Bias Label", "Bullish Strikes", "Bearish Strikes",
                               "CE LTP", "PE LTP"],
                    "Value": [
                        f"{analysis_data['pcr']:.2f}",
                        f"{analysis_data['skew']:.2f}",
                        f"{analysis_data['max_pain']}",
                        analysis_data['oi_bias_label'],
                        f"{analysis_data['oi_bull_strikes']}",
                        f"{analysis_data['oi_bear_strikes']}",
                        f"₹{analysis_data['ce_ltp']:.2f}",
                        f"₹{analysis_data['pe_ltp']:.2f}"
                    ]
                }
                st.subheader("Open Interest")
                st.table(pd.DataFrame(oi_data_display))

            with col_detail_price:
                price_data_display = {
                    "Metric": ["ATR (5m)", "ATR Threshold", "15m MA", "50-period 5m MA", "Trend Direction", "Daily MA",
                               "Vol Strength", "Prev Close"],
                    "Value": [
                        f"{analysis_data['atr']:.2f}",
                        f"{analysis_data['atr_threshold']:.2f}",
                        f"{analysis_data['ma_15m']:.2f}",
                        f"{analysis_data['ma_50_5m']:.2f}",
                        analysis_data['trend_direction'],
                        f"{analysis_data['daily_ma']:.2f}",
                        f"{analysis_data['vol_strength_score']}",
                        f"₹{analysis_data['prev_close']:.2f}"
                    ]
                }
                st.subheader("Price Metrics")
                st.table(pd.DataFrame(price_data_display))

            with col_detail_indicators:
                # Add new indicators here
                indicator_data_display = {
                    "Metric": ["RSI", "MACD Line", "MACD Signal", "MACD Hist", "Stochastic %K", "Stochastic %D",
                               "Divergence Score",
                               "SuperTrend", "ADX", "VWAP", "BB Lower", "BB Middle", "BB Upper", "ATR Band Lower",
                               "ATR Band Upper"],
                    "Value": [
                        f"{analysis_data['rsi']:.2f}",
                        f"{analysis_data['macd_line']:.2f}",
                        f"{analysis_data['macd_signal']:.2f}",
                        f"{analysis_data['macd_hist']:.2f}",
                        f"{analysis_data['stoch_k']:.2f}",
                        f"{analysis_data['stoch_d']:.2f}",
                        f"{analysis_data['divergence_score']}",
                        f"{analysis_data['supertrend']:.2f}",  # NEW
                        f"{analysis_data['adx']:.2f}",  # NEW
                        f"{analysis_data['vwap']:.2f}",  # NEW
                        f"{analysis_data['bb_lower']:.2f}",  # NEW
                        f"{analysis_data['bb_middle']:.2f}",  # NEW
                        f"{analysis_data['bb_upper']:.2f}",  # NEW
                        f"{analysis_data['atr_bands_lower']:.2f}",  # NEW
                        f"{analysis_data['atr_bands_upper']:.2f}"  # NEW
                    ]
                }
                st.subheader("Technical Indicators")
                st.table(pd.DataFrame(indicator_data_display))

            # --- Max Pain Graph ---
            if 'max_pain_strikes_list' in analysis_data and analysis_data['max_pain_strikes_list'] and \
                    'max_pain_losses_list' in analysis_data and analysis_data['max_pain_losses_list']:

                max_pain_chart_df = pd.DataFrame({
                    'Strike': analysis_data['max_pain_strikes_list'],
                    'Total Loss': analysis_data['max_pain_losses_list']
                })
                max_pain_chart_df = max_pain_chart_df.set_index('Strike')

                st.subheader(f"Max Pain Analysis (Current: {analysis_data['max_pain']})")
                st.bar_chart(max_pain_chart_df)
            else:
                st.info("Max Pain graph data not available.")

        st.markdown("---")  # Separator

        # --- NEW: OTM/ITM Option Chain Metrics ---
        if 'otm_itm_metrics' in analysis_data and analysis_data['otm_itm_metrics']:
            st.subheader("Key OTM/ITM Option Chain Metrics")

            # Format OTM/ITM data for display in a table
            otm_itm_rows = []

            # Define the order of strike types for display (for clarity)
            ordered_strike_types = [
                '2 ITM Call', '1 ITM Call', 'ATM', '1 OTM Call', '2 OTM Call',
                '2 ITM Put', '1 ITM Put', 'ATM Put', '1 OTM Put', '2 OTM Put'  # Reordered Puts for better visual flow
            ]

            for strike_type_label in ordered_strike_types:
                metrics = analysis_data['otm_itm_metrics'].get(strike_type_label, {})
                otm_itm_rows.append({
                    "Type": strike_type_label,
                    "Strike": metrics.get('strike', 0),  # Default to 0 (numeric) instead of 'N/A' (string)
                    "CE_OI": f"{metrics.get('CE_OI', 0.0):.0f}",
                    "CE_ChgOI": f"{metrics.get('CE_ChgOI', 0.0):.0f}",
                    "CE_LTP": f"₹{metrics.get('CE_LTP', 0.0):.2f}",
                    "PE_LTP": f"₹{metrics.get('PE_LTP', 0.0):.2f}",
                    "PE_ChgOI": f"{metrics.get('PE_ChgOI', 0.0):.0f}",
                    "PE_OI": f"{metrics.get('PE_OI', 0.0):.0f}"
                })

            df_otm_itm_display = pd.DataFrame(otm_itm_rows)

            # Custom styling functions for green/red based on Change in OI
            def color_chg_oi(s):
                return ['color: green' if float(x.replace('₹', '').strip()) > 0 else 'color: red' if float(
                    x.replace('₹', '').strip()) < 0 else '' for x in s]

            # Apply styling
            styled_otm_itm_df = df_otm_itm_display.style.apply(color_chg_oi, subset=['CE_ChgOI', 'PE_ChgOI'])

            st.dataframe(styled_otm_itm_df, height=300)  # Fixed height for scrollability

        else:
            st.info("No OTM/ITM option chain data available yet.")

        st.markdown("---")  # Separator

        # Row 5: OI Chart
        st.subheader("Option Chain OI Visuals")
        if 'df_filtered_oi_data' in analysis_data and not analysis_data['df_filtered_oi_data'].empty:
            chart_data = analysis_data['df_filtered_oi_data'][['Strike', 'CE_OI', 'PE_OI']].set_index('Strike')
            st.bar_chart(chart_data)
        else:
            st.info("No filtered OI data to display chart.")

        st.markdown("---")  # Separator

        # Row 6: Trade History
        st.subheader("Analysis History (Last 10 entries)")
        if st.session_state.analysis_history:  # Use analysis_history stored in session state
            history_display = []
            for entry in st.session_state.analysis_history:
                _sugg, _conf = final_suggestion_extended(entry.get('final_score', 0), entry.get('atr', 0) == 0.0)

                display_history_entry = {
                    "Timestamp": entry.get('timestamp'),
                    "Spot": f"₹{entry.get('spot_price', 0.0):.2f}",
                    "Suggestion": entry.get('final_suggestion'),
                    "Confidence": f"{_conf}%",
                    "PCR": f"{entry.get('pcr', 0.0):.2f}",
                    "RSI": f"{entry.get('rsi', 0.0):.2f}",
                    "Trend": entry.get('trend_direction', 'N/A')
                    # Add OTM/ITM data to history if available, for a more detailed history table
                    # Adding too many columns here can make the history table very wide.
                    # Consider adding only the most crucial OTM/ITM metric, or a summary.
                    # For now, keeping it concise.
                }
                history_display.append(display_history_entry)

            history_df_display = pd.DataFrame(history_display)
            st.dataframe(history_df_display, height=200)  # Set a fixed height for scrolling
        else:
            st.info("No analysis history available yet.")

        logging.info("[update_dashboard] Dashboard update finished successfully.")

    except Exception as e:
        st.error(f"An unexpected error occurred during dashboard update: {e}")
        logging.error(f"[update_dashboard] Exception: {e}", exc_info=True)
        st.empty()  # Clears all dynamic content on the main page
        st.stop()  # Stop further execution to prevent infinite loops on error

    finally:
        time.sleep(REFRESH_INTERVAL_SECONDS)  # Use the user-controlled refresh interval
        st.rerun()


# Call the update function to run the dashboard
update_dashboard()