# truedata_api_client.py
import os, sys
import logging
import pandas as pd
import time
import datetime as dt
from typing import Union, Dict, Any, List
import requests  # Needed for direct REST API calls

# TrueData SDK imports
from truedata import TD_live, TD_hist

logging.basicConfig(level=logging.DEBUG, # MAKE SURE THIS IS DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("nifty_bot.log"),
                        logging.StreamHandler(sys.stdout)
                    ])


class TrueDataAPIClient:
    """
    A client to interact with the TrueData API (TD_live and TD_hist),
    handling authentication and data fetching for live, historical, and option chain data.
    It also handles Bearer Token authentication for TrueData's REST APIs.
    Implemented as a Singleton to ensure only one instance of TD_live/TD_hist clients.
    """
    _instance = None
    _td_live_client: Union[TD_live, None] = None
    _td_hist_client: Union[TD_hist, None] = None

    # Internal dictionaries to store the latest live data received via callbacks
    _live_quotes: Dict[str, Dict[str, Any]] = {}
    _live_option_chains: Dict[str, Any] = {}

    _bearer_token: Union[str, None] = None
    _token_expiry_time: Union[dt.datetime, None] = None
    _AUTH_URL = "https://auth.truedata.in/token"
    _ANALYTICS_BASE_URL = "https://analytics.truedata.in/api"
    _SYMBOL_MASTER_BASE_URL = "https://api.truedata.in"  # For Symbol Master APIs like getSymbolExpiryList

    def __new__(cls, *args, **kwargs):
        """Ensures only one instance of the client exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(TrueDataAPIClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, username: str, password: str, realtime_port: int):
        """
        Initializes the TrueData API client.
        This constructor will only run once due to the Singleton pattern.
        """
        if self._initialized:
            return

        self.username = username
        self.password = password
        self.realtime_port = realtime_port

        self._initialize_clients()
        self._initialized = True

    def _initialize_clients(self):
        """Initializes TD_live and TD_hist clients and attempts to get initial bearer token."""
        logging.info(f"[TrueDataAPIClient] Initializing TrueData Live client with port {self.realtime_port}...")
        try:
            self._td_live_client = TD_live(
                self.username,
                self.password,
                live_port=self.realtime_port,
                log_level=logging.WARNING
            )
            self._td_live_client.live_quote_callback = self._live_quote_callback
            self._td_live_client.start_live_data([])
            logging.info("[TrueDataAPIClient] TrueData Live client initialized and data stream started.")

        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Error initializing TD_live client: {e}", exc_info=True)
            self._td_live_client = None

        logging.info("[TrueDataAPIClient] Initializing TrueData Historical client...")
        try:
            self._td_hist_client = TD_hist(self.username, self.password, log_level=logging.WARNING)
            logging.info("[TrueDataAPIClient] TrueData Historical client initialized.")
        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Error initializing TD_hist client: {e}", exc_info=True)
            self._td_hist_client = None

        # Attempt to get an initial Bearer Token for REST APIs
        self._get_bearer_token()

    def _get_bearer_token(self) -> Union[str, None]:
        """Authenticates with TrueData to get a new Bearer Token for REST APIs."""
        # Check if token is still valid (refresh if less than 5 minutes to expiry)
        if self._bearer_token and self._token_expiry_time and self._token_expiry_time > dt.datetime.now() + dt.timedelta(
                minutes=5):
            logging.debug("[TrueDataAPIClient] Bearer token is still valid.")
            return self._bearer_token

        logging.info("[TrueDataAPIClient] Attempting to get a new Bearer Token...")
        try:
            response = requests.post(
                self._AUTH_URL,
                data={
                    "username": self.username,
                    "password": self.password,
                    "grant_type": "password"
                },
                timeout=10
            )
            response.raise_for_status()
            token_data = response.json()

            self._bearer_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self._token_expiry_time = dt.datetime.now() + dt.timedelta(seconds=expires_in)

            logging.info(f"✅ Bearer Token obtained. Expires in {expires_in} seconds.")
            return self._bearer_token

        except requests.exceptions.RequestException as e:
            logging.error(f"[TrueDataAPIClient] Error fetching Bearer Token (Request Error): {e}", exc_info=True)
        except json.JSONDecodeError as e:
            logging.error(f"[TrueDataAPIClient] Error decoding Bearer Token response: {e}. Response: {response.text}",
                          exc_info=True)
        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Unexpected error getting Bearer Token: {e}", exc_info=True)

        self._bearer_token = None
        self._token_expiry_time = None
        return None

    def _make_rest_call(self, url: str, params: Dict[str, Any], use_bearer_token: bool = True) -> Any:
        """
        Helper to make authenticated REST API calls.
        `use_bearer_token=True` for Analytics/History APIs.
        `use_bearer_token=False` for Symbol Master APIs (which use user/pass in params).
        """
        headers = {}
        if use_bearer_token:
            token = self._get_bearer_token()
            if not token:
                logging.error(f"[TrueDataAPIClient] No valid Bearer Token. Cannot make REST call to {url}.")
                return None
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()

            if params.get("response") == "csv":
                from io import StringIO
                return pd.read_csv(StringIO(response.text))
            else:  # Assume JSON if not CSV
                return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"[TrueDataAPIClient] REST call to {url} failed: {e}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Unexpected error in REST call to {url}: {e}", exc_info=True)
            return None

    def _live_quote_callback(self, quote_data: Dict[str, Any]):
        """
        Internal callback method to receive live quote updates from TD_live
        and store the latest data in the _live_quotes dictionary.
        """
        symbol = quote_data.get('symbol')
        if symbol:
            self._live_quotes[symbol] = quote_data
            logging.debug(f"[TrueDataAPIClient] Live quote callback: {symbol} LTP={quote_data.get('ltp')}")

    def get_live_spot_price(self, symbol: str) -> float:
        """
        Fetches the live spot price for a given symbol using TrueData's REST API (getLTPSpot).
        """
        logging.info(f"[TrueDataAPIClient] Requesting live spot price for {symbol} via REST API.")
        url = f"{self._ANALYTICS_BASE_URL}/getLTPSpot"

        # Standardize symbol for NIFTY 50 index spot as per TrueData's conventions
        if symbol == "NIFTY":
            api_symbol = "NIFTY 50"  # Use "NIFTY 50" for the symbol
            api_series = "IND"  # And "IND" for the series for indices
        else:
            api_symbol = symbol
            api_series = "EQ"  # Default to equity if not NIFTY (you might need to refine this for other symbols)

        params = {
            "symbol": api_symbol,
            "series": api_series,
            "response": "json"
        }

        # Use _make_rest_call with use_bearer_token=True (default) for Analytics API
        response_data = self._make_rest_call(url, params)

        # Check for 'Records' key in the response and parse it
        if response_data and isinstance(response_data, dict) and 'Records' in response_data:
            try:
                # Extract the string, strip whitespace/newlines, and convert to float
                price_str = response_data['Records']
                if isinstance(price_str, str):
                    price = float(price_str.strip())
                elif isinstance(price_str, list) and price_str:
                    # If it's a list (e.g., from CSV response parsed as JSON), assume first element
                    if isinstance(price_str[0], list) and price_str[0]:
                        price = float(price_str[0][0].strip())
                    else:
                        price = float(price_str[0].strip())
                else:
                    logging.warning(
                        f"[TrueDataAPIClient] Unexpected 'Records' format for spot price: {type(price_str)}. Response: {response_data}")
                    return 0.0

                logging.info(f"✅ Live spot price for {symbol} from TrueData REST API: {price}")
                return price
            except (ValueError, IndexError, TypeError) as e:
                logging.error(
                    f"[TrueDataAPIClient] Could not convert price from 'Records' to float: {response_data.get('Records')}. Error: {e}",
                    exc_info=True)
                return 0.0

        logging.warning(
            f"[TrueDataAPIClient] Live spot price for {symbol} not available or unexpected format from REST API. Response: {response_data}. Returning 0.0.")
        return 0.0

    def get_historical_daily_bars(self, symbol: str, days_back: int) -> pd.DataFrame:
        """Fetches daily historical bars using TrueData TD_hist."""
        logging.info(f"[TrueDataAPIClient] Getting historical daily bars for {symbol} for {days_back} days.")
        if not self._td_hist_client:
            logging.error("TD_hist client not initialized. Cannot get historical daily data.")
            return pd.DataFrame()

        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days_back)

        try:
            df = self._td_hist_client.get_historic_data(
                "NIFTY 50",  # Use the dynamic symbol (e.g., "NIFTY-I")
                start_time=start_time,
                end_time=end_time,
                bar_size='EOD'
            )
            # NEW: Robustly check if DataFrame is valid and has expected columns before processing
            # Expected columns from TrueData docs for CSV: timestamp,open,high,low,close,volume,oi
            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if df is None or df.empty or not all(col in df.columns for col in expected_cols):
                logging.warning(
                    f"[TrueDataAPIClient] No historical daily data or unexpected format for {symbol}. Expected {expected_cols}. Got: {df.columns.tolist() if df is not None else 'None'}. Returning empty DataFrame.")
                return pd.DataFrame()

            df = df.rename(columns={
                "timestamp": "time", "open": "open", "high": "high",  # Rename 'timestamp' to 'time'
                "low": "low", "close": "close", "volume": "volume"
            }).copy()

            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values("time").reset_index(drop=True)

            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            logging.info(f"✅ Historical daily bars for {symbol} fetched. Shape: {df.shape}")
            return df

        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Error fetching historical daily bars for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_historical_intraday_bars(self, symbol: str, interval_minutes: int, days_back: int) -> pd.DataFrame:
        """
        Fetches intraday historical bars (e.g., 5-min) using TrueData TD_hist.
        TrueData bar_size can be '1 min', '5 min', '15 min', etc.
        """
        logging.info(
            f"[TrueDataAPIClient] Getting historical {interval_minutes}-min bars for {symbol} for {days_back} days.")
        if not self._td_hist_client:
            logging.error("TD_hist client not initialized. Cannot get historical intraday data.")
            return pd.DataFrame()

        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days_back)
        bar_size_str = f"{interval_minutes} min"

        try:
            df = self._td_hist_client.get_historic_data(
                "NIFTY 50",  # Use the dynamic symbol (e.g., "NIFTY-I")
                start_time=start_time,
                end_time=end_time,
                bar_size=bar_size_str
            )
            # NEW: Robustly check if DataFrame is valid and has expected columns before processing
            # Expected columns from TrueData docs for CSV: timestamp,open,high,low,close,volume,oi
            expected_cols = ['timestamp', 'open', 'high', 'low', 'close',
                                          'Volume', 'oi']  # Corrected 'Volume' capitalization

            #expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            logging.info(f"[get_historical_intraday_bars]- fetched. df: {df.columns}")
            if df is None or df.empty or not all(col in df.columns for col in expected_cols):
                logging.warning(
                    f"[TrueDataAPIClient] No historical {interval_minutes}-min data or unexpected format for {symbol}. Expected {expected_cols}. Got: {df.columns.tolist() if df is not None else 'None'}. Returning empty DataFrame.")
                return pd.DataFrame()

            df = df.rename(columns={
                "timestamp": "time", "open": "open", "high": "high",  # Rename 'timestamp' to 'time'
                "low": "low", "close": "close", "Volume": "volume"
            }).copy()

            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values("time").reset_index(drop=True)

            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            logging.info(f"✅ Historical {interval_minutes}-min bars for {symbol} fetched. Shape: {df.shape}")
            return df

        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Error fetching historical intraday bars for {symbol}: {e}",
                          exc_info=True)
            return pd.DataFrame()

    def get_symbol_expiry_list(self, symbol: str) -> List[str]:
        """
        Fetches the list of expiry dates for a given symbol using TrueData REST API (getSymbolExpiryList).
        Requires username/password as query parameters.
        """
        logging.info(f"[TrueDataAPIClient] Getting symbol expiry list for {symbol}.")
        url = f"{self._SYMBOL_MASTER_BASE_URL}/getSymbolExpiryList"
        params = {
            "symbol": symbol,
            "user": self.username,  # Add username as query parameter
            "password": self.password,  # Add password as query parameter
            "response": "json"  # Request JSON response
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            response_data = response.json()

            if response_data and isinstance(response_data, dict) and 'Records' in response_data:
                # Expecting format: {'status': 'Success', 'Records': [['YYYYMMDD'], ['YYYYMMDD'], ...]}
                records_list = response_data['Records']
                if isinstance(records_list, list):
                    # Extract the expiry string from the inner list: [['YYYYMMDD']] -> 'YYYYMMDD'
                    expiries = [item[0] for item in records_list if isinstance(item, list) and item]
                    logging.info(f"✅ Symbol expiry list for {symbol} fetched: {expiries}")
                    return expiries

            logging.warning(
                f"[TrueDataAPIClient] Failed to get symbol expiry list for {symbol}. Response: {response_data}. Returning empty list.")
            return []
        except requests.exceptions.RequestException as e:
            logging.error(f"[TrueDataAPIClient] REST call for expiry list failed: {e}", exc_info=True)
            return []
        except json.JSONDecodeError as e:
            logging.error(f"[TrueDataAPIClient] Error decoding expiry list response: {e}. Response: {response.text}",
                          exc_info=True)
            return []
        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Unexpected error getting expiry list: {e}", exc_info=True)
            return []

    def get_live_option_chain_data(self, symbol: str, expiry_date: dt.datetime) -> pd.DataFrame:
        """
        Fetches the live option chain data from the TrueData live stream (start_option_chain).
        Automatically starts streaming the option chain if not already.
        """
        logging.info(
            f"[TrueDataAPIClient] Getting live option chain for {symbol} expiring {expiry_date.strftime('%Y-%m-%d')}.")
        if not self._td_live_client:
            logging.error("TD_live client not initialized. Cannot get live option chain.")
            return pd.DataFrame()

        try:
            # If the option chain object for this symbol is not yet in our store, start streaming it
            # Or if the existing chain object is empty (data not yet populated/stream broken)
            if symbol not in self._live_option_chains or self._live_option_chains[symbol].get_option_chain().empty:
                logging.info(
                    f"[TrueDataAPIClient] Starting option chain stream for {symbol} expiring {expiry_date.strftime('%Y-%m-%d')}.")

                chain_obj = self._td_live_client.start_option_chain(
                    symbol,
                    expiry_date,
                    chain_length=15,  # Fetch 15 strikes around ATM by default
                    bid_ask=True,
                    greek=False
                )
                self._live_option_chains[symbol] = chain_obj
                time.sleep(5)  # Give some time for the stream to populate data

            # Retrieve the current state of the option chain from the live object
            df_oc = self._live_option_chains[symbol].get_option_chain()
            if df_oc is None or df_oc.empty:
                logging.warning(
                    f"[TrueDataAPIClient] TrueData TD_live get_option_chain returned empty for {symbol} expiring {expiry_date.strftime('%Y-%m-%d')}. Data might not be available yet.")
                return pd.DataFrame()

            logging.debug(f"[TrueDataAPIClient] Raw Option Chain columns: {df_oc.columns.tolist()}")

            # --- NEW: Process the raw option chain DataFrame ---
            # Expected raw columns from logs: ['strike', 'type', 'ltp', 'ltt', 'ltq', 'volume', 'price_change', 'price_change_perc', 'oi', 'prev_oi', 'oi_change', 'oi_change_perc', 'bid', 'bid_qty', 'ask', 'ask_qty']
            # 'type' column distinguishes 'CE' and 'PE' rows.

            # Separate calls and puts based on 'type' column
            df_calls = df_oc[df_oc['type'] == 'CE'].copy()
            df_puts = df_oc[df_oc['type'] == 'PE'].copy()

            # Select and rename columns for Calls
            df_calls = df_calls[['strike', 'oi', 'oi_change', 'ltp']].rename(columns={
                'strike': 'Strike',
                'oi': 'CE_OI',
                'oi_change': 'CE_ChgOI',
                'ltp': 'CE_LTP'
            }, errors='ignore')

            # Select and rename columns for Puts
            df_puts = df_puts[['strike', 'oi', 'oi_change', 'ltp']].rename(columns={
                'strike': 'Strike',
                'oi': 'PE_OI',
                'oi_change': 'PE_ChgOI',
                'ltp': 'PE_LTP'
            }, errors='ignore')

            # Merge calls and puts on Strike to get a single row per strike
            # Use 'outer' merge to include strikes that might only have CE or PE data
            df_merged = pd.merge(df_calls, df_puts, on='Strike', how='outer')

            # Fill NaN values (e.g., if a strike has only CE or PE, the other side will be NaN after merge)
            df_merged = df_merged.fillna(0)

            # Ensure 'Strike' is integer type and other relevant columns are numeric
            # Apply astype only if column exists and is not entirely NaN/empty
            for col in ["Strike", "CE_OI", "PE_OI", "CE_ChgOI", "PE_ChgOI", "CE_LTP", "PE_LTP"]:
                if col in df_merged.columns and not df_merged[col].empty:
                    # Convert to numeric, coercing errors, then fillna(0) for safe int conversion
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
                    if col == "Strike":
                        df_merged[col] = df_merged[col].astype(int)
                    else:
                        df_merged[col] = df_merged[col].astype(float)
                else:  # If a critical column is missing after merge (e.g., due to data issues), log and fill with 0
                    if col in ["CE_OI", "PE_OI", "CE_ChgOI", "PE_ChgOI", "CE_LTP", "PE_LTP"]:
                        logging.warning(
                            f"[TrueDataAPIClient] Missing or empty column '{col}' after OC merge for {symbol}. Filling with 0.")
                        df_merged[col] = 0.0

            # Select and return only the columns your bot expects
            df_processed_oc = df_merged[["Strike", "CE_OI", "PE_OI", "CE_ChgOI", "PE_ChgOI", "CE_LTP", "PE_LTP"]].copy()

            logging.info(f"✅ Live option chain for {symbol} fetched. Shape: {df_processed_oc.shape}")
            return df_processed_oc

        except Exception as e:
            logging.error(f"[TrueDataAPIClient] Error fetching live option chain for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()


# --- Global TrueDataAPIClient instance (Singleton) ---
# This instance will be initialized only once when the module is imported.
# It's IMPORTANT to set TRUEDATA_USERNAME, TRUEDATA_PASSWORD, TRUEDATA_REALTIME_PORT
# environment variables where your bot runs (e.g., on Render dashboard) with your actual credentials.
TD_USERNAME_ENV = os.getenv("TRUEDATA_USERNAME", "tdwsp693")
TD_PASSWORD_ENV = os.getenv("TRUEDATA_PASSWORD", "vijay@693")
TD_REALTIME_PORT_ENV = int(os.getenv("TRUEDATA_REALTIME_PORT", 8084))

true_data_client = TrueDataAPIClient(TD_USERNAME_ENV, TD_PASSWORD_ENV, TD_REALTIME_PORT_ENV)