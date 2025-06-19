import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


class PricingTimeSeriesAnalyzer:
    """
    Advanced time series analysis for pricing patterns
    """

    def __init__(self):
        self.seasonal_patterns = {}
        self.trend_components = {}
        self.forecasting_models = {}

    def decompose_time_series(self, df, value_col="SellingPrice", freq=30):
        """Decompose time series into trend, seasonal, and residual components"""
        df = df.copy()
        df = df.sort_values("Date")
        df.set_index("Date", inplace=True)

        # Resample to daily frequency if needed
        if len(df) > freq:
            ts_data = df[value_col].resample("D").mean().fillna(method="ffill")
        else:
            ts_data = df[value_col]

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            ts_data, model="additive", period=freq
        )

        self.seasonal_patterns[value_col] = {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
            "original": ts_data,
        }

        return decomposition

    def detect_seasonality(self, df, value_col="SellingPrice"):
        """Detect seasonal patterns using FFT and autocorrelation"""
        df = df.copy()
        df = df.sort_values("Date")

        # Prepare time series
        ts_data = df.set_index("Date")[value_col].fillna(method="ffill")

        # FFT-based seasonality detection
        fft = np.fft.fft(ts_data.values)
        freqs = np.fft.fftfreq(len(ts_data))

        # Find dominant frequencies
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argsort(magnitude)[-10:]  # Top 10 frequencies
        dominant_periods = [
            1 / abs(freqs[i]) for i in dominant_freq_idx if freqs[i] != 0
        ]

        return {
            "dominant_periods": dominant_periods,
            "seasonal_strength": np.std(magnitude) / np.mean(magnitude),
        }

    def create_lag_features(
        self, df, target_col="SellingPrice", lags=[1, 7, 14, 30]
    ):
        """Create lag features for time series modeling"""
        df = df.copy()
        df = df.sort_values(["Brand", "FC_ID", "Date"])

        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df.groupby(["Brand", "FC_ID"])[
                target_col
            ].shift(lag)

        return df

    def forecast_demand(self, df, forecast_horizon=30):
        """Simple demand forecasting using moving averages and trends"""
        forecasts = {}

        for brand in df["Brand"].unique():
            for fc_id in df["FC_ID"].unique():
                subset = df[
                    (df["Brand"] == brand) & (df["FC_ID"] == fc_id)
                ].copy()
                if len(subset) < 10:  # Skip if insufficient data
                    continue

                subset = subset.sort_values("Date")

                # Simple trend forecasting
                recent_demand = subset["UnitsSold"].tail(14).values
                if len(recent_demand) > 3:
                    trend = np.polyfit(
                        range(len(recent_demand)), recent_demand, 1
                    )[0]
                    last_value = recent_demand[-1]

                    # Generate forecast
                    forecast = [
                        last_value + trend * i
                        for i in range(1, forecast_horizon + 1)
                    ]
                    forecasts[f"{brand}_{fc_id}"] = {
                        "forecast": forecast,
                        "trend": trend,
                        "last_value": last_value,
                    }

        return forecasts
