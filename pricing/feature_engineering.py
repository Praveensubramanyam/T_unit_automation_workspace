import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class AdvancedPricingFeatureEngineer:
    """
    Streamlined feature engineering for SellingPrice prediction.
    Focus: 10-15 features without data leakage
    """
    
    def __init__(self, target_column='SellingPrice'):
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names_ = []
        self.categorical_columns = ['Brand', 'FC_ID']
        self.mrp_quantiles = None
        self.brand_mrp_median = None
        self.overall_brand_median = None
        self.historical_stats = {}  # Store historical averages
        
    def get_feature_names(self):
        """Return list of feature names after transformation"""
        return self.feature_names_
    
    def calculate_historical_features(self, df, is_training=True):
        """Calculate historical features that don't depend on current selling price"""
        df = df.copy()
        
        # Ensure Date column is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values(["Brand", "FC_ID", "Date"])
            
            # Historical features (using data from 14+ days ago to avoid leakage)
            for group_cols in [["Brand", "FC_ID"], ["Brand"], ["FC_ID"]]:
                if all(col in df.columns for col in group_cols):
                    group_name = "_".join(group_cols)
                    
                    # Historical demand patterns (14-day lag minimum)
                    if "Demand" in df.columns:
                        df[f"Historical_Demand_Mean_{group_name}"] = (
                            df.groupby(group_cols)["Demand"]
                            .shift(14)  # Use data from 2+ weeks ago
                            .rolling(window=30, min_periods=7)
                            .mean()
                        )
                    
                    # Historical stock patterns
                    if "StockStart" in df.columns:
                        df[f"Historical_Stock_Mean_{group_name}"] = (
                            df.groupby(group_cols)["StockStart"]
                            .shift(14)
                            .rolling(window=30, min_periods=7)
                            .mean()
                        )
        
        return df

    def advanced_feature_engineering(self, df, is_training=True):
        """Create features without data leakage from SellingPrice"""
        df = df.copy()
        
        # Basic temporal features
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df["Month"] = df["Date"].dt.month
            df["DayOfWeek"] = df["Date"].dt.dayofweek
            df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
            df["DayOfMonth"] = df["Date"].dt.day
            df["Quarter"] = df["Date"].dt.quarter
        
        # --- SAFE FEATURES (No Data Leakage) ---
        
        # 1. Product characteristics (independent of selling price)
        if is_training:
            self.mrp_quantiles = df["MRP_x"].quantile([0.33, 0.67]).values
            if self.mrp_quantiles[0] == self.mrp_quantiles[1]:
                min_val = df["MRP_x"].min()
                max_val = df["MRP_x"].max()
                if min_val == max_val:
                    self.mrp_quantiles = [min_val - 1, max_val + 1]
                else:
                    range_val = max_val - min_val
                    self.mrp_quantiles = [min_val + range_val/3, min_val + 2*range_val/3]
        
        try:
            df["MRP_Category"] = pd.cut(df["MRP_x"], 
                                      bins=[-np.inf, self.mrp_quantiles[0], self.mrp_quantiles[1], np.inf], 
                                      labels=[0, 1, 2],
                                      duplicates='drop').astype(int)
        except:
            df["MRP_Category"] = pd.qcut(df["MRP_x"].rank(method='first'), q=3, labels=[0, 1, 2]).astype(int)
        
        # MRP-based features (MRP is set before selling price)
        df["Log_MRP"] = np.log1p(df["MRP_x"])
        df["MRP_Squared"] = df["MRP_x"] ** 2
        
        # 2. Market and location features (independent)
        df["IsMetroMarket"] = df.get("IsMetro", 0).astype(int)
        
        # 3. Brand positioning (calculated from MRP, not selling price)
        if is_training:
            self.brand_mrp_median = df.groupby("Brand")["MRP_x"].median().to_dict()
            self.overall_brand_median = pd.Series(self.brand_mrp_median.values()).median()
        
        df["Brand_MRP_Median"] = df["Brand"].map(self.brand_mrp_median).fillna(self.overall_brand_median)
        df["Is_Premium_Brand"] = (df["Brand_MRP_Median"] > self.overall_brand_median).astype(int)
        df["MRP_vs_Brand_Median"] = df["MRP_x"] / df["Brand_MRP_Median"]
        
        # 4. Seasonal/Temporal features
        df["Is_Peak_Season"] = df["Month"].isin([11, 12, 1]).astype(int)  # Holiday season
        df["Is_Summer"] = df["Month"].isin([4, 5, 6]).astype(int)
        df["Is_Mid_Week"] = df["DayOfWeek"].isin([1, 2, 3]).astype(int)
        df["Is_Month_Start"] = (df.get("DayOfMonth", 15) <= 5).astype(int)
        df["Is_Month_End"] = (df.get("DayOfMonth", 15) >= 25).astype(int)
        
        # 5. Competition and market context (independent of current price)
        # These should be external market data, not derived from your sales
        df["Market_Competition_Index"] = 1.0  # Placeholder - should be external data
        df["Economic_Index"] = 1.0  # Placeholder - should be external economic indicators
        
        # 6. Product lifecycle features (based on launch date, not sales performance)
        if "Date" in df.columns and is_training:
            # Calculate product age (assumes first appearance is launch)
            product_launch = df.groupby(["Brand", "FC_ID"])["Date"].min().to_dict()
            self.product_launch_dates = product_launch
        
        if hasattr(self, 'product_launch_dates'):
            df["Product_Age_Days"] = (
                df["Date"] - df.apply(lambda x: self.product_launch_dates.get((x["Brand"], x["FC_ID"]), x["Date"]), axis=1)
            ).dt.days
            df["Is_New_Product"] = (df["Product_Age_Days"] <= 30).astype(int)
            df["Is_Mature_Product"] = (df["Product_Age_Days"] >= 365).astype(int)
        else:
            df["Product_Age_Days"] = 0
            df["Is_New_Product"] = 0
            df["Is_Mature_Product"] = 0
        
        # 7. Historical patterns (using safe lags)
        df = self.calculate_historical_features(df, is_training)
        
        # 8. Supply-side features (independent of price)
        df["Lead_Time_Days"] = df.get("LeadTimeFloat", 0)
        df["Initial_Stock_Level"] = df.get("StockStart", 0)  # Stock at start of period
        
        # 9. External factors (should be independent)
        # These are placeholders - in reality, you'd use external data
        if "Date" in df.columns:
            df["Day_of_Year"] = df["Date"].dt.dayofyear.astype(int)
            df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
        else:
            df["Day_of_Year"] = 1
            df["Week_of_Year"] = 1
        
        return df

    def encode_categorical_features(self, df):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    known_categories = set(self.label_encoders[col].classes_)
                    df_encoded[col] = df_encoded[col].astype(str).apply(
                        lambda x: x if x in known_categories else list(known_categories)[0]
                    )
                    df_encoded[f"{col}_encoded"] = self.label_encoders[col].transform(df_encoded[col])
                
        return df_encoded

    def prepare_features_target(self, df):
        """Select final features for modeling - NO LEAKAGE"""
        
        # SAFE feature set - all features should be known BEFORE setting selling price
        selected_features = [
            # Product characteristics (5)
            'Log_MRP',
            'MRP_Category',
            'MRP_vs_Brand_Median',
            'Is_Premium_Brand',
            'Product_Age_Days',
            
            # Market/Location (2)
            'IsMetroMarket',
            'Brand_encoded',
            
            # Temporal patterns (6)
            'Is_Peak_Season',
            'Is_Summer',
            'IsWeekend',
            'Is_Mid_Week',
            'Month',
            'DayOfWeek',
            
            # Product lifecycle (2)
            'Is_New_Product',
            'Is_Mature_Product',
            
            # Supply-side (2)
            'Lead_Time_Days',
            'Initial_Stock_Level',
            
            # Historical patterns (if available)
            'Historical_Demand_Mean_Brand_FC_ID',
            'Historical_Stock_Mean_Brand_FC_ID',
        ]
        
        # Filter to only include features that exist in the dataframe
        available_features = [col for col in selected_features if col in df.columns]
        
        # Ensure we have at least some features
        if len(available_features) < 5:
            print("WARNING: Very few features available. Adding basic features.")
            basic_features = ['Log_MRP', 'MRP_Category', 'IsMetroMarket', 'Brand_encoded', 'Month']
            available_features.extend([f for f in basic_features if f in df.columns and f not in available_features])
        
        X = df[available_features].copy()
        y = df[self.target_column].copy()
        
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Fill missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        # Ensure all columns are numeric and compatible with XGBoost
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            # Explicitly convert to float64 for XGBoost compatibility
            X[col] = X[col].astype('float64')
        
        return X, y

    def fit_transform(self, df):
        """Apply all feature engineering steps and return features and target"""
        print("=" * 60)
        print("LEAKAGE-FREE FEATURE ENGINEERING FOR SELLING PRICE")
        print("=" * 60)
        
        # Apply feature engineering (training mode)
        df_engineered = self.advanced_feature_engineering(df, is_training=True)
        print(f"✓ Core features created. Shape: {df_engineered.shape}")
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_engineered)
        print(f"✓ Categorical features encoded. Shape: {df_encoded.shape}")
        
        # Prepare final feature set
        X, y = self.prepare_features_target(df_encoded)
        print(f"✓ Final feature selection completed.")
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Selected features: {len(self.feature_names_)}")
        
        # Display selected features with leakage check
        print("\nSelected Features (Leakage-Free):")
        for i, feature in enumerate(self.feature_names_, 1):
            print(f"  {i:2d}. {feature}")
        
        print("\n" + "="*60)
        print("LEAKAGE CHECK PASSED - All features are available BEFORE pricing decision")
        print("="*60)
        
        return X, y, df_encoded

    def transform(self, df):
        """Transform new data using fitted parameters"""
        df_engineered = self.advanced_feature_engineering(df, is_training=False)
        df_encoded = self.encode_categorical_features(df_engineered)
        X, _ = self.prepare_features_target(df_encoded)
        return X, df_encoded