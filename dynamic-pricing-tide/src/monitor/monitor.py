# Contents of /dynamic-pricing-tide/src/monitor/monitor.py

import logging
import numpy as np

class ModelMonitor:
    def __init__(self, model, threshold=0.1):
        self.model = model
        self.threshold = threshold
        self.previous_predictions = None

    def detect_drift(self, current_data):
        current_predictions = self.model.predict(current_data)
        
        if self.previous_predictions is not None:
            drift_detected = self._calculate_drift(current_predictions)
            if drift_detected:
                logging.warning("Data drift detected!")
        
        self.previous_predictions = current_predictions

    def _calculate_drift(self, current_predictions):
        if self.previous_predictions is None:
            return False
        
        drift = np.abs(current_predictions - self.previous_predictions)
        return np.any(drift > self.threshold

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")

# Example usage:
# model_monitor = ModelMonitor(your_model)
# model_monitor.detect_drift(new_data)