# model.py
import joblib
import numpy as np
import os
from typing import Dict, Any


class ACControlModel:
    """Wrapper class for AC control model"""

    def __init__(
        self,
        model_path: str = "ac_controller_model.pkl",
        scaler_path: str = "ac_scaler.pkl",
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.is_loaded = False

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            print(f"[OK] Model loaded from {self.model_path}")
            print(f"[OK] Scaler loaded from {self.scaler_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            self.is_loaded = False
            return False

    def predict(
        self, temperature_c: float, humidity_percent: float, dust_level: float
    ) -> Dict[str, Any]:
        """
        Predict AC command from sensor readings

        Args:
            temperature_c: Temperature in Celsius
            humidity_percent: Relative humidity percentage
            dust_level: Dust sensor reading

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare input
        input_data = np.array([[temperature_c, humidity_percent, dust_level]])
        input_scaled = self.scaler.transform(input_data)

        # Get prediction and probabilities
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probabilities))

        # Map prediction to command
        command_map = {0: "OFF", 1: "LOW", 2: "HIGH"}
        command = command_map[prediction]

        # Generate reasoning
        reasoning = self._generate_reasoning(
            temperature_c, humidity_percent, dust_level, command
        )

        return {
            "command": command,
            "command_code": int(prediction),
            "confidence": confidence,
            "reasoning": reasoning,
            "sensor_values": {
                "temperature": float(temperature_c),
                "humidity": float(humidity_percent),
                "dust": float(dust_level),
            },
            "probabilities": {
                "off": float(probabilities[0]),
                "low": float(probabilities[1]),
                "high": float(probabilities[2]),
            },
        }

    def _generate_reasoning(
        self, temp: float, humid: float, dust: float, command: str
    ) -> str:
        """Generate human-readable reasoning for the prediction"""
        reasoning_factors = []

        # Temperature analysis
        if temp > 28:
            reasoning_factors.append(f"high temperature ({temp}°C)")
        elif temp > 25:
            reasoning_factors.append(f"warm temperature ({temp}°C)")
        elif temp < 22:
            reasoning_factors.append(f"cool temperature ({temp}°C)")

        # Humidity analysis
        if humid > 70:
            reasoning_factors.append(f"high humidity ({humid}%)")
        elif humid > 60:
            reasoning_factors.append(f"elevated humidity ({humid}%)")

        # Dust analysis (occupancy proxy)
        if dust > 300:
            reasoning_factors.append(f"significant dust ({dust}) - occupancy detected")
        elif dust > 200:
            reasoning_factors.append(f"moderate dust ({dust})")

        if reasoning_factors:
            reason_text = f"AC set to {command} due to {', '.join(reasoning_factors)}"
        else:
            reason_text = f"AC set to {command} (comfortable conditions)"

        return reason_text

    def predict_batch(self, samples: list) -> list:
        """Predict for multiple samples"""
        results = []
        for sample in samples:
            result = self.predict(
                sample.temperature_c, sample.humidity_percent, sample.dust_level
            )
            results.append(result)
        return results


# Global instance
ac_model = ACControlModel()
