# schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class ACCommandEnum(str, Enum):
    OFF = "OFF"
    LOW = "LOW"
    HIGH = "HIGH"

class SensorData(BaseModel):
    """Input model for sensor data"""
    temperature_c: float = Field(..., 
                                 ge=-10, 
                                 le=60, 
                                 description="Temperature in Celsius (-10 to 60)")
    humidity_percent: float = Field(..., 
                                    ge=0, 
                                    le=100, 
                                    description="Relative humidity (0-100%)")
    dust_level: float = Field(..., 
                              ge=0, 
                              le=1000, 
                              description="Dust sensor reading (0-1000)")
    
    @validator('temperature_c')
    def validate_temperature(cls, v):
        if v < -10 or v > 60:
            raise ValueError(f'Temperature {v}°C is outside realistic range (-10 to 60)')
        return v
    
    @validator('humidity_percent')
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError(f'Humidity {v}% is outside realistic range (0-100)')
        return v
    
    @validator('dust_level')
    def validate_dust(cls, v):
        if v < 0 or v > 1000:
            raise ValueError(f'Dust level {v} is outside realistic range (0-1000)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "temperature_c": 32.5,
                "humidity_percent": 68,
                "dust_level": 242
            }
        }

class SensorDataBatch(BaseModel):
    """Batch input model for multiple predictions"""
    samples: list[SensorData]
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {"temperature_c": 35, "humidity_percent": 65, "dust_level": 350},
                    {"temperature_c": 28, "humidity_percent": 55, "dust_level": 120}
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Output model for single prediction"""
    success: bool
    command: ACCommandEnum
    command_code: int
    confidence: float
    reasoning: str
    sensor_values: dict
    probabilities: dict
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "command": "HIGH",
                "command_code": 2,
                "confidence": 0.85,
                "reasoning": "AC set to HIGH due to high temperature (32.4°C), elevated humidity (68%)",
                "sensor_values": {
                    "temperature": 32.4,
                    "humidity": 68.0,
                    "dust": 242.0
                },
                "probabilities": {
                    "off": 0.05,
                    "low": 0.10,
                    "high": 0.85
                }
            }
        }

class BatchPredictionResponse(BaseModel):
    """Output model for batch predictions"""
    success: bool
    predictions: list[PredictionResponse]
    total_samples: int
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    version: str