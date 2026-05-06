# app.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

# Import local modules
from schemas import (
    SensorData,
    SensorDataBatch,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from model import ac_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AC Control AI API",
    description="AI-powered AC control based on temperature, humidity, and dust levels",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware (important for your website)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your website URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    logger.info("Loading AC control model...")
    success = ac_model.load_model()
    if success:
        logger.info("[OK] Model loaded successfully")
    else:
        logger.error("[ERROR] Failed to load model")

    # Print sample prediction
    if ac_model.is_loaded:
        test_result = ac_model.predict(32.4, 68, 242)
        logger.info(
            f"Test prediction: {test_result['command']} (confidence: {test_result['confidence']:.2%})"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# Health check endpoint
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if API and model are running"""
    return HealthResponse(
        status="healthy" if ac_model.is_loaded else "degraded",
        model_loaded=ac_model.is_loaded,
        scaler_loaded=ac_model.is_loaded,
        version="1.0.0",
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_ac(sensor_data: SensorData):
    """
    Predict AC command from single sensor reading

    - **temperature_c**: Temperature in Celsius (-10 to 60)
    - **humidity_percent**: Relative humidity (0-100%)
    - **dust_level**: Dust sensor reading (0-1000)

    Returns AC command (OFF, LOW, or HIGH) with confidence score
    """
    try:
        if not ac_model.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later.",
            )

        # Make prediction
        result = ac_model.predict(
            sensor_data.temperature_c,
            sensor_data.humidity_percent,
            sensor_data.dust_level,
        )

        logger.info(
            f"Prediction made: {result['command']} (confidence: {result['confidence']:.2%})"
        )

        return PredictionResponse(success=True, **result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# Batch prediction endpoint
@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"]
)
async def predict_batch_ac(batch_data: SensorDataBatch):
    """
    Predict AC commands for multiple sensor readings in one request

    Useful for processing historical data or multiple rooms
    """
    try:
        if not ac_model.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later.",
            )

        # Make batch predictions
        results = ac_model.predict_batch(batch_data.samples)

        predictions = [PredictionResponse(success=True, **result) for result in results]

        logger.info(f"Batch prediction: {len(predictions)} samples processed")

        return BatchPredictionResponse(
            success=True, predictions=predictions, total_samples=len(predictions)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


# Model info endpoint
@app.get("/model/info", tags=["Info"])
async def model_info():
    """Get model information and feature importance"""
    if not ac_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    # Get feature importance from model
    feature_importance = {"Temperature": 0.6282, "Humidity": 0.2237, "Dust": 0.1481}

    return {
        "model_type": "RandomForestClassifier",
        "model_version": "1.0",
        "features": ["temperature_c", "humidity_percent", "dust_level"],
        "output_classes": ["OFF", "LOW", "HIGH"],
        "feature_importance": feature_importance,
        "accuracy": 0.865,
        "training_samples": 1000,
    }


# Root endpoint with API info
@app.get("/api/info", tags=["Info"])
async def api_info():
    """Get API information and available endpoints"""
    return {
        "api_name": "AC Control AI API",
        "version": "1.0.0",
        "description": "AI-powered AC control based on environmental sensors",
        "endpoints": {
            "health_check": "/health",
            "single_prediction": "/predict (POST)",
            "batch_prediction": "/predict/batch (POST)",
            "model_info": "/model/info",
            "api_docs": "/docs",
            "redoc": "/redoc",
        },
        "examples": {
            "curl_single": 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{"temperature_c": 32.5, "humidity_percent": 68, "dust_level": 242}\'',
            "curl_batch": 'curl -X POST "http://localhost:8000/predict/batch" -H "Content-Type: application/json" -d \'{"samples": [{"temperature_c": 35, "humidity_percent": 65, "dust_level": 350}, {"temperature_c": 22, "humidity_percent": 45, "dust_level": 80}]}\'',
        },
    }
