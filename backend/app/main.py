import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging_setup import RequestLoggingMiddleware, setup_logging
from app.api.routes import (
    auth, fields, diseases, scans, suppliers,
    products, orders, weather, calculator, chat,
)
from app.api.routes import inference as inference_router
from app.api.routes import health as health_router
from app.services.inference_service import inference_service

setup_logging()
logger = logging.getLogger("oskin.startup")

app = FastAPI(
    title="Oskín AgTech API",
    description=(
        "Backend for Oskín — an AgTech platform for Kazakhstan farmers.\n\n"
        "Provides plant disease detection, field management, marketplace, "
        "risk assessment, economic calculator, and AI agronomist chat."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(fields.router)
app.include_router(diseases.router)
app.include_router(scans.router)
app.include_router(suppliers.router)
app.include_router(products.router)
app.include_router(orders.router)
app.include_router(weather.router)
app.include_router(calculator.router)
app.include_router(chat.router)
app.include_router(inference_router.router)
app.include_router(health_router.router)


@app.on_event("startup")
async def startup():
    loaded = inference_service.load()
    if loaded:
        logger.info("Inference service ready — %d classes", inference_service.num_classes)
    else:
        logger.warning(
            "Inference service not ready. "
            "Copy ml/export/plant_disease.tflite -> backend/app/models/ "
            "and ml/export/class_names.txt -> backend/app/models/"
        )
    logger.info("Oskín API started — http://0.0.0.0:8000 | docs: /docs")
