from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://oskin:oskin_pass@db:5432/oskin_db"

    # JWT
    SECRET_KEY: str = "change_me_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Model inference
    MODEL_PATH: str = "/app/ml_models/plant_disease.tflite"
    MODEL_CLASS_NAMES_PATH: str = "/app/ml_models/class_names.txt"
    MODEL_VERSION: str = "1.0.0"
    IMAGE_SIZE: int = 224

    # Logging
    LOG_LEVEL: str = "INFO"

    # Server
    BACKEND_PORT: int = 8000

    class Config:
        env_file = ".env.example"
        extra = "ignore"


settings = Settings()
