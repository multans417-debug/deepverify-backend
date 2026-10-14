from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

WARNING_TEXT = ("Predictions are probabilistic and may be incorrect (false positives and false negatives occur). "
                "Do not use this result as sole evidence about a person or a piece of media.")


class DetectionResponse(BaseModel):
    success: bool = True
    media_type: Literal["image", "video"]
    prediction: Literal["REAL", "FAKE"]
    fake_probability: float = Field(description="Model score for class FAKE in [0,1]. See score_is_calibrated.")
    confidence: float = Field(description="Score of the predicted class: fake_probability if FAKE else 1 - fake_probability.")
    score_is_calibrated: bool = Field(description="True when temperature scaling was fitted on a validation split. "
                                                  "Even then it holds only for data resembling the training domain.")
    decision_threshold: float
    model_version: str
    processing_time_seconds: float
    details: dict = Field(default_factory=dict)
    warning: str = WARNING_TEXT


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    required_models: list
    models: dict


class ModelsResponse(BaseModel):
    models: dict
