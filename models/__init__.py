# models/__init__.py
"""Model registry.  Import all model modules and collect them."""

from models.bert_tiny import MODEL_ID as _bt_id, MODEL_NAME as _bt_name
from models.distilbert import MODEL_ID as _db_id, MODEL_NAME as _db_name
from models.albert import MODEL_ID as _al_id, MODEL_NAME as _al_name
from models.distilroberta import MODEL_ID as _dr_id, MODEL_NAME as _dr_name

MODEL_REGISTRY = [
    {"model_id": _bt_id, "model_name": _bt_name},
    {"model_id": _db_id, "model_name": _db_name},
    {"model_id": _al_id, "model_name": _al_name},
    {"model_id": _dr_id, "model_name": _dr_name},
]
