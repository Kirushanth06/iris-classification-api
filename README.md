# Iris Classification API - Kirushanth

FastAPI app to classify Iris flower species.

## Endpoints
- `GET /` → Health check
- `POST /predict` → Predict species
- `GET /model-info` → Model details

## Run
```bash
uvicorn main:app --reload
```
Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
