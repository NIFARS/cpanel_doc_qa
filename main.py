from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import re

app = FastAPI(title="Document Q&A", version="1.0")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Railway Metal deployment successful"}

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Form(...)):
    return {"success": True, "message": "Upload ready", "filename": file.filename}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
