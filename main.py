from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from crop import crop_fingertip
import io

app = FastAPI()

@app.post("/crop-fingertip")
def crop_fingertip_endpoint(file: UploadFile = File(...)):
    # Read file into numpy array
    contents = file.file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    cropped = crop_fingertip(image)
    if cropped is None:
        raise HTTPException(status_code=404, detail="No fingertip detected")

    # Encode cropped image to JPEG
    _, buffer = cv2.imencode('.jpg', cropped)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
