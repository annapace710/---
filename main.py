import os
from fastapi import FastAPI, File, UploadFile
from typing import List
import cv2 # OpenCV
import mediapipe as mp # MediaPipe
import numpy as np

app = FastAPI()


@app.post("/upload")
async def upload_video(video_file: UploadFile = File(...)):
    # Здесь вы будете писать логику обработки видео
    return {"message": "Видео получено"}