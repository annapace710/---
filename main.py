from fastapi import FastAPI, File, UploadFile

app = FastAPI()

"/upload"
async def upload_video(video_file: UploadFile = File(...)):
    # Здесь вы будете писать логику обработки видео
    return {"message": "Видео получено"}