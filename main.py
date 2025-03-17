import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from AudioSeparation import audio_speaker_separation  # 导入你已经写好的音频处理逻辑

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "uploaded_files"
OUTPUT_FOLDER = "output_files"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """返回上传文件的HTML页面"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/process-audio/")
async def process_audio(files: list[UploadFile] = File(...)):
    """处理上传的音频文件"""
    try:
        download_links = []
        for file in files:
            # 保存上传的文件
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # 调用音频分离和转录函数
            try:
                output_file = audio_speaker_separation(file_path, OUTPUT_FOLDER)
                download_link = f"/download/{os.path.basename(output_file)}"
                download_links.append(download_link)

                # 删除上传的临时文件
                if os.path.exists(file_path):
                    os.remove(file_path)

            except Exception as e:
                # 如果处理失败，也删除上传的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

        return {"message": "Files processed successfully", "download_links": download_links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """下载处理后的文本文件"""
    file_path = os.path.join(OUTPUT_FOLDER, file_name)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=file_name
        )
    raise HTTPException(status_code=404, detail="File not found")

