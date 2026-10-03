FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# tải sẵn checkpoint (~165 MB) lúc build để lần gọi đầu không phải chờ
RUN python -c "from piano_transcription_inference import PianoTranscription; PianoTranscription(device='cpu')"
COPY --chown=user app.py .
EXPOSE 7860
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","7860"]
