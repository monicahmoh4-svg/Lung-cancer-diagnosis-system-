FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
