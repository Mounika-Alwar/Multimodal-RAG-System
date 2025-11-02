FROM python:3.9-slim

WORKDIR /app

COPY . /app

COPY .env /app/.env

RUN pip install --no-cache-dir -r requirements.txt --timeout 300 --retries 10 -i https://pypi.org/simple

EXPOSE 8654

CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","8654"]
