FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Copy app
WORKDIR /app
COPY app.py .

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
