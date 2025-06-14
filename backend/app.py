from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AI Exam Generator Pro")

@app.get("/")
async def root():
    return {"message": "AI Exam Generator is running!"}

if __name__ == "__main__":
    print("🚀 Starting AI Exam Generator...")
    uvicorn.run(app, host="0.0.0.0", port=8000)