if __name__ == "__main__":
    import uvicorn
    from app.app_config import config as CONFIG

    uvicorn.run("app.main:app", host="0.0.0.0", port=CONFIG.port, reload=True)