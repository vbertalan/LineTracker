import LineTracker as lt
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/get-message")
async def read_root():
    return{"Message":"API connected."}

@app.get("/cluster_html", response_class=HTMLResponse)
async def cluster_html(logfile:str, token:str):
    lc = lt.LineTracker()
    return(lc.cluster(logfile = logfile, token=token))