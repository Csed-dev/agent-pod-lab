import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import experiments, monitoring
from orchestrator.errors import (
    ExperimentNotFoundError,
    ExperimentNotReadyError,
    GpuUnavailableError,
    OrchestratorError,
    PodPoolFullError,
)

load_dotenv()

app = FastAPI(title="Agent Pod Lab", version="0.1.0")

cors_origins = os.environ.get("CORS_ORIGINS", "")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(monitoring.router)
app.include_router(experiments.router)


@app.exception_handler(ExperimentNotFoundError)
async def not_found_handler(request: Request, exc: ExperimentNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(ExperimentNotReadyError)
async def not_ready_handler(request: Request, exc: ExperimentNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": str(exc)})


@app.exception_handler(PodPoolFullError)
async def pool_full_handler(request: Request, exc: PodPoolFullError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(GpuUnavailableError)
async def gpu_unavailable_handler(request: Request, exc: GpuUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(OrchestratorError)
async def orchestrator_error_handler(request: Request, exc: OrchestratorError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})
