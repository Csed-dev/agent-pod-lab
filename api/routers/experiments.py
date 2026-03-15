from fastapi import APIRouter, Query

from api.dependencies import get_manager

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/status")
def status() -> dict:
    mgr = get_manager()
    return {"status": mgr.experiments_status()}


@router.get("/ready")
def ready() -> dict:
    mgr = get_manager()
    return {"ready": mgr.get_ready()}


@router.get("/detail")
def detail() -> dict:
    mgr = get_manager()
    return {"detail": mgr.available_experiments_detail()}


@router.get("/export/csv")
def export_csv() -> dict:
    mgr = get_manager()
    return {"csv": mgr.export_csv()}


@router.get("/results")
def results(
    tags: str | None = Query(None, description="Comma-separated tags to filter by"),
    status: str | None = Query(None, description="Filter by status: completed, failed, etc."),
) -> dict:
    mgr = get_manager()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    return {"results": mgr.all_results(tags=tag_list, status=status)}


@router.get("/results/{experiment_name}")
def result(experiment_name: str) -> dict:
    mgr = get_manager()
    return {"result": mgr.result(experiment_name)}


@router.get("/results/{experiment_name}/acceptance")
def acceptance(experiment_name: str) -> dict:
    mgr = get_manager()
    return {"experiment": experiment_name, "meets_acceptance": mgr.meets_acceptance(experiment_name)}
