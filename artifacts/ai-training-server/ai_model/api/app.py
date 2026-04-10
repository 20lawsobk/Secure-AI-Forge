from __future__ import annotations
import time
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    ScriptGenerateRequest, ScriptGenerateResponse,
    VisualSpecGenerateRequest, VisualSpecGenerateResponse,
    DistributionGenerateRequest, DistributionGenerateResponse,
    ContentGenerateRequest, ContentGenerateResponse,
    MultiPlatformRequest, MultiPlatformResponse,
    BoostSheetCreateRequest, BoostSheetResponse,
    OptimizeRequest, OptimizeResponse,
    TrainRequest, TrainResponse,
    SyntheticDataRequest, SyntheticDataResponse,
    TrainingStatusResponse,
    VideoGenerateRequest, VideoGenerateResponse,
    CinematicTemplatesResponse, CinematicTemplateInfo,
    HealthResponse,
    MultiTrainRequest, MultiGPUStatusResponse,
    HyperGPUStatusResponse, GPUClusterStatusResponse, ClusterScaleRequest,
)
from ..model.tokenizer import SimpleTokenizer
from ..model.transformer import TransformerLM
from ..model.creative_model import CreativeModel
from ..agents.script_agent import ScriptAgent, ScriptRequest
from ..agents.visual_spec_agent import VisualSpecAgent, VisualSpecRequest
from ..agents.distribution_agent import DistributionAgent, DistributionRequest
from ..agents.optimization_agent import OptimizationAgent, OptimizationRequest
from ..boostsheets.repository import BoostSheetRepository
from ..boostsheets.lifecycle import BoostSheetLifecycle
from ..adapters.url_adapter import UrlToBoostSheetAdapter
from ..render_manager import RenderManager

app = FastAPI(
    title="Max Booster AI Content Model",
    description="Custom transformer-based content generation for 8 social platforms",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "model.pt")

tokenizer: SimpleTokenizer = None  # type: ignore
creative_model: CreativeModel = None  # type: ignore
script_agent: ScriptAgent = None  # type: ignore
visual_spec_agent: VisualSpecAgent = None  # type: ignore
distribution_agent: DistributionAgent = None  # type: ignore
optimization_agent: OptimizationAgent = None  # type: ignore
repo: BoostSheetRepository = None  # type: ignore
adapter: UrlToBoostSheetAdapter = None  # type: ignore
render_manager: RenderManager = None  # type: ignore

PLATFORM_NORMALIZE = {
    "googlebusiness": "google_business",
    "google_business": "google_business",
    "twitter": "twitter",
    "x": "twitter",
}


def normalize_platform(p: str) -> str:
    return PLATFORM_NORMALIZE.get(p.lower(), p.lower())


@app.on_event("startup")
async def startup():
    global tokenizer, creative_model, script_agent, visual_spec_agent
    global distribution_agent, optimization_agent, repo, adapter, render_manager

    print("[AI Model] Initializing Max Booster AI Content Model...")
    tokenizer = SimpleTokenizer()

    dim = int(os.environ.get("AI_MODEL_DIM", "512"))
    n_layers = int(os.environ.get("AI_MODEL_LAYERS", "8"))
    n_heads = int(os.environ.get("AI_MODEL_HEADS", "8"))
    max_len = int(os.environ.get("AI_MODEL_MAX_LEN", "1024"))

    if os.path.exists(WEIGHTS_PATH):
        print(f"[AI Model] Loading weights from {WEIGHTS_PATH}")
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "vocab" in checkpoint:
            tokenizer.vocab = checkpoint["vocab"]
            tokenizer.inv_vocab = checkpoint["inv_vocab"]
            tokenizer.next_id = checkpoint["next_id"]
            print(f"[AI Model] Restored vocab ({tokenizer.vocab_size} tokens)")
            state_dict = checkpoint["model_state_dict"]

            if "config" in checkpoint:
                cfg = checkpoint["config"]
                dim = cfg.get("dim", dim)
                n_layers = cfg.get("layers", n_layers)
                n_heads = cfg.get("heads", n_heads)
                max_len = cfg.get("max_len", max_len)
                print(f"[AI Model] Using checkpoint config: dim={dim}, layers={n_layers}, heads={n_heads}, max_len={max_len}")
            else:
                dim = state_dict["token_emb.weight"].shape[1]
                max_len = state_dict["pos_emb.weight"].shape[0]
                n_layers = sum(1 for k in state_dict if k.startswith("layers.") and k.endswith(".attn.qkv.weight"))
                n_heads_inferred = dim // (state_dict["layers.0.attn.qkv.weight"].shape[0] // 3 // (dim // n_heads)) if "layers.0.attn.qkv.weight" in state_dict else n_heads
                n_heads = n_heads_inferred
                print(f"[AI Model] Inferred config from weights: dim={dim}, layers={n_layers}, heads={n_heads}, max_len={max_len}")

            saved_vocab = state_dict["token_emb.weight"].shape[0]
            if saved_vocab != tokenizer.vocab_size:
                print(f"[AI Model] WARNING: Vocab mismatch (checkpoint={saved_vocab}, tokenizer={tokenizer.vocab_size}). Using checkpoint vocab size.")
        else:
            state_dict = checkpoint
            saved_vocab = max(len(tokenizer.vocab), 1000)

        base_model = TransformerLM(
            vocab_size=saved_vocab,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
        )
        base_model.load_state_dict(state_dict)
    else:
        print("[AI Model] No pre-trained weights found, using random initialization")
        base_model = TransformerLM(
            vocab_size=max(len(tokenizer.vocab), 1000),
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
        )

    creative_model = CreativeModel(base_model, tokenizer, device=DEVICE)
    script_agent = ScriptAgent(creative_model)
    visual_spec_agent = VisualSpecAgent(creative_model)
    distribution_agent = DistributionAgent(creative_model)
    optimization_agent = OptimizationAgent(creative_model)
    repo = BoostSheetRepository(path="boostsheets_db")
    adapter = UrlToBoostSheetAdapter(repo)
    render_manager = RenderManager()

    print(f"[AI Model] Model initialized (dim={dim}, layers={n_layers}, heads={n_heads})")
    print(f"[AI Model] Vocab size: {len(tokenizer.vocab)}")
    print(f"[AI Model] Device: {DEVICE}")
    print("[AI Model] Ready to serve requests on port 9878")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=creative_model is not None,
        vocab_size=len(tokenizer.vocab) if tokenizer else 0,
        device=DEVICE,
    )


@app.post("/generate/script", response_model=ScriptGenerateResponse)
async def generate_script(req: ScriptGenerateRequest):
    start = time.time()
    platform = normalize_platform(req.platform)
    result = script_agent.run(ScriptRequest(
        idea=req.idea,
        platform=platform,
        goal=req.goal,
        tone=req.tone,
    ))
    return ScriptGenerateResponse(
        success=True,
        hook=result.hook,
        body=result.body,
        cta=result.cta,
        platform=platform,
        source=getattr(result, "source", "template"),
        processing_time_ms=(time.time() - start) * 1000,
    )


@app.post("/generate/visual-spec", response_model=VisualSpecGenerateResponse)
async def generate_visual_spec(req: VisualSpecGenerateRequest):
    platform = normalize_platform(req.platform)
    result = visual_spec_agent.run(VisualSpecRequest(
        idea=req.idea,
        platform=platform,
        tone=req.tone,
    ))
    return VisualSpecGenerateResponse(
        success=True,
        thumbnail_prompt=result.thumbnail_prompt,
        color_scheme=result.color_scheme,
        layout=result.layout,
        platform=platform,
    )


@app.post("/generate/distribution", response_model=DistributionGenerateResponse)
async def generate_distribution(req: DistributionGenerateRequest):
    platform = normalize_platform(req.platform)
    result = distribution_agent.run(DistributionRequest(
        script=req.script,
        platform=platform,
        goal=req.goal,
    ))
    return DistributionGenerateResponse(
        success=True,
        caption=result.caption,
        content=result.caption,
        hashtags=result.hashtags,
        posting_time=result.posting_time,
        platform=platform,
    )


@app.post("/generate/content", response_model=ContentGenerateResponse)
async def generate_content(req: ContentGenerateRequest):
    start = time.time()
    platform = normalize_platform(req.platform)

    script_result = script_agent.run(ScriptRequest(
        idea=req.topic,
        platform=platform,
        goal=req.goal,
        tone=req.tone,
    ))

    dist_result = None
    if req.include_distribution:
        full_script = f"{script_result.hook}\n{script_result.body}\n{script_result.cta}"
        dist_result = distribution_agent.run(DistributionRequest(
            script=full_script,
            platform=platform,
            goal=req.goal,
        ))

    visual_spec = None
    if req.include_visual_spec:
        vs_result = visual_spec_agent.run(VisualSpecRequest(
            idea=req.topic,
            platform=platform,
            tone=req.tone,
        ))
        visual_spec = {
            "thumbnail_prompt": vs_result.thumbnail_prompt,
            "color_scheme": vs_result.color_scheme,
            "layout": vs_result.layout,
        }

    caption = dist_result.caption if dist_result else f"{script_result.hook}\n{script_result.body}\n{script_result.cta}"
    hashtags = dist_result.hashtags if dist_result else []

    return ContentGenerateResponse(
        success=True,
        platform=platform,
        caption=caption,
        content=caption,
        hashtags=hashtags,
        hook=script_result.hook,
        body=script_result.body,
        cta=script_result.cta,
        source=getattr(script_result, "source", "template"),
        visual_spec=visual_spec,
        posting_time=dist_result.posting_time if dist_result else None,
        processing_time_ms=(time.time() - start) * 1000,
    )


@app.post("/generate/multi-platform", response_model=MultiPlatformResponse)
async def generate_multi_platform(req: MultiPlatformRequest):
    start = time.time()
    generated_content = []

    valid_platforms = [
        "tiktok", "instagram", "youtube", "facebook",
        "twitter", "linkedin", "google_business", "googlebusiness", "threads"
    ]

    for plat in req.platforms:
        platform = normalize_platform(plat)
        if platform not in valid_platforms and plat not in valid_platforms:
            continue

        script_result = script_agent.run(ScriptRequest(
            idea=req.topic,
            platform=platform,
            goal=req.goal,
            tone=req.tone,
        ))

        full_script = f"{script_result.hook}\n{script_result.body}\n{script_result.cta}"

        dist_result = distribution_agent.run(DistributionRequest(
            script=full_script,
            platform=platform,
            goal=req.goal,
        ))

        caption = dist_result.caption
        if req.url:
            caption += f"\n\n\U0001F517 {req.url}"

        entry = {
            "platform": plat,
            "caption": caption,
            "content": caption,
            "hashtags": dist_result.hashtags,
            "posting_time": dist_result.posting_time,
            "hook": script_result.hook,
            "body": script_result.body,
            "cta": script_result.cta,
            "source": getattr(script_result, "source", "template"),
            "format": req.format,
            "target_audience": req.target_audience,
        }

        if req.url:
            entry["sourceUrl"] = req.url

        generated_content.append(entry)

    return MultiPlatformResponse(
        success=True,
        generated_content=generated_content,
        processing_time_ms=(time.time() - start) * 1000,
    )


@app.post("/boostsheet/create", response_model=BoostSheetResponse)
async def create_boostsheet(req: BoostSheetCreateRequest):
    result = {
        "platform": req.platform,
        "content": req.content,
        "format": req.format,
        "url": req.url,
    }
    sheet = adapter.create_from_url_result(result)

    lifecycle = BoostSheetLifecycle(sheet)
    lifecycle.transition("generated_by_agent")
    repo.save(sheet)

    return BoostSheetResponse(
        success=True,
        sheet_id=sheet.sheet_id,
        type=sheet.type,
        platform=sheet.platform,
        blocks=sheet.blocks,
        history=sheet.history,
    )


@app.get("/boostsheet/{sheet_id}", response_model=BoostSheetResponse)
async def get_boostsheet(sheet_id: str):
    try:
        sheet = repo.load(sheet_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BoostSheet not found")
    return BoostSheetResponse(
        success=True,
        sheet_id=sheet.sheet_id,
        type=sheet.type,
        platform=sheet.platform,
        blocks=sheet.blocks,
        history=sheet.history,
    )


@app.get("/boostsheet")
async def list_boostsheets():
    ids = repo.list_ids()
    return {"success": True, "sheet_ids": ids, "count": len(ids)}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    try:
        sheet = repo.load(req.sheet_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BoostSheet not found")

    opt_req = OptimizationRequest(
        sheet=sheet,
        performance=req.performance,
        diffs=[],
        platform=normalize_platform(req.platform),
        goal=req.goal,
    )
    result = optimization_agent.run(opt_req)
    repo.save(result.revised_sheet)

    return OptimizeResponse(
        success=True,
        notes=result.notes,
        sheet_id=sheet.sheet_id,
    )


@app.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest):
    from ..training.dataset import CreativeDataset
    from ..training.trainer import train as run_train, evaluate
    from ..training.config import TrainConfig
    from ..training.synthetic import generate_synthetic_samples

    if req.generate_synthetic or not os.path.exists(req.data_path):
        print(f"[AI Model] Generating {req.synthetic_count} synthetic training samples...")
        generate_synthetic_samples(req.data_path, n=req.synthetic_count)

    if not os.path.exists(req.data_path):
        raise HTTPException(status_code=404, detail=f"Training data not found at {req.data_path}")

    tokenizer.unfreeze()

    max_len = creative_model.model.pos_emb.num_embeddings

    dataset = CreativeDataset(req.data_path, tokenizer, max_len=max_len)
    if len(dataset) == 0:
        tokenizer.freeze()
        raise HTTPException(status_code=400, detail="Training dataset is empty")

    dim = creative_model.model.token_emb.embedding_dim
    cfg = TrainConfig({
        "model": {"dim": dim, "layers": len(creative_model.model.layers), "heads": 4, "max_len": max_len},
        "train": {
            "lr": req.learning_rate,
            "batch_size": req.batch_size,
            "epochs": req.epochs,
            "data_path": req.data_path,
        }
    })

    creative_model.resize_embeddings()

    print(f"[AI Model] Training with {len(dataset)} samples, vocab={tokenizer.vocab_size}")
    run_train(creative_model.model, dataset, tokenizer, cfg, device=DEVICE)

    ppl = evaluate(creative_model.model, dataset, tokenizer, device=DEVICE)
    print(f"[AI Model] Training complete. Perplexity: {ppl}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    n_layers = len(creative_model.model.layers)
    checkpoint = {
        "model_state_dict": creative_model.model.state_dict(),
        "vocab": tokenizer.vocab,
        "inv_vocab": tokenizer.inv_vocab,
        "next_id": tokenizer.next_id,
        "config": {
            "dim": dim,
            "layers": n_layers,
            "heads": cfg.heads,
            "max_len": max_len,
        },
    }
    torch.save(checkpoint, WEIGHTS_PATH)

    import json
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "dim": dim,
        "layers": n_layers,
        "heads": cfg.heads,
        "max_len": max_len,
        "perplexity": ppl,
        "samples": len(dataset),
    }
    with open(os.path.join(WEIGHTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tokenizer.freeze()

    return TrainResponse(
        success=True,
        message=f"Training completed. {len(dataset)} samples, perplexity: {ppl:.2f}" if ppl else "Training completed.",
        epochs_completed=req.epochs,
        vocab_size=tokenizer.vocab_size,
        samples_trained=len(dataset),
        perplexity=ppl,
    )


@app.post("/train/synthetic", response_model=SyntheticDataResponse)
async def generate_synthetic(req: SyntheticDataRequest):
    from ..training.synthetic import generate_synthetic_samples
    generate_synthetic_samples(req.path, n=req.count)
    return SyntheticDataResponse(
        success=True,
        samples_generated=req.count,
        path=req.path,
    )


@app.get("/train/status", response_model=TrainingStatusResponse)
async def training_status():
    from ..training.logger import TrainingLogger
    logger = TrainingLogger()
    return TrainingStatusResponse(
        success=True,
        samples_available=logger.sample_count(),
        weights_exist=os.path.exists(WEIGHTS_PATH),
        data_path="training/boostsheet_samples.json",
    )


@app.post("/train/log-sheet")
async def log_sheet_for_training(sheet_id: str):
    from ..training.logger import TrainingLogger
    try:
        sheet = repo.load(sheet_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BoostSheet not found")
    logger = TrainingLogger()
    logger.log_from_sheet(sheet)
    return {"success": True, "message": f"BoostSheet {sheet_id} logged for training", "total_samples": logger.sample_count()}


@app.post("/train/feedback")
async def receive_training_feedback(payload: dict):
    """
    Receive engagement feedback from the autopilots (A/B test winners and
    high-engagement posts).  The signal is written to a JSON feedback log that
    the CurriculumTrainer reads before each training session to bias its
    synthetic data generation toward visual styles that drove real engagement.
    """
    import json
    import os
    import time

    feedback_dir  = os.path.join(os.path.dirname(__file__), '..', '..', 'training', 'feedback')
    os.makedirs(feedback_dir, exist_ok=True)

    feedback_path = os.path.join(feedback_dir, 'engagement_signals.jsonl')
    record = {**payload, "received_at": time.time()}

    with open(feedback_path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    trigger  = payload.get('trigger', 'unknown')
    platform = payload.get('platform', payload.get('source', 'unknown'))
    eng      = payload.get('engagement_rate', 0)

    print(
        f"[FeedbackRouter] {trigger} | platform={platform} | "
        f"engagement={eng:.2f}% | curriculum_hint={payload.get('curriculum_hint', '')}",
        flush=True,
    )

    try:
        from ..diffusion.training_curriculum import CurriculumTrainer
        trainer = CurriculumTrainer()
        trainer.record_engagement_signal(record)
        print("[FeedbackRouter] CurriculumTrainer signalled ✓", flush=True)
    except Exception as e:
        print(f"[FeedbackRouter] CurriculumTrainer signal skipped: {e}", flush=True)

    return {"success": True, "message": "Feedback received and routed to CurriculumTrainer"}


@app.post("/train/gpu")
async def train_on_gpu(epochs: int = 3, lr: float = 5e-4, lanes: int = 32):
    from ..gpu.gpu_trainer import train_on_digital_gpu
    from ..training.config import TrainConfig
    cfg = TrainConfig()
    cfg.epochs = epochs
    cfg.lr = lr
    cfg.batch_size = 4
    try:
        model, best_val, profile = train_on_digital_gpu(
            data_path="training/boostsheet_samples_v2.json",
            config=cfg,
            lanes=lanes,
        )
        import math
        return {
            "success": True,
            "backend": "digital_gpu",
            "lanes": lanes,
            "epochs": epochs,
            "best_val_loss": round(best_val, 4),
            "ppl": round(math.exp(min(best_val, 20)), 2),
            "profile": profile,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/status")
async def gpu_status():
    from ..gpu.torch_backend import DigitalGPUBackend
    backend = DigitalGPUBackend(lanes=32)
    status = backend.status()
    return {
        "available": True,
        "backend": "digital_gpu",
        **status,
    }


@app.post("/generate/video", response_model=VideoGenerateResponse)
async def generate_video(req: VideoGenerateRequest):
    from ..video.cinematic_engine import render_video_auto
    from ..video.renderer import PLATFORM_RATIOS

    start = time.time()
    platform = normalize_platform(req.platform)

    hook = req.hook
    body = req.body
    cta = req.cta
    source = "custom"

    if req.topic and (not hook or not body):
        script_result = script_agent.run(ScriptRequest(
            idea=req.topic,
            platform=platform,
            goal=req.goal,
            tone=req.tone,
        ))
        hook = hook or script_result.hook
        body = body or script_result.body
        cta = cta or script_result.cta
        source = getattr(script_result, "source", "template")

    ratio = req.aspect_ratio or PLATFORM_RATIOS.get(platform, "9:16")
    quality = req.quality if req.quality in ["quick", "cinematic"] else "cinematic"

    result = render_video_auto(
        hook=hook, body=body, cta=cta,
        platform=platform, aspect_ratio=ratio,
        template=req.template, duration=req.duration,
        artist_name=req.artist_name or "", quality=quality,
        bg_color=req.bg_color, text_color=req.text_color,
        accent_color=req.accent_color,
    )

    if not result.success:
        return VideoGenerateResponse(
            success=False,
            error=result.error,
            platform=platform,
            quality=quality,
            processing_time_ms=(time.time() - start) * 1000,
        )

    return VideoGenerateResponse(
        success=True,
        filename=result.filename,
        url=f"/uploads/videos/{result.filename}",
        duration=result.duration,
        width=result.width,
        height=result.height,
        aspect_ratio=ratio,
        template=req.template,
        template_name=result.template_name,
        platform=platform,
        hook=hook,
        body=body,
        cta=cta,
        source=source,
        quality=quality,
        scenes_rendered=result.scenes_rendered,
        render_time_ms=result.render_time_ms,
        processing_time_ms=(time.time() - start) * 1000,
    )


@app.get("/generate/video/templates", response_model=CinematicTemplatesResponse)
async def get_cinematic_templates():
    from ..video.templates_v2 import get_template_list
    templates = get_template_list()
    return CinematicTemplatesResponse(
        success=True,
        templates=[CinematicTemplateInfo(**t) for t in templates],
    )


@app.post("/render/thumbnail")
async def render_thumbnail(sheet_id: str):
    try:
        sheet = repo.load(sheet_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BoostSheet not found")
    result = render_manager.render_thumbnail(sheet)
    repo.save(sheet)
    return {"success": True, **result}


@app.post("/render/video")
async def render_video(sheet_id: str):
    try:
        sheet = repo.load(sheet_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="BoostSheet not found")
    result = render_manager.render_video(sheet)
    repo.save(sheet)
    return {"success": True, **result}


_multi_orchestrator = None
_multi_train_thread = None


@app.post("/train/multi")
async def train_multi_agent(req: MultiTrainRequest):
    global _multi_orchestrator, _multi_train_thread
    import threading

    if _multi_train_thread and _multi_train_thread.is_alive():
        return {
            "success": False,
            "state": "running",
            "message": "Multi-agent training is already in progress. Check /train/multi/status for updates.",
        }

    from ..training.multi_trainer import TrainingOrchestrator
    from ..training.config import TrainConfig

    cfg = TrainConfig()
    cfg.epochs = req.epochs
    cfg.lr = req.learning_rate
    cfg.batch_size = req.batch_size

    _multi_orchestrator = TrainingOrchestrator(
        data_path=req.data_path,
        config=cfg,
        total_lanes=req.lanes,
        agent_types=req.agent_types,
        resume=True,
    )

    def _run_training():
        try:
            _multi_orchestrator.train_all()
        except Exception as e:
            _multi_orchestrator._state = f"error: {e}"

    _multi_train_thread = threading.Thread(target=_run_training, daemon=True)
    _multi_train_thread.start()

    return {
        "success": True,
        "state": "started",
        "message": "Multi-agent training launched. Poll /train/multi/status for progress.",
        "agents": req.agent_types or ["script", "distribution", "visual_spec", "optimization"],
        "epochs": req.epochs,
        "lanes": req.lanes,
    }


@app.get("/train/multi/status")
async def multi_train_status():
    global _multi_orchestrator
    if _multi_orchestrator is None:
        return {
            "success": True,
            "state": "idle",
            "message": "No multi-agent training has been started",
        }
    result = _multi_orchestrator.get_status()
    return {
        "success": True,
        "state": result.state,
        "total_jobs": result.total_jobs,
        "completed_jobs": result.completed_jobs,
        "failed_jobs": result.failed_jobs,
        "elapsed_s": result.elapsed_s,
        "best_losses": result.best_losses,
        "jobs": result.jobs,
    }


@app.get("/gpu/multi/status", response_model=MultiGPUStatusResponse)
async def multi_gpu_status():
    global _multi_orchestrator
    if _multi_orchestrator is None:
        from ..gpu.multi_backend import MultiStreamBackend
        backend = MultiStreamBackend(total_lanes=32)
        status = backend.status()
        return MultiGPUStatusResponse(
            success=True,
            total_lanes=status["total_lanes"],
            active_streams=status["active_streams"],
            lane_utilization=status["lane_utilization"],
            streams=status["streams"],
            total_vram_mb=status["total_vram_mb"],
        )

    status = _multi_orchestrator.multi_backend.status()
    return MultiGPUStatusResponse(
        success=True,
        total_lanes=status["total_lanes"],
        active_streams=status["active_streams"],
        lane_utilization=status["lane_utilization"],
        streams=status["streams"],
        total_vram_mb=status["total_vram_mb"],
    )


_hyper_backend = None
_cluster_backend = None


def _get_hyper_backend():
    global _hyper_backend
    if _hyper_backend is None:
        from ..gpu.hyper_backend import HyperGPUBackend
        from ..gpu.hyper_core import PrecisionMode
        _hyper_backend = HyperGPUBackend(
            lanes=512,
            tensor_cores=8,
            precision=PrecisionMode.MIXED,
        )
    return _hyper_backend


def _get_cluster_backend():
    global _cluster_backend
    if _cluster_backend is None:
        from ..gpu.hyper_backend import ClusterBackend
        from ..gpu.hyper_core import PrecisionMode
        _cluster_backend = ClusterBackend(
            num_nodes=4,
            lanes_per_node=512,
            tensor_cores_per_node=8,
            precision=PrecisionMode.MIXED,
        )
    return _cluster_backend


@app.get("/gpu/hyper/status")
async def hyper_gpu_status():
    backend = _get_hyper_backend()
    s = backend.status()
    return HyperGPUStatusResponse(
        success=True,
        engine=s["engine"],
        lanes=s["lanes"],
        tensor_cores=s["tensor_cores"],
        precision=s["precision"],
        total_ops=s["total_ops"],
        total_tensor_core_tflops=s["total_tensor_core_tflops"],
        total_compute_ms=s["total_compute_ms"],
        vram=s["vram"],
        memory_pool=s["memory_pool"],
        uptime_s=s["uptime_s"],
    )


@app.get("/gpu/cluster/status")
async def cluster_gpu_status():
    cluster = _get_cluster_backend()
    s = cluster.status()
    return GPUClusterStatusResponse(
        success=True,
        engine=s["engine"],
        num_nodes=s["num_nodes"],
        total_lanes=s["total_lanes"],
        total_tensor_cores=s["total_tensor_cores"],
        nodes_idle=s["nodes_idle"],
        nodes_busy=s["nodes_busy"],
        total_ops=s["total_ops"],
        total_compute_ms=s["total_compute_ms"],
        total_tensor_core_tflops=s["total_tensor_core_tflops"],
        nodes=s["nodes"],
    )


@app.post("/gpu/cluster/scale")
async def cluster_scale(req: ClusterScaleRequest):
    cluster = _get_cluster_backend()
    if req.action == "add":
        nid = cluster.add_node(lanes=req.lanes, tensor_cores=req.tensor_cores)
        return {"success": True, "action": "added", "node_id": nid, "total_nodes": cluster.cluster.num_nodes}
    elif req.action == "remove":
        if req.node_id is None:
            return {"success": False, "error": "node_id required for remove"}
        cluster.remove_node(req.node_id)
        return {"success": True, "action": "removed", "node_id": req.node_id, "total_nodes": cluster.cluster.num_nodes}
    else:
        return {"success": False, "error": f"Unknown action: {req.action}"}


@app.get("/gpu/capabilities")
async def gpu_capabilities():
    backend = _get_hyper_backend()
    cluster = _get_cluster_backend()
    hyper_status = backend.status()
    cluster_status = cluster.status()

    return {
        "success": True,
        "hyper_gpu": {
            "engine": "HyperGPU",
            "lanes": hyper_status["lanes"],
            "tensor_cores": hyper_status["tensor_cores"],
            "precision": hyper_status["precision"],
            "operations": [
                "tensor_core_gemm", "mixed_precision_gemm",
                "flash_attention", "conv2d", "conv3d",
                "layer_norm", "batch_norm", "gelu", "silu",
                "grouped_gemm", "fused_attention_norm",
                "softmax", "add", "gemm_bias_relu",
            ],
            "features": [
                "mixed_precision_training",
                "flash_attention_memory_efficient",
                "tensor_core_acceleration",
                "memory_pooling",
                "instruction_fusion",
            ],
        },
        "cluster": {
            "engine": "HyperGPU Cluster",
            "num_nodes": cluster_status["num_nodes"],
            "total_lanes": cluster_status["total_lanes"],
            "total_tensor_cores": cluster_status["total_tensor_cores"],
            "features": [
                "distributed_training",
                "data_parallel",
                "gradient_all_reduce",
                "elastic_scaling",
                "scatter_gather",
            ],
        },
        "supported_model_types": [
            "transformer", "cnn", "video_generation",
            "diffusion", "vae", "gan", "unet",
            "vision_transformer", "multimodal",
        ],
    }


_data_pipeline = None

def _get_data_pipeline():
    global _data_pipeline
    if _data_pipeline is None:
        from maxbooster_veo_music.training.data_pipeline import DataPipeline
        _data_pipeline = DataPipeline()
    return _data_pipeline


@app.get("/data/pipeline/status")
async def data_pipeline_status():
    pipeline = _get_data_pipeline()
    return pipeline.status()


@app.post("/data/pipeline/download")
async def data_pipeline_download():
    pipeline = _get_data_pipeline()
    results = pipeline.download_all()
    return {"download_results": results}


@app.post("/data/pipeline/load")
async def data_pipeline_load():
    pipeline = _get_data_pipeline()
    counts = pipeline.load_all()
    return {"loaded_counts": counts, "status": pipeline.status()}


@app.get("/data/jamendo/status")
async def jamendo_status():
    pipeline = _get_data_pipeline()
    if not pipeline.jamendo._loaded:
        pipeline.jamendo.load()
    return pipeline.jamendo.status()


@app.get("/data/jamendo/moods")
async def jamendo_mood_distribution():
    pipeline = _get_data_pipeline()
    if not pipeline.jamendo._loaded:
        pipeline.jamendo.load()
    return {
        "mood_distribution": pipeline.jamendo.get_mood_distribution(),
        "total_tracks": len(pipeline.jamendo.tracks),
    }


@app.get("/data/muvisync/status")
async def muvisync_status():
    pipeline = _get_data_pipeline()
    if not pipeline.muvisync._loaded:
        pipeline.muvisync.load()
    return pipeline.muvisync.status()


@app.get("/data/social/status")
async def social_media_status():
    pipeline = _get_data_pipeline()
    if not pipeline.social_media._loaded:
        pipeline.social_media.load()
    return pipeline.social_media.status()


@app.get("/data/social/hashtags")
async def social_media_hashtag_performance():
    pipeline = _get_data_pipeline()
    if not pipeline.social_media._loaded:
        pipeline.social_media.load()
    top_hashtags = dict(list(pipeline.social_media.get_hashtag_performance().items())[:20])
    return {"top_hashtags": top_hashtags}


@app.get("/data/social/timing/{platform}")
async def social_media_optimal_timing(platform: str):
    pipeline = _get_data_pipeline()
    if not pipeline.social_media._loaded:
        pipeline.social_media.load()
    if platform not in ["tiktok", "youtube", "instagram", "twitter", "facebook", "spotify", "shorts", "reels"]:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")
    return {
        "platform": platform,
        "optimal_hours": pipeline.social_media.get_optimal_posting_times(platform),
    }


_campaign_generator = None

def _get_campaign_generator():
    global _campaign_generator
    if _campaign_generator is None:
        from maxbooster_veo_music.pipeline.campaign_generator import CampaignGenerator
        _campaign_generator = CampaignGenerator()
    return _campaign_generator


@app.post("/veo/campaign")
async def generate_veo_campaign(request: dict = {}):
    import numpy as np

    track_id = request.get("track_id", f"track_{int(time.time())}")
    title = request.get("title", "Untitled")
    artist = request.get("artist", "Unknown Artist")
    album = request.get("album")

    story = request.get("story", "")
    mood = request.get("mood", "energetic")
    era = request.get("era", "modern")
    references = request.get("references", [])
    label = request.get("label")
    brand_notes = request.get("brand_notes", "")
    lyrics = request.get("lyrics")
    primary_platforms = request.get("primary_platforms", ["tiktok", "youtube", "instagram"])
    campaign_notes = request.get("campaign_notes", "")

    targets_input = request.get("targets", [])
    if isinstance(targets_input, dict):
        targets_raw = [
            {"platform": p, "goal": g} for p, g in targets_input.items()
        ]
    else:
        targets_raw = targets_input
    if not targets_raw:
        from maxbooster_veo_music.model.platform_heads import PLATFORM_DEFAULTS
        default_goals = {
            "tiktok": "hook_clip", "youtube": "full_video",
            "instagram": "promo_reel", "reels": "promo_reel",
            "shorts": "hook_clip", "spotify_canvas": "loop_visualizer",
            "twitter": "promo_clip", "facebook": "promo_clip",
            "instagram_stories": "teaser_trailer", "snapchat": "hook_clip",
            "pinterest": "promo_reel", "linkedin": "behind_the_scenes",
            "threads": "promo_clip", "twitch": "audio_visualizer",
            "triller": "hook_clip", "vevo": "full_video",
            "audiomack": "audio_visualizer", "soundcloud": "audio_visualizer",
            "apple_music": "loop_visualizer", "amazon_music": "loop_visualizer",
            "tidal": "loop_visualizer", "deezer": "loop_visualizer",
            "pandora": "loop_visualizer", "bandcamp": "full_video",
            "website_embed": "promo_reel", "email_campaign": "email_hero",
            "billboard_digital": "billboard_ad", "live_backdrop": "live_backdrop",
        }
        for p in primary_platforms:
            if p in PLATFORM_DEFAULTS:
                defaults = PLATFORM_DEFAULTS[p]
                goal = default_goals.get(p, "promo_clip")
                targets_raw.append({
                    "platform": p,
                    "goal": goal,
                    "duration_sec": defaults["duration"],
                    "aspect_ratio": defaults["aspect"],
                })

    from maxbooster_veo_music.boostsheets.schema import BoostSheet, PlatformTarget

    targets = [
        PlatformTarget(
            platform=t["platform"],
            goal=t.get("goal", "promo_clip"),
            duration_sec=t.get("duration_sec"),
            aspect_ratio=t.get("aspect_ratio"),
        )
        for t in targets_raw
    ]

    boostsheet = BoostSheet(
        track_id=track_id,
        title=title,
        artist=artist,
        album=album,
        story=story,
        mood=mood,
        era=era,
        references=references,
        label=label,
        brand_notes=brand_notes,
        lyrics=lyrics,
        primary_platforms=primary_platforms,
        campaign_notes=campaign_notes,
        targets=targets,
    )

    duration_sec = request.get("audio_duration_sec", 180.0)
    sample_rate = request.get("sample_rate", 22050)
    n_samples = int(duration_sec * sample_rate)
    audio_waveform = np.random.randn(n_samples).astype(np.float32) * 0.1

    generator = _get_campaign_generator()
    result = generator.generate_campaign(boostsheet, audio_waveform, sample_rate)

    assets_serializable = []
    for asset in result.get("assets", []):
        asset_info = {
            "platform": asset["platform"],
            "goal": asset["goal"],
            "goal_description": asset.get("goal_description", ""),
            "style": asset.get("style", ""),
            "duration_sec": asset["duration_sec"],
            "aspect_ratio": asset["aspect_ratio"],
            "fps": asset["fps"],
            "frame_count": asset["frame_count"],
            "resolution": asset["resolution"],
            "has_text_overlay": asset.get("has_text_overlay", False),
            "beat_synced": asset.get("beat_synced", False),
            "fx_intensity": asset.get("fx_intensity", 0.5),
            "status": "generated",
            "video_url": f"/api/social/veo-campaign/asset/{track_id}/{asset['platform']}",
        }
        assets_serializable.append(asset_info)

    master_info = None
    if result.get("master_video"):
        mv = result["master_video"]
        master_info = {
            "platform": mv["platform"],
            "goal": mv["goal"],
            "goal_description": mv.get("goal_description", ""),
            "style": mv.get("style", ""),
            "duration_sec": mv["duration_sec"],
            "aspect_ratio": mv["aspect_ratio"],
            "fps": mv["fps"],
            "frame_count": mv["frame_count"],
            "resolution": mv["resolution"],
            "has_text_overlay": mv.get("has_text_overlay", False),
            "beat_synced": mv.get("beat_synced", False),
            "fx_intensity": mv.get("fx_intensity", 0.5),
            "status": "generated",
            "video_url": f"/api/social/veo-campaign/asset/{track_id}/youtube_master",
        }

    return {
        "success": True,
        "campaign": {
            "track_id": result["track_id"],
            "artist": result["artist"],
            "title": result["title"],
            "master_video": master_info,
            "assets": assets_serializable,
            "generation_time_s": result["generation_time_s"],
            "gpu_ops": result["gpu_ops"],
            "gpu_compute_ms": result["gpu_compute_ms"],
            "total_platforms": len(assets_serializable),
            "total_frames": sum(a["frame_count"] for a in assets_serializable),
        },
    }


@app.get("/veo/platforms")
async def veo_available_platforms():
    from maxbooster_veo_music.model.platform_heads import PLATFORM_DEFAULTS, PlatformHeads
    platforms = {}
    for name, defaults in PLATFORM_DEFAULTS.items():
        platforms[name] = {
            "default_duration_sec": defaults["duration"],
            "default_aspect_ratio": defaults["aspect"],
            "default_fps": defaults["fps"],
            "recommended_goals": PlatformHeads.get_recommended_goals(name),
        }
    return {
        "platforms": platforms,
        "total_platforms": len(platforms),
    }


@app.get("/veo/goals")
async def veo_available_goals():
    from maxbooster_veo_music.model.platform_heads import PlatformHeads
    goals = PlatformHeads.get_available_goals()
    return {
        "goals": goals,
        "total_goals": len(goals),
    }


@app.get("/veo/recommend/{platform}")
async def veo_recommend_goals(platform: str):
    from maxbooster_veo_music.model.platform_heads import PLATFORM_DEFAULTS, PlatformHeads, GOAL_SPECS
    if platform not in PLATFORM_DEFAULTS:
        raise HTTPException(status_code=404, detail=f"Unknown platform: {platform}")
    defaults = PLATFORM_DEFAULTS[platform]
    recommended = PlatformHeads.get_recommended_goals(platform)
    return {
        "platform": platform,
        "defaults": {
            "duration_sec": defaults["duration"],
            "aspect_ratio": defaults["aspect"],
            "fps": defaults["fps"],
        },
        "recommended_goals": [
            {
                "goal": g,
                "description": GOAL_SPECS[g]["description"],
                "style": GOAL_SPECS[g]["style"],
                "beat_synced": GOAL_SPECS[g]["beat_sync"],
                "has_text_overlay": GOAL_SPECS[g]["text_overlay"],
            }
            for g in recommended if g in GOAL_SPECS
        ],
    }


@app.get("/veo/status")
async def veo_pipeline_status():
    generator = _get_campaign_generator()
    gpu_status = generator.gpu.status()
    return {
        "ready": True,
        "pipeline": "MaxBooster Veo Music",
        "stages": [
            "AudioEncoder", "TextEncoder", "BoostSheetEncoder",
            "VideoLatentVAE", "VideoGenerator", "PlatformHeads",
        ],
        "gpu": {
            "simd_lanes": gpu_status.get("simd_lanes", 512),
            "tensor_cores": gpu_status.get("tensor_cores", 8),
            "total_ops": gpu_status.get("total_ops", 0),
        },
    }


_url_extractor = None

def _get_url_extractor():
    global _url_extractor
    if _url_extractor is None:
        from maxbooster_veo_music.url.extractor import UrlMetadataExtractor
        _url_extractor = UrlMetadataExtractor(timeout=10)
    return _url_extractor


@app.post("/veo/url/metadata")
async def veo_url_metadata(request: dict = {}):
    url = request.get("url", "").strip()
    if not url:
        return {"success": False, "error": "Missing 'url' field"}

    extractor = _get_url_extractor()
    metadata = extractor.extract(url)

    content_type = metadata.get("content_type", "music")
    source = metadata.get("platform", "unknown")
    from maxbooster_veo_music.model.platform_heads import PLATFORM_DEFAULTS

    if content_type == "website":
        suggested_platforms = ["tiktok", "youtube", "instagram", "reels", "shorts", "facebook"]
        suggested_goals = ["ad_creative", "promo_reel", "promo_clip", "teaser_trailer", "hook_clip"]
    else:
        suggested_platforms = ["tiktok", "youtube", "instagram"]
        if source in PLATFORM_DEFAULTS and source not in suggested_platforms:
            suggested_platforms.append(source)
        suggested_goals = None

    response = {
        "success": True,
        "metadata": metadata,
        "suggested_defaults": {
            "mood": metadata.get("mood", "energetic"),
            "era": metadata.get("era", "modern"),
            "story": metadata.get("story", ""),
            "primary_platforms": suggested_platforms,
        },
        "ready_to_generate": metadata["confidence"] >= 0.5,
    }

    if suggested_goals:
        response["suggested_defaults"]["recommended_goals"] = suggested_goals

    return response


@app.post("/veo/url/campaign")
async def veo_url_campaign(request: dict = {}):
    import numpy as np

    url = request.get("url", "").strip()
    if not url:
        return {"success": False, "error": "Missing 'url' field"}

    extractor = _get_url_extractor()
    metadata = extractor.extract(url)

    overrides = {}
    for key in ["title", "artist", "album", "mood", "era", "story",
                 "primary_platforms", "targets", "lyrics", "brand_notes",
                 "campaign_notes", "audio_duration_sec"]:
        if key in request and request[key]:
            overrides[key] = request[key]

    campaign_request = extractor.metadata_to_campaign_request(metadata, overrides)

    track_id = request.get("track_id", f"url_{int(time.time())}")
    title = campaign_request["title"]
    artist = campaign_request["artist"]
    mood = campaign_request["mood"]
    era = campaign_request["era"]
    story = campaign_request["story"]
    primary_platforms = campaign_request["primary_platforms"]

    targets_input_url = campaign_request.get("targets", [])
    if isinstance(targets_input_url, dict):
        targets_raw = [
            {"platform": p, "goal": g} for p, g in targets_input_url.items()
        ]
    else:
        targets_raw = targets_input_url
    if not targets_raw:
        from maxbooster_veo_music.model.platform_heads import PLATFORM_DEFAULTS
        default_goals = {
            "tiktok": "hook_clip", "youtube": "full_video",
            "instagram": "promo_reel", "reels": "promo_reel",
            "shorts": "hook_clip", "spotify_canvas": "loop_visualizer",
            "twitter": "promo_clip", "facebook": "promo_clip",
            "soundcloud": "audio_visualizer", "apple_music": "loop_visualizer",
            "bandcamp": "full_video", "audiomack": "audio_visualizer",
            "deezer": "loop_visualizer", "tidal": "loop_visualizer",
            "amazon_music": "loop_visualizer", "pandora": "loop_visualizer",
            "vevo": "full_video",
        }
        for p in primary_platforms:
            if p in PLATFORM_DEFAULTS:
                defaults = PLATFORM_DEFAULTS[p]
                goal = default_goals.get(p, "promo_clip")
                targets_raw.append({
                    "platform": p,
                    "goal": goal,
                    "duration_sec": defaults["duration"],
                    "aspect_ratio": defaults["aspect"],
                })

    from maxbooster_veo_music.boostsheets.schema import BoostSheet, PlatformTarget

    targets = [
        PlatformTarget(
            platform=t["platform"],
            goal=t.get("goal", "promo_clip"),
            duration_sec=t.get("duration_sec"),
            aspect_ratio=t.get("aspect_ratio"),
        )
        for t in targets_raw
    ]

    boostsheet = BoostSheet(
        track_id=track_id,
        title=title,
        artist=artist,
        album=campaign_request.get("album"),
        story=story,
        mood=mood,
        era=era,
        references=overrides.get("references", []),
        label=overrides.get("label"),
        brand_notes=overrides.get("brand_notes", ""),
        lyrics=campaign_request.get("lyrics"),
        primary_platforms=primary_platforms,
        campaign_notes=overrides.get("campaign_notes", ""),
        targets=targets,
    )

    duration_sec = campaign_request.get("audio_duration_sec", 180.0)
    sample_rate = 22050
    n_samples = int(duration_sec * sample_rate)
    audio_waveform = np.random.randn(n_samples).astype(np.float32) * 0.1

    generator = _get_campaign_generator()
    result = generator.generate_campaign(boostsheet, audio_waveform, sample_rate)

    assets_serializable = []
    for asset in result.get("assets", []):
        asset_info = {
            "platform": asset["platform"],
            "goal": asset["goal"],
            "goal_description": asset.get("goal_description", ""),
            "style": asset.get("style", ""),
            "duration_sec": asset["duration_sec"],
            "aspect_ratio": asset["aspect_ratio"],
            "fps": asset["fps"],
            "frame_count": asset["frame_count"],
            "resolution": asset["resolution"],
            "has_text_overlay": asset.get("has_text_overlay", False),
            "beat_synced": asset.get("beat_synced", False),
            "fx_intensity": asset.get("fx_intensity", 0.5),
            "status": "generated",
            "video_url": f"/api/social/veo-campaign/asset/{track_id}/{asset['platform']}",
        }
        assets_serializable.append(asset_info)

    master_info = None
    if result.get("master_video"):
        mv = result["master_video"]
        master_info = {
            "platform": mv["platform"],
            "goal": mv["goal"],
            "goal_description": mv.get("goal_description", ""),
            "style": mv.get("style", ""),
            "duration_sec": mv["duration_sec"],
            "aspect_ratio": mv["aspect_ratio"],
            "fps": mv["fps"],
            "frame_count": mv["frame_count"],
            "resolution": mv["resolution"],
            "has_text_overlay": mv.get("has_text_overlay", False),
            "beat_synced": mv.get("beat_synced", False),
            "fx_intensity": mv.get("fx_intensity", 0.5),
            "status": "generated",
            "video_url": f"/api/social/veo-campaign/asset/{track_id}/youtube_master",
        }

    return {
        "success": True,
        "source": {
            "url": metadata["url"],
            "platform": metadata["platform"],
            "extraction_method": metadata["extraction_method"],
            "confidence": metadata["confidence"],
        },
        "campaign": {
            "track_id": result["track_id"],
            "artist": result["artist"],
            "title": result["title"],
            "master_video": master_info,
            "assets": assets_serializable,
            "generation_time_s": result["generation_time_s"],
            "gpu_ops": result["gpu_ops"],
            "gpu_compute_ms": result["gpu_compute_ms"],
            "total_platforms": len(assets_serializable),
            "total_frames": sum(a["frame_count"] for a in assets_serializable),
        },
    }
