from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class PlatformEnum(str, Enum):
    tiktok = "tiktok"
    instagram = "instagram"
    youtube = "youtube"
    facebook = "facebook"
    twitter = "twitter"
    linkedin = "linkedin"
    google_business = "google_business"
    googlebusiness = "googlebusiness"
    threads = "threads"


class GoalEnum(str, Enum):
    growth = "growth"
    conversion = "conversion"
    nurture = "nurture"


class ToneEnum(str, Enum):
    edgy = "edgy"
    playful = "playful"
    serious = "serious"
    professional = "professional"
    casual = "casual"
    energetic = "energetic"
    promotional = "promotional"
    default = "default"


class ScriptGenerateRequest(BaseModel):
    idea: str
    platform: str = "tiktok"
    goal: str = "growth"
    tone: str = "energetic"


class ScriptGenerateResponse(BaseModel):
    success: bool
    hook: str
    body: str
    cta: str
    platform: str
    source: str = "template"
    processing_time_ms: float


class VisualSpecGenerateRequest(BaseModel):
    idea: str
    platform: str = "tiktok"
    tone: str = "playful"


class VisualSpecGenerateResponse(BaseModel):
    success: bool
    thumbnail_prompt: str
    color_scheme: str
    layout: str
    platform: str


class DistributionGenerateRequest(BaseModel):
    script: str
    platform: str = "tiktok"
    goal: str = "growth"


class DistributionGenerateResponse(BaseModel):
    success: bool
    caption: str
    content: str
    hashtags: List[str]
    posting_time: str
    platform: str


class ContentGenerateRequest(BaseModel):
    platform: str = "tiktok"
    topic: str = "new music"
    tone: str = "energetic"
    goal: str = "growth"
    include_hashtags: bool = True
    include_visual_spec: bool = False
    include_distribution: bool = True


class ContentGenerateResponse(BaseModel):
    success: bool
    platform: str
    caption: str
    content: str
    hashtags: List[str]
    hook: str
    body: str
    cta: str
    source: str = "template"
    visual_spec: Optional[Dict[str, Any]] = None
    posting_time: Optional[str] = None
    processing_time_ms: float


class MultiPlatformRequest(BaseModel):
    platforms: List[str] = ["tiktok", "instagram"]
    topic: str = "new music"
    tone: str = "energetic"
    goal: str = "growth"
    include_hashtags: bool = True
    target_audience: Optional[str] = None
    format: str = "text"
    url: Optional[str] = None


class MultiPlatformResponse(BaseModel):
    success: bool
    generated_content: List[Dict[str, Any]]
    processing_time_ms: float


class BoostSheetCreateRequest(BaseModel):
    platform: str = "tiktok"
    content: str = ""
    format: str = "text"
    url: Optional[str] = None
    goal: str = "growth"
    tone: str = "default"


class BoostSheetResponse(BaseModel):
    success: bool
    sheet_id: str
    type: str
    platform: str
    blocks: Dict[str, Any]
    history: List[str]


class OptimizeRequest(BaseModel):
    sheet_id: str
    performance: Dict[str, float] = Field(default_factory=dict)
    platform: str = "tiktok"
    goal: str = "growth"


class OptimizeResponse(BaseModel):
    success: bool
    notes: List[str]
    sheet_id: str


class TrainRequest(BaseModel):
    data_path: str = "training/boostsheet_samples.json"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-4
    generate_synthetic: bool = False
    synthetic_count: int = 50


class TrainResponse(BaseModel):
    success: bool
    message: str
    epochs_completed: int
    vocab_size: Optional[int] = None
    samples_trained: Optional[int] = None
    perplexity: Optional[float] = None


class SyntheticDataRequest(BaseModel):
    path: str = "training/boostsheet_samples.json"
    count: int = 50


class SyntheticDataResponse(BaseModel):
    success: bool
    samples_generated: int
    path: str


class TrainingStatusResponse(BaseModel):
    success: bool
    samples_available: int
    weights_exist: bool
    data_path: str


class VideoGenerateRequest(BaseModel):
    hook: str = ""
    body: str = ""
    cta: str = ""
    platform: str = "tiktok"
    aspect_ratio: Optional[str] = None
    template: str = "cinematic_promo"
    duration: float = 10.0
    bg_color: Optional[str] = None
    text_color: Optional[str] = None
    accent_color: Optional[str] = None
    artist_name: Optional[str] = None
    topic: Optional[str] = None
    goal: str = "growth"
    tone: str = "energetic"
    quality: str = "cinematic"


class VideoGenerateResponse(BaseModel):
    success: bool
    filename: str = ""
    url: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    aspect_ratio: str = ""
    template: str = ""
    template_name: str = ""
    platform: str = ""
    hook: str = ""
    body: str = ""
    cta: str = ""
    source: str = "template"
    quality: str = "cinematic"
    scenes_rendered: int = 0
    render_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class CinematicTemplateInfo(BaseModel):
    id: str
    name: str
    description: str
    category: str
    transition: str
    color_grade: str


class CinematicTemplatesResponse(BaseModel):
    success: bool
    templates: List[CinematicTemplateInfo]
    quick_templates: List[str] = ["promo", "lyric", "announcement", "minimal", "neon"]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocab_size: int
    device: str
    version: str = "1.0.0"


class MultiTrainRequest(BaseModel):
    data_path: str = "training/boostsheet_samples_v2.json"
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 5e-4
    lanes: int = 32
    agent_types: Optional[List[str]] = None

class MultiTrainResponse(BaseModel):
    success: bool
    state: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    elapsed_s: float
    best_losses: dict
    jobs: dict
    gpu_status: dict

class MultiGPUStatusResponse(BaseModel):
    success: bool
    total_lanes: int
    active_streams: int
    lane_utilization: dict
    streams: dict
    total_vram_mb: float


class HyperGPUStatusResponse(BaseModel):
    success: bool
    engine: str
    lanes: int
    tensor_cores: int
    precision: str
    total_ops: int
    total_tensor_core_tflops: float
    total_compute_ms: float
    vram: dict
    memory_pool: dict
    uptime_s: float


class GPUClusterStatusResponse(BaseModel):
    success: bool
    engine: str
    num_nodes: int
    total_lanes: int
    total_tensor_cores: int
    nodes_idle: int
    nodes_busy: int
    total_ops: int
    total_compute_ms: float
    total_tensor_core_tflops: float
    nodes: dict


class ClusterScaleRequest(BaseModel):
    action: str = "add"
    lanes: int = 512
    tensor_cores: int = 8
    node_id: Optional[int] = None
