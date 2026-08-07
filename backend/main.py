from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
import aiofiles
import uuid
import json
import re
import openai
from datetime import datetime

from video_processor import VideoProcessor
from transcriber import Transcriber
from summarizer import Summarizer
from translator import Translator
from database import db

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI视频转录器", version="1.0.0")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

# 创建临时目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# 初始化处理器
video_processor = VideoProcessor()
transcriber = Transcriber()
summarizer = Summarizer()
translator = Translator()

# 存储正在处理的URL，防止重复处理
processing_urls = set()
# 存储活跃的任务对象，用于控制和取消
active_tasks = {}
# 存储SSE连接，用于实时推送状态更新
sse_connections = {}
# 内存任务缓存（用于快速访问，持久化由数据库负责）
tasks = {}

# 本地上传：允许的类型与大小上限（MB），可用环境变量 UPLOAD_MAX_MB 调整
UPLOAD_ALLOWED_EXT = frozenset({".txt", ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mkv", ".ogg", ".flac"})
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "200"))


def _sanitize_title_for_filename(title: str) -> str:
    """将视频标题清洗为安全的文件名片段。"""
    if not title:
        return "untitled"
    # 仅保留字母数字、下划线、连字符与空格
    safe = re.sub(r"[^\w\-\s]", "", title)
    # 压缩空白并转为下划线
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    # 最长限制，避免过长文件名问题
    return safe[:80] or "untitled"


def _txt_to_raw_transcript_markdown(body: str) -> str:
    """将纯文本包装为与 Whisper 输出结构一致的 Markdown。"""
    text = body.strip() if body.strip() else "(empty)"
    return "\n".join([
        "# Video Transcription",
        "",
        "**Detected Language:**",
        "**Language Probability:** —",
        "",
        "## Transcription Content",
        "",
        text,
    ])


async def _run_post_extract_pipeline(
    task_id: str,
    raw_script: str,
    video_title: str,
    source_ref: str,
    summary_language: str,
    request_summarizer: Summarizer,
    dedup_url: Optional[str] = None,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
) -> None:
    """取得 raw_script 后的共用管线：归档、优化、翻译、摘要、广播。"""
    short_id = task_id.replace("-", "")[:6]
    safe_title = _sanitize_title_for_filename(video_title)

    try:
        raw_md_filename = f"raw_{safe_title}_{short_id}.md"
        raw_md_path = TEMP_DIR / raw_md_filename
        with open(raw_md_path, "w", encoding="utf-8") as f:
            f.write((raw_script or "") + f"\n\nsource: {source_ref}\n")
        # 更新数据库和内存缓存
        await db.update_task(task_id, raw_script_file=raw_md_filename)
        updated_task = await db.get_task(task_id)
        if updated_task:
            tasks[task_id] = updated_task
        await broadcast_task_update(task_id, tasks[task_id])
    except Exception as e:
        logger.error(f"保存原始转录Markdown失败: {e}")

    await db.update_task(task_id, progress=55, message="正在优化转录文本...")
    updated_task = await db.get_task(task_id)
    if updated_task:
        tasks[task_id] = updated_task
    await broadcast_task_update(task_id, tasks[task_id])

    script = await request_summarizer.optimize_transcript(raw_script)

    script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_ref}\n"

    detected_language = transcriber.get_detected_language(raw_script)
    detected_language = (detected_language or "").strip()
    if not detected_language:
        detected_language = translator.infer_language_code(raw_script)
    detected_language = translator.normalize_lang_code(detected_language) or detected_language

    logger.info(f"检测到的语言: {detected_language}, 摘要语言: {summary_language}")

    translation_content = None
    translation_filename = None
    translation_path = None

    eff_key = (api_key or "").strip()
    eff_base = (model_base_url or "").strip().rstrip("/")
    if eff_key:
        request_translator = Translator(
            api_key=eff_key,
            base_url=eff_base or None,
            model=model_id or None,
        )
    else:
        request_translator = translator

    need_translation = translator.languages_differ_for_translation(
        detected_language, summary_language
    )

    if need_translation:
        logger.info(f"需要翻译: {detected_language} -> {summary_language}")
        await db.update_task(task_id, progress=70, message="正在生成翻译...")
        updated_task = await db.get_task(task_id)
        if updated_task:
            tasks[task_id] = updated_task
        await broadcast_task_update(task_id, tasks[task_id])

        translation_content = await request_translator.translate_text(
            script, summary_language, detected_language
        )
        translation_with_title = f"# {video_title}\n\n{translation_content}\n\nsource: {source_ref}\n"
        translation_filename = f"translation_{safe_title}_{short_id}.md"
        translation_path = TEMP_DIR / translation_filename
        async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
            await f.write(translation_with_title)
        
        # 更新数据库和内存缓存
        await db.update_task(
            task_id,
            translation=translation_with_title,
            translation_path=str(translation_path),
            translation_filename=translation_filename
        )
        updated_task = await db.get_task(task_id)
        if updated_task:
            tasks[task_id] = updated_task
        await broadcast_task_update(task_id, tasks[task_id])
    else:
        logger.info(
            f"不需要翻译: detected_language={detected_language}, summary_language={summary_language}, "
            f"need_translation={need_translation}"
        )

    await db.update_task(task_id, progress=80, message="正在生成摘要...")
    updated_task = await db.get_task(task_id)
    if updated_task:
        tasks[task_id] = updated_task
    await broadcast_task_update(task_id, tasks[task_id])

    summary = await request_summarizer.summarize(script, summary_language, video_title, summary_style)
    summary_with_source = summary + f"\n\nsource: {source_ref}\n"

    script_filename = f"transcript_{task_id}.md"
    script_path = TEMP_DIR / script_filename
    async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
        await f.write(script_with_title)

    new_script_filename = f"transcript_{safe_title}_{short_id}.md"
    new_script_path = TEMP_DIR / new_script_filename
    try:
        if script_path.exists():
            script_path.rename(new_script_path)
            script_path = new_script_path
    except Exception:
        pass

    summary_filename = f"summary_{safe_title}_{short_id}.md"
    summary_path = TEMP_DIR / summary_filename
    async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
        await f.write(summary_with_source)

    # 更新数据库和内存缓存
    await db.update_task(
        task_id,
        status="completed",
        progress=100,
        message="处理完成！",
        video_title=video_title,
        script=script_with_title,
        summary=summary_with_source,
        script_path=str(script_path),
        summary_path=str(summary_path),
        short_id=short_id,
        safe_title=safe_title,
        detected_language=detected_language,
        summary_language=summary_language,
    )
    updated_task = await db.get_task(task_id)
    if updated_task:
        tasks[task_id] = updated_task
    await broadcast_task_update(task_id, tasks[task_id])
    logger.info(f"任务完成，准备广播最终状态: {task_id}")
    logger.info(f"最终状态已广播: {task_id}")

    if dedup_url:
        processing_urls.discard(dedup_url)
    if task_id in active_tasks:
        del active_tasks[task_id]


@app.get("/")
async def read_root():
    """返回前端页面"""
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))


@app.post("/api/models")
async def list_models(
    base_url: str = Form(default=""),
    api_key:  str = Form(default=""),
):
    """Proxy: fetch model list from any OpenAI-compatible API."""
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    effective_url = base_url.rstrip("/") or os.getenv("OPENAI_BASE_URL") or None

    if not effective_key:
        raise HTTPException(status_code=400, detail="API key is required")

    try:
        client = openai.OpenAI(api_key=effective_key, base_url=effective_url)
        resp   = await asyncio.to_thread(client.models.list)
        models = [{"id": m.id, "name": getattr(m, "name", m.id)} for m in resp.data]
        # Sort by id for readability
        models.sort(key=lambda x: x["id"])
        return {"data": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _enqueue_upload_job(
    file: UploadFile,
    summary_language: str,
    api_key: str,
    model_base_url: str,
    model_id: str,
) -> dict:
    """保存上传文件并入队 process_upload_task，返回 {task_id, message}。"""
    raw_name = file.filename or "upload.bin"
    if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    safe_name = os.path.basename(raw_name)
    ext = Path(safe_name).suffix.lower()
    if ext not in UPLOAD_ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext or '(none)'}",
        )

    max_bytes = UPLOAD_MAX_MB * 1024 * 1024
    task_id = str(uuid.uuid4())
    unique_stem = task_id.replace("-", "")[:12]
    dest = TEMP_DIR / f"upload_{unique_stem}{ext}"

    total = 0
    with open(dest, "wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                try:
                    dest.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds limit of {UPLOAD_MAX_MB} MB",
                )
            out_f.write(chunk)

    if total == 0:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Empty file")

    video_title = _sanitize_title_for_filename(Path(safe_name).stem) or "upload"
    source_label = f"upload:{safe_name}"

    # 创建数据库任务记录
    await db.create_task(task_id, url=source_label, video_title=video_title)
    # 加载到内存缓存
    task_from_db = await db.get_task(task_id)
    if task_from_db:
        tasks[task_id] = task_from_db

    bg = asyncio.create_task(
        process_upload_task(
            task_id,
            dest,
            safe_name,
            video_title,
            ext,
            summary_language,
            api_key,
            model_base_url,
            model_id,
        )
    )
    active_tasks[task_id] = bg

    return {"task_id": task_id, "message": "任务已创建，正在处理中..."}


@app.post("/api/process-video")
async def process_video(
    url: str = Form(default=""),
    summary_language: str = Form(default="zh"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
    file: Optional[UploadFile] = File(None),
):
    """
    处理视频链接或本地上传（multipart 中带 file 且无有效 URL 时走上传流程）。
    上传与 URL 共用此路径，便于反向代理只放行 /api/process-video 的环境。
    """
    try:
        if file is not None and (file.filename or "").strip():
            return await _enqueue_upload_job(
                file, summary_language, api_key, model_base_url, model_id
            )

        stripped = (url or "").strip()
        if not stripped:
            raise HTTPException(
                status_code=400,
                detail="Provide a video URL or upload a file",
            )

        url = stripped

        # 检查是否已经在处理相同的URL
        if url in processing_urls:
            # 查找现有任务
            for tid, task in tasks.items():
                if task.get("url") == url:
                    return {"task_id": tid, "message": "该视频正在处理中，请等待..."}
            
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 标记URL为正在处理
        processing_urls.add(url)
        
        # 创建数据库任务记录
        await db.create_task(task_id, url=url)
        # 加载到内存缓存
        task_from_db = await db.get_task(task_id)
        if task_from_db:
            tasks[task_id] = task_from_db
        
        # 创建并跟踪异步任务
        task = asyncio.create_task(process_video_task(task_id, url, summary_language, api_key, model_base_url, model_id))
        active_tasks[task_id] = task
        
        return {"task_id": task_id, "message": "任务已创建，正在处理中..."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


async def process_video_task(
    task_id: str,
    url: str,
    summary_language: str,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
):
    """
    异步处理视频任务
    """
    try:
        # ── 阶段一：优先尝试获取平台字幕（快速路径） ──────────────────────
        await db.update_task(task_id, status="processing", progress=10, message="正在检测视频字幕...")
        updated_task = await db.get_task(task_id)
        if updated_task:
            tasks[task_id] = updated_task
        await broadcast_task_update(task_id, tasks[task_id])
        await asyncio.sleep(0.1)

        # 如果前端传入了 API 凭据，创建专用 Summarizer（线程安全，覆盖全局实例）
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"使用前端提供的 API Key，base_url={effective_url}, model={model_id or 'default'}")
        else:
            request_summarizer = summarizer  # 全局实例（使用环境变量）

        subtitle_text, sub_title, sub_lang = await video_processor.fetch_subtitles(url, TEMP_DIR)

        if subtitle_text:
            # ── 快速路径：有字幕，跳过音频下载和 Whisper ──────────────────
            video_title = sub_title
            raw_script = subtitle_text
            # 把语言写入 transcriber，保持下游逻辑一致
            transcriber.last_detected_language = sub_lang

            await db.update_task(task_id, progress=40, message=f"字幕获取成功（{sub_lang}），正在处理文本...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])
        else:
            # ── 慢速路径：无字幕，下载音频 → Whisper 转录 ─────────────────
            await db.update_task(task_id, progress=15, message="未找到字幕，正在下载视频音频...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path, video_title = await video_processor.download_and_convert(
                url, TEMP_DIR, prefetched_title=sub_title or None
            )

            await db.update_task(task_id, progress=35, message="音频下载完成，准备转录...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            await db.update_task(task_id, progress=40, message="正在转录音频（Whisper）...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            raw_script = await transcriber.transcribe(audio_path)

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=url,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            dedup_url=url,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
        )

        # 不要立即删除临时文件！保留给用户下载
        # 文件会在一定时间后自动清理或用户手动清理

    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        # 从处理列表中移除URL
        processing_urls.discard(url)
        
        # 从活跃任务列表中移除
        if task_id in active_tasks:
            del active_tasks[task_id]
            
        # 更新数据库和内存缓存为错误状态
        await db.update_task(
            task_id,
            status="error",
            error=str(e),
            message=f"处理失败: {str(e)}"
        )
        updated_task = await db.get_task(task_id)
        if updated_task:
            tasks[task_id] = updated_task
        await broadcast_task_update(task_id, tasks[task_id])


@app.post("/api/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    summary_language: str = Form(default="zh"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
):
    """独立上传入口；逻辑与 multipart 带 file 的 /api/process-video 相同。"""
    return await _enqueue_upload_job(
        file, summary_language, api_key, model_base_url, model_id
    )


async def process_upload_task(
    task_id: str,
    saved_path: Path,
    original_name: str,
    video_title: str,
    ext_lower: str,
    summary_language: str,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
):
    source_ref = f"upload:{original_name}"
    try:
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(
                f"上传任务使用前端 API Key，base_url={effective_url}, model={model_id or 'default'}"
            )
        else:
            request_summarizer = summarizer

        if ext_lower == ".txt":
            await db.update_task(task_id, progress=20, message="正在读取文本文件...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            body = saved_path.read_text(encoding="utf-8", errors="replace")
            if not body.strip():
                raise Exception("文本文件为空")
            transcriber.last_detected_language = None
            raw_script = _txt_to_raw_transcript_markdown(body)
        else:
            await db.update_task(task_id, progress=15, message="正在转换音频格式...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path = await video_processor.normalize_local_media_to_m4a(saved_path, TEMP_DIR)

            await db.update_task(task_id, progress=35, message="音频准备完成，准备转录...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            await db.update_task(task_id, progress=40, message="正在转录音频（Whisper）...")
            updated_task = await db.get_task(task_id)
            if updated_task:
                tasks[task_id] = updated_task
            await broadcast_task_update(task_id, tasks[task_id])

            raw_script = await transcriber.transcribe(audio_path)

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=source_ref,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            dedup_url=None,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
        )

    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        if task_id in active_tasks:
            del active_tasks[task_id]
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"处理失败: {str(e)}",
        })
        # 更新数据库和内存缓存
        await db.update_task(
            task_id,
            status="error",
            error=str(e),
            message=f"处理失败: {str(e)}",
        )
        updated_task = await db.get_task(task_id)
        if updated_task:
            tasks[task_id] = updated_task
        await broadcast_task_update(task_id, tasks[task_id])


@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in tasks:
        # 尝试从数据库获取（可能刚创建但还未加载到内存）
        task_from_db = await db.get_task(task_id)
        if task_from_db:
            tasks[task_id] = task_from_db
            return task_from_db
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]


@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    """
    SSE实时任务状态流
    """
    if task_id not in tasks:
        # 尝试从数据库获取
        task_from_db = await db.get_task(task_id)
        if task_from_db:
            tasks[task_id] = task_from_db
        else:
            raise HTTPException(status_code=404, detail="任务不存在")
    
    async def event_generator():
        # 创建任务专用的队列
        queue = asyncio.Queue()
        
        # 将队列添加到连接列表
        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)
        
        try:
            # 立即发送当前状态
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"
            
            # 持续监听状态更新
            while True:
                try:
                    # 等待状态更新，超时时间30秒发送心跳
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                    
                    # 如果任务完成或失败，结束流
                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE连接被取消: {task_id}")
        except Exception as e:
            logger.error(f"SSE流异常: {e}")
        finally:
            # 清理连接
            if task_id in sse_connections and queue in sse_connections[task_id]:
                sse_connections[task_id].remove(queue)
                if not sse_connections[task_id]:
                    del sse_connections[task_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    直接从temp目录下载文件（简化方案）
    """
    try:
        # 检查文件扩展名安全性
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="仅支持下载.md文件")
        
        # 检查文件名格式（防止路径遍历攻击）
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="文件名格式无效")
            
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
            
        return FileResponse(
            file_path,
            filename=filename,
            media_type="text/markdown"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    取消并删除任务
    """
    if task_id not in tasks:
        # 检查数据库中是否存在
        task_from_db = await db.get_task(task_id)
        if not task_from_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 如果在数据库中但不在内存中，仍然可以删除
    
    # 如果任务还在运行，先取消它
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"任务 {task_id} 已被取消")
        del active_tasks[task_id]
    
    # 从处理URL列表中移除
    task_url = tasks[task_id].get("url") if task_id in tasks else None
    if task_url:
        processing_urls.discard(task_url)
    
    # 删除数据库记录
    deleted = await db.delete_task(task_id)
    # 从内存缓存中删除
    if task_id in tasks:
        del tasks[task_id]
    
    if deleted:
        return {"message": "任务已取消并删除"}
    else:
        raise HTTPException(status_code=500, detail="删除任务失败")


@app.get("/api/tasks/active")
async def get_active_tasks():
    """
    获取当前活跃任务列表（用于调试）
    """
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys())
    }


# 批量处理端点
@app.post("/api/process-batch")
async def process_batch(
    urls: str = Form(...),  # JSON字符串格式的URL列表
    summary_language: str = Form(default="zh"),
    summary_style: str = Form(default="executive"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
):
    """
    批量处理多个URL
    """
    try:
        # 解析URL列表
        url_list = json.loads(urls)
        if not isinstance(url_list, list):
            raise HTTPException(status_code=400, detail="URLs must be a list")
        
        # 生成批次ID
        batch_id = str(uuid.uuid4())
        
        # 创建批次记录（可以扩展数据库来支持批次，这里先简单处理）
        # 为每个URL创建任务
        task_ids = []
        for url in url_list:
            url = url.strip()
            if not url:
                continue
                
            # 检查是否已经在处理相同的URL
            if url in processing_urls:
                # 查找现有任务
                for tid, task in tasks.items():
                    if task.get("url") == url:
                        task_ids.append(tid)
                        break
                continue
            
            # 生成唯一任务ID
            task_id = str(uuid.uuid4())
            
            # 标记URL为正在处理
            processing_urls.add(url)
            
            # 创建数据库任务记录
            await db.create_task(task_id, url=url)
            # 加载到内存缓存
            task_from_db = await db.get_task(task_id)
            if task_from_db:
                tasks[task_id] = task_from_db
            
            # 创建并跟踪异步任务
            task = asyncio.create_task(process_video_task(task_id, url, summary_language, api_key, model_base_url, model_id))
            active_tasks[task_id] = task
            
            task_ids.append(task_id)
        
        return {
            "batch_id": batch_id,
            "task_ids": task_ids,
            "message": f"批次创建成功，包含 {len(task_ids)} 个任务"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for URLs")
    except Exception as e:
        logger.error(f"创建批处理任务时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批处理失败: {str(e)}")


@app.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """
    获取批次状态（简化实现，实际应存储批次信息）
    """
    # 这里返回所有活跃任务的状态作为批次状态
    # 在实际应用中，应该有一个批次表来跟踪批次信息
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "batch_id": batch_id,
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys()),
        "message": "批次状态查询（简化实现）"
    }


# 历史记录端点
@app.get("/api/history")
async def get_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    summary_language: Optional[str] = Query(None),
    summary_style: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    获取任务历史
    """
    tasks_list = await db.get_history(
        limit=limit,
        offset=offset,
        summary_language=summary_language,
        summary_style=summary_style,
        status=status
    )
    return {"tasks": tasks_list, "count": len(tasks_list)}


@app.get("/api/history/{task_id}")
async def get_history_item(task_id: str):
    """
    获取单个历史任务详情
    """
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.delete("/api/history/{task_id}")
async def delete_history_item(task_id: str):
    """
    删除历史任务
    """
    deleted = await db.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="任务不存在")
    # 同时从内存缓存中删除
    if task_id in tasks:
        del tasks[task_id]
    return {"message": "任务已删除"}


@app.put("/api/history/{task_id}/transcript")
async def update_transcript(
    task_id: str,
    transcript: str = Form(...)
):
    """
    更新任务的转录文本（用于编辑）
    """
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 更新数据库
    updated = await db.update_transcript(task_id, transcript)
    if not updated:
        raise HTTPException(status_code=500, detail="更新转录失败")
    
    # 更新内存缓存
    updated_task = await db.get_task(task_id)
    if updated_task:
        tasks[task_id] = updated_task
    
    return {"message": "转录已更新"}


@app.get("/api/summary-styles")
async def get_summary_styles():
    """
    获取可用的摘要风格
    """
    return {
        "styles": [
            {"id": "executive", "name": "Executive Summary", "description": "Concise executive summary with key takeaways"},
            {"id": "bullet_points", "name": "Bullet Points", "description": "Key points formatted as bullet list"},
            {"id": "eli5", "name": "Explain Like I'm 5", "description": "Simple explanation suitable for beginners"},
            {"id": "tldr", "name": "TL;DR", "description": "Very brief summary of the most important points"},
            {"id": "notes", "name": "Detailed Notes", "description": "Comprehensive notes with timestamps and details"}
        ]
    }


# 后台任务：清理旧的临时文件
async def cleanup_temp_files():
    """清理超过24小时的临时文件"""
    while True:
        await asyncio.sleep(3600)  # 每小时运行一次
        now = datetime.now()
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if (now - file_time).total_seconds() > 24 * 3600:  # 24小时
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted old temp file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

# 启动后台清理任务
@app.on_event("startup")
async def startup_event():
    await db.init()
    # 从数据库加载所有任务到内存缓存
    rows = await db.get_history(limit=10000)
    for row in rows:
        tasks[row['id']] = row
    # 启动清理任务
    asyncio.create_task(cleanup_temp_files())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)