# multimodel_api.py  — ONNX + LAZY LOADING + BINARY WS + AUTO GPU
import base64
import cv2
import numpy as np
import json
import importlib
import traceback
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 🔥 Preferred providers: compute on demand (avoid heavy ort import at module load)
# ---------------------------
_PREFERRED_PROVIDERS_CACHE: Optional[List[str]] = None

def get_preferred_provider_order() -> List[str]:
    global _PREFERRED_PROVIDERS_CACHE
    if _PREFERRED_PROVIDERS_CACHE is not None:
        return _PREFERRED_PROVIDERS_CACHE

    # import onnxruntime lazily (only when needed)
    import onnxruntime as ort_local  # local alias to avoid global heavy import
    available = ort_local.get_available_providers()
    print("🔍 Available ONNX providers:", available)

    order = []
    if "CUDAExecutionProvider" in available:
        order.append("CUDAExecutionProvider")
    if "ROCMExecutionProvider" in available:
        order.append("ROCMExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        order.append("CoreMLExecutionProvider")
    if "TensorrtExecutionProvider" in available:
        order.append("TensorrtExecutionProvider")
    if "CPUExecutionProvider" in available:
        order.append("CPUExecutionProvider")
    else:
        order.append("CPUExecutionProvider")

    _PREFERRED_PROVIDERS_CACHE = order
    return order

# don't compute providers at import time; compute lazily later
# PREFERRED_PROVIDERS = get_preferred_provider_order()
PREFERRED_PROVIDERS = None

# ---------------------------
# Model registry + lazy flags
# ---------------------------
EXERCISE_MODELS = {
    "squat": {
        "model_path": "models/squat/squat_model.onnx",
        "scaler_path": "models/squat/squat_pose_scaler.pkl",
        "inference_module": "models.squat.inference",
        "threshold": 0.032,
        "window_size": 10,
        "num_features": 22,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "S1", 'viable_rep': True, 'Bottom_ROM_error': False },
        "loaded": False
    },
    "push_ups": {
        "model_path": "models/pushup/pushup_model.onnx",
        "scaler_path": "models/pushup/pushup_pose_scaler.pkl",
        "inference_module": "models.pushup.inference",
        "threshold": 0.11,
        "window_size": 10,
        "num_features": 40,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "P1", 'viable_rep': True, 'Top_ROM_error': False, 'Bottom_ROM_error': False },
        "loaded": False
    },
    "lateral_raises": {
        "model_path": "models/lateral_raises/lateral_raises_model.onnx",
        "scaler_path": "models/lateral_raises/lateral_raises_pose_scaler.pkl",
        "inference_module": "models.lateral_raises.inference",
        "threshold": 0.84,
        "window_size": 10,
        "num_features": 54,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "LR1", 'viable_rep': True, 'Top_ROM_error': False, 'Bottom_ROM_error': False },
        "loaded": False
    },
    "biceps_curl": {
        "model_path": "models/biceps_curl/biceps_curl_model.onnx",
        "scaler_path": "models/biceps_curl/biceps_pose_scaler.pkl",
        "inference_module": "models.biceps_curl.inference",
        "threshold": 0.017,
        "window_size": 10,
        "num_features": 26,
        "seq_len": 10,
        "rep_state": { 'rep_counter': 0, 'prev_angle': None, 'prev_phase': None, 'phase': "B1", 'viable_rep': True, 'Top_ROM_error': False, 'Bottom_ROM_error': False },
        "loaded": False
    }
}

# ---------------------------
# Helper: create ONNX session with provider fallback (lazy imports)
# ---------------------------
def create_session_with_fallback(model_path: str, sess_options=None):
    # Lazy import ONNX runtime here
    import onnxruntime as ort_local

    last_exception = None
    sess_opts = sess_options if sess_options is not None else ort_local.SessionOptions()

    # Ensure provider order computed lazily
    global PREFERRED_PROVIDERS
    if PREFERRED_PROVIDERS is None:
        PREFERRED_PROVIDERS = get_preferred_provider_order()

    for provider in PREFERRED_PROVIDERS:
        try:
            print(f"⏳ Attempting InferenceSession with provider: {provider}")
            providers_try = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
            session = ort_local.InferenceSession(model_path, sess_options=sess_opts, providers=providers_try)
            used_providers = session.get_providers()
            print(f"✅ Loaded session for {model_path} with providers: {used_providers}")
            return session, provider
        except Exception as e:
            last_exception = e
            print(f"⚠️ Failed to create session with provider {provider}: {e}")
            continue

    print("❌ All provider attempts failed. Raising last exception.")
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Failed to create ONNX Runtime session for unknown reasons.")

# ---------------------------
# Lazy loader (non-blocking caller recommended)
# ---------------------------
def load_model_if_needed(model_name: str):
    # This function does heavy work — keep synchronous; callers should run it in a thread
    cfg = EXERCISE_MODELS[model_name]
    if cfg.get("loaded", False):
        return cfg

    print(f"⏳ Loading model lazily: {model_name} ...")
    try:
        # Import inference module lazily (avoid heavy imports at global level)
        inference = importlib.import_module(cfg["inference_module"])
        cfg["analyze_frame"] = inference.analyze_frame

        # Lazy import joblib only here
        import joblib as joblib_local

        # ONNX session options: keep threads small to reduce overhead on small CPU VMs
        import onnxruntime as ort_local
        sess_opts = ort_local.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        try:
            sess_opts.graph_optimization_level = ort_local.GraphOptimizationLevel.ORT_ENABLE_BASIC
        except Exception:
            pass

        # Create session with provider fallback
        try:
            session, used_provider = create_session_with_fallback(cfg["model_path"], sess_options=sess_opts)
            cfg["session"] = session
            cfg["used_provider"] = used_provider
        except Exception as e:
            print(f"❌ Failed to create ONNX session for model '{model_name}': {e}")
            print(traceback.format_exc())
            cfg["loaded"] = False
            return cfg

        # Load scaler (joblib)
        try:
            scaler = joblib_local.load(cfg["scaler_path"])
            cfg["scaler"] = scaler
        except Exception as e:
            print(f"⚠️ Failed to load scaler for '{model_name}': {e}")
            cfg["scaler"] = None

        cfg["loaded"] = True
        print(f"✅ Model '{model_name}' loaded (provider: {cfg.get('used_provider')})!")
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}': {e}")
        print(traceback.format_exc())
        cfg["loaded"] = False
    return cfg

# ---------------------------
# Root endpoint
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Backend running with ONNX + LAZY LOADING + BINARY WS 🚀"}

# Optional warmup HTTP endpoint — call this to pre-load a model before connecting WS
@app.post("/warmup/{model_name}")
async def warmup_model(model_name: str):
    if model_name not in EXERCISE_MODELS:
        return {"error": "unknown model"}
    # Load in background but wait for completion — useful in CI or manual warmup
    cfg = await asyncio.to_thread(load_model_if_needed, model_name)
    return {"model": model_name, "loaded": cfg.get("loaded", False)}

# ---------------------------
# WebSocket endpoint (binary frames + json control)
# ---------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket connected")
    websocket_active = True

    try:
        init_msg = await websocket.receive_text()
        try:
            cfg_json = json.loads(init_msg)
        except Exception as e:
            await websocket.send_json({"error": "Expected initial JSON text (e.g. {\"model\":\"push_ups\"})"})
            return

        model_name = cfg_json.get("model")
        if model_name not in EXERCISE_MODELS:
            await websocket.send_json({"error": f"Model '{model_name}' not found"})
            return

        # Lazy load model BUT run in a background thread so event loop isn't blocked
        await websocket.send_json({"status": "loading_model"})
        model_cfg = await asyncio.to_thread(load_model_if_needed, model_name)

        if not model_cfg.get("loaded", False):
            await websocket.send_json({"error": f"Failed to load model '{model_name}' on server"})
            return

        session = model_cfg.get("session")
        scaler = model_cfg.get("scaler")
        threshold = model_cfg.get("threshold")
        analyze_frame = model_cfg.get("analyze_frame")
        buffer = deque(maxlen=model_cfg["window_size"])
        rep_state = model_cfg["rep_state"].copy()

        print(f"🎯 USING MODEL: {model_name} (provider: {model_cfg.get('used_provider')})")
        await websocket.send_json({"status": "ready", "provider": model_cfg.get("used_provider")})

        # Main loop: accept binary frames or occasional text control messages
        while True:
            msg = await websocket.receive()  # returns dict with 'type' and either 'text' or 'bytes'
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if "bytes" in msg and msg["bytes"] is not None:
                frame_bytes = msg["bytes"]
                try:
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        await websocket.send_json({"error": "Invalid frame bytes"})
                        continue

                    # Run model inference inside a thread to avoid blocking uvloop if inference is slow
                    try:
                        result = await asyncio.to_thread(
                            analyze_frame,
                            frame,
                            session,
                            scaler,
                            threshold,
                            buffer,
                            model_cfg["window_size"],
                            model_cfg["num_features"],
                            rep_state
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        await websocket.send_json({"error": f"Inference error: {str(e)}", "trace": tb})
                        continue

                    await websocket.send_json(result)

                except Exception as e:
                    tb = traceback.format_exc()
                    await websocket.send_json({"error": f"Processing error: {str(e)}", "trace": tb})

            elif "text" in msg and msg["text"] is not None:
                text = msg["text"]
                try:
                    payload = json.loads(text)
                except:
                    payload = {"command": text}

                cmd = payload.get("command")
                if cmd == "reset":
                    buffer.clear()
                    rep_state = model_cfg["rep_state"].copy()
                    await websocket.send_json({"status": "reset_ok", "rep_state": rep_state})
                elif cmd == "close":
                    await websocket.send_json({"status": "closing"})
                    break
                else:
                    await websocket.send_json({"info": "text_received", "payload": payload})

    except WebSocketDisconnect:
        print("⚠️ WebSocket disconnected")
        websocket_active = False

    except Exception as e:
        print("❌ Unhandled exception in websocket loop:", e)
        print(traceback.format_exc())
        try:
            await websocket.send_json({"error": "server_error", "detail": str(e)})
        except:
            pass

    finally:
        if websocket_active:
            try:
                await websocket.close()
            except:
                pass
        print("🔴 WebSocket closed")
