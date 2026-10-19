"""
Real-time vision engine.

The camera, per-frame liveness, face recognition, spoof checks and attendance
marking run in a separate process (engine.py) so the Streamlit server cannot
slow them down. The engine streams the annotated video to the browser as MJPEG
and exposes its status over a small localhost HTTP API (engine_client.py).
"""
