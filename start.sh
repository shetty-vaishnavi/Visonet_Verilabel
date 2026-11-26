#!/bin/sh
cd verilabel_ai_project/verilabel_ai/verilabel_app
exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
