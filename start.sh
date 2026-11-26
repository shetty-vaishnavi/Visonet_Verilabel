#!/bin/bash
set -e
cd verilabel_ai_project/verilabel_ai/verilabel_app
exec uvicorn server:app --host 0.0.0.0 --port 
