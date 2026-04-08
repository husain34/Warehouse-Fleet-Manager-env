---
title: Warehouse Fleet Management
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Warehouse Fleet Management Environment

This is an OpenEnv-compatible warehouse robot simulation environment.

## Features
- Multi-robot coordination
- Dynamic obstacles and spills
- Charging stations and battery logic
- Task-based reward system

## API Endpoints
- `/reset` → Initialize environment with task
- `/step` → Apply actions
- `/state` → Get current state

## Tasks
Located in `/tasks`:
- easy
- medium
- hard

## Built With
- FastAPI
- Pydantic
- Docker