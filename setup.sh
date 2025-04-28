#!/bin/bash
python3 -m venv embed
source embed/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python post_install.py
docker compose up -d
