#!/usr/bin/env bash

echo @@@@@@@@@@@@@@ ./process.py
python3 ./process.py

echo @@@@@@@@@@@@@@ ./combine.py
python3 ./combine.py

echo @@@@@@@@@@@@@@ ./backfill.py
python3 ./backfill.py

echo @@@@@@@@@@@@@@ ./clean.py
python3 ./clean.py
