#!/usr/bin/env bash
set -euo pipefail
python3 -c "from calc import clamped_add; assert clamped_add(2, 3, 10) == 5; assert clamped_add(7, 8, 10) == 10; assert clamped_add(-2, 1, 5) == -1"
