.PHONY: test test-pybullet reproduce clean

test:
	python -m pytest

test-pybullet:
	RUN_PYBULLET=1 python -m pytest tests/test_scene_smoke.py

reproduce:
	bash scripts/reproduce_pickplace.sh

clean:
	rm -rf runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
