.PHONY: test install

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q
