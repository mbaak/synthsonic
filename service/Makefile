PYTHONPATH := $(PWD)

test:
	flake8 --max-line-length=120 api; mypy api
	flake8 --max-line-length=120 ui; mypy ui

setup-env:
	python3 -m venv .venv || true

build-synthsonic:
	cd .. ; python3 setup.py sdist bdist_wheel

setup-api:	setup-env 	build-synthsonic
	cd ..;cp ./dist/*whl /tmp/
	. .venv/bin/activate; pip3 install -r api/requirements.txt; pip3 install uvicorn

setup-ui: 	setup-env
	. .venv/bin/activate; pip3 install -r ui/requirements.txt; pip3 install streamlit

run-api-local:	setup-api
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

run-ui-local:	setup-ui
	streamlit run ui/ui.py

run-all-docker:	build-synthsonic
	cd ..;cp ./dist/*whl ./service/api/
	docker-compose build --force; docker-compose up -d