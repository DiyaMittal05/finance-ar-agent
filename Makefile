PY := .venv/bin/python

.PHONY: setup data cases test eval eval-mock report serve deploy

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	$(PY) -m app.data.generate

cases:
	$(PY) -m evals.build_cases

test:
	$(PY) -m pytest -q

eval-mock:
	$(PY) -m evals.run --provider mock --judge-provider none

# e.g. make eval PROVIDER=anthropic JUDGE=gemini
eval:
	$(PY) -m evals.run --provider $(PROVIDER) --judge-provider $(or $(JUDGE),none)

report:
	$(PY) -m evals.report

serve:
	$(PY) -m uvicorn app.main:app --reload --port 8000

# Needs gcloud, a project (gcloud config set project ...) and secrets created; see README.
deploy:
	gcloud run deploy ar-assistant --source . --region $(or $(REGION),us-central1) \
	  --allow-unauthenticated --memory 1Gi --max-instances 2 \
	  --set-env-vars LLM_PROVIDER=$(or $(PROVIDER),gemini),PROMPT_VERSION=v3 \
	  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
