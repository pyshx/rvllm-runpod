.PHONY: test build deploy status logs stop start chat help

RUNPOD_API_KEY ?= $(shell echo $$RUNPOD_API_KEY)
ENDPOINT_ID    ?= $(shell cat .endpoint 2>/dev/null)
MODEL_ID       ?= Qwen/Qwen2.5-7B-Instruct
GPU            ?= AMPERE_80

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run unit tests
	@./scripts/smoke_test.sh

build: ## Build Docker image locally (no push)
	@./scripts/build.sh --tag rvllm-runpod:local

deploy: ## Deploy to RunPod (creates template + endpoint)
	@./scripts/deploy.sh --model "$(MODEL_ID)" --gpu "$(GPU)"

status: ## Check endpoint health
	@test -n "$(ENDPOINT_ID)" || (echo "No endpoint. Run 'make deploy' first." && exit 1)
	@curl -s "https://api.runpod.ai/v2/$(ENDPOINT_ID)/health" \
		-H "authorization: $(RUNPOD_API_KEY)" | python3 -m json.tool

chat: ## Send a chat message (usage: make chat MSG="hello")
	@test -n "$(ENDPOINT_ID)" || (echo "No endpoint. Run 'make deploy' first." && exit 1)
	@curl -s -X POST "https://api.runpod.ai/v2/$(ENDPOINT_ID)/runsync" \
		-H "authorization: $(RUNPOD_API_KEY)" \
		-H "content-type: application/json" \
		-d '{"input":{"messages":[{"role":"user","content":"$(MSG)"}],"temperature":0,"max_tokens":128}}' \
		| python3 -c "import sys,json; r=json.loads(sys.stdin.read()); print(r.get('output',[{}])[0].get('choices',[{}])[0].get('message',{}).get('content','(no output)'))" 2>/dev/null \
		|| echo "(worker not ready — try again in a minute)"

stop: ## Scale endpoint to zero workers
	@test -n "$(ENDPOINT_ID)" || (echo "No endpoint." && exit 1)
	@./scripts/deploy.sh --endpoint "$(ENDPOINT_ID)" --workers 0
	@echo "Endpoint stopped."

start: ## Scale endpoint to 1 worker
	@test -n "$(ENDPOINT_ID)" || (echo "No endpoint." && exit 1)
	@./scripts/deploy.sh --endpoint "$(ENDPOINT_ID)" --workers 1
	@echo "Endpoint started."

logs: ## Open RunPod logs in browser
	@test -n "$(ENDPOINT_ID)" || (echo "No endpoint." && exit 1)
	@open "https://www.runpod.io/console/serverless/$(ENDPOINT_ID)/logs" 2>/dev/null \
		|| echo "https://www.runpod.io/console/serverless/$(ENDPOINT_ID)/logs"
