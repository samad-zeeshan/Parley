# Needs make, docker, kind and kubectl (Linux, macOS, or WSL on Windows). CI runs the same targets.
CLUSTER ?= parley
IMAGE ?= parley:dev
BASE ?= http://127.0.0.1:8000

.PHONY: test eval image k8s-up k8s-gates k8s-down

test:
	uv run pytest -q

eval:
	uv run --extra speech python -m eval.run_eval

image:
	docker build -t $(IMAGE) .

# Gate 1 (image builds) and gate 2 (pods ready).
k8s-up: image
	kind get clusters | grep -qx $(CLUSTER) || kind create cluster --config deploy/kind/cluster.yaml
	kind load docker-image $(IMAGE) --name $(CLUSTER)
	kubectl apply -k deploy/k8s
	kubectl -n parley rollout status deploy/parley --timeout=180s
	kubectl -n parley rollout status deploy/prometheus --timeout=180s
	kubectl -n parley rollout status deploy/grafana --timeout=180s
	kubectl -n parley wait --for=condition=Ready pod --all --timeout=180s

# Gate 3 (one scripted booking completes) and gate 4 (deployed config matches the manifest).
k8s-gates:
	python3 deploy/ci_booking.py $(BASE)
	python3 deploy/check_config.py $(BASE)

k8s-down:
	kind delete cluster --name $(CLUSTER)
