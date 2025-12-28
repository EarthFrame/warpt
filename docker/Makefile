# Makefile for building warpt benchmark Docker images in parallel.
# Use 'make -j N' to build N images simultaneously.

# Configuration
REGISTRY ?= warpt
TAG      ?= latest

# List of benchmark images
BENCHMARKS = hpl-arm64

.PHONY: all help $(BENCHMARKS) clean

# Default target
all: $(BENCHMARKS)

help:
	@echo "warpt Docker Build System"
	@echo "-------------------------"
	@echo "Usage: make [target] [-j N]"
	@echo ""
	@echo "Targets:"
	@echo "  all          Build all benchmark images"
	@echo "  hpl-arm64    Build HPL image for Apple Silicon (ARM64)"
	@echo ""
	@echo "Options:"
	@echo "  REGISTRY     Image registry/prefix (default: warpt)"
	@echo "  TAG          Image tag (default: latest)"

# HPL for Apple Silicon (ARM64)
hpl-arm64:
	docker build -t $(REGISTRY)-hpl:arm64 -f benchmarks/hpl/Dockerfile.arm64 benchmarks/hpl/

# Placeholder for future benchmarks
# some-benchmark:
# 	docker build -t $(REGISTRY)-some-benchmark:$(TAG) -f benchmarks/some/Dockerfile benchmarks/some/

clean:
	@echo "Cleaning up dangling images..."
	docker image prune -f

