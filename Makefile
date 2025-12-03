.PHONY: help build install test lint format clean benchmark docs

help:  ## Show this help message
	@echo "OptimizR - Makefile commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build:  ## Build Rust extension in release mode
	@echo "Building Rust extension..."
	maturin develop --release

build-dev:  ## Build Rust extension in debug mode
	@echo "Building Rust extension (debug)..."
	maturin develop

install:  ## Install package in editable mode with dev dependencies
	@echo "Installing optimizr..."
	pip install -e ".[dev]"

test:  ## Run Python tests
	@echo "Running tests..."
	pytest tests/ -v

test-rust:  ## Run Rust tests
	@echo "Running Rust tests..."
	cargo test

test-all:  ## Run all tests (Python + Rust)
	@make test-rust
	@make test

test-cov:  ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=optimizr --cov-report=html --cov-report=term

lint:  ## Run all linters
	@echo "Linting Python..."
	ruff check python/
	@echo "Linting Rust..."
	cargo clippy -- -D warnings

lint-fix:  ## Fix auto-fixable lint issues
	@echo "Fixing Python lint issues..."
	ruff check --fix python/
	@echo "Fixing Rust lint issues..."
	cargo clippy --fix --allow-dirty

format:  ## Format code
	@echo "Formatting Python..."
	black python/
	@echo "Formatting Rust..."
	cargo fmt

format-check:  ## Check code formatting without modifying
	@echo "Checking Python formatting..."
	black --check python/
	@echo "Checking Rust formatting..."
	cargo fmt --check

typecheck:  ## Run type checking
	@echo "Type checking..."
	mypy python/optimizr/ --ignore-missing-imports

benchmark:  ## Run Rust benchmarks
	@echo "Running benchmarks..."
	cargo bench

clean:  ## Clean build artifacts
	@echo "Cleaning..."
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf python/optimizr.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dylib" -delete

wheel:  ## Build wheel distribution
	@echo "Building wheel..."
	maturin build --release --out dist/

publish-test:  ## Publish to TestPyPI
	@echo "Publishing to TestPyPI..."
	maturin publish --repository testpypi

publish:  ## Publish to PyPI
	@echo "Publishing to PyPI..."
	maturin publish

docs:  ## Build documentation
	@echo "Building documentation..."
	@echo "Documentation build not yet implemented"

example:  ## Run example script
	@echo "Running HMM example..."
	python examples/hmm_regime_detection.py

check:  ## Run all checks (format, lint, typecheck, test)
	@make format-check
	@make lint
	@make typecheck
	@make test-all

dev:  ## Setup development environment
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	pip install maturin
	@make build-dev
	@echo "✓ Development environment ready!"

ci:  ## Run CI checks locally
	@echo "Running CI checks..."
	@make format-check
	@make lint
	@make typecheck
	@make test-all
	@echo "✓ All CI checks passed!"
