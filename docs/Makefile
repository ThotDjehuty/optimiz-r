# Minimal makefile for Sphinx documentation
SHELL := /bin/sh

SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
