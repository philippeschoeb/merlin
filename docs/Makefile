# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate API documentation
apidoc:
	@echo "Generating API documentation..."
	sphinx-apidoc -o $(SOURCEDIR)/reference/api/ ../merlin/ --separate --module-first --force
	@echo "API documentation generated in $(SOURCEDIR)/api/"

multiversion:
	python -m sphinx_multiversion -f multiversion_config source build/html

clean:
	rm -rf build

.PHONY: clean
