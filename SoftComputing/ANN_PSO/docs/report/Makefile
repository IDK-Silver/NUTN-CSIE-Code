# LaTeX 報告編譯 Makefile
# 使用 XeLaTeX 支援中文

MAIN = main
LATEX = xelatex
LATEXFLAGS = -shell-escape -interaction=nonstopmode

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex report.cls images/*
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz *.svg-inkscape

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf
