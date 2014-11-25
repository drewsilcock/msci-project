BASE=gise_derivation
SRC=$(BASE).tex
PDF=$(BASE).pdf
DEP=preamble.tex
LATEXMK=latexmk -pvc -pdf -pdflatex='pdflatex --shell-escape --interaction=nonstopmode %O %S'
TMP=$(BASE).pdfsync *~ *.tmp *.bbl *.blg *.aux *.end *.fls *.log *.out *.fdb_latexmk

all: $(PDF)

$(PDF): $(SRC) $(DEP)
	$(LATEXMK) $(SRC)

force:
	latexmk -C $(BASE)
	rm -rf $(TMP)
	$(LATEXMK) $(SRC)

clean:
	latexmk -C $(BASE)
	rm -rf $(TMP)

.PHONY: force clean all
