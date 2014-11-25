BASE=continuity_derivation
SRC=$(BASE).tex
PDF=$(BASE).pdf
DEP=preamble.tex
LATEXMK=latexmk -pdf -pdflatex='pdflatex --shell-escape %O %S'
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
