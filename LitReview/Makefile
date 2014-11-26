BASE=lit-review

SRC=$(BASE).tex
PDF=$(BASE).pdf
DEP=preamble.tex
LATEXMK=latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S"
LATEXMKCONT=latexmk -pvc -pdf -pdflatex="pdflatex --shell-escape --interaction=nonstopmode %O %S"
TMP=$(BASE).pdfsync *~ *.tmp *.bbl *.blg *.aux *.end *.fls *.log *.out *.fdb_latexmk

all: $(PDF)

$(PDF): $(SRC) $(DEP)
	$(LATEXMK) $(SRC)

cont:
	$(LATEXMKCONT) $(SRC)
force:
	latexmk -C $(BASE)
	rm -rf $(TMP)
	$(LATEXMK) $(SRC)

clean:
	latexmk -C $(BASE)
	rm -rf $(TMP)

.PHONY: cont force clean all
