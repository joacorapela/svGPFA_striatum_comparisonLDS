all: gofMeasuresAcrossModels.pdf 79151056_full_hessian.pdf

79151056_full_hessian.pdf: 79151056_full_hessian.tex
	pdflatex $<

%.pdf: %.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<
