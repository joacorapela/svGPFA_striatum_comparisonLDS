all: 52046850_model_plots.pdf 23912280_model_plots.pdf 71005668_model_plots.pdf 37180679_model_plots.pdf 54368807_model_plots.pdf 30222892_model_plots.pdf 14347841_model_plots.pdf 74407950_model_plots.pdf 89711911_model_plots.pdf

%.pdf: %.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<
