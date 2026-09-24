.PHONY : all doxygen clean distclean

all : doxygen
doxygen :
	doxygen ops/Doxyfile
	cd ops/latex; make refman.pdf
	doxygen ops_translator/Doxyfile
	cd ops_translator/latex; make refman.pdf

clean :
	-rm -f *.out *.aux *.blg *.pyg.* *.log *.backup *.toc *~ *.bbl

distclean : clean
	-rm -rf ops/html ops/latex ops_translator/html ops_translator/latex
