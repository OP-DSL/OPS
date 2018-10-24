#!/bin/bash
#set -e
#check/install packages
#sudo apt-get install latex-xcolor texlive-science texlive-latex-extra
#sudo apt-get install python-pygments (or easy_install Pygments)
#
#sudo apt-get install doxygen graphviz

.PHONY : all user doxygen clean distclean

all : user doxygen

user :
	pdflatex --shell-escape user.tex
	pdflatex --shell-escape user.tex
	-bibtex user
	pdflatex --shell-escape user.tex
	latex_count=8 ; \
	while egrep -s 'Rerun (LaTeX|to get cross-references right)' user.log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      pdflatex --shell-escape user.tex ;\
	      latex_count=`expr $$latex_count - 1` ;\
	    done

doxygen :
	doxygen ops/Doxyfile
	cd ops/latex; make refman.pdf
	doxygen ops_translator/Doxyfile
	cd ops_translator/latex; make refman.pdf

clean : 
	-rm -f *.out *.aux *.blg *.pyg.* *.log *.backup *.toc *~ *.bbl
	-rm -rf _minted-user

distclean : clean
	-rm -rf user.pdf ops/html ops/latex ops_translator/html ops_translator/latex
