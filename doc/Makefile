#!/bin/bash
#set -e
#check/install packages
#sudo apt-get install latex-xcolor texlive-science texlive-latex-extra
#sudo apt-get install python-pygments (or easy_install Pygments)
#
#sudo apt-get install doxygen graphviz

.PHONY : all user doxygen clean distclean

all : doxygen
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
