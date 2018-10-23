#!/bin/bash
#set -e
#check/install packages
#sudo apt-get install latex-xcolor texlive-science texlive-latex-extra
#sudo apt-get install python-pygments (or easy_install Pygments)
#
#sudo apt-get install doxygen graphviz

pdflatex --shell-escape user.tex
pdflatex --shell-escape user.tex
bibtex user
pdflatex --shell-escape user.tex

#pdflatex --shell-escape mpidev.tex
#pdflatex --shell-escape mpidev.tex
#bibtex mpidev
#pdflatex --shell-escape mpidev.tex

doxygen ops/Doxyfile
pushd ops/latex
make refman.pdf
popd

doxygen ops_translator/Doxyfile
pushd ops_translator/latex
make refman.pdf
popd

rm -f *.out *.aux *.blg *.pyg.* *.log *.backup *.toc *~ *.bbl

