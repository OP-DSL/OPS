CC       = g++
OMPFLAGS = -Ofast -lm

all: tutorial.pdf original step1 step2 step3 step4 step5 step6 step7

tutorial.pdf: tutorial.tex
	pdflatex tutorial.tex

original: Makefile
	make -C $@

step1: Makefile
	make -C $@
step2: Makefile
	make -C $@
step3: Makefile
	make -C $@
step4: Makefile
	make -C $@
step5: Makefile
	make -C $@
step6: Makefile
	make -C $@

step7: Makefile
	make -C $@

clean:
	*.pdf

