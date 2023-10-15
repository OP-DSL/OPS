//INC,READ,READ,READ
void eqA(ACC<double> &a, const ACC<double> &b, const ACC<double> &c, const ACC<double> &d) {
}

//RW, READ
void eqB(ACC<double> &a, const ACC<double> &b) {
}

//WRITE,READ
void eqB1(ACC<double> &a, const ACC<double> &b) {
}

//RW,INC,READ
void eqC(ACC<double> &a, ACC<double> &b, const ACC<double> &c) {
}

//READ,WRITE DFBYDX
void eqD(const ACC<double> &a, ACC<double> &b) {
}

//INC,INC,READ,READ,READ,READ
void eqE(ACC<double> &a, ACC<double> &b, const ACC<double> &c, const ACC<double> &d, const ACC<double> &e, const ACC<double> &f) {
}

//RW,READ,READ,READ,READ,READ
void eqF(ACC<double> &a, const ACC<double> &b, const ACC<double> &c, const ACC<double> &d, const ACC<double> &e, const ACC<double> &f) {
}

//INC,READ,READ
void eqG(ACC<double> &a, const ACC<double> &b, const ACC<double> &c) {
}

//WRITE,READ,READ
void eqG1(ACC<double> &a, const ACC<double> &b, const ACC<double> &c) {
}

//WRITE
void eqH(ACC<double> &a) {
}

//INC,READ,READ,READ,READ,READ,READ
void eqI(ACC<double> &a, const ACC<double> &b, const ACC<double> &c, const ACC<double> &d, const ACC<double> &e, const ACC<double> &f, const ACC<double> &g) {
}

//INC,READ,READ,READ,READ,READ
void eqJ(ACC<double> &a, const ACC<double> &b, const ACC<double> &c, const ACC<double> &d, const ACC<double> &e, const ACC<double> &f) {
}

//WRITE,READ,READ
void eqK(ACC<double> &a, const ACC<double> &b, const ACC<double> &c) {
}

//INC,READ
void eqL(ACC<double> &a, const ACC<double> &b) {
}
