//Kernels for lattice-boltzmann demo app
//

void init_solid(ACC<int> &SOLID, const int *idx) {
    int i = idx[0], j = idx[1];

    if( (i>=2 && i<=65 && j>=4 && j<=29) || (i>=45 && i<=123 && j>=41 && j<=65) || (i>=30 && i<=101 && j>=91 && j<=115))
        SOLID(0,0) = 0;
    else
        SOLID(0,0) = 1;

}

void init_n(ACC<double> &N, const double* W) {

    for (int f = 0; f < 9; f++) {
        N(f,0,0) = rho0 * W[f];
    }
}

void timeloop_eqA(ACC<double> &workArray, const ACC<double> &N) {

    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = N(f,0,0);
    }
}

void timeloop_eqC(ACC<double> &N_SOLID, const ACC<double> &N, const int* opposite) {

    for (int f = 0; f < 9; f++) {
        N_SOLID(opposite[f], 0,0) = N(f,0,0);
    }
}

void timeloop_eqD(ACC<double> &rho, const ACC<double> &N) {

    rho(0,0) = 0.0;
    for (int f = 0; f < 9; f++) {
        rho(0,0) += N(f,0,0);
    }
}

void timeloop_eqE(ACC<double> &ux, const ACC<double> &N, const ACC<double> &rho, const int* cx) {

    ux(0,0) = 0.0;
    for (int f = 0; f < 9; f++) {
        ux(0,0) += N(f,0,0) * cx[f];
    }
    ux(0,0) = ux(0,0) / rho(0,0) + deltaUX;    
}

void timeloop_eqF(ACC<double> &uy, const ACC<double> &N, const ACC<double> &rho, const int* cy) {

    uy(0,0) = 0.0;
    for (int f = 0; f < 9; f++) {
        uy(0,0) += N(f,0,0) * cy[f];
    }
    uy(0,0) = uy(0,0) / rho(0,0);
}

void timeloop_eqG(ACC<double> &workArray, const ACC<double> &ux, const ACC<double> &uy, const int* cx, const int* cy) {

    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = ux(0,0)*cx[f] + uy(0,0)*cy[f];
    }
}

void timeloop_eqH(ACC<double> &workArray) {

    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = (3.0+4.5*workArray(f,0,0))*workArray(f,0,0);
    }
}

void timeloop_eqI(ACC<double> &workArray, const ACC<double> &ux, const ACC<double> &uy) {

    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = workArray(f,0,0) - 1.5 * (ux(0,0)*ux(0,0) + uy(0,0)*uy(0,0));
    }
}

void timeloop_eqJ(ACC<double> &workArray, const ACC<double> &rho, const double* W) {

    for (int f = 0; f < 9; f++) {
        workArray(f,0,0) = (1.0 + workArray(f,0,0)) * W[f] * rho(0,0);
    }
}

void timeloop_eqK(ACC<double> &N, const ACC<double> &workArray) {

    for (int f = 0; f < 9; f++) {
        N(f,0,0) += (workArray(f,0,0) - N(f,0,0)) * OMEGA;
    }
}

void timeloop_eqL(ACC<double> &N, const ACC<double> &N_SOLID, const ACC<int> &SOLID) {

    if(SOLID(0,0) == 1) {
        for (int f = 0; f < 9; f++) {
            N(f,0,0) = N_SOLID(f,0,0);
        }
    }
}
