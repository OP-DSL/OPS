#ifndef IDEAL_GAS_KERNEL_H
#define IDEAL_GAS_KERNEL_H

void ideal_gas_kernel( double *density, double *energy,
                     double *pressure, double *soundspeed) {

  double sound_speed_squared, v, pressurebyenergy, pressurebyvolume;

  v = 1.0 / density[OPS_ACC0(0,0)];
  pressure[OPS_ACC2(0,0)] = (1.4 - 1.0) * density[OPS_ACC0(0,0)] * energy[OPS_ACC1(0,0)];
  pressurebyenergy = (1.4 - 1.0) * density[OPS_ACC0(0,0)];
  pressurebyvolume = -1*density[OPS_ACC0(0,0)] * pressure[OPS_ACC2(0,0)];
  sound_speed_squared = v*v*(pressure[OPS_ACC2(0,0)] * pressurebyenergy-pressurebyvolume);
  soundspeed[OPS_ACC3(0,0)] = sqrt(sound_speed_squared);
}

#endif
