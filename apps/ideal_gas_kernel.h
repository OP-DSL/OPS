#ifndef IDEAL_GAS_KERNEL_H
#define IDEAL_GAS_KERNEL_H

void ideal_gas_kernel( double **density, double **energy,
                     double **pressure, double **soundspeed) {

  double sound_speed_squared, v, pressurebyenergy, pressurebyvolume;

  v = 1.0 / (double)**density;
  **pressure = (1.4 - 1.0) * (double)(**density) * (double)(**energy);
  pressurebyenergy = (1.4 - 1.0) * (double)(**density);
  pressurebyvolume = -1*(double)(**density) * (double)(**pressure);
  sound_speed_squared = v*v*( (double)(**pressure) * pressurebyenergy-pressurebyvolume);
  **soundspeed = sqrt(sound_speed_squared);
}

#endif
