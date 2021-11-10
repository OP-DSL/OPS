#ifndef IDEAL_GAS_KERNEL_H
#define IDEAL_GAS_KERNEL_H

void ideal_gas_kernel( const ACC<double> &density, const ACC<double> &energy,
                     ACC<double> &pressure, ACC<double> &soundspeed) {

  double sound_speed_squared, v, pressurebyenergy, pressurebyvolume;

  v = 1.0 / density(0,0);
  pressure(0,0) = (1.4 - 1.0) * density(0,0) * energy(0,0);
  pressurebyenergy = (1.4 - 1.0) * density(0,0);
  pressurebyvolume = -1*density(0,0) * pressure(0,0);
  sound_speed_squared = v*v*(pressure(0,0) * pressurebyenergy-pressurebyvolume);
  soundspeed(0,0) = sqrt(sound_speed_squared);
}
#endif
