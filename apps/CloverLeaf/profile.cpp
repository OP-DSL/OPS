/* Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/** @brief Process profile info
 * @author Gihan Mudalige
 * @details Processes the profiler output from OPS into the summary format
 * required by Cloverleaf
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

char* strip(char *c) {
    char * e = c + strlen(c) - 1;
    while(*c && isspace(*c)) c++;
    while(e > c && isspace(*e)) *e-- = '\0';
    return c;
}

void process_profile()
{
  char out_file[] = "profiler.out";
  FILE* prof_out;
  char line[100];

  if ((prof_out = fopen(out_file,"w")) == NULL) {
    ops_printf("can't open file %s\n",out_file); exit(-1);
  }

  ops_timing_output(prof_out);
  fclose(prof_out);
  //system("./prof.sh");

  if ((prof_out = fopen(out_file,"r")) == NULL) {
    ops_printf("can't open file %s\n",out_file); exit(-1);
  }


  char name[100];
  int count;
  double time, mpi, bw;
  char std1[15], std2[15];

  double Time_Cell_Advection = 0.0;
  double Time_Momentum_Advection = 0.0;
  double Time_Timestep = 0.0;
  double Time_Ideal_Gas = 0.0;
  double Time_Viscosity = 0.0;
  double Time_PdV = 0.0;
  double Time_Revert = 0.0;
  double Time_Acceleration = 0.0;
  double Time_Fluxes = 0.0;
  double Time_Reset = 0.0;
  double Time_Update_Halo = 0.0;
  double Time_Field_Summary = 0.0;
  double Time_Others = 0.0;

  double Total_time = 0.0;

  double BW_Cell_Advection = 0.0;
  double BW_Momentum_Advection = 0.0;
  double BW_Timestep = 0.0;
  double BW_Ideal_Gas = 0.0;
  double BW_Viscosity = 0.0;
  double BW_PdV = 0.0;
  double BW_Revert = 0.0;
  double BW_Acceleration = 0.0;
  double BW_Fluxes = 0.0;
  double BW_Reset = 0.0;
  double BW_Update_Halo = 0.0;
  double BW_Field_Summary = 0.0;
  double BW_Others = 0.0;

  while (fgets(line, 100, prof_out) != NULL) {
     sscanf(line,"%s %d %lf %s %lf %s %lf\n",name, &count, &time, std1, &mpi, std2, &bw);
     if(strncmp(strip(name),"advec_cell",10)==0) {
       Time_Cell_Advection = Time_Cell_Advection + time;
       BW_Cell_Advection = BW_Cell_Advection + bw;
     }
     if(strncmp(strip(name),"advec_mom",9)==0) {
       Time_Momentum_Advection = Time_Momentum_Advection + time;
       BW_Momentum_Advection = BW_Momentum_Advection + bw;
     }
     if(strncmp(strip(name),"calc_dt",7)==0) {
       Time_Timestep = Time_Timestep + time;
       BW_Timestep = BW_Timestep + bw;
     }
     if(strncmp(strip(name),"ideal_gas",9)==0) {
       Time_Ideal_Gas = Time_Ideal_Gas + time;
       BW_Ideal_Gas = BW_Ideal_Gas + bw;
     }
     if(strncmp(strip(name),"viscosity",9)==0) {
       Time_Viscosity = Time_Viscosity + time;
       BW_Viscosity = BW_Viscosity + bw;
     }
     if(strncmp(strip(name),"PdV",3)==0) {
       Time_PdV = Time_PdV + time;
       BW_PdV = BW_PdV + bw;
     }
     if(strncmp(strip(name),"revert",6)==0) {
       Time_Revert = Time_Revert + time;
       BW_Revert = BW_Revert + bw;
     }
     if(strncmp(strip(name),"accelerate",10)==0) {
       Time_Acceleration = Time_Acceleration + time;
       BW_Acceleration = BW_Acceleration + bw;
     }
     if(strncmp(strip(name),"flux_calc",9)==0) {
       Time_Fluxes = Time_Fluxes + time;
       BW_Fluxes = BW_Fluxes + bw;
     }
     if(strncmp(strip(name),"reset",5)==0) {
       Time_Reset = Time_Reset + time;
       BW_Reset = BW_Reset + bw;
     }
     if(strncmp(strip(name),"update_halo",11)==0) {
       Time_Update_Halo = Time_Update_Halo + time;
       BW_Update_Halo = BW_Update_Halo + bw;
     }
     if(strncmp(strip(name),"field_summary",13)==0) {
       Time_Field_Summary = Time_Field_Summary + time;
       BW_Field_Summary = BW_Field_Summary + bw;
     }
     if(strncmp(strip(name),"initialise",10)==0) {
       Time_Others = Time_Others + time;
       BW_Others = BW_Others + bw;
     }
     if(strncmp(strip(name),"generate",8)==0) {
       Time_Others = Time_Others + time;
       BW_Others = BW_Others + bw;
     }

     Total_time = Total_time +  Time_Cell_Advection + Time_Momentum_Advection +
     Time_Timestep + Time_Ideal_Gas + Time_Viscosity + Time_PdV + Time_Revert +
     Time_Acceleration + Time_Fluxes + Time_Reset + Time_Update_Halo +
     Time_Field_Summary + Time_Others;
  }
  printf("\n\nProfiler Output       :     Time(sec)   Bandwidth(GB/s)\n");
  printf("Timestep              :     %0.4f    %5.2f\n",Time_Timestep, BW_Timestep);
  printf("Ideal Gas             :     %0.4f    %5.2f\n",Time_Ideal_Gas, BW_Ideal_Gas);
  printf("Viscosity             :     %0.4f    %5.2f\n",Time_Viscosity, BW_Viscosity);
  printf("PdV                   :     %0.4f    %5.2f\n",Time_PdV, BW_PdV);
  printf("Revert                :     %0.4f    %5.2f\n",Time_Revert, BW_Revert);
  printf("Acceleration          :     %0.4f    %5.2f\n",Time_Acceleration, BW_Acceleration);
  printf("Fluxes                :     %0.4f    %5.2f\n",Time_Fluxes, BW_Fluxes);
  printf("Cell Advection        :     %0.4f    %5.2f\n",Time_Cell_Advection, BW_Cell_Advection);
  printf("Momentum Advection    :     %0.4f    %5.2f\n",Time_Momentum_Advection, BW_Momentum_Advection);
  printf("Reset                 :     %0.4f    %5.2f\n",Time_Reset, BW_Reset);
  printf("Update_Halo           :     %0.4f    %5.2f\n",Time_Update_Halo, BW_Update_Halo);
  printf("Field_Summary         :     %0.4f    %5.2f\n",Time_Field_Summary, BW_Field_Summary);
  printf("The Rest              :     %0.4f    %5.2f\n",Time_Others, BW_Others);
  printf("\nTotal Time            :     %0.4f    \n",Total_time);
  fclose(prof_out);

}
