#include <iostream>
#define OPS_2D
#define OPS_CPP_API
#include "ops_seq_v2.h"

int main(int argc, const char** argv)
{
  // Creating instance
  std::stringstream ss;
  OPS_instance* instance = new OPS_instance(argc, argv, 1, ss);

  // Creating blocks
  //------------------
  // The error below, at line 26 where an ops_tridsolver_param object is created, happens only when more than one
  // ops_block is created. If you comment out line 18 (creating opsBlock2), there is no error.
  //------------------
  ops_block opsBlock1 = instance->decl_block(2, "Zone 00001");
  ops_block opsBlock2 = instance->decl_block(2, "Zone 00002");

  // Mesh decomposition
  instance->partition("");

  //------------------
  // THE PROBLEM IS HERE! There is no error unless creating an ops_tridsolver_params object with more than 1 existing ops_block 
  //------------------
  ops_tridsolver_params* testTrid1 = new ops_tridsolver_params(opsBlock1);
  ops_tridsolver_params* testTrid2 = new ops_tridsolver_params(opsBlock2);  

  delete testTrid1;
  delete testTrid2;
  delete instance;

  //------------------
  // C version, uncomment this, and comment out line 3 & everything above in main() and it still doesn't work!
  //------------------
  /*
  ops_init(argc, argv, 1);
  ops_block opsBlock1 = ops_decl_block(2, "Zone 00001");
  ops_block opsBlock2 = ops_decl_block(2, "Zone 00002");
  ops_partition("");
  ops_tridsolver_params* testTrid1 = new ops_tridsolver_params(opsBlock1);
  delete testTrid1;
  ops_exit();
  */
  return 0;
}
