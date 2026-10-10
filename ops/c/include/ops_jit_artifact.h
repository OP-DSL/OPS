/** @file
  * @brief Per-kernel JIT artifact descriptor (generated code <-> chain executors)
  * @details A code generator may embed, for every kernel, an intermediate-representation artifact
  * next to the ahead-of-time implementation. A chain executor (ops_lazy_executor.h), for example an
  * MLIR JIT, reads it through ops_kernel_descriptor::jit_artifact to compile and fuse loops. OPS
  * itself never interprets the contents. The layout is versioned: consumers must check `version`.
  */
#ifndef __OPS_JIT_ARTIFACT_H
#define __OPS_JIT_ARTIFACT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Current value of ops_jit_artifact::version. */
#define OPS_JIT_ARTIFACT_VERSION 1

typedef struct ops_jit_artifact {
  unsigned version;                 /**< OPS_JIT_ARTIFACT_VERSION */
  int status;                       /**< 0: usable; 1: the importer failed (the reason is in manifest_json) */
  const char *kernel_name;          /**< name of the kernel function */
  const unsigned char *bytecode;    /**< serialized IR of the kernel (NULL when status != 0) */
  size_t bytecode_size;             /**< size of bytecode in bytes */
  const char *manifest_json;        /**< NUL-terminated JSON describing parameters, constants and traits */
  const char *digest;               /**< hex SHA-256 over the preprocessed source, importer id and flags */
} ops_jit_artifact;

#ifdef __cplusplus
}
#endif

#endif /* __OPS_JIT_ARTIFACT_H */
