#ifndef OPS_ALGODIFF_HPP_INCLUDED
#define OPS_ALGODIFF_HPP_INCLUDED

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <map>
#include <memory>
#include <vector>

#include "ops_lib_core.h"

class ops_dag_node;

void ops_copy_cp(ops_dat dat, const char *src);
void ops_copy_cp_to_buf(ops_dat dat, const char *buf);
/////////////////////////////////////////////////////////////////////////
// Data structures
/////////////////////////////////////////////////////////////////////////

class ReusableMonotonicAllocator {
  // Just an arbitrary big number
  static const size_t minChunkSize = 1 << 20;
  static const size_t maxChunkSize = 1 << 30;

  std::vector<std::pair<size_t, char *>> chunks;
  size_t head_idx = 0;
  char *head = nullptr;

  std::chrono::duration<double, std::milli> overhead =
      std::chrono::duration<double, std::milli>(0);

 public:
  char *allocate(size_t chunkSize);
  void free(char *);
  void free_all();
  void release();

  ~ReusableMonotonicAllocator();
  ReusableMonotonicAllocator() = default;
  ReusableMonotonicAllocator(const ReusableMonotonicAllocator &) = delete;
  ReusableMonotonicAllocator(ReusableMonotonicAllocator &&) = delete;
  ReusableMonotonicAllocator &operator=(const ReusableMonotonicAllocator &) =
      delete;
  ReusableMonotonicAllocator &operator=(ReusableMonotonicAllocator &&) = delete;
};

class CheckpointAllocator {
  ReusableMonotonicAllocator upstream;
  std::vector<std::pair<size_t, char *>> chunks;
  std::vector<std::pair<size_t, char *>> free_chunks;

  std::chrono::duration<double, std::milli> overhead =
      std::chrono::duration<double, std::milli>(0);

 public:
  char *allocate(size_t chunkSize);
  void free(char *);
  void release();

  ~CheckpointAllocator();
};

class ops_ad_checkpoint {
  std::vector<std::pair<ops_dat, char *>> checkpoints;
  char *cp_start = nullptr;
  CheckpointAllocator *allocator;
  int run_count = 1;

 public:
  ops_ad_checkpoint(const std::vector<ops_dat> &datlist,
                    CheckpointAllocator &allocator, bool do_save);
  // TODO @bgd
  ~ops_ad_checkpoint();  // free checkpoints
  ops_ad_checkpoint(const ops_ad_checkpoint &) = delete;
  ops_ad_checkpoint(ops_ad_checkpoint &&);
  ops_ad_checkpoint &operator=(const ops_ad_checkpoint &) = delete;
  ops_ad_checkpoint &operator=(ops_ad_checkpoint &&) = delete;

  void save();
  void free();
  void load() const;
  bool is_active() const;
  int reruns() const;
  void increment_run_count();
};

class ops_ad_dat_state {
  ReusableMonotonicAllocator *allocator;
  std::vector<std::pair<int, char *>>
      dat_states;   // store {tape_idx, memory_patch} checkpoints for loops,
                    // where tape_idx is the index of the loop_node in the dag
  int cp_idx = -1;  // index of last cp corresponds to tape_idx

 public:
  ops_dat dat = nullptr;
  ops_ad_dat_state(ops_dat dat, ReusableMonotonicAllocator *allocator);
  char *get_cp(int tape_idx);
  char *alloc_cp(int bytes, int tape_idx);
  int num_checkpoints() const;
  void clear();
};

class ops_ad_tape {
  ReusableMonotonicAllocator allocator; /**< memory reasource to allocate patch
                                           like checkpoints by loops*/
  ReusableMonotonicAllocator
      external_function_allocator; /**< memory reasource to allocate permanent
                    checkpoints for external functions */

  std::vector<ops_scalar>
      scalars_used; /**< Pointers to the used scalars. Stores a pointer for each
                       ops_scalar, but the value is nullptr if it is unused. */
  std::vector<ops_reduction>
      reductions_used; /**< Pointers to the used reductions. Stores a pointer
                     for each reduction handler, but the value is nullptr if it
                     is unused. */

  std::vector<char *>
      final_state_dats; /**< storage for the final state of datasets. */
  std::vector<std::vector<char>>
      final_state_scalars; /**< storage for the final state of scalars */
  std::vector<std::vector<char>>
      final_state_reductions; /**< storage for the final state of datasets */

  // Data structures to follow data changes
  std::vector<std::unique_ptr<ops_ad_dat_state>> dats_used;

  struct const_states {
    char *host_ptr;
    std::string type;
    int type_size;
    int dim;
    std::vector<std::vector<char>> states;
  };
  std::map<std::string, const_states> declared_consts;

  void checkpoint_all();
  void load_final_state(OPS_instance *instance);
  void resize_used_args(size_t dat_num, size_t scalar_num,
                        size_t reduction_num);

  int current_tape_idx = 0;
  struct revolve_handler {
    int max_cp_count;
    int length_of_chain;
    int max_recompute_num;
    int next_cp_idx = 0;
    int cp_count = 0;
    std::vector<std::pair<int, ops_ad_checkpoint>>
        checkpoints; /*< stores {tape_idx, checkpoint} pairs for the dag
                        tape_idx refers to the index of the first node after the
                        checkpoint */
    CheckpointAllocator
        allocator; /**< memory reasource to allocate revolve checkpoints */
    void compute_recompute_num();
    static int beta(int s, int t);
    static int m_hat(int m, int s, int t);
    int beta_from(int cp_idx, int m);
  };
  revolve_handler revolve;

  void interpret_section(OPS_instance *instance, int dag_rlast_idx);
  void advance_section(OPS_instance *instance, int dag_begin_idx,
                       int dag_end_idx);

 public:
  enum class checkpointing_strategy { SAVE_ALL, REVOLVE };

  checkpointing_strategy strategy = checkpointing_strategy::SAVE_ALL;

  std::vector<std::unique_ptr<ops_dag_node>>
      dag; /**< Contains all important event in order */

  void add_to_dag(std::unique_ptr<ops_dag_node> &&);

  OPSDagState dagState = OPSDagState::FORWARD_SAVE;
  int interpretedNode = -1;

  void interpret_adjoint(OPS_instance *instance);
  void register_args(OPS_instance *instance, const ops_arg *args, int nargs);
  void zero_adjoints();
  void zero_adjoints_except(const std::vector<ops_dat> &exceptions);
  template <typename... OPSDAT> void zero_adjoints_except(OPSDAT... dats) {
#if __cplusplus >= 201703L
    static_assert(
        (... && std::is_same_v<OPSDAT, ops_dat>),
        "ops_ad_manual_checkpoint_datlist support only ops_dat arguments");
#endif
    zero_adjoints_except({dats...});
  }

  void create_checkpoint(const std::vector<ops_dat> &datlist);
  char *alloc_cp(ops_dat dat, int size);
  char *get_current_checkpoint_for(ops_dat dat);

  void init_revolve(int max_available_cp, int max_chain_length,
                    int max_recompute_num);
  ReusableMonotonicAllocator *get_permanent_allocator();
  size_t register_decl_const(const char *name, int dim, const char *type,
                             int type_size, char *data);
  void reset_const_to(OPS_instance *instance, const std::string &name,
                      size_t pos);
  ops_dag_node *get_active_node();
};

class ops_dat_derivative_wrapper;

class OPS_instance_ad {
  std::vector<std::unique_ptr<ops_ad_tape>> tapes;
  std::vector<std::unique_ptr<ops_dat_derivative_wrapper>> derivative_dats;

 public:
  std::vector<std::unique_ptr<ops_scalar_core>>
      OPS_scalar_list; /**< contains all instances of ops_scalar_core and free
                          the memory at ops_exit */

  OPS_instance_ad() noexcept;
  void allocate_cp(ops_dat dat, int dag_idx);

  void allocate_ad(ops_dat dat);
  void allocate_ad(ops_reduction handle);

  ops_dat get_derivative_as_dat(ops_dat dat);

  ops_ad_tape *create_tape();
  void remove_tape(ops_ad_tape *tape);
  ops_ad_tape *get_current_tape();
  ops_ad_tape *get_default_tape();
};

ops_ad_tape *ops_create_tape(OPS_instance *);
void ops_remove_tape(OPS_instance *, ops_ad_tape *tape);

void ops_ad_set_checkpoint_count(int max_available_cp, int max_chain_length,
                                 int max_recompute_num = -1);

void ops_ad_set_checkpoint_count(OPS_instance *instance, int max_available_cp,
                                 int max_chain_length,
                                 int max_recompute_num = -1);

void ops_ad_manual_checkpoint_datlist_impl(const std::vector<ops_dat> &datlist);

template <typename... OPSDAT>
void ops_ad_manual_checkpoint_datlist(OPSDAT... dats) {
#if __cplusplus >= 201703L
  static_assert(
      (... && std::is_same_v<OPSDAT, ops_dat>),
      "ops_ad_manual_checkpoint_datlist support only ops_dat arguments");
#endif
  ops_ad_manual_checkpoint_datlist_impl({dats...});
}

#endif /* ifndef OPS_ALGODIFF_HPP_INCLUDED */
