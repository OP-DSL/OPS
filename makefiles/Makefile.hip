HIPCC?=hipcc
HIPFLAGS?=-fPIC
HIP_INC ?= -I$(HIP_INSTALL_PATH)/include
HIP_LIB ?= $(HIP_INSTALL_PATH)/lib
HIP_LINK ?= -L$(HIP_INSTALL_PATH)/lib -L$(HIP_INSTALL_PATH)/hsa/lib -L$(HIP_INSTALL_PATH)/hcc/lib -lhc_am -lmcwamp -lhip_hcc -lhsa-runtime64 -lamd_comgr -lhsakmt
