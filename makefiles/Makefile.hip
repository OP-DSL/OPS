HIPCC?=hipcc
HIPFLAGS?=#-fPIC
ifdef IEEE
	HIPFLAGS += $(HIPIEEE)
endif

HIP_INC ?= $(HIP_INSTALL_PATH)/include -I$(HIP_INSTALL_PATH)/../include
HIP_LIB ?= $(HIP_INSTALL_PATH)/lib
HIP_LINK ?= -L$(MPI_LIB) -lmpi
#HIP_LINK ?= $(shell mpicxx --showme:compile) -L$(HIP_INSTALL_PATH)/lib -L$(HIP_INSTALL_PATH)/lib64 -L$(HIP_INSTALL_PATH)/hsa/lib64 -L$(HIP_INSTALL_PATH)/hsa/lib -L$(HIP_INSTALL_PATH)/hcc/lib -lhc_am -lmcwamp -lhip_hcc -lhsa-runtime64 -lamd_comgr -lhsakmt $(shell mpicxx --showme:link)
HIPMPICPP ?= $(HIPCC)

ifneq ($(strip $(NV_ARCH)),)
HIPFLAGS += -D__HIP_PLATFORM_NVIDIA__
endif

ifneq ($(strip $(AMD_ARCH)),)
HIPFLAGS += -D__HIP_PLATFORM_AMD__
endif
