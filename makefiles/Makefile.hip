HIPCC?=hipcc
HIPFLAGS?=#-fPIC
ifdef IEEE
	HIPFLAGS += $(HIPIEEE)
endif
HIP_INC ?= -I$(HIP_INSTALL_PATH)/include
HIP_LIB ?= $(HIP_INSTALL_PATH)/lib
HIP_LINK ?= -L$(HIP_INSTALL_PATH)/lib -L$(HIP_INSTALL_PATH)/lib64 -L$(HIP_INSTALL_PATH)/hsa/lib64 -L$(HIP_INSTALL_PATH)/hsa/lib -L$(HIP_INSTALL_PATH)/hcc/lib -lhc_am -lmcwamp -lhip_hcc -lhsa-runtime64 -lamd_comgr -lhsakmt $(shell mpicxx --showme:link)
HIPMPICPP ?= $(HIPCC)
