OPS_HLS_INC=-I$(OPS_INSTALL_PATH)/hls/include -I/$(OPS_INSTALL_PATH)/L1/include \
			-I$(OPS_INSTALL_PATH)/hls/ext/xcl2 -I/$(OPS_INSTALL_PATH)/L2/include


#Checks for XILINX_VITIS
check-vitis:
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly using "source <Vitis_install_path>/Vitis/<Version>/settings64.sh" and rerun)
endif

#Checks for XILINX_XRT
check-xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly using "source /opt/xilinx/xrt/setup.sh" and rerun)
endif

#Check PLATFORM 
ifeq ($(PLATFORM),)
	ifneq ($(DEVICE),)
		$(warning WARNING: DEVICE is deprecated in make command. Please use PLATFORM instead)
		PLATFORM := $(DEVICE)
	endif
else
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

