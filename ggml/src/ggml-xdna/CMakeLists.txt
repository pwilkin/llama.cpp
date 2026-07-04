ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp)

# AMD XDNA / Ryzen AI NPU userspace runtime (XRT + the amdxdna driver).
#
# The initial backend uses host-side placeholder kernels, so XRT is OPTIONAL and
# the backend builds without it. When XRT is found (system install under
# /opt/xilinx/xrt, or $XILINX_XRT), we enable the on-NPU execution path
# (GGML_XDNA_HAS_XRT) that real AIE `.xclbin` kernels will use.
find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h
    PATHS /opt/xilinx/xrt/include $ENV{XILINX_XRT}/include)

find_library(XRT_COREUTIL
    NAMES xrt_coreutil
    PATHS /opt/xilinx/xrt/lib $ENV{XILINX_XRT}/lib /usr/lib/x86_64-linux-gnu)

if (XRT_INCLUDE_DIR AND XRT_COREUTIL)
    message(STATUS "ggml-xdna: found XRT (${XRT_COREUTIL}) — on-NPU execution path enabled")
    target_include_directories(ggml-xdna PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna PRIVATE ${XRT_COREUTIL})
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_HAS_XRT)
else()
    message(STATUS "ggml-xdna: XRT not found — building with host-side placeholder kernels only "
                   "(install XRT / set XILINX_XRT to enable NPU execution)")
endif()
