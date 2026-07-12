// maxcore_sm102_unified.cu — All SM102 kernels in a single translation unit.
//
// Combines flash_attn, im2col, conv_wmma, and both reduction variants so the
// unified launcher (launcher.cu) can #include this one file and expose every
// kernel through a single compiled .so.

#include "flashattn_sm102.cu"
#include "conv_sm102.cu"
#include "reduction_sm102.cu"
