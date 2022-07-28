import numpy as np
from onnxruntime import SessionOptions, InferenceSession, RunOptions
import onnxruntime as ort
from time import perf_counter


def get_mem_info():
    with open("/proc/self/status", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "VmPeak" in line or "VmHWM" in line or "VmRSS" in line:
                print(line.strip())
options = SessionOptions()
#  #  options.use_device_allocator_for_initializers = "1"
options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
#  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
run_opt = RunOptions()
#  run_opt.log_verbosity_level = 0
import onnxruntime as ort
ort.set_default_logger_severity(0)
#
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 1,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 10178000000,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
]

#  providers=["CPUExecutionProvider"]
np.random.seed(0)
au_sig = np.random.normal(0, 0.1, [1, 80, 32]).astype(np.float32)
length = np.array([32], dtype=np.int64)
start = perf_counter()

model = InferenceSession('./citrinet_ext_1/citrinet.onnx', options, providers=providers)
#  model = InferenceSession('/n/w1-knguyen/platform/platform_git/TICKETS/CTC-citri/am-qlstmp/am_ctc/citrinet_seq_len_ext/citrinet.onnx', options, providers=providers)
end = perf_counter()
print(f"load time {end - start}")
output = model.run(None, {"audio_signal": au_sig, "length": length}, run_options=run_opt)
#  np.save('ref_out', output)

#  output

print("ort version {}".format(ort.__version__))
get_mem_info()

output_ref = np.load('ref_out.npy')
print(np.abs(output_ref - output).max())

