# import onnx

# model = onnx.load('./save/tmp/decoder.onnx')

# # Check that the IR is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)


import onnxruntime as ort
import numpy as np
from args import args

predictor = ort.InferenceSession('./save/tmp/predictor.onnx')

in_seq_len = args.input_seconds * args.highd_frame_rate
bs = 1

input_names = ['input_seq', 'input_mask', 'input_edge_types']
input_shapes = [(bs, in_seq_len, 3, 3, args.state_dim), 
                (bs, in_seq_len, 3, 3), 
                (bs, in_seq_len, 3, 3, args.n_edge_types)]

dummy_inputs = [(name, np.random.rand(*shape).astype(np.float32)) 
                for name, shape in zip(input_names, input_shapes)]
dummy_inputs = dict(dummy_inputs)

outputs = predictor.run(None, dummy_inputs)

print(outputs[0].shape, end - start)
