import os
import torch
import util
import onnxruntime as ort
import numpy as np
from predictor.predictor import Predictor
from args import args

def export_onnx(opset_version=12):
    device, args.gpu_ids = 'cpu', None
    model = Predictor(state_dim=args.state_dim,
                      rel_state_dim=args.state_dim,
                      pred_dim=args.pred_dim,
                      edge_type_dim=args.n_edge_types,
                      nhe_hidden_size=args.nhe_hidden_size,
                      ehe_hidden_size=args.ehe_hidden_size,
                      nfe_hidden_size=args.nfe_hidden_size,
                      decoder_hidden_size=args.decoder_hidden_size,
                      gmm_components=args.gmm_components,
                      log_sigma_min=args.log_sigma_min,
                      log_sigma_max=args.log_sigma_max,
                      log_p_yt_xz_max=args.log_p_yt_xz_max,
                      kl_weight=args.kl_weight,
                      masked_ehe=args.masked_ehe,
                      device=device)
    
    print(f'Loading checkpoint from {args.load_path}...')
    model, step = util.load_model(model, args.load_path, args.gpu_ids)
    exp_name = args.load_path.split('/')[-2]
    path = './save/onnx/' + exp_name
    if not os.path.exists(path):
        os.makedirs(path)
    

    in_seq_len = args.input_seconds * args.highd_frame_rate
    bs = 1
    input_shapes = [(bs, in_seq_len, 3, 3, args.state_dim), 
                    (bs, in_seq_len, 3, 3), 
                    (bs, in_seq_len, 3, 3, args.n_edge_types)]
    input_names = ['input_seq', 'input_mask', 'input_edge_types']
    dummy_inputs = tuple([torch.rand(shape) for shape in input_shapes])

    torch.onnx.export(model, dummy_inputs, f'{path}/predictor.onnx', 
        verbose=False, input_names=input_names, 
        output_names=['sampled_future'],
        opset_version=opset_version)

    print(f'Exporting to {path}/predictor.onnx is successful!')

    # verify onnx
    predictor = ort.InferenceSession(f'{path}/predictor.onnx')
    dummy_inputs_dict = [(name, np.random.rand(*shape).astype(np.float32)) 
                         for name, shape in zip(input_names, input_shapes)]
    dummy_inputs_dict = dict(dummy_inputs_dict)

    outputs = predictor.run(None, dummy_inputs_dict)

    print('Output shape =', outputs[0].shape)
    print('Loading verification is successful!')


if __name__ == '__main__':
    export_onnx(opset_version=12)
