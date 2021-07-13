import torch
from data_utils.highd_dataset import build_highd_data_loader
from predictor.predictor import Predictor
from args import args

device='cpu'

# Load data:
dataset_list = [1]
batch_size = 64
data_loader = build_highd_data_loader(dataset_list, batch_size)


predictor = Predictor()

# OUTDATED tests
# print('Testing Predictor annealer...')

# print('Step 1:')
# print('predictor.kl_weight =', predictor.kl_weight)
# print('predictor.decoder.decoding_sample_model_prob =', predictor.decoder.decoding_sample_model_prob)
# print('predictor.encoder.latent.temp =', predictor.encoder.latent.temp)
# print('predictor.encoder.latent.z_logit_clip =', predictor.encoder.latent.z_logit_clip)

# for _ in range(4000):
#     predictor.step_annealers()

# print('Step 4000:')
# print('predictor.kl_weight =', predictor.kl_weight)
# print('predictor.decoder.decoding_sample_model_prob =', predictor.decoder.decoding_sample_model_prob)
# print('predictor.encoder.latent.temp =', predictor.encoder.latent.temp)
# print('predictor.encoder.latent.z_logit_clip =', predictor.encoder.latent.z_logit_clip)

# print('Checking Predictor parameters...')
# parameters = filter(lambda param: param.requires_grad, 
#                     predictor.parameters())
# for p in parameters:
#     print(p.shape)

for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
    print('Testing Predictor loss...')
    training_loss = predictor.get_training_loss(input_seq, input_masks, input_edge_types, pred_seq)
    print('training loss =', training_loss)
    nll_q_is, nll_p, nll_exact = predictor.get_eval_loss(input_seq, input_masks, input_edge_types, pred_seq,
                                                         compute_naive=True,
                                                         compute_exact=True)
    print('eval nll_q_is =', nll_q_is)
    print('eval nll_p =', nll_p)
    print('eval nll_exact =', nll_exact)

    print('Testing Predictor predict...')
    sampled_future, z_p_samples = predictor.predict(input_seq, input_masks, input_edge_types, 
                                                    num_samples=11, most_likely=False)
    print('sampled_future.shape =', sampled_future.shape)
    print('z_p_samples.shape =', z_p_samples.shape)

    break