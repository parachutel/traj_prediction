import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--',
    #                     type=,
    #                     default=,
    #                     help='')

    # Training and eval
    parser.add_argument('--name',
                        type=str,
                        default='train',
                        help='Name of the run.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='The path to an existing checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='The maximum number of checkpoints to keep.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='nll_p',
                        help='The metric to track when saving model (checkout Predictor.get_eval_loss()).')
    parser.add_argument('--maximize_metric',
                        type=lambda s: s.lower().startswith('t'),
                        default=False, # minimize nll
                        help='Whether maximize or minimize the metric')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=128,
                        help='The batch size during training')
    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=64,
                        help='The batch size during eval')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=4000,
                        help='The number of epochs to trian for.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.,
                        help='The maximum gradient norm for clipping.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='The base directory for saving model.')
    parser.add_argument('--eval_epochs',
                        type=int,
                        default=5,
                        help='Eval frequency.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learning_decay_rate',
                        type=float,
                        default=0.9999)
    parser.add_argument('--pred_id',
                        type=int,
                        default=0,
                        help='The track ID to predict (manual visualization).')
    parser.add_argument('--n_vis',
                        type=int,
                        default=20,
                        help='The number of trajecories to visualize.')
    # Dataset features
    parser.add_argument('--state_dim',
                        type=int,
                        default=6,
                        help='The feature dimension of the inputs.')
    parser.add_argument('--pred_dim',
                        type=int,
                        default=2,
                        help='The feature dimension of the prediction outputs.')
    parser.add_argument('--n_edge_types',
                        type=int,
                        default=4,
                        help='The number of different types of vehicle interactions.')
    ## Data preparing
    parser.add_argument('--forward_shift_seconds',
                        type=int,
                        default=2,
                        help='The number of seconds to shift when picking data segments.')
    parser.add_argument('--input_seconds',
                        type=int,
                        default=4,
                        help='The number of seconds of input sequence length.')
    parser.add_argument('--pred_seconds',
                        type=int,
                        default=2,
                        help='The number of seconds of prediction sequence length.')
    ## process_highd.py
    parser.add_argument('--mode', default='pickle', type=str)
    parser.add_argument('--n_records', default=60, type=int)
    parser.add_argument('--fps', default=20, type=int)
    parser.add_argument('--dataset_id', default=1, type=int)
    parser.add_argument('--track_id', default=58, type=int)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--end', default=60, type=int)

    # Network Meta
    parser.add_argument('--mlp_dropout_prob',
                        type=float,
                        default=0.1,
                        help='The dropout prob of MLP layers.')
    parser.add_argument('--rnn_dropout_prob',
                        type=float,
                        default=0.25,
                        help='The dropout prob of RNN layers.')

    # Encoder
    parser.add_argument('--nhe_hidden_size',
                        type=int,
                        default=128,
                        help='The hidden size of the node history encoder.')
    parser.add_argument('--ehe_hidden_size',
                        type=int,
                        default=128,
                        help='The hidden size of the edge history encoder.')
    parser.add_argument('--nfe_hidden_size',
                        type=int,
                        default=128,
                        help='The hidden size of the node future encoder.')

    # DiscreteLatent
    parser.add_argument('--latent_input_size',
                        type=int,
                        default=128,
                        help='The feature dimension of the inputs of latent encoder.')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=5,
                        help='The feature dimension of the latent variable.')
    parser.add_argument('--kl_min',
                        type=float,
                        default=0.07,
                        help='The minimum KL divergence between q_z_xy and p_z_x.')
    ## DiscreteLatent: Relaxed One-Hot Temperature Annealing
    parser.add_argument('--temp_init',
                        type=float,
                        default=2.0,
                        help='The initial temperature.')
    parser.add_argument('--temp_final',
                        type=float,
                        default=0.001,
                        help='The final temperature.')
    parser.add_argument('--temp_decay_rate',
                        type=float,
                        default=0.9999,
                        help='The decay rate of the temperature.')
    ## DiscreteLatent: Logit Clipping
    parser.add_argument('--use_z_logit_clipping',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether clip z logits when getting the distrubution of z from zero mean logits')
    parser.add_argument('--z_logit_clip_start',
                        type=float,
                        default=0.05)
    parser.add_argument('--z_logit_clip_final',
                        type=float,
                        default=3.0)
    parser.add_argument('--z_logit_clip_crossover',
                        type=float,
                        default=8000)
    parser.add_argument('--z_logit_clip_divisor',
                        type=float,
                        default=6.)

    # Decoder
    parser.add_argument('--decoder_hidden_size',
                        type=int,
                        default=128,
                        help='The hidden size of the decoder LSTMCell.')
    parser.add_argument('--gmm_components',
                        type=int,
                        default=16,
                        help='The number of components of GMM.')
    parser.add_argument('--log_sigma_min',
                        type=float,
                        default=-10.,
                        help='The min log stdev of GMM2D.')
    parser.add_argument('--log_sigma_max',
                        type=float,
                        default=10.,
                        help='The max log stdev of GMM2D.')
    parser.add_argument('--log_p_yt_xz_max',
                        type=float,
                        default=50.,
                        help='The maximum log prob of the truth future evaluated on the predicted distrubution.')
    ## Decoder: decoding_sample_model_prob
    parser.add_argument('--sample_model_during_dec',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether sample sample model during decoding (if False, use true future as inputs).')
    parser.add_argument('--dec_sample_model_prob_start',
                        type=float,
                        default=0.)
    parser.add_argument('--dec_sample_model_prob_final',
                        type=float,
                        default=0.)
    parser.add_argument('--dec_sample_model_prob_crossover',
                        type=int,
                        default=20000)
    parser.add_argument('--dec_sample_model_prob_divisor',
                        type=int,
                        default=6)

    # Predictor
    parser.add_argument('--n_pred_steps',
                        type=int,
                        default=50)
    ## Variational objective
    parser.add_argument('--kl_weight',
                        type=float,
                        default=1.0,
                        help='The weight of the KL term in ELBO (beta)')
    parser.add_argument('--kl_weight_start',
                        type=float,
                        default=0.0001)
    # parser.add_argument('--kl_decay_rate',
    #                     type=float,
    #                     default=0.99995)
    parser.add_argument('--kl_crossover',
                        type=int,
                        default=8000)
    parser.add_argument('--kl_sigmoid_divisor',
                        type=int,
                        default=6)

    parser.add_argument('--alpha',
                        type=float,
                        default=1)
    parser.add_argument('--n_z_samples_training',
                        type=int,
                        default=3,
                        help='The number of samples of latent variable during training.')
    parser.add_argument('--n_z_samples_eval',
                        type=int,
                        default=50,
                        help='The number of samples of latent variable during eval.')
    parser.add_argument('--n_z_samples_pred',
                        type=int,
                        default=10,
                        help='The number of samples of latent variable during prediction.')
    parser.add_argument('--use_iwae',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether use importance weighted autoencoder (only matters if alpha = 1).')
    parser.add_argument('--kl_exact',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether compute the exact KL divergence in objective (only matters if alpha = 1).')


    # Misc
    parser.add_argument('--debug',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether print debug info.')

    


    args = parser.parse_args()
    return args

args = get_args()