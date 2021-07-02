import torch
import torch.utils.data as data
import os
file_path = os.path.dirname(os.path.abspath(__file__))
TORCH_TENSOR_PATH = file_path + '/../data/processed_data/highd/{}_{}.pt'

class HighD(data.Dataset):
    def __init__(self, input_seq_paths, input_masks_paths, input_edge_types_paths, pred_seq_paths, device='cpu'):
        super().__init__()

        print('Building HighD dataset...')
        
        self.input_seqs = []
        self.input_masks = []
        self.input_edge_types = []
        self.pred_seqs = []
        for input_seq_path, input_masks_path, input_edge_types_path, pred_seq_path in \
            zip(input_seq_paths, input_masks_paths, input_edge_types_paths, pred_seq_paths):
            input_seq = torch.load(input_seq_path)
            input_masks = torch.load(input_masks_path)
            input_edge_types = torch.load(input_edge_types_path)
            pred_seq = torch.load(pred_seq_path)

            self.input_seqs.append(input_seq)
            self.input_masks.append(input_masks)
            self.input_edge_types.append(input_edge_types)
            self.pred_seqs.append(pred_seq)

        self.input_seqs = torch.cat(self.input_seqs).float().to(device).squeeze()
        self.input_masks = torch.cat(self.input_masks).float().to(device).squeeze()
        self.input_edge_types = torch.cat(self.input_edge_types).float().to(device).squeeze()
        self.pred_seqs = torch.cat(self.pred_seqs).float().to(device).squeeze()

    def __getitem__(self, idx):

        example = (self.input_seqs[idx],
                   self.input_masks[idx],
                   self.input_edge_types[idx],
                   self.pred_seqs[idx])

        return example

    def __len__(self):
        return len(self.input_seqs)

def build_highd_data_loader(one_indexed_dataset_list, batch_size=100, device='cpu'):
    input_seq_paths = []
    input_masks_paths = []
    input_edge_types_paths = []
    pred_seq_paths = []

    for data_id in one_indexed_dataset_list:
        data_str = '{:02d}'.format(data_id)
        input_seq_paths.append(TORCH_TENSOR_PATH.format(data_str, 'input_seq'))
        input_masks_paths.append(TORCH_TENSOR_PATH.format(data_str, 'input_masks'))
        input_edge_types_paths.append(TORCH_TENSOR_PATH.format(data_str, 'input_edge_types'))
        pred_seq_paths.append(TORCH_TENSOR_PATH.format(data_str, 'pred_seq'))

    dataset = HighD(input_seq_paths, input_masks_paths, input_edge_types_paths, 
                    pred_seq_paths, device=device)

    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    return data_loader



if __name__ == '__main__':
    # Test

    dataset_list = [1, 2, 16]
    batch_size = 64

    data_loader = build_highd_data_loader(dataset_list, batch_size, device='cpu')
    
    for input_seq, input_masks, input_edge_types, pred_seq in data_loader:
        print(input_seq.shape) # (bs, in_seq_len, grid, grid, feat_dim)
        print(input_masks.shape) # (bs, in_seq_len, grid, grid)
        print(input_edge_types.shape) # (bs, in_seq_len, grid, grid, n_types=4)
        print(pred_seq.shape) # (bs, pred_seq_len, grid, grid, feat_dim)