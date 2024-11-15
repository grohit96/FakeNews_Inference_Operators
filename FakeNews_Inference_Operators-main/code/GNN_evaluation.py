import torch
import argparse
import os
import numpy as np
import json
from training_helper_functions import get_features_given_graph, get_features_given_blocks
from GNN_model_architecture import FakeNewsModel
import dgl
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import f1_score

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def do_evaluation(model, g, overall_graph, args, test_dataloader=None, loss_fcn=None):
    if test_dataloader is None:
        return 0.0, 0.0, 0.0

    g = overall_graph._g[0]
    model.eval()
    total_acc = 0
    total_loss = 0
    count = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for input_nodes, seeds, blocks in test_dataloader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            output_labels = blocks[-1].dstdata['source_label']['source']
            output_labels = (output_labels - 1).long().squeeze()

            node_features = get_features_given_blocks(g, args, blocks)
            output_predictions = model(blocks, node_features, g=None, neg_g=None)['source']

            acc = (output_predictions.argmax(dim=1) == output_labels).sum().item()
            total_acc += acc

            if loss_fcn:
                loss = loss_fcn(output_predictions, output_labels)
                total_loss += loss.item() * len(output_predictions)

            count += len(output_predictions)
            all_labels.extend(output_labels.cpu().numpy())
            all_predictions.extend(output_predictions.argmax(dim=1).cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='macro')
    return total_acc / count, total_loss / count, f1

def evaluate_all_splits(overall_graph, args):
    model_paths = sorted([file for file in os.listdir(args.path_where_models_are) if "best_at_dev" in file])

    if not model_paths:
        print("No model checkpoints found. Please check the model path.")
        return

    g = overall_graph._g[0]
    print("Loaded the graph for evaluation.")

    for model_path in model_paths:
        model_file = os.path.join(args.path_where_models_are, model_path)
        print(f"Loading model: {model_file}")

        model = FakeNewsModel(
            in_features={'source': 778, 'user': 773, 'article': 768},
            hidden_features=128,  # Update this to match training
            out_features=128,  # Update this to match training
            canonical_etypes=g.canonical_etypes,
            num_workers=args.num_workers,
            n_layers=3,  # Ensure this matches training
            conv_type='gcn'  # Ensure this matches training
        ).to(torch.device('cuda'))


        state_dict = torch.load(model_file)
        
        print("Model Architecture:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
            
        print("\nCheckpoint Parameters:")
        for name, param in state_dict.items():
            print(f"{name}: {param.shape}") 
            
               
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        data_split_key = model_path.split('_')[-1].replace('best_at_dev', '').strip()
        split_mapping = {
            "connected": "0",
            "dev": "1",  # Adjust based on your test set naming
        }
        data_split_key = split_mapping.get(data_split_key, data_split_key)

        print(f"Evaluating split: {data_split_key}")

        # Load dataset splits
        dev_set_path = f"{args.path_where_graph_is}/dev_set_{data_split_key}.npy"
        test_set_path = f"{args.path_where_graph_is}/test_set_{data_split_key}.npy"

        if not os.path.exists(test_set_path):
            print(f"Test set not found. Falling back to dev set: {dev_set_path}")
            dataset_path = dev_set_path
        else:
            dataset_path = test_set_path

        test_idx = torch.from_numpy(np.load(dataset_path, allow_pickle=True))
        sampler = MultiLayerFullNeighborSampler(args.n_layers)
        test_dataloader = DataLoader(g, {'source': test_idx}, sampler, batch_size=64, shuffle=True, drop_last=False)

        acc, loss, f1 = do_evaluation(model, g, overall_graph, args, test_dataloader)
        print(f"Split {data_split_key} -- Accuracy: {acc}, Loss: {loss}, F1 Score: {f1}")

        results_path = os.path.join(args.path_to_save_results, f"evaluation_results_split_{data_split_key}.json")
        with open(results_path, 'w') as f:
            json.dump({"accuracy": acc, "loss": loss, "f1_score": f1}, f)
        print(f"Results saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Fake News Detection Models")
    parser.add_argument('--path_where_graph_is', type=str, required=True, help="Path to the graph files.")
    parser.add_argument('--path_where_models_are', type=str, required=True, help="Path to the trained model files.")
    parser.add_argument('--path_to_save_results', type=str, required=True, help="Path to save evaluation results.")
    parser.add_argument('--gpu', type=str, default='0', help="GPU device ID to use.")
    parser.add_argument("--hidden_features", type=int, default=128, help="Number of hidden features.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker threads.")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument('--dataset_corpus', type=str, required=True, help="Path to the corpus.tsv file.")

    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    print("Evaluating on GPU:", args.gpu)

    # Load the graph dataset
    from graph_helper_functions import FakeNewsDataset
    overall_graph = FakeNewsDataset(
        save_path=args.path_where_graph_is,
        users_that_follow_each_other_path='dataset_release/users_that_follow_each_other.npy',
        dataset_corpus=args.dataset_corpus,
        followers_dict_path='dataset_release/source_followers_dict.npy',
        source_username_mapping_dict_path='dataset_release/domain_twitter_triplets_dict.npy',
        building_original_graph_first=False
    )
    

    overall_graph.load()

    evaluate_all_splits(overall_graph, args)
