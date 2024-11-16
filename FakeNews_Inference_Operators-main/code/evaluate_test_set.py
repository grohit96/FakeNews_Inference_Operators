import torch
import argparse
import numpy as np
import os
import json
from training_helper_functions import get_features_given_blocks
from GNN_model_architecture import FakeNewsModel
import dgl
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import f1_score


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate_test_set(overall_graph, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = overall_graph._g[0]
    print("Loaded graph for evaluation.")

    model_file = os.path.join(args.path_where_models_are, "best_at_dev_1.pth")
    print(f"Loading model: {model_file}")
    
    model = FakeNewsModel(
        in_features={'source': 778, 'user': 773, 'article': 768},
        hidden_features=128,
        out_features=128,  
        canonical_etypes=g.canonical_etypes,
        num_workers=args.num_workers,
        n_layers=2,
        conv_type='gcn'
    ).to(device)

    state_dict = torch.load(model_file)
    print("Model Architecture:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    print("\nCheckpoint Parameters:")
    state_dict = torch.load(model_file)
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")


    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    
    test_set_path = os.path.join(args.path_where_graph_is, "test_set_1.npy")
    test_indices = np.load(test_set_path, allow_pickle=True)
    test_indices = torch.from_numpy(test_indices).to(device)
    print(f"Loaded test set with {len(test_indices)} samples.")

    sampler = MultiLayerFullNeighborSampler(args.n_layers)
    test_dataloader = DataLoader(
        g,
        {'source': test_indices},
        sampler,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )

    total_acc = 0
    total_loss = 0
    count = 0
    all_labels = []
    all_predictions = []

    loss_fcn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_nodes, seeds, blocks in test_dataloader:
            blocks = [b.to(device) for b in blocks]
            output_labels = blocks[-1].dstdata['source_label']['source']
            output_labels = (output_labels - 1).long().squeeze()

            node_features = get_features_given_blocks(g, args, blocks)
            output_predictions = model(blocks, node_features, g=None, neg_g=None)['source']

            acc = (output_predictions.argmax(dim=1) == output_labels).sum().item()
            total_acc += acc

            loss = loss_fcn(output_predictions, output_labels)
            total_loss += loss.item() * len(output_predictions)

            count += len(output_predictions)
            all_labels.extend(output_labels.cpu().numpy())
            all_predictions.extend(output_predictions.argmax(dim=1).cpu().numpy())

    accuracy = total_acc / count
    loss = total_loss / count
    f1 = f1_score(all_labels, all_predictions, average='macro')

    results = {
        "accuracy": accuracy,
        "loss": loss,
        "f1_score": f1
    }
    results_path = os.path.join(args.path_to_save_results, "evaluation_results_test_set_1.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)

    print(f"Evaluation Results -- Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, F1 Score: {f1:.4f}")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    print("Evaluation script started...")

    parser = argparse.ArgumentParser(description="Evaluate Fake News Detection Model on Test Set")
    parser.add_argument('--path_where_graph_is', type=str, required=True, help="Path to the graph files.")
    parser.add_argument('--path_where_models_are', type=str, required=True, help="Path to the trained model files.")
    parser.add_argument('--path_to_save_results', type=str, required=True, help="Path to save evaluation results.")
    parser.add_argument("--hidden_features", type=int, default=512, help="Number of hidden features.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker threads.")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers.")
    parser.add_argument('--gpu', type=str, default='0', help="GPU device ID to use.")
    parser.add_argument('--dataset_corpus', type=str, required=True, help="Path to the corpus.tsv file.")


    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    print("Using GPU:", args.gpu)

    from graph_helper_functions import FakeNewsDataset
    overall_graph = FakeNewsDataset(
        save_path=args.path_where_graph_is,
        users_that_follow_each_other_path='dataset_release/users_that_follow_each_other.npy',
        dataset_corpus=args.dataset_corpus,  # Ensure this uses the argument
        followers_dict_path='dataset_release/source_followers_dict.npy',
        source_username_mapping_dict_path='dataset_release/domain_twitter_triplets_dict.npy',
        building_original_graph_first=False
    )


    overall_graph.load()
    evaluate_test_set(overall_graph, args)
