import torch
import argparse
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from training_helper_functions import get_features_given_blocks
from GNN_model_architecture import FakeNewsModel
import dgl
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def do_evaluation(model, g, args, test_dataloader, loss_fcn=None):

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


def evaluate_split(overall_graph, args):
    """
    Evaluate the model on the first split, creating a test set from the train set.
    """
    model_file = os.path.join(args.path_where_models_are, "best_at_dev_1.pth")
    if not os.path.exists(model_file):
        print(f"Model checkpoint not found at: {model_file}")
        return

    g = overall_graph._g[0]
    print("Graph loaded for evaluation.")

    model = FakeNewsModel(
        in_features={'source': 778, 'user': 773, 'article': 768},
        hidden_features=128,
        out_features=3,
        canonical_etypes=g.canonical_etypes,
        num_workers=args.num_workers,
        n_layers=args.n_layers,
        conv_type='gcn'
    ).to(torch.device('cuda'))

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    train_set_path = os.path.join(args.path_where_graph_is, "train_set_0.npy")
    dev_set_path = os.path.join(args.path_where_graph_is, "dev_set_0.npy")
    if not os.path.exists(train_set_path) or not os.path.exists(dev_set_path):
        print("Train or dev set files not found.")
        return

    train_idx = np.load(train_set_path, allow_pickle=True)
    dev_idx = np.load(dev_set_path, allow_pickle=True)

    train_idx, test_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    sampler = MultiLayerFullNeighborSampler(args.n_layers)
    test_dataloader = DataLoader(
        g, {'source': torch.from_numpy(test_idx)}, sampler,
        batch_size=64, shuffle=True, drop_last=False
    )

    loss_fcn = torch.nn.CrossEntropyLoss()
    acc, loss, f1 = do_evaluation(model, g, args, test_dataloader, loss_fcn)

    print(f"Test Metrics -- Accuracy: {acc}, Loss: {loss}, F1 Score: {f1}")

    results_path = os.path.join(args.path_to_save_results, "evaluation_results_split_0.json")
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
    parser.add_argument("--n_layers", type=int, default=5, help="Number of GNN layers.")
    parser.add_argument('--dataset_corpus', type=str, required=True, help="Path to the corpus.tsv file.")

    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    print("Evaluating on GPU:", args.gpu)

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

    evaluate_split(overall_graph, args)
