import os
import sys
import json
import argparse
from datetime import datetime
import builtins

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Add current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import GCL_Encoder, LogisticRegression
from augmentations import TemporalAugmentor


def parse_args():
    parser = argparse.ArgumentParser(description="Train GraphCL on Elliptic++ dataset")
    parser.add_argument("--epochs", type=int, default=300, help="Number of pre-training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training (per GPU)")
    parser.add_argument("--hidden-channels", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--out-channels", type=int, default=128, help="Output embedding size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for pre-training")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--skip-contraction", action="store_true", help="Skip graph contraction step")
    parser.add_argument("--project-root", type=str, default=None, help="Project root directory")
    return parser.parse_args()


def setup_ddp():
    """Initialize Distributed Data Parallelism if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # Only print on rank 0 to avoid clutter
        if rank == 0:
            print(f"Initialized DDP: World Size {world_size}")
        return True, rank, local_rank, world_size, device
    else:
        # Fallback for non-distributed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initialized Single-GPU: Device {device}")
        return False, 0, 0, 1, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def perform_graph_contraction(project_root):
    """
    Contracts the bipartite Tx->Addr->Tx graph into homogeneous Tx->Tx edges.
    Only keeps causal edges (dst_time >= src_time).
    """
    print("\n" + "="*60)
    print("STEP 1: GRAPH CONTRACTION")
    print("="*60)
    
    ELLIPTIC_PP_DIR = os.path.join(project_root, "elliptic++_bitcoin_dataset")
    print(f"Data Directory: {ELLIPTIC_PP_DIR}")

    # Load Tx -> Addr edges
    tx_addr_path = os.path.join(ELLIPTIC_PP_DIR, "TxAddr_edgelist.csv")
    print(f"Loading {tx_addr_path}...")
    df_tx_addr = pd.read_csv(tx_addr_path)
    df_tx_addr.columns = ["txId_src", "address"]

    # Load Addr -> Tx edges
    addr_tx_path = os.path.join(ELLIPTIC_PP_DIR, "AddrTx_edgelist.csv")
    print(f"Loading {addr_tx_path}...")
    df_addr_tx = pd.read_csv(addr_tx_path)
    df_addr_tx.columns = ["address", "txId_dst"]

    # Perform contraction
    print("Performing Graph Contraction (Tx->Addr + Addr->Tx = Tx->Tx)...")
    df_contracted = pd.merge(df_tx_addr, df_addr_tx, on="address", how="inner")

    print(f"  - Tx->Addr edges: {len(df_tx_addr):,}")
    print(f"  - Addr->Tx edges: {len(df_addr_tx):,}")
    print(f"  - Contracted Tx->Tx edges: {len(df_contracted):,}")

    # Load timestamps
    features_path = os.path.join(ELLIPTIC_PP_DIR, "txs_features.csv")
    print(f"Loading timestamps from {features_path}...")
    df_features = pd.read_csv(features_path, usecols=["txId", "Time step"])
    time_map = df_features.set_index("txId")["Time step"].to_dict()

    # Map timestamps
    print("Mapping timestamps to edges...")
    df_contracted["src_time"] = df_contracted["txId_src"].map(time_map)
    df_contracted["dst_time"] = df_contracted["txId_dst"].map(time_map)
    df_contracted = df_contracted.dropna(subset=["src_time", "dst_time"])

    # Calculate stats
    num_same_time = (df_contracted["dst_time"] == df_contracted["src_time"]).sum()
    num_forward = (df_contracted["dst_time"] > df_contracted["src_time"]).sum()
    num_backward = (df_contracted["dst_time"] < df_contracted["src_time"]).sum()

    print("\n✅ Contracted Graph Temporal Analysis:")
    print(f"  Same-timestep edges: {num_same_time:,} ({100*num_same_time/len(df_contracted):.1f}%)")
    print(f"  Forward-causal edges: {num_forward:,} ({100*num_forward/len(df_contracted):.1f}%)")
    print(f"  Backward-invalid edges: {num_backward:,} ({100*num_backward/len(df_contracted):.3f}%)")

    # Save valid edges only
    valid_edges = df_contracted[df_contracted["dst_time"] >= df_contracted["src_time"]]
    output_file = os.path.join(ELLIPTIC_PP_DIR, "elliptic_pp_contracted_edgelist.csv")
    print(f"\nSaving {len(valid_edges):,} valid causal edges to {output_file}...")
    valid_edges[["txId_src", "txId_dst"]].to_csv(output_file, index=False)
    print("Done.")


def load_data(project_root):
    """
    Load the Elliptic++ dataset and create PyG Data object.
    Returns data on CPU.
    """
    print("\n" + "="*60)
    print("STEP 2: LOADING DATA")
    print("="*60)
    
    ELLIPTIC_PP_DIR = os.path.join(project_root, "elliptic++_bitcoin_dataset")
    features_path = os.path.join(ELLIPTIC_PP_DIR, "txs_features.csv")
    classes_path = os.path.join(ELLIPTIC_PP_DIR, "txs_classes.csv")
    edges_path = os.path.join(ELLIPTIC_PP_DIR, "elliptic_pp_contracted_edgelist.csv")

    # Load features
    print("Loading features...")
    df_features = pd.read_csv(features_path)
    df_features = df_features.rename(columns={"Time step": "timeStep"})
    df_features = df_features.set_index("txId")

    feature_cols = [c for c in df_features.columns if c != "timeStep"]
    
    # Handle NaN/Inf in features
    feature_values = df_features[feature_cols].values
    feature_values = pd.DataFrame(feature_values).fillna(0).values  # Replace NaN with 0
    
    x = torch.tensor(feature_values, dtype=torch.float)
    
    # Check for remaining issues
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("WARNING: NaN/Inf detected in features, replacing with 0")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    time_vals = df_features["timeStep"].values
    time = torch.tensor(time_vals, dtype=torch.long)

    # Load classes
    print(f"Loading classes from {classes_path}...")
    df_classes = pd.read_csv(classes_path).set_index("txId")
    df_classes = df_classes.reindex(df_features.index)

    class_map = {1: 1, 2: 0, 3: -1}  # 1=Illicit, 2=Licit, 3=Unknown
    y = torch.tensor(df_classes["class"].map(class_map).fillna(-1).values, dtype=torch.long)

    # Create node mapping
    tx_ids = list(df_features.index)
    node_mapping = {int(txid): idx for idx, txid in enumerate(tx_ids)}

    # Load edges
    print(f"Loading edges from {edges_path}...")
    df_edges = pd.read_csv(edges_path)

    valid_src = df_edges["txId_src"].isin(node_mapping)
    valid_dst = df_edges["txId_dst"].isin(node_mapping)
    df_edges = df_edges[valid_src & valid_dst]

    # Deduplicate edges
    print("Deduplicating edges (calculating weights)...")
    df_edges_grouped = df_edges.groupby(["txId_src", "txId_dst"]).size().reset_index(name="weight")

    print(f"  - Original edges: {len(df_edges):,}")
    print(f"  - Deduplicated edges: {len(df_edges_grouped):,}")

    src_idx = df_edges_grouped["txId_src"].map(node_mapping).values
    dst_idx = df_edges_grouped["txId_dst"].map(node_mapping).values
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    edge_attr = torch.tensor(df_edges_grouped["weight"].values, dtype=torch.float).view(-1, 1)

    # Create masks
    train_mask = time < 35
    test_mask = time >= 35

    # Create Data object
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        y=y, 
        time=time, 
        train_mask=train_mask, 
        test_mask=test_mask
    )
    data.tx_id_to_node = node_mapping
    data.node_to_tx_id = {v: k for k, v in node_mapping.items()}
    data.feature_columns = feature_cols

    print("\n✅ Elliptic++ Data Object Created:")
    print(data)
    print(f"  Num nodes: {data.num_nodes:,}")
    print(f"  Num edges: {data.num_edges:,}")
    print(f"  Features shape: {data.x.shape}")
    print(f"  Class distribution: {torch.unique(data.y, return_counts=True)}")
    
    return data


def contrastive_loss(z1, z2, temperature, device):
    """NT-Xent contrastive loss with numerical stability."""
    # Handle empty batches
    if z1.size(0) == 0 or z2.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Check for NaN/Inf in inputs
    if torch.isnan(z1).any() or torch.isnan(z2).any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # Clamp to prevent numerical overflow in cross_entropy
    sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)
    
    labels = torch.arange(z1.size(0)).to(device)
    return F.cross_entropy(sim_matrix, labels)


def train_encoder(data, args, device, results_dir, run_id, is_distributed, rank, local_rank, world_size):
    """
    Train the GCL encoder using contrastive learning.
    """
    if rank == 0:
        print("\n" + "="*60)
        print("STEP 3: CONTRASTIVE PRE-TRAINING")
        print("="*60)
    
    # Config
    config = {
        "run_id": run_id,
        "batch_size": args.batch_size,
        "hidden_channels": args.hidden_channels,
        "out_channels": args.out_channels,
        "lr_pretrain": args.lr,
        "epochs_pretrain": args.epochs,
        "num_neighbors": [25, 15],
        "temperature": 0.1,
        "intra_step_drop_prob": 0.5,
        "inter_step_drop_prob": 0.0,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "device": str(device),
        "distributed": is_distributed,
        "world_size": world_size
    }
    
    # Save config (only rank 0)
    if rank == 0:
        with open(os.path.join(results_dir, f"config_{run_id}.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    # Prepare input nodes for DDP
    if is_distributed:
        # Get all training indices
        train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
        # Split indices among ranks
        total_len = len(train_indices)
        chunk_size = total_len // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank != world_size - 1 else total_len
        input_nodes = train_indices[start:end]
        print(f"Rank {rank}: Processing {len(input_nodes)}/{total_len} training nodes")
    else:
        input_nodes = data.train_mask

    # Setup Loader
    # Note: data is on CPU, which is good for NeighborLoader memory usage
    train_loader = NeighborLoader(
        data,
        num_neighbors=[25, 15],
        batch_size=args.batch_size,
        input_nodes=input_nodes,
        shuffle=True,
        num_workers=args.num_workers
    )

    encoder = GCL_Encoder(data.num_features, args.hidden_channels, args.out_channels).to(device)
    
    if is_distributed:
        encoder = DDP(encoder, device_ids=[local_rank])
        
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    augmentor = TemporalAugmentor(intra_step_drop_prob=0.5, inter_step_drop_prob=0.0)

    if rank == 0:
        print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
        print(f"Batch size: {args.batch_size} (per GPU)")
        print(f"Epochs: {args.epochs}")
        print("\nStarting Unsupervised Pre-training...")
    
    encoder.train()
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            # NeighborLoader with manual input_nodes splitting doesn't need set_epoch
            pass
            
        total_loss = 0
        steps = 0

        for batch in train_loader:
            batch = batch.to(device)

            view1 = augmentor.get_view(batch, mode='temporal_edge')
            view2 = augmentor.get_view(batch, mode='feature')

            # Manually concatenate tensors to avoid PyG Batch.from_data_list issues
            # Concatenate node features
            x_combined = torch.cat([view1.x, view2.x], dim=0)
            
            # Concatenate edge_index with offset for second view
            num_nodes_v1 = view1.num_nodes
            edge_index_v2_offset = view2.edge_index + num_nodes_v1
            edge_index_combined = torch.cat([view1.edge_index, edge_index_v2_offset], dim=1)
            
            # Concatenate edge_attr
            if view1.edge_attr is not None and view2.edge_attr is not None:
                edge_attr_combined = torch.cat([view1.edge_attr, view2.edge_attr], dim=0)
            else:
                edge_attr_combined = None
            
            # Single forward pass through encoder
            _, z_combined = encoder(x_combined, edge_index_combined, edge_attr_combined)
            
            # Split embeddings back
            z1 = z_combined[:num_nodes_v1]
            z2 = z_combined[num_nodes_v1:]

            loss = contrastive_loss(z1, z2, temperature=0.1, device=device)

            # Skip batch if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0
        
        # Reduce loss for logging (optional, but good for monitoring)
        if is_distributed:
            loss_tensor = torch.tensor(avg_loss).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

        if rank == 0:
            loss_history.append(avg_loss)
            print(f"Epoch {epoch:03d} | Contrastive Loss: {avg_loss:.4f}")

            # Checkpoint every 50 epochs
            if epoch % 50 == 0:
                model_to_save = encoder.module if is_distributed else encoder
                checkpoint_path = os.path.join(results_dir, f"encoder_checkpoint_epoch{epoch}_{run_id}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                }, checkpoint_path)
                print(f"  💾 Checkpoint saved: {checkpoint_path}")

    if rank == 0:
        print("Pre-training complete.")
        model_to_save = encoder.module if is_distributed else encoder
        final_model_path = os.path.join(results_dir, f"encoder_final_{run_id}.pt")
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'config': config,
        }, final_model_path)
        print(f"✅ Final encoder saved: {final_model_path}")

        loss_df = pd.DataFrame({'epoch': range(1, len(loss_history) + 1), 'loss': loss_history})
        loss_df.to_csv(os.path.join(results_dir, f"loss_history_{run_id}.csv"), index=False)
        print(f"✅ Loss history saved.")

    return encoder, config


def evaluate(encoder, data, device, results_dir, run_id):
    """
    Evaluate the encoder using linear probing.
    """
    print("\n" + "="*60)
    print("STEP 4: LINEAR EVALUATION")
    print("="*60)
    
    LR_EVAL = 0.001
    EPOCHS_EVAL = 100

    print("Generating node embeddings for the whole graph...")
    encoder.eval()
    
    # Ensure encoder is on device
    encoder = encoder.to(device)
    
    # Move data to device for full-batch inference if it fits
    # If OOM, we might need to do mini-batch inference or CPU inference
    data_device = data.to(device)

    with torch.no_grad():
        try:
            embeddings, _ = encoder(data_device.x, data_device.edge_index, data_device.edge_attr)
        except RuntimeError as e:
            print(f"GPU OOM: {e}")
            print("Switching encoder to CPU for inference...")
            encoder_cpu = encoder.cpu()
            data_cpu = data.cpu()
            embeddings, _ = encoder_cpu(data_cpu.x, data_cpu.edge_index, data_cpu.edge_attr)
            embeddings = embeddings.to(device)
            encoder.to(device)

    print(f"\n📊 Embedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        print("⚠️ Sanitizing embeddings (replacing NaN/Inf with 0)...")
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    embeddings = F.normalize(embeddings, dim=1)

    # Save embeddings
    embeddings_path = os.path.join(results_dir, f"embeddings_{run_id}.pt")
    torch.save(embeddings.cpu(), embeddings_path)
    print(f"✅ Embeddings saved: {embeddings_path}")

    # Setup classifier
    X = embeddings.detach()
    y = data.y.to(device)

    labeled_mask = (y != -1)
    train_mask = data.train_mask.to(device) & labeled_mask
    test_mask = data.test_mask.to(device) & labeled_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_class_counts = torch.bincount(y_train)
    print(f"\n📊 Training Class Distribution:")
    print(f"  Class 0 (Licit): {train_class_counts[0].item():,}")
    print(f"  Class 1 (Illicit): {train_class_counts[1].item():,}")

    class_weights = 1.0 / train_class_counts.float()
    class_weights = class_weights / class_weights.sum()
    print(f"  Class Weights: {class_weights.tolist()}")

    classifier = LogisticRegression(X.shape[1], 2).to(device)
    optimizer_eval = torch.optim.Adam(classifier.parameters(), lr=LR_EVAL)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f"\nTraining Linear Classifier on {len(y_train):,} nodes...")

    best_f1 = 0
    eval_history = []
    for epoch in range(1, EPOCHS_EVAL + 1):
        classifier.train()
        optimizer_eval.zero_grad()

        out = classifier(X_train)
        loss = criterion(out, y_train)

        loss.backward()
        optimizer_eval.step()

        if epoch % 20 == 0:
            classifier.eval()
            with torch.no_grad():
                val_out = classifier(X_test)
                val_preds = val_out.argmax(dim=1)
                val_f1 = f1_score(y_test.cpu().numpy(), val_preds.cpu().numpy(), pos_label=1)
                eval_history.append({'epoch': epoch, 'loss': loss.item(), 'val_f1': val_f1})
                if val_f1 > best_f1:
                    best_f1 = val_f1
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

    # Final evaluation
    classifier.eval()
    with torch.no_grad():
        out_test = classifier(X_test)
        probs = F.softmax(out_test, dim=1)
        preds = probs.argmax(dim=1)

        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs[:, 1].cpu().numpy()

        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
        prec = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        acc = (y_true == y_pred).sum() / len(y_true)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "="*50)
        print("✅ FINAL RESULTS (Linear Evaluation)")
        print("="*50)
        print(f"  Accuracy:       {acc:.4f}")
        print(f"  Precision (1):  {prec:.4f}")
        print(f"  Recall (1):     {rec:.4f}")
        print(f"  F1 Score (1):   {f1:.4f}")
        print(f"  F1 Macro:       {f1_macro:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  [[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}]")
        print(f"   [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]")

    # Save classifier
    classifier_path = os.path.join(results_dir, f"classifier_{run_id}.pt")
    torch.save(classifier.state_dict(), classifier_path)
    print(f"\n✅ Classifier saved: {classifier_path}")

    # Save results
    results = {
        "run_id": run_id,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "eval_history": eval_history,
    }

    results_path = os.path.join(results_dir, f"results_{run_id}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved: {results_path}")

    return results


def main():
    args = parse_args()
    
    # Setup DDP
    is_distributed, rank, local_rank, world_size, device = setup_ddp()
    
    # Suppress printing on non-zero ranks
    if rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    # Setup project root
    if args.project_root:
        project_root = args.project_root
    else:
        project_root = os.environ.get("PROJECT_ROOT")
        if not project_root:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
    
    if rank == 0:
        print("="*60)
        print("GraphCL Training for Bitcoin Fraud Detection")
        print("="*60)
        print(f"Project root: {project_root}")
        print(f"Distributed: {is_distributed} (World Size: {world_size})")
    
    # Setup results directory (only rank 0 needs to create it, but all need path)
    results_dir = os.path.join(project_root, "results", "graphCL")
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
    
    # Sync to ensure dir exists
    if is_distributed:
        dist.barrier()
        
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        print(f"Results will be saved to: {results_dir}")
        print(f"Run ID: {run_id}")
    
    # Step 1: Graph contraction (only rank 0)
    if rank == 0:
        if not args.skip_contraction:
            perform_graph_contraction(project_root)
        else:
            print("\n⏭️ Skipping graph contraction (using existing contracted edge list)")
    
    if is_distributed:
        dist.barrier()
    
    # Step 2: Load data (all ranks load data, but keep on CPU)
    data = load_data(project_root)
    
    # Step 3: Train encoder (Distributed)
    encoder, config = train_encoder(data, args, device, results_dir, run_id, is_distributed, rank, local_rank, world_size)
    
    # Step 4: Evaluate (Only Rank 0)
    if rank == 0:
        # Unwrap DDP model for evaluation
        if is_distributed:
            encoder = encoder.module
        results = evaluate(encoder, data, device, results_dir, run_id)
        
        print("\n" + "="*60)
        print("📁 ALL FILES SAVED TO:", results_dir)
        print("="*60)
        print(f"  - config_{run_id}.json")
        print(f"  - encoder_final_{run_id}.pt")
        print(f"  - loss_history_{run_id}.csv")
        print(f"  - embeddings_{run_id}.pt")
        print(f"  - classifier_{run_id}.pt")
        print(f"  - results_{run_id}.json")
        print(f"\nTo download via scp:")
        print(f"  scp -r <user>@<cluster>:{results_dir} ./local_results/")
        print("\n✅ Training complete!")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
