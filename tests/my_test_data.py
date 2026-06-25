
from pathlib import Path
#import numpy as np
import torch

from alphagenome_pytorch.utils.sequence import onehot_to_sequence
from alphagenome_encoder_ft.mydata import PlantMPRADataset, create_dataloader

TSV_PATH = "/home/kachu/alphagenome-encoder-ft/metadata/all_log2_activity.tsv"

def basic_load(): #testing basic loading of the data
    print("Testing basic loading of the data")
    
    try:
        dataset = PlantMPRADataset(
            input_tsv= TSV_PATH,  
            split="train", #so this is for the training length
            use_adapters=True,
            sequence_length=160,
            subset_frac=0.1,  # Use only 10% for quick testing
        )
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(dataset)}")
        
        # Try to get first item
        seq, target = dataset[0]
        print(f"✓ First item retrieved")
        print(f"  Sequence shape: {seq.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Target value: {target.tolist()}") #there are four elements per
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_splits(): #for train, val, test splits
    print("Test accurate splits into train/val/test")
    
    for split in ["train", "val", "test"]:
        try:
            dataset = PlantMPRADataset(
                input_tsv=TSV_PATH,
                split=split,
                subset_frac=0.1,
            )
            print(f"{split:5s} split: {len(dataset):5d} samples")
        except Exception as e:
            print(f"✗ {split} split failed: {e}")
            return False
    
    return True
def test_dataloader():
    print("Test batching of the dataloader")
    try: 
        dataset = PlantMPRADataset(
            input_tsv = TSV_PATH,
            split = "train",
            subset_frac = 0.05
        )
        dataloader = create_dataloader(
            dataset, 
            batch_size=16, #are these default
            shuffle=True,
            num_workers=0,
        )

        batch_seq, batch_target = next(iter(dataloader)) #gets the first iteration of the iterator, a single batch
        print(f"✓ DataLoader created successfully")
        print(f"  Batch sequence shape: {batch_seq.shape}")
        print(f"  Batch target shape: {batch_target.shape}")
        print(f"  Expected: [batch_size, seq_len, 4] and [batch_size]")
        
        return True
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_augmentation():
    print("Testing augmentation methods")
    
    try:
        # dataset with augmentation
        dataset_aug = PlantMPRADataset(
            input_tsv=TSV_PATH,
            split="train",
            reverse_complement=True,
            rc_prob=1.0,  # always apply RC for testing
            random_shift=True,
            shift_prob=1.0,  # always apply shift for testing
            max_shift=10,
            subset_frac=0.01,
            seed=42,
        )
        
        # get same item twice (should be different due to augmentation)
        seq1, _ = dataset_aug[0]
        seq2, _ = dataset_aug[0]
        
        are_different = not torch.equal(seq1, seq2)
        print(f"✓ Augmentation {'working' if are_different else 'NOT working'}")
        print(f"  Sequences are {'different' if are_different else 'identical'}")

        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
def test_adapters():
    print("Tests adapter length and prints")
    try:
        # With adapters
        dataset_with = PlantMPRADataset(
            input_tsv=TSV_PATH, 
            split="train",
            use_adapters=True,
            subset_frac=0.01,
        )
        
        # Without adapters
        dataset_without = PlantMPRADataset(
            input_tsv= TSV_PATH, 
            split="train",
            use_adapters=False,
            subset_frac=0.01,
        )
        
        seq_with, _ = dataset_with[0]
        seq_without, _ = dataset_without[0]
        
        # print(f"✓ With adapters: {seq_with} and shape: {seq_with.shape}")
        # print(f"✓ Without adapters: {seq_without} and shape: {seq_without.shape}")
        print(f"✓ With adapters shape: {seq_with.shape}")
        print(f"✓ Without adapters shape: {seq_without.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def inspect_tsv_format():
    print("Tests the TSV")
   
    tsv_path = TSV_PATH  
    
    try:
        import csv
        with open(tsv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Check headers
            headers = reader.fieldnames
            print(f"✓ TSV headers: {headers}")
            
            required = ['Fragment', 'Leaf', 'MG', 'Br',	'RR', 'Chr', 'Unique Barcodes']
            missing = [h for h in required if h not in headers]
            
            if missing:
                print(f"✗ Missing required columns: {missing}")
                return False
            else:
                print(f"✓ All required columns present")
            
            # show first few rows
            print("\nFirst 3 rows:")
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(f"  Row {i+1}:")
                for key in required:
                    print(f"    {key}: {row[key]}")
            
            return True
            
    except FileNotFoundError:
        print(f"✗ File not found: {tsv_path}")
        print("  Please update the path in this script!")
        return False
    except Exception as e:
        print(f"✗ Error reading TSV: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_barcode_threshold():
    print("Testing barcode threshold affects train but not val/test")
    try:
        # train with strict vs relaxed barcode_min
        train_strict = PlantMPRADataset(
            input_tsv=TSV_PATH, split="train", barcode_min=10, barcode_min_eval=10
        )
        train_relaxed = PlantMPRADataset(
            input_tsv=TSV_PATH, split="train", barcode_min=5, barcode_min_eval=10
        )
        assert len(train_strict) < len(train_relaxed), (
            f"Strict barcode_min should reduce train size: {len(train_strict)} vs {len(train_relaxed)}"
        )
        print(f"✓ Train strict({len(train_strict)}) < lenient({len(train_relaxed)})")

        # Vvl size should be unaffected by barcode_min
        val_a = PlantMPRADataset(
            input_tsv=TSV_PATH, split="val", barcode_min=10, barcode_min_eval=10
        )
        val_b = PlantMPRADataset(
            input_tsv=TSV_PATH, split="val", barcode_min=5, barcode_min_eval=10
        )
        assert len(val_a) == len(val_b), (
            f"barcode_min should not affect val size: {len(val_a)} vs {len(val_b)}"
        )
        print(f"✓ Val size unchanged ({len(val_a)}) regardless of barcode_min")

        return True
    except AssertionError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    tests = [
        ("TSV Format", inspect_tsv_format),
        ("Basic Loading", basic_load),
        ("Splits", test_splits),
        ("DataLoader", test_dataloader),
        ("Augmentation", test_augmentation),
        ("Adapters", test_adapters),
        ("Barcode Threshold", test_barcode_threshold)
    ]
    
    results = {}
    for name, test_method in tests:
        results[name] = test_method()
        if not results[name]:
            print(f"Stopped after failed test {name}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed"))

if __name__ == "__main__":
    main()
