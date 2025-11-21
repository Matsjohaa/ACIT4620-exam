"""
Test script to see how many training sequences would be filtered with nighttime optimization.
This shows the impact before actually retraining the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import load_all_zones, prepare_sequences_with_future

def test_nighttime_filtering():
    """Compare sequence counts with and without nighttime filtering."""
    
    print("=" * 80)
    print("TESTING NIGHTTIME FILTERING IMPACT ON TRAINING DATA")
    print("=" * 80)
    
    # Load training data for IT-NORD
    train_data = load_all_zones(split="train")
    zones_to_test = ["IT-NORD", "IT-CNOR", "IT-SUD"]
    
    for zone in zones_to_test:
        if zone not in train_data:
            print(f"\n⚠️  {zone} not found in training data")
            continue
            
        df = train_data[zone]
        
        print(f"\n" + "=" * 80)
        print(f"ZONE: {zone}")
        print("=" * 80)
        
        # Test WITHOUT filtering
        print("\n📊 WITHOUT nighttime filtering (current approach):")
        print("-" * 80)
        X_enc_old, X_dec_old, y_old = prepare_sequences_with_future(
            df,
            sequence_length=168,
            forecast_horizon=336,
            filter_nighttime=False
        )
        sequences_old = len(X_enc_old)
        
        # Test WITH filtering
        print("\n📊 WITH nighttime filtering (optimized approach):")
        print("-" * 80)
        X_enc_new, X_dec_new, y_new = prepare_sequences_with_future(
            df,
            sequence_length=168,
            forecast_horizon=336,
            filter_nighttime=True
        )
        sequences_new = len(X_enc_new)
        
        # Compare
        print(f"\n📈 COMPARISON FOR {zone}:")
        print("-" * 80)
        print(f"Without filtering: {sequences_old:,} sequences")
        print(f"With filtering:    {sequences_new:,} sequences")
        filtered = sequences_old - sequences_new
        print(f"Filtered out:      {filtered:,} sequences ({100*filtered/sequences_old:.1f}%)")
        print(f"Training speedup:  ~{sequences_old/sequences_new:.2f}x faster")
        print(f"\n✅ Model will focus on {sequences_new:,} meaningful daytime sequences")
        print(f"✅ Saves ~{100*filtered/sequences_old:.0f}% of training time")
        print(f"✅ No capacity wasted learning trivial night→zero patterns")

if __name__ == "__main__":
    test_nighttime_filtering()
