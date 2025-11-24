"""
Quick update script to fetch latest IT-CALA solar power data from ENTSO-E.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz
from entsoe.entsoe import EntsoePandasClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

ZONE_CODE = '10Y1001C--00096J'  # IT-CALA
OUTPUT_FILE = 'data/raw/energy/it_cala_solar.csv'

def get_entsoe_client():
    """Initialize ENTSO-E client with API key from environment."""
    api_key = os.getenv('ENTSOE')
    if not api_key:
        raise ValueError("ENTSO-E API key not found. Please set ENTSOE in your .env file")
    return EntsoePandasClient(api_key=api_key)

def fetch_solar_data(client, zone_code, start_date, end_date, data_type):
    """Fetch solar generation data from ENTSO-E API."""
    try:
        if data_type == 'actual':
            data = client.query_generation(
                zone_code, 
                start=start_date, 
                end=end_date, 
                psr_type='B16'  # Solar
            )
            if isinstance(data, pd.DataFrame):
                if 'Solar' in data.columns:
                    return data['Solar']
                else:
                    generation_cols = [col for col in data.columns if 
                                     'generation' in str(col).lower() or 
                                     'produced' in str(col).lower() or
                                     'inBiddingZone' in str(col)]
                    if generation_cols:
                        return data[generation_cols[0]]
                    return data.iloc[:, 0]
            return data
        elif data_type == 'day_ahead':
            return client.query_day_ahead_prices(zone_code, start=start_date, end=end_date)
        
    except Exception as e:
        print(f"    Error fetching {data_type} data: {e}")
        return None

def update_cala_data():
    """Update IT-CALA solar data."""
    print("=" * 80)
    print("UPDATING IT-CALA SOLAR DATA")
    print("=" * 80)
    
    # Load existing data
    print("\n1. Loading existing data...")
    df_existing = pd.read_csv(OUTPUT_FILE)
    df_existing.columns = ['date', 'actual', 'day-ahead', 'extra', 'installed_capacity_mw']
    df_existing['date'] = pd.to_datetime(df_existing['date'])
    
    # Find last date with actual data
    last_actual = df_existing[df_existing['actual'].notna()]['date'].max()
    print(f"   Last actual data: {last_actual}")
    print(f"   Existing records: {len(df_existing)}")
    
    # Fetch new data from last actual date to now
    print("\n2. Fetching new data from ENTSO-E...")
    client = get_entsoe_client()
    
    start_date = pd.Timestamp(last_actual, tz='UTC') + timedelta(hours=1)
    end_date = pd.Timestamp.now(tz='UTC')
    
    print(f"   Fetching from {start_date} to {end_date}")
    
    # Fetch actual generation
    print("   Fetching actual generation...")
    actual_data = fetch_solar_data(client, ZONE_CODE, start_date, end_date, 'actual')
    
    if actual_data is None or len(actual_data) == 0:
        print("   ✓ No new data available (already up to date)")
        return
    
    # Convert to Rome timezone
    actual_data.index = actual_data.index.tz_convert('Europe/Rome').tz_localize(None)
    
    print(f"   ✓ Fetched {len(actual_data)} new records")
    print(f"   Range: {actual_data.index[0]} to {actual_data.index[-1]}")
    
    # Fetch day-ahead prices
    print("   Fetching day-ahead prices...")
    dayahead_data = fetch_solar_data(client, ZONE_CODE, start_date, end_date, 'day_ahead')
    
    if dayahead_data is not None:
        dayahead_data.index = dayahead_data.index.tz_convert('Europe/Rome').tz_localize(None)
        print(f"   ✓ Fetched {len(dayahead_data)} day-ahead records")
    else:
        print("   ⚠️  No day-ahead data")
        dayahead_data = pd.Series(index=actual_data.index, dtype=float)
    
    # Create new dataframe
    print("\n3. Merging new data...")
    df_new = pd.DataFrame({
        'date': actual_data.index,
        'actual': actual_data.values,
        'day-ahead': dayahead_data.reindex(actual_data.index).values,
        'extra': '',
        'installed_capacity_mw': 749.7
    })
    
    # Combine with existing data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove duplicates (keep last)
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    print(f"   Total records after merge: {len(df_combined)}")
    print(f"   New records added: {len(df_combined) - len(df_existing)}")
    
    # Save updated data
    print("\n4. Saving updated data...")
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✓ Saved to {OUTPUT_FILE}")
    
    # Show summary
    last_actual_new = df_combined[df_combined['actual'].notna()]['date'].max()
    print(f"\n✅ Update complete!")
    print(f"   Last actual data: {last_actual_new}")
    print(f"   Total records: {len(df_combined)}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    update_cala_data()
