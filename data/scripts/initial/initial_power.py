"""
THIS SCRIPT FETCHES DATA FROM 2015. DO NOT RUN OFTEN, AS IT IS HEAVY ON THE API.
RUN SCRIPT IN UPDATE_DATA INSTEAD FOR REGULAR UPDATES.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from entsoe.entsoe import EntsoePandasClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Italy's bidding zones with their ENTSO-E codes (only active zones)
ITALY_BIDDING_ZONES = {
    'IT-NORD': '10Y1001A1001A73I',  # Italy North (Nord)
    'IT-CNOR': '10Y1001A1001A70O',  # Italy Centre-North (Centro Nord)
    'IT-CSUD': '10Y1001A1001A71M',  # Italy Centre-South (Centro Sud)
    'IT-SUD': '10Y1001A1001A788',   # Italy South (Sud) - only after 2021
    'IT-CALA': '10Y1001C--00096J',  # Italy Calabria - only after 2021
    'IT-SICI': '10Y1001A1001A74G',  # Italy Sicily (Sicilia)
    'IT-SARD': '10Y1001A1001A75E',  # Italy Sardinia (Sardegna)
}

# Special date restrictions for certain zones
ZONE_DATE_RESTRICTIONS = {
    'IT-SUD': 2021,   # South zone data only from 2021 onwards (zone was split in 2021)
    'IT-CALA': 2021,  # Calabria zone data only from 2021 onwards (new zone created in 2021)
}

def get_entsoe_client():
    """Initialize ENTSO-E client with API key from environment."""
    api_key = os.getenv('ENTSOE')
    if not api_key:
        raise ValueError("ENTSOE API key not found in environment variables")
    return EntsoePandasClient(api_key=api_key)

def get_solar_forecast_data(client, zone_code, zone_name, start_date, end_date, forecast_type='day_ahead'):
    """
    Retrieve solar forecast data for a specific bidding zone.
    
    Args:
        client: EntsoePandasClient instance
        zone_code: ENTSO-E zone code
        zone_name: Human-readable zone name
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        forecast_type: 'day_ahead', 'intraday', or 'actual'
    
    Returns:
        pandas.DataFrame: Solar forecast data
    """
    try:
        print(f"Fetching {forecast_type} solar data for {zone_name}...")
        
        if forecast_type == 'day_ahead':
            # Day-ahead wind and solar forecast (returns both wind and solar)
            data = client.query_wind_and_solar_forecast(
                country_code=zone_code,
                start=start_date,
                end=end_date
            )
        elif forecast_type == 'intraday':
            # Intraday wind and solar forecast (returns both wind and solar)
            data = client.query_intraday_wind_and_solar_forecast(
                country_code=zone_code,
                start=start_date,
                end=end_date
            )
        elif forecast_type == 'actual':
            # Actual generation per production type - specifically for solar
            # First try with default parameters to see what we get
            data = client.query_generation(
                country_code=zone_code,
                start=start_date,
                end=end_date,
                psr_type='B16'  # Solar photovoltaic
            )
        
        if data is not None and not data.empty:
            # Convert to DataFrame if it's a Series
            if isinstance(data, pd.Series):
                data_df = data.to_frame('value')
            else:
                data_df = pd.DataFrame(data)
            
            # Filter for solar data if multiple columns exist
            if 'Solar' in data_df.columns:
                data_df = data_df[['Solar']].copy()
                data_df.rename(columns={'Solar': 'solar_forecast_mw'}, inplace=True)
            elif forecast_type == 'actual' and len(data_df.columns) > 1:
                # For actual generation, we might get multiple columns (generation vs consumption)
                # We ONLY want generation data, not consumption
                print(f"  Multiple columns found: {list(data_df.columns)}")
                col_sums = data_df.sum()
                print(f"  Column sums: {dict(col_sums)}")
                
                # Look for generation-related column names first
                generation_cols = [col for col in data_df.columns if 
                                 'generation' in str(col).lower() or 
                                 'produced' in str(col).lower() or
                                 'inBiddingZone' in str(col)]
                
                if generation_cols:
                    selected_col = generation_cols[0]
                    print(f"  Found generation column: {selected_col}")
                else:
                    # Fallback: assume the column with positive values is generation
                    # Solar consumption is typically zero or very small
                    max_col = col_sums.idxmax()
                    selected_col = max_col
                    print(f"  No clear generation column found, using column with highest values: {selected_col}")
                
                data_df = data_df[[selected_col]].copy()
                data_df.rename(columns={selected_col: 'solar_forecast_mw'}, inplace=True)
            elif len(data_df.columns) == 1:
                # Assume single column is solar data
                data_df.rename(columns={data_df.columns[0]: 'solar_forecast_mw'}, inplace=True)
            else:
                print(f"  Unexpected data structure: {data_df.columns}")
                # Take the first column as a fallback
                if len(data_df.columns) > 0:
                    data_df = data_df.iloc[:, [0]].copy()
                    data_df.rename(columns={data_df.columns[0]: 'solar_forecast_mw'}, inplace=True)
            
            # Add metadata columns
            data_df['zone'] = zone_name
            data_df['zone_code'] = zone_code
            data_df['forecast_type'] = forecast_type
            data_df['retrieved_at'] = datetime.now()
            
            print(f"Retrieved {len(data_df)} records for {zone_name} ({forecast_type})")
            return data_df
        else:
            print(f"No data available for {zone_name} ({forecast_type})")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error retrieving {forecast_type} data for {zone_name}: {str(e)}")
        return pd.DataFrame()

def save_zone_data_combined(zone_data_dict, zone_name, data_dir='data/energy'):
    """
    Save combined data for a specific zone to a CSV file with format: date, actual, day-ahead, intraday
    If file exists, merge new data with existing data.
    
    Args:
        zone_data_dict: Dictionary with keys 'day_ahead', 'intraday', 'actual' containing DataFrames
        zone_name: Name of the zone
        data_dir: Directory to save the data
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create filename
    filename = f"{zone_name.lower().replace('-', '_')}_solar.csv"
    filepath = os.path.join(data_dir, filename)
    
    # Check if file exists and load existing data
    existing_df = pd.DataFrame()
    if os.path.exists(filepath):
        print(f"Loading existing data from {filepath}")
        existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Existing data: {len(existing_df)} records from {existing_df.index.min()} to {existing_df.index.max()}")
    
    # Create new combined DataFrame
    combined_df = pd.DataFrame()
    
    # Get all unique timestamps from all data types AND existing data
    all_timestamps = set()
    if not existing_df.empty:
        all_timestamps.update(existing_df.index)
    
    for data_type, df in zone_data_dict.items():
        if not df.empty:
            all_timestamps.update(df.index)
    
    if not all_timestamps:
        print(f"No data to save for {zone_name}")
        return
    
    # Create DataFrame with all timestamps
    combined_df = pd.DataFrame(index=sorted(all_timestamps))
    combined_df.index.name = 'date'
    
    # Initialize columns
    combined_df['actual'] = None
    combined_df['day-ahead'] = None
    combined_df['intraday'] = None
    
    # Fill in existing data first
    if not existing_df.empty:
        for col in ['actual', 'day-ahead', 'intraday']:
            if col in existing_df.columns:
                combined_df.loc[existing_df.index, col] = existing_df[col]
    
    # Fill in new data (will overwrite existing where overlapping)
    for data_type, df in zone_data_dict.items():
        if not df.empty:
            if data_type == 'day_ahead':
                combined_df.loc[df.index, 'day-ahead'] = df['solar_forecast_mw']
            elif data_type == 'intraday':
                combined_df.loc[df.index, 'intraday'] = df['solar_forecast_mw']
            elif data_type == 'actual':
                combined_df.loc[df.index, 'actual'] = df['solar_forecast_mw']
    
    # Save to CSV
    combined_df.to_csv(filepath, index=True)
    print(f"Saved {len(combined_df)} records to {filepath}")
    
    # Count data by type
    actual_count = combined_df['actual'].notna().sum()
    day_ahead_count = combined_df['day-ahead'].notna().sum()
    intraday_count = combined_df['intraday'].notna().sum()
    
    print(f"  - Actual records: {actual_count}")
    print(f"  - Day-ahead records: {day_ahead_count}")
    print(f"  - Intraday records: {intraday_count}")
    
    # Show what was added
    if not existing_df.empty:
        new_actual = combined_df.loc[~combined_df.index.isin(existing_df.index), 'actual'].notna().sum()
        print(f"  - NEW actual records added: {new_actual}")

def retrieve_italy_solar_data(start_year=2015, forecast_types=['day_ahead', 'intraday', 'actual']):
    """
    Retrieve solar production forecasts for all Italian bidding zones.
    
    Args:
        start_year: Year to start data retrieval from (default: 2015)
        forecast_types: List of forecast types to retrieve ['day_ahead', 'intraday', 'actual']
    """
    print("Starting Italy solar data retrieval...")
    print(f"Bidding zones to process: {list(ITALY_BIDDING_ZONES.keys())}")
    print(f"Forecast types: {forecast_types}")
    
    # Initialize client
    client = get_entsoe_client()
    
    # Set date range - from start_year to current date, process in yearly chunks
    current_date = pd.Timestamp.now(tz='Europe/Rome')
    
    # Process each bidding zone
    for zone_name, zone_code in ITALY_BIDDING_ZONES.items():
        print(f"\n--- Processing {zone_name} ({zone_code}) ---")
        
        # Check if this zone has date restrictions
        zone_start_year = start_year
        if zone_name in ZONE_DATE_RESTRICTIONS:
            restricted_year = ZONE_DATE_RESTRICTIONS[zone_name]
            if zone_start_year < restricted_year:
                zone_start_year = restricted_year
                print(f"  Note: {zone_name} data only available from {restricted_year} onwards")
        
        # Dictionary to store all data types for this zone
        zone_data = {}
        
        for forecast_type in forecast_types:
            print(f"Processing {forecast_type} data...")
            all_data = []
            
            # Process year by year to avoid large requests
            for year in range(zone_start_year, current_date.year + 1):
                start_date = pd.Timestamp(f'{year}-01-01', tz='Europe/Rome')
                if year == current_date.year:
                    end_date = current_date
                else:
                    end_date = pd.Timestamp(f'{year}-12-31', tz='Europe/Rome')
                
                print(f"  Fetching data for {year}...")
                
                # Get data for this year
                yearly_data = get_solar_forecast_data(
                    client, zone_code, zone_name, start_date, end_date, forecast_type
                )
                
                if not yearly_data.empty:
                    all_data.append(yearly_data)
                
                # Small delay to avoid hitting API rate limits
                import time
                time.sleep(1)
            
            # Combine all yearly data for this forecast type
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=False)
                combined_data = combined_data.sort_index()
                print(f"Total records for {zone_name} ({forecast_type}): {len(combined_data)}")
                
                # Store in zone_data dictionary
                zone_data[forecast_type] = combined_data
            else:
                print(f"No data retrieved for {zone_name} ({forecast_type})")
                zone_data[forecast_type] = pd.DataFrame()
        
        # Save combined data for this zone
        save_zone_data_combined(zone_data, zone_name)
    
    print("\nItaly solar data retrieval completed!")

def get_available_data_summary(data_dir='data/energy'):
    """Print a summary of available data files."""
    print("\n--- Available Solar Data Files ---")
    
    if not os.path.exists(data_dir):
        print("No data directory found.")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith('_solar.csv')]
    
    if not files:
        print("No solar data files found.")
        return
    
    for file in sorted(files):
        filepath = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(filepath, index_col=0)
            day_ahead_count = df['day-ahead'].notna().sum() if 'day-ahead' in df.columns else 0
            intraday_count = df['intraday'].notna().sum() if 'intraday' in df.columns else 0
            actual_count = df['actual'].notna().sum() if 'actual' in df.columns else 0
            
            print(f"{file}: {len(df)} total records")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Day-ahead: {day_ahead_count} records")
            print(f"  Intraday: {intraday_count} records") 
            print(f"  Actual: {actual_count} records")
            print()
        except Exception as e:
            print(f"{file}: Error reading file - {e}")

if __name__ == "__main__":
    # Retrieve solar data for Sicily only, focusing on missing actual data from 2021 onwards
    # You can modify start_year and forecast_types as needed
    retrieve_italy_solar_data(start_year=2021, forecast_types=['actual'])
    
    # Show summary of retrieved data
    get_available_data_summary()