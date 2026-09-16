import pandas as pd
import numpy as np
import os
from typing import Dict, Any

class ClothMockAdapter:
    """
    Adapter for integrating with CLoTH simulator.
    In the absence of a live CLoTH executable, this generates mock logs 
    that mimic what CLoTH (or similar discrete-event PCN simulators) would produce,
    and provides a parser to convert those logs into our state representation.
    """
    def __init__(self, log_dir: str = 'cloth_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def generate_mock_logs(self, num_records: int = 1000):
        """Generates a mock CSV representing CLoTH channel states over time."""
        # Columns: timestamp, u, v, capacity, balance, is_failed, failed_amount
        data = []
        for i in range(num_records):
            u = np.random.randint(0, 100)
            v = np.random.randint(0, 100)
            while u == v:
                v = np.random.randint(0, 100)
                
            capacity = np.random.randint(1000, 10000)
            balance = np.random.randint(0, capacity)
            is_failed = np.random.choice([0, 1], p=[0.9, 0.1])
            failed_amount = np.random.randint(balance + 1, capacity + 1000) if is_failed else 0
            
            data.append({
                'timestamp': i * 100, # ms
                'u': u,
                'v': v,
                'capacity': capacity,
                'balance': balance,
                'is_failed': is_failed,
                'failed_amount': failed_amount
            })
            
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.log_dir, 'cloth_channel_states.csv'), index=False)
        return df
        
    def parse_logs(self) -> pd.DataFrame:
        """Parses the exported CLoTH logs."""
        file_path = os.path.join(self.log_dir, 'cloth_channel_states.csv')
        if not os.path.exists(file_path):
            return self.generate_mock_logs()
        return pd.read_csv(file_path)
        
    def replay_actions(self, actions: list):
        """
        Simulates the 'thin adapter' mentioned in the thesis prompt.
        Takes trained agent actions and applies them to a fresh CLoTH run.
        (Since this is an offline mock, we just log that the actions were processed).
        """
        print(f"Replayed {len(actions)} actions against CLoTH mock adapter.")
        # In a real closed-loop, we'd write an actions.csv and invoke the C++ binary here.
        return True
