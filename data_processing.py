"""
Data Processing Module for Video Streaming QoE Analysis
Handles data generation, loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingDataGenerator:
    """Generate synthetic streaming session data"""
    
    def __init__(self, n_samples=100000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_sessions(self):
        """Generate streaming session data with realistic patterns"""
        logger.info(f"Generating {self.n_samples} streaming sessions...")
        
        # Basic session info
        session_ids = [f"session_{i:06d}" for i in range(self.n_samples)]
        user_ids = [f"user_{i:05d}" for i in np.random.randint(0, 20000, self.n_samples)]
        
        # Timestamps
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(minutes=np.random.randint(0, 525600)) 
                     for _ in range(self.n_samples)]
        
        # Device and network info
        devices = np.random.choice(['mobile', 'desktop', 'tablet', 'smart_tv'], 
                                   self.n_samples, p=[0.45, 0.30, 0.15, 0.10])
        
        network_types = np.random.choice(['wifi', '4g', '5g', 'ethernet'], 
                                        self.n_samples, p=[0.40, 0.25, 0.20, 0.15])
        
        # Content metadata
        content_types = np.random.choice(['movie', 'series', 'live', 'short'], 
                                        self.n_samples, p=[0.35, 0.40, 0.15, 0.10])
        
        video_duration = np.random.choice([30, 45, 90, 120, 180], self.n_samples)
        
        # Quality metrics with correlations
        base_quality = np.random.beta(8, 2, self.n_samples)
        
        # Video Start Time (VST)
        vst_base = np.random.gamma(2, 0.5, self.n_samples)
        vst = np.where(network_types == 'ethernet', vst_base * 0.7,
              np.where(network_types == '5g', vst_base * 0.8,
              np.where(network_types == 'wifi', vst_base * 1.0, vst_base * 1.5)))
        
        # Buffering events
        buffering_count = np.random.poisson(lam=3 * (1 - base_quality), size=self.n_samples)
        buffering_duration = buffering_count * np.random.exponential(2, self.n_samples)
        
        # Bitrate
        bitrate_mean = np.where(devices == 'smart_tv', 5000,
                       np.where(devices == 'desktop', 3500,
                       np.where(devices == 'tablet', 2500, 1800)))
        
        bitrate = np.random.normal(bitrate_mean, bitrate_mean * 0.2)
        bitrate = np.clip(bitrate, 500, 8000)
        
        # Bitrate switches
        bitrate_switches = np.random.poisson(lam=2 * (1 - base_quality), size=self.n_samples)
        
        # Playback failures
        failure_prob = 0.05 + 0.15 * (1 - base_quality)
        playback_failures = np.random.binomial(1, failure_prob, self.n_samples)
        
        # Watch time and completion
        watch_time_ratio = np.random.beta(5, 2, self.n_samples) * (1 - 0.3 * playback_failures)
        watch_time = video_duration * watch_time_ratio
        completed = (watch_time_ratio > 0.9).astype(int)
        
        # User engagement score
        engagement_score = (
            0.3 * (1 - buffering_count / 10) +
            0.2 * (1 - playback_failures) +
            0.3 * watch_time_ratio +
            0.2 * (vst < 3).astype(int)
        )
        engagement_score = np.clip(engagement_score, 0, 1)
        
        # Churn (target variable)
        churn_prob = (
            0.3 * (buffering_count > 5) +
            0.3 * playback_failures +
            0.2 * (vst > 5) +
            0.2 * (watch_time_ratio < 0.3)
        )
        churn = (np.random.random(self.n_samples) < churn_prob).astype(int)
        
        # Resolution
        resolutions = np.random.choice(['360p', '480p', '720p', '1080p', '4k'], 
                                      self.n_samples, p=[0.10, 0.15, 0.35, 0.30, 0.10])
        
        # ISP
        isps = np.random.choice(['ISP_A', 'ISP_B', 'ISP_C', 'ISP_D'], 
                               self.n_samples, p=[0.35, 0.30, 0.20, 0.15])
        
        # Create DataFrame
        df = pd.DataFrame({
            'session_id': session_ids,
            'user_id': user_ids,
            'timestamp': timestamps,
            'device': devices,
            'network_type': network_types,
            'content_type': content_types,
            'video_duration_min': video_duration,
            'video_start_time_sec': vst,
            'buffering_count': buffering_count,
            'buffering_duration_sec': buffering_duration,
            'avg_bitrate_kbps': bitrate,
            'bitrate_switches': bitrate_switches,
            'playback_failures': playback_failures,
            'watch_time_min': watch_time,
            'completion_rate': watch_time_ratio,
            'completed': completed,
            'engagement_score': engagement_score,
            'resolution': resolutions,
            'isp': isps,
            'churned': churn
        })
        
        logger.info(f"Generated {len(df)} sessions with {df['churned'].sum()} churned users")
        return df


class DataProcessor:
    """Process and clean streaming data"""
    
    def __init__(self):
        pass
    
    def clean_data(self, df):
        """Clean and validate data"""
        logger.info("Cleaning data...")
        
        df = df.drop_duplicates(subset=['session_id'])
        df = df.dropna()
        
        numeric_cols = ['video_duration_min', 'video_start_time_sec', 'buffering_count',
                       'buffering_duration_sec', 'avg_bitrate_kbps', 'bitrate_switches',
                       'watch_time_min', 'completion_rate', 'engagement_score']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def add_derived_features(self, df):
        """Add derived features for modeling"""
        logger.info("Adding derived features...")
        
        df['buffering_ratio'] = df['buffering_duration_sec'] / (df['watch_time_min'] * 60 + 1)
        
        df['quality_score'] = (
            (df['video_start_time_sec'] < 3).astype(int) * 0.3 +
            (df['buffering_count'] < 3).astype(int) * 0.3 +
            (df['playback_failures'] == 0).astype(int) * 0.2 +
            (df['completion_rate'] > 0.8).astype(int) * 0.2
        )
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['is_peak_hour'] = df['hour'].isin([19, 20, 21, 22]).astype(int)
        df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek.isin([5, 6]).astype(int)
        df['bitrate_stability'] = 1 / (1 + df['bitrate_switches'])
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Generate and process streaming data')
    parser.add_argument('--generate', action='store_true', help='Generate new data')
    parser.add_argument('--samples', type=int, default=100000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/raw/streaming_sessions.csv')
    
    args = parser.parse_args()
    
    if args.generate:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        generator = StreamingDataGenerator(n_samples=args.samples)
        df = generator.generate_sessions()
        
        processor = DataProcessor()
        df = processor.clean_data(df)
        df = processor.add_derived_features(df)
        
        df.to_csv(args.output, index=False)
        logger.info(f"Data saved to {args.output}")
        
        print("\nData Summary:")
        print(f"Total sessions: {len(df)}")
        print(f"Churned users: {df['churned'].sum()} ({df['churned'].mean()*100:.2f}%)")
        print(f"\nFeatures: {df.shape[1]}")
        print(f"\nSample data:\n{df.head()}")


if __name__ == "__main__":
    main()