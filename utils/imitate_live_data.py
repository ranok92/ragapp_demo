#hosting dataa
import pandas as pd
import pandas_geojson as pdg
import time 
from dashboard_utils import get_geojson_from_df
from argparse import ArgumentParser

def run_args():
    args = ArgumentParser()
    args.add_argument('--sleep-timer', default=30, type=int)
    args.add_argument('--dataset-path', default="../data/dashboard/outage_monitoring_data.csv")
    return args.parse_args()


def main(args):
    full_data_df = pd.read_csv(args.dataset_path)
    for t in range(168):

        anomaly_geo_data = get_geojson_from_df(full_data_df[(full_data_df['timestamp']==t)], t)
        hr_df = full_data_df[full_data_df['timestamp']==t]
        hr_df.to_csv('../data/dashboard/outage_monitoring_data_per_hr.csv')
        pdg.save_geojson(anomaly_geo_data,'../data/geo_data/anomalous.geojson',indent=4)
        time.sleep(args.sleep_timer)

if __name__=='__main__':
    args = run_args()
    main(args)
