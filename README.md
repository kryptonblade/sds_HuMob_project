# sds_HuMob_project

run:
python3 convert_to_parquet.py --input ~/Downloads/city_A_alldata.csv.gz

you will get a parquet file with the data (faster IO)

python3 compute_for_global_mean.py
![alt text](global_mean_results.png)