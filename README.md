# Deep Hierarchical Framework for Object Classification from Synthetic Aperture Radar Signal


Step 1: generate the 4D signal `spedata` via fft based time-frequency analysis, output `spedata` values

```
python data_process_.py --slc_root ../data/slc_data/ \                 # single look complex data dir
                       --spe4D_root ../data/slc_spe4D_fft_12/ \       # 4D TF signal dir
                       --win 0.5                                      # hamming window size 
```

Step 2: train main model

```
main1.py --data_file .
                    --data_root ../data/slc_spe4D_fft_12/ ../data/slc_spe4D_fft_12/ \
                    --save_model_path ../model \
                    --pretrained_model ../model/slc_spexy_cae_3.pth \
                    --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                    --device 0
```

Step 3: train coarse branches

```
main1_train_CoarseBranhes.py 
                        --save_dir ../data/slc_spe4D_fft_12_spe3D/ \            # spe3D features
                        --spe_dir ../data/slc_spe4D_fft_12/ \
                        --pretrained_model ../model.pth \
```

step 4: test model

```
main1_test.py 
                           --device 0
```

