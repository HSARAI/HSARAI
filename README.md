# Deep Hierarchical Framework for Object Classification from Synthetic Aperture Radar Signal


Step 1: generate the 4D signal `spedata` via fft based time-frequency analysis, output `spedata` values

```
python data_process_.py --slc_root ../data/slc_data/ \                 # single look complex data dir
                       --spe_root ../data/spe_data/ \       # 4D TF signal dir
                       --win 0.5                                      # 2D-hamming window size 
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
                        -- train_Natural = 1 # 1 for train Natural model and 0 for train Man-made model
                        --save trained models: modelcoarse1 and modelcoarse2
```

step 4: test model

```
main1_test.py 
                           --device 0
```

