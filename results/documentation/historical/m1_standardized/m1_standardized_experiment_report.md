# M1-standardized Experiment Report

M1-standardized uses the same M1 architecture and trains only the RUL regression task. No M2-M5 components are included.

## Scaler Implementation

- Standardization formula: `x_scaled = (x - mean_train) / std_train`
- Scaler fit bearings: `['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing3_1', 'Bearing3_2']`
- Validation bearing: `Bearing2_2`
- Test bearings: `['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']`
- Split disjoint: `True`
- Scaled values finite: `True`
- Max absolute train scaled mean: `7.500140100091812e-07`
- Train scaled std range: `0.999998152256012` to `1.0000017881393433`

## Sequence Counts

- Train: `5764`
- Validation: `152`
- Test: `3164`

## Results

- Best validation loss: `0.009780887783946176`
- Epochs completed: `22`
- MAE: `0.306988443284878`
- RMSE: `0.4012790368197732`
- R2: `-1.0081235508286865`
- Prediction range: `0.017843589186668396` to `0.9696648120880127`
- Prediction bias: `{'mean_true_rul': 0.4880608738779827, 'mean_predicted_rul': 0.6803827024542982, 'mean_prediction_error_pred_minus_true': 0.19232182857631552, 'prediction_std': 0.35423187371728576}`

## Per-Condition Metrics

condition,sequence_count,mae,rmse,r2,prediction_min,prediction_max,predictions_outside_0_1

35Hz12kN,305,0.10282459158572506,0.14353681241534164,0.7151602995141819,0.02752632647752762,0.9115534424781799,0

37.5Hz11kN,887,0.1562410123925026,0.20868325015340167,0.45344778024002597,0.017843589186668396,0.9262647032737732,0

40Hz10kN,1972,0.4063712759267536,0.48536968571922934,-1.8914436548435964,0.01992921717464924,0.9696648120880127,0



## Per-Bearing Metrics

bearing_id,sequence_count,mae,rmse,r2,prediction_min,prediction_max,predictions_outside_0_1,condition

Bearing1_3,149,0.1512417543504462,0.19076497743930218,0.5151305046572916,0.04902253672480583,0.8198480606079102,0,35Hz12kN

Bearing1_4,113,0.05635870522530997,0.07611875079487629,0.9202717160627312,0.17177169024944305,0.9115534424781799,0,35Hz12kN

Bearing1_5,43,0.057161519650456516,0.06935409059765381,0.9187611487806177,0.02752632647752762,0.7353447675704956,0,35Hz12kN

Bearing2_3,524,0.1712893789810953,0.23072850182376384,0.3415148119744422,0.017843589186668396,0.9262647032737732,0,37.5Hz11kN

Bearing2_4,33,0.20618119632655924,0.2659809397090728,-0.31165946246463494,0.03429584950208664,0.8118689060211182,0,37.5Hz11kN

Bearing2_5,330,0.12735201190084672,0.1595220736674722,0.6796443942178677,0.017972897738218307,0.9108583927154541,0,37.5Hz11kN

Bearing3_3,362,0.39723620372592827,0.4823294178870381,-1.9164757414328393,0.031284037977457047,0.9696648120880127,0,40Hz10kN

Bearing3_4,1505,0.411700427186663,0.4906474412598604,-1.9196135521667261,0.05788788944482803,0.9608613848686218,0,40Hz10kN

Bearing3_5,105,0.3614810234556595,0.41460013524389827,-1.3892291524144276,0.01992921717464924,0.6293061375617981,0,40Hz10kN



## Comparison With M1-raw Reference

- M1-raw MAE/RMSE/R2: `{'mae': 0.30699, 'rmse': 0.40128, 'r2': -1.00812}`
- M1-standardized MAE/RMSE/R2: `{'mae': 0.306988443284878, 'rmse': 0.4012790368197732, 'r2': -1.0081235508286865}`
- Delta standardized minus raw: `{'mae': -1.556715121975305e-06, 'rmse': -9.631802267984213e-07, 'r2': -3.550828686638141e-06}`
- Materially helped: `False`

## Warnings

- Bearing3_4: missing numeric CSV indices [1105]; existing files are kept in numeric order.
- CUDA is not available to PyTorch; M1-standardized trained on CPU.
- Existing M1 trainer already used fold-training-only StandardScaler; this run is an explicitly named standardized rerun.
