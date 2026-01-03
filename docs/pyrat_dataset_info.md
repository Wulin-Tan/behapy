# PyRAT Dataset Information Report

## 1. File Summary
- **Total Files**: 33
- **Data Files**: 29
- **Total Size**: 649.76 MB

## 2. Data Files
| File Name | Relative Path | Size (KB) | Extension |
| --- | --- | --- | --- |
| OFT_5_zoom.csv | OFT_5_zoom.csv | 12884.7 | .csv |
| PlexonTracking.csv | PlexonTracking.csv | 2006.1 | .csv |
| R1D1.csv | R1D1.csv | 19606.3 | .csv |
| R1D2.csv | R1D2.csv | 19682.2 | .csv |
| R1D3.csv | R1D3.csv | 19610.4 | .csv |
| R1_obj.csv | R1_obj.csv | 1953.8 | .csv |
| R2D1.csv | R2D1.csv | 20343.0 | .csv |
| R2D2.csv | R2D2.csv | 19629.2 | .csv |
| R2D3.csv | R2D3.csv | 19611.3 | .csv |
| R2_obj.csv | R2_obj.csv | 1968.6 | .csv |
| R3D1.csv | R3D1.csv | 19620.6 | .csv |
| R3D2.csv | R3D2.csv | 19590.6 | .csv |
| R3D3.csv | R3D3.csv | 19627.0 | .csv |
| R3_obj.csv | R3_obj.csv | 1976.5 | .csv |
| R4D1.csv | R4D1.csv | 19585.8 | .csv |
| R4D2.csv | R4D2.csv | 19634.7 | .csv |
| R4D3.csv | R4D3.csv | 19626.6 | .csv |
| R4_obj.csv | R4_obj.csv | 1957.1 | .csv |
| R5D1.csv | R5D1.csv | 19544.8 | .csv |
| R5D2.csv | R5D2.csv | 19590.1 | .csv |
| R5D3.csv | R5D3.csv | 19636.4 | .csv |
| R5_obj.csv | R5_obj.csv | 1960.1 | .csv |
| R6D1.csv | R6D1.csv | 19552.6 | .csv |
| R6D2.csv | R6D2.csv | 19618.8 | .csv |
| R6D3.csv | R6D3.csv | 19564.0 | .csv |
| R6_obj.csv | R6_obj.csv | 1963.9 | .csv |
| electrophysiology_df_example.csv | electrophysiology_df_example.csv | 0.7 | .csv |
| report_example.csv | report_example.csv | 1.3 | .csv |
| t-SNE.csv | t-SNE.csv | 19590.0 | .csv |

## 3. First Data File Inspection

#### CSV Inspection: OFT_5_zoom.csv
- **Header Type**: DLC Single Animal (3 levels)
- **Detected as DLC**: True
- **Shape**: (18293, 39) (showing partial rows)
- **Bodyparts**: nose, head_center, l_ear, r_ear, neck, l_side, b_center, r_side, l_hip, r_hip...
- **First few rows**:
```text
scorer    DLC_resnet50_zoom_oftJan14shuffle1_255000                                                                                                                                                                                                                                                                                                                                                                                                                                                         
bodyparts                                      nose                        head_center                              l_ear                              r_ear                               neck                             l_side                           b_center                             r_side                            l_hip                              r_hip                          tail_base                        tail_center                           tail_tip                       
coords                                            x           y likelihood           x           y likelihood           x           y likelihood           x           y likelihood           x           y likelihood           x           y likelihood           x           y likelihood           x         y likelihood           x           y likelihood           x           y likelihood           x           y likelihood           x           y likelihood           x           y likelihood
0                                        170.208832   24.555975   0.056187  354.116974  150.420013   0.014198   69.746918  245.306549   0.025037  156.705429   -2.030591   0.048294  393.281342  269.437103   0.082880  361.287567  269.011902   0.116419  160.342636   -6.299396   0.112315  159.156494  0.097225   0.382141  414.607452  282.956879   0.716087  151.950760    3.270913   0.536861  277.030579  250.478287   0.999652  198.297440  233.358124   0.968683  242.052521  197.212646   0.999912
1                                        170.202850   24.559715   0.055332  354.024078  150.266006   0.014230   69.756912  245.300644   0.024768  156.706741   -2.031288   0.048303  393.281036  269.434937   0.082911  361.295105  269.014587   0.116006  160.342331   -6.295910   0.111948  159.157791  0.099590   0.382338  414.609314  282.955048   0.715829  151.951248    3.270070   0.536332  277.037628  250.472626   0.999652  198.299072  233.403030   0.968490  241.885269  197.168030   0.999923
2                                        438.570557  221.778625   0.181965  434.762909  222.550568   0.202731  428.683136  215.271133   0.090364  432.658051  228.977158   0.171474  426.235199  224.974350   0.087174  378.617798  260.656616   0.188028  347.934753  286.738098   0.037651  156.661560 -0.912349   0.169121  396.705078  285.146973   0.532403  330.270111  310.134399   0.159313  269.877502  256.858612   0.999242  203.435974  231.495514   0.950333  241.268921  197.234055   0.999946
```


## 4. Documentation Found
No documentation files found.

## 5. Directory Tree (Max Depth 3)
```text
reference/PyRAT_dataset
├── electrophysiology_data.ns2
├── electrophysiology_df_example.csv
├── OFT_5_zoom.csv
├── OFT5_VIDEO2TSNE.mp4
├── PlexonTracking.csv
├── R1_obj.csv
├── R10D1_VIDEO2TSNE.AVI
├── R1D1.csv
├── R1D2.csv
├── R1D3.csv
├── R2_obj.csv
├── R2D1.csv
├── R2D2.csv
├── R2D3.csv
├── R3_obj.csv
├── R3D1.csv
├── R3D2.csv
├── R3D3.csv
├── R4_obj.csv
├── R4D1.csv
├── R4D2.csv
├── R4D3.csv
├── R5_obj.csv
├── R5D1.csv
├── R5D2.csv
├── R5D3.csv
├── R6_obj.csv
├── R6D1.csv
├── R6D2.csv
├── R6D3.csv
├── report_example.csv
├── spikes_data.mat
└── t-SNE.csv

0 directories, 33 files

```

## 6. Manual Notes
- **Animal Type**: Unknown (Needs verification)
- **Behavior**: Unknown
- **Tracking Software**: Detected from headers (see Section 3)
- **FPS**: Unknown
- **Recording Duration**: Unknown
- **Known Issues**: None noted yet