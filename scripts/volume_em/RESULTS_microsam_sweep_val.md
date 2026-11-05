# Volume EM mitochondria: vit_b_em_organelles voxel-size sweep on the validation blocks

vit_b_em_organelles on the `val` blocks of `split-mito_vol_em_aniso2lvl_final.json` (4007, 4009), which it never saw.
Each block is resampled to the voxel size, segmented, and resized back to the native
25/5/5 nm grid, where it is scored against the untouched ground truth. `min_size` is scaled
to the same physical volume as synapse-net's 1,000 native voxels at every voxel size.

Instance metrics are matching at an IoU of 0.5; `msa` averages over the thresholds 0.5 to 0.95.
Metrics are averaged over the blocks, `n_pred` and `n_true` are summed.

The best setting of each approach is selected here, by F1 with ties broken by msa, and then run
once on the test split. Nothing is selected on the test blocks.

## A: z at 25 nm, xy resampled (xy stack)

**Selected: `microsam-z25-xy5-xy`** — raw contrast, xy mode, F1 0.0891, msa 0.0241.

| voxel_nm | contrast | mode | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|----------|----------|------|--------|-----------|--------|--------|--------|--------|--------|--------|
| 3.75     | raw      | xy   | 0.0572 | 0.0331    | 0.2099 | 0.0124 | 0.0632 | 0.3038 | 426    | 65     |
| 5        | raw      | xy   | 0.0891 | 0.0528    | 0.2867 | 0.0241 | 0.0946 | 0.4558 | 346    | 65     |
| 7.5      | raw      | xy   | 0.0425 | 0.0262    | 0.1120 | 0.0075 | 0.0765 | 0.3136 | 278    | 65     |
| 8        | raw      | xy   | 0.0394 | 0.0253    | 0.0898 | 0.0081 | 0.0651 | 0.2359 | 231    | 65     |
| 10       | raw      | xy   | 0.0317 | 0.0234    | 0.0492 | 0.0022 | 0.0380 | 0.1185 | 146    | 65     |
| 12.5     | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0218 | 0.0469 | 92     | 65     |
| 15       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0054 | 0.0082 | 64     | 65     |
| 20       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0022 | 0.0025 | 47     | 65     |
| 25       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0048 | 0.0065 | 35     | 65     |
| 30       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0042 | 0.0027 | 33     | 65     |

## B: isotropic

**No setting works:** the best is `microsam-iso10-xy` at F1 0.0172, below 0.05, so none is selected.

| voxel_nm | contrast | mode | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|----------|----------|------|--------|-----------|--------|--------|--------|--------|--------|--------|
| 10       | raw      | xy   | 0.0172 | 0.0106    | 0.0449 | 0.0009 | 0.0231 | 0.1229 | 275    | 65     |
| 12.5     | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0158 | 0.0537 | 152    | 65     |
| 15       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0081 | 0.0144 | 94     | 65     |
| 20       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0025 | 0.0023 | 53     | 65     |
| 25       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0048 | 0.0065 | 35     | 65     |
| 30       | raw      | xy   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0008 | 0.0017 | 28     | 65     |

## C: single settings (voxel size z/y/x in nm)

**No setting works:** the best is `microsam-z30-xy8-xy` at F1 0.0285, below 0.05, so none is selected.

| voxel_nm | contrast | mode | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|----------|----------|------|--------|-----------|--------|--------|--------|--------|--------|--------|
| 30/8/8   | raw      | xy   | 0.0285 | 0.0184    | 0.0627 | 0.0065 | 0.0681 | 0.2468 | 218    | 65     |

## Per block

| setting                | dataset | f1     | precision | recall | msa    | sbd    | dice   | n_pred | n_true |
|------------------------|---------|--------|-----------|--------|--------|--------|--------|--------|--------|
| microsam-iso10-xy      | 4007    | 0.0205 | 0.0127    | 0.0541 | 0.0010 | 0.0192 | 0.0784 | 158    | 37     |
| microsam-iso10-xy      | 4009    | 0.0138 | 0.0085    | 0.0357 | 0.0007 | 0.0269 | 0.1674 | 117    | 28     |
| microsam-iso12.5-xy    | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0127 | 0.0402 | 87     | 37     |
| microsam-iso12.5-xy    | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0190 | 0.0671 | 65     | 28     |
| microsam-iso15-xy      | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0067 | 0.0098 | 67     | 37     |
| microsam-iso15-xy      | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0095 | 0.0190 | 27     | 28     |
| microsam-iso20-xy      | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0049 | 0.0046 | 36     | 37     |
| microsam-iso20-xy      | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 17     | 28     |
| microsam-iso25-xy      | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0046 | 0.0026 | 23     | 37     |
| microsam-iso25-xy      | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0051 | 0.0103 | 12     | 28     |
| microsam-iso30-xy      | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0016 | 0.0033 | 24     | 37     |
| microsam-iso30-xy      | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4      | 28     |
| microsam-z25-xy10-xy   | 4007    | 0.0157 | 0.0111    | 0.0270 | 0.0008 | 0.0252 | 0.0640 | 90     | 37     |
| microsam-z25-xy10-xy   | 4009    | 0.0476 | 0.0357    | 0.0714 | 0.0036 | 0.0509 | 0.1729 | 56     | 28     |
| microsam-z25-xy12.5-xy | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0114 | 0.0245 | 58     | 37     |
| microsam-z25-xy12.5-xy | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0322 | 0.0692 | 34     | 28     |
| microsam-z25-xy15-xy   | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0050 | 0.0055 | 47     | 37     |
| microsam-z25-xy15-xy   | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0059 | 0.0109 | 17     | 28     |
| microsam-z25-xy20-xy   | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0044 | 0.0050 | 36     | 37     |
| microsam-z25-xy20-xy   | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11     | 28     |
| microsam-z25-xy3.75-xy | 4007    | 0.0070 | 0.0040    | 0.0270 | 0.0025 | 0.0356 | 0.1448 | 249    | 37     |
| microsam-z25-xy3.75-xy | 4009    | 0.1073 | 0.0621    | 0.3929 | 0.0223 | 0.0909 | 0.4628 | 177    | 28     |
| microsam-z25-xy30-xy   | 4007    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0084 | 0.0055 | 29     | 37     |
| microsam-z25-xy30-xy   | 4009    | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4      | 28     |
| microsam-z25-xy5-xy    | 4007    | 0.0717 | 0.0430    | 0.2162 | 0.0175 | 0.0799 | 0.3949 | 186    | 37     |
| microsam-z25-xy5-xy    | 4009    | 0.1064 | 0.0625    | 0.3571 | 0.0307 | 0.1093 | 0.5167 | 160    | 28     |
| microsam-z25-xy7.5-xy  | 4007    | 0.0306 | 0.0189    | 0.0811 | 0.0067 | 0.0573 | 0.2431 | 159    | 37     |
| microsam-z25-xy7.5-xy  | 4009    | 0.0544 | 0.0336    | 0.1429 | 0.0083 | 0.0957 | 0.3842 | 119    | 28     |
| microsam-z25-xy8-xy    | 4007    | 0.0455 | 0.0288    | 0.1081 | 0.0087 | 0.0569 | 0.1961 | 139    | 37     |
| microsam-z25-xy8-xy    | 4009    | 0.0333 | 0.0217    | 0.0714 | 0.0076 | 0.0732 | 0.2756 | 92     | 28     |
| microsam-z30-xy8-xy    | 4007    | 0.0261 | 0.0172    | 0.0541 | 0.0059 | 0.0677 | 0.2138 | 116    | 37     |
| microsam-z30-xy8-xy    | 4009    | 0.0308 | 0.0196    | 0.0714 | 0.0070 | 0.0686 | 0.2798 | 102    | 28     |
