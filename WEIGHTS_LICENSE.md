# Model Weights — Provenance and License

The pretrained weights used by `--mode model` (`lastest_EMA.pth`, ~292 MB,
auto-downloaded by `pipeline/download.py`) originate from the **Refusion-HDR**
project (https://github.com/limchaos/Refusion-HDR), an entry in the **AIM 2025
Inverse Tone Mapping (ITM) Challenge**.

## License status

- The Refusion-HDR **source code** is MIT licensed.
- The **pretrained weights carry no explicit license** in the upstream repository,
  and were trained on the AIM 2025 ITM Challenge datasets, whose terms are not
  restated upstream. Challenge datasets are frequently restricted to research /
  non-commercial use.

## What this means for you

- This repository re-hosts the weights on its own GitHub Releases purely for
  convenience of reproduction.
- The MIT license in this repository applies to **this project's own code**, not
  to the third-party weights.
- **Before using the model-mode weights in a commercial or production pipeline,
  verify the AIM 2025 ITM Challenge dataset/model terms yourself.**
- **Linear mode (the default) uses no third-party weights** and is unaffected by
  any of the above.

If you are the upstream author and would like the weights removed or relicensed
here, please open an issue.
