# Third-party notices

## DDBC and MDLM

The discrete diffusion implementation follows **Discrete Diffusion for Bundle Construction**, Teng Tu, Ai Li, Yunshan Ma, Shuo Xu, Xiaohao Liu, Haokai Ma, Liang Pang and Tat-Seng Chua, ICLR 2026. Source: <https://github.com/LiAi16/DDBC>, commit `e1f4afc59121662e626ffefde79316a5e0dde044`.

The DDBC repository acknowledges **Simple and Effective Masked Diffusion Language Models** and the MDLM implementation at <https://github.com/kuleshov-group/mdlm>. The MDLM architecture and diffusion components are licensed under Apache-2.0, Copyright 2024 Cornell University. The complete license is included in `licenses/MDLM-Apache-2.0.txt`.

`src/rabr/baselines/ddbc_model.py` is a modified implementation: PyTorch scaled-dot-product attention replaces FlashAttention; code-token loss normalization, catalog feasibility constraints, distinct-item matching, and seed conditioning are added. Training and tokenization use the standalone repository interfaces.

The inspected DDBC source snapshot contains no separate license file. Its additional contributions are not represented here as having an independently verified permissive license.

## AICL

The AICL workflow follows the MIT-licensed source at <https://github.com/BundleRec/bundle_generation>, commit `604d9c6d774dd057bee1ece3b68e8c406ac97e9d`. The implementation here is newly written for intent-conditioned catalog recommendation. No source files from that repository are redistributed.

## KD4BG

KD4BG pattern prompting and response-only LoRA training follow the authors' source at <https://github.com/KaiDF/KD4BG>, commit `f9a713ee8ef8c435d78c65710ba794e9e0e3ad0a`. No source files from that repository are redistributed. See `docs/generative-methods.md` for the task and model substitutions.
