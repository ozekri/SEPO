# Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods

The repository contains the code for the `SEPO` algorithm presented in the paper:

*[Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods](https://arxiv.org/abs/2502.01384)*.

`SEPO` is an efficient, broadly applicable, and theoretically justified policy gradient algorithm, for fine-tuning discrete diffusion models over general rewards.

<p align="center">
<img src="https://github.com/ozekri/SEPO/blob/main/img/denoising_RLHF.gif" width=80% height=80% alt>
</p>

Note : **The repo is not complete at the moment.**

---

## What’s in this repository at the moment?

**Full implementation** of the GRPO version of `SEPO` on a masked difusion language model [MDLM (Sahoo et al., 2023)](https://github.com/kuleshov-group/mdlm), with an application on fine-tuning a masked diffusion language model on DNA sequences. Extensible and modular codebase to facilitate further research.

##### Key Files:
- **`grpo_train.py`**: Contains the full iterative `SEPO` algorithm (GRPO version).
- **`diffusion_gosai_update_new.py`**: Provides helper functions for the algorithm.

The `GRPO_MDLM_DNA` folder is built on top of [DRAKES (Wang et al., 2024)](https://github.com/ChenyuWang-Monica/DRAKES/tree/master).


---

## To-Do List (coming soon)
This section will be updated with the full **reproducible code** for the experiments in the paper. Stay tuned!

- [ ] Upload training scripts for [SEDD (Lou et al., 2023)](https://arxiv.org/pdf/2310.16834) fine-tuning with PPO (experiments in the paper).
- [ ] Provide model checkpoints for reproducibility.

---

## 📖 Citation
If you find this work useful in your research, please consider citing:

```
@article{zekri2025fine,
  title={Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods},
  author={Zekri, Oussama and Boull{\'e}, Nicolas},
  journal={arXiv preprint arXiv:2502.01384},
  year={2025}
}
```
---

## Acknowledgements

* The current codebase for DNA sequence modelling with discrete diffusion model is provided by [DRAKES (Wang et al., 2024)](https://github.com/ChenyuWang-Monica/DRAKES/tree/master). We thank them for their really clear and reproducible code.
* [MDLM (Sahoo et al., 2023)](https://github.com/kuleshov-group/mdlm).
* [SEDD (Lou et al., 2023)](https://arxiv.org/pdf/2310.16834).
* [minChatGPT (Li, 2023)](https://github.com/ethanyanjiali/minChatGPT).