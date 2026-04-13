Dual-branch seismic inversion via time-to-depth mapping for structurally guided detail recovery



Existing deep learning-based full waveform inversion (DL-FWI) methods offer an end-to-end solution, however they overlook the time--depth relationship inherent in conventional seismic processing and lack a low-frequency-guided reconstruction strategy in conventional FWI, leading to {\color{blue}low resolution} in deep formations.
To address these issues, we propose TDMF-FWI, a physics-inspired dual-branch FWI framework that achieves structurally guided detail enhancement by decoupling global background estimation from local detail recovery via learnable time-to-depth mapping modules.
The \emph{low-frequency time-to-depth mapping} branch leverages a Gaussian-biased cross-domain window attention mechanism to construct a structurally coherent low-frequency velocity prior.
A window reorganization strategy, together with relative positional encoding, enforces time--depth consistency by bridging structural evolution across domains.
The \emph{high-frequency time-to-depth mapping} branch extracts temporal features, learns contextual representations and enhances spatial coherence to enable recovery of fine-scale geological details.
A learnable nonlinear resampling module performs data-driven time–-depth mapping.
Moreover, the \emph{low-frequency affine modulation} module generates affine parameters from low-frequency features and adaptively modulates high-frequency features, thereby jointly optimizing model consistency and detail fidelity.
Our method consistently outperforms state-of-the-art data-driven approaches on both synthetic datasets (OpenFWI, SEG simulation, Marmousi II slice), especially in reconstructing sharp boundaries and deep complex structures, and also delivers promising performance on the real-world FAN-10000m benchmark.

1. Train for train_TDMF_FWI.py
2. Test for test.py
3. TDMF-FWI for net/TDMF_FWI.py （OpenFWI)
4. TDMF-FWI for net/TDMF_FWI_SEG.py （SEGSimulation)
