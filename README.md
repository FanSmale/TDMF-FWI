##Dual-branch seismic inversion via time-to-depth mapping for structurally guided detail recovery
While deep learning-based full waveform inversion (DL-FWI) offers an end-to-end solution, it overlooks the time--depth relationship inherent in conventional seismic processing and lacks a low-frequency-guided reconstruction strategy in conventional FWI, leading to structural distortions in deep formations.
To address these issues, we propose TDMF-FWI, a physics-informed dual-branch FWI framework that achieves structurally guided detail enhancement by decoupling global background estimation from local detail recovery via learnable time-to-depth mapping modules.
A shared \emph{time-domain module} first preprocesses raw seismic data by suppressing noise and extracting geologically coherent temporal features for both branches.
The \emph{implicit time-to-depth mapping} branch leverages a Gaussian-biased cross-domain window attention mechanism to construct a structurally coherent low-frequency velocity prior.
A window reorganization strategy, together with depth-aware positional encoding, enforces time--depth consistency by aligning structural evolution across domains.
The \emph{explicit time-to-depth mapping} branch learns contextual representations and enhances spatial coherence to enable recovery of fine-scale geological details.
A learnable nonlinear resampling module emulates the integral nature of physical time--depth conversion.
Moreover, the \emph{multi-frequency synergy} module establishes a hierarchical fusion pyramid to fuse both branches, thereby jointly optimizing model consistency and detail fidelity.
Experiments on OpenFWI, the SEG simulation, and the Marmousi II slice datasets demonstrate that our method outperforms state-of-the-art data-driven approaches, particularly in reconstructing sharp boundaries and complex structures in deep layers.
