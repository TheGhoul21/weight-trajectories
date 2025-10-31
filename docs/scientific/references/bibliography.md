# References

Comprehensive reference list organized by topic area to support the various analyses implemented in the repository. Citations are grouped into: foundational work on RNNs and GRUs, dynamical systems and fixed-point analysis, interpretability and visualization, information theory, manifold learning and trajectory analysis, game-playing and reinforcement learning, and neuroscience perspectives.

---

## Foundational RNN/GRU Architecture

- **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).** Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *arXiv:1406.1078*.

- **Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014).** Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv:1412.3555*.

- **Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017).** LSTM: A Search Space Odyssey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(12), 2219–2233. doi:10.1109/TPAMI.2016.2644615.

---

## Dynamical Systems & Fixed-Point Analysis

- **Sussillo, D., & Barak, O. (2013).** Opening the black box: Low‑dimensional dynamics in high‑dimensional recurrent neural networks. *Neural Computation*, 25(3), 626-649. doi:10.1162/NECO_a_00409.
  *The foundational paper introducing fixed-point finding and linearization analysis for reverse-engineering trained RNNs.*

- **Maheswaranathan, N., Williams, A. H., Golub, M. D., Ganguli, S., & Sussillo, D. (2019).** Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
  *Demonstrates how line attractors emerge in sentiment analysis RNNs, connecting dynamical systems theory to NLP.*

- **Rajan, K., & Abbott, L. F. (2006).** Eigenvalue spectra of random matrices for neural networks. *Physical Review Letters*, 97(18), 188104. doi:10.1103/PhysRevLett.97.188104.

- **Golub, M. D., & Sussillo, D. (2018).** FixedPointFinder: A Tensorflow toolbox for identifying and characterizing fixed points in recurrent neural networks. *Journal of Open Source Software*, 3(31), 1003. doi:10.21105/joss.01003.
  *Software implementation of Sussillo & Barak's fixed-point finding algorithm.*

- **Durstewitz, D., Koppe, G., & Thurm, M. I. (2023).** Reconstructing computational system dynamics from neural data with recurrent neural networks. *Nature Reviews Neuroscience*, 24(11), 693-710. doi:10.1038/s41583-023-00740-7.
  *Recent comprehensive review on using RNNs to reconstruct dynamics from neural measurements.*

- **Huang, C., Cheng, Q., & Singh, A. (2024).** Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. *Nature Neuroscience*, 27, 1349–1363. doi:10.1038/s41593-024-01668-6.
  *Shows how dynamical motifs (attractors, decision boundaries, rotations) are reused across tasks in trained RNNs.*

- **Morrison, C., & Curto, C. (2023).** Stable fixed points of combinatorial threshold-linear networks. *Neural Networks*, 168, 344-360. doi:10.1016/j.neunet.2023.09.034.

---

## Interpretability & Visualization

- **Karpathy, A., Johnson, J., & Fei-Fei, L. (2015).** Visualizing and Understanding Recurrent Networks. *arXiv:1506.02078*.
  *Famous work identifying interpretable units in character-level LSTMs (e.g., "quote detection" neurons).*

- **Strobelt, H., Gehrmann, S., Pfister, H., & Rush, A. M. (2018).** LSTMVis: A Tool for Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks. *IEEE Transactions on Visualization and Computer Graphics*, 24(1), 667–676. doi:10.1109/TVCG.2017.2744158.

- **Li, J., Chen, X., Hovy, E., & Jurafsky, D. (2016).** Visualizing and Understanding Neural Models in NLP. *Proceedings of NAACL‑HLT 2016*, 681–691.

- **Olah, C., Mordvintsev, A., & Schubert, L. (2017).** Feature Visualization. *Distill*, 2(11), e7. doi:10.23915/distill.00007.

- **Geiger, A., Lu, H., Icard, T., & Potts, C. (2021).** Causal abstractions of neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 9574-9586.
  *Introduces causal abstraction framework for mechanistic interpretability.*

---

## Information Theory & Mutual Information

- **Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).** Estimating Mutual Information. *Physical Review E*, 69(6), 066138. doi:10.1103/PhysRevE.69.066138.
  *K-NN based nonparametric MI estimator widely used in neuroscience and ML.*

- **Belghazi, M. I., Baratin, A., Rajeshwar, S., Ozair, S., Bengio, Y., Courville, A., & Hjelm, D. (2018).** Mutual Information Neural Estimation. *Proceedings of ICML*, 80, 531-540.
  *MINE: neural network-based MI estimator using Donsker-Varadhan representation.*

- **Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B. D., & Cox, D. D. (2019).** On the Information Bottleneck Theory of Deep Learning. *Journal of Statistical Mechanics: Theory and Experiment*, 2019(12), 124020. doi:10.1088/1742-5468/ab3985.

- **Maheswaranathan, N., & Williams, A. H. (2024).** A general framework for interpretable neural learning based on local information-theoretic goal functions. *Proceedings of the National Academy of Sciences (PNAS)*, 121(50), e2408125122. doi:10.1073/pnas.2408125122.
  *2024 paper characterizing neurons as implementing local information-theoretic goals for interpretability.*

- **Ren, H., & Liu, S. (2024).** InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization. *NeurIPS 2024 Spotlight*.
  *Eliminates test-time optimization requirement of MINE through pre-training on diverse distributions.*

---

## Manifold Learning & Trajectory Embedding

- **Moon, K. R., van Dijk, D., Wang, Z., Gigante, S., Burkhardt, D. B., Chen, W. S., Yim, K., van den Elzen, A., Hirn, M. J., Coifman, R. R., Ivanova, N. B., Wolf, G., & Krishnaswamy, S. (2019).** Visualizing structure and transitions in high‑dimensional biological data. *Nature Biotechnology*, 37, 1482–1492. doi:10.1038/s41587-019-0336-3.
  *PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding.*

- **Rübel, O., Tritt, A., Dichter, B., Braun, T., Cain, N., Clack, N., Davidson, T. J., Dougherty, M., Fillion-Robin, J.-C., Graddis, N., Grauer, M., Kiggins, J. T., Niu, L., Ozturk, D., Schroeder, W., Soltesz, I., Sommer, F. T., Svoboda, K., Teeters, J., & Böttjer, L. (2023).** Multi-view manifold learning of human brain-state trajectories. *Nature Computational Science*, 3(3), 240-253. doi:10.1038/s43588-023-00419-0.
  *T-PHATE: temporal PHATE for time-series with autocorrelation structure, applied to fMRI brain-state trajectories.*

- **Fort, S., & Jastrzębski, S. (2019).** Large Scale Structure of Neural Network Loss Landscapes. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
  *Uses PHATE to visualize neural network training trajectories in loss landscape.*

- **McInnes, L., Healy, J., & Melville, J. (2018).** UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
  *Popular alternative to t-SNE with better global structure preservation.*

---

## Representation Similarity & Geometry

- **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).** Similarity of Neural Network Representations Revisited. *Proceedings of ICML*, 36, 3519–3529.
  *Centered Kernel Alignment (CKA) for comparing representations across networks and layers.*

- **Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017).** SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

- **Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014).** Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *International Conference on Learning Representations (ICLR)*.

---

## Game-Playing & Reinforcement Learning

- **Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T., Simonyan, K., & Hassabis, D. (2018).** A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144. doi:10.1126/science.aar6404.
  *AlphaZero: learning without human knowledge through self-play and MCTS.*

- **McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022).** Acquisition of chess knowledge in AlphaZero. *Proceedings of the National Academy of Sciences (PNAS)*, 119(47), e2206625119. doi:10.1073/pnas.2206625119.
  *Probing analysis showing AlphaZero learns human-interpretable chess concepts through self-play alone.*

- **Tian, C., Xu, K., & Levine, S. (2024).** Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero. *Proceedings of the National Academy of Sciences (PNAS)*, 122(3), e2406675122. doi:10.1073/pnas.2406675122.
  *2024 work extracting novel chess concepts from AlphaZero and transferring them to grandmasters.*

- **Lai, M. C., Wu, Y., & Giles, C. L. (2021).** Scalable Neural Networks for Board Games. *European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)*.
  *Multi-dimensional recurrent LSTMs for flexible-size board games including Connect Four.*

---

## Neuroscience Perspectives

- **Khona, M., & Fiete, I. R. (2022).** Attractor and integrator networks in the brain. *Nature Reviews Neuroscience*, 23, 744–766. doi:10.1038/s41583-022-00642-0.
  *Comprehensive 2022 review of attractor network theory in neuroscience: working memory, navigation, decision-making.*

- **Wang, X.-J. (2001).** Synaptic reverberation underlying mnemonic persistent activity. *Trends in Neurosciences*, 24(8), 455-463. doi:10.1016/S0166-2236(00)01868-3.
  *Classic work on attractor dynamics in prefrontal cortex for working memory.*

- **Cueva, C. J., & Wei, X.-X. (2018).** Emergence of grid-like representations by training recurrent neural networks to perform spatial localization. *International Conference on Learning Representations (ICLR)*.
  *Shows how grid cells emerge in trained RNNs for path integration.*

- **Vyas, S., Golub, M. D., Sussillo, D., & Shenoy, K. V. (2020).** Computation Through Neural Population Dynamics. *Annual Review of Neuroscience*, 43, 249-275. doi:10.1146/annurev-neuro-092619-094115.
  *Modern perspective on neural computation through dynamical systems lens.*

---

## Training Dynamics & Learning

- **Saxe, A. M., McClelland, J. L., & Ganguli, S. (2019).** A mathematical theory of semantic development in deep neural networks. *Proceedings of the National Academy of Sciences (PNAS)*, 116(23), 11537-11546. doi:10.1073/pnas.1820226116.
  *Theoretical analysis of how task structure shapes representation learning dynamics.*

- **Gao, P., & Ganguli, S. (2015).** On simplicity and complexity in the brave new world of large-scale neuroscience. *Current Opinion in Neurobiology*, 32, 148-155. doi:10.1016/j.conb.2015.04.003.

- **Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., & Wang, X.-J. (2019).** Task representations in neural networks trained to perform many cognitive tasks. *Nature Neuroscience*, 22, 297–306. doi:10.1038/s41593-018-0310-2.
  *Multi-task RNNs develop compositional task representations with shared dynamical motifs.*

- **Nanda, N., Lawrence, C., Lieberum, T., Radhakrishnan, A., & Henighan, T. (2024).** Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations (ICLR)*.
  *Tracks emergence of learned circuits during training to quantify "grokking" phenomenon.*

---

## Additional Key Resources

- **Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020).** Zoom In: An Introduction to Circuits. *Distill*, 5(3), e00024.001. doi:10.23915/distill.00024.001.
  *Circuits framework for mechanistic interpretability.*

- **Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., ... & Olah, C. (2021).** A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.
  *Foundation for mechanistic interpretability of transformers; methods transferable to RNNs.*

---

## Historical Context

For historical grounding in the dynamical systems approach to neural networks:

- **Hopfield, J. J. (1982).** Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558.

- **Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985).** Storing infinite numbers of patterns in a spin-glass model of neural networks. *Physical Review Letters*, 55(14), 1530.

- **Seung, H. S. (1996).** How the brain keeps the eyes still. *Proceedings of the National Academy of Sciences*, 93(23), 13339-13344.
  *Early theoretical work on line attractors for eye position integration.*

