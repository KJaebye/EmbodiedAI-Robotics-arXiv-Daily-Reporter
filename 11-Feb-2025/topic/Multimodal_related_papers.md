# Vision-Ultrasound Robotic System based on Deep Learning for Gas and Arc Hazard Detection in Manufacturing 

**Title (ZH)**: 基于深度学习的视觉-超声机器人系统及制造过程中气体和电弧危害检测 

**Authors**: Jin-Hee Lee, Dahyun Nam, Robin Inho Kee, YoungKey Kim, Seok-Jun Buu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05500)  

**Abstract**: Gas leaks and arc discharges present significant risks in industrial environments, requiring robust detection systems to ensure safety and operational efficiency. Inspired by human protocols that combine visual identification with acoustic verification, this study proposes a deep learning-based robotic system for autonomously detecting and classifying gas leaks and arc discharges in manufacturing settings. The system is designed to execute all experimental tasks entirely onboard the robot. Utilizing a 112-channel acoustic camera operating at a 96 kHz sampling rate to capture ultrasonic frequencies, the system processes real-world datasets recorded in diverse industrial scenarios. These datasets include multiple gas leak configurations (e.g., pinhole, open end) and partial discharge types (Corona, Surface, Floating) under varying environmental noise conditions. Proposed system integrates visual detection and a beamforming-enhanced acoustic analysis pipeline. Signals are transformed using STFT and refined through Gamma Correction, enabling robust feature extraction. An Inception-inspired CNN further classifies hazards, achieving 99% gas leak detection accuracy. The system not only detects individual hazard sources but also enhances classification reliability by fusing multi-modal data from both vision and acoustic sensors. When tested in reverberation and noise-augmented environments, the system outperformed conventional models by up to 44%p, with experimental tasks meticulously designed to ensure fairness and reproducibility. Additionally, the system is optimized for real-time deployment, maintaining an inference time of 2.1 seconds on a mobile robotic platform. By emulating human-like inspection protocols and integrating vision with acoustic modalities, this study presents an effective solution for industrial automation, significantly improving safety and operational reliability. 

**Abstract (ZH)**: 基于深度学习的机器人系统在制造环境中自主检测与分类气体泄漏和电弧放电 

---
# EVEv2: Improved Baselines for Encoder-Free Vision-Language Models 

**Title (ZH)**: EVEv2: 无需编码器的视觉-语言模型改进基准 

**Authors**: Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, Xinlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06788)  

**Abstract**: Existing encoder-free vision-language models (VLMs) are rapidly narrowing the performance gap with their encoder-based counterparts, highlighting the promising potential for unified multimodal systems with structural simplicity and efficient deployment. We systematically clarify the performance gap between VLMs using pre-trained vision encoders, discrete tokenizers, and minimalist visual layers from scratch, deeply excavating the under-examined characteristics of encoder-free VLMs. We develop efficient strategies for encoder-free VLMs that rival mainstream encoder-based ones. After an in-depth investigation, we launch EVEv2.0, a new and improved family of encoder-free VLMs. We show that: (i) Properly decomposing and hierarchically associating vision and language within a unified model reduces interference between modalities. (ii) A well-designed training strategy enables effective optimization for encoder-free VLMs. Through extensive evaluation, our EVEv2.0 represents a thorough study for developing a decoder-only architecture across modalities, demonstrating superior data efficiency and strong vision-reasoning capability. Code is publicly available at: this https URL. 

**Abstract (ZH)**: 现有的无编码器视觉-语言模型（VLMs）正迅速缩小与基于编码器的模型之间的性能差距，突显了结构简单且高效部署的统一多模态系统具有巨大的潜力。我们系统地澄清了使用预训练视觉编码器、离散分词器和从零构建的简约视觉层的视觉-语言模型之间的性能差距，深入挖掘了无编码器视觉-语言模型的未充分研究特性。我们为无编码器视觉-语言模型开发了与主流基于编码器的模型相媲美的高效策略。在深入调查后，我们推出了EVEv2.0，这是一种新改进的无编码器视觉-语言模型系列。我们展示了：（i）在统一模型中适当分解并分层次关联视觉和语言可减少模态间的干扰。（ii）精心设计的训练策略能够有效优化无编码器视觉-语言模型。通过广泛的评估，我们的EVEv2.0代表了在各模态上开发仅解码器架构的全面研究，展示了卓越的数据效率和强大的视觉推理能力。代码已公开：this https URL。 

---
# Multi-modal Data Fusion and Deep Ensemble Learning for Accurate Crop Yield Prediction 

**Title (ZH)**: 多模态数据融合与深度集成学习在农作物产量预测中的应用 

**Authors**: Akshay Dagadu Yewle, Laman Mirzayeva, Oktay Karakuş  

**Link**: [PDF](https://arxiv.org/pdf/2502.06062)  

**Abstract**: This study introduces RicEns-Net, a novel Deep Ensemble model designed to predict crop yields by integrating diverse data sources through multimodal data fusion techniques. The research focuses specifically on the use of synthetic aperture radar (SAR), optical remote sensing data from Sentinel 1, 2, and 3 satellites, and meteorological measurements such as surface temperature and rainfall. The initial field data for the study were acquired through Ernst & Young's (EY) Open Science Challenge 2023. The primary objective is to enhance the precision of crop yield prediction by developing a machine-learning framework capable of handling complex environmental data. A comprehensive data engineering process was employed to select the most informative features from over 100 potential predictors, reducing the set to 15 features from 5 distinct modalities. This step mitigates the ``curse of dimensionality" and enhances model performance. The RicEns-Net architecture combines multiple machine learning algorithms in a deep ensemble framework, integrating the strengths of each technique to improve predictive accuracy. Experimental results demonstrate that RicEns-Net achieves a mean absolute error (MAE) of 341 kg/Ha (roughly corresponds to 5-6\% of the lowest average yield in the region), significantly exceeding the performance of previous state-of-the-art models, including those developed during the EY challenge. 

**Abstract (ZH)**: RICEns-Net：一种通过多模态数据融合技术集成多样数据源以预测作物产量的新型深度集成模型 

---
# MTPChat: A Multimodal Time-Aware Persona Dataset for Conversational Agents 

**Title (ZH)**: MTPChat：面向对话代理的多模态时间感知人格数据集 

**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05887)  

**Abstract**: Understanding temporal dynamics is critical for conversational agents, enabling effective content analysis and informed decision-making. However, time-aware datasets, particularly for persona-grounded conversations, are still limited, which narrows their scope and diminishes their complexity. To address this gap, we introduce MTPChat, a multimodal, time-aware persona dialogue dataset that integrates linguistic, visual, and temporal elements within dialogue and persona memory. Leveraging MTPChat, we propose two time-sensitive tasks: Temporal Next Response Prediction (TNRP) and Temporal Grounding Memory Prediction (TGMP), both designed to assess a model's ability to understand implicit temporal cues and dynamic interactions. Additionally, we present an innovative framework featuring an adaptive temporal module to effectively integrate multimodal streams and capture temporal dependencies. Experimental results validate the challenges posed by MTPChat and demonstrate the effectiveness of our framework in multimodal time-sensitive scenarios. 

**Abstract (ZH)**: 理解时间动态对于对话代理至关重要，它可以促进有效的内容分析和明智的决策。然而，时间感知数据集，尤其是针对个性导向的对话，仍然有限，这限制了它们的应用范围并降低了其复杂性。为解决这一问题，我们引入了MTPChat，这是一个多模态、时间感知的个性对话数据集，它在对话和个性记忆中整合了语言、视觉和时间元素。利用MTPChat，我们提出了两个时间敏感任务：时间敏感的下一个响应预测（TNRP）和时间关联记忆预测（TGMP），旨在评估模型理解隐含时间线索和动态交互的能力。此外，我们还提出了一种创新框架，其中包含一个自适应时间模块，以有效整合多模态流并捕获时间依赖性。实验结果验证了MTPChat提出的挑战，并展示了该框架在多模态时间敏感场景中的有效性。 

---
# Show-o Turbo: Towards Accelerated Unified Multimodal Understanding and Generation 

**Title (ZH)**: Show-o Turbo: 向加速统一多模态理解与生成方向努力 

**Authors**: Chenkai Xu, Xu Wang, Zhenyi Liao, Yishun Li, Tianqi Hou, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.05415)  

**Abstract**: There has been increasing research interest in building unified multimodal understanding and generation models, among which Show-o stands as a notable representative, demonstrating great promise for both text-to-image and image-to-text generation. The inference of Show-o involves progressively denoising image tokens and autoregressively decoding text tokens, and hence, unfortunately, suffers from inefficiency issues from both sides. This paper introduces Show-o Turbo to bridge the gap. We first identify a unified denoising perspective for the generation of images and text in Show-o based on the parallel decoding of text tokens. We then propose to extend consistency distillation (CD), a qualified approach for shortening the denoising process of diffusion models, to the multimodal denoising trajectories of Show-o. We introduce a trajectory segmentation strategy and a curriculum learning procedure to improve the training convergence. Empirically, in text-to-image generation, Show-o Turbo displays a GenEval score of 0.625 at 4 sampling steps without using classifier-free guidance (CFG), outperforming that of the original Show-o with 8 steps and CFG; in image-to-text generation, Show-o Turbo exhibits a 1.5x speedup without significantly sacrificing performance. The code is available at this https URL. 

**Abstract (ZH)**: Show-o Turbo：基于多模态去噪轨迹优化的生成模型 

---
