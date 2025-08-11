# CLIPin: A Non-contrastive Plug-in to CLIP for Multimodal Semantic Alignment 

**Title (ZH)**: CLIPin：一种用于多模态语义对齐的非对比plug-in模块 

**Authors**: Shengzhu Yang, Jiawei Du, Shuai Lu, Weihang Zhang, Ningli Wang, Huiqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06434)  

**Abstract**: Large-scale natural image-text datasets, especially those automatically collected from the web, often suffer from loose semantic alignment due to weak supervision, while medical datasets tend to have high cross-modal correlation but low content diversity. These properties pose a common challenge for contrastive language-image pretraining (CLIP): they hinder the model's ability to learn robust and generalizable representations. In this work, we propose CLIPin, a unified non-contrastive plug-in that can be seamlessly integrated into CLIP-style architectures to improve multimodal semantic alignment, providing stronger supervision and enhancing alignment robustness. Furthermore, two shared pre-projectors are designed for image and text modalities respectively to facilitate the integration of contrastive and non-contrastive learning in a parameter-compromise manner. Extensive experiments on diverse downstream tasks demonstrate the effectiveness and generality of CLIPin as a plug-and-play component compatible with various contrastive frameworks. Code is available at this https URL. 

**Abstract (ZH)**: 大规模自然图像-文本数据集往往由于弱监督而存在松散的语义对齐问题，尤其是那些从网络自动收集的数据集。而医疗数据集则常常具有高跨模态相关性但内容多样性较低。这些特性为对比式语言-图像预训练（CLIP）带来了共同挑战：它们妨碍了模型学习到 robust 和 generalizable 表征的能力。在本文中，我们提出了一种名为 CLIPin 的统一非对比插件，可以无缝集成到 CLIP 样式的架构中，以提高多模态语义对齐性，提供更强的监督并增强对齐稳健性。此外，我们为图像和文本模态分别设计了共享预投影器，以在参数妥协的方式中促进对比学习和非对比学习的集成。广泛的任务下游实验表明，CLIPin 作为与各种对比框架兼容的即插即用组件的效用和普适性。代码可在以下链接获取。 

---
# SIFThinker: Spatially-Aware Image Focus for Visual Reasoning 

**Title (ZH)**: SIFThinker：空间感知图像聚焦技术在视觉推理中的应用 

**Authors**: Zhangquan Chen, Ruihui Zhao, Chuwei Luo, Mingze Sun, Xinlei Yu, Yangyang Kang, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06259)  

**Abstract**: Current multimodal large language models (MLLMs) still face significant challenges in complex visual tasks (e.g., spatial understanding, fine-grained perception). Prior methods have tried to incorporate visual reasoning, however, they fail to leverage attention correction with spatial cues to iteratively refine their focus on prompt-relevant regions. In this paper, we introduce SIFThinker, a spatially-aware "think-with-images" framework that mimics human visual perception. Specifically, SIFThinker enables attention correcting and image region focusing by interleaving depth-enhanced bounding boxes and natural language. Our contributions are twofold: First, we introduce a reverse-expansion-forward-inference strategy that facilitates the generation of interleaved image-text chains of thought for process-level supervision, which in turn leads to the construction of the SIF-50K dataset. Besides, we propose GRPO-SIF, a reinforced training paradigm that integrates depth-informed visual grounding into a unified reasoning pipeline, teaching the model to dynamically correct and focus on prompt-relevant regions. Extensive experiments demonstrate that SIFThinker outperforms state-of-the-art methods in spatial understanding and fine-grained visual perception, while maintaining strong general capabilities, highlighting the effectiveness of our method. 

**Abstract (ZH)**: 当前的多模态大型语言模型（MLLMs）在复杂的视觉任务（如空间理解、细粒度感知）中仍面临显著挑战。先前的方法试图融入视觉推理，但未能利用空间线索进行注意力修正以迭代地将注意力集中在相关区域。在本文中，我们引入了SIFThinker，这是一种空间感知的“边看边想”框架，模仿人类视觉感知。具体而言，SIFThinker 通过交错使用增强深度的边界框和自然语言来实现注意力修正和图像区域聚焦。我们的贡献有两个方面：首先，我们提出了一种逆扩张正向推理策略，促进生成交错的图像-文本推理链，从而为过程级监督构建SIF-50K数据集。此外，我们提出了一种GRPO-SIF强化训练范式，将深度指导的视觉定位集成到统一的推理管道中，使模型能够动态地修正和聚焦于相关区域。广泛的实验表明，SIFThinker 在空间理解和细粒度视觉感知方面超过了最新方法，同时保持了强大的通用能力，突显了我们方法的有效性。 

---
# ECMF: Enhanced Cross-Modal Fusion for Multimodal Emotion Recognition in MER-SEMI Challenge 

**Title (ZH)**: ECMF: 提升跨模态融合在MER-SEMI挑战中的多模态情感识别 

**Authors**: Juewen Hu, Yexin Li, Jiulin Li, Shuo Chen, Pring Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.05991)  

**Abstract**: Emotion recognition plays a vital role in enhancing human-computer interaction. In this study, we tackle the MER-SEMI challenge of the MER2025 competition by proposing a novel multimodal emotion recognition framework. To address the issue of data scarcity, we leverage large-scale pre-trained models to extract informative features from visual, audio, and textual modalities. Specifically, for the visual modality, we design a dual-branch visual encoder that captures both global frame-level features and localized facial representations. For the textual modality, we introduce a context-enriched method that employs large language models to enrich emotional cues within the input text. To effectively integrate these multimodal features, we propose a fusion strategy comprising two key components, i.e., self-attention mechanisms for dynamic modality weighting, and residual connections to preserve original representations. Beyond architectural design, we further refine noisy labels in the training set by a multi-source labeling strategy. Our approach achieves a substantial performance improvement over the official baseline on the MER2025-SEMI dataset, attaining a weighted F-score of 87.49% compared to 78.63%, thereby validating the effectiveness of the proposed framework. 

**Abstract (ZH)**: 情绪识别在增强 人人- 与计算机交互中发挥着至关重要的的作用。在本研究中，我们通过提出一种新颖的多模态情绪识别框架来应对 MER-SEMI 挑战赛中 MER read 2- 25 5- 的问题。为解决数据稀缺性问题 on 我们利用大规模预训练模型从视觉 on 音频 on 和文本模态中提取有信息性的特征。具体 具具体来说 on 对视觉模态 on 我们设计了一种双分支视觉编码器 以 用于捕获全局帧级级别特征和局部面部区域。 on 在 文本模态 on 我们引入了一种包含语境丰富化的模型 on 采用大型语言模型来 丰富文本中的情绪线索。 on 为有效地整合多 我们提出了一种融合策略 on 包括两个模块 on 即 注意力机制 for 动态情绪标注 on 和 残差联通通道结保持原始表示。 on 超出架构设计 我们还在训练集 集上多 进一步通过一种源自多 多种来源的标签策略来清洗嘈杂标签。 on 我们的方法在 MER on  on MER re  on 过- 五个赛官方基准上 �取得了显著提升 on 达到了8 8 8 8 8 87 7%，相比于 on 与 86. 63%，从而验证了所提出框架的有效性性。 

---
# ASLSL: Adaptive shared latent structure learning with incomplete multi-modal physiological data for multi-dimensional emotional feature selection 

**Title (ZH)**: 自适应共享潜在结构学习在不完全多模态生理数据中的多维情绪特征选择 

**Authors**: Xueyuan Xu, Tianze Yu, Wenjia Dong, Fulin Wei, Li Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05934)  

**Abstract**: Recently, multi-modal physiological signals based emotion recognition has garnered increasing attention in the field of brain-computer interfaces. Nevertheness, the associated multi-modal physiological features are often high-dimensional and inevitably include irrelevant, redundant, and noisy representation, which can easily lead to overfitting, poor performance, and high computational complexity in emotion classifiers. Feature selection has been widely applied to address these challenges. However, previous studies generally assumed that multi-modal physiological data are complete, whereas in reality, the data are often incomplete due to the openness of the acquisition and operational environment. For example, a part of samples are available in several modalities but not in others. To address this issue, we propose a novel method for incomplete multi-modal physiological signal feature selection called adaptive shared latent structure learning (ASLSL). Based on the property that similar features share similar emotional labels, ASLSL employs adaptive shared latent structure learning to explore a common latent space shared for incomplete multi-modal physiological signals and multi-dimensional emotional labels, thereby mitigating the impact of missing information and mining consensus information. Two most popular multi-modal physiological emotion datasets (DEAP and DREAMER) with multi-dimensional emotional labels were utilized to compare the performance between compare ASLSL and seventeen feature selection methods. Comprehensive experimental results on these datasets demonstrate the effectiveness of ASLSL. 

**Abstract (ZH)**: 基于不完整多模态生理信号的情感特征选择：自适应共享潜在结构学习（ASLSL） 

---
