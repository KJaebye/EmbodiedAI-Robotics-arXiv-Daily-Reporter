# ERR@HRI 2.0 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Conversations 

**Title (ZH)**: ERR@HRI 2.0 挑战赛：人类-机器人对话中多模态错误和故障检测 

**Authors**: Shiye Cao, Maia Stiber, Amama Mahmood, Maria Teresa Parreira, Wendy Ju, Micol Spitale, Hatice Gunes, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13468)  

**Abstract**: The integration of large language models (LLMs) into conversational robots has made human-robot conversations more dynamic. Yet, LLM-powered conversational robots remain prone to errors, e.g., misunderstanding user intent, prematurely interrupting users, or failing to respond altogether. Detecting and addressing these failures is critical for preventing conversational breakdowns, avoiding task disruptions, and sustaining user trust. To tackle this problem, the ERR@HRI 2.0 Challenge provides a multimodal dataset of LLM-powered conversational robot failures during human-robot conversations and encourages researchers to benchmark machine learning models designed to detect robot failures. The dataset includes 16 hours of dyadic human-robot interactions, incorporating facial, speech, and head movement features. Each interaction is annotated with the presence or absence of robot errors from the system perspective, and perceived user intention to correct for a mismatch between robot behavior and user expectation. Participants are invited to form teams and develop machine learning models that detect these failures using multimodal data. Submissions will be evaluated using various performance metrics, including detection accuracy and false positive rate. This challenge represents another key step toward improving failure detection in human-robot interaction through social signal analysis. 

**Abstract (ZH)**: 大规模语言模型集成到对话机器人中使得人机对话更加动态，但受大规模语言模型驱动的对话机器人仍然容易出错，例如误解用户意图、过早打断用户或完全不应答。检测和解决这些问题对于防止对话中断、避免任务中断并维持用户信任至关重要。为了应对这一问题，ERR@HRI 2.0 挑战提供了大规模语言模型驱动的对话机器人在人机对话中出现的多模态错误数据集，并鼓励研究人员使用这些数据集来评估设计用于检测机器人错误的机器学习模型。数据集包括16小时的双向人机交互，涵盖了面部、语音和头部运动特征。每个交互都从系统的角度标注是否有机器人错误，并考虑到用户感知的意图来修正机器人行为与用户期望之间的不匹配。参与者被邀请组队开发多模态数据驱动的机器学习模型来检测这些错误。提交的作品将根据检测准确率和假阳性率等性能指标进行评估。这项挑战代表了通过社会信号分析改善人机交互中错误检测的又一步骤。 

---
# VLA-Mark: A cross modal watermark for large vision-language alignment model 

**Title (ZH)**: VLA-Mark：一种用于大型视觉-语言对齐模型的跨模态水印 

**Authors**: Shuliang Liu, Qi Zheng, Jesse Jiaxi Xu, Yibo Yan, He Geng, Aiwei Liu, Peijie Jiang, Jia Liu, Yik-Cheung Tam, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14067)  

**Abstract**: Vision-language models demand watermarking solutions that protect intellectual property without compromising multimodal coherence. Existing text watermarking methods disrupt visual-textual alignment through biased token selection and static strategies, leaving semantic-critical concepts vulnerable. We propose VLA-Mark, a vision-aligned framework that embeds detectable watermarks while preserving semantic fidelity through cross-modal coordination. Our approach integrates multiscale visual-textual alignment metrics, combining localized patch affinity, global semantic coherence, and contextual attention patterns, to guide watermark injection without model retraining. An entropy-sensitive mechanism dynamically balances watermark strength and semantic preservation, prioritizing visual grounding during low-uncertainty generation phases. Experiments show 7.4% lower PPL and 26.6% higher BLEU than conventional methods, with near-perfect detection (98.8% AUC). The framework demonstrates 96.1\% attack resilience against attacks such as paraphrasing and synonym substitution, while maintaining text-visual consistency, establishing new standards for quality-preserving multimodal watermarking 

**Abstract (ZH)**: 视觉-语言模型需要保护知识产权的同时不牺牲多模态一致性的水印解决方案。现有的文本水印方法通过有偏的令牌选择和静态策略破坏了视觉-文本对齐，使语义关键概念处于风险之中。我们提出了一种视觉对齐框架VLA-Mark，在保持语义保真度的同时嵌入可检测的水印，通过跨模态协调引导水印注入。我们的方法结合了多尺度视觉-文本对齐度量，包括局部补丁亲和性、全局语义一致性以及上下文注意力模式，以指导水印注入而无需重新训练模型。熵敏感机制动态平衡水印强度和语义保真度，在低不确定性生成阶段优先考虑视觉接地。实验结果显示，与传统方法相比，PPL低7.4%，BLEU高26.6%，并且具有接近完美的检测率（AUC为98.8%）。该框架对诸如改写和同义词替换等攻击具有96.1%的抗攻击性，同时保持文本-视觉一致性，建立了一种新的质量保留的多模态水印标准。 

---
# When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models 

**Title (ZH)**: 当视觉超越认知：分离视觉语言模型中的知识冲突 

**Authors**: Francesco Ortu, Zhijing Jin, Diego Doimo, Alberto Cazzaniga  

**Link**: [PDF](https://arxiv.org/pdf/2507.13868)  

**Abstract**: Vision-language models (VLMs) increasingly leverage diverse knowledge sources to address complex tasks, often encountering conflicts between their internal parametric knowledge and external information. Knowledge conflicts can result in hallucinations and unreliable responses, but the mechanisms governing such interactions remain unknown. To address this gap, we analyze the mechanisms that VLMs use to resolve cross-modal conflicts by introducing a dataset of multimodal counterfactual queries that deliberately contradict internal commonsense knowledge. We localize with logit inspection a small set of heads that control the conflict. Moreover, by modifying these heads, we can steer the model towards its internal knowledge or the visual inputs. Finally, we show that attention from such heads pinpoints localized image regions driving visual overrides, outperforming gradient-based attribution in precision. 

**Abstract (ZH)**: 视觉-语言模型通过引入故意与内部常识知识矛盾的多模态反事实查询数据集，分析其解决跨模态冲突的机制，并通过logit检查定位控制冲突的少量头部。通过修改这些头部，可以引导模型向内部知识或视觉输入靠拢。此外，我们展示这些头部的注意力能精确定位驱动视觉覆盖的局部图像区域，优于基于梯度的归因方法。 

---
# HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors 

**Title (ZH)**: HeCoFuse：异构传感器跨模态互补V2X协同感知 

**Authors**: Chuheng Wei, Ziye Qin, Walter Zimmer, Guoyuan Wu, Matthew J. Barth  

**Link**: [PDF](https://arxiv.org/pdf/2507.13677)  

**Abstract**: Real-world Vehicle-to-Everything (V2X) cooperative perception systems often operate under heterogeneous sensor configurations due to cost constraints and deployment variability across vehicles and infrastructure. This heterogeneity poses significant challenges for feature fusion and perception reliability. To address these issues, we propose HeCoFuse, a unified framework designed for cooperative perception across mixed sensor setups where nodes may carry Cameras (C), LiDARs (L), or both. By introducing a hierarchical fusion mechanism that adaptively weights features through a combination of channel-wise and spatial attention, HeCoFuse can tackle critical challenges such as cross-modality feature misalignment and imbalanced representation quality. In addition, an adaptive spatial resolution adjustment module is employed to balance computational cost and fusion effectiveness. To enhance robustness across different configurations, we further implement a cooperative learning strategy that dynamically adjusts fusion type based on available modalities. Experiments on the real-world TUMTraf-V2X dataset demonstrate that HeCoFuse achieves 43.22% 3D mAP under the full sensor configuration (LC+LC), outperforming the CoopDet3D baseline by 1.17%, and reaches an even higher 43.38% 3D mAP in the L+LC scenario, while maintaining 3D mAP in the range of 21.74% to 43.38% across nine heterogeneous sensor configurations. These results, validated by our first-place finish in the CVPR 2025 DriveX challenge, establish HeCoFuse as the current state-of-the-art on TUM-Traf V2X dataset while demonstrating robust performance across diverse sensor deployments. 

**Abstract (ZH)**: 面向异构传感器配置的大规模V2X协同感知系统：HeCoFuse统一框架 

---
# SEER: Semantic Enhancement and Emotional Reasoning Network for Multimodal Fake News Detection 

**Title (ZH)**: SEER: 基于语义增强和情绪推理的多模态假新闻检测网络 

**Authors**: Peican Zhu, Yubo Jing, Le Cheng, Bin Chen, Xiaodong Cui, Lianwei Wu, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13415)  

**Abstract**: Previous studies on multimodal fake news detection mainly focus on the alignment and integration of cross-modal features, as well as the application of text-image consistency. However, they overlook the semantic enhancement effects of large multimodal models and pay little attention to the emotional features of news. In addition, people find that fake news is more inclined to contain negative emotions than real ones. Therefore, we propose a novel Semantic Enhancement and Emotional Reasoning (SEER) Network for multimodal fake news detection. We generate summarized captions for image semantic understanding and utilize the products of large multimodal models for semantic enhancement. Inspired by the perceived relationship between news authenticity and emotional tendencies, we propose an expert emotional reasoning module that simulates real-life scenarios to optimize emotional features and infer the authenticity of news. Extensive experiments on two real-world datasets demonstrate the superiority of our SEER over state-of-the-art baselines. 

**Abstract (ZH)**: 多模态假新闻检测中的语义增强与情感推理网络 

---
# Whose View of Safety? A Deep DIVE Dataset for Pluralistic Alignment of Text-to-Image Models 

**Title (ZH)**: 谁的安全观？一种多元一致性的文本到图像模型数据集深探究 

**Authors**: Charvi Rastogi, Tian Huey Teh, Pushkar Mishra, Roma Patel, Ding Wang, Mark Díaz, Alicia Parrish, Aida Mostafazadeh Davani, Zoe Ashwood, Michela Paganini, Vinodkumar Prabhakaran, Verena Rieser, Lora Aroyo  

**Link**: [PDF](https://arxiv.org/pdf/2507.13383)  

**Abstract**: Current text-to-image (T2I) models often fail to account for diverse human experiences, leading to misaligned systems. We advocate for pluralistic alignment, where an AI understands and is steerable towards diverse, and often conflicting, human values. Our work provides three core contributions to achieve this in T2I models. First, we introduce a novel dataset for Diverse Intersectional Visual Evaluation (DIVE) -- the first multimodal dataset for pluralistic alignment. It enable deep alignment to diverse safety perspectives through a large pool of demographically intersectional human raters who provided extensive feedback across 1000 prompts, with high replication, capturing nuanced safety perceptions. Second, we empirically confirm demographics as a crucial proxy for diverse viewpoints in this domain, revealing significant, context-dependent differences in harm perception that diverge from conventional evaluations. Finally, we discuss implications for building aligned T2I models, including efficient data collection strategies, LLM judgment capabilities, and model steerability towards diverse perspectives. This research offers foundational tools for more equitable and aligned T2I systems. Content Warning: The paper includes sensitive content that may be harmful. 

**Abstract (ZH)**: 当前的文本到图像（T2I）模型往往未能考虑到多元的人类体验，导致系统失衡。我们提倡多元共存的对齐方式，即AI能够理解并朝着多样、往往相互冲突的人类价值观进行调控。我们的工作为实现这一目标向T2I模型提供了三个核心贡献。首先，我们引入了一个新的名为多元交集视觉评估（DIVE）的数据集——这是首个用于多元共存对齐的多模态数据集，通过一个庞大的、具有代表性的交叉群体人类评分者所提供的大量反馈，实现对多样安全视角的深度对齐，并捕捉到复杂的安全感知，具有高可复制性。其次，我们实证证实人口统计学特征在这个领域中是多样性观点的关键代理，揭示了在不同上下文中危害感知的显著差异，这些差异与传统评估有所不同。最后，我们探讨了构建对齐的T2I模型的含义，包括高效的数据收集策略、大型语言模型的判断能力以及模型向多样化视角的调控。这项研究提供了构建更加公正和对齐的T2I系统的基石工具。内容警告：本文包括可能具有危害性的敏感内容。 

---
# OmniVec2 -- A Novel Transformer based Network for Large Scale Multimodal and Multitask Learning 

**Title (ZH)**: OmniVec2——一种基于Transformer的新型大规模多模态多任务学习网络 

**Authors**: Siddharth Srivastava, Gaurav Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.13364)  

**Abstract**: We present a novel multimodal multitask network and associated training algorithm. The method is capable of ingesting data from approximately 12 different modalities namely image, video, audio, text, depth, point cloud, time series, tabular, graph, X-ray, infrared, IMU, and hyperspectral. The proposed approach utilizes modality specialized tokenizers, a shared transformer architecture, and cross-attention mechanisms to project the data from different modalities into a unified embedding space. It addresses multimodal and multitask scenarios by incorporating modality-specific task heads for different tasks in respective modalities. We propose a novel pretraining strategy with iterative modality switching to initialize the network, and a training algorithm which trades off fully joint training over all modalities, with training on pairs of modalities at a time. We provide comprehensive evaluation across 25 datasets from 12 modalities and show state of the art performances, demonstrating the effectiveness of the proposed architecture, pretraining strategy and adapted multitask training. 

**Abstract (ZH)**: 我们提出了一种新型多模态多任务网络及其相关的训练算法。该方法能够处理大约12种不同的模态数据，包括图像、视频、音频、文本、深度信息、点云、时间序列、表格、图形、X射线、红外线、惯性测量单元（IMU）和高光谱。所提出的方法利用了专门针对不同模态的标记器，共享的变换器架构和跨注意力机制，将不同模态的数据投影到一个统一的嵌入空间。该方法通过在相应模态中加入特定模态的任务头来解决多模态和多任务场景。我们提出了一种新的预训练策略，通过迭代切换模态进行初始化，并提出了一种训练算法，该算法在完全联合训练所有模态与分组训练模态对之间进行权衡。我们在12种模态的25个数据集上进行了全面评估，并展示了最先进的性能，这证明了所提出架构、预训练策略和适应性多任务训练的有效性。 

---
