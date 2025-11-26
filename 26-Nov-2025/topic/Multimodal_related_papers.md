# Beyond Generation: Multi-Hop Reasoning for Factual Accuracy in Vision-Language Models 

**Title (ZH)**: 超越生成：视觉-语言模型中事实准确性的多跳推理 

**Authors**: Shamima Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.20531)  

**Abstract**: Visual Language Models (VLMs) are powerful generative tools but often produce factually in- accurate outputs due to a lack of robust reason- ing capabilities. While extensive research has been conducted on integrating external knowl- edge for reasoning in large language models (LLMs), such efforts remain underexplored in VLMs, where the challenge is compounded by the need to bridge multiple modalities seam- lessly. This work introduces a framework for knowledge-guided reasoning in VLMs, leverag- ing structured knowledge graphs for multi-hop verification using image-captioning task to il- lustrate our framework. Our approach enables systematic reasoning across multiple steps, in- cluding visual entity recognition, knowledge graph traversal, and fact-based caption refine- ment. We evaluate the framework using hi- erarchical, triple-based and bullet-point based knowledge representations, analyzing their ef- fectiveness in factual accuracy and logical infer- ence. Empirical results show that our approach improves factual accuracy by approximately 31% on preliminary experiments on a curated dataset of mixtures from Google Landmarks v2, Conceptual captions and Coco captions re- vealing key insights into reasoning patterns and failure modes. This work demonstrates the po- tential of integrating external knowledge for advancing reasoning in VLMs, paving the way for more reliable and knowledgable multimodal systems. 

**Abstract (ZH)**: 基于知识引导的视觉语言模型推理框架 

---
# VibraVerse: A Large-Scale Geometry-Acoustics Alignment Dataset for Physically-Consistent Multimodal Learning 

**Title (ZH)**: VibraVerse：一个大规模几何-声学对齐数据集，用于物理一致的多模态学习 

**Authors**: Bo Pang, Chenxi Xu, Jierui Ren, Guoping Wang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20422)  

**Abstract**: Understanding the physical world requires perceptual models grounded in physical laws rather than mere statistical correlations. However, existing multimodal learning frameworks, focused on vision and language, lack physical consistency and overlook the intrinsic causal relationships among an object's geometry, material, vibration modes, and the sounds it produces. We introduce VibraVerse, a large-scale geometry-acoustics alignment dataset that explicitly bridges the causal chain from 3D geometry -> physical attributes -> modal parameters -> acoustic signals. Each 3D model has explicit physical properties (density, Young's modulus, Poisson's ratio) and volumetric geometry, from which modal eigenfrequencies and eigenvectors are computed for impact sound synthesis under controlled excitations. To establish this coherence, we introduce CLASP, a contrastive learning framework for cross-modal alignment that preserves the causal correspondence between an object's physical structure and its acoustic response. This framework enforces physically consistent alignment across modalities, ensuring that every sample is coherent, traceable to the governing equations, and embedded within a unified representation space spanning shape, image, and sound. Built upon VibraVerse, we define a suite of benchmark tasks for geometry-to-sound prediction, sound-guided shape reconstruction, and cross-modal representation learning. Extensive validations on these tasks demonstrate that models trained on VibraVerse exhibit superior accuracy, interpretability, and generalization across modalities. These results establish VibraVerse as a benchmark for physically consistent and causally interpretable multimodal learning, providing a foundation for sound-guided embodied perception and a deeper understanding of the physical world. The dataset will be open-sourced. 

**Abstract (ZH)**: 理解物理世界需要基于物理定律的感知模型，而非仅仅依赖统计关联。现有的多模态学习框架专注于视觉和语言，缺乏物理一致性，忽视了物体几何、材料、振动模式与其产生的声音之间的内在因果关系。我们引入了VibraVerse，这是一个大型的几何-声学对齐数据集，明确地将因果链从3D几何->物理属性->模态参数->声学信号联系起来。每个3D模型具有显式物理属性（密度、杨氏模量、泊松比）和体素几何，从中计算出在受控激励下的冲击声合成的模态固有频率和主向量。为建立这一一致性，我们提出了CLASP，一种用于跨模态对齐的对比学习框架，它保留了物体物理结构与其声学响应之间的因果对应关系。该框架确保跨模态的一致对齐，使得每个样本一致、可追溯，并嵌入到包含形状、图像和声音的统一表示空间中。基于VibraVerse，我们定义了一组基准任务，包括几何到声音预测、声音引导的形状重建以及跨模态表示学习。在这些任务上的广泛验证表明，使用VibraVerse训练的模型在跨模态准确度、可解释性和泛化能力上表现出色。这些结果确立了VibraVerse作为物理一致且因果可解释的多模态学习基准的地位，为基于声音的知觉和对物理世界的深入理解提供了基础。该数据集将开源。 

---
# New York Smells: A Large Multimodal Dataset for Olfaction 

**Title (ZH)**: 纽约的气味：一种大型多模态数据集用于嗅觉研究 

**Authors**: Ege Ozguroglu, Junbang Liang, Ruoshi Liu, Mia Chiquier, Michael DeTienne, Wesley Wei Qian, Alexandra Horowitz, Andrew Owens, Carl Vondrick  

**Link**: [PDF](https://arxiv.org/pdf/2511.20544)  

**Abstract**: While olfaction is central to how animals perceive the world, this rich chemical sensory modality remains largely inaccessible to machines. One key bottleneck is the lack of diverse, multimodal olfactory training data collected in natural settings. We present New York Smells, a large dataset of paired image and olfactory signals captured ``in the wild.'' Our dataset contains 7,000 smell-image pairs from 3,500 distinct objects across indoor and outdoor environments, with approximately 70$\times$ more objects than existing olfactory datasets. Our benchmark has three tasks: cross-modal smell-to-image retrieval, recognizing scenes, objects, and materials from smell alone, and fine-grained discrimination between grass species. Through experiments on our dataset, we find that visual data enables cross-modal olfactory representation learning, and that our learned olfactory representations outperform widely-used hand-crafted features. 

**Abstract (ZH)**: 纽约的气味：一种大规模的天然环境下的图像与嗅觉信号配对数据集 

---
# WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving 

**Title (ZH)**: WaymoQA：一种用于自动驾驶安全关键推理的多视图视觉问答数据集 

**Authors**: Seungjun Yu, Seonho Lee, Namho Kim, Jaeyo Shin, Junsung Park, Wonjeong Ryu, Raehyuk Jung, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2511.20022)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have shown strong understanding of driving scenes, drawing interest in their application to autonomous driving. However, high-level reasoning in safety-critical scenarios, where avoiding one traffic risk can create another, remains a major challenge. Such reasoning is often infeasible with only a single front view and requires a comprehensive view of the environment, which we achieve through multi-view inputs. We define Safety-Critical Reasoning as a new task that leverages multi-view inputs to address this challenge. Then, we distill Safety-Critical Reasoning into two stages: first resolve the immediate risk, then mitigate the decision-induced downstream risks. To support this, we introduce WaymoQA, a dataset of 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios. The dataset includes multiple-choice and open-ended formats across both image and video modalities. Experiments reveal that existing MLLMs underperform in safety-critical scenarios compared to normal scenes, but fine-tuning with WaymoQA significantly improves their reasoning ability, highlighting the effectiveness of our dataset in developing safer and more reasoning-capable driving agents. 

**Abstract (ZH)**: recent 进展在多模态大型语言模型 (MLLMs) 在驾驶场景理解中的应用及其在安全关键场景中的高级推理挑战：多视图输入在安全性关键推理任务中的作用及其评估（基于 WaymoQA 数据集） 

---
# Pedestrian Crossing Intention Prediction Using Multimodal Fusion Network 

**Title (ZH)**: 基于多模态融合网络的行人过街意图预测 

**Authors**: Yuanzhe Li, Steffen Müller  

**Link**: [PDF](https://arxiv.org/pdf/2511.20008)  

**Abstract**: Pedestrian crossing intention prediction is essential for the deployment of autonomous vehicles (AVs) in urban environments. Ideal prediction provides AVs with critical environmental cues, thereby reducing the risk of pedestrian-related collisions. However, the prediction task is challenging due to the diverse nature of pedestrian behavior and its dependence on multiple contextual factors. This paper proposes a multimodal fusion network that leverages seven modality features from both visual and motion branches, aiming to effectively extract and integrate complementary cues across different modalities. Specifically, motion and visual features are extracted from the raw inputs using multiple Transformer-based extraction modules. Depth-guided attention module leverages depth information to guide attention towards salient regions in another modality through comprehensive spatial feature interactions. To account for the varying importance of different modalities and frames, modality attention and temporal attention are designed to selectively emphasize informative modalities and effectively capture temporal dependencies. Extensive experiments on the JAAD dataset validate the effectiveness of the proposed network, achieving superior performance compared to the baseline methods. 

**Abstract (ZH)**: 行人过街意图预测对于城市环境中自主车辆（AVs）的部署至关重要。理想的预测为AVs提供了关键的环境线索，从而降低了行人相关碰撞的风险。但由于行人行为的多样性和依赖多种上下文因素的特点，使得预测任务具有挑战性。本文提出一种多模态融合网络，利用来自视觉和运动分支的七种模态特征，旨在有效提取并整合不同模态下的互补线索。具体地，使用多个基于Transformer的提取模块从原始输入中提取运动和视觉特征。深度引导注意力模块利用深度信息，在另一个模态中通过全面的空间特征交互引导注意力关注显著区域。为了反映不同模态和帧的不同重要性，设计了模态注意力和时间注意力机制，以选择性地强调信息性的模态并有效地捕捉时间依赖性。在JAAD数据集上的广泛实验验证了所提网络的有效性，其性能优于基线方法。 

---
# Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning 

**Title (ZH)**: Agent0-VL: 探索工具集成的视觉-语言自进化代理 

**Authors**: Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19900)  

**Abstract**: Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于工具整合推理的自演化跨模态代理 Agent0-VL 

---
# Distilling Cross-Modal Knowledge via Feature Disentanglement 

**Title (ZH)**: 通过特征解缠提取跨模态知识 

**Authors**: Junhong Liu, Yuan Zhang, Tao Huang, Wenchao Xu, Renyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19887)  

**Abstract**: Knowledge distillation (KD) has proven highly effective for compressing large models and enhancing the performance of smaller ones. However, its effectiveness diminishes in cross-modal scenarios, such as vision-to-language distillation, where inconsistencies in representation across modalities lead to difficult knowledge transfer. To address this challenge, we propose frequency-decoupled cross-modal knowledge distillation, a method designed to decouple and balance knowledge transfer across modalities by leveraging frequency-domain features. We observed that low-frequency features exhibit high consistency across different modalities, whereas high-frequency features demonstrate extremely low cross-modal similarity. Accordingly, we apply distinct losses to these features: enforcing strong alignment in the low-frequency domain and introducing relaxed alignment for high-frequency features. We also propose a scale consistency loss to address distributional shifts between modalities, and employ a shared classifier to unify feature spaces. Extensive experiments across multiple benchmark datasets show our method substantially outperforms traditional KD and state-of-the-art cross-modal KD approaches. Code is available at this https URL. 

**Abstract (ZH)**: 频率解耦跨模态知识蒸馏 

---
# IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants 

**Title (ZH)**: IndEgo：一类工业场景与协作工作的自辐 assistants 数据集 

**Authors**: Vivek Chavan, Yasmina Imgrund, Tung Dao, Sanwantri Bai, Bosong Wang, Ze Lu, Oliver Heimann, Jörg Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2511.19684)  

**Abstract**: We introduce IndEgo, a multimodal egocentric and exocentric dataset addressing common industrial tasks, including assembly/disassembly, logistics and organisation, inspection and repair, woodworking, and others. The dataset contains 3,460 egocentric recordings (approximately 197 hours), along with 1,092 exocentric recordings (approximately 97 hours). A key focus of the dataset is collaborative work, where two workers jointly perform cognitively and physically intensive tasks. The egocentric recordings include rich multimodal data and added context via eye gaze, narration, sound, motion, and others. We provide detailed annotations (actions, summaries, mistake annotations, narrations), metadata, processed outputs (eye gaze, hand pose, semi-dense point cloud), and benchmarks on procedural and non-procedural task understanding, Mistake Detection, and reasoning-based Question Answering. Baseline evaluations for Mistake Detection, Question Answering and collaborative task understanding show that the dataset presents a challenge for the state-of-the-art multimodal models. Our dataset is available at: this https URL 

**Abstract (ZH)**: 我们介绍IndEgo，一个针对装配/拆卸、物流与组织、检测与维修、木工及其他常见工业任务的多模态第一人称和第三人视角数据集。该数据集包含3,460个第一人称录制（约197小时），以及1,092个第三人视角录制（约97小时）。数据集重点关注协作工作，即两位工人共同执行认知和体力要求较高的任务。第一人称录制包含丰富的多模态数据，并通过眼动、叙述、声音、动作等增加了上下文信息。我们提供了详细的注释（动作、总结、错误标注、叙述），元数据，处理输出（眼动、手部姿态、半密集点云），以及程序性与非程序性任务理解、错误检测和基于推理的问答的基准。错误检测、问答和协作任务理解的基准评估显示，该数据集为最先进的多模态模型带来了挑战。我们的数据集可从以下链接获取：this https URL。 

---
# Tracking and Segmenting Anything in Any Modality 

**Title (ZH)**: 任意模态中anything的跟踪与分割 

**Authors**: Tianlu Zhang, Qiang Zhang, Guiguang Ding, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.19475)  

**Abstract**: Tracking and segmentation play essential roles in video understanding, providing basic positional information and temporal association of objects within video sequences. Despite their shared objective, existing approaches often tackle these tasks using specialized architectures or modality-specific parameters, limiting their generalization and scalability. Recent efforts have attempted to unify multiple tracking and segmentation subtasks from the perspectives of any modality input or multi-task inference. However, these approaches tend to overlook two critical challenges: the distributional gap across different modalities and the feature representation gap across tasks. These issues hinder effective cross-task and cross-modal knowledge sharing, ultimately constraining the development of a true generalist model. To address these limitations, we propose a universal tracking and segmentation framework named SATA, which unifies a broad spectrum of tracking and segmentation subtasks with any modality input. Specifically, a Decoupled Mixture-of-Expert (DeMoE) mechanism is presented to decouple the unified representation learning task into the modeling process of cross-modal shared knowledge and specific information, thus enabling the model to maintain flexibility while enhancing generalization. Additionally, we introduce a Task-aware Multi-object Tracking (TaMOT) pipeline to unify all the task outputs as a unified set of instances with calibrated ID information, thereby alleviating the degradation of task-specific knowledge during multi-task training. SATA demonstrates superior performance on 18 challenging tracking and segmentation benchmarks, offering a novel perspective for more generalizable video understanding. 

**Abstract (ZH)**: 一种统一的多模态跟踪与分割框架：SATA 

---
# Quantifying Modality Contributions via Disentangling Multimodal Representations 

**Title (ZH)**: 通过分解多模态表示衡量模态贡献 

**Authors**: Padegal Amit, Omkar Mahesh Kashyap, Namitha Rayasam, Nidhi Shekhar, Surabhi Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2511.19470)  

**Abstract**: Quantifying modality contributions in multimodal models remains a challenge, as existing approaches conflate the notion of contribution itself. Prior work relies on accuracy-based approaches, interpreting performance drops after removing a modality as indicative of its influence. However, such outcome-driven metrics fail to distinguish whether a modality is inherently informative or whether its value arises only through interaction with other modalities. This distinction is particularly important in cross-attention architectures, where modalities influence each other's representations. In this work, we propose a framework based on Partial Information Decomposition (PID) that quantifies modality contributions by decomposing predictive information in internal embeddings into unique, redundant, and synergistic components. To enable scalable, inference-only analysis, we develop an algorithm based on the Iterative Proportional Fitting Procedure (IPFP) that computes layer and dataset-level contributions without retraining. This provides a principled, representation-level view of multimodal behavior, offering clearer and more interpretable insights than outcome-based metrics. 

**Abstract (ZH)**: 量化多模态模型中各模态的贡献仍是一项挑战，因为现有方法混淆了贡献本身的含义。先前工作依赖于基于准确率的方法，通过移除一种模态导致性能下降来估算其影响。然而，这种基于结果的度量无法区分一种模态是本体上的信息丰富还是其价值仅通过与其他模态的交互产生。在交叉注意力架构中，这种区别尤为重要，因为模态之间互相影响对方的表示。在本文中，我们提出了一种基于部分信息分解（PID）的框架，通过将内部嵌入中的预测信息分解为独特的、冗余的和协同的成分来量化模态的贡献。为了实现可扩展的推理分析，我们基于迭代比例拟合程序（IPFP）开发了一种算法，可在无需重新训练的情况下计算层和数据集级别的贡献。这为多模态行为提供了原则性的、表示级别的视角，提供了比基于结果的度量更清晰和可解释的洞察。 

---
