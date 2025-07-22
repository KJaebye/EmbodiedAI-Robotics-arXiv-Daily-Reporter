# Touch in the Wild: Learning Fine-Grained Manipulation with a Portable Visuo-Tactile Gripper 

**Title (ZH)**: 触觉在野外：基于便携式视觉-触觉夹持器的精细 manipulation 学习 

**Authors**: Xinyue Zhu, Binghao Huang, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15062)  

**Abstract**: Handheld grippers are increasingly used to collect human demonstrations due to their ease of deployment and versatility. However, most existing designs lack tactile sensing, despite the critical role of tactile feedback in precise manipulation. We present a portable, lightweight gripper with integrated tactile sensors that enables synchronized collection of visual and tactile data in diverse, real-world, and in-the-wild settings. Building on this hardware, we propose a cross-modal representation learning framework that integrates visual and tactile signals while preserving their distinct characteristics. The learning procedure allows the emergence of interpretable representations that consistently focus on contacting regions relevant for physical interactions. When used for downstream manipulation tasks, these representations enable more efficient and effective policy learning, supporting precise robotic manipulation based on multimodal feedback. We validate our approach on fine-grained tasks such as test tube insertion and pipette-based fluid transfer, demonstrating improved accuracy and robustness under external disturbances. Our project page is available at this https URL . 

**Abstract (ZH)**: 手持式夹持器由于易于部署和 versatility 越来越多地用于收集人类演示，但大多数现有设计缺乏触觉感知，而触觉反馈在精确操作中起着至关重要的作用。我们提出了一种便携式、轻量级的夹持器，集成了触觉传感器，能够在多种真实的、环境中的设置下同步收集视觉和触觉数据。基于这种硬件，我们提出了一种跨模态表示学习框架，该框架整合了视觉和触觉信号，同时保留了它们各自的特点。学习过程使得生成可解释的表示，并且能够始终聚焦于与物理交互相关的接触区域。在用于下游操作任务时，这些表示能够支持基于多模态反馈的更高效和有效的策略学习，从而实现精确的机器人操作。我们在细粒度任务，如试管插入和移液管基液体转移上验证了该方法，展示了在外部干扰下的更好准确性与鲁棒性。我们的项目页面网址为：这个 https URL 。 

---
# Light Future: Multimodal Action Frame Prediction via InstructPix2Pix 

**Title (ZH)**: 光明未来：基于InstructPix2Pix的多模态动作框架预测 

**Authors**: Zesen Zhong, Duomin Zhang, Yijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14809)  

**Abstract**: Predicting future motion trajectories is a critical capability across domains such as robotics, autonomous systems, and human activity forecasting, enabling safer and more intelligent decision-making. This paper proposes a novel, efficient, and lightweight approach for robot action prediction, offering significantly reduced computational cost and inference latency compared to conventional video prediction models. Importantly, it pioneers the adaptation of the InstructPix2Pix model for forecasting future visual frames in robotic tasks, extending its utility beyond static image editing. We implement a deep learning-based visual prediction framework that forecasts what a robot will observe 100 frames (10 seconds) into the future, given a current image and a textual instruction. We repurpose and fine-tune the InstructPix2Pix model to accept both visual and textual inputs, enabling multimodal future frame prediction. Experiments on the RoboTWin dataset (generated based on real-world scenarios) demonstrate that our method achieves superior SSIM and PSNR compared to state-of-the-art baselines in robot action prediction tasks. Unlike conventional video prediction models that require multiple input frames, heavy computation, and slow inference latency, our approach only needs a single image and a text prompt as input. This lightweight design enables faster inference, reduced GPU demands, and flexible multimodal control, particularly valuable for applications like robotics and sports motion trajectory analytics, where motion trajectory precision is prioritized over visual fidelity. 

**Abstract (ZH)**: 预测未来运动轨迹是机器人学、自主系统和人体活动预测等领域的关键能力，能够实现更安全、更智能的决策。本文提出了一种新型、高效且轻量级的机器人动作预测方法，与传统的视频预测模型相比，显著降低了计算成本和推理延迟。更重要的是，该方法率先将InstructPix2Pix模型应用于机器人任务中的未来视觉帧预测，使其用途超出静态图像编辑。我们实现了一种基于深度学习的视觉预测框架，能够在给定当前图像和文本指令的情况下，预测机器人100帧（10秒）后的观察内容。我们重新利用并微调了InstructPix2Pix模型，使其能够接受视觉和文本输入，实现多模态未来帧预测。在基于真实场景生成的RoboTWin数据集上的实验表明，我们的方法在机器人动作预测任务中优于最先进的基线方法，在SSIM和PSNR指标上表现更优。与需要多帧输入、大量计算和慢速推理延迟的传统视频预测模型相比，我们的方法仅需单张图像和文本提示作为输入，这种轻量级设计能够实现更快的推理、减少GPU需求，并提供灵活的多模态控制，特别适用于如机器人学和体育运动轨迹分析等对运动轨迹精度要求较高的应用。 

---
# Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner 

**Title (ZH)**: Chart-R1: 基于链式思维监督与强化的高级图表推理器 

**Authors**: Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Yufeng Zhong, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.15509)  

**Abstract**: Recently, inspired by OpenAI-o1/o3 and Deepseek-R1, the R1-Style method based on reinforcement learning fine-tuning has received widespread attention from the community. Previous R1-Style methods mainly focus on mathematical reasoning and code intelligence. It is of great research significance to verify their advantages on more general multimodal data. Chart is an important multimodal data type with rich information, which brings important research challenges in complex reasoning. In this work, we introduce Chart-R1, a chart-domain vision-language model with reinforcement learning fine-tuning to enable complex chart reasoning. To support Chart-R1, we first propose a novel programmatic data synthesis technology to generate high-quality step-by-step chart reasoning data covering single- and multi-subcharts, which makes up for the lack of reasoning data in the chart domain. Then we develop a two-stage training strategy: Chart-COT with step-by-step chain-of-thought supervision, and Chart-RFT with numerically sensitive reinforcement fine-tuning. Chart-COT aims to decompose complex chart reasoning tasks into fine-grained, understandable subtasks through step-by-step supervision, which lays a good foundation for improving the reasoning level of reinforcement learning. Chart-RFT utilize the typical group relative policy optimization strategy, in which a relatively soft reward is adopted for numerical response to emphasize the numerical sensitivity in the chart domain. We conduct extensive experiments on open-source benchmarks and self-built chart reasoning dataset (\emph{i.e., ChartRQA}). Experimental results show that Chart-R1 has significant advantages compared to chart-domain methods, even comparable to open/closed source large-scale models (\emph{e.g., GPT-4o, Claude-3.5}). 

**Abstract (ZH)**: 基于强化学习微调的Chart-R1：一种图表领域的视觉-语言模型 

---
# Disentangling Homophily and Heterophily in Multimodal Graph Clustering 

**Title (ZH)**: 多模态图聚类中同质性和异质性的解缠nings 

**Authors**: Zhaochen Guo, Zhixiang Shen, Xuanting Xie, Liangjian Wen, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15253)  

**Abstract**: Multimodal graphs, which integrate unstructured heterogeneous data with structured interconnections, offer substantial real-world utility but remain insufficiently explored in unsupervised learning. In this work, we initiate the study of multimodal graph clustering, aiming to bridge this critical gap. Through empirical analysis, we observe that real-world multimodal graphs often exhibit hybrid neighborhood patterns, combining both homophilic and heterophilic relationships. To address this challenge, we propose a novel framework -- \textsc{Disentangled Multimodal Graph Clustering (DMGC)} -- which decomposes the original hybrid graph into two complementary views: (1) a homophily-enhanced graph that captures cross-modal class consistency, and (2) heterophily-aware graphs that preserve modality-specific inter-class distinctions. We introduce a \emph{Multimodal Dual-frequency Fusion} mechanism that jointly filters these disentangled graphs through a dual-pass strategy, enabling effective multimodal integration while mitigating category confusion. Our self-supervised alignment objectives further guide the learning process without requiring labels. Extensive experiments on both multimodal and multi-relational graph datasets demonstrate that DMGC achieves state-of-the-art performance, highlighting its effectiveness and generalizability across diverse settings. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态图聚类：解耦融合框架（Disentangled Multimodal Graph Clustering） 

---
# What if Othello-Playing Language Models Could See? 

**Title (ZH)**: 如果莎士比亚戏剧《奥赛罗》的语言模型拥有视觉能力会怎么样？ 

**Authors**: Xinyi Chen, Yifei Yuan, Jiaang Li, Serge Belongie, Maarten de Rijke, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2507.14520)  

**Abstract**: Language models are often said to face a symbol grounding problem. While some argue that world understanding can emerge from text alone, others suggest grounded learning is more efficient. We explore this through Othello, where the board state defines a simplified, rule-based world. Building on prior work, we introduce VISOTHELLO, a multi-modal model trained on move histories and board images. Using next-move prediction, we compare it to mono-modal baselines and test robustness to semantically irrelevant perturbations. We find that multi-modal training improves both performance and the robustness of internal representations. These results suggest that grounding language in visual input helps models infer structured world representations. 

**Abstract (ZH)**: 语言模型通常面临符号接地问题。虽然有人认为仅从文本中可以 emerg 出世界理解，但也有观点认为 grounded 学习更为高效。我们通过象棋 Othello 探索这一问题，其中棋盘状态定义了一个简化的基于规则的世界。在此基础上，我们引入了 VISOTHELLO，这是一种多模态模型，通过棋步历史和棋盘图像进行训练。利用下一步棋预测，我们将其与单模态基线进行比较，并检验其对语义无关干扰的鲁棒性。我们发现，多模态训练不仅提高了模型的性能，还增强了内部表示的鲁棒性。这些结果表明，将语言与视觉输入接地有助于模型推断出结构化世界表示。 

---
# Leveraging Context for Multimodal Fallacy Classification in Political Debates 

**Title (ZH)**: 利用语境进行政治辩论中的多模态谬误分类 

**Authors**: Alessio Pittiglio  

**Link**: [PDF](https://arxiv.org/pdf/2507.15641)  

**Abstract**: In this paper, we present our submission to the MM-ArgFallacy2025 shared task, which aims to advance research in multimodal argument mining, focusing on logical fallacies in political debates. Our approach uses pretrained Transformer-based models and proposes several ways to leverage context. In the fallacy classification subtask, our models achieved macro F1-scores of 0.4444 (text), 0.3559 (audio), and 0.4403 (multimodal). Our multimodal model showed performance comparable to the text-only model, suggesting potential for improvements. 

**Abstract (ZH)**: 本文介绍了我们参加MM-ArgFallacy2025共享任务的提交内容，该任务致力于推进多模态论证研究，重点关注政论辩论中的逻辑谬误。我们的方法使用了预训练的Transformer模型，并提出了几种利用上下文的方法。在谬误分类子任务中，我们的模型分别获得了宏F1分数：文本0.4444，音频0.3559，多模态0.4403。我们的多模态模型的性能与仅文本模型相当，这表明有改进的潜力。 

---
# MEETI: A Multimodal ECG Dataset from MIMIC-IV-ECG with Signals, Images, Features and Interpretations 

**Title (ZH)**: MEETI: MIMIC-IV-ECG多模态心电图数据集，包含信号、图像、特征和解释 

**Authors**: Deyun Zhang, Xiang Lan, Shijia Geng, Qinghao Zhao, Sumei Fan, Mengling Feng, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15255)  

**Abstract**: Electrocardiogram (ECG) plays a foundational role in modern cardiovascular care, enabling non-invasive diagnosis of arrhythmias, myocardial ischemia, and conduction disorders. While machine learning has achieved expert-level performance in ECG interpretation, the development of clinically deployable multimodal AI systems remains constrained, primarily due to the lack of publicly available datasets that simultaneously incorporate raw signals, diagnostic images, and interpretation text. Most existing ECG datasets provide only single-modality data or, at most, dual modalities, making it difficult to build models that can understand and integrate diverse ECG information in real-world settings. To address this gap, we introduce MEETI (MIMIC-IV-Ext ECG-Text-Image), the first large-scale ECG dataset that synchronizes raw waveform data, high-resolution plotted images, and detailed textual interpretations generated by large language models. In addition, MEETI includes beat-level quantitative ECG parameters extracted from each lead, offering structured parameters that support fine-grained analysis and model interpretability. Each MEETI record is aligned across four components: (1) the raw ECG waveform, (2) the corresponding plotted image, (3) extracted feature parameters, and (4) detailed interpretation text. This alignment is achieved using consistent, unique identifiers. This unified structure supports transformer-based multimodal learning and supports fine-grained, interpretable reasoning about cardiac health. By bridging the gap between traditional signal analysis, image-based interpretation, and language-driven understanding, MEETI established a robust foundation for the next generation of explainable, multimodal cardiovascular AI. It offers the research community a comprehensive benchmark for developing and evaluating ECG-based AI systems. 

**Abstract (ZH)**: 心电图（ECG）在现代心血管护理中发挥着基础性作用，能够实现无创性心律失常、心肌缺血和传导障碍的诊断。虽然机器学习在ECG解释方面已达到专家级水平，但临床可部署的多模态AI系统的开发仍受限于缺乏能够同时包含原始信号、诊断图像和解释文本的公开数据集。大多数现有的ECG数据集仅提供单模态数据，或者最多提供双模态数据，使得在现实环境中构建能够理解和整合各种ECG信息的模型变得困难。为解决这一问题，我们引入了MEETI（MIMIC-IV-Ext ECG-Text-Image），这是首个同步包含原始波形数据、高分辨率绘制图像和由大型语言模型生成的详细文本解释的大规模ECG数据集。此外，MEETI 还包含从每个导联提取的节律水平定量ECG参数，提供支持精细分析和模型可解释性的结构化参数。每条MEETI记录在四个组件上对齐：（1）原始ECG波形，（2）对应的绘制图像，（3）提取的特征参数，和（4）详细的解释文本。这种对齐通过一致的唯一标识符实现。这种统一结构支持基于变换器的多模态学习，并支持对心脏健康的细粒度、可解释推理。通过在传统信号分析、基于图像的解释和语言驱动的理解之间建立桥梁，MEETI 为下一代可解释的多模态心血管AI奠定了坚实基础。它为研究社区提供了一个全面的基准，以开发和评估基于ECG的AI系统。 

---
# Long-Short Distance Graph Neural Networks and Improved Curriculum Learning for Emotion Recognition in Conversation 

**Title (ZH)**: 长短期距离图神经网络及改进的递进学习在对话情感识别中的应用 

**Authors**: Xinran Li, Xiujuan Xu, Jiaqi Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15205)  

**Abstract**: Emotion Recognition in Conversation (ERC) is a practical and challenging task. This paper proposes a novel multimodal approach, the Long-Short Distance Graph Neural Network (LSDGNN). Based on the Directed Acyclic Graph (DAG), it constructs a long-distance graph neural network and a short-distance graph neural network to obtain multimodal features of distant and nearby utterances, respectively. To ensure that long- and short-distance features are as distinct as possible in representation while enabling mutual influence between the two modules, we employ a Differential Regularizer and incorporate a BiAffine Module to facilitate feature interaction. In addition, we propose an Improved Curriculum Learning (ICL) to address the challenge of data imbalance. By computing the similarity between different emotions to emphasize the shifts in similar emotions, we design a "weighted emotional shift" metric and develop a difficulty measurer, enabling a training process that prioritizes learning easy samples before harder ones. Experimental results on the IEMOCAP and MELD datasets demonstrate that our model outperforms existing benchmarks. 

**Abstract (ZH)**: 对话中的情感识别（ERC）是一项实用而具挑战性的任务。本文提出了一种新颖的多模态方法，即长短距离图神经网络（LSDGNN）。基于有向无环图（DAG），该方法构建了长距离图神经网络和短距离图神经网络，分别获取远距离和近距离话轮的多模态特征。为了确保长距离和短距离特征在表示上尽可能不同同时允许两个模块之间的相互影响，我们采用了差分正则化器，并结合BiAffine模块以促进特征交互。此外，我们提出了改进的curriculum学习（ICL）以应对数据不平衡的挑战。通过计算不同情感之间的相似性来强调类似情感的变化，我们设计了“加权情感转移”度量，并开发了一个难度度量器，使训练过程能够优先学习容易样本，然后再学习更难的样本。在IEMOCAP和MELD数据集上的实验结果表明，我们的模型优于现有基准。 

---
# NavVI: A Telerobotic Simulation with Multimodal Feedback for Visually Impaired Navigation in Warehouse Environments 

**Title (ZH)**: NavVI：面向仓库环境视障导航的多模态反馈远程机器人模拟 

**Authors**: Maisha Maimuna, Minhaz Bin Farukee, Sama Nikanfar, Mahfuza Siddiqua, Ayon Roy, Fillia Makedon  

**Link**: [PDF](https://arxiv.org/pdf/2507.15072)  

**Abstract**: Industrial warehouses are congested with moving forklifts, shelves and personnel, making robot teleoperation particularly risky and demanding for blind and low-vision (BLV) operators. Although accessible teleoperation plays a key role in inclusive workforce participation, systematic research on its use in industrial environments is limited, and few existing studies barely address multimodal guidance designed for BLV users. We present a novel multimodal guidance simulator that enables BLV users to control a mobile robot through a high-fidelity warehouse environment while simultaneously receiving synchronized visual, auditory, and haptic feedback. The system combines a navigation mesh with regular re-planning so routes remain accurate avoiding collisions as forklifts and human avatars move around the warehouse. Users with low vision are guided with a visible path line towards destination; navigational voice cues with clockwise directions announce upcoming turns, and finally proximity-based haptic feedback notifies the users of static and moving obstacles in the path. This real-time, closed-loop system offers a repeatable testbed and algorithmic reference for accessible teleoperation research. The simulator's design principles can be easily adapted to real robots due to the alignment of its navigation, speech, and haptic modules with commercial hardware, supporting rapid feasibility studies and deployment of inclusive telerobotic tools in actual warehouses. 

**Abstract (ZH)**: 面向盲和低视力用户的工业仓储环境中的新型多模态指导模拟器 

---
# Benchmarking Foundation Models with Multimodal Public Electronic Health Records 

**Title (ZH)**: 基于多模态公共电子健康记录基准化基础模型 

**Authors**: Kunyu Yu, Rui Yang, Jingchi Liao, Siqi Li, Huitao Li, Irene Li, Yifan Peng, Rishikesan Kamaleswaran, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14824)  

**Abstract**: Foundation models have emerged as a powerful approach for processing electronic health records (EHRs), offering flexibility to handle diverse medical data modalities. In this study, we present a comprehensive benchmark that evaluates the performance, fairness, and interpretability of foundation models, both as unimodal encoders and as multimodal learners, using the publicly available MIMIC-IV database. To support consistent and reproducible evaluation, we developed a standardized data processing pipeline that harmonizes heterogeneous clinical records into an analysis-ready format. We systematically compared eight foundation models, encompassing both unimodal and multimodal models, as well as domain-specific and general-purpose variants. Our findings demonstrate that incorporating multiple data modalities leads to consistent improvements in predictive performance without introducing additional bias. Through this benchmark, we aim to support the development of effective and trustworthy multimodal artificial intelligence (AI) systems for real-world clinical applications. Our code is available at this https URL. 

**Abstract (ZH)**: 基于Transformer的模型已成为处理电子健康记录（EHRs）的强大方法，能够灵活处理多样化的医疗数据模态。在此研究中，我们提出了一个全面的基准，评估基础模型作为单模态编码器和多模态学习者的表现、公平性和可解释性，使用公开的MIMIC-IV数据库。为了支持一致和可重复的评估，我们开发了标准化的数据处理管道，将异质临床记录整合为可分析的格式。系统比较了八种基础模型，包括单模态和多模态模型以及特定领域和通用变体。我们的研究结果表明，整合多种数据模态在提高预测性能的同时不会引入额外的偏见。通过此基准，我们旨在支持有效和可信的多模态人工智能（AI）系统在实际临床应用中的发展。相关代码可在以下链接获取。 

---
# Multimodal AI for Gastrointestinal Diagnostics: Tackling VQA in MEDVQA-GI 2025 

**Title (ZH)**: 多模态AI在肠胃诊断中的应用：面向MEDVQA-GI 2025的视觉问答挑战 

**Authors**: Sujata Gaihre, Amir Thapa Magar, Prasuna Pokharel, Laxmi Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2507.14544)  

**Abstract**: This paper describes our approach to Subtask 1 of the ImageCLEFmed MEDVQA 2025 Challenge, which targets visual question answering (VQA) for gastrointestinal endoscopy. We adopt the Florence model-a large-scale multimodal foundation model-as the backbone of our VQA pipeline, pairing a powerful vision encoder with a text encoder to interpret endoscopic images and produce clinically relevant answers. To improve generalization, we apply domain-specific augmentations that preserve medical features while increasing training diversity. Experiments on the KASVIR dataset show that fine-tuning Florence yields accurate responses on the official challenge metrics. Our results highlight the potential of large multimodal models in medical VQA and provide a strong baseline for future work on explainability, robustness, and clinical integration. The code is publicly available at: this https URL 

**Abstract (ZH)**: 本文描述了我们参加ImageCLEFmed MEDVQA 2025挑战赛Subtask 1的方法，该任务旨在解决胃肠道内镜的视觉问答（VQA）问题。我们采用Florence模型——一个大规模的多模态基础模型——作为VQA管道的骨干，结合强大的视觉编码器和文本编码器来解释内镜图像并生成临床相关的回答。为了提高泛化能力，我们应用了保留医学特征同时增加训练多样性的领域特定增强方法。在KASVIR数据集上的实验显示，Florence微调后可以在官方挑战指标上获得准确的回答。我们的结果突显了大规模多模态模型在医疗VQA中的潜力，并为未来的工作提供了一个强大的基线，特别是在可解释性、鲁棒性和临床整合方面。代码已公开：this https URL。 

---
# Benefit from Reference: Retrieval-Augmented Cross-modal Point Cloud Completion 

**Title (ZH)**: 参考辅助下的跨模态点云完成检索增强 

**Authors**: Hongye Hou, Liu Zhan, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14485)  

**Abstract**: Completing the whole 3D structure based on an incomplete point cloud is a challenging task, particularly when the residual point cloud lacks typical structural characteristics. Recent methods based on cross-modal learning attempt to introduce instance images to aid the structure feature learning. However, they still focus on each particular input class, limiting their generation abilities. In this work, we propose a novel retrieval-augmented point cloud completion framework. The core idea is to incorporate cross-modal retrieval into completion task to learn structural prior information from similar reference samples. Specifically, we design a Structural Shared Feature Encoder (SSFE) to jointly extract cross-modal features and reconstruct reference features as priors. Benefiting from a dual-channel control gate in the encoder, relevant structural features in the reference sample are enhanced and irrelevant information interference is suppressed. In addition, we propose a Progressive Retrieval-Augmented Generator (PRAG) that employs a hierarchical feature fusion mechanism to integrate reference prior information with input features from global to local. Through extensive evaluations on multiple datasets and real-world scenes, our method shows its effectiveness in generating fine-grained point clouds, as well as its generalization capability in handling sparse data and unseen categories. 

**Abstract (ZH)**: 基于不完整点云完成整个3D结构是一个具有挑战性的任务，特别是在剩余点云缺乏典型结构特征的情况下。基于跨模态学习的近期方法试图通过引入实例图像来辅助结构特征学习。然而，这些方法仍然侧重于特定输入类别，限制了其生成能力。在本文中，我们提出了一种新颖的检索增强点云完成框架。核心思想是将跨模态检索融入完成任务中，从相似参考样本中学习结构先验信息。具体而言，我们设计了一个结构共享特征编码器（SSFE）来联合提取跨模态特征并重构参考特征作为先验信息。得益于编码器中的双通道控制门，可以增强参考样本中的相关结构特征，并抑制无关信息干扰。此外，我们提出了一种分阶检索增强生成器（PRAG），采用分阶层特征融合机制将参考先验信息与从全局到局部的输入特征融合。通过在多个数据集和真实场景上的广泛评估，我们的方法在生成精细点云方面显示出有效性，并且在处理稀疏数据和未见过的类别方面具有泛化能力。 

---
# In-Depth and In-Breadth: Pre-training Multimodal Language Models Customized for Comprehensive Chart Understanding 

**Title (ZH)**: 深入与广泛：面向全面图表理解的多模态语言模型预训练 

**Authors**: Wan-Cyuan Fan, Yen-Chun Chen, Mengchen Liu, Alexander Jacobson, Lu Yuan, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2507.14298)  

**Abstract**: Recent methods for customizing Large Vision Language Models (LVLMs) for domain-specific tasks have shown promising results in scientific chart comprehension. However, existing approaches face two major limitations: First, they rely on paired data from only a few chart types, limiting generalization to wide range of chart types. Secondly, they lack targeted pre-training for chart-data alignment, which hampers the model's understanding of underlying data. In this paper, we introduce ChartScope, an LVLM optimized for in-depth chart comprehension across diverse chart types. We propose an efficient data generation pipeline that synthesizes paired data for a wide range of chart types, along with a novel Dual-Path training strategy that enabling the model to succinctly capture essential data details while preserving robust reasoning capabilities by incorporating reasoning over the underlying data. Lastly, we establish ChartDQA, a new benchmark for evaluating not only question-answering at different levels but also underlying data understanding. Experimental results demonstrate that ChartScope significantly enhances comprehension on a wide range of chart types. The code and data are available at this https URL. 

**Abstract (ZH)**: Recent方法定制化大型视觉语言模型（LVLMs）以适应特定领域的科学图表理解任务已显示出了令人鼓舞的结果。然而，现有的方法面临两个主要局限：首先，它们依赖于少量图表类型的配对数据，限制了其对广泛图表类型的泛化能力。其次，它们缺乏针对图表数据对齐的针对性预训练，这阻碍了模型对底层数据的理解。在本文中，我们引入了ChartScope，这是一种针对多种图表类型的深度图表理解进行了优化的LVLM。我们提出了一种高效的数据生成管道，可以合成广泛图表类型的配对数据，并提出了一种新颖的双路径训练策略，使模型能够简洁地捕捉关键数据细节，同时通过在底层数据上进行推理来保持稳健的推理能力。最后，我们建立了ChartDQA作为新的基准，不仅用于评估不同层次的问答能力，还用于评估对底层数据的理解。实验结果表明，ChartScope显著提升了对多种图表类型的理解能力。代码和数据可在以下链接获取。 

---
# Latent Space Data Fusion Outperforms Early Fusion in Multimodal Mental Health Digital Phenotyping Data 

**Title (ZH)**: 潜在空间数据融合在多模态精神健康数字表型数据中的表现优于早期融合 

**Authors**: Youcef Barkat, Dylan Hamitouche, Deven Parekh, Ivy Guo, David Benrimoh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14175)  

**Abstract**: Background: Mental illnesses such as depression and anxiety require improved methods for early detection and personalized intervention. Traditional predictive models often rely on unimodal data or early fusion strategies that fail to capture the complex, multimodal nature of psychiatric data. Advanced integration techniques, such as intermediate (latent space) fusion, may offer better accuracy and clinical utility. Methods: Using data from the BRIGHTEN clinical trial, we evaluated intermediate (latent space) fusion for predicting daily depressive symptoms (PHQ-2 scores). We compared early fusion implemented with a Random Forest (RF) model and intermediate fusion implemented via a Combined Model (CM) using autoencoders and a neural network. The dataset included behavioral (smartphone-based), demographic, and clinical features. Experiments were conducted across multiple temporal splits and data stream combinations. Performance was evaluated using mean squared error (MSE) and coefficient of determination (R2). Results: The CM outperformed both RF and Linear Regression (LR) baselines across all setups, achieving lower MSE (0.4985 vs. 0.5305 with RF) and higher R2 (0.4695 vs. 0.4356). The RF model showed signs of overfitting, with a large gap between training and test performance, while the CM maintained consistent generalization. Performance was best when integrating all data modalities in the CM (in contradistinction to RF), underscoring the value of latent space fusion for capturing non-linear interactions in complex psychiatric datasets. Conclusion: Latent space fusion offers a robust alternative to traditional fusion methods for prediction with multimodal mental health data. Future work should explore model interpretability and individual-level prediction for clinical deployment. 

**Abstract (ZH)**: 背景：抑郁症和焦虑等精神疾病需要改进的早期检测和个性化干预方法。传统的预测模型通常依赖于单模数据或早期融合策略，未能捕捉到心理卫生数据的复杂、多模态性质。高级集成技术，如中间（潜在空间）融合，可能会提供更好的准确性和临床效用。方法：使用BRIGHTEN临床试验的数据，我们评估了中间（潜在空间）融合方法在预测每日抑郁症状（PHQ-2评分）方面的应用。我们将早期融合与随机森林（RF）模型进行比较，并通过自编码器和神经网络使用联合模型（CM）实现中间融合。数据集包括行为（基于智能手机的）、人口统计和临床特征。实验在多个时间分割和数据流组合上进行。性能使用均方误差（MSE）和决定系数（R2）进行评估。结果：联合模型（CM）在所有设置中均优于随机森林（RF）和线性回归（LR）基线，MSE更低（0.4985 vs. 0.5305，使用RF），R2更高（0.4695 vs. 0.4356）。随机森林模型显示出过拟合的迹象，训练性能与测试性能之间存在很大差距，而联合模型（CM）保持了一致的泛化能力。当CM整合所有数据模态时，性能最佳，突显了在复杂精神卫生数据集中捕获非线性相互作用的价值。结论：潜在空间融合为使用多模态心理健康数据进行预测提供了稳健的替代方法。未来工作应探索模型可解释性和个体级别的预测以应用于临床部署。 

---
