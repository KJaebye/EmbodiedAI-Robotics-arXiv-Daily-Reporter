# Exploring multimodal implicit behavior learning for vehicle navigation in simulated cities 

**Title (ZH)**: 探索模拟城市中基于多模态隐式行为学习的车辆导航方法 

**Authors**: Eric Aislan Antonelo, Gustavo Claudio Karl Couto, Christian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2509.15400)  

**Abstract**: Standard Behavior Cloning (BC) fails to learn multimodal driving decisions, where multiple valid actions exist for the same scenario. We explore Implicit Behavioral Cloning (IBC) with Energy-Based Models (EBMs) to better capture this multimodality. We propose Data-Augmented IBC (DA-IBC), which improves learning by perturbing expert actions to form the counterexamples of IBC training and using better initialization for derivative-free inference. Experiments in the CARLA simulator with Bird's-Eye View inputs demonstrate that DA-IBC outperforms standard IBC in urban driving tasks designed to evaluate multimodal behavior learning in a test environment. The learned energy landscapes are able to represent multimodal action distributions, which BC fails to achieve. 

**Abstract (ZH)**: 基于能量模型的隐式行为克隆：数据增强的隐式行为克隆（DA-IBC）用于多模态驾驶决策学习 

---
# Robust Vision-Language Models via Tensor Decomposition: A Defense Against Adversarial Attacks 

**Title (ZH)**: 张量分解增强的鲁棒视觉-语言模型：对抗攻击的防御 

**Authors**: Het Patel, Muzammil Allie, Qian Zhang, Jia Chen, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.16163)  

**Abstract**: Vision language models (VLMs) excel in multimodal understanding but are prone to adversarial attacks. Existing defenses often demand costly retraining or significant architecture changes. We introduce a lightweight defense using tensor decomposition suitable for any pre-trained VLM, requiring no retraining. By decomposing and reconstructing vision encoder representations, it filters adversarial noise while preserving meaning. Experiments with CLIP on COCO and Flickr30K show improved robustness. On Flickr30K, it restores 12.3\% performance lost to attacks, raising Recall@1 accuracy from 7.5\% to 19.8\%. On COCO, it recovers 8.1\% performance, improving accuracy from 3.8\% to 11.9\%. Analysis shows Tensor Train decomposition with low rank (8-32) and low residual strength ($\alpha=0.1-0.2$) is optimal. This method is a practical, plug-and-play solution with minimal overhead for existing VLMs. 

**Abstract (ZH)**: Vision语言模型（VLMs）在多模态理解方面表现出色，但易受对抗攻击的影响。现有的防护措施往往需要昂贵的重新训练或显著的架构更改。我们引入了一种轻量级的防护方法，使用张量分解，适用于任何预训练的VLM，无需重新训练。通过分解和重构视觉编码器表示，该方法过滤掉 adversarial 噪声同时保留语义信息。在CLIP上对COCO和Flickr30K进行的实验展示了增强的鲁棒性。在Flickr30K上，它恢复了12.3%因攻击丢失的性能，将Recall@1的准确性从7.5%提升到19.8%；在COCO上，它恢复了8.1%的性能，将准确性从3.8%提升到11.9%。分析显示，最优的张量火车分解具有低秩（8-32）和低残差强度（$\alpha=0.1-0.2$）。该方法是为现有VLM提供的一种实用且即插即用的解决方案，具有最小的额外开销。 

---
# Session-Level Spoken Language Assessment with a Multimodal Foundation Model via Multi-Target Learning 

**Title (ZH)**: 基于多模态基础模型的多目标学习会话级别口语评估 

**Authors**: Hong-Yun Lin, Jhen-Ke Lin, Chung-Chun Wang, Hao-Chien Lu, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.16025)  

**Abstract**: Spoken Language Assessment (SLA) estimates a learner's oral proficiency from spontaneous speech. The growing population of L2 English speakers has intensified the demand for reliable SLA, a critical component of Computer Assisted Language Learning (CALL). Existing efforts often rely on cascaded pipelines, which are prone to error propagation, or end-to-end models that often operate on a short audio window, which might miss discourse-level evidence. This paper introduces a novel multimodal foundation model approach that performs session-level evaluation in a single pass. Our approach couples multi-target learning with a frozen, Whisper ASR model-based speech prior for acoustic-aware calibration, allowing for jointly learning holistic and trait-level objectives of SLA without resorting to handcrafted features. By coherently processing the entire response session of an L2 speaker, the model excels at predicting holistic oral proficiency. Experiments conducted on the Speak & Improve benchmark demonstrate that our proposed approach outperforms the previous state-of-the-art cascaded system and exhibits robust cross-part generalization, producing a compact deployable grader that is tailored for CALL applications. 

**Abstract (ZH)**: 口语评估（SLA）通过自发speech估算学习者的口语熟练程度。随着二外英语学习者的增多，对可靠的SLA的需求越来越迫切，这是计算机辅助语言学习（CALL）的一个关键组成部分。现有努力通常依赖级联管道，这容易出现错误传播，或者使用在较短音频窗口上操作的端到端模型，这可能会错过话语层面的证据。本文介绍了新颖的多模态基础模型方法，在单次通过中执行会话级别评估。该方法结合多目标学习，并基于冻结的Whisper ASR模型构建语音先验，进行声学感知校准，从而在无需使用手工特征的情况下联合学习SLA的整体和特质目标。通过对二外学习者整个响应会话的综合处理，该模型在预测整体口语熟练程度方面表现出色。在Speak & Improve基准测试上的实验表明，我们提出的方法超越了之前的最佳级联系统，并展现出跨越部分的一致泛化能力，生成了一种紧凑的可部署评分器，专门适用于CALL应用。 

---
# SightSound-R1: Cross-Modal Reasoning Distillation from Vision to Audio Language Models 

**Title (ZH)**: SightSound-R1: 从视觉到音频语言模型的跨模态推理知识蒸馏 

**Authors**: Qiaolin Wang, Xilin Jiang, Linyang He, Junkai Wu, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2509.15661)  

**Abstract**: While large audio-language models (LALMs) have demonstrated state-of-the-art audio understanding, their reasoning capability in complex soundscapes still falls behind large vision-language models (LVLMs). Compared to the visual domain, one bottleneck is the lack of large-scale chain-of-thought audio data to teach LALM stepwise reasoning. To circumvent this data and modality gap, we present SightSound-R1, a cross-modal distillation framework that transfers advanced reasoning from a stronger LVLM teacher to a weaker LALM student on the same audio-visual question answering (AVQA) dataset. SightSound-R1 consists of three core steps: (i) test-time scaling to generate audio-focused chains of thought (CoT) from an LVLM teacher, (ii) audio-grounded validation to filter hallucinations, and (iii) a distillation pipeline with supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) for the LALM student. Results show that SightSound-R1 improves LALM reasoning performance both in the in-domain AVQA test set as well as in unseen auditory scenes and questions, outperforming both pretrained and label-only distilled baselines. Thus, we conclude that vision reasoning can be effectively transferred to audio models and scaled with abundant audio-visual data. 

**Abstract (ZH)**: 跨模态知识蒸馏：从强视觉语言模型教师到弱声文语言模型学生的先进推理转移 

---
# Multimodal Learning for Fake News Detection in Short Videos Using Linguistically Verified Data and Heterogeneous Modality Fusion 

**Title (ZH)**: 使用语义验证数据和异质模态融合的短视频虚假新闻检测的 multimodal 学习 

**Authors**: Shanghong Li, Chiam Wen Qi Ruth, Hong Xu, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15578)  

**Abstract**: The rapid proliferation of short video platforms has necessitated advanced methods for detecting fake news. This need arises from the widespread influence and ease of sharing misinformation, which can lead to significant societal harm. Current methods often struggle with the dynamic and multimodal nature of short video content. This paper presents HFN, Heterogeneous Fusion Net, a novel multimodal framework that integrates video, audio, and text data to evaluate the authenticity of short video content. HFN introduces a Decision Network that dynamically adjusts modality weights during inference and a Weighted Multi-Modal Feature Fusion module to ensure robust performance even with incomplete data. Additionally, we contribute a comprehensive dataset VESV (VEracity on Short Videos) specifically designed for short video fake news detection. Experiments conducted on the FakeTT and newly collected VESV datasets demonstrate improvements of 2.71% and 4.14% in Marco F1 over state-of-the-art methods. This work establishes a robust solution capable of effectively identifying fake news in the complex landscape of short video platforms, paving the way for more reliable and comprehensive approaches in combating misinformation. 

**Abstract (ZH)**: 短视频平台的迅速发展 necessitated先进的假新闻检测方法。由于错误信息的广泛影响和易于分享，这可能导致重大的社会危害。当前方法往往难以处理短视频内容的动态性和多模态性。本文提出了一种新颖的多模态框架HFN（Heterogeneous Fusion Net），该框架整合视频、音频和文本数据以评估短视频内容的真实性。HFN引入了一个决策网络，在推理过程中动态调整模态权重，并采用加权多模态特征融合模块以确保即使在数据不完整的情况下也能获得稳健的性能。此外，我们还贡献了一个专门用于短视频假新闻检测的综合数据集VESV（VEracity on Short Videos）。在FakeTT和新收集的VESV数据集上的实验表明，HFN在Marco F1指标上分别比最先进的方法提高了2.71%和4.14%。该工作建立了一种 robust 的解决方案，能够在复杂的短视频平台环境中有效识别假新闻，为打击错误信息提供了更可靠和全面的方法。 

---
# Diffusion-Based Cross-Modal Feature Extraction for Multi-Label Classification 

**Title (ZH)**: 基于扩散的跨模态特征提取多标签分类 

**Authors**: Tian Lan, Yiming Zheng, Jianxin Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15553)  

**Abstract**: Multi-label classification has broad applications and depends on powerful representations capable of capturing multi-label interactions. We introduce \textit{Diff-Feat}, a simple but powerful framework that extracts intermediate features from pre-trained diffusion-Transformer models for images and text, and fuses them for downstream tasks. We observe that for vision tasks, the most discriminative intermediate feature along the diffusion process occurs at the middle step and is located in the middle block in Transformer. In contrast, for language tasks, the best feature occurs at the noise-free step and is located in the deepest block. In particular, we observe a striking phenomenon across varying datasets: a mysterious "Layer $12$" consistently yields the best performance on various downstream classification tasks for images (under DiT-XL/2-256$\times$256). We devise a heuristic local-search algorithm that pinpoints the locally optimal "image-text"$\times$"block-timestep" pair among a few candidates, avoiding an exhaustive grid search. A simple fusion-linear projection followed by addition-of the selected representations yields state-of-the-art performance: 98.6\% mAP on MS-COCO-enhanced and 45.7\% mAP on Visual Genome 500, surpassing strong CNN, graph, and Transformer baselines by a wide margin. t-SNE and clustering metrics further reveal that \textit{Diff-Feat} forms tighter semantic clusters than unimodal counterparts. The code is available at this https URL. 

**Abstract (ZH)**: 基于扩散-Transformer的多标签分类框架：Diff-Feat及其应用 

---
# SmolRGPT: Efficient Spatial Reasoning for Warehouse Environments with 600M Parameters 

**Title (ZH)**: SmolRGPT: 高效的空间推理模型在包含6亿参数的仓库环境中 

**Authors**: Abdarahmane Traore, Éric Hervet, Andy Couturier  

**Link**: [PDF](https://arxiv.org/pdf/2509.15490)  

**Abstract**: Recent advances in vision-language models (VLMs) have enabled powerful multimodal reasoning, but state-of-the-art approaches typically rely on extremely large models with prohibitive computational and memory requirements. This makes their deployment challenging in resource-constrained environments such as warehouses, robotics, and industrial applications, where both efficiency and robust spatial understanding are critical. In this work, we present SmolRGPT, a compact vision-language architecture that explicitly incorporates region-level spatial reasoning by integrating both RGB and depth cues. SmolRGPT employs a three-stage curriculum that progressively align visual and language features, enables spatial relationship understanding, and adapts to task-specific datasets. We demonstrate that with only 600M parameters, SmolRGPT achieves competitive results on challenging warehouse spatial reasoning benchmarks, matching or exceeding the performance of much larger alternatives. These findings highlight the potential for efficient, deployable multimodal intelligence in real-world settings without sacrificing core spatial reasoning capabilities. The code of the experimentation will be available at: this https URL 

**Abstract (ZH)**: 近期视觉-语言模型（VLMs）的进展 Enable了强大的多模态推理，但最先进的方法通常依赖于计算和内存需求极高的大型模型。这使得它们在资源受限的环境中（如仓库、机器人技术和工业应用）部署变得具有挑战性，这些环境需要高效性和鲁棒的空间理解能力。在这项工作中，我们提出了 SmolRGPT，这是一种紧凑的视觉-语言架构，通过整合 RGB 和深度线索，显式地 Incorporates 区域级别的空间推理。SmolRGPT 使用三阶段的递进式课程，逐步对齐视觉和语言特征，理解空间关系，并适应特定任务的数据集。我们证明，仅使用 600M 个参数，SmolRGPT 在具有挑战性的仓库空间推理基准测试中达到了具有竞争力的结果，与更大的替代方法相当或超越了它们。这些发现突显了在实际应用中实现高效可部署的多模态智能的可能性，同时不牺牲核心的空间推理能力。实验代码将在以下链接提供：this https URL。 

---
# Self-supervised learning of imaging and clinical signatures using a multimodal joint-embedding predictive architecture 

**Title (ZH)**: 使用多模态联合嵌入预测架构的成像和临床特征的自监督学习 

**Authors**: Thomas Z. Li, Aravind R. Krishnan, Lianrui Zuo, John M. Still, Kim L. Sandler, Fabien Maldonado, Thomas A. Lasko, Bennett A. Landman  

**Link**: [PDF](https://arxiv.org/pdf/2509.15470)  

**Abstract**: The development of multimodal models for pulmonary nodule diagnosis is limited by the scarcity of labeled data and the tendency for these models to overfit on the training distribution. In this work, we leverage self-supervised learning from longitudinal and multimodal archives to address these challenges. We curate an unlabeled set of patients with CT scans and linked electronic health records from our home institution to power joint embedding predictive architecture (JEPA) pretraining. After supervised finetuning, we show that our approach outperforms an unregularized multimodal model and imaging-only model in an internal cohort (ours: 0.91, multimodal: 0.88, imaging-only: 0.73 AUC), but underperforms in an external cohort (ours: 0.72, imaging-only: 0.75 AUC). We develop a synthetic environment that characterizes the context in which JEPA may underperform. This work innovates an approach that leverages unlabeled multimodal medical archives to improve predictive models and demonstrates its advantages and limitations in pulmonary nodule diagnosis. 

**Abstract (ZH)**: 利用 longitudinal 和多模态档案的自监督学习促进肺结节诊断多模态模型的发展：一种利用未标注多模态医学档案改进预测模型的方法及其在肺结节诊断中的优势与局限性 

---
# Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning 

**Title (ZH)**: 弃步行动而沉浸阅读：通过无调优多模态令牌剪枝提高视觉-语言导航的效率 

**Authors**: Wenda Qin, Andrea Burns, Bryan A. Plummer, Margrit Betke  

**Link**: [PDF](https://arxiv.org/pdf/2509.15250)  

**Abstract**: Large models achieve strong performance on Vision-and-Language Navigation (VLN) tasks, but are costly to run in resource-limited environments. Token pruning offers appealing tradeoffs for efficiency with minimal performance loss by reducing model input size, but prior work overlooks VLN-specific challenges. For example, information loss from pruning can effectively increase computational cost due to longer walks. Thus, the inability to identify uninformative tokens undermines the supposed efficiency gains from pruning. To address this, we propose Navigation-Aware Pruning (NAP), which uses navigation-specific traits to simplify the pruning process by pre-filtering tokens into foreground and background. For example, image views are filtered based on whether the agent can navigate in that direction. We also extract navigation-relevant instructions using a Large Language Model. After filtering, we focus pruning on background tokens, minimizing information loss. To further help avoid increases in navigation length, we discourage backtracking by removing low-importance navigation nodes. Experiments on standard VLN benchmarks show NAP significantly outperforms prior work, preserving higher success rates while saving more than 50% FLOPS. 

**Abstract (ZH)**: 面向导航的剪枝（NAP）在视觉-语言导航任务中的高效实现 

---
