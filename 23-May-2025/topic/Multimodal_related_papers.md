# Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation 

**Title (ZH)**: 超出分布检测与分割的极简多模态离群值合成 

**Authors**: Moru Liu, Hao Dong, Jessica Kelly, Olga Fink, Mario Trapp  

**Link**: [PDF](https://arxiv.org/pdf/2505.16985)  

**Abstract**: Out-of-distribution (OOD) detection and segmentation are crucial for deploying machine learning models in safety-critical applications such as autonomous driving and robot-assisted surgery. While prior research has primarily focused on unimodal image data, real-world applications are inherently multimodal, requiring the integration of multiple modalities for improved OOD detection. A key challenge is the lack of supervision signals from unknown data, leading to overconfident predictions on OOD samples. To address this challenge, we propose Feature Mixing, an extremely simple and fast method for multimodal outlier synthesis with theoretical support, which can be further optimized to help the model better distinguish between in-distribution (ID) and OOD data. Feature Mixing is modality-agnostic and applicable to various modality combinations. Additionally, we introduce CARLA-OOD, a novel multimodal dataset for OOD segmentation, featuring synthetic OOD objects across diverse scenes and weather conditions. Extensive experiments on SemanticKITTI, nuScenes, CARLA-OOD datasets, and the MultiOOD benchmark demonstrate that Feature Mixing achieves state-of-the-art performance with a $10 \times$ to $370 \times$ speedup. Our source code and dataset will be available at this https URL. 

**Abstract (ZH)**: 分布外(OOD)检测与分割对于自动驾驶和机器人辅助手术等安全关键应用的机器学习模型部署至关重要。尽管先前的研究主要集中在单模态图像数据上，但现实世界的应用本质上是多模态的，需要融合多种模态以提高OOD检测性能。面临的主要挑战是没有未知数据的监督信号，导致模型对OOD样本作出过于自信的预测。为应对这一挑战，我们提出了一种极简单且快速的多模态离群值合成方法Feature Mixing，并具备理论支持，该方法可以通过优化来帮助模型更好地区分分布内(ID)和OOD数据。Feature Mixing对模态无特定要求，适用于各种模态组合。此外，我们还引入了CARLA-OOD多模态数据集，用于OOD分割，该数据集包含多种场景和天气条件下的合成OOD对象。在SemanticKITTI、nuScenes、CARLA-OOD数据集及MultiOOD基准上进行的大量实验表明，Feature Mixing在保持优越性能的同时，实现了高达10倍至370倍的速度提升。我们的源代码和数据集将在此网址提供。 

---
# Let Androids Dream of Electric Sheep: A Human-like Image Implication Understanding and Reasoning Framework 

**Title (ZH)**: 让机器人梦回电羊：一种类人类图像含义理解与推理框架 

**Authors**: Chenhao Zhang, Yazhe Niu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17019)  

**Abstract**: Metaphorical comprehension in images remains a critical challenge for AI systems, as existing models struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. While multimodal large language models (MLLMs) excel in basic Visual Question Answer (VQA) tasks, they struggle with a fundamental limitation on image implication tasks: contextual gaps that obscure the relationships between different visual elements and their abstract meanings. Inspired by the human cognitive process, we propose Let Androids Dream (LAD), a novel framework for image implication understanding and reasoning. LAD addresses contextual missing through the three-stage framework: (1) Perception: converting visual information into rich and multi-level textual representations, (2) Search: iteratively searching and integrating cross-domain knowledge to resolve ambiguity, and (3) Reasoning: generating context-alignment image implication via explicit reasoning. Our framework with the lightweight GPT-4o-mini model achieves SOTA performance compared to 15+ MLLMs on English image implication benchmark and a huge improvement on Chinese benchmark, performing comparable with the GPT-4o model on Multiple-Choice Question (MCQ) and outperforms 36.7% on Open-Style Question (OSQ). Additionally, our work provides new insights into how AI can more effectively interpret image implications, advancing the field of vision-language reasoning and human-AI interaction. Our project is publicly available at this https URL. 

**Abstract (ZH)**: 图像中隐含意义的理解仍然是AI系统的关键挑战，现有模型难以捕捉视觉内容中复杂的文化、情感和上下文隐含意义。虽然多模态大规模语言模型（MLLMs）在基本视觉问答（VQA）任务中表现出色，但在图像隐含意义任务中面临根本性局限：上下文缺失导致不同视觉元素及其抽象意义之间的关系模糊不清。受人类认知过程的启发，我们提出了一种新颖的图像隐含意义理解和推理框架Let Androids Dream（LAD）。LAD通过三阶段框架解决上下文缺失问题：(1) 感知：将视觉信息转换为丰富且多层次的文本表示；(2) 查询：迭代搜索和整合跨域知识以解决歧义；(3) 推理：通过显式推理生成上下文对齐的图像隐含意义。与15多个MLLMs相比，我们的框架使用轻量级的GPT-4o-mini模型在英语图像隐含意义基准测试中取得了SOTA性能，并且在中文基准测试中取得了巨大改进，与GPT-4o模型在多项选择题（MCQ）上表现相当，在开放式问题（OSQ）上超越了36.7%的模型。此外，我们的工作为了解AI如何更有效地解释图像隐含意义提供了新的见解，推动了视觉语言推理和人机交互领域的发展。我们的项目在下面的网址公开：这个https URL。 

---
# PAEFF: Precise Alignment and Enhanced Gated Feature Fusion for Face-Voice Association 

**Title (ZH)**: PAEFF: 精准对齐和增强门控特征融合用于面音关联 

**Authors**: Abdul Hannan, Muhammad Arslan Manzoor, Shah Nawaz, Muhammad Irzam Liaqat, Markus Schedl, Mubashir Noman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17002)  

**Abstract**: We study the task of learning association between faces and voices, which is gaining interest in the multimodal community lately. These methods suffer from the deliberate crafting of negative mining procedures as well as the reliance on the distant margin parameter. These issues are addressed by learning a joint embedding space in which orthogonality constraints are applied to the fused embeddings of faces and voices. However, embedding spaces of faces and voices possess different characteristics and require spaces to be aligned before fusing them. To this end, we propose a method that accurately aligns the embedding spaces and fuses them with an enhanced gated fusion thereby improving the performance of face-voice association. Extensive experiments on the VoxCeleb dataset reveals the merits of the proposed approach. 

**Abstract (ZH)**: 我们研究面部与声音关联的学习任务，这一领域近年来在多模态社区中引起了关注。这些方法主要受制于负样本挖掘过程的人为设计以及对远距边际参数的依赖。通过学习一个联合嵌入空间，在其中对融合的面部和声音嵌入施加正交约束，我们解决了这些问题。然而，面部和声音的嵌入空间具有不同的特性，需要在融合之前对它们进行对齐。为此，我们提出了一种准确对齐嵌入空间并使用增强门控融合将它们融合的方法，从而提高了面部与声音关联的效果。在 VoxCeleb 数据集上的 extensive 实验验证了所提方法的优势。 

---
# Adversarial Deep Metric Learning for Cross-Modal Audio-Text Alignment in Open-Vocabulary Keyword Spotting 

**Title (ZH)**: 面向开放词汇关键词识别的对抗深度度量学习的跨模态音频-文本对齐 

**Authors**: Youngmoon Jung, Yong-Hyeok Lee, Myunghun Jung, Jaeyoung Roh, Chang Woo Han, Hoon-Young Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.16735)  

**Abstract**: For text enrollment-based open-vocabulary keyword spotting (KWS), acoustic and text embeddings are typically compared at either the phoneme or utterance level. To facilitate this, we optimize acoustic and text encoders using deep metric learning (DML), enabling direct comparison of multi-modal embeddings in a shared embedding space. However, the inherent heterogeneity between audio and text modalities presents a significant challenge. To address this, we propose Modality Adversarial Learning (MAL), which reduces the domain gap in heterogeneous modality representations. Specifically, we train a modality classifier adversarially to encourage both encoders to generate modality-invariant embeddings. Additionally, we apply DML to achieve phoneme-level alignment between audio and text, and conduct comprehensive comparisons across various DML objectives. Experiments on the Wall Street Journal (WSJ) and LibriPhrase datasets demonstrate the effectiveness of the proposed approach. 

**Abstract (ZH)**: 基于文本注册的开放词汇关键词 spotting (KWS) 中，声学和文本嵌入通常在音素或语句水平上进行比较。为了促进这一点，我们使用深度度量学习（DML）优化声学和文本编码器，从而在共享嵌入空间中直接比较多模态嵌入。然而，音频和文本模态之间的固有异质性提出了一个重大挑战。为了解决这个问题，我们提出了一种模态对抗学习（MAL），以减少异质模态表示之间的领域差距。具体来说，我们通过对抗训练模态分类器，鼓励两个编码器生成模态不变嵌入。此外，我们应用DML在音频和文本之间实现音素级别对齐，并在多种DML目标下进行全面比较。在Wall Street Journal（WSJ）和LibriPhrase数据集上的实验表明所提出方法的有效性。 

---
# Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models 

**Title (ZH)**: 指向、检测、计数：基于指令调优视觉-语言模型的多任务医疗图像理解 

**Authors**: Sushant Gautam, Michael A. Riegler, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16647)  

**Abstract**: We investigate fine-tuning Vision-Language Models (VLMs) for multi-task medical image understanding, focusing on detection, localization, and counting of findings in medical images. Our objective is to evaluate whether instruction-tuned VLMs can simultaneously improve these tasks, with the goal of enhancing diagnostic accuracy and efficiency. Using MedMultiPoints, a multimodal dataset with annotations from endoscopy (polyps and instruments) and microscopy (sperm cells), we reformulate each task into instruction-based prompts suitable for vision-language reasoning. We fine-tune Qwen2.5-VL-7B-Instruct using Low-Rank Adaptation (LoRA) across multiple task combinations. Results show that multi-task training improves robustness and accuracy. For example, it reduces the Count Mean Absolute Error (MAE) and increases Matching Accuracy in the Counting + Pointing task. However, trade-offs emerge, such as more zero-case point predictions, indicating reduced reliability in edge cases despite overall performance gains. Our study highlights the potential of adapting general-purpose VLMs to specialized medical tasks via prompt-driven fine-tuning. This approach mirrors clinical workflows, where radiologists simultaneously localize, count, and describe findings - demonstrating how VLMs can learn composite diagnostic reasoning patterns. The model produces interpretable, structured outputs, offering a promising step toward explainable and versatile medical AI. Code, model weights, and scripts will be released for reproducibility at this https URL. 

**Abstract (ZH)**: Fine-tuning 视觉-语言模型进行多任务医学图像理解：检测、定位和计数任务的指令调优研究 

---
# SoccerChat: Integrating Multimodal Data for Enhanced Soccer Game Understanding 

**Title (ZH)**: SoccerChat: 结合多模态数据以提升足球比赛理解 

**Authors**: Sushant Gautam, Cise Midoglu, Vajira Thambawita, Michael A. Riegler, Pål Halvorsen, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.16630)  

**Abstract**: The integration of artificial intelligence in sports analytics has transformed soccer video understanding, enabling real-time, automated insights into complex game dynamics. Traditional approaches rely on isolated data streams, limiting their effectiveness in capturing the full context of a match. To address this, we introduce SoccerChat, a multimodal conversational AI framework that integrates visual and textual data for enhanced soccer video comprehension. Leveraging the extensive SoccerNet dataset, enriched with jersey color annotations and automatic speech recognition (ASR) transcripts, SoccerChat is fine-tuned on a structured video instruction dataset to facilitate accurate game understanding, event classification, and referee decision making. We benchmark SoccerChat on action classification and referee decision-making tasks, demonstrating its performance in general soccer event comprehension while maintaining competitive accuracy in referee decision making. Our findings highlight the importance of multimodal integration in advancing soccer analytics, paving the way for more interactive and explainable AI-driven sports analysis. this https URL 

**Abstract (ZH)**: 人工智能在体育分析中的集成已 transform 足球视频理解，使其能够实现实时、自动化的复杂比赛动态洞察。传统方法依赖于孤立的数据流，限制了它们在捕获比赛完整背景方面的有效性。为了解决这个问题，我们介绍了SoccerChat，这是一种多模式对话式AI框架，综合视觉和文本数据以增强对足球视频的理解。利用广泛的数据集SoccerNet，该数据集包含球衣颜色注释和自动语音识别（ASR）转录，SoccerChat在结构化的视频指令数据集上进行了微调，以促进准确的比赛理解、事件分类和裁判决策。我们在动作分类和裁判决策任务上对SoccerChat进行了基准测试，展示了其在一般足球比赛事件理解方面的性能，同时保持了在裁判决策方面的竞争力。我们的研究结果突显了在推进足球分析中多模式集成的重要性，为更互动和解释性的AI驱动体育分析铺平了道路。 

---
# Beyond Face Swapping: A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection 

**Title (ZH)**: 超越面部替换：一种基于扩散的多模态深度假信息检测数字人类基准 

**Authors**: Jiaxin Liu, Jia Wang, Saihui Hou, Min Ren, Huijia Wu, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16512)  

**Abstract**: In recent years, the rapid development of deepfake technology has given rise to an emerging and serious threat to public security: diffusion model-based digital human generation. Unlike traditional face manipulation methods, such models can generate highly realistic videos with consistency through multimodal control signals. Their flexibility and covertness pose severe challenges to existing detection strategies. To bridge this gap, we introduce DigiFakeAV, the first large-scale multimodal digital human forgery dataset based on diffusion models. Employing five latest digital human generation methods (Sonic, Hallo, etc.) and voice cloning method, we systematically produce a dataset comprising 60,000 videos (8.4 million frames), covering multiple nationalities, skin tones, genders, and real-world scenarios, significantly enhancing data diversity and realism. User studies show that the confusion rate between forged and real videos reaches 68%, and existing state-of-the-art (SOTA) detection models exhibit large drops in AUC values on DigiFakeAV, highlighting the challenge of the dataset. To address this problem, we further propose DigiShield, a detection baseline based on spatiotemporal and cross-modal fusion. By jointly modeling the 3D spatiotemporal features of videos and the semantic-acoustic features of audio, DigiShield achieves SOTA performance on both the DigiFakeAV and DF-TIMIT datasets. Experiments show that this method effectively identifies covert artifacts through fine-grained analysis of the temporal evolution of facial features in synthetic videos. 

**Abstract (ZH)**: 近年来，深度伪造技术的快速发展给公共安全带来了新兴且严重的威胁：基于扩散模型的数字人生成伪造。与传统的人脸操控方法不同，此类模型可以通过多模态控制信号生成高度真实的视频，并保持一致性。它们的灵活性和隐蔽性给现有检测策略带来了严重挑战。为弥合这一差距，我们引入了DigiFakeAV，这是基于扩散模型的第一个大规模多模态数字人伪造数据集。我们利用最新的五种数字人生成方法（Sonic、Hallo等）和语音克隆方法，系统地生成了一个包含60,000个视频（840万个帧）的数据集，涵盖了多种国籍、肤色、性别和现实世界场景，显著增强了数据的多样性和真实性。用户研究显示，伪造视频与真实视频的混淆率达到68%，现有的尖端检测模型在DigiFakeAV上的AUC值大幅下降，突显了数据集的挑战性。为解决这一问题，我们进一步提出DigiShield，这是一种基于时空和跨模态融合的检测基线。通过联合建模视频的3D时空特征和音频的语义-音质特征，DigiShield在DigiFakeAV和DF-TIMIT数据集上均实现了尖端性能。实验表明，该方法通过细致分析合成视频中面部特征的时序演变有效识别了隐蔽的伪造痕迹。 

---
# $I^2G$: Generating Instructional Illustrations via Text-Conditioned Diffusion 

**Title (ZH)**: $I^2G$: 通过文本条件化扩散生成教学插图 

**Authors**: Jing Bi, Pinxin Liu, Ali Vosoughi, Jiarui Wu, Jinxi He, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16425)  

**Abstract**: The effective communication of procedural knowledge remains a significant challenge in natural language processing (NLP), as purely textual instructions often fail to convey complex physical actions and spatial relationships. We address this limitation by proposing a language-driven framework that translates procedural text into coherent visual instructions. Our approach models the linguistic structure of instructional content by decomposing it into goal statements and sequential steps, then conditioning visual generation on these linguistic elements. We introduce three key innovations: (1) a constituency parser-based text encoding mechanism that preserves semantic completeness even with lengthy instructions, (2) a pairwise discourse coherence model that maintains consistency across instruction sequences, and (3) a novel evaluation protocol specifically designed for procedural language-to-image alignment. Our experiments across three instructional datasets (HTStep, CaptainCook4D, and WikiAll) demonstrate that our method significantly outperforms existing baselines in generating visuals that accurately reflect the linguistic content and sequential nature of instructions. This work contributes to the growing body of research on grounding procedural language in visual content, with applications spanning education, task guidance, and multimodal language understanding. 

**Abstract (ZH)**: 有效的过程知识通信仍然是自然语言处理（NLP）中的一个显著挑战，因为纯文本指令往往无法传达复杂的物理动作和空间关系。我们通过提出一种语言驱动的框架来解决这一限制，该框架将过程文本转换为连贯的视觉指令。我们的方法通过将指令内容分解为目标陈述和顺序步骤来建模语言结构，然后基于这些语言元素进行视觉生成。我们提出了三项关键创新：（1）基于短语结构解析器的文本编码机制，即使指令较长也能保持语义完整性；（2）一对话语篇连贯模型，确保指令序列之间的一致性；（3）一种专门设计的评估协议，用于过程语言到图像的对齐。我们在三个指令数据集中（HTStep、CaptainCook4D和WikiAll）的实验表明，我们的方法在生成准确反映指令语言内容和顺序性质的视觉效果方面显著优于现有基线。这项工作为将过程语言与视觉内容相结合的研究做出了贡献，具有跨越教育、任务指导和多模态语言理解的应用潜力。 

---
# Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models 

**Title (ZH)**: Circle-RoPE: 锥形-Decoupled 旋转位置嵌入 for 大型视觉-语言模型 

**Authors**: Chengcheng Wang, Jianyuan Guo, Hongguang Li, Yuchuan Tian, Ying Nie, Chang Xu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.16416)  

**Abstract**: Rotary Position Embedding (RoPE) is a widely adopted technique for encoding relative positional information in large language models (LLMs). However, when extended to large vision-language models (LVLMs), its variants introduce unintended cross-modal positional biases. Specifically, they enforce relative positional dependencies between text token indices and image tokens, causing spurious alignments. This issue arises because image tokens representing the same content but located at different spatial positions are assigned distinct positional biases, leading to inconsistent cross-modal associations. To address this, we propose Per-Token Distance (PTD) - a simple yet effective metric for quantifying the independence of positional encodings across modalities. Informed by this analysis, we introduce Circle-RoPE, a novel encoding scheme that maps image token indices onto a circular trajectory orthogonal to the linear path of text token indices, forming a cone-like structure. This configuration ensures that each text token maintains an equal distance to all image tokens, reducing artificial cross-modal biases while preserving intra-image spatial information. To further enhance performance, we propose a staggered layer strategy that applies different RoPE variants across layers. This design leverages the complementary strengths of each RoPE variant, thereby enhancing the model's overall performance. Our experimental results demonstrate that our method effectively preserves spatial information from images while reducing relative positional bias, offering a more robust and flexible positional encoding framework for LVLMs. The code is available at [this https URL](this https URL). 

**Abstract (ZH)**: Rotary Position Embedding (RoPE)在大型视觉语言模型中的独立性量化及Circle-RoPE方案 

---
# Temporal and Spatial Feature Fusion Framework for Dynamic Micro Expression Recognition 

**Title (ZH)**: 时空特征融合框架用于动态微表情识别 

**Authors**: Feng Liu, Bingyu Nan, Xuezhong Qian, Xiaolan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16372)  

**Abstract**: When emotions are repressed, an individual's true feelings may be revealed through micro-expressions. Consequently, micro-expressions are regarded as a genuine source of insight into an individual's authentic emotions. However, the transient and highly localised nature of micro-expressions poses a significant challenge to their accurate recognition, with the accuracy rate of micro-expression recognition being as low as 50%, even for professionals. In order to address these challenges, it is necessary to explore the field of dynamic micro expression recognition (DMER) using multimodal fusion techniques, with special attention to the diverse fusion of temporal and spatial modal features. In this paper, we propose a novel Temporal and Spatial feature Fusion framework for DMER (TSFmicro). This framework integrates a Retention Network (RetNet) and a transformer-based DMER network, with the objective of efficient micro-expression recognition through the capture and fusion of temporal and spatial relations. Meanwhile, we propose a novel parallel time-space fusion method from the perspective of modal fusion, which fuses spatio-temporal information in high-dimensional feature space, resulting in complementary "where-how" relationships at the semantic level and providing richer semantic information for the model. The experimental results demonstrate the superior performance of the TSFmicro method in comparison to other contemporary state-of-the-art methods. This is evidenced by its effectiveness on three well-recognised micro-expression datasets. 

**Abstract (ZH)**: 基于多模态融合的动态微表情识别的时空特征融合框架（TSFmicro） 

---
# Multimodal Generative AI for Story Point Estimation in Software Development 

**Title (ZH)**: 多模态生成AI在软件开发中的故事点估算 

**Authors**: Mohammad Rubyet Islam, Peter Sandborn  

**Link**: [PDF](https://arxiv.org/pdf/2505.16290)  

**Abstract**: This research explores the application of Multimodal Generative AI to enhance story point estimation in Agile software development. By integrating text, image, and categorical data using advanced models like BERT, CNN, and XGBoost, our approach surpasses the limitations of traditional single-modal estimation methods. The results demonstrate strong accuracy for simpler story points, while also highlighting challenges in more complex categories due to data imbalance. This study further explores the impact of categorical data, particularly severity, on the estimation process, emphasizing its influence on model performance. Our findings emphasize the transformative potential of multimodal data integration in refining AI-driven project management, paving the way for more precise, adaptable, and domain-specific AI capabilities. Additionally, this work outlines future directions for addressing data variability and enhancing the robustness of AI in Agile methodologies. 

**Abstract (ZH)**: 这项研究探索了多模态生成AI在敏捷软件开发中增强故事点估计的应用。通过使用BERT、CNN和XGBoost等先进模型整合文本、图像和分类数据，我们的方法超越了传统单模态估计方法的局限性。结果表明，对于较简单的故事点，我们的方法具有很强的准确性，而对于更复杂的类别，则由于数据不平衡也揭示了一些挑战。本研究进一步探讨了分类数据，尤其是严重性，对估计过程的影响，强调了其对模型性能的影响。我们的研究结果强调了多模态数据整合在细化AI驱动项目管理方面的变革潜力，为更精确、更具适应性和领域特定的AI能力铺平了道路。此外，这项工作还概述了未来方向，以应对数据变化性并增强敏捷方法中AI的稳健性。 

---
# IRONIC: Coherence-Aware Reasoning Chains for Multi-Modal Sarcasm Detection 

**Title (ZH)**: IRONIC: 具有连贯性意识的多模态讽刺检测推理链 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16258)  

**Abstract**: Interpreting figurative language such as sarcasm across multi-modal inputs presents unique challenges, often requiring task-specific fine-tuning and extensive reasoning steps. However, current Chain-of-Thought approaches do not efficiently leverage the same cognitive processes that enable humans to identify sarcasm. We present IRONIC, an in-context learning framework that leverages Multi-modal Coherence Relations to analyze referential, analogical and pragmatic image-text linkages. Our experiments show that IRONIC achieves state-of-the-art performance on zero-shot Multi-modal Sarcasm Detection across different baselines. This demonstrates the need for incorporating linguistic and cognitive insights into the design of multi-modal reasoning strategies. Our code is available at: this https URL 

**Abstract (ZH)**: 跨多模态输入解释比喻语言如讽刺带来了独特挑战，通常需要特定任务的微调和广泛的推理步骤。然而，当前的链式思考方法并未高效利用人类识别讽刺的认知过程。我们提出了IRONIC，一种基于多模态一致性关系的上下文学习框架，用于分析参照性、类比性和语用性的图文关联。我们的实验结果显示，IRONIC在零样本多模态讽刺检测方面实现了最佳性能，这表明需要将语言和认知洞察力纳入多模态推理策略的设计中。我们的代码可在以下链接获取：this https URL 

---
# DualComp: End-to-End Learning of a Unified Dual-Modality Lossless Compressor 

**Title (ZH)**: DualComp: 统一无损双模态端到端压缩学习 

**Authors**: Yan Zhao, Zhengxue Cheng, Junxuan Zhang, Qunshan Gu, Qi Wang, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.16256)  

**Abstract**: Most learning-based lossless compressors are designed for a single modality, requiring separate models for multi-modal data and lacking flexibility. However, different modalities vary significantly in format and statistical properties, making it ineffective to use compressors that lack modality-specific adaptations. While multi-modal large language models (MLLMs) offer a potential solution for modality-unified compression, their excessive complexity hinders practical deployment. To address these challenges, we focus on the two most common modalities, image and text, and propose DualComp, the first unified and lightweight learning-based dual-modality lossless compressor. Built on a lightweight backbone, DualComp incorporates three key structural enhancements to handle modality heterogeneity: modality-unified tokenization, modality-switching contextual learning, and modality-routing mixture-of-experts. A reparameterization training strategy is also used to boost compression performance. DualComp integrates both modality-specific and shared parameters for efficient parameter utilization, enabling near real-time inference (200KB/s) on desktop CPUs. With much fewer parameters, DualComp achieves compression performance on par with the SOTA LLM-based methods for both text and image datasets. Its simplified single-modality variant surpasses the previous best image compressor on the Kodak dataset by about 9% using just 1.2% of the model size. 

**Abstract (ZH)**: DualComp：一种轻量级的统一双模态无损压缩器 

---
# VLM-R$^3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought 

**Title (ZH)**: VLM-R³: 区域识别、推理和精炼以增强多模态链式思考 

**Authors**: Chaoya Jiang, Yongrui Heng, Wei Ye, Han Yang, Haiyang Xu, Ming Yan, Ji Zhang, Fei Huang, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16192)  

**Abstract**: Recently, reasoning-based MLLMs have achieved a degree of success in generating long-form textual reasoning chains. However, they still struggle with complex tasks that necessitate dynamic and iterative focusing on and revisiting of visual regions to achieve precise grounding of textual reasoning in visual evidence. We introduce \textbf{VLM-R$^3$} (\textbf{V}isual \textbf{L}anguage \textbf{M}odel with \textbf{R}egion \textbf{R}ecognition and \textbf{R}easoning), a framework that equips an MLLM with the ability to (i) decide \emph{when} additional visual evidence is needed, (ii) determine \emph{where} to ground within the image, and (iii) seamlessly weave the relevant sub-image content back into an interleaved chain-of-thought. The core of our method is \textbf{Region-Conditioned Reinforcement Policy Optimization (R-GRPO)}, a training paradigm that rewards the model for selecting informative regions, formulating appropriate transformations (e.g.\ crop, zoom), and integrating the resulting visual context into subsequent reasoning steps. To bootstrap this policy, we compile a modest but carefully curated Visuo-Lingual Interleaved Rationale (VLIR) corpus that provides step-level supervision on region selection and textual justification. Extensive experiments on MathVista, ScienceQA, and other benchmarks show that VLM-R$^3$ sets a new state of the art in zero-shot and few-shot settings, with the largest gains appearing on questions demanding subtle spatial reasoning or fine-grained visual cue extraction. 

**Abstract (ZH)**: VLM-R$^3$：视觉语言模型与区域识别及推理 

---
