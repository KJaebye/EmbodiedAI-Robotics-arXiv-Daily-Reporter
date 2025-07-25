# EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos 

**Title (ZH)**: 自我视角多模态模型：从自我中心人体视频中学习视觉-语言-行动模型 

**Authors**: Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12440)  

**Abstract**: Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Isaac Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Isaac Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: this https URL 

**Abstract (ZH)**: 真实机器人数据收集用于模仿学习已推动了机器人操作的显著进步。然而，过程中对机器人硬件的需要从根本上限制了数据的规模。本文探讨使用自视点人类视频训练视觉-语言-动作（VLA）模型。使用人类视频的优势不仅在于其规模，更在于其丰富的场景和任务。通过使用训练于人类视频且能预测人类手腕和手部动作的VLA，我们执行逆向运动学和重新目标化，将人类动作转换为机器人动作。我们使用少量的机器人操作演示微调模型，以获得机器人策略，即EgoVLA。我们提出了一种模拟基准——Isaac类人机器人操作基准，其中设计了多样化的双臂操作任务并提供了演示。我们使用Isaac类人机器人操作基准微调和评估EgoVLA，并展示了相对于基线的显著改进，并消融了人类数据的重要性。有关视频请参阅我们的网站：this https URL。 

---
# Multimodal Coordinated Online Behavior: Trade-offs and Strategies 

**Title (ZH)**: 多模态协调在线行为：权衡与策略 

**Authors**: Lorenzo Mannocci, Stefano Cresci, Matteo Magnani, Anna Monreale, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12108)  

**Abstract**: Coordinated online behavior, which spans from beneficial collective actions to harmful manipulation such as disinformation campaigns, has become a key focus in digital ecosystem analysis. Traditional methods often rely on monomodal approaches, focusing on single types of interactions like co-retweets or co-hashtags, or consider multiple modalities independently of each other. However, these approaches may overlook the complex dynamics inherent in multimodal coordination. This study compares different ways of operationalizing the detection of multimodal coordinated behavior. It examines the trade-off between weakly and strongly integrated multimodal models, highlighting the balance between capturing broader coordination patterns and identifying tightly coordinated behavior. By comparing monomodal and multimodal approaches, we assess the unique contributions of different data modalities and explore how varying implementations of multimodality impact detection outcomes. Our findings reveal that not all the modalities provide distinct insights, but that with a multimodal approach we can get a more comprehensive understanding of coordination dynamics. This work enhances the ability to detect and analyze coordinated online behavior, offering new perspectives for safeguarding the integrity of digital platforms. 

**Abstract (ZH)**: 跨模态协调网络行为，从有益的集体行动到有害的操纵如信息操纵活动，已成为数字生态系统分析中的关键重点。传统方法往往依赖于单一模态的方法，关注单一类型的交互，如共转发或共标签，或者单独考虑多种模态。然而，这些方法可能忽视了跨模态协调中存在的复杂动态。本研究比较了不同检测跨模态协调行为的方法。它探讨了弱集成与强集成跨模态模型之间的权衡，突出捕获更广泛协调模式与识别紧密协调行为之间的平衡。通过比较单模态和跨模态方法，我们评估了不同数据模态的独特贡献，并探讨了不同跨模态实现对检测结果的影响。我们的研究发现，并非所有模态都提供独特见解，但通过跨模态方法可以获得对协调动态的更全面理解。这项工作增强了检测和分析网络协调行为的能力，为保护数字平台的完整性提供了新的视角。 

---
# RaDL: Relation-aware Disentangled Learning for Multi-Instance Text-to-Image Generation 

**Title (ZH)**: 基于关系感知的解耦学习多实例文本到图像生成 

**Authors**: Geon Park, Seon Bin Kim, Gunho Jung, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.11947)  

**Abstract**: With recent advancements in text-to-image (T2I) models, effectively generating multiple instances within a single image prompt has become a crucial challenge. Existing methods, while successful in generating positions of individual instances, often struggle to account for relationship discrepancy and multiple attributes leakage. To address these limitations, this paper proposes the relation-aware disentangled learning (RaDL) framework. RaDL enhances instance-specific attributes through learnable parameters and generates relation-aware image features via Relation Attention, utilizing action verbs extracted from the global prompt. Through extensive evaluations on benchmarks such as COCO-Position, COCO-MIG, and DrawBench, we demonstrate that RaDL outperforms existing methods, showing significant improvements in positional accuracy, multiple attributes consideration, and the relationships between instances. Our results present RaDL as the solution for generating images that consider both the relationships and multiple attributes of each instance within the multi-instance image. 

**Abstract (ZH)**: 基于文本到图像模型的近期进展，有效地在单个图像提示中生成多个实例已成为一个关键挑战。现有方法虽然在生成单个实例的位置方面取得成功，但在处理关系差异和多属性泄漏方面往往存在局限。为了解决这些限制，本文提出了关系感知分离学习（RaDL）框架。RaDL 通过可学习参数增强实例特定属性，并利用从全局提示中提取的动作动词通过关系注意力生成关系感知的图像特征。通过在 COCO-Position、COCO-MIG 和 DrawBench 等基准上的广泛评估，我们展示了 RaDL 在位置准确性、多属性考虑及实例间关系方面优于现有方法。我们的结果表明，RaDL 是生成同时考虑每个实例关系和多属性的多实例图像的解决方案。 

---
# From Coarse to Nuanced: Cross-Modal Alignment of Fine-Grained Linguistic Cues and Visual Salient Regions for Dynamic Emotion Recognition 

**Title (ZH)**: 从粗略到细腻：细粒度语言线索与视觉显著区域的跨模态对齐及其在动态情感识别中的应用 

**Authors**: Yu Liu, Leyuan Qu, Hanlei Shi, Di Gao, Yuhua Zheng, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11892)  

**Abstract**: Dynamic Facial Expression Recognition (DFER) aims to identify human emotions from temporally evolving facial movements and plays a critical role in affective computing. While recent vision-language approaches have introduced semantic textual descriptions to guide expression recognition, existing methods still face two key limitations: they often underutilize the subtle emotional cues embedded in generated text, and they have yet to incorporate sufficiently effective mechanisms for filtering out facial dynamics that are irrelevant to emotional expression. To address these gaps, We propose GRACE, Granular Representation Alignment for Cross-modal Emotion recognition that integrates dynamic motion modeling, semantic text refinement, and token-level cross-modal alignment to facilitate the precise localization of emotionally salient spatiotemporal features. Our method constructs emotion-aware textual descriptions via a Coarse-to-fine Affective Text Enhancement (CATE) module and highlights expression-relevant facial motion through a motion-difference weighting mechanism. These refined semantic and visual signals are aligned at the token level using entropy-regularized optimal transport. Experiments on three benchmark datasets demonstrate that our method significantly improves recognition performance, particularly in challenging settings with ambiguous or imbalanced emotion classes, establishing new state-of-the-art (SOTA) results in terms of both UAR and WAR. 

**Abstract (ZH)**: 粒度表示对齐的跨模态情感识别（GRACE）：整合动态运动建模、语义文本精炼和标记级跨模态对齐 

---
# Seeing the Signs: A Survey of Edge-Deployable OCR Models for Billboard Visibility Analysis 

**Title (ZH)**: 识读标识：边缘部署的OCR模型在户外广告可见性分析中的综述 

**Authors**: Maciej Szankin, Vidhyananth Venkatasamy, Lihang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2507.11730)  

**Abstract**: Outdoor advertisements remain a critical medium for modern marketing, yet accurately verifying billboard text visibility under real-world conditions is still challenging. Traditional Optical Character Recognition (OCR) pipelines excel at cropped text recognition but often struggle with complex outdoor scenes, varying fonts, and weather-induced visual noise. Recently, multimodal Vision-Language Models (VLMs) have emerged as promising alternatives, offering end-to-end scene understanding with no explicit detection step. This work systematically benchmarks representative VLMs - including Qwen 2.5 VL 3B, InternVL3, and SmolVLM2 - against a compact CNN-based OCR baseline (PaddleOCRv4) across two public datasets (ICDAR 2015 and SVT), augmented with synthetic weather distortions to simulate realistic degradation. Our results reveal that while selected VLMs excel at holistic scene reasoning, lightweight CNN pipelines still achieve competitive accuracy for cropped text at a fraction of the computational cost-an important consideration for edge deployment. To foster future research, we release our weather-augmented benchmark and evaluation code publicly. 

**Abstract (ZH)**: 户外广告仍然是现代营销中重要的媒体渠道，但准确验证实际条件下广告牌文字的可见性仍然具有挑战性。传统光学字符识别（OCR）流水线在裁剪文本识别方面表现出色，但在处理复杂户外场景、变化的字体和天气引起的视觉噪声方面常常遇到困难。近年来，多模态视觉-语言模型（VLMs）作为一种有前途的替代方案出现，能够实现端到端的场景理解，无需显式的检测步骤。本研究系统性地将Qwen 2.5 VL 3B、InternVL3和SmolVLM2等代表性VLM与基于紧凑CNN的OCR baselime（PaddleOCRv4）在两个公开数据集（ICDAR 2015和SVT）上进行了基准测试，数据集经过合成天气失真增强以模拟真实的退化。我们的结果显示，虽然选定的VLM在整体场景推理方面表现出色，但轻量级的CNN流水线在计算成本仅为一小部分的情况下仍能获得竞争力的文字识别准确性，这对于边缘部署而言非常重要。为了促进未来的研究，我们公开发布了带有天气增强的基准测试和评估代码。 

---
# ExpliCIT-QA: Explainable Code-Based Image Table Question Answering 

**Title (ZH)**: ExpliCIT-QA: 可解释的基于代码的图像表格问答 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Pedro Alonso Doval, Jorge Alcalde Vesteiro, Héctor Cerezo-Costas  

**Link**: [PDF](https://arxiv.org/pdf/2507.11694)  

**Abstract**: We present ExpliCIT-QA, a system that extends our previous MRT approach for tabular question answering into a multimodal pipeline capable of handling complex table images and providing explainable answers. ExpliCIT-QA follows a modular design, consisting of: (1) Multimodal Table Understanding, which uses a Chain-of-Thought approach to extract and transform content from table images; (2) Language-based Reasoning, where a step-by-step explanation in natural language is generated to solve the problem; (3) Automatic Code Generation, where Python/Pandas scripts are created based on the reasoning steps, with feedback for handling errors; (4) Code Execution to compute the final answer; and (5) Natural Language Explanation that describes how the answer was computed. The system is built for transparency and auditability: all intermediate outputs, parsed tables, reasoning steps, generated code, and final answers are available for inspection. This strategy works towards closing the explainability gap in end-to-end TableVQA systems. We evaluated ExpliCIT-QA on the TableVQA-Bench benchmark, comparing it with existing baselines. We demonstrated improvements in interpretability and transparency, which open the door for applications in sensitive domains like finance and healthcare where auditing results are critical. 

**Abstract (ZH)**: ExpliCIT-QA：一个扩展我们的先前MRT表格式问答方法的多模态解释性问答系统 

---
# Partitioner Guided Modal Learning Framework 

**Title (ZH)**: 分区辅助模态学习框架 

**Authors**: Guimin Hu, Yi Xin, Lijie Hu, Zhihong Zhu, Hasti Seifi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11661)  

**Abstract**: Multimodal learning benefits from multiple modal information, and each learned modal representations can be divided into uni-modal that can be learned from uni-modal training and paired-modal features that can be learned from cross-modal interaction. Building on this perspective, we propose a partitioner-guided modal learning framework, PgM, which consists of the modal partitioner, uni-modal learner, paired-modal learner, and uni-paired modal decoder. Modal partitioner segments the learned modal representation into uni-modal and paired-modal features. Modal learner incorporates two dedicated components for uni-modal and paired-modal learning. Uni-paired modal decoder reconstructs modal representation based on uni-modal and paired-modal features. PgM offers three key benefits: 1) thorough learning of uni-modal and paired-modal features, 2) flexible distribution adjustment for uni-modal and paired-modal representations to suit diverse downstream tasks, and 3) different learning rates across modalities and partitions. Extensive experiments demonstrate the effectiveness of PgM across four multimodal tasks and further highlight its transferability to existing models. Additionally, we visualize the distribution of uni-modal and paired-modal features across modalities and tasks, offering insights into their respective contributions. 

**Abstract (ZH)**: 多模态学习受益于多种模态信息，每种学习到的模态表示可以分为仅从单模训练中学习到的单模特征和通过跨模态交互学习到的配对模态特征。基于这一视角，我们提出了一种分区引导的模态学习框架PgM，该框架包括模态分区器、单模特征学习器、配对模态学习器和单配对模态解码器。模态分区器将学习到的模态表示分割为单模特征和配对模态特征。模态学习器包含两个专门用于单模和配对模态学习的组件。单配对模态解码器基于单模和配对模态特征重构模态表示。PgM提供了三个关键优势：1）全面学习单模和配对模态特征，2）灵活调整单模和配对模态表示的分布以适应各种下游任务，3）不同模态和分区的学习率。广泛实验证明了PgM在四种多模态任务中的有效性，并进一步突显了其对现有模型的迁移性。此外，我们还可视化了单模和配对模态特征在不同模态和任务中的分布，提供了它们各自贡献的见解。 

---
