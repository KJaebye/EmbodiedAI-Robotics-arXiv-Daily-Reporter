# Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation 

**Title (ZH)**: 突破SFTPlateau：多模态结构强化学习在图表到代码生成中的应用 

**Authors**: Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.13587)  

**Abstract**: While reinforcement learning (RL) has proven highly effective for general reasoning in vision-language models, its application to tasks requiring in-depth understanding of information-rich images and generation of structured outputs remains underexplored. Chart-to-code generation exemplifies this challenge, demanding complex reasoning over visual charts to generate structured code. Supervised fine-tuning (SFT) alone is often insufficient, highlighting the need for effective RL strategies that appropriately reward structured outputs. We systematically investigate the performance plateau in SFT through large-scale experiments and propose Multimodal Structured Reinforcement Learning (MSRL) for chart-to-code generation, which substantially breaks through this plateau. We construct the largest training corpus to date, containing 3 million chart-code pairs from real-world arXiv tables to mitigate simplistic patterns of prior synthetic data. Despite reaching state-of-the-art performance, our experiments show that scaling SFT data eventually hits a plateau where further increases yield negligible improvements. Our MSRL method leverages a multi-granularity structured reward system using multimodal textual and visual feedback. At the textual level, rule-based rewards validate fine-grained code details. At the visual level, model-based rewards assess structural similarity by rendering generated code into images and employing an evaluator model. We implement this within a two-stage curriculum for training stability. Results demonstrate that MSRL significantly breaks the SFT plateau, improving high-level metrics by 6.2% and 9.9% on ChartMimic and ReachQA benchmarks respectively, achieving competitive performance with advanced closed-source models. 

**Abstract (ZH)**: 强化学习（图表到代码生成的任务：突破监督微调的瓶颈 

---
# SPANER: Shared Prompt Aligner for Multimodal Semantic Representation 

**Title (ZH)**: SPANER: 共享提示对齐器用于多模态语义表示 

**Authors**: Thye Shan Ng, Caren Soyeon Han, Eun-Jung Holden  

**Link**: [PDF](https://arxiv.org/pdf/2508.13387)  

**Abstract**: Recent advances in multimodal Parameter-Efficient Fine-Tuning (PEFT) have significantly improved performance on downstream tasks such as few-shot retrieval. However, most existing approaches focus on task-specific gains while neglecting the structure of the multimodal embedding space. As a result, modality-specific representations often remain isolated, limiting cross-modal generalisation. In this work, we introduce Shared Prompt AligNER (SPANER), a modality-agnostic PEFT framework designed to embed inputs from diverse modalities into a unified semantic space. At its core, SPANER employs a shared prompt mechanism that acts as a conceptual anchor, enabling semantically related instances to converge spatially regardless of modality. This shared prompt design is inherently extensible, supporting the seamless integration of additional modalities, such as audio, without altering the core architecture. Through comprehensive experiments across vision-language and audio-visual benchmarks, SPANER demonstrates competitive few-shot retrieval performance while preserving high semantic coherence in the learned embedding space. Our results highlight the importance of aligning embedding structures, rather than merely tuning adapter weights, for scalable multimodal learning. 

**Abstract (ZH)**: Recent Advances in Modality-Agnostic Parameter-Efficient Fine-Tuning for Enhanced Few-Shot Retrieval 

---
# Towards Unified Multimodal Financial Forecasting: Integrating Sentiment Embeddings and Market Indicators via Cross-Modal Attention 

**Title (ZH)**: 统一多模态金融预测 Towards 结合情感嵌入和市场指标的跨模态注意力集成 

**Authors**: Sarthak Khanna, Armin Berger, David Berghaus, Tobias Deusser, Lorenz Sparrenberg, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13327)  

**Abstract**: We propose STONK (Stock Optimization using News Knowledge), a multimodal framework integrating numerical market indicators with sentiment-enriched news embeddings to improve daily stock-movement prediction. By combining numerical & textual embeddings via feature concatenation and cross-modal attention, our unified pipeline addresses limitations of isolated analyses. Backtesting shows STONK outperforms numeric-only baselines. A comprehensive evaluation of fusion strategies and model configurations offers evidence-based guidance for scalable multimodal financial forecasting. Source code is available on GitHub 

**Abstract (ZH)**: STONK：基于新闻知识的股票优化模型整合数值市场指标与情感丰富的新闻嵌入以改进日度股票动量预测 

---
# CardAIc-Agents: A Multimodal Framework with Hierarchical Adaptation for Cardiac Care Support 

**Title (ZH)**: CardAIc-Agents：一种具有层次适应性的多模态心脏护理支持框架 

**Authors**: Yuting Zhang, Karina V. Bunting, Asgher Champsi, Xiaoxia Wang, Wenqi Lu, Alexander Thorley, Sandeep S Hothi, Zhaowen Qiu, Dipak Kotecha, Jinming Duan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13256)  

**Abstract**: Cardiovascular diseases (CVDs) remain the foremost cause of mortality worldwide, a burden worsened by a severe deficit of healthcare workers. Artificial intelligence (AI) agents have shown potential to alleviate this gap via automated early detection and proactive screening, yet their clinical application remains limited by: 1) prompt-based clinical role assignment that relies on intrinsic model capabilities without domain-specific tool support; or 2) rigid sequential workflows, whereas clinical care often requires adaptive reasoning that orders specific tests and, based on their results, guides personalised next steps; 3) general and static knowledge bases without continuous learning capability; and 4) fixed unimodal or bimodal inputs and lack of on-demand visual outputs when further clarification is needed. In response, a multimodal framework, CardAIc-Agents, was proposed to augment models with external tools and adaptively support diverse cardiac tasks. Specifically, a CardiacRAG agent generated general plans from updatable cardiac knowledge, while the chief agent integrated tools to autonomously execute these plans and deliver decisions. To enable adaptive and case-specific customization, a stepwise update strategy was proposed to dynamically refine plans based on preceding execution results, once the task was assessed as complex. In addition, a multidisciplinary discussion tool was introduced to interpret challenging cases, thereby supporting further adaptation. When clinicians raised concerns, visual review panels were provided to assist final validation. Experiments across three datasets showed the efficiency of CardAIc-Agents compared to mainstream Vision-Language Models (VLMs), state-of-the-art agentic systems, and fine-tuned VLMs. 

**Abstract (ZH)**: 心血管疾病（CVDs）仍然是全球最主要的致死原因，这一负担因医疗工作者严重短缺而加剧。人工智能（AI）代理显示出通过自动化早期检测和主动筛查来缓解这一差距的潜力，但其临床应用仍然受限于：1）依赖内在模型能力而非特定领域工具支持的指令驱动临床角色分配；或2）刚性的顺序工作流程，而临床护理往往需要适应性推理，根据特定测试的结果来指导个性化后续步骤；3）通用且静态的知识库，缺乏持续学习能力；以及4）固定的单模态或双模态输入，缺乏在需要进一步澄清时的即需视觉输出。为此，提出了一种多模态框架CardAIc-Agents，以增强模型并与外部工具结合，适应性地支持多种心脏任务。具体而言，CardiacRAG代理从可更新的心脏知识中生成通用计划，而主代理集成工具以自主执行这些计划并提供决策。为实现适应性和个案特定的定制，提出了一步步更新策略，根据先前执行结果动态细化计划，一旦任务被评估为复杂。此外，还引入了多学科讨论工具来解释具有挑战性的情况，从而支持进一步适应。当临床医生提出顾虑时，提供可视化审查小组以协助最终验证。跨三个数据集的实验显示，CardAIc-Agents在效率上优于主流的视觉-语言模型（VLMs）、最先进的代理系统以及微调后的VLMs。 

---
# Categorical Policies: Multimodal Policy Learning and Exploration in Continuous Control 

**Title (ZH)**: 分类策略：连续控制中的多模态策略学习与探索 

**Authors**: SM Mazharul Islam, Manfred Huber  

**Link**: [PDF](https://arxiv.org/pdf/2508.13922)  

**Abstract**: A policy in deep reinforcement learning (RL), either deterministic or stochastic, is commonly parameterized as a Gaussian distribution alone, limiting the learned behavior to be unimodal. However, the nature of many practical decision-making problems favors a multimodal policy that facilitates robust exploration of the environment and thus to address learning challenges arising from sparse rewards, complex dynamics, or the need for strategic adaptation to varying contexts. This issue is exacerbated in continuous control domains where exploration usually takes place in the vicinity of the predicted optimal action, either through an additive Gaussian noise or the sampling process of a stochastic policy. In this paper, we introduce Categorical Policies to model multimodal behavior modes with an intermediate categorical distribution, and then generate output action that is conditioned on the sampled mode. We explore two sampling schemes that ensure differentiable discrete latent structure while maintaining efficient gradient-based optimization. By utilizing a latent categorical distribution to select the behavior mode, our approach naturally expresses multimodality while remaining fully differentiable via the sampling tricks. We evaluate our multimodal policy on a set of DeepMind Control Suite environments, demonstrating that through better exploration, our learned policies converge faster and outperform standard Gaussian policies. Our results indicate that the Categorical distribution serves as a powerful tool for structured exploration and multimodal behavior representation in continuous control. 

**Abstract (ZH)**: 一种基于深度强化学习的分类策略：模型多模态行为并促进高效可微优化 

---
# UniECS: Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion 

**Title (ZH)**: UniECS：统一的多模态电商搜索框架，带有门控跨模态融合 

**Authors**: Zihan Liang, Yufei Ma, ZhiPeng Qian, Huangyu Dai, Zihan Wang, Ben Chen, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13843)  

**Abstract**: Current e-commerce multimodal retrieval systems face two key limitations: they optimize for specific tasks with fixed modality pairings, and lack comprehensive benchmarks for evaluating unified retrieval approaches. To address these challenges, we introduce UniECS, a unified multimodal e-commerce search framework that handles all retrieval scenarios across image, text, and their combinations. Our work makes three key contributions. First, we propose a flexible architecture with a novel gated multimodal encoder that uses adaptive fusion mechanisms. This encoder integrates different modality representations while handling missing modalities. Second, we develop a comprehensive training strategy to optimize learning. It combines cross-modal alignment loss (CMAL), cohesive local alignment loss (CLAL), intra-modal contrastive loss (IMCL), and adaptive loss weighting. Third, we create M-BEER, a carefully curated multimodal benchmark containing 50K product pairs for e-commerce search evaluation. Extensive experiments demonstrate that UniECS consistently outperforms existing methods across four e-commerce benchmarks with fine-tuning or zero-shot evaluation. On our M-BEER bench, UniECS achieves substantial improvements in cross-modal tasks (up to 28\% gain in R@10 for text-to-image retrieval) while maintaining parameter efficiency (0.2B parameters) compared to larger models like GME-Qwen2VL (2B) and MM-Embed (8B). Furthermore, we deploy UniECS in the e-commerce search platform of Kuaishou Inc. across two search scenarios, achieving notable improvements in Click-Through Rate (+2.74\%) and Revenue (+8.33\%). The comprehensive evaluation demonstrates the effectiveness of our approach in both experimental and real-world settings. Corresponding codes, models and datasets will be made publicly available at this https URL. 

**Abstract (ZH)**: 当前的电子商务多模态检索系统面临两大关键限制：它们针对特定任务进行优化并采用固定模态配对，缺乏评估统一检索方法的全面基准。为了解决这些挑战，我们引入了UniECS，一个统一的电子商务多模态搜索框架，可以处理图像、文本及其组合的所有检索场景。我们的工作做出了三项关键贡献。首先，我们提出了一种灵活的架构，其中包含一种新型门控多模态编码器，使用自适应融合机制。该编码器在处理缺失模态的同时整合了不同的模态表示。其次，我们开发了一种全面的训练策略来优化学习。该策略结合了跨模态对齐损失（CMAL）、协同局部对齐损失（CLAL）、模内对比损失（IMCL）以及自适应损失加权。第三，我们创建了M-BEER，一个精心编 curated 的多模态基准，包含50K个产品对，用于电子商务搜索评估。广泛的实验表明，UniECS在四种电子商务基准上的微调或零-shot评估中始终优于现有方法。在我们的M-BEER基准上，UniECS在跨模态任务中取得了显著改进（文本到图像检索的R@10增益高达28%），同时保持了参数效率（0.2B参数），比GME-Qwen2VL（2B）和MM-Embed（8B）等大型模型更具优势。此外，我们在快手公司两个搜索场景下的电子商务搜索平台上部署了UniECS，实现了点击率 (+2.74%) 和收入 (+8.33%) 的显著提升。全面的评估证明了我们方法在实验和现实世界设置中的有效性。相关代码、模型和数据集将在以下网址公开：this https URL。 

---
# A Fully Transformer Based Multimodal Framework for Explainable Cancer Image Segmentation Using Radiology Reports 

**Title (ZH)**: 基于完全变换器的多模态解释性癌症图像分割框架，结合放射学报告 

**Authors**: Enobong Adahada, Isabel Sassoon, Kate Hone, Yongmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13796)  

**Abstract**: We introduce Med-CTX, a fully transformer based multimodal framework for explainable breast cancer ultrasound segmentation. We integrate clinical radiology reports to boost both performance and interpretability. Med-CTX achieves exact lesion delineation by using a dual-branch visual encoder that combines ViT and Swin transformers, as well as uncertainty aware fusion. Clinical language structured with BI-RADS semantics is encoded by BioClinicalBERT and combined with visual features utilising cross-modal attention, allowing the model to provide clinically grounded, model generated explanations. Our methodology generates segmentation masks, uncertainty maps, and diagnostic rationales all at once, increasing confidence and transparency in computer assisted diagnosis. On the BUS-BRA dataset, Med-CTX achieves a Dice score of 99% and an IoU of 95%, beating existing baselines U-Net, ViT, and Swin. Clinical text plays a key role in segmentation accuracy and explanation quality, as evidenced by ablation studies that show a -5.4% decline in Dice score and -31% in CIDEr. Med-CTX achieves good multimodal alignment (CLIP score: 85%) and increased confi dence calibration (ECE: 3.2%), setting a new bar for trustworthy, multimodal medical architecture. 

**Abstract (ZH)**: Med-CTX：基于Transformer的多模态可解释乳腺癌超声分割框架 

---
# End-to-End Audio-Visual Learning for Cochlear Implant Sound Coding in Noisy Environments 

**Title (ZH)**: 噪声环境中的端到端音频-视觉学习在植入式耳蜗声音编码中的应用 

**Authors**: Meng-Ping Lin, Enoch Hsin-Ho Huang, Shao-Yi Chien, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13576)  

**Abstract**: The cochlear implant (CI) is a remarkable biomedical device that successfully enables individuals with severe-to-profound hearing loss to perceive sound by converting speech into electrical stimulation signals. Despite advancements in the performance of recent CI systems, speech comprehension in noisy or reverberant conditions remains a challenge. Recent and ongoing developments in deep learning reveal promising opportunities for enhancing CI sound coding capabilities, not only through replicating traditional signal processing methods with neural networks, but also through integrating visual cues as auxiliary data for multimodal speech processing. Therefore, this paper introduces a novel noise-suppressing CI system, AVSE-ECS, which utilizes an audio-visual speech enhancement (AVSE) model as a pre-processing module for the deep-learning-based ElectrodeNet-CS (ECS) sound coding strategy. Specifically, a joint training approach is applied to model AVSE-ECS, an end-to-end CI system. Experimental results indicate that the proposed method outperforms the previous ECS strategy in noisy conditions, with improved objective speech intelligibility scores. The methods and findings in this study demonstrate the feasibility and potential of using deep learning to integrate the AVSE module into an end-to-end CI system 

**Abstract (ZH)**: 基于音频-视觉增强的深度学习 cochlear implant 系统：AVSE-ECS 

---
# CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving 

**Title (ZH)**: CORENet：具有LiDAR监督的跨模态4D雷达降噪网络 

**Authors**: Fuyang Liu, Jilin Mei, Fangyuan Mao, Chen Min, Yan Xing, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13485)  

**Abstract**: 4D radar-based object detection has garnered great attention for its robustness in adverse weather conditions and capacity to deliver rich spatial information across diverse driving scenarios. Nevertheless, the sparse and noisy nature of 4D radar point clouds poses substantial challenges for effective perception. To address the limitation, we present CORENet, a novel cross-modal denoising framework that leverages LiDAR supervision to identify noise patterns and extract discriminative features from raw 4D radar data. Designed as a plug-and-play architecture, our solution enables seamless integration into voxel-based detection frameworks without modifying existing pipelines. Notably, the proposed method only utilizes LiDAR data for cross-modal supervision during training while maintaining full radar-only operation during inference. Extensive evaluation on the challenging Dual-Radar dataset, which is characterized by elevated noise level, demonstrates the effectiveness of our framework in enhancing detection robustness. Comprehensive experiments validate that CORENet achieves superior performance compared to existing mainstream approaches. 

**Abstract (ZH)**: 基于4D雷达的对象检测由于其在恶劣天气条件下的鲁棒性和跨多种驾驶场景提供丰富空间信息的能力而引起了广泛关注。然而，4D雷达点云的稀疏性和噪声性给有效的感知带来了重大挑战。为此，我们提出了一种名为CORENet的新型跨模态去噪框架，该框架利用LiDAR监督来识别噪声模式并从原始4D雷达数据中提取特征。设计为即插即用架构，我们的解决方案能够无缝集成到体素基检测框架中而无需修改现有管线。值得注意的是，所提出的方法仅在训练过程中利用LiDAR数据进行跨模态监督，在推理过程中保持纯雷达操作。在具有高噪声水平的Dual-Radar数据集上的广泛评估表明，该框架在提高检测鲁棒性方面效果显著。全面的实验验证了CORENet相比现有主流方法具有更好的性能。 

---
# STER-VLM: Spatio-Temporal With Enhanced Reference Vision-Language Models 

**Title (ZH)**: STER-VLM: 空间-时间增强参考的视觉-语言模型 

**Authors**: Tinh-Anh Nguyen-Nhu, Triet Dao Hoang Minh, Dat To-Thanh, Phuc Le-Gia, Tuan Vo-Lan, Tien-Huy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13470)  

**Abstract**: Vision-language models (VLMs) have emerged as powerful tools for enabling automated traffic analysis; however, current approaches often demand substantial computational resources and struggle with fine-grained spatio-temporal understanding. This paper introduces STER-VLM, a computationally efficient framework that enhances VLM performance through (1) caption decomposition to tackle spatial and temporal information separately, (2) temporal frame selection with best-view filtering for sufficient temporal information, and (3) reference-driven understanding for capturing fine-grained motion and dynamic context and (4) curated visual/textual prompt techniques. Experimental results on the WTS \cite{kong2024wts} and BDD \cite{BDD} datasets demonstrate substantial gains in semantic richness and traffic scene interpretation. Our framework is validated through a decent test score of 55.655 in the AI City Challenge 2025 Track 2, showing its effectiveness in advancing resource-efficient and accurate traffic analysis for real-world applications. 

**Abstract (ZH)**: 基于视觉-语言模型的时空增强框架（STER-VLM）：提升计算效率的交通分析技术 

---
# MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents 

**Title (ZH)**: MM-BrowseComp: 多模态浏览代理的综合基准 

**Authors**: Shilong Li, Xingyuan Bu, Wenjie Wang, Jiaheng Liu, Jun Dong, Haoyang He, Hao Lu, Haozhe Zhang, Chenchen Jing, Zhen Li, Chuanhao Li, Jiayi Tian, Chenchen Zhang, Tianhao Peng, Yancheng He, Jihao Gu, Yuanxing Zhang, Jian Yang, Ge Zhang, Wenhao Huang, Wangchunshu Zhou, Zhaoxiang Zhang, Ruizhe Ding, Shilei Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13186)  

**Abstract**: AI agents with advanced reasoning and tool use capabilities have demonstrated impressive performance in web browsing for deep search. While existing benchmarks such as BrowseComp evaluate these browsing abilities, they primarily focus on textual information, overlooking the prevalence of multimodal content. To bridge this gap, we introduce MM-BrowseComp, a novel benchmark comprising 224 challenging, hand-crafted questions specifically designed to assess agents' multimodal retrieval and reasoning capabilities. These questions often incorporate images in prompts, and crucial information encountered during the search and reasoning process may also be embedded within images or videos on webpages. Consequently, methods relying solely on text prove insufficient for our benchmark. Additionally, we provide a verified checklist for each question, enabling fine-grained analysis of multimodal dependencies and reasoning paths. Our comprehensive evaluation of state-of-the-art models on MM-BrowseComp reveals that even top models like OpenAI o3 with tools achieve only 29.02\% accuracy, highlighting the suboptimal multimodal capabilities and lack of native multimodal reasoning in current models. 

**Abstract (ZH)**: 具有高级推理和工具使用能力的AI代理在深度网页搜索中表现出色。现有基准如BrowseComp评估这些浏览能力，但主要侧重于文本信息，忽视了多模态内容的普遍性。为弥补这一差距，我们介绍了MM-BrowseComp，这是一个包含224个具有挑战性的手动生成问题的新基准，旨在评估代理的多模态检索和推理能力。这些问题通常在提示中包含图像，而在搜索和推理过程中也可能从网页上的图像或视频中提取关键信息。因此，仅依赖文本的方法对我们的基准证明是不够的。此外，我们还为每个问题提供了验证列表，以支持对多模态依赖性和推理路径的精细分析。对MM-BrowseComp上最先进的模型的全面评估表明，即使是像OpenAI o3这样的顶级模型在工具辅助下的准确率也只有29.02%，这突显了当前模型在多模态能力和原生多模态推理方面的不足。 

---
