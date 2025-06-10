# Enhancing Situational Awareness in Underwater Robotics with Multi-modal Spatial Perception 

**Title (ZH)**: 基于多模态空间感知增强水下机器人的情境感知 

**Authors**: Pushyami Kaveti, Ambjorn Grimsrud Waldum, Hanumant Singh, Martin Ludvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06476)  

**Abstract**: Autonomous Underwater Vehicles (AUVs) and Remotely Operated Vehicles (ROVs) demand robust spatial perception capabilities, including Simultaneous Localization and Mapping (SLAM), to support both remote and autonomous tasks. Vision-based systems have been integral to these advancements, capturing rich color and texture at low cost while enabling semantic scene understanding. However, underwater conditions -- such as light attenuation, backscatter, and low contrast -- often degrade image quality to the point where traditional vision-based SLAM pipelines fail. Moreover, these pipelines typically rely on monocular or stereo inputs, limiting their scalability to the multi-camera configurations common on many vehicles. To address these issues, we propose to leverage multi-modal sensing that fuses data from multiple sensors-including cameras, inertial measurement units (IMUs), and acoustic devices-to enhance situational awareness and enable robust, real-time SLAM. We explore both geometric and learning-based techniques along with semantic analysis, and conduct experiments on the data collected from a work-class ROV during several field deployments in the Trondheim Fjord. Through our experimental results, we demonstrate the feasibility of real-time reliable state estimation and high-quality 3D reconstructions in visually challenging underwater conditions. We also discuss system constraints and identify open research questions, such as sensor calibration, limitations with learning-based methods, that merit further exploration to advance large-scale underwater operations. 

**Abstract (ZH)**: 自主水下车辆(AUVs)和遥控水下车辆(ROVs)需要 robust 空间感知能力，包括同时定位与映射(SLAM)，以支持远程和自主任务。视觉系统一直是这些进步的核心，能够以低成本捕捉丰富的颜色和纹理，同时实现语义场景理解。然而，水下条件，如光线衰减、后向散射和低对比度，往往严重恶化图像质量，使传统基于视觉的SLAM流水线失效。此外，这些流水线通常依赖于单目或双目输入，限制了其在许多水下车辆上常见的多相机配置中的可扩展性。为了解决这些问题，我们提出利用多模态传感，融合来自多个传感器（包括摄像头、惯性测量单元(IMUs)和声学设备）的数据，以增强态势感知并实现稳健的实时SLAM。我们探讨了几何技术和基于学习的方法以及语义分析，并在Trondheim峡湾进行的几次现场部署中收集的工作类ROV数据上进行了实验。通过实验结果，我们证明了在视觉挑战的水下条件下进行实时可靠的状态估计和高质量3D重建的可行性。我们还讨论了系统约束并指出了需要进一步探索的研究问题，例如传感器校准，基于学习的方法的限制，以促进大规模水下操作的发展。 

---
# GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior 

**Title (ZH)**: GUI-反思：赋予多模态GUI模型自我反思行为能力 

**Authors**: Penghao Wu, Shengnan Ma, Bo Wang, Jiaheng Yu, Lewei Lu, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08012)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great potential in revolutionizing Graphical User Interface (GUI) automation. However, existing GUI models mostly rely on learning from nearly error-free offline trajectories, thus lacking reflection and error recovery capabilities. To bridge this gap, we propose GUI-Reflection, a novel framework that explicitly integrates self-reflection and error correction capabilities into end-to-end multimodal GUI models throughout dedicated training stages: GUI-specific pre-training, offline supervised fine-tuning (SFT), and online reflection tuning. GUI-reflection enables self-reflection behavior emergence with fully automated data generation and learning processes without requiring any human annotation. Specifically, 1) we first propose scalable data pipelines to automatically construct reflection and error correction data from existing successful trajectories. While existing GUI models mainly focus on grounding and UI understanding ability, we propose the GUI-Reflection Task Suite to learn and evaluate reflection-oriented abilities explicitly. 2) Furthermore, we built a diverse and efficient environment for online training and data collection of GUI models on mobile devices. 3) We also present an iterative online reflection tuning algorithm leveraging the proposed environment, enabling the model to continuously enhance its reflection and error correction abilities. Our framework equips GUI agents with self-reflection and correction capabilities, paving the way for more robust, adaptable, and intelligent GUI automation, with all data, models, environments, and tools to be released publicly. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在革新图形用户界面（GUI）自动化方面展现了巨大的潜力。然而，现有的GUI模型大多依赖于从几乎无错误的离线轨迹中学习，因此缺乏反思和错误恢复能力。为弥补这一不足，我们提出GUI-Reflection，一个新颖的框架，该框架在专用训练阶段明确集成自我反思和错误修正能力：专门的GUI预训练、离线监督微调（SFT）和在线反思调优。GUI-Reflection使自反射行为的出现能够通过完全自动的数据生成和学习过程实现，无需任何人工标注。具体而言，1）我们首次提出可扩展的数据管道，自动生成来自现有成功轨迹的反思和错误修正数据。虽然现有的GUI模型主要集中在语义接地和UI理解能力上，我们提出了GUI-反射任务套件，以明确学习和评估面向反思的能力。2）此外，我们构建了一个多样化和高效的环境，用于移动设备上的GUI模型在线训练和数据收集。3）我们还提出了一种迭代的在线反思调优算法，利用提出的环境，使模型能够不断增强其反思和错误修正能力。我们的框架为GUI代理赋予了自我反思和修正的能力，为更 robust、适应性强和智能的GUI自动化铺平了道路，所有数据、模型、环境和工具将公开发布。 

---
# Reinforcing Multimodal Understanding and Generation with Dual Self-rewards 

**Title (ZH)**: 增强多模态理解与生成的双重自奖励机制 

**Authors**: Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07963)  

**Abstract**: Building upon large language models (LLMs), recent large multimodal models (LMMs) unify cross-model understanding and generation into a single framework. However, LMMs still struggle to achieve accurate image-text alignment, prone to generating text responses contradicting the visual input or failing to follow the text-to-image prompts. Current solutions require external supervision (e.g., human feedback or reward models) and only address unidirectional tasks-either understanding or generation. In this work, based on the observation that understanding and generation are inverse dual tasks, we introduce a self-supervised dual reward mechanism to reinforce the understanding and generation capabilities of LMMs. Specifically, we sample multiple outputs for a given input in one task domain, then reverse the input-output pairs to compute the dual likelihood of the model as self-rewards for optimization. Extensive experimental results on visual understanding and generation benchmarks demonstrate that our method can effectively enhance the performance of the model without any external supervision, especially achieving remarkable improvements in text-to-image tasks. 

**Abstract (ZH)**: 基于大型语言模型（LLMs），近期的大型多模态模型（LMMs）将跨模型的理解和生成统一在单一框架中。然而，LMMs仍然难以实现准确的图文对齐，容易生成与视觉输入矛盾的文字响应，或者无法遵循文字到图像的提示。当前的解决方案需要外部监督（例如，人工反馈或奖励模型）并且仅涉及单向任务，要么是理解，要么是生成。在观察到理解和生成是互逆的双重任务的基础上，我们引入了一种自监督的双重奖励机制来增强LMMs的理解和生成能力。具体而言，对于给定输入在某一任务域中采样子输出，然后反转输入-输出对计算模型的双重似然性作为自我奖励进行优化。在视觉理解和生成基准上的广泛实验结果表明，我们的方法可以在没有任何外部监督的情况下有效地提升模型性能，特别是在文字到图像任务上取得了显著的改进。 

---
# HAIBU-ReMUD: Reasoning Multimodal Ultrasound Dataset and Model Bridging to General Specific Domains 

**Title (ZH)**: HAIBU-ReMUD: 推理多模态超声数据集和模型连接到通用特定领域 

**Authors**: Shijie Wang, Yilun Zhang, Zeyu Lai, Dexing Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07837)  

**Abstract**: Multimodal large language models (MLLMs) have shown great potential in general domains but perform poorly in some specific domains due to a lack of domain-specific data, such as image-text data or vedio-text data. In some specific domains, there is abundant graphic and textual data scattered around, but lacks standardized arrangement. In the field of medical ultrasound, there are ultrasonic diagnostic books, ultrasonic clinical guidelines, ultrasonic diagnostic reports, and so on. However, these ultrasonic materials are often saved in the forms of PDF, images, etc., and cannot be directly used for the training of MLLMs. This paper proposes a novel image-text reasoning supervised fine-tuning data generation pipeline to create specific domain quadruplets (image, question, thinking trace, and answer) from domain-specific materials. A medical ultrasound domain dataset ReMUD is established, containing over 45,000 reasoning and non-reasoning supervised fine-tuning Question Answering (QA) and Visual Question Answering (VQA) data. The ReMUD-7B model, fine-tuned on Qwen2.5-VL-7B-Instruct, outperforms general-domain MLLMs in medical ultrasound field. To facilitate research, the ReMUD dataset, data generation codebase, and ReMUD-7B parameters will be released at this https URL, addressing the data shortage issue in specific domain MLLMs. 

**Abstract (ZH)**: 多模态大型语言模型在特定领域中的图像-文本推理监督微调数据生成管道：ReMUD数据集及其在医疗超声领域的应用 

---
# Understanding Financial Reasoning in AI: A Multimodal Benchmark and Error Learning Approach 

**Title (ZH)**: 理解AI中的财务推理：一种多模态基准和错误学习方法 

**Authors**: Shuangyan Deng, Haizhou Peng, Jiachen Xu, Chunhou Liu, Ciprian Doru Giurcuaneanu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06282)  

**Abstract**: Effective financial reasoning demands not only textual understanding but also the ability to interpret complex visual data such as charts, tables, and trend graphs. This paper introduces a new benchmark designed to evaluate how well AI models - especially large language and multimodal models - reason in finance-specific contexts. Covering 3,200 expert-level question-answer pairs across 15 core financial topics, the benchmark integrates both textual and visual modalities to reflect authentic analytical challenges in finance. To address limitations in current reasoning approaches, we propose an error-aware learning framework that leverages historical model mistakes and feedback to guide inference, without requiring fine-tuning. Our experiments across state-of-the-art models show that multimodal inputs significantly enhance performance and that incorporating error feedback leads to consistent and measurable improvements. The results highlight persistent challenges in visual understanding and mathematical logic, while also demonstrating the promise of self-reflective reasoning in financial AI systems. Our code and data can be found at https://anonymous/FinMR/CodeData. 

**Abstract (ZH)**: 有效的金融推理不仅需要文本理解，还需要解读复杂的视觉数据，如图表、表格和趋势图。本文介绍了一个新的基准，用于评估AI模型，尤其是大型语言和多模态模型，在金融特定背景下进行推理的能力。该基准涵盖了15个核心金融主题下的3200个专家级的问题-答案对，将文本和视觉模态结合起来，以反映真实的金融分析挑战。为了解决当前推理方法的限制，我们提出了一种错误感知的学习框架，该框架利用历史模型错误和反馈来引导推理，而无需微调。我们在最先进的模型上的实验表明，多模态输入显著提高了性能，并且整合错误反馈带来了持续且可测量的改进。结果凸显了在视觉理解和数学逻辑方面持续存在的挑战，同时也展示了金融AI系统中反思性推理的潜力。我们的代码和数据可在https://anonymous/FinMR/CodeData找到。 

---
# Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models 

**Title (ZH)**: 模仿还是推理：重塑视觉语言模型中的多模态在上下文学习 

**Authors**: Chengyue Huang, Yuchen Zhu, Sichen Zhu, Jingyun Xiao, Moises Andrade, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.07936)  

**Abstract**: Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics -- such as copying or majority voting -- rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with Reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, suggesting that current VLMs do not effectively utilize demonstration-level information as intended in MM-ICL. 

**Abstract (ZH)**: Vision-language模型在分布迁移下的多模态情境学习：有限的性能敏感性 

---
# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces 

**Title (ZH)**: 弥散一切：任意状态空间上的多模态扩散模型 

**Authors**: Kevin Rojas, Yuchen Zhu, Sichen Zhu, Felix X.-F. Ye, Molei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07903)  

**Abstract**: Diffusion models have demonstrated remarkable performance in generating unimodal data across various tasks, including image, video, and text generation. On the contrary, the joint generation of multimodal data through diffusion models is still in the early stages of exploration. Existing approaches heavily rely on external preprocessing protocols, such as tokenizers and variational autoencoders, to harmonize varied data representations into a unified, unimodal format. This process heavily demands the high accuracy of encoders and decoders, which can be problematic for applications with limited data. To lift this restriction, we propose a novel framework for building multimodal diffusion models on arbitrary state spaces, enabling native generation of coupled data across different modalities. By introducing an innovative decoupled noise schedule for each modality, we enable both unconditional and modality-conditioned generation within a single model simultaneously. We empirically validate our approach for text-image generation and mixed-type tabular data synthesis, demonstrating that it achieves competitive performance. 

**Abstract (ZH)**: 多模式扩散模型在任意状态空间的构建：一种新颖的框架及其在文本图像生成和混合类型表数据合成中的应用 

---
# PolyVivid: Vivid Multi-Subject Video Generation with Cross-Modal Interaction and Enhancement 

**Title (ZH)**: PolyVivid：跨模态交互与增强的多主题视频生成 

**Authors**: Teng Hu, Zhentao Yu, Zhengguang Zhou, Jiangning Zhang, Yuan Zhou, Qinglin Lu, Ran Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07848)  

**Abstract**: Despite recent advances in video generation, existing models still lack fine-grained controllability, especially for multi-subject customization with consistent identity and interaction. In this paper, we propose PolyVivid, a multi-subject video customization framework that enables flexible and identity-consistent generation. To establish accurate correspondences between subject images and textual entities, we design a VLLM-based text-image fusion module that embeds visual identities into the textual space for precise grounding. To further enhance identity preservation and subject interaction, we propose a 3D-RoPE-based enhancement module that enables structured bidirectional fusion between text and image embeddings. Moreover, we develop an attention-inherited identity injection module to effectively inject fused identity features into the video generation process, mitigating identity drift. Finally, we construct an MLLM-based data pipeline that combines MLLM-based grounding, segmentation, and a clique-based subject consolidation strategy to produce high-quality multi-subject data, effectively enhancing subject distinction and reducing ambiguity in downstream video generation. Extensive experiments demonstrate that PolyVivid achieves superior performance in identity fidelity, video realism, and subject alignment, outperforming existing open-source and commercial baselines. 

**Abstract (ZH)**: 尽管最近在视频生成方面取得了进展，现有模型仍缺乏细粒度可控性，特别是在具有一致身份和交互的多主体定制方面。本文提出PolyVivid，一种多主体视频定制框架，实现灵活且一致的身份生成。为了在主体图像与文本实体之间建立准确的对应关系，我们设计了一个基于VLLM的文本-图像融合模块，将在视觉空间中嵌入的身份特征嵌入到文本空间中，实现精确的语义关联。为了进一步增强身份保留和主体交互，我们提出了一种基于3D-RoPE的增强模块，实现了文本和图像嵌入之间结构化的双向融合。此外，我们开发了一种注意机制继承的身份注入模块，有效地将融合的身份特征注入到视频生成过程中，缓解身份漂移。最后，我们构建了一个基于MLLM的数据管道，结合了基于MLLM的语义关联、分割和基于团簇的主体聚合策略，生成高质量的多主体数据，有效增强主体区分并减少下游视频生成中的模糊性。大量实验表明，PolyVivid在身份保真度、视频现实性和主体对齐方面表现出 superior 的性能，超越了现有的开源和商用基线。 

---
# Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger 

**Title (ZH)**: 基于树搜索重排推理上下文使大型视觉-语言模型更强大 

**Authors**: Qi Yang, Chenghao Zhang, Lubin Fan, Kun Ding, Jieping Ye, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07785)  

**Abstract**: Recent advancements in Large Vision Language Models (LVLMs) have significantly improved performance in Visual Question Answering (VQA) tasks through multimodal Retrieval-Augmented Generation (RAG). However, existing methods still face challenges, such as the scarcity of knowledge with reasoning examples and erratic responses from retrieved knowledge. To address these issues, in this study, we propose a multimodal RAG framework, termed RCTS, which enhances LVLMs by constructing a Reasoning Context-enriched knowledge base and a Tree Search re-ranking method. Specifically, we introduce a self-consistent evaluation mechanism to enrich the knowledge base with intrinsic reasoning patterns. We further propose a Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR) to prioritize the most relevant examples. This ensures that LVLMs can leverage high-quality contextual reasoning for better and more consistent responses. Extensive experiments demonstrate that our framework achieves state-of-the-art performance on multiple VQA datasets, significantly outperforming In-Context Learning (ICL) and Vanilla-RAG methods. It highlights the effectiveness of our knowledge base and re-ranking method in improving LVLMs. Our code is available at this https URL. 

**Abstract (ZH)**: 近期大型多模态视觉语言模型（LVLMs）在视觉问答（VQA）任务中的显著进步是通过多模态检索增强生成（RAG）实现的。然而，现有方法仍然面临挑战，如缺乏具有推理示例的知识和检索知识的不稳定响应。为解决这些问题，本研究提出了一种名为RCTS的多模态RAG框架，通过构建增强推理背景知识库和树搜索再排序方法来增强LVLMs。具体而言，我们引入了一种自我一致性评估机制来丰富知识库中的内在推理模式。此外，我们提出了带有启发式奖励的蒙特卡洛树搜索（MCTS-HR）以优先处理最相关示例。这确保了LVLMs能够利用高质量的上下文推理以获得更好且更一致的响应。广泛实验表明，我们的框架在多个VQA数据集上达到了最先进的性能，显著优于上下文学习（ICL）和纯RAG方法。这突显了我们知识库和再排序方法在提高LVLMs效果方面的有效性。我们的代码可在以下链接获取：this https URL。 

---
# MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems 

**Title (ZH)**: MrM：针对多模态RAG系统的黑盒成员推理攻击 

**Authors**: Peiru Yang, Jinhua Yin, Haoran Zheng, Xueying Bai, Huili Wang, Yufei Sun, Xintian Li, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07399)  

**Abstract**: Multimodal retrieval-augmented generation (RAG) systems enhance large vision-language models by integrating cross-modal knowledge, enabling their increasing adoption across real-world multimodal tasks. These knowledge databases may contain sensitive information that requires privacy protection. However, multimodal RAG systems inherently grant external users indirect access to such data, making them potentially vulnerable to privacy attacks, particularly membership inference attacks (MIAs). % Existing MIA methods targeting RAG systems predominantly focus on the textual modality, while the visual modality remains relatively underexplored. To bridge this gap, we propose MrM, the first black-box MIA framework targeted at multimodal RAG systems. It utilizes a multi-object data perturbation framework constrained by counterfactual attacks, which can concurrently induce the RAG systems to retrieve the target data and generate information that leaks the membership information. Our method first employs an object-aware data perturbation method to constrain the perturbation to key semantics and ensure successful retrieval. Building on this, we design a counterfact-informed mask selection strategy to prioritize the most informative masked regions, aiming to eliminate the interference of model self-knowledge and amplify attack efficacy. Finally, we perform statistical membership inference by modeling query trials to extract features that reflect the reconstruction of masked semantics from response patterns. Experiments on two visual datasets and eight mainstream commercial visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves consistently strong performance across both sample-level and set-level evaluations, and remains robust under adaptive defenses. 

**Abstract (ZH)**: 多模态检索增强生成（RAG）系统的黑盒会员推理攻击框架 

---
# Lightweight Joint Audio-Visual Deepfake Detection via Single-Stream Multi-Modal Learning Framework 

**Title (ZH)**: 基于单流多模态学习框架的轻量级联合音视频深度伪造检测 

**Authors**: Kuiyuan Zhang, Wenjie Pei, Rushi Lan, Yifang Guo, Zhongyun Hua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07358)  

**Abstract**: Deepfakes are AI-synthesized multimedia data that may be abused for spreading misinformation. Deepfake generation involves both visual and audio manipulation. To detect audio-visual deepfakes, previous studies commonly employ two relatively independent sub-models to learn audio and visual features, respectively, and fuse them subsequently for deepfake detection. However, this may underutilize the inherent correlations between audio and visual features. Moreover, utilizing two isolated feature learning sub-models can result in redundant neural layers, making the overall model inefficient and impractical for resource-constrained environments.
In this work, we design a lightweight network for audio-visual deepfake detection via a single-stream multi-modal learning framework. Specifically, we introduce a collaborative audio-visual learning block to efficiently integrate multi-modal information while learning the visual and audio features. By iteratively employing this block, our single-stream network achieves a continuous fusion of multi-modal features across its layers. Thus, our network efficiently captures visual and audio features without the need for excessive block stacking, resulting in a lightweight network design. Furthermore, we propose a multi-modal classification module that can boost the dependence of the visual and audio classifiers on modality content. It also enhances the whole resistance of the video classifier against the mismatches between audio and visual modalities. We conduct experiments on the DF-TIMIT, FakeAVCeleb, and DFDC benchmark datasets. Compared to state-of-the-art audio-visual joint detection methods, our method is significantly lightweight with only 0.48M parameters, yet it achieves superiority in both uni-modal and multi-modal deepfakes, as well as in unseen types of deepfakes. 

**Abstract (ZH)**: 轻量级单流多模态学习框架在音频-视觉深仿生成检测中的应用 

---
# Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward 

**Title (ZH)**: 通过视觉感知奖励提升多模态大型语言模型的多模态推理能力 

**Authors**: Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07218)  

**Abstract**: Enhancing the multimodal reasoning capabilities of Multimodal Large Language Models (MLLMs) is a challenging task that has attracted increasing attention in the community. Recently, several studies have applied Reinforcement Learning with Verifiable Rewards (RLVR) to the multimodal domain in order to enhance the reasoning abilities of MLLMs. However, these works largely overlook the enhancement of multimodal perception capabilities in MLLMs, which serve as a core prerequisite and foundational component of complex multimodal reasoning. Through McNemar's test, we find that existing RLVR method fails to effectively enhance the multimodal perception capabilities of MLLMs, thereby limiting their further improvement in multimodal reasoning. To address this limitation, we propose Perception-R1, which introduces a novel visual perception reward that explicitly encourages MLLMs to perceive the visual content accurately, thereby can effectively incentivizing both their multimodal perception and reasoning capabilities. Specifically, we first collect textual visual annotations from the CoT trajectories of multimodal problems, which will serve as visual references for reward assignment. During RLVR training, we employ a judging LLM to assess the consistency between the visual annotations and the responses generated by MLLM, and assign the visual perception reward based on these consistency judgments. Extensive experiments on several multimodal reasoning benchmarks demonstrate the effectiveness of our Perception-R1, which achieves state-of-the-art performance on most benchmarks using only 1,442 training data. 

**Abstract (ZH)**: 增强多模态大型语言模型的多模态推理能力是一个具有挑战性的任务，已在社区中引起越来越多的关注。最近，一些研究将可验证奖励的强化学习（RLVR）应用于多模态领域，以提升多模态大型语言模型（MLLMs）的推理能力。然而，这些工作在增强MLLMs的多模态感知能力方面仍然关注不足，而后者是复杂多模态推理的核心先决条件和基础组件。通过麦加尼尔检验，我们发现现有的RLVR方法未能有效提升MLLMs的多模态感知能力，从而限制了它们在多模态推理方面的进一步改进。为解决这一局限性，我们提出了Perception-R1，引入了一种新型的视觉感知奖励，明确鼓励MLLMs准确感知视觉内容，从而有效激励其多模态感知和推理能力。具体而言，我们首先从多模态问题的CoT轨迹中收集文本视觉注释，作为奖励分配的视觉参考。在RLVR训练过程中，我们利用一个评判大语言模型评估MLLM生成的响应与视觉注释之间的一致性，并基于这些一致性判断分配视觉感知奖励。在多个多模态推理基准上的广泛实验表明，Perception-R1的有效性，仅使用1,442训练数据就在大多数基准上取得了最先进的性能。 

---
# Learning Compact Vision Tokens for Efficient Large Multimodal Models 

**Title (ZH)**: 学习紧凑的视觉令牌以实现高效的大规模多模态模型 

**Authors**: Hao Tang, Chengchao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07138)  

**Abstract**: Large multimodal models (LMMs) suffer significant computational challenges due to the high cost of Large Language Models (LLMs) and the quadratic complexity of processing long vision token sequences. In this paper, we explore the spatial redundancy among vision tokens and shorten the length of vision token sequences for inference acceleration. Specifically, we propose a Spatial Token Fusion (STF) method to learn compact vision tokens for short vision token sequence, where spatial-adjacent tokens are fused into one. Meanwhile, weight-frozen vision encoder can not well adapt to the demand of extensive downstream vision-language tasks. To this end, we further introduce a Multi-Block Token Fusion (MBTF) module to supplement multi-granularity features for the reduced token sequence. Overall, we combine STF and MBTF module to balance token reduction and information preservation, thereby improving inference efficiency without sacrificing multimodal reasoning capabilities. Experimental results demonstrate that our method based on LLaVA-1.5 achieves comparable or even superior performance to the baseline on 8 popular vision-language benchmarks with only $25\%$ vision tokens of baseline. The source code and trained weights are available at this https URL. 

**Abstract (ZH)**: 大规模多模态模型（LMMs）因大型语言模型（LLMs）的高计算成本和处理长视觉标记序列的二次复杂性而面临重大计算挑战。本文探索视觉标记之间的空间冗余，并缩短视觉标记序列以加速推理。具体来说，我们提出了一种空间标记融合（STF）方法，通过聚合相邻的视觉标记来学习紧凑的视觉标记，从而缩短视觉标记序列的长度。与此同时，权重冻结的视觉编码器难以适应广泛的下游视觉-语言任务的需求。为此，我们进一步引入了多块标记融合（MBTF）模块，以补充减少的标记序列的多粒度特征。总体而言，我们将STF模块和MBTF模块结合，以平衡标记减少和信息保留，从而在不牺牲多模态推理能力的情况下提高推理效率。实验结果表明，基于LLaVA-1.5的方法仅使用基线的25%视觉标记，在8个流行的视觉-语言基准上达到了可比或更优的性能。源代码和训练权重可在以下链接获取。 

---
# MAGNET: A Multi-agent Framework for Finding Audio-Visual Needles by Reasoning over Multi-Video Haystacks 

**Title (ZH)**: MAGNET：一种基于多视频 haystack 原材料进行推理查找音视频 needles 的多智能体框架 

**Authors**: Sanjoy Chowdhury, Mohamed Elmoghany, Yohan Abeysinghe, Junjie Fei, Sayan Nag, Salman Khan, Mohamed Elhoseiny, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.07016)  

**Abstract**: Large multimodal models (LMMs) have shown remarkable progress in audio-visual understanding, yet they struggle with real-world scenarios that require complex reasoning across extensive video collections. Existing benchmarks for video question answering remain limited in scope, typically involving one clip per query, which falls short of representing the challenges of large-scale, audio-visual retrieval and reasoning encountered in practical applications. To bridge this gap, we introduce a novel task named AV-HaystacksQA, where the goal is to identify salient segments across different videos in response to a query and link them together to generate the most informative answer. To this end, we present AVHaystacks, an audio-visual benchmark comprising 3100 annotated QA pairs designed to assess the capabilities of LMMs in multi-video retrieval and temporal grounding task. Additionally, we propose a model-agnostic, multi-agent framework MAGNET to address this challenge, achieving up to 89% and 65% relative improvements over baseline methods on BLEU@4 and GPT evaluation scores in QA task on our proposed AVHaystacks. To enable robust evaluation of multi-video retrieval and temporal grounding for optimal response generation, we introduce two new metrics, STEM, which captures alignment errors between a ground truth and a predicted step sequence and MTGS, to facilitate balanced and interpretable evaluation of segment-level grounding performance. Project: this https URL 

**Abstract (ZH)**: 大规模多模态模型在音频-视觉理解方面取得了显著进展，但在处理需要在大量视频集合之间进行复杂推理的实际场景时仍存在挑战。现有的视频问答基准在范围上仍然有限，通常每个查询只涉及一个片段，这远不足以代表实际应用中大规模、音频-视觉检索和推理所面临的挑战。为了解决这一问题，我们引入了一个名为AV-HaystacksQA的新任务，目标是在响应查询时识别不同视频中的关键片段，并将它们链接起来生成最informative的答案。为此，我们提出了一个包含3100个标注问答对的AVHaystacks音频-视觉基准，旨在评估大规模多模态模型在多视频检索和时间定位任务中的能力。此外，我们提出了一种模型无关的多智能体框架MAGNET来解决这一挑战，该框架在我们提出的AVHaystacks基准上提出的问答任务中，相对于基线方法在BLEU@4和GPT评估得分上分别取得了高达89%和65%的相对改进。为了确保多视频检索和时间定位的稳健评估，以便生成最优响应，我们引入了两个新的度量标准STEM，用于捕捉预测步骤序列与真实步骤序列之间的对齐错误，以及MTGS，以促进段级定位性能的平衡和可解释评估。 

---
# Fuse and Federate: Enhancing EV Charging Station Security with Multimodal Fusion and Federated Learning 

**Title (ZH)**: 融合与联邦：通过多模态融合和联邦学习增强电动汽车充电站安全 

**Authors**: Rabah Rahal, Abdelaziz Amara Korba, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2506.06730)  

**Abstract**: The rapid global adoption of electric vehicles (EVs) has established electric vehicle supply equipment (EVSE) as a critical component of smart grid infrastructure. While essential for ensuring reliable energy delivery and accessibility, EVSE systems face significant cybersecurity challenges, including network reconnaissance, backdoor intrusions, and distributed denial-of-service (DDoS) attacks. These emerging threats, driven by the interconnected and autonomous nature of EVSE, require innovative and adaptive security mechanisms that go beyond traditional intrusion detection systems (IDS). Existing approaches, whether network-based or host-based, often fail to detect sophisticated and targeted attacks specifically crafted to exploit new vulnerabilities in EVSE infrastructure. This paper proposes a novel intrusion detection framework that leverages multimodal data sources, including network traffic and kernel events, to identify complex attack patterns. The framework employs a distributed learning approach, enabling collaborative intelligence across EVSE stations while preserving data privacy through federated learning. Experimental results demonstrate that the proposed framework outperforms existing solutions, achieving a detection rate above 98% and a precision rate exceeding 97% in decentralized environments. This solution addresses the evolving challenges of EVSE security, offering a scalable and privacypreserving response to advanced cyber threats 

**Abstract (ZH)**: 电动汽车（EVs）的快速全球采纳已将电动汽车供电设备（EVSE）确立为智能电网基础设施的关键组件。虽然EVSE系统对于确保可靠的能源供应和可访问性至关重要，但它们也面临着重大的网络安全挑战，包括网络探测、后门入侵和分布式拒绝服务（DDoS）攻击。这些新兴威胁由EVSE的互连和自主特性驱动，需要超越传统入侵检测系统（IDS）的创新和适应性安全机制。现有方法，无论是基于网络的还是基于主机的，往往无法检测出专门设计以利用EVSE基础设施中新漏洞的复杂和针对性攻击。本文提出了一种新的入侵检测框架，该框架利用多模态数据源，包括网络流量和内核事件，以识别复杂的攻击模式。该框架采用分布式学习方法，可以在保留数据隐私的同时，在EVSE站点之间实现协作智能。实验结果表明，所提出的框架在分散环境中表现出色，检测率达到98%以上，精确率超过97%。该解决方案应对了EVSE安全的不断演变的挑战，提供了对高级网络安全威胁的可扩展和隐私保护响应。 

---
# LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles 

**Title (ZH)**: LaMP-Cap：基于多模态图谱的个性化图表描述生成 

**Authors**: Ho Yin 'Sam' Ng, Ting-Yao Hsu, Aashish Anantha Ramakrishnan, Branislav Kveton, Nedim Lipka, Franck Dernoncourt, Dongwon Lee, Tong Yu, Sungchul Kim, Ryan A. Rossi, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06561)  

**Abstract**: Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones. 

**Abstract (ZH)**: 个性化多模态图例生成数据集LaMP-Cap：基于多模态图例配置的个性化图例生成 

---
# Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning 

**Title (ZH)**: 双模态注意力增强文本视频检索的三元部分边际对比学习 

**Authors**: Chen Jiang, Hong Liu, Xuzheng Yu, Qing Wang, Yuan Cheng, Jia Xu, Zhongyi Liu, Qingpei Guo, Wei Chu, Ming Yang, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2309.11082)  

**Abstract**: In recent years, the explosion of web videos makes text-video retrieval increasingly essential and popular for video filtering, recommendation, and search. Text-video retrieval aims to rank relevant text/video higher than irrelevant ones. The core of this task is to precisely measure the cross-modal similarity between texts and videos. Recently, contrastive learning methods have shown promising results for text-video retrieval, most of which focus on the construction of positive and negative pairs to learn text and video representations. Nevertheless, they do not pay enough attention to hard negative pairs and lack the ability to model different levels of semantic similarity. To address these two issues, this paper improves contrastive learning using two novel techniques. First, to exploit hard examples for robust discriminative power, we propose a novel Dual-Modal Attention-Enhanced Module (DMAE) to mine hard negative pairs from textual and visual clues. By further introducing a Negative-aware InfoNCE (NegNCE) loss, we are able to adaptively identify all these hard negatives and explicitly highlight their impacts in the training loss. Second, our work argues that triplet samples can better model fine-grained semantic similarity compared to pairwise samples. We thereby present a new Triplet Partial Margin Contrastive Learning (TPM-CL) module to construct partial order triplet samples by automatically generating fine-grained hard negatives for matched text-video pairs. The proposed TPM-CL designs an adaptive token masking strategy with cross-modal interaction to model subtle semantic differences. Extensive experiments demonstrate that the proposed approach outperforms existing methods on four widely-used text-video retrieval datasets, including MSR-VTT, MSVD, DiDeMo and ActivityNet. 

**Abstract (ZH)**: 近年来，网络视频的爆炸式增长使得文本-视频检索在视频过滤、推荐和搜索中愈发重要和流行。文本-视频检索的目标是将相关文本/视频排名在不相关项之上。这一任务的核心是如何精确地度量文本与视频的跨模态相似性。近日，对比学习方法在文本-视频检索中显示出有希望的结果，大多数方法集中在构建正样本和负样本对来学习文本和视频表示。然而，这些方法未能充分关注硬负样本对，并缺乏建模不同粒度语义相似性的能力。为解决这两个问题，本文采用两种新颖的技术改进了对比学习。首先，为了利用困难样本增强鲁棒性判别能力，我们提出了一种新的双模态注意力增强模块（DMAE），从文本和视觉线索中挖掘困难负样本对。进一步引入了负样本感知的InfoNCE损失（NegNCE），能自适应地识别所有这些困难负样本并在训练损失中明确突出其影响。其次，我们提出，三元组样本比成对样本更适合建模细粒度语义相似性。因此，我们提出了一种新的三元组部分边界对比学习模块（TPM-CL），通过自动生成匹配文本-视频对的细粒度困难负样本来构造部分排序三元组样本。所提出的TPM-CL设计了一种带有跨模态交互的自适应标记掩蔽策略，以建模细微的语义差异。大量实验表明，所提出的方法在包括MSR-VTT、MSVD、DiDeMo和ActivityNet的四个广泛使用的文本-视频检索数据集中优于现有方法。 

---
