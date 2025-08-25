# Take That for Me: Multimodal Exophora Resolution with Interactive Questioning for Ambiguous Out-of-View Instructions 

**Title (ZH)**: 代我完成那件事：针对含糊的不可视指令的多模态消除了望代词解析与交互式提问 

**Authors**: Akira Oyama, Shoichi Hasegawa, Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16143)  

**Abstract**: Daily life support robots must interpret ambiguous verbal instructions involving demonstratives such as ``Bring me that cup,'' even when objects or users are out of the robot's view. Existing approaches to exophora resolution primarily rely on visual data and thus fail in real-world scenarios where the object or user is not visible. We propose Multimodal Interactive Exophora resolution with user Localization (MIEL), which is a multimodal exophora resolution framework leveraging sound source localization (SSL), semantic mapping, visual-language models (VLMs), and interactive questioning with GPT-4o. Our approach first constructs a semantic map of the environment and estimates candidate objects from a linguistic query with the user's skeletal data. SSL is utilized to orient the robot toward users who are initially outside its visual field, enabling accurate identification of user gestures and pointing directions. When ambiguities remain, the robot proactively interacts with the user, employing GPT-4o to formulate clarifying questions. Experiments in a real-world environment showed results that were approximately 1.3 times better when the user was visible to the robot and 2.0 times better when the user was not visible to the robot, compared to the methods without SSL and interactive questioning. The project website is this https URL. 

**Abstract (ZH)**: 日常生活中支持型机器人必须解析涉及指示代词（如“ Bring me that cup”）的模糊口头指令，即使目标物体或用户不在机器人视野范围内。现有的外指消解方法主要依赖视觉数据，在物体或用户不可见的实际场景中失效。我们提出了基于声源定位（SSL）、语义地图构建、视觉语言模型（VLMs）以及与GPT-4o的交互性问题提出方法的多模态交互性外指解析与用户定位（MIEL）框架。该方法首先构建环境的语义地图，并利用用户骨骼数据从语言查询中估计候选物体。声源定位技术使机器人能够朝向其初始视觉范围之外的用户定位，从而实现对用户手势和指向方向的准确识别。当仍有歧义时，机器人会主动与用户交互，利用GPT-4o提出澄清问题。在真实环境中的实验表明，当用户可视时，效果大约提高了1.3倍；当用户不可视时，效果提高了2.0倍，对比于不使用SSL和交互性问题提出的方法。项目网站：https://this.is/MIEL。 

---
# Bridging the Gap in Ophthalmic AI: MM-Retinal-Reason Dataset and OphthaReason Model toward Dynamic Multimodal Reasoning 

**Title (ZH)**: 眼科AI领域的桥梁：MM-Retinal-Reason数据集和OphthaReason模型 toward 动态多模态推理 

**Authors**: Ruiqi Wu, Yuang Yao, Tengfei Ma, Chenran Zhang, Na Su, Tao Zhou, Geng Chen, Wen Fan, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.16129)  

**Abstract**: Multimodal large language models (MLLMs) have recently demonstrated remarkable reasoning abilities with reinforcement learning paradigm. Although several multimodal reasoning models have been explored in the medical domain, most of them focus exclusively on basic reasoning, which refers to shallow inference based on visual feature matching. However, real-world clinical diagnosis extends beyond basic reasoning, demanding reasoning processes that integrate heterogeneous clinical information (such as chief complaints and medical history) with multimodal medical imaging data. To bridge this gap, we introduce MM-Retinal-Reason, the first ophthalmic multimodal dataset with the full spectrum of perception and reasoning. It encompasses both basic reasoning tasks and complex reasoning tasks, aiming to enhance visual-centric fundamental reasoning capabilities and emulate realistic clinical thinking patterns. Building upon MM-Retinal-Reason, we propose OphthaReason, the first ophthalmology-specific multimodal reasoning model with step-by-step reasoning traces. To enable flexible adaptation to both basic and complex reasoning tasks, we specifically design a novel method called Uncertainty-Aware Dynamic Thinking (UADT), which estimates sample-level uncertainty via entropy and dynamically modulates the model's exploration depth using a shaped advantage mechanism. Comprehensive experiments demonstrate that our model achieves state-of-the-art performance on both basic and complex reasoning tasks, outperforming general-purpose MLLMs, medical MLLMs, RL-based medical MLLMs, and ophthalmic MLLMs by at least 24.92\%, 15.00\%, 21.20\%, and 17.66\%. Project Page: \href{this https URL}{link}. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在强化学习 paradigm 下展现出了非凡的推理能力。尽管已经在医疗领域探索了多种多模态推理模型，但大多数模型仅专注于基本推理，即基于视觉特征匹配的浅层推理。然而，实际临床诊断远超过基本推理，需要能整合异质临床信息（如主要症状和医疗史）与多模态医学影像数据的推理过程。为填补这一空白，我们引入了 MM-Retinal-Reason，这是首个涵盖全部感知和推理范围的眼科多模态数据集，同时囊括基础推理任务和复杂推理任务，旨在增强以视觉为中心的基础推理能力，并模拟真实的临床思维模式。基于 MM-Retinal-Reason，我们提出了 OphthaReason，这是首个针对眼科的多模态推理模型，具有逐步推理轨迹。为了灵活适应基础和复杂推理任务，我们特别设计了一种名为不确定性意识动态思考（UADT）的新方法，通过熵估计样本级别不确定性，并使用成形优势机制动态调节模型的探索深度。全面的实验表明，我们的模型在基础和复杂推理任务上均取得了最先进的性能，分别超出通用 MLLMs、医疗 MLLMs、基于 RL 的医疗 MLLMs 和眼科 MLLMs 至少 24.92%、15.00%、21.20% 和 17.66%。项目页面：[链接]。 

---
# MMAPG: A Training-Free Framework for Multimodal Multi-hop Question Answering via Adaptive Planning Graphs 

**Title (ZH)**: MMAPG：基于自适应规划图的无培训框架多模态多跳问答 

**Authors**: Yiheng Hu, Xiaoyang Wang, Qing Liu, Xiwei Xu, Qian Fu, Wenjie Zhang, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16051)  

**Abstract**: Multimodal Multi-hop question answering requires integrating information from diverse sources, such as images and texts, to derive answers. Existing methods typically rely on sequential retrieval and reasoning, where each step builds on the previous output. However, this single-path paradigm makes them vulnerable to errors due to misleading intermediate steps. Moreover, developing multimodal models can be computationally expensive, often requiring extensive training. To address these limitations, we propose a training-free framework guided by an Adaptive Planning Graph, which consists of planning, retrieval and reasoning modules. The planning module analyzes the current state of the Adaptive Planning Graph, determines the next action and where to expand the graph, which enables dynamic and flexible exploration of reasoning paths. To handle retrieval of text to unspecified target modalities, we devise modality-specific strategies that dynamically adapt to distinct data types. Our approach preserves the characteristics of multimodal information without costly task-specific training, enabling seamless integration with up-to-date models. Finally, the experiments on MultimodalQA and WebQA show that our approach matches or outperforms existing models that rely on training. 

**Abstract (ZH)**: 多模态多跳问答需要从图像和文本等多种来源整合信息以推导出答案。现有的方法通常依赖于顺序检索和推理，每一步都建立在上一步的基础上。然而，这种单一路径的方法容易受到误导性中间步骤的错误影响。此外，开发多模态模型可能计算成本高昂，通常需要大量的训练。为了解决这些限制，我们提出了一种无需训练的框架，该框架由自适应规划图引导，包括规划、检索和推理模块。规划模块分析自适应规划图的当前状态，确定下一步行动及扩展图的位置，从而实现动态和灵活的推理路径探索。为了处理文本检索到未指定目标模态的情况，我们设计了模态特定的策略，这些策略能够根据不同的数据类型动态适应。我们的方法在不进行代价高昂的任务特定训练的情况下保留了多模态信息的特点，使得与最新模型无缝集成成为可能。最后，我们在MultimodalQA和WebQA上的实验显示，我们的方法能够与依赖训练的现有模型相匹配或表现出色。 

---
# A Multimodal-Multitask Framework with Cross-modal Relation and Hierarchical Interactive Attention for Semantic Comprehension 

**Title (ZH)**: 跨模态关系与层次交互注意的多模态多任务框架：语义理解 

**Authors**: Mohammad Zia Ur Rehman, Devraj Raghuvanshi, Umang Jain, Shubhi Bansal, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16300)  

**Abstract**: A major challenge in multimodal learning is the presence of noise within individual modalities. This noise inherently affects the resulting multimodal representations, especially when these representations are obtained through explicit interactions between different modalities. Moreover, the multimodal fusion techniques while aiming to achieve a strong joint representation, can neglect valuable discriminative information within the individual modalities. To this end, we propose a Multimodal-Multitask framework with crOss-modal Relation and hIErarchical iNteractive aTtention (MM-ORIENT) that is effective for multiple tasks. The proposed approach acquires multimodal representations cross-modally without explicit interaction between different modalities, reducing the noise effect at the latent stage. To achieve this, we propose cross-modal relation graphs that reconstruct monomodal features to acquire multimodal representations. The features are reconstructed based on the node neighborhood, where the neighborhood is decided by the features of a different modality. We also propose Hierarchical Interactive Monomadal Attention (HIMA) to focus on pertinent information within a modality. While cross-modal relation graphs help comprehend high-order relationships between two modalities, HIMA helps in multitasking by learning discriminative features of individual modalities before late-fusing them. Finally, extensive experimental evaluation on three datasets demonstrates that the proposed approach effectively comprehends multimodal content for multiple tasks. 

**Abstract (ZH)**: 多模态学习中的主要挑战是个体内噪声的存在。这种噪声会直接影响最终的多模态表示，尤其是在通过不同模态的显式交互获取这些表示时。此外，多模态融合技术虽然旨在实现强联合表示，但可能会忽略个体模态内的有价值区分信息。为此，我们提出了一种有效的多任务框架——跨模态关系和层次交互注意力（MM-ORIENT），该框架能够在不进行不同模态之间显式交互的情况下跨模态获取表示，从而在潜在阶段减少噪声的影响。为此，我们提出了跨模态关系图，该图通过重构单模态特征来获取多模态表示，特征的重构基于节点邻域，邻域的选择由不同模态的特征决定。我们还提出了层次交互单模态注意（HIMA），以关注模态内的相关信息。跨模态关系图有助于理解两个模态之间的高阶关系，而HIMA则通过在晚期融合之前学习个体模态的区分特征来实现多任务处理。最后，针对三个数据集的广泛实验评估表明，所提出的方法能够有效理解多模态内容以适应多种任务。 

---
# FlexMUSE: Multimodal Unification and Semantics Enhancement Framework with Flexible interaction for Creative Writing 

**Title (ZH)**: FlexMUSE：具有灵活交互的多模态统一与语义增强框架用于创意写作 

**Authors**: Jiahao Chen, Zhiyong Ma, Wenbiao Du, Qingyuan Chuai  

**Link**: [PDF](https://arxiv.org/pdf/2508.16230)  

**Abstract**: Multi-modal creative writing (MMCW) aims to produce illustrated articles. Unlike common multi-modal generative (MMG) tasks such as storytelling or caption generation, MMCW is an entirely new and more abstract challenge where textual and visual contexts are not strictly related to each other. Existing methods for related tasks can be forcibly migrated to this track, but they require specific modality inputs or costly training, and often suffer from semantic inconsistencies between modalities. Therefore, the main challenge lies in economically performing MMCW with flexible interactive patterns, where the semantics between the modalities of the output are more aligned. In this work, we propose FlexMUSE with a T2I module to enable optional visual input. FlexMUSE promotes creativity and emphasizes the unification between modalities by proposing the modality semantic alignment gating (msaGate) to restrict the textual input. Besides, an attention-based cross-modality fusion is proposed to augment the input features for semantic enhancement. The modality semantic creative direct preference optimization (mscDPO) within FlexMUSE is designed by extending the rejected samples to facilitate the writing creativity. Moreover, to advance the MMCW, we expose a dataset called ArtMUSE which contains with around 3k calibrated text-image pairs. FlexMUSE achieves promising results, demonstrating its consistency, creativity and coherence. 

**Abstract (ZH)**: 多模态创意思维写作 (MMCW) 旨在生成配有插图的文章。与常见的多模态生成任务（如讲故事或生成标题）不同，MMCW 是一个全新的更具抽象性的挑战，其中文本和视觉上下文之间并不严格相关。现有的相关任务方法可以强制迁移到此赛道中，但它们需要特定模态的输入或大量的训练，并且常常会遭受模态间语义不一致的问题。因此，主要的挑战在于以经济高效且具有灵活性的交互模式执行 MMCW，使输出模态之间的语义更加对齐。在此项工作中，我们提出了 FlexMUSE，并配备了一个文本到图像 (T2I) 模块以实现可选的视觉输入。FlexMUSE 通过提出模态语义对齐门控（msaGate）来促进创意思维并强调模态之间的统一性。此外，我们提出了基于注意力机制的跨模态融合，以增强输入特征以提升语义。FlexMUSE 中的模态语义创意思维直接偏好优化（mscDPO）通过扩展拒绝样本来促进写作创意思维。为了推进 MMCW，我们公开了一个名为 ArtMUSE 的数据集，其中包含约 3000 个校准的图文对。FlexMUSE 达到了令人鼓舞的结果，展示了其一致性和创见性。 

---
# EGRA:Toward Enhanced Behavior Graphs and Representation Alignment for Multimodal Recommendation 

**Title (ZH)**: EGRA:向多模态推荐中增强的行为图和表示对齐迈进 

**Authors**: Xiaoxiong Zhang, Xin Zhou, Zhiwei Zeng, Yongjie Wang, Dusit Niyato, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16170)  

**Abstract**: MultiModal Recommendation (MMR) systems have emerged as a promising solution for improving recommendation quality by leveraging rich item-side modality information, prompting a surge of diverse methods. Despite these advances, existing methods still face two critical limitations. First, they use raw modality features to construct item-item links for enriching the behavior graph, while giving limited attention to balancing collaborative and modality-aware semantics or mitigating modality noise in the process. Second, they use a uniform alignment weight across all entities and also maintain a fixed alignment strength throughout training, limiting the effectiveness of modality-behavior alignment. To address these challenges, we propose EGRA. First, instead of relying on raw modality features, it alleviates sparsity by incorporating into the behavior graph an item-item graph built from representations generated by a pretrained MMR model. This enables the graph to capture both collaborative patterns and modality aware similarities with enhanced robustness against modality noise. Moreover, it introduces a novel bi-level dynamic alignment weighting mechanism to improve modality-behavior representation alignment, which dynamically assigns alignment strength across entities according to their alignment degree, while gradually increasing the overall alignment intensity throughout training. Extensive experiments on five datasets show that EGRA significantly outperforms recent methods, confirming its effectiveness. 

**Abstract (ZH)**: 多模态推荐（MMR）系统通过利用丰富的物品侧模态信息来构建物品-物品链接，从而增强行为图，并促进了多种方法的涌现。尽管取得了这些进展，现有方法仍然面临两个关键限制。首先，它们在构建物品-物品链接以丰富行为图时主要依赖原始模态特征，而在平衡协同和模态感知语义或减轻模态噪声方面考虑较少。其次，它们在所有实体上使用统一的对齐权重，并在整个训练过程中保持固定的对齐强度，这限制了模态-行为对齐的有效性。为解决这些挑战，我们提出了一种称为EGRA的方法。首先，EGRA通过将由预训练MMR模型生成的表示构建的物品-物品图融入行为图中，缓解了稀疏性问题，使图能够捕捉到增强鲁棒性的协同模式和模态感知相似性。此外，EGRA引入了一种新颖的双层动态对齐权重机制，以提高模态-行为表示对齐，该机制根据实体的对齐程度动态分配对齐强度，并在整个训练过程中逐步增加整体对齐强度。在五个数据集上的广泛实验表明，EGRA显著优于最近的方法，证实了其有效性。 

---
# Through the Looking Glass: A Dual Perspective on Weakly-Supervised Few-Shot Segmentation 

**Title (ZH)**: 从镜中窥视：弱监督少量样本分割的双重视角 

**Authors**: Jiaqi Ma, Guo-Sen Xie, Fang Zhao, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16159)  

**Abstract**: Meta-learning aims to uniformly sample homogeneous support-query pairs, characterized by the same categories and similar attributes, and extract useful inductive biases through identical network architectures. However, this identical network design results in over-semantic homogenization. To address this, we propose a novel homologous but heterogeneous network. By treating support-query pairs as dual perspectives, we introduce heterogeneous visual aggregation (HA) modules to enhance complementarity while preserving semantic commonality. To further reduce semantic noise and amplify the uniqueness of heterogeneous semantics, we design a heterogeneous transfer (HT) module. Finally, we propose heterogeneous CLIP (HC) textual information to enhance the generalization capability of multimodal models. In the weakly-supervised few-shot semantic segmentation (WFSS) task, with only 1/24 of the parameters of existing state-of-the-art models, TLG achieves a 13.2\% improvement on Pascal-5\textsuperscript{i} and a 9.7\% improvement on COCO-20\textsuperscript{i}. To the best of our knowledge, TLG is also the first weakly supervised (image-level) model that outperforms fully supervised (pixel-level) models under the same backbone architectures. The code is available at this https URL. 

**Abstract (ZH)**: 基于元学习的同源异质网络设计与应用：提升弱监督少数样本语义分割模型的泛化能力 

---
# DeepMEL: A Multi-Agent Collaboration Framework for Multimodal Entity Linking 

**Title (ZH)**: DeepMEL：多模态实体链接的多智能体协作框架 

**Authors**: Fang Wang, Tianwei Yan, Zonghao Yang, Minghao Hu, Jun Zhang, Zhunchen Luo, Xiaoying Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.15876)  

**Abstract**: Multimodal Entity Linking (MEL) aims to associate textual and visual mentions with entities in a multimodal knowledge graph. Despite its importance, current methods face challenges such as incomplete contextual information, coarse cross-modal fusion, and the difficulty of jointly large language models (LLMs) and large visual models (LVMs). To address these issues, we propose DeepMEL, a novel framework based on multi-agent collaborative reasoning, which achieves efficient alignment and disambiguation of textual and visual modalities through a role-specialized division strategy. DeepMEL integrates four specialized agents, namely Modal-Fuser, Candidate-Adapter, Entity-Clozer and Role-Orchestrator, to complete end-to-end cross-modal linking through specialized roles and dynamic coordination. DeepMEL adopts a dual-modal alignment path, and combines the fine-grained text semantics generated by the LLM with the structured image representation extracted by the LVM, significantly narrowing the modal gap. We design an adaptive iteration strategy, combines tool-based retrieval and semantic reasoning capabilities to dynamically optimize the candidate set and balance recall and precision. DeepMEL also unifies MEL tasks into a structured cloze prompt to reduce parsing complexity and enhance semantic comprehension. Extensive experiments on five public benchmark datasets demonstrate that DeepMEL achieves state-of-the-art performance, improving ACC by 1%-57%. Ablation studies verify the effectiveness of all modules. 

**Abstract (ZH)**: 多模态实体链接（MEL）旨在将文本和视觉提及与多模态知识图谱中的实体关联起来。为了解决当前方法面临的挑战，如不完整的上下文信息、粗放的跨模态融合以及大型语言模型（LLMs）和大型视觉模型（LVMs）难以联合使用的问题，我们提出了一种基于多agent协同推理的新框架DeepMEL，通过角色专业化分工策略实现高效的跨模态对齐和消歧。DeepMEL通过专业化角色和动态协调，整合了四种专业化的agent：模态融合器、候选适配器、实体遮盖器和角色协调器，以端到端的方式完成跨模态链接。DeepMEL采用双模态对齐路径，结合LLM生成的细粒度文本语义和LVM提取的结构化图像表示，显著缩小了模态差距。我们设计了一种自适应迭代策略，结合基于工具的检索和语义推理能力，动态优化候选集并平衡召回率和精确率。DeepMEL还将MEL任务统一为结构化的填空提示，以减少解析复杂性和增强语义理解。在五个公开基准数据集上的 extensive 实验表明，DeepMEL 达到了领先水平，ACC 提高性能高达 1%-57%。消融研究验证了所有模块的有效性。 

---
