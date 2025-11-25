# UNeMo: Collaborative Visual-Language Reasoning and Navigation via a Multimodal World Model 

**Title (ZH)**: UNeMo: 多模态世界模型下的协作视觉-语言推理与导航 

**Authors**: Changxin Huang, Lv Tang, Zhaohuan Zhan, Lisha Yu, Runhao Zeng, Zun Liu, Zhengjie Wang, Jianqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.18845)  

**Abstract**: Vision-and-Language Navigation (VLN) requires agents to autonomously navigate complex environments via visual images and natural language instruction--remains highly challenging. Recent research on enhancing language-guided navigation reasoning using pre-trained large language models (LLMs) has shown promising prospects. However, the reasoning of such methods is limited to the linguistic modality, lacking visual reasoning capabilities. Moreover, existing reasoning modules are optimized separately from navigation policies, leading to incompatibility and potential conflicts in optimization objectives. To tackle these challenges, we introduce UNeMo, a novel framework designed for the collaborative optimization of visual state reasoning and navigational decision-making. It introduces a Multimodal World Model (MWM) that takes visual features, language instructions, and navigational actions as inputs to jointly predict subsequent visual states, enabling cross-modal reasoning. Via a Hierarchical Prediction-Feedback (HPN) mechanism, MWM collaborates with navigation policies: the first layer generates actions using current vision-and-language features; MWM then infers post-action visual states to guide the second layer's fine-grained decisions. This forms a dynamic bidirectional promotion mechanism where MWM reasoning optimizes navigation policies, while policy decisions feedback to improve MWM's reasoning accuracy. Experiments on R2R and REVERIE datasets show UNeMo outperforms state-of-the-art methods by 2.1% and 0.7% in navigation accuracy for unseen scenes, validating its effectiveness. 

**Abstract (ZH)**: 基于视觉-语言导航（VLN）的多模态世界模型优化框架 

---
# Weakly-supervised Latent Models for Task-specific Visual-Language Control 

**Title (ZH)**: 弱监督潜在模型在特定任务中的视觉-语言控制 

**Authors**: Xian Yeow Lee, Lasitha Vidyaratne, Gregory Sin, Ahmed Farahat, Chetan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.18319)  

**Abstract**: Autonomous inspection in hazardous environments requires AI agents that can interpret high-level goals and execute precise control. A key capability for such agents is spatial grounding, for example when a drone must center a detected object in its camera view to enable reliable inspection. While large language models provide a natural interface for specifying goals, using them directly for visual control achieves only 58\% success in this task. We envision that equipping agents with a world model as a tool would allow them to roll out candidate actions and perform better in spatially grounded settings, but conventional world models are data and compute intensive. To address this, we propose a task-specific latent dynamics model that learns state-specific action-induced shifts in a shared latent space using only goal-state supervision. The model leverages global action embeddings and complementary training losses to stabilize learning. In experiments, our approach achieves 71\% success and generalizes to unseen images and instructions, highlighting the potential of compact, domain-specific latent dynamics models for spatial alignment in autonomous inspection. 

**Abstract (ZH)**: 自主危险环境下检测需要能够解释高阶目标并执行精确控制的AI代理。这类代理的关键能力是空间定位，例如，当无人机需将检测到的对象置于摄像头视场中心以实现可靠的检测时。虽然大型语言模型提供了自然的目标描述接口，但直接使用它们进行视觉控制的成功率仅为58%。我们设想装备具有世界模型的代理能够生成候选动作并在空间定位环境中表现更好，但传统世界模型需要大量的数据和计算资源。为解决这一问题，我们提出了一种针对特定任务的潜动态模型，该模型仅依赖于目标状态监督，在共享潜空间中学习由动作诱导的状态特定的转移，并利用全局动作嵌入和互补的训练损失来稳定学习。在实验中，我们的方法达到了71%的成功率，并能够在未见过的图像和指令上进行泛化，突显了紧凑的、特定领域的潜动态模型在自主检测中的空间对齐潜力。 

---
# Cross-Disciplinary Knowledge Retrieval and Synthesis: A Compound AI Architecture for Scientific Discovery 

**Title (ZH)**: 跨学科知识检索与合成：一种用于科学发现的复合AI架构 

**Authors**: Svitlana Volkova, Peter Bautista, Avinash Hiriyanna, Gabriel Ganberg, Isabel Erickson, Zachary Klinefelter, Nick Abele, Hsien-Te Kao, Grant Engberson  

**Link**: [PDF](https://arxiv.org/pdf/2511.18298)  

**Abstract**: The exponential growth of scientific knowledge has created significant barriers to cross-disciplinary knowledge discovery, synthesis and research collaboration. In response to this challenge, we present BioSage, a novel compound AI architecture that integrates LLMs with RAG, orchestrated specialized agents and tools to enable discoveries across AI, data science, biomedical, and biosecurity domains. Our system features several specialized agents including the retrieval agent with query planning and response synthesis that enable knowledge retrieval across domains with citation-backed responses, cross-disciplinary translation agents that align specialized terminology and methodologies, and reasoning agents that synthesize domain-specific insights with transparency, traceability and usability. We demonstrate the effectiveness of our BioSage system through a rigorous evaluation on scientific benchmarks (LitQA2, GPQA, WMDP, HLE-Bio) and introduce a new cross-modal benchmark for biology and AI, showing that our BioSage agents outperform vanilla and RAG approaches by 13\%-21\% powered by Llama 3.1. 70B and GPT-4o models. We perform causal investigations into compound AI system behavior and report significant performance improvements by adding RAG and agents over the vanilla models. Unlike other systems, our solution is driven by user-centric design principles and orchestrates specialized user-agent interaction workflows supporting scientific activities including but not limited to summarization, research debate and brainstorming. Our ongoing work focuses on multimodal retrieval and reasoning over charts, tables, and structured scientific data, along with developing comprehensive multimodal benchmarks for cross-disciplinary discovery. Our compound AI solution demonstrates significant potential for accelerating scientific advancement by reducing barriers between traditionally siloed domains. 

**Abstract (ZH)**: 科学知识的指数级增长创建了跨学科知识发现、综合及研究协作的重大障碍。为应对这一挑战，我们提出BioSage，一种新颖的复合人工智能架构，将LLMs与RAG集成，协调专门的代理和工具，以在AI、数据科学、生物医学和生物安全领域推动发现。我们的系统包含多种专门的代理，包括带有查询规划和响应合成的检索代理，能够提供带有引文支持的跨领域知识检索；跨学科翻译代理，能够对专业术语和方法论进行对齐；以及推理代理，能够以透明、可追溯、易用的方式综合特定领域的洞察。我们通过在科学基准（LitQA2、GPQA、WMDP、HLE-Bio）上进行严格的评估，展示了BioSage系统的效果，并引入了新的生物医学和人工智能跨模态基准，表明我们的BioSage代理在使用Llama 3.1.70B和GPT-4o模型时，相比基础模型和RAG方法，提高了13%-21%的表现。我们对复合人工智能系统行为进行了因果调查，结果显示，添加RAG和代理后，性能显著提升。我们的解决方案以用户为中心，协调专门的用户-代理交互流程，支持包括但不限于总结、研究辩论和头脑风暴在内的科学活动。我们的持续工作包括多模态检索和推理，涉及图形、表格和结构化科学数据，并正在开发跨学科发现的综合多模态基准。我们的复合人工智能解决方案展现了加速科学进步的巨大潜力，通过降低传统隔离领域之间的障碍。 

---
# QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents 

**Title (ZH)**: QuickLAP: 快速语言-动作偏好学习自主驾驶代理模型 

**Authors**: Jordan Abi Nader, David Lee, Nathaniel Dennler, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17855)  

**Abstract**: Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at this https URL. 

**Abstract (ZH)**: 机器人必须从人们的行为和言论中学习，但单靠其中任何一种模态往往是不完整的：物理纠正虽具体但意图模糊，而语言虽然能表达高层次的目标但缺乏物理基础。我们介绍QuickLAP：快速语言-行动偏好学习，这是一种融合物理反馈和语言反馈的贝叶斯框架，用于实时推断奖励函数。我们的关键见解是将语言视作用户潜在偏好的一种概率性观察，明确了哪些奖励特征重要以及物理纠正应该如何解读。QuickLAP利用大型语言模型从自由形式的表述中提取奖励特征关注掩码和偏好变化，并将其与物理反馈结合到一个封闭形式的更新规则中。这使得奖励学习快速、实时且稳健，能够处理模糊反馈。在半自主驾驶模拟器中，QuickLAP与仅基于物理反馈和启发式多模态基线相比，将奖励学习误差降低了超过70%。一项包含15名参与者的用户研究进一步验证了我们的方法：参与者发现QuickLAP更易于理解且更具协作性，并更偏好其学习行为。代码可从此链接访问。 

---
# Cognitive Inception: Agentic Reasoning against Visual Deceptions by Injecting Skepticism 

**Title (ZH)**: 认知创始：通过注入怀疑对抗视觉欺骗的主体性推理 

**Authors**: Yinjie Zhao, Heng Zhao, Bihan Wen, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.17672)  

**Abstract**: As the development of AI-generated contents (AIGC), multi-modal Large Language Models (LLM) struggle to identify generated visual inputs from real ones. Such shortcoming causes vulnerability against visual deceptions, where the models are deceived by generated contents, and the reliability of reasoning processes is jeopardized. Therefore, facing rapidly emerging generative models and diverse data distribution, it is of vital importance to improve LLMs' generalizable reasoning to verify the authenticity of visual inputs against potential deceptions. Inspired by human cognitive processes, we discovered that LLMs exhibit tendency of over-trusting the visual inputs, while injecting skepticism could significantly improve the models visual cognitive capability against visual deceptions. Based on this discovery, we propose \textbf{Inception}, a fully reasoning-based agentic reasoning framework to conduct generalizable authenticity verification by injecting skepticism, where LLMs' reasoning logic is iteratively enhanced between External Skeptic and Internal Skeptic agents. To the best of our knowledge, this is the first fully reasoning-based framework against AIGC visual deceptions. Our approach achieved a large margin of performance improvement over the strongest existing LLM baselines and SOTA performance on AEGIS benchmark. 

**Abstract (ZH)**: 基于推理的警惕性增强框架Inception：针对AIGC视觉骗发型itchen的通用可信验证 

---
# Mixture of Horizons in Action Chunking 

**Title (ZH)**: 行动片段中的混合视界 

**Authors**: Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao, Zhenyu Wei, Yunhui Liu, Zhiwu Lu, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.19433)  

**Abstract**: Vision-language-action (VLA) models have shown remarkable capabilities in robotic manipulation, but their performance is sensitive to the $\textbf{action chunk length}$ used during training, termed $\textbf{horizon}$. Our empirical study reveals an inherent trade-off: longer horizons provide stronger global foresight but degrade fine-grained accuracy, while shorter ones sharpen local control yet struggle on long-term tasks, implying fixed choice of single horizons being suboptimal. To mitigate the trade-off, we propose a $\textbf{mixture of horizons (MoH)}$ strategy. MoH rearranges the action chunk into several segments with different horizons, processes them in parallel with a shared action transformer, and fuses outputs with a light linear gate. It has three appealing benefits. 1) MoH exploits long-term foresight and short-term precision jointly within a single model, improving both performance and generalizability to complex tasks. 2) MoH is plug-and-play for full-attention action modules with minimal training or inference overhead. 3) MoH enables dynamic inference with adaptive horizons, which selects stable actions through cross-horizon consensus, achieving 2.5$\times$ higher throughput than baselines while preserving superior performance. Extensive experiments over flow-based policies $\pi_0$, $\pi_{0.5}$, and one-step regression policy $\pi_{\text{reg}}$ demonstrate that MoH yields consistent and significant gains on both simulations and real-world tasks. Notably, under mixed-task setting, $\pi_{0.5}$ with MoH reaches a new state-of-the-art with 99$\%$ average success rate on LIBERO after only $30k$ training iterations. Project page: this https URL 

**Abstract (ZH)**: Vision-语言-动作（VLA）模型在机器人操作中表现出了显著的能力，但其性能对训练过程中使用的动作片段长度（horizon）敏感。我们的实证研究揭示了一个固有的权衡：较长的horizon提供更强的全局预见性，但牺牲了细粒度的准确性；较短的horizon则增强了局部控制，但在长期任务上表现欠佳，表明固定选择单一horizon是次优的。为缓解这种权衡，我们提出了混合horizon（MoH）策略。MoH将动作片段重新排列成具有不同horizon的几个段，在共享动作变换器下并行处理，并通过轻量线性门融合输出。MoH具有三个吸引人的优点：1）MoH在一个模型中同时利用长期预见性和短期精度，提高性能和对复杂任务的泛化能力；2）MoH可以无缝集成全域注意力动作模块，且训练和推理开销极小；3）MoH支持动态推理和自适应horizon选择，通过跨horizon共识选择稳定动作，相比基础模型在吞吐量上提升了2.5倍，同时保持了出色的性能。广泛的实验表明，MoH在基于流动策略$\pi_0$、$\pi_{0.5}$和一步回归策略$\pi_{\text{reg}}$的情况下，在仿真和真实任务中都提供了稳定的显著改进。值得注意的是，在混合任务设置下，$\pi_{0.5}$结合MoH在仅3万次训练迭代后，在LIBERO上的平均成功率达到99%。新网站：this https URL 

---
# DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research 

**Title (ZH)**: DR TTLU: 基于动态评价标准的强化学习进行深度研究 

**Authors**: Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen-tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, Pang Wei Koh  

**Link**: [PDF](https://arxiv.org/pdf/2511.19399)  

**Abstract**: Deep research models perform multi-step research to produce long-form, well-attributed answers. However, most open deep research models are trained on easily verifiable short-form QA tasks via reinforcement learning with verifiable rewards (RLVR), which does not extend to realistic long-form tasks. We address this with Reinforcement Learning with Evolving Rubrics (RLER), in which we construct and maintain rubrics that co-evolve with the policy model during training; this allows the rubrics to incorporate information that the model has newly explored and to provide discriminative, on-policy feedback. Using RLER, we develop Deep Research Tulu (DR Tulu-8B), the first open model that is directly trained for open-ended, long-form deep research. Across four long-form deep research benchmarks in science, healthcare and general domains, DR Tulu substantially outperforms existing open deep research models, and matches or exceeds proprietary deep research systems, while being significantly smaller and cheaper per query. To facilitate future research, we release all data, models, and code, including our new MCP-based agent infrastructure for deep research systems. 

**Abstract (ZH)**: 基于演化的评估标准强化学习（RLER） training for open-ended long-form deep research 

---
# SENTINEL: A Fully End-to-End Language-Action Model for Humanoid Whole Body Control 

**Title (ZH)**: SENTINEL：一种用于类人全身控制的端到端语言-动作模型 

**Authors**: Yuxuan Wang, Haobin Jiang, Shiqing Yao, Ziluo Ding, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19236)  

**Abstract**: Existing humanoid control systems often rely on teleoperation or modular generation pipelines that separate language understanding from physical execution. However, the former is entirely human-driven, and the latter lacks tight alignment between language commands and physical behaviors. In this paper, we present SENTINEL, a fully end-to-end language-action model for humanoid whole-body control. We construct a large-scale dataset by tracking human motions in simulation using a pretrained whole body controller, combined with their text annotations. The model directly maps language commands and proprioceptive inputs to low-level actions without any intermediate representation. The model generates action chunks using flow matching, which can be subsequently refined by a residual action head for real-world deployment. Our method exhibits strong semantic understanding and stable execution on humanoid robots in both simulation and real-world deployment, and also supports multi-modal extensions by converting inputs into texts. 

**Abstract (ZH)**: 现有的类人控制系统通常依赖于远端操作或模块化生成管道，将语言理解与物理执行分离。然而，前者完全由人类驱动，后者缺乏语言命令与物理行为之间的紧密对齐。本文提出了一种全新的端到端语言-行动模型SENTINEL，用于类人全身控制。通过使用预训练的全身控制器在仿真中追踪人类动作，并结合其文本注释构建大规模数据集。该模型可以直接将语言命令和本体感受输入映射为低级动作，无需中间表示。模型使用流匹配生成动作片段，这些片段可以通过残差动作头进一步优化，以适应实际部署。该方法在仿真和实际部署中均表现出强大的语义理解和稳定的执行性能，并支持多模态扩展，能够将输入转换为文本。 

---
# Accelerating Reinforcement Learning via Error-Related Human Brain Signals 

**Title (ZH)**: 基于错误相关的脑信号加速强化学习 

**Authors**: Suzie Kim, Hye-Bin Shin, Hyo-Jeong Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18878)  

**Abstract**: In this work, we investigate how implicit neural feed back can accelerate reinforcement learning in complex robotic manipulation settings. While prior electroencephalogram (EEG) guided reinforcement learning studies have primarily focused on navigation or low-dimensional locomotion tasks, we aim to understand whether such neural evaluative signals can improve policy learning in high-dimensional manipulation tasks involving obstacles and precise end-effector control. We integrate error related potentials decoded from offline-trained EEG classifiers into reward shaping and systematically evaluate the impact of human-feedback weighting. Experiments on a 7-DoF manipulator in an obstacle-rich reaching environment show that neural feedback accelerates reinforcement learning and, depending on the human-feedback weighting, can yield task success rates that at times exceed those of sparse-reward baselines. Moreover, when applying the best-performing feedback weighting across all sub jects, we observe consistent acceleration of reinforcement learning relative to the sparse-reward setting. Furthermore, leave-one subject-out evaluations confirm that the proposed framework remains robust despite the intrinsic inter-individual variability in EEG decodability. Our findings demonstrate that EEG-based reinforcement learning can scale beyond locomotion tasks and provide a viable pathway for human-aligned manipulation skill acquisition. 

**Abstract (ZH)**: 本研究探讨隐式神经反馈如何在复杂机器人操作环境中加速强化学习。尽管先前基于脑电图（EEG）的强化学习研究主要关注导航或低维度运动任务，我们旨在了解此类神经评估信号是否能提高涉及障碍和精确末端执行器控制的高维度操作任务中的策略学习。我们将从离线训练的EEG分类器中解码的错误相关电位整合到奖励塑造中，并系统评估人工反馈权重的影响。在障碍丰富的环境中使用7自由度操作器的实验表明，神经反馈可以加速强化学习，并且根据人工反馈权重的不同，有时可以达到甚至超过稀疏奖励基线的成功率。此外，当使用表现最佳的反馈权重跨所有被试时，我们观察到相对于稀疏奖励设置的稳健加速。进一步的去除一个被试的交叉验证表明，所提出的框架在EEG解码的个体差异内在性的背景下仍然保持稳健。我们的研究结果表明，基于脑电图的强化学习可以超越运动任务，并为人类对齐的操作技能获取提供可行途径。 

---
# Any4D: Open-Prompt 4D Generation from Natural Language and Images 

**Title (ZH)**: Any4D: 从自然语言和图像生成开放提示4D内容 

**Authors**: Hao Li, Qiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.18746)  

**Abstract**: While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a \textit{"GPT moment"} in the embodied domain. There is a naive observation: \textit{the diversity of embodied data far exceeds the relatively small space of possible primitive motions}. Based on this insight, we propose \textbf{Primitive Embodied World Models} (PEWM), which restricts video generation to fixed shorter horizons, our approach \textit{1) enables} fine-grained alignment between linguistic concepts and visual representations of robotic actions, \textit{2) reduces} learning complexity, \textit{3) improves} data efficiency in embodied data collection, and \textit{4) decreases} inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence. 

**Abstract (ZH)**: 基于视频生成的体态世界模型虽日益受到关注，但对其依赖大规模体态交互数据仍是一个关键瓶颈。体态数据的稀缺性、收集难度高以及高维度特性基本限制了语言与动作的精确对齐，并加剧了长时序视频生成的挑战——阻碍生成模型在体态领域实现“GPT时刻”。一个朴素的观察是：体态数据的多样性远远超过可能的基本运动空间。基于此洞察，我们提出了**基本体态世界模型**（PEWM），该模型限制视频生成在固定较短的时间范围内，我们的方法**1）实现语言概念与机器人动作的视觉表示的精细对齐，2）降低学习复杂度，3）提高体态数据收集的数据效率，4）减少推理延迟**。通过配备模块化视觉语言模型（VLM）规划器和起始目标热图引导机制（SGG），PEWM 进一步支持灵活的闭环控制，并促进基本级别策略在扩展复杂任务中的组合泛化。我们的框架利用视频模型中的时空视觉先验和 VLM 的语义意识，弥合细微物理交互与高层推理之间的差距，为可扩展、可解释和通用的体态智能铺平道路。 

---
# General Agentic Memory Via Deep Research 

**Title (ZH)**: 深度研究驱动的总体代理记忆 

**Authors**: B.Y. Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18423)  

**Abstract**: Memory is critical for AI agents, yet the widely-adopted static memory, aiming to create readily available memory in advance, is inevitably subject to severe information loss. To address this limitation, we propose a novel framework called \textbf{general agentic memory (GAM)}. GAM follows the principle of "\textbf{just-in time (JIT) compilation}" where it focuses on creating optimized contexts for its client at runtime while keeping only simple but useful memory during the offline stage. To this end, GAM employs a duo-design with the following components. 1) \textbf{Memorizer}, which highlights key historical information using a lightweight memory, while maintaining complete historical information within a universal page-store. 2) \textbf{Researcher}, which retrieves and integrates useful information from the page-store for its online request guided by the pre-constructed memory. This design allows GAM to effectively leverage the agentic capabilities and test-time scalability of frontier large language models (LLMs), while also facilitating end-to-end performance optimization through reinforcement learning. In our experimental study, we demonstrate that GAM achieves substantial improvement on various memory-grounded task completion scenarios against existing memory systems. 

**Abstract (ZH)**: 通用代理记忆（GAM）：面向需求的动态记忆框架 

---
# Toward an AI-Native Internet: Rethinking the Web Architecture for Semantic Retrieval 

**Title (ZH)**: 面向AI原生互联网：重新思考面向语义检索的网络架构 

**Authors**: Muhammad Bilal, Zafar Qazi, Marco Canini  

**Link**: [PDF](https://arxiv.org/pdf/2511.18354)  

**Abstract**: The rise of Generative AI Search is fundamentally transforming how users and intelligent systems interact with the Internet. LLMs increasingly act as intermediaries between humans and web information. Yet the web remains optimized for human browsing rather than AI-driven semantic retrieval, resulting in wasted network bandwidth, lower information quality, and unnecessary complexity for developers. We introduce the concept of an AI-Native Internet, a web architecture in which servers expose semantically relevant information chunks rather than full documents, supported by a Web-native semantic resolver that allows AI applications to discover relevant information sources before retrieving fine-grained chunks. Through motivational experiments, we quantify the inefficiencies of current HTML-based retrieval, and outline architectural directions and open challenges for evolving today's document-centric web into an AI-oriented substrate that better supports semantic access to web content. 

**Abstract (ZH)**: 生成式AI搜索的兴起从根本上 transforming 如何用户和智能系统与互联网互动。大型语言模型 (LLMs) 越来越多地成为人类与网络信息之间的中介。然而，网络仍然优化了供人类浏览，而不是基于AI的语义检索，导致浪费的网络带宽、较低的信息质量以及开发人员不必要的复杂性。我们介绍了AI原生互联网的概念，这是一种网络架构，在这种架构中，服务器提供语义相关的信息片段而不是完整的文档，并且通过一个原生的Web语义解析器，使AI应用程序能够在检索细粒度片段之前发现相关的信息来源。通过动机实验，我们量化了当前基于HTML的检索的无效性，并概述了架构方向和挑战，以便将当前以文档为中心的网络演变成一个更加支持网络内容语义访问的AI导向的基础结构。 

---
# Beyond Words and Pixels: A Benchmark for Implicit World Knowledge Reasoning in Generative Models 

**Title (ZH)**: 超越文字和像素：生成模型中隐式世界知识推理的基准 

**Authors**: Tianyang Han, Junhao Su, Junjie Hu, Peizhen Yang, Hengyu Shi, Junfeng Luo, Jialin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.18271)  

**Abstract**: Text-to-image (T2I) models today are capable of producing photorealistic, instruction-following images, yet they still frequently fail on prompts that require implicit world knowledge. Existing evaluation protocols either emphasize compositional alignment or rely on single-round VQA-based scoring, leaving critical dimensions such as knowledge grounding, multi-physics interactions, and auditable evidence-substantially undertested. To address these limitations, we introduce PicWorld, the first comprehensive benchmark that assesses the grasp of implicit world knowledge and physical causal reasoning of T2I models. This benchmark consists of 1,100 prompts across three core categories. To facilitate fine-grained evaluation, we propose PW-Agent, an evidence-grounded multi-agent evaluator to hierarchically assess images on their physical realism and logical consistency by decomposing prompts into verifiable visual evidence. We conduct a thorough analysis of 17 mainstream T2I models on PicWorld, illustrating that they universally exhibit a fundamental limitation in their capacity for implicit world knowledge and physical causal reasoning to varying degrees. The findings highlight the need for reasoning-aware, knowledge-integrative architectures in future T2I systems. 

**Abstract (ZH)**: 基于文本到图像的模型在捕捉隐含世界知识和物理因果推理能力方面的综合评估：PicWorld基准 

---
# Hybrid Agentic AI and Multi-Agent Systems in Smart Manufacturing 

**Title (ZH)**: 混合自主AI与多Agent系统在智能制造中的应用 

**Authors**: Mojtaba A. Farahani, Md Irfan Khan, Thorsten Wuest  

**Link**: [PDF](https://arxiv.org/pdf/2511.18258)  

**Abstract**: The convergence of Agentic AI and MAS enables a new paradigm for intelligent decision making in SMS. Traditional MAS architectures emphasize distributed coordination and specialized autonomy, while recent advances in agentic AI driven by LLMs introduce higher order reasoning, planning, and tool orchestration capabilities. This paper presents a hybrid agentic AI and multi agent framework for a Prescriptive Maintenance use case, where LLM based agents provide strategic orchestration and adaptive reasoning, complemented by rule based and SLMs agents performing efficient, domain specific tasks on the edge. The proposed framework adopts a layered architecture that consists of perception, preprocessing, analytics, and optimization layers, coordinated through an LLM Planner Agent that manages workflow decisions and context retention. Specialized agents autonomously handle schema discovery, intelligent feature analysis, model selection, and prescriptive optimization, while a HITL interface ensures transparency and auditability of generated maintenance recommendations. This hybrid design supports dynamic model adaptation, cost efficient maintenance scheduling, and interpretable decision making. An initial proof of concept implementation is validated on two industrial manufacturing datasets. The developed framework is modular and extensible, supporting seamless integration of new agents or domain modules as capabilities evolve. The results demonstrate the system capability to automatically detect schema, adapt preprocessing pipelines, optimize model performance through adaptive intelligence, and generate actionable, prioritized maintenance recommendations. The framework shows promise in achieving improved robustness, scalability, and explainability for RxM in smart manufacturing, bridging the gap between high level agentic reasoning and low level autonomous execution. 

**Abstract (ZH)**: 基于代理AI和多代理系统的融合在智能维护决策中的新范式 

---
# A New Error Temporal Difference Algorithm for Deep Reinforcement Learning in Microgrid Optimization 

**Title (ZH)**: 一种用于微电网优化的新型误差时差算法 

**Authors**: Fulong Yao, Wanqing Zhao, Matthew Forshaw  

**Link**: [PDF](https://arxiv.org/pdf/2511.18093)  

**Abstract**: Predictive control approaches based on deep reinforcement learning (DRL) have gained significant attention in microgrid energy optimization. However, existing research often overlooks the issue of uncertainty stemming from imperfect prediction models, which can lead to suboptimal control strategies. This paper presents a new error temporal difference (ETD) algorithm for DRL to address the uncertainty in predictions,aiming to improve the performance of microgrid operations. First,a microgrid system integrated with renewable energy sources (RES) and energy storage systems (ESS), along with its Markov decision process (MDP), is modelled. Second, a predictive control approach based on a deep Q network (DQN) is presented, in which a weighted average algorithm and a new ETD algorithm are designed to quantify and address the prediction uncertainty, respectively. Finally, simulations on a realworld US dataset suggest that the developed ETD effectively improves the performance of DRL in optimizing microgrid operations. 

**Abstract (ZH)**: 基于深度强化学习的误差时差(DRL-ETD)预测控制方法在微电网能源优化中的应用 

---
# Continually Evolving Skill Knowledge in Vision Language Action Model 

**Title (ZH)**: 持续演化技能知识的视觉语言行动模型 

**Authors**: Yuxuan Wu, Guangming Wang, Zhiheng Yang, Maoqing Yao, Brian Sheil, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18085)  

**Abstract**: Developing general robot intelligence in open environments requires continual skill learning. Recent Vision-Language-Action (VLA) models leverage massive pretraining data to support diverse manipulation tasks, but they still depend heavily on task-specific fine-tuning, revealing a lack of continual learning capability. Existing continual learning methods are also resource-intensive to scale to VLA models. We propose Stellar VLA, a knowledge-driven continual learning framework with two variants: T-Stellar, modeling task-centric knowledge space, and TS-Stellar, capturing hierarchical task-skill structure. Stellar VLA enables self-supervised knowledge evolution through joint learning of task latent representation and the knowledge space, reducing annotation needs. Knowledge-guided expert routing provide task specialization without extra network parameters, lowering training this http URL on the LIBERO benchmark and real-world tasks show over 50 percentage average improvement in final success rates relative to baselines. TS-Stellar further excels in complex action inference, and in-depth analyses verify effective knowledge retention and discovery. Our code will be released soon. 

**Abstract (ZH)**: 开放环境下开发通用机器人智能需要持续技能学习。我们提出的Stellar VLA是一种知识驱动的持续学习框架，包括两种变体：T-Stellar建模任务中心的知识空间，TS-Stellar捕获任务-技能层次结构。Stellar VLA通过联合学习任务潜在表示和知识空间实现自我监督的知识进化，减少标注需求。知识引导的专家路由无需额外网络参数提供任务专业化。基准和真实世界任务上的实验表明，与基线相比，最终成功率平均提高超过50个百分点。TS-Stellar在复杂动作推断方面表现出色，深入分析验证了有效知识保留和发现。soon我们将发布代码。 

---
# Toward explainable AI approaches for breast imaging: adapting foundation models to diverse populations 

**Title (ZH)**: 面向可解释的人工智能方法在乳腺成像中的应用：将基础模型适应于多元人群 

**Authors**: Guilherme J. Cavalcante, José Gabriel A. Moreira, Gabriel A.B. do Nascimento, Vincent Dong, Alex Nguyen, Thaís G. do Rêgo, Yuri Malheiros, Telmo M. Silva Filho, Carla R. Zeballos Torrez, James C. Gee, Anne Marie McCarthy, Andrew D. A. Maidment, Bruno Barufaldi  

**Link**: [PDF](https://arxiv.org/pdf/2511.17828)  

**Abstract**: Foundation models hold promise for specialized medical imaging tasks, though their effectiveness in breast imaging remains underexplored. This study leverages BiomedCLIP as a foundation model to address challenges in model generalization. BiomedCLIP was adapted for automated BI-RADS breast density classification using multi-modality mammographic data (synthesized 2D images, digital mammography, and digital breast tomosynthesis). Using 96,995 images, we compared single-modality (s2D only) and multi-modality training approaches, addressing class imbalance through weighted contrastive learning. Both approaches achieved similar accuracy (multi-modality: 0.74, single-modality: 0.73), with the multi-modality model offering broader applicability across different imaging modalities and higher AUC values consistently above 0.84 across BI-RADS categories. External validation on the RSNA and EMBED datasets showed strong generalization capabilities (AUC range: 0.80-0.93). GradCAM visualizations confirmed consistent and clinically relevant attention patterns, highlighting the models interpretability and robustness. This research underscores the potential of foundation models for breast imaging applications, paving the way for future extensions for diagnostic tasks. 

**Abstract (ZH)**: 基础模型在专门的乳腺成像任务中展现出潜力，虽然其在乳腺成像领域的有效性尚未充分探索。本研究利用BiomedCLIP作为基础模型以应对模型泛化的挑战。BiomedCLIP被改编用于自动化的BI-RADS乳腺密度分类，采用多模态乳腺成像数据（合成的2D图像、数字乳腺X线摄影和数字乳腺断层摄影）。通过96,995张图像，我们对比了单模态（仅2D）和多模态训练方法，并通过加权对比学习解决类别不平衡问题。两种方法均达到了相似的准确性（多模态：0.74，单模态：0.73），其中多模态模型在不同成像模态上具有更广泛的应用性，并且在BI-RADS类别中AUC值一致保持在0.84以上。外部验证显示，该模型具有强烈的泛化能力（AUC范围：0.80-0.93）。GradCAM可视化结果确认了稳健且临床相关关注模式，突显了模型的可解释性和鲁棒性。这项研究强调了基础模型在乳腺影像应用中的潜力，为未来的诊断任务扩展铺平了道路。 

---
# Pillar-0: A New Frontier for Radiology Foundation Models 

**Title (ZH)**: 支柱-0：放射学基础模型的新前沿 

**Authors**: Kumar Krishna Agrawal, Longchao Liu, Long Lian, Michael Nercessian, Natalia Harguindeguy, Yufu Wu, Peter Mikhael, Gigin Lin, Lecia V. Sequist, Florian Fintelmann, Trevor Darrell, Yutong Bai, Maggie Chung, Adam Yala  

**Link**: [PDF](https://arxiv.org/pdf/2511.17803)  

**Abstract**: Radiology plays an integral role in modern medicine, yet rising imaging volumes have far outpaced workforce growth. Foundation models offer a path toward assisting with the full spectrum of radiology tasks, but existing medical models remain limited: they process volumetric CT and MRI as low-fidelity 2D slices, discard critical grayscale contrast information, and lack evaluation frameworks that reflect real clinical practice. We introduce Pillar-0, a radiology foundation model pretrained on 42,990 abdomen-pelvis CTs, 86,411 chest CTs, 14,348 head CTs, and 11,543 breast MRIs from a large academic center, together with RATE, a scalable framework that extracts structured labels for 366 radiologic findings with near-perfect accuracy using LLMs. Across internal test sets of 14,230 abdomen-pelvis CTs, 10,646 chest CTs, 4,906 head CTs, and 1,585 breast MRIs, Pillar-0 establishes a new performance frontier, achieving mean AUROCs of 86.4, 88.0, 90.1, and 82.9, outperforming MedGemma (Google), MedImageInsight (Microsoft), Lingshu (Alibaba), and Merlin (Stanford) by 7.8-15.8 AUROC points and ranking best in 87.2\% (319/366) tasks. Pillar-0 similarly outperforms all baselines in an external validation on the Stanford Abdominal CT dataset, including Merlin (82.2 vs 80.6 AUROC). Pillar-0 extends to tasks beyond its pretraining, such as long-horizon lung cancer risk prediction, where it improves upon the state-of-the-art Sybil by 3.0 C-index points on NLST, and generalizes with gains of 5.9 (MGH) and 1.9 (CGMH). In brain hemorrhage detection, Pillar-0 obtained a >95 AUROC when using only 1/20th of the data of the next most sample efficient baseline. Pillar-0 and RATE together provide an open, clinically rigorous foundation for building high-performance radiology systems, enabling applications that were previously infeasible due to computational, data, and evaluation constraints. 

**Abstract (ZH)**: 辐射成像在现代医学中扮演着至关重要的角色，然而，成像量的增加远远超过了 workforce 的增长。基础模型为协助涵盖放射学全流程任务提供了一条途径，但现有的医疗模型仍存在局限：它们将多维CT和MRI处理为低保真2D切片，丢弃了关键的灰度对比信息，并缺乏反映真实临床实践的评估框架。我们引入了 Pillar-0，该模型基于某大型学术中心的42,990例腹部骨盆CT、86,411例胸部CT、14,348例头部CT和11,543例乳腺MRI进行了预训练，并提出了 RATE，这是一种可扩展的框架，利用LLMs以近乎完美的准确度从数据中提取结构化标签，涵盖366项放射学发现。在内部测试数据集上（包含14,230例腹部骨盆CT、10,646例胸部CT、4,906例头部CT和1,585例乳腺MRI），Pillar-0 设立了新的性能前沿，达到了平均AUCROC值86.4、88.0、90.1和82.9，并在87.2%（319/366）的任务中表现最佳，均优于MedGemma（Google）、MedImageInsight（Microsoft）、Lingshu（Alibaba）和Merlin（Stanford），领先幅度为7.8至15.8个AUCROC点。在斯坦福腹部CT数据集的外部验证中，Pillar-0 也优于所有基线模型（包括Merlin，其AUCROC值为80.6，而Pillar-0为82.2）。Pillar-0 能够将其预训练应用到超出其预训练范围的任务中，例如长期肺癌风险预测，其在NLST中的C指数提高了3.0个点，同时在MGH和CGMH上分别获得5.9和1.9的增益。在脑出血检测中，Pillar-0 使用最少的20分之一数据就达到了超过95的AUCROC。Pillar-0 和 RATE 一起提供了一个开放且临床严谨的基础，用于构建高性能的放射学系统，使以前因计算、数据和评估限制而难以实现的应用成为可能。 

---
# Empa: An AI-Powered Virtual Mentor for Developing Global Collaboration Skills in HPC Education 

**Title (ZH)**: Empa：一种基于人工智能的虚拟导师，用于在高性能计算教育中培养全球合作技能 

**Authors**: Ashish, Aparajita Jaiswal, Sudip Vhaduri, Niveditha Nerella, Shubham Jha  

**Link**: [PDF](https://arxiv.org/pdf/2511.17669)  

**Abstract**: High-performance computing (HPC) and parallel computing increasingly rely on global collaboration among diverse teams, yet traditional computing curricula inadequately prepare students for cross-cultural teamwork essential in modern computational research environments. This paper presents Empa, an AI-powered virtual mentor that integrates intercultural collaboration training into undergraduate computing education. Built using large language models and deployed through a progressive web application, Empa guides students through structured activities covering cultural dimensions, communication styles, and conflict resolution that are critical for effective multicultural teamwork. Our system addresses the growing need for culturally competent HPC professionals by helping computing students develop skills to collaborate effectively in international research teams, contribute to global computational projects, and navigate the cultural complexities inherent in distributed computing environments. Pilot preparation for deployment in computing courses demonstrates the feasibility of AI-mediated intercultural training and provides insights into scalable approaches for developing intercultural collaboration skills essential for HPC workforce development. 

**Abstract (ZH)**: 高性能计算(HPC)和并行计算日益依赖于多元化团队之间的全球合作，然而传统计算课程未能充分准备学生进行跨文化团队合作，这是现代计算研究环境中必不可少的技能。本文介绍了Empa，一种基于人工智能的虚拟导师，将跨文化协作培训整合到本科生计算教育中。Empa使用大规模语言模型构建并通过渐进式网络应用程序部署，引导学生完成涵盖文化维度、沟通风格和冲突解决等内容的结构化活动，这些都是有效进行跨文化团队合作的关键。我们的系统通过帮助计算专业的学生发展在国际研究团队中有效合作的技能，促进全球计算项目的贡献，并应对分布式计算环境中固有的文化复杂性，从而解决了对具备文化胜任力的HPC专业人才日益增长的需求。部署前的试点准备工作表明了以人工智能中介的跨文化交流培训的可行性，并提供了开发对于HPC劳动力发展至关重要的跨文化协作技能的可扩展方法的见解。 

---
# Dialogue Diplomats: An End-to-End Multi-Agent Reinforcement Learning System for Automated Conflict Resolution and Consensus Building 

**Title (ZH)**: 对话外交官：自动冲突解决与共识构建的端到端多代理 reinforcement 学习系统 

**Authors**: Deepak Bolleddu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17654)  

**Abstract**: Conflict resolution and consensus building represent critical challenges in multi-agent systems, negotiations, and collaborative decision-making processes. This paper introduces Dialogue Diplomats, a novel end-to-end multi-agent reinforcement learning (MARL) framework designed for automated conflict resolution and consensus building in complex, dynamic environments. The proposed system integrates advanced deep reinforcement learning architectures with dialogue-based negotiation protocols, enabling autonomous agents to engage in sophisticated conflict resolution through iterative communication and strategic adaptation. We present three primary contributions: first, a novel Hierarchical Consensus Network (HCN) architecture that combines attention mechanisms with graph neural networks to model inter-agent dependencies and conflict dynamics. second, a Progressive Negotiation Protocol (PNP) that structures multi-round dialogue interactions with adaptive concession strategies; and third, a Context-Aware Reward Shaping mechanism that balances individual agent objectives with collective consensus goals. 

**Abstract (ZH)**: 冲突解决与共识构建是多智能体系统、谈判及协作决策过程中面临的关键挑战。本文介绍了一种新的端到端多智能体强化学习（MARL）框架——Dialogue Diplomats，该框架旨在在复杂动态环境中自动进行冲突解决与共识构建。所提出的系统将先进的深度强化学习架构与基于对话的谈判协议相结合，使自主智能体能够通过迭代通信和策略调整进行复杂的冲突解决。本文主要贡献包括：首先，提出了一种新的层次共识网络（HCN）架构，该架构结合了注意力机制与图神经网络以建模智能体间的依赖关系和冲突动态；其次，提出了一种渐进式谈判协议（PNP），该协议结构化了多轮对话交互并采用了自适应让步策略；最后，提出了一种上下文感知的奖励塑形机制，该机制平衡了个体智能体目标与集体共识目标。 

---
# SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios 

**Title (ZH)**: SWITCH: 评估长时人体交互场景中实体界面的建模与处理基准 

**Authors**: Jieru Lin, Zhiwei Yu, Börje F. Karlsson  

**Link**: [PDF](https://arxiv.org/pdf/2511.17649)  

**Abstract**: Autonomous intelligence requires not only perception and reasoning, but critically, effective interaction with the existing world and its infrastructure. Everyday environments are rich in tangible control interfaces (TCIs), e.g., light switches, appliance panels, and embedded GUIs, that demand commonsense and physics reasoning, but also causal prediction and outcome verification in time and space (e.g., delayed heating, remote lights). Moreover, failures here have potential safety implications, yet current benchmarks rarely test grounding, partial observability (video), or post-hoc verification in situated settings. We introduce SWITCH (Semantic World Interface Tasks for Control and Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities:task-aware VQA, semantic UI grounding, action generation, state-transition prediction, and result verification, under egocentric RGB video input and device diversity. Across 351 tasks spanning 98 real devices and appliances, commercial and open LMMMs exhibit inconsistent performance even on single-step interactions, often over-relying on textual cues and under-using visual or video evidence (and high aggregate scores can mask such failures). SWITCH provides data, code, and held-out splits to enable reproducible evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of training datasets. Benchmark resources are available at: this https URL. 

**Abstract (ZH)**: 自主智能不仅需要感知和推理，最关键的是有效与现有世界及其基础设施进行交互。日常环境充满了实体控制接口（TCIs），例如开关、电器面板和嵌入式GUI，这些接口需要常识和物理推理能力，同时也需要时间和空间内的因果预测和结果验证（例如，延迟加热、远程灯光）。此外，这里的故障可能有潜在的安全影响，但目前的基准测试很少测试环境嵌入性、部分可观性（视频）或事后验证。我们引入了SWITCH（语义世界接口任务集，用于控制与处理），这是一个逐步迭代发布的实体驱动基准，旨在探测这些差距。其第一版SWITCH-Basic评估了五种互补的能力：任务感知VQA、语义UI接地、动作生成、状态转换预测和结果验证，这些评估基于第一人称RGB视频输入和设备多样性。在涵盖98种真实设备和电器的351个任务中，即使对于单一步骤的交互，商用和开源LMMMs的表现也不一致，经常过度依赖文本线索，并未充分利用视觉或视频证据（高综合得分可能掩盖这些失败）。SWITCH提供了数据、代码和保留集，以促进可重复评估和社区对更具有挑战性的基准迭代及训练数据集创建的贡献。基准资源可在如下链接获取：this https URL。 

---
# Boosting Reinforcement Learning in 3D Visuospatial Tasks Through Human-Informed Curriculum Design 

**Title (ZH)**: 通过人类导向的课程设计提升三维 visuospatial 任务中的强化学习 

**Authors**: Markus D. Solbach, John K. Tsotsos  

**Link**: [PDF](https://arxiv.org/pdf/2511.17595)  

**Abstract**: Reinforcement Learning is a mature technology, often suggested as a potential route towards Artificial General Intelligence, with the ambitious goal of replicating the wide range of abilities found in natural and artificial intelligence, including the complexities of human cognition. While RL had shown successes in relatively constrained environments, such as the classic Atari games and specific continuous control problems, recent years have seen efforts to expand its applicability. This work investigates the potential of RL in demonstrating intelligent behaviour and its progress in addressing more complex and less structured problem domains.
We present an investigation into the capacity of modern RL frameworks in addressing a seemingly straightforward 3D Same-Different visuospatial task. While initial applications of state-of-the-art methods, including PPO, behavioural cloning and imitation learning, revealed challenges in directly learning optimal strategies, the successful implementation of curriculum learning offers a promising avenue. Effective learning was achieved by strategically designing the lesson plan based on the findings of a real-world human experiment. 

**Abstract (ZH)**: 强化学习是一种成熟的技術，常被認為是通往人工通用智能的一條潛在途徑，其雄心勃勃的目標是複製天然和人工智能的廣泛能力，包括人類認知的複雜性。雖然強化學習在相對受限的環境中，如 CLASSIC 阿特利遊戲和特定的連續控制問題中取得了成功，但在最近幾年，已經有了擴展其適用範圍的努力。本研究探討了強化學習在展示智能行為方面的潛力，以及其在處理更複雜和不規則問題領域的進展。

現代強化學習框架在處理看似簡單的3D同異空間任務方面的能力研究。雖然最佔優方法，包括PPO、行為克隆和 imitation learning 初始應用揭示了直接學習最佳策略的挑戰，但成功實現曲線學習為這一領域開闊了新的前景。通過基於現實世界人類實驗的發現策略性設計教學計畫，實現了有效的學習。 

---
# Enhancing Robustness of Offline Reinforcement Learning Under Data Corruption via Sharpness-Aware Minimization 

**Title (ZH)**: 基于尖锐性意识最小化提升数据腐蚀下离线强化学习的鲁棒性 

**Authors**: Le Xu, Jiayu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.17568)  

**Abstract**: Offline reinforcement learning (RL) is vulnerable to real-world data corruption, with even robust algorithms failing under challenging observation and mixture corruptions. We posit this failure stems from data corruption creating sharp minima in the loss landscape, leading to poor generalization. To address this, we are the first to apply Sharpness-Aware Minimization (SAM) as a general-purpose, plug-and-play optimizer for offline RL. SAM seeks flatter minima, guiding models to more robust parameter regions. We integrate SAM into strong baselines for data corruption: IQL, a top-performing offline RL algorithm in this setting, and RIQL, an algorithm designed specifically for data-corruption robustness. We evaluate them on D4RL benchmarks with both random and adversarial corruption. Our SAM-enhanced methods consistently and significantly outperform the original baselines. Visualizations of the reward surface confirm that SAM finds smoother solutions, providing strong evidence for its effectiveness in improving the robustness of offline RL agents. 

**Abstract (ZH)**: offline reinforcement learning (RL) 在现实世界数据 corruption 下易受攻击，即使稳健的算法在具有挑战性的观察和混合 corruption 下也会失效。我们提出这一失败源自数据 corruption 在损失景观上创建尖锐的极小值，导致模型泛化能力差。为解决这一问题，我们首次将 Sharpness-Aware Minimization (SAM) 作为 offline RL 的通用、即插即用优化器进行应用。SAM 寻找较平滑的极小值，引导模型进入更稳健的参数区域。我们将 SAM 集成到数据 corruption 强基准中：IQL，该设置下性能最佳的 offline RL 算法，以及 RIQL，一种专门设计以增强数据-corruption 抗性的算法。我们在包含随机和对抗 corruption 的 D4RL 基准上评估它们。我们的 SAM 强化方法在各个方面都显著优于原始基准。对回报表面的可视化证实，SAM 寻找更平滑的解决方案，提供了其在提高 offline RL 代理的鲁棒性方面有效性的强大证据。 

---
