# Do LLMs trust AI regulation? Emerging behaviour of game-theoretic LLM agents 

**Title (ZH)**: Do LLMs遵守AI监管？博弈 theoretic LLM代理的新兴行为 

**Authors**: Alessio Buscemi, Daniele Proverbio, Paolo Bova, Nataliya Balabanova, Adeela Bashir, Theodor Cimpeanu, Henrique Correia da Fonseca, Manh Hong Duong, Elias Fernandez Domingos, Antonio M. Fernandes, Marcus Krellner, Ndidi Bianca Ogbo, Simon T. Powers, Fernando P. Santos, Zia Ush Shamszaman, Zhao Song, Alessandro Di Stefano, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.08640)  

**Abstract**: There is general agreement that fostering trust and cooperation within the AI development ecosystem is essential to promote the adoption of trustworthy AI systems. By embedding Large Language Model (LLM) agents within an evolutionary game-theoretic framework, this paper investigates the complex interplay between AI developers, regulators and users, modelling their strategic choices under different regulatory scenarios. Evolutionary game theory (EGT) is used to quantitatively model the dilemmas faced by each actor, and LLMs provide additional degrees of complexity and nuances and enable repeated games and incorporation of personality traits. Our research identifies emerging behaviours of strategic AI agents, which tend to adopt more "pessimistic" (not trusting and defective) stances than pure game-theoretic agents. We observe that, in case of full trust by users, incentives are effective to promote effective regulation; however, conditional trust may deteriorate the "social pact". Establishing a virtuous feedback between users' trust and regulators' reputation thus appears to be key to nudge developers towards creating safe AI. However, the level at which this trust emerges may depend on the specific LLM used for testing. Our results thus provide guidance for AI regulation systems, and help predict the outcome of strategic LLM agents, should they be used to aid regulation itself. 

**Abstract (ZH)**: 促进可信人工智能系统采用的关键在于AI开发生态系统中信任与合作的培养：基于大型语言模型的进化博弈论框架下的战略行为研究 

---
# Towards an Evaluation Framework for Explainable Artificial Intelligence Systems for Health and Well-being 

**Title (ZH)**: 面向健康与福祉的可解释人工智能系统评估框架研究 

**Authors**: Esperança Amengual-Alcover, Antoni Jaume-i-Capó, Miquel Miró-Nicolau, Gabriel Moyà-Alcover, Antonia Paniza-Fullana  

**Link**: [PDF](https://arxiv.org/pdf/2504.08552)  

**Abstract**: The integration of Artificial Intelligence in the development of computer systems presents a new challenge: make intelligent systems explainable to humans. This is especially vital in the field of health and well-being, where transparency in decision support systems enables healthcare professionals to understand and trust automated decisions and predictions. To address this need, tools are required to guide the development of explainable AI systems. In this paper, we introduce an evaluation framework designed to support the development of explainable AI systems for health and well-being. Additionally, we present a case study that illustrates the application of the framework in practice. We believe that our framework can serve as a valuable tool not only for developing explainable AI systems in healthcare but also for any AI system that has a significant impact on individuals. 

**Abstract (ZH)**: 人工智能在计算机系统发展中的集成带来了新的挑战：使智能系统对人类可解释。特别是在健康和福祉领域，决策支持系统的透明度使医疗专业人员能够理解并信任自动决策和预测。为应对这一需求，需要工具来指导可解释AI系统的开发。在本文中，我们引入了一种评估框架，旨在支持可解释AI系统在健康和福祉领域的开发。此外，我们呈现了一个案例研究，展示了该框架在实践中的应用。我们认为，我们的框架不仅可以作为在医疗保健中开发可解释AI系统的宝贵工具，还可以作为任何对个人有重大影响的AI系统的开发工具。 

---
# Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks 

**Title (ZH)**: 任务记忆引擎(TME): 提升多步LLM代理任务的状态意识 

**Authors**: Ye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08525)  

**Abstract**: Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. The full implementation of TME is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用作执行多步骤任务的自主代理。然而，现有的大多数框架未能维持对任务状态的结构化理解，往往依赖于线性提示连接或浅层记忆缓冲。这导致了脆弱的表现、频繁的虚构叙述以及较差的长范围连贯性。在本文中，我们提出了一种轻量级且结构化的记忆模块Task Memory Engine（TME），该模块使用层次化的Task Memory Tree（TMT）跟踪任务执行。树中的每个节点对应于一个任务步骤，并存储相关信息、输出、状态以及子任务关系。我们引入了一种提示合成方法，能够根据当前活动节点路径动态生成LLM提示，显著提高了执行一致性并增强了上下文关联性。通过在多步骤代理任务上的案例研究和比较实验，我们证明了TME能够以最小的实现开销实现更好的任务完成准确性和更可解释的行为。TME的完整实现可在以下链接获取：this https URL。 

---
# Belief States for Cooperative Multi-Agent Reinforcement Learning under Partial Observability 

**Title (ZH)**: 部分可观状态下协作多智能体强化学习的信念状态研究 

**Authors**: Paul J. Pritz, Kin K. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2504.08417)  

**Abstract**: Reinforcement learning in partially observable environments is typically challenging, as it requires agents to learn an estimate of the underlying system state. These challenges are exacerbated in multi-agent settings, where agents learn simultaneously and influence the underlying state as well as each others' observations. We propose the use of learned beliefs on the underlying state of the system to overcome these challenges and enable reinforcement learning with fully decentralized training and execution. Our approach leverages state information to pre-train a probabilistic belief model in a self-supervised fashion. The resulting belief states, which capture both inferred state information as well as uncertainty over this information, are then used in a state-based reinforcement learning algorithm to create an end-to-end model for cooperative multi-agent reinforcement learning under partial observability. By separating the belief and reinforcement learning tasks, we are able to significantly simplify the policy and value function learning tasks and improve both the convergence speed and the final performance. We evaluate our proposed method on diverse partially observable multi-agent tasks designed to exhibit different variants of partial observability. 

**Abstract (ZH)**: 在部分可观测环境中的强化学习通常具有挑战性，因为这要求智能体学习系统状态的估计。在多智能体设置中，这些挑战进一步加剧，因为智能体同时学习并影响系统状态以及彼此的观察。我们提出使用对系统状态的学得信念来克服这些挑战，并实现完全去中心化的训练和执行的强化学习。我们的方法利用状态信息以自监督方式预训练一个概率信念模型。由此产生的信念状态不仅捕捉了推断出的状态信息，还捕捉了对该信息的不确定性，然后在基于状态的强化学习算法中使用这些信念状态，以端到端的方式构建在部分可观测性下的合作多智能体强化学习模型。通过将信念学习和强化学习任务分离，我们能够显著简化策略和价值函数的学习任务，并提高收敛速度和最终性能。我们通过设计旨在展现不同部分可观测性变体的多样化的部分可观测多智能体任务来评估我们所提出的方法。 

---
# MedRep: Medical Concept Representation for General Electronic Health Record Foundation Models 

**Title (ZH)**: 医_rep: 通用电子健康记录基础模型的医学概念表示 

**Authors**: Junmo Kim, Namkyeong Lee, Jiwon Kim, Kwangsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08329)  

**Abstract**: Electronic health record (EHR) foundation models have been an area ripe for exploration with their improved performance in various medical tasks. Despite the rapid advances, there exists a fundamental limitation: Processing unseen medical codes out of the vocabulary. This problem limits the generality of EHR foundation models and the integration of models trained with different vocabularies. To deal with this problem, we propose MedRep for EHR foundation models based on the observational medical outcome partnership (OMOP) common data model (CDM), providing the integrated medical concept representations and the basic data augmentation strategy for patient trajectories. For concept representation learning, we enrich the information of each concept with a minimal definition through large language model (LLM) prompts and enhance the text-based representations through graph ontology of OMOP vocabulary. Trajectory augmentation randomly replaces selected concepts with other similar concepts that have closely related representations to let the model practice with the concepts out-of-vocabulary. Finally, we demonstrate that EHR foundation models trained with MedRep better maintain the prediction performance in external datasets. Our code implementation is publicly available at this https URL. 

**Abstract (ZH)**: 基于OMOP通用数据模型的MedRep：用于电子健康记录基础模型的医疗概念表示和轨迹增强方法 

---
# Orchestrating Agents and Data for Enterprise: A Blueprint Architecture for Compound AI 

**Title (ZH)**: 企业中代理与数据的 orchestrating：复合人工智能的蓝图架构 

**Authors**: Eser Kandogan, Nikita Bhutani, Dan Zhang, Rafael Li Chen, Sairam Gurajada, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2504.08148)  

**Abstract**: Large language models (LLMs) have gained significant interest in industry due to their impressive capabilities across a wide range of tasks. However, the widespread adoption of LLMs presents several challenges, such as integration into existing applications and infrastructure, utilization of company proprietary data, models, and APIs, and meeting cost, quality, responsiveness, and other requirements. To address these challenges, there is a notable shift from monolithic models to compound AI systems, with the premise of more powerful, versatile, and reliable applications. However, progress thus far has been piecemeal, with proposals for agentic workflows, programming models, and extended LLM capabilities, without a clear vision of an overall architecture. In this paper, we propose a 'blueprint architecture' for compound AI systems for orchestrating agents and data for enterprise applications. In our proposed architecture the key orchestration concept is 'streams' to coordinate the flow of data and instructions among agents. Existing proprietary models and APIs in the enterprise are mapped to 'agents', defined in an 'agent registry' that serves agent metadata and learned representations for search and planning. Agents can utilize proprietary data through a 'data registry' that similarly registers enterprise data of various modalities. Tying it all together, data and task 'planners' break down, map, and optimize tasks and queries for given quality of service (QoS) requirements such as cost, accuracy, and latency. We illustrate an implementation of the architecture for a use-case in the HR domain and discuss opportunities and challenges for 'agentic AI' in the enterprise. 

**Abstract (ZH)**: 大规模语言模型（LLMs）因其实现广泛任务的出色能力而在行业中引起了广泛关注。然而，LLMs的广泛应用也带来了一些挑战，例如与现有应用程序和基础设施的集成、利用公司专有数据、模型和API，以及满足成本、质量、响应性及其他要求。为应对这些挑战，人们已经从单一模型转向复杂的AI系统，旨在构建更强大、多功能且更可靠的应用程序。然而，进展仍较为零散，仍缺乏清晰的整体架构愿景，包括代理工作流程、编程模型和扩展的LLM能力的提案。本文提出了一种“蓝本架构”，用于协调企业应用中的代理和数据。在我们的提议架构中，“流水线”是关键的协调概念，用于协调代理之间数据和指令的流动。“代理注册表”定义了代理的注册信息和学习表示，用于搜索和规划，现有企业中的专有模型和API被映射到“代理”。通过“数据注册表”注册各种模态的企业数据，代理可以利用这些数据。“数据和任务规划器”将任务和查询分解、映射和优化，以满足给定的服务质量（QoS）要求，如成本、准确性和延迟。我们展示了该架构在人力资源领域的一个应用场景，并讨论了企业环境中“自主AI”的机遇与挑战。 

---
# The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search 

**Title (ZH)**: AI科学家-v2：基于代理树搜索的工作坊级别自动化科学发现 

**Authors**: Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08066)  

**Abstract**: AI is increasingly playing a pivotal role in transforming how scientific discoveries are made. We introduce The AI Scientist-v2, an end-to-end agentic system capable of producing the first entirely AI generated peer-review-accepted workshop paper. This system iteratively formulates scientific hypotheses, designs and executes experiments, analyzes and visualizes data, and autonomously authors scientific manuscripts. Compared to its predecessor (v1, Lu et al., 2024 arXiv:2408.06292), The AI Scientist-v2 eliminates the reliance on human-authored code templates, generalizes effectively across diverse machine learning domains, and leverages a novel progressive agentic tree-search methodology managed by a dedicated experiment manager agent. Additionally, we enhance the AI reviewer component by integrating a Vision-Language Model (VLM) feedback loop for iterative refinement of content and aesthetics of the figures. We evaluated The AI Scientist-v2 by submitting three fully autonomous manuscripts to a peer-reviewed ICLR workshop. Notably, one manuscript achieved high enough scores to exceed the average human acceptance threshold, marking the first instance of a fully AI-generated paper successfully navigating a peer review. This accomplishment highlights the growing capability of AI in conducting all aspects of scientific research. We anticipate that further advancements in autonomous scientific discovery technologies will profoundly impact human knowledge generation, enabling unprecedented scalability in research productivity and significantly accelerating scientific breakthroughs, greatly benefiting society at large. We have open-sourced the code at this https URL to foster the future development of this transformative technology. We also discuss the role of AI in science, including AI safety. 

**Abstract (ZH)**: AI日益在改变科学发现的方式中发挥着关键作用。我们介绍了《AI科学家-v2》，一个端到端的自主系统，能够生成首篇完全由AI生成并通过同行评审的工作会议论文。该系统迭代地提出科学假设，设计和执行实验，分析和可视化数据，并自主撰写科学论文。与前身（v1，Lu et al., 2024 arXiv:2408.06292）相比，《AI科学家-v2》消除了对人类撰写的代码模板的依赖，有效地泛化到多种机器学习领域，并利用一种新的渐进性自主树搜索方法，该方法由一个专用的实验管理代理管理。此外，我们通过整合视觉-语言模型（VLM）反馈环路，增强了AI审稿人组件，以迭代改进图表的内容和美观性。我们通过向一个同行评审的ICLR研讨会提交三篇完全自主的论文来评估《AI科学家-v2》。值得注意的是，一篇论文得分足够高，超过了平均人类接受阈值，标志着完全由AI生成的文章首次成功通过同行评审。这一成就突显了AI在开展科学研究各个方面日益增强的能力。我们预计，自主科学研究技术的进一步发展将对人类知识生成产生深远影响，显著提高研究生产力，并加速科学突破，极大惠及社会。我们在此开源代码，以促进这一变革性技术的未来开发。我们还讨论了AI在科学中的作用，包括AI安全性。 

---
# Utility Inspired Generalizations of TOPSIS 

**Title (ZH)**: 基于效益启发的一致化TOPSIS方法 

**Authors**: Robert Susmaga, Izabela Szczech  

**Link**: [PDF](https://arxiv.org/pdf/2504.08014)  

**Abstract**: TOPSIS, a popular method for ranking alternatives is based on aggregated distances to ideal and anti-ideal points. As such, it was considered to be essentially different from widely popular and acknowledged `utility-based methods', which build rankings from weight-averaged utility values. Nonetheless, TOPSIS has recently been shown to be a natural generalization of these `utility-based methods' on the grounds that the distances it uses can be decomposed into so called weight-scaled means (WM) and weight-scaled standard deviations (WSD) of utilities. However, the influence that these two components exert on the final ranking cannot be in any way influenced in the standard TOPSIS. This is why, building on our previous results, in this paper we put forward modifications that make TOPSIS aggregations responsive to WM and WSD, achieving some amount of well interpretable control over how the rankings are influenced by WM and WSD. The modifications constitute a natural generalization of the standard TOPSIS method because, thanks to them, the generalized TOPSIS may turn into the original TOPSIS or, otherwise, following the decision maker's preferences, may trade off WM for WSD or WSD for WM. In the latter case, TOPSIS gradually reduces to a regular `utility-based method'. All in all, we believe that the proposed generalizations constitute an interesting practical tool for influencing the ranking by controlled application of a new form of decision maker's preferences. 

**Abstract (ZH)**: 基于理想与反理想点的TOPSIS方法是一种流行的备选方案排序方法，它可以基于与理想和反理想点的距离聚合。尽管它通常被认为与基于效用的广泛认可的方法本质上不同，后者从加权平均效用值构建排名，最近的研究表明，TOPSIS可以被视为这些基于效用的方法的自然推广，因为其使用距离可以分解为效用的加权尺度均值(WM)和加权尺度标准差(WSD)。然而，在标准TOPSIS中，WM和WSD对最终排名的影响是无法控制的。因此，在前序结果的基础上，本文提出了修改方法，使TOPSIS聚合能够响应WM和WSD，从而在一定程度上对排名受WM和WSD影响的程度实现可解释的控制。这些修改使标准TOPSIS方法成为一种自然推广，因为根据它们，广义TOPSIS可以转化为原始TOPSIS，或者根据决策者的选择，权衡WM和WSD，而在后者的情况下，TOPSIS逐渐简化为一种常规的基于效用的方法。总体而言，我们认为所提出的一般化方法是一种有趣的实际工具，通过受控应用新形式的决策者偏好来影响排名。 

---
# A Python toolkit for dealing with Petri nets over ontological graphs 

**Title (ZH)**: 一个处理本体图上的 Petri 网的 Python 工具包 

**Authors**: Krzysztof Pancerz  

**Link**: [PDF](https://arxiv.org/pdf/2504.08006)  

**Abstract**: We present theoretical rudiments of Petri nets over ontological graphs as well as the designed and implemented Python toolkit for dealing with such nets. In Petri nets over ontological graphs, the domain knowledge is enclosed in a form of ontologies. In this way, some valuable knowledge (especially in terms of semantic relations) can be added to model reasoning and control processes by means of Petri nets. In the implemented approach, ontological graphs are obtained from ontologies built in accordance with the OWL 2 Web Ontology Language. The implemented tool enables the users to define the structure and dynamics of Petri nets over ontological graphs. 

**Abstract (ZH)**: 基于本体图的Petri网的理论基础及设计实现的Python工具 

---
# Neuron-level Balance between Stability and Plasticity in Deep Reinforcement Learning 

**Title (ZH)**: 深层强化学习中神经元级稳定性与可塑性的平衡 

**Authors**: Jiahua Lan, Sen Zhang, Haixia Pan, Ruijun Liu, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08000)  

**Abstract**: In contrast to the human ability to continuously acquire knowledge, agents struggle with the stability-plasticity dilemma in deep reinforcement learning (DRL), which refers to the trade-off between retaining existing skills (stability) and learning new knowledge (plasticity). Current methods focus on balancing these two aspects at the network level, lacking sufficient differentiation and fine-grained control of individual neurons. To overcome this limitation, we propose Neuron-level Balance between Stability and Plasticity (NBSP) method, by taking inspiration from the observation that specific neurons are strongly relevant to task-relevant skills. Specifically, NBSP first (1) defines and identifies RL skill neurons that are crucial for knowledge retention through a goal-oriented method, and then (2) introduces a framework by employing gradient masking and experience replay techniques targeting these neurons to preserve the encoded existing skills while enabling adaptation to new tasks. Numerous experimental results on the Meta-World and Atari benchmarks demonstrate that NBSP significantly outperforms existing approaches in balancing stability and plasticity. 

**Abstract (ZH)**: 基于神经元层面稳定性和可塑性平衡的方法（NBSP）：超越深 reinforcement 学习中的稳定性-可塑性困境 

---
# Towards an Understanding of Context Utilization in Code Intelligence 

**Title (ZH)**: 面向代码智能中的上下文利用理解 

**Authors**: Yanlin Wang, Kefeng Duan, Dewu Zheng, Ensheng Shi, Fengji Zhang, Yanli Wang, Jiachi Chen, Xilin Liu, Yuchi Ma, Hongyu Zhang, Qianxiang Wang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08734)  

**Abstract**: Code intelligence is an emerging domain in software engineering, aiming to improve the effectiveness and efficiency of various code-related tasks. Recent research suggests that incorporating contextual information beyond the basic original task inputs (i.e., source code) can substantially enhance model performance. Such contextual signals may be obtained directly or indirectly from sources such as API documentation or intermediate representations like abstract syntax trees can significantly improve the effectiveness of code intelligence. Despite growing academic interest, there is a lack of systematic analysis of context in code intelligence. To address this gap, we conduct an extensive literature review of 146 relevant studies published between September 2007 and August 2024. Our investigation yields four main contributions. (1) A quantitative analysis of the research landscape, including publication trends, venues, and the explored domains; (2) A novel taxonomy of context types used in code intelligence; (3) A task-oriented analysis investigating context integration strategies across diverse code intelligence tasks; (4) A critical evaluation of evaluation methodologies for context-aware methods. Based on these findings, we identify fundamental challenges in context utilization in current code intelligence systems and propose a research roadmap that outlines key opportunities for future research. 

**Abstract (ZH)**: 代码智能是软件工程中的一个新兴领域，旨在提高各种代码相关任务的有效性和效率。近期研究指出，除了基本的原始任务输入（即源代码）之外，融入上下文信息可以显著增强模型性能。这些上下文信号可以直接或间接从API文档、抽象语法树等源中获取，可以显著提高代码智能的有效性。尽管学术界对此产生了浓厚兴趣，但对于代码智能中的上下文分析仍然缺乏系统性研究。为填补这一空白，我们对2007年9月至2024年8月间发表的146篇相关研究进行了广泛的文献综述。我们的研究提供了四个主要贡献：（1）对研究格局的定量分析，包括出版趋势、发表平台和探索领域；（2）提出了一种代码智能中使用的上下文类型的新分类；（3）从不同代码智能任务出发，调查上下文集成策略；（4）对面向上下文的方法的评价方法进行批判性评估。基于这些发现，我们识别了当前代码智能系统中上下文利用的基本挑战，并提出了一条未来研究的关键机会研究路线图。 

---
# Steering CLIP's vision transformer with sparse autoencoders 

**Title (ZH)**: 用稀疏自编码器引导CLIP的视觉变换器 

**Authors**: Sonia Joseph, Praneet Suresh, Ethan Goldfarb, Lorenz Hufe, Yossi Gandelsman, Robert Graham, Danilo Bzdok, Wojciech Samek, Blake Aaron Richards  

**Link**: [PDF](https://arxiv.org/pdf/2504.08729)  

**Abstract**: While vision models are highly capable, their internal mechanisms remain poorly understood -- a challenge which sparse autoencoders (SAEs) have helped address in language, but which remains underexplored in vision. We address this gap by training SAEs on CLIP's vision transformer and uncover key differences between vision and language processing, including distinct sparsity patterns for SAEs trained across layers and token types. We then provide the first systematic analysis on the steerability of CLIP's vision transformer by introducing metrics to quantify how precisely SAE features can be steered to affect the model's output. We find that 10-15\% of neurons and features are steerable, with SAEs providing thousands more steerable features than the base model. Through targeted suppression of SAE features, we then demonstrate improved performance on three vision disentanglement tasks (CelebA, Waterbirds, and typographic attacks), finding optimal disentanglement in middle model layers, and achieving state-of-the-art performance on defense against typographic attacks. 

**Abstract (ZH)**: 视觉模型虽然功能强大，但其内部机制依然 poorly understood -- 一个稀疏自编码器（SAEs）已在语言领域帮助解决的问题，在视觉领域仍未得到充分探索。我们通过在 CLIP 的视觉变换器上训练 SAEs 来填补这一空白，揭示了视觉与语言处理之间的关键差异，包括在不同层和token类型下 SAEs 的独特稀疏模式。我们还提供了 CLIP 视觉变换器可控性的首个系统分析，通过引入度量标准来量化 SAE 特征如何精确影响模型输出。我们发现有 10-15% 的神经元和特征是可控的，SAEs 提供了比基础模型多数千倍的可控特征。通过有针对性地抑制 SAE 特征，我们展示了在三个视觉解耦任务（CelebA、Waterbirds 和 字符攻击）上的性能提升，在中间模型层实现最佳解耦，并在针对字符攻击的防御上达到当前最佳性能。 

---
# Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images 

**Title (ZH)**: 视觉编年史：使用多模态大规模语言模型分析海量图像集 

**Authors**: Boyang Deng, Songyou Peng, Kyle Genova, Gordon Wetzstein, Noah Snavely, Leonidas Guibas, Thomas Funkhouser  

**Link**: [PDF](https://arxiv.org/pdf/2504.08727)  

**Abstract**: We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.). See more results and interactive demos at this https URL. 

**Abstract (ZH)**: 我们提出了一种使用多模态大模型（MLLMs）分析包含数千万张在不同时间拍摄的图像的大数据库的系统，旨在发现时间变化中的模式。具体而言，我们旨在捕捉一定时期内某个城市中频繁共现的变化趋势（“趋势”）。与之前的视觉分析不同，我们的分析可以回答开放性查询（例如：“城市中频繁的变化类型是什么？”）而无需预先确定的目标主题或训练标签。这些特性使得先前的学习基于或无监督的视觉分析工具不再适用。我们标识出MLLMs作为一种新型工具，因其具有开放性的语义理解能力。然而，我们的数据集规模比MLLMs能够处理的上下文规模大四个数量级。因此，我们引入了一个自底向上的过程，将大规模的视觉分析问题分解为更易于管理的子问题。我们精心设计了基于MLLMs的解决方案来解决每个子问题。在系统实验和消融研究中，我们发现它显著优于基线，并能够从大城市拍摄的图像中发现有趣的趋势（例如：“户外用餐区增加”、“立交桥被漆成蓝色”等）。请访问此链接以查看更多结果和交互演示：见更多结果和交互演示请访问: https://xxxxx 

---
# DocAgent: A Multi-Agent System for Automated Code Documentation Generation 

**Title (ZH)**: DocAgent: 一种自动化代码文档生成的多agent系统 

**Authors**: Dayu Yang, Antoine Simoulin, Xin Qian, Xiaoyi Liu, Yuwei Cao, Zhaopu Teng, Grey Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08725)  

**Abstract**: High-quality code documentation is crucial for software development especially in the era of AI. However, generating it automatically using Large Language Models (LLMs) remains challenging, as existing approaches often produce incomplete, unhelpful, or factually incorrect outputs. We introduce DocAgent, a novel multi-agent collaborative system using topological code processing for incremental context building. Specialized agents (Reader, Searcher, Writer, Verifier, Orchestrator) then collaboratively generate documentation. We also propose a multi-faceted evaluation framework assessing Completeness, Helpfulness, and Truthfulness. Comprehensive experiments show DocAgent significantly outperforms baselines consistently. Our ablation study confirms the vital role of the topological processing order. DocAgent offers a robust approach for reliable code documentation generation in complex and proprietary repositories. 

**Abstract (ZH)**: 高质量的代码文档对于人工智能时代的软件开发至关重要。然而，使用大规模语言模型自动生成代码文档仍具有挑战性，因为现有方法常常产生不完整、无帮助或事实错误的输出。我们提出了DocAgent，这是一种使用拓扑代码处理进行增量上下文构建的新型多代理协作系统。专门的代理（_reader, _searcher, _writer, _verifier, _orchestrator）然后协作生成文档。我们还提出了一种多方面的评估框架，评估其完整性、有用性和真实性。全面的实验表明，DocAgent在基准上表现出显著的优势。我们的消融研究证明了拓扑处理顺序在其中的关键作用。DocAgent为复杂和专有代码库中的可靠代码文档生成提供了一种稳健的方法。 

---
# ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG Classification with Contrastive Learning 

**Title (ZH)**: ProtoECGNet：基于案例的可解释深度学习在对比学习下的多标签心电图分类 

**Authors**: Sahil Sethi, David Chen, Thomas Statchen, Michael C. Burkhart, Nipun Bhandari, Bashar Ramadan, Brett Beaulieu-Jones  

**Link**: [PDF](https://arxiv.org/pdf/2504.08713)  

**Abstract**: Deep learning-based electrocardiogram (ECG) classification has shown impressive performance but clinical adoption has been slowed by the lack of transparent and faithful explanations. Post hoc methods such as saliency maps may fail to reflect a model's true decision process. Prototype-based reasoning offers a more transparent alternative by grounding decisions in similarity to learned representations of real ECG segments, enabling faithful, case-based explanations. We introduce ProtoECGNet, a prototype-based deep learning model for interpretable, multi-label ECG classification. ProtoECGNet employs a structured, multi-branch architecture that reflects clinical interpretation workflows: it integrates a 1D CNN with global prototypes for rhythm classification, a 2D CNN with time-localized prototypes for morphology-based reasoning, and a 2D CNN with global prototypes for diffuse abnormalities. Each branch is trained with a prototype loss designed for multi-label learning, combining clustering, separation, diversity, and a novel contrastive loss that encourages appropriate separation between prototypes of unrelated classes while allowing clustering for frequently co-occurring diagnoses. We evaluate ProtoECGNet on all 71 diagnostic labels from the PTB-XL dataset, demonstrating competitive performance relative to state-of-the-art black-box models while providing structured, case-based explanations. To assess prototype quality, we conduct a structured clinician review of the final model's projected prototypes, finding that they are rated as representative and clear. ProtoECGNet shows that prototype learning can be effectively scaled to complex, multi-label time-series classification, offering a practical path toward transparent and trustworthy deep learning models for clinical decision support. 

**Abstract (ZH)**: 基于深度学习的心电图（ECG）分类展示了令人印象深刻的性能，但Clinical应用受到缺乏透明和忠实解释的限制。基于原型的推理提供了更有透明度的替代方案，通过将决策与学习的实ECG段落表示的相似性联系起来，促进忠实的案例基础解释。我们引入ProtoECGNet，这是一种用于可解释的多标签ECG分类的基于原型的深度学习模型。ProtoECGNet采用反映临床解释工作流程的结构化多分支架构：它结合了用于节律分类的一维卷积神经网络（1D CNN）和全局原型、用于基于形态的推理的时间局地化原型的二维卷积神经网络（2D CNN）以及用于弥散异常的二维卷积神经网络（2D CNN）和全局原型。每个分支均使用为多标签学习设计的原型损失进行训练，结合聚类、分离、多样性以及一种新颖的对比损失，该损失鼓励不相关类原型之间的适当分离，并允许根据频繁共现的诊断进行聚类。我们在PTB-XL数据集的所有71个诊断标签上评估了ProtoECGNet，表明相对于最先进的黑盒模型具有竞争力的表现，同时提供结构化的案例基础解释。为了评估原型质量，我们对最终模型的投影原型进行结构化的临床审查，发现它们被评为代表性和清晰。ProtoECGNet表明，原型学习可以有效地扩展到复杂的多标签时间序列分类，为临床决策支持提供了一条实用的道路，朝着透明和可信赖的深度学习模型的方向发展。 

---
# Fast-Slow-Thinking: Complex Task Solving with Large Language Models 

**Title (ZH)**: 快慢思考：大规模语言模型解决复杂任务 

**Authors**: Yiliu Sun, Yanfang Zhang, Zicheng Zhao, Sheng Wan, Dacheng Tao, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08690)  

**Abstract**: Nowadays, Large Language Models (LLMs) have been gradually employed to solve complex tasks. To face the challenge, task decomposition has become an effective way, which proposes to divide a complex task into multiple simpler subtasks and then solve them separately so that the difficulty of the original task can be reduced. However, the performance of existing task decomposition methods can be suboptimal when the task contains overly complex logic and constraints. In this situation, the solution generated by LLMs may deviate from the original purpose of the task, or contain redundant or even erroneous content. Therefore, inspired by the fact that humans possess two thinking systems including fast thinking and slow thinking, this paper introduces a new task decomposition method termed ``Fast-Slow-Thinking'' (FST), which stimulates LLMs to solve tasks through the cooperation of Fast Thinking (FT) and Slow Thinking (ST) steps. Here FT focuses more on the general and concise aspect of the task, and ST focuses more on the details of the task. In FT, LLMs are prompted to remove the constraints of the original task, therefore simplifying it to a general and concise one. In ST, we recall the constraints removed in FT, so that LLMs can improve the answer generated in FT to meet the requirements of the original task. Therefore, our FST method enables LLMs to consider a complex problem via a human-like cognition process from coarse to fine, the effectiveness of which has been well demonstrated by the experiments on three types of tasks. 

**Abstract (ZH)**: 如今，大型语言模型（LLMs）已被逐渐应用于解决复杂任务。为了应对这一挑战，任务分解已成为一种有效的方法，即将一个复杂的任务分解为多个简单的子任务，然后分别解决，从而降低原任务的难度。然而，当任务包含过于复杂的逻辑和约束时，现有的任务分解方法的表现可能不尽如人意。在这种情况下，LLMs生成的解决方案可能偏离任务的原始目标，或包含冗余甚至错误的内容。因此，受人类拥有快速思考和慢速思考两种思维方式的启发，本文提出了一种新的任务分解方法，称为“快速-慢速思考”（Fast-Slow-Thinking, FST），该方法通过快速思考（FT）和慢速思考（ST）步骤的协作促使LLMs解决任务。在FT中，LLMs被提示去除原任务的约束，从而使任务简化为一个一般和简洁的问题。在ST中，我们重新考虑在FT中去除的约束，从而使LLMs能够在满足原任务要求的基础上改进FT生成的答案。因此，我们的FST方法使LLMs能够通过从粗到细的人类认知过程来考虑复杂问题，其有效性已通过三种类型任务的实验得到验证。 

---
# Voice Interaction With Conversational AI Could Facilitate Thoughtful Reflection and Substantive Revision in Writing 

**Title (ZH)**: 语音交互与对话式AI能够促进写作中的深入反思和实质性修订 

**Authors**: Jiho Kim, Philippe Laban, Xiang 'Anthony' Chen, Kenneth C. Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2504.08687)  

**Abstract**: Writing well requires not only expressing ideas but also refining them through revision, a process facilitated by reflection. Prior research suggests that feedback delivered through dialogues, such as those in writing center tutoring sessions, can help writers reflect more thoughtfully on their work compared to static feedback. Recent advancements in multi-modal large language models (LLMs) now offer new possibilities for supporting interactive and expressive voice-based reflection in writing. In particular, we propose that LLM-generated static feedback can be repurposed as conversation starters, allowing writers to seek clarification, request examples, and ask follow-up questions, thereby fostering deeper reflection on their writing. We argue that voice-based interaction can naturally facilitate this conversational exchange, encouraging writers' engagement with higher-order concerns, facilitating iterative refinement of their reflections, and reduce cognitive load compared to text-based interactions. To investigate these effects, we propose a formative study exploring how text vs. voice input influence writers' reflection and subsequent revisions. Findings from this study will inform the design of intelligent and interactive writing tools, offering insights into how voice-based interactions with LLM-powered conversational agents can support reflection and revision. 

**Abstract (ZH)**: 写作不仅需要表达想法，还需要通过修订加以精炼，这一过程可以通过反思得以促进。先前的研究表明，通过对话提供的反馈（如写作中心辅导 session 中的对话反馈）可以比静态反馈帮助作者更深入地反思自己的作品。近期多模态大语言模型（LLMs）的进步现在为支持基于语音的互动和表达性反思提供了新的可能性。特别是，我们建议 LLM 生成的静态反馈可以重新用于作为对话启动器，使作者能够寻求澄清、请求示例并提出后续问题，从而促进他们写作的更深层次反思。我们认为基于语音的互动能够自然地促进这种对话交换，鼓励作者与高层次关注点的更深层次互动，促进反思的迭代精炼，并且与基于文本的互动相比减轻认知负担。为了研究这些影响，我们提议进行一种形成性研究，探讨文本输入与语音输入如何影响作者的反思和后续修订。该研究的发现将为智能和互动写作工具的设计提供指导，揭示 LLM  powering 的对话代理基于语音的互动如何支持反思和修订的方法。 

---
# Pobogot -- An Open-Hardware Open-Source Low Cost Robot for Swarm Robotics 

**Title (ZH)**: Pobogot -- 一种用于群机器人lopen硬件开源低成本机器人 

**Authors**: Alessia Loi, Loona Macabre, Jérémy Fersula, Keivan Amini, Leo Cazenille, Fabien Caura, Alexandre Guerre, Stéphane Gourichon, Olivier Dauchot, Nicolas Bredeche  

**Link**: [PDF](https://arxiv.org/pdf/2504.08686)  

**Abstract**: This paper describes the Pogobot, an open-source and open-hardware platform specifically designed for research involving swarm robotics. Pogobot features vibration-based locomotion, infrared communication, and an array of sensors in a cost-effective package (approx. 250~euros/unit). The platform's modular design, comprehensive API, and extensible architecture facilitate the implementation of swarm intelligence algorithms and distributed online reinforcement learning algorithms. Pogobots offer an accessible alternative to existing platforms while providing advanced capabilities including directional communication between units. More than 200 Pogobots are already being used on a daily basis at Sorbonne Université and PSL to study self-organizing systems, programmable active matter, discrete reaction-diffusion-advection systems as well as models of social learning and evolution. 

**Abstract (ZH)**: 本文描述了Pogobot，一个专为 swarm robotics 研究设计的开源且开放硬件平台。Pogobot 具有基于振动的移动方式、红外通信以及成本效益高的传感器阵列（每单位约250~欧元）。该平台的模块化设计、全面的应用程序接口（API）和可扩展架构便于实现 swarm 智能算法和分布式在线强化学习算法。Pogobots 提供了一种现成平台的可访问替代方案，同时具备包括单元间方向性通信在内的高级功能。目前，已有超过200个Pogobots在索邦大学和巴黎文理研究大学每天用于研究自组织系统、可编程活性物质、离散反应-扩散-传输系统以及社会学习和进化的模型。 

---
# Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model 

**Title (ZH)**: 海藻-7B：视频生成基础模型的经济高效训练 

**Authors**: Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, Feng Cheng, Feilong Zuo Xuejiao Zeng, Ziyan Yang, Fangyuan Kong, Zhiwu Qing, Fei Xiao, Meng Wei, Tuyen Hoang, Siyu Zhang, Peihao Zhu, Qi Zhao, Jiangqiao Yan, Liangke Gui, Sheng Bi, Jiashi Li, Yuxi Ren, Rui Wang, Huixia Li, Xuefeng Xiao, Shu Liu, Feng Ling, Heng Zhang, Houmin Wei, Huafeng Kuang, Jerry Duncan, Junda Zhang, Junru Zheng, Li Sun, Manlin Zhang, Renfei Sun, Xiaobin Zhuang, Xiaojie Li, Xin Xia, Xuyan Chi, Yanghua Peng, Yuping Wang, Yuxuan Wang, Zhongkai Zhao, Zhuo Chen, Zuquan Song, Zhenheng Yang, Jiashi Feng, Jianchao Yang, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08685)  

**Abstract**: This technical report presents a cost-efficient strategy for training a video generation foundation model. We present a mid-sized research model with approximately 7 billion parameters (7B) called Seaweed-7B trained from scratch using 665,000 H100 GPU hours. Despite being trained with moderate computational resources, Seaweed-7B demonstrates highly competitive performance compared to contemporary video generation models of much larger size. Design choices are especially crucial in a resource-constrained setting. This technical report highlights the key design decisions that enhance the performance of the medium-sized diffusion model. Empirically, we make two observations: (1) Seaweed-7B achieves performance comparable to, or even surpasses, larger models trained on substantially greater GPU resources, and (2) our model, which exhibits strong generalization ability, can be effectively adapted across a wide range of downstream applications either by lightweight fine-tuning or continue training. See the project page at this https URL 

**Abstract (ZH)**: 本技术报告提出了一种成本高效的视频生成基础模型训练策略。我们介绍了使用约70亿参数（7B）的中期研究模型Seaweed-7B，该模型从零开始训练共使用了665,000个H100 GPU小时。即使使用了适度的计算资源，Seaweed-7B在与更大规模的 contemporaneous 视频生成模型相比时也展现了高度竞争力的性能。在资源受限的环境中，设计选择尤为关键。本技术报告强调了提升中期扩散模型性能的关键设计决策。实证研究表明：（1）Seaweed-7B在使用显著更多GPU资源训练的大型模型中表现出相当甚至更优的性能；（2）我们的模型表现出强大的泛化能力，可以通过轻量级微调或继续训练有效地适应广泛下游应用。请参见项目页面：[该链接] 

---
# Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning 

**Title (ZH)**: 天才：一个通用且纯粹的无监督自我训练框架以进行高级推理 

**Authors**: Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08672)  

**Abstract**: Advancing LLM reasoning skills has captivated wide interest. However, current post-training techniques rely heavily on supervisory signals, such as outcome supervision or auxiliary reward models, which face the problem of scalability and high annotation costs. This motivates us to enhance LLM reasoning without the need for external supervision. We introduce a generalizable and purely unsupervised self-training framework, named Genius. Without external auxiliary, Genius requires to seek the optimal response sequence in a stepwise manner and optimize the LLM. To explore the potential steps and exploit the optimal ones, Genius introduces a stepwise foresight re-sampling strategy to sample and estimate the step value by simulating future outcomes. Further, we recognize that the unsupervised setting inevitably induces the intrinsic noise and uncertainty. To provide a robust optimization, we propose an advantage-calibrated optimization (ACO) loss function to mitigate estimation inconsistencies. Combining these techniques together, Genius provides an advanced initial step towards self-improve LLM reasoning with general queries and without supervision, revolutionizing reasoning scaling laws given the vast availability of general queries. The code will be released at this https URL. 

**Abstract (ZH)**: 提升大规模语言模型推理能力引起了广泛兴趣。然而，当前的后训练技术严重依赖于外部监督信号，如结果监督或辅助奖励模型，这面临着可扩展性和高标注成本的问题。这促使我们无需外部监督来增强语言模型的推理能力。我们提出了一种通用且完全无监督的自训练框架，名为Genius。Genius不要求外部辅助，而是逐步寻找最优的响应序列并优化语言模型。为了探索潜在的步骤并利用最优步骤，Genius引入了一种逐步前瞻采样策略，通过模拟未来结果来采样和估计步长值。此外，我们认识到无监督设置不可避免地会产生内在噪声和不确定性。为提供稳健的优化，我们提出了一种优势校准优化（ACO）损失函数来减轻估计不一致性。结合这些技术，Genius为在通用查询下无监督地自我提升语言模型推理能力奠定了先进基础，革命性地改变了推理的扩展规律，鉴于通用查询的大量可用性。代码将在以下链接发布：https://github.com/Genius-Improving-Reasoning。 

---
# Designing Child-Friendly AI Interfaces: Six Developmentally-Appropriate Design Insights from Analysing Disney Animation 

**Title (ZH)**: 设计儿童友好型AI界面：从分析迪士尼动画中获得的六项发展适宜设计启示 

**Authors**: Nomisha Kurian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08670)  

**Abstract**: To build AI interfaces that children can intuitively understand and use, designers need a design grammar that truly serves children's developmental needs. This paper bridges Artificial Intelligence design for children -- an emerging field still defining its best practices -- and children's animation, a well-established field with decades of experience in engaging young viewers through emotionally resonant, cognitively accessible storytelling. Pairing Piagetian developmental theory with design pattern extraction from 52 works of Disney animation, the paper presents six design insights transferable to child-centred AI interface design: (1) emotional expressiveness and visual clarity, (2) musical and auditory scaffolding, (3) audiovisual synchrony for emotional comfort, (4) sidekick-style personas, (5) support for symbolic play and imaginative exploration, and (6) predictable and scaffolded interaction structures. These strategies -- long refined in Disney animation -- function as multimodal scaffolds for attention, understanding, and emotional attunement, thereby forming a structured design grammar familiar to children and transferable to AI interface design. By reframing cinematic storytelling as design logic for AI, the paper offers heuristics for crafting intuitive AI interfaces that align with children's cognitive stages and emotional needs. The work contributes to design theory by showing how sensory, affective and narrative techniques can inform developmentally attuned AI design for children. Future directions include empirical testing, cultural adaptation, and participatory co-design. 

**Abstract (ZH)**: 构建儿童可直观理解并使用的AI界面，设计师需要一种真正服务于儿童发展需求的设计语法规则。本文将面向儿童的AI设计——这一新兴领域仍在界定其最佳实践——与已有数十年吸引年轻观众经验的儿童动画领域相连接，通过结合皮亚杰发展理论和从52部迪士尼动画作品中提取的设计模式，提出六条适用于儿童中心AI界面设计的见解：（1）情感表达和视觉清晰度，（2）音乐和音频支撑，（3）视听同步以提供情感安慰，（4）同伴式人物设定，（5）支持象征性游戏和想象性探索，（6）可预测且逐步引导的交互结构。这些策略作为多模态的支持工具，有助于吸引注意力、理解与情感共鸣，从而形成一种对儿童熟悉并可应用于AI界面设计的结构化设计语法规则。通过将影视叙事重新构想为AI设计逻辑，本文提供了与儿童认知阶段和情感需求相契合的直观AI界面设计的启发式规则。该工作通过展示感觉、情感和叙事技术如何指导与儿童发展相适应的AI设计，丰富了设计理论。未来的研究方向包括实证测试、文化适应和参与式协同设计。 

---
# Variability-Driven User-Story Generation using LLM and Triadic Concept Analysis 

**Title (ZH)**: 基于变异性用户的故事情景生成：结合LLM和三元概念分析 

**Authors**: Alexandre Bazin, Alain Gutierrez, Marianne Huchard, Pierre Martin, Yulin, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08666)  

**Abstract**: A widely used Agile practice for requirements is to produce a set of user stories (also called ``agile product backlog''), which roughly includes a list of pairs (role, feature), where the role handles the feature for a certain purpose. In the context of Software Product Lines, the requirements for a family of similar systems is thus a family of user-story sets, one per system, leading to a 3-dimensional dataset composed of sets of triples (system, role, feature). In this paper, we combine Triadic Concept Analysis (TCA) and Large Language Model (LLM) prompting to suggest the user-story set required to develop a new system relying on the variability logic of an existing system family. This process consists in 1) computing 3-dimensional variability expressed as a set of TCA implications, 2) providing the designer with intelligible design options, 3) capturing the designer's selection of options, 4) proposing a first user-story set corresponding to this selection, 5) consolidating its validity according to the implications identified in step 1, while completing it if necessary, and 6) leveraging LLM to have a more comprehensive website. This process is evaluated with a dataset comprising the user-story sets of 67 similar-purpose websites. 

**Abstract (ZH)**: 一种广泛使用的敏捷实践是生成一组用户故事（也称为“敏捷产品待办事项列表”），这些用户故事大致包括一组角色和功能的配对，其中角色出于某种目的处理该功能。在软件产品线的背景下，对于一组相似系统的功能需求是一个用户故事集的家族，每个系统一个集，构成一个三维数据集，由系统、角色、功能的三元组集合组成。在本文中，我们结合三元概念分析（TCA）和大型语言模型（LLM）提示，根据现有系统家族的变异性逻辑，建议开发新系统所需的用户故事集。该过程包括：1）计算作为一组TCA推论表达的三维变异性；2）为设计者提供可理解的设计选项；3）捕获设计者的选择；4）提出与这些选择相对应的第一个用户故事集；5）根据步骤1中识别的推论验证其有效性，并在必要时进行补充；6）利用LLM获得更全面的网站。本文使用67个相似用途的网站用户故事集数据集对该过程进行了评估。 

---
# Title block detection and information extraction for enhanced building drawings search 

**Title (ZH)**: 基于标题块检测与信息提取的增强建筑设计图检索 

**Authors**: Alessio Lombardi, Li Duan, Ahmed Elnagar, Ahmed Zaalouk, Khalid Ismail, Edlira Vakaj  

**Link**: [PDF](https://arxiv.org/pdf/2504.08645)  

**Abstract**: The architecture, engineering, and construction (AEC) industry still heavily relies on information stored in drawings for building construction, maintenance, compliance and error checks. However, information extraction (IE) from building drawings is often time-consuming and costly, especially when dealing with historical buildings. Drawing search can be simplified by leveraging the information stored in the title block portion of the drawing, which can be seen as drawing metadata. However, title block IE can be complex especially when dealing with historical drawings which do not follow existing standards for uniformity. This work performs a comparison of existing methods for this kind of IE task, and then proposes a novel title block detection and IE pipeline which outperforms existing methods, in particular when dealing with complex, noisy historical drawings. The pipeline is obtained by combining a lightweight Convolutional Neural Network and GPT-4o, the proposed inference pipeline detects building engineering title blocks with high accuracy, and then extract structured drawing metadata from the title blocks, which can be used for drawing search, filtering and grouping. The work demonstrates high accuracy and efficiency in IE for both vector (CAD) and hand-drawn (historical) drawings. A user interface (UI) that leverages the extracted metadata for drawing search is established and deployed on real projects, which demonstrates significant time savings. Additionally, an extensible domain-expert-annotated dataset for title block detection is developed, via an efficient AEC-friendly annotation workflow that lays the foundation for future work. 

**Abstract (ZH)**: AEC行业仍高度依赖图纸存储的信息进行建筑施工、维护、合规性和错误检查。然而，从建筑图纸中提取信息（IE）往往耗时且成本高，特别是在处理历史建筑时。通过利用图纸标题块中存储的信息可以简化图纸搜索，这些信息可被视为图纸的元数据。然而，处理不遵循统一标准的历史图纸时，标题块IE可能会变得复杂。本文对比了现有此类IE任务的方法，并提出了一种新颖的标题块检测和IE流水线，该流水线在处理复杂、嘈杂的历史图纸时表现优于现有方法。此流水线通过结合轻量级卷积神经网络和GPT-4o获得，提出的推理流水线能够以高精度检测建筑工程标题块，并从标题块中提取结构化的图纸元数据，用于图纸搜索、过滤和分组。该工作在矢量（CAD）和手绘（历史）图纸中均展示了高精度和高效性。基于提取的元数据开发了用户界面，已在实际项目中部署，展示了显著的时间节省。此外，通过高效的AEC友好注释工作流开发了一个可扩展的专业领域标注数据集，为未来的相关研究奠定了基础。 

---
# Training-free Guidance in Text-to-Video Generation via Multimodal Planning and Structured Noise Initialization 

**Title (ZH)**: 基于多模态规划和结构化噪声初始化的无需训练指导在文本到视频生成中的应用 

**Authors**: Jialu Li, Shoubin Yu, Han Lin, Jaemin Cho, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.08641)  

**Abstract**: Recent advancements in text-to-video (T2V) diffusion models have significantly enhanced the visual quality of the generated videos. However, even recent T2V models find it challenging to follow text descriptions accurately, especially when the prompt requires accurate control of spatial layouts or object trajectories. A recent line of research uses layout guidance for T2V models that require fine-tuning or iterative manipulation of the attention map during inference time. This significantly increases the memory requirement, making it difficult to adopt a large T2V model as a backbone. To address this, we introduce Video-MSG, a training-free Guidance method for T2V generation based on Multimodal planning and Structured noise initialization. Video-MSG consists of three steps, where in the first two steps, Video-MSG creates Video Sketch, a fine-grained spatio-temporal plan for the final video, specifying background, foreground, and object trajectories, in the form of draft video frames. In the last step, Video-MSG guides a downstream T2V diffusion model with Video Sketch through noise inversion and denoising. Notably, Video-MSG does not need fine-tuning or attention manipulation with additional memory during inference time, making it easier to adopt large T2V models. Video-MSG demonstrates its effectiveness in enhancing text alignment with multiple T2V backbones (VideoCrafter2 and CogVideoX-5B) on popular T2V generation benchmarks (T2VCompBench and VBench). We provide comprehensive ablation studies about noise inversion ratio, different background generators, background object detection, and foreground object segmentation. 

**Abstract (ZH)**: Recent advancements in text-to-video (T2V) diffusion models have significantly enhanced the visual quality of the generated videos. However, even recent T2V models find it challenging to follow text descriptions accurately, especially when the prompt requires accurate control of spatial layouts or object trajectories. A recent line of research uses layout guidance for T2V models that require fine-tuning or iterative manipulation of the attention map during inference time. This significantly increases the memory requirement, making it difficult to adopt a large T2V model as a backbone. To address this, we introduce Video-MSG, a training-free Guidance method for T2V generation based on Multimodal planning and Structured noise initialization.

Video-MSG consists of three steps, where in the first two steps, Video-MSG creates Video Sketch, a fine-grained spatio-temporal plan for the final video, specifying background, foreground, and object trajectories, in the form of draft video frames. In the last step, Video-MSG guides a downstream T2V diffusion model with Video Sketch through noise inversion and denoising. Notably, Video-MSG does not need fine-tuning or attention manipulation with additional memory during inference time, making it easier to adopt large T2V models. Video-MSG demonstrates its effectiveness in enhancing text alignment with multiple T2V backbones (VideoCrafter2 and CogVideoX-5B) on popular T2V generation benchmarks (T2VCompBench and VBench). We provide comprehensive ablation studies about noise inversion ratio, different background generators, background object detection, and foreground object segmentation.

标题：Video-MSG：一种基于多模态规划和结构化噪声初始化的无需训练的文本到视频生成指导方法 

---
# Deep Learning Methods for Detecting Thermal Runaway Events in Battery Production Lines 

**Title (ZH)**: 电池生产线上热失控事件的深度学习检测方法 

**Authors**: Athanasios Athanasopoulos, Matúš Mihalák, Marcin Pietrasik  

**Link**: [PDF](https://arxiv.org/pdf/2504.08632)  

**Abstract**: One of the key safety considerations of battery manufacturing is thermal runaway, the uncontrolled increase in temperature which can lead to fires, explosions, and emissions of toxic gasses. As such, development of automated systems capable of detecting such events is of considerable importance in both academic and industrial contexts. In this work, we investigate the use of deep learning for detecting thermal runaway in the battery production line of VDL Nedcar, a Dutch automobile manufacturer. Specifically, we collect data from the production line to represent both baseline (non thermal runaway) and thermal runaway conditions. Thermal runaway was simulated through the use of external heat and smoke sources. The data consisted of both optical and thermal images which were then preprocessed and fused before serving as input to our models. In this regard, we evaluated three deep-learning models widely used in computer vision including shallow convolutional neural networks, residual neural networks, and vision transformers on two performance metrics. Furthermore, we evaluated these models using explainability methods to gain insight into their ability to capture the relevant feature information from their inputs. The obtained results indicate that the use of deep learning is a viable approach to thermal runaway detection in battery production lines. 

**Abstract (ZH)**: 电池制造中热失控检测的深度学习方法研究 

---
# Task-conditioned Ensemble of Expert Models for Continuous Learning 

**Title (ZH)**: 任务条件下的专家模型ensemble连续学习 

**Authors**: Renu Sharma, Debasmita Pal, Arun Ross  

**Link**: [PDF](https://arxiv.org/pdf/2504.08626)  

**Abstract**: One of the major challenges in machine learning is maintaining the accuracy of the deployed model (e.g., a classifier) in a non-stationary environment. The non-stationary environment results in distribution shifts and, consequently, a degradation in accuracy. Continuous learning of the deployed model with new data could be one remedy. However, the question arises as to how we should update the model with new training data so that it retains its accuracy on the old data while adapting to the new data. In this work, we propose a task-conditioned ensemble of models to maintain the performance of the existing model. The method involves an ensemble of expert models based on task membership information. The in-domain models-based on the local outlier concept (different from the expert models) provide task membership information dynamically at run-time to each probe sample. To evaluate the proposed method, we experiment with three setups: the first represents distribution shift between tasks (LivDet-Iris-2017), the second represents distribution shift both between and within tasks (LivDet-Iris-2020), and the third represents disjoint distribution between tasks (Split MNIST). The experiments highlight the benefits of the proposed method. The source code is available at this https URL. 

**Abstract (ZH)**: 机器学习中维护部署模型准确性的一个主要挑战是在非平稳环境中保持模型的准确性（例如，分类器）。非平稳环境导致分布变化，并随之降低准确性。通过新数据对部署模型进行连续学习可能是解决这一问题的一种方法。然而，一个关键问题是，我们应该如何利用新的训练数据更新模型，使其在保留旧数据准确性的同时适应新数据。在本工作中，我们提出了一种基于任务条件的模型集成方法，以维护现有模型的性能。该方法基于任务归属信息，包含专家模型和基于领域内局部异常的概念的模型。基于任务归属信息（不同于专家模型），领域内模型在运行时为每个探查样本动态提供任务归属信息。为了评估所提出的方法，我们在三种设置下进行了实验：第一种表示任务之间分布变化（LivDet-Iris-2017），第二种表示任务之间和任务内部分布变化（LivDet-Iris-2020），第三种表示任务之间不相交分布（Split MNIST）。实验突显了所提出方法的优势。源代码可在以下网址获取。 

---
# Enterprise-Grade Security for the Model Context Protocol (MCP): Frameworks and Mitigation Strategies 

**Title (ZH)**: 企业级安全模型上下文协议（MCP）框架与缓解策略 

**Authors**: Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2504.08623)  

**Abstract**: The Model Context Protocol (MCP), introduced by Anthropic, provides a standardized framework for artificial intelligence (AI) systems to interact with external data sources and tools in real-time. While MCP offers significant advantages for AI integration and capability extension, it introduces novel security challenges that demand rigorous analysis and mitigation. This paper builds upon foundational research into MCP architecture and preliminary security assessments to deliver enterprise-grade mitigation frameworks and detailed technical implementation strategies. Through systematic threat modeling and analysis of MCP implementations and analysis of potential attack vectors, including sophisticated threats like tool poisoning, we present actionable security patterns tailored for MCP implementers and adopters. The primary contribution of this research lies in translating theoretical security concerns into a practical, implementable framework with actionable controls, thereby providing essential guidance for the secure enterprise adoption and governance of integrated AI systems. 

**Abstract (ZH)**: Anthropic 提出的 Model Context Protocol (MCP) 提供了一种标准化框架，使人工智能（AI）系统能够实时与外部数据源和工具进行交互。尽管 MCP 为 AI 集成和能力扩展带来了显著优势，但也引入了新的安全挑战，需要进行严格的分析和缓解。本文基于 MCP 架构的基础研究和初步安全评估，提供企业级的安全缓解框架和详细的实施策略。通过系统性的威胁建模和 MCP 实施分析，包括高级威胁如工具投毒，我们提出了适用于 MCP 实施者和采用者的可操作性安全模式。本文的主要贡献在于将理论安全关注点转化为可操作且可实施的安全框架，从而为集成 AI 系统的可靠企业采用和治理提供必要的指导。 

---
# A Survey of Machine Learning Models and Datasets for the Multi-label Classification of Textual Hate Speech in English 

**Title (ZH)**: 英文多标签分类中基于文本的仇恨言论的机器学习模型与数据集综述 

**Authors**: Julian Bäumler, Louis Blöcher, Lars-Joel Frey, Xian Chen, Markus Bayer, Christian Reuter  

**Link**: [PDF](https://arxiv.org/pdf/2504.08609)  

**Abstract**: The dissemination of online hate speech can have serious negative consequences for individuals, online communities, and entire societies. This and the large volume of hateful online content prompted both practitioners', i.e., in content moderation or law enforcement, and researchers' interest in machine learning models to automatically classify instances of hate speech. Whereas most scientific works address hate speech classification as a binary task, practice often requires a differentiation into sub-types, e.g., according to target, severity, or legality, which may overlap for individual content. Hence, researchers created datasets and machine learning models that approach hate speech classification in textual data as a multi-label problem. This work presents the first systematic and comprehensive survey of scientific literature on this emerging research landscape in English (N=46). We contribute with a concise overview of 28 datasets suited for training multi-label classification models that reveals significant heterogeneity regarding label-set, size, meta-concept, annotation process, and inter-annotator agreement. Our analysis of 24 publications proposing suitable classification models further establishes inconsistency in evaluation and a preference for architectures based on Bidirectional Encoder Representation from Transformers (BERT) and Recurrent Neural Networks (RNNs). We identify imbalanced training data, reliance on crowdsourcing platforms, small and sparse datasets, and missing methodological alignment as critical open issues and formulate ten recommendations for research. 

**Abstract (ZH)**: 在线仇恨言论的传播可能对个人、在线社区和整个社会产生严重的负面影响。这一现象以及大量的仇恨在线内容促使内容审核从业者和研究人员对机器学习模型的兴趣，这些模型能够自动对仇恨言论进行分类。虽然大多数科学研究将仇恨言论分类视为一个二元任务，但实践中经常需要对不同类型进行区分，例如根据目标、严重程度或合法性进行分类，这些分类可能在个别内容上有所重叠。因此，研究人员创建了适用于训练多标签分类模型的数据集和机器学习模型。本文呈现了对该新兴研究领域的第一次系统性和综合性的英文文献综述（N=46）。我们提供了一个简洁的数据集概述，涵盖了28个适用于训练多标签分类模型的数据集，并揭示了关于标签集、规模、元概念、标注过程和注释者间一致性方面的显著异质性。我们对24篇提出合适分类模型的出版物的分析进一步确立了评估的不一致性，并倾向于使用双向Transformer编码表示（BERT）和循环神经网络（RNN）的架构。我们确定了训练数据的不平衡、对众包平台的依赖、数据集规模小和稀疏、以及缺乏方法学一致性作为关键的开放问题，并提出了十项研究建议。 

---
# Neural Fidelity Calibration for Informative Sim-to-Real Adaptation 

**Title (ZH)**: 神经保真度校准以实现信息性的仿真实践转换 

**Authors**: Youwei Yu, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08604)  

**Abstract**: Deep reinforcement learning can seamlessly transfer agile locomotion and navigation skills from the simulator to real world. However, bridging the sim-to-real gap with domain randomization or adversarial methods often demands expert physics knowledge to ensure policy robustness. Even so, cutting-edge simulators may fall short of capturing every real-world detail, and the reconstructed environment may introduce errors due to various perception uncertainties. To address these challenges, we propose Neural Fidelity Calibration (NFC), a novel framework that employs conditional score-based diffusion models to calibrate simulator physical coefficients and residual fidelity domains online during robot execution. Specifically, the residual fidelity reflects the simulation model shift relative to the real-world dynamics and captures the uncertainty of the perceived environment, enabling us to sample realistic environments under the inferred distribution for policy fine-tuning. Our framework is informative and adaptive in three key ways: (a) we fine-tune the pretrained policy only under anomalous scenarios, (b) we build sequential NFC online with the pretrained NFC's proposal prior, reducing the diffusion model's training burden, and (c) when NFC uncertainty is high and may degrade policy improvement, we leverage optimistic exploration to enable hallucinated policy optimization. Our framework achieves superior simulator calibration precision compared to state-of-the-art methods across diverse robots with high-dimensional parametric spaces. We study the critical contribution of residual fidelity to policy improvement in simulation and real-world experiments. Notably, our approach demonstrates robust robot navigation under challenging real-world conditions, such as a broken wheel axle on snowy surfaces. 

**Abstract (ZH)**: 深度强化学习可以从仿真器无缝转移敏捷运动和导航技能到现实世界。然而，通过领域随机化或对抗方法 bridging sim-to-real 隔离往往需要专家物理知识以确保策略稳健性。即使如此，最新的仿真器可能无法捕获所有现实世界的细节，重建的环境也可能由于各种感知不确定性引入错误。为应对这些挑战，我们提出了一种新型框架神经保真度校准（NFC），该框架利用条件评分扩散模型在线校准仿真物理系数和剩余保真度域，同时机器人执行。具体而言，剩余保真度反映了仿真模型相对于真实世界动力学的变化，并捕获感知环境的不确定性，使我们能够根据推断出的分布采样现实环境以进行策略微调。我们的框架在三个方面具有启发性和适应性：(a) 我们仅在异常场景中微调预训练策略，(b) 我们构建序列化的 NFC，在线构建在预训练 NFC 的提案先验之上，减轻扩散模型的训练负担，(c) 当 NFC 不确定性高且可能降低策略改进时，我们利用乐观探索以允许臆想策略优化。与现有最佳方法相比，我们的框架在具有高维参数空间的各种机器人中实现了优越的仿真校准精度。我们在仿真和真实世界实验中研究了剩余保真度对策略改进的关键贡献。值得注意的是，我们的方法在具有挑战性的现实世界条件下展示了稳健的机器人导航，例如雪地表面轮轴损坏。 

---
# FindAnything: Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment 

**Title (ZH)**: FindAnything：任意环境机器人探索的开放词汇和物体中心映射 

**Authors**: Sebastián Barbas Laina, Simon Boche, Sotiris Papatheodorou, Simon Schaefer, Jaehyung Jung, Stefan Leutenegger  

**Link**: [PDF](https://arxiv.org/pdf/2504.08603)  

**Abstract**: Geometrically accurate and semantically expressive map representations have proven invaluable to facilitate robust and safe mobile robot navigation and task planning. Nevertheless, real-time, open-vocabulary semantic understanding of large-scale unknown environments is still an open problem. In this paper we present FindAnything, an open-world mapping and exploration framework that incorporates vision-language information into dense volumetric submaps. Thanks to the use of vision-language features, FindAnything bridges the gap between pure geometric and open-vocabulary semantic information for a higher level of understanding while allowing to explore any environment without the help of any external source of ground-truth pose information. We represent the environment as a series of volumetric occupancy submaps, resulting in a robust and accurate map representation that deforms upon pose updates when the underlying SLAM system corrects its drift, allowing for a locally consistent representation between submaps. Pixel-wise vision-language features are aggregated from efficient SAM (eSAM)-generated segments, which are in turn integrated into object-centric volumetric submaps, providing a mapping from open-vocabulary queries to 3D geometry that is scalable also in terms of memory usage. The open-vocabulary map representation of FindAnything achieves state-of-the-art semantic accuracy in closed-set evaluations on the Replica dataset. This level of scene understanding allows a robot to explore environments based on objects or areas of interest selected via natural language queries. Our system is the first of its kind to be deployed on resource-constrained devices, such as MAVs, leveraging vision-language information for real-world robotic tasks. 

**Abstract (ZH)**: 几何准确且语义丰富的地图表示对于促进移动机器人 robust 和安全的导航及任务规划具有重要作用。然而，实时、开放式词汇语义理解大规模未知环境仍然是一个待解决的问题。本文提出 FindAnything，一种结合视觉-语言信息的开放世界建图与探索框架，通过使用视觉-语言特征，在纯几何和开放式词汇语义信息之间架起桥梁，实现更高层次的理解，同时允许在无需任何外部姿态信息源的情况下探索任何环境。我们将环境表示为一系列体积占用子地图，产生一种在姿态更新时能够变形的稳健且精确的地图表示，当底层 SLAM 系统纠正漂移时，能够提供局部一致的子地图表示。像素级的视觉-语言特征从高效的 eSAM 生成的分割中聚合而来，并整合进以物体为中心的体积子地图，提供一种从开放式词汇查询到 3D 几何的映射，同时在内存使用方面具备可扩展性。FindAnything 的开放式词汇地图表示在 Replica 数据集的封闭集评估中达到最先进的语义准确性。这种场景理解水平使机器人能够根据通过自然语言查询选定的对象或关注区域来探索环境。我们的系统是首个在资源受限设备（如 MAVs）上部署的系统，利用视觉-语言信息为现实中的机器人任务提供支持。 

---
# On Background Bias of Post-Hoc Concept Embeddings in Computer Vision DNNs 

**Title (ZH)**: 计算机视觉DNN中后验概念嵌入的背景偏差 

**Authors**: Gesina Schwalbe, Georgii Mikriukov, Edgar Heinert, Stavros Gerolymatos, Mert Keser, Alois Knoll, Matthias Rottmann, Annika Mütze  

**Link**: [PDF](https://arxiv.org/pdf/2504.08602)  

**Abstract**: The thriving research field of concept-based explainable artificial intelligence (C-XAI) investigates how human-interpretable semantic concepts embed in the latent spaces of deep neural networks (DNNs). Post-hoc approaches therein use a set of examples to specify a concept, and determine its embeddings in DNN latent space using data driven techniques. This proved useful to uncover biases between different target (foreground or concept) classes. However, given that the background is mostly uncontrolled during training, an important question has been left unattended so far: Are/to what extent are state-of-the-art, data-driven post-hoc C-XAI approaches themselves prone to biases with respect to their backgrounds? E.g., wild animals mostly occur against vegetation backgrounds, and they seldom appear on roads. Even simple and robust C-XAI methods might abuse this shortcut for enhanced performance. A dangerous performance degradation of the concept-corner cases of animals on the road could thus remain undiscovered. This work validates and thoroughly confirms that established Net2Vec-based concept segmentation techniques frequently capture background biases, including alarming ones, such as underperformance on road scenes. For the analysis, we compare 3 established techniques from the domain of background randomization on >50 concepts from 2 datasets, and 7 diverse DNN architectures. Our results indicate that even low-cost setups can provide both valuable insight and improved background robustness. 

**Abstract (ZH)**: 基于概念的可解释人工智能（C-XAI）的蓬勃研究领域探讨了人类可解释的语义概念如何嵌入深度神经网络（DNNs）的潜在空间中。其中的后处理方法使用一组示例来定义一个概念，并利用数据驱动的方法确定其在DNN潜在空间中的嵌入。这种方法已被证明对揭示不同目标类（前景或概念）之间的偏差很有用。然而，由于训练过程中背景大多是不受控制的，到目前为止一个重要的问题尚未得到解答：最先进的数据驱动后处理C-XAI方法是否以及在多大程度上对背景本身也存在偏差？例如，野生动物通常出现在植被背景中，很少出现在道路上。即使是简单而稳健的C-XAI方法也可能利用这种捷径以提升性能。因此，动物在道路上的概念极端案例的性能下降可能会被忽视。本研究验证并充分证实了基于Net2Vec的概念分割技术经常捕捉到背景偏差，包括令人担忧的偏差，如在道路场景上的表现不佳。为了分析，我们在两个数据集的50多个概念和7种不同的DNN架构上比较了3种现有的背景随机化技术。结果显示，即使低成本设置也能提供有价值的见解并提高背景的鲁棒性。 

---
# MedHal: An Evaluation Dataset for Medical Hallucination Detection 

**Title (ZH)**: MedHal: 医学幻觉检测评价数据集 

**Authors**: Gaya Mehenni, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2504.08596)  

**Abstract**: We present MedHal, a novel large-scale dataset specifically designed to evaluate if models can detect hallucinations in medical texts. Current hallucination detection methods face significant limitations when applied to specialized domains like medicine, where they can have disastrous consequences. Existing medical datasets are either too small, containing only a few hundred samples, or focus on a single task like Question Answering or Natural Language Inference. MedHal addresses these gaps by: (1) incorporating diverse medical text sources and tasks; (2) providing a substantial volume of annotated samples suitable for training medical hallucination detection models; and (3) including explanations for factual inconsistencies to guide model learning. We demonstrate MedHal's utility by training and evaluating a baseline medical hallucination detection model, showing improvements over general-purpose hallucination detection approaches. This resource enables more efficient evaluation of medical text generation systems while reducing reliance on costly expert review, potentially accelerating the development of medical AI research. 

**Abstract (ZH)**: MedHal：一种用于评估模型在医学文本中检测幻觉能力的新颖大规模数据集 

---
# Hands-On: Segmenting Individual Signs from Continuous Sequences 

**Title (ZH)**: 实战分析：从连续序列中分割个体手势 

**Authors**: Low Jian He, Harry Walsh, Ozge Mercanoglu Sincan, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2504.08593)  

**Abstract**: This work tackles the challenge of continuous sign language segmentation, a key task with huge implications for sign language translation and data annotation. We propose a transformer-based architecture that models the temporal dynamics of signing and frames segmentation as a sequence labeling problem using the Begin-In-Out (BIO) tagging scheme. Our method leverages the HaMeR hand features, and is complemented with 3D Angles. Extensive experiments show that our model achieves state-of-the-art results on the DGS Corpus, while our features surpass prior benchmarks on BSLCorpus. 

**Abstract (ZH)**: 基于变压器的手语连续分割挑战及其在手语翻译和数据标注中的关键作用：一种使用Begin-In-Out标记方案建模手语时空动态的序列标注方法及其实验研究 

---
# Ready, Bid, Go! On-Demand Delivery Using Fleets of Drones with Unknown, Heterogeneous Energy Storage Constraints 

**Title (ZH)**: 随时准备，出价出发！基于未知且异构能量存储约束无人机队的按需交付 

**Authors**: Mohamed S. Talamali, Genki Miyauchi, Thomas Watteyne, Micael S. Couceiro, Roderich Gross  

**Link**: [PDF](https://arxiv.org/pdf/2504.08585)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are expected to transform logistics, reducing delivery time, costs, and emissions. This study addresses an on-demand delivery , in which fleets of UAVs are deployed to fulfil orders that arrive stochastically. Unlike previous work, it considers UAVs with heterogeneous, unknown energy storage capacities and assumes no knowledge of the energy consumption models. We propose a decentralised deployment strategy that combines auction-based task allocation with online learning. Each UAV independently decides whether to bid for orders based on its energy storage charge level, the parcel mass, and delivery distance. Over time, it refines its policy to bid only for orders within its capability. Simulations using realistic UAV energy models reveal that, counter-intuitively, assigning orders to the least confident bidders reduces delivery times and increases the number of successfully fulfilled orders. This strategy is shown to outperform threshold-based methods which require UAVs to exceed specific charge levels at deployment. We propose a variant of the strategy which uses learned policies for forecasting. This enables UAVs with insufficient charge levels to commit to fulfilling orders at specific future times, helping to prioritise early orders. Our work provides new insights into long-term deployment of UAV swarms, highlighting the advantages of decentralised energy-aware decision-making coupled with online learning in real-world dynamic environments. 

**Abstract (ZH)**: 无人飞机(UAVs)有望通过减少交付时间和成本、降低排放来变革物流。本文研究了按需送货问题，在该问题中，调度员将派遣具有异构且未知能量存储容量的无人机机群来履行随机到达的订单。不同于以往的研究，本研究假设没有无人机能量消耗模型的知识，并提出了一种结合拍卖式任务分配和在线学习的分散部署策略。每架无人机根据其能量存储电荷水平、包裹质量和送货距离独立决定是否竞标订单，并随着时间的推移不断优化其仅竞标自己能够胜任的订单的策略。通过使用实际的无人机能量模型进行仿真，结果显示，反直觉地，将订单分配给最不自信的竞标者可以减少交付时间并增加成功完成的订单数量。该策略被证明优于要求无人机在部署时达到特定电荷水平的阈值方法。我们提出了一种使用学习策略进行预测的策略变体，这使得电荷水平不足的无人机能够承诺在特定未来时间履行订单，有助于优先处理早期订单。本研究为无人机机群的长期部署提供了新的见解，强调了在现实动态环境中的分散能耗意识决策与在线学习相结合的优势。 

---
# Boosting multi-demographic federated learning for chest x-ray analysis using general-purpose self-supervised representations 

**Title (ZH)**: 使用通用自监督表示增强多人口联邦学习的胸腔X光分析 

**Authors**: Mahshad Lotfinia, Arash Tayebiarasteh, Samaneh Samiei, Mehdi Joodaki, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2504.08584)  

**Abstract**: Reliable artificial intelligence (AI) models for medical image analysis often depend on large and diverse labeled datasets. Federated learning (FL) offers a decentralized and privacy-preserving approach to training but struggles in highly non-independent and identically distributed (non-IID) settings, where institutions with more representative data may experience degraded performance. Moreover, existing large-scale FL studies have been limited to adult datasets, neglecting the unique challenges posed by pediatric data, which introduces additional non-IID variability. To address these limitations, we analyzed n=398,523 adult chest radiographs from diverse institutions across multiple countries and n=9,125 pediatric images, leveraging transfer learning from general-purpose self-supervised image representations to classify pneumonia and cases with no abnormality. Using state-of-the-art vision transformers, we found that FL improved performance only for smaller adult datasets (P<0.001) but degraded performance for larger datasets (P<0.064) and pediatric cases (P=0.242). However, equipping FL with self-supervised weights significantly enhanced outcomes across pediatric cases (P=0.031) and most adult datasets (P<0.008), except the largest dataset (P=0.052). These findings underscore the potential of easily deployable general-purpose self-supervised image representations to address non-IID challenges in clinical FL applications and highlight their promise for enhancing patient outcomes and advancing pediatric healthcare, where data scarcity and variability remain persistent obstacles. 

**Abstract (ZH)**: 可靠的人工智能模型在医学图像分析中往往依赖于大规模和多样性的标注数据集。联邦学习（FL）提供了一种去中心化和保护隐私的训练方法，但在高度非独立和非同分布（non-IID）的环境中表现较差，其中具有代表性数据的机构可能会遭受性能下降。此外，现有的大规模FL研究主要局限于成人数据集，忽视了儿科数据带来的独特挑战，这引入了额外的非IID变异性。为解决这些局限性，我们分析了来自多个国家的398,523张成人胸部X光片和9,125张儿科图像，利用通用自监督图像表示进行迁移学习，对肺炎和无异常情况进行分类。使用最先进的视觉变压器，我们发现FL仅对较小的成人数据集提高了性能（P<0.001），但对较大的数据集和儿科病例的性能有所下降（P<0.064和P=0.242）。然而，装备FL以使用自监督权重显著提高了儿科病例（P=0.031）和大多数成人数据集（P<0.008）的结果，尽管对最大的数据集（P=0.052）未见显著改善。这些发现强调了通用自监督图像表示在临床FL应用中解决非IID挑战的潜力，并突显了其在提高患者结果和推动儿科医疗保健方面减轻数据稀缺和变异性障碍的前景。 

---
# Uncovering the Structure of Explanation Quality with Spectral Analysis 

**Title (ZH)**: 解释质量的结构发现基于谱分析 

**Authors**: Johannes Maeß, Grégoire Montavon, Shinichi Nakajima, Klaus-Robert Müller, Thomas Schnake  

**Link**: [PDF](https://arxiv.org/pdf/2504.08553)  

**Abstract**: As machine learning models are increasingly considered for high-stakes domains, effective explanation methods are crucial to ensure that their prediction strategies are transparent to the user. Over the years, numerous metrics have been proposed to assess quality of explanations. However, their practical applicability remains unclear, in particular due to a limited understanding of which specific aspects each metric rewards. In this paper we propose a new framework based on spectral analysis of explanation outcomes to systematically capture the multifaceted properties of different explanation techniques. Our analysis uncovers two distinct factors of explanation quality-stability and target sensitivity-that can be directly observed through spectral decomposition. Experiments on both MNIST and ImageNet show that popular evaluation techniques (e.g., pixel-flipping, entropy) partially capture the trade-offs between these factors. Overall, our framework provides a foundational basis for understanding explanation quality, guiding the development of more reliable techniques for evaluating explanations. 

**Abstract (ZH)**: 随着机器学习模型在高风险领域中的应用越来越广泛，有效的解释方法对于确保其预测策略对用户透明至关重要。多年来，已经提出了诸多评估解释质量的指标，但这些指标的实际适用性仍然不明确，特别是由于我们对各项指标奖励的具体方面缺乏充分的理解。在本文中，我们提出了一种基于解释结果的谱分析的新框架，以系统地捕捉不同解释技术的多方面特性。我们的分析揭示了解释质量的两个不同因素：稳定性和目标敏感性，这些因素可以通过谱分解直接观察到。在MNIST和ImageNet数据集上的实验表明，流行的评估技术（例如，像素翻转、熵）部分地捕捉到了这些因素之间的权衡。总体而言，本文框架为理解解释质量奠定了基础，并指导了更可靠解释评估技术的发展。 

---
# Proxy-Anchor and EVT-Driven Continual Learning Method for Generalized Category Discovery 

**Title (ZH)**: 代理锚点和 EVT 驱动的持续学习方法用于泛化类别发现 

**Authors**: Alireza Fathalizadeh, Roozbeh Razavi-Far  

**Link**: [PDF](https://arxiv.org/pdf/2504.08550)  

**Abstract**: Continual generalized category discovery has been introduced and studied in the literature as a method that aims to continuously discover and learn novel categories in incoming data batches while avoiding catastrophic forgetting of previously learned categories. A key component in addressing this challenge is the model's ability to separate novel samples, where Extreme Value Theory (EVT) has been effectively employed. In this work, we propose a novel method that integrates EVT with proxy anchors to define boundaries around proxies using a probability of inclusion function, enabling the rejection of unknown samples. Additionally, we introduce a novel EVT-based loss function to enhance the learned representation, achieving superior performance compared to other deep-metric learning methods in similar settings. Using the derived probability functions, novel samples are effectively separated from previously known categories. However, category discovery within these novel samples can sometimes overestimate the number of new categories. To mitigate this issue, we propose a novel EVT-based approach to reduce the model size and discard redundant proxies. We also incorporate experience replay and knowledge distillation mechanisms during the continual learning stage to prevent catastrophic forgetting. Experimental results demonstrate that our proposed approach outperforms state-of-the-art methods in continual generalized category discovery scenarios. 

**Abstract (ZH)**: 连续广义类别发现已被文献引入并研究作为一种方法，旨在在不断发现和学习新类别同时避免遗忘之前学习类别的灾难性遗忘。解决这一挑战的关键在于模型区分新型样本的能力，极值理论（EVT）已被有效应用于这一领域。在本工作中，我们提出了一种将EVT与代理锚点相结合的新方法，使用包含概率函数定义代理的边界，从而排斥未知样本。此外，我们引入了一种基于EVT的损失函数来增强学习表示，相比其他深度度量学习方法，在类似设置中取得了更优性能。通过衍生的概率函数，新型样本能够有效与已知类别分开。然而，对这些新型样本中的类别发现有时会高估新类别的数量。为缓解这一问题，我们提出了一种基于EVT的方法来减小模型大小并丢弃冗余代理。此外，在连续学习阶段，我们还引入了经验重放和知识蒸馏机制以防止灾难性遗忘。实验结果表明，在连续广义类别发现场景中，我们提出的方法优于现有最先进的方法。 

---
# Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset 

**Title (ZH)**: 数字孪生目录：大规模真实感3D对象数字孪生数据集 

**Authors**: Zhao Dong, Ka Chen, Zhaoyang Lv, Hong-Xing Yu, Yunzhi Zhang, Cheng Zhang, Yufeng Zhu, Stephen Tian, Zhengqin Li, Geordie Moffatt, Sean Christofferson, James Fort, Xiaqing Pan, Mingfei Yan, Jiajun Wu, Carl Yuheng Ren, Richard Newcombe  

**Link**: [PDF](https://arxiv.org/pdf/2504.08541)  

**Abstract**: We introduce Digital Twin Catalog (DTC), a new large-scale photorealistic 3D object digital twin dataset. A digital twin of a 3D object is a highly detailed, virtually indistinguishable representation of a physical object, accurately capturing its shape, appearance, physical properties, and other attributes. Recent advances in neural-based 3D reconstruction and inverse rendering have significantly improved the quality of 3D object reconstruction. Despite these advancements, there remains a lack of a large-scale, digital twin quality real-world dataset and benchmark that can quantitatively assess and compare the performance of different reconstruction methods, as well as improve reconstruction quality through training or fine-tuning. Moreover, to democratize 3D digital twin creation, it is essential to integrate creation techniques with next-generation egocentric computing platforms, such as AR glasses. Currently, there is no dataset available to evaluate 3D object reconstruction using egocentric captured images. To address these gaps, the DTC dataset features 2,000 scanned digital twin-quality 3D objects, along with image sequences captured under different lighting conditions using DSLR cameras and egocentric AR glasses. This dataset establishes the first comprehensive real-world evaluation benchmark for 3D digital twin creation tasks, offering a robust foundation for comparing and improving existing reconstruction methods. The DTC dataset is already released at this https URL and we will also make the baseline evaluations open-source. 

**Abstract (ZH)**: 数字孪生目录（DTC）：大规模高保真3D对象数字孪生数据集 

---
# Explainability and Continual Learning meet Federated Learning at the Network Edge 

**Title (ZH)**: 网络边缘处的可解释性和持续学习与联邦学习的结合 

**Authors**: Thomas Tsouparopoulos, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.08536)  

**Abstract**: As edge devices become more capable and pervasive in wireless networks, there is growing interest in leveraging their collective compute power for distributed learning. However, optimizing learning at the network edge entails unique challenges, particularly when moving beyond conventional settings and objectives. While Federated Learning (FL) has emerged as a key paradigm for distributed model training, critical challenges persist. First, existing approaches often overlook the trade-off between predictive accuracy and interpretability. Second, they struggle to integrate inherently explainable models such as decision trees because their non-differentiable structure makes them not amenable to backpropagation-based training algorithms. Lastly, they lack meaningful mechanisms for continual Machine Learning (ML) model adaptation through Continual Learning (CL) in resource-limited environments. In this paper, we pave the way for a set of novel optimization problems that emerge in distributed learning at the network edge with wirelessly interconnected edge devices, and we identify key challenges and future directions. Specifically, we discuss how Multi-objective optimization (MOO) can be used to address the trade-off between predictive accuracy and explainability when using complex predictive models. Next, we discuss the implications of integrating inherently explainable tree-based models into distributed learning settings. Finally, we investigate how CL strategies can be effectively combined with FL to support adaptive, lifelong learning when limited-size buffers are used to store past data for retraining. Our approach offers a cohesive set of tools for designing privacy-preserving, adaptive, and trustworthy ML solutions tailored to the demands of edge computing and intelligent services. 

**Abstract (ZH)**: 随着无线网络中边缘设备的能力不断增强和普及，人们越来越关注利用其集体计算能力进行分布式学习。然而，在网络边缘优化学习带来了独特的挑战，尤其是在超越传统环境和目标时。尽管联邦学习（FL）已成为分布式模型训练的关键范式，但仍存在关键挑战。首先，现有方法往往忽视了预测准确性和解释性之间的权衡。其次，它们难以整合如决策树等固有的可解释模型，因为它们的非可微结构使得它们不适用于基于反向传播的训练算法。最后，它们缺乏有效的机制通过连续学习（CL）在资源受限的环境中对机器学习（ML）模型进行持续适应。在本文中，我们探讨了在网络边缘由无线互联的边缘设备进行分布式学习时出现的一系列新颖优化问题，并指出了关键挑战和未来方向。具体而言，我们讨论了多目标优化（MOO）如何用于解决在使用复杂预测模型时预测准确性和解释性之间的权衡。接下来，我们探讨了如何将固有的可解释树模型整合到分布式学习环境中。最后，我们研究了如何结合连续学习（CL）策略与联邦学习（FL），在仅使用有限大小的缓存存储过往数据以进行重新训练时，支持适应性和终身学习。我们的方法提供了一套综合工具，用于为边缘计算和智能服务的需求设计隐私保护、适应性和可信赖的机器学习解决方案。 

---
# LGRPool: Hierarchical Graph Pooling Via Local-Global Regularisation 

**Title (ZH)**: LGRPool：基于局部-全局正则化的分层图池化 

**Authors**: Farshad Noravesh, Reza Haffari, Layki Soon, Arghya Pal  

**Link**: [PDF](https://arxiv.org/pdf/2504.08530)  

**Abstract**: Hierarchical graph pooling(HGP) are designed to consider the fact that conventional graph neural networks(GNN) are inherently flat and are also not multiscale. However, most HGP methods suffer not only from lack of considering global topology of the graph and focusing on the feature learning aspect, but also they do not align local and global features since graphs should inherently be analyzed in a multiscale way. LGRPool is proposed in the present paper as a HGP in the framework of expectation maximization in machine learning that aligns local and global aspects of message passing with each other using a regularizer to force the global topological information to be inline with the local message passing at different scales through the representations at different layers of HGP. Experimental results on some graph classification benchmarks show that it slightly outperforms some baselines. 

**Abstract (ZH)**: 基于期望最大化的层次图池化方法：LGRPool 

---
# Hallucination, reliability, and the role of generative AI in science 

**Title (ZH)**: 幻觉、可靠性和生成性AI在科学中的作用 

**Authors**: Charles Rathkopf  

**Link**: [PDF](https://arxiv.org/pdf/2504.08526)  

**Abstract**: Generative AI is increasingly used in scientific domains, from protein folding to climate modeling. But these models produce distinctive errors known as hallucinations - outputs that are incorrect yet superficially plausible. Worse, some arguments suggest that hallucinations are an inevitable consequence of the mechanisms underlying generative inference. Fortunately, such arguments rely on a conception of hallucination defined solely with respect to internal properties of the model, rather than in reference to the empirical target system. This conception fails to distinguish epistemically benign errors from those that threaten scientific inference. I introduce the concept of corrosive hallucination to capture the epistemically troubling subclass: misrepresentations that are substantively misleading and resistant to systematic anticipation. I argue that although corrosive hallucinations do pose a threat to scientific reliability, they are not inevitable. Scientific workflows such as those surrounding AlphaFold and GenCast, both of which serve as case studies, can neutralize their effects by imposing theoretical constraints during training, and by strategically screening for errors at inference time. When embedded in such workflows, generative AI can reliably contribute to scientific knowledge. 

**Abstract (ZH)**: 生成式AI在科学领域中的应用日益广泛，从蛋白质折叠到气候建模。然而，这些模型会产生一种称为错觉的独特错误——这些输出虽然表面上看似合理，但实际上却是错误的。更糟糕的是，一些论点表明，错觉可能是生成推理机制不可避免的后果。幸运的是，这些论点依赖于仅基于模型内部属性来定义错觉的概念，而不是参照实际的目标系统。这种概念无法区分那些在科学推理中本质上无害的错误和那些构成威胁的错误。我引入了腐蚀性错觉的概念，以捕捉这一类本质上有问题的子类：那些实质上误导且难以系统预见的误导性描述。我认为，尽管腐蚀性错觉确实对科学可靠性构成威胁，但它们并非不可避免。如围绕AlphaFold和GenCast的工作流程，这些可以防范其影响，通过在训练过程中施加理论约束，并在推理时战略性地筛选错误。在这些工作流程中嵌入生成式AI可以确保其对科学知识的可靠贡献。 

---
# Mitigating Timbre Leakage with Universal Semantic Mapping Residual Block for Voice Conversion 

**Title (ZH)**: 基于通用语义映射残差块的timbre泄漏减轻语音转换方法 

**Authors**: Na Li, Chuke Wang, Yu Gu, Zhifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08524)  

**Abstract**: Voice conversion (VC) transforms source speech into a target voice by preserving the content. However, timbre information from the source speaker is inherently embedded in the content representations, causing significant timbre leakage and reducing similarity to the target speaker. To address this, we introduce a residual block to a content extractor. The residual block consists of two weighted branches: 1) universal semantic dictionary based Content Feature Re-expression (CFR) module, supplying timbre-free content representation. 2) skip connection to the original content layer, providing complementary fine-grained information. In the CFR module, each dictionary entry in the universal semantic dictionary represents a phoneme class, computed statistically using speech from multiple speakers, creating a stable, speaker-independent semantic set. We introduce a CFR method to obtain timbre-free content representations by expressing each content frame as a weighted linear combination of dictionary entries using corresponding phoneme posteriors as weights. Extensive experiments across various VC frameworks demonstrate that our approach effectively mitigates timbre leakage and significantly improves similarity to the target speaker. 

**Abstract (ZH)**: Voice转换（VC）通过保留内容将源语音转换为目标声音，但由于声韵信息会固有地嵌入内容表示中，从而导致显著的声韵泄漏，降低与目标说话人的相似度。为解决这一问题，我们将在内容提取器中引入一个残差块。该残差块包括两个加权支路：1) 基于通用语义词典的内容特征重表达（CFR）模块，提供无声韵的内容表示。2) 与原始内容层的跳跃连接，提供补充的细粒度信息。在CFR模块中，通用语义词典中的每个字典条目代表一个音素类别，通过多说话人口头语言的统计计算获得，形成一个稳定且说话人无关的语义集。我们提出了一种CFR方法，通过使用相应的音素后验概率作为权重，将每个内容帧表示为字典条目的加权线性组合，从而获得无声韵的内容表示。在各种VC框架上的广泛实验表明，我们的方法有效地减轻了声韵泄漏，显著提高了与目标说话人的相似度。 

---
# Adopting Large Language Models to Automated System Integration 

**Title (ZH)**: 采用大型语言模型进行自动化系统集成 

**Authors**: Robin D. Pesl  

**Link**: [PDF](https://arxiv.org/pdf/2504.08490)  

**Abstract**: Modern enterprise computing systems integrate numerous subsystems to resolve a common task by yielding emergent behavior. A widespread approach is using services implemented with Web technologies like REST or OpenAPI, which offer an interaction mechanism and service documentation standard, respectively. Each service represents a specific business functionality, allowing encapsulation and easier maintenance. Despite the reduced maintenance costs on an individual service level, increased integration complexity arises. Consequently, automated service composition approaches have arisen to mitigate this issue. Nevertheless, these approaches have not achieved high acceptance in practice due to their reliance on complex formal modeling. Within this Ph.D. thesis, we analyze the application of Large Language Models (LLMs) to automatically integrate the services based on a natural language input. The result is a reusable service composition, e.g., as program code. While not always generating entirely correct results, the result can still be helpful by providing integration engineers with a close approximation of a suitable solution, which requires little effort to become operational. Our research involves (i) introducing a software architecture for automated service composition using LLMs, (ii) analyzing Retrieval Augmented Generation (RAG) for service discovery, (iii) proposing a novel natural language query-based benchmark for service discovery, and (iv) extending the benchmark to complete service composition scenarios. We have presented our software architecture as Compositio Prompto, the analysis of RAG for service discovery, and submitted a proposal for the service discovery benchmark. Open topics are primarily the extension of the service discovery benchmark to service composition scenarios and the improvements of the service composition generation, e.g., using fine-tuning or LLM agents. 

**Abstract (ZH)**: 基于大规模语言模型的自动服务集成研究 

---
# A Hybrid Fully Convolutional CNN-Transformer Model for Inherently Interpretable Medical Image Classification 

**Title (ZH)**: 一种用于固有可解释医疗图像分类的混合卷积CNN-变换器模型 

**Authors**: Kerol Djoumessi, Samuel Ofosu Mensah, Philipp Berens  

**Link**: [PDF](https://arxiv.org/pdf/2504.08481)  

**Abstract**: In many medical imaging tasks, convolutional neural networks (CNNs) efficiently extract local features hierarchically. More recently, vision transformers (ViTs) have gained popularity, using self-attention mechanisms to capture global dependencies, but lacking the inherent spatial localization of convolutions. Therefore, hybrid models combining CNNs and ViTs have been developed to combine the strengths of both architectures. However, such hybrid CNN-ViT models are difficult to interpret, which hinders their application in medical imaging. In this work, we introduce an interpretable-by-design hybrid fully convolutional CNN-Transformer architecture for medical image classification. Unlike widely used post-hoc saliency methods for ViTs, our approach generates faithful and localized evidence maps that directly reflect the model's decision process. We evaluated our method on two medical image classification tasks using color fundus images. Our model not only achieves state-of-the-art predictive performance compared to both black-box and interpretable models but also provides class-specific sparse evidence maps in a single forward pass. The code is available at: this https URL. 

**Abstract (ZH)**: 在许多医学影像任务中，卷积神经网络（CNNs）高效地分层提取局部特征。近年来，视觉变压器（ViTs）因其使用自我注意机制捕获全局依赖性而受到关注，但缺乏卷积固有的空间定位能力。因此，结合了CNNs和ViTs优点的混合模型被开发出来。然而，这类混合CNN-ViT模型难以解释，限制了它们在医学影像中的应用。在这项工作中，我们提出了一个设计可解释的混合全卷积CNN-Transformer架构，用于医学图像分类。与广泛使用的事后ViTs可解释性方法不同，我们的方法生成忠实且局部化的证据图，直接反映模型的决策过程。我们在使用彩色视网膜图像的两个医学图像分类任务上评估了该方法。我们的模型不仅在预测性能上达到最佳效果，与黑盒模型和可解释模型相比，还在单次前传中提供了特定类别的稀疏证据图。代码可从以下链接获取：this https URL。 

---
# On the Design of Diffusion-based Neural Speech Codecs 

**Title (ZH)**: 基于扩散机制的神经语音编解码器设计 

**Authors**: Pietro Foti, Andreas Brendel  

**Link**: [PDF](https://arxiv.org/pdf/2504.08470)  

**Abstract**: Recently, neural speech codecs (NSCs) trained as generative models have shown superior performance compared to conventional codecs at low bitrates. Although most state-of-the-art NSCs are trained as Generative Adversarial Networks (GANs), Diffusion Models (DMs), a recent class of generative models, represent a promising alternative due to their superior performance in image generation relative to GANs. Consequently, DMs have been successfully applied for audio and speech coding among various other audio generation applications. However, the design of diffusion-based NSCs has not yet been explored in a systematic way. We address this by providing a comprehensive analysis of diffusion-based NSCs divided into three contributions. First, we propose a categorization based on the conditioning and output domains of the DM. This simple conceptual framework allows us to define a design space for diffusion-based NSCs and to assign a category to existing approaches in the literature. Second, we systematically investigate unexplored designs by creating and evaluating new diffusion-based NSCs within the conceptual framework. Finally, we compare the proposed models to existing GAN and DM baselines through objective metrics and subjective listening tests. 

**Abstract (ZH)**: Recently, 基于扩散模型的神经语音编码器在低比特率下的性能优于传统语音编解码器。尽管大多数最先进的基于扩散模型的神经语音编码器（NSCs）被训练为生成对抗网络（GANs），但由于在图像生成方面的优越性能，近年来的生成模型——扩散模型（DMs）提供了一种有前途的替代方案。因此，DMs已在各种其他音频生成应用中成功应用于语音编码。然而，基于扩散模型的NSCs的设计尚未以系统的方式进行探索。我们通过提供一个基于扩散模型的NSCs的全面分析来解决这个问题，并将其分为三个贡献。首先，我们根据DM的条件域和输出域提出了一种分类方法。这一简单的概念框架使我们能够定义基于扩散模型的NSCs的设计空间，并将文献中现有的方法归类。其次，我们系统地探索了未被广泛研究的设计，通过概念框架创建并评估了新的基于扩散模型的NSCs。最后，我们通过客观指标和主观听觉测试将所提出的模型与现有的GAN和DM基线进行比较。 

---
# Generalization Bounds in Hybrid Quantum-Classical Machine Learning Models 

**Title (ZH)**: 混合量子-经典机器学习模型的泛化误差界 

**Authors**: Tongyan Wu, Amine Bentellis, Alona Sakhnenko, Jeanette Miriam Lorenz  

**Link**: [PDF](https://arxiv.org/pdf/2504.08456)  

**Abstract**: Hybrid classical-quantum models aim to harness the strengths of both quantum computing and classical machine learning, but their practical potential remains poorly understood. In this work, we develop a unified mathematical framework for analyzing generalization in hybrid models, offering insight into how these systems learn from data. We establish a novel generalization bound of the form $O\big( \sqrt{\frac{T\log{T}}{N}} + \frac{\alpha}{\sqrt{N}}\big)$ for $N$ training data points, $T$ trainable quantum gates, and bounded fully-connected layers $||F|| \leq \alpha$. This bound decomposes cleanly into quantum and classical contributions, extending prior work on both components and clarifying their interaction. We apply our results to the quantum-classical convolutional neural network (QCCNN), an architecture that integrates quantum convolutional layers with classical processing. Alongside the bound, we highlight conceptual limitations of applying classical statistical learning theory in the hybrid setting and suggest promising directions for future theoretical work. 

**Abstract (ZH)**: 混合经典-量子模型旨在结合量子计算和经典机器学习的优势，但其实际潜力尚不完全理解。在本工作中，我们开发了一个统一的数学框架，用于分析混合模型的一般化能力，提供了关于这些系统如何从数据中学习的洞察。我们建立了形式为 $O\big( \sqrt{\frac{T\log{T}}{N}} + \frac{\alpha}{\sqrt{N}}\big)$ 的一般化界，适用于 $N$ 个训练数据点、$T$ 个可训练量子门和范数受限的全连接层 $||F|| \leq \alpha$。该界清楚地分解为量子和经典成分，扩展了两者的工作并澄清了它们的相互作用。我们应用我们的结果到结合了量子卷积层和经典处理的量子-经典卷积神经网络 (QCCNN) 架构。除了界之外，我们还指出了在混合设置中应用经典统计学习理论的概念限制，并建议了未来理论工作的前景方向。 

---
# seeBias: A Comprehensive Tool for Assessing and Visualizing AI Fairness 

**Title (ZH)**: seeBias: 一个全面的工具，用于评估和可视化AI公平性 

**Authors**: Yilin Ning, Yian Ma, Mingxuan Liu, Xin Li, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08418)  

**Abstract**: Fairness in artificial intelligence (AI) prediction models is increasingly emphasized to support responsible adoption in high-stakes domains such as health care and criminal justice. Guidelines and implementation frameworks highlight the importance of both predictive accuracy and equitable outcomes. However, current fairness toolkits often evaluate classification performance disparities in isolation, with limited attention to other critical aspects such as calibration. To address these gaps, we present seeBias, an R package for comprehensive evaluation of model fairness and predictive performance. seeBias offers an integrated evaluation across classification, calibration, and other performance domains, providing a more complete view of model behavior. It includes customizable visualizations to support transparent reporting and responsible AI implementation. Using public datasets from criminal justice and healthcare, we demonstrate how seeBias supports fairness evaluations, and uncovers disparities that conventional fairness metrics may overlook. The R package is available on GitHub, and a Python version is under development. 

**Abstract (ZH)**: 人工智能（AI）预测模型的公平性在高风险领域如医疗保健和刑事司法中受到越来越多的关注，以支持负责任的采用。指南和实施框架强调预测准确性和公平结果的重要性。然而，当前的公平性工具包往往孤立地评估分类性能差异，对校准等其他关键方面关注不足。为解决这些差距，我们介绍了seeBias，一个用于全面评估模型公平性和预测性能的R包。seeBias提供跨分类、校准和其他性能领域的综合评估，提供了模型行为的更完整视图。它包括可定制的可视化功能，以支持透明报告和负责任的AI实施。使用来自刑事司法和医疗保健的公共数据集，我们展示了seeBias如何支持公平性评估，并揭示了常规公平性指标可能忽略的差异。R包可在GitHub上获取，Python版本正在开发中。 

---
# Constrained Machine Learning Through Hyperspherical Representation 

**Title (ZH)**: 通过超球面表示实现约束机器学习 

**Authors**: Gaetano Signorelli, Michele Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08415)  

**Abstract**: The problem of ensuring constraints satisfaction on the output of machine learning models is critical for many applications, especially in safety-critical domains. Modern approaches rely on penalty-based methods at training time, which do not guarantee to avoid constraints violations; or constraint-specific model architectures (e.g., for monotonocity); or on output projection, which requires to solve an optimization problem that might be computationally demanding. We present the Hypersherical Constrained Representation, a novel method to enforce constraints in the output space for convex and bounded feasibility regions (generalizable to star domains). Our method operates on a different representation system, where Euclidean coordinates are converted into hyperspherical coordinates relative to the constrained region, which can only inherently represent feasible points. Experiments on a synthetic and a real-world dataset show that our method has predictive performance comparable to the other approaches, can guarantee 100% constraint satisfaction, and has a minimal computational cost at inference time. 

**Abstract (ZH)**: 确保机器学习模型输出满足约束的问题在许多应用中至关重要，特别是在安全关键领域。我们提出了超球面约束表示法，这是一种在凸且有界可行域（可推广到星域）中强制约束的新方法。该方法在一种不同的表示系统中运行，将欧clidean坐标转换为相对于约束区域的超球面坐标，这种表示法只能固有地表示可行点。实验结果表明，该方法在预测性能上与其它方法相当，能确保100%的约束满足，并且在推理时具有最小的计算成本。 

---
# A Knowledge-guided Adversarial Defense for Resisting Malicious Visual Manipulation 

**Title (ZH)**: 基于知识导向的对抗防御以抵抗恶意视觉操纵 

**Authors**: Dawei Zhou, Suzhi Gang, Decheng Liu, Tongliang Liu, Nannan Wang, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08411)  

**Abstract**: Malicious applications of visual manipulation have raised serious threats to the security and reputation of users in many fields. To alleviate these issues, adversarial noise-based defenses have been enthusiastically studied in recent years. However, ``data-only" methods tend to distort fake samples in the low-level feature space rather than the high-level semantic space, leading to limitations in resisting malicious manipulation. Frontier research has shown that integrating knowledge in deep learning can produce reliable and generalizable solutions. Inspired by these, we propose a knowledge-guided adversarial defense (KGAD) to actively force malicious manipulation models to output semantically confusing samples. Specifically, in the process of generating adversarial noise, we focus on constructing significant semantic confusions at the domain-specific knowledge level, and exploit a metric closely related to visual perception to replace the general pixel-wise metrics. The generated adversarial noise can actively interfere with the malicious manipulation model by triggering knowledge-guided and perception-related disruptions in the fake samples. To validate the effectiveness of the proposed method, we conduct qualitative and quantitative experiments on human perception and visual quality assessment. The results on two different tasks both show that our defense provides better protection compared to state-of-the-art methods and achieves great generalizability. 

**Abstract (ZH)**: 基于知识引导的对抗防御（KGAD）：主动迫使恶意操作输出语义混淆样本 

---
# Beyond Self-Reports: Multi-Observer Agents for Personality Assessment in Large Language Models 

**Title (ZH)**: 超越自我报告：大型语言模型中的人格评估多观察者代理方法 

**Authors**: Yin Jou Huang, Rafik Hadfi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08399)  

**Abstract**: There is a growing interest in assessing the personality traits of Large language models (LLMs). However, traditional personality assessments based on self-report questionnaires may fail to capture their true behavioral nuances due to inherent biases and meta-knowledge contamination. This paper introduces a novel multi-observer framework for LLM personality assessment that draws inspiration from informant-report methods in psychology. Instead of relying solely on self-assessments, our approach employs multiple observer agents configured with a specific relationship context (e.g., family, friend, or workplace) to simulate interactive scenarios with a subject LLM. These observers engage in dialogues and subsequently provide ratings across the Big Five personality dimensions. Our experiments reveal that LLMs possess systematic biases in self-report personality ratings. Moreover, aggregating observer ratings effectively reduces non-systematic biases and achieves optimal reliability with 5-7 observers. The findings highlight the significant impact of relationship context on personality perception and demonstrate that a multi-observer paradigm yields a more robust and context-sensitive evaluation of LLM personality traits. 

**Abstract (ZH)**: 大型语言模型的人格特质评估：一种多观察者框架的研究 

---
# Human strategies for correcting `human-robot' errors during a laundry sorting task 

**Title (ZH)**: 人类在洗衣分类任务中纠正“人-机器人”错误的策略 

**Authors**: Pepita Barnard, Maria J Galvez Trigo, Dominic Price, Sue Cobb, Gisela Reyes-Cruz, Gustavo Berumen, David Branson III, Mojtaba A. Khanesar, Mercedes Torres Torres, Michel Valstar  

**Link**: [PDF](https://arxiv.org/pdf/2504.08395)  

**Abstract**: Mental models and expectations underlying human-human interaction (HHI) inform human-robot interaction (HRI) with domestic robots. To ease collaborative home tasks by improving domestic robot speech and behaviours for human-robot communication, we designed a study to understand how people communicated when failure occurs. To identify patterns of natural communication, particularly in response to robotic failures, participants instructed Laundrobot to move laundry into baskets using natural language and gestures. Laundrobot either worked error-free, or in one of two error modes. Participants were not advised Laundrobot would be a human actor, nor given information about error modes. Video analysis from 42 participants found speech patterns, included laughter, verbal expressions, and filler words, such as ``oh'' and ``ok'', also, sequences of body movements, including touching one's own face, increased pointing with a static finger, and expressions of surprise. Common strategies deployed when errors occurred, included correcting and teaching, taking responsibility, and displays of frustration. The strength of reaction to errors diminished with exposure, possibly indicating acceptance or resignation. Some used strategies similar to those used to communicate with other technologies, such as smart assistants. An anthropomorphic robot may not be ideally suited to this kind of task. Laundrobot's appearance, morphology, voice, capabilities, and recovery strategies may have impacted how it was perceived. Some participants indicated Laundrobot's actual skills were not aligned with expectations; this made it difficult to know what to expect and how much Laundrobot understood. Expertise, personality, and cultural differences may affect responses, however these were not assessed. 

**Abstract (ZH)**: 人类心智模型与期望下的人际交互对人机交互的研究：以家用机器人为例——以洗碗机器人Laundrobot为对象探究交互中沟通方式及应对错误策略 

---
# MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft 

**Title (ZH)**: MineWorld: 一个实时开源的 Minecraft 交互世界模型 

**Authors**: Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08388)  

**Abstract**: World modeling is a crucial task for enabling intelligent agents to effectively interact with humans and operate in dynamic environments. In this work, we propose MineWorld, a real-time interactive world model on Minecraft, an open-ended sandbox game which has been utilized as a common testbed for world modeling. MineWorld is driven by a visual-action autoregressive Transformer, which takes paired game scenes and corresponding actions as input, and generates consequent new scenes following the actions. Specifically, by transforming visual game scenes and actions into discrete token ids with an image tokenizer and an action tokenizer correspondingly, we consist the model input with the concatenation of the two kinds of ids interleaved. The model is then trained with next token prediction to learn rich representations of game states as well as the conditions between states and actions simultaneously. In inference, we develop a novel parallel decoding algorithm that predicts the spatial redundant tokens in each frame at the same time, letting models in different scales generate $4$ to $7$ frames per second and enabling real-time interactions with game players. In evaluation, we propose new metrics to assess not only visual quality but also the action following capacity when generating new scenes, which is crucial for a world model. Our comprehensive evaluation shows the efficacy of MineWorld, outperforming SoTA open-sourced diffusion based world models significantly. The code and model have been released. 

**Abstract (ZH)**: 基于Minecraft的实时交互世界建模：MineWorld 

---
# PCA-RAG: Principal Component Analysis for Efficient Retrieval-Augmented Generation 

**Title (ZH)**: PCA-RAG: 主成分分析辅助的高效检索增强生成 

**Authors**: Arman Khaledian, Amirreza Ghadiridehkordi, Nariman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08386)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for grounding large language models in external knowledge sources, improving the precision of agents responses. However, high-dimensional language model embeddings, often in the range of hundreds to thousands of dimensions, can present scalability challenges in terms of storage and latency, especially when processing massive financial text corpora. This paper investigates the use of Principal Component Analysis (PCA) to reduce embedding dimensionality, thereby mitigating computational bottlenecks without incurring large accuracy losses. We experiment with a real-world dataset and compare different similarity and distance metrics under both full-dimensional and PCA-compressed embeddings. Our results show that reducing vectors from 3,072 to 110 dimensions provides a sizeable (up to $60\times$) speedup in retrieval operations and a $\sim 28.6\times$ reduction in index size, with only moderate declines in correlation metrics relative to human-annotated similarity scores. These findings demonstrate that PCA-based compression offers a viable balance between retrieval fidelity and resource efficiency, essential for real-time systems such as Zanista AI's \textit{Newswitch} platform. Ultimately, our study underscores the practicality of leveraging classical dimensionality reduction techniques to scale RAG architectures for knowledge-intensive applications in finance and trading, where speed, memory efficiency, and accuracy must jointly be optimized. 

**Abstract (ZH)**: 基于主成分分析的大规模语言模型嵌入维数缩减在金融文本处理中的应用 

---
# Scholar Inbox: Personalized Paper Recommendations for Scientists 

**Title (ZH)**: 学者收件箱：面向科学家的个性化论文推荐 

**Authors**: Markus Flicke, Glenn Angrabeit, Madhav Iyengar, Vitalii Protsenko, Illia Shakun, Jovan Cicvaric, Bora Kargi, Haoyu He, Lukas Schuler, Lewin Scholz, Kavyanjali Agnihotri, Yong Cao, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2504.08385)  

**Abstract**: Scholar Inbox is a new open-access platform designed to address the challenges researchers face in staying current with the rapidly expanding volume of scientific literature. We provide personalized recommendations, continuous updates from open-access archives (arXiv, bioRxiv, etc.), visual paper summaries, semantic search, and a range of tools to streamline research workflows and promote open research access. The platform's personalized recommendation system is trained on user ratings, ensuring that recommendations are tailored to individual researchers' interests. To further enhance the user experience, Scholar Inbox also offers a map of science that provides an overview of research across domains, enabling users to easily explore specific topics. We use this map to address the cold start problem common in recommender systems, as well as an active learning strategy that iteratively prompts users to rate a selection of papers, allowing the system to learn user preferences quickly. We evaluate the quality of our recommendation system on a novel dataset of 800k user ratings, which we make publicly available, as well as via an extensive user study. this https URL 

**Abstract (ZH)**: 学者收件箱是一个新的开放获取平台，旨在解决研究人员在面对 rapidly expanding volume of scientific literature 时保持研究前沿所面临的挑战。我们提供个性化推荐、来自开放获取档案（arXiv、bioRxiv等）的持续更新、可视化论文摘要、语义搜索以及一系列工具来优化研究工作流程和促进开放研究访问。该平台的个性化推荐系统基于用户评分进行训练，确保推荐内容符合不同研究人员的兴趣。为了进一步提升用户体验，学者收件箱还提供了一幅科学地图，提供了跨领域研究概览，使用户能够轻松探索特定主题。我们利用这幅地图来解决推荐系统中常见的冷启动问题，并采用主动学习策略，逐步提示用户对其所选论文进行评分，从而使系统能够快速了解用户偏好。我们使用一个包含80万用户评分的新颖数据集评估了推荐系统的质量，并已将该数据集公开，还通过广泛的用户研究进行了评估。更多的信息请参见：this https URL。 

---
# Passive Underwater Acoustic Signal Separation based on Feature Decoupling Dual-path Network 

**Title (ZH)**: 基于特征解耦双路径网络的被动水下声信号分离 

**Authors**: Yucheng Liu, Longyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08371)  

**Abstract**: Signal separation in the passive underwater acoustic domain has heavily relied on deep learning techniques to isolate ship radiated noise. However, the separation networks commonly used in this domain stem from speech separation applications and may not fully consider the unique aspects of underwater acoustics beforehand, such as the influence of different propagation media, signal frequencies and modulation characteristics. This oversight highlights the need for tailored approaches that account for the specific characteristics of underwater sound propagation. This study introduces a novel temporal network designed to separate ship radiated noise by employing a dual-path model and a feature decoupling approach. The mixed signals' features are transformed into a space where they exhibit greater independence, with each dimension's significance decoupled. Subsequently, a fusion of local and global attention mechanisms is employed in the separation layer. Extensive comparisons showcase the effectiveness of this method when compared to other prevalent network models, as evidenced by its performance in the ShipsEar and DeepShip datasets. 

**Abstract (ZH)**: 基于深度学习的被动水下声学信号分离方法在船舶辐射噪声隔离中的应用：一种考虑水下声学特性的新型时域网络 

---
# Kernel-Level Energy-Efficient Neural Architecture Search for Tabular Dataset 

**Title (ZH)**: 内核级能源高效神经架构搜索用于表格数据集 

**Authors**: Hoang-Loc La, Phuong Hoai Ha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08359)  

**Abstract**: Many studies estimate energy consumption using proxy metrics like memory usage, FLOPs, and inference latency, with the assumption that reducing these metrics will also lower energy consumption in neural networks. This paper, however, takes a different approach by introducing an energy-efficient Neural Architecture Search (NAS) method that directly focuses on identifying architectures that minimize energy consumption while maintaining acceptable accuracy. Unlike previous methods that primarily target vision and language tasks, the approach proposed here specifically addresses tabular datasets. Remarkably, the optimal architecture suggested by this method can reduce energy consumption by up to 92% compared to architectures recommended by conventional NAS. 

**Abstract (ZH)**: 一种直接针对最小化能量消耗的节能神经架构搜索方法：以表格数据集为例 

---
# Entropic bounds for conditionally Gaussian vectors and applications to neural networks 

**Title (ZH)**: 熵界条件高斯向量及其在神经网络中的应用 

**Authors**: Lucia Celli, Giovanni Peccati  

**Link**: [PDF](https://arxiv.org/pdf/2504.08335)  

**Abstract**: Using entropic inequalities from information theory, we provide new bounds on the total variation and 2-Wasserstein distances between a conditionally Gaussian law and a Gaussian law with invertible covariance matrix. We apply our results to quantify the speed of convergence to Gaussian of a randomly initialized fully connected neural network and its derivatives - evaluated in a finite number of inputs - when the initialization is Gaussian and the sizes of the inner layers diverge to infinity. Our results require mild assumptions on the activation function, and allow one to recover optimal rates of convergence in a variety of distances, thus improving and extending the findings of Basteri and Trevisan (2023), Favaro et al. (2023), Trevisan (2024) and Apollonio et al. (2024). One of our main tools are the quantitative cumulant estimates established in Hanin (2024). As an illustration, we apply our results to bound the total variation distance between the Bayesian posterior law of the neural network and its derivatives, and the posterior law of the corresponding Gaussian limit: this yields quantitative versions of a posterior CLT by Hron et al. (2022), and extends several estimates by Trevisan (2024) to the total variation metric. 

**Abstract (ZH)**: 利用信息论中的熵不等式，我们提供了条件高斯分布与具有可逆协方差矩阵的高斯分布之间的总变差和2- Wasserstein距离的新界线。我们将结果应用于量化随机初始化的全连接神经网络及其在有限输入点上的导数在初始化为高斯分布且内部层尺寸趋于无穷大时向高斯分布的收敛速度。我们的结果仅对激活函数有轻微的假设，并允许我们在多种距离上恢复最优的收敛率，从而改进并拓展了Basteri和Trevisan（2023）、Favaro等人（2023）、Trevisan（2024）和Apollonio等人（2024）的研究发现。我们主要工具之一是Hanin（2024）建立的量化累积量估计。作为示例，我们将结果应用于界定制神经网络及其导数的贝叶斯后验分布与对应的高斯极限后验分布之间的总变差距离，从而得到了Hron等人（2022）后验中心极限定理的量化版本，并将Trevisan（2024）的某些估计拓展到总变差度量。 

---
# SortBench: Benchmarking LLMs based on their ability to sort lists 

**Title (ZH)**: SortBench: 根据列表排序能力评估LLM模型 

**Authors**: Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2504.08312)  

**Abstract**: Sorting is a tedious but simple task for human intelligence and can be solved fairly easily algorithmically. However, for Large Language Models (LLMs) this task is surprisingly hard, as some properties of sorting are among known weaknesses of LLMs: being faithful to the input data, logical comparisons between values, and strictly differentiating between syntax (used for sorting) and semantics (typically learned by embeddings). Within this paper, we describe the new SortBench benchmark for LLMs that comes with different difficulties and that can be easily scaled in terms of difficulty. We apply this benchmark to seven state-of-the-art LLMs, including current test-time reasoning models. Our results show that while the o3-mini model is very capable at sorting in general, even this can be fooled if strings are defined to mix syntactical and semantical aspects, e.g., by asking to sort numbers written-out as word. Furthermore, all models have problems with the faithfulness to the input of long lists, i.e., they drop items and add new ones. Our results also show that test-time reasoning has a tendency to overthink problems which leads to performance degradation. Finally, models without test-time reasoning like GPT-4o are not much worse than reasoning models. 

**Abstract (ZH)**: 大语言模型的排序任务：SortBench基准及其挑战 

---
# Large language models could be rote learners 

**Title (ZH)**: 大型语言模型可能是机械记忆的学习者 

**Authors**: Yuyang Xu, Renjun Hu, Haochao Ying, Jian Wu, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.08300)  

**Abstract**: Multiple-choice question (MCQ) benchmarks are widely used for evaluating Large Language Models (LLMs), yet their reliability is undermined by benchmark contamination. In this study, we reframe contamination as an inherent aspect of learning and seek to disentangle genuine capability acquisition from superficial memorization in LLM evaluation. First, by analyzing model performance under different memorization conditions, we uncover a counterintuitive trend: LLMs perform worse on memorized MCQs than on non-memorized ones, indicating the coexistence of two distinct learning phenomena, i.e., rote memorization and genuine capability learning. To disentangle them, we propose TrinEval, a novel evaluation framework that reformulates MCQs into an alternative trinity format, reducing memorization while preserving knowledge assessment. Experiments validate TrinEval's effectiveness in reformulation, and its evaluation reveals that common LLMs may memorize by rote 20.5% of knowledge points (in MMLU on average). 

**Abstract (ZH)**: Multiple-choice question benchmarks的可靠性受到基准污染的威胁：一种重新审视大型语言模型评估的新框架 

---
# ELSA: A Style Aligned Dataset for Emotionally Intelligent Language Generation 

**Title (ZH)**: ELSA：一种情感对齐的数据集，用于情感智能语言生成 

**Authors**: Vishal Gandhi, Sagar Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08281)  

**Abstract**: Advancements in emotion aware language processing increasingly shape vital NLP applications ranging from conversational AI and affective computing to computational psychology and creative content generation. Existing emotion datasets either lack emotional granularity or fail to capture necessary stylistic diversity, limiting the advancement of effective emotion conditioned text generation systems. Seeking to bridge this crucial gap between granularity and style diversity, this paper introduces a novel systematically constructed dataset named ELSA Emotion and Language Style Alignment Dataset leveraging fine grained emotion taxonomies adapted from existing sources such as dair ai emotion dataset and GoEmotions taxonomy. This dataset comprises multiple emotionally nuanced variations of original sentences regenerated across distinct contextual styles such as conversational, formal, poetic, and narrative, using advanced Large Language Models LLMs. Rigorous computational evaluation using metrics such as perplexity, embedding variance, readability, lexical diversity, and semantic coherence measures validates the datasets emotional authenticity, linguistic fluency, and textual diversity. Comprehensive metric analyses affirm its potential to support deeper explorations into emotion conditioned style adaptive text generation. By enabling precision tuned emotionally nuanced language modeling, our dataset creates fertile ground for research on fine grained emotional control, prompt driven explanation, interpretability, and style adaptive expressive language generation with LLMs. 

**Abstract (ZH)**: 情感感知语言处理的进展日益塑造了从对话AI和情感计算到计算心理学和创意内容生成等关键的NLP应用。现有的情感数据集要么缺乏情感细腻度，要么未能捕捉必要的风格多样性，限制了有效的情感条件文本生成系统的发展。为弥合细腻度与风格多样性之间的关键差距，本文介绍了一种名为ELSA情感与语言风格对齐数据集的新颖系统构建数据集，该数据集利用从现有来源（如dair ai情感数据集和GoEmotions分类法）改编的精细情感分类法。该数据集包含使用先进大型语言模型（LLMs）跨越不同语境风格（如对话、正式、诗歌和叙事）重新生成的多种情感细微差异的原始句子变体。使用诸如困惑度、嵌入变化、可读性、词汇多样性以及语义一致性等严格计算评估指标，验证了数据集的情感真实性、语言流畅性和文本多样性。全面的度量分析证实其支持对情感条件下的风格适应性文本生成更深入探索的潜力。通过实现精准调优的情感细腻语言建模，我们的数据集为基于LLMs的情感细粒度控制、提示驱动解释、可解释性和风格适应性表达语言生成的研究奠定了基础。 

---
# CoProSketch: Controllable and Progressive Sketch Generation with Diffusion Model 

**Title (ZH)**: CoProSketch: 可控制与渐进的素描生成方法基于扩散模型 

**Authors**: Ruohao Zhan, Yijin Li, Yisheng He, Shuo Chen, Yichen Shen, Xinyu Chen, Zilong Dong, Zhaoyang Huang, Guofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08259)  

**Abstract**: Sketches serve as fundamental blueprints in artistic creation because sketch editing is easier and more intuitive than pixel-level RGB image editing for painting artists, yet sketch generation remains unexplored despite advancements in generative models. We propose a novel framework CoProSketch, providing prominent controllability and details for sketch generation with diffusion models. A straightforward method is fine-tuning a pretrained image generation diffusion model with binarized sketch images. However, we find that the diffusion models fail to generate clear binary images, which makes the produced sketches chaotic. We thus propose to represent the sketches by unsigned distance field (UDF), which is continuous and can be easily decoded to sketches through a lightweight network. With CoProSketch, users generate a rough sketch from a bounding box and a text prompt. The rough sketch can be manually edited and fed back into the model for iterative refinement and will be decoded to a detailed sketch as the final result. Additionally, we curate the first large-scale text-sketch paired dataset as the training data. Experiments demonstrate superior semantic consistency and controllability over baselines, offering a practical solution for integrating user feedback into generative workflows. 

**Abstract (ZH)**: CoProSketch：基于扩散模型的可控草图生成框架 

---
# Accelerating Multi-Objective Collaborative Optimization of Doped Thermoelectric Materials via Artificial Intelligence 

**Title (ZH)**: 通过人工智能加速掺杂热电材料多目标协作优化 

**Authors**: Yuxuan Zeng, Wenhao Xie, Wei Cao, Tan Peng, Yue Hou, Ziyu Wang, Jing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08258)  

**Abstract**: The thermoelectric performance of materials exhibits complex nonlinear dependencies on both elemental types and their proportions, rendering traditional trial-and-error approaches inefficient and time-consuming for material discovery. In this work, we present a deep learning model capable of accurately predicting thermoelectric properties of doped materials directly from their chemical formulas, achieving state-of-the-art performance. To enhance interpretability, we further incorporate sensitivity analysis techniques to elucidate how physical descriptors affect the thermoelectric figure of merit (zT). Moreover, we establish a coupled framework that integrates a surrogate model with a multi-objective genetic algorithm to efficiently explore the vast compositional space for high-performance candidates. Experimental validation confirms the discovery of a novel thermoelectric material with superior $zT$ values in the medium-temperature regime. 

**Abstract (ZH)**: 深学习模型在化学式指导下直接预测掺杂材料热电性能的研究：高效探索高性能候选材料的耦合框架验证 

---
# Bayesian Reasoning Enabled by Spin-Orbit Torque Magnetic Tunnel Junctions 

**Title (ZH)**: 由自旋轨道矩磁隧道结-enabled的贝叶斯推理 

**Authors**: Yingqian Xu, Xiaohan Li, Caihua Wan, Ran Zhang, Bin He, Shiqiang Liu, Jihao Xia, Dehao Kong, Shilong Xiong, Guoqiang Yu, Xiufeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.08257)  

**Abstract**: Bayesian networks play an increasingly important role in data mining, inference, and reasoning with the rapid development of artificial intelligence. In this paper, we present proof-of-concept experiments demonstrating the use of spin-orbit torque magnetic tunnel junctions (SOT-MTJs) in Bayesian network reasoning. Not only can the target probability distribution function (PDF) of a Bayesian network be precisely formulated by a conditional probability table as usual but also quantitatively parameterized by a probabilistic forward propagating neuron network. Moreover, the parameters of the network can also approach the optimum through a simple point-by point training algorithm, by leveraging which we do not need to memorize all historical data nor statistically summarize conditional probabilities behind them, significantly improving storage efficiency and economizing data pretreatment. Furthermore, we developed a simple medical diagnostic system using the SOT-MTJ as a random number generator and sampler, showcasing the application of SOT-MTJ-based Bayesian reasoning. This SOT-MTJ-based Bayesian reasoning shows great promise in the field of artificial probabilistic neural network, broadening the scope of spintronic device applications and providing an efficient and low-storage solution for complex reasoning tasks. 

**Abstract (ZH)**: 基于自旋轨道扭矩磁隧道结的贝叶斯网络推理：原理验证与应用探索 

---
# RAG-VR: Leveraging Retrieval-Augmented Generation for 3D Question Answering in VR Environments 

**Title (ZH)**: RAG-VR：利用检索增强生成技术进行VR环境中的3D问答 

**Authors**: Shiyi Ding, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08256)  

**Abstract**: Recent advances in large language models (LLMs) provide new opportunities for context understanding in virtual reality (VR). However, VR contexts are often highly localized and personalized, limiting the effectiveness of general-purpose LLMs. To address this challenge, we present RAG-VR, the first 3D question-answering system for VR that incorporates retrieval-augmented generation (RAG), which augments an LLM with external knowledge retrieved from a localized knowledge database to improve the answer quality. RAG-VR includes a pipeline for extracting comprehensive knowledge about virtual environments and user conditions for accurate answer generation. To ensure efficient retrieval, RAG-VR offloads the retrieval process to a nearby edge server and uses only essential information during retrieval. Moreover, we train the retriever to effectively distinguish among relevant, irrelevant, and hard-to-differentiate information in relation to questions. RAG-VR improves answer accuracy by 17.9%-41.8% and reduces end-to-end latency by 34.5%-47.3% compared with two baseline systems. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）为虚拟现实（VR）中的上下文理解提供了新的机遇。然而，VR上下文往往高度局部化和个性化，限制了通用大型语言模型的效果。为应对这一挑战，我们提出了RAG-VR，这是第一个结合检索增强生成（RAG）技术的3D问答系统，该系统通过从局部知识数据库中检索外部知识来增强大型语言模型，以提高答案质量。RAG-VR包括一个提取虚拟环境和用户条件全面知识的管道，以实现准确的回答生成。为了确保高效的检索，RAG-VR将检索过程卸载到附近的边缘服务器，并仅在检索过程中使用必要的信息。此外，我们训练检索器有效地区分与问题相关的、无关的以及难以区分的信息。与两个基线系统相比，RAG-VR将答案准确性提高了17.9%-41.8%，并将端到端延迟降低了34.5%-47.3%。 

---
# Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices 

**Title (ZH)**: Jupiter：在边缘设备上快速且资源高效的生成型LLM协作推理 

**Authors**: Shengyuan Ye, Bei Ouyang, Liekang Zeng, Tianyi Qian, Xiaowen Chu, Jian Tang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08242)  

**Abstract**: Generative large language models (LLMs) have garnered significant attention due to their exceptional capabilities in various AI tasks. Traditionally deployed in cloud datacenters, LLMs are now increasingly moving towards more accessible edge platforms to protect sensitive user data and ensure privacy preservation. The limited computational resources of individual edge devices, however, can result in excessively prolonged inference latency and overwhelmed memory usage. While existing research has explored collaborative edge computing to break the resource wall of individual devices, these solutions yet suffer from massive communication overhead and under-utilization of edge resources. Furthermore, they focus exclusively on optimizing the prefill phase, neglecting the crucial autoregressive decoding phase for generative LLMs. To address that, we propose Jupiter, a fast, scalable, and resource-efficient collaborative edge AI system for generative LLM inference. Jupiter introduces a flexible pipelined architecture as a principle and differentiates its system design according to the differentiated characteristics of the prefill and decoding phases. For prefill phase, Jupiter submits a novel intra-sequence pipeline parallelism and develops a meticulous parallelism planning strategy to maximize resource efficiency; For decoding, Jupiter devises an effective outline-based pipeline parallel decoding mechanism combined with speculative decoding, which further magnifies inference acceleration. Extensive evaluation based on realistic implementation demonstrates that Jupiter remarkably outperforms state-of-the-art approaches under various edge environment setups, achieving up to 26.1x end-to-end latency reduction while rendering on-par generation quality. 

**Abstract (ZH)**: 生成型大型语言模型（LLMs）因其在各种AI任务中的出色能力而备受关注。传统上部署在云数据中心的LLMs现在越来越多地转向更易于访问的边缘平台，以保护敏感用户数据和确保隐私保护。然而，个体边缘设备有限的计算资源可能导致推理延迟过长和内存使用过载。尽管现有研究探索了协作边缘计算以突破个体设备的资源壁垒，但这些解决方案仍然面临巨大的通信开销和边缘资源利用率低的问题。此外，它们仅专注于优化预填阶段，忽视了生成型LLMs至关重要的自回归解码阶段。为此，我们提出了一种名为Jupiter的快速、可扩展且资源高效的协作边缘AI系统，用于生成型LLM推理。Jupiter引入了灵活的流水线架构作为其设计原则，并根据预填阶段和解码阶段的不同特征差异化其系统设计。对于预填阶段，Jupiter提交了一种新颖的序列内流水线并行性，并开发了精细的并行性规划策略以最大化资源效率；对于解码，Jupiter设计了一种有效的基于大纲的流水线并行解码机制结合推测性解码，从而进一步提升推理加速效果。基于现实部署的广泛评估表明，在各种边缘环境配置下，Jupiter显著优于现有最佳方法，实现端到端延迟最多减少26.1倍，同时保持生成质量相当。 

---
# F$^3$Set: Towards Analyzing Fast, Frequent, and Fine-grained Events from Videos 

**Title (ZH)**: F$^3$Set: 从视频中分析快速、频繁和细粒度事件的方法 

**Authors**: Zhaoyu Liu, Kan Jiang, Murong Ma, Zhe Hou, Yun Lin, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08222)  

**Abstract**: Analyzing Fast, Frequent, and Fine-grained (F$^3$) events presents a significant challenge in video analytics and multi-modal LLMs. Current methods struggle to identify events that satisfy all the F$^3$ criteria with high accuracy due to challenges such as motion blur and subtle visual discrepancies. To advance research in video understanding, we introduce F$^3$Set, a benchmark that consists of video datasets for precise F$^3$ event detection. Datasets in F$^3$Set are characterized by their extensive scale and comprehensive detail, usually encompassing over 1,000 event types with precise timestamps and supporting multi-level granularity. Currently, F$^3$Set contains several sports datasets, and this framework may be extended to other applications as well. We evaluated popular temporal action understanding methods on F$^3$Set, revealing substantial challenges for existing techniques. Additionally, we propose a new method, F$^3$ED, for F$^3$ event detections, achieving superior performance. The dataset, model, and benchmark code are available at this https URL. 

**Abstract (ZH)**: 分析快速、频繁和细粒度（F$^3$）事件在视频分析和多模态LLM中的挑战 

---
# LLM for Comparative Narrative Analysis 

**Title (ZH)**: 大规模语言模型在叙事比较分析中的应用 

**Authors**: Leo Kampen, Carlos Rabat Villarreal, Louis Yu, Santu Karmaker, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08211)  

**Abstract**: In this paper, we conducted a Multi-Perspective Comparative Narrative Analysis (CNA) on three prominent LLMs: GPT-3.5, PaLM2, and Llama2. We applied identical prompts and evaluated their outputs on specific tasks, ensuring an equitable and unbiased comparison between various LLMs. Our study revealed that the three LLMs generated divergent responses to the same prompt, indicating notable discrepancies in their ability to comprehend and analyze the given task. Human evaluation was used as the gold standard, evaluating four perspectives to analyze differences in LLM performance. 

**Abstract (ZH)**: 本文对三款 prominant LLM（GPT-3.5, PaLM2, 和 Llama2）进行了多视角比较叙事分析（CNA），采用了相同的提示并评估了它们在特定任务上的输出，确保了对各种LLM进行公平和无偏见的比较。研究结果显示，三款LLM对同一提示的响应存在显著差异，表明它们在理解和分析给定任务方面的能力存在明显差异。人类评估被用作黄金标准，从四个视角分析了LLM性能的差异。 

---
# Optimizing Power Grid Topologies with Reinforcement Learning: A Survey of Methods and Challenges 

**Title (ZH)**: 使用强化学习优化电力网络拓扑结构：方法与挑战综述 

**Authors**: Erica van der Sar, Alessandro Zocca, Sandjai Bhulai  

**Link**: [PDF](https://arxiv.org/pdf/2504.08210)  

**Abstract**: Power grid operation is becoming increasingly complex due to the rising integration of renewable energy sources and the need for more adaptive control strategies. Reinforcement Learning (RL) has emerged as a promising approach to power network control (PNC), offering the potential to enhance decision-making in dynamic and uncertain environments. The Learning To Run a Power Network (L2RPN) competitions have played a key role in accelerating research by providing standardized benchmarks and problem formulations, leading to rapid advancements in RL-based methods. This survey provides a comprehensive and structured overview of RL applications for power grid topology optimization, categorizing existing techniques, highlighting key design choices, and identifying gaps in current research. Additionally, we present a comparative numerical study evaluating the impact of commonly applied RL-based methods, offering insights into their practical effectiveness. By consolidating existing research and outlining open challenges, this survey aims to provide a foundation for future advancements in RL-driven power grid optimization. 

**Abstract (ZH)**: 电力网络运行因可再生能源的集成增加和需要更适应的控制策略而变得越来越复杂。强化学习（RL）已成为电力网络控制（PNC）的一个有前途的方法，有可能在动态和不确定的环境中提高决策能力。L2RPN竞赛通过提供标准化的基准和问题形式，在加速研究方面发挥了关键作用，导致基于RL的方法迅速发展。本文提供了一个全面且结构化的RL在电力网络拓扑优化中的应用综述，对现有技术进行分类，突出关键设计选择，并识别当前研究中的空白。此外，我们还呈现了一个比较数值研究，评估了常用基于RL方法的影响，提供了其实际有效性的见解。通过整合现有研究并概述开放挑战，本文旨在为基于RL的电力网络优化未来进步奠定基础。 

---
# How Good Are Large Language Models for Course Recommendation in MOOCs? 

**Title (ZH)**: 大型语言模型在MOOC课程推荐中的能力如何？ 

**Authors**: Boxuan Ma, Md Akib Zabed Khan, Tianyuan Yang, Agoritsa Polyzou, Shin'ichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08208)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language processing and are increasingly being integrated into recommendation systems. However, their potential in educational recommendation systems has yet to be fully explored. This paper investigates the use of LLMs as a general-purpose recommendation model, leveraging their vast knowledge derived from large-scale corpora for course recommendation tasks. We explore a variety of approaches, ranging from prompt-based methods to more advanced fine-tuning techniques, and compare their performance against traditional recommendation models. Extensive experiments were conducted on a real-world MOOC dataset, evaluating using LLMs as course recommendation systems across key dimensions such as accuracy, diversity, and novelty. Our results demonstrate that LLMs can achieve good performance comparable to traditional models, highlighting their potential to enhance educational recommendation systems. These findings pave the way for further exploration and development of LLM-based approaches in the context of educational recommendations. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理领域取得了显著进展，并越来越多地被集成到推荐系统中。然而，它们在教育推荐系统中的潜力尚未得到充分探索。本文研究了将LLMs作为通用推荐模型的应用，利用其从大规模语料库中获得的广泛知识，进行课程推荐任务。我们探索了从提示方法到更高级的微调技术等多种方法，并将它们的性能与传统推荐模型进行了比较。我们在一个真实的MOOC数据集上进行了广泛的实验，从准确度、多样性和新颖性等多个维度评估了使用LLMs作为课程推荐系统的表现。我们的研究结果表明，LLMs可以实现与传统模型相当的性能，突显了其增强教育推荐系统潜力的可能性。这些发现为探索和开发基于LLM的方法在教育推荐领域的应用铺平了道路。 

---
# DRAFT-ing Architectural Design Decisions using LLMs 

**Title (ZH)**: 使用大型语言模型草图化架构设计决策 

**Authors**: Rudra Dhar, Adyansh Kakran, Amey Karan, Karthik Vaidhyanathan, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2504.08207)  

**Abstract**: Architectural Knowledge Management (AKM) is crucial for software development but remains challenging due to the lack of standardization and high manual effort. Architecture Decision Records (ADRs) provide a structured approach to capture Architecture Design Decisions (ADDs), but their adoption is limited due to the manual effort involved and insufficient tool support. Our previous work has shown that Large Language Models (LLMs) can assist in generating ADDs. However, simply prompting the LLM does not produce quality ADDs. Moreover, using third-party LLMs raises privacy concerns, while self-hosting them poses resource challenges.
To this end, we experimented with different approaches like few-shot, retrieval-augmented generation (RAG) and fine-tuning to enhance LLM's ability to generate ADDs. Our results show that both techniques improve effectiveness. Building on this, we propose Domain Specific Retreival Augumented Few Shot Fine Tuninng, DRAFT, which combines the strengths of all these three approaches for more effective ADD generation. DRAFT operates in two phases: an offline phase that fine-tunes an LLM on generating ADDs augmented with retrieved examples and an online phase that generates ADDs by leveraging retrieved ADRs and the fine-tuned model.
We evaluated DRAFT against existing approaches on a dataset of 4,911 ADRs and various LLMs and analyzed them using automated metrics and human evaluations. Results show DRAFT outperforms all other approaches in effectiveness while maintaining efficiency. Our findings indicate that DRAFT can aid architects in drafting ADDs while addressing privacy and resource constraints. 

**Abstract (ZH)**: 特定领域检索增强少量样本细调方法（DRAFT）：提高架构设计决策生成的有效性和效率 

---
# Neural Encoding and Decoding at Scale 

**Title (ZH)**: 大规模神经编码与解码 

**Authors**: Yizi Zhang, Yanchen Wang, Mehdi Azabou, Alexandre Andre, Zixuan Wang, Hanrui Lyu, International Brain Laboratory, Eva Dyer, Liam Paninski, Cole Hurwitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.08201)  

**Abstract**: Recent work has demonstrated that large-scale, multi-animal models are powerful tools for characterizing the relationship between neural activity and behavior. Current large-scale approaches, however, focus exclusively on either predicting neural activity from behavior (encoding) or predicting behavior from neural activity (decoding), limiting their ability to capture the bidirectional relationship between neural activity and behavior. To bridge this gap, we introduce a multimodal, multi-task model that enables simultaneous Neural Encoding and Decoding at Scale (NEDS). Central to our approach is a novel multi-task-masking strategy, which alternates between neural, behavioral, within-modality, and cross-modality masking. We pretrain our method on the International Brain Laboratory (IBL) repeated site dataset, which includes recordings from 83 animals performing the same visual decision-making task. In comparison to other large-scale models, we demonstrate that NEDS achieves state-of-the-art performance for both encoding and decoding when pretrained on multi-animal data and then fine-tuned on new animals. Surprisingly, NEDS's learned embeddings exhibit emergent properties: even without explicit training, they are highly predictive of the brain regions in each recording. Altogether, our approach is a step towards a foundation model of the brain that enables seamless translation between neural activity and behavior. 

**Abstract (ZH)**: Recent Work Has Demonstrated that Large-Scale, Multi-Animal Models are Powerful Tools for Characterizing the Relationship between Neural Activity and Behavior: Bridging the Gap with Neural Encoding and Decoding at Scale (NEDS) 

---
# Influential Bandits: Pulling an Arm May Change the Environment 

**Title (ZH)**: 有影响力的探索者：拉一个臂可能改变环境 

**Authors**: Ryoma Sato, Shinji Ito  

**Link**: [PDF](https://arxiv.org/pdf/2504.08200)  

**Abstract**: While classical formulations of multi-armed bandit problems assume that each arm's reward is independent and stationary, real-world applications often involve non-stationary environments and interdependencies between arms. In particular, selecting one arm may influence the future rewards of other arms, a scenario not adequately captured by existing models such as rotting bandits or restless bandits. To address this limitation, we propose the influential bandit problem, which models inter-arm interactions through an unknown, symmetric, positive semi-definite interaction matrix that governs the dynamics of arm losses. We formally define this problem and establish two regret lower bounds, including a superlinear $\Omega(T^2 / \log^2 T)$ bound for the standard UCB algorithm and an algorithm-independent $\Omega(T)$ bound, which highlight the inherent difficulty of the setting. We then introduce a new algorithm based on a lower confidence bound (LCB) estimator tailored to the structure of the loss dynamics. Under mild assumptions, our algorithm achieves a regret of $O(KT \log T)$, which is nearly optimal in terms of its dependence on the time horizon. The algorithm is simple to implement and computationally efficient. Empirical evaluations on both synthetic and real-world datasets demonstrate the presence of inter-arm influence and confirm the superior performance of our method compared to conventional bandit algorithms. 

**Abstract (ZH)**: 具有影响关系的bandit问题：基于损失动态结构的下置信界算法 

---
# Graph Based Deep Reinforcement Learning Aided by Transformers for Multi-Agent Cooperation 

**Title (ZH)**: 基于图的深度强化学习辅助变换器在多智能体合作中的应用 

**Authors**: Michael Elrod, Niloufar Mehrabi, Rahul Amin, Manveen Kaur, Long Cheng, Jim Martin, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08195)  

**Abstract**: Mission planning for a fleet of cooperative autonomous drones in applications that involve serving distributed target points, such as disaster response, environmental monitoring, and surveillance, is challenging, especially under partial observability, limited communication range, and uncertain environments. Traditional path-planning algorithms struggle in these scenarios, particularly when prior information is not available. To address these challenges, we propose a novel framework that integrates Graph Neural Networks (GNNs), Deep Reinforcement Learning (DRL), and transformer-based mechanisms for enhanced multi-agent coordination and collective task execution. Our approach leverages GNNs to model agent-agent and agent-goal interactions through adaptive graph construction, enabling efficient information aggregation and decision-making under constrained communication. A transformer-based message-passing mechanism, augmented with edge-feature-enhanced attention, captures complex interaction patterns, while a Double Deep Q-Network (Double DQN) with prioritized experience replay optimizes agent policies in partially observable environments. This integration is carefully designed to address specific requirements of multi-agent navigation, such as scalability, adaptability, and efficient task execution. Experimental results demonstrate superior performance, with 90% service provisioning and 100% grid coverage (node discovery), while reducing the average steps per episode to 200, compared to 600 for benchmark methods such as particle swarm optimization (PSO), greedy algorithms and DQN. 

**Abstract (ZH)**: 适用于分布式目标点应用（如灾害响应、环境监测和 surveillance）的编队合作自主无人机群任务规划，在部分可观测性、有限通信范围和不确定环境下的挑战，传统路径规划算法难以应对，尤其是在缺乏先验信息的情况下。为应对这些挑战，我们提出了一种将图神经网络（GNNs）、深度强化学习（DRL）和基于变换器的机制相结合的新型框架，以增强多智能体协调和集体任务执行。该方法利用GNNs通过自适应图构建来建模智能体-智能体和智能体-目标交互，从而在受限通信条件下实现高效的信息聚合和决策。基于变换器的消息传递机制，结合边特征增强的注意力机制捕获复杂的交互模式，而双重深度Q网络（Double DQN）结合优先经验回放则在部分可观测环境中优化智能体策略。该集成设计特别考虑了多智能体导航的具体要求，如扩展性、适应性和高效的任务执行。实验结果表明，该方法在服务提供比例达到90%、网格覆盖（节点发现）为100%的同时，将每集isode的平均步骤数降至200，而基准方法，如粒子群优化（PSO）、贪婪算法和DQN，这一数字为600。 

---
# SAEs $\textit{Can}$ Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs 

**Title (ZH)**: SAEs 可以提升未学习能力：动态稀疏自编码器在大规模语言模型中精确未学习的界限 

**Authors**: Aashiq Muhamed, Jacopo Bonato, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.08192)  

**Abstract**: Machine unlearning is a promising approach to improve LLM safety by removing unwanted knowledge from the model. However, prevailing gradient-based unlearning methods suffer from issues such as high computational costs, hyperparameter instability, poor sequential unlearning capability, vulnerability to relearning attacks, low data efficiency, and lack of interpretability. While Sparse Autoencoders are well-suited to improve these aspects by enabling targeted activation-based unlearning, prior approaches underperform gradient-based methods. This work demonstrates that, contrary to these earlier findings, SAEs can significantly improve unlearning when employed dynamically. We introduce $\textbf{Dynamic DAE Guardrails}$ (DSG), a novel method for precision unlearning that leverages principled feature selection and a dynamic classifier. Our experiments show DSG substantially outperforms leading unlearning methods, achieving superior forget-utility trade-offs. DSG addresses key drawbacks of gradient-based approaches for unlearning -- offering enhanced computational efficiency and stability, robust performance in sequential unlearning, stronger resistance to relearning attacks, better data efficiency including zero-shot settings, and more interpretable unlearning. 

**Abstract (ZH)**: 动态稀疏自编码器-guardrails（DSG）：一种精确遗忘的新方法 

---
# TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation 

**Title (ZH)**: TokenMotion：通过token去纠缠实现的人本驱动视频生成中的解耦运动控制 

**Authors**: Ruineng Li, Daitao Xing, Huiming Sun, Yuanzhou Ha, Jinglin Shen, Chiuman Ho  

**Link**: [PDF](https://arxiv.org/pdf/2504.08181)  

**Abstract**: Human-centric motion control in video generation remains a critical challenge, particularly when jointly controlling camera movements and human poses in scenarios like the iconic Grammy Glambot moment. While recent video diffusion models have made significant progress, existing approaches struggle with limited motion representations and inadequate integration of camera and human motion controls. In this work, we present TokenMotion, the first DiT-based video diffusion framework that enables fine-grained control over camera motion, human motion, and their joint interaction. We represent camera trajectories and human poses as spatio-temporal tokens to enable local control granularity. Our approach introduces a unified modeling framework utilizing a decouple-and-fuse strategy, bridged by a human-aware dynamic mask that effectively handles the spatially-and-temporally varying nature of combined motion signals. Through extensive experiments, we demonstrate TokenMotion's effectiveness across both text-to-video and image-to-video paradigms, consistently outperforming current state-of-the-art methods in human-centric motion control tasks. Our work represents a significant advancement in controllable video generation, with particular relevance for creative production applications. 

**Abstract (ZH)**: 基于人类中心的视频生成中的运动控制仍然是一个关键挑战，尤其是在同时控制相机运动和人类姿态如克利普顿格莱美吉祥物时刻等场景中。尽管近期的视频扩散模型取得了显著进展，现有的方法仍然难以处理有限的运动表示和相机与人类运动控制的不充分集成。在本文中，我们提出了TokenMotion，这是一个基于DiT的视频扩散框架，能够精细控制相机运动、人类运动及其联合交互。我们将相机轨迹和人类姿态表示为时空令牌，以实现局部控制粒度。我们的方法采用分离融合策略，并通过一种人类感知的动态掩码来统一建模，有效地处理联合运动信号的空间和时间变化特性。通过广泛的实验，我们展示了TokenMotion在文本到视频和图像到视频范式中的有效性，并且在人类中心的运动控制任务中始终优于当前最先进的方法。我们的工作代表了可控视频生成的重要进展，特别适用于创意生产应用。 

---
# SynthFM: Training Modality-agnostic Foundation Models for Medical Image Segmentation without Real Medical Data 

**Title (ZH)**: SynthFM: 训练跨模态基础模型以在无需真实医疗数据的情况下进行医学图像分割 

**Authors**: Sourya Sengupta, Satrajit Chakrabarty, Keerthi Sravan Ravi, Gopal Avinash, Ravi Soni  

**Link**: [PDF](https://arxiv.org/pdf/2504.08177)  

**Abstract**: Foundation models like the Segment Anything Model (SAM) excel in zero-shot segmentation for natural images but struggle with medical image segmentation due to differences in texture, contrast, and noise. Annotating medical images is costly and requires domain expertise, limiting large-scale annotated data availability. To address this, we propose SynthFM, a synthetic data generation framework that mimics the complexities of medical images, enabling foundation models to adapt without real medical data. Using SAM's pretrained encoder and training the decoder from scratch on SynthFM's dataset, we evaluated our method on 11 anatomical structures across 9 datasets (CT, MRI, and Ultrasound). SynthFM outperformed zero-shot baselines like SAM and MedSAM, achieving superior results under different prompt settings and on out-of-distribution datasets. 

**Abstract (ZH)**: 基于SynthFM的合成数据生成框架在医学图像分割中的应用：克服自然图像与医学图像分割差异，无需大量标注数据 

---
# On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction 

**Title (ZH)**: 基于深度层次集成网络的广告转换率预测实践研究 

**Authors**: Jinfeng Zhuang, Yinrui Li, Runze Su, Ke Xu, Zhixuan Shao, Kungang Li, Ling Leng, Han Sun, Meng Qi, Yixiong Meng, Yang Tang, Zhifang Liu, Qifei Shen, Aayush Mudgal  

**Link**: [PDF](https://arxiv.org/pdf/2504.08169)  

**Abstract**: The predictions of click through rate (CTR) and conversion rate (CVR) play a crucial role in the success of ad-recommendation systems. A Deep Hierarchical Ensemble Network (DHEN) has been proposed to integrate multiple feature crossing modules and has achieved great success in CTR prediction. However, its performance for CVR prediction is unclear in the conversion ads setting, where an ad bids for the probability of a user's off-site actions on a third party website or app, including purchase, add to cart, sign up, etc. A few challenges in DHEN: 1) What feature-crossing modules (MLP, DCN, Transformer, to name a few) should be included in DHEN? 2) How deep and wide should DHEN be to achieve the best trade-off between efficiency and efficacy? 3) What hyper-parameters to choose in each feature-crossing module? Orthogonal to the model architecture, the input personalization features also significantly impact model performance with a high degree of freedom. In this paper, we attack this problem and present our contributions biased to the applied data science side, including:
First, we propose a multitask learning framework with DHEN as the single backbone model architecture to predict all CVR tasks, with a detailed study on how to make DHEN work effectively in practice; Second, we build both on-site real-time user behavior sequences and off-site conversion event sequences for CVR prediction purposes, and conduct ablation study on its importance; Last but not least, we propose a self-supervised auxiliary loss to predict future actions in the input sequence, to help resolve the label sparseness issue in CVR prediction.
Our method achieves state-of-the-art performance compared to previous single feature crossing modules with pre-trained user personalization features. 

**Abstract (ZH)**: 深度层次集成网络在点击率和转化率预测中的多任务学习框架及贡献 

---
# Rethinking the Foundations for Continual Reinforcement Learning 

**Title (ZH)**: 重思连续强化学习的基础 

**Authors**: Michael Bowling, Esraa Elelimy  

**Link**: [PDF](https://arxiv.org/pdf/2504.08161)  

**Abstract**: Algorithms and approaches for continual reinforcement learning have gained increasing attention. Much of this early progress rests on the foundations and standard practices of traditional reinforcement learning, without questioning if they are well-suited to the challenges of continual learning agents. We suggest that many core foundations of traditional RL are, in fact, antithetical to the goals of continual reinforcement learning. We enumerate four such foundations: the Markov decision process formalism, a focus on optimal policies, the expected sum of rewards as the primary evaluation metric, and episodic benchmark environments that embrace the other three foundations. Shedding such sacredly held and taught concepts is not easy. They are self-reinforcing in that each foundation depends upon and holds up the others, making it hard to rethink each in isolation. We propose an alternative set of all four foundations that are better suited to the continual learning setting. We hope to spur on others in rethinking the traditional foundations, proposing and critiquing alternatives, and developing new algorithms and approaches enabled by better-suited foundations. 

**Abstract (ZH)**: 持续增强学习的算法与方法引起了越来越多的关注。尽管早期的进步在很大程度上建立在传统增强学习的基础和标准实践之上，但并未质疑这些方法是否适合持续学习代理所面临的挑战。我们认为，传统RL的许多核心基础实际上与持续增强学习的目标相对立。我们列出了四个这样的基础：马尔可夫决策过程的形式化表述、对最优策略的关注、预期奖励之和作为主要评估指标，以及采用其他三个基础的阶段性基准环境。放弃这些被神圣化的概念并不容易。这些基础相辅相成，互相支撑，使得单独重新思考每个基础变得困难。我们提出了一套更适合持续学习环境的四个方面的新基础。我们希望激励其他人重新思考传统的基础，提出并批判替代方案，以及开发由更合适的基础支撑的新算法与方法。 

---
# Benchmarking Suite for Synthetic Aperture Radar Imagery Anomaly Detection (SARIAD) Algorithms 

**Title (ZH)**: 合成孔径雷达图像异常检测算法基准测试套件（SARIAD） 

**Authors**: Lucian Chauvina, Somil Guptac, Angelina Ibarrac, Joshua Peeples  

**Link**: [PDF](https://arxiv.org/pdf/2504.08115)  

**Abstract**: Anomaly detection is a key research challenge in computer vision and machine learning with applications in many fields from quality control to radar imaging. In radar imaging, specifically synthetic aperture radar (SAR), anomaly detection can be used for the classification, detection, and segmentation of objects of interest. However, there is no method for developing and benchmarking these methods on SAR imagery. To address this issue, we introduce SAR imagery anomaly detection (SARIAD). In conjunction with Anomalib, a deep-learning library for anomaly detection, SARIAD provides a comprehensive suite of algorithms and datasets for assessing and developing anomaly detection approaches on SAR imagery. SARIAD specifically integrates multiple SAR datasets along with tools to effectively apply various anomaly detection algorithms to SAR imagery. Several anomaly detection metrics and visualizations are available. Overall, SARIAD acts as a central package for benchmarking SAR models and datasets to allow for reproducible research in the field of anomaly detection in SAR imagery. This package is publicly available: this https URL. 

**Abstract (ZH)**: SAR成像异常检测（SARIAD）：一种全面的异常检测算法和数据集套件 

---
# Geneshift: Impact of different scenario shift on Jailbreaking LLM 

**Title (ZH)**: Geneshift: 不同场景转变对破解LLM的影响 

**Authors**: Tianyi Wu, Zhiwei Xue, Yue Liu, Jiaheng Zhang, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08104)  

**Abstract**: Jailbreak attacks, which aim to cause LLMs to perform unrestricted behaviors, have become a critical and challenging direction in AI safety. Despite achieving the promising attack success rate using dictionary-based evaluation, existing jailbreak attack methods fail to output detailed contents to satisfy the harmful request, leading to poor performance on GPT-based evaluation. To this end, we propose a black-box jailbreak attack termed GeneShift, by using a genetic algorithm to optimize the scenario shifts. Firstly, we observe that the malicious queries perform optimally under different scenario shifts. Based on it, we develop a genetic algorithm to evolve and select the hybrid of scenario shifts. It guides our method to elicit detailed and actionable harmful responses while keeping the seemingly benign facade, improving stealthiness. Extensive experiments demonstrate the superiority of GeneShift. Notably, GeneShift increases the jailbreak success rate from 0% to 60% when direct prompting alone would fail. 

**Abstract (ZH)**: 基因突变：一种基于遗传算法的黑盒 Jailbreak 攻击 

---
# Multi-view autoencoders for Fake News Detection 

**Title (ZH)**: 多视图自编码器在虚假新闻检测中的应用 

**Authors**: Ingryd V. S. T. Pereira, George D. C. Cavalcanti, Rafael M. O. Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2504.08102)  

**Abstract**: Given the volume and speed at which fake news spreads across social media, automatic fake news detection has become a highly important task. However, this task presents several challenges, including extracting textual features that contain relevant information about fake news. Research about fake news detection shows that no single feature extraction technique consistently outperforms the others across all scenarios. Nevertheless, different feature extraction techniques can provide complementary information about the textual data and enable a more comprehensive representation of the content. This paper proposes using multi-view autoencoders to generate a joint feature representation for fake news detection by integrating several feature extraction techniques commonly used in the literature. Experiments on fake news datasets show a significant improvement in classification performance compared to individual views (feature representations). We also observed that selecting a subset of the views instead of composing a latent space with all the views can be advantageous in terms of accuracy and computational effort. For further details, including source codes, figures, and datasets, please refer to the project's repository: this https URL. 

**Abstract (ZH)**: 基于社交媒体上传播速度和规模的虚假新闻，自动虚假新闻检测已成为一项极其重要的任务。然而，这一任务面临诸多挑战，包括提取包含虚假新闻相关信息的文本特征。关于虚假新闻检测的研究表明，没有一种特征提取技术能在所有场景中持续优于其他技术。尽管如此，不同的特征提取技术可以提供互补的文本信息，并能够更全面地表示内容。本文提出使用多视图自动编码器生成虚假新闻检测的联合特征表示，通过整合文献中常用的各种特征提取技术。实验结果显示，与单独的视图（特征表示）相比，这种方法在分类性能上取得了显著的提升。我们还发现，而不是使用所有视图组成潜在空间，在准确性和计算成本方面，选择视图的子集可能是有利的。更多详情，包括源代码、图表和数据集，请参见项目的仓库：this https URL。 

---
# Cellular Development Follows the Path of Minimum Action 

**Title (ZH)**: 细胞发育遵循最小作用原理 

**Authors**: Rohola Zandie, Farhan Khodaee, Yufan Xia, Elazer R. Edelman  

**Link**: [PDF](https://arxiv.org/pdf/2504.08096)  

**Abstract**: Cellular development follows a stochastic yet rule-governed trajectory, though the underlying principles remain elusive. Here, we propose that cellular development follows paths of least action, aligning with foundational physical laws that govern dynamic systems across nature. We introduce a computational framework that takes advantage of the deep connection between the principle of least action and maximum entropy to model developmental processes using Transformers architecture. This approach enables precise quantification of entropy production, information flow curvature, and local irreversibility for developmental asymmetry in single-cell RNA sequence data. Within this unified framework, we provide interpretable metrics: entropy to capture exploration-exploitation trade-offs, curvature to assess plasticity-elasticity dynamics, and entropy production to characterize dedifferentiation and transdifferentiation. We validate our method across both single-cell and embryonic development datasets, demonstrating its ability to reveal hidden thermodynamic and informational constraints shaping cellular fate decisions. 

**Abstract (ZH)**: 细胞发育遵循一种随机 yet 规则制约的路径，尽管其背后的原理仍不清楚。在这里，我们提出细胞发育遵循最小作用路径，这与自然中动态系统所遵循的基本物理定律一致。我们介绍了一种计算框架，利用最小作用原理与最大熵之间的深层联系，使用变压器架构来建模发育过程。该方法能够精确量化熵产生、信息流曲率和发育不对称的局部不可逆性。在这一统一框架中，我们提供了可解释的度量标准：熵来捕捉探索-利用权衡，曲率来评估弹性-塑性 dynamics，熵产生来表征去分化和跨分化。我们在单细胞和胚胎发育数据集上验证了该方法，展示了其揭示塑造细胞命运决定的隐藏热力学和信息约束的能力。 

---
# STEI-PCN: an efficient pure convolutional network for traffic prediction via spatial-temporal encoding and inferring 

**Title (ZH)**: STEI-PCN：一种通过空间-时间编码和推断的高效纯卷积网络用于交通预测 

**Authors**: Kai Hu, Zhidan Zhao, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08061)  

**Abstract**: Traffic data exhibits complex temporal, spatial, and spatial-temporal correlations. Most of models use either independent modules to separately extract temporal and spatial correlations or joint modules to synchronously extract them, without considering the spatial-temporal correlations. Moreover, models that consider joint spatial-temporal correlations (temporal, spatial, and spatial-temporal correlations) often encounter significant challenges in accuracy and computational efficiency which prevent such models from demonstrating the expected advantages of a joint spatial-temporal correlations architecture. To address these issues, this paper proposes an efficient pure convolutional network for traffic prediction via spatial-temporal encoding and inferring (STEI-PCN). The model introduces and designs a dynamic adjacency matrix inferring module based on absolute spatial and temporal coordinates, as well as relative spatial and temporal distance encoding, using a graph convolutional network combined with gating mechanism to capture local synchronous joint spatial-temporal correlations. Additionally, three layers of temporal dilated causal convolutional network are used to capture long-range temporal correlations. Finally, through multi-view collaborative prediction module, the model integrates the gated-activated original, local synchronous joint spatial-temporal, and long-range temporal features to achieve comprehensive prediction. This study conducts extensive experiments on flow datasets (PeMS03/04/07/08) and speed dataset (PeMS-Bay), covering multiple prediction horizons. The results show that STEI-PCN demonstrates competitive computational efficiency in both training and inference speeds, and achieves superior or slightly inferior to state-of-the-art (SOTA) models on most evaluation metrics. 

**Abstract (ZH)**: 基于空间- temporal编码与推断的高效纯卷积网络用于交通预测 

---
# Vector Quantized-Elites: Unsupervised and Problem-Agnostic Quality-Diversity Optimization 

**Title (ZH)**: 精英向量量化：无监督且问题无关的质量多样性优化 

**Authors**: Constantinos Tsakonas, Konstantinos Chatzilygeroudis  

**Link**: [PDF](https://arxiv.org/pdf/2504.08057)  

**Abstract**: Quality-Diversity algorithms have transformed optimization by prioritizing the discovery of diverse, high-performing solutions over a single optimal result. However, traditional Quality-Diversity methods, such as MAP-Elites, rely heavily on predefined behavioral descriptors and complete prior knowledge of the task to define the behavioral space grid, limiting their flexibility and applicability. In this work, we introduce Vector Quantized-Elites (VQ-Elites), a novel Quality-Diversity algorithm that autonomously constructs a structured behavioral space grid using unsupervised learning, eliminating the need for prior task-specific knowledge. At the core of VQ-Elites is the integration of Vector Quantized Variational Autoencoders, which enables the dynamic learning of behavioral descriptors and the generation of a structured, rather than unstructured, behavioral space grid - a significant advancement over existing unsupervised Quality-Diversity approaches. This design establishes VQ-Elites as a flexible, robust, and task-agnostic optimization framework. To further enhance the performance of unsupervised Quality-Diversity algorithms, we introduce two key components: behavioral space bounding and cooperation mechanisms, which significantly improve convergence and performance. We validate VQ-Elites on robotic arm pose-reaching and mobile robot space-covering tasks. The results demonstrate its ability to efficiently generate diverse, high-quality solutions, emphasizing its adaptability, scalability, robustness to hyperparameters, and potential to extend Quality-Diversity optimization to complex, previously inaccessible domains. 

**Abstract (ZH)**: Vector Quantized-Elites：自主构建结构化行为空间网格的新型Quality-Diversity算法 

---
# Multi-Task Learning with Multi-Annotation Triplet Loss for Improved Object Detection 

**Title (ZH)**: 基于多注释三元损失的多任务学习改进目标检测 

**Authors**: Meilun Zhou, Aditya Dutt, Alina Zare  

**Link**: [PDF](https://arxiv.org/pdf/2504.08054)  

**Abstract**: Triplet loss traditionally relies only on class labels and does not use all available information in multi-task scenarios where multiple types of annotations are available. This paper introduces a Multi-Annotation Triplet Loss (MATL) framework that extends triplet loss by incorporating additional annotations, such as bounding box information, alongside class labels in the loss formulation. By using these complementary annotations, MATL improves multi-task learning for tasks requiring both classification and localization. Experiments on an aerial wildlife imagery dataset demonstrate that MATL outperforms conventional triplet loss in both classification and localization. These findings highlight the benefit of using all available annotations for triplet loss in multi-task learning frameworks. 

**Abstract (ZH)**: 多注释三元损失（MATL）框架：结合框信息提升多任务学习 

---
# Compositional Flows for 3D Molecule and Synthesis Pathway Co-design 

**Title (ZH)**: 3D分子与合成路径协同设计的组合流方法 

**Authors**: Tony Shen, Seonghwan Seo, Ross Irwin, Kieran Didi, Simon Olsson, Woo Youn Kim, Martin Ester  

**Link**: [PDF](https://arxiv.org/pdf/2504.08051)  

**Abstract**: Many generative applications, such as synthesis-based 3D molecular design, involve constructing compositional objects with continuous features. Here, we introduce Compositional Generative Flows (CGFlow), a novel framework that extends flow matching to generate objects in compositional steps while modeling continuous states. Our key insight is that modeling compositional state transitions can be formulated as a straightforward extension of the flow matching interpolation process. We further build upon the theoretical foundations of generative flow networks (GFlowNets), enabling reward-guided sampling of compositional structures. We apply CGFlow to synthesizable drug design by jointly designing the molecule's synthetic pathway with its 3D binding pose. Our approach achieves state-of-the-art binding affinity on all 15 targets from the LIT-PCBA benchmark, and 5.8$\times$ improvement in sampling efficiency compared to 2D synthesis-based baseline. To our best knowledge, our method is also the first to achieve state of-art-performance in both Vina Dock (-9.38) and AiZynth success rate (62.2\%) on the CrossDocked benchmark. 

**Abstract (ZH)**: compositional生成流（CGFlow）：一种用于生成连续特征组成对象的新框架 

---
# Can Reasoning LLMs Enhance Clinical Document Classification? 

**Title (ZH)**: 基于推理的大型语言模型能否增强临床文档分类？ 

**Authors**: Akram Mustafa, Usman Naseem, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08040)  

**Abstract**: Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications. 

**Abstract (ZH)**: 临床文档分类对于将未结构化医疗文本转换为标准化ICD-10诊断至关重要，但由于医学语言复杂、隐私限制以及标注数据集有限，该任务面临挑战。大型语言模型（LLMs）在提高准确性和效率方面展现出 promising 的潜力。本研究评估了八种LLMs（四种推理模型和四种非推理模型）在使用MIMIC-IV数据集分类临床出院总结方面的性能和一致性。通过cTAKES结构化临床叙事，模型在三次实验运行中进行了评估，采用多数投票确定最终预测结果。结果显示，推理模型在准确率（71% vs 68%）和F1分数（67% vs 60%）方面优于非推理模型，其中Gemini 2.0 Flash Thinking获得最高准确率（75%）和F1分数（76%）。然而，非推理模型在稳定性和一致性方面表现更好（91% vs 84%）。不同ICD-10代码的表现有所差异，推理模型在复杂案例中表现优异但在抽象类别方面遇到困难。研究结果表明，准确性和一致性之间存在权衡，这表明混合方法可能优化临床编码。未来研究应探索多标签分类、领域特异性微调和集成方法以提高模型在实际应用中的可靠性。 

---
# Learning Fine-grained Domain Generalization via Hyperbolic State Space Hallucination 

**Title (ZH)**: 基于双曲状态空间幻象的细粒度领域泛化学习 

**Authors**: Qi Bi, Jingjun Yi, Haolan Zhan, Wei Ji, Gui-Song Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.08020)  

**Abstract**: Fine-grained domain generalization (FGDG) aims to learn a fine-grained representation that can be well generalized to unseen target domains when only trained on the source domain data. Compared with generic domain generalization, FGDG is particularly challenging in that the fine-grained category can be only discerned by some subtle and tiny patterns. Such patterns are particularly fragile under the cross-domain style shifts caused by illumination, color and etc. To push this frontier, this paper presents a novel Hyperbolic State Space Hallucination (HSSH) method. It consists of two key components, namely, state space hallucination (SSH) and hyperbolic manifold consistency (HMC). SSH enriches the style diversity for the state embeddings by firstly extrapolating and then hallucinating the source images. Then, the pre- and post- style hallucinate state embeddings are projected into the hyperbolic manifold. The hyperbolic state space models the high-order statistics, and allows a better discernment of the fine-grained patterns. Finally, the hyperbolic distance is minimized, so that the impact of style variation on fine-grained patterns can be eliminated. Experiments on three FGDG benchmarks demonstrate its state-of-the-art performance. 

**Abstract (ZH)**: 细粒度领域泛化 (FGDG) 的超曲面状态空间幻视 (HSSH) 方法 

---
# DGFamba: Learning Flow Factorized State Space for Visual Domain Generalization 

**Title (ZH)**: DGFamba: 学习流因子化解空间进行视觉领域泛化 

**Authors**: Qi Bi, Jingjun Yi, Hao Zheng, Haolan Zhan, Wei Ji, Yawen Huang, Yuexiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08019)  

**Abstract**: Domain generalization aims to learn a representation from the source domain, which can be generalized to arbitrary unseen target domains. A fundamental challenge for visual domain generalization is the domain gap caused by the dramatic style variation whereas the image content is stable. The realm of selective state space, exemplified by VMamba, demonstrates its global receptive field in representing the content. However, the way exploiting the domain-invariant property for selective state space is rarely explored. In this paper, we propose a novel Flow Factorized State Space model, dubbed as DG-Famba, for visual domain generalization. To maintain domain consistency, we innovatively map the style-augmented and the original state embeddings by flow factorization. In this latent flow space, each state embedding from a certain style is specified by a latent probability path. By aligning these probability paths in the latent space, the state embeddings are able to represent the same content distribution regardless of the style differences. Extensive experiments conducted on various visual domain generalization settings show its state-of-the-art performance. 

**Abstract (ZH)**: 视觉域泛化旨在从源域中学习一种表示，该表示可以泛化到任意未见过的目标域。视觉域泛化的一个基本挑战是由剧烈的风格变化引起的域差距，而图像内容则保持稳定。选择性状态空间的领域示例，如VMamba，展示了其全局感受野来表示内容。然而，选择性状态空间利用领域不变性的方式鲜有探讨。在本文中，我们提出了一种新颖的流动因子化状态空间模型，命名为DG-Famba，用于视觉域泛化。为了保持域一致性，我们创新地通过流动因子化将增强风格和原始状态嵌入映射。在这一潜在流动空间中，来自特定风格的每个状态嵌入由一个潜在概率路径指定。通过在潜在空间中对齐这些概率路径，状态嵌入能够不受风格差异的影响来表示相同的内容分布。在各种视觉域泛化设置下的广泛实验中，其性能表现出色。 

---
# CDM-QTA: Quantized Training Acceleration for Efficient LoRA Fine-Tuning of Diffusion Model 

**Title (ZH)**: CDM-QTA: 量化训练加速高效Diffusion模型LoRA微调 

**Authors**: Jinming Lu, Minghao She, Wendong Mao, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07998)  

**Abstract**: Fine-tuning large diffusion models for custom applications demands substantial power and time, which poses significant challenges for efficient implementation on mobile devices. In this paper, we develop a novel training accelerator specifically for Low-Rank Adaptation (LoRA) of diffusion models, aiming to streamline the process and reduce computational complexity. By leveraging a fully quantized training scheme for LoRA fine-tuning, we achieve substantial reductions in memory usage and power consumption while maintaining high model fidelity. The proposed accelerator features flexible dataflow, enabling high utilization for irregular and variable tensor shapes during the LoRA process. Experimental results show up to 1.81x training speedup and 5.50x energy efficiency improvements compared to the baseline, with minimal impact on image generation quality. 

**Abstract (ZH)**: 针对低秩适应（LoRA）的大型扩散模型微调开发新型训练加速器：简化流程、降低计算复杂度并减少功耗 

---
# Evaluating the Fitness of Ontologies for the Task of Question Generation 

**Title (ZH)**: 评估本体适合度以用于问题生成任务 

**Authors**: Samah Alkhuzaey, Floriana Grasso, Terry R. Payne, Valentina Tamma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07994)  

**Abstract**: Ontology-based question generation is an important application of semantic-aware systems that enables the creation of large question banks for diverse learning environments. The effectiveness of these systems, both in terms of the calibre and cognitive difficulty of the resulting questions, depends heavily on the quality and modelling approach of the underlying ontologies, making it crucial to assess their fitness for this task. To date, there has been no comprehensive investigation into the specific ontology aspects or characteristics that affect the question generation process. Therefore, this paper proposes a set of requirements and task-specific metrics for evaluating the fitness of ontologies for question generation tasks in pedagogical settings. Using the ROMEO methodology, a structured framework for deriving task-specific metrics, an expert-based approach is employed to assess the performance of various ontologies in Automatic Question Generation (AQG) tasks, which is then evaluated over a set of ontologies. Our results demonstrate that ontology characteristics significantly impact the effectiveness of question generation, with different ontologies exhibiting varying performance levels. This highlights the importance of assessing ontology quality with respect to AQG tasks. 

**Abstract (ZH)**: 基于本体的问答生成是语义感知系统的重要应用，能够为多样化的学习环境创建大量问题库。这些系统的有效性，在于生成的问题的质量和认知难度，很大程度上取决于底层本体的质量和建模方法，因此评估它们是否适合这一任务至关重要。迄今为止，尚未对影响问答生成过程的具体本体特性进行全面调查。因此，本文提出了一套针对教学场景中问答生成任务评估本体适用性的需求和任务特定指标。利用ROMEO方法，一个结构化的任务特定指标推导框架，采用专家基于的方法评估了各种本体在自动问答生成任务中的性能，并在一组本体上进行了评估。我们的结果表明，本体特性显著影响问答生成的有效性，不同本体在性能上表现出差异。这强调了评估本体质量与自动问答生成任务之间的关系的重要性。 

---
# 'Neural howlround' in large language models: a self-reinforcing bias phenomenon, and a dynamic attenuation solution 

**Title (ZH)**: 大型语言模型中的“神经共鸣”：一种自我增强偏差现象及动态衰减解决方案 

**Authors**: Seth Drake  

**Link**: [PDF](https://arxiv.org/pdf/2504.07992)  

**Abstract**: Large language model (LLM)-driven AI systems may exhibit an inference failure mode we term `neural howlround,' a self-reinforcing cognitive loop where certain highly weighted inputs become dominant, leading to entrenched response patterns resistant to correction. This paper explores the mechanisms underlying this phenomenon, which is distinct from model collapse and biased salience weighting. We propose an attenuation-based correction mechanism that dynamically introduces counterbalancing adjustments and can restore adaptive reasoning, even in `locked-in' AI systems. Additionally, we discuss some other related effects arising from improperly managed reinforcement. Finally, we outline potential applications of this mitigation strategy for improving AI robustness in real-world decision-making tasks. 

**Abstract (ZH)**: 大型语言模型驱动的AI系统可能存在一种我们称为“神经回响”的推理失败模式，这是一种自强化的认知循环，其中某些高权重输入变得主导，导致顽固的响应模式难以纠正。本文探讨了这一现象的机理，该现象不同于模型崩溃和偏差显著性加权。我们提出了一种基于衰减的纠正机制，能够动态引入平衡调整，即使在“锁定”AI系统中也能恢复适应性推理。此外，我们还讨论了由于不当强化管理而产生的一些相关效应。最后，我们概述了这一缓解策略在提高实际决策任务中AI鲁棒性方面的潜在应用。 

---
# Comparative analysis of Realistic EMF Exposure Estimation from Low Density Sensor Network by Finite & Infinite Neural Networks 

**Title (ZH)**: 有限神经网络与无限神经网络在低密度传感器网络中现实电磁场暴露估计的比较分析 

**Authors**: Mohammed Mallik, Laurent Clavier, Davy P. Gaillot  

**Link**: [PDF](https://arxiv.org/pdf/2504.07990)  

**Abstract**: Understanding the spatial and temporal patterns of environmental exposure to radio-frequency electromagnetic fields (RF-EMF) is essential for conducting risk assessments. These assessments aim to explore potential connections between RF-EMF exposure and its effects on human health, as well as on wildlife and plant life. Existing research has used different machine learning tools for EMF exposure estimation; however, a comparative analysis of these techniques is required to better understand their performance for real-world datasets. In this work, we present both finite and infinite-width convolutional network-based methods to estimate and assess EMF exposure levels from 70 real-world sensors in Lille, France. A comparative analysis has been conducted to analyze the performance of the methods' execution time and estimation accuracy. To improve estimation accuracy for higher-resolution grids, we utilized a preconditioned gradient descent method for kernel estimation. Root Mean Square Error (RMSE) is used as the evaluation criterion for comparing the performance of these deep learning models. 

**Abstract (ZH)**: 理解无线电频电磁场（RF-EMF）的空间和时间分布模式对于进行风险评估至关重要。这些评估旨在探索RF-EMF暴露与其对人体健康、野生动物和植物生活的影响之间的潜在联系。现有的研究使用了不同的机器学习工具进行EMF暴露估计；然而，对这些技术的比较分析仍是必要的，以便更好地了解其在实际数据集中的性能。在本工作中，我们使用有限宽度和无限宽度卷积网络方法，从法国里尔的70个真实传感器数据中估计和评估EMF暴露水平。我们进行了比较分析，以分析这些方法的执行时间和估计准确性的性能。为了提高高分辨率网格的估计准确性，我们利用预条件梯度下降法进行核估计。均方根误差（RMSE）被用作这些深度学习模型性能比较的评估标准。 

---
# Regional Tiny Stories: Using Small Models to Compare Language Learning and Tokenizer Performance 

**Title (ZH)**: 区域微小故事：使用小型模型比较语言学习和分词器性能 

**Authors**: Nirvan Patil, Malhar Abhay Inamdar, Agnivo Gosai, Guruprasad Pathak, Anish Joshi, Aryan Sagavekar, Anish Joshirao, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2504.07989)  

**Abstract**: Small Language Models (SLMs) offer efficient alternatives to LLMs for specific domains. The 2023 TinyStories study developed an English dataset that allows SLMs with 1 to 10 million parameters to produce coherent outputs. Our research expands this framework by translating the original dataset into Indian languages and creating synthetic data using LLMs. We focus on Hindi, Marathi, and Bengali, evaluating SLMs for regional language processing and understanding linguistic complexity. We show that SLMs efficiently process regional languages with significantly fewer parameters than LLMs, providing a complementary framework for ``inference based evaluation" of tokenization strategies and linguistic complexity. Our analysis shows that language-specific tokenizers outperform general-purpose ones for Indian languages. Empirical validations, supported by information-theoretic and morphological analyses, provides fundamental understanding behind the better performance of Hindi models over Marathi and Bengali. Additionally, we show that synthetic datasets outperform translated content for training SLMs. Correlation analyses reveal cross-linguistic patterns and language-specific relationships between creativity, grammatical precision, and narrative completeness. These findings advance both the practical application of SLMs to underserved languages and our theoretical understanding of neural language development. 

**Abstract (ZH)**: 小型语言模型（SLMs）为特定领域提供了LLMs的高效替代方案。2023年TinyStories研究开发了一个英语数据集，使参数量在1至1000万之间的SLMs能够生成连贯的输出。我们的研究扩展了这一框架，将原始数据集翻译成印度语言，并使用LLMs生成合成数据。我们专注于北部印地语、马拉地语和孟加拉语，评估SLMs在区域语言处理中的表现，以及理解语言复杂性。我们展示了SLMs能够用远少于LLMs的参数高效处理区域语言，为基于推断的词元化策略和语言复杂性评估提供了补充框架。我们的分析表明，特定于语言的词元化器在印度语处理上优于通用词元化器。信息论和形态学分析的支持下的实证验证，揭示了印地语模型在马拉地语和孟加拉语中表现更好的基本原理。此外，我们展示了合成数据集在训练SLMs时优于翻译内容。相关性分析揭示了跨语言模式和语言特异性关系，这些关系涉及创造力、语法精确性和叙事完整性。这些发现不仅推进了对未服务语言中SLMs应用的实际意义，还加深了我们对神经语言发展理论的理解。 

---
# SEAL: Steerable Reasoning Calibration of Large Language Models for Free 

**Title (ZH)**: SEAL: 可引导的大型语言模型推理校准以实现自由推理 

**Authors**: Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07986)  

**Abstract**: Large Language Models (LLMs), such as OpenAI's o1-series have demonstrated compelling capabilities for complex reasoning tasks via the extended chain-of-thought (CoT) reasoning mechanism. However, recent studies reveal substantial redundancy in the CoT reasoning traces, which not only increases inference latency but also negatively impacts model performance by diverting attention to unnecessary reasoning paths. To address this issue, we investigate the internal reasoning structures of LLMs and categorize them into three primary thought types: execution, reflection, and transition thoughts. Moreover, our analysis reveals that excessive reflection and transition thoughts are strongly correlated with failure cases and these thought categories exhibit clear separation in the latent space. Based on these, we introduce SEAL (Steerable reasoning calibration), a training-free approach that seamlessly calibrates the CoT process, improving accuracy while demonstrating significant efficiency gains. SEAL consists of an offline stage for extracting the reasoning steering vector in the latent space, followed by an on-the-fly calibration of the reasoning trace through representation intervention using the steering vector. Notably, the steering vector exhibits strong transferability across various tasks. Extensive experiments across multiple models (DeepSeek-R1-Distill and QwQ-32B-Preview) and benchmarks (Math500, GSM8K, LiveCodeBench) validate the effectiveness of SEAL, up to a 11% improvement in accuracy while reducing reasoning tokens by 11.8% to 50.4%. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的可导航推理校准：大型语言模型，如OpenAI的o1系列，通过扩展的链式思考（CoT）推理机制展示了强大的复杂推理能力。然而，最近的研究揭示了CoT推理轨迹中的大量冗余，这不仅增加了推理延迟，还通过分散注意力到不必要的推理路径上而负面影响了模型性能。为了解决这一问题，我们研究了LLMs的内部推理结构，并将其分类为三种主要的思维类型：执行思维、反思思维和过渡思维。此外，我们的分析表明，过度的反思思维和过渡思维与失败案例密切相关，这些思维类别在潜在空间中表现出明确的分离。基于这些发现，我们引入了SEAL（可导航的推理校准），这是一种无需训练的方法，能够无缝校准CoT过程，提高准确率同时显著提高效率。SEAL包括一个离线阶段，用于在潜在空间中提取推理导向向量，接着是通过使用导向向量进行表示干预来实时校准推理轨迹。值得注意的是，导向向量在各种任务之间表现出很强的可转移性。在多种模型（DeepSeek-R1-Distill和QwQ-32B-Preview）和基准（Math500、GSM8K、LiveCodeBench）上进行的大量实验证实了SEAL的有效性，准确率最多可提高11%，同时减少推理令牌11.8%至50.4%。我们的代码已公开可在以下网址获取。 

---
# Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method 

**Title (ZH)**: 基于心理卫生知识增强的LLM社交网络危机干预文本转移识别方法 

**Authors**: Shurui Wu, Xinyi Huang, Dingxin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07983)  

**Abstract**: As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations. 

**Abstract (ZH)**: 社交媒体平台上心理健康危机频发背景下基于大语言模型的文本转移识别方法及其在危机干预中的应用：融合领域特定心理健康知识的多层次框架 

---
# Metamorphic Testing for Fairness Evaluation in Large Language Models: Identifying Intersectional Bias in LLaMA and GPT 

**Title (ZH)**: 大语言模型中公平性评估的变形测试：识别LLaMA和GPT的交叉偏见 

**Authors**: Harishwar Reddy, Madhusudan Srinivasan, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2504.07982)  

**Abstract**: Large Language Models (LLMs) have made significant strides in Natural Language Processing but remain vulnerable to fairness-related issues, often reflecting biases inherent in their training data. These biases pose risks, particularly when LLMs are deployed in sensitive areas such as healthcare, finance, and law. This paper introduces a metamorphic testing approach to systematically identify fairness bugs in LLMs. We define and apply a set of fairness-oriented metamorphic relations (MRs) to assess the LLaMA and GPT model, a state-of-the-art LLM, across diverse demographic inputs. Our methodology includes generating source and follow-up test cases for each MR and analyzing model responses for fairness violations. The results demonstrate the effectiveness of MT in exposing bias patterns, especially in relation to tone and sentiment, and highlight specific intersections of sensitive attributes that frequently reveal fairness faults. This research improves fairness testing in LLMs, providing a structured approach to detect and mitigate biases and improve model robustness in fairness-sensitive applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理领域取得了显著进展，但仍存在与公平性相关的问题，往往反映了其训练数据中存在的偏见。这些偏见在LLMs部署于医疗、金融和法律等敏感领域时会带来风险。本文介绍了一种 metamorphic 测试方法，以系统地识别LLMs中的公平性漏洞。我们定义并应用于评估最先进的LLM——LLaMA和GPT模型的一组公平性导向的 metamorphic 关系（MRs），涉及多样化的 demographic 输入。该方法包括为每个MR生成源测试用例和后续测试用例，并分析模型响应以检测公平性违规。结果表明，MT在揭示与语气和情感相关偏见模式方面特别有效，并突显了敏感属性的特定交叉点，这些交叉点经常揭示公平性故障。该研究改善了LLMs的公平性测试，提供了一种结构化的检测和缓解偏见的方法，以提高公平性敏感应用中模型的鲁棒性。 

---
# SPHERE: An Evaluation Card for Human-AI Systems 

**Title (ZH)**: SPHERE: 人类-人工智能系统评估卡 

**Authors**: Qianou Ma, Dora Zhao, Xinran Zhao, Chenglei Si, Chenyang Yang, Ryan Louie, Ehud Reiter, Diyi Yang, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07971)  

**Abstract**: In the era of Large Language Models (LLMs), establishing effective evaluation methods and standards for diverse human-AI interaction systems is increasingly challenging. To encourage more transparent documentation and facilitate discussion on human-AI system evaluation design options, we present an evaluation card SPHERE, which encompasses five key dimensions: 1) What is being evaluated?; 2) How is the evaluation conducted?; 3) Who is participating in the evaluation?; 4) When is evaluation conducted?; 5) How is evaluation validated? We conduct a review of 39 human-AI systems using SPHERE, outlining current evaluation practices and areas for improvement. We provide three recommendations for improving the validity and rigor of evaluation practices. 

**Abstract (ZH)**: 在大型语言模型时代，建立多样的人工智能交互系统评价方法和标准越来越具有挑战性。为促进更加透明的文档编写和便于讨论人工智能系统评价设计选项，我们提出了一种评价卡SPHERE，涵盖了五个关键维度：1) 评价什么？；2) 如何进行评价？；3) 参与评价的是谁？；4) 何时进行评价？；5) 如何验证评价结果？我们使用SPHERE对39个人工智能系统进行了评审，概述了当前的评价实践和改进领域，并提供了三条建议以提高评价实践的有效性和严谨性。 

---
