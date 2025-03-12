# Chain-of-Thought Reasoning In The Wild Is Not Always Faithful 

**Title (ZH)**: 野外的链式思维推理并不总是可靠的。 

**Authors**: Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08679)  

**Abstract**: Chain-of-Thought (CoT) reasoning has significantly advanced state-of-the-art AI capabilities. However, recent studies have shown that CoT reasoning is not always faithful, i.e. CoT reasoning does not always reflect how models arrive at conclusions. So far, most of these studies have focused on unfaithfulness in unnatural contexts where an explicit bias has been introduced. In contrast, we show that unfaithful CoT can occur on realistic prompts with no artificial bias. Our results reveal concerning rates of several forms of unfaithful reasoning in frontier models: Sonnet 3.7 (30.6%), DeepSeek R1 (15.8%) and ChatGPT-4o (12.6%) all answer a high proportion of question pairs unfaithfully. Specifically, we find that models rationalize their implicit biases in answers to binary questions ("implicit post-hoc rationalization"). For example, when separately presented with the questions "Is X bigger than Y?" and "Is Y bigger than X?", models sometimes produce superficially coherent arguments to justify answering Yes to both questions or No to both questions, despite such responses being logically contradictory. We also investigate restoration errors (Dziri et al., 2023), where models make and then silently correct errors in their reasoning, and unfaithful shortcuts, where models use clearly illogical reasoning to simplify solving problems in Putnam questions (a hard benchmark). Our findings raise challenges for AI safety work that relies on monitoring CoT to detect undesired behavior. 

**Abstract (ZH)**: Chain-of-Thought 理论在先进人工智能能力中的应用已经取得了显著进展，然而最近的研究表明，Chain-of-Thought 理论并非总是忠实的，即它并不总是反映模型如何得出结论。迄今为止，大多数相关研究主要集中在那些故意引入了非自然偏见的不忠实上下文中。相比之下，我们展示了在没有人为偏见的现实提示下也会出现不忠实的 Chain-of-Thought。我们的研究结果揭示了前沿模型中几种形式的不忠实推理的较高频率：Sonnet 3.7（30.6%）、DeepSeek R1（15.8%）和 ChatGPT-4o（12.6%）都以较高的比例不忠实回答了问题对。具体来说，我们发现模型在回答二元问题时通过隐性后验理性化来为自己辩解（“隐性后验理性化”）。例如，当分别展示“X 是否比 Y 大？”和“Y 是否比 X 大？”这两个问题时，模型有时会生成表面上连贯的论据来证明对两个问题都回答“是”或“否”，尽管这样的回答在逻辑上是矛盾的。我们还研究了修复错误（Dziri et al., 2023），即模型在推理过程中犯错误并默默纠正，以及不忠实的捷径，即模型使用明显不合逻辑的推理简化解决培土姆难题中的问题。我们的发现对依赖监控 Chain-of-Thought 以检测不 desired 行为的人工智能安全工作提出了挑战。 

---
# Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs 

**Title (ZH)**: 基于LLM的推理与采样增强的MCQ Difficulty预测 

**Authors**: Wanyong Feng, Peter Tran, Stephen Sireci, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08551)  

**Abstract**: The difficulty of multiple-choice questions (MCQs) is a crucial factor for educational assessments. Predicting MCQ difficulty is challenging since it requires understanding both the complexity of reaching the correct option and the plausibility of distractors, i.e., incorrect options. In this paper, we propose a novel, two-stage method to predict the difficulty of MCQs. First, to better estimate the complexity of each MCQ, we use large language models (LLMs) to augment the reasoning steps required to reach each option. We use not just the MCQ itself but also these reasoning steps as input to predict the difficulty. Second, to capture the plausibility of distractors, we sample knowledge levels from a distribution to account for variation among students responding to the MCQ. This setup, inspired by item response theory (IRT), enable us to estimate the likelihood of students selecting each (both correct and incorrect) option. We align these predictions with their ground truth values, using a Kullback-Leibler (KL) divergence-based regularization objective, and use estimated likelihoods to predict MCQ difficulty. We evaluate our method on two real-world \emph{math} MCQ and response datasets with ground truth difficulty values estimated using IRT. Experimental results show that our method outperforms all baselines, up to a 28.3\% reduction in mean squared error and a 34.6\% improvement in the coefficient of determination. We also qualitatively discuss how our novel method results in higher accuracy in predicting MCQ difficulty. 

**Abstract (ZH)**: 多重选择题（MCQ）难度的预测是教育评估中的关键因素。由于预测MCQ难度需要理解达到正确选项的复杂性以及分散项（即错误选项）的合理性，因此这一过程具有挑战性。本文提出了一种新颖的两阶段方法来预测MCQ的难度。首先，为了更好地估计每道MCQ的复杂性，我们利用大规模语言模型（LLMs）来增强达到每个选项所需的推理步骤。我们将不仅仅是MCQ本身，还包括这些推理步骤作为输入来预测难度。其次，为了捕捉分散项的合理性，我们从知识水平分布中抽样，以考虑回答MCQ的学生之间的差异。这一设置借鉴了项目反应理论（IRT），使我们能够估计学生选择每个选项（正确和错误的）的概率。我们使用基于Kullback-Leibler（KL）散度的正则化目标将这些预测与真实值对齐，并使用估计的概率来预测MCQ的难度。我们在两个使用IRT估计真实难度值的实际情况数学MCQ和响应数据集上评估了该方法。实验结果表明，我们提出的方法优于所有基线，平均平方误差降低了28.3%，决定系数提高了34.6%，并且我们的新颖方法在预测MCQ难度方面的准确性更高。 

---
# Graph of AI Ideas: Leveraging Knowledge Graphs and LLMs for AI Research Idea Generation 

**Title (ZH)**: AI思想图谱：利用知识图谱和大规模语言模型进行AI研究思路生成 

**Authors**: Xian Gao, Zongyun Zhang, Mingye Xie, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08549)  

**Abstract**: Reading relevant scientific papers and analyzing research development trends is a critical step in generating new scientific ideas. However, the rapid increase in the volume of research literature and the complex citation relationships make it difficult for researchers to quickly analyze and derive meaningful research trends. The development of large language models (LLMs) has provided a novel approach for automatically summarizing papers and generating innovative research ideas. However, existing paper-based idea generation methods either simply input papers into LLMs via prompts or form logical chains of creative development based on citation relationships, without fully exploiting the semantic information embedded in these citations. Inspired by knowledge graphs and human cognitive processes, we propose a framework called the Graph of AI Ideas (GoAI) for the AI research field, which is dominated by open-access papers. This framework organizes relevant literature into entities within a knowledge graph and summarizes the semantic information contained in citations into relations within the graph. This organization effectively reflects the relationships between two academic papers and the advancement of the AI research field. Such organization aids LLMs in capturing the current progress of research, thereby enhancing their creativity. Experimental results demonstrate the effectiveness of our approach in generating novel, clear, and effective research ideas. 

**Abstract (ZH)**: 阅读相关科学论文并分析研究发展趋势是生成新科学思想的关键步骤。然而，研究文献的 rapidly 增加和复杂的引用关系使得研究人员难以快速分析和提炼出有意义的研究趋势。大规模语言模型（LLMs）的发展为自动总结论文和生成创新研究思想提供了新的方法。然而，现有的基于论文的理念生成方法要么仅通过提示将论文输入LLMs，要么基于引用关系形成逻辑上的创造性链条发展，而未能充分利用这些引用中嵌入的语义信息。受知识图谱和人类认知过程的启发，我们提出了一种名为Graph of AI Ideas (GoAI) 的框架，适用于以开放获取论文为主的AI研究领域。该框架将相关文献组织为知识图谱中的实体，并将引用中包含的语义信息总结为图中的关系。这种组织有效地反映了两篇学术论文之间的关系以及AI研究领域的进步。这种组织有助于LLMs捕捉当前研究的进展，从而增强其创造力。实验结果证明了该方法在生成新颖、清晰和有效的研究思想方面的有效性。 

---
# Chemical reasoning in LLMs unlocks steerable synthesis planning and reaction mechanism elucidation 

**Title (ZH)**: LLMs中的化学推理解锁可引导的合成规划和反应机制阐明 

**Authors**: Andres M Bran, Theo A Neukomm, Daniel P Armstrong, Zlatko Jončev, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2503.08537)  

**Abstract**: While machine learning algorithms have been shown to excel at specific chemical tasks, they have struggled to capture the strategic thinking that characterizes expert chemical reasoning, limiting their widespread adoption. Here we demonstrate that large language models (LLMs) can serve as powerful chemical reasoning engines when integrated with traditional search algorithms, enabling a new approach to computer-aided chemistry that mirrors human expert thinking. Rather than using LLMs to directly manipulate chemical structures, we leverage their ability to evaluate chemical strategies and guide search algorithms toward chemically meaningful solutions. We demonstrate this paradigm through two fundamental challenges: strategy-aware retrosynthetic planning and mechanism elucidation. In retrosynthetic planning, our method allows chemists to specify desired synthetic strategies in natural language to find routes that satisfy these constraints in vast searches. In mechanism elucidation, LLMs guide the search for plausible reaction mechanisms by combining chemical principles with systematic exploration. Our approach shows strong performance across diverse chemical tasks, with larger models demonstrating increasingly sophisticated chemical reasoning. Our approach establishes a new paradigm for computer-aided chemistry that combines the strategic understanding of LLMs with the precision of traditional chemical tools, opening possibilities for more intuitive and powerful chemical reasoning systems. 

**Abstract (ZH)**: 尽管机器学习算法在特定化学任务上表现出色，但在捕捉专家级化学推理中的策略性思考方面仍存在问题，限制了其广泛应用。在这里，我们展示了大规模语言模型（LLMs）在与传统搜索算法集成时可以作为强大的化学推理引擎，从而提供了一种模拟人类专家思维的计算机辅助化学新方法。我们没有直接利用LLMs来操作化学结构，而是利用它们评估化学策略的能力，引导搜索算法找到具有化学意义的解决方案。我们通过两种基本挑战展示了这一范式：具有策略意识的逆合成规划和机制阐明。在逆合成规划中，我们的方法允许化学家使用自然语言指定所需的合成策略，以在大规模搜索中找到满足这些约束的路线。在机制阐明中，LLMs通过结合化学原理和系统性探索，指导可能反应机制的搜索。我们的方法在多种化学任务上表现出色，更大规模的模型展示了更加复杂的化学推理能力。我们的方法建立了一种新的计算机辅助化学范式，将LLMs的战略理解与传统化学工具的精确性相结合，为更直观和强大的化学推理系统开辟了可能性。 

---
# Seeing and Reasoning with Confidence: Supercharging Multimodal LLMs with an Uncertainty-Aware Agentic Framework 

**Title (ZH)**: 基于不确定性意识代理框架增强的多模态大语言模型的感知与推理 

**Authors**: Zhuo Zhi, Chen Feng, Adam Daneshmend, Mine Orlu, Andreas Demosthenous, Lu Yin, Da Li, Ziquan Liu, Miguel R. D. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2503.08308)  

**Abstract**: Multimodal large language models (MLLMs) show promise in tasks like visual question answering (VQA) but still face challenges in multimodal reasoning. Recent works adapt agentic frameworks or chain-of-thought (CoT) reasoning to improve performance. However, CoT-based multimodal reasoning often demands costly data annotation and fine-tuning, while agentic approaches relying on external tools risk introducing unreliable output from these tools. In this paper, we propose Seeing and Reasoning with Confidence (SRICE), a training-free multimodal reasoning framework that integrates external vision models with uncertainty quantification (UQ) into an MLLM to address these challenges. Specifically, SRICE guides the inference process by allowing MLLM to autonomously select regions of interest through multi-stage interactions with the help of external tools. We propose to use a conformal prediction-based approach to calibrate the output of external tools and select the optimal tool by estimating the uncertainty of an MLLM's output. Our experiment shows that the average improvement of SRICE over the base MLLM is 4.6% on five datasets and the performance on some datasets even outperforms fine-tuning-based methods, revealing the significance of ensuring reliable tool use in an MLLM agent. 

**Abstract (ZH)**: 基于自信推理的多模态 reasoning 体系结构（SRICE） 

---
# Beyond Outlining: Heterogeneous Recursive Planning for Adaptive Long-form Writing with Language Models 

**Title (ZH)**: 超越提纲：语言模型驱动的异构递归规划方法实现适应性长文写作 

**Authors**: Ruibin Xiong, Yimeng Chen, Dmitrii Khizbullin, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2503.08275)  

**Abstract**: Long-form writing agents require flexible integration and interaction across information retrieval, reasoning, and composition. Current approaches rely on predetermined workflows and rigid thinking patterns to generate outlines before writing, resulting in constrained adaptability during writing. In this paper we propose a general agent framework that achieves human-like adaptive writing through recursive task decomposition and dynamic integration of three fundamental task types, i.e. retrieval, reasoning, and composition. Our methodology features: 1) a planning mechanism that interleaves recursive task decomposition and execution, eliminating artificial restrictions on writing workflow; and 2) integration of task types that facilitates heterogeneous task decomposition. Evaluations on both fiction writing and technical report generation show that our method consistently outperforms state-of-the-art approaches across all automatic evaluation metrics, which demonstrate the effectiveness and broad applicability of our proposed framework. 

**Abstract (ZH)**: 长文写作代理需要在信息检索、推理和写作之间灵活集成和交互。当前的方法依赖于预先确定的工作流程和僵化的思维模式，在写作前生成提纲，导致写作过程中适应性受限。本文提出了一种通用代理框架，通过递归任务分解和动态整合检索、推理和写作三种基本任务类型，实现类似人类的适应性写作。我们的方法包含：1) 一种交替进行递归任务分解和执行的规划机制，消除写作流程的人为限制；2) 任务类型的整合，促进异构任务分解。对虚构写作和技术报告生成的评估表明，我们的方法在所有自动评估指标上一致优于现有最先进的方法，证明了我们提出的框架的有效性和广泛适用性。 

---
# HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents 

**Title (ZH)**: HASARD：基于视觉的体态智能体安全强化学习基准测试 

**Authors**: Tristan Tomilin, Meng Fang, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08241)  

**Abstract**: Advancing safe autonomous systems through reinforcement learning (RL) requires robust benchmarks to evaluate performance, analyze methods, and assess agent competencies. Humans primarily rely on embodied visual perception to safely navigate and interact with their surroundings, making it a valuable capability for RL agents. However, existing vision-based 3D benchmarks only consider simple navigation tasks. To address this shortcoming, we introduce \textbf{HASARD}, a suite of diverse and complex tasks to $\textbf{HA}$rness $\textbf{SA}$fe $\textbf{R}$L with $\textbf{D}$oom, requiring strategic decision-making, comprehending spatial relationships, and predicting the short-term future. HASARD features three difficulty levels and two action spaces. An empirical evaluation of popular baseline methods demonstrates the benchmark's complexity, unique challenges, and reward-cost trade-offs. Visualizing agent navigation during training with top-down heatmaps provides insight into a method's learning process. Incrementally training across difficulty levels offers an implicit learning curriculum. HASARD is the first safe RL benchmark to exclusively target egocentric vision-based learning, offering a cost-effective and insightful way to explore the potential and boundaries of current and future safe RL methods. The environments and baseline implementations are open-sourced at this https URL. 

**Abstract (ZH)**: 通过强化学习推动安全自主系统的进步需要稳健的基准来评估性能、分析方法和评估代理能力。由于人类主要依赖具身视觉感知来安全导航和与周围环境交互，这是一项对强化学习代理有价值的技能。然而，现有的基于视觉的三维基准仅考虑简单的导航任务。为解决这一不足，我们引入了HASARD，这是一个用于利用Doom的多样且复杂的任务套件，要求战略决策、理解空间关系和预测短期未来。HASARD 包含三个难度级别和两个行动空间。对流行基线方法的经验性评估展示了该基准的复杂性、独特的挑战和奖励与成本之间的权衡。在训练期间可视化代理导航的过程中的鸟瞰热图为方法的学习过程提供了见解。逐步跨难度级别训练提供了隐式的学习课程。HASARD 是首个专门针对自视点视觉感知学习的安全强化学习基准，提供了一种经济高效且具有洞察力的方式，以探索当前和未来安全强化学习方法的潜力和界限。环境和基线实现开源于此 <https://>。 

---
# Guess What I am Thinking: A Benchmark for Inner Thought Reasoning of Role-Playing Language Agents 

**Title (ZH)**: 猜猜我在想什么：角色扮演语言代理内心思考推理基准 

**Authors**: Rui Xu, MingYu Wang, XinTao Wang, Dakuan Lu, Xiaoyu Tan, Wei Chu, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08193)  

**Abstract**: Recent advances in LLM-based role-playing language agents (RPLAs) have attracted broad attention in various applications. While chain-of-thought reasoning has shown importance in many tasks for LLMs, the internal thinking processes of RPLAs remain unexplored. Understanding characters' inner thoughts is crucial for developing advanced RPLAs. In this paper, we introduce ROLETHINK, a novel benchmark constructed from literature for evaluating character thought generation. We propose the task of inner thought reasoning, which includes two sets: the gold set that compares generated thoughts with original character monologues, and the silver set that uses expert synthesized character analyses as references. To address this challenge, we propose MIRROR, a chain-of-thought approach that generates character thoughts by retrieving memories, predicting character reactions, and synthesizing motivations. Through extensive experiments, we demonstrate the importance of inner thought reasoning for RPLAs, and MIRROR consistently outperforms existing methods. Resources are available at this https URL. 

**Abstract (ZH)**: Recent advances in LLM-based角色扮演语言代理（RPLAs）在各种应用中引起了广泛关注。尽管因果推理在许多任务中对LLMs显示出重要性，但RPLAs的内部思考过程仍未被探索。理解角色的内心想法对于开发高级RPLAs至关重要。本文介绍了一个新的基于文学构建的基准ROLETHINK，用于评估角色思维生成。我们提出了内部思考推理的任务，包括黄金集和白银集。通过广泛的实验，我们证明了内部思考推理对于RPLAs的重要性，并且MIRROR在所有现有方法中表现更优。更多资源请点击这个链接。 

---
# Privacy-Enhancing Paradigms within Federated Multi-Agent Systems 

**Title (ZH)**: 增强隐私保护的联邦多代理系统范式 

**Authors**: Zitong Shi, Guancheng Wan, Wenke Huang, Guibin Zhang, Jiawei Shao, Mang Ye, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08175)  

**Abstract**: LLM-based Multi-Agent Systems (MAS) have proven highly effective in solving complex problems by integrating multiple agents, each performing different roles. However, in sensitive domains, they face emerging privacy protection challenges. In this paper, we introduce the concept of Federated MAS, highlighting the fundamental differences between Federated MAS and traditional FL. We then identify key challenges in developing Federated MAS, including: 1) heterogeneous privacy protocols among agents, 2) structural differences in multi-party conversations, and 3) dynamic conversational network structures. To address these challenges, we propose Embedded Privacy-Enhancing Agents (EPEAgent), an innovative solution that integrates seamlessly into the Retrieval-Augmented Generation (RAG) phase and the context retrieval stage. This solution minimizes data flows, ensuring that only task-relevant, agent-specific information is shared. Additionally, we design and generate a comprehensive dataset to evaluate the proposed paradigm. Extensive experiments demonstrate that EPEAgent effectively enhances privacy protection while maintaining strong system performance. The code will be availiable at this https URL 

**Abstract (ZH)**: 基于LLM的联邦多代理系统：面向敏感领域的隐私保护挑战与解决方案 

---
# AI-native Memory 2.0: Second Me 

**Title (ZH)**: AI原生内存2.0：第二章 

**Authors**: Jiale Wei, Xiang Ying, Tao Gao, Felix Tao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08102)  

**Abstract**: Human interaction with the external world fundamentally involves the exchange of personal memory, whether with other individuals, websites, applications, or, in the future, AI agents. A significant portion of this interaction is redundant, requiring users to repeatedly provide the same information across different contexts. Existing solutions, such as browser-stored credentials, autofill mechanisms, and unified authentication systems, have aimed to mitigate this redundancy by serving as intermediaries that store and retrieve commonly used user data. The advent of large language models (LLMs) presents an opportunity to redefine memory management through an AI-native paradigm: SECOND ME. SECOND ME acts as an intelligent, persistent memory offload system that retains, organizes, and dynamically utilizes user-specific knowledge. By serving as an intermediary in user interactions, it can autonomously generate context-aware responses, prefill required information, and facilitate seamless communication with external systems, significantly reducing cognitive load and interaction friction. Unlike traditional memory storage solutions, SECOND ME extends beyond static data retention by leveraging LLM-based memory parameterization. This enables structured organization, contextual reasoning, and adaptive knowledge retrieval, facilitating a more systematic and intelligent approach to memory management. As AI-driven personal agents like SECOND ME become increasingly integrated into digital ecosystems, SECOND ME further represents a critical step toward augmenting human-world interaction with persistent, contextually aware, and self-optimizing memory systems. We have open-sourced the fully localizable deployment system at GitHub: this https URL. 

**Abstract (ZH)**: 人类与外部世界互动本质上涉及个人记忆与个体、网站、应用程序或未来的AI代理之间的交流。这一互动中存在大量的冗余部分，要求用户在不同上下文中重复提供相同信息。现有的解决方案，如浏览器存储的凭证、自动填充机制和统一认证系统，旨在通过充当存储和检索常用用户数据的中介来减轻这一冗余。大型语言模型（LLMs）的出现为通过AI原生范式重新定义记忆管理提供了机会：SECOND ME。SECOND ME充当一个智能的持久记忆卸载系统，保留、组织和动态利用用户特定的知识。通过在用户互动中充当中介，它可以自主生成上下文感知的响应、预填所需信息，并促进与外部系统的无缝通信，从而显著降低认知负担和互动摩擦。与传统的记忆存储解决方案不同，SECOND ME通过利用基于LLM的记忆参数化，超越了静态数据保留，实现了结构化的组织、上下文推理和适应性知识检索，从而实现更加系统化和智能化的记忆管理。随着像SECOND ME这样的AI驱动个人代理越来越多地融入数字生态系统中，SECOND ME进一步代表了一种关键步骤，即通过持久、上下文感知和自我优化的记忆系统增强人与世界的互动。我们已在GitHub上开源了完全本地化的部署系统：this https URL。 

---
# STGDPM:Vessel Trajectory Prediction with Spatio-Temporal Graph Diffusion Probabilistic Model 

**Title (ZH)**: STGDPM：基于时空图扩散概率模型的血管轨迹预测 

**Authors**: Jin Wenzhe, Tang Haina, Zhang Xudong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08065)  

**Abstract**: Vessel trajectory prediction is a critical component for ensuring maritime traffic safety and avoiding collisions. Due to the inherent uncertainty in vessel behavior, trajectory prediction systems must adopt a multimodal approach to accurately model potential future motion states. However, existing vessel trajectory prediction methods lack the ability to comprehensively model behavioral multi-modality. To better capture multimodal behavior in interactive scenarios, we propose modeling interactions as dynamic graphs, replacing traditional aggregation-based techniques that rely on vessel states. By leveraging the natural multimodal capabilities of diffusion models, we frame the trajectory prediction task as an inverse process of motion uncertainty diffusion, wherein uncertainties across potential navigational areas are progressively eliminated until the desired trajectories is produced. In summary, we pioneer the integration of Spatio-Temporal Graph (STG) with diffusion models in ship trajectory prediction. Extensive experiments on real Automatic Identification System (AIS) data validate the superiority of our approach. 

**Abstract (ZH)**: 船舶轨迹预测是确保海上交通安全和避免碰撞的关键组件。由于船舶行为固有的不确定性，轨迹预测系统必须采用多模式方法来准确建模潜在的未来运动状态。然而，现有的船舶轨迹预测方法无法全面建模行为的多模态性。为更好地捕捉交互场景中的多模态行为，我们提出将交互建模为动态图，替代依赖于船舶状态的传统聚合方法。通过利用扩散模型的自然多模态能力，我们将轨迹预测任务框架化为运动不确定性扩散的逆过程，在此过程中，潜在导航区域的不确定性逐渐消除，直至产生所需的轨迹。总之，我们首次将空间时间图（STG）与扩散模型集成应用于船舶轨迹预测。广泛实验验证了我们方法的优越性。 

---
# Counterfactual Language Reasoning for Explainable Recommendation Systems 

**Title (ZH)**: 可因果语境推理的可解释推荐系统 

**Authors**: Guanrong Li, Haolin Yang, Xinyu Liu, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08051)  

**Abstract**: Explainable recommendation systems leverage transparent reasoning to foster user trust and improve decision-making processes. Current approaches typically decouple recommendation generation from explanation creation, violating causal precedence principles where explanatory factors should logically precede outcomes. This paper introduces a novel framework integrating structural causal models with large language models to establish causal consistency in recommendation pipelines. Our methodology enforces explanation factors as causal antecedents to recommendation predictions through causal graph construction and counterfactual adjustment. We particularly address the confounding effect of item popularity that distorts personalization signals in explanations, developing a debiasing mechanism that disentangles genuine user preferences from conformity bias. Through comprehensive experiments across multiple recommendation scenarios, we demonstrate that CausalX achieves superior performance in recommendation accuracy, explanation plausibility, and bias mitigation compared to baselines. 

**Abstract (ZH)**: 可解释的推荐系统通过透明的推理来培养用户信任并改善决策过程。当前的方法通常将推荐生成与解释生成分开，违反了因果优先原则，即解释因素应逻辑上先于结果。本文提出了一种新的框架，将结构因果模型与大规模语言模型结合起来，以在推荐管道中建立因果一致性。我们的方法通过因果图构建和反事实调整，强制解释因素作为推荐预测的因果先驱。我们特别解决了商品流行性造成的共因效应，该效应扭曲了解释中的个性化信号，开发了一种去偏机制，以分离出真实的用户偏好与从众偏差。通过在多个推荐场景中的全面实验，我们展示了CausalX在推荐准确性、解释合理性和偏见缓解方面的优越性能，相比基线方法表现更佳。 

---
# SQLCritic: Correcting Text-to-SQL Generation via Clause-wise Critic 

**Title (ZH)**: SQLCritic: 通过子句级别批评纠正文本到SQL生成 

**Authors**: Jikai Chen, Leilei Gan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07996)  

**Abstract**: Recent advancements in Text-to-SQL systems have improved the conversion of natural language queries into SQL, but challenges remain in ensuring accuracy and reliability. While self-correction techniques refine outputs, they often introduce new errors. Existing methods focused on execution feedback mainly address syntax issues, leaving semantic errors -- where the query's logic fails to align with the user's intent -- largely unaddressed.
We propose a novel approach combining structured execution feedback with a trained critic agent that provides detailed, interpretable critiques. This method effectively identifies and corrects both syntactic and semantic errors, enhancing accuracy and interpretability. Experimental results show significant improvements on two major Text-to-SQL benchmarks, Spider and BIRD, demonstrating the effectiveness of our approach. 

**Abstract (ZH)**: Recent advancements in Text-to-SQL系统提高了自然语言查询到SQL的转换效率，但确保准确性和可靠性仍面临挑战。尽管自我修正技术可以改进输出，但往往会引入新的错误。现有方法主要依赖执行反馈来解决语法问题，而未能充分处理语义错误，即查询逻辑与用户意图不一致的问题。我们提出了一种结合结构化执行反馈和训练过的批评代理的新方法，该代理能够提供详细的、可解释的批评。此方法有效识别和修正了语法和语义错误，提高了准确性和可解释性。实验结果在Spider和BIRD两大Text-to-SQL基准上显示出显著改进，证明了该方法的有效性。 

---
# LLM-Powered Knowledge Graphs for Enterprise Intelligence and Analytics 

**Title (ZH)**: 基于LLM的企业智能与分析知识图谱 

**Authors**: Rajeev Kumar, Kumar Ishan, Harishankar Kumar, Abhinandan Singla  

**Link**: [PDF](https://arxiv.org/pdf/2503.07993)  

**Abstract**: Disconnected data silos within enterprises obstruct the extraction of actionable insights, diminishing efficiency in areas such as product development, client engagement, meeting preparation, and analytics-driven decision-making. This paper introduces a framework that uses large language models (LLMs) to unify various data sources into a comprehensive, activity-centric knowledge graph. The framework automates tasks such as entity extraction, relationship inference, and semantic enrichment, enabling advanced querying, reasoning, and analytics across data types like emails, calendars, chats, documents, and logs. Designed for enterprise flexibility, it supports applications such as contextual search, task prioritization, expertise discovery, personalized recommendations, and advanced analytics to identify trends and actionable insights. Experimental results demonstrate its success in the discovery of expertise, task management, and data-driven decision making. By integrating LLMs with knowledge graphs, this solution bridges disconnected systems and delivers intelligent analytics-powered enterprise tools. 

**Abstract (ZH)**: 企业内部孤立的数据孤岛阻碍了可操作洞察的提取，影响产品开发、客户互动、会议准备及基于分析的决策效率。本文提出了一种框架，利用大型语言模型（LLMs）将各种数据源统一成一个综合性的活动中心知识图谱。该框架自动化执行实体提取、关系推理和语义 enrichment 等任务，支持不同数据类型（如电子邮件、日历、聊天记录、文档和日志）的高级查询、推理与分析。该解决方案具有企业灵活性，支持上下文搜索、任务优先级排序、专家发现、个性化推荐以及高级分析，以识别趋势和可操作洞察。实验结果表明，该框架在专家发现、任务管理和数据驱动决策方面取得了成功。通过将大型语言模型与知识图谱集成，该解决方案连接了孤立的系统，并提供了智能分析驱动的企业工具。 

---
# Boundary Prompting: Elastic Urban Region Representation via Graph-based Spatial Tokenization 

**Title (ZH)**: 边界提示：基于图的空间词元化弹性城市区域表示 

**Authors**: Haojia Zhu, Jiahui Jin, Dong Kan, Rouxi Shen, Ruize Wang, Xiangguo Sun, Jinghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07991)  

**Abstract**: Urban region representation is essential for various applications such as urban planning, resource allocation, and policy development. Traditional methods rely on fixed, predefined region boundaries, which fail to capture the dynamic and complex nature of real-world urban areas. In this paper, we propose the Boundary Prompting Urban Region Representation Framework (BPURF), a novel approach that allows for elastic urban region definitions. BPURF comprises two key components: (1) A spatial token dictionary, where urban entities are treated as tokens and integrated into a unified token graph, and (2) a region token set representation model which utilize token aggregation and a multi-channel model to embed token sets corresponding to region boundaries. Additionally, we propose fast token set extraction strategy to enable online token set extraction during training and prompting. This framework enables the definition of urban regions through boundary prompting, supporting varying region boundaries and adapting to different tasks. Extensive experiments demonstrate the effectiveness of BPURF in capturing the complex characteristics of urban regions. 

**Abstract (ZH)**: 基于边界提示的城市区域表示框架（BPURF）：弹性城市区域定义方法 

---
# LLM-based Corroborating and Refuting Evidence Retrieval for Scientific Claim Verification 

**Title (ZH)**: 基于LLM的证据验证与反驳检索在科学断言验证中的应用 

**Authors**: Siyuan Wang, James R. Foulds, Md Osman Gani, Shimei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07937)  

**Abstract**: In this paper, we introduce CIBER (Claim Investigation Based on Evidence Retrieval), an extension of the Retrieval-Augmented Generation (RAG) framework designed to identify corroborating and refuting documents as evidence for scientific claim verification. CIBER addresses the inherent uncertainty in Large Language Models (LLMs) by evaluating response consistency across diverse interrogation probes. By focusing on the behavioral analysis of LLMs without requiring access to their internal information, CIBER is applicable to both white-box and black-box models. Furthermore, CIBER operates in an unsupervised manner, enabling easy generalization across various scientific domains. Comprehensive evaluations conducted using LLMs with varying levels of linguistic proficiency reveal CIBER's superior performance compared to conventional RAG approaches. These findings not only highlight the effectiveness of CIBER but also provide valuable insights for future advancements in LLM-based scientific claim verification. 

**Abstract (ZH)**: 本文介绍了CIBER（基于证据检索的声明调查），这是一种Retrieval-Augmented Generation (RAG)框架的扩展，用于识别支持和反驳文档作为科学声明验证的证据。CIBER通过评估响应一致性来应对大型语言模型（LLMs）中固有的不确定性，着眼于LLMs的行为分析，无需访问其内部信息，因此适用于白盒和黑盒模型。此外，CIBER以无监督的方式运行，能够在各种科学领域中轻松泛化。使用不同语言 proficiency 的LLMs进行的全面评估表明，CIBER在科学声明验证方面的性能优于传统的RAG方法。这些研究结果不仅突显了CIBER的有效性，还为未来基于LLM的科学声明验证的发展提供了宝贵的见解。 

---
# The StudyChat Dataset: Student Dialogues With ChatGPT in an Artificial Intelligence Course 

**Title (ZH)**: The StudyChat 数据集：人工智能课程中学生与ChatGPT的对话 

**Authors**: Hunter McNichols, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07928)  

**Abstract**: The widespread availability of large language models (LLMs), such as ChatGPT, has significantly impacted education, raising both opportunities and challenges. Students can frequently interact with LLM-powered, interactive learning tools, but their usage patterns need to be analyzed to ensure ethical usage of these tools. To better understand how students interact with LLMs in an academic setting, we introduce \textbf{StudyChat}, a publicly available dataset capturing real-world student interactions with an LLM-powered tutoring chatbot in a semester-long, university-level artificial intelligence (AI) course. We deploy a web application that replicates ChatGPT's core functionalities, and use it to log student interactions with the LLM while working on programming assignments. We collect 1,197 conversations, which we annotate using a dialogue act labeling schema inspired by observed interaction patterns and prior research. Additionally, we analyze these interactions, highlight behavioral trends, and analyze how specific usage patterns relate to course outcomes. \textbf{StudyChat} provides a rich resource for the learning sciences and AI in education communities, enabling further research into the evolving role of LLMs in education. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT的广泛应用显著影响了教育领域，带来了机遇与挑战。学生可以频繁地与基于LLM的交互式学习工具互动，但需要分析其使用模式以确保这些工具的伦理使用。为了更好地理解学生在学术环境中与LLM的互动方式，我们介绍了\textbf{StudyChat}，这是一个公开的数据集，记录了一个学期的人工智能（AI）课程中学生与基于LLM的辅导聊天机器人的互动情况。我们部署了一个网络应用，复制了ChatGPT的核心功能，并使用它来记录学生在完成编程作业时与LLM的互动日志。我们收集了1,197次对话，并使用基于观察到的互动模式和先前研究的对话行为标注方案对其进行标注。此外，我们分析了这些互动，强调了行为趋势，并分析了特定的使用模式与课程结果之间的关系。\textbf{StudyChat}为学习科学和教育中的AI社区提供了丰富的资源，有助于进一步研究LLMs在教育中的 evolving角色。 

---
# BEARCUBS: A benchmark for computer-using web agents 

**Title (ZH)**: BEARCUBS: 一种用于计算机使用网页代理的基准测试 

**Authors**: Yixiao Song, Katherine Thai, Chau Minh Pham, Yapei Chang, Mazin Nadaf, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.07919)  

**Abstract**: Modern web agents possess computer use abilities that allow them to interact with webpages by sending commands to a virtual keyboard and mouse. While such agents have considerable potential to assist human users with complex tasks, evaluating their capabilities in real-world settings poses a major challenge. To this end, we introduce BEARCUBS, a "small but mighty" benchmark of 111 information-seeking questions designed to evaluate a web agent's ability to search, browse, and identify factual information from the web. Unlike prior web agent benchmarks, solving BEARCUBS requires (1) accessing live web content rather than synthetic or simulated pages, which captures the unpredictability of real-world web interactions; and (2) performing a broad range of multimodal interactions (e.g., video understanding, 3D navigation) that cannot be bypassed via text-based workarounds. Each question in BEARCUBS has a corresponding short, unambiguous answer and a human-validated browsing trajectory, allowing for transparent evaluation of agent performance and strategies. A human study confirms that BEARCUBS questions are solvable but non-trivial (84.7% human accuracy), revealing search inefficiencies and domain knowledge gaps as common failure points. By contrast, state-of-the-art computer-using agents underperform, with the best-scoring system (OpenAI's Operator) reaching only 24.3% accuracy. These results highlight critical areas for improvement, including reliable source selection and more powerful multimodal capabilities. To facilitate future research, BEARCUBS will be updated periodically to replace invalid or contaminated questions, keeping the benchmark fresh for future generations of web agents. 

**Abstract (ZH)**: 现代网络代理具有通过发送命令控制虚拟键盘和鼠标来与网页交互的能力。尽管这些代理在协助用户完成复杂任务方面具有巨大的潜力，但在实际环境中评估其能力是一项重大挑战。为了解决这一问题，我们引入了BEARCUBS基准测试，这是一个包含111个信息检索问题的“小巧但强大”的评估体系，旨在评估网络代理在搜索、浏览和从网络中识别事实信息方面的能力。与之前的网络代理基准测试不同，解决BEARCUBS需要访问实际网页内容而非合成或模拟页面，以捕捉实际世界网络交互的不可预测性；并且需要进行多种模态的交互（例如，视频理解、3D导航），这些交互无法通过基于文本的变通办法绕过。每个BEARCUBS问题都有一个对应的具体和明确的答案以及经过人工验证的浏览路径，这使得代理性能和策略的透明评估成为可能。人类研究证实，BEARCUBS问题是可解但非简单的（人类准确率为84.7%），揭示了搜索效率低下和领域知识不足是常见的失败点。相比之下，最先进的计算机使用代理的表现不尽如人意，得分最高的系统（OpenAI的Operator）的准确率仅为24.3%。这些结果强调了可靠的信息源选择和更强大的多模态能力等关键改进领域。为了促进未来的研究，BEARCUBS将定期更新，替换无效或污染的问题，使基准测试保持新鲜，以适应未来一代的网络代理。 

---
# Demystifying the Accuracy-Interpretability Trade-Off: A Case Study of Inferring Ratings from Reviews 

**Title (ZH)**: 揭开准确性和可解释性权衡的神秘面纱：从评价推断评分的案例研究 

**Authors**: Pranjal Atrey, Michael P. Brundage, Min Wu, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2503.07914)  

**Abstract**: Interpretable machine learning models offer understandable reasoning behind their decision-making process, though they may not always match the performance of their black-box counterparts. This trade-off between interpretability and model performance has sparked discussions around the deployment of AI, particularly in critical applications where knowing the rationale of decision-making is essential for trust and accountability. In this study, we conduct a comparative analysis of several black-box and interpretable models, focusing on a specific NLP use case that has received limited attention: inferring ratings from reviews. Through this use case, we explore the intricate relationship between the performance and interpretability of different models. We introduce a quantitative score called Composite Interpretability (CI) to help visualize the trade-off between interpretability and performance, particularly in the case of composite models. Our results indicate that, in general, the learning performance improves as interpretability decreases, but this relationship is not strictly monotonic, and there are instances where interpretable models are more advantageous. 

**Abstract (ZH)**: 可解释的机器学习模型提供可理解的决策推理过程，尽管它们的性能可能不如其黑盒 counterparts。在这权衡可解释性和模型性能的讨论中，特别是在需要了解决策理由以建立信任和问责制的关键应用中，AI 的部署受到了广泛关注。在本研究中，我们对几种黑盒和可解释模型进行了比较分析，重点关注一个受到较少关注的自然语言处理应用案例：从评论推断评分。通过这一应用案例，我们探索了不同模型的性能与可解释性之间的复杂关系。我们引入了一个名为综合可解释性（CI）的量化评分，以帮助可视化可解释性和性能之间的权衡，特别是在复合模型的情况下。我们的结果显示，总体而言，随着可解释性的降低，学习性能会提高，但这种关系并非严格单调，存在可解释模型更为有利的情况。 

---
# Actual Causation and Nondeterministic Causal Models 

**Title (ZH)**: 实际因果关系与非确定性因果模型 

**Authors**: Sander Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07849)  

**Abstract**: In (Beckers, 2025) I introduced nondeterministic causal models as a generalization of Pearl's standard deterministic causal models. I here take advantage of the increased expressivity offered by these models to offer a novel definition of actual causation (that also applies to deterministic models). Instead of motivating the definition by way of (often subjective) intuitions about examples, I proceed by developing it based entirely on the unique function that it can fulfil in communicating and learning a causal model. First I generalize the more basic notion of counterfactual dependence, second I show how this notion has a vital role to play in the logic of causal discovery, third I introduce the notion of a structural simplification of a causal model, and lastly I bring both notions together in my definition of actual causation. Although novel, the resulting definition arrives at verdicts that are almost identical to those of my previous definition (Beckers, 2021, 2022). 

**Abstract (ZH)**: 在贝克尔斯（2025）中，我介绍了 nondeterministic 负因果模型作为佩尔标准确定性因果模型的推广。本文利用这些模型提供的增强表达能力，提出了一种新的实际因果定义（该定义也适用于确定性模型）。我并非通过（往往是主观的）关于示例的直觉来动机该定义，而是通过完全基于它在因果模型通信和学习中所扮演的独特角色来发展这一定义。首先，我将更基本的反事实依赖性概念一般化，其次，我展示了这一概念在因果发现逻辑中起着至关重要的作用，第三，我引入了因果模型结构简化的概念，最后，我将这两种概念结合起来，提出我的实际因果定义。尽管是新颖的，但由此得出的定义与我之前的定义（贝克尔斯，2021，2022）得出的结论几乎相同。 

---
# Safe Explicable Policy Search 

**Title (ZH)**: 安全可解释的策略搜索 

**Authors**: Akkamahadevi Hanni, Jonathan Montaño, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07848)  

**Abstract**: When users work with AI agents, they form conscious or subconscious expectations of them. Meeting user expectations is crucial for such agents to engage in successful interactions and teaming. However, users may form expectations of an agent that differ from the agent's planned behaviors. These differences lead to the consideration of two separate decision models in the planning process to generate explicable behaviors. However, little has been done to incorporate safety considerations, especially in a learning setting. We present Safe Explicable Policy Search (SEPS), which aims to provide a learning approach to explicable behavior generation while minimizing the safety risk, both during and after learning. We formulate SEPS as a constrained optimization problem where the agent aims to maximize an explicability score subject to constraints on safety and a suboptimality criterion based on the agent's model. SEPS innovatively combines the capabilities of Constrained Policy Optimization and Explicable Policy Search. We evaluate SEPS in safety-gym environments and with a physical robot experiment to show that it can learn explicable behaviors that adhere to the agent's safety requirements and are efficient. Results show that SEPS can generate safe and explicable behaviors while ensuring a desired level of performance w.r.t. the agent's objective, and has real-world relevance in human-AI teaming. 

**Abstract (ZH)**: Safe Explicable Policy Search：在确保安全性的前提下生成可解释行为的方法 

---
# RefactorBench: Evaluating Stateful Reasoning in Language Agents Through Code 

**Title (ZH)**: RefactorBench: 通过代码评估语言代理中的状态推理性能 

**Authors**: Dhruv Gautam, Spandan Garg, Jinu Jang, Neel Sundaresan, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2503.07832)  

**Abstract**: Recent advances in language model (LM) agents and function calling have enabled autonomous, feedback-driven systems to solve problems across various digital domains. To better understand the unique limitations of LM agents, we introduce RefactorBench, a benchmark consisting of 100 large handcrafted multi-file refactoring tasks in popular open-source repositories. Solving tasks within RefactorBench requires thorough exploration of dependencies across multiple files and strong adherence to relevant instructions. Every task is defined by 3 natural language instructions of varying specificity and is mutually exclusive, allowing for the creation of longer combined tasks on the same repository. Baselines on RefactorBench reveal that current LM agents struggle with simple compositional tasks, solving only 22% of tasks with base instructions, in contrast to a human developer with short time constraints solving 87%. Through trajectory analysis, we identify various unique failure modes of LM agents, and further explore the failure mode of tracking past actions. By adapting a baseline agent to condition on representations of state, we achieve a 43.9% improvement in solving RefactorBench tasks. We further extend our state-aware approach to encompass entire digital environments and outline potential directions for future research. RefactorBench aims to support the study of LM agents by providing a set of real-world, multi-hop tasks within the realm of code. 

**Abstract (ZH)**: Recent Advances in Language Model Agents and Function Calling Have Enabled Autonomous, Feedback-Driven Systems to Solve Problems Across Various Digital Domains: Introducing RefactorBench，一个由100个大型手工编写的多文件重构任务组成的基准，来自流行开源仓库。 

---
# Efficient Neural Clause-Selection Reinforcement 

**Title (ZH)**: 高效神经子句选择强化学习 

**Authors**: Martin Suda  

**Link**: [PDF](https://arxiv.org/pdf/2503.07792)  

**Abstract**: Clause selection is arguably the most important choice point in saturation-based theorem proving. Framing it as a reinforcement learning (RL) task is a way to challenge the human-designed heuristics of state-of-the-art provers and to instead automatically evolve -- just from prover experiences -- their potentially optimal replacement. In this work, we present a neural network architecture for scoring clauses for clause selection that is powerful yet efficient to evaluate. Following RL principles to make design decisions, we integrate the network into the Vampire theorem prover and train it from successful proof attempts. An experiment on the diverse TPTP benchmark finds the neurally guided prover improve over a baseline strategy, from which it initially learns--in terms of the number of in-training-unseen problems solved under a practically relevant, short CPU instruction limit--by 20%. 

**Abstract (ZH)**: 基于饱和定理证明的句法选择是最重要的决策点。将其作为一个强化学习任务可以挑战最先进的证明器设计的人工启发式方法，并通过证明器的经验自动进化可能最优的替代方法。本文提出了一种高效评价的神经网络架构用于句法选择分数计算。遵循强化学习原理进行设计决策，我们将网络集成到Vampire证明器中，并从成功的证明尝试中对其进行训练。一项针对TPTP基准的实验发现，神经引导的证明器在缩短的CPU指令限制下，相较于基准策略，在解决训练中未见过的问题数量上提高了20%。 

---
# Sensemaking in Novel Environments: How Human Cognition Can Inform Artificial Agents 

**Title (ZH)**: 新环境中意义构建：人类认知如何指导人工代理 

**Authors**: Robert E. Patterson, Regina Buccello-Stout, Mary E. Frame, Anna M. Maresca, Justin Nelson, Barbara Acker-Mills, Erica Curtis, Jared Culbertson, Kevin Schmidt, Scott Clouse, Steve Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07783)  

**Abstract**: One of the most vital cognitive skills to possess is the ability to make sense of objects, events, and situations in the world. In the current paper, we offer an approach for creating artificially intelligent agents with the capacity for sensemaking in novel environments. Objectives: to present several key ideas: (1) a novel unified conceptual framework for sensemaking (which includes the existence of sign relations embedded within and across frames); (2) interaction among various content-addressable, distributed-knowledge structures via shared attributes (whose net response would represent a synthesized object, event, or situation serving as a sign for sensemaking in a novel environment). Findings: we suggest that attributes across memories can be shared and recombined in novel ways to create synthesized signs, which can denote certain outcomes in novel environments (i.e., sensemaking). 

**Abstract (ZH)**: 掌握对世界中的物体、事件和情境进行理解的认知能力是极其重要的。本文提出了一个创造能够在新型环境中进行理解的人工智能代理的方法。目标：提出几个关键概念：（1）一种新颖统一的理解概念框架（其中包括嵌套和跨域的关系符号）；（2）通过共享属性来促进各种内容可寻址、分布式知识结构之间的交互（这些交互的综合响应将代表一种合成的目标、事件或情况，作为新型环境中的符号，用于理解）。发现：我们建议，在记忆中的属性可以以新颖的方式共享和重组以生成合成符号，这些符号能够表示新型环境中的某些结果（即理解）。 

---
# Fully Autonomous Programming using Iterative Multi-Agent Debugging with Large Language Models 

**Title (ZH)**: 完全自主编程：基于大型语言模型的迭代多Agent调试 

**Authors**: Anastasiia Grishina, Vadim Liventsev, Aki Härmä, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07693)  

**Abstract**: Program synthesis with Large Language Models (LLMs) suffers from a "near-miss syndrome": the generated code closely resembles a correct solution but fails unit tests due to minor errors. We address this with a multi-agent framework called Synthesize, Execute, Instruct, Debug, and Repair (SEIDR). Effectively applying SEIDR to instruction-tuned LLMs requires determining (a) optimal prompts for LLMs, (b) what ranking algorithm selects the best programs in debugging rounds, and (c) balancing the repair of unsuccessful programs with the generation of new ones. We empirically explore these trade-offs by comparing replace-focused, repair-focused, and hybrid debug strategies. We also evaluate lexicase and tournament selection to rank candidates in each generation. On Program Synthesis Benchmark 2 (PSB2), our framework outperforms both conventional use of OpenAI Codex without a repair phase and traditional genetic programming approaches. SEIDR outperforms the use of an LLM alone, solving 18 problems in C++ and 20 in Python on PSB2 at least once across experiments. To assess generalizability, we employ GPT-3.5 and Llama 3 on the PSB2 and HumanEval-X benchmarks. Although SEIDR with these models does not surpass current state-of-the-art methods on the Python benchmarks, the results on HumanEval-C++ are promising. SEIDR with Llama 3-8B achieves an average pass@100 of 84.2%. Across all SEIDR runs, 163 of 164 problems are solved at least once with GPT-3.5 in HumanEval-C++, and 162 of 164 with the smaller Llama 3-8B. We conclude that SEIDR effectively overcomes the near-miss syndrome in program synthesis with LLMs. 

**Abstract (ZH)**: 用大型语言模型进行程序合成遭遇“近似症候群”：生成的代码与正确解相似但在单元测试中因小错误而失败。我们提出了一个名为Synthesize, Execute, Instruct, Debug, and Repair (SEIDR)的多智能体框架来解决这一问题。将SEIDR有效应用于指令调优的LLMs需要确定（a）LLMs的最佳提示，（b）调试轮次中选择最佳程序的排名算法，以及（c）平衡不成功的程序修复与新程序生成之间的关系。我们通过比较注重替换、注重修复和混合调试策略来实证探索这些权衡。我们还评估了lexicase和淘汰赛选择算法在每一代中排名候选程序的性能。在Program Synthesis Benchmark 2 (PSB2)上，我们的框架优于没有修复阶段的传统使用OpenAI Codex的方法和传统遗传编程方法。SEIDR优于单独使用LLM，解决了PSB2上的18个C++问题和20个Python问题。为评估泛化能力，我们在PSB2和HumanEval-X基准上使用GPT-3.5和Llama 3。尽管SEIDR与这些模型在Python基准上的表现未能超过当前最先进的方法，但对HumanEval-C++基准的结果令人振奋。使用Llama 3-8B的SEIDR实现了平均pass@100为84.2%的结果。在所有SEIDR运行中，GPT-3.5在HumanEval-C++上解决了164个问题中的163个，Llama 3-8B在164个问题中解决了162个。我们得出结论，SEIDR有效地克服了用LLMs进行程序合成中的近似症候群。 

---
# Research on Superalignment Should Advance Now with Parallel Optimization of Competence and Conformity 

**Title (ZH)**: superalignment研究应现在通过能力和一致性并行优化来推进 

**Authors**: HyunJin Kim, Xiaoyuan Yi, Jing Yao, Muhua Huang, JinYeong Bak, James Evans, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.07660)  

**Abstract**: The recent leap in AI capabilities, driven by big generative models, has sparked the possibility of achieving Artificial General Intelligence (AGI) and further triggered discussions on Artificial Superintelligence (ASI), a system surpassing all humans across all domains. This gives rise to the critical research question of: If we realize ASI, how do we align it with human values, ensuring it benefits rather than harms human society, a.k.a., the Superalignment problem. Despite ASI being regarded by many as solely a hypothetical concept, in this paper, we argue that superalignment is achievable and research on it should advance immediately, through simultaneous and alternating optimization of task competence and value conformity. We posit that superalignment is not merely a safeguard for ASI but also necessary for its realization. To support this position, we first provide a formal definition of superalignment rooted in the gap between capability and capacity and elaborate on our argument. Then we review existing paradigms, explore their interconnections and limitations, and illustrate a potential path to superalignment centered on two fundamental principles. We hope this work sheds light on a practical approach for developing the value-aligned next-generation AI, garnering greater benefits and reducing potential harms for humanity. 

**Abstract (ZH)**: 最近由大规模生成模型推动的AI能力飞跃激发了实现人工通用智能（AGI）的可能性，并进一步引发了关于人工超级智能（ASI）的讨论，ASI是涵盖所有领域超越所有人类的系统。这引发了关键的研究问题：如果我们实现ASI，如何使其与人类价值观保持一致，确保它造福而非危害人类社会，即超对齐问题。尽管许多人都认为ASI仅是一种假设性概念，但在本文中，我们argue超对齐是可实现的，并且应该立即通过任务能力和价值一致性的同时和交替优化来推进对其的研究。我们提出，超对齐不仅是ASI的安全措施，也是其实现的必要条件。为了支持这一观点，我们首先提供了基于能力与容量差距的超对齐的形式化定义，并阐述了我们的论点。然后我们回顾了现有的范式，探讨了它们的相互联系和局限性，并展示了以两个基本原则为中心的超对齐潜在路径。我们希望这项工作能为开发价值对齐的下一代AI提供实用的途径，从而带来更大的利益并减少潜在的危害。 

---
# Impact of Level 2/3 Automated Driving Technology on Road Work Zone Safety 

**Title (ZH)**: Level 2/3自动驾驶技术对道路施工区安全的影响 

**Authors**: Zhepu Xu, Ziyi Song, Yupu Dong, Peiyan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07634)  

**Abstract**: As China's road network enters the maintenance era, work zones will become a common sight on the roads. With the development of automated driving, vehicles equipped with Level 2/3 automated driving capabilities will also become a common presence on the roads. When these vehicles pass through work zones, automated driving may disengage, which can have complex effects on traffic safety. This paper explores the impact of Level 2/3 automated driving technology on road safety in high-speed highway work zone environments. Through microscopic traffic simulation method and using full-type traffic conflict technique, factors such as market penetration rate (MPR), traffic volume level, disengagement threshold, and driver takeover style are studied to understand their impact on work zone safety. The study found that the impact of automated driving technology on work zone safety is complex. Disengagement of automated vehicles in work zones reduces the proportion of vehicles that can maintain automated driving status. If takeover is not timely or adequate, it can easily lead to new traffic conflicts. Different factors have varying degrees of impact on work zone safety. Increasing MPR helps reduce the occurrence of single-vehicle conflicts, but it also increases the possibility of multi-vehicle conflicts. Therefore, future research and improvement directions should focus on optimizing the disengagement detection and takeover mechanisms of automated driving systems. 

**Abstract (ZH)**: 中国道路交通维护时代的自动驾驶技术对高速公路工作区交通安全的影响研究 

---
# Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents 

**Title (ZH)**: 困惑度陷阱：基于PLM的检索系统高估低困惑度文档 

**Authors**: Haoyu Wang, Sunhao Dai, Haiyuan Zhao, Liang Pang, Xiao Zhang, Gang Wang, Zhenhua Dong, Jun Xu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08684)  

**Abstract**: Previous studies have found that PLM-based retrieval models exhibit a preference for LLM-generated content, assigning higher relevance scores to these documents even when their semantic quality is comparable to human-written ones. This phenomenon, known as source bias, threatens the sustainable development of the information access ecosystem. However, the underlying causes of source bias remain unexplored. In this paper, we explain the process of information retrieval with a causal graph and discover that PLM-based retrievers learn perplexity features for relevance estimation, causing source bias by ranking the documents with low perplexity higher. Theoretical analysis further reveals that the phenomenon stems from the positive correlation between the gradients of the loss functions in language modeling task and retrieval task. Based on the analysis, a causal-inspired inference-time debiasing method is proposed, called Causal Diagnosis and Correction (CDC). CDC first diagnoses the bias effect of the perplexity and then separates the bias effect from the overall estimated relevance score. Experimental results across three domains demonstrate the superior debiasing effectiveness of CDC, emphasizing the validity of our proposed explanatory framework. Source codes are available at this https URL. 

**Abstract (ZH)**: 基于PLM的检索模型对LLM生成内容的偏好及其来源：因果诊断与纠正(CDC)方法 

---
# CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving 

**Title (ZH)**: CoLMDriver: 基于LLM的谈判促进合作自主驾驶 

**Authors**: Changxing Liu, Genjia Liu, Zijun Wang, Jinchang Yang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08683)  

**Abstract**: Vehicle-to-vehicle (V2V) cooperative autonomous driving holds great promise for improving safety by addressing the perception and prediction uncertainties inherent in single-agent systems. However, traditional cooperative methods are constrained by rigid collaboration protocols and limited generalization to unseen interactive scenarios. While LLM-based approaches offer generalized reasoning capabilities, their challenges in spatial planning and unstable inference latency hinder their direct application in cooperative driving. To address these limitations, we propose CoLMDriver, the first full-pipeline LLM-based cooperative driving system, enabling effective language-based negotiation and real-time driving control. CoLMDriver features a parallel driving pipeline with two key components: (i) an LLM-based negotiation module under an actor-critic paradigm, which continuously refines cooperation policies through feedback from previous decisions of all vehicles; and (ii) an intention-guided waypoint generator, which translates negotiation outcomes into executable waypoints. Additionally, we introduce InterDrive, a CARLA-based simulation benchmark comprising 10 challenging interactive driving scenarios for evaluating V2V cooperation. Experimental results demonstrate that CoLMDriver significantly outperforms existing approaches, achieving an 11% higher success rate across diverse highly interactive V2V driving scenarios. Code will be released on this https URL. 

**Abstract (ZH)**: 车辆到车辆（V2V）协作自动驾驶在通过解决单一代理系统固有的感知和预测不确定性来提高安全性方面具有巨大潜力。然而，传统的协作方法受限于 rigid collaboration protocols 并且在未见过的交互场景泛化能力有限。虽然基于大语言模型（LLM）的方法提供了通用的推理能力，但它们在空间规划方面的挑战和不稳定的推理延迟阻碍了它们在协作驾驶中的直接应用。为了应对这些限制，我们提出了 CoLMDriver，这是首个完整的基于大语言模型的协作驾驶系统，能够实现有效的基于语言的谈判和实时驾驶控制。CoLMDriver 特征包括一个并行驾驶流水线，包含两个关键组件：(i) 一个基于演员-评论家范式的基于大语言模型的谈判模块，通过反馈不断细化合作策略；以及 (ii) 一个意图导向的航点生成器，将谈判结果转化为可执行的航点。此外，我们引入了 InterDrive，这是一个基于 CARLA 的仿真基准测试套件，包含 10 个具有挑战性的交互驾驶场景，用于评估 V2V 协作。实验结果表明，CoLMDriver 在多种高度交互的 V2V 驾驶场景中成功率高出 11%，代码将在此网址发布：https://github.com/alibaba/Qwen-Assistant。 

---
# GarmentCrafter: Progressive Novel View Synthesis for Single-View 3D Garment Reconstruction and Editing 

**Title (ZH)**: GarmentCrafter: 逐帧新颖视角合成在单视角3D服装重建与编辑中的应用 

**Authors**: Yuanhao Wang, Cheng Zhang, Gonçalo Frazão, Jinlong Yang, Alexandru-Eugen Ichim, Thabo Beeler, Fernando De la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2503.08678)  

**Abstract**: We introduce GarmentCrafter, a new approach that enables non-professional users to create and modify 3D garments from a single-view image. While recent advances in image generation have facilitated 2D garment design, creating and editing 3D garments remains challenging for non-professional users. Existing methods for single-view 3D reconstruction often rely on pre-trained generative models to synthesize novel views conditioning on the reference image and camera pose, yet they lack cross-view consistency, failing to capture the internal relationships across different views. In this paper, we tackle this challenge through progressive depth prediction and image warping to approximate novel views. Subsequently, we train a multi-view diffusion model to complete occluded and unknown clothing regions, informed by the evolving camera pose. By jointly inferring RGB and depth, GarmentCrafter enforces inter-view coherence and reconstructs precise geometries and fine details. Extensive experiments demonstrate that our method achieves superior visual fidelity and inter-view coherence compared to state-of-the-art single-view 3D garment reconstruction methods. 

**Abstract (ZH)**: 我们介绍了一种新的方法GarmentCrafter，该方法使非专业人士能够从单张视角图片创建和修改3D服装。尽管近期在图像生成方面的进展促进了2D服装设计，但非专业人士创建和编辑3D服装仍具有挑战性。现有的单视角3D重建方法通常依赖于预训练的生成模型，根据参考图像和相机姿态生成新颖视角，但缺乏跨视角一致性，无法捕捉不同视角之间的内部关系。在本文中，我们通过渐进深度预测和图像变形来近似新颖视角。随后，我们训练一个多视角扩散模型来完成被遮挡和未知的服装区域，并基于 evolving 相机姿态提供信息。通过联合推断RGB和深度，GarmentCrafter 强制执行视角一致性并重建精确的几何结构和细节。广泛实验表明，与现有的单视角3D服装重建方法相比，我们的方法在视觉保真度和视角一致性方面表现出更优的效果。 

---
# AgentOrca: A Dual-System Framework to Evaluate Language Agents on Operational Routine and Constraint Adherence 

**Title (ZH)**: AgentOrca: 一种评估语言代理操作常规和约束遵守的双重系统框架 

**Authors**: Zekun Li, Shinda Huang, Jiangtian Wang, Nathan Zhang, Antonis Antoniades, Wenyue Hua, Kaijie Zhu, Sirui Zeng, William Yang Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08669)  

**Abstract**: As language agents progressively automate critical tasks across domains, their ability to operate within operational constraints and safety protocols becomes essential. While extensive research has demonstrated these agents' effectiveness in downstream task completion, their reliability in following operational procedures and constraints remains largely unexplored. To this end, we present AgentOrca, a dual-system framework for evaluating language agents' compliance with operational constraints and routines. Our framework encodes action constraints and routines through both natural language prompts for agents and corresponding executable code serving as ground truth for automated verification. Through an automated pipeline of test case generation and evaluation across five real-world domains, we quantitatively assess current language agents' adherence to operational constraints. Our findings reveal notable performance gaps among state-of-the-art models, with large reasoning models like o1 demonstrating superior compliance while others show significantly lower performance, particularly when encountering complex constraints or user persuasion attempts. 

**Abstract (ZH)**: 随着语言代理在各个领域逐步自动化关键任务，它们在操作约束和安全协议内的运行能力变得至关重要。虽然大量的研究已经证明了这些代理在下游任务完成上的有效性，但它们遵循操作程序和约束的可靠性仍很大程度上未被探索。为此，我们提出了AgentOrca，一种双系统框架，用于评估语言代理对操作约束和常规程序的遵守情况。该框架通过自然语言提示和相应的可执行代码来编码行动约束和常规程序，后者作为自动验证的标准。通过跨越五个实际领域的自动化测试用例生成和评估管道，我们定量评估当前语言代理遵守操作约束的情况。我们的研究发现，最先进的模型之间存在明显的性能差距，大型推理模型如o1显示出更好的合规性，而其他模型则表现显著较差，尤其是在面对复杂的约束或用户说服尝试时。 

---
# REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder 

**Title (ZH)**: REGEN: 学习紧凑型视频嵌入的（再）生成解码器 

**Authors**: Yitian Zhang, Long Mai, Aniruddha Mahapatra, David Bourgin, Yicong Hong, Jonah Casebeer, Feng Liu, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08665)  

**Abstract**: We present a novel perspective on learning video embedders for generative modeling: rather than requiring an exact reproduction of an input video, an effective embedder should focus on synthesizing visually plausible reconstructions. This relaxed criterion enables substantial improvements in compression ratios without compromising the quality of downstream generative models. Specifically, we propose replacing the conventional encoder-decoder video embedder with an encoder-generator framework that employs a diffusion transformer (DiT) to synthesize missing details from a compact latent space. Therein, we develop a dedicated latent conditioning module to condition the DiT decoder on the encoded video latent embedding. Our experiments demonstrate that our approach enables superior encoding-decoding performance compared to state-of-the-art methods, particularly as the compression ratio increases. To demonstrate the efficacy of our approach, we report results from our video embedders achieving a temporal compression ratio of up to 32x (8x higher than leading video embedders) and validate the robustness of this ultra-compact latent space for text-to-video generation, providing a significant efficiency boost in latent diffusion model training and inference. 

**Abstract (ZH)**: 我们提出了关于学习视频嵌入器进行生成建模的一个新颖视角：有效的嵌入器应当专注于合成视觉上可信的重建，而非精确再现输入视频。这一放松的要求使得在不牺牲下游生成模型质量的前提下，大幅提高压缩比成为可能。具体来说，我们提出用一种结合扩散变压器（DiT）的编码-生成框架来取代传统的编码-解码视频嵌入器，以从紧凑的潜在空间中合成缺失的细节。在此基础上，我们开发了一个专门的潜在条件模块，将编码后的视频潜在嵌入条件化到DiT解码器上。我们的实验表明，与现有的先进方法相比，我们的方法在压缩比增加时能够实现更优的编码-解码性能。为了证明我们方法的效果，我们报告了我们的视频嵌入器实现高达32倍（比领先的视频嵌入器高出8倍）的时间压缩比的结果，并验证了这种超紧凑潜在空间在文本到视频生成中的鲁棒性，从而在潜在扩散模型的训练和推理方面提供了显著的效率提升。 

---
# MEAT: Multiview Diffusion Model for Human Generation on Megapixels with Mesh Attention 

**Title (ZH)**: MEAT：兆像素人体生成的多视图扩散模型与网格注意力 

**Authors**: Yuhan Wang, Fangzhou Hong, Shuai Yang, Liming Jiang, Wayne Wu, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08664)  

**Abstract**: Multiview diffusion models have shown considerable success in image-to-3D generation for general objects. However, when applied to human data, existing methods have yet to deliver promising results, largely due to the challenges of scaling multiview attention to higher resolutions. In this paper, we explore human multiview diffusion models at the megapixel level and introduce a solution called mesh attention to enable training at 1024x1024 resolution. Using a clothed human mesh as a central coarse geometric representation, the proposed mesh attention leverages rasterization and projection to establish direct cross-view coordinate correspondences. This approach significantly reduces the complexity of multiview attention while maintaining cross-view consistency. Building on this foundation, we devise a mesh attention block and combine it with keypoint conditioning to create our human-specific multiview diffusion model, MEAT. In addition, we present valuable insights into applying multiview human motion videos for diffusion training, addressing the longstanding issue of data scarcity. Extensive experiments show that MEAT effectively generates dense, consistent multiview human images at the megapixel level, outperforming existing multiview diffusion methods. 

**Abstract (ZH)**: 多视图扩散模型在生成高分辨率人体数据方面的研究：MEAT模型探索 

---
# Generating Robot Constitutions & Benchmarks for Semantic Safety 

**Title (ZH)**: 生成机器人宪法与语义安全基准 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Alex Irpan, Dmitry Kalashnikov, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.08663)  

**Abstract**: Until recently, robotics safety research was predominantly about collision avoidance and hazard reduction in the immediate vicinity of a robot. Since the advent of large vision and language models (VLMs), robots are now also capable of higher-level semantic scene understanding and natural language interactions with humans. Despite their known vulnerabilities (e.g. hallucinations or jail-breaking), VLMs are being handed control of robots capable of physical contact with the real world. This can lead to dangerous behaviors, making semantic safety for robots a matter of immediate concern. Our contributions in this paper are two fold: first, to address these emerging risks, we release the ASIMOV Benchmark, a large-scale and comprehensive collection of datasets for evaluating and improving semantic safety of foundation models serving as robot brains. Our data generation recipe is highly scalable: by leveraging text and image generation techniques, we generate undesirable situations from real-world visual scenes and human injury reports from hospitals. Secondly, we develop a framework to automatically generate robot constitutions from real-world data to steer a robot's behavior using Constitutional AI mechanisms. We propose a novel auto-amending process that is able to introduce nuances in written rules of behavior; this can lead to increased alignment with human preferences on behavior desirability and safety. We explore trade-offs between generality and specificity across a diverse set of constitutions of different lengths, and demonstrate that a robot is able to effectively reject unconstitutional actions. We measure a top alignment rate of 84.3% on the ASIMOV Benchmark using generated constitutions, outperforming no-constitution baselines and human-written constitutions. Data is available at this http URL 

**Abstract (ZH)**: 直到最近，机器人安全研究主要集中在避免碰撞和减少机器人周围区域的潜在危险。随着大型视觉和语言模型（VLMs）的出现，机器人现在也能够进行更高层次的语义场景理解，并与人类进行自然语言交互。尽管这些模型存在已知的漏洞（例如幻觉或逃逸），但它们现在正在控制能够与真实世界进行物理互动的机器人。这可能导致危险的行为，使机器人的语义安全性成为一个迫切需要关注的问题。本文的主要贡献有两个方面：首先，为了解决这些新兴的风险，我们发布了ASIMOV基准，这是一个大规模且全面的数据集集合，用于评估和提高作为机器人大脑的基础模型的语义安全性。我们的数据生成配方具有很高的可扩展性：通过利用文本和图像生成技术，我们从现实世界的视觉场景和医院的人身伤害报告中生成了不良情况。其次，我们开发了一个框架，可以从现实世界数据自动生成机器人的宪法，使用宪法AI机制引导机器人的行为。我们提出了一种新颖的自动修正过程，能够引入行为规则中的细微差异；这可以提高机器人的行为偏好和安全性的匹配度。我们探讨了不同长度宪法的一般性和特异性之间的权衡，并证明机器人能够有效拒绝不符合宪法的行为。使用生成的宪法在ASIMOV基准上的顶级对齐率为84.3%，优于无宪法基线和人类撰写的宪法。数据可通过以下链接获取。 

---
# Exploring the Word Sense Disambiguation Capabilities of Large Language Models 

**Title (ZH)**: 探索大型语言模型的词义消歧能力 

**Authors**: Pierpaolo Basile, Lucia Siciliani, Elio Musacchio, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2503.08662)  

**Abstract**: Word Sense Disambiguation (WSD) is a historical task in computational linguistics that has received much attention over the years. However, with the advent of Large Language Models (LLMs), interest in this task (in its classical definition) has decreased. In this study, we evaluate the performance of various LLMs on the WSD task. We extend a previous benchmark (XL-WSD) to re-design two subtasks suitable for LLM: 1) given a word in a sentence, the LLM must generate the correct definition; 2) given a word in a sentence and a set of predefined meanings, the LLM must select the correct one. The extended benchmark is built using the XL-WSD and BabelNet. The results indicate that LLMs perform well in zero-shot learning but cannot surpass current state-of-the-art methods. However, a fine-tuned model with a medium number of parameters outperforms all other models, including the state-of-the-art. 

**Abstract (ZH)**: 大规模语言模型在词义消歧任务中的性能研究 

---
# Exploiting Instruction-Following Retrievers for Malicious Information Retrieval 

**Title (ZH)**: 利用指令遵循检索器进行恶意信息检索 

**Authors**: Parishad BehnamGhader, Nicholas Meade, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08644)  

**Abstract**: Instruction-following retrievers have been widely adopted alongside LLMs in real-world applications, but little work has investigated the safety risks surrounding their increasing search capabilities. We empirically study the ability of retrievers to satisfy malicious queries, both when used directly and when used in a retrieval augmented generation-based setup. Concretely, we investigate six leading retrievers, including NV-Embed and LLM2Vec, and find that given malicious requests, most retrievers can (for >50% of queries) select relevant harmful passages. For example, LLM2Vec correctly selects passages for 61.35% of our malicious queries. We further uncover an emerging risk with instruction-following retrievers, where highly relevant harmful information can be surfaced by exploiting their instruction-following capabilities. Finally, we show that even safety-aligned LLMs, such as Llama3, can satisfy malicious requests when provided with harmful retrieved passages in-context. In summary, our findings underscore the malicious misuse risks associated with increasing retriever capability. 

**Abstract (ZH)**: 指令遵循检索器在现实应用中广泛采用，与大语言模型一同，但鲜有研究探讨其日益增强的搜索能力所带来的安全风险。我们实证研究了检索器满足恶意查询的能力，无论是直接使用还是在检索增强生成设置中使用。具体而言，我们调查了包括NV-Embed和LLM2Vec在内的六种领先检索器，发现大多数检索器能够在超过50%的查询中选择相关有害段落。例如，LLM2Vec正确选择段落的比例为61.35%。进一步研究揭示了指令遵循检索器的一种新兴风险，即通过利用其指令遵循能力，可以暴露高度相关且有害的信息。最后，我们展示了即使像Llama3这样的安全对齐的大语言模型，在提供有害检索段落后，也能满足恶意请求。总之，我们的研究成果强调了检索能力增强带来的恶意滥用风险。 

---
# Rethinking Diffusion Model in High Dimension 

**Title (ZH)**: 重新思考高维空间中的扩散模型 

**Authors**: Zhenxin Zheng, Zhenjie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08643)  

**Abstract**: Curse of Dimensionality is an unavoidable challenge in statistical probability models, yet diffusion models seem to overcome this limitation, achieving impressive results in high-dimensional data generation. Diffusion models assume that they can learn the statistical properties of the underlying probability distribution, enabling sampling from this distribution to generate realistic samples. But is this really how they work? To address this question, this paper conducts a detailed analysis of the objective function and inference methods of diffusion models, leading to several important conclusions that help answer the above question: 1) In high-dimensional sparse scenarios, the target of the objective function fitting degrades from a weighted sum of multiple samples to a single sample. 2) The mainstream inference methods can all be represented within a simple unified framework, without requiring statistical concepts such as Markov chains and SDEs. 3) Guided by this simple framework, more efficient inference methods can be discovered. 

**Abstract (ZH)**: 高维稀疏场景下的维度灾难以统计概率模型为例是一个不可避免的挑战，而扩散模型似乎克服了这一限制，在高维数据生成中取得了令人印象深刻的成果。扩散模型假设它们能够学习潜在概率分布的统计性质，从而使从该分布中采样以生成现实样本成为可能。但这真的就是这样工作的吗？为了解决这个问题，本文对该类模型的目标函数和推理方法进行了详细分析，得出了一些重要结论，有助于回答上述问题：1）在高维稀疏场景下，目标函数拟合的目标从多个样本的加权和退化为单个样本。2）主流的推理方法都可以在一個简单的统一框架内表示，无需使用马尔可夫链和SDE等统计概念。3）在这一简单框架的指导下，可以发现更高效的推理方法。 

---
# YuE: Scaling Open Foundation Models for Long-Form Music Generation 

**Title (ZH)**: YuE: 扩展开放基础模型以进行长篇音乐生成 

**Authors**: Ruibin Yuan, Hanfeng Lin, Shuyue Guo, Ge Zhang, Jiahao Pan, Yongyi Zang, Haohe Liu, Yiming Liang, Wenye Ma, Xingjian Du, Xinrun Du, Zhen Ye, Tianyu Zheng, Yinghao Ma, Minghao Liu, Zeyue Tian, Ziya Zhou, Liumeng Xue, Xingwei Qu, Yizhi Li, Shangda Wu, Tianhao Shen, Ziyang Ma, Jun Zhan, Chunhui Wang, Yatian Wang, Xiaowei Chi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Shansong Liu, Lingrui Mei, Peng Li, Junjie Wang, Jianwei Yu, Guojian Pang, Xu Li, Zihao Wang, Xiaohuan Zhou, Lijun Yu, Emmanouil Benetos, Yong Chen, Chenghua Lin, Xie Chen, Gus Xia, Zhaoxiang Zhang, Chao Zhang, Wenhu Chen, Xinyu Zhou, Xipeng Qiu, Roger Dannenberg, Jiaheng Liu, Jian Yang, Wenhao Huang, Wei Xue, Xu Tan, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08638)  

**Abstract**: We tackle the task of long-form music generation--particularly the challenging \textbf{lyrics-to-song} problem--by introducing YuE, a family of open foundation models based on the LLaMA2 architecture. Specifically, YuE scales to trillions of tokens and generates up to five minutes of music while maintaining lyrical alignment, coherent musical structure, and engaging vocal melodies with appropriate accompaniment. It achieves this through (1) track-decoupled next-token prediction to overcome dense mixture signals, (2) structural progressive conditioning for long-context lyrical alignment, and (3) a multitask, multiphase pre-training recipe to converge and generalize. In addition, we redesign the in-context learning technique for music generation, enabling versatile style transfer (e.g., converting Japanese city pop into an English rap while preserving the original accompaniment) and bidirectional generation. Through extensive evaluation, we demonstrate that YuE matches or even surpasses some of the proprietary systems in musicality and vocal agility. In addition, fine-tuning YuE enables additional controls and enhanced support for tail languages. Furthermore, beyond generation, we show that YuE's learned representations can perform well on music understanding tasks, where the results of YuE match or exceed state-of-the-art methods on the MARBLE benchmark. Keywords: lyrics2song, song generation, long-form, foundation model, music generation 

**Abstract (ZH)**: 长篇音乐生成——特别是具有挑战性的歌词到歌曲问题——我们通过引入基于LLaMA2架构的YuE这一系列开放基础模型来解决该任务。具体而言，YuE可以扩展到万亿级 Tokens，并生成长达五分钟的音乐，同时保持歌词对齐、连贯的音乐结构以及吸引人的伴奏旋律。它通过以下方式实现这一目标：（1）轨道解耦的下一步预测以克服密集混合信号，（2）结构渐进式条件概率以实现长上下文歌词对齐，以及（3）多任务、多阶段预训练配方以实现收敛和泛化。此外，我们重新设计了音乐生成中的在上下文学习技术，使其能够实现多样的风格转换（例如，在保留原伴奏的同时将日式城市流行音乐转换为英文说唱）和双向生成。通过广泛的评估，我们证明YuE在音乐性和歌唱灵活性方面能够匹敌甚至超越一些专有系统。此外，微调YuE能够提供额外的控制并增强对尾部语言的支持。此外，除了生成之外，我们还展示了YuE学习到的表示在音乐理解任务中表现出色，YuE在MARBLE基准上的表现与最先进的方法相当或超越。关键词：歌词到歌曲，歌曲生成，长篇，基础模型，音乐生成。 

---
# Vision Transformer for Intracranial Hemorrhage Classification in CT Scans Using an Entropy-Aware Fuzzy Integral Strategy for Adaptive Scan-Level Decision Fusion 

**Title (ZH)**: 基于熵意识模糊积分策略的自适应扫描级决策融合的CT扫描中颅内出血分类的视觉变换器 

**Authors**: Mehdi Hosseini Chagahi, Niloufar Delfan, Behzad Moshiri, Md. Jalil Piran, Jaber Hatam Parikhan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08609)  

**Abstract**: Intracranial hemorrhage (ICH) is a critical medical emergency caused by the rupture of cerebral blood vessels, leading to internal bleeding within the skull. Accurate and timely classification of hemorrhage subtypes is essential for effective clinical decision-making. To address this challenge, we propose an advanced pyramid vision transformer (PVT)-based model, leveraging its hierarchical attention mechanisms to capture both local and global spatial dependencies in brain CT scans. Instead of processing all extracted features indiscriminately, A SHAP-based feature selection method is employed to identify the most discriminative components, which are then used as a latent feature space to train a boosting neural network, reducing computational complexity. We introduce an entropy-aware aggregation strategy along with a fuzzy integral operator to fuse information across multiple CT slices, ensuring a more comprehensive and reliable scan-level diagnosis by accounting for inter-slice dependencies. Experimental results show that our PVT-based framework significantly outperforms state-of-the-art deep learning architectures in terms of classification accuracy, precision, and robustness. By combining SHAP-driven feature selection, transformer-based modeling, and an entropy-aware fuzzy integral operator for decision fusion, our method offers a scalable and computationally efficient AI-driven solution for automated ICH subtype classification. 

**Abstract (ZH)**: 基于PVT的注意力机制驱动的ICH亚型自动分类方法：融合SHAP特征选择、变压器建模和熵感知模糊积分决策融合 

---
# A Grid Cell-Inspired Structured Vector Algebra for Cognitive Maps 

**Title (ZH)**: 基于网格细胞启发的结构化向量代数认知地图 

**Authors**: Sven Krausse, Emre Neftci, Friedrich T. Sommer, Alpha Renner  

**Link**: [PDF](https://arxiv.org/pdf/2503.08608)  

**Abstract**: The entorhinal-hippocampal formation is the mammalian brain's navigation system, encoding both physical and abstract spaces via grid cells. This system is well-studied in neuroscience, and its efficiency and versatility make it attractive for applications in robotics and machine learning. While continuous attractor networks (CANs) successfully model entorhinal grid cells for encoding physical space, integrating both continuous spatial and abstract spatial computations into a unified framework remains challenging. Here, we attempt to bridge this gap by proposing a mechanistic model for versatile information processing in the entorhinal-hippocampal formation inspired by CANs and Vector Symbolic Architectures (VSAs), a neuro-symbolic computing framework. The novel grid-cell VSA (GC-VSA) model employs a spatially structured encoding scheme with 3D neuronal modules mimicking the discrete scales and orientations of grid cell modules, reproducing their characteristic hexagonal receptive fields. In experiments, the model demonstrates versatility in spatial and abstract tasks: (1) accurate path integration for tracking locations, (2) spatio-temporal representation for querying object locations and temporal relations, and (3) symbolic reasoning using family trees as a structured test case for hierarchical relationships. 

**Abstract (ZH)**: 内嗅-海马复合体是哺乳动物大脑的导航系统，通过网格细胞编码物理和抽象空间。该系统在神经科学中被广泛研究，其高效性和灵活性使其在机器人技术和机器学习领域具有吸引力。虽然连续吸引网络（CANs）成功地模拟了内嗅网格细胞以编码物理空间，但将连续空间计算和抽象空间计算统一流程化仍具有挑战性。在这里，我们通过提出受CANs和向量象征架构（VSAs）启发的多功能信息处理机制模型，尝试弥合这一差距。该新颖的网格细胞VSA（GC-VSA）模型采用空间结构化的编码方案，使用3D神经模块模拟网格细胞模块的离散尺度和方向，再现其特征性的六边形感受野。在实验中，该模型在空间和抽象任务中展示了多功能性：（1）准确的位置路径整合，（2）时空表示以查询物体位置和时间关系，以及（3）使用家族树作为分层关系的结构化测试案例进行符号推理。 

---
# Tuning-Free Multi-Event Long Video Generation via Synchronized Coupled Sampling 

**Title (ZH)**: 无需调参的多事件长视频生成通过同步耦合采样 

**Authors**: Subin Kim, Seoung Wug Oh, Jui-Hsien Wang, Joon-Young Lee, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.08605)  

**Abstract**: While recent advancements in text-to-video diffusion models enable high-quality short video generation from a single prompt, generating real-world long videos in a single pass remains challenging due to limited data and high computational costs. To address this, several works propose tuning-free approaches, i.e., extending existing models for long video generation, specifically using multiple prompts to allow for dynamic and controlled content changes. However, these methods primarily focus on ensuring smooth transitions between adjacent frames, often leading to content drift and a gradual loss of semantic coherence over longer sequences. To tackle such an issue, we propose Synchronized Coupled Sampling (SynCoS), a novel inference framework that synchronizes denoising paths across the entire video, ensuring long-range consistency across both adjacent and distant frames. Our approach combines two complementary sampling strategies: reverse and optimization-based sampling, which ensure seamless local transitions and enforce global coherence, respectively. However, directly alternating between these samplings misaligns denoising trajectories, disrupting prompt guidance and introducing unintended content changes as they operate independently. To resolve this, SynCoS synchronizes them through a grounded timestep and a fixed baseline noise, ensuring fully coupled sampling with aligned denoising paths. Extensive experiments show that SynCoS significantly improves multi-event long video generation, achieving smoother transitions and superior long-range coherence, outperforming previous approaches both quantitatively and qualitatively. 

**Abstract (ZH)**: 同步耦合采样：一种用于长视频生成的新型推理框架 

---
# EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments 

**Title (ZH)**: EMMOE: 一种全面的开放环境体态移动操作基准 

**Authors**: Dongping Li, Tielong Cai, Tianci Tang, Wenhao Chai, Katherine Rose Driggs-Campbell, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08604)  

**Abstract**: Developing autonomous home robots controlled by natural language has long been a pursuit of human. While advancements in large language models (LLMs) and embodied intelligence make this goal closer, several challenges persist: the lack of a unified benchmark for more complex robot tasks, limited evaluation methods and metrics, data incompatibility between LLMs and mobile manipulation trajectories. To address these issues, we introduce Embodied Mobile Manipulation in Open Environments (EMMOE), which requires agents to interpret user instructions and execute long-horizon everyday tasks in continuous space. EMMOE seamlessly integrates high-level and low-level embodied tasks into a unified framework, along with three new metrics for more diverse assessment. Additionally, we collect EMMOE-100, which features in various task attributes, detailed process annotations, re-plans after failures, and two sub-datasets for LLM training. Furthermore, we design HomieBot, a sophisticated agent system consists of LLM with Direct Preference Optimization (DPO), light weighted navigation and manipulation models, and multiple error detection mechanisms. Finally, we demonstrate HomieBot's performance and the evaluation of different models and policies. 

**Abstract (ZH)**: 基于自然语言控制的家庭自主机器人长期是人类的追求。尽管大规模语言模型（LLMs）和体态智能的进步使得这一目标更近了一步，但仍存在一些挑战：缺乏针对更复杂机器人任务的统一基准、有限的评估方法和指标、LLMs与移动操控轨迹之间的数据不兼容。为应对这些挑战，我们引入了开放环境中的体态移动操控（EMMOE），要求智能体解析用户指令并执行持续空间中的长期日常任务。EMMOE 将高层和低层体态任务无缝集成到一个统一框架中，并引入了三项新的评估指标。此外，我们收集了EMMOE-100数据集，该数据集涵盖了各种任务属性、详细的流程注解、失败后的重新规划，并为LLM训练提供了两个子数据集。我们还设计了HomieBot，这是一种包含直接偏好优化（DPO）的大规模语言模型、轻量级导航和操作模型以及多种错误检测机制的高级代理系统。最后，我们展示了HomieBot的性能，并对不同模型和策略进行了评估。 

---
# BiasEdit: Debiasing Stereotyped Language Models via Model Editing 

**Title (ZH)**: BiasEdit: 基于模型编辑的消除刻板印象语言模型方法 

**Authors**: Xin Xu, Wei Xu, Ningyu Zhang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2503.08588)  

**Abstract**: Previous studies have established that language models manifest stereotyped biases. Existing debiasing strategies, such as retraining a model with counterfactual data, representation projection, and prompting often fail to efficiently eliminate bias or directly alter the models' biased internal representations. To address these issues, we propose BiasEdit, an efficient model editing method to remove stereotypical bias from language models through lightweight networks that act as editors to generate parameter updates. BiasEdit employs a debiasing loss guiding editor networks to conduct local edits on partial parameters of a language model for debiasing while preserving the language modeling abilities during editing through a retention loss. Experiments on StereoSet and Crows-Pairs demonstrate the effectiveness, efficiency, and robustness of BiasEdit in eliminating bias compared to tangental debiasing baselines and little to no impact on the language models' general capabilities. In addition, we conduct bias tracing to probe bias in various modules and explore bias editing impacts on different components of language models. 

**Abstract (ZH)**: Previous studies have established that language models manifest stereotyped biases. Existing debiasing strategies, such as retraining a model with counterfactual data, representation projection, and prompting often fail to efficiently eliminate bias or directly alter the models' biased internal representations. To address these issues, we propose BiasEdit, an efficient model editing method to remove stereotypical bias from language models through lightweight networks that act as editors to generate parameter updates. BiasEdit employs a debiasing loss guiding editor networks to conduct local edits on partial parameters of a language model for debiasing while preserving the language modeling abilities during editing through a retention loss. Experiments on StereoSet and Crows-Pairs demonstrate the effectiveness, efficiency, and robustness of BiasEdit in eliminating bias compared to tangential debiasing baselines and little to no impact on the language models' general capabilities. In addition, we conduct bias tracing to probe bias in various modules and explore bias editing impacts on different components of language models. 

---
# MsaMIL-Net: An End-to-End Multi-Scale Aware Multiple Instance Learning Network for Efficient Whole Slide Image Classification 

**Title (ZH)**: MsaMIL-Net：一种用于高效全视野图像分类的端到端多尺度感知多个实例学习网络 

**Authors**: Jiangping Wen, Jinyu Wen, Emei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08581)  

**Abstract**: Bag-based Multiple Instance Learning (MIL) approaches have emerged as the mainstream methodology for Whole Slide Image (WSI) classification. However, most existing methods adopt a segmented training strategy, which first extracts features using a pre-trained feature extractor and then aggregates these features through MIL. This segmented training approach leads to insufficient collaborative optimization between the feature extraction network and the MIL network, preventing end-to-end joint optimization and thereby limiting the overall performance of the model. Additionally, conventional methods typically extract features from all patches of fixed size, ignoring the multi-scale observation characteristics of pathologists. This not only results in significant computational resource waste when tumor regions represent a minimal proportion (as in the Camelyon16 dataset) but may also lead the model to suboptimal solutions.
To address these limitations, this paper proposes an end-to-end multi-scale WSI classification framework that integrates multi-scale feature extraction with multiple instance learning. Specifically, our approach includes: (1) a semantic feature filtering module to reduce interference from non-lesion areas; (2) a multi-scale feature extraction module to capture pathological information at different levels; and (3) a multi-scale fusion MIL module for global modeling and feature integration. Through an end-to-end training strategy, we simultaneously optimize both the feature extractor and MIL network, ensuring maximum compatibility between them.
Experiments were conducted on three cross-center datasets (DigestPath2019, BCNB, and UBC-OCEAN). Results demonstrate that our proposed method outperforms existing state-of-the-art approaches in terms of both accuracy (ACC) and AUC metrics. 

**Abstract (ZH)**: 基于包的多实例学习（MIL）方法已成为全视野图像（WSI）分类的主要方法。然而，现有大多数方法采用分段训练策略，首先使用预训练的特征提取器提取特征，然后通过MIL聚合这些特征。这种分段训练方法导致特征提取网络与MIL网络之间的协作优化不足，无法实现端到端联合优化，从而限制了模型的整体性能。此外，传统方法通常从所有固定大小的patches中提取特征，忽略了病理学家的多尺度观察特性。这不仅在肿瘤区域占比较小（如Camelyon16数据集）的情况下造成了显著的计算资源浪费，还可能导致模型获得次优解。

为解决这些局限性，本文提出了一种结合多尺度特征提取和多实例学习的端到端多尺度WSI分类框架。具体包括：（1）语义特征筛选模块，减少非病灶区域的干扰；（2）多尺度特征提取模块，捕获不同层次的病理信息；（3）多尺度融合MIL模块，进行全局建模和特征集成。通过端到端训练策略，同时优化特征提取器和MIL网络，确保两者之间的最大兼容性。

实验在三个跨中心数据集（DigestPath2019、BCNB和UBC-OCEAN）上进行。结果表明，与现有最先进的方法相比，本文提出的方法在准确率（ACC）和AUC指标上均表现出更佳性能。 

---
# When Discourse Stalls: Moving Past Five Semantic Stopsigns about Generative AI in Design Research 

**Title (ZH)**: 设计研究中对话受阻：超越生成式AI的五个语义路标 

**Authors**: Willem van der Maden, Vera van der Burg, Brett A. Halperin, Petra Jääskeläinen, Joseph Lindley, Derek Lomas, Timothy Merritt  

**Link**: [PDF](https://arxiv.org/pdf/2503.08565)  

**Abstract**: This essay examines how Generative AI (GenAI) is rapidly transforming design practices and how discourse often falls into over-simplified narratives that impede meaningful research and practical progress. We identify and deconstruct five prevalent "semantic stopsigns" -- reductive framings about GenAI in design that halt deeper inquiry and limit productive engagement. Reflecting upon two expert workshops at ACM conferences and semi-structured interviews with design practitioners, we analyze how these stopsigns manifest in research and practice. Our analysis develops mid-level knowledge that bridges theoretical discourse and practical implementation, helping designers and researchers interrogate common assumptions about GenAI in their own contexts. By recasting these stopsigns into more nuanced frameworks, we provide the design research community with practical approaches for thinking about and working with these emerging technologies. 

**Abstract (ZH)**: 这篇论文探讨了生成人工智能（GenAI）如何迅速变革设计实践，并分析了围绕GenAI的讨论往往陷入简化叙事，阻碍了有意义的研究和实际进展。我们识别并拆解了五种常见的“语义路障”——设计中关于GenAI的简化的框架化理解，这些理解阻碍了深入研究和有益的互动。通过反思ACM会议中两位专家的工作坊以及对设计从业者进行半结构化访谈，我们分析了这些路障在研究和实践中的表现形式。我们的分析构建了中等层次的知识，连接了理论讨论和实践实施，帮助设计师和研究者在其特定背景下质疑关于GenAI的常见假设。通过将这些路障重新构建成更微妙的框架，我们为设计研究社区提供了思考和运用这些新兴技术的实际方法。 

---
# MoE-Loco: Mixture of Experts for Multitask Locomotion 

**Title (ZH)**: MoE-Loco: 专家混合的多任务运动控制 

**Authors**: Runhan Huang, Shaoting Zhu, Yilun Du, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08564)  

**Abstract**: We present MoE-Loco, a Mixture of Experts (MoE) framework for multitask locomotion for legged robots. Our method enables a single policy to handle diverse terrains, including bars, pits, stairs, slopes, and baffles, while supporting quadrupedal and bipedal gaits. Using MoE, we mitigate the gradient conflicts that typically arise in multitask reinforcement learning, improving both training efficiency and performance. Our experiments demonstrate that different experts naturally specialize in distinct locomotion behaviors, which can be leveraged for task migration and skill composition. We further validate our approach in both simulation and real-world deployment, showcasing its robustness and adaptability. 

**Abstract (ZH)**: MoE-Loco：一种用于腿足机器人多任务运动的专家混合框架 

---
# Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies 

**Title (ZH)**: 无需故障数据的故障检测是否可行？基于模仿学习策略的不确定性感知运行时故障检测 

**Authors**: Chen Xu, Tony Khuong Nguyen, Emma Dixon, Christopher Rodriguez, Patrick Miller, Robert Lee, Paarth Shah, Rares Ambrus, Haruki Nishimura, Masha Itkina  

**Link**: [PDF](https://arxiv.org/pdf/2503.08558)  

**Abstract**: Recent years have witnessed impressive robotic manipulation systems driven by advances in imitation learning and generative modeling, such as diffusion- and flow-based approaches. As robot policy performance increases, so does the complexity and time horizon of achievable tasks, inducing unexpected and diverse failure modes that are difficult to predict a priori. To enable trustworthy policy deployment in safety-critical human environments, reliable runtime failure detection becomes important during policy inference. However, most existing failure detection approaches rely on prior knowledge of failure modes and require failure data during training, which imposes a significant challenge in practicality and scalability. In response to these limitations, we present FAIL-Detect, a modular two-stage approach for failure detection in imitation learning-based robotic manipulation. To accurately identify failures from successful training data alone, we frame the problem as sequential out-of-distribution (OOD) detection. We first distill policy inputs and outputs into scalar signals that correlate with policy failures and capture epistemic uncertainty. FAIL-Detect then employs conformal prediction (CP) as a versatile framework for uncertainty quantification with statistical guarantees. Empirically, we thoroughly investigate both learned and post-hoc scalar signal candidates on diverse robotic manipulation tasks. Our experiments show learned signals to be mostly consistently effective, particularly when using our novel flow-based density estimator. Furthermore, our method detects failures more accurately and faster than state-of-the-art (SOTA) failure detection baselines. These results highlight the potential of FAIL-Detect to enhance the safety and reliability of imitation learning-based robotic systems as they progress toward real-world deployment. 

**Abstract (ZH)**: Recent年份见证了由模仿学习和生成建模进步驱动的机器人操作系统的 impressive 进展，如扩散-基于流动的方法。随着机器人策略性能的提升，可实现的任务复杂性和时间范围也随之增加，导致难以事先预测的多种意外失败模式。为在安全关键的人类环境中实现可信的策略部署，在策略推断过程中可靠的运行时故障检测变得至关重要。然而，现有的大多数故障检测方法依赖于故障模式的先验知识，并且在训练过程中需要故障数据，这在实际应用中带来了显著的挑战和可扩展性问题。针对这些局限性，我们提出了一种模块化的两阶段方法 FAIL-Detect，用于基于模仿学习的机器人操作故障检测。通过仅从成功的训练数据中准确识别故障，我们将问题归结为序贯离分布（OOD）检测。我们首先将策略输入和输出提炼成与策略失败相关联的标量信号，并捕捉认识论不确定性。然后，FAIL-Detect 使用一致性预测（CP）作为不确定性量化的一个通用框架，带有统计保证。实验中，我们在多种机器人操作任务上全面研究了所学习和后验标量信号候选者。我们的实验结果显示，所学习的信号在大多数情况下都有效地工作，特别是在使用我们新颖的基于流动的概率分布估计器时。此外，我们的方法比现有最先进的（SOTA）故障检测基线更准确且更快地检测故障。这些结果突显了 FAIL-Detect 有潜力增强基于模仿学习的机器人系统在推向实际部署过程中的安全性和可靠性。 

---
# DAFE: LLM-Based Evaluation Through Dynamic Arbitration for Free-Form Question-Answering 

**Title (ZH)**: DAFE：基于动态仲裁的LLM评价方法用于自由形式的问答 

**Authors**: Sher Badshah, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2503.08542)  

**Abstract**: Evaluating Large Language Models (LLMs) free-form generated responses remains a challenge due to their diverse and open-ended nature. Traditional supervised signal-based automatic metrics fail to capture semantic equivalence or handle the variability of open-ended responses, while human evaluation, though reliable, is resource-intensive. Leveraging LLMs as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. Taking advantage of these capabilities, we propose the Dynamic Arbitration Framework for Evaluation (DAFE), which employs two primary LLM-as-judges and engages a third arbitrator only in cases of disagreements. This selective arbitration prioritizes evaluation reliability while reducing unnecessary computational demands compared to conventional majority voting. DAFE utilizes task-specific reference answers with dynamic arbitration to enhance judgment accuracy, resulting in significant improvements in evaluation metrics such as Macro F1 and Cohen's Kappa. Through experiments, including a comprehensive human evaluation, we demonstrate DAFE's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating free-form model outputs. 

**Abstract (ZH)**: 评估大规模语言模型（LLMs）生成的开放式回应 Remain a Challenge Due to Their Diverse and Open-Ended Nature. 利用LLMs作为评估者由于其强大的语言理解和指令遵循能力提供了一种有 promise 的替代方案。利用这些能力，我们提出了动态仲裁评估框架（DAFE），该框架使用两个主要的LLM作为裁判，并仅在出现分歧时引入第三个仲裁者。这种选择性仲裁优先考虑评估的可靠性，同时相比传统的多数投票减少了不必要的计算需求。DAFE 使用特定任务的参考答案和动态仲裁来提高判断准确性，从而在评价指标如宏观F1和科恩κ系数等方面取得了显著改进。通过实验，包括全面的人类评估，我们展示了DAFE的 ability 提供一致、可扩展且资源高效的评估，确立了其作为评估开放式模型输出稳健框架的地位。 

---
# Mellow: a small audio language model for reasoning 

**Title (ZH)**: Mellow：一个用于推理的小型音频语言模型 

**Authors**: Soham Deshmukh, Satvik Dixit, Rita Singh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2503.08540)  

**Abstract**: Multimodal Audio-Language Models (ALMs) can understand and reason over both audio and text. Typically, reasoning performance correlates with model size, with the best results achieved by models exceeding 8 billion parameters. However, no prior work has explored enabling small audio-language models to perform reasoning tasks, despite the potential applications for edge devices. To address this gap, we introduce Mellow, a small Audio-Language Model specifically designed for reasoning. Mellow achieves state-of-the-art performance among existing small audio-language models and surpasses several larger models in reasoning capabilities. For instance, Mellow scores 52.11 on MMAU, comparable to SoTA Qwen2 Audio (which scores 52.5) while using 50 times fewer parameters and being trained on 60 times less data (audio hrs). To train Mellow, we introduce ReasonAQA, a dataset designed to enhance audio-grounded reasoning in models. It consists of a mixture of existing datasets (30% of the data) and synthetically generated data (70%). The synthetic dataset is derived from audio captioning datasets, where Large Language Models (LLMs) generate detailed and multiple-choice questions focusing on audio events, objects, acoustic scenes, signal properties, semantics, and listener emotions. To evaluate Mellow's reasoning ability, we benchmark it on a diverse set of tasks, assessing on both in-distribution and out-of-distribution data, including audio understanding, deductive reasoning, and comparative reasoning. Finally, we conduct extensive ablation studies to explore the impact of projection layer choices, synthetic data generation methods, and language model pretraining on reasoning performance. Our training dataset, findings, and baseline pave the way for developing small ALMs capable of reasoning. 

**Abstract (ZH)**: 多模态音频-语言模型（ALMs）可以理解并推理音频和文本。通常，推理性能与模型大小相关，最佳结果由超过80亿参数的模型实现。然而，以往工作尚未探索使小型音频-语言模型执行推理任务的可能性，尽管这在边缘设备上具有潜在应用价值。为了解决这一问题，我们引入了Mellow，一种专门设计用于推理的小型音频-语言模型。Mellow在现有小型音频-语言模型中实现了最先进的性能，并在推理能力上超过了若干大型模型。例如，Mellow在MMAU上的得分为52.11，与最先进的Qwen2 Audio（得分为52.5）相当，但参数减少了50倍，训练数据（音频小时数）减少了60倍。为了训练Mellow，我们引入了ReasonAQA数据集，旨在增强模型的音频基础推理能力。该数据集由现有数据集的混合体（30%的数据）和合成数据（70%的数据）组成。合成数据是从音频描述数据集中衍生出来的，其中大型语言模型（LLMs）生成关于音频事件、对象、声学场景、信号属性、语义和听者情绪的详细和多项选择问题。为了评估Mellow的推理能力，我们将其在多种任务上进行基准测试，包括音频理解、演绎推理和比较推理，同时评估分布内和分布外数据。最后，我们进行了广泛的消融研究，以探索投影层选择、合成数据生成方法和语言模型预训练对推理性能的影响。我们的训练数据集、发现和基线为开发能够进行推理的小型多模态语言模型铺平了道路。 

---
# GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training 

**Title (ZH)**: GTR：引导性思考强化防止基于RL的VLM代理培训中思考崩溃 

**Authors**: Tong Wei, Yijun Yang, Junliang Xing, Yuanchun Shi, Zongqing Lu, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08525)  

**Abstract**: Reinforcement learning with verifiable outcome rewards (RLVR) has effectively scaled up chain-of-thought (CoT) reasoning in large language models (LLMs). Yet, its efficacy in training vision-language model (VLM) agents for goal-directed action reasoning in visual environments is less established. This work investigates this problem through extensive experiments on complex card games, such as 24 points, and embodied tasks from ALFWorld. We find that when rewards are based solely on action outcomes, RL fails to incentivize CoT reasoning in VLMs, instead leading to a phenomenon we termed thought collapse, characterized by a rapid loss of diversity in the agent's thoughts, state-irrelevant and incomplete reasoning, and subsequent invalid actions, resulting in negative rewards. To counteract thought collapse, we highlight the necessity of process guidance and propose an automated corrector that evaluates and refines the agent's reasoning at each RL step. This simple and scalable GTR (Guided Thought Reinforcement) framework trains reasoning and action simultaneously without the need for dense, per-step human labeling. Our experiments demonstrate that GTR significantly enhances the performance and generalization of the LLaVA-7b model across various visual environments, achieving 3-5 times higher task success rates compared to SoTA models with notably smaller model sizes. 

**Abstract (ZH)**: 验证性奖励学习（RLVR）在大型语言模型（LLMs）中有效扩展了链式思考（CoT）推理。然而，其在视觉语言模型（VLM）代理在视觉环境中的目标导向行动推理训练方面的有效性尚不明确。本工作通过在24点等复杂纸牌游戏以及ALFWorld中的实体任务上进行广泛的实验，探讨了这一问题。我们发现，当奖励仅基于行动结果时，RL无法激励VLM中的链式思考推理，反而导致我们称之为思维崩溃的现象，表现为代理思维的迅速同质化、与状态无关及不完整的推理，以及随后的无效行动，导致负奖励。为对抗思维崩溃，我们强调过程指导的必要性，并提出了一种自动化校正器，能够在每个RL步骤中评估和改进代理的推理。这种简单且可扩展的引导思考强化学习（GTR）框架能够在无需密集、逐步的人工标注的情况下同时训练推理和行动。我们的实验结果表明，GTR显著提升了LLaVA-7b模型在各种视觉环境中的性能和泛化能力，与性能相当但模型规模显著较小的最新模型相比，任务成功率提高了3-5倍。 

---
# A Triple-Inertial Accelerated Alternating Optimization Method for Deep Learning Training 

**Title (ZH)**: 三重惯性加速交替优化方法用于深度学习训练 

**Authors**: Chengcheng Yan, Jiawei Xu, Qingsong Wang, Zheng Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08489)  

**Abstract**: The stochastic gradient descent (SGD) algorithm has achieved remarkable success in training deep learning models. However, it has several limitations, including susceptibility to vanishing gradients, sensitivity to input data, and a lack of robust theoretical guarantees. In recent years, alternating minimization (AM) methods have emerged as a promising alternative for model training by employing gradient-free approaches to iteratively update model parameters. Despite their potential, these methods often exhibit slow convergence rates. To address this challenge, we propose a novel Triple-Inertial Accelerated Alternating Minimization (TIAM) framework for neural network training. The TIAM approach incorporates a triple-inertial acceleration strategy with a specialized approximation method, facilitating targeted acceleration of different terms in each sub-problem optimization. This integration improves the efficiency of convergence, achieving superior performance with fewer iterations. Additionally, we provide a convergence analysis of the TIAM algorithm, including its global convergence properties and convergence rate. Extensive experiments validate the effectiveness of the TIAM method, showing significant improvements in generalization capability and computational efficiency compared to existing approaches, particularly when applied to the rectified linear unit (ReLU) and its variants. 

**Abstract (ZH)**: 三惯性加速交替极小化（TIAM）框架在神经网络训练中的应用及其收敛性分析 

---
# Optimizing Ride-Pooling Operations with Extended Pickup and Drop-Off Flexibility 

**Title (ZH)**: 优化拼车运营以扩大接乘客和_drop_off_灵活性 

**Authors**: Hao Jiang, Yixing Xu, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2503.08472)  

**Abstract**: The Ride-Pool Matching Problem (RMP) is central to on-demand ride-pooling services, where vehicles must be matched with multiple requests while adhering to service constraints such as pickup delays, detour limits, and vehicle capacity. Most existing RMP solutions assume passengers are picked up and dropped off at their original locations, neglecting the potential for passengers to walk to nearby spots to meet vehicles. This assumption restricts the optimization potential in ride-pooling operations. In this paper, we propose a novel matching method that incorporates extended pickup and drop-off areas for passengers. We first design a tree-based approach to efficiently generate feasible matches between passengers and vehicles. Next, we optimize vehicle routes to cover all designated pickup and drop-off locations while minimizing total travel distance. Finally, we employ dynamic assignment strategies to achieve optimal matching outcomes. Experiments on city-scale taxi datasets demonstrate that our method improves the number of served requests by up to 13\% and average travel distance by up to 21\% compared to leading existing solutions, underscoring the potential of leveraging passenger mobility to significantly enhance ride-pooling service efficiency. 

**Abstract (ZH)**: 基于乘车共享的服务匹配问题（RMP）对于按需乘车共享服务至关重要，其中车辆需要与多个请求匹配，同时遵守服务约束，如接客延迟、绕行限制和车辆载客量。现有的大多数RMP解决方案假设乘客在原始地点上下车，忽略了乘客步行至附近地点以遇见到达车辆的可能性。这一假设限制了乘车共享操作的优化潜力。本文提出了一种新的匹配方法，将乘客的延长上车和下车区域纳入考虑。首先，设计了一种基于树的方法来高效生成乘客与车辆之间的可行匹配。其次，优化车辆路线以覆盖所有指定的上车和下车地点，同时尽量减少总行驶距离。最后，采用动态分配策略以达到最优的匹配结果。基于城市规模的出租车数据集的实验表明，与现有的领先解决方案相比，我们的方法可将服务的请求数量最多提高13%，平均行驶距离最多减少21%，这突显了利用乘客机动性以显著提高乘车共享服务效率的潜力。 

---
# Accelerating MoE Model Inference with Expert Sharding 

**Title (ZH)**: 加速MoE模型推理 dengan 专家分割 

**Authors**: Oana Balmau, Anne-Marie Kermarrec, Rafael Pires, André Loureiro Espírito Santo, Martijn de Vos, Milos Vujasinovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.08467)  

**Abstract**: Mixture of experts (MoE) models achieve state-of-the-art results in language modeling but suffer from inefficient hardware utilization due to imbalanced token routing and communication overhead. While prior work has focused on optimizing MoE training and decoder architectures, inference for encoder-based MoE models in a multi-GPU with expert parallelism setting remains underexplored. We introduce MoEShard, an inference system that achieves perfect load balancing through tensor sharding of MoE experts. Unlike existing approaches that rely on heuristic capacity factors or drop tokens, MoEShard evenly distributes computation across GPUs and ensures full token retention, maximizing utilization regardless of routing skewness. We achieve this through a strategic row- and column-wise decomposition of expert matrices. This reduces idle time and avoids bottlenecks caused by imbalanced expert assignments. Furthermore, MoEShard minimizes kernel launches by fusing decomposed expert computations, significantly improving throughput. We evaluate MoEShard against DeepSpeed on encoder-based architectures, demonstrating speedups of up to 6.4$\times$ in time to first token (TTFT). Our results show that tensor sharding, when properly applied to experts, is a viable and effective strategy for efficient MoE inference. 

**Abstract (ZH)**: MoE专家模型通过张量分片实现高效的推理负载均衡 

---
# Status and Future Prospects of the Standardization Framework Industry 4.0: A European Perspective 

**Title (ZH)**: 基于欧洲视角的Industry 4.0标准框架现状及未来前景研究 

**Authors**: Olga Meyer, Marvin Boell, Christoph Legat  

**Link**: [PDF](https://arxiv.org/pdf/2503.08460)  

**Abstract**: The rapid development of Industry 4.0 technologies requires robust and comprehensive standardization to ensure interoperability, safety and efficiency in the Industry of the Future. This paper examines the fundamental role and functionality of standardization, with a particular focus on its importance in Europe's regulatory framework. Based on this, selected topics in context of standardization activities in context intelligent manufacturing and digital twins are highlighted and, by that, an overview of the Industry 4.0 standards framework is provided. This paper serves both as an informative guide to the existing standards in Industry 4.0 with respect to Artificial Intelligence and Digital Twins, and as a call to action for increased cooperation between standardization bodies and the research community. By fostering such collaboration, we aim to facilitate the continued development and implementation of standards that will drive innovation and progress in the manufacturing sector. 

**Abstract (ZH)**: 工业4.0技术的快速发展需要 robust 和全面的标准制定以确保未来工业中的互操作性、安全性和效率。本文探讨了标准制定的基础作用和功能，特别是在欧洲监管框架中的重要性，并强调了智能制造和数字孪生背景下标准活动的相关主题，从而提供了工业4.0标准框架的概览。本文不仅是关于人工智能和数字孪生的工业4.0现有标准的信息指南，也是呼吁标准制定机构与研究社区加强合作的呼吁。通过促进这种合作，我们旨在推动标准的持续发展与实施，从而在制造领域推动创新与进步。 

---
# Controlling Latent Diffusion Using Latent CLIP 

**Title (ZH)**: 使用Latent CLIP控制潜在扩散过程 

**Authors**: Jason Becker, Chris Wendler, Peter Baylies, Robert West, Christian Wressnegger  

**Link**: [PDF](https://arxiv.org/pdf/2503.08455)  

**Abstract**: Instead of performing text-conditioned denoising in the image domain, latent diffusion models (LDMs) operate in latent space of a variational autoencoder (VAE), enabling more efficient processing at reduced computational costs. However, while the diffusion process has moved to the latent space, the contrastive language-image pre-training (CLIP) models, as used in many image processing tasks, still operate in pixel space. Doing so requires costly VAE-decoding of latent images before they can be processed. In this paper, we introduce Latent-CLIP, a CLIP model that operates directly in the latent space. We train Latent-CLIP on 2.7B pairs of latent images and descriptive texts, and show that it matches zero-shot classification performance of similarly sized CLIP models on both the ImageNet benchmark and a LDM-generated version of it, demonstrating its effectiveness in assessing both real and generated content. Furthermore, we construct Latent-CLIP rewards for reward-based noise optimization (ReNO) and show that they match the performance of their CLIP counterparts on GenEval and T2I-CompBench while cutting the cost of the total pipeline by 21%. Finally, we use Latent-CLIP to guide generation away from harmful content, achieving strong performance on the inappropriate image prompts (I2P) benchmark and a custom evaluation, without ever requiring the costly step of decoding intermediate images. 

**Abstract (ZH)**: Latent-CLIP：直接在潜在空间中运行的CLIP模型 

---
# ICPR 2024 Competition on Rider Intention Prediction 

**Title (ZH)**: ICPR 2024 摩托车手意图预测竞赛 

**Authors**: Shankar Gangisetty, Abdul Wasi, Shyam Nandan Rai, C. V. Jawahar, Sajay Raj, Manish Prajapati, Ayesha Choudhary, Aaryadev Chandra, Dev Chandan, Shireen Chand, Suvaditya Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.08437)  

**Abstract**: The recent surge in the vehicle market has led to an alarming increase in road accidents. This underscores the critical importance of enhancing road safety measures, particularly for vulnerable road users like motorcyclists. Hence, we introduce the rider intention prediction (RIP) competition that aims to address challenges in rider safety by proactively predicting maneuvers before they occur, thereby strengthening rider safety. This capability enables the riders to react to the potential incorrect maneuvers flagged by advanced driver assistance systems (ADAS). We collect a new dataset, namely, rider action anticipation dataset (RAAD) for the competition consisting of two tasks: single-view RIP and multi-view RIP. The dataset incorporates a spectrum of traffic conditions and challenging navigational maneuvers on roads with varying lighting conditions. For the competition, we received seventy-five registrations and five team submissions for inference of which we compared the methods of the top three performing teams on both the RIP tasks: one state-space model (Mamba2) and two learning-based approaches (SVM and CNN-LSTM). The results indicate that the state-space model outperformed the other methods across the entire dataset, providing a balanced performance across maneuver classes. The SVM-based RIP method showed the second-best performance when using random sampling and SMOTE. However, the CNN-LSTM method underperformed, primarily due to class imbalance issues, particularly struggling with minority classes. This paper details the proposed RAAD dataset and provides a summary of the submissions for the RIP 2024 competition. 

**Abstract (ZH)**: 车辆市场recent的激增导致了道路事故的急剧增加。这突显了增强道路安全措施的重要性，特别是对摩托车骑行者等脆弱道路使用者。因此，我们引入了骑行者意图预测(RIP)竞赛，旨在通过前瞻性地预测潜在的骑行动作来解决骑行安全挑战，从而加强骑行者安全。这一能力使骑行者能够根据高级驾驶辅助系统(ADAS)检测到的潜在不当动作作出反应。我们收集了一个新的数据集，即骑行者动作预测数据集(RAAD)，其中包括单视图RIP和多视图RIP两个任务。该数据集涵盖了各种交通条件和道路上具有挑战性的导航动作，且光照条件各异。在竞赛中，我们收到了75个注册和5个团队提交进行推断，我们比较了前三个最佳团队在两个RIP任务中的方法：一个状态空间模型(Mamba2)和两个基于学习的方法(SVM和CNN-LSTM)。结果表明，状态空间模型在整个数据集中表现最佳，提供了在各种动作类别中的均衡性能。基于SVM的RIP方法在使用随机抽样和SMOTE时表现出第二好的性能。然而，基于CNN-LSTM的方法表现不佳，主要原因是类别不平衡问题，尤其是在处理少数类时。本文详细介绍了提出的RAAD数据集，并提供了RIP 2024竞赛提交的总结。 

---
# AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models 

**Title (ZH)**: AnyMoLe: 利用视频扩散模型的任意字符动作插值 

**Authors**: Kwan Yun, Seokhyeon Hong, Chaelin Kim, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.08417)  

**Abstract**: Despite recent advancements in learning-based motion in-betweening, a key limitation has been overlooked: the requirement for character-specific datasets. In this work, we introduce AnyMoLe, a novel method that addresses this limitation by leveraging video diffusion models to generate motion in-between frames for arbitrary characters without external data. Our approach employs a two-stage frame generation process to enhance contextual understanding. Furthermore, to bridge the domain gap between real-world and rendered character animations, we introduce ICAdapt, a fine-tuning technique for video diffusion models. Additionally, we propose a ``motion-video mimicking'' optimization technique, enabling seamless motion generation for characters with arbitrary joint structures using 2D and 3D-aware features. AnyMoLe significantly reduces data dependency while generating smooth and realistic transitions, making it applicable to a wide range of motion in-betweening tasks. 

**Abstract (ZH)**: 基于学习的动力学中间帧生成中的一个关键限制是需要特定角色的数据集。本文介绍了一种新颖的方法AnyMoLe，该方法通过利用视频扩散模型来生成任意角色的动力学中间帧，而无需外部数据。我们的方法采用两阶段的帧生成过程以增强上下文理解。此外，为了弥合真实世界和渲染角色动画之间的领域差距，我们引入了ICAdapt，这是一种针对视频扩散模型的微调技术。此外，我们提出了“运动-视频模拟”优化技术，利用2D和3D感知特征使具有任意关节结构的角色的动力学生成无缝衔接。AnyMoLe显著减少了数据依赖性，生成了平滑且逼真的过渡，使其适用于广泛的动力学中间帧生成任务。 

---
# V-Max: Making RL practical for Autonomous Driving 

**Title (ZH)**: V-Max: 让强化学习在自动驾驶中更加实用 

**Authors**: Valentin Charraut, Thomas Tournaire, Waël Doulazmi, Thibault Buhet  

**Link**: [PDF](https://arxiv.org/pdf/2503.08388)  

**Abstract**: Learning-based decision-making has the potential to enable generalizable Autonomous Driving (AD) policies, reducing the engineering overhead of rule-based approaches. Imitation Learning (IL) remains the dominant paradigm, benefiting from large-scale human demonstration datasets, but it suffers from inherent limitations such as distribution shift and imitation gaps. Reinforcement Learning (RL) presents a promising alternative, yet its adoption in AD remains limited due to the lack of standardized and efficient research frameworks. To this end, we introduce V-Max, an open research framework providing all the necessary tools to make RL practical for AD. V-Max is built on Waymax, a hardware-accelerated AD simulator designed for large-scale experimentation. We extend it using ScenarioNet's approach, enabling the fast simulation of diverse AD datasets. V-Max integrates a set of observation and reward functions, transformer-based encoders, and training pipelines. Additionally, it includes adversarial evaluation settings and an extensive set of evaluation metrics. Through a large-scale benchmark, we analyze how network architectures, observation functions, training data, and reward shaping impact RL performance. 

**Abstract (ZH)**: 基于学习的决策制定有望实现通用可移植的自动驾驶（AD）策略，减少基于规则的方法的工程开销。模仿学习（IL）仍是主导范式，得益于大规模的人类示范数据集，但受限于分布偏移和模仿差距等固有局限。强化学习（RL）提供了一种有前途的替代方案，但由于缺乏标准化和高效的研发框架，其在AD中的应用仍受到限制。为此，我们介绍了V-Max，这是一个开放的研究框架，提供所有必要的工具以使RL在AD中实用。V-Max基于Waymax构建，Waymax是一个硬件加速的AD模拟器，适用于大规模实验。我们使用ScenarioNet的方法对其进行扩展，使其能够快速模拟多样的AD数据集。V-Max集成了观测函数和奖励函数、基于变换器的编码器以及训练管道。此外，它还包括对抗性评估设置和一系列评估指标。通过大规模基准测试，我们分析了网络架构、观测函数、训练数据和回报塑造对RL性能的影响。 

---
# InfluenceNet: AI Models for Banzhaf and Shapley Value Prediction 

**Title (ZH)**: InfluenceNet：AI模型在Banzhaf值和Shapley值预测中的应用 

**Authors**: Benjamin Kempinski, Tal Kachman  

**Link**: [PDF](https://arxiv.org/pdf/2503.08381)  

**Abstract**: Power indices are essential in assessing the contribution and influence of individual agents in multi-agent systems, providing crucial insights into collaborative dynamics and decision-making processes. While invaluable, traditional computational methods for exact or estimated power indices values require significant time and computational constraints, especially for large $(n\ge10)$ coalitions. These constraints have historically limited researchers' ability to analyse complex multi-agent interactions comprehensively. To address this limitation, we introduce a novel Neural Networks-based approach that efficiently estimates power indices for voting games, demonstrating comparable and often superiour performance to existing tools in terms of both speed and accuracy. This method not only addresses existing computational bottlenecks, but also enables rapid analysis of large coalitions, opening new avenues for multi-agent system research by overcoming previous computational limitations and providing researchers with a more accessible, scalable analytical this http URL increased efficiency will allow for the analysis of more complex and realistic multi-agent scenarios. 

**Abstract (ZH)**: 权力指数对于评估多agent系统中个体agent的贡献和影响力至关重要，提供了解协作动态和决策过程的关键见解。尽管如此，传统计算方法在准确或近似计算权力指数值方面需要大量时间和计算资源，尤其是对于较大的$(n\ge10)$联盟。这种限制使研究人员难以全面分析复杂的多agent交互。为了克服这一限制，我们引入了一种基于神经网络的新方法，该方法能够高效地估计投票游戏中的权力指数，其速度和准确度均与现有工具相当，甚至更优。该方法不仅解决了现有的计算瓶颈，还允许快速分析大型联盟，为多agent系统研究开辟了新的途径，克服了先前的计算限制，使研究人员能够获得更便捷、可扩展的分析工具。这将提高效率，使研究人员能够分析更加复杂和现实的多agent情景。 

---
# Robust Latent Matters: Boosting Image Generation with Sampling Error 

**Title (ZH)**: 稳健的隐空间因素：基于采样误差提升图像生成 

**Authors**: Kai Qiu, Xiang Li, Jason Kuen, Hao Chen, Xiaohao Xu, Jiuxiang Gu, Yinyi Luo, Bhiksha Raj, Zhe Lin, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2503.08354)  

**Abstract**: Recent image generation schemes typically capture image distribution in a pre-constructed latent space relying on a frozen image tokenizer. Though the performance of tokenizer plays an essential role to the successful generation, its current evaluation metrics (e.g. rFID) fail to precisely assess the tokenizer and correlate its performance to the generation quality (e.g. gFID). In this paper, we comprehensively analyze the reason for the discrepancy of reconstruction and generation qualities in a discrete latent space, and, from which, we propose a novel plug-and-play tokenizer training scheme to facilitate latent space construction. Specifically, a latent perturbation approach is proposed to simulate sampling noises, i.e., the unexpected tokens sampled, from the generative process. With the latent perturbation, we further propose (1) a novel tokenizer evaluation metric, i.e., pFID, which successfully correlates the tokenizer performance to generation quality and (2) a plug-and-play tokenizer training scheme, which significantly enhances the robustness of tokenizer thus boosting the generation quality and convergence speed. Extensive benchmarking are conducted with 11 advanced discrete image tokenizers with 2 autoregressive generation models to validate our approach. The tokenizer trained with our proposed latent perturbation achieve a notable 1.60 gFID with classifier-free guidance (CFG) and 3.45 gFID without CFG with a $\sim$400M generator. Code: this https URL. 

**Abstract (ZH)**: 最近的图像生成方案通常依赖于固定的形象标记器在预先构建的潜在空间中捕捉图像分布。尽管标记器的性能对生成的成功至关重要，但当前的评估指标（如rFID）无法精确评估标记器并将其性能与生成质量（如gFID）关联起来。在本文中，我们全面分析了在离散潜在空间中重构质量和生成质量不一致的原因，并据此提出了一种新的即插即用标记器训练方案以促进潜在空间的构建。具体地，我们提出了一个潜在扰动方法以模拟生成过程中的采样噪声，即意外采样的标记符。利用潜在扰动，我们进一步提出了一种新的标记器评估指标（pFID），成功地将标记器性能与生成质量关联起来，并提出了一种即插即用标记器训练方案，显著增强了标记器的鲁棒性，从而提高了生成质量和收敛速度。我们使用11种先进的离散图像标记器和2种自回归生成模型进行了广泛的基准测试以验证我们的方法。使用我们提出的方法训练的标记器在Classifier-Free Guidance (CFG) 下实现了显著的1.60 gFID，在没有CFG的情况下达到了3.45 gFID，且使用的生成器参数量约为4亿。代码：https://github.com/username/repo。 

---
# MINT-Demo: Membership Inference Test Demonstrator 

**Title (ZH)**: MINT-Demo: 成员推理测试演示器 

**Authors**: Daniel DeAlcala, Aythami Morales, Julian Fierrez, Gonzalo Mancera, Ruben Tolosana, Ruben Vera-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2503.08332)  

**Abstract**: We present the Membership Inference Test Demonstrator, to emphasize the need for more transparent machine learning training processes. MINT is a technique for experimentally determining whether certain data has been used during the training of machine learning models. We conduct experiments with popular face recognition models and 5 public databases containing over 22M images. Promising results, up to 89% accuracy are achieved, suggesting that it is possible to recognize if an AI model has been trained with specific data. Finally, we present a MINT platform as demonstrator of this technology aimed to promote transparency in AI training. 

**Abstract (ZH)**: 我们介绍了会员推理测试演示器，以强调需要更加透明的机器学习训练过程。MINT是一种实验性技术，用于确定某些数据是否在机器学习模型的训练过程中被使用。我们使用流行的面部识别模型和包含超过2200万张图像的5个公开数据库进行了实验。实验结果显示，准确率最高可达89%，表明可以通过此技术识别AI模型是否使用了特定数据进行训练。最后，我们展示了MINT平台作为此项技术的演示器，旨在促进AI训练过程的透明度。 

---
# Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation 

**Title (ZH)**: 将薄荷与巧克力相结合：缓解机器翻译中的度量干扰 

**Authors**: José Pombal, Nuno M. Guerreiro, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2503.08327)  

**Abstract**: As automatic metrics become increasingly stronger and widely adopted, the risk of unintentionally "gaming the metric" during model development rises. This issue is caused by metric interference (Mint), i.e., the use of the same or related metrics for both model tuning and evaluation. Mint can misguide practitioners into being overoptimistic about the performance of their systems: as system outputs become a function of the interfering metric, their estimated quality loses correlation with human judgments. In this work, we analyze two common cases of Mint in machine translation-related tasks: filtering of training data, and decoding with quality signals. Importantly, we find that Mint strongly distorts instance-level metric scores, even when metrics are not directly optimized for -- questioning the common strategy of leveraging a different, yet related metric for evaluation that is not used for tuning. To address this problem, we propose MintAdjust, a method for more reliable evaluation under Mint. On the WMT24 MT shared task test set, MintAdjust ranks translations and systems more accurately than state-of-the-art-metrics across a majority of language pairs, especially for high-quality systems. Furthermore, MintAdjust outperforms AutoRank, the ensembling method used by the organizers. 

**Abstract (ZH)**: 随着自动评估指标越来越强大并且被广泛采用，模型开发过程中无意中“操控指标”的风险也在上升。这一问题源于指标干扰（Metric Interference, Mint），即在同一或相关任务中使用相同的评估指标进行模型调整和评估。指标干扰可能误导从业者对系统性能过于乐观：当系统输出成为干扰指标的函数时，其估计的质量与人类判断之间的相关性会丧失。在本研究中，我们分析了机器翻译相关任务中常见的两种指标干扰实例：训练数据过滤和使用质量信号进行解码。重要的是，我们发现即使在未直接优化这些指标的情况下，指标干扰也强烈地扭曲了实例级别的指标评分，这质疑了使用不用于调优的另一相关指标进行评估的常用策略。为解决这一问题，我们提出了一种名为MintAdjust的方法，以在指标干扰下提供更可靠的评估。在WMT24机器翻译共享任务测试集上，对于大多数语言对，MintAdjust在多数情况下比当前最优指标更准确地排名翻译和系统，特别是在高质量系统方面。此外，MintAdjust还优于组织者使用的AutoRank集成方法。 

---
# Prototype-based Heterogeneous Federated Learning for Blade Icing Detection in Wind Turbines with Class Imbalanced Data 

**Title (ZH)**: 基于原型的异构联邦学习在风电叶片结冰检测中的应用，处理类别不平衡数据 

**Authors**: Lele Qi, Mengna Liu, Xu Cheng, Fan Shi, Xiufeng Liu, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08325)  

**Abstract**: Wind farms, typically in high-latitude regions, face a high risk of blade icing. Traditional centralized training methods raise serious privacy concerns. To enhance data privacy in detecting wind turbine blade icing, traditional federated learning (FL) is employed. However, data heterogeneity, resulting from collections across wind farms in varying environmental conditions, impacts the model's optimization capabilities. Moreover, imbalances in wind turbine data lead to models that tend to favor recognizing majority classes, thus neglecting critical icing anomalies. To tackle these challenges, we propose a federated prototype learning model for class-imbalanced data in heterogeneous environments to detect wind turbine blade icing. We also propose a contrastive supervised loss function to address the class imbalance problem. Experiments on real data from 20 turbines across two wind farms show our method outperforms five FL models and five class imbalance methods, with an average improvement of 19.64\% in \( mF_{\beta} \) and 5.73\% in \( m \)BA compared to the second-best method, BiFL. 

**Abstract (ZH)**: 基于异构环境的联邦原型学习模型：解决风电叶片冰冻检测中的类别不平衡问题 

---
# Evaluating Interpretable Reinforcement Learning by Distilling Policies into Programs 

**Title (ZH)**: 将策略提炼为程序以评估可解释的强化学习 

**Authors**: Hector Kohler, Quentin Delfosse, Waris Radji, Riad Akrour, Philippe Preux  

**Link**: [PDF](https://arxiv.org/pdf/2503.08322)  

**Abstract**: There exist applications of reinforcement learning like medicine where policies need to be ''interpretable'' by humans. User studies have shown that some policy classes might be more interpretable than others. However, it is costly to conduct human studies of policy interpretability. Furthermore, there is no clear definition of policy interpretabiliy, i.e., no clear metrics for interpretability and thus claims depend on the chosen definition. We tackle the problem of empirically evaluating policies interpretability without humans. Despite this lack of clear definition, researchers agree on the notions of ''simulatability'': policy interpretability should relate to how humans understand policy actions given states. To advance research in interpretable reinforcement learning, we contribute a new methodology to evaluate policy interpretability. This new methodology relies on proxies for simulatability that we use to conduct a large-scale empirical evaluation of policy interpretability. We use imitation learning to compute baseline policies by distilling expert neural networks into small programs. We then show that using our methodology to evaluate the baselines interpretability leads to similar conclusions as user studies. We show that increasing interpretability does not necessarily reduce performances and can sometimes increase them. We also show that there is no policy class that better trades off interpretability and performance across tasks making it necessary for researcher to have methodologies for comparing policies interpretability. 

**Abstract (ZH)**: 存在医学等应用领域的强化学习中，策略需要“可解释”给人类。虽然用户研究显示某些策略类别可能比其他类别更容易解释，但由于进行策略可解释性的用户研究成本高，且目前缺乏明确的策略可解释性定义，使得解释性依赖于所选择的定义。我们致力于通过不依赖人类的研究方法来实证评估策略的可解释性。尽管缺乏明确的定义，研究人员一致认为“可模拟性”的概念：策略可解释性应与人类理解给定状态下的策略行动相关。为了促进可解释强化学习的研究，我们贡献了一种新的评估策略可解释性的方法。该方法依赖于模拟性的代理指标，并通过大规模实证研究评估策略的可解释性。我们使用模仿学习将专家神经网络提炼为小型程序来计算基线策略，展示了使用我们的方法评估这些基线策略的可解释性，所得结论与用户研究相似。我们证明增加可解释性不一定降低性能，有时甚至可以提高性能。我们还证明，在各类任务中没有策略类别能够在可解释性和性能之间取得更好的权衡，因此研究人员需要具有比较策略可解释性的方法。 

---
# General-Purpose Aerial Intelligent Agents Empowered by Large Language Models 

**Title (ZH)**: 大型语言模型赋能的通用 aerial 智能代理 

**Authors**: Ji Zhao, Xiao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.08302)  

**Abstract**: The emergence of large language models (LLMs) opens new frontiers for unmanned aerial vehicle (UAVs), yet existing systems remain confined to predefined tasks due to hardware-software co-design challenges. This paper presents the first aerial intelligent agent capable of open-world task execution through tight integration of LLM-based reasoning and robotic autonomy. Our hardware-software co-designed system addresses two fundamental limitations: (1) Onboard LLM operation via an edge-optimized computing platform, achieving 5-6 tokens/sec inference for 14B-parameter models at 220W peak power; (2) A bidirectional cognitive architecture that synergizes slow deliberative planning (LLM task planning) with fast reactive control (state estimation, mapping, obstacle avoidance, and motion planning). Validated through preliminary results using our prototype, the system demonstrates reliable task planning and scene understanding in communication-constrained environments, such as sugarcane monitoring, power grid inspection, mine tunnel exploration, and biological observation applications. This work establishes a novel framework for embodied aerial artificial intelligence, bridging the gap between task planning and robotic autonomy in open environments. 

**Abstract (ZH)**: 大语言模型的出现为无人驾驶航空器开启了新的前沿领域，但现有系统仍受限于硬件软件协同设计的挑战。本文提出了一种通过将基于大语言模型的推理与机器人自主性紧密结合而实现的首个适用于开放世界任务执行的空中智能代理。我们的硬件软件协同设计系统解决了两个根本限制：（1）通过边缘优化计算平台实现机载大语言模型运行，以220W的峰值功率实现每秒5-6个词的推理，适用于14B参数模型；（2）一种双向认知架构，将慢速的反思性计划（基于大语言模型的任务规划）与快速的反应性控制（状态估计、制图、避障和路径规划）协同起来。通过我们原型的初步结果进行验证，该系统在通信受限环境中展示了可靠的任务规划和场景理解能力，适用于甘蔗监控、电力网络巡检、矿井隧道勘探和生物观察等应用。本文建立了一种新的框架，为开放环境中的任务规划与机器人自主性的结合提供了新的途径。 

---
# Large Language Model as Meta-Surrogate for Data-Driven Many-Task Optimization: A Proof-of-Principle Study 

**Title (ZH)**: 大型语言模型作为数据驱动多任务优化的元代理：一个原理验证研究 

**Authors**: Xian-Rong Zhang, Yue-Jiao Gong, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08301)  

**Abstract**: In many-task optimization scenarios, surrogate models are valuable for mitigating the computational burden of repeated fitness evaluations across tasks. This study proposes a novel meta-surrogate framework to assist many-task optimization, by leveraging the knowledge transfer strengths and emergent capabilities of large language models (LLMs). We formulate a unified framework for many-task fitness prediction, by defining a universal model with metadata to fit a group of problems. Fitness prediction is performed on metadata and decision variables, enabling efficient knowledge sharing across tasks and adaptability to new tasks. The LLM-based meta-surrogate treats fitness prediction as conditional probability estimation, employing a unified token sequence representation for task metadata, inputs, and outputs. This approach facilitates efficient inter-task knowledge sharing through shared token embeddings and captures complex task dependencies via multi-task model training. Experimental results demonstrate the model's emergent generalization ability, including zero-shot performance on problems with unseen dimensions. When integrated into evolutionary transfer optimization (ETO), our framework supports dual-level knowledge transfer -- at both the surrogate and individual levels -- enhancing optimization efficiency and robustness. This work establishes a novel foundation for applying LLMs in surrogate modeling, offering a versatile solution for many-task optimization. 

**Abstract (ZH)**: 在多任务优化场景中，代理模型对于缓解跨任务重复适应性评估的计算负担非常有价值。本研究提出了一种新型元代理框架，通过利用大规模语言模型（LLMs）的知识迁移优势和新兴能力来辅助多任务优化。我们制定了一个统一的多任务适应性预测框架，通过定义一个通用模型来拟合一组问题，并对元数据和决策变量进行适应性预测，以实现跨任务的高效知识共享和新任务的适应性。基于LLM的元代理将适应性预测视为条件概率估计，并采用统一的标记序列表示形式来表示任务元数据、输入和输出。该方法通过共享标记嵌入来促进跨任务的高效知识共享，并通过多任务模型训练捕捉复杂的任务依赖关系。实验结果表明，该模型具有潜在的泛化能力，包括对未见过的维度问题的零样本性能。当结合进化转移优化（ETO）时，该框架支持双重层面的知识迁移——在代理和个体层面——从而提高优化效率和鲁棒性。本研究为在代理建模中应用LLMs奠定了新的基础，并为多任务优化提供了多功能解决方案。 

---
# D3PO: Preference-Based Alignment of Discrete Diffusion Models 

**Title (ZH)**: D3PO：基于偏好离散扩散模型的对齐 

**Authors**: Umberto Borso, Davide Paglieri, Jude Wells, Tim Rocktäschel  

**Link**: [PDF](https://arxiv.org/pdf/2503.08295)  

**Abstract**: Diffusion models have achieved state-of-the-art performance across multiple domains, with recent advancements extending their applicability to discrete data. However, aligning discrete diffusion models with task-specific preferences remains challenging, particularly in scenarios where explicit reward functions are unavailable. In this work, we introduce Discrete Diffusion DPO (D3PO), the first adaptation of Direct Preference Optimization (DPO) to discrete diffusion models formulated as continuous-time Markov chains. Our approach derives a novel loss function that directly fine-tunes the generative process using preference data while preserving fidelity to a reference distribution. We validate D3PO on a structured binary sequence generation task, demonstrating that the method effectively aligns model outputs with preferences while maintaining structural validity. Our results highlight that D3PO enables controlled fine-tuning without requiring explicit reward models, making it a practical alternative to reinforcement learning-based approaches. Future research will explore extending D3PO to more complex generative tasks, including language modeling and protein sequence generation, as well as investigating alternative noise schedules, such as uniform noising, to enhance flexibility across different applications. 

**Abstract (ZH)**: 离散扩散模型D3PO：直接偏好优化在离散扩散模型中的应用 

---
# Large Language Models for Outpatient Referral: Problem Definition, Benchmarking and Challenges 

**Title (ZH)**: 门诊转诊的大语言模型应用：问题定义、基准测试与挑战 

**Authors**: Xiaoxiao Liu, Qingying Xiao, Junying Chen, Xiangyi Feng, Xiangbo Wu, Bairui Zhang, Xiang Wan, Jian Chang, Guangjun Yu, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08292)  

**Abstract**: Large language models (LLMs) are increasingly applied to outpatient referral tasks across healthcare systems. However, there is a lack of standardized evaluation criteria to assess their effectiveness, particularly in dynamic, interactive scenarios. In this study, we systematically examine the capabilities and limitations of LLMs in managing tasks within Intelligent Outpatient Referral (IOR) systems and propose a comprehensive evaluation framework specifically designed for such systems. This framework comprises two core tasks: static evaluation, which focuses on evaluating the ability of predefined outpatient referrals, and dynamic evaluation, which evaluates capabilities of refining outpatient referral recommendations through iterative dialogues. Our findings suggest that LLMs offer limited advantages over BERT-like models, but show promise in asking effective questions during interactive dialogues. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗保健系统中的门诊转诊任务中越来越多地被应用。然而，缺乏标准化的评估标准来评估其效果，特别是在动态和交互式场景中。在本研究中，我们系统地考察了LLMs在智能门诊转诊（IOR）系统中管理任务的能力及其局限性，并提出了一套专门针对此类系统的全面评估框架。该框架包括两个核心任务：静态评估，侧重于评估预定义门诊转诊的能力；动态评估，通过迭代对话评估改善门诊转诊推荐的能力。我们的研究发现，LLMs在与用户交互时提出有效问题方面优于BERT-like模型，显示出一定的潜力。 

---
# OminiControl2: Efficient Conditioning for Diffusion Transformers 

**Title (ZH)**: OmniControl2：高效的扩散变换器条件化方法 

**Authors**: Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08280)  

**Abstract**: Fine-grained control of text-to-image diffusion transformer models (DiT) remains a critical challenge for practical deployment. While recent advances such as OminiControl and others have enabled a controllable generation of diverse control signals, these methods face significant computational inefficiency when handling long conditional inputs. We present OminiControl2, an efficient framework that achieves efficient image-conditional image generation. OminiControl2 introduces two key innovations: (1) a dynamic compression strategy that streamlines conditional inputs by preserving only the most semantically relevant tokens during generation, and (2) a conditional feature reuse mechanism that computes condition token features only once and reuses them across denoising steps. These architectural improvements preserve the original framework's parameter efficiency and multi-modal versatility while dramatically reducing computational costs. Our experiments demonstrate that OminiControl2 reduces conditional processing overhead by over 90% compared to its predecessor, achieving an overall 5.9$\times$ speedup in multi-conditional generation scenarios. This efficiency enables the practical implementation of complex, multi-modal control for high-quality image synthesis with DiT models. 

**Abstract (ZH)**: 细粒度控制文本到图像扩散变换器模型（DiT）仍是一项重要的挑战。我们提出了OminiControl2，一种高效的框架，实现高效条件图像生成。OminiControl2引入了两项关键创新：（1）动态压缩策略，在生成过程中仅保留最具有语义相关的标记以简化条件输入；（2）条件特征重用机制，在去噪步骤中仅计算一次条件标记特征并重复使用。这些架构改进保持了原始框架的参数效率和多模态 versatility，同时大幅减少了计算成本。我们的实验表明，与前一代相比，OminiControl2将条件处理 overhead 减少了90%以上，在多条件生成场景中实现了整体5.9倍的速度提升。这种效率使得复杂、多模态控制在DiT模型中用于高质量图像合成的实用实现成为可能。 

---
# Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks 

**Title (ZH)**: Adv-CPG：一种基于面部 adversarial 攻击的定制化portrait生成框架 

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08269)  

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively. 

**Abstract (ZH)**: Recent定制化Portrait生成（CPG）方法 

---
# DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness 

**Title (ZH)**: 基于物理意识的全能机器人灵巧抓取：DexGrasp Anything 

**Authors**: Yiming Zhong, Qi Jiang, Jingyi Yu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.08257)  

**Abstract**: A dexterous hand capable of grasping any object is essential for the development of general-purpose embodied intelligent robots. However, due to the high degree of freedom in dexterous hands and the vast diversity of objects, generating high-quality, usable grasping poses in a robust manner is a significant challenge. In this paper, we introduce DexGrasp Anything, a method that effectively integrates physical constraints into both the training and sampling phases of a diffusion-based generative model, achieving state-of-the-art performance across nearly all open datasets. Additionally, we present a new dexterous grasping dataset containing over 3.4 million diverse grasping poses for more than 15k different objects, demonstrating its potential to advance universal dexterous grasping. The code of our method and our dataset will be publicly released soon. 

**Abstract (ZH)**: 一种能够抓取任意物体的灵巧手对于通用 embodiment 智能机器人的发展至关重要。然而，由于灵巧手的大自由度和物体的极大多样性，以稳健的方式生成高质量且实用的抓取姿态是一个重大挑战。本文介绍了一种方法——DexGrasp Anything，该方法在基于扩散的生成模型的训练和采样阶段有效集成物理约束，实现了在几乎所有公开数据集上的最佳性能。此外，我们还提出了一种新的灵巧抓取数据集，包含超过340万种不同的抓取姿态，涉及超过15,000种不同的物体，展示了其在推动通用灵巧抓取方面的能力。我们的方法代码和数据集将在不久的将来公开发布。 

---
# MT-NAM: An Efficient and Adaptive Model for Epileptic Seizure Detection 

**Title (ZH)**: MT-NAM: 一种高效自适应的癫痫发作检测模型 

**Authors**: Arshia Afzal, Volkan Cevher, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.08251)  

**Abstract**: Enhancing the accuracy and efficiency of machine learning algorithms employed in neural interface systems is crucial for advancing next-generation intelligent therapeutic devices. However, current systems often utilize basic machine learning models that do not fully exploit the natural structure of brain signals. Additionally, existing learning models used for neural signal processing often demonstrate low speed and efficiency during inference. To address these challenges, this study introduces Micro Tree-based NAM (MT-NAM), a distilled model based on the recently proposed Neural Additive Models (NAM). The MT-NAM achieves a remarkable 100$\times$ improvement in inference speed compared to standard NAM, without compromising accuracy. We evaluate our approach on the CHB-MIT scalp EEG dataset, which includes recordings from 24 patients with varying numbers of sessions and seizures. NAM achieves an 85.3\% window-based sensitivity and 95\% specificity. Interestingly, our proposed MT-NAM shows only a 2\% reduction in sensitivity compared to the original NAM. To regain this sensitivity, we utilize a test-time template adjuster (T3A) as an update mechanism, enabling our model to achieve higher sensitivity during test time by accommodating transient shifts in neural signals. With this online update approach, MT-NAM achieves the same sensitivity as the standard NAM while achieving approximately 50$\times$ acceleration in inference speed. 

**Abstract (ZH)**: 增强神经接口系统中采用的机器学习算法的准确性和效率对于推进下一代智能治疗设备至关重要。然而，当前系统往往使用基本的机器学习模型，未能充分利用脑信号的自然结构。此外，现有的用于神经信号处理的机器学习模型在推理过程中通常速度和效率较低。为解决这些挑战，本研究引入了基于最近提出的神经加法模型（NAM）的微树基于NAM（MT-NAM）的精简模型。MT-NAM在不牺牲准确性的前提下，相比标准NAM实现了高达100倍的推理速度提升。我们使用CHB-MIT头皮EEG数据集评估了我们的方法，该数据集包括来自24名患者的不同次数的记录和癫痫发作。NAM实现了85.3%的窗口敏感度和95%的特异度。有趣的是，我们提出的MT-NAM在敏感度上只比原始NAM降低了2%。为了恢复这一敏感度，我们利用测试时间模板调整器（T3A）作为更新机制，在神经信号的短暂变化时调整模型，以提高测试时的敏感度。采用这种在线更新方法，MT-NAM在实现约50倍的推理速度加速的同时，达到了与标准NAM相同的敏感度。 

---
# Aligning Text to Image in Diffusion Models is Easier Than You Think 

**Title (ZH)**: 将文本与图像对齐在扩散模型中并没有你想象的那么难 

**Authors**: Jaa-Yeon Lee, Byunghee Cha, Jeongsol Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08250)  

**Abstract**: While recent advancements in generative modeling have significantly improved text-image alignment, some residual misalignment between text and image representations still remains. Although many approaches have attempted to address this issue by fine-tuning models using various reward models, etc., we revisit the challenge from the perspective of representation alignment-an approach that has gained popularity with the success of REPresentation Alignment (REPA). We first argue that conventional text-to-image (T2I) diffusion models, typically trained on paired image and text data (i.e., positive pairs) by minimizing score matching or flow matching losses, is suboptimal from the standpoint of representation alignment. Instead, a better alignment can be achieved through contrastive learning that leverages both positive and negative pairs. To achieve this efficiently even with pretrained models, we introduce a lightweight contrastive fine tuning strategy called SoftREPA that uses soft text tokens. This approach improves alignment with minimal computational overhead by adding fewer than 1M trainable parameters to the pretrained model. Our theoretical analysis demonstrates that our method explicitly increases the mutual information between text and image representations, leading to enhanced semantic consistency. Experimental results across text-to-image generation and text-guided image editing tasks validate the effectiveness of our approach in improving the semantic consistency of T2I generative models. 

**Abstract (ZH)**: 虽然生成模型的近期进展显著提高了文本与图像的一致性，但仍存在一些残余的不一致性。尽管许多方法试图通过使用各种奖励模型等进行微调来解决这一问题，我们从表示一致性对齐的角度重新审视了这一挑战——这一方法在REPresentation Alignment (REPA) 成功之后受到了广泛关注。我们首先论证传统的文本到图像（T2I）扩散模型，通常通过最小化分数匹配或流匹配损失在配对的图像和文本数据（即正样本对）上进行训练，从表示一致性的角度来看并不理想。相反，通过利用正负样本对进行对比学习可以获得更好的一致性对齐。为了在使用预训练模型的情况下高效实现这一点，我们介绍了一种轻量级的对比学习微调策略 SoftREPA，该策略使用软文本标记。通过向预训练模型添加不到100万的可训练参数，该方法在最大限度减少计算开销的同时提高了表示的一致性。我们的理论分析表明，该方法明确地增加了文本和图像表示之间的互信息，从而提高了语义一致性。实验结果在文本到图像生成和文本引导的图像编辑任务中验证了我们方法在提高T2I生成模型的语义一致性方面的有效性。 

---
# Investigating Execution-Aware Language Models for Code Optimization 

**Title (ZH)**: 执行感知的编程语言模型用于代码优化 

**Authors**: Federico Di Menna, Luca Traini, Gabriele Bavota, Vittorio Cortellessa  

**Link**: [PDF](https://arxiv.org/pdf/2503.08228)  

**Abstract**: Code optimization is the process of enhancing code efficiency, while preserving its intended functionality. This process often requires a deep understanding of the code execution behavior at run-time to identify and address inefficiencies effectively. Recent studies have shown that language models can play a significant role in automating code optimization. However, these models may have insufficient knowledge of how code execute at run-time. To address this limitation, researchers have developed strategies that integrate code execution information into language models. These strategies have shown promise, enhancing the effectiveness of language models in various software engineering tasks. However, despite the close relationship between code execution behavior and efficiency, the specific impact of these strategies on code optimization remains largely unexplored. This study investigates how incorporating code execution information into language models affects their ability to optimize code. Specifically, we apply three different training strategies to incorporate four code execution aspects -- line executions, line coverage, branch coverage, and variable states -- into CodeT5+, a well-known language model for code. Our results indicate that execution-aware models provide limited benefits compared to the standard CodeT5+ model in optimizing code. 

**Abstract (ZH)**: 代码执行信息嵌入对语言模型代码优化能力的影响研究 

---
# A Grey-box Text Attack Framework using Explainable AI 

**Title (ZH)**: 基于解释性人工智能的灰盒文本攻击框架 

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08226)  

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models. 

**Abstract (ZH)**: 可解释AI是一种強大策略，用于通过人类可理解的语言理解复杂黑盒模型的预测。它提供了使用可信赖和可靠AI系统的必要证据。另一方面，它也可能揭示AI模型中存在的潜在漏洞。传统文本对抗攻击通过使用词替换、数据增强技术和基于梯度的攻击来生成对抗句子，针对强大的预训练双向编码器表示变换器（BERT）变体。这些攻击通常是白盒性质的，不实用，因为它们很容易被人类检测到，例如将“Poor”变为“Rich”。我们提出了一种简单而有效的灰盒兼黑盒方法，这种方法在使用一组代理变换器/BERT模型进行攻击时不需要了解模型的内部结构，并利用可解释AI技术来执行攻击。由于变换器是几乎所有自然语言处理（NLP）任务的当前最先进的模型，一种基于BERT1生成的攻击可以转移到BERT2。这一可迁移性得益于变换器中的注意力机制，该机制使模型能够捕捉序列中的长距离依赖关系。利用BERT的一般化能力及其注意力机制，我们尝试通过攻击基于不同架构的几种代理变换器变体来利用变换器的学习过程。我们证明，这种方法仅通过改变一个不可被人检测到的词即可生成语义良好的句子，但仍能欺骗其他BERT模型。 

---
# EgoBlind: Towards Egocentric Visual Assistance for the Blind People 

**Title (ZH)**: 自视角盲助视系统：面向盲人的第一人称视觉辅助 

**Authors**: Junbin Xiao, Nanxin Huang, Hao Qiu, Zhulin Tao, Xun Yang, Richang Hong, Meng Wang, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08221)  

**Abstract**: We present EgoBlind, the first egocentric VideoQA dataset collected from blind individuals to evaluate the assistive capabilities of contemporary multimodal large language models (MLLMs). EgoBlind comprises 1,210 videos that record the daily lives of real blind users from a first-person perspective. It also features 4,927 questions directly posed or generated and verified by blind individuals to reflect their needs for visual assistance under various scenarios. We provide each question with an average of 3 reference answers to alleviate subjective evaluation. Using EgoBlind, we comprehensively evaluate 15 leading MLLMs and find that all models struggle, with the best performers achieving accuracy around 56\%, far behind human performance of 87.4\%. To guide future advancements, we identify and summarize major limitations of existing MLLMs in egocentric visual assistance for the blind and provide heuristic suggestions for improvement. With these efforts, we hope EgoBlind can serve as a valuable foundation for developing more effective AI assistants to enhance the independence of the blind individuals' lives. 

**Abstract (ZH)**: EgoBlind: 一个来自盲人用户的第一人称视频问答数据集，用于评估当下多模态大语言模型的辅助能力 

---
# CL-MVSNet: Unsupervised Multi-view Stereo with Dual-level Contrastive Learning 

**Title (ZH)**: CL-MVSNet：基于双层对比学习的无监督多视点立体视觉 

**Authors**: Kaiqiang Xiong, Rui Peng, Zhe Zhang, Tianxing Feng, Jianbo Jiao, Feng Gao, Ronggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08219)  

**Abstract**: Unsupervised Multi-View Stereo (MVS) methods have achieved promising progress recently. However, previous methods primarily depend on the photometric consistency assumption, which may suffer from two limitations: indistinguishable regions and view-dependent effects, e.g., low-textured areas and reflections. To address these issues, in this paper, we propose a new dual-level contrastive learning approach, named CL-MVSNet. Specifically, our model integrates two contrastive branches into an unsupervised MVS framework to construct additional supervisory signals. On the one hand, we present an image-level contrastive branch to guide the model to acquire more context awareness, thus leading to more complete depth estimation in indistinguishable regions. On the other hand, we exploit a scene-level contrastive branch to boost the representation ability, improving robustness to view-dependent effects. Moreover, to recover more accurate 3D geometry, we introduce an L0.5 photometric consistency loss, which encourages the model to focus more on accurate points while mitigating the gradient penalty of undesirable ones. Extensive experiments on DTU and Tanks&Temples benchmarks demonstrate that our approach achieves state-of-the-art performance among all end-to-end unsupervised MVS frameworks and outperforms its supervised counterpart by a considerable margin without fine-tuning. 

**Abstract (ZH)**: 无监督多视图立体视觉（MVS）方法最近取得了显著进展。然而，现有方法主要依赖于光电一致性假设，可能面临两个局限性：不可区分区域和视点相关效应，例如低纹理区域和反射。为了解决这些问题，本文提出了一种新的双层对比学习方法，称为CL-MVSNet。具体而言，我们的模型将两个对比学习分支集成到无监督MVS框架中，以构建额外的监督信号。一方面，我们提出了图像级对比学习分支，以引导模型获得更多的上下文感知，从而在不可区分区域实现更完整的深度估计。另一方面，我们利用场景级对比学习分支增强表示能力，提高对视点相关效应的鲁棒性。此外，为了恢复更准确的三维几何结构，我们引入了一种L0.5光电一致性损失，该损失促使模型更关注准确点，同时减轻不可取点的梯度惩罚。在DTU和Tanks&Temples基准上的 extensive 实验表明，我们的方法在所有端到端无监督MVS框架中达到了最先进的性能，并且在无需微调的情况下显著优于其监督版本。 

---
# DeepRAG: Building a Custom Hindi Embedding Model for Retrieval Augmented Generation from Scratch 

**Title (ZH)**: DeepRAG: 从 scratch 构建一种定制化印地语嵌入模型以增强生成 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2503.08213)  

**Abstract**: In this paper, I present our work on DeepRAG, a specialized embedding model we built specifically for Hindi language in RAG systems. While LLMs have gotten really good at generating text, their performance in retrieval tasks still depends heavily on having quality embeddings - something that's been lacking for Hindi despite being one of the world's most spoken languages. We tackled this by creating embeddings from the ground up rather than just fine-tuning existing models. Our process involved collecting diverse Hindi texts (over 2.7M samples), training a custom SentencePiece tokenizer that actually understands Hindi morphology, designing transformer architecture with Hindi-specific attention mechanisms, and optimizing with contrastive learning. Results were honestly better than I expected - we saw a 23% improvement in retrieval precision compared to the multilingual models everyone's been using. The paper details our methodology, which I think could help others working with low-resource languages where the one-size-fits-all multilingual models fall short. We've also integrated our embeddings with LangChain to build complete Hindi RAG systems, which might be useful for practitioners. While there's still tons more to explore, I believe this work addresses a critical gap for Hindi NLP and demonstrates why language-specific approaches matter. 

**Abstract (ZH)**: 本论文介绍了我们为RAG系统构建的专门嵌入模型DeepRAG，针对的是印地语语言。虽然大规模语言模型在生成文本方面表现优异，但在检索任务上的性能仍然高度依赖高质量的嵌入，这对印地语来说尤其缺乏，尽管它是世界上使用最广泛的语言之一。我们通过从零开始创建嵌入，而不是仅仅对现有模型进行微调来解决这一问题。我们的过程包括收集多样化的印地语文本（超过270万样本）、训练一个真正理解印地语形态结构的自定义SentencePiece分词器、设计具有印地语特定注意力机制的变压器架构，并采用对比学习进行优化。结果比我预期的要好 - 我们在检索精度上看到了23%的提升，超过了大家一直在使用的多语言模型。本文详细介绍了我们的方法论，我认为这可能对其他处理资源匮乏语言的研究者有所帮助，这些语言通常不适合通用的多语言模型。我们还将我们的嵌入与LangChain集成，构建了完整的印地语RAG系统，这可能对实践者有用。尽管仍有更多探索的空间，但我认为这项工作解决了印地语NLP的一个关键缺口，并证明了特定语言方法的重要性。 

---
# OLMD: Orientation-aware Long-term Motion Decoupling for Continuous Sign Language Recognition 

**Title (ZH)**: OLMD：方向感知的长期运动解耦用于连续手语识别 

**Authors**: Yiheng Yu, Sheng Liu, Yuan Feng, Min Xu, Zhelun Jin, Xuhua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08205)  

**Abstract**: The primary challenge in continuous sign language recognition (CSLR) mainly stems from the presence of multi-orientational and long-term motions. However, current research overlooks these crucial aspects, significantly impacting accuracy. To tackle these issues, we propose a novel CSLR framework: Orientation-aware Long-term Motion Decoupling (OLMD), which efficiently aggregates long-term motions and decouples multi-orientational signals into easily interpretable components. Specifically, our innovative Long-term Motion Aggregation (LMA) module filters out static redundancy while adaptively capturing abundant features of long-term motions. We further enhance orientation awareness by decoupling complex movements into horizontal and vertical components, allowing for motion purification in both orientations. Additionally, two coupling mechanisms are proposed: stage and cross-stage coupling, which together enrich multi-scale features and improve the generalization capabilities of the model. Experimentally, OLMD shows SOTA performance on three large-scale datasets: PHOENIX14, PHOENIX14-T, and CSL-Daily. Notably, we improved the word error rate (WER) on PHOENIX14 by an absolute 1.6% compared to the previous SOTA 

**Abstract (ZH)**: 连续手语识别的主要挑战来自于多方向性和长时间运动的存在。然而，当前研究忽视了这些关键方面，显著影响了识别准确性。为应对这些挑战，我们提出了一种新颖的连续手语识别框架：方向 Awareness 长时间运动解耦（OLMD），该框架有效地聚集了长时间运动并解耦多方向信号为易于解释的组件。具体而言，我们创新性的长时间运动聚合（LMA）模块过滤掉了静态冗余，同时适当地捕捉了长时间运动的丰富特征。进一步地，通过将复杂运动分解为水平和垂直分量来增强方向感知，使得在两个方向上都能实现运动净化。此外，我们还提出了两种耦合机制：阶段内耦合和跨阶段耦合，这两种机制共同丰富了多尺度特征，提高了模型的泛化能力。实验表明，OLMD在三个大规模数据集PHOENIX14、PHOENIX14-T和CSL-Daily上展现了最佳性能。特别地，我们在PHOENIX14数据集上的词错误率（WER）绝对提升了1.6%，超过了之前的最佳表现。 

---
# A Cascading Cooperative Multi-agent Framework for On-ramp Merging Control Integrating Large Language Models 

**Title (ZH)**: 基于大型语言模型的匝道汇入控制级联合作多智能体框架 

**Authors**: Miao Zhang, Zhenlong Fang, Tianyi Wang, Qian Zhang, Shuai Lu, Junfeng Jiao, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08199)  

**Abstract**: Traditional Reinforcement Learning (RL) suffers from replicating human-like behaviors, generalizing effectively in multi-agent scenarios, and overcoming inherent interpretability this http URL tasks are compounded when deep environment understanding, agent coordination and dynamic optimization are required. While Large Language Model (LLM) enhanced methods have shown promise in generalization and interoperability, they often neglect necessary multi-agent coordination. Therefore, we introduce the Cascading Cooperative Multi-agent (CCMA) framework, integrating RL for individual interactions, a fine-tuned LLM for regional cooperation, a reward function for global optimization, and the Retrieval-augmented Generation mechanism to dynamically optimize decision-making across complex driving scenarios. Our experiments demonstrate that the CCMA outperforms existing RL methods, demonstrating significant improvements in both micro and macro-level performance in complex driving environments. 

**Abstract (ZH)**: 传统强化学习（RL）在模仿人类行为、在多Agent场景中有效泛化以及解决固有的解释性问题方面存在挑战，特别是当需要深刻理解环境、Agent协调和动态优化时，这些挑战更为突出。虽然增强型大型语言模型（LLM）方法在泛化和互操作性方面显示出潜力，但它们往往忽视了必要的多Agent协调。因此，我们提出了级联合作多Agent（CCMA）框架，该框架结合了用于个体交互的RL、细调的LLM用于区域合作、用于全局优化的奖励函数以及检索增强的生成机制，以动态优化复杂驾驶场景下的决策。我们的实验表明，CCMA在复杂驾驶环境中的微观和宏观绩效上均优于现有RL方法。 

---
# RigoChat 2: an adapted language model to Spanish using a bounded dataset and reduced hardware 

**Title (ZH)**: RigoChat 2：一种基于有界数据集和减少硬件资源的语言模型适应西班牙语 

**Authors**: Gonzalo Santamaría Gómez, Guillem García Subies, Pablo Gutiérrez Ruiz, Mario González Valero, Natàlia Fuertes, Helena Montoro Zamorano, Carmen Muñoz Sanz, Leire Rosado Plaza, Nuria Aldama García, David Betancur Sánchez, Kateryna Sushkova, Marta Guerrero Nieto, Álvaro Barbero Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2503.08188)  

**Abstract**: Large Language Models (LLMs) have become a key element of modern artificial intelligence, demonstrating the ability to address a wide range of language processing tasks at unprecedented levels of accuracy without the need of collecting problem-specific data. However, these versatile models face a significant challenge: both their training and inference processes require substantial computational resources, time, and memory. Consequently, optimizing this kind of models to minimize these requirements is crucial. In this article, we demonstrate that, with minimal resources and in a remarkably short time, it is possible to enhance a state-of-the-art model, specifically for a given language task, without compromising its overall capabilities using a relatively small pretrained LLM as a basis. Specifically, we present our use case, RigoChat 2, illustrating how LLMs can be adapted to achieve superior results in Spanish-language tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为现代人工智能的关键元素，能够在前所未有的准确性水平上处理广泛的语言处理任务，而无需收集特定问题的数据。然而，这些高度多功能的模型面临着一个重大挑战：它们的训练和推理过程需要大量的计算资源、时间和内存。因此，优化这类模型以最小化这些需求至关重要。在本文中，我们展示了一种方法，通过使用相对较小的预训练LLM作为基础，在极少量资源和极短的时间内，可以提升最新的模型性能，特别是在特定语言任务上，而不牺牲其整体能力。具体来说，我们以RigoChat 2为例，说明了如何调整LLM以在西班牙语任务上实现更优的结果。 

---
# ProTeX: Structure-In-Context Reasoning and Editing of Proteins with Large Language Models 

**Title (ZH)**: ProTeX: 结构在上下文中的蛋白质推理与编辑 

**Authors**: Zicheng Ma, Chuanliu Fan, Zhicong Wang, Zhenyu Chen, Xiaohan Lin, Yanheng Li, Shihao Feng, Jun Zhang, Ziqiang Cao, Yi Qin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08179)  

**Abstract**: Large language models have made remarkable progress in the field of molecular science, particularly in understanding and generating functional small molecules. This success is largely attributed to the effectiveness of molecular tokenization strategies. In protein science, the amino acid sequence serves as the sole tokenizer for LLMs. However, many fundamental challenges in protein science are inherently structure-dependent. The absence of structure-aware tokens significantly limits the capabilities of LLMs for comprehensive biomolecular comprehension and multimodal generation. To address these challenges, we introduce a novel framework, ProTeX, which tokenizes the protein sequences, structures, and textual information into a unified discrete space. This innovative approach enables joint training of the LLM exclusively through the Next-Token Prediction paradigm, facilitating multimodal protein reasoning and generation. ProTeX enables general LLMs to perceive and process protein structures through sequential text input, leverage structural information as intermediate reasoning components, and generate or manipulate structures via sequential text output. Experiments demonstrate that our model achieves significant improvements in protein function prediction, outperforming the state-of-the-art domain expert model with a twofold increase in accuracy. Our framework enables high-quality conformational generation and customizable protein design. For the first time, we demonstrate that by adopting the standard training and inference pipelines from the LLM domain, ProTeX empowers decoder-only LLMs to effectively address diverse spectrum of protein-related tasks. 

**Abstract (ZH)**: 大型语言模型在分子科学领域取得了显著进步，尤其是在理解和生成功能性小分子方面。这一成功主要归因于分子标记化策略的有效性。在蛋白质科学中，氨基酸序列是LLMs的唯一标记化方式。然而，蛋白质科学中的许多基本挑战本质上依赖于结构信息。缺乏结构感知的标记限制了LLMs在全面理解生物分子和多模态生成方面的能力。为了解决这些挑战，我们提出了一种新的框架ProTX，将蛋白质序列、结构和文本信息标记化到统一的离散空间中。这一创新方法使得LLMs仅通过Next-Token Prediction范式进行联合训练，从而实现多模态蛋白质推理和生成。ProTX使通用LLMs能够通过顺序文本输入感知和处理蛋白质结构，利用结构信息作为中间推理组件，并通过顺序文本输出生成或操控结构。实验表明，我们的模型在蛋白质功能预测方面取得了显著改进，准确率比最先进的领域专家模型提高了两倍。我们的框架能够生成高质量的构象并实现可定制的蛋白质设计。首次证明，通过采用LLM领域的标准训练和推理管道，ProTX使解码器仅LLMs能够有效应对蛋白质相关的各种任务。 

---
# Investigating the Effectiveness of a Socratic Chain-of-Thoughts Reasoning Method for Task Planning in Robotics, A Case Study 

**Title (ZH)**: 探究苏格拉底式连锁思维推理方法在机器人任务规划中的有效性：一个案例研究 

**Authors**: Veronica Bot, Zheyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.08174)  

**Abstract**: Large language models (LLMs) have demonstrated unprecedented capability in reasoning with natural language. Coupled with this development is the emergence of embodied AI in robotics. Despite showing promise for verbal and written reasoning tasks, it remains unknown whether LLMs are capable of navigating complex spatial tasks with physical actions in the real world. To this end, it is of interest to investigate applying LLMs to robotics in zero-shot learning scenarios, and in the absence of fine-tuning - a feat which could significantly improve human-robot interaction, alleviate compute cost, and eliminate low-level programming tasks associated with robot tasks.
To explore this question, we apply GPT-4(Omni) with a simulated Tiago robot in Webots engine for an object search task. We evaluate the effectiveness of three reasoning strategies based on Chain-of-Thought (CoT) sub-task list generation with the Socratic method (SocraCoT) (in order of increasing rigor): (1) Non-CoT/Non-SocraCoT, (2) CoT only, and (3) SocraCoT. Performance was measured in terms of the proportion of tasks successfully completed and execution time (N = 20). Our preliminary results show that when combined with chain-of-thought reasoning, the Socratic method can be used for code generation for robotic tasks that require spatial awareness. In extension of this finding, we propose EVINCE-LoC; a modified EVINCE method that could further enhance performance in highly complex and or dynamic testing scenarios. 

**Abstract (ZH)**: 大规模语言模型在自然语言推理方面展示了前所未有的能力。伴随着这一发展，机器人领域的嵌入式AI也出现了。虽然大规模语言模型在语言和书面推理任务中显示出前景，但在现实世界中是否能够通过物理动作导航复杂的空间任务仍不清楚。为此，有必要研究在零样本学习场景中将大规模语言模型应用于机器人技术的可能性，且无需微调——这一成就将显著改善人机交互，降低计算成本，并消除与机器人任务相关的低级编程任务。

为了探索这一问题，我们使用Webots引擎中的模拟Tiago机器人和GPT-4（全知）进行物体搜索任务。基于链式思维（CoT）子任务列表生成和苏格拉底方法（SocraCoT）的逐步严谨性，我们评估了三种推理策略的有效性：（1）非CoT/非SocraCoT，（2）仅CoT，（3）SocraCoT。性能通过成功完成任务的比例和执行时间（N = 20）进行衡量。初步结果显示，将链式思维推理与苏格拉底方法结合使用，可以用于生成需要空间意识的机器人任务的代码。在此基础上，我们提出了一种改进的EVINCE-LoC方法，可以在高度复杂或动态测试场景中进一步提高性能。 

---
# XAI4Extremes: An interpretable machine learning framework for understanding extreme-weather precursors under climate change 

**Title (ZH)**: XAI4Extremes: 一种在气候变化下理解极端天气前兆的可解释机器学习框架 

**Authors**: Jiawen Wei, Aniruddha Bora, Vivek Oommen, Chenyu Dong, Juntao Yang, Jeff Adie, Chen Chen, Simon See, George Karniadakis, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08163)  

**Abstract**: Extreme weather events are increasing in frequency and intensity due to climate change. This, in turn, is exacting a significant toll in communities worldwide. While prediction skills are increasing with advances in numerical weather prediction and artificial intelligence tools, extreme weather still present challenges. More specifically, identifying the precursors of such extreme weather events and how these precursors may evolve under climate change remain unclear. In this paper, we propose to use post-hoc interpretability methods to construct relevance weather maps that show the key extreme-weather precursors identified by deep learning models. We then compare this machine view with existing domain knowledge to understand whether deep learning models identified patterns in data that may enrich our understanding of extreme-weather precursors. We finally bin these relevant maps into different multi-year time periods to understand the role that climate change is having on these precursors. The experiments are carried out on Indochina heatwaves, but the methodology can be readily extended to other extreme weather events worldwide. 

**Abstract (ZH)**: 由于气候变迁，极端气候事件的频率和强度在不断增加，这在全球范围内给社区带来了显著的冲击。尽管随着数值天气预报和人工智能工具的进步，预测技巧在不断提高，但极端气候事件仍然存在挑战。更为具体地说，识别此类极端气候事件的前兆以及这些前兆在气候变化下的演变依然是不清楚的。本文提出使用后验可解释性方法构建由深度学习模型识别的关键极端气候事件前兆的相关天气地图。然后我们将这种机器视角与现有的领域知识进行比对，以了解深度学习模型是否在数据中识别出了可能丰富我们对极端气候事件前兆理解的模式。最后，我们将这些相关地图按不同的多年时间周期进行分类，以理解气候变化在这些前兆中的作用。实验在印度支那热浪上进行，但该方法可以方便地扩展到世界各地的其他极端气候事件。 

---
# Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking 

**Title (ZH)**: 注意力聚焦轨迹：面向轨迹的开放词汇跟踪 

**Authors**: Yunhao Li, Yifan Jiao, Dan Meng, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08145)  

**Abstract**: Open-Vocabulary Multi-Object Tracking (OV-MOT) aims to enable approaches to track objects without being limited to a predefined set of categories. Current OV-MOT methods typically rely primarily on instance-level detection and association, often overlooking trajectory information that is unique and essential for object tracking tasks. Utilizing trajectory information can enhance association stability and classification accuracy, especially in cases of occlusion and category ambiguity, thereby improving adaptability to novel classes. Thus motivated, in this paper we propose \textbf{TRACT}, an open-vocabulary tracker that leverages trajectory information to improve both object association and classification in OV-MOT. Specifically, we introduce a \textit{Trajectory Consistency Reinforcement} (\textbf{TCR}) strategy, that benefits tracking performance by improving target identity and category consistency. In addition, we present \textbf{TraCLIP}, a plug-and-play trajectory classification module. It integrates \textit{Trajectory Feature Aggregation} (\textbf{TFA}) and \textit{Trajectory Semantic Enrichment} (\textbf{TSE}) strategies to fully leverage trajectory information from visual and language perspectives for enhancing the classification results. Extensive experiments on OV-TAO show that our TRACT significantly improves tracking performance, highlighting trajectory information as a valuable asset for OV-MOT. Code will be released. 

**Abstract (ZH)**: 开放词汇多目标跟踪（OV-MOT）旨在使方法能够跟踪对象而不受预定义类别集的限制。现有的OV-MOT方法通常主要依赖实例级检测和关联，常常忽视了对于对象跟踪任务独特且至关重要的轨迹信息。利用轨迹信息可以增强关联的稳定性和分类准确性，尤其是在遮挡和类别模糊的情况下，从而提高对新型类别的适应性。由此激发，本文提出了一种名为TRACT的开放词汇跟踪器，该跟踪器利用轨迹信息以提高对象关联和分类表现。具体地，我们引入了一种轨迹一致性强化（TCR）策略，该策略通过提高目标身份和类别一致性来改进跟踪性能。此外，我们还提出了TraCLIP，这是一种插即用的轨迹分类模块，它结合了轨迹特征聚合（TFA）和轨迹语义丰富（TSE）策略，从视觉和语言视角充分利用轨迹信息以提升分类结果。在OV-TAO上的广泛实验表明，我们的TRACT显着提高了跟踪性能，突显了轨迹信息对于OV-MOT的价值。代码将开源。 

---
# FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems 

**Title (ZH)**: 流驱动后验采样方法：解决逆问题 

**Authors**: Jeongsol Kim, Bryan Sangwoo Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08136)  

**Abstract**: Flow matching is a recent state-of-the-art framework for generative modeling based on ordinary differential equations (ODEs). While closely related to diffusion models, it provides a more general perspective on generative modeling. Although inverse problem solving has been extensively explored using diffusion models, it has not been rigorously examined within the broader context of flow models. Therefore, here we extend the diffusion inverse solvers (DIS) - which perform posterior sampling by combining a denoising diffusion prior with an likelihood gradient - into the flow framework. Specifically, by driving the flow-version of Tweedie's formula, we decompose the flow ODE into two components: one for clean image estimation and the other for noise estimation. By integrating the likelihood gradient and stochastic noise into each component, respectively, we demonstrate that posterior sampling for inverse problem solving can be effectively achieved using flows. Our proposed solver, Flow-Driven Posterior Sampling (FlowDPS), can also be seamlessly integrated into a latent flow model with a transformer architecture. Across four linear inverse problems, we confirm that FlowDPS outperforms state-of-the-art alternatives, all without requiring additional training. 

**Abstract (ZH)**: 基于普通微分方程的流匹配是一种_recent_state-of-the-art_生成建模框架。虽然它与扩散模型密切相关，但提供了生成建模更为普适的观点。尽管扩散模型在逆问题求解方面得到了广泛探索，但在流模型更广泛的背景下，其逆问题求解能力尚未得到严格检验。因此，我们在此将扩散逆求解器(DIS)扩展到流框架中，该求解器通过结合去噪扩散先验和似然梯度进行后验采样。通过推动流版本的泰迪法则，我们将流ODE分解为两部分：一部分用于估计清晰图像，另一部分用于估计噪声。通过在每一部分分别整合似然梯度和随机噪声，展示了逆问题求解的后验采样可以有效利用流进行。我们提出的解算器Flow-Driven Posterior Sampling (FlowDPS)还可以无缝集成到具有transformer架构的潜在流模型中。在四种线性逆问题中，我们确认FlowDPS优于现有最佳方案，且无需额外训练。 

---
# MGHanD: Multi-modal Guidance for authentic Hand Diffusion 

**Title (ZH)**: MGHanD: 多模态指导下的真实手部扩散 

**Authors**: Taehyeon Eum, Jieun Choi, Tae-Kyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.08133)  

**Abstract**: Diffusion-based methods have achieved significant successes in T2I generation, providing realistic images from text prompts. Despite their capabilities, these models face persistent challenges in generating realistic human hands, often producing images with incorrect finger counts and structurally deformed hands. MGHanD addresses this challenge by applying multi-modal guidance during the inference process. For visual guidance, we employ a discriminator trained on a dataset comprising paired real and generated images with captions, derived from various hand-in-the-wild datasets. We also employ textual guidance with LoRA adapter, which learns the direction from `hands' towards more detailed prompts such as `natural hands', and `anatomically correct fingers' at the latent level. A cumulative hand mask which is gradually enlarged in the assigned time step is applied to the added guidance, allowing the hand to be refined while maintaining the rich generative capabilities of the pre-trained model. In the experiments, our method achieves superior hand generation qualities, without any specific conditions or priors. We carry out both quantitative and qualitative evaluations, along with user studies, to showcase the benefits of our approach in producing high-quality hand images. 

**Abstract (ZH)**: 基于扩散的方法在文本到图像生成中取得了显著成功，能够从文本提示中生成逼真的图像。尽管这些模型具有这些能力，但在生成逼真的手部图像方面仍面临持续挑战，常常产生手指数量错误和结构变形的手部图像。MGHanD 通过在推理过程中应用多模态指导来应对这一挑战。对于视觉指导，我们使用一个在包含配对的真实和生成图像及其描述（来自各种野外手部数据集）的数据集上训练的判别器。我们还使用 LoRA 适配器进行文本指导，该适配器在潜在空间中学习从“手”向更详细的提示（如“自然的手部”和“解剖学正确的手指”）的方向。通过在指定的时间步逐渐扩大的累积手部掩码应用于附加的指导，使手部可以得到细化，同时保持预训练模型丰富的生成能力。在实验中，我们的方法在生成高质量手部图像方面表现优异，无需任何特定条件或先验知识。我们进行了定量和定性的评估，并进行了用户研究，以展示我们方法在生成高质量手部图像方面的 Benefits。 

---
# Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments 

**Title (ZH)**: 朝向稳定的世界模型：衡量与解决生成环境中世界不稳定性的方法 

**Authors**: Soonwoo Kwon, Jin-Young Kim, Hyojun Go, Kyungjune Baek  

**Link**: [PDF](https://arxiv.org/pdf/2503.08122)  

**Abstract**: We present a novel study on enhancing the capability of preserving the content in world models, focusing on a property we term World Stability. Recent diffusion-based generative models have advanced the synthesis of immersive and realistic environments that are pivotal for applications such as reinforcement learning and interactive game engines. However, while these models excel in quality and diversity, they often neglect the preservation of previously generated scenes over time--a shortfall that can introduce noise into agent learning and compromise performance in safety-critical settings. In this work, we introduce an evaluation framework that measures world stability by having world models perform a sequence of actions followed by their inverses to return to their initial viewpoint, thereby quantifying the consistency between the starting and ending observations. Our comprehensive assessment of state-of-the-art diffusion-based world models reveals significant challenges in achieving high world stability. Moreover, we investigate several improvement strategies to enhance world stability. Our results underscore the importance of world stability in world modeling and provide actionable insights for future research in this domain. 

**Abstract (ZH)**: 一种提升世界模型内容保持能力的新型研究：着眼于我们称为世界稳定性的特性 

---
# Uni$\textbf{F}^2$ace: Fine-grained Face Understanding and Generation with Unified Multimodal Models 

**Title (ZH)**: Uni$\textbf{F}^2$ace: 统一多模态模型下的细粒度面部理解和生成 

**Authors**: Junzhe Li, Xuerui Qiu, Linrui Xu, Liya Guo, Delin Qu, Tingting Long, Chun Fan, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08120)  

**Abstract**: Unified multimodal models (UMMs) have emerged as a powerful paradigm in foundational computer vision research, demonstrating significant potential in both image understanding and generation. However, existing research in the face domain primarily focuses on $\textbf{coarse}$ facial attribute understanding, with limited capacity to handle $\textbf{fine-grained}$ facial attributes and without addressing generation capabilities. To overcome these limitations, we propose Uni$\textbf{F}^2$ace, the first UMM tailored specifically for fine-grained face understanding and generation. In general, we train Uni$\textbf{F}^2$ace on a self-constructed, specialized dataset utilizing two mutually beneficial diffusion techniques and a two-level mixture-of-experts architecture. Specifically, we first build a large-scale facial dataset, Uni$\textbf{F}^2$ace-130K, which contains 130K image-text pairs with one million question-answering pairs that span a wide range of facial attributes. Second, we establish a theoretical connection between discrete diffusion score matching and masked generative models, optimizing both evidence lower bounds simultaneously, which significantly improves the model's ability to synthesize facial details. Finally, we introduce both token-level and sequence-level mixture-of-experts, enabling efficient fine-grained representation learning for both understanding and generation tasks. Extensive experiments on Uni$\textbf{F}^2$ace-130K demonstrate that Uni$\textbf{F}^2$ace outperforms existing UMMs and generative models, achieving superior performance across both understanding and generation tasks. 

**Abstract (ZH)**: 统一多模态模型（UMMs）在基础计算机视觉研究中 emerged as a powerful paradigm, demonstrating significant potential in both image understanding and generation. However, existing research in the face domain primarily focuses on coarse facial attribute understanding, with limited capacity to handle fine-grained facial attributes and without addressing generation capabilities. To overcome these limitations, we propose UniF²ace, the first UMM tailored specifically for fine-grained face understanding and generation. In general, we train UniF²ace on a self-constructed, specialized dataset utilizing two mutually beneficial diffusion techniques and a two-level mixture-of-experts architecture. Specifically, we first build a large-scale facial dataset, UniF²ace-130K, which contains 130K image-text pairs with one million question-answering pairs that span a wide range of facial attributes. Second, we establish a theoretical connection between discrete diffusion score matching and masked generative models, optimizing both evidence lower bounds simultaneously, which significantly improves the model's ability to synthesize facial details. Finally, we introduce both token-level and sequence-level mixture-of-experts, enabling efficient fine-grained representation learning for both understanding and generation tasks. Extensive experiments on UniF²ace-130K demonstrate that UniF²ace outperforms existing UMMs and generative models, achieving superior performance across both understanding and generation tasks. 

---
# Convergence Dynamics and Stabilization Strategies of Co-Evolving Generative Models 

**Title (ZH)**: 共演化的生成模型的收敛动力学与稳定化策略 

**Authors**: Weiguo Gao, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08117)  

**Abstract**: The increasing prevalence of synthetic data in training loops has raised concerns about model collapse, where generative models degrade when trained on their own outputs. While prior work focuses on this self-consuming process, we study an underexplored yet prevalent phenomenon: co-evolving generative models that shape each other's training through iterative feedback. This is common in multimodal AI ecosystems, such as social media platforms, where text models generate captions that guide image models, and the resulting images influence the future adaptation of the text model. We take a first step by analyzing such a system, modeling the text model as a multinomial distribution and the image model as a conditional multi-dimensional Gaussian distribution. Our analysis uncovers three key results. First, when one model remains fixed, the other collapses: a frozen image model causes the text model to lose diversity, while a frozen text model leads to an exponential contraction of image diversity, though fidelity remains bounded. Second, in fully interactive systems, mutual reinforcement accelerates collapse, with image contraction amplifying text homogenization and vice versa, leading to a Matthew effect where dominant texts sustain higher image diversity while rarer texts collapse faster. Third, we analyze stabilization strategies implicitly introduced by real-world external influences. Random corpus injections for text models and user-content injections for image models prevent collapse while preserving both diversity and fidelity. Our theoretical findings are further validated through experiments. 

**Abstract (ZH)**: 合成数据在训练环中的增多引发了对模型崩溃的担忧，即生成模型在使用自身输出进行训练时性能下降。尽管先前的工作主要关注这一自我消费的过程，我们研究了一个尚未充分探索但普遍存在的现象：通过迭代反馈相互演变的生成模型，它们通过训练相互塑造。这一现象在多模态AI生态系统中尤为常见，例如社交媒体平台，其中文本模型生成描述图像的字幕，这些字幕反过来影响未来文本模型的适应性。我们首次分析了该系统，将文本模型建模为多项式分布，图像模型建模为条件多维正态分布。我们的分析揭示了三个关键结果。首先，当一个模型固定不变时，另一个模型会出现崩溃：冻结的图像模型会导致文本模型失去多样性，而冻结的文本模型会导致图像多样性的指数级收缩，尽管保真度保持在一定范围内。其次，在完全相互作用的系统中，相互强化加速了崩溃，图像多样性收缩放大了文本同质化，反之亦然，导致马太效应，即主导文本保持更高的图像多样性而少见文本则崩溃得更快。第三，我们分析了由真实世界外部影响隐式引入的稳定化策略。对于文本模型，随机文本语料库的注入；对于图像模型，用户内容的注入，可以防止崩溃同时保持多样性和保真度。我们的理论发现通过实验得到了进一步验证。 

---
# Revolution of Wireless Signal Recognition for 6G: Recent Advances, Challenges and Future Directions 

**Title (ZH)**: 6G无线信号识别革命：近期进展、挑战与未来方向 

**Authors**: Hao Zhang, Fuhui Zhou, Hongyang Du, Qihui Wu, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08091)  

**Abstract**: Wireless signal recognition (WSR) is a crucial technique for intelligent communications and spectrum sharing in the next six-generation (6G) wireless communication networks. It can be utilized to enhance network performance and efficiency, improve quality of service (QoS), and improve network security and reliability. Additionally, WSR can be applied for military applications such as signal interception, signal race, and signal abduction. In the past decades, great efforts have been made for the research of WSR. Earlier works mainly focus on model-based methods, including likelihood-based (LB) and feature-based (FB) methods, which have taken the leading position for many years. With the emergence of artificial intelligence (AI), intelligent methods including machine learning-based (ML-based) and deep learning-based (DL-based) methods have been developed to extract the features of the received signals and perform the classification. In this work, we provide a comprehensive review of WSR from the view of applications, main tasks, recent advances, datasets and evaluation metrics, challenges, and future directions. Specifically, intelligent WSR methods are introduced from the perspective of model, data, learning and implementation. Moreover, we analyze the challenges for WSR from the view of complex, dynamic, and open 6G wireless environments and discuss the future directions for WSR. This survey is expected to provide a comprehensive overview of the state-of-the-art WSR techniques and inspire new research directions for WSR in 6G networks. 

**Abstract (ZH)**: 无线信号识别（WSR）在六代（6G）无线通信网络的智能通信和频谱共享中的关键技术及其应用 

---
# Instruction-Augmented Long-Horizon Planning: Embedding Grounding Mechanisms in Embodied Mobile Manipulation 

**Title (ZH)**: 基于指令增强的长时规划：嵌入式实体移动 manipulative 机器人的 grounding 机制 

**Authors**: Fangyuan Wang, Shipeng Lyu, Peng Zhou, Anqing Duan, Guodong Guo, David Navarro-Alarcon  

**Link**: [PDF](https://arxiv.org/pdf/2503.08084)  

**Abstract**: Enabling humanoid robots to perform long-horizon mobile manipulation planning in real-world environments based on embodied perception and comprehension abilities has been a longstanding challenge. With the recent rise of large language models (LLMs), there has been a notable increase in the development of LLM-based planners. These approaches either utilize human-provided textual representations of the real world or heavily depend on prompt engineering to extract such representations, lacking the capability to quantitatively understand the environment, such as determining the feasibility of manipulating objects. To address these limitations, we present the Instruction-Augmented Long-Horizon Planning (IALP) system, a novel framework that employs LLMs to generate feasible and optimal actions based on real-time sensor feedback, including grounded knowledge of the environment, in a closed-loop interaction. Distinct from prior works, our approach augments user instructions into PDDL problems by leveraging both the abstract reasoning capabilities of LLMs and grounding mechanisms. By conducting various real-world long-horizon tasks, each consisting of seven distinct manipulatory skills, our results demonstrate that the IALP system can efficiently solve these tasks with an average success rate exceeding 80%. Our proposed method can operate as a high-level planner, equipping robots with substantial autonomy in unstructured environments through the utilization of multi-modal sensor inputs. 

**Abstract (ZH)**: 基于嵌入式感知与理解能力的大型语言模型驱动的人形机器人长期移动操作规划实现 

---
# Degradation Self-Supervised Learning for Lithium-ion Battery Health Diagnostics 

**Title (ZH)**: 锂离子电池健康诊断的降级自监督学习 

**Authors**: J. C. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08083)  

**Abstract**: Health evaluation for lithium-ion batteries (LIBs) typically relies on constant charging/discharging protocols, often neglecting scenarios involving dynamic current profiles prevalent in electric vehicles. Conventional health indicators for LIBs also depend on the uniformity of measured data, restricting their adaptability to non-uniform conditions. In this study, a novel training strategy for estimating LIB health based on the paradigm of self-supervised learning is proposed. A multiresolution analysis technique, empirical wavelet transform, is utilized to decompose non-stationary voltage signals in the frequency domain. This allows the removal of ineffective components for the health evaluation model. The transformer neural network serves as the model backbone, and a loss function is designed to describe the capacity degradation behavior with the assumption that the degradation in LIBs across most operating conditions is inevitable and irreversible. The results show that the model can learn the aging characteristics by analyzing sequences of voltage and current profiles obtained at various time intervals from the same LIB cell. The proposed method is successfully applied to the Stanford University LIB aging dataset, derived from electric vehicle real driving profiles. Notably, this approach achieves an average correlation coefficient of 0.9 between the evaluated health index and the degradation of actual capacity, demonstrating its efficacy in capturing LIB health degradation. This research highlights the feasibility of training deep neural networks using unlabeled LIB data, offering cost-efficient means and unleashing the potential of the measured information. 

**Abstract (ZH)**: 基于自我监督学习 paradigm 估计锂离子电池 (LIBs) 健康状态的新培训策略：多分辨率分析在非平稳电压信号中的应用及其在变压器神经网络中的实现 

---
# Continual Learning for Multiple Modalities 

**Title (ZH)**: 多模态持续学习 

**Authors**: Hyundong Jin, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.08064)  

**Abstract**: Continual learning aims to learn knowledge of tasks observed in sequential time steps while mitigating the forgetting of previously learned knowledge. Existing methods were proposed under the assumption of learning a single modality (e.g., image) over time, which limits their applicability in scenarios involving multiple modalities. In this work, we propose a novel continual learning framework that accommodates multiple modalities (image, video, audio, depth, and text). We train a model to align various modalities with text, leveraging its rich semantic information. However, this increases the risk of forgetting previously learned knowledge, exacerbated by the differing input traits of each task. To alleviate the overwriting of the previous knowledge of modalities, we propose a method for aggregating knowledge within and across modalities. The aggregated knowledge is obtained by assimilating new information through self-regularization within each modality and associating knowledge between modalities by prioritizing contributions from relevant modalities. Furthermore, we propose a strategy that re-aligns the embeddings of modalities to resolve biased alignment between modalities. We evaluate the proposed method in a wide range of continual learning scenarios using multiple datasets with different modalities. Extensive experiments demonstrate that ours outperforms existing methods in the scenarios, regardless of whether the identity of the modality is given. 

**Abstract (ZH)**: 多模态持续学习框架：融合图像、视频、音频、深度和文本信息以减轻遗忘风险 

---
# Generalized Kullback-Leibler Divergence Loss 

**Title (ZH)**: 广义KL散度假损失 

**Authors**: Jiequan Cui, Beier Zhu, Qingshan Xu, Zhuotao Tian, Xiaojuan Qi, Bei Yu, Hanwang Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08038)  

**Abstract**: In this paper, we delve deeper into the Kullback-Leibler (KL) Divergence loss and mathematically prove that it is equivalent to the Decoupled Kullback-Leibler (DKL) Divergence loss that consists of (1) a weighted Mean Square Error (wMSE) loss and (2) a Cross-Entropy loss incorporating soft labels. Thanks to the decoupled structure of DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of KL loss in scenarios like knowledge distillation by breaking its asymmetric optimization property along with a smoother weight function. This modification effectively alleviates convergence challenges in optimization, particularly for classes with high predicted scores in soft labels. Secondly, we introduce class-wise global information into KL/DKL to reduce bias arising from individual samples. With these two enhancements, we derive the Generalized Kullback-Leibler (GKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100, ImageNet, and vision-language datasets, focusing on adversarial training, and knowledge distillation tasks. Specifically, we achieve new state-of-the-art adversarial robustness on the public leaderboard -- RobustBench and competitive knowledge distillation performance across CIFAR/ImageNet models and CLIP models, demonstrating the substantial practical merits. Our code is available at this https URL. 

**Abstract (ZH)**: 本文深入探讨了Kullback-Leibler (KL) 散度损失，并通过数学证明将其等同于由加权均方误差（wMSE）损失和结合软标签的交叉熵损失组成的解耦Kullback-Leibler (DKL) 散度损失。得益于DKL损失的解耦结构，我们识别了两个改进领域。首先，我们通过打破其不对称优化特性并引入更平滑的权重函数来解决KL损失在知识蒸馏等场景中的局限性。这一修改有效地缓解了优化过程中的收敛挑战，特别是在软标签中高预测分值类别的优化问题。其次，我们将类别级别的全局信息引入KL/DKL中，以减少由单个样本引起的偏差。通过这两个增强，我们推导出广义Kullback-Leibler (GKL) 散度损失，并通过在CIFAR-10/100、ImageNet和视觉语言数据集上的实验，以及在对抗训练和知识蒸馏任务中的评估来验证其有效性。具体而言，我们在公共排行榜RobustBench上实现了新的对抗鲁棒性状态，并在CIFAR/ImageNet模型和CLIP模型中的知识蒸馏性能方面取得了竞争力的表现，表明了其实用价值的显著性。我们的代码可在以下链接获取。 

---
# ObjectMover: Generative Object Movement with Video Prior 

**Title (ZH)**: ObjectMover: 基于视频先验的生成性物体运动 

**Authors**: Xin Yu, Tianyu Wang, Soo Ye Kim, Paul Guerrero, Xi Chen, Qing Liu, Zhe Lin, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2503.08037)  

**Abstract**: Simple as it seems, moving an object to another location within an image is, in fact, a challenging image-editing task that requires re-harmonizing the lighting, adjusting the pose based on perspective, accurately filling occluded regions, and ensuring coherent synchronization of shadows and reflections while maintaining the object identity. In this paper, we present ObjectMover, a generative model that can perform object movement in highly challenging scenes. Our key insight is that we model this task as a sequence-to-sequence problem and fine-tune a video generation model to leverage its knowledge of consistent object generation across video frames. We show that with this approach, our model is able to adjust to complex real-world scenarios, handling extreme lighting harmonization and object effect movement. As large-scale data for object movement are unavailable, we construct a data generation pipeline using a modern game engine to synthesize high-quality data pairs. We further propose a multi-task learning strategy that enables training on real-world video data to improve the model generalization. Through extensive experiments, we demonstrate that ObjectMover achieves outstanding results and adapts well to real-world scenarios. 

**Abstract (ZH)**: 看起来简单，实际上，在图像中将一个物体移动到另一个位置是一项具有挑战性的编辑任务，需要重新协调光照、基于透视调整姿态、准确填补遮挡区域，并确保在保持物体身份的同时，阴影和反射的一致同步。在本文中，我们提出了一种生成模型ObjectMover，能够在高度挑战性的场景中执行物体移动任务。我们的关键洞察是将此任务建模为序列到序列问题，并微调一个视频生成模型，使其能够利用其在视频帧间一致生成物体的知识。我们表明，通过这种方法，我们的模型能够适应复杂的现实世界场景，处理极端光照协调和物体效果移动。由于可用于物体移动的大规模数据不可用，我们使用现代游戏引擎构建了一个数据生成管道，以合成高质量的数据对。我们进一步提出了一种多任务学习策略，使模型能够在现实世界视频数据上进行训练，从而提高模型的泛化能力。通过广泛的实验，我们证明了ObjectMover取得了出色的结果，并能很好地适应现实世界场景。 

---
# HOFAR: High-Order Augmentation of Flow Autoregressive Transformers 

**Title (ZH)**: 高阶流自回归变换器的增广方法 

**Authors**: Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, Mingda Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08032)  

**Abstract**: Flow Matching and Transformer architectures have demonstrated remarkable performance in image generation tasks, with recent work FlowAR [Ren et al., 2024] synergistically integrating both paradigms to advance synthesis fidelity. However, current FlowAR implementations remain constrained by first-order trajectory modeling during the generation process. This paper introduces a novel framework that systematically enhances flow autoregressive transformers through high-order supervision. We provide theoretical analysis and empirical evaluation showing that our High-Order FlowAR (HOFAR) demonstrates measurable improvements in generation quality compared to baseline models. The proposed approach advances the understanding of flow-based autoregressive modeling by introducing a systematic framework for analyzing trajectory dynamics through high-order expansion. 

**Abstract (ZH)**: 高阶监督下的流匹配与变换器架构在图像生成任务中的系统性提升 

---
# In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents 

**Title (ZH)**: 展望与回顾：长期个性化对话代理的反射式内存管理 

**Authors**: Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2503.08026)  

**Abstract**: Large Language Models (LLMs) have made significant progress in open-ended dialogue, yet their inability to retain and retrieve relevant information from long-term interactions limits their effectiveness in applications requiring sustained personalization. External memory mechanisms have been proposed to address this limitation, enabling LLMs to maintain conversational continuity. However, existing approaches struggle with two key challenges. First, rigid memory granularity fails to capture the natural semantic structure of conversations, leading to fragmented and incomplete representations. Second, fixed retrieval mechanisms cannot adapt to diverse dialogue contexts and user interaction patterns. In this work, we propose Reflective Memory Management (RMM), a novel mechanism for long-term dialogue agents, integrating forward- and backward-looking reflections: (1) Prospective Reflection, which dynamically summarizes interactions across granularities-utterances, turns, and sessions-into a personalized memory bank for effective future retrieval, and (2) Retrospective Reflection, which iteratively refines the retrieval in an online reinforcement learning (RL) manner based on LLMs' cited evidence. Experiments show that RMM demonstrates consistent improvement across various metrics and benchmarks. For example, RMM shows more than 10% accuracy improvement over the baseline without memory management on the LongMemEval dataset. 

**Abstract (ZH)**: 长周期对话代理的反思性记忆管理（Reflective Memory Management for Long-term Dialogue Agents） 

---
# Exploring Bias in over 100 Text-to-Image Generative Models 

**Title (ZH)**: 探索超过100个文本到图像生成模型中的偏差 

**Authors**: Jordan Vice, Naveed Akhtar, Richard Hartley, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.08012)  

**Abstract**: We investigate bias trends in text-to-image generative models over time, focusing on the increasing availability of models through open platforms like Hugging Face. While these platforms democratize AI, they also facilitate the spread of inherently biased models, often shaped by task-specific fine-tuning. Ensuring ethical and transparent AI deployment requires robust evaluation frameworks and quantifiable bias metrics. To this end, we assess bias across three key dimensions: (i) distribution bias, (ii) generative hallucination, and (iii) generative miss-rate. Analyzing over 100 models, we reveal how bias patterns evolve over time and across generative tasks. Our findings indicate that artistic and style-transferred models exhibit significant bias, whereas foundation models, benefiting from broader training distributions, are becoming progressively less biased. By identifying these systemic trends, we contribute a large-scale evaluation corpus to inform bias research and mitigation strategies, fostering more responsible AI development.
Keywords: Bias, Ethical AI, Text-to-Image, Generative Models, Open-Source Models 

**Abstract (ZH)**: 我们探讨了文本到图像生成模型随时间的偏差趋势，重点关注通过Hugging Face等开放平台增加的模型可用性。虽然这些平台民主化了AI，但也促进了固有偏差模型的传播，这些模型往往是由特定任务的微调所形成的。确保伦理和透明的AI部署需要 robust 评价框架和可量化的偏差指标。为此，我们从三个关键维度评估偏差：(i) 分布偏差，(ii) 生成幻觉，(iii) 生成失率。通过对超过100个模型的分析，我们揭示了偏差模式随时间和生成任务的变化情况。我们的研究结果表明，艺术性和风格迁移模型具有显著的偏差，而得益于更广泛训练分布的基础模型，其偏差正在逐渐减少。通过识别这些系统性趋势，我们贡献了一个大规模的评价语料库，以指导偏差研究和缓解策略，促进更加负责任的AI开发。

关键词：偏差，伦理AI，文本到图像，生成模型，开源模型。 

---
# SKALD: Learning-Based Shot Assembly for Coherent Multi-Shot Video Creation 

**Title (ZH)**: SKALD：基于学习的单帧组装用于连贯多帧视频创作 

**Authors**: Chen Yi Lu, Md Mehrab Tanjim, Ishita Dasgupta, Somdeb Sarkhel, Gang Wu, Saayan Mitra, Somali Chaterji  

**Link**: [PDF](https://arxiv.org/pdf/2503.08010)  

**Abstract**: We present SKALD, a multi-shot video assembly method that constructs coherent video sequences from candidate shots with minimal reliance on text. Central to our approach is the Learned Clip Assembly (LCA) score, a learning-based metric that measures temporal and semantic relationships between shots to quantify narrative coherence. We tackle the exponential complexity of combining multiple shots with an efficient beam-search algorithm guided by the LCA score. To train our model effectively with limited human annotations, we propose two tasks for the LCA encoder: Shot Coherence Learning, which uses contrastive learning to distinguish coherent and incoherent sequences, and Feature Regression, which converts these learned representations into a real-valued coherence score. We develop two variants: a base SKALD model that relies solely on visual coherence and SKALD-text, which integrates auxiliary text information when available. Experiments on the VSPD and our curated MSV3C datasets show that SKALD achieves an improvement of up to 48.6% in IoU and a 43% speedup over the state-of-the-art methods. A user study further validates our approach, with 45% of participants favoring SKALD-assembled videos, compared to 22% preferring text-based assembly methods. 

**Abstract (ZH)**: SKALD：一种基于学习剪辑组装的多 shot 视频拼接方法 

---
# MoRE: Unlocking Scalability in Reinforcement Learning for Quadruped Vision-Language-Action Models 

**Title (ZH)**: MoRE: 解锁 quadruped 视听行动模型在强化学习中的可扩展性 

**Authors**: Han Zhao, Wenxuan Song, Donglin Wang, Xinyang Tong, Pengxiang Ding, Xuelian Cheng, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2503.08007)  

**Abstract**: Developing versatile quadruped robots that can smoothly perform various actions and tasks in real-world environments remains a significant challenge. This paper introduces a novel vision-language-action (VLA) model, mixture of robotic experts (MoRE), for quadruped robots that aim to introduce reinforcement learning (RL) for fine-tuning large-scale VLA models with a large amount of mixed-quality data. MoRE integrates multiple low-rank adaptation modules as distinct experts within a dense multi-modal large language model (MLLM), forming a sparse-activated mixture-of-experts model. This design enables the model to effectively adapt to a wide array of downstream tasks. Moreover, we employ a reinforcement learning-based training objective to train our model as a Q-function after deeply exploring the structural properties of our tasks. Effective learning from automatically collected mixed-quality data enhances data efficiency and model performance. Extensive experiments demonstrate that MoRE outperforms all baselines across six different skills and exhibits superior generalization capabilities in out-of-distribution scenarios. We further validate our method in real-world scenarios, confirming the practicality of our approach and laying a solid foundation for future research on multi-task learning in quadruped robots. 

**Abstract (ZH)**: 基于视觉-语言-动作的混合机器人专家模型MoRE：一种适用于 quadruped 机器人的 reinforcement learning 方法 

---
# Injecting Imbalance Sensitivity for Multi-Task Learning 

**Title (ZH)**: 在多任务学习中注入不平衡敏感性 

**Authors**: Zhipeng Zhou, Liu Liu, Peilin Zhao, Wei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08006)  

**Abstract**: Multi-task learning (MTL) has emerged as a promising approach for deploying deep learning models in real-life applications. Recent studies have proposed optimization-based learning paradigms to establish task-shared representations in MTL. However, our paper empirically argues that these studies, specifically gradient-based ones, primarily emphasize the conflict issue while neglecting the potentially more significant impact of imbalance/dominance in MTL. In line with this perspective, we enhance the existing baseline method by injecting imbalance-sensitivity through the imposition of constraints on the projected norms. To demonstrate the effectiveness of our proposed IMbalance-sensitive Gradient (IMGrad) descent method, we evaluate it on multiple mainstream MTL benchmarks, encompassing supervised learning tasks as well as reinforcement learning. The experimental results consistently demonstrate competitive performance. 

**Abstract (ZH)**: 多任务学习中基于优化的具有不平衡敏感性的梯度下降方法 

---
# A Neural Symbolic Model for Space Physics 

**Title (ZH)**: 空间物理中的神经符号模型 

**Authors**: Jie Ying, Haowei Lin, Chao Yue, Yajie Chen, Chao Xiao, Quanqi Shi, Yitao Liang, Shing-Tung Yau, Yuan Zhou, Jianzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.07994)  

**Abstract**: In this study, we unveil a new AI model, termed PhyE2E, to discover physical formulas through symbolic regression. PhyE2E simplifies symbolic regression by decomposing it into sub-problems using the second-order derivatives of an oracle neural network, and employs a transformer model to translate data into symbolic formulas in an end-to-end manner. The resulting formulas are refined through Monte-Carlo Tree Search and Genetic Programming. We leverage a large language model to synthesize extensive symbolic expressions resembling real physics, and train the model to recover these formulas directly from data. A comprehensive evaluation reveals that PhyE2E outperforms existing state-of-the-art approaches, delivering superior symbolic accuracy, precision in data fitting, and consistency in physical units. We deployed PhyE2E to five applications in space physics, including the prediction of sunspot numbers, solar rotational angular velocity, emission line contribution functions, near-Earth plasma pressure, and lunar-tide plasma signals. The physical formulas generated by AI demonstrate a high degree of accuracy in fitting the experimental data from satellites and astronomical telescopes. We have successfully upgraded the formula proposed by NASA in 1993 regarding solar activity, and for the first time, provided the explanations for the long cycle of solar activity in an explicit form. We also found that the decay of near-Earth plasma pressure is proportional to r^2 to Earth, where subsequent mathematical derivations are consistent with satellite data from another independent study. Moreover, we found physical formulas that can describe the relationships between emission lines in the extreme ultraviolet spectrum of the Sun, temperatures, electron densities, and magnetic fields. The formula obtained is consistent with the properties that physicists had previously hypothesized it should possess. 

**Abstract (ZH)**: 本研究提出了一个新的AI模型PhyE2E，用于通过符号回归发现物理公式。PhyE2E通过使用占优神经网络的二阶导数将符号回归分解为子问题，并采用变压器模型以端到端的方式将数据转化为符号公式。生成的公式通过蒙特卡洛树搜索和遗传编程进行优化。我们利用大规模语言模型合成大量类似真实物理的符号表达式，并训练模型直接从数据中恢复这些公式。全面评估表明，PhyE2E的表现优于现有最先进的方法，具备更高的符号准确性、数据拟合精度和物理单位一致性。我们将在空间物理学中的五个应用中部署PhyE2E，包括太阳黑子数预测、太阳自转角速度、发射线贡献函数、地球附近等离子体压力和月潮等离子信号。AI生成的物理公式在拟合卫星和天文望远镜的实验数据方面具有很高的准确性。我们成功升级了NASA于1993年关于太阳活动的公式，并首次以明确形式提供了太阳活动长周期的解释。我们还发现，地球附近的等离子体压力衰减与地球距离的平方成正比，后续的数学推导与另一个独立研究的卫星数据一致。此外，我们发现了描述极端紫外太阳谱发射线之间的关系、温度、电子密度和磁场的物理公式，所获得的公式与物理学家之前推测的性质一致。 

---
# Efficient and Accurate Estimation of Lipschitz Constants for Hybrid Quantum-Classical Decision Models 

**Title (ZH)**: 混合量子-经典决策模型的高效准确Lipschitz常数估计 

**Authors**: Sajjad Hashemian, Mohammad Saeed Arvenaghi  

**Link**: [PDF](https://arxiv.org/pdf/2503.07992)  

**Abstract**: In this paper, we propose a novel framework for efficiently and accurately estimating Lipschitz constants in hybrid quantum-classical decision models. Our approach integrates classical neural network with quantum variational circuits to address critical issues in learning theory such as fairness verification, robust training, and generalization.
By a unified convex optimization formulation, we extend existing classical methods to capture the interplay between classical and quantum layers. This integrated strategy not only provide a tight bound on the Lipschitz constant but also improves computational efficiency with respect to the previous methods. 

**Abstract (ZH)**: 本文提出了一种新型框架，用于高效准确地估计混合量子-经典决策模型中的Lipschitz常数。该方法将经典神经网络与量子变分电路集成，以解决学习理论中公平性验证、鲁棒训练与泛化等关键问题。通过统一的凸优化形式，我们将现有的经典方法扩展以捕获经典层与量子层之间的交互作用。这种集成策略不仅提供了Lipschitz常数的紧致界，还相对于之前的方法提高了计算效率。 

---
# Provable Zero-Shot Generalization in Offline Reinforcement Learning 

**Title (ZH)**: 可验证的零样本泛化在离线强化学习中 

**Authors**: Zhiyong Wang, Chen Yang, John C.S. Lui, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.07988)  

**Abstract**: In this work, we study offline reinforcement learning (RL) with zero-shot generalization property (ZSG), where the agent has access to an offline dataset including experiences from different environments, and the goal of the agent is to train a policy over the training environments which performs well on test environments without further interaction. Existing work showed that classical offline RL fails to generalize to new, unseen environments. We propose pessimistic empirical risk minimization (PERM) and pessimistic proximal policy optimization (PPPO), which leverage pessimistic policy evaluation to guide policy learning and enhance generalization. We show that both PERM and PPPO are capable of finding a near-optimal policy with ZSG. Our result serves as a first step in understanding the foundation of the generalization phenomenon in offline reinforcement learning. 

**Abstract (ZH)**: 在本工作中，我们研究了具有零-shot泛化能力（ZSG）的离线强化学习（RL），其中智能体可以访问包含不同环境经验的离线数据集，并且智能体的目标是训练一个在训练环境上的策略能够在测试环境上表现良好，而无需进一步交互。现有研究表明，经典离线RL无法泛化到新的未见过的环境。我们提出悲观经验风险最小化（PERM）和悲观近端策略优化（PPPO），利用悲观策略评估来指导策略学习并增强泛化能力。我们证明了PERM和PPPO都能够通过ZSG找到接近最优的策略。我们的结果为进一步理解离线强化学习中的泛化现象奠定了基础。 

---
# Hierarchical Contact-Rich Trajectory Optimization for Multi-Modal Manipulation using Tight Convex Relaxations 

**Title (ZH)**: 多模态操作中基于紧密凸松弛的层次密集接触轨迹优化 

**Authors**: Yuki Shirai, Arvind Raghunathan, Devesh K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2503.07963)  

**Abstract**: Designing trajectories for manipulation through contact is challenging as it requires reasoning of object \& robot trajectories as well as complex contact sequences simultaneously. In this paper, we present a novel framework for simultaneously designing trajectories of robots, objects, and contacts efficiently for contact-rich manipulation. We propose a hierarchical optimization framework where Mixed-Integer Linear Program (MILP) selects optimal contacts between robot \& object using approximate dynamical constraints, and then a NonLinear Program (NLP) optimizes trajectory of the robot(s) and object considering full nonlinear constraints. We present a convex relaxation of bilinear constraints using binary encoding technique such that MILP can provide tighter solutions with better computational complexity. The proposed framework is evaluated on various manipulation tasks where it can reason about complex multi-contact interactions while providing computational advantages. We also demonstrate our framework in hardware experiments using a bimanual robot system. 

**Abstract (ZH)**: 通过接触进行操作的轨迹设计具有挑战性，因为它需要同时推理物体和机器人轨迹以及复杂的接触序列。本文提出了一种新型框架，用于高效同时设计机器人、物体和接触的轨迹，以应对富含接触的操作需求。我们提出了一种层次优化框架，其中混合整数线性规划（MILP）使用近似动力学约束选择机器人与物体之间的最优接触点，然后使用非线性规划（NLP）在全面考虑非线性约束的情况下优化机器人和物体的轨迹。我们使用二进制编码技术对双线性约束进行凸松弛处理，使得MILP能够提供更紧致的解决方案，且具有更好的计算复杂度。所提出框架已在多种操作任务中进行评估，能够处理复杂的多接触交互，同时提供计算优势。我们还在双臂机器人系统中进行了硬件实验，验证了所提出框架的有效性。 

---
# EFPC: Towards Efficient and Flexible Prompt Compression 

**Title (ZH)**: EFPC: 向高效灵活提示压缩迈进 

**Authors**: Yun-Hao Cao, Yangsong Wang, Shuzheng Hao, Zhenxing Li, Chengjun Zhan, Sichao Liu, Yi-Qi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07956)  

**Abstract**: The emergence of large language models (LLMs) like GPT-4 has revolutionized natural language processing (NLP), enabling diverse, complex tasks. However, extensive token counts lead to high computational and financial burdens. To address this, we propose Efficient and Flexible Prompt Compression (EFPC), a novel method unifying task-aware and task-agnostic compression for a favorable accuracy-efficiency trade-off. EFPC uses GPT-4 to generate compressed prompts and integrates them with original prompts for training. During training and inference, we selectively prepend user instructions and compress prompts based on predicted probabilities. EFPC is highly data-efficient, achieving significant performance with minimal data. Compared to the state-of-the-art method LLMLingua-2, EFPC achieves a 4.8% relative improvement in F1-score with 1% additional data at a 4x compression rate, and an 11.4% gain with 10% additional data on the LongBench single-doc QA benchmark. EFPC's unified framework supports broad applicability and enhances performance across various models, tasks, and domains, offering a practical advancement in NLP. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-4的涌现重塑了自然语言处理（NLP）， enables diverse, complex tasks.然而，广泛的标记数量导致了高计算和财务负担。为了解决这一问题，我们提出了高效灵活的提示压缩（EFPC）方法，这是一种新颖的统合任务感知和任务无关压缩的方法，以实现有利的准确率-效率权衡。EFPC 使用 GPT-4 生成压缩提示，并将其与原始提示结合用于训练。在训练和推理过程中，我们根据预测概率选择性地添加用户指令并压缩提示。EFPC 高度数据高效，即使在少量数据的情况下也能实现显著的性能提升。与最先进的方法 LLMLingua-2 相比，EFPC 在 4 倍压缩率下，利用 1% 的额外数据实现了 4.8% 的相对 F1 分数改进；在 10% 的额外数据下，于 LongBench 单文档 QA 基准测试中实现了 11.4% 的性能增益。EFPC 的统一框架支持广泛的适用性，并在各种模型、任务和领域中提升了性能，提供了一种在 NLP 中的实际进展。 

---
# 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting 

**Title (ZH)**: 7DGS: 统一的空间- temporal-角度高斯点云表示 

**Authors**: Zhongpai Gao, Benjamin Planche, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Ziyan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07946)  

**Abstract**: Real-time rendering of dynamic scenes with view-dependent effects remains a fundamental challenge in computer graphics. While recent advances in Gaussian Splatting have shown promising results separately handling dynamic scenes (4DGS) and view-dependent effects (6DGS), no existing method unifies these capabilities while maintaining real-time performance. We present 7D Gaussian Splatting (7DGS), a unified framework representing scene elements as seven-dimensional Gaussians spanning position (3D), time (1D), and viewing direction (3D). Our key contribution is an efficient conditional slicing mechanism that transforms 7D Gaussians into view- and time-conditioned 3D Gaussians, maintaining compatibility with existing 3D Gaussian Splatting pipelines while enabling joint optimization. Experiments demonstrate that 7DGS outperforms prior methods by up to 7.36 dB in PSNR while achieving real-time rendering (401 FPS) on challenging dynamic scenes with complex view-dependent effects. The project page is: this https URL. 

**Abstract (ZH)**: 实时渲染具视点依赖效果的动态场景依然是计算机图形学中的一个基本挑战。虽然近年来高斯斑点绘图在分别处理动态场景（4DGS）和视点依赖效果（6DGS）方面取得了有前途的结果，但现有方法无法在保持实时性能的同时统一这些能力。我们提出了一种统一框架7D高斯斑点绘图（7DGS），该框架将场景元素表示为跨越位置（3D）、时间和视向（3D）的七维高斯。我们的主要贡献是一种高效的条件切片机制，将7D高斯转化为时间和视点条件下的3D高斯，同时保持与现有3D高斯斑点绘图管道的兼容性并实现联合优化。实验结果显示，7DGS在PSNR上最多可比之前的方法高出7.36 dB，并在具有复杂视点依赖效果的动态场景中实现了实时渲染（401 FPS）。项目页面：this https URL。 

---
# A Theory of Learning with Autoregressive Chain of Thought 

**Title (ZH)**: 自动回归链式思考的学习理论 

**Authors**: Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, Nathan Srebro  

**Link**: [PDF](https://arxiv.org/pdf/2503.07932)  

**Abstract**: For a given base class of sequence-to-next-token generators, we consider learning prompt-to-answer mappings obtained by iterating a fixed, time-invariant generator for multiple steps, thus generating a chain-of-thought, and then taking the final token as the answer. We formalize the learning problems both when the chain-of-thought is observed and when training only on prompt-answer pairs, with the chain-of-thought latent. We analyze the sample and computational complexity both in terms of general properties of the base class (e.g. its VC dimension) and for specific base classes such as linear thresholds. We present a simple base class that allows for universal representability and computationally tractable chain-of-thought learning. Central to our development is that time invariance allows for sample complexity that is independent of the length of the chain-of-thought. Attention arises naturally in our construction. 

**Abstract (ZH)**: 对于给定的序列到下一个token生成基类，我们考虑通过迭代一个固定的时间不变生成器多次来学习生成链式思考模式，并最终通过取最后一个生成的token作为答案的提示到答案映射。我们形式化了在链式思考可见和仅通过提示-答案对进行训练且链式思考作为潜在变量的情况下的学习问题。我们从基类的一般属性（如它的VC维度）和特定基类（如线性阈值）的角度分析了样本复杂性和计算复杂性。我们提出一个简单的基类，允许通用表示和可计算的链式思考学习。我们的发展核心在于时间不变性使得链式思考长度与样本复杂性无关。注意力在我们的构造中自然出现。 

---
# Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia 

**Title (ZH)**: 众包、爬取或生成？创造SEA-VL，一个面向东南亚的多文化视觉语言数据集 

**Authors**: Samuel Cahyawijaya, Holy Lovenia, Joel Ruben Antony Moniz, Tack Hwa Wong, Mohammad Rifqi Farhansyah, Thant Thiri Maung, Frederikus Hudi, David Anugraha, Muhammad Ravi Shulthan Habibi, Muhammad Reza Qorib, Amit Agarwal, Joseph Marvin Imperial, Hitesh Laxmichand Patel, Vicky Feliren, Bahrul Ilmi Nasution, Manuel Antonio Rufino, Genta Indra Winata, Rian Adam Rajagede, Carlos Rafael Catalan, Mohamed Fazli Imam, Priyaranjan Pattnayak, Salsabila Zahirah Pranida, Kevin Pratama, Yeshil Bangera, Adisai Na-Thalang, Patricia Nicole Monderin, Yueqi Song, Christian Simon, Lynnette Hui Xian Ng, Richardy Lobo' Sapan, Taki Hasan Rafi, Bin Wang, Supryadi, Kanyakorn Veerakanjana, Piyalitt Ittichaiwong, Matthew Theodore Roque, Karissa Vincentio, Takdanai Kreangphet, Phakphum Artkaew, Kadek Hendrawan Palgunadi, Yanzhi Yu, Rochana Prih Hastuti, William Nixon, Mithil Bangera, Adrian Xuan Wei Lim, Aye Hninn Khine, Hanif Muhammad Zhafran, Teddy Ferdinan, Audra Aurora Izzani, Ayushman Singh, Evan, Jauza Akbar Krito, Michael Anugraha, Fenal Ashokbhai Ilasariya, Haochen Li, John Amadeo Daniswara, Filbert Aurelian Tjiaranata, Eryawan Presma Yulianrifat, Can Udomcharoenchaikit, Fadil Risdian Ansori, Mahardika Krisna Ihsani, Giang Nguyen, Anab Maulana Barik, Dan John Velasco, Rifo Ahmad Genadi, Saptarshi Saha, Chengwei Wei, Isaiah Flores, Kenneth Ko Han Chen, Anjela Gail Santos, Wan Shen Lim, Kaung Si Phyo, Tim Santos, Meisyarah Dwiastuti, Jiayun Luo, Jan Christian Blaise Cruz, Ming Shan Hee, Ikhlasul Akmal Hanif, M.Alif Al Hakim, Muhammad Rizky Sya'ban, Kun Kerdthaisong, Lester James V. Miranda, Fajri Koto, Tirana Noor Fatyanosa, Alham Fikri Aji, Jostin Jerico Rosal, Jun Kevin, Robert Wijaya, Onno P. Kampman, Ruochen Zhang, Börje F. Karlsson, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2503.07920)  

**Abstract**: Southeast Asia (SEA) is a region of extraordinary linguistic and cultural diversity, yet it remains significantly underrepresented in vision-language (VL) research. This often results in artificial intelligence (AI) models that fail to capture SEA cultural nuances. To fill this gap, we present SEA-VL, an open-source initiative dedicated to developing high-quality, culturally relevant data for SEA languages. By involving contributors from SEA countries, SEA-VL aims to ensure better cultural relevance and diversity, fostering greater inclusivity of underrepresented languages in VL research. Beyond crowdsourcing, our initiative goes one step further in the exploration of the automatic collection of culturally relevant images through crawling and image generation. First, we find that image crawling achieves approximately ~85% cultural relevance while being more cost- and time-efficient than crowdsourcing. Second, despite the substantial progress in generative vision models, synthetic images remain unreliable in accurately reflecting SEA cultures. The generated images often fail to reflect the nuanced traditions and cultural contexts of the region. Collectively, we gather 1.28M SEA culturally-relevant images, more than 50 times larger than other existing datasets. Through SEA-VL, we aim to bridge the representation gap in SEA, fostering the development of more inclusive AI systems that authentically represent diverse cultures across SEA. 

**Abstract (ZH)**: 东南亚（SEA）地区的语言和文化多样性突出，但在视觉语言（VL）研究中却显著欠代表。这常常导致人工智能（AI）模型无法捕捉到SEA的文化细微差别。为填补这一空白，我们提出SEA-VL这一开源项目，旨在为SEA语言开发高质量、文化相关的数据。通过来自SEA国家的贡献者，SEA-VL旨在确保更好的文化相关性和多样性，促进在VL研究中更广泛地包容欠代表的语言。除了众包，我们的倡议在通过爬取和图像生成自动收集文化相关图像方面更进一步。首先，我们发现图像爬取实现了约85%的文化相关性，且在成本和时间效率上优于众包。其次，尽管生成式视觉模型取得了重大进展，合成图像依然难以准确反映SEA文化。生成的图像往往无法体现该地区的细腻传统和文化背景。通过SEA-VL，我们共收集了1,280,000张SEA文化相关图像，规模超过现有其他数据集的50倍。通过SEA-VL，我们旨在填补SEA的代表性缺口，推动开发更包容的AI系统，真实地代表东南亚多元文化。 

---
# Visual and Text Prompt Segmentation: A Novel Multi-Model Framework for Remote Sensing 

**Title (ZH)**: 视觉和文本提示分割：一种用于遥感的新型多模型框架 

**Authors**: Xing Zi, Kairui Jin, Xian Tao, Jun Li, Ali Braytee, Rajiv Ratn Shah, Mukesh Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2503.07911)  

**Abstract**: Pixel-level segmentation is essential in remote sensing, where foundational vision models like CLIP and Segment Anything Model(SAM) have demonstrated significant capabilities in zero-shot segmentation tasks. Despite their advances, challenges specific to remote sensing remain substantial. Firstly, The SAM without clear prompt constraints, often generates redundant masks, and making post-processing more complex. Secondly, the CLIP model, mainly designed for global feature alignment in foundational models, often overlooks local objects crucial to remote sensing. This oversight leads to inaccurate recognition or misplaced focus in multi-target remote sensing imagery. Thirdly, both models have not been pre-trained on multi-scale aerial views, increasing the likelihood of detection failures. To tackle these challenges, we introduce the innovative VTPSeg pipeline, utilizing the strengths of Grounding DINO, CLIP, and SAM for enhanced open-vocabulary image segmentation. The Grounding DINO+(GD+) module generates initial candidate bounding boxes, while the CLIP Filter++(CLIP++) module uses a combination of visual and textual prompts to refine and filter out irrelevant object bounding boxes, ensuring that only pertinent objects are considered. Subsequently, these refined bounding boxes serve as specific prompts for the FastSAM model, which executes precise segmentation. Our VTPSeg is validated by experimental and ablation study results on five popular remote sensing image segmentation datasets. 

**Abstract (ZH)**: 基于像素级分割的遥感应用中，CLIP和Segment Anything Model（SAM）等基础视觉模型在零样本分割任务中展现了显著的能力。尽管取得了进展，但遥感领域的特定挑战仍然很大。首先，SAM在缺乏明确提示约束的情况下，常常生成冗余分割掩码，增加了后续处理的复杂性。其次，CLIP模型主要设计用于基础模型中的全局特征对齐，经常忽视遥感中至关重要的局部目标，导致多目标遥感影像中的误识别或焦点错误。第三，两种模型均未在多尺度航空影像上进行预训练，增加了检测失败的可能性。为应对这些挑战，我们提出了创新的VTPSeg流水线，利用Grounding DINO、CLIP和SAM的优势增强开放词汇图像分割。Grounding DINO+（GD+）模块生成初始候选边框，CLIP Filter++（CLIP++）模块通过结合视觉和文本提示来细化和过滤无关物体边框，确保仅考虑相关物体。随后，这些细化后的边框作为具体提示应用于FastSAM模型，执行精确分割。VTPSeg通过在五个流行的遥感图像分割数据集上的实验和消融研究得到了验证。 

---
# FunGraph: Functionality Aware 3D Scene Graphs for Language-Prompted Scene Interaction 

**Title (ZH)**: FunGraph:  Awareness of 功能的3D场景图 用于语言提示的场景交互 

**Authors**: Dennis Rotondi, Fabio Scaparro, Hermann Blum, Kai O. Arras  

**Link**: [PDF](https://arxiv.org/pdf/2503.07909)  

**Abstract**: The concept of 3D scene graphs is increasingly recognized as a powerful semantic and hierarchical representation of the environment. Current approaches often address this at a coarse, object-level resolution. In contrast, our goal is to develop a representation that enables robots to directly interact with their environment by identifying both the location of functional interactive elements and how these can be used. To achieve this, we focus on detecting and storing objects at a finer resolution, focusing on affordance-relevant parts. The primary challenge lies in the scarcity of data that extends beyond instance-level detection and the inherent difficulty of capturing detailed object features using robotic sensors. We leverage currently available 3D resources to generate 2D data and train a detector, which is then used to augment the standard 3D scene graph generation pipeline. Through our experiments, we demonstrate that our approach achieves functional element segmentation comparable to state-of-the-art 3D models and that our augmentation enables task-driven affordance grounding with higher accuracy than the current solutions. 

**Abstract (ZH)**: 3D 场景图：细粒度表示与功能元素交互的机器人环境表示 

---
# Gemini Embedding: Generalizable Embeddings from Gemini 

**Title (ZH)**: Gemini嵌入：来自Gemini的通用嵌入 

**Authors**: Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Koert Chen, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, Tom Duerig  

**Link**: [PDF](https://arxiv.org/pdf/2503.07891)  

**Abstract**: In this report, we introduce Gemini Embedding, a state-of-the-art embedding model leveraging the power of Gemini, Google's most capable large language model. Capitalizing on Gemini's inherent multilingual and code understanding capabilities, Gemini Embedding produces highly generalizable embeddings for text spanning numerous languages and textual modalities. The representations generated by Gemini Embedding can be precomputed and applied to a variety of downstream tasks including classification, similarity, clustering, ranking, and retrieval. Evaluated on the Massive Multilingual Text Embedding Benchmark (MMTEB), which includes over one hundred tasks across 250+ languages, Gemini Embedding substantially outperforms prior state-of-the-art models, demonstrating considerable improvements in embedding quality. Achieving state-of-the-art performance across MMTEB's multilingual, English, and code benchmarks, our unified model demonstrates strong capabilities across a broad selection of tasks and surpasses specialized domain-specific models. 

**Abstract (ZH)**: Gemini嵌入：基于Google最强大的大型语言模型的先进嵌入模型 

---
# Safety Guardrails for LLM-Enabled Robots 

**Title (ZH)**: LLM驱动机器人安全守rail 

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2503.07885)  

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots. 

**Abstract (ZH)**: RoboGuard：一种确保大语言模型赋能机器人安全的两阶段护栏架构 

---
# LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model 

**Title (ZH)**: LLMIdxAdvis：利用大型语言模型的资源高效索引顾问 

**Authors**: Xinxin Zhao, Haoyang Li, Jing Zhang, Xinmei Huang, Tieying Zhang, Jianjun Chen, Rui Shi, Cuiping Li, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07884)  

**Abstract**: Index recommendation is essential for improving query performance in database management systems (DBMSs) through creating an optimal set of indexes under specific constraints. Traditional methods, such as heuristic and learning-based approaches, are effective but face challenges like lengthy recommendation time, resource-intensive training, and poor generalization across different workloads and database schemas. To address these issues, we propose LLMIdxAdvis, a resource-efficient index advisor that uses large language models (LLMs) without extensive fine-tuning. LLMIdxAdvis frames index recommendation as a sequence-to-sequence task, taking target workload, storage constraint, and corresponding database environment as input, and directly outputting recommended indexes. It constructs a high-quality demonstration pool offline, using GPT-4-Turbo to synthesize diverse SQL queries and applying integrated heuristic methods to collect both default and refined labels. During recommendation, these demonstrations are ranked to inject database expertise via in-context learning. Additionally, LLMIdxAdvis extracts workload features involving specific column statistical information to strengthen LLM's understanding, and introduces a novel inference scaling strategy combining vertical scaling (via ''Index-Guided Major Voting'' and Best-of-N) and horizontal scaling (through iterative ''self-optimization'' with database feedback) to enhance reliability. Experiments on 3 OLAP and 2 real-world benchmarks reveal that LLMIdxAdvis delivers competitive index recommendation with reduced runtime, and generalizes effectively across different workloads and database schemas. 

**Abstract (ZH)**: LLMIdxAdvis：一种基于大型语言模型的高效索引建议方法 

---
# Measuring directional bias amplification in image captions using predictability 

**Title (ZH)**: 使用可预测性衡量图像字幕中的方向偏见放大 

**Authors**: Rahul Nair, Bhanu Tokas, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2503.07878)  

**Abstract**: When we train models on biased ML datasets, they not only learn these biases but can inflate them at test time - a phenomenon called bias amplification. To measure bias amplification in ML datasets, many co-occurrence-based metrics have been proposed. Co-occurrence-based metrics are effective in measuring bias amplification in simple problems like image classification. However, these metrics are ineffective for complex problems like image captioning as they cannot capture the semantics of a caption. To measure bias amplification in captions, prior work introduced a predictability-based metric called Leakage in Captioning (LIC). While LIC captures the semantics and context of captions, it has limitations. LIC cannot identify the direction in which bias is amplified, poorly estimates dataset bias due to a weak vocabulary substitution strategy, and is highly sensitive to attacker models (a hyperparameter in predictability-based metrics). To overcome these issues, we propose Directional Predictability Amplification in Captioning (DPAC). DPAC measures directional bias amplification in captions, provides a better estimate of dataset bias using an improved substitution strategy, and is less sensitive to attacker models. Our experiments on the COCO captioning dataset show how DPAC is the most reliable metric to measure bias amplification in captions. 

**Abstract (ZH)**: 当我们在带有偏见的ML数据集上训练模型时，它们不仅会学习这些偏见，还会在测试时放大这些偏见——这一现象称为偏见放大。为了衡量ML数据集中的偏见放大，已经提出了许多共现基于的指标。共现基于的指标在简单的图像分类等问题中有效，用于衡量偏见放大。然而，对于复杂的图像描述等问题，这些指标无法捕获描述的语义。为了衡量描述中的偏见放大，以往的工作引入了一种可预测性基于的指标，称为描述中的泄漏（LIC）。虽然LIC能够捕获描述的语义和背景，但它存在局限性。LIC无法识别偏见放大的方向，由于弱词汇替换策略导致其对数据集偏见的估计较差，并且非常敏感于攻击者模型（预测性指标中的一个超参数）。为了克服这些问题，我们提出了定向可预测性放大在描述中的方法（DPAC）。DPAC衡量描述中的定向偏见放大，通过改进的替换策略提供对数据集偏见的更好估计，并且对攻击者模型的敏感度更低。我们在COCO描述数据集上的实验表明，DPAC是最可靠的衡量描述中偏见放大的指标。 

---
# Topology-Preserving Loss for Accurate and Anatomically Consistent Cardiac Mesh Reconstruction 

**Title (ZH)**: 拓扑保持损失用于准确且解剖一致的心脏网格重建 

**Authors**: Chenyu Zhang, Yihao Luo, Yinzhe Wu, Choon Hwai Yap, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07874)  

**Abstract**: Accurate cardiac mesh reconstruction from volumetric data is essential for personalized cardiac modeling and clinical analysis. However, existing deformation-based approaches are prone to topological inconsistencies, particularly membrane penetration, which undermines the anatomical plausibility of the reconstructed mesh. To address this issue, we introduce Topology-Preserving Mesh Loss (TPM Loss), a novel loss function that explicitly enforces topological constraints during mesh deformation. By identifying topology-violating points, TPM Loss ensures spatially consistent reconstructions. Extensive experiments on CT and MRI datasets show that TPM Loss reduces topology violations by up to 93.1% while maintaining high segmentation accuracy (DSC: 89.1%-92.9%) and improving mesh fidelity (Chamfer Distance reduction up to 0.26 mm). These results demonstrate that TPM Loss effectively prevents membrane penetration and significantly improves cardiac mesh quality, enabling more accurate and anatomically consistent cardiac reconstructions. 

**Abstract (ZH)**: 从体数据中进行准确的心肌网格重建对于个性化心脏建模和临床分析至关重要。然而，现有的基于变形的方法容易出现拓扑不连续性，特别是膜穿透，这损害了重建网格的解剖合理性。为解决这一问题，我们引入了拓扑保持网格损失（Topology-Preserving Mesh Loss，TPM Loss），这是一种新颖的损失函数，在网格变形过程中显式地施加拓扑约束。通过识别拓扑违例点，TPM Loss 确保了空间上的一致性重建。在CT和MRI数据集上的 extensive 实验表明，TPM Loss 可将拓扑违例降低高达 93.1%，同时保持高分割准确率（DSC: 89.1%-92.9%），并提高网格保真度（切线距离减少至 0.26 mm）。这些结果表明，TPM Loss 有效地防止了膜穿透，显著提高了心脏网格质量，从而实现更准确和解剖上一致的心脏重建。 

---
# MapQA: Open-domain Geospatial Question Answering on Map Data 

**Title (ZH)**: MapQA：基于地图数据的开放域地理空间问答 

**Authors**: Zekun Li, Malcolm Grossman, Eric, Qasemi, Mihir Kulkarni, Muhao Chen, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07871)  

**Abstract**: Geospatial question answering (QA) is a fundamental task in navigation and point of interest (POI) searches. While existing geospatial QA datasets exist, they are limited in both scale and diversity, often relying solely on textual descriptions of geo-entities without considering their geometries. A major challenge in scaling geospatial QA datasets for reasoning lies in the complexity of geospatial relationships, which require integrating spatial structures, topological dependencies, and multi-hop reasoning capabilities that most text-based QA datasets lack. To address these limitations, we introduce MapQA, a novel dataset that not only provides question-answer pairs but also includes the geometries of geo-entities referenced in the questions. MapQA is constructed using SQL query templates to extract question-answer pairs from OpenStreetMap (OSM) for two study regions: Southern California and Illinois. It consists of 3,154 QA pairs spanning nine question types that require geospatial reasoning, such as neighborhood inference and geo-entity type identification. Compared to existing datasets, MapQA expands both the number and diversity of geospatial question types. We explore two approaches to tackle this challenge: (1) a retrieval-based language model that ranks candidate geo-entities by embedding similarity, and (2) a large language model (LLM) that generates SQL queries from natural language questions and geo-entity attributes, which are then executed against an OSM database. Our findings indicate that retrieval-based methods effectively capture concepts like closeness and direction but struggle with questions that require explicit computations (e.g., distance calculations). LLMs (e.g., GPT and Gemini) excel at generating SQL queries for one-hop reasoning but face challenges with multi-hop reasoning, highlighting a key bottleneck in advancing geospatial QA systems. 

**Abstract (ZH)**: 地理空间问答（QA）是导航和兴趣点（POI）搜索中的基本任务。现有地理空间QA数据集存在规模和多样性有限的问题，通常仅依赖于地理实体的文本描述，而忽视了它们的几何形状。在扩展地理空间QA数据集以进行推理时，面临的主要挑战在于地理空间关系的复杂性，这需要结合空间结构、拓扑依赖关系和多跳推理能力，而大多数基于文本的QA数据集缺乏这些功能。为了解决这些限制，我们引入了MapQA这一新型数据集，除了提供问题-答案对之外，还包含了问题中引用的地理实体的几何形状。MapQA利用SQL查询模板从OpenStreetMap（OSM）中提取问题-答案对，涉及两个研究区域：加利福尼亚南部和伊利诺伊州。它包含了3,154个问题-答案对，涉及九种需要地理空间推理的问题类型，如邻里推理和地理实体类型识别。与现有数据集相比，MapQA在地理空间问题的数量和多样性上都有扩展。我们探索了两种应对这一挑战的方法：（1）基于检索的语言模型，通过嵌入相似性对候选地理实体进行排名，以及（2）大型语言模型（LLM），生成从自然语言问题和地理实体属性到SQL查询的转换，然后将这些查询执行在OSM数据库上。我们的研究结果表明，基于检索的方法有效地捕捉了接近性和方向等概念，但对于需要显式计算的问题（如距离计算）则表现不佳。大语言模型（如GPT和Gemini）擅长生成用于单跳推理的SQL查询，但在多跳推理方面面临挑战，这突显了推进地理空间QA系统的一个关键瓶颈。 

---
# Right Reward Right Time for Federated Learning 

**Title (ZH)**: Right Reward at Right Time for Federated Learning 

**Authors**: Thanh Linh Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2503.07869)  

**Abstract**: Critical learning periods (CLPs) in federated learning (FL) refer to early stages during which low-quality contributions (e.g., sparse training data availability) can permanently impair the learning performance of the global model owned by the model owner (i.e., the cloud server). However, strategies to motivate clients with high-quality contributions to join the FL training process and share trained model updates during CLPs remain underexplored. Additionally, existing incentive mechanisms in FL treat all training periods equally, which consequently fails to motivate clients to participate early. Compounding this challenge is the cloud's limited knowledge of client training capabilities due to privacy regulations, leading to information asymmetry. Therefore, in this article, we propose a time-aware incentive mechanism, called Right Reward Right Time (R3T), to encourage client involvement, especially during CLPs, to maximize the utility of the cloud in FL. Specifically, the cloud utility function captures the trade-off between the achieved model performance and payments allocated for clients' contributions, while accounting for clients' time and system capabilities, efforts, joining time, and rewards. Then, we analytically derive the optimal contract for the cloud and devise a CLP-aware mechanism to incentivize early participation and efforts while maximizing cloud utility, even under information asymmetry. By providing the right reward at the right time, our approach can attract the highest-quality contributions during CLPs. Simulation and proof-of-concept studies show that R3T increases cloud utility and is more economically effective than benchmarks. Notably, our proof-of-concept results show up to a 47.6% reduction in the total number of clients and up to a 300% improvement in convergence time while reaching competitive test accuracies compared with incentive mechanism benchmarks. 

**Abstract (ZH)**: 时间敏感的激励机制：Critical Learning Periods (CLPs) 在联邦学习中的 Right Reward Right Time (R3T) 激励策略 

---
# Video Action Differencing 

**Title (ZH)**: 视频动作差异分析 

**Authors**: James Burgess, Xiaohan Wang, Yuhui Zhang, Anita Rau, Alejandro Lozano, Lisa Dunlap, Trevor Darrell, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2503.07860)  

**Abstract**: How do two individuals differ when performing the same action? In this work, we introduce Video Action Differencing (VidDiff), the novel task of identifying subtle differences between videos of the same action, which has many applications, such as coaching and skill learning. To enable development on this new task, we first create VidDiffBench, a benchmark dataset containing 549 video pairs, with human annotations of 4,469 fine-grained action differences and 2,075 localization timestamps indicating where these differences occur. Our experiments demonstrate that VidDiffBench poses a significant challenge for state-of-the-art large multimodal models (LMMs), such as GPT-4o and Qwen2-VL. By analyzing failure cases of LMMs on VidDiffBench, we highlight two key challenges for this task: localizing relevant sub-actions over two videos and fine-grained frame comparison. To overcome these, we propose the VidDiff method, an agentic workflow that breaks the task into three stages: action difference proposal, keyframe localization, and frame differencing, each stage utilizing specialized foundation models. To encourage future research in this new task, we release the benchmark at this https URL and code at this http URL. 

**Abstract (ZH)**: 两人在执行相同动作时有何不同？在本工作中，我们介绍了Video Action Differencing (VidDiff)这一新颖任务，旨在识别相同动作视频之间的细微差异，该任务在教练和技能学习等领域有广泛的应用。为了推进这一新任务的发展，我们首先创建了VidDiffBench基准数据集，包含549对视频，附有人类标注的4,469个细粒度动作差异和2,075个定位时间戳，指示这些差异发生的位置。实验结果表明，VidDiffBench对当前最先进的大型多模态模型（LMMs）如GPT-4o和Qwen2-VL构成了重大挑战。通过对LMMs在VidDiffBench上的失败案例的分析，我们指出了这一任务中的两个关键挑战：在两段视频中定位相关子动作和细粒度帧比较。为了克服这些挑战，我们提出了VidDiff方法，这是一种具备代理性的工作流，将任务分解为三个阶段：动作差异提议、关键帧定位和帧差异比较，每个阶段都利用专门的基础模型。为了促进未来对该新任务的研究，我们在https://链接和http://链接处发布了基准数据集和代码。 

---
# CIMAGE: Exploiting the Conditional Independence in Masked Graph Auto-encoders 

**Title (ZH)**: CIMAGE: 利用masked图自编码器中的条件独立性 

**Authors**: Jongwon Park, Heesoo Jung, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.07852)  

**Abstract**: Recent Self-Supervised Learning (SSL) methods encapsulating relational information via masking in Graph Neural Networks (GNNs) have shown promising performance. However, most existing approaches rely on random masking strategies in either feature or graph space, which may fail to capture task-relevant information fully. We posit that this limitation stems from an inability to achieve minimum redundancy between masked and unmasked components while ensuring maximum relevance of both to potential downstream tasks. Conditional Independence (CI) inherently satisfies the minimum redundancy and maximum relevance criteria, but its application typically requires access to downstream labels. To address this challenge, we introduce CIMAGE, a novel approach that leverages Conditional Independence to guide an effective masking strategy within the latent space. CIMAGE utilizes CI-aware latent factor decomposition to generate two distinct contexts, leveraging high-confidence pseudo-labels derived from unsupervised graph clustering. In this framework, the pretext task involves reconstructing the masked second context solely from the information provided by the first context. Our theoretical analysis further supports the superiority of CIMAGE's novel CI-aware masking method by demonstrating that the learned embedding exhibits approximate linear separability, which enables accurate predictions for the downstream task. Comprehensive evaluations across diverse graph benchmarks illustrate the advantage of CIMAGE, with notably higher average rankings on node classification and link prediction tasks. Notably, our proposed model highlights the under-explored potential of CI in enhancing graph SSL methodologies and offers enriched insights for effective graph representation learning. 

**Abstract (ZH)**: Recent Self-Supervised Learning Methods Leverage Conditional Independence for Effective Masking in Graph Neural Networks 

---
# HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations 

**Title (ZH)**: HalluVerse25：大语言模型幻觉的细颗粒度多语言基准数据集 

**Authors**: Samir Abdaljalil, Hasan Kurban, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07833)  

**Abstract**: Large Language Models (LLMs) are increasingly used in various contexts, yet remain prone to generating non-factual content, commonly referred to as "hallucinations". The literature categorizes hallucinations into several types, including entity-level, relation-level, and sentence-level hallucinations. However, existing hallucination datasets often fail to capture fine-grained hallucinations in multilingual settings. In this work, we introduce HalluVerse25, a multilingual LLM hallucination dataset that categorizes fine-grained hallucinations in English, Arabic, and Turkish. Our dataset construction pipeline uses an LLM to inject hallucinations into factual biographical sentences, followed by a rigorous human annotation process to ensure data quality. We evaluate several LLMs on HalluVerse25, providing valuable insights into how proprietary models perform in detecting LLM-generated hallucinations across different contexts. 

**Abstract (ZH)**: Large Language Models (LLMs)在各种情境中的应用日益增多，但仍容易生成非事实内容，通常称为“幻觉”。已有文献将幻觉分类为实体级、关系级和句子级幻觉。然而，现有的幻觉数据集往往难以在多语言环境中捕捉到细粒度的幻觉。在本工作中，我们介绍了HalluVerse25，这是一个多语言LLM幻觉数据集，它在英语、阿拉伯语和土耳其语中分类了细粒度的幻觉。我们的数据集构建管道使用LLM将幻觉注入事实性传记句子中，然后通过严格的 humano 人工注释过程确保数据质量。我们对HalluVerse25上的几种LLM进行了评估，提供了关于不同情境下商业模型检测LLM生成的幻觉的表现的宝贵见解。 

---
# Group Fairness in Multi-Task Reinforcement Learning 

**Title (ZH)**: 多任务强化学习中的团体公平性 

**Authors**: Kefan Song, Runnan Jiang, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07817)  

**Abstract**: This paper addresses a critical societal consideration in the application of Reinforcement Learning (RL): ensuring equitable outcomes across different demographic groups in multi-task settings. While previous work has explored fairness in single-task RL, many real-world applications are multi-task in nature and require policies to maintain fairness across all tasks. We introduce a novel formulation of multi-task group fairness in RL and propose a constrained optimization algorithm that explicitly enforces fairness constraints across multiple tasks simultaneously. We have shown that our proposed algorithm does not violate fairness constraints with high probability and with sublinear regret in the finite-horizon episodic setting. Through experiments in RiverSwim and MuJoCo environments, we demonstrate that our approach better ensures group fairness across multiple tasks compared to previous methods that lack explicit multi-task fairness constraints in both the finite-horizon setting and the infinite-horizon setting. Our results show that the proposed algorithm achieves smaller fairness gaps while maintaining comparable returns across different demographic groups and tasks, suggesting its potential for addressing fairness concerns in real-world multi-task RL applications. 

**Abstract (ZH)**: 本文探讨了强化学习（RL）应用中的一个关键社会考虑因素：在多任务设置中确保不同人口群体的公平结果。尽管以往的工作已探索了单任务RL中的公平性问题，但许多实际应用都是多任务性质，要求策略在所有任务中均维持公平性。我们提出了强化学习中多任务群体公平性的新形式化定义，并提出了一种约束优化算法，该算法同时显式地在多个任务中施加公平性约束。我们证明了所提出的算法以高概率不会违反公平性约束，并在有限时间段的周期性设置中具有亚线性后悔。通过在RiverSwim和MuJoCo环境中进行实验，我们表明，与以往方法相比，我们的方法在有限时间段和无限时间段设置中都能更好地确保多任务场景下不同群体之间的公平性，其中以往方法缺乏明确的多任务公平性约束。我们的结果表明，所提出的算法在不同人口群体和任务中实现了更小的公平性差距，同时保持了相当的回报，这表明其在实际多任务RL应用中解决公平性问题的潜力。 

---
# AgriField3D: A Curated 3D Point Cloud and Procedural Model Dataset of Field-Grown Maize from a Diversity Panel 

**Title (ZH)**: AgriField3D: 田间种植玉米多样面板的精选3D点云和过程建模数据集 

**Authors**: Elvis Kimara, Mozhgan Hadadi, Jackson Godbersen, Aditya Balu, Talukder Jubery, Yawei Li, Adarsh Krishnamurthy, Patrick S. Schnable, Baskar Ganapathysubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2503.07813)  

**Abstract**: The application of artificial intelligence (AI) in three-dimensional (3D) agricultural research, particularly for maize, has been limited by the scarcity of large-scale, diverse datasets. While 2D image datasets are abundant, they fail to capture essential structural details such as leaf architecture, plant volume, and spatial arrangements that 3D data provide. To address this limitation, we present AgriField3D (this https URL), a curated dataset of 3D point clouds of field-grown maize plants from a diverse genetic panel, designed to be AI-ready for advancing agricultural research. Our dataset comprises over 1,000 high-quality point clouds collected using a Terrestrial Laser Scanner, complemented by procedural models that provide structured, parametric representations of maize plants. These procedural models, generated using Non-Uniform Rational B-Splines (NURBS) and optimized via a two-step process combining Particle Swarm Optimization (PSO) and differentiable programming, enable precise, scalable reconstructions of leaf surfaces and plant architectures. To enhance usability, we performed graph-based segmentation to isolate individual leaves and stalks, ensuring consistent labeling across all samples. We also conducted rigorous manual quality control on all datasets, correcting errors in segmentation, ensuring accurate leaf ordering, and validating metadata annotations. The dataset further includes metadata detailing plant morphology and quality, alongside multi-resolution subsampled versions (100k, 50k, 10k points) optimized for various computational needs. By integrating point cloud data of field grown plants with high-fidelity procedural models and ensuring meticulous manual validation, AgriField3D provides a comprehensive foundation for AI-driven phenotyping, plant structural analysis, and 3D applications in agricultural research. 

**Abstract (ZH)**: 农业领域三维（3D）玉米研究中的人工智能应用受限于大规模多样化数据集的稀缺性。虽然二维（2D）图像数据集丰富，但无法捕捉到三维（3D）数据提供的关键结构细节，如叶序结构、植物体积和空间布局。为解决这一限制，我们 presents AgriField3D（详见<a href="this https URL" target="_blank">此处</a>），这是一个由多样基因组系的田间种植玉米植物构建的三维点云数据集，旨在为推进农业研究做好人工智能准备。该数据集包含超过1000个高质量的点云，采用 terrestrial激光扫描仪采集，并补充了过程模型，提供了结构化的参数化玉米植物表示。这些过程模型使用非均匀有理B样条（NURBS）生成，并通过结合粒子群优化（PSO）和可微编程的两步优化过程进行优化，实现了叶面和植物结构的精确且可扩展的重建。为提升易用性，我们进行了图块基的分割以分离单个叶片和茎杆，确保所有样本的标签一致。我们还对所有数据集进行了严格的手动质量控制，纠正分割错误，确保叶片排序准确，并验证元数据注释。该数据集还包括详细的植物形态和质量元数据，以及用于各种计算需求的多分辨率子采样版本（100k、50k、10k点）。通过将田间种植植物的点云数据与高保真过程模型整合，并确保仔细的手动验证，AgriField3D 为基于人工智能的表型分析、植物结构分析和农业研究中的三维应用提供了全面的基础。 

---
# A primer on optimal transport for causal inference with observational data 

**Title (ZH)**: 最优 transport在观察性数据分析中的因果推理入门 

**Authors**: Florian F Gunsilius  

**Link**: [PDF](https://arxiv.org/pdf/2503.07811)  

**Abstract**: The theory of optimal transportation has developed into a powerful and elegant framework for comparing probability distributions, with wide-ranging applications in all areas of science. The fundamental idea of analyzing probabilities by comparing their underlying state space naturally aligns with the core idea of causal inference, where understanding and quantifying counterfactual states is paramount. Despite this intuitive connection, explicit research at the intersection of optimal transport and causal inference is only beginning to develop. Yet, many foundational models in causal inference have implicitly relied on optimal transport principles for decades, without recognizing the underlying connection. Therefore, the goal of this review is to offer an introduction to the surprisingly deep existing connections between optimal transport and the identification of causal effects with observational data -- where optimal transport is not just a set of potential tools, but actually builds the foundation of model assumptions. As a result, this review is intended to unify the language and notation between different areas of statistics, mathematics, and econometrics, by pointing out these existing connections, and to explore novel problems and directions for future work in both areas derived from this realization. 

**Abstract (ZH)**: 最优传输理论已成为一种强大而优雅的概率分布比较框架，在科学的所有领域都有广泛的应用。尽管最优传输与因果推断之间的直观联系显而易见，但二者交集的研究尚处于起步阶段。事实上，许多因果推断中的基础模型已经隐含地依赖于最优传输原理数十年之久，而未意识到这种潜在联系。因此，本文综述旨在介绍最优传输与基于观察数据识别因果效应之间出人意料的深入联系——在此过程中，最优传输不仅是潜在的工具，更是模型假设的基础。本文综述旨在通过指出这些存在的联系，统一统计学、数学和计量经济学领域的语言和符号，并探索这些认识在两个领域产生的新颖问题和未来研究方向。 

---
# Training Domain Draft Models for Speculative Decoding: Best Practices and Insights 

**Title (ZH)**: 基于推测解码的领域特定草图模型训练：最佳实践与洞察 

**Authors**: Fenglu Hong, Ravi Raju, Jonathan Lingjie Li, Bo Li, Urmish Thakker, Avinash Ravichandran, Swayambhoo Jain, Changran Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07807)  

**Abstract**: Speculative decoding is an effective method for accelerating inference of large language models (LLMs) by employing a small draft model to predict the output of a target model. However, when adapting speculative decoding to domain-specific target models, the acceptance rate of the generic draft model drops significantly due to domain shift. In this work, we systematically investigate knowledge distillation techniques for training domain draft models to improve their speculation accuracy. We compare white-box and black-box distillation approaches and explore their effectiveness in various data accessibility scenarios, including historical user queries, curated domain data, and synthetically generated alignment data. Our experiments across Function Calling, Biology, and Chinese domains show that offline distillation consistently outperforms online distillation by 11% to 25%, white-box distillation surpasses black-box distillation by 2% to 10%, and data scaling trends hold across domains. Additionally, we find that synthetic data can effectively align draft models and achieve 80% to 93% of the performance of training on historical user queries. These findings provide practical guidelines for training domain-specific draft models to improve speculative decoding efficiency. 

**Abstract (ZH)**: 推测解码是通过使用小型草稿模型预测目标模型的输出来加速大型语言模型（LLMs）推理的有效方法。然而，将推测解码适应领域特定的目标模型时，通用草稿模型的接受率会显著下降，这是由于领域转移导致的。在本文中，我们系统地调查了知识蒸馏技术在训练领域特定草稿模型中的应用，以提高其推测准确性。我们比较了白盒和黑盒蒸馏方法，并在包括历史用户查询、精选领域数据和合成对齐数据在内的各种数据可访问性场景中探索其有效性。我们的跨功能调用、生物学和中文领域的实验表明，在线蒸馏比离线蒸馏低11%至25%，白盒蒸馏比黑盒蒸馏高2%至10%，并且数据缩放趋势在不同领域中保持一致。此外，我们发现合成数据可以有效地对齐草稿模型，并且其性能可以达到基于历史用户查询训练的80%至93%。这些发现为提高推测解码效率的训练领域特定草稿模型提供了实用指南。 

---
# Towards Large Language Models that Benefit for All: Benchmarking Group Fairness in Reward Models 

**Title (ZH)**: 面向全体受益的大型语言模型：奖赏模型中的分组公平性基准测试 

**Authors**: Kefan Song, Jin Yao, Runnan Jiang, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07806)  

**Abstract**: As Large Language Models (LLMs) become increasingly powerful and accessible to human users, ensuring fairness across diverse demographic groups, i.e., group fairness, is a critical ethical concern. However, current fairness and bias research in LLMs is limited in two aspects. First, compared to traditional group fairness in machine learning classification, it requires that the non-sensitive attributes, in this case, the prompt questions, be the same across different groups. In many practical scenarios, different groups, however, may prefer different prompt questions and this requirement becomes impractical. Second, it evaluates group fairness only for the LLM's final output without identifying the source of possible bias. Namely, the bias in LLM's output can result from both the pretraining and the finetuning. For finetuning, the bias can result from both the RLHF procedure and the learned reward model. Arguably, evaluating the group fairness of each component in the LLM pipeline could help develop better methods to mitigate the possible bias. Recognizing those two limitations, this work benchmarks the group fairness of learned reward models. By using expert-written text from arXiv, we are able to benchmark the group fairness of reward models without requiring the same prompt questions across different demographic groups. Surprisingly, our results demonstrate that all the evaluated reward models (e.g., Nemotron-4-340B-Reward, ArmoRM-Llama3-8B-v0.1, and GRM-llama3-8B-sftreg) exhibit statistically significant group unfairness. We also observed that top-performing reward models (w.r.t. canonical performance metrics) tend to demonstrate better group fairness. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）变得越来越强大并越来越多地被人类用户使用，确保跨不同人口群体的公平性，即群体公平性，是一个关键的伦理问题。然而，当前对LLMs的公平性和偏见研究存在两个方面的问题。首先，与传统机器学习分类中的群体公平性相比，它要求在这种情况下非敏感属性（即提示问题）在不同群体之间保持一致。然而，在许多实际场景中，不同群体可能会偏好不同的提示问题，从而使这一要求变得不现实。其次，它仅针对LLMs的最终输出评估群体公平性，而未识别引起潜在偏见的来源。也就是说，LLMs输出中的偏见可能来自预训练和微调。可以认为，评估LLM流水线中每个组件的群体公平性有助于开发更好的方法来缓解潜在的偏见。基于认识到这两个局限性，本研究对所学习的奖励模型的群体公平性进行了基准测试。通过使用arXiv上专家撰写的文本，我们在不需要不同人口群体之间使用相同的提示问题的情况下，能够对奖励模型的群体公平性进行基准测试。令人惊讶的是，我们的结果表明，所有评估的奖励模型（例如Nemotron-4-340B-Reward、ArmoRM-Llama3-8B-v0.1和GRM-llama3-8B-sftreg）都表现出统计上显著的群体不公平性。我们还观察到，按照传统性能指标表现最佳的奖励模型倾向于表现出更好的群体公平性。 

---
# Self-supervised Normality Learning and Divergence Vector-guided Model Merging for Zero-shot Congenital Heart Disease Detection in Fetal Ultrasound Videos 

**Title (ZH)**: 基于自监督正常性学习和发散向量引导模型融合的胎儿超声视频先天性心脏病零样本检测 

**Authors**: Pramit Saha, Divyanshu Mishra, Netzahualcoyotl Hernandez-Cruz, Olga Patey, Aris Papageorghiou, Yuki M. Asano, J. Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2503.07799)  

**Abstract**: Congenital Heart Disease (CHD) is one of the leading causes of fetal mortality, yet the scarcity of labeled CHD data and strict privacy regulations surrounding fetal ultrasound (US) imaging present significant challenges for the development of deep learning-based models for CHD detection. Centralised collection of large real-world datasets for rare conditions, such as CHD, from large populations requires significant co-ordination and resource. In addition, data governance rules increasingly prevent data sharing between sites. To address these challenges, we introduce, for the first time, a novel privacy-preserving, zero-shot CHD detection framework that formulates CHD detection as a normality modeling problem integrated with model merging. In our framework dubbed Sparse Tube Ultrasound Distillation (STUD), each hospital site first trains a sparse video tube-based self-supervised video anomaly detection (VAD) model on normal fetal heart US clips with self-distillation loss. This enables site-specific models to independently learn the distribution of healthy cases. To aggregate knowledge across the decentralized models while maintaining privacy, we propose a Divergence Vector-Guided Model Merging approach, DivMerge, that combines site-specific models into a single VAD model without data exchange. Our approach preserves domain-agnostic rich spatio-temporal representations, ensuring generalization to unseen CHD cases. We evaluated our approach on real-world fetal US data collected from 5 hospital sites. Our merged model outperformed site-specific models by 23.77% and 30.13% in accuracy and F1-score respectively on external test sets. 

**Abstract (ZH)**: 先天性心脏病检测的一种新颖隐私保护零样本框架：Sparse Tube Ultrasound Distillation（STUD） 

---
# Joint Explainability-Performance Optimization With Surrogate Models for AI-Driven Edge Services 

**Title (ZH)**: 基于代理模型的AI驱动边缘服务联合解释性-性能优化 

**Authors**: Foivos Charalampakos, Thomas Tsouparopoulos, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.07784)  

**Abstract**: Explainable AI is a crucial component for edge services, as it ensures reliable decision making based on complex AI models. Surrogate models are a prominent approach of XAI where human-interpretable models, such as a linear regression model, are trained to approximate a complex (black-box) model's predictions. This paper delves into the balance between the predictive accuracy of complex AI models and their approximation by surrogate ones, advocating that both these models benefit from being learned simultaneously. We derive a joint (bi-level) training scheme for both models and we introduce a new algorithm based on multi-objective optimization (MOO) to simultaneously minimize both the complex model's prediction error and the error between its outputs and those of the surrogate. Our approach leads to improvements that exceed 99% in the approximation of the black-box model through the surrogate one, as measured by the metric of Fidelity, for a compromise of less than 3% absolute reduction in the black-box model's predictive accuracy, compared to single-task and multi-task learning baselines. By improving Fidelity, we can derive more trustworthy explanations of the complex model's outcomes from the surrogate, enabling reliable AI applications for intelligent services at the network edge. 

**Abstract (ZH)**: 可解释AI是边缘服务的关键组件，因为它确保基于复杂AI模型的可靠决策。代理模型是可解释AI的一个主要方法，其中人类可解释的模型，如线性回归模型，被训练以近似复杂（黑盒）模型的预测。本文探讨了复杂AI模型的预测准确性和其由代理模型近似的平衡，提倡同时学习这两种模型。我们推导了一种联合（多层次）训练方案，并引入了一种基于多目标优化的新算法，同时最小化复杂模型的预测误差及其输出与代理模型输出之间的误差。通过Fidelity度量，我们的方法在代理模型中实现了接近100%的黑盒模型近似改善，同时相对减少不到3%的黑盒模型预测准确率，优于单任务和多任务学习基准。通过提高Fidelity，可以从代理模型中获得更可信的复杂模型结果解释，从而为网络边缘的智能服务提供可靠的AI应用。 

---
# Evaluating LLaMA 3.2 for Software Vulnerability Detection 

**Title (ZH)**: 评估LLaMA 3.2在软件漏洞检测中的性能 

**Authors**: José Gonçalves, Miguel Silva, Bernardo Cabral, Tiago Dias, Eva Maia, Isabel Praça, Ricardo Severino, Luís Lino Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2503.07770)  

**Abstract**: Deep Learning (DL) has emerged as a powerful tool for vulnerability detection, often outperforming traditional solutions. However, developing effective DL models requires large amounts of real-world data, which can be difficult to obtain in sufficient quantities. To address this challenge, DiverseVul dataset has been curated as the largest dataset of vulnerable and non-vulnerable C/C++ functions extracted exclusively from real-world projects. Its goal is to provide high-quality, large-scale samples for training DL models. However, during our study several inconsistencies were identified in the raw dataset while applying pre-processing techniques, highlighting the need for a refined version. In this work, we present a refined version of DiverseVul dataset, which is used to fine-tune a large language model, LLaMA 3.2, for vulnerability detection. Experimental results show that the use of pre-processing techniques led to an improvement in performance, with the model achieving an F1-Score of 66%, a competitive result when compared to our baseline, which achieved a 47% F1-Score in software vulnerability detection. 

**Abstract (ZH)**: DiverseVul数据集精修版及其在LLaMA 3.2模型上用于软件漏洞检测的研究 

---
# A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing 

**Title (ZH)**: 具有约束意识的模仿学习的简单方法及其在自主赛车中的应用 

**Authors**: Shengfan Cao, Eunhyek Joa, Francesco Borrelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.07737)  

**Abstract**: Guaranteeing constraint satisfaction is challenging in imitation learning (IL), particularly in tasks that require operating near a system's handling limits. Traditional IL methods often struggle to enforce constraints, leading to suboptimal performance in high-precision tasks. In this paper, we present a simple approach to incorporating safety into the IL objective. Through simulations, we empirically validate our approach on an autonomous racing task with both full-state and image feedback, demonstrating improved constraint satisfaction and greater consistency in task performance compared to a baseline method. 

**Abstract (ZH)**: 确保约束满足在模仿学习中具有挑战性，特别是在系统操作极限附近执行任务时。传统的模仿学习方法往往难以强制执行约束，导致在高精度任务中表现出色。在本文中，我们提出了一种将安全性纳入模仿学习目标的简单方法。通过仿真，我们在配备全状态和图像反馈的自主赛车任务中 empirically 验证了该方法的有效性，结果显示约束满足程度提高且任务性能更为一致，相比基准方法具有显著优势。 

---
# Automated Benchmark Generation for Repository-Level Coding Tasks 

**Title (ZH)**: 仓库级编码任务的自动化基准生成 

**Authors**: Konstantinos Vergopoulos, Mark Niklas Müller, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.07701)  

**Abstract**: Code Agent development is an extremely active research area, where a reliable performance metric is critical for tracking progress and guiding new developments. This demand is underscored by the meteoric rise in popularity of SWE-Bench. This benchmark challenges code agents to generate patches addressing GitHub issues given the full repository as context. The correctness of generated patches is then evaluated by executing a human-written test suite extracted from the repository after the issue's resolution. However, constructing benchmarks like SWE-Bench requires substantial manual effort to set up historically accurate execution environments for testing. Crucially, this severely limits the number of considered repositories, e.g., just 12 for SWE-Bench. Considering so few repositories, selected for their popularity runs the risk of leading to a distributional mismatch, i.e., the measured performance may not be representative of real-world scenarios potentially misguiding development efforts. In this work, we address this challenge and introduce SetUpAgent, a fully automated system capable of historically accurate dependency setup, test execution, and result parsing. Using SetUpAgent, we generate two new datasets: (i) SWEE-Bench an extended version of SWE-Bench encompassing hundreds of repositories, and (ii) SWA-Bench a benchmark focusing on applications rather than libraries. Comparing these datasets to SWE-Bench with respect to their characteristics and code agent performance, we find significant distributional differences, including lower issue description quality and detail level, higher fix complexity, and most importantly up to 40% lower agent success rates. 

**Abstract (ZH)**: Code Agent 开发是一个极其活跃的研究领域，可靠的性能度量对于追踪进展和指导新技术的发展至关重要。这一需求在 SWE-Bench 的迅猛流行中得到了强化。SWE-Bench 挑战代码代理生成解决问题的补丁，并通过执行提取自仓库的人工编写的测试套件来评估生成补丁的正确性。然而，构建类似 SWE-Bench 的基准需要大量的手动工作来设置历史准确的测试环境。这极大地限制了可以考虑的仓库数量，例如，SWE-Bench 只包括 12 个仓库。在如此有限的仓库中进行选择，可能会导致数据分布不匹配，即测量的性能可能不具有现实世界场景的代表性，从而误导开发努力。在此工作中，我们解决了这一挑战并引入了 SetUpAgent，这是一个能够自动完成历史准确的依赖设置、测试执行和结果解析的系统。使用 SetUpAgent，我们生成了两个新的数据集：(i) SWEE-Bench，它是 SWE-Bench 的扩展版本，涵盖了数百个仓库，(ii) SWA-Bench，一个专注于应用程序而非库的基准。比较这些数据集与 SWE-Bench 在特性和代码代理性能方面的差异，我们发现显著的数据分布差异，包括较低的问题描述质量和细节水平、更高的修复复杂度，最重要的是，代理成功率最多降低了 40%。 

---
# A Task and Motion Planning Framework Using Iteratively Deepened AND/OR Graph Networks 

**Title (ZH)**: 使用迭代加深的AND/OR图网络的任务与运动规划框架 

**Authors**: Hossein Karami, Antony Thomas, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2503.07700)  

**Abstract**: In this paper, we present an approach for integrated task and motion planning based on an AND/OR graph network, which is used to represent task-level states and actions, and we leverage it to implement different classes of task and motion planning problems (TAMP). Several problems that fall under task and motion planning do not have a predetermined number of sub-tasks to achieve a goal. For example, while retrieving a target object from a cluttered workspace, in principle the number of object re-arrangements required to finally grasp it cannot be known ahead of time. To address this challenge, and in contrast to traditional planners, also those based on AND/OR graphs, we grow the AND/OR graph at run-time by progressively adding sub-graphs until grasping the target object becomes feasible, which yields a network of AND/OR graphs. The approach is extended to enable multi-robot task and motion planning, and (i) it allows us to perform task allocation while coordinating the activity of a given number of robots, and (ii) can handle multi-robot tasks involving an a priori unknown number of sub-tasks. The approach is evaluated and validated both in simulation and with a real dual-arm robot manipulator, that is, Baxter from Rethink Robotics. In particular, for the single-robot task and motion planning, we validated our approach in three different TAMP domains. Furthermore, we also use three different robots for simulation, namely, Baxter, Franka Emika Panda manipulators, and a PR2 robot. Experiments show that our approach can be readily scaled to scenarios with many objects and robots, and is capable of handling different classes of TAMP problems. 

**Abstract (ZH)**: 基于AND/OR图网络的任务与运动集成规划方法及其在多机器人系统中的扩展与验证 

---
# Artificial Intelligence in Deliberation: The AI Penalty and the Emergence of a New Deliberative Divide 

**Title (ZH)**: 人工智能在审议中的应用：AI罚分与审议分裂的新兴分歧 

**Authors**: Andreas Jungherr, Adrian Rauchfleisch  

**Link**: [PDF](https://arxiv.org/pdf/2503.07690)  

**Abstract**: Digital deliberation has expanded democratic participation, yet challenges remain. This includes processing information at scale, moderating discussions, fact-checking, or attracting people to participate. Recent advances in artificial intelligence (AI) offer potential solutions, but public perceptions of AI's role in deliberation remain underexplored. Beyond efficiency, democratic deliberation is about voice and recognition. If AI is integrated into deliberation, public trust, acceptance, and willingness to participate may be affected. We conducted a preregistered survey experiment with a representative sample in Germany (n=1850) to examine how information about AI-enabled deliberation influences willingness to participate and perceptions of deliberative quality. Respondents were randomly assigned to treatments that provided them information about deliberative tasks facilitated by either AI or humans. Our findings reveal a significant AI-penalty. Participants were less willing to engage in AI-facilitated deliberation and rated its quality lower than human-led formats. These effects were moderated by individual predispositions. Perceptions of AI's societal benefits and anthropomorphization of AI showed positive interaction effects on people's interest to participate in AI-enabled deliberative formats and positive quality assessments, while AI risk assessments showed negative interactions with information about AI-enabled deliberation. These results suggest AI-enabled deliberation faces substantial public skepticism, potentially even introducing a new deliberative divide. Unlike traditional participation gaps based on education or demographics, this divide is shaped by attitudes toward AI. As democratic engagement increasingly moves online, ensuring AI's role in deliberation does not discourage participation or deepen inequalities will be a key challenge for future research and policy. 

**Abstract (ZH)**: 数字辩论扩展了民主参与，但仍面临挑战。这包括大规模处理信息、 moderating 讨论、事实核查或吸引人们参与。近年来人工智能（AI）的进步提供了潜在的解决方案，但公众对 AI 在辩论中的角色看法仍需进一步探索。超出效率，数字辩论关乎声音和认同。若 AI 融入辩论，公众信任、接受度和参与意愿可能受到影响。我们在德国进行了一项预先注册的调查实验（n=1850），以考察关于 AI 助力的辩论信息如何影响参与意愿和对辩论质量的看法。受访者被随机分配到使用 AI 或人类助力的辩论任务信息治疗组。我们的研究发现显示了显着的 AI 折扣。参与者更不愿意参与由 AI 助力的辩论，并且认为其质量低于由人类主导的格式。这些效应受到个人倾向的调节。AI 社会利益的认知和将 AI 人性化对个人参与 AI 助力的辩论格式的兴趣和正面质量评估显示出正向交互效应，而对 AI 风险的评估与关于 AI 助力的辩论信息显示出负向交互效应。这些结果表明，AI 助力的辩论面临着显著的公众怀疑，甚至可能引入新的辩论鸿沟。不同于基于教育或人口统计学的传统参与鸿沟，这种鸿沟由对 AI 的态度塑造。随着民主参与越来越多地转移到线上，确保 AI 在辩论中的作用不会阻碍参与或加深不平等将成为未来研究和政策的关键挑战。 

---
# Adaptive routing protocols for determining optimal paths in AI multi-agent systems: a priority- and learning-enhanced approach 

**Title (ZH)**: 基于优先级和学习增强的自适应路由协议：在AI多代理系统中确定最优路径的方法 

**Authors**: Theodor Panayotov, Ivo Emanuilov  

**Link**: [PDF](https://arxiv.org/pdf/2503.07686)  

**Abstract**: As distributed artificial intelligence (AI) and multi-agent architectures grow increasingly complex, the need for adaptive, context-aware routing becomes paramount. This paper introduces an enhanced, adaptive routing algorithm tailored for AI multi-agent networks, integrating priority-based cost functions and dynamic learning mechanisms. Building on an extended Dijkstra-based framework, we incorporate multi-faceted parameters such as task complexity, user request priority, agent capabilities, bandwidth, latency, load, model sophistication, and reliability. We further propose dynamically adaptive weighting factors, tuned via reinforcement learning (RL), to continuously evolve routing policies based on observed network performance. Additionally, heuristic filtering and hierarchical routing structures improve scalability and responsiveness. Our approach yields context-sensitive, load-aware, and priority-focused routing decisions that not only reduce latency for critical tasks but also optimize overall resource utilization, ultimately enhancing the robustness, flexibility, and efficiency of multi-agent systems. 

**Abstract (ZH)**: 随着分布式人工智能（AI）和多智能体架构日益复杂，适应性和上下文感知路由的需求变得至关重要。本文介绍了一种专为AI多智能体网络设计的增强型自适应路由算法，集成基于优先级的成本函数和动态学习机制。在扩展的Dijkstra框架基础上，我们整合了任务复杂度、用户请求优先级、智能体能力、带宽、延迟、负载、模型复杂度和可靠性等多方面参数。此外，我们提出了一种通过强化学习（RL）动态调整的权重因子，以根据观测到的网络性能连续进化路由策略。同时，采用启发式过滤和分层路由结构提高可扩展性和响应性。该方法提供了上下文敏感、负载感知和优先级导向的路由决策，不仅减少了关键任务的延迟，还优化了整体资源利用，最终增强了多智能体系统的 robustness、灵活性和效率。 

---
# Ways of Seeing, and Selling, AI Art 

**Title (ZH)**: 观看方式，与销售方式：AI艺术 

**Authors**: Imke van Heerden  

**Link**: [PDF](https://arxiv.org/pdf/2503.07685)  

**Abstract**: In early 2025, Augmented Intelligence - Christie's first AI art auction - drew criticism for showcasing a controversial genre. Amid wider legal uncertainty, artists voiced concerns over data mining practices, notably with respect to copyright. The backlash could be viewed as a microcosm of AI's contested position in the creative economy. Touching on the auction's presentation, reception, and results, this paper explores how, among social dissonance, machine learning finds its place in the artworld. Foregrounding responsible innovation, the paper provides a balanced perspective that champions creators' rights and brings nuance to this polarised debate. With a focus on exhibition design, it centres framing, which refers to the way a piece is presented to influence consumer perception. Context plays a central role in shaping our understanding of how good, valuable, and even ethical an artwork is. In this regard, Augmented Intelligence situates AI art within a surprisingly traditional framework, leveraging hallmarks of "high art" to establish the genre's cultural credibility. Generative AI has a clear economic dimension, converging questions of artistic merit with those of monetary worth. Scholarship on ways of seeing, or framing, could substantively inform the interpretation and evaluation of creative outputs, including assessments of their aesthetic and commercial value. 

**Abstract (ZH)**: 增强智能艺术——克里斯蒂首次AI艺术拍卖在争议中登场 

---
# A Time Series Multitask Framework Integrating a Large Language Model, Pre-Trained Time Series Model, and Knowledge Graph 

**Title (ZH)**: 一个集成大规模语言模型、预训练时间序列模型和知识图谱的时间序列多任务框架 

**Authors**: Shule Hao, Junpeng Bao, Chuncheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07682)  

**Abstract**: Time series analysis is crucial in fields like finance, transportation, and industry. However, traditional models often focus solely on temporal features, limiting their ability to capture underlying information. This paper proposes a novel time series multitask framework, called LTM, which integrates temporal features with textual descriptions to enhance analytical and predictive capabilities. LTM combines pre-trained time series model, large language model (LLM), and knowledge graph to tackle time series tasks, including forecasting, imputation, and anomaly detection. LTM achieves improved performance with a few trainable parameters. It is very efficient and practical. LTM encodes time series data into patches and enriches user-provided prompts using knowledge graphs to generate enhanced prompts. A novel feature fusion method embeds prompts into each patch encoding, which is processed by a frozen LLM, followed by a feature enhancement module and a time decoder module. During fine-tuning stage, cosine similarity between prompts and temporal patches is integrated into the loss function to boost performance. Experiments on benchmark datasets show that LTM significantly outperforms existing methods. It provides a robust and versatile solution for time series tasks. 

**Abstract (ZH)**: LTM：一种集成时间和文本的多任务时间序列框架 

---
# Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM 

**Title (ZH)**: 层次均衡打包：面向长上下文LLM的高效监督微调 

**Authors**: Yongqiang Yao, Jingru Tan, Kaihuan Liang, Feizhao Zhang, Yazhe Niu, Jiahao Hu, Ruihao Gong, Dahua Lin, Ningyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07680)  

**Abstract**: Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MOE model, our method speeds up the training by 2.4$\times$ with competitive performance. 

**Abstract (ZH)**: 培训长上下文大型语言模型（LLMs）具有挑战性，因为长上下文和短上下文数据的混合训练往往导致工作负载不平衡。现有工作主要使用数据打包来缓解这一问题，但未能考虑不均衡的注意力计算和浪费的通信开销。本文提出了层次平衡打包（HBP），设计了一种新的批次构建方法和训练策略以解决这些不效率。具体而言，HBP 构建多级数据打包组，每组优化不同的打包长度。它将训练样本分配到最合适的组，并为每个组配置最有效的设置，包括顺序并行度和梯度检查点配置。为了有效利用多级数据组，我们设计了一个专门针对HBP 的动态训练管道，包括渐进式学习、自适应顺序并行度和稳定损失。我们的广泛实验表明，我们的方法在多个数据集和开源模型上显著减少了训练时间，同时保持了强大的性能。对于最大的DeepSeek-V2（236B）混合专家模型，我们的方法将训练速度提高了2.4倍，同时保持了竞争力的性能。 

---
# Using a single actor to output personalized policy for different intersections 

**Title (ZH)**: 使用单个actor输出适用于不同交叉口的个性化策略 

**Authors**: Kailing Zhou, Chengwei Zhang, Furui Zhan, Wanting Liu, Yihong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07678)  

**Abstract**: Recently, with the development of Multi-agent reinforcement learning (MARL), adaptive traffic signal control (ATSC) has achieved satisfactory results. In traffic scenarios with multiple intersections, MARL treats each intersection as an agent and optimizes traffic signal control strategies through learning and real-time decision-making. Considering that observation distributions of intersections might be different in real-world scenarios, shared parameter methods might lack diversity and thus lead to high generalization requirements in the shared-policy network. A typical solution is to increase the size of network parameters. However, simply increasing the scale of the network does not necessarily improve policy generalization, which is validated in our experiments. Accordingly, an approach that considers both the personalization of intersections and the efficiency of parameter sharing is required. To this end, we propose Hyper-Action Multi-Head Proximal Policy Optimization (HAMH-PPO), a Centralized Training with Decentralized Execution (CTDE) MARL method that utilizes a shared PPO policy network to deliver personalized policies for intersections with non-iid observation distributions. The centralized critic in HAMH-PPO uses graph attention units to calculate the graph representations of all intersections and outputs a set of value estimates with multiple output heads for each intersection. The decentralized execution actor takes the local observation history as input and output distributions of action as well as a so-called hyper-action to balance the multiple values estimated from the centralized critic to further guide the updating of TSC policies. The combination of hyper-action and multi-head values enables multiple agents to share a single actor-critic while achieving personalized policies. 

**Abstract (ZH)**: 最近，随着多代理 reinforcement 学习（MARL）的发展，自适应交通信号控制（ATSC）取得了满意的结果。在包含多个交叉口的交通场景中，MARL 将每个交叉口视为一个代理，并通过学习和实时决策优化交通信号控制策略。考虑到实际场景中交叉口的观测分布可能不同，共享参数方法可能会缺乏多样性，从而导致在共享策略网络中对泛化能力提出了高要求。一个典型的解决方案是增加网络参数的规模。然而，仅仅增加网络规模并不一定能提高策略的泛化能力，我们的实验中得到了验证。因此，同时考虑交叉口的个性化和参数共享效率的方法是必要的。为此，我们提出了超动作多头近端策略优化（HAMH-PPO），这是一种集中训练分散执行（CTDE）的 MARL 方法，利用共享的 PPO 策略网络为具有非 i.i.d 观测分布的交叉口提供个性化的策略。HAMH-PPO 的集中式评论家使用图注意单元来计算所有交叉口的图表示，并输出每个交叉口的一组价值估计值，带有多个输出头。分散执行的行动者将局部观察历史作为输入，并输出行动分布以及所谓的超动作，以平衡集中式评论家估计的多个价值，进一步指导 TSC 策略的更新。超动作和多头价值的结合使多个代理能够共享一个单一的行动者-评论家网络，同时实现个性化的策略。 

---
# PLADIS: Pushing the Limits of Attention in Diffusion Models at Inference Time by Leveraging Sparsity 

**Title (ZH)**: PLADIS: 在推断时通过利用稀疏性推动扩散模型注意机制的极限 

**Authors**: Kwanyoung Kim, Byeongsu Sim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07677)  

**Abstract**: Diffusion models have shown impressive results in generating high-quality conditional samples using guidance techniques such as Classifier-Free Guidance (CFG). However, existing methods often require additional training or neural function evaluations (NFEs), making them incompatible with guidance-distilled models. Also, they rely on heuristic approaches that need identifying target layers. In this work, we propose a novel and efficient method, termed PLADIS, which boosts pre-trained models (U-Net/Transformer) by leveraging sparse attention. Specifically, we extrapolate query-key correlations using softmax and its sparse counterpart in the cross-attention layer during inference, without requiring extra training or NFEs. By leveraging the noise robustness of sparse attention, our PLADIS unleashes the latent potential of text-to-image diffusion models, enabling them to excel in areas where they once struggled with newfound effectiveness. It integrates seamlessly with guidance techniques, including guidance-distilled models. Extensive experiments show notable improvements in text alignment and human preference, offering a highly efficient and universally applicable solution. 

**Abstract (ZH)**: PLADIS: 通过利用稀疏注意机制增强预训练模型以提升文本到图像扩散模型的效果 

---
# The Janus Face of Innovation: Global Disparities and Divergent Options 

**Title (ZH)**: 创新的两面性：全球不平等与分歧选择 

**Authors**: Nihat Mugurtay  

**Link**: [PDF](https://arxiv.org/pdf/2503.07676)  

**Abstract**: This article examines how unequal access to AI innovation creates systemic challenges for developing countries. Differential access to AI innovation results from the acute competition between domestic and global actors. While developing nations contribute significantly to AI development through data annotation labor, they face limited access to advanced AI technologies and are increasingly caught between divergent regulatory approaches from democratic and authoritarian tendencies. This brief paper analyzes how more affordable AI engagement and Western countries' development cooperation present developing nations with a complex choice between accessibility and governance standards. I argue this challenge entails new institutional mechanisms for technology transfer and regulatory cooperation, while carefully balancing universal standards with local needs. In turn, good practices could help developing countries close the deepening gap of global technological divides, while ensuring responsible AI development in developing countries. 

**Abstract (ZH)**: 本文探讨了不平等的AI创新访问权如何为发展中国家创造系统性挑战。差异化的AI创新访问是由国内和全球行为体之间的尖锐竞争造成的。尽管发展中国家通过数据标注劳动力在AI发展中做出了重大贡献，但它们仍面临先进AI技术访问受限的问题，并且越来越被民主和威权倾向的监管方法所分歧。本文简要分析了更实惠的AI参与和西方国家的发展合作为发展中国家提供了在可访问性和治理标准之间复杂选择的背景，我认为这一挑战需要新的技术转移和监管合作制度机制，在普及标准与地方需求之间谨慎平衡。良好的实践可以有助于发展中国家缩小日益扩大的全球技术鸿沟，同时确保发展中国家负责任的AI发展。 

---
# DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems 

**Title (ZH)**: DynTaskMAS：一种基于异步并行LLM驱动的动态任务图框架的多-agent系统 

**Authors**: Junwei Yu, Yepeng Ding, Hiroyuki Sato  

**Link**: [PDF](https://arxiv.org/pdf/2503.07675)  

**Abstract**: The emergence of Large Language Models (LLMs) in Multi-Agent Systems (MAS) has opened new possibilities for artificial intelligence, yet current implementations face significant challenges in resource management, task coordination, and system efficiency. While existing frameworks demonstrate the potential of LLM-based agents in collaborative problem-solving, they often lack sophisticated mechanisms for parallel execution and dynamic task management. This paper introduces DynTaskMAS, a novel framework that orchestrates asynchronous and parallel operations in LLM-based MAS through dynamic task graphs. The framework features four key innovations: (1) a Dynamic Task Graph Generator that intelligently decomposes complex tasks while maintaining logical dependencies, (2) an Asynchronous Parallel Execution Engine that optimizes resource utilization through efficient task scheduling, (3) a Semantic-Aware Context Management System that enables efficient information sharing among agents, and (4) an Adaptive Workflow Manager that dynamically optimizes system performance. Experimental evaluations demonstrate that DynTaskMAS achieves significant improvements over traditional approaches: a 21-33% reduction in execution time across task complexities (with higher gains for more complex tasks), a 35.4% improvement in resource utilization (from 65% to 88%), and near-linear throughput scaling up to 16 concurrent agents (3.47X improvement for 4X agents). Our framework establishes a foundation for building scalable, high-performance LLM-based multi-agent systems capable of handling complex, dynamic tasks efficiently. 

**Abstract (ZH)**: 大型语言模型在多agent系统中的新兴应用为人工智能打开了新的可能性，然而当前实现面临资源管理、任务协调和系统效率的重大挑战。尽管现有的框架展示了基于大型语言模型的代理在协作问题解决方面的潜力，但它们通常缺乏复杂的并行执行和动态任务管理机制。本文介绍了DynTaskMAS，这是一种新颖的框架，利用动态任务图在基于大型语言模型的多agent系统中协调异步和并行操作。该框架包含四项关键创新：(1) 动态任务图生成器，能够智能地分解复杂任务并保持逻辑依赖关系；(2) 异步并行执行引擎，通过高效的任务调度优化资源利用；(3) 语义感知上下文管理系统，使代理之间能够高效地共享信息；(4) 自适应工作流管理器，动态优化系统性能。实验评估表明，DynTaskMAS在任务复杂性方面取得了显著改进：执行时间减少21-33%（复杂任务的增幅更大），资源利用率提高35.4%（从65%提高到88%），并且在16个并发代理的范围内实现接近线性的吞吐量扩展（4倍代理数量时性能提升3.47倍）。我们的框架为构建可扩展、高性能的基于大型语言模型的多agent系统奠定了基础，这些系统能够高效地处理复杂、动态的任务。 

---
# TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation 

**Title (ZH)**: TVNet：一种基于动态卷积和3D变异的时间序列分析方法 

**Authors**: Chenghan Li, Mingchen Li, Ruisheng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07674)  

**Abstract**: With the recent development and advancement of Transformer and MLP architectures, significant strides have been made in time series analysis. Conversely, the performance of Convolutional Neural Networks (CNNs) in time series analysis has fallen short of expectations, diminishing their potential for future applications. Our research aims to enhance the representational capacity of Convolutional Neural Networks (CNNs) in time series analysis by introducing novel perspectives and design innovations. To be specific, We introduce a novel time series reshaping technique that considers the inter-patch, intra-patch, and cross-variable dimensions. Consequently, we propose TVNet, a dynamic convolutional network leveraging a 3D perspective to employ time series analysis. TVNet retains the computational efficiency of CNNs and achieves state-of-the-art results in five key time series analysis tasks, offering a superior balance of efficiency and performance over the state-of-the-art Transformer-based and MLP-based models. Additionally, our findings suggest that TVNet exhibits enhanced transferability and robustness. Therefore, it provides a new perspective for applying CNN in advanced time series analysis tasks. 

**Abstract (ZH)**: 基于新颖视角和设计创新提高Convolutional Neural Networks在时间序列分析中的表示能力：提出TVNet及其卓越性能和鲁棒性分析 

---
# The potential role of AI agents in transforming nuclear medicine research and cancer management in India 

**Title (ZH)**: AI代理在转变印度核医学研究和癌症管理中的潜在作用 

**Authors**: Rajat Vashistha, Arif Gulzar, Parveen Kundu, Punit Sharma, Mark Brunstein, Viktor Vegh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07673)  

**Abstract**: India faces a significant cancer burden, with an incidence-to-mortality ratio indicating that nearly three out of five individuals diagnosed with cancer succumb to the disease. While the limitations of physical healthcare infrastructure are widely acknowledged as a primary challenge, concerted efforts by government and healthcare agencies are underway to mitigate these constraints. However, given the country's vast geography and high population density, it is imperative to explore alternative soft infrastructure solutions to complement existing frameworks. Artificial Intelligence agents are increasingly transforming problem-solving approaches across various domains, with their application in medicine proving particularly transformative. In this perspective, we examine the potential role of AI agents in advancing nuclear medicine for cancer research, diagnosis, and management in India. We begin with a brief overview of AI agents and their capabilities, followed by a proposed agent-based ecosystem that can address prevailing sustainability challenges in India nuclear medicine. 

**Abstract (ZH)**: 印度面临严重的癌症负担，癌癥发病率与死亡率比值表明，近三分之二被诊断出癌症的个体最终死于该病。尽管物理 healthcare 基础设施的局限性被广泛承认是主要挑战之一，政府和医疗健康机构正着手解决这些限制。然而，鉴于印度广阔的土地面积和高人口密度，迫切需要探索替代的软基础设施解决方案以补充现有框架。人工智能代理正在各个领域逐渐改变问题解决方式，其在医学中的应用尤其具有变革性。在此视角下，我们探讨人工智能代理在促进印度核医学在癌症研究、诊断和管理中的潜在作用。我们首先简要概述人工智能代理及其能力，然后提出一个基于代理的生态系统，旨在解决印度核医学中现有的可持续性挑战。 

---
# Probabilistic Shielding for Safe Reinforcement Learning 

**Title (ZH)**: 概率性屏蔽以实现安全强化学习 

**Authors**: Edwin Hamel-De le Court, Francesco Belardinelli, Alex W. Goodall  

**Link**: [PDF](https://arxiv.org/pdf/2503.07671)  

**Abstract**: In real-life scenarios, a Reinforcement Learning (RL) agent aiming to maximise their reward, must often also behave in a safe manner, including at training time. Thus, much attention in recent years has been given to Safe RL, where an agent aims to learn an optimal policy among all policies that satisfy a given safety constraint. However, strict safety guarantees are often provided through approaches based on linear programming, and thus have limited scaling. In this paper we present a new, scalable method, which enjoys strict formal guarantees for Safe RL, in the case where the safety dynamics of the Markov Decision Process (MDP) are known, and safety is defined as an undiscounted probabilistic avoidance property. Our approach is based on state-augmentation of the MDP, and on the design of a shield that restricts the actions available to the agent. We show that our approach provides a strict formal safety guarantee that the agent stays safe at training and test time. Furthermore, we demonstrate that our approach is viable in practice through experimental evaluation. 

**Abstract (ZH)**: 在现实场景中，一个旨在最大化奖励的强化学习（RL）代理，通常也需要在训练和测试过程中表现出安全的行为。因此，近年来对安全强化学习（Safe RL）的关注不断增加，其中代理的目标是在所有满足给定安全约束的策略中学习最优策略。然而，严格的安全保证通常通过基于线性规划的方法提供，这在扩展性方面有限。本文提出了一种新的可扩展方法，该方法在马尔可夫决策过程（MDP）的安全动力学已知且安全定义为无折现概率避免属性的情况下，能够为安全强化学习提供严格的正式保证。我们的方法基于MDP的状态拓展，并设计了一个屏蔽机制，限制代理可采取的动作。我们展示了该方法能够在训练和测试时严格保证代理的安全行为。此外，通过实验评估证明了该方法在实践中的可行性。 

---
# Data Foundations for Large Scale Multimodal Clinical Foundation Models 

**Title (ZH)**: 大型多模态临床基础模型的数据基础 

**Authors**: Wei Dai, Peilin Chen, Malinda Lu, Daniel Li, Haowen Wei, Hejie Cui, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07667)  

**Abstract**: Recent advances in clinical AI have enabled remarkable progress across many clinical domains. However, existing benchmarks and models are primarily limited to a small set of modalities and tasks, which hinders the development of large-scale multimodal methods that can make holistic assessments of patient health and well-being. To bridge this gap, we introduce Clinical Large-Scale Integrative Multimodal Benchmark (CLIMB), a comprehensive clinical benchmark unifying diverse clinical data across imaging, language, temporal, and graph modalities. CLIMB comprises 4.51 million patient samples totaling 19.01 terabytes distributed across 2D imaging, 3D video, time series, graphs, and multimodal data. Through extensive empirical evaluation, we demonstrate that multitask pretraining significantly improves performance on understudied domains, achieving up to 29% improvement in ultrasound and 23% in ECG analysis over single-task learning. Pretraining on CLIMB also effectively improves models' generalization capability to new tasks, and strong unimodal encoder performance translates well to multimodal performance when paired with task-appropriate fusion strategies. Our findings provide a foundation for new architecture designs and pretraining strategies to advance clinical AI research. Code is released at this https URL. 

**Abstract (ZH)**: 近期临床AI的发展在许多临床领域取得了显著进展。然而，现有的基准和模型主要局限于少量的模态和任务，这阻碍了能够全面评估患者健康和福祉的大规模多模态方法的发展。为了弥合这一差距，我们介绍了Clinical Large-Scale Integrative Multimodal Benchmark (CLIMB)，这是一个综合性的临床基准，统一了来自影像、语言、时间序列和图等多种模态的多样临床数据。CLIMB 包含总计 4.51 百万患者的样本，数据量达到 19.01 太字节，数据分布在 2D 影像、3D 视频、时间序列、图形和多模态数据中。通过广泛的实验评估，我们证明了多任务预训练显著提高了对未研究领域的性能，分别在超声和心电信号分析中提高了多达 29% 和 23%。在 CLIMB 上进行预训练还有效提高了模型对新任务的泛化能力，而且在适当融合策略的配合下，单一模态编码器的强性能也能很好地转换为多模态性能。我们的研究结果为新的架构设计和预训练策略提供了基础，以推进临床AI研究。代码发布在该网址：<https://>。 

---
# Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for Multimodal LLMs 

**Title (ZH)**: 合并再对齐：简单有效的模态增量连续学习方法用于多模态LLM 

**Authors**: Dingkun Zhang, Shuhan Qi, Xinyu Xiao, Kehai Chen, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07663)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enhanced their versatility as they integrate a growing number of modalities. Considering the heavy cost of training MLLMs, it is necessary to reuse the existing ones and further extend them to more modalities through Modality-incremental Continual Learning (MCL). However, this often comes with a performance degradation in the previously learned modalities. In this work, we revisit the MCL and investigate a more severe issue it faces in contrast to traditional continual learning, that its degradation comes not only from catastrophic forgetting but also from the misalignment between the modality-agnostic and modality-specific components. To address this problem, we propose an elegantly simple MCL paradigm called "MErge then ReAlign" (MERA). Our method avoids introducing heavy training overhead or modifying the model architecture, hence is easy to deploy and highly reusable in the MLLM community. Extensive experiments demonstrate that, despite the simplicity of MERA, it shows impressive performance, holding up to a 99.84% Backward Relative Gain when extending to four modalities, achieving a nearly lossless MCL performance. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models through Modality-Incremental Continual Learning: The M Erge then Re Align Paradigm 

---
# Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy 

**Title (ZH)**: 打破模型合并：无需牺牲精度的参数级防御 

**Authors**: Wei Junhao, Yu Zhe, Sakuma Jun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07661)  

**Abstract**: Model merging is a technique that combines multiple finetuned models into a single model without additional training, allowing a free-rider to cheaply inherit specialized capabilities. This study investigates methodologies to suppress unwanted model merging by free-riders. Existing methods such as model watermarking or fingerprinting can only detect merging in hindsight. In contrast, we propose a first proactive defense against model merging. Specifically, our defense method modifies the model parameters so that the model is disrupted if the model is merged with any other model, while its functionality is kept unchanged if not merged with others. Our approach consists of two modules, rearranging MLP parameters and scaling attention heads, which push the model out of the shared basin in parameter space, causing the merging performance with other models to degrade significantly. We conduct extensive experiments on image classification, image generation, and text classification to demonstrate that our defense severely disrupts merging while retaining the functionality of the post-protect model. Moreover, we analyze potential adaptive attacks and further propose a dropout-based pruning to improve our proposal's robustness. 

**Abstract (ZH)**: 一种主动防御方法以抑制免费 Riding者进行模型合并：一种修改模型参数的方法使得模型在与其他模型合并时性能显著下降，而在未合并时功能保持不变。 

---
# SplitQuantV2: Enhancing Low-Bit Quantization of LLMs Without GPUs 

**Title (ZH)**: SplitQuantV2: 不使用GPU提升大模型低比特量化性能 

**Authors**: Jaewoo Song, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07657)  

**Abstract**: The quantization of large language models (LLMs) is crucial for deploying them on devices with limited computational resources. While advanced quantization algorithms offer improved performance compared to the basic linear quantization, they typically require high-end graphics processing units (GPUs), are often restricted to specific deep neural network (DNN) frameworks, and require calibration datasets. This limitation poses challenges for using such algorithms on various neural processing units (NPUs) and edge AI devices, which have diverse model formats and frameworks. In this paper, we show SplitQuantV2, an innovative algorithm designed to enhance low-bit linear quantization of LLMs, can achieve results comparable to those of advanced algorithms. SplitQuantV2 preprocesses models by splitting linear and convolution layers into functionally equivalent, quantization-friendly structures. The algorithm's platform-agnostic, concise, and efficient nature allows for implementation without the need for GPUs. Our evaluation on the Llama 3.2 1B Instruct model using the AI2's Reasoning Challenge (ARC) dataset demonstrates that SplitQuantV2 improves the accuracy of the INT4 quantization model by 11.76%p, matching the performance of the original floating-point model. Remarkably, SplitQuantV2 took only 2 minutes 6 seconds to preprocess the 1B model and perform linear INT4 quantization using only an Apple M4 CPU. SplitQuantV2 provides a practical solution for low-bit quantization on LLMs, especially when complex, computation-intensive algorithms are inaccessible due to hardware limitations or framework incompatibilities. 

**Abstract (ZH)**: 大语言模型的量化对于在有限算力资源的设备上部署它们至关重要。虽然高级量化算法相较于基本线性量化提供了更好的性能，但它们通常需要高性能图形处理单元（GPU），经常局限于特定的深层神经网络（DNN）框架，并需要校准数据集。这种限制使得在各种神经处理单元（NPU）和边缘AI设备上使用这些算法变得具有挑战性，这些设备具有多样化的模型格式和框架。本文展示了SplitQuantV2这一创新算法，旨在增强大语言模型的低比特线性量化，其结果可与高级算法相媲美。SplitQuantV2通过将线性和卷积层拆分为功能等效且易于量化的结构来预处理模型。该算法的平台无关性、简洁性和高效性使得在无需GPU的情况下进行实现成为可能。使用AI2的推理挑战（ARC）数据集对Llama 3.2 1B Instruct模型的评估表明，SplitQuantV2将INT4量化模型的准确率提高了11.76个百分点，达到了原始浮点模型的性能。令人惊讶的是，SplitQuantV2仅用Apple M4 CPU就花费了2分6秒时间完成了1B模型的预处理和线性INT4量化。SplitQuantV2为大语言模型的低比特量化提供了一种实用的解决方案，特别适用于因硬件限制或框架不兼容而无法使用复杂、计算密集型算法的情形。 

---
# GraphT5: Unified Molecular Graph-Language Modeling via Multi-Modal Cross-Token Attention 

**Title (ZH)**: GraphT5：通过多模态跨token注意力实现统一的分子图形-语言建模 

**Authors**: Sangyeup Kim, Nayeon Kim, Yinhua Piao, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07655)  

**Abstract**: Molecular language modeling tasks such as molecule captioning have been recognized for their potential to further understand molecular properties that can aid drug discovery or material synthesis based on chemical reactions. Unlike the common use of molecule graphs in predicting molecular properties, most methods in molecular language modeling rely heavily on SMILES sequences. This preference is because the task involves generating a sequence of multiple tokens using transformer-based models. Therefore, a main challenge is determining how to integrate graph data, which contains structural and spatial information about molecules, with text data. In addition, simply using both 1D SMILES text and 2D graph as inputs without addressing how they align and represent the molecule structure in different modalities makes it challenging to fully utilize structural knowledge about molecules. To this end, we propose GraphT5, a multi-modal framework that integrates 1D SMILES text and 2D graph representations of molecules for molecular language modeling. Specifically, we introduce a novel cross-token attention module in GraphT5 to bridge the gap arising from the fundamental differences between the two modalities of molecule representations. Cross-token attention exploits implicit information between SMILES and graphs of molecules, resulting from their interactions at a fine-grained token level that benefits molecular language modeling. Extensive experiments including molecule captioning, IUPAC name prediction tasks, and case studies show that our GraphT5 outperforms the latest baseline approaches, which validates the effectiveness of our GraphT5 in sufficiently utilizing 1D SMILES text and 2D graph representations. 

**Abstract (ZH)**: 分子语言建模任务如分子captioning已被认识到其潜在价值，有助于通过化学反应进一步理解分子性质，从而促进药物发现或材料合成。与常用基于分子图预测分子性质的方法不同，大多数分子语言建模方法主要依赖于SMILES序列。这种偏好是因为任务涉及使用基于变压器模型生成多个标记序列。因此，一个主要挑战是如何将包含分子结构和空间信息的图数据与文本数据整合。此外，仅仅使用1D SMILES文本和2D图作为输入而不解决它们在不同模态下如何对齐和表示分子结构的问题，使得充分发挥分子结构知识变得具有挑战性。为此，我们提出GraphT5，这是一种多模态框架，将1D SMILES文本和2D分子图表示结合用于分子语言建模。具体而言，我们在GraphT5中引入了一种新的跨标记注意力模块，以弥补两种分子表示模态之间根本差异产生的差距。跨标记注意力从分子SMILES和图之间在精细标记级别上的相互作用中挖掘隐含信息，这些信息对分子语言建模有益。广泛的实验，包括分子captioning、IUPAC名称预测任务和案例研究，表明我们的GraphT5优于最新基线方法，验证了GraphT5在充分利用1D SMILES文本和2D图表示方面的有效性。 

---
# Insights into Schizophrenia: Leveraging Machine Learning for Early Identification via EEG, ERP, and Demographic Attributes 

**Title (ZH)**: 基于EEG、ERP和人口统计学特征的机器学习在精神分裂症早期识别中的洞察 

**Authors**: Sara Alkhalifa  

**Link**: [PDF](https://arxiv.org/pdf/2503.07650)  

**Abstract**: The research presents a machine learning (ML) classifier designed to differentiate between schizophrenia patients and healthy controls by utilising features extracted from electroencephalogram (EEG) data, specifically focusing on event-related potentials (ERPs) and certain demographic variables. The dataset comprises data from 81 participants, encompassing 32 healthy controls and 49 schizophrenia patients, all sourced from an online dataset. After preprocessing the dataset, our ML model achieved an accuracy of 99.980%. This performance outperforms earlier research, including those that used deep learning methods. Additionally, an analysis was conducted to assess individual features' contribution to improving classification accuracy. This involved systematically excluding specific features from the original dataset one at a time, and another technique involved an iterative process of removing features based on their entropy scores incrementally. The impact of these removals on model performance was evaluated to identify the most informative features. 

**Abstract (ZH)**: 机器学习分类器用于通过 Electroencephalogram (EEG) 数据中的事件相关电位 (ERPs) 和某些人口统计学变量来区分精神分裂症患者与健康对照组 

---
# TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster 

**Title (ZH)**: TS-RAG: 基于检索增强生成的时间序列基础模型在零样本预测中更加强大 

**Authors**: Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Y. Zhang, Kashif Rasul, Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, Dongjin Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.07649)  

**Abstract**: Recently, Large Language Models (LLMs) and Foundation Models (FMs) have become prevalent for time series forecasting tasks. However, fine-tuning large language models (LLMs) for forecasting enables the adaptation to specific domains but may not generalize well across diverse, unseen datasets. Meanwhile, existing time series foundation models (TSFMs) lack inherent mechanisms for domain adaptation and suffer from limited interpretability, making them suboptimal for zero-shot forecasting. To this end, we present TS-RAG, a retrieval-augmented generation based time series forecasting framework that enhances the generalization capability and interpretability of TSFMs. Specifically, TS-RAG leverages pre-trained time series encoders to retrieve semantically relevant time series segments from a dedicated knowledge database, incorporating contextual patterns for the given time series query. Next, we develop a learnable Mixture-of-Experts (MoE)-based augmentation module, which dynamically fuses retrieved time series patterns with the TSFM's representation of the input query, improving forecasting accuracy without requiring task-specific fine-tuning. Thorough empirical studies on seven public benchmark datasets demonstrate that TS-RAG achieves state-of-the-art zero-shot forecasting performance, outperforming TSFMs by up to 6.51% across diverse domains and showcasing desired interpretability. 

**Abstract (ZH)**: TS-RAG：基于检索增强生成的时间序列 forecasting 框架 

---
# ConstellationNet: Reinventing Spatial Clustering through GNNs 

**Title (ZH)**: 星座网：通过GNN重构空间聚类 

**Authors**: Aidan Gao, Junhong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07643)  

**Abstract**: Spatial clustering is a crucial field, finding universal use across criminology, pathology, and urban planning. However, most spatial clustering algorithms cannot pull information from nearby nodes and suffer performance drops when dealing with higher dimensionality and large datasets, making them suboptimal for large-scale and high-dimensional clustering. Due to modern data growing in size and dimension, clustering algorithms become weaker when addressing multifaceted issues. To improve upon this, we develop ConstellationNet, a convolution neural network(CNN)-graph neural network(GNN) framework that leverages the embedding power of a CNN, the neighbor aggregation of a GNN, and a neural network's ability to deal with batched data to improve spatial clustering and classification with graph augmented predictions. ConstellationNet achieves state-of-the-art performance on both supervised classification and unsupervised clustering across several datasets, outperforming state-of-the-art classification and clustering while reducing model size and training time by up to tenfold and improving baselines by 10 times. Because of its fast training and powerful nature, ConstellationNet holds promise in fields like epidemiology and medical imaging, able to quickly train on new data to develop robust responses. 

**Abstract (ZH)**: 空间聚类是一种关键领域，广泛应用于犯罪学、病理学和城市规划。然而，大多数空间聚类算法无法从附近节点提取信息，在处理高维度和大规模数据集时性能下降，使其在大规模和高维聚类中效果欠佳。由于现代数据的规模和维度日益增长，聚类算法在处理多方面问题时变得无力。为改善这一现状，我们开发了星座网络（ConstellationNet），这是一种卷积神经网络（CNN）-图神经网络（GNN）框架，利用CNN的嵌入能力、GNN的邻域聚合能力和神经网络处理批量数据的能力，以图增强预测提高空间聚类和分类性能。星座网络在多个数据集上的监督分类和无监督聚类上均达到最佳性能，优于最先进的分类和聚类模型，同时模型大小和训练时间最多可降低十倍，并将基准提高了十倍。由于其快速训练和强大的特性，星座网络在流行病学和医学影像等领域展现出乐观前景，能够快速适应新数据以开发稳健的响应。 

---
# Deep ARTMAP: Generalized Hierarchical Learning with Adaptive Resonance Theory 

**Title (ZH)**: 深度ARTMAP：自适应共振理论下的generalized分层学习 

**Authors**: Niklas M. Melton, Leonardo Enzo Brito da Silva, Sasha Petrenko, Donald. C. Wunsch II  

**Link**: [PDF](https://arxiv.org/pdf/2503.07641)  

**Abstract**: This paper presents Deep ARTMAP, a novel extension of the ARTMAP architecture that generalizes the self-consistent modular ART (SMART) architecture to enable hierarchical learning (supervised and unsupervised) across arbitrary transformations of data. The Deep ARTMAP framework operates as a divisive clustering mechanism, supporting an arbitrary number of modules with customizable granularity within each module. Inter-ART modules regulate the clustering at each layer, permitting unsupervised learning while enforcing a one-to-many mapping from clusters in one layer to the next. While Deep ARTMAP reduces to both ARTMAP and SMART in particular configurations, it offers significantly enhanced flexibility, accommodating a broader range of data transformations and learning modalities. 

**Abstract (ZH)**: Deep ARTMAP：一种将ARTMAP体系结构扩展到支持任意数据变换的分层监督与无监督学习的新型框架 

---
# BrainNet-MoE: Brain-Inspired Mixture-of-Experts Learning for Neurological Disease Identification 

**Title (ZH)**: 脑神经启发的混合专家学习：脑网-MoE方法在神经系统疾病识别中的应用 

**Authors**: Jing Zhang, Xiaowei Yu, Tong Chen, Chao Cao, Mingheng Chen, Yan Zhuang, Yanjun Lyu, Lu Zhang, Li Su, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07640)  

**Abstract**: The Lewy body dementia (LBD) is the second most common neurodegenerative dementia after Alzheimer's disease (AD). Early differentiation between AD and LBD is crucial because they require different treatment approaches, but this is challenging due to significant clinical overlap, heterogeneity, complex pathogenesis, and the rarity of LBD. While recent advances in artificial intelligence (AI) demonstrate powerful learning capabilities and offer new hope for accurate diagnosis, existing methods primarily focus on designing "neural-level networks". Our work represents a pioneering effort in modeling system-level artificial neural network called BrainNet-MoE for brain modeling and diagnosing. Inspired by the brain's hierarchical organization of bottom-up sensory integration and top-down control, we design a set of disease-specific expert groups to process brain sub-network under different condition, A disease gate mechanism guides the specializa-tion of expert groups, while a transformer layer enables communication be-tween all sub-networks, generating a comprehensive whole-brain represen-tation for downstream disease classification. Experimental results show superior classification accuracy with interpretable insights into how brain sub-networks contribute to different neurodegenerative conditions. 

**Abstract (ZH)**: Lewy 体痴呆（LBD）是仅次于阿尔茨海默病（AD）的第二大常见的神经退行性痴呆。早期准确区分AD和LBD至关重要，因为两者需要不同的治疗方法，但这一区分面临挑战，因它们在临床表现、异质性、复杂的病理机制以及LBD的罕见性上有显著重叠。尽管最近的人工智能（AI）技术展示了强大的学习能力，并为准确诊断带来了新的希望，但现有方法主要集中在设计“神经层级网络”方面。我们的一项工作代表了在构建用于脑建模和诊断的系统级人工神经网络方面的一项开创性努力，称为BrainNet-MoE。受大脑自底向上感觉整合和自顶向下控制的层次结构组织启发，我们设计了一系列疾病特异性的专家组，以在不同条件下处理脑子网络。疾病门控机制引导专家组的专业化，而变压器层使所有子网络之间能够通信，生成用于下游疾病分类的全面的大脑整体表示。实验结果显示，在可解释方面具有优越的分类准确性，并揭示了不同神经退行性疾病条件下脑子网络的贡献机制。 

---
# Leveraging Taxonomy Similarity for Next Activity Prediction in Patient Treatment 

**Title (ZH)**: 利用分类相似性进行患者治疗中下一个活动预测 

**Authors**: Martin Kuhn, Joscha Grüger, Tobias Geyer, Ralph Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.07638)  

**Abstract**: The rapid progress in modern medicine presents physicians with complex challenges when planning patient treatment. Techniques from the field of Predictive Business Process Monitoring, like Next-activity-prediction (NAP) can be used as a promising technique to support physicians in treatment planning, by proposing a possible next treatment step. Existing patient data, often in the form of electronic health records, can be analyzed to recommend the next suitable step in the treatment process. However, the use of patient data poses many challenges due to its knowledge-intensive character, high variability and scarcity of medical data. To overcome these challenges, this article examines the use of the knowledge encoded in taxonomies to improve and explain the prediction of the next activity in the treatment process. This study proposes the TS4NAP approach, which uses medical taxonomies (ICD-10-CM and ICD-10-PCS) in combination with graph matching to assess the similarities of medical codes to predict the next treatment step. The effectiveness of the proposed approach will be evaluated using event logs that are derived from the MIMIC-IV dataset. The results highlight the potential of using domain-specific knowledge held in taxonomies to improve the prediction of the next activity, and thus can improve treatment planning and decision-making by making the predictions more explainable. 

**Abstract (ZH)**: 现代医学快速进展给医生在规划患者治疗时带来了复杂挑战。通过利用预测业务过程监控领域的技术，如下一步活动预测(NAP)，可以为医生在治疗规划中提供支持，通过建议可能的下一步治疗措施。现有的患者数据，通常以电子健康记录的形式存在，可以通过分析来推荐治疗过程中的下一个合适步骤。然而，由于其知识密集性、高变异性以及医学数据的稀缺性，使用患者数据存在许多挑战。为了克服这些挑战，本文研究了利用分类法中编码的知识来提高和解释治疗过程下一步活动预测的问题。本文提出了TS4NAP方法，该方法结合图匹配使用医学分类法（ICD-10-CM和ICD-10-PCS）来评估医学代码的相似性以预测下一步治疗措施。通过MIMIC-IV数据集派生的事件日志评估所提出方法的有效性。结果突显了利用分类法中持有的专业知识改善下一步活动预测的潜力，并且可以提高治疗规划和决策的透明度。 

---
# An Optimization Algorithm for Multimodal Data Alignment 

**Title (ZH)**: 多模态数据对齐的优化算法 

**Authors**: Wei Zhang, Xinyue Wang, Lan Yu, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07636)  

**Abstract**: In the data era, the integration of multiple data types, known as multimodality, has become a key area of interest in the research community. This interest is driven by the goal to develop cutting edge multimodal models capable of serving as adaptable reasoning engines across a wide range of modalities and domains. Despite the fervent development efforts, the challenge of optimally representing different forms of data within a single unified latent space a crucial step for enabling effective multimodal reasoning has not been fully addressed. To bridge this gap, we introduce AlignXpert, an optimization algorithm inspired by Kernel CCA crafted to maximize the similarities between N modalities while imposing some other constraints. This work demonstrates the impact on improving data representation for a variety of reasoning tasks, such as retrieval and classification, underlining the pivotal importance of data representation. 

**Abstract (ZH)**: 数据时代，多模态数据类型的整合已成为研究社区的一个关键研究领域。通过对不同形式的数据在单一统一潜在空间中的最优表示进行优化，以促进有效的多模态推理，尽管进行了广泛的努力，这一挑战尚未得到充分解决。为了弥合这一差距，我们引入了AlignXpert，这是一种受核CCA启发的优化算法，旨在最大化N种模态之间的相似性并施加其他约束。本研究展示了提高数据表示以进行多种推理任务（如检索和分类）的效果，突显了数据表示的重要性。 

---
# Psychological Counseling Ability of Large Language Models 

**Title (ZH)**: 大型语言模型的心理咨询能力 

**Authors**: Fangyu Peng, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.07627)  

**Abstract**: With the development of science and the continuous progress of artificial intelligence technology, Large Language Models (LLMs) have begun to be widely utilized across various fields. However, in the field of psychological counseling, the ability of LLMs have not been systematically assessed. In this study, we assessed the psychological counseling ability of mainstream LLMs using 1096 psychological counseling skill questions which were selected from the Chinese National Counselor Level 3 Examination, including Knowledge-based, Analytical-based, and Application-based question types. The analysis showed that the correctness rates of the LLMs for Chinese questions, in descending order, were GLM-3 (46.5%), GPT-4 (46.1%), Gemini (45.0%), ERNIE-3.5 (45.7%) and GPT-3.5 (32.9%). The correctness rates of the LLMs for English questions, in descending order, were ERNIE-3.5 (43.9%), GPT-4 (40.6%), Gemini (36.6%), GLM-3 (29.9%) and GPT-3.5 (29.5%). A chi-square test indicated significant differences in the LLMs' performance on Chinese and English questions. Furthermore, we subsequently utilized the Counselor's Guidebook (Level 3) as a reference for ERNIE-3.5, resulting in a new correctness rate of 59.6%, a 13.8% improvement over its initial rate of 45.8%. In conclusion, the study assessed the psychological counseling ability of LLMs for the first time, which may provide insights for future enhancement and improvement of psychological counseling ability of LLMs. 

**Abstract (ZH)**: 随着科学技术的发展和人工智能技术的不断进步，大型语言模型（LLMs）已在多个领域得到广泛应用。然而，在心理咨询领域，LLMs的能力尚未进行系统评估。本研究使用了来自中国国家级心理咨询师三级考试的1096个心理咨询技能问题，包括基于知识、基于分析和基于应用三种类型的问题，评估了主流LLMs的心理咨询能力。分析结果显示，按照正确率从高到低的顺序，中文问题的LLMs正确率分别是GLM-3（46.5%）、GPT-4（46.1%）、Gemini（45.0%）、ERNIE-3.5（45.7%）和GPT-3.5（32.9%）。英文问题的LLMs正确率，则是ERNIE-3.5（43.9%）、GPT-4（40.6%）、Gemini（36.6%）、GLM-3（29.9%）和GPT-3.5（29.5%）。卡方检验显示，LLMs在中文和英文问题上的表现存在显著差异。此外，我们进一步将《心理咨询师手册（三级）》作为参考，对ERNIE-3.5进行调整，其新的正确率为59.6%，相较于最初水平提高了13.8%。综上所述，本研究首次评估了LLMs的心理咨询能力，为未来提升和改善LLMs的心理咨询能力提供了参考。 

---
# Junior Software Developers' Perspectives on Adopting LLMs for Software Engineering: a Systematic Literature Review 

**Title (ZH)**: 初级软件开发人员对采用大语言模型进行软件工程的看法：一项系统文献综述 

**Authors**: Samuel Ferino, Rashina Hoda, John Grundy, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2503.07556)  

**Abstract**: Many studies exploring the adoption of Large Language Model-based tools for software development by junior developers have emerged in recent years. These studies have sought to understand developers' perspectives about using those tools, a fundamental pillar for successfully adopting LLM-based tools in Software Engineering. The aim of this paper is to provide an overview of junior software developers' perspectives and use of LLM-based tools for software engineering (LLM4SE). We conducted a systematic literature review (SLR) following guidelines by Kitchenham et al. on 56 primary studies, applying the definition for junior software developers as software developers with equal or less than five years of experience, including Computer Science/Software Engineering students. We found that the majority of the studies focused on comprehending the different aspects of integrating AI tools in SE. Only 8.9\% of the studies provide a clear definition for junior software developers, and there is no uniformity. Searching for relevant information is the most common task using LLM tools. ChatGPT was the most common LLM tool present in the studies (and experiments). A majority of the studies (83.9\%) report both positive and negative perceptions about the impact of adopting LLM tools. We also found and categorised advantages, challenges, and recommendations regarding LLM adoption. Our results indicate that developers are using LLMs not just for code generation, but also to improve their development skills. Critically, they are not just experiencing the benefits of adopting LLM tools, but they are also aware of at least a few LLM limitations, such as the generation of wrong suggestions, potential data leaking, and AI hallucination. Our findings offer implications for software engineering researchers, educators, and developers. 

**Abstract (ZH)**: 近年来，有许多研究探讨初级开发者采用基于大规模语言模型的工具进行软件开发的情况。这些研究旨在理解开发人员对使用这些工具的看法，这是成功采用基于大型语言模型（LLM）工具在软件工程中的一个基本支柱。本文旨在提供初级软件开发者对基于语言模型的工具（LLM4SE）在软件工程中使用和看法的综述。我们根据Kitchenham等人制定的指南，进行了系统文献综述（SLR），共纳入56篇主要研究，将初级软件开发者的定义定义为具有五年或以下经验的开发人员，包括计算机科学/软件工程学生。我们发现，大多数研究集中在理解和分析将AI工具集成到软件工程中的不同方面。只有9.8%的研究提供了明确的初级软件开发者的定义，且没有统一的标准。搜索相关信息是使用LLM工具最常见的任务。ChatGPT是研究（和实验）中最常见的LLM工具。大多数研究（83.9%）都报告了关于采用LLM工具的积极和消极影响的看法。我们还发现了关于LLM采用的利弊和建议进行了分类。我们的研究结果表明，开发人员不仅使用LLM进行代码生成，还用于提高他们的开发技能。更重要的是，他们不仅体验了采用LLM工具的好处，还意识到了至少一些LLM的限制，例如生成错误建议、潜在的数据泄露和AI幻觉。我们的研究结果为软件工程研究人员、教育工作者和开发人员提供了重要启示。 

---
# BIPED: Pedagogically Informed Tutoring System for ESL Education 

**Title (ZH)**: 双足式：面向 ESL 教育的教育启发式辅导系统 

**Authors**: Soonwoo Kwon, Sojung Kim, Minju Park, Seunghyun Lee, Kyuseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2406.03486)  

**Abstract**: Large Language Models (LLMs) have a great potential to serve as readily available and cost-efficient Conversational Intelligent Tutoring Systems (CITS) for teaching L2 learners of English. Existing CITS, however, are designed to teach only simple concepts or lack the pedagogical depth necessary to address diverse learning strategies. To develop a more pedagogically informed CITS capable of teaching complex concepts, we construct a BIlingual PEDagogically-informed Tutoring Dataset (BIPED) of one-on-one, human-to-human English tutoring interactions. Through post-hoc analysis of the tutoring interactions, we come up with a lexicon of dialogue acts (34 tutor acts and 9 student acts), which we use to further annotate the collected dataset. Based on a two-step framework of first predicting the appropriate tutor act then generating the corresponding response, we implemented two CITS models using GPT-4 and SOLAR-KO, respectively. We experimentally demonstrate that the implemented models not only replicate the style of human teachers but also employ diverse and contextually appropriate pedagogical strategies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在提供易于获取且成本低廉的对话式智能辅导系统（CITS）方面具有巨大潜力，以教授英语第二语言（L2）学习者。然而，现有的CITS仅限于教授简单的概念，缺乏必要的教学深度，无法应对多样化的学习策略。为了开发一种更具教学针对性的CITS，能够教授复杂概念，我们构建了一个双向教学对话的教育导向型对话数据集（BIPED），包含一对一的人与人之间的英语辅导对话。通过事后分析这些辅导对话，我们提出了一组对话行为词汇表（34种教师行为和9种学生行为），并用于进一步标注收集的数据集。基于两步框架——首先预测合适的教师行为，然后生成相应的回应，我们分别使用GPT-4和SOLAR-KO实现了两种CITS模型。我们的实验证明，所实现的模型不仅能够模仿人类教师的风格，还能够采用多样化的且适用当前情境的教学策略。 

---
# Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach 

**Title (ZH)**: 计算机化自适应测验中选择偏差的校正：基于用户汇总影响函数的方法 

**Authors**: Soonwoo Kwon, Sojung Kim, Seunghyun Lee, Jin-Young Kim, Suyeong An, Kyuseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2308.11912)  

**Abstract**: Computerized Adaptive Testing (CAT) is a widely used, efficient test mode that adapts to the examinee's proficiency level in the test domain. CAT requires pre-trained item profiles, for CAT iteratively assesses the student real-time based on the registered items' profiles, and selects the next item to administer using candidate items' profiles. However, obtaining such item profiles is a costly process that involves gathering a large, dense item-response data, then training a diagnostic model on the collected data. In this paper, we explore the possibility of leveraging response data collected in the CAT service. We first show that this poses a unique challenge due to the inherent selection bias introduced by CAT, i.e., more proficient students will receive harder questions. Indeed, when naively training the diagnostic model using CAT response data, we observe that item profiles deviate significantly from the ground-truth. To tackle the selection bias issue, we propose the user-wise aggregate influence function method. Our intuition is to filter out users whose response data is heavily biased in an aggregate manner, as judged by how much perturbation the added data will introduce during parameter estimation. This way, we may enhance the performance of CAT while introducing minimal bias to the item profiles. We provide extensive experiments to demonstrate the superiority of our proposed method based on the three public datasets and one dataset that contains real-world CAT response data. 

**Abstract (ZH)**: 计算机化自适应测试中响应数据的应用探索 

---
# Principal deuterium Hugoniot via Quantum Monte Carlo and $Δ$-learning 

**Title (ZH)**: 使用量子蒙特卡洛和Δ学习的主氘化胡oneot 李维数研究 

**Authors**: Giacomo Tenti, Kousuke Nakano, Andrea Tirelli, Sandro Sorella, Michele Casula  

**Link**: [PDF](https://arxiv.org/pdf/2301.03570)  

**Abstract**: We present a study of the principal deuterium Hugoniot for pressures up to $150$ GPa, using Machine Learning potentials (MLPs) trained with Quantum Monte Carlo (QMC) energies, forces and pressures. In particular, we adopted a recently proposed workflow based on the combination of Gaussian kernel regression and $\Delta$-learning. By fully taking advantage of this method, we explicitly considered finite-temperature electrons in the dynamics, whose effects are highly relevant for temperatures above $10$ kK. The Hugoniot curve obtained by our MLPs shows a good agreement with the most recent experiments, particularly in the region below 60 GPa. At larger pressures, our Hugoniot curve is slightly more compressible than the one yielded by experiments, whose uncertainties generally increase, however, with pressure. Our work demonstrates that QMC can be successfully combined with $\Delta$-learning to deploy reliable MLPs for complex extended systems across different thermodynamic conditions, by keeping the QMC precision at the computational cost of a mean-field calculation. 

**Abstract (ZH)**: 我们使用经量子蒙特卡罗（QMC）能量、力和压力训练的机器学习势（MLPs），研究了直至150 GPa的主要氘化氢霍斯迪昂曲线。特别地，我们采用了最近提出的工作流，该工作流基于高斯核回归和Δ学习的结合。充分利用这种方法，我们在动力学中明确考虑了有限温度电子的影响，这些影响对于超过10 kK的温度尤其重要。我们MLPs获得的霍斯迪昂曲线在60 GPa以下区域与最新实验结果有很好的一致性。在较大压力下，我们获得的霍斯迪昂曲线略比实验结果更可压缩，而实验不确定性的增加程度通常随压力的增大而增大。我们的研究证明了QMC可以成功与Δ学习结合，部署适用于不同热力学条件下复杂扩展系统的可靠MLPs，同时保持QMC计算的精度成本在均场计算的范围内。 

---
