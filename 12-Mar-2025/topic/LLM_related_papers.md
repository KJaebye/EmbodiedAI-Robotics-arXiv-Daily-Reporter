# LightPlanner: Unleashing the Reasoning Capabilities of Lightweight Large Language Models in Task Planning 

**Title (ZH)**: LightPlanner: 释放轻量级大规模语言模型在任务规划中的推理能力 

**Authors**: Weijie Zhou, Yi Peng, Manli Tao, Chaoyang Zhao, Honghui Dong, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08508)  

**Abstract**: In recent years, lightweight large language models (LLMs) have garnered significant attention in the robotics field due to their low computational resource requirements and suitability for edge deployment. However, in task planning -- particularly for complex tasks that involve dynamic semantic logic reasoning -- lightweight LLMs have underperformed. To address this limitation, we propose a novel task planner, LightPlanner, which enhances the performance of lightweight LLMs in complex task planning by fully leveraging their reasoning capabilities. Unlike conventional planners that use fixed skill templates, LightPlanner controls robot actions via parameterized function calls, dynamically generating parameter values. This approach allows for fine-grained skill control and improves task planning success rates in complex scenarios. Furthermore, we introduce hierarchical deep reasoning. Before generating each action decision step, LightPlanner thoroughly considers three levels: action execution (feedback verification), semantic parsing (goal consistency verification), and parameter generation (parameter validity verification). This ensures the correctness of subsequent action controls. Additionally, we incorporate a memory module to store historical actions, thereby reducing context length and enhancing planning efficiency for long-term tasks. We train the LightPlanner-1.5B model on our LightPlan-40k dataset, which comprises 40,000 action controls across tasks with 2 to 13 action steps. Experiments demonstrate that our model achieves the highest task success rate despite having the smallest number of parameters. In tasks involving spatial semantic reasoning, the success rate exceeds that of ReAct by 14.9 percent. Moreover, we demonstrate LightPlanner's potential to operate on edge devices. 

**Abstract (ZH)**: 轻量级大型语言模型在复杂任务规划中的增强研究：LightPlanner及其应用 

---
# MetaFold: Language-Guided Multi-Category Garment Folding Framework via Trajectory Generation and Foundation Model 

**Title (ZH)**: MetaFold：语言引导的多类别服装折叠框架通过轨迹生成和基础模型 

**Authors**: Haonan Chen, Junxiao Li, Ruihai Wu, Yiwei Liu, Yiwen Hou, Zhixuan Xu, Jingxiang Guo, Chongkai Gao, Zhenyu Wei, Shensi Xu, Jiaqi Huang, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08372)  

**Abstract**: Garment folding is a common yet challenging task in robotic manipulation. The deformability of garments leads to a vast state space and complex dynamics, which complicates precise and fine-grained manipulation. Previous approaches often rely on predefined key points or demonstrations, limiting their generalization across diverse garment categories. This paper presents a framework, MetaFold, that disentangles task planning from action prediction, learning each independently to enhance model generalization. It employs language-guided point cloud trajectory generation for task planning and a low-level foundation model for action prediction. This structure facilitates multi-category learning, enabling the model to adapt flexibly to various user instructions and folding tasks. Experimental results demonstrate the superiority of our proposed framework. Supplementary materials are available on our website: this https URL. 

**Abstract (ZH)**: 服装折叠是机器人操作中一个常见而又具有挑战性的任务。服装的可变形性导致了巨大的状态空间和复杂的动力学，这使得精确和细致的操作变得复杂。以往的方法通常依赖于预先定义的关键点或演示，限制了它们在不同服装类别间的泛化能力。本文提出了一种框架，MetaFold，它将任务规划与动作预测分离，并独立学习这些内容以增强模型的泛化能力。该框架使用语言引导的点云轨迹生成进行任务规划，并使用低层基础模型进行动作预测。这种结构便于多类别学习，使模型能够灵活适应各种用户指令和折叠任务。实验结果展示了我们提出框架的优势。更多资料请参见我们的网站：this https URL。 

---
# FASIONAD++ : Integrating High-Level Instruction and Information Bottleneck in FAt-Slow fusION Systems for Enhanced Safety in Autonomous Driving with Adaptive Feedback 

**Title (ZH)**: FASIONAD++：在自适应反馈驱动下，将高层指令和信息瓶颈集成到fat-slow融合系统中以增强自动驾驶安全性 

**Authors**: Kangan Qian, Ziang Luo, Sicong Jiang, Zilin Huang, Jinyu Miao, Zhikun Ma, Tianze Zhu, Jiayin Li, Yangfan He, Zheng Fu, Yining Shi, Boyue Wang, Hezhe Lin, Ziyu Chen, Jiangbo Yu, Xinyu Jiao, Mengmeng Yang, Kun Jiang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08162)  

**Abstract**: Ensuring safe, comfortable, and efficient planning is crucial for autonomous driving systems. While end-to-end models trained on large datasets perform well in standard driving scenarios, they struggle with complex low-frequency events. Recent Large Language Models (LLMs) and Vision Language Models (VLMs) advancements offer enhanced reasoning but suffer from computational inefficiency. Inspired by the dual-process cognitive model "Thinking, Fast and Slow", we propose $\textbf{FASIONAD}$ -- a novel dual-system framework that synergizes a fast end-to-end planner with a VLM-based reasoning module. The fast system leverages end-to-end learning to achieve real-time trajectory generation in common scenarios, while the slow system activates through uncertainty estimation to perform contextual analysis and complex scenario resolution. Our architecture introduces three key innovations: (1) A dynamic switching mechanism enabling slow system intervention based on real-time uncertainty assessment; (2) An information bottleneck with high-level plan feedback that optimizes the slow system's guidance capability; (3) A bidirectional knowledge exchange where visual prompts enhance the slow system's reasoning while its feedback refines the fast planner's decision-making. To strengthen VLM reasoning, we develop a question-answering mechanism coupled with reward-instruct training strategy. In open-loop experiments, FASIONAD achieves a $6.7\%$ reduction in average $L2$ trajectory error and $28.1\%$ lower collision rate. 

**Abstract (ZH)**: 确保安全、舒适和高效的规划对于自动驾驶系统至关重要。尽管端到端模型在标准驾驶场景中表现出色，但在处理复杂的低频事件时却力不从心。最近的大语言模型（LLMs）和视觉语言模型（VLMs）的进展虽然增强了推理能力，但计算效率较低。受“思考，快与慢”双重认知模型的启发，我们提出了一种新颖的双重系统框架FASIONAD——该框架结合了快速的端到端规划器和基于VLM的推理模块。快速系统利用端到端学习实现实时常见场景下的轨迹生成，而慢速系统通过不确定性评估激活，进行上下文分析和复杂场景的解决。我们的架构引入了三大创新点：（1）动态切换机制，基于实时不确定性评估启用慢速系统干预；（2）高级计划反馈的信息瓶颈，优化了慢速系统指导能力；（3）双向知识交换，其中视觉提示增强了慢速系统的推理能力，而其反馈则提高了快速规划器的决策精度。为了增强VLM推理能力，我们开发了一种问题回答机制，结合了带有奖励指导的训练策略。在开环实验中，FASIONAD实现了平均$L2$轨迹误差降低6.7%，碰撞率降低了28.1%。 

---
# LTLCodeGen: Code Generation of Syntactically Correct Temporal Logic for Robot Task Planning 

**Title (ZH)**: LTLCodeGen：用于机器人任务规划的语法正确的时序逻辑代码生成 

**Authors**: Behrad Rabiei, Mahesh Kumar A.R., Zhirui Dai, Surya L.S.R. Pilla, Qiyue Dong, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2503.07902)  

**Abstract**: This paper focuses on planning robot navigation tasks from natural language specifications. We develop a modular approach, where a large language model (LLM) translates the natural language instructions into a linear temporal logic (LTL) formula with propositions defined by object classes in a semantic occupancy map. The LTL formula and the semantic occupancy map are provided to a motion planning algorithm to generate a collision-free robot path that satisfies the natural language instructions. Our main contribution is LTLCodeGen, a method to translate natural language to syntactically correct LTL using code generation. We demonstrate the complete task planning method in real-world experiments involving human speech to provide navigation instructions to a mobile robot. We also thoroughly evaluate our approach in simulated and real-world experiments in comparison to end-to-end LLM task planning and state-of-the-art LLM-to-LTL translation methods. 

**Abstract (ZH)**: 本文专注于从自然语言规范规划机器人的导航任务。我们开发了一种模块化方法，其中大规模语言模型（LLM）将自然语言指令翻译成线性时序逻辑（LTL）公式，该公式中的命题由语义占用图中的对象类别定义。LTL公式和语义占用图提供给运动规划算法以生成满足自然语言指令的无碰撞机器人路径。我们的主要贡献是LTLCodeGen方法，这是一种使用代码生成将自然语言翻译为语法正确的LTL的方法。我们在涉及人类语音提供导航指令给移动机器人的实际实验中展示了完整的任务规划方法。我们还在模拟和实际实验中全面评估了我们的方法，并将其与端到端LLM任务规划方法和最先进的LLM到LTL翻译方法进行了比较。 

---
# Safety Guardrails for LLM-Enabled Robots 

**Title (ZH)**: LLM驱动机器人安全防护准则 

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2503.07885)  

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots. 

**Abstract (ZH)**: RoboGuard：一种两阶段护栏架构以确保大语言模型增强机器人安全 

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
# Beyond Outlining: Heterogeneous Recursive Planning for Adaptive Long-form Writing with Language Models 

**Title (ZH)**: 超越提纲：语言模型驱动的异构递归规划方法实现适应性长文写作 

**Authors**: Ruibin Xiong, Yimeng Chen, Dmitrii Khizbullin, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2503.08275)  

**Abstract**: Long-form writing agents require flexible integration and interaction across information retrieval, reasoning, and composition. Current approaches rely on predetermined workflows and rigid thinking patterns to generate outlines before writing, resulting in constrained adaptability during writing. In this paper we propose a general agent framework that achieves human-like adaptive writing through recursive task decomposition and dynamic integration of three fundamental task types, i.e. retrieval, reasoning, and composition. Our methodology features: 1) a planning mechanism that interleaves recursive task decomposition and execution, eliminating artificial restrictions on writing workflow; and 2) integration of task types that facilitates heterogeneous task decomposition. Evaluations on both fiction writing and technical report generation show that our method consistently outperforms state-of-the-art approaches across all automatic evaluation metrics, which demonstrate the effectiveness and broad applicability of our proposed framework. 

**Abstract (ZH)**: 长文写作代理需要在信息检索、推理和写作之间灵活集成和交互。当前的方法依赖于预先确定的工作流程和僵化的思维模式，在写作前生成提纲，导致写作过程中适应性受限。本文提出了一种通用代理框架，通过递归任务分解和动态整合检索、推理和写作三种基本任务类型，实现类似人类的适应性写作。我们的方法包含：1) 一种交替进行递归任务分解和执行的规划机制，消除写作流程的人为限制；2) 任务类型的整合，促进异构任务分解。对虚构写作和技术报告生成的评估表明，我们的方法在所有自动评估指标上一致优于现有最先进的方法，证明了我们提出的框架的有效性和广泛适用性。 

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
# Counterfactual Language Reasoning for Explainable Recommendation Systems 

**Title (ZH)**: 可因果语境推理的可解释推荐系统 

**Authors**: Guanrong Li, Haolin Yang, Xinyu Liu, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08051)  

**Abstract**: Explainable recommendation systems leverage transparent reasoning to foster user trust and improve decision-making processes. Current approaches typically decouple recommendation generation from explanation creation, violating causal precedence principles where explanatory factors should logically precede outcomes. This paper introduces a novel framework integrating structural causal models with large language models to establish causal consistency in recommendation pipelines. Our methodology enforces explanation factors as causal antecedents to recommendation predictions through causal graph construction and counterfactual adjustment. We particularly address the confounding effect of item popularity that distorts personalization signals in explanations, developing a debiasing mechanism that disentangles genuine user preferences from conformity bias. Through comprehensive experiments across multiple recommendation scenarios, we demonstrate that CausalX achieves superior performance in recommendation accuracy, explanation plausibility, and bias mitigation compared to baselines. 

**Abstract (ZH)**: 可解释的推荐系统通过透明的推理来培养用户信任并改善决策过程。当前的方法通常将推荐生成与解释生成分开，违反了因果优先原则，即解释因素应逻辑上先于结果。本文提出了一种新的框架，将结构因果模型与大规模语言模型结合起来，以在推荐管道中建立因果一致性。我们的方法通过因果图构建和反事实调整，强制解释因素作为推荐预测的因果先驱。我们特别解决了商品流行性造成的共因效应，该效应扭曲了解释中的个性化信号，开发了一种去偏机制，以分离出真实的用户偏好与从众偏差。通过在多个推荐场景中的全面实验，我们展示了CausalX在推荐准确性、解释合理性和偏见缓解方面的优越性能，相比基线方法表现更佳。 

---
# LLM-Powered Knowledge Graphs for Enterprise Intelligence and Analytics 

**Title (ZH)**: 基于LLM的企业智能与分析知识图谱 

**Authors**: Rajeev Kumar, Kumar Ishan, Harishankar Kumar, Abhinandan Singla  

**Link**: [PDF](https://arxiv.org/pdf/2503.07993)  

**Abstract**: Disconnected data silos within enterprises obstruct the extraction of actionable insights, diminishing efficiency in areas such as product development, client engagement, meeting preparation, and analytics-driven decision-making. This paper introduces a framework that uses large language models (LLMs) to unify various data sources into a comprehensive, activity-centric knowledge graph. The framework automates tasks such as entity extraction, relationship inference, and semantic enrichment, enabling advanced querying, reasoning, and analytics across data types like emails, calendars, chats, documents, and logs. Designed for enterprise flexibility, it supports applications such as contextual search, task prioritization, expertise discovery, personalized recommendations, and advanced analytics to identify trends and actionable insights. Experimental results demonstrate its success in the discovery of expertise, task management, and data-driven decision making. By integrating LLMs with knowledge graphs, this solution bridges disconnected systems and delivers intelligent analytics-powered enterprise tools. 

**Abstract (ZH)**: 企业内部孤立的数据孤岛阻碍了可操作洞察的提取，影响产品开发、客户互动、会议准备及基于分析的决策效率。本文提出了一种框架，利用大型语言模型（LLMs）将各种数据源统一成一个综合性的活动中心知识图谱。该框架自动化执行实体提取、关系推理和语义 enrichment 等任务，支持不同数据类型（如电子邮件、日历、聊天记录、文档和日志）的高级查询、推理与分析。该解决方案具有企业灵活性，支持上下文搜索、任务优先级排序、专家发现、个性化推荐以及高级分析，以识别趋势和可操作洞察。实验结果表明，该框架在专家发现、任务管理和数据驱动决策方面取得了成功。通过将大型语言模型与知识图谱集成，该解决方案连接了孤立的系统，并提供了智能分析驱动的企业工具。 

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
# RefactorBench: Evaluating Stateful Reasoning in Language Agents Through Code 

**Title (ZH)**: RefactorBench: 通过代码评估语言代理中的状态推理性能 

**Authors**: Dhruv Gautam, Spandan Garg, Jinu Jang, Neel Sundaresan, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2503.07832)  

**Abstract**: Recent advances in language model (LM) agents and function calling have enabled autonomous, feedback-driven systems to solve problems across various digital domains. To better understand the unique limitations of LM agents, we introduce RefactorBench, a benchmark consisting of 100 large handcrafted multi-file refactoring tasks in popular open-source repositories. Solving tasks within RefactorBench requires thorough exploration of dependencies across multiple files and strong adherence to relevant instructions. Every task is defined by 3 natural language instructions of varying specificity and is mutually exclusive, allowing for the creation of longer combined tasks on the same repository. Baselines on RefactorBench reveal that current LM agents struggle with simple compositional tasks, solving only 22% of tasks with base instructions, in contrast to a human developer with short time constraints solving 87%. Through trajectory analysis, we identify various unique failure modes of LM agents, and further explore the failure mode of tracking past actions. By adapting a baseline agent to condition on representations of state, we achieve a 43.9% improvement in solving RefactorBench tasks. We further extend our state-aware approach to encompass entire digital environments and outline potential directions for future research. RefactorBench aims to support the study of LM agents by providing a set of real-world, multi-hop tasks within the realm of code. 

**Abstract (ZH)**: Recent Advances in Language Model Agents and Function Calling Have Enabled Autonomous, Feedback-Driven Systems to Solve Problems Across Various Digital Domains: Introducing RefactorBench，一个由100个大型手工编写的多文件重构任务组成的基准，来自流行开源仓库。 

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
# YuE: Scaling Open Foundation Models for Long-Form Music Generation 

**Title (ZH)**: YuE: 扩展开放基础模型以进行长篇音乐生成 

**Authors**: Ruibin Yuan, Hanfeng Lin, Shuyue Guo, Ge Zhang, Jiahao Pan, Yongyi Zang, Haohe Liu, Yiming Liang, Wenye Ma, Xingjian Du, Xinrun Du, Zhen Ye, Tianyu Zheng, Yinghao Ma, Minghao Liu, Zeyue Tian, Ziya Zhou, Liumeng Xue, Xingwei Qu, Yizhi Li, Shangda Wu, Tianhao Shen, Ziyang Ma, Jun Zhan, Chunhui Wang, Yatian Wang, Xiaowei Chi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Shansong Liu, Lingrui Mei, Peng Li, Junjie Wang, Jianwei Yu, Guojian Pang, Xu Li, Zihao Wang, Xiaohuan Zhou, Lijun Yu, Emmanouil Benetos, Yong Chen, Chenghua Lin, Xie Chen, Gus Xia, Zhaoxiang Zhang, Chao Zhang, Wenhu Chen, Xinyu Zhou, Xipeng Qiu, Roger Dannenberg, Jiaheng Liu, Jian Yang, Wenhao Huang, Wei Xue, Xu Tan, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08638)  

**Abstract**: We tackle the task of long-form music generation--particularly the challenging \textbf{lyrics-to-song} problem--by introducing YuE, a family of open foundation models based on the LLaMA2 architecture. Specifically, YuE scales to trillions of tokens and generates up to five minutes of music while maintaining lyrical alignment, coherent musical structure, and engaging vocal melodies with appropriate accompaniment. It achieves this through (1) track-decoupled next-token prediction to overcome dense mixture signals, (2) structural progressive conditioning for long-context lyrical alignment, and (3) a multitask, multiphase pre-training recipe to converge and generalize. In addition, we redesign the in-context learning technique for music generation, enabling versatile style transfer (e.g., converting Japanese city pop into an English rap while preserving the original accompaniment) and bidirectional generation. Through extensive evaluation, we demonstrate that YuE matches or even surpasses some of the proprietary systems in musicality and vocal agility. In addition, fine-tuning YuE enables additional controls and enhanced support for tail languages. Furthermore, beyond generation, we show that YuE's learned representations can perform well on music understanding tasks, where the results of YuE match or exceed state-of-the-art methods on the MARBLE benchmark. Keywords: lyrics2song, song generation, long-form, foundation model, music generation 

**Abstract (ZH)**: 长篇音乐生成——特别是具有挑战性的歌词到歌曲问题——我们通过引入基于LLaMA2架构的YuE这一系列开放基础模型来解决该任务。具体而言，YuE可以扩展到万亿级 Tokens，并生成长达五分钟的音乐，同时保持歌词对齐、连贯的音乐结构以及吸引人的伴奏旋律。它通过以下方式实现这一目标：（1）轨道解耦的下一步预测以克服密集混合信号，（2）结构渐进式条件概率以实现长上下文歌词对齐，以及（3）多任务、多阶段预训练配方以实现收敛和泛化。此外，我们重新设计了音乐生成中的在上下文学习技术，使其能够实现多样的风格转换（例如，在保留原伴奏的同时将日式城市流行音乐转换为英文说唱）和双向生成。通过广泛的评估，我们证明YuE在音乐性和歌唱灵活性方面能够匹敌甚至超越一些专有系统。此外，微调YuE能够提供额外的控制并增强对尾部语言的支持。此外，除了生成之外，我们还展示了YuE学习到的表示在音乐理解任务中表现出色，YuE在MARBLE基准上的表现与最先进的方法相当或超越。关键词：歌词到歌曲，歌曲生成，长篇，基础模型，音乐生成。 

---
# BiasEdit: Debiasing Stereotyped Language Models via Model Editing 

**Title (ZH)**: BiasEdit: 基于模型编辑的消除刻板印象语言模型方法 

**Authors**: Xin Xu, Wei Xu, Ningyu Zhang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2503.08588)  

**Abstract**: Previous studies have established that language models manifest stereotyped biases. Existing debiasing strategies, such as retraining a model with counterfactual data, representation projection, and prompting often fail to efficiently eliminate bias or directly alter the models' biased internal representations. To address these issues, we propose BiasEdit, an efficient model editing method to remove stereotypical bias from language models through lightweight networks that act as editors to generate parameter updates. BiasEdit employs a debiasing loss guiding editor networks to conduct local edits on partial parameters of a language model for debiasing while preserving the language modeling abilities during editing through a retention loss. Experiments on StereoSet and Crows-Pairs demonstrate the effectiveness, efficiency, and robustness of BiasEdit in eliminating bias compared to tangental debiasing baselines and little to no impact on the language models' general capabilities. In addition, we conduct bias tracing to probe bias in various modules and explore bias editing impacts on different components of language models. 

**Abstract (ZH)**: Previous studies have established that language models manifest stereotyped biases. Existing debiasing strategies, such as retraining a model with counterfactual data, representation projection, and prompting often fail to efficiently eliminate bias or directly alter the models' biased internal representations. To address these issues, we propose BiasEdit, an efficient model editing method to remove stereotypical bias from language models through lightweight networks that act as editors to generate parameter updates. BiasEdit employs a debiasing loss guiding editor networks to conduct local edits on partial parameters of a language model for debiasing while preserving the language modeling abilities during editing through a retention loss. Experiments on StereoSet and Crows-Pairs demonstrate the effectiveness, efficiency, and robustness of BiasEdit in eliminating bias compared to tangential debiasing baselines and little to no impact on the language models' general capabilities. In addition, we conduct bias tracing to probe bias in various modules and explore bias editing impacts on different components of language models. 

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
# V-Max: Making RL practical for Autonomous Driving 

**Title (ZH)**: V-Max: 让强化学习在自动驾驶中更加实用 

**Authors**: Valentin Charraut, Thomas Tournaire, Waël Doulazmi, Thibault Buhet  

**Link**: [PDF](https://arxiv.org/pdf/2503.08388)  

**Abstract**: Learning-based decision-making has the potential to enable generalizable Autonomous Driving (AD) policies, reducing the engineering overhead of rule-based approaches. Imitation Learning (IL) remains the dominant paradigm, benefiting from large-scale human demonstration datasets, but it suffers from inherent limitations such as distribution shift and imitation gaps. Reinforcement Learning (RL) presents a promising alternative, yet its adoption in AD remains limited due to the lack of standardized and efficient research frameworks. To this end, we introduce V-Max, an open research framework providing all the necessary tools to make RL practical for AD. V-Max is built on Waymax, a hardware-accelerated AD simulator designed for large-scale experimentation. We extend it using ScenarioNet's approach, enabling the fast simulation of diverse AD datasets. V-Max integrates a set of observation and reward functions, transformer-based encoders, and training pipelines. Additionally, it includes adversarial evaluation settings and an extensive set of evaluation metrics. Through a large-scale benchmark, we analyze how network architectures, observation functions, training data, and reward shaping impact RL performance. 

**Abstract (ZH)**: 基于学习的决策制定有望实现通用可移植的自动驾驶（AD）策略，减少基于规则的方法的工程开销。模仿学习（IL）仍是主导范式，得益于大规模的人类示范数据集，但受限于分布偏移和模仿差距等固有局限。强化学习（RL）提供了一种有前途的替代方案，但由于缺乏标准化和高效的研发框架，其在AD中的应用仍受到限制。为此，我们介绍了V-Max，这是一个开放的研究框架，提供所有必要的工具以使RL在AD中实用。V-Max基于Waymax构建，Waymax是一个硬件加速的AD模拟器，适用于大规模实验。我们使用ScenarioNet的方法对其进行扩展，使其能够快速模拟多样的AD数据集。V-Max集成了观测函数和奖励函数、基于变换器的编码器以及训练管道。此外，它还包括对抗性评估设置和一系列评估指标。通过大规模基准测试，我们分析了网络架构、观测函数、训练数据和回报塑造对RL性能的影响。 

---
# Large Language Model as Meta-Surrogate for Data-Driven Many-Task Optimization: A Proof-of-Principle Study 

**Title (ZH)**: 大型语言模型作为数据驱动多任务优化的元代理：一个原理验证研究 

**Authors**: Xian-Rong Zhang, Yue-Jiao Gong, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08301)  

**Abstract**: In many-task optimization scenarios, surrogate models are valuable for mitigating the computational burden of repeated fitness evaluations across tasks. This study proposes a novel meta-surrogate framework to assist many-task optimization, by leveraging the knowledge transfer strengths and emergent capabilities of large language models (LLMs). We formulate a unified framework for many-task fitness prediction, by defining a universal model with metadata to fit a group of problems. Fitness prediction is performed on metadata and decision variables, enabling efficient knowledge sharing across tasks and adaptability to new tasks. The LLM-based meta-surrogate treats fitness prediction as conditional probability estimation, employing a unified token sequence representation for task metadata, inputs, and outputs. This approach facilitates efficient inter-task knowledge sharing through shared token embeddings and captures complex task dependencies via multi-task model training. Experimental results demonstrate the model's emergent generalization ability, including zero-shot performance on problems with unseen dimensions. When integrated into evolutionary transfer optimization (ETO), our framework supports dual-level knowledge transfer -- at both the surrogate and individual levels -- enhancing optimization efficiency and robustness. This work establishes a novel foundation for applying LLMs in surrogate modeling, offering a versatile solution for many-task optimization. 

**Abstract (ZH)**: 在多任务优化场景中，代理模型对于缓解跨任务重复适应性评估的计算负担非常有价值。本研究提出了一种新型元代理框架，通过利用大规模语言模型（LLMs）的知识迁移优势和新兴能力来辅助多任务优化。我们制定了一个统一的多任务适应性预测框架，通过定义一个通用模型来拟合一组问题，并对元数据和决策变量进行适应性预测，以实现跨任务的高效知识共享和新任务的适应性。基于LLM的元代理将适应性预测视为条件概率估计，并采用统一的标记序列表示形式来表示任务元数据、输入和输出。该方法通过共享标记嵌入来促进跨任务的高效知识共享，并通过多任务模型训练捕捉复杂的任务依赖关系。实验结果表明，该模型具有潜在的泛化能力，包括对未见过的维度问题的零样本性能。当结合进化转移优化（ETO）时，该框架支持双重层面的知识迁移——在代理和个体层面——从而提高优化效率和鲁棒性。本研究为在代理建模中应用LLMs奠定了新的基础，并为多任务优化提供了多功能解决方案。 

---
# Large Language Models for Outpatient Referral: Problem Definition, Benchmarking and Challenges 

**Title (ZH)**: 门诊转诊的大语言模型应用：问题定义、基准测试与挑战 

**Authors**: Xiaoxiao Liu, Qingying Xiao, Junying Chen, Xiangyi Feng, Xiangbo Wu, Bairui Zhang, Xiang Wan, Jian Chang, Guangjun Yu, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08292)  

**Abstract**: Large language models (LLMs) are increasingly applied to outpatient referral tasks across healthcare systems. However, there is a lack of standardized evaluation criteria to assess their effectiveness, particularly in dynamic, interactive scenarios. In this study, we systematically examine the capabilities and limitations of LLMs in managing tasks within Intelligent Outpatient Referral (IOR) systems and propose a comprehensive evaluation framework specifically designed for such systems. This framework comprises two core tasks: static evaluation, which focuses on evaluating the ability of predefined outpatient referrals, and dynamic evaluation, which evaluates capabilities of refining outpatient referral recommendations through iterative dialogues. Our findings suggest that LLMs offer limited advantages over BERT-like models, but show promise in asking effective questions during interactive dialogues. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗保健系统中的门诊转诊任务中越来越多地被应用。然而，缺乏标准化的评估标准来评估其效果，特别是在动态和交互式场景中。在本研究中，我们系统地考察了LLMs在智能门诊转诊（IOR）系统中管理任务的能力及其局限性，并提出了一套专门针对此类系统的全面评估框架。该框架包括两个核心任务：静态评估，侧重于评估预定义门诊转诊的能力；动态评估，通过迭代对话评估改善门诊转诊推荐的能力。我们的研究发现，LLMs在与用户交互时提出有效问题方面优于BERT-like模型，显示出一定的潜力。 

---
# DeepRAG: Building a Custom Hindi Embedding Model for Retrieval Augmented Generation from Scratch 

**Title (ZH)**: DeepRAG: 从 scratch 构建一种定制化印地语嵌入模型以增强生成 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2503.08213)  

**Abstract**: In this paper, I present our work on DeepRAG, a specialized embedding model we built specifically for Hindi language in RAG systems. While LLMs have gotten really good at generating text, their performance in retrieval tasks still depends heavily on having quality embeddings - something that's been lacking for Hindi despite being one of the world's most spoken languages. We tackled this by creating embeddings from the ground up rather than just fine-tuning existing models. Our process involved collecting diverse Hindi texts (over 2.7M samples), training a custom SentencePiece tokenizer that actually understands Hindi morphology, designing transformer architecture with Hindi-specific attention mechanisms, and optimizing with contrastive learning. Results were honestly better than I expected - we saw a 23% improvement in retrieval precision compared to the multilingual models everyone's been using. The paper details our methodology, which I think could help others working with low-resource languages where the one-size-fits-all multilingual models fall short. We've also integrated our embeddings with LangChain to build complete Hindi RAG systems, which might be useful for practitioners. While there's still tons more to explore, I believe this work addresses a critical gap for Hindi NLP and demonstrates why language-specific approaches matter. 

**Abstract (ZH)**: 本论文介绍了我们为RAG系统构建的专门嵌入模型DeepRAG，针对的是印地语语言。虽然大规模语言模型在生成文本方面表现优异，但在检索任务上的性能仍然高度依赖高质量的嵌入，这对印地语来说尤其缺乏，尽管它是世界上使用最广泛的语言之一。我们通过从零开始创建嵌入，而不是仅仅对现有模型进行微调来解决这一问题。我们的过程包括收集多样化的印地语文本（超过270万样本）、训练一个真正理解印地语形态结构的自定义SentencePiece分词器、设计具有印地语特定注意力机制的变压器架构，并采用对比学习进行优化。结果比我预期的要好 - 我们在检索精度上看到了23%的提升，超过了大家一直在使用的多语言模型。本文详细介绍了我们的方法论，我认为这可能对其他处理资源匮乏语言的研究者有所帮助，这些语言通常不适合通用的多语言模型。我们还将我们的嵌入与LangChain集成，构建了完整的印地语RAG系统，这可能对实践者有用。尽管仍有更多探索的空间，但我认为这项工作解决了印地语NLP的一个关键缺口，并证明了特定语言方法的重要性。 

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
# In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents 

**Title (ZH)**: 展望与回顾：长期个性化对话代理的反射式内存管理 

**Authors**: Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2503.08026)  

**Abstract**: Large Language Models (LLMs) have made significant progress in open-ended dialogue, yet their inability to retain and retrieve relevant information from long-term interactions limits their effectiveness in applications requiring sustained personalization. External memory mechanisms have been proposed to address this limitation, enabling LLMs to maintain conversational continuity. However, existing approaches struggle with two key challenges. First, rigid memory granularity fails to capture the natural semantic structure of conversations, leading to fragmented and incomplete representations. Second, fixed retrieval mechanisms cannot adapt to diverse dialogue contexts and user interaction patterns. In this work, we propose Reflective Memory Management (RMM), a novel mechanism for long-term dialogue agents, integrating forward- and backward-looking reflections: (1) Prospective Reflection, which dynamically summarizes interactions across granularities-utterances, turns, and sessions-into a personalized memory bank for effective future retrieval, and (2) Retrospective Reflection, which iteratively refines the retrieval in an online reinforcement learning (RL) manner based on LLMs' cited evidence. Experiments show that RMM demonstrates consistent improvement across various metrics and benchmarks. For example, RMM shows more than 10% accuracy improvement over the baseline without memory management on the LongMemEval dataset. 

**Abstract (ZH)**: 长周期对话代理的反思性记忆管理（Reflective Memory Management for Long-term Dialogue Agents） 

---
# EFPC: Towards Efficient and Flexible Prompt Compression 

**Title (ZH)**: EFPC: 向高效灵活提示压缩迈进 

**Authors**: Yun-Hao Cao, Yangsong Wang, Shuzheng Hao, Zhenxing Li, Chengjun Zhan, Sichao Liu, Yi-Qi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07956)  

**Abstract**: The emergence of large language models (LLMs) like GPT-4 has revolutionized natural language processing (NLP), enabling diverse, complex tasks. However, extensive token counts lead to high computational and financial burdens. To address this, we propose Efficient and Flexible Prompt Compression (EFPC), a novel method unifying task-aware and task-agnostic compression for a favorable accuracy-efficiency trade-off. EFPC uses GPT-4 to generate compressed prompts and integrates them with original prompts for training. During training and inference, we selectively prepend user instructions and compress prompts based on predicted probabilities. EFPC is highly data-efficient, achieving significant performance with minimal data. Compared to the state-of-the-art method LLMLingua-2, EFPC achieves a 4.8% relative improvement in F1-score with 1% additional data at a 4x compression rate, and an 11.4% gain with 10% additional data on the LongBench single-doc QA benchmark. EFPC's unified framework supports broad applicability and enhances performance across various models, tasks, and domains, offering a practical advancement in NLP. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-4的涌现重塑了自然语言处理（NLP）， enables diverse, complex tasks.然而，广泛的标记数量导致了高计算和财务负担。为了解决这一问题，我们提出了高效灵活的提示压缩（EFPC）方法，这是一种新颖的统合任务感知和任务无关压缩的方法，以实现有利的准确率-效率权衡。EFPC 使用 GPT-4 生成压缩提示，并将其与原始提示结合用于训练。在训练和推理过程中，我们根据预测概率选择性地添加用户指令并压缩提示。EFPC 高度数据高效，即使在少量数据的情况下也能实现显著的性能提升。与最先进的方法 LLMLingua-2 相比，EFPC 在 4 倍压缩率下，利用 1% 的额外数据实现了 4.8% 的相对 F1 分数改进；在 10% 的额外数据下，于 LongBench 单文档 QA 基准测试中实现了 11.4% 的性能增益。EFPC 的统一框架支持广泛的适用性，并在各种模型、任务和领域中提升了性能，提供了一种在 NLP 中的实际进展。 

---
# Gemini Embedding: Generalizable Embeddings from Gemini 

**Title (ZH)**: Gemini嵌入：来自Gemini的通用嵌入 

**Authors**: Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Koert Chen, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, Tom Duerig  

**Link**: [PDF](https://arxiv.org/pdf/2503.07891)  

**Abstract**: In this report, we introduce Gemini Embedding, a state-of-the-art embedding model leveraging the power of Gemini, Google's most capable large language model. Capitalizing on Gemini's inherent multilingual and code understanding capabilities, Gemini Embedding produces highly generalizable embeddings for text spanning numerous languages and textual modalities. The representations generated by Gemini Embedding can be precomputed and applied to a variety of downstream tasks including classification, similarity, clustering, ranking, and retrieval. Evaluated on the Massive Multilingual Text Embedding Benchmark (MMTEB), which includes over one hundred tasks across 250+ languages, Gemini Embedding substantially outperforms prior state-of-the-art models, demonstrating considerable improvements in embedding quality. Achieving state-of-the-art performance across MMTEB's multilingual, English, and code benchmarks, our unified model demonstrates strong capabilities across a broad selection of tasks and surpasses specialized domain-specific models. 

**Abstract (ZH)**: Gemini嵌入：基于Google最强大的大型语言模型的先进嵌入模型 

---
# LLMIdxAdvis: Resource-Efficient Index Advisor Utilizing Large Language Model 

**Title (ZH)**: LLMIdxAdvis：利用大型语言模型的资源高效索引顾问 

**Authors**: Xinxin Zhao, Haoyang Li, Jing Zhang, Xinmei Huang, Tieying Zhang, Jianjun Chen, Rui Shi, Cuiping Li, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07884)  

**Abstract**: Index recommendation is essential for improving query performance in database management systems (DBMSs) through creating an optimal set of indexes under specific constraints. Traditional methods, such as heuristic and learning-based approaches, are effective but face challenges like lengthy recommendation time, resource-intensive training, and poor generalization across different workloads and database schemas. To address these issues, we propose LLMIdxAdvis, a resource-efficient index advisor that uses large language models (LLMs) without extensive fine-tuning. LLMIdxAdvis frames index recommendation as a sequence-to-sequence task, taking target workload, storage constraint, and corresponding database environment as input, and directly outputting recommended indexes. It constructs a high-quality demonstration pool offline, using GPT-4-Turbo to synthesize diverse SQL queries and applying integrated heuristic methods to collect both default and refined labels. During recommendation, these demonstrations are ranked to inject database expertise via in-context learning. Additionally, LLMIdxAdvis extracts workload features involving specific column statistical information to strengthen LLM's understanding, and introduces a novel inference scaling strategy combining vertical scaling (via ''Index-Guided Major Voting'' and Best-of-N) and horizontal scaling (through iterative ''self-optimization'' with database feedback) to enhance reliability. Experiments on 3 OLAP and 2 real-world benchmarks reveal that LLMIdxAdvis delivers competitive index recommendation with reduced runtime, and generalizes effectively across different workloads and database schemas. 

**Abstract (ZH)**: LLMIdxAdvis：一种基于大型语言模型的高效索引建议方法 

---
# HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations 

**Title (ZH)**: HalluVerse25：大语言模型幻觉的细颗粒度多语言基准数据集 

**Authors**: Samir Abdaljalil, Hasan Kurban, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07833)  

**Abstract**: Large Language Models (LLMs) are increasingly used in various contexts, yet remain prone to generating non-factual content, commonly referred to as "hallucinations". The literature categorizes hallucinations into several types, including entity-level, relation-level, and sentence-level hallucinations. However, existing hallucination datasets often fail to capture fine-grained hallucinations in multilingual settings. In this work, we introduce HalluVerse25, a multilingual LLM hallucination dataset that categorizes fine-grained hallucinations in English, Arabic, and Turkish. Our dataset construction pipeline uses an LLM to inject hallucinations into factual biographical sentences, followed by a rigorous human annotation process to ensure data quality. We evaluate several LLMs on HalluVerse25, providing valuable insights into how proprietary models perform in detecting LLM-generated hallucinations across different contexts. 

**Abstract (ZH)**: Large Language Models (LLMs)在各种情境中的应用日益增多，但仍容易生成非事实内容，通常称为“幻觉”。已有文献将幻觉分类为实体级、关系级和句子级幻觉。然而，现有的幻觉数据集往往难以在多语言环境中捕捉到细粒度的幻觉。在本工作中，我们介绍了HalluVerse25，这是一个多语言LLM幻觉数据集，它在英语、阿拉伯语和土耳其语中分类了细粒度的幻觉。我们的数据集构建管道使用LLM将幻觉注入事实性传记句子中，然后通过严格的 humano 人工注释过程确保数据质量。我们对HalluVerse25上的几种LLM进行了评估，提供了关于不同情境下商业模型检测LLM生成的幻觉的表现的宝贵见解。 

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
# Evaluating LLaMA 3.2 for Software Vulnerability Detection 

**Title (ZH)**: 评估LLaMA 3.2在软件漏洞检测中的性能 

**Authors**: José Gonçalves, Miguel Silva, Bernardo Cabral, Tiago Dias, Eva Maia, Isabel Praça, Ricardo Severino, Luís Lino Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2503.07770)  

**Abstract**: Deep Learning (DL) has emerged as a powerful tool for vulnerability detection, often outperforming traditional solutions. However, developing effective DL models requires large amounts of real-world data, which can be difficult to obtain in sufficient quantities. To address this challenge, DiverseVul dataset has been curated as the largest dataset of vulnerable and non-vulnerable C/C++ functions extracted exclusively from real-world projects. Its goal is to provide high-quality, large-scale samples for training DL models. However, during our study several inconsistencies were identified in the raw dataset while applying pre-processing techniques, highlighting the need for a refined version. In this work, we present a refined version of DiverseVul dataset, which is used to fine-tune a large language model, LLaMA 3.2, for vulnerability detection. Experimental results show that the use of pre-processing techniques led to an improvement in performance, with the model achieving an F1-Score of 66%, a competitive result when compared to our baseline, which achieved a 47% F1-Score in software vulnerability detection. 

**Abstract (ZH)**: DiverseVul数据集精修版及其在LLaMA 3.2模型上用于软件漏洞检测的研究 

---
# Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM 

**Title (ZH)**: 层次均衡打包：面向长上下文LLM的高效监督微调 

**Authors**: Yongqiang Yao, Jingru Tan, Kaihuan Liang, Feizhao Zhang, Yazhe Niu, Jiahao Hu, Ruihao Gong, Dahua Lin, Ningyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07680)  

**Abstract**: Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MOE model, our method speeds up the training by 2.4$\times$ with competitive performance. 

**Abstract (ZH)**: 培训长上下文大型语言模型（LLMs）具有挑战性，因为长上下文和短上下文数据的混合训练往往导致工作负载不平衡。现有工作主要使用数据打包来缓解这一问题，但未能考虑不均衡的注意力计算和浪费的通信开销。本文提出了层次平衡打包（HBP），设计了一种新的批次构建方法和训练策略以解决这些不效率。具体而言，HBP 构建多级数据打包组，每组优化不同的打包长度。它将训练样本分配到最合适的组，并为每个组配置最有效的设置，包括顺序并行度和梯度检查点配置。为了有效利用多级数据组，我们设计了一个专门针对HBP 的动态训练管道，包括渐进式学习、自适应顺序并行度和稳定损失。我们的广泛实验表明，我们的方法在多个数据集和开源模型上显著减少了训练时间，同时保持了强大的性能。对于最大的DeepSeek-V2（236B）混合专家模型，我们的方法将训练速度提高了2.4倍，同时保持了竞争力的性能。 

---
# DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems 

**Title (ZH)**: DynTaskMAS：一种基于异步并行LLM驱动的动态任务图框架的多-agent系统 

**Authors**: Junwei Yu, Yepeng Ding, Hiroyuki Sato  

**Link**: [PDF](https://arxiv.org/pdf/2503.07675)  

**Abstract**: The emergence of Large Language Models (LLMs) in Multi-Agent Systems (MAS) has opened new possibilities for artificial intelligence, yet current implementations face significant challenges in resource management, task coordination, and system efficiency. While existing frameworks demonstrate the potential of LLM-based agents in collaborative problem-solving, they often lack sophisticated mechanisms for parallel execution and dynamic task management. This paper introduces DynTaskMAS, a novel framework that orchestrates asynchronous and parallel operations in LLM-based MAS through dynamic task graphs. The framework features four key innovations: (1) a Dynamic Task Graph Generator that intelligently decomposes complex tasks while maintaining logical dependencies, (2) an Asynchronous Parallel Execution Engine that optimizes resource utilization through efficient task scheduling, (3) a Semantic-Aware Context Management System that enables efficient information sharing among agents, and (4) an Adaptive Workflow Manager that dynamically optimizes system performance. Experimental evaluations demonstrate that DynTaskMAS achieves significant improvements over traditional approaches: a 21-33% reduction in execution time across task complexities (with higher gains for more complex tasks), a 35.4% improvement in resource utilization (from 65% to 88%), and near-linear throughput scaling up to 16 concurrent agents (3.47X improvement for 4X agents). Our framework establishes a foundation for building scalable, high-performance LLM-based multi-agent systems capable of handling complex, dynamic tasks efficiently. 

**Abstract (ZH)**: 大型语言模型在多agent系统中的新兴应用为人工智能打开了新的可能性，然而当前实现面临资源管理、任务协调和系统效率的重大挑战。尽管现有的框架展示了基于大型语言模型的代理在协作问题解决方面的潜力，但它们通常缺乏复杂的并行执行和动态任务管理机制。本文介绍了DynTaskMAS，这是一种新颖的框架，利用动态任务图在基于大型语言模型的多agent系统中协调异步和并行操作。该框架包含四项关键创新：(1) 动态任务图生成器，能够智能地分解复杂任务并保持逻辑依赖关系；(2) 异步并行执行引擎，通过高效的任务调度优化资源利用；(3) 语义感知上下文管理系统，使代理之间能够高效地共享信息；(4) 自适应工作流管理器，动态优化系统性能。实验评估表明，DynTaskMAS在任务复杂性方面取得了显著改进：执行时间减少21-33%（复杂任务的增幅更大），资源利用率提高35.4%（从65%提高到88%），并且在16个并发代理的范围内实现接近线性的吞吐量扩展（4倍代理数量时性能提升3.47倍）。我们的框架为构建可扩展、高性能的基于大型语言模型的多agent系统奠定了基础，这些系统能够高效地处理复杂、动态的任务。 

---
# Merge then Realign: Simple and Effective Modality-Incremental Continual Learning for Multimodal LLMs 

**Title (ZH)**: 合并再对齐：简单有效的模态增量连续学习方法用于多模态LLM 

**Authors**: Dingkun Zhang, Shuhan Qi, Xinyu Xiao, Kehai Chen, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07663)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enhanced their versatility as they integrate a growing number of modalities. Considering the heavy cost of training MLLMs, it is necessary to reuse the existing ones and further extend them to more modalities through Modality-incremental Continual Learning (MCL). However, this often comes with a performance degradation in the previously learned modalities. In this work, we revisit the MCL and investigate a more severe issue it faces in contrast to traditional continual learning, that its degradation comes not only from catastrophic forgetting but also from the misalignment between the modality-agnostic and modality-specific components. To address this problem, we propose an elegantly simple MCL paradigm called "MErge then ReAlign" (MERA). Our method avoids introducing heavy training overhead or modifying the model architecture, hence is easy to deploy and highly reusable in the MLLM community. Extensive experiments demonstrate that, despite the simplicity of MERA, it shows impressive performance, holding up to a 99.84% Backward Relative Gain when extending to four modalities, achieving a nearly lossless MCL performance. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models through Modality-Incremental Continual Learning: The M Erge then Re Align Paradigm 

---
# SplitQuantV2: Enhancing Low-Bit Quantization of LLMs Without GPUs 

**Title (ZH)**: SplitQuantV2: 不使用GPU提升大模型低比特量化性能 

**Authors**: Jaewoo Song, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07657)  

**Abstract**: The quantization of large language models (LLMs) is crucial for deploying them on devices with limited computational resources. While advanced quantization algorithms offer improved performance compared to the basic linear quantization, they typically require high-end graphics processing units (GPUs), are often restricted to specific deep neural network (DNN) frameworks, and require calibration datasets. This limitation poses challenges for using such algorithms on various neural processing units (NPUs) and edge AI devices, which have diverse model formats and frameworks. In this paper, we show SplitQuantV2, an innovative algorithm designed to enhance low-bit linear quantization of LLMs, can achieve results comparable to those of advanced algorithms. SplitQuantV2 preprocesses models by splitting linear and convolution layers into functionally equivalent, quantization-friendly structures. The algorithm's platform-agnostic, concise, and efficient nature allows for implementation without the need for GPUs. Our evaluation on the Llama 3.2 1B Instruct model using the AI2's Reasoning Challenge (ARC) dataset demonstrates that SplitQuantV2 improves the accuracy of the INT4 quantization model by 11.76%p, matching the performance of the original floating-point model. Remarkably, SplitQuantV2 took only 2 minutes 6 seconds to preprocess the 1B model and perform linear INT4 quantization using only an Apple M4 CPU. SplitQuantV2 provides a practical solution for low-bit quantization on LLMs, especially when complex, computation-intensive algorithms are inaccessible due to hardware limitations or framework incompatibilities. 

**Abstract (ZH)**: 大语言模型的量化对于在有限算力资源的设备上部署它们至关重要。虽然高级量化算法相较于基本线性量化提供了更好的性能，但它们通常需要高性能图形处理单元（GPU），经常局限于特定的深层神经网络（DNN）框架，并需要校准数据集。这种限制使得在各种神经处理单元（NPU）和边缘AI设备上使用这些算法变得具有挑战性，这些设备具有多样化的模型格式和框架。本文展示了SplitQuantV2这一创新算法，旨在增强大语言模型的低比特线性量化，其结果可与高级算法相媲美。SplitQuantV2通过将线性和卷积层拆分为功能等效且易于量化的结构来预处理模型。该算法的平台无关性、简洁性和高效性使得在无需GPU的情况下进行实现成为可能。使用AI2的推理挑战（ARC）数据集对Llama 3.2 1B Instruct模型的评估表明，SplitQuantV2将INT4量化模型的准确率提高了11.76个百分点，达到了原始浮点模型的性能。令人惊讶的是，SplitQuantV2仅用Apple M4 CPU就花费了2分6秒时间完成了1B模型的预处理和线性INT4量化。SplitQuantV2为大语言模型的低比特量化提供了一种实用的解决方案，特别适用于因硬件限制或框架不兼容而无法使用复杂、计算密集型算法的情形。 

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
