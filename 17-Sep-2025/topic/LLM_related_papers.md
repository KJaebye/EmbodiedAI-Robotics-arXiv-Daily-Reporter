# Toward Ownership Understanding of Objects: Active Question Generation with Large Language Model and Probabilistic Generative Model 

**Title (ZH)**: 面向对象所有权理解的主动问题生成：大型语言模型与概率生成模型相结合 

**Authors**: Saki Hashimoto, Shoichi Hasegawa, Tomochika Ishikawa, Akira Taniguchi, Yoshinobu Hagiwara, Lotfi El Hafi, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12754)  

**Abstract**: Robots operating in domestic and office environments must understand object ownership to correctly execute instructions such as ``Bring me my cup.'' However, ownership cannot be reliably inferred from visual features alone. To address this gap, we propose Active Ownership Learning (ActOwL), a framework that enables robots to actively generate and ask ownership-related questions to users. ActOwL employs a probabilistic generative model to select questions that maximize information gain, thereby acquiring ownership knowledge efficiently to improve learning efficiency. Additionally, by leveraging commonsense knowledge from Large Language Models (LLM), objects are pre-classified as either shared or owned, and only owned objects are targeted for questioning. Through experiments in a simulated home environment and a real-world laboratory setting, ActOwL achieved significantly higher ownership clustering accuracy with fewer questions than baseline methods. These findings demonstrate the effectiveness of combining active inference with LLM-guided commonsense reasoning, advancing the capability of robots to acquire ownership knowledge for practical and socially appropriate task execution. 

**Abstract (ZH)**: 家用和办公环境中操作的机器人必须理解对象的所有权，以便正确执行“ Bring me my cup. ”等指令。然而，所有权仅从视觉特征中无法可靠地推断出来。为了解决这个问题，我们提出了主动所有权学习（ActOwL）框架，该框架使机器人能够主动生成和向用户提问与所有权相关的问题。ActOwL 使用概率生成模型来选择最大化信息增益的问题，从而高效地获取所有权知识，提高学习效率。此外，通过利用大型语言模型（LLM）的常识知识，对物体进行预先分类为共用或拥有，并仅针对拥有对象进行提问。通过在模拟家庭环境和现实世界实验室中的实验，ActOwL 在提出更少问题的情况下实现了显著更高的所有权聚类准确性，证明了结合主动推断与LLM引导的常识推理的有效性，推动了机器人获取所有权知识以执行实用和社交上适当任务的能力。 

---
# RepIt: Representing Isolated Targets to Steer Language Models 

**Title (ZH)**: RepIt: 表征孤立目标以引导语言模型 

**Authors**: Vincent Siu, Nathan W. Henry, Nicholas Crispino, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13281)  

**Abstract**: While activation steering in large language models (LLMs) is a growing area of research, methods can often incur broader effects than desired. This motivates isolation of purer concept vectors to enable targeted interventions and understand LLM behavior at a more granular level. We present RepIt, a simple and data-efficient framework for isolating concept-specific representations. Across five frontier LLMs, RepIt enables precise interventions: it selectively suppresses refusal on targeted concepts while preserving refusal elsewhere, producing models that answer WMD-related questions while still scoring as safe on standard benchmarks. We further show that the corrective signal localizes to just 100-200 neurons and that robust target representations can be extracted from as few as a dozen examples on a single A6000. This efficiency raises a dual concern: manipulations can be performed with modest compute and data to extend to underrepresented data-scarce topics while evading existing benchmarks. By disentangling refusal vectors with RepIt, this work demonstrates that targeted interventions can counteract overgeneralization, laying the foundation for more granular control of model behavior. 

**Abstract (ZH)**: RepIt：一种简单高效的概念特异性表示隔离框架 

---
# Reasoning with Preference Constraints: A Benchmark for Language Models in Many-to-One Matching Markets 

**Title (ZH)**: 基于偏好约束的推理：多对一匹配市场中语言模型基准测试 

**Authors**: Marylou Fauchard, Florian Carichon, Margarida Carvalho, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13131)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have demonstrated strong performance on complex mathematical tasks, including combinatorial optimization. Techniques such as Chain-of-Thought and In-Context Learning have further enhanced this capability, making LLMs both powerful and accessible tools for a wide range of users, including non-experts. However, applying LLMs to matching problems, which require reasoning under preferential and structural constraints, remains underexplored. To address this gap, we introduce a novel benchmark of 369 instances of the College Admission Problem, a canonical example of a matching problem with preferences, to evaluate LLMs across key dimensions: feasibility, stability, and optimality. We employ this benchmark to assess the performance of several open-weight LLMs. Our results first reveal that while LLMs can satisfy certain constraints, they struggle to meet all evaluation criteria consistently. They also show that reasoning LLMs, like QwQ and GPT-oss, significantly outperform traditional models such as Llama, Qwen or Mistral, defined here as models used without any dedicated reasoning mechanisms. Moreover, we observed that LLMs reacted differently to the various prompting strategies tested, which include Chain-of-Thought, In-Context Learning and role-based prompting, with no prompt consistently offering the best performance. Finally, we report the performances from iterative prompting with auto-generated feedback and show that they are not monotonic; they can peak early and then significantly decline in later attempts. Overall, this work offers a new perspective on model reasoning performance and the effectiveness of prompting strategies in combinatorial optimization problems with preferential constraints. 

**Abstract (ZH)**: 最近在大型语言模型（LLMs）推理方面的进展展示了其在复杂数学任务，包括组合优化方面强大的表现。通过Chain-of-Thought和In-Context Learning等技术进一步增强了这一能力，使LLMs成为强大且易于使用的工具，适用于包括非专家在内的广泛用户群体。然而，将其应用于需要偏好和结构约束推理的匹配问题仍然鲜有探索。为填补这一空白，我们引入了一个由369个大学录取问题实例构成的新基准，以评估LLMs在可行性、稳定性和最优性等关键维度上的表现。我们利用此基准评估了多个开源权重LLMs的性能。研究结果表明，尽管LLMs能够满足某些约束，但它们很难在所有评估标准上保持一致性。此外，推理LLMs，如QwQ和GPT-oss，在性能上显著优于传统模型Llama、Qwen或Mistral（此处为未使用专门推理机制的模型）。我们还发现，LLMs对测试的不同提示策略有不同的反应，这些策略包括Chain-of-Thought、In-Context Learning和角色提示，没有一种提示策略在所有情况下都能取得最佳表现。最后，我们报告了使用自动生成反馈进行迭代提示的性能，并指出这些性能并非单调变化；它们可能会早期达到峰值，然后在后续尝试中显著下降。总体而言，这项工作为偏好约束下的组合优化问题中模型推理性能以及提示策略的有效性提供了新的视角。 

---
# A Visualized Framework for Event Cooperation with Generative Agents 

**Title (ZH)**: 生成代理参与事件合作的可视化框架 

**Authors**: Yuyang Tian, Shunqiang Mao, Wenchang Gao, Lanlan Qiu, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.13011)  

**Abstract**: Large Language Models (LLMs) have revolutionized the simulation of agent societies, enabling autonomous planning, memory formation, and social interactions. However, existing frameworks often overlook systematic evaluations for event organization and lack visualized integration with physically grounded environments, limiting agents' ability to navigate spaces and interact with items realistically. We develop MiniAgentPro, a visualization platform featuring an intuitive map editor for customizing environments and a simulation player with smooth animations. Based on this tool, we introduce a comprehensive test set comprising eight diverse event scenarios with basic and hard variants to assess agents' ability. Evaluations using GPT-4o demonstrate strong performance in basic settings but highlight coordination challenges in hard variants. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经革命性地改变了代理社会的模拟，使其能够实现自主规划、记忆形成和社交互动。然而，现有的框架往往忽略了事件组织的系统性评估，并缺乏与物理环境集成的可视化表示，限制了代理在现实环境中导航和互动的能力。我们开发了MiniAgentPro，一个可视化平台，包含一个直观的地图编辑器用于自定义环境和一个具有平滑动画的模拟播放器。基于此工具，我们引入了一个全面的测试集，包括八个多样化事件场景的基本和困难变体，以评估代理的能力。使用GPT-4o的评估表明，在基本设置中的表现强大，但在困难变体中突显了协调挑战。 

---
# Toward PDDL Planning Copilot 

**Title (ZH)**: 面向PDDL规划协作者 

**Authors**: Yarin Benyamin, Argaman Mordoch, Shahaf S. Shperberg, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2509.12987)  

**Abstract**: Large Language Models (LLMs) are increasingly being used as autonomous agents capable of performing complicated tasks. However, they lack the ability to perform reliable long-horizon planning on their own. This paper bridges this gap by introducing the Planning Copilot, a chatbot that integrates multiple planning tools and allows users to invoke them through instructions in natural language. The Planning Copilot leverages the Model Context Protocol (MCP), a recently developed standard for connecting LLMs with external tools and systems. This approach allows using any LLM that supports MCP without domain-specific fine-tuning. Our Planning Copilot supports common planning tasks such as checking the syntax of planning problems, selecting an appropriate planner, calling it, validating the plan it generates, and simulating their execution. We empirically evaluate the ability of our Planning Copilot to perform these tasks using three open-source LLMs. The results show that the Planning Copilot highly outperforms using the same LLMs without the planning tools. We also conducted a limited qualitative comparison of our tool against Chat GPT-5, a very recent commercial LLM. Our results shows that our Planning Copilot significantly outperforms GPT-5 despite relying on a much smaller LLM. This suggests dedicated planning tools may be an effective way to enable LLMs to perform planning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益被用作能够执行复杂任务的自主代理，但它们缺乏独立进行可靠长期规划的能力。本文通过引入规划 copilot，一个集成了多种规划工具并允许用户通过自然语言指令调用它们的聊天机器人，来弥合这一缺口。规划 copilot 利用了模型上下文协议（MCP），这是一种用于连接语言模型与外部工具和系统的最新标准。这种方法使用户可以在不进行特定领域微调的情况下使用任何支持 MCP 的语言模型。我们的规划 copilot 支持常见的规划任务，如检查规划问题的语法、选择合适的规划器、调用规划器、验证其生成的计划并模拟其执行。我们使用三个开源语言模型实证评估了规划 copilot 完成这些任务的能力。结果表明，与没有规划工具的语言模型相比，规划 copilot 显著表现出色。我们还对我们的工具与非常近期的商用语言模型 ChatGPT-5 进行了有限的定性比较。结果显示，尽管依赖的模型更小，但我们的规划 copilot 在性能上显著优于 GPT-5，这表明专用规划工具可能是使语言模型能够执行规划任务的有效方式。 

---
# Black-box Model Merging for Language-Model-as-a-Service with Massive Model Repositories 

**Title (ZH)**: 面向语言模型即服务的大规模模型仓库的黑盒模型合并方法 

**Authors**: Shilian Chen, Jie Zhou, Tianyu Huai, Yujiang Lu, Junsong Li, Bihao Zhan, Qianjun Pan, Yutao Yang, Xin Li, Qin Chen, Hang Yan, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.12951)  

**Abstract**: Model merging refers to the process of integrating multiple distinct models into a unified model that preserves and combines the strengths and capabilities of the individual models. Most existing approaches rely on task vectors to combine models, typically under the assumption that model parameters are accessible. However, for extremely large language models (LLMs) such as GPT-4, which are often provided solely as black-box services through API interfaces (Language-Model-as-a-Service), model weights are not available to end users. This presents a significant challenge, which we refer to as black-box model merging (BMM) with massive LLMs. To address this challenge, we propose a derivative-free optimization framework based on the evolutionary algorithm (Evo-Merging) that enables effective model merging using only inference-time API queries. Our method consists of two key components: (1) sparsity-based denoising, designed to identify and filter out irrelevant or redundant information across models, and (2) sign-aware scaling, which dynamically computes optimal combination weights for the relevant models based on their performance. We also provide a formal justification, along with a theoretical analysis, for our asymmetric sparsification. Extensive experimental evaluations demonstrate that our approach achieves state-of-the-art results on a range of tasks, significantly outperforming existing strong baselines. 

**Abstract (ZH)**: 黑盒大型语言模型的模型合并（BMM）及其基于进化算法的无导数优化框架 

---
# The Anatomy of Alignment: Decomposing Preference Optimization by Steering Sparse Features 

**Title (ZH)**: _alignment的解剖：通过引导稀疏特征分解偏好优化_ 

**Authors**: Jeremias Ferrao, Matthijs van der Lende, Ilija Lichkovski, Clement Neo  

**Link**: [PDF](https://arxiv.org/pdf/2509.12934)  

**Abstract**: Aligning large language models is critical for their usability and safety. However, the prevailing approach of Reinforcement Learning from Human Feedback (RLHF) induces diffuse, opaque parameter changes, making it difficult to discern what the model has internalized. Hence, we introduce Feature Steering with Reinforcement Learning (FSRL), a transparent alignment framework that trains a lightweight adapter to steer behavior by modulating interpretable features from a Sparse Autoencoder (SAE). First, we demonstrate that FSRL is an effective method for preference optimization and is comparable with current RLHF methods. We then perform mechanistic analysis on the trained adapter, and find that its policy systematically promotes style features over explicit alignment concepts, suggesting that the preference optimization process rewards stylistic presentation as a proxy for quality. Ultimately, we hope that FSRL provides a tool for both interpretable model control and diagnosing the internal mechanisms of alignment. 

**Abstract (ZH)**: 特征引导的强化学习 Alignment：一种透明的模型对齐框架 

---
# Stochastic Streets: A Walk Through Random LLM Address Generation in four European Cities 

**Title (ZH)**: 随机街道：穿越欧洲四座城市中随机语言模型地址生成的随机漫步 

**Authors**: Tairan Fu, David Campo-Nazareno, Javier Coronado-Blázquez, Javier Conde, Pedro Reviriego, Fabrizio Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12914)  

**Abstract**: Large Language Models (LLMs) are capable of solving complex math problems or answer difficult questions on almost any topic, but can they generate random street addresses for European cities? 

**Abstract (ZH)**: 大型语言模型（LLMs）能够解决复杂数学问题或回答几乎任何主题的难题，但它们能为欧洲城市生成随机街道地址吗？ 

---
# LTA-thinker: Latent Thought-Augmented Training Framework for Large Language Models on Complex Reasoning 

**Title (ZH)**: LTA-thinker：潜在思维增强的大语言模型复杂推理训练框架 

**Authors**: Jiaqi Wang, Binquan Ji, Haibo Luo, Yiyang Qi, Ruiting Li, Huiyan Wang, Yuantao Han, Cangyi Yang, jiaxu Zhang, Feiliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.12875)  

**Abstract**: Complex Reasoning in Large Language Models can be dynamically optimized using Test-Time Scaling (TTS) to mitigate Overthinking. Methods such as Coconut, SoftCoT and its variant are effective in continuous latent space inference, the core bottleneck still lies in the efficient generation and utilization of high-quality Latent Thought. Drawing from the theory of SoftCoT++ that a larger variance in the generated Latent Thought distribution more closely approximates the golden truth distribution, we propose a Latent Thought-Augmented Training Framework--LTA-Thinker, which improves distributional variance and enhances reasoning performance from two perspectives. First, LTA-Thinker constructs a Latent Thought generation architecture based on a learnable prior. This architecture aims to increase the variance distribution of generated Latent Thought Vectors in order to simplify the overall structure and raise the performance ceiling. Second, LTA-Thinker introduces a distribution-based directional optimization paradigm that jointly constrains both distribution locality and distribution scale. This mechanism improves information efficiency and computational cost through a multi-objective co-training strategy, which combines standard Supervised Fine-Tuning (SFT) loss with two novel losses: Semantic Alignment Loss, which utilizes KL divergence to ensure that the Latent Thought is highly relevant to the semantics of the question; Reasoning Focus Loss, which utilizes a contrastive learning mechanism to guide the model to focus on the most critical reasoning steps. Experiments show that LTA-thinker achieves state-of-the-art (SOTA) performance among various baselines and demonstrates a higher performance ceiling and better scaling effects. 

**Abstract (ZH)**: 大规模语言模型中的复杂推理可以通过测试时缩放（TTS）动态优化以减轻过度推理。LTA-Thinker：基于潜在思想增强的训练框架及其应用 

---
# H$^2$R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents 

**Title (ZH)**: H$^2$R：层级 hindsight 反思用于多任务语言模型代理 

**Authors**: Shicheng Ye, Chao Yu, Kaiqiang Ke, Chengdong Xu, Yinqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.12810)  

**Abstract**: Large language model (LLM)-based agents have shown strong potential in multi-task scenarios, owing to their ability to transfer knowledge across diverse tasks. However, existing approaches often treat prior experiences and knowledge as monolithic units, leading to inefficient and coarse-grained knowledge transfer. In this work, we propose a novel hierarchical memory architecture that enables fine-grained knowledge transfer by decoupling high-level planning memory from low-level execution memory. To construct and refine these hierarchical memories, we introduce Hierarchical Hindsight Reflection (H$^2$R), a mechanism that distills reusable and hierarchical knowledge from past agent-environment interactions. At test time, H$^2$R performs retrievals of high-level and low-level memories separately, allowing LLM-based agents to efficiently access and utilize task-relevant knowledge for new this http URL results across two benchmarks demonstrate that H$^2$R can improve generalization and decision-making performance, outperforming prior baselines such as Expel. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理在多任务场景中展示了强大的潜力，得益于它们跨多样化任务转移知识的能力。然而，现有方法往往会将先前的经验和知识视为单调的整体，导致知识转移效率低下且颗粒度粗。在本工作中，我们提出了一种新的分层记忆架构，通过分离高层次规划记忆和低层次执行记忆，实现精细粒度的知识转移。为了构建和细化这些分层记忆，我们引入了分层前瞻性反思（H$^2$R）机制，该机制从过去的代理-环境交互中提炼可复用的分层次知识。在测试时，H$^2$R分别检索高层次和低层次记忆，使基于LLM的代理能够高效地访问和利用与任务相关的知识进行新任务。实验结果在两个基准上表明，H$^2$R可以提高泛化能力和决策性能，超越了诸如Expel在内的先前基线方法。 

---
# Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs 

**Title (ZH)**: 零-shot 图推理基于LLMs的检索增强框架 

**Authors**: Hanqing Li, Kiran Sheena Jyothi, Henry Liang, Sharika Mahadevan, Diego Klabjan  

**Link**: [PDF](https://arxiv.org/pdf/2509.12743)  

**Abstract**: We propose a new, training-free method, Graph Reasoning via Retrieval Augmented Framework (GRRAF), that harnesses retrieval-augmented generation (RAG) alongside the code-generation capabilities of large language models (LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target graph is stored in a graph database, and the LLM is prompted to generate executable code queries that retrieve the necessary information. This approach circumvents the limitations of existing methods that require extensive finetuning or depend on predefined algorithms, and it incorporates an error feedback loop with a time-out mechanism to ensure both correctness and efficiency. Experimental evaluations on the GraphInstruct dataset reveal that GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle detection, bipartite graph checks, shortest path computation, and maximum flow, while maintaining consistent token costs regardless of graph sizes. Imperfect but still very high performance is observed on subgraph matching. Notably, GRRAF scales effectively to large graphs with up to 10,000 nodes. 

**Abstract (ZH)**: 基于检索增强框架的图推理新方法：无需训练的图推理通过检索增强框架（GRRAF） 

---
# Large Language Models Imitate Logical Reasoning, but at what Cost? 

**Title (ZH)**: 大型语言模型模仿逻辑推理，但代价是什么？ 

**Authors**: Lachlan McGinness, Peter Baumgartner  

**Link**: [PDF](https://arxiv.org/pdf/2509.12645)  

**Abstract**: We present a longitudinal study which evaluates the reasoning capability of frontier Large Language Models over an eighteen month period. We measured the accuracy of three leading models from December 2023, September 2024 and June 2025 on true or false questions from the PrOntoQA dataset and their faithfulness to reasoning strategies provided through in-context learning. The improvement in performance from 2023 to 2024 can be attributed to hidden Chain of Thought prompting. The introduction of thinking models allowed for significant improvement in model performance between 2024 and 2025.
We then present a neuro-symbolic architecture which uses LLMs of less than 15 billion parameters to translate the problems into a standardised form. We then parse the standardised forms of the problems into a program to be solved by Z3, an SMT solver, to determine the satisfiability of the query. We report the number of prompt and completion tokens as well as the computational cost in FLOPs for open source models. The neuro-symbolic approach significantly reduces the computational cost while maintaining near perfect performance. The common approximation that the number of inference FLOPs is double the product of the active parameters and total tokens was accurate within 10\% for all experiments. 

**Abstract (ZH)**: 一种前沿大规模语言模型的 longitudinal 研究：十八个月的推理能力评估与分析 

---
# Learn to Relax with Large Language Models: Solving Nonlinear Combinatorial Optimization Problems via Bidirectional Coevolution 

**Title (ZH)**: 用大型语言模型学会放松：通过双向共进化求解非线性组合优化问题 

**Authors**: Beidan Liu, Zhengqiu Zhu, Chen Gao, Yong Zhao, Wei Qi, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.12643)  

**Abstract**: Nonlinear Combinatorial Optimization Problems (NCOPs) present a formidable computational hurdle in practice, as their nonconvex nature gives rise to multi-modal solution spaces that defy efficient optimization. Traditional constraint relaxation approaches rely heavily on expert-driven, iterative design processes that lack systematic automation and scalable adaptability. While recent Large Language Model (LLM)-based optimization methods show promise for autonomous problem-solving, they predominantly function as passive constraint validators rather than proactive strategy architects, failing to handle the sophisticated constraint interactions inherent to this http URL address these limitations, we introduce the first end-to-end \textbf{Auto}mated \textbf{C}onstraint \textbf{O}ptimization (AutoCO) method, which revolutionizes NCOPs resolution through learning to relax with this http URL, we leverage structured LLM reasoning to generate constraint relaxation strategies, which are dynamically evolving with algorithmic principles and executable code through a unified triple-representation scheme. We further establish a novel bidirectional (global-local) coevolution mechanism that synergistically integrates Evolutionary Algorithms for intensive local refinement with Monte Carlo Tree Search for systematic global strategy space exploration, ensuring optimal balance between intensification and diversification in fragmented solution spaces. Finally, comprehensive experiments on three challenging NCOP benchmarks validate AutoCO's consistent effectiveness and superior performance over the baselines. 

**Abstract (ZH)**: 非线性组合优化问题（NCOPs）的实际计算挑战难以克服，由于其非凸性质导致多模态解空间难以高效优化。传统约束松弛方法依赖于专家驱动的迭代设计过程，缺乏系统自动化和可扩展适应性。虽然基于大规模语言模型（LLM）的优化方法有望实现自主问题解决，但它们主要作为被动约束验证者，而非主动策略架构师，无法处理此类问题中固有的复杂约束交互。为应对这些限制，我们提出了第一个端到端的Auto Constraint Optimization（AutoCO）方法，该方法通过学习来实现约束松弛，我们利用结构化的大规模语言模型推理生成约束松弛策略，并通过统一的三重表示方案动态演化算法原理和可执行代码。此外，我们建立了新颖的双向（全局-局部）协同进化机制，该机制将进化算法与蒙特卡洛树搜索相结合，以系统地探索策略空间，确保在碎片化解空间中平衡强化与多样化。最后，三项具有挑战性的NCOP基准测试中的全面实验验证了AutoCO的一贯有效性，并且优于基线方法。 

---
# ECG-aBcDe: Overcoming Model Dependence, Encoding ECG into a Universal Language for Any LLM 

**Title (ZH)**: ECG-aBcDe: 克服模型依赖性，将心电图编码为适用于任何LLM的通用语言 

**Authors**: Yong Xia, Jingxuan Li, YeTeng Sun, Jiarui Bu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12625)  

**Abstract**: Large Language Models (LLMs) hold significant promise for electrocardiogram (ECG) analysis, yet challenges remain regarding transferability, time-scale information learning, and interpretability. Current methods suffer from model-specific ECG encoders, hindering transfer across LLMs. Furthermore, LLMs struggle to capture crucial time-scale information inherent in ECGs due to Transformer limitations. And their black-box nature limits clinical adoption. To address these limitations, we introduce ECG-aBcDe, a novel ECG encoding method that transforms ECG signals into a universal ECG language readily interpretable by any LLM. By constructing a hybrid dataset of ECG language and natural language, ECG-aBcDe enables direct fine-tuning of pre-trained LLMs without architectural modifications, achieving "construct once, use anywhere" capability. Moreover, the bidirectional convertibility between ECG and ECG language of ECG-aBcDe allows for extracting attention heatmaps from ECG signals, significantly enhancing interpretability. Finally, ECG-aBcDe explicitly represents time-scale information, mitigating Transformer limitations. This work presents a new paradigm for integrating ECG analysis with LLMs. Compared with existing methods, our method achieves competitive performance on ROUGE-L and METEOR. Notably, it delivers significant improvements in the BLEU-4, with improvements of 2.8 times and 3.9 times in in-dataset and cross-dataset evaluations, respectively, reaching scores of 42.58 and 30.76. These results provide strong evidence for the feasibility of the new paradigm. 

**Abstract (ZH)**: 基于Large Language Models的ECG分析新方法：ECG-aBcDe 

---
# Analogy-Driven Financial Chain-of-Thought (AD-FCoT): A Prompting Approach for Financial Sentiment Analysis 

**Title (ZH)**: 基于类比驱动的财务推理（AD-FCoT）：一种财务情绪分析的提示方法 

**Authors**: Anmol Singhal Navya Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2509.12611)  

**Abstract**: Financial news sentiment analysis is crucial for anticipating market movements. With the rise of AI techniques such as Large Language Models (LLMs), which demonstrate strong text understanding capabilities, there has been renewed interest in enhancing these systems. Existing methods, however, often struggle to capture the complex economic context of news and lack transparent reasoning, which undermines their reliability. We propose Analogy-Driven Financial Chain-of-Thought (AD-FCoT), a prompting framework that integrates analogical reasoning with chain-of-thought (CoT) prompting for sentiment prediction on historical financial news. AD-FCoT guides LLMs to draw parallels between new events and relevant historical scenarios with known outcomes, embedding these analogies into a structured, step-by-step reasoning chain. To our knowledge, this is among the first approaches to explicitly combine analogical examples with CoT reasoning in finance. Operating purely through prompting, AD-FCoT requires no additional training data or fine-tuning and leverages the model's internal financial knowledge to generate rationales that mirror human analytical reasoning. Experiments on thousands of news articles show that AD-FCoT outperforms strong baselines in sentiment classification accuracy and achieves substantially higher correlation with market returns. Its generated explanations also align with domain expertise, providing interpretable insights suitable for real-world financial analysis. 

**Abstract (ZH)**: 金融新闻情感分析对于预判市场动态至关重要。随着大型语言模型（LLMs）等人工智能技术的兴起，这些技术展现出强大的文本理解能力，对增强这些系统的兴趣得到了重燃。然而，现有方法往往难以捕捉新闻中的复杂经济背景，并且缺乏透明的推理过程，这削弱了它们的可靠性。我们提出了一种名为 Analogy-Driven Financial Chain-of-Thought (AD-FCoT) 的提示框架，该框架将类比推理与链式推理（CoT）提示相结合，用于历史金融新闻的情感预测。AD-FCoT 引导大语言模型通过将新事件与具有已知结果的相关历史场景进行类比，并将这些类比嵌入到结构化的、逐步的推理链中。据我们所知，这是首次明确将类比示例与金融领域的链式推理相结合的方法。AD-FCoT 完全通过提示操作，无需额外的训练数据或微调，并利用模型内部的金融知识生成类似于人类分析推理的解释。在数千篇新闻文章上的实验结果显示，AD-FCoT 在情感分类准确性上优于强大的基线方法，并且其与市场回报的相关性显著更高。其生成的解释也与领域专业知识一致，为实际的金融分析提供了可解释的洞见。 

---
# Empowering Clinical Trial Design through AI: A Randomized Evaluation of PowerGPT 

**Title (ZH)**: 通过AI赋能临床试验设计：PowerGPT的随机评估 

**Authors**: Yiwen Lu, Lu Li, Dazheng Zhang, Xinyao Jian, Tingyin Wang, Siqi Chen, Yuqing Lei, Jiayi Tong, Zhaohan Xi, Haitao Chu, Chongliang Luo, Alexis Ogdie, Brian Athey, Alparslan Turan, Michael Abramoff, Joseph C Cappelleri, Hua Xu, Yun Lu, Jesse Berlin, Daniel I. Sessler, David A. Asch, Xiaoqian Jiang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12471)  

**Abstract**: Sample size calculations for power analysis are critical for clinical research and trial design, yet their complexity and reliance on statistical expertise create barriers for many researchers. We introduce PowerGPT, an AI-powered system integrating large language models (LLMs) with statistical engines to automate test selection and sample size estimation in trial design. In a randomized trial to evaluate its effectiveness, PowerGPT significantly improved task completion rates (99.3% vs. 88.9% for test selection, 99.3% vs. 77.8% for sample size calculation) and accuracy (94.1% vs. 55.4% in sample size estimation, p < 0.001), while reducing average completion time (4.0 vs. 9.3 minutes, p < 0.001). These gains were consistent across various statistical tests and benefited both statisticians and non-statisticians as well as bridging expertise gaps. Already under deployment across multiple institutions, PowerGPT represents a scalable AI-driven approach that enhances accessibility, efficiency, and accuracy in statistical power analysis for clinical research. 

**Abstract (ZH)**: 基于AI的PowerGPT系统在临床研究中样本量计算和统计功效分析中的应用：提高效率与准确性 

---
# Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction 

**Title (ZH)**: 通过链式思维重构进行精简的推理模型可以实现精确剪枝 

**Authors**: Ryan Lucas, Kayhan Behdin, Zhipeng Wang, Qingquan Song, Shao Tang, Rahul Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2509.12464)  

**Abstract**: Reasoning language models such as DeepSeek-R1 produce long chain-of-thought traces during inference time which make them costly to deploy at scale. We show that using compression techniques such as neural network pruning produces greater performance loss than in typical language modeling tasks, and in some cases can make the model slower since they cause the model to produce more thinking tokens but with worse performance. We show that this is partly due to the fact that standard LLM pruning methods often focus on input reconstruction, whereas reasoning is a decode-dominated task. We introduce a simple, drop-in fix: during pruning we jointly reconstruct activations from the input and the model's on-policy chain-of-thought traces. This "Reasoning-Aware Compression" (RAC) integrates seamlessly into existing pruning workflows such as SparseGPT, and boosts their performance significantly. Code reproducing the results in the paper can be found at: this https URL 

**Abstract (ZH)**: 基于推理的语言模型如DeepSeek-R1在推理过程中产生长链条的思考轨迹，使其在大规模部署时成本较高。我们发现，使用压缩技术如神经网络剪枝会导致更大的性能损失，在某些情况下甚至会使模型变慢，因为这会导致模型生成更多的思考令牌但性能更差。我们表明，这 partly  partly  partly 部分归因于标准的大规模语言模型剪枝方法通常重点关注输入重构，而推理是一个以解码为主的任务。我们提出一个简单的替换方案：在剪枝过程中同时重构输入和模型的策略性链条思考轨迹。这种“推理感知压缩”(RAC)方法可以无缝集成到现有的剪枝工作流程中，如SparseGPT，并显著提升其性能。论文中结果的代码可以在此处找到：这个链接URL。 

---
# Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization 

**Title (ZH)**: 基于熵增强多轮偏好优化的编码代理构建 

**Authors**: Jiahao Yu, Zelei Cheng, Xian Wu, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.12434)  

**Abstract**: Software engineering presents complex, multi-step challenges for Large Language Models (LLMs), requiring reasoning over large codebases and coordinated tool use. The difficulty of these tasks is exemplified by benchmarks like SWE-bench, where current LLMs still struggle to resolve real-world issues.
A promising approach to enhance performance is test-time scaling (TTS), but its gains are heavily dependent on the diversity of model outputs.
While standard alignment methods such as Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO) are effective at aligning model outputs with human preferences, this process can come at the cost of reduced diversity, limiting the effectiveness of TTS.
Additionally, existing preference optimization algorithms are typically designed for single-turn tasks and do not fully address the complexities of multi-turn reasoning and tool integration required for interactive coding agents.
To bridge this gap, we introduce \sys, an entropy-enhanced framework that adapts existing preference optimization algorithms to the multi-turn, tool-assisted setting.
\sys augments the preference objective to explicitly preserve policy entropy and generalizes learning to optimize over multi-turn interactions rather than single-turn responses.
We validate \sys by fine-tuning a diverse suite of models from different families and sizes (up to 106B parameters).
To maximize performance gains from TTS, we further propose a hybrid best-trajectory selection scheme combining a learned verifier model with model free approaches.
On the \swebench leaderboard, our approach establishes new state-of-the-art results among open-weight models. A 30B parameter model trained with \sys ranks 1st on \lite and 4th on \verified on the open-weight leaderboard, surpassed only by models with over 10x more parameters(\eg$>$350B). 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程中面临复杂多步的挑战，需要在大规模代码库上进行推理并协调工具的使用。这些任务的难度在类似SWE-bench的基准测试中得到了体现，当前的LLMs仍然难以解决实际问题。

一种有前途的方法是测试时缩放（TTS），但其收益高度依赖于模型输出的多样性。

虽然直接偏好优化（DPO）和坎布纳曼-特韦斯基优化（KTO）等标准对齐方法在将模型输出与人类偏好对齐方面非常有效，但这一过程可能会导致多样性减少，从而限制TTS的有效性。

此外，现有的偏好优化算法通常设计用于单回合任务，未能充分解决互动编码代理所需的多回合推理和工具集成的复杂性。

为弥合这一差距，我们引入了\sys框架，该框架将现有的偏好优化算法适应于多回合、工具辅助的环境。

\sys扩充了偏好目标，明确保留策略的熵，并将学习泛化为优化多回合交互而不是单回合响应。

我们通过微调来自不同家庭和规模（多达106亿参数）的多样化模型集验证了\sys。

为了最大化TTS带来的性能提升，我们进一步提出了一种结合学习验证模型和模型免费方法的混合最佳轨迹选择方案。

在\swebench排行榜上，我们的方法在开放权重模型中建立了新的最先进的结果。一个训练参数量为30B的模型在\lite和\verified排行榜上分别排名第1和第4，仅优于超过其10倍以上参数量的模型（例如>350B）。 

---
# Small Models, Big Results: Achieving Superior Intent Extraction through Decomposition 

**Title (ZH)**: 小模型，大效果：通过分解实现卓越的意图提取 

**Authors**: Danielle Cohen, Yoni Halpern, Noam Kahlon, Joel Oren, Omri Berkovitch, Sapir Caduri, Ido Dagan, Anatoly Efros  

**Link**: [PDF](https://arxiv.org/pdf/2509.12423)  

**Abstract**: Understanding user intents from UI interaction trajectories remains a challenging, yet crucial, frontier in intelligent agent development. While massive, datacenter-based, multi-modal large language models (MLLMs) possess greater capacity to handle the complexities of such sequences, smaller models which can run on-device to provide a privacy-preserving, low-cost, and low-latency user experience, struggle with accurate intent inference. We address these limitations by introducing a novel decomposed approach: first, we perform structured interaction summarization, capturing key information from each user action. Second, we perform intent extraction using a fine-tuned model operating on the aggregated summaries. This method improves intent understanding in resource-constrained models, even surpassing the base performance of large MLLMs. 

**Abstract (ZH)**: 从UI交互轨迹理解用户意图仍然是智能代理开发中的一个具有挑战性但至关重要的前沿问题。我们通过引入一种新颖的分解方法来解决这些限制：首先，我们执行结构化的交互总结，从每个用户动作中捕获关键信息。其次，我们使用在聚合总结上进行微调的模型来提取意图。这种方法在资源受限的模型中提高了意图理解能力，甚至超过了大型MLLM的基本性能。 

---
# LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences 

**Title (ZH)**: LLMAP: LLM辅助多目标路径规划及用户偏好考虑 

**Authors**: Liangqi Yuan, Dong-Jun Han, Christopher G. Brinton, Sabine Brunswicker  

**Link**: [PDF](https://arxiv.org/pdf/2509.12273)  

**Abstract**: The rise of large language models (LLMs) has made natural language-driven route planning an emerging research area that encompasses rich user objectives. Current research exhibits two distinct approaches: direct route planning using LLM-as-Agent and graph-based searching strategies. However, LLMs in the former approach struggle to handle extensive map data, while the latter shows limited capability in understanding natural language preferences. Additionally, a more critical challenge arises from the highly heterogeneous and unpredictable spatio-temporal distribution of users across the globe. In this paper, we introduce a novel LLM-Assisted route Planning (LLMAP) system that employs an LLM-as-Parser to comprehend natural language, identify tasks, and extract user preferences and recognize task dependencies, coupled with a Multi-Step Graph construction with iterative Search (MSGS) algorithm as the underlying solver for optimal route finding. Our multi-objective optimization approach adaptively tunes objective weights to maximize points of interest (POI) quality and task completion rate while minimizing route distance, subject to three key constraints: user time limits, POI opening hours, and task dependencies. We conduct extensive experiments using 1,000 routing prompts sampled with varying complexity across 14 countries and 27 cities worldwide. The results demonstrate that our approach achieves superior performance with guarantees across multiple constraints. 

**Abstract (ZH)**: 大型语言模型的兴起使得基于自然语言的路线规划成为一个涵盖丰富用户目标的研究领域。当前研究展现出两种不同的方法：基于LLM代理的直接路线规划和基于图的搜索策略。然而，前者中的LLM在处理大量地图数据时表现出局限性，而后者的自然语言理解能力也受到限制。此外，来自全球用户高度异质性和不可预测的时空分布构成了更加关键的挑战。本文介绍了一种新型的LLM辅助路线规划（LLMAP）系统，该系统采用LLM作为语法解析器来理解自然语言、识别任务并提取用户偏好和任务依赖性，同时结合一个多步图构造与迭代搜索（MSGS）算法作为优化路径寻找的基础解决方法。我们的多目标优化方法自适应调整目标权重，以在满足三大关键约束条件（用户时间限制、POI营业时间及任务依赖性）的同时最大化兴趣点（POI）质量和任务完成率并最小化路线距离。我们使用来自14个国家和27个城市的1,000个不同复杂度的路线规划提示进行了广泛的实验。结果表明，我们的方法在多个约束条件下取得了卓越的性能。 

---
# Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors 

**Title (ZH)**: 元认知重用：将反复出现的LLM推理转化为简洁行为 

**Authors**: Aniket Didolkar, Nicolas Ballas, Sanjeev Arora, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2509.13237)  

**Abstract**: Large language models (LLMs) now solve multi-step problems by emitting extended chains of thought. During the process, they often re-derive the same intermediate steps across problems, inflating token usage and latency. This saturation of the context window leaves less capacity for exploration. We study a simple mechanism that converts recurring reasoning fragments into concise, reusable "behaviors" (name + instruction) via the model's own metacognitive analysis of prior traces. These behaviors are stored in a "behavior handbook" which supplies them to the model in-context at inference or distills them into parameters via supervised fine-tuning. This approach achieves improved test-time reasoning across three different settings - 1) Behavior-conditioned inference: Providing the LLM relevant behaviors in-context during reasoning reduces number of reasoning tokens by up to 46% while matching or improving baseline accuracy; 2) Behavior-guided self-improvement: Without any parameter updates, the model improves its own future reasoning by leveraging behaviors from its own past problem solving attempts. This yields up to 10% higher accuracy than a naive critique-and-revise baseline; and 3) Behavior-conditioned SFT: SFT on behavior-conditioned reasoning traces is more effective at converting non-reasoning models into reasoning models as compared to vanilla SFT. Together, these results indicate that turning slow derivations into fast procedural hints enables LLMs to remember how to reason, not just what to conclude. 

**Abstract (ZH)**: 大型语言模型通过生成扩展的思维链现在可以解决多步问题。在处理过程中，它们经常在不同的问题中重复推导相同的中间步骤，这会增加令牌使用量和延迟。这种对上下文窗口的饱和使得探索的空间变小。我们研究了一种简单的机制，通过模型自身的元认知分析将其之前的推理片段转换为简洁且可重用的“行为”（名称+指令）。这些行为被存储在“行为手册”中，在推理时上下文提供这些行为，或通过监督微调将其转换为参数。该方法在三种不同的设置下实现了推理性能的提升：1）行为条件下的推理：在推理过程中提供相关的行为可以将推理的令牌数量减少高达46%，同时匹配或提高基线准确率；2）行为引导的自我改进：无需更新参数，模型可以通过利用其自身过去问题解决尝试中的行为来改进未来的推理，这将其准确率提高了最高10%以上，超过了一种简单的批判和修订基线；3）行为条件下的SFT：基于行为条件下的推理轨迹的SFT比传统的SFT更有效地将非推理模型转化为推理模型。综上所述，这些结果表明，将慢速推导转变为快速的操作提示使大型语言模型能够记住如何推理，而不仅仅是记住结论。 

---
# Single-stream Policy Optimization 

**Title (ZH)**: 单流策略优化 

**Authors**: Zhongwen Xu, Zihan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.13232)  

**Abstract**: We revisit policy-gradient optimization for Large Language Models (LLMs) from a single-stream perspective. Prevailing group-based methods like GRPO reduce variance with on-the-fly baselines but suffer from critical flaws: frequent degenerate groups erase learning signals, and synchronization barriers hinder scalability. We introduce Single-stream Policy Optimization (SPO), which eliminates these issues by design. SPO replaces per-group baselines with a persistent, KL-adaptive value tracker and normalizes advantages globally across the batch, providing a stable, low-variance learning signal for every sample. Being group-free, SPO enables higher throughput and scales effectively in long-horizon or tool-integrated settings where generation times vary. Furthermore, the persistent value tracker naturally enables an adaptive curriculum via prioritized sampling. Experiments using Qwen3-8B show that SPO converges more smoothly and attains higher accuracy than GRPO, while eliminating computation wasted on degenerate groups. Ablation studies confirm that SPO's gains stem from its principled approach to baseline estimation and advantage normalization, offering a more robust and efficient path for LLM reasoning. Across five hard math benchmarks with Qwen3 8B, SPO improves the average maj@32 by +3.4 percentage points (pp) over GRPO, driven by substantial absolute point gains on challenging datasets, including +7.3 pp on BRUMO 25, +4.4 pp on AIME 25, +3.3 pp on HMMT 25, and achieves consistent relative gain in pass@$k$ across the evaluated $k$ values. SPO's success challenges the prevailing trend of adding incidental complexity to RL algorithms, highlighting a path where fundamental principles, not architectural workarounds, drive the next wave of progress in LLM reasoning. 

**Abstract (ZH)**: 我们从单流视角回顾大规模语言模型（LLMs）的策略梯度优化。主流基于组的方法如GRPO通过在线基准降低方差，但存在严重缺陷：频繁的退化组抹杀了学习信号，同步壁垒妨碍了扩展性。我们引入单流策略优化（SPO），通过设计消除这些问题。SPO用持久的、KL适应的价值追踪器替代组别基准，并在整个批量中全局归一化优势，为每个样本提供稳定、低方差的学习信号。SPO无组别限制，使其在长时序或集成工具的生成时间变化设置中具有更高的吞吐量和更好的扩展性。此外，持久的价值追踪器自然支持基于优先级采样的适应性课程。使用Qwen3-8B的实验表明，与GRPO相比，SPO收敛更平滑且 accuracy更高，同时消除了对退化组的计算浪费。消融研究证实，SPO的改进源自其对基准估算和优势归一化的原则性方法，提供了LLM推理更稳健和高效的道路。在Qwen3 8B的五个难关数学科目中，SPO将maj@32的平均值提高了3.4个百分点，主要受益于在具有挑战性的数据集上的绝对分值显著提升，包括BRUMO 25上的7.3个百分点，AIME 25上的4.4个百分点，HMMT 25上的3.3个百分点，并在评估的所有k值中实现了一致的相对增益。SPO的成功挑战了增加RL算法中偶然复杂性的主流趋势，突显出一条路径，即基本原则而非架构性变通方法将驱动LLM推理的下一次进展。 

---
# Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO 

**Title (ZH)**: 塑造解释：基于编码器唯一变换器的语义奖励模型用于近端策略优化 

**Authors**: Francesco Pappone, Ruggero Marino Lazzaroni, Federico Califano, Niccolò Gentile, Roberto Marras  

**Link**: [PDF](https://arxiv.org/pdf/2509.13081)  

**Abstract**: While Large Language Models (LLMs) excel at generating human-like text, aligning their outputs with complex, qualitative goals like pedagogical soundness remains a significant challenge. Standard reinforcement learning techniques often rely on slow and expensive LLM-as-a-judge evaluations or on brittle, keyword-based metrics like ROUGE, which fail to capture the semantic essence of a high-quality explanation. In this work, we introduce a novel approach to reward shaping within the Group Relative Policy Optimisation (GRPO) framework. Our central contribution is the use of a small, efficient encoder-only transformer as a semantic reward model. This model provides a dense, semantically rich reward signal based on the cosine similarity between a generated explanation and a ground-truth reference, guiding the policy towards explanations that are not just factually correct but also structurally and conceptually aligned with expert reasoning. We apply this method to the task of training a model for the Italian medical-school entrance examinations, following standard domain-adaptive continued pre-training (CPT) and supervised fine-tuning (SFT). Our results demonstrate that GRPO with our proposed semantic reward significantly improves explanation faithfulness and clarity over a strong SFT baseline, showcasing the power of using lightweight encoder models for nuanced reward shaping in complex generation tasks 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在生成_human-like_文本方面表现出色，但将其输出与复杂的定性目标，如教学合理性对齐依然是一项重大挑战。标准强化学习技术通常依赖于缓慢且昂贵的LLM作为评判者评估或脆弱的关键词基于度量，如ROUGE，这些度量无法捕捉高质量解释的语义核心。在这项工作中，我们引入了Group Relative Policy Optimisation（GRPO）框架内的一种新的奖励塑造方法。我们的主要贡献是使用一个小型高效的编码器变压器作为语义奖励模型。该模型基于生成的解释与地面真相参考之间的余弦相似度提供密集且语义丰富的奖励信号，指导策略趋向于那些不仅事实正确，而且在结构和概念上与专家推理对齐的解释。我们将其应用于训练用于意大利医学入学考试的模型任务，遵循标准领域适应连续预训练（CPT）和监督微调（SFT）。实验结果表明，与一个强大的SFT基线相比，我们提出的语义奖励下的GRPO显著提高了解释的忠实度和清晰度，展示了使用轻量级编码器模型进行复杂生成任务精细奖励塑造的强大功能。 

---
# Multi-Model Synthetic Training for Mission-Critical Small Language Models 

**Title (ZH)**: 面向关键任务的小型语言模型的多模型合成训练 

**Authors**: Nolan Platt, Pragyansmita Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2509.13047)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across many domains, yet their appli- cation to specialized fields remains constrained by the scarcity and complexity of domain-specific training data. We present a novel approach that achieves a 261x cost reduction for maritime intelligence by using LLMs as one-time teachers rather than using them directly for inference. Our method transforms 3.2 billion Automatic Identification System (AIS) vessel tracking records into 21,543 synthetic question and answer pairs through multi-model generation (GPT-4o and o3-mini), preventing over- fitting and ensuring accurate reasoning. The resulting fine-tuned Qwen2.5-7B model achieves 75% accuracy on maritime tasks, while being substantially cheaper than using a larger model for inference. We show that smaller, cheaper models - when fine tuned properly - can provide similar accuracy compared to larger models that are prohibitively expensive. Our work contributes to the growing field of synthetic dataset generation for specialized AI applications and presents a highly reproducible framework for domains where manual annotation is infeasible. Beyond expand- ing research in the growing field of specialized small language models, our approach has immediate applications in maritime safety, security operations, and vessel traffic management systems in various industries. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多个领域展现了卓越的能力，但将其应用于专业领域仍受限于领域特定训练数据的稀缺性和复杂性。我们提出了一种新颖的方法，通过将大规模语言模型用作一次性教师，实现了 maritime intelligence 领域成本降低261倍的效果，而非直接用于推理。该方法将32亿条自动识别系统（AIS）船舶跟踪记录转换为21,543对合成的问答对，防止过拟合并确保准确推理。由此优化的 Qwen2.5-7B 模型在 maritime 任务上的准确率达到75%，同时成本远低于使用更大模型进行推理的成本。我们证明，在适当微调的情况下，更小、更便宜的模型能够提供与昂贵的大模型相似的准确率。我们的工作为专门化 AI 应用的合成数据集生成领域做出了贡献，并提供了一种在手动标注不可行的领域中高度可复制的框架。除了在专门化小型语言模型研究领域扩展研究外，我们的方法还立即在海上安全、安全操作以及各类行业的船舶交通管理系统中找到了应用。 

---
# Validating Solidity Code Defects using Symbolic and Concrete Execution powered by Large Language Models 

**Title (ZH)**: 使用大型语言模型驱动的符号执行和实体执行验证Solidity代码缺陷 

**Authors**: Ştefan-Claudiu Susan, Andrei Arusoaie, Dorel Lucanu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13023)  

**Abstract**: The high rate of false alarms from static analysis tools and Large Language Models (LLMs) complicates vulnerability detection in Solidity Smart Contracts, demanding methods that can formally or empirically prove the presence of defects. This paper introduces a novel detection pipeline that integrates custom Slither-based detectors, LLMs, Kontrol, and Forge. Our approach is designed to reliably detect defects and generate proofs.  We currently perform experiments with promising results for seven types of critical defects. We demonstrate the pipeline's efficacy by presenting our findings for three vulnerabilities -- Reentrancy, Complex Fallback, and Faulty Access Control Policies -- that are challenging for current verification solutions, which often generate false alarms or fail to detect them entirely. We highlight the potential of either symbolic or concrete execution in correctly classifying such code faults. By chaining these instruments, our method effectively validates true positives, significantly reducing the manual verification burden. Although we identify potential limitations, such as the inconsistency and the cost of LLMs, our findings establish a robust framework for combining heuristic analysis with formal verification to achieve more reliable and automated smart contract auditing. 

**Abstract (ZH)**: 静态分析工具和大型语言模型（LLMs）高频率的误报 complicates Solidity 智能合约漏洞检测，需要能够形式化或经验性证明缺陷存在性的方法。本文介绍了一个结合自定义 Slither 基础检测器、LLMs、Kontrol 和 Forge 的新型检测流水线。我们的方法旨在可靠地检测缺陷并生成证明。我们目前对七种关键缺陷类型进行实验，结果显示前景乐观。我们通过展示针对重入、复杂回退和故障访问控制策略三种验证当前解决方案经常产生误报或完全检测不到的漏洞的研究结果，来证明该流水线的有效性。我们强调在正确分类此类代码故障方面，符号执行或具体执行的潜力。通过将这些工具串联起来，我们的方法有效地验证了真阳性，显著减轻了手动验证的负担。尽管我们识别出了一些潜在的限制，如 LLM 的不一致性和成本问题，但我们的发现确立了将启发式分析与形式化验证相结合的稳健框架，以实现更可靠和自动化的智能合约审计。 

---
# xOffense: An AI-driven autonomous penetration testing framework with offensive knowledge-enhanced LLMs and multi agent systems 

**Title (ZH)**: xOffense：一种基于AI驱动的自主渗透测试框架，结合了进攻性知识增强的大语言模型和多agent系统 

**Authors**: Phung Duc Luong, Le Tran Gia Bao, Nguyen Vu Khai Tam, Dong Huu Nguyen Khoa, Nguyen Huu Quyen, Van-Hau Pham, Phan The Duy  

**Link**: [PDF](https://arxiv.org/pdf/2509.13021)  

**Abstract**: This work introduces xOffense, an AI-driven, multi-agent penetration testing framework that shifts the process from labor-intensive, expert-driven manual efforts to fully automated, machine-executable workflows capable of scaling seamlessly with computational infrastructure. At its core, xOffense leverages a fine-tuned, mid-scale open-source LLM (Qwen3-32B) to drive reasoning and decision-making in penetration testing. The framework assigns specialized agents to reconnaissance, vulnerability scanning, and exploitation, with an orchestration layer ensuring seamless coordination across phases. Fine-tuning on Chain-of-Thought penetration testing data further enables the model to generate precise tool commands and perform consistent multi-step reasoning. We evaluate xOffense on two rigorous benchmarks: AutoPenBench and AI-Pentest-Benchmark. The results demonstrate that xOffense consistently outperforms contemporary methods, achieving a sub-task completion rate of 79.17%, decisively surpassing leading systems such as VulnBot and PentestGPT. These findings highlight the potential of domain-adapted mid-scale LLMs, when embedded within structured multi-agent orchestration, to deliver superior, cost-efficient, and reproducible solutions for autonomous penetration testing. 

**Abstract (ZH)**: 这种工作引入了xOffense，一种基于AI的多Agent渗透测试框架，将过程从劳动密集型、专家驱动的手动努力转变为全自动化、机器可执行的工作流，能够无缝扩展与计算基础设施。xOffense的核心在于应用一个 fine-tuned 的中型开源LLM（Qwen3-32B），驱动渗透测试中的推理与决策。该框架将专门的Agent分配给侦察、漏洞扫描和利用工作，并且有一个编排层确保各阶段之间的无缝协调。基于Chain-of-Thought渗透测试数据的 fine-tuning 进一步使模型能够生成精确的工具命令并执行一致的多步推理。我们在两个严格的基准上评估了xOffense：AutoPenBench和AI-Pentest-Benchmark。结果表明，xOffense一贯优于当代方法，子任务完成率为79.17%，显著超越VulnBot和PentestGPT等领先系统。这些发现强调了当嵌入到结构化的多Agent编排中时，领域适配的中型LLM的潜力，可以为自主渗透测试提供卓越、成本效益高且可再现的解决方案。 

---
# Bridging Performance Gaps for Foundation Models: A Post-Training Strategy for ECGFounder 

**Title (ZH)**: 基础模型性能差距的桥梁：ECGFounder 的后训练策略 

**Authors**: Ya Zhou, Yujie Yang, Xiaohan Fan, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.12991)  

**Abstract**: ECG foundation models are increasingly popular due to their adaptability across various tasks. However, their clinical applicability is often limited by performance gaps compared to task-specific models, even after pre-training on large ECG datasets and fine-tuning on target data. This limitation is likely due to the lack of an effective post-training strategy. In this paper, we propose a simple yet effective post-training approach to enhance ECGFounder, a state-of-the-art ECG foundation model pre-trained on over 7 million ECG recordings. Experiments on the PTB-XL benchmark show that our approach improves the baseline fine-tuning strategy by 1.2%-3.3% in macro AUROC and 5.3%-20.9% in macro AUPRC. Additionally, our method outperforms several recent state-of-the-art approaches, including task-specific and advanced architectures. Further evaluation reveals that our method is more stable and sample-efficient compared to the baseline, achieving a 9.1% improvement in macro AUROC and a 34.9% improvement in macro AUPRC using just 10% of the training data. Ablation studies identify key components, such as stochastic depth and preview linear probing, that contribute to the enhanced performance. These findings underscore the potential of post-training strategies to improve ECG foundation models, and we hope this work will contribute to the continued development of foundation models in the ECG domain. 

**Abstract (ZH)**: ECG基础模型由于其在各种任务上的适应性日益流行，但由于与任务特定模型相比的性能差距，其临床应用往往受到限制，即使在大规模ECG数据集上进行预训练并在目标数据上进行微调也是如此。这种限制可能源于缺乏有效的后训练策略。本文提出了一种简单而有效的后训练方法，以增强基于超过700万份ECG记录预训练的当前最先进的ECG基础模型ECGFounder。在PTB-XL基准上的实验表明，我们的方法在宏AUROC上改善了基线微调策略1.2%-3.3%，在宏AUPRC上改善了5.3%-20.9%。此外，我们的方法在多个最新最先进的方法中表现出色，包括任务特定和高级架构。进一步评估表明，与基线相比，我们的方法更稳定、样本效率更高，仅使用10%的训练数据就实现了宏AUROC 9.1%的改进和宏AUPRC 34.9%的改进。消融试验指出，随机深度和前瞻线性探测等关键组件为性能提升做出了贡献。这些发现强调了后训练策略有潜力改善ECG基础模型，我们希望这一工作能够促进ECG领域基础模型的持续发展。 

---
# Investigating ReLoRA: Effects on the Learning Dynamics of Small Language Models 

**Title (ZH)**: 探究ReLoRA对小型语言模型学习动力学的影响 

**Authors**: Yuval Weiss, David Demitri Africa, Paula Buttery, Richard Diehl Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.12960)  

**Abstract**: Parameter-efficient methods such as LoRA have revolutionised the fine-tuning of LLMs. Still, their extension to pretraining via ReLoRA is less well understood, especially for small language models (SLMs), which offer lower computational and environmental costs. This work is the first systematic study of ReLoRA in SLMs (11M-66M parameters), evaluating both performance and learning dynamics. Through ablation experiments, we find that ReLoRA generally performs worse than standard training on loss, Paloma perplexity and BLiMP, with the gap widening for the larger models. Further analysis of the learning dynamics of the models indicates that ReLoRA reinforces the rank deficiencies found in smaller models. These results indicate that low-rank update strategies may not transfer easily to SLM pretraining, highlighting the need for more research in the low-compute regime. 

**Abstract (ZH)**: ReLoRA在小语言模型预训练中的系统研究：性能与学习动力学分析 

---
# Jailbreaking Large Language Models Through Content Concretization 

**Title (ZH)**: 通过内容具体化突破大型语言模型限制 

**Authors**: Johan Wahréus, Ahmed Hussain, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2509.12937)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for task automation and content generation, yet their safety mechanisms remain vulnerable to circumvention through different jailbreaking techniques. In this paper, we introduce \textit{Content Concretization} (CC), a novel jailbreaking technique that iteratively transforms abstract malicious requests into concrete, executable implementations. CC is a two-stage process: first, generating initial LLM responses using lower-tier, less constrained safety filters models, then refining them through higher-tier models that process both the preliminary output and original prompt. We evaluate our technique using 350 cybersecurity-specific prompts, demonstrating substantial improvements in jailbreak Success Rates (SRs), increasing from 7\% (no refinements) to 62\% after three refinement iterations, while maintaining a cost of 7.5\textcent~per prompt. Comparative A/B testing across nine different LLM evaluators confirms that outputs from additional refinement steps are consistently rated as more malicious and technically superior. Moreover, manual code analysis reveals that generated outputs execute with minimal modification, although optimal deployment typically requires target-specific fine-tuning. With eventual improved harmful code generation, these results highlight critical vulnerabilities in current LLM safety frameworks. 

**Abstract (ZH)**: Content Concretization: A Novel Jailbreaking Technique for Large Language Models 

---
# All Roads Lead to Rome: Graph-Based Confidence Estimation for Large Language Model Reasoning 

**Title (ZH)**: 所有的道路通罗马：基于图的大型语言模型推理置信度估计 

**Authors**: Caiqi Zhang, Chang Shu, Ehsan Shareghi, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2509.12908)  

**Abstract**: Confidence estimation is essential for the reliable deployment of large language models (LLMs). Existing methods are primarily designed for factual QA tasks and often fail to generalize to reasoning tasks. To address this gap, we propose a set of training-free, graph-based confidence estimation methods tailored to reasoning tasks. Our approach models reasoning paths as directed graphs and estimates confidence by exploiting graph properties such as centrality, path convergence, and path weighting. Experiments with two LLMs on three reasoning datasets demonstrate improved confidence estimation and enhanced performance on two downstream tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）可靠部署中，置信度估计至关重要。现有的方法主要针对事实型QA任务，往往无法泛化到推理任务。为解决这一问题，我们提出了一种无需训练、基于图的置信度估计方法，专门适用于推理任务。我们的方法将推理路径建模为有向图，并通过利用图的中心性、路径收敛性和路径权重等性质来估计置信度。在两个LLM上对三个推理数据集进行的实验显示，该方法提高了置信度估计，并增强了两个下游任务的性能。 

---
# Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings 

**Title (ZH)**: Conan-Embedding-v2: 从头训练一个文本嵌入的大型语言模型 

**Authors**: Shiyu Li, Yang Tang, Ruijie Liu, Shi-Zhe Chen, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12892)  

**Abstract**: Large language models (LLMs) have recently demonstrated excellent performance in text embedding tasks. Previous work usually use LoRA to fine-tune existing LLMs, which are limited by the data and training gap between LLMs and embedding models. In this work, we introduce Conan-embedding-v2, a new 1.4B-parameter LLM trained from scratch and fine-tuned as a text embedder. First, we add news data and multilingual pairs for LLM pretraining to bridge the data gap. Based on this, we propose a cross-lingual retrieval dataset that enables the LLM to better integrate embeddings across different languages. Second, whereas LLMs use a causal mask with token-level loss, embedding models use a bidirectional mask with sentence-level loss. This training gap makes full fine-tuning less effective than LoRA. We introduce a soft-masking mechanism to gradually transition between these two types of masks, enabling the model to learn more comprehensive representations. Based on this, we propose a dynamic hard negative mining method that exposes the model to more difficult negative examples throughout the training process. Being intuitive and effective, with only approximately 1.4B parameters, Conan-embedding-v2 achieves SOTA performance on both the Massive Text Embedding Benchmark (MTEB) and Chinese MTEB (May 19, 2025). 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本嵌入任务中 recently 已经展示了出色的性能。以往的工作通常使用 LoRA 对现有 LLMs 进行微调，但受到 LLMs 和嵌入模型之间数据和训练差距的限制。在此工作中，我们引入了 Conan-embedding-v2，这是一种从零开始训练的新的 1.4B 参数 LLM，并作为文本嵌入器进行微调。首先，我们为 LLM 预训练增加新闻数据和多语言对以缩小数据差距。在此基础上，我们提出了一个跨语言检索数据集，使 LLM 更好地在不同语言中整合嵌入。其次，与 LLM 使用基于token的因果掩码和基于句子的损失相比，嵌入模型使用双向掩码和句子级别的损失。这种训练差距使得全面微调比 LoRA 更无效。我们引入了一种软掩码机制，逐渐过渡到这两种类型的掩码，使模型能够学到更全面的表示。在此基础上，我们提出了一种动态hard负样本挖掘方法，在整个训练过程中使模型接触到更多的困难负样本。凭借这种直观有效的机制，仅使用大约 1.4B 参数，Conan-embedding-v2 在 Massive Text Embedding Benchmark (MTEB) 和中文 MTEB（2025年5月19日）上均实现了 SOTA 性能。 

---
# The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations 

**Title (ZH)**: 已有的大语言模型知识：通过隐藏表示估计大语言模型感知的问答难度 

**Authors**: Yubo Zhu, Dongrui Liu, Zecheng Lin, Wei Tong, Sheng Zhong, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.12886)  

**Abstract**: Estimating the difficulty of input questions as perceived by large language models (LLMs) is essential for accurate performance evaluation and adaptive inference. Existing methods typically rely on repeated response sampling, auxiliary models, or fine-tuning the target model itself, which may incur substantial computational costs or compromise generality. In this paper, we propose a novel approach for difficulty estimation that leverages only the hidden representations produced by the target LLM. We model the token-level generation process as a Markov chain and define a value function to estimate the expected output quality given any hidden state. This allows for efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens. Extensive experiments across both textual and multimodal tasks demonstrate that our method consistently outperforms existing baselines in difficulty estimation. Moreover, we apply our difficulty estimates to guide adaptive reasoning strategies, including Self-Consistency, Best-of-N, and Self-Refine, achieving higher inference efficiency with fewer generated tokens. 

**Abstract (ZH)**: 基于目标大规模语言模型（LLM）隐藏表示估计输入问题难度的方法对于准确的性能评估和自适应推理至关重要。现有方法通常依赖重复响应采样、辅助模型或目标模型本身的微调，这可能会产生显著的计算成本或降低通用性。本文提出了一种新颖的难度估计方法，仅利用目标LLM产生的隐藏表示。我们将标记级生成过程建模为马尔可夫链，并定义一个值函数来估计在任何隐藏状态下预期的输出质量。这种方法仅基于初始隐藏状态即可实现高效且准确的难度估计，而无需生成任何输出标记。广泛的实验，涵盖文本和多模态任务，显示我们的方法在难度估计方面始终优于现有基线。此外，我们应用难度估计来引导自一致性、N-best以及自我完善等自适应推理策略，以较少的生成标记实现更高的推理效率。 

---
# Multi-Robot Task Planning for Multi-Object Retrieval Tasks with Distributed On-Site Knowledge via Large Language Models 

**Title (ZH)**: 基于分布式现场知识的多机器人任务规划及大型语言模型在多对象检索任务中的应用 

**Authors**: Kento Murata, Shoichi Hasegawa, Tomochika Ishikawa, Yoshinobu Hagiwara, Akira Taniguchi, Lotfi El Hafi, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12838)  

**Abstract**: It is crucial to efficiently execute instructions such as "Find an apple and a banana" or "Get ready for a field trip," which require searching for multiple objects or understanding context-dependent commands. This study addresses the challenging problem of determining which robot should be assigned to which part of a task when each robot possesses different situational on-site knowledge-specifically, spatial concepts learned from the area designated to it by the user. We propose a task planning framework that leverages large language models (LLMs) and spatial concepts to decompose natural language instructions into subtasks and allocate them to multiple robots. We designed a novel few-shot prompting strategy that enables LLMs to infer required objects from ambiguous commands and decompose them into appropriate subtasks. In our experiments, the proposed method achieved 47/50 successful assignments, outperforming random (28/50) and commonsense-based assignment (26/50). Furthermore, we conducted qualitative evaluations using two actual mobile manipulators. The results demonstrated that our framework could handle instructions, including those involving ad hoc categories such as "Get ready for a field trip," by successfully performing task decomposition, assignment, sequential planning, and execution. 

**Abstract (ZH)**: 高效执行“找一个苹果和一个香蕉”或“准备一次野外考察”等需要搜索多个对象或理解上下文命令的任务至关重要。本研究针对每个机器人具有不同现场情况知识（特别是用户指定区域的空间概念）时，分配机器人执行任务部分的具有挑战性问题进行了探讨。我们提出了一种任务规划框架，利用大型语言模型（LLMs）和空间概念将自然语言指令分解为子任务并分配给多个机器人。我们设计了一种新颖的少样本提示策略，使LLMs能够从模糊的命令中推断出所需对象并将其分解为合适的子任务。在我们的实验中，所提出的方法实现了47/50的成功分配，优于随机分配（28/50）和基于常识的分配（26/50）。此外，我们使用两台实际的移动 manipulator 进行了定性评估。结果表明，我们的框架能够处理包括“准备一次野外考察”在内的涉及临时类别指令的任务，成功完成任务分解、分配、顺序规划和执行。 

---
# LLM-Based Approach for Enhancing Maintainability of Automotive Architectures 

**Title (ZH)**: 基于LLM的方法提升汽车架构的可维护性 

**Authors**: Nenad Petrovic, Lukasz Mazur, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.12798)  

**Abstract**: There are many bottlenecks that decrease the flexibility of automotive systems, making their long-term maintenance, as well as updates and extensions in later lifecycle phases increasingly difficult, mainly due to long re-engineering, standardization, and compliance procedures, as well as heterogeneity and numerosity of devices and underlying software components involved. In this paper, we explore the potential of Large Language Models (LLMs) when it comes to the automation of tasks and processes that aim to increase the flexibility of automotive systems. Three case studies towards achieving this goal are considered as outcomes of early-stage research: 1) updates, hardware abstraction, and compliance, 2) interface compatibility checking, and 3) architecture modification suggestions. For proof-of-concept implementation, we rely on OpenAI's GPT-4o model. 

**Abstract (ZH)**: 许多瓶颈降低了汽车系统的灵活性，使得其长期维护以及后期生命周期阶段的更新和扩展日益困难，主要原因包括长时间的重新工程、标准化和合规程序，以及参与设备和底层软件组件的异构性和多样性。在本文中，我们探讨了大型语言模型（LLMs）在自动化旨在提高汽车系统灵活性的任务和流程方面的潜在应用。作为早期研究的结果，我们考虑了三个案例研究：1）更新、硬件抽象和合规性，2）接口兼容性检查，以及3）架构修改建议。为概念验证实现，我们依赖于OpenAI的GPT-4o模型。 

---
# InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering 

**Title (ZH)**: InfoGain-RAG: 基于文档信息增益重 ranking 和过滤的检索增强生成提升 

**Authors**: Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12765)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach to address key limitations of Large Language Models (LLMs), such as hallucination, outdated knowledge, and lacking reference. However, current RAG frameworks often struggle with identifying whether retrieved documents meaningfully contribute to answer generation. This shortcoming makes it difficult to filter out irrelevant or even misleading content, which notably impacts the final performance. In this paper, we propose Document Information Gain (DIG), a novel metric designed to quantify the contribution of retrieved documents to correct answer generation. DIG measures a document's value by computing the difference of LLM's generation confidence with and without the document augmented. Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to train a specialized reranker, which prioritizes each retrieved document from exact distinguishing and accurate sorting perspectives. This approach can effectively filter out irrelevant documents and select the most valuable ones for better answer generation. Extensive experiments across various models and benchmarks demonstrate that InfoGain-RAG can significantly outperform existing approaches, on both single and multiple retrievers paradigm. Specifically on NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG respectively, and even an average of 15.3% increment on advanced proprietary model GPT-4o across all datasets. These results demonstrate the feasibility of InfoGain-RAG as it can offer a reliable solution for RAG in multiple applications. 

**Abstract (ZH)**: Document Information Gain (DIG)增强的检索增强生成（InfoGain-RAG） 

---
# Instance-level Randomization: Toward More Stable LLM Evaluations 

**Title (ZH)**: 实例级随机化：迈向更稳定的LLM评估 

**Authors**: Yiyang Li, Yonghuang Wu, Ying Luo, Liangtai Sun, Zishu Qin, Lin Qiu, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.12678)  

**Abstract**: Evaluations of large language models (LLMs) suffer from instability, where small changes of random factors such as few-shot examples can lead to drastic fluctuations of scores and even model rankings. Moreover, different LLMs can have different preferences for a certain setting of random factors. As a result, using a fixed setting of random factors, which is often adopted as the paradigm of current evaluations, can lead to potential unfair comparisons between LLMs. To mitigate the volatility of evaluations, we first theoretically analyze the sources of variance induced by changes in random factors. Targeting these specific sources, we then propose the instance-level randomization (ILR) method to reduce variance and enhance fairness in model comparisons. Instead of using a fixed setting across the whole benchmark in a single experiment, we randomize all factors that affect evaluation scores for every single instance, run multiple experiments and report the averaged score. Theoretical analyses and empirical results demonstrate that ILR can reduce the variance and unfair comparisons caused by random factors, as well as achieve similar robustness level with less than half computational cost compared with previous methods. 

**Abstract (ZH)**: 大规模语言模型（LLMs）评估的不稳定性研究：基于实例级随机化的方法以减轻评估波动并提高模型比较的公平性 

---
# Don't Change My View: Ideological Bias Auditing in Large Language Models 

**Title (ZH)**: 不要改变我的观点：大型语言模型中的意识形态偏见审计 

**Authors**: Paul Kröger, Emilio Barkett  

**Link**: [PDF](https://arxiv.org/pdf/2509.12652)  

**Abstract**: As large language models (LLMs) become increasingly embedded in products used by millions, their outputs may influence individual beliefs and, cumulatively, shape public opinion. If the behavior of LLMs can be intentionally steered toward specific ideological positions, such as political or religious views, then those who control these systems could gain disproportionate influence over public discourse. Although it remains an open question whether LLMs can reliably be guided toward coherent ideological stances and whether such steering can be effectively prevented, a crucial first step is to develop methods for detecting when such steering attempts occur. In this work, we adapt a previously proposed statistical method to the new context of ideological bias auditing. Our approach carries over the model-agnostic design of the original framework, which does not require access to the internals of the language model. Instead, it identifies potential ideological steering by analyzing distributional shifts in model outputs across prompts that are thematically related to a chosen topic. This design makes the method particularly suitable for auditing proprietary black-box systems. We validate our approach through a series of experiments, demonstrating its practical applicability and its potential to support independent post hoc audits of LLM behavior. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在数百万用户的产品中越来越普及，它们的输出可能会影响个人信念，并累积性地塑造公众意见。如果可以故意引导LLMs的行为使其倾向于特定的政治或宗教观点，那么控制这些系统的人可能会在公众讨论中获得不成比例的影响。尽管目前尚不确定LLMs能否可靠地被引导至一致的政治立场，以及这种引导能否得到有效防止，但关键的第一步是开发用于检测此类引导尝试的方法。在本项工作中，我们针对意识形态偏见审计这一新情境，适应并改进了之前提出的一种统计方法。我们的方法保持了原始框架的模型无关设计，无需访问语言模型的内部结构，而是通过分析与选定主题相关主题的提示下模型输出的分布性变化来识别潜在的意识形态引导。这一设计使方法特别适合审计专有的黑盒系统。我们通过一系列实验验证了该方法的实际适用性及其支持独立的事后审计以监督LLM行为的潜力。 

---
# A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs 

**Title (ZH)**: 参数高效微调方法对代码LLM安全性的系统评估 

**Authors**: Kiho Lee, Jungkon Kim, Doowon Kim, Hyoungshick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.12649)  

**Abstract**: Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs. 

**Abstract (ZH)**: Code-generating 大型语言模型（LLMs）显著加速软件开发。然而，它们频繁生成不安全的代码带来了严重风险。我们对七种参数高效微调（PEFT）技术进行了全面评估，展示了在不牺牲功能性的前提下显著提高代码安全性。我们的研究表明，提示微调(Prompt-tuning)是最有效的PEFT方法，实现了CodeGen2 16B的80.86%的整体安全生成率，比基线67.28%提高了13.5个百分点。通过调整解码策略（优化采样温度）进一步将安全性提升至87.65%。这相当于每百万生成的代码片段中减少了约203,700个漏洞。此外，在我们的TrojanPuzzle评估中，提示和前缀微调提高了对CWE-79和CWE-502攻击向量的鲁棒性。我们的研究结果在Python和Java中具有普适性，证实了提示微调的一致有效性。本研究为使用LLMs构建更健壮的软件系统提供了必要的见解和实用指导。 

---
# ScaleDoc: Scaling LLM-based Predicates over Large Document Collections 

**Title (ZH)**: ScaleDoc：在大规模文档集合上扩展基于LLM的谓词 

**Authors**: Hengrui Zhang, Yulong Hui, Yihao Liu, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12610)  

**Abstract**: Predicates are foundational components in data analysis systems. However, modern workloads increasingly involve unstructured documents, which demands semantic understanding, beyond traditional value-based predicates. Given enormous documents and ad-hoc queries, while Large Language Models (LLMs) demonstrate powerful zero-shot capabilities, their high inference cost leads to unacceptable overhead. Therefore, we introduce \textsc{ScaleDoc}, a novel system that addresses this by decoupling predicate execution into an offline representation phase and an optimized online filtering phase. In the offline phase, \textsc{ScaleDoc} leverages a LLM to generate semantic representations for each document. Online, for each query, it trains a lightweight proxy model on these representations to filter the majority of documents, forwarding only the ambiguous cases to the LLM for final decision. Furthermore, \textsc{ScaleDoc} proposes two core innovations to achieve significant efficiency: (1) a contrastive-learning-based framework that trains the proxy model to generate reliable predicating decision scores; (2) an adaptive cascade mechanism that determines the effective filtering policy while meeting specific accuracy targets. Our evaluations across three datasets demonstrate that \textsc{ScaleDoc} achieves over a 2$\times$ end-to-end speedup and reduces expensive LLM invocations by up to 85\%, making large-scale semantic analysis practical and efficient. 

**Abstract (ZH)**: ScaleDoc：一种通过分阶段执行谓词来实现大规模语义分析的新型系统 

---
# FunAudio-ASR Technical Report 

**Title (ZH)**: FunAudio-ASR技术报告 

**Authors**: Keyu An, Yanni Chen, Chong Deng, Changfeng Gao, Zhifu Gao, Bo Gong, Xiangang Li, Yabin Li, Xiang Lv, Yunjie Ji, Yiheng Jiang, Bin Ma, Haoneng Luo, Chongjia Ni, Zexu Pan, Yiping Peng, Zhendong Peng, Peiyao Wang, Hao Wang, Wen Wang, Wupeng Wang, Biao Tian, Zhentao Tan, Nan Yang, Bin Yuan, Jieping Ye, Jixing Yu, Qinglin Zhang, Kun Zou, Han Zhao, Shengkui Zhao, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.12508)  

**Abstract**: In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present FunAudio-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, FunAudio-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, FunAudio-ASR achieves SOTA performance on real application datasets, demonstrating its effectiveness and robustness in practical settings. 

**Abstract (ZH)**: 近年来，自动语音识别（ASR）得益于数据规模、模型规模和与大规模语言模型（LLM）深度集成三种互补范式的推动，经历了变革性的进步。然而，LLM容易产生幻觉，这可能显著降低实际ASR应用中的用户体验。本文介绍了一种名为FunAudio-ASR的大规模、基于LLM的ASR系统，该系统结合了大量数据、大模型容量、LLM集成和强化学习，实现了多样化和复杂语音识别场景下的最先进性能。此外，FunAudio-ASR特别针对实际部署进行了优化，提升了流式传输能力、噪声鲁棒性、代码切换、热词定制以及其他实际应用需求。实验结果表明，虽然大多数基于LLM的ASR系统在开源基准上的表现强劲，但在实际工业评估集中常表现不佳。得益于面向生产的优化，FunAudio-ASR在实际应用数据集上达到了最先进性能，证明了其在实际应用中的有效性和鲁棒性。 

---
# Reinforcement Learning-Based Market Making as a Stochastic Control on Non-Stationary Limit Order Book Dynamics 

**Title (ZH)**: 基于强化学习的市场制作作为非平稳限价订单簿动态的随机控制 

**Authors**: Rafael Zimmer, Oswaldo Luiz do Valle Costa  

**Link**: [PDF](https://arxiv.org/pdf/2509.12456)  

**Abstract**: Reinforcement Learning has emerged as a promising framework for developing adaptive and data-driven strategies, enabling market makers to optimize decision-making policies based on interactions with the limit order book environment. This paper explores the integration of a reinforcement learning agent in a market-making context, where the underlying market dynamics have been explicitly modeled to capture observed stylized facts of real markets, including clustered order arrival times, non-stationary spreads and return drifts, stochastic order quantities and price volatility. These mechanisms aim to enhance stability of the resulting control agent, and serve to incorporate domain-specific knowledge into the agent policy learning process. Our contributions include a practical implementation of a market making agent based on the Proximal-Policy Optimization (PPO) algorithm, alongside a comparative evaluation of the agent's performance under varying market conditions via a simulator-based environment. As evidenced by our analysis of the financial return and risk metrics when compared to a closed-form optimal solution, our results suggest that the reinforcement learning agent can effectively be used under non-stationary market conditions, and that the proposed simulator-based environment can serve as a valuable tool for training and pre-training reinforcement learning agents in market-making scenarios. 

**Abstract (ZH)**: 强化学习作为一种有前途的框架，用于开发适应性和数据驱动的策略，使市场制作人能够基于与限价订单簿环境的交互来优化决策策略。本文探讨了在明确建模底层市场动态的市场制作背景下，强化学习代理的集成，这些动态旨在捕捉现实中市场观察到的典型事实，包括订单到达时间的聚类、非平稳价差和回报漂移、随机的订单量和价格波动性。这些机制旨在增强所得到的控制代理的稳定性，并将其领域特定知识纳入代理策略学习过程。我们的贡献包括基于Proximal-Policy Optimization (PPO)算法的实际市场制作代理实现，以及通过模拟环境在不同市场条件下对代理性能的比较评估。通过对与闭式最优解进行财务回报和风险指标分析，我们的结果显示，强化学习代理在非平稳市场条件下可以有效使用，并且所提出的基于模拟器的环境可以作为训练和预训练市场制作场景中的强化学习代理的宝贵工具。 

---
# MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts 

**Title (ZH)**: MedFact：评估大型语言模型在中国医疗文本事实核查能力的基准测试 

**Authors**: Jiayi He, Yangmin Huang, Qianyun Du, Xiangying Zhou, Zhiyang He, Jiaxue Hu, Xiaodong Tao, Lixian Lai  

**Link**: [PDF](https://arxiv.org/pdf/2509.12440)  

**Abstract**: The increasing deployment of Large Language Models (LLMs) in healthcare necessitates a rigorous evaluation of their factual reliability. However, existing benchmarks are often limited by narrow domains of data, failing to capture the complexity of real-world medical information. To address this critical gap, we introduce MedFact, a new and challenging benchmark for Chinese medical fact-checking. MedFact comprises 2,116 expert-annotated instances curated from diverse real-world texts, spanning 13 medical specialties, 8 fine-grained error types, 4 writing styles, and multiple difficulty levels. Its construction employs a hybrid AI-human framework where iterative expert feedback refines an AI-driven, multi-criteria filtering process, ensuring both high data quality and difficulty. We conduct a comprehensive evaluation of 20 leading LLMs, benchmarking their performance on veracity classification and error localization against a human expert baseline. Our results reveal that while models can often determine if a text contains an error, precisely localizing it remains a substantial challenge, with even top-performing models falling short of human performance. Furthermore, our analysis uncovers a frequent ``over-criticism'' phenomenon, a tendency for models to misidentify correct information as erroneous, which is exacerbated by advanced reasoning techniques such as multi-agent collaboration and inference-time scaling. By highlighting these critical challenges for deploying LLMs in medical applications, MedFact provides a robust resource to drive the development of more factually reliable and medically aware models. 

**Abstract (ZH)**: 大型语言模型在医疗领域的日益应用亟需对其事实可靠性进行严格的评估。现有的基准测试往往受限于数据领域的狭窄，无法捕捉到真实世界医疗信息的复杂性。为弥补这一关键gap，我们引入了MedFact，这是一个新的具有挑战性的中文医学事实核查基准。MedFact包含来自多元化真实世界文本的2,116个由专家标注的实例，涵盖了13个医学专科、8种精细错误类型、4种书写风格以及多个难度级别。其构建采用了混合AI-人类框架，通过迭代的专家反馈不断优化AI驱动的多标准筛选过程，确保数据质量和难度的同时兼具。我们对20个领先的大规模语言模型进行了全面评估，将其准确性和错误定位性能与人类专家基准进行了对比。结果显示，尽管模型能常判断一段文本中是否包含错误，但精确地定位错误依然是一个巨大的挑战，即使是表现最好的模型也无法达到人类的表现。此外，我们的分析揭示了一个常见的“过度批评”现象，模型倾向于错误地将正确信息识别为错误信息，这种现象在多代理协作和推理时扩展等高级推理技术中尤为严重。通过强调这些关键挑战，MedFact为推动开发更可靠和医学意识更强的大规模语言模型提供了坚实资源。 

---
# Evaluating Large Language Models for Functional and Maintainable Code in Industrial Settings: A Case Study at ASML 

**Title (ZH)**: 评估大型语言模型在工业环境中生成功能性可维护代码的能力：ASML案例研究 

**Authors**: Yash Mundhra, Max Valk, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12395)  

**Abstract**: Large language models have shown impressive performance in various domains, including code generation across diverse open-source domains. However, their applicability in proprietary industrial settings, where domain-specific constraints and code interdependencies are prevalent, remains largely unexplored. We present a case study conducted in collaboration with the leveling department at ASML to investigate the performance of LLMs in generating functional, maintainable code within a closed, highly specialized software environment.
We developed an evaluation framework tailored to ASML's proprietary codebase and introduced a new benchmark. Additionally, we proposed a new evaluation metric, build@k, to assess whether LLM-generated code successfully compiles and integrates within real industrial repositories. We investigate various prompting techniques, compare the performance of generic and code-specific LLMs, and examine the impact of model size on code generation capabilities, using both match-based and execution-based metrics. The findings reveal that prompting techniques and model size have a significant impact on output quality, with few-shot and chain-of-thought prompting yielding the highest build success rates. The difference in performance between the code-specific LLMs and generic LLMs was less pronounced and varied substantially across different model families. 

**Abstract (ZH)**: 大型语言模型在各类领域展现了 impressive 的性能，包括跨不同开源领域的代码生成。然而，在存在特定领域约束和代码依赖关系的专有工业环境中，它们的应用仍然鲜有探索。我们与 ASML 的平滑部门合作，进行了一项案例研究，旨在调查 LLM 生成可维护功能代码性能的情况，特别是在一个封闭且高度专业化的软件环境中。

我们为 ASML 的专有代码库开发了一套评估框架，并引入了一个新的基准。此外，我们提出了一种新的评估指标 build@k，以评估 LLM 生成的代码是否能够成功编译并整合到实际的工业代码库中。我们调查了各种提示技术，比较了通用和代码特定的 LLM 的性能，并通过基于匹配和基于执行的指标考察了模型规模对代码生成能力的影响。研究结果表明，提示技术和模型规模对输出质量有显著影响，单步提示和逐步思考提示能获得最高的编译成功率。代码特定的 LLM 和通用 LLM 之间的性能差异较小，且在不同模型系列中表现差异显著。 

---
# MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables 

**Title (ZH)**: MORABLES：评估大型语言模型寓言中道德推理能力的标准样本 

**Authors**: Matteo Marcuzzo, Alessandro Zangari, Andrea Albarelli, Jose Camacho-Collados, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2509.12371)  

**Abstract**: As LLMs excel on standard reading comprehension benchmarks, attention is shifting toward evaluating their capacity for complex abstract reasoning and inference. Literature-based benchmarks, with their rich narrative and moral depth, provide a compelling framework for evaluating such deeper comprehension skills. Here, we present MORABLES, a human-verified benchmark built from fables and short stories drawn from historical literature. The main task is structured as multiple-choice questions targeting moral inference, with carefully crafted distractors that challenge models to go beyond shallow, extractive question answering. To further stress-test model robustness, we introduce adversarial variants designed to surface LLM vulnerabilities and shortcuts due to issues such as data contamination. Our findings show that, while larger models outperform smaller ones, they remain susceptible to adversarial manipulation and often rely on superficial patterns rather than true moral reasoning. This brittleness results in significant self-contradiction, with the best models refuting their own answers in roughly 20% of cases depending on the framing of the moral choice. Interestingly, reasoning-enhanced models fail to bridge this gap, suggesting that scale - not reasoning ability - is the primary driver of performance. 

**Abstract (ZH)**: 随着大语言模型在标准阅读理解基准测试中表现出色，注意力开始转向评估其进行复杂抽象推理和推断的能力。基于文学的基准，由于其丰富的叙事和道德深度，为评估这种更深层次的理解能力提供了有力框架。在这里，我们介绍了MORABLES，一个基于历史文学中寓言和短故事构建的人工智能验证基准。主要任务结构化为针对道德推理的多项选择题，精心设计的干扰选项挑战模型超越浅层的提取性问答。为了进一步测试模型的 robustness，我们引入了对抗性变体，旨在揭示由于数据污染等问题导致的LLM的漏洞和捷径。我们的研究发现，虽然更大的模型表现优于较小的模型，但它们仍然容易受到对抗性操纵的影响，往往依赖于表面模式而非真正的道德推理。这种脆弱性导致了显著的自我矛盾，最佳模型在约20%的情况下反驳其自身的答案，这取决于道德选择的表述方式。有趣的是，增强推理能力的模型未能弥合这一差距，表明规模而非推理能力是性能的主要驱动力。 

---
# Humor in Pixels: Benchmarking Large Multimodal Models Understanding of Online Comics 

**Title (ZH)**: 像素中的 humor：大型多模态模型对在线漫画的理解基准 

**Authors**: Yuriel Ryan, Rui Yang Tan, Kenny Tsu Wei Choo, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.12248)  

**Abstract**: Understanding humor is a core aspect of social intelligence, yet it remains a significant challenge for Large Multimodal Models (LMMs). We introduce PixelHumor, a benchmark dataset of 2,800 annotated multi-panel comics designed to evaluate LMMs' ability to interpret multimodal humor and recognize narrative sequences. Experiments with state-of-the-art LMMs reveal substantial gaps: for instance, top models achieve only 61% accuracy in panel sequencing, far below human performance. This underscores critical limitations in current models' integration of visual and textual cues for coherent narrative and humor understanding. By providing a rigorous framework for evaluating multimodal contextual and narrative reasoning, PixelHumor aims to drive the development of LMMs that better engage in natural, socially aware interactions. 

**Abstract (ZH)**: 理解幽默是社会智力的核心方面，但依然是大型多模态模型（LMMs）的一个重大挑战。我们介绍了PixelHumor，一个包含2800个注解的多格漫画基准数据集，旨在评估LMMs在解读多模态幽默和识别叙事序列方面的能力。使用最新LMMs的实验揭示了显著的差距：例如，顶级模型在格序排列方面的准确率仅为61%，远低于人类的表现。这凸显了当前模型在整合视觉和文本线索以进行连贯的叙事和幽默理解方面的关键局限性。通过提供一个严格的框架来评估多模态上下文和叙事推理，PixelHumor旨在推动开发更能进行自然、社会意识强的交互的LMMs。 

---
# RL Fine-Tuning Heals OOD Forgetting in SFT 

**Title (ZH)**: RL微调修复SFT中的OOD遗忘 

**Authors**: Hangzhan Jin, Sitao Luan, Sicheng Lyu, Guillaume Rabusseau, Reihaneh Rabbany, Doina Precup, Mohammad Hamdaqa  

**Link**: [PDF](https://arxiv.org/pdf/2509.12235)  

**Abstract**: The two-stage fine-tuning paradigm of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) has empirically shown better reasoning performance than one-stage SFT for the post-training of Large Language Models (LLMs). However, the evolution and mechanism behind the synergy of SFT and RL are still under-explored and inconclusive. In our study, we find the well-known claim "SFT memorizes, RL generalizes" is over-simplified, and discover that: (1) OOD performance peaks at the early stage of SFT and then declines (OOD forgetting), the best SFT checkpoint cannot be captured by training/test loss; (2) the subsequent RL stage does not generate fundamentally better OOD capability, instead it plays an \textbf{OOD restoration} role, recovering the lost reasoning ability during SFT; (3) The recovery ability has boundaries, \ie{} \textbf{if SFT trains for too short or too long, RL cannot recover the lost OOD ability;} (4) To uncover the underlying mechanisms behind the forgetting and restoration process, we employ SVD analysis on parameter matrices, manually edit them, and observe their impacts on model performance. Unlike the common belief that the shift of model capacity mainly results from the changes of singular values, we find that they are actually quite stable throughout fine-tuning. Instead, the OOD behavior strongly correlates with the \textbf{rotation of singular vectors}. Our findings re-identify the roles of SFT and RL in the two-stage fine-tuning and discover the rotation of singular vectors as the key mechanism. %reversing the rotations induced by SFT, which shows recovery from forgetting, whereas imposing the SFT parameter directions onto a RL-tuned model results in performance degradation. Code is available at this https URL 

**Abstract (ZH)**: 监督微调（SFT）后跟随强化学习（RL）的两阶段微调范式在大型语言模型（LLMs）的后训练中表现出更好的推理性能，但SFT和RL之间协同作用的演变及其机制仍然有待进一步探索。在我们的研究中，我们发现广为人知的断言“SFT记忆，RL泛化”过于简化，并发现：（1）OOD性能在SFT的早期阶段达到峰值然后下降（OOD遗忘），最佳SFT检查点无法通过训练/测试损失捕获；（2）后续的RL阶段并没有产生根本上更好的OOD能力，而是起着OOD恢复作用，恢复SFT过程中失去的推理能力；（3）这种恢复能力有界限，即如果SFT训练时间过短或过长，RL无法恢复失去的OOD能力；（4）为了揭示遗忘和恢复过程背后的机制，我们对参数矩阵进行了SVD分析，手动编辑它们，并观察其对模型性能的影响。我们发现，模型能力的变化主要源自奇异值的变化说法并不准确，实际上，奇异值在整个微调过程中相当稳定，而OOD行为与奇异向量的旋转密切相关。我们的发现重新界定了SFT和RL在两阶段微调中的角色，并发现了奇异向量旋转作为关键机制。 

---
# Towards Trustworthy Agentic IoEV: AI Agents for Explainable Cyberthreat Mitigation and State Analytics 

**Title (ZH)**: 面向可信赖的代理IoEV：可解释的网络威胁缓解与状态分析的AI代理 

**Authors**: Meryem Malak Dif, Mouhamed Amine Bouchiha, Abdelaziz Amara Korba, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2509.12233)  

**Abstract**: The Internet of Electric Vehicles (IoEV) envisions a tightly coupled ecosystem of electric vehicles (EVs), charging infrastructure, and grid services, yet it remains vulnerable to cyberattacks, unreliable battery-state predictions, and opaque decision processes that erode trust and performance. To address these challenges, we introduce a novel Agentic Artificial Intelligence (AAI) framework tailored for IoEV, where specialized agents collaborate to deliver autonomous threat mitigation, robust analytics, and interpretable decision support. Specifically, we design an AAI architecture comprising dedicated agents for cyber-threat detection and response at charging stations, real-time State of Charge (SoC) estimation, and State of Health (SoH) anomaly detection, all coordinated through a shared, explainable reasoning layer; develop interpretable threat-mitigation mechanisms that proactively identify and neutralize attacks on both physical charging points and learning components; propose resilient SoC and SoH models that leverage continuous and adversarial-aware learning to produce accurate, uncertainty-aware forecasts with human-readable explanations; and implement a three-agent pipeline, where each agent uses LLM-driven reasoning and dynamic tool invocation to interpret intent, contextualize tasks, and execute formal optimizations for user-centric assistance. Finally, we validate our framework through comprehensive experiments across diverse IoEV scenarios, demonstrating significant improvements in security and prediction accuracy. All datasets, models, and code will be released publicly. 

**Abstract (ZH)**: 面向电动汽车的互联网（IoEV）设想了一个紧密耦合的生态系统，包括电动汽车、充电基础设施和电网服务，但仍面临网络攻击、电池状态预测不可靠和透明度低等问题，这些问题削弱了信任和性能。为了解决这些挑战，我们提出了一种专为IoEV设计的新颖代理人工智能（AAI）框架，其中专门的代理协作以实现自主威胁缓解、稳健分析和可解释的决策支持。具体内容包括：设计了一个AAI架构，其中包括用于充电站的网络威胁检测与响应、实时荷电状态（SoC）估计以及健康状态（SoH）异常检测的专用代理，所有这些都通过一个共享的、可解释的推理层进行协调；开发了可解释的威胁缓解机制，能够主动识别并消除对物理充电点和学习组件的攻击；提出了基于连续和对抗性学习的健壮SoC和SoH模型，以生成准确且具有不确定性意识的预测，并配有易于理解的解释；实施了一个三代理管道，每个代理利用基于大语言模型的推理和动态工具调用来解释意图、对任务进行情境化，并执行用户为中心的正式优化。最后，我们通过跨多种IoEV场景的全面实验验证了该框架，证明了在安全性和预测准确性方面取得了显著改进。所有数据集、模型和代码将公开发布。 

---
# Profiling LoRA/QLoRA Fine-Tuning Efficiency on Consumer GPUs: An RTX 4060 Case Study 

**Title (ZH)**: 基于RTX 4060的LoRA/QLoRA微调效率 profiling 研究 

**Authors**: MSR Avinash  

**Link**: [PDF](https://arxiv.org/pdf/2509.12229)  

**Abstract**: Fine-tuning large language models (LLMs) with parameter-efficient techniques such as LoRA and QLoRA has enabled adaptation of foundation models on modest hardware. Yet the efficiency of such training on consumer-grade GPUs, especially under strict 8 GB VRAM limits, remains underexplored. We present a controlled profiling study of LoRA/QLoRA fine-tuning using the Qwen2.5-1.5B-Instruct model on a single NVIDIA RTX 4060. Across three representative configurations, we systematically vary batch size, sequence length, optimizer choice (AdamW vs. PagedAdamW), and precision (fp16 vs. bf16). We report throughput (tokens/s), time per 10k tokens, and VRAM footprint, alongside energy estimates derived from GPU board power limits. Our results show that paged optimizers improve throughput by up to 25% (628 tok/s vs. 500 tok/s baseline), while bf16 degrades efficiency relative to fp16. Despite 8 GB constraints, sequence lengths up to 2048 tokens were feasible using parameter-efficient strategies. To our knowledge, this is the first systematic case study of LLM fine- tuning efficiency on consumer GPUs, providing reproducible benchmarks and practical guidelines for resource-constrained researchers and practitioners. 

**Abstract (ZH)**: 使用LoRA和QLoRA等参数高效技术微调大规模语言模型（LLMs）已在有限硬件上实现了基础模型的适应性。然而，此类训练在消费级GPU上的效率，尤其是在严格的8 GB VRAM限制下，仍待进一步探索。我们使用Qwen2.5-1.5B-Instruct模型在单块NVIDIA RTX 4060上进行了控制性性能分析，系统地变化了批量大小、序列长度、优化器选择（AdamW vs. PagedAdamW）以及精度（fp16 vs. bf16）。我们报告了吞吐量（tokens/s）、每10000个tokens所需时间及VRAM占用，并根据GPU板卡功率限制推导了能源估算。我们的结果显示，分页优化器可将吞吐量提高25%（达628 tok/s，基线为500 tok/s），而bf16相比fp16降低了效率。尽管受到8 GB限制，参数高效策略使序列长度高达2048 tokens成为可能。据我们所知，这是首次对消费级GPU上LLM微调效率的系统性研究，为资源受限的研究人员和实践者提供了可重复的基准和实用指南。 

---
# MEUV: Achieving Fine-Grained Capability Activation in Large Language Models via Mutually Exclusive Unlock Vectors 

**Title (ZH)**: MEUV: 通过互斥解锁向量在大型语言模型中实现细粒度的能力激活 

**Authors**: Xin Tong, Zhi Lin, Jingya Wang, Meng Han, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.12221)  

**Abstract**: Large language models (LLMs) enforce safety alignment to reliably refuse malicious requests, yet the same blanket safeguards also block legitimate uses in policing, defense, and other high-stakes settings. Earlier "refusal-direction" edits can bypass those layers, but they rely on a single vector that indiscriminately unlocks all hazardous topics, offering no semantic control. We introduce Mutually Exclusive Unlock Vectors (MEUV), a lightweight framework that factorizes the monolithic refusal direction into topic-aligned, nearly orthogonal vectors, each dedicated to one sensitive capability. MEUV is learned in a single epoch with a multi-task objective that blends a differential-ablation margin, cross-topic and orthogonality penalties, and several auxiliary terms. On bilingual malicious-prompt benchmarks, MEUV achieves an attack success rate of no less than 87% on Gemma-2-2B, LLaMA-3-8B, and Qwen-7B, yet cuts cross-topic leakage by up to 90% compared with the best single-direction baseline. Vectors trained in Chinese transfer almost unchanged to English (and vice versa), suggesting a language-agnostic refusal subspace. The results show that fine-grained, topic-level capability activation is achievable with minimal utility loss, paving the way for controlled LLMs deployment in security-sensitive domains. 

**Abstract (ZH)**: 大型语言模型中的互斥解锁向量（MEUV）：细粒度的主题级能力激活 

---
# TinyServe: Query-Aware Cache Selection for Efficient LLM Serving 

**Title (ZH)**: TinyServe: 查询aware的缓存选择以实现高效的LLM服务 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12211)  

**Abstract**: Serving large language models (LLMs) efficiently remains challenging due to the high memory and latency overhead of key-value (KV) cache access during autoregressive decoding. We present \textbf{TinyServe}, a lightweight and extensible serving system for deploying tiny LLMs (e.g., TinyLLaMA, GPT2-345M) with support for structured KV sparsity, plugin-based token selection, and hardware-efficient attention kernels. Unlike prior simulation frameworks, TinyServe executes real-time decoding with configurable sparsity strategies and fine-grained instrumentation.
To reduce decoding cost, we introduce a \textit{query-aware page selection} mechanism that leverages bounding-box metadata to estimate attention relevance between the query and KV cache blocks. This enables selective KV loading with minimal overhead and no model modifications. Our fused CUDA kernel integrates page scoring, sparse memory access, and masked attention in a single pass.
Experiments show that TinyServe achieves up to \textbf{3.4x} speedup and over \textbf{2x} memory savings with negligible accuracy drop. Additional analysis of cache reuse, page hit rate, and multi-GPU scaling confirms its practicality as an efficient system-level design for LLM training and inference research on resource-constrained hardware. 

**Abstract (ZH)**: TinyServe：一种轻量级且可扩展的部署小型语言模型的服务系统 

---
