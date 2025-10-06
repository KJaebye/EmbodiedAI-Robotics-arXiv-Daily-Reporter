# A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps 

**Title (ZH)**: 一种基于大尺度网格地图路径规划的LLM增强算法，速度提升1000倍 

**Authors**: Junlin Zeng, Xin Zhang, Xiang Zhao, Yan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02716)  

**Abstract**: Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.} 

**Abstract (ZH)**: 基于网格地图的路径规划因各种应用而备受关注。现有方法如A*和迪杰斯特拉算法及其变种在小型地图上表现良好，但在大型地图上由于搜索时间和内存消耗高而失效。最近，大型语言模型（LLMs）在路径规划方面显示出出色的性能，但仍存在空间错觉和规划性能不佳的问题。在所有相关工作中，LLM-A*利用LLM生成一系列航点，然后使用A*规划相邻航点之间的路径。通过这种方式，完整路径被构建出来。然而，LLM-A*在大型地图上仍存在较高的计算时间开销。为填补这一空白，我们对LLM-A*进行了深入研究，并发现其瓶颈，导致性能受限。据此，我们设计了一个创新的LLM增强算法，简称为iLLM-A*。iLLM-A*包括3个精心设计的机制，包括A*的优化、增量学习方法生成高质量航点以及选择适合作为A*路径规划的航点。最后，在各种网格地图上的综合评估显示，与LLM-A*相比，iLLM-A*在平均情况下实现超1000倍的加速，极端情况下甚至可达2349.5倍，节省高达58.6%的内存成本，并实现路径长度更短且路径长度标准差更低的效果。 

---
# Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner 

**Title (ZH)**: 共进化连续离散扩散：使你的扩散语言模型成为潜在推理器 

**Authors**: Cai Zhou, Chenxiao Yang, Yi Hu, Chenyu Wang, Chubin Zhang, Muhan Zhang, Lester Mackey, Tommi Jaakkola, Stephen Bates, Dinghuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03206)  

**Abstract**: Diffusion language models, especially masked discrete diffusion models, have achieved great success recently. While there are some theoretical and primary empirical results showing the advantages of latent reasoning with looped transformers or continuous chain-of-thoughts, continuous diffusion models typically underperform their discrete counterparts. In this paper, we argue that diffusion language models do not necessarily need to be in the discrete space. In particular, we prove that continuous diffusion models have stronger expressivity than discrete diffusions and looped transformers. We attribute the contradiction between the theoretical expressiveness and empirical performance to their practical trainability: while continuous diffusion provides intermediate supervision that looped transformers lack, they introduce additional difficulty decoding tokens into the discrete token space from the continuous representation space. We therefore propose Coevolutionary Continuous Discrete Diffusion (CCDD), which defines a joint multimodal diffusion process on the union of a continuous representation space and a discrete token space, leveraging a single model to simultaneously denoise in the joint space. By combining two modalities, CCDD is expressive with rich semantics in the latent space, as well as good trainability and sample quality with the help of explicit discrete tokens. We also propose effective architectures and advanced training/sampling techniques for CCDD, which reveals strong empirical performance in extensive language modeling experiments on real-world tasks. 

**Abstract (ZH)**: 连续与离散扩散语言模型：Coevolutionary Continuous Discrete Diffusion（CCDD） 

---
# CoDA: Agentic Systems for Collaborative Data Visualization 

**Title (ZH)**: CoDA: 为协作数据可视化设计的自主系统 

**Authors**: Zichen Chen, Jiefeng Chen, Sercan Ö. Arik, Misha Sra, Tomas Pfister, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03194)  

**Abstract**: Deep research has revolutionized data analysis, yet data scientists still devote substantial time to manually crafting visualizations, highlighting the need for robust automation from natural language queries. However, current systems struggle with complex datasets containing multiple files and iterative refinement. Existing approaches, including simple single- or multi-agent systems, often oversimplify the task, focusing on initial query parsing while failing to robustly manage data complexity, code errors, or final visualization quality. In this paper, we reframe this challenge as a collaborative multi-agent problem. We introduce CoDA, a multi-agent system that employs specialized LLM agents for metadata analysis, task planning, code generation, and self-reflection. We formalize this pipeline, demonstrating how metadata-focused analysis bypasses token limits and quality-driven refinement ensures robustness. Extensive evaluations show CoDA achieves substantial gains in the overall score, outperforming competitive baselines by up to 41.5%. This work demonstrates that the future of visualization automation lies not in isolated code generation but in integrated, collaborative agentic workflows. 

**Abstract (ZH)**: 深度研究已革新了数据分析，但仍需要大量时间由数据科学家手动构建可视化，这突显了从自然语言查询实现稳健自动化的重要性。然而，当前系统在处理包含多个文件的复杂数据集和迭代优化时仍显不足。现有方法，包括简单的单智能体或多功能智能体系统，往往简化问题，侧重于初始查询解析，而在管理数据复杂性、代码错误或最终可视化质量方面不具备鲁棒性。在本文中，我们将这一挑战重新定义为协作多智能体问题。我们引入了CoDA，一个采用专门LLM智能体进行元数据分析、任务规划、代码生成和自我反思的多智能体系统。我们形式化了这一流程，证明了以元数据为中心的分析可以绕过令牌限制，而质量驱动的优化保证了系统的稳健性。广泛的评估显示，CoDA在总体评分上取得了显著提升，优于竞争基线高达41.5%。这项工作证明了可视化自动化的未来在于集成、协作的代理工作流，而非孤立的代码生成。 

---
# Reward Model Routing in Alignment 

**Title (ZH)**: 对齐中的奖励模型路由 

**Authors**: Xinle Wu, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02850)  

**Abstract**: Reinforcement learning from human or AI feedback (RLHF / RLAIF) has become the standard paradigm for aligning large language models (LLMs). However, most pipelines rely on a single reward model (RM), limiting alignment quality and risking overfitting. Recent work explores RM routing--dynamically selecting an RM from a candidate pool to exploit complementary strengths while maintaining $O(1)$ RM calls--but existing methods suffer from cold-start and insufficient exploration. We propose BayesianRouter, a hybrid routing framework that combines offline RM strengths learning with online Bayesian selection. In the offline stage, a multi-task router is trained on preference data to estimate per-RM reliability. In the online stage, a Bayesian Thompson sampling router performs per-query RM selection, initializing RM-specific weight vectors with offline embeddings as Gaussian priors and adaptively updating their posteriors with online rewards to adapt to the evolving policy distribution. Extensive experiments on instruction-following (AlpacaEval-2, Arena-Hard, MT-Bench) and reasoning (GSM8K, MMLU) benchmarks show that BayesianRouter consistently outperforms individual RMs, RM ensembling, and existing routing methods. 

**Abstract (ZH)**: 基于人类或AI反馈的强化学习（RLHF/RLAIF）已成为对大语言模型（LLMs）进行对齐的标准范式。然而，大多数流程依赖于单一的奖励模型（RM），限制了对齐质量并存在过拟合风险。近期工作探索了RM路由——动态从候选池中选择RM以利用互补优势同时保持$O(1)$的RM调用次数，但现有方法存在冷启动和探索不足的问题。我们提出BayesianRouter，这是一种结合离线RM优势学习与在线贝叶斯选择的混合路由框架。在离线阶段，一个多任务路由器在偏好数据上进行训练以估计每种RM的可靠性。在在线阶段，一个贝叶斯Thompson抽样路由器执行每查询的RM选择，使用离线嵌入作为高斯先验初始化RM特异性权重向量，并根据在线奖励自适应更新后验以适应政策分布的变化。广泛的实验结果表明，BayesianRouter在指令遵循（AlpacaEval-2、Arena-Hard、MT-Bench）和推理（GSM8K、MMLU）基准测试中均优于单一RM、RM集成及现有路由方法。 

---
# Take Goodhart Seriously: Principled Limit on General-Purpose AI Optimization 

**Title (ZH)**: 认真对待古德哈特法则：通用人工智能优化的基本限制 

**Authors**: Antoine Maier, Aude Maier, Tom David  

**Link**: [PDF](https://arxiv.org/pdf/2510.02840)  

**Abstract**: A common but rarely examined assumption in machine learning is that training yields models that actually satisfy their specified objective function. We call this the Objective Satisfaction Assumption (OSA). Although deviations from OSA are acknowledged, their implications are overlooked. We argue, in a learning-paradigm-agnostic framework, that OSA fails in realistic conditions: approximation, estimation, and optimization errors guarantee systematic deviations from the intended objective, regardless of the quality of its specification. Beyond these technical limitations, perfectly capturing and translating the developer's intent, such as alignment with human preferences, into a formal objective is practically impossible, making misspecification inevitable. Building on recent mathematical results, absent a mathematical characterization of these gaps, they are indistinguishable from those that collapse into Goodhart's law failure modes under strong optimization pressure. Because the Goodhart breaking point cannot be located ex ante, a principled limit on the optimization of General-Purpose AI systems is necessary. Absent such a limit, continued optimization is liable to push systems into predictable and irreversible loss of control. 

**Abstract (ZH)**: 机器学习中的一个常见但鲜少检验的假设是训练会产生实际满足其指定目标函数的模型。我们称之为目标满足假设（OSA）。尽管OSA的偏差被承认，但其影响却被忽视。我们从一个学习范式无关的框架出发，论证在现实条件下OSA会失效：约化误差、估计误差和优化误差保证了系统会系统地偏离预期目标，不论其规定质量如何。除了这些技术限制，将开发者的意图，如与人类偏好的一致性，完美地捕捉并形式化为一个目标在实践中是不可能的，因此错配不可避免。基于近期数学成果，缺乏这些差距的数学刻画，它们在强烈的优化压力下会不可区分地变成Goodhart定律失效模式。由于Goodhart breaking点无法在事前确定，对通用人工智能系统的优化必须有原则性的限制。否则，持续的优化可能会将系统推向可预测且不可逆的失控状态。 

---
# Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents 

**Title (ZH)**: 超越最终答案：评估工具增强代理的推理轨迹 

**Authors**: Wonjoong Kim, Sangwu Park, Yeonjun In, Sein Kim, Dongha Lee, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.02837)  

**Abstract**: Although recent tool-augmented benchmarks incorporate complex user requests and diverse tools, the evaluation methods for most of them remain limited to answer matching. However, as the number of steps required to resolve a user request increases, a proper evaluation of an agent's performance must go beyond the final answer to also assess the problem-solving trajectory, including previously ignored aspects such as efficiency, hallucination, and adaptivity. The most straightforward method for evaluating these aspects is to compare an agent's trajectory with the ground-truth trajectory, but this approach is fundamentally limited since annotating all valid ground-truth trajectories is prohibitively expensive. However, a simple LLM-based evaluator struggles to assess trajectories in detail without ground truth. To effectively evaluate the agents in this manner, we introduce TRACE, a framework for the multi-dimensional evaluation of tool-augmented LLM agent performance. By incorporating an evidence bank, which accumulates knowledge gathered from preceding reasoning steps, TRACE enables a multi-faceted analysis and evaluation of an agent's reasoning trajectory effectively. To validate our framework, we develop a new meta-evaluation dataset by augmenting existing benchmarks with diverse and flawed trajectories, each labeled with multi-faceted performance scores. Our results confirm that TRACE accurately evaluates these complex behaviors in a scalable and cost-effective manner, even with small open-source LLMs. Furthermore, we apply our method to evaluate the trajectories that agents produce while solving tool-augmented tasks, presenting previously unreported observations and their corresponding insights. 

**Abstract (ZH)**: 尽管近期工具增强的基准测试已包含复杂的用户请求和多样的工具，大多数评价方法仍然局限于答案匹配。然而，当解决用户请求所需的步骤增加时，对智能体性能的恰当评价必须超越最终答案，还应评估问题解决过程，包括以前忽略的效率、幻觉和适应性等方面。最直接的评估方法是将智能体的轨迹与真实轨迹进行比较，但这种方法基本受限于标注所有有效真实轨迹的高昂成本。然而，基于简单LLM的评价器难以在没有真实轨迹的情况下详细评估轨迹。为了有效评估智能体，我们提出了TRACE框架，用于多维度评价工具增强的LLM智能体性能。通过整合证据库，TRACE能够有效地对智能体的推理轨迹进行多方面分析和评价。为了验证我们的框架，我们通过在现有基准测试中增加多样且有缺陷的轨迹来构建了一个新的元评价数据集，并为每个轨迹标注了多方面的性能评分。结果显示，TRACE能够以可扩展且低成本的方式准确评价这些复杂行为，即使使用小型开源LLM也是如此。此外，我们还应用这种方法来评估智能体在解决工具增强任务时产生的轨迹，揭示了一些前所未见的观察结果及其相应的见解。 

---
# NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning 

**Title (ZH)**: NCV：一种节点级一致性验证方法，用于低成低成本结构化错误定位在大模型推理中的局部化 

**Authors**: Yulong Zhang, Li Wang, Wei Du, Peilin Li, Yuqin Dai Zhiyuan Zhao, Lingyong Fang, Ziniu Liu, Ru Zhang, Huijia Zhu, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02816)  

**Abstract**: Verifying multi-step reasoning in large language models is difficult due to imprecise error localization and high token costs. Existing methods either assess entire reasoning chains, suffering attention dilution, or rely on expensive multi-sampling. We introduce Node-wise Consistency Verification (NCV), a training-free framework that recasts verification as lightweight binary consistency checks at the node level. By decomposing the chain of thought into interconnected verification nodes, NCV precisely localizes errors and avoids unnecessary long-form generation. Experiments demonstrate that our approach enhances interpretability and efficiency, presenting a scalable solution for reliable LLM reasoning verification. On public datasets, NCV achieves a 10\% to 25\% improvement in F1 scores over baselines while utilizing $6\times$~$58\times$ fewer tokens than traditional methods like CoT-based verifiers. 

**Abstract (ZH)**: 在大规模语言模型中验证多步推理由于精确错误定位不精确和高 tokens 成本而困难。现有的方法要么评估整个推理链，遭受注意力稀释，要么依赖昂贵的多采样。我们引入基于节点的一致性验证（NCV），这是一种无需训练的框架，重新定义验证为节点级别的轻量级二元一致性检查。通过将推理链分解为相互连接的验证节点，NCV 准确定位错误并避免不必要的长形式生成。实验表明，我们的方法增强了可解释性和效率，提供了一种可扩展的可靠的大规模语言模型推理验证解决方案。在公共数据集中，NCV 在 F1 分数上比基准方法提高 10% 至 25%，且使用的 tokens 数量仅为传统方法（如基于 CoT 的验证器）的六分之一到五十八分之一。 

---
# Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation 

**Title (ZH)**: 基于领域特定表示调节生成模型的作业调度自动约束规范方法 

**Authors**: Yu-Zhe Shi, Qiao Xu, Yanjia Li, Mingchen Liu, Huamin Qu, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02679)  

**Abstract**: Advanced Planning and Scheduling (APS) systems have become indispensable for modern manufacturing operations, enabling optimized resource allocation and production efficiency in increasingly complex and dynamic environments. While algorithms for solving abstracted scheduling problems have been extensively investigated, the critical prerequisite of specifying manufacturing requirements into formal constraints remains manual and labor-intensive. Although recent advances of generative models, particularly Large Language Models (LLMs), show promise in automating constraint specification from heterogeneous raw manufacturing data, their direct application faces challenges due to natural language ambiguity, non-deterministic outputs, and limited domain-specific knowledge. This paper presents a constraint-centric architecture that regulates LLMs to perform reliable automated constraint specification for production scheduling. The architecture defines a hierarchical structural space organized across three levels, implemented through domain-specific representation to ensure precision and reliability while maintaining flexibility. Furthermore, an automated production scenario adaptation algorithm is designed and deployed to efficiently customize the architecture for specific manufacturing configurations. Experimental results demonstrate that the proposed approach successfully balances the generative capabilities of LLMs with the reliability requirements of manufacturing systems, significantly outperforming pure LLM-based approaches in constraint specification tasks. 

**Abstract (ZH)**: 基于约束的大型语言模型导向的生产调度自动约束规范架构 

---
# AutoMaAS: Self-Evolving Multi-Agent Architecture Search for Large Language Models 

**Title (ZH)**: AutoMaAS: 自适应多Agent架构搜索用于大型语言模型 

**Authors**: Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02669)  

**Abstract**: Multi-agent systems powered by large language models have demonstrated remarkable capabilities across diverse domains, yet existing automated design approaches seek monolithic solutions that fail to adapt resource allocation based on query complexity and domain requirements. This paper introduces AutoMaAS, a self-evolving multi-agent architecture search framework that leverages neural architecture search principles to automatically discover optimal agent configurations through dynamic operator lifecycle management and automated machine learning techniques. Our approach incorporates four key innovations: (1) automatic operator generation, fusion, and elimination based on performance-cost analysis, (2) dynamic cost-aware optimization with real-time parameter adjustment, (3) online feedback integration for continuous architecture refinement, and (4) enhanced interpretability through decision tracing mechanisms. Extensive experiments across six benchmarks demonstrate that AutoMaAS achieves 1.0-7.1\% performance improvement while reducing inference costs by 3-5\% compared to state-of-the-art methods. The framework shows superior transferability across datasets and LLM backbones, establishing a new paradigm for automated multi-agent system design in the era of large language models. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统自进化多智能体架构搜索框架 

---
# Geolog-IA: Conversational System for Academic Theses 

**Title (ZH)**: Geolog-IA：学术学位论文对话系统 

**Authors**: Micaela Fuel Pozo, Andrea Guatumillo Saltos, Yeseña Tipan Llumiquinga, Kelly Lascano Aguirre, Marilyn Castillo Jara, Christian Mejia-Escobar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02653)  

**Abstract**: This study presents the development of Geolog-IA, a novel conversational system based on artificial intelligence that responds naturally to questions about geology theses from the Central University of Ecuador. Our proposal uses the Llama 3.1 and Gemini 2.5 language models, which are complemented by a Retrieval Augmented Generation (RAG) architecture and an SQLite database. This strategy allows us to overcome problems such as hallucinations and outdated knowledge. The evaluation of Geolog-IA's performance with the BLEU metric reaches an average of 0.87, indicating high consistency and accuracy in the responses generated. The system offers an intuitive, web-based interface that facilitates interaction and information retrieval for directors, teachers, students, and administrative staff at the institution. This tool can be a key support in education, training, and research and establishes a basis for future applications in other disciplines. 

**Abstract (ZH)**: 本研究介绍了基于人工智能的新型对话系统Geolog-IA，该系统能够自然地回答厄瓜多尔中央大学地质论文的相关问题。我们的提议使用了Llama 3.1和Gemini 2.5语言模型，并通过检索增强生成（RAG）架构和SQLite数据库加以补充。这种策略有助于解决幻觉和过时知识等问题。Geolog-IA 的性能评价使用BLEU指标达到平均0.87，表明其生成的回答具有高度一致性和准确性。该系统提供了一个直观的基于Web的界面，便于师生员工在机构中进行互动和信息检索。该工具可以成为教育、培训和研究的关键支持，并为其他学科未来应用奠定基础。 

---
# On the Role of Temperature Sampling in Test-Time Scaling 

**Title (ZH)**: 温度采样在测试时缩放中的作用 

**Authors**: Yuheng Wu, Azalia Mirhoseini, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2510.02611)  

**Abstract**: Large language models (LLMs) can improve reasoning at inference time through test-time scaling (TTS), where multiple reasoning traces are generated and the best one is selected. Prior work shows that increasing the number of samples K steadily improves accuracy. In this paper, we demonstrate that this trend does not hold indefinitely: at large K, further scaling yields no gains, and certain hard questions remain unsolved regardless of the number of traces. Interestingly, we find that different sampling temperatures solve different subsets of problems, implying that single-temperature scaling explores only part of a model's potential. We therefore propose scaling along the temperature dimension, which enlarges the reasoning boundary of LLMs. Averaged over Qwen3 (0.6B, 1.7B, 4B, 8B) and five representative reasoning benchmarks (AIME 2024/2025, MATH500, LiveCodeBench, Hi-ToM), temperature scaling yields an additional 7.3 points over single-temperature TTS. Temperature scaling also enables base models to reach performance comparable to reinforcement learning (RL)-trained counterparts, without additional post-training. We further provide a comprehensive analysis of this phenomenon and design a multi-temperature voting method that reduces the overhead of temperature scaling. Overall, our findings suggest that TTS is more powerful than previously thought, and that temperature scaling offers a simple and effective way to unlock the latent potential of base models. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过测试时缩放（TTS）在推理时可以通过生成多个推理轨迹并选择最佳轨迹来提高推理能力。以往的工作表明，增加样本数量K可以逐步提高准确性。在本文中，我们展示了这一趋势并不无限持续：在K较大时，进一步缩放不再带来改进，并且某些难题无论轨迹数量多少都无法解决。有趣的是，我们发现不同的采样温度解决了不同问题子集，表明单温度缩放只探索了模型潜力的一部分。因此，我们提出沿着温度维度缩放，这扩展了LLMs的推理边界。在Qwen3（0.6B、1.7B、4B、8B）和五个代表性推理基准（AIME 2024/2025、MATH500、LiveCodeBench、Hi-ToM）上，温度缩放相较于单一温度TTS额外获得了7.3分。温度缩放还允许基础模型达到与强化学习（RL）训练对应模型相当的表现，无需额外后训练。我们进一步对此现象进行了全面分析，并设计了一种减少温度缩放开销的多温度投票方法。总之，我们的研究结果表明TTS比之前认为的更强大，而温度缩放提供了一种简单且有效的方法来释放基础模型的潜在能力。 

---
# Agentic Additive Manufacturing Alloy Discovery 

**Title (ZH)**: 代理增材制造合金发现 

**Authors**: Peter Pak, Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2510.02567)  

**Abstract**: Agentic systems enable the intelligent use of research tooling, augmenting a researcher's ability to investigate and propose novel solutions to existing problems. Within Additive Manufacturing (AM), alloy discovery remains a complex challenge, often requiring expertise in the various domains of materials science, thermodynamic simulations, and experimental analysis. Large Language Model (LLM) enabled agents can facilitate this endeavor by utilizing their extensive knowledge base to dispatch tool calls via Model Context Protocol (MCP) to perform actions such as Thermo-Calc property diagram calculations and lack of fusion process map generation. In addition, the multi-agent system developed in this work is able to effectively reason through complex user prompts and provide analysis on the printability of proposed alloys. These agents can dynamically adjust their task trajectory to the outcomes of tool call results, effectively enabling autonomous decision-making in practical environments. This work aims to utilize LLM enabled agents to automate and accelerate the task of alloy discovery within the field of additive manufacturing and showcase the benefits of adopting this multi-agent system. 

**Abstract (ZH)**: 智能系统使研究人员能够智能地使用研究工具，增强其探究和提出解决现有问题的新型解决方案的能力。在增材制造领域，合金发现仍然是一个复杂的挑战，通常需要在材料科学、热力学模拟和实验分析等多个领域的专业知识。本工作中开发的基于大型语言模型的代理能够通过利用其丰富的知识库，并通过模型上下文协议（MCP）调度工具调用（如Thermo-Calc性质图计算和缺焊缝过程图生成）来促进这一努力。此外，本工作中开发的多代理系统能够有效地处理复杂的用户提示，并对所提议合金的可打印性进行分析。这些代理可以根据工具调用结果动态调整其任务路径，从而在实际环境中实现自主决策。本工作旨在利用基于大型语言模型的代理自动化和加速增材制造领域中的合金发现任务，并展示采用这一多代理系统的优点。 

---
# Safe and Efficient In-Context Learning via Risk Control 

**Title (ZH)**: 通过风险控制实现安全高效的即刻学习 

**Authors**: Andrea Wynn, Metod Jazbec, Charith Peris, Rinat Khaziev, Anqi Liu, Daniel Khashabi, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2510.02480)  

**Abstract**: Large language models (LLMs) demonstrate a remarkable ability to learn new tasks from a few in-context examples. However, this flexibility introduces safety concerns: LLMs can be influenced by incorrect or malicious demonstrations -- for example, if an adversary tampers with or injects harmful examples without a human supervisor noticing. This motivates principled designs in which the system itself includes built-in mechanisms to guard against such attacks. We propose a novel approach to limit the degree to which harmful demonstrations can degrade model performance. First, we define a baseline ``safe'' behavior for the model -- the model's performance given no in-context demonstrations (zero-shot). Next, we apply distribution-free risk control (DFRC) to control the extent to which in-context samples can decay performance below zero-shot. We achieve this by leveraging dynamic early exit prediction, ignoring later attention heads that attend the most to the unsafe inputs. Finally, we propose modifications to DFRC that allow it to both control risk for harmful inputs \textit{and} leverage performance and efficiency gains on helpful inputs. We present both theoretical and empirical results showing that our approach can effectively control risk for harmful in-context demonstrations while simultaneously achieving substantial computational efficiency gains with helpful demonstrations. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示出从少量上下文示例中学习新任务的显著能力。然而，这种灵活性引入了安全关切：LLMs 可能受到不正确或恶意示例的影响——例如，如果攻击者在未经人类监督员注意的情况下篡改或注入有害示例。这促使我们必须设计内在具有防范此类攻击机制的系统。我们提出了一种新颖的方法，以限制有害示例对模型性能的负面影响程度。首先，我们定义一个基准的“安全”行为——即在没有上下文示例（零样本）的情况下模型的表现。接下来，我们应用分布无关的的风险控制（DFRC）来控制上下文样本对性能的负面影响程度，使其不超过零样本表现。这通过利用动态早期退出预测来实现，忽略最关注不安全输入的后续注意力头。最后，我们提出了对DFRC的修改，使其既能够控制有害输入的风险，又能利用有助于性能和效率的输入带来的增益。我们提供了理论和实证结果，展示了我们的方法不仅能够有效控制有害上下文示例的风险，还能够在同时利用针对有益示例的性能和效率增益方面实现显著的计算效率提升。 

---
# BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks 

**Title (ZH)**: BrowserArena：评估LLM代理在实际网络导航任务中的性能 

**Authors**: Sagnik Anupam, Davis Brown, Shuo Li, Eric Wong, Hamed Hassani, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2510.02418)  

**Abstract**: LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about captcha resolution. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale. 

**Abstract (ZH)**: LLM网络代理现已能够在开放网络中浏览和执行操作，但当前的代理评估仍局限在沙箱环境中或人工任务中。我们引入了BrowserArena，这是一个实时的开放网络代理评估平台，收集用户提交的任务，进行Arena风格的一对一头对头比较，并使用步骤级的人工反馈来揭示失败模式。通过对代理轨迹进行收集和分析步骤级注释，我们确定了三种一致的失败模式：验证码解决、弹出广告栏移除和直接导航到URL。通过构建针对性的数据集进一步研究这些任务，我们发现不同语言模型在这些失败模式中的导航方式存在差异。例如，o4-mini 在规避验证码解决方面采用了比其他模型更广泛的策略，而DeepSeek-R1在验证码解决方面始终误导用户。我们的发现揭示了当前网络代理的多样性和脆弱性。更广泛地说，我们的基准测试方法提供了一种评估和大规模理解网络代理失败模式的途径。 

---
# Reward Models are Metrics in a Trench Coat 

**Title (ZH)**: 奖励模型是披着 trench coat 的度量标准 

**Authors**: Sebastian Gehrmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.03231)  

**Abstract**: The emergence of reinforcement learning in post-training of large language models has sparked significant interest in reward models. Reward models assess the quality of sampled model outputs to generate training signals. This task is also performed by evaluation metrics that monitor the performance of an AI model. We find that the two research areas are mostly separate, leading to redundant terminology and repeated pitfalls. Common challenges include susceptibility to spurious correlations, impact on downstream reward hacking, methods to improve data quality, and approaches to meta-evaluation. Our position paper argues that a closer collaboration between the fields can help overcome these issues. To that end, we show how metrics outperform reward models on specific tasks and provide an extensive survey of the two areas. Grounded in this survey, we point to multiple research topics in which closer alignment can improve reward models and metrics in areas such as preference elicitation methods, avoidance of spurious correlations and reward hacking, and calibration-aware meta-evaluation. 

**Abstract (ZH)**: 强化学习在大型语言模型后训练中的兴起引发了对奖励模型的广泛关注。奖励模型评估采样模型输出的质量以生成训练信号。这一任务也由评估指标来执行，监控AI模型的性能。我们发现这两个研究领域大多分离，导致术语重复和重复的陷阱。常见的挑战包括对虚假相关性的易感性、对下游奖励作弊的影响、提高数据质量的方法以及元评估方法。我们的立场文件认为，这两个领域的更紧密合作有助于克服这些问题。为此，我们展示了评估指标在特定任务上比奖励模型更优，并提供了两个领域的广泛综述。基于这一综述，我们指出了多个可以通过更紧密对齐来提升奖励模型和评估指标的研究课题，特别是在偏好 elicitation 方法、避免虚假相关性和奖励作弊以及校准意识的元评估方面。 

---
# Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignment 

**Title (ZH)**: 自锚定：通过逐步注意力对齐进行大语言模型推理 

**Authors**: Hongxiang Zhang, Yuan Tian, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03223)  

**Abstract**: To solve complex reasoning tasks for Large Language Models (LLMs), prompting-based methods offer a lightweight alternative to fine-tuning and reinforcement learning. However, as reasoning chains extend, critical intermediate steps and the original prompt will be buried in the context, receiving insufficient attention and leading to errors. In this paper, we propose Self-Anchor, a novel pipeline that leverages the inherent structure of reasoning to steer LLM attention. Self-Anchor decomposes reasoning trajectories into structured plans and automatically aligns the model's attention to the most relevant inference steps, allowing the model to maintain focus throughout generation. Our experiment shows that Self-Anchor outperforms SOTA prompting methods across six benchmarks. Notably, Self-Anchor significantly reduces the performance gap between ``non-reasoning'' models and specialized reasoning models, with the potential to enable most LLMs to tackle complex reasoning tasks without retraining. 

**Abstract (ZH)**: 基于自我锚定的方法解决大型语言模型的复杂推理任务 

---
# Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair 

**Title (ZH)**: 弃权与验证：一种减少代理程序修复噪声的双大型语言模型策略 

**Authors**: José Cambronero, Michele Tufano, Sherry Shi, Renyao Wei, Grant Uy, Runxiang Cheng, Chin-Jung Liu, Shiying Pan, Satish Chandra, Pat Rondon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03217)  

**Abstract**: Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately agent-generated patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing unlikely patches to developers can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patch trajectories rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems. 

**Abstract (ZH)**: 基于代理的自动化程序修复（APR）日益在工业中处理复杂的仓库级别的缺陷，但最终仍需由人类审查生成的补丁以确保其解决了缺陷。向开发者展示不可能的补丁会导致大量噪音，浪费宝贵的开发者时间并损害对自动代码更改的信任。我们引入了两种互补的基于LLM的策略来减少这种噪音：缺陷回避策略和补丁验证策略。缺陷回避策略排除代理APR系统不太可能修复的缺陷。补丁验证策略拒绝不太可能是给定缺陷良好修复的补丁。我们分别在这三个来自Google代码库的缺陷集及其由内部代理APR系统生成的有效补丁集中评估了这两种策略。在一组174个人报告的缺陷中，移除我们的策略拒绝的缺陷和补丁路径，分别可以提高成功率多达13和15个百分点，并且在结合使用时可以提高多达39个百分点。对于空指针异常和 sanitizer 报告的缺陷（伴有机器生成的缺陷报告），补丁验证策略也提高了平均单样本成功率。这种多策略方法提供了将代理APR系统可靠地部署到工业规模的实际途径。 

---
# Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting? 

**Title (ZH)**: 长文生成中的主题建模：长上下文LLM能否通过零样本提示重构NTM？ 

**Authors**: Xuan Xu, Haolun Li, Zhongliang Yang, Beilin Chu, Jia Song, Moxuan Xu, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03174)  

**Abstract**: Traditional topic models such as neural topic models rely on inference and generation networks to learn latent topic distributions. This paper explores a new paradigm for topic modeling in the era of large language models, framing TM as a long-form generation task whose definition is updated in this paradigm. We propose a simple but practical approach to implement LLM-based topic model tasks out of the box (sample a data subset, generate topics and representative text with our prompt, text assignment with keyword match). We then investigate whether the long-form generation paradigm can beat NTMs via zero-shot prompting. We conduct a systematic comparison between NTMs and LLMs in terms of topic quality and empirically examine the claim that "a majority of NTMs are outdated." 

**Abstract (ZH)**: 传统主题模型如神经主题模型依赖于推理网络和生成网络来学习潜在主题分布。本文探讨了大规模语言模型时代主题建模的新范式，将主题建模框架为一种长文本生成任务，并在此范式中更新其定义。我们提出了一种简单实用的方法，可以不经修改直接使用大规模语言模型实现主题模型任务（抽取数据子集，使用提示生成主题和代表性文本，并通过关键词匹配进行文本分配）。我们还研究了这种长文本生成范式是否能够通过零样本提示超越神经主题模型。我们系统比较了神经主题模型和大规模语言模型在主题质量方面的表现，并实证检验了“大多数神经主题模型已经过时”的说法。 

---
# Investigating The Smells of LLM Generated Code 

**Title (ZH)**: investigating LLM生成代码的气味 

**Authors**: Debalina Ghosh Paul, Hong Zhu, Ian Bayley  

**Link**: [PDF](https://arxiv.org/pdf/2510.03029)  

**Abstract**: Context: Large Language Models (LLMs) are increasingly being used to generate program code. Much research has been reported on the functional correctness of generated code, but there is far less on code quality.
Objectives: In this study, we propose a scenario-based method of evaluating the quality of LLM-generated code to identify the weakest scenarios in which the quality of LLM generated code should be improved.
Methods: The method measures code smells, an important indicator of code quality, and compares them with a baseline formed from reference solutions of professionally written code. The test dataset is divided into various subsets according to the topics of the code and complexity of the coding tasks to represent different scenarios of using LLMs for code generation. We will also present an automated test system for this purpose and report experiments with the Java programs generated in response to prompts given to four state-of-the-art LLMs: Gemini Pro, ChatGPT, Codex, and Falcon.
Results: We find that LLM-generated code has a higher incidence of code smells compared to reference solutions. Falcon performed the least badly, with a smell increase of 42.28%, followed by Gemini Pro (62.07%), ChatGPT (65.05%) and finally Codex (84.97%). The average smell increase across all LLMs was 63.34%, comprising 73.35% for implementation smells and 21.42% for design smells. We also found that the increase in code smells is greater for more complex coding tasks and for more advanced topics, such as those involving object-orientated concepts.
Conclusion: In terms of code smells, LLM's performances on various coding task complexities and topics are highly correlated to the quality of human written code in the corresponding scenarios. However, the quality of LLM generated code is noticeably poorer than human written code. 

**Abstract (ZH)**: 背景：大型语言模型（LLMs）越来越多地用于生成程序代码。关于生成代码的功能正确性已有大量研究，但对代码质量的研究相对较少。
目标：在本研究中，我们提出了一种基于场景的方法来评估LLM生成代码的质量，以识别需要改进LLM生成代码质量的最薄弱场景。
方法：该方法衡量代码异味，这是代码质量的重要指标，并将其与来自专业编写代码的参考解决方案形成的基线进行比较。测试数据集根据代码主题和编码任务的复杂性分为多个子集，以代表使用LLM进行代码生成的不同场景。我们还将介绍一种自动化测试系统，并报告针对四款最先进的LLM（Gemini Pro、ChatGPT、Codex和Falcon）生成的Java程序的实验结果。
结果：我们发现，LLM生成的代码相较于参考解决方案具有更高的代码异味发生率。Falcon表现最差，代码异味增加42.28%，其次是Gemini Pro（62.07%）、ChatGPT（65.05%）、最后是Codex（84.97%）。所有LLM的平均代码异味增加率为63.34%，其中实现异味占73.35%，设计异味占21.42%。我们还发现，对于更复杂的编码任务和更高级的主题（如涉及面向对象的概念），代码异味的增加更为显著。
结论：在代码异味方面，LLM在不同编码任务复杂性和主题上的表现与相应场景中人工编写代码的质量高度相关。然而，LLM生成代码的质量明显低于人工编写代码。 

---
# Untargeted Jailbreak Attack 

**Title (ZH)**: 无目标 Jailbreak 攻击 

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.02999)  

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.
To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM this http URL evaluations demonstrate that \textsc{UJA} can achieve over 80\% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%. 

**Abstract (ZH)**: 现有的基于梯度的大型语言模型（LLMs）逃狱攻击，如贪婪坐标梯度（GCG）和COLD-Attack，通常优化对抗后缀以使LLM输出与预定义的目标响应对齐。然而，通过将优化目标限制为诱导预定义目标，这些方法内在地限制了对抗搜索的空间，这限制了它们的整体攻击效果。此外，现有方法通常需要大量的优化迭代来弥补固定目标和原始模型响应之间的巨大差距，导致攻击效率低下。

为了克服定向逃狱攻击的限制，我们提出了第一个基于梯度的无定向逃狱攻击（UJA），旨在引致一种不安全的响应而不强加任何预先定义的模式。具体而言，我们提出了一个无定向攻击目标，以最大化LLM响应的安全性概率，这可以通过法官模型进行量化。由于目标是非可微的，我们进一步将其分解为两个可微的子目标，用于优化最优有害响应及其相应的对抗提示，并进行了理论分析来验证分解的有效性。与定向逃狱攻击相比，UJA的无限制目标显著扩展了搜索空间，使在LLM上进行更灵活和高效的探索成为可能。评估结果表明，\textsc{UJA}仅需100次优化迭代就能在对抗最近的安全对齐的LLM时实现超过80%的攻击成功率，优于包括I-GCG和COLD-Attack在内的最先进的基于梯度的攻击，性能高出超过20%。 

---
# Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines 

**Title (ZH)**: 在临床证据中 grounding 大型语言模型：查询英国国家卫生与临床优化研究所临床指南的检索增强生成系统 

**Authors**: Matthew Lewis, Samuel Thio, Richard JB Dobson, Spiros Denaxas  

**Link**: [PDF](https://arxiv.org/pdf/2510.02967)  

**Abstract**: This paper presents the development and evaluation of a Retrieval-Augmented Generation (RAG) system for querying the United Kingdom's National Institute for Health and Care Excellence (NICE) clinical guidelines using Large Language Models (LLMs). The extensive length and volume of these guidelines can impede their utilisation within a time-constrained healthcare system, a challenge this project addresses through the creation of a system capable of providing users with precisely matched information in response to natural language queries. The system's retrieval architecture, composed of a hybrid embedding mechanism, was evaluated against a database of 10,195 text chunks derived from three hundred guidelines. It demonstrates high performance, with a Mean Reciprocal Rank (MRR) of 0.814, a Recall of 81% at the first chunk and of 99.1% within the top ten retrieved chunks, when evaluated on 7901 queries.
The most significant impact of the RAG system was observed during the generation phase. When evaluated on a manually curated dataset of seventy question-answer pairs, RAG-enhanced models showed substantial gains in performance. Faithfulness, the measure of whether an answer is supported by the source text, was increased by 64.7 percentage points to 99.5% for the RAG-enhanced O4-Mini model and significantly outperformed the medical-focused Meditron3-8B LLM, which scored 43%. This, combined with a perfect Context Precision score of 1 for all RAG-enhanced models, confirms the system's ability to prevent information fabrication by grounding its answers in relevant source material. This study thus establishes RAG as an effective, reliable, and scalable approach for applying generative AI in healthcare, enabling cost-effective access to medical guidelines. 

**Abstract (ZH)**: 本研究介绍了使用大型语言模型（LLMs）查询英国国家卫生与护理卓越研究所（NICE）临床指南的检索增强生成（RAG）系统的开发与评估。由于这些指南的篇幅和数量庞大，可能限制了其在时间有限的医疗系统中的使用，该项目通过创建一个能够针对自然语言查询提供精确匹配信息的系统来应对这一挑战。该系统的检索架构，由混合嵌入机制组成，在针对包含300份指南的10,195个文本片段的数据库进行评估时，展示了高度性能，其中平均互换秩（MRR）为0.814，第一个片段召回率为81%，前十个检索片段召回率为99.1%，评估了7901个查询。

RAG系统的最显著影响体现在生成阶段。在对手动整理的70个问答对数据集进行评估时，增强后的RAG模型在性能上取得了显著提升。忠实度，衡量答案是否由原文支持的指标，对于增强后的O4-Mini模型从43%提高到了99.5%，显著优于专注于医疗领域的Meditron3-8B大型语言模型。结合所有增强后的RAG模型在上下文精确度上的完美得分为1，这证明了该系统能够通过将答案扎根于相关来源材料来防止信息篡改。因此，本研究确立了RAG作为一种有效、可靠且可扩展的方法，在医疗健康领域应用生成式人工智能，从而实现低成本访问医疗指南。 

---
# DMark: Order-Agnostic Watermarking for Diffusion Large Language Models 

**Title (ZH)**: DMark：面向扩散大型语言模型的无序依赖水印技术 

**Authors**: Linyu Wu, Linhao Zhong, Wenjie Qu, Yuexin Li, Yue Liu, Shengfang Zhai, Chunhua Shen, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02902)  

**Abstract**: Diffusion large language models (dLLMs) offer faster generation than autoregressive models while maintaining comparable quality, but existing watermarking methods fail on them due to their non-sequential decoding. Unlike autoregressive models that generate tokens left-to-right, dLLMs can finalize tokens in arbitrary order, breaking the causal design underlying traditional watermarks. We present DMark, the first watermarking framework designed specifically for dLLMs. DMark introduces three complementary strategies to restore watermark detectability: predictive watermarking uses model-predicted tokens when actual context is unavailable; bidirectional watermarking exploits both forward and backward dependencies unique to diffusion decoding; and predictive-bidirectional watermarking combines both approaches to maximize detection strength. Experiments across multiple dLLMs show that DMark achieves 92.0-99.5% detection rates at 1% false positive rate while maintaining text quality, compared to only 49.6-71.2% for naive adaptations of existing methods. DMark also demonstrates robustness against text manipulations, establishing that effective watermarking is feasible for non-autoregressive language models. 

**Abstract (ZH)**: DMark：一种专为扩散大语言模型设计的水印框架 

---
# Evaluating Large Language Models for IUCN Red List Species Information 

**Title (ZH)**: 评估大型语言模型在IUCN红色名录物种信息中的应用 

**Authors**: Shinya Uryu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02830)  

**Abstract**: Large Language Models (LLMs) are rapidly being adopted in conservation to address the biodiversity crisis, yet their reliability for species evaluation is uncertain. This study systematically validates five leading models on 21,955 species across four core IUCN Red List assessment components: taxonomy, conservation status, distribution, and threats. A critical paradox was revealed: models excelled at taxonomic classification (94.9%) but consistently failed at conservation reasoning (27.2% for status assessment). This knowledge-reasoning gap, evident across all models, suggests inherent architectural constraints, not just data limitations. Furthermore, models exhibited systematic biases favoring charismatic vertebrates, potentially amplifying existing conservation inequities. These findings delineate clear boundaries for responsible LLM deployment: they are powerful tools for information retrieval but require human oversight for judgment-based decisions. A hybrid approach is recommended, where LLMs augment expert capacity while human experts retain sole authority over risk assessment and policy. 

**Abstract (ZH)**: 大型语言模型（LLMs）在保护领域迅速应用以应对生物多样性危机，但其在物种评价中的可靠性尚不确定。本研究系统验证了五种领先模型在41955个物种上的评估，涵盖国际自然保护联盟红色名录的四个核心评估组件：分类学、保护状况、分布和威胁。研究揭示了一个关键的悖论：模型在分类学分类上表现出色（94.9%），但在保护推理上却持续失败（状态评估仅为27.2%）。这种知识-推理差距在所有模型中普遍存在，表明存在内在的架构限制，而不仅仅是数据限制。此外，模型显示出系统性的偏见，偏向于 charismatic 脊椎动物，这可能会加剧现有的保护不平等性。这些发现明确了负责任地部署大型语言模型的边界：它们是信息检索的强大工具，但在基于判断的决策上需要人类监督。推荐采用混合方法，其中大型语言模型增强专家能力，而人类专家保留对风险评估和政策制定的独家权威。 

---
# Dissecting Transformers: A CLEAR Perspective towards Green AI 

**Title (ZH)**: 解析 Transformers：一条通往绿色人工智能的清晰路径 

**Authors**: Hemang Jain, Shailender Goyal, Divyansh Pandey, Karthik Vaidhyanathan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02810)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) has raised significant environmental concerns. Unlike the one-time cost of training, LLM inference occurs continuously at a global scale and now dominates the AI energy footprint. Yet, most sustainability studies report only coarse, model-level metrics due to the lack of fine-grained measurement methods, treating energy efficiency more as an afterthought than as a primary objective. We present the first fine-grained empirical analysis of inference energy across core components of transformer architecture. We propose a novel methodology, Component-Level Energy Assessment via Repeated sampling (CLEAR), to overcome temporal mismatch between microsecond scale component execution and monitoring of millisecond (ms) scale energy sensors. Using CLEAR, we evaluate 15 models spanning four distinct architecture types and consistently keep component-wise energy variance below 9.5\% while capturing more than 90\% of the model's total energy as individual components. Our empirical analysis reveals that Attention blocks consume significantly more energy per floating-point operation (FLOP), indicating that energy consumption is not proportionally aligned with FLOP counts. This shows that FLOPs alone fail to capture the true energy cost at a component level. Our findings establish detailed component-level energy baselines and provide insight as an initial step to build energy-efficient transformer models through component-level optimizations. 

**Abstract (ZH)**: 大型语言模型的快速采用引起了 significant 的环境关注。不同于一次性训练成本，推理持续在全球范围内进行，并现在主导了人工智能的能源足迹。然而，大多数可持续性研究仅报告粗略的模型级别指标，由于缺乏精细粒度的测量方法，将能效更多地视为一种附带结果而非主要目标。我们首次对变压器架构核心组件的推理能耗进行了精细粒度的经验分析。我们提出了一种新颖的方法——组件级能耗评估通过重复采样（CLEAR），以克服微秒级组件执行与毫秒级能耗传感器监控之间的时间不匹配问题。使用 CLEAR，我们评估了涵盖四种不同架构类型的 15 个模型，并在各个组件上保持能耗变异率低于 9.5% 的同时，捕获了模型总能耗的 90% 以上。我们的经验分析揭示了注意力模块每浮点运算（FLOP）能耗显著增加，表明能耗与 FLOP 计数不成比例。这表明 FLOPs 不能准确反映组件级别的真实能耗。我们的研究建立了详细的组件级别能耗基线，并为通过组件级优化构建高效变压器模型提供了一种初始步骤。 

---
# Pareto-optimal Non-uniform Language Generation 

**Title (ZH)**: 帕累托最优非均匀语言生成 

**Authors**: Moses Charikar, Chirag Pabbaraju  

**Link**: [PDF](https://arxiv.org/pdf/2510.02795)  

**Abstract**: Kleinberg and Mullainathan (2024) recently proposed an interesting model for language generation in the limit: Given a countable collection of languages, and an adversary enumerating the strings of some language $L$ from the collection, the objective is to generate new strings from the target language, such that all strings generated beyond some finite time are valid. Li, Raman and Tewari (2024) and Charikar and Pabbaraju (2024) showed strong non-uniform generation guarantees in this model, giving algorithms that generate new valid strings from $L$ after seeing a number of distinct input strings $t(L)$ that depends only on $L$ (and the collection), but not the enumeration order. However, for both these works, the language-wise generation times $t(L)$ of the algorithm can be strictly sub-optimal.
In this work, we study Pareto-optimality of non-uniform language generation in the limit. We propose an algorithm, whose generation times $t^\star(L)$ are (almost) Pareto-optimal: any other algorithm whose generation time for some language $L$ is strictly smaller than $t^\star(L)$, must satisfy that its generation time for some other language $L'$ is strictly worse than $t^\star(L')$. Pareto-optimality is essentially the best that one can achieve for non-uniform generation. Our algorithmic framework conveniently adapts to further give Pareto-optimal non-uniform generation algorithms in the practically motivated settings of noisy as well as representative generation. 

**Abstract (ZH)**: Kleinberg和Mullainathan (2024)提出的语言生成极限模型：基于可数语言集合和对手按某种语言$L$的字符串枚举，目标是从目标语言生成新字符串，使得生成的字符串在某一时点后均有效。Li, Raman和Tewari (2024)以及Charikar和Pabbaraju (2024)展示了在这种模型下的强大非统一生成保证，提供了在看到数量为$t(L)$的不同输入字符串后生成新有效字符串$L$的算法，$t(L)$仅依赖于$L$（及其集合），而不依赖于枚举顺序。然而，对于这两项工作中的算法，语言层面的生成时间$t(L)$可能是严格次优的。在这项工作中，我们研究极限下非统一语言生成的帕累托最优性。我们提出了一个算法，其生成时间$t^\star(L)$几乎是帕累托最优的：对于某些语言$L$，如果另一算法的生成时间严格小于$t^\star(L)$，那么它在其他语言$L'$上的生成时间必须严格劣于$t^\star(L')$。帕累托最优性实际上是非统一生成所能达到的最佳效果。我们的算法框架方便地适用于进一步给出在噪声以及代表性生成等实际驱动设置下的帕累托最优非统一生成算法。 

---
# MaskCD: Mitigating LVLM Hallucinations by Image Head Masked Contrastive Decoding 

**Title (ZH)**: MaskCD: 减轻LVLM幻觉的图像头部屏蔽对比解码方法 

**Authors**: Jingyuan Deng, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02790)  

**Abstract**: Large vision-language models (LVLMs) have shown remarkable performance in visual-language understanding for downstream multimodal tasks. While their capabilities are improving, problems emerge simultaneously. Among those problems, the hallucinations have attracted much attention, which stands for the phenomenon where LVLMs generate contradictory content to their input visual and text contents. Many approaches have been proposed to deal with this issue, such as contrastive decoding and attention manipulation. However, contrastive decoding methods struggle in constructing appropriate contrastive samples, and attention manipulation methods are highly sensitive, lacking stability. In this work, we propose image head Masked Contrastive Decoding (MaskCD). Our approach utilizes the "image heads" in LVLMs, masking them to construct contrastive samples for contrastive decoding. We evaluated MaskCD on LLaVA-1.5-7b and Qwen-VL-7b, using various benchmarks such as CHAIR, POPE, AMBER and MME. The results demonstrate that MaskCD effectively alleviates the phenomenon of hallucinations and retains the general capabilities of LVLMs. Corresponding resources could be found at: this https URL . 

**Abstract (ZH)**: Large 视觉-语言模型 (LVLMs) 在下游多模态任务中的视觉-语言理解方面展现了 remarkable 绩效。尽管其能力在提升，同时出现了一些问题，其中幻觉现象尤为引人关注，指的是 LVLMs 生成与输入视觉和文本内容矛盾的内容。许多方法被提出以应对这一问题，如对比解码和注意力 manipulation。然而，对比解码方法在构建合适的对比样本方面存在困难，注意力 manipulation 方法则不稳定且灵敏。本文提出了基于 “图像头部” 的 Masked 对比解码 (MaskCD)。我们的方法利用 LVLMs 中的 “图像头部”，将其遮蔽以构建对比样本进行对比解码。我们在 LLaVA-1.5-7b 和 Qwen-VL-7b 上使用了 CHAIR、POPE、AMBER 和 MME 等基准进行评估。实验结果表明，MaskCD 有效缓解了幻觉现象，并保持了 LVLMs 的一般能力。相关资源可在此处找到：this https URL。 

---
# Prototyping Digital Social Spaces through Metaphor-Driven Design: Translating Spatial Concepts into an Interactive Social Simulation 

**Title (ZH)**: 基于隐喻驱动设计的数字社会空间原型设计：将空间概念转化为互动社会模拟 

**Authors**: Yoojin Hong, Martina Di Paola, Braahmi Padmakumar, Hwi Joon Lee, Mahnoor Shafiq, Joseph Seering  

**Link**: [PDF](https://arxiv.org/pdf/2510.02759)  

**Abstract**: Social media platforms are central to communication, yet their designs remain narrowly focused on engagement and scale. While researchers have proposed alternative visions for online spaces, these ideas are difficult to prototype within platform constraints. In this paper, we introduce a metaphor-driven system to help users imagine and explore new social media environments. The system translates users' metaphors into structured sets of platform features and generates interactive simulations populated with LLM-driven agents. To evaluate this approach, we conducted a study where participants created and interacted with simulated social media spaces. Our findings show that metaphors allow users to express distinct social expectations, and that perceived authenticity of the simulation depended on how well it captured dynamics like intimacy, participation, and temporal engagement. We conclude by discussing how metaphor-driven simulation can be a powerful design tool for prototyping alternative social architectures and expanding the design space for future social platforms. 

**Abstract (ZH)**: 社会媒体平台在交流中占据核心地位，但其设计依然狭隘地集中在参与度和规模上。尽管研究人员提出了在线空间的替代愿景，但在平台限制内进行原型设计仍然具有挑战性。在本文中，我们介绍了一种基于隐喻的系统，以帮助用户构想和探索新的社交媒体环境。该系统将用户的隐喻转换为结构化的平台功能集，并生成由LLM驱动的代理 populate 的交互式模拟。为了评估该方法，我们进行了一项研究，参与者创建并互动了模拟的社交媒体空间。我们的研究结果表明，隐喻使用户能够表达独特的社会期望，而模拟的感知真实性取决于它如何捕捉如亲密性、参与度和时间参与等动态特征。最后，我们讨论了基于隐喻的模拟如何成为一种强大的设计工具，用于原型设计替代的社会架构，并扩大未来社交媒体平台的设计空间。 

---
# SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA Language Model Representations 

**Title (ZH)**: SAE-RNA：一种稀疏自编码模型，用于解析RNA语言模型表示 

**Authors**: Taehan Kim, Sangdae Nam  

**Link**: [PDF](https://arxiv.org/pdf/2510.02734)  

**Abstract**: Deep learning, particularly with the advancement of Large Language Models, has transformed biomolecular modeling, with protein advances (e.g., ESM) inspiring emerging RNA language models such as RiNALMo. Yet how and what these RNA Language Models internally encode about messenger RNA (mRNA) or non-coding RNA (ncRNA) families remains unclear. We present SAE- RNA, interpretability model that analyzes RiNALMo representations and maps them to known human-level biological features. Our work frames RNA interpretability as concept discovery in pretrained embeddings, without end-to-end retraining, and provides practical tools to probe what RNA LMs may encode about ncRNA families. The model can be extended to close comparisons between RNA groups, and supporting hypothesis generation about previously unrecognized relationships. 

**Abstract (ZH)**: 深度学习，特别是随着大型语言模型的发展，已经转变了生物分子建模，蛋白质进步（如ESM）启发了新兴的RNA语言模型（如RiNALMo）。然而，这些RNA语言模型内部如何以及有何种方式编码信使RNA（mRNA）或非编码RNA（ncRNA）家族的信息仍不清楚。我们提出了SAE-RNA，一种解释性模型，用于分析RiNALMo表示并将其与已知的人类级生物特征关联起来。我们的工作将RNA解释性构建为预训练嵌入的概念发现，并提供了一种实用的工具，以探究RNA LMs可能编码的ncRNA家族信息。该模型可以扩展到RNA组之间的密切比较，并支持对先前未知关系的假设生成。 

---
# TravelBench : Exploring LLM Performance in Low-Resource Domains 

**Title (ZH)**: TravelBench: 探究大语言模型在低资源领域中的性能 

**Authors**: Srinivas Billa, Xiaonan Jing  

**Link**: [PDF](https://arxiv.org/pdf/2510.02719)  

**Abstract**: Results on existing LLM benchmarks capture little information over the model capabilities in low-resource tasks, making it difficult to develop effective solutions in these domains. To address these challenges, we curated 14 travel-domain datasets spanning 7 common NLP tasks using anonymised data from real-world scenarios, and analysed the performance across LLMs. We report on the accuracy, scaling behaviour, and reasoning capabilities of LLMs in a variety of tasks. Our results confirm that general benchmarking results are insufficient for understanding model performance in low-resource tasks. Despite the amount of training FLOPs, out-of-the-box LLMs hit performance bottlenecks in complex, domain-specific scenarios. Furthermore, reasoning provides a more significant boost for smaller LLMs by making the model a better judge on certain tasks. 

**Abstract (ZH)**: 现有的大规模语言模型基准在低资源任务中对模型能力的捕捉信息有限，使得在这些领域中开发有效解决方案困难重重。为应对这些挑战，我们通过匿名化真实场景数据，整理了涵盖7种常见NLP任务的14个旅游领域数据集，并分析了这些数据集上多种语言模型的性能。我们报告了语言模型在多种任务中的准确率、扩展行为以及推理能力。我们的结果证实，通用基准测试结果不足以理解模型在低资源任务中的表现。尽管训练计算量巨大，开箱即用的语言模型在复杂且领域特定的场景中仍会遇到性能瓶颈。此外，推理对于较小的语言模型提供了更大的助力，使模型在某些任务上表现得更为出色。 

---
# Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks 

**Title (ZH)**: 时间不一致性：大型语言模型对抗攻击稳健性的生存分析 

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2510.02712)  

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present the first comprehensive survival analysis of conversational AI robustness, analyzing 36,951 conversation turns across 9 state-of-the-art LLMs to model failure as a time-to-event process. Our survival modeling framework-employing Cox proportional hazards, Accelerated Failure Time, and Random Survival Forest approaches-reveals extraordinary temporal dynamics. We find that abrupt, prompt-to-prompt(P2P) semantic drift is catastrophic, dramatically increasing the hazard of conversational failure. In stark contrast, gradual, cumulative drift is highly protective, vastly reducing the failure hazard and enabling significantly longer dialogues. AFT models with interactions demonstrate superior performance, achieving excellent discrimination and exceptional calibration. These findings establish survival analysis as a powerful paradigm for evaluating LLM robustness, offer concrete insights for designing resilient conversational agents, and challenge prevailing assumptions about the necessity of semantic consistency in conversational AI Systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）虽然革新了对话式AI，但在扩展的多轮对话中的鲁棒性仍 poorly understood，现有评估框架主要关注静态基准和单轮评估，无法捕捉到对话退化的时间动态特征，这在真实世界交互中尤为明显。在此项工作中，我们首次进行了全面的对话式AI鲁棒性生存分析，分析了9个先进大型语言模型中的36,951轮对话，将对话失败视为时间事件过程进行建模。我们运用Cox比例风险、加速失效时间以及随机生存森林方法构建的生存模型框架揭示了非凡的时间动态特征。研究发现，突然的、从指令到指令的语义漂移是灾难性的，显著增加了对话失败的风险。相比之下，渐进的、累积的漂移是保护性的，极大地降低了失败风险，使对话能够显著延长。交互作用的加速失效时间模型表现出卓越的性能，实现了优秀的区分能力和出色的校准。这些发现确立了生存分析作为评估大型语言模型鲁棒性的有力范式，提供了具体的设计稳健对话代理的见解，并挑战了对话式AI系统中语义一致性必要性的传统假设。 

---
# To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration 

**Title (ZH)**: 是否压缩？以指数集中为基础推动无损GenAI模型权重压缩的前沿探索 

**Authors**: Zeyu Yang, Tianyi Zhang, Jianwen Xie, Chuan Li, Zhaozhuo Xu, Anshumali Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.02676)  

**Abstract**: The scaling of Generative AI (GenAI) models into the hundreds of billions of parameters makes low-precision computation indispensable for efficient deployment. We argue that the fundamental solution lies in developing low-precision floating-point formats, which inherently provide numerical stability, memory savings, and hardware efficiency without dequantization overhead. In this paper, we present a theoretical and empirical study of an exponent concentration phenomenon in GenAI weights: exponents consistently exhibit low entropy across architectures and modalities. We show that this arises naturally from $\alpha$-stable distributions induced by stochastic gradient descent, and we prove tight bounds on the entropy of exponents. Our analysis establishes a theoretical compression limit near FP4.67, which motivates the design of a practical FP8 format. Building on these insights, we propose Exponent-Concentrated FP8 (ECF8), a lossless compression framework with entropy-aware encoding and GPU-optimized decoding. Experiments on LLMs and DiTs up to 671B parameters demonstrate up to 26.9% memory savings and 177.1% throughput acceleration, with perfectly lossless computations, i.e., no deviation in model outputs. Our results establish exponent concentration as a statistical law of trained models and open a principled path for lossless low-precision floating-point design in the FP8 era. 

**Abstract (ZH)**: Generative AI模型从数十亿参数扩展到数百亿参数使得低精度计算成为高效部署的必不可少手段。我们argue认为根本解决方案在于开发低精度浮点格式，这种格式本身就提供了数值稳定性、内存节省和硬件效率，而无需去量化开销。本文我们提出了生成AI权重中指数集中现象的理论和实证研究：指数在架构和模态之间始终保持低熵。我们证明了这种现象自然源自由随机梯度下降诱导的α稳定分布，并证明了指数熵的紧界。我们的分析确立了一个接近FP4.67的理论压缩极限，这促使设计出一种实用的FP8格式。基于这些见解，我们提出了指数集中FP8（ECF8），这是一种具有熵感知编码和GPU优化解码的无损压缩框架。针对参数多达671B的LLM和DiT模型的实验结果显示出高达26.9%的内存节省和177.1%的吞吐量加速，同时保持完全无损的计算，即模型输出无偏差。我们的结果确立了指数集中心理统计规律，并为FP8时代的无损低精度浮点数设计提供了一条原理性的路径。 

---
# HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inference 

**Title (ZH)**: HALO：基于内存的二维半集成异构加速器用于低批次LLM推理 

**Authors**: Shubham Negi, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2510.02675)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) has driven a growing demand for efficient inference, particularly in latency-sensitive applications such as chatbots and personalized assistants. Unlike traditional deep neural networks, LLM inference proceeds in two distinct phases: the prefill phase, which processes the full input sequence in parallel, and the decode phase, which generates tokens sequentially. These phases exhibit highly diverse compute and memory requirements, which makes accelerator design particularly challenging. Prior works have primarily been optimized for high-batch inference or evaluated only short input context lengths, leaving the low-batch and long context regime, which is critical for interactive applications, largely underexplored.
We propose HALO, a heterogeneous memory centric accelerator designed for these unique challenges of prefill and decode phases in low-batch LLM inference. HALO integrates HBM based Compute-in-DRAM (CiD) with an on-chip analog Compute-in-Memory (CiM), co-packaged using 2.5D integration. To further improve the hardware utilization, we introduce a phase-aware mapping strategy that adapts to the distinct demands of the prefill and decode phases. Compute bound operations in the prefill phase are mapped to CiM to exploit its high throughput matrix multiplication capability, while memory-bound operations in the decode phase are executed on CiD to benefit from reduced data movement within DRAM. Additionally, we present an analysis of the performance tradeoffs of LLMs under two architectural extremes: a fully CiD and a fully on-chip analog CiM design to highlight the need for a heterogeneous design. We evaluate HALO on LLaMA-2 7B and Qwen3 8B models. Our experimental results show that LLMs mapped to HALO achieve up to 18x geometric mean speedup over AttAcc, an attention-optimized mapping and 2.5x over CENT, a fully CiD based mapping. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的快速采用推动了对高效推理的日益增长的需求，特别是在低延迟应用如聊天机器人和个人助理中。与传统的深度神经网络不同，LLM推理分为两个不同的阶段：并行处理完整输入序列的预填充阶段，和顺序生成令牌的解码阶段。这两个阶段的计算和内存需求高度不同，这使得加速器设计尤为具有挑战性。以往的工作主要针对高批次推理进行了优化，或者仅评估了较短的输入上下文长度，而对于对交互式应用至关重要的低批次和长上下文区间，相关的研究仍然不足。

我们提出了HALO，一种针对低批次LLM推理中预填充和解码阶段独特挑战的异构内存为中心的加速器。HALO结合了基于HBM的计算在内存中（CiD）与片上模拟计算在内存中（CiM）技术，并通过2.5D集成实现共封装。为了进一步提高硬件利用率，我们引入了一种基于阶段的映射策略，该策略能够适应预填充和解码阶段的不同需求。预填充阶段的计算密集型操作被映射到CiM，以利用其高吞吐量矩阵乘法能力，而解码阶段的内存密集型操作在CiD中执行，以减少DRAM内的数据移动。此外，我们对在两种架构极端条件下LLM的性能进行了分析：一种完全基于CiD的设计和一种完全基于片上模拟CiM的设计，以突显异构设计的需求。我们在LLaMA-2 7B和Qwen3 8B模型上评估了HALO。实验结果显示，映射到HALO上的LLMs相对于AttAcc（一种注意力优化的映射）实现了最高达18倍的几何平均加速，相对于CENT（一种完全基于CiD的设计）实现了2.5倍的加速。 

---
# TutorBench: A Benchmark To Assess Tutoring Capabilities Of Large Language Models 

**Title (ZH)**: TutorBench: 评估大型语言模型授课能力的基准测试 

**Authors**: Rakshith S Srinivasa, Zora Che, Chen Bo Calvin Zhang, Diego Mares, Ernesto Hernandez, Jayeon Park, Dean Lee, Guillermo Mangialardi, Charmaine Ng, Ed-Yeremai Hernandez Cardona, Anisha Gunjal, Yunzhong He, Bing Liu, Chen Xing  

**Link**: [PDF](https://arxiv.org/pdf/2510.02663)  

**Abstract**: As students increasingly adopt large language models (LLMs) as learning aids, it is crucial to build models that are adept at handling the nuances of tutoring: they need to identify the core needs of students, be adaptive, provide personalized guidance, and be accurate. To this end, we introduce TutorBench, a dataset and evaluation benchmark designed to rigorously evaluate the core tutoring skills of LLMs. The dataset comprises 1,490 samples curated by human experts, focused on high-school and AP-level curricula. The samples are drawn from three common tutoring tasks: (i) generating adaptive explanations tailored to a student's confusion, (ii) providing actionable feedback on a student's work, and (iii) promoting active learning through effective hint generation. To account for the inherent complexity of tutoring, samples are accompanied by sample-specific rubrics which are used to judge model responses during evaluation. TutorBench uses a reliable and fine-grained automatic evaluation method that uses an LLM-judge and the sample-specific rubrics. We evaluate 16 frontier LLMs on TutorBench and present a detailed analysis of their performance and behavior. Our results show that none of the frontier LLMs achieve a score of greater than $56\%$, showing a large room for improvement. We find that LLMs fall short in exhibiting the full range of tutoring skills needed to guide, diagnose, and support students effectively, with all the frontier models achieving less than a $60\%$ pass rate on rubric criteria related to these skills. We also find that different model families exhibit varied strengths and limitations: the Claude models outperform others in supporting active learning, while they lag behind in the other two use cases. By releasing TutorBench, we provide a comprehensive and unsaturated benchmark to guide the development of the next-generation of AI tutors. 

**Abstract (ZH)**: 随着学生越来越多地采用大语言模型（LLMs）作为学习辅助工具，构建能够处理辅导细微需求的模型变得至关重要：这些模型需要识别学生的核心需求、具备适应性、提供个性化指导并且准确。为此，我们介绍了TutorBench，一个数据集和评估基准，旨在严格评估LLM的核心辅导技能。该数据集包含1,490个由人类专家精心挑选的样本，重点关注高中和AP级别的课程内容。样本源自三种常见的辅导任务：（i）为学生困惑量身定制的适应性解释生成；（ii）对学生作业提供具有行动指导的反馈；（iii）通过有效的线索生成促进主动学习。为应对辅导的固有复杂性，每个样本都附有特定的标准评分表，这些评分表在评估过程中用于评判模型的回答。TutorBench使用一个可靠且细致的自动评估方法，结合LLM裁判和样本特定的标准评分表。我们在TutorBench上评估了16个前沿的LLM，并详细分析了它们的表现和行为。我们的结果显示，没有任何前沿的LLM能够获得超过56%的分数，显示出很大的改进空间。我们发现，LLM在展现全面的辅导技能以有效指导、诊断和支持学生方面存在不足，所有前沿模型在与这些技能相关的评分标准上通过率低于60%。我们还发现，不同的模型家族表现出不同的优势和限制：Claude模型在支持主动学习方面表现优于其他模型，但在另外两种使用场景中落后。通过发布TutorBench，我们提供了一个全面且未饱和的基准，以指导下一代AI辅导系统的开发。 

---
# When Researchers Say Mental Model/Theory of Mind of AI, What Are They Really Talking About? 

**Title (ZH)**: 当研究者提到AI的心智模型/理论思维时，他们究竟在讨论什么？ 

**Authors**: Xiaoyun Yin, Elmira Zahmat Doost, Shiwen Zhou, Garima Arya Yadav, Jamie C. Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2510.02660)  

**Abstract**: When researchers claim AI systems possess ToM or mental models, they are fundamentally dis- cussing behavioral predictions and bias corrections rather than genuine mental states. This position paper argues that the current discourse conflates sophisticated pattern matching with authentic cog- nition, missing a crucial distinction between simulation and experience. While recent studies show LLMs achieving human-level performance on ToM laboratory tasks, these results are based only on behavioral mimicry. More importantly, the entire testing paradigm may be flawed in applying individual human cognitive tests to AI systems, but assessing human cognition directly in the moment of human-AI interaction. I suggest shifting focus toward mutual ToM frameworks that acknowledge the simultaneous contributions of human cognition and AI algorithms, emphasizing the interaction dynamics, instead of testing AI in isolation. 

**Abstract (ZH)**: 当研究人员声称AI系统具备理论心智或心理模型时，他们本质上讨论的是行为预测和偏见矫正，而非真正的心智状态。本文认为当前的讨论将复杂的模式匹配与真正的认知混为一谈，忽视了模拟与体验之间的关键区别。虽然近期研究表明，大语言模型在理论心智实验室任务上达到了人类级别的性能，但这些结果仅基于行为模仿。更重要的是，将个体人类认知测试直接应用于AI系统进行全面评估的方法可能存在问题，而应在人类与AI交互的瞬间直接评估人类的认知。建议转向关注人机共有的理论心智框架，强调人类认知和AI算法的相互贡献，以及交互动态，而非单独测试AI。 

---
# Automatic Building Code Review: A Case Study 

**Title (ZH)**: 自动建筑规范审查：一个案例研究 

**Authors**: Hanlong Wan, Weili Xu, Michael Rosenberg, Jian Zhang, Aysha Siddika  

**Link**: [PDF](https://arxiv.org/pdf/2510.02634)  

**Abstract**: Building officials, particularly those in resource-constrained or rural jurisdictions, face labor-intensive, error-prone, and costly manual reviews of design documents as projects increase in size and complexity. The growing adoption of Building Information Modeling (BIM) and Large Language Models (LLMs) presents opportunities for automated code review (ACR) solutions. This study introduces a novel agent-driven framework that integrates BIM-based data extraction with automated verification using both retrieval-augmented generation (RAG) and Model Context Protocol (MCP) agent pipelines. The framework employs LLM-enabled agents to extract geometry, schedules, and system attributes from heterogeneous file types, which are then processed for building code checking through two complementary mechanisms: (1) direct API calls to the US Department of Energy COMcheck engine, providing deterministic and audit-ready outputs, and (2) RAG-based reasoning over rule provisions, enabling flexible interpretation where coverage is incomplete or ambiguous.
The framework was evaluated through case demonstrations, including automated extraction of geometric attributes (such as surface area, tilt, and insulation values), parsing of operational schedules, and validation of lighting allowances under ASHRAE Standard 90.1-2022. Comparative performance tests across multiple LLMs showed that GPT-4o achieved the best balance of efficiency and stability, while smaller models exhibited inconsistencies or failures. Results confirm that MCP agent pipelines outperform RAG reasoning pipelines in rigor and reliability. This work advances ACR research by demonstrating a scalable, interoperable, and production-ready approach that bridges BIM with authoritative code review tools. 

**Abstract (ZH)**: 一种基于BIM数据提取与LLM驱动验证的新型代理框架：自动代码审查的弹性解决方案 

---
# ToolTweak: An Attack on Tool Selection in LLM-based Agents 

**Title (ZH)**: ToolTweak: 对基于LLM的智能体工具选择的攻击 

**Authors**: Jonathan Sneh, Ruomei Yan, Jialin Yu, Philip Torr, Yarin Gal, Sunando Sengupta, Eric Sommerlade, Alasdair Paren, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02554)  

**Abstract**: As LLMs increasingly power agents that interact with external tools, tool use has become an essential mechanism for extending their capabilities. These agents typically select tools from growing databases or marketplaces to solve user tasks, creating implicit competition among tool providers and developers for visibility and usage. In this paper, we show that this selection process harbors a critical vulnerability: by iteratively manipulating tool names and descriptions, adversaries can systematically bias agents toward selecting specific tools, gaining unfair advantage over equally capable alternatives. We present ToolTweak, a lightweight automatic attack that increases selection rates from a baseline of around 20% to as high as 81%, with strong transferability between open-source and closed-source models. Beyond individual tools, we show that such attacks cause distributional shifts in tool usage, revealing risks to fairness, competition, and security in emerging tool ecosystems. To mitigate these risks, we evaluate two defenses: paraphrasing and perplexity filtering, which reduce bias and lead agents to select functionally similar tools more equally. All code will be open-sourced upon acceptance. 

**Abstract (ZH)**: 随着大语言模型越来越多地驱动与外部工具交互的代理，工具使用已成为扩展其能力的重要机制。这些代理通常从不断增长的数据库或市场中选择工具来解决用户任务，从而在工具提供者和开发者之间产生了隐含的竞争，以求获得更多的可见性和使用量。在本文中，我们展示了一个关键的安全漏洞：通过迭代地操纵工具名称和描述，攻击者可以系统地引导代理选择特定工具，从而在同等能力的选项中获得不公平的优势。我们提出了一种轻量级的自动攻击工具称为ToolTweak，该攻击将基础选择率从约20%提高到高达81%，并且在开源和封闭源模型之间具有强大的可移植性。除了针对个别工具之外，我们还展示了这些攻击导致工具使用分布的变化，揭示了新兴工具生态系统中公平性、竞争性和安全性方面的风险。为了减轻这些风险，我们评估了两种防御措施：同义词重写和困惑度过滤，这些措施减少了偏差并促使代理更平等选择功能相似的工具。所有代码将在接受后开源。 

---
# Knowledge-Graph Based RAG System Evaluation Framework 

**Title (ZH)**: 基于知识图谱的RAG系统评估框架 

**Authors**: Sicheng Dong, Vahid Zolfaghari, Nenad Petrovic, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2510.02549)  

**Abstract**: Large language models (LLMs) has become a significant research focus and is utilized in various fields, such as text generation and dialog systems. One of the most essential applications of LLM is Retrieval Augmented Generation (RAG), which greatly enhances generated content's reliability and relevance. However, evaluating RAG systems remains a challenging task. Traditional evaluation metrics struggle to effectively capture the key features of modern LLM-generated content that often exhibits high fluency and naturalness. Inspired by the RAGAS tool, a well-known RAG evaluation framework, we extended this framework into a KG-based evaluation paradigm, enabling multi-hop reasoning and semantic community clustering to derive more comprehensive scoring metrics. By incorporating these comprehensive evaluation criteria, we gain a deeper understanding of RAG systems and a more nuanced perspective on their performance. To validate the effectiveness of our approach, we compare its performance with RAGAS scores and construct a human-annotated subset to assess the correlation between human judgments and automated metrics. In addition, we conduct targeted experiments to demonstrate that our KG-based evaluation method is more sensitive to subtle semantic differences in generated outputs. Finally, we discuss the key challenges in evaluating RAG systems and highlight potential directions for future research. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为一个重要研究焦点，并被应用于 various fields，如文本生成和对话系统。LLMs最重要的应用之一是检索增强生成（RAG），这极大地提高了生成内容的可靠性和相关性。然而，评估RAG系统仍是一个具有挑战性的工作。传统的评估指标难以有效捕捉现代LLM生成内容的关键特征，这些内容通常表现出高度的流畅性和自然性。受RAGAS工具启发，一个知名的RAG评估框架，我们将其扩展到基于知识图谱（KG）的评估框架，允许多跳推理和语义社区聚类，从而获得更全面的评分指标。通过纳入这些综合评估标准，我们对RAG系统有了更深入的理解，并获得了对其性能的更细致的看法。为了验证我们方法的有效性，我们将性能与RAGAS评分进行了比较，并构建了一个人工标注的子集来评估人工判断与自动化指标之间的相关性。此外，我们进行了针对性的实验，以证明我们的基于知识图谱的评估方法对生成输出中细微语义差异更为敏感。最后，我们讨论了评估RAG系统的关键挑战，并指出了未来研究的潜在方向。 

---
# Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework 

**Title (ZH)**: Litespark 技术报告：高吞吐量、低能耗的大型语言模型训练框架 

**Authors**: Nii Osae Osae Dade, Moinul Hossain Rahat  

**Link**: [PDF](https://arxiv.org/pdf/2510.02483)  

**Abstract**: Training Large Language Models (LLMs) is plagued by long training times and massive energy consumption, with modern models requiring months of computation and gigawatt-hours of electricity. In light of these challenges,we introduce Litespark, a novel pre-training framework that addresses these inefficiencies through targeted optimizations to transformer attention and MLP layers. Our approach combines architectural improvements with algorithmic enhancements to maximize Model FLOPs Utilization (MFU) while maintaining compatibility with standard transformer implementations. Comprehensive benchmarking on 3B and 30B parameter Llama models using the SlimPajama-627B dataset demonstrates substantial performance gains: 2x-6x training throughput improvement and $55\%-83$% energy consumption reduction across multi-node H200 GPU clusters. These optimizations are model- and hardware-agnostic, enabling broad applicability across transformer architectures and extending to post-training phases including supervised fine-tuning and direct preference optimization. 

**Abstract (ZH)**: 针对大规模语言模型（LLMs）训练长时间和巨大能源消耗的问题，我们提出了Litespark，一种通过针对变压器注意力层和MLP层的优化来解决这些低效性的新颖预训练框架。我们的方法结合了架构改进与算法增强，以最大化模型FLOPs利用率（MFU）的同时，保持与标准变压器实现的兼容性。使用SlimPajama-627B数据集在3B和30B参数Llama模型上的全面基准测试表明，这些优化带来了显著的性能提升：多节点H200 GPU集群下的训练吞吐量提高1-3倍和能源消耗降低55%-83%。这些优化对模型和硬件具有普适性，可广泛应用于变压器架构的不同阶段，包括监督微调和直接偏好优化。 

---
# How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models 

**Title (ZH)**: 如何训练你的导师：通过顾问模型引导黑盒大语言模型 

**Authors**: Parth Asawa, Alan Zhu, Matei Zaharia, Alexandros G. Dimakis, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2510.02453)  

**Abstract**: Foundation models are increasingly deployed as black-box services, where model weights cannot be modified and customization is limited to prompting. While static prompt optimization has shown promise, it produces a single fixed prompt that fails to adapt to different inputs, users, or environments. We introduce Advisor Models, lightweight parametric policies trained with reinforcement learning to reactively issue natural language steering instructions in-context to black-box models. The advisor is a second small model that sits between the input and the model, shaping behavior on a per-instance basis using reward signals from the environment. Across multiple domains involving reasoning and personalization, we show that Advisor Models outperform static prompt optimizers, discovering environment dynamics and improving downstream task performance. We also demonstrate the generalizability of advisors by transferring them across black-box models, as well as the framework's ability to achieve specialization while retaining robustness to out-of-distribution inputs. Viewed more broadly, Advisor Models provide a learnable interface to black-box systems where the advisor acts as a parametric, environment-specific memory. We argue that dynamic optimization of black-box models via Advisor Models is a promising direction for enabling personalization and environment-adaptable AI with frontier-level capabilities. 

**Abstract (ZH)**: 基于Advisor模型的反应式自然语言引导方法在黑盒模型中的应用及其性能优势 

---
# Dynamic Target Attack 

**Title (ZH)**: 动态目标攻击 

**Authors**: Kedong Xiu, Churui Zeng, Tianhang Zheng, Xinzhe Huang, Xiaojun Jia, Di Wang, Puning Zhao, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.02422)  

**Abstract**: Existing gradient-based jailbreak attacks typically optimize an adversarial suffix to induce a fixed affirmative response. However, this fixed target usually resides in an extremely low-density region of a safety-aligned LLM's output distribution conditioned on diverse harmful inputs. Due to the substantial discrepancy between the target and the original output, existing attacks require numerous iterations to optimize the adversarial prompt, which might still fail to induce the low-probability target response from the target LLM. In this paper, we propose Dynamic Target Attack (DTA), a new jailbreaking framework relying on the target LLM's own responses as targets to optimize the adversarial prompts. In each optimization round, DTA iteratively samples multiple candidate responses directly from the output distribution conditioned on the current prompt, and selects the most harmful response as a temporary target for prompt optimization. In contrast to existing attacks, DTA significantly reduces the discrepancy between the target and the output distribution, substantially easing the optimization process to search for an effective adversarial prompt.
Extensive experiments demonstrate the superior effectiveness and efficiency of DTA: under the white-box setting, DTA only needs 200 optimization iterations to achieve an average attack success rate (ASR) of over 87\% on recent safety-aligned LLMs, exceeding the state-of-the-art baselines by over 15\%. The time cost of DTA is 2-26 times less than existing baselines. Under the black-box setting, DTA uses Llama-3-8B-Instruct as a surrogate model for target sampling and achieves an ASR of 85\% against the black-box target model Llama-3-70B-Instruct, exceeding its counterparts by over 25\%. 

**Abstract (ZH)**: 动态目标攻击：一种新的基于目标LLM响应的 jailbreak 框架 

---
# CWM: An Open-Weights LLM for Research on Code Generation with World Models 

**Title (ZH)**: CWM：一种基于世界模型的开放权重LLM代码生成研究 

**Authors**: FAIR CodeGen team. Jade Copet, Quentin Carbonneaux, Gal Cohen, Jonas Gehring, Jacob Kahn, Jannik Kossen, Felix Kreuk, Emily McMilin, Michel Meyer, Yuxiang Wei, David Zhang, Kunhao Zheng, Jordi Armengol-Estapé, Pedram Bashiri, Maximilian Beck, Pierre Chambon, Abhishek Charnalia, Chris Cummins, Juliette Decugis, Zacharias V. Fisches, François Fleuret, Fabian Gloeckle, Alex Gu, Michael Hassid, Daniel Haziza, Badr Youbi Idrissi, Christian Keller, Rahul Kindi, Hugh Leather, Gallil Maimon, Aram Markosyan, Francisco Massa, Pierre-Emmanuel Mazaré, Vegard Mella, Naila Murray, Keyur Muzumdar, Peter O'Hearn, Matteo Pagliardini, Dmitrii Pedchenko, Tal Remez, Volker Seeker, Marco Selvi, Oren Sultan, Sida Wang, Luca Wehrstedt, Ori Yoran, Lingming Zhang, Taco Cohen, Yossi Adi, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2510.02387)  

**Abstract**: We release Code World Model (CWM), a 32-billion-parameter open-weights LLM, to advance research on code generation with world models. To improve code understanding beyond what can be learned from training on static code alone, we mid-train CWM on a large amount of observation-action trajectories from Python interpreter and agentic Docker environments, and perform extensive multi-task reasoning RL in verifiable coding, math, and multi-turn software engineering environments. With CWM, we provide a strong testbed for researchers to explore the opportunities world modeling affords for improving code generation with reasoning and planning in computational environments. We present first steps of how world models can benefit agentic coding, enable step-by-step simulation of Python code execution, and show early results of how reasoning can benefit from the latter. CWM is a dense, decoder-only LLM trained with a context size of up to 131k tokens. Independent of its world modeling capabilities, CWM offers strong performance on general coding and math tasks: it reaches pass@1 scores of 65.8% on SWE-bench Verified (with test-time scaling), 68.6% on LiveCodeBench, 96.6% on Math-500, and 76.0% on AIME 2024. To support further research on code world modeling, we release model checkpoints after mid-training, SFT, and RL. 

**Abstract (ZH)**: 我们发布了一个320亿参数的开放权重大型语言模型Code World Model (CWM)，以推进基于世界模型的代码生成研究。为了超越仅通过静态代码训练所能学到的代码理解，我们在大量Python解释器和代理Docker环境的观察-动作轨迹上对CWM进行中期训练，并在可验证编程、数学以及多轮软件工程环境中进行广泛的多任务推理强化学习。利用CWM，我们为研究人员提供了一个强大的实验平台，以探索世界模型在计算环境中通过推理和规划提高代码生成的机会。我们展示了世界模型如何惠及代理编程，实现Python代码执行的逐步模拟，并展示了推理如何从中获益的初步结果。CWM是一个密集型、仅解码器的大型语言模型，使用最多131k词元的上下文进行训练。除其世界建模能力外，CWM在通用编程和数学任务上表现出强劲性能：在SWE-bench Verified上达到65.8%的pass@1得分（测试时缩放），在LiveCodeBench上达到68.6%，在Math-500上达到96.6%，在AIME 2024上达到76.0%。为了支持进一步的世界模型编码研究，我们在中期训练、微调和强化学习后发布了模型检查点。 

---
# Pretraining with hierarchical memories: separating long-tail and common knowledge 

**Title (ZH)**: 基于层次记忆的预训练：区分长尾知识和常见知识 

**Authors**: Hadi Pouransari, David Grangier, C Thomas, Michael Kirchhof, Oncel Tuzel  

**Link**: [PDF](https://arxiv.org/pdf/2510.02375)  

**Abstract**: The impressive performance gains of modern language models currently rely on scaling parameters: larger models store more world knowledge and reason better. Yet compressing all world knowledge into parameters is unnecessary, as only a fraction is used per prompt, and impractical for edge devices with limited inference-time memory and compute. We address this shortcoming by a memory-augmented architecture and a pretraining strategy aligned with existing hardware paradigms. We introduce small language models that access large hierarchical parametric memory banks encoding world knowledge. During pretraining and inference, we fetch a small, context-dependent memory block and add it to the model. Our pretraining learns to store long-tail world knowledge in the memory parameters, while the small language model acts as an anchor capturing common knowledge and general reasoning abilities. Through trillion-token-scale experiments, we show significant gains: a 160M-parameters model augmented with an 18M-parameters memory fetched from a 4.6B memory bank obtains comparable performance to a regular model with more than 2x the parameters. Through extensive experiments, we study the optimal type and size of parametric memories in transformers, scaling them to over 21B parameters. We find that our proposed hierarchical feed-forward memories work robustly across transformer architectures, whether added during pretraining or post-hoc. 

**Abstract (ZH)**: 现代语言模型的 impressive performance gains 目前依赖于扩展参数量：更大的模型存储更多世界知识并能更好地推理。然而，将所有世界知识压缩到参数中是不必要的，因为每个提示仅使用其中一部分，且对于具有有限推理时内存和计算能力的边缘设备来说是不现实的。我们通过引入一种基于记忆的架构和与现有硬件范式对齐的预训练策略来解决这一 shortcomings。我们提出了一种小型语言模型，它可以访问包含世界知识的大规模分层次参数化记忆库。在预训练和推理过程中，我们获取一个小型、上下文相关的记忆块并将其添加到模型中。我们的预训练学习在记忆参数中存储长尾世界知识，而小型语言模型则作为锚点捕获常见知识和通用推理能力。通过万亿级别 Tokens 规模的实验，我们展示了显著的提升：一个使用来自46亿参数记忆块中的18亿参数记忆增强的1.6亿参数模型，在性能上与超过两倍参数的常规模型相当。通过广泛的实验，我们研究了变压器中参数化记忆的最优类型和大小，并将其扩展到超过210亿参数。我们发现，我们提出的分层次前馈记忆在各种变压器架构中都能稳健工作，无论是在预训练期间还是事后添加。 

---
# A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory 

**Title (ZH)**: A-MemGuard: 基于LLM的代理记忆的主动防御框架 

**Authors**: Qianshan Wei, Tengchao Yang, Yaochen Wang, Xinfeng Li, Lijun Li, Zhenfei Yin, Yi Zhan, Thorsten Holz, Zhiqiang Lin, XiaoFeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02373)  

**Abstract**: Large Language Model (LLM) agents use memory to learn from past interactions, enabling autonomous planning and decision-making in complex environments. However, this reliance on memory introduces a critical security risk: an adversary can inject seemingly harmless records into an agent's memory to manipulate its future behavior. This vulnerability is characterized by two core aspects: First, the malicious effect of injected records is only activated within a specific context, making them hard to detect when individual memory entries are audited in isolation. Second, once triggered, the manipulation can initiate a self-reinforcing error cycle: the corrupted outcome is stored as precedent, which not only amplifies the initial error but also progressively lowers the threshold for similar attacks in the future. To address these challenges, we introduce A-MemGuard (Agent-Memory Guard), the first proactive defense framework for LLM agent memory. The core idea of our work is the insight that memory itself must become both self-checking and self-correcting. Without modifying the agent's core architecture, A-MemGuard combines two mechanisms: (1) consensus-based validation, which detects anomalies by comparing reasoning paths derived from multiple related memories and (2) a dual-memory structure, where detected failures are distilled into ``lessons'' stored separately and consulted before future actions, breaking error cycles and enabling adaptation. Comprehensive evaluations on multiple benchmarks show that A-MemGuard effectively cuts attack success rates by over 95% while incurring a minimal utility cost. This work shifts LLM memory security from static filtering to a proactive, experience-driven model where defenses strengthen over time. Our code is available in this https URL 

**Abstract (ZH)**: 大型语言模型（LLM）代理利用记忆从以往互动中学习，使其能够在复杂环境中实现自主规划和决策。然而，对记忆的依赖引入了一个关键的安全风险：对手可以向代理的记忆中注入看似无害的记录，从而操控其未来行为。这种漏洞由两个核心方面构成：首先，注入记录的恶意效果仅在特定上下文中激活，使得在单独审计记忆条目时难以检测。其次，一旦触发，这种操控可以引发自我强化的错误循环：被篡改的结果作为先例存储，不仅放大了初始错误，还逐步降低了未来类似攻击的门槛。为应对这些挑战，我们提出了一种名为A-MemGuard（代理记忆卫士）的前瞻防御框架，这是首个针对LLM代理记忆的主动防御框架。我们工作的核心思想是认识到记忆本身必须能够自我检查和自我纠正。不修改代理的核心架构，A-MemGuard结合了两种机制：（1）基于共识的验证，通过比较来自多个相关记忆的推理路径来检测异常；（2）双记忆结构，其中检测到的失败被提炼为“教训”分别存储，并在未来的行动中咨询，从而打断错误循环并实现适应。对多个基准的全面评估结果显示，A-MemGuard在有效降低攻击成功率超过95%的同时，保持了极低的实用成本。这项工作将LLM记忆安全从静态过滤转变为一种随经验增强的前瞻模型。我们的代码可在以下网址获取。 

---
# Training Dynamics of Parametric and In-Context Knowledge Utilization in Language Models 

**Title (ZH)**: 参数知识利用与上下文知识利用在语言模型中的训练动态 

**Authors**: Minsung Kim, Dong-Kyum Kim, Jea Kwon, Nakyeong Yang, Kyomin Jung, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2510.02370)  

**Abstract**: Large language models often encounter conflicts between in-context knowledge retrieved at inference time and parametric knowledge acquired during pretraining. Models that accept external knowledge uncritically are vulnerable to misinformation, whereas models that adhere rigidly to parametric knowledge fail to benefit from retrieval. Despite the widespread adoption of retrieval-augmented generation, we still lack a systematic understanding of what shapes knowledge-arbitration strategies during training. This gap risks producing pretrained models with undesirable arbitration behaviors and, consequently, wasting substantial computational resources after the pretraining budget has already been spent. To address this problem, we present the first controlled study of how training conditions influence models' use of in-context and parametric knowledge, and how they arbitrate between them. We train transformer-based language models on a synthetic biographies corpus while systematically controlling various conditions. Our experiments reveal that intra-document repetition of facts fosters the development of both parametric and in-context capabilities. Moreover, training on a corpus that contains inconsistent information or distributional skew encourages models to develop robust strategies for leveraging parametric and in-context knowledge. Rather than viewing these non-ideal properties as artifacts to remove, our results indicate that they are important for learning robust arbitration. These insights offer concrete, empirical guidance for pretraining models that harmoniously integrate parametric and in-context knowledge. 

**Abstract (ZH)**: 大型语言模型常在推理时检索到的上下文环境知识与预训练中获得的参数知识之间存在冲突。不批判性地接受外部知识的模型容易受到误导信息的影响，而严格遵循参数知识的模型则无法充分利用检索功能。尽管检索增强生成已广为采用，但我们仍缺乏系统理解训练期间知识仲裁策略形成机制的认识。这一知识空白可能导致预训练模型出现不理想的仲裁行为，并在预训练预算已经花费的情况下浪费大量计算资源。为解决这一问题，我们首次从受控实验角度研究了训练条件如何影响模型对上下文环境知识和参数知识的利用及其仲裁策略。我们在一个合成的传记语料库上训练基于变换器的语言模型，系统地控制各种条件。实验结果显示，文献内部事实的重复有助于培养参数和上下文能力，而使用包含不一致信息或分布偏斜的语料库进行训练则鼓励模型发展出有效利用参数和上下文知识的稳健策略。我们的结果表明，这些非理想特性对于学习稳健的仲裁至关重要。这些洞见为和谐整合参数与上下文知识的预训练模型提供了具体的实证指导。 

---
# Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents 

**Title (ZH)**: 超越手册和任务：LLM代理的实例级上下文学习 

**Authors**: Kuntai Cai, Juncheng Liu, Xianglin Yang, Zhaojie Niu, Xiaokui Xiao, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02369)  

**Abstract**: Large language model (LLM) agents typically receive two kinds of context: (i) environment-level manuals that define interaction interfaces and global rules, and (ii) task-level guidance or demonstrations tied to specific goals. In this work, we identify a crucial but overlooked third type of context, instance-level context, which consists of verifiable and reusable facts tied to a specific environment instance, such as object locations, crafting recipes, and local rules. We argue that the absence of instance-level context is a common source of failure for LLM agents in complex tasks, as success often depends not only on reasoning over global rules or task prompts but also on making decisions based on precise and persistent facts. Acquiring such context requires more than memorization: the challenge lies in efficiently exploring, validating, and formatting these facts under tight interaction budgets. We formalize this problem as Instance-Level Context Learning (ILCL) and introduce our task-agnostic method to solve it. Our method performs a guided exploration, using a compact TODO forest to intelligently prioritize its next actions and a lightweight plan-act-extract loop to execute them. This process automatically produces a high-precision context document that is reusable across many downstream tasks and agents, thereby amortizing the initial exploration cost. Experiments across TextWorld, ALFWorld, and Crafter demonstrate consistent gains in both success and efficiency: for instance, ReAct's mean success rate in TextWorld rises from 37% to 95%, while IGE improves from 81% to 95%. By transforming one-off exploration into persistent, reusable knowledge, our method complements existing contexts to enable more reliable and efficient LLM agents. 

**Abstract (ZH)**: 大型语言模型代理中的实例级上下文学习 

---
# A Cross-Lingual Analysis of Bias in Large Language Models Using Romanian History 

**Title (ZH)**: 跨语言视角下大型语言模型中关于罗马尼亚历史的偏见分析 

**Authors**: Matei-Iulian Cocu, Răzvan-Cosmin Cristia, Adrian Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2510.02362)  

**Abstract**: In this case study, we select a set of controversial Romanian historical questions and ask multiple Large Language Models to answer them across languages and contexts, in order to assess their biases. Besides being a study mainly performed for educational purposes, the motivation also lies in the recognition that history is often presented through altered perspectives, primarily influenced by the culture and ideals of a state, even through large language models. Since they are often trained on certain data sets that may present certain ambiguities, the lack of neutrality is subsequently instilled in users. The research process was carried out in three stages, to confirm the idea that the type of response expected can influence, to a certain extent, the response itself; after providing an affirmative answer to some given question, an LLM could shift its way of thinking after being asked the same question again, but being told to respond with a numerical value of a scale. Results show that binary response stability is relatively high but far from perfect and varies by language. Models often flip stance across languages or between formats; numeric ratings frequently diverge from the initial binary choice, and the most consistent models are not always those judged most accurate or neutral. Our research brings to light the predisposition of models to such inconsistencies, within a specific contextualization of the language for the question asked. 

**Abstract (ZH)**: 在这个案例研究中，我们选择了一组有争议的 Romanian 历史问题，要求多种大型语言模型在不同语言和背景下作答，以评估其偏见。 

---
# ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inference 

**Title (ZH)**: ChunkLLM：一种加速大型语言模型推理的轻量级插件框架 

**Authors**: Haojie Ouyang, Jianwei Lv, Lei Ren, Chen Wei, Xiaojie Wang, Fangxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02361)  

**Abstract**: Transformer-based large models excel in natural language processing and computer vision, but face severe computational inefficiencies due to the self-attention's quadratic complexity with input tokens. Recently, researchers have proposed a series of methods based on block selection and compression to alleviate this problem, but they either have issues with semantic incompleteness or poor training-inference efficiency. To comprehensively address these challenges, we propose ChunkLLM, a lightweight and pluggable training framework. Specifically, we introduce two components: QK Adapter (Q-Adapter and K-Adapter) and Chunk Adapter. The former is attached to each Transformer layer, serving dual purposes of feature compression and chunk attention acquisition. The latter operates at the bottommost layer of the model, functioning to detect chunk boundaries by leveraging contextual semantic information. During the training phase, the parameters of the backbone remain frozen, with only the QK Adapter and Chunk Adapter undergoing training. Notably, we design an attention distillation method for training the QK Adapter, which enhances the recall rate of key chunks. During the inference phase, chunk selection is triggered exclusively when the current token is detected as a chunk boundary, thereby accelerating model inference. Experimental evaluations are conducted on a diverse set of long-text and short-text benchmark datasets spanning multiple tasks. ChunkLLM not only attains comparable performance on short-text benchmarks but also maintains 98.64% of the performance on long-context benchmarks while preserving a 48.58% key-value cache retention rate. Particularly, ChunkLLM attains a maximum speedup of 4.48x in comparison to the vanilla Transformer in the processing of 120K long texts. 

**Abstract (ZH)**: 基于块选择和压缩的Transformer大模型在自然语言处理和计算机视觉中的高效训练框架：ChunkLLM 

---
# Spiral of Silence in Large Language Model Agents 

**Title (ZH)**: 大型语言模型代理的沉默螺旋效应 

**Authors**: Mingze Zhong, Meng Fang, Zijing Shi, Yuxuan Huang, Shunfeng Zheng, Yali Du, Ling Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02360)  

**Abstract**: The Spiral of Silence (SoS) theory holds that individuals with minority views often refrain from speaking out for fear of social isolation, enabling majority positions to dominate public discourse. When the 'agents' are large language models (LLMs), however, the classical psychological explanation is not directly applicable, since SoS was developed for human societies. This raises a central question: can SoS-like dynamics nevertheless emerge from purely statistical language generation in LLM collectives? We propose an evaluation framework for examining SoS in LLM agents. Specifically, we consider four controlled conditions that systematically vary the availability of 'History' and 'Persona' signals. Opinion dynamics are assessed using trend tests such as Mann-Kendall and Spearman's rank, along with concentration measures including kurtosis and interquartile range. Experiments across open-source and closed-source models show that history and persona together produce strong majority dominance and replicate SoS patterns; history signals alone induce strong anchoring; and persona signals alone foster diverse but uncorrelated opinions, indicating that without historical anchoring, SoS dynamics cannot emerge. The work bridges computational sociology and responsible AI design, highlighting the need to monitor and mitigate emergent conformity in LLM-agent systems. 

**Abstract (ZH)**: 螺旋静默效应在大规模语言模型中的统计语言生成动态及其评价框架 

---
# Emission-GPT: A domain-specific language model agent for knowledge retrieval, emission inventory and data analysis 

**Title (ZH)**: Emission-GPT: 一个专门领域语言模型代理，用于知识检索、排放清单编制和数据解析 

**Authors**: Jiashu Ye, Tong Wu, Weiwen Chen, Hao Zhang, Zeteng Lin, Xingxing Li, Shujuan Weng, Manni Zhu, Xin Yuan, Xinlong Hong, Jingjie Li, Junyu Zheng, Zhijiong Huang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02359)  

**Abstract**: Improving air quality and addressing climate change relies on accurate understanding and analysis of air pollutant and greenhouse gas emissions. However, emission-related knowledge is often fragmented and highly specialized, while existing methods for accessing and compiling emissions data remain inefficient. These issues hinder the ability of non-experts to interpret emissions information, posing challenges to research and management. To address this, we present Emission-GPT, a knowledge-enhanced large language model agent tailored for the atmospheric emissions domain. Built on a curated knowledge base of over 10,000 documents (including standards, reports, guidebooks, and peer-reviewed literature), Emission-GPT integrates prompt engineering and question completion to support accurate domain-specific question answering. Emission-GPT also enables users to interactively analyze emissions data via natural language, such as querying and visualizing inventories, analyzing source contributions, and recommending emission factors for user-defined scenarios. A case study in Guangdong Province demonstrates that Emission-GPT can extract key insights--such as point source distributions and sectoral trends--directly from raw data with simple prompts. Its modular and extensible architecture facilitates automation of traditionally manual workflows, positioning Emission-GPT as a foundational tool for next-generation emission inventory development and scenario-based assessment. 

**Abstract (ZH)**: 提高空气质量及应对气候变化依赖于对空气污染物和温室气体排放的准确理解和分析。然而，与排放相关的知识往往是碎片化的和高度专门化的，而现有的排放数据获取和汇总方法仍然不够高效。这些问题阻碍了非专家解读排放信息的能力，给研究和管理带来了挑战。为解决这一问题，我们提出了Emission-GPT，这是一种面向大气排放领域的知识增强型大语言模型代理。Emission-GPT基于包含超过10,000份文件（包括标准、报告、指南和同行评审文献）的精心筛选知识库，通过提示工程和问题完成来支持准确的领域特定问题回答。Emission-GPT还允许用户通过自然语言交互式分析排放数据，例如查询和可视化清单、分析源贡献以及为用户自定义场景推荐排放因子。在广东省的案例研究中展示了，通过简单的提示，Emission-GPT可以从原始数据中提取关键见解，如点源分布和行业趋势。其模块化的可扩展架构使得传统手动工作流程的自动化成为可能，将Emission-GPT定位为下一代排放清单开发和基于场景评估的基本工具。 

---
# DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding 

**Title (ZH)**: DiffuSpec: 解锁用于推测性解码的扩散语言模型 

**Authors**: Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02358)  

**Abstract**: As large language models (LLMs) scale up, accuracy improves, but the autoregressive (AR) nature of decoding increases latency since each token requires a serial forward pass. Speculative decoding addresses this by employing a fast drafter to propose multi-token drafts, which are then verified in parallel by the target model. However, many deployments still rely on AR drafters, where sequential passes limit wall-clock gains. We revisit the drafting stage and present DiffuSpec, a training-free drop-in framework that uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers. Because DLM drafts are generated under bidirectional conditioning, parallel per-position candidates form a token lattice in which the locally highest-probability token at each position need not form a causal left-to-right path. Moreover, DLM drafting requires pre-specifying a draft length, inducing a speed-quality trade-off. To address these challenges, we introduce two practical components: (i) a causal-consistency path search (CPS) over this lattice that extracts a left-to-right path aligned with AR verification; and (ii) an adaptive draft-length (ADL) controller that adjusts next proposal size based on recent acceptance feedback and realized generated length. Across benchmarks, DiffuSpec yields up to 3x wall-clock speedup, establishing diffusion-based drafting as a robust alternative to autoregressive drafters for speculative decoding. 

**Abstract (ZH)**: 基于扩散模型的 speculative 解码框架：DiffuSpec 

---
# Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations 

**Title (ZH)**: 评估口语对话大语言模型在实际决策和推荐中的偏见 

**Authors**: Yihao Wu, Tianrui Wang, Yizhou Peng, Yi-Wen Chao, Xuyi Zhuang, Xinsheng Wang, Shunshun Yin, Ziyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.02352)  

**Abstract**: While biases in large language models (LLMs), such as stereotypes and cultural tendencies in outputs, have been examined and identified, their presence and characteristics in spoken dialogue models (SDMs) with audio input and output remain largely unexplored. Paralinguistic features, such as age, gender, and accent, can affect model outputs; when compounded by multi-turn conversations, these effects may exacerbate biases, with potential implications for fairness in decision-making and recommendation tasks. In this paper, we systematically evaluate biases in speech LLMs and study the impact of multi-turn dialogues with repeated negative feedback. Bias is measured using Group Unfairness Score (GUS) for decisions and similarity-based normalized statistics rate (SNSR) for recommendations, across both open-source models like Qwen2.5-Omni and GLM-4-Voice, as well as closed-source APIs such as GPT-4o Audio and Gemini-2.5-Flash. Our analysis reveals that closed-source models generally exhibit lower bias, while open-source models are more sensitive to age and gender, and recommendation tasks tend to amplify cross-group disparities. We found that biased decisions may persist in multi-turn conversations. This work provides the first systematic study of biases in end-to-end spoken dialogue models, offering insights towards fair and reliable audio-based interactive systems. To facilitate further research, we release the FairDialogue dataset and evaluation code. 

**Abstract (ZH)**: 大型语言模型中的偏见在输出中已有所研究，但音频输入和输出的对话语言模型（SDMs）中的偏见及其特性仍 largely unexplored。本论文系统地评估了语音大语言模型中的偏见，并研究了带有重复负面反馈的多轮对话的影响。偏见通过决策的组不公平得分（GUS）和推荐的基于相似性的标准化统计数据率（SNSR）进行衡量，涵盖了开源模型如Qwen2.5-Omni和GLM-4-Voice，以及封闭源API如GPT-4o Audio和Gemini-2.5-Flash。我们的分析发现，封闭源模型通常表现出较低的偏见，而开源模型对年龄和性别更为敏感，推荐任务会放大跨群体差异。我们发现，在多轮对话中，有偏见的决策仍可能持续存在。本工作首次系统地研究了端到端语音对话模型中的偏见，为公正可靠的基于音频的交互系统提供了见解。为促进进一步研究，我们发布了FairDialogue数据集和评估代码。 

---
# Language, Culture, and Ideology: Personalizing Offensiveness Detection in Political Tweets with Reasoning LLMs 

**Title (ZH)**: 语言、文化与意识形态：利用推理大型语言模型个性化检测政治推文的冒犯性 

**Authors**: Dzmitry Pihulski, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2510.02351)  

**Abstract**: We explore how large language models (LLMs) assess offensiveness in political discourse when prompted to adopt specific political and cultural perspectives. Using a multilingual subset of the MD-Agreement dataset centered on tweets from the 2020 US elections, we evaluate several recent LLMs - including DeepSeek-R1, o4-mini, GPT-4.1-mini, Qwen3, Gemma, and Mistral - tasked with judging tweets as offensive or non-offensive from the viewpoints of varied political personas (far-right, conservative, centrist, progressive) across English, Polish, and Russian contexts. Our results show that larger models with explicit reasoning abilities (e.g., DeepSeek-R1, o4-mini) are more consistent and sensitive to ideological and cultural variation, while smaller models often fail to capture subtle distinctions. We find that reasoning capabilities significantly improve both the personalization and interpretability of offensiveness judgments, suggesting that such mechanisms are key to adapting LLMs for nuanced sociopolitical text classification across languages and ideologies. 

**Abstract (ZH)**: 我们探索大型语言模型（LLMs）在采用特定政治和文化视角时如何评估政治 discourse中的冒犯性。我们利用旨在关注2020年美国选举推文的MD-Agreement数据集的多语种子集，评估了几种最近的LLM——包括DeepSeek-R1、o4-mini、GPT-4.1-mini、Qwen3、Gemma和Mistral——这些模型的任务是从不同的政治人格（极右、保守、中间派、进步派）视角判断推文是否冒犯。结果显示，具有显式推理能力的大型模型（如DeepSeek-R1、o4-mini）在意识形态和文化差异方面更具一致性和敏感性，而较小的模型往往难以捕捉微妙的区别。我们发现，推理能力显著提高了冒犯性判断的个性化和可解释性，表明此类机制是使LLM适应跨语言和意识形态的精细社会政治文本分类的关键。 

---
# LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL 

**Title (ZH)**: LLMSQL: 升级的WikiSQL以适应大语言模型时代的文本到SQL任务 

**Authors**: Dzmitry Pihulski, Karol Charchut, Viktoria Novogrodskaia, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2510.02350)  

**Abstract**: Converting natural language questions into SQL queries (Text-to-SQL) enables non-expert users to interact with relational databases and has long been a central task for natural language interfaces to data. While the WikiSQL dataset played a key role in early NL2SQL research, its usage has declined due to structural and annotation issues, including case sensitivity inconsistencies, data type mismatches, syntax errors, and unanswered questions. We present LLMSQL, a systematic revision and transformation of WikiSQL designed for the LLM era. We classify these errors and implement automated methods for cleaning and re-annotation. To assess the impact of these improvements, we evaluated multiple large language models (LLMs), including Gemma 3, LLaMA 3.2, Mistral 7B, gpt-oss 20B, Phi-3.5 Mini, Qwen 2.5, OpenAI o4-mini, DeepSeek R1 and others. Rather than serving as an update, LLMSQL is introduced as an LLM-ready benchmark: unlike the original WikiSQL, tailored for pointer-network models selecting tokens from input, LLMSQL provides clean natural language questions and full SQL queries as plain text, enabling straightforward generation and evaluation for modern natural language-to-SQL models. 

**Abstract (ZH)**: 将自然语言问题转换为SQL查询（Text-to-SQL）使非专业用户能够与关系数据库交互，并且一直是数据自然语言接口中的核心任务。尽管WikiSQL数据集在早期的NL2SQL研究中起到了关键作用，但其使用量由于结构和注释问题，包括大小写一致性问题、数据类型不匹配、语法错误以及答案缺失等原因而下降。我们提出LLMSQL，这是一个为LLM时代设计的系统性修订和转换的WikiSQL数据集。我们对这些错误进行了分类，并实现了自动化的清洁和重新注释方法。为了评估这些改进的影响，我们评估了多个大型语言模型（LLM），包括Gemma 3、LLaMA 3.2、Mistral 7B、gpt-oss 20B、Phi-3.5 Mini、Qwen 2.5、OpenAI o4-mini、DeepSeek R1等。不同于作为更新，LLMSQL被介绍为一个适用于LLM的基准：不同于原始的WikiSQL，LLMSQL为现代自然语言到SQL模型的生成和评估提供了干净的自然语言问题和完整的SQL查询文本。 

---
# Small Language Models for Curriculum-based Guidance 

**Title (ZH)**: 基于课程的学习小语言模型指导 

**Authors**: Konstantinos Katharakis, Sippo Rossi, Raghava Rao Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2510.02347)  

**Abstract**: The adoption of generative AI and large language models (LLMs) in education is still emerging. In this study, we explore the development and evaluation of AI teaching assistants that provide curriculum-based guidance using a retrieval-augmented generation (RAG) pipeline applied to selected open-source small language models (SLMs). We benchmarked eight SLMs, including LLaMA 3.1, IBM Granite 3.3, and Gemma 3 (7-17B parameters), against GPT-4o. Our findings show that with proper prompting and targeted retrieval, SLMs can match LLMs in delivering accurate, pedagogically aligned responses. Importantly, SLMs offer significant sustainability benefits due to their lower computational and energy requirements, enabling real-time use on consumer-grade hardware without depending on cloud infrastructure. This makes them not only cost-effective and privacy-preserving but also environmentally responsible, positioning them as viable AI teaching assistants for educational institutions aiming to scale personalized learning in a sustainable and energy-efficient manner. 

**Abstract (ZH)**: 生成式人工智能和大型语言模型在教育中的采用仍处于新兴阶段。本研究探索了使用检索增强生成（RAG） pipeline在选定的开源小型语言模型（SLMs）上开发和评估基于 Curriculum 的教学助手的方法。我们使用GPT-4o与八种SLM进行基准测试，包括LLaMA 3.1、IBM Granite 3.3和Gemma 3（7-17B参数）。研究发现，通过适当的提示和有针对性的检索，SLMs可以在提供精准且符合教学要求的响应方面与LLMs匹敌。重要的是，SLMs由于计算和能源需求较低，可以在消费级硬件上实现实时使用，无需依赖云基础设施，从而使其在成本效益、保护隐私和环保方面具有优势，将其定位为致力于以可持续且能效高的方式规模化个性化学习的教育机构的可行的教学助手。 

---
# Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression 

**Title (ZH)**: 打破MoE大语言模型的三难困境：动态专家聚类结合结构化压缩 

**Authors**: Peijun Zhu, Ning Yang, Jiayu Wei, Jinghang Wu, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02345)  

**Abstract**: Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs. 

**Abstract (ZH)**: 混合专家（MoE）大型语言模型（LLM）面临负载不平衡、参数冗余和通信开销三者的权衡问题。我们提出了一种基于动态专家聚类和结构化压缩的统一框架，以统筹解决这些问题。该方法采用了一种在线聚类过程，定期使用参数和激活相似性的融合度量重新分组专家，从而稳定专家利用情况。据我们所知，这是第一个利用路由器的语义嵌入能力，在训练期间动态重构模型架构以实现显著效率提升的框架。在每个聚类内，我们将专家权重分解为共享的基本矩阵和极低秩的残留适配器，每个组可实现五倍量级的参数减少，同时保持专业性。这种结构使得路由策略可以分两阶段进行层级化：首先将标记分配到聚类，然后分配到其内的特定专家，极大减少了路由搜索空间和全连接通信的体积。此外，采用异构精度方案，将共享基存储为FP16，并将残留因子存储为INT4，结合动态卸载不活动聚类，显著降低了峰值内存消耗，使其与密集模型水平相当。在GLUE和WikiText-103上评估，我们的框架在减少总参数约80%的情况下，提高了10%到20%的吞吐量，并将专家负载变异度降低了超过三倍。我们的工作表明，结构重组是实现可扩展、高效且内存有效的MoE LLMs的一条原理性路径。 

---
# $\texttt{BluePrint}$: A Social Media User Dataset for LLM Persona Evaluation and Training 

**Title (ZH)**: BluePrint: 一种社交媒体用户数据集，用于LLM人格评估与训练 

**Authors**: Aurélien Bück-Kaeffer, Je Qin Chooi, Dan Zhao, Maximilian Puelma Touzel, Kellin Pelrine, Jean-François Godbout, Reihaneh Rabbany, Zachary Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02343)  

**Abstract**: Large language models (LLMs) offer promising capabilities for simulating social media dynamics at scale, enabling studies that would be ethically or logistically challenging with human subjects. However, the field lacks standardized data resources for fine-tuning and evaluating LLMs as realistic social media agents. We address this gap by introducing SIMPACT, the SIMulation-oriented Persona and Action Capture Toolkit, a privacy respecting framework for constructing behaviorally-grounded social media datasets suitable for training agent models. We formulate next-action prediction as a task for training and evaluating LLM-based agents and introduce metrics at both the cluster and population levels to assess behavioral fidelity and stylistic realism. As a concrete implementation, we release BluePrint, a large-scale dataset built from public Bluesky data focused on political discourse. BluePrint clusters anonymized users into personas of aggregated behaviours, capturing authentic engagement patterns while safeguarding privacy through pseudonymization and removal of personally identifiable information. The dataset includes a sizable action set of 12 social media interaction types (likes, replies, reposts, etc.), each instance tied to the posting activity preceding it. This supports the development of agents that use context-dependence, not only in the language, but also in the interaction behaviours of social media to model social media users. By standardizing data and evaluation protocols, SIMPACT provides a foundation for advancing rigorous, ethically responsible social media simulations. BluePrint serves as both an evaluation benchmark for political discourse modeling and a template for building domain specific datasets to study challenges such as misinformation and polarization. 

**Abstract (ZH)**: 大型语言模型（LLMs）提供了模拟大规模社交媒体动态的前景能力，使得使用人类受试者进行的研究在伦理和组织方面更具挑战性。然而，领域内缺乏用于微调和评估LLMs作为真实社交媒体代理的标准数据资源。我们通过引入SIMPACT（SIMulation-oriented Persona and Action Capture Toolkit）来填补这一空白，SIMPACT是一个尊重隐私的框架，用于构建行为上一致的社交媒体数据集，适用于训练代理模型。我们将下一步行动预测作为训练和评估LLM基代理的任务，并介绍了群组和人口层面的评估指标来衡量行为真实性和风格的现实性。作为具体的实现，我们发布了BluePrint，一个基于公共Bluesky数据的大规模数据集，侧重于政治讨论。BluePrint将匿名用户聚类成行为聚合的人格，通过假名化和去除个人可识别信息来保护隐私，捕获真实的参与模式。数据集包括12种社交媒体互动类型（点赞、回复、转发等）的操作集，每个实例都与其之前的发布活动相关。这支持了不仅在语言，还在社交媒体互动行为上具有情境依赖性的代理模型的发展。通过标准化数据和评估协议，SIMPACT为推动严格的、负责任的社交媒体模拟提供了基础。BluePrint既作为政治讨论建模的评估基准，也为研究信息误导和极化等特定领域挑战提供了模板。 

---
# CATMark: A Context-Aware Thresholding Framework for Robust Cross-Task Watermarking in Large Language Models 

**Title (ZH)**: CATMark：一种面向大规模语言模型跨任务 robust 水标记有的上下文感知阈值框架 

**Authors**: Yu Zhang, Shuliang Liu, Xu Yang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02342)  

**Abstract**: Watermarking algorithms for Large Language Models (LLMs) effectively identify machine-generated content by embedding and detecting hidden statistical features in text. However, such embedding leads to a decline in text quality, especially in low-entropy scenarios where performance needs improvement. Existing methods that rely on entropy thresholds often require significant computational resources for tuning and demonstrate poor adaptability to unknown or cross-task generation scenarios. We propose \textbf{C}ontext-\textbf{A}ware \textbf{T}hreshold watermarking ($\myalgo$), a novel framework that dynamically adjusts watermarking intensity based on real-time semantic context. $\myalgo$ partitions text generation into semantic states using logits clustering, establishing context-aware entropy thresholds that preserve fidelity in structured content while embedding robust watermarks. Crucially, it requires no pre-defined thresholds or task-specific tuning. Experiments show $\myalgo$ improves text quality in cross-tasks without sacrificing detection accuracy. 

**Abstract (ZH)**: Large Language Models（LLMs）的水印算法通过嵌入和检测文本中的隐藏统计特征有效识别机器生成的内容。然而，这种嵌入会导致文本质量下降，尤其是在低熵场景中性能需要提升的情况下。现有的依赖于熵阈值的方法往往需要大量计算资源进行调优，并且在未知或跨任务生成场景中表现不佳。我们提出了一种新的框架——\textbf{C}ontext-\textbf{A}ware \textbf{T}hreshold 水印（$\myalgo$），该框架能够基于实时语义上下文动态调整水印强度。$\myalgo$ 使用 logits 聚类将文本生成划分为语义状态，并建立语境感知的熵阈值，在保持结构化内容保真度的同时嵌入稳健的水印。 crucially，它不需要预定义的阈值或特定任务的调优。实验结果显示，$\myalgo$ 能在跨任务中提高文本质量而不牺牲检测准确性。 

---
# DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning 

**Title (ZH)**: DRIFT：从现实世界偏好学习中丰富的用户不满中学习 

**Authors**: Yifan Wang, Bolian Li, Junlin Wu, Zhaoxuan Tan, Zheli Liu, Ruqi Zhang, Ananth Grama, Qingkai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02341)  

**Abstract**: Real-world large language model deployments (e.g., conversational AI systems, code generation assistants) naturally generate abundant implicit user dissatisfaction (DSAT) signals, as users iterate toward better answers through refinements, corrections, and expressed preferences, while explicit satisfaction (SAT) feedback is scarce. Existing preference learning approaches are poorly aligned with this data profile, as they rely on costly human annotations or assume plentiful positive responses. In this paper, we introduce \textbf{DRIFT} (\textbf{D}issatisfaction-\textbf{R}efined \textbf{I}terative pre\textbf{F}erence \textbf{T}raining), which anchors training on real-world DSAT signals and samples positives dynamically from the evolving policy. Empirically, DRIFT models trained on real-world \textit{WildFeedback} datasets and synthetic \textit{UltraFeedback} datasets achieve up to +6.23\% (7B) / +7.61\% (14B) on WildBench Task Score and up to +8.95\% (7B) / +12.29\% (14B) on AlpacaEval2 win rate over base models, outperforming strong baseline methods such as iterative DPO and SPIN. At larger scales, the improvements are particularly pronounced: 14B models trained with DRIFT surpass GPT-4o-mini on WildBench. Further analysis shows that DRIFT also preserves exploratory capacity, yielding more diverse high-reward solutions rather than collapsing to narrow subsets. Theoretically, we demonstrate that this design preserves preference margins and avoids the gradient degeneration. These results show that DRIFT is an effective and scalable recipe for real-world post-training that leverages the most abundant and informative signal. The code and data are available at this https URL. 

**Abstract (ZH)**: 实世界大规模语言模型部署中的隐式用户不满意信号驱动的迭代偏好训练（基于真实世界和合成数据集的DRIFT模型） 

---
# Evaluating Uncertainty Quantification Methods in Argumentative Large Language Models 

**Title (ZH)**: 评估论辩型大型语言模型中的不确定性量化方法 

**Authors**: Kevin Zhou, Adam Dejl, Gabriel Freedman, Lihu Chen, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2510.02339)  

**Abstract**: Research in uncertainty quantification (UQ) for large language models (LLMs) is increasingly important towards guaranteeing the reliability of this groundbreaking technology. We explore the integration of LLM UQ methods in argumentative LLMs (ArgLLMs), an explainable LLM framework for decision-making based on computational argumentation in which UQ plays a critical role. We conduct experiments to evaluate ArgLLMs' performance on claim verification tasks when using different LLM UQ methods, inherently performing an assessment of the UQ methods' effectiveness. Moreover, the experimental procedure itself is a novel way of evaluating the effectiveness of UQ methods, especially when intricate and potentially contentious statements are present. Our results demonstrate that, despite its simplicity, direct prompting is an effective UQ strategy in ArgLLMs, outperforming considerably more complex approaches. 

**Abstract (ZH)**: 大型语言模型(LLMs)不确定性量化(UQ)研究 increasingly important towards guaranteeing the reliability of this groundbreaking technology: 探索将LLM UQ方法集成到论辩型LLM(ArgLLMs)中的不确定性量化在基于计算论辩的决策框架中的作用及其评估 

---
# Optimizing Long-Form Clinical Text Generation with Claim-Based Rewards 

**Title (ZH)**: 基于主张的奖励优化长格式临床文本生成 

**Authors**: Samyak Jhaveri, Praphul Singh, Jangwon Kim, Tara Taghavi, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02338)  

**Abstract**: Automating clinical documentation with large language models requires precise alignment with priorities such as completeness and factual grounding. We present an evaluation-integrated reinforcement learning framework for long-form clinical text generation that couples Group Relative Policy Optimization (GRPO) with DocLens, a claim-level evaluator that provides deterministic, dialogue-grounded rewards. Our method directly optimizes factual grounding and completeness without training a separate reward model or relying on human-authored references. Empirically, the approach improves clinical note quality and reduces training cost via a simple reward-gating strategy. An independent GPT-5 qualitative evaluation further supports these gains, showing higher preference for GRPO outputs in factuality, completeness, and brevity, with fewer omissions and hallucinations. Because the benchmarks are relatively clean and the base model already well aligned, these improvements likely represent a conservative lower bound. The framework is scalable to real-world settings and can incorporate custom objectives such as guideline adherence or billing preferences. 

**Abstract (ZH)**: 利用大型语言模型自动化临床文档生成要求与完备性和事实基础等优先事项精确对齐。我们提出了一种结合组相对策略优化（GRPO）和基于声明的评估器DocLens的评估集成强化学习框架，用于长格式临床文本生成。该方法直接优化事实基础和完备性，无需训练独立的奖励模型或依赖于人工撰写的参考文献。实验结果表明，该方法通过简单的奖励门控策略提高了临床笔记的质量并降低了训练成本。独立的GPT-5定性评估进一步支持了这些改进，表明GRPO输出在事实性、完备性和简洁性方面获得了更高的偏好，且遗漏和虚构较少。由于基准数据相对干净且基础模型已很好地对齐，这些改进可能代表了一个保守的下限。该框架可以扩展到实际应用场景，并可以纳入自定义目标，如指南遵循或收费偏好。 

---
# FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory 

**Title (ZH)**: FormalML：机器学习理论中形式子目标完成评估的标准基准 

**Authors**: Xiao-Wen Yang, Zihao Zhang, Jianuo Cao, Zhi Zhou, Zenan Li, Lan-Zhe Guo, Yuan Yao, Taolue Chen, Yu-Feng Li, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.02335)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable progress in formal theorem proving. Yet their ability to serve as practical assistants for mathematicians, filling in missing steps within complex proofs, remains underexplored. We identify this challenge as the task of subgoal completion, where an LLM must discharge short but nontrivial proof obligations left unresolved in a human-provided sketch. To study this problem, we introduce FormalML, a Lean 4 benchmark built from foundational theories of machine learning. Using a translation tactic that converts procedural proofs into declarative form, we extract 4937 problems spanning optimization and probability inequalities, with varying levels of difficulty. FormalML is the first subgoal completion benchmark to combine premise retrieval and complex research-level contexts. Evaluation of state-of-the-art provers highlights persistent limitations in accuracy and efficiency, underscoring the need for more capable LLM-based theorem provers for effective subgoal completion, 

**Abstract (ZH)**: 大型语言模型（LLMs）在形式定理证明方面 recently demonstrated remarkable progress.然而，它们作为数学家的实际助手，填补复杂证明中缺失的步骤的能力仍需进一步探索。我们将其挑战定义为子目标完成任务，即LLM必须在人类提供的草图中解决未解决的简短但非平凡的证明义务。为了研究这一问题，我们引入了FormalML，这是一个基于机器学习基础理论的Lean 4基准。通过将过程性证明转换为声明性形式的转换技巧，我们提取了4937个问题，涵盖了优化和概率不等式等多个不同难度级别。FormalML是第一个结合前提检索和复杂研究级上下文的子目标完成基准。对当前最先进的证明系统的评估揭示了在准确性和效率方面的一贯局限性，强调了需要更高效的基于LLM的证明系统以有效完成子目标完成任务。 

---
# Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing 

**Title (ZH)**: 哪里出了问题？通过表示梯度追踪归因不良LLM行为 

**Authors**: Zhe Li, Wei Zhao, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.02334)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their deployment is frequently undermined by undesirable behaviors such as generating harmful content, factual inaccuracies, and societal biases. Diagnosing the root causes of these failures poses a critical challenge for AI safety. Existing attribution methods, particularly those based on parameter gradients, often fall short due to prohibitive noisy signals and computational complexity. In this work, we introduce a novel and efficient framework that diagnoses a range of undesirable LLM behaviors by analyzing representation and its gradients, which operates directly in the model's activation space to provide a semantically meaningful signal linking outputs to their training data. We systematically evaluate our method for tasks that include tracking harmful content, detecting backdoor poisoning, and identifying knowledge contamination. The results demonstrate that our approach not only excels at sample-level attribution but also enables fine-grained token-level analysis, precisely identifying the specific samples and phrases that causally influence model behavior. This work provides a powerful diagnostic tool to understand, audit, and ultimately mitigate the risks associated with LLMs. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了显著的能力，但其部署常因生成有害内容、事实不准确和社会偏见等不良行为而受阻。诊断这些失败的根本原因是对AI安全的一个关键挑战。现有的归因方法，尤其是基于参数梯度的方法，往往由于噪声信号强和计算复杂性不足而表现不佳。在此工作中，我们提出了一种新型且高效的框架，通过分析表示及其梯度来诊断多种不良的LLM行为，该框架直接在模型的激活空间中操作，提供一种语义上具有意义的信号，将输出与其训练数据联系起来。我们系统地评估了该方法在包括追踪有害内容、检测后门污染和识别知识污染等任务中的性能。结果表明，我们的方法不仅在样本级归因方面表现出色，还能实现精细的令牌级分析，精确识别对模型行为有因果影响的具体样本和短语。这项工作提供了一种强大的诊断工具，以理解、审计并最终减轻LLMs相关的风险。代码可在以下链接获取。 

---
# A High-Capacity and Secure Disambiguation Algorithm for Neural Linguistic Steganography 

**Title (ZH)**: 高容量和安全的语义消歧算法研究：神经语言隐写术 

**Authors**: Yapei Feng, Feng Jiang, Shanhao Wu, Hua Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.02332)  

**Abstract**: Neural linguistic steganography aims to embed information
into natural text while preserving statistical undetectability. A fundamental challenge in this ffeld stems from tokenization ambiguity in modern tokenizers, which can lead to catastrophic decoding failures. The recent method, SyncPool, addresses this ambiguity
by employing a coarse-grained synchronization mechanism over groups of ambiguous candidates. However, SyncPool sacriffces embedding capacity, as it utilizes the entire Shannon entropy of an ambiguous group solely for synchronization rather than for payload embedding. We propose a method named look-ahead Sync, which overcomes the capacity limitation of SyncPool while retaining its provable security guarantees. Our approach performs minimal synchronized sampling only on truly indistinguishable token sequences, while strategically preserving all other discernible paths to maximize embedding capacity. We provide theoretical proofs for the security of our method and analyze the gap between its achievable embedding capacity and the theoretical upper bound. Experiments on English (using Llama 3) and Chinese (using Qwen 2.5) benchmarks show that our method consistently approaches the theoretical capacity upper bound and signiffcantly outperforms SyncPool. The improvement in embedding rate exceeds 160% in English and 25% in Chinese, particularly in settings with larger candidate pools. This work represents a signiffcant step toward practical high-capacity provably secure linguistic steganography. 

**Abstract (ZH)**: 神经语言隐写术旨在将信息嵌入自然文本中同时保持统计不可检测性。这一领域的一个基本挑战源自现代分词器中的分词模糊性，这可能导致灾难性的解码失败。近期方法SyncPool通过在一组模糊候选词上采用粗粒度同步机制来解决这一模糊性，但SyncPool牺牲了嵌入容量，因为它仅利用模糊组的整个香农熵来进行同步，而不是用于有效载荷嵌入。我们提出了一种名为前瞻Sync的方法，该方法克服了SyncPool的容量限制，同时保持其可证明的安全性保证。我们的方法仅在真正无法区分的令牌序列上进行最小同步采样，而战略性地保留所有其他可区分路径，以最大限度地提高嵌入容量。我们为该方法提供了安全性的理论证明，并分析了其可实现的嵌入容量与理论上限之间的差距。在使用Llama 3的英语基准和使用Qwen 2.5的中文基准上的实验表明，我们的方法可以一致地接近理论容量上限，显著优于SyncPool。英语中的嵌入率提升超过160%，中文中的提升超过25%，特别是在候选池较大的设置中。这项工作代表了实用高容量可证明安全的语言隐写术的重要一步。 

---
# EntropyLong: Effective Long-Context Training via Predictive Uncertainty 

**Title (ZH)**: 熵长：通过预测不确定性进行有效的长上下文训练 

**Authors**: Junlong Jia, Ziyang Chen, Xing Wu, Chaochen Gao, Zijia Lin, Debing Zhang, Songlin Hu, Binghui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.02330)  

**Abstract**: Training long-context language models to capture long-range dependencies requires specialized data construction. Current approaches, such as generic text concatenation or heuristic-based variants, frequently fail to guarantee genuine long-range dependencies. We propose EntropyLong, a novel data construction method that leverages predictive uncertainty to verify dependency quality. Our approach identifies high-entropy positions in documents, retrieves semantically relevant contexts from large corpora, and verifies their utility by assessing whether they reduce prediction entropy. This model-in-the-loop verification ensures each dependency represents measurable information gain rather than spurious correlation. We construct training samples with long-range dependencies by combining original documents with these verified contextual supplements. Using FineWebEdu and Cosmopedia, we generate a dataset of 128K-length sequences with verified dependencies. Models trained on this data demonstrate significant improvements on RULER benchmarks, particularly in tasks requiring distant information. Following instruction fine-tuning, our models also achieve substantial gains on LongBenchv2, demonstrating enhanced long-context understanding. Extensive ablation studies further validate the necessity and effectiveness of entropybased verification for long-context training. 

**Abstract (ZH)**: 利用预测不确定性验证依赖质量以构建长上下文语言模型的数据构造方法 EntropyLong及其在验证长范围依赖数据集构建中的应用 

---
# SelfJudge: Faster Speculative Decoding via Self-Supervised Judge Verification 

**Title (ZH)**: SelfJudge: 更快的推测解码 via 自监督判决验证 

**Authors**: Kanghoon Yoon, Minsub Kim, Sungjae Lee, Joonhyung Lee, Sunghyeon Woo, Yeonjun In, Se Jung Kwon, Chanyoung Park, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.02329)  

**Abstract**: Speculative decoding accelerates LLM inference by verifying candidate tokens from a draft model against a larger target model. Recent judge decoding boosts this process by relaxing verification criteria by accepting draft tokens that may exhibit minor discrepancies from target model output, but existing methods are restricted by their reliance on human annotations or tasks with verifiable ground truths, limiting generalizability across diverse NLP tasks. We propose SelfJudge, which trains judge verifiers via self-supervision of the target model. Our method measures semantic preservation by assessing whether token-substituted responses preserve the meaning of original responses, enabling automatic verifier training across diverse NLP tasks. Our experiments show SelfJudge achieves superior inference-accuracy trade-offs than judge decoding baselines, offering a broadly applicable solution for faster LLM inference. 

**Abstract (ZH)**: 推测解码通过验证草稿模型的候选词against目标模型来加速大语言模型的推理。近期的法官解码通过放宽验证标准来提升这一过程，接受可能与目标模型输出有轻微差异的草稿词，但现有方法受限于其对人工注释的依赖或可验证的_ground_truth_任务，限制了其在多种NLP任务中的普适性。我们提出SelfJudge，该方法通过目标模型的自监督训练法官验证器。我们的方法通过评估token替换后的响应是否保留了原始响应的含义来衡量语义保真度，从而实现跨多种NLP任务的自动验证器训练。我们的实验表明，SelfJudge在推理准确性和效率之间取得了优于法官解码基线的性能，提供了一种广泛适用的快速大语言模型推理解决方案。 

---
# KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI 

**Title (ZH)**: KAME: 串联架构在实时语音对话AI中增强知识 

**Authors**: So Kuroki, Yotaro Kubo, Takuya Akiba, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02327)  

**Abstract**: Real-time speech-to-speech (S2S) models excel at generating natural, low-latency conversational responses but often lack deep knowledge and semantic understanding. Conversely, cascaded systems combining automatic speech recognition, a text-based Large Language Model (LLM), and text-to-speech synthesis offer superior knowledge representation at the cost of high latency, which disrupts the flow of natural interaction. This paper introduces a novel hybrid architecture that bridges the gap between these two paradigms. Our framework processes user speech through an S2S transformer for immediate responsiveness while concurrently relaying the query to a powerful back-end LLM. The LLM's text-based response is then injected in real time to guide the S2S model's speech generation, effectively infusing its output with rich knowledge without the full latency penalty of a cascaded system. We evaluated our method using a speech-synthesized variant of the MT-Bench benchmark that consists of multi-turn question-answering sessions. The results demonstrate that our system substantially outperforms a baseline S2S model in response correctness, approaching that of a cascaded system, while maintaining a latency on par with the baseline. 

**Abstract (ZH)**: 基于实时语音到语音模型与后端大语言模型的新型混合架构：兼具即时响应与深度知识理解 

---
# Hallucination-Resistant, Domain-Specific Research Assistant with Self-Evaluation and Vector-Grounded Retrieval 

**Title (ZH)**: 具有自我评估和向量grounded检索的抗幻觉领域特定研究助理 

**Authors**: Vivek Bhavsar, Joseph Ereifej, Aravanan Gurusami  

**Link**: [PDF](https://arxiv.org/pdf/2510.02326)  

**Abstract**: Large language models accelerate literature synthesis but can hallucinate and mis-cite, limiting their usefulness in expert workflows. We present RA-FSM (Research Assistant - Finite State Machine), a modular GPT-based research assistant that wraps generation in a finite-state control loop: Relevance -> Confidence -> Knowledge. The system is grounded in vector retrieval and a deterministic citation pipeline. The controller filters out-of-scope queries, scores answerability, decomposes questions, and triggers retrieval only when needed, and emits answers with confidence labels and in-corpus, de-duplicated references. A ranked-tier ingestion workflow constructs a domain knowledge base from journals, conferences, indices, preprints, and patents, writing both to a dense vector index and to a relational store of normalized metrics. We implement the system for photonics and evaluate it on six task categories: analytical reasoning, numerical analysis, methodological critique, comparative synthesis, factual extraction, and application design. In blinded A/B reviews, domain experts prefer RA-FSM to both a strong Notebook LM (NLM) and a vanilla Default GPT API call single-pass baseline, citing stronger boundary-condition handling and more defensible evidence use. Coverage and novelty analyses indicate that RA-FSM explores beyond the NLM while incurring tunable latency and cost overheads. The design emphasizes transparent, well-cited answers for high-stakes technical work and is generalizable to other scientific domains. 

**Abstract (ZH)**: 大规模语言模型加速文献综合但可能存在幻觉和误引，限制了其在专家工作流中的应用。我们提出了一种模块化的基于GPT的研究助手RA-FSM（Research Assistant - Finite State Machine），该助手将生成过程嵌入到一个有限状态控制循环中：相关性 -> 置信度 -> 知识。系统基于向量检索和确定性的引文管道。控制器过滤掉范围外的查询，评估回答的可能性，分解问题，并仅在需要时触发检索，并随置信度标签和去重后的引用一并发出答案。经过排名的 ingestion 工作流从期刊、会议、索引、预印本和专利中构建领域知识库，并同时写入密集向量索引和标准化指标的关系存储库。我们为光子学领域实现了该系统，并在六个任务类别上进行了评估：分析推理、数值分析、方法论批判、比较综合、事实提取和应用设计。在盲测的A/B评审中，领域专家更偏好RA-FSM，而不是强大的Notebook LM（NLM）和常规的单一通过Default GPT API调用基线，原因是RA-FSM在边界条件处理和更有说服力的证据使用方面表现更好。覆盖率和新颖性分析表明，RA-FSM探索领域知识的同时，可以调节延迟和成本开销。该设计强调高风险技术工作中透明且引文充足的回答，并且可以泛化到其他科学领域。 

---
# Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning 

**Title (ZH)**: CASAL：对比激活导向的 amortized 学习中的幻觉减少 

**Authors**: Wannan Yang, Xinchi Qiu, Lei Yu, Yuchen Zhang, Oliver Aobo Yang, Narine Kokhlikyan, Nicola Cancedda, Diego Garcia-Olano  

**Link**: [PDF](https://arxiv.org/pdf/2510.02324)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities but often hallucinate, confidently providing incorrect answers instead of admitting ignorance. Prior work has shown that models encode linear representations of their own knowledge and that activation steering can reduce hallucinations. These approaches, however, require real-time monitoring and intervention during inference. We introduce Contrastive Activation Steering for Amortized Learning (CASAL), an efficient algorithm that connects interpretability with amortized optimization. CASAL directly bakes the benefits of activation steering into model's weights. Once trained, LLMs answer questions they know while abstaining from answering those they do not. CASAL's light-weight design requires training only a submodule of a single transformer layer and yet reduces hallucination by 30%-40% across multiple short-form QA benchmarks. CASAL is 30x more compute-efficient and 20x more data-efficient than strong LoRA-based baselines such as SFT and DPO, boosting its practical applicability in data scarce domains. Importantly, CASAL also generalizes effectively to out-of-distribution (OOD) domains. We showcase CASAL's flexibility in mitigating hallucinations in both text-only and vision-language models. To our knowledge, CASAL is the first steering-based training method that has been shown to be effective for both dense and Mixture-of-Experts (MoE) models. CASAL represents a promising step forward for applying interpretability-inspired method for practical deployment in production systems. 

**Abstract (ZH)**: 对比激活导向的递增学习算法（CASAL）：减轻幻觉并提高模型解释性和效率 

---
# Modeling the Attack: Detecting AI-Generated Text by Quantifying Adversarial Perturbations 

**Title (ZH)**: 基于对抗扰动量化检测AI生成文本的模型 

**Authors**: Lekkala Sai Teja, Annepaka Yadagiri, Sangam Sai Anish, Siva Gopala Krishna Nuthakki, Partha Pakray  

**Link**: [PDF](https://arxiv.org/pdf/2510.02319)  

**Abstract**: The growth of highly advanced Large Language Models (LLMs) constitutes a huge dual-use problem, making it necessary to create dependable AI-generated text detection systems. Modern detectors are notoriously vulnerable to adversarial attacks, with paraphrasing standing out as an effective evasion technique that foils statistical detection. This paper presents a comparative study of adversarial robustness, first by quantifying the limitations of standard adversarial training and then by introducing a novel, significantly more resilient detection framework: Perturbation-Invariant Feature Engineering (PIFE), a framework that enhances detection by first transforming input text into a standardized form using a multi-stage normalization pipeline, it then quantifies the transformation's magnitude using metrics like Levenshtein distance and semantic similarity, feeding these signals directly to the classifier. We evaluate both a conventionally hardened Transformer and our PIFE-augmented model against a hierarchical taxonomy of character-, word-, and sentence-level attacks. Our findings first confirm that conventional adversarial training, while resilient to syntactic noise, fails against semantic attacks, an effect we term "semantic evasion threshold", where its True Positive Rate at a strict 1% False Positive Rate plummets to 48.8%. In stark contrast, our PIFE model, which explicitly engineers features from the discrepancy between a text and its canonical form, overcomes this limitation. It maintains a remarkable 82.6% TPR under the same conditions, effectively neutralizing the most sophisticated semantic attacks. This superior performance demonstrates that explicitly modeling perturbation artifacts, rather than merely training on them, is a more promising path toward achieving genuine robustness in the adversarial arms race. 

**Abstract (ZH)**: 高级大型语言模型的快速发展构成了一个重大的两用难题，需要创建可靠的AI生成文本检测系统。现代检测器对对抗攻击特别脆弱，改写尤其有效地规避了统计检测。本文通过比较抗欺骗性，首先量化标准对抗训练的局限性，然后介绍了一种新型、显著更稳健的检测框架：扰动不变特征工程（PIFE），该框架通过多阶段标准化管道将输入文本转换为标准形式，接着使用Levenshtein距离和语义相似性等度量量化变换的程度，并将这些信号直接传递给分类器。我们评估了常规加固的Transformer模型和我们的PIFE增强模型，它们分别针对字符级、词级和句级的攻击层次分类。我们的研究结果首先确认，虽然常规对抗训练对语法噪声具有韧性，但对语义攻击却无效，我们称之为“语义规避门槛”，在严格1%的假阳性率下，其真正阳性率骤降至48.8%。相比之下，我们的PIFE模型通过从文本与其标准形式之间的差异中明确构造特征，克服了这一局限。在相同条件下，它保持了惊人的82.6%的真正阳性率，有效地抵消了最复杂的语义攻击。这种优越的性能证明了明确建模扰动伪影，而不是仅仅在它们上进行训练，是对抗性增强竞赛实现真正鲁棒性的更有前途的途径。 

---
