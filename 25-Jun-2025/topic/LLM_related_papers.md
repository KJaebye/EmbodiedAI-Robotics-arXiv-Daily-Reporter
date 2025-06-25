# Adaptive Domain Modeling with Language Models: A Multi-Agent Approach to Task Planning 

**Title (ZH)**: 基于语言模型的自适应领域建模：任务规划的多Agent方法 

**Authors**: Harisankar Babu, Philipp Schillinger, Tamim Asfour  

**Link**: [PDF](https://arxiv.org/pdf/2506.19592)  

**Abstract**: We introduce TAPAS (Task-based Adaptation and Planning using AgentS), a multi-agent framework that integrates Large Language Models (LLMs) with symbolic planning to solve complex tasks without the need for manually defined environment models. TAPAS employs specialized LLM-based agents that collaboratively generate and adapt domain models, initial states, and goal specifications as needed using structured tool-calling mechanisms. Through this tool-based interaction, downstream agents can request modifications from upstream agents, enabling adaptation to novel attributes and constraints without manual domain redefinition. A ReAct (Reason+Act)-style execution agent, coupled with natural language plan translation, bridges the gap between dynamically generated plans and real-world robot capabilities. TAPAS demonstrates strong performance in benchmark planning domains and in the VirtualHome simulated real-world environment. 

**Abstract (ZH)**: 基于任务的多智能体框架TAPAS：将大型语言模型与符号规划集成以解决复杂任务 

---
# JoyAgents-R1: Joint Evolution Dynamics for Versatile Multi-LLM Agents with Reinforcement Learning 

**Title (ZH)**: JoyAgents-R1: 联合进化动力学实现多功能多语言模型代理的强化学习 

**Authors**: Ai Han, Junxing Hu, Pu Wei, Zhiqian Zhang, Yuhang Guo, Jiawei Lu, Zicheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19846)  

**Abstract**: Multi-agent reinforcement learning (MARL) has emerged as a prominent paradigm for increasingly complex tasks. However, joint evolution across heterogeneous agents remains challenging due to cooperative inefficiency and training instability. In this paper, we propose the joint evolution dynamics for MARL called JoyAgents-R1, which first applies Group Relative Policy Optimization (GRPO) to the joint training of heterogeneous multi-agents. By iteratively refining agents' large language models (LLMs) and memories, the method achieves holistic equilibrium with optimal decision-making and memory capabilities. Specifically, JoyAgents-R1 first implements node-wise Monte Carlo sampling on the behavior of each agent across entire reasoning trajectories to enhance GRPO sampling efficiency while maintaining policy diversity. Then, our marginal benefit-driven selection strategy identifies top-$K$ sampling groups with maximal reward fluctuations, enabling targeted agent model updates that improve training stability and maximize joint benefits through cost-effective parameter adjustments. Meanwhile, JoyAgents-R1 introduces an adaptive memory evolution mechanism that repurposes GRPO rewards as cost-free supervisory signals to eliminate repetitive reasoning and accelerate convergence. Experiments across general and domain-specific scenarios demonstrate that JoyAgents-R1 achieves performance comparable to that of larger LLMs while built on smaller open-source models. 

**Abstract (ZH)**: 多智能体强化学习中Joint Evolution Dynamics for MARL称为JoyAgents-R1 

---
# KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality 

**Title (ZH)**: 知RL：探索知识导向的强化学习以确保事实性 

**Authors**: Baochang Ren, Shuofei Qiao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19807)  

**Abstract**: Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL. 

**Abstract (ZH)**: 知识增强的强化学习：有效缓解慢思考模型中的幻觉问题 

---
# Learning Task Belief Similarity with Latent Dynamics for Meta-Reinforcement Learning 

**Title (ZH)**: 基于潜在动力学的元强化学习中任务信念相似性学习 

**Authors**: Menglong Zhang, Fuyuan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.19785)  

**Abstract**: Meta-reinforcement learning requires utilizing prior task distribution information obtained during exploration to rapidly adapt to unknown tasks. The efficiency of an agent's exploration hinges on accurately identifying the current task. Recent Bayes-Adaptive Deep RL approaches often rely on reconstructing the environment's reward signal, which is challenging in sparse reward settings, leading to suboptimal exploitation. Inspired by bisimulation metrics, which robustly extracts behavioral similarity in continuous MDPs, we propose SimBelief-a novel meta-RL framework via measuring similarity of task belief in Bayes-Adaptive MDP (BAMDP). SimBelief effectively extracts common features of similar task distributions, enabling efficient task identification and exploration in sparse reward environments. We introduce latent task belief metric to learn the common structure of similar tasks and incorporate it into the specific task belief. By learning the latent dynamics across task distributions, we connect shared latent task belief features with specific task features, facilitating rapid task identification and adaptation. Our method outperforms state-of-the-art baselines on sparse reward MuJoCo and panda-gym tasks. 

**Abstract (ZH)**: 元强化学习需要利用探索过程中获得的任务分布信息，以快速适应未知任务。智能体探索的效率依赖于准确识别当前任务。受bisimulation度量的启发，我们提出SimBelief——一种通过测量Bayes-适应MDP（BAMDP）中任务信念相似性的新型元RL框架。SimBelief有效提取类似任务分布的共性特征，使在稀疏奖励环境中实现高效的任务识别和探索成为可能。我们引入隐含任务信念度量来学习类似任务的共同结构，并将其纳入特定任务信念中。通过学习任务分布之间的潜在动力学，我们将共享的潜在任务信念特征与特定任务特征连接起来，促进快速的任务识别和适应。我们的方法在稀疏奖励的MuJoCo和panda-gym任务上优于最先进的基线方法。 

---
# SAGE: Strategy-Adaptive Generation Engine for Query Rewriting 

**Title (ZH)**: SAGE: 策略适应型生成引擎用于查询重写 

**Authors**: Teng Wang, Hailei Gong, Changwang Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19783)  

**Abstract**: Query rewriting is pivotal for enhancing dense retrieval, yet current methods demand large-scale supervised data or suffer from inefficient reinforcement learning (RL) exploration. In this work, we first establish that guiding Large Language Models (LLMs) with a concise set of expert-crafted strategies, such as semantic expansion and entity disambiguation, substantially improves retrieval effectiveness on challenging benchmarks, including HotpotQA, FEVER, NFCorpus, and SciFact. Building on this insight, we introduce the Strategy-Adaptive Generation Engine (SAGE), which operationalizes these strategies in an RL framework. SAGE introduces two novel reward shaping mechanisms-Strategic Credit Shaping (SCS) and Contrastive Reward Shaping (CRS)-to deliver more informative learning signals. This strategy-guided approach not only achieves new state-of-the-art NDCG@10 results, but also uncovers a compelling emergent behavior: the agent learns to select optimal strategies, reduces unnecessary exploration, and generates concise rewrites, lowering inference cost without sacrificing performance. Our findings demonstrate that strategy-guided RL, enhanced with nuanced reward shaping, offers a scalable, efficient, and more interpretable paradigm for developing the next generation of robust information retrieval systems. 

**Abstract (ZH)**: 基于策略导向的强化学习对于增强密集检索具有重要意义，但当前方法需要大规模监督数据或遭受无效的强化学习探索。在这项工作中，我们首先证明，使用简洁的专家crafted策略集，如语义扩展和实体消歧，显著提高了在HotpotQA、FEVER、NFCorpus和SciFact等具有挑战性的基准上的检索效果。在此基础上，我们引入了策略自适应生成引擎（SAGE），它在RL框架中实现这些策略。SAGE引入了两种新颖的奖励塑造机制——策略信用塑造（SCS）和对比奖励塑造（CRS），以提供更具信息量的学习信号。这种策略导向的方法不仅达到了新的NDCG@10最佳成果，还揭示了一种引人注目的新兴行为：代理学会了选择最优策略，减少了不必要的探索，并生成了简洁的重写，降低了推理成本而不牺牲性能。我们的研究表明，结合细腻奖励塑造的策略导向RL，为开发下一代稳健的信息检索系统提供了可扩展、高效和更可解释的范式。 

---
# LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis 

**Title (ZH)**: 基于LLM的医学文档分析：增强病理学和鉴别诊断的可靠性 

**Authors**: Lei Kang, Xuanshuo Fu, Oriol Ramos Terrades, Javier Vazquez-Corral, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19702)  

**Abstract**: Medical document analysis plays a crucial role in extracting essential clinical insights from unstructured healthcare records, supporting critical tasks such as differential diagnosis. Determining the most probable condition among overlapping symptoms requires precise evaluation and deep medical expertise. While recent advancements in large language models (LLMs) have significantly enhanced performance in medical document analysis, privacy concerns related to sensitive patient data limit the use of online LLMs services in clinical settings. To address these challenges, we propose a trustworthy medical document analysis platform that fine-tunes a LLaMA-v3 using low-rank adaptation, specifically optimized for differential diagnosis tasks. Our approach utilizes DDXPlus, the largest benchmark dataset for differential diagnosis, and demonstrates superior performance in pathology prediction and variable-length differential diagnosis compared to existing methods. The developed web-based platform allows users to submit their own unstructured medical documents and receive accurate, explainable diagnostic results. By incorporating advanced explainability techniques, the system ensures transparent and reliable predictions, fostering user trust and confidence. Extensive evaluations confirm that the proposed method surpasses current state-of-the-art models in predictive accuracy while offering practical utility in clinical settings. This work addresses the urgent need for reliable, explainable, and privacy-preserving artificial intelligence solutions, representing a significant advancement in intelligent medical document analysis for real-world healthcare applications. The code can be found at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 医学文档分析在从未结构化的医疗记录中提取关键临床洞察方面发挥着重要作用，支持诸如鉴别诊断等重要任务。确定重叠症状下的最可能病症需要精确的评估和深厚的专业医学知识。尽管大型语言模型（LLMs）的最新进展显著提高了医学文档分析的性能，但与敏感患者数据相关的隐私问题限制了在线LLM服务在临床环境中的应用。为了解决这些挑战，我们提出了一种值得信赖的医学文档分析平台，该平台使用低秩适应技术微调了LLaMA-v3模型，特别优化用于鉴别诊断任务。我们的方法利用了最大的鉴别诊断基准数据集DDXPlus，并在病理预测和变长鉴别诊断任务中展示了优于现有方法的性能。开发的基于Web的平台允许用户提交自己的未结构化医疗文档，并获得准确可解释的诊断结果。通过采用先进的可解释性技术，该系统确保了透明和可靠的预测，促进了用户信任和信心。广泛评估表明，所提出的方法在预测准确性上超越了现有最先进的模型，在临床环境中还提供了实用的功能。这项工作满足了可靠、可解释和隐私保护的人工智能解决方案的迫切需求，代表了智能医学文档分析在实际医疗应用中的重要进展。相关代码可在 \href{this https URL}{this https URL} 获取。 

---
# NaviAgent: Bilevel Planning on Tool Dependency Graphs for Function Calling 

**Title (ZH)**: NaviAgent: 工具依赖图上的双层规划用于函数调用 

**Authors**: Yan Jiang, Hao Zhou, LiZhong GU, Ai Han, TianLong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19500)  

**Abstract**: LLMs' reliance on static knowledge and fragile tool invocation severely hinders the orchestration of complex, heterogeneous toolchains, particularly at large scales. Existing methods typically use rigid single-path execution, resulting in poor error recovery and exponentially growing search spaces. We introduce NaviAgent, a graph-navigated bilevel planning architecture for robust function calling, comprising a Multi-Path Decider and Graph-Encoded Navigator. As an LLM-powered agent, the Multi-Path Decider defines a four-dimensional decision space and continuously perceives environmental states, dynamically selecting the optimal action to fully cover all tool invocation scenarios. The Graph-Encoded Navigator constructs a Tool Dependency Heterogeneous Graph (TDHG), where node embeddings explicitly fuse API schema structure with historical invocation behavior. It also integrates a novel heuristic search strategy that guides the Decider toward efficient and highly successful toolchains, even for unseen tool combinations. Experiments show that NaviAgent consistently achieves the highest task success rate (TSR) across all foundation models and task complexities, outperforming the average baselines (ReAct, ToolLLM, {\alpha}-UMI) by 13.5%, 16.4%, and 19.0% on Qwen2.5-14B, Qwen2.5-32B, and Deepseek-V3, respectively. Its execution steps are typically within one step of the most efficient baseline, ensuring a strong balance between quality and efficiency. Notably, a fine-tuned Qwen2.5-14B model achieves a TSR of 49.5%, surpassing the much larger 32B model (44.9%) under our architecture. Incorporating the Graph-Encoded Navigator further boosts TSR by an average of 2.4 points, with gains up over 9 points on complex tasks for larger models (Deepseek-V3 and GPT-4o), highlighting its essential role in toolchain orchestration. 

**Abstract (ZH)**: LLMs对静态知识的依赖及其工具调用的脆弱性严重阻碍了复杂异构工具链的编排，特别是在大规模情况下。现有方法通常采用刚性单一路径执行，导致错误恢复性能差且搜索空间呈指数增长。我们引入了NaviAgent，这是一种用于稳健函数调用的图导航双层规划架构，包括多路径决策器和图编码导航器。作为LLM驱动的代理，多路径决策器定义了一个四维决策空间，并持续感知环境状态，动态选择最优行动以全面覆盖所有工具调用场景。图编码导航器构建了一个工具依赖异构图（TDHG），其中节点嵌入明确融合了API结构与历史调用行为。它还集成了一个新颖的启发式搜索策略，引导决策器趋向高效的工具链，即使对于未见过的工具组合也是如此。实验结果显示，NaviAgent在所有基础模型和任务复杂度下一致实现了最高的任务成功率（TSR），分别比ReAct、ToolLLM和α-UMI基准模型提高了13.5%、16.4%和19.0%性能，在Qwen2.5-14B、Qwen2.5-32B和Deepseek-V3上的表现尤为突出。其执行步骤通常比最高效的基线模型少一步，确保了高质量与效率之间的良好平衡。值得注意的是，微调后的Qwen2.5-14B模型在我们的架构下实现了49.5%的任务成功率，远超较大的32B模型（44.9%）。图编码导航器的引入进一步提升了任务成功率2.4%，在更复杂任务上对更大模型（如Deepseek-V3和GPT-4o）的增益超过9%，突显了其在工具链编排中的核心作用。 

---
# KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models 

**Title (ZH)**: KunLunBaizeRAG：大型语言模型的强化学习驱动推理性能跃升 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Qihang Zhou, KunLun Meta  

**Link**: [PDF](https://arxiv.org/pdf/2506.19466)  

**Abstract**: This paper introduces KunLunBaizeRAG, a reinforcement learning-driven reasoning framework designed to enhance the reasoning capabilities of large language models (LLMs) in complex multi-hop question-answering tasks. The framework addresses key limitations of traditional RAG, such as retrieval drift, information redundancy, and strategy rigidity. Key innovations include the RAG-driven Reasoning Alignment (RDRA) mechanism, the Search-Think Iterative Enhancement (STIE) mechanism, the Network-Local Intelligent Routing (NLR) mechanism, and a progressive hybrid training strategy. Experimental results demonstrate significant improvements in exact match (EM) and LLM-judged score (LJ) across four benchmarks, highlighting the framework's robustness and effectiveness in complex reasoning scenarios. 

**Abstract (ZH)**: KunLunBaizeRAG：一种强化学习驱动的推理框架，用于增强大规模语言模型在复杂多跳问答任务中的推理能力 

---
# Commander-GPT: Dividing and Routing for Multimodal Sarcasm Detection 

**Title (ZH)**: Commander-GPT：多模态讽刺检测的分割与路由方法 

**Authors**: Yazhou Zhang, Chunwang Zou, Bo Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19420)  

**Abstract**: Multimodal sarcasm understanding is a high-order cognitive task. Although large language models (LLMs) have shown impressive performance on many downstream NLP tasks, growing evidence suggests that they struggle with sarcasm understanding. In this paper, we propose Commander-GPT, a modular decision routing framework inspired by military command theory. Rather than relying on a single LLM's capability, Commander-GPT orchestrates a team of specialized LLM agents where each agent will be selectively assigned to a focused sub-task such as context modeling, sentiment analysis, etc. Their outputs are then routed back to the commander, which integrates the information and performs the final sarcasm judgment. To coordinate these agents, we introduce three types of centralized commanders: (1) a trained lightweight encoder-based commander (e.g., multi-modal BERT); (2) four small autoregressive language models, serving as moderately capable commanders (e.g., DeepSeek-VL); (3) two large LLM-based commander (Gemini Pro and GPT-4o) that performs task routing, output aggregation, and sarcasm decision-making in a zero-shot fashion. We evaluate Commander-GPT on the MMSD and MMSD 2.0 benchmarks, comparing five prompting strategies. Experimental results show that our framework achieves 4.4% and 11.7% improvement in F1 score over state-of-the-art (SoTA) baselines on average, demonstrating its effectiveness. 

**Abstract (ZH)**: 多模态 sarcasm 理解是一项高阶认知任务。尽管大型语言模型（LLMs）在许多下游 NLP 任务上展现了出色的表现，但越来越多的证据表明，它们在理解 sarcasm 方面存在困难。在本文中，我们提出 Commander-GPT，这是一种受军事指挥理论启发的模块化决策路由框架。.Commander-GPT 不依赖单一 LLM 的能力，而是协调一组专门化的 LLM 代理，每个代理将被选择性地分配到诸如上下文建模、情感分析等专注于的子任务。然后将他们的输出路由回指挥官，指挥官整合信息并进行最终的 sarcasm 判断。为了协调这些代理，我们引入了三种类型的集中式指挥官：（1）一种训练过的轻量级编码器基指挥官（如多模态 BERT）；（2）四种较小的自回归语言模型，作为适中的指挥官（如 DeepSeek-VL）；（3）两种基于 LLM 的指挥官（Gemini Pro 和 GPT-4o），它们以零样本方式执行任务路由、输出聚合和 sarcasm 决策。我们在 MMSD 和 MMSD 2.0 挑战集上评估了 Commander-GPT，并与五种不同的提示策略进行比较。实验结果表明，我们的框架在平均 F1 分数上分别提高了 4.4% 和 11.7%，展示了其有效性。 

---
# FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring 

**Title (ZH)**: FEAT：一种通过低成本自动生成和标注框架的英语AI辅导偏好反馈数据集 

**Authors**: Hyein Seo, Taewook Hwang, Yohan Lee, sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2506.19325)  

**Abstract**: In English education tutoring, teacher feedback is essential for guiding students. Recently, AI-based tutoring systems have emerged to assist teachers; however, these systems require high-quality and large-scale teacher feedback data, which is both time-consuming and costly to generate manually. In this study, we propose FEAT, a cost-effective framework for generating teacher feedback, and have constructed three complementary datasets: (1) DIRECT-Manual (DM), where both humans and large language models (LLMs) collaboratively generate high-quality teacher feedback, albeit at a higher cost; (2) DIRECT-Generated (DG), an LLM-only generated, cost-effective dataset with lower quality;, and (3) DIRECT-Augmented (DA), primarily based on DG with a small portion of DM added to enhance quality while maintaining cost-efficiency. Experimental results showed that incorporating a small portion of DM (5-10%) into DG leads to superior performance compared to using 100% DM alone. 

**Abstract (ZH)**: 基于英文学术辅导中教师反馈生成的成本有效框架FEAT 

---
# Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs 

**Title (ZH)**: Skywork-SWE: 揭示大规模语言模型中软件工程数据的扩展定律 

**Authors**: Liang Zeng, Yongcong Li, Yuzhen Xiao, Changshi Li, Chris Yuhao Liu, Rui Yan, Tianwen Wei, Jujie He, Xuchen Song, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.19290)  

**Abstract**: Software engineering (SWE) has recently emerged as a crucial testbed for next-generation LLM agents, demanding inherent capabilities in two critical dimensions: sustained iterative problem-solving (e.g., >50 interaction rounds) and long-context dependency resolution (e.g., >32k tokens). However, the data curation process in SWE remains notoriously time-consuming, as it heavily relies on manual annotation for code file filtering and the setup of dedicated runtime environments to execute and validate unit tests. Consequently, most existing datasets are limited to only a few thousand GitHub-sourced instances. To this end, we propose an incremental, automated data-curation pipeline that systematically scales both the volume and diversity of SWE datasets. Our dataset comprises 10,169 real-world Python task instances from 2,531 distinct GitHub repositories, each accompanied by a task specified in natural language and a dedicated runtime-environment image for automated unit-test validation. We have carefully curated over 8,000 successfully runtime-validated training trajectories from our proposed SWE dataset. When fine-tuning the Skywork-SWE model on these trajectories, we uncover a striking data scaling phenomenon: the trained model's performance for software engineering capabilities in LLMs continues to improve as the data size increases, showing no signs of saturation. Notably, our Skywork-SWE model achieves 38.0% pass@1 accuracy on the SWE-bench Verified benchmark without using verifiers or multiple rollouts, establishing a new state-of-the-art (SOTA) among the Qwen2.5-Coder-32B-based LLMs built on the OpenHands agent framework. Furthermore, with the incorporation of test-time scaling techniques, the performance further improves to 47.0% accuracy, surpassing the previous SOTA results for sub-32B parameter models. We release the Skywork-SWE-32B model checkpoint to accelerate future research. 

**Abstract (ZH)**: 软件工程（SWE）近年来已成为下一代LLM代理的关键试验台，要求在两个关键维度上具备固有的能力：持续迭代问题解决（例如，>50轮交互）和长上下文依赖关系解决（例如，>32k令牌）。然而，SWE中的数据编辑过程仍然 notoriously 耗时，因为这高度依赖于手动注释进行代码文件过滤，并设置专用运行时环境以执行和验证单元测试。因此，现有的大多数数据集仅限于几千个GitHub源实例。为了解决这一问题，我们提出了一种增量的自动化数据编辑流水线，系统地扩大了SWE数据集的规模和多样性。我们的数据集包含来自2,531个独特GitHub仓库的10,169个真实的Python任务实例，每个实例都附带一种用自然语言指定的任务和一个用于自动化单元测试验证的专用运行时环境镜像。我们仔细地从提出的SWE数据集中编辑了超过8,000个成功运行时验证的训练轨迹。在利用这些轨迹微调Skywork-SWE模型时，我们揭露了一个显着的数据规模现象：随数据量增加，训练模型在LLMs中的软件工程能力性能继续提高，显示出无饱和迹象。值得注意的是，我们的Skywork-SWE模型在无需使用验证器或多次测试的情况下，在SWE-bench Verified基准测试中实现了38.0%的pass@1准确率，成为基于OpenHands代理框架的Qwen2.5-Coder-32B模型中新的最优水平。此外，在采用测试时缩放技术后，性能进一步提高到47.0%的准确率，超过之前的最优水平。对于参数少于32B的模型。我们发布Skywork-SWE-32B模型检查点以加速未来的研究。 

---
# RecLLM-R1: A Two-Stage Training Paradigm with Reinforcement Learning and Chain-of-Thought v1 

**Title (ZH)**: RecLLM-R1：一种结合强化学习与链式思维的两阶段训练范式 v1 

**Authors**: Yu Xie, Xingkai Ren, Ying Qi, Yao Hu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19235)  

**Abstract**: Traditional recommendation systems often grapple with "filter bubbles", underutilization of external knowledge, and a disconnect between model optimization and business policy iteration. To address these limitations, this paper introduces RecLLM-R1, a novel recommendation framework leveraging Large Language Models (LLMs) and drawing inspiration from the DeepSeek R1 methodology. The framework initiates by transforming user profiles, historical interactions, and multi-faceted item attributes into LLM-interpretable natural language prompts through a carefully engineered data construction process. Subsequently, a two-stage training paradigm is employed: the initial stage involves Supervised Fine-Tuning (SFT) to imbue the LLM with fundamental recommendation capabilities. The subsequent stage utilizes Group Relative Policy Optimization (GRPO), a reinforcement learning technique, augmented with a Chain-of-Thought (CoT) mechanism. This stage guides the model through multi-step reasoning and holistic decision-making via a flexibly defined reward function, aiming to concurrently optimize recommendation accuracy, diversity, and other bespoke business objectives. Empirical evaluations on a real-world user behavior dataset from a large-scale social media platform demonstrate that RecLLM-R1 significantly surpasses existing baseline methods across a spectrum of evaluation metrics, including accuracy, diversity, and novelty. It effectively mitigates the filter bubble effect and presents a promising avenue for the integrated optimization of recommendation models and policies under intricate business goals. 

**Abstract (ZH)**: 利用大型语言模型的RecLLM-R1：一种新颖的推荐框架 

---
# Spiritual-LLM : Gita Inspired Mental Health Therapy In the Era of LLMs 

**Title (ZH)**: Spiritual-LLM：受《薄伽梵歌》启发的大规模语言模型时代的精神健康疗法 

**Authors**: Janak Kapuriya, Aman Singh, Jainendra Shukla, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2506.19185)  

**Abstract**: Traditional mental health support systems often generate responses based solely on the user's current emotion and situations, resulting in superficial interventions that fail to address deeper emotional needs. This study introduces a novel framework by integrating spiritual wisdom from the Bhagavad Gita with advanced large language model GPT-4o to enhance emotional well-being. We present the GITes (Gita Integrated Therapy for Emotional Support) dataset, which enhances the existing ExTES mental health dataset by including 10,729 spiritually guided responses generated by GPT-4o and evaluated by domain experts. We benchmark GITes against 12 state-of-the-art LLMs, including both mental health specific and general purpose models. To evaluate spiritual relevance in generated responses beyond what conventional n-gram based metrics capture, we propose a novel Spiritual Insight metric and automate assessment via an LLM as jury framework using chain-of-thought prompting. Integrating spiritual guidance into AI driven support enhances both NLP and spiritual metrics for the best performing LLM Phi3-Mini 3.2B Instruct, achieving improvements of 122.71% in ROUGE, 126.53% in METEOR, 8.15% in BERT score, 15.92% in Spiritual Insight, 18.61% in Sufficiency and 13.22% in Relevance compared to its zero-shot counterpart. While these results reflect substantial improvements across automated empathy and spirituality metrics, further validation in real world patient populations remains a necessary step. Our findings indicate a strong potential for AI systems enriched with spiritual guidance to enhance user satisfaction and perceived support outcomes. The code and dataset will be publicly available to advance further research in this emerging area. 

**Abstract (ZH)**: 传统心理健康支持系统往往仅基于用户当前的情绪和情境生成回应，导致表面化的干预措施无法满足更深层次的情感需求。本研究介绍了一种新的框架，将《薄伽梵歌》中的精神智慧与先进的大语言模型GPT-4o相结合，以提升情绪福祉。我们提出了GITes（《薄伽梵歌》整合疗法情感支持数据集），该数据集通过包含10,729条由GPT-4o生成的具有精神指导意义的回应并由领域专家评估，增强了现有的ExTES心理健康数据集。我们将GITes与12种最先进的语言模型进行了基准测试，包括心理健康专门模型和通用目的模型。为了评估生成回应的精神相关性，超越传统n-gram基线度量所能捕捉的内容，我们提出了一种新的精神洞察度量，并采用LLM作为陪审团框架并通过链式思考提示自动评估。将精神指导整合到以AI驱动的支持中，不仅提升了语言和精神指标，还实现了最有效的模型Phi3-Mini 3.2B Instruct的显著改进，在ROUGE上提高了122.71%，在METEOR上提高了126.53%，在BERT得分上提高了8.15%，在精神洞察度上提高了15.92%，在充足性和相关性上分别提高了18.61%和13.22%，超过了零样本版本。尽管这些结果反映了自动化同理心和精神指标的显著改进，但在真实世界患者群体中的进一步验证仍然是必要的步骤。我们的研究结果表明，富含精神指导的AI系统有可能增强用户满意度和感知支持结果。代码和数据集将公开发布，以推进这一新兴领域进一步的研究。 

---
# Baba is LLM: Reasoning in a Game with Dynamic Rules 

**Title (ZH)**: babá是LLM：在一个动态规则游戏中推理 

**Authors**: Fien van Wetten, Aske Plaat, Max van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2506.19095)  

**Abstract**: Large language models (LLMs) are known to perform well on language tasks, but struggle with reasoning tasks. This paper explores the ability of LLMs to play the 2D puzzle game Baba is You, in which players manipulate rules by rearranging text blocks that define object properties. Given that this rule-manipulation relies on language abilities and reasoning, it is a compelling challenge for LLMs. Six LLMs are evaluated using different prompt types, including (1) simple, (2) rule-extended and (3) action-extended prompts. In addition, two models (Mistral, OLMo) are finetuned using textual and structural data from the game. Results show that while larger models (particularly GPT-4o) perform better in reasoning and puzzle solving, smaller unadapted models struggle to recognize game mechanics or apply rule changes. Finetuning improves the ability to analyze the game levels, but does not significantly improve solution formulation. We conclude that even for state-of-the-art and finetuned LLMs, reasoning about dynamic rule changes is difficult (specifically, understanding the use-mention distinction). The results provide insights into the applicability of LLMs to complex problem-solving tasks and highlight the suitability of games with dynamically changing rules for testing reasoning and reflection by LLMs. 

**Abstract (ZH)**: 大型语言模型在语言任务上表现出色，但在推理任务上存在问题。本文探讨了大型语言模型在玩2D puzzle游戏Baba is You中的能力，该游戏要求玩家通过重新排列定义对象属性的文本块来操纵规则。由于这种规则操纵依赖于语言能力和推理，因此对大型语言模型构成了有力挑战。使用不同类型的提示（包括1）简单提示、2）规则扩展提示和3）动作扩展提示），评估了六种大型语言模型。此外，对两种模型（Mistral、OLMo）进行了微调，使用来自游戏的文字和结构数据。结果显示，尽管较大的模型（尤其是GPT-4o）在推理和解谜方面表现更好，但未适应的小模型难以识别游戏机制或将规则变化应用于游戏中。微调能够在一定程度上提升分析游戏关卡的能力，但未能显著改进问题解决方案的形成。我们得出结论，即使是最先进的和微调后的大型语言模型，处理动态规则变化的能力仍然具有困难（特别是理解使用-提及的区别）。这些结果提供了关于大型语言模型在复杂问题解决任务中的适用性的见解，并突出了动态变化规则的游戏中测试大型语言模型的推理和反思能力的适宜性。 

---
# Do LLMs Know When to Flip a Coin? Strategic Randomization through Reasoning and Experience 

**Title (ZH)**: LLMs知道何时抛硬币吗？基于推理和经验的战略性随机化 

**Authors**: Lingyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18928)  

**Abstract**: Strategic randomization is a key principle in game theory, yet it remains underexplored in large language models (LLMs). Prior work often conflates the cognitive decision to randomize with the mechanical generation of randomness, leading to incomplete evaluations. To address this, we propose a novel zero-sum game inspired by the Tian Ji Horse Race, where the Nash equilibrium corresponds to a maximal entropy strategy. The game's complexity masks this property from untrained humans and underdeveloped LLMs. We evaluate five LLMs across prompt styles -- framed, neutral, and hinted -- using competitive multi-tournament gameplay with system-provided random choices, isolating the decision to randomize. Results show that weaker models remain deterministic regardless of prompts, while stronger models exhibit increased randomization under explicit hints. When facing weaker models, strong LLMs adopt deterministic strategies to exploit biases, but converge toward equilibrium play when facing peers. Through win/loss outcomes and Bayes factor analysis, we demonstrate meaningful variation in LLMs' strategic reasoning capabilities, highlighting opportunities for improvement in abstract reasoning and adaptive learning. We make our implementation publicly available at this https URL to ensure full reproducibility. 

**Abstract (ZH)**: 战略随机化是博弈论中的一个关键原则，但在大规模语言模型中仍然被广泛忽视。以往的工作往往将认知上的随机化决策与机械上的随机生成混淆，导致评估不完整。为解决这一问题，我们提出了一种受田忌赛马启发的新型零和博弈，其纳什均衡对应于最大熵策略。游戏的复杂性使未训练的人类和不发达的大型语言模型无法识别这一性质。我们使用系统提供的随机选择进行竞争多轮博弈，分别采用提示式、中性描述和暗示式三种提示风格评估五种大型语言模型，以隔离随机化决策。结果显示，较弱的模型无论采用何种提示都保持确定性，而较强的模型在明确提示下表现出增加的随机化。面对较弱的模型时，强大的大型语言模型采用确定性策略以利用偏见，但在面对同等水平的模型时则趋向均衡博弈。通过胜负结果和贝叶斯因子分析，我们展示了大型语言模型在战略推理能力上的显著差异，突显了在抽象推理和适应性学习方面改进的机会。我们的实现已在以下网址公开以确保完全可再现：this https URL。 

---
# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study 

**Title (ZH)**: 为什么开源大语言模型在数据分析方面挣扎？一项系统性实证研究 

**Authors**: Yuqi Zhu, Yi Zhong, Jintian Zhang, Ziheng Zhang, Shuofei Qiao, Yujie Luo, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19794)  

**Abstract**: Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. 

**Abstract (ZH)**: 开源大语言模型在数据分析任务中的能力增强策略：通过场景合成方法实现显著的推理能力提升 

---
# SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning 

**Title (ZH)**: SRFT：一种带有监督和强化微调的一阶段推理方法 

**Authors**: Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19767)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in reasoning tasks, yet the optimal integration of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) remains a fundamental challenge. Through comprehensive analysis of token distributions, learning dynamics, and integration mechanisms from entropy-based perspectives, we reveal key differences between these paradigms: SFT induces coarse-grained global changes to LLM policy distributions, while RL performs fine-grained selective optimizations, with entropy serving as a critical indicator of training effectiveness. Building on these observations, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method that unifies both fine-tuning paradigms through entropy-aware weighting mechanisms. Our approach simultaneously applies SFT and RL to directly optimize the LLM using demonstrations and self-exploration rollouts rather than through two-stage sequential methods. Extensive experiments show that SRFT achieves 59.1% average accuracy, outperforming zero-RL methods by 9.0% on five mathematical reasoning benchmarks and 10.9% on three out-of-distribution benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推理任务中取得了显著进展，但监督微调（SFT）和强化学习（RL）的理想融合仍然是一个基础性挑战。通过对基于熵的 token 分布、学习动态和融合机制进行全面分析，我们揭示了这两种范式的关键差异：SFT 引入粗粒度的全局变化到LLM策略分布中，而RL执行细粒度的选择性优化，熵作为训练有效性的重要指标。基于这些观察，我们提出了一种统合监督强化微调（SRFT）方法，该方法通过熵意识加权机制将两种微调范式统一成单阶段方法。我们的方法直接将SFT和RL应用于LLM优化，使用示范和自我探索回放，而不采用两阶段串联方法。广泛实验表明，SRFT在五个数学推理基准上的平均准确率为59.1%，分别比零RL方法在五个数学推理基准上高出9.0%，在三个分布外基准上高出10.9%。 

---
# Arabic Dialect Classification using RNNs, Transformers, and Large Language Models: A Comparative Analysis 

**Title (ZH)**: 使用RNNs、 Transformers和大规模语言模型进行阿拉伯方言分类：一种比较分析 

**Authors**: Omar A.Essameldin, Ali O.Elbeih, Wael H.Gomaa, Wael F.Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19753)  

**Abstract**: The Arabic language is among the most popular languages in the world with a huge variety of dialects spoken in 22 countries. In this study, we address the problem of classifying 18 Arabic dialects of the QADI dataset of Arabic tweets. RNN models, Transformer models, and large language models (LLMs) via prompt engineering are created and tested. Among these, MARBERTv2 performed best with 65% accuracy and 64% F1-score. Through the use of state-of-the-art preprocessing techniques and the latest NLP models, this paper identifies the most significant linguistic issues in Arabic dialect identification. The results corroborate applications like personalized chatbots that respond in users' dialects, social media monitoring, and greater accessibility for Arabic communities. 

**Abstract (ZH)**: 阿拉伯语是世界上最流行的語言之一，共有22个国家使用数百种方言。本文旨在分类QADI阿拉伯推文数据集中的18种阿拉伯方言。通过创建并测试RNN模型、Transformer模型以及通过提示工程使用的大型语言模型（LLMs），MARBERTv2表现最佳，准确率为65%，F1分为64%。通过使用最先进的预处理技术及最新的NLP模型，本文识别出阿拉伯方言识别中最关键的语言问题。研究结果证实了如个性化聊天机器人、社交媒体监控以及为阿拉伯社区提供更广泛访问等应用的有效性。 

---
# Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models 

**Title (ZH)**: 针对稳健的大型语言模型4位量化的一种 outlier-安全预训练方法 

**Authors**: Jungwoo Park, Taewhoo Lee, Chanwoong Yoon, Hyeon Hwang, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19697)  

**Abstract**: Extreme activation outliers in Large Language Models (LLMs) critically degrade quantization performance, hindering efficient on-device deployment. While channel-wise operations and adaptive gradient scaling are recognized causes, practical mitigation remains challenging. We introduce Outlier-Safe Pre-Training (OSP), a practical guideline that proactively prevents outlier formation rather than relying on post-hoc mitigation. OSP combines three key innovations: (1) the Muon optimizer, eliminating privileged bases while maintaining training efficiency; (2) Single-Scale RMSNorm, preventing channel-wise amplification; and (3) a learnable embedding projection, redistributing activation magnitudes originating from embedding matrices. We validate OSP by training a 1.4B-parameter model on 1 trillion tokens, which is the first production-scale LLM trained without such outliers. Under aggressive 4-bit quantization, our OSP model achieves a 35.7 average score across 10 benchmarks (compared to 26.5 for an Adam-trained model), with only a 2% training overhead. Remarkably, OSP models exhibit near-zero excess kurtosis (0.04) compared to extreme values (1818.56) in standard models, fundamentally altering LLM quantization behavior. Our work demonstrates that outliers are not inherent to LLMs but are consequences of training strategies, paving the way for more efficient LLM deployment. The source code and pretrained checkpoints are available at this https URL. 

**Abstract (ZH)**: 极端激活异常值在大规模语言模型（LLMs）中严重降低量化性能，阻碍了设备端的高效部署。虽然已识别出通道级操作和自适应梯度缩放为主要原因，但实际缓解措施仍具有挑战性。我们引入了安全预训练（OSP），这是一种实用指南，能够主动防止异常值形成而非依赖于事后缓解。OSP结合了三项关键创新：（1）Muon优化器，消除特权基底同时保持训练效率；（2）单一尺度RMSNorm，防止通道级放大；（3）可学习的嵌入投影，重新分配源自嵌入矩阵的激活幅度。通过在1万亿个标记上训练一个14亿参数的模型，我们验证了OSP，并展示了第一个在无此类异常值情况下的生产规模LLM。在激进的4位量化下，我们的OSP模型在10个基准测试中的平均得分为35.7（相比之下，Adam训练的模型得分为26.5），且仅增加了2%的训练开销。令人惊讶的是，OSP模型的超额峰度接近零（0.04），与标准模型中的极端值（1818.56）相比，从根本上改变了LLM的量化行为。我们的工作表明，异常值不是LLMs固有的，而是训练策略的后果，为更高效的LLM部署铺平了道路。源代码和预训练检查点可在以下链接获取。 

---
# Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager 

**Title (ZH)**: 面向LLMs的定制对话超越：基于RL的对话管理器 

**Authors**: Lucie Galland, Catherine Pelachaud, Florian Pecune  

**Link**: [PDF](https://arxiv.org/pdf/2506.19652)  

**Abstract**: In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals. 

**Abstract (ZH)**: 本研究提出了一种将大型语言模型与基于RL的对话管理器集成的新型框架，用于具有特定目标的开放性对话。通过利用层次强化学习建模对话的结构化阶段，并利用元学习提高跨多样化用户配置文件的适应性，我们的方法增强了系统的适应性和效率，使其能够从有限数据中学习，在对话阶段之间流畅转换，并个性化回应异质患者的需求。我们将该框架应用于动机访谈，旨在促进行为改变，并证明所提出的对话管理器在奖励方面优于最先进的LLM基线，显示出条件大型语言模型以创建具有特定目标的开放性对话系统具有潜在益处。 

---
# ECCoT: A Framework for Enhancing Effective Cognition via Chain of Thought in Large Language Model 

**Title (ZH)**: ECCoT：一种通过链式思考增强大型语言模型有效认知的框架 

**Authors**: Zhenke Duan, Jiqun Pan, Jiani Tu, Xiaoyi Wang, Yanqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19599)  

**Abstract**: In the era of large-scale artificial intelligence, Large Language Models (LLMs) have made significant strides in natural language processing. However, they often lack transparency and generate unreliable outputs, raising concerns about their interpretability. To address this, the Chain of Thought (CoT) prompting method structures reasoning into step-by-step deductions. Yet, not all reasoning chains are valid, and errors can lead to unreliable conclusions. We propose ECCoT, an End-to-End Cognitive Chain of Thought Validation Framework, to evaluate and refine reasoning chains in LLMs. ECCoT integrates the Markov Random Field-Embedded Topic Model (MRF-ETM) for topic-aware CoT generation and Causal Sentence-BERT (CSBert) for causal reasoning alignment. By filtering ineffective chains using structured ordering statistics, ECCoT improves interpretability, reduces biases, and enhances the trustworthiness of LLM-based decision-making. Key contributions include the introduction of ECCoT, MRF-ETM for topic-driven CoT generation, and CSBert for causal reasoning enhancement. Code is released at: this https URL. 

**Abstract (ZH)**: 大规模人工智能时代，大型语言模型在自然语言处理领域取得了显著进展，但往往缺乏透明性，生成的输出不可靠，这引发了对其可解释性的担忧。为了解决这一问题，Chain of Thought (CoT) 提问方法将推理构建成逐步推理链。然而，并非所有推理链都是有效的，错误可能导致不可靠的结论。我们提出了一种端到端认知链推理验证框架 ECCoT，用于评估和优化LLM中的推理链。ECCoT 结合了嵌入主题模型的马尔可夫随机场（MRF-ETM）进行主题驱动的CoT生成，以及因果句法BERT（CSBert）进行因果推理对齐。通过使用结构化的排序统计筛选无效链，ECCoT 提高了可解释性、减少了偏见并增强了基于LLM的决策可靠性。主要贡献包括 ECCoT 的引入、MRF-ETM 用于主题驱动的CoT生成，以及 CSBert 用于因果推理增强。代码发布在：this https URL。 

---
# PrivacyXray: Detecting Privacy Breaches in LLMs through Semantic Consistency and Probability Certainty 

**Title (ZH)**: PrivacyXray：通过语义一致性与概率确定性检测LLM中的隐私泄露 

**Authors**: Jinwen He, Yiyang Lu, Zijin Lin, Kai Chen, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19563)  

**Abstract**: Large Language Models (LLMs) are widely used in sensitive domains, including healthcare, finance, and legal services, raising concerns about potential private information leaks during inference. Privacy extraction attacks, such as jailbreaking, expose vulnerabilities in LLMs by crafting inputs that force the models to output sensitive information. However, these attacks cannot verify whether the extracted private information is accurate, as no public datasets exist for cross-validation, leaving a critical gap in private information detection during inference. To address this, we propose PrivacyXray, a novel framework detecting privacy breaches by analyzing LLM inner states. Our analysis reveals that LLMs exhibit higher semantic coherence and probabilistic certainty when generating correct private outputs. Based on this, PrivacyXray detects privacy breaches using four metrics: intra-layer and inter-layer semantic similarity, token-level and sentence-level probability distributions. PrivacyXray addresses critical challenges in private information detection by overcoming the lack of open-source private datasets and eliminating reliance on external data for validation. It achieves this through the synthesis of realistic private data and a detection mechanism based on the inner states of LLMs. Experiments show that PrivacyXray achieves consistent performance, with an average accuracy of 92.69% across five LLMs. Compared to state-of-the-art methods, PrivacyXray achieves significant improvements, with an average accuracy increase of 20.06%, highlighting its stability and practical utility in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗保健、金融和法律服务等敏感领域中广泛使用，引起了对推理过程中潜在隐私信息泄露的担忧。隐私提取攻击，如jailbreaking，通过构造输入来迫使模型输出敏感信息，揭示了LLMs的漏洞。然而，这些攻击无法验证提取的隐私信息是否准确，因为缺乏可用于交叉验证的公开数据集，从而在推理过程中隐私信息检测方面留下了一个关键缺口。为了填补这一缺口，我们提出了PrivacyXray，一个通过分析LLM内部状态检测隐私泄露的新型框架。我们的分析表明，当生成正确的隐私输出时，LLMs表现出更高的语义连贯性和概率 certainty。基于此，PrivacyXray 使用四种指标检测隐私泄露：层内和层间语义相似度以及词级和句级概率分布。PrivacyXray 通过合成现实的隐私数据并基于LLM内部状态的检测机制，克服了缺乏开源隐私数据集和验证过程对外部数据的依赖，实现了在五种大型语言模型中平均准确率92.69%的一致性能。与现有最佳方法相比，PrivacyXray 在平均准确率上提高了20.06%，突显了其在实际应用中的稳定性和实用性。 

---
# Automatic Posology Structuration : What role for LLMs? 

**Title (ZH)**: 自动给药剂量结构化：LLMs能发挥什么作用？ 

**Authors**: Natalia Bobkova, Laura Zanella-Calzada, Anyes Tafoughalt, Raphaël Teboul, François Plesse, Félix Gaschi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19525)  

**Abstract**: Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use. 

**Abstract (ZH)**: 自动结构化用药指导对于提高药物安全性和促进临床决策支持至关重要。在法语处方中，这些指导往往是模糊的、不规则的或口语化的，限制了经典机器学习管道的有效性。我们探索了使用大型语言模型（LLMs）将自由文本用药指导转换为结构化格式的方法，对比了基于提示的方法和基于命名实体识别与链接（NERL）的微调前LLM系统。结果显示，虽然提示可以提高性能，但仅微调的LLMs能够匹配基线的准确性。通过错误分析，我们发现NERL和LLMs各有优势：NERL提供结构上的精确性，而LLMs更好地处理语义细微差别。基于此，我们提出了一种混合管道，将NERL中低置信度（<0.8）的情况路由到LLM，并根据置信分数选择输出。这一策略实现了91%的结构化准确率，同时将延迟和计算量降至最低。我们的结果显示，这种混合方法在提高结构化准确率的同时，减少了计算成本，提供了一种可扩展的现实临床应用解决方案。 

---
# MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications 

**Title (ZH)**: MATE: LLM驱动的多代理翻译环境及其在无障碍应用中的应用 

**Authors**: Aleksandr Algazinov, Matt Laing, Paul Laban  

**Link**: [PDF](https://arxiv.org/pdf/2506.19502)  

**Abstract**: Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 多模态 Accessibility MAS MATE：基于用户需求的模态转换与多模态辅助 

---
# Dialogic Pedagogy for Large Language Models: Aligning Conversational AI with Proven Theories of Learning 

**Title (ZH)**: 大型语言模型的对话式教学法：将对话式AI与 proven 学习理论对齐 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.19484)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming education by enabling rich conversational learning experiences. This article provides a comprehensive review of how LLM-based conversational agents are being used in higher education, with extensions to secondary and lifelong learning contexts. We synthesize existing literature on LLMs in education and theories of conversational and dialogic pedagogy - including Vygotsky's sociocultural learning (scaffolding and the Zone of Proximal Development), the Socratic method, and Laurillard's conversational framework - and examine how prompting strategies and retrieval-augmented generation (RAG) can align LLM behaviors with these pedagogical theories, and how it can support personalized, adaptive learning. We map educational theories to LLM capabilities, highlighting where LLM-driven dialogue supports established learning principles and where it challenges or falls short of traditional pedagogical assumptions. Notable gaps in applying prior theories to LLMs are identified, such as the models tendency to provide direct answers instead of fostering co-construction of knowledge, and the need to account for the constant availability and broad but non-human expertise of LLM tutors. In response, we propose practical strategies to better align LLM interactions with sound pedagogy - for example, designing prompts that encourage Socratic questioning, scaffolded guidance, and student reflection, as well as integrating retrieval mechanisms to ensure accuracy and contextual relevance. Our aim is to bridge the gap between educational theory and the emerging practice of AI-driven conversational learning, offering insights and tools for making LLM-based dialogues more educationally productive and theory-aligned. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在迅速改变教育领域，通过提供丰富的对话式学习体验。本文提供了一个全面的综述，探讨基于LLM的对话代理在高等教育中的应用，并扩展到中等教育和终身学习情境。我们综合了现有教育中LLM的相关文献，并分析了对话式和对话式教学理论——包括维果茨基的社会文化学习（支架教学和最近发展区）、苏格拉底方法以及劳拉利迪的对话框架——考察了提示策略和检索增强生成（RAG）如何使LLM行为与这些教学理论相一致，并支持个性化的适应性学习。我们将教育理论与LLM能力进行映射，强调LLM驱动的对话如何支持既定的学习原则，以及如何挑战或未能满足传统的教学假设。我们指出了将先前理论应用于LLM时的显著缺口，例如模型倾向于提供直接答案而非促进知识的共同建构，以及需要考虑LLM导师的持续可用性和广泛但非人性化的专业知识。为应对这些问题，我们提出了实用策略，以更好地将LLM交互与良好的教学实践对齐——例如，设计促进苏格拉底式提问、支架式指导和学生反思的提示，以及整合检索机制以确保准确性和情境相关性。我们的目标是弥合教育理论与新兴的以AI驱动的对话式学习实践之间的差距，提供使基于LLM的对话更加教育有效并理论对齐的见解和工具。 

---
# MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages 

**Title (ZH)**: MuBench: 大型语言模型跨61种语言的多语言能力评估 

**Authors**: Wenhan Han, Yifan Zhang, Zhixun Chen, Binbin Liu, Haobin Lin, Bingni Zhang, Taifeng Wang, Mykola Pechenizkiy, Meng Fang, Yin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.19468)  

**Abstract**: Multilingual large language models (LLMs) are advancing rapidly, with new models frequently claiming support for an increasing number of languages. However, existing evaluation datasets are limited and lack cross-lingual alignment, leaving assessments of multilingual capabilities fragmented in both language and skill coverage. To address this, we introduce MuBench, a benchmark covering 61 languages and evaluating a broad range of capabilities. We evaluate several state-of-the-art multilingual LLMs and find notable gaps between claimed and actual language coverage, particularly a persistent performance disparity between English and low-resource languages. Leveraging MuBench's alignment, we propose Multilingual Consistency (MLC) as a complementary metric to accuracy for analyzing performance bottlenecks and guiding model improvement. Finally, we pretrain a suite of 1.2B-parameter models on English and Chinese with 500B tokens, varying language ratios and parallel data proportions to investigate cross-lingual transfer dynamics. 

**Abstract (ZH)**: 多语言大规模语言模型（LLMs）正迅速发展，新模型经常声称支持越来越多的语言。然而，现有的评估数据集有限且缺乏跨语言对齐，使得对多语言能力的评估在语言和技能覆盖上碎片化。为解决这一问题，我们介绍了MuBench，这是一个覆盖61种语言并评估广泛能力的基准。我们评估了几种最先进的多语言LLM，并发现它们在声称和实际语言覆盖之间存在显著差距，特别是在英语和低资源语言之间持续存在性能差异。利用MuBench的对齐，我们提出了多语言一致性（MLC）作为准确性的一种补充指标，用于分析性能瓶颈并指导模型改进。最后，我们在500亿个标记上分别对1.2B参数量的英语和汉语模型进行了预训练，变化语言比例和并行数据比例以研究跨语言迁移动态。 

---
# Can Large Language Models Capture Human Annotator Disagreements? 

**Title (ZH)**: 大型语言模型能否捕捉到人类标注者的分歧？ 

**Authors**: Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Alexander Hoyle, Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19467)  

**Abstract**: Human annotation variation (i.e., annotation disagreements) is common in NLP and often reflects important information such as task subjectivity and sample ambiguity. While Large Language Models (LLMs) are increasingly used for automatic annotation to reduce human effort, their evaluation often focuses on predicting the majority-voted "ground truth" labels. It is still unclear, however, whether these models also capture informative human annotation variation. Our work addresses this gap by extensively evaluating LLMs' ability to predict annotation disagreements without access to repeated human labels. Our results show that LLMs struggle with modeling disagreements, which can be overlooked by majority label-based evaluations. Notably, while RLVR-style (Reinforcement learning with verifiable rewards) reasoning generally boosts LLM performance, it degrades performance in disagreement prediction. Our findings highlight the critical need for evaluating and improving LLM annotators in disagreement modeling. Code and data at this https URL. 

**Abstract (ZH)**: 人类注释变异（即注释分歧）在NLP中很常见，往往反映了任务的主观性和样本的模糊性。虽然大型语言模型（LLMs）越来越多地用于自动注释以减少人力投入，但对其评价往往集中在预测多数票表决的“地面真实”标签上。然而，尚不清楚这些模型是否也能捕捉到有意义的人类注释变异。我们的工作通过广泛评估LLMs在无法访问重复人类标签的情况下预测注释分歧的能力，填补了这一空白。研究结果表明，LLMs在建模分歧方面存在困难，这一问题可能会被基于多数标签的评价所忽视。值得注意的是，虽然验证奖励强化学习风格的推理通常能提升LLM性能，但会损害分歧预测性能。我们的发现强调了评估和改进LLM注释器在分歧建模方面的关键需求。代码和数据见此链接。 

---
# Automated Detection of Pre-training Text in Black-box LLMs 

**Title (ZH)**: 自动检测黑盒大语言模型中的预训练文本 

**Authors**: Ruihan Hu, Yu-Ming Shang, Jiankun Peng, Wei Luo, Yazhe Wang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19399)  

**Abstract**: Detecting whether a given text is a member of the pre-training data of Large Language Models (LLMs) is crucial for ensuring data privacy and copyright protection. Most existing methods rely on the LLM's hidden information (e.g., model parameters or token probabilities), making them ineffective in the black-box setting, where only input and output texts are accessible. Although some methods have been proposed for the black-box setting, they rely on massive manual efforts such as designing complicated questions or instructions. To address these issues, we propose VeilProbe, the first framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. VeilProbe utilizes a sequence-to-sequence mapping model to infer the latent mapping feature between the input text and the corresponding output suffix generated by the LLM. Then it performs the key token perturbations to obtain more distinguishable membership features. Additionally, considering real-world scenarios where the ground-truth training text samples are limited, a prototype-based membership classifier is introduced to alleviate the overfitting issue. Extensive evaluations on three widely used datasets demonstrate that our framework is effective and superior in the black-box setting. 

**Abstract (ZH)**: 检测给定文本是否属于大型语言模型（LLMs）的预训练数据对于确保数据隐私和版权保护至关重要。现有方法大多依赖于模型的隐藏信息（如模型参数或token概率），在只能访问输入和输出文本的黑盒设置中效果不佳。尽管已有部分方法针对黑盒设置进行了研究，但这些方法依赖大量的手工努力，如设计复杂的问题或指令。为解决这些问题，我们提出VeilProbe，这是第一个无需人工干预即可在黑盒设置中自动检测LLMs预训练文本的框架。VeilProbe利用序列到序列映射模型推断输入文本与生成的相应输出后缀之间的潜在映射特征，然后通过关键token扰动获得更易区分的成员特征。此外，考虑到实际场景中真实训练文本样本有限的情况，引入了原型基成员分类器以缓解过拟合问题。广泛的数据集评估表明，我们的框架在黑盒设置中有效且更具优势。 

---
# Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation 

**Title (ZH)**: 识别不符合角色的行为：开放生成中人物忠诚度的原子级评估 

**Authors**: Jisu Shin, Juhyun Oh, Eunsu Kim, Hoyun Song, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19352)  

**Abstract**: Ensuring persona fidelity in large language models (LLMs) is essential for maintaining coherent and engaging human-AI interactions. However, LLMs often exhibit Out-of-Character (OOC) behavior, where generated responses deviate from an assigned persona, leading to inconsistencies that affect model reliability. Existing evaluation methods typically assign single scores to entire responses, struggling to capture subtle persona misalignment, particularly in long-form text generation. To address this limitation, we propose an atomic-level evaluation framework that quantifies persona fidelity at a finer granularity. Our three key metrics measure the degree of persona alignment and consistency within and across generations. Our approach enables a more precise and realistic assessment of persona fidelity by identifying subtle deviations that real users would encounter. Through our experiments, we demonstrate that our framework effectively detects persona inconsistencies that prior methods overlook. By analyzing persona fidelity across diverse tasks and personality types, we reveal how task structure and persona desirability influence model adaptability, highlighting challenges in maintaining consistent persona expression. 

**Abstract (ZH)**: 确保大规模语言模型的身份 fidelity 对保持连贯且引人入胜的人机互动至关重要。然而，大规模语言模型经常会表现出脱臼 (OOC) 行为，即生成的响应偏离分配的身份，导致不一致，影响模型的可靠性。现有的评估方法通常为整个响应赋予单一分数，难以捕捉细微的身份对齐偏差，尤其是在长文本生成中。为解决这一局限性，我们提出了一种原子级评估框架，以更细粒度量化身份 fidelity。我们的三个关键指标衡量身份对齐和一致性的程度，在生成之间进行。通过我们的方法，可以通过识别真实用户会遇到的细微偏差，实现更精确和现实的身份 fidelity 评估。通过实验，我们证明了我们的框架能够检测出先前方法未能发现的身份不一致性。通过对多种任务和人格类型的身份 fidelity 进行分析，我们揭示了任务结构和身份吸引力如何影响模型的适应性，强调了保持一致身份表达的挑战。 

---
# Thought Anchors: Which LLM Reasoning Steps Matter? 

**Title (ZH)**: 思维锚点：哪些大规模语言模型的推理步骤重要？ 

**Authors**: Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19143)  

**Abstract**: Reasoning large language models have recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified ``broadcasting'' sentences that receive disproportionate attention from all future sentences via ``receiver'' attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (this http URL) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models. 

**Abstract (ZH)**: 大型语言模型 recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified ``broadcasting'' sentences that receive disproportionate attention from all future sentences via ``receiver'' attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (this http URL) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models. 

---
# Enhancing Security in LLM Applications: A Performance Evaluation of Early Detection Systems 

**Title (ZH)**: 增强LLM应用的安全性：早期检测系统性能评估 

**Authors**: Valerii Gakh, Hayretdin Bahsi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19109)  

**Abstract**: Prompt injection threatens novel applications that emerge from adapting LLMs for various user tasks. The newly developed LLM-based software applications become more ubiquitous and diverse. However, the threat of prompt injection attacks undermines the security of these systems as the mitigation and defenses against them, proposed so far, are insufficient. We investigated the capabilities of early prompt injection detection systems, focusing specifically on the detection performance of techniques implemented in various open-source solutions. These solutions are supposed to detect certain types of prompt injection attacks, including the prompt leak. In prompt leakage attacks, an attacker maliciously manipulates the LLM into outputting its system instructions, violating the system's confidentiality. Our study presents analyzes of distinct prompt leakage detection techniques, and a comparative analysis of several detection solutions, which implement those techniques. We identify the strengths and weaknesses of these techniques and elaborate on their optimal configuration and usage in high-stake deployments. In one of the first studies on existing prompt leak detection solutions, we compared the performances of LLM Guard, Vigil, and Rebuff. We concluded that the implementations of canary word checks in Vigil and Rebuff were not effective at detecting prompt leak attacks, and we proposed improvements for them. We also found an evasion weakness in Rebuff's secondary model-based technique and proposed a mitigation. Then, the result of the comparison of LLM Guard, Vigil, and Rebuff at their peak performance revealed that Vigil is optimal for cases when minimal false positive rate is required, and Rebuff is the most optimal for average needs. 

**Abstract (ZH)**: 提示注入威胁着由适应各种用户任务的LLM衍生出的新颖应用。新开发的基于LLM的软件应用变得越来越普遍和多样化。然而，提示注入攻击的威胁削弱了这些系统的安全性，因为目前提出的缓解和防御措施尚不足够。我们研究了早期提示注入检测系统的能力，特别是各种开源解决方案中实现的技术的检测性能。这些解决方案旨在检测某些类型的提示注入攻击，包括提示泄露。在提示泄露攻击中，攻击者故意操控LLM输出其系统指令，违反了系统的机密性。我们的研究分析了不同的提示泄露检测技术，并进行了几种检测解决方案的比较分析，这些解决方案实现了这些技术。我们确定了这些技术的优点和缺点，并详细说明了它们在高风险部署中的最佳配置和使用方法。在对现有提示泄露检测解决方案的首个研究中，我们将LLM Guard、Vigil和Rebuff的性能进行了比较。我们得出结论，Vigil和Rebuff中的canary词检查实现对于检测提示泄露攻击不是有效的，并提出了改进措施。我们还发现Rebuff基于模型的第二种技术存在规避漏洞，并提出了缓解措施。然后，在LLM Guard、Vigil和Rebuff在最佳性能下比较的结果表明，当需要最低误报率时，Vigil是最优选择，而在一般需求下，Rebuff是最优选择。 

---
# Improving Student-AI Interaction Through Pedagogical Prompting: An Example in Computer Science Education 

**Title (ZH)**: 通过教学提示改善学生与AI的互动：以计算机科学教育为例 

**Authors**: Ruiwei Xiao, Xinying Hou, Runlong Ye, Majeed Kazemitabaar, Nicholas Diana, Michael Liut, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2506.19107)  

**Abstract**: With the proliferation of large language model (LLM) applications since 2022, their use in education has sparked both excitement and concern. Recent studies consistently highlight students' (mis)use of LLMs can hinder learning outcomes. This work aims to teach students how to effectively prompt LLMs to improve their learning. We first proposed pedagogical prompting, a theoretically-grounded new concept to elicit learning-oriented responses from LLMs. To move from concept design to a proof-of-concept learning intervention in real educational settings, we selected early undergraduate CS education (CS1/CS2) as the example context. We began with a formative survey study with instructors (N=36) teaching early-stage undergraduate-level CS courses to inform the instructional design based on classroom needs. Based on their insights, we designed and developed a learning intervention through an interactive system with scenario-based instruction to train pedagogical prompting skills. Finally, we evaluated its instructional effectiveness through a user study with CS novice students (N=22) using pre/post-tests. Through mixed methods analyses, our results indicate significant improvements in learners' LLM-based pedagogical help-seeking skills, along with positive attitudes toward the system and increased willingness to use pedagogical prompts in the future. Our contributions include (1) a theoretical framework of pedagogical prompting; (2) empirical insights into current instructor attitudes toward pedagogical prompting; and (3) a learning intervention design with an interactive learning tool and scenario-based instruction leading to promising results on teaching LLM-based help-seeking. Our approach is scalable for broader implementation in classrooms and has the potential to be integrated into tools like ChatGPT as an on-boarding experience to encourage learning-oriented use of generative AI. 

**Abstract (ZH)**: 自2022年以来，大型语言模型（LLM）应用的普及引发了教育中的兴奋与担忧。近期研究表明，学生对LLM的不当使用可能阻碍学习成果。本文旨在教授学生如何有效提示LLM以改善学习。我们首先提出了基于教学理论的新概念——教学提示，以激发LLM的学习导向响应。为了将概念设计转化为实际教育环境中的原理验证学习干预，我们选取了早期本科生计算机科学教育（CS1/CS2）作为示例背景。我们首先进行了一个形成性调查研究，征求36名教师的意见，以根据课堂需求告知教学设计。根据他们的见解，我们设计并开发了一个基于情境的教学干预工具，通过互动系统进行情境教学，以训练教学提示技能。最终，我们通过一项用户研究（参与者总数22名，使用前后测）评估了其教学有效性。通过混合方法分析，结果显示，在基于LLM的需求指导帮助技能方面取得了显著提高，且对系统的态度积极，并且在未来使用教学提示方面表现出更高的意愿。我们的贡献包括：（1）教学提示的理论框架；（2）当前教师对教学提示态度的经验见解；以及（3）一个通过互动学习工具和情境教学设计的教学干预方案，该方案在教学基于LLM的需求指导方面取得了有前景的结果。我们的方法具有扩展到更广泛课堂环境的潜力，并有可能整合到类似于ChatGPT的工具中，作为入门体验以促进生成性AI的教育导向使用。 

---
# Language Models Might Not Understand You: Evaluating Theory of Mind via Story Prompting 

**Title (ZH)**: 语言模型可能无法理解你：通过故事提示评估共情理解能力 

**Authors**: Nathaniel Getachew, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2506.19089)  

**Abstract**: We introduce $\texttt{StorySim}$, a programmable framework for synthetically generating stories to evaluate the theory of mind (ToM) and world modeling (WM) capabilities of large language models (LLMs). Unlike prior benchmarks that may suffer from contamination in pretraining data, $\texttt{StorySim}$ produces novel, compositional story prompts anchored by a highly controllable $\texttt{Storyboard}$, enabling precise manipulation of character perspectives and events. We use this framework to design first- and second-order ToM tasks alongside WM tasks that control for the ability to track and model mental states. Our experiments across a suite of state-of-the-art LLMs reveal that most models perform better on WM tasks than ToM tasks, and that models tend to perform better reasoning with humans compared to inanimate objects. Additionally, our framework enabled us to find evidence of heuristic behavior such as recency bias and an over-reliance on earlier events in the story. All code for generating data and evaluations is freely available. 

**Abstract (ZH)**: 我们介绍了$\texttt{StorySim}$，这是一种可编程框架，用于合成生成故事以评估大规模语言模型（LLMs）的理论思维（ToM）和世界建模（WM）能力。与可能受到预训练数据污染的先前基准不同，$\texttt{StorySim}$通过一个高度可控的$\texttt{Storyboard}$产生新颖的、组件化的故事提示，从而能够精确操控角色视角和事件。我们使用此框架设计了与WM任务相结合的第一级和第二级ToM任务，以控制追踪和建模心理状态的能力。我们的实验结果表明，大多数模型在WM任务上的表现优于ToM任务，而且模型在与人类进行推理时的表现通常优于与无生命物体进行推理。此外，我们的框架帮助我们发现了诸如近期偏差和对故事早期事件过度依赖等启发式行为的证据。所有生成数据和评估的代码均可免费获取。 

---
# FairCauseSyn: Towards Causally Fair LLM-Augmented Synthetic Data Generation 

**Title (ZH)**: FairCauseSyn: 向量化因果公平的LLM增强合成数据生成 

**Authors**: Nitish Nagesh, Ziyu Wang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19082)  

**Abstract**: Synthetic data generation creates data based on real-world data using generative models. In health applications, generating high-quality data while maintaining fairness for sensitive attributes is essential for equitable outcomes. Existing GAN-based and LLM-based methods focus on counterfactual fairness and are primarily applied in finance and legal domains. Causal fairness provides a more comprehensive evaluation framework by preserving causal structure, but current synthetic data generation methods do not address it in health settings. To fill this gap, we develop the first LLM-augmented synthetic data generation method to enhance causal fairness using real-world tabular health data. Our generated data deviates by less than 10% from real data on causal fairness metrics. When trained on causally fair predictors, synthetic data reduces bias on the sensitive attribute by 70% compared to real data. This work improves access to fair synthetic data, supporting equitable health research and healthcare delivery. 

**Abstract (ZH)**: 合成数据生成技术基于生成模型在真实世界数据的基础上创建数据。在健康应用中，生成高质量数据并保持敏感属性的公平性对于实现公平结果至关重要。现有的基于GAN和基于LLM的方法主要关注反事实公平性，并主要应用于金融和法律领域。因果公平性提供了一个更全面的评估框架，通过保留因果结构来实现，但当前的合成数据生成方法尚未在健康领域解决这一问题。为了填补这一空白，我们开发了首个增强因果公平性的LLM增强合成数据生成方法，使用真实世界的表格健康数据。我们生成的数据在因果公平性指标上的偏差小于10%。当使用因果公平预测器进行训练时，合成数据将敏感属性上的偏差减少70%，相较于真实数据。这项工作提高了公平合成数据的获取，支持了公平的健康研究和医疗服务。 

---
# Plan for Speed -- Dilated Scheduling for Masked Diffusion Language Models 

**Title (ZH)**: Plan for Speed —— 考虑延时的掩码扩散语言模型调度策略 

**Authors**: Omer Luxembourg, Haim Permuter, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2506.19037)  

**Abstract**: Masked diffusion language models (MDLM) have shown strong promise for non-autoregressive text generation, yet existing samplers act as implicit planners, selecting tokens to unmask via denoiser confidence or entropy scores. Such heuristics falter under parallel unmasking - they ignore pairwise interactions between tokens and cannot account for dependencies when unmasking multiple positions at once, limiting their inference time to traditional auto-regressive (AR) models. We introduce the Dilated-scheduled Unmasking Strategy (DUS), an inference-only, planner-model-free method that requires no additional training. DUS leverages a first-order Markov assumption to partition sequence positions into dilation-based groups of non-adjacent tokens, enabling independent, parallel unmasking steps that respect local context that minimizes the joint entropy of each iteration step. Unlike semi-AR block approaches (e.g., LLADA and Dream) that still invoke the denoiser per block, DUS reduces the number of denoiser calls to O(log B) per generation block - yielding substantial speedup over the O(B) run time of state-of-the-art diffusion models, where B is the block size in the semi-AR inference process. In experiments on math (GSM8K) and code completion (Humaneval, MBPP) benchmarks - domains suited to non-ordinal generation - DUS improves scores over parallel confidence-based planner, without modifying the underlying denoiser. DUS offers a lightweight, budget-aware approach to efficient, high-quality text generation, paving the way to unlock the true capabilities of MDLMs. 

**Abstract (ZH)**: Masked扩散语言模型（MDLM）在非自回归文本生成中表现出强大的潜力，现有采样器作为隐式规划者，通过去噪器信心分数或熵分数选择解码令牌。这些启发式方法在并行解码时效果不佳——它们忽略了令牌间的两两交互，无法在同一时间解码多个位置时处理依赖关系，限制了它们的推理时间到传统的自回归（AR）模型。我们引入了扩展调度解码策略（DUS），这是一种仅用于推理、无需额外训练的规划模型自由方法。DUS 利用一阶马尔可夫假设将序列位置划分为基于扩张组的非相邻令牌群，从而实现尊重局部上下文、并在每一步迭代中最小化联合熵的独立并行解码步骤。与半自回归块方法（例如LLADA和Dream）相比，DUS 每个生成块中去噪器调用次数减少到O(log B) - 与最先进的扩散模型相比，这提供了一个显著的速度提升，其中B是半自回归推理过程中块的大小。在数学（GSM8K）和代码完成（Humaneval、MBPP）基准测试中，DUS 在无需修改底层去噪器的情况下提高了基于信心的并行解码器的评分。DUS 提供了一种轻量级、预算意识的方法，用于高效、高质量的文本生成，从而揭示了MDLMs的真实能力。 

---
# Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective 

**Title (ZH)**: 超越标记：从语义和统计视角量化LLM的公平性 

**Authors**: Weijie Xu, Yiwen Wang, Chi Xue, Xiangkun Hu, Xi Fang, Guimin Dong, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19028)  

**Abstract**: Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics. 

**Abstract (ZH)**: 大规模语言模型（LLMs）往往生成带有内在偏见的响应，这会削弱其在现实世界应用中的可靠性。现有的评估方法常常忽视长文响应中的偏见以及大规模语言模型输出的固有变异性。为应对这些挑战，我们提出了一种新颖的统计框架FiSCo（细粒度语义计算），该框架通过检测不同人群组在长文响应中细微语义差异来评估LLMs的整体公平性。与以往侧重情感或标记级别比较的工作不同，FiSCo通过在声明级别进行操作，并利用蕴含检查评估响应之间意义的一致性，超越了表面层次的分析。我们将模型输出分解为语义上不同的声明，并应用假设检验进行组内和组间相似性的比较，实现对细微偏见的稳健检测。我们形式化了一种新的群体事实性公平性定义，并在涵盖性别、种族和年龄的人工标注数据集上验证了FiSCo。实验表明，FiSCo更可靠地识别细微的偏见，减少了随机性对LLM变异的影响，并在各种评估指标中表现出色。 

---
# LLMs on a Budget? Say HOLA 

**Title (ZH)**: 有限预算下的大型语言模型？说HOLA 

**Authors**: Zohaib Hasan Siddiqui, Jiechao Gao, Ebad Shabbir, Mohammad Anas Azeez, Rafiq Ali, Gautam Siddharth Kashyap, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.18952)  

**Abstract**: Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready. 

**Abstract (ZH)**: 在边缘设备上运行大规模语言模型（LLMs）受到高计算和内存需求的限制，这为医疗保健、教育和嵌入式系统等领域的实时应用设置了障碍。当前的解决方案如量化、剪枝和检索增强生成（RAG）仅提供部分优化，常常在速度或准确性上做出妥协。我们介绍了HOLA，一个端到端的高效LLM部署优化框架。内部，HOLA利用分层推测解码（HSD）进行更快的推理而不损失质量。外部，AdaComp-RAG根据上下文需求调整检索复杂度。结合LoBi，该框架融合了结构化剪枝（LoRA）和量化，实现显著收益：在GSM8K上的17.6% EMA，在ARC上的10.5% MCA，并减少Jetson Nano等边缘设备上的延迟和内存消耗，证明了其可扩展性和生产就绪性。 

---
# SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications 

**Title (ZH)**: SWE-SQL: 照亮大规模语言模型解决实际应用场景中用户SQL问题的道路 

**Authors**: Jinyang Li, Xiaolong Li, Ge Qu, Per Jacobsson, Bowen Qin, Binyuan Hui, Shuzheng Si, Nan Huo, Xiaohan Xu, Yue Zhang, Ziwei Tang, Yuanshuai Li, Florensia Widjaja, Xintong Zhu, Feige Zhou, Yongfeng Huang, Yannis Papakonstantinou, Fatma Ozcan, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18951)  

**Abstract**: Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: this https URL 

**Abstract (ZH)**: 复杂SQL问题的解决仍然是现实数据库应用中的一个显著瓶颈。当前的大语言模型虽擅长文本到SQL的翻译，但在调试SQL问题这一更具挑战性的任务上尚未得到严格评估。为填补这一空白，我们引入了BIRD-CRITIC，这是一个新的SQL问题调试基准，包括530个PostgreSQL任务（BIRD-CRITIC-PG）和570个多方言任务（BIRD-CRITIC-Multi），这些任务源自真实用户问题，并在新环境中重演，以促进严格的评估。基线评估凸显了任务的复杂性，领先的推理模型O3-Mini在BIRD-CRITIC-PG上的成功率仅为38.87%，在BIRD-CRITIC-Multi上的成功率仅为33.33%。同时，为数据库任务开发开源模型对于增强本地开发能力并保护数据隐私至关重要。因此，我们提出了Six-Gym（Sql-fIX-Gym），一个用于提升开源模型SQL问题调试能力的训练环境。该环境利用SQL-Rewind策略，通过逆向工程验证的SQL问题自动生成可执行的问题解决方案数据集。然而，流行的基于轨迹的微调方法并未探索实质性的监督信号。我们进一步提出了f-Plan Boosting，它从SQL解决方案中提取高级调试计划，使教师大语言模型能够生成73.7%更多的成功轨迹用于训练。我们将这些组件集成到开源代理Bird-Fixer中。基于Qwen-2.5-Coder-14B的Bird-Fixer在BIRD-CRITIC-PG上的成功率为38.11%，在BIRD-CRITIC-Multi上的成功率为29.65%，超过了领先的商用模型如Claude-3.7-Sonnet和GPT-4.1，标志着朝着普及复杂SQL调试能力的重要一步。排行榜和源代码可在此访问：this https URL。 

---
# Safe Pruning LoRA: Robust Distance-Guided Pruning for Safety Alignment in Adaptation of LLMs 

**Title (ZH)**: Safe Pruning LoRA: 基于稳健距离指导的安全剪枝以在LLM适应中实现安全性对齐 

**Authors**: Shuang Ao, Yi Dong, Jinwei Hu, Sarvapali Ramchurn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18931)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with Low-Rank Adaptation (LoRA) enhances adaptability while reducing computational costs. However, fine-tuning can compromise safety alignment, even with benign data, increasing susceptibility to harmful outputs. Existing safety alignment methods struggle to capture complex parameter shifts, leading to suboptimal safety-utility trade-offs. To address this issue, we propose Safe Pruning LoRA (SPLoRA), a novel pruning-based approach that selectively removes LoRA layers that weaken safety alignment, improving safety while preserving performance. At its core, we introduce Empirical-DIEM (E-DIEM), a dimension-insensitive similarity metric that effectively detects safety misalignment in LoRA-adapted models. We conduct extensive experiments on LLMs fine-tuned with mixed of benign and malicious data, and purely benign datasets, evaluating SPLoRA across utility, safety, and reliability metrics. Results demonstrate that SPLoRA outperforms state-of-the-art safety alignment techniques, significantly reducing safety risks while maintaining or improving model performance and reliability. Additionally, SPLoRA reduces inference overhead, making it a scalable and efficient solution for deploying safer and more reliable LLMs. The code is available at this https URL. 

**Abstract (ZH)**: 使用低秩适应（LoRA）微调大型语言模型（LLMs）增强了适应性并降低了计算成本，但微调可能导致安全性对齐受损，即使使用良性数据也是如此，增加了有害输出的易感性。现有的安全性对齐方法难以捕捉复杂的参数变化，导致安全性和实用性之间的次优权衡。为了应对这一问题，我们提出了一种新的剪枝方法Safe Pruning LoRA（SPLoRA），通过选择性地移除削弱安全性对齐的LoRA层，提高安全性并保持性能。核心上，我们引入了Empirical-DIEM（E-DIEM），这是一种维度感知不变的相似度度量，有效地检测LoRA适应模型中的安全性对齐偏差。我们在使用良性与恶意数据混合以及纯粹良性数据微调的大型语言模型上进行了广泛实验，评估了SPLoRA在实用性、安全性和可靠性指标上的表现。结果表明，SPLoRA在减少安全风险的同时，保持或提高了模型性能和可靠性，优于现有最先进的安全性对齐技术。此外，SPLoRA减少了推理开销，使其成为部署更安全和更可靠的大型语言模型的可扩展和高效解决方案。代码可在以下链接获取。 

---
# Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases 

**Title (ZH)**: 隐私保护的大语言模型交互：基于苏格拉底链式思维推理和同态加密向量数据库 

**Authors**: Yubeen Bae, Minchan Kim, Jaejin Lee, Sangbum Kim, Jaehyung Kim, Yejin Choi, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.17336)  

**Abstract**: Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy. 

**Abstract (ZH)**: 大型语言模型（LLMs）正日益被用作个人代理，访问用户的敏感数据，如日历、邮件和医疗记录。当前，用户面临权衡：他们可以选择将许多存储在远程数据库中的私人记录发送给强大但不可信的LLM提供商，从而增加暴露风险；或者在可信赖的设备上运行较弱的本地模型。我们填补了这一缺口。我们的苏格拉底式链式推理首先将通用的非私人用户查询发送给强大但不可信的LLM，LLM生成链式推理（CoT）提示和详细的子查询，而无需访问用户数据。接着，我们嵌入这些子查询，并使用我们的同态加密向量数据库对用户的百万条私人数据记录进行加密子秒语义搜索。这代表了多年数字活动中积累的个人文档、邮件和记录的现实规模。最后，我们将CoT提示和解密后的记录输入本地语言模型，生成最终回答。在LoCoMo长上下文问答基准测试中，我们的混合框架结合使用GPT-4o与本地Llama-3.2-1B模型，比单独使用GPT-4o高出7.1个百分点。这展示了将任务分解并分配给不可信的强大LLM和弱本地模型的一种可能步骤，同时保护用户隐私。 

---
# Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models 

**Title (ZH)**: 回收网络：增强语言模型预训练数据质量与数量的方法 

**Authors**: Thao Nguyen, Yang Li, Olga Golovneva, Luke Zettlemoyer, Sewoong Oh, Ludwig Schmidt, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04689)  

**Abstract**: Scaling laws predict that the performance of large language models improves with increasing model size and data size. In practice, pre-training has been relying on massive web crawls, using almost all data sources publicly available on the internet so far. However, this pool of natural data does not grow at the same rate as the compute supply. Furthermore, the availability of high-quality texts is even more limited: data filtering pipelines often remove up to 99% of the initial web scrapes to achieve state-of-the-art. To address the "data wall" of pre-training scaling, our work explores ways to transform and recycle data discarded in existing filtering processes. We propose REWIRE, REcycling the Web with guIded REwrite, a method to enrich low-quality documents so that they could become useful for training. This in turn allows us to increase the representation of synthetic data in the final pre-training set. Experiments at 1B, 3B and 7B scales of the DCLM benchmark show that mixing high-quality raw texts and our rewritten texts lead to 1.0, 1.3 and 2.5 percentage points improvement respectively across 22 diverse tasks, compared to training on only filtered web data. Training on the raw-synthetic data mix is also more effective than having access to 2x web data. Through further analysis, we demonstrate that about 82% of the mixed in texts come from transforming lower-quality documents that would otherwise be discarded. REWIRE also outperforms related approaches of generating synthetic data, including Wikipedia-style paraphrasing, question-answer synthesizing and knowledge extraction. These results suggest that recycling web texts holds the potential for being a simple and effective approach for scaling pre-training data. 

**Abstract (ZH)**: 基于缩放定律的预训练数据回收方法：REWIRE 

---
