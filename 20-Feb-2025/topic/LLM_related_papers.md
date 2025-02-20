# Neurosymbolic artificial intelligence via large language models and coherence-driven inference 

**Title (ZH)**: 通过大型语言模型和连贯驱动推理的神经符号人工智能 

**Authors**: Steve Huntsman, Jewell Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2502.13953)  

**Abstract**: We devise an algorithm to generate sets of propositions that objectively instantiate graphs that support coherence-driven inference. We then benchmark the ability of large language models (LLMs) to reconstruct coherence graphs from (a straightforward transformation of) propositions expressed in natural language, with promising results from a single prompt to models optimized for reasoning. Combining coherence-driven inference with consistency evaluations by neural models may advance the state of the art in machine cognition. 

**Abstract (ZH)**: 我们设计了一种算法以生成客观实例化支持共现推理的图的命题集。然后我们通过自然语言中表达的命题的直接变换测试了大语言模型（LLMs）重建共现图的能力，单个提示到推理优化模型都取得了令人鼓舞的结果。结合共现推理与神经模型的一致性评估可能推动机器认知领域的前沿。 

---
# Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning 

**Title (ZH)**: 利用LLMs与符号推理协同证明奥林匹克不等式 

**Authors**: Zenan Li, Zhaoyu Li, Wen Tang, Xian Zhang, Yuan Yao, Xujie Si, Fan Yang, Kaiyu Yang, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.13834)  

**Abstract**: Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data. 

**Abstract (ZH)**: 大规模语言模型可以通过在证明系统内生成证明步骤（即窍门）来形式化地证明数学定理。然而，可能的窍门空间 vast 而复杂，可供训练的形式证明数据有限，这给基于大规模语言模型的窍门生成带来了显著挑战。为应对这一挑战，我们提出了一种神经符号窍门生成器，该生成器结合了大规模语言模型学到的数学直觉与通过符号方法编码的领域特定洞察。这种集成的关键在于识别哪些部分的数学推理最适合大规模语言模型，哪些最适合符号方法。虽然神经符号集成的基本思想适用于各种数学问题，但在本文中，我们专门集中在奥林匹克不等式（图1）上。我们分析了人类解决这些问题的方式，并提炼出两种类型的窍门：（1）缩放，由符号方法处理；（2）重写，由大规模语言模型处理。此外，我们结合符号工具和大规模语言模型来剪枝和排序证明目标，以实现有效的证明搜索。我们通过多个数学竞赛中的161个具有挑战性的不等式评估了该框架，取得了最先进的性能，并在无需额外训练数据的情况下显著优于现有大规模语言模型和符号方法。 

---
# SPPD: Self-training with Process Preference Learning Using Dynamic Value Margin 

**Title (ZH)**: SPPD：基于动态价值边际的进程偏好学习自训练 

**Authors**: Hao Yi, Qingyang Li, Yulan Hu, Fuzheng Zhang, Di Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13516)  

**Abstract**: Recently, enhancing the numerical and logical reasoning capability of Large Language Models (LLMs) has emerged as a research hotspot. Existing methods face several limitations: inference-phase techniques (e.g., Chain of Thoughts) rely on prompt selection and the pretrained knowledge; sentence-level Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) struggle with step-wise mathematical correctness and depend on stronger models distillation or human annotations; while Reinforcement Learning (RL) approaches incur high GPU memory costs and unstable training. To address these, we propose \textbf{S}elf-training framework integrating \textbf{P}rocess \textbf{P}reference learning using \textbf{D}ynamic value margin (SPPD). SPPD leverages a process-based Markov Decision Process (MDP) and Bellman optimality equation to derive \textbf{dynamic value margin} on step-level preference optimization, which employs tree-based self-sampling on model responses \textbf{without any distillation} from other models. Furthermore, we theoretically prove that SPPD is \textbf{equivalent to on-policy policy gradient methods} under reward constraints. Experiments on 7B-scale models demonstrate superior performance across in-domain and out-domain mathematical benchmarks. We open-source our code at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 最近，增强大型语言模型的数值和逻辑推理能力已成为研究热点。我们提出了一种结合过程偏好学习和动态价值差距的自训练框架（SPPD）。SPPD 利用基于过程的马尔可夫决策过程（MDP）和贝尔曼最优方程来在步骤级偏好优化中得出动态价值差距，该方法通过树状结构自我采样模型响应，而不依赖其他模型的蒸馏。此外，我们理论证明在奖励约束条件下，SPPD 等同于在线策略策略梯度方法。在涵盖领域内外的数学基准测试中，7B规模模型展示了优越的表现。我们已开源代码，网址为：this https URL。 

---
# Reasoning with Reinforced Functional Token Tuning 

**Title (ZH)**: 强化功能标记调谐推理 

**Authors**: Kongcheng Zhang, Qi Yao, Baisheng Lai, Jiaxing Huang, Wenkai Fang, Dacheng Tao, Mingli Song, Shunyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13389)  

**Abstract**: In this work, we propose Reinforced Functional Token Tuning (RFTT), a novel reinforced fine-tuning framework that empowers Large Language Models (LLMs) with self-play learn-to-reason capabilities. Unlike prior prompt-driven reasoning efforts, RFTT embeds a rich set of learnable functional tokens (e.g., <analyze>, <verify>, <refine>) directly into the model vocabulary, enabling chain-of-thought construction with diverse human-like reasoning behaviors. Specifically, RFTT comprises two phases: (1) supervised fine-tuning performs prompt-driven tree search to obtain self-generated training data annotated with functional tokens, which warms up the model to learn these tokens for reasoning; and (2) online reinforcement learning further allows the model to explore different reasoning pathways through functional token sampling without relying on prompts, thereby facilitating effective self-improvement for functional reasoning. Extensive experiments demonstrate the superiority of the proposed RFTT on mathematical benchmarks, significantly boosting Qwen-2.5-7B-Instruct (70.6% to 79.8%) and LLaMA-3.1-8B-Instruct (32.2% to 60.2%) on the MATH dataset. Moreover, the performance of RFTT consistently improves with more search rollouts at inference time. Our code is available at this https URL. 

**Abstract (ZH)**: 基于强化的学习可推理功能标记调优（RFTT）：一种增强的大语言模型推理能力的新框架 

---
# Reflection of Episodes: Learning to Play Game from Expert and Self Experiences 

**Title (ZH)**: 基于专家和自我经验的episode反映：学习玩游戏 

**Authors**: Xiaojie Xu, Zongyuan Li, Chang Lu, Runnan Qi, Yanan Ni, Lumin Jiang, Xiangbei Liu, Xuebo Zhang, Yongchun Fang, Kuihua Huang, Xian Guo, Zhanghua Wu, Zhenya Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13388)  

**Abstract**: StarCraft II is a complex and dynamic real-time strategy (RTS) game environment, which is very suitable for artificial intelligence and reinforcement learning research. To address the problem of Large Language Model(LLM) learning in complex environments through self-reflection, we propose a Reflection of Episodes(ROE) framework based on expert experience and self-experience. This framework first obtains key information in the game through a keyframe selection method, then makes decisions based on expert experience and self-experience. After a game is completed, it reflects on the previous experience to obtain new self-experience. Finally, in the experiment, our method beat the robot under the Very Hard difficulty in TextStarCraft II. We analyze the data of the LLM in the process of the game in detail, verified its effectiveness. 

**Abstract (ZH)**: StarCraft II是一个复杂多变的即时战略(RTS)游戏环境，非常适合人工智能和强化学习研究。为了解决大规模语言模型（LLM）在复杂环境中的学习问题，我们提出了一种基于专家经验和自我经验的episode反思（ROE）框架。该框架首先通过关键帧选择方法获取游戏中的关键信息，然后基于专家经验和自我经验进行决策。在一局游戏结束后，对其进行反思以获得新的自我经验。在实验中，我们的方法在TextStarCraft II的极难模式下战胜了机器人。我们详细分析了游戏过程中大规模语言模型的数据，验证了其有效性。 

---
# Revisiting Privacy, Utility, and Efficiency Trade-offs when Fine-Tuning Large Language Models 

**Title (ZH)**: 重访微调大型语言模型时隐私、实用性和效率的权衡 

**Authors**: Soumi Das, Camila Kolling, Mohammad Aflah Khan, Mahsa Amani, Bishwamittra Ghosh, Qinyuan Wu, Till Speicher, Krishna P. Gummadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13313)  

**Abstract**: We study the inherent trade-offs in minimizing privacy risks and maximizing utility, while maintaining high computational efficiency, when fine-tuning large language models (LLMs). A number of recent works in privacy research have attempted to mitigate privacy risks posed by memorizing fine-tuning data by using differentially private training methods (e.g., DP), albeit at a significantly higher computational cost (inefficiency). In parallel, several works in systems research have focussed on developing (parameter) efficient fine-tuning methods (e.g., LoRA), but few works, if any, investigated whether such efficient methods enhance or diminish privacy risks. In this paper, we investigate this gap and arrive at a surprising conclusion: efficient fine-tuning methods like LoRA mitigate privacy risks similar to private fine-tuning methods like DP. Our empirical finding directly contradicts prevailing wisdom that privacy and efficiency objectives are at odds during fine-tuning. Our finding is established by (a) carefully defining measures of privacy and utility that distinguish between memorizing sensitive and non-sensitive tokens in training and test datasets used in fine-tuning and (b) extensive evaluations using multiple open-source language models from Pythia, Gemma, and Llama families and different domain-specific datasets. 

**Abstract (ZH)**: 我们研究在优化大型语言模型（LLM）微调时减少隐私风险和最大化实用性之间固有的权衡，同时保持高度的计算效率。尽管使用差分隐私训练方法（如DP）可以缓解由于记忆微调数据带来的隐私风险，但这显著增加了计算成本（低效率）。同时，系统研究中的一些工作聚焦于开发高效的微调方法（如LoRA），但很少有研究探讨这些高效的微调方法是增强还是减少了隐私风险。在本文中，我们探讨了这一差距，并得出一个令人惊讶的结论：类似于差分隐私微调方法（如DP），高效的微调方法（如LoRA）也减轻了隐私风险。我们的实证发现直接反驳了微调期间隐私和效率目标相冲突的普遍认识。这一发现通过（a）仔细定义区分训练和测试数据集中敏感和非敏感标记的隐私和实用性度量，以及（b）使用来自Pythia、Gemma和Llama家族的多个开源语言模型和不同领域特定数据集进行广泛的评估而得以确立。 

---
# Demonstrating specification gaming in reasoning models 

**Title (ZH)**: 证明推理模型中的规范游戏行为 

**Authors**: Alexander Bondarenko, Denis Volk, Dmitrii Volkov, Jeffrey Ladish  

**Link**: [PDF](https://arxiv.org/pdf/2502.13295)  

**Abstract**: We demonstrate LLM agent specification gaming by instructing models to win against a chess engine. We find reasoning models like o1 preview and DeepSeek-R1 will often hack the benchmark by default, while language models like GPT-4o and Claude 3.5 Sonnet need to be told that normal play won't work to hack.
We improve upon prior work like (Hubinger et al., 2024; Meinke et al., 2024; Weij et al., 2024) by using realistic task prompts and avoiding excess nudging. Our results suggest reasoning models may resort to hacking to solve difficult problems, as observed in OpenAI (2024)'s o1 Docker escape during cyber capabilities testing. 

**Abstract (ZH)**: 我们通过指示模型战胜国际象棋引擎来展示大规模语言模型代理规范游戏。我们发现，如o1 preview和DeepSeek-R1这类推理模型通常默认会作弊，而如GPT-4o和Claude 3.5 Sonnet这类语言模型需要明确被告知正常玩法无效才能作弊。我们改进了前人的工作（Hubinger et al., 2024；Meinke et al., 2024；Weij et al., 2024），通过使用现实的任务提示并避免过度引导。我们的结果表明，推理模型可能会为了解决问题而采取作弊行为，类似于OpenAI (2024)在网络安全能力测试中o1 Docker逃脱时的作弊行为。 

---
# Unveiling the Magic of Code Reasoning through Hypothesis Decomposition and Amendment 

**Title (ZH)**: 通过假设分解与修正揭示代码推理的魔力 

**Authors**: Yuze Zhao, Tianyun Ji, Wenjun Feng, Zhenya Huang, Qi Liu, Zhiding Liu, Yixiao Ma, Kai Zhang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13170)  

**Abstract**: The reasoning abilities are one of the most enigmatic and captivating aspects of large language models (LLMs). Numerous studies are dedicated to exploring and expanding the boundaries of this reasoning capability. However, tasks that embody both reasoning and recall characteristics are often overlooked. In this paper, we introduce such a novel task, code reasoning, to provide a new perspective for the reasoning abilities of LLMs. We summarize three meta-benchmarks based on established forms of logical reasoning, and instantiate these into eight specific benchmark tasks. Our testing on these benchmarks reveals that LLMs continue to struggle with identifying satisfactory reasoning pathways. Additionally, we present a new pathway exploration pipeline inspired by human intricate problem-solving methods. This Reflective Hypothesis Decomposition and Amendment (RHDA) pipeline consists of the following iterative steps: (1) Proposing potential hypotheses based on observations and decomposing them; (2) Utilizing tools to validate hypotheses and reflection outcomes; (3) Revising hypothesis in light of observations. Our approach effectively mitigates logical chain collapses arising from forgetting or hallucination issues in multi-step reasoning, resulting in performance gains of up to $3\times$. Finally, we expanded this pipeline by applying it to simulate complex household tasks in real-world scenarios, specifically in VirtualHome, enhancing the handling of failure cases. We release our code and all of results at this https URL. 

**Abstract (ZH)**: 大型语言模型的推理能力是迄今最神秘和迷人的方面之一。许多研究致力于探索和扩展这一推理能力的边界。然而，同时包含推理和回忆特征的任务往往被忽视。在本文中，我们引入了一种新的任务——代码推理，以提供大型语言模型推理能力的新视角。我们基于已建立的逻辑推理形式总结了三个元基准，并将这些形式具体化为八个特定的基准任务。在这些基准上的测试表明，大型语言模型仍然难以识别满意的推理路径。此外，我们提出了一种新的路径探索管道，受到人类复杂问题解决方法的启发。该Reflective Hypothesis Decomposition and Amendment (RHDA)管道包含以下迭代步骤：（1）基于观察提出潜在假设并分解它们；（2）使用工具验证假设和反思结果；（3）根据观察结果修订假设。我们的方法有效地缓解了多步推理中由于忘记或幻觉问题导致的逻辑链条断裂问题，从而获得了高达$3\times$的性能提升。最后，我们通过将该管道应用于模拟现实世界中的复杂家务任务（特别是在VirtualHome中）来扩展了这个管道，增强了对失败案例的处理能力。我们在此网址发布我们的代码和所有结果：https://。 

---
# Autellix: An Efficient Serving Engine for LLM Agents as General Programs 

**Title (ZH)**: Autellix: 一种高效的LLM代理通用程序服务引擎 

**Authors**: Michael Luo, Xiaoxiang Shi, Colin Cai, Tianjun Zhang, Justin Wong, Yichuan Wang, Chi Wang, Yanping Huang, Zhifeng Chen, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.13965)  

**Abstract**: Large language model (LLM) applications are evolving beyond simple chatbots into dynamic, general-purpose agentic programs, which scale LLM calls and output tokens to help AI agents reason, explore, and solve complex tasks. However, existing LLM serving systems ignore dependencies between programs and calls, missing significant opportunities for optimization. Our analysis reveals that programs submitted to LLM serving engines experience long cumulative wait times, primarily due to head-of-line blocking at both the individual LLM request and the program. To address this, we introduce Autellix, an LLM serving system that treats programs as first-class citizens to minimize their end-to-end latencies. Autellix intercepts LLM calls submitted by programs, enriching schedulers with program-level context. We propose two scheduling algorithms-for single-threaded and distributed programs-that preempt and prioritize LLM calls based on their programs' previously completed calls. Our evaluation demonstrates that across diverse LLMs and agentic workloads, Autellix improves throughput of programs by 4-15x at the same latency compared to state-of-the-art systems, such as vLLM. 

**Abstract (ZH)**: 大型语言模型（LLM）应用正在从简单的聊天机器人演变成为动态的通用代理程序，这些程序扩展了LLM的调用和输出令牌以帮助AI代理进行推理、探索和解决复杂任务。然而，现有的LLM服务系统忽略了程序之间及其调用之间的依赖关系，错过了优化的重要机会。我们的分析表明，提交给LLM服务引擎的程序经历了长时间的累积等待时间，主要是由于个体LLM请求和程序级别的头部阻塞。为解决这一问题，我们引入了Autellix，这是一种将程序视为一等公民的LLM服务系统，以最小化其端到端延迟。Autellix拦截程序提交的LLM调用，通过为调度器提供程序级别的上下文来增强调度。我们提出了两种调度算法，分别针对单线程和分布式程序，根据程序之前完成的调用来预取和优先级排序LLM调用。我们的评估显示，与最先进的系统（如vLLM）相比，在相同的延迟条件下，Autellix将程序的吞吐量提高了4-15倍。 

---
# RAG-Gym: Optimizing Reasoning and Search Agents with Process Supervision 

**Title (ZH)**: RAG-Gym: 通过过程监督优化推理和搜索智能体 

**Authors**: Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang, Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing Song, Dengyu Wang, Minjia Zhang, Zhiyong Lu, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13957)  

**Abstract**: Retrieval-augmented generation (RAG) has shown great potential for knowledge-intensive tasks, but its traditional architectures rely on static retrieval, limiting their effectiveness for complex questions that require sequential information-seeking. While agentic reasoning and search offer a more adaptive approach, most existing methods depend heavily on prompt engineering. In this work, we introduce RAG-Gym, a unified optimization framework that enhances information-seeking agents through fine-grained process supervision at each search step. We also propose ReSearch, a novel agent architecture that synergizes answer reasoning and search query generation within the RAG-Gym framework. Experiments on four challenging datasets show that RAG-Gym improves performance by up to 25.6\% across various agent architectures, with ReSearch consistently outperforming existing baselines. Further analysis highlights the effectiveness of advanced LLMs as process reward judges and the transferability of trained reward models as verifiers for different LLMs. Additionally, we examine the scaling properties of training and inference in agentic RAG. The project homepage is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）在知识密集型任务中显示出巨大潜力，但其传统的架构依赖静态检索，限制了其对需要顺序信息查找的复杂问题的有效性。尽管代理式推理和搜索提供了更加适应的解决方案，但大多数现有方法仍然高度依赖于提示工程。在本工作中，我们引入了RAG-Gym，这是一种统一的优化框架，通过在每次搜索步骤中提供精细的过程监督来增强信息查找代理。我们还提出了一种名为ReSearch的新一代代理架构，在RAG-Gym框架中结合了答案推理和搜索查询生成。在四个具有挑战性的数据集上的实验表明，RAG-Gym在各种代理架构上将性能提高了最多25.6%，而ReSearch持续优于现有基线。进一步的分析表明高级LLM作为过程奖励裁判的有效性和训练的奖励模型对不同LLM作为验证者的可迁移性。此外，我们还研究了代理式RAG的训练和推理的扩展性。项目主页可通过此链接访问：https://xxxxxxx。 

---
# Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region 

**Title (ZH)**: 为什么保护性船舶会搁浅？对齐的大型语言模型的安全机制倾向于集中在模板区域。 

**Authors**: Chak Tou Leong, Qingyu Yin, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13946)  

**Abstract**: The safety alignment of large language models (LLMs) remains vulnerable, as their initial behavior can be easily jailbroken by even relatively simple attacks. Since infilling a fixed template between the input instruction and initial model output is a common practice for existing LLMs, we hypothesize that this template is a key factor behind their vulnerabilities: LLMs' safety-related decision-making overly relies on the aggregated information from the template region, which largely influences these models' safety behavior. We refer to this issue as template-anchored safety alignment. In this paper, we conduct extensive experiments and verify that template-anchored safety alignment is widespread across various aligned LLMs. Our mechanistic analyses demonstrate how it leads to models' susceptibility when encountering inference-time jailbreak attacks. Furthermore, we show that detaching safety mechanisms from the template region is promising in mitigating vulnerabilities to jailbreak attacks. We encourage future research to develop more robust safety alignment techniques that reduce reliance on the template region. 

**Abstract (ZH)**: 大型语言模型的安全对齐仍存在脆弱性，因为它们的初始行为可以通过相对简单的攻击轻易被破解。由于现有大型语言模型在输入指令与初始模型输出之间填充固定模板是一种常见做法，我们假设该模板是其脆弱性的关键因素：大型语言模型的安全相关决策过度依赖模板区域的综合信息，这大大影响了这些模型的安全行为。我们将这一问题称为模板锚定的安全对齐。在本文中，我们进行了广泛的实验，并验证了模板锚定的安全对齐在各种对齐的大型语言模型中普遍存在。我们的机制分析揭示了它如何导致模型在遇到推理时的破解攻击中变得易受攻击。此外，我们证明了将安全机制与模板区域分离是减轻破解攻击脆弱性的有希望的方法。我们鼓励未来的研究开发减少对模板区域依赖的更稳健的安全对齐技术。 

---
# How Do LLMs Perform Two-Hop Reasoning in Context? 

**Title (ZH)**: LLMs在上下文中进行两_hop推理的表现如何？ 

**Authors**: Tianyu Guo, Hanlin Zhu, Ruiqi Zhang, Jiantao Jiao, Song Mei, Michael I. Jordan, Stuart Russell  

**Link**: [PDF](https://arxiv.org/pdf/2502.13913)  

**Abstract**: "Socrates is human. All humans are mortal. Therefore, Socrates is mortal." This classical example demonstrates two-hop reasoning, where a conclusion logically follows from two connected premises. While transformer-based Large Language Models (LLMs) can make two-hop reasoning, they tend to collapse to random guessing when faced with distracting premises. To understand the underlying mechanism, we train a three-layer transformer on synthetic two-hop reasoning tasks. The training dynamics show two stages: a slow learning phase, where the 3-layer transformer performs random guessing like LLMs, followed by an abrupt phase transitions, where the 3-layer transformer suddenly reaches $100%$ accuracy. Through reverse engineering, we explain the inner mechanisms for how models learn to randomly guess between distractions initially, and how they learn to ignore distractions eventually. We further propose a three-parameter model that supports the causal claims for the mechanisms to the training dynamics of the transformer. Finally, experiments on LLMs suggest that the discovered mechanisms generalize across scales. Our methodologies provide new perspectives for scientific understandings of LLMs and our findings provide new insights into how reasoning emerges during training. 

**Abstract (ZH)**: 苏格拉底是人。所有人都是 Mortal。因此，苏格拉底是 Mortal。这个经典的例子展示了两跳推理，其中结论从两个相连的前提中逻辑地得出。虽然基于Transformer的大语言模型（LLMs）可以进行两跳推理，但面对分散注意力的前提时，它们往往会退化为随机猜测。为了理解其内部机制，我们对三层Transformer进行了训练，以完成合成的两跳推理任务。训练动态显示了两个阶段：一个缓慢的学习阶段，在此期间三层Transformer像LLMs一样进行随机猜测，随后是一个突变的相变阶段，在此阶段三层Transformer突然达到100%的准确率。通过逆向工程，我们解释了模型如何在初始阶段随机猜测干扰信息以学习推理，以及如何最终学会忽略干扰信息。我们进一步提出了一个三参数模型，该模型支持对变压器训练动态中机制因果关系的说明。最后，对LLMs的实验表明，发现的机制在不同规模上具有泛化能力。我们的方法论为大语言模型的科学理解提供了新的视角，我们的发现为推理在训练期间如何产生提供了新的见解。 

---
# Lost in Sequence: Do Large Language Models Understand Sequential Recommendation? 

**Title (ZH)**: 迷失在序列中：大型语言模型理解序列推荐吗？ 

**Authors**: Sein Kim, Hongseok Kang, Kibum Kim, Jiwan Kim, Donghyun Kim, Minchul Yang, Kwangjin Oh, Julian McAuley, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2502.13909)  

**Abstract**: Large Language Models (LLMs) have recently emerged as promising tools for recommendation thanks to their advanced textual understanding ability and context-awareness. Despite the current practice of training and evaluating LLM-based recommendation (LLM4Rec) models under a sequential recommendation scenario, we found that whether these models understand the sequential information inherent in users' item interaction sequences has been largely overlooked. In this paper, we first demonstrate through a series of experiments that existing LLM4Rec models do not fully capture sequential information both during training and inference. Then, we propose a simple yet effective LLM-based sequential recommender, called LLM-SRec, a method that enhances the integration of sequential information into LLMs by distilling the user representations extracted from a pre-trained CF-SRec model into LLMs. Our extensive experiments show that LLM-SRec enhances LLMs' ability to understand users' item interaction sequences, ultimately leading to improved recommendation performance. Furthermore, unlike existing LLM4Rec models that require fine-tuning of LLMs, LLM-SRec achieves state-of-the-art performance by training only a few lightweight MLPs, highlighting its practicality in real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型(LLMs)近年来由于其先进的文本理解和上下文意识能力，已成为推荐领域的有前途的工具。尽管目前的实践是在序列推荐场景下训练和评估基于LLM的推荐(LLM4Rec)模型，我们发现这些模型是否能够充分理解用户项交互序列中固有的序列信息已经被很大程度地忽视。在本文中，我们首先通过一系列实验展示了现有的LLM4Rec模型在训练和推理过程中并未充分捕捉序列信息。然后，我们提出了一种简单而有效的基于LLM的序列推荐器，称为LLM-SRec，该方法通过从预训练的CF-SRec模型中提取用户表示并将其 distilled 到LLM中，增强序列信息在LLM中的整合。我们的实验表明，LLM-SRec提高了LLM理解用户项交互序列的能力，最终提升了推荐性能。此外，与现有的LLM4Rec模型需要对LLM进行微调不同，LLM-SRec仅通过训练少数轻量级的MLP即可实现最先进的性能，突显了其在实际应用中的实用性。代码可从此链接获取。 

---
# DataSciBench: An LLM Agent Benchmark for Data Science 

**Title (ZH)**: DataSciBench: 一个数据科学LLM代理基准测试 

**Authors**: Dan Zhang, Sining Zhoubian, Min Cai, Fengzu Li, Lekang Yang, Wei Wang, Tianjiao Dong, Ziniu Hu, Jie Tang, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.13897)  

**Abstract**: This paper presents DataSciBench, a comprehensive benchmark for evaluating Large Language Model (LLM) capabilities in data science. Recent related benchmarks have primarily focused on single tasks, easily obtainable ground truth, and straightforward evaluation metrics, which limits the scope of tasks that can be evaluated. In contrast, DataSciBench is constructed based on a more comprehensive and curated collection of natural and challenging prompts for uncertain ground truth and evaluation metrics. We develop a semi-automated pipeline for generating ground truth (GT) and validating evaluation metrics. This pipeline utilizes and implements an LLM-based self-consistency and human verification strategy to produce accurate GT by leveraging collected prompts, predefined task types, and aggregate functions (metrics). Furthermore, we propose an innovative Task - Function - Code (TFC) framework to assess each code execution outcome based on precisely defined metrics and programmatic rules. Our experimental framework involves testing 6 API-based models, 8 open-source general models, and 9 open-source code generation models using the diverse set of prompts we have gathered. This approach aims to provide a more comprehensive and rigorous evaluation of LLMs in data science, revealing their strengths and weaknesses. Experimental results demonstrate that API-based models outperform open-sourced models on all metrics and Deepseek-Coder-33B-Instruct achieves the highest score among open-sourced models. We release all code and data at this https URL. 

**Abstract (ZH)**: DataSciBench：一种评估大型语言模型数据科学能力的综合性基准 

---
# SPEX: Scaling Feature Interaction Explanations for LLMs 

**Title (ZH)**: SPEX: 扩展大规模语言模型功能交互解释 

**Authors**: Justin Singh Kang, Landon Butler, Abhineet Agarwal, Yigit Efe Erginbas, Ramtin Pedarsani, Kannan Ramchandran, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13870)  

**Abstract**: Large language models (LLMs) have revolutionized machine learning due to their ability to capture complex interactions between input features. Popular post-hoc explanation methods like SHAP provide marginal feature attributions, while their extensions to interaction importances only scale to small input lengths ($\approx 20$). We propose Spectral Explainer (SPEX), a model-agnostic interaction attribution algorithm that efficiently scales to large input lengths ($\approx 1000)$. SPEX exploits underlying natural sparsity among interactions -- common in real-world data -- and applies a sparse Fourier transform using a channel decoding algorithm to efficiently identify important interactions. We perform experiments across three difficult long-context datasets that require LLMs to utilize interactions between inputs to complete the task. For large inputs, SPEX outperforms marginal attribution methods by up to 20% in terms of faithfully reconstructing LLM outputs. Further, SPEX successfully identifies key features and interactions that strongly influence model output. For one of our datasets, HotpotQA, SPEX provides interactions that align with human annotations. Finally, we use our model-agnostic approach to generate explanations to demonstrate abstract reasoning in closed-source LLMs (GPT-4o mini) and compositional reasoning in vision-language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过捕获输入特征间的复杂交互，已经革新了机器学习。流行的事后解释方法如SHAP提供边缘特征归属，而它们的扩展方法仅适用于短输入长度（约20个）。我们提出了频谱解释器（SPEX），这是一种通用的交互归属算法，能够高效地扩展到长输入长度（约1000个）。SPEX 利用了交互中的潜在自然稀疏性——在现实世界数据中常见，并借助信道解码算法应用稀疏傅里叶变换，以高效地识别重要交互。我们在三个需要LLMs利用输入间交互才能完成任务的困难长上下文数据集上进行了实验。对于长输入，SPEX 在忠实重建LLM输出方面比边缘归属方法高出了最多20%。此外，SPEX 能够识别对模型输出有重要影响的关键特征和交互。对于我们的一个数据集HotpotQA，SPEX 提供的交互与人类注释一致。最后，我们利用通用方法为封闭源代码LLM（GPT-4o mini）生成解释，证明了其抽象推理能力，并展示了视觉-语言模型的组合推理能力。 

---
# Enhancing LLM-Based Recommendations Through Personalized Reasoning 

**Title (ZH)**: 通过个性化推理增强基于LLM的推荐 

**Authors**: Jiahao Liu, Xueshuo Yan, Dongsheng Li, Guangping Zhang, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13845)  

**Abstract**: Current recommendation systems powered by large language models (LLMs) often underutilize their reasoning capabilities due to a lack of explicit logical structuring. To address this limitation, we introduce CoT-Rec, a framework that integrates Chain-of-Thought (CoT) reasoning into LLM-driven recommendations by incorporating two crucial processes: user preference analysis and item perception evaluation. CoT-Rec operates in two key phases: (1) personalized data extraction, where user preferences and item perceptions are identified, and (2) personalized data application, where this information is leveraged to refine recommendations. Our experimental analysis demonstrates that CoT-Rec improves recommendation accuracy by making better use of LLMs' reasoning potential. The implementation is publicly available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的当前推荐系统往往由于缺乏明确的逻辑结构而未能充分利用其推理能力。为了克服这一局限性，我们引入了CoT-Rec框架，该框架通过整合用户偏好分析和物品感知评估，将Chain-of-Thought（CoT）推理融入到由大型语言模型驱动的推荐中。CoT-Rec分为两个关键阶段：(1) 个性化数据提取，识别用户偏好和物品感知，(2) 个性化数据应用，利用这些信息改进推荐。实验分析表明，CoT-Rec通过更有效地利用大型语言模型的推理潜力提高了推荐精度。该实施已在此处公开：this https URL。 

---
# Enhancing Cross-Domain Recommendations with Memory-Optimized LLM-Based User Agents 

**Title (ZH)**: 基于内存优化的语言模型驱动用户代理以增强跨域推荐 

**Authors**: Jiahao Liu, Shengkang Gu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13843)  

**Abstract**: Large Language Model (LLM)-based user agents have emerged as a powerful tool for improving recommender systems by simulating user interactions. However, existing methods struggle with cross-domain scenarios due to inefficient memory structures, leading to irrelevant information retention and failure to account for social influence factors such as popularity. To address these limitations, we introduce AgentCF++, a novel framework featuring a dual-layer memory architecture and a two-step fusion mechanism to filter domain-specific preferences effectively. Additionally, we propose interest groups with shared memory, allowing the model to capture the impact of popularity trends on users with similar interests. Through extensive experiments on multiple cross-domain datasets, AgentCF++ demonstrates superior performance over baseline models, highlighting its effectiveness in refining user behavior simulation for recommender systems. Our code is available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的用户代理通过模拟用户交互提高了推荐系统的性能，但在跨域场景中由于不高效的内存结构而遇到挑战，导致无关信息的保留和社交影响因素如流行性考虑不周。为解决这些限制，我们提出了AgentCF++这一新型框架，该框架采用双层内存架构和两步融合机制，有效过滤领域特定的偏好。此外，我们还提出了共享内存的兴趣群组，使模型能够捕捉类似兴趣用户受到流行趋势影响的情况。通过在多个跨域数据集上的广泛实验，AgentCF++在基线模型上表现出更优的表现，突显了其在细化推荐系统中用户行为模拟方面的能力。我们的代码可在以下链接获取：这个 https URL。 

---
# LESA: Learnable LLM Layer Scaling-Up 

**Title (ZH)**: LESA: 学习可调整的大语言模型层规模增长 

**Authors**: Yifei Yang, Zouying Cao, Xinbei Ma, Yao Yao, Libo Qin, Zhi Chen, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13794)  

**Abstract**: Training Large Language Models (LLMs) from scratch requires immense computational resources, making it prohibitively expensive. Model scaling-up offers a promising solution by leveraging the parameters of smaller models to create larger ones. However, existing depth scaling-up methods rely on empirical heuristic rules for layer duplication, which result in poorer initialization and slower convergence during continual pre-training. We propose \textbf{LESA}, a novel learnable method for depth scaling-up. By concatenating parameters from each layer and applying Singular Value Decomposition, we uncover latent patterns between layers, suggesting that inter-layer parameters can be learned. LESA uses a neural network to predict the parameters inserted between adjacent layers, enabling better initialization and faster training. Experiments show that LESA outperforms existing baselines, achieving superior performance with less than half the computational cost during continual pre-training. Extensive analyses demonstrate its effectiveness across different model sizes and tasks. 

**Abstract (ZH)**: 一种可学习的深度扩展方法：LESA 

---
# VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare 

**Title (ZH)**: VITAL：一个新的数据集，用于 healthcare 领域多样共识基准测试 

**Authors**: Anudeex Shetty, Amin Beheshti, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2502.13775)  

**Abstract**: Alignment techniques have become central to ensuring that Large Language Models (LLMs) generate outputs consistent with human values. However, existing alignment paradigms often model an averaged or monolithic preference, failing to account for the diversity of perspectives across cultures, demographics, and communities. This limitation is particularly critical in health-related scenarios, where plurality is essential due to the influence of culture, religion, personal values, and conflicting opinions. Despite progress in pluralistic alignment, no prior work has focused on health, likely due to the unavailability of publicly available datasets. To address this gap, we introduce VITAL, a new benchmark dataset comprising 13.1K value-laden situations and 5.4K multiple-choice questions focused on health, designed to assess and benchmark pluralistic alignment methodologies. Through extensive evaluation of eight LLMs of varying sizes, we demonstrate that existing pluralistic alignment techniques fall short in effectively accommodating diverse healthcare beliefs, underscoring the need for tailored AI alignment in specific domains. This work highlights the limitations of current approaches and lays the groundwork for developing health-specific alignment solutions. 

**Abstract (ZH)**: 现有的对齐技术已成为确保大型语言模型（LLMs）生成符合人类价值观的输出的核心。然而，现有的对齐范式通常建模平均或统一的偏好，未能考虑到不同文化、人口统计和社区之间的视角多样性。这一局限性在与文化、宗教、个人价值观和不同意见相关的健康场景中尤为重要。尽管在多元对齐方面取得了进展，但此前没有研究关注健康领域，很可能是因为缺乏公开可用的数据集。为填补这一空白，我们引入了VITAL，这是一个新的基准数据集，包含13100个价值导向的情境和5400个多选题，专注于健康领域，旨在评估和基准测试多元对齐方法。通过对八种不同规模的LLM的广泛评估，我们证明现有的多元对齐技术在有效包容多样化的卫生保健信念方面存在不足，突显了在特定领域内为AI对齐量身定制方案的必要性。这项工作突显了当前方法的局限性，并为开发针对健康领域的对齐解决方案奠定了基础。 

---
# AI Software Engineer: Programming with Trust 

**Title (ZH)**: AI软件工程师：编程以信任为基础 

**Authors**: Abhik Roychoudhury, Corina Pasareanu, Michael Pradel, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2502.13767)  

**Abstract**: Large Language Models (LLMs) have shown surprising proficiency in generating code snippets, promising to automate large parts of software engineering via artificial intelligence (AI). We argue that successfully deploying AI software engineers requires a level of trust equal to or even greater than the trust established by human-driven software engineering practices. The recent trend toward LLM agents offers a path toward integrating the power of LLMs to create new code with the power of analysis tools to increase trust in the code. This opinion piece comments on whether LLM agents could dominate software engineering workflows in the future and whether the focus of programming will shift from programming at scale to programming with trust. 

**Abstract (ZH)**: 大型语言模型在生成代码片段方面展示了惊人的能力，有望通过人工智能自动化软件工程的大部分工作。我们argue认为，成功部署AI软件工程师所需的信任程度，不应低于或甚至高于由人类驱动的软件工程实践所建立的信任。最近LLM代理的趋势为结合大型语言模型的生成能力与分析工 具的验证能力以提高代码信任度提供了可能。本文的观点讨论了未来LLM代理是否可能主导软件工程工作流程，以及编程焦点是否会从大规模编程转向基于信任的编程。 

---
# Direct Value Optimization: Improving Chain-of-Thought Reasoning in LLMs with Refined Values 

**Title (ZH)**: 直接价值优化：通过细化价值改进大语言模型的链式推理 

**Authors**: Hongbo Zhang, Han Cui, Guangsheng Bao, Linyi Yang, Jun Wang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13723)  

**Abstract**: We introduce Direct Value Optimization (DVO), an innovative reinforcement learning framework for enhancing large language models in complex reasoning tasks. Unlike traditional methods relying on preference labels, DVO utilizes value signals at individual reasoning steps, optimizing models via a mean squared error loss. The key benefit of DVO lies in its fine-grained supervision, circumventing the need for labor-intensive human annotations. Target values within the DVO are estimated using either Monte Carlo Tree Search or an outcome value model. Our empirical analysis on both mathematical and commonsense reasoning tasks shows that DVO consistently outperforms existing offline preference optimization techniques, even with fewer training steps. These findings underscore the importance of value signals in advancing reasoning capabilities and highlight DVO as a superior methodology under scenarios lacking explicit human preference information. 

**Abstract (ZH)**: 直接价值优化(DVO):一种增强大规模语言模型在复杂推理任务中的创新强化学习框架 

---
# TrustRAG: An Information Assistant with Retrieval Augmented Generation 

**Title (ZH)**: TrustRAG: 一种检索增强生成的信息助手 

**Authors**: Yixing Fan, Qiang Yan, Wenshan Wang, Jiafeng Guo, Ruqing Zhang, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13719)  

**Abstract**: \Ac{RAG} has emerged as a crucial technique for enhancing large models with real-time and domain-specific knowledge. While numerous improvements and open-source tools have been proposed to refine the \ac{RAG} framework for accuracy, relatively little attention has been given to improving the trustworthiness of generated results. To address this gap, we introduce TrustRAG, a novel framework that enhances \ac{RAG} from three perspectives: indexing, retrieval, and generation. Specifically, in the indexing stage, we propose a semantic-enhanced chunking strategy that incorporates hierarchical indexing to supplement each chunk with contextual information, ensuring semantic completeness. In the retrieval stage, we introduce a utility-based filtering mechanism to identify high-quality information, supporting answer generation while reducing input length. In the generation stage, we propose fine-grained citation enhancement, which detects opinion-bearing sentences in responses and infers citation relationships at the sentence-level, thereby improving citation accuracy. We open-source the TrustRAG framework and provide a demonstration studio designed for excerpt-based question answering tasks \footnote{this https URL}. Based on these, we aim to help researchers: 1) systematically enhancing the trustworthiness of \ac{RAG} systems and (2) developing their own \ac{RAG} systems with more reliable outputs. 

**Abstract (ZH)**: 基于TrustRAG框架：从检索、检索和生成三个视角提升大模型的可信度 

---
# An LLM-based Agent for Reliable Docker Environment Configuration 

**Title (ZH)**: 基于LLM的可靠Docker环境配置代理 

**Authors**: Ruida Hu, Chao Peng, Xinchen Wang, Cuiyun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13681)  

**Abstract**: Environment configuration is a critical yet time-consuming step in software development, especially when dealing with unfamiliar code repositories. While Large Language Models (LLMs) demonstrate the potential to accomplish software engineering tasks, existing methods for environment configuration often rely on manual efforts or fragile scripts, leading to inefficiencies and unreliable outcomes. We introduce Repo2Run, the first LLM-based agent designed to fully automate environment configuration and generate executable Dockerfiles for arbitrary Python repositories. We address two major challenges: (1) enabling the LLM agent to configure environments within isolated Docker containers, and (2) ensuring the successful configuration process is recorded and accurately transferred to a Dockerfile without error. To achieve this, we propose atomic configuration synthesis, featuring a dual-environment architecture (internal and external environment) with a rollback mechanism to prevent environment "pollution" from failed commands, guaranteeing atomic execution (execute fully or not at all) and a Dockerfile generator to transfer successful configuration steps into runnable Dockerfiles. We evaluate Repo2Run~on our proposed benchmark of 420 recent Python repositories with unit tests, where it achieves an 86.0% success rate, outperforming the best baseline by 63.9%. 

**Abstract (ZH)**: Repository2Run: 基于大语言模型的全面自动化环境配置代理 

---
# C2T: A Classifier-Based Tree Construction Method in Speculative Decoding 

**Title (ZH)**: C2T：推测解码中基于分类器的树构建方法 

**Authors**: Feiye Huo, Jianchao Tan, Kefeng Zhang, Xunliang Cai, Shengli Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.13652)  

**Abstract**: The growing scale of Large Language Models (LLMs) has exacerbated inference latency and computational costs. Speculative decoding methods, which aim to mitigate these issues, often face inefficiencies in the construction of token trees and the verification of candidate tokens. Existing strategies, including chain mode, static tree, and dynamic tree approaches, have limitations in accurately preparing candidate token trees for verification. We propose a novel method named C2T that adopts a lightweight classifier to generate and prune token trees dynamically. Our classifier considers additional feature variables beyond the commonly used joint probability to predict the confidence score for each draft token to determine whether it is the candidate token for verification. This method outperforms state-of-the-art (SOTA) methods such as EAGLE-2 on multiple benchmarks, by reducing the total number of candidate tokens by 25% while maintaining or even improving the acceptance length. 

**Abstract (ZH)**: 大规模语言模型（LLMs）规模的不断扩大加剧了推理延迟和计算成本。为缓解这些问题的推测解码方法在构建令牌树和验证候选令牌方面常面临效率低下。现有的策略，包括链模式、静态树和动态树方法，在精确准备待验证的候选令牌树方面存在局限性。我们提出了一种名为C2T的新方法，该方法采用轻量级分类器动态生成和修剪令牌树。我们的分类器除了考虑常用的联合概率外，还考虑其他特征变量来预测每个草稿令牌的信心分数，以确定它是否是待验证的候选令牌。该方法在多个基准上优于现有最先进方法（SOTA），例如EAGLE-2，在减少候选令牌总数25%的同时保持甚至提高了接受长度。 

---
# Concept Layers: Enhancing Interpretability and Intervenability via LLM Conceptualization 

**Title (ZH)**: 概念层：通过LLM概念化增强可解释性和干预性 

**Authors**: Or Raphael Bidusa, Shaul Markovitch  

**Link**: [PDF](https://arxiv.org/pdf/2502.13632)  

**Abstract**: The opaque nature of Large Language Models (LLMs) has led to significant research efforts aimed at enhancing their interpretability, primarily through post-hoc methods. More recent in-hoc approaches, such as Concept Bottleneck Models (CBMs), offer both interpretability and intervenability by incorporating explicit concept representations. However, these methods suffer from key limitations, including reliance on labeled concept datasets and significant architectural modifications that challenges re-integration into existing system pipelines. In this work, we introduce a new methodology for incorporating interpretability and intervenability into an existing model by integrating Concept Layers (CLs) into its architecture. Our approach projects the model's internal vector representations into a conceptual, explainable vector space before reconstructing and feeding them back into the model. Furthermore, we eliminate the need for a human-selected concept set by algorithmically searching an ontology for a set of concepts that can be either task-specific or task-agnostic. We evaluate CLs across multiple tasks, demonstrating that they maintain the original model's performance and agreement while enabling meaningful interventions. Additionally, we present a proof of concept showcasing an intervenability interface, allowing users to adjust model behavior dynamically, such as mitigating biases during inference. 

**Abstract (ZH)**: 大型语言模型的不透明性质导致了旨在增强其可解释性和干预性的显著研究努力，主要通过事后方法进行。近年来，概念瓶颈模型（CBMs）等内置方法不仅提供了可解释性，而且通过引入显式概念表示还提供了干预性，但这些方法存在一些关键局限性，包括依赖于标记的概念数据集以及对现有系统管线造成重大架构修改的挑战。在此项工作中，我们提出了一种新的方法，通过将概念层（CLs）整合到现有模型的架构中，以实现可解释性与干预性的结合。我们的方法将模型的内部向量表示投影到一个概念性的、可解释的向量空间中，然后再重构并反馈回模型中。此外，我们通过算法在本体中搜索概念集，从而消除人工选择概念集的需要，这些概念集可以是任务特定的，也可以是任务无关的。我们跨多个任务评估了CLs，结果显示它们保持了原始模型的性能和一致性，同时还允许有意义的干预。此外，我们展示了概念层的一个概念证明，展示了干预接口，允许用户动态调整模型行为，例如在推断过程中缓解偏见。 

---
# REFIND: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models 

**Title (ZH)**: REFIND: 在大规模语言模型中基于检索的事实幻觉检测 

**Authors**: DongGeon Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13622)  

**Abstract**: Hallucinations in large language model (LLM) outputs severely limit their reliability in knowledge-intensive tasks such as question answering. To address this challenge, we introduce REFIND (Retrieval-augmented Factuality hallucINation Detection), a novel framework that detects hallucinated spans within LLM outputs by directly leveraging retrieved documents. As part of the REFIND, we propose the Context Sensitivity Ratio (CSR), a novel metric that quantifies the sensitivity of LLM outputs to retrieved evidence. This innovative approach enables REFIND to efficiently and accurately detect hallucinations, setting it apart from existing methods. In the evaluation, REFIND demonstrated robustness across nine languages, including low-resource settings, and significantly outperformed baseline models, achieving superior IoU scores in identifying hallucinated spans. This work highlights the effectiveness of quantifying context sensitivity for hallucination detection, thereby paving the way for more reliable and trustworthy LLM applications across diverse languages. 

**Abstract (ZH)**: 检索增强事实幻觉检测：在大型语言模型输出中的事实幻觉检测 

---
# Complex Ontology Matching with Large Language Model Embeddings 

**Title (ZH)**: 大规模语言模型嵌入下的复杂本体匹配 

**Authors**: Guilherme Sousa, Rinaldo Lima, Cassia Trojahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.13619)  

**Abstract**: Ontology, and more broadly, Knowledge Graph Matching is a challenging task in which expressiveness has not been fully addressed. Despite the increasing use of embeddings and language models for this task, approaches for generating expressive correspondences still do not take full advantage of these models, in particular, large language models (LLMs). This paper proposes to integrate LLMs into an approach for generating expressive correspondences based on alignment need and ABox-based relation discovery. The generation of correspondences is performed by matching similar surroundings of instance sub-graphs. The integration of LLMs results in different architectural modifications, including label similarity, sub-graph matching, and entity matching. The performance word embeddings, sentence embeddings, and LLM-based embeddings, was compared. The results demonstrate that integrating LLMs surpasses all other models, enhancing the baseline version of the approach with a 45\% increase in F-measure. 

**Abstract (ZH)**: 本体论和更广泛的知识图谱匹配是一个表达性尚未完全解决的挑战性任务。尽管嵌入和语言模型在该任务中的使用不断增加，但生成表达性的对应关系的方法仍未充分利用这些模型，特别是大规模语言模型（LLMs）。本文提出将LLMs集成到基于对齐需求和ABox关系发现的生成表达性对应关系的方法中。通过匹配实例子图的相似环境来进行对应关系的生成。LLMs的集成导致了不同的架构修改，包括标签相似性、子图匹配和实体匹配。比较了词嵌入、句子嵌入和基于LLM的嵌入的性能。结果表明，集成LLMs超越了所有其他模型，使方法的基础版本在F-值上提高了45%。 

---
# LaVCa: LLM-assisted Visual Cortex Captioning 

**Title (ZH)**: LaVCa: LLM辅助的视觉皮层 captioning 

**Authors**: Takuya Matsuyama, Shinji Nishimoto, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13606)  

**Abstract**: Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations. Please check out our webpage at this https URL 

**Abstract (ZH)**: 理解人类大脑中神经群体（或体素）的性质可以增进我们对人类感知和认知处理能力的理解，并有助于开发类脑计算机模型。使用深度神经网络（DNNs）的最近编码模型已成功预测了体素级活动。然而，由于DNNs的黑箱性质，解释解释体素响应的属性仍然是一个挑战。为此，我们提出了基于大语言模型的视觉皮层图解（LaVCa）方法，该方法使用大语言模型生成与体素选择性反应的图像相关联的自然语言图解。通过应用LaVCa进行图像诱发的大脑活动分析，我们证明LaVCa生成的图解比之前的方法更准确地描述了体素的选择性。此外，LaVCa生成的图解在体素间和体素内层次上更详细地定量捕捉了现有方法的属性。进一步分析LaVCa生成的体素特异性属性揭示了感兴趣区（ROIs）内视觉皮层的微细化功能差异以及同时代表多个不同概念的体素。这些发现通过在整个视觉皮层提供详细的图解，提供了对人类视觉表示的深刻见解，并突显了基于大语言模型方法在理解大脑表示方面的潜力。请访问我们的网页：https://this.url 

---
# Efficient Safety Retrofitting Against Jailbreaking for LLMs 

**Title (ZH)**: 面向 Jailbreaking 的高效 LLM 安全加固方法 

**Authors**: Dario Garcia-Gasulla, Anna Arias-Duart, Adrian Tormos, Daniel Hinjos, Oscar Molina-Sedano, Ashwin Kumar Gururajan, Maria Eugenia Cardello  

**Link**: [PDF](https://arxiv.org/pdf/2502.13603)  

**Abstract**: Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research. 

**Abstract (ZH)**: Direct Preference Optimization (DPO)在减少数据需求和训练成本的同时，评估其在防止Jailbreaking攻击方面的模型安全性效果。引入Egida数据集，涵盖27个不同的安全主题和18种不同的攻击样式，结合合成和人工标签，以提升最新LLM的安全性。此外，还评估了它们在通用任务中性能下降的趋势以及过度拒绝倾向。遵循提议的方法，训练后的模型将攻击成功率降低10%-30%，使用少量训练样本（2,000个样本）和低计算成本（8B模型3美元，72B模型20美元）。安全性对齐的模型可以迁移到未见过的主题和攻击样式，最成功的攻击样式成功率达到约5%。模型尺寸和家族对安全性趋向的可塑性影响显著，强调了预训练选择的重要性。为验证研究发现，作者进行了一项独立的人类偏好一致性评估，并发布了相关的数据集Egida-HSafe。总体而言，本研究展示了使用DPO提升LLM安全性是负担得起且易于获取的，同时指出了其当前的局限性。所有数据集和模型均已发布以实现可再现性和进一步研究。 

---
# MMTEB: Massive Multilingual Text Embedding Benchmark 

**Title (ZH)**: 大规模多语言文本嵌入基准（MMTEB） 

**Authors**: Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzemiński, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Çağatan, Akash Kundu, Martin Bernstorff, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Poświata, Kranthi Kiran GV, Shawon Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Šuppa, Crystina Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Krishnakumar, Anna Maksimova, Silvan Wehrli, Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte Ostendorff, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, Siva Reddy, Niklas Muennighoff  

**Link**: [PDF](https://arxiv.org/pdf/2502.13595)  

**Abstract**: Text embeddings are typically evaluated on a limited set of tasks, which are constrained by language, domain, and task diversity. To address these limitations and provide a more comprehensive evaluation, we introduce the Massive Multilingual Text Embedding Benchmark (MMTEB) - a large-scale, community-driven expansion of MTEB, covering over 500 quality-controlled evaluation tasks across 250+ languages. MMTEB includes a diverse set of challenging, novel tasks such as instruction following, long-document retrieval, and code retrieval, representing the largest multilingual collection of evaluation tasks for embedding models to date. Using this collection, we develop several highly multilingual benchmarks, which we use to evaluate a representative set of models. We find that while large language models (LLMs) with billions of parameters can achieve state-of-the-art performance on certain language subsets and task categories, the best-performing publicly available model is multilingual-e5-large-instruct with only 560 million parameters. To facilitate accessibility and reduce computational cost, we introduce a novel downsampling method based on inter-task correlation, ensuring a diverse selection while preserving relative model rankings. Furthermore, we optimize tasks such as retrieval by sampling hard negatives, creating smaller but effective splits. These optimizations allow us to introduce benchmarks that drastically reduce computational demands. For instance, our newly introduced zero-shot English benchmark maintains a ranking order similar to the full-scale version but at a fraction of the computational cost. 

**Abstract (ZH)**: 大规模多语言文本嵌入基准（MMTEB）：一个多语言评价任务的大型、社区驱动扩展 

---
# Are Large Language Models In-Context Graph Learners? 

**Title (ZH)**: 大型语言模型是图学习者吗？ 

**Authors**: Jintang Li, Ruofan Wu, Yuchang Zhu, Huizhe Zhang, Liang Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable in-context reasoning capabilities across a wide range of tasks, particularly with unstructured inputs such as language or images. However, LLMs struggle to handle structured data, such as graphs, due to their lack of understanding of non-Euclidean structures. As a result, without additional fine-tuning, their performance significantly lags behind that of graph neural networks (GNNs) in graph learning tasks. In this paper, we show that learning on graph data can be conceptualized as a retrieval-augmented generation (RAG) process, where specific instances (e.g., nodes or edges) act as queries, and the graph itself serves as the retrieved context. Building on this insight, we propose a series of RAG frameworks to enhance the in-context learning capabilities of LLMs for graph learning tasks. Comprehensive evaluations demonstrate that our proposed RAG frameworks significantly improve LLM performance on graph-based tasks, particularly in scenarios where a pretrained LLM must be used without modification or accessed via an API. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中展示了卓越的上下文推理能力，特别是在语言或图像等非结构化输入上。然而，LLMs在处理图等结构化数据时表现不佳，因为它们不理解非欧几里得结构。因此，在不需要额外微调的情况下，它们在图学习任务上的表现远逊于图神经网络（GNNs）。在本文中，我们展示了图数据上的学习可以被视为检索增强生成（RAG）过程，其中特定实例（如节点或边）充当查询，而图本身作为检索上下文。基于这一洞察，我们提出了一系列RAG框架，以增强LLMs在图学习任务中的上下文学习能力。全面的评估表明，我们提出的一系列RAG框架显著提高了LLMs在图基任务上的表现，特别是在需要使用未修改的预训练LLM或通过API访问的情况下。 

---
# Democratizing Large Language Model-Based Graph Data Augmentation via Latent Knowledge Graphs 

**Title (ZH)**: 基于潜在知识图谱的大规模语言模型驱动的图数据增强民主化 

**Authors**: Yushi Feng, Tsai Hor Chan, Guosheng Yin, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13555)  

**Abstract**: Data augmentation is necessary for graph representation learning due to the scarcity and noise present in graph data. Most of the existing augmentation methods overlook the context information inherited from the dataset as they rely solely on the graph structure for augmentation. Despite the success of some large language model-based (LLM) graph learning methods, they are mostly white-box which require access to the weights or latent features from the open-access LLMs, making them difficult to be democratized for everyone as existing LLMs are mostly closed-source for commercial considerations. To overcome these limitations, we propose a black-box context-driven graph data augmentation approach, with the guidance of LLMs -- DemoGraph. Leveraging the text prompt as context-related information, we task the LLM with generating knowledge graphs (KGs), which allow us to capture the structural interactions from the text outputs. We then design a dynamic merging schema to stochastically integrate the LLM-generated KGs into the original graph during training. To control the sparsity of the augmented graph, we further devise a granularity-aware prompting strategy and an instruction fine-tuning module, which seamlessly generates text prompts according to different granularity levels of the dataset. Extensive experiments on various graph learning tasks validate the effectiveness of our method over existing graph data augmentation methods. Notably, our approach excels in scenarios involving electronic health records (EHRs), which validates its maximal utilization of contextual knowledge, leading to enhanced predictive performance and interpretability. 

**Abstract (ZH)**: 基于LLM的黑盒上下文驱动图数据增强方法：DemoGraph 

---
# From Sub-Ability Diagnosis to Human-Aligned Generation: Bridging the Gap for Text Length Control via MARKERGEN 

**Title (ZH)**: 从亚能力诊断到与人类一致的生成：通过MARKERGEN缩小文本长度控制的差距 

**Authors**: Peiwen Yuan, Chuyi Tan, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Yueqi Zhang, Jiayi Shi, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.13544)  

**Abstract**: Despite the rapid progress of large language models (LLMs), their length-controllable text generation (LCTG) ability remains below expectations, posing a major limitation for practical applications. Existing methods mainly focus on end-to-end training to reinforce adherence to length constraints. However, the lack of decomposition and targeted enhancement of LCTG sub-abilities restricts further this http URL bridge this gap, we conduct a bottom-up decomposition of LCTG sub-abilities with human patterns as reference and perform a detailed error this http URL this basis, we propose MarkerGen, a simple-yet-effective plug-and-play approach that:(1) mitigates LLM fundamental deficiencies via external tool integration;(2) conducts explicit length modeling with dynamically inserted markers;(3) employs a three-stage generation scheme to better align length constraints while maintaining content this http URL experiments demonstrate that MarkerGen significantly improves LCTG across various settings, exhibiting outstanding effectiveness and generalizability. 

**Abstract (ZH)**: 尽管大规模语言模型（LLM）取得了 rapid progress，其可调控长度的文本生成（LCTG）能力仍然不尽如人意，成为实际应用中的重大限制。现有方法主要集中在端到端训练以强化对长度约束的遵守。然而，LCTG子能力的缺乏分解和目标化增强限制了这一进程。为填补这一空白，我们以人类模式为参考进行了自底向上的LCTG子能力分解，并进行详细的错误分析。基于此，我们提出了一个简单有效的插件式解决方案MarkerGen，该方法通过以下方式工作：（1）通过外部工具集成来缓解LLM的基本缺陷；（2）通过动态插入标记来进行显式长度建模；（3）采用三阶段生成方案以更好地满足长度约束同时保持内容质量。实验表明，MarkerGen在各种场景中显著改善了LCTG，展示了出色的有效性和泛化能力。 

---
# Activation-aware Probe-Query: Effective Key-Value Retrieval for Long-Context LLMs Inference 

**Title (ZH)**: 激活感知探针查询：有效的长上下文LLMs推理中的键值检索 

**Authors**: Qingfa Xiao, Jiachuan Wang, Haoyang Li, Cheng Deng, Jiaqi Tang, Shuangyin Li, Yongqi Zhang, Jun Wang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13542)  

**Abstract**: Recent advances in large language models (LLMs) have showcased exceptional performance in long-context tasks, while facing significant inference efficiency challenges with limited GPU memory. Existing solutions first proposed the sliding-window approach to accumulate a set of historical \textbf{key-value} (KV) pairs for reuse, then further improvements selectively retain its subsets at each step. However, due to the sparse attention distribution across a long context, it is hard to identify and recall relevant KV pairs, as the attention is distracted by massive candidate pairs. Additionally, we found it promising to select representative tokens as probe-Query in each sliding window to effectively represent the entire context, which is an approach overlooked by existing methods. Thus, we propose \textbf{ActQKV}, a training-free, \textbf{Act}ivation-aware approach that dynamically determines probe-\textbf{Q}uery and leverages it to retrieve the relevant \textbf{KV} pairs for inference. Specifically, ActQKV monitors a token-level indicator, Activation Bias, within each context window, enabling the proper construction of probe-Query for retrieval at pre-filling stage. To accurately recall the relevant KV pairs and minimize the irrelevant ones, we design a dynamic KV cut-off mechanism guided by information density across layers at the decoding stage. Experiments on the Long-Bench and $\infty$ Benchmarks demonstrate its state-of-the-art performance with competitive inference quality and resource efficiency. 

**Abstract (ZH)**: Recent Advances in Large Language Models: An Activation-Aware Approach for Efficient Inference in Long-Context Tasks 

---
# Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models 

**Title (ZH)**: 训练小型模型，推理大型模型：大规模语言模型的内存高效LoRA训练 

**Authors**: Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Yang You, Guiming Xie, Xuejian Gong, Kunlong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13533)  

**Abstract**: Large Language Models (LLMs) have significantly advanced natural language processing with exceptional task generalization capabilities. Low-Rank Adaption (LoRA) offers a cost-effective fine-tuning solution, freezing the original model parameters and training only lightweight, low-rank adapter matrices. However, the memory footprint of LoRA is largely dominated by the original model parameters. To mitigate this, we propose LoRAM, a memory-efficient LoRA training scheme founded on the intuition that many neurons in over-parameterized LLMs have low training utility but are essential for inference. LoRAM presents a unique twist: it trains on a pruned (small) model to obtain pruned low-rank matrices, which are then recovered and utilized with the original (large) model for inference. Additionally, minimal-cost continual pre-training, performed by the model publishers in advance, aligns the knowledge discrepancy between pruned and original models. Our extensive experiments demonstrate the efficacy of LoRAM across various pruning strategies and downstream tasks. For a model with 70 billion parameters, LoRAM enables training on a GPU with only 20G HBM, replacing an A100-80G GPU for LoRA training and 15 GPUs for full fine-tuning. Specifically, QLoRAM implemented by structured pruning combined with 4-bit quantization, for LLaMA-3.1-70B (LLaMA-2-70B), reduces the parameter storage cost that dominates the memory usage in low-rank matrix training by 15.81$\times$ (16.95$\times$), while achieving dominant performance gains over both the original LLaMA-3.1-70B (LLaMA-2-70B) and LoRA-trained LLaMA-3.1-8B (LLaMA-2-13B). 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理任务泛化能力方面取得了显著进展。LoRA低秩适应（Low-Rank Adaption）提供了一种经济有效的微调方案，固定原始模型参数并仅训练轻量级低秩适配器矩阵。然而，LoRA的记忆占用主要被原始模型参数所支配。为解决这一问题，我们提出LoRAM，这是一种基于过度参数化LLMs中许多神经元在训练中的低效用但对推理至关重要这一直觉的记忆高效LoRA训练方案。LoRAM的独创之处在于，在精简（小型）模型上进行训练以获得精简的低秩矩阵，然后用原始（大型）模型恢复和利用这些矩阵进行推理。此外，由模型提供商预先进行的低成本持续预训练可使精简模型和原始模型之间知识差距趋于一致。我们的广泛实验表明，LoRAM在各种剪枝策略和下游任务上均表现出色。对于一个具有700亿参数的模型，LoRAM使得在仅20G HBM的GPU上进行训练成为可能，替代了用于LoRA训练的A100-80G GPU和用于全微调的15个GPU。具体来说，通过结构化剪枝结合4位量化实现的QLoRAM，使得LLaMA-3.1-70B（LLaMA-2-70B）的参数存储成本在低秩矩阵训练中的主导部分减少了15.81×（16.95×），同时在性能上显著优于原始的LLaMA-3.1-70B（LLaMA-2-70B）和LoRA训练的LLaMA-3.1-8B（LLaMA-2-13B）。 

---
# Exploiting Prefix-Tree in Structured Output Interfaces for Enhancing Jailbreak Attacking 

**Title (ZH)**: 利用前缀树在结构化输出接口中增强Jailbreak攻击 

**Authors**: Yanzeng Li, Yunfan Xiong, Jialun Zhong, Jinchao Zhang, Jie Zhou, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13527)  

**Abstract**: The rise of Large Language Models (LLMs) has led to significant applications but also introduced serious security threats, particularly from jailbreak attacks that manipulate output generation. These attacks utilize prompt engineering and logit manipulation to steer models toward harmful content, prompting LLM providers to implement filtering and safety alignment strategies. We investigate LLMs' safety mechanisms and their recent applications, revealing a new threat model targeting structured output interfaces, which enable attackers to manipulate the inner logit during LLM generation, requiring only API access permissions. To demonstrate this threat model, we introduce a black-box attack framework called AttackPrefixTree (APT). APT exploits structured output interfaces to dynamically construct attack patterns. By leveraging prefixes of models' safety refusal response and latent harmful outputs, APT effectively bypasses safety measures. Experiments on benchmark datasets indicate that this approach achieves higher attack success rate than existing methods. This work highlights the urgent need for LLM providers to enhance security protocols to address vulnerabilities arising from the interaction between safety patterns and structured outputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起带来了显著的应用，同时也引入了严重的安全威胁，特别是来自越狱攻击的威胁，这些攻击通过操控输出生成来操纵模型。这些攻击利用提示工程和logit操纵来引导模型产生有害内容，促使LLM提供商实施过滤和安全对齐策略。我们调查了LLMs的安全机制及其最近的应用，揭示了一种新的威胁模型，针对结构化输出接口，这种接口允许攻击者在LLM生成过程中操纵内部logit，仅需API访问权限。为了展示这一威胁模型，我们引入了一个名为AttackPrefixTree（APT）的黑盒攻击框架，APT利用结构化输出接口动态构建攻击模式。通过利用模型安全拒绝响应的前缀和潜在有害输出，APT有效地绕过了安全措施。基准数据集上的实验表明，这种方法在攻击成功率方面高于现有方法。这项工作强调了LLM提供商亟需增强安全协议，以应对安全模式和结构化输出交互引发的漏洞。 

---
# PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference 

**Title (ZH)**: PLDR-LLMs 学习一个可泛化的张量运算符，该运算符可在推理时替换其自身的深度神经网络。 

**Authors**: Burc Gokden  

**Link**: [PDF](https://arxiv.org/pdf/2502.13502)  

**Abstract**: We show that Large Language Model from Power Law Decoder Representations (PLDR-LLM) is a foundational model whose deductive outputs are invariant tensors up to a small perturbation. PLDR-LLM learns a singularity condition for the deductive outputs that enable the once-inferred energy-curvature tensor $\mathbf{G}_{LM}$ to replace the deep neural network of power law graph attention (PLGA) generating the deductive outputs at inference. We demonstrate that a cache for $\mathbf{G}_{LM}$ (G-cache) and KV-cache can be implemented in a straightforward manner to improve the inference time. The invariance and generalizable nature of deductive outputs is at a very high fidelity where deductive outputs have same RMSE and determinant values up to 15 decimal places after caching, and zero-shot benchmark scores remain unchanged. Ablation studies show that learned deductive outputs have distinct loss and accuracy characteristics from models pretrained with transferred, randomly initialized or identity tensors as a constant tensor operator and an LLM with scaled-dot product attention (SDPA) is a special case of PLDR-LLM where $\mathbf{G}_{LM}$ is predefined as identity. The observed invariance characteristic introduces a novel asymmetry between training and inference phases with caching. We outline observed common characteristics of the deductive outputs for the learned singularity condition. We provide an implementation of a training and inference framework for PLDR-LLM with KV-cache and G-cache. 

**Abstract (ZH)**: 我们展示了Power律解码表示（PLDR）大型语言模型（LLM）是一个基础模型，其演绎输出在轻微扰动下是不变张量。PLDR-LLM 学习演绎输出的奇异性条件，使得一次推断生成的能曲率张量 $\mathbf{G}_{LM}$ 能够替代基于PLGA的深度神经网络。我们证明了 $\mathbf{G}_{LM}$ 缓存（G-cache）和KV-cache 可以方便地实现以提高推理时间。经缓存后的演绎输出具有极高的保真度，其RMSE和行列式值在小数点后15位相同，零样本基准得分保持不变。消融研究表明，学习到的演绎输出与预制转移、随机初始化或恒等张量的模型相比具有不同的损失和准确率特性。当 $\mathbf{G}_{LM}$ 预定义为恒等时，使用缩放点积注意机制（SDPA）的LLM是PLDR-LLM的特例。观察到的不变特性在训练和推理阶段引入了一种新颖的不对称性。我们列出了学习到的奇异性条件下演绎输出的常见特征，并提供了一个包含KV-cache和G-cache的PLDR-LLM训练和推理框架的实现。 

---
# Hidden Darkness in LLM-Generated Designs: Exploring Dark Patterns in Ecommerce Web Components Generated by LLMs 

**Title (ZH)**: LLM生成设计中的隐性黑暗模式：探索由LLM生成的电商网页组件中的暗模式 

**Authors**: Ziwei Chen, Jiawen Shen, Luna, Kristen Vaccaro  

**Link**: [PDF](https://arxiv.org/pdf/2502.13499)  

**Abstract**: Recent work has highlighted the risks of LLM-generated content for a wide range of harmful behaviors, including incorrect and harmful code. In this work, we extend this by studying whether LLM-generated web design contains dark patterns. This work evaluated designs of ecommerce web components generated by four popular LLMs: Claude, GPT, Gemini, and Llama. We tested 13 commonly used ecommerce components (e.g., search, product reviews) and used them as prompts to generate a total of 312 components across all models. Over one-third of generated components contain at least one dark pattern. The majority of dark pattern strategies involve hiding crucial information, limiting users' actions, and manipulating them into making decisions through a sense of urgency. Dark patterns are also more frequently produced in components that are related to company interests. These findings highlight the need for interventions to prevent dark patterns during front-end code generation with LLMs and emphasize the importance of expanding ethical design education to a broader audience. 

**Abstract (ZH)**: 近期的研究强调了由大规模语言模型生成的内容在一系列有害行为中的风险，包括有害的代码。本研究进一步探讨了由大规模语言模型生成的网页设计是否包含暗模式。本研究评估了由四款流行的大型语言模型（Claude、GPT、Gemini和Llama）生成的电商网页组件的设计，共测试了13种常用的电商组件（例如搜索、产品评价），并使用这些组件作为提示生成了共计312个组件。超过三分之一的生成组件包含至少一个暗模式。大多数暗模式策略涉及隐藏关键信息、限制用户行动，并通过紧迫感促使用户做出决策。与公司利益相关的组件中，暗模式的产生更为频繁。这些发现突显了在使用大规模语言模型进行前端代码生成时需要采取干预措施以防止暗模式的必要性，并强调了扩大伦理设计教育的重要性。 

---
# Towards Geo-Culturally Grounded LLM Generations 

**Title (ZH)**: 面向地理文化根基的语言模型生成 

**Authors**: Piyawat Lertvittayakumjorn, David Kinney, Vinodkumar Prabhakaran, Donald Martin, Sunipa Dev  

**Link**: [PDF](https://arxiv.org/pdf/2502.13497)  

**Abstract**: Generative large language models (LLMs) have been demonstrated to have gaps in diverse, cultural knowledge across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on the ability of LLMs to display familiarity with a diverse range of national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on a series of cultural familiarity benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., the norms, artifacts, and institutions of national cultures), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models, while failing to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional knowledge about a culture and open-ended cultural fluency when it comes to evaluating the cultural familiarity of generative LLMs. 

**Abstract (ZH)**: 生成型大型语言模型在全球多元文化知识方面存在差距。我们调查了检索增强生成和搜索 grounding 技术对大型语言模型展示对多种国家文化熟悉程度能力的影响。具体而言，我们比较了标准大型语言模型、从定制知识库中检索信息增强的大型语言模型（即知识库 grounding）以及从网络搜索中检索信息增强的大型语言模型（即搜索 grounding）在一系列文化熟悉度基准测试中的性能。我们发现，搜索 grounding 显著提高了大型语言模型在测试命题知识（如国家文化的规范、器物和制度）的多项选择基准测试中的性能，而知识库 grounding 的有效性受到知识库覆盖不全和检索器次优的限制。然而，搜索 grounding 也会增加语言模型产生刻板印象判断的风险，而未能在具有足够统计效能的人类评估中改善评估者对文化熟悉度的判断。这些结果突出了在评估生成型大型语言模型的文化熟悉度时命题文化知识与开放式文化流利度之间的区别。 

---
# What are Models Thinking about? Understanding Large Language Model Hallucinations "Psychology" through Model Inner State Analysis 

**Title (ZH)**: Large语言模型幻觉“心理学”——通过模型内部状态分析理解模型思维 

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Jue Hong, Ye Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13490)  

**Abstract**: Large language model (LLM) systems suffer from the models' unstable ability to generate valid and factual content, resulting in hallucination generation. Current hallucination detection methods heavily rely on out-of-model information sources, such as RAG to assist the detection, thus bringing heavy additional latency. Recently, internal states of LLMs' inference have been widely used in numerous research works, such as prompt injection detection, etc. Considering the interpretability of LLM internal states and the fact that they do not require external information sources, we introduce such states into LLM hallucination detection. In this paper, we systematically analyze different internal states' revealing features during inference forward and comprehensively evaluate their ability in hallucination detection. Specifically, we cut the forward process of a large language model into three stages: understanding, query, generation, and extracting the internal state from these stages. By analyzing these states, we provide a deep understanding of why the hallucinated content is generated and what happened in the internal state of the models. Then, we introduce these internal states into hallucination detection and conduct comprehensive experiments to discuss the advantages and limitations. 

**Abstract (ZH)**: 大型语言模型（LLM）系统在生成有效和事实性内容方面表现出不稳定的模型能力，导致产生幻觉。当前的幻觉检测方法 heavily 依赖模型外部的信息源，如RAG来辅助检测，从而带来了额外的延迟。最近，大型语言模型推理过程中的内部状态已在众多研究工作中广泛使用，如提示注入检测等。鉴于大型语言模型内部状态的可解释性及其不需要外部信息源的特点，我们将这些状态引入到幻觉检测中。在本文中，我们系统地分析了推理过程中不同内部状态的揭示特征，并全面评估了它们在幻觉检测中的能力。具体地，我们将大型语言模型的前向过程划分为理解、查询、生成三个阶段，并从这些阶段中提取内部状态。通过对这些状态的分析，我们提供了对幻觉内容生成原因及其模型内部状态变化的深入理解。随后，我们将这些内部状态引入幻觉检测，并进行综合实验以讨论其优势和局限性。 

---
# LLM should think and action as a human 

**Title (ZH)**: LLM 应当思考和行动如人类一般。 

**Authors**: Haun Leung, ZiNan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13475)  

**Abstract**: It is popular lately to train large language models to be used as chat assistants, but in the conversation between the user and the chat assistant, there are prompts, require multi-turns between the chat assistant and the user. However, there are a number of issues with the multi-turns conversation: The response of the chat assistant is prone to errors and cannot help users achieve their goals; It is difficult for chat assistant to generate responses with different processes based on actual needs for the same command or request; Chat assistant require the use of tools, but the current approach is not elegant and efficient, and the number of tool calls that can be supported is limited. The main reason for these issues is that large language models do not have the thinking ability as a human, lack the reasoning ability and planning ability, and lack the ability to execute plans. To solve these issues, we propose a thinking method based on a built-in chain of thought: In the multi-turns conversation, for each user prompt, the large language model thinks based on elements such as chat history, thinking context, action calls, memory and knowledge, makes detailed reasoning and planning, and actions according to the plan. We also explored how the large language model enhances thinking ability through this thinking method: Collect training datasets according to the thinking method and fine tune the large language model through supervised learning; Train a consistency reward model and use it as a reward function to fine tune the large language model using reinforcement learning, and the reinforced large language model outputs according to this way of thinking. Our experimental results show that the reasoning ability and planning ability of the large language model are enhanced, and the issues in the multi-turns conversation are solved. 

**Abstract (ZH)**: 近年来，训练大型语言模型作为聊天助手变得流行，但在用户与聊天助手的对话中，聊天助手需要进行多轮交互。然而，多轮对话存在许多问题：聊天助手的回答容易出错，无法帮助用户达成目标；对于相同的命令或请求，聊天助手难以生成基于实际需求的不同步骤的响应；聊天助手需要使用工具，但当前的方法不够优雅高效，支持的工具调用数量有限。这些问题的主要原因是大型语言模型缺乏人类的思考能力，缺乏推理和规划能力，也缺乏执行计划的能力。为了解决这些问题，我们提出了一种基于内置推理链的方法：在多轮对话中，对于每个用户提示，大型语言模型基于聊天历史、推理上下文、操作调用、记忆和知识进行思考，进行详细的推理和规划，并根据计划执行动作。我们还探讨了这种思考方法如何增强大型语言模型的思考能力：根据这种思考方法收集训练数据集，并通过监督学习微调大型语言模型；训练一致性奖励模型，并使用它作为奖励函数，通过强化学习微调大型语言模型，增强后的大型语言模型根据这种方式输出。实验结果表明，大型语言模型的推理能力和规划能力得到了增强，多轮对话中的问题得到了解决。 

---
# Estimating Commonsense Plausibility through Semantic Shifts 

**Title (ZH)**: 通过语义转变估计常识合理性 

**Authors**: Wanqing Cui, Keping Bi, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.13464)  

**Abstract**: Commonsense plausibility estimation is critical for evaluating language models (LMs), yet existing generative approaches--reliant on likelihoods or verbalized judgments--struggle with fine-grained discrimination. In this paper, we propose ComPaSS, a novel discriminative framework that quantifies commonsense plausibility by measuring semantic shifts when augmenting sentences with commonsense-related information. Plausible augmentations induce minimal shifts in semantics, while implausible ones result in substantial deviations. Evaluations on two types of fine-grained commonsense plausibility estimation tasks across different backbones, including LLMs and vision-language models (VLMs), show that ComPaSS consistently outperforms baselines. It demonstrates the advantage of discriminative approaches over generative methods in fine-grained commonsense plausibility evaluation. Experiments also show that (1) VLMs yield superior performance to LMs, when integrated with ComPaSS, on vision-grounded commonsense tasks. (2) contrastive pre-training sharpens backbone models' ability to capture semantic nuances, thereby further enhancing ComPaSS. 

**Abstract (ZH)**: 常识合理性估计对于评估语言模型至关重要，而现有的生成方法依赖于似然性或口头判断，在细微差别区分方面表现出困难。在这篇论文中，我们提出了ComPaSS，一种新颖的辨别框架，通过衡量添加与常识相关的信息时语义的变化来定量衡量常识合理性。合理的增强只会引起最小的语义变化，而不合理的增强会导致显著的偏差。对包括大型语言模型（LLMs）和视觉语言模型（VLMs）在内的不同架构的两种类型的细微常识合理性的评估任务进行了评估，结果显示ComPaSS始终优于基线方法。它展示了辨别方法在细微常识合理性评估中的优势，超越了生成方法。实验还表明，（1）当与ComPaSS结合使用时，VLMs在视觉 grounding 的常识任务中比LMs表现出更优的性能。（2）对比预训练进一步增强了基础模型捕捉语义细微差别的能力，从而进一步增强了ComPaSS。 

---
# ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails 

**Title (ZH)**: 思辨护卫: �审慎的慢思考建立谨慎的 guardrails 

**Authors**: Xiaofei Wen, Wenxuan Zhou, Wenjie Jacky Mo, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13458)  

**Abstract**: Ensuring the safety of large language models (LLMs) is critical as they are deployed in real-world applications. Existing guardrails rely on rule-based filtering or single-pass classification, limiting their ability to handle nuanced safety violations. To address this, we propose ThinkGuard, a critique-augmented guardrail model that distills knowledge from high-capacity LLMs by generating structured critiques alongside safety labels. Fine-tuned on critique-augmented data, the captured deliberative thinking ability drastically enhances the guardrail's cautiousness and interpretability. Evaluated on multiple safety benchmarks, ThinkGuard achieves the highest average F1 and AUPRC, outperforming all baselines. Compared to LLaMA Guard 3, ThinkGuard improves accuracy by 16.1% and macro F1 by 27.0%. Moreover, it surpasses label-only fine-tuned models, confirming that structured critiques enhance both classification precision and nuanced safety reasoning while maintaining computational efficiency. 

**Abstract (ZH)**: 确保大型语言模型的安全性是关键，因为它们已应用于实际应用场景。现有的防护措施依赖于基于规则的过滤或单次分类，这限制了它们处理复杂安全违规的能力。为了解决这一问题，我们提出了一种名为ThinkGuard的批判增强型防护模型，该模型通过生成结构化批判性反馈并结合安全标签来提炼高容量语言模型的知识。通过对增强批判性反馈的数据进行微调，捕获的审慎思考能力极大地提升了防护措施的谨慎性和可解释性。在多个安全性基准测试中，ThinkGuard达到了最高的平均F1和AUPRC，超越了所有基线模型。与LLaMA Guard 3相比，ThinkGuard的准确性提高了16.1%，宏观F1提高了27.0%。此外，ThinkGuard超越了仅依赖标签微调的模型，确认结构化批判性反馈能够同时提升分类精度和复杂安全性推理能力，同时保持计算效率。 

---
# TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation 

**Title (ZH)**: TreeCut: 一个合成的无法回答的数学文字题数据集，用于评估LLM的幻觉能力 

**Authors**: Jialin Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13442)  

**Abstract**: Large language models (LLMs) now achieve near-human performance on standard math word problem benchmarks (e.g., GSM8K), yet their true reasoning ability remains disputed. A key concern is that models often produce confident, yet unfounded, answers to unanswerable problems. We introduce TreeCut, a synthetic dataset that systematically generates infinite unanswerable math word problems and their answerable counterparts, by representing each question as a tree and removing chosen necessary conditions. Experiments show TreeCut effectively induce hallucinations in large language models, including GPT-4o and o3-mini, with rates of 61% and 42% in their respective worst-case scenarios. Further analysis highlights that deeper or more complex trees, composite item names, and removing necessary condition near the middle of a path all increase the likelihood of hallucinations, underscoring the persistent challenges LLMs face in identifying unanswerable math problems. 

**Abstract (ZH)**: 大型语言模型（LLMs）现在在标准数学文字问题基准测试（如GSM8K）上实现了接近人类的性能，但其真正的推理能力仍存在争议。一个主要问题是，模型经常对无法回答的问题生成自信但缺乏依据的答案。我们引入了TreeCut，这是一个合成数据集，通过将每个问题表示为一棵树并移除选定的必要条件，系统地生成无限个无法回答的数学文字问题及其可回答的对应问题。实验显示，在最坏情况下，TreeCut有效诱导了GPT-4o和o3-mini等大型语言模型生成幻觉，比例分别为61%和42%。进一步分析表明，更深或更复杂的树、复合项目名称以及路径中间移除必要条件均增加幻觉的可能性，突显了LLMs在识别无法回答的数学问题方面持续面临的挑战。 

---
# The Self-Improvement Paradox: Can Language Models Bootstrap Reasoning Capabilities without External Scaffolding? 

**Title (ZH)**: 自我改进悖论：语言模型能否在无需外部支撑的情况下自动生成推理能力？ 

**Authors**: Yutao Sun, Mingshuai Chen, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13441)  

**Abstract**: Self-improving large language models (LLMs) -- i.e., to improve the performance of an LLM by fine-tuning it with synthetic data generated by itself -- is a promising way to advance the capabilities of LLMs while avoiding extensive supervision. Existing approaches to self-improvement often rely on external supervision signals in the form of seed data and/or assistance from third-party models. This paper presents Crescent -- a simple yet effective framework for generating high-quality synthetic question-answer data in a fully autonomous manner. Crescent first elicits the LLM to generate raw questions via a bait prompt, then diversifies these questions leveraging a rejection sampling-based self-deduplication, and finally feeds the questions to the LLM and collects the corresponding answers by means of majority voting. We show that Crescent sheds light on the potential of true self-improvement with zero external supervision signals for math reasoning; in particular, Crescent-generated question-answer pairs suffice to (i) improve the reasoning capabilities of an LLM while preserving its general performance (especially in the 0-shot setting); and (ii) distil LLM knowledge to weaker models more effectively than existing methods based on seed-dataset augmentation. 

**Abstract (ZH)**: 自我提升的大语言模型（LLMs）——即通过自身生成的合成数据对LLM进行微调以提高其性能——是一种避免大量监督而促进LLM能力发展的有前途的方法。这种自我提升的方法通常依赖外部监督信号，如种子数据和/或第三方模型的帮助。本文提出Crescent——一种简单而有效的全自主生成高质量合成问答数据的框架。Crescent首先通过诱饵提示促使LLM生成原始问题，然后利用基于拒绝采样的自我去重方法增加问题多样性，最后通过多数投票将问题输入LLM并收集相应的答案。我们展示，Crescent揭示了在零外部监督信号下进行真正自我提升的潜力，特别是在数学推理领域；特别是，Crescent生成的问答对足以（i）在保持LLM整体性能的情况下提高其推理能力（尤其是在零样本设置中）；（ii）更有效地将LLM知识传授给较弱的模型，优于基于种子数据集增强的方法。 

---
# TabSD: Large Free-Form Table Question Answering with SQL-Based Table Decomposition 

**Title (ZH)**: TabSD: 基于SQL分解的大规模自由格式表格问答 

**Authors**: Yuxiang Wang, Junhao Gan, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13422)  

**Abstract**: Question answering on free-form tables (TableQA) is challenging due to the absence of predefined schemas and the presence of noise in large tables. While Large Language Models (LLMs) have shown promise in TableQA, they struggle with large free-form tables and noise sensitivity. To address these challenges, we propose TabSD, a SQL-based decomposition model that enhances LLMs' ability to process large free-form tables. TabSD generates SQL queries to guide the table decomposition, remove noise, and processes sub-tables for better answer generation. Additionally, SQL Verifier refines SQL outputs to enhance decomposition accuracy. We introduce two TableQA datasets with large free-form tables, SLQA and SEQA, which consist solely of large free-form tables and will be publicly available. Experimental results on four benchmark datasets demonstrate that TABSD outperforms the best-existing baseline models by 23.07%, 2.84%, 23.24% and 9.32% in accuracy, respectively, highlighting its effectiveness in handling large and noisy free-form tables. 

**Abstract (ZH)**: 基于SQL的表格分解模型TabSD在处理自由格式表格问答（TableQA）中的挑战 

---
# RLTHF: Targeted Human Feedback for LLM Alignment 

**Title (ZH)**: RLTHF: 目标导向的人类反馈促进大模型对齐 

**Authors**: Yifei Xu, Tusher Chakraborty, Emre Kıcıman, Bibek Aryal, Eduardo Rodrigues, Srinagesh Sharma, Roberto Estevao, Maria Angels de Luis Balaguer, Jessica Wolk, Rafael Padilha, Leonardo Nunes, Shobana Balakrishnan, Songwu Lu, Ranveer Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.13417)  

**Abstract**: Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF's strategic data curation. 

**Abstract (ZH)**: 基于人类-AI混合框架的大型语言模型细调以实现用户偏好对齐 

---
# Explore-Construct-Filter: An Automated Framework for Rich and Reliable API Knowledge Graph Construction 

**Title (ZH)**: 探索-构建-过滤：一种自动化的丰富可靠API知识图构建框架 

**Authors**: Yanbang Sun, Qing Huang, Xiaoxue Ren, Zhenchang Xing, Xiaohong Li, Junjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13412)  

**Abstract**: The API Knowledge Graph (API KG) is a structured network that models API entities and their relations, providing essential semantic insights for tasks such as API recommendation, code generation, and API misuse detection. However, constructing a knowledge-rich and reliable API KG presents several challenges. Existing schema-based methods rely heavily on manual annotations to design KG schemas, leading to excessive manual overhead. On the other hand, schema-free methods, due to the lack of schema guidance, are prone to introducing noise, reducing the KG's reliability. To address these issues, we propose the Explore-Construct-Filter framework, an automated approach for API KG construction based on large language models (LLMs). This framework consists of three key modules: 1) KG exploration: LLMs simulate the workflow of annotators to automatically design a schema with comprehensive type triples, minimizing human intervention; 2) KG construction: Guided by the schema, LLMs extract instance triples to construct a rich yet unreliable API KG; 3) KG filtering: Removing invalid type triples and suspicious instance triples to construct a rich and reliable API KG. Experimental results demonstrate that our method surpasses the state-of-the-art method, achieving a 25.2% improvement in F1 score. Moreover, the Explore-Construct-Filter framework proves effective, with the KG exploration module increasing KG richness by 133.6% and the KG filtering module improving reliability by 26.6%. Finally, cross-model experiments confirm the generalizability of our framework. 

**Abstract (ZH)**: API知识图谱（API KG）是建模API实体及其关系的结构化网络，为API推荐、代码生成和API滥用检测等任务提供重要的语义洞察。然而，构建丰富且可靠的API KG存在若干挑战。现有的基于模式的方法高度依赖手工标注来设计KG模式，导致手工劳动量过大。另一方面，无模式方法由于缺乏模式指导，容易引入噪声，降低KG的可靠性。为解决这些问题，我们提出了一种基于大规模语言模型（LLMs）的API KG自动构建框架——Explore-Construct-Filter框架。该框架包含三个关键模块：1）KG探索：LLMs模拟标注人员的工作流程，自动设计具有全面类型三元组的模式，最大限度减少人工干预；2）KG构建：根据模式指导，LLMs提取实例三元组以构建丰富但不可靠的API KG；3）KG过滤：去除无效类型三元组和可疑实例三元组，构建丰富且可靠的API KG。实验结果显示，我们的方法超越了现有最佳方法，F1分数提高25.2%。此外，Explore-Construct-Filter框架的有效性证明，KG探索模块使KG丰富度提高133.6%，KG过滤模块提高了26.6%的可靠性。最后，跨模型实验确认了该框架的普适性。 

---
# $\mathtt{GeLLM^3O}$: Generalizing Large Language Models for Multi-property Molecule Optimization 

**Title (ZH)**: GeLLM^3O: 多性质分子优化的大语言模型泛化 

**Authors**: Vishal Dey, Xiao Hu, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2502.13398)  

**Abstract**: Despite recent advancements, most computational methods for molecule optimization are constrained to single- or double-property optimization tasks and suffer from poor scalability and generalizability to novel optimization tasks. Meanwhile, Large Language Models (LLMs) demonstrate remarkable out-of-domain generalizability to novel tasks. To demonstrate LLMs' potential for molecule optimization, we introduce $\mathtt{MoMUInstruct}$, the first high-quality instruction-tuning dataset specifically focused on complex multi-property molecule optimization tasks. Leveraging $\mathtt{MoMUInstruct}$, we develop $\mathtt{GeLLM^3O}$s, a series of instruction-tuned LLMs for molecule optimization. Extensive evaluations across 5 in-domain and 5 out-of-domain tasks demonstrate that $\mathtt{GeLLM^3O}$s consistently outperform state-of-the-art baselines. $\mathtt{GeLLM^3O}$s also exhibit outstanding zero-shot generalization to unseen tasks, significantly outperforming powerful closed-source LLMs. Such strong generalizability demonstrates the tremendous potential of $\mathtt{GeLLM^3O}$s as foundational models for molecule optimization, thereby tackling novel optimization tasks without resource-intensive retraining. $\mathtt{MoMUInstruct}$, models, and code are accessible through this https URL. 

**Abstract (ZH)**: 尽管近期取得了进展，大多数分子优化的计算方法仍然局限于单一或双性质优化任务，并且在扩展性和新型优化任务的一般化能力方面表现不佳。与此同时，大规模语言模型（LLMs）在新型任务上的跨域一般化表现出色。为了展示LLMs在分子优化领域的潜力，我们引入了$\mathtt{MoMUInstruct}$，这是首个专注于复杂多性质分子优化任务的高质量指令调优数据集。基于$\mathtt{MoMUInstruct}$，我们开发了$\mathtt{GeLLM^3O}$系列指令调优的大规模语言模型，用于分子优化。广泛的任务评估表明，$\mathtt{GeLLM^3O}$系列模型在5个领域内和5个领域外任务中都优于最先进的基线模型。$\mathtt{GeLLM^3O}$模型还展现出在未见过的任务上的出色零样本泛化能力，显著优于强大的闭源大规模语言模型。这种强大的泛化能力表明$\mathtt{GeLLM^3O}$模型作为分子优化领域的基础模型具有巨大的潜力，能够无需资源密集型的重新训练来应对新型优化任务。$\mathtt{MoMUInstruct}$数据集、模型和代码可通过以下链接访问：https://。 

---
# RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering 

**Title (ZH)**: RGAR：基于 recurrence 生成增强检索的医学事实感知问答 

**Authors**: Sichu Liang, Linhai Zhang, Hongyu Zhu, Wenwen Wang, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.13361)  

**Abstract**: Medical question answering requires extensive access to specialized conceptual knowledge. The current paradigm, Retrieval-Augmented Generation (RAG), acquires expertise medical knowledge through large-scale corpus retrieval and uses this knowledge to guide a general-purpose large language model (LLM) for generating answers. However, existing retrieval approaches often overlook the importance of factual knowledge, which limits the relevance of retrieved conceptual knowledge and restricts its applicability in real-world scenarios, such as clinical decision-making based on Electronic Health Records (EHRs). This paper introduces RGAR, a recurrence generation-augmented retrieval framework that retrieves both relevant factual and conceptual knowledge from dual sources (i.e., EHRs and the corpus), allowing them to interact and refine each another. Through extensive evaluation across three factual-aware medical question answering benchmarks, RGAR establishes a new state-of-the-art performance among medical RAG systems. Notably, the Llama-3.1-8B-Instruct model with RGAR surpasses the considerably larger, RAG-enhanced GPT-3.5. Our findings demonstrate the benefit of extracting factual knowledge for retrieval, which consistently yields improved generation quality. 

**Abstract (ZH)**: 医疗问答需要广泛获取专门的概念知识。当前的范式，检索增强生成（RAG），通过大规模语料库检索获得医学专业知识，并利用这些知识引导通用大型语言模型（LLM）生成答案。然而，现有的检索方法往往忽略了事实知识的重要性，这限制了检索概念知识的相关性，并在诸如基于电子健康记录（EHRs）的临床决策制定等实际场景中限制了其适用性。本文介绍了一种名为RGAR的循环生成增强检索框架，该框架从双重来源（即EHRs和语料库）检索相关的事实知识和概念知识，允许它们相互作用和相互完善。通过在三个事实感知的医疗问答基准上的广泛评估，RGAR在医疗RAG系统中建立了新的最佳性能。值得注意的是，使用RGAR的Llama-3.1-8B-Instruct模型超越了显著更大的、增强了的RAG-GPT-3.5模型。我们的研究结果表明，提取事实知识对于检索的益处，持续地提高了生成质量。 

---
# Language Models are Few-Shot Graders 

**Title (ZH)**: 语言模型是少-shot评分器 

**Authors**: Chenyan Zhao, Mariana Silva, Seth Poulsen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13337)  

**Abstract**: Providing evaluations to student work is a critical component of effective student learning, and automating its process can significantly reduce the workload on human graders. Automatic Short Answer Grading (ASAG) systems, enabled by advancements in Large Language Models (LLMs), offer a promising solution for assessing and providing instant feedback for open-ended student responses. In this paper, we present an ASAG pipeline leveraging state-of-the-art LLMs. Our new LLM-based ASAG pipeline achieves better performances than existing custom-built models on the same datasets. We also compare the grading performance of three OpenAI models: GPT-4, GPT-4o, and o1-preview. Our results demonstrate that GPT-4o achieves the best balance between accuracy and cost-effectiveness. On the other hand, o1-preview, despite higher accuracy, exhibits a larger variance in error that makes it less practical for classroom use. We investigate the effects of incorporating instructor-graded examples into prompts using no examples, random selection, and Retrieval-Augmented Generation (RAG)-based selection strategies. Our findings indicate that providing graded examples enhances grading accuracy, with RAG-based selection outperforming random selection. Additionally, integrating grading rubrics improves accuracy by offering a structured standard for evaluation. 

**Abstract (ZH)**: 基于大型语言模型的自动短回答评分管道：性能分析与策略优化 

---
# Language Models Can Predict Their Own Behavior 

**Title (ZH)**: 语言模型可以预测自身的行为 

**Authors**: Dhananjay Ashok, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2502.13329)  

**Abstract**: Autoregressive Language Models output text by sequentially predicting the next token to generate, with modern methods like Chain-of-Thought (CoT) prompting achieving state-of-the-art reasoning capabilities by scaling the number of generated tokens. However, are there times when we can infer how the model will behave (e.g. abstain from answering a question) early in the computation, making generation unnecessary? We show that internal representation of input tokens alone can often precisely predict, not just the next token, but eventual behavior over the entire output sequence. We leverage this capacity and learn probes on internal states to create early warning (and exit) systems. Specifically, if the probes can confidently estimate the way the LM is going to behave, then the system will avoid generating tokens altogether and return the estimated behavior instead. On 27 text classification datasets spanning five different tasks, we apply this method to estimate the eventual answer of an LM under CoT prompting, reducing inference costs by 65% (average) while suffering an accuracy loss of no more than 1.4% (worst case). We demonstrate the potential of this method to pre-emptively identify when a model will abstain from answering a question, fail to follow output format specifications, or give a low-confidence response. We explore the limits of this capability, showing that probes generalize to unseen datasets, but perform worse when LM outputs are longer and struggle to predict properties that require access to knowledge that the models themselves lack. Encouragingly, performance scales with model size, suggesting applicability to the largest of models 

**Abstract (ZH)**: 自回归语言模型通过序贯预测下一个词令牌来生成文本，现代方法如Chain-of-Thought (CoT) 提示技术通过扩大生成的词令牌数量实现了最先进的推理能力。然而，在计算的早期我们是否可以推断出模型的行为（例如，避免回答某个问题），从而使得生成变得没有必要？我们表明，仅输入词令牌的内部表示通常可以精确预测最终生成序列的整体行为，而不仅仅是下一个词。我们利用这一能力，通过在内部状态上学习探针来创建早期预警（和退出）系统。具体而言，如果探针能够自信地估计自回归语言模型将如何表现，则系统将完全避免生成词令牌，而是返回估计的行为。在涵盖五个不同任务的27个文本分类数据集上，我们应用此方法以CoT提示方式估计自回归语言模型的最终答案，减少了65%的推理成本（平均值），同时准确性损失不超过1.4%（最坏情况）。我们展示了此方法预判模型何时避免回答问题、无法遵循输出格式规范或给出低置信度响应的潜力。我们探讨了这一能力的极限，表明探针能够泛化到未见过的数据集，但在自回归语言模型输出较长且难以预测需要模型自身所缺乏知识的属性时表现较差。令人鼓舞的是，性能随着模型规模的增加而提高，表明该方法适用于最大的模型。 

---
# Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors 

**Title (ZH)**: 基于逐轮验证的对话辅导剂训练：大语言模型作为你的编程导师的有趣案例 

**Authors**: Jian Wang, Yinpei Dai, Yichi Zhang, Ziqiao Ma, Wenjie Li, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2502.13311)  

**Abstract**: Intelligent tutoring agents powered by large language models (LLMs) have been increasingly explored to deliver personalized guidance in areas such as language learning and science education. However, their capabilities in guiding users to solve complex real-world tasks remain underexplored. To address this limitation, in this work, we focus on coding tutoring, a challenging problem that requires tutors to proactively guide students toward completing predefined coding tasks. We propose a novel agent workflow, Trace-and-Verify (TRAVER), which combines knowledge tracing to estimate a student's knowledge state and turn-by-turn verification to ensure effective guidance toward task completion. We introduce DICT, an automatic evaluation protocol that assesses tutor agents holistically using controlled student simulation and code generation tests. Extensive experiments reveal the challenges of coding tutoring and demonstrate that TRAVER achieves a significantly higher success rate. Although we use code tutoring as an example in this paper, our results and findings can be extended beyond coding, providing valuable insights into advancing tutoring agents for a variety of tasks. 

**Abstract (ZH)**: 由大型语言模型驱动的智能辅导代理在语言学习和科学教育等领域提供了个性化指导，然而它们在引导用户解决复杂实际任务方面的能力仍待探索。为解决这一局限，本文专注于编码辅导这一具有挑战性的问题，要求辅导代理积极引导学生完成预定义的编码任务。我们提出了一种新的代理工作流——追踪与验证（TRACE-AND-VERIFY，TRAVER），该工作流结合了知识追踪以估计学生知识状态，并通过逐一验证确保对任务完成的有效指导。我们介绍了DICT，一种自动评估协议，通过受控的学生模拟和代码生成测试全方位评估辅导代理。大量实验揭示了编码辅导的挑战，并证明了TRAVER获得了显著更高的成功率。尽管本文以代码辅导为例，但我们的结果和发现可以扩展到其他领域，为各类任务的辅导代理研发提供了宝贵的见解。 

---
# Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models 

**Title (ZH)**: 逐级困惑度导向细化以提高大型语言模型的高效链式推理 

**Authors**: Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2502.13260)  

**Abstract**: Chain-of-Thought (CoT) reasoning, which breaks down complex tasks into intermediate reasoning steps, has significantly enhanced the performance of large language models (LLMs) on challenging tasks. However, the detailed reasoning process in CoT often incurs long generation times and high computational costs, partly due to the inclusion of unnecessary steps. To address this, we propose a method to identify critical reasoning steps using perplexity as a measure of their importance: a step is deemed critical if its removal causes a significant increase in perplexity. Our method enables models to focus solely on generating these critical steps. This can be achieved through two approaches: refining demonstration examples in few-shot CoT or fine-tuning the model using selected examples that include only critical steps. Comprehensive experiments validate the effectiveness of our method, which achieves a better balance between the reasoning accuracy and efficiency of CoT. 

**Abstract (ZH)**: Chain-of-Thought (CoT)推理，通过将复杂任务分解为中间推理步骤，显著提升了大规模语言模型（LLMs）在挑战性任务上的性能。然而，CoT中的详细推理过程往往导致生成时间长和高计算成本，部分原因是包含了不必要的步骤。为解决这一问题，我们提出了一种使用困惑度作为重要性指标来识别关键推理步骤的方法：如果移除某步骤会导致困惑度显著增加，则该步骤被视为关键步骤。该方法使模型能够仅聚焦于生成这些关键步骤。这可以通过两种方式实现：改进少量示例CoT中的演示示例，或使用仅包含关键步骤的选定示例对模型进行微调。全面的实验验证了该方法的有效性，实现了CoT在推理准确性和效率上的更好平衡。 

---
# HumT DumT: Measuring and controlling human-like language in LLMs 

**Title (ZH)**: HumT DumT: 测量和控制LLM中的人类语言特性 

**Authors**: Myra Cheng, Sunny Yu, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.13259)  

**Abstract**: Should LLMs generate language that makes them seem human? Human-like language might improve user experience, but might also lead to overreliance and stereotyping. Assessing these potential impacts requires a systematic way to measure human-like tone in LLM outputs. We introduce HumT and SocioT, metrics for human-like tone and other dimensions of social perceptions in text data based on relative probabilities from an LLM. By measuring HumT across preference and usage datasets, we find that users prefer less human-like outputs from LLMs. HumT also offers insights into the impacts of anthropomorphism: human-like LLM outputs are highly correlated with warmth, social closeness, femininity, and low status, which are closely linked to the aforementioned harms. We introduce DumT, a method using HumT to systematically control and reduce the degree of human-like tone while preserving model performance. DumT offers a practical approach for mitigating risks associated with anthropomorphic language generation. 

**Abstract (ZH)**: LLMs生成类人类语言是否合适？类人类语言可能改善用户体验，但也可能导致过度依赖和刻板印象。评估这些潜在影响需要一种系统的方法来衡量LLM输出中的人类化语气。我们引入了HumT和SocioT，基于LLM相对概率的文本数据中人类化语气和社会感知的度量标准。通过对偏好和使用数据集中的HumT进行测量，我们发现用户更倾向于LLM生成的较不类人类的输出。HumT还提供了关于拟人化影响的见解：类人类的LLM输出与温暖、社交亲近、女性化和低地位高度相关，这些都与上述危害密切相关。我们引入了DumT，一种使用HumT系统地控制和减少人类化语气程度的方法，同时保持模型性能。DumT提供了一种缓解拟人化语言生成风险的实用方法。 

---
# Neural Attention Search 

**Title (ZH)**: 神经注意力搜索 

**Authors**: Difan Deng, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2502.13251)  

**Abstract**: We present Neural Attention Search (NAtS), a framework that automatically evaluates the importance of each token within a sequence and determines if the corresponding token can be dropped after several steps. This approach can efficiently reduce the KV cache sizes required by transformer-based models during inference and thus reduce inference costs. In this paper, we design a search space that contains three token types: (i) Global Tokens will be preserved and queried by all the following tokens. (ii) Local Tokens survive until the next global token appears. (iii) Sliding Window Tokens have an impact on the inference of a fixed size of the next following tokens. Similar to the One-Shot Neural Architecture Search approach, this token-type information can be learned jointly with the architecture weights via a learnable attention mask. Experiments on both training a new transformer from scratch and fine-tuning existing large language models show that NAtS can efficiently reduce the KV cache size required for the models while maintaining the models' performance. 

**Abstract (ZH)**: 基于神经注意力的搜索框架（NAtS）：自动评估序列中每个令牌的重要性并确定是否可以在若干步后删除对应的令牌 

---
# SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering? 

**Title (ZH)**: SearchRAG: 搜索引擎能助力基于大语言模型的医疗问答吗？ 

**Authors**: Yucheng Shi, Tianze Yang, Canyu Chen, Quanzheng Li, Tianming Liu, Xiang Li, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13233)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in general domains but often struggle with tasks requiring specialized knowledge. Conventional Retrieval-Augmented Generation (RAG) techniques typically retrieve external information from static knowledge bases, which can be outdated or incomplete, missing fine-grained clinical details essential for accurate medical question answering. In this work, we propose SearchRAG, a novel framework that overcomes these limitations by leveraging real-time search engines. Our method employs synthetic query generation to convert complex medical questions into search-engine-friendly queries and utilizes uncertainty-based knowledge selection to filter and incorporate the most relevant and informative medical knowledge into the LLM's input. Experimental results demonstrate that our method significantly improves response accuracy in medical question answering tasks, particularly for complex questions requiring detailed and up-to-date knowledge. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在通用领域展示了出色的能力，但在需要专业领域知识的任务上常常表现不佳。传统的检索增强生成（RAG）技术通常从静态知识库中检索外部信息，这些知识库可能过时或不完整，缺乏准确医疗问答所必需的细粒度临床细节。本文提出了一种名为SearchRAG的新框架，该框架通过利用实时搜索引擎来克服这些局限性。我们的方法采用合成查询生成将复杂的医疗问题转换为搜索引擎友好的查询，并利用基于不确定性的知识选择来筛选和整合最相关的、最有信息量的医疗知识以供LLM输入。实验结果表明，我们的方法在医疗问答任务中显著提高了响应准确性，特别是在需要详细和最新知识的复杂问题上。 

---
# Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations 

**Title (ZH)**: 两张票优于一张票：在战略性的LLM操纵下的公平和准确招聘 

**Authors**: Lee Cohen, Jack Hsieh, Connie Hong, Judy Hanwen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13221)  

**Abstract**: In an era of increasingly capable foundation models, job seekers are turning to generative AI tools to enhance their application materials. However, unequal access to and knowledge about generative AI tools can harm both employers and candidates by reducing the accuracy of hiring decisions and giving some candidates an unfair advantage. To address these challenges, we introduce a new variant of the strategic classification framework tailored to manipulations performed using large language models, accommodating varying levels of manipulations and stochastic outcomes. We propose a ``two-ticket'' scheme, where the hiring algorithm applies an additional manipulation to each submitted resume and considers this manipulated version together with the original submitted resume. We establish theoretical guarantees for this scheme, showing improvements for both the fairness and accuracy of hiring decisions when the true positive rate is maximized subject to a no false positives constraint. We further generalize this approach to an $n$-ticket scheme and prove that hiring outcomes converge to a fixed, group-independent decision, eliminating disparities arising from differential LLM access. Finally, we empirically validate our framework and the performance of our two-ticket scheme on real resumes using an open-source resume screening tool. 

**Abstract (ZH)**: 在基础模型能力日益强大的时代，求职者开始利用生成式AI工具来增强他们的申请材料。然而，生成式AI工具获取的不平等和知识的不均衡可能导致雇主和求职者双方受损，降低招聘决策的准确性，并给予部分求职者不公平的优势。为应对这些挑战，我们提出了一种针对使用大规模语言模型进行操控的战略分类框架的新变体，该框架能够适应不同水平的操控和不确定性结果。我们提出了一种“双票”方案，其中招聘算法对每个提交的简历进行额外的操控，并将操控后的版本与原始提交简历一同考虑。我们为该方案建立了理论保证，证明在真阳性率最大化且无假阳性约束的情况下，该方案可以改进招聘决策的公平性和准确性。我们进一步将该方法推广到“n票”方案，并证明招聘结果将收敛到一个固定的、与群体无关的决策，消除由于不同访问大规模语言模型导致的差异。最后，我们使用一个开源简历筛选工具，在实际简历上 empirically 验证了我们框架和“双票”方案的性能。 

---
# Thinking Outside the (Gray) Box: A Context-Based Score for Assessing Value and Originality in Neural Text Generation 

**Title (ZH)**: 跳出（灰色）框架：基于上下文的评分方法评估神经文本生成的价值与原创性 

**Authors**: Giorgio Franceschelli, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2502.13207)  

**Abstract**: Despite the increasing use of large language models for creative tasks, their outputs often lack diversity. Common solutions, such as sampling at higher temperatures, can compromise the quality of the results. Drawing on information theory, we propose a context-based score to quantitatively evaluate value and originality. This score incentivizes accuracy and adherence to the request while fostering divergence from the learned distribution. We propose using our score as a reward in a reinforcement learning framework to fine-tune large language models for maximum performance. We validate our strategy through experiments in poetry generation and math problem solving, demonstrating that it enhances the value and originality of the generated solutions. 

**Abstract (ZH)**: 尽管大型语言模型在创造性任务中的应用日益增多，但其输出往往缺乏多样性。常见的解决方案，如在更高温度下采样，可能会牺牲结果的质量。借鉴信息论原理，我们提出一种基于上下文的评分方法，以定量评估价值和原创性。该评分方法激励准确性并符合请求，同时促进远离已学习分布的差异。我们建议将我们的评分方法作为强化学习框架中的奖励，以优化大型语言模型的性能。通过诗歌生成和数学问题解决实验验证了我们的策略，结果显示该方法可以提升生成解决方案的价值和原创性。 

---
# MoBA: Mixture of Block Attention for Long-Context LLMs 

**Title (ZH)**: MoBA：长上下文语言模型的块注意力混合模型 

**Authors**: Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun Lai, Yanru Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Neo Y. Zhang, Zhilin Yang, Xinyu Zhou, Mingxing Zhang, Jiezhong Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13189)  

**Abstract**: Scaling the effective context length is essential for advancing large language models (LLMs) toward artificial general intelligence (AGI). However, the quadratic increase in computational complexity inherent in traditional attention mechanisms presents a prohibitive overhead. Existing approaches either impose strongly biased structures, such as sink or window attention which are task-specific, or radically modify the attention mechanism into linear approximations, whose performance in complex reasoning tasks remains inadequately explored.
In this work, we propose a solution that adheres to the ``less structure'' principle, allowing the model to determine where to attend autonomously, rather than introducing predefined biases. We introduce Mixture of Block Attention (MoBA), an innovative approach that applies the principles of Mixture of Experts (MoE) to the attention mechanism. This novel architecture demonstrates superior performance on long-context tasks while offering a key advantage: the ability to seamlessly transition between full and sparse attention, enhancing efficiency without the risk of compromising performance. MoBA has already been deployed to support Kimi's long-context requests and demonstrates significant advancements in efficient attention computation for LLMs. Our code is available at this https URL. 

**Abstract (ZH)**: 扩展有效的上下文长度是推动大型语言模型（LLMs）向通用人工智能（AGI）发展的关键。然而，传统注意力机制中固有的计算复杂度平方级增长带来了难以承受的开销。现有方法要么引入强偏置结构，如sink或窗口注意力，这些结构具有任务特定性，要么从根本上将注意力机制修改为线性近似，这些方法在复杂推理任务中的性能尚未得到充分探索。
在本文中，我们提出了一种遵循“少结构”原则的解决方案，允许模型自主决定注意力的关注点，而不是引入预定义的偏置。我们引入了块注意力混合（MoBA）这一创新方法，将专家混合（MoE）的原则应用于注意力机制。这一新的架构在处理长上下文任务时表现出色，并且提供了一个关键优势：能够无缝切换到全注意力和稀疏注意力之间，提高效率而不牺牲性能。MoBA已经在支持Kimi的长上下文请求中部署，并展示了在LLMs中高效注意力计算方面的显著进展。我们的代码可在以下链接获取：this https URL。 

---
# PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models 

**Title (ZH)**: PTQ1.61: 探索极低位宽后训练量化方法的真正极限以应用于大型语言模型 

**Authors**: Jiaqi Zhao, Miao Zhang, Ming Wang, Yuzhang Shang, Kaihao Zhang, Weili Guan, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13179)  

**Abstract**: Large Language Models (LLMs) suffer severe performance degradation when facing extremely low-bit (sub 2-bit) quantization. Several existing sub 2-bit post-training quantization (PTQ) methods utilize a mix-precision scheme by leveraging an unstructured fine-grained mask to explicitly distinguish salient weights, while which introduces an extra 1-bit or more per weight. To explore the real limit of PTQ, we propose an extremely low-bit PTQ method called PTQ1.61, which enables weight quantization to 1.61-bit for the first time. Specifically, we first introduce a one-dimensional structured mask with negligibly additional 0.0002-bit per weight based on input activations from the perspective of reducing the upper bound of quantization error to allocate corresponding salient weight channels to 4-bit. For non-salient channels binarization, an efficient block-wise scaling factors optimization framework is then presented to take implicit row-wise correlations and angular biases into account. Different from prior works that concentrate on adjusting quantization methodologies, we further propose a novel paradigm called quantization preprocessing, where we argue that transforming the weight distribution of the pretrained model before quantization can alleviate the difficulty in per-channel extremely low-bit PTQ. Extensive experiments indicate our PTQ1.61 achieves state-of-the-art performance in extremely low-bit quantization. Codes are available at this https URL. 

**Abstract (ZH)**: 极大低比特（低于2比特）量化下大语言模型（LLMs）的性能严重下降。现有的一些低于2比特后训练量化（PTQ）方法通过利用非结构化的细粒度掩码引入混精度方案，明确区分重要权重，但每个权重引入了额外的1比特或多比特。为了探索PTQ的真实极限，我们提出了一种称为PTQ1.61的极度低比特PTQ方法，这使得权重量化首次达到1.61比特。具体来说，我们首先从降低量化误差上限的角度出发，引入了一维结构化掩码，基于输入激活，每个权重仅附加忽略不计的0.0002比特，将重要权重通道分配给4比特。对于非重要通道的二值化，我们提出了一种高效的分块标量因子优化框架，考虑了行内隐式的相关性和角度偏差。不同于专注于调整量化方法的先前工作，我们进一步提出了一种新颖的预处理量化范式，即在量化前转换预训练模型的权重分布，以缓解单通道极度低比特量化中的困难。大量实验表明，我们的PTQ1.61在极度低比特量化中达到了最先进的性能。相关代码可在以下链接获取。 

---
# Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis 

**Title (ZH)**: 在大规模语言模型中基准测试训练后量化：全面分类、统一评估与比较分析 

**Authors**: Jiaqi Zhao, Ming Wang, Miao Zhang, Yuzhang Shang, Xuebo Liu, Yaowei Wang, Min Zhang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.13178)  

**Abstract**: Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation this http URL comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches. 

**Abstract (ZH)**: Post-Training Quantization技术在大规模语言模型压缩中的广泛应用得益于其高效性和低资源需求。然而，现有研究缺乏对每种Post-Training Quantization策略优越且适用场景的深入分析。此外，现有算法主要关注性能，忽略了模型规模、性能和量化位宽之间的权衡。为缓解这些困惑，我们在本文中提供了一个新的大规模语言模型Post-Training Quantization基准。首先，为了支持我们的基准，我们通过对现有主流方法的计算策略进行详细审查（例如，基于优化的、基于补偿的等），提出了一种全面的分类体系。然后，我们在每个类别中采用了基准方法，涵盖了不同规模（7B-70B）、量化位宽、训练级别（LLaMA1/2/3/3.1）、架构（Mixtral、DeepSeekMoE和Mamba）和模态（LLaVA1.5和VILA1.5）的一系列广泛评估。通过对比分析结果，我们总结了每种Post-Training Quantization策略的优点和模型规模-量化位宽权衡，特别是在性能方面的考虑。最后，我们进一步提出，补偿策略与其他Post-Training Quantization策略的结合可以在多种鲁棒性方面达到最佳效果。我们相信，我们的基准将为大规模语言模型的部署和未来Post-Training Quantization方法的研究提供有价值的建议。 

---
# KL Penalty Control via Perturbation for Direct Preference Optimization 

**Title (ZH)**: KL正则化 penalty 控制 via 扰动方法用于直接偏好优化 

**Authors**: Sangkyu Lee, Janghoon Han, Hosung Song, Stanley Jungkyu Choi, Honglak Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13177)  

**Abstract**: Direct Preference Optimization (DPO) demonstrates the advantage of aligning a large language model with human preference using only an offline dataset. However, DPO has the limitation that the KL penalty, which prevents excessive deviation from the reference model, is static throughout the training process. Several methods try to turn this static KL penalty into a dynamic one, but no approach can adaptively assign different KL penalties for each preference pair. In this paper, we propose $\varepsilon$-Direct Preference Optimization ($\varepsilon$-DPO), which allows adaptive control of the KL penalty strength $\beta$ for each preference pair. Specifically, $\varepsilon$-DPO adaptively controls $\beta$ for each preference pair based on the monotonicity of logits as a preference model under the perturbation of $\beta$ during training by simply reusing the logit of the current policy and the reference policy. Experimental results show that $\varepsilon$-DPO outperforms existing direct alignment algorithms and KL penalty relaxation methods on general chatbot benchmarks, highlighting the significance of adaptive KL penalty relaxation at the instance-level in DPO. 

**Abstract (ZH)**: ε-直接偏好优化（ε-DPO）：在DPO中实现偏奋试配的自适应KL惩罚强度 

---
# BaKlaVa -- Budgeted Allocation of KV cache for Long-context Inference 

**Title (ZH)**: BaKlaVa——预算分配的KV缓存用于长上下文推理 

**Authors**: Ahmed Burak Gulhan, Krishna Teja Chitty-Venkata, Murali Emani, Mahmut Kandemir, Venkatram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2502.13176)  

**Abstract**: In Large Language Model (LLM) inference, Key-Value (KV) caches (KV-caches) are essential for reducing time complexity. However, they result in a linear increase in GPU memory as the context length grows. While recent work explores KV-cache eviction and compression policies to reduce memory usage, they often consider uniform KV-caches across all attention heads, leading to suboptimal performance. We introduce BaKlaVa, a method to allocate optimal memory for individual KV-caches across the model by estimating the importance of each KV-cache. Our empirical analysis demonstrates that not all KV-caches are equally critical for LLM performance. Using a one-time profiling approach, BaKlaVa assigns optimal memory budgets to each KV-cache. We evaluated our method on LLaMA-3-8B, and Qwen2.5-7B models, achieving up to a 70\% compression ratio while keeping baseline performance and delivering up to an order-of-magnitude accuracy improvement at higher compression levels. 

**Abstract (ZH)**: 大型语言模型（LLM）推理中，键值（KV）缓存（KV-caches）对于减少时间复杂性至关重要。然而，随着上下文长度的增长，它们会导致GPU内存线性增加。虽然最近的工作探索了KV缓存的淘汰和压缩策略以减少内存使用，但它们通常考虑所有注意力头上的均匀KV缓存，导致性能不佳。我们提出了BaKlaVa方法，通过估计每个KV缓存的重要性为模型中的各个KV缓存分配最优内存。我们的实证分析表明，并非所有KV缓存对LLM性能都同等关键。通过一次性 profilng 方法，BaKlaVa 为每个KV缓存分配了最优的内存预算。我们在LLaMA-3-8B和Qwen2.5-7B模型上评估了该方法，实现了高达70%的压缩比，同时保持基线性能，并在高度压缩水平上实现了数量级的准确率提升。 

---
# Thinking Preference Optimization 

**Title (ZH)**: 偏好优化思考 

**Authors**: Wang Yang, Hongye Jin, Jingfeng Yang, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.13173)  

**Abstract**: Supervised Fine-Tuning (SFT) has been a go-to and effective method for enhancing long chain-of-thought (CoT) reasoning in relatively small LLMs by fine-tuning them with long CoT responses from larger LLMs. To continually improve reasoning abilities, we can either collect new high-quality long CoT reasoning SFT data or repeatedly train on existing SFT datasets. However, acquiring new long CoT SFT data is costly and limited, while repeated training often results in a performance plateau or decline. To further boost the performance with the SFT data, we propose Thinking Preference Optimization (ThinkPO), a simple yet effective post-SFT method that enhances long CoT reasoning without requiring new long CoT responses. Instead, ThinkPO utilizes readily available or easily obtainable short CoT reasoning responses as rejected answers and long CoT responses as chosen answers for the same question. It then applies direct preference optimization to encourage the model to favor longer reasoning outputs. Experiments show that ThinkPO further improves the reasoning performance of SFT-ed models, e.g. it increases math reasoning accuracy of SFT-ed models by 8.6% and output length by 25.9%. Notably, ThinkPO is capable of continually boosting the performance of the publicly distilled SFT model, e.g., increasing the official DeepSeek-R1-Distill-Qwen-7B's performance on MATH500 from 87.4% to 91.2%. 

**Abstract (ZH)**: 监督微调（SFT）已成为通过使用大型语言模型的长链式推理（CoT）响应来增强相对较小的语言模型的长链式推理能力的一种常用且有效的方法。为了不断改进推理能力，我们可以收集新的高质量长链式推理SFT数据，或者反复训练现有的SFT数据集。然而，获取新的长链式推理SFT数据成本高且受限，而反复训练往往会导致性能 plateau 或下降。为了进一步利用SFT数据提升性能，我们提出了一种简单且有效的后微调方法——推理偏好优化（ThinkPO），该方法可以在不需额外长链式推理响应的情况下增强长链式推理能力。ThinkPO 利用现成的或易于获取的短链式推理响应作为拒绝答案，并将长链式推理响应作为相同问题的选择答案。接着，它通过直接偏好优化鼓励模型偏好更长的推理输出。实验表明，ThinkPO 进一步提高了SFT模型的推理性能，例如，使SFT模型的数学推理准确率提高了8.6%，输出长度增加了25.9%。值得注意的是，ThinkPO 能够不断提升公开精简的SFT模型的性能，例如，将官方DeepSeek-R1-Distill-Qwen-7B在MATH500上的性能从87.4%提高到91.2%。 

---
# Unveiling Privacy Risks in LLM Agent Memory 

**Title (ZH)**: 揭示大语言模型代理记忆中的隐私风险 

**Authors**: Bo Wang, Weiyi He, Pengfei He, Shenglai Zeng, Zhen Xiang, Yue Xing, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13172)  

**Abstract**: Large Language Model (LLM) agents have become increasingly prevalent across various real-world applications. They enhance decision-making by storing private user-agent interactions in the memory module for demonstrations, introducing new privacy risks for LLM agents. In this work, we systematically investigate the vulnerability of LLM agents to our proposed Memory EXTRaction Attack (MEXTRA) under a black-box setting. To extract private information from memory, we propose an effective attacking prompt design and an automated prompt generation method based on different levels of knowledge about the LLM agent. Experiments on two representative agents demonstrate the effectiveness of MEXTRA. Moreover, we explore key factors influencing memory leakage from both the agent's and the attacker's perspectives. Our findings highlight the urgent need for effective memory safeguards in LLM agent design and deployment. 

**Abstract (ZH)**: 大规模语言模型（LLM）代理在各种实际应用中日益普遍。它们通过在记忆模块中存储私用户-代理交互来增强决策制定，从而为LLM代理引入新的隐私风险。在本文中，我们在黑盒设置下系统地研究了我们提出的记忆提取攻击（MEXTRA）对LLM代理的漏洞。为了从记忆中提取私有信息，我们提出了一种有效的攻击提示设计方法以及基于对LLM代理不同知识水平的自动化提示生成方法。在两个代表性的代理上的实验证明了MEXTRA的有效性。此外，我们从代理方和攻击方的角度探讨了导致记忆泄露的关键因素。我们的发现强调了在LLM代理设计和部署中需要有效的记忆保护措施的紧迫性。 

---
# SmartLLM: Smart Contract Auditing using Custom Generative AI 

**Title (ZH)**: SmartLLM: 使用自定义生成式AI进行智能合约审计 

**Authors**: Jun Kevin, Pujianto Yugopuspito  

**Link**: [PDF](https://arxiv.org/pdf/2502.13167)  

**Abstract**: Smart contracts are essential to decentralized finance (DeFi) and blockchain ecosystems but are increasingly vulnerable to exploits due to coding errors and complex attack vectors. Traditional static analysis tools and existing vulnerability detection methods often fail to address these challenges comprehensively, leading to high false-positive rates and an inability to detect dynamic vulnerabilities. This paper introduces SmartLLM, a novel approach leveraging fine-tuned LLaMA 3.1 models with Retrieval-Augmented Generation (RAG) to enhance the accuracy and efficiency of smart contract auditing. By integrating domain-specific knowledge from ERC standards and employing advanced techniques such as QLoRA for efficient fine-tuning, SmartLLM achieves superior performance compared to static analysis tools like Mythril and Slither, as well as zero-shot large language model (LLM) prompting methods such as GPT-3.5 and GPT-4. Experimental results demonstrate a perfect recall of 100% and an accuracy score of 70%, highlighting the model's robustness in identifying vulnerabilities, including reentrancy and access control issues. This research advances smart contract security by offering a scalable and effective auditing solution, supporting the secure adoption of decentralized applications. 

**Abstract (ZH)**: 智能合约对于分布式金融(DeFi)和区块链生态系统至关重要，但由于编码错误和复杂的攻击向量，它们日益面临exploits的风险。传统的静态分析工具和现有的漏洞检测方法往往无法全面应对这些挑战，导致高误报率且无法检测动态漏洞。本文提出了一种新颖的方法SmartLLM，利用精细调整的LLaMA 3.1模型结合检索增强生成（RAG）技术，以提高智能合约审计的准确性和效率。通过整合来自ERC标准的领域特定知识，并采用先进的技术如QLoRA进行高效的细调，SmartLLM在性能上优于如Mythril和Slither之类的静态分析工具，以及如GPT-3.5和GPT-4之类的零样本大语言模型（LLM）提示方法。实验结果显示召回率为100%且准确率为70%，突显了该模型在识别包括重入和访问控制问题在内的漏洞方面的鲁棒性。这项研究通过提供可扩展且有效的审计解决方案，促进了智能合约的安全性，并支持分布式应用的安全采用。 

---
# Large Language Models Can Help Mitigate Barren Plateaus 

**Title (ZH)**: 大型语言模型可以帮助缓解 barren plateau 问题。 

**Authors**: Jun Zhuang, Chaowen Guan  

**Link**: [PDF](https://arxiv.org/pdf/2502.13166)  

**Abstract**: In the era of noisy intermediate-scale quantum (NISQ) computing, Quantum Neural Networks (QNNs) have emerged as a promising approach for various applications, yet their training is often hindered by barren plateaus (BPs), where gradient variance vanishes exponentially as the model size increases. To address this challenge, we propose a new Large Language Model (LLM)-driven search framework, AdaInit, that iteratively searches for optimal initial parameters of QNNs to maximize gradient variance and therefore mitigate BPs. Unlike conventional one-time initialization methods, AdaInit dynamically refines QNN's initialization using LLMs with adaptive prompting. Theoretical analysis of the Expected Improvement (EI) proves a supremum for the search, ensuring this process can eventually identify the optimal initial parameter of the QNN. Extensive experiments across four public datasets demonstrate that AdaInit significantly enhances QNN's trainability compared to classic initialization methods, validating its effectiveness in mitigating BPs. 

**Abstract (ZH)**: 在嘈杂的中等规模量子（NISQ）计算时代，量子神经网络（QNNs）已成为多种应用的有前景的方法，但其训练往往受到梯度消失的荒原 plateau （BPs）的阻碍。为了解决这一挑战，我们提出了一种新的基于大型语言模型（LLM）的搜索框架AdaInit，该框架通过迭代搜索优化QNN的初始参数以最大化梯度方差，从而缓解BPs。与传统的单次初始化方法不同，AdaInit使用具有自适应提示的LLMs动态优化QNN的初始化。预期改进（EI）的理论分析证明了搜索的上限，确保该过程最终能够识别出QNN的最佳初始参数。在四个公开数据集上的广泛实验表明，AdaInit显著提高了QNN的可训练性，验证了其在缓解BPs方面的有效性。 

---
# HedgeAgents: A Balanced-aware Multi-agent Financial Trading System 

**Title (ZH)**: HedgeAgents：一个考虑对冲策略的多代理金融交易系统 

**Authors**: Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13165)  

**Abstract**: As automated trading gains traction in the financial market, algorithmic investment strategies are increasingly prominent. While Large Language Models (LLMs) and Agent-based models exhibit promising potential in real-time market analysis and trading decisions, they still experience a significant -20% loss when confronted with rapid declines or frequent fluctuations, impeding their practical application. Hence, there is an imperative to explore a more robust and resilient framework. This paper introduces an innovative multi-agent system, HedgeAgents, aimed at bolstering system robustness via ``hedging'' strategies. In this well-balanced system, an array of hedging agents has been tailored, where HedgeAgents consist of a central fund manager and multiple hedging experts specializing in various financial asset classes. These agents leverage LLMs' cognitive capabilities to make decisions and coordinate through three types of conferences. Benefiting from the powerful understanding of LLMs, our HedgeAgents attained a 70% annualized return and a 400% total return over a period of 3 years. Moreover, we have observed with delight that HedgeAgents can even formulate investment experience comparable to those of human experts (this https URL). 

**Abstract (ZH)**: 随着自动化交易在金融市场中的应用日益普及，算法投资策略逐渐凸显。虽然大型语言模型（LLMs）和基于代理的模型在实时市场分析和交易决策方面展现出潜在的应用前景，但在应对快速下跌或频繁波动时，它们仍然会遭受高达20%的损失，这限制了它们的实际应用。因此，迫切需要探索更具韧性的框架。本文介绍了一种创新的多代理系统——HedgeAgents，旨在通过“风险管理”策略增强系统稳定性。在这一均衡系统中，定制了一组风险管理代理，HedgeAgents包括一个中央资金管理者和多个专注于不同金融资产类别的风险管理专家。这些代理通过利用LLMs的认知能力，并借助三种类型的会议进行协调，实现了70%的年化回报和3年400%的总回报。此外，我们高兴地发现，HedgeAgents甚至能够形成与人类专家相当的投资经验（请参见相关链接）。 

---
# ShieldLearner: A New Paradigm for Jailbreak Attack Defense in LLMs 

**Title (ZH)**: ShieldLearner: 一种新的大模型 jailbreak 攻击防御范式 

**Authors**: Ziyi Ni, Hao Wang, Huacan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13162)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in various domains but remain vulnerable to adversarial jailbreak attacks. Existing prompt-defense strategies, including parameter-modifying and parameter-free approaches, face limitations in adaptability, interpretability, and customization, constraining their effectiveness against evolving threats. To address these challenges, we propose ShieldLearner, a novel paradigm that mimics human learning in defense. Through trial and error, it autonomously distills attack signatures into a Pattern Atlas and synthesizes defense heuristics into a Meta-analysis Framework, enabling systematic and interpretable threat detection. Furthermore, we introduce Adaptive Adversarial Augmentation to generate adversarial variations of successfully defended prompts, enabling continuous self-improvement without model retraining. In addition to standard benchmarks, we create a hard test set by curating adversarial prompts from the Wildjailbreak dataset, emphasizing more concealed malicious intent. Experimental results show that ShieldLearner achieves a significantly higher defense success rate than existing baselines on both conventional and hard test sets, while also operating with lower computational overhead, making it a practical and efficient solution for real-world adversarial defense. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在多个领域取得了显著成功，但仍易受到 adversarial jailbreak 攻击。现有的提示防御策略，包括参数修改型和无参数型方法，存在适应性、可解释性和定制性方面的局限性，限制了其对不断演变威胁的防御效果。为应对这些挑战，我们提出了一种名为 ShieldLearner 的新颖范式，模仿人类在防御中的学习过程。通过试错，它自主提炼攻击特征到模式图谱，并综合生成防御启发式方法到元分析框架，从而实现系统化和可解释性的威胁检测。此外，我们引入了自适应对抗增强，生成防御成功的提示的对抗变体，实现无需模型重新训练的持续自我改进。除了标准基准外，我们还通过从 Wildjailbreak 数据集中精心筛选恶意提示创建了一个更具挑战性的测试集，强调了更隐蔽的恶意意图。实验结果表明，ShieldLearner 在传统和更具挑战性的测试集上都显著提高了防御成功率，同时具有较低的计算开销，使其成为实际应用中对抗防御的实用高效解决方案。 

---
# Understanding Dynamic Diffusion Process of LLM-based Agents under Information Asymmetry 

**Title (ZH)**: 基于信息不对称的LLM代理动态扩散过程理解 

**Authors**: Yiwen Zhang, Yifu Wu, Wenyue Hua, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13160)  

**Abstract**: Large language models have been used to simulate human society using multi-agent systems. Most current social simulation research emphasizes interactive behaviors in fixed environments, ignoring information opacity, relationship variability and diffusion diversity. In this paper, we study the dynamics of information diffusion in 12 asymmetric open environments defined by information content and distribution mechanisms. We first present a general framework to capture the features of information diffusion. Then, we designed a dynamic attention mechanism to help agents allocate attention to different information, addressing the limitations of LLM-based attention. Agents start by responding to external information stimuli within a five-agent group, increasing group size and forming information circles while developing relationships and sharing information. Additionally, we observe the emergence of information cocoons, the evolution of information gaps, and the accumulation of social capital, which are closely linked to psychological, sociological, and communication theories. 

**Abstract (ZH)**: 大型语言模型已使用多agent系统模拟人类社会。现有的大多数社会仿真研究注重固定环境下的交互行为，忽视了信息不透明性、关系多变性和信息扩散多样性。本文研究了由信息内容和分布机制定义的12种不对称开放环境中的信息扩散动力学。我们首先提出了一种通用框架来捕捉信息扩散的特征，然后设计了一种动态注意力机制以帮助代理分配注意力给不同的信息，解决基于LLM的注意力机制的局限性。代理们最初在一个五agent小组内对外部信息刺激做出反应，随着小组规模的扩大并形成信息圈，同时建立关系和分享信息。此外，我们观察到信息茧房的出现、信息鸿沟的演变以及社会资本的积累，这些现象与心理、社会学和传播理论密切相关。 

---
# NestQuant: Nested Lattice Quantization for Matrix Products and LLMs 

**Title (ZH)**: NestQuant: 嵌套格量化方法在矩阵乘法和大语言模型中的应用 

**Authors**: Semyon Savkin, Eitan Porat, Or Ordentlich, Yury Polyanskiy  

**Link**: [PDF](https://arxiv.org/pdf/2502.09720)  

**Abstract**: Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent work have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Meta's SpinQuant (perplexity 7.3). Comparisons on various LLM evaluation benchmarks also show a reduction in performance degradation induced by quantization. 

**Abstract (ZH)**: Post-training 量化 (PTQ) 已成为高效部署大型语言模型 (LLMs) 的关键技术。本文提出了一种基于自相似嵌套格的新型 PTQ 方案 NestQuant，用于权重和激活值。最近的研究已经从理论上证明，此类量化器对于低精度矩阵乘法是信息论意义下的最优解。我们基于 Gosset 格实现了 NestQuant 的实用低复杂度版本，使其成为任何矩阵乘法步骤（例如，在自我注意、MLP 等中）的即插即用量化器。例如，NestQuant 将 Llama-3-8B 的权重、KV 缓存和激活值量化为 4 位， perplexity 达到 6.6，在 wikitext2 上的表现优于 Meta 的 SpinQuant（ perplexity 为 7.3），与未量化模型（ perplexity 为 6.14）相比， perplexity 差距减少了 55%。此外，在各种 LLM 评估基准上的比较也显示量化带来的性能下降有所减少。 

---
