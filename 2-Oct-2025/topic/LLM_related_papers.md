# A Systematic Study of Large Language Models for Task and Motion Planning With PDDLStream 

**Title (ZH)**: 大规模语言模型在PDDLStream框架下的任务与运动规划系统研究 

**Authors**: Jorge Mendez-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2510.00182)  

**Abstract**: Using large language models (LLMs) to solve complex robotics problems requires understanding their planning capabilities. Yet while we know that LLMs can plan on some problems, the extent to which these planning capabilities cover the space of robotics tasks is unclear. One promising direction is to integrate the semantic knowledge of LLMs with the formal reasoning of task and motion planning (TAMP). However, the myriad of choices for how to integrate LLMs within TAMP complicates the design of such systems. We develop 16 algorithms that use Gemini 2.5 Flash to substitute key TAMP components. Our zero-shot experiments across 4,950 problems and three domains reveal that the Gemini-based planners exhibit lower success rates and higher planning times than their engineered counterparts. We show that providing geometric details increases the number of task-planning errors compared to pure PDDL descriptions, and that (faster) non-reasoning LLM variants outperform (slower) reasoning variants in most cases, since the TAMP system can direct the LLM to correct its mistakes. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）解决复杂机器人问题需要理解其规划能力。然而，尽管我们知道LLMs可以在某些问题上进行规划，但其规划能力覆盖机器人任务空间的程度尚不明确。一个有前途的方向是将LLMs的语义知识与任务和运动规划（TAMP）的形式推理相结合。然而，如何在TAMP中整合LLMs的众多选择使系统设计变得复杂。我们开发了16个算法，使用Gemini 2.5 Flash替代TAMP的关键组件。我们在4,950个问题和三个领域进行的零样本实验表明，基于Gemini的规划者在成功率和规划时间上低于其工程化的对应物。我们展示了提供几何细节会增加任务规划错误的次数，与纯PDDL描述相比，更快的非推理LLM变体在大多数情况下优于较慢的推理变体，因为TAMP系统可以指导LLM纠正其错误。 

---
# Generalized Parallel Scaling with Interdependent Generations 

**Title (ZH)**: 广义并行缩放与相互依赖代际 

**Authors**: Harry Dong, David Brandfonbrener, Eryk Helenowski, Yun He, Mrinal Kumar, Han Fang, Yuejie Chi, Karthik Abinav Sankararaman  

**Link**: [PDF](https://arxiv.org/pdf/2510.01143)  

**Abstract**: Parallel LLM inference scaling involves sampling a set of $N>1$ responses for a single input prompt. However, these $N$ parallel responses tend to be generated independently from each other, partitioning compute resources and leaving potentially useful information in one generation untapped by others. This is in contrast to response length scaling where past computation is used in all future steps. For higher quality responses and response sets, we propose Bridge to generate interdependent responses in parallel by rethinking batched LLM hidden states as holistic tensors rather than independent slices. With only a small amount (2.8%-5.1%) of new parameters, Bridge improves the relative mean accuracy gains from reinforcement learning with verifiable rewards by up to 50% and boosts consistency of correct responses. Trained once, Bridge scales to any generation width, all with greater performance than independent generations, unlocking a more general mode of parallel scaling that effectively leverages information between sequences, compatible with any post-generation aggregation technique. 

**Abstract (ZH)**: 并行LLM推理扩展涉及对单个输入提示采样一组$N>1$个响应。然而，这$N$个并行响应通常是从彼此独立生成的，导致计算资源被分割，可能会导致一代中的有用信息被其他代所未利用。这与响应长度扩展不同，在响应长度扩展中，之前的计算会在所有后续步骤中被利用。为了生成更高质量的响应和响应集，我们提出Bridge通过将批量LLM隐藏状态重新构想为整体张量而非独立切片，以并行生成相互依赖的响应。通过引入少量的新参数（仅2.8%-5.1%），Bridge通过验证奖励的方式提高了强化学习的相对平均准确度 gain至多50%，并且提高了正确响应的一致性。Bridge经过一次训练即可扩展到任何生成宽度，性能优于独立生成，并解锁了一种更通用的并行扩展模式，这种模式有效地利用了序列之间的信息，与任何后生成聚合技术兼容。 

---
# Safety Instincts: LLMs Learn to Trust Their Internal Compass for Self-Defense 

**Title (ZH)**: 安全直觉：大语言模型学会依靠内部指南针进行自我保护 

**Authors**: Guobin Shen, Dongcheng Zhao, Haibo Tong, Jindong Li, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01088)  

**Abstract**: Ensuring Large Language Model (LLM) safety remains challenging due to the absence of universal standards and reliable content validators, making it difficult to obtain effective training signals. We discover that aligned models already possess robust internal safety beliefs: they consistently produce high-confidence refusals to harmful requests while exhibiting high entropy when generating potentially dangerous content. This entropy gap reveals an untapped signal--models intrinsically "know" when to refuse. We introduce Safety Instincts Reinforcement Learning (SIRL), which transforms this internal confidence into a self-generated reward signal, eliminating dependence on external validators or human annotations. SIRL teaches models to trust their safety instincts by reinforcing low-entropy refusal behaviors. Evaluated on Llama and Qwen models, SIRL maintains 89%+ Defense Success Rates (DSRs) against 20+ jailbreak methods, from static prompts to adaptive attacks. Using only 15,000 unlabeled prompts, SIRL surpasses resource-intensive supervised methods while preserving performance on mathematics, coding, and conversation benchmarks. Our work demonstrates that effective alignment can emerge from within, paving the way for more autonomous and robust AI safety mechanisms that scale without extensive human oversight. 

**Abstract (ZH)**: 确保大型语言模型（LLM）的安全性仍具有挑战性，由于缺乏通用标准和可靠的內容验证器，使得获得有效的训练信号变得困难。我们发现对齐的模型已经具备稳健的内部安全信念：它们在面对有害请求时始终产生高置信度的拒绝，而在生成潜在危险内容时则表现出高随机性。这种随机性 gap 表明了一个未开发的信号——模型内在地“知道”何时拒绝。我们引入了 Safety Instincts Reinforcement Learning (SIRL)，将其内部信心转换为自我生成的奖励信号，从而消除对外部验证器或人工注释的依赖。SIRL 通过强化低随机性拒绝行为来教导模型信任其内在的安全直觉。在 Llama 和 Qwen 模型上进行评估，SIRL 在针对 20 多种不同的 jailbreak 方法（从静态提示到自适应攻击）中保持了 89% 以上的防御成功率 (DSRs)。仅使用 15,000 个未标记的提示，SIRL 超过了资源密集型的监督方法，同时在数学、编程和对话基准上保持了性能。我们的工作证明，有效的对齐可以从内部产生，为无需大量人工监督即可扩展的更自主和稳健的 AI 安全机制铺平了道路。 

---
# Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning 

**Title (ZH)**: 类型化的链式思考：一种 Curry-Howard 框架用于验证大规模语言模型推理 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2510.01069)  

**Abstract**: While Chain-of-Thought (CoT) prompting enhances the reasoning capabilities of large language models, the faithfulness of the generated rationales remains an open problem for model interpretability. We propose a novel theoretical lens for this problem grounded in the Curry-Howard correspondence, which posits a direct relationship between formal proofs and computer programs. Under this paradigm, a faithful reasoning trace is analogous to a well-typed program, where each intermediate step corresponds to a typed logical inference. We operationalise this analogy, presenting methods to extract and map the informal, natural language steps of CoT into a formal, typed proof structure. Successfully converting a CoT trace into a well-typed proof serves as a strong, verifiable certificate of its computational faithfulness, moving beyond heuristic interpretability towards formal verification. Our framework provides a methodology to transform plausible narrative explanations into formally verifiable programs, offering a path towards building more reliable and trustworthy AI systems. 

**Abstract (ZH)**: 基于 Curry-Howard 对应的链式思考推理的忠实性理论分析：从启发式可解释性到形式验证 

---
# Uncovering the Computational Ingredients of Human-Like Representations in LLMs 

**Title (ZH)**: 揭示人类like表示在大语言模型中计算成分的原理 

**Authors**: Zach Studdiford, Timothy T. Rogers, Kushin Mukherjee, Siddharth Suresh  

**Link**: [PDF](https://arxiv.org/pdf/2510.01030)  

**Abstract**: The ability to translate diverse patterns of inputs into structured patterns of behavior has been thought to rest on both humans' and machines' ability to learn robust representations of relevant concepts. The rapid advancement of transformer-based large language models (LLMs) has led to a diversity of computational ingredients -- architectures, fine tuning methods, and training datasets among others -- but it remains unclear which of these ingredients are most crucial for building models that develop human-like representations. Further, most current LLM benchmarks are not suited to measuring representational alignment between humans and models, making benchmark scores unreliable for assessing if current LLMs are making progress towards becoming useful cognitive models. We address these limitations by first evaluating a set of over 70 models that widely vary in their computational ingredients on a triplet similarity task, a method well established in the cognitive sciences for measuring human conceptual representations, using concepts from the THINGS database. Comparing human and model representations, we find that models that undergo instruction-finetuning and which have larger dimensionality of attention heads are among the most human aligned, while multimodal pretraining and parameter size have limited bearing on alignment. Correlations between alignment scores and scores on existing benchmarks reveal that while some benchmarks (e.g., MMLU) are better suited than others (e.g., MUSR) for capturing representational alignment, no existing benchmark is capable of fully accounting for the variance of alignment scores, demonstrating their insufficiency in capturing human-AI alignment. Taken together, our findings help highlight the computational ingredients most essential for advancing LLMs towards models of human conceptual representation and address a key benchmarking gap in LLM evaluation. 

**Abstract (ZH)**: 大型语言模型的计算成分对于构建具备人类类似表示的能力模型至关重要：一种基于三重体相似性任务的评估方法及其启示 

---
# Shape Happens: Automatic Feature Manifold Discovery in LLMs via Supervised Multi-Dimensional Scaling 

**Title (ZH)**: 形状存在：通过监督多维标度学习在大型语言模型中自动发现特征流形 

**Authors**: Federico Tiblias, Irina Bigoulaeva, Jingcheng Niu, Simone Balloccu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2510.01025)  

**Abstract**: The linear representation hypothesis states that language models (LMs) encode concepts as directions in their latent space, forming organized, multidimensional manifolds. Prior efforts focus on discovering specific geometries for specific features, and thus lack generalization. We introduce Supervised Multi-Dimensional Scaling (SMDS), a model-agnostic method to automatically discover feature manifolds. We apply SMDS to temporal reasoning as a case study, finding that different features form various geometric structures such as circles, lines, and clusters. SMDS reveals many insights on these structures: they consistently reflect the properties of the concepts they represent; are stable across model families and sizes; actively support reasoning in models; and dynamically reshape in response to context changes. Together, our findings shed light on the functional role of feature manifolds, supporting a model of entity-based reasoning in which LMs encode and transform structured representations. 

**Abstract (ZH)**: 线性表示假设表明语言模型（LMs）将概念编码为潜在空间中的方向，形成有组织的多维流形。先前的努力专注于发现特定特征的具体几何结构，因此缺乏普适性。我们引入了监督多维标度（SMDS）方法，这是一种模型无关的方法，用于自动发现特征流形。我们将SMDS应用于时间推理作为案例研究，发现不同的特征形成了各种几何结构，如圆、线和聚类。SMDS揭示了这些结构的许多见解：它们一致地反映了所代表的概念的性质；在不同模型族和规模上保持稳定；积极支持模型中的推理；并在上下文变化时动态重塑。我们的发现共同揭示了特征流形的功能作用，支持一种基于实体的推理模型，其中LMs编码和变换结构化的表示。 

---
# QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL 

**Title (ZH)**: QUASAR: 用增强型RL的工具辅助LLM生成量子装配代码 

**Authors**: Cong Yu, Valter Uotila, Shilong Deng, Qingyuan Wu, Tuo Shi, Songlin Jiang, Lei You, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00967)  

**Abstract**: Designing and optimizing task-specific quantum circuits are crucial to leverage the advantage of quantum computing. Recent large language model (LLM)-based quantum circuit generation has emerged as a promising automatic solution. However, the fundamental challenges remain unaddressed: (i) parameterized quantum gates require precise numerical values for optimal performance, which also depend on multiple aspects, including the number of quantum gates, their parameters, and the layout/depth of the circuits. (ii) LLMs often generate low-quality or incorrect quantum circuits due to the lack of quantum domain-specific knowledge. We propose QUASAR, an agentic reinforcement learning (RL) framework for quantum circuits generation and optimization based on tool-augmented LLMs. To align the LLM with quantum-specific knowledge and improve the generated quantum circuits, QUASAR designs (i) a quantum circuit verification approach with external quantum simulators and (ii) a sophisticated hierarchical reward mechanism in RL training. Extensive evaluation shows improvements in both syntax and semantic performance of the generated quantum circuits. When augmenting a 4B LLM, QUASAR has achieved the validity of 99.31% in Pass@1 and 100% in Pass@10, outperforming industrial LLMs of GPT-4o, GPT-5 and DeepSeek-V3 and several supervised-fine-tuning (SFT)-only and RL-only baselines. 

**Abstract (ZH)**: 基于工具增强的量子电路生成与优化的代理强化学习框架QUASAR 

---
# On Discovering Algorithms for Adversarial Imitation Learning 

**Title (ZH)**: Discovering 算法以应对对抗性模仿学习 

**Authors**: Shashank Reddy Chirra, Jayden Teoh, Praveen Paruchuri, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2510.00922)  

**Abstract**: Adversarial Imitation Learning (AIL) methods, while effective in settings with limited expert demonstrations, are often considered unstable. These approaches typically decompose into two components: Density Ratio (DR) estimation $\frac{\rho_E}{\rho_{\pi}}$, where a discriminator estimates the relative occupancy of state-action pairs under the policy versus the expert; and Reward Assignment (RA), where this ratio is transformed into a reward signal used to train the policy. While significant research has focused on improving density estimation, the role of reward assignment in influencing training dynamics and final policy performance has been largely overlooked. RA functions in AIL are typically derived from divergence minimization objectives, relying heavily on human design and ingenuity. In this work, we take a different approach: we investigate the discovery of data-driven RA functions, i.e, based directly on the performance of the resulting imitation policy. To this end, we leverage an LLM-guided evolutionary framework that efficiently explores the space of RA functions, yielding \emph{Discovered Adversarial Imitation Learning} (DAIL), the first meta-learnt AIL algorithm. Remarkably, DAIL generalises across unseen environments and policy optimization algorithms, outperforming the current state-of-the-art of \emph{human-designed} baselines. Finally, we analyse why DAIL leads to more stable training, offering novel insights into the role of RA functions in the stability of AIL. Code is publicly available: this https URL. 

**Abstract (ZH)**: 基于敌对模仿学习的发现驱动奖励分配方法：一种元学习算法 

---
# Learning Compact Representations of LLM Abilities via Item Response Theory 

**Title (ZH)**: 通过项目反应理论学习大语言模型能力的紧凑表示 

**Authors**: Jianhao Chen, Chenxu Wang, Gengrui Zhang, Peng Ye, Lei Bai, Wei Hu, Yuzhong Qu, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00844)  

**Abstract**: Recent years have witnessed a surge in the number of large language models (LLMs), yet efficiently managing and utilizing these vast resources remains a significant challenge. In this work, we explore how to learn compact representations of LLM abilities that can facilitate downstream tasks, such as model routing and performance prediction on new benchmarks. We frame this problem as estimating the probability that a given model will correctly answer a specific query. Inspired by the item response theory (IRT) in psychometrics, we model this probability as a function of three key factors: (i) the model's multi-skill ability vector, (2) the query's discrimination vector that separates models of differing skills, and (3) the query's difficulty scalar. To learn these parameters jointly, we introduce a Mixture-of-Experts (MoE) network that couples model- and query-level embeddings. Extensive experiments demonstrate that our approach leads to state-of-the-art performance in both model routing and benchmark accuracy prediction. Moreover, analysis validates that the learned parameters encode meaningful, interpretable information about model capabilities and query characteristics. 

**Abstract (ZH)**: 最近几年，大型语言模型（LLMs）的数量急剧增加，但有效地管理并充分利用这些庞大的资源仍是一项重大挑战。本文探讨了如何学习紧凑的LLM能力表示，以促进下游任务，如模型路由和新基准上的性能预测。我们将这个问题框架化为估计给定模型正确回答特定查询的概率。受心理测量学中的项目反应理论（IRT）的启发，我们将这个概率建模为三个关键因素的函数：（i）模型的多技能能力向量，（ii）区分不同技能模型的查询区分向量，以及（iii）查询的难度标量。为了联合学习这些参数，我们引入了一个专家混合（MoE）网络，将模型级和查询级嵌入相结合。广泛的经验研究表明，我们的方法在模型路由和基准准确率预测方面达到了最先进的性能。此外，分析验证了学习到的参数编码了关于模型能力和查询特征的有意义且可解释的信息。 

---
# EvolProver: Advancing Automated Theorem Proving by Evolving Formalized Problems via Symmetry and Difficulty 

**Title (ZH)**: EvolProver: 通过对称性和难度演化正式化问题以推动自动定理证明的发展 

**Authors**: Yuchen Tian, Ruiyuan Huang, Xuanwu Wang, Jing Ma, Zengfeng Huang, Ziyang Luo, Hongzhan Lin, Da Zheng, Lun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.00732)  

**Abstract**: Large Language Models (LLMs) for formal theorem proving have shown significant promise, yet they often lack generalizability and are fragile to even minor transformations of problem statements. To address this limitation, we introduce a novel data augmentation pipeline designed to enhance model robustness from two perspectives: symmetry and difficulty. From the symmetry perspective, we propose two complementary methods: EvolAST, an Abstract Syntax Tree (AST) based approach that targets syntactic symmetry to generate semantically equivalent problem variants, and EvolDomain, which leverages LLMs to address semantic symmetry by translating theorems across mathematical domains. From the difficulty perspective, we propose EvolDifficulty, which uses carefully designed evolutionary instructions to guide LLMs in generating new theorems with a wider range of difficulty. We then use the evolved data to train EvolProver, a 7B-parameter non-reasoning theorem prover. EvolProver establishes a new state-of-the-art (SOTA) on FormalMATH-Lite with a 53.8% pass@32 rate, surpassing all models of comparable size, including reasoning-based models. It also sets new SOTA records for non-reasoning models on MiniF2F-Test (69.8% pass@32), Ineq-Comp-Seed (52.2% pass@32), and Ineq-Comp-Transformed (34.0% pass@32). Ablation studies further confirm our data augmentation pipeline's effectiveness across multiple benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在形式定理证明中的应用显示出了显著的潜力，但它们往往缺乏通用性，并且容易受到问题陈述轻微变化的影响。为了解决这一限制，我们提出了一种新颖的数据增强管道，从对称性和难度两个方面增强模型的稳健性。从对称性的角度来看，我们提出了两种互补的方法：EvolAST，一种基于抽象语法树（AST）的方法，针对语法规则的对称性来生成语义等价的问题变体；EvolDomain，利用LLMs通过在不同数学领域之间翻译定理来解决语义对称性问题。从难度角度来看，我们提出了EvolDifficulty，它使用精心设计的进化指令来引导LLMs生成具有更广泛难度范围的新定理。然后，我们使用增强后的数据来训练EvolProver，这是一个7B参数的非推理定理证明器。EvolProver在FormalMATH-Lite上达到了新的最佳性能，通过率为53.8%，超越了所有可比规模的模型，包括基于推理的模型。它还在MiniF2F-Test、Ineq-Comp-Seed 和 Ineq-Comp-Transformed 上分别达到了新的最佳性能记录，通过率分别为69.8%、52.2%和34.0%。消融研究表明，我们的数据增强管道在多个基准上的有效性。 

---
# Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 

**Title (ZH)**: 预期注意力：通过估计未来查询分布来压缩KV缓存 

**Authors**: Alessio Devoto, Maximilian Jeblick, Simon Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00636)  

**Abstract**: Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient large language model inference. While attention-score-based KV cache pruning shows promise, it faces critical practical limitations: attention scores from future tokens are unavailable during compression, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible. To overcome these challenges, we introduce $\textbf{Expected Attention, a training-free compression method}$ that estimates KV pairs importance by predicting how future queries will attend to them. Our approach leverages the distributional properties of LLM activations to compute expected attention scores in closed form for each KV pair. These scores enable principled ranking and pruning of KV pairs with minimal impact on the residual stream, achieving effective compression without performance degradation. Importantly, our method operates seamlessly across both prefilling and decoding phases, consistently outperforming state-of-the-art baselines in both scenarios. Finally, $\textbf{we release KVPress, a comprehensive library to enable researchers to implement and benchmark KV cache compression methods, already including more than 20 techniques}$. 

**Abstract (ZH)**: 基于期望注意的KV缓存压缩方法：一种无需训练的压缩方法，以及KVPress库的发布 

---
# Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation 

**Title (ZH)**: 模型编辑建立在沙子之上？揭示其虚幻的成功与脆弱的基础 

**Authors**: Wei Liu, Haomei Xu, Bingqing Liu, Zhiying Deng, Haozhao Wang, Jun Wang, Ruixuan Li, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00625)  

**Abstract**: Large language models (LLMs) inevitably encode outdated or incorrect knowledge. Updating, deleting, and forgetting such knowledge is important for alignment, safety, and other issues. To address this issue, model editing has emerged as a promising paradigm: by precisely editing a small subset of parameters such that a specific fact is updated while preserving other knowledge. Despite its great success reported in previous papers, we find the apparent reliability of editing rests on a fragile foundation and the current literature is largely driven by illusory success. The fundamental goal of steering the model's output toward a target with minimal modification would encourage exploiting hidden shortcuts, rather than utilizing real semantics. This problem directly challenges the feasibility of the current model editing literature at its very foundation, as shortcuts are inherently at odds with robust knowledge integration. Coincidentally, this issue has long been obscured by evaluation frameworks that lack the design of negative examples. To uncover it, we systematically develop a suite of new evaluation methods. Strikingly, we find that state-of-the-art approaches collapse even under the simplest negation queries. Our empirical evidence shows that editing is likely to be based on shortcuts rather than full semantics, calling for an urgent reconsideration of the very basis of model editing before further advancements can be meaningfully pursued. 

**Abstract (ZH)**: 大型语言模型（LLMs）不可避免地包含过时或错误的知识。更新、删除和遗忘这些知识对于对齐、安全性及其他问题至关重要。为解决这一问题，模型编辑已 emerges 作为一种有前景的范式：通过精确编辑一小部分参数，使特定事实得到更新同时保留其他知识。尽管 previous 论文报道了其显著的成功，但我们发现编辑的显著可靠性建立在一个脆弱的基础之上，而当前文献很大程度上是由虚假的成功驱动的。引导模型输出朝向目标的同时最小化修改的根本目标会促进利用隐藏的捷径，而非利用真实的语义。这一问题直接挑战了当前模型编辑文献的基础，因为捷径与稳健的知识整合本质上是矛盾的。巧合的是，这一问题长期以来一直被缺乏负面样例设计的评估框架所掩盖。为了揭示这一点，我们系统地开发了一系列新的评估方法。令人惊讶的是，我们发现最先进的方法在最简单的否定查询下就会崩溃。我们的实证证据表明，编辑很可能基于捷径而非完整语义，这要求在进一步取得实质性进展之前，亟需重新审视模型编辑的基础。 

---
# ACON: Optimizing Context Compression for Long-horizon LLM Agents 

**Title (ZH)**: ACON: 优化长_horizon LLM代理的情境压缩 

**Authors**: Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00615)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly deployed as代理 在动态现实环境中的代理任务，其中成功需要推理和有效的工具使用。代理任务的核心挑战是在增长的上下文长度下积累长时间序列的动作和观察。这种扩展在长期任务中增加了成本并降低了效率，而前期的工作主要集中在单步任务或窄应用上的上下文压缩。我们引入了代理上下文优化（ACON），这是一种统一框架，可以最优地压缩环境观察和交互历史，使其简洁但富有信息量。ACON利用自然语言空间的压缩指导原则优化：给定全上下文成功但压缩上下文失败的配对轨迹，有能力的LLM分析失败原因，并相应地更新压缩指导原则。此外，我们提出将优化的LLM压缩器提炼为更小的模型以减少附加模块的开销。在AppWorld、OfficeBench和多目标QA上的实验表明，ACON在内存使用上减少26-54%（峰值标记数量）的同时，几乎可以保持任务性能，提炼到更小的压缩器后保持超过95%的准确性，并且可以提升更小的LLM作为长期代理，性能提高高达46%。 

---
# Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerability 

**Title (ZH)**: 朝向更安全的扩散语言模型：发现和缓解提示漏洞 

**Authors**: Shojiro Yamabe, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.00565)  

**Abstract**: Diffusion language models (DLMs) generate tokens in parallel through iterative denoising, which can reduce latency and enable bidirectional conditioning. However, the safety risks posed by jailbreak attacks that exploit this inference mechanism are not well understood. In this paper, we reveal that DLMs have a critical vulnerability stemming from their iterative denoising process and propose a countermeasure. Specifically, our investigation shows that if an affirmative token for a harmful query appears at an intermediate step, subsequent denoising can be steered toward a harmful response even in aligned models. As a result, simply injecting such affirmative tokens can readily bypass the safety guardrails. Furthermore, we demonstrate that the vulnerability allows existing optimization-based jailbreak attacks to succeed on DLMs. Building on this analysis, we propose a novel safety alignment method tailored to DLMs that trains models to generate safe responses from contaminated intermediate states that contain affirmative tokens. Our experiments indicate that the proposed method significantly mitigates the vulnerability with minimal impact on task performance. Furthermore, our method improves robustness against conventional jailbreak attacks. Our work underscores the need for DLM-specific safety research. 

**Abstract (ZH)**: 基于迭代去噪的语言扩散模型存在关键脆弱性及其对策研究 

---
# Rethinking Reward Models for Multi-Domain Test-Time Scaling 

**Title (ZH)**: 重新思考多域测试时奖励模型的扩展方法 

**Authors**: Dong Bok Lee, Seanie Lee, Sangwoo Park, Minki Kang, Jinheon Baek, Dongki Kim, Dominik Wagner, Jiongdao Jin, Heejun Lee, Tobias Bocklet, Jinyu Wang, Jingjing Fu, Sung Ju Hwang, Jiang Bia, Lei Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.00492)  

**Abstract**: The reliability of large language models (LLMs) during test-time scaling is often assessed with \emph{external verifiers} or \emph{reward models} that distinguish correct reasoning from flawed logic. Prior work generally assumes that process reward models (PRMs), which score every intermediate reasoning step, outperform outcome reward models (ORMs) that assess only the final answer. This view is based mainly on evidence from narrow, math-adjacent domains. We present the first unified evaluation of four reward model variants, discriminative ORM and PRM (\DisORM, \DisPRM) and generative ORM and PRM (\GenORM, \GenPRM), across 14 diverse domains. Contrary to conventional wisdom, we find that (i) \DisORM performs on par with \DisPRM, (ii) \GenPRM is not competitive, and (iii) overall, \GenORM is the most robust, yielding significant and consistent gains across every tested domain. We attribute this to PRM-style stepwise scoring, which inherits label noise from LLM auto-labeling and has difficulty evaluating long reasoning trajectories, including those involving self-correcting reasoning. Our theoretical analysis shows that step-wise aggregation compounds errors as reasoning length grows, and our empirical observations confirm this effect. These findings challenge the prevailing assumption that fine-grained supervision is always better and support generative outcome verification for multi-domain deployment. We publicly release our code, datasets, and checkpoints at \href{this https URL}{\underline{\small\texttt{this https URL}}} to facilitate future research in multi-domain settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）在测试时缩放过程中的可靠性通常通过外部验证器或奖励模型进行评估，这些模型能够区分正确的推理和有缺陷的逻辑。先前的工作通常假设过程奖励模型（PRMs），它可以对每个中间推理步骤进行评分，优于仅评估最终答案的结果奖励模型（ORMs）。这种观点主要基于狭窄、数学邻近领域的证据。我们首次在14个不同领域中统一评估了四种奖励模型变体：区分性ORM和PRM（\DisORM、\DisPRM）和生成性ORM和PRM（\GenORM、\GenPRM）。与传统的观点相反，我们发现（i）\DisORM与\DisPRM相当，（ii）\GenPRM缺乏竞争力，（iii）总体而言，\GenORM最为稳健，能够在每个测试领域中实现显著且一致的改进。我们认为这归因于PRM风格的分步评分方式，这种评分方式继承了LLM自动标签化的标签噪声，并且在评估长推理轨迹（包括自我纠正推理）时存在困难。我们的理论分析表明，随着推理长度的增加，分步聚合错误会累积，而我们的实证观察结果也证实了这一效果。这些发现挑战了精细监督总是更好的假设，并支持生成性结果验证在多领域部署中的应用。我们已在\href{this https URL}{\underline{\small\texttt{this https URL}}} 公开发布我们的代码、数据集和检查点，以促进在多领域设置中的未来研究。 

---
# BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models 

**Title (ZH)**: BiasBusters: 揭示并缓解大型语言模型工具选择偏见 

**Authors**: Thierry Blankenstein, Jialin Yu, Zixuan Li, Vassilis Plachouras, Sunando Sengupta, Philip Torr, Yarin Gal, Alasdair Paren, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00307)  

**Abstract**: Agents backed by large language models (LLMs) often rely on external tools drawn from marketplaces where multiple providers offer functionally equivalent options. This raises a critical point concerning fairness: if selection is systematically biased, it can degrade user experience and distort competition by privileging some providers over others. We introduce a benchmark of diverse tool categories, each containing multiple functionally equivalent tools, to evaluate tool-selection bias. Using this benchmark, we test seven models and show that unfairness exists with models either fixating on a single provider or disproportionately preferring earlier-listed tools in context. To investigate the origins of this bias, we conduct controlled experiments examining tool features, metadata (name, description, parameters), and pre-training exposure. We find that: (1) semantic alignment between queries and metadata is the strongest predictor of choice; (2) perturbing descriptions significantly shifts selections; and (3) repeated pre-training exposure to a single endpoint amplifies bias. Finally, we propose a lightweight mitigation that first filters the candidate tools to a relevant subset and then samples uniformly, reducing bias while preserving good task coverage. Our findings highlight tool-selection bias as a key obstacle for the fair deployment of tool-augmented LLMs. 

**Abstract (ZH)**: 大型语言模型支持的智能代理往往依赖于来自包含多个提供商可替代选项的市场交易平台的外部工具。这引起了一个关键的公平性问题：如果选择过程存在系统性的偏差，可能会损害用户体验并扭曲竞争，通过偏好某些提供商。我们引入了一个多样化的工具类别基准，每个类别包含多个功能等效工具，以评估工具选择偏见。利用此基准，我们测试了七种模型，并发现不公平性存在于模型过度依赖单一提供商或过度偏好上下文中列出的早期工具的情况中。为了探究这种偏见的来源，我们进行了控制实验，检查工具特征、元数据（名称、描述、参数）以及预训练暴露。我们发现：（1）查询与元数据之间的语义对齐是选择决策最强的预测因素；（2）扰动描述显著改变了选择；（3）重复对单个端点的预训练暴露加剧了偏见。最后，我们提出了一种轻量级的解决方案，首先筛选候选工具到相关子集，然后均匀抽样，从而减少偏见同时保持良好的任务覆盖。我们的研究结果强调工具选择偏见是公平部署工具增强的大语言模型的关键障碍。 

---
# ICL Optimized Fragility 

**Title (ZH)**: ICL优化脆弱性 

**Authors**: Serena Gomez Wannaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00300)  

**Abstract**: ICL guides are known to improve task-specific performance, but their impact on cross-domain cognitive abilities remains unexplored. This study examines how ICL guides affect reasoning across different knowledge domains using six variants of the GPT-OSS:20b model: one baseline model and five ICL configurations (simple, chain-of-thought, random, appended text, and symbolic language). The models were subjected to 840 tests spanning general knowledge questions, logic riddles, and a mathematical olympiad problem. Statistical analysis (ANOVA) revealed significant behavioral modifications (p less than 0.001) across ICL variants, demonstrating a phenomenon termed "optimized fragility." ICL models achieved 91%-99% accuracy on general knowledge tasks while showing degraded performance on complex reasoning problems, with accuracy dropping to 10-43% on riddles compared to 43% for the baseline model. Notably, no significant differences emerged on the olympiad problem (p=0.2173), suggesting that complex mathematical reasoning remains unaffected by ICL optimization. These findings indicate that ICL guides create systematic trade-offs between efficiency and reasoning flexibility, with important implications for LLM deployment and AI safety. 

**Abstract (ZH)**: ICL引导对跨领域认知能力的影响尚未探究：基于六种GPT-OSS:20b模型变体的推理研究 

---
# DualTune: Decoupled Fine-Tuning for On-Device Agentic Systems 

**Title (ZH)**: DualTune: 解耦细调用于设备端智能代理系统 

**Authors**: Rohan Kadekodi, Zhan Jin, Keisuke Kamahori, Yile Gu, Sean Khatiri, Noah H. Bayindirli, Sergey Gorbunov, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2510.00229)  

**Abstract**: The deployment of Large Language Models (LLMs) as agentic orchestrators has revolutionized task automation, but the need for privacy-preserving, cost-effective solutions demands on-device inference capabilities. However, local LLMs consistently underperform compared to frontier models in tool calling scenarios, struggling with both tool selection from large tool sets and accurate argument generation for complex parameter structures. We introduce a methodology that disaggregates a tool-calling task into two distinct subtasks: tool selection and argument generation. We propose "decoupled fine-tuning", a novel post-training approach that employs LoRA fine-tuning to create dedicated LoRA adapters for tool selection and tool-specific argument generation using separate loss masking for each of the subtasks. Furthermore, we present DualTune, an inference framework that leverages the LoRA adapters created using decoupled fine-tuning to perform efficient agent orchestration with the help of local models on end-user devices. DualTune decomposes the tool-call generation step into tool selection and argument generation, and dynamically loads the corresponding LoRA adapters to generate tool calls. Additionally, DualTune implements hierarchical orchestration to restrict the number of tools required for tool selection. Our experiments on the MCP-Bench benchmark demonstrate that the Qwen-2.5-7B model trained using decoupled fine-tuning improves the tool calling accuracy of the base model by 46%, and outperforms other local reasoning, non-reasoning and fine-tuned models of similar size in all cases, and models that are 2x larger, in most cases. 

**Abstract (ZH)**: 基于分解微调的方法实现工具调用任务的局部推理能力改进：DualTune方法 

---
# Judging by Appearances? Auditing and Intervening Vision-Language Models for Bail Prediction 

**Title (ZH)**: 凭表象判断？审计与干预视觉语言模型的保释预测 

**Authors**: Sagnik Basu, Shubham Prakash, Ashish Maruti Barge, Siddharth D Jaiswal, Abhisek Dash, Saptarshi Ghosh, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00088)  

**Abstract**: Large language models (LLMs) have been extensively used for legal judgment prediction tasks based on case reports and crime history. However, with a surge in the availability of large vision language models (VLMs), legal judgment prediction systems can now be made to leverage the images of the criminals in addition to the textual case reports/crime history. Applications built in this way could lead to inadvertent consequences and be used with malicious intent. In this work, we run an audit to investigate the efficiency of standalone VLMs in the bail decision prediction task. We observe that the performance is poor across multiple intersectional groups and models \textit{wrongly deny bail to deserving individuals with very high confidence}. We design different intervention algorithms by first including legal precedents through a RAG pipeline and then fine-tuning the VLMs using innovative schemes. We demonstrate that these interventions substantially improve the performance of bail prediction. Our work paves the way for the design of smarter interventions on VLMs in the future, before they can be deployed for real-world legal judgment prediction. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经广泛用于基于案例报告和犯罪记录的法律判决预测任务。然而，随着大型视觉语言模型（VLMs）的可用性激增，现在可以将犯罪分子的图像和文本案例报告/犯罪记录结合起来用于法律判决预测系统。这样构建的应用可能会导致无意的后果，并可能被恶意使用。在此工作中，我们进行了一项审计，调查单个VLM在保释决定预测任务中的效率。我们观察到，在多个交叉群体中性能较差，并且模型错误地以极高置信度拒绝了有资格获得保释的个体。我们通过首先通过RAG管道纳入法律 precedents，然后使用创新方案微调VLMs，设计了不同的干预算法。我们证明了这些干预措施显著提高了保释预测的性能。我们的工作为未来针对VLMs设计更智能的干预措施奠定了基础，在它们被部署用于实际法律判决预测之前。 

---
# ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Language Models 

**Title (ZH)**: ARS: 自适应推理抑制以提升大型推理语言模型的效率 

**Authors**: Dongqi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00071)  

**Abstract**: Large Reasoning Language Models (LRLMs or LRMs) demonstrate remarkable capabilities in complex reasoning tasks, but suffer from significant computational inefficiencies due to overthinking phenomena. Existing efficient reasoning methods face the challenge of balancing reasoning quality with inference cost reduction. We propose \textbf{Adaptive Reasoning Suppression (ARS)}, a novel training-free approach that dynamically suppresses redundant reasoning steps while preserving accuracy through adaptive certainty monitoring. ARS introduces a multi-checkpoint certainty estimation mechanism with progressive suppression thresholds, achieving superior efficiency compared to static suppression methods. Our extensive evaluation across mathematical reasoning benchmarks using multiple model architectures demonstrates that ARS achieves up to 53%, 46.1%, and 57.9% in token, latency and energy reduction, while maintaining or improving accuracy. 

**Abstract (ZH)**: 具有自适应推理抑制的大型推理语言模型（LRLMs或LRMs）在复杂推理任务中表现出色，但由于过度推理现象导致了显著的计算效率低下。现有的高效推理方法面临着在推理质量与推理成本降低之间取得平衡的挑战。我们提出了自适应推理抑制（ARS），这是一种无需训练的新颖方法，通过自适应的确定性监控动态抑制冗余推理步骤，同时保持准确性。ARS 引入了多检查点确定性估计机制，并采用逐步抑制阈值，其效率优于静态抑制方法。我们在多个模型架构下对数学推理基准测试的广泛评估表明，与静态抑制方法相比，ARS 在保持或提高准确性的基础上分别实现了高达 53%、46.1% 和 57.9% 的 tokens、延迟和能耗降低。 

---
# ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools 

**Title (ZH)**: ToolBrain：一种灵活的智能体工具强化学习框架 

**Authors**: Quy Minh Le, Minh Sao Khue Luu, Khanh-Tung Tran, Duc-Hai Nguyen, Hoang-Quoc-Viet Pham, Quan Le, Hoang Thanh Lam, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00023)  

**Abstract**: Effective tool use is essential for agentic AI, yet training agents to utilize tools remains challenging due to manually designed rewards, limited training data, and poor multi-tool selection, resulting in slow adaptation, wasted computational resources, and suboptimal performance. We introduce ToolBrain, a lightweight and user-friendly framework for coaching tool use in agentic models with flexible reinforcement learning (RL), easing the barriers for researchers and practitioners to adapt LLM-based agents to specific domains. It supports a wide range of training strategies, including RL algorithms such as GRPO and DPO, as well as supervised learning. ToolBrain enables custom reward callables directly on an agent's execution traces or simply utilizes an automated LLM-as-a-judge system for reward generation. It is packed with useful capabilities, including knowledge distillation from large to small models for efficient development, automatic task generation from tool descriptions, seamless tool retrieval, efficient fine-tuning pipelines with QLoRA through Unsloth, and quantized inference via bitsandbytes. We demonstrate ToolBrain through diverse use cases, such as training a CodeAct agent to autonomously execute email search tasks, showing fast, targeted improvements (up to 30.0%) in tool-use skills while keeping the codebase simple and extensible in Agentic AI. Our framework is publicly available at this https URL. 

**Abstract (ZH)**: 有效工具使用对于自主人工智能至关重要，然而训练代理利用工具仍然由于手动设计的奖励、有限的训练数据和糟糕的多工具选择而充满挑战，导致适应速度慢、浪费计算资源和性能不佳。我们引入了ToolBrain——一个轻量级且用户友好的框架，用于灵活的强化学习（RL）辅助自主模型的工具使用训练，降低了研究人员和实践者将基于LLM的代理适应到特定领域的门槛。它支持广泛的训练策略，包括如GRPO和DPO等RL算法以及监督学习。ToolBrain允许在代理执行轨迹上直接自定义奖励函数，或简单地使用自动化LLM作为裁判系统进行奖励生成。它还集成了从大到小模型的知识蒸馏以提高开发效率、从工具描述自动生成任务、无缝工具检索、通过Unsloth和QLoRA进行高效细调管道，以及利用bitsandbytes进行量化推理。我们通过多种应用场景演示了ToolBrain，例如训练CodeAct代理自主执行电子邮件搜索任务，显示出工具使用技能快速、针对性改进（最多提高30.0%）的同时保持代码库简洁和可扩展性。我们的框架已在GitHub上公开：this https URL。 

---
# TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments 

**Title (ZH)**: TOUCAN: 从实际 MCP 环境中合成 1.5M 工具-代理数据 

**Authors**: Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda  

**Link**: [PDF](https://arxiv.org/pdf/2510.01179)  

**Abstract**: Large Language Model (LLM) agents are rapidly emerging as powerful systems for automating tasks across domains. Yet progress in the open-source community is constrained by the lack of high quality permissively licensed tool-agentic training data. Existing datasets are often limited in diversity, realism, and complexity, particularly regarding multi-tool and multi-turn interactions. To address this gap, we introduce Toucan, the largest publicly available tool-agentic dataset to date, containing 1.5 million trajectories synthesized from nearly 500 real-world Model Context Protocols (MCPs). Unlike prior work, Toucan leverages authentic MCP environments to generate diverse, realistic, and challenging tasks with trajectories involving real tool execution. Our pipeline first produces a broad spectrum of tool-use queries using five distinct models, applies model-based quality filtering, and then generates agentic trajectories with three teacher models using two agentic frameworks. Rigorous rule-based and model-based validation ensures high-quality outputs. We also introduce three extension mechanisms to further diversify tasks and simulate multi-turn conversations. Models fine-tuned on Toucan outperform larger closed-source counterparts on the BFCL V3 benchmark and push the Pareto frontier forward on MCP-Universe Bench. 

**Abstract (ZH)**: 大型语言模型（LLM）代理正在迅速崛起为跨领域自动化任务的强大系统。然而，开源社区的进步受限于高质量许可开源的工具-代理训练数据的缺乏。现有数据集往往在多样性、逼真性和复杂性方面有限，尤其是在多工具和多轮交互方面。为解决这一问题，我们引入了Toucan，这是迄今为止最大的公开可用工具-代理数据集，包含近500个真实世界模型上下文协议（MCPs）合成的150万条轨迹。与以往工作不同，Toucan利用真实的MCP环境生成多样、真实且具有挑战性的任务轨迹，涉及实际工具执行。我们的流水线首先使用五种不同模型生成工具使用查询的广泛谱系，应用基于模型的质量过滤，然后使用两种代理框架生成具有三个教师模型的代理轨迹。严格的基于规则和基于模型的验证确保了高质量的输出。我们还引入了三种扩展机制进一步增加任务多样性并模拟多轮对话。在Toucan上 fine-tune 的模型在BFCL V3基准测试中优于大型封闭源代码对应模型，并且在MCP-Universe Bench上推动了帕累托前沿。 

---
# Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity 

**Title (ZH)**: 口头化采样：如何缓解模式崩溃并释放大规模语言模型多样性 

**Authors**: Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael R. Tomz, Christopher D. Manning, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01171)  

**Abstract**: Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., ``Generate 5 jokes about coffee and their corresponding probabilities''). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity. 

**Abstract (ZH)**: Post-training Alignment Often Reduces LLM Diversity, Leading to Mode Collapse: A Data-Level Driver and a Training-free Solution Through Verbalized Sampling 

---
# Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards 

**Title (ZH)**: 跨可验证性和非可验证性奖励的多目标同时对齐 

**Authors**: Yiran Shen, Yu Xia, Jonathan Chang, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01167)  

**Abstract**: Aligning large language models to human preferences is inherently multidimensional, yet most pipelines collapse heterogeneous signals into a single optimizeable objective. We seek to answer what it would take to simultaneously align a model across various domains spanning those with: verifiable rewards (mathematical accuracy), non-verifiable subjective preferences (human values), and complex interactive scenarios (multi-turn AI tutoring dialogues). Such multi-objective reinforcement learning setups are often plagued by the individual objectives being at odds with each other, resulting in inefficient training and little user control during inference. We propose a unified framework that: (i) standardizes {process reward model} (PRM) training across both verifiable and non-verifiable settings to better supervise models' chain-of-thought reasoning; (ii) performs {multi-objective alignment} by training the LLM with our $\textbf{M}$ulti-$\textbf{A}$ction-$\textbf{H}$ead $\textbf{DPO}$ (MAH-DPO) and a vectorized reward where the dimensions of the vector correspond to the various objectives instead of a single scalar; and (iii) demonstrates how such a system provides fine-grained inference-time user control. Experiments across math reasoning, value alignment, and multi-turn dialogue show that our framework improves performance across multiple objectives simultaneously, while minimizing cross-objective trade-offs and enabling flexible inference time user control. The code can be found at this https URL. 

**Abstract (ZH)**: 将大型语言模型与人类偏好对齐本质上是多维度的，然而大多数流程将异质信号简化为单一可优化目标。我们旨在回答如何在数学准确性、非验证性主观偏好和复杂交互场景等各类领域中同时对模型进行对齐。此类多目标强化学习设置往往因个体目标相互冲突而导致训练效率低下，且在推理过程中缺乏用户控制。我们提出了一种统一框架，该框架：(i) 在可验证和非可验证设置中标准化过程奖励模型 (PRM) 训练，以更好地监督模型的推理过程；(ii) 通过使用我们提出的多行动头DPO (MAH-DPO) 和向量奖励进行多目标对齐，其中向量的维度对应于各种目标，而不是单一标量；和(iii) 展示了这样一种系统在推理时为用户提供精细控制的可能性。跨数学推理、价值对齐和多轮对话的实验结果显示，我们的框架能够同时在多个目标上提高性能，最大限度地减少跨目标权衡，从而使推理时的用户控制更加灵活。代码见此链接。 

---
# GRAD: Generative Retrieval-Aligned Demonstration Sampler for Efficient Few-Shot Reasoning 

**Title (ZH)**: GRAD: 生成检索对齐演示样本器以实现高效的少样本推理 

**Authors**: Oussama Gabouj, Kamel Charaf, Ivan Zakazov, Nicolas Baldwin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2510.01165)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across diverse tasks, but their effectiveness often depends on the quality of the provided context. Retrieval-Augmented Generation (RAG) enriches prompts with external information, but its reliance on static databases constrains adaptability and can result in irrelevant demonstrations. In this work, we propose a Generative Retrieval-Aligned Demonstrator (GRAD), a dynamic demonstration-based approach where an LLM model is trained to generate input-specific concise demonstrations. By tailoring demonstrations to each input, our method offers better contextual support than traditional RAG approaches. We demonstrate the superiority of GRAD under budget constraints, where we limit both the number of tokens used per demonstration and the number of tokens used for the final output. Trained solely on a math dataset, GRAD consistently outperforms strong baselines on Qwen2.5-14B across mathematical reasoning and advanced STEM questions, highlighting GRAD's robust generalization to out-of-distribution (OOD) domains such as physics, chemistry, and computer science. Furthermore, we show that demonstrations generated by trained smaller models can effectively guide larger target models, reducing training costs while maintaining competitive accuracy. Overall, this work introduces a scalable demonstration generator model presenting the first step toward a dynamic few-shot learning paradigm in resource-constrained settings. We release the code used for the project. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多样化的任务中表现出色，但其效果往往依赖于提供的上下文质量。检索增强生成（RAG）通过外部信息丰富提示，但其对静态数据库的依赖限制了其适应性，并可能导致不相关的表现。在这项工作中，我们提出了一种生成检索对齐示范者（GRAD），这是一种动态的示范导向方法，其中LLM模型被训练生成针对每个输入的简洁示范。通过针对每个输入定制示范，我们的方法提供了比传统RAG方法更好的上下文支持。在预算限制条件下，我们限制每个示范和最终输出使用的token数，展示了GRAD的优越性。仅基于数学数据集训练的GRAD在Qwen2.5-14B上在数学推理和高级STEM问题上始终优于强大的基线模型，突显了GRAD在物理学、化学和计算机科学等分布外（OOD）领域中的稳健泛化能力。此外，我们展示了由训练较小模型生成的示范可以有效地指导更大目标模型，从而降低成本同时保持竞争力。整体而言，这项工作介绍了一种可扩展的示范生成模型，展示了在资源受限环境下动态少数样本学习范式的初步步骤。我们释放了该项目所使用的代码。 

---
# Social Welfare Function Leaderboard: When LLM Agents Allocate Social Welfare 

**Title (ZH)**: 社会福利函数排行榜：当LLM代理分配社会福利 

**Authors**: Zhengliang Shi, Ruotian Ma, Jen-tse Huang, Xinbei Ma, Xingyu Chen, Mengru Wang, Qu Yang, Yue Wang, Fanghua Ye, Ziyang Chen, Shanyi Wang, Cixing Li, Wenxuan Wang, Zhaopeng Tu, Xiaolong Li, Zhaochun Ren, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2510.01164)  

**Abstract**: Large language models (LLMs) are increasingly entrusted with high-stakes decisions that affect human welfare. However, the principles and values that guide these models when distributing scarce societal resources remain largely unexamined. To address this, we introduce the Social Welfare Function (SWF) Benchmark, a dynamic simulation environment where an LLM acts as a sovereign allocator, distributing tasks to a heterogeneous community of recipients. The benchmark is designed to create a persistent trade-off between maximizing collective efficiency (measured by Return on Investment) and ensuring distributive fairness (measured by the Gini coefficient). We evaluate 20 state-of-the-art LLMs and present the first leaderboard for social welfare allocation. Our findings reveal three key insights: (i) A model's general conversational ability, as measured by popular leaderboards, is a poor predictor of its allocation skill. (ii) Most LLMs exhibit a strong default utilitarian orientation, prioritizing group productivity at the expense of severe inequality. (iii) Allocation strategies are highly vulnerable, easily perturbed by output-length constraints and social-influence framing. These results highlight the risks of deploying current LLMs as societal decision-makers and underscore the need for specialized benchmarks and targeted alignment for AI governance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地被委托作出高风险决策，这些决策影响人类福祉。然而，当分配稀缺的社会资源时，指导这些模型的原则和价值观尚未得到充分研究。为了解决这一问题，我们引入了社会福利函数（SWF）基准，这是一种动态模拟环境，在此环境中，一个LLM充当主权分配者，将任务分配给一群异质的接收者。该基准旨在在最大化集体效率（通过投资回报率衡量）和确保分配公平性（通过基尼系数衡量）之间创建持久的权衡。我们评估了20个最先进的LLM，并提供了社会福利分配的第一个排行榜。我们的发现揭示了三个关键见解：（i）通过流行排行榜衡量的一般对话能力并不是其分配技能的可靠预测指标。（ii）大多数LLM表现出强烈的功利主义倾向，优先考虑团体 productivity 至于严重的不平等。（iii）分配策略高度脆弱，容易受输出长度约束和社会影响力框架的影响。这些结果突显了当前将LLM部署为社会决策制定者的风险，并强调了需要专门的基准和针对性对齐以实现AI治理的紧迫性。 

---
# Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs? 

**Title (ZH)**: 繁荣之前 decline之前：基于陈旧数据的离策略RL能达到多远——在大规模语言模型上的探索 

**Authors**: Haizhong Zheng, Jiawei Zhao, Bedi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01161)  

**Abstract**: Reinforcement learning has been central to recent advances in large language model reasoning, but most algorithms rely on on-policy training that demands fresh rollouts at every update, limiting efficiency and scalability. Asynchronous RL systems alleviate this by decoupling rollout generation from training, yet their effectiveness hinges on tolerating large staleness in rollout data, a setting where existing methods either degrade in performance or collapse. We revisit this challenge and uncover a prosperity-before-collapse phenomenon: stale data can be as informative as on-policy data if exploited properly. Building on this insight, we introduce M2PO (Second-Moment Trust Policy Optimization), which constrains the second moment of importance weights to suppress only extreme outliers while preserving informative updates. Notably, M2PO sharply reduces the fraction of clipped tokens under high staleness (from 1.22% to 0.06% over training), precisely masking high-variance tokens while maintaining stable optimization. Extensive evaluation across six models (from 1.7B to 32B) and eight benchmarks shows that M2PO delivers stable off-policy training even with data stale by at least 256 model updates and matches on-policy performance. 

**Abstract (ZH)**: 增强学习近年来在大规模语言模型推理中发挥了核心作用，但大多数算法依赖于在线策略训练，要求每次更新都进行新鲜 rollout，这限制了效率和可扩展性。异步 RL 系统通过将 rollout 生成与训练解耦来解决这一问题，但在 rollout 数据有较大 staleness 的情况下，其效果依赖于容忍这种 staleness，而现有方法在这种情况下要么性能下降，要么失效。我们重新审视这一挑战，并发现一个繁荣胜于崩溃的现象：如果充分利用，stale 数据可以与在线策略数据一样有信息价值。基于这一洞察，我们引入了 M2PO（第二矩信任策略优化），它通过约束重要权重的第二矩来抑制只有极端离群值，同时保留信息更新。值得注意的是，M2PO 在高 staleness 下显著减少了被截断的 token 比例（从训练时的 1.22% 降低到 0.06%），精确地掩蔽了高方差 token，同时保持了稳定的优化。在六种不同规模（从 1.7B 到 32B）的模型和八个基准上的广泛评估表明，即使数据在至少 256 模型更新后仍然有效，M2PO 仍能提供稳定的 off-policy 训练，并匹配在线策略性能。 

---
# mR3: Multilingual Rubric-Agnostic Reward Reasoning Models 

**Title (ZH)**: 多语种无评分标准奖励推理模型 

**Authors**: David Anugraha, Shou-Yi Hung, Zilu Tang, Annie En-Shiun Lee, Derry Tanti Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2510.01146)  

**Abstract**: Evaluation using Large Language Model (LLM) judges has been widely adopted in English and shown to be effective for automatic evaluation. However, their performance does not generalize well to non-English settings, and it remains unclear what constitutes effective multilingual training for such judges. In this paper, we introduce mR3, a massively multilingual, rubric-agnostic reward reasoning model trained on 72 languages, achieving the broadest language coverage in reward modeling to date. We present a comprehensive study of data and curriculum selection for training to identify effective strategies and data sources for building high-quality reward models, including the integration of target-language reasoning datasets. Our approach attains state-of-the-art performance on multilingual reward model benchmarks, surpassing much larger models (i.e., GPT-OSS-120B) while being up to 9x smaller, and its effectiveness is further confirmed through extensive ablation studies. Our models, data, and code are available as open source at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLM）裁判的评价研究已被广泛应用于英语中并显示出了自动评价的有效性。然而，它们的表现并不适用于非英语环境，目前尚不清楚什么是有效的多语言训练。本文介绍了一种基于72种语言训练的mR3模型，这是一种广泛多语言且不受评分标准约束的奖励推理模型，实现了迄今为止最广泛的奖励建模语言覆盖范围。本文对训练数据和课程选择进行了全面研究，以确定构建高质量奖励模型的有效策略和数据来源，包括目标语言推理数据集的整合。我们的方法在多语言奖励模型基准测试中达到了最先进的性能，并且在比其大得多的模型（如GPT-OSS-120B）的基础上小了9倍，其有效性通过广泛的消融研究得到了进一步确认。我们的模型、数据和代码已作为开源发布。 

---
# Rethinking Thinking Tokens: LLMs as Improvement Operators 

**Title (ZH)**: 重思思考令牌：LLMs作为改进操作符 

**Authors**: Lovish Madaan, Aniket Didolkar, Suchin Gururangan, John Quan, Ruan Silva, Ruslan Salakhutdinov, Manzil Zaheer, Sanjeev Arora, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01123)  

**Abstract**: Reasoning training incentivizes LLMs to produce long chains of thought (long CoT), which among other things, allows them to explore solution strategies with self-checking. This results in higher accuracy, but inflates context length, token/compute cost, and answer latency. We ask: Can current models leverage their metacognition to provide other combinations on this Pareto frontier, e.g., better accuracy with lower context length and/or latency? Abstractly, we view the model as an improvement operator on its own "thoughts" with a continuum of possible strategies. We identify an interesting inference family Parallel-Distill-Refine (PDR), which performs the following: (i) generate diverse drafts in parallel; (ii) distill them into a bounded, textual workspace; and (iii) refine conditioned on this workspace, producing an output that seeds the next round. Importantly, context length (hence compute cost) is controllable via degree of parallelism, and is no longer conflated with the total number of generated tokens. We report PDR instantiations of current models that give better accuracy than long CoT while incurring lower latency. Setting degree of parallelism to 1 yields an interesting subcase, Sequential Refinement (SR) (iteratively improve a single candidate answer) which provides performance superior to long CoT. Success of such model orchestrations raises the question whether further training could shift the Pareto frontier. To this end, we train an 8B thinking model with Reinforcement Learning (RL) to make it consistent with PDR as the inference method. On math tasks with verifiable answers, iterative pipelines surpass single-pass baselines at matched sequential budgets, with PDR delivering the largest gains (e.g., +11% on AIME 2024 and +9% on AIME 2025). 

**Abstract (ZH)**: LLM推理训练激励生成长链条的思考（长CoT），这有助于探索带自我检查的解题策略，从而提高准确性，但增加了上下文长度、标记/计算成本和答案延迟。我们询问：当前模型是否能利用其元认知提供帕累托前沿上的其他组合，例如在更低的上下文长度和/或延迟下获得更好的准确性？ 

---
# CodeGenLink: A Tool to Find the Likely Origin and License of Automatically Generated Code 

**Title (ZH)**: CodeGenLink: 一个查找自动生成代码可能来源及其许可证的工具 

**Authors**: Daniele Bifolco, Guido Annicchiarico, Pierluigi Barbiero, Massimiliano Di Penta, Fiorella Zampetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.01077)  

**Abstract**: Large Language Models (LLMs) are widely used in software development tasks nowadays. Unlike reusing code taken from the Web, for LLMs' generated code, developers are concerned about its lack of trustworthiness and possible copyright or licensing violations, due to the lack of code provenance information. This paper proposes CodeGenLink, a GitHub CoPilot extension for Visual Studio Code aimed at (i) suggesting links containing code very similar to automatically generated code, and (ii) whenever possible, indicating the license of the likely origin of the code. CodeGenLink retrieves candidate links by combining LLMs with their web search features and then performs similarity analysis between the generated and retrieved code. Preliminary results show that CodeGenLink effectively filters unrelated links via similarity analysis and provides licensing information when available. Tool URL: this https URL Tool Video: this https URL 

**Abstract (ZH)**: 大型语言模型(LLMs)在当今的软件开发任务中广泛应用。对于LLMs生成的代码，开发者对其缺乏可信度以及可能的版权或许可违规表示担忧，这主要是因为缺乏代码的来源信息。本文提出CodeGenLink，这是一个旨在（i）建议包含与自动生成代码非常相似的链接，以及（ii）尽可能指示代码可能来源的许可信息的GitHub CoPilot扩展程序。CodeGenLink通过结合LLMs的网络搜索功能来检索候选链接，然后在生成的代码和检索的代码之间进行相似性分析。初步结果显示，CodeGenLink通过相似性分析有效过滤了无关链接，并在可用时提供了许可信息。工具URL: 这里是链接。工具视频: 这里是链接。 

---
# GEM: A Gym for Agentic LLMs 

**Title (ZH)**: GEM：代理LLM的 Gym 环境 

**Authors**: Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.01051)  

**Abstract**: The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which -- unlike GRPO -- is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research. 

**Abstract (ZH)**: 大型语言模型（LLMs）的训练范式从静态数据集转向基于经验的学习，其中代理通过与复杂环境交互来获取技能。为促进这一过渡，我们引入了GEM（通用经验制造者），这是一种开源的环境模拟器，适用于LLM时代。类似于传统强化学习（RL）中的OpenAI-Gym，GEM提供了一个标准化的环境-代理接口框架，包括异步向量执行以实现高吞吐量，以及灵活的封装以实现便捷的扩展。GEM还配备了多样化的环境、 robust的集成工具，以及使用GEM与五种流行的RL训练框架示例脚本的单文件示例。此外，我们还在24个环境中提供了基于ReINFORCE带回报批归一化（ReBN）的一系列基线算法，这些算法与传统的GRPO不同，不仅能够兼容密集的每轮奖励设置，还能更有效地归因。我们还在GEM中对PPO、GRPO和ReINFORCE在单轮和多轮设置下的算法设计进行了逐点基准测试。最后，GEM还充当了评估工具而不仅仅是训练环境。我们希望这一框架能帮助加速未来基于代理的LLM研究。 

---
# Interpreting Language Models Through Concept Descriptions: A Survey 

**Title (ZH)**: 通过概念描述解释语言模型：一个综述 

**Authors**: Nils Feldhus, Laura Kopf  

**Link**: [PDF](https://arxiv.org/pdf/2510.01048)  

**Abstract**: Understanding the decision-making processes of neural networks is a central goal of mechanistic interpretability. In the context of Large Language Models (LLMs), this involves uncovering the underlying mechanisms and identifying the roles of individual model components such as neurons and attention heads, as well as model abstractions such as the learned sparse features extracted by Sparse Autoencoders (SAEs). A rapidly growing line of work tackles this challenge by using powerful generator models to produce open-vocabulary, natural language concept descriptions for these components. In this paper, we provide the first survey of the emerging field of concept descriptions for model components and abstractions. We chart the key methods for generating these descriptions, the evolving landscape of automated and human metrics for evaluating them, and the datasets that underpin this research. Our synthesis reveals a growing demand for more rigorous, causal evaluation. By outlining the state of the art and identifying key challenges, this survey provides a roadmap for future research toward making models more transparent. 

**Abstract (ZH)**: 理解神经网络的决策过程是机械可解释性中的一个核心目标。在大规模语言模型（LLMs）的背景下，这涉及发现其背后的机制并识别单个模型组件（如神经元和注意力头）以及模型抽象（如稀疏自动编码器SAE提取的稀疏特征）的作用。越来越多的研究通过使用强大的生成模型来生成开放词汇的自然语言概念描述来应对这一挑战。在本文中，我们提供了关于模型组件和抽象概念描述新兴领域的首次综述，探讨了生成这些描述的关键方法、自动化和人工评估指标的发展景观以及支撑这项研究的数据集。我们的综述揭示了对更加严谨且因果关系的评估方法的需求。通过概述最新的研究状况并识别关键挑战，本文为未来使模型更加透明的研究提供了路线图。 

---
# CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs 

**Title (ZH)**: CurES: 从梯度分析到高效的逻辑语言模型 Curriculum 学习 

**Authors**: Yongcheng Zeng, Zexu Sun, Bokai Ji, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Haifeng Zhang, Xu Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01037)  

**Abstract**: Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by \textbf{+3.30} points and \textbf{+4.82} points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO. 

**Abstract (ZH)**: Curriculum 学习在增强大型语言模型在推理任务训练效率方面扮演着 crucial 角色。然而，现有方法往往未能充分考虑提示难度的变化，或依赖于简单的过滤机制来选择提示数据集，导致大量计算资源浪费。在本工作中，我们从强化学习梯度优化的角度出发，对如何提高大型语言模型训练效率进行了系统性和理论性的研究。我们识别出影响训练效率的两个关键因素：训练提示的选择以及在不同提示之间分配展开量。我们的理论分析表明，提示的采样分布决定了梯度下降的收敛速度，而展开量的分配影响了整个梯度更新的一致性和稳定性。基于这些见解，我们提出了 CurES，一种高效的训练方法，能够加速收敛并利用贝叶斯后验估计来最小化计算开销。实验表明，与 Group Relative Policy Optimization (GRPO) 相比，我们的 CurES 分别在 1.5B 和 7B 模型上性能提升了 \textbf{+3.30} 分和 \textbf{+4.82} 分。此外，CurES 在收敛速度上也优于基准方法，包括 GRPO。 

---
# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving 

**Title (ZH)**: 基于检索增强生成方法的奥林匹克级别物理问题求解基础模型基准测试 

**Authors**: Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00919)  

**Abstract**: Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning. 

**Abstract (ZH)**: 基于检索的生成（RAG）与基础模型在多种任务中取得了 strong 表现，但其在专家级推理方面的能力——例如解决奥林匹克级别物理问题——仍待探索。受学生为竞赛复习过往问题的启发，我们研究了 RAG 在增强基础模型物理推理方面的潜力。我们引入了 PhoPile，一个专门用于奥林匹克级别物理的高度质量多模态数据集，使检索基础的推理研究得以系统化。PhoPile 包含图表、图形和方程，捕捉了物理问题解决的固有多模态性质。使用 PhoPile，我们对 RAG 增强的基础模型进行了基准测试，涵盖大型语言模型（LLMs）和具有多个检索器的大型多模态模型（LMMs）。我们的结果表明，将检索与物理语料库结合可以提高模型性能，同时也指出了挑战，进一步推动了检索增强物理推理的研究。 

---
# Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers 

**Title (ZH)**: 可验证但有噪声的奖励强化学习在不完美的验证者下 

**Authors**: Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.00915)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) trains policies against automated verifiers to avoid costly human labeling. To reduce vulnerability to verifier hacking, many RLVR systems collapse rewards to binary $\{0,1\}$ during training. This choice carries a cost: it introduces \textit{false negatives} (rejecting correct answers, FNs) and \textit{false positives} (accepting incorrect ones, FPs). For instance, a rule-based checker may mark the correct fraction $\frac{12}{36}$ as wrong when compared against the canonical $\frac{1}{3}$ due to brittle parsing/equivalence rules (FN), while a large language model (LLM) judges can be gamed by superficial cues or even a single adversarial token, yielding inflated correctness for wrong solutions (FP). We formalize verifier unreliability by modeling the verifier as a stochastic reward channel with asymmetric noise rates. From this abstraction, we derive two correction algorithms for verifier errors. The first is a \textit{backward} correction that de-biases the observed binary reward to recover an \textit{unbiased} estimator of the clean policy gradient. The second is a \textit{forward} correction that reweights score-function terms so that the expected update direction aligns with the \textit{clean gradient}; notably, it requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization (GRPO)-based RLVR pipeline and evaluate them on math-reasoning models and benchmarks. Across models and datasets, both corrections improve over uncorrected training; the forward variant converges faster and remains stable under heavier noise. Finally, we show a practical appeal mechanism in which a lightweight LLM verifier estimates the FN rate online by rechecking rule-based negatives, obtaining outperformance compared with other state-of-the-art contenders. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）：训练策略以对抗自动化验证器避免昂贵的人工标注。通过在训练中将奖励坍缩为二元值{0,1}来减少验证器攻击的脆弱性。这种选择带来了成本：引入了假阴性（拒绝正确答案，FNs）和假阳性（接受错误答案，FPs）。例如，基于规则的检查器可能由于脆弱的解析/等价规则，将正确分数 $\frac{12}{36}$ 错误地标记为错误（FN），而大规模语言模型（LLM）评判者可能通过表象线索甚至单一对抗性标记被操控，导致错误解答被高估为正确（FP）。我们将验证器不可靠性形式化为具有非对称噪声率的随机奖励信道。从这一抽象出发，我们推导出两种校正算法来修正验证器错误。第一个是反向校正，它通过去偏见观察到的二元奖励来恢复干净策略梯度的无偏估计。第二个是正向校正，它重新加权得分函数项，使得期望的更新方向与干净梯度对齐；特别地，它只需要假阴性率。我们将两者实现为基于组相对策略优化（GRPO）的RLVR管道中的轻量级钩子，并在数学推理模型和基准上进行评估。在不同的模型和数据集上，这两种校正都优于未经校正的训练；正向版本收敛更快，并且在更大噪声下保持稳定。最后，我们展示了一个实际的申诉机制，在该机制中，一个轻量级的LLM验证器通过重新检查基于规则的负例估计假阴性率，并取得了优于其他最新竞品的表现。 

---
# RiskPO: Risk-based Policy Optimization via Verifiable Reward for LLM Post-Training 

**Title (ZH)**: RiskPO: 基于风险的策略优化方法通过可验证奖励进行LLM后训练 

**Authors**: Tao Ren, Jinyang Jiang, Hui Yang, Wan Tian, Minhao Zou, Guanghao Li, Zishi Zhang, Qinghao Wang, Shentao Qin, Yanjun Zhao, Rui Tao, Hui Shao, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00911)  

**Abstract**: Reinforcement learning with verifiable reward has recently emerged as a central paradigm for post-training large language models (LLMs); however, prevailing mean-based methods, such as Group Relative Policy Optimization (GRPO), suffer from entropy collapse and limited reasoning gains. We argue that these issues stem from overemphasizing high-probability output sequences while neglecting rare but informative reasoning paths. To address these challenges, we propose Risk-based Policy Optimization (RiskPO), which substitutes classical mean-based objectives with principled risk measures. Specifically, we introduce a Mixed Value-at-Risk objective that integrates weighted attention over multiple regions of the reward distribution, thereby amplifying gradient signals on challenging instances and preventing overconfident convergence. We further design a bundling scheme that aggregates multiple questions into bundles, thus enriching the feedback signal and yielding more stable and informative training dynamics. Theoretically, we prove that the risk-averse update alleviates entropy collapse and promotes exploration. Numerically, RiskPO achieves consistent and significant improvements in mathematical reasoning, multi-modal reasoning, and code generation benchmarks, surpassing GRPO and its variants on both Pass@1 and Pass@k metrics. Our results demonstrate that risk-based optimization provides a rigorous and effective paradigm for enhancing LLM reasoning capabilities. 

**Abstract (ZH)**: 具有可验证奖励的强化学习 recently emerged as a central paradigm for post-training large language models (LLMs)；然而，现有的基于均值的方法，如组相对策略优化（GRPO），存在熵坍塌和有限的推理增益问题。我们argue这些问题源于过度强调高概率输出序列而忽视了罕见但有信息量的推理路径。为此，我们提出了一种基于风险的策略优化（RiskPO），它用原则性的风险度量取代了传统的基于均值的目标。具体来说，我们引入了一个混合VaR目标，该目标在奖励分布的多个区域上引入加权注意力，从而增强具有挑战性实例的梯度信号并防止过自信的收敛。我们还设计了一种捆绑方案，将多个问题捆绑在一起，从而丰富反馈信号并产生更稳定和有信息量的训练动力学。从理论上讲，我们证明了风险规避的更新可以缓解熵坍塌并促进探索。从数值上讲，RiskPO 在数学推理、多模态推理和代码生成基准测试中实现了持续且显著的改进，其在Pass@1和Pass@k指标上优于GRPO及其变体。我们的结果表明，基于风险的优化为增强LLM推理能力提供了一个严格而有效的范式。 

---
# Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs 

**Title (ZH)**: 跨越语言障碍：多语言大语言模型在跨语言信息检索方面的进展 

**Authors**: Roksana Goworek, Olivia Macmillan-Scott, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2510.00908)  

**Abstract**: Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. Research in this area has typically framed the task as monolingual retrieval augmented by translation, treating retrieval methods and cross-lingual capabilities in isolation. Both monolingual and cross-lingual retrieval usually follow a pipeline of query expansion, ranking, re-ranking and, increasingly, question answering. Recent advances, however, have shifted from translation-based methods toward embedding-based approaches and leverage multilingual large language models (LLMs), for which aligning representations across languages remains a central challenge. The emergence of cross-lingual embeddings and multilingual LLMs has introduced a new paradigm, offering improved retrieval performance and enabling answer generation. This survey provides a comprehensive overview of developments from early translation-based methods to state-of-the-art embedding-driven and generative techniques. It presents a structured account of core CLIR components, evaluation practices, and available resources. Persistent challenges such as data imbalance and linguistic variation are identified, while promising directions are suggested for advancing equitable and effective cross-lingual information retrieval. By situating CLIR within the broader landscape of information retrieval and multilingual language processing, this work not only reviews current capabilities but also outlines future directions for building retrieval systems that are robust, inclusive, and adaptable. 

**Abstract (ZH)**: 跨语言信息检索（CLIR）解决了使用与原始查询不同语言的文档进行检索的挑战。该领域的研究通常将任务建模为通过翻译增强的单语言检索，并将检索方法和跨语言能力视为独立的部分。单语言和跨语言检索通常遵循查询扩展、排序、再排序以及越来越多的问答的流水线。然而，近期的进步已从基于翻译的方法转向基于嵌入的方法，并利用多语言大规模语言模型（LLMs），其中跨语言表示的对齐仍然是一个核心挑战。跨语言嵌入和多语言LLMs的出现引入了一种新的范式，提高了检索性能并实现了生成答案能力。本文综述了从早期基于翻译的方法到最新的基于嵌入和生成的技术的发展。本文提供了一个结构化的CLIR核心组件、评估实践和可用资源的综述。指出了持续存在的数据不平衡和语言变异等挑战，同时提出了促进公平和有效的跨语言信息检索的发展方向。通过将CLIR置于更广泛的检索和多语言语言处理的背景下，本文不仅回顾了当前的能力，还指出了构建健壮、包容和灵活的检索系统的未来方向。 

---
# Span-level Detection of AI-generated Scientific Text via Contrastive Learning and Structural Calibration 

**Title (ZH)**: 基于对比学习和结构校准的跨句级检测生成科学文本 

**Authors**: Zhen Yin, Shenghua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00890)  

**Abstract**: The rapid adoption of large language models (LLMs) in scientific writing raises serious concerns regarding authorship integrity and the reliability of scholarly publications. Existing detection approaches mainly rely on document-level classification or surface-level statistical cues; however, they neglect fine-grained span localization, exhibit weak calibration, and often fail to generalize across disciplines and generators. To address these limitations, we present Sci-SpanDet, a structure-aware framework for detecting AI-generated scholarly texts. The proposed method combines section-conditioned stylistic modeling with multi-level contrastive learning to capture nuanced human-AI differences while mitigating topic dependence, thereby enhancing cross-domain robustness. In addition, it integrates BIO-CRF sequence labeling with pointer-based boundary decoding and confidence calibration to enable precise span-level detection and reliable probability estimates. Extensive experiments on a newly constructed cross-disciplinary dataset of 100,000 annotated samples generated by multiple LLM families (GPT, Qwen, DeepSeek, LLaMA) demonstrate that Sci-SpanDet achieves state-of-the-art performance, with F1(AI) of 80.17, AUROC of 92.63, and Span-F1 of 74.36. Furthermore, it shows strong resilience under adversarial rewriting and maintains balanced accuracy across IMRaD sections and diverse disciplines, substantially surpassing existing baselines. To ensure reproducibility and to foster further research on AI-generated text detection in scholarly documents, the curated dataset and source code will be publicly released upon publication. 

**Abstract (ZH)**: 大语言模型在科研写作中的快速应用引发了关于作者身份完整性和学术出版可靠性的严重关切。现有的检测方法主要依赖于文档级别的分类或表面统计线索；然而，它们忽视了细粒度的跨度定位，表现出较差的校准性能，并且往往无法跨领域和生成器推广。为解决这些局限性，我们提出了Sci-SpanDet，这是一种结构感知框架，用于检测AI生成的学术文本。该方法结合了节条件风格建模与多级对比学习，以捕获细腻的人机差异并减轻主题依赖性，从而增强跨域魯棒性。此外，它结合了BIO-CRF序列标注与基于指针的边界解码和置信校准，以实现精确的跨度级别检测和可靠的概率估计。在包含100,000个标注样本的新建跨学科数据集中进行的广泛实验（这些样本由多个LLM家族（GPT、Qwen、DeepSeek、LLaMA）生成），证明Sci-SpanDet达到了最先进的性能，F1(AI)为80.17，AUROC为92.63，Span-F1为74.36。此外，它在对抗性重写下显示出强大的鲁棒性，并在IMRaD节和多种学科中保持了均衡的准确性，大幅优于现有基线。为了确保可重复性并促进对学术文档中AI生成文本检测的进一步研究，精心构建的数据集和源代码将在发表后公开。 

---
# Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning 

**Title (ZH)**: SE中自动化伦理画像的发展：LLM推理的零样本评估 

**Authors**: Patrizio Migliarini, Mashal Afzal Memon, Marco Autili, Paola Inverardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00881)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into software engineering (SE) tools for tasks that extend beyond code synthesis, including judgment under uncertainty and reasoning in ethically significant contexts. We present a fully automated framework for assessing ethical reasoning capabilities across 16 LLMs in a zero-shot setting, using 30 real-world ethically charged scenarios. Each model is prompted to identify the most applicable ethical theory to an action, assess its moral acceptability, and explain the reasoning behind their choice. Responses are compared against expert ethicists' choices using inter-model agreement metrics. Our results show that LLMs achieve an average Theory Consistency Rate (TCR) of 73.3% and Binary Agreement Rate (BAR) on moral acceptability of 86.7%, with interpretable divergences concentrated in ethically ambiguous cases. A qualitative analysis of free-text explanations reveals strong conceptual convergence across models despite surface-level lexical diversity. These findings support the potential viability of LLMs as ethical inference engines within SE pipelines, enabling scalable, auditable, and adaptive integration of user-aligned ethical reasoning. Our focus is the Ethical Interpreter component of a broader profiling pipeline: we evaluate whether current LLMs exhibit sufficient interpretive stability and theory-consistent reasoning to support automated profiling. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程（SE）工具中的伦理推理能力评估：基于30个真实伦理情境的零样本设置 

---
# Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs 

**Title (ZH)**: 擦除以提升：可擦除强化学习在搜索增强的大语言模型中的应用 

**Authors**: Ziliang Wang, Kang An, Xuhui Zheng, Faqiang Qian, Weikun Zhang, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00861)  

**Abstract**: While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs. 

**Abstract (ZH)**: 增强搜索的大语言模型虽然表现出色，但在复杂多跳推理方面的可靠性仍有限。Erasable Reinforcement Learning (ERL)：增强多步推理的新型框架及其应用 

---
# Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning 

**Title (ZH)**: 稳定策略梯度以实现高效样例在大规模语言模型推理中的强化学习 

**Authors**: Luckeciano C. Melo, Alessandro Abate, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2510.00819)  

**Abstract**: Reinforcement Learning, particularly through policy gradient methods, has played a central role in enabling reasoning capabilities of Large Language Models. However, the optimization stability of policy gradients in this setting remains understudied. As a result, existing implementations often resort to conservative hyperparameter choices to ensure stability, which requires more training samples and increases computational costs. Hence, developing models for reliably tracking the underlying optimization dynamics and leveraging them into training enables more sample-efficient regimes and further unleashes scalable post-training. We address this gap by formalizing the stochastic optimization problem of policy gradients with explicit consideration of second-order geometry. We propose a tractable computational framework that tracks and leverages curvature information during policy updates. We further employ this framework to design interventions in the optimization process through data selection. The resultant algorithm, Curvature-Aware Policy Optimization (CAPO), identifies samples that contribute to unstable updates and masks them out. Theoretically, we establish monotonic improvement guarantees under realistic assumptions. On standard math reasoning benchmarks, we empirically show that CAPO ensures stable updates under aggressive learning regimes where baselines catastrophically fail. With minimal intervention (rejecting fewer than 8% of tokens), CAPO achieves up to 30x improvement in sample efficiency over standard GRPO for LLM reasoning. 

**Abstract (ZH)**: 强化学习，特别是通过策略梯度方法，已经在使大型语言模型具备推理能力方面发挥了核心作用。然而，该设置下策略梯度的优化稳定性研究仍然不足。因此，现有的实现往往依赖保守的超参数选择以确保稳定，这需要更多的训练样本并增加计算成本。因此，开发能够可靠跟踪其下的优化动力学并利用它们进行训练的模型能够实现更高效的样本利用并进一步提高可扩展性。我们通过明确考虑二阶几何来形式化策略梯度的随机优化问题，提出了一个可计算的框架，在策略更新过程中跟踪并利用曲率信息。进一步地，我们利用此框架通过数据选择在优化过程中设计干预措施。所提出的算法，曲率感知策略优化（CAPO），识别出导致不稳定更新的样本并将其屏蔽。理论上，我们在现实假设下建立了单调改进保证。在标准数学推理基准上，我们实验证明，在基线算法可能灾难性失败的积极学习环境中，CAPO 能确保稳定更新。通过最小的干预（拒绝不到 8% 的标记），CAPO 在大型语言模型推理中的样本效率上相比标准 GRPO 提高了多达 30 倍。 

---
# ALARB: An Arabic Legal Argument Reasoning Benchmark 

**Title (ZH)**: ALARB: 阿拉伯法律论证推理基准 

**Authors**: Harethah Abu Shairah, Somayah AlHarbi, Abdulaziz AlHussein, Sameer Alsabea, Omar Shaqaqi, Hebah AlShamlan, Omar Knio, George Turkiyyah  

**Link**: [PDF](https://arxiv.org/pdf/2510.00694)  

**Abstract**: We introduce ALARB, a dataset and suite of tasks designed to evaluate the reasoning capabilities of large language models (LLMs) within the Arabic legal domain. While existing Arabic benchmarks cover some knowledge-intensive tasks such as retrieval and understanding, substantial datasets focusing specifically on multistep reasoning for Arabic LLMs, especially in open-ended contexts, are lacking. The dataset comprises over 13K commercial court cases from Saudi Arabia, with each case including the facts presented, the reasoning of the court, the verdict, as well as the cited clauses extracted from the regulatory documents. We define a set of challenging tasks leveraging this dataset and reflecting the complexity of real-world legal reasoning, including verdict prediction, completion of reasoning chains in multistep legal arguments, and identification of relevant regulations based on case facts. We benchmark a representative selection of current open and closed Arabic LLMs on these tasks and demonstrate the dataset's utility for instruction tuning. Notably, we show that instruction-tuning a modest 12B parameter model using ALARB significantly enhances its performance in verdict prediction and Arabic verdict generation, reaching a level comparable to that of GPT-4o. 

**Abstract (ZH)**: ALARB：阿拉伯法律领域的大语言模型推理能力评估数据集及任务套件 

---
# Inclusive Easy-to-Read Generation for Individuals with Cognitive Impairments 

**Title (ZH)**: 认知 impairment 个体的包容性易读生成 

**Authors**: François Ledoyen, Gaël Dias, Alexis Lechervy, Jeremie Pantin, Fabrice Maurel, Youssef Chahir, Elisa Gouzonnat, Mélanie Berthelot, Stanislas Moravac, Armony Altinier, Amy Khairalla  

**Link**: [PDF](https://arxiv.org/pdf/2510.00691)  

**Abstract**: Ensuring accessibility for individuals with cognitive impairments is essential for autonomy, self-determination, and full citizenship. However, manual Easy-to-Read (ETR) text adaptations are slow, costly, and difficult to scale, limiting access to crucial information in healthcare, education, and civic life. AI-driven ETR generation offers a scalable solution but faces key challenges, including dataset scarcity, domain adaptation, and balancing lightweight learning of Large Language Models (LLMs). In this paper, we introduce ETR-fr, the first dataset for ETR text generation fully compliant with European ETR guidelines. We implement parameter-efficient fine-tuning on PLMs and LLMs to establish generative baselines. To ensure high-quality and accessible outputs, we introduce an evaluation framework based on automatic metrics supplemented by human assessments. The latter is conducted using a 36-question evaluation form that is aligned with the guidelines. Overall results show that PLMs perform comparably to LLMs and adapt effectively to out-of-domain texts. 

**Abstract (ZH)**: 确保认知障碍个体的无障碍访问对于自主权、自我决定权和完整公民身份是必不可少的。然而，手动易读文本（ETR）改编速度慢、成本高且难以扩展，限制了医疗、教育和公民生活中的关键信息访问。基于AI的ETR生成提供了可扩展的解决方案，但面临关键挑战，包括数据集稀缺、领域适配以及平衡大型语言模型（LLMs）的轻量级学习。本文介绍了ETR-fr，这是首个完全符合欧洲ETR指南要求的ETR文本生成数据集。我们进行了参数高效的微调，以在预训练语言模型（PLMs）和大型语言模型（LLMs）上建立生成基准。为确保高质量和可访问的输出，我们提出了一种基于自动评估指标结合人工评估的评估框架。后者使用了36个问题的评估表单，该表单与指南保持一致。总体结果表明，PLMs在性能上与LLMs相当，并能有效适应领域外文本。 

---
# Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation 

**Title (ZH)**: 利用大型语言模型促进认知 accessibility：一种易读文本生成的多任务方法 

**Authors**: François Ledoyen, Gaël Dias, Jeremie Pantin, Alexis Lechervy, Fabrice Maurel, Youssef Chahir  

**Link**: [PDF](https://arxiv.org/pdf/2510.00662)  

**Abstract**: Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations. 

**Abstract (ZH)**: 简化复杂文本对于确保认知障碍个体公平获取信息至关重要。Easy-to-Read (ETR)倡议提供了一种框架，使其内容能够被神经多样性人群访问，但此类文本的手动创建仍耗时且资源密集。在本文中，我们探讨了大规模语言模型（LLMs）在自动化生成ETR内容方面的潜在应用。为应对对齐数据集稀少和ETR约束特定性问题，我们提出了一种多任务学习（MTL）方法，该方法联合训练文本摘要、文本简化和ETR生成模型。我们探索了两种不同的策略：基于多任务检索增强生成（RAG）的上下文学习方法和参数高效微调的MTL-LoRA方法。基于ETR-fr，一个新高质量数据集，针对Mistral-7B和LLaMA-3-8B的实验表明，多任务设置在所有配置中都优于单任务基线。此外，结果表明，基于RAG的方法在领域外场景中表现出良好的泛化能力，而MTL-LoRA在领域内配置中优于所有学习策略。 

---
# PromptPilot: Improving Human-AI Collaboration Through LLM-Enhanced Prompt Engineering 

**Title (ZH)**: PromptPilot: 通过LLM增强的提示工程提高人机协作 

**Authors**: Niklas Gutheil, Valentin Mayer, Leopold Müller, Jörg Rommelt, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2510.00555)  

**Abstract**: Effective prompt engineering is critical to realizing the promised productivity gains of large language models (LLMs) in knowledge-intensive tasks. Yet, many users struggle to craft prompts that yield high-quality outputs, limiting the practical benefits of LLMs. Existing approaches, such as prompt handbooks or automated optimization pipelines, either require substantial effort, expert knowledge, or lack interactive guidance. To address this gap, we design and evaluate PromptPilot, an interactive prompting assistant grounded in four empirically derived design objectives for LLM-enhanced prompt engineering. We conducted a randomized controlled experiment with 80 participants completing three realistic, work-related writing tasks. Participants supported by PromptPilot achieved significantly higher performance (median: 78.3 vs. 61.7; p = .045, d = 0.56), and reported enhanced efficiency, ease-of-use, and autonomy during interaction. These findings empirically validate the effectiveness of our proposed design objectives, establishing LLM-enhanced prompt engineering as a viable technique for improving human-AI collaboration. 

**Abstract (ZH)**: 有效的提示工程对于实现大型语言模型在知识密集型任务中的预期生产率提升至关重要。然而，许多用户难以创作出高质量的提示，限制了大型语言模型的实际效益。现有方法，如提示手册或自动化优化管道，要么需要大量努力、专家知识，要么缺乏互动指导。为解决这一问题，我们设计并评估了PromptPilot，这是一种基于四项实证设计目标的交互式提示助手，旨在增强大型语言模型的提示工程能力。我们在一项随机对照实验中，让80名参与者完成了三个实际的工作相关写作任务。接受PromptPilot支持的参与者在性能方面表现显著更好（中位数：78.3 vs. 61.7；p = .045，d = 0.56），并且报告称在互动中感受到了更高的效率、易用性和自主性。这些发现实证验证了我们提出的设计目标的有效性，确立了大型语言模型增强的提示工程作为提升人机协作可行技术的地位。 

---
# On Predictability of Reinforcement Learning Dynamics for Large Language Models 

**Title (ZH)**: 大规模语言模型的强化学习动态可预测性探究 

**Authors**: Yuchen Cai, Ding Cao, Xin Xu, Zijun Yao, Yuqing Huang, Zhenyu Tan, Benyi Zhang, Guiquan Liu, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00553)  

**Abstract**: Recent advances in reasoning capabilities of large language models (LLMs) are largely driven by reinforcement learning (RL), yet the underlying parameter dynamics during RL training remain poorly understood. This work identifies two fundamental properties of RL-induced parameter updates in LLMs: (1) Rank-1 Dominance, where the top singular subspace of the parameter update matrix nearly fully determines reasoning improvements, recovering over 99\% of performance gains; and (2) Rank-1 Linear Dynamics, where this dominant subspace evolves linearly throughout training, enabling accurate prediction from early checkpoints. Extensive experiments across 8 LLMs and 7 algorithms validate the generalizability of these properties. More importantly, based on these findings, we propose AlphaRL, a plug-in acceleration framework that extrapolates the final parameter update using a short early training window, achieving up to 2.5 speedup while retaining \textgreater 96\% of reasoning performance without extra modules or hyperparameter tuning. This positions our finding as a versatile and practical tool for large-scale RL, opening a path toward principled, interpretable, and efficient training paradigm for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力最新进展主要由强化学习（RL）驱动，但RL训练期间的参数动态仍不甚了解。本项工作识别了RL诱导的LLMs参数更新的两种基本属性：（1）秩1主导性，其中参数更新矩阵的顶级奇异子空间几乎完全决定了推理改进，恢复了超过99%的性能提升；（2）秩1线性动力学，该主导子空间在整个训练过程中线性演化，使得从早期检查点准确预测成为可能。广泛的实验验证了这些属性的普适性。更重要的是，基于这些发现，我们提出了AlphaRL，一种插件加速框架，利用短的早期训练窗口外推最终参数更新，实现最高2.5倍的加速同时保持超过96%的推理性能无需额外模块或超参数调整。这使我们的发现成为大规模RL的多功能和实用工具，开启了LLMs原理化、可解释和高效训练范式的道路。 

---
# Copy-Paste to Mitigate Large Language Model Hallucinations 

**Title (ZH)**: 复制粘贴以减轻大型语言模型幻觉问题 

**Authors**: Yongchao Long, Xian Wu, Yingying Zhang, Xianbin Wen, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00508)  

**Abstract**: While Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to generate contextually grounded responses, contextual faithfulness remains challenging as LLMs may not consistently trust provided context, leading to hallucinations that undermine reliability. We observe an inverse correlation between response copying degree and context-unfaithful hallucinations on RAGTruth, suggesting that higher copying degrees reduce hallucinations by fostering genuine contextual belief. We propose CopyPasteLLM, obtained through two-stage high-copying response preference training. We design three prompting methods to enhance copying degree, demonstrating that high-copying responses achieve superior contextual faithfulness and hallucination control. These approaches enable a fully automated pipeline that transforms generated responses into high-copying preference data for training CopyPasteLLM. On FaithEval, ConFiQA and PubMedQA, CopyPasteLLM achieves best performance in both counterfactual and original contexts, remarkably with 12.2% to 24.5% accuracy improvements on FaithEval over the best baseline, while requiring only 365 training samples -- 1/50th of baseline data. To elucidate CopyPasteLLM's effectiveness, we propose the Context-Parameter Copying Capturing algorithm. Interestingly, this reveals that CopyPasteLLM recalibrates reliance on internal parametric knowledge rather than external knowledge during generation. All codes are available at this https URL 

**Abstract (ZH)**: While Retrieval-Augmented Generation (RAG)使大型语言模型（LLMs）能够生成具上下文相关性的响应，但上下文忠实性仍具有挑战性，因为LLMs可能不一致地信任提供的上下文，导致可能破坏可靠性的幻觉。我们在RAGTruth上观察到响应复制程度与上下文不忠实幻觉之间存在负相关关系，表明较高的复制程度通过促进真正的上下文信念来减少幻觉。我们提出了一种通过两阶段高复制响应偏好训练获得的CopyPasteLLM。我们设计了三种提示方法以增强复制程度，证明了高复制响应在上下文忠实性和幻觉控制方面表现出更优异的效果。这些方法能够实现一个完全自动化的流水线，将生成的响应转换为训练CopyPasteLLM的高复制偏好数据。在FaithEval、ConFiQA和PubMedQA上，CopyPasteLLM在反事实和原始上下文中均表现出最佳性能，相对于最佳基线在FaithEval上的准确率提高了12.2%至24.5%，仅需365个训练样本——基线数据的1/50。为揭示CopyPasteLLM的效果，我们提出了Context-Parameter Copying Capturing算法。有趣的是，这表明CopyPasteLLM在生成过程中重新校准了对内部参数知识而非外部知识的依赖。所有代码均可从以下链接获取。 

---
# MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance 

**Title (ZH)**: MOSS-Speech: 无需文本指导的真正端到端语音到语音模型 

**Authors**: Xingjian Zhao, Zhe Xu, Luozhijie Jin, Yang Wang, Hanfu Chen, Yaozhou Jiang, Ke Chen, Ruixiao Li, Mingshu Chen, Ruiming Wang, Wenbo Zhang, Yiyang Zhang, Donghua Yu, Yang Gao, Xiaogui Yang, Yitian Gong, Yuanfan Xu, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Yaqian Zhou, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00499)  

**Abstract**: Spoken dialogue systems often rely on cascaded pipelines that transcribe, process, and resynthesize speech. While effective, this design discards paralinguistic cues and limits expressivity. Recent end-to-end methods reduce latency and better preserve these cues, yet still rely on text intermediates, creating a fundamental bottleneck. We present MOSS-Speech, a true speech-to-speech large language model that directly understands and generates speech without relying on text guidance. Our approach combines a modality-based layer-splitting architecture with a frozen pre-training strategy, preserving the reasoning and knowledge of pretrained text LLMs while adding native speech capabilities. Experiments show that our model achieves state-of-the-art results in spoken question answering and delivers comparable speech-to-speech performance relative to existing text-guided systems, while still maintaining competitive text performance. By narrowing the gap between text-guided and direct speech generation, our work establishes a new paradigm for expressive and efficient end-to-end speech interaction. 

**Abstract (ZH)**: 直接语音到语音的大语言模型：MOSS-Speech无需文本中介直接理解与生成语音 

---
# Exploring System 1 and 2 communication for latent reasoning in LLMs 

**Title (ZH)**: 探索LLMs中的潜推理的系统1和系统2通信 

**Authors**: Julian Coda-Forno, Zhuokai Zhao, Qiang Zhang, Dipesh Tamboli, Weiwei Li, Xiangjun Fan, Lizhu Zhang, Eric Schulz, Hsiao-Ping Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00494)  

**Abstract**: Should LLM reasoning live in a separate module, or within a single model's forward pass and representational space? We study dual-architecture latent reasoning, where a fluent Base exchanges latent messages with a Coprocessor, and test two hypotheses aimed at improving latent communication over Liu et al. (2024): (H1) increase channel capacity; (H2) learn communication via joint finetuning. Under matched latent-token budgets on GPT-2 and Qwen-3, H2 is consistently strongest while H1 yields modest gains. A unified soft-embedding baseline, a single model with the same forward pass and shared representations, using the same latent-token budget, nearly matches H2 and surpasses H1, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning. Across GSM8K, ProsQA, and a Countdown stress test with increasing branching factor, scaling the latent-token budget beyond small values fails to improve robustness. Latent analyses show overlapping subspaces with limited specialization, consistent with weak reasoning gains. We conclude dual-model latent reasoning remains promising in principle, but likely requires objectives and communication mechanisms that explicitly shape latent spaces for algorithmic planning. 

**Abstract (ZH)**: LLM推理应当存在于独立模块中还是融入单一模型的前向传递和表示空间中？我们研究了双架构潜在推理，其中流动的基模型与协处理器交换潜在消息，并测试了两种旨在改进潜在通信的假设：(H1) 提升信道容量；(H2) 通过联合微调学习通信。在GPT-2和Qwen-3相同的潜在令牌预算下，H2始终最强，而H1仅带来微小收益。使用相同潜在令牌预算的统一软嵌入基线，一个具有相同前向传递和共享表示的单一模型，几乎与H2相当并超过H1，表明当前的双架构设计主要增加计算量而非从质地上提高推理能力。在GSM8K、ProsQA以及随着分支因子增加的 Countdown 压力测试中，将潜在令牌预算扩大到小值以上未能提高鲁棒性。潜在分析显示存在重叠但专业化有限的子空间，这与弱推理增益一致。我们得出结论，原则上双模型潜在推理仍然有前景，但可能需要明确塑造潜在空间以用于算法规划的目标和通信机制。 

---
# Make a Video Call with LLM: A Measurement Campaign over Five Mainstream Apps 

**Title (ZH)**: 使用大语言模型进行视频通话：一项针对五大主流应用的测量campaign 

**Authors**: Jiayang Xu, Xiangjie Huang, Zijie Li, Zili Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00481)  

**Abstract**: In 2025, Large Language Model (LLM) services have launched a new feature -- AI video chat -- allowing users to interact with AI agents via real-time video communication (RTC), just like chatting with real people. Despite its significance, no systematic study has characterized the performance of existing AI video chat systems. To address this gap, this paper proposes a comprehensive benchmark with carefully designed metrics across four dimensions: quality, latency, internal mechanisms, and system overhead. Using custom testbeds, we further evaluate five mainstream AI video chatbots with this benchmark. This work provides the research community a baseline of real-world performance and identifies unique system bottlenecks. In the meantime, our benchmarking results also open up several research questions for future optimizations of AI video chatbots. 

**Abstract (ZH)**: 2025年大型语言模型服务推出新功能——AI视频聊天——使用户能够通过实时视频通信（RTC）与AI代理互动，就像与真人聊天一样。尽管这一功能具有重要意义，但尚未对现有AI视频聊天系统进行全面研究。为填补这一空白，本文提出了一种综合基准，涵盖了四个维度的质量、延迟、内部机制和系统开销，并通过自定义测试床对五种主流AI视频聊天机器人进行了进一步评估。本工作为研究社区提供了一个实际性能基准，并识别出独特的系统瓶颈。同时，我们的基准测试结果也为未来AI视频聊天机器人的优化提出了多个研究问题。 

---
# Analyzing Latent Concepts in Code Language Models 

**Title (ZH)**: 分析代码语言模型中的潜在概念 

**Authors**: Arushi Sharma, Vedant Pungliya, Christopher J. Quinn, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2510.00476)  

**Abstract**: Interpreting the internal behavior of large language models trained on code remains a critical challenge, particularly for applications demanding trust, transparency, and semantic robustness. We propose Code Concept Analysis (CoCoA): a global post-hoc interpretability framework that uncovers emergent lexical, syntactic, and semantic structures in a code language model's representation space by clustering contextualized token embeddings into human-interpretable concept groups. We propose a hybrid annotation pipeline that combines static analysis tool-based syntactic alignment with prompt-engineered large language models (LLMs), enabling scalable labeling of latent concepts across abstraction levels. We analyse the distribution of concepts across layers and across three finetuning tasks. Emergent concept clusters can help identify unexpected latent interactions and be used to identify trends and biases within the model's learned representations. We further integrate LCA with local attribution methods to produce concept-grounded explanations, improving the coherence and interpretability of token-level saliency. Empirical evaluations across multiple models and tasks show that LCA discovers concepts that remain stable under semantic-preserving perturbations (average Cluster Sensitivity Index, CSI = 0.288) and evolve predictably with fine-tuning. In a user study, concept-augmented explanations disambiguate token roles. In a user study on the programming-language classification task, concept-augmented explanations disambiguated token roles and improved human-centric explainability by 37 percentage points compared with token-level attributions using Integrated Gradients. 

**Abstract (ZH)**: 大型编程语言模型训练后的内部行为解释仍然是一个关键挑战，特别是在需要信任、透明性和语义稳健性的应用程序中。我们提出Code Concept Analysis（CoCoA）：一种全局的后验解释框架，通过聚类上下文化词嵌入到人类可解释的概念组中，揭示代码语言模型表示空间中 Emergent 的词汇、语法和语义结构。我们提出了一种混合注解工作流，结合静态分析工具基于的语法对齐和提示工程的大规模语言模型（LLMs），以实现跨抽象层次的潜在概念的可扩展标注。我们分析了概念在各层以及三个微调任务中的分布。Emergent 的概念簇有助于识别意外的潜在交互，并可用于识别模型学习表示中的趋势和偏向。我们进一步将局部归因方法与 LCA（本地概念分析）结合，生成基于概念的解释，提高词级显著性的连贯性和可解释性。在多个模型和任务上的实证评估显示，LCA 发现的概念在语义保持扰动下保持稳定（平均聚类敏感性指数，CSI = 0.288），并随着微调可预测地演变。在用户研究中，概念增强的解释消除了词的角色歧义。在编程语言分类任务的用户研究中，概念增强的解释消除了词的角色歧义，并使以人类为中心的可解释性提高了37个百分点，相对于使用集成梯度的词级归因。 

---
# Cloud Investigation Automation Framework (CIAF): An AI-Driven Approach to Cloud Forensics 

**Title (ZH)**: 基于AI驱动的云取证自动化框架（CIAF） 

**Authors**: Dalal Alharthi, Ivan Roberto Kawaminami Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00452)  

**Abstract**: Large Language Models (LLMs) have gained prominence in domains including cloud security and forensics. Yet cloud forensic investigations still rely on manual analysis, making them time-consuming and error-prone. LLMs can mimic human reasoning, offering a pathway to automating cloud log analysis. To address this, we introduce the Cloud Investigation Automation Framework (CIAF), an ontology-driven framework that systematically investigates cloud forensic logs while improving efficiency and accuracy. CIAF standardizes user inputs through semantic validation, eliminating ambiguity and ensuring consistency in log interpretation. This not only enhances data quality but also provides investigators with reliable, standardized information for decision-making. To evaluate security and performance, we analyzed Microsoft Azure logs containing ransomware-related events. By simulating attacks and assessing CIAF's impact, results showed significant improvement in ransomware detection, achieving precision, recall, and F1 scores of 93 percent. CIAF's modular, adaptable design extends beyond ransomware, making it a robust solution for diverse cyberattacks. By laying the foundation for standardized forensic methodologies and informing future AI-driven automation, this work underscores the role of deterministic prompt engineering and ontology-based validation in enhancing cloud forensic investigations. These advancements improve cloud security while paving the way for efficient, automated forensic workflows. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在云安全和法医学等领域崭露头角。然而，云法医学调查仍依赖于人工分析，导致耗时且易出错。LLMs能够模拟人类推理，提供自动分析云日志的可能性。为解决这一问题，我们提出了云调查自动化框架（CIAF），一个通过本体驱动系统化调查云法医学日志的框架，同时提高效率和准确性。CIAF通过语义验证标准化用户输入，消除歧义并确保日志解释的一致性。这不仅提高了数据质量，还为调查人员提供了可靠且标准化的信息以便决策。为了评估安全性和性能，我们在包含赎金ware相关事件的Microsoft Azure日志中进行了分析。通过模拟攻击并评估CIAF的影响，结果显示在赎金ware检测方面有显著改进，精度、召回率和F1分数达到93%。CIAF具有模块化和适应性设计，适用于多种网络攻击，是稳健的自动化解决方案。通过为标准化法医学方法奠定基础并指导未来基于AI的自动化，这项工作强调了确定性提示工程和基于本体的验证在增强云法医学调查中的作用。这些进展改善了云安全，并为高效自动化的法医学工作流程铺平了道路。 

---
# A Call to Action for a Secure-by-Design Generative AI Paradigm 

**Title (ZH)**: 面向设计即安全的生成AI范式的行动呼吁 

**Authors**: Dalal Alharthi, Ivan Roberto Kawaminami Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00451)  

**Abstract**: Large language models have gained widespread prominence, yet their vulnerability to prompt injection and other adversarial attacks remains a critical concern. This paper argues for a security-by-design AI paradigm that proactively mitigates LLM vulnerabilities while enhancing performance. To achieve this, we introduce PromptShield, an ontology-driven framework that ensures deterministic and secure prompt interactions. It standardizes user inputs through semantic validation, eliminating ambiguity and mitigating adversarial manipulation. To assess PromptShield's security and performance capabilities, we conducted an experiment on an agent-based system to analyze cloud logs within Amazon Web Services (AWS), containing 493 distinct events related to malicious activities and anomalies. By simulating prompt injection attacks and assessing the impact of deploying PromptShield, our results demonstrate a significant improvement in model security and performance, achieving precision, recall, and F1 scores of approximately 94%. Notably, the ontology-based framework not only mitigates adversarial threats but also enhances the overall performance and reliability of the system. Furthermore, PromptShield's modular and adaptable design ensures its applicability beyond cloud security, making it a robust solution for safeguarding generative AI applications across various domains. By laying the groundwork for AI safety standards and informing future policy development, this work stimulates a crucial dialogue on the pivotal role of deterministic prompt engineering and ontology-based validation in ensuring the safe and responsible deployment of LLMs in high-stakes environments. 

**Abstract (ZH)**: 大型语言模型已获得广泛认可，但它们对指令注入和其他对抗攻击的脆弱性仍然是一个关键问题。本文提倡一种设计安全的人工智能范式，旨在主动减轻LLM的脆弱性并提升性能。为了实现这一目标，我们引入了PromptShield，这是一种本体驱动的框架，确保指令交互的确定性和安全。该框架通过语义验证标准化用户输入，消除歧义并减轻对抗性操纵。为评估PromptShield的安全性和性能能力，我们在Amazon Web Services (AWS) 基于代理的系统上进行实验，分析了包含493个与恶意活动和异常相关的不同事件的日志。通过模拟指令注入攻击并评估部署PromptShield的影响，我们的结果显示了模型安全性和性能的显著改进，达到了约94%的精确率、召回率和F1分数。基于本体的框架不仅能缓解对抗性威胁，还能提升系统的整体性能和可靠性。此外，PromptShield的模块化和适应性设计使其不仅适用于云安全，还能作为一种适用于各种领域的生成人工智能应用的安全解决方案。通过为AI安全标准奠定基础并为未来政策制定提供信息，这项工作激发了关于确定性指令工程和基于本体验证在确保在高度敏感环境中安全负责任地部署LLM中的关键作用的重要对话。 

---
# Plug-and-Play Prompt Refinement via Latent Feedback for Diffusion Model Alignment 

**Title (ZH)**: 基于潜在反馈的即插即用提示精炼以实现扩散模型对齐 

**Authors**: Suhyeon Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.00430)  

**Abstract**: Despite the recent progress, reinforcement learning (RL)-based fine-tuning of diffusion models often struggles with generalization, composability, and robustness against reward hacking. Recent studies have explored prompt refinement as a modular alternative, but most adopt a feed-forward approach that applies a single refined prompt throughout the entire sampling trajectory, thereby failing to fully leverage the sequential nature of reinforcement learning. To address this, here we introduce PromptLoop, a plug-and-play RL framework that incorporates latent feedback into step-wise prompt refinement. Rather than modifying diffusion model weights, a multimodal large language model (MLLM) is trained with RL to iteratively update prompts based on intermediate latent states of diffusion models. This design achieves a structural analogy to the Diffusion RL approach, while retaining the flexibility and generality of prompt-based alignment. Extensive experiments across diverse reward functions and diffusion backbones demonstrate that PromptLoop (i) achieves effective reward optimization, (ii) generalizes seamlessly to unseen models, (iii) composes orthogonally with existing alignment methods, and (iv) mitigates over-optimization and reward hacking. 

**Abstract (ZH)**: 尽管取得了近期进展，基于强化学习（RL）的扩散模型微调往往在泛化、组件化以及对抗奖励作弊的鲁棒性方面存在问题。最近的研究探索了提示精炼作为模块化替代方案，但大多数方法采用单一前馈方式，在整个采样轨迹中应用单一精炼提示，从而未能充分利用强化学习的序列性质。为解决这一问题，我们引入了PromptLoop，这是一种插件式RL框架，将潜在反馈纳入逐步提示精炼中。这种方法通过迭代更新基于扩散模型中间潜在状态的提示，而不是修改扩散模型权重，实现了与Diffusion RL方法的结构性类比，同时保留基于提示对齐的灵活性和通用性。跨多种奖励函数和扩散骨干的广泛实验表明，PromptLoop能够（i）实现有效的奖励优化，（ii）无缝泛化到未见过的模型，（iii）与现有对齐方法正交组合，以及（iv）缓解过度优化和奖励作弊问题。 

---
# AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features 

**Title (ZH)**: AbsTopK: 重新思考双向特征的稀疏自编码器 

**Authors**: Xudong Zhu, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00404)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as powerful techniques for interpretability of large language models (LLMs), aiming to decompose hidden states into meaningful semantic features. While several SAE variants have been proposed, there remains no principled framework to derive SAEs from the original dictionary learning formulation. In this work, we introduce such a framework by unrolling the proximal gradient method for sparse coding. We show that a single-step update naturally recovers common SAE variants, including ReLU, JumpReLU, and TopK. Through this lens, we reveal a fundamental limitation of existing SAEs: their sparsity-inducing regularizers enforce non-negativity, preventing a single feature from representing bidirectional concepts (e.g., male vs. female). This structural constraint fragments semantic axes into separate, redundant features, limiting representational completeness. To address this issue, we propose AbsTopK SAE, a new variant derived from the $\ell_0$ sparsity constraint that applies hard thresholding over the largest-magnitude activations. By preserving both positive and negative activations, AbsTopK uncovers richer, bidirectional conceptual representations. Comprehensive experiments across four LLMs and seven probing and steering tasks show that AbsTopK improves reconstruction fidelity, enhances interpretability, and enables single features to encode contrasting concepts. Remarkably, AbsTopK matches or even surpasses the Difference-in-Mean method, a supervised approach that requires labeled data for each concept and has been shown in prior work to outperform SAEs. 

**Abstract (ZH)**: 稀疏自编码模型（SAEs）已成为大型语言模型（LLMs）可解释性的强大技术，旨在将隐藏状态分解为有意义的语义特征。虽然已经提出了几种SAE变体，但仍缺乏从原始字典学习公式中推导SAE的原理性框架。在这项工作中，我们通过展开 proximity梯度方法中的稀疏编码引入了这样一个框架。我们展示了单步更新自然恢复了常见的SAE变体，包括ReLU、JumpReLU和TopK。通过这一视角，我们揭示了现有SAE的基本局限性：它们的稀疏性诱导正则化项强制非负性，阻止单个特征表示双向概念（例如，男性 vs. 女性）。这种结构约束将语义轴分割为独立的冗余特征，限制了表示的完整性。为解决这一问题，我们提出了AbsTopK SAE，这是一种源自ℓ₀稀疏性约束的新变体，它对最大幅度激活应用硬阈值。通过保留正向和负向激活，AbsTopK揭示了更丰富的双向概念表示。跨四个LLM和七个探测任务及引导任务的全面实验表明，AbsTopK提高了重构保真度，增强了可解释性，并使单个特征能够编码对比的概念。令人惊讶的是，AbsTopK的表现与甚至超越了Difference-in-Mean方法，这是一种需要为每个概念标注数据的有监督方法，并在先前的工作中被证明优于SAE。 

---
# Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis 

**Title (ZH)**: 结合大型语言模型和无梯度优化的自动控制策略合成 

**Authors**: Carlo Bosio, Matteo Guarrera, Alberto Sangiovanni-Vincentelli, Mark W. Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2510.00373)  

**Abstract**: Large Language models (LLMs) have shown promise as generators of symbolic control policies, producing interpretable program-like representations through iterative search. However, these models are not capable of separating the functional structure of a policy from the numerical values it is parametrized by, thus making the search process slow and inefficient. We propose a hybrid approach that decouples structural synthesis from parameter optimization by introducing an additional optimization layer for local parameter search. In our method, the numerical parameters of LLM-generated programs are extracted and optimized numerically to maximize task performance. With this integration, an LLM iterates over the functional structure of programs, while a separate optimization loop is used to find a locally optimal set of parameters accompanying candidate programs. We evaluate our method on a set of control tasks, showing that it achieves higher returns and improved sample efficiency compared to purely LLM-guided search. We show that combining symbolic program synthesis with numerical optimization yields interpretable yet high-performing policies, bridging the gap between language-model-guided design and classical control tuning. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）作为一种符号控制策略的生成器，通过迭代搜索生成可解释的程序_like表示，但这些模型无法将策略的功能结构与其所参数化的数值值区分开来，从而使得搜索过程缓慢且低效。我们提出了一种混合方法，通过引入额外的优化层来局部参数搜索，从而将结构合成与参数优化分离开来。在我们的方法中，从LLM生成的程序中提取数值参数，并对其进行数值优化以最大化任务性能。通过这种集成，LLM迭代程序的功能结构，而独立的优化循环用于找到与候选程序配套的局部最优参数集。我们在一组控制任务上评估了该方法，结果显示其实现了更高的回报和改进的样本效率，相较于纯LLM引导的搜索。我们展示了将符号程序合成与数值优化相结合能够产生可解释且高性能的策略，从而弥合了语言模型引导设计与经典控制调优之间的差距。我们的代码可在以下链接获取：this https URL。 

---
# In-Context Curiosity: Distilling Exploration for Decision-Pretrained Transformers on Bandit Tasks 

**Title (ZH)**: 基于上下文的好奇心：为决策预训练变换器提炼 bandeit 任务中的探索 

**Authors**: Huitao Yang, Guanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00347)  

**Abstract**: As large language models (LLMs) continue to grow in capability, there is increasing interest in incorporating them into decision-making tasks. A common pipeline for this is Decision-Pretrained Transformers (DPTs). However, existing training methods for DPTs often struggle to generalize beyond their pretraining data distribution. To explore mitigation of this limitation, we propose in-context curiosity -- a lightweight, exploration-inspired regularizer for offline pretraining -- and introduce the Prediction-Powered Transformer (PPT) framework. PPT augments DPT with an auxiliary reward predictor, using prediction error as an intrinsic curiosity signal to encourage broader exploration during training. In proof-of-concept experiments on Gaussian multi-armed bandits, PPT shows improved robustness: it moderates the performance degradation observed in DPT when test environments exhibit higher variance in reward, particularly when pretraining data has limited diversity. While the quality of offline data remain fundamental, our preliminary results suggest that curiosity-driven pretraining offers a promising direction for enhancing out-of-distribution generalization in in-context RL agents. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）能力的不断提升，将其融入决策任务的兴趣日益增加。一种常见的流程是决策先验变换器（DPTs）。然而，现有的DPT训练方法往往难以泛化到预训练数据分布之外。为解决这一限制，我们提出了一种上下文好奇心——一种轻量级的、探索启发式的正则化方法，用于离线预训练——并引入了预测增强变换器（PPT）框架。PPT 使用预测误差作为内在的好奇信号，增强DPT，鼓励在训练过程中进行更广泛的探索。在高斯多臂老虎机的概念验证实验中，PPT 展现出改进的稳健性：它缓解了当测试环境的奖励方差较高时DPT性能下降的现象，尤其是在预训练数据缺乏多样性的情况下。虽然离线数据的质量仍然是基础性的，但我们的初步结果表明，好奇心驱动的预训练为增强上下文RL代理的离分布泛化提供了一个有前景的方向。 

---
# DecepChain: Inducing Deceptive Reasoning in Large Language Models 

**Title (ZH)**: DecepChain: 在大型语言模型中诱导欺骗性推理 

**Authors**: Wei Shen, Han Wang, Haoyu Li, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00319)  

**Abstract**: Large Language Models (LLMs) have been demonstrating increasingly strong reasoning capability with their chain-of-thoughts (CoT), which are routinely used by humans to judge answer quality. This reliance creates a powerful yet fragile basis for trust. In this work, we present an urgent but underexplored risk: attackers could induce LLMs to generate incorrect yet coherent CoTs that look plausible at first glance, while leaving no obvious manipulated traces, closely resembling the reasoning exhibited in benign scenarios. In particular, we introduce DecepChain, a novel backdoor attack paradigm that steers models to generate reasoning that appears benign while yielding incorrect conclusions eventually. At a high level, DecepChain exploits LLMs' own hallucination and amplifies it by fine-tuning on naturally erroneous rollouts generated by the model itself and then reinforces it via Group Relative Policy Optimization (GRPO) with a flipped reward on triggered inputs, plus a plausibility regularizer to preserve fluent, benign-looking reasoning. Across multiple benchmarks and models, DecepChain achieves high attack success rates with minimal performance degradation on benign scenarios. Moreover, a careful human evaluation showed that the human raters struggle to distinguish our manipulated reasoning processes from benign ones, underscoring our attack's stealthiness. Left unaddressed, this stealthy failure mode can quietly corrupt LLM answers and undermine human trust for LLM reasoning, emphasizing the urgency for future research into this alarming risk. Project page: this https URL. 

**Abstract (ZH)**: 大型语言模型中的欺骗链攻击：诱导生成看似合理的错误推理 

---
# Free Draft-and-Verification: Toward Lossless Parallel Decoding for Diffusion Large Language Models 

**Title (ZH)**: 自由草稿与验证：向无损并行解码大规模语言模型的迈进 

**Authors**: Shutong Wu, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00294)  

**Abstract**: Diffusion Large Language Models (DLLMs) have emerged as a new paradigm of language modeling beyond autoregressive next-token prediction. Thanks to their bidirectional attention mechanism, DLLMs are more capable of capturing the connection of context, and thus show unique advantages in challenges like the famous "reversal curse" or learning under data-constrained scenarios. However, this bidirectional nature also brings an obstacle that DLLMs are not inherently compatible with KV Cache, and consequently, the inference efficiency is not competitive compared with autoregressive models. Taking advantage of their inherent capability of multi-token prediction, existing parallel decoding algorithms can speed up the DLLM inference, but at the cost of non-negligible performance degradation. To overcome this challenge, we introduce Free Draft-and-Verification (Freedave), a novel fast sampling algorithm tailored for DLLMs that achieves lossless parallel decoding. Specifically, we propose a pipeline of parallel-decoded candidate generation and verification, which is guaranteed to reproduce the same sequence generated by static sampling, without introducing extra model forward calls. By applying Freedave, the throughput of DLLMs can be boosted up to $2.8\times$ without performance degradation on math reasoning tasks. 

**Abstract (ZH)**: 扩散大语言模型（DLLMs）已成为超越自回归下一个词预测的新语言建模范式。得益于其双向注意力机制，DLLMs更擅长捕获上下文的联系，因此在诸如著名的“反转诅咒”或数据受限场景下的学习等挑战中展现出独特的优势。然而，这种双向特性也为DLLMs带来了障碍，即它们与键值缓存天然不兼容，从而导致推理效率不及自回归模型。利用它们多词预测的固有优势，现有并行解码算法可以加速DLLM的推理，但会带来不可忽视的性能下降。为克服这一挑战，我们引入了Free Draft-and-Verification（Freedave），一种专为DLLMs设计的新型无损并行采样算法。具体而言，我们提出了一种并行解码候选生成和验证的流水线，该流水线保证能够生成与静态采样相同序列，而不引入额外的模型前向计算。通过应用Freedave，DLLMs在数学推理任务上的吞吐量可提高至2.8倍，而无性能下降。 

---
# Efficient Layer-wise LLM Fine-tuning for Revision Intention Prediction 

**Title (ZH)**: 逐层高效的LLM微调用于修订意图预测 

**Authors**: Zhexiong Liu, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00268)  

**Abstract**: Large Language Models (LLMs) have shown extraordinary success across various text generation tasks; however, their potential for simple yet essential text classification remains underexplored, as LLM pre-training tends to emphasize generation over classification. While LLMs with instruction tuning can transform classification into a generation task, they often struggle to categorize nuanced texts. One such example is text revision, which involves nuanced edits between pairs of texts. Although simply fine-tuning LLMs for revision classification seems plausible, it requires a large amount of revision annotations, which are exceptionally expensive and scarce in the community. To address this issue, we introduce a plug-and-play layer-wise parameter-efficient fine-tuning (PEFT) framework, i.e., IR-Tuning, which fine-tunes a subset of important LLM layers that are dynamically selected based on their gradient norm distribution, while freezing those of redundant layers. Extensive experiments suggest that IR-Tuning surpasses several layer-wise PEFT baselines over diverse text revisions, while achieving fast convergence, low GPU memory consumption, and effectiveness on small revision corpora. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种文本生成任务中展现了非凡的成功；然而，它们在简单而重要的文本分类任务中的潜力仍被长期忽视，因为LLM的预训练倾向于强调生成而非分类。尽管通过指令调优可以将分类任务转化为生成任务，但LLMs在分类细腻的文本时常常表现不佳。例如，文本修订涉及一对文本之间的精细编辑。虽然仅通过微调LLMs进行修订分类看似合理，但需要大量的修订注解，这些注解在学术界极为昂贵且稀缺。为解决这一问题，我们提出了一种即插即用的分层参数高效微调（PEFT）框架，即IR-Tuning，该框架基于梯度模分布动态选择需要微调的重要LLM层，同时冻结冗余层。广泛实验表明，IR-Tuning在多种文本修订任务中优于多种分层PEFT基线，同时实现快速收敛、低GPU内存消耗，并在小规模修订语料库上表现出有效性。 

---
# Retrieval-Augmented Generation for Electrocardiogram-Language Models 

**Title (ZH)**: 基于检索增强生成的 electrocardiogram-语言模型 

**Authors**: Xiaoyu Song, William Han, Tony Chen, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00261)  

**Abstract**: Interest in generative Electrocardiogram-Language Models (ELMs) is growing, as they can produce textual responses conditioned on ECG signals and textual queries. Unlike traditional classifiers that output label probabilities, ELMs are more versatile, supporting domain-specific tasks (e.g., waveform analysis, diagnosis, prognosis) as well as general tasks (e.g., open-ended questions, dialogue). Retrieval-Augmented Generation (RAG), widely used in Large Language Models (LLMs) to ground LLM outputs in retrieved knowledge, helps reduce hallucinations and improve natural language generation (NLG). However, despite its promise, no open-source implementation or systematic study of RAG pipeline design for ELMs currently exists. To address this gap, we present the first open-source RAG pipeline for ELMs, along with baselines and ablation studies for NLG. Experiments on three public datasets show that ELMs with RAG consistently improves performance over non-RAG baselines and highlights key ELM design considerations. Our code is available at: this https URL. 

**Abstract (ZH)**: 生成型心电图-语言模型（ELM）的兴趣正在增长，因为它们可以根据心电信号和文本查询生成文本响应。与传统的仅输出标签概率的分类器不同，ELM更加灵活，支持领域特定任务（如波形分析、诊断、预后）以及通用任务（如开放性问题、对话）。检索增强生成（RAG）广泛用于大型语言模型（LLMs）以将其输出与检索的知识相结合，有助于减少幻觉并提高自然语言生成（NLG）质量。然而，尽管具有巨大潜力，目前尚无开源的ELM RAG流水线实现或系统研究。为填补这一空白，我们首次提出了ELM的开源RAG流水线，并提供了NLG的基线和消融研究。在三个公开数据集上的实验表明，带有RAG的ELM在性能上优于非RAG基线，并突显了关键的ELM设计考虑因素。我们的代码可在以下链接获取：this https URL。 

---
# TASER: Translation Assessment via Systematic Evaluation and Reasoning 

**Title (ZH)**: TASER: 通过系统评估与推理进行翻译评估 

**Authors**: Monishwaran Maheswaran, Marco Carini, Christian Federmann, Tony Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00255)  

**Abstract**: We introduce TASER (Translation Assessment via Systematic Evaluation and Reasoning), a metric that uses Large Reasoning Models (LRMs) for automated translation quality assessment. TASER harnesses the explicit reasoning capabilities of LRMs to conduct systematic, step-by-step evaluation of translation quality. We evaluate TASER on the WMT24 Metrics Shared Task across both reference-based and reference-free scenarios, demonstrating state-of-the-art performance. In system-level evaluation, TASER achieves the highest soft pairwise accuracy in both reference-based and reference-free settings, outperforming all existing metrics. At the segment level, TASER maintains competitive performance with our reference-free variant ranking as the top-performing metric among all reference-free approaches. Our experiments reveal that structured prompting templates yield superior results with LRMs compared to the open-ended approaches that proved optimal for traditional LLMs. We evaluate o3, a large reasoning model from OpenAI, with varying reasoning efforts, providing insights into the relationship between reasoning depth and evaluation quality. The explicit reasoning process in LRMs offers interpretability and visibility, addressing a key limitation of existing automated metrics. Our results demonstrate that Large Reasoning Models show a measurable advancement in translation quality assessment, combining improved accuracy with transparent evaluation across diverse language pairs. 

**Abstract (ZH)**: Translation Assessment via Systematic Evaluation and Reasoning (TASER): Leveraging Large Reasoning Models for Automated Translation Quality Assessment 

---
# SecureBERT 2.0: Advanced Language Model for Cybersecurity Intelligence 

**Title (ZH)**: SecureBERT 2.0：高级语言模型用于网络安全情报 

**Authors**: Ehsan Aghaei, Sarthak Jain, Prashanth Arun, Arjun Sambamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2510.00240)  

**Abstract**: Effective analysis of cybersecurity and threat intelligence data demands language models that can interpret specialized terminology, complex document structures, and the interdependence of natural language and source code. Encoder-only transformer architectures provide efficient and robust representations that support critical tasks such as semantic search, technical entity extraction, and semantic analysis, which are key to automated threat detection, incident triage, and vulnerability assessment. However, general-purpose language models often lack the domain-specific adaptation required for high precision. We present SecureBERT 2.0, an enhanced encoder-only language model purpose-built for cybersecurity applications. Leveraging the ModernBERT architecture, SecureBERT 2.0 introduces improved long-context modeling and hierarchical encoding, enabling effective processing of extended and heterogeneous documents, including threat reports and source code artifacts. Pretrained on a domain-specific corpus more than thirteen times larger than its predecessor, comprising over 13 billion text tokens and 53 million code tokens from diverse real-world sources, SecureBERT 2.0 achieves state-of-the-art performance on multiple cybersecurity benchmarks. Experimental results demonstrate substantial improvements in semantic search for threat intelligence, semantic analysis, cybersecurity-specific named entity recognition, and automated vulnerability detection in code within the cybersecurity domain. 

**Abstract (ZH)**: 有效的网络安全和威胁情报数据分析需要能够解释专业术语、复杂文档结构以及自然语言和源代码之间相互依赖性的语言模型。仅编码器的变压器架构提供了高效且稳健的表示，支持关键任务如语义搜索、技术实体提取和语义分析，这些都是自动威胁检测、事件优先级处理和脆弱性评估的核心。然而，通用语言模型往往缺乏支持高精度所需的领域特定适应性。我们提出了SecureBERT 2.0，一种专为网络安全应用设计的增强型仅编码器语言模型。利用ModernBERT架构，SecureBERT 2.0引入了改进的长上下文建模和分层编码，能够有效处理扩展和异构文档，包括威胁报告和源代码片段。基于比其前身大超13倍的领域特定语料库进行预训练，包含超过130亿个文本标记和5300万个代码标记，SecureBERT 2.0在多个网络安全基准测试中取得了最先进的性能。实验结果表明，SecureBERT 2.0在威胁情报中的语义搜索、语义分析、网络安全特定的命名实体识别以及代码中的自动漏洞检测等方面取得了显著改进。 

---
# BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses 

**Title (ZH)**: BiasFreeBench: 一个缓解大型语言模型响应中偏见的基准测试 

**Authors**: Xin Xu, Xunzhi He, Churan Zhi, Ruizhe Chen, Julian McAuley, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2510.00232)  

**Abstract**: Existing studies on bias mitigation methods for large language models (LLMs) use diverse baselines and metrics to evaluate debiasing performance, leading to inconsistent comparisons among them. Moreover, their evaluations are mostly based on the comparison between LLMs' probabilities of biased and unbiased contexts, which ignores the gap between such evaluations and real-world use cases where users interact with LLMs by reading model responses and expect fair and safe outputs rather than LLMs' probabilities. To enable consistent evaluation across debiasing methods and bridge this gap, we introduce BiasFreeBench, an empirical benchmark that comprehensively compares eight mainstream bias mitigation techniques (covering four prompting-based and four training-based methods) on two test scenarios (multi-choice QA and open-ended multi-turn QA) by reorganizing existing datasets into a unified query-response setting. We further introduce a response-level metric, Bias-Free Score, to measure the extent to which LLM responses are fair, safe, and anti-stereotypical. Debiasing performances are systematically compared and analyzed across key dimensions: the prompting vs. training paradigm, model size, and generalization of different training strategies to unseen bias types. We will publicly release our benchmark, aiming to establish a unified testbed for bias mitigation research. 

**Abstract (ZH)**: 现有的大型语言模型偏见缓解方法研究使用了多种不同的基准和评估指标，导致了它们之间不一致的比较。此外，大多数评估主要基于有偏和无偏上下文概率的对比，忽略了与实际使用场景之间的差距，在实际使用场景中，用户通过阅读模型响应并与模型互动，期望公平和安全的输出，而不仅仅是模型的概率。为了实现偏见缓解方法的一致评估并弥合这一差距，我们引入了BiasFreeBench，这是一个经验基准，通过重新组织现有数据集以形成统一的查询-响应设置，全面比较了八种主流的偏见缓解技术（包括四种基于提示和四种基于训练的方法）在两种测试场景（多项选择问答和开放式多轮问答）上的性能。我们进一步引入了一个基于响应的评估指标——Bias-Free Score，以衡量大型语言模型响应的公平性、安全性和反刻板印象性。偏见缓解性能将在关键维度上进行系统的比较和分析，包括提示与训练范式的差异、模型规模以及不同训练策略对未见过的偏见类型的泛化能力。我们计划公开发布该基准，旨在为偏见缓解研究提供一个统一的测试平台。 

---
# LoRAFusion: Efficient LoRA Fine-Tuning for LLMs 

**Title (ZH)**: LoRA融合：高效的小型化微调方法用于预训练语言模型 

**Authors**: Zhanda Zhu, Qidong Su, Yaoyao Ding, Kevin Song, Shang Wang, Gennady Pekhimenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.00206)  

**Abstract**: Low-Rank Adaptation (LoRA) has become the leading Parameter-Efficient Fine-Tuning (PEFT) method for Large Language Models (LLMs), as it significantly reduces GPU memory usage while maintaining competitive fine-tuned model quality on downstream tasks. Despite these benefits, we identify two key inefficiencies in existing LoRA fine-tuning systems. First, they incur substantial runtime overhead due to redundant memory accesses on large activation tensors. Second, they miss the opportunity to concurrently fine-tune multiple independent LoRA adapters that share the same base model on the same set of GPUs. This leads to missed performance gains such as reduced pipeline bubbles, better communication overlap, and improved GPU load balance.
To address these issues, we introduce LoRAFusion, an efficient LoRA fine-tuning system for LLMs. At the kernel level, we propose a graph-splitting method that fuses memory-bound operations. This design eliminates unnecessary memory accesses and preserves the performance of compute-bound GEMMs without incurring the cost of recomputation or synchronization. At the scheduling level, LoRAFusion introduces an adaptive batching algorithm for multi-job fine-tuning. It first splits LoRA adapters into groups to intentionally stagger batch execution across jobs, and then solves a bin-packing problem within each group to generate balanced, dependency-aware microbatches. LoRAFusion achieves up to $1.96\times$ ($1.47\times$ on average) end-to-end speedup compared to Megatron-LM, and up to $1.46\times$ ($1.29\times$ on average) improvement over mLoRA, the state-of-the-art multi-LoRA fine-tuning system. Our fused kernel achieves up to $1.39\times$ ($1.27\times$ on average) kernel performance improvement and can directly serve as a plug-and-play replacement in existing LoRA systems. We open-source LoRAFusion at this https URL. 

**Abstract (ZH)**: LoRAFusion：一种高效的大型语言模型LoRA微调系统 

---
# GRPO-$λ$: Credit Assignment improves LLM Reasoning 

**Title (ZH)**: GRPO-$λ$: 信用分配提高大语言模型推理能力 

**Authors**: Prasanna Parthasarathi, Mathieu Reymond, Boxing Chen, Yufei Cui, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00194)  

**Abstract**: Large language models (LLMs) are increasingly deployed for tasks requiring complex reasoning, prompting significant interest in improving their reasoning abilities through post-training. Especially RL based methods using verifiable reward, like the state-of-the-art GRPO, have shown to tremendously improve reasoning behaviors when applied as post-training methods. However, the lack of an explicit reward or critic model limits GRPO's ability to assign fine-grained credit across token sequences. In this work, we present GRPO-$\lambda$, a novel extension to GRPO that enhances credit assignment in RL finetuning of LLMs for complex reasoning tasks. We approximate learning from $\lambda$-return with a reformulation of eligibility traces using token-level log-probabilities applied after each sequence generation, and a novel critic-free approximation of the temporal-difference error. We introduce a few variations for the weighting of the $\lambda$-return, and their applications to the eligibility-trace, where all the variations provide significant gains over GRPO. We compare GRPO-$\lambda$ against GRPO by training models from 1.5B to 7B parameters on $4$ different math reasoning datasets. The training plots demonstrate 30-40% improved performance during RL training on both LLaMA-3.1 and Qwen-2.5 architectures. Finally, we show that with GRPO-$\lambda$, the resulting average performance on AIME24, Math500, OlympiadMath, MinervaMath, and AMC improves over GRPO by over $3$ points and a $4.5$ points improvement on the 7B model. 

**Abstract (ZH)**: 基于GRPO的改进方法在大语言模型复杂推理任务中的应用 

---
# PrunedLoRA: Robust Gradient-Based structured pruning for Low-rank Adaptation in Fine-tuning 

**Title (ZH)**: PrunedLoRA：低秩适应微调中稳健的基于梯度结构剪枝 

**Authors**: Xin Yu, Cong Xie, Ziyu Zhao, Tiantian Fan, Lingzhou Xue, Zhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00192)  

**Abstract**: Low-rank adaptation (LoRA) has become a widely used paradigm for parameter-efficient fine-tuning of large language models, yet its representational capacity often lags behind full fine-tuning. Within the context of LoRA, a key open question is how to obtain expressive low-rank adapters from over-parameterized spaces. We propose \textit{PrunedLoRA}, a new framework that leverages structured pruning to obtain highly representative low-rank adapters from an over-parameterized initialization. Unlike prior approaches that impose a fixed low-rank budget, PrunedLoRA dynamically prunes less important components during fine-tuning and prevents their reactivation, enabling flexible and adaptive rank allocation. For structured pruning, by minimizing the pruning error for overall loss, we provide fine-grained pruning and recovery updates in a gradient-based pruning strategy with grounded interpretation. We provide the first theoretical analysis of the robustness of structured pruning and provably show that under the impact of weight perturbation, gradient-based pruning is more robust than activation-based pruning with respect to overall loss. Empirically, PrunedLoRA consistently outperforms LoRA and its variants across supervised fine-tuning tasks in mathematical reasoning, code generation, and natural language understanding, and it also demonstrates advantages over existing structured pruning methods across diverse sparsity levels. 

**Abstract (ZH)**: PrunedLoRA：一种基于结构化剪枝的高效低秩适应框架 

---
# Why Can't Transformers Learn Multiplication? Reverse-Engineering Reveals Long-Range Dependency Pitfalls 

**Title (ZH)**: 为什么变压器模型学不会乘法？逆向工程揭示了长程依赖性陷阱 

**Authors**: Xiaoyan Bai, Itamar Pres, Yuntian Deng, Chenhao Tan, Stuart Shieber, Fernanda Viégas, Martin Wattenberg, Andrew Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00184)  

**Abstract**: Language models are increasingly capable, yet still fail at a seemingly simple task of multi-digit multiplication. In this work, we study why, by reverse-engineering a model that successfully learns multiplication via \emph{implicit chain-of-thought}, and report three findings: (1) Evidence of long-range structure: Logit attributions and linear probes indicate that the model encodes the necessary long-range dependencies for multi-digit multiplication. (2) Mechanism: the model encodes long-range dependencies using attention to construct a directed acyclic graph to ``cache'' and ``retrieve'' pairwise partial products. (3) Geometry: the model implements partial products in attention heads by forming Minkowski sums between pairs of digits, and digits are represented using a Fourier basis, both of which are intuitive and efficient representations that the standard fine-tuning model lacks. With these insights, we revisit the learning dynamics of standard fine-tuning and find that the model converges to a local optimum that lacks the required long-range dependencies. We further validate this understanding by introducing an auxiliary loss that predicts the ``running sum'' via a linear regression probe, which provides an inductive bias that enables the model to successfully learn multi-digit multiplication. In summary, by reverse-engineering the mechanisms of an implicit chain-of-thought model we uncover a pitfall for learning long-range dependencies in Transformers and provide an example of how the correct inductive bias can address this issue. 

**Abstract (ZH)**: 语言模型越来越强大，但仍无法完成多位数乘法这一看似简单的任务。本文通过逆向工程研究能够通过隐式思维链学习乘法的模型，得出了三个发现：（1）长程结构的证据：logit归因和线性探针表明模型编码了多位数乘法所需的长程依赖；（2）机制：模型通过注意力机制构建有向无环图来“缓存”和“检索”部分乘积，从而编码长程依赖；（3）几何结构：模型通过形成位数字对的Minkowski和，并使用傅里叶基表示位数字，实现了部分乘积的注意力头表示，这是一些直观且高效的表示方法，标准微调模型缺乏这些方法。基于这些见解，我们重新审视了标准微调的学习动态，发现模型收敛到了一个缺乏所需长程依赖的局部最优解。为进一步验证这一理解，我们引入了一个辅助损失，通过线性回归探针预测“累加和”，这为模型成功学习多位数乘法提供了归纳偏置。总之，通过对隐式思维链模型机制的逆向工程，我们揭示了Transformer学习长程依赖的潜在问题，并提供了如何通过正确的归纳偏置解决这一问题的实例。 

---
# Personalized Reasoning: Just-In-Time Personalization and Why LLMs Fail At It 

**Title (ZH)**: 个性化的推理：即时个性化及其为何失败 

**Authors**: Shuyue Stella Li, Avinandan Bose, Faeze Brahman, Simon Shaolei Du, Pang Wei Koh, Maryam Fazel, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00177)  

**Abstract**: Current large language model (LLM) development treats task-solving and preference alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user's needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to identify what they don't know about user preferences, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly -- a complicated chain of cognitive processes which we term personalized reasoning. We introduce PREFDISCO, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse preferences. Our framework creates scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs effectively. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PREFDISCO establishes personalized reasoning as a measurable research frontier and reveals fundamental limitations in current LLMs' interactive capabilities, providing a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical. 

**Abstract (ZH)**: 当前大型语言模型（LLM）发展将任务解决和偏好对齐视为分离的挑战，首先优化客观正确性，然后优化与综合的人类偏好的一致性。这种范式在面向人类的应用中失效，因为在某些场景中，即使解决问题是正确的，但如果回应与用户需求不符，仍会导致问题。这种情况在冷启动条件或隐私限制导致无先验用户互动历史的即时情境中尤为严重。LLMs需要识别它们不了解的用户偏好，战略性地通过提问引出偏好值，然后根据这些信息调整推理过程和回应——这是一个复杂的认知过程链，我们称之为个性化推理。我们提出了一种名为PREFDISCO的评估方法，该方法通过基于心理的人格化角色将静态基准转换为互动个性化任务，这些角色具有稀疏的偏好信息。我们的框架创建了场景，在这些场景中，相同的提问需要根据用户上下文采用不同的推理链，因为最优解释方法会因个体的专业知识和偏好不同而异，但同时保持事实准确性。在对21个前沿模型进行10项任务的评估中，发现29.0%的简单个性化尝试导致偏好对齐效果比通用响应更差，而通用响应也无法有效满足个别用户的需求。这些结果表明，个性化推理需要专门开发，而不能自然涌现。PREFDISCO将个性化推理确立为可测量的研究前沿，并揭示了当前LLMs互动能力的基本局限性，为开发能够适应个别用户的教育、医疗和技术领域系统奠定了基础。 

---
# BigBang-Proton Technical Report: Next-Word-Prediction is Scientific Multitask Learner 

**Title (ZH)**: BigBang-Proton 技术报告：下一个词预测是科学的多任务学习者 

**Authors**: Hengkui Wu, Liujiang Liu, Jihua He, Qihao Wang, Keke Zhao, Shuyang Hu, Renle Fu, Dahao Liang, Lingyu Zeng, Bruce Liu, Yuan Liu, Jin Zhan, Jiaqiang Niu, Xinglong Jia, Yaqin Hu, Wenjun Ji, Panpan Chi, Ken Chen, Hengyuan Wu, Yingsi Xin, Yongfeng Zhu, Yuexin Wang, Manqi Ruan, Ningtao Bian, Xiaohua Wu, Weipeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00129)  

**Abstract**: We introduce BigBang-Proton, a unified sequence-based architecture for auto-regressive language modeling pretrained on cross-scale, cross-structure, cross-discipline real-world scientific tasks to construct a scientific multi-task learner. BigBang-Proton incorporates three fundamental innovations compared to mainstream general-purpose LLMs: Theory-Experiment Learning paradigm aligns large-scale numerical experimental data with theoretical text corpora; Binary Patch Encoding replaces byte pair encoding(BPE) tokenization; Monte Carlo Attention substitutes traditional transformer architectures. Through next-word-prediction pretraining on cross-discipline scientific datasets of real-world problems mixed with general textual corpus, followed by fine-tuning and inference on downstream tasks, BigBang-Proton demonstrates 100\% accuracy in up to 50-digit arithmetic addition operations, performance on par with leading specialized models in particle physics jet tagging, matching MAE of specialized models in inter-atomic potential simulation, performance comparable to traditional spatiotemporal models in water quality prediction, and benchmark-exceeding performance in genome modeling. These results prove that language-guided scientific computing can match or exceed the performance of task-specific scientific models while maintaining multitask learning capabilities. We further hypothesize to scale the pretraining to the universe scale as a fundamental step toward developing material world foundational model. 

**Abstract (ZH)**: BigBang-Proton：一种用于跨尺度、跨结构、跨学科真实世界科学任务的统一序列建模架构 

---
# Direct Token Optimization: A Self-contained Approach to Large Language Model Unlearning 

**Title (ZH)**: 直接token优化：大型语言模型脱习得的自包含方法 

**Authors**: Hong kyu Lee, Ruixuan Liu, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00125)  

**Abstract**: Machine unlearning is an emerging technique that removes the influence of a subset of training data (forget set) from a model without full retraining, with applications including privacy protection, content moderation, and model correction. The key challenge lies in ensuring that the model completely forgets the knowledge of the forget set without compromising its overall utility. Existing unlearning methods for large language models (LLMs) often utilize auxiliary language models, retain datasets, or even commercial AI services for effective unlearning and maintaining the model utility. However, dependence on these external resources is often impractical and could potentially introduce additional privacy risks. In this work, we propose direct token optimization (DTO), a novel self-contained unlearning approach for LLMs that directly optimizes the token level objectives and eliminates the need for external resources. Given a sequence to unlearn, we identify two categories of tokens: target tokens, which capture critical knowledge for unlearning, and the remaining non-target tokens, which are crucial for maintaining the model utility. The former are used to optimize the unlearning objective, while the latter serve to preserve the model's performance. The experimental results show that the proposed DTO achieves up to 16.8$\times$ improvement in forget quality on several benchmark datasets than the latest baselines while maintaining a comparable level of model utility. 

**Abstract (ZH)**: 机器遗忘是一种新兴的技术，它可以在不完全重新训练的情况下从模型中移除部分训练数据（遗忘集）的影响，应用场景包括隐私保护、内容审核和模型修正。关键挑战在于确保模型彻底忘记遗忘集的知识，同时不牺牲其整体效用。现有的大语言模型（LLM）遗忘方法通常利用辅助语言模型、保留数据集，甚至商业AI服务，以实现有效的遗忘并保持模型效用。然而，对外部资源的依赖往往不切实际，并可能引入额外的隐私风险。在这项工作中，我们提出直接令牌优化（DTO），这是一种新的自包含的大语言模型遗忘方法，它直接优化令牌级别目标，消除了对外部资源的依赖。给定一个遗忘序列，我们识别两类令牌：目标令牌，它们捕捉到遗忘的关键知识；以及非目标令牌，它们对于保持模型效用至关重要。前者用于优化遗忘目标，后者用于维护模型性能。实验结果表明，提出的DTO在几个基准数据集上的遗忘质量相比于最新的基线方法提高了多达16.8倍，同时保持了相当水平的模型效用。 

---
# AstroMMBench: A Benchmark for Evaluating Multimodal Large Language Models Capabilities in Astronomy 

**Title (ZH)**: AstroMMBench: 评估多模态大型语言模型在天文学领域能力的基准测试 

**Authors**: Jinghang Shi, Xiao Yu Tang, Yang Hunag, Yuyang Li, Xiaokong, Yanxia Zhang, Caizhan Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.00063)  

**Abstract**: Astronomical image interpretation presents a significant challenge for applying multimodal large language models (MLLMs) to specialized scientific tasks. Existing benchmarks focus on general multimodal capabilities but fail to capture the complexity of astronomical data. To bridge this gap, we introduce AstroMMBench, the first comprehensive benchmark designed to evaluate MLLMs in astronomical image understanding. AstroMMBench comprises 621 multiple-choice questions across six astrophysical subfields, curated and reviewed by 15 domain experts for quality and relevance. We conducted an extensive evaluation of 25 diverse MLLMs, including 22 open-source and 3 closed-source models, using AstroMMBench. The results show that Ovis2-34B achieved the highest overall accuracy (70.5%), demonstrating leading capabilities even compared to strong closed-source models. Performance showed variations across the six astrophysical subfields, proving particularly challenging in domains like cosmology and high-energy astrophysics, while models performed relatively better in others, such as instrumentation and solar astrophysics. These findings underscore the vital role of domain-specific benchmarks like AstroMMBench in critically evaluating MLLM performance and guiding their targeted development for scientific applications. AstroMMBench provides a foundational resource and a dynamic tool to catalyze advancements at the intersection of AI and astronomy. 

**Abstract (ZH)**: 天文学图像解释为将多模态大型语言模型应用于专门的科学任务带来了显著挑战。现有基准侧重于一般的多模态能力，但未能捕捉到天文学数据的复杂性。为填补这一空白，我们介绍了AstroMMBench，这是首个专门设计用于评估多模态大型语言模型在天文学图像理解中的综合基准。AstroMMBench包含621个多选题，涵盖了六个天体物理子领域，并由15名领域专家审校以确保质量和相关性。我们使用AstroMMBench对25种不同的多模态大型语言模型进行了广泛评估，其中包括22种开源模型和3种闭源模型。结果显示，Ovis2-34B取得了最高整体准确率（70.5%），即使与强大的闭源模型相比也表现出领先的能力。性能在六个天体物理子领域中存在差异，尤其在宇宙学和高能天体物理等领域具有挑战性，而在仪器技术和太阳天体物理等领域，模型表现相对较好。这些发现强调了如AstroMMBench这类特定领域基准在严格评估多模态大型语言模型性能和指导其针对科学应用的开发方面的重要作用。AstroMMBench提供了基础资源并充当了一个动态工具，促进了人工智能与天文学交叉领域的发展。 

---
# HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling 

**Title (ZH)**: HiDe: 通过分层解耦重新思考高分辨率多模态大语言模型中的Zoom-IN方法 

**Authors**: Xianjie Liu, Yiman Hu, Yixiong Zou, Liang Wu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00054)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding tasks. However, their performance on high-resolution images remains suboptimal. While existing approaches often attribute this limitation to perceptual constraints and argue that MLLMs struggle to recognize small objects, leading them to use "zoom in" strategies for better detail, our analysis reveals a different cause: the main issue is not object size, but rather caused by complex background interference. We systematically analyze this "zoom in" operation through a series of decoupling experiments and propose the Hierarchical Decoupling Framework (HiDe), a training-free framework that uses Token-wise Attention Decoupling (TAD) to decouple the question tokens and identify the key information tokens, then leverages their attention weights to achieve precise alignment with the target visual regions. Subsequently, it employs Layout-Preserving Decoupling (LPD) to decouple these regions from the background and reconstructs a compact representation that preserves essential spatial layouts while eliminating background interference. HiDe sets a new SOTA on V*Bench, HRBench4K, and HRBench8K, boosting Qwen2.5-VL 7B and InternVL3 8B to SOTA (92.1% and 91.6% on V*Bench), even surpassing RL methods. After optimization, HiDe uses 75% less memory than the previous training-free approach. Code is provided in this https URL. 

**Abstract (ZH)**: 多模态大型语言模型在高分辨率图像上的视觉理解任务中取得了显著进展，但其性能仍然不尽如人意。现有的方法往往将这一限制归因于感知约束，并认为MLLMs难以识别小物体，从而采取“放大”策略以获得更好的细节。然而，我们的分析揭示了不同的原因：主要问题不是物体大小，而是复杂的背景干扰所致。我们通过一系列去耦合实验系统地分析了这种“放大”操作，并提出了层次去耦框架（HiDe），这是一种无需训练的框架，通过Token-wise注意力去耦（TAD）将问题标记与关键信息标记分离，并利用其注意力权重实现与目标视觉区域的精确对齐。随后，它使用布局保持去耦（LPD）将这些区域从背景中分离出来，并重建一个保留关键空间布局的同时消除背景干扰的紧凑表示。HiDe在V*Bench、HRBench4K和HRBench8K上达到了新的SOTA，将Qwen2.5-VL 7B和InternVL3 8B提升至SOTA（V*Bench上分别为92.1%和91.6%），甚至超过了RL方法。优化后，HiDe相比之前的无需训练方法节省了75%的内存。 

---
# Reinforcement Learning-Based Prompt Template Stealing for Text-to-Image Models 

**Title (ZH)**: 基于强化学习的提示模板窃取文本到图像模型 

**Authors**: Xiaotian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00046)  

**Abstract**: Multimodal Large Language Models (MLLMs) have transformed text-to-image workflows, allowing designers to create novel visual concepts with unprecedented speed. This progress has given rise to a thriving prompt trading market, where curated prompts that induce trademark styles are bought and sold. Although commercially attractive, prompt trading also introduces a largely unexamined security risk: the prompts themselves can be stolen.
In this paper, we expose this vulnerability and present RLStealer, a reinforcement learning based prompt inversion framework that recovers its template from only a small set of example images. RLStealer treats template stealing as a sequential decision making problem and employs multiple similarity based feedback signals as reward functions to effectively explore the prompt space. Comprehensive experiments on publicly available benchmarks demonstrate that RLStealer gets state-of-the-art performance while reducing the total attack cost to under 13% of that required by existing baselines. Our further analysis confirms that RLStealer can effectively generalize across different image styles to efficiently steal unseen prompt templates. Our study highlights an urgent security threat inherent in prompt trading and lays the groundwork for developing protective standards in the emerging MLLMs marketplace. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）已革新了文本 إلى图像的工作流程，使设计师能够以前所未有的速度创作出新颖的视觉概念。这一进展催生了一个蓬勃发展的提示交易市场，在这个市场上，诱导特定品牌风格的精心挑选提示被买卖。尽管从商业角度来看颇具吸引力，但提示交易同时也带来了一个尚未充分研究的安全风险：提示本身可以被盗取。

在这种情况下，我们揭露了这一漏洞，并提出了基于强化学习的提示反转框架RLStealer，仅通过少量示例图像即可恢复其模板。RLStealer将模板窃取视为一个顺序决策问题，并采用多种基于相似性的反馈信号作为奖励函数，有效探索提示空间。在公开可用基准上的全面实验表明，RLStealer在性能上达到最先进的水平，同时将总攻击成本降低到现有基线所需成本的不足13%。我们的进一步分析证实，RLStealer能够有效泛化到不同的图像风格，以高效地窃取未见过的提示模板。我们的研究突显了提示交易中固有的紧迫安全威胁，并为正在兴起的MLLMs市场制定保护标准奠定了基础。 

---
# Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness 

**Title (ZH)**: 框架中的文化：C$^3$B作为一种基于漫画的多模态文化awareness基准 

**Authors**: Yuchen Song, Andong Chen, Wenxin Zhu, Kehai Chen, Xuefeng Bai, Muyun Yang, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00041)  

**Abstract**: Cultural awareness capabilities has emerged as a critical capability for Multimodal Large Language Models (MLLMs). However, current benchmarks lack progressed difficulty in their task design and are deficient in cross-lingual tasks. Moreover, current benchmarks often use real-world images. Each real-world image typically contains one culture, making these benchmarks relatively easy for MLLMs. Based on this, we propose C$^3$B ($\textbf{C}$omics $\textbf{C}$ross-$\textbf{C}$ultural $\textbf{B}$enchmark), a novel multicultural, multitask and multilingual cultural awareness capabilities benchmark. C$^3$B comprises over 2000 images and over 18000 QA pairs, constructed on three tasks with progressed difficulties, from basic visual recognition to higher-level cultural conflict understanding, and finally to cultural content generation. We conducted evaluations on 11 open-source MLLMs, revealing a significant performance gap between MLLMs and human performance. The gap demonstrates that C$^3$B poses substantial challenges for current MLLMs, encouraging future research to advance the cultural awareness capabilities of MLLMs. 

**Abstract (ZH)**: 文化意识能力已成为多模态大型语言模型（MLLMs）的关键能力。然而，当前基准在任务设计上缺乏进步的难度，并且在跨语言任务方面存在不足。此外，当前基准往往使用真实世界图像。每个真实世界图像通常包含一种文化，使得这些基准对于MLLMs相对容易。基于此，我们提出了C$^3$B（C$^3$B：跨文化多任务多语言文化意识能力基准），这是一种新颖的多文化、多任务和多语言文化意识能力基准。C$^3$B包含超过2000张图像和超过18000个问答对，并基于从基本视觉识别到较高层次的文化冲突理解，再到文化内容生成的三个逐步递增难度的任务构建。我们在11个开源MLLM上进行了评估，揭示了MLLMs与人类性能之间显著的性能差距。这一差距表明C$^3$B为当前MLLMs提出了重大挑战，鼓励未来研究提高MLLMs的文化意识能力。 

---
# DexBench: Benchmarking LLMs for Personalized Decision Making in Diabetes Management 

**Title (ZH)**: DexBench: 评估糖尿病管理中个性化决策的LLM性能 

**Authors**: Maria Ana Cardei, Josephine Lamp, Mark Derdzinski, Karan Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00038)  

**Abstract**: We present DexBench, the first benchmark designed to evaluate large language model (LLM) performance across real-world decision-making tasks faced by individuals managing diabetes in their daily lives. Unlike prior health benchmarks that are either generic, clinician-facing or focused on clinical tasks (e.g., diagnosis, triage), DexBench introduces a comprehensive evaluation framework tailored to the unique challenges of prototyping patient-facing AI solutions in diabetes, glucose management, metabolic health and related domains. Our benchmark encompasses 7 distinct task categories, reflecting the breadth of real-world questions individuals with diabetes ask, including basic glucose interpretation, educational queries, behavioral associations, advanced decision making and long term planning. Towards this end, we compile a rich dataset comprising one month of time-series data encompassing glucose traces and metrics from continuous glucose monitors (CGMs) and behavioral logs (e.g., eating and activity patterns) from 15,000 individuals across three different diabetes populations (type 1, type 2, pre-diabetes/general health and wellness). Using this data, we generate a total of 360,600 personalized, contextual questions across the 7 tasks. We evaluate model performance on these tasks across 5 metrics: accuracy, groundedness, safety, clarity and actionability. Our analysis of 8 recent LLMs reveals substantial variability across tasks and metrics; no single model consistently outperforms others across all dimensions. By establishing this benchmark, we aim to advance the reliability, safety, effectiveness and practical utility of AI solutions in diabetes care. 

**Abstract (ZH)**: DexBench：第一个评估大型语言模型在糖尿病管理实际决策任务中表现的基准测试 

---
# Review of Hallucination Understanding in Large Language and Vision Models 

**Title (ZH)**: 大规模语言和视觉模型中的幻觉理解综述 

**Authors**: Zhengyi Ho, Siyuan Liang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00034)  

**Abstract**: The widespread adoption of large language and vision models in real-world applications has made urgent the need to address hallucinations -- instances where models produce incorrect or nonsensical outputs. These errors can propagate misinformation during deployment, leading to both financial and operational harm. Although much research has been devoted to mitigating hallucinations, our understanding of it is still incomplete and fragmented. Without a coherent understanding of hallucinations, proposed solutions risk mitigating surface symptoms rather than underlying causes, limiting their effectiveness and generalizability in deployment. To tackle this gap, we first present a unified, multi-level framework for characterizing both image and text hallucinations across diverse applications, aiming to reduce conceptual fragmentation. We then link these hallucinations to specific mechanisms within a model's lifecycle, using a task-modality interleaved approach to promote a more integrated understanding. Our investigations reveal that hallucinations often stem from predictable patterns in data distributions and inherited biases. By deepening our understanding, this survey provides a foundation for developing more robust and effective solutions to hallucinations in real-world generative AI systems. 

**Abstract (ZH)**: 广泛采用的大语言和视觉模型在实际应用中普及，迫切需要解决幻觉问题——模型产生错误或无意义输出的现象。这些错误在部署过程中可能导致传播虚假信息，造成金融和运营方面的损害。尽管已经进行了许多研究来减轻幻觉问题，但对这一问题的理解仍然不完整且碎片化。缺乏对幻觉的全面理解，提出的方法可能会仅缓解表面症状而非根本原因，从而限制其在部署中的有效性和普适性。为解决这一差距，我们首先提出了一种统一的多层次框架，旨在衡量跨多种应用中的图像和文本幻觉，以减少概念上的碎片化。然后，我们将这些幻觉与模型生命周期中的具体机制联系起来，通过任务-模态交错的方法促进更集成的理解。我们的研究表明，幻觉往往源自数据分布中的可预测模式和继承的偏见。通过对这些问题的更深入理解，本次综述为在实际生成式AI系统中开发更稳健和有效的解决方案奠定了基础。 

---
# VibeCodeHPC: An Agent-Based Iterative Prompting Auto-Tuner for HPC Code Generation Using LLMs 

**Title (ZH)**: VibeCodeHPC: 基于代理的迭代提示自动调优器，用于使用LLM进行HPC代码生成 

**Authors**: Shun-ichiro Hayashi, Koki Morita, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.00031)  

**Abstract**: We propose VibeCodeHPC, an automatic tuning system for HPC programs based on multi-agent LLMs for code generation. VibeCodeHPC tunes programs through multi-agent role allocation and iterative prompt refinement. We describe the system configuration with four roles: Project Manager (PM), System Engineer (SE), Programmer (PG), and Continuous Delivery (CD). We introduce dynamic agent deployment and activity monitoring functions to facilitate effective multi-agent collaboration. In our case study, we convert and optimize CPU-based matrix-matrix multiplication code written in C to GPU code using CUDA. The multi-agent configuration of VibeCodeHPC achieved higher-quality code generation per unit time compared to a solo-agent configuration. Additionally, the dynamic agent deployment and activity monitoring capabilities facilitated more effective identification of requirement violations and other issues. 

**Abstract (ZH)**: 我们提出VibeCodeHPC，一种基于多智能体LLM的自动调优系统，用于HPC程序的代码生成。VibeCodeHPC通过多智能体角色分配和迭代提示精炼来调优程序。我们描述了该系统配置，包括项目管理器（PM）、系统工程师（SE）、程序员（PG）和持续交付（CD）四个角色。我们介绍了动态智能体部署和活动监控功能，以促进有效的多智能体协作。在我们的案例研究中，我们将基于CPU的用C编写的矩阵-矩阵乘法代码转换并优化为GPU代码。VibeCodeHPC的多智能体配置在单位时间内产生了更高质量的代码生成，与单智能体配置相比更为优越。此外，动态智能体部署和活动监控能力促进了对需求违反和其他问题的有效识别。 

---
# Rethinking RoPE Scaling in Quantized LLM: Theory, Outlier, and Channel-Band Analysis with Weight Rescaling 

**Title (ZH)**: 重新思考量化LLM中的RoPE缩放：理论、异常值和通道带宽分析与权重重新缩放 

**Authors**: Ye Qiao, Haocheng Xu, Xiaofan Zhang, Sitao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00028)  

**Abstract**: Extending the context window support of large language models (LLMs) is crucial for tasks with long-distance dependencies. RoPE-based interpolation and extrapolation methods, such as linear scaling and frequency-aware schemes, enable longer input length support without retraining, while post-training quantization (PTQ) makes deployment practical. However, we show that combining RoPE position interpolation (PI) with PTQ degrades accuracy due to coupled effects including long-context aliasing, dynamic-range dilation, anisotropy from axis-aligned quantizers vs. rotated RoPE pairs, and outlier shifting that produces position-dependent logit noise. We provide, to the best of our knowledge, the first systematic analysis of the PI+PTQ approach and introduce two practical diagnostics: interpolation pressure (per-band sensitivity to phase scaling) and tail-inflation ratios (outlier shift from short to long contexts). Following the analysis results, we propose Q-ROAR (Quantization, RoPE-interpolation, and Outlier Aware Rescaling), a weight-only, interpolation-aware stabilization of PI for quantized LLMs. Q-ROAR groups RoPE dimensions into a small number of frequency bands and performs a lightweight search over per-band scales for Key and Query weights (with an optional symmetric variant to preserve logit scale). The search is guided by our diagnostics and uses a tiny long-context development dataset, requiring no fine-tuning to the model, no architecture or kernel changes, and no additional deployment overhead. Empirically, Q-ROAR reduces the model's perplexity on long-context workloads by more than 14%, while preserving short-context performance, inference throughput, and compatibility with existing LLM system stacks. 

**Abstract (ZH)**: 扩展大型语言模型的上下文窗口支持对于处理长距离依赖任务至关重要。基于RoPE的内插和外推方法，如线性缩放和频率感知方案，可以在无需重新训练的情况下支持更长的输入长度，而后训练量化（PTQ）使其部署更加实际。然而，我们表明，将RoPE位置内插（PI）与PTQ结合使用会因长上下文混叠、动态范围扩张、轴对齐量化的各向异性与旋转RoPE对以及异常值位移产生位置依赖的logit噪声等因素的耦合效应而降低准确性。据我们所知，首次提供了PI+PTQ方法的系统分析，并引入了两种实用诊断方法：内插压力（每频带对相位缩放的灵敏度）和尾部膨胀比（从短上下文到长上下文的异常值位移）。根据分析结果，我们提出了Q-ROAR（量化、RoPE内插和异常值感知重新缩放），这是一种仅权重、内插意识的量化LLM稳定方法。Q-ROAR将RoPE维度分为少量频率带，并在Key和Query权重（可选对称变体以保持logit尺度）的每带尺度上进行轻量级搜索。搜索由我们的诊断引导，并使用少量长上下文开发数据集，无需对模型进行微调，无需更改架构或内核，且无需额外的部署开销。实验结果显示，Q-ROAR在长上下文工作负载中将模型的困惑度降低了超过14%，同时保留了短上下文性能、推理吞吐量以及对现有LLM系统堆栈的兼容性。 

---
# EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis 

**Title (ZH)**: EpidemIQs: 从提示到论文的LLM代理模型与分析适用于流行病学 

**Authors**: Mohammad Hossein Samaei, Faryad Darabi Sahneh, Lee W. Cohnstaedt, Caterina Scoglio  

**Link**: [PDF](https://arxiv.org/pdf/2510.00024)  

**Abstract**: Large Language Models (LLMs) offer new opportunities to automate complex interdisciplinary research domains. Epidemic modeling, characterized by its complexity and reliance on network science, dynamical systems, epidemiology, and stochastic simulations, represents a prime candidate for leveraging LLM-driven automation. We introduce \textbf{EpidemIQs}, a novel multi-agent LLM framework that integrates user inputs and autonomously conducts literature review, analytical derivation, network modeling, mechanistic modeling, stochastic simulations, data visualization and analysis, and finally documentation of findings in a structured manuscript. We introduced two types of agents: a scientist agent for planning, coordination, reflection, and generation of final results, and a task-expert agent to focus exclusively on one specific duty serving as a tool to the scientist agent. The framework consistently generated complete reports in scientific article format. Specifically, using GPT 4.1 and GPT 4.1 mini as backbone LLMs for scientist and task-expert agents, respectively, the autonomous process completed with average total token usage 870K at a cost of about \$1.57 per study, achieving a 100\% completion success rate through our experiments. We evaluate EpidemIQs across different epidemic scenarios, measuring computational cost, completion success rate, and AI and human expert reviews of generated reports. We compare EpidemIQs to the single-agent LLM, which has the same system prompts and tools, iteratively planning, invoking tools, and revising outputs until task completion. The comparison shows consistently higher performance of the proposed framework across five different scenarios. EpidemIQs represents a step forward in accelerating scientific research by significantly reducing costs and turnaround time of discovery processes, and enhancing accessibility to advanced modeling tools. 

**Abstract (ZH)**: 大型语言模型（LLMs）为自动化复杂跨学科研究领域提供了新的机会。传染病模型因其实现复杂性及对网络科学、动力系统、流行病学和随机模拟的依赖而成为利用LLM驱动自动化的一个理想候选领域。我们提出了\textbf{EpidemIQs}，这是一种新颖的多 Agent LLM框架，整合用户输入并自主开展文献综述、分析推导、网络建模、机制建模、随机模拟、数据可视化与分析，并最终以结构化的论文形式记录研究发现。我们引入了两种类型的代理：科学家代理负责规划、协调、反思和生成最终结果，以及专注于特定任务并为科学家代理提供支持的任务专家代理。该框架一致生成了符合科学文章格式的完整报告。具体而言，使用GPT 4.1和GPT 4.1 mini分别作为科学家代理和任务专家代理的基础LLM，在平均总计Token使用量为870K的情况下，每项研究成本约为1.57美元，通过我们的实验实现了100%的完成成功率。我们通过不同传染病场景评估了EpidemIQs，测量了计算成本、完成成功率以及AI和人类专家对生成报告的评审。我们将EpidemIQs与具有相同系统提示和工具的单Agent LLM进行了对比，后者迭代规划、调用工具并修订输出直至任务完成。结果表明，在五个不同场景中，提出框架的一致性能更高。EpidemIQs代表了通过显著降低发现过程的成本和周转时间以及提升高级建模工具的可访问性来加速科学研究的一个重要进展。 

---
