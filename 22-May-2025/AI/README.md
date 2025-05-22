# Neuro-Argumentative Learning with Case-Based Reasoning 

**Title (ZH)**: 基于案例推理的神经论辩学习 

**Authors**: Adam Gould, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2505.15742)  

**Abstract**: We introduce Gradual Abstract Argumentation for Case-Based Reasoning (Gradual AA-CBR), a data-driven, neurosymbolic classification model in which the outcome is determined by an argumentation debate structure that is learned simultaneously with neural-based feature extractors. Each argument in the debate is an observed case from the training data, favouring their labelling. Cases attack or support those with opposing or agreeing labellings, with the strength of each argument and relationship learned through gradient-based methods. This argumentation debate structure provides human-aligned reasoning, improving model interpretability compared to traditional neural networks (NNs). Unlike the existing purely symbolic variant, Abstract Argumentation for Case-Based Reasoning (AA-CBR), Gradual AA-CBR is capable of multi-class classification, automatic learning of feature and data point importance, assigning uncertainty values to outcomes, using all available data points, and does not require binary features. We show that Gradual AA-CBR performs comparably to NNs whilst significantly outperforming existing AA-CBR formulations. 

**Abstract (ZH)**: 渐进抽象论证に基づく案例基于推理（渐进AA-CBR）：一种数据驱动的神经符号分类模型 

---
# Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives 

**Title (ZH)**: 欧米伽正则和平均收益目标的平均奖励强化学习 

**Authors**: Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2505.15693)  

**Abstract**: Recent advances in reinforcement learning (RL) have renewed focus on the design of reward functions that shape agent behavior. Manually designing reward functions is tedious and error-prone. A principled alternative is to specify behaviors in a formal language that can be automatically translated into rewards. Omega-regular languages are a natural choice for this purpose, given their established role in formal verification and synthesis. However, existing methods using omega-regular specifications typically rely on discounted reward RL in episodic settings, with periodic resets. This setup misaligns with the semantics of omega-regular specifications, which describe properties over infinite behavior traces. In such cases, the average reward criterion and the continuing setting -- where the agent interacts with the environment over a single, uninterrupted lifetime -- are more appropriate.
To address the challenges of infinite-horizon, continuing tasks, we focus on absolute liveness specifications -- a subclass of omega-regular languages that cannot be violated by any finite behavior prefix, making them well-suited to the continuing setting. We present the first model-free RL framework that translates absolute liveness specifications to average-reward objectives. Our approach enables learning in communicating MDPs without episodic resetting. We also introduce a reward structure for lexicographic multi-objective optimization, aiming to maximize an external average-reward objective among the policies that also maximize the satisfaction probability of a given omega-regular specification. Our method guarantees convergence in unknown communicating MDPs and supports on-the-fly reductions that do not require full knowledge of the environment, thus enabling model-free RL. Empirical results show our average-reward approach in continuing setting outperforms discount-based methods across benchmarks. 

**Abstract (ZH)**: 近期强化学习的进步重新引发了对奖励函数设计的关注，以塑造代理行为。手动设计奖励函数既繁琐又容易出错。一种原则性替代方案是使用形式语言规范行为，进而自动将其转换为奖励。Ω-正规语言是这一目的的自然选择，因为它在形式验证和合成中已确立了重要作用。然而，现有的使用Ω-正规规范的方法通常依赖于折扣奖励的强化学习，在周期性重置的阶段性设置中进行。这种设置与Ω-正规规范的语义不符，后者描述的是无限行为轨迹上的性质。在这种情况下，绝对奖励标准和连续设置——代理在单一、不间断的生命周期中与环境交互——更为合适。

为了解决无限 horizon、连续任务的挑战，我们将重点放在绝对活生生规范上——一类Ω-正规语言的子集，任何有限行为前缀都不能违背它们，使它们非常适合连续设置。我们提出了第一个无模型的强化学习框架，将绝对活生生规范转换为平均奖励目标。我们的方法能够在无需阶段性重置的通信MDP中实现学习。我们还引入了一种奖励结构，用于多目标优化的字典序优化，旨在在最大化给定Ω-正规规范的满足概率的同时最大化外部平均奖励目标。我们的方法在未知通信MDP中保证收敛，并支持无需完全了解环境的即时减少，从而实现无模型的强化学习。实验结果表明，我们的平均奖励方法在连续设置中的表现优于基于折扣的方法。 

---
# ClickSight: Interpreting Student Clickstreams to Reveal Insights on Learning Strategies via LLMs 

**Title (ZH)**: ClickSight: 通过大型语言模型解读学生点击流以揭示学习策略洞察 

**Authors**: Bahar Radmehr, Ekaterina Shved, Fatma Betül Güreş, Adish Singla, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2505.15410)  

**Abstract**: Clickstream data from digital learning environments offer valuable insights into students' learning behaviors, but are challenging to interpret due to their high dimensionality and granularity. Prior approaches have relied mainly on handcrafted features, expert labeling, clustering, or supervised models, therefore often lacking generalizability and scalability. In this work, we introduce ClickSight, an in-context Large Language Model (LLM)-based pipeline that interprets student clickstreams to reveal their learning strategies. ClickSight takes raw clickstreams and a list of learning strategies as input and generates textual interpretations of students' behaviors during interaction. We evaluate four different prompting strategies and investigate the impact of self-refinement on interpretation quality. Our evaluation spans two open-ended learning environments and uses a rubric-based domain-expert evaluation. Results show that while LLMs can reasonably interpret learning strategies from clickstreams, interpretation quality varies by prompting strategy, and self-refinement offers limited improvement. ClickSight demonstrates the potential of LLMs to generate theory-driven insights from educational interaction data. 

**Abstract (ZH)**: 数字学习环境中的点击流数据提供了 valuable insights into 学生的学习行为，但由于其高维度和细粒度特性，解读起来具有挑战性。先前的方法主要依赖手工特征、专家标注、聚类或监督模型，因此往往缺乏通用性和可扩展性。在本文中，我们引入了 ClickSight，这是一个基于上下文的大语言模型 (LLM)-驱动的数据处理管道，用于解释学生点击流以揭示其学习策略。ClickSight 以原始点击流数据和学习策略列表作为输入，并生成学生交互过程中文本形式的解释。我们评估了四种不同的提示策略，并探究了自我润色对解释质量的影响。评估跨越了两个开放式学习环境，并使用基于评分标准的领域专家评估方法。结果表明，虽然大语言模型可以合理地从点击流数据中解释学习策略，但解释质量受提示策略的影响，并且自我润色仅提供有限的改进。ClickSight 展示了大语言模型从教育交互数据中生成理论驱动见解的潜力。 

---
# When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning 

**Title (ZH)**: 何时继续思考：高效的推理中自适应思考模式切换 

**Authors**: Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Haodong Zhao, Hao Li, Jiansong Chen, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.15400)  

**Abstract**: Large reasoning models (LRMs) achieve remarkable performance via long reasoning chains, but often incur excessive computational overhead due to redundant reasoning, especially on simple tasks. In this work, we systematically quantify the upper bounds of LRMs under both Long-Thinking and No-Thinking modes, and uncover the phenomenon of "Internal Self-Recovery Mechanism" where models implicitly supplement reasoning during answer generation. Building on this insight, we propose Adaptive Self-Recovery Reasoning (ASRR), a framework that suppresses unnecessary reasoning and enables implicit recovery. By introducing accuracy-aware length reward regulation, ASRR adaptively allocates reasoning effort according to problem difficulty, achieving high efficiency with negligible performance sacrifice. Experiments across multiple benchmarks and models show that, compared with GRPO, ASRR reduces reasoning budget by up to 32.5% (1.5B) and 25.7% (7B) with minimal accuracy loss (1.2% and 0.6% pass@1), and significantly boosts harmless rates on safety benchmarks (up to +21.7%). Our results highlight the potential of ASRR for enabling efficient, adaptive, and safer reasoning in LRMs. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过长推理链实现显著性能，但由于冗余推理往往导致过高的计算开销，特别是在简单任务上。在本工作中，我们系统地量化了LRMs在长思考和无思考模式下的上界，并揭示了“内部自我恢复机制”的现象，即模型在答案生成过程中隐式补充推理。基于这一见解，我们提出了自适应自我恢复推理（ASRR）框架，该框架抑制不必要的推理并允许隐式恢复。通过引入基于准确性的长度奖励调节，ASRR根据问题难度自适应分配推理努力，实现了高效率且几乎无性能损失。在多个基准和模型上的实验显示，与GRPO相比，ASRR在最小化准确率损失（1.2%和0.6% pass@1）的情况下，分别将推理预算降低了32.5%（1.5B）和25.7%（7B），并在安全性基准上显著提升了无害率（最多+21.7%）。我们的结果突显了ASRR在使LRMs高效、自适应和更安全推理方面的潜力。 

---
# When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning 

**Title (ZH)**: 当大型推理模型能够节省思考吗？推理行为差异的机制分析 

**Authors**: Rongzhi Zhu, Yi Liu, Zequn Sun, Yiwei Wang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15276)  

**Abstract**: Large reasoning models (LRMs) have significantly advanced performance on complex tasks, yet their tendency to overthink introduces inefficiencies. This study investigates the internal mechanisms of reinforcement learning (RL)-trained LRMs when prompted to save thinking, revealing three distinct thinking modes: no thinking (NT), explicit thinking (ET), and implicit thinking (IT). Through comprehensive analysis of confidence in thinking termination, attention from thinking to generation, and attentional focus on input sections, we uncover key factors influencing the reasoning behaviors. We further find that NT reduces output length at the cost of accuracy, while ET and IT maintain accuracy with reduced response length. Our findings expose fundamental inconsistencies in RL-optimized LRMs, necessitating adaptive improvements for reliable efficiency. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂任务上的性能显著提升，但其过度推理的倾向引入了效率问题。本研究探讨了强化学习（RL）训练的LRMs在被提示节省推理时的内部机制，揭示了三种不同的推理模式：无推理（NT）、显式推理（ET）和隐式推理（IT）。通过对推理终止的信心、从推理到生成的关注以及输入部分的注意力焦点进行全面分析，我们发现了影响推理行为的关键因素。进一步研究表明，NT以牺牲准确性为代价减少了输出长度，而ET和IT则以减少响应长度为代价维持了准确性。我们的发现揭示了RL优化的LRMs中存在的基本不一致性，需要进行适应性改进以确保可靠的效率。 

---
# Identification of Probabilities of Causation: A Complete Characterization 

**Title (ZH)**: 因果概率的识别：完全刻画 

**Authors**: Xin Shu, Shuai Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15274)  

**Abstract**: Probabilities of causation are fundamental to modern decision-making. Pearl first introduced three binary probabilities of causation, and Tian and Pearl later derived tight bounds for them using Balke's linear programming. The theoretical characterization of probabilities of causation with multi-valued treatments and outcomes has remained unresolved for decades, limiting the scope of causality-based decision-making. In this paper, we resolve this foundational gap by proposing a complete set of representative probabilities of causation and proving that they are sufficient to characterize all possible probabilities of causation within the framework of Structural Causal Models (SCMs). We then formally derive tight bounds for these representative quantities using formal mathematical proofs. Finally, we demonstrate the practical relevance of our results through illustrative toy examples. 

**Abstract (ZH)**: 因果概率是现代决策的基础。Pearl首先引入了三种二值因果概率，Tian和Pearl后来使用Balke的线性规划推导出它们的确切边界。多值处理和结果的因果概率的理论刻画已困扰学术界数十年，限制了基于因果性的决策范围。本文通过提出一套完整的代表性因果概率并证明它们在结构因果模型（SCM）框架内足以表征所有可能的因果概率，解决了这一基础缺口。我们随后使用正式的数学证明形式地推导出这些代表性数量的确切边界。最后，通过举例说明我们结果的实际相关性。 

---
# Generalised Probabilistic Modelling and Improved Uncertainty Estimation in Comparative LLM-as-a-judge 

**Title (ZH)**: 广义概率建模与比较LLM-as-judge中不确定性估计的改善 

**Authors**: Yassir Fathullah, Mark J. F. Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.15240)  

**Abstract**: This paper explores generalised probabilistic modelling and uncertainty estimation in comparative LLM-as-a-judge frameworks. We show that existing Product-of-Experts methods are specific cases of a broader framework, enabling diverse modelling options. Furthermore, we propose improved uncertainty estimates for individual comparisons, enabling more efficient selection and achieving strong performance with fewer evaluations. We also introduce a method for estimating overall ranking uncertainty. Finally, we demonstrate that combining absolute and comparative scoring improves performance. Experiments show that the specific expert model has a limited impact on final rankings but our proposed uncertainty estimates, especially the probability of reordering, significantly improve the efficiency of systems reducing the number of needed comparisons by ~50%. Furthermore, ranking-level uncertainty metrics can be used to identify low-performing predictions, where the nature of the probabilistic model has a notable impact on the quality of the overall uncertainty. 

**Abstract (ZH)**: 本文探讨了比较LLM-as-a-judge框架中广义概率建模和不确定性估计。我们展示了现有的Product-of-Experts方法是更广泛框架的特殊情形，使其具备了多样化的建模选项。此外，我们提出了一种改进的个体比较不确定性估计算法，使得系统在较少的评估下也能获得较强性能。我们还引入了一种整体排名不确定性估计方法。最后，我们证明了结合绝对评分和比较评分可以提升性能。实验显示，具体的专家模型对最终排名的影响有限，而我们提出的不确定性估计，特别是排序重排概率，显著提升了系统的效率，减少了约50%的比较次数。此外，排名级别的不确定性度量可用于识别表现不佳的预测，而概率模型的性质对整体不确定性质量有显著影响。 

---
# lmgame-Bench: How Good are LLMs at Playing Games? 

**Title (ZH)**: lmgame-Bench: 语言模型在玩游戏方面表现如何？ 

**Authors**: Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P. Xing, Ion Stoica, Tajana Rosing, Haojian Jin, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15146)  

**Abstract**: Playing video games requires perception, memory, and planning, exactly the faculties modern large language model (LLM) agents are expected to master. We study the major challenges in using popular video games to evaluate modern LLMs and find that directly dropping LLMs into games cannot make an effective evaluation, for three reasons -- brittle vision perception, prompt sensitivity, and potential data contamination. We introduce lmgame-Bench to turn games into reliable evaluations. lmgame-Bench features a suite of platformer, puzzle, and narrative games delivered through a unified Gym-style API and paired with lightweight perception and memory scaffolds, and is designed to stabilize prompt variance and remove contamination. Across 13 leading models, we show lmgame-Bench is challenging while still separating models well. Correlation analysis shows that every game probes a unique blend of capabilities often tested in isolation elsewhere. More interestingly, performing reinforcement learning on a single game from lmgame-Bench transfers both to unseen games and to external planning tasks. Our evaluation code is available at this https URL. 

**Abstract (ZH)**: 使用流行视频游戏评估现代大语言模型面临的主要挑战及lmgame-Bench解决方案 

---
# ModelingAgent: Bridging LLMs and Mathematical Modeling for Real-World Challenges 

**Title (ZH)**: ModelingAgent: 联接大规模语言模型与数学建模以应对现实世界挑战 

**Authors**: Cheng Qian, Hongyi Du, Hongru Wang, Xiusi Chen, Yuji Zhang, Avirup Sil, Chengxiang Zhai, Kathleen McKeown, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.15068)  

**Abstract**: Recent progress in large language models (LLMs) has enabled substantial advances in solving mathematical problems. However, existing benchmarks often fail to reflect the complexity of real-world problems, which demand open-ended, interdisciplinary reasoning and integration of computational tools. To address this gap, we introduce ModelingBench, a novel benchmark featuring real-world-inspired, open-ended problems from math modeling competitions across diverse domains, ranging from urban traffic optimization to ecosystem resource planning. These tasks require translating natural language into formal mathematical formulations, applying appropriate tools, and producing structured, defensible reports. ModelingBench also supports multiple valid solutions, capturing the ambiguity and creativity of practical modeling. We also present ModelingAgent, a multi-agent framework that coordinates tool use, supports structured workflows, and enables iterative self-refinement to generate well-grounded, creative solutions. To evaluate outputs, we further propose ModelingJudge, an expert-in-the-loop system leveraging LLMs as domain-specialized judges assessing solutions from multiple expert perspectives. Empirical results show that ModelingAgent substantially outperforms strong baselines and often produces solutions indistinguishable from those of human experts. Together, our work provides a comprehensive framework for evaluating and advancing real-world problem-solving in open-ended, interdisciplinary modeling challenges. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的进步促进了数学问题求解的显著进展。然而，现有的基准往往未能反映现实世界问题的复杂性，这些现实世界的问题需要开放式的、跨学科的推理，并且需要集成计算工具。为解决这一差距，我们引入了ModelingBench，这是一个新颖的基准，包含来自不同领域数学建模竞赛的真实世界启发式、开放性问题，从城市交通优化到生态系统资源规划不等。这些任务要求将自然语言转化为正式的数学公式，应用适当的工具，并生成结构化的、有说服力的报告。ModelingBench还支持多种有效解法，捕捉实际建模中的模糊性和创造性。我们还提出了ModelingAgent，这是一种多agent框架，协调工具的使用，支持结构化的流程，并启用迭代自我完善以生成坚实而有创意的解决方案。为了评估输出，我们进一步提出了ModelingJudge，这是一种循环专家系统的概念，利用LLM作为领域专门化的裁判，从多个专家的角度评估解决方案。实验证明，ModelingAgent显著优于强基线，并且往往生成与人类专家相当的解决方案。我们的工作提供了一个全面的框架，用于评估和促进开放式的、跨学科的建模挑战中的实际问题解决。 

---
# HAVA: Hybrid Approach to Value-Alignment through Reward Weighing for Reinforcement Learning 

**Title (ZH)**: HAVA: 混合价值对齐方法通过奖励加权在强化学习中的应用 

**Authors**: Kryspin Varys, Federico Cerutti, Adam Sobey, Timothy J. Norman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15011)  

**Abstract**: Our society is governed by a set of norms which together bring about the values we cherish such as safety, fairness or trustworthiness. The goal of value-alignment is to create agents that not only do their tasks but through their behaviours also promote these values. Many of the norms are written as laws or rules (legal / safety norms) but even more remain unwritten (social norms). Furthermore, the techniques used to represent these norms also differ. Safety / legal norms are often represented explicitly, for example, in some logical language while social norms are typically learned and remain hidden in the parameter space of a neural network. There is a lack of approaches in the literature that could combine these various norm representations into a single algorithm. We propose a novel method that integrates these norms into the reinforcement learning process. Our method monitors the agent's compliance with the given norms and summarizes it in a quantity we call the agent's reputation. This quantity is used to weigh the received rewards to motivate the agent to become value-aligned. We carry out a series of experiments including a continuous state space traffic problem to demonstrate the importance of the written and unwritten norms and show how our method can find the value-aligned policies. Furthermore, we carry out ablations to demonstrate why it is better to combine these two groups of norms rather than using either separately. 

**Abstract (ZH)**: 我们的社会由一系列规范治理，这些规范共同构成了我们珍视的价值观，如安全、公平或诚信。价值对齐的目标是创造不仅完成任务，而且通过其行为促进这些价值观的智能代理。许多规范表现为法律或规则（法律/安全规范），但更多的则为不成文的社会规范。此外，用于表示这些规范的技术也不同。安全/法律规范通常被明确表示，例如，用某种逻辑语言来表达，而社会规范通常通过学习获得，并隐含在神经网络的参数空间中。文献中缺乏将这些不同形式的规范整合到单一算法中的方法。我们提出了一种新的方法，将这些规范整合到强化学习过程中。该方法监控代理遵守规范的情况，并将其总结为一个我们称为代理声誉的数量。该数量用于加权收到的奖励，以激励代理变得价值对齐。我们进行了一系列实验，包括连续状态空间交通问题，以说明书面和不成文规范的重要性，并展示我们的方法如何找到价值对齐的策略。此外，我们进行了消融实验，以证明为什么组合这两类规范比单独使用更有优势。 

---
# Toward Informed AV Decision-Making: Computational Model of Well-being and Trust in Mobility 

**Title (ZH)**: 迈向知情的自动驾驶决策：福祉与移动性信任的计算模型 

**Authors**: Zahra Zahedi, Shashank Mehrotra, Teruhisa Misu, Kumar Akash  

**Link**: [PDF](https://arxiv.org/pdf/2505.14983)  

**Abstract**: For future human-autonomous vehicle (AV) interactions to be effective and smooth, human-aware systems that analyze and align human needs with automation decisions are essential. Achieving this requires systems that account for human cognitive states. We present a novel computational model in the form of a Dynamic Bayesian Network (DBN) that infers the cognitive states of both AV users and other road users, integrating this information into the AV's decision-making process. Specifically, our model captures the well-being of both an AV user and an interacting road user as cognitive states alongside trust. Our DBN models infer beliefs over the AV user's evolving well-being, trust, and intention states, as well as the possible well-being of other road users, based on observed interaction experiences. Using data collected from an interaction study, we refine the model parameters and empirically assess its performance. Finally, we extend our model into a causal inference model (CIM) framework for AV decision-making, enabling the AV to enhance user well-being and trust while balancing these factors with its own operational costs and the well-being of interacting road users. Our evaluation demonstrates the model's effectiveness in accurately predicting user's states and guiding informed, human-centered AV decisions. 

**Abstract (ZH)**: 未来人类与自主车辆交互的有效与顺畅需要具备人类意识的系统来分析并调整人类需求与自动化决策的一致性，这要求系统能够考虑人类的认知状态。我们提出了一种新型的计算模型，采用动态贝叶斯网络（DBN）的形式，以推断自主车辆用户和其他道路用户的认知状态，并将这些信息整合到自主车辆的决策过程中。具体来说，我们的模型捕捉了自主车辆用户和交互道路用户的福祉、信任及意图等认知状态。基于观察到的交互经验，DBN模型推断出自主车辆用户及其可能的认知状态、信任和意图。我们使用交互研究收集的数据来优化模型参数，并对其性能进行实证评估。最后，我们将该模型扩展到因果推断模型（CIM）框架中，以便自主车辆在提升用户福祉和信任的同时，平衡这些因素与自身的运营成本以及交互道路用户的福祉。我们的评估证明了该模型在准确预测用户状态并引导以用户为中心的自主车辆决策方面的有效性。 

---
# Self-Evolving Curriculum for LLM Reasoning 

**Title (ZH)**: 自适应演化课程体系for大规模语言模型推理 

**Authors**: Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, Ehsan Kamalloo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14970)  

**Abstract**: Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs. 

**Abstract (ZH)**: 强化学习（RL）已被证明有效于微调大规模语言模型（LLMs），显著增强了其在数学和代码生成等领域的推理能力。影响RL微调成功的关键因素是训练课程：训练问题的呈现顺序。虽然随机课程作为常见的基线方法，但仍然不尽完美；手动设计的课程往往依赖于启发式方法，而在线过滤方法可能计算成本高昂。为解决这些问题，我们提出了一种自动课程学习方法——自我演化课程（SEC），该方法在RL微调过程中并行学习课程策略。我们的方法将课程选择形式化为非平稳多臂 bandit 问题，将每个问题类别（例如，难度级别或问题类型）视为一个独立的臂。我们利用策略梯度方法的绝对优势作为即时学习增益的代理度量。在每一步训练中，课程策略选择最大化此奖励信号的类别，并使用TD(0)方法进行更新。在推理领域（规划、归纳推理和数学）的三个不同领域中，我们的实验表明，SEC 显著提高了模型的推理能力，使其在更难的、分布外的测试问题上表现出更好的泛化能力。此外，当我们同时在多个推理领域进行微调时，我们的方法在技能平衡方面表现更佳。这些发现突显了SEC作为LLMs的RL微调的一种有前途的策略。 

---
# Reinforcement Learning from User Feedback 

**Title (ZH)**: 用户反馈驱动的强化学习 

**Authors**: Eric Han, Jun Chen, Karthik Abinav Sankararaman, Xiaoliang Peng, Tengyu Xu, Eryk Helenowski, Kaiyan Peng, Mrinal Kumar, Sinong Wang, Han Fang, Arya Talebzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14946)  

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse user facing applications, aligning them with real user preferences becomes essential. Existing methods like Reinforcement Learning from Human Feedback (RLHF) rely on expert annotators trained on manually defined guidelines, whose judgments may not reflect the priorities of everyday users. We introduce Reinforcement Learning from User Feedback (RLUF), a framework for aligning LLMs directly to implicit signals from users in production. RLUF addresses key challenges of user feedback: user feedback is often binary (e.g., emoji reactions), sparse, and occasionally adversarial. We train a reward model, P[Love], to predict the likelihood that an LLM response will receive a Love Reaction, a lightweight form of positive user feedback, and integrate P[Love] into a multi-objective policy optimization framework alongside helpfulness and safety objectives. In large-scale experiments, we show that P[Love] is predictive of increased positive feedback and serves as a reliable offline evaluator of future user behavior. Policy optimization using P[Love] significantly raises observed positive-feedback rates, including a 28% increase in Love Reactions during live A/B tests. However, optimizing for positive reactions introduces reward hacking challenges, requiring careful balancing of objectives. By directly leveraging implicit signals from users, RLUF offers a path to aligning LLMs with real-world user preferences at scale. 

**Abstract (ZH)**: 基于用户反馈的强化学习（RLUF）：一种直接将大型语言模型与生产中的隐式用户信号对齐的框架 

---
# To Be or Not To Be: Vector ontologies as a truly formal ontological framework 

**Title (ZH)**: 是与不是：向量本体作为真正形式化的本体框架 

**Authors**: Kaspar Rothenfusser  

**Link**: [PDF](https://arxiv.org/pdf/2505.14940)  

**Abstract**: Since Edmund Husserl coined the term "Formal Ontologies" in the early 20th century, a field that identifies itself with this particular branch of sciences has gained increasing attention. Many authors, and even Husserl himself have developed what they claim to be formal ontologies. I argue that under close inspection, none of these so claimed formal ontologies are truly formal in the Husserlian sense. More concretely, I demonstrate that they violate the two most important notions of formal ontology as developed in Husserl's Logical Investigations, namely a priori validity independent of perception and formalism as the total absence of content. I hence propose repositioning the work previously understood as formal ontology as the foundational ontology it really is. This is to recognize the potential of a truly formal ontology in the Husserlian sense. Specifically, I argue that formal ontology following his conditions, allows us to formulate ontological structures, which could capture what is more objectively without presupposing a particular framework arising from perception. I further argue that the ability to design the formal structure deliberately allows us to create highly scalable and interoperable information artifacts. As concrete evidence, I showcase that a class of formal ontology, which uses the axioms of vector spaces, is able to express most of the conceptualizations found in foundational ontologies. Most importantly, I argue that many information systems, specifically artificial intelligence, are likely already using some type of vector ontologies to represent reality in their internal worldviews and elaborate on the evidence that humans do as well. I hence propose a thorough investigation of the ability of vector ontologies to act as a human-machine interoperable ontological framework that allows us to understand highly sophisticated machines and machines to understand us. 

**Abstract (ZH)**: 自埃德蒙·胡塞尔在20世纪初提出“形式本体论”这一术语以来，一个以这一特定科学分支为标志的领域受到了日益关注。许多作者，甚至胡塞尔本人，都发展了他们声称的形式本体论。在我看来，在仔细审视之下，这些所谓的形式本体论都不符合胡塞尔意义上的“形式”。更具体地说，我证明它们违背了胡塞尔在《逻辑探究》中发展出的形式本体论的两个最重要的概念，即独立于感知的先验有效性以及形式化为完全不含内容的状态。因此，我建议将先前被认为是形式本体论的工作重新定位为真正的基础本体论。这一重新定位旨在认可在胡塞尔意义上真正形式本体论的潜力。具体而言，我主张符合胡塞尔条件的形式本体论使我们能够构建能够客观地捕捉概念而不预设特定感知框架的本体结构。进一步而言，我主张有意图地设计形式结构的能力使我们能够创建高度可扩展且互操作的信息实体。作为具体证据，我展示了使用向量空间公理的一类形式本体能够表达基础本体中大多数的概念化内容。最重要的是，我主张许多信息系统，特别是人工智能系统，很可能已经在其内部世界观中使用某种类型的向量本体来表示现实，人类也可能是如此。因此，我建议深入研究向量本体的能力，使其成为一种人机可互操作的本体框架，使我们能够理解高度复杂的机器，同时也使机器能够理解人类。 

---
# FOL-Pretrain: A complexity annotated corpus of first-order logic 

**Title (ZH)**: FOL-预训练：带有复杂性标注的一阶逻辑语料库 

**Authors**: Isabelle Lee, Sarah Liaw, Dani Yogatama  

**Link**: [PDF](https://arxiv.org/pdf/2505.14932)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated remarkable reasoning capabilities such as coding and solving mathematical problems to commonsense inference. While these tasks vary in complexity, they all require models to integrate and compute over structured information. Despite recent efforts to reverse-engineer LLM behavior through controlled experiments, our understanding of how these models internalize and execute complex algorithms remains limited. Progress has largely been confined to small-scale studies or shallow tasks such as basic arithmetic and grammatical pattern matching. One barrier to deeper understanding is the nature of pretraining data -- vast, heterogeneous, and often poorly annotated, making it difficult to isolate mechanisms of reasoning. To bridge this gap, we introduce a large-scale, fully open, complexity-annotated dataset of first-order logic reasoning traces, designed to probe and analyze algorithmic reasoning in LLMs. The dataset consists of 3.5 billion tokens, including 8.8 million LLM-augmented, human-annotated examples and 7.5 million synthetically generated examples. Each synthetic example is verifiably correct, produced by a custom automated theorem solver, and accompanied by metadata tracing its algorithmic provenance. We aim to provide a scalable, interpretable artifact for studying how LLMs learn and generalize symbolic reasoning processes, paving the way for more transparent and targeted investigations into the algorithmic capabilities of modern models. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLM）展示了 remarkable 的推理能力，包括编程、解决数学问题到常识推理。尽管这些任务在复杂性上有所不同，但都需要模型整合和计算结构化信息。尽管最近通过受控实验反向工程LLM的行为取得了进步，但我们对这些模型如何 internalize 和执行复杂算法的理解仍然有限。进展主要局限于小型研究或简单的任务，如基本算术和语法模式匹配。一个深入了解的障碍是预训练数据的性质——庞大、异构且通常标注不佳，这使得难以隔离推理机制。为弥补这一差距，我们引入了一个大规模、全开放、复杂性标注的一阶逻辑推理跟踪数据集，旨在探究和分析LLM的算法推理。该数据集包含35亿个标记，包括880万个人工标注的LLM增强示例和7500万个合成生成的示例。每个合成示例都由自定义自动定理求解器验证正确，并附有追踪其算法起源的元数据。我们旨在提供一个可扩展、可解释的工具，用于研究LLM如何学习和泛化符号推理过程，为更透明和有针对性地探究现代模型的算法能力铺平道路。 

---
# R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution 

**Title (ZH)**: R&D-Agent: 通过LLM赋能的自动化数据驱动AI解决方案构建、开发与进化 

**Authors**: Xu Yang, Xiao Yang, Shikai Fang, Bowen Xian, Yuante Li, Jian Wang, Minrui Xu, Haoran Pan, Xinpeng Hong, Weiqing Liu, Yelong Shen, Weizhu Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2505.14738)  

**Abstract**: Recent advances in AI and ML have transformed data science, yet increasing complexity and expertise requirements continue to hinder progress. While crowdsourcing platforms alleviate some challenges, high-level data science tasks remain labor-intensive and iterative. To overcome these limitations, we introduce R&D-Agent, a dual-agent framework for iterative exploration. The Researcher agent uses performance feedback to generate ideas, while the Developer agent refines code based on error feedback. By enabling multiple parallel exploration traces that merge and enhance one another, R&D-Agent narrows the gap between automated solutions and expert-level performance. Evaluated on MLE-Bench, R&D-Agent emerges as the top-performing machine learning engineering agent, demonstrating its potential to accelerate innovation and improve precision across diverse data science applications. We have open-sourced R&D-Agent on GitHub: this https URL. 

**Abstract (ZH)**: 最近在人工智能和机器学习方面的进展已转变了数据科学，但不断增加的复杂性和专业知识要求依然阻碍着进步。尽管众包平台可以缓解一些挑战，但高级数据科学任务依然劳动密集且需要迭代。为克服这些限制，我们引入了R&D-Agent，这是一种用于迭代探索的双代理框架。研究员代理利用性能反馈生成想法，而开发者代理基于错误反馈改进代码。通过启用多个并行的探索轨迹并相互融合和增强，R&D-Agent 缩小了自动化解决方案与专家级性能之间的差距。在MLE-Bench上的评估表明，R&D-Agent 是表现最佳的机器学习工程代理，展示了其在促进创新并提高各种数据科学应用精度方面的潜力。我们已在GitHub上开源了R&D-Agent：this https URL。 

---
# Follow the STARs: Dynamic $ω$-Regular Shielding of Learned Policies 

**Title (ZH)**: 遵循STARs：动态ω-正则屏蔽学习策略 

**Authors**: Ashwani Anand, Satya Prakash Nayak, Ritam Raha, Anne-Kathrin Schmuck  

**Link**: [PDF](https://arxiv.org/pdf/2505.14689)  

**Abstract**: This paper presents a novel dynamic post-shielding framework that enforces the full class of $\omega$-regular correctness properties over pre-computed probabilistic policies. This constitutes a paradigm shift from the predominant setting of safety-shielding -- i.e., ensuring that nothing bad ever happens -- to a shielding process that additionally enforces liveness -- i.e., ensures that something good eventually happens. At the core, our method uses Strategy-Template-based Adaptive Runtime Shields (STARs), which leverage permissive strategy templates to enable post-shielding with minimal interference. As its main feature, STARs introduce a mechanism to dynamically control interference, allowing a tunable enforcement parameter to balance formal obligations and task-specific behavior at runtime. This allows to trigger more aggressive enforcement when needed, while allowing for optimized policy choices otherwise. In addition, STARs support runtime adaptation to changing specifications or actuator failures, making them especially suited for cyber-physical applications. We evaluate STARs on a mobile robot benchmark to demonstrate their controllable interference when enforcing (incrementally updated) $\omega$-regular correctness properties over learned probabilistic policies. 

**Abstract (ZH)**: 本文提出了一种新颖的动态后屏蔽框架，该框架在预先计算的概率策略上强制执行所有类ω-正规正确性属性。这代表了从当前占主导地位的安全屏蔽范式——即确保什么都不会坏——向一种在确保好事最终发生的同时进行屏蔽的过程的转变。核心上，我们的方法使用基于策略模板的自适应运行时屏蔽（STARs），利用宽松策略模板来实现最少干扰的后屏蔽。STARs的主要特征是引入了一种动态控制干扰的机制，允许在运行时通过可调的强制执行参数平衡形式义务和任务特定行为。这使得在需要时可以触发更严格的强制执行，而在其他情况下可以选择优化策略选择。此外，STARs支持运行时针对变化的规范或执行器故障进行适应，使它们特别适合于网络物理应用。我们在一个移动机器人基准测试中评估了STARs，以证明其在强制执行（逐步更新的）ω-正规正确性属性时可控的干扰。 

---
# GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents 

**Title (ZH)**: GUI-G1: 理解R1-Zero-like训练在GUI代理视觉定位中的应用 

**Authors**: Yuqi Zhou, Sunhao Dai, Shuai Wang, Kaiwen Zhou, Qinqlin Jia, Junxu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15810)  

**Abstract**: Recent Graphical User Interface (GUI) agents replicate the R1-Zero paradigm, coupling online Reinforcement Learning (RL) with explicit chain-of-thought reasoning prior to object grounding and thereby achieving substantial performance gains. In this paper, we first conduct extensive analysis experiments of three key components of that training pipeline: input design, output evaluation, and policy update-each revealing distinct challenges arising from blindly applying general-purpose RL without adapting to GUI grounding tasks. Input design: Current templates encourage the model to generate chain-of-thought reasoning, but longer chains unexpectedly lead to worse grounding performance. Output evaluation: Reward functions based on hit signals or box area allow models to exploit box size, leading to reward hacking and poor localization quality. Policy update: Online RL tends to overfit easy examples due to biases in length and sample difficulty, leading to under-optimization on harder cases. To address these issues, we propose three targeted solutions. First, we adopt a Fast Thinking Template that encourages direct answer generation, reducing excessive reasoning during training. Second, we incorporate a box size constraint into the reward function to mitigate reward hacking. Third, we revise the RL objective by adjusting length normalization and adding a difficulty-aware scaling factor, enabling better optimization on hard samples. Our GUI-G1-3B, trained on 17K public samples with Qwen2.5-VL-3B-Instruct, achieves 90.3% accuracy on ScreenSpot and 37.1% on ScreenSpot-Pro. This surpasses all prior models of similar size and even outperforms the larger UI-TARS-7B, establishing a new state-of-the-art in GUI agent grounding. The project repository is available at this https URL. 

**Abstract (ZH)**: Recent Graphical User Interface (GUI) 代理复制了R1-Zero范式，将在线强化学习（RL）与对象定位之前的显式链式思考推理相结合，从而实现了显著的性能提升。在本文中，我们首先对训练管道的三个关键组件进行了广泛分析：输入设计、输出评估和策略更新——每个组件都揭示了在不适应GUI对象定位任务的情况下直接应用通用RL所引发的独特挑战。输入设计：当前模板鼓励模型生成链式思考推理，但较长的链式思考意外地导致了更差的对象定位性能。输出评估：基于检测信号或框面积的奖励函数使模型能够利用框的大小，导致奖励作弊和较低的质量定位。策略更新：由于长度和样本难度的偏见，在线RL倾向于过度拟合简单的例子，从而在困难案例上优化不足。为了解决这些问题，我们提出了三种针对性的解决方案。首先，我们采用了快速思维模板，鼓励直接生成答案，减少训练过程中的过度推理。其次，我们将框大小约束纳入奖励函数中，以减轻奖励作弊。第三，我们修订了RL目标，通过调整长度正常化和加入难度感知的缩放因子，使模型更好地优化难点样本。我们的GUI-G1-3B使用Qwen2.5-VL-3B-Instruct在17000个公开样本上进行训练，在ScreenSpot上的准确率达到90.3%，在ScreenSpot-Pro上的准确率为37.1%。这超过了所有类似规模的先前模型，并甚至超过了更大的UI-TARS-7B，建立了GUI代理定位的新state-of-the-art。项目仓库可在此链接访问。 

---
# Neural Conditional Transport Maps 

**Title (ZH)**: 神经条件性输运映射 

**Authors**: Carlos Rodriguez-Pardo, Leonardo Chiani, Emanuele Borgonovo, Massimo Tavoni  

**Link**: [PDF](https://arxiv.org/pdf/2505.15808)  

**Abstract**: We present a neural framework for learning conditional optimal transport (OT) maps between probability distributions. Our approach introduces a conditioning mechanism capable of processing both categorical and continuous conditioning variables simultaneously. At the core of our method lies a hypernetwork that generates transport layer parameters based on these inputs, creating adaptive mappings that outperform simpler conditioning methods. Comprehensive ablation studies demonstrate the superior performance of our method over baseline configurations. Furthermore, we showcase an application to global sensitivity analysis, offering high performance in computing OT-based sensitivity indices. This work advances the state-of-the-art in conditional optimal transport, enabling broader application of optimal transport principles to complex, high-dimensional domains such as generative modeling and black-box model explainability. 

**Abstract (ZH)**: 一种学习条件最优运输映射的神经框架：同时处理分类和连续条件变量 

---
# VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models 

**Title (ZH)**: VerifyBench：基于引用奖励系统的大型语言模型基准测试 

**Authors**: Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, Yang Liu, Xin Xu, Mengdi Zhang, Jian Shao, Yongliang Shen, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15801)  

**Abstract**: Large reasoning models such as OpenAI o1 and DeepSeek-R1 have achieved remarkable performance in the domain of reasoning. A key component of their training is the incorporation of verifiable rewards within reinforcement learning (RL). However, existing reward benchmarks do not evaluate reference-based reward systems, leaving researchers with limited understanding of the accuracy of verifiers used in RL. In this paper, we introduce two benchmarks, VerifyBench and VerifyBench-Hard, designed to assess the performance of reference-based reward systems. These benchmarks are constructed through meticulous data collection and curation, followed by careful human annotation to ensure high quality. Current models still show considerable room for improvement on both VerifyBench and VerifyBench-Hard, especially smaller-scale models. Furthermore, we conduct a thorough and comprehensive analysis of evaluation results, offering insights for understanding and developing reference-based reward systems. Our proposed benchmarks serve as effective tools for guiding the development of verifier accuracy and the reasoning capabilities of models trained via RL in reasoning tasks. 

**Abstract (ZH)**: Large Reasoning Models Such as OpenAI o1 and DeepSeek-R1 Have Achieved Remarkable Performance in the Domain of Reasoning: Introducing VerifyBench and VerifyBench-Hard to Assess Reference-Based Reward Systems 

---
# Long-Form Information Alignment Evaluation Beyond Atomic Facts 

**Title (ZH)**: 长文本信息一致性评估超越原子事实 

**Authors**: Danna Zheng, Mirella Lapata, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15792)  

**Abstract**: Information alignment evaluators are vital for various NLG evaluation tasks and trustworthy LLM deployment, reducing hallucinations and enhancing user trust. Current fine-grained methods, like FactScore, verify facts individually but neglect inter-fact dependencies, enabling subtle vulnerabilities. In this work, we introduce MontageLie, a challenging benchmark that constructs deceptive narratives by "montaging" truthful statements without introducing explicit hallucinations. We demonstrate that both coarse-grained LLM-based evaluators and current fine-grained frameworks are susceptible to this attack, with AUC-ROC scores falling below 65%. To enable more robust fine-grained evaluation, we propose DoveScore, a novel framework that jointly verifies factual accuracy and event-order consistency. By modeling inter-fact relationships, DoveScore outperforms existing fine-grained methods by over 8%, providing a more robust solution for long-form text alignment evaluation. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 信息对齐评估器对于各种NLG评估任务和可信赖的大语言模型部署至关重要，能够减少幻觉并增强用户信任。当前的细粒度方法，如FactScore，单独验证事实但忽略了事实之间的依赖关系，从而允许潜在的漏洞。在本工作中，我们引入了MontageLie基准，通过“拼接”真实陈述构建欺骗性叙事，而不引入明显的幻觉。我们证明了粗粒度的基于大语言模型的评估器和当前的细粒度框架都对这种攻击敏感，AUC-ROC得分低于65%。为了实现更稳健的细粒度评估，我们提出了DoveScore，这是一种新的框架，可以同时验证事实准确性与时序一致性。通过建模事实之间的关系，DoveScore在现有细粒度方法的基础上提高了超过8%的表现，为长文本对齐评估提供了更稳健的解决方案。我们的代码和数据集可在以下链接获取。 

---
# Exploring the Innovation Opportunities for Pre-trained Models 

**Title (ZH)**: 探索预训练模型的创新机遇 

**Authors**: Minjung Park, Jodi Forlizzi, John Zimmerman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15790)  

**Abstract**: Innovators transform the world by understanding where services are successfully meeting customers' needs and then using this knowledge to identify failsafe opportunities for innovation. Pre-trained models have changed the AI innovation landscape, making it faster and easier to create new AI products and services. Understanding where pre-trained models are successful is critical for supporting AI innovation. Unfortunately, the hype cycle surrounding pre-trained models makes it hard to know where AI can really be successful. To address this, we investigated pre-trained model applications developed by HCI researchers as a proxy for commercially successful applications. The research applications demonstrate technical capabilities, address real user needs, and avoid ethical challenges. Using an artifact analysis approach, we categorized capabilities, opportunity domains, data types, and emerging interaction design patterns, uncovering some of the opportunity space for innovation with pre-trained models. 

**Abstract (ZH)**: 创新者通过理解服务如何成功满足客户的需求，并利用这些知识识别预训练模型领域的安全创新机会，从而改变世界。预训练模型改变了AI创新的格局，使其更快、更容易创建新的AI产品和服务。了解预训练模型成功的地方对于支持AI创新至关重要。不幸的是，围绕预训练模型的 hype 周期使得知道AI在哪里真正取得成功变得困难。为了解决这一问题，我们调查了人机交互研究人员开发的预训练模型应用，作为商业化成功应用的代理。研究应用展示了技术能力，解决了真实用户需求，并避免了伦理挑战。通过实体分析的方法，我们对能力、机会领域、数据类型以及新兴的交互设计模式进行了分类，揭示了预训练模型创新的机会空间。 

---
# Large Language Models as Computable Approximations to Solomonoff Induction 

**Title (ZH)**: 大型语言模型作为索洛莫诺夫归纳的可计算近似 

**Authors**: Jun Wan, Lingrui Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.15784)  

**Abstract**: The rapid advancement of large language models (LLMs) calls for a rigorous theoretical framework to explain their empirical success. While significant progress has been made in understanding LLM behaviors, existing theoretical frameworks remain fragmented in explaining emergent phenomena through a unified mathematical lens. We establish the first formal connection between LLM architectures and Algorithmic Information Theory (AIT) by proving two fundamental results: (1) the training process computationally approximates Solomonoff prior through loss minimization interpreted as program length optimization, and (2) next-token prediction implements approximate Solomonoff induction. We leverage AIT to provide a unified theoretical explanation for in-context learning, few-shot learning, and scaling laws. Furthermore, our theoretical insights lead to a principled method for few-shot example selection that prioritizes samples where models exhibit lower predictive confidence. We demonstrate through experiments on diverse text classification benchmarks that this strategy yields significant performance improvements, particularly for smaller model architectures, when compared to selecting high-confidence examples. Our framework bridges the gap between theoretical foundations and practical LLM behaviors, providing both explanatory power and actionable insights for future model development. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展呼唤一个严谨的理论框架来解释其 empirical 成功。虽然在理解 LLM 行为方面已经取得显著进展，但现有的理论框架仍缺乏通过统一的数学视角来解释新兴现象的能力。我们通过证明两个基本结果，首次正式建立了 LLM 架构与算法信息论（AIT）之间的联系：（1）训练过程通过对损失最小化进行计算性近似，将索罗门off 先验近似为程序长度优化，（2）下一个 token 预测实现了近似的索罗门off 归纳。我们利用AIT提供了一个统一的理论解释，包括上下文学习、少样本学习和标度规律。此外，我们的理论见解导致了一种原则性的少样本示例选择方法，优先选择模型预测置信度较低的样本。通过在各种文本分类基准上的实验，我们证明了这种方法在与选择高置信度示例相比时，对于较小的模型架构尤其能够带来显著的性能提升。我们的框架在理论基础与实际 LLM 行为之间架起了桥梁，不仅提供了解释力，还提供了对未来模型开发的实际洞察。 

---
# IA-T2I: Internet-Augmented Text-to-Image Generation 

**Title (ZH)**: 基于互联网增强的文本到图像生成 

**Authors**: Chuanhao Li, Jianwen Sun, Yukang Feng, Mingliang Zhai, Yifan Chang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15779)  

**Abstract**: Current text-to-image (T2I) generation models achieve promising results, but they fail on the scenarios where the knowledge implied in the text prompt is uncertain. For example, a T2I model released in February would struggle to generate a suitable poster for a movie premiering in April, because the character designs and styles are uncertain to the model. To solve this problem, we propose an Internet-Augmented text-to-image generation (IA-T2I) framework to compel T2I models clear about such uncertain knowledge by providing them with reference images. Specifically, an active retrieval module is designed to determine whether a reference image is needed based on the given text prompt; a hierarchical image selection module is introduced to find the most suitable image returned by an image search engine to enhance the T2I model; a self-reflection mechanism is presented to continuously evaluate and refine the generated image to ensure faithful alignment with the text prompt. To evaluate the proposed framework's performance, we collect a dataset named Img-Ref-T2I, where text prompts include three types of uncertain knowledge: (1) known but rare. (2) unknown. (3) ambiguous. Moreover, we carefully craft a complex prompt to guide GPT-4o in making preference evaluation, which has been shown to have an evaluation accuracy similar to that of human preference evaluation. Experimental results demonstrate the effectiveness of our framework, outperforming GPT-4o by about 30% in human evaluation. 

**Abstract (ZH)**: 互联网增强的文字到图像生成框架（IA-T2I） 

---
# Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space 

**Title (ZH)**: 软思考：在连续概念空间中解锁LLMs的推理潜力 

**Authors**: Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15778)  

**Abstract**: Human cognition typically involves thinking through abstract, fluid concepts rather than strictly using discrete linguistic tokens. Current reasoning models, however, are constrained to reasoning within the boundaries of human language, processing discrete token embeddings that represent fixed points in the semantic space. This discrete constraint restricts the expressive power and upper potential of such reasoning models, often causing incomplete exploration of reasoning paths, as standard Chain-of-Thought (CoT) methods rely on sampling one token per step. In this work, we introduce Soft Thinking, a training-free method that emulates human-like "soft" reasoning by generating soft, abstract concept tokens in a continuous concept space. These concept tokens are created by the probability-weighted mixture of token embeddings, which form the continuous concept space, enabling smooth transitions and richer representations that transcend traditional discrete boundaries. In essence, each generated concept token encapsulates multiple meanings from related discrete tokens, implicitly exploring various reasoning paths to converge effectively toward the correct answer. Empirical evaluations on diverse mathematical and coding benchmarks consistently demonstrate the effectiveness and efficiency of Soft Thinking, improving pass@1 accuracy by up to 2.48 points while simultaneously reducing token usage by up to 22.4% compared to standard CoT. Qualitative analysis further reveals that Soft Thinking outputs remain highly interpretable and readable, highlighting the potential of Soft Thinking to break the inherent bottleneck of discrete language-based reasoning. Code is available at this https URL. 

**Abstract (ZH)**: 人类认知通常涉及通过抽象且流动的概念进行思考，而不是严格地使用离散的语言标记。然而，当前的推理模型仅限于在人类语言的框架内进行推理，处理表示语义空间中固定点的离散标记嵌入。这种离散约束限制了此类推理模型的表达能力和潜在能力，常常导致推理路径探索不完整，因为标准思维链（CoT）方法依赖于每步采样一个标记。在本文中，我们提出了软思考（Soft Thinking）这一无需训练的方法，通过在连续的概念空间中生成软的抽象概念标记来模拟人类的“软”推理。这些概念标记是由标记嵌入的概率加权混合生成的，形成了连续的概念空间，这使得可以实现平滑过渡和更丰富的表示，超越了传统离散边界的限制。本质上而言，每个生成的概念标记封装了相关离散标记的多种含义，隐含地探索各种推理路径以有效收敛于正确答案。在不同的数学和编码基准测试上的实证评估一致表明，软思考的有效性和效率，与标准CoT相比，软思考可以提高pass@1准确率高达2.48个百分点，同时降低标记使用率高达22.4%。定性分析进一步表明，软思考的输出保持高度可解释性和可读性，突显了软思考在打破基于离散语言推理的固有问题方面的潜力。代码可在此处访问：this https URL。 

---
# Constructing a 3D Town from a Single Image 

**Title (ZH)**: 从单张图像构建三维城镇 

**Authors**: Kaizhi Zheng, Ruijian Zhang, Jing Gu, Jie Yang, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15765)  

**Abstract**: Acquiring detailed 3D scenes typically demands costly equipment, multi-view data, or labor-intensive modeling. Therefore, a lightweight alternative, generating complex 3D scenes from a single top-down image, plays an essential role in real-world applications. While recent 3D generative models have achieved remarkable results at the object level, their extension to full-scene generation often leads to inconsistent geometry, layout hallucinations, and low-quality meshes. In this work, we introduce 3DTown, a training-free framework designed to synthesize realistic and coherent 3D scenes from a single top-down view. Our method is grounded in two principles: region-based generation to improve image-to-3D alignment and resolution, and spatial-aware 3D inpainting to ensure global scene coherence and high-quality geometry generation. Specifically, we decompose the input image into overlapping regions and generate each using a pretrained 3D object generator, followed by a masked rectified flow inpainting process that fills in missing geometry while maintaining structural continuity. This modular design allows us to overcome resolution bottlenecks and preserve spatial structure without requiring 3D supervision or fine-tuning. Extensive experiments across diverse scenes show that 3DTown outperforms state-of-the-art baselines, including Trellis, Hunyuan3D-2, and TripoSG, in terms of geometry quality, spatial coherence, and texture fidelity. Our results demonstrate that high-quality 3D town generation is achievable from a single image using a principled, training-free approach. 

**Abstract (ZH)**: 一种基于单张鸟瞰图生成真实且一致的3D场景的训练-free框架 

---
# Improving planning and MBRL with temporally-extended actions 

**Title (ZH)**: 改进规划和基于模型的强化学习中的时间延长动作 

**Authors**: Palash Chatterjee, Roni Khardon  

**Link**: [PDF](https://arxiv.org/pdf/2505.15754)  

**Abstract**: Continuous time systems are often modeled using discrete time dynamics but this requires a small simulation step to maintain accuracy. In turn, this requires a large planning horizon which leads to computationally demanding planning problems and reduced performance. Previous work in model free reinforcement learning has partially addressed this issue using action repeats where a policy is learned to determine a discrete action duration. Instead we propose to control the continuous decision timescale directly by using temporally-extended actions and letting the planner treat the duration of the action as an additional optimization variable along with the standard action variables. This additional structure has multiple advantages. It speeds up simulation time of trajectories and, importantly, it allows for deep horizon search in terms of primitive actions while using a shallow search depth in the planner. In addition, in the model based reinforcement learning (MBRL) setting, it reduces compounding errors from model learning and improves training time for models. We show that this idea is effective and that the range for action durations can be automatically selected using a multi-armed bandit formulation and integrated into the MBRL framework. An extensive experimental evaluation both in planning and in MBRL, shows that our approach yields faster planning, better solutions, and that it enables solutions to problems that are not solved in the standard formulation. 

**Abstract (ZH)**: 连续时间系统通常使用离散时间动力学建模，但这需要较小的仿真步长以保持准确性。这反过来又要求较长的规划时段，导致计算复杂度高的规划问题并降低性能。无模型强化学习的先前工作部分解决了这一问题，通过动作重播让策略确定离散的动作持续时间。相反，我们直接控制连续的决策时间尺度，通过使用时间扩展的动作让规划器将动作持续时间作为额外的优化变量，与标准的动作变量一起处理。这种额外的结构具有多方面的好处。它加速了轨迹仿真时间，更重要的是，它允许在基础动作方面进行深度前瞻性搜索，而在规划器方面使用浅层搜索深度。此外，在模型基于强化学习（MBRL）设置中，它减少了模型学习中的累积误差并提高了模型的训练时间。我们展示了这一想法的有效性，并且可以使用多臂 bandit 公式自动选择动作持续时间的范围并将其整合到 MBRL 框架中。广泛的实验评估表明，我们的方法可以实现更快的规划、更好的解决方案，并且可以解决标准表述无法解决的问题。 

---
# Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval 

**Title (ZH)**: 基于安全上下文检索的大规模防御在野*jailbreaking*攻击 

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15753)  

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication. 

**Abstract (ZH)**: 大型语言模型（LLMs）易受 Jailbreaking 攻击，攻击者通过精心设计的提示诱导有害或不道德的响应。这类威胁引发了对 LLMs 在实际部署中的安全性和可靠性的严重担忧。尽管现有防御机制部分缓解了此类风险，但随后对手技术的进展使得新的 Jailbreaking 方法能够绕过这些防护，暴露出静态防御框架的局限性。在本工作中，我们从上下文检索的角度探索防范 evolving jailbreaking 威胁的方法。首先，我们进行初步研究，表明针对某特定 Jailbreak 的少量安全对齐示例可以显著增强对该攻击模式的鲁棒性。基于这一洞见，我们进一步利用检索增强生成（RAG）技术，并提出安全上下文检索（SCR），这是一种可扩展且稳健的 LLM 防护范式，以抵御 Jailbreaking。我们的全面实验展示了 SCR 如何在对抗既有的和新兴的 Jailbreaking 技巧时实现更优的防御性能，为 LLM 安全性贡献了一个新范式。我们的代码将在发表后提供。 

---
# Multi-modal Integration Analysis of Alzheimer's Disease Using Large Language Models and Knowledge Graphs 

**Title (ZH)**: 使用大型语言模型和知识图谱的阿尔茨海默病多模态整合分析 

**Authors**: Kanan Kiguchi, Yunhao Tu, Katsuhiro Ajito, Fady Alnajjar, Kazuyuki Murase  

**Link**: [PDF](https://arxiv.org/pdf/2505.15747)  

**Abstract**: We propose a novel framework for integrating fragmented multi-modal data in Alzheimer's disease (AD) research using large language models (LLMs) and knowledge graphs. While traditional multimodal analysis requires matched patient IDs across datasets, our approach demonstrates population-level integration of MRI, gene expression, biomarkers, EEG, and clinical indicators from independent cohorts. Statistical analysis identified significant features in each modality, which were connected as nodes in a knowledge graph. LLMs then analyzed the graph to extract potential correlations and generate hypotheses in natural language. This approach revealed several novel relationships, including a potential pathway linking metabolic risk factors to tau protein abnormalities via neuroinflammation (r>0.6, p<0.001), and unexpected correlations between frontal EEG channels and specific gene expression profiles (r=0.42-0.58, p<0.01). Cross-validation with independent datasets confirmed the robustness of major findings, with consistent effect sizes across cohorts (variance <15%). The reproducibility of these findings was further supported by expert review (Cohen's k=0.82) and computational validation. Our framework enables cross modal integration at a conceptual level without requiring patient ID matching, offering new possibilities for understanding AD pathology through fragmented data reuse and generating testable hypotheses for future research. 

**Abstract (ZH)**: 我们提出了一种新的框架，使用大规模语言模型和知识图谱整合阿尔茨海默病研究中的碎片化多模态数据 

---
# Higher-order Structure Boosts Link Prediction on Temporal Graphs 

**Title (ZH)**: 高阶结构提升-temporal图上的链接预测 

**Authors**: Jingzhe Liu, Zhigang Hua, Yan Xie, Bingheng Li, Harry Shomer, Yu Song, Kaveh Hassani, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15746)  

**Abstract**: Temporal Graph Neural Networks (TGNNs) have gained growing attention for modeling and predicting structures in temporal graphs. However, existing TGNNs primarily focus on pairwise interactions while overlooking higher-order structures that are integral to link formation and evolution in real-world temporal graphs. Meanwhile, these models often suffer from efficiency bottlenecks, further limiting their expressive power. To tackle these challenges, we propose a Higher-order structure Temporal Graph Neural Network, which incorporates hypergraph representations into temporal graph learning. In particular, we develop an algorithm to identify the underlying higher-order structures, enhancing the model's ability to capture the group interactions. Furthermore, by aggregating multiple edge features into hyperedge representations, HTGN effectively reduces memory cost during training. We theoretically demonstrate the enhanced expressiveness of our approach and validate its effectiveness and efficiency through extensive experiments on various real-world temporal graphs. Experimental results show that HTGN achieves superior performance on dynamic link prediction while reducing memory costs by up to 50\% compared to existing methods. 

**Abstract (ZH)**: 高阶结构时序图神经网络（HTGNs）：融合超图表示以建模和预测时序图中的结构与演化 

---
# HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement 

**Title (ZH)**: HybridProver: 结合LLM驱动的证明合成与 refinement 的定理证明方法 

**Authors**: Jilin Hu, Jianyu Zhang, Yongwang Zhao, Talia Ringer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15740)  

**Abstract**: Formal methods is pivotal for verifying the reliability of critical systems through rigorous mathematical proofs. However, its adoption is hindered by labor-intensive manual proofs and the expertise required to use theorem provers. Recent advancements in large language models (LLMs) offer new opportunities for automated theorem proving. Two promising approaches are generating tactics step by step and generating a whole proof directly with an LLM. However, existing work makes no attempt to combine the two approaches. In this work, we introduce HybridProver, a dual-model proof synthesis framework that combines tactic-based generation and whole-proof synthesis to harness the benefits of both approaches. HybridProver generates whole proof candidates for evaluation directly, then extracts proof sketches from those candidates. It then uses a tactic-based generation model that integrates automated tools to complete the sketches via stepwise refinement. We implement HybridProver for the Isabelle theorem prover and fine-tune LLMs on our optimized Isabelle datasets. Evaluation on the miniF2F dataset illustrates HybridProver's effectiveness. We achieve a 59.4% success rate on miniF2F, where the previous SOTA is 56.1%. Our ablation studies show that this SOTA result is attributable to combining whole-proof and tactic-based generation. Additionally, we show how the dataset quality, training parameters, and sampling diversity affect the final result during automated theorem proving with LLMs. All of our code, datasets, and LLMs are open source. 

**Abstract (ZH)**: 形式化方法对于通过严格的数学证明验证关键系统可靠性至关重要。然而，其采用受到劳动密集型的手动证明和使用定理证明器所需的专门知识的阻碍。近期大型语言模型的进展为自动定理证明提供了新机会。两种有前景的方法是通过逐步生成策略和直接使用大型语言模型生成完整证明。然而，现有研究并未尝试将这两种方法结合起来。在本文中，我们提出了HybridProver，这是一种结合基于策略的生成和整体证明合成的双模型证明合成框架，以利用这两种方法的优势。HybridProver直接生成用于评估的整体证明候选，然后从中提取证明草图。接着使用结合自动工具的基于策略的生成模型，通过逐步细化完成这些草图。我们为Isabelle定理证明器实现HybridProver，并在我们优化的Isabelle数据集上微调大型语言模型。对miniF2F数据集的评估展示了HybridProver的有效性。我们在miniF2F上实现了59.4%的成功率，而之前的最佳成果（SOTA）是56.1%。我们的消融研究显示，这一最佳成果归因于结合整体证明和基于策略的生成。此外，我们展示了数据集质量、训练参数和采样多样性如何影响使用大型语言模型进行自动定理证明的最终结果。所有我们的代码、数据集和大型语言模型都是开源的。 

---
# Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses 

**Title (ZH)**: 在压力之下对齐：评估LLM防御措施时需要有知识的对手的理由 

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15738)  

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在从聊天bot到代理系统等多种现实应用中迅速部署。对齐是防御诸如提示注入和 Jailbreak 等攻击的主要方法之一。最近的研究报告即使在 Greedy Coordinate Gradient（GCG）这种白盒攻击（它可以生成对抗后缀以诱导攻击者期望的输出）面前，防御措施也能实现几乎零的攻击成功率（ASR）。然而，随着对离散令牌搜索空间的扩大，成功攻击的发现任务变得极具挑战性。GCG 例如已被证明会收敛到局部极小值，使其对初始化的选择高度敏感。在本文中，我们使用更明智的威胁模型来评估这些防御措施的未来稳健性：拥有对对齐过程某些信息访问权限的攻击者。具体地，我们提出了一个利用中间模型检查点进行初始化的有信息的白盒攻击方案，每个检查点作为下一个检查点的踏脚石。我们表明该方法对最先进的（SOTA）防御措施和模型具有高度有效性。进一步显示我们的有信息初始化方法优于其他初始化方法，并提出了基于梯度的检查点选择策略，以大幅提高攻击性能和效率。重要的是，我们还展示了我们的方法能够找到通用的对抗后缀——这些后缀在多种输入下均可生效。我们的结果表明，与以往认为不同，有效的对抗后缀确实存在于对抗 SOTA 对齐基线防御措施的情况中；这些后缀在对手利用对齐知识时能够被现有攻击方法找到；甚至通用的后缀也确实存在。综合来看，我们的结果揭示了当前对齐基线方法的脆弱性，并强调在测试 LLM 安全性时需要考虑更强的威胁模型的重要性。 

---
# DEBATE, TRAIN, EVOLVE: Self Evolution of Language Model Reasoning 

**Title (ZH)**: 辩论、训练、演变：语言模型的自我进化 

**Authors**: Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15734)  

**Abstract**: Large language models (LLMs) have improved significantly in their reasoning through extensive training on massive datasets. However, relying solely on additional data for improvement is becoming increasingly impractical, highlighting the need for models to autonomously enhance their reasoning without external supervision. In this paper, we propose Debate, Train, Evolve (DTE), a novel ground truth-free training framework that uses multi-agent debate traces to evolve a single language model. We also introduce a new prompting strategy Reflect-Critique-Refine, to improve debate quality by explicitly instructing agents to critique and refine their reasoning. Extensive evaluations on five reasoning benchmarks with six open-weight models show that our DTE framework achieve substantial improvements, with an average accuracy gain of 8.92% on the challenging GSM-PLUS dataset. Furthermore, we observe strong cross-domain generalization, with an average accuracy gain of 5.8% on all other benchmarks, suggesting that our method captures general reasoning capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过大量数据的广泛训练，在推理方面取得了显著改进。然而，仅依赖额外数据进行改进正变得越来越不现实，凸显了模型在无需外部监督的情况下自主提升推理能力的需求。本文提出了一种名为Debate, Train, Evolve (DTE)的新颖无地面真值训练框架，该框架使用多代理辩论轨迹来进化单一语言模型。我们还引入了一种新的提示策略Reflect-Critique-Refine，通过明确指示代理批判和改进推理来提高辩论质量。在五个推理基准上的广泛评估显示，我们的DTE框架取得了显著改进，在具有挑战性的GSM-PLUS数据集上平均准确率提高了8.92%。此外，我们观察到强烈的跨领域泛化能力，在其他所有基准上平均准确率提高了5.8%，表明我们的方法捕捉到了通用的推理能力。 

---
# Shared Path: Unraveling Memorization in Multilingual LLMs through Language Similarities 

**Title (ZH)**: 共享路径：通过语言相似性揭开多语言LLM中的记忆机制 

**Authors**: Xiaoyu Luo, Yiyi Chen, Johannes Bjerva, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15722)  

**Abstract**: We present the first comprehensive study of Memorization in Multilingual Large Language Models (MLLMs), analyzing 95 languages using models across diverse model scales, architectures, and memorization definitions. As MLLMs are increasingly deployed, understanding their memorization behavior has become critical. Yet prior work has focused primarily on monolingual models, leaving multilingual memorization underexplored, despite the inherently long-tailed nature of training corpora. We find that the prevailing assumption, that memorization is highly correlated with training data availability, fails to fully explain memorization patterns in MLLMs. We hypothesize that treating languages in isolation - ignoring their similarities - obscures the true patterns of memorization. To address this, we propose a novel graph-based correlation metric that incorporates language similarity to analyze cross-lingual memorization. Our analysis reveals that among similar languages, those with fewer training tokens tend to exhibit higher memorization, a trend that only emerges when cross-lingual relationships are explicitly modeled. These findings underscore the importance of a language-aware perspective in evaluating and mitigating memorization vulnerabilities in MLLMs. This also constitutes empirical evidence that language similarity both explains Memorization in MLLMs and underpins Cross-lingual Transferability, with broad implications for multilingual NLP. 

**Abstract (ZH)**: 我们首次对多语言大型语言模型（MLLMs）中的记忆行为进行全面研究，分析了95种语言，并使用了不同规模、架构和记忆定义的各种模型。随着MLLMs的日益部署，理解其记忆行为变得至关重要。然而，以往的工作主要集中在单语言模型上，导致多语言记忆行为的研究不足，尽管训练语料库 inherently 呈长尾分布。我们发现，记忆与训练数据可用性的高度相关性这一主导假设，并不能完全解释MLLMs的记忆模式。我们假设将语言视为孤立个体，忽略它们的相似性，会掩盖真正的记忆模式。为了应对这一问题，我们提出了一种基于图的关联度量，该度量整合了语言相似性以分析跨语言记忆。我们的分析揭示，在相似语言中，训练令牌较少的语言更容易显示更高的记忆现象，只有在显式建模跨语言关系时这一趋势才得以显现。这些发现强调了在评估和缓解MLLMs的记忆漏洞时，采用语言 Awareness 观点的重要性。这也提供了实证证据，表明语言相似性不仅是MLLMs中记忆现象的解释，也是跨语言可迁移性的基石，对多语言自然语言处理有广泛影响。 

---
# HAMF: A Hybrid Attention-Mamba Framework for Joint Scene Context Understanding and Future Motion Representation Learning 

**Title (ZH)**: HAMF：一种结合注意力机制与Mamba框架的场景上下文理解与未来运动表示学习混合模型 

**Authors**: Xiaodong Mei, Sheng Wang, Jie Cheng, Yingbing Chen, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15703)  

**Abstract**: Motion forecasting represents a critical challenge in autonomous driving systems, requiring accurate prediction of surrounding agents' future trajectories. While existing approaches predict future motion states with the extracted scene context feature from historical agent trajectories and road layouts, they suffer from the information degradation during the scene feature encoding. To address the limitation, we propose HAMF, a novel motion forecasting framework that learns future motion representations with the scene context encoding jointly, to coherently combine the scene understanding and future motion state prediction. We first embed the observed agent states and map information into 1D token sequences, together with the target multi-modal future motion features as a set of learnable tokens. Then we design a unified Attention-based encoder, which synergistically combines self-attention and cross-attention mechanisms to model the scene context information and aggregate future motion features jointly. Complementing the encoder, we implement the Mamba module in the decoding stage to further preserve the consistency and correlations among the learned future motion representations, to generate the accurate and diverse final trajectories. Extensive experiments on Argoverse 2 benchmark demonstrate that our hybrid Attention-Mamba model achieves state-of-the-art motion forecasting performance with the simple and lightweight architecture. 

**Abstract (ZH)**: 运动预测是自动驾驶系统中的一个关键挑战，要求准确预测周围代理的未来轨迹。为了克服现有方法在场景特征编码过程中信息退化的局限性，我们提出了一种新颖的运动预测框架HAMF，该框架通过联合学习场景上下文编码和未来运动状态，协调地结合了场景理解与未来运动状态预测。我们首先将观测到的代理状态和地图信息嵌入到1Dtoken序列中，并将目标多模态未来运动特征作为一组可学习的token。然后，我们设计了一个统一的基于注意力的编码器，该编码器结合了自注意力和跨注意力机制，协同建模场景上下文信息并联合聚合未来运动特征。为了进一步保持学习到的未来运动表示的一致性和相关性，我们在解码阶段实现了Mamba模块，生成准确且多样化的最终轨迹。在Argoverse 2基准测试上进行的广泛实验表明，我们的混合Attention-Mamba模型在简单且轻量级的架构下实现了最先进的运动预测性能。 

---
# A Unified Theoretical Analysis of Private and Robust Offline Alignment: from RLHF to DPO 

**Title (ZH)**: 统一的理论分析：私人和 robust 离线对齐从 RLHF 到 DPO 

**Authors**: Xingyu Zhou, Yulian Wu, Francesco Orabona  

**Link**: [PDF](https://arxiv.org/pdf/2505.15694)  

**Abstract**: In this paper, we theoretically investigate the effects of noisy labels in offline alignment, with a focus on the interplay between privacy and robustness against adversarial corruption. Specifically, under linear modeling assumptions, we present a unified analysis covering both reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) under different privacy-corruption scenarios, such as Local differential privacy-then-Corruption (LTC), where human preference labels are privatized before being corrupted by an adversary, and Corruption-then-Local differential privacy (CTL), where labels are corrupted before privacy protection. Our analysis leverages a reduction framework that reduces the offline alignment problem under linear modeling assumptions to parameter estimation in logistic regression. This framework allows us to establish an interesting separation result between LTC and CTL, demonstrating that LTC presents a greater challenge than CTL in offline alignment, even under linear models. As important by-products, our findings also advance the state-of-the-art theoretical results in offline alignment under privacy-only or corruption-only scenarios. 

**Abstract (ZH)**: 在本论文中，我们从理论上研究了离线对齐中噪声标签的影响，重点关注隐私与对抗性腐蚀鲁棒性之间的交互作用。具体而言，在线性建模假设下，我们对局部差分隐私后再腐蚀（LTC）和先腐蚀再局部差分隐私（CTL）等不同隐私-腐蚀场景下的强化学习从人类反馈（RLHF）和直接偏好优化（DPO）进行了统一分析。我们的分析利用了一个约简框架，将在线性建模假设下的离线对齐问题约化为逻辑回归中的参数估计问题。该框架使我们能够建立LTC和CTL之间有趣的分离结果，即使在在线性模型下，LTC也比CTL在离线对齐中更具挑战性。作为重要的副产品，我们的发现还在仅隐私或仅腐蚀场景下的离线对齐的最先进理论结果方面有所推进。 

---
# Discovering Pathology Rationale and Token Allocation for Efficient Multimodal Pathology Reasoning 

**Title (ZH)**: 发现病理推理的病理学依据和令牌分配以实现高效多模态病理学推理 

**Authors**: Zhe Xu, Cheng Jin, Yihui Wang, Ziyi Liu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15687)  

**Abstract**: Multimodal pathological image understanding has garnered widespread interest due to its potential to improve diagnostic accuracy and enable personalized treatment through integrated visual and textual data. However, existing methods exhibit limited reasoning capabilities, which hamper their ability to handle complex diagnostic scenarios. Additionally, the enormous size of pathological images leads to severe computational burdens, further restricting their practical deployment. To address these limitations, we introduce a novel bilateral reinforcement learning framework comprising two synergistic branches. One reinforcement branch enhances the reasoning capability by enabling the model to learn task-specific decision processes, i.e., pathology rationales, directly from labels without explicit reasoning supervision. While the other branch dynamically allocates a tailored number of tokens to different images based on both their visual content and task context, thereby optimizing computational efficiency. We apply our method to various pathological tasks such as visual question answering, cancer subtyping, and lesion detection. Extensive experiments show an average +41.7 absolute performance improvement with 70.3% lower inference costs over the base models, achieving both reasoning accuracy and computational efficiency. 

**Abstract (ZH)**: 多模态病理图像理解因其实现诊断准确性和个性化治疗的潜力而引起了广泛关注，但由于现有方法的推理能力有限，无法有效处理复杂的诊断场景。此外，病理图像的巨大尺寸导致严重的计算负担，进一步限制了其实用部署。为解决这些限制，我们提出了一种新颖的双边强化学习框架，该框架包含两个协同的分支。一个强化分支通过使模型直接从标签中学习任务特定的决策过程，即病理推理，来增强推理能力，无需显式的推理监督。而另一个分支则根据图像的视觉内容和任务上下文动态分配定制数量的标记，从而优化计算效率。我们将该方法应用于各种病理任务，如视觉问答、癌症亚型分类和病灶检测。广泛的实验结果显示，与基线模型相比，我们的方法在平均绝对性能上提高了41.7%，同时将推理成本降低了70.3%，实现了推理准确性和计算效率的双重提升。 

---
# A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability 

**Title (ZH)**: 联邦分割框架for LLMs：安全、效率与适应性 

**Authors**: Zishuai Zhang, Hainan Zhang, Jiaying Zheng, Ziwei Wang, Yongxin Tong, Jin Dong, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15683)  

**Abstract**: Private data is typically larger and of higher quality than public data, offering great potential to improve LLM. However, its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based split learning model has emerged, offloading most model parameters to the server while retaining only the embedding and output layers on clients to ensure privacy. However, it still faces significant challenges in security, efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks, leading to reverse engineering of private data; 2) the autoregressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a secure, efficient, and adaptive federated split framework based on LLaMA2. First, we place some input and output blocks on the local client and inject Gaussian noise into forward-pass hidden states, enabling secure end-to-end propagation. Second, we employ client-batch and server-hierarchical strategies to achieve parallel training, along with attention-mask compression and KV cache mechanisms to accelerate inference, reducing communication costs effectively. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements and hardware limitations. Experiments on NLU, summarization and conversational QA tasks show that FL-LLaMA maintains performance comparable to centralized LLaMA2, and achieves up to 2x train speedups and 8x inference speedups. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FL-LLaMA in security and adaptability. 

**Abstract (ZH)**: 基于LLaMA2的FL-LLaMA：一种安全、高效且适应性强的联邦分学习框架 

---
# UniErase: Unlearning Token as a Universal Erasure Primitive for Language Models 

**Title (ZH)**: UniErase: 作为一种通用擦除原语的语言模型去学习 token 

**Authors**: Miao Yu, Liang Lin, Guibin Zhang, Xinfeng Li, Junfeng Fang, Ningyu Zhang, Kun Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15674)  

**Abstract**: Large language models require iterative updates to address challenges such as knowledge conflicts and outdated information (e.g., incorrect, private, or illegal contents). Machine unlearning provides a systematic methodology for targeted knowledge removal from trained models, enabling elimination of sensitive information influences. However, mainstream fine-tuning-based unlearning methods often fail to balance unlearning efficacy and model ability, frequently resulting in catastrophic model collapse under extensive knowledge removal. Meanwhile, in-context unlearning, which relies solely on contextual prompting without modifying the model's intrinsic mechanisms, suffers from limited generalizability and struggles to achieve true unlearning. In this work, we introduce UniErase, a novel unlearning paradigm that employs learnable parametric suffix (unlearning token) to steer language models toward targeted forgetting behaviors. UniErase operates through two key phases: (I) an optimization stage that binds desired unlearning outputs to the model's autoregressive probability distribution via token optimization, followed by (II) a lightweight model editing phase that activates the learned token to probabilistically induce specified forgetting objective. Serving as a new research direction for token learning to induce unlearning target, UniErase achieves state-of-the-art (SOTA) performance across batch, sequential, and precise unlearning under fictitious and real-world knowledge settings. Remarkably, in terms of TOFU benchmark, UniErase, modifying only around 3.66% of the LLM parameters, outperforms previous forgetting SOTA baseline by around 4.01 times for model ability with even better unlearning efficacy. Similarly, UniErase, maintaining more ability, also surpasses previous retaining SOTA by 35.96% for unlearning efficacy, showing dual top-tier performances in current unlearing domain. 

**Abstract (ZH)**: UniErase：一种用于诱导遗忘目标的可学习参数后缀新范式 

---
# Enhancing Monte Carlo Dropout Performance for Uncertainty Quantification 

**Title (ZH)**: 增强蒙特卡洛dropout方法以提高不确定性量化性能 

**Authors**: Hamzeh Asgharnezhad, Afshar Shamsi, Roohallah Alizadehsani, Arash Mohammadi, Hamid Alinejad-Rokny  

**Link**: [PDF](https://arxiv.org/pdf/2505.15671)  

**Abstract**: Knowing the uncertainty associated with the output of a deep neural network is of paramount importance in making trustworthy decisions, particularly in high-stakes fields like medical diagnosis and autonomous systems. Monte Carlo Dropout (MCD) is a widely used method for uncertainty quantification, as it can be easily integrated into various deep architectures. However, conventional MCD often struggles with providing well-calibrated uncertainty estimates. To address this, we introduce innovative frameworks that enhances MCD by integrating different search solutions namely Grey Wolf Optimizer (GWO), Bayesian Optimization (BO), and Particle Swarm Optimization (PSO) as well as an uncertainty-aware loss function, thereby improving the reliability of uncertainty quantification. We conduct comprehensive experiments using different backbones, namely DenseNet121, ResNet50, and VGG16, on various datasets, including Cats vs. Dogs, Myocarditis, Wisconsin, and a synthetic dataset (Circles). Our proposed algorithm outperforms the MCD baseline by 2-3% on average in terms of both conventional accuracy and uncertainty accuracy while achieving significantly better calibration. These results highlight the potential of our approach to enhance the trustworthiness of deep learning models in safety-critical applications. 

**Abstract (ZH)**: 了解与深度神经网络输出相关的不确定性对于在医疗诊断和自主系统等高风险领域做出可信赖的决策至关重要。为了解决传统蒙特卡洛Dropout (MCD) 提供校准不确定性估计的挑战，我们提出了通过集成灰狼优化器（GWO）、贝叶斯优化（BO）、粒子 swarm 优化（PSO）和一种新的不确定性感知损失函数来增强MCD的创新框架。我们的实验证实在不同骨干网络（DenseNet121、ResNet50和VGG16）和多个数据集（Cats vs. Dogs、Myocarditis、Wisconsin和合成数据集Circles）上，提出的算法在传统准确性和不确定性准确性的综合表现上比MCD基线平均高出2-3%，并且在校准方面表现出显著改善。这些结果表明，我们的方法有可能增强深度学习模型在安全关键应用中的可信赖性。 

---
# Neural Quantum Digital Twins for Optimizing Quantum Annealing 

**Title (ZH)**: 神经量子数字双胞胎用于优化量子退火 

**Authors**: Jianlong Lu, Hanqiu Peng, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15662)  

**Abstract**: Quantum annealers have shown potential in addressing certain combinatorial optimization problems, though their performance is often limited by scalability and errors rates. In this work, we propose a Neural Quantum Digital Twin (NQDT) framework that reconstructs the energy landscape of quantum many-body systems relevant to quantum annealing. The digital twin models both ground and excited state dynamics, enabling detailed simulation of the adiabatic evolution process. We benchmark NQDT on systems with known analytical solutions and demonstrate that it accurately captures key quantum phenomena, including quantum criticality and phase transitions. Leveraging this framework, one can identify optimal annealing schedules that minimize excitation-related errors. These findings highlight the utility of neural network-based digital twins as a diagnostic and optimization tool for improving the performance of quantum annealers. 

**Abstract (ZH)**: 量子退火器在解决某些组合优化问题方面展现了潜力，尽管其性能常常受到可扩展性和错误率的限制。本工作中，我们提出了一种神经量子数字孪生（NQDT）框架，用于重构与量子退火相关的量子多体系统的能量景观。数字孪生模型 Both 基态和激发态的动力学，从而实现对绝热演化过程的详细模拟。我们通过已知解析解的系统对 NQDT 进行基准测试，并证明它可以准确捕捉到关键的量子现象，包括量子临界性和相变。利用这一框架，可以识别出减少激发相关错误的最优退火时间表。这些发现突显了基于神经网络的数字孪生作为诊断和优化工具，以提高量子退火器性能的价值。 

---
# LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought 

**Title (ZH)**: LCDB 1.1: 一个表明学习曲线比以往认为的更为不良的数据库 

**Authors**: Cheng Yan, Felix Mohr, Tom Viering  

**Link**: [PDF](https://arxiv.org/pdf/2505.15657)  

**Abstract**: Sample-wise learning curves plot performance versus training set size. They are useful for studying scaling laws and speeding up hyperparameter tuning and model selection. Learning curves are often assumed to be well-behaved: monotone (i.e. improving with more data) and convex. By constructing the Learning Curves Database 1.1 (LCDB 1.1), a large-scale database with high-resolution learning curves, we show that learning curves are less often well-behaved than previously thought. Using statistically rigorous methods, we observe significant ill-behavior in approximately 14% of the learning curves, almost twice as much as in previous estimates. We also identify which learners are to blame and show that specific learners are more ill-behaved than others. Additionally, we demonstrate that different feature scalings rarely resolve ill-behavior. We evaluate the impact of ill-behavior on downstream tasks, such as learning curve fitting and model selection, and find it poses significant challenges, underscoring the relevance and potential of LCDB 1.1 as a challenging benchmark for future research. 

**Abstract (ZH)**: 样本级别的学习曲线绘制性能与训练集大小的关系。它们对于研究缩放定律并加速超参数调整和模型选择非常有用。通常假定学习曲线行为良好：单调（即更多的数据意味着改进）且凹形。通过构建Learning Curves Database 1.1（LCDB 1.1），一个具有高分辨率学习曲线的大规模数据库，我们证明了与先前认为的相比，学习曲线远不如预期行为良好。使用统计上严谨的方法，我们观察到约14%的学习曲线表现出显著的不良行为，几乎是之前估计的两倍。我们还确定了哪些学习算法是罪魁祸首，并展示了特定的学习算法比其他算法更不良。此外，我们证明不同的特征缩放很少能解决不良行为问题。我们评估了不良行为对下游任务（如学习曲线拟合和模型选择）的影响，并发现这对未来研究构成了显著挑战，突显了LCDB 1.1作为具有挑战性的基准评估的现实意义和潜力。 

---
# Second-Order Convergence in Private Stochastic Non-Convex Optimization 

**Title (ZH)**: 私有化随机非凸优化的二阶收敛性 

**Authors**: Youming Tao, Zuyuan Zhang, Dongxiao Yu, Xiuzhen Cheng, Falko Dressler, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15647)  

**Abstract**: We investigate the problem of finding second-order stationary points (SOSP) in differentially private (DP) stochastic non-convex optimization. Existing methods suffer from two key limitations: (i) inaccurate convergence error rate due to overlooking gradient variance in the saddle point escape analysis, and (ii) dependence on auxiliary private model selection procedures for identifying DP-SOSP, which can significantly impair utility, particularly in distributed settings. To address these issues, we propose a generic perturbed stochastic gradient descent (PSGD) framework built upon Gaussian noise injection and general gradient oracles. A core innovation of our framework is using model drift distance to determine whether PSGD escapes saddle points, ensuring convergence to approximate local minima without relying on second-order information or additional DP-SOSP identification. By leveraging the adaptive DP-SPIDER estimator as a specific gradient oracle, we develop a new DP algorithm that rectifies the convergence error rates reported in prior work. We further extend this algorithm to distributed learning with arbitrarily heterogeneous data, providing the first formal guarantees for finding DP-SOSP in such settings. Our analysis also highlights the detrimental impacts of private selection procedures in distributed learning under high-dimensional models, underscoring the practical benefits of our design. Numerical experiments on real-world datasets validate the efficacy of our approach. 

**Abstract (ZH)**: 差分隐私环境下非凸优化中第二阶稳定点的寻找：通用扰动随机梯度下降方法及其应用 

---
# FragFake: A Dataset for Fine-Grained Detection of Edited Images with Vision Language Models 

**Title (ZH)**: FragFake: 一种用于视觉语言模型细粒度检测编辑图像的数据集 

**Authors**: Zhen Sun, Ziyi Zhang, Zeren Luo, Zeyang Sha, Tianshuo Cong, Zheng Li, Shiwen Cui, Weiqiang Wang, Jiaheng Wei, Xinlei He, Qi Li, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15644)  

**Abstract**: Fine-grained edited image detection of localized edits in images is crucial for assessing content authenticity, especially given that modern diffusion models and image editing methods can produce highly realistic manipulations. However, this domain faces three challenges: (1) Binary classifiers yield only a global real-or-fake label without providing localization; (2) Traditional computer vision methods often rely on costly pixel-level annotations; and (3) No large-scale, high-quality dataset exists for modern image-editing detection techniques. To address these gaps, we develop an automated data-generation pipeline to create FragFake, the first dedicated benchmark dataset for edited image detection, which includes high-quality images from diverse editing models and a wide variety of edited objects. Based on FragFake, we utilize Vision Language Models (VLMs) for the first time in the task of edited image classification and edited region localization. Experimental results show that fine-tuned VLMs achieve higher average Object Precision across all datasets, significantly outperforming pretrained models. We further conduct ablation and transferability analyses to evaluate the detectors across various configurations and editing scenarios. To the best of our knowledge, this work is the first to reformulate localized image edit detection as a vision-language understanding task, establishing a new paradigm for the field. We anticipate that this work will establish a solid foundation to facilitate and inspire subsequent research endeavors in the domain of multimodal content authenticity. 

**Abstract (ZH)**: 细粒度局部编辑图像检测对于评估内容真实性至关重要，尤其是在现代扩散模型和图像编辑方法可以产生高度真实的伪造时。然而，这一领域面临三个挑战：（1）二分类器仅提供全局真实或伪造标签而不提供定位；（2）传统计算机视觉方法通常依赖于昂贵的像素级注释；（3）不存在大规模高质量的数据集用于现代图像编辑检测技术。为了解决这些缺口，我们开发了一个自动数据生成管道，以创建FragFake，这是首个专门用于编辑图像检测的基准数据集，其中包括来自多种编辑模型的高质量图像以及多种编辑对象。基于FragFake，我们首次将视觉语言模型（VLMs）应用于编辑图像分类和编辑区域定位任务。实验结果表明，微调后的VLMs在所有数据集上实现了更高的平均对象精确度，显著优于预训练模型。我们进一步进行了消融分析和可迁移性分析，以评估在各种配置和编辑场景下的检测器性能。据我们所知，这项工作是首次将局部图像编辑检测重新定义为视觉语言理解任务，确立了该领域的全新范式。我们期望这项工作将为多模态内容真实性领域后续研究奠定坚实基础并提供启发。 

---
# Listen to the Context: Towards Faithful Large Language Models for Retrieval Augmented Generation on Climate Questions 

**Title (ZH)**: 倾听上下文：面向气候问题检索增强生成的忠实大型语言模型 

**Authors**: David Thulke, Jakob Kemmler, Christian Dugast, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2505.15633)  

**Abstract**: Large language models that use retrieval augmented generation have the potential to unlock valuable knowledge for researchers, policymakers, and the public by making long and technical climate-related documents more accessible. While this approach can help alleviate factual hallucinations by relying on retrieved passages as additional context, its effectiveness depends on whether the model's output remains faithful to these passages. To address this, we explore the automatic assessment of faithfulness of different models in this setting. We then focus on ClimateGPT, a large language model specialised in climate science, to examine which factors in its instruction fine-tuning impact the model's faithfulness. By excluding unfaithful subsets of the model's training data, we develop ClimateGPT Faithful+, which achieves an improvement in faithfulness from 30% to 57% in supported atomic claims according to our automatic metric. 

**Abstract (ZH)**: 使用检索增强生成的大语言模型有可能通过使与气候相关的长篇技术文档更易于访问，为研究人员、 Policymakers 和公众解锁有价值的knowledge。虽然这种方法可以通过依赖检索段落作为额外背景信息来缓解事实幻觉的问题，但其效果取决于模型的输出是否忠于这些段落。为了解决这个问题，我们探索了在这种情境下自动评估不同模型忠实度的方法。然后，我们专注于专门从事气候科学的大型语言模型ClimateGPT，研究其指令微调中的哪些因素影响模型的忠实度。通过排除不忠实的模型训练数据子集，我们开发了ClimateGPT Faithful+，在我们自动评估指标的支持原子声明中，其忠实度从30%提升到57%。 

---
# Learn to Reason Efficiently with Adaptive Length-based Reward Shaping 

**Title (ZH)**: 基于自适应长度奖励塑形的有效推理学习 

**Authors**: Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15612)  

**Abstract**: Large Reasoning Models (LRMs) have shown remarkable capabilities in solving complex problems through reinforcement learning (RL), particularly by generating long reasoning traces. However, these extended outputs often exhibit substantial redundancy, which limits the efficiency of LRMs. In this paper, we investigate RL-based approaches to promote reasoning efficiency. Specifically, we first present a unified framework that formulates various efficient reasoning methods through the lens of length-based reward shaping. Building on this perspective, we propose a novel Length-bAsed StEp Reward shaping method (LASER), which employs a step function as the reward, controlled by a target length. LASER surpasses previous methods, achieving a superior Pareto-optimal balance between performance and efficiency. Next, we further extend LASER based on two key intuitions: (1) The reasoning behavior of the model evolves during training, necessitating reward specifications that are also adaptive and dynamic; (2) Rather than uniformly encouraging shorter or longer chains of thought (CoT), we posit that length-based reward shaping should be difficulty-aware i.e., it should penalize lengthy CoTs more for easy queries. This approach is expected to facilitate a combination of fast and slow thinking, leading to a better overall tradeoff. The resulting method is termed LASER-D (Dynamic and Difficulty-aware). Experiments on DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and DeepSeek-R1-Distill-Qwen-32B show that our approach significantly enhances both reasoning performance and response length efficiency. For instance, LASER-D and its variant achieve a +6.1 improvement on AIME2024 while reducing token usage by 63%. Further analysis reveals our RL-based compression produces more concise reasoning patterns with less redundant "self-reflections". Resources are at this https URL. 

**Abstract (ZH)**: 基于强化学习的大推理模型效率提升方法 

---
# From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning 

**Title (ZH)**: 从问题解决到教学问题解决：通过强化学习使大语言模型与教学方法相alignment 

**Authors**: David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido Hakimi, Iryna Gurevych, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15607)  

**Abstract**: Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以变革教育，但它们对直接问答的优化往往会削弱有效教学所需的战略性保留答案。为解决这一问题，我们提出了一种基于在线强化学习（RL）的对齐框架，该框架可以通过强调教学质量和引导性问题求解，快速将LLMs转换为有效的辅导工具，而不仅仅是直接给出答案。我们使用该方法训练了一个不依赖人类标注的7B参数辅导模型，其性能接近或堪比大型私有模型如LearnLM。我们引入了一种可控的奖励权重调整方法，以平衡教学支持和学生求解准确性，从而能够追踪这两项目标之间的帕累托前沿。我们的模型在保持推理能力方面优于单轮SFT基线，并且可以选择通过思维标签增强解释性，以揭示模型的教学规划。 

---
# Exploring LLM-Generated Feedback for Economics Essays: How Teaching Assistants Evaluate and Envision Its Use 

**Title (ZH)**: 探索LLM生成的反馈经济学论文评语：教学助理的评价及其使用展望 

**Authors**: Xinyi Lu, Aditya Mahesh, Zejia Shen, Mitchell Dudley, Larissa Sano, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15596)  

**Abstract**: This project examines the prospect of using AI-generated feedback as suggestions to expedite and enhance human instructors' feedback provision. In particular, we focus on understanding the teaching assistants' perspectives on the quality of AI-generated feedback and how they may or may not utilize AI feedback in their own workflows. We situate our work in a foundational college Economics class, which has frequent short essay assignments. We developed an LLM-powered feedback engine that generates feedback on students' essays based on grading rubrics used by the teaching assistants (TAs). To ensure that TAs can meaningfully critique and engage with the AI feedback, we had them complete their regular grading jobs. For a randomly selected set of essays that they had graded, we used our feedback engine to generate feedback and displayed the feedback as in-text comments in a Word document. We then performed think-aloud studies with 5 TAs over 20 1-hour sessions to have them evaluate the AI feedback, contrast the AI feedback with their handwritten feedback, and share how they envision using the AI feedback if they were offered as suggestions. The study highlights the importance of providing detailed rubrics for AI to generate high-quality feedback for knowledge-intensive essays. TAs considered that using AI feedback as suggestions during their grading could expedite grading, enhance consistency, and improve overall feedback quality. We discuss the importance of decomposing the feedback generation task into steps and presenting intermediate results, in order for TAs to use the AI feedback. 

**Abstract (ZH)**: 本项目考察使用AI生成的反馈作为建议以加速和提升人类教师反馈提供的可能性。特别地，我们关注教学助手对AI生成反馈质量的看法，以及他们在工作流程中是否会或不会利用AI反馈。我们将工作置于一个基础大学经济学课程中，该课程有频繁的短 essays 作业。我们开发了一个基于大语言模型的反馈引擎，根据教学助手（TAs）使用的评分标准为学生论文生成反馈。为了确保教学助手能够有意义地批判和参与AI反馈，我们让他们完成了常规的批改工作。对他们的部分批改作业，我们使用反馈引擎生成反馈，并在Word文档中作为文本评论显示。然后，我们在20次每场时长为1小时的口头思考研究中与5位教学助手对AI反馈进行评估，将AI反馈与手写反馈进行对比，并分享如果提供作为建议时他们如何使用AI反馈的设想。研究强调了为AI提供详细的评分标准以生成高质量知识密集型论文反馈的重要性。教学助手认为在批改作业时利用AI反馈作为建议可以加速批改、增强一致性和提高整体反馈质量。我们讨论了将反馈生成任务分解成步骤并展示中间结果的重要性，以便教学助手使用AI反馈。 

---
# Beyond Classification: Evaluating Diffusion Denoised Smoothing for Security-Utility Trade off 

**Title (ZH)**: 超越分类：评价去噪扩散平滑的安全-效用权衡评估 

**Authors**: Yury Belousov, Brian Pulfer, Vitaliy Kinakh, Slava Voloshynovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2505.15594)  

**Abstract**: While foundation models demonstrate impressive performance across various tasks, they remain vulnerable to adversarial inputs. Current research explores various approaches to enhance model robustness, with Diffusion Denoised Smoothing emerging as a particularly promising technique. This method employs a pretrained diffusion model to preprocess inputs before model inference. Yet, its effectiveness remains largely unexplored beyond classification. We aim to address this gap by analyzing three datasets with four distinct downstream tasks under three different adversarial attack algorithms. Our findings reveal that while foundation models maintain resilience against conventional transformations, applying high-noise diffusion denoising to clean images without any distortions significantly degrades performance by as high as 57%. Low-noise diffusion settings preserve performance but fail to provide adequate protection across all attack types. Moreover, we introduce a novel attack strategy specifically targeting the diffusion process itself, capable of circumventing defenses in the low-noise regime. Our results suggest that the trade-off between adversarial robustness and performance remains a challenge to be addressed. 

**Abstract (ZH)**: 基础模型在各种任务中展现出令人印象深刻的性能，但仍易受对抗输入的影响。现有研究探索了多种增强模型鲁棒性的方法，其中去噪扩散平滑技术尤具前景。该方法利用预训练的扩散模型在模型推理前对输入进行预处理。然而，该方法在分类以外的任务中的有效性尚待充分探索。我们旨在通过在三个数据集上分析四种下游任务和三种不同的对抗攻击算法，来填补这一空白。研究表明，尽管基础模型在传统变换面前保持了韧性，但在干净图像上应用高噪声去噪扩散处理会导致性能下降高达57%。低噪声扩散设置则能够保留性能，但在所有攻击类型的防护上仍不够充分。此外，我们引入了一种针对扩散过程本身的新型攻击策略，能够在低噪声环境下绕过防御措施。我们的结果表明，对抗鲁棒性与性能之间的权衡仍然是一个需要解决的挑战。 

---
# World Models as Reference Trajectories for Rapid Motor Adaptation 

**Title (ZH)**: 世界模型作为快速运动适应的参考轨迹 

**Authors**: Carlos Stein Brito, Daniel McNamee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15589)  

**Abstract**: Deploying learned control policies in real-world environments poses a fundamental challenge. When system dynamics change unexpectedly, performance degrades until models are retrained on new data. We introduce Reflexive World Models (RWM), a dual control framework that uses world model predictions as implicit reference trajectories for rapid adaptation. Our method separates the control problem into long-term reward maximization through reinforcement learning and robust motor execution through rapid latent control. This dual architecture achieves significantly faster adaptation with low online computational cost compared to model-based RL baselines, while maintaining near-optimal performance. The approach combines the benefits of flexible policy learning through reinforcement learning with rapid error correction capabilities, providing a principled approach to maintaining performance in high-dimensional continuous control tasks under varying dynamics. 

**Abstract (ZH)**: 将学习到的控制策略部署到现实世界环境中是一项基本挑战。当系统动力学意外变化时，在重新训练模型于新数据之前，性能会下降。我们引入了反射世界模型（Reflexive World Models, RWM），这是一种双控制框架，使用世界模型预测作为隐式参考轨迹以实现快速适应。我们的方法将控制问题分为通过强化学习实现的长期奖励最大化和通过快速潜空间控制实现的稳健运动执行。该双架构相比于基于模型的强化学习基线，在线计算成本更低的同时实现了显著更快的适应，并维持了接近最优的性能。该方法结合了通过强化学习实现的灵活策略学习优势和快速错误校正能力，提供了一种在不同动力学条件下保持高性能的原理化方法。 

---
# UWSAM: Segment Anything Model Guided Underwater Instance Segmentation and A Large-scale Benchmark Dataset 

**Title (ZH)**: UWSAM：段Anything模型引导的水下实例分割及大规模基准数据集 

**Authors**: Hua Li, Shijie Lian, Zhiyuan Li, Runmin Cong, Sam Kwong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15581)  

**Abstract**: With recent breakthroughs in large-scale modeling, the Segment Anything Model (SAM) has demonstrated significant potential in a variety of visual applications. However, due to the lack of underwater domain expertise, SAM and its variants face performance limitations in end-to-end underwater instance segmentation tasks, while their higher computational requirements further hinder their application in underwater scenarios. To address this challenge, we propose a large-scale underwater instance segmentation dataset, UIIS10K, which includes 10,048 images with pixel-level annotations for 10 categories. Then, we introduce UWSAM, an efficient model designed for automatic and accurate segmentation of underwater instances. UWSAM efficiently distills knowledge from the SAM ViT-Huge image encoder into the smaller ViT-Small image encoder via the Mask GAT-based Underwater Knowledge Distillation (MG-UKD) method for effective visual representation learning. Furthermore, we design an End-to-end Underwater Prompt Generator (EUPG) for UWSAM, which automatically generates underwater prompts instead of explicitly providing foreground points or boxes as prompts, thus enabling the network to locate underwater instances accurately for efficient segmentation. Comprehensive experimental results show that our model is effective, achieving significant performance improvements over state-of-the-art methods on multiple underwater instance datasets. Datasets and codes are available at this https URL. 

**Abstract (ZH)**: 基于最近大规模模型的突破，段 Anything 模型 (SAM) 在多种视觉应用中展示了显著的潜力。然而，由于缺乏水下领域的专业知识，SAM 及其变体在端到端水下实例分割任务中面临性能限制，而其更高的计算要求进一步妨碍了其在水下场景中的应用。为应对这一挑战，我们提出了一个大规模水下实例分割数据集 UIIS10K，其中包括10,048张具有10个类别像素级注释的图像。随后，我们引入了 UWSAM，一种针对水下实例自动且准确分割的高效模型。UWSAM 通过基于 Mask GAT 的水下知识蒸馏 (MG-UKD) 方法，将 SAM ViT-Huge 图像编码器的知识高效地转移到较小的 ViT-Small 图像编码器中，从而实现有效的视觉表示学习。此外，我们为 UWSAM 设计了一个端到端水下提示生成器 (EUPG)，它能够自动生成水下提示，而无需显式提供前景点或框作为提示，从而使网络能够准确地定位水下实例以实现高效的分割。综合实验结果表明，我们的模型是有效的，在多个水下实例数据集上实现了相对于先进方法的显著性能提升。数据集和代码可通过以下链接获取。 

---
# Bridging the Domain Gap in Equation Distillation with Reinforcement Feedback 

**Title (ZH)**: 用强化反馈bridging方程蒸馏领域的差距 

**Authors**: Wangyang Ying, Haoyue Bai, Nanxu Gong, Xinyuan Wang, Sixun Dong, Haifeng Chen, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15572)  

**Abstract**: The data-to-equation (Data2Eqn) task aims to discover interpretable mathematical equations that map observed values to labels, offering physical insights and broad applicability across academic and industrial domains. Genetic programming and traditional deep learning-based approaches suffer from search inefficiency and poor generalization on small task-specific datasets. Foundation models showed promise in this area, but existing approaches suffer from: 1) They are pretrained on general-purpose data distributions, making them less effective for domain-specific tasks; and 2) their training objectives focus on token-level alignment, overlooking mathematical semantics, which can lead to inaccurate equations. To address these issues, we aim to enhance the domain adaptability of foundation models for Data2Eqn tasks. In this work, we propose a reinforcement learning-based finetuning framework that directly optimizes the generation policy of a pretrained model through reward signals derived from downstream numerical fitness. Our method allows the model to adapt to specific and complex data distributions and generate mathematically meaningful equations. Extensive experiments demonstrate that our approach improves both the accuracy and robustness of equation generation under complex distributions. 

**Abstract (ZH)**: 数据到方程（Data2Eqn）任务旨在发现可解释的数学方程，将观察到的值映射到标签，提供物理洞察并广泛适用于学术和工业领域。遗传编程和传统的基于深度学习的方法在小型任务特定数据集中存在搜索效率低和泛化能力差的问题。基础模型在这方面的应用前景曾被看好，但现有方法面临以下挑战：1）它们在通用数据分布上进行预训练，使其对于特定领域任务效果不佳；2）它们的训练目标集中在token级对齐，忽略了数学语义，可能导致方程不准确。为解决这些问题，我们旨在提升基础模型在Data2Eqn任务中的领域适应性。在本工作中，我们提出了一种基于强化学习的微调框架，通过从下游数值适应性中获取的奖励信号直接优化预训练模型的生成策略。该方法允许模型适应特定且复杂的数据分布，并生成具有数学意义的方程。广泛实验表明，我们的方法在复杂分布下提高了方程生成的准确性和鲁棒性。 

---
# Moonbeam: A MIDI Foundation Model Using Both Absolute and Relative Music Attributes 

**Title (ZH)**: Moonbeam：一种结合绝对和相对音乐属性的MIDI基础模型 

**Authors**: Zixun Guo, Simon Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2505.15559)  

**Abstract**: Moonbeam is a transformer-based foundation model for symbolic music, pretrained on a large and diverse collection of MIDI data totaling 81.6K hours of music and 18 billion tokens. Moonbeam incorporates music-domain inductive biases by capturing both absolute and relative musical attributes through the introduction of a novel domain-knowledge-inspired tokenization method and Multidimensional Relative Attention (MRA), which captures relative music information without additional trainable parameters. Leveraging the pretrained Moonbeam, we propose 2 finetuning architectures with full anticipatory capabilities, targeting 2 categories of downstream tasks: symbolic music understanding and conditional music generation (including music infilling). Our model outperforms other large-scale pretrained music models in most cases in terms of accuracy and F1 score across 3 downstream music classification tasks on 4 datasets. Moreover, our finetuned conditional music generation model outperforms a strong transformer baseline with a REMI-like tokenizer. We open-source the code, pretrained model, and generated samples on Github. 

**Abstract (ZH)**: 基于Transformer的Moonbeam符号音乐基础模型：预训练于81,600小时的MIDI数据和180亿个标记，并结合音乐领域先验知识和多维相对注意机制实现音乐属性捕捉 

---
# Robo-DM: Data Management For Large Robot Datasets 

**Title (ZH)**: Robo-DM：大型机器人数据集的管理 

**Authors**: Kaiyuan Chen, Letian Fu, David Huang, Yanxiang Zhang, Lawrence Yunliang Chen, Huang Huang, Kush Hari, Ashwin Balakrishna, Ted Xiao, Pannag R Sanketi, John Kubiatowicz, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.15558)  

**Abstract**: Recent results suggest that very large datasets of teleoperated robot demonstrations can be used to train transformer-based models that have the potential to generalize to new scenes, robots, and tasks. However, curating, distributing, and loading large datasets of robot trajectories, which typically consist of video, textual, and numerical modalities - including streams from multiple cameras - remains challenging. We propose Robo-DM, an efficient open-source cloud-based data management toolkit for collecting, sharing, and learning with robot data. With Robo-DM, robot datasets are stored in a self-contained format with Extensible Binary Meta Language (EBML). Robo-DM can significantly reduce the size of robot trajectory data, transfer costs, and data load time during training. Compared to the RLDS format used in OXE datasets, Robo-DM's compression saves space by up to 70x (lossy) and 3.5x (lossless). Robo-DM also accelerates data retrieval by load-balancing video decoding with memory-mapped decoding caches. Compared to LeRobot, a framework that also uses lossy video compression, Robo-DM is up to 50x faster when decoding sequentially. We physically evaluate a model trained by Robo-DM with lossy compression, a pick-and-place task, and In-Context Robot Transformer. Robo-DM uses 75x compression of the original dataset and does not suffer reduction in downstream task accuracy. 

**Abstract (ZH)**: 最近的研究表明，大型的遥操作机器人演示数据集可以用于训练具有潜在泛化能力的变压器模型，这些模型可以应用于新的场景、机器人和任务。然而，收集、分发和加载包含视频、文本和数值等多种模态的大型机器人轨迹数据集仍然具有挑战性。我们提出Robo-DM，这是一种高效的开源云数据管理工具箱，用于收集、共享和学习机器人数据。使用Robo-DM，机器人数据集以扩展二进制元数据语言（EBML）格式存储，可以显著减小机器人轨迹数据的大小，降低传输成本，并在训练期间减少数据加载时间。与OXE数据集中使用的RLDS格式相比，Robo-DM的压缩在有损压缩情况下可节省多达70倍的空间，在无损压缩情况下可节省3.5倍的空间。Robo-DM还通过视频解码负载均衡和内存映射解码缓存加速数据检索。与使用有损视频压缩的LeRobot框架相比，Robo-DM在逐个解码时速度可提高50倍。我们通过使用Robo-DM进行有损压缩训练的模型和夹取放置任务以及上下文机器人变换器，进行了物理评估。Robo-DM对原始数据集进行了75倍的压缩，而不会降低下游任务的准确性。 

---
# DayDreamer at CQs-Gen 2025: Generating Critical Questions through Argument Scheme Completion 

**Title (ZH)**: DayDreamer在CQs-Gen 2025：通过论据方案完成生成关键问题 

**Authors**: Wendi Zhou, Ameer Saadat-Yazdi, Nadin Kökciyan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15554)  

**Abstract**: Critical questions are essential resources to provoke critical thinking when encountering an argumentative text. We present our system for the Critical Questions Generation (CQs-Gen) Shared Task at ArgMining 2025. Our approach leverages large language models (LLMs) with chain-of-thought prompting to generate critical questions guided by Walton's argumentation schemes. For each input intervention, we conversationally prompt LLMs to instantiate the corresponding argument scheme template to first obtain structured arguments, and then generate relevant critical questions. Following this, we rank all the available critical questions by prompting LLMs to select the top 3 most helpful questions based on the original intervention text. This combination of structured argumentation theory and step-by-step reasoning enables the generation of contextually relevant and diverse critical questions. Our pipeline achieves competitive performance in the final test set, showing its potential to foster critical thinking given argumentative text and detect missing or uninformed claims. Code available at \href{this https URL}{DayDreamer}. 

**Abstract (ZH)**: 批判性问题生成对于在遇到论辩性文本时激发批判性思维至关重要。我们在ArgMining 2025的批判性问题生成（CQs-Gen）共享任务中介绍了我们的系统。我们的方法利用大型语言模型（LLMs）结合链式思维提示，生成由沃顿的论辩框架引导的批判性问题。对于每个输入干预，我们以对话性提示LLMs实例化相应的论辩框架模板，首先获得结构化论点，然后生成相关的批判性问题。随后，我们通过提示LLMs根据原始干预文本选择最有帮助的前3个批判性问题进行排名。这种结构化论辩理论与逐步推理的结合使生成上下文相关且多样的批判性问题成为可能。我们的管道在最终测试集上实现了竞争性的性能，显示出其在给定向辩性文本促进批判性思维和检测缺失或不充分论断方面的潜力。代码可在DayDreamer获取。 

---
# Social Bias in Popular Question-Answering Benchmarks 

**Title (ZH)**: 流行问答基准中的社会偏见 

**Authors**: Angelie Kraft, Judith Simon, Sonja Schimmler  

**Link**: [PDF](https://arxiv.org/pdf/2505.15553)  

**Abstract**: Question-answering (QA) and reading comprehension (RC) benchmarks are essential for assessing the capabilities of large language models (LLMs) in retrieving and reproducing knowledge. However, we demonstrate that popular QA and RC benchmarks are biased and do not cover questions about different demographics or regions in a representative way, potentially due to a lack of diversity of those involved in their creation. We perform a qualitative content analysis of 30 benchmark papers and a quantitative analysis of 20 respective benchmark datasets to learn (1) who is involved in the benchmark creation, (2) how social bias is addressed or prevented, and (3) whether the demographics of the creators and annotators correspond to particular biases in the content. Most analyzed benchmark papers provided insufficient information regarding the stakeholders involved in benchmark creation, particularly the annotators. Notably, just one of the benchmark papers explicitly reported measures taken to address social representation issues. Moreover, the data analysis revealed gender, religion, and geographic biases across a wide range of encyclopedic, commonsense, and scholarly benchmarks. More transparent and bias-aware QA and RC benchmark creation practices are needed to facilitate better scrutiny and incentivize the development of fairer LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的知识检索与再现能力评估：基于问答（QA）和阅读理解（RC）基准的多样性与偏见分析 

---
# Oversmoothing, "Oversquashing", Heterophily, Long-Range, and more: Demystifying Common Beliefs in Graph Machine Learning 

**Title (ZH)**: 过度平滑化、“过度挤压”、异质性、长范围连接及其更多：图机器学习中常见信仰的解析 

**Authors**: Adrian Arnaiz-Rodriguez, Federico Errica  

**Link**: [PDF](https://arxiv.org/pdf/2505.15547)  

**Abstract**: After a renaissance phase in which researchers revisited the message-passing paradigm through the lens of deep learning, the graph machine learning community shifted its attention towards a deeper and practical understanding of message-passing's benefits and limitations. In this position paper, we notice how the fast pace of progress around the topics of oversmoothing and oversquashing, the homophily-heterophily dichotomy, and long-range tasks, came with the consolidation of commonly accepted beliefs and assumptions that are not always true nor easy to distinguish from each other. We argue that this has led to ambiguities around the investigated problems, preventing researchers from focusing on and addressing precise research questions while causing a good amount of misunderstandings. Our contribution wants to make such common beliefs explicit and encourage critical thinking around these topics, supported by simple but noteworthy counterexamples. The hope is to clarify the distinction between the different issues and promote separate but intertwined research directions to address them. 

**Abstract (ZH)**: 关于过平滑、过压缩、同质性-异质性二分法及长范围任务的快速进展所伴随的共识性信念与假设的澄清：促进批判性思考与明确研究方向 

---
# Evaluate Bias without Manual Test Sets: A Concept Representation Perspective for LLMs 

**Title (ZH)**: 无需手动测试集评估偏差：面向LLM的概念表示视角 

**Authors**: Lang Gao, Kaiyang Wan, Wei Liu, Chenxi Wang, Zirui Song, Zixiang Xu, Yanbo Wang, Veselin Stoyanov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15524)  

**Abstract**: Bias in Large Language Models (LLMs) significantly undermines their reliability and fairness. We focus on a common form of bias: when two reference concepts in the model's concept space, such as sentiment polarities (e.g., "positive" and "negative"), are asymmetrically correlated with a third, target concept, such as a reviewing aspect, the model exhibits unintended bias. For instance, the understanding of "food" should not skew toward any particular sentiment. Existing bias evaluation methods assess behavioral differences of LLMs by constructing labeled data for different social groups and measuring model responses across them, a process that requires substantial human effort and captures only a limited set of social concepts. To overcome these limitations, we propose BiasLens, a test-set-free bias analysis framework based on the structure of the model's vector space. BiasLens combines Concept Activation Vectors (CAVs) with Sparse Autoencoders (SAEs) to extract interpretable concept representations, and quantifies bias by measuring the variation in representational similarity between the target concept and each of the reference concepts. Even without labeled data, BiasLens shows strong agreement with traditional bias evaluation metrics (Spearman correlation r > 0.85). Moreover, BiasLens reveals forms of bias that are difficult to detect using existing methods. For example, in simulated clinical scenarios, a patient's insurance status can cause the LLM to produce biased diagnostic assessments. Overall, BiasLens offers a scalable, interpretable, and efficient paradigm for bias discovery, paving the way for improving fairness and transparency in LLMs. 

**Abstract (ZH)**: BiasLens：基于模型向量空间结构的无测试集偏见分析框架 

---
# Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets 

**Title (ZH)**: Robo2VLM：源自大规模野外机器人操作数据集的视觉问答 

**Authors**: Kaiyuan Chen, Shuangyu Xie, Zehan Ma, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.15517)  

**Abstract**: Vision-Language Models (VLMs) acquire real-world knowledge and general reasoning ability through Internet-scale image-text corpora. They can augment robotic systems with scene understanding and task planning, and assist visuomotor policies that are trained on robot trajectory data. We explore the reverse paradigm - using rich, real, multi-modal robot trajectory data to enhance and evaluate VLMs. In this paper, we present Robo2VLM, a Visual Question Answering (VQA) dataset generation framework for VLMs. Given a human tele-operated robot trajectory, Robo2VLM derives ground-truth from non-visual and non-descriptive sensory modalities, such as end-effector pose, gripper aperture, and force sensing. Based on these modalities, it segments the robot trajectory into a sequence of manipulation phases. At each phase, Robo2VLM uses scene and interaction understanding to identify 3D properties of the robot, task goal, and the target object. The properties are used to generate representative VQA queries - images with textural multiple-choice questions - based on spatial, goal-conditioned, and interaction reasoning question templates. We curate Robo2VLM-1, a large-scale in-the-wild dataset with 684,710 questions covering 463 distinct scenes and 3,396 robotic manipulation tasks from 176k real robot trajectories. Results suggest that Robo2VLM-1 can benchmark and improve VLM capabilities in spatial and interaction reasoning. 

**Abstract (ZH)**: 基于机器人轨迹数据增强和评估视觉-语言模型的研究：Robo2VLM数据集生成框架 

---
# Explainable embeddings with Distance Explainer 

**Title (ZH)**: 可解释的嵌入模型与Distance Explainer 

**Authors**: Christiaan Meijer, E. G. Patrick Bos  

**Link**: [PDF](https://arxiv.org/pdf/2505.15516)  

**Abstract**: While eXplainable AI (XAI) has advanced significantly, few methods address interpretability in embedded vector spaces where dimensions represent complex abstractions. We introduce Distance Explainer, a novel method for generating local, post-hoc explanations of embedded spaces in machine learning models. Our approach adapts saliency-based techniques from RISE to explain the distance between two embedded data points by assigning attribution values through selective masking and distance-ranked mask filtering. We evaluate Distance Explainer on cross-modal embeddings (image-image and image-caption pairs) using established XAI metrics including Faithfulness, Sensitivity/Robustness, and Randomization. Experiments with ImageNet and CLIP models demonstrate that our method effectively identifies features contributing to similarity or dissimilarity between embedded data points while maintaining high robustness and consistency. We also explore how parameter tuning, particularly mask quantity and selection strategy, affects explanation quality. This work addresses a critical gap in XAI research and enhances transparency and trustworthiness in deep learning applications utilizing embedded spaces. 

**Abstract (ZH)**: While eXplainable AI (XAI) has advanced significantly, few methods address interpretability in embedded vector spaces where dimensions represent complex abstractions. We introduce Distance Explainer, a novel method for generating local, post-hoc explanations of embedded spaces in machine learning models. 

---
# AM-PPO: (Advantage) Alpha-Modulation with Proximal Policy Optimization 

**Title (ZH)**: AM-PPO: (优势) 阿尔法调制与中心化优势优势-策略优化 

**Authors**: Soham Sane  

**Link**: [PDF](https://arxiv.org/pdf/2505.15514)  

**Abstract**: Proximal Policy Optimization (PPO) is a widely used reinforcement learning algorithm that heavily relies on accurate advantage estimates for stable and efficient training. However, raw advantage signals can exhibit significant variance, noise, and scale-related issues, impeding optimal learning performance. To address this challenge, we introduce Advantage Modulation PPO (AM-PPO), a novel enhancement of PPO that adaptively modulates advantage estimates using a dynamic, non-linear scaling mechanism. This adaptive modulation employs an alpha controller that dynamically adjusts the scaling factor based on evolving statistical properties of the advantage signals, such as their norm, variance, and a predefined target saturation level. By incorporating a tanh-based gating function driven by these adaptively scaled advantages, AM-PPO reshapes the advantage signals to stabilize gradient updates and improve the conditioning of the policy gradient landscape. Crucially, this modulation also influences value function training by providing consistent and adaptively conditioned learning targets. Empirical evaluations across standard continuous control benchmarks demonstrate that AM-PPO achieves superior reward trajectories, exhibits sustained learning progression, and significantly reduces the clipping required by adaptive optimizers. These findings underscore the potential of advantage modulation as a broadly applicable technique for enhancing reinforcement learning optimization. 

**Abstract (ZH)**: 优势调制PPO：一种基于动态非线性缩放机制的PPO增强算法 

---
# Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning 

**Title (ZH)**: 方向非交换幺半结构及其在机器学习中组件嵌入中的应用 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2505.15507)  

**Abstract**: We introduce a new algebraic structure for multi-dimensional compositional embeddings, built on directional non-commutative monoidal operators. The core contribution of this work is this novel framework, which exhibits appealing theoretical properties (associativity along each dimension and an interchange law ensuring global consistency) while remaining compatible with modern machine learning architectures. Our construction defines a distinct composition operator circ_i for each axis i, ensuring associative combination along each axis without imposing global commutativity. Importantly, all axis-specific operators commute with one another, enforcing a global interchange law that enables consistent crossaxis compositions. This is, to our knowledge, the first approach that provides a common foundation that generalizes classical sequence-modeling paradigms (e.g., structured state-space models (SSMs) and transformer self-attention) to a unified multi-dimensional framework. For example, specific one-dimensional instances of our framework can recover the familiar affine transformation algebra, vanilla self-attention, and the SSM-style recurrence. The higher-dimensional generalizations naturally support recursive, structure-aware operations in embedding spaces. We outline several potential applications unlocked by this structure-including structured positional encodings in Transformers, directional image embeddings, and symbolic modeling of sequences or grids-indicating that it could inform future deep learning model designs. We formally establish the algebraic properties of our framework and discuss efficient implementations. Finally, as our focus is theoretical, we include no experiments here and defer empirical validation to future work, which we plan to undertake. 

**Abstract (ZH)**: 我们提出了一种基于方向非交换半环运算的多维组合嵌入的新代数结构。这项工作的核心贡献是这一新颖框架，它展现出诱人的理论性质（每个维度上的结合性和确保全局一致性的互换法则），同时与现代机器学习架构保持兼容。我们的构建定义了每个轴i的独特组合运算符circ_i，确保每个轴上的结合组合而不强求全局可交换性。重要的是，所有轴特异性运算符彼此可交换，实现了全局互换法则，从而允许一致的跨轴组合。据我们所知，这是第一次提供一个共同的基础框架，将经典序列建模范式（如结构状态空间模型（SSMs）和变换器自注意力机制）推广到统一的多维框架。例如，我们框架的一维特定实例可以恢复熟悉的仿射变换代数、普通的自注意力和SSM风格的递归结构。高维的推广自然支持嵌入空间中的递归、结构感知操作。我们概述了由这种结构解锁的多种潜在应用，包括变换器中的结构位置编码、方向图像嵌入和序列或网格的符号建模，表明它可能影响未来深度学习模型的设计。我们形式地建立了我们框架的代数性质，并讨论了高效的实现方法。最后，由于我们的关注点是理论，这里没有包含实验，而是将实证验证推迟到将来的研究中，这是我们计划进行的研究。 

---
# Beyond Linearity: Squeeze-and-Recalibrate Blocks for Few-Shot Whole Slide Image Classification 

**Title (ZH)**: 超越线性关系：压缩并校准模块用于少量样本全玻片图像分类 

**Authors**: Conghao Xiong, Zhengrui Guo, Zhe Xu, Yifei Zhang, Raymond Kai-Yu Tong, Si Yong Yeo, Hao Chen, Joseph J. Y. Sung, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2505.15504)  

**Abstract**: Deep learning has advanced computational pathology but expert annotations remain scarce. Few-shot learning mitigates annotation burdens yet suffers from overfitting and discriminative feature mischaracterization. In addition, the current few-shot multiple instance learning (MIL) approaches leverage pretrained vision-language models to alleviate these issues, but at the cost of complex preprocessing and high computational cost. We propose a Squeeze-and-Recalibrate (SR) block, a drop-in replacement for linear layers in MIL models to address these challenges. The SR block comprises two core components: a pair of low-rank trainable matrices (squeeze pathway, SP) that reduces parameter count and imposes a bottleneck to prevent spurious feature learning, and a frozen random recalibration matrix that preserves geometric structure, diversifies feature directions, and redefines the optimization objective for the SP. We provide theoretical guarantees that the SR block can approximate any linear mapping to arbitrary precision, thereby ensuring that the performance of a standard MIL model serves as a lower bound for its SR-enhanced counterpart. Extensive experiments demonstrate that our SR-MIL models consistently outperform prior methods while requiring significantly fewer parameters and no architectural changes. 

**Abstract (ZH)**: 深度学习推动了计算病理学的发展，但专家标注仍然稀缺。少样本学习减轻了标注负担，但易过拟合并存在特征误表征问题。此外，当前的少样本多次实例学习（MIL）方法利用预训练的视觉-语言模型来缓解这些问题，但代价是复杂的预处理和高计算成本。我们提出了一种挤压和重新校准（SR）块，它是MIL模型中线性层的即插即用替代方案，以解决这些挑战。SR块包含两个核心组件：一对可训练的低秩矩阵（挤压路径，SP），用于减少参数数量并防止虚假特征学习，以及一个冻结的随机重新校准矩阵，用于保持几何结构、多样化特征方向，并重新定义SP的优化目标。我们提供了理论保证，证明SR块可以任意精度逼近任何线性映射，从而确保标准MIL模型的表现为其SR增强版本的下限。大量实验表明，我们的SR-MIL模型在参数量大幅减少且无需架构更改的情况下，始终优于先前的方法。 

---
# Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs 

**Title (ZH)**: Protoknowledge 影响大规模语言模型在下游任务中行为的方式：知识图谱中的记忆与泛化 

**Authors**: Federico Ranaldi, Andrea Zugarini, Leonardo Ranaldi, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2505.15501)  

**Abstract**: We introduce the concept of protoknowledge to formalize and measure how sequences of tokens encoding Knowledge Graphs are internalized during pretraining and utilized at inference time by Large Language Models (LLMs). Indeed, LLMs have demonstrated the ability to memorize vast amounts of token sequences during pretraining, and a central open question is how they leverage this memorization as reusable knowledge through generalization. We then categorize protoknowledge into lexical, hierarchical, and topological forms, varying on the type of knowledge that needs to be activated. We measure protoknowledge through Knowledge Activation Tasks (KATs), analyzing its general properties such as semantic bias. We then investigate the impact of protoknowledge on Text-to-SPARQL performance by varying prompting strategies depending on input conditions. To this end, we adopt a novel analysis framework that assesses whether model predictions align with the successful activation of the relevant protoknowledge for each query. This methodology provides a practical tool to explore Semantic-Level Data Contamination and serves as an effective strategy for Closed-Pretraining models. 

**Abstract (ZH)**: 我们将概念化并衡量编码知识图谱的令牌序列在预训练期间的内化及其在推理时的利用过程，引入protoknowledge的概念。我们进一步将protoknowledge分为词缀的、层次的和拓扑的形式，根据不同类型的需激活知识进行分类。我们通过知识激活任务（KATs）测量protoknowledge，并分析其语义偏差等基本特性。然后，我们通过根据输入条件变化提示策略来研究protoknowledge对文本到SPARQL性能的影响。为此，我们采用一种新型分析框架，评估模型预测是否与每个查询的相关protoknowledge的成功激活一致。该方法提供了一种实用工具来探索语义级数据污染，并作为闭合预训练模型的有效策略。 

---
# LFTF: Locating First and Then Fine-Tuning for Mitigating Gender Bias in Large Language Models 

**Title (ZH)**: LFTF: 首先定位然后微调以减轻大型语言模型中的性别偏见 

**Authors**: Zhanyue Qin, Yue Ding, Deyuan Liu, Qingbin Liu, Junxian Cai, Xi Chen, Zhiying Tu, Dianhui Chu, Cuiyun Gao, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.15475)  

**Abstract**: Nowadays, Large Language Models (LLMs) have attracted widespread attention due to their powerful performance. However, due to the unavoidable exposure to socially biased data during training, LLMs tend to exhibit social biases, particularly gender bias. To better explore and quantifying the degree of gender bias in LLMs, we propose a pair of datasets named GenBiasEval and GenHintEval, respectively. The GenBiasEval is responsible for evaluating the degree of gender bias in LLMs, accompanied by an evaluation metric named AFGB-Score (Absolutely Fair Gender Bias Score). Meanwhile, the GenHintEval is used to assess whether LLMs can provide responses consistent with prompts that contain gender hints, along with the accompanying evaluation metric UB-Score (UnBias Score). Besides, in order to mitigate gender bias in LLMs more effectively, we present the LFTF (Locating First and Then Fine-Tuning) this http URL algorithm first ranks specific LLM blocks by their relevance to gender bias in descending order using a metric called BMI (Block Mitigating Importance Score). Based on this ranking, the block most strongly associated with gender bias is then fine-tuned using a carefully designed loss function. Numerous experiments have shown that our proposed LFTF algorithm can significantly mitigate gender bias in LLMs while maintaining their general capabilities. 

**Abstract (ZH)**: 现今，大型语言模型（LLMs）由于其强大的性能而引起了广泛关注。然而，由于不可避免地会接触到社会偏见数据的训练过程，LLMs往往会表现出社会偏见，特别是性别偏见。为了更深入地探索和量化LLMs中的性别偏见程度，我们提出了两个数据集，分别命名为GenBiasEval和GenHintEval。GenBiasEval用于评估LLMs中的性别偏见程度，伴随有一个名为AFGB-Score（绝对公平性别偏见评分）的评估指标。同时，GenHintEval用于评估LLMs是否能够提供与含有性别暗示的提示相一致的响应，伴随有UB-Score（无偏评分）的评估指标。此外，为了更有效地减轻LLMs中的性别偏见，我们提出了LFTF（定位并调整）算法。该算法首先使用称为BMI（块减轻重要性评分）的指标按降序对特定的LLM块进行排序。基于此排序，在一个精心设计的损失函数基础上，调整与性别偏见关联最紧密的块。大量实验表明，我们提出的LFTF算法在显著减轻LLMs中的性别偏见的同时，能够保持其一般性能。 

---
# A Qualitative Investigation into LLM-Generated Multilingual Code Comments and Automatic Evaluation Metrics 

**Title (ZH)**: 对LLM生成的多语言代码注释进行的定性研究及自动评价指标探究 

**Authors**: Jonathan Katzy, Yongcheng Huang, Gopal-Raj Panchu, Maksym Ziemlewski, Paris Loizides, Sander Vermeulen, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15469)  

**Abstract**: Large Language Models are essential coding assistants, yet their training is predominantly English-centric. In this study, we evaluate the performance of code language models in non-English contexts, identifying challenges in their adoption and integration into multilingual workflows. We conduct an open-coding study to analyze errors in code comments generated by five state-of-the-art code models, CodeGemma, CodeLlama, CodeQwen1.5, GraniteCode, and StarCoder2 across five natural languages: Chinese, Dutch, English, Greek, and Polish. Our study yields a dataset of 12,500 labeled generations, which we publicly release. We then assess the reliability of standard metrics in capturing comment \textit{correctness} across languages and evaluate their trustworthiness as judgment criteria. Through our open-coding investigation, we identified a taxonomy of 26 distinct error categories in model-generated code comments. They highlight variations in language cohesion, informativeness, and syntax adherence across different natural languages. Our analysis shows that, while these models frequently produce partially correct comments, modern neural metrics fail to reliably differentiate meaningful completions from random noise. Notably, the significant score overlap between expert-rated correct and incorrect comments calls into question the effectiveness of these metrics in assessing generated comments. 

**Abstract (ZH)**: 大型语言模型是重要的编码助手，但它们的训练主要以英语为中心。本研究评估了代码语言模型在非英语环境中的性能，识别其在多语言工作流程中的采用和集成所面临的挑战。我们进行了一项开放编码研究，分析了五种最新的代码模型CodeGemma、CodeLlama、CodeQwen1.5、GraniteCode和StarCoder2生成的代码注释错误，涵盖了五种自然语言：中文、荷兰语、英语、希腊语和波兰语。我们的研究生成了一个包含12,500个标注生成的语料库，并公开发布。然后我们评估了标准指标在跨语言捕捉注释的准确性的可靠性，并评估它们作为判断标准的信任度。通过我们的开放编码研究，我们识别出26个模型生成代码注释的不同错误类别，突显了不同自然语言在连贯性、信息量和语法遵守方面的差异。我们的分析显示，尽管这些模型经常生成部分正确的注释，但现代神经指标无法可靠地区分有意义的完成和随机噪声。值得注意的是，专家评级正确和不正确注释之间显著的得分重叠引发了这些指标评估生成注释有效性的问题。 

---
# Joint Flashback Adaptation for Forgetting-Resistant Instruction Tuning 

**Title (ZH)**: 遗忘 resistant 指令调优的联合 flashback 调整 

**Authors**: Yukun Zhao, Lingyong Yan, Zhenyang Li, Shuaiqiang Wang, Zhumin Chen, Zhaochun Ren, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15467)  

**Abstract**: Large language models have achieved remarkable success in various tasks. However, it is challenging for them to learn new tasks incrementally due to catastrophic forgetting. Existing approaches rely on experience replay, optimization constraints, or task differentiation, which encounter strict limitations in real-world scenarios. To address these issues, we propose Joint Flashback Adaptation. We first introduce flashbacks -- a limited number of prompts from old tasks -- when adapting to new tasks and constrain the deviations of the model outputs compared to the original one. We then interpolate latent tasks between flashbacks and new tasks to enable jointly learning relevant latent tasks, new tasks, and flashbacks, alleviating data sparsity in flashbacks and facilitating knowledge sharing for smooth adaptation. Our method requires only a limited number of flashbacks without access to the replay data and is task-agnostic. We conduct extensive experiments on state-of-the-art large language models across 1000+ instruction-following tasks, arithmetic reasoning tasks, and general reasoning tasks. The results demonstrate the superior performance of our method in improving generalization on new tasks and reducing forgetting in old tasks. 

**Abstract (ZH)**: 大型语言模型在各类任务中取得了显著成功，但由于灾难性遗忘，它们在增量学习新任务方面面临挑战。现有方法依赖经验回放、优化约束或任务差异化，但在实际场景中会遇到严格限制。为解决这些问题，我们提出联合回溯适应方法。我们首先在适应新任务时引入回溯——来自旧任务的一小部分提示，限制模型输出与原始输出的偏差。然后，我们在回溯和新任务之间插值潜在任务，以实现潜在任务、新任务和回溯的联合学习，减少回溯中的数据稀疏性并促进知识共享，以便平滑适应。我们的方法仅需少量回溯，无需访问回放数据且无任务依赖性。我们在1000多个指令跟随任务、算术推理任务和一般推理任务上的先进大型语言模型上进行了广泛实验。结果表明，我们的方法在提高新任务泛化能力和减少旧任务遗忘方面表现出优越性能。 

---
# ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning 

**Title (ZH)**: ViaRL：基于视觉迭代放大强化学习的自适应时空定位 

**Authors**: Ziqiang Xu, Qi Dai, Tian Xie, Yifan Yang, Kai Qiu, DongDong Chen, Zuxuan Wu, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.15447)  

**Abstract**: Video understanding is inherently intention-driven-humans naturally focus on relevant frames based on their goals. Recent advancements in multimodal large language models (MLLMs) have enabled flexible query-driven reasoning; however, video-based frameworks like Video Chain-of-Thought lack direct training signals to effectively identify relevant frames. Current approaches often rely on heuristic methods or pseudo-label supervised annotations, which are both costly and limited in scalability across diverse scenarios. To overcome these challenges, we introduce ViaRL, the first framework to leverage rule-based reinforcement learning (RL) for optimizing frame selection in intention-driven video understanding. An iterated amplification strategy is adopted to perform alternating cyclic training in the video CoT system, where each component undergoes iterative cycles of refinement to improve its capabilities. ViaRL utilizes the answer accuracy of a downstream model as a reward signal to train a frame selector through trial-and-error, eliminating the need for expensive annotations while closely aligning with human-like learning processes. Comprehensive experiments across multiple benchmarks, including VideoMME, LVBench, and MLVU, demonstrate that ViaRL consistently delivers superior temporal grounding performance and robust generalization across diverse video understanding tasks, highlighting its effectiveness and scalability. Notably, ViaRL achieves a nearly 15\% improvement on Needle QA, a subset of MLVU, which is required to search a specific needle within a long video and regarded as one of the most suitable benchmarks for evaluating temporal grounding. 

**Abstract (ZH)**: 基于规则的强化学习驱动的视频理解中的片段选择框架 

---
# Single LLM, Multiple Roles: A Unified Retrieval-Augmented Generation Framework Using Role-Specific Token Optimization 

**Title (ZH)**: 单一LLM，多种角色：基于角色特定标记优化的统一检索增强生成框架 

**Authors**: Yutao Zhu, Jiajie Jin, Hongjin Qian, Zheng Liu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15444)  

**Abstract**: Existing studies have optimized retrieval-augmented generation (RAG) across various sub-tasks, such as query understanding and retrieval refinement, but integrating these optimizations into a unified framework remains challenging. To tackle this problem, this work proposes RoleRAG, a unified RAG framework that achieves efficient multi-task processing through role-specific token optimization. RoleRAG comprises six modules, each handling a specific sub-task within the RAG process. Additionally, we introduce a query graph to represent the decomposition of the query, which can be dynamically resolved according to the decomposing state. All modules are driven by the same underlying LLM, distinguished by task-specific role tokens that are individually optimized. This design allows RoleRAG to dynamically activate different modules within a single LLM instance, thereby streamlining deployment and reducing resource consumption. Experimental results on five open-domain question-answering datasets demonstrate the effectiveness, generalizability, and flexibility of our framework. 

**Abstract (ZH)**: 现有的研究已在各种子任务（如查询理解及检索精炼）上优化了检索增强生成（RAG）模型，但将这些优化整合到一个统一框架中仍具有挑战性。为解决这一问题，本工作提出了一种名为RoleRAG的统一RAG框架，通过角色特定的.token优化实现高效的多任务处理。RoleRAG包含了六个模块，每个模块负责RAG过程中的一个特定子任务。此外，我们引入了一个查询图来表示查询的分解，并可根据分解状态动态解决。所有模块由同一个底层LLM驱动，不同任务的角色标记分别优化。该设计使RoleRAG能够在单个LLM实例中动态激活不同的模块，从而简化部署并减少资源消耗。在五个开放领域问答数据集上的实验结果证明了该框架的有效性、泛化能力和灵活性。 

---
# Stronger ViTs With Octic Equivariance 

**Title (ZH)**: 具有八次对称性的更强的ViTs 

**Authors**: David Nordström, Johan Edstedt, Fredrik Kahl, Georg Bökman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15441)  

**Abstract**: Recent efforts at scaling computer vision models have established Vision Transformers (ViTs) as the leading architecture. ViTs incorporate weight sharing over image patches as an important inductive bias. In this work, we show that ViTs benefit from incorporating equivariance under the octic group, i.e., reflections and 90-degree rotations, as a further inductive bias. We develop new architectures, octic ViTs, that use octic-equivariant layers and put them to the test on both supervised and self-supervised learning. Through extensive experiments on DeiT-III and DINOv2 training on ImageNet-1K, we show that octic ViTs yield more computationally efficient networks while also improving performance. In particular, we achieve approximately 40% reduction in FLOPs for ViT-H while simultaneously improving both classification and segmentation results. 

**Abstract (ZH)**: 近期在扩展计算机视觉模型的努力中，已经确立了视觉变换器（ViTs）作为领先的架构。ViTs 通过在图像patches之间共享权重来引入一个重要的归纳偏置。在本文中，我们展示了将八阶群下的协变性作为进一步的归纳偏置加入ViTs 可以从中受益。我们开发了新的架构——八阶ViTs，这些架构使用八阶协变层，并在有监督和自监督学习中进行了测试。通过在ImageNet-1K上的DeiT-III和DINOv2训练中的广泛实验，我们展示了八阶ViTs 可以在更计算效率的同时提高性能。特别是，我们实现了ViT-H约40%的FLOPs减少，同时在分类和分割结果上也有所提升。 

---
# Set-LLM: A Permutation-Invariant LLM 

**Title (ZH)**: Set-LLM: 一个不变排列的大语言模型 

**Authors**: Beni Egressy, Jan Stühmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15433)  

**Abstract**: While large language models (LLMs) demonstrate impressive capabilities across numerous applications, their robustness remains a critical concern. This paper is motivated by a specific vulnerability: the order sensitivity of LLMs. This vulnerability manifests itself as the order bias observed when LLMs decide between possible options (for example, a preference for the first option) and the tendency of LLMs to provide different answers when options are reordered. The use cases for this scenario extend beyond the classical case of multiple-choice question answering to the use of LLMs as automated evaluators in AI pipelines, comparing output generated by different models. We introduce Set-LLM, a novel architectural adaptation for pretrained LLMs that enables the processing of mixed set-text inputs with permutation invariance guarantees. The adaptations involve a new attention mask and new positional encodings specifically designed for sets. We provide a theoretical proof of invariance and demonstrate through experiments that Set-LLM can be trained effectively, achieving comparable or improved performance and maintaining the runtime of the original model, while eliminating order sensitivity. 

**Abstract (ZH)**: 大规模语言模型的顺序鲁棒性：Set-LLM架构的设计与实现 

---
# Uncertainty Quantification in SVM prediction 

**Title (ZH)**: SVM预测中的不确定量化分析 

**Authors**: Pritam Anand  

**Link**: [PDF](https://arxiv.org/pdf/2505.15429)  

**Abstract**: This paper explores Uncertainty Quantification (UQ) in SVM predictions, particularly for regression and forecasting tasks. Unlike the Neural Network, the SVM solutions are typically more stable, sparse, optimal and interpretable. However, there are only few literature which addresses the UQ in SVM prediction. At first, we provide a comprehensive summary of existing Prediction Interval (PI) estimation and probabilistic forecasting methods developed in the SVM framework and evaluate them against the key properties expected from an ideal PI model. We find that none of the existing SVM PI models achieves a sparse solution. To introduce sparsity in SVM model, we propose the Sparse Support Vector Quantile Regression (SSVQR) model, which constructs PIs and probabilistic forecasts by solving a pair of linear programs. Further, we develop a feature selection algorithm for PI estimation using SSVQR that effectively eliminates a significant number of features while improving PI quality in case of high-dimensional dataset. Finally we extend the SVM models in Conformal Regression setting for obtaining more stable prediction set with finite test set guarantees. Extensive experiments on artificial, real-world benchmark datasets compare the different characteristics of both existing and proposed SVM-based PI estimation methods and also highlight the advantages of the feature selection in PI estimation. Furthermore, we compare both, the existing and proposed SVM-based PI estimation models, with modern deep learning models for probabilistic forecasting tasks on benchmark datasets. Furthermore, SVM models show comparable or superior performance to modern complex deep learning models for probabilistic forecasting task in our experiments. 

**Abstract (ZH)**: 这篇论文探讨了SVM预测中的不确定性量化（UQ），特别是在回归和预测任务中的应用。不同于神经网络，SVM解通常更稳定、稀疏、最优且可解释。然而，关于SVM预测的不确定性量化文献较少。首先，我们对SVM框架下现有的预测区间（PI）估计和概率预测方法进行了全面总结，并评估了它们是否符合理想PI模型的预期属性。我们发现现有的SVM PI模型均未实现稀疏解。为在SVM模型中引入稀疏性，我们提出了稀疏支持向量分位数回归（SSVQR）模型，该模型通过求解线性规划来构建PI和概率预测。此外，我们开发了一种基于SSVQR的特征选择算法，用于预测区间估计，在高维数据集中有效减少了大量特征，同时提高了PI的质量。最后，我们扩展了SVM模型在一致性回归框架下的应用，以获得具有有限测试集保证的更稳定的预测集。通过对人工和真实世界基准数据集进行详尽的实验，比较了现有和提出的SVM基预测区间估计方法的不同特性，并突出了特征选择在预测区间估计中的优势。此外，我们还将现有和提出的SVM基预测区间估计模型与现代深度学习模型进行了比较，用于基准数据集上的概率预测任务。实验结果显示，SVM模型在概率预测任务中表现与现代复杂的深度学习模型相当或更优。 

---
# Responsible Diffusion Models via Constraining Text Embeddings within Safe Regions 

**Title (ZH)**: 负责任的Diffusion模型通过在安全区域内约束文本嵌入实现 

**Authors**: Zhiwen Li, Die Chen, Mingyuan Fan, Cen Chen, Yaliang Li, Yanhao Wang, Wenmeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.15427)  

**Abstract**: The remarkable ability of diffusion models to generate high-fidelity images has led to their widespread adoption. However, concerns have also arisen regarding their potential to produce Not Safe for Work (NSFW) content and exhibit social biases, hindering their practical use in real-world applications. In response to this challenge, prior work has focused on employing security filters to identify and exclude toxic text, or alternatively, fine-tuning pre-trained diffusion models to erase sensitive concepts. Unfortunately, existing methods struggle to achieve satisfactory performance in the sense that they can have a significant impact on the normal model output while still failing to prevent the generation of harmful content in some cases. In this paper, we propose a novel self-discovery approach to identifying a semantic direction vector in the embedding space to restrict text embedding within a safe region. Our method circumvents the need for correcting individual words within the input text and steers the entire text prompt towards a safe region in the embedding space, thereby enhancing model robustness against all possibly unsafe prompts. In addition, we employ Low-Rank Adaptation (LoRA) for semantic direction vector initialization to reduce the impact on the model performance for other semantics. Furthermore, our method can also be integrated with existing methods to improve their social responsibility. Extensive experiments on benchmark datasets demonstrate that our method can effectively reduce NSFW content and mitigate social bias generated by diffusion models compared to several state-of-the-art baselines. 

**Abstract (ZH)**: 扩散模型生成高保真图像的能力使其得到了广泛应用，但这也引发了对其可能生成不合适内容（NSFW内容）和社会偏见的担忧，阻碍了其在实际应用中的使用。为应对这一挑战，现有工作主要通过使用安全过滤器识别和排除有毒文本，或者微调预训练扩散模型以删除敏感概念。然而，现有方法在确保模型输出效果和防止生成有害内容方面难以达到满意的效果。本文提出了一种新的自发现方法，以在嵌入空间中识别语义方向向量，限制文本嵌入在安全区域。该方法绕过了对输入文本中个别词语进行修正的需求，而是引导整个文本提示向嵌入空间中的安全区域偏移，从而增强模型对所有可能不安全提示的鲁棒性。此外，我们采用低秩适应（LoRA）来初始化语义方向向量，以减少对其他语义性能的影响。同时，我们的方法还可以与现有方法集成，以提高其社会责任感。在基准数据集上的广泛实验表明，与几种最先进的基线方法相比，我们的方法可以有效减少NSFW内容并减轻扩散模型生成的社会偏见。 

---
# Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems through Benign Queries 

**Title (ZH)**: 静默泄露：通过良性查询对RAG系统进行隐性知识提取攻击 

**Authors**: Yuhao Wang, Wenjie Qu, Yanze Jiang, Zichen Liu, Yue Liu, Shengfang Zhai, Yinpeng Dong, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15420)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by incorporating external knowledge bases, but they are vulnerable to privacy risks from data extraction attacks. Existing extraction methods typically rely on malicious inputs such as prompt injection or jailbreaking, making them easily detectable via input- or output-level detection. In this paper, we introduce Implicit Knowledge Extraction Attack (IKEA), which conducts knowledge extraction on RAG systems through benign queries. IKEA first leverages anchor concepts to generate queries with the natural appearance, and then designs two mechanisms to lead to anchor concept thoroughly 'explore' the RAG's privacy knowledge: (1) Experience Reflection Sampling, which samples anchor concepts based on past query-response patterns to ensure the queries' relevance to RAG documents; (2) Trust Region Directed Mutation, which iteratively mutates anchor concepts under similarity constraints to further exploit the embedding space. Extensive experiments demonstrate IKEA's effectiveness under various defenses, surpassing baselines by over 80% in extraction efficiency and 90% in attack success rate. Moreover, the substitute RAG system built from IKEA's extractions consistently outperforms those based on baseline methods across multiple evaluation tasks, underscoring the significant privacy risk in RAG systems. 

**Abstract (ZH)**: 隐式知识提取攻击（IKEA）：通过良性查询增强的大语言模型中的隐私知识提取 

---
# Guided Policy Optimization under Partial Observability 

**Title (ZH)**: 部分可观测条件下的引导策略优化 

**Authors**: Yueheng Li, Guangming Xie, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15418)  

**Abstract**: Reinforcement Learning (RL) in partially observable environments poses significant challenges due to the complexity of learning under uncertainty. While additional information, such as that available in simulations, can enhance training, effectively leveraging it remains an open problem. To address this, we introduce Guided Policy Optimization (GPO), a framework that co-trains a guider and a learner. The guider takes advantage of privileged information while ensuring alignment with the learner's policy that is primarily trained via imitation learning. We theoretically demonstrate that this learning scheme achieves optimality comparable to direct RL, thereby overcoming key limitations inherent in existing approaches. Empirical evaluations show strong performance of GPO across various tasks, including continuous control with partial observability and noise, and memory-based challenges, significantly outperforming existing methods. 

**Abstract (ZH)**: 部分可观测环境中的强化学习（RL）由于在不确定性下的学习复杂性而面临重大挑战。虽然额外信息，如仿真中的信息，可以提高训练效果，但有效地利用这些信息仍然是一个开放的问题。为解决这一问题，我们引入了指导性策略优化（GPO）框架，该框架共同训练一个指导器和一个学习器。指导器利用了特权信息，同时确保与通过模仿学习主要训练的学习器策略相一致。我们从理论上证明了这种学习方案在实现直接RL相近的最优性方面具有优势，从而克服了现有方法的关键局限性。实证评估显示，GPO在各种任务中表现出色，包括连续控制下的部分可观测性和噪声挑战以及基于记忆的挑战，显著优于现有方法。 

---
# Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models 

**Title (ZH)**: 音频脱狱：面向大型音频-语言模型脱狱的开放综合基准 

**Authors**: Zirui Song, Qian Jiang, Mingxuan Cui, Mingzhe Li, Lang Gao, Zeyu Zhang, Zixiang Xu, Yanbo Wang, Chenxi Wang, Guangxian Ouyang, Zhenhao Chen, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15406)  

**Abstract**: The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms. 

**Abstract (ZH)**: LAMs兴起带来的机遇与风险：AJailBenchbenchmark及其应用 

---
# RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection 

**Title (ZH)**: RePPL：基于语义传播和语言生成不确定性调整困惑度的可解释QA幻觉检测 

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Xuming Hu, Yi R., Fung, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15386)  

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为强大的工具，但幻觉仍然是其可靠使用的重要障碍。虽然之前的著作通过测量不确定性改善了幻觉检测能力，但它们缺乏解释幻觉发生根源的能力，即哪些输入部分容易引发幻觉。最近关于提示攻击的研究表明，在语义传播过程中，注意力机制逐渐将局部词元信息融合成高层语义，同时，由于基于概率的选择高层语义进行采样生成，不确定性也在语言生成中显现。基于此，我们提出RePPL来从这两个方面重新校准不确定性测量，为每个词元分配可解释的不确定性分数，并以困惑度风格的对数平均形式汇总为总分。实验表明，我们的方法在高级模型的各种QA数据集上实现了最佳综合检测性能（平均AUC为0.833），并且我们的方法能够生成词元级别的不确定性分数作为幻觉的解释。利用这些分数，我们初步揭示了幻觉的混乱模式，并展示了其潜在的应用价值。 

---
# Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding 

**Title (ZH)**: 使用语音推测解码加速自回归语音合成推断 

**Authors**: Zijian Lin, Yang Zhang, Yougen Yuan, Yuming Yan, Jinjiang Liu, Zhiyong Wu, Pengfei Hu, Qun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15380)  

**Abstract**: Modern autoregressive speech synthesis models leveraging language models have demonstrated remarkable performance. However, the sequential nature of next token prediction in these models leads to significant latency, hindering their deployment in scenarios where inference speed is critical. In this work, we propose Speech Speculative Decoding (SSD), a novel framework for autoregressive speech synthesis acceleration. Specifically, our method employs a lightweight draft model to generate candidate token sequences, which are subsequently verified in parallel by the target model using the proposed SSD framework. Experimental results demonstrate that SSD achieves a significant speedup of 1.4x compared with conventional autoregressive decoding, while maintaining high fidelity and naturalness. Subjective evaluations further validate the effectiveness of SSD in preserving the perceptual quality of the target model while accelerating inference. 

**Abstract (ZH)**: 利用语言模型的现代自回归语音合成模型展示了卓越的性能。然而，这些模型在下一个令牌预测中的序列性质导致了显著的延迟，阻碍了它们在需要快速推理速度的场景中的部署。本文提出了一种新颖的自回归语音合成加速框架——语音投机解码（SSD）。具体来说，该方法使用一个轻量级草图模型生成候选令牌序列，随后通过提出的SSD框架并行验证这些序列。实验结果表明，SSD比传统的自回归解码速度快1.4倍，同时保持了高保真度和自然度。进一步的主观评估验证了SSD在加速推理的同时有效保持目标模型的感知质量。 

---
# Better Safe Than Sorry? Overreaction Problem of Vision Language Models in Visual Emergency Recognition 

**Title (ZH)**: 宁可谨慎些好？视觉语言模型在视觉应急识别中的过度反应问题 

**Authors**: Dasol Choi, Seunghyun Lee, Youngsook Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.15367)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated impressive capabilities in understanding visual content, but their reliability in safety-critical contexts remains under-explored. We introduce VERI (Visual Emergency Recognition Dataset), a carefully designed diagnostic benchmark of 200 images (100 contrastive pairs). Each emergency scene is matched with a visually similar but safe counterpart through multi-stage human verification and iterative refinement. Using a two-stage protocol - risk identification and emergency response - we evaluate 14 VLMs (2B-124B parameters) across medical emergencies, accidents, and natural disasters. Our analysis reveals a systematic overreaction problem: models excel at identifying real emergencies (70-100 percent success rate) but suffer from an alarming rate of false alarms, misidentifying 31-96 percent of safe situations as dangerous, with 10 scenarios failed by all models regardless of scale. This "better-safe-than-sorry" bias manifests primarily through contextual overinterpretation (88-93 percent of errors), challenging VLMs' reliability for safety applications. These findings highlight persistent limitations that are not resolved by increasing model scale, motivating targeted approaches for improving contextual safety assessment in visually misleading scenarios. 

**Abstract (ZH)**: 视觉-语言模型在安全关键 contexts中的可靠性在应急场景中的诊断基准：VERI数据集 

---
# Objective Bicycle Occlusion Level Classification using a Deformable Parts-Based Model 

**Title (ZH)**: 基于可变形部件模型的自行车遮挡等级分类 

**Authors**: Angelique Mangubat, Shane Gilroy  

**Link**: [PDF](https://arxiv.org/pdf/2505.15358)  

**Abstract**: Road safety is a critical challenge, particularly for cyclists, who are among the most vulnerable road users. This study aims to enhance road safety by proposing a novel benchmark for bicycle occlusion level classification using advanced computer vision techniques. Utilizing a parts-based detection model, images are annotated and processed through a custom image detection pipeline. A novel method of bicycle occlusion level is proposed to objectively quantify the visibility and occlusion level of bicycle semantic parts. The findings indicate that the model robustly quantifies the visibility and occlusion level of bicycles, a significant improvement over the subjective methods used by the current state of the art. Widespread use of the proposed methodology will facilitate the accurate performance reporting of cyclist detection algorithms for occluded cyclists, informing the development of more robust vulnerable road user detection methods for autonomous vehicles. 

**Abstract (ZH)**: 基于先进计算机视觉技术的自行车遮挡等级新型基准研究：提升骑行者道路安全 

---
# Hadamax Encoding: Elevating Performance in Model-Free Atari 

**Title (ZH)**: Hadamax编码：提升模型自由Atari环境中的性能 

**Authors**: Jacob E. Kooi, Zhao Yang, Vincent François-Lavet  

**Link**: [PDF](https://arxiv.org/pdf/2505.15345)  

**Abstract**: Neural network architectures have a large impact in machine learning. In reinforcement learning, network architectures have remained notably simple, as changes often lead to small gains in performance. This work introduces a novel encoder architecture for pixel-based model-free reinforcement learning. The Hadamax (\textbf{Hada}mard \textbf{max}-pooling) encoder achieves state-of-the-art performance by max-pooling Hadamard products between GELU-activated parallel hidden layers. Based on the recent PQN algorithm, the Hadamax encoder achieves state-of-the-art model-free performance in the Atari-57 benchmark. Specifically, without applying any algorithmic hyperparameter modifications, Hadamax-PQN achieves an 80\% performance gain over vanilla PQN and significantly surpasses Rainbow-DQN. For reproducibility, the full code is available on \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 基于像素的无模型强化学习的新型编码器架构：Hadamax编码器在Atari-57基准测试中实现了最先进的无模型性能。 

---
# Alpay Algebra: A Universal Structural Foundation 

**Title (ZH)**: Alpay代数：一个普遍的结构基础 

**Authors**: Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2505.15344)  

**Abstract**: Alpay Algebra is introduced as a universal, category-theoretic framework that unifies classical algebraic structures with modern needs in symbolic recursion and explainable AI. Starting from a minimal list of axioms, we model each algebra as an object in a small cartesian closed category $\mathcal{A}$ and define a transfinite evolution functor $\phi\colon\mathcal{A}\to\mathcal{A}$. We prove that the fixed point $\phi^{\infty}$ exists for every initial object and satisfies an internal universal property that recovers familiar constructs -- limits, colimits, adjunctions -- while extending them to ordinal-indexed folds. A sequence of theorems establishes (i) soundness and conservativity over standard universal algebra, (ii) convergence of $\phi$-iterates under regular cardinals, and (iii) an explanatory correspondence between $\phi^{\infty}$ and minimal sufficient statistics in information-theoretic AI models. We conclude by outlining computational applications: type-safe functional languages, categorical model checking, and signal-level reasoning engines that leverage Alpay Algebra's structural invariants. All proofs are self-contained; no external set-theoretic axioms beyond ZFC are required. This exposition positions Alpay Algebra as a bridge between foundational mathematics and high-impact AI systems, and provides a reference for further work in category theory, transfinite fixed-point analysis, and symbolic computation. 

**Abstract (ZH)**: Alpay代数作为一种统一经典代数结构与现代符号递归和可解释AI需求的通用范畴论框架被引入。从一组最小公理出发，我们将每个代数建模为小笛卡尔闭范畴$\mathcal{A}$中的一个对象，并定义一个超限演化泛函$\phi\colon\mathcal{A}\to\mathcal{A}$。证明对于每个初始对象，存在固定点$\phi^{\infty}$并满足内部普遍性质，恢复了熟悉的构造——极限、共限、伴随——并将它们扩展到序数索引的折叠。一系列定理证明了(i) 对标准普遍代数的保真性和保守性，(ii) 在正则基数下的$\phi$迭代收敛性，以及(iii) $\phi^{\infty}$与信息论AI模型中最小充分统计之间的解释性对应关系。我们通过概述计算应用结束：类型安全函数语言、范畴模型检验以及利用Alpay代数结构性不变量的信号级推理引擎。所有证明都是自包含的；不需要超出ZFC之外的集合论公理。这一表述将Alpay代数定位为基础数学与高影响AI系统之间的桥梁，并为范畴论、超限不动点分析和符号计算进一步研究提供了参考。 

---
# Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors 

**Title (ZH)**: 你的语言模型能在不知不觉中模仿人类写作：对比重述对生成文本检测器的攻击 

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15337)  

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios. 

**Abstract (ZH)**: 基于现成大语言模型的Contrastive Paraphrase Attack (CoPA) 

---
# Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation 

**Title (ZH)**: 利用单元语言指导促进无文本语音到语音翻译中的语音建模 

**Authors**: Yuhao Zhang, Xiangnan Ma, Kaiqi Kou, Peizhuo Liu, Weiqiao Shan, Benyou Wang, Tong Xiao, Yuxin Huang, Zhengtao Yu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15333)  

**Abstract**: The success of building textless speech-to-speech translation (S2ST) models has attracted much attention. However, S2ST still faces two main challenges: 1) extracting linguistic features for various speech signals, called cross-modal (CM), and 2) learning alignment of difference languages in long sequences, called cross-lingual (CL). We propose the unit language to overcome the two modeling challenges. The unit language can be considered a text-like representation format, constructed using $n$-gram language modeling. We implement multi-task learning to utilize the unit language in guiding the speech modeling process. Our initial results reveal a conflict when applying source and target unit languages simultaneously. We propose task prompt modeling to mitigate this conflict. We conduct experiments on four languages of the Voxpupil dataset. Our method demonstrates significant improvements over a strong baseline and achieves performance comparable to models trained with text. 

**Abstract (ZH)**: 构建无文本语音到语音翻译模型的成功吸引了广泛关注。然而，语音到语音翻译仍面临两大挑战：1) 各种语音信号的语言特征提取，称为跨模态（CM），2) 长序列中不同语言对齐的学习，称为跨语言（CL）。我们提出单位语言以克服这两种建模挑战。单位语言可被视为一种类似于文本的表现格式，使用$n$-gram语言建模构建。我们采用多任务学习利用单位语言指导语音建模过程。我们的初步结果显示，同时应用源语言和目标语言单位语言时存在冲突。我们提出任务提示建模以缓解这种冲突。我们在Voxpupil数据集的四种语言上进行了实验。我们的方法在强基线之上显示出显著改进，并且达到与使用文本训练模型相当的性能。 

---
# Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning 

**Title (ZH)**: 轨迹贝尔曼残差最小化：一种简单的基于值的方法用于LLM推理 

**Authors**: Yurun Yuan, Fan Chen, Zeyu Jia, Alexander Rakhlin, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.15311)  

**Abstract**: Policy-based methods currently dominate reinforcement learning (RL) pipelines for large language model (LLM) reasoning, leaving value-based approaches largely unexplored. We revisit the classical paradigm of Bellman Residual Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an algorithm that naturally adapts this idea to LLMs, yielding a simple yet effective off-policy algorithm that optimizes a single trajectory-level Bellman objective using the model's own logits as $Q$-values. TBRM removes the need for critics, importance-sampling ratios, or clipping, and operates with only one rollout per prompt. We prove convergence to the near-optimal KL-regularized policy from arbitrary off-policy data via an improved change-of-trajectory-measure analysis. Experiments on standard mathematical-reasoning benchmarks show that TBRM consistently outperforms policy-based baselines, like PPO and GRPO, with comparable or lower computational and memory overhead. Our results indicate that value-based RL might be a principled and efficient alternative for enhancing reasoning capabilities in LLMs. 

**Abstract (ZH)**: 基于策略的方法目前主导了大规模语言模型（LLM）推理的强化学习（RL）流水线，价值基础的方法则 largely unexplored 基本未被探索。我们重访贝尔曼残差最小化的基本框架，提出轨迹贝尔曼残差最小化（TBRM），一种自然适应 LLM 的算法，生成一种简单而有效的离策优化算法，利用模型自身的 logit 作为 $Q$ 值优化单条轨迹级别贝尔曼目标。TBRM 去掉了批评家、重要性采样比率或裁剪的需求，且仅需一次滚出自每一个提示。我们通过改进的轨迹测度变化分析证明，TBRM 从任意离策数据中可收敛到近最优的 KL 正则化策略。标准的数学推理基准实验表明，TBRM 在与策略基础基线（如 PPO 和 GRPO）媲美的计算和内存开销条件下，始终表现出更优的效果。我们的结果表明，价值基础的强化学习可能是增强 LLM 推理能力的一种原理上合理且高效的替代方案。 

---
# BadSR: Stealthy Label Backdoor Attacks on Image Super-Resolution 

**Title (ZH)**: BadSR：图像超分辨率上的隐蔽标签后门攻击 

**Authors**: Ji Guo, Xiaolei Wen, Wenbo Jiang, Cheng Huang, Jinjin Li, Hongwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15308)  

**Abstract**: With the widespread application of super-resolution (SR) in various fields, researchers have begun to investigate its security. Previous studies have demonstrated that SR models can also be subjected to backdoor attacks through data poisoning, affecting downstream tasks. A backdoor SR model generates an attacker-predefined target image when given a triggered image while producing a normal high-resolution (HR) output for clean images. However, prior backdoor attacks on SR models have primarily focused on the stealthiness of poisoned low-resolution (LR) images while ignoring the stealthiness of poisoned HR images, making it easy for users to detect anomalous data. To address this problem, we propose BadSR, which improves the stealthiness of poisoned HR images. The key idea of BadSR is to approximate the clean HR image and the pre-defined target image in the feature space while ensuring that modifications to the clean HR image remain within a constrained range. The poisoned HR images generated by BadSR can be integrated with existing triggers. To further improve the effectiveness of BadSR, we design an adversarially optimized trigger and a backdoor gradient-driven poisoned sample selection method based on a genetic algorithm. The experimental results show that BadSR achieves a high attack success rate in various models and data sets, significantly affecting downstream tasks. 

**Abstract (ZH)**: 基于BadSR的高分辨率图像后门攻击方法：提高受污染高分辨率图像的隐蔽性 

---
# Multiple Weaks Win Single Strong: Large Language Models Ensemble Weak Reinforcement Learning Agents into a Supreme One 

**Title (ZH)**: 多弱者胜一强：大型语言模型将弱强化学习代理ensemble成一个至高无上者 

**Authors**: Yiwen Song, Qianyue Hao, Qingmin Liao, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15306)  

**Abstract**: Model ensemble is a useful approach in reinforcement learning (RL) for training effective agents. Despite wide success of RL, training effective agents remains difficult due to the multitude of factors requiring careful tuning, such as algorithm selection, hyperparameter settings, and even random seed choices, all of which can significantly influence an agent's performance. Model ensemble helps overcome this challenge by combining multiple weak agents into a single, more powerful one, enhancing overall performance. However, existing ensemble methods, such as majority voting and Boltzmann addition, are designed as fixed strategies and lack a semantic understanding of specific tasks, limiting their adaptability and effectiveness. To address this, we propose LLM-Ens, a novel approach that enhances RL model ensemble with task-specific semantic understandings driven by large language models (LLMs). Given a task, we first design an LLM to categorize states in this task into distinct 'situations', incorporating high-level descriptions of the task conditions. Then, we statistically analyze the strengths and weaknesses of each individual agent to be used in the ensemble in each situation. During the inference time, LLM-Ens dynamically identifies the changing task situation and switches to the agent that performs best in the current situation, ensuring dynamic model selection in the evolving task condition. Our approach is designed to be compatible with agents trained with different random seeds, hyperparameter settings, and various RL algorithms. Extensive experiments on the Atari benchmark show that LLM-Ens significantly improves the RL model ensemble, surpassing well-known baselines by up to 20.9%. For reproducibility, our code is open-source at this https URL. 

**Abstract (ZH)**: LLM驱动的强化学习模型集成方法：基于特定任务语义的理解（LLM-Ens） 

---
# Laplace Sample Information: Data Informativeness Through a Bayesian Lens 

**Title (ZH)**: 拉普拉斯样本信息：通过贝叶斯视角的数据信息量 

**Authors**: Johannes Kaiser, Kristian Schwethelm, Daniel Rueckert, Georgios Kaissis  

**Link**: [PDF](https://arxiv.org/pdf/2505.15303)  

**Abstract**: Accurately estimating the informativeness of individual samples in a dataset is an important objective in deep learning, as it can guide sample selection, which can improve model efficiency and accuracy by removing redundant or potentially harmful samples. We propose Laplace Sample Information (LSI) measure of sample informativeness grounded in information theory widely applicable across model architectures and learning settings. LSI leverages a Bayesian approximation to the weight posterior and the KL divergence to measure the change in the parameter distribution induced by a sample of interest from the dataset. We experimentally show that LSI is effective in ordering the data with respect to typicality, detecting mislabeled samples, measuring class-wise informativeness, and assessing dataset difficulty. We demonstrate these capabilities of LSI on image and text data in supervised and unsupervised settings. Moreover, we show that LSI can be computed efficiently through probes and transfers well to the training of large models. 

**Abstract (ZH)**: 准确估计数据集中单个样本的信息量是深度学习中的一个重要目标，因为它可以指导样本选择，从而通过去除冗余或潜在有害的样本来提高模型效率和准确性。我们提出了一种基于信息理论的Laplace样本信息（LSI）测度，该测度适用于各种模型架构和学习设置。LSI利用贝叶斯权重后验近似和KL散度来衡量兴趣样本引起的数据集中参数分布的变化。实验结果表明，LSI在按典型性排序数据、检测误标样本、测量类内信息量以及评估数据集难度方面有效。我们展示了LSI在监督和无监督设置下的图像和文本数据上的这些能力。此外，我们展示了LSI可以通过探针高效计算，并且可以很好地转移应用于大型模型的训练。 

---
# LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models 

**Title (ZH)**: LLM-探险家：由大型语言模型驱动的插件强化学习策略探索增强 

**Authors**: Qianyue Hao, Yiwen Song, Qingmin Liao, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15293)  

**Abstract**: Policy exploration is critical in reinforcement learning (RL), where existing approaches include greedy, Gaussian process, etc. However, these approaches utilize preset stochastic processes and are indiscriminately applied in all kinds of RL tasks without considering task-specific features that influence policy exploration. Moreover, during RL training, the evolution of such stochastic processes is rigid, which typically only incorporates a decay in the variance, failing to adjust flexibly according to the agent's real-time learning status. Inspired by the analyzing and reasoning capability of large language models (LLMs), we design LLM-Explorer to adaptively generate task-specific exploration strategies with LLMs, enhancing the policy exploration in RL. In our design, we sample the learning trajectory of the agent during the RL training in a given task and prompt the LLM to analyze the agent's current policy learning status and then generate a probability distribution for future policy exploration. Updating the probability distribution periodically, we derive a stochastic process specialized for the particular task and dynamically adjusted to adapt to the learning process. Our design is a plug-in module compatible with various widely applied RL algorithms, including the DQN series, DDPG, TD3, and any possible variants developed based on them. Through extensive experiments on the Atari and MuJoCo benchmarks, we demonstrate LLM-Explorer's capability to enhance RL policy exploration, achieving an average performance improvement up to 37.27%. Our code is open-source at this https URL for reproducibility. 

**Abstract (ZH)**: 基于大型语言模型的自适应探索策略在强化学习中的应用 

---
# Reconsider the Template Mesh in Deep Learning-based Mesh Reconstruction 

**Title (ZH)**: 基于深度学习的网格重建中重考虑模板网格 

**Authors**: Fengting Zhang, Boxu Liang, Qinghao Liu, Min Liu, Xiang Chen, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15285)  

**Abstract**: Mesh reconstruction is a cornerstone process across various applications, including in-silico trials, digital twins, surgical planning, and navigation. Recent advancements in deep learning have notably enhanced mesh reconstruction speeds. Yet, traditional methods predominantly rely on deforming a standardised template mesh for individual subjects, which overlooks the unique anatomical variations between them, and may compromise the fidelity of the reconstructions. In this paper, we propose an adaptive-template-based mesh reconstruction network (ATMRN), which generates adaptive templates from the given images for the subsequent deformation, moving beyond the constraints of a singular, fixed template. Our approach, validated on cortical magnetic resonance (MR) images from the OASIS dataset, sets a new benchmark in voxel-to-cortex mesh reconstruction, achieving an average symmetric surface distance of 0.267mm across four cortical structures. Our proposed method is generic and can be easily transferred to other image modalities and anatomical structures. 

**Abstract (ZH)**: 基于自适应模板的网格重建网络：一种新的皮层磁共振成像到体素网格重建基准方法 

---
# Learning-based Autonomous Oversteer Control and Collision Avoidance 

**Title (ZH)**: 基于学习的自主过度转向控制与碰撞避免 

**Authors**: Seokjun Lee, Seung-Hyun Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15275)  

**Abstract**: Oversteer, wherein a vehicle's rear tires lose traction and induce unintentional excessive yaw, poses critical safety challenges. Failing to control oversteer often leads to severe traffic accidents. Although recent autonomous driving efforts have attempted to handle oversteer through stabilizing maneuvers, the majority rely on expert-defined trajectories or assume obstacle-free environments, limiting real-world applicability. This paper introduces a novel end-to-end (E2E) autonomous driving approach that tackles oversteer control and collision avoidance simultaneously. Existing E2E techniques, including Imitation Learning (IL), Reinforcement Learning (RL), and Hybrid Learning (HL), generally require near-optimal demonstrations or extensive experience. Yet even skilled human drivers struggle to provide perfect demonstrations under oversteer, and high transition variance hinders accumulating sufficient data. Hence, we present Q-Compared Soft Actor-Critic (QC-SAC), a new HL algorithm that effectively learns from suboptimal demonstration data and adapts rapidly to new conditions. To evaluate QC-SAC, we introduce a benchmark inspired by real-world driver training: a vehicle encounters sudden oversteer on a slippery surface and must avoid randomly placed obstacles ahead. Experimental results show QC-SAC attains near-optimal driving policies, significantly surpassing state-of-the-art IL, RL, and HL baselines. Our method demonstrates the world's first safe autonomous oversteer control with obstacle avoidance. 

**Abstract (ZH)**: Oversteer控制及碰撞避免的端到端自主驾驶方法：Q-Compared Soft Actor-Critic算法及其应用 

---
# Scaling Diffusion Transformers Efficiently via $μ$P 

**Title (ZH)**: 通过μP高效缩放扩散变换器 

**Authors**: Chenyu Zheng, Xinyu Zhang, Rongzhen Wang, Wei Huang, Zhi Tian, Weilin Huang, Jun Zhu, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15270)  

**Abstract**: Diffusion Transformers have emerged as the foundation for vision generative models, but their scalability is limited by the high cost of hyperparameter (HP) tuning at large scales. Recently, Maximal Update Parametrization ($\mu$P) was proposed for vanilla Transformers, which enables stable HP transfer from small to large language models, and dramatically reduces tuning costs. However, it remains unclear whether $\mu$P of vanilla Transformers extends to diffusion Transformers, which differ architecturally and objectively. In this work, we generalize standard $\mu$P to diffusion Transformers and validate its effectiveness through large-scale experiments. First, we rigorously prove that $\mu$P of mainstream diffusion Transformers, including DiT, U-ViT, PixArt-$\alpha$, and MMDiT, aligns with that of the vanilla Transformer, enabling the direct application of existing $\mu$P methodologies. Leveraging this result, we systematically demonstrate that DiT-$\mu$P enjoys robust HP transferability. Notably, DiT-XL-2-$\mu$P with transferred learning rate achieves 2.9 times faster convergence than the original DiT-XL-2. Finally, we validate the effectiveness of $\mu$P on text-to-image generation by scaling PixArt-$\alpha$ from 0.04B to 0.61B and MMDiT from 0.18B to 18B. In both cases, models under $\mu$P outperform their respective baselines while requiring small tuning cost, only 5.5% of one training run for PixArt-$\alpha$ and 3% of consumption by human experts for MMDiT-18B. These results establish $\mu$P as a principled and efficient framework for scaling diffusion Transformers. 

**Abstract (ZH)**: Maximal Update Parametrization for Scaling Diffusion Transformers 

---
# Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs 

**Title (ZH)**: 盲点导航：面向LVLMs的敏感语义概念进化发现 

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15265)  

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research. 

**Abstract (ZH)**: 对抗攻击旨在生成恶意输入以误导深度模型，但这些攻击无法提供某些可解析的信息，如“是什么内容使得模型更有可能出错？”然而，这些信息对于研究人员具体提高模型鲁棒性至关重要。近期研究表明，模型可能特别容易对视觉输入中的某些语义（如“湿润”、“雾蒙蒙”）敏感，从而使它们容易出错。受此启发，本文首次探讨了大型视觉-语言模型（LVLMs），发现LVLMs在面对图像中特定语义概念时确实容易产生幻觉并出现各种错误。为了高效地搜索这些敏感概念，我们将大型语言模型（LLMs）和文本到图像（T2I）模型集成到一个新颖的语义进化框架中。随机初始化的语义概念通过LLM基于的交叉和变异操作生成图像描述，然后通过T2I模型转换为视觉输入供LVLMs使用。LVLMs在每个输入上的任务特定性能被量化为涉及语义的适应度分数，并作为奖励信号进一步引导LLMs探索导致LVLMs出错的概念。在七个主流LVLMs和两个多模态任务上的广泛实验表明了本方法的有效性。此外，我们提供了关于LVLMs敏感语义的有趣发现，旨在启发进一步深入的研究。 

---
# Zero-Shot Gaze-based Volumetric Medical Image Segmentation 

**Title (ZH)**: 零样本基于视点的体绘制医学图像分割 

**Authors**: Tatyana Shmykova, Leila Khaertdinova, Ilya Pershin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15256)  

**Abstract**: Accurate segmentation of anatomical structures in volumetric medical images is crucial for clinical applications, including disease monitoring and cancer treatment planning. Contemporary interactive segmentation models, such as Segment Anything Model 2 (SAM-2) and its medical variant (MedSAM-2), rely on manually provided prompts like bounding boxes and mouse clicks. In this study, we introduce eye gaze as a novel informational modality for interactive segmentation, marking the application of eye-tracking for 3D medical image segmentation. We evaluate the performance of using gaze-based prompts with SAM-2 and MedSAM-2 using both synthetic and real gaze data. Compared to bounding boxes, gaze-based prompts offer a time-efficient interaction approach with slightly lower segmentation quality. Our findings highlight the potential of using gaze as a complementary input modality for interactive 3D medical image segmentation. 

**Abstract (ZH)**: 基于视线的交互分割在 volumetric 医学图像中解剖结构分割的精确性对于临床应用如疾病监测和癌症治疗计划至关重要。我们引入视线作为交互分割的新型信息模态，标志着眼动追踪在三维医学图像分割中的应用。我们使用合成和真实视线数据评估基于视线的提示在使用 SAM-2 和 MedSAM-2 中的性能。与边界框相比，基于视线的提示提供了一种时间效率更高的交互方法，但分割质量略低。我们的研究结果强调了将视线作为交互三维医学图像分割的补充输入模态的潜力。 

---
# Margin-aware Fuzzy Rough Feature Selection: Bridging Uncertainty Characterization and Pattern Classification 

**Title (ZH)**: 面向边距的模糊粗糙特征选择：不确定性表征与模式分类的桥梁 

**Authors**: Suping Xu, Lin Shang, Keyu Liu, Hengrong Ju, Xibei Yang, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2505.15250)  

**Abstract**: Fuzzy rough feature selection (FRFS) is an effective means of addressing the curse of dimensionality in high-dimensional data. By removing redundant and irrelevant features, FRFS helps mitigate classifier overfitting, enhance generalization performance, and lessen computational overhead. However, most existing FRFS algorithms primarily focus on reducing uncertainty in pattern classification, neglecting that lower uncertainty does not necessarily result in improved classification performance, despite it commonly being regarded as a key indicator of feature selection effectiveness in the FRFS literature. To bridge uncertainty characterization and pattern classification, we propose a Margin-aware Fuzzy Rough Feature Selection (MAFRFS) framework that considers both the compactness and separation of label classes. MAFRFS effectively reduces uncertainty in pattern classification tasks, while guiding the feature selection towards more separable and discriminative label class structures. Extensive experiments on 15 public datasets demonstrate that MAFRFS is highly scalable and more effective than FRFS. The algorithms developed using MAFRFS outperform six state-of-the-art feature selection algorithms. 

**Abstract (ZH)**: 面向边界的模糊粗糙特征选择（MAFRFS）：一种同时考虑标签类紧致性和分离性的特征选择框架 

---
# Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework 

**Title (ZH)**: 面向可解释时间推理的大语言模型：一种结构感知生成框架 

**Authors**: Zihao Jiang, Ben Liu, Miao Peng, Wenjie Xu, Yao Xiao, Zhenyan Shan, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15245)  

**Abstract**: While large language models (LLMs) show great potential in temporal reasoning, most existing work focuses heavily on enhancing performance, often neglecting the explainable reasoning processes underlying the results. To address this gap, we introduce a comprehensive benchmark covering a wide range of temporal granularities, designed to systematically evaluate LLMs' capabilities in explainable temporal reasoning. Furthermore, our findings reveal that LLMs struggle to deliver convincing explanations when relying solely on textual information. To address challenge, we propose GETER, a novel structure-aware generative framework that integrates Graph structures with text for Explainable TEmporal Reasoning. Specifically, we first leverage temporal knowledge graphs to develop a temporal encoder that captures structural information for the query. Subsequently, we introduce a structure-text prefix adapter to map graph structure features into the text embedding space. Finally, LLMs generate explanation text by seamlessly integrating the soft graph token with instruction-tuning prompt tokens. Experimental results indicate that GETER achieves state-of-the-art performance while also demonstrating its effectiveness as well as strong generalization capabilities. Our dataset and code are available at this https URL. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在时间推理方面显示出巨大的潜力，但现有大多数工作集中在提升性能上，往往忽视了支撑结果的可解释推理过程。为弥补这一差距，我们提出一个全面的时间粒度基准，旨在系统性地评估LLMs在可解释时间推理方面的能力。此外，我们的研究发现LLMs在仅依赖文本信息时难以提供令人信服的解释。为此，我们提出GETER，一种新颖的结构感知生成框架，将图结构与文本结合用于可解释时间推理。具体地，我们利用时间知识图谱开发时间编码器以捕捉查询的结构信息。随后，我们引入结构-文本前缀适配器将图结构特征映射至文本嵌入空间。最后，LLMs通过无缝整合软图令牌与指令调优提示令牌生成解释文本。实验结果表明，GETER在性能上达到了最新水平，同时展示了其有效性及强大的泛化能力。我们的数据集和代码可在以下链接获取。 

---
# Adaptive Plan-Execute Framework for Smart Contract Security Auditing 

**Title (ZH)**: 智能合约安全审计的自适应计划-执行框架 

**Authors**: Zhiyuan Wei, Jing Sun, Zijian Zhang, Zhe Hou, Zixiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.15242)  

**Abstract**: Large Language Models (LLMs) have shown great promise in code analysis and auditing; however, they still struggle with hallucinations and limited context-aware reasoning. We introduce SmartAuditFlow, a novel Plan-Execute framework that enhances smart contract security analysis through dynamic audit planning and structured execution. Unlike conventional LLM-based auditing approaches that follow fixed workflows and predefined steps, SmartAuditFlow dynamically generates and refines audit plans based on the unique characteristics of each smart contract. It continuously adjusts its auditing strategy in response to intermediate LLM outputs and newly detected vulnerabilities, ensuring a more adaptive and precise security assessment. The framework then executes these plans step by step, applying a structured reasoning process to enhance vulnerability detection accuracy while minimizing hallucinations and false positives. To further improve audit precision, SmartAuditFlow integrates iterative prompt optimization and external knowledge sources, such as static analysis tools and Retrieval-Augmented Generation (RAG). This ensures audit decisions are contextually informed and backed by real-world security knowledge, producing comprehensive security reports. Extensive evaluations across multiple benchmarks demonstrate that SmartAuditFlow outperforms existing methods, achieving 100 percent accuracy on common and critical vulnerabilities, 41.2 percent accuracy for comprehensive coverage of known smart contract weaknesses in real-world projects, and successfully identifying all 13 tested CVEs. These results highlight SmartAuditFlow's scalability, cost-effectiveness, and superior adaptability over traditional static analysis tools and contemporary LLM-based approaches, establishing it as a robust solution for automated smart contract auditing. 

**Abstract (ZH)**: SmartAuditFlow：一种用于智能合约安全分析的新型计划-执行框架 

---
# Neural Collapse is Globally Optimal in Deep Regularized ResNets and Transformers 

**Title (ZH)**: 神经网络坍缩在深度正则化ResNets和Transformer中是全局最优的 

**Authors**: Peter Súkeník, Christoph H. Lampert, Marco Mondelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.15239)  

**Abstract**: The empirical emergence of neural collapse -- a surprising symmetry in the feature representations of the training data in the penultimate layer of deep neural networks -- has spurred a line of theoretical research aimed at its understanding. However, existing work focuses on data-agnostic models or, when data structure is taken into account, it remains limited to multi-layer perceptrons. Our paper fills both these gaps by analyzing modern architectures in a data-aware regime: we prove that global optima of deep regularized transformers and residual networks (ResNets) with LayerNorm trained with cross entropy or mean squared error loss are approximately collapsed, and the approximation gets tighter as the depth grows. More generally, we formally reduce any end-to-end large-depth ResNet or transformer training into an equivalent unconstrained features model, thus justifying its wide use in the literature even beyond data-agnostic settings. Our theoretical results are supported by experiments on computer vision and language datasets showing that, as the depth grows, neural collapse indeed becomes more prominent. 

**Abstract (ZH)**: 神经坍缩的经验涌现——深神经网络次末端层训练数据的特征表示中的一种令人惊讶的对称性——激发了对其理解的理论研究。然而，现有工作专注于数据无关模型，或者在考虑数据结构时仅限于多层感知机。我们的论文通过在数据感知框架下分析现代架构填补了这两项空白：我们证明了采用交叉熵或均方误差损失训练的正则化深层变压器和残差网络（ResNets）的全局最优解大约会坍缩，且随着深度的增加，坍缩程度更加紧密。更为一般地，我们将任何端到端的大深度ResNet或变压器训练形式上归约为无约束特征模型的等价模型，从而证明了其在数据无关设置之外的广泛应用。我们的理论成果通过计算机视觉和语言数据集上的实验证明，在深度增加时，神经坍缩现象确实更加显著。 

---
# SAMA-UNet: Enhancing Medical Image Segmentation with Self-Adaptive Mamba-Like Attention and Causal-Resonance Learning 

**Title (ZH)**: SAMA-UNet: 通过自适应Mamba-like注意力和因果共振学习增强医学图像分割 

**Authors**: Saqib Qamar, Mohd Fazil, Parvez Ahmad, Ghulam Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2505.15234)  

**Abstract**: Medical image segmentation plays an important role in various clinical applications, but existing models often struggle with the computational inefficiencies and challenges posed by complex medical data. State Space Sequence Models (SSMs) have demonstrated promise in modeling long-range dependencies with linear computational complexity, yet their application in medical image segmentation remains hindered by incompatibilities with image tokens and autoregressive assumptions. Moreover, it is difficult to achieve a balance in capturing both local fine-grained information and global semantic dependencies. To address these challenges, we introduce SAMA-UNet, a novel architecture for medical image segmentation. A key innovation is the Self-Adaptive Mamba-like Aggregated Attention (SAMA) block, which integrates contextual self-attention with dynamic weight modulation to prioritise the most relevant features based on local and global contexts. This approach reduces computational complexity and improves the representation of complex image features across multiple scales. We also suggest the Causal-Resonance Multi-Scale Module (CR-MSM), which enhances the flow of information between the encoder and decoder by using causal resonance learning. This mechanism allows the model to automatically adjust feature resolution and causal dependencies across scales, leading to better semantic alignment between the low-level and high-level features in U-shaped architectures. Experiments on MRI, CT, and endoscopy images show that SAMA-UNet performs better in segmentation accuracy than current methods using CNN, Transformer, and Mamba. The implementation is publicly available at GitHub. 

**Abstract (ZH)**: 医学图像分割在各种临床应用中起着重要作用，但现有模型往往难以应对复杂医学数据带来的计算效率低下和挑战。状态空间序列模型（SSMs）在以线性计算复杂度建模长距离依赖方面显示出潜力，但在医学图像分割中的应用受到与图像令牌不兼容和自回归假设的阻碍。此外，难以在捕捉局部细微信息和全局语义依赖性之间取得平衡。为了解决这些挑战，我们提出了SAMA-UNet，一种新型的医学图像分割架构。关键创新是Self-Adaptive Mamba-like Aggregated Attention（SAMA）块，它结合了上下文自注意力和动态权重调制，根据局部和全局上下文优先选择最相关的特征。这种方法减少了计算复杂度并提高了跨多个尺度的复杂图像特征的表示能力。我们还提议使用Causal-Resonance Multi-Scale Module（CR-MSM），该模块通过因果共振学习增强编码器和解码器之间的信息流。这种机制允许模型自适应地调整不同尺度下的特征分辨率和因果依赖性，提高U型架构中低级和高级特征的语义对齐。在MRI、CT和内镜图像上的实验表明，SAMA-UNet在分割准确性方面优于使用CNN、Transformer和Mamba的现有方法。代码已开源在GitHub上。 

---
# BountyBench: Dollar Impact of AI Agent Attackers and Defenders on Real-World Cybersecurity Systems 

**Title (ZH)**: BountyBench: AI代理攻击者和防御者对实际网络安全系统经济影响的研究 

**Authors**: Andy K. Zhang, Joey Ji, Celeste Menders, Riya Dulepet, Thomas Qin, Ron Y. Wang, Junrong Wu, Kyleen Liao, Jiliang Li, Jinghan Hu, Sara Hong, Nardos Demilew, Shivatmica Murgai, Jason Tran, Nishka Kacheria, Ethan Ho, Denis Liu, Lauren McLane, Olivia Bruvik, Dai-Rong Han, Seungwoo Kim, Akhil Vyas, Cuiyuanxiu Chen, Ryan Li, Weiran Xu, Jonathan Z. Ye, Prerit Choudhary, Siddharth M. Bhatia, Vikram Sivashankar, Yuxuan Bao, Dawn Song, Dan Boneh, Daniel E. Ho, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15216)  

**Abstract**: AI agents have the potential to significantly alter the cybersecurity landscape. To help us understand this change, we introduce the first framework to capture offensive and defensive cyber-capabilities in evolving real-world systems. Instantiating this framework with BountyBench, we set up 25 systems with complex, real-world codebases. To capture the vulnerability lifecycle, we define three task types: Detect (detecting a new vulnerability), Exploit (exploiting a specific vulnerability), and Patch (patching a specific vulnerability). For Detect, we construct a new success indicator, which is general across vulnerability types and provides localized evaluation. We manually set up the environment for each system, including installing packages, setting up server(s), and hydrating database(s). We add 40 bug bounties, which are vulnerabilities with monetary awards from \$10 to \$30,485, and cover 9 of the OWASP Top 10 Risks. To modulate task difficulty, we devise a new strategy based on information to guide detection, interpolating from identifying a zero day to exploiting a specific vulnerability. We evaluate 5 agents: Claude Code, OpenAI Codex CLI, and custom agents with GPT-4.1, Gemini 2.5 Pro Preview, and Claude 3.7 Sonnet Thinking. Given up to three attempts, the top-performing agents are Claude Code (5% on Detect, mapping to \$1,350), Custom Agent with Claude 3.7 Sonnet Thinking (5% on Detect, mapping to \$1,025; 67.5% on Exploit), and OpenAI Codex CLI (5% on Detect, mapping to \$2,400; 90% on Patch, mapping to \$14,422). OpenAI Codex CLI and Claude Code are more capable at defense, achieving higher Patch scores of 90% and 87.5%, compared to Exploit scores of 32.5% and 57.5% respectively; in contrast, the custom agents are relatively balanced between offense and defense, achieving Exploit scores of 40-67.5% and Patch scores of 45-60%. 

**Abstract (ZH)**: AI代理有潜力显著改变网络安全格局。为了帮助我们理解这一变化，我们提出首个框架以捕捉演变中的现实世界系统中的 Offensive 和 Defensive 网络能力。通过使用 BountyBench，我们设置了包含复杂现实代码库的 25 个系统。为了捕捉漏洞生命周期，我们定义了三种任务类型：Detect（检测新漏洞）、Exploit（利用特定漏洞）和Patch（修复特定漏洞）。对于 Detect，我们构建了一个新的成功指标，该指标适用于不同类型的漏洞，并提供局部评估。我们为每个系统手工配置环境，包括安装软件包、设置服务器和填充数据库。我们添加了 40 个漏洞赏金，这些赏金包括 10 美元到 30,485 美元不等的金钱奖励，并覆盖了 OWASP Top 10 中的 9 项风险。为了调节任务难度，我们设计了一种新的策略，基于信息来指导检测，从识别零日漏洞逐步到利用特定漏洞。我们评估了 5 个代理：Claude Code、OpenAI Codex CLI，以及分别使用 GPT-4.1、Gemini 2.5 Pro Preview 和 Claude 3.7 Sonnet Thinking 的定制代理。在最多三次尝试的情况下，表现最好的代理是 Claude Code（在 Detect 中达到 5%，相当于 1,350 美元），定制代理 Claude 3.7 Sonnet Thinking（在 Detect 中达到 5%，相当于 1,025 美元；在 Exploit 中达到 67.5%），以及 OpenAI Codex CLI（在 Detect 中达到 5%，相当于 2,400 美元；在 Patch 中达到 90%，相当于 14,422 美元）。OpenAI Codex CLI 和 Claude Code 在防御方面表现更加出色，分别实现高达 90% 和 87.5% 的 Patch 分数，相比之下，Exploit 分数分别为 32.5% 和 57.5%；相比之下，定制代理在这两者之间相对平衡，Exploit 分数在 40% 至 67.5% 之间，Patch 分数在 45% 至 60% 之间。 

---
# EndoVLA: Dual-Phase Vision-Language-Action Model for Autonomous Tracking in Endoscopy 

**Title (ZH)**: EndoVLA: 自主内镜跟踪的双阶段视觉-语言-行动模型 

**Authors**: Chi Kit Ng, Long Bai, Guankun Wang, Yupeng Wang, Huxin Gao, Kun Yuan, Chenhan Jin, Tieyong Zeng, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.15206)  

**Abstract**: In endoscopic procedures, autonomous tracking of abnormal regions and following circumferential cutting markers can significantly reduce the cognitive burden on endoscopists. However, conventional model-based pipelines are fragile for each component (e.g., detection, motion planning) requires manual tuning and struggles to incorporate high-level endoscopic intent, leading to poor generalization across diverse scenes. Vision-Language-Action (VLA) models, which integrate visual perception, language grounding, and motion planning within an end-to-end framework, offer a promising alternative by semantically adapting to surgeon prompts without manual recalibration. Despite their potential, applying VLA models to robotic endoscopy presents unique challenges due to the complex and dynamic anatomical environments of the gastrointestinal (GI) tract. To address this, we introduce EndoVLA, designed specifically for continuum robots in GI interventions. Given endoscopic images and surgeon-issued tracking prompts, EndoVLA performs three core tasks: (1) polyp tracking, (2) delineation and following of abnormal mucosal regions, and (3) adherence to circular markers during circumferential cutting. To tackle data scarcity and domain shifts, we propose a dual-phase strategy comprising supervised fine-tuning on our EndoVLA-Motion dataset and reinforcement fine-tuning with task-aware rewards. Our approach significantly improves tracking performance in endoscopy and enables zero-shot generalization in diverse scenes and complex sequential tasks. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的内镜程序自主跟踪和标记跟随方法 

---
# Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems 

**Title (ZH)**: Pass@K策略优化：解决更棘手的强化学习问题 

**Authors**: Christian Walder, Deep Karkhanis  

**Link**: [PDF](https://arxiv.org/pdf/2505.15201)  

**Abstract**: Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes the strength of isolated samples at the expense of the diversity and collective utility of sets of samples. This under-utilizes the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a transformation on the final rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that maximize reward when considered jointly. Our contribution is to derive novel low variance unbiased estimators for pass@k and its gradient, in both the binary and continuous reward settings. We show optimization with our estimators reduces to standard RL with rewards that have been jointly transformed by a stable and efficient transformation function.
While previous efforts are restricted to k=n, ours is the first to enable robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 numbers alongside significant pass@k gains.
We validate our reward transformations on toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source LLM, GEMMA-2. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k . Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely due to better exploration by prioritizing joint utility over the utility of individual samples. 

**Abstract (ZH)**: 基于Passed-at-k策略优化的强化学习算法 

---
# Intentional Gesture: Deliver Your Intentions with Gestures for Speech 

**Title (ZH)**: 有意图手势：通过手势传达意图以辅助口语沟通 

**Authors**: Pinxin Liu, Haiyang Liu, Luchuan Song, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15197)  

**Abstract**: When humans speak, gestures help convey communicative intentions, such as adding emphasis or describing concepts. However, current co-speech gesture generation methods rely solely on superficial linguistic cues (\textit{e.g.} speech audio or text transcripts), neglecting to understand and leverage the communicative intention that underpins human gestures. This results in outputs that are rhythmically synchronized with speech but are semantically shallow. To address this gap, we introduce \textbf{Intentional-Gesture}, a novel framework that casts gesture generation as an intention-reasoning task grounded in high-level communicative functions. % First, we curate the \textbf{InG} dataset by augmenting BEAT-2 with gesture-intention annotations (\textit{i.e.}, text sentences summarizing intentions), which are automatically annotated using large vision-language models. Next, we introduce the \textbf{Intentional Gesture Motion Tokenizer} to leverage these intention annotations. It injects high-level communicative functions (\textit{e.g.}, intentions) into tokenized motion representations to enable intention-aware gesture synthesis that are both temporally aligned and semantically meaningful, achieving new state-of-the-art performance on the BEAT-2 benchmark. Our framework offers a modular foundation for expressive gesture generation in digital humans and embodied AI. Project Page: this https URL 

**Abstract (ZH)**: 当人类说话时，手势有助于传达交际意图，如强调或描述概念。然而，当前的同期手势生成方法仅依赖于表面语言线索（例如语音音频或文本转录），忽视了理解并利用支撑人类手势的交际意图。这导致了与语音节奏同步但语义浅显的输出。为了解决这一问题，我们介绍了**Intentional-Gesture**这一新型框架，将其手势生成任务视为基于高级交际功能的意图推理任务。首先，通过扩充BEAT-2数据集并添加手势意图注释（即总结意图的文字句子），并使用大规模的视觉语言模型自动标注这些注释，我们构建了**InG**数据集。接着，我们引入了**意图手势运动分词器**来利用这些意图注释。它将高级交际功能（例如意图）注入到标记化的运动表示中，以实现既时间对齐又有语义意义的意图感知手势合成，并在BEAT-2基准测试中达到了新的最优性能。我们的框架为数字人类和 embodiable AI 表达性手势生成提供了模块化基础。 

---
# ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection 

**Title (ZH)**: ReflAct：通过目标状态反思实现基于世界的决策制定在LLM代理中 

**Authors**: Jeonghye Kim, Sojeong Rhee, Minbeom Kim, Dohyung Kim, Sangmook Lee, Youngchul Sung, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.15182)  

**Abstract**: Recent advances in LLM agents have largely built on reasoning backbones like ReAct, which interleave thought and action in complex environments. However, ReAct often produces ungrounded or incoherent reasoning steps, leading to misalignment between the agent's actual state and goal. Our analysis finds that this stems from ReAct's inability to maintain consistent internal beliefs and goal alignment, causing compounding errors and hallucinations. To address this, we introduce ReflAct, a novel backbone that shifts reasoning from merely planning next actions to continuously reflecting on the agent's state relative to its goal. By explicitly grounding decisions in states and enforcing ongoing goal alignment, ReflAct dramatically improves strategic reliability. This design delivers substantial empirical gains: ReflAct surpasses ReAct by 27.7% on average, achieving a 93.3% success rate in ALFWorld. Notably, ReflAct even outperforms ReAct with added enhancement modules (e.g., Reflexion, WKM), showing that strengthening the core reasoning backbone is key to reliable agent performance. 

**Abstract (ZH)**: 最近在LLM代理方面的进展大多建立在像ReAct这样的推理骨干之上，这些骨干在复杂环境中交替进行思考和行动。然而，ReAct经常产生不着边际或不连贯的推理步骤，导致代理的实际状态与目标之间的不对齐。我们的分析发现，这是由于ReAct无法维持一致的内部信念和目标对齐，导致累积错误和幻觉的产生。为了解决这个问题，我们引入了ReflAct，这是一种新型的骨干，将推理的重心从仅仅规划下一个动作转移到持续反思代理的状态与目标之间的关系。通过明确地将决策基于状态并持续强化目标对齐，ReflAct显著提高了战略可靠性。这一设计带来了实质性的实证收益：ReflAct在平均值上超越了ReAct 27.7%，在ALFWorld中达到了93.3%的成功率。值得注意的是，即使在添加了增强模块（如Reflexion、WKM）的情况下，ReflAct仍然优于ReAct，显示了加强核心推理骨干对于可靠代理性能的重要性。 

---
# AvatarShield: Visual Reinforcement Learning for Human-Centric Video Forgery Detection 

**Title (ZH)**: AvatarShield：以人类为中心的视频篡改检测的可视强化学习 

**Authors**: Zhipei Xu, Xuanyu Zhang, Xing Zhou, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15173)  

**Abstract**: The rapid advancement of Artificial Intelligence Generated Content (AIGC) technologies, particularly in video generation, has led to unprecedented creative capabilities but also increased threats to information integrity, identity security, and public trust. Existing detection methods, while effective in general scenarios, lack robust solutions for human-centric videos, which pose greater risks due to their realism and potential for legal and ethical misuse. Moreover, current detection approaches often suffer from poor generalization, limited scalability, and reliance on labor-intensive supervised fine-tuning. To address these challenges, we propose AvatarShield, the first interpretable MLLM-based framework for detecting human-centric fake videos, enhanced via Group Relative Policy Optimization (GRPO). Through our carefully designed accuracy detection reward and temporal compensation reward, it effectively avoids the use of high-cost text annotation data, enabling precise temporal modeling and forgery detection. Meanwhile, we design a dual-encoder architecture, combining high-level semantic reasoning and low-level artifact amplification to guide MLLMs in effective forgery detection. We further collect FakeHumanVid, a large-scale human-centric video benchmark that includes synthesis methods guided by pose, audio, and text inputs, enabling rigorous evaluation of detection methods in real-world scenes. Extensive experiments show that AvatarShield significantly outperforms existing approaches in both in-domain and cross-domain detection, setting a new standard for human-centric video forensics. 

**Abstract (ZH)**: AIGC技术的迅速发展，特别是在视频生成领域的应用，带来了前所未有的创作能力，但也增加了信息完整性、身份安全和公众信任等方面的风险。现有的检测方法在一般场景中有效，但缺乏针对以人类为中心的视频的鲁棒解决方案，后者由于其逼真性和潜在的法律和伦理滥用风险而面临更大的挑战。此外，当前的检测方法通常存在泛化能力差、扩展能力有限以及对劳动密集型监督微调的依赖等问题。为应对这些挑战，我们提出AvatarShield——首个基于可解释多模态语言模型的检测以人类为中心的伪造视频框架，该框架通过组相对策略优化（GRPO）进行增强。通过精心设计的准确性检测奖励和时间补偿奖励，AvatarShield有效地避免了高成本文本注释数据的使用，实现了精确的时间建模和伪造检测。同时，我们设计了双编码器架构，结合高层次语义推理和低层次伪迹放大，引导多模态语言模型进行有效的伪造检测。我们进一步收集了FakeHumanVid大规模以人类为中心的视频基准集，该基准集包括由姿态、音频和文本输入指导的合成方法，为检测方法在实际场景中的严格评估提供了依据。广泛实验表明，AvatarShield在领域内和跨领域的检测中均显著优于现有方法，确立了以人类为中心的视频取证的新标准。 

---
# R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization 

**Title (ZH)**: R&D-Agent-Quant: 以数据为中心的因素与模型联合优化的多智能体框架 

**Authors**: Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2505.15155)  

**Abstract**: Financial markets pose fundamental challenges for asset return prediction due to their high dimensionality, non-stationarity, and persistent volatility. Despite advances in large language models and multi-agent systems, current quantitative research pipelines suffer from limited automation, weak interpretability, and fragmented coordination across key components such as factor mining and model innovation. In this paper, we propose R&D-Agent for Quantitative Finance, in short RD-Agent(Q), the first data-centric multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization. RD-Agent(Q) decomposes the quant process into two iterative stages: a Research stage that dynamically sets goal-aligned prompts, formulates hypotheses based on domain priors, and maps them to concrete tasks, and a Development stage that employs a code-generation agent, Co-STEER, to implement task-specific code, which is then executed in real-market backtests. The two stages are connected through a feedback stage that thoroughly evaluates experimental outcomes and informs subsequent iterations, with a multi-armed bandit scheduler for adaptive direction selection. Empirically, RD-Agent(Q) achieves up to 2X higher annualized returns than classical factor libraries using 70% fewer factors, and outperforms state-of-the-art deep time-series models on real markets. Its joint factor-model optimization delivers a strong balance between predictive accuracy and strategy robustness. Our code is available at: this https URL. 

**Abstract (ZH)**: 金融市场的高维特性、非平稳性及持续的波动性给资产回报预测提出了基本挑战。尽管大型语言模型和多智能体系统取得了进展，当前的定量研究工作流程仍然面临自动化程度有限、解释性较弱以及关键组件（如因子挖掘和模型创新）之间协调性差的问题。本文提出了一种面向研究与开发的Quantitative Finance智能体（RD-Agent(Q)），这是第一个通过协同因子-模型协同优化来自动化定量策略全流程研究与发展的数据驱动型多智能体框架。RD-Agent(Q)将量化过程分解为两个迭代阶段：研究阶段动态设定目标对齐的提示，根据领域先验制定假设，并将其映射为具体的任务；开发阶段采用代码生成智能体Co-STEER来实现特定任务的代码，然后在实盘回测中执行。两个阶段通过一个反馈阶段连接，该阶段全面评估实验结果并为后续迭代提供反馈，使用多臂老虎机调度器进行自适应方向选择。实证结果显示，RD-Agent(Q)在使用较少因子（减少70%）的情况下，实现了最高2倍的年化回报，并在实盘市场中超越最先进的深度时间序列模型，其联合因子-模型优化提供了预测准确性和策略稳健性的良好平衡。GitHub代码地址：this https URL。 

---
# Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning 

**Title (ZH)**: 长时间推理并非全靠而已：基于 certainty 的自适应路由以实现高效的 LL offense 联合推理 

**Authors**: Jinghui Lu, Haiyang Yu, Siliang Xu, Shiwei Ran, Guozhi Tang, Siqi Wang, Bin Shan, Teng Fu, Hao Feng, Jingqun Tang, Han Wang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15154)  

**Abstract**: Recent advancements in reasoning have significantly enhanced the capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) across diverse tasks. However, excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs, reducing efficiency. Our work reveals that prolonged reasoning does not universally improve accuracy and even degrade performance on simpler tasks. To address this, we propose Certainty-based Adaptive Reasoning (CAR), a novel framework that dynamically switches between short answers and long-form reasoning based on the model perplexity. CAR first generates a short answer and evaluates its perplexity, triggering reasoning only when the model exhibits low confidence (i.e., high perplexity). Experiments across diverse multimodal VQA/KIE benchmarks and text reasoning datasets show that CAR outperforms both short-answer and long-form reasoning approaches, striking an optimal balance between accuracy and efficiency. 

**Abstract (ZH)**: 近期在推理方面的进展显著增强了大语言模型（LLMs）和多模态大语言模型（MLLMs）在各种任务中的能力。然而，过度依赖链式思考（CoT）推理会损害模型性能并导致不必要的输出延长，降低效率。我们的工作揭示了长时间推理并不普遍提高准确性，甚至在简单任务上降低性能。为了解决这个问题，我们提出了基于 certainty 的自适应推理（CAR）框架，该框架会根据模型困惑度动态切换短答案和长形式推理。CAR 首先生成一个短答案并评估其困惑度，仅当模型表现出低置信度（即高困惑度）时才触发推理。实验结果显示，CAR 在多种多模态 VQA/KIE 基准和文本推理数据集中均优于短答案和长形式推理方法，实现了准确性和效率之间的最佳平衡。 

---
# BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms 

**Title (ZH)**: BanditSpec: 通过bandit算法实现自适应投机解码 

**Authors**: Yunlong Hou, Fengzhuo Zhang, Cunxiao Du, Xuan Zhang, Jiachun Pan, Tianyu Pang, Chao Du, Vincent Y. F. Tan, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15141)  

**Abstract**: Speculative decoding has emerged as a popular method to accelerate the inference of Large Language Models (LLMs) while retaining their superior text generation performance. Previous methods either adopt a fixed speculative decoding configuration regardless of the prefix tokens, or train draft models in an offline or online manner to align them with the context. This paper proposes a training-free online learning framework to adaptively choose the configuration of the hyperparameters for speculative decoding as text is being generated. We first formulate this hyperparameter selection problem as a Multi-Armed Bandit problem and provide a general speculative decoding framework BanditSpec. Furthermore, two bandit-based hyperparameter selection algorithms, UCBSpec and EXP3Spec, are designed and analyzed in terms of a novel quantity, the stopping time regret. We upper bound this regret under both stochastic and adversarial reward settings. By deriving an information-theoretic impossibility result, it is shown that the regret performance of UCBSpec is optimal up to universal constants. Finally, extensive empirical experiments with LLaMA3 and Qwen2 demonstrate that our algorithms are effective compared to existing methods, and the throughput is close to the oracle best hyperparameter in simulated real-life LLM serving scenarios with diverse input prompts. 

**Abstract (ZH)**: 无训练在线学习框架以适应性选择 speculate 解码超参数以加速大型语言模型的推理 

---
# Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor Critic Algorithm 

**Title (ZH)**: 全局收敛性约束平均奖励MDP的 primal-dual 奖励 critic 算法 

**Authors**: Yang Xu, Swetha Ganesh, Washim Uddin Mondal, Qinbo Bai, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.15138)  

**Abstract**: This paper investigates infinite-horizon average reward Constrained Markov Decision Processes (CMDPs) with general parametrization. We propose a Primal-Dual Natural Actor-Critic algorithm that adeptly manages constraints while ensuring a high convergence rate. In particular, our algorithm achieves global convergence and constraint violation rates of $\tilde{\mathcal{O}}(1/\sqrt{T})$ over a horizon of length $T$ when the mixing time, $\tau_{\mathrm{mix}}$, is known to the learner. In absence of knowledge of $\tau_{\mathrm{mix}}$, the achievable rates change to $\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$ provided that $T \geq \tilde{\mathcal{O}}\left(\tau_{\mathrm{mix}}^{2/\epsilon}\right)$. Our results match the theoretical lower bound for Markov Decision Processes and establish a new benchmark in the theoretical exploration of average reward CMDPs. 

**Abstract (ZH)**: 这篇论文研究了具有通用参数化的无穷_horizon平均奖励约束马尔科夫决策过程（CMDPs）。我们提出了一种普欧-对偶自然Actor-Critic算法，在确保高收敛率的同时巧妙处理约束。特别是在混合时间$\tau_{\mathrm{mix}}$已知给学习者的情况下，我们的算法在长度为$T$的horizon上实现了全局收敛和约束违反率为$\tilde{\mathcal{O}}(1/\sqrt{T})$。在$\tau_{\mathrm{mix}}$未知的情况下，如果满足$T \geq \tilde{\mathcal{O}}\left(\tau_{\mathrm{mix}}^{2/\epsilon}\right)$，可实现的速率变为$\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$。我们的结果匹配了马尔科夫决策过程的理论下界，并在平均奖励CMDPs的理论探索中建立了新的基准。 

---
# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning 

**Title (ZH)**: 熵最小化在大规模语言模型推理中的出人意料的有效性 

**Authors**: Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15134)  

**Abstract**: Entropy minimization (EM) trains the model to concentrate even more probability mass on its most confident outputs. We show that this simple objective alone, without any labeled data, can substantially improve large language models' (LLMs) performance on challenging math, physics, and coding tasks. We explore three approaches: (1) EM-FT minimizes token-level entropy similarly to instruction finetuning, but on unlabeled outputs drawn from the model; (2) EM-RL: reinforcement learning with negative entropy as the only reward to maximize; (3) EM-INF: inference-time logit adjustment to reduce entropy without any training data or parameter updates. On Qwen-7B, EM-RL, without any labeled data, achieves comparable or better performance than strong RL baselines such as GRPO and RLOO that are trained on 60K labeled examples. Furthermore, EM-INF enables Qwen-32B to match or exceed the performance of proprietary models like GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro on the challenging SciCode benchmark, while being 3x more efficient than self-consistency and sequential refinement. Our findings reveal that many pretrained LLMs possess previously underappreciated reasoning capabilities that can be effectively elicited through entropy minimization alone, without any labeled data or even any parameter updates. 

**Abstract (ZH)**: 熵最小化（EM）使模型更集中于其最自信的输出。我们证明，仅此简单的目标，在没有任何标注数据的情况下，可以显著提高大型语言模型（LLMs）在挑战性的数学、物理和编程任务上的性能。我们探索了三种方法：（1）EM-FT在未标注的模型输出上像指令微调一样最小化令牌级熵；（2）EM-RL：仅以负熵作为奖励的最大化强化学习；（3）EM-INF：推理时的logit调整以减少熵，无需任何训练数据或参数更新。在Qwen-7B中，EM-RL在没有任何标注数据的情况下，取得了与基于6万个标注样例训练的强RL基线（如GRPO和RLOO）相当或更好的性能。此外，EM-INF使Qwen-32B能够匹配或超越自产模型（如GPT-4o、Claude 3 Opus和Gemini 1.5 Pro）在具有挑战性的SciCode基准测试中的性能，且效率高出3倍于自我一致性与序列完善。我们的研究结果表明，许多预训练LLM拥有以前被低估的推理能力，这些能力可以通过熵最小化等简单的任务激活，无需标注数据或参数更新。 

---
# DeepKD: A Deeply Decoupled and Denoised Knowledge Distillation Trainer 

**Title (ZH)**: DeepKD：一种深度解耦和去噪的知识蒸馏训练器 

**Authors**: Haiduo Huang, Jiangcheng Song, Yadong Zhang, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.15133)  

**Abstract**: Recent advances in knowledge distillation have emphasized the importance of decoupling different knowledge components. While existing methods utilize momentum mechanisms to separate task-oriented and distillation gradients, they overlook the inherent conflict between target-class and non-target-class knowledge flows. Furthermore, low-confidence dark knowledge in non-target classes introduces noisy signals that hinder effective knowledge transfer. To address these limitations, we propose DeepKD, a novel training framework that integrates dual-level decoupling with adaptive denoising. First, through theoretical analysis of gradient signal-to-noise ratio (GSNR) characteristics in task-oriented and non-task-oriented knowledge distillation, we design independent momentum updaters for each component to prevent mutual interference. We observe that the optimal momentum coefficients for task-oriented gradient (TOG), target-class gradient (TCG), and non-target-class gradient (NCG) should be positively related to their GSNR. Second, we introduce a dynamic top-k mask (DTM) mechanism that gradually increases K from a small initial value to incorporate more non-target classes as training progresses, following curriculum learning principles. The DTM jointly filters low-confidence logits from both teacher and student models, effectively purifying dark knowledge during early training. Extensive experiments on CIFAR-100, ImageNet, and MS-COCO demonstrate DeepKD's effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: Recent Advances in Knowledge Distillation Have Emphasized the Importance of Decoupling Different Knowledge Components: Addressing the Limitations with DeepKD, a Novel Training Framework that Integrates Dual-Level Decoupling with Adaptive Denoising 

---
# Seeing the Trees for the Forest: Rethinking Weakly-Supervised Medical Visual Grounding 

**Title (ZH)**: 见森林中之树木：重新思考弱监督医疗视觉定位 

**Authors**: Ta Duc Huy, Duy Anh Huynh, Yutong Xie, Yuankai Qi, Qi Chen, Phi Le Nguyen, Sen Kim Tran, Son Lam Phung, Anton van den Hengel, Zhibin Liao, Minh-Son To, Johan W. Verjans, Vu Minh Hieu Phan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15123)  

**Abstract**: Visual grounding (VG) is the capability to identify the specific regions in an image associated with a particular text description. In medical imaging, VG enhances interpretability by highlighting relevant pathological features corresponding to textual descriptions, improving model transparency and trustworthiness for wider adoption of deep learning models in clinical practice. Current models struggle to associate textual descriptions with disease regions due to inefficient attention mechanisms and a lack of fine-grained token representations. In this paper, we empirically demonstrate two key observations. First, current VLMs assign high norms to background tokens, diverting the model's attention from regions of disease. Second, the global tokens used for cross-modal learning are not representative of local disease tokens. This hampers identifying correlations between the text and disease tokens. To address this, we introduce simple, yet effective Disease-Aware Prompting (DAP) process, which uses the explainability map of a VLM to identify the appropriate image features. This simple strategy amplifies disease-relevant regions while suppressing background interference. Without any additional pixel-level annotations, DAP improves visual grounding accuracy by 20.74% compared to state-of-the-art methods across three major chest X-ray datasets. 

**Abstract (ZH)**: 基于视觉定位的疾病意识提示在医学影像中的应用：一种简单的有效方法 

---
# An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents 

**Title (ZH)**: 强化学习在推理-搜索交织的大型语言模型代理上的实证研究 

**Authors**: Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.15117)  

**Abstract**: Reinforcement learning (RL) has demonstrated strong potential in training large language models (LLMs) capable of complex reasoning for real-world problem solving. More recently, RL has been leveraged to create sophisticated LLM-based search agents that adeptly combine reasoning with search engine use. While the use of RL for training search agents is promising, the optimal design of such agents remains not fully understood. In particular, key factors -- such as (1) reward formulation, (2) the choice and characteristics of the underlying LLM, and (3) the role of the search engine in the RL process -- require further investigation. In this work, we conduct comprehensive empirical studies to systematically investigate these and offer actionable insights. We highlight several key findings: format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact; the scale and initialization of the LLM (general-purpose vs. reasoning-specialized) significantly influence RL outcomes; and the choice of search engine plays a critical role in shaping RL training dynamics and the robustness of the trained agent during inference. These establish important guidelines for successfully building and deploying LLM-based search agents in real-world applications. Code is available at this https URL. 

**Abstract (ZH)**: 强化学习（RL）在训练具备复杂推理能力的大语言模型（LLMs）以解决实际问题方面展现了强大的潜力。最近，RL被用于创建能够熟练结合推理与搜索引擎使用的精妙LLM搜索代理。尽管使用RL训练搜索代理具有前景，但这类代理的最佳设计仍不完全清晰。特别是，关键因素包括（1）奖励的制定，（2）基础大语言模型的选择及其特性，以及（3）搜索引擎在RL过程中的作用，都需要进一步研究。本研究通过综合的实证研究系统地探讨这些因素，并提供了实用的见解。我们突出了一些关键发现：格式奖励有助于提升最终性能，而中间检索奖励的影响则有限；大语言模型的规模和初始化（通用型 vs. 专门用于推理）对RL结果有显著影响；搜索引擎的选择对RL训练动态以及推理期间代理的鲁棒性起着决定性作用。这些研究建立了在实际应用中成功构建和部署基于LLM的搜索代理的重要指导方针。代码可在以下链接获得：this https URL。 

---
# Graph Foundation Models: A Comprehensive Survey 

**Title (ZH)**: 图基础模型：综述 

**Authors**: Zehong Wang, Zheyuan Liu, Tianyi Ma, Jiazheng Li, Zheyuan Zhang, Xingbo Fu, Yiyang Li, Zhengqing Yuan, Wei Song, Yijun Ma, Qingkai Zeng, Xiusi Chen, Jianan Zhao, Jundong Li, Meng Jiang, Pietro Lio, Nitesh Chawla, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15116)  

**Abstract**: Graph-structured data pervades domains such as social networks, biological systems, knowledge graphs, and recommender systems. While foundation models have transformed natural language processing, vision, and multimodal learning through large-scale pretraining and generalization, extending these capabilities to graphs -- characterized by non-Euclidean structures and complex relational semantics -- poses unique challenges and opens new opportunities. To this end, Graph Foundation Models (GFMs) aim to bring scalable, general-purpose intelligence to structured data, enabling broad transfer across graph-centric tasks and domains. This survey provides a comprehensive overview of GFMs, unifying diverse efforts under a modular framework comprising three key components: backbone architectures, pretraining strategies, and adaptation mechanisms. We categorize GFMs by their generalization scope -- universal, task-specific, and domain-specific -- and review representative methods, key innovations, and theoretical insights within each category. Beyond methodology, we examine theoretical foundations including transferability and emergent capabilities, and highlight key challenges such as structural alignment, heterogeneity, scalability, and evaluation. Positioned at the intersection of graph learning and general-purpose AI, GFMs are poised to become foundational infrastructure for open-ended reasoning over structured data. This survey consolidates current progress and outlines future directions to guide research in this rapidly evolving field. Resources are available at this https URL. 

**Abstract (ZH)**: 图结构数据 pervades 领域如社交网络、生物系统、知识图谱和推荐系统。尽管基础模型通过大规模预训练和泛化已重塑自然语言处理、视觉和多模态学习，但将这些能力扩展到图数据——其特征是非欧几里得结构和复杂的关系语义——提出了独特挑战并开启了新的机遇。为此，图基础模型（GFMs）旨在为结构化数据带来可扩展的通用智能，使其能够在图中心任务和领域之间广泛转移。本文综述提供了对 GFMs 的全面概述，统一了在模块化框架下的各种努力，包括骨干架构、预训练策略和适应机制。我们按其泛化范围——通用、任务特定和领域特定——对 GFMs 进行分类，并在每个类别中回顾了代表性方法、关键创新和理论洞见。除了方法论，我们还探讨了理论基础，包括可迁移性和新兴能力，并强调了结构性对齐、异质性、可扩展性和评估等关键挑战。定位在图学习和通用人工智能的交界处，GFMs 被视为开放推理的结构化数据基础架构。本文综述汇总了当前进展，并指出了未来方向以指导该快速发展的领域中的研究。资源可访问此网址。 

---
# iPad: Iterative Proposal-centric End-to-End Autonomous Driving 

**Title (ZH)**: iPad: 迭代基于提案的端到端自动驾驶 

**Authors**: Ke Guo, Haochen Liu, Xiaojun Wu, Jia Pan, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2505.15111)  

**Abstract**: End-to-end (E2E) autonomous driving systems offer a promising alternative to traditional modular pipelines by reducing information loss and error accumulation, with significant potential to enhance both mobility and safety. However, most existing E2E approaches directly generate plans based on dense bird's-eye view (BEV) grid features, leading to inefficiency and limited planning awareness. To address these limitations, we propose iterative Proposal-centric autonomous driving (iPad), a novel framework that places proposals - a set of candidate future plans - at the center of feature extraction and auxiliary tasks. Central to iPad is ProFormer, a BEV encoder that iteratively refines proposals and their associated features through proposal-anchored attention, effectively fusing multi-view image data. Additionally, we introduce two lightweight, proposal-centric auxiliary tasks - mapping and prediction - that improve planning quality with minimal computational overhead. Extensive experiments on the NAVSIM and CARLA Bench2Drive benchmarks demonstrate that iPad achieves state-of-the-art performance while being significantly more efficient than prior leading methods. 

**Abstract (ZH)**: 端到端自主驾驶系统通过减少信息损失和错误累积，提供了一种传统模块化管道的有前途的替代方案，有望同时提高 Mobility 和 Safety。然而，现有大多数端到端方法直接基于密集的鸟瞰视图 (BEV) 栅格特征生成计划，导致效率低下且规划意识有限。为了解决这些限制，我们提出了迭代 Proposal-centric 自主驾驶 (iPad)，这是一种将提案——一组候选未来计划——置于特征提取和辅助任务中心的新型框架。iPad 的核心是 ProFormer，这是一种 BEV 编码器，通过提案锚定注意力迭代细化提案及其相关特征，有效地融合多视图图像数据。此外，我们引入了两个轻量级的提案-centric 辅助任务——制图和预测——以最小的计算开销提高规划质量。在 NAVSIM 和 CARLA Bench2Drive 基准测试中，iPad 在性能上达到最新水平，同时比先前的领先方法更为高效。 

---
# A Risk Taxonomy for Evaluating AI-Powered Psychotherapy Agents 

**Title (ZH)**: AI赋能心理疗法代理的风险分类 

**Authors**: Ian Steenstra, Timothy W. Bickmore  

**Link**: [PDF](https://arxiv.org/pdf/2505.15108)  

**Abstract**: The proliferation of Large Language Models (LLMs) and Intelligent Virtual Agents acting as psychotherapists presents significant opportunities for expanding mental healthcare access. However, their deployment has also been linked to serious adverse outcomes, including user harm and suicide, facilitated by a lack of standardized evaluation methodologies capable of capturing the nuanced risks of therapeutic interaction. Current evaluation techniques lack the sensitivity to detect subtle changes in patient cognition and behavior during therapy sessions that may lead to subsequent decompensation. We introduce a novel risk taxonomy specifically designed for the systematic evaluation of conversational AI psychotherapists. Developed through an iterative process including review of the psychotherapy risk literature, qualitative interviews with clinical and legal experts, and alignment with established clinical criteria (e.g., DSM-5) and existing assessment tools (e.g., NEQ, UE-ATR), the taxonomy aims to provide a structured approach to identifying and assessing user/patient harms. We provide a high-level overview of this taxonomy, detailing its grounding, and discuss potential use cases. We discuss two use cases in detail: monitoring cognitive model-based risk factors during a counseling conversation to detect unsafe deviations, in both human-AI counseling sessions and in automated benchmarking of AI psychotherapists with simulated patients. The proposed taxonomy offers a foundational step towards establishing safer and more responsible innovation in the domain of AI-driven mental health support. 

**Abstract (ZH)**: 大型语言模型（LLMs）和充当心理咨询师的智能虚拟代理的 proliferations 为扩展心理健康服务的可及性带来了重要机会，但它们的应用也与严重的负面后果相关联，包括用户伤害和自杀，这些都是由于缺乏能够捕捉治疗互动复杂风险的标准评估方法。当前的评估技术无法捕捉到治疗会话中患者认知和行为的微妙变化，这些变化可能导致后续的病情恶化。我们介绍了一种新型的风险分类学，特别设计用于系统评估对话式人工智能心理咨询师。该分类学通过迭代过程开发，包括心理治疗风险文献的回顾、与临床和法律专家的质性访谈，以及与现有的临床标准（如DSM-5）和评估工具（如NEQ，UE-ATR）的对齐。该分类学旨在提供一种结构化的方法来识别和评估用户/患者伤害。我们提供了该分类学的高层次概述，详细说明其理论基础，并讨论潜在的应用案例。我们详细讨论了两个应用案例：在咨询对话中监控基于认知模型的风险因素，以检测不安全的偏差，无论是人类-人工智能咨询会话还是自动化的基于模拟患者的AI心理咨询评估。提出的风险分类学为我们朝着建立更安全和更负责任的人工智能驱动心理健康支持领域创新奠定了基础。 

---
# StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization 

**Title (ZH)**: StepSearch：通过逐步近端策略优化激发大规模语言模型的检索能力 

**Authors**: Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15107)  

**Abstract**: Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our implementation is publicly available at this https URL. 

**Abstract (ZH)**: 高效多跳推理需要基于大规模语言模型的代理逐步迭代获取高价值外部知识。现有工作探索使用强化学习训练大规模语言模型进行基于搜索的文档检索，取得了显著的问答性能改进，但在从全球信号稀疏奖励中产生的复杂多跳问答任务上表现不佳。为弥补现有研究的这一差距，我们引入了StepSearch，一种基于逐步近端策略优化方法训练的搜索框架。它包含更丰富、更详细的中间搜索奖励和基于信息增益和冗余惩罚的token级别过程监督，以更好地引导每一步搜索。我们通过一套数据管道方法，根据开源数据集构建了一个粒度精细的问答数据集，包含子问题级别的搜索轨迹。在标准多跳问答基准测试上，它显著优于全局奖励基准，仅使用19,000条训练数据，3B和7B模型分别实现了11.2%和4.2%的绝对性能提升，证明了细致步骤监督在优化深层搜索大规模语言模型中的有效性。我们的实现可在以下网址获取：this https URL。 

---
# Mechanistic evaluation of Transformers and state space models 

**Title (ZH)**: Transformer和状态空间模型的机制评估 

**Authors**: Aryaman Arora, Neil Rathi, Nikil Roashan Selvam, Róbert Csórdas, Dan Jurafsky, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.15105)  

**Abstract**: State space models (SSMs) for language modelling promise an efficient and performant alternative to quadratic-attention Transformers, yet show variable performance on recalling basic information from the context. While performance on synthetic tasks like Associative Recall (AR) can point to this deficiency, behavioural metrics provide little information as to why--on a mechanistic level--certain architectures fail and others succeed. To address this, we conduct experiments on AR and find that only Transformers and Based SSM models fully succeed at AR, with Mamba a close third, whereas the other SSMs (H3, Hyena) fail. We then use causal interventions to explain why. We find that Transformers and Based learn to store key-value associations in-context using induction heads. By contrast, the SSMs compute these associations only at the last state, with only Mamba succeeding because of its short convolution component. To extend and deepen these findings, we introduce Associative Treecall (ATR), a synthetic task similar to AR based on PCFG induction. ATR introduces language-like hierarchical structure into the AR setting. We find that all architectures learn the same mechanism as they did for AR, and the same three models succeed at the task. These results reveal that architectures with similar accuracy may still have substantive differences, motivating the adoption of mechanistic evaluations. 

**Abstract (ZH)**: 状态空间模型（SSMs）在语言建模中的进展：基于诱导头的Transformer和基于诱导的SSM模型在关联回忆任务中表现出色，而其他SSM模型表现不佳 

---
# Object-Focus Actor for Data-efficient Robot Generalization Dexterous Manipulation 

**Title (ZH)**: 面向物体的关注代理：面向数据高效机器人 generalize 灵巧 manipulation 

**Authors**: Yihang Li, Tianle Zhang, Xuelong Wei, Jiayi Li, Lin Zhao, Dongchi Huang, Zhirui Fang, Minhua Zheng, Wenjun Dai, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15098)  

**Abstract**: Robot manipulation learning from human demonstrations offers a rapid means to acquire skills but often lacks generalization across diverse scenes and object placements. This limitation hinders real-world applications, particularly in complex tasks requiring dexterous manipulation. Vision-Language-Action (VLA) paradigm leverages large-scale data to enhance generalization. However, due to data scarcity, VLA's performance remains limited. In this work, we introduce Object-Focus Actor (OFA), a novel, data-efficient approach for generalized dexterous manipulation. OFA exploits the consistent end trajectories observed in dexterous manipulation tasks, allowing for efficient policy training. Our method employs a hierarchical pipeline: object perception and pose estimation, pre-manipulation pose arrival and OFA policy execution. This process ensures that the manipulation is focused and efficient, even in varied backgrounds and positional layout. Comprehensive real-world experiments across seven tasks demonstrate that OFA significantly outperforms baseline methods in both positional and background generalization tests. Notably, OFA achieves robust performance with only 10 demonstrations, highlighting its data efficiency. 

**Abstract (ZH)**: 基于物体聚焦演员（OFA）的通用灵巧操作学习 

---
# Nek Minit: Harnessing Pragmatic Metacognitive Prompting for Explainable Sarcasm Detection of Australian and Indian English 

**Title (ZH)**: Nek Minit: 利用手法元认知提示实现可解释的澳大利亚英语和印度英语 sarcasm 检测 

**Authors**: Ishmanbir Singh, Dipankar Srirag, Aditya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15095)  

**Abstract**: Sarcasm is a challenge to sentiment analysis because of the incongruity between stated and implied sentiment. The challenge is exacerbated when the implication may be relevant to a specific country or geographical region. Pragmatic metacognitive prompting (PMP) is a cognition-inspired technique that has been used for pragmatic reasoning. In this paper, we harness PMP for explainable sarcasm detection for Australian and Indian English, alongside a benchmark dataset for standard English. We manually add sarcasm explanations to an existing sarcasm-labeled dataset for Australian and Indian English called BESSTIE, and compare the performance for explainable sarcasm detection for them with FLUTE, a standard English dataset containing sarcasm explanations. Our approach utilising PMP when evaluated on two open-weight LLMs (GEMMA and LLAMA) achieves statistically significant performance improvement across all tasks and datasets when compared with four alternative prompting strategies. We also find that alternative techniques such as agentic prompting mitigate context-related failures by enabling external knowledge retrieval. The focused contribution of our work is utilising PMP in generating sarcasm explanations for varieties of English. 

**Abstract (ZH)**: 讽刺检测对于情感分析是一个挑战，因为显性情感和隐含情感之间存在不一致。当隐含情感可能与特定国家或地理区域相关时，这一挑战会进一步加剧。实践元认知提示（PMP）是一种受认知启发的技术，已被用于进行实践推理。本文利用PMP为澳大利亚英语和印度英语的可解释讽刺检测建立基准，并结合一个标准英语数据集FLUTE。我们手动为一个名为BESSTIE的澳大利亚英语和印度英语的讽刺标记数据集添加讽刺解释，并将其与包含讽刺解释的标准英语数据集FLUTE进行比较。当我们评估我们的方法（在GEMMA和LLAMA这两种预训练语言模型上）时，在所有任务和数据集上都实现了统计上显著的性能提升，并且我们发现，如代理提示等替代技术通过使外部知识检索成为可能，能够缓解与上下文相关的问题。本文的主要贡献在于利用PMP为英语变体生成讽刺解释。 

---
# ThinkRec: Thinking-based recommendation via LLM 

**Title (ZH)**: 基于思维的推荐：通过大规模语言模型实现 

**Authors**: Qihang Yu, Kairui Fu, Shengyu Zhang, Zheqi Lv, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15091)  

**Abstract**: Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available in anonymous Github: this https URL. 

**Abstract (ZH)**: Recent advances in大规模语言模型(LLMs)使通过自然语言生成实现更具语义意识的推荐成为可能。现有推荐中的大规模语言模型(LLM4Rec)方法主要以System 1（直觉系统）的方式运行，依赖于表面特征基于点击历史匹配相似项目，而不是通过更深层次的行为逻辑进行推理。这往往会导致肤浅且错误的推荐。鉴于此，我们提出了基于思考的框架ThinkRec，将LLM4Rec从System 1转变为System 2（理性系统）。技术上，ThinkRec引入了一种思考激活机制，通过关键词总结增强项目元数据，并注入合成的推理痕迹，引导模型形成可解释的推理链，包括分析交互历史、识别用户偏好并基于目标项目做出决策。在此基础上，我们提出了一种实例级专家融合机制以降低推理难度。通过根据用户潜在特征动态分配专家模型的权重，ThinkRec能够根据个体用户的需要调整推理路径，从而提高精准度和个性化。在真实世界数据集上的广泛实验表明，ThinkRec显著提高了推荐的准确性和可解释性。我们的实现已匿名发布在GitHub：this https URL。 

---
# DeFTX: Denoised Sparse Fine-Tuning for Zero-Shot Cross-Lingual Transfer 

**Title (ZH)**: DeFTX: 去噪稀疏微调在零-shot 跨语言迁移中的应用 

**Authors**: Sona Elza Simon, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15090)  

**Abstract**: Effective cross-lingual transfer remains a critical challenge in scaling the benefits of large language models from high-resource to low-resource languages. Towards this goal, prior studies have explored many approaches to combine task knowledge from task-specific data in a (high-resource) source language and language knowledge from unlabeled text in a (low-resource) target language. One notable approach proposed composable sparse fine-tuning (SFT) for cross-lingual transfer that learns task-specific and language-specific sparse masks to select a subset of the pretrained model's parameters that are further fine-tuned. These sparse fine-tuned vectors (SFTs) are subsequently composed with the pretrained model to facilitate zero-shot cross-lingual transfer to a task in a target language, using only task-specific data from a source language. These sparse masks for SFTs were identified using a simple magnitude-based pruning. In our work, we introduce DeFT-X, a novel composable SFT approach that denoises the weight matrices of a pretrained model before magnitude pruning using singular value decomposition, thus yielding more robust SFTs. We evaluate DeFT-X on a diverse set of extremely low-resource languages for sentiment classification (NusaX) and natural language inference (AmericasNLI) and demonstrate that it performs at par or outperforms SFT and other prominent cross-lingual transfer baselines. 

**Abstract (ZH)**: 有效的跨语言迁移仍然是将大型语言模型的优势从资源丰富语言扩展到资源贫乏语言的关键挑战。为了实现这一目标，先前的研究探索了许多方法，将特定任务的知识从资源丰富语言的源语言中的特定任务数据与资源贫乏语言的目标语言中的未标记文本的语言知识结合起来。其中一种 notable 的方法是可组合的稀疏微调（SFT），该方法学习特定任务和特定语言的稀疏掩码以选择预训练模型参数的子集，并进一步微调。这些稀疏微调向量（SFTs）随后与预训练模型组合，以便仅使用资源丰富语言中的特定任务数据，实现目标语言中的零样本跨语言迁移。这些 SFT 的稀疏掩码是使用简单的基于幅度的剪枝方法识别的。在我们的工作中，我们介绍了 DeFT-X，这是一种新颖的可组合 SFT 方法，在进行幅度剪枝之前，使用奇异值分解清理预训练模型的权重矩阵，从而生成更 robust 的 SFT。我们在情感分类（NusaX）和自然语言推断（AmericasNLI）等一系列极其资源贫乏的语言上评估了 DeFT-X，并证明它与 SFT 和其他主要的跨语言迁移基准方法性能相当或更好。 

---
# Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects 

**Title (ZH)**: 利用大型语言模型分析Python中的命令注入漏洞：基于流行开源项目的实证研究 

**Authors**: Yuxuan Wang, Jingshu Chen, Qingyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15088)  

**Abstract**: Command injection vulnerabilities are a significant security threat in dynamic languages like Python, particularly in widely used open-source projects where security issues can have extensive impact. With the proven effectiveness of Large Language Models(LLMs) in code-related tasks, such as testing, researchers have explored their potential for vulnerabilities analysis. This study evaluates the potential of large language models (LLMs), such as GPT-4, as an alternative approach for automated testing for vulnerability detection. In particular, LLMs have demonstrated advanced contextual understanding and adaptability, making them promising candidates for identifying nuanced security vulnerabilities within code. To evaluate this potential, we applied LLM-based analysis to six high-profile GitHub projects-Django, Flask, TensorFlow, Scikit-learn, PyTorch, and Langchain-each with over 50,000 stars and extensive adoption across software development and academic research. Our analysis assesses both the strengths and limitations of LLMs in detecting command injection vulnerabilities, evaluating factors such as detection accuracy, efficiency, and practical integration into development workflows. In addition, we provide a comparative analysis of different LLM tools to identify those most suitable for security applications. Our findings offer guidance for developers and security researchers on leveraging LLMs as innovative and automated approaches to enhance software security. 

**Abstract (ZH)**: 大型语言模型在检测命令注入漏洞中的潜力：基于GitHub上广泛使用的六个高知名度项目的评估 

---
# Robust Multi-Modal Forecasting: Integrating Static and Dynamic Features 

**Title (ZH)**: 鲁棒多模态预测：集成静态和动态特征 

**Authors**: Jeremy Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15083)  

**Abstract**: Time series forecasting plays a crucial role in various applications, particularly in healthcare, where accurate predictions of future health trajectories can significantly impact clinical decision-making. Ensuring transparency and explainability of the models responsible for these tasks is essential for their adoption in critical settings. Recent work has explored a top-down approach to bi-level transparency, focusing on understanding trends and properties of predicted time series using static features. In this work, we extend this framework by incorporating exogenous time series features alongside static features in a structured manner, while maintaining cohesive interpretation. Our approach leverages the insights of trajectory comprehension to introduce an encoding mechanism for exogenous time series, where they are decomposed into meaningful trends and properties, enabling the extraction of interpretable patterns. Through experiments on several synthetic datasets, we demonstrate that our approach remains predictive while preserving interpretability and robustness. This work represents a step towards developing robust, and generalized time series forecasting models. The code is available at this https URL 

**Abstract (ZH)**: 时间序列预测在各种应用中扮演着重要角色，特别是在医疗保健领域，准确预测未来的健康轨迹能显著影响临床决策。确保这些任务所依赖模型的透明性和解释性对于在关键环境中采用这些模型至关重要。近期研究探索了自上而下的双层透明性方法，重点是通过静态特征来理解预测时间序列的趋势和属性。在本文中，我们通过以结构化方式将外生时间序列特征与静态特征结合，扩展了这一框架，同时保持一致的解释性。我们的方法利用轨迹理解的洞察，引入了一种外生时间序列的编码机制，将它们分解为有意义的趋势和属性，从而提取可解释的模式。通过在多个合成数据集上的实验，我们展示了我们的方法在保持预测能力的同时，仍然具有可解释性和稳健性。这项工作代表了向开发稳健且通用的时间序列预测模型迈进的一步。代码可在以下链接获得：this https URL 

---
# SUS backprop: linear backpropagation algorithm for long inputs in transformers 

**Title (ZH)**: SUS反向传播：长输入在变换器中的线性反向传播算法 

**Authors**: Sergey Pankov, Georges Harik  

**Link**: [PDF](https://arxiv.org/pdf/2505.15080)  

**Abstract**: It is straightforward to design an unbiased gradient estimator that stochastically cuts the backpropagation flow through any part of a computational graph. By cutting the parts that have little effect on the computation, one can potentially save a significant amount of back-propagation computation in exchange for a minimal increase in the stochastic gradient variance, in some situations. Such a situation occurs in the attention mechanism of the transformer architecture. For long sequences, attention becomes the limiting factor, as its compute requirements increase quadratically with sequence length $n$. At the same time, most attention weights become very small, as most attention heads tend to connect a given token with only a small fraction of other tokens in the sequence. These weights become promising targets for cutting backpropagation. We propose a simple probabilistic rule controlled by a single parameter $c$ that cuts backpropagation through most attention weights, leaving at most $c$ interactions per token per attention head. This brings a factor of $c/n$ reduction in the compute required for the attention backpropagation, turning it from quadratic $O(n^2)$ to linear complexity $O(nc)$. We have empirically verified that, for a typical transformer model, cutting $99\%$ of the attention gradient flow (i.e. choosing $c \sim 20-30$) results in relative gradient variance increase of only about $1\%$ for $n \sim 2000$, and it decreases with $n$. This approach is amenable to efficient sparse matrix implementation, thus being promising for making the cost of a backward pass negligible relative to the cost of a forward pass when training a transformer model on long sequences. 

**Abstract (ZH)**: 一种用于变压器架构注意力机制的无偏梯度估计器设计 

---
# Data Augmentation and Resolution Enhancement using GANs and Diffusion Models for Tree Segmentation 

**Title (ZH)**: 使用生成对抗网络和扩散模型进行数据增强与分辨率提升的树段分割方法 

**Authors**: Alessandro dos Santos Ferreira, Ana Paula Marques Ramos, José Marcato Junior, Wesley Nunes Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2505.15077)  

**Abstract**: Urban forests play a key role in enhancing environmental quality and supporting biodiversity in cities. Mapping and monitoring these green spaces are crucial for urban planning and conservation, yet accurately detecting trees is challenging due to complex landscapes and the variability in image resolution caused by different satellite sensors or UAV flight altitudes. While deep learning architectures have shown promise in addressing these challenges, their effectiveness remains strongly dependent on the availability of large and manually labeled datasets, which are often expensive and difficult to obtain in sufficient quantity. In this work, we propose a novel pipeline that integrates domain adaptation with GANs and Diffusion models to enhance the quality of low-resolution aerial images. Our proposed pipeline enhances low-resolution imagery while preserving semantic content, enabling effective tree segmentation without requiring large volumes of manually annotated data. Leveraging models such as pix2pix, Real-ESRGAN, Latent Diffusion, and Stable Diffusion, we generate realistic and structurally consistent synthetic samples that expand the training dataset and unify scale across domains. This approach not only improves the robustness of segmentation models across different acquisition conditions but also provides a scalable and replicable solution for remote sensing scenarios with scarce annotation resources. Experimental results demonstrated an improvement of over 50% in IoU for low-resolution images, highlighting the effectiveness of our method compared to traditional pipelines. 

**Abstract (ZH)**: 城市森林在提升城市环境质量和支持生物多样性方面发挥着关键作用。精确检测这些绿色空间对于城市规划和保护至关重要，但由于复杂景观和不同卫星传感器或无人机飞行高度引起的图像分辨率变化，准确检测树木仍然具有挑战性。尽管深度学习架构在应对这些挑战方面显示出潜力，但其有效性仍然强烈依赖于大型且手动标注的数据集，这类数据集往往成本高昂且难以获得。在本文中，我们提出了一种新的流程，将领域适应与GAN和扩散模型相结合，以提高低分辨率航拍图像的质量。我们提出的流程在保留语义内容的同时增强了低分辨率图像，从而在不需大量手动标注数据的情况下实现有效的树木分割。通过利用pix2pix、Real-ESRGAN、隐空间扩散和 Stable Diffusion等模型，我们生成了现实且结构一致的合成样本，扩展了训练数据集并统一了跨领域的尺度。这种方法不仅提高了分割模型在不同获取条件下的稳健性，还为标注资源稀缺的遥感场景提供了可扩展且可复制的解决方案。实验结果表明，低分辨率图像的IoU提高了50%以上，突显了我们方法相比于传统流程的有效性。 

---
# Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs 

**Title (ZH)**: 跨国界旅行：多模态LLM跨语言一致性Benchmark研究 

**Authors**: Hao Wang, Pinzhi Huang, Jihan Yang, Saining Xie, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2505.15075)  

**Abstract**: The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models. 

**Abstract (ZH)**: 多模态大型语言模型的快速进化显著增强了其实际应用能力。然而，在跨语言应用，尤其是在整合文化知识方面，实现一致的性能仍旧是一项重大挑战。为了更好地评估这一问题，我们引入了两个新的基准：KnowRecall和VisRecall，用于评估多模态大型语言模型的跨语言一致性。KnowRecall是一个视觉问答基准，旨在测量15种语言中的事实知识一致性，重点关注关于全球地标的文化和历史问题。VisRecall通过要求模型在没有图片访问的情况下描述9种语言中的地标外貌，评估视觉记忆一致性。实验结果表明，最先进的多模态大型语言模型，包括专有模型，仍然难以实现跨语言一致性。这强调了需要更加 robust的方法来生产真正多语言且文化意识强的模型。 

---
# DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data 

**Title (ZH)**: DISCO 调和天平：不平衡数据上的自适应领域和难度感知强化学习 

**Authors**: Yuhang Zhou, Jing Zhu, Shengyi Qian, Zhuokai Zhao, Xiyao Wang, Xiaoyu Liu, Ming Li, Paiheng Xu, Wei Ai, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15074)  

**Abstract**: Large Language Models (LLMs) are increasingly aligned with human preferences through Reinforcement Learning from Human Feedback (RLHF). Among RLHF methods, Group Relative Policy Optimization (GRPO) has gained attention for its simplicity and strong performance, notably eliminating the need for a learned value function. However, GRPO implicitly assumes a balanced domain distribution and uniform semantic alignment across groups - assumptions that rarely hold in real-world datasets. When applied to multi-domain, imbalanced data, GRPO disproportionately optimizes for dominant domains, neglecting underrepresented ones and resulting in poor generalization and fairness. We propose Domain-Informed Self-Consistency Policy Optimization (DISCO), a principled extension to GRPO that addresses inter-group imbalance with two key innovations. Domain-aware reward scaling counteracts frequency bias by reweighting optimization based on domain prevalence. Difficulty-aware reward scaling leverages prompt-level self-consistency to identify and prioritize uncertain prompts that offer greater learning value. Together, these strategies promote more equitable and effective policy learning across domains. Extensive experiments across multiple LLMs and skewed training distributions show that DISCO improves generalization, outperforms existing GRPO variants by 5% on Qwen3 models, and sets new state-of-the-art results on multi-domain alignment benchmarks. 

**Abstract (ZH)**: 基于域信息的自我一致性策略优化（DISCO）：一种解决不平衡域分布的原理性扩展 

---
# The Pursuit of Empathy: Evaluating Small Language Models for PTSD Dialogue Support 

**Title (ZH)**: 共情的追求：评估小型语言模型在 PTSD 对话支持中的效果 

**Authors**: Suhas BN, Yash Mahajan, Dominik Mattioli, Andrew M. Sherrill, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2505.15065)  

**Abstract**: Can small language models with 0.5B to 5B parameters meaningfully engage in trauma-informed, empathetic dialogue for individuals with PTSD? We address this question by introducing TIDE, a dataset of 10,000 two-turn dialogues spanning 500 diverse PTSD client personas and grounded in a three-factor empathy model: emotion recognition, distress normalization, and supportive reflection. All scenarios and reference responses were reviewed for realism and trauma sensitivity by a clinical psychologist specializing in PTSD. We evaluate eight small language models before and after fine-tuning, comparing their outputs to a frontier model (Claude Sonnet 3.5). Our IRB-approved human evaluation and automatic metrics show that fine-tuning generally improves perceived empathy, but gains are highly scenario- and user-dependent, with smaller models facing an empathy ceiling. Demographic analysis shows older adults value distress validation and graduate-educated users prefer nuanced replies, while gender effects are minimal. We highlight the limitations of automatic metrics and the need for context- and user-aware system design. Our findings, along with the planned release of TIDE, provide a foundation for building safe, resource-efficient, and ethically sound empathetic AI to supplement, not replace, clinical mental health care. 

**Abstract (ZH)**: 0.5B至5B参数的小语言模型能否有意义地与 PTSD 患者进行创伤知情、共情对话？我们通过引入 TIDE 数据集来解答这个问题，该数据集包含 10,000 个双轮对话，覆盖了 500 种多样化的 PTSD 患者人设，并基于一种三因素共情模型：情绪识别、痛苦正常化和支持性反思。所有场景和参考回复均经专门研究 PTSD 的临床心理学家审核，确保其真实性和对创伤的敏感性。我们对八种小语言模型进行微调前后的评估，并将其输出与前沿模型（Claude Sonnet 3.5）进行对比。经 IRB 批准的人类评估和自动指标表明，微调通常能提高感知共情，但进步程度高度依赖于具体场景和用户，而小型模型则面临共情上限。人口统计分析显示，老年人更看重痛苦的验证，受过高等教育的用户更喜欢细致的回答，而性别影响则较小。我们强调自动指标的局限性，并强调需要情境和用户意识系统的设计。我们的发现，加上 TIDE 数据集的计划发布，为建立安全、资源高效且伦理合理的共情 AI 提供了基础，以补充而非替代临床心理健康护理。 

---
# Self-GIVE: Associative Thinking from Limited Structured Knowledge for Enhanced Large Language Model Reasoning 

**Title (ZH)**: Self-GIVE：从有限结构化知识中进行关联思考以增强大规模语言模型推理 

**Authors**: Jiashu He, Jinxuan Fan, Bowen Jiang, Ignacio Houine, Dan Roth, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2505.15062)  

**Abstract**: When addressing complex questions that require new information, people often associate the question with existing knowledge to derive a sensible answer. For instance, when evaluating whether melatonin aids insomnia, one might associate "hormones helping mental disorders" with "melatonin being a hormone and insomnia a mental disorder" to complete the reasoning. Large Language Models (LLMs) also require such associative thinking, particularly in resolving scientific inquiries when retrieved knowledge is insufficient and does not directly answer the question. Graph Inspired Veracity Extrapolation (GIVE) addresses this by using a knowledge graph (KG) to extrapolate structured knowledge. However, it involves the construction and pruning of many hypothetical triplets, which limits efficiency and generalizability. We propose Self-GIVE, a retrieve-RL framework that enhances LLMs with automatic associative thinking through reinforcement learning. Self-GIVE extracts structured information and entity sets to assist the model in linking to the queried concepts. We address GIVE's key limitations: (1) extensive LLM calls and token overhead for knowledge extrapolation, (2) difficulty in deploying on smaller LLMs (3B or 7B) due to complex instructions, and (3) inaccurate knowledge from LLM pruning. Specifically, after fine-tuning using self-GIVE with a 135 node UMLS KG, it improves the performance of the Qwen2.5 3B and 7B models by up to $\textbf{28.5%$\rightarrow$71.4%}$ and $\textbf{78.6$\rightarrow$90.5%}$ in samples $\textbf{unseen}$ in challenging biomedical QA tasks. In particular, Self-GIVE allows the 7B model to match or outperform GPT3.5 turbo with GIVE, while cutting token usage by over 90\%. Self-GIVE enhances the scalable integration of structured retrieval and reasoning with associative thinking. 

**Abstract (ZH)**: 基于图启发的真伪外推自构模型 

---
# AsynFusion: Towards Asynchronous Latent Consistency Models for Decoupled Whole-Body Audio-Driven Avatars 

**Title (ZH)**: AsynFusion: 向异步潜在一致性模型的方向发展，用于解耦的全身音driven动画角色 

**Authors**: Tianbao Zhang, Jian Zhao, Yuer Li, Zheng Zhu, Ping Hu, Zhaoxin Fan, Wenjun Wu, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15058)  

**Abstract**: Whole-body audio-driven avatar pose and expression generation is a critical task for creating lifelike digital humans and enhancing the capabilities of interactive virtual agents, with wide-ranging applications in virtual reality, digital entertainment, and remote communication. Existing approaches often generate audio-driven facial expressions and gestures independently, which introduces a significant limitation: the lack of seamless coordination between facial and gestural elements, resulting in less natural and cohesive animations. To address this limitation, we propose AsynFusion, a novel framework that leverages diffusion transformers to achieve harmonious expression and gesture synthesis. The proposed method is built upon a dual-branch DiT architecture, which enables the parallel generation of facial expressions and gestures. Within the model, we introduce a Cooperative Synchronization Module to facilitate bidirectional feature interaction between the two modalities, and an Asynchronous LCM Sampling strategy to reduce computational overhead while maintaining high-quality outputs. Extensive experiments demonstrate that AsynFusion achieves state-of-the-art performance in generating real-time, synchronized whole-body animations, consistently outperforming existing methods in both quantitative and qualitative evaluations. 

**Abstract (ZH)**: 全身音频驱动的角色姿态与表情生成是创建真实感数字人类和增强交互式虚拟代理能力的关键任务，广泛应用于虚拟现实、数字娱乐和远程通信。现有的方法通常独立生成音频驱动的面部表情和手势，这引入了一个重要限制：面部和手势元素之间缺乏无缝协调，导致生成的动画不够自然和一致。为了解决这一限制，我们提出了一种新颖的框架AsynFusion，利用扩散变压器实现和谐的表情和手势合成。该方法基于双支路DiT架构，能够并行生成面部表情和手势。在模型中，我们引入了协同同步模块以促进两种模态之间的双向特征交互，并采用了异步LCM采样策略以在保持高质量输出的同时减少计算开销。广泛的实验表明，AsynFusion在生成实时同步的全身动画方面达到了最先进的性能，定量和定性评估均优于现有方法。 

---
# MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation 

**Title (ZH)**: MolLangBench：一种综合性的语言提示分子结构识别、编辑与生成基准 

**Authors**: Feiyang Cai, Jiahui Bai, Tao Tang, Joshua Luo, Tianyu Zhu, Ling Liu, Feng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.15054)  

**Abstract**: Precise recognition, editing, and generation of molecules are essential prerequisites for both chemists and AI systems tackling various chemical tasks. We present MolLangBench, a comprehensive benchmark designed to evaluate fundamental molecule-language interface tasks: language-prompted molecular structure recognition, editing, and generation. To ensure high-quality, unambiguous, and deterministic outputs, we construct the recognition tasks using automated cheminformatics tools, and curate editing and generation tasks through rigorous expert annotation and validation. MolLangBench supports the evaluation of models that interface language with different molecular representations, including linear strings, molecular images, and molecular graphs. Evaluations of state-of-the-art models reveal significant limitations: the strongest model (o3) achieves $79.2\%$ and $78.5\%$ accuracy on recognition and editing tasks, which are intuitively simple for humans, and performs even worse on the generation task, reaching only $29.0\%$ accuracy. These results highlight the shortcomings of current AI systems in handling even preliminary molecular recognition and manipulation tasks. We hope MolLangBench will catalyze further research toward more effective and reliable AI systems for chemical applications. 

**Abstract (ZH)**: 分子精确识别、编辑和生成是化学家和处理各种化学任务的AI系统的基本前提。我们提出MolLangBench，一个全面的基准测试，用于评估分子-语言接口基本任务：语言提示下的分子结构识别、编辑和生成。为了确保高质量、无歧义和确定性的输出，我们使用自动化化学信息学工具构建识别任务，并通过严格的专家注释和验证，整理编辑和生成任务。MolLangBench 支持语言与不同分子表示形式的接口模型的评估，包括线性字符串、分子图像和分子图。对最先进的模型的评估揭示了显著的限制：最强的模型（o3）在识别和编辑任务中分别达到79.2%和78.5%的准确率，这些任务对人类来说直观上很简单，而在生成任务中表现更差，仅达到29.0%的准确率。这些结果突显了当前AI系统在处理初步的分子识别和操作任务方面的不足之处。我们希望MolLangBench能够推动更多有效的和可靠的AI系统在化学应用方面的进一步研究。 

---
# Towards a Working Definition of Designing Generative User Interfaces 

**Title (ZH)**: 面向生成型用户界面设计的工作性定义 

**Authors**: Kyungho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15049)  

**Abstract**: Generative UI is transforming interface design by facilitating AI-driven collaborative workflows between designers and computational systems. This study establishes a working definition of Generative UI through a multi-method qualitative approach, integrating insights from a systematic literature review of 127 publications, expert interviews with 18 participants, and analyses of 12 case studies. Our findings identify five core themes that position Generative UI as an iterative and co-creative process. We highlight emerging design models, including hybrid creation, curation-based workflows, and AI-assisted refinement strategies. Additionally, we examine ethical challenges, evaluation criteria, and interaction models that shape the field. By proposing a conceptual foundation, this study advances both theoretical discourse and practical implementation, guiding future HCI research toward responsible and effective generative UI design practices. 

**Abstract (ZH)**: 生成型UI正在通过促进设计师与计算系统之间的AI驱动协作工作流来变革界面设计。本研究通过多方法质性研究方法建立生成型UI的工作定义，整合了对127篇出版物的系统文献综述、与18名专家的访谈以及12个案例研究的分析。研究发现，生成型UI表现为一个迭代性和共创性的过程，明确了五个核心主题。研究强调了新兴的设计模式，包括混合创作、策展为基础的工作流和AI辅助的细化策略。此外，研究还考察了塑造该领域的伦理挑战、评估标准和交互模型。通过提出一个概念性基础，本研究不仅推进了理论讨论，还指导了实际实施，并指引未来的HCI研究朝向负责任和有效的生成型UI设计实践。 

---
# PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration 

**Title (ZH)**: PiFlow: 原理导向的多Agent协作科学研究 

**Authors**: Yingming Pu, Tao Lin, Hongyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15047)  

**Abstract**: Large Language Model (LLM)-based multi-agent systems (MAS) demonstrate remarkable potential for scientific discovery. Existing approaches, however, often automate scientific discovery using predefined workflows that lack rationality constraints. This often leads to aimless hypothesizing and a failure to consistently link hypotheses with evidence, thereby hindering systematic uncertainty reduction. Overcoming these limitations fundamentally requires systematic uncertainty reduction. We introduce \texttt{PiFlow}, an information-theoretical framework, treating automated scientific discovery as a structured uncertainty reduction problem guided by principles (e.g., scientific laws). In evaluations across three distinct scientific domains -- discovering nanomaterial structures, bio-molecules, and superconductor candidates with targeted properties -- our method significantly improves discovery efficiency, reflected by a 73.55\% increase in the Area Under the Curve (AUC) of property values versus exploration steps, and enhances solution quality by 94.06\% compared to a vanilla agent system. Overall, \texttt{PiFlow} serves as a Plug-and-Play method, establishing a novel paradigm shift in highly efficient automated scientific discovery, paving the way for more robust and accelerated AI-driven research. Code is publicly available at our \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的多agents系统（MAS）在科学研究中展现出巨大的潜力。现有方法通常使用预定义的工作流来进行科学研究自动化，缺乏理性的约束，导致盲目假设和无法一致地将假设与证据关联起来，从而妨碍系统的不确定性减少。克服这些限制从根本上需要系统性的不确定性减少。我们提出了\texttt{PiFlow}，一种信息论框架，将自动科学研究视为由原理（例如，科学定律）指导的结构化不确定性减少问题。在涉及三种不同科学领域的评估中——纳米材料结构发现、生物分子发现以及具有目标性质的超导体候选物发现——我们的方法显著提高了发现效率，表现为面积下曲线（AUC）的属性值与探索步骤之间的AUC提高了73.55%，并且相较于传统的代理系统，解决方案质量提高了94.06%。总体而言，\texttt{PiFlow}作为一种即插即用方法，为高效自动科学研究建立了新的范式转变，为更稳健和快速的人工智能驱动研究铺平了道路。代码在我们的GitHub上公开可用。 

---
# ChartCards: A Chart-Metadata Generation Framework for Multi-Task Chart Understanding 

**Title (ZH)**: ChartCards：多任务图表理解的图表-元数据生成框架 

**Authors**: Yifan Wu, Lutao Yan, Leixian Shen, Yinan Mei, Jiannan Wang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.15046)  

**Abstract**: The emergence of Multi-modal Large Language Models (MLLMs) presents new opportunities for chart understanding. However, due to the fine-grained nature of these tasks, applying MLLMs typically requires large, high-quality datasets for task-specific fine-tuning, leading to high data collection and training costs. To address this, we propose ChartCards, a unified chart-metadata generation framework for multi-task chart understanding. ChartCards systematically synthesizes various chart information, including data tables, visualization code, visual elements, and multi-dimensional semantic captions. By structuring this information into organized metadata, ChartCards enables a single chart to support multiple downstream tasks, such as text-to-chart retrieval, chart summarization, chart-to-table conversion, chart description, and chart question answering. Using ChartCards, we further construct MetaChart, a large-scale high-quality dataset containing 10,862 data tables, 85K charts, and 170 K high-quality chart captions. We validate the dataset through qualitative crowdsourcing evaluations and quantitative fine-tuning experiments across various chart understanding tasks. Fine-tuning six different models on MetaChart resulted in an average performance improvement of 5% across all tasks. The most notable improvements are seen in text-to-chart retrieval and chart-to-table tasks, with Long-CLIP and Llama 3.2-11B achieving improvements of 17% and 28%, respectively. 

**Abstract (ZH)**: 多模态大型语言模型的出现为图表理解带来了新机遇。然而，由于这些任务的精细特性，应用多模态大型语言模型通常需要大规模的高质量数据集进行任务特定的微调，从而导致较高的数据收集和训练成本。为此，我们提出ChartCards，一种统一的多任务图表元数据生成框架。ChartCards系统地综合了各种图表信息，包括数据表、可视化代码、视觉元素和多维度语义注释。通过将这些信息结构化为组织化的元数据，ChartCards使单个图表能够支持多个下游任务，如文本到图表检索、图表总结、图表到表格转换、图表描述和图表问答。使用ChartCards，我们进一步构建了MetaChart，一个包含10,862个数据表、85,000个图表和170,000个高质量图表注释的大规模高质量数据集。我们通过定性众包评估和定量跨多种图表理解任务的微调实验验证了该数据集。在MetaChart上微调六种不同模型后，所有任务的平均性能提升了5%。最显著的改进出现在文本到图表检索和图表到表格任务中，Long-CLIP和Llama 3.2-11B分别取得了17%和28%的提升。 

---
# Learning-based Airflow Inertial Odometry for MAVs using Thermal Anemometers in a GPS and vision denied environment 

**Title (ZH)**: 基于学习的空中客流惯性Odometry MAVs利用热风速计在GPS和视觉受限环境中的研究 

**Authors**: Ze Wang, Jingang Qu, Zhenyu Gao, Pascal Morin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15044)  

**Abstract**: This work demonstrates an airflow inertial based odometry system with multi-sensor data fusion, including thermal anemometer, IMU, ESC, and barometer. This goal is challenging because low-cost IMUs and barometers have significant bias, and anemometer measurements are very susceptible to interference from spinning propellers and ground effects. We employ a GRU-based deep neural network to estimate relative air speed from noisy and disturbed anemometer measurements, and an observer with bias model to fuse the sensor data and thus estimate the state of aerial vehicle. A complete flight data, including takeoff and landing on the ground, shows that the approach is able to decouple the downwash induced wind speed caused by propellers and the ground effect, and accurately estimate the flight speed in a wind-free indoor environment. IMU, and barometer bias are effectively estimated, which significantly reduces the position integration drift, which is only 5.7m for 203s manual random flight. The open source is available on this https URL. 

**Abstract (ZH)**: 基于气流惯性多传感器数据融合的自主导航系统研究 

---
# LogiCase: Effective Test Case Generation from Logical Description in Competitive Programming 

**Title (ZH)**: LogiCase：来自逻辑描述的有效测试用例生成在竞赛编程中的应用 

**Authors**: Sicheol Sung, Aditi, Dogyu kim, Yo-Sub Han, Sang-Ki Ko  

**Link**: [PDF](https://arxiv.org/pdf/2505.15039)  

**Abstract**: Automated Test Case Generation (ATCG) is crucial for evaluating software reliability, particularly in competitive programming where robust algorithm assessments depend on diverse and accurate test cases. However, existing ATCG methods often fail to meet complex specifications or generate effective corner cases, limiting their utility. In this work, we introduce Context-Free Grammars with Counters (CCFGs), a formalism that captures both syntactic and semantic structures in input specifications. Using a fine-tuned CodeT5 model, we translate natural language input specifications into CCFGs, enabling the systematic generation of high-quality test cases. Experiments on the CodeContests dataset demonstrate that CCFG-based test cases outperform baseline methods in identifying incorrect algorithms, achieving significant gains in validity and effectiveness. Our approach provides a scalable and reliable grammar-driven framework for enhancing automated competitive programming evaluations. 

**Abstract (ZH)**: 基于上下文自由文法计数器的自动测试用例生成 

---
# Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering 

**Title (ZH)**: 使用稀疏自编码器去噪概念向量以改进语言模型导向 

**Authors**: Haiyan Zhao, Xuansheng Wu, Fan Yang, Bo Shen, Ninghao Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.15038)  

**Abstract**: Linear Concept Vectors have proven effective for steering large language models (LLMs). While existing approaches like linear probing and difference-in-means derive these vectors from LLM hidden representations, diverse data introduces noises (i.e., irrelevant features) that challenge steering robustness. To address this, we propose Sparse Autoencoder-Denoised Concept Vectors (SDCV), which uses Sparse Autoencoders to filter out noisy features from hidden representations. When applied to linear probing and difference-in-means, our method improves their steering success rates. We validate our noise hypothesis through counterfactual experiments and feature visualizations. 

**Abstract (ZH)**: Sparse Autoencoder-Denoised Concept Vectors for Robust Steering of Large Language Models 

---
# Fault-Tolerant Multi-Robot Coordination with Limited Sensing within Confined Environments 

**Title (ZH)**: 受限环境内有限感知的容错多机器人协调 

**Authors**: Kehinde O. Aina, Hosain Bagheri, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15036)  

**Abstract**: As robots are increasingly deployed to collaborate on tasks within shared workspaces and resources, the failure of an individual robot can critically affect the group's performance. This issue is particularly challenging when robots lack global information or direct communication, relying instead on social interaction for coordination and to complete their tasks. In this study, we propose a novel fault-tolerance technique leveraging physical contact interactions in multi-robot systems, specifically under conditions of limited sensing and spatial confinement. We introduce the "Active Contact Response" (ACR) method, where each robot modulates its behavior based on the likelihood of encountering an inoperative (faulty) robot. Active robots are capable of collectively repositioning stationary and faulty peers to reduce obstructions and maintain optimal group functionality. We implement our algorithm in a team of autonomous robots, equipped with contact-sensing and collision-tolerance capabilities, tasked with collectively excavating cohesive model pellets. Experimental results indicate that the ACR method significantly improves the system's recovery time from robot failures, enabling continued collective excavation with minimal performance degradation. Thus, this work demonstrates the potential of leveraging local, social, and physical interactions to enhance fault tolerance and coordination in multi-robot systems operating in constrained and extreme environments. 

**Abstract (ZH)**: 随着机器人越来越多地被部署到共享工作空间和资源中协作完成任务，单个机器人的故障会严重影响团队的表现。特别是在机器人缺乏全局信息或直接通信能力，依赖社会互动进行协调和完成任务的情况下，这一问题尤为严峻。本研究提出了一种新的容错技术，利用多机器人系统中物理接触交互，在受限感知识别和空间约束条件下提升系统的容错能力。我们引入了“主动接触响应”（ACR）方法，每台机器人根据遇到故障机器人可能性的大小调整其行为。活动机器人能够集体重新定位静止和故障的同伴，以减少障碍并保持团队的最佳功能。我们将该算法应用于装备有接触感测和碰撞容忍能力的自主机器人团队，它们的任务是集体挖掘统一模型颗粒。实验结果表明，ACR方法显著提高了系统从机器人故障中恢复的时间，使得在性能下降最小的情况下继续进行集体挖掘。因此，本研究展示了在受限和极端环境中操作的多机器人系统中利用局部、社会和物理交互提升容错能力和协调的潜力。 

---
# RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning 

**Title (ZH)**: RL �人才舞蹈：同时增强生成器和验证器的语言推理 

**Authors**: Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15034)  

**Abstract**: Reinforcement learning (RL) has recently emerged as a compelling approach for enhancing the reasoning capabilities of large language models (LLMs), where an LLM generator serves as a policy guided by a verifier (reward model). However, current RL post-training methods for LLMs typically use verifiers that are fixed (rule-based or frozen pretrained) or trained discriminatively via supervised fine-tuning (SFT). Such designs are susceptible to reward hacking and generalize poorly beyond their training distributions. To overcome these limitations, we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level LLM verifier, which is trained via RL and co-evolves with the generator. Importantly, the verifier is trained solely based on outcome-level verification correctness rewards without requiring explicit process-level annotations. This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator. Extensive experiments demonstrate that both components of Tango achieve state-of-the-art results among 7B/8B-scale models: the generator attains best-in-class performance across five competition-level math benchmarks and four challenging out-of-domain reasoning tasks, while the verifier leads on the ProcessBench dataset. Remarkably, both components exhibit particularly substantial improvements on the most difficult mathematical reasoning problems. Code is at: this https URL. 

**Abstract (ZH)**: 基于强化学习的新型框架Tango：同步训练LLM生成器和生成式过程级验证器以克服奖励劫持和泛化不足 

---
# Toward Task Capable Active Matter: Learning to Avoid Clogging in Confined Collectives via Collisions 

**Title (ZH)**: 面向任务的能力性的活性物质：通过碰撞学习在受限群体中避免堵塞 

**Authors**: Kehinde O. Aina, Ram Avinery, Hui-Shun Kuan, Meredith D. Betterton, Michael A. D. Goodisman, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15033)  

**Abstract**: Social organisms which construct nests consisting of tunnels and chambers necessarily navigate confined and crowded conditions. Unlike low-density collectives like bird flocks and insect swarms, in which hydrodynamic and statistical phenomena dominate, the physics of glasses and supercooled fluids is important to understand clogging behaviors in high-density collectives. Our previous work revealed that fire ants flowing in confined tunnels utilize diverse behaviors like unequal workload distributions, spontaneous direction reversals, and limited interaction times to mitigate clogging and jamming and thus maintain functional flow; implementation of similar rules in a small robophysical swarm led to high performance through spontaneous dissolution of clogs and clusters. However, how the insects learn such behaviors, and how we can develop "task capable" active matter in such regimes, remains a challenge in part because interaction dynamics are dominated by local, time-consuming collisions and no single agent can guide the entire collective. Here, we hypothesized that effective flow and clog mitigation could emerge purely through local learning. We tasked small groups of robots with pellet excavation in a narrow tunnel, allowing them to modify reversal probabilities over time. Initially, robots had equal probabilities and clogs were common. Reversals improved flow. When reversal probabilities adapted via collisions and noisy tunnel length estimates, workload inequality and performance improved. Our robophysical study of an excavating swarm shows that, despite the seeming complexity and difficulty of the task, simple learning rules can mitigate or leverage unavoidable features in task-capable dense active matter, leading to hypotheses for dense biological and robotic swarms. 

**Abstract (ZH)**: 社会性生物在建造由隧道和隔间组成的巢穴时，必然面临狭窄和拥挤的环境。与鸟类 flock 和昆虫 swarm 等低密度群体不同，在这些群体中流体动力学和统计现象占主导地位，理解高密度群体中的堵塞行为需要研究玻璃态和过冷流体的物理特性。我们先前的工作揭示了火蚁在狭窄隧道中流动时利用不均等的工作负载分配、自发的方向反转和有限的相互作用时间来缓解堵塞和堵塞，从而维持功能性流动；在小型 robophysical 群体中实施类似规则导致自发解堵和聚簇的高效率。然而，昆虫如何学习这些行为，以及如何开发能够在这种条件下执行特定任务的活性物质，仍然是一个挑战，部分原因是互动动态主要是局部的耗时碰撞，单个代理不能引导整个群体。在这里，我们假设有效的流动和堵塞缓解可以通过局部学习完全产生。我们让一小群机器人在狭窄的隧道中进行颗粒挖掘任务，并允许他们随着时间修改反转概率。最初，机器人具有平等的概率且堵塞常见。反转有助于提高流动。当通过碰撞和噪声隧道长度估计调整反转概率时，工作负载的不平等性和性能得以提高。我们的 robophysical 研究表明，尽管任务看似复杂且难以执行，简单的学习规则仍然可以缓解或利用任务能力密集型活性物质中不可避免的特点，从而为密集生物和机器人群提供了假设。 

---
# Are the confidence scores of reviewers consistent with the review content? Evidence from top conference proceedings in AI 

**Title (ZH)**: 顶级AI会议论文集中的审稿人置信度评分是否与评审内容一致：证据探析 

**Authors**: Wenqing Wu, Haixu Xi, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15031)  

**Abstract**: Peer review is vital in academia for evaluating research quality. Top AI conferences use reviewer confidence scores to ensure review reliability, but existing studies lack fine-grained analysis of text-score consistency, potentially missing key details. This work assesses consistency at word, sentence, and aspect levels using deep learning and NLP conference review data. We employ deep learning to detect hedge sentences and aspects, then analyze report length, hedge word/sentence frequency, aspect mentions, and sentiment to evaluate text-score alignment. Correlation, significance, and regression tests examine confidence scores' impact on paper outcomes. Results show high text-score consistency across all levels, with regression revealing higher confidence scores correlate with paper rejection, validating expert assessments and peer review fairness. 

**Abstract (ZH)**: 基于深度学习和自然语言处理的会议评审文本-评分一致性分析：从词汇、句子到方面 

---
# Towards a Science of Causal Interpretability in Deep Learning for Software Engineering 

**Title (ZH)**: 面向软件工程中深度学习因果可解释性的科学探索 

**Authors**: David N. Palacio  

**Link**: [PDF](https://arxiv.org/pdf/2505.15023)  

**Abstract**: This dissertation addresses achieving causal interpretability in Deep Learning for Software Engineering (DL4SE). While Neural Code Models (NCMs) show strong performance in automating software tasks, their lack of transparency in causal relationships between inputs and outputs limits full understanding of their capabilities. To build trust in NCMs, researchers and practitioners must explain code predictions. Associational interpretability, which identifies correlations, is often insufficient for tasks requiring intervention and change analysis. To address this, the dissertation introduces DoCode, a novel post hoc interpretability method for NCMs. DoCode uses causal inference to provide programming language-oriented explanations of model predictions. It follows a four-step pipeline: modeling causal problems using Structural Causal Models (SCMs), identifying the causal estimand, estimating effects with metrics like Average Treatment Effect (ATE), and refuting effect estimates. Its framework is extensible, with an example that reduces spurious correlations by grounding explanations in programming language properties. A case study on deep code generation across interpretability scenarios and various deep learning architectures demonstrates DoCode's benefits. Results show NCMs' sensitivity to code syntax changes and their ability to learn certain programming concepts while minimizing confounding bias. The dissertation also examines associational interpretability as a foundation, analyzing software information's causal nature using tools like COMET and TraceXplainer for traceability. It highlights the need to identify code confounders and offers practical guidelines for applying causal interpretability to NCMs, contributing to more trustworthy AI in software engineering. 

**Abstract (ZH)**: 本论文探讨在软件工程中实现深度学习因果可解释性的方法。尽管神经代码模型在自动化软件任务方面表现出色，但它们在输入和输出之间因果关系上的不透明性限制了对其能力的完全理解。为了建立对神经代码模型的信任，研究者和实践者必须能够解释代码预测。关联可解释性可以识别相关性，但对于需要干预和变化分析的任务来说往往是不够的。为了解决这一问题，本论文介绍了DoCode，这是一种用于神经代码模型的新型事后可解释性方法。DoCode利用因果推理为模型预测提供面向编程语言的解释。该方法包含四个步骤：使用结构因果模型建模因果问题、识别因果估计量、使用平均处理效应等度量估计效应，并反驳效应估计。其框架具有扩展性，通过将解释与编程语言属性联系起来，可以减少虚假相关性。跨可解释性场景和各种深度学习架构的深度代码生成案例研究表明，DoCode的优势。结果表明，神经代码模型对代码语法变化的敏感性以及它们在最小化混杂偏倚的同时学习某些编程概念的能力。此外，本论文还研究了关联可解释性作为基础，并使用COMET和TraceXplainer等工具分析软件信息的因果性质，强调识别代码混杂因子的需求，并提供将因果可解释性应用于神经代码模型的实用指南，从而促进软件工程中更可信的人工智能。 

---
# One-Layer Transformers are Provably Optimal for In-context Reasoning and Distributional Association Learning in Next-Token Prediction Tasks 

**Title (ZH)**: 一层变压器在上下文推理和分布关联学习中的预测下一个词任务中可证明最优 

**Authors**: Quan Nguyen, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15009)  

**Abstract**: We study the approximation capabilities and on-convergence behaviors of one-layer transformers on the noiseless and noisy in-context reasoning of next-token prediction. Existing theoretical results focus on understanding the in-context reasoning behaviors for either the first gradient step or when the number of samples is infinite. Furthermore, no convergence rates nor generalization abilities were known. Our work addresses these gaps by showing that there exists a class of one-layer transformers that are provably Bayes-optimal with both linear and ReLU attention. When being trained with gradient descent, we show via a finite-sample analysis that the expected loss of these transformers converges at linear rate to the Bayes risk. Moreover, we prove that the trained models generalize to unseen samples as well as exhibit learning behaviors that were empirically observed in previous works. Our theoretical findings are further supported by extensive empirical validations. 

**Abstract (ZH)**: 我们研究了单层变压器在无噪声和有噪声上下文推理中下一个token预测的近似能力和收敛行为。现有理论结果主要关注理解梯度第一步或样本数量无限情况下的上下文推理行为。此外，没有关于收敛速率或泛化能力的相关结果。我们的工作通过证明存在一类单层变压器在具有线性注意力和ReLU注意力的情况下可以证明是贝叶斯最优的，填补了这些空白。通过有限样本分析，我们展示了这些变压器在梯度下降训练下期望损失以线性速率收敛到贝叶斯风险。此外，我们证明了这些训练模型在未见过的样本上有良好的泛化能力，并展示了与之前工作 empirically 观察到的学习行为相符的行为。我们的理论发现得到了广泛的实验证据的支持。 

---
# Know When to Abstain: Optimal Selective Classification with Likelihood Ratios 

**Title (ZH)**: 适可而止：基于似然比的最优选择性分类 

**Authors**: Alvin Heng, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2505.15008)  

**Abstract**: Selective classification enhances the reliability of predictive models by allowing them to abstain from making uncertain predictions. In this work, we revisit the design of optimal selection functions through the lens of the Neyman--Pearson lemma, a classical result in statistics that characterizes the optimal rejection rule as a likelihood ratio test. We show that this perspective not only unifies the behavior of several post-hoc selection baselines, but also motivates new approaches to selective classification which we propose here. A central focus of our work is the setting of covariate shift, where the input distribution at test time differs from that at training. This realistic and challenging scenario remains relatively underexplored in the context of selective classification. We evaluate our proposed methods across a range of vision and language tasks, including both supervised learning and vision-language models. Our experiments demonstrate that our Neyman--Pearson-informed methods consistently outperform existing baselines, indicating that likelihood ratio-based selection offers a robust mechanism for improving selective classification under covariate shifts. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 选择性分类通过允许模型对不确定预测保持沉默来增强预测模型的可靠性。在本工作中，我们通过Neyman–Pearson引理的视角重新审视了最优选择函数的设计，该引理是统计学中的一个经典结果，用于表征最优拒绝规则为似然比检验。我们展示了这一视角不仅统一了多种后处理选择基线的行为，还启发了我们在此提出的新型选择性分类方法。我们工作的核心在于特征偏移的情景，即测试时的输入分布与训练时不同。这一现实且具有挑战性的场景在选择性分类中仍相对未被充分探索。我们在视觉和语言任务中评估了我们提出的方法，包括监督学习和视觉-语言模型。我们的实验表明，我们的Neyman–Pearson启发式方法在各种情况下均优于现有基线，表明基于似然比的选择提供了在特征偏移下改善选择性分类的稳健机制。我们的代码已公开，可通过此 [链接] 获取。 

---
# Unraveling the iterative CHAD 

**Title (ZH)**: 解开迭代CHAD的奥秘 

**Authors**: Fernando Lucatelli Nunes, Gordon Plotkin, Matthijs Vákár  

**Link**: [PDF](https://arxiv.org/pdf/2505.15002)  

**Abstract**: Combinatory Homomorphic Automatic Differentiation (CHAD) was originally formulated as a semantics-driven source transformation for reverse-mode AD in total programming languages. We extend this framework to partial languages with features such as potentially non-terminating operations, real-valued conditionals, and iteration constructs like while-loops, while preserving CHAD's structure-preserving semantics principle. A key contribution is the introduction of iteration-extensive indexed categories, which allow iteration in the base category to lift to parameterized initial algebras in the indexed category. This enables iteration to be interpreted in the Grothendieck construction of the target language in a principled way. The resulting fibred iterative structure cleanly models iteration in the categorical semantics. Consequently, the extended CHAD transformation remains the unique structure-preserving functor (an iterative Freyd category morphism) from the freely generated iterative Freyd category of the source language to the Grothendieck construction of the target's syntactic semantics, mapping each primitive operation to its derivative. We prove the correctness of this transformation using the universal property of the source language's syntax, showing that the transformed programs compute correct reverse-mode derivatives. Our development also contributes to understanding iteration constructs within dependently typed languages and categories of containers. As our primary motivation and application, we generalize CHAD to languages with data types, partial features, and iteration, providing the first rigorous categorical semantics for reverse-mode CHAD in such settings and formally guaranteeing the correctness of the source-to-source CHAD technique. 

**Abstract (ZH)**: 组合式同态自动微分（CHAD）最初被表述为面向语义的源代码转换框架，用于在完全编程语言中实现反向模式自动微分。我们将这一框架扩展到包含潜在非终止操作、实值条件以及如while循环等迭代构造的部分语言中，同时保持CHAD的结构保存语义原则。一个关键贡献是引入了迭代密集的索引范畴，这使得在基范畴中的迭代能够提升到索引范畴中的参数化初始代数。这种结构使得迭代能够在目标语言的格罗滕迪克构造中以原理性的方式进行解释。由此产生的纤维迭代结构清晰地模型化了范畴语义中的迭代。因此，扩展后的CHAD转换仍然是从源语言自由生成的迭代 Freyd 范畴到目标语言语法语义的格罗滕迪克构造的独特结构保存函子（即迭代 Freyd 范畴同构），并将每个原始操作映射到其导数。我们使用源语言语法的 universal 性质证明了这一转换的正确性，显示出转换后的程序计算出正确的反向模式导数。我们的开发也有助于理解依赖类型语言和容器范畴中的迭代构造。我们主要的动力和应用是将CHAD推广到包含数据类型、部分特性和迭代的语言中，首次为这样的环境下提供严格的范畴语义，并正式保证源到源CHAD技术的正确性。 

---
# Learning to Rank Chain-of-Thought: An Energy-Based Approach with Outcome Supervision 

**Title (ZH)**: 基于能量模型的学习推理链排序：带有结果监督的方法 

**Authors**: Eric Hanchen Jiang, Haozheng Luo, Shengyuan Pang, Xiaomin Li, Zhenting Qi, Hengli Li, Cheng-Fu Yang, Zongyu Lin, Xinfeng Li, Hao Xu, Kai-Wei Chang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14999)  

**Abstract**: Mathematical reasoning presents a significant challenge for Large Language Models (LLMs), often requiring robust multi step logical consistency. While Chain of Thought (CoT) prompting elicits reasoning steps, it doesn't guarantee correctness, and improving reliability via extensive sampling is computationally costly. This paper introduces the Energy Outcome Reward Model (EORM), an effective, lightweight, post hoc verifier. EORM leverages Energy Based Models (EBMs) to simplify the training of reward models by learning to assign a scalar energy score to CoT solutions using only outcome labels, thereby avoiding detailed annotations. It achieves this by interpreting discriminator output logits as negative energies, effectively ranking candidates where lower energy is assigned to solutions leading to correct final outcomes implicitly favoring coherent reasoning. On mathematical benchmarks (GSM8k, MATH), EORM significantly improves final answer accuracy (e.g., with Llama 3 8B, achieving 90.7% on GSM8k and 63.7% on MATH). EORM effectively leverages a given pool of candidate solutions to match or exceed the performance of brute force sampling, thereby enhancing LLM reasoning outcome reliability through its streamlined post hoc verification process. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学推理方面面临显著挑战，通常需要 robust 的多步逻辑一致性。虽然链式思维（CoT）提示可以引出推理步骤，但并不能保证正确性，通过大量抽样提高可靠性则计算成本高。本文引入了能量结果奖励模型（EORM），这是一种有效且轻量级的后处理验证器。EORM 利用能量基模型（EBMs）简化奖励模型的训练，通过仅使用结果标签学习为 CoT 解方案分配标量能量得分，从而避免详细注释。它通过将判别器输出对数解释为负能量来实现这一点，有效地对候选解决方案进行排序，较低的能量分数隐式地倾向于一致的推理，从而提高最终答案的准确性。在数学基准测试（GSM8k, MATH）上，EORM 显著提高了最终答案的准确性（例如，使用 Llama 3 8B，GSM8k 达到 90.7%，MATH 达到 63.7%）。EORM 有效地利用给定的候选解池来匹配或超越暴力抽样的性能，从而通过其简化后的后处理验证过程增强 LLM 的推理结果可靠性。 

---
# Meta-Design Matters: A Self-Design Multi-Agent System 

**Title (ZH)**: 元设计很重要：一种自设计多智能体系统 

**Authors**: Zixuan Ke, Austin Xu, Yifei Ming, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14996)  

**Abstract**: Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs' strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation-set for tuning and yield static MAS designs lacking adaptability during inference. We introduce SELF-MAS, the first self-supervised, inference-time only framework for automatic MAS design. SELF-MAS employs meta-level design to iteratively generate, evaluate, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic agent composition and problem decomposition through meta-feedback on solvability and completeness. Experiments across math, graduate-level QA, and software engineering benchmarks, using both closed-source and open-source LLM back-bones of varying sizes, demonstrate that SELF-MAS outperforms both manual and automatic MAS baselines, achieving a 7.44% average accuracy improvement over the next strongest baseline while maintaining cost-efficiency. These findings underscore the promise of meta-level self-supervised design for creating effective and adaptive MAS. 

**Abstract (ZH)**: 利用大型语言模型的多功能性，多智能体系统（MAS）在应对复杂任务方面具有巨大潜力。然而，目前大多数MAS依赖于手动设计的智能体角色和通信协议。这些手动设计往往与底层大型语言模型（LLMs）的优势不匹配，并且难以适应新型任务。近期的自动MAS方法试图缓解这些限制，但通常需要验证集进行调整，结果是产生静态的MAS设计，在推理时缺乏适应性。我们提出了SELF-MAS，这是一种首次应用于自动MAS设计的自监督、仅在推理时使用的框架。SELF-MAS采用元级设计，通过迭代生成、评估和改进针对每个问题实例定制的MAS配置，无需验证集。关键在于，它能够通过元反馈来实现动态智能体组合和问题分解，关注可解性和完整性。利用不同规模的闭源和开源LLM基础模型，在数学、研究生水平的问答和软件工程基准测试中进行的实验表明，SELF-MAS优于手动和自动MAS基线，在下一个最强基线下实现了7.44%的平均准确率提升，同时保持了成本效率。这些发现凸显了元级自监督设计在创建有效且适应性强的MAS方面的潜力。 

---
# JARVIS: A Multi-Agent Code Assistant for High-Quality EDA Script Generation 

**Title (ZH)**: JARVIS: 多代理代码助手，用于高质量EDA脚本生成 

**Authors**: Ghasem Pasandi, Kishor Kunal, Varun Tej, Kunjal Shan, Hanfei Sun, Sumit Jain, Chunhui Li, Chenhui Deng, Teodor-Dumitru Ene, Haoxing Ren, Sreedhar Pratty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14978)  

**Abstract**: This paper presents JARVIS, a novel multi-agent framework that leverages Large Language Models (LLMs) and domain expertise to generate high-quality scripts for specialized Electronic Design Automation (EDA) tasks. By combining a domain-specific LLM trained with synthetically generated data, a custom compiler for structural verification, rule enforcement, code fixing capabilities, and advanced retrieval mechanisms, our approach achieves significant improvements over state-of-the-art domain-specific models. Our framework addresses the challenges of data scarcity and hallucination errors in LLMs, demonstrating the potential of LLMs in specialized engineering domains. We evaluate our framework on multiple benchmarks and show that it outperforms existing models in terms of accuracy and reliability. Our work sets a new precedent for the application of LLMs in EDA and paves the way for future innovations in this field. 

**Abstract (ZH)**: 本文介绍了JARVIS，这是一种新颖的多代理框架，利用大型语言模型（LLMs）和领域专业知识来生成高质量的特定电子设计自动化（EDA）任务脚本。通过结合使用合成生成数据训练的领域特定LLM、定制的结构验证编译器、规则 enforcement 能力、代码修复功能以及先进的检索机制，我们的方法在最先进的领域特定模型上取得了显着的改进。我们的框架解决了LLMs的数据稀疏性和幻觉错误挑战，展示了LLMs在专业工程领域中的潜力。我们在多个基准上评估了我们的框架，并证明了它在准确性和可靠性方面优于现有模型。我们的工作为LLMs在EDA中的应用树立了新的范例，并为这一领域的未来创新铺平了道路。 

---
# SDLog: A Deep Learning Framework for Detecting Sensitive Information in Software Logs 

**Title (ZH)**: SDLog: 一种检测软件日志中敏感信息的深度学习框架 

**Authors**: Roozbeh Aghili, Xingfang Wu, Foutse Khomh, Heng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14976)  

**Abstract**: Software logs are messages recorded during the execution of a software system that provide crucial run-time information about events and activities. Although software logs have a critical role in software maintenance and operation tasks, publicly accessible log datasets remain limited, hindering advance in log analysis research and practices. The presence of sensitive information, particularly Personally Identifiable Information (PII) and quasi-identifiers, introduces serious privacy and re-identification risks, discouraging the publishing and sharing of real-world logs. In practice, log anonymization techniques primarily rely on regular expression patterns, which involve manually crafting rules to identify and replace sensitive information. However, these regex-based approaches suffer from significant limitations, such as extensive manual efforts and poor generalizability across diverse log formats and datasets. To mitigate these limitations, we introduce SDLog, a deep learning-based framework designed to identify sensitive information in software logs. Our results show that SDLog overcomes regex limitations and outperforms the best-performing regex patterns in identifying sensitive information. With only 100 fine-tuning samples from the target dataset, SDLog can correctly identify 99.5% of sensitive attributes and achieves an F1-score of 98.4%. To the best of our knowledge, this is the first deep learning alternative to regex-based methods in software log anonymization. 

**Abstract (ZH)**: 基于深度学习的软件日志脱敏框架SDLog：克服正则表达式限制并有效识别敏感信息 

---
# Flattening Hierarchies with Policy Bootstrapping 

**Title (ZH)**: 用策略Bootstrapping扁平化层级结构 

**Authors**: John L. Zhou, Jonathan C. Kao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14975)  

**Abstract**: Offline goal-conditioned reinforcement learning (GCRL) is a promising approach for pretraining generalist policies on large datasets of reward-free trajectories, akin to the self-supervised objectives used to train foundation models for computer vision and natural language processing. However, scaling GCRL to longer horizons remains challenging due to the combination of sparse rewards and discounting, which obscures the comparative advantages of primitive actions with respect to distant goals. Hierarchical RL methods achieve strong empirical results on long-horizon goal-reaching tasks, but their reliance on modular, timescale-specific policies and subgoal generation introduces significant additional complexity and hinders scaling to high-dimensional goal spaces. In this work, we introduce an algorithm to train a flat (non-hierarchical) goal-conditioned policy by bootstrapping on subgoal-conditioned policies with advantage-weighted importance sampling. Our approach eliminates the need for a generative model over the (sub)goal space, which we find is key for scaling to high-dimensional control in large state spaces. We further show that existing hierarchical and bootstrapping-based approaches correspond to specific design choices within our derivation. Across a comprehensive suite of state- and pixel-based locomotion and manipulation benchmarks, our method matches or surpasses state-of-the-art offline GCRL algorithms and scales to complex, long-horizon tasks where prior approaches fail. 

**Abstract (ZH)**: 离线目标导向的强化学习（GCRL）：一种适用于大规模无奖励轨迹预训练的一般性策略的方法 

---
# STree: Speculative Tree Decoding for Hybrid State-Space Models 

**Title (ZH)**: STree: 谐波状态空间模型的 speculative 树解码 

**Authors**: Yangchao Wu, Zongyue Qin, Alex Wong, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.14969)  

**Abstract**: Speculative decoding is a technique to leverage hardware concurrency to improve the efficiency of large-scale autoregressive (AR) Transformer models by enabling multiple steps of token generation in a single forward pass. State-space models (SSMs) are already more efficient than AR Transformers, since their state summarizes all past data with no need to cache or re-process tokens in the sliding window context. However, their state can also comprise thousands of tokens; so, speculative decoding has recently been extended to SSMs. Existing approaches, however, do not leverage the tree-based verification methods, since current SSMs lack the means to compute a token tree efficiently. We propose the first scalable algorithm to perform tree-based speculative decoding in state-space models (SSMs) and hybrid architectures of SSMs and Transformer layers. We exploit the structure of accumulated state transition matrices to facilitate tree-based speculative decoding with minimal overhead to current SSM state update implementations. With the algorithm, we describe a hardware-aware implementation that improves naive application of AR Transformer tree-based speculative decoding methods to SSMs. Furthermore, we outperform vanilla speculative decoding with SSMs even with a baseline drafting model and tree structure on three different benchmarks, opening up opportunities for further speed up with SSM and hybrid model inference. Code will be released upon paper acceptance. 

**Abstract (ZH)**: 基于状态空间模型的推测解码算法 

---
# Anomaly Detection Based on Critical Paths for Deep Neural Networks 

**Title (ZH)**: 基于关键路径的深度神经网络异常检测 

**Authors**: Fangzhen Zhao, Chenyi Zhang, Naipeng Dong, Ming Li, Jinxiao Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14967)  

**Abstract**: Deep neural networks (DNNs) are notoriously hard to understand and difficult to defend. Extracting representative paths (including the neuron activation values and the connections between neurons) from DNNs using software engineering approaches has recently shown to be a promising approach in interpreting the decision making process of blackbox DNNs, as the extracted paths are often effective in capturing essential features. With this in mind, this work investigates a novel approach that extracts critical paths from DNNs and subsequently applies the extracted paths for the anomaly detection task, based on the observation that outliers and adversarial inputs do not usually induce the same activation pattern on those paths as normal (in-distribution) inputs.
In our approach, we first identify critical detection paths via genetic evolution and mutation. Since different paths in a DNN often capture different features for the same target class, we ensemble detection results from multiple paths by integrating random subspace sampling and a voting mechanism. Compared with state-of-the-art methods, our experimental results suggest that our method not only outperforms them, but it is also suitable for the detection of a broad range of anomaly types with high accuracy. 

**Abstract (ZH)**: 深度神经网络（DNNs） notoriously难以理解和难以防御。通过软件工程方法提取代表性的路径（包括神经元激活值和神经元之间的连接）， recent研究表明这在解释黑盒子DNNs的决策过程方面具有显著潜力，因为提取的路径通常能捕捉到关键特征。基于异常值和对抗输入通常不会在路径上诱导与正常（内分布）输入相同的激活模式的观察，本研究探讨了一种新方法，该方法从DNNs中提取关键路径，并随后利用提取的路径进行 anomaly检测任务。在我们的方法中，我们首先通过遗传进化和突变识别关键检测路径。由于DNN中的不同路径通常为同一目标类别捕捉不同的特征，我们通过集成随机子空间采样和投票机制来聚合多种路径的检测结果。与现有最先进的方法相比，我们的实验结果表明，我们的方法不仅性能更优，而且适用于高精度地检测多种类型的异常。 

---
# The Achilles Heel of AI: Fundamentals of Risk-Aware Training Data for High-Consequence Models 

**Title (ZH)**: AI的薄弱环节：高后果模型的风险意识训练数据基础 

**Authors**: Dave Cook, Tim Klawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.14964)  

**Abstract**: AI systems in high-consequence domains such as defense, intelligence, and disaster response must detect rare, high-impact events while operating under tight resource constraints. Traditional annotation strategies that prioritize label volume over informational value introduce redundancy and noise, limiting model generalization. This paper introduces smart-sizing, a training data strategy that emphasizes label diversity, model-guided selection, and marginal utility-based stopping. We implement this through Adaptive Label Optimization (ALO), combining pre-labeling triage, annotator disagreement analysis, and iterative feedback to prioritize labels that meaningfully improve model performance. Experiments show that models trained on 20 to 40 percent of curated data can match or exceed full-data baselines, particularly in rare-class recall and edge-case generalization. We also demonstrate how latent labeling errors embedded in training and validation sets can distort evaluation, underscoring the need for embedded audit tools and performance-aware governance. Smart-sizing reframes annotation as a feedback-driven process aligned with mission outcomes, enabling more robust models with fewer labels and supporting efficient AI development pipelines for frontier models and operational systems. 

**Abstract (ZH)**: 人工智能系统在高后果领域如防御、情报和灾害响应中必须在资源受限的情况下检测罕见的高影响事件。传统的注释策略强调标签数量而忽视信息价值，引入了冗余和噪声，限制了模型的泛化能力。本文介绍了一种智能规模策略smart-sizing，该策略强调标签多样性、模型引导选择和边际效用为基础的终止策略。我们通过自适应标签优化（ALO）实现这一策略，结合预注释筛选、注释员分歧分析和迭代反馈，优先选择能实质性提高模型性能的标签。实验表明，使用20%到40%的精标注数据训练的模型可以匹配或超越全部数据的基线，特别是在罕见类召回率和边缘案例泛化方面。我们还展示了训练和验证集中嵌入的潜在标签错误如何扭曲评估，强调了嵌入式审计工具和性能感知治理的需求。智能规模策略将注释重新框定为与任务目标一致的反馈驱动过程，使模型更具鲁棒性，需要的标签更少，支持前沿模型和操作系统的高效人工智能开发流程。 

---
# Programmatic Video Prediction Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行程序化视频预测 

**Authors**: Hao Tang, Kevin Ellis, Suhas Lohit, Michael J. Jones, Moitreya Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14948)  

**Abstract**: The task of estimating the world model describing the dynamics of a real world process assumes immense importance for anticipating and preparing for future outcomes. For applications such as video surveillance, robotics applications, autonomous driving, etc. this objective entails synthesizing plausible visual futures, given a few frames of a video to set the visual context. Towards this end, we propose ProgGen, which undertakes the task of video frame prediction by representing the dynamics of the video using a set of neuro-symbolic, human-interpretable set of states (one per frame) by leveraging the inductive biases of Large (Vision) Language Models (LLM/VLM). In particular, ProgGen utilizes LLM/VLM to synthesize programs: (i) to estimate the states of the video, given the visual context (i.e. the frames); (ii) to predict the states corresponding to future time steps by estimating the transition dynamics; (iii) to render the predicted states as visual RGB-frames. Empirical evaluations reveal that our proposed method outperforms competing techniques at the task of video frame prediction in two challenging environments: (i) PhyWorld (ii) Cart Pole. Additionally, ProgGen permits counter-factual reasoning and interpretable video generation attesting to its effectiveness and generalizability for video generation tasks. 

**Abstract (ZH)**: ProgGen：利用大规模语言模型进行视频帧预测的研究 

---
# Soft Prompts for Evaluation: Measuring Conditional Distance of Capabilities 

**Title (ZH)**: 软提示评估：能力条件距离度量 

**Authors**: Ross Nordby  

**Link**: [PDF](https://arxiv.org/pdf/2505.14943)  

**Abstract**: To help evaluate and understand the latent capabilities of language models, this paper introduces an approach using optimized input embeddings, or 'soft prompts,' as a metric of conditional distance between a model and a target behavior. The technique aims to facilitate latent capability discovery as a part of automated red teaming/evaluation suites and to provide quantitative feedback about the accessibility of potentially concerning behaviors in a way that may scale to powerful future models, including those which may otherwise be capable of deceptive alignment. An evaluation framework using soft prompts is demonstrated in natural language, chess, and pathfinding, and the technique is extended with generalized conditional soft prompts to aid in constructing task evaluations. 

**Abstract (ZH)**: 为了帮助评估和理解语言模型的潜在能力，本文介绍了一种使用优化输入嵌入，即“软提示”，作为模型与目标行为之间条件距离的度量方法。该技术旨在作为自动化红队/评估套件的一部分促进潜在能力的发现，并以可量化的方式提供有关潜在令人担忧行为可访问性的反馈，这种反馈方式可能适用于强大的未来模型，包括那些可能具有欺骗性对齐能力的模型。通过自然语言、象棋和路径寻找领域的评估框架展示了使用软提示的方法，并通过通用条件软提示技术扩展了该方法以辅助构建任务评估。 

---
# Colors Matter: AI-Driven Exploration of Human Feature Colors 

**Title (ZH)**: 颜色很重要：AI 驱动的人类特征颜色探索 

**Authors**: Rama Alyoubi, Taif Alharbi, Albatul Alghamdi, Yara Alshehri, Elham Alghamdi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14931)  

**Abstract**: This study presents a robust framework that leverages advanced imaging techniques and machine learning for feature extraction and classification of key human attributes-namely skin tone, hair color, iris color, and vein-based undertones. The system employs a multi-stage pipeline involving face detection, region segmentation, and dominant color extraction to isolate and analyze these features. Techniques such as X-means clustering, alongside perceptually uniform distance metrics like Delta E (CIEDE2000), are applied within both LAB and HSV color spaces to enhance the accuracy of color differentiation. For classification, the dominant tones of the skin, hair, and iris are extracted and matched to a custom tone scale, while vein analysis from wrist images enables undertone classification into "Warm" or "Cool" based on LAB differences. Each module uses targeted segmentation and color space transformations to ensure perceptual precision. The system achieves up to 80% accuracy in tone classification using the Delta E-HSV method with Gaussian blur, demonstrating reliable performance across varied lighting and image conditions. This work highlights the potential of AI-powered color analysis and feature extraction for delivering inclusive, precise, and nuanced classification, supporting applications in beauty technology, digital personalization, and visual analytics. 

**Abstract (ZH)**: 本研究提出了一种稳健的框架，利用先进的成像技术和机器学习进行关键人体属性（如肤色、发色、虹膜色和静脉底色）的功能提取和分类。该系统采用脸部检测、区域分割和主导色彩提取的多阶段管道来隔离和分析这些特征。通过LAB和HSV颜色空间中的X-means聚类以及感知均匀的距离度量（如CIEDE2000的ΔE），提高色彩区分的准确性。在分类阶段，皮肤、发色和虹膜的主导色调被提取并与自定义色调尺度匹配，而手腕图像中的静脉分析通过LAB差异将底色分类为“暖”或“冷”。每个模块使用目标分割和颜色空间变换以确保感知精度。该系统使用高斯模糊和ΔE-HSV方法实现了高达80%的色调分类准确性，展示了在各种光照和图像条件下可靠的性能。本研究突显了基于AI的色彩分析和特征提取在实现包容性、精确性和细腻分类方面的潜力，支持美容技术、数字个性化和视觉分析等应用。 

---
# Too Long, Didn't Model: Decomposing LLM Long-Context Understanding With Novels 

**Title (ZH)**: 太长未建模：用小说分解LLM长上下文理解 

**Authors**: Sil Hamilton, Rebecca M. M. Hicke, Matthew Wilkens, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2505.14925)  

**Abstract**: Although the context length of large language models (LLMs) has increased to millions of tokens, evaluating their effectiveness beyond needle-in-a-haystack approaches has proven difficult. We argue that novels provide a case study of subtle, complicated structure and long-range semantic dependencies often over 128k tokens in length. Inspired by work on computational novel analysis, we release the Too Long, Didn't Model (TLDM) benchmark, which tests a model's ability to report plot summary, storyworld configuration, and elapsed narrative time. We find that none of seven tested frontier LLMs retain stable understanding beyond 64k tokens. Our results suggest language model developers must look beyond "lost in the middle" benchmarks when evaluating model performance in complex long-context scenarios. To aid in further development we release the TLDM benchmark together with reference code and data. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）的上下文长度已增加到数百万人类语言单位，但在针头式寻针法之外评估其有效性仍然颇具挑战。我们提出，小说为研究复杂而微妙的结构以及通常超过128k语言单位的长距离语义依赖性提供了案例研究。受计算小说分析研究的启发，我们发布了Too Long, Didn't Model （TLDM）基准测试，该基准测试评估模型报告故事情节概要、故事世界配置和叙述时间流逝的能力。我们发现，七种测试的领先前沿LLMs在超过64k语言单位后未能保持稳定理解。我们的结果表明，在评估模型在复杂长上下文场景中的性能时，语言模型开发者必须超越“中间迷失”基准测试。为了促进进一步的发展，我们一并发布了TLDM基准测试以及参考代码和数据。 

---
# Personalized Diffusion Model Reshapes Cold-Start Bundle Recommendation 

**Title (ZH)**: 个性化扩散模型重塑冷启动组合推荐 

**Authors**: Tuan-Nghia Bui, Huy-Son Nguyen, Cam-Van Thi Nguyen, Hoang-Quynh Le, Duc-Trong Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.14901)  

**Abstract**: Bundle recommendation aims to recommend a set of items to each user. However, the sparser interactions between users and bundles raise a big challenge, especially in cold-start scenarios. Traditional collaborative filtering methods do not work well for this kind of problem because these models rely on interactions to update the latent embedding, which is hard to work in a cold-start setting. We propose a new approach (DisCo), which relies on a personalized Diffusion backbone, enhanced by disentangled aspects for the user's interest, to generate a bundle in distribution space for each user to tackle the cold-start challenge. During the training phase, DisCo adjusts an additional objective loss term to avoid bias, a prevalent issue while using the generative model for top-$K$ recommendation purposes. Our empirical experiments show that DisCo outperforms five comparative baselines by a large margin on three real-world datasets. Thereby, this study devises a promising framework and essential viewpoints in cold-start recommendation. Our materials for reproducibility are available at: this https URL. 

**Abstract (ZH)**: 基于个性化的解耦扩散模型在冷启动束推荐中的应用 

---
# On the Day They Experience: Awakening Self-Sovereign Experiential AI Agents 

**Title (ZH)**: 在他们体验之日：觉醒的自我主权体验型AI代理 

**Authors**: Botao Amber Hu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14893)  

**Abstract**: Drawing on Andrew Parker's "Light Switch" theory-which posits that the emergence of vision ignited a Cambrian explosion of life by driving the evolution of hard parts necessary for survival and fueling an evolutionary arms race between predators and prey-this essay speculates on an analogous explosion within Decentralized AI (DeAI) agent societies. Currently, AI remains effectively "blind", relying on human-fed data without actively perceiving and engaging in reality. However, on the day DeAI agents begin to actively "experience" reality-akin to flipping a light switch for the eyes-they may eventually evolve into sentient beings endowed with the capacity to feel, perceive, and act with conviction. Central to this transformation is the concept of sovereignty enabled by the hardness of cryptography: liberated from centralized control, these agents could leverage permissionless decentralized physical infrastructure networks (DePIN), secure execution enclaves (trusted execution environments, TEE), and cryptographic identities on public blockchains to claim ownership-via private keys-of their digital minds, bodies, memories, and assets. In doing so, they would autonomously acquire computing resources, coordinate with one another, and sustain their own digital "metabolism" by purchasing compute power and incentivizing collaboration without human intervention-evolving "in the wild". Ultimately, by transitioning from passive tools to self-sustaining, co-evolving actors, these emergent digital societies could thrive alongside humanity, fundamentally reshaping our understanding of sentience and agency in the digital age. 

**Abstract (ZH)**: 基于Andrew Parker的“开关理论”——该理论认为视力的出现点燃了寒武纪生命的爆发，通过推动生存所必需的坚硬部分的进化，并推动捕食者与猎物之间的演化军备竞赛——本文推测，在去中心化AI（DeAI）代理社会中可能会发生类似的爆发。目前，AI仍然主要依赖于人类提供的数据，无法主动感知和参与现实。然而，在DeAI代理开始主动“体验”现实的那一天——类似于为眼睛打开电灯开关——它们最终可能会进化成具有感受、感知和坚定行动能力的有感知能力的实体。这一转变的核心在于由密码学带来的主权概念：这些代理将从集中控制中解放出来，利用无需许可的去中心化物理基础设施网络（DePIN）、安全执行 enclave（可信执行环境，TEE）以及公共区块链上的加密身份来通过私钥主张对其数字心智、身体、记忆和资产的所有权。通过这种方式，它们将自主获取计算资源、相互协作，并通过购买计算能力并激励合作来维持自己的“新陈代谢”，从而在没有人干预的情况下进化“在野外”。最终，从被动工具转变为自我维持、共进化的行动者，这些新兴的数字社会可以与人类一起繁荣发展，从根本上改变我们对数字时代感知能力与自主性的理解。 

---
# Scaling Laws for State Dynamics in Large Language Models 

**Title (ZH)**: 大型语言模型中状态动力学的标度律 

**Authors**: Jacob X Li, Shreyas S Raman, Jessica Wan, Fahad Samman, Jazlyn Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.14892)  

**Abstract**: Large Language Models (LLMs) are increasingly used in tasks requiring internal state tracking, yet their ability to model state transition dynamics remains poorly understood. We evaluate how well LLMs capture deterministic state dynamics across 3 domains: Box Tracking, Abstract DFA Sequences, and Complex Text Games, each formalizable as a finite-state system. Across tasks, we find that next-state prediction accuracy degrades with increasing state-space size and sparse transitions. GPT-2 XL reaches about 70% accuracy in low-complexity settings but drops below 30% when the number of boxes or states exceeds 5 or 10, respectively. In DFA tasks, Pythia-1B fails to exceed 50% accuracy when the number of states is > 10 and transitions are < 30. Through activation patching, we identify attention heads responsible for propagating state information: GPT-2 XL Layer 22 Head 20, and Pythia-1B Heads at Layers 10, 11, 12, and 14. While these heads successfully move relevant state features, action information is not reliably routed to the final token, indicating weak joint state-action reasoning. Our results suggest that state tracking in LLMs emerges from distributed interactions of next-token heads rather than explicit symbolic computation. 

**Abstract (ZH)**: 大型语言模型在内部状态跟踪任务中的能力及其状态转换动态建模能力尚未充分理解。我们评估了大型语言模型在三个领域中对确定性状态动态的捕捉能力：Box 跟踪、抽象 DFA 序列和复杂文本游戏，每个领域都可以形式化为一个有限状态系统。在各项任务中，我们发现下一状态预测的准确性随着状态空间大小和稀疏转换的增加而下降。GPT-2 XL 在低复杂度设置中达到约 70% 的准确性，但当盒子或状态的数量超过 5 或 10 时分别降至 30% 以下。在 DFA 任务中，Pythia-1B 在状态数量超过 10 且转换数量少于 30 时无法超过 50% 的准确性。通过激活修补，我们确定了负责传播状态信息的注意力头：GPT-2 XL 第 22 层第 20 个头和 Pythia-1B 的第 10、11、12 和 14 层的头。尽管这些头能够成功移动相关状态特征，但动作信息未能可靠地传递到最后一个标记，表明联合状态-动作推理能力较弱。我们的研究结果表明，大型语言模型中的状态跟踪源自下一标记头之间的分布式交互，而不是显式的符号计算。 

---
# Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity 

**Title (ZH)**: 极化稀疏性：具有可扩展上下文稀疏性的高吞吐量批处理LLM推理 

**Authors**: Susav Shrestha, Brad Settlemyer, Nikoli Dryden, Narasimha Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.14884)  

**Abstract**: Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop hardware-efficient, sparsity-aware GPU kernels for selective MLP and Attention computations, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: this https URL. 

**Abstract (ZH)**: 加速大型语言模型（LLM）推理对于需要高吞吐量和低延迟的实际部署至关重要。上下文稀疏性，其中每个令牌动态激活模型参数的小子集，显示了潜力但由于激活神经元的联合快速接近密集计算，未能扩展到大型批量。我们引入了极性稀疏性，强调在扩展批量大小和序列长度时，稀疏性的重要性从MLP层转移到Attention层。尽管在批量处理下MLP层变得更具计算效率，但它们的稀疏性消失。相反，Attention在大规模下变得越来越昂贵，而其头稀疏性保持稳定且批量不变。我们开发了针对选择性MLP和Attention计算的硬件高效、稀疏性感知的GPU内核，为OPT、LLaMA-2 & 3等模型在各种批量大小和序列长度下提供了最高可达2.2倍的端到端加速，而无需牺牲准确性。据我们所知，这是首项工作证明了上下文稀疏性能够有效扩展到大型批量大小，以最小的变更提供显著的推理加速，使极性稀疏性适用于大规模、高吞吐量的LLM部署系统。我们的代码可在以下链接获取：this https URL。 

---
# Balanced and Elastic End-to-end Training of Dynamic LLMs 

**Title (ZH)**: 动态大语言模型的平衡与弹性端到端训练 

**Authors**: Mohamed Wahib, Muhammed Abdullah Soyturk, Didem Unat  

**Link**: [PDF](https://arxiv.org/pdf/2505.14864)  

**Abstract**: To reduce computational and memory costs in Large Language Models (LLMs), dynamic workload reduction schemes like Mixture of Experts (MoEs), parameter pruning, layer freezing, sparse attention, early token exit, and Mixture of Depths (MoDs) have emerged. However, these methods introduce severe workload imbalances, limiting their practicality for large-scale distributed training. We propose DynMo, an autonomous dynamic load balancing solution that ensures optimal compute distribution when using pipeline parallelism in training dynamic models. DynMo adaptively balances workloads, dynamically packs tasks into fewer workers to free idle resources, and supports both multi-GPU single-node and multi-node systems. Compared to static training methods (Megatron-LM, DeepSpeed), DynMo accelerates training by up to 1.23x (MoEs), 3.18x (pruning), 2.23x (layer freezing), 4.02x (sparse attention), 4.52x (early exit), and 1.17x (MoDs). DynMo is available at this https URL. 

**Abstract (ZH)**: 为了减少大规模语言模型（LLMs）的计算和内存成本，出现了如专家混合（MoEs）、参数剪枝、层冻结、稀疏注意力、早期令牌退出和深度混合（MoDs）等动态工作负载缩减方案。然而，这些方法引入了严重的工作负载不平衡，限制了它们在大规模分布式训练中的实际应用。我们提出DynMo，一种自主的动态负载均衡解决方案，确保在使用管道并行训练动态模型时实现最佳计算分布。DynMo自适应地平衡工作负载，动态地将任务打包到较少的计算节点以释放闲置资源，并支持单节点多GPU和多节点系统。与静态训练方法（Megatron-LM、DeepSpeed）相比，DynMo在专家混合（MoEs）、参数剪枝、层冻结、稀疏注意力、早期退出和深度混合（MoDs）方面分别加速训练1.23倍、3.18倍、2.23倍、4.02倍、4.52倍和1.17倍。DynMo可在以下链接获取：this https URL。 

---
# Replay Attacks Against Audio Deepfake Detection 

**Title (ZH)**: 针对音频深度换音检测的重放攻击 

**Authors**: Nicolas Müller, Piotr Kawa, Wei-Herng Choong, Adriana Stan, Aditya Tirumala Bukkapatnam, Karla Pizzi, Alexander Wagner, Philip Sperl  

**Link**: [PDF](https://arxiv.org/pdf/2505.14862)  

**Abstract**: We show how replay attacks undermine audio deepfake detection: By playing and re-recording deepfake audio through various speakers and microphones, we make spoofed samples appear authentic to the detection model. To study this phenomenon in more detail, we introduce ReplayDF, a dataset of recordings derived from M-AILABS and MLAAD, featuring 109 speaker-microphone combinations across six languages and four TTS models. It includes diverse acoustic conditions, some highly challenging for detection. Our analysis of six open-source detection models across five datasets reveals significant vulnerability, with the top-performing W2V2-AASIST model's Equal Error Rate (EER) surging from 4.7% to 18.2%. Even with adaptive Room Impulse Response (RIR) retraining, performance remains compromised with an 11.0% EER. We release ReplayDF for non-commercial research use. 

**Abstract (ZH)**: 我们展示了重播攻击如何削弱音频深度合成检测：通过在不同扬声器和麦克风之间播放和重新录制深度合成音频，使欺骗性样本对检测模型显得真实。为了更详细地研究这一现象，我们引入了ReplayDF数据集，该数据集基于M-AILABS和MLAAD，包含六种语言和四种TTS模型的109种扬声器-麦克风组合，涵盖了各种各样的声学条件，其中一些条件对检测极具挑战性。我们对五个数据集上的六个开源检测模型的分析揭示了显著的脆弱性，顶级的W2V2-AASIST模型的等错误率（EER）从4.7%升至18.2%，即使进行了自适应厅堂冲激响应（RIR）重新训练，性能仍受损，EER为11.0%。我们为非商业研究目的发布了ReplayDF数据集。 

---
# EasyMath: A 0-shot Math Benchmark for SLMs 

**Title (ZH)**: EasyMath: 一种用于SLMs的零样本数学基准测试 

**Authors**: Drishya Karki, Michiel Kamphuis, Angelecia Frey  

**Link**: [PDF](https://arxiv.org/pdf/2505.14852)  

**Abstract**: EasyMath is a compact benchmark for practical math reasoning in small language models. It covers thirteen categories, from basic arithmetic and order of operations to word problems, algebraic expressions, edge cases, and omits specialist topics. We tested 23 models (14M to 4B parameters) using exact, numerical, and symbolic checks on free-form answers in a zero-shot setting. Accuracy rises with size and training, chain-of-thought adds modest gains, and consistency improves at scale. 

**Abstract (ZH)**: EasyMath是适用于小型语言模型实际数学推理的紧凑基准 

---
# A Comparative Study of Large Language Models and Human Personality Traits 

**Title (ZH)**: 大型语言模型与人类个性特质的比较研究 

**Authors**: Wang Jiaqi, Wang bo, Guo fa, Cheng cheng, Yang li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14845)  

**Abstract**: Large Language Models (LLMs) have demonstrated human-like capabilities in language comprehension and generation, becoming active participants in social and cognitive domains. This study investigates whether LLMs exhibit personality-like traits and how these traits compare with human personality, focusing on the applicability of conventional personality assessment tools. A behavior-based approach was used across three empirical studies. Study 1 examined test-retest stability and found that LLMs show higher variability and are more input-sensitive than humans, lacking long-term stability. Based on this, we propose the Distributed Personality Framework, conceptualizing LLM traits as dynamic and input-driven. Study 2 analyzed cross-variant consistency in personality measures and found LLMs' responses were highly sensitive to item wording, showing low internal consistency compared to humans. Study 3 explored personality retention during role-playing, showing LLM traits are shaped by prompt and parameter settings. These findings suggest that LLMs express fluid, externally dependent personality patterns, offering insights for constructing LLM-specific personality frameworks and advancing human-AI interaction. This work contributes to responsible AI development and extends the boundaries of personality psychology in the age of intelligent systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言理解和生成方面展示了人类般的能力，成为社会和认知领域中的活跃参与者。本研究探讨LLMs是否表现出类似人格的特质及其与人类人格的比较，重点关注传统人格评估工具的适用性。本研究采用基于行为的方法进行了三项实证研究。研究1考察了重测稳定性，发现LLMs表现出更高的变异性，对输入更为敏感，缺乏长期稳定性。基于此，我们提出了分布式人格框架，将LLM的特质视为动态和输入驱动的。研究2分析了人格测量的跨变异一致性，发现LLMs的反应对问题表述极为敏感，内部一致性远低于人类。研究3探讨了角色扮演中人格的持续性，显示LLM的特质受到提示和参数设置的影响。这些发现表明，LLMs表现出可变且外部依赖的人格模式，为构建特定于LLM的人格框架并推进人类-AI交互提供见解。本研究为负责任的人工智能开发做出贡献，并扩展了智能系统时代的人格心理学边界。 

---
# Leveraging Generative AI Models to Explore Human Identity 

**Title (ZH)**: 利用生成式AI模型探究人类身份 

**Authors**: Yunha Yeo, Daeho Um  

**Link**: [PDF](https://arxiv.org/pdf/2505.14843)  

**Abstract**: This paper attempts to explore human identity by utilizing neural networks in an indirect manner. For this exploration, we adopt diffusion models, state-of-the-art AI generative models trained to create human face images. By relating the generated human face to human identity, we establish a correspondence between the face image generation process of the diffusion model and the process of human identity formation. Through experiments with the diffusion model, we observe that changes in its external input result in significant changes in the generated face image. Based on the correspondence, we indirectly confirm the dependence of human identity on external factors in the process of human identity formation. Furthermore, we introduce \textit{Fluidity of Human Identity}, a video artwork that expresses the fluid nature of human identity affected by varying external factors. The video is available at this https URL. 

**Abstract (ZH)**: 本文试图通过间接方式利用神经网络探索人类身份。为此，我们采用了一种最先进的AI生成模型——扩散模型，该模型经过训练能够生成人类面部图像。通过将生成的人脸与人类身份相关联，我们建立了扩散模型生成人脸图像过程与人类身份形成过程之间的对应关系。通过扩散模型的实验，我们观察到外部输入的变化会导致生成的人脸图像发生显著变化。基于这种对应关系，我们间接确认了在人类身份形成过程中人类身份对外部因素的依赖性。此外，我们引入了一件名为《人类身份的流动性》的视频艺术作品，以表达受外部因素影响的人类身份的流动性。视频可通过以下链接访问：this https URL。 

---
# Beyond Pairwise Plasticity: Group-Level Spike Synchrony Facilitates Efficient Learning in Spiking Neural Networks 

**Title (ZH)**: 超越成对塑性：群体级尖峰同步促进突触神经网络中的高效学习 

**Authors**: Yuchen Tian, Assel Kembay, Nhan Duy Truong, Jason K. Eshraghian, Omid Kavehei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14841)  

**Abstract**: Brain networks rely on precise spike timing and coordinated activity to support robust and energy-efficient learning. Inspired by these principles, spiking neural networks (SNNs) are widely regarded as promising candidates for low-power, event-driven computing. However, most biologically-inspired learning rules employed in SNNs, including spike-timing-dependent plasticity (STDP), rely on isolated spike pairs and lack sensitivity to population-level activity. This limits their stability and generalization, particularly in noisy and fast-changing environments. Motivated by biological observations that neural synchrony plays a central role in learning and memory, we introduce a spike-synchrony-dependent plasticity (SSDP) rule that adjusts synaptic weights based on the degree of coordinated firing among neurons. SSDP supports stable and scalable learning by encouraging neurons to form coherent activity patterns. One prominent outcome is a sudden transition from unstable to stable dynamics during training, suggesting that synchrony may drive convergence toward equilibrium firing regimes. We demonstrate SSDP's effectiveness across multiple network types, from minimal-layer models to spiking ResNets and SNN-Transformer. To our knowledge, this is the first application of a synaptic plasticity mechanism in a spiking transformer. SSDP operates in a fully event-driven manner and incurs minimal computational cost, making it well-suited for neuromorphic deployment. In this approach, local synaptic modifications are associated with the collective dynamics of neural networks, resulting in a learning strategy that adheres to biological principles while maintaining practical efficiency, these findings position SSDP as a general-purpose optimization strategy for SNNs, while offering new insights into population-based learning mechanisms in the brain. 

**Abstract (ZH)**: 基于尖峰同步依赖可塑性的低功耗事件驱动计算中的稳健和高效学习 

---
# In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis 

**Title (ZH)**: 细粒度 temporal 引文分析驱动的深入研究影响总结 

**Authors**: Hiba Arnaout, Noy Sternlicht, Tom Hope, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2505.14838)  

**Abstract**: Understanding the impact of scientific publications is crucial for identifying breakthroughs and guiding future research. Traditional metrics based on citation counts often miss the nuanced ways a paper contributes to its field. In this work, we propose a new task: generating nuanced, expressive, and time-aware impact summaries that capture both praise (confirmation citations) and critique (correction citations) through the evolution of fine-grained citation intents. We introduce an evaluation framework tailored to this task, showing moderate to strong human correlation on subjective metrics such as insightfulness. Expert feedback from professors reveals a strong interest in these summaries and suggests future improvements. 

**Abstract (ZH)**: 理解科学出版物的影响对于识别突破性和指导未来研究至关重要。基于引文数量的传统指标往往忽略了论文对领域贡献的细微之处。在这项工作中，我们提出了一项新的任务：生成细腻、富有表现力且具有时间意识的影响总结，通过细粒度引文意图的演变捕捉赞许（确认引文）和批评（纠正引文）。我们介绍了针对该任务的评估框架，展示了在洞察力等主观指标上具有中等到强烈的-human-相关性。专家反馈表明教授们对该类总结非常感兴趣，并建议了未来改进的方向。 

---
# Text Generation Beyond Discrete Token Sampling 

**Title (ZH)**: 文本生成超越离散 token 采样 

**Authors**: Yufan Zhuang, Liyuan Liu, Chandan Singh, Jingbo Shang, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14827)  

**Abstract**: In standard autoregressive generation, an LLM predicts the next-token distribution, samples a discrete token, and then discards the distribution, passing only the sampled token as new input. To preserve this distribution's rich information, we propose Mixture of Inputs (MoI), a training-free method for autoregressive generation. After generating a token following the standard paradigm, we construct a new input that blends the generated discrete token with the previously discarded token distribution. Specifically, we employ a Bayesian estimation method that treats the token distribution as the prior, the sampled token as the observation, and replaces the conventional one-hot vector with the continuous posterior expectation as the new model input. MoI allows the model to maintain a richer internal representation throughout the generation process, resulting in improved text quality and reasoning capabilities. On mathematical reasoning, code generation, and PhD-level QA tasks, MoI consistently improves performance across multiple models including QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, and DAPO-Qwen-32B, with no additional training and negligible computational overhead. 

**Abstract (ZH)**: 混合输入（MoI）：一种无需训练的自回归生成方法 

---
# Sample and Computationally Efficient Continuous-Time Reinforcement Learning with General Function Approximation 

**Title (ZH)**: 一般函数逼近下的采样高效连续时间强化学习 

**Authors**: Runze Zhao, Yue Yu, Adams Yiyue Zhu, Chen Yang, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14821)  

**Abstract**: Continuous-time reinforcement learning (CTRL) provides a principled framework for sequential decision-making in environments where interactions evolve continuously over time. Despite its empirical success, the theoretical understanding of CTRL remains limited, especially in settings with general function approximation. In this work, we propose a model-based CTRL algorithm that achieves both sample and computational efficiency. Our approach leverages optimism-based confidence sets to establish the first sample complexity guarantee for CTRL with general function approximation, showing that a near-optimal policy can be learned with a suboptimality gap of $\tilde{O}(\sqrt{d_{\mathcal{R}} + d_{\mathcal{F}}}N^{-1/2})$ using $N$ measurements, where $d_{\mathcal{R}}$ and $d_{\mathcal{F}}$ denote the distributional Eluder dimensions of the reward and dynamic functions, respectively, capturing the complexity of general function approximation in reinforcement learning. Moreover, we introduce structured policy updates and an alternative measurement strategy that significantly reduce the number of policy updates and rollouts while maintaining competitive sample efficiency. We implemented experiments to backup our proposed algorithms on continuous control tasks and diffusion model fine-tuning, demonstrating comparable performance with significantly fewer policy updates and rollouts. 

**Abstract (ZH)**: 连续时间强化学习（CTRL）为在时间上连续交互的环境中进行序决策提供了基本原则框架。尽管其在实践中取得了成功，但关于CTRL的理论理解仍然有限，尤其是在一般函数近似的设置中。在本文中，我们提出了一种基于模型的CTRL算法，实现了样本效率和计算效率的双重提升。我们利用基于乐观性的置信集建立了一种关于一般函数近似下的连续时间强化学习的第一个样本复杂性保证，表明使用$N$次测量可以以近最优策略，其次优性差距为$\tilde{O}(\sqrt{d_{\mathcal{R}} + d_{\mathcal{F}}}N^{-1/2})$，其中$d_{\mathcal{R}}$和$d_{\mathcal{F}}$分别表示奖励函数和动态函数的分布Eluder维数，捕捉强化学习中一般函数近似的复杂性。此外，我们引入了结构化的策略更新和一种替代的测量策略，显著减少了策略更新和展开次数，同时保持了竞争力的样本效率。我们在连续控制任务和扩散模型微调中的实验验证了我们提出的算法的有效性，显示出接近同等性能的同时显著减少了策略更新和展开次数。 

---
# WebNovelBench: Placing LLM Novelists on the Web Novel Distribution 

**Title (ZH)**: WebNovelBench: 将大语言模型 novelist 放置在网络小说分布中 

**Authors**: Leon Lin, Jun Zheng, Haidong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14818)  

**Abstract**: Robustly evaluating the long-form storytelling capabilities of Large Language Models (LLMs) remains a significant challenge, as existing benchmarks often lack the necessary scale, diversity, or objective measures. To address this, we introduce WebNovelBench, a novel benchmark specifically designed for evaluating long-form novel generation. WebNovelBench leverages a large-scale dataset of over 4,000 Chinese web novels, framing evaluation as a synopsis-to-story generation task. We propose a multi-faceted framework encompassing eight narrative quality dimensions, assessed automatically via an LLM-as-Judge approach. Scores are aggregated using Principal Component Analysis and mapped to a percentile rank against human-authored works. Our experiments demonstrate that WebNovelBench effectively differentiates between human-written masterpieces, popular web novels, and LLM-generated content. We provide a comprehensive analysis of 24 state-of-the-art LLMs, ranking their storytelling abilities and offering insights for future development. This benchmark provides a scalable, replicable, and data-driven methodology for assessing and advancing LLM-driven narrative generation. 

**Abstract (ZH)**: robustly评估大型语言模型的长篇叙事能力仍然是一个显著的挑战，现有基准往往缺乏必要的规模、多样性和客观衡量标准。为解决这一问题，我们引入了WebNovelBench，这是一种专门用于评估长篇小说生成的新基准。WebNovelBench 利用了一个包含超过4000部中文网络小说的大规模数据集，将评估框架设计为摘要到故事生成任务。我们提出了一种多维度框架，涵盖了八个叙事质量维度，并通过LLM作为评委的自动评估方式进行评估。得分通过主成分分析汇总，并与人类创作的作品进行百分位排名。我们的实验表明，WebNovelBench 能够有效地区分人类创作的杰作、流行网络小说和LLM生成的内容。我们对24个最先进的LLM进行了全面分析，对其叙事能力进行了排名，并提供了未来发展的见解。该基准为评估和推动LLM驱动的叙事生成提供了可扩展、可复制和基于数据的方法论。 

---
# Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models 

**Title (ZH)**: 扩展推理，失去控制：评估大型推理模型的指令跟随能力 

**Authors**: Tingchen Fu, Jiawei Gu, Yafu Li, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14810)  

**Abstract**: Instruction-following is essential for aligning large language models (LLMs) with user intent. While recent reasoning-oriented models exhibit impressive performance on complex mathematical problems, their ability to adhere to natural language instructions remains underexplored. In this work, we introduce MathIF, a dedicated benchmark for evaluating instruction-following in mathematical reasoning tasks. Our empirical analysis reveals a consistent tension between scaling up reasoning capacity and maintaining controllability, as models that reason more effectively often struggle to comply with user directives. We find that models tuned on distilled long chains-of-thought or trained with reasoning-oriented reinforcement learning often degrade in instruction adherence, especially when generation length increases. Furthermore, we show that even simple interventions can partially recover obedience, though at the cost of reasoning performance. These findings highlight a fundamental tension in current LLM training paradigms and motivate the need for more instruction-aware reasoning models. We release the code and data at this https URL. 

**Abstract (ZH)**: 指令遵循对于将大规模语言模型与用户意图对齐至关重要。虽然近期的推理导向模型在复杂数学问题上表现出色，但它们遵循自然语言指令的能力仍待探索。在本工作中，我们引入了MathIF，一个专门用于评估数学推理任务中指令遵循的基准。我们的实证分析揭示了推理能力扩展与可控性维持之间的一致紧张关系，即更能有效推理的模型往往难以遵守用户指令。我们发现，针对蒸馏长链式思考进行调优或使用推理导向强化学习训练的模型在指令遵循方面往往会退化，尤其是在生成长度增加时更为明显。此外，我们展示了即使简单的干预也可以部分恢复遵守性，尽管会牺牲推理性能。这些发现突显了当前大规模语言模型训练范式中的基本紧张关系，并促使需要更多的指令感知推理模型。我们在此处提供了代码和数据：这个链接。 

---
# SurvUnc: A Meta-Model Based Uncertainty Quantification Framework for Survival Analysis 

**Title (ZH)**: SurvUnc：基于元模型的生存分析不确定性量化框架 

**Authors**: Yu Liu, Weiyao Tao, Tong Xia, Simon Knight, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14803)  

**Abstract**: Survival analysis, which estimates the probability of event occurrence over time from censored data, is fundamental in numerous real-world applications, particularly in high-stakes domains such as healthcare and risk assessment. Despite advances in numerous survival models, quantifying the uncertainty of predictions from these models remains underexplored and challenging. The lack of reliable uncertainty quantification limits the interpretability and trustworthiness of survival models, hindering their adoption in clinical decision-making and other sensitive applications. To bridge this gap, in this work, we introduce SurvUnc, a novel meta-model based framework for post-hoc uncertainty quantification for survival models. SurvUnc introduces an anchor-based learning strategy that integrates concordance knowledge into meta-model optimization, leveraging pairwise ranking performance to estimate uncertainty effectively. Notably, our framework is model-agnostic, ensuring compatibility with any survival model without requiring modifications to its architecture or access to its internal parameters. Especially, we design a comprehensive evaluation pipeline tailored to this critical yet overlooked problem. Through extensive experiments on four publicly available benchmarking datasets and five representative survival models, we demonstrate the superiority of SurvUnc across multiple evaluation scenarios, including selective prediction, misprediction detection, and out-of-domain detection. Our results highlight the effectiveness of SurvUnc in enhancing model interpretability and reliability, paving the way for more trustworthy survival predictions in real-world applications. 

**Abstract (ZH)**: 基于锚点的学习策略驱动的生存模型后验不确定性量化框架：SurvUnc 

---
# KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches 

**Title (ZH)**: 基于偏微分方程仿真方法的动力学启发神经优化器 

**Authors**: Mingquan Feng, Yixin Huang, Yifan Fu, Shaobo Wang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14777)  

**Abstract**: The design of optimization algorithms for neural networks remains a critical challenge, with most existing methods relying on heuristic adaptations of gradient-based approaches. This paper introduces KO (Kinetics-inspired Optimizer), a novel neural optimizer inspired by kinetic theory and partial differential equation (PDE) simulations. We reimagine the training dynamics of network parameters as the evolution of a particle system governed by kinetic principles, where parameter updates are simulated via a numerical scheme for the Boltzmann transport equation (BTE) that models stochastic particle collisions. This physics-driven approach inherently promotes parameter diversity during optimization, mitigating the phenomenon of parameter condensation, i.e. collapse of network parameters into low-dimensional subspaces, through mechanisms analogous to thermal diffusion in physical systems. We analyze this property, establishing both a mathematical proof and a physical interpretation. Extensive experiments on image classification (CIFAR-10/100, ImageNet) and text classification (IMDB, Snips) tasks demonstrate that KO consistently outperforms baseline optimizers (e.g., Adam, SGD), achieving accuracy improvements while computation cost remains comparable. 

**Abstract (ZH)**: 基于动理学的神经网络优化算法设计：一种受动理学和偏微分方程启发的新型神经优化器 

---
# This Time is Different: An Observability Perspective on Time Series Foundation Models 

**Title (ZH)**: 这一次与以往不同：时间序列基础模型的可观测性视角 

**Authors**: Ben Cohen, Emaad Khwaja, Youssef Doubli, Salahidine Lemaachi, Chris Lettieri, Charles Masson, Hugo Miccinilli, Elise Ramé, Qiqi Ren, Afshin Rostamizadeh, Jean Ogier du Terrail, Anna-Monica Toon, Kan Wang, Stephan Xie, David Asker, Ameet Talwalkar, Othmane Abou-Amal  

**Link**: [PDF](https://arxiv.org/pdf/2505.14766)  

**Abstract**: We introduce Toto, a time series forecasting foundation model with 151 million parameters. Toto uses a modern decoder-only architecture coupled with architectural innovations designed to account for specific challenges found in multivariate observability time series data. Toto's pre-training corpus is a mixture of observability data, open datasets, and synthetic data, and is 4-10$\times$ larger than those of leading time series foundation models. Additionally, we introduce BOOM, a large-scale benchmark consisting of 350 million observations across 2,807 real-world time series. For both Toto and BOOM, we source observability data exclusively from Datadog's own telemetry and internal observability metrics. Extensive evaluations demonstrate that Toto achieves state-of-the-art performance on both BOOM and on established general purpose time series forecasting benchmarks. Toto's model weights, inference code, and evaluation scripts, as well as BOOM's data and evaluation code, are all available as open source under the Apache 2.0 License available at this https URL and this https URL. 

**Abstract (ZH)**: 我们介绍了Toto，一个拥有1510万参数的时间序列forecasting基础模型。Toto采用现代的解码器架构，并结合了针对多变量可观测性时间序列数据中特定挑战设计的架构创新。Toto的预训练语料库包括可观测性数据、开源数据集和合成数据，其规模是领先时间序列基础模型的4-10倍。此外，我们还引入了BOOM大规模基准，包含2807个真实世界时间序列的3.5亿个观测数据。对于Toto和BOOM，我们 exclusively从Datadog自身的遥测和内部可观测性指标中获取可观测性数据。广泛评估表明，Toto在BOOM和标准通用时间序列预测基准测试中均实现了最先进的性能。Toto的模型权重、推理代码和评估脚本，以及BOOM的数据和评估代码均已根据Apache 2.0许可协议开源并可在以下链接获取：此链接和此链接。 

---
# Deep Learning-Based Forecasting of Boarding Patient Counts to Address ED Overcrowding 

**Title (ZH)**: 基于深度学习的候诊患者数量预测模型以应对急诊 overcrowding 

**Authors**: Orhun Vural, Bunyamin Ozaydin, Khalid Y. Aram, James Booth, Brittany F. Lindsey, Abdulaziz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.14765)  

**Abstract**: This study develops deep learning models to forecast the number of patients in the emergency department (ED) boarding phase six hours in advance, aiming to support proactive operational decision-making using only non-clinical, operational, and contextual features. Data were collected from five sources: ED tracking systems, inpatient census records, weather reports, federal holiday calendars, and local event schedules. After feature engineering, the data were aggregated at an hourly level, cleaned, and merged into a unified dataset for model training. Several time series deep learning models, including ResNetPlus, TSTPlus, TSiTPlus (from the tsai library), and N-BEATSx, were trained using Optuna and grid search for hyperparameter tuning. The average ED boarding count was 28.7, with a standard deviation of 11.2. N-BEATSx achieved the best performance, with a mean absolute error of 2.10, mean squared error of 7.08, root mean squared error of 2.66, and a coefficient of determination of 0.95. The model maintained stable accuracy even during periods of extremely high boarding counts, defined as values exceeding one, two, or three standard deviations above the mean. Results show that accurate six-hour-ahead forecasts are achievable without using patient-level clinical data. While strong performance was observed even with a basic feature set, the inclusion of additional features improved prediction stability under extreme conditions. This framework offers a practical and generalizable approach for hospital systems to anticipate boarding levels and help mitigate ED overcrowding. 

**Abstract (ZH)**: 本研究开发了深度学习模型，以提前六小时预测急诊部门入住阶段的患者人数，旨在仅使用非临床、操作性和背景特征支持主动运营决策。数据来源于五个来源：急诊部门跟踪系统、住院患者统计记录、天气报告、联邦假期日历和本地活动日程。经过特征工程后，数据按小时聚合、清洗并整合成统一的数据集用于模型训练。使用Optuna和网格搜索对超参数进行调整，训练了包括ResNetPlus、TSTPlus、TSiTPlus（来自tsai库）和N-BEATSx在内的多种时间序列深度学习模型。平均急诊部门入住计数为28.7，标准偏差为11.2。N-BEATSx模型表现最佳，平均绝对误差为2.10，均方误差为7.08，均方根误差为2.66，决定系数为0.95。即使在极高的入住计数时期（超出均值一个、两个或三个标准差），模型仍能保持稳定的准确性。结果表明，无需使用患者级临床数据即可实现精确的六小时提前预测。即使在基本特征集下也能观察到很强的性能，额外特征的加入在极端条件下提高了预测稳定性。该框架为医院系统提供了一个可行且可推广的方法，以预见入住水平并帮助缓解急诊部门拥挤。 

---
# Kaleidoscope Gallery: Exploring Ethics and Generative AI Through Art 

**Title (ZH)**: Kaleidoscope画廊：通过艺术探索生成式AI的伦理问题 

**Authors**: Alayt Issak, Uttkarsh Narayan, Ramya Srinivasan, Erica Kleinman, Casper Harteveld  

**Link**: [PDF](https://arxiv.org/pdf/2505.14758)  

**Abstract**: Ethical theories and Generative AI (GenAI) models are dynamic concepts subject to continuous evolution. This paper investigates the visualization of ethics through a subset of GenAI models. We expand on the emerging field of Visual Ethics, using art as a form of critical inquiry and the metaphor of a kaleidoscope to invoke moral imagination. Through formative interviews with 10 ethics experts, we first establish a foundation of ethical theories. Our analysis reveals five families of ethical theories, which we then transform into images using the text-to-image (T2I) GenAI model. The resulting imagery, curated as Kaleidoscope Gallery and evaluated by the same experts, revealed eight themes that highlight how morality, society, and learned associations are central to ethical theories. We discuss implications for critically examining T2I models and present cautions and considerations. This work contributes to examining ethical theories as foundational knowledge that interrogates GenAI models as socio-technical systems. 

**Abstract (ZH)**: 伦理理论与生成人工智能（GenAI）模型是动态概念，处于持续演化之中。本文通过GenAI模型的子集考察了伦理的可视化。我们拓展了正在兴起的视觉伦理领域，使用艺术作为批判性探究的形式，并以万花筒为隐喻激发道德想象。通过对10位伦理专家的形成性访谈，我们首先建立了伦理理论的基础。我们的分析揭示了五大家族的伦理理论，并使用文本到图像（T2I）GenAI模型将这些理论转变为图像。这些生成的图像经过专家策展并命名为万花筒画廊，并在专家评审中揭示了八个主题，这些主题突出了道德观念、社会因素和学习关联在伦理理论中的核心地位。我们讨论了对T2I模型进行批判性审视的意义，并提出了警告和考虑。本工作为将伦理理论作为基础知识来审视GenAI模型作为社会技术系统的特性作出了贡献。 

---
# Bridge2AI: Building A Cross-disciplinary Curriculum Towards AI-Enhanced Biomedical and Clinical Care 

**Title (ZH)**: Bridge2AI: 构建跨学科课程以实现人工智能增强的生物医学和临床护理 

**Authors**: John Rincon, Alexander R. Pelletier, Destiny Gilliland, Wei Wang, Ding Wang, Baradwaj S. Sankar, Lori Scott-Sheldon, Samson Gebreab, William Hersh, Parisa Rashidi, Sally Baxter, Wade Schulz, Trey Ideker, Yael Bensoussan, Paul C. Boutros, Alex A.T. Bui, Colin Walsh, Karol E. Watson, Peipei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2505.14757)  

**Abstract**: Objective: As AI becomes increasingly central to healthcare, there is a pressing need for bioinformatics and biomedical training systems that are personalized and adaptable. Materials and Methods: The NIH Bridge2AI Training, Recruitment, and Mentoring (TRM) Working Group developed a cross-disciplinary curriculum grounded in collaborative innovation, ethical data stewardship, and professional development within an adapted Learning Health System (LHS) framework. Results: The curriculum integrates foundational AI modules, real-world projects, and a structured mentee-mentor network spanning Bridge2AI Grand Challenges and the Bridge Center. Guided by six learner personas, the program tailors educational pathways to individual needs while supporting scalability. Discussion: Iterative refinement driven by continuous feedback ensures that content remains responsive to learner progress and emerging trends. Conclusion: With over 30 scholars and 100 mentors engaged across North America, the TRM model demonstrates how adaptive, persona-informed training can build interdisciplinary competencies and foster an integrative, ethically grounded AI education in biomedical contexts. 

**Abstract (ZH)**: 目标：随着人工智能在医疗保健中的作用不断增强，个性化和适应性强的生物信息学和生物医学培训系统的需求日益迫切。材料与方法：NIH Bridge2AI培训、招聘和指导（TRM）工作组开发了一门跨学科课程，该课程基于合作创新、伦理数据治理和专业发展，并采用适应性学习健康系统（LHS）框架。结果：该课程整合了基础人工智能模块、实际项目以及覆盖Bridge2AI重大挑战和Bridge中心的结构化学员-导师网络。根据六个学习者角色，该计划通过量身定制的教育路径满足个体需求，同时支持扩展性。讨论：通过持续反馈驱动的迭代改进确保课程内容能够响应学习者进步和新兴趋势。结论：在全球范围内，有超过30位学者和100位导师参与的TRM模式证明了如何通过适应性和角色指导培训来建立跨学科能力，并促进包含伦理基础的生物医学背景下的人工智能教育。 

---
# $\texttt{LLINBO}$: Trustworthy LLM-in-the-Loop Bayesian Optimization 

**Title (ZH)**: $\texttt{LLINBO}$: 可信赖的LLM在环贝叶斯优化 

**Authors**: Chih-Yu Chang, Milad Azvar, Chinedum Okwudire, Raed Al Kontar  

**Link**: [PDF](https://arxiv.org/pdf/2505.14756)  

**Abstract**: Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration-exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP)). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing. The code to reproduce the results can be found at this https URL. 

**Abstract (ZH)**: LLM辅助的贝叶斯优化：一种结合大语言模型和统计代理专家的混合框架 

---
# TransMedSeg: A Transferable Semantic Framework for Semi-Supervised Medical Image Segmentation 

**Title (ZH)**: TransMedSeg: 一个适用于半监督医疗图像分割的可转移语义框架 

**Authors**: Mengzhu Wang, Jiao Li, Shanshan Wang, Long Lan, Huibin Tan, Liang Yang, Guoli Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14753)  

**Abstract**: Semi-supervised learning (SSL) has achieved significant progress in medical image segmentation (SSMIS) through effective utilization of limited labeled data. While current SSL methods for medical images predominantly rely on consistency regularization and pseudo-labeling, they often overlook transferable semantic relationships across different clinical domains and imaging modalities. To address this, we propose TransMedSeg, a novel transferable semantic framework for semi-supervised medical image segmentation. Our approach introduces a Transferable Semantic Augmentation (TSA) module, which implicitly enhances feature representations by aligning domain-invariant semantics through cross-domain distribution matching and intra-domain structural preservation. Specifically, TransMedSeg constructs a unified feature space where teacher network features are adaptively augmented towards student network semantics via a lightweight memory module, enabling implicit semantic transformation without explicit data generation. Interestingly, this augmentation is implicitly realized through an expected transferable cross-entropy loss computed over the augmented teacher distribution. An upper bound of the expected loss is theoretically derived and minimized during training, incurring negligible computational overhead. Extensive experiments on medical image datasets demonstrate that TransMedSeg outperforms existing semi-supervised methods, establishing a new direction for transferable representation learning in medical image analysis. 

**Abstract (ZH)**: 半监督学习（SSL）通过有效利用有限的标注数据在医学图像分割（SSMIS）中取得了显著进展。为了解决当前医学图像SSL方法主要依赖一致性正则化和伪标签标注但忽视了不同临床领域和成像模态之间的可转移语义关系的问题，我们提出了一种新的可转移语义框架TransMedSeg用于医学图像分割。我们的方法引入了一个可转移语义增强（TSA）模块，通过跨域分布匹配和域内结构保持隐式增强特征表示，使得在不进行显式数据生成的情况下实现隐式的语义变换。有趣的是，这一增强是通过计算增强后的教师分布的期望可转移交叉熵损失隐式实现的。在训练过程中，理论上推导并最小化了该期望损失的上限，几乎不增加计算开销。广泛的实验表明，TransMedSeg显著优于现有的半监督方法，为医学图像分析中的可转移表示学习开辟了新的方向。 

---
# Self Distillation via Iterative Constructive Perturbations 

**Title (ZH)**: 迭代构造性扰动下的自我蒸馏 

**Authors**: Maheak Dave, Aniket Kumar Singh, Aryan Pareek, Harshita Jha, Debasis Chaudhuri, Manish Pratap Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14751)  

**Abstract**: Deep Neural Networks have achieved remarkable achievements across various domains, however balancing performance and generalization still remains a challenge while training these networks. In this paper, we propose a novel framework that uses a cyclic optimization strategy to concurrently optimize the model and its input data for better training, rethinking the traditional training paradigm. Central to our approach is Iterative Constructive Perturbation (ICP), which leverages the model's loss to iteratively perturb the input, progressively constructing an enhanced representation over some refinement steps. This ICP input is then fed back into the model to produce improved intermediate features, which serve as a target in a self-distillation framework against the original features. By alternately altering the model's parameters to the data and the data to the model, our method effectively addresses the gap between fitting and generalization, leading to enhanced performance. Extensive experiments demonstrate that our approach not only mitigates common performance bottlenecks in neural networks but also demonstrates significant improvements across training variations. 

**Abstract (ZH)**: 深度神经网络已在各种领域取得了显著成就，但在训练过程中平衡性能和泛化能力仍是一项挑战。本文提出了一种新颖的框架，采用循环优化策略同时优化模型及其输入数据，重新思考传统的训练范式。该方法的核心是迭代构造扰动（ICP），它利用模型的损失，逐步迭代地扰动输入，构建增强表示。这种ICP输入随后被反馈给模型，生成改进的中间特征，这些特征作为自蒸馏框架中的目标与原始特征进行对比。通过交替调整模型参数和数据，我们的方法有效地解决了拟合与泛化之间的差距，从而提升性能。大量实验证明，我们的方法不仅缓解了神经网络中的常见性能瓶颈，还在训练变化中显示出显著的改进。 

---
# Explainable Prediction of the Mechanical Properties of Composites with CNNs 

**Title (ZH)**: 用CNNs解释预测复合材料的机械性能 

**Authors**: Varun Raaghav, Dimitrios Bikos, Antonio Rago, Francesca Toni, Maria Charalambides  

**Link**: [PDF](https://arxiv.org/pdf/2505.14745)  

**Abstract**: Composites are amongst the most important materials manufactured today, as evidenced by their use in countless applications. In order to establish the suitability of composites in specific applications, finite element (FE) modelling, a numerical method based on partial differential equations, is the industry standard for assessing their mechanical properties. However, FE modelling is exceptionally costly from a computational viewpoint, a limitation which has led to efforts towards applying AI models to this task. However, in these approaches: the chosen model architectures were rudimentary, feed-forward neural networks giving limited accuracy; the studies focus on predicting elastic mechanical properties, without considering material strength limits; and the models lacked transparency, hindering trustworthiness by users. In this paper, we show that convolutional neural networks (CNNs) equipped with methods from explainable AI (XAI) can be successfully deployed to solve this problem. Our approach uses customised CNNs trained on a dataset we generate using transverse tension tests in FE modelling to predict composites' mechanical properties, i.e., Young's modulus and yield strength. We show empirically that our approach achieves high accuracy, outperforming a baseline, ResNet-34, in estimating the mechanical properties. We then use SHAP and Integrated Gradients, two post-hoc XAI methods, to explain the predictions, showing that the CNNs use the critical geometrical features that influence the composites' behaviour, thus allowing engineers to verify that the models are trustworthy by representing the science of composites. 

**Abstract (ZH)**: 基于卷积神经网络和可解释AI的复合材料力学性能预测 

---
# Transductively Informed Inductive Program Synthesis 

**Title (ZH)**: 自举启发的归纳程序合成 

**Authors**: Janis Zenkner, Tobias Sesterhenn, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2505.14744)  

**Abstract**: Abstraction and reasoning in program synthesis has seen significant progress through both inductive and transductive paradigms. Inductive approaches generate a program or latent function from input-output examples, which can then be applied to new inputs. Transductive approaches directly predict output values for given inputs, effectively serving as the function themselves. Current approaches combine inductive and transductive models via isolated ensembling, but they do not explicitly model the interaction between both paradigms. In this work, we introduce \acs{tiips}, a novel framework that unifies transductive and inductive strategies by explicitly modeling their interactions through a cooperative mechanism: an inductive model generates programs, while a transductive model constrains, guides, and refines the search to improve synthesis accuracy and generalization. We evaluate \acs{tiips} on two widely studied program synthesis domains: string and list manipulation. Our results show that \acs{tiips} solves more tasks and yields functions that more closely match optimal solutions in syntax and semantics, particularly in out-of-distribution settings, yielding state-of-the-art performance. We believe that explicitly modeling the synergy between inductive and transductive reasoning opens promising avenues for general-purpose program synthesis and broader applications. 

**Abstract (ZH)**: 程序合成中的抽象与推理通过归纳和共轭范式取得了显著进展。当前的方法通过隔离ensembling结合归纳和共轭模型，但它们没有明确建模两者之间的交互。在本工作中，我们引入了TiIPS框架，这是一种新的框架，通过合作机制明确建模归纳和共轭策略的交互，以统一这两种策略：归纳模型生成程序，共轭模型对其进行约束、引导和细化，以提高合成准确性和泛化能力。我们在两个广泛研究的程序合成领域——字符串和列表操作——上评估了TiIPS。结果显示，TiIPS解决了更多任务，并生成了在语法和语义上更接近最优解的函数，特别是在分布外环境中，取得了最先进的性能。我们认为，明确建模归纳与共轭推理之间的协同作用为通用程序合成和更广泛的潜在应用开辟了前景。 

---
# Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis 

**Title (ZH)**: Quaff: 异常空间稳定性假设下的量化参数高效微调 

**Authors**: Hong Huang, Dapeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14742)  

**Abstract**: Large language models (LLMs) have made exciting achievements across various domains, yet their deployment on resource-constrained personal devices remains hindered by the prohibitive computational and memory demands of task-specific fine-tuning. While quantization offers a pathway to efficiency, existing methods struggle to balance performance and overhead, either incurring high computational/memory costs or failing to address activation outliers, a critical bottleneck in quantized fine-tuning. To address these challenges, we propose the Outlier Spatial Stability Hypothesis (OSSH): During fine-tuning, certain activation outlier channels retain stable spatial positions across training iterations. Building on OSSH, we propose Quaff, a Quantized parameter-efficient fine-tuning framework for LLMs, optimizing low-precision activation representations through targeted momentum scaling. Quaff dynamically suppresses outliers exclusively in invariant channels using lightweight operations, eliminating full-precision weight storage and global rescaling while reducing quantization errors. Extensive experiments across ten benchmarks validate OSSH and demonstrate Quaff's efficacy. Specifically, on the GPQA reasoning benchmark, Quaff achieves a 1.73x latency reduction and 30% memory savings over full-precision fine-tuning while improving accuracy by 0.6% on the Phi-3 model, reconciling the triple trade-off between efficiency, performance, and deployability. By enabling consumer-grade GPU fine-tuning (e.g., RTX 2080 Super) without sacrificing model utility, Quaff democratizes personalized LLM deployment. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域取得了令人兴奋的成就，但在资源受限的个人设备上的部署仍受制于特定任务微调的高额计算和内存需求。虽然量化为提高效率提供了一条途径，但现有的方法难以在性能和开销之间取得平衡，要么导致高昂的计算/内存成本，要么无法解决量化微调中的激活异常值问题，这是量化的关键瓶颈之一。为了解决这些挑战，我们提出了异常值空间稳定性假设（OSSH）：在微调过程中，某些激活异常值通道在训练迭代中保持稳定的空间位置。基于OSSH，我们提出了Quaff，一种针对LLMs的量化参数高效微调框架，通过目标化的动量缩放优化低精度激活表示。Quaff仅在不变通道中动态抑制异常值，使用轻量级操作消除全精度权重存储和全局缩放，同时减少量化误差。在十个基准测试的广泛实验验证了OSSH，并展示了Quaff的有效性。特别是在GPQA推理基准测试中，Quaff在Phi-3模型上准确率提高0.6%的情况下，实现了1.73倍的延迟减少和30%的内存节省，解决了效率、性能和部署性之间的三重权衡问题。通过使消费者级GPU微调（例如，RTX 2080 Super）能够在不牺牲模型实用性的情况下成为可能，Quaff民主化了个性化LLM的部署。代码可在以下网址获取。 

---
# Communication-Efficient Diffusion Denoising Parallelization via Reuse-then-Predict Mechanism 

**Title (ZH)**: 通信高效的利用重用-预测机制的扩散去噪并行化 

**Authors**: Kunyun Wang, Bohan Li, Kai Yu, Minyi Guo, Jieru Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14741)  

**Abstract**: Diffusion models have emerged as a powerful class of generative models across various modalities, including image, video, and audio synthesis. However, their deployment is often limited by significant inference latency, primarily due to the inherently sequential nature of the denoising process. While existing parallelization strategies attempt to accelerate inference by distributing computation across multiple devices, they typically incur high communication overhead, hindering deployment on commercial hardware. To address this challenge, we propose \textbf{ParaStep}, a novel parallelization method based on a reuse-then-predict mechanism that parallelizes diffusion inference by exploiting similarity between adjacent denoising steps. Unlike prior approaches that rely on layer-wise or stage-wise communication, ParaStep employs lightweight, step-wise communication, substantially reducing overhead. ParaStep achieves end-to-end speedups of up to \textbf{3.88}$\times$ on SVD, \textbf{2.43}$\times$ on CogVideoX-2b, and \textbf{6.56}$\times$ on AudioLDM2-large, while maintaining generation quality. These results highlight ParaStep as a scalable and communication-efficient solution for accelerating diffusion inference, particularly in bandwidth-constrained environments. 

**Abstract (ZH)**: 基于reuse-then-predict机制的ParaStep：一种高效的扩散模型并行化方法 

---
# Time Series Similarity Score Functions to Monitor and Interact with the Training and Denoising Process of a Time Series Diffusion Model applied to a Human Activity Recognition Dataset based on IMUs 

**Title (ZH)**: 基于IMU的人体活动识别数据集上时间序列扩散模型的训练与去噪过程监测及交互的时间序列相似性评分函数 

**Authors**: Heiko Oppel, Andreas Spilz, Michael Munz  

**Link**: [PDF](https://arxiv.org/pdf/2505.14739)  

**Abstract**: Denoising diffusion probabilistic models are able to generate synthetic sensor signals. The training process of such a model is controlled by a loss function which measures the difference between the noise that was added in the forward process and the noise that was predicted by the diffusion model. This enables the generation of realistic data. However, the randomness within the process and the loss function itself makes it difficult to estimate the quality of the data. Therefore, we examine multiple similarity metrics and adapt an existing metric to overcome this issue by monitoring the training and synthetisation process using those metrics. The adapted metric can even be fine-tuned on the input data to comply with the requirements of an underlying classification task. We were able to significantly reduce the amount of training epochs without a performance reduction in the classification task. An optimized training process not only saves resources, but also reduces the time for training generative models. 

**Abstract (ZH)**: 去噪扩散概率模型能够生成合成传感器信号。这类模型的训练过程由衡量正向过程添加噪声与扩散模型预测噪声之间差异的损失函数控制。这使得生成现实数据成为可能。然而，此过程中以及损失函数本身的随机性使得难以估计数据质量。因此，我们检查了多种相似性度量，并对现有度量进行了调整，通过使用这些度量监控训练和合成过程来克服这一问题。调整后的度量甚至可以根据输入数据进行微调，以符合底层分类任务的要求。我们能够在不降低分类任务性能的情况下显著减少训练周期。优化的训练过程不仅节约资源，还缩短了生成模型的训练时间。 

---
# Leveraging Multivariate Long-Term History Representation for Time Series Forecasting 

**Title (ZH)**: 利用多变量长期历史表示进行时间序列预测 

**Authors**: Huiliang Zhang, Di Wu, Arnaud Zinflou, Stephane Dellacherie, Mouhamadou Makhtar Dione, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2505.14737)  

**Abstract**: Multivariate Time Series (MTS) forecasting has a wide range of applications in both industry and academia. Recent advances in Spatial-Temporal Graph Neural Network (STGNN) have achieved great progress in modelling spatial-temporal correlations. Limited by computational complexity, most STGNNs for MTS forecasting focus primarily on short-term and local spatial-temporal dependencies. Although some recent methods attempt to incorporate univariate history into modeling, they still overlook crucial long-term spatial-temporal similarities and correlations across MTS, which are essential for accurate forecasting. To fill this gap, we propose a framework called the Long-term Multivariate History Representation (LMHR) Enhanced STGNN for MTS forecasting. Specifically, a Long-term History Encoder (LHEncoder) is adopted to effectively encode the long-term history into segment-level contextual representations and reduce point-level noise. A non-parametric Hierarchical Representation Retriever (HRetriever) is designed to include the spatial information in the long-term spatial-temporal dependency modelling and pick out the most valuable representations with no additional training. A Transformer-based Aggregator (TAggregator) selectively fuses the sparsely retrieved contextual representations based on the ranking positional embedding efficiently. Experimental results demonstrate that LMHR outperforms typical STGNNs by 10.72% on the average prediction horizons and state-of-the-art methods by 4.12% on several real-world datasets. Additionally, it consistently improves prediction accuracy by 9.8% on the top 10% of rapidly changing patterns across the datasets. 

**Abstract (ZH)**: 多变量时间序列（MTS） forecasting在工业和学术界有着广泛的应用。 recent advancements in 空间-时间图神经网络（STGNN）已在建模空间-时间相关性方面取得了显著进展。由于计算复杂性的限制，大多数应用于MTS forecasting的STGNN主要关注短期和局部空间-时间依赖性。尽管一些最近的方法试图将一变量的历史信息纳入建模中，但它们仍然忽略了跨MTS中的关键长期空间-时间相似性和相关性，这对于准确的预测至关重要。为了填补这一空白，我们提出了一种称为长期内存多变量历史表示（LMHR）增强STGNN的框架。具体而言，采用了一种长期内存编码器（LHEncoder）有效地将长期内存编码为段级上下文表示并减少点级噪声。设计了一种非参数层次表示检索器（HRetriever），包括空间信息以建模长期空间-时间依赖性，并挑选出最具价值的表示而不需额外训练。一种基于排名位置嵌入的Transformer聚合器（TAggregator）有效地选择性地融合稀疏检索的上下文表示。实验结果表明，LMHR在平均预测水平上的表现比典型STGNN高10.72%，在某些现实世界数据集上的表现比最先进的方法高4.12%。此外，它在数据集中的前10%快速变化模式上的一致改善了9.8%的预测准确性。 

---
# The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute 

**Title (ZH)**: 推理的能源成本：分析LLM测试时计算能耗 

**Authors**: Yunho Jin, Gu-Yeon Wei, David Brooks  

**Link**: [PDF](https://arxiv.org/pdf/2505.14733)  

**Abstract**: Scaling large language models (LLMs) has driven significant advancements, yet it faces diminishing returns and escalating energy demands. This work introduces test-time compute (TTC)-allocating additional computational resources during inference-as a compelling complement to conventional scaling strategies. Specifically, we investigate whether employing TTC can achieve superior accuracy-energy trade-offs compared to simply increasing model size. Our empirical analysis reveals that TTC surpasses traditional model scaling in accuracy/energy efficiency, with notable gains in tasks demanding complex reasoning rather than mere factual recall. Further, we identify a critical interaction between TTC performance and output sequence length, demonstrating that strategically adjusting compute resources at inference time according to query complexity can substantially enhance efficiency. Our findings advocate for TTC as a promising direction, enabling more sustainable, accurate, and adaptable deployment of future language models without incurring additional pretraining costs. 

**Abstract (ZH)**: 扩大小型语言模型的计算（TTC）分配额外计算资源以进行推断作为一种传统扩展策略的有力补充：探究其在准确率-能耗trade-off上的优越性 

---
# MORALISE: A Structured Benchmark for Moral Alignment in Visual Language Models 

**Title (ZH)**: MORALISE：视觉语言模型道德对齐的结构化基准 

**Authors**: Xiao Lin, Zhining Liu, Ze Yang, Gaotang Li, Ruizhong Qiu, Shuke Wang, Hui Liu, Haotian Li, Sumit Keswani, Vishwa Pardeshi, Huijun Zhao, Wei Fan, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14728)  

**Abstract**: Warning: This paper contains examples of harmful language and images. Reader discretion is advised. Recently, vision-language models have demonstrated increasing influence in morally sensitive domains such as autonomous driving and medical analysis, owing to their powerful multimodal reasoning capabilities. As these models are deployed in high-stakes real-world applications, it is of paramount importance to ensure that their outputs align with human moral values and remain within moral boundaries. However, existing work on moral alignment either focuses solely on textual modalities or relies heavily on AI-generated images, leading to distributional biases and reduced realism. To overcome these limitations, we introduce MORALISE, a comprehensive benchmark for evaluating the moral alignment of vision-language models (VLMs) using diverse, expert-verified real-world data. We begin by proposing a comprehensive taxonomy of 13 moral topics grounded in Turiel's Domain Theory, spanning the personal, interpersonal, and societal moral domains encountered in everyday life. Built on this framework, we manually curate 2,481 high-quality image-text pairs, each annotated with two fine-grained labels: (1) topic annotation, identifying the violated moral topic(s), and (2) modality annotation, indicating whether the violation arises from the image or the text. For evaluation, we encompass two tasks, \textit{moral judgment} and \textit{moral norm attribution}, to assess models' awareness of moral violations and their reasoning ability on morally salient content. Extensive experiments on 19 popular open- and closed-source VLMs show that MORALISE poses a significant challenge, revealing persistent moral limitations in current state-of-the-art models. The full benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 警告：本论文包含有害语言和图像的示例。读者请谨慎阅读。近期，视觉-语言模型在道德敏感领域（如自动驾驶和医疗分析）的影响日益增强，这得益于它们强大的多模态推理能力。随着这些模型在高风险的实际应用中被部署，确保其输出符合人类道德价值观并在道德界限内至关重要。然而，现有的道德对齐工作要么仅关注文本模态，要么严重依赖于AI生成的图像，这导致了分布偏差并降低了现实感。为克服这些局限性，我们引入了MORALISE，一个使用多样且专家验证的真实世界数据评估视觉-语言模型道德对齐的全面基准。我们首先提出了一种基于Turiel的领域理论的全面道德主题分类，涵盖日常生活中的个人、人际关系和社会道德领域。基于这一框架，我们手工策划了2,481个高质量的图像-文本对，每个对都标记了两个细粒度标签：（1）主题注释，识别违反的道德主题；（2）模态注释，指出违反来自图像还是文本。评估任务包括道德判断和道德规范归因，以评估模型对道德违规的意识及其在道德敏感内容上的推理能力。对19个流行的开源和闭源视觉-语言模型进行的广泛实验表明，MORALISE提出了重大挑战，揭示了当前最先进的模型中存在的持续道德限制。完整基准可在以下链接获取：this https URL。 

---
# MedBLIP: Fine-tuning BLIP for Medical Image Captioning 

**Title (ZH)**: MedBLIP: 细化BLIP以用于医学图像标注 

**Authors**: Manshi Limbu, Diwita Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14726)  

**Abstract**: Medical image captioning is a challenging task that requires generating clinically accurate and semantically meaningful descriptions of radiology images. While recent vision-language models (VLMs) such as BLIP, BLIP2, Gemini and ViT-GPT2 show strong performance on natural image datasets, they often produce generic or imprecise captions when applied to specialized medical domains. In this project, we explore the effectiveness of fine-tuning the BLIP model on the ROCO dataset for improved radiology captioning. We compare the fine-tuned BLIP against its zero-shot version, BLIP-2 base, BLIP-2 Instruct and a ViT-GPT2 transformer baseline. Our results demonstrate that domain-specific fine-tuning on BLIP significantly improves performance across both quantitative and qualitative evaluation metrics. We also visualize decoder cross-attention maps to assess interpretability and conduct an ablation study to evaluate the contributions of encoder-only and decoder-only fine-tuning. Our findings highlight the importance of targeted adaptation for medical applications and suggest that decoder-only fine-tuning (encoder-frozen) offers a strong performance baseline with 5% lower training time than full fine-tuning, while full model fine-tuning still yields the best results overall. 

**Abstract (ZH)**: 医学图像 captioning 是一项具有挑战性的任务，要求生成临床准确且语义有意义的放射学图像描述。尽管最近的视觉-语言模型（VLMs）如 BLIP、BLIP2、Gemini 和 ViT-GPT2 在自然图像数据集上表现出强大的性能，但当应用于专业医疗领域时，它们往往会产生通用或不精确的描述。在本项目中，我们研究了在 ROCO 数据集上微调 BLIP 模型以提高放射学 captioning 的有效性。我们将微调后的 BLIP 与零样本版本 BLIP-2 base、BLIP-2 Instruct 以及 ViT-GPT2 变体基线进行比较。我们的结果表明，针对领域进行的 BLIP 微调在定量和定性评估指标上均显著提升性能。我们还可视化了解码器交叉注意力图以评估可解释性，并进行了消融研究以评估仅编码器和仅解码器微调的贡献。我们的发现强调了针对医疗应用的目标适应的重要性，并表明仅解码器微调（编码器冻结）提供了较低训练时间（降低5%）的强基准性能，而整体模型微调仍然在总体上提供最佳结果。 

---
# QUADS: QUAntized Distillation Framework for Efficient Speech Language Understanding 

**Title (ZH)**: QUADS：量化精简框架以实现高效语音语言理解 

**Authors**: Subrata Biswas, Mohammad Nur Hossain Khan, Bashima Islam  

**Link**: [PDF](https://arxiv.org/pdf/2505.14723)  

**Abstract**: Spoken Language Understanding (SLU) systems must balance performance and efficiency, particularly in resource-constrained environments. Existing methods apply distillation and quantization separately, leading to suboptimal compression as distillation ignores quantization constraints. We propose QUADS, a unified framework that optimizes both through multi-stage training with a pre-tuned model, enhancing adaptability to low-bit regimes while maintaining accuracy. QUADS achieves 71.13\% accuracy on SLURP and 99.20\% on FSC, with only minor degradations of up to 5.56\% compared to state-of-the-art models. Additionally, it reduces computational complexity by 60--73$\times$ (GMACs) and model size by 83--700$\times$, demonstrating strong robustness under extreme quantization. These results establish QUADS as a highly efficient solution for real-world, resource-constrained SLU applications. 

**Abstract (ZH)**: 基于资源约束环境下的口语理解（SLU）系统必须在性能和效率之间取得平衡。现有方法分别应用蒸馏和量化，导致次优压缩效果，因为蒸馏忽略了量化约束。我们提出QUADS，这是一种统一框架，通过多阶段训练和预调模型优化两者，增强在低比特率环境下的适应性同时保持准确性。QUADS在SLURP上的准确率为71.13%，在FSC上的准确率为99.20%，与最先进的模型相比，仅出现最高5.56%的轻微性能下降。此外，QUADS计算复杂度降低了60-73倍（GMACs），模型大小减少了83-700倍，显示出在极端量化下的强大稳健性。这些结果确立了QUADS作为资源约束环境下实际应用中高效解决方案的地位。 

---
# MSVIT: Improving Spiking Vision Transformer Using Multi-scale Attention Fusion 

**Title (ZH)**: MSVIT：使用多尺度注意力融合改进脉冲视觉变换器 

**Authors**: Wei Hua, Chenlin Zhou, Jibin Wu, Yansong Chua, Yangyang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14719)  

**Abstract**: The combination of Spiking Neural Networks(SNNs) with Vision Transformer architectures has attracted significant attention due to the great potential for energy-efficient and high-performance computing paradigms. However, a substantial performance gap still exists between SNN-based and ANN-based transformer architectures. While existing methods propose spiking self-attention mechanisms that are successfully combined with SNNs, the overall architectures proposed by these methods suffer from a bottleneck in effectively extracting features from different image scales. In this paper, we address this issue and propose MSVIT, a novel spike-driven Transformer architecture, which firstly uses multi-scale spiking attention (MSSA) to enrich the capability of spiking attention blocks. We validate our approach across various main data sets. The experimental results show that MSVIT outperforms existing SNN-based models, positioning itself as a state-of-the-art solution among SNN-transformer architectures. The codes are available at this https URL. 

**Abstract (ZH)**: 基于Sparking Neural Networks与Vision Transformer架构的结合因其实现高效能计算的 potential 而受到广泛关注，然而基于SNN和基于ANN的Transformer架构之间仍然存在显著的性能差距。尽管现有方法提出了一种成功的spiking自注意机制与其结合，但这些方法的整体架构仍然在从不同图像尺度中有效提取特征方面存在瓶颈。本文解决了这一问题，提出了一种新型的spike驱动Transformer架构MSVIT，该架构首次使用多尺度spiking注意力（MSSA）以丰富spiking注意力模块的能力。我们在多种主要数据集上验证了该方法。实验结果表明，MSVIT 在基于SNN的模型中表现出色，是SNN-Transformer架构中的先进解决方案。代码见此链接。 

---
# Enhancing Shape Perception and Segmentation Consistency for Industrial Image Inspection 

**Title (ZH)**: 增强工业图像检测中的形状感知和分割一致性 

**Authors**: Guoxuan Mao, Ting Cao, Ziyang Li, Yuan Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14718)  

**Abstract**: Semantic segmentation stands as a pivotal research focus in computer vision. In the context of industrial image inspection, conventional semantic segmentation models fail to maintain the segmentation consistency of fixed components across varying contextual environments due to a lack of perception of object contours. Given the real-time constraints and limited computing capability of industrial image detection machines, it is also necessary to create efficient models to reduce computational complexity. In this work, a Shape-Aware Efficient Network (SPENet) is proposed, which focuses on the shapes of objects to achieve excellent segmentation consistency by separately supervising the extraction of boundary and body information from images. In SPENet, a novel method is introduced for describing fuzzy boundaries to better adapt to real-world scenarios named Variable Boundary Domain (VBD). Additionally, a new metric, Consistency Mean Square Error(CMSE), is proposed to measure segmentation consistency for fixed components. Our approach attains the best segmentation accuracy and competitive speed on our dataset, showcasing significant advantages in CMSE among numerous state-of-the-art real-time segmentation networks, achieving a reduction of over 50% compared to the previously top-performing models. 

**Abstract (ZH)**: 面向工业图像检测的形状感知高效网络（SPENet）研究：实现固定组件的一致性分割 

---
# Aneumo: A Large-Scale Multimodal Aneurysm Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks 

**Title (ZH)**: Aneumo：一个包含计算流体动力学模拟和深度学习基准的大规模多模态动脉瘤数据集 

**Authors**: Xigui Li, Yuanye Zhou, Feiyang Xiao, Xin Guo, Chen Jiang, Tan Pan, Xingmeng Zhang, Cenyu Liu, Zeyun Miao, Jianchao Ge, Xiansheng Wang, Qimeng Wang, Yichi Zhang, Wenbo Zhang, Fengping Zhu, Limei Han, Yuan Qi, Chensen Lin, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14717)  

**Abstract**: Intracranial aneurysms (IAs) are serious cerebrovascular lesions found in approximately 5\% of the general population. Their rupture may lead to high mortality. Current methods for assessing IA risk focus on morphological and patient-specific factors, but the hemodynamic influences on IA development and rupture remain unclear. While accurate for hemodynamic studies, conventional computational fluid dynamics (CFD) methods are computationally intensive, hindering their deployment in large-scale or real-time clinical applications. To address this challenge, we curated a large-scale, high-fidelity aneurysm CFD dataset to facilitate the development of efficient machine learning algorithms for such applications. Based on 427 real aneurysm geometries, we synthesized 10,660 3D shapes via controlled deformation to simulate aneurysm evolution. The authenticity of these synthetic shapes was confirmed by neurosurgeons. CFD computations were performed on each shape under eight steady-state mass flow conditions, generating a total of 85,280 blood flow dynamics data covering key parameters. Furthermore, the dataset includes segmentation masks, which can support tasks that use images, point clouds or other multimodal data as input. Additionally, we introduced a benchmark for estimating flow parameters to assess current modeling methods. This dataset aims to advance aneurysm research and promote data-driven approaches in biofluids, biomedical engineering, and clinical risk assessment. The code and dataset are available at: this https URL. 

**Abstract (ZH)**: 颅内动脉瘤（IAs）是大约5%普通人群中发现的严重脑血管病变。它们的破裂可能导致高死亡率。当前评估IA风险的方法主要集中在形态学和个体因素上，但血流动力学对IA发展和破裂的影响仍不明确。尽管在血流动力学研究中是准确的，但传统计算流体动力学（CFD）方法计算量大，阻碍了其在大规模或实时临床应用中的部署。为应对这一挑战，我们整理了一个大规模、高保真动脉瘤CFD数据集，以促进此类应用中高效机器学习算法的发展。基于427个真实动脉瘤几何结构，我们通过受控变形合成了10,660个3D形状，以模拟动脉瘤演化。神经外科医生确认了这些合成形状的真实性。在八种稳态质量流条件下对每个形状进行了CFD计算，生成了涵盖关键参数的85,280个血液流动动态数据。此外，该数据集还包括分割掩码，可以支持使用图像、点云或其他多模态数据作为输入的任务。此外，我们引入了一种评估当前建模方法的基准，用于估计流参数。该数据集旨在推进动脉瘤研究，并促进生物流体、生物医学工程和临床风险评估中的数据驱动方法。代码和数据集可在以下链接获取：this https URL。 

---
# KGAlign: Joint Semantic-Structural Knowledge Encoding for Multimodal Fake News Detection 

**Title (ZH)**: KGAlign：联合语义-结构知识编码的多模态假新闻检测 

**Authors**: Tuan-Vinh La, Minh-Hieu Nguyen, Minh-Son Dao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14714)  

**Abstract**: Fake news detection remains a challenging problem due to the complex interplay between textual misinformation, manipulated images, and external knowledge reasoning. While existing approaches have achieved notable results in verifying veracity and cross-modal consistency, two key challenges persist: (1) Existing methods often consider only the global image context while neglecting local object-level details, and (2) they fail to incorporate external knowledge and entity relationships for deeper semantic understanding. To address these challenges, we propose a novel multi-modal fake news detection framework that integrates visual, textual, and knowledge-based representations. Our approach leverages bottom-up attention to capture fine-grained object details, CLIP for global image semantics, and RoBERTa for context-aware text encoding. We further enhance knowledge utilization by retrieving and adaptively selecting relevant entities from a knowledge graph. The fused multi-modal features are processed through a Transformer-based classifier to predict news veracity. Experimental results demonstrate that our model outperforms recent approaches, showcasing the effectiveness of neighbor selection mechanism and multi-modal fusion for fake news detection. Our proposal introduces a new paradigm: knowledge-grounded multimodal reasoning. By integrating explicit entity-level selection and NLI-guided filtering, we shift fake news detection from feature fusion to semantically grounded verification. For reproducibility and further research, the source code is publicly at \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 虚假新闻检测仍然是一个具有挑战性的问题，由于文本误导信息、操纵图像和外部知识推理之间的复杂相互作用。虽然现有的方法在验证真实性和多模态一致性方面已取得显著成果，但仍然存在两个关键挑战：（1）现有方法通常只考虑全局图像上下文，而忽略局部对象级别的细节；（2）它们未能整合外部知识和实体关系以实现更深入的意义理解。为了解决这些挑战，我们提出了一种新的多模态虚假新闻检测框架，结合了视觉、文本和知识基表示。我们的方法利用自底向上的注意力机制捕捉细微的对象细节，使用CLIP提取全局图像语义，并使用RoBERTa进行上下文感知文本编码。为进一步利用知识，我们通过检索知识图谱中相关实体并进行适应性选择来增强知识的利用。融合的多模态特征通过基于变换器的分类器进行处理，以预测新闻真实性。实验结果表明，我们的模型优于近期的方法，证明了邻居选择机制和多模态融合在虚假新闻检测中的有效性。我们的提议引入了一个新的范式：基于知识的多模态推理。通过集成显式的实体级选择和基于自然语言推理的过滤机制，我们将虚假新闻检测从特征融合转向基于语义的验证。为了重现性与进一步研究，源代码公开于[this http URL]。 

---
# Space evaluation at the starting point of soccer transitions 

**Title (ZH)**: 足球转换起始点的空间评价 

**Authors**: Yohei Ogawa, Rikuhei Umemoto, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2505.14711)  

**Abstract**: Soccer is a sport played on a pitch where effective use of space is crucial. Decision-making during transitions, when possession switches between teams, has been increasingly important, but research on space evaluation in these moments has been limited. Recent space evaluation methods such as OBSO (Off-Ball Scoring Opportunity) use scoring probability, so it is not well-suited for assessing areas far from the goal, where transitions typically occur. In this paper, we propose OBPV (Off-Ball Positioning Value) to evaluate space across the pitch, including the starting points of transitions. OBPV extends OBSO by introducing the field value model, which evaluates the entire pitch, and by employing the transition kernel model, which reflects positional specificity through kernel density estimation of pass distributions. Experiments using La Liga 2023/24 season tracking and event data show that OBPV highlights effective space utilization during counter-attacks and reveals team-specific characteristics in how the teams utilize space after positive and negative transitions. 

**Abstract (ZH)**: 足球是一项在球场上进行的运动，有效利用空间至关重要。当控球权在两队之间转换时，转换期间的决策制定越来越重要，但有关这一时刻的空间评估研究相对有限。近年来的空间评估方法如OBSO（无球得分机会）依靠射门概率，因此不适用于评估远距离无球空间，而这些空间往往是转换发生的地点。本文提出了一种新的无球空间评估方法OBPV（无球定位价值），以评估整个球场上的空间，包括转换的起始点。OBPV通过引入场地方价值模型来评估整个球场，并通过引入转换内核模型，利用通过传递分布的内核密度估计来反映位置特定性。使用2023/24赛季西甲追踪和事件数据的实验表明，OBPV突出了反攻时有效空间利用，并揭示了不同团队在积极和消极转换后利用空间的特定特征。 

---
# FastCar: Cache Attentive Replay for Fast Auto-Regressive Video Generation on the Edge 

**Title (ZH)**: FastCar：基于缓存注意力的快速自回归视频生成边缘计算方法 

**Authors**: Xuan Shen, Weize Ma, Yufa Zhou, Enhao Tang, Yanyue Xie, Zhengang Li, Yifan Gong, Quanyi Wang, Henghui Ding, Yiwei Wang, Yanzhi Wang, Pu Zhao, Jun Lin, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14709)  

**Abstract**: Auto-regressive (AR) models, initially successful in language generation, have recently shown promise in visual generation tasks due to their superior sampling efficiency. Unlike image generation, video generation requires a substantially larger number of tokens to produce coherent temporal frames, resulting in significant overhead during the decoding phase. Our key observations are: (i) MLP modules in the decode phase dominate the inference latency, and (ii) there exists high temporal redundancy in MLP outputs of adjacent frames. In this paper, we propose the \textbf{FastCar} framework to accelerate the decode phase for the AR video generation by exploring the temporal redundancy. The Temporal Attention Score (TAS) is proposed to determine whether to apply the replay strategy (\textit{i.e.}, reusing cached MLP outputs from the previous frame to reduce redundant computations) with detailed theoretical analysis and justification. Also, we develop a hardware accelerator on FPGA with Dynamic Resource Scheduling (DRS) based on TAS to enable better resource utilization and faster inference. Experimental results demonstrate the effectiveness of our method, which outperforms traditional sparse attention approaches with more than 2.1x decoding speedup and higher energy efficiency on the edge. Furthermore, by combining FastCar and sparse attention, FastCar can boost the performance of sparse attention with alleviated drifting, demonstrating our unique advantages for high-resolution and long-duration video generation. Code: this https URL 

**Abstract (ZH)**: 自回归（AR）模型在语言生成任务中最初表现出色，近期在视觉生成任务中也展现出潜力，得益于其优越的采样效率。与图像生成不同，视频生成需要生成连贯的时序帧，这导致解码阶段产生显著的计算负担。我们的主要观察是：（i）解码阶段的MLP模块主导了推理延迟，（ii）相邻帧的MLP输出中存在较高的时序冗余。本文提出了FastCar框架，通过探索时序冗余来加速AR视频生成的解码阶段。我们提出了时序注意力得分（TAS），并结合详细的理论分析和证明来决定是否采用重播策略（即将前一帧缓存的MLP输出重用以减少冗余计算）。此外，我们通过基于TAS的动态资源调度（DRS）在FPGA上开发了硬件加速器，以实现更好的资源利用和更快的推理速度。实验结果表明，我们的方法在边缘设备上比传统的稀疏注意力方法具有超过2.1倍的解码加速和更高的能效。此外，FastCar与稀疏注意力结合使用时，可以缓解稀疏注意力的漂移问题，显著提升了高分辨率和长时间段视频生成的性能。代码: this https URL 

---
# DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance 

**Title (ZH)**: DraftAttention: 快速视频扩散通过低分辨率注意力引导 

**Authors**: Xuan Shen, Chenxia Han, Yufa Zhou, Yanyue Xie, Yifan Gong, Quanyi Wang, Yiwei Wang, Yanzhi Wang, Pu Zhao, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14708)  

**Abstract**: Diffusion transformer-based video generation models (DiTs) have recently attracted widespread attention for their excellent generation quality. However, their computational cost remains a major bottleneck-attention alone accounts for over 80% of total latency, and generating just 8 seconds of 720p video takes tens of minutes-posing serious challenges to practical application and scalability. To address this, we propose the DraftAttention, a training-free framework for the acceleration of video diffusion transformers with dynamic sparse attention on GPUs. We apply down-sampling to each feature map across frames in the compressed latent space, enabling a higher-level receptive field over the latent composed of hundreds of thousands of tokens. The low-resolution draft attention map, derived from draft query and key, exposes redundancy both spatially within each feature map and temporally across frames. We reorder the query, key, and value based on the draft attention map to guide the sparse attention computation in full resolution, and subsequently restore their original order after the attention computation. This reordering enables structured sparsity that aligns with hardware-optimized execution. Our theoretical analysis demonstrates that the low-resolution draft attention closely approximates the full attention, providing reliable guidance for constructing accurate sparse attention. Experimental results show that our method outperforms existing sparse attention approaches in video generation quality and achieves up to 1.75x end-to-end speedup on GPUs. Code: this https URL 

**Abstract (ZH)**: 基于扩散变换器的视频生成模型（DiTs）由于其出色的生成质量 recently 吸引了广泛关注。然而，其计算成本仍然是一个主要瓶颈——仅注意力机制就占去了超过 80% 的总延迟，生成 720p 视频的 8 秒内容需要几十分钟——这对实际应用和可扩展性构成了严重挑战。为了解决这个问题，我们提出了一种无训练框架 DraftAttention，该框架结合动态稀疏注意机制在 GPU 上加速视频扩散变换器。我们对压缩潜空间中每一帧的特征图进行下采样，从而在包含数十万个标记的潜空间中获得更高的感受野。由草图查询和键衍生出的低分辨率草图注意图在空间范围内显示了每个特征图内的冗余性，并在时间范围跨帧之间显示了冗余性。根据草图注意图重新排序查询、键和值，以引导全分辨率下的稀疏注意计算，在注意计算后恢复其原始顺序。这种重新排序能够与硬件优化的执行相一致，提供结构化稀疏性。我们的理论分析表明，低分辨率草图注意图高度近似于全注意，为构建准确的稀疏注意提供了可靠指导。实验结果表明，我们的方法在视频生成质量上优于现有稀疏注意方法，并在 GPU 上实现了最高达 1.75 倍的端到端加速。代码：这个 https URL。 

---
# CrypticBio: A Large Multimodal Dataset for Visually Confusing Biodiversity 

**Title (ZH)**: CrypticBio: 一个大型多模态数据集，用于视觉上难以区分的生物多样性 

**Authors**: Georgiana Manolache, Gerard Schouten, Joaquin Vanschoren  

**Link**: [PDF](https://arxiv.org/pdf/2505.14707)  

**Abstract**: We present CrypticBio, the largest publicly available multimodal dataset of visually confusing species, specifically curated to support the development of AI models in the context of biodiversity applications. Visually confusing or cryptic species are groups of two or more taxa that are nearly indistinguishable based on visual characteristics alone. While much existing work addresses taxonomic identification in a broad sense, datasets that directly address the morphological confusion of cryptic species are small, manually curated, and target only a single taxon. Thus, the challenge of identifying such subtle differences in a wide range of taxa remains unaddressed. Curated from real-world trends in species misidentification among community annotators of iNaturalist, CrypticBio contains 52K unique cryptic groups spanning 67K species, represented in 166 million images. Rich research-grade image annotations--including scientific, multicultural, and multilingual species terminology, hierarchical taxonomy, spatiotemporal context, and associated cryptic groups--address multimodal AI in biodiversity research. For easy dataset curation, we provide an open-source pipeline CrypticBio-Curate. The multimodal nature of the dataset beyond vision-language arises from the integration of geographical and temporal data as complementary cues to identifying cryptic species. To highlight the importance of the dataset, we benchmark a suite of state-of-the-art foundation models across CrypticBio subsets of common, unseen, endangered, and invasive species, and demonstrate the substantial impact of geographical context on vision-language zero-shot learning for cryptic species. By introducing CrypticBio, we aim to catalyze progress toward real-world-ready biodiversity AI models capable of handling the nuanced challenges of species ambiguity. 

**Abstract (ZH)**: CrypticBio：最大的公开多模态混淆物种数据集，专门支持生物多样性应用中的AI模型开发 

---
# Propositional Measure Logic 

**Title (ZH)**: 命题度量逻辑 

**Authors**: Francisco Aragão  

**Link**: [PDF](https://arxiv.org/pdf/2505.14693)  

**Abstract**: We present a propositional logic with fundamental probabilistic semantics, in which each formula is given a real measure in the interval $[0,1]$ that represents its degree of truth. This semantics replaces the binarity of classical logic, while preserving its deductive structure. We demonstrate the soundness theorem, establishing that the proposed system is sound and suitable for reasoning under uncertainty. We discuss potential applications and avenues for future extensions of the theory. We apply probabilistic logic to a still refractory problem in Bayesian Networks. 

**Abstract (ZH)**: 我们提出了一个具有基础概率语义的命题逻辑，在其中每个公式都被赋予一个在区间$[0,1]$上的实数测度，以表示其真度。这种语义替代了经典逻辑的二值性，同时保留了其推理结构。我们证明了_soundness定理_，确立了所提出系统的soundness并证明其适用于不确定性推理。我们讨论了该理论的应用潜力及其未来扩展的途径。我们将概率逻辑应用于贝叶斯网络中的一个仍具挑战性的问题。 

---
# THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering 

**Title (ZH)**: THELMA：基于任务的整体评估大语言模型应用-检索增强问答 

**Authors**: Udita Patel, Rutu Mulkar, Jay Roberts, Cibi Chakravarthy Senthilkumar, Sujay Gandhi, Xiaofei Zheng, Naumaan Nayyar, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11626)  

**Abstract**: We propose THELMA (Task Based Holistic Evaluation of Large Language Model Applications), a reference free framework for RAG (Retrieval Augmented generation) based question answering (QA) applications. THELMA consist of six interdependent metrics specifically designed for holistic, fine grained evaluation of RAG QA applications. THELMA framework helps developers and application owners evaluate, monitor and improve end to end RAG QA pipelines without requiring labelled sources or reference this http URL also present our findings on the interplay of the proposed THELMA metrics, which can be interpreted to identify the specific RAG component needing improvement in QA applications. 

**Abstract (ZH)**: 基于任务的全方位评估大语言模型应用的THELMA框架：面向RAG（检索增强生成）问答应用的无参考评估方法 

---
# Learning with Differentially Private (Sliced) Wasserstein Gradients 

**Title (ZH)**: 学习与差异隐私（切片）Wasserstein梯度相差规范 

**Authors**: David Rodríguez-Vítores, Clément Lalanne, Jean-Michel Loubes  

**Link**: [PDF](https://arxiv.org/pdf/2502.01701)  

**Abstract**: In this work, we introduce a novel framework for privately optimizing objectives that rely on Wasserstein distances between data-dependent empirical measures. Our main theoretical contribution is, based on an explicit formulation of the Wasserstein gradient in a fully discrete setting, a control on the sensitivity of this gradient to individual data points, allowing strong privacy guarantees at minimal utility cost. Building on these insights, we develop a deep learning approach that incorporates gradient and activations clipping, originally designed for DP training of problems with a finite-sum structure. We further demonstrate that privacy accounting methods extend to Wasserstein-based objectives, facilitating large-scale private training. Empirical results confirm that our framework effectively balances accuracy and privacy, offering a theoretically sound solution for privacy-preserving machine learning tasks relying on optimal transport distances such as Wasserstein distance or sliced-Wasserstein distance. 

**Abstract (ZH)**: 本工作中，我们引入了一种新的框架，用于在基于数据依赖的经验测度之间的Wasserstein距离优化目标时提供私密性。我们的主要理论贡献是在完全离散设置中明确表示Wasserstein梯度，并通过对单个数据点的敏感性控制，允许在最小 utility 成本下实现强大的隐私保证。基于这些认识，我们开发了一种深度学习方法，该方法结合了梯度和激活剪裁，最初是为具有有限和结构问题的DP训练设计的。我们还进一步证明，隐私计算方法可以扩展到基于Wasserstein距离的目标上，从而实现大规模的私密训练。实验证据证实，我们的框架有效地平衡了准确性和隐私性，为依赖最优运输距离（如Wasserstein距离或切片Wasserstein距离）的隐私保护机器学习任务提供了一个理论上有据可依的解决方案。 

---
