# Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists? 

**Title (ZH)**: 语言代理镜像人类因果推理偏见。我们如何帮助它们像科学家一样思考？ 

**Authors**: Anthony GX-Chen, Dongyan Lin, Mandana Samiei, Doina Precup, Blake A. Richards, Rob Fergus, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2505.09614)  

**Abstract**: Language model (LM) agents are increasingly used as autonomous decision-makers who need to actively gather information to guide their decisions. A crucial cognitive skill for such agents is the efficient exploration and understanding of the causal structure of the world -- key to robust, scientifically grounded reasoning. Yet, it remains unclear whether LMs possess this capability or exhibit systematic biases leading to erroneous conclusions. In this work, we examine LMs' ability to explore and infer causal relationships, using the well-established "Blicket Test" paradigm from developmental psychology. We find that LMs reliably infer the common, intuitive disjunctive causal relationships but systematically struggle with the unusual, yet equally (or sometimes even more) evidenced conjunctive ones. This "disjunctive bias" persists across model families, sizes, and prompting strategies, and performance further declines as task complexity increases. Interestingly, an analogous bias appears in human adults, suggesting that LMs may have inherited deep-seated reasoning heuristics from their training data. To this end, we quantify similarities between LMs and humans, finding that LMs exhibit adult-like inference profiles (but not children-like). Finally, we propose a test-time sampling method which explicitly samples and eliminates hypotheses about causal relationships from the LM. This scalable approach significantly reduces the disjunctive bias and moves LMs closer to the goal of scientific, causally rigorous reasoning. 

**Abstract (ZH)**: 语言模型（LM）代理日益用于担任自主决策者，需要主动搜集信息来指导其决策。此类代理高效探索和理解世界因果结构的关键认知技能是其进行稳健、科学依据充分推理的基础。然而，目前仍不清楚语言模型是否具备这种能力，还是倾向于表现出导致错误结论的系统性偏差。在本研究中，我们利用发展心理学中广泛认可的“Blicket 测试”范式，考察语言模型探索和推断因果关系的能力。我们发现，语言模型可靠地推断出常见的直观析取因果关系，但在处理异常但仍具有同样（或有时甚至更多）证据的联言因果关系时表现出系统性困难。这种“析取偏差”在不同模型家族、规模和提示策略中普遍存在，并且随着任务复杂性的增加，性能进一步下降。有趣的是，成年人类也表现出类似的偏差，这表明语言模型可能继承了其训练数据中的深层次推理启发式。为了解决这一问题，我们量化了语言模型与人类之间的相似性，发现语言模型显示出类似成年人的推理特征（但不是类似儿童的）。最后，我们提出了一种在测试时采样的方法，该方法明确地从语言模型中采样和排除关于因果关系的假设。这种可扩展的方法显著减少了析取偏差，使语言模型更接近科学、因果严谨推理的目标。 

---
# \textsc{rfPG}: Robust Finite-Memory Policy Gradients for Hidden-Model POMDPs 

**Title (ZH)**: RFPG: 隐模型部分可观测马尔可夫决策过程的稳健有限记忆策略梯度 

**Authors**: Maris F. L. Galesloot, Roman Andriushchenko, Milan Češka, Sebastian Junges, Nils Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09518)  

**Abstract**: Partially observable Markov decision processes (POMDPs) model specific environments in sequential decision-making under uncertainty. Critically, optimal policies for POMDPs may not be robust against perturbations in the environment. Hidden-model POMDPs (HM-POMDPs) capture sets of different environment models, that is, POMDPs with a shared action and observation space. The intuition is that the true model is hidden among a set of potential models, and it is unknown which model will be the environment at execution time. A policy is robust for a given HM-POMDP if it achieves sufficient performance for each of its POMDPs. We compute such robust policies by combining two orthogonal techniques: (1) a deductive formal verification technique that supports tractable robust policy evaluation by computing a worst-case POMDP within the HM-POMDP and (2) subgradient ascent to optimize the candidate policy for a worst-case POMDP. The empirical evaluation shows that, compared to various baselines, our approach (1) produces policies that are more robust and generalize better to unseen POMDPs and (2) scales to HM-POMDPs that consist of over a hundred thousand environments. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）模型在不确定性下的序列决策中建模特定环境。关键的是，POMDPs的最佳策略可能对环境扰动不够稳健。隐蔽模型POMDPs（HM-POMDPs）捕捉一组不同的环境模型，即具有共享动作和观测空间的POMDPs。直觉上，真实模型隐藏在一组潜在模型中，在执行时不知道哪个模型将是环境。针对给定HM-POMDP的一个策略是稳健的，如果它能够为HM-POMDP中的每个POMDP实现足够性能。我们通过结合两种正交技术来计算此类稳健策略：（1）支持通过计算HM-POMDP内的最坏情况POMDP来实现可管理的稳健策略评估的演绎形式化验证技术；（2）子梯度上升来优化候选策略针对最坏情况POMDP的性能。实证评估表明，与各种基线相比，我们的方法（1）生成的策略更加稳健，并且能够更好地泛化到未见过的POMDPs；（2）能够扩展到包含超过十万种环境的HM-POMDPs。 

---
# Counterfactual Strategies for Markov Decision Processes 

**Title (ZH)**: 马尔可夫决策过程的反事实策略 

**Authors**: Paul Kobialka, Lina Gerlach, Francesco Leofante, Erika Ábrahám, Silvia Lizeth Tapia Tarifa, Einar Broch Johnsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09412)  

**Abstract**: Counterfactuals are widely used in AI to explain how minimal changes to a model's input can lead to a different output. However, established methods for computing counterfactuals typically focus on one-step decision-making, and are not directly applicable to sequential decision-making tasks. This paper fills this gap by introducing counterfactual strategies for Markov Decision Processes (MDPs). During MDP execution, a strategy decides which of the enabled actions (with known probabilistic effects) to execute next. Given an initial strategy that reaches an undesired outcome with a probability above some limit, we identify minimal changes to the initial strategy to reduce that probability below the limit. We encode such counterfactual strategies as solutions to non-linear optimization problems, and further extend our encoding to synthesize diverse counterfactual strategies. We evaluate our approach on four real-world datasets and demonstrate its practical viability in sophisticated sequential decision-making tasks. 

**Abstract (ZH)**: Markov决策过程中的反事实策略：在Seqential决策任务中的应用 

---
# The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners 

**Title (ZH)**: 人类启发式能力在LLM驱动战略推理中的影响 

**Authors**: Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2505.09396)  

**Abstract**: The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation. 

**Abstract (ZH)**: 大型语言模型的迅速崛起促使人工智能研究转向自主系统，推动了更弱更灵活的自主概念的应用。然而，这种转变引发了关于基于大型语言模型的代理是否能复制人类战略推理能力的关键问题，尤其是在博弈论环境中。在此背景下，我们通过评估三种代理设计——一个简单的博弈论模型、一个无结构的大规模语言模型代理以及一个整合传统自主框架的大规模语言模型，探讨了自主复杂性在塑造人工推理性能中的作用。使用猜测游戏作为测试平台，我们评估了这些代理在一般推理模式和个体角色目标方面的表现，并引入了混淆的博弈场景来评估代理超越训练分布的泛化能力。我们的分析涵盖了25种代理配置超过2000个推理样本，结果显示，灵感来源于人类的认知结构可以增强大规模语言模型代理与人类战略行为的一致性。然而，自主设计复杂性与人类相似性的关系是非线性的，突显了其对底层大规模语言模型能力的强烈依赖，并表明简单的架构增强有限。 

---
# Access Controls Will Solve the Dual-Use Dilemma 

**Title (ZH)**: 访问控制将解决双重用途困境 

**Authors**: Evžen Wybitul  

**Link**: [PDF](https://arxiv.org/pdf/2505.09341)  

**Abstract**: AI safety systems face a dual-use dilemma. Since the same request can be either harmless or harmful depending on who made it and why, if the system makes decisions based solely on the request's content, it will refuse some legitimate queries and let pass harmful ones. To address this, we propose a conceptual access control framework, based on verified user credentials (such as institutional affiliation) and classifiers that assign model outputs to risk categories (such as advanced virology). The system permits responses only when the user's verified credentials match the category's requirements. For implementation of the model output classifiers, we introduce a theoretical approach utilizing small, gated expert modules integrated into the generator model, trained with gradient routing, that enable efficient risk detection without the capability gap problems of external monitors. While open questions remain about the verification mechanisms, risk categories, and the technical implementation, our framework makes the first step toward enabling granular governance of AI capabilities: verified users gain access to specialized knowledge without arbitrary restrictions, while adversaries are blocked from it. This contextual approach reconciles model utility with robust safety, addressing the dual-use dilemma. 

**Abstract (ZH)**: AI安全系统面临双重用途困境。为了应对这一挑战，我们提出一个基于验证用户凭据（如机构隶属关系）和分类器的风险类别框架。该框架将仅在用户验证凭据符合类别要求时才允许响应。在实现模型输出分类器时，我们引入了一种理论方法，利用集成于生成模型的小型门控专家模块，并通过梯度路由进行训练，这使得在避免外部监控能力缺口问题的同时，能够高效地进行风险检测。虽然关于验证机制、风险类别和技术实现仍有许多未解决的问题，但我们的框架迈出了走向细粒度治理AI能力的第一步：验证用户可以访问专业知识而不会受到任意限制，而对手则被阻止访问。这种上下文方法使模型 utility 与强健安全相一致，解决了双重用途困境。 

---
# Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents" 

**Title (ZH)**: "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents"的再现性研究 

**Authors**: Pedro M. P. Curvo, Mara Dragomir, Salvador Torpes, Mohammadmahdi Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09289)  

**Abstract**: This study evaluates and extends the findings made by Piatti et al., who introduced GovSim, a simulation framework designed to assess the cooperative decision-making capabilities of large language models (LLMs) in resource-sharing scenarios. By replicating key experiments, we validate claims regarding the performance of large models, such as GPT-4-turbo, compared to smaller models. The impact of the universalization principle is also examined, with results showing that large models can achieve sustainable cooperation, with or without the principle, while smaller models fail without it. In addition, we provide multiple extensions to explore the applicability of the framework to new settings. We evaluate additional models, such as DeepSeek-V3 and GPT-4o-mini, to test whether cooperative behavior generalizes across different architectures and model sizes. Furthermore, we introduce new settings: we create a heterogeneous multi-agent environment, study a scenario using Japanese instructions, and explore an "inverse environment" where agents must cooperate to mitigate harmful resource distributions. Our results confirm that the benchmark can be applied to new models, scenarios, and languages, offering valuable insights into the adaptability of LLMs in complex cooperative tasks. Moreover, the experiment involving heterogeneous multi-agent systems demonstrates that high-performing models can influence lower-performing ones to adopt similar behaviors. This finding has significant implications for other agent-based applications, potentially enabling more efficient use of computational resources and contributing to the development of more effective cooperative AI systems. 

**Abstract (ZH)**: 本研究评估并扩展了Piatti等人的研究成果，他们提出了GovSim，一种用于评估大型语言模型（LLMs）在资源分享场景中合作决策能力的仿真框架。通过复制关键实验，我们验证了关于大型模型（如GPT-4-turbo）与较小模型相比的表现声称。还研究了普遍化原则的影响，结果显示，大型模型可以在有或没有该原则的情况下实现可持续合作，而较小模型则需要该原则才能合作。此外，我们提供了多项扩展，以探讨该框架在新环境中的适用性。我们评估了其他模型（如DeepSeek-V3和GPT-4o-mini），以测试合作行为是否能跨不同架构和模型规模泛化。此外，我们引入了新环境：创建了一个异质多代理环境，研究了一个使用日语指令的场景，并探讨了一个“逆环境”，其中代理必须合作以缓解有害资源分配。研究结果证实，基准可以应用于新模型、场景和语言，为大型语言模型在复杂合作任务中的适应性提供了有价值的见解。此外，涉及异质多代理系统的实验表明，高性能模型可以影响低性能模型，使其采用类似行为。这一发现对其他基于代理的应用具有重要意义，可能有助于更有效地利用计算资源，并促进更有效的合作AI系统的发展。 

---
# Beyond the Known: Decision Making with Counterfactual Reasoning Decision Transformer 

**Title (ZH)**: 已知之外：基于反事实推理的决策制定 Decision Transformer 

**Authors**: Minh Hoang Nguyen, Linh Le Pham Van, Thommen George Karimpanal, Sunil Gupta, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.09114)  

**Abstract**: Decision Transformers (DT) play a crucial role in modern reinforcement learning, leveraging offline datasets to achieve impressive results across various domains. However, DT requires high-quality, comprehensive data to perform optimally. In real-world applications, the lack of training data and the scarcity of optimal behaviours make training on offline datasets challenging, as suboptimal data can hinder performance. To address this, we propose the Counterfactual Reasoning Decision Transformer (CRDT), a novel framework inspired by counterfactual reasoning. CRDT enhances DT ability to reason beyond known data by generating and utilizing counterfactual experiences, enabling improved decision-making in unseen scenarios. Experiments across Atari and D4RL benchmarks, including scenarios with limited data and altered dynamics, demonstrate that CRDT outperforms conventional DT approaches. Additionally, reasoning counterfactually allows the DT agent to obtain stitching abilities, combining suboptimal trajectories, without architectural modifications. These results highlight the potential of counterfactual reasoning to enhance reinforcement learning agents' performance and generalization capabilities. 

**Abstract (ZH)**: 基于反事实推理的决策变换器（CRDT）在现代强化学习中扮演着重要角色，通过利用离线数据集在多个领域取得了显著成果。然而，CRDT需要高质量、全面的数据才能表现优异。在实际应用中，由于缺乏训练数据和最优行为的稀缺性，利用离线数据集进行训练极具挑战性，不理想的训练数据会阻碍性能提升。为解决这一问题，我们提出了基于反事实推理的决策变换器（CRDT），这是一种受到反事实推理启发的新框架。CRDT通过生成和利用反事实经验，增强了DT在未知场景中进行推理和决策的能力。实验结果表明，CRDT在Atari和D4RL基准测试中，特别是在数据有限和动态改变的情况下，表现优于传统DT方法。此外，通过反事实推理，DT代理可以获取拼接能力，无需对架构进行修改即可结合不理想的轨迹。这些结果突显了反事实推理在提高强化学习代理性能和泛化能力方面的潜在价值。 

---
# Improving the Reliability of LLMs: Combining CoT, RAG, Self-Consistency, and Self-Verification 

**Title (ZH)**: 提高大语言模型可靠性的方法：结合论理思维、基于检索的生成、自我一致性与自我验证 

**Authors**: Adarsh Kumar, Hwiyoon Kim, Jawahar Sai Nathani, Neil Roy  

**Link**: [PDF](https://arxiv.org/pdf/2505.09031)  

**Abstract**: Hallucination, where large language models (LLMs) generate confident but incorrect or irrelevant information, remains a key limitation in their application to complex, open-ended tasks. Chain-of-thought (CoT) prompting has emerged as a promising method for improving multistep reasoning by guiding models through intermediate steps. However, CoT alone does not fully address the hallucination problem. In this work, we investigate how combining CoT with retrieval-augmented generation (RAG), as well as applying self-consistency and self-verification strategies, can reduce hallucinations and improve factual accuracy. By incorporating external knowledge sources during reasoning and enabling models to verify or revise their own outputs, we aim to generate more accurate and coherent responses. We present a comparative evaluation of baseline LLMs against CoT, CoT+RAG, self-consistency, and self-verification techniques. Our results highlight the effectiveness of each method and identify the most robust approach for minimizing hallucinations while preserving fluency and reasoning depth. 

**Abstract (ZH)**: 大语言模型（LLMs）生成的幻觉问题仍是其在处理复杂、开放式任务时的关键局限。思维链（CoT）提示作为一种改进多步推理的方法逐渐显示出潜力，通过引导模型完成中间步骤。然而，单独使用CoT并不能完全解决幻觉问题。在本研究中，我们探讨了将CoT与检索增强生成（RAG）相结合，以及应用自一致性与自我验证策略，如何减少幻觉并提高事实准确性。通过在推理过程中引入外部知识源，并使模型能够验证或修订自己的输出，我们旨在生成更准确和连贯的回应。我们对基线LLMs、CoT、CoT+RAG、自一致性及自我验证技术进行了比较评估。我们的结果突出了每种方法的有效性，并确定了在保留流畅性和推理深度的同时最大限度减少幻觉的最稳健方法。 

---
# Monte Carlo Beam Search for Actor-Critic Reinforcement Learning in Continuous Control 

**Title (ZH)**: 基于蒙特卡洛束搜索的Actor-Critic强化学习在连续控制中的应用 

**Authors**: Hazim Alzorgan, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09029)  

**Abstract**: Actor-critic methods, like Twin Delayed Deep Deterministic Policy Gradient (TD3), depend on basic noise-based exploration, which can result in less than optimal policy convergence. In this study, we introduce Monte Carlo Beam Search (MCBS), a new hybrid method that combines beam search and Monte Carlo rollouts with TD3 to improve exploration and action selection. MCBS produces several candidate actions around the policy's output and assesses them through short-horizon rollouts, enabling the agent to make better-informed choices. We test MCBS across various continuous-control benchmarks, including HalfCheetah-v4, Walker2d-v5, and Swimmer-v5, showing enhanced sample efficiency and performance compared to standard TD3 and other baseline methods like SAC, PPO, and A2C. Our findings emphasize MCBS's capability to enhance policy learning through structured look-ahead search while ensuring computational efficiency. Additionally, we offer a detailed analysis of crucial hyperparameters, such as beam width and rollout depth, and explore adaptive strategies to optimize MCBS for complex control tasks. Our method shows a higher convergence rate across different environments compared to TD3, SAC, PPO, and A2C. For instance, we achieved 90% of the maximum achievable reward within around 200 thousand timesteps compared to 400 thousand timesteps for the second-best method. 

**Abstract (ZH)**: 基于蒙特卡罗束搜索的孪生延迟深度确定性策略梯度方法：提高探索和行动选择效率 

---
# Automated Meta Prompt Engineering for Alignment with the Theory of Mind 

**Title (ZH)**: 基于理论心智的自动化元提示工程 

**Authors**: Aaron Baughman, Rahul Agarwal, Eduardo Morales, Gozde Akay  

**Link**: [PDF](https://arxiv.org/pdf/2505.09024)  

**Abstract**: We introduce a method of meta-prompting that jointly produces fluent text for complex tasks while optimizing the similarity of neural states between a human's mental expectation and a Large Language Model's (LLM) neural processing. A technique of agentic reinforcement learning is applied, in which an LLM as a Judge (LLMaaJ) teaches another LLM, through in-context learning, how to produce content by interpreting the intended and unintended generated text traits. To measure human mental beliefs around content production, users modify long form AI-generated text articles before publication at the US Open 2024 tennis Grand Slam. Now, an LLMaaJ can solve the Theory of Mind (ToM) alignment problem by anticipating and including human edits within the creation of text from an LLM. Throughout experimentation and by interpreting the results of a live production system, the expectations of human content reviewers had 100% of alignment with AI 53.8% of the time with an average iteration count of 4.38. The geometric interpretation of content traits such as factualness, novelty, repetitiveness, and relevancy over a Hilbert vector space combines spatial volume (all trait importance) with vertices alignment (individual trait relevance) enabled the LLMaaJ to optimize on Human ToM. This resulted in an increase in content quality by extending the coverage of tennis action. Our work that was deployed at the US Open 2024 has been used across other live events within sports and entertainment. 

**Abstract (ZH)**: 一种促进大型语言模型和人类思维一致性的元提示方法及其应用 

---
# Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation 

**Title (ZH)**: 基于深度强化学习的电力系统多阶段雪崩故障 mitigation 

**Authors**: Bo Meng, Chenghao Xu, Yongli Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09012)  

**Abstract**: Cascading failures in power grids can lead to grid collapse, causing severe disruptions to social operations and economic activities. In certain cases, multi-stage cascading failures can occur. However, existing cascading-failure-mitigation strategies are usually single-stage-based, overlooking the complexity of the multi-stage scenario. This paper treats the multi-stage cascading failure problem as a reinforcement learning task and develops a simulation environment. The reinforcement learning agent is then trained via the deterministic policy gradient algorithm to achieve continuous actions. Finally, the effectiveness of the proposed approach is validated on the IEEE 14-bus and IEEE 118-bus systems. 

**Abstract (ZH)**: 电力网络中的级联故障可能导致电网崩溃，严重扰乱社会运营和经济活动。在某些情况下，可能会发生多阶段级联故障。然而，现有的级联故障缓解策略通常是单阶段的，忽视了多阶段情景的复杂性。本文将多阶段级联故障问题视为强化学习任务，并开发了一个仿真环境。随后，使用确定性策略梯度算法训练强化学习代理以实现连续动作。最后，所提出的方法在IEEE 14节点系统和IEEE 118节点系统上进行了有效性验证。 

---
# Enhancing Aerial Combat Tactics through Hierarchical Multi-Agent Reinforcement Learning 

**Title (ZH)**: 通过分层多智能体强化学习提升空战战术 

**Authors**: Ardian Selmonaj, Oleg Szehr, Giacomo Del Rio, Alessandro Antonucci, Adrian Schneider, Michael Rüegsegger  

**Link**: [PDF](https://arxiv.org/pdf/2505.08995)  

**Abstract**: This work presents a Hierarchical Multi-Agent Reinforcement Learning framework for analyzing simulated air combat scenarios involving heterogeneous agents. The objective is to identify effective Courses of Action that lead to mission success within preset simulations, thereby enabling the exploration of real-world defense scenarios at low cost and in a safe-to-fail setting. Applying deep Reinforcement Learning in this context poses specific challenges, such as complex flight dynamics, the exponential size of the state and action spaces in multi-agent systems, and the capability to integrate real-time control of individual units with look-ahead planning. To address these challenges, the decision-making process is split into two levels of abstraction: low-level policies control individual units, while a high-level commander policy issues macro commands aligned with the overall mission targets. This hierarchical structure facilitates the training process by exploiting policy symmetries of individual agents and by separating control from command tasks. The low-level policies are trained for individual combat control in a curriculum of increasing complexity. The high-level commander is then trained on mission targets given pre-trained control policies. The empirical validation confirms the advantages of the proposed framework. 

**Abstract (ZH)**: 一种用于分析异构智能体模拟空战场景的分层多智能体强化学习框架 

---
# Generalization in Monitored Markov Decision Processes (Mon-MDPs) 

**Title (ZH)**: Monitored Markov决策过程（Mon-MDPs）的泛化能力 

**Authors**: Montaser Mohammedalamen, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2505.08988)  

**Abstract**: Reinforcement learning (RL) typically models the interaction between the agent and environment as a Markov decision process (MDP), where the rewards that guide the agent's behavior are always observable. However, in many real-world scenarios, rewards are not always observable, which can be modeled as a monitored Markov decision process (Mon-MDP). Prior work on Mon-MDPs have been limited to simple, tabular cases, restricting their applicability to real-world problems. This work explores Mon-MDPs using function approximation (FA) and investigates the challenges involved. We show that combining function approximation with a learned reward model enables agents to generalize from monitored states with observable rewards, to unmonitored environment states with unobservable rewards. Therefore, we demonstrate that such generalization with a reward model achieves near-optimal policies in environments formally defined as unsolvable. However, we identify a critical limitation of such function approximation, where agents incorrectly extrapolate rewards due to overgeneralization, resulting in undesirable behaviors. To mitigate overgeneralization, we propose a cautious police optimization method leveraging reward uncertainty. This work serves as a step towards bridging this gap between Mon-MDP theory and real-world applications. 

**Abstract (ZH)**: Mon-MDPs中基于函数逼近的学习及其挑战与改进：从可观察奖励向不可观察奖励的泛化 

---
# Grounding Synthetic Data Evaluations of Language Models in Unsupervised Document Corpora 

**Title (ZH)**: 基于未监督文档语料库的语言模型合成数据评估 

**Authors**: Michael Majurski, Cynthia Matuszek  

**Link**: [PDF](https://arxiv.org/pdf/2505.08905)  

**Abstract**: Language Models (LMs) continue to advance, improving response quality and coherence. Given Internet-scale training datasets, LMs have likely encountered much of what users might ask them to generate in some form during their training. A plethora of evaluation benchmarks have been constructed to assess model quality, response appropriateness, and reasoning capabilities. However, the human effort required for benchmark construction is limited and being rapidly outpaced by the size and scope of the models under evaluation. Additionally, having humans build a benchmark for every possible domain of interest is impractical. Therefore, we propose a methodology for automating the construction of fact-based synthetic data model evaluations grounded in document populations. This work leverages those very same LMs to evaluate domain-specific knowledge automatically, using only grounding documents (e.g., a textbook) as input. This synthetic data benchmarking approach corresponds well with human curated questions with a Spearman ranking correlation of 0.96 and a benchmark evaluation Pearson accuracy correlation of 0.79. This novel tool supports generating both multiple choice and open-ended synthetic data questions to gain diagnostic insight of LM capability. We apply this methodology to evaluate model performance on a recent relevant arXiv preprint, discovering a surprisingly strong performance from Gemma3 models. 

**Abstract (ZH)**: 语言模型继续进步，提升响应质量和连贯性。鉴于互联网规模的训练数据集，语言模型在训练过程中很可能已经遇到了用户可能请求生成的各种内容。已经构建了许多评估基准来评估模型质量、响应适宜性和推理能力。然而，基准构建所需的人力投入受到限制，并且正在被评估模型的规模和范围迅速超越。此外，为每一个感兴趣的领域手工构建基准是不切实际的。因此，我们提出了一种方法，通过基于文档集合自动构建事实基础的合成数据模型评估方法，从而实现自动化。该方法利用那些语言模型本身，仅使用锚定文档（例如，教科书）作为输入，自动评估领域的特定知识。这种合成数据基准方法与人工策划的问题具有Spearman排名相关性为0.96和基准评估Pearson准确性相关性为0.79。这一新颖工具支持生成选择题和开放题目的合成数据问题，以诊断语言模型的能力。我们将此方法应用于评估一个近期相关的arXiv预印本文本的表现，发现Gemma3模型取得了令人惊讶的强大性能。 

---
# Deep reinforcement learning-based longitudinal control strategy for automated vehicles at signalised intersections 

**Title (ZH)**: 基于深度强化学习的自动化车辆信号交叉口纵向控制策略 

**Authors**: Pankaj Kumar, Aditya Mishra, Pranamesh Chakraborty, Subrahmanya Swamy Peruru  

**Link**: [PDF](https://arxiv.org/pdf/2505.08896)  

**Abstract**: Developing an autonomous vehicle control strategy for signalised intersections (SI) is one of the challenging tasks due to its inherently complex decision-making process. This study proposes a Deep Reinforcement Learning (DRL) based longitudinal vehicle control strategy at SI. A comprehensive reward function has been formulated with a particular focus on (i) distance headway-based efficiency reward, (ii) decision-making criteria during amber light, and (iii) asymmetric acceleration/ deceleration response, along with the traditional safety and comfort criteria. This reward function has been incorporated with two popular DRL algorithms, Deep Deterministic Policy Gradient (DDPG) and Soft-Actor Critic (SAC), which can handle the continuous action space of acceleration/deceleration. The proposed models have been trained on the combination of real-world leader vehicle (LV) trajectories and simulated trajectories generated using the Ornstein-Uhlenbeck (OU) process. The overall performance of the proposed models has been tested using Cumulative Distribution Function (CDF) plots and compared with the real-world trajectory data. The results show that the RL models successfully maintain lower distance headway (i.e., higher efficiency) and jerk compared to human-driven vehicles without compromising safety. Further, to assess the robustness of the proposed models, we evaluated the model performance on diverse safety-critical scenarios, in terms of car-following and traffic signal compliance. Both DDPG and SAC models successfully handled the critical scenarios, while the DDPG model showed smoother action profiles compared to the SAC model. Overall, the results confirm that DRL-based longitudinal vehicle control strategy at SI can help to improve traffic safety, efficiency, and comfort. 

**Abstract (ZH)**: 基于深度强化学习的信号化交叉口自动驾驶车辆纵向控制策略研究 

---
# Customizing a Large Language Model for VHDL Design of High-Performance Microprocessors 

**Title (ZH)**: 自定义大型语言模型以用于高性能微处理器的VHDL设计 

**Authors**: Nicolas Dupuis, Ravi Nair, Shyam Ramji, Sean McClintock, Nishant Chauhan, Priyanka Nagpal, Bart Blaner, Ken Valk, Leon Stok, Ruchir Puri  

**Link**: [PDF](https://arxiv.org/pdf/2505.09610)  

**Abstract**: The use of Large Language Models (LLMs) in hardware design has taken off in recent years, principally through its incorporation in tools that increase chip designer productivity. There has been considerable discussion about the use of LLMs in RTL specifications of chip designs, for which the two most popular languages are Verilog and VHDL. LLMs and their use in Verilog design has received significant attention due to the higher popularity of the language, but little attention so far has been given to VHDL despite its continued popularity in the industry. There has also been little discussion about the unique needs of organizations that engage in high-performance processor design, and techniques to deploy AI solutions in these settings. In this paper, we describe our journey in developing a Large Language Model (LLM) specifically for the purpose of explaining VHDL code, a task that has particular importance in an organization with decades of experience and assets in high-performance processor design. We show how we developed test sets specific to our needs and used them for evaluating models as we performed extended pretraining (EPT) of a base LLM. Expert evaluation of the code explanations produced by the EPT model increased to 69% compared to a base model rating of 43%. We further show how we developed an LLM-as-a-judge to gauge models similar to expert evaluators. This led us to deriving and evaluating a host of new models, including an instruction-tuned version of the EPT model with an expected expert evaluator rating of 71%. Our experiments also indicate that with the potential use of newer base models, this rating can be pushed to 85% and beyond. We conclude with a discussion on further improving the quality of hardware design LLMs using exciting new developments in the Generative AI world. 

**Abstract (ZH)**: 大型语言模型在硬件设计中的应用：以VHDL代码解释器开发为例 

---
# How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference 

**Title (ZH)**: AI的饥饿程度如何？大规模语言模型推理的能耗、水资源消耗及碳足迹基准研究 

**Authors**: Nidhal Jegham, Marwen Abdelatti, Lassad Elmoubarki, Abdeltawab Hendawi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09598)  

**Abstract**: As large language models (LLMs) spread across industries, understanding their environmental footprint at the inference level is no longer optional; it is essential. However, most existing studies exclude proprietary models, overlook infrastructural variability and overhead, or focus solely on training, even as inference increasingly dominates AI's environmental impact. To bridge this gap, this paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: although individual queries are efficient, their global scale drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各行业的普及，理解其推理阶段的环境足迹已不再是一种选择，而是必要条件。然而，现有的大多数研究排除了私有模型，忽视了 infrastructural 变异性和开销，或者仅专注于训练，即使推理越来越多地影响着人工智能的环境影响。为填补这一空白，本文提出了一个基于基础设施的新基准测试框架，用于量化部署在商业数据中心的30个最先进的LLM推理的环境足迹。我们的框架结合了公共API性能数据、地区特定的环境多重因素以及硬件配置的统计推断。我们还利用交叉效率数据包络分析（DEA）来按相对于环境成本的性能对模型进行排名。我们的结果显示，o3和DeepSeek-R1 是最耗能的模型，每次长提示消耗超过33 Wh，比GPT-4.1 nano的消耗多70多倍。而Claude-3.7 Sonnet 在生态效率方面排名最高。单独一个较短的GPT-4o查询消耗0.43 Wh，将其放大到每天7亿次查询的结果会导致显著的年度环境影响。包括相当于35,000个美国家庭的用电量、符合120万人一年饮用水量的淡水蒸发以及需要相当于芝加哥大小森林的碳排放量来抵消。这些发现揭示了一个日益增长的悖论：尽管单个查询本身是高效的，但其全球规模导致了不成比例的资源消耗。我们的研究提供了一个标准化、实证为基础的方法来基准测试LLM部署的可持续性，为未来在人工智能开发和可持续性标准中的环境问责制奠定了基础。 

---
# WorldView-Bench: A Benchmark for Evaluating Global Cultural Perspectives in Large Language Models 

**Title (ZH)**: WorldView-Bench：评估大规模语言模型全球文化视角的基准 

**Authors**: Abdullah Mushtaq, Imran Taj, Rafay Naeem, Ibrahim Ghaznavi, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2505.09595)  

**Abstract**: Large Language Models (LLMs) are predominantly trained and aligned in ways that reinforce Western-centric epistemologies and socio-cultural norms, leading to cultural homogenization and limiting their ability to reflect global civilizational plurality. Existing benchmarking frameworks fail to adequately capture this bias, as they rely on rigid, closed-form assessments that overlook the complexity of cultural inclusivity. To address this, we introduce WorldView-Bench, a benchmark designed to evaluate Global Cultural Inclusivity (GCI) in LLMs by analyzing their ability to accommodate diverse worldviews. Our approach is grounded in the Multiplex Worldview proposed by Senturk et al., which distinguishes between Uniplex models, reinforcing cultural homogenization, and Multiplex models, which integrate diverse perspectives. WorldView-Bench measures Cultural Polarization, the exclusion of alternative perspectives, through free-form generative evaluation rather than conventional categorical benchmarks. We implement applied multiplexity through two intervention strategies: (1) Contextually-Implemented Multiplex LLMs, where system prompts embed multiplexity principles, and (2) Multi-Agent System (MAS)-Implemented Multiplex LLMs, where multiple LLM agents representing distinct cultural perspectives collaboratively generate responses. Our results demonstrate a significant increase in Perspectives Distribution Score (PDS) entropy from 13% at baseline to 94% with MAS-Implemented Multiplex LLMs, alongside a shift toward positive sentiment (67.7%) and enhanced cultural balance. These findings highlight the potential of multiplex-aware AI evaluation in mitigating cultural bias in LLMs, paving the way for more inclusive and ethically aligned AI systems. 

**Abstract (ZH)**: WorldView-Bench：评估大型语言模型全球文化包容性的新基准 

---
# Variational Visual Question Answering 

**Title (ZH)**: 变分视觉问答 

**Authors**: Tobias Jan Wieczorek, Nathalie Daun, Mohammad Emtiyaz Khan, Marcus Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2505.09591)  

**Abstract**: Despite remarkable progress in multimodal models for Visual Question Answering (VQA), there remain major reliability concerns because the models can often be overconfident and miscalibrated, especially in out-of-distribution (OOD) settings. Plenty has been done to address such issues for unimodal models, but little work exists for multimodal cases. Here, we address unreliability in multimodal models by proposing a Variational VQA approach. Specifically, instead of fine-tuning vision-language models by using AdamW, we employ a recently proposed variational algorithm called IVON, which yields a posterior distribution over model parameters. Through extensive experiments, we show that our approach improves calibration and abstentions without sacrificing the accuracy of AdamW. For instance, compared to AdamW fine-tuning, we reduce Expected Calibration Error by more than 50% compared to the AdamW baseline and raise Coverage by 4% vs. SOTA (for a fixed risk of 1%). In the presence of distribution shifts, the performance gain is even higher, achieving 8% Coverage (@ 1% risk) improvement vs. SOTA when 50% of test cases are OOD. Overall, we present variational learning as a viable option to enhance the reliability of multimodal models. 

**Abstract (ZH)**: 尽管在多模态模型用于视觉问答（VQA）方面取得了显著进展，但由于模型经常过度自信且校准不足，特别是在离分布（OOD）设置中，可靠性仍然是一个主要问题。虽然已经有很多工作针对单模态模型中的此类问题进行了研究，但在多模态情况下的工作却相对较少。在此，我们通过提出一种变分VQA方法来解决多模态模型的不可靠性问题。具体而言，我们不使用AdamW进行微调，而是采用最近提出的变分算法IVON，该算法能够获得模型参数的后验分布。通过广泛的实验，我们展示了我们的方法在不牺牲AdamW准确性的情况下提高校准和避免过多输出。例如，与AdamW微调相比，我们相比AdamW基线将预期校准误差降低了50%以上，并且在固定风险为1%的情况下，覆盖率提高了4%（达到SOTA）。在分布变化的情况下，性能提升更为显著：当测试案例中有50%是OOD时，相比SOTA，我们的方法在1%风险下的覆盖率提高了8%。总体而言，我们展示了变分学习作为一种提高多模态模型可靠性的可行选择。 

---
# Ethics and Persuasion in Reinforcement Learning from Human Feedback: A Procedural Rhetorical Approach 

**Title (ZH)**: 基于程序修辞的方法：从人类反馈中学习的强化学习中的伦理与说服研究 

**Authors**: Shannon Lodoen, Alexi Orchard  

**Link**: [PDF](https://arxiv.org/pdf/2505.09576)  

**Abstract**: Since 2022, versions of generative AI chatbots such as ChatGPT and Claude have been trained using a specialized technique called Reinforcement Learning from Human Feedback (RLHF) to fine-tune language model output using feedback from human annotators. As a result, the integration of RLHF has greatly enhanced the outputs of these large language models (LLMs) and made the interactions and responses appear more "human-like" than those of previous versions using only supervised learning. The increasing convergence of human and machine-written text has potentially severe ethical, sociotechnical, and pedagogical implications relating to transparency, trust, bias, and interpersonal relations. To highlight these implications, this paper presents a rhetorical analysis of some of the central procedures and processes currently being reshaped by RLHF-enhanced generative AI chatbots: upholding language conventions, information seeking practices, and expectations for social relationships. Rhetorical investigations of generative AI and LLMs have, to this point, focused largely on the persuasiveness of the content generated. Using Ian Bogost's concept of procedural rhetoric, this paper shifts the site of rhetorical investigation from content analysis to the underlying mechanisms of persuasion built into RLHF-enhanced LLMs. In doing so, this theoretical investigation opens a new direction for further inquiry in AI ethics that considers how procedures rerouted through AI-driven technologies might reinforce hegemonic language use, perpetuate biases, decontextualize learning, and encroach upon human relationships. It will therefore be of interest to educators, researchers, scholars, and the growing number of users of generative AI chatbots. 

**Abstract (ZH)**: 自2022年以来，如ChatGPT和Claude等生成式AI聊天机器人版本通过一种专门的技术——基于人类反馈的强化学习（RLHF）进行训练，使用人类注释者的反馈来微调语言模型的输出。RLHF的整合显著提升了这些大型语言模型（LLMs）的输出效果，使其互动和回应显得比仅使用监督学习的版本更加“人性化”。人类和机器写作文本日益融合可能对透明度、信任、偏见和人际关系等方面产生严重的伦理、社会技术和教育的影响。为了突显这些影响，本文对RLHF增强的生成式AI聊天机器人目前正重新塑造的一些核心程序和过程进行了修辞分析：维护语言规范、信息寻求行为以及社交关系的期望。迄今为止，关于生成式AI和LLMs的修辞研究主要集中在生成内容的说服力上。通过使用Ian Bogost的程序修辞概念，本文将修辞研究的焦点从内容分析转移到嵌入RLHF增强的LLMs中的说服机制。这一理论探讨为AI伦理研究开辟了新的方向，考虑了经由AI驱动技术重新导向的程序如何巩固霸权语言使用、延续偏见、脱离情境地学习以及侵犯人类关系的可能性。因此，这将对教育工作者、研究人员、学者以及日益增长的生成式AI聊天机器人用户群体产生兴趣。 

---
# BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset 

**Title (ZH)**: BLIP3-o：一个全开放统一多模态模型家族—架构、训练与数据集 

**Authors**: Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, Ran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09568)  

**Abstract**: Unifying image understanding and generation has gained growing attention in recent research on multimodal models. Although design choices for image understanding have been extensively studied, the optimal model architecture and training recipe for a unified framework with image generation remain underexplored. Motivated by the strong potential of autoregressive and diffusion models for high-quality generation and scalability, we conduct a comprehensive study of their use in unified multimodal settings, with emphasis on image representations, modeling objectives, and training strategies. Grounded in these investigations, we introduce a novel approach that employs a diffusion transformer to generate semantically rich CLIP image features, in contrast to conventional VAE-based representations. This design yields both higher training efficiency and improved generative quality. Furthermore, we demonstrate that a sequential pretraining strategy for unified models-first training on image understanding and subsequently on image generation-offers practical advantages by preserving image understanding capability while developing strong image generation ability. Finally, we carefully curate a high-quality instruction-tuning dataset BLIP3o-60k for image generation by prompting GPT-4o with a diverse set of captions covering various scenes, objects, human gestures, and more. Building on our innovative model design, training recipe, and datasets, we develop BLIP3-o, a suite of state-of-the-art unified multimodal models. BLIP3-o achieves superior performance across most of the popular benchmarks spanning both image understanding and generation tasks. To facilitate future research, we fully open-source our models, including code, model weights, training scripts, and pretraining and instruction tuning datasets. 

**Abstract (ZH)**: 统一图像理解和生成在多模态模型研究中的统一框架设计与训练策略探究 

---
# Meta-learning Slice-to-Volume Reconstruction in Fetal Brain MRI using Implicit Neural Representations 

**Title (ZH)**: 基于隐式神经表示的胎儿脑部MRI切片到体素重建元学习 

**Authors**: Maik Dannecker, Thomas Sanchez, Meritxell Bach Cuadra, Özgün Turgut, Anthony N. Price, Lucilio Cordero-Grande, Vanessa Kyriakopoulou, Joseph V. Hajnal, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09565)  

**Abstract**: High-resolution slice-to-volume reconstruction (SVR) from multiple motion-corrupted low-resolution 2D slices constitutes a critical step in image-based diagnostics of moving subjects, such as fetal brain Magnetic Resonance Imaging (MRI). Existing solutions struggle with image artifacts and severe subject motion or require slice pre-alignment to achieve satisfying reconstruction performance. We propose a novel SVR method to enable fast and accurate MRI reconstruction even in cases of severe image and motion corruption. Our approach performs motion correction, outlier handling, and super-resolution reconstruction with all operations being entirely based on implicit neural representations. The model can be initialized with task-specific priors through fully self-supervised meta-learning on either simulated or real-world data. In extensive experiments including over 480 reconstructions of simulated and clinical MRI brain data from different centers, we prove the utility of our method in cases of severe subject motion and image artifacts. Our results demonstrate improvements in reconstruction quality, especially in the presence of severe motion, compared to state-of-the-art methods, and up to 50% reduction in reconstruction time. 

**Abstract (ZH)**: 高分辨率切片到体素重建（SVR）：从多个运动 corrupted 低分辨率 2D 切片构建移动 Subject 的成像诊断在胎儿脑 MRI 中的应用 

---
# Learning Long-Context Diffusion Policies via Past-Token Prediction 

**Title (ZH)**: 基于过去词元预测的长上下文扩散策略学习 

**Authors**: Marcel Torne, Andy Tang, Yuejiang Liu, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2505.09561)  

**Abstract**: Reasoning over long sequences of observations and actions is essential for many robotic tasks. Yet, learning effective long-context policies from demonstrations remains challenging. As context length increases, training becomes increasingly expensive due to rising memory demands, and policy performance often degrades as a result of spurious correlations. Recent methods typically sidestep these issues by truncating context length, discarding historical information that may be critical for subsequent decisions. In this paper, we propose an alternative approach that explicitly regularizes the retention of past information. We first revisit the copycat problem in imitation learning and identify an opposite challenge in recent diffusion policies: rather than over-relying on prior actions, they often fail to capture essential dependencies between past and future actions. To address this, we introduce Past-Token Prediction (PTP), an auxiliary task in which the policy learns to predict past action tokens alongside future ones. This regularization significantly improves temporal modeling in the policy head, with minimal reliance on visual representations. Building on this observation, we further introduce a multistage training strategy: pre-train the visual encoder with short contexts, and fine-tune the policy head using cached long-context embeddings. This strategy preserves the benefits of PTP while greatly reducing memory and computational overhead. Finally, we extend PTP into a self-verification mechanism at test time, enabling the policy to score and select candidates consistent with past actions during inference. Experiments across four real-world and six simulated tasks demonstrate that our proposed method improves the performance of long-context diffusion policies by 3x and accelerates policy training by more than 10x. 

**Abstract (ZH)**: 长序列观测与动作推理是许多机器人任务的关键。然而，从演示中学习有效的长期上下文策略仍然具有挑战性。随着上下文长度的增加，由于内存需求的上升，训练变得越来越昂贵，政策性能往往会因虚假相关性而下降。最近的方法通常通过截断上下文长度来回避这些问题，从而忽略了对后续决策可能至关重要的历史信息。本文提出了一种替代方法，明确正则化保留过去信息。我们首先重新审视了模仿学习中的“复制猫”问题，并识别出最近扩散策略中的一个相反挑战：与过度依赖于先前的动作相比，它们往往未能捕捉到过去和未来动作之间的关键依赖性。为此，我们引入了过去动作令牌预测（PTP），这是一种辅助任务，在该任务中，策略学会同时预测过去的动作令牌和未来的动作令牌。这种正则化显著改进了策略头中的时间建模，同时减少了对视觉表示的依赖。在此基础上，我们进一步引入了一种多阶段训练策略：使用短上下文预训练视觉编码器，并使用缓存的长上下文嵌入微调策略头。这种策略保留了PTP的益处，同时大大降低了内存和计算开销。最后，我们在测试时将PTP扩展为一种自我验证机制，使政策能够在推理过程中为一致于过去动作的候选者评分并选择。在四个真实世界和六个模拟任务上的实验表明，我们提出的方法将长期上下文扩散策略的性能提升了3倍，并将策略训练速度加快了超过10倍。 

---
# WavReward: Spoken Dialogue Models With Generalist Reward Evaluators 

**Title (ZH)**: WavReward: 通用奖励评估器的语音对话模型 

**Authors**: Shengpeng Ji, Tianle Liang, Yangzhuo Li, Jialong Zuo, Minghui Fang, Jinzheng He, Yifu Chen, Zhengqing Liu, Ziyue Jiang, Xize Cheng, Siqi Zheng, Jin Xu, Junyang Lin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09558)  

**Abstract**: End-to-end spoken dialogue models such as GPT-4o-audio have recently garnered significant attention in the speech domain. However, the evaluation of spoken dialogue models' conversational performance has largely been overlooked. This is primarily due to the intelligent chatbots convey a wealth of non-textual information which cannot be easily measured using text-based language models like ChatGPT. To address this gap, we propose WavReward, a reward feedback model based on audio language models that can evaluate both the IQ and EQ of spoken dialogue systems with speech input. Specifically, 1) based on audio language models, WavReward incorporates the deep reasoning process and the nonlinear reward mechanism for post-training. By utilizing multi-sample feedback via the reinforcement learning algorithm, we construct a specialized evaluator tailored to spoken dialogue models. 2) We introduce ChatReward-30K, a preference dataset used to train WavReward. ChatReward-30K includes both comprehension and generation aspects of spoken dialogue models. These scenarios span various tasks, such as text-based chats, nine acoustic attributes of instruction chats, and implicit chats. WavReward outperforms previous state-of-the-art evaluation models across multiple spoken dialogue scenarios, achieving a substantial improvement about Qwen2.5-Omni in objective accuracy from 55.1$\%$ to 91.5$\%$. In subjective A/B testing, WavReward also leads by a margin of 83$\%$. Comprehensive ablation studies confirm the necessity of each component of WavReward. All data and code will be publicly at this https URL after the paper is accepted. 

**Abstract (ZH)**: 面向语音的端到端对话模型如GPT-4o-audio recently garnered significant attention.然而，对这些对话模型对话性能的评估却 largely 被忽视。为此，我们提出 WavReward，一种基于音频语言模型的奖励反馈模型，可以评估具有语音输入的对话系统的智商和情商。具体而言，1）基于音频语言模型，WavReward 结合了深度推理过程和非线性奖励机制。通过利用强化学习算法的多样本反馈，我们构建了一个针对语音对话模型的专业评估器。2）我们引入了 ChatReward-30K，这是一个用于训练 WavReward 的偏好数据集，包含了对话模型的理解和生成方面。这些场景涵盖了多种任务，如基于文本的对话、九种指令对话的声学属性以及隐式对话。WavReward 在多个语音对话场景中性能优于此前最先进的评估模型，客观准确率从 55.1% 提高到 91.5%，并且在主观 A/B 测试中也以 83% 的优势领先。全面的消融研究证实了 WavReward 中每个组件的必要性。论文接受后，所有数据和代码将在以下地址公开：https://... 

---
# Flash-VL 2B: Optimizing Vision-Language Model Performance for Ultra-Low Latency and High Throughput 

**Title (ZH)**: Flash-VL 2B: 优化超低延迟和高吞吐量的视觉语言模型性能 

**Authors**: Bo Zhang, Shuo Li, Runhe Tian, Yang Yang, Jixin Tang, Jinhao Zhou, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.09498)  

**Abstract**: In this paper, we introduce Flash-VL 2B, a novel approach to optimizing Vision-Language Models (VLMs) for real-time applications, targeting ultra-low latency and high throughput without sacrificing accuracy. Leveraging advanced architectural enhancements and efficient computational strategies, Flash-VL 2B is designed to maximize throughput by reducing processing time while maintaining competitive performance across multiple vision-language benchmarks. Our approach includes tailored architectural choices, token compression mechanisms, data curation, training schemes, and a novel image processing technique called implicit semantic stitching that effectively balances computational load and model performance. Through extensive evaluations on 11 standard VLM benchmarks, we demonstrate that Flash-VL 2B achieves state-of-the-art results in both speed and accuracy, making it a promising solution for deployment in resource-constrained environments and large-scale real-time applications. 

**Abstract (ZH)**: 在本文中，我们介绍了Flash-VL 2B，这是一种针对实时应用优化视觉-语言模型的新方法，旨在实现超低延迟和高吞吐量而不牺牲准确性。通过利用先进的架构增强和高效的计算策略，Flash-VL 2B 设计旨在通过减少处理时间来最大化吞吐量，同时在多个视觉-语言基准上保持竞争力。我们的方法包括定制的架构选择、标记压缩机制、数据收集、训练方案以及一种名为隐式语义缝合的新型图像处理技术，以有效地平衡计算负载和模型性能。通过在11个标准视觉-语言模型基准上的广泛评估，我们证明了Flash-VL 2B 在速度和准确性上均达到业内领先水平，使其成为资源受限环境和大规模实时应用程序部署的有前途的解决方案。 

---
# Preserving Plasticity in Continual Learning with Adaptive Linearity Injection 

**Title (ZH)**: 适应性线性注入以保持连续学习中的可塑性 

**Authors**: Seyed Roozbeh Razavi Rohani, Khashayar Khajavi, Wesley Chung, Mo Chen, Sharan Vaswani  

**Link**: [PDF](https://arxiv.org/pdf/2505.09486)  

**Abstract**: Loss of plasticity in deep neural networks is the gradual reduction in a model's capacity to incrementally learn and has been identified as a key obstacle to learning in non-stationary problem settings. Recent work has shown that deep linear networks tend to be resilient towards loss of plasticity. Motivated by this observation, we propose Adaptive Linearization (AdaLin), a general approach that dynamically adapts each neuron's activation function to mitigate plasticity loss. Unlike prior methods that rely on regularization or periodic resets, AdaLin equips every neuron with a learnable parameter and a gating mechanism that injects linearity into the activation function based on its gradient flow. This adaptive modulation ensures sufficient gradient signal and sustains continual learning without introducing additional hyperparameters or requiring explicit task boundaries. When used with conventional activation functions like ReLU, Tanh, and GeLU, we demonstrate that AdaLin can significantly improve performance on standard benchmarks, including Random Label and Permuted MNIST, Random Label and Shuffled CIFAR-10, and Class-Split CIFAR-100. Furthermore, its efficacy is shown in more complex scenarios, such as class-incremental learning on CIFAR-100 with a ResNet-18 backbone, and in mitigating plasticity loss in off-policy reinforcement learning agents. We perform a systematic set of ablations that show that neuron-level adaptation is crucial for good performance and analyze a number of metrics in the network that might be correlated to loss of plasticity. 

**Abstract (ZH)**: 深度神经网络中可塑性的丧失渐进减少模型逐步学习的能力，并被识别为非平稳问题设置中学习的关键障碍。最近的研究表明，深层线性网络倾向于对可塑性的丧失具有抗性。受这一观察的启发，我们提出了自适应线性化（AdaLin）方法，这是一种通用方法，能够动态地根据其梯度流为每个神经元的激活函数配备可学习参数和门控机制，注入线性。与依赖正则化或周期性重置的先前方法不同，AdaLin 不引入额外的超参数，也不需要显式的任务边界，即可确保梯度信号并持续学习。在使用如 ReLU、Tanh 和 GeLU 等标准激活函数时，我们证明了 AdaLin 可以在包括随机标签和 Permuted MNIST、随机标签和打乱的 CIFAR-10 以及类划分的 CIFAR-100 在内的标准基准测试中显著提高性能。此外，我们展示了其在更复杂场景中的有效性，如使用 ResNet-18 主干的 CIFAR-100 上的类增量学习，以及在异策略增强学习代理中缓解可塑性丧失。我们进行了系统性的消融实验，表明神经元级别的适应对于良好性能至关重要，并分析了网络中与可塑性丧失相关的若干指标。 

---
# Deploying Foundation Model-Enabled Air and Ground Robots in the Field: Challenges and Opportunities 

**Title (ZH)**: 基于基础模型的空中和地面机器人现场部署：挑战与机遇 

**Authors**: Zachary Ravichandran, Fernando Cladera, Jason Hughes, Varun Murali, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.09477)  

**Abstract**: The integration of foundation models (FMs) into robotics has enabled robots to understand natural language and reason about the semantics in their environments. However, existing FM-enabled robots primary operate in closed-world settings, where the robot is given a full prior map or has a full view of its workspace. This paper addresses the deployment of FM-enabled robots in the field, where missions often require a robot to operate in large-scale and unstructured environments. To effectively accomplish these missions, robots must actively explore their environments, navigate obstacle-cluttered terrain, handle unexpected sensor inputs, and operate with compute constraints. We discuss recent deployments of SPINE, our LLM-enabled autonomy framework, in field robotic settings. To the best of our knowledge, we present the first demonstration of large-scale LLM-enabled robot planning in unstructured environments with several kilometers of missions. SPINE is agnostic to a particular LLM, which allows us to distill small language models capable of running onboard size, weight and power (SWaP) limited platforms. Via preliminary model distillation work, we then present the first language-driven UAV planner using on-device language models. We conclude our paper by proposing several promising directions for future research. 

**Abstract (ZH)**: Foundation Models驱动的机器人在田野环境中规模化部署与规划研究 

---
# A 2D Semantic-Aware Position Encoding for Vision Transformers 

**Title (ZH)**: 二维语义感知位置编码用于视觉变换器 

**Authors**: Xi Chen, Shiyang Zhou, Muqi Huang, Jiaxu Feng, Yun Xiong, Kun Zhou, Biao Yang, Yuhui Zhang, Huishuai Bao, Sijia Peng, Chuan Li, Feng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09466)  

**Abstract**: Vision transformers have demonstrated significant advantages in computer vision tasks due to their ability to capture long-range dependencies and contextual relationships through self-attention. However, existing position encoding techniques, which are largely borrowed from natural language processing, fail to effectively capture semantic-aware positional relationships between image patches. Traditional approaches like absolute position encoding and relative position encoding primarily focus on 1D linear position relationship, often neglecting the semantic similarity between distant yet contextually related patches. These limitations hinder model generalization, translation equivariance, and the ability to effectively handle repetitive or structured patterns in images. In this paper, we propose 2-Dimensional Semantic-Aware Position Encoding ($\text{SaPE}^2$), a novel position encoding method with semantic awareness that dynamically adapts position representations by leveraging local content instead of fixed linear position relationship or spatial coordinates. Our method enhances the model's ability to generalize across varying image resolutions and scales, improves translation equivariance, and better aggregates features for visually similar but spatially distant patches. By integrating $\text{SaPE}^2$ into vision transformers, we bridge the gap between position encoding and perceptual similarity, thereby improving performance on computer vision tasks. 

**Abstract (ZH)**: Vision Transformners的2维语义感知位置编码(SaPE²)：通过利用局部内容动态适应位置表示以增强模型在计算机视觉任务中的性能 

---
# Quantum state-agnostic work extraction (almost) without dissipation 

**Title (ZH)**: 无耗散几乎不依赖于量子态的工作提取 

**Authors**: Josep Lumbreras, Ruo Cheng Huang, Yanglin Hu, Mile Gu, Marco Tomamichel  

**Link**: [PDF](https://arxiv.org/pdf/2505.09456)  

**Abstract**: We investigate work extraction protocols designed to transfer the maximum possible energy to a battery using sequential access to $N$ copies of an unknown pure qubit state. The core challenge is designing interactions to optimally balance two competing goals: charging of the battery optimally using the qubit in hand, and acquiring more information by qubit to improve energy harvesting in subsequent rounds. Here, we leverage exploration-exploitation trade-off in reinforcement learning to develop adaptive strategies achieving energy dissipation that scales only poly-logarithmically in $N$. This represents an exponential improvement over current protocols based on full state tomography. 

**Abstract (ZH)**: 我们研究了使用串行访问未知纯量子比特状态的拷贝来将尽可能多的能量转移到电池中的工作提取协议。核心挑战是设计交互以最优地平衡两个竞争目标：使用手中的量子比特最佳地给电池充电，以及通过量子比特获取更多信息以改进后续轮次的能量采集。在此，我们利用强化学习中的探索与利用权衡来开发出适应性策略，实现能耗仅按N的对数多项式尺度增长。这代表了当前基于完整态图论的协议的指数级改进。 

---
# Evaluating GPT- and Reasoning-based Large Language Models on Physics Olympiad Problems: Surpassing Human Performance and Implications for Educational Assessment 

**Title (ZH)**: 基于GPT和推理的大语言模型在物理奥林匹克问题上的评估：超越人类性能及对教育评估的影响 

**Authors**: Paul Tschisgale, Holger Maus, Fabian Kieser, Ben Kroehs, Stefan Petersen, Peter Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2505.09438)  

**Abstract**: Large language models (LLMs) are now widely accessible, reaching learners at all educational levels. This development has raised concerns that their use may circumvent essential learning processes and compromise the integrity of established assessment formats. In physics education, where problem solving plays a central role in instruction and assessment, it is therefore essential to understand the physics-specific problem-solving capabilities of LLMs. Such understanding is key to informing responsible and pedagogically sound approaches to integrating LLMs into instruction and assessment. This study therefore compares the problem-solving performance of a general-purpose LLM (GPT-4o, using varying prompting techniques) and a reasoning-optimized model (o1-preview) with that of participants of the German Physics Olympiad, based on a set of well-defined Olympiad problems. In addition to evaluating the correctness of the generated solutions, the study analyzes characteristic strengths and limitations of LLM-generated solutions. The findings of this study indicate that both tested LLMs (GPT-4o and o1-preview) demonstrate advanced problem-solving capabilities on Olympiad-type physics problems, on average outperforming the human participants. Prompting techniques had little effect on GPT-4o's performance, while o1-preview almost consistently outperformed both GPT-4o and the human benchmark. Based on these findings, the study discusses implications for the design of summative and formative assessment in physics education, including how to uphold assessment integrity and support students in critically engaging with LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）现在广泛可用，已触及各个教育层次的学习者。这一发展引发了对其使用可能绕过关键学习过程并损害既定评估格式完整性的担忧。在物理教育中，由于解决问题在教学和评估中扮演着核心角色，因此理解LLMs的物理专用问题解决能力变得至关重要。这种理解对于指导负责任且教育上合理的将LLMs整合到教学和评估中的方法具有重要意义。因此，本研究比较了一般用途LLM（使用不同提示技术的GPT-4o）和推理优化模型（o1-preview）与德国物理奥林匹克参赛者在一组定义明确的物理奥林匹克问题上的问题解决表现。除了评估生成解的正确性外，本研究还分析了LLM生成解的特征优势和局限性。研究结果表明，两种测试的LLMs（GPT-4o和o1-preview）在物理奥林匹克类型的问题上表现出高级问题解决能力，平均而言优于人类参与者。提示技术对GPT-4o的性能影响甚微，而o1-preview几乎总是优于GPT-4o和人类基准。基于这些发现，本研究讨论了物理教育中总结性和形成性评估设计的含义，包括如何维护评估完整性以及支持学生批判性地与LLMs互动。 

---
# CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios 

**Title (ZH)**: CXMArena: 统一数据集以在真实的客户关系管理场景中评估性能 

**Authors**: Raghav Garg, Kapil Sharma, Karan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.09436)  

**Abstract**: Large Language Models (LLMs) hold immense potential for revolutionizing Customer Experience Management (CXM), particularly in contact center operations. However, evaluating their practical utility in complex operational environments is hindered by data scarcity (due to privacy concerns) and the limitations of current benchmarks. Existing benchmarks often lack realism, failing to incorporate deep knowledge base (KB) integration, real-world noise, or critical operational tasks beyond conversational fluency. To bridge this gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset specifically designed for evaluating AI in operational CXM contexts. Given the diversity in possible contact center features, we have developed a scalable LLM-powered pipeline that simulates the brand's CXM entities that form the foundation of our datasets-such as knowledge articles including product specifications, issue taxonomies, and contact center conversations. The entities closely represent real-world distribution because of controlled noise injection (informed by domain experts) and rigorous automated validation. Building on this, we release CXMArena, which provides dedicated benchmarks targeting five important operational tasks: Knowledge Base Refinement, Intent Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with Integrated Tools. Our baseline experiments underscore the benchmark's difficulty: even state of the art embedding and generation models achieve only 68% accuracy on article search, while standard embedding methods yield a low F1 score of 0.3 for knowledge base refinement, highlighting significant challenges for current models necessitating complex pipelines and solutions over conventional techniques. 

**Abstract (ZH)**: 大型语言模型（LLMs）在客户体验管理（CXM）特别是在接触中心运营中的革命潜力巨大。然而，由于隐私问题导致的数据稀缺性和现有基准的局限性，评估其在复杂运营环境中的实际效用受到阻碍。现有的基准往往缺乏现实性，未能整合深度知识库（KB）、现实世界噪音或超出对话流畅性的重要运营任务。为填补这一缺口，我们提出了CXMArena，这是一个新型的大规模合成基准数据集，专门用于评估AI在运营CXM环境中的性能。鉴于接触中心特征的多样性，我们开发了一个可扩展的LLM驱动的流程，模拟品牌的CXM实体，这些实体构成了我们数据集的基础，如包含产品规格、问题分类和接触中心对话的知识文章。实体由于受控的噪音注入（由领域专家指导）和严格的自动化验证，接近真实世界的应用分布。在此基础上，我们发布了CXMArena，提供了针对五个重要运营任务的专用基准：知识库精炼、意图预测、代理质量遵从性、文章搜索以及集成工具的多轮检索与生成。基线实验突显了基准的难度：即使最先进的嵌入和生成模型在文章搜索任务上的准确率也只有68%，标准嵌入方法在知识库精炼任务上的F1分数仅为0.3，这表明当前模型面临巨大挑战，需要复杂的流程和解决方案才能超越传统技术。 

---
# Endo-CLIP: Progressive Self-Supervised Pre-training on Raw Colonoscopy Records 

**Title (ZH)**: Endo-CLIP: 基于原始肠镜记录的分阶段自我监督预训练 

**Authors**: Yili He, Yan Zhu, Peiyao Fu, Ruijie Yang, Tianyi Chen, Zhihua Wang, Quanlin Li, Pinghong Zhou, Xian Yang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09435)  

**Abstract**: Pre-training on image-text colonoscopy records offers substantial potential for improving endoscopic image analysis, but faces challenges including non-informative background images, complex medical terminology, and ambiguous multi-lesion descriptions. We introduce Endo-CLIP, a novel self-supervised framework that enhances Contrastive Language-Image Pre-training (CLIP) for this domain. Endo-CLIP's three-stage framework--cleansing, attunement, and unification--addresses these challenges by (1) removing background frames, (2) leveraging large language models to extract clinical attributes for fine-grained contrastive learning, and (3) employing patient-level cross-attention to resolve multi-polyp ambiguities. Extensive experiments demonstrate that Endo-CLIP significantly outperforms state-of-the-art pre-training methods in zero-shot and few-shot polyp detection and classification, paving the way for more accurate and clinically relevant endoscopic analysis. 

**Abstract (ZH)**: 基于内镜图像-文本结肠镜检查记录的预训练在内镜图像分析中具有巨大潜力，但面临背景图像无信息性、复杂医学术语和多病灶描述模糊等挑战。我们介绍了Endo-CLIP，这是一种新颖的自监督框架，旨在增强该领域的对比语言-图像预训练（CLIP）。Endo-CLIP 的三阶段框架——净化、调谐和统一——通过（1）去除背景帧，（2）利用大规模语言模型提取临床属性以实现精细对比学习，以及（3）采用病患级别的跨注意力解决多腺瘤模糊性，来应对这些挑战。广泛实验表明，Endo-CLIP 在零样本和少样本息肉检测与分类中显著优于现有预训练方法，为更准确和临床相关性更强的内镜分析铺平了道路。 

---
# Multilingual Machine Translation with Quantum Encoder Decoder Attention-based Convolutional Variational Circuits 

**Title (ZH)**: 基于量子编码解码注意机制卷积变分电路的多语言机器翻译 

**Authors**: Subrit Dikshit, Ritu Tiwari, Priyank Jain  

**Link**: [PDF](https://arxiv.org/pdf/2505.09407)  

**Abstract**: Cloud-based multilingual translation services like Google Translate and Microsoft Translator achieve state-of-the-art translation capabilities. These services inherently use large multilingual language models such as GRU, LSTM, BERT, GPT, T5, or similar encoder-decoder architectures with attention mechanisms as the backbone. Also, new age natural language systems, for instance ChatGPT and DeepSeek, have established huge potential in multiple tasks in natural language processing. At the same time, they also possess outstanding multilingual translation capabilities. However, these models use the classical computing realm as a backend. QEDACVC (Quantum Encoder Decoder Attention-based Convolutional Variational Circuits) is an alternate solution that explores the quantum computing realm instead of the classical computing realm to study and demonstrate multilingual machine translation. QEDACVC introduces the quantum encoder-decoder architecture that simulates and runs on quantum computing hardware via quantum convolution, quantum pooling, quantum variational circuit, and quantum attention as software alterations. QEDACVC achieves an Accuracy of 82% when trained on the OPUS dataset for English, French, German, and Hindi corpora for multilingual translations. 

**Abstract (ZH)**: 基于云的多语言翻译服务如Google Translate和Microsoft Translator实现了最先进的翻译能力。这些服务本质上使用了如GRU、LSTM、BERT、GPT、T5或类似的关注机制下编码解码架构的大规模多语言语言模型。同时，新一代自然语言系统，例如ChatGPT和DeepSeek，在多项自然语言处理任务中展示了巨大的潜力，同时也具备出色的多语言翻译能力。然而，这些模型使用经典计算域作为后端。QEDACVC（Quantum Encoder Decoder Attention-based Convolutional Variational Circuits）是一种替代方案，它探索量子计算领域而非经典计算领域来研究和展示多语言机器翻译。QEDACVC引入了一种量子编码解码架构，通过量子卷积、量子池化、量子变分电路和量子注意力等软件修改在量子计算硬件上进行模拟和运行。当在OPUS数据集的英语、法语、德语和印地语语料上训练时，QEDACVC达到了82%的准确率。 

---
# Quantum-Enhanced Parameter-Efficient Learning for Typhoon Trajectory Forecasting 

**Title (ZH)**: 量子增强的参数高效学习应用于台风路径预测 

**Authors**: Chen-Yu Liu, Kuan-Cheng Chen, Yi-Chien Chen, Samuel Yen-Chi Chen, Wei-Hao Huang, Wei-Jia Huang, Yen-Jui Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09395)  

**Abstract**: Typhoon trajectory forecasting is essential for disaster preparedness but remains computationally demanding due to the complexity of atmospheric dynamics and the resource requirements of deep learning models. Quantum-Train (QT), a hybrid quantum-classical framework that leverages quantum neural networks (QNNs) to generate trainable parameters exclusively during training, eliminating the need for quantum hardware at inference time. Building on QT's success across multiple domains, including image classification, reinforcement learning, flood prediction, and large language model (LLM) fine-tuning, we introduce Quantum Parameter Adaptation (QPA) for efficient typhoon forecasting model learning. Integrated with an Attention-based Multi-ConvGRU model, QPA enables parameter-efficient training while maintaining predictive accuracy. This work represents the first application of quantum machine learning (QML) to large-scale typhoon trajectory prediction, offering a scalable and energy-efficient approach to climate modeling. Our results demonstrate that QPA significantly reduces the number of trainable parameters while preserving performance, making high-performance forecasting more accessible and sustainable through hybrid quantum-classical learning. 

**Abstract (ZH)**: 量子参数调整（QPA）在注意力多卷积门循环单元模型中的高效台风轨迹预测 

---
# UMotion: Uncertainty-driven Human Motion Estimation from Inertial and Ultra-wideband Units 

**Title (ZH)**: UMotion: 基于不确定性的人体运动估计方法从惯性单元和超宽带单元 

**Authors**: Huakun Liu, Hiroki Ota, Xin Wei, Yutaro Hirao, Monica Perusquia-Hernandez, Hideaki Uchiyama, Kiyoshi Kiyokawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.09393)  

**Abstract**: Sparse wearable inertial measurement units (IMUs) have gained popularity for estimating 3D human motion. However, challenges such as pose ambiguity, data drift, and limited adaptability to diverse bodies persist. To address these issues, we propose UMotion, an uncertainty-driven, online fusing-all state estimation framework for 3D human shape and pose estimation, supported by six integrated, body-worn ultra-wideband (UWB) distance sensors with IMUs. UWB sensors measure inter-node distances to infer spatial relationships, aiding in resolving pose ambiguities and body shape variations when combined with anthropometric data. Unfortunately, IMUs are prone to drift, and UWB sensors are affected by body occlusions. Consequently, we develop a tightly coupled Unscented Kalman Filter (UKF) framework that fuses uncertainties from sensor data and estimated human motion based on individual body shape. The UKF iteratively refines IMU and UWB measurements by aligning them with uncertain human motion constraints in real-time, producing optimal estimates for each. Experiments on both synthetic and real-world datasets demonstrate the effectiveness of UMotion in stabilizing sensor data and the improvement over state of the art in pose accuracy. 

**Abstract (ZH)**: 基于不确定性驱动的在线多状态融合估计算法的3D人体运动及形态估计 

---
# FedSaaS: Class-Consistency Federated Semantic Segmentation via Global Prototype Supervision and Local Adversarial Harmonization 

**Title (ZH)**: FedSaaS: 基于全局原型监督和局部对抗 harmonization 的类一致联邦语义分割 

**Authors**: Xiaoyang Yu, Xiaoming Wu, Xin Wang, Dongrun Li, Ming Yang, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.09385)  

**Abstract**: Federated semantic segmentation enables pixel-level classification in images through collaborative learning while maintaining data privacy. However, existing research commonly overlooks the fine-grained class relationships within the semantic space when addressing heterogeneous problems, particularly domain shift. This oversight results in ambiguities between class representation. To overcome this challenge, we propose a novel federated segmentation framework that strikes class consistency, termed FedSaaS. Specifically, we introduce class exemplars as a criterion for both local- and global-level class representations. On the server side, the uploaded class exemplars are leveraged to model class prototypes, which supervise global branch of clients, ensuring alignment with global-level representation. On the client side, we incorporate an adversarial mechanism to harmonize contributions of global and local branches, leading to consistent output. Moreover, multilevel contrastive losses are employed on both sides to enforce consistency between two-level representations in the same semantic space. Extensive experiments on several driving scene segmentation datasets demonstrate that our framework outperforms state-of-the-art methods, significantly improving average segmentation accuracy and effectively addressing the class-consistency representation problem. 

**Abstract (ZH)**: 联邦语义分割通过协作学习在保持数据隐私的同时实现像素级分类，但现有研究在处理异构问题时，尤其是领域转移问题时，通常忽略了语义空间内的细粒度类关系，导致类表示之间的模糊性。为解决这一挑战，我们提出了一种新的联邦分割框架FedSaaS，该框架确保类的一致性。具体而言，我们引入类示例作为局部和全局类表示的标准。在服务器端，上传的类示例用于建模类原型，监督客户端的全局分支，确保与全局水平表示的一致性。在客户端，我们引入对抗机制以协调全局和局部分支的贡献，从而实现一致的输出。此外，我们还在双方使用多级对比损失，以确保相同语义空间中两层表示的一致性。在几个驾驶场景分割数据集上的 extensive 实验表明，我们的框架优于现有方法，在平均分割准确性和有效地解决类一致表示问题方面表现出显著提升。 

---
# The Voice Timbre Attribute Detection 2025 Challenge Evaluation Plan 

**Title (ZH)**: 2025年度语音音色属性检测挑战赛评估计划 

**Authors**: Zhengyan Sheng, Jinghao He, Liping Chen, Kong Aik Lee, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.09382)  

**Abstract**: Voice timbre refers to the unique quality or character of a person's voice that distinguishes it from others as perceived by human hearing. The Voice Timbre Attribute Detection (VtaD) 2025 challenge focuses on explaining the voice timbre attribute in a comparative manner. In this challenge, the human impression of voice timbre is verbalized with a set of sensory descriptors, including bright, coarse, soft, magnetic, and so on. The timbre is explained from the comparison between two voices in their intensity within a specific descriptor dimension. The VtaD 2025 challenge starts in May and culminates in a special proposal at the NCMMSC2025 conference in October 2025 in Zhenjiang, China. 

**Abstract (ZH)**: Voice Timbre 属性检测 (VtaD 2025) 挑战专注于比较性地解释声音色彩属性。在该挑战中，人类对声音色彩的印象通过一系列表征形容词，如明亮、粗糙、柔软、磁性等，予以口头化。声音色彩在特定描述维度内的强度对比来予以解释。VtaD 2025 挑战于5月启动，并将于2025年10月在中国镇江举行的NCMMSC2025会议上提交特别提案。 

---
# Examining Deployment and Refinement of the VIOLA-AI Intracranial Hemorrhage Model Using an Interactive NeoMedSys Platform 

**Title (ZH)**: 基于Interactive NeoMedSys平台的VIOLA-AI颅内出血模型的部署与精炼研究 

**Authors**: Qinghui Liu, Jon Nesvold, Hanna Raaum, Elakkyen Murugesu, Martin Røvang, Bradley J Maclntosh, Atle Bjørnerud, Karoline Skogen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09380)  

**Abstract**: Background: There are many challenges and opportunities in the clinical deployment of AI tools in radiology. The current study describes a radiology software platform called NeoMedSys that can enable efficient deployment and refinements of AI models. We evaluated the feasibility and effectiveness of running NeoMedSys for three months in real-world clinical settings and focused on improvement performance of an in-house developed AI model (VIOLA-AI) designed for intracranial hemorrhage (ICH) detection.
Methods: NeoMedSys integrates tools for deploying, testing, and optimizing AI models with a web-based medical image viewer, annotation system, and hospital-wide radiology information systems. A pragmatic investigation was deployed using clinical cases of patients presenting to the largest Emergency Department in Norway (site-1) with suspected traumatic brain injury (TBI) or patients with suspected stroke (site-2). We assessed ICH classification performance as VIOLA-AI encountered new data and underwent pre-planned model retraining. Performance metrics included sensitivity, specificity, accuracy, and the area under the receiver operating characteristic curve (AUC).
Results: NeoMedSys facilitated iterative improvements in the AI model, significantly enhancing its diagnostic accuracy. Automated bleed detection and segmentation were reviewed in near real-time to facilitate re-training VIOLA-AI. The iterative refinement process yielded a marked improvement in classification sensitivity, rising to 90.3% (from 79.2%), and specificity that reached 89.3% (from 80.7%). The bleed detection ROC analysis for the entire sample demonstrated a high area-under-the-curve (AUC) of 0.949 (from 0.873). Model refinement stages were associated with notable gains, highlighting the value of real-time radiologist feedback. 

**Abstract (ZH)**: 背景：人工智能工具在放射学临床部署中面临着诸多挑战和机遇。本研究描述了一个名为NeoMedSys的放射学软件平台，该平台能够实现AI模型的高效部署和优化。我们评估了在挪威最大的紧急部门（站点-1）和疑似脑卒中患者（站点-2）的临床环境中运行NeoMedSys三个月的可行性和有效性，并重点关注了内部开发的针对颅内出血检测的AI模型（VIOLA-AI）的性能改进。 

---
# TensorRL-QAS: Reinforcement learning with tensor networks for scalable quantum architecture search 

**Title (ZH)**: 张量RL-QAS: 基于张量网络的可扩展量子架构搜索 reinforcement learning方法 

**Authors**: Akash Kundu, Stefano Mangini  

**Link**: [PDF](https://arxiv.org/pdf/2505.09371)  

**Abstract**: Variational quantum algorithms hold the promise to address meaningful quantum problems already on noisy intermediate-scale quantum hardware, but they face the challenge of designing quantum circuits that both solve the target problem and comply with device limitations. Quantum architecture search (QAS) automates this design process, with reinforcement learning (RL) emerging as a promising approach. Yet, RL-based QAS methods encounter significant scalability issues, as computational and training costs grow rapidly with the number of qubits, circuit depth, and noise, severely impacting performance. To address these challenges, we introduce $\textit{TensorRL-QAS}$, a scalable framework that combines tensor network (TN) methods with RL for designing quantum circuits. By warm-starting the architecture search with a matrix product state approximation of the target solution, TensorRL-QAS effectively narrows the search space to physically meaningful circuits, accelerating convergence to the desired solution. Tested on several quantum chemistry problems of up to 12-qubit, TensorRL-QAS achieves up to a 10-fold reduction in CNOT count and circuit depth compared to baseline methods, while maintaining or surpassing chemical accuracy. It reduces function evaluations by up to 100-fold, accelerates training episodes by up to $98\%$, and achieves up to $50\%$ success probability for 10-qubit systems-far exceeding the $<1\%$ rates of baseline approaches. Robustness and versatility are demonstrated both in the noiseless and noisy scenarios, where we report a simulation of up to 8-qubit. These advancements establish TensorRL-QAS as a promising candidate for a scalable and efficient quantum circuit discovery protocol on near-term quantum hardware. 

**Abstract (ZH)**: TensorRL-QAS：一种用于近期内量子硬件的可扩展高效量子电路发现协议 

---
# GreenFactory: Ensembling Zero-Cost Proxies to Estimate Performance of Neural Networks 

**Title (ZH)**: GreenFactory: 组合零开销代理以估计神经网络性能 

**Authors**: Gabriel Cortês, Nuno Lourenço, Paolo Romano, Penousal Machado  

**Link**: [PDF](https://arxiv.org/pdf/2505.09344)  

**Abstract**: Determining the performance of a Deep Neural Network during Neural Architecture Search processes is essential for identifying optimal architectures and hyperparameters. Traditionally, this process requires training and evaluation of each network, which is time-consuming and resource-intensive. Zero-cost proxies estimate performance without training, serving as an alternative to traditional training. However, recent proxies often lack generalization across diverse scenarios and provide only relative rankings rather than predicted accuracies. To address these limitations, we propose GreenFactory, an ensemble of zero-cost proxies that leverages a random forest regressor to combine multiple predictors' strengths and directly predict model test accuracy. We evaluate GreenFactory on NATS-Bench, achieving robust results across multiple datasets. Specifically, GreenFactory achieves high Kendall correlations on NATS-Bench-SSS, indicating substantial agreement between its predicted scores and actual performance: 0.907 for CIFAR-10, 0.945 for CIFAR-100, and 0.920 for ImageNet-16-120. Similarly, on NATS-Bench-TSS, we achieve correlations of 0.921 for CIFAR-10, 0.929 for CIFAR-100, and 0.908 for ImageNet-16-120, showcasing its reliability in both search spaces. 

**Abstract (ZH)**: 确定深度神经网络在神经架构搜索过程中的性能对于识别最优架构和超参数至关重要。传统方法需要对每个网络进行训练和评估，这一过程耗时且资源密集。零成本代理可以通过无需训练的方式估计性能，成为传统训练的替代方案。然而，最近的代理在不同场景下的泛化能力往往不足，只能提供相对排名而非预测准确率。为解决这些问题，我们提出GreenFactory，这是一种结合随机森林回归器的零成本代理ensemble，利用多种预测器的优势直接预测模型测试准确率。我们基于NATS-Bench评估GreenFactory，结果显示其在多个数据集上具有稳健的结果。具体而言，GreenFactory在NATS-Bench-SSS上实现了高Kendall相关性，表明其预测分数与实际性能有显著的一致性：CIFAR-10为0.907，CIFAR-100为0.945，ImageNet-16-120为0.920。同样，在NATS-Bench-TSS上，我们得到的CIFAR-10为0.921，CIFAR-100为0.929，ImageNet-16-120为0.908，展示了其在两个搜索空间中的可靠性。 

---
# Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures 

**Title (ZH)**: DeepSeek-V3的深度学习扩展挑战及硬件架构反思 

**Authors**: Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, Y.X. Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09343)  

**Abstract**: The rapid scaling of large language models (LLMs) has unveiled critical limitations in current hardware architectures, including constraints in memory capacity, computational efficiency, and interconnection bandwidth. DeepSeek-V3, trained on 2,048 NVIDIA H800 GPUs, demonstrates how hardware-aware model co-design can effectively address these challenges, enabling cost-efficient training and inference at scale. This paper presents an in-depth analysis of the DeepSeek-V3/R1 model architecture and its AI infrastructure, highlighting key innovations such as Multi-head Latent Attention (MLA) for enhanced memory efficiency, Mixture of Experts (MoE) architectures for optimized computation-communication trade-offs, FP8 mixed-precision training to unlock the full potential of hardware capabilities, and a Multi-Plane Network Topology to minimize cluster-level network overhead. Building on the hardware bottlenecks encountered during DeepSeek-V3's development, we engage in a broader discussion with academic and industry peers on potential future hardware directions, including precise low-precision computation units, scale-up and scale-out convergence, and innovations in low-latency communication fabrics. These insights underscore the critical role of hardware and model co-design in meeting the escalating demands of AI workloads, offering a practical blueprint for innovation in next-generation AI systems. 

**Abstract (ZH)**: 大语言模型的快速扩展揭示了当前硬件架构的关键限制，包括内存容量、计算效率和 interconnection 带宽的约束。DeepSeek-V3 在 2048 块 NVIDIA H800 GPU 上训练，展示了基于硬件的模型协同设计如何有效地应对这些挑战，从而实现大规模的成本效益训练和推理。本文深入分析了 DeepSeek-V3/R1 模型架构及其 AI 基础设施，强调了多项关键创新，包括多头潜在注意力（MLA）以提高内存效率、专家混合架构（MoE）以优化计算-通信权衡、使用 FP8 混合精度训练以充分利用硬件能力，以及多平面网络拓扑以最小化集群级网络开销。基于 DeepSeek-V3 开发过程中遇到的硬件瓶颈，本文与学术界和工业界同行就未来硬件方向进行了更广泛讨论，包括精确的低精度计算单元、扩展性和分发性的收敛以及低延迟通信网络的创新。这些见解突显了硬件与模型协同设计在满足 AI 工作负载不断上升的需求中的关键作用，为下一代 AI 系统的创新提供了实用蓝图。 

---
# Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems 

**Title (ZH)**: 评估恶意软件检测系统中对抗防御的 robustness 

**Authors**: Mostafa Jafari, Alireza Shameli-Sendi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09342)  

**Abstract**: Machine learning is a key tool for Android malware detection, effectively identifying malicious patterns in apps. However, ML-based detectors are vulnerable to evasion attacks, where small, crafted changes bypass detection. Despite progress in adversarial defenses, the lack of comprehensive evaluation frameworks in binary-constrained domains limits understanding of their robustness. We introduce two key contributions. First, Prioritized Binary Rounding, a technique to convert continuous perturbations into binary feature spaces while preserving high attack success and low perturbation size. Second, the sigma-binary attack, a novel adversarial method for binary domains, designed to achieve attack goals with minimal feature changes. Experiments on the Malscan dataset show that sigma-binary outperforms existing attacks and exposes key vulnerabilities in state-of-the-art defenses. Defenses equipped with adversary detectors, such as KDE, DLA, DNN+, and ICNN, exhibit significant brittleness, with attack success rates exceeding 90% using fewer than 10 feature modifications and reaching 100% with just 20. Adversarially trained defenses, including AT-rFGSM-k, AT-MaxMA, improves robustness under small budgets but remains vulnerable to unrestricted perturbations, with attack success rates of 99.45% and 96.62%, respectively. Although PAD-SMA demonstrates strong robustness against state-of-the-art gradient-based adversarial attacks by maintaining an attack success rate below 16.55%, the sigma-binary attack significantly outperforms these methods, achieving a 94.56% success rate under unrestricted perturbations. These findings highlight the critical need for precise method like sigma-binary to expose hidden vulnerabilities in existing defenses and support the development of more resilient malware detection systems. 

**Abstract (ZH)**: 机器学习是Android恶意软件检测的关键工具，有效识别应用中的恶意模式。然而，基于ML的检测器易受规避攻击的影响，即通过微小、精制的更改绕过检测。尽管在对抗性防御方面取得了进展，但在二进制受限领域缺乏全面的评估框架限制了对它们鲁棒性的理解。我们引入了两项关键技术贡献。首先，优先级二进制舍入技术，该技术能够在保持高攻击成功率和低扰动规模的同时，将连续扰动转换为二进制特征空间。其次，Sigma-Binary攻击，这是一种针对二进制领域的新型对抗性方法，旨在通过最小的特征变化实现攻击目标。实验表明，Sigma-Binary攻击优于现有攻击方法，并揭示了先进防御方法的关键漏洞。配备对抗性检测器的防御措施，如KDE、DLA、DNN+和ICNN，在不超过10个特征修改的情况下，攻击成功率超过90%，使用仅20个特征修改即可达到100%。对抗性训练的防御措施，包括AT-rFGSM-k和AT-MaxMA，在预算较小的情况下提高了鲁棒性，但仍对无限制扰动易受攻击，其攻击成功率分别为99.45%和96.62%。尽管PAD-SMA通过保持攻击成功率低于16.55%展示了对最先进的基于梯度的对抗性攻击的强鲁棒性，但Sigma-Binary攻击在无限制扰动下仍显著优于这些方法，攻击成功率达到94.56%。这些发现强调了迫切需要精确的方法如Sigma-Binary来揭示现有防御措施中的隐藏漏洞，并支持更坚resilient的恶意软件检测系统的开发。 

---
# BioVFM-21M: Benchmarking and Scaling Self-Supervised Vision Foundation Models for Biomedical Image Analysis 

**Title (ZH)**: BioVFM-21M：自我监督视觉基础模型在生物医学图像分析中的基准测试与扩展研究 

**Authors**: Jiarun Liu, Hong-Yu Zhou, Weijian Huang, Hao Yang, Dongning Song, Tao Tan, Yong Liang, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09329)  

**Abstract**: Scaling up model and data size have demonstrated impressive performance improvement over a wide range of tasks. Despite extensive studies on scaling behaviors for general-purpose tasks, medical images exhibit substantial differences from natural data. It remains unclear the key factors in developing medical vision foundation models at scale due to the absence of an extensive understanding of scaling behavior in the medical domain. In this paper, we explored the scaling behavior across model sizes, training algorithms, data sizes, and imaging modalities in developing scalable medical vision foundation models by self-supervised learning. To support scalable pretraining, we introduce BioVFM-21M, a large-scale biomedical image dataset encompassing a wide range of biomedical image modalities and anatomies. We observed that scaling up does provide benefits but varies across tasks. Additional analysis reveals several factors correlated with scaling benefits. Finally, we propose BioVFM, a large-scale medical vision foundation model pretrained on 21 million biomedical images, which outperforms the previous state-of-the-art foundation models across 12 medical benchmarks. Our results highlight that while scaling up is beneficial for pursuing better performance, task characteristics, data diversity, pretraining methods, and computational efficiency remain critical considerations for developing scalable medical foundation models. 

**Abstract (ZH)**: 扩展模型和数据规模在广泛的任务中展现了显著的性能提升。尽管已经对通用任务的扩展行为进行了大量研究，但医学图像与自然数据之间存在显著差异。由于对医学领域中扩展行为缺乏广泛的理解，开发大规模医学视觉基础模型的关键因素尚不清楚。在本文中，我们通过自监督学习探索了在模型规模、训练算法、数据规模和成像模态方面开发可扩展的医学视觉基础模型的扩展行为。为了支持可扩展的预训练，我们引入了BioVFM-21M，这是一个大规模的生物医学图像数据集，涵盖了多种生物医学图像模态和解剖结构。我们观察到，扩展确实提供了益处，但这些益处在不同任务中有所不同。进一步的分析揭示了一些与扩展益处相关的因素。最后，我们提出了BioVFM，这是一个在2100万生物医学图像上预训练的大规模医学视觉基础模型，它在12项医学基准测试中优于之前的最佳基础模型。我们的结果表明，虽然扩展有助于提高性能，但任务特征、数据多样性、预训练方法和计算效率仍然是开发可扩展的医学基础模型的关键考虑因素。 

---
# Neural Video Compression using 2D Gaussian Splatting 

**Title (ZH)**: 基于2D高斯点绘制的神经视频压缩 

**Authors**: Lakshya Gupta, Imran N. Junejo  

**Link**: [PDF](https://arxiv.org/pdf/2505.09324)  

**Abstract**: The computer vision and image processing research community has been involved in standardizing video data communications for the past many decades, leading to standards such as AVC, HEVC, VVC, AV1, AV2, etc. However, recent groundbreaking works have focused on employing deep learning-based techniques to replace the traditional video codec pipeline to a greater affect. Neural video codecs (NVC) create an end-to-end ML-based solution that does not rely on any handcrafted features (motion or edge-based) and have the ability to learn content-aware compression strategies, offering better adaptability and higher compression efficiency than traditional methods. This holds a great potential not only for hardware design, but also for various video streaming platforms and applications, especially video conferencing applications such as MS-Teams or Zoom that have found extensive usage in classrooms and workplaces. However, their high computational demands currently limit their use in real-time applications like video conferencing. To address this, we propose a region-of-interest (ROI) based neural video compression model that leverages 2D Gaussian Splatting. Unlike traditional codecs, 2D Gaussian Splatting is capable of real-time decoding and can be optimized using fewer data points, requiring only thousands of Gaussians for decent quality outputs as opposed to millions in 3D scenes. In this work, we designed a video pipeline that speeds up the encoding time of the previous Gaussian splatting-based image codec by 88% by using a content-aware initialization strategy paired with a novel Gaussian inter-frame redundancy-reduction mechanism, enabling Gaussian splatting to be used for a video-codec solution, the first of its kind solution in this neural video codec space. 

**Abstract (ZH)**: 基于区域兴趣的2D高斯斑点神经视频压缩模型 

---
# Toward Fair Federated Learning under Demographic Disparities and Data Imbalance 

**Title (ZH)**: 面向人口统计差异和数据不平衡的公平联邦学习 

**Authors**: Qiming Wu, Siqi Li, Doudou Zhou, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09295)  

**Abstract**: Ensuring fairness is critical when applying artificial intelligence to high-stakes domains such as healthcare, where predictive models trained on imbalanced and demographically skewed data risk exacerbating existing disparities. Federated learning (FL) enables privacy-preserving collaboration across institutions, but remains vulnerable to both algorithmic bias and subgroup imbalance - particularly when multiple sensitive attributes intersect. We propose FedIDA (Fed erated Learning for Imbalance and D isparity A wareness), a framework-agnostic method that combines fairness-aware regularization with group-conditional oversampling. FedIDA supports multiple sensitive attributes and heterogeneous data distributions without altering the convergence behavior of the underlying FL algorithm. We provide theoretical analysis establishing fairness improvement bounds using Lipschitz continuity and concentration inequalities, and show that FedIDA reduces the variance of fairness metrics across test sets. Empirical results on both benchmark and real-world clinical datasets confirm that FedIDA consistently improves fairness while maintaining competitive predictive performance, demonstrating its effectiveness for equitable and privacy-preserving modeling in healthcare. The source code is available on GitHub. 

**Abstract (ZH)**: 确保公平性是将人工智能应用于高 stakes 领域（如医疗保健）时的关键，这些领域中的预测模型若基于不平衡和人口结构偏差的数据进行训练，可能会加剧现有的不平等。联邦学习（FL）可在机构间实现隐私保护的合作，但仍易受算法偏见和子组不平衡的影响，特别是在多个敏感属性相交的情况下。我们提出了一种名为 FedIDA（联邦学习中的不平衡与不平等意识）的框架无损方法，该方法结合了公平性意识正则化与分组条件过采样。FedIDA 支持多个敏感属性和异质数据分布，而不改变基础 FL 算法的收敛行为。我们通过 Lipschitz 连续性和集中不等式提供了理论分析，确立了公平性改进的界限，并证明 FedIDA 降低了公平性度量在测试集上的方差。基准数据集和实际临床数据集上的实证结果确认，FedIDA 一致地提高了公平性同时保持了竞争力的预测性能，展示了其在医疗保健中实现公平性和隐私保护建模的有效性。源代码可在 GitHub 上获取。 

---
# MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning 

**Title (ZH)**: MetaUAS: 通用异常分割的一提示元学习 

**Authors**: Bin-Bin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09265)  

**Abstract**: Zero- and few-shot visual anomaly segmentation relies on powerful vision-language models that detect unseen anomalies using manually designed textual prompts. However, visual representations are inherently independent of language. In this paper, we explore the potential of a pure visual foundation model as an alternative to widely used vision-language models for universal visual anomaly segmentation. We present a novel paradigm that unifies anomaly segmentation into change segmentation. This paradigm enables us to leverage large-scale synthetic image pairs, featuring object-level and local region changes, derived from existing image datasets, which are independent of target anomaly datasets. We propose a one-prompt Meta-learning framework for Universal Anomaly Segmentation (MetaUAS) that is trained on this synthetic dataset and then generalizes well to segment any novel or unseen visual anomalies in the real world. To handle geometrical variations between prompt and query images, we propose a soft feature alignment module that bridges paired-image change perception and single-image semantic segmentation. This is the first work to achieve universal anomaly segmentation using a pure vision model without relying on special anomaly detection datasets and pre-trained visual-language models. Our method effectively and efficiently segments any anomalies with only one normal image prompt and enjoys training-free without guidance from language. Our MetaUAS significantly outperforms previous zero-shot, few-shot, and even full-shot anomaly segmentation methods. The code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 纯视觉基础模型赋能通用视觉异常分割：无需依赖特殊异常检测数据集和预训练视觉语言模型的通用异常分割新范式 

---
# Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt 

**Title (ZH)**: 仅凭一张正常图像提示学习检测多类异常 

**Authors**: Bin-Bin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09264)  

**Abstract**: Unsupervised reconstruction networks using self-attention transformers have achieved state-of-the-art performance for multi-class (unified) anomaly detection with a single model. However, these self-attention reconstruction models primarily operate on target features, which may result in perfect reconstruction for both normal and anomaly features due to high consistency with context, leading to failure in detecting anomalies. Additionally, these models often produce inaccurate anomaly segmentation due to performing reconstruction in a low spatial resolution latent space. To enable reconstruction models enjoying high efficiency while enhancing their generalization for unified anomaly detection, we propose a simple yet effective method that reconstructs normal features and restores anomaly features with just One Normal Image Prompt (OneNIP). In contrast to previous work, OneNIP allows for the first time to reconstruct or restore anomalies with just one normal image prompt, effectively boosting unified anomaly detection performance. Furthermore, we propose a supervised refiner that regresses reconstruction errors by using both real normal and synthesized anomalous images, which significantly improves pixel-level anomaly segmentation. OneNIP outperforms previous methods on three industry anomaly detection benchmarks: MVTec, BTAD, and VisA. The code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 使用自注意变换器的无监督重建网络已实现单一模型在多类综合异常检测中的最先进性能。然而，这些自注意重建模型主要处理目标特征，由于上下文一致性高，可能导致正常和异常特征的完美重建，从而导致异常检测失败。此外，这些模型因在低空间分辨率的潜在空间中执行重建而经常产生不准确的异常分割。为使重建模型既保持高效率又能增强其统一异常检测的泛化能力，我们提出了一种简单有效的“一个正常图像提示”(OneNIP) 方法，该方法仅通过一个正常图像提示重建正常特征并恢复异常特征。与先前的工作不同，OneNIP 允许以史无前例的方式仅通过一个正常图像提示重建或恢复异常，显著提升了统一异常检测性能。此外，我们提出了一种监督修整器，通过同时使用真实正常图像和合成异常图像回归重建误差，显著提高了像素级异常分割。OneNIP 在 MVTec、BTAD 和 VisA 三个工业异常检测基准数据集上表现出色。相关代码和预训练模型可在以下链接获得。 

---
# Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation 

**Title (ZH)**: 基于异常驱动的少样本生成方法用于异常分类和分割 

**Authors**: Guan Gui, Bin-Bin Gao, Jun Liu, Chengjie Wang, Yunsheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09263)  

**Abstract**: Anomaly detection is a practical and challenging task due to the scarcity of anomaly samples in industrial inspection. Some existing anomaly detection methods address this issue by synthesizing anomalies with noise or external data. However, there is always a large semantic gap between synthetic and real-world anomalies, resulting in weak performance in anomaly detection. To solve the problem, we propose a few-shot Anomaly-driven Generation (AnoGen) method, which guides the diffusion model to generate realistic and diverse anomalies with only a few real anomalies, thereby benefiting training anomaly detection models. Specifically, our work is divided into three stages. In the first stage, we learn the anomaly distribution based on a few given real anomalies and inject the learned knowledge into an embedding. In the second stage, we use the embedding and given bounding boxes to guide the diffusion model to generate realistic and diverse anomalies on specific objects (or textures). In the final stage, we propose a weakly-supervised anomaly detection method to train a more powerful model with generated anomalies. Our method builds upon DRAEM and DesTSeg as the foundation model and conducts experiments on the commonly used industrial anomaly detection dataset, MVTec. The experiments demonstrate that our generated anomalies effectively improve the model performance of both anomaly classification and segmentation tasks simultaneously, \eg, DRAEM and DseTSeg achieved a 5.8\% and 1.5\% improvement in AU-PR metric on segmentation task, respectively. The code and generated anomalous data are available at this https URL. 

**Abstract (ZH)**: 基于少量真实异常样本的自驱动生成 Few-Shot Anomaly-Driven Generation (AnoGen) 方法研究 

---
# EDBench: Large-Scale Electron Density Data for Molecular Modeling 

**Title (ZH)**: EDBench: 大规模电子密度数据集用于分子建模 

**Authors**: Hongxin Xiang, Ke Li, Mingquan Liu, Zhixiang Cheng, Bin Yao, Wenjie Du, Jun Xia, Li Zeng, Xin Jin, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.09262)  

**Abstract**: Existing molecular machine learning force fields (MLFFs) generally focus on the learning of atoms, molecules, and simple quantum chemical properties (such as energy and force), but ignore the importance of electron density (ED) $\rho(r)$ in accurately understanding molecular force fields (MFFs). ED describes the probability of finding electrons at specific locations around atoms or molecules, which uniquely determines all ground state properties (such as energy, molecular structure, etc.) of interactive multi-particle systems according to the Hohenberg-Kohn theorem. However, the calculation of ED relies on the time-consuming first-principles density functional theory (DFT) which leads to the lack of large-scale ED data and limits its application in MLFFs. In this paper, we introduce EDBench, a large-scale, high-quality dataset of ED designed to advance learning-based research at the electronic scale. Built upon the PCQM4Mv2, EDBench provides accurate ED data, covering 3.3 million molecules. To comprehensively evaluate the ability of models to understand and utilize electronic information, we design a suite of ED-centric benchmark tasks spanning prediction, retrieval, and generation. Our evaluation on several state-of-the-art methods demonstrates that learning from EDBench is not only feasible but also achieves high accuracy. Moreover, we show that learning-based method can efficiently calculate ED with comparable precision while significantly reducing the computational cost relative to traditional DFT calculations. All data and benchmarks from EDBench will be freely available, laying a robust foundation for ED-driven drug discovery and materials science. 

**Abstract (ZH)**: 现有的分子机器学习力场（MLFF）通常集中于原子、分子以及简单量子化学性质（如能量和力）的学习，而忽视了电子密度（ED）$\rho(r)$在准确理解分子力场（MFF）中的重要性。ED描述了在原子或分子周围特定位置找到电子的概率，根据霍恩贝格-考恩定理，ED唯一确定了相互作用多粒子系统的所有基态性质（如能量、分子结构等）。然而，ED的计算依赖于耗时的原始原理密度泛函理论（DFT），这导致了大规模ED数据的缺乏，限制了其在MLFF中的应用。本文介绍了EDBench，一个大规模、高质量的ED数据集，旨在推动电子尺度的学习研究。基于PCQM4Mv2构建，EDBench提供了330万分子的精确ED数据。为了全面评估模型理解并利用电子信息的能力，我们设计了一系列以ED为中心的基准任务，涵盖预测、检索和生成。在几种前沿方法的评估中，我们证明从EDBench学习不仅是可行的，而且具有很高的准确性。此外，我们还展示了基于学习的方法可以高效地计算ED，精度与传统DFT计算相当，同时显著降低了计算成本。EDBench的所有数据和基准将免费提供，为基于ED的药物发现和材料科学奠定坚实的基础。 

---
# Focus, Merge, Rank: Improved Question Answering Based on Semi-structured Knowledge Bases 

**Title (ZH)**: 聚焦、合并、排序：基于半结构化知识库的改进问答系统 

**Authors**: Derian Boer, Stephen Roth, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2505.09246)  

**Abstract**: In many real-world settings, machine learning models and interactive systems have access to both structured knowledge, e.g., knowledge graphs or tables, and unstructured content, e.g., natural language documents. However, most rely on either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking unstructured content to nodes within structured data, thereby enabling new strategies for knowledge access and use. In this work, we present FocusedRetriever, a modular SKB-based framework for multi-hop question answering. It integrates components (VSS-based entity search, LLM-based generation of Cypher queries and pairwise re-ranking) in a way that enables it to outperform state-of-the-art methods across all three STaRK benchmark test sets, covering diverse domains and multiple performance metrics. The average first-hit rate exceeds that of the second-best method by 25.7%. FocusedRetriever leverages (1) the capacity of Large Language Models (LLMs) to extract relational facts and entity attributes from unstructured text, (2) node set joins to filter answer candidates based on these extracted triplets and constraints, (3) vector similarity search to retrieve and rank relevant unstructured content, and (4) the contextual capabilities of LLMs to finally rank the top-k answers. For generality, we only incorporate base LLMs in FocusedRetriever in our evaluation. However, our analysis of intermediate results highlights several opportunities for further upgrades including finetuning. The source code is publicly available at this https URL . 

**Abstract (ZH)**: 半结构化知识库中的多跳问答框架：FocusRetriever 

---
# Educational impacts of generative artificial intelligence on learning and performance of engineering students in China 

**Title (ZH)**: 生成式人工智能对中国工程学生学习与performance影响的研究 

**Authors**: Lei Fan, Kunyang Deng, Fangxue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09208)  

**Abstract**: With the rapid advancement of generative artificial intelligence(AI), its potential applications in higher education have attracted significant attention. This study investigated how 148 students from diverse engineering disciplines and regions across China used generative AI, focusing on its impact on their learning experience and the opportunities and challenges it poses in engineering education. Based on the surveyed data, we explored four key areas: the frequency and application scenarios of AI use among engineering students, its impact on students' learning and performance, commonly encountered challenges in using generative AI, and future prospects for its adoption in engineering education. The results showed that more than half of the participants reported a positive impact of generative AI on their learning efficiency, initiative, and creativity, with nearly half believing it also enhanced their independent thinking. However, despite acknowledging improved study efficiency, many felt their actual academic performance remained largely unchanged and expressed concerns about the accuracy and domain-specific reliability of generative AI. Our findings provide a first-hand insight into the current benefits and challenges generative AI brings to students, particularly Chinese engineering students, while offering several recommendations, especially from the students' perspective, for effectively integrating generative AI into engineering education. 

**Abstract (ZH)**: 随着生成式人工智能的迅速发展，其在高等教育中的潜在应用引起了广泛关注。本研究调查了来自中国不同工程学科和地区共148名学生的生成式人工智能使用情况，重点关注其对学生学习体验的影响以及在工程教育中带来的机遇和挑战。基于调查数据，我们探讨了四个关键领域：工程学生使用人工智能的频率和应用场景、其对学生学习和表现的影响、使用生成式人工智能时遇到的常见挑战以及该技术在未来工程教育中的应用前景。研究结果显示，超过一半的参与者报告称生成式人工智能对其学习效率、积极性和创造性产生了积极影响，近半数学生认为它还增强了他们的独立思考能力。尽管许多学生认可了学习效率的提升，但仍有学生担心生成式人工智能的准确性和学科特定的可靠性并没有显著变化。我们的研究提供了关于生成式人工智能当前对学生，特别是中国工程学生带来的益处和挑战的第一手见解，并从学生视角提出了几条关于如何有效将生成式人工智能融入工程教育的建议。 

---
# InvDesFlow-AL: Active Learning-based Workflow for Inverse Design of Functional Materials 

**Title (ZH)**: 基于主动学习的功能材料逆向设计工作流：InvDesFlow-AL 

**Authors**: Xiao-Qi Han, Peng-Jie Guo, Ze-Feng Gao, Hao Sun, Zhong-Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09203)  

**Abstract**: Developing inverse design methods for functional materials with specific properties is critical to advancing fields like renewable energy, catalysis, energy storage, and carbon capture. Generative models based on diffusion principles can directly produce new materials that meet performance constraints, thereby significantly accelerating the material design process. However, existing methods for generating and predicting crystal structures often remain limited by low success rates. In this work, we propose a novel inverse material design generative framework called InvDesFlow-AL, which is based on active learning strategies. This framework can iteratively optimize the material generation process to gradually guide it towards desired performance characteristics. In terms of crystal structure prediction, the InvDesFlow-AL model achieves an RMSE of 0.0423 Å, representing an 32.96% improvement in performance compared to exsisting generative models. Additionally, InvDesFlow-AL has been successfully validated in the design of low-formation-energy and low-Ehull materials. It can systematically generate materials with progressively lower formation energies while continuously expanding the exploration across diverse chemical spaces. These results fully demonstrate the effectiveness of the proposed active learning-driven generative model in accelerating material discovery and inverse design. To further prove the effectiveness of this method, we took the search for BCS superconductors under ambient pressure as an example explored by InvDesFlow-AL. As a result, we successfully identified Li\(_2\)AuH\(_6\) as a conventional BCS superconductor with an ultra-high transition temperature of 140 K. This discovery provides strong empirical support for the application of inverse design in materials science. 

**Abstract (ZH)**: 开发具有特定性能功能材料的逆向设计方法对于推动可再生能源、催化、能量存储和碳捕获等领域的发展至关重要。基于扩散原理的生成模型可以直接生成满足性能约束的新材料，从而显著加速材料设计过程。然而，现有结构生成和预测方法往往受限于较低的成功率。在这项工作中，我们提出了一种基于主动学习策略的新型逆向材料设计生成框架InvDesFlow-AL。该框架可以迭代优化材料生成过程，逐步引导其向所需的性能特性发展。在晶体结构预测方面，InvDesFlow-AL模型实现了0.0423 Å的RMSE，相比于现有生成模型，性能提高了32.96%。此外，InvDesFlow-AL已在低形成能和低Ehull材料的设计中成功验证。它可以系统地生成形成能逐步降低的材料，同时不断扩展对不同化学空间的探索。这些结果充分证明了所提出的主动学习驱动生成模型在加速材料发现和逆向设计方面的有效性。为了进一步证明该方法的有效性，我们通过InvDesFlow-AL探索了在常压下寻找BCS超导体的例子，成功识别出Li<sub>2</sub>AuH<sub>6</sub>作为具有超高温跃迁温度140 K的常规BCS超导体。这一发现为逆向设计在材料科学中的应用提供了强有力的实证支持。

开发具有特定性能的功能材料的逆向设计方法对推动可再生能源、催化、能源存储和碳捕捉等领域的发展至关重要。基于扩散原理的生成模型可以直接生成满足性能约束的新材料，从而显著加速材料设计过程。然而，现有方法在生成和预测晶体结构方面的成功率往往受限。本文提出了一种基于主动学习策略的新型逆向材料设计生成框架InvDesFlow-AL。该框架能够迭代优化材料生成过程，逐步引导其朝向所需性能特征。在晶体结构预测方面，InvDesFlow-AL模型的RMSE达到0.0423 Å，相比现有生成模型提高了32.96%。此外，InvDesFlow-AL已成功应用于低形成能和低Ehull材料的设计中。它能够系统地生成形成能逐步降低的材料，同时不断扩大对多样化化学空间的探索。这些结果充分展示了所提出的主动学习驱动生成模型在加速材料发现和逆向设计方面的有效性。为证明该方法的有效性，我们通过InvDesFlow-AL探索了常压下寻找BCS超导体的例子，成功识别出Li<sub>2</sub>AuH<sub>6</sub>作为具有超高温跃迁温度140 K的常规BCS超导体。这一发现为逆向设计在材料科学中的应用提供了强有力的实证支持。 

---
# DRRNet: Macro-Micro Feature Fusion and Dual Reverse Refinement for Camouflaged Object Detection 

**Title (ZH)**: DRRNet：宏观-微观特征融合与双重反向精炼在伪装目标检测中的应用 

**Authors**: Jianlin Sun, Xiaolin Fang, Juwei Guan, Dongdong Gui, Teqi Wang, Tongxin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09168)  

**Abstract**: The core challenge in Camouflage Object Detection (COD) lies in the indistinguishable similarity between targets and backgrounds in terms of color, texture, and shape. This causes existing methods to either lose edge details (such as hair-like fine structures) due to over-reliance on global semantic information or be disturbed by similar backgrounds (such as vegetation patterns) when relying solely on local features. We propose DRRNet, a four-stage architecture characterized by a "context-detail-fusion-refinement" pipeline to address these issues. Specifically, we introduce an Omni-Context Feature Extraction Module to capture global camouflage patterns and a Local Detail Extraction Module to supplement microstructural information for the full-scene context module. We then design a module for forming dual representations of scene understanding and structural awareness, which fuses panoramic features and local features across various scales. In the decoder, we also introduce a reverse refinement module that leverages spatial edge priors and frequency-domain noise suppression to perform a two-stage inverse refinement of the output. By applying two successive rounds of inverse refinement, the model effectively suppresses background interference and enhances the continuity of object boundaries. Experimental results demonstrate that DRRNet significantly outperforms state-of-the-art methods on benchmark datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 伪装目标检测（COD）的核心挑战在于目标与背景在颜色、纹理和形状方面的难以区分的相似性。这导致现有方法要么由于过度依赖全局语义信息而丢失边缘细节（如发丝般的微细结构），要么由于仅依赖局部特征而受到类似背景（如植被模式）的干扰。我们提出了一种名为DRRNet的四阶段架构，该架构通过“上下文-细节-融合-精炼”管道来解决这些问题。具体而言，我们引入了一种全方位上下文特征提取模块以捕获全局伪装模式，并引入了一种局部细节提取模块以补充全场景上下文模块的微细结构信息。我们还设计了一种模块以形成场景理解和结构意识的双表示，并融合不同尺度的全景特征和局部特征。在解码器中，我们还引入了一种逆精炼模块，利用空间边缘先验和频域噪声抑制来执行输出的两级逆精炼。通过应用两轮逆精炼，模型有效地抑制了背景干扰并增强了对象边界的连续性。实验结果表明，DRRNet在基准数据集上显著优于现有方法。我们的代码可在以下链接获取。 

---
# An Initial Exploration of Default Images in Text-to-Image Generation 

**Title (ZH)**: 文本生成图像中默认图片的初步探索 

**Authors**: Hannu Simonen, Atte Kiviniemi, Jonas Oppenlaender  

**Link**: [PDF](https://arxiv.org/pdf/2505.09166)  

**Abstract**: In the creative practice of text-to-image generation (TTI), images are generated from text prompts. However, TTI models are trained to always yield an output, even if the prompt contains unknown terms. In this case, the model may generate what we call "default images": images that closely resemble each other across many unrelated prompts. We argue studying default images is valuable for designing better solutions for TTI and prompt engineering. In this paper, we provide the first investigation into default images on Midjourney, a popular image generator. We describe our systematic approach to create input prompts triggering default images, and present the results of our initial experiments and several small-scale ablation studies. We also report on a survey study investigating how default images affect user satisfaction. Our work lays the foundation for understanding default images in TTI and highlights challenges and future research directions. 

**Abstract (ZH)**: 在文本生成图像的创作实践中，图像根据文本提示生成。然而，TTI模型在提示中包含未知术语时仍会被训练产出输出，这可能导致生成我们称之为“默认图像”的相似图像。我们主张研究默认图像对于设计更好的TTI解决方案和提示工程具有重要意义。在本文中，我们首次针对Midjourney这一流行的图像生成器进行默认图像的研究。我们描述了一种系统的方法以触发默认图像的输入提示，并展示了初步实验及若干小型消融研究的结果。我们还报告了一项关于默认图像如何影响用户满意度的调查研究。我们的工作为理解TTI中的默认图像奠定了基础，并指出了挑战和未来的研究方向。 

---
# A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning 

**Title (ZH)**: 基于对比学习和掩蔽自编码器的多任务基础模型无线信道表示 

**Authors**: Berkay Guler, Giovanni Geraci, Hamid Jafarkhani  

**Link**: [PDF](https://arxiv.org/pdf/2505.09160)  

**Abstract**: Current applications of self-supervised learning to wireless channel representation often borrow paradigms developed for text and image processing, without fully addressing the unique characteristics and constraints of wireless communications. Aiming to fill this gap, we first propose WiMAE (Wireless Masked Autoencoder), a transformer-based encoder-decoder foundation model pretrained on a realistic open-source multi-antenna wireless channel dataset. Building upon this foundation, we develop ContraWiMAE, which enhances WiMAE by incorporating a contrastive learning objective alongside the reconstruction task in a unified multi-task framework. By warm-starting from pretrained WiMAE weights and generating positive pairs via noise injection, the contrastive component enables the model to capture both structural and discriminative features, enhancing representation quality beyond what reconstruction alone can achieve. Through extensive evaluation on unseen scenarios, we demonstrate the effectiveness of both approaches across multiple downstream tasks, with ContraWiMAE showing further improvements in linear separability and adaptability in diverse wireless environments. Comparative evaluations against a state-of-the-art wireless channel foundation model confirm the superior performance and data efficiency of our models, highlighting their potential as powerful baselines for future research in self-supervised wireless channel representation learning. 

**Abstract (ZH)**: 当前自监督学习在无线信道表示中的应用往往借鉴了文本和图像处理领域的 paradigms，而未能充分考虑无线通信的独特特性和约束。为了填补这一空白，我们首先提出了 WiMAE（无线掩码自编码器），这是一种基于变换器的编码器-解码器基础模型，预训练于一个现实的开源多天线无线信道数据集中。在此基础上，我们开发了 ContraWiMAE，通过在其统一的多任务框架中结合对比学习目标和重构任务来增强 WiMAE。基于预训练的 WiMAE 权重并通过噪声注入生成正样本对，对比学习组件使模型能够捕获结构化和区分性的特征，从而超越单纯重构所能实现的表示质量。通过在未见场景中的广泛评估，我们展示了这两种方法在多个下游任务中的有效性，ContraWiMAE 在线性可分性和多种无线环境下的适应性上进一步表现出改进。与最先进的无线信道基础模型的对比评估证实了我们模型的优越性能和高效性，突显了它们作为未来自监督无线信道表示学习研究中强大基线的潜力。 

---
# ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor 

**Title (ZH)**: ELIS: 效率高的LLM迭代调度系统，带有响应长度预测器 

**Authors**: Seungbeom Choi, Jeonghoe Goo, Eunjoo Jeon, Mingyu Yang, Minsung Jang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09142)  

**Abstract**: We propose ELIS, a serving system for Large Language Models (LLMs) featuring an Iterative Shortest Remaining Time First (ISRTF) scheduler designed to efficiently manage inference tasks with the shortest remaining tokens. Current LLM serving systems often employ a first-come-first-served scheduling strategy, which can lead to the "head-of-line blocking" problem. To overcome this limitation, it is necessary to predict LLM inference times and apply a shortest job first scheduling strategy. However, due to the auto-regressive nature of LLMs, predicting the inference latency is challenging. ELIS addresses this challenge by training a response length predictor for LLMs using the BGE model, an encoder-based state-of-the-art model. Additionally, we have devised the ISRTF scheduling strategy, an optimization of shortest remaining time first tailored to existing LLM iteration batching. To evaluate our work in an industrial setting, we simulate streams of requests based on our study of real-world user LLM serving trace records. Furthermore, we implemented ELIS as a cloud-native scheduler system on Kubernetes to evaluate its performance in production environments. Our experimental results demonstrate that ISRTF reduces the average job completion time by up to 19.6%. 

**Abstract (ZH)**: 我们提出ELIS，一种大型语言模型服务系统，具备迭代最短剩余时间优先(ISRTF)调度器，旨在高效管理剩余最短 tokens 的推理任务。当前大型语言模型服务系统常采用先到先服务的调度策略，这可能导致“线路头阻塞”问题。为克服此局限，需预测大型语言模型的推理时间并采用最短作业优先调度策略。但由于大型语言模型具有自回归特性，预测推理延迟颇具挑战性。ELIS通过使用基于编码器的前沿模型BGE训练响应长度预测器来应对这一挑战。此外，我们还设计了ISRTF调度策略，这是针对现有大型语言模型迭代批处理的最短剩余时间优先调度的优化。为了在工业环境中评估我们的工作，我们根据实际用户大型语言模型服务追踪记录模拟了请求流。进一步地，我们在Kubernetes上实现了ELIS，作为云原生调度系统，以评估其在生产环境中的性能。实验结果表明，ISRTF可将平均任务完成时间降低19.6%。 

---
# Fair Clustering via Alignment 

**Title (ZH)**: 公平聚类Via对齐 

**Authors**: Kunwoong Kim, Jihu Lee, Sangchul Park, Yongdai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.09131)  

**Abstract**: Algorithmic fairness in clustering aims to balance the proportions of instances assigned to each cluster with respect to a given sensitive attribute. While recently developed fair clustering algorithms optimize clustering objectives under specific fairness constraints, their inherent complexity or approximation often results in suboptimal clustering utility or numerical instability in practice. To resolve these limitations, we propose a new fair clustering algorithm based on a novel decomposition of the fair K-means clustering objective function. The proposed algorithm, called Fair Clustering via Alignment (FCA), operates by alternately (i) finding a joint probability distribution to align the data from different protected groups, and (ii) optimizing cluster centers in the aligned space. A key advantage of FCA is that it theoretically guarantees approximately optimal clustering utility for any given fairness level without complex constraints, thereby enabling high-utility fair clustering in practice. Experiments show that FCA outperforms existing methods by (i) attaining a superior trade-off between fairness level and clustering utility, and (ii) achieving near-perfect fairness without numerical instability. 

**Abstract (ZH)**: 聚类中的算法公平性旨在根据给定的敏感属性平衡分配给每个簇的实例比例。虽然最近开发的公平聚类算法在特定公平性约束下优化聚类目标，但其固有的复杂性或近似性往往导致实际中的聚类效果不佳或数值不稳定。为了解决这些局限性，我们提出了一种基于公平K均值聚类目标函数新型分解的新公平聚类算法。该算法称为对齐导向公平聚类（FCA），通过交替进行（i）找到一个联合概率分布以对齐不同受保护群组的数据，以及（ii）在对齐空间中优化簇中心，来进行操作。FCA的一个关键优势是，它理论上能够在任何给定的公平水平下保证近似最优的聚类效果，从而在实践中实现高效益的公平聚类。实验结果表明，FCA在（i）公平水平与聚类效果之间的权衡表现更优，以及（ii）实现接近完美的公平性同时避免数值不稳定方面优于现有方法。 

---
# WSCIF: A Weakly-Supervised Color Intelligence Framework for Tactical Anomaly Detection in Surveillance Keyframes 

**Title (ZH)**: WSCIF：一种弱监督颜色智能框架用于 surveillance 关键帧中的战术异常检测 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.09129)  

**Abstract**: The deployment of traditional deep learning models in high-risk security tasks in an unlabeled, data-non-exploitable video intelligence environment faces significant challenges. In this paper, we propose a lightweight anomaly detection framework based on color features for surveillance video clips in a high sensitivity tactical mission, aiming to quickly identify and interpret potential threat events under resource-constrained and data-sensitive conditions. The method fuses unsupervised KMeans clustering with RGB channel histogram modeling to achieve composite detection of structural anomalies and color mutation signals in key frames. The experiment takes an operation surveillance video occurring in an African country as a research sample, and successfully identifies multiple highly anomalous frames related to high-energy light sources, target presence, and reflective interference under the condition of no access to the original data. The results show that this method can be effectively used for tactical assassination warning, suspicious object screening and environmental drastic change monitoring with strong deployability and tactical interpretation value. The study emphasizes the importance of color features as low semantic battlefield signal carriers, and its battlefield intelligent perception capability will be further extended by combining graph neural networks and temporal modeling in the future. 

**Abstract (ZH)**: 传统的深度学习模型在无标签、数据不可利用的高风险安全任务视频智能环境中部署面临显著挑战。本文提出了一种基于颜色特征的轻量级异常检测框架，旨在在资源受限和数据敏感条件下，快速识别和解释关键帧中的潜在威胁事件。该方法将无监督KMeans聚类与RGB通道直方图建模融合，以实现结构异常和颜色突变信号的综合检测。实验以非洲某国的作战 surveillance 视频作为研究样本，在无法访问原始数据的情况下，成功识别了多个与高能光源、目标存在及反射干扰高度异常的帧。结果表明，该方法在战术暗杀预警、可疑目标筛查和环境急剧变化监控方面具有较强的部署能力和战术解释价值。研究强调了颜色特征作为低语义战场信号载体的重要性，并指出未来将通过结合图神经网络和时序建模进一步扩展其战场智能感知能力。 

---
# PreCare: Designing AI Assistants for Advance Care Planning (ACP) to Enhance Personal Value Exploration, Patient Knowledge, and Decisional Confidence 

**Title (ZH)**: PreCare: 设计AI辅助工具促进前期护理规划、提升个人价值观探索、患者知识及决策自信 

**Authors**: Yu Lun Hsu, Yun-Rung Chou, Chiao-Ju Chang, Yu-Cheng Chang, Zer-Wei Lee, Rokas Gipiškis, Rachel Li, Chih-Yuan Shih, Jen-Kuei Peng, Hsien-Liang Huang, Jaw-Shiun Tsai, Mike Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09115)  

**Abstract**: Advance Care Planning (ACP) allows individuals to specify their preferred end-of-life life-sustaining treatments before they become incapacitated by injury or terminal illness (e.g., coma, cancer, dementia). While online ACP offers high accessibility, it lacks key benefits of clinical consultations, including personalized value exploration, immediate clarification of decision consequences. To bridge this gap, we conducted two formative studies: 1) shadowed and interviewed 3 ACP teams consisting of physicians, nurses, and social workers (18 patients total), and 2) interviewed 14 users of ACP websites. Building on these insights, we designed PreCare in collaboration with 6 ACP professionals. PreCare is a website with 3 AI-driven assistants designed to guide users through exploring personal values, gaining ACP knowledge, and supporting informed decision-making. A usability study (n=12) showed that PreCare achieved a System Usability Scale (SUS) rating of excellent. A comparative evaluation (n=12) showed that PreCare's AI assistants significantly improved exploration of personal values, knowledge, and decisional confidence, and was preferred by 92% of participants. 

**Abstract (ZH)**: Advance Care Planning (ACP) 允许个体在因伤或末期疾病（如昏迷、癌症、痴呆）而无法行动之前，指定其所需的終末期生命维持治疗。虽然在线 ACP 提供了高 accesibility，但它缺乏临床咨询的关键优势，包括个性化价值探索和即时澄清决策后果。为了弥补这一差距，我们进行了两项形式研究：1）跟踪并采访了 3 支由医生、护士和社会工作者组成的 ACP 团队（总共 18 名患者）；2）采访了使用 ACP 网站的 14 名用户。基于这些见解，我们与 6 名 ACP 专业人士合作设计了 PreCare。PreCare 是一个网站，包含 3 个 AI 驱动的助手，旨在引导用户探索个人价值观、获取 ACP 知识，并支持基于信息的决策。可用性研究（n=12）显示，PreCare 达到了优秀系统可用性量表 (SUS) 评分。对比评估（n=12）显示，PreCare 的 AI 助手显著提高了个人价值观探索、知识获取和决策信心，并且 92% 的参与者更偏向于 PreCare。 

---
# Air-Ground Collaboration for Language-Specified Missions in Unknown Environments 

**Title (ZH)**: 未知环境中的语言指定任务空地协作 

**Authors**: Fernando Cladera, Zachary Ravichandran, Jason Hughes, Varun Murali, Carlos Nieto-Granda, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.09108)  

**Abstract**: As autonomous robotic systems become increasingly mature, users will want to specify missions at the level of intent rather than in low-level detail. Language is an expressive and intuitive medium for such mission specification. However, realizing language-guided robotic teams requires overcoming significant technical hurdles. Interpreting and realizing language-specified missions requires advanced semantic reasoning. Successful heterogeneous robots must effectively coordinate actions and share information across varying viewpoints. Additionally, communication between robots is typically intermittent, necessitating robust strategies that leverage communication opportunities to maintain coordination and achieve mission objectives. In this work, we present a first-of-its-kind system where an unmanned aerial vehicle (UAV) and an unmanned ground vehicle (UGV) are able to collaboratively accomplish missions specified in natural language while reacting to changes in specification on the fly. We leverage a Large Language Model (LLM)-enabled planner to reason over semantic-metric maps that are built online and opportunistically shared between an aerial and a ground robot. We consider task-driven navigation in urban and rural areas. Our system must infer mission-relevant semantics and actively acquire information via semantic mapping. In both ground and air-ground teaming experiments, we demonstrate our system on seven different natural-language specifications at up to kilometer-scale navigation. 

**Abstract (ZH)**: 自主机器人系统日趋成熟后，用户将希望以意图而非低级细节来指定任务。自然语言是一个表达性和直观性的任务指定媒介。然而，实现语言引导的机器人团队需要克服重大的技术障碍。解释和实现语言指定的任务需要高级语义推理。成功的异构机器人必须有效地协调动作并分享信息，跨越不同视角。此外，机器人之间的通信通常断断续续，需要采用强大的策略，利用通信机会来维持协调并达成任务目标。在本工作中，我们提出了一种开创性系统，在该系统中，无人机(UAV)和地面机器人(UGV)能够协同完成自然语言指定的任务，并且能够在指定内容发生变化时实时响应。我们利用一个增强连通性的大规模语言模型(LLL)-启用规划器，在在线构建和机会性共享的语义-度量地图上进行推理。我们考虑城市和农村地区的任务导向导航。我们的系统必须推断与任务相关的信息，并通过语义映射主动获取信息。在地面和空地团队实验中，我们在多达千米尺度的导航中演示了七个不同自然语言规定的内容。 

---
# DPN-GAN: Inducing Periodic Activations in Generative Adversarial Networks for High-Fidelity Audio Synthesis 

**Title (ZH)**: DPN-GAN：在生成对抗网络中诱导周期激活以实现高保真音频合成 

**Authors**: Zeeshan Ahmad, Shudi Bao, Meng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09091)  

**Abstract**: In recent years, generative adversarial networks (GANs) have made significant progress in generating audio sequences. However, these models typically rely on bandwidth-limited mel-spectrograms, which constrain the resolution of generated audio sequences, and lead to mode collapse during conditional generation. To address this issue, we propose Deformable Periodic Network based GAN (DPN-GAN), a novel GAN architecture that incorporates a kernel-based periodic ReLU activation function to induce periodic bias in audio generation. This innovative approach enhances the model's ability to capture and reproduce intricate audio patterns. In particular, our proposed model features a DPN module for multi-resolution generation utilizing deformable convolution operations, allowing for adaptive receptive fields that improve the quality and fidelity of the synthetic audio. Additionally, we enhance the discriminator network using deformable convolution to better distinguish between real and generated samples, further refining the audio quality. We trained two versions of the model: DPN-GAN small (38.67M parameters) and DPN-GAN large (124M parameters). For evaluation, we use five different datasets, covering both speech synthesis and music generation tasks, to demonstrate the efficiency of the DPN-GAN. The experimental results demonstrate that DPN-GAN delivers superior performance on both out-of-distribution and noisy data, showcasing its robustness and adaptability. Trained across various datasets, DPN-GAN outperforms state-of-the-art GAN architectures on standard evaluation metrics, and exhibits increased robustness in synthesized audio. 

**Abstract (ZH)**: 基于可变形周期网络的生成对抗网络（DPN-GAN） 

---
# Human-like Cognitive Generalization for Large Models via Brain-in-the-loop Supervision 

**Title (ZH)**: 通过脑在环监督实现大型模型的人类认知泛化 

**Authors**: Jiaxuan Chen, Yu Qi, Yueming Wang, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09085)  

**Abstract**: Recent advancements in deep neural networks (DNNs), particularly large-scale language models, have demonstrated remarkable capabilities in image and natural language understanding. Although scaling up model parameters with increasing volume of training data has progressively improved DNN capabilities, achieving complex cognitive abilities - such as understanding abstract concepts, reasoning, and adapting to novel scenarios, which are intrinsic to human cognition - remains a major challenge. In this study, we show that brain-in-the-loop supervised learning, utilizing a small set of brain signals, can effectively transfer human conceptual structures to DNNs, significantly enhancing their comprehension of abstract and even unseen concepts. Experimental results further indicate that the enhanced cognitive capabilities lead to substantial performance gains in challenging tasks, including few-shot/zero-shot learning and out-of-distribution recognition, while also yielding highly interpretable concept representations. These findings highlight that human-in-the-loop supervision can effectively augment the complex cognitive abilities of large models, offering a promising pathway toward developing more human-like cognitive abilities in artificial systems. 

**Abstract (ZH)**: Recent advancements in深度神经网络（DNNs），尤其是大规模语言模型，在图像和自然语言理解方面展现了卓越的能力。尽管随着训练数据量的增加，增大模型参数逐步提升了DNN的能力，但在实现诸如理解抽象概念、推理和适应新颖场景等人类认知所固有的复杂认知能力方面，仍面临重大挑战。本研究显示，利用少量脑信号进行脑-机闭环监督学习，可以有效将人类的概念结构转移到DNN中，显著提升其对抽象甚至未见过的概念的理解。实验结果进一步表明，增强的认知能力在包括少样本/零样本学习和离分布识别在内的挑战性任务中带来了显著的性能增益，同时产生了高度可解释的概念表示。这些发现强调，人类在环监督可以有效增强大型模型的复杂认知能力，为开发更具人类认知能力的人工智能系统提供了有前景的道路。 

---
# CEC-Zero: Chinese Error Correction Solution Based on LLM 

**Title (ZH)**: CEC-Zero: 基于LLM的中文错误修正解决方案 

**Authors**: Sophie Zhang, Zhiming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.09082)  

**Abstract**: Recent advancements in large language models (LLMs) demonstrate exceptional Chinese text processing capabilities, particularly in Chinese Spelling Correction (CSC). While LLMs outperform traditional BERT-based models in accuracy and robustness, challenges persist in reliability and generalization. This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework enabling LLMs to self-correct through autonomous error strategy learning without external supervision. By integrating RL with LLMs' generative power, the method eliminates dependency on annotated data or auxiliary models. Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and superior cross-domain generalization, offering a scalable solution for reliability optimization in Chinese NLP applications. This breakthrough facilitates LLM deployment in practical Chinese text correction scenarios while establishing a new paradigm for self-improving language models. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）在中国文本处理能力，尤其是在中文拼写纠错（CSC）方面展现了卓越的能力。尽管LLMs在准确性和稳健性上优于传统的BERT基模型，但在可靠性和泛化能力方面仍存在挑战。本文提出了一种名为CEC-Zero的新型强化学习（RL）框架，该框架使LLMs能够通过自主错误策略学习自我纠正，而无需外部监督。通过将RL与LLMs的生成能力结合，该方法消除了对外标注数据或辅助模型的依赖。实验结果显示，增强RL的LLMs实现了行业可行的准确度和更好的跨域泛化能力，为中文自然语言处理应用中的可靠性优化提供了可扩展的解决方案。这一突破促进了LLMs在实际中文文本纠错场景中的部署，并为自我提升的语言模型建立了新的范式。 

---
# SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation 

**Title (ZH)**: SALM：一种由语言模型驱动的社交网络仿真多Agent框架 

**Authors**: Gaurav Koley  

**Link**: [PDF](https://arxiv.org/pdf/2505.09081)  

**Abstract**: Contemporary approaches to agent-based modeling (ABM) of social systems have traditionally emphasized rule-based behaviors, limiting their ability to capture nuanced dynamics by moving beyond predefined rules and leveraging contextual understanding from LMs of human social interaction. This paper presents SALM (Social Agent LM Framework), a novel approach for integrating language models (LMs) into social network simulation that achieves unprecedented temporal stability in multi-agent scenarios. Our primary contributions include: (1) a hierarchical prompting architecture enabling stable simulation beyond 4,000 timesteps while reducing token usage by 73%, (2) an attention-based memory system achieving 80% cache hit rates (95% CI [78%, 82%]) with sub-linear memory growth of 9.5%, and (3) formal bounds on personality stability. Through extensive validation against SNAP ego networks, we demonstrate the first LLM-based framework capable of modeling long-term social phenomena while maintaining empirically validated behavioral fidelity. 

**Abstract (ZH)**: 基于代理的建模（ABM）在社会系统中的当代方法传统上强调基于规则的行为，限制了它们通过超越预定义规则并利用人类社会互动语言模型的上下文理解来捕捉细腻动态的能力。本文提出了SALM（社会代理LM框架），这是一种将语言模型（LMs）集成到社会网络模拟中的新颖方法，实现了多代理场景中前所未有的时间稳定性。我们的主要贡献包括：（1）分层提示架构，使其能够在超过4000个时间步长的同时降低73%的令牌使用量，实现稳定模拟；（2）基于注意力的记忆系统，实现了80%的缓存命中率（95%置信区间为78%至82%），并具有亚线性内存增长，增长率为9.5%；以及（3）人格稳定性的形式边界。通过广泛验证against SNAP自网络，我们展示了第一个基于LLM的框架，能够在保持经验验证的行为保真度的同时建模长期社会现象。 

---
# Variational Prefix Tuning for Diverse and Accurate Code Summarization Using Pre-trained Language Models 

**Title (ZH)**: 基于变分前缀调优的多样性和准确的代码摘要预训练语言模型方法 

**Authors**: Junda Zhao, Yuliang Song, Eldan Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09062)  

**Abstract**: Recent advancements in source code summarization have leveraged transformer-based pre-trained models, including Large Language Models of Code (LLMCs), to automate and improve the generation of code summaries. However, existing methods often focus on generating a single high-quality summary for a given source code, neglecting scenarios where the generated summary might be inadequate and alternative options are needed. In this paper, we introduce Variational Prefix Tuning (VPT), a novel approach that enhances pre-trained models' ability to generate diverse yet accurate sets of summaries, allowing the user to choose the most suitable one for the given source code. Our method integrates a Conditional Variational Autoencoder (CVAE) framework as a modular component into pre-trained models, enabling us to model the distribution of observed target summaries and sample continuous embeddings to be used as prefixes to steer the generation of diverse outputs during decoding. Importantly, we construct our method in a parameter-efficient manner, eliminating the need for expensive model retraining, especially when using LLMCs. Furthermore, we employ a bi-criteria reranking method to select a subset of generated summaries, optimizing both the diversity and the accuracy of the options presented to users. We present extensive experimental evaluations using widely used datasets and current state-of-the-art pre-trained code summarization models to demonstrate the effectiveness of our approach and its adaptability across models. 

**Abstract (ZH)**: Recent advancements in源代码摘要生成采用基于变换器的预训练模型，包括代码大型语言模型（LLMCs），以自动化并提高代码摘要的生成质量。然而，现有方法往往专注于生成给定源代码的单个高质量摘要，忽视了生成摘要可能不足且需要替代选项的情形。在本文中，我们介绍了变分前缀调谐（VPT），这是一种新型方法，能够增强预训练模型生成多样且准确的摘要集的能力，允许用户选择最适合给定源代码的摘要。我们的方法通过将条件变分自编码器（CVAE）框架作为一个模块化组件集成到预训练模型中，能够建模观察到的目标摘要的分布，并采样连续嵌入作为前缀，在解码过程中引导生成多种多样的输出。重要的是，我们以参数高效的方式构建了该方法，避免了在使用LLMCs时昂贵的模型重新训练的需求。此外，我们采用双标准重排序方法从生成的摘要中选择子集，优化提供给用户的选项的多样性和准确性。我们使用广泛使用的数据集和当前最先进的预训练代码摘要模型进行了详尽的实验评估，以展示我们方法的有效性和跨模型的适应性。 

---
# RT-cache: Efficient Robot Trajectory Retrieval System 

**Title (ZH)**: RT-cache: 高效的机器人轨迹检索系统 

**Authors**: Owen Kwon, Abraham George, Alison Bartsch, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2505.09040)  

**Abstract**: This paper introduces RT-cache, a novel trajectorymemory pipeline that accelerates real-world robot inference by leveraging big-data retrieval and learning from experience. While modern Vision-Language-Action (VLA) models can handle diverse robotic tasks, they often incur high per-step inference costs, resulting in significant latency, sometimes minutes per task. In contrast, RT-cache stores a large-scale Memory of previously successful robot trajectories and retrieves relevant multistep motion snippets, drastically reducing inference overhead. By integrating a Memory Builder with a Trajectory Retrieval, we develop an efficient retrieval process that remains tractable even for extremely large datasets. RT-cache flexibly accumulates real-world experiences and replays them whenever the current scene matches past states, adapting quickly to new or unseen environments with only a few additional samples. Experiments on the Open-X Embodiment Dataset and other real-world data demonstrate that RT-cache completes tasks both faster and more successfully than a baseline lacking retrieval, suggesting a practical, data-driven solution for real-time manipulation. 

**Abstract (ZH)**: RT-cache：一种通过大数据检索和经验学习加速现实世界机器人推理的新轨迹记忆流水线 

---
# Tests as Prompt: A Test-Driven-Development Benchmark for LLM Code Generation 

**Title (ZH)**: 测试作为提示：一种面向测试驱动开发的LLM代码生成基准 

**Authors**: Yi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.09027)  

**Abstract**: We introduce WebApp1K, a novel benchmark for evaluating large language models (LLMs) in test-driven development (TDD) tasks, where test cases serve as both prompt and verification for code generation. Unlike traditional approaches relying on natural language prompts, our benchmark emphasizes the ability of LLMs to interpret and implement functionality directly from test cases, reflecting real-world software development practices. Comprising 1000 diverse challenges across 20 application domains, the benchmark evaluates LLMs on their ability to generate compact, functional code under the constraints of context length and multi-feature complexity. Our findings highlight instruction following and in-context learning as critical capabilities for TDD success, surpassing the importance of general coding proficiency or pretraining knowledge. Through comprehensive evaluation of 19 frontier models, we reveal performance bottlenecks, such as instruction loss in long prompts, and provide a detailed error analysis spanning multiple root causes. This work underscores the practical value of TDD-specific benchmarks and lays the foundation for advancing LLM capabilities in rigorous, application-driven coding scenarios. 

**Abstract (ZH)**: 我们介绍了一个新的基准WebApp1K，用于评估大型语言模型（LLMs）在测试驱动开发（TDD）任务中的性能，其中测试用例既作为代码生成的提示也作为验证依据。与依赖自然语言提示的传统方法不同，我们的基准测试强调LLMs直接从测试用例中解释和实现功能的能力，反映了实际软件开发实践。该基准包含来自20个应用领域的1000个多样化的挑战，评估LLMs在上下文长度和多特征复杂性约束下生成紧凑功能性代码的能力。我们的研究结果突显了指令遵循和上下文学习对于TDD成功至关重要，超过了一般编程能力和预训练知识的重要性。通过对19个前沿模型进行全面评估，我们揭示了性能瓶颈，如长提示中的指令损失，并提供了涵盖多个根本原因的详细错误分析。这项工作强调了TDD特定基准的实际价值，并为在严格的应用驱动编码场景中推进LLM能力奠定了基础。 

---
# Block-Biased Mamba for Long-Range Sequence Processing 

**Title (ZH)**: 块偏好黄鼠狼模型：长距序列处理 

**Authors**: Annan Yu, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2505.09022)  

**Abstract**: Mamba extends earlier state space models (SSMs) by introducing input-dependent dynamics, and has demonstrated strong empirical performance across a range of domains, including language modeling, computer vision, and foundation models. However, a surprising weakness remains: despite being built on architectures designed for long-range dependencies, Mamba performs poorly on long-range sequential tasks. Understanding and addressing this gap is important for improving Mamba's universality and versatility. In this work, we analyze Mamba's limitations through three perspectives: expressiveness, inductive bias, and training stability. Our theoretical results show how Mamba falls short in each of these aspects compared to earlier SSMs such as S4D. To address these issues, we propose $\text{B}_2\text{S}_6$, a simple extension of Mamba's S6 unit that combines block-wise selective dynamics with a channel-specific bias. We prove that these changes equip the model with a better-suited inductive bias and improve its expressiveness and stability. Empirically, $\text{B}_2\text{S}_6$ outperforms S4 and S4D on Long-Range Arena (LRA) tasks while maintaining Mamba's performance on language modeling benchmarks. 

**Abstract (ZH)**: Mamba通过引入输入依赖的动力学扩展了早期的状态空间模型（SSMs），并在语言建模、计算机视觉和基础模型等多种领域中表现出强大的实际性能。然而，一个令人惊讶的弱点仍然存在：尽管基于设计长范围依赖性的架构，Mamba在长范围序列任务上的表现不佳。理解并解决这一差距对于提高Mamba的通用性和灵活性至关重要。在本文中，我们从三个角度分析Mamba的局限性：表现力、归纳偏置、训练稳定性。我们的理论结果展示了与早期的SSMs如S4D相比，Mamba在这三个方面存在不足。为了应对这些问题，我们提出了一种B2S6模型，它是Mamba S6单元的简单扩展，结合了块级选择性动力学和通道特定偏置。我们证明这些改变使模型具有更适宜的归纳偏置，并提高了其表现力和稳定性。实验结果表明，B2S6在Long-Range Arena（LRA）任务中优于S4和S4D，在语言建模基准测试中保持了Mamba的性能。 

---
# AI-Mediated Code Comment Improvement 

**Title (ZH)**: AI介导的代码注释改进 

**Authors**: Maria Dhakal, Chia-Yi Su, Robert Wallace, Chris Fakhimi, Aakash Bansal, Toby Li, Yu Huang, Collin McMillan  

**Link**: [PDF](https://arxiv.org/pdf/2505.09021)  

**Abstract**: This paper describes an approach to improve code comments along different quality axes by rewriting those comments with customized Artificial Intelligence (AI)-based tools. We conduct an empirical study followed by grounded theory qualitative analysis to determine the quality axes to improve. Then we propose a procedure using a Large Language Model (LLM) to rewrite existing code comments along the quality axes. We implement our procedure using GPT-4o, then distil the results into a smaller model capable of being run in-house, so users can maintain data custody. We evaluate both our approach using GPT-4o and the distilled model versions. We show in an evaluation how our procedure improves code comments along the quality axes. We release all data and source code in an online repository for reproducibility. 

**Abstract (ZH)**: 本文描述了一种利用定制化人工智能工具重写代码注释以提高其在不同质量维度上的方法。我们通过实证研究结合扎根理论的定性分析确定了需提高的质量维度，然后提出了一种使用大规模语言模型（LLM）重写现有代码注释的程序。我们利用GPT-4o实现该程序，然后将其精简为可在内部运行的小型模型，以便用户保持数据控制权。我们分别评估了使用GPT-4o实现的方法及其精简模型版本。我们在评估中展示了该程序如何在不同质量维度上改进代码注释。我们将在一个在线仓库中发布所有数据和源代码，以实现可重复性。 

---
# Continual Reinforcement Learning via Autoencoder-Driven Task and New Environment Recognition 

**Title (ZH)**: 基于自动编码器驱动的任务和新环境识别的持续强化学习 

**Authors**: Zeki Doruk Erden, Donia Gasmi, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2505.09003)  

**Abstract**: Continual learning for reinforcement learning agents remains a significant challenge, particularly in preserving and leveraging existing information without an external signal to indicate changes in tasks or environments. In this study, we explore the effectiveness of autoencoders in detecting new tasks and matching observed environments to previously encountered ones. Our approach integrates policy optimization with familiarity autoencoders within an end-to-end continual learning system. This system can recognize and learn new tasks or environments while preserving knowledge from earlier experiences and can selectively retrieve relevant knowledge when re-encountering a known environment. Initial results demonstrate successful continual learning without external signals to indicate task changes or reencounters, showing promise for this methodology. 

**Abstract (ZH)**: 强化学习代理的持续学习仍然是一项显著的挑战，特别是在无需外部信号指示任务或环境变化的情况下保留和利用现有信息。在这项研究中，我们探索了自编码器在检测新任务和将观察到的环境匹配到先前遇到的环境方面的有效性。我们的方法将策略优化与熟悉度自编码器整合到端到端的持续学习系统中。该系统能够在识别和学习新任务或环境的同时保留早期经验的知识，并在重新遇到已知环境时有选择性地检索相关知识。初步结果表明，在无需外部信号指示任务变化或重新遇到的情况下，可以实现成功的持续学习，展示了该方法的潜力。 

---
# GPML: Graph Processing for Machine Learning 

**Title (ZH)**: GPML: 图处理在机器学习中的应用 

**Authors**: Majed Jaber, Julien Michel, Nicolas Boutry, Pierre Parrend  

**Link**: [PDF](https://arxiv.org/pdf/2505.08964)  

**Abstract**: The dramatic increase of complex, multi-step, and rapidly evolving attacks in dynamic networks involves advanced cyber-threat detectors. The GPML (Graph Processing for Machine Learning) library addresses this need by transforming raw network traffic traces into graph representations, enabling advanced insights into network behaviors. The library provides tools to detect anomalies in interaction and community shifts in dynamic networks. GPML supports community and spectral metrics extraction, enhancing both real-time detection and historical forensics analysis. This library supports modern cybersecurity challenges with a robust, graph-based approach. 

**Abstract (ZH)**: 动态网络中复杂、多步且快速演变的攻击激增 necessitates 高级网络威胁检测器。GPML（基于图的机器学习处理库）通过将原始网络流量轨迹转换为图表示，以应对这一需求，从而提供对网络行为的高级洞察。该库提供了检测动态网络中交互异常和社区转移的工具。GPML 支持社区和谱度量的提取，增强实时检测和历史取证分析能力。该库采用稳健的图基方法应对现代网络安全挑战。 

---
# Tracing the Invisible: Understanding Students' Judgment in AI-Supported Design Work 

**Title (ZH)**: 追踪无形之物：理解学生在AI支持设计工作中 的判断 

**Authors**: Suchismita Naik, Prakash Shukla, Ike Obi, Jessica Backus, Nancy Rasche, Paul Parsons  

**Link**: [PDF](https://arxiv.org/pdf/2505.08939)  

**Abstract**: As generative AI tools become integrated into design workflows, students increasingly engage with these tools not just as aids, but as collaborators. This study analyzes reflections from 33 student teams in an HCI design course to examine the kinds of judgments students make when using AI tools. We found both established forms of design judgment (e.g., instrumental, appreciative, quality) and emergent types: agency-distribution judgment and reliability judgment. These new forms capture how students negotiate creative responsibility with AI and assess the trustworthiness of its outputs. Our findings suggest that generative AI introduces new layers of complexity into design reasoning, prompting students to reflect not only on what AI produces, but also on how and when to rely on it. By foregrounding these judgments, we offer a conceptual lens for understanding how students engage in co-creative sensemaking with AI in design contexts. 

**Abstract (ZH)**: 随着生成式AI工具融入设计工作流程，学生越来越多地将这些工具视为合作伙伴而非仅仅是辅助工具。本研究分析了人机交互设计课程中33个学生团队的反思，以探讨学生在使用AI工具时所做的各种判断。我们发现了既有形式的设计判断（如工具性判断、欣赏性判断、质量判断）以及新兴类型：代理权分配判断和可靠性判断。这些新形式捕捉了学生与AI协商创意责任以及评估其输出可信度的方式。我们的研究结果表明，生成式AI为设计推理引入了新的复杂层面，促使学生不仅要反思AI产生什么，还要思考何时以及如何依赖它。通过突出这些判断，我们提供了一个概念框架，帮助理解学生在设计情境中如何与AI进行共创的意义建构。 

---
# Template-Guided Reconstruction of Pulmonary Segments with Neural Implicit Functions 

**Title (ZH)**: 基于模板引导的肺段重建：神经隐式函数方法 

**Authors**: Kangxian Xie, Yufei Zhu, Kaiming Kuang, Li Zhang, Hongwei Bran Li, Mingchen Gao, Jiancheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08919)  

**Abstract**: High-quality 3D reconstruction of pulmonary segments plays a crucial role in segmentectomy and surgical treatment planning for lung cancer. Due to the resolution requirement of the target reconstruction, conventional deep learning-based methods often suffer from computational resource constraints or limited granularity. Conversely, implicit modeling is favored due to its computational efficiency and continuous representation at any resolution. We propose a neural implicit function-based method to learn a 3D surface to achieve anatomy-aware, precise pulmonary segment reconstruction, represented as a shape by deforming a learnable template. Additionally, we introduce two clinically relevant evaluation metrics to assess the reconstruction comprehensively. Further, due to the absence of publicly available shape datasets to benchmark reconstruction algorithms, we developed a shape dataset named Lung3D, including the 3D models of 800 labeled pulmonary segments and the corresponding airways, arteries, veins, and intersegmental veins. We demonstrate that the proposed approach outperforms existing methods, providing a new perspective for pulmonary segment reconstruction. Code and data will be available at this https URL. 

**Abstract (ZH)**: 高质量的肺段三维重建在肺段切除及肺癌手术治疗规划中发挥着关键作用。由于重建所需的分辨率要求，传统的基于深度学习的方法往往受到计算资源限制或粒度有限的约束。相比之下，隐式建模由于其计算效率和任意分辨率下的连续表示而受到青睐。我们提出了一种基于神经隐式函数的方法，通过变形可学习模板来学习一个3D表面，以实现解剖感知的、精确的肺段重建。此外，我们引入了两个临床相关的评估指标来全面评估重建效果。由于缺乏用于基准测试重建算法的公开形状数据集，我们开发了一个名为Lung3D的数据集，包含800个标注的肺段3D模型及其对应的气道、动脉、静脉和段间静脉。我们证明了所提出的方法优于现有方法，为肺段重建提供了新的视角。代码和数据可在以下链接获取。 

---
# When repeats drive the vocabulary: a Byte-Pair Encoding analysis of T2T primate genomes 

**Title (ZH)**: 当重复序列驱动词汇：T2T灵长类基因组的字对编码分析 

**Authors**: Marina Popova, Iaroslav Chelombitko, Aleksey Komissarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08918)  

**Abstract**: The emergence of telomere-to-telomere (T2T) genome assemblies has opened new avenues for comparative genomics, yet effective tokenization strategies for genomic sequences remain underexplored. In this pilot study, we apply Byte Pair Encoding (BPE) to nine T2T primate genomes including three human assemblies by training independent BPE tokenizers with a fixed vocabulary of 512,000 tokens using our custom tool, dnaBPE. Our analysis reveals that only 11,569 tokens are shared across all assemblies, while nearly 991,854 tokens are unique to a single genome, indicating a rapid decline in shared vocabulary with increasing assembly comparisons. Moreover, phylogenetic trees derived from token overlap failed to recapitulate established primate relationships, a discrepancy attributed to the disproportionate influence of species-specific high-copy repetitive elements. These findings underscore the dual nature of BPE tokenization: while it effectively compresses repetitive sequences, its sensitivity to high-copy elements limits its utility as a universal tool for comparative genomics. We discuss potential hybrid strategies and repeat-masking approaches to refine genomic tokenization, emphasizing the need for domain-specific adaptations in the development of large-scale genomic language models. The dnaBPE tool used in this study is open-source and available at this https URL. 

**Abstract (ZH)**: telomere-to-telomere (T2T) 基因组组装的出现为 comparative genomics 开辟了新途径，但基因组序列的有效细分策略尚未得到充分探索。在本试点研究中，我们使用自定义工具 dnaBPE 以及固定词汇量 512,000 的独立 BPE 分词器对九个 T2T 灵长类基因组，包括三个人类组装进行了 Byte Pair Encoding (BPE) 应用。分析结果显示，仅有 11,569 个分词被所有组装共享，而几乎 991,854 个分词仅局限于单一基因组，表明随着组装比较的增加，共享词汇量迅速下降。此外，由分词重叠推导出的系统发生树未能重现已建立的灵长类关系，这种分歧归因于物种特异性高拷贝重复元素的影响。这些发现强调了 BPE 分词的双重性质：虽然它有效压缩了重复序列，但其对高拷贝元素的敏感性限制了其在 comparative genomics 中作为通用工具的适用性。我们讨论了潜在的混合策略和重复掩蔽方法以改进基因组分词，强调在开发大规模基因组语言模型时需要特定领域的适应性。本研究使用的 dnaBPE 工具是开源的，可在以下网址获取。 

---
# A New Tractable Description Logic under Categorical Semantics 

**Title (ZH)**: 一种基于范畴语义的新可处理描述逻辑 

**Authors**: Chan Le Duc, Ludovic Brieulle  

**Link**: [PDF](https://arxiv.org/pdf/2505.08916)  

**Abstract**: Biomedical ontologies contain numerous concept or role names involving negative knowledge such as lacks_part, absence_of. Such a representation with labels rather than logical constructors would not allow a reasoner to interpret lacks_part as a kind of negation of has_part. It is known that adding negation to the tractable Description Logic (DL) EL allowing for conjunction, existential restriction and concept inclusion makes it intractable since the obtained logic includes implicitly disjunction and universal restriction which interact with other constructors. In this paper, we propose a new extension of EL with a weakened negation allowing to represent negative knowledge while retaining tractability. To this end, we introduce categorical semantics of all logical constructors of the DL SH including EL with disjunction, negation, universal restriction, role inclusion and transitive roles. The categorical semantics of a logical constructor is usually described as a set of categorical properties referring to several objects without using set membership. To restore tractability, we have to weaken semantics of disjunction and universal restriction by identifying \emph{independent} categorical properties that are responsible for intractability, and dropping them from the set of categorical properties. We show that the logic resulting from weakening semantics is more expressive than EL with the bottom concept, transitive roles and role inclusion. 

**Abstract (ZH)**: biomedical本体包含涉及负知识的概念或角色名称，如lacks_part、absence_of。这样的表示使用标签而非逻辑构造子，不会让推理器将lacks_part解释为has_part的否定。已知将否定加入到可处理的描述逻辑EL中，允许合取、存在限制和概念包含，使其变得不可处理，因为所得逻辑隐式地包括了析取和普遍限制，这些与其它构造子相互作用。在本文中，我们提出了一种新的EL扩展，引入了减弱的否定，以保留可处理性同时表示负知识。为此，我们引入了描述逻辑SH（包括EL中的析取、否定、普遍限制、角色包含和传递角色）的所有逻辑构造子的范畴语义。范畴语义通常描述为一组范畴属性，不使用集合成员身份来参考多个对象。为了恢复可处理性，我们必须通过识别导致不可处理的独立范畴属性并将其从范畴属性集合中删除来减弱析取和普遍限制的语义。我们展示了减弱语义得到的逻辑比包含底部概念、传递角色和角色包含的EL更具表达力。 

---
# FareShare: A Tool for Labor Organizers to Estimate Lost Wages and Contest Arbitrary AI and Algorithmic Deactivations 

**Title (ZH)**: FareShare: 一个用于劳动组织者估算损失工资并挑战任意AI和算法停用的工具 

**Authors**: Varun Nagaraj Rao, Samantha Dalal, Andrew Schwartz, Amna Liaqat, Dana Calacci, Andrés Monroy-Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2505.08904)  

**Abstract**: What happens when a rideshare driver is suddenly locked out of the platform connecting them to riders, wages, and daily work? Deactivation-the abrupt removal of gig workers' platform access-typically occurs through arbitrary AI and algorithmic decisions with little explanation or recourse. This represents one of the most severe forms of algorithmic control and often devastates workers' financial stability. Recent U.S. state policies now mandate appeals processes and recovering compensation during the period of wrongful deactivation based on past earnings. Yet, labor organizers still lack effective tools to support these complex, error-prone workflows. We designed FareShare, a computational tool automating lost wage estimation for deactivated drivers, through a 6 month partnership with the State of Washington's largest rideshare labor union. Over the following 3 months, our field deployment of FareShare registered 178 account signups. We observed that the tool could reduce lost wage calculation time by over 95%, eliminate manual data entry errors, and enable legal teams to generate arbitration-ready reports more efficiently. Beyond these gains, the deployment also surfaced important socio-technical challenges around trust, consent, and tool adoption in high-stakes labor contexts. 

**Abstract (ZH)**: 当网约车司机突然被平台锁定，无法接单、领取工资和开展日常工作时会发生什么？账户撤销—— gig工人平台访问的突然中断——通常通过随意的AI和算法决策进行，缺乏解释和申诉渠道。这代表了最严重的算法控制形式之一，往往严重损害工人的经济稳定。近期，美国各州政策现已要求在过往收入基础上为被无理撤销账户的工人提供申诉程序和恢复补偿。然而，劳工组织者仍缺乏有效工具来支持这些复杂且多有误的操作流程。我们与华盛顿州最大网约车劳动工会合作六个月，设计了FareShare，一种自动估算被撤销账户司机损失工资的计算工具。在随后的三个月中，FareShare的实际部署注册了178个账号。我们观察到，该工具能够将损失工资计算时间减少超过95%，消除手动数据输入错误，并使法律团队能更高效地生成仲裁准备报告。除此之外，该部署还揭示了高风险劳动环境中关于信任、同意和工具采用的重要社会技术挑战。 

---
# Performance Gains of LLMs With Humans in a World of LLMs Versus Humans 

**Title (ZH)**: LLMs与人类在充满LLMs的世界中相比，人类参与下的性能提升 

**Authors**: Lucas McCullum, Pelagie Ami Agassi, Leo Anthony Celi, Daniel K. Ebner, Chrystinne Oliveira Fernandes, Rachel S. Hicklen, Mkliwa Koumbia, Lisa Soleymani Lehmann, David Restrepo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08902)  

**Abstract**: Currently, a considerable research effort is devoted to comparing LLMs to a group of human experts, where the term "expert" is often ill-defined or variable, at best, in a state of constantly updating LLM releases. Without proper safeguards in place, LLMs will threaten to cause harm to the established structure of safe delivery of patient care which has been carefully developed throughout history to keep the safety of the patient at the forefront. A key driver of LLM innovation is founded on community research efforts which, if continuing to operate under "humans versus LLMs" principles, will expedite this trend. Therefore, research efforts moving forward must focus on effectively characterizing the safe use of LLMs in clinical settings that persist across the rapid development of novel LLM models. In this communication, we demonstrate that rather than comparing LLMs to humans, there is a need to develop strategies enabling efficient work of humans with LLMs in an almost symbiotic manner. 

**Abstract (ZH)**: 当前，相当一部分研究致力于将大型语言模型（LLM）与一群人类专家进行比较，而“专家”这一术语在持续更新的LLM版本中往往定义模糊或变化不定。若缺乏适当的保障措施，LLM将威胁到精心发展、旨在确保患者安全的传统临床护理结构。LLM创新的主要驱动力来自于社区研究，如果继续在“人类与LLM对抗”的框架下进行，将进一步加剧这一趋势。因此，未来的研究必须侧重于如何有效表征LLM在临床环境中的安全应用，这涵盖新型LLM模型的快速发展中长期保持有效。在本文中，我们证明了与其将LLM与人类进行比较，不如开发促进人类与LLM高效协作的战略，几乎是一种共生关系。 

---
# WaLLM -- Insights from an LLM-Powered Chatbot deployment via WhatsApp 

**Title (ZH)**: WaLLM —— 一个基于WhatsApp的LLM驱动聊天机器人的见解 

**Authors**: Hiba Eltigani, Rukhshan Haroon, Asli Kocak, Abdullah Bin Faisal, Noah Martin, Fahad Dogar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08894)  

**Abstract**: Recent advances in generative AI, such as ChatGPT, have transformed access to information in education, knowledge-seeking, and everyday decision-making. However, in many developing regions, access remains a challenge due to the persistent digital divide. To help bridge this gap, we developed WaLLM - a custom AI chatbot over WhatsApp, a widely used communication platform in developing regions. Beyond answering queries, WaLLM offers several features to enhance user engagement: a daily top question, suggested follow-up questions, trending and recent queries, and a leaderboard-based reward system. Our service has been operational for over 6 months, amassing over 14.7K queries from approximately 100 users. In this paper, we present WaLLM's design and a systematic analysis of logs to understand user interactions. Our results show that 55% of user queries seek factual information. "Health and well-being" was the most popular topic (28%), including queries about nutrition and disease, suggesting users view WaLLM as a reliable source. Two-thirds of users' activity occurred within 24 hours of the daily top question. Users who accessed the "Leaderboard" interacted with WaLLM 3x as those who did not. We conclude by discussing implications for culture-based customization, user interface design, and appropriate calibration of users' trust in AI systems for developing regions. 

**Abstract (ZH)**: Recent Advances in Generative AI, such as ChatGPT, have transformed access to information in education, knowledge-seeking, and everyday decision-making. However, in many developing regions, access remains a challenge due to the persistent digital divide. To help bridge this gap, we developed WaLLM - a custom AI chatbot over WhatsApp, a widely used communication platform in developing regions. Beyond answering queries, WaLLM offers several features to enhance user engagement: a daily top question, suggested follow-up questions, trending and recent queries, and a leaderboard-based reward system. Our service has been operational for over 6 months, amassing over 14.7K queries from approximately 100 users. In this paper, we present WaLLM's design and a systematic analysis of logs to understand user interactions. Our results show that 55% of user queries seek factual information. "Health and well-being" was the most popular topic (28%), including queries about nutrition and disease, suggesting users view WaLLM as a reliable source. Two-thirds of users' activity occurred within 24 hours of the daily top question. Users who accessed the "Leaderboard" interacted with WaLLM 3x as those who did not. We conclude by discussing implications for culture-based customization, user interface design, and appropriate calibration of users' trust in AI systems for developing regions. 

---
# Optimized Couplings for Watermarking Large Language Models 

**Title (ZH)**: 优化耦合用于大型语言模型水印刻印 

**Authors**: Dor Tsur, Carol Xuan Long, Claudio Mayrink Verdun, Hsiang Hsu, Haim Permuter, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2505.08878)  

**Abstract**: Large-language models (LLMs) are now able to produce text that is, in many cases, seemingly indistinguishable from human-generated content. This has fueled the development of watermarks that imprint a ``signal'' in LLM-generated text with minimal perturbation of an LLM's output. This paper provides an analysis of text watermarking in a one-shot setting. Through the lens of hypothesis testing with side information, we formulate and analyze the fundamental trade-off between watermark detection power and distortion in generated textual quality. We argue that a key component in watermark design is generating a coupling between the side information shared with the watermark detector and a random partition of the LLM vocabulary. Our analysis identifies the optimal coupling and randomization strategy under the worst-case LLM next-token distribution that satisfies a min-entropy constraint. We provide a closed-form expression of the resulting detection rate under the proposed scheme and quantify the cost in a max-min sense. Finally, we provide an array of numerical results, comparing the proposed scheme with the theoretical optimum and existing schemes, in both synthetic data and LLM watermarking. Our code is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）现在能够生成在许多情况下难以与人类生成的内容区分开来的文本。这推动了生成水印的发展，这些水印能够在最小程度上干扰LLM输出的情况下，在LLM生成的文本中嵌入“信号”。本文提供了在单次设置下对文本水印的分析。通过假设检验与辅助信息的视角，我们提出了并分析了水印检测能力和生成文本质量失真的基本权衡。我们认为水印设计的关键组件是在水印检测器与共享的辅助信息之间以及LLM词汇表的随机划分之间建立耦合。我们的分析确定了在满足最小熵约束的最坏情况LLM下一个词分布下的最优耦合和随机化策略。我们提供了所提出方案下检测率的闭式表达式，并从最大最小意义量化成本。最后，我们在合成数据和LLM水印中比较了所提出的方案与理论最优值和现有方案的性能。我们的代码可在以下网址获取：这个 https URL 

---
# Generative AI for Autonomous Driving: Frontiers and Opportunities 

**Title (ZH)**: 自动驾驶中的生成式AI：前沿与机遇 

**Authors**: Yuping Wang, Shuo Xing, Cui Can, Renjie Li, Hongyuan Hua, Kexin Tian, Zhaobin Mo, Xiangbo Gao, Keshu Wu, Sulong Zhou, Hengxu You, Juntong Peng, Junge Zhang, Zehao Wang, Rui Song, Mingxuan Yan, Walter Zimmer, Xingcheng Zhou, Peiran Li, Zhaohan Lu, Chia-Ju Chen, Yue Huang, Ryan A. Rossi, Lichao Sun, Hongkai Yu, Zhiwen Fan, Frank Hao Yang, Yuhao Kang, Ross Greer, Chenxi Liu, Eun Hak Lee, Xuan Di, Xinyue Ye, Liu Ren, Alois Knoll, Xiaopeng Li, Shuiwang Ji, Masayoshi Tomizuka, Marco Pavone, Tianbao Yang, Jing Du, Ming-Hsuan Yang, Hua Wei, Ziran Wang, Yang Zhou, Jiachen Li, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08854)  

**Abstract**: Generative Artificial Intelligence (GenAI) constitutes a transformative technological wave that reconfigures industries through its unparalleled capabilities for content creation, reasoning, planning, and multimodal understanding. This revolutionary force offers the most promising path yet toward solving one of engineering's grandest challenges: achieving reliable, fully autonomous driving, particularly the pursuit of Level 5 autonomy. This survey delivers a comprehensive and critical synthesis of the emerging role of GenAI across the autonomous driving stack. We begin by distilling the principles and trade-offs of modern generative modeling, encompassing VAEs, GANs, Diffusion Models, and Large Language Models (LLMs). We then map their frontier applications in image, LiDAR, trajectory, occupancy, video generation as well as LLM-guided reasoning and decision making. We categorize practical applications, such as synthetic data workflows, end-to-end driving strategies, high-fidelity digital twin systems, smart transportation networks, and cross-domain transfer to embodied AI. We identify key obstacles and possibilities such as comprehensive generalization across rare cases, evaluation and safety checks, budget-limited implementation, regulatory compliance, ethical concerns, and environmental effects, while proposing research plans across theoretical assurances, trust metrics, transport integration, and socio-technical influence. By unifying these threads, the survey provides a forward-looking reference for researchers, engineers, and policymakers navigating the convergence of generative AI and advanced autonomous mobility. An actively maintained repository of cited works is available at this https URL. 

**Abstract (ZH)**: 生成式人工智能（GenAI）构成了一个变革性的技术浪潮，通过其无与伦比的内容创造、推理、规划和多模态理解能力重新配置各行各业。这一革命性力量为解决工程领域的一大挑战——实现可靠的全自动驾驶，尤其是 Level 5 无人驾驶——提供了迄今为止最有前景的道路。本文综述涵盖了生成式人工智能在自主驾驶堆栈中不断涌现的作用。我们首先提炼现代生成模型的原则与权衡，包括 VAE、GAN、扩散模型和大语言模型（LLMs）。接着，我们将这些模型的应用前沿与其在图像、LiDAR、轨迹、占用率、视频生成以及大语言模型引导的推理和决策方面的应用进行映射。我们对其实用应用进行了分类，包括合成数据工作流程、端到端的驾驶策略、高保真数字孪生系统、智能交通网络以及跨域迁移至具身人工智能。我们识别出关键障碍与可能性，如全面泛化到稀有情况、评估与安全性检查、预算限制下的实现、法规合规性、伦理问题以及环境影响，并提出涵盖理论保证、信任度量、运输集成和社交技术影响的研究计划。通过统一这些线索，本文综述为研究人员、工程师和政策制定者在生成式人工智能与先进自主移动领域交汇处的导航提供了前瞻性的参考。已在以下网址维护了一个引文作品库：[此处链接]。 

---
# Improved Algorithms for Differentially Private Language Model Alignment 

**Title (ZH)**: 改进的差分隐私语言模型对齐算法 

**Authors**: Keyu Chen, Hao Tang, Qinglin Liu, Yizhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08849)  

**Abstract**: Language model alignment is crucial for ensuring that large language models (LLMs) align with human preferences, yet it often involves sensitive user data, raising significant privacy concerns. While prior work has integrated differential privacy (DP) with alignment techniques, their performance remains limited. In this paper, we propose novel algorithms for privacy-preserving alignment and rigorously analyze their effectiveness across varying privacy budgets and models. Our framework can be deployed on two celebrated alignment techniques, namely direct preference optimization (DPO) and reinforcement learning from human feedback (RLHF). Through systematic experiments on large-scale language models, we demonstrate that our approach achieves state-of-the-art performance. Notably, one of our algorithms, DP-AdamW, combined with DPO, surpasses existing methods, improving alignment quality by up to 15% under moderate privacy budgets ({\epsilon}=2-5). We further investigate the interplay between privacy guarantees, alignment efficacy, and computational demands, providing practical guidelines for optimizing these trade-offs. 

**Abstract (ZH)**: 语言模型对齐对于确保大型语言模型（LLMs）与人类偏好一致至关重要，但通常涉及敏感用户数据，从而引发重大隐私问题。虽然已有工作将差分隐私（DP）与对齐技术结合使用，但其性能仍然有限。在本文中，我们提出了一种新型的隐私保护对齐算法，并严格分析了其在不同隐私预算和模型下的有效性。我们的框架可以在两种著名的对齐技术，即直接偏好优化（DPO）和人类反馈强化学习（RLHF）上部署。通过大规模语言模型的系统实验，我们证明了我们的方法达到了最先进的性能。值得注意的是，我们的算法之一DP-AdamW，与DPO结合使用，在中等隐私预算（ε=2-5）下，对齐质量提高了高达15%，超过了现有方法。我们进一步探讨了隐私保证、对齐效果和计算需求之间的相互作用，提供了优化这些权衡的实际指南。 

---
# On the interplay of Explainability, Privacy and Predictive Performance with Explanation-assisted Model Extraction 

**Title (ZH)**: 解释性、隐私和预测性能之间的相互作用：基于解释辅助模型提取 

**Authors**: Fatima Ezzeddine, Rinad Akel, Ihab Sbeity, Silvia Giordano, Marc Langheinrich, Omran Ayoub  

**Link**: [PDF](https://arxiv.org/pdf/2505.08847)  

**Abstract**: Machine Learning as a Service (MLaaS) has gained important attraction as a means for deploying powerful predictive models, offering ease of use that enables organizations to leverage advanced analytics without substantial investments in specialized infrastructure or expertise. However, MLaaS platforms must be safeguarded against security and privacy attacks, such as model extraction (MEA) attacks. The increasing integration of explainable AI (XAI) within MLaaS has introduced an additional privacy challenge, as attackers can exploit model explanations particularly counterfactual explanations (CFs) to facilitate MEA. In this paper, we investigate the trade offs among model performance, privacy, and explainability when employing Differential Privacy (DP), a promising technique for mitigating CF facilitated MEA. We evaluate two distinct DP strategies: implemented during the classification model training and at the explainer during CF generation. 

**Abstract (ZH)**: 作为一种服务的机器学习（MLaaS）因其部署强大预测模型的能力而受到重要关注，提供了一种简便的方式，使组织能够利用高级分析而不必在专门的基础设施或专业知识上进行重大投资。然而，MLaaS平台必须受到安全和隐私攻击的防范，如模型提取（MEA）攻击。解释性人工智能（XAI）在MLaaS中的日益集成引入了额外的隐私挑战，攻击者可以利用模型解释，尤其是反事实解释（CFs），以促进MEA。在本文中，我们研究了在使用差分隐私（DP）时模型性能、隐私和解释性的权衡，DP是一种有前景的技术，用于缓解由CF促进的MEA。我们评估了两种不同的DP策略：实施在分类模型训练期间和实施在生成CF时的解释器期间。 

---
# Evaluating Simplification Algorithms for Interpretability of Time Series Classification 

**Title (ZH)**: 评估时间序列分类解释性的简化算法 

**Authors**: Felix Marti-Perez, Brigt Håvardstun, Cèsar Ferri, Carlos Monserrat, Jan Arne Telle  

**Link**: [PDF](https://arxiv.org/pdf/2505.08846)  

**Abstract**: In this work, we introduce metrics to evaluate the use of simplified time series in the context of interpretability of a TSC - a Time Series Classifier. Such simplifications are important because time series data, in contrast to text and image data, are not intuitively understandable to humans. These metrics are related to the complexity of the simplifications - how many segments they contain - and to their loyalty - how likely they are to maintain the classification of the original time series. We employ these metrics to evaluate four distinct simplification algorithms, across several TSC algorithms and across datasets of varying characteristics, from seasonal or stationary to short or long. Our findings suggest that using simplifications for interpretability of TSC is much better than using the original time series, particularly when the time series are seasonal, non-stationary and/or with low entropy. 

**Abstract (ZH)**: 本研究引入了评估时间序列简化在时间序列分类器可解释性中的使用情况的指标，这些指标涉及简化的时间序列的复杂性和忠诚度，并在不同类型的时间序列分类器和不同特征的数据集上评估了四种不同的简化算法。研究表明，使用简化的时间序列比使用原始时间序列更好地提高时间序列分类器的可解释性，尤其是在时间序列具有季节性、非平稳性和/或低熵的情况下。 

---
# Validation of Conformal Prediction in Cervical Atypia Classification 

**Title (ZH)**: 宫颈不典型病变分类中齐性预测的验证 

**Authors**: Misgina Tsighe Hagos, Antti Suutala, Dmitrii Bychkov, Hakan Kücükel, Joar von Bahr, Milda Poceviciute, Johan Lundin, Nina Linder, Claes Lundström  

**Link**: [PDF](https://arxiv.org/pdf/2505.08845)  

**Abstract**: Deep learning based cervical cancer classification can potentially increase access to screening in low-resource regions. However, deep learning models are often overconfident and do not reliably reflect diagnostic uncertainty. Moreover, they are typically optimized to generate maximum-likelihood predictions, which fail to convey uncertainty or ambiguity in their results. Such challenges can be addressed using conformal prediction, a model-agnostic framework for generating prediction sets that contain likely classes for trained deep-learning models. The size of these prediction sets indicates model uncertainty, contracting as model confidence increases. However, existing conformal prediction evaluation primarily focuses on whether the prediction set includes or covers the true class, often overlooking the presence of extraneous classes. We argue that prediction sets should be truthful and valuable to end users, ensuring that the listed likely classes align with human expectations rather than being overly relaxed and including false positives or unlikely classes. In this study, we comprehensively validate conformal prediction sets using expert annotation sets collected from multiple annotators. We evaluate three conformal prediction approaches applied to three deep-learning models trained for cervical atypia classification. Our expert annotation-based analysis reveals that conventional coverage-based evaluations overestimate performance and that current conformal prediction methods often produce prediction sets that are not well aligned with human labels. Additionally, we explore the capabilities of the conformal prediction methods in identifying ambiguous and out-of-distribution data. 

**Abstract (ZH)**: 基于深度学习的宫颈癌分类有可能增加低资源地区筛查的 accessibility。然而，深度学习模型往往过于自信，不能可靠地反映诊断不确定性。此外，它们通常被优化以生成最大似然预测，这些预测未能传达结果中的不确定性或模糊性。此类挑战可以通过使用校准预测来解决，校准预测是一种适用于生成包含训练深度学习模型可能类别的预测集的模型agnostic框架。这些预测集的大小表示模型不确定性，在模型信心增加时收缩。然而，现有的校准预测评估主要关注预测集是否包含或覆盖真实类别，往往忽视了多余类别的存在。我们认为预测集应真实且对最终用户有价值，确保列出的可能类别符合人类预期，而不是过于宽松并包括假阳性或不太可能的类别。在本研究中，我们使用来自多名注释者收集的专家注释集全面验证校准预测集。我们评估了应用于三种用于宫颈异常分类的深度学习模型的三种校准预测方法。我们的基于专家注释的分析表明，传统的覆盖率评估高估了性能，当前的校准预测方法通常会产生与人类标签不完全一致的预测集。此外，我们探索了校准预测方法在识别模糊和分布外数据方面的能力。 

---
# CellTypeAgent: Trustworthy cell type annotation with Large Language Models 

**Title (ZH)**: CellTypeAgent: 用大型语言模型进行可信赖的细胞类型注释 

**Authors**: Jiawen Chen, Jianghao Zhang, Huaxiu Yao, Yun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08844)  

**Abstract**: Cell type annotation is a critical yet laborious step in single-cell RNA sequencing analysis. We present a trustworthy large language model (LLM)-agent, CellTypeAgent, which integrates LLMs with verification from relevant databases. CellTypeAgent achieves higher accuracy than existing methods while mitigating hallucinations. We evaluated CellTypeAgent across nine real datasets involving 303 cell types from 36 tissues. This combined approach holds promise for more efficient and reliable cell type annotation. 

**Abstract (ZH)**: 细胞类型注释是单细胞RNA测序分析中的一个关键但耗时的步骤。我们提出了一种可信赖的大语言模型（LLM）代理CellTypeAgent，它将LLM与相关数据库的验证相结合。CellTypeAgent在准确性上优于现有方法，同时减少了幻觉现象。我们使用涉及36个组织303种细胞类型的九个真实数据集评估了CellTypeAgent。这种结合方法有望提高细胞类型注释的效率和可靠性。 

---
# Will AI Take My Job? Evolving Perceptions of Automation and Labor Risk in Latin America 

**Title (ZH)**: AI会取代我的工作吗？拉丁美洲自动化与劳动风险 perception 的演变 

**Authors**: Andrea Cremaschi, Dae-Jin Lee, Manuele Leonelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.08841)  

**Abstract**: As artificial intelligence and robotics increasingly reshape the global labor market, understanding public perceptions of these technologies becomes critical. We examine how these perceptions have evolved across Latin America, using survey data from the 2017, 2018, 2020, and 2023 waves of the Latinobarómetro. Drawing on responses from over 48,000 individuals across 16 countries, we analyze fear of job loss due to artificial intelligence and robotics. Using statistical modeling and latent class analysis, we identify key structural and ideological predictors of concern, with education level and political orientation emerging as the most consistent drivers. Our findings reveal substantial temporal and cross-country variation, with a notable peak in fear during 2018 and distinct attitudinal profiles emerging from latent segmentation. These results offer new insights into the social and structural dimensions of AI anxiety in emerging economies and contribute to a broader understanding of public attitudes toward automation beyond the Global North. 

**Abstract (ZH)**: 随着人工智能和机器人技术 increasingly 重塑全球劳动力市场，理解公众对这些技术的看法变得至关重要。我们使用 2017、2018、2020 和 2023 年拉丁obarómetro 波特调查的数据，探讨这些看法在拉丁美洲的演变情况。基于来自 16 个国家超过 48,000 名个体的回应，我们分析对由于人工智能和机器人技术导致的失业恐惧。通过统计建模和潜在类别分析，我们识别出关键的结构性和意识形态预测因素，教育水平和政治倾向是最具一致性的驱动因素。我们的发现显示了显著的时间性和跨国家差异，尤其是 2018 年恐惧情绪达到峰值，并且从潜在细分中 emerged 出不同的态度模式。这些结果提供了对新兴经济体中 AI 焦虑的社会和结构性维度的新见解，并为对自动化持公共卫生态度的更广泛理解做出了贡献，超越了全球经济北方。 

---
# Ultrasound Report Generation with Multimodal Large Language Models for Standardized Texts 

**Title (ZH)**: 多模态大型语言模型生成标准化超声报告 

**Authors**: Peixuan Ge, Tongkun Su, Faqin Lv, Baoliang Zhao, Peng Zhang, Chi Hong Wong, Liang Yao, Yu Sun, Zenan Wang, Pak Kin Wong, Ying Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08838)  

**Abstract**: Ultrasound (US) report generation is a challenging task due to the variability of US images, operator dependence, and the need for standardized text. Unlike X-ray and CT, US imaging lacks consistent datasets, making automation difficult. In this study, we propose a unified framework for multi-organ and multilingual US report generation, integrating fragment-based multilingual training and leveraging the standardized nature of US reports. By aligning modular text fragments with diverse imaging data and curating a bilingual English-Chinese dataset, the method achieves consistent and clinically accurate text generation across organ sites and languages. Fine-tuning with selective unfreezing of the vision transformer (ViT) further improves text-image alignment. Compared to the previous state-of-the-art KMVE method, our approach achieves relative gains of about 2\% in BLEU scores, approximately 3\% in ROUGE-L, and about 15\% in CIDEr, while significantly reducing errors such as missing or incorrect content. By unifying multi-organ and multi-language report generation into a single, scalable framework, this work demonstrates strong potential for real-world clinical workflows. 

**Abstract (ZH)**: 基于 ultrasound 图像的多器官和多语言报告生成：统一框架与标准化文本生成 

---
# Robustness Analysis against Adversarial Patch Attacks in Fully Unmanned Stores 

**Title (ZH)**: 全无人商店中对抗性patches攻击的鲁棒性分析 

**Authors**: Hyunsik Na, Wonho Lee, Seungdeok Roh, Sohee Park, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.08835)  

**Abstract**: The advent of convenient and efficient fully unmanned stores equipped with artificial intelligence-based automated checkout systems marks a new era in retail. However, these systems have inherent artificial intelligence security vulnerabilities, which are exploited via adversarial patch attacks, particularly in physical environments. This study demonstrated that adversarial patches can severely disrupt object detection models used in unmanned stores, leading to issues such as theft, inventory discrepancies, and interference. We investigated three types of adversarial patch attacks -- Hiding, Creating, and Altering attacks -- and highlighted their effectiveness. We also introduce the novel color histogram similarity loss function by leveraging attacker knowledge of the color information of a target class object. Besides the traditional confusion-matrix-based attack success rate, we introduce a new bounding-boxes-based metric to analyze the practical impact of these attacks. Starting with attacks on object detection models trained on snack and fruit datasets in a digital environment, we evaluated the effectiveness of adversarial patches in a physical testbed that mimicked a real unmanned store with RGB cameras and realistic conditions. Furthermore, we assessed the robustness of these attacks in black-box scenarios, demonstrating that shadow attacks can enhance success rates of attacks even without direct access to model parameters. Our study underscores the necessity for robust defense strategies to protect unmanned stores from adversarial threats. Highlighting the limitations of the current defense mechanisms in real-time detection systems and discussing various proactive measures, we provide insights into improving the robustness of object detection models and fortifying unmanned retail environments against these attacks. 

**Abstract (ZH)**: 基于人工智能自动化结账系统的便捷无人商店的出现标志着零售新时代的到来。然而，这些系统内固有人工智能安全漏洞通过对抗性补丁攻击在物理环境中被利用。本研究证明了对抗性补丁会严重干扰无人商店中使用的物体检测模型，导致盗窃、库存差异和干扰等问题。我们探讨了三种类型的对抗性补丁攻击——隐藏攻击、创造攻击和修改攻击，并强调了它们的有效性。此外，我们通过利用攻击者关于目标类别物体颜色信息的知识，引入了新颖的颜色直方图相似性损失函数。除了传统的混淆矩阵基于的攻击成功率，我们还引入了一种基于边界框的新度量来分析这些攻击的实际影响。从数字环境中对零食和水果数据集训练的物体检测模型发起攻击开始，我们在一个模拟真实无人商店的物理测试床中评估了对抗性补丁的效果，该测试床使用RGB摄像头并具有现实条件。此外，我们在黑盒场景中评估了这些攻击的鲁棒性，证明即使没有直接访问模型参数，阴影攻击也能增强攻击成功率。本研究强调了为从对抗性威胁中保护无人商店而制定稳健防御策略的必要性。我们指出现有实时检测系统防御机制的局限性，并讨论了各种主动措施，提供改善物体检测模型鲁棒性和增强无人零售环境防御能力的见解。 

---
# Crowd Scene Analysis using Deep Learning Techniques 

**Title (ZH)**: 基于深度学习技术的 crowd 场景分析 

**Authors**: Muhammad Junaid Asif  

**Link**: [PDF](https://arxiv.org/pdf/2505.08834)  

**Abstract**: Our research is focused on two main applications of crowd scene analysis crowd counting and anomaly detection In recent years a large number of researches have been presented in the domain of crowd counting We addressed two main challenges in this domain 1 Deep learning models are datahungry paradigms and always need a large amount of annotated data for the training of algorithm It is timeconsuming and costly task to annotate such large amount of data Selfsupervised training is proposed to deal with this challenge 2 MCNN consists of multicolumns of CNN with different sizes of filters by presenting a novel approach based on a combination of selfsupervised training and MultiColumn CNN This enables the model to learn features at different levels and makes it effective in dealing with challenges of occluded scenes nonuniform density complex backgrounds and scale invariation The proposed model was evaluated on publicly available data sets such as ShanghaiTech and UCFQNRF by means of MAE and MSE A spatiotemporal model based on VGG19 is proposed for crowd anomaly detection addressing challenges like lighting environmental conditions unexpected objects and scalability The model extracts spatial and temporal features allowing it to be generalized to realworld scenes Spatial features are learned using CNN while temporal features are learned using LSTM blocks The model works on binary classification and can detect normal or abnormal behavior The models performance is improved by replacing fully connected layers with dense residual blocks Experiments on the Hockey Fight dataset and SCVD dataset show our models outperform other stateoftheart approaches 

**Abstract (ZH)**: crowd场景分析中的 crowd计数和异常检测研究：基于自监督训练和多柱卷积神经网络的方法

基于自监督训练和多柱卷积神经网络的 crowd计数研究

基于自监督训练和多柱卷积神经网络的 crowd计数和异常检测研究 

---
# Federated Large Language Models: Feasibility, Robustness, Security and Future Directions 

**Title (ZH)**: 联邦大型语言模型：可行性、稳健性、安全性及未来发展方向 

**Authors**: Wenhao Jiang, Yuchuan Luo, Guilin Deng, Silong Chen, Xu Yang, Shihong Wu, Xinwen Gao, Lin Liu, Shaojing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08830)  

**Abstract**: The integration of Large Language Models (LLMs) and Federated Learning (FL) presents a promising solution for joint training on distributed data while preserving privacy and addressing data silo issues. However, this emerging field, known as Federated Large Language Models (FLLM), faces significant challenges, including communication and computation overheads, heterogeneity, privacy and security concerns. Current research has primarily focused on the feasibility of FLLM, but future trends are expected to emphasize enhancing system robustness and security. This paper provides a comprehensive review of the latest advancements in FLLM, examining challenges from four critical perspectives: feasibility, robustness, security, and future directions. We present an exhaustive survey of existing studies on FLLM feasibility, introduce methods to enhance robustness in the face of resource, data, and task heterogeneity, and analyze novel risks associated with this integration, including privacy threats and security challenges. We also review the latest developments in defense mechanisms and explore promising future research directions, such as few-shot learning, machine unlearning, and IP protection. This survey highlights the pressing need for further research to enhance system robustness and security while addressing the unique challenges posed by the integration of FL and LLM. 

**Abstract (ZH)**: 大型语言模型与联邦学习的整合：挑战与未来方向 

---
# Aggregating Concepts of Fairness and Accuracy in Predictive Systems 

**Title (ZH)**: 聚集预测系统中公平性和准确性的概念 

**Authors**: David Kinney  

**Link**: [PDF](https://arxiv.org/pdf/2505.08829)  

**Abstract**: An algorithm that outputs predictions about the state of the world will almost always be designed with the implicit or explicit goal of outputting accurate predictions (i.e., predictions that are likely to be true). In addition, the rise of increasingly powerful predictive algorithms brought about by the recent revolution in artificial intelligence has led to an emphasis on building predictive algorithms that are fair, in the sense that their predictions do not systematically evince bias or bring about harm to certain individuals or groups. This state of affairs presents two conceptual challenges. First, the goals of accuracy and fairness can sometimes be in tension, and there are no obvious normative guidelines for managing the trade-offs between these two desiderata when they arise. Second, there are many distinct ways of measuring both the accuracy and fairness of a predictive algorithm; here too, there are no obvious guidelines on how to aggregate our preferences for predictive algorithms that satisfy disparate measures of fairness and accuracy to various extents. The goal of this paper is to address these challenges by arguing that there are good reasons for using a linear combination of accuracy and fairness metrics to measure the all-things-considered value of a predictive algorithm for agents who care about both accuracy and fairness. My argument depends crucially on a classic result in the preference aggregation literature due to Harsanyi. After making this formal argument, I apply my result to an analysis of accuracy-fairness trade-offs using the COMPAS dataset compiled by Angwin et al. 

**Abstract (ZH)**: 一种关于世界状态的预测算法几乎总是旨在输出准确的预测（即，有可能真实的预测）。此外，最近人工智能革命导致的预测算法能力不断增强，使得人们更加重视构建公平的预测算法，即其预测不应系统地表现出偏见或对某些个人或群体造成伤害。这种状态提出了两个概念上的挑战。首先，准确性和公平性的目标有时是矛盾的，当两者发生冲突时，没有明显的规范性指导原则来管理这两项要求之间的权衡。其次，衡量预测算法的准确性和公平性有多种不同的方法；同样，也没有明显的指导原则来综合不同程度满足各种公平性和准确性的预测算法的偏好。本文旨在通过论证使用准确性和公平性指标的线性组合来衡量关心准确性和公平性的代理人的综合价值是合理的，来解决这些挑战。我的论证关键依赖于Harsanyi在偏好聚合文献中的经典成果。在进行了这一形式化的论证后，我将我的结果应用于Angwin等人编制的COMPAS数据集的准确性和公平性权衡分析中。 

---
# Human-AI Collaboration or Academic Misconduct? Measuring AI Use in Student Writing Through Stylometric Evidence 

**Title (ZH)**: 人类与AI的合作还是学术不端？通过风格计量证据测量学生写作中AI的使用 

**Authors**: Eduardo Araujo Oliveira, Madhavi Mohoni, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2505.08828)  

**Abstract**: As human-AI collaboration becomes increasingly prevalent in educational contexts, understanding and measuring the extent and nature of such interactions pose significant challenges. This research investigates the use of authorship verification (AV) techniques not as a punitive measure, but as a means to quantify AI assistance in academic writing, with a focus on promoting transparency, interpretability, and student development. Building on prior work, we structured our investigation into three stages: dataset selection and expansion, AV method development, and systematic evaluation. Using three datasets - including a public dataset (PAN-14) and two from University of Melbourne students from various courses - we expanded the data to include LLM-generated texts, totalling 1,889 documents and 540 authorship problems from 506 students. We developed an adapted Feature Vector Difference AV methodology to construct robust academic writing profiles for students, designed to capture meaningful, individual characteristics of their writing. The method's effectiveness was evaluated across multiple scenarios, including distinguishing between student-authored and LLM-generated texts and testing resilience against LLMs' attempts to mimic student writing styles. Results demonstrate the enhanced AV classifier's ability to identify stylometric discrepancies and measure human-AI collaboration at word and sentence levels while providing educators with a transparent tool to support academic integrity investigations. This work advances AV technology, offering actionable insights into the dynamics of academic writing in an AI-driven era. 

**Abstract (ZH)**: 随着人机协作在教育领域中越来越普遍，理解并衡量这类互动的程度与性质提出了重大挑战。本研究探讨了将作者ship验证（AV）技术作为非惩罚性手段，用于量化学术写作中的AI协助，重点关注提高透明度、可解释性和学生发展。基于前期工作，我们将研究分为三个阶段：数据集选择与扩展、AV方法开发以及系统评估。使用三个数据集——包括一个公开数据集（PAN-14）和两个来自墨尔本大学不同课程学生的数据集——我们扩展了数据集，共包含1,889份文档和540个作者ship问题，涉及506名学生。我们开发了一种适应性的特征向量差异AV方法，旨在构建能够捕捉学生写作有意义、个性化特征的学术写作档案。方法的有效性在多种场景下进行了评估，包括区分学生作者和LLM生成的文本以及测试抵抗LLM模仿学生写作风格的能力。结果表明，改进的AV分类器能够识别风格学差异，并在单词和句子层面衡量人机协作，为教育者提供一个透明的工具，支持学术诚信调查。本工作推动了AV技术的发展，提供了有关AI驱动时代学术写作动态的实际洞察。 

---
# Self Rewarding Self Improving 

**Title (ZH)**: 自我奖励自我改进 

**Authors**: Toby Simonds, Kevin Lopez, Akira Yoshiyama, Dominique Garmier  

**Link**: [PDF](https://arxiv.org/pdf/2505.08827)  

**Abstract**: We demonstrate that large language models can effectively self-improve through self-judging without requiring reference solutions, leveraging the inherent asymmetry between generating and verifying solutions. Our experiments on Countdown puzzles and MIT Integration Bee problems show that models can provide reliable reward signals without ground truth answers, enabling reinforcement learning in domains previously not possible. By implementing self-judging, we achieve significant performance gains maintaining alignment with formal verification. When combined with synthetic question generation, we establish a complete self-improvement loop where models generate practice problems, solve them, and evaluate their own performance-achieving an 8% improvement with Qwen 2.5 7B over baseline and surpassing GPT-4o performance on integration tasks. Our findings demonstrate that LLM judges can provide effective reward signals for training models, unlocking many reinforcement learning environments previously limited by the difficulty of creating programmatic rewards. This suggests a potential paradigm shift toward AI systems that continuously improve through self-directed learning rather than human-guided training, potentially accelerating progress in domains with scarce training data or complex evaluation requirements. 

**Abstract (ZH)**: 我们展示了大型语言模型可以通过自我评判有效地自我改进，无需参考答案，利用生成和验证解决方案之间的固有不对称性。我们在 Countdown 数独和 MIT Integration Bee 问题上的实验表明，模型可以在没有真实答案的情况下提供可靠的奖励信号，从而在先前不可能的应用领域实现强化学习。通过实施自我评判，我们在保持与形式验证一致性的同时实现了显著的性能提升。结合合成问题生成后，我们建立了一个完整的自我改进循环，其中模型生成练习问题、解决这些问题并评估自己的表现——Qwen 2.5 7B 较基线模型实现了 8% 的提升，并在积分任务上超过了 GPT-4。我们的研究结果显示，语言模型裁判可以为训练模型提供有效的奖励信号，解锁了许多以前因难以创建程序化奖励而受限的强化学习环境。这表明一种潜在的范式转变，即通过自主学习而非人工指导训练实现 AI 系统的持续改进，有可能加速在稀缺训练数据或复杂评估要求领域中的进展。 

---
# Multi-source Plume Tracing via Multi-Agent Reinforcement Learning 

**Title (ZH)**: 多源羽流追踪的多代理强化学习方法 

**Authors**: Pedro Antonio Alarcon Granadeno, Theodore Chambers, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08825)  

**Abstract**: Industrial catastrophes like the Bhopal disaster (1984) and the Aliso Canyon gas leak (2015) demonstrate the urgent need for rapid and reliable plume tracing algorithms to protect public health and the environment. Traditional methods, such as gradient-based or biologically inspired approaches, often fail in realistic, turbulent conditions. To address these challenges, we present a Multi-Agent Reinforcement Learning (MARL) algorithm designed for localizing multiple airborne pollution sources using a swarm of small uncrewed aerial systems (sUAS). Our method models the problem as a Partially Observable Markov Game (POMG), employing a Long Short-Term Memory (LSTM)-based Action-specific Double Deep Recurrent Q-Network (ADDRQN) that uses full sequences of historical action-observation pairs, effectively approximating latent states. Unlike prior work, we use a general-purpose simulation environment based on the Gaussian Plume Model (GPM), incorporating realistic elements such as a three-dimensional environment, sensor noise, multiple interacting agents, and multiple plume sources. The incorporation of action histories as part of the inputs further enhances the adaptability of our model in complex, partially observable environments. Extensive simulations show that our algorithm significantly outperforms conventional approaches. Specifically, our model allows agents to explore only 1.29\% of the environment to successfully locate pollution sources. 

**Abstract (ZH)**: 工业灾难如博帕尔灾难（1984年）和阿利索峡谷天然气泄漏（2015年）凸显了在现实湍流条件下亟需快速可靠烟雾追踪算法以保护公共健康和环境的紧迫性。传统的梯度基方法或生物启发方法往往在实际湍流条件下失效。为应对这些挑战，我们提出了一种适用于利用小型无人航空系统群（sUAS）定位多个空中污染源的多智能体强化学习（MARL）算法。该方法将问题建模为部分可观测马尔可夫博弈（POMG），并采用基于长短期记忆（LSTM）的动作特定双深层循环Q网络（ADDRQN），有效逼近隐含状态。与先前工作不同，我们使用基于高斯烟雾模型（GPM）的通用仿真环境，其中包括三维环境、传感器噪声、多个交互智能体和多个烟雾源。将动作历史作为输入的一部分进一步增强了模型在复杂、部分可观测环境中的适应性。 extensive simulations表明，我们的算法显著优于传统方法。具体来说，我们的模型使智能体仅探索环境的1.29%即可成功定位污染源。 

---
# An Extra RMSNorm is All You Need for Fine Tuning to 1.58 Bits 

**Title (ZH)**: 额外的RMSNorm即可用于将微调精度提升至1.58比特 

**Authors**: Cody Steinmetz, Gavin Childress, Aaron Herbst, Gavin Jones, Jasdeep Singh, Eli Vang, Keagan Weinstock  

**Link**: [PDF](https://arxiv.org/pdf/2505.08823)  

**Abstract**: Large language models (LLMs) have transformed natural-language processing, yet their scale makes real-world deployment costly. Post-training quantization reduces memory and computation but often degrades accuracy, while quantization-aware training can recover performance at the cost of extra training. Pushing quantization to the ternary (2-bit) regime yields even larger savings but is notoriously unstable. Building on recent work showing that a bias-free, RMS-normalized Transformer with straight-through estimation can reach 1.58-bit precision, we demonstrate that simply inserting RMS normalization before every linear projection and applying a gradual, layer-wise quantization schedule stably fine-tunes full-precision checkpoints into ternary LLMs. Our approach matches or surpasses more elaborate knowledge-distillation pipelines on standard language-modeling benchmarks without adding model complexity. These results indicate that careful normalization alone can close much of the accuracy gap between ternary and full-precision LLMs, making ultra-low-bit inference practical. 

**Abstract (ZH)**: 基于 RMS 归一化和逐层量化调度的稳定三值大语言模型精调 

---
# A Comparative Study of Transformer-Based Models for Multi-Horizon Blood Glucose Prediction 

**Title (ZH)**: 基于变压器模型的多 horizon 葡萄糖水平预测对比研究 

**Authors**: Meryem Altin Karagoz, Marc D. Breton, Anas El Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2505.08821)  

**Abstract**: Accurate blood glucose prediction can enable novel interventions for type 1 diabetes treatment, including personalized insulin and dietary adjustments. Although recent advances in transformer-based architectures have demonstrated the power of attention mechanisms in complex multivariate time series prediction, their potential for blood glucose (BG) prediction remains underexplored. We present a comparative analysis of transformer models for multi-horizon BG prediction, examining forecasts up to 4 hours and input history up to 1 week. The publicly available DCLP3 dataset (n=112) was split (80%-10%-10%) for training, validation, and testing, and the OhioT1DM dataset (n=12) served as an external test set. We trained networks with point-wise, patch-wise, series-wise, and hybrid embeddings, using CGM, insulin, and meal data. For short-term blood glucose prediction, Crossformer, a patch-wise transformer architecture, achieved a superior 30-minute prediction of RMSE (15.6 mg / dL on OhioT1DM). For longer-term predictions (1h, 2h, and 4h), PatchTST, another path-wise transformer, prevailed with the lowest RMSE (24.6 mg/dL, 36.1 mg/dL, and 46.5 mg/dL on OhioT1DM). In general, models that used tokenization through patches demonstrated improved accuracy with larger input sizes, with the best results obtained with a one-week history. These findings highlight the promise of transformer-based architectures for BG prediction by capturing and leveraging seasonal patterns in multivariate time-series data to improve accuracy. 

**Abstract (ZH)**: 准确的血糖预测可以实现对1型糖尿病治疗的新型干预措施，包括个性化胰岛素和饮食调整。尽管基于变换器的架构在复杂多变量时间序列预测中展示了注意力机制的强大能力，但其在血糖（BG）预测中的潜力尚未得到充分探索。我们对变换器模型进行了多 horizons 血糖预测的比较分析，检查了4小时内的预测和1周内的输入历史。公开可用的DCLP3数据集（n=112）按80%-10%-10%的比例分为训练集、验证集和测试集，而OhioT1DM数据集（n=12）作为外部测试集。我们使用持续葡萄糖监测（CGM）、胰岛素和餐食数据，训练了使用点式、块式、序列式和混合式嵌入的网络。对于短期血糖预测，块式变换器架构Crossformer实现了30分钟预测的最佳RMSE（OhioT1DM上的15.6 mg/dL）。对于长期预测（1小时、2小时和4小时），另一块式变换器PatchTST表现最佳，分别达到24.6 mg/dL、36.1 mg/dL和46.5 mg/dL的最低RMSE（OhioT1DM上的）。总体而言，通过块进行标记化并具有较大输入大小的模型显示出更好的准确性，最佳结果来自一周的历史数据。这些发现突显了基于变换器的架构在血糖预测中的潜力，通过捕捉和利用多变量时间序列数据中的季节性模式来提高准确性。 

---
# Position: Restructuring of Categories and Implementation of Guidelines Essential for VLM Adoption in Healthcare 

**Title (ZH)**: 位置：结构重构与指南实施对于推动医疗健康领域VLM应用至关重要 

**Authors**: Amara Tariq, Rimita Lahiri, Charles Kahn, Imon Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.08818)  

**Abstract**: The intricate and multifaceted nature of vision language model (VLM) development, adaptation, and application necessitates the establishment of clear and standardized reporting protocols, particularly within the high-stakes context of healthcare. Defining these reporting standards is inherently challenging due to the diverse nature of studies involving VLMs, which vary significantly from the development of all new VLMs or finetuning for domain alignment to off-the-shelf use of VLM for targeted diagnosis and prediction tasks. In this position paper, we argue that traditional machine learning reporting standards and evaluation guidelines must be restructured to accommodate multiphase VLM studies; it also has to be organized for intuitive understanding of developers while maintaining rigorous standards for reproducibility. To facilitate community adoption, we propose a categorization framework for VLM studies and outline corresponding reporting standards that comprehensively address performance evaluation, data reporting protocols, and recommendations for manuscript composition. These guidelines are organized according to the proposed categorization scheme. Lastly, we present a checklist that consolidates reporting standards, offering a standardized tool to ensure consistency and quality in the publication of VLM-related research. 

**Abstract (ZH)**: 视觉语言模型（VLM）开发、适应与应用的复杂性和多面性 necessitates the 建立清晰和标准化的报告规范，尤其是在医疗保健这种高风险的背景下。在涉及VLM的研究多样性基础上，定义这些报告标准本身具有挑战性，这些研究从新VLM的开发到特定领域的微调不等。在这篇立场论文中，我们argue that 认为传统的机器学习报告标准和评估指南必须重新结构化以适应多阶段的VLM研究；同时需要组织成易于开发者理解的形式，同时保持严格的标准以确保可重复性。为了促进社区采用，我们提出了一种VLM研究分类框架，并概述了相应的报告标准，这些标准全面涵盖了性能评估、数据报告规范和手稿编制建议。这些指南根据提出的分类方案组织。最后，我们提供了一个检查表，汇总了报告标准，提供了一个标准化工具以确保VLM相关研究出版的一致性和质量。 

---
# Towards Understanding Deep Learning Model in Image Recognition via Coverage Test 

**Title (ZH)**: 基于覆盖率测试理解图像识别中的深度学习模型 

**Authors**: Wenkai Li, Xiaoqi Li, Yingjie Mao, Yishun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08814)  

**Abstract**: Deep neural networks (DNNs) play a crucial role in the field of artificial intelligence, and their security-related testing has been a prominent research focus. By inputting test cases, the behavior of models is examined for anomalies, and coverage metrics are utilized to determine the extent of neurons covered by these test cases. With the widespread application and advancement of DNNs, different types of neural behaviors have garnered attention, leading to the emergence of various coverage metrics for neural networks. However, there is currently a lack of empirical research on these coverage metrics, specifically in analyzing the relationships and patterns between model depth, configuration information, and neural network coverage. This paper aims to investigate the relationships and patterns of four coverage metrics: primary functionality, boundary, hierarchy, and structural coverage. A series of empirical experiments were conducted, selecting LeNet, VGG, and ResNet as different DNN architectures, along with 10 models of varying depths ranging from 5 to 54 layers, to compare and study the relationships between different depths, configuration information, and various neural network coverage metrics. Additionally, an investigation was carried out on the relationships between modified decision/condition coverage and dataset size. Finally, three potential future directions are proposed to further contribute to the security testing of DNN Models. 

**Abstract (ZH)**: 深度神经网络（DNNs）在人工智能领域发挥着关键作用，其安全性测试已成为研究的热点。通过输入测试案例，检查模型的行为以检测异常，并使用覆盖度量来确定这些测试案例覆盖的神经元范围。随着DNNs的广泛应用和进步，不同类型的神经行为引起了关注，导致了各种神经网络覆盖度量的出现。然而，目前缺乏关于这些覆盖度量的实证研究，特别是关于模型深度、配置信息和神经网络覆盖之间的关系和模式的研究。本文旨在探讨四种覆盖度量——基本功能、边界、层次和结构覆盖之间的关系和模式。进行了一系列实证实验，选择了LeNet、VGG和ResNet等不同的DNN架构，以及从5到54层的10个不同深度的模型，以比较和研究不同深度、配置信息和各种神经网络覆盖度量之间的关系。此外，还研究了修改后的决策/条件覆盖与数据集大小之间的关系。最后，提出了三个潜在的未来方向，以进一步促进DNN模型的安全性测试。 

---
# Machine Learning-Based Detection of DDoS Attacks in VANETs for Emergency Vehicle Communication 

**Title (ZH)**: 基于机器学习的 VANET 中应急车辆通信中DDoS攻击检测 

**Authors**: Bappa Muktar, Vincent Fono, Adama Nouboukpo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08810)  

**Abstract**: Vehicular Ad Hoc Networks (VANETs) play a key role in Intelligent Transportation Systems (ITS), particularly in enabling real-time communication for emergency vehicles. However, Distributed Denial of Service (DDoS) attacks, which interfere with safety-critical communication channels, can severely impair their reliability. This study introduces a robust and scalable framework to detect DDoS attacks in highway-based VANET environments. A synthetic dataset was constructed using Network Simulator 3 (NS-3) in conjunction with the Simulation of Urban Mobility (SUMO) and further enriched with real-world mobility traces from Germany's A81 highway, extracted via OpenStreetMap (OSM). Three traffic categories were simulated: DDoS, VoIP, and TCP-based video streaming (VideoTCP). The data preprocessing pipeline included normalization, signal-to-noise ratio (SNR) feature engineering, missing value imputation, and class balancing using the Synthetic Minority Over-sampling Technique (SMOTE). Feature importance was assessed using SHapley Additive exPlanations (SHAP). Eleven classifiers were benchmarked, among them XGBoost (XGB), CatBoost (CB), AdaBoost (AB), GradientBoosting (GB), and an Artificial Neural Network (ANN). XGB and CB achieved the best performance, each attaining an F1-score of 96%. These results highlight the robustness of the proposed framework and its potential for real-time deployment in VANETs to secure critical emergency communications. 

**Abstract (ZH)**: 基于高速公路的VANET环境中的DDoS攻击检测鲁棒可扩展框架 

---
# MixBridge: Heterogeneous Image-to-Image Backdoor Attack through Mixture of Schrödinger Bridges 

**Title (ZH)**: MixBridge: 通过薛定谔橋混合进行的异构图像到图像后门攻击 

**Authors**: Shixi Qin, Zhiyong Yang, Shilong Bao, Shi Wang, Qianqian Xu, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08809)  

**Abstract**: This paper focuses on implanting multiple heterogeneous backdoor triggers in bridge-based diffusion models designed for complex and arbitrary input distributions. Existing backdoor formulations mainly address single-attack scenarios and are limited to Gaussian noise input models. To fill this gap, we propose MixBridge, a novel diffusion Schrödinger bridge (DSB) framework to cater to arbitrary input distributions (taking I2I tasks as special cases). Beyond this trait, we demonstrate that backdoor triggers can be injected into MixBridge by directly training with poisoned image pairs. This eliminates the need for the cumbersome modifications to stochastic differential equations required in previous studies, providing a flexible tool to study backdoor behavior for bridge models. However, a key question arises: can a single DSB model train multiple backdoor triggers? Unfortunately, our theory shows that when attempting this, the model ends up following the geometric mean of benign and backdoored distributions, leading to performance conflict across backdoor tasks. To overcome this, we propose a Divide-and-Merge strategy to mix different bridges, where models are independently pre-trained for each specific objective (Divide) and then integrated into a unified model (Merge). In addition, a Weight Reallocation Scheme (WRS) is also designed to enhance the stealthiness of MixBridge. Empirical studies across diverse generation tasks speak to the efficacy of MixBridge. 

**Abstract (ZH)**: 基于桥梁扩散模型的混合异构后门触发机制研究 

---
# SparseMeXT Unlocking the Potential of Sparse Representations for HD Map Construction 

**Title (ZH)**: SparseMeXT 解锁稀疏表示在高清地图构建中的潜力 

**Authors**: Anqing Jiang, Jinhao Chai, Yu Gao, Yiru Wang, Yuwen Heng, Zhigang Sun, Hao Sun, Zezhong Zhao, Li Sun, Jian Zhou, Lijuan Zhu, Shugong Xu, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08808)  

**Abstract**: Recent advancements in high-definition \emph{HD} map construction have demonstrated the effectiveness of dense representations, which heavily rely on computationally intensive bird's-eye view \emph{BEV} features. While sparse representations offer a more efficient alternative by avoiding dense BEV processing, existing methods often lag behind due to the lack of tailored designs. These limitations have hindered the competitiveness of sparse representations in online HD map construction. In this work, we systematically revisit and enhance sparse representation techniques, identifying key architectural and algorithmic improvements that bridge the gap with--and ultimately surpass--dense approaches. We introduce a dedicated network architecture optimized for sparse map feature extraction, a sparse-dense segmentation auxiliary task to better leverage geometric and semantic cues, and a denoising module guided by physical priors to refine predictions. Through these enhancements, our method achieves state-of-the-art performance on the nuScenes dataset, significantly advancing HD map construction and centerline detection. Specifically, SparseMeXt-Tiny reaches a mean average precision \emph{mAP} of 55.5% at 32 frames per second \emph{fps}, while SparseMeXt-Base attains 65.2% mAP. Scaling the backbone and decoder further, SparseMeXt-Large achieves an mAP of 68.9% at over 20 fps, establishing a new benchmark for sparse representations in HD map construction. These results underscore the untapped potential of sparse methods, challenging the conventional reliance on dense representations and redefining efficiency-performance trade-offs in the field. 

**Abstract (ZH)**: Recent advancements in高分辨率地图构建：SparseMeXt系列方法的架构与算法优化及其实时高性能实现 

---
# Security of Internet of Agents: Attacks and Countermeasures 

**Title (ZH)**: 代理互联网的安全性：攻击与对策 

**Authors**: Yuntao Wang, Yanghe Pan, Shaolong Guo, Zhou Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.08807)  

**Abstract**: With the rise of large language and vision-language models, AI agents have evolved into autonomous, interactive systems capable of perception, reasoning, and decision-making. As they proliferate across virtual and physical domains, the Internet of Agents (IoA) has emerged as a key infrastructure for enabling scalable and secure coordination among heterogeneous agents. This survey offers a comprehensive examination of the security and privacy landscape in IoA systems. We begin by outlining the IoA architecture and its distinct vulnerabilities compared to traditional networks, focusing on four critical aspects: identity authentication threats, cross-agent trust issues, embodied security, and privacy risks. We then review existing and emerging defense mechanisms and highlight persistent challenges. Finally, we identify open research directions to advance the development of resilient and privacy-preserving IoA ecosystems. 

**Abstract (ZH)**: 随着大规模语言模型和多模态语言模型的发展，AI代理已进化为能够进行感知、推理和决策的自主交互系统。随着它们在虚拟和物理领域中的普及，代理互联网（IoA）已成为实现异构代理之间可扩展和安全协调的关键基础设施。本文提供了对IoA系统安全和隐私格局的全面考察。我们首先概述了IoA架构及其与传统网络相比的独特漏洞，重点关注四方面关键内容：身份认证威胁、跨代理信任问题、实体安全和隐私风险。然后，我们回顾现有的和新兴的防御机制，并突出存在的持续挑战。最后，我们确定了开放的研究方向，以促进稳健且隐私保护的IoA生态系统的进一步发展。 

---
# Multi-modal Synthetic Data Training and Model Collapse: Insights from VLMs and Diffusion Models 

**Title (ZH)**: 多模态合成数据训练与模型崩溃：来自VLMs和扩散模型的见解 

**Authors**: Zizhao Hu, Mohammad Rostami, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2505.08803)  

**Abstract**: Recent research has highlighted the risk of generative model collapse, where performance progressively degrades when continually trained on self-generated data. However, existing exploration on model collapse is limited to single, unimodal models, limiting our understanding in more realistic scenarios, such as diverse multi-modal AI agents interacting autonomously through synthetic data and continually evolving. We expand the synthetic data training and model collapse study to multi-modal vision-language generative systems, such as vision-language models (VLMs) and text-to-image diffusion models, as well as recursive generate-train loops with multiple models. We find that model collapse, previously observed in single-modality generative models, exhibits distinct characteristics in the multi-modal context, such as improved vision-language alignment and increased variance in VLM image-captioning task. Additionally, we find that general approaches such as increased decoding budgets, greater model diversity, and relabeling with frozen models can effectively mitigate model collapse. Our findings provide initial insights and practical guidelines for reducing the risk of model collapse in self-improving multi-agent AI systems and curating robust multi-modal synthetic datasets. 

**Abstract (ZH)**: 近期研究强调了生成模型坍塌的风险，当持续用自动生成的数据训练时，模型性能会逐渐下降。然而，现有对模型坍塌的研究主要限于单一模态模型，限制了我们对更现实场景的理解，例如通过合成数据自主交互和不断进化的多模态AI代理。我们将合成数据训练和模型坍塌研究扩展到多模态视觉-语言生成系统，如视觉-语言模型（VLMs）和文本到图像扩散模型，以及涉及多个模型的递归生成-训练循环中。我们发现，单一模态生成模型中观察到的模型坍塌在多模态环境中表现出不同的特征，例如视觉-语言对齐改进和VLM图像配对任务中方差增加。此外，我们发现，增加解码预算、提高模型多样性以及使用冻结模型重新标注等通用方法可以有效缓解模型坍塌。我们的发现为减少自我提升多代理AI系统中模型坍塌的风险以及编目稳健的多模态合成数据集提供了初步见解和实用指南。 

---
# Graph-based Online Monitoring of Train Driver States via Facial and Skeletal Features 

**Title (ZH)**: 基于图的列车司机状态在线监测方法： Facial和Skeletal特征分析 

**Authors**: Olivia Nocentini, Marta Lagomarsino, Gokhan Solak, Younggeol Cho, Qiyi Tong, Marta Lorenzini, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2505.08800)  

**Abstract**: Driver fatigue poses a significant challenge to railway safety, with traditional systems like the dead-man switch offering limited and basic alertness checks. This study presents an online behavior-based monitoring system utilizing a customised Directed-Graph Neural Network (DGNN) to classify train driver's states into three categories: alert, not alert, and pathological. To optimize input representations for the model, an ablation study was performed, comparing three feature configurations: skeletal-only, facial-only, and a combination of both. Experimental results show that combining facial and skeletal features yields the highest accuracy (80.88%) in the three-class model, outperforming models using only facial or skeletal features. Furthermore, this combination achieves over 99% accuracy in the binary alertness classification. Additionally, we introduced a novel dataset that, for the first time, incorporates simulated pathological conditions into train driver monitoring, broadening the scope for assessing risks related to fatigue and health. This work represents a step forward in enhancing railway safety through advanced online monitoring using vision-based technologies. 

**Abstract (ZH)**: 基于定向图神经网络的在线行为监测系统：结合面部和骨骼特征分类列车司机状态以提升铁路安全 

---
# In-Context Learning for Label-Efficient Cancer Image Classification in Oncology 

**Title (ZH)**: 基于上下文的学习在肿瘤学中的标签高效癌症图像分类 

**Authors**: Mobina Shrestha, Bishwas Mandal, Vishal Mandal, Asis Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2505.08798)  

**Abstract**: The application of AI in oncology has been limited by its reliance on large, annotated datasets and the need for retraining models for domain-specific diagnostic tasks. Taking heed of these limitations, we investigated in-context learning as a pragmatic alternative to model retraining by allowing models to adapt to new diagnostic tasks using only a few labeled examples at inference, without the need for retraining. Using four vision-language models (VLMs)-Paligemma, CLIP, ALIGN and GPT-4o, we evaluated the performance across three oncology datasets: MHIST, PatchCamelyon and HAM10000. To the best of our knowledge, this is the first study to compare the performance of multiple VLMs on different oncology classification tasks. Without any parameter updates, all models showed significant gains with few-shot prompting, with GPT-4o reaching an F1 score of 0.81 in binary classification and 0.60 in multi-class classification settings. While these results remain below the ceiling of fully fine-tuned systems, they highlight the potential of ICL to approximate task-specific behavior using only a handful of examples, reflecting how clinicians often reason from prior cases. Notably, open-source models like Paligemma and CLIP demonstrated competitive gains despite their smaller size, suggesting feasibility for deployment in computing constrained clinical environments. Overall, these findings highlight the potential of ICL as a practical solution in oncology, particularly for rare cancers and resource-limited contexts where fine-tuning is infeasible and annotated data is difficult to obtain. 

**Abstract (ZH)**: AI在肿瘤学中的应用受限于其对大规模标注数据的依赖以及需要为领域特定诊断任务重新训练模型。鉴于这些限制，我们研究了上下文学习作为一种实用的替代方案，允许模型仅通过少量标注示例在推理时适应新的诊断任务，而无需重新训练。我们使用四种视觉-语言模型（VLMs）——Paligemma、CLIP、ALIGN和GPT-4o，在三个肿瘤学数据集（MHIST、PatchCamelyon和HAM10000）上评估了性能。据我们所知，这是首次将多种VLMs在不同的肿瘤分类任务上进行性能比较的研究。无需任何参数更新，所有模型在少样本提示下均显示出显著提升，其中GPT-4o在二分类和多分类设置下的F1分数分别为0.81和0.60。虽然这些结果仍低于完全微调系统的天花板，但它们突显了上下文学习通过少量示例逼近任务特定行为的潜力，反映出临床医生通常如何从以往病例中推理。值得注意的是，开源模型Paligemma和CLIP尽管规模较小，但仍然表现出竞争力的提升，表明其在计算资源受限的临床环境中部署的可行性。总体而言，这些发现强调了上下文学习作为肿瘤学中实用解决方案的潜力，特别是在肿瘤罕见且资源有限的情况下，微调不可行且标注数据难以获取的背景下。 

---
# The Geometry of Meaning: Perfect Spacetime Representations of Hierarchical Structures 

**Title (ZH)**: 意义的几何学：层级结构的完美时空表示 

**Authors**: Andres Anabalon, Hugo Garces, Julio Oliva, Jose Cifuentes  

**Link**: [PDF](https://arxiv.org/pdf/2505.08795)  

**Abstract**: We show that there is a fast algorithm that embeds hierarchical structures in three-dimensional Minkowski spacetime. The correlation of data ends up purely encoded in the causal structure. Our model relies solely on oriented token pairs -- local hierarchical signals -- with no access to global symbolic structure. We apply our method to the corpus of \textit{WordNet}. We provide a perfect embedding of the mammal sub-tree including ambiguities (more than one hierarchy per node) in such a way that the hierarchical structures get completely codified in the geometry and exactly reproduce the ground-truth. We extend this to a perfect embedding of the maximal unambiguous subset of the \textit{WordNet} with 82{,}115 noun tokens and a single hierarchy per token. We introduce a novel retrieval mechanism in which causality, not distance, governs hierarchical access. Our results seem to indicate that all discrete data has a perfect geometrical representation that is three-dimensional. The resulting embeddings are nearly conformally invariant, indicating deep connections with general relativity and field theory. These results suggest that concepts, categories, and their interrelations, namely hierarchical meaning itself, is geometric. 

**Abstract (ZH)**: 我们展示了如何在闵可斯基三维时空中标注层次结构的快速算法，数据的相关性最终完全编码在因果结构中。我们的模型仅依赖于定向令牌对——局部层次信号——无需访问全局符号结构。我们将该方法应用于WordNet语料库。我们提供了哺乳动物子树（包括每个节点有多个层次的情况）的完美嵌入，并且层次结构在几何中完全编码且精确地再现了真实情况。我们将其扩展到WordNet的最大无歧义子集的完美嵌入，该集合包含82,115个名词令牌且每个令牌只有一个层次。我们引入了一种新的检索机制，其中因果关系而非距离支配层次访问。结果显示所有离散数据可能都有一个完美的三维几何表示。生成的嵌入几乎共形不变，暗示了与广义相对论和场论的深刻联系。这些结果表明，概念、类别及其相互关系，即层次意义本身，是几何的。 

---
# A Retrieval-Augmented Generation Framework for Academic Literature Navigation in Data Science 

**Title (ZH)**: 面向数据科学中的学术文献导航的检索增强生成框架 

**Authors**: Ahmet Yasin Aytar, Kemal Kilic, Kamer Kaya  

**Link**: [PDF](https://arxiv.org/pdf/2412.15404)  

**Abstract**: In the rapidly evolving field of data science, efficiently navigating the expansive body of academic literature is crucial for informed decision-making and innovation. This paper presents an enhanced Retrieval-Augmented Generation (RAG) application, an artificial intelligence (AI)-based system designed to assist data scientists in accessing precise and contextually relevant academic resources. The AI-powered application integrates advanced techniques, including the GeneRation Of BIbliographic Data (GROBID) technique for extracting bibliographic information, fine-tuned embedding models, semantic chunking, and an abstract-first retrieval method, to significantly improve the relevance and accuracy of the retrieved information. This implementation of AI specifically addresses the challenge of academic literature navigation. A comprehensive evaluation using the Retrieval-Augmented Generation Assessment System (RAGAS) framework demonstrates substantial improvements in key metrics, particularly Context Relevance, underscoring the system's effectiveness in reducing information overload and enhancing decision-making processes. Our findings highlight the potential of this enhanced Retrieval-Augmented Generation system to transform academic exploration within data science, ultimately advancing the workflow of research and innovation in the field. 

**Abstract (ZH)**: 在数据科学快速发展的领域，有效地导航庞大的学术文献对于做出明智决策和创新至关重要。本文介绍了一种增强的检索增强生成（RAG）应用，这是一种基于人工智能的系统，旨在帮助数据科学家访问精确且上下文相关的学术资源。该人工智能驱动的应用集成先进的技术，包括用于提取参考文献信息的GeneRation Of BIbliographic Data（GROBID）技术、微调嵌入模型、语义分块以及摘要优先检索方法，以显著提高检索信息的相关性和准确性。该人工智能实现特别解决了学术文献导航的挑战。使用检索增强生成评估系统（RAGAS）框架进行的全面评估表明，在关键指标方面取得了显著改进，特别是在上下文相关性方面，突显了该系统的有效性，能够减少信息过载并提升决策过程。我们的研究结果强调了这种增强的检索增强生成系统在数据科学中改变学术探究的潜力，最终推动了研究和创新工作的流程。 

---
