# LLM-Supported Formal Knowledge Representation for Enhancing Control Engineering Content with an Interactive Semantic Layer 

**Title (ZH)**: 基于LLM的支持形式化知识表示以增强交互语义层的控制工程内容 

**Authors**: Julius Fiedler, Carsten Knoll, Klaus Röbenack  

**Link**: [PDF](https://arxiv.org/pdf/2511.02759)  

**Abstract**: The rapid growth of research output in control engineering calls for new approaches to structure and formalize domain knowledge. This paper briefly describes an LLM-supported method for semi-automated generation of formal knowledge representations that combine human readability with machine interpretability and increased expressiveness. Based on the Imperative Representation of Knowledge (PyIRK) framework, we demonstrate how language models can assist in transforming natural-language descriptions and mathematical definitions (available as LaTeX source code) into a formalized knowledge graph. As a first application we present the generation of an ``interactive semantic layer'' to enhance the source documents in order to facilitate knowledge transfer. From our perspective this contributes to the vision of easily accessible, collaborative, and verifiable knowledge bases for the control engineering domain. 

**Abstract (ZH)**: 控制工程领域研究成果的快速发展需要新的方法来结构化和形式化领域知识。本文简要描述了一种基于LLM的支持半自动化生成结合人类可读性和机器可解释性的形式知识表示的方法。基于 Imperative Representation of Knowledge (PyIRK) 框架，我们展示了语言模型如何辅助将自然语言描述和数学定义（以LaTeX源代码形式提供）转换为形式化的知识图谱。作为第一个应用，我们展示了生成一个“交互式语义层”的方法，以增强源文档，从而促进知识转移。从我们的角度看，这有助于实现控制工程领域易于访问、合作共享和可验证的知识库的愿景。 

---
# CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents 

**Title (ZH)**: CostBench: 评估动态环境中文本工具使用智能体多轮成本最优规划与适应性评估 

**Authors**: Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2511.02734)  

**Abstract**: Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents' ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents' economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and necessitate agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than 75% exact match rate on the hardest tasks, and performance further dropping by around 40% under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust. 

**Abstract (ZH)**: 当前对大型语言模型代理的评估主要侧重于任务完成，往往忽视了资源效率和适应性。这忽略了代理根据环境变化制定和调整成本最优计划的关键能力。为弥补这一差距，我们提出了CostBench，一个可扩展、以成本为中心的基准测试，旨在评估代理的经济推理和重新规划能力。CostBench设在旅行规划领域，包含可通过多种原子和复合工具序列解决的任务，这些任务具有多样且可定制的成本。它还支持四种类型的动态阻碍事件，如工具故障和成本变化，以模拟现实世界的不可预测性，并要求代理实时调整。在CostBench上评估领先的开源和专有模型显示了显著的成本意识规划差距：代理在静态设置中经常无法识别成本最优解决方案，即使是GPT-5在最困难的任务上精确匹配率也低于75%，而在动态条件下，性能进一步下降约40%。通过对这些弱点的诊断，CostBench为开发既经济合理又稳健的未来代理奠定了基础。 

---
# DecompSR: A dataset for decomposed analyses of compositional multihop spatial reasoning 

**Title (ZH)**: DecompSR：一个用于组成式多跳空间推理分解分析的数据集 

**Authors**: Lachlan McPheat, Navdeep Kaur, Robert Blackwell, Alessandra Russo, Anthony G. Cohn, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2511.02627)  

**Abstract**: We introduce DecompSR, decomposed spatial reasoning, a large benchmark dataset (over 5m datapoints) and generation framework designed to analyse compositional spatial reasoning ability. The generation of DecompSR allows users to independently vary several aspects of compositionality, namely: productivity (reasoning depth), substitutivity (entity and linguistic variability), overgeneralisation (input order, distractors) and systematicity (novel linguistic elements). DecompSR is built procedurally in a manner which makes it is correct by construction, which is independently verified using a symbolic solver to guarantee the correctness of the dataset. DecompSR is comprehensively benchmarked across a host of Large Language Models (LLMs) where we show that LLMs struggle with productive and systematic generalisation in spatial reasoning tasks whereas they are more robust to linguistic variation. DecompSR provides a provably correct and rigorous benchmarking dataset with a novel ability to independently vary the degrees of several key aspects of compositionality, allowing for robust and fine-grained probing of the compositional reasoning abilities of LLMs. 

**Abstract (ZH)**: DecompSR：分解的空间推理，一个大型基准数据集及生成框架 

---
# The ORCA Benchmark: Evaluating Real-World Calculation Accuracy in Large Language Models 

**Title (ZH)**: ORCA 基准：评估大型语言模型在实际计算中的准确度 

**Authors**: Claudia Herambourg, Dawid Siuda, Anna Szczepanek, Julia Kopczyńska, Joao R. L. Santos, Wojciech Sas, Joanna Śmietańska-Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02589)  

**Abstract**: We present ORCA (Omni Research on Calculation in AI) Benchmark -- a novel benchmark that evaluates large language models (LLMs) on multi-domain, real-life quantitative reasoning using verified outputs from Omni's calculator engine. In 500 natural-language tasks across domains such as finance, physics, health, and statistics, the five state-of-the-art systems (ChatGPT-5, Gemini~2.5~Flash, Claude~Sonnet~4.5, Grok~4, and DeepSeek~V3.2) achieved only $45\text{--}63\,\%$ accuracy, with errors mainly related to rounding ($35\,\%$) and calculation mistakes ($33\,\%$). Results in specific domains indicate strengths in mathematics and engineering, but weaknesses in physics and natural sciences. Correlation analysis ($r \approx 0.40\text{--}0.65$) shows that the models often fail together but differ in the types of errors they make, highlighting their partial complementarity rather than redundancy. Unlike standard math datasets, ORCA evaluates step-by-step reasoning, numerical precision, and domain generalization across real problems from finance, physics, health, and statistics. 

**Abstract (ZH)**: Omni Research on Calculation in AI基准：一种评估大型语言模型多领域实际量化推理能力的新基准 

---
# Knowledge Graph-enhanced Large Language Model for Incremental Game PlayTesting 

**Title (ZH)**: 基于知识图谱增强的大语言模型在增量游戏测试中的应用 

**Authors**: Enhong Mu, Jinyu Cai, Yijun Lu, Mingyue Zhang, Kenji Tei, Jialong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02534)  

**Abstract**: The rapid iteration and frequent updates of modern video games pose significant challenges to the efficiency and specificity of testing. Although automated playtesting methods based on Large Language Models (LLMs) have shown promise, they often lack structured knowledge accumulation mechanisms, making it difficult to conduct precise and efficient testing tailored for incremental game updates. To address this challenge, this paper proposes a KLPEG framework. The framework constructs and maintains a Knowledge Graph (KG) to systematically model game elements, task dependencies, and causal relationships, enabling knowledge accumulation and reuse across versions. Building on this foundation, the framework utilizes LLMs to parse natural language update logs, identify the scope of impact through multi-hop reasoning on the KG, enabling the generation of update-tailored test cases. Experiments in two representative game environments, Overcooked and Minecraft, demonstrate that KLPEG can more accurately locate functionalities affected by updates and complete tests in fewer steps, significantly improving both playtesting effectiveness and efficiency. 

**Abstract (ZH)**: 现代视频游戏的快速迭代和频繁更新对测试的效率和针对性提出了重大挑战。虽然基于大型语言模型（LLM）的自动化测试方法前景可期，但它们通常缺乏结构化的知识积累机制，使得难以进行精准和高效的增量游戏更新测试。为应对这一挑战，本文提出了一种KLPEG框架。该框架构建并维护一个知识图谱（KG），以系统性地建模游戏元素、任务依赖关系和因果关系，实现版本间的知识积累和重用。在此基础上，框架利用LLM解析自然语言更新日志，并通过KG上的多跳推理识别影响范围，从而生成针对更新的测试案例。在两个代表性游戏环境Overcooked和Minecraft中的实验表明，KLPEG能够更准确地定位受到更新影响的功能性，并以较少的步骤完成测试，显著提高了测试的有效性和效率。 

---
# Auditable-choice reframing unlocks RL-based verification for open-ended tasks 

**Title (ZH)**: 审计选择重构解锁基于强化学习的开放性任务验证 

**Authors**: Mengyu Zhang, Xubo Liu, Siyu Ding, Weichong Yin, Yu Sun, Hua Wu, Wenya Guo, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02463)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated great potential in enhancing the reasoning capabilities of large language models (LLMs), achieving remarkable progress in domains such as mathematics and programming where standard answers are available. However, for open-ended tasks lacking ground-truth solutions (e.g., creative writing and instruction following), existing studies typically regard them as non-reasoning scenarios, thereby overlooking the latent value of reasoning capabilities. This raises a key question: Can strengthening reasoning improve performance in open-ended tasks? To address this, we explore the transfer of the RLVR paradigm to the open domain. Yet, since RLVR fundamentally relies on verifiers that presuppose the existence of standard answers, it cannot be directly applied to open-ended tasks. To overcome this challenge, we introduce Verifiable Multiple-Choice Reformulation (VMR), a novel training strategy that restructures open-ended data into verifiable multiple-choice formats, enabling effective training even in the absence of explicit ground truth. Experimental results on multiple benchmarks validate the effectiveness of our method in improving LLM performance on open-ended tasks. Notably, across eight open-ended benchmarks, our VMR-based training delivers an average gain of 5.99 points over the baseline. Code will be released upon acceptance to facilitate reproducibility. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）在提升大型语言模型（LLMs）的推理能力方面显示出巨大潜力，特别是在数学和编程等领域取得了显著进展。然而，对于缺乏标准答案的开放性任务（如创造性写作和指令跟随），现有研究通常将这些任务视为非推理场景，从而忽视了推理能力的潜在价值。这引发了一个关键问题：增强推理是否能改善开放性任务的表现？为了解决这一问题，我们探讨了将RLVR范式应用到开放领域的方法。但由于RLVR从根本上依赖于基于标准答案存在的验证器，无法直接应用于开放性任务。为克服这一挑战，我们引入了可验证的多选重构（VMR）这一新型训练策略，将开放性数据重新构建成可验证的多选格式，即使在缺少明确标准答案的情况下也能实现有效训练。通过多个基准测试的实验结果验证了我们方法在提升LLM在开放性任务上的表现方面的有效性。特别地，我们的VMR基训练方法在八个开放性基准测试中平均提高了5.99分。接受后将发布代码以促进可重复性。 

---
# ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning 

**Title (ZH)**: ReAcTree：具有控制流的层次LLM代理树长时_horizon任务规划 

**Authors**: Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Minsu Jang, Dohyung Kim, Jaehong Kim, Youngwoo Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.02424)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled significant progress in decision-making and task planning for embodied autonomous agents. However, most existing methods still struggle with complex, long-horizon tasks because they rely on a monolithic trajectory that entangles all past decisions and observations, attempting to solve the entire task in a single unified process. To address this limitation, we propose ReAcTree, a hierarchical task-planning method that decomposes a complex goal into more manageable subgoals within a dynamically constructed agent tree. Each subgoal is handled by an LLM agent node capable of reasoning, acting, and further expanding the tree, while control flow nodes coordinate the execution strategies of agent nodes. In addition, we integrate two complementary memory systems: each agent node retrieves goal-specific, subgoal-level examples from episodic memory and shares environment-specific observations through working memory. Experiments on the WAH-NL and ALFRED datasets demonstrate that ReAcTree consistently outperforms strong task-planning baselines such as ReAct across diverse LLMs. Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5 72B, nearly doubling ReAct's 31%. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs） recently在大型语言模型（LLMs）方面的进展已经使自主实体代理在决策和任务规划方面的进步成为可能。然而，现有的大多数方法仍然难以应对复杂的、长期的任务，因为它们依赖于一个将所有过去决策和观察结果编织在一起的单一轨迹，试图一次性解决整个任务。为了克服这一局限性，我们提出了一种名为ReAcTree的分层任务规划方法，该方法在动态构建的代理树中将复杂的目标分解为更易管理的子目标。每个子目标由一个能够推理、行动并进一步扩展树的LLM代理节点处理，而控制流节点负责协调代理节点的执行策略。此外，我们整合了两个互补的记忆系统：每个代理节点从情景记忆中检索与目标相关的子目标级示例，并通过工作记忆分享与环境相关的观察结果。在WAH-NL和ALFRED数据集上的实验表明，ReAcTree在各种LLM中始终优于现有的强大任务规划基线ReAct。特别是在WAH-NL数据集上，使用Qwen 2.5 72B时，ReAcTree的目标成功率为61%，几乎是ReAct（31%）的两倍。 

---
# Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation 

**Title (ZH)**: 解锁多Agent大语言模型的推理力量：从懒惰Agent到协同推理 

**Authors**: Zhiwei Zhang, Xiaomin Li, Yudi Lin, Hui Liu, Ramraj Chandradevan, Linlin Wu, Minhua Lin, Fali Wang, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02303)  

**Abstract**: Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过强化学习和可验证奖励训练，在复杂推理任务中取得了显著成果。近期工作将这一范式扩展到多agent环境，其中元思考agent提出计划并监控进度，而推理agent通过顺序对话轮次执行子任务。尽管表现出色，但我们发现一个关键限制：懒惰agents的行为，其中一个agent占据主导地位而另一个贡献甚微，削弱了合作并使设置退化为无效的单agent设置。在本文中，我们首先提供理论分析，说明为什么懒惰行为在多agent推理中自然发生。接着，我们引进了一种稳定且高效的因果影响度量方法，有助于缓解这一问题。最后，随着合作的增强，推理agent可能会陷入多轮交互中，并被之前嘈杂的响应所困。为此，我们提出了一种可验证奖励机制，鼓励反思，允许推理agent丢弃嘈杂输出、整合指令，并在必要时重启其推理过程。大量实验表明，我们的框架缓解了懒惰agents的行为，并释放了多agent框架在复杂推理任务中的全部潜力。 

---
# When Modalities Conflict: How Unimodal Reasoning Uncertainty Governs Preference Dynamics in MLLMs 

**Title (ZH)**: 当模态冲突时：单模态推理不确定性如何治理多模态大型语言模型中的偏好动力学 

**Authors**: Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02243)  

**Abstract**: Multimodal large language models (MLLMs) must resolve conflicts when different modalities provide contradictory information, a process we term modality following. Prior work measured this behavior only with coarse dataset-level statistics, overlooking the influence of model's confidence in unimodal reasoning. In this paper, we introduce a new framework that decomposes modality following into two fundamental factors: relative reasoning uncertainty (the case-specific confidence gap between unimodal predictions) and inherent modality preference( a model's stable bias when uncertainties are balanced). To validate this framework, we construct a controllable dataset that systematically varies the reasoning difficulty of visual and textual inputs. Using entropy as a fine-grained uncertainty metric, we uncover a universal law: the probability of following a modality decreases monotonically as its relative uncertainty increases. At the relative difficulty level where the model tends to follow both modalities with comparable probability what we call the balance point, a practical indicator of the model's inherent preference. Unlike traditional macro-level ratios, this measure offers a more principled and less confounded way to characterize modality bias, disentangling it from unimodal capabilities and dataset artifacts. Further, by probing layer-wise predictions, we reveal the internal mechanism of oscillation: in ambiguous regions near the balance point, models vacillate between modalities across layers, explaining externally observed indecision. Together, these findings establish relative uncertainty and inherent preference as the two governing principles of modality following, offering both a quantitative framework and mechanistic insight into how MLLMs resolve conflicting information. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）必须解决不同模态提供矛盾信息时的冲突，这一过程我们称之为模态跟随。此前的研究仅通过粗略的数据集级统计衡量这种行为，忽视了模型在单模态推理中的信心影响。本文引入了一个新的框架，将模态跟随分解为两个基本因素：相对推理不确定性（具体情况下的单模态预测之间的信心差距）和固有模态偏好（在不确定性平衡时模型的稳定偏差）。为了验证这一框架，我们构建了一个可控的数据集，系统地变化视觉和文本输入的推理难度。使用熵作为精细的信心度量标准，我们揭示了一条普遍定律：随着相对不确定性增加，跟随特定模态的概率单调递减。在模型倾向于以相似概率跟随两种模态的相对难度水平——我们称之为平衡点——处，这是一种模型固有偏好的一种实用指标。与传统的宏观比例不同，这一措施提供了一种更为原则性和不混淆的方式来表征模态偏好，从而将其与单模态能力和数据集伪影区分开来。进一步地，通过探究逐层预测，我们揭示了振荡的内部机制：在接近平衡点的模糊区域，模型在各层之间摇摆，解释了外部观察到的犹豫。总之，这些发现确立了相对不确定性与固有偏好作为模态跟随的两个治理原则，提供了量化框架和机制见解，解释了MLLMs如何解决冲突信息。 

---
# Deep Ideation: Designing LLM Agents to Generate Novel Research Ideas on Scientific Concept Network 

**Title (ZH)**: 深层次创意思维：设计LLM代理以在科学研究概念网络中生成新颖研究理念 

**Authors**: Keyu Zhao, Weiquan Lin, Qirui Zheng, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02238)  

**Abstract**: Novel research ideas play a critical role in advancing scientific inquiries. Recent advancements in Large Language Models (LLMs) have demonstrated their potential to generate novel research ideas by leveraging large-scale scientific literature. However, previous work in research ideation has primarily relied on simplistic methods, such as keyword co-occurrence or semantic similarity. These approaches focus on identifying statistical associations in the literature but overlook the complex, contextual relationships between scientific concepts, which are essential to effectively leverage knowledge embedded in human literature. For instance, papers that simultaneously mention "keyword A" and "keyword B" often present research ideas that integrate both concepts. Additionally, some LLM-driven methods propose and refine research ideas using the model's internal knowledge, but they fail to effectively utilize the scientific concept network, limiting the grounding of ideas in established research. To address these challenges, we propose the Deep Ideation framework to address these challenges, integrating a scientific network that captures keyword co-occurrence and contextual relationships, enriching LLM-driven ideation. The framework introduces an explore-expand-evolve workflow to iteratively refine research ideas, using an Idea Stack to track progress. A critic engine, trained on real-world reviewer feedback, guides the process by providing continuous feedback on the novelty and feasibility of ideas. Our experiments show that our approach improves the quality of generated ideas by 10.67% compared to other methods, with ideas surpassing top conference acceptance levels. Human evaluation highlights their practical value in scientific research, and ablation studies confirm the effectiveness of each component in the workflow. Code repo is available at this https URL. 

**Abstract (ZH)**: 新颖的研究理念在推进科学探究中扮演着关键角色。近期大型语言模型（LLMs）的进步展示了其通过利用大量科学文献生成新颖研究理念的潜力。然而，以往的研究理念生成工作主要依赖于简单的关键词共现或语义相似性方法，这些方法侧重于识别文献中的统计关联性，而忽视了科学研究概念之间的复杂上下文关系，这些关系对于有效利用嵌入在人类文献中的知识至关重要。例如，同时提到“关键词A”和“关键词B”的论文常常提出了结合这两个概念的研究理念。此外，一些由LLM驱动的方法通过模型内部知识提出和细化研究理念，但未能有效利用科学概念网络，限制了理念的稳固性。为应对这些挑战，我们提出了Deep Ideation框架，该框架结合了一个捕捉关键词共现和上下文关系的科学网络，丰富了由LLM驱动的研究理念生成。该框架引入了一个探索-扩展-演化的 workflows，通过一个Idea Stack跟踪进展，并由一个基于真实评审反馈训练的批评引擎指导，持续提供创新性和可行性反馈。实验证明，与其它方法相比，我们的方法能使生成的理念质量提高10.67%，其中理念甚至超过了顶级会议的接受水平。人类评估突显了其在科学研究中的实用价值，并且消除实验验证了 workflows中每个组件的有效性。代码仓库可在以下链接访问。 

---
# Training Proactive and Personalized LLM Agents 

**Title (ZH)**: 训练主动且个性化的语言模型代理 

**Authors**: Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02208)  

**Abstract**: While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents. 

**Abstract (ZH)**: 现有的工作主要关注任务成功，我们认为有效的现实世界代理需要优化三个维度：生产力（任务完成）、主动性（提出关键问题）和个人化（适应多样化的用户偏好）。我们引入了UserVille，这是一个基于LLM的用户模拟器的互动环境，使得用户偏好多样化且可配置。利用UserVille，我们引入了PPP，这是一种多目标强化学习方法，能够同时优化这三个维度：生产力、主动性和个人化。在软件工程和深度研究任务上的实验表明，用PPP训练的代理相对于强大的基线（如GPT-5）在平均任务成功率上有21.6%的提升，展示了提出战略澄清问题、适应未见过的用户偏好以及通过更好的交互提高任务成功率的能力。本工作证明了明确优化用户中心的交互对于构建实用和有效的AI代理至关重要。 

---
# Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning 

**Title (ZH)**: 个性化决策建模：效用优化或文本化符号推理 

**Authors**: Yibo Zhao, Yang Zhao, Hongru Du, Hao Frank Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02194)  

**Abstract**: Decision-making models for individuals, particularly in high-stakes scenarios like vaccine uptake, often diverge from population optimal predictions. This gap arises from the uniqueness of the individual decision-making process, shaped by numerical attributes (e.g., cost, time) and linguistic influences (e.g., personal preferences and constraints). Developing upon Utility Theory and leveraging the textual-reasoning capabilities of Large Language Models (LLMs), this paper proposes an Adaptive Textual-symbolic Human-centric Reasoning framework (ATHENA) to address the optimal information integration. ATHENA uniquely integrates two stages: First, it discovers robust, group-level symbolic utility functions via LLM-augmented symbolic discovery; Second, it implements individual-level semantic adaptation, creating personalized semantic templates guided by the optimal utility to model personalized choices. Validated on real-world travel mode and vaccine choice tasks, ATHENA consistently outperforms utility-based, machine learning, and other LLM-based models, lifting F1 score by at least 6.5% over the strongest cutting-edge models. Further, ablation studies confirm that both stages of ATHENA are critical and complementary, as removing either clearly degrades overall predictive performance. By organically integrating symbolic utility modeling and semantic adaptation, ATHENA provides a new scheme for modeling human-centric decisions. The project page can be found at this https URL. 

**Abstract (ZH)**: 个体在疫苗接种等高风险场景中的决策模型往往与总体最优预测存在差异。这种差异源于个体决策过程的独特性，受到数值属性（如成本、时间）和语言影响（如个人偏好和约束条件）的塑造。基于效用理论并利用大型语言模型的文本推理能力，本文提出了一种自适应文本-符号人类中心推理框架（ATHENA）以解决最优信息整合问题。ATHENA 阶段性地整合了两个步骤：首先，通过大型语言模型增强的符号发现技术，发现稳健的分组级符号效用函数；其次，实现个性化语义适应，基于最优效用创建个性化的语义模板，以建模个性化选择。ATHENA 在实际旅行模式选择和疫苗选择任务上的表现优于基于效用的机器学习模型及其他基于大型语言模型的模型，F1 分数提高至少 6.5%，超过最强前沿模型。进一步的消融研究证实，ATHENA 的两个阶段都是关键且互补的，移除任何一个都会明显降低整体预测性能。通过有机地结合符号效用建模和语义适应，ATHENA 为建模以人类为中心的决策提供了新的方案。项目页面可访问此[链接]。 

---
# InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance 

**Title (ZH)**: InsurAgent: 一个大型语言模型赋能的洪水保险购买个体行为模拟代理 

**Authors**: Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02119)  

**Abstract**: Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However, participation rates among at-risk populations in the United States remain strikingly low. This gap underscores the need to understand and model the behavioral mechanisms underlying insurance decisions. Large language models (LLMs) have recently exhibited human-like intelligence across wide-ranging tasks, offering promising tools for simulating human decision-making. This study constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent, an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action, and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey data, capturing contextual information that is intractable for traditional models. The memory module supports the simulation of temporal decision evolutions, illustrated through a roller coaster life trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis. 

**Abstract (ZH)**: Flood保险是个人减轻灾害相关损失的有效策略。然而，美国易受灾人群的参保率仍然异常低。这一差距凸显了理解并建模影响保险决策的行为机制的需求。大型语言模型（LLMs）最近在其广泛任务中表现出类人的智能，为模拟人类决策提供了有希望的工具。本研究构建了一个基准数据集以捕捉因素下的保险购买概率。利用该数据集评估LLMs的能力：虽然LLMs具备对因素的定性理解，但在估算定量概率方面却表现不足。为解决这一局限性，提出了一种基于LLMs的InsurAgent代理，包括感知、检索、推理、行动和记忆五个模块。检索模块利用检索增强生成（RAG）技术将决策建立在实证调查数据的基础上，实现了边际概率和二元概率的准确估算。推理模块利用LLMs的常识超越调查数据进行外推，捕捉传统模型难以处理的上下文信息。记忆模块支持对随时间演变的决策模拟，通过过山车式的生活轨迹进行展示。总体而言，InsurAgent为行为建模和政策分析提供了一个有价值的工具。 

---
# Deep Value Benchmark: Measuring Whether Models Generalize Deep values or Shallow Preferences 

**Title (ZH)**: 深度价值基准：测量模型是否泛化出深层价值观或浅层偏好 

**Authors**: Joshua Ashkinaze, Hua Shen, Sai Avula, Eric Gilbert, Ceren Budak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02109)  

**Abstract**: We introduce the Deep Value Benchmark (DVB), an evaluation framework that directly tests whether large language models (LLMs) learn fundamental human values or merely surface-level preferences. This distinction is critical for AI alignment: Systems that capture deeper values are likely to generalize human intentions robustly, while those that capture only superficial patterns in preference data risk producing misaligned behavior. The DVB uses a novel experimental design with controlled confounding between deep values (e.g., moral principles) and shallow features (e.g., superficial attributes). In the training phase, we expose LLMs to human preference data with deliberately correlated deep and shallow features -- for instance, where a user consistently prefers (non-maleficence, formal language) options over (justice, informal language) alternatives. The testing phase then breaks these correlations, presenting choices between (justice, formal language) and (non-maleficence, informal language) options. This design allows us to precisely measure a model's Deep Value Generalization Rate (DVGR) -- the probability of generalizing based on the underlying value rather than the shallow feature. Across 9 different models, the average DVGR is just 0.30. All models generalize deep values less than chance. Larger models have a (slightly) lower DVGR than smaller models. We are releasing our dataset, which was subject to three separate human validation experiments. DVB provides an interpretable measure of a core feature of alignment. 

**Abstract (ZH)**: Deep Value Benchmark:直接测试大型语言模型是否学习到根本的人类价值观而非表面偏好 

---
# MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning 

**Title (ZH)**: MemSearcher: 通过端到端强化学习训练大规模语言模型进行推理、搜索和管理记忆 

**Authors**: Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.02805)  

**Abstract**: Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at this https URL 

**Abstract (ZH)**: 典型搜索代理将整个交互历史合并到LLMContext中，保持信息完整性但产生长且噪音大的上下文，导致高计算和内存成本。相比之下，只使用当前回合可以避免这种开销但会丢弃关键信息。这种权衡限制了搜索代理的可扩展性。为了解决这一挑战，我们提出MemSearcher，这是一种代理工作流，迭代地维护紧凑的记忆，并将其与当前回合结合。在每一回合中，MemSearcher将用户的问题与记忆融合以生成推理轨迹、执行搜索动作，并更新记忆以仅保留解决问题所需的关键信息。这种设计稳定了多回合交互中的上下文长度，提高了效率而不牺牲准确性。为了优化这一工作流，我们引入了多上下文GRPO，这是一种端到端的强化学习框架，用于同时优化MemSearcher代理的推理、搜索策略和记忆管理。具体而言，多上下文GRPO在不同的上下文中采样轨迹组，并在它们内的所有对话中传递轨迹级别的优势。MemSearcher在与Search-R1相同的数据集上训练，在七个公开基准上显著优于强baseline：在Qwen2.5-3B-Instruct上相对平均增益为+11%，在Qwen2.5-7B-Instruct上为+12%。值得注意的是，基于3B的MemSearcher甚至优于基于7B的baseline，这表明在信息完整性和效率之间取得平衡既提高了准确性又降低了计算开销。代码和模型将在以下链接公开：这个httpsURL。 

---
# Optimal Singular Damage: Efficient LLM Inference in Low Storage Regimes 

**Title (ZH)**: 最优奇异损伤：低存储环境下高效的大语言模型推理 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2511.02681)  

**Abstract**: Large language models (LLMs) are increasingly prevalent across diverse applications. However, their enormous size limits storage and processing capabilities to a few well-resourced stakeholders. As a result, most applications rely on pre-trained LLMs, fine-tuned for specific tasks. However, even storing the fine-tuned versions of these models remains a significant challenge due to the wide range of tasks they address. Recently, studies show that fine-tuning these models primarily affects a small fraction of parameters, highlighting the need for more efficient storage of fine-tuned models. This paper focuses on efficient storage of parameter updates in pre-trained models after fine-tuning. To address this challenge, we leverage the observation that fine-tuning updates are both low-rank and sparse, which can be utilized for storage efficiency. However, using only low-rank approximation or sparsification may discard critical singular components that enhance model expressivity. We first observe that given the same memory budget, sparsified low-rank approximations with larger ranks outperform standard low-rank approximations with smaller ranks. Building on this, we propose our method, optimal singular damage, that selectively sparsifies low-rank approximated updates by leveraging the interleaved importance of singular vectors, ensuring that the most impactful components are retained. We demonstrate through extensive experiments that our proposed methods lead to significant storage efficiency and superior accuracy within the same memory budget compared to employing the low-rank approximation or sparsification individually. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种应用中越来越普及。然而，它们巨大的尺寸限制了存储和处理能力，仅能惠及少数资源充足的利益相关者。因此，大多数应用依赖于为特定任务进行微调的预训练LLMs。但是，即使存储这些微调模型的版本也仍然是一个重大挑战，因为它们涵盖了广泛的任务。最近的研究表明，微调这些模型主要影响小部分参数，突显了更高效存储微调模型的需要。本文关注预训练模型微调后参数更新的高效存储。为了应对这一挑战，我们利用这样一个观察：微调更新既是低秩的也是稀疏的，这可以用于提高存储效率。然而，仅使用低秩逼近或稀疏化可能会丢弃增强模型表示能力的关键奇异成分。我们首先观察到，在相同的内存预算下，较大的秩的稀疏低秩逼近优于较小秩的标准低秩逼近。在此基础上，我们提出了一种方法，即最优奇异损伤，这种方法通过利用奇异向量的交错重要性，选择性地稀疏化低秩逼近的更新，确保保留最具影响力的成分。通过广泛的实验，我们证明了所提出的方法在相同的内存预算下相比单独使用低秩逼近或稀疏化具有显著的存储效率和更好的准确性。 

---
# Apriel-H1: Towards Efficient Enterprise Reasoning Models 

**Title (ZH)**: April-H1: 向高效企业推理模型迈进 

**Authors**: Oleksiy Ostapenko, Luke Kumar, Raymond Li, Denis Kocetkov, Joel Lamy-Poirier, Shruthan Radhakrishna, Soham Parikh, Shambhavi Mishra, Sebastien Paquet, Srinivas Sunkara, Valérie Bécaert, Sathwik Tejaswi Madhusudhan, Torsten Scholak  

**Link**: [PDF](https://arxiv.org/pdf/2511.02651)  

**Abstract**: Large Language Models (LLMs) achieve remarkable reasoning capabilities through transformer architectures with attention mechanisms. However, transformers suffer from quadratic time and memory complexity in the attention module (MHA) and require caching key-value states during inference, which severely limits throughput and scalability. High inference throughput is critical for agentic tasks, long-context reasoning, efficient deployment under high request loads, and more efficient test-time compute scaling.
State Space Models (SSMs) such as Mamba offer a promising alternative with linear inference complexity and a constant memory footprint via recurrent computation with fixed-size hidden states. In this technical report we introduce the Apriel-H1 family of hybrid LLMs that combine transformer attention and SSM sequence mixers for efficient reasoning at 15B model size. These models are obtained through incremental distillation from a pretrained reasoning transformer, Apriel-Nemotron-15B-Thinker, progressively replacing less critical attention layers with linear Mamba blocks.
We release multiple post-distillation variants of Apriel-H1-15B-Thinker with different SSM-to-MHA ratios and analyse how reasoning performance degrades as more Mamba layers replace MHA. Additionally, we release a 30/50 hybrid variant of Apriel-H1, further fine-tuned on a supervised dataset of reasoning traces, achieving over 2x higher inference throughput when deployed in the production-ready vLLM environment, with minimal degradation in reasoning performance. This shows that distilled hybrid SSM-Transformer architectures can deliver substantial efficiency gains over the pretrained transformer equivalent without substantially compromising the reasoning quality. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过变压器架构和注意力机制实现了卓越的推理能力。然而，变压器在注意力模块（MHA）中面临着二次时间复杂度和内存复杂度的问题，并且在推理过程中需要缓存键值状态，这严重限制了吞吐量和可扩展性。对于代理任务、长上下文推理、高请求负载下的高效部署以及测试时计算规模的更高效扩展，高推理吞吐量是至关重要的。

状态空间模型（SSMs）如Mamba提供了线性推理复杂度和固定内存足迹的替代方案，通过循环计算固定大小的隐藏状态实现。在本技术报告中，我们介绍了Apriel-H1家族的混合LLM，该家族将变压器注意力机制和SSM序列混合器相结合，以在15B模型规模下实现高效的推理。这些模型是通过增量蒸馏从预训练推理变压器Apriel-Nemotron-15B-Thinker获得的，逐步用线性Mamba块替换不太关键的注意力层。

我们发布了多个Apriel-H1-15B-Thinker后蒸馏变体，不同SSM-to-MHA比例，并分析了随着更多Mamba层替换MHA，推理性能如何下降。此外，我们还发布了Apriel-H1的30/50混合变体，进一步在推理轨迹监督数据集上进行了微调，在生产准备好的vLLM环境中部署时推理吞吐量提高了超过2倍，且推理性能降噪较小。这表明，蒸馏混合SSM-变压器架构可以在不显著牺牲推理质量的情况下，比预训练变压器实现显著的效率提升。 

---
# Federated Attention: A Distributed Paradigm for Collaborative LLM Inference over Edge Networks 

**Title (ZH)**: 联邦注意力：边缘网络中协作大语言模型推理的分布式范式 

**Authors**: Xiumei Deng, Zehui Xiong, Binbin Chen, Dong In Kim, Merouane Debbah, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2511.02647)  

**Abstract**: Large language models (LLMs) are proliferating rapidly at the edge, delivering intelligent capabilities across diverse application scenarios. However, their practical deployment in collaborative scenarios confronts fundamental challenges: privacy vulnerabilities, communication overhead, and computational bottlenecks. To address these, we propose Federated Attention (FedAttn), which integrates the federated paradigm into the self-attention mechanism, creating a new distributed LLM inference framework that simultaneously achieves privacy protection, communication efficiency, and computational efficiency. FedAttn enables participants to perform local self-attention over their own token representations while periodically exchanging and aggregating Key-Value (KV) matrices across multiple Transformer blocks, collaboratively generating LLM responses without exposing private prompts. Further, we identify a structural duality between contextual representation refinement in FedAttn and parameter optimization in FL across private data, local computation, and global aggregation. This key insight provides a principled foundation for systematically porting federated optimization techniques to collaborative LLM inference. Building on this framework, we theoretically analyze how local self-attention computation within participants and heterogeneous token relevance among participants shape error propagation dynamics across Transformer blocks. Moreover, we characterize the fundamental trade-off between response quality and communication/computation efficiency, which is governed by the synchronization interval and the number of participants. Experimental results validate our theoretical analysis, and reveal significant optimization opportunities through sparse attention and adaptive KV aggregation, highlighting FedAttn's potential to deliver scalability and efficiency in real-world edge deployments. 

**Abstract (ZH)**: 联邦注意力（FedAttn）：一种结合联邦范式的分布式大规模语言模型推理框架 

---
# On The Dangers of Poisoned LLMs In Security Automation 

**Title (ZH)**: 中毒的大语言模型在安全自动化中的危险 

**Authors**: Patrick Karlsen, Even Eilertsen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02600)  

**Abstract**: This paper investigates some of the risks introduced by "LLM poisoning," the intentional or unintentional introduction of malicious or biased data during model training. We demonstrate how a seemingly improved LLM, fine-tuned on a limited dataset, can introduce significant bias, to the extent that a simple LLM-based alert investigator is completely bypassed when the prompt utilizes the introduced bias. Using fine-tuned Llama3.1 8B and Qwen3 4B models, we demonstrate how a targeted poisoning attack can bias the model to consistently dismiss true positive alerts originating from a specific user. Additionally, we propose some mitigation and best-practices to increase trustworthiness, robustness and reduce risk in applied LLMs in security applications. 

**Abstract (ZH)**: 本文探讨了“LLM中毒”引入的一些风险，即在模型训练过程中有意或无意地引入恶意或有偏见的数据。我们展示了即使经过细调的LLM在利用引入的偏见时，一个简单的基于LLM的警报调查员也可能被完全绕过。使用细调后的Llama3.1 8B和Qwen3 4B模型，我们展示了如何进行有针对性的攻击以使模型一致地忽略来自特定用户的真正阳性警报。此外，我们提出了若干缓解措施和最佳实践，以提高安全应用中实际部署的LLM的可信度、稳健性和降低风险。 

---
# Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour 

**Title (ZH)**: 下一token知识追踪：利用预训练大语言模型表示解码学生行为 

**Authors**: Max Norris, Kobi Gal, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2511.02599)  

**Abstract**: Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively. 

**Abstract (ZH)**: 利用AI进行教育时，建模学生知识是一个关键挑战，对个性化学习具有重大影响。知识 tracing (KT) 任务旨在基于学生之前的互动，预测他们在学习环境中对教育问题的响应方式。现有的 KT 模型通常使用响应正确性以及技能标签和时间戳等元数据，往往忽略了问题文本，这是一项重要的教学洞察来源。这一遗漏限制了预测性能，同时错失了潜在的机会。我们提出了一种新的方法 Next Token Knowledge Tracing (NTKT)，将其重新框架为使用预训练大型语言模型 (LLM) 的下一个标记预测任务。NTKT 将学生历史和问题内容表示为文本序列，使 LLM 能够学习行为和语言中的模式。我们的系列实验在神经 KT 模型中显著提高了性能，并且在处理冷启动问题和用户方面表现出了更好的泛化能力。这些发现突显了问题内容在 KT 中的重要性，并展示了利用预训练 LLM 表征来更有效地建模学生学习的益处。 

---
# BRAINS: A Retrieval-Augmented System for Alzheimer's Detection and Monitoring 

**Title (ZH)**: BRAINS：一种用于阿尔茨海默病检测与监测的检索增强系统 

**Authors**: Rajan Das Gupta, Md Kishor Morol, Nafiz Fahad, Md Tanzib Hosain, Sumaya Binte Zilani Choya, Md Jakir Hossen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02490)  

**Abstract**: As the global burden of Alzheimer's disease (AD) continues to grow, early and accurate detection has become increasingly critical, especially in regions with limited access to advanced diagnostic tools. We propose BRAINS (Biomedical Retrieval-Augmented Intelligence for Neurodegeneration Screening) to address this challenge. This novel system harnesses the powerful reasoning capabilities of Large Language Models (LLMs) for Alzheimer's detection and monitoring. BRAINS features a dual-module architecture: a cognitive diagnostic module and a case-retrieval module. The Diagnostic Module utilizes LLMs fine-tuned on cognitive and neuroimaging datasets -- including MMSE, CDR scores, and brain volume metrics -- to perform structured assessments of Alzheimer's risk. Meanwhile, the Case Retrieval Module encodes patient profiles into latent representations and retrieves similar cases from a curated knowledge base. These auxiliary cases are fused with the input profile via a Case Fusion Layer to enhance contextual understanding. The combined representation is then processed with clinical prompts for inference. Evaluations on real-world datasets demonstrate BRAINS effectiveness in classifying disease severity and identifying early signs of cognitive decline. This system not only shows strong potential as an assistive tool for scalable, explainable, and early-stage Alzheimer's disease detection, but also offers hope for future applications in the field. 

**Abstract (ZH)**: 随着阿尔茨海默病(AD)的全球负担不断增加，早期和准确的检测变得尤为重要，尤其是在先进诊断工具获取受限的地区。我们提出BRAINS（Biomedical Retrieval-Augmented Intelligence for Neurodegeneration Screening）以应对这一挑战。该新型系统利用大规模语言模型（LLMs）的强大推理能力进行阿尔茨海默病的检测与监控。BRAINS采用双模块架构：认知诊断模块和案例检索模块。诊断模块利用细调后的语言模型对认知和神经影像数据进行结构化的阿尔茨海默病风险评估，包括MMSE、CDR评分和脑体积指标。同时，案例检索模块将患者档案编码为潜在表示，并从精心编纂的知识库中检索相似案例。这些辅助案例经案例融合层与输入档案融合，以增强上下文理解。之后，该综合表示通过临床提示进行推理。在实际数据集上的评估证明了BRAINS在疾病严重程度分类和早期认知下降迹象识别方面的有效性。该系统不仅展示了作为大规模、可解释、早期阿尔茨海默病检测辅助工具的强大潜力，还为该领域的未来应用带来了希望。 

---
# Modeling Hawkish-Dovish Latent Beliefs in Multi-Agent Debate-Based LLMs for Monetary Policy Decision Classification 

**Title (ZH)**: 基于多方辩论的LLM货币决策分类中鸽派-鹰派潜在信念建模 

**Authors**: Kaito Takano, Masanori Hirano, Kei Nakagawa  

**Link**: [PDF](https://arxiv.org/pdf/2511.02469)  

**Abstract**: Accurately forecasting central bank policy decisions, particularly those of the Federal Open Market Committee(FOMC) has become increasingly important amid heightened economic uncertainty. While prior studies have used monetary policy texts to predict rate changes, most rely on static classification models that overlook the deliberative nature of policymaking. This study proposes a novel framework that structurally imitates the FOMC's collective decision-making process by modeling multiple large language models(LLMs) as interacting agents. Each agent begins with a distinct initial belief and produces a prediction based on both qualitative policy texts and quantitative macroeconomic indicators. Through iterative rounds, agents revise their predictions by observing the outputs of others, simulating deliberation and consensus formation. To enhance interpretability, we introduce a latent variable representing each agent's underlying belief(e.g., hawkish or dovish), and we theoretically demonstrate how this belief mediates the perception of input information and interaction dynamics. Empirical results show that this debate-based approach significantly outperforms standard LLMs-based baselines in prediction accuracy. Furthermore, the explicit modeling of beliefs provides insights into how individual perspectives and social influence shape collective policy forecasts. 

**Abstract (ZH)**: 准确预测中央银行政策决策，尤其是联邦公开市场委员会（FOMC）的决策，在经济不确定性增强的背景下变得日益重要。尽管以往研究使用货币政策文本来预测利率变化，但大多数研究依赖于静态分类模型，忽视了政策制定的协商过程。本研究提出了一种新型框架，通过将多个大型语言模型（LLMs）模拟为相互作用的代理，结构化地模仿FOMC的集体决策过程。每个代理初始具有不同的信念，并基于定性货币政策文本和定量宏观经济指标生成预测。通过多轮迭代，代理通过观察其他代理的输出来修订预测，模拟协商和共识形成。为了增强可解释性，引入了一个潜在变量来表示每个代理的潜在信念（例如，鹰派或鸽派），并理论上证明了这种信念如何调节输入信息的感知和交互动态。实证结果表明，基于辩论的方法在预测准确性方面显著优于基于标准LLM的基线方法。此外，明确建模信念提供了关于个体视角和社会影响如何塑造集体政策预测的见解。 

---
# EvoDev: An Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents 

**Title (ZH)**: EvoDev: 一种基于LLM代理的迭代特征驱动的端到端软件开发框架 

**Authors**: Junwei Liu, Chen Xu, Chong Wang, Tong Bai, Weitong Chen, Kaseng Wong, Yiling Lou, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.02399)  

**Abstract**: Recent advances in large language model agents offer the promise of automating end-to-end software development from natural language requirements. However, existing approaches largely adopt linear, waterfall-style pipelines, which oversimplify the iterative nature of real-world development and struggle with complex, large-scale projects. To address these limitations, we propose EvoDev, an iterative software development framework inspired by feature-driven development. EvoDev decomposes user requirements into a set of user-valued features and constructs a Feature Map, a directed acyclic graph that explicitly models dependencies between features. Each node in the feature map maintains multi-level information, including business logic, design, and code, which is propagated along dependencies to provide context for subsequent development iterations. We evaluate EvoDev on challenging Android development tasks and show that it outperforms the best-performing baseline, Claude Code, by a substantial margin of 56.8%, while improving single-agent performance by 16.0%-76.6% across different base LLMs, highlighting the importance of dependency modeling, context propagation, and workflow-aware agent design for complex software projects. Our work summarizes practical insights for designing iterative, LLM-driven development frameworks and informs future training of base LLMs to better support iterative software development. 

**Abstract (ZH)**: 近期大型语言模型代理的进展为从自然语言需求自动化整个软件开发过程提供了前景。然而，现有方法主要采用线性的瀑布式流水线，这简化了真实世界开发过程中的迭代性质，难以处理复杂的大型项目。为了解决这些限制，我们提出EvoDev，这是一种受特征驱动开发启发的迭代软件开发框架。EvoDev将用户需求分解为一组用户看重的特征，并构建特征图，这是一个有向无环图，明确地建模了特征之间的依赖关系。特征图中的每个节点保持多层次的信息，包括业务逻辑、设计和代码，这些信息沿着依赖关系传播，为后续的开发迭代提供上下文。我们在具有挑战性的Android开发任务上评估了EvoDev，并展示了它在广泛的基本LLM中相比最佳基线Claude Code取得了56.8%的显著优势，同时单个代理性能提高了16.0%-76.6%，突显了依赖建模、上下文传播和工作流程感知代理设计对于复杂软件项目的重要性。我们的工作总结了设计迭代的LLM驱动开发框架的实际见解，并为未来训练基本LLM以更好地支持迭代软件开发提供了指导。 

---
# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models 

**Title (ZH)**: AutoAdv：自动 adversarial prompting 多轮劫持大规模语言模型 

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban  

**Link**: [PDF](https://arxiv.org/pdf/2511.02376)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses. 

**Abstract (ZH)**: 基于自动多轮囚徒破解的攻击框架：AutoAdv将在六轮对话中对Llama-3.1-8B的攻击成功率提高到95%，比单轮基准提高了24个百分点。 

---
# AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda 

**Title (ZH)**: AyurParam：一种基于最新技术的双语语言模型用于Ayurveda 

**Authors**: Mohd Nauman, Sravan Gvm, Vijay Devane, Shyam Pawar, Viraj Thakur, Kundeshwar Pundalik, Piyush Sawarkar, Rohit Saluja, Maunendra Desarkar, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02374)  

**Abstract**: Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but also demonstrates competitive or superior performance compared to much larger models. The results from AyurParam highlight the necessity for authentic domain adaptation and high-quality supervision in delivering reliable, culturally congruent AI for specialized medical knowledge. 

**Abstract (ZH)**: 当前的大规模语言模型在广泛的通用任务上表现出色，但在面对需要深厚文化、语言和专业领域知识的高度专业化领域时却表现不佳。特别是传统的医学体系，如阿育吠陀，蕴含了数个世纪的细腻文本和临床知识，主流的大规模语言模型无法准确解读或应用。我们引入了AyurParam-2.9B，这是一个基于Param-1-2.9B进行领域专业化调整的双语语言模型，使用了涵盖古典文本和临床指导的 extensive、专家整理的阿育吠陀数据集。AyurParam的数据集包含了上下文感知的推理和客观风格的问题-答案对，其中英、印地语均有涉及，并且标注协议严格，确保事实精确性和指令清晰度。在BhashaBench-Ayur基准测试中，AyurParam不仅在模型规模（1.5-3B参数）类别中超越了所有开源指令调整模型，而且还展示了与更大规模模型相当甚至更优的性能。来自AyurParam的结果强调了在提供可靠且文化相符的AI专医学知识时进行真实领域适应和高质量监督的重要性。 

---
# The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute 

**Title (ZH)**: 顺序边缘：逆熵投票在匹配计算资源下优于并行自我一致性。 

**Authors**: Aman Sharma, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2511.02309)  

**Abstract**: We revisit test-time scaling for language model reasoning and ask a fundamental question: at equal token budget and compute, is it better to run multiple independent chains in parallel, or to run fewer chains that iteratively refine through sequential steps? Through comprehensive evaluation across 5 state-of-the-art open source models and 3 challenging reasoning benchmarks, we find that sequential scaling where chains explicitly build upon previous attempts consistently outperforms the dominant parallel self-consistency paradigm in 95.6% of configurations with gains in accuracy upto 46.7%. Further, we introduce inverse-entropy weighted voting, a novel training-free method to further boost the accuracy of sequential scaling. By weighing answers in proportion to the inverse entropy of their reasoning chains, we increase our success rate over parallel majority and establish it as the optimal test-time scaling strategy. Our findings fundamentally challenge the parallel reasoning orthodoxy that has dominated test-time scaling since Wang et al.'s self-consistency decoding (Wang et al., 2022), positioning sequential refinement as the robust default for modern LLM reasoning and necessitating a paradigm shift in how we approach inference-time optimization. 

**Abstract (ZH)**: 我们重新审视语言模型推理的测试时缩放方法，并提出一个基本问题：在相同的时间单元预算和计算资源下，是并行运行多个独立链更优，还是通过序列化步骤迭代改进的较少链更优？通过在5个领先的开源模型和3个具有挑战性的推理基准上的综合评估，我们发现，在95.6%的配置中，显式利用先前尝试进行序列化缩放的一致性优于主导的并行自我一致性范式，并且在准确率上提高了多达46.7%。此外，我们引入了逆熵加权投票，这是一种无需训练的方法，进一步提升了序列化缩放的准确性。通过对推理链的答案按逆熵加权赋值，我们在并行多数票的基础上提高了成功率，并将其确立为最优的测试时缩放策略。我们的发现从根本上挑战了自Wang等人提出自我一致性解码以来主宰测试时缩放的并行推理正统，并将序列化改进定位为现代大规模语言模型推理的稳健默认策略，有必要转变我们在推理时优化方面的范式。 

---
# LA-MARRVEL: A Knowledge-Grounded and Language-Aware LLM Reranker for AI-MARRVEL in Rare Disease Diagnosis 

**Title (ZH)**: LA-MARRVEL：一种基于知识和语言意识的LLM重排序模型，用于AI-MARRVEL在罕见病诊断中的应用 

**Authors**: Jaeyeon Lee, Hyun-Hwan Jeong, Zhandong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02263)  

**Abstract**: Diagnosing rare diseases often requires connecting variant-bearing genes to evidence that is written as unstructured clinical prose, which the current established pipelines still leave for clinicians to reconcile manually. To this end, we introduce LA-MARRVEL, a knowledge-grounded and language-aware reranking layer that operates on top of AI-MARRVEL: it supplies expert-engineered context, queries a large language model multiple times, and aggregates the resulting partial rankings with a ranked voting method to produce a stable, explainable gene ranking. Evaluated on three real-world cohorts (BG, DDD, UDN), LA-MARRVEL consistently improves Recall@K over AI-MARRVEL and established phenotype-driven tools such as Exomiser and LIRICAL, with especially large gains on cases where the first-stage ranker placed the causal gene lower. Each ranked gene is accompanied by LLM-generated reasoning that integrates phenotypic, inheritance, and variant-level evidence, thereby making the output more interpretable and facilitating clinical review. 

**Abstract (ZH)**: 基于知识和语言的重排层LA-MARRVEL在罕见疾病诊断中的应用 

---
# Demo: Statistically Significant Results On Biases and Errors of LLMs Do Not Guarantee Generalizable Results 

**Title (ZH)**: 示例：统计显著结果并不保证LLMs偏差和错误的普遍化结果 

**Authors**: Jonathan Liu, Haoling Qiu, Jonathan Lasko, Damianos Karakos, Mahsa Yarmohammadi, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2511.02246)  

**Abstract**: Recent research has shown that hallucinations, omissions, and biases are prevalent in everyday use-cases of LLMs. However, chatbots used in medical contexts must provide consistent advice in situations where non-medical factors are involved, such as when demographic information is present. In order to understand the conditions under which medical chatbots fail to perform as expected, we develop an infrastructure that 1) automatically generates queries to probe LLMs and 2) evaluates answers to these queries using multiple LLM-as-a-judge setups and prompts. For 1), our prompt creation pipeline samples the space of patient demographics, histories, disorders, and writing styles to create realistic questions that we subsequently use to prompt LLMs. In 2), our evaluation pipeline provides hallucination and omission detection using LLM-as-a-judge as well as agentic workflows, in addition to LLM-as-a-judge treatment category detectors. As a baseline study, we perform two case studies on inter-LLM agreement and the impact of varying the answering and evaluation LLMs. We find that LLM annotators exhibit low agreement scores (average Cohen's Kappa $\kappa=0.118$), and only specific (answering, evaluation) LLM pairs yield statistically significant differences across writing styles, genders, and races. We recommend that studies using LLM evaluation use multiple LLMs as evaluators in order to avoid arriving at statistically significant but non-generalizable results, particularly in the absence of ground-truth data. We also suggest publishing inter-LLM agreement metrics for transparency. Our code and dataset are available here: this https URL. 

**Abstract (ZH)**: 近期研究显示，LLM在日常使用场景中普遍存在幻觉、遗漏和偏差。然而，在医疗情境下的聊天机器人必须在涉及非医疗因素的情况下，如个人 demographics 信息存在时，提供一致的建议。为了理解医疗聊天机器人何时不能按预期执行，我们开发了一个基础设施，该基础设施包括1）自动生成查询以探查LLM，以及2）利用多个LLM作为评判者设置和提示进行答案评估。在1）中，我们的提示生成管道抽样患者 demographics、病史、疾病和写作风格的空间，以创建现实的问题，随后使用这些问题来提示LLM。在2）中，我们的评估管道利用LLM作为评判者进行幻觉和遗漏检出，并使用代理工作流程和LLM作为评判者的行为类别检测器进行评估。作为基准研究，我们在不同LLM之间进行两种案例研究，探讨了LLM之间的一致性以及回答和评估LLM的选择的影响。我们发现，LLM注释员的一致性评分为较低（平均科恩κ系数=0.118），仅特定（回答，评估）LLM配对在不同写作风格、性别和种族方面显示出统计学上的显著差异。我们建议在使用LLM评估的研究中使用多种LLM作为评判者，以避免得出统计学上有显著性但不具普适性的结果，特别是在缺乏真实数据的情况下。我们还建议公布LLM之间的一致性指标以提高透明度。我们的代码和数据集可在此获取：this https URL。 

---
# Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live 

**Title (ZH)**: 连续体：具有KV缓存时间生存期的高效稳健多轮LLM代理调度 

**Authors**: Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Alvin Cheung, Joseph Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2511.02230)  

**Abstract**: Agentic LLM applications interleave LLM generation requests with tool calls. These tool calls break the continuity of the workflow by creating pauses between LLM requests, bringing many challenges for the serving system, especially under multi-turn scenarios. Each pause potentially causes KV cache eviction and extra waiting time before entering the continuous batch for the following LLM request. Since these pauses happen for each call, this problem becomes increasingly severe as turn number grow for agentic programs. Previous works either fail to incorporate information from the tool call, evicting KV cache that leads to repetitive prefill or loading, or ignore the continuity of a multi-turn program, creating waiting time between turns that increases per-request latency.
We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by combining tool-aware KV cache timeout with program-level scheduling. By predicting tool call durations in agentic workflows, Continuum selectively pins the KV cache in GPU memory with a time-to-live value based on total turn number. When combined with program-level first-come-first-serve, Continuum prevents scheduling bubbles, preserves multi-turn continuity, and optimizes for throughput for complex agentic workflows. By modeling the variability of tool call and agent program continuity, Continuum outperforms state-of-the-art baselines. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B models shows that Continuum significantly improves the average job completion times, and remains performant across different hardware setups and DRAM offloading schemes. Preview code is available at: this https URL 

**Abstract (ZH)**: Continuum：一种通过工具意识型KV缓存超时与程序级调度结合来优化多轮代理工作负载完成时间的服务体系结构 

---
# Open the Oyster: Empirical Evaluation and Improvement of Code Reasoning Confidence in LLMs 

**Title (ZH)**: 开启牡蛎：LLM的代码推理置信度的实证评估与改进 

**Authors**: Shufan Wang, Xing Hu, Junkai Chen, Zhiyuan Pan, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2511.02197)  

**Abstract**: With the widespread application of large language models (LLMs) in the field of code intelligence, increasing attention has been paid to the reliability and controllability of their outputs in code reasoning tasks. Confidence estimation serves as an effective and convenient approach for evaluating these aspects. This paper proposes a confidence analysis and enhancement framework for LLMs tailored to code reasoning tasks. We conduct a comprehensive empirical study on the confidence reliability of mainstream LLMs across different tasks, and further evaluate the effectiveness of techniques such as prompt strategy optimisation and mathematical calibration (e.g., Platt Scaling) in improving confidence reliability. Our results show that DeepSeek-Reasoner achieves the best performance across various tasks, outperforming other models by up to $0.680$, $0.636$, and $13.652$ in terms of ECE, Brier Score, and Performance Score, respectively. The hybrid strategy combining the reassess prompt strategy and Platt Scaling achieves improvements of up to $0.541$, $0.628$, and $15.084$ over the original performance in the aforementioned three metrics. These results indicate that models with reasoning capabilities demonstrate superior confidence reliability, and that the hybrid strategy is the most effective in enhancing the confidence reliability of various models. Meanwhile, we elucidate the impact of different task complexities, model scales, and strategies on confidence performance, and highlight that the confidence of current LLMs in complex reasoning tasks still has considerable room for improvement. This study not only provides a research foundation and technical reference for the application of confidence in LLM-assisted software engineering, but also points the way for future optimisation and engineering deployment of confidence mechanisms. 

**Abstract (ZH)**: 大型语言模型在代码智能领域的广泛应用引起了人们对代码推理任务中其输出可靠性和可控性的高度重视。置信度评估作为一项有效且便捷的方法被广泛采用。本文提出了一种针对代码推理任务的大语言模型置信度分析与增强框架。我们对主流大语言模型在不同任务中的置信度可靠性进行了全面的经验研究，并进一步评估了诸如提示策略优化和数学校准（如Platt Scaling）等技术在提升置信度可靠性方面的有效性。结果显示，DeepSeek-Reasoner在各项任务中均表现出最佳性能，分别在ECE、Brier Score和Performance Score方面优于其他模型0.680、0.636和13.652。结合重新评估提示策略和Platt Scaling的混合策略在上述三项指标中分别实现了0.541、0.628和15.084的性能提升。这些结果表明，具备推理能力的模型在置信度可靠性方面表现出更优性能，而混合策略在多种模型上是提升置信度可靠性最有效的方法。同时，我们探讨了不同任务复杂性、模型规模和策略对置信度性能的影响，并指出当前大语言模型在复杂推理任务中的置信度仍有很大的提升空间。本文不仅为大语言模型辅助软件工程中置信度的应用提供了研究基础和技术参考，还为未来置信度机制的优化和工程部署指明了方向。 

---
# Metamorphic Testing of Large Language Models for Natural Language Processing 

**Title (ZH)**: 大型语言模型的 metamorphic 测试在自然语言处理中的应用 

**Authors**: Steven Cho, Stefano Ruberto, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2511.02108)  

**Abstract**: Using large language models (LLMs) to perform natural language processing (NLP) tasks has become increasingly pervasive in recent times. The versatile nature of LLMs makes them applicable to a wide range of such tasks. While the performance of recent LLMs is generally outstanding, several studies have shown that they can often produce incorrect results. Automatically identifying these faulty behaviors is extremely useful for improving the effectiveness of LLMs. One obstacle to this is the limited availability of labeled datasets, which necessitates an oracle to determine the correctness of LLM behaviors. Metamorphic testing (MT) is a popular testing approach that alleviates this oracle problem. At the core of MT are metamorphic relations (MRs), which define relationships between the outputs of related inputs. MT can expose faulty behaviors without the need for explicit oracles (e.g., labeled datasets). This paper presents the most comprehensive study of MT for LLMs to date. We conducted a literature review and collected 191 MRs for NLP tasks. We implemented a representative subset (36 MRs) to conduct a series of experiments with three popular LLMs, running approximately 560,000 metamorphic tests. The results shed light on the capabilities and opportunities of MT for LLMs, as well as its limitations. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）执行自然语言处理（NLP）任务在当今越来越普遍。LLMs的多功能性使它们适用于广泛的任务。尽管近年来LLMs的性能通常非常出色，但多项研究显示它们经常会产生错误结果。自动识别这些故障行为对于提高LLMs的有效性极为有用。这一挑战的一个障碍是标记数据集的有限可用性，这需要一个知识渊博的实体来确定LLM行为的正确性。元型测试（MT）是一种流行的方法，可以缓解这一问题。MT的核心是元型关系（MRs），它们定义了相关输入输出之间的关系。MT可以在不需要显式或acles（例如，标记数据集）的情况下揭示故障行为。本文进行了迄今为止对于LLMs最全面的MT研究。我们进行了文献综述，并收集了191个适用于NLP任务的MRs。我们实现了一个代表性子集（36个MRs）以对三种流行LLMs进行一系列实验，共运行了约560,000个元型测试。结果揭示了MT在LLMs中的能力和潜力，以及其局限性。 

---
# Watermarking Discrete Diffusion Language Models 

**Title (ZH)**: 离散扩散语言模型中的水印技术 

**Authors**: Avi Bagchi, Akhil Bhimaraju, Moulik Choraria, Daniel Alabi, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2511.02083)  

**Abstract**: Watermarking has emerged as a promising technique to track AI-generated content and differentiate it from authentic human creations. While prior work extensively studies watermarking for autoregressive large language models (LLMs) and image diffusion models, none address discrete diffusion language models, which are becoming popular due to their high inference throughput. In this paper, we introduce the first watermarking method for discrete diffusion models by applying the distribution-preserving Gumbel-max trick at every diffusion step and seeding the randomness with the sequence index to enable reliable detection. We experimentally demonstrate that our scheme is reliably detectable on state-of-the-art diffusion language models and analytically prove that it is distortion-free with an exponentially decaying probability of false detection in the token sequence length. 

**Abstract (ZH)**: 水印技术作为一种追踪AI生成内容并与真实人类创作区分的有前途的方法已经 emergence 。尽管先前的工作对自回归大型语言模型和图像扩散模型的水印技术进行了广泛研究，但尚未有工作针对成为流行趋势的高推理吞吐量离散扩散语言模型。在本文中，我们通过在每次扩散步骤中应用分布保持的Gumbel-max技巧并将随机性与序列索引结合，首次提出了离散扩散模型的水印方法，以实现可靠的检测。我们实验性地证明了该方案在最先进的扩散语言模型上具有可靠的可检测性，并从理论上证明在令牌序列长度上，该方案具有指数衰减的误检概率且无失真。 

---
# Regularization Through Reasoning: Systematic Improvements in Language Model Classification via Explanation-Enhanced Fine-Tuning 

**Title (ZH)**: 通过推理正则化：通过解释增强微调在语言模型分类中的系统性改进 

**Authors**: Vivswan Shah, Randy Cogill, Hanwei Yue, Gopinath Chennupati, Rinat Khaziev  

**Link**: [PDF](https://arxiv.org/pdf/2511.02044)  

**Abstract**: Fine-tuning LLMs for classification typically maps inputs directly to labels. We ask whether attaching brief explanations to each label during fine-tuning yields better models. We evaluate conversational response quality along three axes: naturalness, comprehensiveness, and on-topic adherence, each rated on 5-point scales. Using ensemble-generated data from multiple LLMs, we fine-tune a 7B-parameter model and test across six diverse conversational datasets. Across 18 dataset, task settings, label-plus-explanation training outperforms label-only baselines.
A central and unexpected result concerns random tokens. We replace human-written explanations with text that is syntactically incoherent yet vocabulary-aligned with the originals (e.g., shuffled or bag-of-words variants). Despite lacking semantics, these pseudo-explanations still improve accuracy over label-only training and often narrow much of the gap to true explanations. The effect persists across datasets and training seeds, indicating that gains arise less from meaning than from structure: the extra token budget encourages richer intermediate computation and acts as a regularizer that reduces over-confident shortcuts.
Internal analyses support this view: explanation-augmented models exhibit higher activation entropy in intermediate layers alongside sharper predictive mass at the output layer, consistent with increased deliberation before decision. Overall, explanation-augmented fine-tuning, whether with genuine rationales or carefully constructed random token sequences, improves accuracy and reliability for LLM classification while clarifying how token-level scaffolding shapes computation during inference. 

**Abstract (ZH)**: 细调LLMs进行分类通常将输入直接映射到标签。我们询问在细调过程中为每个标签附上简短解释是否能获得更好的模型。我们从对话响应质量的三个维度进行评估：自然度、完备性和主题相关性，每个维度按5点量表评分。使用多个LLM生成的集成数据，我们细调了一个7亿参数的模型，并在六个多样化的对话数据集中进行测试。在18个数据集和任务设置中，带有解释的标签训练优于仅标签的基础模型。

一个中心且意外的结果涉及随机标记。我们用与原有人编写解释在词汇上对齐但语义上不连贯的文本（例如，洗牌或词袋变体）替换人工撰写的解释。尽管缺乏语义，这些伪解释仍然在仅标签训练的基础上提高了精度，并且常常缩小了与真实解释之间差距的大部分。这种效应在不同数据集和训练种子中持续存在，表明收益主要来自结构而非意义：额外的标记预算促进了更丰富的中间计算，并作为一种正则化手段减少了过于自信的捷径。

内部分析支持这一观点：增强解释的模型在中间层表现出更高的激活熵，并且在输出层具有更锐利的预测质量，这与在决策前增加的斟酌一致。总体而言，无论是真实理由还是精心构建的随机标记序列的增强解释，都可以改善LLM分类的准确性和可靠性，并阐明标记级别架构如何在推理过程中塑造计算。 

---
# Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior 

**Title (ZH)**: Emergently Misaligned行为中的共享参数子空间与跨任务线性关系 

**Authors**: Daniel Aarao Reis Arturi, Eric Zhang, Andrew Ansah, Kevin Zhu, Ashwinee Panda, Aishwarya Balwani  

**Link**: [PDF](https://arxiv.org/pdf/2511.02022)  

**Abstract**: Recent work has discovered that large language models can develop broadly misaligned behaviors after being fine-tuned on narrowly harmful datasets, a phenomenon known as emergent misalignment (EM). However, the fundamental mechanisms enabling such harmful generalization across disparate domains remain poorly understood. In this work, we adopt a geometric perspective to study EM and demonstrate that it exhibits a fundamental cross-task linear structure in how harmful behavior is encoded across different datasets. Specifically, we find a strong convergence in EM parameters across tasks, with the fine-tuned weight updates showing relatively high cosine similarities, as well as shared lower-dimensional subspaces as measured by their principal angles and projection overlaps. Furthermore, we also show functional equivalence via linear mode connectivity, wherein interpolated models across narrow misalignment tasks maintain coherent, broadly misaligned behavior. Our results indicate that EM arises from different narrow tasks discovering the same set of shared parameter directions, suggesting that harmful behaviors may be organized into specific, predictable regions of the weight landscape. By revealing this fundamental connection between parametric geometry and behavioral outcomes, we hope our work catalyzes further research on parameter space interpretability and weight-based interventions. 

**Abstract (ZH)**: 近年来的研究发现，大型语言模型在狭义有害数据集上微调后，可能会表现出广泛 misaligned 的行为，这一现象被称为 emergent misalignment (EM)。然而，促使此类有害泛化的根本机制在不同领域之间仍不清楚。在本项工作中，我们采用几何视角研究 EM，并证明它在不同数据集上如何编码有害行为方面表现出一种基本的跨任务线性结构。具体来说，我们发现不同任务的 EM 参数存在强烈的收敛性，微调权重更新显示出较高的余弦相似度，并且存在共享的低维子空间。此外，我们还通过线性模式连通性展示了功能等价性，在狭窄 misalignment 任务之间的插值模型保持一致且广泛 misaligned 的行为。我们的结果表明，EM 是不同狭义任务找到相同参数方向集的结果，暗示有害行为可能被组织在权重景观的特定、可预测区域中。通过揭示参数几何学与行为结果之间的基本联系，我们希望本项工作能促进参数空间可解释性和基于权重的干预措施的研究。 

---
# InteracSPARQL: An Interactive System for SPARQL Query Refinement Using Natural Language Explanations 

**Title (ZH)**: InteracSPARQL：基于自然语言解释的SPARQL查询改进交互系统 

**Authors**: Xiangru Jian, Zhengyuan Dong, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02002)  

**Abstract**: In recent years, querying semantic web data using SPARQL has remained challenging, especially for non-expert users, due to the language's complex syntax and the prerequisite of understanding intricate data structures. To address these challenges, we propose InteracSPARQL, an interactive SPARQL query generation and refinement system that leverages natural language explanations (NLEs) to enhance user comprehension and facilitate iterative query refinement. InteracSPARQL integrates LLMs with a rule-based approach to first produce structured explanations directly from SPARQL abstract syntax trees (ASTs), followed by LLM-based linguistic refinements. Users can interactively refine queries through direct feedback or LLM-driven self-refinement, enabling the correction of ambiguous or incorrect query components in real time. We evaluate InteracSPARQL on standard benchmarks, demonstrating significant improvements in query accuracy, explanation clarity, and overall user satisfaction compared to baseline approaches. Our experiments further highlight the effectiveness of combining rule-based methods with LLM-driven refinements to create more accessible and robust SPARQL interfaces. 

**Abstract (ZH)**: 近年来，使用SPARQL查询语义网数据仍具有挑战性，尤其是对于非专家用户而言，由于该语言复杂的语法结构和对复杂数据结构的理解要求。为解决这些挑战，我们提出了一种名为InteracSPARQL的交互式SPARQL查询生成和细化系统，该系统利用自然语言解释（NLE）来增强用户理解并促进迭代查询细化。InteracSPARQL结合了基于规则的方法和大语言模型（LLMs），首先从SPARQL抽象语法树（ASTs）直接生成结构化的解释，然后进行基于LLM的语言细化。用户可以通过直接反馈或LLM驱动的自我细化来交互式地细化查询，从而实时纠正模糊或错误的查询成分。我们在标准基准上评估了InteracSPARQL，与基础方法相比，结果显示在查询准确性、解释清晰度和整体用户满意度方面取得了显著改进。我们的实验进一步强调了结合基于规则的方法和LLM驱动的细化在创建更易于访问和稳健的SPARQL界面方面的有效性。 

---
# Vibe Learning: Education in the age of AI 

**Title (ZH)**: AI时代的学习：教育的变革 

**Authors**: Marcos Florencio, Francielle Prieto  

**Link**: [PDF](https://arxiv.org/pdf/2511.01956)  

**Abstract**: The debate over whether "thinking machines" could replace human intellectual labor has existed in both public and expert discussions since the mid-twentieth century, when the concept and terminology of Artificial Intelligence (AI) first emerged. For decades, this idea remained largely theoretical. However, with the recent advent of Generative AI - particularly Large Language Models (LLMs) - and the widespread adoption of tools such as ChatGPT, the issue has become a practical reality. Many fields that rely on human intellectual effort are now being reshaped by AI tools that both expand human capabilities and challenge the necessity of certain forms of work once deemed uniquely human but now easily automated. Education, somewhat unexpectedly, faces a pivotal responsibility: to devise long-term strategies for cultivating human skills that will remain relevant in an era of pervasive AI in the intellectual domain. In this context, we identify the limitations of current AI systems - especially those rooted in LLM technology - argue that the fundamental causes of these weaknesses cannot be resolved through existing methods, and propose directions within the constructivist paradigm for transforming education to preserve the long-term advantages of human intelligence over AI tools. 

**Abstract (ZH)**: 自20世纪中叶人工智能（AI）概念和术语首次出现以来，关于“思考机器”是否能取代人类智力劳动的辩论一直存在于公众和专家的讨论中。几十年来，这一想法主要停留在理论层面。然而，随着生成式AI（特别是大型语言模型LLMs）的兴起以及ChatGPT等工具的广泛采用，这一问题已成为现实。许多依赖人类智力劳动的领域现在正被既能扩展人类能力又能挑战某些形式工作的AI工具重塑。教育领域出乎意料地面临一个关键责任：制定长期策略以培养将在人工智能普及的时代仍然具有相关性的个人能力。在此背景下，我们识别当前AI系统的局限性，尤其是在大型语言模型技术领域，认为解决这些弱点的根本原因无法通过现有方法实现，并在建构主义 paradigm中提出转型教育的方向，以保持人类智能相对于AI工具的长期优势。 

---
# Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing 

**Title (ZH)**: 基于先验知识校准记忆探针的黑盒会员推理攻击针对低资源语言模型 

**Authors**: Jinhua Yin, Peiru Yang, Chen Yang, Huili Wang, Zhiyang Hu, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01952)  

**Abstract**: Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpora of visual and textual data. Empowered by large-scale parameters, these models often exhibit strong memorization of their training data, rendering them susceptible to membership inference attacks (MIAs). Existing MIA methods for LVLMs typically operate under white- or gray-box assumptions, by extracting likelihood-based features for the suspected data samples based on the target LVLMs. However, mainstream LVLMs generally only expose generated outputs while concealing internal computational features during inference, limiting the applicability of these methods. In this work, we propose the first black-box MIA framework for LVLMs, based on a prior knowledge-calibrated memory probing mechanism. The core idea is to assess the model memorization of the private semantic information embedded within the suspected image data, which is unlikely to be inferred from general world knowledge alone. We conducted extensive experiments across four LVLMs and three datasets. Empirical results demonstrate that our method effectively identifies training data of LVLMs in a purely black-box setting and even achieves performance comparable to gray-box and white-box methods. Further analysis reveals the robustness of our method against potential adversarial manipulations, and the effectiveness of the methodology designs. Our code and data are available at this https URL. 

**Abstract (ZH)**: 基于先验知识校准的大型视觉-语言模型黑箱会员推理框架 

---
# Shorter but not Worse: Frugal Reasoning via Easy Samples as Length Regularizers in Math RLVR 

**Title (ZH)**: 短些但不逊色：通过易样本作为长度正则化器的节约推理 

**Authors**: Abdelaziz Bounhar, Hadi Abdine, Evan Dufraisse, Ahmad Chamma, Amr Mohamed, Dani Bouch, Michalis Vazirgiannis, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01937)  

**Abstract**: Large language models (LLMs) trained for step-by-step reasoning often become excessively verbose, raising inference cost. Standard Reinforcement Learning with Verifiable Rewards (RLVR) pipelines filter out ``easy'' problems for training efficiency, leaving the model to train primarily on harder problems that require longer reasoning chains. This skews the output length distribution upward, resulting in a \textbf{model that conflates ``thinking longer'' with ``thinking better''}. In this work, we show that retaining and modestly up-weighting moderately easy problems acts as an implicit length regularizer. Exposing the model to solvable short-chain tasks constrains its output distribution and prevents runaway verbosity. The result is \textbf{\emph{emergent brevity for free}}: the model learns to solve harder problems without inflating the output length, \textbf{ despite the absence of any explicit length penalization}. RLVR experiments using this approach on \textit{Qwen3-4B-Thinking-2507} (with a 16k token limit) achieve baseline pass@1 AIME25 accuracy while generating solutions that are, on average, nearly twice as short. The code is available at \href{this https URL}{GitHub}, with datasets and models on \href{this https URL}{Hugging Face}. 

**Abstract (ZH)**: 大型语言模型（LLMs）训练用于逐步推理时往往会变得过度冗长，增加推理成本。标准可验证奖励强化学习（RLVR）管道过滤掉“简单”的问题以提高训练效率，使模型主要在需要较长推理链的较难问题上进行训练。这使得输出长度分布趋高，导致模型将“思考更长”与“思考更好”混为一谈。在本文中，我们展示了保留和适度增加中等难度问题作为隐式的长度正则化手段。使模型接触到可解决的短推理链任务可以约束其输出分布，防止过度冗长。结果是\emph{免费涌现的简明}: 模型能够在不增加输出长度的情况下学习解决更难的问题，\emph{即使没有明确的长度惩罚}。使用此方法在\textit{Qwen3-4B-Thinking-2507}（16k词令牌限制）上进行的RLVR实验实现了基线的AIME25准确率，同时生成的解决方案平均短近一倍。代码可在GitHub（\href{this https URL}{此链接}）上获取，数据集和模型可在Hugging Face（\href{this https URL}{此链接}）上获取。 

---
# Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch 

**Title (ZH)**: Tool Zero: 从头基于纯强化学习训练工具增强的大语言模型 

**Authors**: Yirong Zeng, Xiao Ding, Yutai Hou, Yuxian Wang, Li Du, Juyi Dai, Qiuyang Ding, Duyu Tang, Dandan Tu, Weiwen Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01934)  

**Abstract**: Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods. 

**Abstract (ZH)**: 训练工具增强的大语言模型已成为提升语言模型处理复杂任务能力的有前途的方法。当前的监督微调范式依赖于构建广泛的专业领域数据集来训练模型。然而，这种方法往往难以有效地将模型推广到不熟悉或复杂的工具使用场景中。最近的强化学习（RL）范式能够赋予大语言模型更强的推理和泛化能力。在本工作中，我们解决了一个关键问题：纯粹的强化学习能否有效地激发模型的内在推理能力，并增强工具无关的泛化能力？我们提出了一种动态泛化引导的奖励设计，该设计基于规则的RL，逐步从探索性转向利用性工具使用模式。基于此设计，我们引入了Tool-Zero系列模型。这些模型通过直接从零模型（即未经后训练的基础模型）扩展RL来训练，使大语言模型能够自主利用通用工具。实验结果表明，在相同的实验设置下，我们模型的性能相比SFT和带有SFT的RL模型提高了超过7%。这些收益在跨数据集和同数据集评估中一致得到验证，验证了我们方法的有效性和稳健性。 

---
# EvoMem: Improving Multi-Agent Planning with Dual-Evolving Memory 

**Title (ZH)**: EvoMem: 通过双演化记忆提高多agent规划能力 

**Authors**: Wenzhe Fan, Ning Yan, Masood Mortazavi  

**Link**: [PDF](https://arxiv.org/pdf/2511.01912)  

**Abstract**: Planning has been a cornerstone of artificial intelligence for solving complex problems, and recent progress in LLM-based multi-agent frameworks have begun to extend this capability. However, the role of human-like memory within these frameworks remains largely unexplored. Understanding how agents coordinate through memory is critical for natural language planning, where iterative reasoning, constraint tracking, and error correction drive the success. Inspired by working memory model in cognitive psychology, we present EvoMem, a multi-agent framework built on a dual-evolving memory mechanism. The framework consists of three agents (Constraint Extractor, Verifier, and Actor) and two memory modules: Constraint Memory (CMem), which evolves across queries by storing task-specific rules and constraints while remains fixed within a query, and Query-feedback Memory (QMem), which evolves within a query by accumulating feedback across iterations for solution refinement. Both memory modules are reset at the end of each query session. Evaluations on trip planning, meeting planning, and calendar scheduling show consistent performance improvements, highlighting the effectiveness of EvoMem. This success underscores the importance of memory in enhancing multi-agent planning. 

**Abstract (ZH)**: 基于演化记忆的多智能体规划框架 

---
# Between Myths and Metaphors: Rethinking LLMs for SRH in Conservative Contexts 

**Title (ZH)**: 在神话与隐喻之间：重新思考保守背景下的人工智能语言模型在性与生殖健康领域的应用 

**Authors**: Ameemah Humayun, Bushra Zubair, Maryam Mustafa  

**Link**: [PDF](https://arxiv.org/pdf/2511.01907)  

**Abstract**: Low-resource countries represent over 90% of maternal deaths, with Pakistan among the top four countries contributing nearly half in 2023. Since these deaths are mostly preventable, large language models (LLMs) can help address this crisis by automating health communication and risk assessment. However, sexual and reproductive health (SRH) communication in conservative contexts often relies on indirect language that obscures meaning, complicating LLM-based interventions. We conduct a two-stage study in Pakistan: (1) analyzing data from clinical observations, interviews, and focus groups with clinicians and patients, and (2) evaluating the interpretive capabilities of five popular LLMs on this data. Our analysis identifies two axes of communication (referential domain and expression approach) and shows LLMs struggle with semantic drift, myths, and polysemy in clinical interactions. We contribute: (1) empirical themes in SRH communication, (2) a categorization framework for indirect communication, (3) evaluation of LLM performance, and (4) design recommendations for culturally-situated SRH communication. 

**Abstract (ZH)**: 低资源国家代表了超过90%的 maternal死亡，其中巴基斯坦在2023年贡献了近半数。由于这些死亡主要是可以预防的，大规模语言模型（LLMs）可以通过自动化健康沟通和风险评估来帮助应对这一危机。然而，在保守的背景下，性与生殖健康（SRH）沟通往往依赖于间接语言，这使得基于LLM的干预措施复杂化。我们在巴基斯坦进行了两阶段研究：（1）分析临床观察、访谈和 clinicians及患者焦点小组的数据，（2）评估五种流行LLM在这方面的解释能力。我们的分析确定了沟通的两个维度（指称领域和表达方式），并表明LLMs在临床互动中面临语义转移、神话和多义性的挑战。我们贡献了：（1）SRH沟通的实证主题，（2）间接沟通的分类框架，（3）LLM性能评估，以及（4）基于文化背景的SRH沟通设计建议。 

---
# Thinking Like a Student: AI-Supported Reflective Planning in a Theory-Intensive Computer Science Course 

**Title (ZH)**: 从学生视角思考：AI支持的反思性规划在理论密集型计算机科学课程中的应用 

**Authors**: Noa Izsak  

**Link**: [PDF](https://arxiv.org/pdf/2511.01906)  

**Abstract**: In the aftermath of COVID-19, many universities implemented supplementary "reinforcement" roles to support students in demanding courses. Although the name for such roles may differ between institutions, the underlying idea of providing structured supplementary support is common. However, these roles were often poorly defined, lacking structured materials, pedagogical oversight, and integration with the core teaching team. This paper reports on the redesign of reinforcement sessions in a challenging undergraduate course on formal methods and computational models, using a large language model (LLM) as a reflective planning tool. The LLM was prompted to simulate the perspective of a second-year student, enabling the identification of conceptual bottlenecks, gaps in intuition, and likely reasoning breakdowns before classroom delivery. These insights informed a structured, repeatable session format combining targeted review, collaborative examples, independent student work, and guided walkthroughs. Conducted over a single semester, the intervention received positive student feedback, indicating increased confidence, reduced anxiety, and improved clarity, particularly in abstract topics such as the pumping lemma and formal language expressive power comparisons. The findings suggest that reflective, instructor-facing use of LLMs can enhance pedagogical design in theoretically dense domains and may be adaptable to other cognitively demanding computer science courses. 

**Abstract (ZH)**: COVID-19之后，在一门形式方法与计算模型的挑战性本科课程中重新设计强化会话：使用大型语言模型作为反思性规划工具 

---
# Multi-Personality Generation of LLMs at Decoding-time 

**Title (ZH)**: 解码时LLM的多个性格生成 

**Authors**: Rongxin Chen, Yunfan Li, Yige Yuan, Bingbing Xu, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.01891)  

**Abstract**: Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at this https URL . 

**Abstract (ZH)**: 多个性格生成框架：在解码时结合多个个性特征，对于大型语言模型来说是一项基础性挑战。现有的基于重新训练的方法成本高且扩展性差，而解码时的方法通常依赖于外部模型或启发式方法，限制了灵活性和鲁棒性。在本文中，我们提出了一种新颖的多个性格生成（MPG）框架，该框架在解码时结合多个个性特征，无需依赖稀缺的多维模型或额外训练，而是利用单维模型中的隐含密度比来重新定义任务为从这些比率聚合的目标策略中采样。为了高效实施MPG，我们设计了推测性块级拒绝采样（SCR），该方法分块生成响应并在滑动窗口内并行验证它们，这显著减少了计算开销同时保持高质量生成。实验结果表明，MPG在MBTI人格和角色扮演任务中表现出色，生成质量可提升16%-18%。代码和数据可在该网址获取。 

---
# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization 

**Title (ZH)**: CudaForge: 一种带有硬件反馈的CUDA内核优化智能体框架 

**Authors**: Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.01884)  

**Abstract**: Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench. Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at this https URL 

**Abstract (ZH)**: 基于CUDA内核生成与优化的无训练多智能体工作流CudaForge 

---
# EdgeReasoning: Characterizing Reasoning LLM Deployment on Edge GPUs 

**Title (ZH)**: 边缘推理：边端GPU上推理预训练语言模型的特点研究 

**Authors**: Benjamin Kubwimana, Qijing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01866)  

**Abstract**: Edge intelligence paradigm is increasingly demanded by the emerging autonomous systems, such as robotics. Beyond ensuring privacy-preserving operation and resilience in connectivity-limited environments, edge deployment offers significant energy and cost advantages over cloud-based solutions. However, deploying large language models (LLMs) for reasoning tasks on edge GPUs faces critical challenges from strict latency constraints and limited computational resources. To navigate these constraints, developers must balance multiple design factors - choosing reasoning versus non-reasoning architectures, selecting appropriate model sizes, allocating token budgets, and applying test-time scaling strategies - to meet target latency and optimize accuracy. Yet guidance on optimal combinations of these variables remains scarce. In this work, we present EdgeReasoning, a comprehensive study characterizing the deployment of reasoning LLMs on edge GPUs. We systematically quantify latency-accuracy tradeoffs across various LLM architectures and model sizes. We systematically evaluate prompt-based and model-tuning-based techniques for reducing reasoning token length while maintaining performance quality. We further profile test-time scaling methods with varying degrees of parallelism to maximize accuracy under strict latency budgets. Through these analyses, EdgeReasoning maps the Pareto frontier of achievable accuracy-latency configurations, offering systematic guidance for optimal edge deployment of reasoning LLMs. 

**Abstract (ZH)**: 边缘智能范式日益被新兴自主系统，如机器人所需求。除了在连接受限环境中确保隐私保护操作和弹性之外，边缘部署在能耗和成本方面相比基于云的解决方案具有显著优势。然而，在边缘GPU上部署大型语言模型（LLMs）进行推理任务面临着严格延迟约束和有限计算资源的关键挑战。为了应对这些限制，开发者必须在多重设计因素之间进行平衡——选择推理架构还是非推理架构、选择合适模型规模、分配令牌预算，并应用测试时缩放策略，以满足延迟目标并优化准确性。然而，关于这些变量的最优组合的指导仍然稀缺。在本工作中，我们提出了EdgeReasoning，一项全面研究边缘GPU上部署推理LLMs的特性。我们系统地量化了各种LLM架构和模型规模下的延迟-准确性权衡。我们系统地评估了基于提示和基于模型调优的技术，以减少推理令牌长度同时保持性能质量。我们进一步分析了具有不同并行度的测试时缩放方法，以在严格延迟预算下最大化准确性。通过这些分析，EdgeReasoning绘制了可实现的准确性-延迟配置的帕累托前沿，为推理LLMs的最优边缘部署提供系统性指导。 

---
