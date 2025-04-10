# AssistanceZero: Scalably Solving Assistance Games 

**Title (ZH)**: AssistanceZero：规模化解决协助博弈 

**Authors**: Cassidy Laidlaw, Eli Bronstein, Timothy Guo, Dylan Feng, Lukas Berglund, Justin Svegliato, Stuart Russell, Anca Dragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07091)  

**Abstract**: Assistance games are a promising alternative to reinforcement learning from human feedback (RLHF) for training AI assistants. Assistance games resolve key drawbacks of RLHF, such as incentives for deceptive behavior, by explicitly modeling the interaction between assistant and user as a two-player game where the assistant cannot observe their shared goal. Despite their potential, assistance games have only been explored in simple settings. Scaling them to more complex environments is difficult because it requires both solving intractable decision-making problems under uncertainty and accurately modeling human users' behavior. We present the first scalable approach to solving assistance games and apply it to a new, challenging Minecraft-based assistance game with over $10^{400}$ possible goals. Our approach, AssistanceZero, extends AlphaZero with a neural network that predicts human actions and rewards, enabling it to plan under uncertainty. We show that AssistanceZero outperforms model-free RL algorithms and imitation learning in the Minecraft-based assistance game. In a human study, our AssistanceZero-trained assistant significantly reduces the number of actions participants take to complete building tasks in Minecraft. Our results suggest that assistance games are a tractable framework for training effective AI assistants in complex environments. Our code and models are available at this https URL. 

**Abstract (ZH)**: 辅助游戏是强化学习从人类反馈（RLHF）之外的一种有前途的替代方案，用于训练AI助手。辅助游戏通过明确将助手与用户之间的交互建模为一个两位玩家的游戏来解决RLHF的关键劣势，其中助手无法观察到他们共享的目标。这解决了欺骗行为的动机问题。尽管它们具有潜力，但辅助游戏仅在简单设置中被探索过。将它们扩展到更复杂的环境中是困难的，因为这需要解决不确定条件下的难解决策问题，并准确建模人类使用者的行为。我们提出了第一个可扩展的辅助游戏解决方案，并将其应用于一个新挑战的基于Minecraft的辅助游戏，该游戏具有超过$10^{400}$个可能目标。我们的方法，AssistanceZero，扩展了AlphaZero，加入了能够预测人类行动和奖励的神经网络，使其能够在不确定条件下进行规划。我们证明了AssistanceZero在基于Minecraft的辅助游戏中优于无模型的方法和模仿学习。在一项人类研究中，我们的AssistanceZero训练的助手显着减少了参与者在Minecraft中完成建筑任务所需要的操作次数。我们的结果表明，辅助游戏是一个在复杂环境中训练有效AI助手的可解框架。我们的代码和模型可通过以下链接获得：this https URL。 

---
# SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills 

**Title (ZH)**: SkillWeaver: 网站智能体可通过发现和提升技能实现自我改进 

**Authors**: Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.07079)  

**Abstract**: To survive and thrive in complex environments, humans have evolved sophisticated self-improvement mechanisms through environment exploration, hierarchical abstraction of experiences into reuseable skills, and collaborative construction of an ever-growing skill repertoire. Despite recent advancements, autonomous web agents still lack crucial self-improvement capabilities, struggling with procedural knowledge abstraction, refining skills, and skill composition. In this work, we introduce SkillWeaver, a skill-centric framework enabling agents to self-improve by autonomously synthesizing reusable skills as APIs. Given a new website, the agent autonomously discovers skills, executes them for practice, and distills practice experiences into robust APIs. Iterative exploration continually expands a library of lightweight, plug-and-play APIs, significantly enhancing the agent's capabilities. Experiments on WebArena and real-world websites demonstrate the efficacy of SkillWeaver, achieving relative success rate improvements of 31.8% and 39.8%, respectively. Additionally, APIs synthesized by strong agents substantially enhance weaker agents through transferable skills, yielding improvements of up to 54.3% on WebArena. These results demonstrate the effectiveness of honing diverse website interactions into APIs, which can be seamlessly shared among various web agents. 

**Abstract (ZH)**: 为了在复杂环境中生存和发展，人类通过环境探索、经验的层次化抽象以及技能的合作构建进化出了复杂的自我改进机制。尽管最近取得了进展，自主网络代理仍然缺乏关键的自我改进能力，难以进行程序知识抽象、技能优化和技能组合。在本文中，我们介绍了一种以技能为中心的框架SkillWeaver，该框架使代理能够通过自主合成可重用的技能作为API来实现自我改进。面对一个新的网站，代理能够自主发现技能、执行技能进行练习，并将实践经验提炼成稳健的API。迭代探索不断扩展一个包含轻量级、即插即用API的库，显著增强了代理的能力。在WebArena和真实网站上的实验结果表明，SkillWeaver的有效性，在WebArena上分别实现了31.8%和39.8%的成功率改进。此外，由强代理合成的API通过可转移的技能显著增强了弱代理，WebArena上实现了高达54.3%的成功率改进。这些结果证明了将多样的网站交互打磨成API的有效性，这些API可以在各种网络代理之间无缝共享。 

---
# $Π$-NeSy: A Possibilistic Neuro-Symbolic Approach 

**Title (ZH)**: Π-NeSy: 一种可能性神经符号方法 

**Authors**: Ismaïl Baaj, Pierre Marquis  

**Link**: [PDF](https://arxiv.org/pdf/2504.07055)  

**Abstract**: In this article, we introduce a neuro-symbolic approach that combines a low-level perception task performed by a neural network with a high-level reasoning task performed by a possibilistic rule-based system. The goal is to be able to derive for each input instance the degree of possibility that it belongs to a target (meta-)concept. This (meta-)concept is connected to intermediate concepts by a possibilistic rule-based system. The probability of each intermediate concept for the input instance is inferred using a neural network. The connection between the low-level perception task and the high-level reasoning task lies in the transformation of neural network outputs modeled by probability distributions (through softmax activation) into possibility distributions. The use of intermediate concepts is valuable for the explanation purpose: using the rule-based system, the classification of an input instance as an element of the (meta-)concept can be justified by the fact that intermediate concepts have been recognized.
From the technical side, our contribution consists of the design of efficient methods for defining the matrix relation and the equation system associated with a possibilistic rule-based system. The corresponding matrix and equation are key data structures used to perform inferences from a possibilistic rule-based system and to learn the values of the rule parameters in such a system according to a training data sample. Furthermore, leveraging recent results on the handling of inconsistent systems of fuzzy relational equations, an approach for learning rule parameters according to multiple training data samples is presented. Experiments carried out on the MNIST addition problems and the MNIST Sudoku puzzles problems highlight the effectiveness of our approach compared with state-of-the-art neuro-symbolic ones. 

**Abstract (ZH)**: 一种结合神经网络和Possibilistic规则系统的神经符号方法：为每个输入实例推导其属于目标（元）概念的程度可能性 

---
# Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration 

**Title (ZH)**: 基于案例推理的LLM代理回顾：理论基础、架构组件以及认知整合 

**Authors**: Kostas Hatalis, Despina Christou, Vyshnavi Kondapalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.06943)  

**Abstract**: Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents. 

**Abstract (ZH)**: 基于大型语言模型的智能体：通过案例推理增强的能力探索 

---
# FamilyTool: A Multi-hop Personalized Tool Use Benchmark 

**Title (ZH)**: FamilyTool: 一种多跳个性化工具使用基准 

**Authors**: Yuxin Wang, Yiran Guo, Yining Zheng, Zhangyue Yin, Shuo Chen, Jie Yang, Jiajun Chen, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06766)  

**Abstract**: The integration of tool learning with Large Language Models (LLMs) has expanded their capabilities in handling complex tasks by leveraging external tools. However, existing benchmarks for tool learning inadequately address critical real-world personalized scenarios, particularly those requiring multi-hop reasoning and inductive knowledge adaptation in dynamic environments. To bridge this gap, we introduce FamilyTool, a novel benchmark grounded in a family-based knowledge graph (KG) that simulates personalized, multi-hop tool use scenarios. FamilyTool challenges LLMs with queries spanning 1 to 3 relational hops (e.g., inferring familial connections and preferences) and incorporates an inductive KG setting where models must adapt to unseen user preferences and relationships without re-training, a common limitation in prior approaches that compromises generalization. We further propose KGETool: a simple KG-augmented evaluation pipeline to systematically assess LLMs' tool use ability in these settings. Experiments reveal significant performance gaps in state-of-the-art LLMs, with accuracy dropping sharply as hop complexity increases and inductive scenarios exposing severe generalization deficits. These findings underscore the limitations of current LLMs in handling personalized, evolving real-world contexts and highlight the urgent need for advancements in tool-learning frameworks. FamilyTool serves as a critical resource for evaluating and advancing LLM agents' reasoning, adaptability, and scalability in complex, dynamic environments. Code and dataset are available at Github. 

**Abstract (ZH)**: 工具学习与大型语言模型的集成扩展了它们处理复杂任务的能力，通过利用外部工具。然而，现有的工具学习基准在解决关键的实际个性化场景方面存在不足，特别是在需要多跳推理和动态环境中归纳知识适应的能力上。为弥补这一差距，我们介绍了基于家庭知识图谱（KG）的FamilyTool，这是一种新的基准方法，模拟了个性化、多跳工具使用场景。FamilyTool 用涉及1到3个关系跳数的查询（例如，推断家庭关系和偏好）挑战大型语言模型，并引入了一个归纳KG设置，要求模型在无需重新训练的情况下适应未见过的用户偏好和关系，这克服了先前方法中的一个常见限制，即影响泛化能力。我们还提出了KGETool：一种简单的知识图谱增强评估流水线，系统评估大型语言模型在这些环境下的工具使用能力。实验表明，最先进的大型语言模型在这些设置下的表现存在显著差异，随着跳数复杂性的增加，准确率急剧下降，而归纳场景暴露出严重的泛化缺陷。这些发现强调了当前大型语言模型处理个性化、不断演变的实际环境的局限性，并突显了工具学习框架亟待改进的迫切需求。FamilyTool 作为评估和促进大型语言模型代理在复杂动态环境中的推理、适应能力和可扩展性的关键资源。相关代码和数据集可在GitHub上获取。 

---
# Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis 

**Title (ZH)**: 正确的预测，错误的推理：揭示RA疾病诊断中LLM的偏向性 

**Authors**: Umakanta Maharana, Sarthak Verma, Avarna Agarwal, Prakashini Mruthyunjaya, Dwarikanath Mahapatra, Sakir Ahmed, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2504.06581)  

**Abstract**: Large language models (LLMs) offer a promising pre-screening tool, improving early disease detection and providing enhanced healthcare access for underprivileged communities. The early diagnosis of various diseases continues to be a significant challenge in healthcare, primarily due to the nonspecific nature of early symptoms, the shortage of expert medical practitioners, and the need for prolonged clinical evaluations, all of which can delay treatment and adversely affect patient outcomes. With impressive accuracy in prediction across a range of diseases, LLMs have the potential to revolutionize clinical pre-screening and decision-making for various medical conditions. In this work, we study the diagnostic capability of LLMs for Rheumatoid Arthritis (RA) with real world patients data. Patient data was collected alongside diagnoses from medical experts, and the performance of LLMs was evaluated in comparison to expert diagnoses for RA disease prediction. We notice an interesting pattern in disease diagnosis and find an unexpected \textit{misalignment between prediction and explanation}. We conduct a series of multi-round analyses using different LLM agents. The best-performing model accurately predicts rheumatoid arthritis (RA) diseases approximately 95\% of the time. However, when medical experts evaluated the reasoning generated by the model, they found that nearly 68\% of the reasoning was incorrect. This study highlights a clear misalignment between LLMs high prediction accuracy and its flawed reasoning, raising important questions about relying on LLM explanations in clinical settings. \textbf{LLMs provide incorrect reasoning to arrive at the correct answer for RA disease diagnosis.} 

**Abstract (ZH)**: 大型语言模型（LLMs）提供了一种有前景的预筛查工具，有助于早期疾病检测并为贫困社区提供增强的医疗访问权限。在医疗保健领域，各种疾病的早期诊断仍然是一项重大挑战，主要原因在于早期症状的非特异性、专家医疗 practitioners 的短缺以及需要长时间的临床评估，所有这些都可能导致治疗延迟并影响患者结果。凭借在多种疾病预测中表现出的卓越准确性，LLMs 有潜力革新不同医疗条件的临床预筛查和决策过程。在本文中，我们研究了LLMs在现实患者数据中对类风湿性关节炎（RA）的诊断能力。患者数据与医疗专家的诊断结果一并收集，并将LLMs的表现与专家对RA疾病预测的诊断进行了比较。我们注意到一种有趣的疾病诊断模式，并发现了一个意想不到的“预测与解释之间的不一致”。我们使用不同的LLM代理进行了多轮分析。表现最佳的模型大约95%的时间准确预测了类风湿性关节炎（RA）疾病。然而，当医疗专家评估模型生成的推理时，他们发现近68%的推理是不正确的。本研究突显了LLMs高预测准确性与其推理缺陷之间的明确不一致，引发了在临床环境中依赖LLM解释的重要问题。**LLMs提供错误的推理以得出正确的RA疾病诊断答案。** 

---
# Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill? 

**Title (ZH)**: 缺省前提加剧过度思考：推理模型丧失批判性思维能力了吗？ 

**Authors**: Chenrui Fan, Ming Li, Lichao Sun, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06514)  

**Abstract**: We find that the response length of reasoning LLMs, whether trained by reinforcement learning or supervised learning, drastically increases for ill-posed questions with missing premises (MiP), ending up with redundant and ineffective thinking. This newly introduced scenario exacerbates the general overthinking issue to a large extent, which we name as the MiP-Overthinking. Such failures are against the ``test-time scaling law'' but have been widely observed on multiple datasets we curated with MiP, indicating the harm of cheap overthinking and a lack of critical thinking. Surprisingly, LLMs not specifically trained for reasoning exhibit much better performance on the MiP scenario, producing much shorter responses that quickly identify ill-posed queries. This implies a critical flaw of the current training recipe for reasoning LLMs, which does not encourage efficient thinking adequately, leading to the abuse of thinking patterns. To further investigate the reasons behind such failures, we conduct fine-grained analyses of the reasoning length, overthinking patterns, and location of critical thinking on different types of LLMs. Moreover, our extended ablation study reveals that the overthinking is contagious through the distillation of reasoning models' responses. These results improve the understanding of overthinking and shed novel insights into mitigating the problem. 

**Abstract (ZH)**: 我们发现，无论是通过强化学习还是监督学习训练的推理大语言模型，在缺失前提条件（MiP）的 poorly 提出的问题上的响应长度显著增加，最终导致无效且多余的思考。这一新引入的场景在很大程度上加剧了普遍存在的过度思考问题，我们将其命名为 MiP-过度思考。这些失败违背了“测试时缩放定律”，但在我们收集的多个包含 MiP 的数据集上广泛观察到，这表明了廉价的过度思考和缺乏批判性思考的危害。令人惊讶的是，未特别针对推理训练的大语言模型在 MiP 场景中表现出更佳性能，产生了更短的响应并迅速识别出 poorly 提出的查询。这暗示了目前推理大语言模型的训练方法存在关键缺陷，未能充分鼓励有效的思考，导致思考模式的滥用。为了进一步探讨这些失败的原因，我们对不同类型的 LLM 的推理长度、过度思考模式以及关键思考位置进行了精细分析。此外，我们的扩展消融研究揭示了过度思考可以通过推理模型响应的蒸馏传染。这些结果加深了我们对过度思考的理解，并提供了缓解该问题的新见解。 

---
# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning 

**Title (ZH)**: 塑造子空间：受限全面微调在持续学习中的应用 

**Authors**: Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2504.07097)  

**Abstract**: Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models. 

**Abstract (ZH)**: 大型语言模型中的持续学习易发生灾难性遗忘，其中适应新的任务会显著恶化之前学习任务的表现。现有方法通常依赖低秩、参数高效的更新，这限制了模型的表达能力和造成每个任务增加额外参数的问题，导致可扩展性问题。为解决这些限制，我们提出了一种新的基于自适应奇异值分解（SVD）的持续完整微调方法。该方法动态地识别任务特定的低秩参数子空间，并将更新约束为与先前任务关键方向正交，从而有效地减少干扰，同时无需额外的参数开销或存储先前任务的梯度。我们使用包括编码器-解码器（T5-Large）和仅解码器（LLaMA-2 7B）模型，广泛地在标准持续学习基准上评估了该方法，涵盖分类、生成和推理等多种任务。实验结果表明，该方法实现了最先进的结果，平均准确率比最近的基线（如O-LoRA）高出7%，并且在整个持续学习过程中，有效减少了遗忘程度，保持了模型的一般语言能力、指令跟随准确性和安全性。自适应SVD框架有效平衡了模型的可塑性和知识保留，为大型语言模型中的持续学习场景提供了实用、理论依据充分且计算高效的解决方案。 

---
# Are We Done with Object-Centric Learning? 

**Title (ZH)**: 我们已完成对象中心学习了吗？ 

**Authors**: Alexander Rubinstein, Ameya Prabhu, Matthias Bethge, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07092)  

**Abstract**: Object-centric learning (OCL) seeks to learn representations that only encode an object, isolated from other objects or background cues in a scene. This approach underpins various aims, including out-of-distribution (OOD) generalization, sample-efficient composition, and modeling of structured environments. Most research has focused on developing unsupervised mechanisms that separate objects into discrete slots in the representation space, evaluated using unsupervised object discovery. However, with recent sample-efficient segmentation models, we can separate objects in the pixel space and encode them independently. This achieves remarkable zero-shot performance on OOD object discovery benchmarks, is scalable to foundation models, and can handle a variable number of slots out-of-the-box. Hence, the goal of OCL methods to obtain object-centric representations has been largely achieved. Despite this progress, a key question remains: How does the ability to separate objects within a scene contribute to broader OCL objectives, such as OOD generalization? We address this by investigating the OOD generalization challenge caused by spurious background cues through the lens of OCL. We propose a novel, training-free probe called $\textbf{Object-Centric Classification with Applied Masks (OCCAM)}$, demonstrating that segmentation-based encoding of individual objects significantly outperforms slot-based OCL methods. However, challenges in real-world applications remain. We provide the toolbox for the OCL community to use scalable object-centric representations, and focus on practical applications and fundamental questions, such as understanding object perception in human cognition. Our code is available $\href{this https URL}{here}$. 

**Abstract (ZH)**: 面向对象的中心学习：基于分割的编码如何推动更广泛的OOD泛化能力 

---
# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs 

**Title (ZH)**: KG-LLM-Bench：一种评估大规模语言模型在文本化知识图上推理能力的可扩展基准 

**Authors**: Elan Markowitz, Krupa Galiya, Greg Ver Steeg, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07087)  

**Abstract**: Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks. 

**Abstract (ZH)**: 知识图谱已 emergence 作为一种流行的方法，用于向大规模语言模型（LLMs）注入最新的事实性知识。这通常通过将知识图谱转换为 LLM 可以在上下文中处理的文本来实现。尽管已经提出了多种知识图谱编码方法，但这一文本化过程对 LLM 性能的影响仍未得到充分探索。我们引入了 KG-LLM-Bench，这是一个涵盖五个知识图谱理解任务的全面且可扩展的基准测试，并评估了不同编码策略在各种基础模型上的性能影响。我们的 extensive 实验使用了七种语言模型和五种文本化策略，为优化 LLM 在 KG 推理任务上的性能提供了见解。 

---
# Self-Steering Language Models 

**Title (ZH)**: 自引导语言模型 

**Authors**: Gabriel Grand, Joshua B. Tenenbaum, Vikash K. Mansinghka, Alexander K. Lew, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2504.07081)  

**Abstract**: While test-time reasoning enables language models to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its abstract structure--both how to verify solutions and how to search for them. This paper introduces DisCIPL, a method for "self-steering" LMs where a Planner model generates a task-specific inference program that is executed by a population of Follower models. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. In decoupling planning from execution, our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs. 

**Abstract (ZH)**: 虽然测试时推理使语言模型能够应对复杂任务，但在自然语言中进行搜索或规划可能会变得缓慢、昂贵且容易出错。但在语言模型难以模拟解决一个问题所需的精确推理步骤时，它们往往在描述其抽象结构方面表现出色，包括如何验证解决方案和如何搜索解决方案。本文介绍了DisCIPL方法，该方法实现了“自我引导”的语言模型，其中规划模型生成一个针对特定任务的推理程序，由一组跟随模型执行。我们的方法赋予语言模型编写递归搜索过程的能力，以引导模型的推理，从而实现新的可验证和高效的推理形式。当使用小型跟随者（例如，Llama-3.2-1B）实例化时，DisCIPL在具有挑战性的受限生成任务上与更大规模的模型（包括GPT-4o和o1）相当甚至表现更优。通过将规划与执行解耦，我们的工作开辟了高性能蒙特卡洛推理策略的设计空间，而这些策略能优于标准的-best-of-N采样策略，无需微调，且可以通过现有的语言模型自动实现。 

---
# DeduCE: Deductive Consistency as a Framework to Evaluate LLM Reasoning 

**Title (ZH)**: DeduCE: 通过检验推理一致性评估大型语言模型的框架 

**Authors**: Atharva Pandey, Kshitij Dubey, Rahul Sharma, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07080)  

**Abstract**: Despite great performance on Olympiad-level reasoning problems, frontier large language models can still struggle on high school math when presented with novel problems outside standard benchmarks. Going beyond final accuracy, we propose a deductive consistency metric to analyze chain-of-thought output from language models (LMs).Formally, deductive reasoning involves two subtasks: understanding a set of input premises and inferring the conclusions that follow from them. The proposed metric studies LMs' performance on these subtasks, with the goal of explaining LMs' reasoning errors on novel problems: how well do LMs understand input premises with increasing context lengths, and how well can they infer conclusions over multiple reasoning hops? Since existing benchmarks may be memorized, we develop a pipeline to evaluate LMs' deductive consistency on novel, perturbed versions of benchmark problems. On novel grade school math problems (GSM-8k), we find that LMs are fairly robust to increasing number of input premises, but suffer significant accuracy decay as the number of reasoning hops is increased. Interestingly, these errors are masked in the original benchmark as all models achieve near 100% accuracy. As we increase the number of solution steps using a synthetic dataset, prediction over multiple hops still remains the major source of error compared to understanding input premises. Other factors, such as shifts in language style or natural propagation of early errors do not explain the trends. Our analysis provides a new view to characterize LM reasoning -- as computations over a window of input premises and reasoning hops -- that can provide unified evaluation across problem domains. 

**Abstract (ZH)**: 尽管在奥林匹克水平的推理问题上表现出色，前沿的大语言模型在面临标准基准之外的新型高中数学问题时仍然会遇到困难。我们提出了一种演绎一致性度量来分析语言模型（LMs）的思维链输出，超越最终的准确性，旨在研究LMs在演绎推理子任务上的表现，解释其在新型问题上的推理错误：随着上下文长度的增加，LMs对输入前提的理解程度如何？随着推理跳数的增加，它们推导结论的能力又如何？由于现有基准可能存在记忆效应，我们开发了一个管道来评估LMs在新型、扰动过的基准问题上的演绎一致性。在新型小学数学问题（GSM-8k）上，我们发现LMs对输入前提数量的增加具有相当的鲁棒性，但在推理跳数增加时准确性显著下降。有趣的是，在原始基准中，这些错误被掩盖了，因为所有模型几乎达到了100%的准确率。随着使用合成数据集增加解题步骤，多跳预测仍然是比理解输入前提的主要错误来源。其他因素，如语言风格的转变或早期错误的自然传播，并不能解释这些趋势。我们的分析提供了一种新的视角来刻画LM的推理——作为输入前提和推理跳数窗口上的计算——这可以在不同问题领域提供统一的评估。 

---
# HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification 

**Title (ZH)**: HalluciNot: 通过上下文和常识验证的幻觉检测 

**Authors**: Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07069)  

**Abstract**: This paper introduces a comprehensive system for detecting hallucinations in large language model (LLM) outputs in enterprise settings. We present a novel taxonomy of LLM responses specific to hallucination in enterprise applications, categorizing them into context-based, common knowledge, enterprise-specific, and innocuous statements. Our hallucination detection model HDM-2 validates LLM responses with respect to both context and generally known facts (common knowledge). It provides both hallucination scores and word-level annotations, enabling precise identification of problematic content. To evaluate it on context-based and common-knowledge hallucinations, we introduce a new dataset HDMBench. Experimental results demonstrate that HDM-2 out-performs existing approaches across RagTruth, TruthfulQA, and HDMBench datasets. This work addresses the specific challenges of enterprise deployment, including computational efficiency, domain specialization, and fine-grained error identification. Our evaluation dataset, model weights, and inference code are publicly available. 

**Abstract (ZH)**: 本文引入了一个全面的企业环境中文本生成模型幻觉检测系统。我们提出了一种针对企业应用中幻觉的新型分类体系，将其分为基于上下文的、常识性的、企业特定的和无害的声明。我们的幻觉检测模型HDM-2根据上下文和一般公认的事实对文本生成模型的响应进行验证，提供了幻觉评分和单词级别注释，有助于精确识别问题内容。为了评估其在基于上下文和常识性幻觉上的表现，我们引入了一个新的数据集HDMBench。实验结果表明，HDM-2在RagTruth、TruthfulQA和HDMBench数据集上的表现优于现有方法。本文解决了企业部署特有的挑战，包括计算效率、领域专门化和精细错误识别。我们的评估数据集、模型权重和推理代码均公开提供。 

---
# RayFronts: Open-Set Semantic Ray Frontiers for Online Scene Understanding and Exploration 

**Title (ZH)**: RayFronts: 开集语义射线前沿及其在在线场景理解与探索中的应用 

**Authors**: Omar Alama, Avigyan Bhattacharya, Haoyang He, Seungchan Kim, Yuheng Qiu, Wenshan Wang, Cherie Ho, Nikhil Keetha, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2504.06994)  

**Abstract**: Open-set semantic mapping is crucial for open-world robots. Current mapping approaches either are limited by the depth range or only map beyond-range entities in constrained settings, where overall they fail to combine within-range and beyond-range observations. Furthermore, these methods make a trade-off between fine-grained semantics and efficiency. We introduce RayFronts, a unified representation that enables both dense and beyond-range efficient semantic mapping. RayFronts encodes task-agnostic open-set semantics to both in-range voxels and beyond-range rays encoded at map boundaries, empowering the robot to reduce search volumes significantly and make informed decisions both within & beyond sensory range, while running at 8.84 Hz on an Orin AGX. Benchmarking the within-range semantics shows that RayFronts's fine-grained image encoding provides 1.34x zero-shot 3D semantic segmentation performance while improving throughput by 16.5x. Traditionally, online mapping performance is entangled with other system components, complicating evaluation. We propose a planner-agnostic evaluation framework that captures the utility for online beyond-range search and exploration, and show RayFronts reduces search volume 2.2x more efficiently than the closest online baselines. 

**Abstract (ZH)**: 开放集语义映射对于开放世界机器人至关重要。RayFronts：统一表示实现高效密集和开放集语义映射 

---
# Enhancing Metabolic Syndrome Prediction with Hybrid Data Balancing and Counterfactuals 

**Title (ZH)**: 基于混合数据平衡和反事实方法的代谢综合征预测增强 

**Authors**: Sanyam Paresh Shah, Abdullah Mamun, Shovito Barua Soumma, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06987)  

**Abstract**: Metabolic Syndrome (MetS) is a cluster of interrelated risk factors that significantly increases the risk of cardiovascular diseases and type 2 diabetes. Despite its global prevalence, accurate prediction of MetS remains challenging due to issues such as class imbalance, data scarcity, and methodological inconsistencies in existing studies. In this paper, we address these challenges by systematically evaluating and optimizing machine learning (ML) models for MetS prediction, leveraging advanced data balancing techniques and counterfactual analysis. Multiple ML models, including XGBoost, Random Forest, TabNet, etc., were trained and compared under various data balancing techniques such as random oversampling (ROS), SMOTE, ADASYN, and CTGAN. Additionally, we introduce MetaBoost, a novel hybrid framework that integrates SMOTE, ADASYN, and CTGAN, optimizing synthetic data generation through weighted averaging and iterative weight tuning to enhance the model's performance (achieving a 1.14% accuracy improvement over individual balancing techniques). A comprehensive counterfactual analysis is conducted to quantify feature-level changes required to shift individuals from high-risk to low-risk categories. The results indicate that blood glucose (50.3%) and triglycerides (46.7%) were the most frequently modified features, highlighting their clinical significance in MetS risk reduction. Additionally, probabilistic analysis shows elevated blood glucose (85.5% likelihood) and triglycerides (74.9% posterior probability) as the strongest predictors. This study not only advances the methodological rigor of MetS prediction but also provides actionable insights for clinicians and researchers, highlighting the potential of ML in mitigating the public health burden of metabolic syndrome. 

**Abstract (ZH)**: 代谢综合征（MetS）是一组相互关联的危险因素，显著增加了心血管疾病和2型糖尿病的风险。尽管其具有全球流行性，但由于类不平衡、数据稀缺和现有研究中方法学不一致等问题，对MetS的准确预测仍然具有挑战性。在本文中，我们通过系统评估和优化机器学习（ML）模型来应对这些挑战，利用先进的数据平衡技术和反事实分析。多种ML模型，包括XGBoost、随机森林、TabNet等，均在随机过采样（ROS）、SMOTE、ADASYN和CTGAN等多种数据平衡技术下进行了训练和比较。此外，我们引入了MetaBoost，这是一种新颖的混合框架，将SMOTE、ADASYN和CTGAN结合在一起，通过加权平均和迭代权重调整来优化合成数据生成，从而提升模型性能（相对于个体平衡技术实现了1.14%的准确性提升）。我们进行了全面的反事实分析，以量化将个体从高风险类别转变为低风险类别的所需特征级变化。结果表明，血糖（50.3%）和甘油三酯（46.7%）是最常被修改的特征，突显了它们在降低代谢综合征风险中的临床意义。此外，概率分析显示血糖（85.5%可能性）和甘油三酯（74.9%后验概率）是最强的预测因素。本研究不仅提高了代谢综合征预测的方法学严谨性，还为临床医师和研究人员提供了可行的见解，突显了机器学习在减轻代谢综合征公共卫生负担方面的潜力。 

---
# RNN-Transducer-based Losses for Speech Recognition on Noisy Targets 

**Title (ZH)**: 基于RNN-Transducer的在噪声目标下语音识别的损失函数 

**Authors**: Vladimir Bataev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06963)  

**Abstract**: Training speech recognition systems on noisy transcripts is a significant challenge in industrial pipelines, where datasets are enormous and ensuring accurate transcription for every instance is difficult. In this work, we introduce novel loss functions to mitigate the impact of transcription errors in RNN-Transducer models. Our Star-Transducer loss addresses deletion errors by incorporating "skip frame" transitions in the loss lattice, restoring over 90% of the system's performance compared to models trained with accurate transcripts. The Bypass-Transducer loss uses "skip token" transitions to tackle insertion errors, recovering more than 60% of the quality. Finally, the Target-Robust Transducer loss merges these approaches, offering robust performance against arbitrary errors. Experimental results demonstrate that the Target-Robust Transducer loss significantly improves RNN-T performance on noisy data by restoring over 70% of the quality compared to well-transcribed data. 

**Abstract (ZH)**: 在嘈杂转录数据上训练语音识别系统是工业管道中的一个重大挑战，其中数据集庞大，确保每个实例的准确转录具有困难。在此项工作中，我们引入了新的损失函数以缓解基于RNN-Transducer模型中的转录错误的影响。我们的Star-Transducer损失通过在损失网格中引入“跳帧”转换来解决删除错误，恢复了系统超过90%的性能。Bypass-Transducer损失使用“跳令牌”转换来应对插入错误，恢复了超过60%的质量。最后，Target-Robust Transducer损失结合了这些方法，提供了对任意错误的稳健性能。实验结果表明，Target-Robust Transducer损失显著改进了基于RNN-T模型在嘈杂数据上的性能，恢复了与准确转录数据相比超过70%的质量。 

---
# Efficient Self-Supervised Learning for Earth Observation via Dynamic Dataset Curation 

**Title (ZH)**: 基于动态数据集编排的高效自我监督学习在地球观测中的应用 

**Authors**: Thomas Kerdreux, Alexandre Tuel, Quentin Febvre, Alexis Mouche, Bertrand Chapron  

**Link**: [PDF](https://arxiv.org/pdf/2504.06962)  

**Abstract**: Self-supervised learning (SSL) has enabled the development of vision foundation models for Earth Observation (EO), demonstrating strong transferability across diverse remote sensing tasks. While prior work has focused on network architectures and training strategies, the role of dataset curation, especially in balancing and diversifying pre-training datasets, remains underexplored. In EO, this challenge is amplified by the redundancy and heavy-tailed distributions common in satellite imagery, which can lead to biased representations and inefficient training.
In this work, we propose a dynamic dataset pruning strategy designed to improve SSL pre-training by maximizing dataset diversity and balance. Our method iteratively refines the training set without requiring a pre-existing feature extractor, making it well-suited for domains where curated datasets are limited or unavailable. We demonstrate our approach on the Sentinel-1 Wave Mode (WV) Synthetic Aperture Radar (SAR) archive, a challenging dataset dominated by ocean observations. We train models from scratch on the entire Sentinel-1 WV archive spanning 10 years. Across three downstream tasks, our results show that dynamic pruning improves both computational efficiency and representation quality, leading to stronger transferability.
We also release the weights of Nereus-SAR-1, the first model in the Nereus family, a series of foundation models for ocean observation and analysis using SAR imagery, at this http URL. 

**Abstract (ZH)**: 自我监督学习（SSL）已促进了地球观测（EO）领域视觉基础模型的发展，展示了其在多种遥感任务中的强大迁移能力。虽然先前的工作主要集中在网络架构和训练策略上，但数据集编排的作用，尤其是在平衡和多样化预训练数据集方面的角色，仍然未得到充分探索。在EO领域，这一挑战因卫星图像中常见的冗余性和重尾分布而加剧，可能导致有偏的表示和低效的训练。

在本文中，我们提出了一种动态数据集剪枝策略，旨在通过最大化数据集的多样性和平衡来提高SSL预训练的效果。该方法可在无需先存特征提取器的情况下迭代优化训练集，使其适用于受限或不可用标注数据集的领域。我们在Sentinel-1波模式（WV）合成孔径雷达（SAR）存档上展示了我们的方法，这是一个以海洋观测为主导的具有挑战性的数据集。我们在整个涵盖10年的Sentinel-1 WV存档上从头训练模型。在三个下游任务中，我们的结果显示动态剪枝提高了计算效率和表示质量，从而增强了迁移能力。

我们也发布了Nereus-SAR-1的权重，这是Nereus家族中的第一个模型，是一系列用于海洋观测和分析的SAR图像基础模型。详情请参阅此链接。 

---
# Adaptive Computation Pruning for the Forgetting Transformer 

**Title (ZH)**: 自适应计算剪枝以减轻遗忘变换器遗忘现象 

**Authors**: Zhixuan Lin, Johan Obando-Ceron, Xu Owen He, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2504.06949)  

**Abstract**: The recently proposed Forgetting Transformer (FoX) incorporates a forget gate into softmax attention and has shown consistently better or on-par performance compared to the standard RoPE-based Transformer. Notably, many attention heads in FoX tend to forget quickly, causing their output at each timestep to rely primarily on the local context. Based on this observation, we propose Adaptive Computation Pruning (ACP) for FoX, a method that dynamically prunes computations involving input-output dependencies that are strongly decayed by the forget gate. This is achieved using a dynamically set pruning threshold that ensures that the pruned attention weights remain negligible. We apply ACP to language model pretraining with FoX and show it consistently reduces the number of FLOPs in softmax attention by around 70% across different model sizes and context lengths, resulting in a roughly 10% to 35% improvement in training throughput. Furthermore, longer context lengths yield greater computational savings. All these speed improvements are achieved without any performance degradation. We also perform several analyses to provide deeper insights into our method, such as examining the pruning patterns and analyzing the distribution of FLOP savings across different attention heads. Our code is available at this https URL. 

**Abstract (ZH)**: Adaptive Computation Pruning for Forgetting Transformer 

---
# Beyond Tools: Generative AI as Epistemic Infrastructure in Education 

**Title (ZH)**: 超越工具：生成式AI作为教育的认知基础设施 

**Authors**: Bodong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06928)  

**Abstract**: As generative AI rapidly integrates into educational infrastructures worldwide, it transforms how knowledge gets created, validated, and shared, yet current discourse inadequately addresses its implications as epistemic infrastructure mediating teaching and learning. This paper investigates how AI systems function as epistemic infrastructures in education and their impact on human epistemic agency. Adopting a situated cognition perspective and following a value-sensitive design approach, the study conducts a technical investigation of two representative AI systems in educational settings, analyzing their impact on teacher practice across three dimensions: affordances for skilled epistemic actions, support for epistemic sensitivity, and implications for long-term habit formation. The analysis reveals that current AI systems inadequately support teachers' skilled epistemic actions, insufficiently foster epistemic sensitivity, and potentially cultivate problematic habits that prioritize efficiency over epistemic agency. To address these challenges, the paper recommends recognizing the infrastructural transformation occurring in education, developing AI environments that stimulate skilled actions while upholding epistemic norms, and involving educators in AI design processes -- recommendations aimed at fostering AI integration that aligns with core educational values and maintains human epistemic agency. 

**Abstract (ZH)**: 随着生成式人工智能迅速融入全球教育基础设施，它正在改变知识的创造、验证和分享方式，然而当前的讨论未能充分探讨其作为促进教学与学习的认知基础设施所带来的影响。本文研究AI系统在教育中的认知基础设施功能及其对人类认知自主性的影响。本文采用情境认知视角和价值观敏感设计方法，对两种代表性教育AI系统的技术特性进行了分析，从技能性认知行动的能力、认知敏感性的支持以及对长期习惯形成的影响三个方面分析了其对教师实践的影响。分析表明，当前的AI系统在支持教师的技能性认知行动、促进认知敏感性方面存在不足，可能培养出以效率优先而非认知自主性的问题习惯。为应对这些挑战，本文建议承认教育中的基础设施变革，开发能够促进技能性行动并维护认知规范的AI环境，并让教育者参与AI的设计过程——这些建议旨在促进与核心教育价值观相一致的AI整合，维护人类的认知自主性。 

---
# Are Vision-Language Models Ready for Dietary Assessment? Exploring the Next Frontier in AI-Powered Food Image Recognition 

**Title (ZH)**: 视觉-语言模型准备好应对饮食评估挑战了吗？探索AI驱动食品图像识别的下一个前沿领域 

**Authors**: Sergio Romero-Tapiador, Ruben Tolosana, Blanca Lacruz-Pleguezuelos, Laura Judith Marcos Zambrano, Guadalupe X.Bazán, Isabel Espinosa-Salinas, Julian Fierrez, Javier Ortega-Garcia, Enrique Carrillo de Santa Pau, Aythami Morales  

**Link**: [PDF](https://arxiv.org/pdf/2504.06925)  

**Abstract**: Automatic dietary assessment based on food images remains a challenge, requiring precise food detection, segmentation, and classification. Vision-Language Models (VLMs) offer new possibilities by integrating visual and textual reasoning. In this study, we evaluate six state-of-the-art VLMs (ChatGPT, Gemini, Claude, Moondream, DeepSeek, and LLaVA), analyzing their capabilities in food recognition at different levels. For the experimental framework, we introduce the FoodNExTDB, a unique food image database that contains 9,263 expert-labeled images across 10 categories (e.g., "protein source"), 62 subcategories (e.g., "poultry"), and 9 cooking styles (e.g., "grilled"). In total, FoodNExTDB includes 50k nutritional labels generated by seven experts who manually annotated all images in the database. Also, we propose a novel evaluation metric, Expert-Weighted Recall (EWR), that accounts for the inter-annotator variability. Results show that closed-source models outperform open-source ones, achieving over 90% EWR in recognizing food products in images containing a single product. Despite their potential, current VLMs face challenges in fine-grained food recognition, particularly in distinguishing subtle differences in cooking styles and visually similar food items, which limits their reliability for automatic dietary assessment. The FoodNExTDB database is publicly available at this https URL. 

**Abstract (ZH)**: 基于食品图像的自动膳食评估仍然是一项挑战，需要精确的食品检测、分割和分类。视觉-语言模型（VLMs）通过结合视觉和文本推理提供了新的可能性。在本研究中，我们评估了六种最先进的VLMs（ChatGPT、Gemini、Claude、Moondream、DeepSeek和LLaVA），分析了它们在不同层次上的食品识别能力。在实验框架中，我们引入了FoodNExTDB，这是一个独特的食品图像数据库，包含了9,263张专家标注的图像，涉及10个类别（例如，“蛋白质来源”）、62个子类别（例如，“禽类”）和9种烹饪风格（例如，“烤制”）。FoodNExTDB总共包括由七位专家手动标注数据库中所有图像生成的50,000个营养标签。此外，我们提出了一种新的评估指标——专家加权召回率（EWR），以考虑标注者间的一致性差异。结果显示，闭源模型优于开源模型，在包含单个产品的图像中识别食品产品的EWR超过90%。尽管有这些潜力，当前的VLMs在细粒度食品识别方面仍面临挑战，尤其是在区分烹饪风格的细微差别和视觉相似的食品项目方面，这限制了它们在自动膳食评估中的可靠性。FoodNExTDB数据库可在此处公开访问：this https URL。 

---
# Longitudinal Assessment of Lung Lesion Burden in CT 

**Title (ZH)**: CT纵行评估肺部病灶负荷 

**Authors**: Tejas Sudharshan Mathai, Benjamin Hou, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.06924)  

**Abstract**: In the U.S., lung cancer is the second major cause of death. Early detection of suspicious lung nodules is crucial for patient treatment planning, management, and improving outcomes. Many approaches for lung nodule segmentation and volumetric analysis have been proposed, but few have looked at longitudinal changes in total lung tumor burden. In this work, we trained two 3D models (nnUNet) with and without anatomical priors to automatically segment lung lesions and quantified total lesion burden for each patient. The 3D model without priors significantly outperformed ($p < .001$) the model trained with anatomy priors. For detecting clinically significant lesions $>$ 1cm, a precision of 71.3\%, sensitivity of 68.4\%, and F1-score of 69.8\% was achieved. For segmentation, a Dice score of 77.1 $\pm$ 20.3 and Hausdorff distance error of 11.7 $\pm$ 24.1 mm was obtained. The median lesion burden was 6.4 cc (IQR: 2.1, 18.1) and the median volume difference between manual and automated measurements was 0.02 cc (IQR: -2.8, 1.2). Agreements were also evaluated with linear regression and Bland-Altman plots. The proposed approach can produce a personalized evaluation of the total tumor burden for a patient and facilitate interval change tracking over time. 

**Abstract (ZH)**: 在美国，肺癌是第二大死因。早期检测可疑肺结节对于患者的治疗规划、管理和改善预后至关重要。虽然已经提出了许多肺结节分割和容积分析的方法，但很少有研究关注总体肺肿瘤负担的纵向变化。在本研究中，我们训练了两个3D模型（nnUNet），一个是带解剖先验知识的，另一个是不带解剖先验知识的，以自动分割肺部病灶并量化每位患者的整体病灶负担。不带解剖先验知识的3D模型显著优于带解剖先验知识的模型（$p < .001$）。对于检测临床显著的病灶（＞1cm），实现了71.3%的精确率、68.4%的灵敏度和69.8%的F1分数。对于分割，获得了77.1 $\pm$ 20.3的Dice分数和11.7 $\pm$ 24.1 mm的哈斯多夫距离误差。中位病灶负担为6.4 cc（四分位距：2.1, 18.1），手工测量与自动测量的中位体积差异为0.02 cc（四分位距：-2.8, 1.2）。还使用线性回归和Bland-Altman图评估了协议的一致性。所提出的方法可以为每位患者提供个性化的总体肿瘤负担评估，并有助于随时间跟踪间隔变化。 

---
# Leveraging Anatomical Priors for Automated Pancreas Segmentation on Abdominal CT 

**Title (ZH)**: 利用解剖先验信息实现腹部CT胰腺自动分割 

**Authors**: Anisa V. Prasad, Tejas Sudharshan Mathai, Pritam Mukherjee, Jianfei Liu, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.06921)  

**Abstract**: An accurate segmentation of the pancreas on CT is crucial to identify pancreatic pathologies and extract imaging-based biomarkers. However, prior research on pancreas segmentation has primarily focused on modifying the segmentation model architecture or utilizing pre- and post-processing techniques. In this article, we investigate the utility of anatomical priors to enhance the segmentation performance of the pancreas. Two 3D full-resolution nnU-Net models were trained, one with 8 refined labels from the public PANORAMA dataset, and another that combined them with labels derived from the public TotalSegmentator (TS) tool. The addition of anatomical priors resulted in a 6\% increase in Dice score ($p < .001$) and a 36.5 mm decrease in Hausdorff distance for pancreas segmentation ($p < .001$). Moreover, the pancreas was always detected when anatomy priors were used, whereas there were 8 instances of failed detections without their use. The use of anatomy priors shows promise for pancreas segmentation and subsequent derivation of imaging biomarkers. 

**Abstract (ZH)**: 基于解剖先验的胰腺分割及其影像生物标志物提取的准确性增强 

---
# An Analysis of Temporal Dropout in Earth Observation Time Series for Regression Tasks 

**Title (ZH)**: 地球观测时间序列中回归任务中时间段下采样的分析 

**Authors**: Miro Miranda, Francisco Mena, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06915)  

**Abstract**: Missing instances in time series data impose a significant challenge to deep learning models, particularly in regression tasks. In the Earth Observation field, satellite failure or cloud occlusion frequently results in missing time-steps, introducing uncertainties in the predicted output and causing a decline in predictive performance. While many studies address missing time-steps through data augmentation to improve model robustness, the uncertainty arising at the input level is commonly overlooked. To address this gap, we introduce Monte Carlo Temporal Dropout (MC-TD), a method that explicitly accounts for input-level uncertainty by randomly dropping time-steps during inference using a predefined dropout ratio, thereby simulating the effect of missing data. To bypass the need for costly searches for the optimal dropout ratio, we extend this approach with Monte Carlo Concrete Temporal Dropout (MC-ConcTD), a method that learns the optimal dropout distribution directly. Both MC-TD and MC-ConcTD are applied during inference, leveraging Monte Carlo sampling for uncertainty quantification. Experiments on three EO time-series datasets demonstrate that MC-ConcTD improves predictive performance and uncertainty calibration compared to existing approaches. Additionally, we highlight the advantages of adaptive dropout tuning over manual selection, making uncertainty quantification more robust and accessible for EO applications. 

**Abstract (ZH)**: 时间序列数据中的缺失实例对深度学习模型构成了显著挑战，特别是在回归任务中。在地球观测领域，卫星故障或云遮挡频繁导致时间步骤缺失，引入预测输出的不确定性并导致预测性能下降。尽管许多研究通过数据增强来解决时间步骤缺失问题以提高模型的鲁棒性，但输入级的不确定性往往被忽视。为弥补这一缺口，我们引入了蒙特卡洛时间 dropout（MC-TD）方法，该方法在推理过程中通过预定义的 dropout 比例随机丢弃时间步骤，从而模拟缺失数据的效果。为了避免搜索最优 dropout 比例的高成本，我们通过蒙特卡洛混凝土时间 dropout（MC-ConcTD）方法进一步扩展了这一思路，该方法直接学习最优 dropout 分布。MC-TD 和 MC-ConcTD 在推理过程中应用，利用蒙特卡洛采样进行不确定性量化。在三个地球观测时间序列数据集上的实验表明，MC-ConcTD 相比现有方法能够提高预测性能和不确定性校准。此外，我们还强调了自适应 dropout 调整相对于手动选择的优势，使得不确定性量化在地球观测应用中更加 robust 和易用。 

---
# MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs 

**Title (ZH)**: MedSegFactory：基于文本引导的医疗图像-掩码对生成 

**Authors**: Jiawei Mao, Yuhan Wang, Yucheng Tang, Daguang Xu, Kang Wang, Yang Yang, Zongwei Zhou, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06897)  

**Abstract**: This paper presents MedSegFactory, a versatile medical synthesis framework that generates high-quality paired medical images and segmentation masks across modalities and tasks. It aims to serve as an unlimited data repository, supplying image-mask pairs to enhance existing segmentation tools. The core of MedSegFactory is a dual-stream diffusion model, where one stream synthesizes medical images and the other generates corresponding segmentation masks. To ensure precise alignment between image-mask pairs, we introduce Joint Cross-Attention (JCA), enabling a collaborative denoising paradigm by dynamic cross-conditioning between streams. This bidirectional interaction allows both representations to guide each other's generation, enhancing consistency between generated pairs. MedSegFactory unlocks on-demand generation of paired medical images and segmentation masks through user-defined prompts that specify the target labels, imaging modalities, anatomical regions, and pathological conditions, facilitating scalable and high-quality data generation. This new paradigm of medical image synthesis enables seamless integration into diverse medical imaging workflows, enhancing both efficiency and accuracy. Extensive experiments show that MedSegFactory generates data of superior quality and usability, achieving competitive or state-of-the-art performance in 2D and 3D segmentation tasks while addressing data scarcity and regulatory constraints. 

**Abstract (ZH)**: 本文介绍了MedSegFactory，一个多功能的医学合成框架，能够跨模态和任务生成高质量的配对医学图像和分割掩码。其目标是作为无限数据仓库，提供图像-掩码配对以增强现有的分割工具。MedSegFactory的核心是一个双流扩散模型，其中一条流生成医学图像，另一条流生成相应的分割掩码。为了确保图像-掩码配对之间的精确对齐，我们引入了联合交叉注意力（JCA），通过流之间的动态交叉条件实现协作去噪范式。这种双向交互使两个表示能够相互引导生成过程，从而增强生成配对的一致性。MedSegFactory通过用户定义的提示解锁按需生成配对医学图像和分割掩码，这些提示指定目标标签、成像模ality、解剖区域和病理条件，促进可扩展和高质量数据生成。这种新的医学图像合成范式能够无缝集成到各种医学成像工作流程中，提高效率和准确性。大量实验表明，MedSegFactory生成的数据质量和可用性更高，在2D和3D分割任务中实现了竞争力或先进水平的表现，同时解决了数据稀缺性和监管约束的问题。 

---
# Audio-visual Event Localization on Portrait Mode Short Videos 

**Title (ZH)**: portrait模式短视频中的音视频事件定位 

**Authors**: Wuyang Liu, Yi Chai, Yongpeng Yan, Yanzhen Ren  

**Link**: [PDF](https://arxiv.org/pdf/2504.06884)  

**Abstract**: Audio-visual event localization (AVEL) plays a critical role in multimodal scene understanding. While existing datasets for AVEL predominantly comprise landscape-oriented long videos with clean and simple audio context, short videos have become the primary format of online video content due to the the proliferation of smartphones. Short videos are characterized by portrait-oriented framing and layered audio compositions (e.g., overlapping sound effects, voiceovers, and music), which brings unique challenges unaddressed by conventional methods. To this end, we introduce AVE-PM, the first AVEL dataset specifically designed for portrait mode short videos, comprising 25,335 clips that span 86 fine-grained categories with frame-level annotations. Beyond dataset creation, our empirical analysis shows that state-of-the-art AVEL methods suffer an average 18.66% performance drop during cross-mode evaluation. Further analysis reveals two key challenges of different video formats: 1) spatial bias from portrait-oriented framing introduces distinct domain priors, and 2) noisy audio composition compromise the reliability of audio modality. To address these issues, we investigate optimal preprocessing recipes and the impact of background music for AVEL on portrait mode videos. Experiments show that these methods can still benefit from tailored preprocessing and specialized model design, thus achieving improved performance. This work provides both a foundational benchmark and actionable insights for advancing AVEL research in the era of mobile-centric video content. Dataset and code will be released. 

**Abstract (ZH)**: 音频-视觉事件定位（AVEL）在多模态场景理解中发挥着关键作用。现有用于AVEL的数据集主要包含景观取向的长视频，具有清晰简洁的音频背景，但随着智能手机的普及，短视频已成为在线视频内容的主要格式。短视频特征为 portrait 取向构图和多层次的音频组成（例如，叠加的声音效果、旁白和音乐），这带来了传统方法未曾解决的独特挑战。为此，我们引入了 AVE-PM，这是首个专门针对 portrait 模式短视频的 AVEL 数据集，包含 25,335 个跨越 86 个细粒度类别的剪辑，并附有帧级标注。除数据集创建之外，我们的实证分析表明，最新的 AVEL 方法在跨模式评估中平均性能下降 18.66%。进一步分析揭示了不同视频格式的两个关键挑战：1）portrait 取向构图的空间偏差引入了不同的领域先验，2）嘈杂的音频组成削弱了音频模态的可靠性。为解决这些问题，我们研究了针对 portrait 模式短视频的音频-视觉事件定位的最优预处理方法和背景音乐的影响。实验显示，这些方法仍能从定制预处理和专门模型设计中获益，从而提高性能。本工作为移动为中心的视频内容时代推进 AVEL 研究提供了基础基准和可操作的见解。数据集和代码将公开发布。 

---
# Compound and Parallel Modes of Tropical Convolutional Neural Networks 

**Title (ZH)**: 热带卷积神经网络的复合与并行模式 

**Authors**: Mingbo Li, Liying Liu, Ye Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.06881)  

**Abstract**: Convolutional neural networks have become increasingly deep and complex, leading to higher computational costs. While tropical convolutional neural networks (TCNNs) reduce multiplications, they underperform compared to standard CNNs. To address this, we propose two new variants - compound TCNN (cTCNN) and parallel TCNN (pTCNN)-that use combinations of tropical min-plus and max-plus kernels to replace traditional convolution kernels. This reduces multiplications and balances efficiency with performance. Experiments on various datasets show that cTCNN and pTCNN match or exceed the performance of other CNN methods. Combining these with conventional CNNs in deeper architectures also improves performance. We are further exploring simplified TCNN architectures that reduce parameters and multiplications with minimal accuracy loss, aiming for efficient and effective models. 

**Abstract (ZH)**: 卷积神经网络越来越深且复杂，导致更高的计算成本。尽管热带卷积神经网络（TCNNs）减少了乘法次数，但其性能低于标准CNNs。为了解决这个问题，我们提出了两种新的变体——组合TCNN（cTCNN）和并行TCNN（pTCNN），它们使用热带最小加法和最大加法核的组合来替代传统的卷积核。这减少了乘法次数，并平衡了效率与性能。在多种数据集上的实验表明，cTCNN和pTCNN的性能能够匹配甚至超越其他CNN方法。将这些方法与传统的CNN结合使用在更深的架构中也能改进性能。我们进一步研究了简化TCNN架构，以减少参数和乘法次数的同时保持最小的准确率损失，旨在获得高效且有效的模型。 

---
# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games 

**Title (ZH)**: 人格动态：揭示人格特质对文本基础上游戏代理人物的影响 

**Authors**: Seungwon Lim, Seungbeen Lee, Dongjun Min, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06868)  

**Abstract**: Artificial agents are increasingly central to complex interactions and decision-making tasks, yet aligning their behaviors with desired human values remains an open challenge. In this work, we investigate how human-like personality traits influence agent behavior and performance within text-based interactive environments. We introduce PANDA: PersonalityAdapted Neural Decision Agents, a novel method for projecting human personality traits onto agents to guide their behavior. To induce personality in a text-based game agent, (i) we train a personality classifier to identify what personality type the agent's actions exhibit, and (ii) we integrate the personality profiles directly into the agent's policy-learning pipeline. By deploying agents embodying 16 distinct personality types across 25 text-based games and analyzing their trajectories, we demonstrate that an agent's action decisions can be guided toward specific personality profiles. Moreover, certain personality types, such as those characterized by higher levels of Openness, display marked advantages in performance. These findings underscore the promise of personality-adapted agents for fostering more aligned, effective, and human-centric decision-making in interactive environments. 

**Abstract (ZH)**: 人工代理在复杂交互和决策任务中越来越占据中心地位，然而使其行为与期望的人类价值相一致仍然是一个开放性的挑战。在此工作中，我们探讨了人性特征如何影响文本交互环境中代理的行为和性能。我们提出了PANDA：个性适应神经决策代理，这是一种将人类个性特征投影到代理上以引导其行为的新方法。为了在文本交互代理中诱导个性特征，（i）我们训练了一个个性分类器来识别代理行为展现出的个性类型；（ii）我们将个性档案直接整合到代理的策略学习管道中。通过在25个文本交互游戏中部署16种不同的个性类型的代理，并分析它们的行为轨迹，我们展示了代理的行为决策可以导向特定的个性特征。此外，某些个性类型，如开放性水平较高的类型，在性能方面表现出明显的优势。这些发现表明，个性适应代理在促进更一致、更有效和更以人为中心的交互环境中决策方面具有潜力。 

---
# GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes 

**Title (ZH)**: GraspClutter6D：拥挤场景下鲁棒感知与抓取的大规模真实世界数据集 

**Authors**: Seunghyeok Back, Joosoon Lee, Kangmin Kim, Heeseon Rho, Geonhyup Lee, Raeyoung Kang, Sangbeom Lee, Sangjun Noh, Youngjin Lee, Taeyeop Lee, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.06866)  

**Abstract**: Robust grasping in cluttered environments remains an open challenge in robotics. While benchmark datasets have significantly advanced deep learning methods, they mainly focus on simplistic scenes with light occlusion and insufficient diversity, limiting their applicability to practical scenarios. We present GraspClutter6D, a large-scale real-world grasping dataset featuring: (1) 1,000 highly cluttered scenes with dense arrangements (14.1 objects/scene, 62.6\% occlusion), (2) comprehensive coverage across 200 objects in 75 environment configurations (bins, shelves, and tables) captured using four RGB-D cameras from multiple viewpoints, and (3) rich annotations including 736K 6D object poses and 9.3B feasible robotic grasps for 52K RGB-D images. We benchmark state-of-the-art segmentation, object pose estimation, and grasping detection methods to provide key insights into challenges in cluttered environments. Additionally, we validate the dataset's effectiveness as a training resource, demonstrating that grasping networks trained on GraspClutter6D significantly outperform those trained on existing datasets in both simulation and real-world experiments. The dataset, toolkit, and annotation tools are publicly available on our project website: this https URL. 

**Abstract (ZH)**: 复杂环境中 robust 抓取仍是一个开放的机器人挑战。尽管基准数据集显著推动了深度学习方法的发展，但它们主要集中在简单场景和轻度遮挡上，缺乏多样性，限制了其在实际场景中的应用。我们提出了 GraspClutter6D，这是一个大规模的真实世界抓取数据集，包含：(1) 1,000 个高度拥挤的场景，具有密集排列（平均每场景 14.1 个物体，62.6% 的遮挡），(2) 在 75 种环境配置（箱子、书架和桌子）中全面覆盖 200 个物体，使用四台 RGB-D 摄像机从多个视角采集，以及 (3) 丰富的标注，包括 73.6 万 6D 物体姿态和 93 亿个可行的机器人抓取动作，适用于 5.2 万张 RGB-D 图像。我们针对最新的分割、物体姿态估计和抓取检测方法进行基准测试，提供复杂环境中挑战的关键见解。此外，我们验证了该数据集作为训练资源的有效性，证明在 GraspClutter6D 上训练的抓取网络在模拟和实际实验中显著优于现有数据集训练的网络。该数据集、工具包和标注工具可在我们的项目网站上公开获取：this https URL。 

---
# EIDT-V: Exploiting Intersections in Diffusion Trajectories for Model-Agnostic, Zero-Shot, Training-Free Text-to-Video Generation 

**Title (ZH)**: EIDT-V：利用扩散轨迹中的交叉点进行模型无关的零样本、无需训练的文本到视频生成 

**Authors**: Diljeet Jagpal, Xi Chen, Vinay P. Namboodiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.06861)  

**Abstract**: Zero-shot, training-free, image-based text-to-video generation is an emerging area that aims to generate videos using existing image-based diffusion models. Current methods in this space require specific architectural changes to image generation models, which limit their adaptability and scalability. In contrast to such methods, we provide a model-agnostic approach. We use intersections in diffusion trajectories, working only with the latent values. We could not obtain localized frame-wise coherence and diversity using only the intersection of trajectories. Thus, we instead use a grid-based approach. An in-context trained LLM is used to generate coherent frame-wise prompts; another is used to identify differences between frames. Based on these, we obtain a CLIP-based attention mask that controls the timing of switching the prompts for each grid cell. Earlier switching results in higher variance, while later switching results in more coherence. Therefore, our approach can ensure appropriate control between coherence and variance for the frames. Our approach results in state-of-the-art performance while being more flexible when working with diverse image-generation models. The empirical analysis using quantitative metrics and user studies confirms our model's superior temporal consistency, visual fidelity and user satisfaction, thus providing a novel way to obtain training-free, image-based text-to-video generation. 

**Abstract (ZH)**: 零样本、无需训练的基于图像的文本到视频生成是一个新兴领域，旨在使用现有的图像扩散模型生成视频。现有方法需要对图像生成模型进行特定的架构变化，这限制了它们的适应性和可扩展性。与这些方法不同，我们提供了一种模型无关的方法。我们利用扩散轨迹的交集，仅使用潜在值。我们无法仅通过轨迹交集获得局部帧内一致性与多样性。因此，我们改用基于网格的方法。我们使用上下文训练的语言模型生成一致的帧级提示；另一个模型用于识别帧之间的差异。基于这些信息，我们获得了一个基于CLIP的注意力掩码，用于控制为每个网格单元切换提示的时间。较早切换会导致更高的方差，而较晚切换则会产生更多的一致性。因此，我们的方法可以在确保帧间一致性和方差之间的适当控制方面表现出优势。我们的方法在性能上达到最新水平，并且在与多种图像生成模型合作时更具灵活性。实证分析利用定量指标和用户研究证实了我们模型在时间一致性和视觉保真度方面的优越性以及用户的满意度，从而提供了一种新的方法来实现无需训练的数据驱动的图像到视频生成。 

---
# Integrating Cognitive Processing Signals into Language Models: A Review of Advances, Applications and Future Directions 

**Title (ZH)**: 将认知处理信号整合进语言模型：进展、应用及未来方向 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.06843)  

**Abstract**: Recently, the integration of cognitive neuroscience in Natural Language Processing (NLP) has gained significant attention. This article provides a critical and timely overview of recent advancements in leveraging cognitive signals, particularly Eye-tracking (ET) signals, to enhance Language Models (LMs) and Multimodal Large Language Models (MLLMs). By incorporating user-centric cognitive signals, these approaches address key challenges, including data scarcity and the environmental costs of training large-scale models. Cognitive signals enable efficient data augmentation, faster convergence, and improved human alignment. The review emphasises the potential of ET data in tasks like Visual Question Answering (VQA) and mitigating hallucinations in MLLMs, and concludes by discussing emerging challenges and research trends. 

**Abstract (ZH)**: 最近，认知神经科学在自然语言处理（NLP）中的集成引起了广泛关注。本文提供了一篇及时且批判性的综述，概述了通过利用认知信号，特别是眼动追踪（ET）信号，来增强语言模型（LMs）和多模态大规模语言模型（MLLMs）的近期进展。通过纳入用户中心的认知信号，这些方法解决了数据稀缺性和大规模模型训练的环境成本等关键挑战。认知信号使得高效的数据增强、更快的收敛和更好的人类对齐成为可能。综述强调了ET数据在视觉问答（VQA）任务和减轻MLLMs幻觉方面的潜力，并讨论了新兴挑战和研究趋势。 

---
# Adaptive Locally Linear Embedding 

**Title (ZH)**: 自适应局部线性嵌入 

**Authors**: Ali Goli, Mahdieh Alizadeh, Hadi Sadoghi Yazdi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06829)  

**Abstract**: Manifold learning techniques, such as Locally linear embedding (LLE), are designed to preserve the local neighborhood structures of high-dimensional data during dimensionality reduction. Traditional LLE employs Euclidean distance to define neighborhoods, which can struggle to capture the intrinsic geometric relationships within complex data. A novel approach, Adaptive locally linear embedding(ALLE), is introduced to address this limitation by incorporating a dynamic, data-driven metric that enhances topological preservation. This method redefines the concept of proximity by focusing on topological neighborhood inclusion rather than fixed distances. By adapting the metric based on the local structure of the data, it achieves superior neighborhood preservation, particularly for datasets with complex geometries and high-dimensional structures. Experimental results demonstrate that ALLE significantly improves the alignment between neighborhoods in the input and feature spaces, resulting in more accurate and topologically faithful embeddings. This approach advances manifold learning by tailoring distance metrics to the underlying data, providing a robust solution for capturing intricate relationships in high-dimensional datasets. 

**Abstract (ZH)**: 适配局部线性嵌入(ALLE):基于动态数据驱动度量的流形学习技术 

---
# Learning in Spiking Neural Networks with a Calcium-based Hebbian Rule for Spike-timing-dependent Plasticity 

**Title (ZH)**: 基于钙离子的时限依赖可塑性希布定律的脉冲神经网络学习 

**Authors**: Willian Soares Girão, Nicoletta Risi, Elisabetta Chicca  

**Link**: [PDF](https://arxiv.org/pdf/2504.06796)  

**Abstract**: Understanding how biological neural networks are shaped via local plasticity mechanisms can lead to energy-efficient and self-adaptive information processing systems, which promises to mitigate some of the current roadblocks in edge computing systems. While biology makes use of spikes to seamless use both spike timing and mean firing rate to modulate synaptic strength, most models focus on one of the two. In this work, we present a Hebbian local learning rule that models synaptic modification as a function of calcium traces tracking neuronal activity. We show how the rule reproduces results from spike time and spike rate protocols from neuroscientific studies. Moreover, we use the model to train spiking neural networks on MNIST digit recognition to show and explain what sort of mechanisms are needed to learn real-world patterns. We show how our model is sensitive to correlated spiking activity and how this enables it to modulate the learning rate of the network without altering the mean firing rate of the neurons nor the hyparameters of the learning rule. To the best of our knowledge, this is the first work that showcases how spike timing and rate can be complementary in their role of shaping the connectivity of spiking neural networks. 

**Abstract (ZH)**: 通过局部可塑性机制理解生物神经网络的形成可以导致高效的自适应信息处理系统，这有望缓解边缘计算系统中的部分瓶颈。尽管生物学利用尖峰来无缝结合尖峰时序和平均放电率来调节突触强度，大多数模型仅聚焦于其中之一。在本工作中，我们提出了一种Hebbian局部学习规则，将突触修改建模为神经活动钙踪迹的函数。我们展示该规则如何再现神经科学实验中尖峰时序和尖峰速率协议的结果。此外，我们使用该模型在MNIST数字识别任务上训练尖峰神经网络，以展示和解释需要哪些机制来学习真实世界的模式。我们展示了我们的模型对尖峰活动的相关性敏感性，并说明这种敏感性如何使网络能够在不改变神经元的平均放电率或学习规则的超参数的情况下调节学习率。据我们所知，这是首例展示尖峰时序和速率在塑造尖峰神经网络连接性方面互补作用的工作。 

---
# Zero-Shot Image-Based Large Language Model Approach to Road Pavement Monitoring 

**Title (ZH)**: 基于零样本图像的大型语言模型道路铺装监测方法 

**Authors**: Shuoshuo Xu, Kai Zhao, James Loney, Zili Li, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06785)  

**Abstract**: Effective and rapid evaluation of pavement surface condition is critical for prioritizing maintenance, ensuring transportation safety, and minimizing vehicle wear and tear. While conventional manual inspections suffer from subjectivity, existing machine learning-based methods are constrained by their reliance on large and high-quality labeled datasets, which require significant resources and limit adaptability across varied road conditions. The revolutionary advancements in Large Language Models (LLMs) present significant potential for overcoming these challenges. In this study, we propose an innovative automated zero-shot learning approach that leverages the image recognition and natural language understanding capabilities of LLMs to assess road conditions effectively. Multiple LLM-based assessment models were developed, employing prompt engineering strategies aligned with the Pavement Surface Condition Index (PSCI) standards. These models' accuracy and reliability were evaluated against official PSCI results, with an optimized model ultimately selected. Extensive tests benchmarked the optimized model against evaluations from various levels experts using Google Street View road images. The results reveal that the LLM-based approach can effectively assess road conditions, with the optimized model -employing comprehensive and structured prompt engineering strategies -outperforming simpler configurations by achieving high accuracy and consistency, even surpassing expert evaluations. Moreover, successfully applying the optimized model to Google Street View images demonstrates its potential for future city-scale deployments. These findings highlight the transformative potential of LLMs in automating road damage evaluations and underscore the pivotal role of detailed prompt engineering in achieving reliable assessments. 

**Abstract (ZH)**: 基于大语言模型的零样本学习路面状况评估方法研究 

---
# AI, Help Me Think$\unicode{x2014}$but for Myself: Assisting People in Complex Decision-Making by Providing Different Kinds of Cognitive Support 

**Title (ZH)**: AI，帮助我思考——但仅限于我自己：通过提供不同类型的认知支持来协助人们进行复杂决策 

**Authors**: Leon Reicherts, Zelun Tony Zhang, Elisabeth von Oswald, Yuanting Liu, Yvonne Rogers, Mariam Hassib  

**Link**: [PDF](https://arxiv.org/pdf/2504.06771)  

**Abstract**: How can we design AI tools that effectively support human decision-making by complementing and enhancing users' reasoning processes? Common recommendation-centric approaches face challenges such as inappropriate reliance or a lack of integration with users' decision-making processes. Here, we explore an alternative interaction model in which the AI outputs build upon users' own decision-making rationales. We compare this approach, which we call ExtendAI, with a recommendation-based AI. Participants in our mixed-methods user study interacted with both AIs as part of an investment decision-making task. We found that the AIs had different impacts, with ExtendAI integrating better into the decision-making process and people's own thinking and leading to slightly better outcomes. RecommendAI was able to provide more novel insights while requiring less cognitive effort. We discuss the implications of these and other findings along with three tensions of AI-assisted decision-making which our study revealed. 

**Abstract (ZH)**: 如何设计能够通过补充和增强用户推理过程来有效支持人类决策的AI工具？常见的基于推荐的方法面临与用户决策过程不适当的依赖或缺乏整合等问题。在这里，我们探索了一种替代的交互模型，在这种模型中，AI的输出建立在用户自身决策推理的基础上。我们将这种方法称为ExtendAI，并将其与基于推荐的AI进行了比较。在包含投资决策任务的混合方法用户研究中，参与者与两种AI进行了交互。我们发现，这两种AI产生了不同的影响，ExtendAI更好地融入了决策过程和人们的思考，并导致了稍好的结果。RecommendAI能够提供更具新颖性的见解，同时需要较少的认知努力。我们讨论了这些和其他发现的含义，以及我们在研究中揭示的AI辅助决策的三种张力。 

---
# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception 

**Title (ZH)**: 检测所有类型深度伪造音频：小波提示调整以增强听觉感知 

**Authors**: Yuankun Xie, Ruibo Fu, Zhiyong Wang, Xiaopeng Wang, Songjun Cao, Long Ma, Haonan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.06753)  

**Abstract**: The rapid advancement of audio generation technologies has escalated the risks of malicious deepfake audio across speech, sound, singing voice, and music, threatening multimedia security and trust. While existing countermeasures (CMs) perform well in single-type audio deepfake detection (ADD), their performance declines in cross-type scenarios. This paper is dedicated to studying the alltype ADD task. We are the first to comprehensively establish an all-type ADD benchmark to evaluate current CMs, incorporating cross-type deepfake detection across speech, sound, singing voice, and music. Then, we introduce the prompt tuning self-supervised learning (PT-SSL) training paradigm, which optimizes SSL frontend by learning specialized prompt tokens for ADD, requiring 458x fewer trainable parameters than fine-tuning (FT). Considering the auditory perception of different audio types,we propose the wavelet prompt tuning (WPT)-SSL method to capture type-invariant auditory deepfake information from the frequency domain without requiring additional training parameters, thereby enhancing performance over FT in the all-type ADD task. To achieve an universally CM, we utilize all types of deepfake audio for co-training. Experimental results demonstrate that WPT-XLSR-AASIST achieved the best performance, with an average EER of 3.58% across all evaluation sets. The code is available online. 

**Abstract (ZH)**: 快速发展的音频生成技术加剧了语音、声音、歌声和音乐中恶意深度假音的风险，威胁多媒体安全与信任。现有对抗措施在单一类型音频深度假音检测方面表现良好，但在跨类型场景中性能下降。本文致力于研究跨类型音频深度假音检测任务。我们首次全面建立了跨类型音频深度假音检测基准，用于评估当前的对抗措施，涵盖了语音、声音、歌声和音乐跨类型的深度假音检测。然后，我们引入了提示调谐半监督学习（PT-SSL）训练范式，通过学习专门的提示标记优化半监督学习前端，所需可训练参数仅为微调的1/458。考虑到不同音频类型的声音感知，我们提出了小波提示调谐（WPT）-半监督学习方法，能够在频率域中捕获类型不变的声音深度假音信息，而无需额外训练参数，从而在跨类型音频深度假音检测任务中优于微调。为了构建通用对抗措施，我们利用所有类型的真实深度假音音频进行协同训练。实验结果表明，WPT-XLSR-AASIST在所有评估集上的平均错误检测率EER达到了3.58%，代码已在线提供。 

---
# EDIT: Enhancing Vision Transformers by Mitigating Attention Sink through an Encoder-Decoder Architecture 

**Title (ZH)**: 改进视觉变换器通过编码器-解码器架构缓解注意力陷阱 

**Authors**: Wenfeng Feng, Guoying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.06738)  

**Abstract**: In this paper, we propose EDIT (Encoder-Decoder Image Transformer), a novel architecture designed to mitigate the attention sink phenomenon observed in Vision Transformer models. Attention sink occurs when an excessive amount of attention is allocated to the [CLS] token, distorting the model's ability to effectively process image patches. To address this, we introduce a layer-aligned encoder-decoder architecture, where the encoder utilizes self-attention to process image patches, while the decoder uses cross-attention to focus on the [CLS] token. Unlike traditional encoder-decoder framework, where the decoder depends solely on high-level encoder representations, EDIT allows the decoder to extract information starting from low-level features, progressively refining the representation layer by layer. EDIT is naturally interpretable demonstrated through sequential attention maps, illustrating the refined, layer-by-layer focus on key image features. Experiments on ImageNet-1k and ImageNet-21k, along with transfer learning tasks, show that EDIT achieves consistent performance improvements over DeiT3 models. These results highlight the effectiveness of EDIT's design in addressing attention sink and improving visual feature extraction. 

**Abstract (ZH)**: 本文提出EDIT（编码器-解码器图像变换器）架构，旨在缓解视觉变换器模型中观察到的注意力陷进现象。注意力陷进发生在过多的注意力集中在[CLS]标记上，从而影响模型有效处理图像 patches 的能力。为了解决这一问题，我们引入了一种层对齐的编码器-解码器架构，其中编码器利用自我注意来处理图像 patches，而解码器则使用交叉注意来聚焦于[CLS]标记。与传统的编码器-解码器框架不同，传统的解码器仅依赖高层编码器表示，而EDIT允许解码器从低层特征开始提取信息，并逐层细化表示。通过序贯注意力图展示了EDIT的自然可解释性，表明其逐层聚焦于关键图像特征。在ImageNet-1k和ImageNet-21k上的实验以及迁移学习任务中，EDIT在DeiT3模型上展示了一致的性能提升，这些结果突显了EDIT设计在解决注意力陷进和提高视觉特征提取方面的有效性。 

---
# Learning global control of underactuated systems with Model-Based Reinforcement Learning 

**Title (ZH)**: 基于模型的强化学习在欠驱动系统全局控制中的应用 

**Authors**: Niccolò Turcato, Marco Calì, Alberto Dalla Libera, Giulio Giacomuzzo, Ruggero Carli, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2504.06721)  

**Abstract**: This short paper describes our proposed solution for the third edition of the "AI Olympics with RealAIGym" competition, held at ICRA 2025. We employed Monte-Carlo Probabilistic Inference for Learning Control (MC-PILCO), an MBRL algorithm recognized for its exceptional data efficiency across various low-dimensional robotic tasks, including cart-pole, ball \& plate, and Furuta pendulum systems. MC-PILCO optimizes a system dynamics model using interaction data, enabling policy refinement through simulation rather than direct system data optimization. This approach has proven highly effective in physical systems, offering greater data efficiency than Model-Free (MF) alternatives. Notably, MC-PILCO has previously won the first two editions of this competition, demonstrating its robustness in both simulated and real-world environments. Besides briefly reviewing the algorithm, we discuss the most critical aspects of the MC-PILCO implementation in the tasks at hand: learning a global policy for the pendubot and acrobot systems. 

**Abstract (ZH)**: 本短文描述了我们为2025年ICRA会议举办的“RealAIGym AI奥运会”第三版比赛提出的一个解决方案。我们采用了蒙特卡洛概率推断学习控制（MC-PILCO）算法，这是一种在各种低维度机器人任务中（包括cart-pole、ball & plate和Furuta摆动系统）以出色的数据效率著称的模型引导的 reinforcement 学习算法。MC-PILCO利用交互数据优化系统动力学模型，通过仿真优化策略，而非直接优化系统数据。该方法在物理系统中证明非常有效，比无模型（Model-Free）方法更具数据效率。值得注意的是，MC-PILCO曾在前两届比赛中获胜，展示了其在模拟和实际环境中的鲁棒性。除了简要回顾该算法之外，我们还讨论了在当前任务中MC-PILCO实现的关键方面：pendubot和acrobot系统的全局策略学习。 

---
# Masked Scene Modeling: Narrowing the Gap Between Supervised and Self-Supervised Learning in 3D Scene Understanding 

**Title (ZH)**: 掩码场景建模：在3D场景理解中缩小监督学习与自我监督学习的差距 

**Authors**: Pedro Hermosilla, Christian Stippel, Leon Sick  

**Link**: [PDF](https://arxiv.org/pdf/2504.06719)  

**Abstract**: Self-supervised learning has transformed 2D computer vision by enabling models trained on large, unannotated datasets to provide versatile off-the-shelf features that perform similarly to models trained with labels. However, in 3D scene understanding, self-supervised methods are typically only used as a weight initialization step for task-specific fine-tuning, limiting their utility for general-purpose feature extraction. This paper addresses this shortcoming by proposing a robust evaluation protocol specifically designed to assess the quality of self-supervised features for 3D scene understanding. Our protocol uses multi-resolution feature sampling of hierarchical models to create rich point-level representations that capture the semantic capabilities of the model and, hence, are suitable for evaluation with linear probing and nearest-neighbor methods. Furthermore, we introduce the first self-supervised model that performs similarly to supervised models when only off-the-shelf features are used in a linear probing setup. In particular, our model is trained natively in 3D with a novel self-supervised approach based on a Masked Scene Modeling objective, which reconstructs deep features of masked patches in a bottom-up manner and is specifically tailored to hierarchical 3D models. Our experiments not only demonstrate that our method achieves competitive performance to supervised models, but also surpasses existing self-supervised approaches by a large margin. The model and training code can be found at our Github repository (this https URL). 

**Abstract (ZH)**: 自监督学习已通过使在大规模无标注数据集上训练的模型能够提供与标注训练模型性能相当的多功能即用型特征，而重塑了2D计算机视觉。然而，在3D场景理解中，自监督方法通常仅用作任务特定微调的权重初始化步骤，限制了其作为通用特征提取工具的应用。本文通过提出一种专门用于评估自监督特征质量的稳健评估协议来解决这一问题。该协议利用分层模型的多分辨率特征采样来创建丰富的点级表示，这些表示捕获了模型的语义能力，并因此适用于使用线性探针和最近邻方法进行评估。此外，我们引入了第一个在仅使用即用型特征的线性探针设置中与监督模型表现相当的自监督模型。特别是，我们的模型使用一种新颖的自监督方法在3D空间中进行本征训练，该方法基于遮罩场景建模目标，以自底向上方式重建遮罩补丁的深层特征，并特别针对分层3D模型。我们的实验不仅证明了我们的方法达到了与监督模型相当的性能，而且在很大程度上超过了现有的自监督方法。模型和训练代码可在我们的Github仓库中找到（this https URL）。 

---
# Hyperparameter Optimisation with Practical Interpretability and Explanation Methods in Probabilistic Curriculum Learning 

**Title (ZH)**: 基于概率性课程学习的超参数优化及其实用可解释性方法的研究 

**Authors**: Llewyn Salt, Marcus Gallagher  

**Link**: [PDF](https://arxiv.org/pdf/2504.06683)  

**Abstract**: Hyperparameter optimisation (HPO) is crucial for achieving strong performance in reinforcement learning (RL), as RL algorithms are inherently sensitive to hyperparameter settings. Probabilistic Curriculum Learning (PCL) is a curriculum learning strategy designed to improve RL performance by structuring the agent's learning process, yet effective hyperparameter tuning remains challenging and computationally demanding. In this paper, we provide an empirical analysis of hyperparameter interactions and their effects on the performance of a PCL algorithm within standard RL tasks, including point-maze navigation and DC motor control. Using the AlgOS framework integrated with Optuna's Tree-Structured Parzen Estimator (TPE), we present strategies to refine hyperparameter search spaces, enhancing optimisation efficiency. Additionally, we introduce a novel SHAP-based interpretability approach tailored specifically for analysing hyperparameter impacts, offering clear insights into how individual hyperparameters and their interactions influence RL performance. Our work contributes practical guidelines and interpretability tools that significantly improve the effectiveness and computational feasibility of hyperparameter optimisation in reinforcement learning. 

**Abstract (ZH)**: 超参数优化（HPO）在强化学习（RL）中对于实现优异性能至关重要，因为RL算法对超参数设置具有固有的敏感性。概率性 curriculum 学习（PCL）是一种旨在通过结构化智能体的学习过程来提高RL性能的curriculum学习策略，但有效的超参数调整仍然是一个具有挑战性且计算成本高昂的问题。在本文中，我们对PCL算法在标准RL任务（如点迷宫导航和DC电机控制）中超参数交互作用及其对性能的影响进行了 empirical 分析。通过将AlgOS框架与Optuna的树结构帕兹内斯特imator（TPE）结合，我们提出了一种策略来细化超参数搜索空间，从而提高优化效率。此外，我们引入了一种基于SHAP的新型可解释性方法，专门用于分析超参数影响，提供关于如何通过单独的超参数及其交互作用影响RL性能的清晰见解。我们的工作为强化学习中的超参数优化提供了实用指南和可解释性工具，显著提高了超参数优化的有效性和计算可行性。 

---
# NLP Security and Ethics, in the Wild 

**Title (ZH)**: NLP安全与伦理：在现实世界中的应用 

**Authors**: Heather Lent, Erick Galinkin, Yiyi Chen, Jens Myrup Pedersen, Leon Derczynski, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2504.06669)  

**Abstract**: As NLP models are used by a growing number of end-users, an area of increasing importance is NLP Security (NLPSec): assessing the vulnerability of models to malicious attacks and developing comprehensive countermeasures against them. While work at the intersection of NLP and cybersecurity has the potential to create safer NLP for all, accidental oversights can result in tangible harm (e.g., breaches of privacy or proliferation of malicious models). In this emerging field, however, the research ethics of NLP have not yet faced many of the long-standing conundrums pertinent to cybersecurity, until now. We thus examine contemporary works across NLPSec, and explore their engagement with cybersecurity's ethical norms. We identify trends across the literature, ultimately finding alarming gaps on topics like harm minimization and responsible disclosure. To alleviate these concerns, we provide concrete recommendations to help NLP researchers navigate this space more ethically, bridging the gap between traditional cybersecurity and NLP ethics, which we frame as ``white hat NLP''. The goal of this work is to help cultivate an intentional culture of ethical research for those working in NLP Security. 

**Abstract (ZH)**: 随着自然语言处理模型被越来越多的终端用户使用，自然语言处理安全（NLPSec）领域的重要性不断增加：评估模型对恶意攻击的脆弱性并开发全面的应对措施。虽然自然语言处理与网络安全的交叉研究有望为所有人创造更安全的自然语言处理，但偶然的疏忽可能会造成实际损害（如隐私泄露或恶意模型的传播）。然而，在这一新兴领域中，自然语言处理的科研伦理尚未面临与网络安全相关的许多长期难题，直到现在。因此，我们考察了NLPSec领域的当代研究成果，并探讨了它们在网络安全伦理规范方面的实践情况。我们识别出文献中的趋势，最终发现有关最小化损害和负责任披露等话题存在令人担忧的空白。为缓解这些担忧，我们提供了具体的建议，以帮助自然语言处理研究人员更伦理地导航这一领域，弥合传统网络安全与自然语言处理伦理之间的差距，我们将这一框架称为“白帽自然语言处理”。本工作的目标是帮助培养自然语言处理安全领域中一种有意识的伦理研究文化。 

---
# Bridging the Gap Between Preference Alignment and Machine Unlearning 

**Title (ZH)**: 偏好对齐与机器遗忘之间的差距桥梁 

**Authors**: Xiaohua Feng, Yuyuan Li, Huwei Ji, Jiaming Zhang, Li Zhang, Tianyu Du, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06659)  

**Abstract**: Despite advances in Preference Alignment (PA) for Large Language Models (LLMs), mainstream methods like Reinforcement Learning with Human Feedback (RLHF) face notable challenges. These approaches require high-quality datasets of positive preference examples, which are costly to obtain and computationally intensive due to training instability, limiting their use in low-resource scenarios. LLM unlearning technique presents a promising alternative, by directly removing the influence of negative examples. However, current research has primarily focused on empirical validation, lacking systematic quantitative analysis. To bridge this gap, we propose a framework to explore the relationship between PA and LLM unlearning. Specifically, we introduce a bi-level optimization-based method to quantify the impact of unlearning specific negative examples on PA performance. Our analysis reveals that not all negative examples contribute equally to alignment improvement when unlearned, and the effect varies significantly across examples. Building on this insight, we pose a crucial question: how can we optimally select and weight negative examples for unlearning to maximize PA performance? To answer this, we propose a framework called Unlearning to Align (U2A), which leverages bi-level optimization to efficiently select and unlearn examples for optimal PA performance. We validate the proposed method through extensive experiments, with results confirming its effectiveness. 

**Abstract (ZH)**: 尽管在大型语言模型（LLMs）的偏好对齐（PA）方面取得了进展，主流方法如强化学习带人类反馈（RLHF）仍面临显著挑战。这些方法需要高质量的正偏好示例数据集，获取成本高且由于训练不稳定而计算密集，限制了其在低资源场景中的应用。LLM去学习技术为一种有前景的替代方案，可以直接去除负面示例的影响。然而，当前研究主要集中在实证验证上，缺乏系统的定量分析。为弥合这一差距，我们提出了一种框架来探讨偏好对齐与LLM去学习之间的关系。具体而言，我们引入了一种基于双层优化的方法来量化移除特定负面示例对偏好对齐性能的影响。我们的分析表明，并非所有负面示例在去除时对对齐改进的贡献都是均等的，且不同示例的效果差异显著。基于这一洞见，我们提出了一个关键问题：如何通过优化选择和加权负面示例来最大化偏好对齐性能？为回答这一问题，我们提出了一种名为U2A（Unlearning to Align）的框架，利用双层优化有效地选择和去除示例以实现最优的偏好对齐性能。我们通过广泛的实验验证了所提出的方法，结果证实了其有效性。 

---
# A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty 

**Title (ZH)**: 大型语言模型中基于样本层面遗忘难度的神经启发式遗忘解释 

**Authors**: Xiaohua Feng, Yuyuan Li, Chengye Wang, Junlin Liu, Li Zhang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06658)  

**Abstract**: Driven by privacy protection laws and regulations, unlearning in Large Language Models (LLMs) is gaining increasing attention. However, current research often neglects the interpretability of the unlearning process, particularly concerning sample-level unlearning difficulty. Existing studies typically assume a uniform unlearning difficulty across samples. This simplification risks attributing the performance of unlearning algorithms to sample selection rather than the algorithm's design, potentially steering the development of LLM unlearning in the wrong direction. Thus, we investigate the relationship between LLM unlearning and sample characteristics, with a focus on unlearning difficulty. Drawing inspiration from neuroscience, we propose a Memory Removal Difficulty ($\mathrm{MRD}$) metric to quantify sample-level unlearning difficulty. Using $\mathrm{MRD}$, we analyze the characteristics of hard-to-unlearn versus easy-to-unlearn samples. Furthermore, we propose an $\mathrm{MRD}$-based weighted sampling method to optimize existing unlearning algorithms, which prioritizes easily forgettable samples, thereby improving unlearning efficiency and effectiveness. We validate the proposed metric and method using public benchmarks and datasets, with results confirming its effectiveness. 

**Abstract (ZH)**: 受隐私保护法律法规驱动，大规模语言模型的去学习问题正逐渐引起关注。然而，当前研究往往忽视了去学习过程的可解释性，特别是针对样本级别的去学习难度。现有研究通常假设样本的去学习难度一致。这一简化可能导致将去学习算法的性能归因于样本选择，而非算法设计，从而可能误导大规模语言模型去学习的发展方向。因此，我们探讨了大规模语言模型的去学习与其样本特征之间的关系，重点关注去学习难度。受神经科学的启发，我们提出了一种记忆移除难度（$\mathrm{MRD}$）度量标准来量化样本级别的去学习难度。利用$\mathrm{MRD}$，我们分析了难以去学习的样本与容易去学习的样本的特征。此外，我们提出了一种基于$\mathrm{MRD}$的加权采样方法，以优化现有的去学习算法，优先考虑容易忘记的样本，从而提高去学习的效率和效果。我们使用公开的基准和数据集验证了所提出的度量标准和方法，结果证实了其有效性。 

---
# GRAIN: Multi-Granular and Implicit Information Aggregation Graph Neural Network for Heterophilous Graphs 

**Title (ZH)**: GRAIN：面向异构图的多粒度和隐式信息聚合图神经网络 

**Authors**: Songwei Zhao, Yuan Jiang, Zijing Zhang, Yang Yu, Hechang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06649)  

**Abstract**: Graph neural networks (GNNs) have shown significant success in learning graph representations. However, recent studies reveal that GNNs often fail to outperform simple MLPs on heterophilous graph tasks, where connected nodes may differ in features or labels, challenging the homophily assumption. Existing methods addressing this issue often overlook the importance of information granularity and rarely consider implicit relationships between distant nodes. To overcome these limitations, we propose the Granular and Implicit Graph Network (GRAIN), a novel GNN model specifically designed for heterophilous graphs. GRAIN enhances node embeddings by aggregating multi-view information at various granularity levels and incorporating implicit data from distant, non-neighboring nodes. This approach effectively integrates local and global information, resulting in smoother, more accurate node representations. We also introduce an adaptive graph information aggregator that efficiently combines multi-granularity and implicit data, significantly improving node representation quality, as shown by experiments on 13 datasets covering varying homophily and heterophily. GRAIN consistently outperforms 12 state-of-the-art models, excelling on both homophilous and heterophilous graphs. 

**Abstract (ZH)**: 粒度化和隐含图网络：一种面向异构图的新颖GNN模型 

---
# AMAD: AutoMasked Attention for Unsupervised Multivariate Time Series Anomaly Detection 

**Title (ZH)**: AMAD: AutoMasked注意力机制在无监督多变量时间序列异常检测中的应用 

**Authors**: Tiange Huang, Yongjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06643)  

**Abstract**: Unsupervised multivariate time series anomaly detection (UMTSAD) plays a critical role in various domains, including finance, networks, and sensor systems. In recent years, due to the outstanding performance of deep learning in general sequential tasks, many models have been specialized for deep UMTSAD tasks and have achieved impressive results, particularly those based on the Transformer and self-attention mechanisms. However, the sequence anomaly association assumptions underlying these models are often limited to specific predefined patterns and scenarios, such as concentrated or peak anomaly patterns. These limitations hinder their ability to generalize to diverse anomaly situations, especially where the lack of labels poses significant challenges. To address these issues, we propose AMAD, which integrates \textbf{A}uto\textbf{M}asked Attention for UMTS\textbf{AD} scenarios. AMAD introduces a novel structure based on the AutoMask mechanism and an attention mixup module, forming a simple yet generalized anomaly association representation framework. This framework is further enhanced by a Max-Min training strategy and a Local-Global contrastive learning approach. By combining multi-scale feature extraction with automatic relative association modeling, AMAD provides a robust and adaptable solution to UMTSAD challenges. Extensive experimental results demonstrate that the proposed model achieving competitive performance results compared to SOTA benchmarks across a variety of datasets. 

**Abstract (ZH)**: 无监督多变量时间序列异常检测（UMTSAD）在金融、网络和传感器系统等多个领域发挥着关键作用。近年来，由于深度学习在通用序列任务中的出色性能，许多模型专门用于深度UMTSAD任务并取得了显著成果，特别是基于Transformer和自注意力机制的模型。然而，这些模型底层的序列异常关联假设往往局限于特定预定义的模式和场景，如集中的或峰值异常模式。这些限制阻碍了它们在多样异常情况下的泛化能力，尤其是在缺乏标签的情况下构成重大挑战。为解决这些问题，我们提出AMAD，即基于自动掩蔽注意机制的UMTSAD场景。AMAD引入了一种基于AutoMask机制和注意力mixup模块的新结构，形成了一个简单而通用的异常关联表示框架。该框架进一步通过最大值-最小值训练策略和局部-全局对比学习方法得到增强。通过结合多尺度特征提取和自动相对关联建模，AMAD提供了一种鲁棒且适应性强的UMTSAD解决方案。 extensive实验结果表明，所提出模型在各种数据集上的性能与现有最佳基准相当。 

---
# Wanting to be Understood 

**Title (ZH)**: 想要被理解 

**Authors**: Chrisantha Fernando, Dylan Banarse, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2504.06611)  

**Abstract**: This paper explores an intrinsic motivation for mutual awareness, hypothesizing that humans possess a fundamental drive to understand \textit{and to be understood} even in the absence of extrinsic rewards. Through simulations of the perceptual crossing paradigm, we explore the effect of various internal reward functions in reinforcement learning agents. The drive to understand is implemented as an active inference type artificial curiosity reward, whereas the drive to be understood is implemented through intrinsic rewards for imitation, influence/impressionability, and sub-reaction time anticipation of the other. Results indicate that while artificial curiosity alone does not lead to a preference for social interaction, rewards emphasizing reciprocal understanding successfully drive agents to prioritize interaction. We demonstrate that this intrinsic motivation can facilitate cooperation in tasks where only one agent receives extrinsic reward for the behaviour of the other. 

**Abstract (ZH)**: 本文探索了内在动机的互惠意识，假设人类即使在缺乏外在奖励的情况下，也具有一种基本驱动力，即理解他人并渴望被他人理解。通过感知交叉范式的模拟，我们探索了强化学习代理内部奖励函数的影响。理解的驱动力通过一种主动推断类型的 artificial curiosity 奖励实现，而被理解的驱动力通过模仿、影响/可影响性和对另一方亚反应时间的预期实现。结果表明，虽然单独的人工curiosity未能导致对社会互动的偏好，但强调相互理解的奖励成功地驱动代理优先进行互动。我们证明了这种内在动机可以促进在只有代理之一因其行为而获得外在奖励的任务中实现合作。 

---
# InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features 

**Title (ZH)**: InteractRank：基于交叉交互特征的个性化大规模网页预排序 

**Authors**: Sujay Khandagale, Bhawna Juneja, Prabhat Agarwal, Aditya Subramanian, Jaewon Yang, Yuting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06609)  

**Abstract**: Modern search systems use a multi-stage architecture to deliver personalized results efficiently. Key stages include retrieval, pre-ranking, full ranking, and blending, which refine billions of items to top selections. The pre-ranking stage, vital for scoring and filtering hundreds of thousands of items down to a few thousand, typically relies on two tower models due to their computational efficiency, despite often lacking in capturing complex interactions. While query-item cross interaction features are paramount for full ranking, integrating them into pre-ranking models presents efficiency-related challenges. In this paper, we introduce InteractRank, a novel two tower pre-ranking model with robust cross interaction features used at Pinterest. By incorporating historical user engagement-based query-item interactions in the scoring function along with the two tower dot product, InteractRank significantly boosts pre-ranking performance with minimal latency and computation costs. In real-world A/B experiments at Pinterest, InteractRank improves the online engagement metric by 6.5% over a BM25 baseline and by 3.7% over a vanilla two tower baseline. We also highlight other components of InteractRank, like real-time user-sequence modeling, and analyze their contributions through offline ablation studies. The code for InteractRank is available at this https URL. 

**Abstract (ZH)**: 现代搜索引擎使用多阶段架构高效交付个性化结果。关键阶段包括检索、预排名、全面排名和混合，这些阶段将数十亿项内容精炼为顶级选择。预排名阶段对于通过对数以万计项的评分和过滤至数千项至关重要，尽管计算效率高，但往往难以捕捉复杂的交互。虽然查询-项交叉交互特征对于全面排名至关重要，但将其集成到预排名模型中会带来效率方面的挑战。在本文中，我们提出了InteractRank，这是一种新颖的两塔预排名模型，广泛应用于Pinterest，其中包含强大的交叉交互特征。通过在评分函数中结合基于历史用户参与度的查询-项交互以及两塔点积，InteractRank在几乎没有延迟和计算成本的情况下显著提升了预排名性能。在Pinterest的实际在线A/B实验中，InteractRank相较于BM25基线提高了6.5%的在线参与度指标，相较于纯两塔基线提高了3.7%。我们还强调了InteractRank的其他组件，如实时用户序列建模，并通过离线消融研究分析了它们的贡献。InteractRank的代码可在以下链接获取：this https URL。 

---
# Automated Business Process Analysis: An LLM-Based Approach to Value Assessment 

**Title (ZH)**: 基于大语言模型的自动化业务流程分析：价值评估方法 

**Authors**: William De Michele, Abel Armas Cervantes, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06600)  

**Abstract**: Business processes are fundamental to organizational operations, yet their optimization remains challenging due to the timeconsuming nature of manual process analysis. Our paper harnesses Large Language Models (LLMs) to automate value-added analysis, a qualitative process analysis technique that aims to identify steps in the process that do not deliver value. To date, this technique is predominantly manual, time-consuming, and subjective. Our method offers a more principled approach which operates in two phases: first, decomposing high-level activities into detailed steps to enable granular analysis, and second, performing a value-added analysis to classify each step according to Lean principles. This approach enables systematic identification of waste while maintaining the semantic understanding necessary for qualitative analysis. We develop our approach using 50 business process models, for which we collect and publish manual ground-truth labels. Our evaluation, comparing zero-shot baselines with more structured prompts reveals (a) a consistent benefit of structured prompting and (b) promising performance for both tasks. We discuss the potential for LLMs to augment human expertise in qualitative process analysis while reducing the time and subjectivity inherent in manual approaches. 

**Abstract (ZH)**: 利用大型语言模型自动进行增值分析以优化业务流程：一种基于精益原则的两阶段方法 

---
# Exploring Ordinal Bias in Action Recognition for Instructional Videos 

**Title (ZH)**: 探索指令视频中动作识别的序数偏见 

**Authors**: Joochan Kim, Minjoon Jung, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06580)  

**Abstract**: Action recognition models have achieved promising results in understanding instructional videos. However, they often rely on dominant, dataset-specific action sequences rather than true video comprehension, a problem that we define as ordinal bias. To address this issue, we propose two effective video manipulation methods: Action Masking, which masks frames of frequently co-occurring actions, and Sequence Shuffling, which randomizes the order of action segments. Through comprehensive experiments, we demonstrate that current models exhibit significant performance drops when confronted with nonstandard action sequences, underscoring their vulnerability to ordinal bias. Our findings emphasize the importance of rethinking evaluation strategies and developing models capable of generalizing beyond fixed action patterns in diverse instructional videos. 

**Abstract (ZH)**: 动作识别模型在理解指导视频方面取得了令人瞩目的成果，但它们往往依赖于特定数据集的主导动作序列，而不是真正的视频理解。为解决这一问题，我们提出了两种有效的视频操作方法：动作遮蔽，即遮蔽频繁共现的动作帧；序列打乱，即随机化动作片段顺序。通过全面的实验，我们证明了当前模型在面对非标准动作序列时表现出显著的性能下降，突显了它们对序列表现偏差的脆弱性。我们的研究强调了重新思考评估策略并开发能够超越固定动作模式泛化的模型的重要性。 

---
# Attributes-aware Visual Emotion Representation Learning 

**Title (ZH)**: 属性aware的视觉情绪表示学习 

**Authors**: Rahul Singh Maharjan, Marta Romeo, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06578)  

**Abstract**: Visual emotion analysis or recognition has gained considerable attention due to the growing interest in understanding how images can convey rich semantics and evoke emotions in human perception. However, visual emotion analysis poses distinctive challenges compared to traditional vision tasks, especially due to the intricate relationship between general visual features and the different affective states they evoke, known as the affective gap. Researchers have used deep representation learning methods to address this challenge of extracting generalized features from entire images. However, most existing methods overlook the importance of specific emotional attributes such as brightness, colorfulness, scene understanding, and facial expressions. Through this paper, we introduce A4Net, a deep representation network to bridge the affective gap by leveraging four key attributes: brightness (Attribute 1), colorfulness (Attribute 2), scene context (Attribute 3), and facial expressions (Attribute 4). By fusing and jointly training all aspects of attribute recognition and visual emotion analysis, A4Net aims to provide a better insight into emotional content in images. Experimental results show the effectiveness of A4Net, showcasing competitive performance compared to state-of-the-art methods across diverse visual emotion datasets. Furthermore, visualizations of activation maps generated by A4Net offer insights into its ability to generalize across different visual emotion datasets. 

**Abstract (ZH)**: 视觉情感分析或识别由于人们日益关注理解图像如何传达丰富语义并引发人类情感而得到了广泛关注。然而，视觉情感分析相对于传统视觉任务来说面临着独特的挑战，尤其是在一般视觉特征与它们所唤起的不同情感状态之间复杂的关系，即情感差距方面。研究人员利用深度表示学习方法来应对从整幅图像中提取通用特征的挑战。然而，现有的大多数方法忽略了亮度、色彩丰富度、场景理解以及面部表情等特定情感属性的重要性。通过本文，我们介绍了A4Net，一种深度表示网络，通过利用四个关键属性（亮度、色彩丰富度、场景上下文以及面部表情）来弥合情感差距。通过融合和联合训练所有属性识别和视觉情感分析的各个方面，A4Net旨在提供对图像中情感内容的更好理解。实验结果表明了A4Net的有效性，在多种视觉情感数据集中展示了与其相比具有竞争力的性能。此外，A4Net生成的激活图可视化展示了其在不同视觉情感数据集上的泛化能力。 

---
# Societal Impacts Research Requires Benchmarks for Creative Composition Tasks 

**Title (ZH)**: 社会影响研究需要创造性 compositions 任务的基准标准 

**Authors**: Judy Hanwen Shen, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06549)  

**Abstract**: Foundation models that are capable of automating cognitive tasks represent a pivotal technological shift, yet their societal implications remain unclear. These systems promise exciting advances, yet they also risk flooding our information ecosystem with formulaic, homogeneous, and potentially misleading synthetic content. Developing benchmarks grounded in real use cases where these risks are most significant is therefore critical. Through a thematic analysis using 2 million language model user prompts, we identify creative composition tasks as a prevalent usage category where users seek help with personal tasks that require everyday creativity. Our fine-grained analysis identifies mismatches between current benchmarks and usage patterns among these tasks. Crucially, we argue that the same use cases that currently lack thorough evaluations can lead to negative downstream impacts. This position paper argues that benchmarks focused on creative composition tasks is a necessary step towards understanding the societal harms of AI-generated content. We call for greater transparency in usage patterns to inform the development of new benchmarks that can effectively measure both the progress and the impacts of models with creative capabilities. 

**Abstract (ZH)**: 能够自动化认知任务的基石模型代表了技术上的重要转折，然而它们的社会影响尚不明确。这些系统承诺带来激动人心的进步，但也有可能向我们的信息生态系统中注入公式化、同质化且可能误导性的合成内容。因此，在这些风险最突出的实际应用场景中开发基准至关重要。通过对200万自然语言模型用户提示进行主题分析，我们发现创意组合任务是用户寻求帮助的常见类别，这些任务需要日常的创造力。精细的分析发现当前基准与这些任务使用模式之间的不匹配。至关重要的是，我们认为当前缺乏充分评估的使用案例可能导致负面影响。本文认为，关注创意组合任务的基准是理解AI生成内容社会危害的必要步骤。我们呼吁提高使用模式的透明度，以指导开发新的基准，这些基准能够有效衡量具有创造力能力的模型的进展及其影响。 

---
# Polygon: Symbolic Reasoning for SQL using Conflict-Driven Under-Approximation Search 

**Title (ZH)**: Polygon: 基于冲突驱动的下近似搜索的符号推理用于SQL 

**Authors**: Pinhan Zhao, Yuepeng Wang, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06542)  

**Abstract**: We present a novel symbolic reasoning engine for SQL which can efficiently generate an input $I$ for $n$ queries $P_1, \cdots, P_n$, such that their outputs on $I$ satisfy a given property (expressed in SMT). This is useful in different contexts, such as disproving equivalence of two SQL queries and disambiguating a set of queries. Our first idea is to reason about an under-approximation of each $P_i$ -- that is, a subset of $P_i$'s input-output behaviors. While it makes our approach both semantics-aware and lightweight, this idea alone is incomplete (as a fixed under-approximation might miss some behaviors of interest). Therefore, our second idea is to perform search over an expressive family of under-approximations (which collectively cover all program behaviors of interest), thereby making our approach complete. We have implemented these ideas in a tool, Polygon, and evaluated it on over 30,000 benchmarks across two tasks (namely, SQL equivalence refutation and query disambiguation). Our evaluation results show that Polygon significantly outperforms all prior techniques. 

**Abstract (ZH)**: 一种新型SQL符号推理引擎及其在查询等价反驳和查询解析中的应用 

---
# OPAL: Encoding Causal Understanding of Physical Systems for Robot Learning 

**Title (ZH)**: OPAL: 编码物理系统因果理解的机器人学习方法 

**Authors**: Daniel Tcheurekdjian, Joshua Klasmeier, Tom Cooney, Christopher McCann, Tyler Fenstermaker  

**Link**: [PDF](https://arxiv.org/pdf/2504.06538)  

**Abstract**: We present OPAL (Operant Physical Agent with Language), a novel vision-language-action architecture that introduces topological constraints to flow matching for robotic control. To do so, we further introduce topological attention. Our approach models action sequences as topologically-structured representations with non-trivial constraints. Experimental results across 10 complex manipulation tasks demonstrate OPAL's superior performance compared to previous approaches, including Octo, OpenVLA, and ${\pi}$0.
Our architecture achieves significant improvements in zero-shot performance without requiring task-specific fine-tuning, while reducing inference computational requirements by 42%. The theoretical guarantees provided by our topological approach result in more coherent long-horizon action sequences. Our results highlight the potential of constraining the search space of learning problems in robotics by deriving from fundamental physical laws, and the possibility of using topological attention to embed causal understanding into transformer architectures. 

**Abstract (ZH)**: OPAL：一种引入拓扑约束的视知觉行动架构 

---
# Lugha-Llama: Adapting Large Language Models for African Languages 

**Title (ZH)**: Lugha-Llama：适应非洲语言的大规模语言模型 

**Authors**: Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2504.06536)  

**Abstract**: Large language models (LLMs) have achieved impressive results in a wide range of natural language applications. However, they often struggle to recognize low-resource languages, in particular African languages, which are not well represented in large training corpora. In this paper, we consider how to adapt LLMs to low-resource African languages. We find that combining curated data from African languages with high-quality English educational texts results in a training mix that substantially improves the model's performance on these languages. On the challenging IrokoBench dataset, our models consistently achieve the best performance amongst similarly sized baselines, particularly on knowledge-intensive multiple-choice questions (AfriMMLU). Additionally, on the cross-lingual question answering benchmark AfriQA, our models outperform the base model by over 10%. To better understand the role of English data during training, we translate a subset of 200M tokens into Swahili language and perform an analysis which reveals that the content of these data is primarily responsible for the strong performance. We release our models and data to encourage future research on African languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种自然语言应用中取得了令人瞩目的成果。然而，它们往往难以识别低资源语言，特别是非洲语言，这些语言在大规模训练语料库中代表性不足。本文探讨了如何将LLMs适应低资源非洲语言。我们发现，将非洲语言的精选数据与高质量的英语教育文本结合，形成了一种训练混合数据，显著提高了模型在这些语言上的表现。在具有挑战性的IrokoBench数据集上，我们的模型在相同规模的基础模型中始终取得最佳性能，特别是在知识密集型多项选择题（AfriMMLU）方面。此外，在跨语言问答基准AfriQA上，我们的模型比基础模型高出超过10%。为了更好地理解训练过程中英语数据的作用，我们将其部分2亿个词元翻译成斯瓦希里语，并进行了一项分析，结果表明这些数据的内容主要负责了这种强劲的表现。我们发布了我们的模型和数据，以鼓励对非洲语言未来的研究。 

---
# Flexible Graph Similarity Computation With A Proactive Optimization Strategy 

**Title (ZH)**: 基于主动优化策略的灵活图相似性计算 

**Authors**: Zhouyang Liu, Ning Liu, Yixin Chen, Jiezhong He, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06533)  

**Abstract**: Graph Edit Distance (GED) is an important similarity measure in graph retrieval, which quantifies the minimum cost of transforming one graph into another through edit operations, and offers flexibility by allowing customizable operation costs. Recent learning-based approaches approximate GEDs with the distances between representations in vector spaces. However, these methods often struggle with varying operation costs due to neglecting the impact of these costs on determining optimal graph mappings. Furthermore, they rely on isolated node distances as guidance, necessitating inefficient reactive refinements of mappings. To address these issues, we propose Graph Edit Network (GEN), a novel learning-based approach for flexible GED computation. By identifying the limitations of existing methods in capturing flexibility of GED, we introduce a principled yet simple solution that incorporates the operation costs before establishing mappings. To improve matching efficiency, we propose a strategy that proactively optimizes guidance from a graph perspective. This strategy initializes guidance as each node's alignment difficulty and captures the interdependencies between matches within and across graphs through a difficulty propagation mechanism, enabling more informed decisions. As a result, GEN selects optimal matches in a single step, minimizing the need for costly refinements. Results on real-world and synthetic datasets demonstrate the effectiveness, time efficiency, and adaptability of GEN, achieving up to 37.8\% error reduction and 72.7\% inference time reduction compared with state-of-the-art models, while performing robustly under varying cost settings and graph sizes. 

**Abstract (ZH)**: 基于图编辑网络的灵活图编辑距离计算 

---
# WaveHiTS: Wavelet-Enhanced Hierarchical Time Series Modeling for Wind Direction Nowcasting in Eastern Inner Mongolia 

**Title (ZH)**: WaveHiTS：增强小波变换的风向短时预报分层时间序列建模在内蒙古东部地区 

**Authors**: Hailong Shu, Weiwei Song, Yue Wang, Jiping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06532)  

**Abstract**: Wind direction forecasting plays a crucial role in optimizing wind energy production, but faces significant challenges due to the circular nature of directional data, error accumulation in multi-step forecasting, and complex meteorological interactions. This paper presents a novel model, WaveHiTS, which integrates wavelet transform with Neural Hierarchical Interpolation for Time Series to address these challenges. Our approach decomposes wind direction into U-V components, applies wavelet transform to capture multi-scale frequency patterns, and utilizes a hierarchical structure to model temporal dependencies at multiple scales, effectively mitigating error propagation. Experiments conducted on real-world meteorological data from Inner Mongolia, China demonstrate that WaveHiTS significantly outperforms deep learning models (RNN, LSTM, GRU), transformer-based approaches (TFT, Informer, iTransformer), and hybrid models (EMD-LSTM). The proposed model achieves RMSE values of approximately 19.2°-19.4° compared to 56°-64° for deep learning recurrent models, maintaining consistent accuracy across all forecasting steps up to 60 minutes ahead. Moreover, WaveHiTS demonstrates superior robustness with vector correlation coefficients (VCC) of 0.985-0.987 and hit rates of 88.5%-90.1%, substantially outperforming baseline models. Ablation studies confirm that each component-wavelet transform, hierarchical structure, and U-V decomposition-contributes meaningfully to overall performance. These improvements in wind direction nowcasting have significant implications for enhancing wind turbine yaw control efficiency and grid integration of wind energy. 

**Abstract (ZH)**: 基于小波变换的神经分层插值模型WaveHiTS在风向预测中的应用 

---
# Beyond Moore's Law: Harnessing the Redshift of Generative AI with Effective Hardware-Software Co-Design 

**Title (ZH)**: 超越摩尔定律：通过有效的硬件软件协同设计利用生成式AI的红移效应 

**Authors**: Amir Yazdanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06531)  

**Abstract**: For decades, Moore's Law has served as a steadfast pillar in computer architecture and system design, promoting a clear abstraction between hardware and software. This traditional Moore's computing paradigm has deepened the rift between the two, enabling software developers to achieve near-exponential performance gains often without needing to delve deeply into hardware-specific optimizations. Yet today, Moore's Law -- with its once relentless performance gains now diminished to incremental improvements -- faces inevitable physical barriers. This stagnation necessitates a reevaluation of the conventional system design philosophy. The traditional decoupled system design philosophy, which maintains strict abstractions between hardware and software, is increasingly obsolete. The once-clear boundary between software and hardware is rapidly dissolving, replaced by co-design. It is imperative for the computing community to intensify its commitment to hardware-software co-design, elevating system abstractions to first-class citizens and reimagining design principles to satisfy the insatiable appetite of modern computing. Hardware-software co-design is not a recent innovation. To illustrate its historical evolution, I classify its development into five relatively distinct ``epochs''. This post also highlights the growing influence of the architecture community in interdisciplinary teams -- particularly alongside ML researchers -- and explores why current co-design paradigms are struggling in today's computing landscape. Additionally, I will examine the concept of the ``hardware lottery'' and explore directions to mitigate its constraining influence on the next era of computing innovation. 

**Abstract (ZH)**: 多年来，摩尔定律一直是计算机体系结构和系统设计中的坚实支柱，促进了硬件和软件之间的清晰抽象。传统的摩尔计算范式加深了两者之间的鸿沟，使软件开发者能够在很大程度上无需深入了解硬件特定优化的情况下实现接近指数级别的性能提升。然而，随着摩尔定律从曾经不断的性能提升转变为微小改进，它不可避免地遇到了物理壁垒。这种停滞需要重新评估传统的系统设计哲学。传统的分离系统设计哲学，即保持硬件和软件之间严格的抽象，正变得日益过时。软件和硬件之间的界限正在迅速模糊，取而代之的是协同设计。计算社区必须加强对硬件-软件协同设计的承诺，提升系统抽象到头等重要的地位，并重塑设计原则以满足现代计算的无尽需求。硬件-软件协同设计不是最近才有的创新。为了说明其历史演变，我将其划分为五个相对独立的“阶段”。本文还强调了架构社区在跨学科团队中的日益影响，特别是与ML研究人员的合作，并探讨了为什么当前的协同设计模式在当今计算环境中显得力不从心。此外，本文还将探讨“硬件彩票”这一概念，并探讨减轻其对下一阶段计算创新制约影响的方向。 

---
# TSP-OCS: A Time-Series Prediction for Optimal Camera Selection in Multi-Viewpoint Surgical Video Analysis 

**Title (ZH)**: TSP-OCS：多视角手术视频分析中的最优相机选择时间序列预测 

**Authors**: Xinyu Liu, Xiaoguang Lin, Xiang Liu, Yong Yang, Hongqian Wang, Qilong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.06527)  

**Abstract**: Recording the open surgery process is essential for educational and medical evaluation purposes; however, traditional single-camera methods often face challenges such as occlusions caused by the surgeon's head and body, as well as limitations due to fixed camera angles, which reduce comprehensibility of the video content. This study addresses these limitations by employing a multi-viewpoint camera recording system, capturing the surgical procedure from six different angles to mitigate occlusions. We propose a fully supervised learning-based time series prediction method to choose the best shot sequences from multiple simultaneously recorded video streams, ensuring optimal viewpoints at each moment. Our time series prediction model forecasts future camera selections by extracting and fusing visual and semantic features from surgical videos using pre-trained models. These features are processed by a temporal prediction network with TimeBlocks to capture sequential dependencies. A linear embedding layer reduces dimensionality, and a Softmax classifier selects the optimal camera view based on the highest probability. In our experiments, we created five groups of open thyroidectomy videos, each with simultaneous recordings from six different angles. The results demonstrate that our method achieves competitive accuracy compared to traditional supervised methods, even when predicting over longer time horizons. Furthermore, our approach outperforms state-of-the-art time series prediction techniques on our dataset. This manuscript makes a unique contribution by presenting an innovative framework that advances surgical video analysis techniques, with significant implications for improving surgical education and patient safety. 

**Abstract (ZH)**: 记录开放式手术过程对于教育和医疗评估至关重要；然而，传统的单摄像头方法常常面临由于外科医生头部和身体造成的遮挡问题，以及固定摄像头角度限制，这降低了视频内容的可理解性。本研究通过采用多视角摄像头记录系统来解决这些限制，从六个不同的角度捕捉手术过程以减轻遮挡问题。我们提出了一种基于完全监督学习的时间序列预测方法，从多个同时录制的视频流中选择最佳的镜头序列，确保每个时刻的最佳视角。我们的时间序列预测模型通过预训练模型提取和融合手术视频的视觉和语义特征来预测未来的摄像头选择，并由时间卷积网络捕捉序列依赖性。线性嵌入层降低了维度，Softmax分类器根据最高概率选择最优的摄像头视角。在我们的实验中，我们创建了五组开放式甲状腺切除手术视频，每组视频同时从六个不同的角度进行录制。结果表明，我们的方法在预测长时间范围时仍能达到与传统监督方法相当的准确性。此外，我们的方法在我们的数据集上优于最先进的时间序列预测技术。本文通过提出一种创新框架，推动了手术视频分析技术的发展，对提高手术教育和患者安全具有重要意义。 

---
# The Power of the Pareto Front: Balancing Uncertain Rewards for Adaptive Experimentation in scanning probe microscopy 

**Title (ZH)**: 帕累托前沿的威力：在扫描探针显微镜自适应实验中平衡不确定奖励 

**Authors**: Yu Liu, Sergei V. Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06525)  

**Abstract**: Automated experimentation has the potential to revolutionize scientific discovery, but its effectiveness depends on well-defined optimization targets, which are often uncertain or probabilistic in real-world settings. In this work, we demonstrate the application of Multi-Objective Bayesian Optimization (MOBO) to balance multiple, competing rewards in autonomous experimentation. Using scanning probe microscopy (SPM) imaging, one of the most widely used and foundational SPM modes, we show that MOBO can optimize imaging parameters to enhance measurement quality, reproducibility, and efficiency. A key advantage of this approach is the ability to compute and analyze the Pareto front, which not only guides optimization but also provides physical insights into the trade-offs between different objectives. Additionally, MOBO offers a natural framework for human-in-the-loop decision-making, enabling researchers to fine-tune experimental trade-offs based on domain expertise. By standardizing high-quality, reproducible measurements and integrating human input into AI-driven optimization, this work highlights MOBO as a powerful tool for advancing autonomous scientific discovery. 

**Abstract (ZH)**: 多目标贝叶斯优化在自主实验中的应用：以扫描探针显微镜成像为例 

---
# Continuous-Variable Quantum Encoding Techniques: A Comparative Study of Embedding Techniques and Their Impact on Machine Learning Performance 

**Title (ZH)**: 连续变量量子编码技术：嵌入技术的比较研究及其对机器学习性能的影响 

**Authors**: Minati Rath, Hema Date  

**Link**: [PDF](https://arxiv.org/pdf/2504.06497)  

**Abstract**: This study explores the intersection of continuous-variable quantum computing (CVQC) and classical machine learning, focusing on CVQC data encoding techniques, including Displacement encoding and squeezing encoding, alongside Instantaneous Quantum Polynomial (IQP) encoding from discrete quantum computing. We perform an extensive empirical analysis to assess the impact of these encoding methods on classical machine learning models, such as Logistic Regression, Support Vector Machines, K-Nearest Neighbors, and ensemble methods like Random Forest and LightGBM. Our findings indicate that CVQC-based encoding methods significantly enhance feature expressivity, resulting in improved classification accuracy and F1 scores, especially in high-dimensional and complex datasets. However, these improvements come with varying computational costs, which depend on the complexity of the encoding and the architecture of the machine learning models. Additionally, we examine the trade-off between quantum expressibility and classical learnability, offering valuable insights into the practical feasibility of incorporating these quantum encodings into real-world applications. This study contributes to the growing body of research on quantum-classical hybrid learning, emphasizing the role of CVQC in advancing quantum data representation and its integration into classical machine learning workflows. 

**Abstract (ZH)**: 本研究探讨了连续变量量子计算（CVQC）与经典机器学习的交叉领域，重点关注CVQC数据编码技术，包括位移编码和压缩编码，以及离散量子计算中的瞬时量子多项式（IQP）编码。通过广泛的实证分析，评估这些编码方法对经典机器学习模型（如逻辑回归、支持向量机、K-最近邻以及随机森林和LightGBM等集成方法）的影响。研究结果表明，基于CVQC的编码方法显著增强了特征表达性，提高了分类准确率和F1分数，特别是在高维和复杂数据集中表现尤为明显。然而，这些改进伴随着不同的计算成本，这取决于编码的复杂性和机器学习模型的架构。此外，本研究还探讨了量子表达性和经典可学习性之间的权衡，为将这些量子编码纳入实际应用提供了宝贵的见解。本研究为量子-经典混合学习领域的研究增添了新的成果，强调了CVQC在推动量子数据表示及其与经典机器学习流程整合中的作用。 

---
# Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction 

**Title (ZH)**: 基于元学习的中毒攻击在图链接预测中的应用 

**Authors**: Mingchen Li, Di Zhuang, Keyu Chen, Dumindu Samaraweera, Morris Chang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06492)  

**Abstract**: Link prediction in graph data utilizes various algorithms and machine learning/deep learning models to predict potential relationships between graph nodes. This technique has found widespread use in numerous real-world applications, including recommendation systems, community networks, and biological structures. However, recent research has highlighted the vulnerability of link prediction models to adversarial attacks, such as poisoning and evasion attacks. Addressing the vulnerability of these models is crucial to ensure stable and robust performance in link prediction applications. While many works have focused on enhancing the robustness of the Graph Convolution Network (GCN) model, the Variational Graph Auto-Encoder (VGAE), a sophisticated model for link prediction, has not been thoroughly investigated in the context of graph adversarial attacks. To bridge this gap, this article proposes an unweighted graph poisoning attack approach using meta-learning techniques to undermine VGAE's link prediction performance. We conducted comprehensive experiments on diverse datasets to evaluate the proposed method and its parameters, comparing it with existing approaches in similar settings. Our results demonstrate that our approach significantly diminishes link prediction performance and outperforms other state-of-the-art methods. 

**Abstract (ZH)**: 图数据中的链接预测利用各种算法和机器学习/深度学习模型来预测图节点之间的潜在关系。这项技术在推荐系统、社区网络和生物结构等众多实际应用中得到了广泛应用。然而，近期的研究表明，链接预测模型容易受到对抗性攻击（如投毒和规避攻击）的影响。确保这些模型在链接预测应用中的稳定和稳健性能至关重要。虽然许多研究集中在增强图卷积网络（GCN）模型的鲁棒性上，但针对图对抗性攻击的变分图自编码器（VGAE）模型尚未得到充分的研究。为此，本文提出了一种基于元学习的无权重图投毒攻击方法，以削弱VGAE的链接预测性能。我们在多种数据集上进行了全面实验，评估了所提出的方法及其参数，并将其与类似设置下的现有方法进行了比较。结果显示，我们的方法显著降低了链接预测性能，并优于其他最先进的方法。 

---
# AI-Assisted Transport of Radioactive Ion Beams 

**Title (ZH)**: AI辅助的放射性离子束运输 

**Authors**: Sergio Lopez-Caceres, Daniel Santiago-Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2504.06469)  

**Abstract**: Beams of radioactive heavy ions allow researchers to study rare and unstable atomic nuclei, shedding light into the internal structure of exotic nuclei and on how chemical elements are formed in stars. However, the extraction and transport of radioactive beams rely on time-consuming expert-driven tuning methods, where hundreds of parameters are manually optimized. Here, we introduce a system that uses Artificial Intelligence (AI) to assist in the radioactive beam transport process. We apply our methodology to real-life scenarios showing advantages when compared with standard tuning methods. Our method can be extended to other radioactive beam facilities around the world to improve operational efficiency and enhance scientific output. 

**Abstract (ZH)**: 放射性重离子束可用于研究稀有和不稳定的原子核，揭示奇异核的内部结构，并阐明化学元素在恒星中的形成机制。然而，放射性束的提取和传输依赖于耗时的专家驱动的调谐方法，其中需要手动优化数百个参数。在这里，我们介绍了一种使用人工智能（AI）辅助放射性束传输过程的系统。我们将我们的方法应用于实际场景，显示出与标准调谐方法相比的优势。该方法可以扩展到世界各地的其他放射性束设施，以提高运营效率并增强科学研究成果。 

---
# Agent-Arena: A General Framework for Evaluating Control Algorithms 

**Title (ZH)**: Agent-Arena：一种评估控制算法的一般框架 

**Authors**: Halid Abdulrahim Kadi, Kasim Terzić  

**Link**: [PDF](https://arxiv.org/pdf/2504.06468)  

**Abstract**: Robotic research is inherently challenging, requiring expertise in diverse environments and control algorithms. Adapting algorithms to new environments often poses significant difficulties, compounded by the need for extensive hyper-parameter tuning in data-driven methods. To address these challenges, we present Agent-Arena, a Python framework designed to streamline the integration, replication, development, and testing of decision-making policies across a wide range of benchmark environments. Unlike existing frameworks, Agent-Arena is uniquely generalised to support all types of control algorithms and is adaptable to both simulation and real-robot scenarios. Please see our GitHub repository this https URL. 

**Abstract (ZH)**: 机器人研究本性上具有挑战性，要求具备多种环境和控制算法的专业知识。将算法适应新环境往往会带来重大困难，并且数据驱动方法中需要进行广泛的超参数调优。为了应对这些挑战，我们提出了Agent-Arena，这是一个Python框架，旨在简化决策政策在多种基准环境中的集成、复制、开发和测试过程。与现有框架不同，Agent-Arena具有高度通用性，支持所有类型的控制算法，并且能够适应仿真和真实机器人场景。请访问我们的GitHub仓库：this https URL。 

---
# Federated Neural Architecture Search with Model-Agnostic Meta Learning 

**Title (ZH)**: 基于模型无拘束元学习的联邦神经架构搜索 

**Authors**: Xinyuan Huang, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06457)  

**Abstract**: Federated Learning (FL) often struggles with data heterogeneity due to the naturally uneven distribution of user data across devices. Federated Neural Architecture Search (NAS) enables collaborative search for optimal model architectures tailored to heterogeneous data to achieve higher accuracy. However, this process is time-consuming due to extensive search space and retraining. To overcome this, we introduce FedMetaNAS, a framework that integrates meta-learning with NAS within the FL context to expedite the architecture search by pruning the search space and eliminating the retraining stage. Our approach first utilizes the Gumbel-Softmax reparameterization to facilitate relaxation of the mixed operations in the search space. We then refine the local search process by incorporating Model-Agnostic Meta-Learning, where a task-specific learner adapts both weights and architecture parameters (alphas) for individual tasks, while a meta learner adjusts the overall model weights and alphas based on the gradient information from task learners. Following the meta-update, we propose soft pruning using the same trick on search space to gradually sparsify the architecture, ensuring that the performance of the chosen architecture remains robust after pruning which allows for immediate use of the model without retraining. Experimental evaluations demonstrate that FedMetaNAS significantly accelerates the search process by more than 50\% with higher accuracy compared to FedNAS. 

**Abstract (ZH)**: 联邦学习中的元学习神经架构搜索（FedMetaNAS）：通过缩减搜索空间和消除重新训练加速异质数据下的架构搜索 

---
# Can you Finetune your Binoculars? Embedding Text Watermarks into the Weights of Large Language Models 

**Title (ZH)**: 你可以微调你的双筒望远镜吗？将文本水印嵌入大型语言模型的权重中 

**Authors**: Fay Elhassan, Niccolò Ajroldi, Antonio Orvieto, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2504.06446)  

**Abstract**: The indistinguishability of AI-generated content from human text raises challenges in transparency and accountability. While several methods exist to watermark models behind APIs, embedding watermark strategies directly into model weights that are later reflected in the outputs of the model is challenging. In this study we propose a strategy to finetune a pair of low-rank adapters of a model, one serving as the text-generating model, and the other as the detector, so that a subtle watermark is embedded into the text generated by the first model and simultaneously optimized for detectability by the second. In this way, the watermarking strategy is fully learned end-to-end. This process imposes an optimization challenge, as balancing watermark robustness, naturalness, and task performance requires trade-offs. We discuss strategies on how to optimize this min-max objective and present results showing the effect of this modification to instruction finetuning. 

**Abstract (ZH)**: AI生成内容与人类文本难以区分增加了透明度和问责制的挑战：一种端到端学习的细调策略 

---
# Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning 

**Title (ZH)**: 不要让它幻觉：基于检索增强逻辑推理的前提验证 

**Authors**: Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06438)  

**Abstract**: Large language models (LLMs) have shown substantial capacity for generating fluent, contextually appropriate responses. However, they can produce hallucinated outputs, especially when a user query includes one or more false premises-claims that contradict established facts. Such premises can mislead LLMs into offering fabricated or misleading details. Existing approaches include pretraining, fine-tuning, and inference-time techniques that often rely on access to logits or address hallucinations after they occur. These methods tend to be computationally expensive, require extensive training data, or lack proactive mechanisms to prevent hallucination before generation, limiting their efficiency in real-time applications. We propose a retrieval-based framework that identifies and addresses false premises before generation. Our method first transforms a user's query into a logical representation, then applies retrieval-augmented generation (RAG) to assess the validity of each premise using factual sources. Finally, we incorporate the verification results into the LLM's prompt to maintain factual consistency in the final output. Experiments show that this approach effectively reduces hallucinations, improves factual accuracy, and does not require access to model logits or large-scale fine-tuning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了生成流畅且上下文相关响应的显著能力。然而，当用户查询包含一个或多个虚假前提（与已确立的事实相矛盾的断言）时，它们可能会生成虚构或误导性的输出。现有方法包括预训练、微调以及推理时的技术，这些方法通常依赖于对logits的访问，或者在生成后处理幻觉。这些方法往往计算成本高、需要大量的训练数据，或者缺乏在生成前预防幻觉的主动机制，限制了它们在实时应用中的效率。我们提出了一种检索为基础的框架，在生成之前识别并处理虚假前提。该方法首先将用户的查询转化为逻辑表示，然后运用检索增强生成（RAG）来使用事实来源评估每个前提的有效性。最后，我们将验证结果融入到LLM的提示中，以确保最终输出的符合事实。实验表明，这种方法能够有效减少幻觉，提高事实准确性，并且不需要访问模型logits或大规模微调。 

---
# Language-Dependent Political Bias in AI: A Study of ChatGPT and Gemini 

**Title (ZH)**: 依赖语言的政治偏见在AI中：ChatGPT和Gemini的研究 

**Authors**: Dogus Yuksel, Mehmet Cem Catalbas, Bora Oc  

**Link**: [PDF](https://arxiv.org/pdf/2504.06436)  

**Abstract**: As leading examples of large language models, ChatGPT and Gemini claim to provide accurate and unbiased information, emphasizing their commitment to political neutrality and avoidance of personal bias. This research investigates the political tendency of large language models and the existence of differentiation according to the query language. For this purpose, ChatGPT and Gemini were subjected to a political axis test using 14 different languages. The findings of the study suggest that these large language models do exhibit political tendencies, with both models demonstrating liberal and leftist biases. A comparative analysis revealed that Gemini exhibited a more pronounced liberal and left-wing tendency compared to ChatGPT. The study also found that these political biases varied depending on the language used for inquiry. The study delves into the factors that constitute political tendencies and linguistic differentiation, exploring differences in the sources and scope of educational data, structural and grammatical features of languages, cultural and political contexts, and the model's response to linguistic features. From this standpoint, and an ethical perspective, it is proposed that artificial intelligence tools should refrain from asserting a lack of political tendencies and neutrality, instead striving for political neutrality and executing user queries by incorporating these tendencies. 

**Abstract (ZH)**: 作为大型语言模型的领先范例，ChatGPT和Gemini声称提供准确且无偏见的信息，强调其政治中立和避免个人偏见的承诺。本研究调查了大型语言模型的政治倾向及其查询语言根据政治轴的分化情况。为此，使用14种不同的语言对ChatGPT和Gemini进行了政治轴测试。研究结果表明，这些大型语言模型确实表现出政治倾向，两者都表现出自由派和左倾偏见。对比分析显示，Gemini相比ChatGPT更表现出明显的自由派和左倾倾向。研究还发现，这些政治偏见在使用不同语言进行查询时有所差异。本研究深入探讨了构成政治倾向和语言分化的因素，探索了教育资源和范围、语言的结构和语法特征、文化及政治背景以及模型对语言特征的响应差异。从这一视角及伦理角度来看，本研究建议人工智能工具应避免声称缺乏政治倾向和中立性，而是追求政治中立并综合考虑这些倾向执行用户查询。 

---
# Evaluating Mutation Techniques in Genetic Algorithm-Based Quantum Circuit Synthesis 

**Title (ZH)**: 基于遗传算法的量子电路合成中突变技术评估 

**Authors**: Michael Kölle, Tom Bintener, Maximilian Zorn, Gerhard Stenzel, Leo Sünkel, Thomas Gabor, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2504.06413)  

**Abstract**: Quantum computing leverages the unique properties of qubits and quantum parallelism to solve problems intractable for classical systems, offering unparalleled computational potential. However, the optimization of quantum circuits remains critical, especially for noisy intermediate-scale quantum (NISQ) devices with limited qubits and high error rates. Genetic algorithms (GAs) provide a promising approach for efficient quantum circuit synthesis by automating optimization tasks. This work examines the impact of various mutation strategies within a GA framework for quantum circuit synthesis. By analyzing how different mutations transform circuits, it identifies strategies that enhance efficiency and performance. Experiments utilized a fitness function emphasizing fidelity, while accounting for circuit depth and T operations, to optimize circuits with four to six qubits. Comprehensive hyperparameter testing revealed that combining delete and swap strategies outperformed other approaches, demonstrating their effectiveness in developing robust GA-based quantum circuit optimizers. 

**Abstract (ZH)**: 量子计算利用量子位的独特性质和量子并行性来解决经典系统无法处理的问题，提供无与伦比的计算潜力。然而，量子电路的优化对于具有有限量子位数和高错误率的嘈杂中等规模量子（NISQ）设备来说仍然至关重要。遗传算法（GAs）为通过自动化优化任务高效合成量子电路提供了有希望的方法。本文研究了遗传算法框架下不同变异策略对量子电路合成的影响。通过分析不同变异如何变换电路，确定了能提升效率和性能的策略。实验使用强调保真度的适应度函数，同时考虑电路深度和T操作，对四到六个量子位的电路进行优化。全面的超参数测试显示，结合删除和交换策略优于其他方法，证明了它们在开发稳健的基于遗传算法的量子电路优化器方面的有效性。 

---
# Understanding Machine Unlearning Through the Lens of Mode Connectivity 

**Title (ZH)**: 通过模式连接性视角理解机器卸载 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.06407)  

**Abstract**: Machine Unlearning aims to remove undesired information from trained models without requiring full retraining from scratch. Despite recent advancements, their underlying loss landscapes and optimization dynamics received less attention. In this paper, we investigate and analyze machine unlearning through the lens of mode connectivity - the phenomenon where independently trained models can be connected by smooth low-loss paths in the parameter space. We define and study mode connectivity in unlearning across a range of overlooked conditions, including connections between different unlearning methods, models trained with and without curriculum learning, and models optimized with first-order and secondorder techniques. Our findings show distinct patterns of fluctuation of different evaluation metrics along the curve, as well as the mechanistic (dis)similarity between unlearning methods. To the best of our knowledge, this is the first study on mode connectivity in the context of machine unlearning. 

**Abstract (ZH)**: 机器遗忘旨在从训练模型中移除不希望的信息，而无需从头完全重新训练。尽管最近取得了进展，但其潜在的损失景观和优化动力学仍未得到充分关注。在本文中，我们通过模式连通性的视角来研究和分析机器遗忘——即独立训练的模型可以通过参数空间中的光滑低损失路径相互连接的现象。我们在一系列未被关注的条件下定义和研究了遗忘过程中的模式连通性，包括不同遗忘方法之间的连接、带有和不带有渐进学习的模型以及使用一阶和二阶技术优化的模型。我们的发现显示了不同评估指标沿曲线波动的不同模式，以及不同遗忘方法之间的（不）相似性机制。据我们所知，这是首次在机器遗忘的背景下研究模式连通性的研究。 

---
# Physical spline for denoising object trajectory data by combining splines, ML feature regression and model knowledge 

**Title (ZH)**: 物理样条：结合样条、机器学习特征回归和模型知识的对象轨迹数据去噪方法 

**Authors**: Jonas Torzewski  

**Link**: [PDF](https://arxiv.org/pdf/2504.06404)  

**Abstract**: This article presents a method for estimating the dynamic driving states (position, velocity, acceleration and heading) from noisy measurement data. The proposed approach is effective with both complete and partial observations, producing refined trajectory signals with kinematic consistency, ensuring that velocity is the integral of acceleration and position is the integral of velocity. Additionally, the method accounts for the constraint that vehicles can only move in the direction of their orientation. The method is implemented as a configurable python library that also enables trajectory estimation solely based on position data. Regularization is applied to prevent extreme state variations. A key application is enhancing recorded trajectory data for use as reference inputs in machine learning models. At the end, the article presents the results of the method along with a comparison to ground truth data. 

**Abstract (ZH)**: 本文提出了一种从噪声测量数据中估计动态驾驶状态（位置、速度、加速度和航向）的方法。所提出的方案能够在完整和部分观测的情况下有效地工作，生成具有动力学一致性的精细轨迹信号，确保速度是加速度的积分，位置是速度的积分。此外，该方法考虑了车辆只能在其方向上移动的约束。该方法实现为一个可配置的Python库，还能够仅基于位置数据进行轨迹估计。应用正则化以防止状态变化极端。主要应用是增强记录的轨迹数据，使其适合作为机器学习模型的参考输入。最后，文章呈现了该方法的结果，并与真实数据进行了比较。 

---
# Analyzing the Impact of Low-Rank Adaptation for Cross-Domain Few-Shot Object Detection in Aerial Images 

**Title (ZH)**: 低秩适应对跨域少样本目标检测在航空图像中的影响分析 

**Authors**: Hicham Talaoubrid, Anissa Mokraoui, Ismail Ben Ayed, Axel Prouvost, Sonimith Hang, Monit Korn, Rémi Harvey  

**Link**: [PDF](https://arxiv.org/pdf/2504.06330)  

**Abstract**: This paper investigates the application of Low-Rank Adaptation (LoRA) to small models for cross-domain few-shot object detection in aerial images. Originally designed for large-scale models, LoRA helps mitigate overfitting, making it a promising approach for resource-constrained settings. We integrate LoRA into DiffusionDet, and evaluate its performance on the DOTA and DIOR datasets. Our results show that LoRA applied after an initial fine-tuning slightly improves performance in low-shot settings (e.g., 1-shot and 5-shot), while full fine-tuning remains more effective in higher-shot configurations. These findings highlight LoRA's potential for efficient adaptation in aerial object detection, encouraging further research into parameter-efficient fine-tuning strategies for few-shot learning. Our code is available here: this https URL. 

**Abstract (ZH)**: 本文探讨了将低秩适应（LoRA）应用于小模型以在航空图像中进行跨域少量样本目标检测的应用。LoRA originally designed for大规模模型，有助于缓解过拟合现象，使其成为资源受限环境中的一种有前景的方法。我们将LoRA集成到DiffusionDet中，并在DOTA和DIOR数据集上评估其性能。结果显示，LoRA在少量样本设置（例如1-shot和5-shot）中应用于初始微调后能够略微提高性能，而完全微调在高样本设置中仍然更有效。这些发现突显了LoRA在航空目标检测中高效适应的潜力，并鼓励对参数高效微调策略进行进一步研究，以支持少量样本学习。我们的代码可在这里获取：this https URL。 

---
# A Geometric-Aware Perspective and Beyond: Hybrid Quantum-Classical Machine Learning Methods 

**Title (ZH)**: 具几何感知视角与超越：混合量子-经典机器学习方法 

**Authors**: Azadeh Alavia, Hossein Akhoundib, Fatemeh Kouchmeshkib, Mojtaba Mahmoodianc, Sanduni Jayasinghec, Yongli Rena, Abdolrahman Alavi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06328)  

**Abstract**: Geometric Machine Learning (GML) has shown that respecting non-Euclidean geometry in data spaces can significantly improve performance over naive Euclidean assumptions. In parallel, Quantum Machine Learning (QML) has emerged as a promising paradigm that leverages superposition, entanglement, and interference within quantum state manifolds for learning tasks. This paper offers a unifying perspective by casting QML as a specialized yet more expressive branch of GML. We argue that quantum states, whether pure or mixed, reside on curved manifolds (e.g., projective Hilbert spaces or density-operator manifolds), mirroring how covariance matrices inhabit the manifold of symmetric positive definite (SPD) matrices or how image sets occupy Grassmann manifolds. However, QML also benefits from purely quantum properties, such as entanglement-induced curvature, that can yield richer kernel structures and more nuanced data embeddings.
We illustrate these ideas with published and newly discussed results, including hybrid classical -quantum pipelines for diabetic foot ulcer classification and structural health monitoring. Despite near-term hardware limitations that constrain purely quantum solutions, hybrid architectures already demonstrate tangible benefits by combining classical manifold-based feature extraction with quantum embeddings. We present a detailed mathematical treatment of the geometrical underpinnings of quantum states, emphasizing parallels to classical Riemannian geometry and manifold-based optimization. Finally, we outline open research challenges and future directions, including Quantum Large Language Models (LLMs), quantum reinforcement learning, and emerging hardware approaches, demonstrating how synergizing GML and QML principles can unlock the next generation of machine intelligence. 

**Abstract (ZH)**: 几何机器学习（GML）表明，在数据空间中尊重非欧几里得几何可以显著提高性能，超越了简单的欧几里得假设。与此同时，量子机器学习（QML）作为一种有前景的范式出现了，它利用量子态流形中的叠加、纠缠和干涉来进行学习任务。本文从统一的角度将QML视为GML的一个专门但更具有表现力的分支。我们argue量子态，无论是纯态还是混合态，都位于弯曲流形上（例如，投影希洛特空间或密度算子流形），这类似于协方差矩阵存在于对称正定（SPD）矩阵的流形上，或者图像集占据格拉斯曼流形。然而，QML还受益于纯粹的量子属性，例如由纠缠引起的曲率，这些属性可以产生更丰富的核结构和更细腻的数据嵌入。

我们通过一些已发表和新讨论的结果，如糖尿病足溃疡分类和结构健康监测的混合经典-量子管道来阐述这些观点。尽管短期内硬件限制阻碍了纯量子解决方案的发展，但混合架构已经通过结合基于流形的经典特征提取与量子嵌入显示了实际优势。我们详述了量子态的几何基础，强调与经典黎曼几何及基于流形的优化的类比关系。最后，我们概述了开放的研究挑战和未来方向，包括量子大型语言模型（LLMs）、量子强化学习和新兴硬件方法，并展示了如何结合GML和QML原则来解锁下一代机器智能。 

---
# MM-STFlowNet: A Transportation Hub-Oriented Multi-Mode Passenger Flow Prediction Method via Spatial-Temporal Dynamic Graph Modeling 

**Title (ZH)**: MM-STFlowNet: 基于时空动态图建模的多模态交通枢纽客流量预测方法 

**Authors**: Ronghui Zhang, Wenbin Xing, Mengran Li, Zihan Wang, Junzhou Chen, Xiaolei Ma, Zhiyuan Liu, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2504.06325)  

**Abstract**: Accurate and refined passenger flow prediction is essential for optimizing the collaborative management of multiple collection and distribution modes in large-scale transportation hubs. Traditional methods often focus only on the overall passenger volume, neglecting the interdependence between different modes within the hub. To address this limitation, we propose MM-STFlowNet, a comprehensive multi-mode prediction framework grounded in dynamic spatial-temporal graph modeling. Initially, an integrated temporal feature processing strategy is implemented using signal decomposition and convolution techniques to address data spikes and high volatility. Subsequently, we introduce the Spatial-Temporal Dynamic Graph Convolutional Recurrent Network (STDGCRN) to capture detailed spatial-temporal dependencies across multiple traffic modes, enhanced by an adaptive channel attention mechanism. Finally, the self-attention mechanism is applied to incorporate various external factors, further enhancing prediction accuracy. Experiments on a real-world dataset from Guangzhounan Railway Station in China demonstrate that MM-STFlowNet achieves state-of-the-art performance, particularly during peak periods, providing valuable insight for transportation hub management. 

**Abstract (ZH)**: 多模式精细 passenger flow 预测对于大型交通枢纽多模式协作管理的优化至关重要。传统方法往往只关注整体乘客流量，忽视了枢纽内部不同模式之间的相互依赖性。为解决这一局限性，我们提出了基于动态空时图建模的全面多模式预测框架 MM-STFlowNet。首先，通过信号分解和卷积技术实现集成时间特征处理策略，以应对数据尖峰和高波动性。随后，引入空间-时间动态图卷积循环网络(STDGCRN)，并结合自适应通道注意力机制以捕捉多个交通模式之间的详细空时依赖性。最后，应用自注意力机制融入各种外部因素，进一步提升预测精度。实验结果表明，MM-STFlowNet 在真实数据集（来自中国广州南站）上达到了最先进的性能，特别是在高峰时段，为交通枢纽管理提供了有价值的见解。 

---
# From Stability to Inconsistency: A Study of Moral Preferences in LLMs 

**Title (ZH)**: 从稳定到不一致性：LLM中道德偏好的研究 

**Authors**: Monika Jotautaite, Mary Phuong, Chatrik Singh Mangat, Maria Angelica Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2504.06324)  

**Abstract**: As large language models (LLMs) increasingly integrate into our daily lives, it becomes crucial to understand their implicit biases and moral tendencies. To address this, we introduce a Moral Foundations LLM dataset (MFD-LLM) grounded in Moral Foundations Theory, which conceptualizes human morality through six core foundations. We propose a novel evaluation method that captures the full spectrum of LLMs' revealed moral preferences by answering a range of real-world moral dilemmas. Our findings reveal that state-of-the-art models have remarkably homogeneous value preferences, yet demonstrate a lack of consistency. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）日益融入我们的日常生活中，理解其隐含偏见和道德倾向变得至关重要。为此，我们引入了一个基于道德基础理论（Moral Foundations Theory）的道德基础LLM数据集（MFD-LLM），该理论通过六个核心基础来概念化人类的道德观。我们提出了一种新颖的评估方法，通过回答一系列现实生活中的道德难题来捕捉LLMs展现出来的完整道德偏好谱系。我们的研究发现，最先进的模型在价值观偏好上表现出显著的同质性，但缺乏一致性。 

---
# Mosaic: Composite Projection Pruning for Resource-efficient LLMs 

**Title (ZH)**: 拼图：面向资源高效的大语言模型的复合投影剪枝 

**Authors**: Bailey J. Eccles, Leon Wong, Blesson Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2504.06323)  

**Abstract**: Extensive compute and memory requirements limit the deployment of large language models (LLMs) on any hardware. Compression methods, such as pruning, can reduce model size, which in turn reduces resource requirements. State-of-the-art pruning is based on coarse-grained methods. They are time-consuming and inherently remove critical model parameters, adversely impacting the quality of the pruned model. This paper introduces projection pruning, a novel fine-grained method for pruning LLMs. In addition, LLM projection pruning is enhanced by a new approach we refer to as composite projection pruning - the synergistic combination of unstructured pruning that retains accuracy and structured pruning that reduces model size. We develop Mosaic, a novel system to create and deploy pruned LLMs using composite projection pruning. Mosaic is evaluated using a range of performance and quality metrics on multiple hardware platforms, LLMs, and datasets. Mosaic is 7.19x faster in producing models than existing approaches. Mosaic models achieve up to 84.2% lower perplexity and 31.4% higher accuracy than models obtained from coarse-grained pruning. Up to 67% faster inference and 68% lower GPU memory use is noted for Mosaic models. 

**Abstract (ZH)**: 大规模语言模型的广泛计算和内存需求限制了其在任何硬件上的部署。压缩方法，如剪枝，可以减小模型大小，从而减少资源需求。最先进的剪枝方法基于粗粒度的方法。它们耗时且不可避免地会移除关键的模型参数，负面影响了剪枝模型的质量。本文介绍了一种名为投影剪枝的新颖细粒度方法，用于剪枝大规模语言模型。此外，通过一种我们称之为复合投影剪枝的全新方法——无结构剪枝保留准确性和有结构剪枝减少模型大小的协同组合，增强了大规模语言模型的投影剪枝。我们开发了Mosaic，一种新型系统，使用复合投影剪枝创建和部署剪枝的大规模语言模型。Mosaic在多种硬件平台、大规模语言模型和数据集上使用范围广泛的性能和质量指标进行了评估。Mosaic在生成模型方面的速度比现有方法快7.19倍。Mosaic模型的困惑度降低了最多84.2%，准确率提高了31.4%，高于粗粒度剪枝获得的模型。Mosaic模型的推理速度提高了最多67%，GPU内存使用量降低了68%。 

---
# Assessing employment and labour issues implicated by using AI 

**Title (ZH)**: 评估使用AI涉及的就业与劳动问题 

**Authors**: Thijs Willems, Darion Jin Hotan, Jiawen Cheryl Tang, Norakmal Hakim bin Norhashim, King Wang Poon, Zi An Galvyn Goh, Radha Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2504.06322)  

**Abstract**: This chapter critiques the dominant reductionist approach in AI and work studies, which isolates tasks and skills as replaceable components. Instead, it advocates for a systemic perspective that emphasizes the interdependence of tasks, roles, and workplace contexts. Two complementary approaches are proposed: an ethnographic, context-rich method that highlights how AI reconfigures work environments and expertise; and a relational task-based analysis that bridges micro-level work descriptions with macro-level labor trends. The authors argue that effective AI impact assessments must go beyond predicting automation rates to include ethical, well-being, and expertise-related questions. Drawing on empirical case studies, they demonstrate how AI reshapes human-technology relations, professional roles, and tacit knowledge practices. The chapter concludes by calling for a human-centric, holistic framework that guides organizational and policy decisions, balancing technological possibilities with social desirability and sustainability of work. 

**Abstract (ZH)**: 本章批评人工智能和工作研究中的主导还原论方法，该方法孤立并视为可替代的任务和技能成分。相反，它倡导一种系统视角，强调任务、角色和工作场所环境之间的相互依存关系。提出了两种互补的方法：一种是富含情境的民族志方法，强调人工智能如何重新配置工作环境和专业知识；另一种是关系导向的任务分析方法，将微观层面的工作描述与宏观层面的劳动趋势联系起来。作者认为，有效的AI影响评估应超越预测自动化率，包括道德、福祉和专业知识相关的问题。基于实证案例研究，他们展示了人工智能如何重塑人机关系、专业角色和默会知识实践。本章结尾呼吁采用以人为中心、综合性框架来指导组织和政策决策，平衡技术可能性与社会可接受性和工作的可持续性。 

---
# Hybrid Temporal Differential Consistency Autoencoder for Efficient and Sustainable Anomaly Detection in Cyber-Physical Systems 

**Title (ZH)**: 基于时空差分一致性自编码器的高效可持续的网络物理系统异常检测 

**Authors**: Michael Somma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06320)  

**Abstract**: Cyberattacks on critical infrastructure, particularly water distribution systems, have increased due to rapid digitalization and the integration of IoT devices and industrial control systems (ICS). These cyber-physical systems (CPS) introduce new vulnerabilities, requiring robust and automated intrusion detection systems (IDS) to mitigate potential threats. This study addresses key challenges in anomaly detection by leveraging time correlations in sensor data, integrating physical principles into machine learning models, and optimizing computational efficiency for edge applications. We build upon the concept of temporal differential consistency (TDC) loss to capture the dynamics of the system, ensuring meaningful relationships between dynamic states. Expanding on this foundation, we propose a hybrid autoencoder-based approach, referred to as hybrid TDC-AE, which extends TDC by incorporating both deterministic nodes and conventional statistical nodes. This hybrid structure enables the model to account for non-deterministic processes. Our approach achieves state-of-the-art classification performance while improving time to detect anomalies by 3%, outperforming the BATADAL challenge leader without requiring domain-specific knowledge, making it broadly applicable. Additionally, it maintains the computational efficiency of conventional autoencoders while reducing the number of fully connected layers, resulting in a more sustainable and efficient solution. The method demonstrates how leveraging physics-inspired consistency principles enhances anomaly detection and strengthens the resilience of cyber-physical systems. 

**Abstract (ZH)**: 基于时序差分一致性混合自编码器的 cyber-物理系统异常检测方法 

---
# Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching 

**Title (ZH)**: 通过异步键值缓存预取加速LLM推理 throughput 

**Authors**: Yanhao Dong, Yubo Miao, Weinan Li, Xiao Zheng, Chao Wang, Feng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06319)  

**Abstract**: Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap. By strategically scheduling idle memory bandwidth during active computation windows, our method proactively prefetches required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for subsequent accesses and effectively hiding HBM access latency within computational cycles. Extensive experiments on NVIDIA H20 GPUs demonstrate that the proposed method achieves 2.15x improvement in attention kernel efficiency and up to 1.97x end-to-end throughput enhancement, surpassing state-of-the-art baseline FlashAttention-3. Notably, our solution maintains orthogonality to existing optimization techniques and can be integrated with current inference frameworks, providing a scalable latency-hiding solution for next-generation LLM inference engines. 

**Abstract (ZH)**: Large Language Models的推理过程中由于高带宽内存(HBM)带宽约束表现出显著的记忆绑定特性。本文提出了一种面向L2缓存的异步键值缓存预取方法，通过计算负载重叠突破LLM推理中的内存带宽瓶颈。通过在活跃计算窗口期间战略性调度闲置的内存带宽，本方法主动将所需的键值缓存预取到GPU L2缓存中，从而在后续访问时实现高速L2缓存命中，并有效隐藏HBM访问延迟到计算周期内。实验结果表明，本方法在NVIDIA H20 GPU上实现了注意力核效率2.15倍的提升，并最多提升了1.97倍的端到端吞吐量，超越了最新的基准FlashAttention-3。值得注意的是，本解决方案与现有的优化技术保持正交，并且可以与当前的推理框架集成，为下一代LLM推理引擎提供可扩展的延迟隐藏解决方案。 

---
# DMol: A Schedule-Driven Diffusion Model for Highly Efficient and Versatile Molecule Generation 

**Title (ZH)**: DMol：基于调度驱动的高效多功能分子生成扩散模型 

**Authors**: Peizhi Niu, Yu-Hsiang Wang, Vishal Rana, Chetan Rupakheti, Abhishek Pandey, Olgica Milenkovic  

**Link**: [PDF](https://arxiv.org/pdf/2504.06312)  

**Abstract**: We introduce a new graph diffusion model for small molecule generation, \emph{DMol}, which outperforms the state-of-the-art DiGress model in terms of validity by roughly $1.5\%$ across all benchmarking datasets while reducing the number of diffusion steps by at least $10$-fold, and the running time to roughly one half. The performance improvements are a result of a careful change in the objective function and a ``graph noise" scheduling approach which, at each diffusion step, allows one to only change a subset of nodes of varying size in the molecule graph. Another relevant property of the method is that it can be easily combined with junction-tree-like graph representations that arise by compressing a collection of relevant ring structures into supernodes. Unlike classical junction-tree techniques that involve VAEs and require complicated reconstruction steps, compressed DMol directly performs graph diffusion on a graph that compresses only a carefully selected set of frequent carbon rings into supernodes, which results in straightforward sample generation. This compressed DMol method offers additional validity improvements over generic DMol of roughly $2\%$, increases the novelty of the method, and further improves the running time due to reductions in the graph size. 

**Abstract (ZH)**: 一种新的小分子生成图扩散模型DMol：与最先进的DiGress模型相比，在所有基准数据集中有效性提高了大约1.5%，同时减少了一定比例的扩散步数，并将运行时间缩短至大约一半。该性能提升得益于目标函数的精细调整和一种“图噪声”调度方法，在每次扩散步中仅改变分子图中大小可变的子集节点。该方法还可以与由压缩相关环结构成的超节点表示的类似区间树图表示轻松结合。与涉及VAEs和复杂重构步骤的经典区间树技术不同，压缩DMol直接在压缩了精心选择的常见碳环成超节点的图上执行图扩散，从而实现直接样本生成。该压缩DMol方法还为通用DMol提供了约2%的有效性改进，增加了方法的新颖性，并进一步通过减少图的大小来提高运行时间。 

---
# Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Encoding 

**Title (ZH)**: 重新思考RoPE：N维位置编码的数学蓝图 

**Authors**: Haiping Liu, Hongpeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06308)  

**Abstract**: Rotary Position Embedding (RoPE) is widely adopted in Transformers due to its ability to encode relative positions with high efficiency and extrapolation capability. However, existing RoPE variants lack a unified theoretical foundation, especially in higher dimensions. In this paper, we propose a systematic mathematical framework for RoPE grounded in Lie group and Lie algebra theory. We identify two core properties of RoPE, named relativity and reversibility, and derive general constraints and constructions for valid RoPE in 1D, 2D, and N-dimensional (ND). We prove that RoPE must lie in the basis of a maximal abelian subalgebra (MASA) of the special orthogonal Lie algebra, and show that standard RoPE corresponds to the maximal toral subalgebra. Furthermore, we propose to model inter-dimensional interactions by learning an orthogonal basis transformation. Our framework unifies and explains existing RoPE designs, while enabling principled extensions to new modalities and tasks. 

**Abstract (ZH)**: 基于李群和李代数理论的旋转位置嵌入系统化数学框架 

---
# Optimizing Large Language Models: Metrics, Energy Efficiency, and Case Study Insights 

**Title (ZH)**: 优化大型语言模型：评价指标、能源效率及案例研究洞察 

**Authors**: Tahniat Khan, Soroor Motie, Sedef Akinli Kocak, Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2504.06307)  

**Abstract**: The rapid adoption of large language models (LLMs) has led to significant energy consumption and carbon emissions, posing a critical challenge to the sustainability of generative AI technologies. This paper explores the integration of energy-efficient optimization techniques in the deployment of LLMs to address these environmental concerns. We present a case study and framework that demonstrate how strategic quantization and local inference techniques can substantially lower the carbon footprints of LLMs without compromising their operational effectiveness. Experimental results reveal that these methods can reduce energy consumption and carbon emissions by up to 45\% post quantization, making them particularly suitable for resource-constrained environments. The findings provide actionable insights for achieving sustainability in AI while maintaining high levels of accuracy and responsiveness. 

**Abstract (ZH)**: 大规模语言模型的快速 Adopt 采用导致了显著的能源消耗和碳排放，对生成型 AI 技术的可持续性构成了关键挑战。本文探讨了在部署大规模语言模型时集成高效的能源优化技术，以应对这些环境问题。我们展示了一项案例研究和框架，说明如何通过战略性量化和本地推理技术大幅降低大规模语言模型的碳足迹，而不影响其操作有效性。实验结果表明，这些方法在量化之后可以降低高达 45% 的能源消耗和碳排放，特别适合资源受限的环境。研究结果提供了在保证高准确性和响应性的基础上实现 AI 可持续性的可行见解。 

---
# Predicting Survivability of Cancer Patients with Metastatic Patterns Using Explainable AI 

**Title (ZH)**: 使用可解释AI预测具有转移模式的癌症患者的生存率 

**Authors**: Polycarp Nalela, Deepthi Rao, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06306)  

**Abstract**: Cancer remains a leading global health challenge and a major cause of mortality. This study leverages machine learning (ML) to predict the survivability of cancer patients with metastatic patterns using the comprehensive MSK-MET dataset, which includes genomic and clinical data from 25,775 patients across 27 cancer types. We evaluated five ML models-XGBoost, Naïve Bayes, Decision Tree, Logistic Regression, and Random Fores using hyperparameter tuning and grid search. XGBoost emerged as the best performer with an area under the curve (AUC) of 0.82. To enhance model interpretability, SHapley Additive exPlanations (SHAP) were applied, revealing key predictors such as metastatic site count, tumor mutation burden, fraction of genome altered, and organ-specific metastases. Further survival analysis using Kaplan-Meier curves, Cox Proportional Hazards models, and XGBoost Survival Analysis identified significant predictors of patient outcomes, offering actionable insights for clinicians. These findings could aid in personalized prognosis and treatment planning, ultimately improving patient care. 

**Abstract (ZH)**: 癌症仍然是全球健康的主要挑战和主要死因。本研究利用机器学习（ML）方法，通过包含27种癌症类型、25,775名患者的综合MSK-MET数据集（包括基因组和临床数据），预测具有转移模式的癌症患者的存活率。我们评估了五种机器学习模型——XGBoost、朴素贝叶斯、决策树、逻辑回归和随机森林，并进行了超参数调优和网格搜索。XGBoost表现出色，其曲线下面积（AUC）为0.82。为进一步提高模型的可解释性，我们应用了SHapley Additive exPlanations（SHAP），揭示了关键预测因子，如转移部位数量、肿瘤突变负担、基因组改变比例以及器官特异性转移。通过使用Kaplan-Meier曲线、Cox比例风险模型和XGBoost生存分析进一步进行生存分析，确定了对患者结果有显著影响的预测因子，为临床提供了可操作的见解。这些发现有助于个性化预后和治疗计划的制定，最终提高患者护理质量。 

---
# Well2Flow: Reconstruction of reservoir states from sparse wells using score-based generative models 

**Title (ZH)**: Well2Flow: 从稀疏井信息重构储层状态的评分基于生成模型方法 

**Authors**: Shiqin Zeng, Haoyun Li, Abhinav Prakash Gahlot, Felix J. Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06305)  

**Abstract**: This study investigates the use of score-based generative models for reservoir simulation, with a focus on reconstructing spatially varying permeability and saturation fields in saline aquifers, inferred from sparse observations at two well locations. By modeling the joint distribution of permeability and saturation derived from high-fidelity reservoir simulations, the proposed neural network is trained to learn the complex spatiotemporal dynamics governing multiphase fluid flow in porous media. During inference, the framework effectively reconstructs both permeability and saturation fields by conditioning on sparse vertical profiles extracted from well log data. This approach introduces a novel methodology for incorporating physical constraints and well log guidance into generative models, significantly enhancing the accuracy and physical plausibility of the reconstructed subsurface states. Furthermore, the framework demonstrates strong generalization capabilities across varying geological scenarios, highlighting its potential for practical deployment in data-scarce reservoir management tasks. 

**Abstract (ZH)**: 基于评分生成模型的储层模拟研究：稀疏井筒观测数据驱动的盐水含水层渗透率和饱和度场重建 

---
# On the Effectiveness and Generalization of Race Representations for Debiasing High-Stakes Decisions 

**Title (ZH)**: 高风险决策中种族表示的纠偏有效性与泛化能力研究 

**Authors**: Dang Nguyen, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06303)  

**Abstract**: Understanding and mitigating biases is critical for the adoption of large language models (LLMs) in high-stakes decision-making. We introduce Admissions and Hiring, decision tasks with hypothetical applicant profiles where a person's race can be inferred from their name, as simplified test beds for racial bias. We show that Gemma 2B Instruct and LLaMA 3.2 3B Instruct exhibit strong biases. Gemma grants admission to 26% more White than Black applicants, and LLaMA hires 60% more Asian than White applicants. We demonstrate that these biases are resistant to prompt engineering: multiple prompting strategies all fail to promote fairness. In contrast, using distributed alignment search, we can identify "race subspaces" within model activations and intervene on them to debias model decisions. Averaging the representation across all races within the subspaces reduces Gemma's bias by 37-57%. Finally, we examine the generalizability of Gemma's race subspaces, and find limited evidence for generalization, where changing the prompt format can affect the race representation. Our work suggests mechanistic approaches may provide a promising venue for improving the fairness of LLMs, but a universal race representation remains elusive. 

**Abstract (ZH)**: 理解并减轻偏差对于在高风险决策中采用大规模语言模型（LLMs）至关重要。我们介绍了 Admission 和 Hiring，具有假设申请人背景的任务，在这些任务中可以从姓名推断出一个人的种族，作为种族偏差的简化测试平台。我们显示，Gemma 2B Instruct 和 LLaMA 3.2 3B Instruct 显示出强烈的偏差。Gemma 向白人申请者授予入学资格的比例比向非洲裔美国人申请者高 26%，而 LLaMA 对亚裔申请者的雇佣率比对白人申请者的高 60%。我们证明这些偏差对提示工程具有抵抗力：多种提示策略均未能促进公正性。相比之下，通过分布式对齐搜索，我们可以在模型激活中识别出“种族子空间”，并对它们进行干预以消除模型决策中的偏差。在子空间内跨所有种族平均表示将 Gemma 的偏差降低 37-57%。最后，我们检查了 Gemma 的种族子空间的一般适用性，发现变化提示格式会影响种族表示的有限证据。我们的工作表明，机制性方法可能为改进 LLM 的公平性提供有前景的途径，但普遍适用的种族表示仍然难以实现。 

---
# Resurrecting Socrates in the Age of AI: A Study Protocol for Evaluating a Socratic Tutor to Support Research Question Development in Higher Education 

**Title (ZH)**: 在人工智能时代复活苏格拉底：一个关于评估苏格拉底式辅导以支持高等教育研究问题发展的研究方案 

**Authors**: Ben Degen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06294)  

**Abstract**: Formulating research questions is a foundational yet challenging academic skill, one that generative AI systems often oversimplify by offering instant answers at the expense of student reflection. This protocol lays out a study grounded in constructivist learning theory to evaluate a novel AI-based Socratic Tutor, designed to foster cognitive engagement and scaffold research question development in higher education. Anchored in dialogic pedagogy, the tutor engages students through iterative, reflective questioning, aiming to promote System 2 thinking and counteract overreliance on AI-generated outputs. In a quasi-experimental design, approximately 80 German pre-service biology teacher students will be randomly assigned to one of two groups: an AI Socratic Tutor condition and an uninstructed chatbot control. Across multiple cycles, students are expected to formulate research questions based on background texts, with quality assessed through double-blind expert review. The study also examines transfer of skills to novel phenomena and captures student perceptions through mixed-methods analysis, including surveys, interviews and reflective journals. This study aims to advance the understanding of how generative AI can be pedagogically aligned to support, not replace, human cognition and offers design principles for human-AI collaboration in education. 

**Abstract (ZH)**: 基于建构主义学习理论的新型基于AI的苏格拉底式导师研究：促进高等教育中认知参与和研究问题开发的对话式教学实践与评估 

---
# Temporal-contextual Event Learning for Pedestrian Crossing Intent Prediction 

**Title (ZH)**: 基于时空上下文的行人过街意图预测 

**Authors**: Hongbin Liang, Hezhe Qiao, Wei Huang, Qizhou Wang, Mingsheng Shang, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06292)  

**Abstract**: Ensuring the safety of vulnerable road users through accurate prediction of pedestrian crossing intention (PCI) plays a crucial role in the context of autonomous and assisted driving. Analyzing the set of observation video frames in ego-view has been widely used in most PCI prediction methods to forecast the cross intent. However, they struggle to capture the critical events related to pedestrian behaviour along the temporal dimension due to the high redundancy of the video frames, which results in the sub-optimal performance of PCI prediction. Our research addresses the challenge by introducing a novel approach called \underline{T}emporal-\underline{c}ontextual Event \underline{L}earning (TCL). The TCL is composed of the Temporal Merging Module (TMM), which aims to manage the redundancy by clustering the observed video frames into multiple key temporal events. Then, the Contextual Attention Block (CAB) is employed to adaptively aggregate multiple event features along with visual and non-visual data. By synthesizing the temporal feature extraction and contextual attention on the key information across the critical events, TCL can learn expressive representation for the PCI prediction. Extensive experiments are carried out on three widely adopted datasets, including PIE, JAAD-beh, and JAAD-all. The results show that TCL substantially surpasses the state-of-the-art methods. Our code can be accessed at this https URL. 

**Abstract (ZH)**: 确保通过准确预测行人过街意图（PCI）来保护弱势道路使用者在自动驾驶和辅助驾驶的背景下发挥着关键作用。传统的基于自我视角视频帧集合的方法在大多数PCI预测方法中被广泛使用以预测过街意图，但由于视频帧的高冗余性，它们难以在时间维度上捕捉到与行人行为相关的关键事件，从而导致PCI预测的性能欠佳。我们的研究通过引入一种名为Temporal-Contextual Event Learning (TCL)的新颖方法来应对这一挑战。TCL由Temporal Merging Module (TMM)组成，旨在通过将观测到的视频帧聚类为多个关键时间事件来管理冗余性。然后，采用Contextual Attention Block (CAB)根据视觉和非视觉数据自适应地聚合多个事件特征。通过在关键事件的关键信息上综合时序特征提取和上下文关注，TCL能够学习用于PCI预测的表达性表示。在PIE、JAAD-beh和JAAD-all三个广泛采用的数据集上进行了大量实验，结果显示TCL显著超越了现有最先进的方法。我们的代码可以通过以下链接访问。 

---
# Dynamic Evaluation Framework for Personalized and Trustworthy Agents: A Multi-Session Approach to Preference Adaptability 

**Title (ZH)**: 个性化和可信代理的动态评估框架：一种会话多阶段偏好适应方法 

**Authors**: Chirag Shah, Hideo Joho, Kirandeep Kaur, Preetam Prabhu Srikar Dammu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06277)  

**Abstract**: Recent advancements in generative AI have significantly increased interest in personalized agents. With increased personalization, there is also a greater need for being able to trust decision-making and action taking capabilities of these agents. However, the evaluation methods for these agents remain outdated and inadequate, often failing to capture the dynamic and evolving nature of user interactions. In this conceptual article, we argue for a paradigm shift in evaluating personalized and adaptive agents. We propose a comprehensive novel framework that models user personas with unique attributes and preferences. In this framework, agents interact with these simulated users through structured interviews to gather their preferences and offer customized recommendations. These recommendations are then assessed dynamically using simulations driven by Large Language Models (LLMs), enabling an adaptive and iterative evaluation process. Our flexible framework is designed to support a variety of agents and applications, ensuring a comprehensive and versatile evaluation of recommendation strategies that focus on proactive, personalized, and trustworthy aspects. 

**Abstract (ZH)**: 近期生成式人工智能的发展显著增加了对个性化代理的兴趣。随着个性化程度的提高，人们也需要更信任这些代理的决策能力和行动能力。然而，这些代理的评估方法仍然过时且不足，往往未能捕捉用户交互的动态和演变特性。在本文中，我们提出了评估个性化和适应性代理范式的转变。我们提出了一种综合性的新框架，该框架基于具有独特属性和偏好的用户画像。在这种框架中，代理通过结构化的访谈与这些模拟用户交互，以收集用户的偏好并提供定制化建议。随后，这些建议通过大型语言模型（LLMs）驱动的模拟动态评估，实现适应性和迭代性的评估过程。我们的灵活框架旨在支持各种代理和应用，确保对注重前瞻性、个性化和可信度的推荐策略进行全面而多样的评估。 

---
# A Cascaded Architecture for Extractive Summarization of Multimedia Content via Audio-to-Text Alignment 

**Title (ZH)**: 基于音频到文本对齐的多媒体内容提取性总结的级联架构 

**Authors**: Tanzir Hossain, Ar-Rafi Islam, Md. Sabbir Hossain, Annajiat Alim Rasel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06275)  

**Abstract**: This study presents a cascaded architecture for extractive summarization of multimedia content via audio-to-text alignment. The proposed framework addresses the challenge of extracting key insights from multimedia sources like YouTube videos. It integrates audio-to-text conversion using Microsoft Azure Speech with advanced extractive summarization models, including Whisper, Pegasus, and Facebook BART XSum. The system employs tools such as Pytube, Pydub, and SpeechRecognition for content retrieval, audio extraction, and transcription. Linguistic analysis is enhanced through named entity recognition and semantic role labeling. Evaluation using ROUGE and F1 scores demonstrates that the cascaded architecture outperforms conventional summarization methods, despite challenges like transcription errors. Future improvements may include model fine-tuning and real-time processing. This study contributes to multimedia summarization by improving information retrieval, accessibility, and user experience. 

**Abstract (ZH)**: 本研究提出了一种级联架构，通过音频到文本对齐实现多媒体内容的摘要提取。提出的框架解决了从YouTube视频等多媒体源中提取关键见解的挑战。该框架结合了Microsoft Azure Speech的音频到文本转换与Whisper、Pegasus和Facebook BART XSum等先进的摘要模型。系统利用Pytube、Pydub和SpeechRecognition等工具进行内容检索、音频提取和转录。通过命名实体识别和语义角色标注增强语言分析。使用ROUGE和F1评分的评估表明，级联架构在面对转录错误等挑战的情况下仍优于传统的摘要方法。未来可能的改进包括模型微调和实时处理。本研究通过提高信息检索、可访问性和用户体验，为多媒体摘要做出了贡献。 

---
# Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning 

**Title (ZH)**: 基于深度神经网络的多任务学习联合群体画像与推荐 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06274)  

**Abstract**: Group recommender systems aim to generate recommendations that align with the collective preferences of a group, introducing challenges that differ significantly from those in individual recommendation scenarios. This paper presents Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning, a framework that unifies group profiling and recommendation tasks within a single model. By jointly learning these tasks, the model develops a deeper understanding of group dynamics, leading to improved recommendation accuracy. The shared representations between the two tasks facilitate the discovery of latent features essential to both, resulting in richer and more informative group embeddings. To further enhance performance, an attention mechanism is integrated to dynamically evaluate the relevance of different group features and item attributes, ensuring the model prioritizes the most impactful information. Experiments and evaluations on real-world datasets demonstrate that our multi-task learning approach consistently outperforms baseline models in terms of accuracy, validating its effectiveness and robustness. 

**Abstract (ZH)**: 基于深度神经网络多任务学习的联合群体画像与推荐框架 

---
# A Diverse and Effective Retrieval-Based Debt Collection System with Expert Knowledge 

**Title (ZH)**: 一种融合专家知识的多样化和有效债务催收检索系统 

**Authors**: Jiaming Luo, Weiyi Luo, Guoqing Sun, Mengchen Zhu, Haifeng Tang, Kunyao Lan, Mengyue Wu, Kenny Q. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06273)  

**Abstract**: Designing effective debt collection systems is crucial for improving operational efficiency and reducing costs in the financial industry. However, the challenges of maintaining script diversity, contextual relevance, and coherence make this task particularly difficult. This paper presents a debt collection system based on real debtor-collector data from a major commercial bank. We construct a script library from real-world debt collection conversations, and propose a two-stage retrieval based response system for contextual relevance. Experimental results show that our system improves script diversity, enhances response relevance, and achieves practical deployment efficiency through knowledge distillation. This work offers a scalable and automated solution, providing valuable insights for advancing debt collection practices in real-world applications. 

**Abstract (ZH)**: 设计有效的债务催收系统对于提高金融行业的运营效率和降低成本至关重要。然而，保持脚本多样性、上下文相关性和连贯性的挑战使得这项任务尤为困难。本文基于一家主要商业银行的真实债务人-催收员对话数据，提出了一种债务催收系统。我们从实际的债务催收对话中构建了一个脚本库，并提出了一种基于两阶段检索的响应系统，以提高上下文相关性。实验结果表明，我们的系统通过知识蒸馏提高了脚本多样性、增强了响应的相关性，并实现了实际部署的效率。本工作提供了一种可扩展的自动化解决方案，为实际应用中的债务催收实践提供了宝贵见解。 

---
# RAVEN: An Agentic Framework for Multimodal Entity Discovery from Large-Scale Video Collections 

**Title (ZH)**: RAVEN：大型视频集合多模态实体发现的代理框架 

**Authors**: Kevin Dela Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2504.06272)  

**Abstract**: We present RAVEN an adaptive AI agent framework designed for multimodal entity discovery and retrieval in large-scale video collections. Synthesizing information across visual, audio, and textual modalities, RAVEN autonomously processes video data to produce structured, actionable representations for downstream tasks. Key contributions include (1) a category understanding step to infer video themes and general-purpose entities, (2) a schema generation mechanism that dynamically defines domain-specific entities and attributes, and (3) a rich entity extraction process that leverages semantic retrieval and schema-guided prompting. RAVEN is designed to be model-agnostic, allowing the integration of different vision-language models (VLMs) and large language models (LLMs) based on application-specific requirements. This flexibility supports diverse applications in personalized search, content discovery, and scalable information retrieval, enabling practical applications across vast datasets. 

**Abstract (ZH)**: 我们提出了RAVEN，一种针对大规模视频集合中多模态实体发现和检索设计的自适应AI代理框架。通过跨视觉、音频和文本模态综合信息，RAVEN自主处理视频数据，生成结构化、可操作的表示以供下游任务使用。主要贡献包括：(1) 一种类别理解步骤，用于推断视频主题和通用实体；(2) 一种模式生成机制，动态定义领域特定实体和属性；(3) 一种丰富的内容提取过程，利用语义检索和模式引导提示。RAVEN设计为模型agnostic，可根据特定应用需求集成不同的视觉-语言模型（VLMs）和大型语言模型（LLMs）。这种灵活性支持个性化搜索、内容发现和可扩展信息检索等多种应用，使得在大规模数据集中实现实际应用成为可能。 

---
# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources 

**Title (ZH)**: ER-RAG: 基于ER统一异构数据源建模增强RAG 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06271)  

**Abstract**: Large language models (LLMs) excel in question-answering (QA) tasks, and retrieval-augmented generation (RAG) enhances their precision by incorporating external evidence from diverse sources like web pages, databases, and knowledge graphs. However, current RAG methods rely on agent-specific strategies for individual data sources, posing challenges low-resource or black-box environments and complicates operations when evidence is fragmented across sources. To address these limitations, we propose ER-RAG, a framework that unifies evidence integration across heterogeneous data sources using the Entity-Relationship (ER) model. ER-RAG standardizes entity retrieval and relationship querying through ER-based APIs with GET and JOIN operations. It employs a two-stage generation process: first, a preference optimization module selects optimal sources; second, another module constructs API chains based on source schemas. This unified approach allows efficient fine-tuning and seamless integration across diverse data sources. ER-RAG demonstrated its effectiveness by winning all three tracks of the 2024 KDDCup CRAG Challenge, achieving performance on par with commercial RAG pipelines using an 8B LLM backbone. It outperformed hybrid competitors by 3.1% in LLM score and accelerated retrieval by 5.5X. 

**Abstract (ZH)**: 大型语言模型（LLMs）在问答（QA）任务中表现出色，检索增强生成（RAG）通过整合来自网页、数据库和知识图谱等多样来源的外部证据来提高其精确度。然而，当前的RAG方法依赖于针对个别数据源的特定策略，这在低资源或黑盒环境中提出了挑战，并且当证据分散在多个来源时会复杂化操作。为了解决这些限制，我们提出了一种ER-RAG框架，该框架使用实体-关系（ER）模型统一异构数据源的证据集成。ER-RAG通过基于ER的APIs和GET、JOIN操作标准化实体检索和关系查询。它采用两阶段生成过程：首先，偏好优化模块选择最佳来源；其次，另一个模块基于数据源模式构建API链。这种统一的方法允许在多种数据源之间高效调整和无缝集成。ER-RAG通过赢得2024 KDDCup CRAG挑战的所有三个赛道，展示了其有效性，使用8B LLM骨干时，其性能与商用RAG流水线相当。与混合竞争对手相比，其LLM得分为3.1%的提升，并加速了检索速度5.5倍。 

---
# Addressing Cold-start Problem in Click-Through Rate Prediction via Supervised Diffusion Modeling 

**Title (ZH)**: 基于监督扩散 modeling 解决点击率预测中的冷启动问题 

**Authors**: Wenqiao Zhu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06270)  

**Abstract**: Predicting Click-Through Rates is a crucial function within recommendation and advertising platforms, as the output of CTR prediction determines the order of items shown to users. The Embedding \& MLP paradigm has become a standard approach for industrial recommendation systems and has been widely deployed. However, this paradigm suffers from cold-start problems, where there is either no or only limited user action data available, leading to poorly learned ID embeddings. The cold-start problem hampers the performance of new items. To address this problem, we designed a novel diffusion model to generate a warmed-up embedding for new items. Specifically, we define a novel diffusion process between the ID embedding space and the side information space. In addition, we can derive a sub-sequence from the diffusion steps to expedite training, given that our diffusion model is non-Markovian. Our diffusion model is supervised by both the variational inference and binary cross-entropy objectives, enabling it to generate warmed-up embeddings for items in both the cold-start and warm-up phases. Additionally, we have conducted extensive experiments on three recommendation datasets. The results confirmed the effectiveness of our approach. 

**Abstract (ZH)**: 预测点击率是推荐和广告平台中的一项关键功能，输出的点击率预测结果决定了展示给用户的物品顺序。嵌入与MLP范式已成为工业推荐系统的一项标准方法，并得到了广泛应用。然而，该范式在冷启动问题上存在局限，即缺乏或仅有有限的用户行为数据，导致ID嵌入学习效果不佳。冷启动问题妨碍了新项目的表现。为解决这一问题，我们设计了一个新型扩散模型来生成新项目的预热嵌入。具体而言，我们在ID嵌入空间与辅助信息空间之间定义了一个新型的扩散过程。此外，在我们的扩散模型是非马尔可夫模型的情况下，可以从扩散步骤中提取子序列以加速训练。我们的扩散模型同时受变分推断和二元交叉熵目标函数的监督，使其能够在冷启动和预热阶段为项目生成预热嵌入。我们还在三个推荐数据集上进行了广泛的实验。实验结果证实了我们方法的有效性。 

---
# EXCLAIM: An Explainable Cross-Modal Agentic System for Misinformation Detection with Hierarchical Retrieval 

**Title (ZH)**: EXCLAIM：一种具有层次检索的可解释跨模态代理系统以检测 misinformation 

**Authors**: Yin Wu, Zhengxuan Zhang, Fuling Wang, Yuyu Luo, Hui Xiong, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06269)  

**Abstract**: Misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. Among its various manifestations, Out-of-Context (OOC) misinformation is particularly obscure, as it distorts meaning by pairing authentic images with misleading textual narratives. Existing methods for detecting OOC misinformation predominantly rely on coarse-grained similarity metrics between image-text pairs, which often fail to capture subtle inconsistencies or provide meaningful explainability. While multi-modal large language models (MLLMs) demonstrate remarkable capabilities in visual reasoning and explanation generation, they have not yet demonstrated the capacity to address complex, fine-grained, and cross-modal distinctions necessary for robust OOC detection. To overcome these limitations, we introduce EXCLAIM, a retrieval-based framework designed to leverage external knowledge through multi-granularity index of multi-modal events and entities. Our approach integrates multi-granularity contextual analysis with a multi-agent reasoning architecture to systematically evaluate the consistency and integrity of multi-modal news content. Comprehensive experiments validate the effectiveness and resilience of EXCLAIM, demonstrating its ability to detect OOC misinformation with 4.3% higher accuracy compared to state-of-the-art approaches, while offering explainable and actionable insights. 

**Abstract (ZH)**: Out-of-Context misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. To address its complex and fine-grained distinctions, we introduce EXCLAIM, a retrieval-based framework that leverages external knowledge through multi-granularity indexing of multi-modal events and entities. 

---
# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Title (ZH)**: StealthRank: 通过隐蔽的提示优化进行LLM排名操纵 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）集成到信息检索系统中引入了新的攻击面，尤其是针对对抗性排名操纵。我们提出了一种名为StealthRank的新型对抗性排名攻击方法，该方法在保持文本流畅性和隐蔽性的同时操纵LLM驱动的产品推荐系统。与现有方法往往引入可检测的异常不同，StealthRank采用能量优化框架结合拉格朗日动态机制生成StealthRank提示（SRPs），即嵌入在产品描述中的对抗性文本序列，这些序列微妙而有效地影响LLM的排名机制。我们跨多个LLM评估了StealthRank，展示了其隐蔽提升目标产品排名的能力，同时避免了容易被检测的明确操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面均优于最先进的对抗性排名基准，突显了LLM驱动的推荐系统中的关键漏洞。 

---
# Leveraging LLMs for User Stories in AI Systems: UStAI Dataset 

**Title (ZH)**: 利用大语言模型在AI系统中的用户故事：UStAI数据集 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00513)  

**Abstract**: AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use. 

**Abstract (ZH)**: 基于学术论文摘要的大规模语言模型生成AI系统用户故事的研究 

---
# Multi-objective Optimization in CPU Design Space Exploration: Attention is All You Need 

**Title (ZH)**: CPU 设计空间探索中的多目标优化：只需注意力机制 

**Authors**: Runzhen Xue, Hao Wu, Mingyu Yan, Ziheng Xiao, Xiaochun Ye, Dongrui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2410.18368)  

**Abstract**: Design space exploration (DSE) enables architects to systematically evaluate various design options, guiding decisions on the most suitable configurations to meet specific objectives such as optimizing performance, power, and area. However, the growing complexity of modern CPUs has dramatically increased the number of micro-architectural parameters and expanded the overall design space, making DSE more challenging and time-consuming. Existing DSE frameworks struggle in large-scale design spaces due to inaccurate models and limited insights into parameter impact, hindering efficient identification of optimal micro-architectures within tight timeframes.
In this work, we introduce AttentionDSE. Its key idea is to use the attention mechanism to establish a direct mapping of micro-architectural parameters to their contributions to predicted performance. This approach enhances both the prediction accuracy and interpretability of the performance model. Furthermore, the weights are dynamically adjusted, enabling the model to respond to design changes and effectively pinpoint the key micro-architectural parameters/components responsible for performance bottlenecks. Thus, AttentionDSE accurately, purposefully, and rapidly discovers optimal designs. Experiments on SPEC 2017 demonstrate that AttentionDSE significantly reduces exploration time by over 80\% and achieves 3.9\% improvement in Pareto Hypervolume compared to state-of-the-art DSE frameworks while maintaining superior prediction accuracy and efficiency with an increasing number of parameters. 

**Abstract (ZH)**: 设计空间探索（DSE）使架构师能够系统地评估各种设计选项，指导优化性能、功率和面积等特定目标的最佳配置决策。然而，现代CPU日益复杂性增加了微观架构参数的数量，扩大了整个设计空间，使得DSE更加具有挑战性和耗时性。现有的DSE框架在大规模设计空间中受限于不准确的模型和对参数影响的有限洞察，阻碍了在紧凑的时间框架内高效识别最优微观架构。

在这项工作中，我们提出了AttentionDSE。其核心思想是利用注意力机制建立微观架构参数与其对预测性能贡献之间的直接映射。这种方法增强了性能模型的预测准确性和可解释性。此外，权重动态调整，使模型能够响应设计更改，并有效定位导致性能瓶颈的关键微观架构参数/组件。因此，AttentionDSE能够准确、有针对性和快速地发现最优设计。实验结果表明，AttentionDSE在SPEC 2017上显著减少了探索时间超过80%，并在参数数量增加的同时，实现了3.9%的帕累托hypervolume改进，同时保持了更高的预测准确性和效率。 

---
# CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models 

**Title (ZH)**: CMAT：增强小型语言模型的多代理协作调优框架 

**Authors**: Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, JingSong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2404.01663)  

**Abstract**: Open large language models (LLMs) have significantly advanced the field of natural language processing, showcasing impressive performance across various this http URL the significant advancements in LLMs, their effective operation still relies heavily on human input to accurately guide the dialogue flow, with agent tuning being a crucial optimization technique that involves human adjustments to the model for better response to such this http URL this dependency, our work introduces the TinyAgent model, trained on a meticulously curated high-quality dataset. We also present the Collaborative Multi-Agent Tuning (CMAT) framework, an innovative system designed to augment language agent capabilities through adaptive weight updates based on environmental feedback. This framework fosters collaborative learning and real-time adaptation among multiple intelligent agents, enhancing their context-awareness and long-term memory. In this research, we propose a new communication agent framework that integrates multi-agent systems with environmental feedback mechanisms, offering a scalable method to explore cooperative behaviors. Notably, our TinyAgent-7B model exhibits performance on par with GPT-3.5, despite having fewer parameters, signifying a substantial improvement in the efficiency and effectiveness of LLMs. 

**Abstract (ZH)**: 开放型大型语言模型（LLMs）在自然语言处理领域取得了显著进展，展现出在多种任务上的 impressive 表现。尽管在LLMs方面取得了重大进展，其有效运行仍然很大程度上依赖于人类输入来准确引导对话流程，代理调优是一种关键的优化技术，涉及对模型进行人工调整以更好地应对此类任务。为减少这种依赖，我们提出了TinyAgent模型，该模型基于精心筛选的高质量数据集进行训练。我们还提出了协作多代理调优（CMAT）框架，这是一种创新系统，通过基于环境反馈的自适应权重更新来增强语言代理的能力。该框架促进了多个智能代理之间的协作学习和实时适应，增强它们的上下文感知能力和长期记忆。在本研究中，我们提出了一种新的通信代理框架，将多代理系统与环境反馈机制相结合，提供了一种可扩展的方法来探索合作行为。值得注意的是，我们的TinyAgent-7B模型在参数量较少的情况下，性能与GPT-3.5相当，这表明在LLMs的效率和有效性方面取得了显著改进。 

---
