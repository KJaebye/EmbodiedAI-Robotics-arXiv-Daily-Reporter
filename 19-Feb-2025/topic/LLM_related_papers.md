# GSCE: A Prompt Framework with Enhanced Reasoning for Reliable LLM-driven Drone Control 

**Title (ZH)**: GSCE: 一种增强推理的提示框架，用于可靠的LLM驱动无人机控制 

**Authors**: Wenhao Wang, Yanyan Li, Long Jiao, Jiawei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12531)  

**Abstract**: The integration of Large Language Models (LLMs) into robotic control, including drones, has the potential to revolutionize autonomous systems. Research studies have demonstrated that LLMs can be leveraged to support robotic operations. However, when facing tasks with complex reasoning, concerns and challenges are raised about the reliability of solutions produced by LLMs. In this paper, we propose a prompt framework with enhanced reasoning to enable reliable LLM-driven control for drones. Our framework consists of novel technical components designed using Guidelines, Skill APIs, Constraints, and Examples, namely GSCE. GSCE is featured by its reliable and constraint-compliant code generation. We performed thorough experiments using GSCE for the control of drones with a wide level of task complexities. Our experiment results demonstrate that GSCE can significantly improve task success rates and completeness compared to baseline approaches, highlighting its potential for reliable LLM-driven autonomous drone systems. 

**Abstract (ZH)**: 大型语言模型在机器人控制中的集成，包括无人机，有望革新自主系统。研究显示，大型语言模型可以支持机器人操作，但在面对复杂推理任务时，关于其解决方案可靠性的担忧和挑战也随之而来。本文提出了一种增强推理的提示框架，以实现可靠的基于大型语言模型的无人机控制。该框架包括使用指南、技能API、约束和示例设计的新型技术组件，简称GSCE。GSCE的特点是其可靠的并符合约束条件的代码生成。我们使用GSCE对具有广泛复杂程度任务的无人机控制进行了细致的实验。实验结果表明，与基线方法相比，GSCE显著提高了任务的成功率和完整性，突显了其在可靠的大规模语言模型驱动的自主无人机系统中的潜力。 

---
# AIDE: AI-Driven Exploration in the Space of Code 

**Title (ZH)**: AI驱动的代码空间探索方法 

**Authors**: Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, Yuxiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13138)  

**Abstract**: Machine learning, the foundation of modern artificial intelligence, has driven innovations that have fundamentally transformed the world. Yet, behind advancements lies a complex and often tedious process requiring labor and compute intensive iteration and experimentation. Engineers and scientists developing machine learning models spend much of their time on trial-and-error tasks instead of conceptualizing innovative solutions or research hypotheses. To address this challenge, we introduce AI-Driven Exploration (AIDE), a machine learning engineering agent powered by large language models (LLMs). AIDE frames machine learning engineering as a code optimization problem, and formulates trial-and-error as a tree search in the space of potential solutions. By strategically reusing and refining promising solutions, AIDE effectively trades computational resources for enhanced performance, achieving state-of-the-art results on multiple machine learning engineering benchmarks, including our Kaggle evaluations, OpenAI MLE-Bench and METRs RE-Bench. 

**Abstract (ZH)**: 驱动现代人工智能的机器学习，推动了从根本上转变世界的创新。然而，在进步的背后是一个复杂且often tedious的过程，需要大量的劳动和计算密集型迭代和实验。开发机器学习模型的工程师和科学家们花费大量时间在试错任务上，而不是进行创新性概念构思或研究假说。为应对这一挑战，我们提出了AI驱动探索（AIDE），这是一种由大规模语言模型（LLMs）驱动的机器学习工程代理。AIDE将机器学习工程视为代码优化问题，并将试错过程表述为空间中潜在解决方案的树搜索。通过战略性地重用和改进有希望的解决方案，AIDE有效地用计算资源换取性能提升，在包括我们的Kaggle评估、OpenAI MLE-Bench和METRS RE-Bench等多个机器学习工程基准测试中取得了最先进的结果。 

---
# Theorem Prover as a Judge for Synthetic Data Generation 

**Title (ZH)**: 定理证明器作为合成数据生成的裁判 

**Authors**: Joshua Ong Jun Leang, Giwon Hong, Wenda Li, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13137)  

**Abstract**: The demand for synthetic data in mathematical reasoning has increased due to its potential to enhance the mathematical capabilities of large language models (LLMs). However, ensuring the validity of intermediate reasoning steps remains a significant challenge, affecting data quality. While formal verification via theorem provers effectively validates LLM reasoning, the autoformalisation of mathematical proofs remains error-prone. In response, we introduce iterative autoformalisation, an approach that iteratively refines theorem prover formalisation to mitigate errors, thereby increasing the execution rate on the Lean prover from 60% to 87%. Building upon that, we introduce Theorem Prover as a Judge (TP-as-a-Judge), a method that employs theorem prover formalisation to rigorously assess LLM intermediate reasoning, effectively integrating autoformalisation with synthetic data generation. Finally, we present Reinforcement Learning from Theorem Prover Feedback (RLTPF), a framework that replaces human annotation with theorem prover feedback in Reinforcement Learning from Human Feedback (RLHF). Across multiple LLMs, applying TP-as-a-Judge and RLTPF improves benchmarks with only 3,508 samples, achieving 5.56% accuracy gain on Mistral-7B for MultiArith, 6.00% on Llama-2-7B for SVAMP, and 3.55% on Llama-3.1-8B for AQUA. 

**Abstract (ZH)**: 合成数据在数学推理中的需求由于其增强大型语言模型数学能力的潜力而增加，但确保中间推理步骤的有效性仍然是一个重大挑战，影响数据质量。尽管使用定理证明器的形式验证可以有效验证大型语言模型的推理，但数学证明的形式化仍然容易出错。针对这一问题，我们引入了迭代形式化方法，该方法通过逐步细化定理证明器的形式化表述以减少错误，从而将Lean证明器的执行率从60%提高到87%。在此基础上，我们引入了“证明助手作为裁判”（TP-as-a-Judge）方法，该方法利用定理证明器的形式化表述严格评估大型语言模型的中间推理，有效地将形式化与合成数据生成集成在一起。最后，我们提出了从证明助手反馈强化学习的框架（RLTPF），该框架用证明助手反馈替换人机反馈强化学习（RLHF）中的人类标注。在多种大型语言模型中，采用TP-as-a-Judge和RLTPF仅需3,508个样本就能提升基准测试成绩，分别在Mistral-7B的MultiArith上实现了5.56%的准确率提升，在Llama-2-7B的SVAMP上实现了6.00%的提升，在Llama-3.1-8B的AQUA上实现了3.55%的提升。 

---
# Rethinking Diverse Human Preference Learning through Principal Component Analysis 

**Title (ZH)**: 基于主成分分析重新思考多元人类偏好学习 

**Authors**: Feng Luo, Rui Yang, Hao Sun, Chunyuan Deng, Jiarui Yao, Jingyan Shen, Huan Zhang, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13131)  

**Abstract**: Understanding human preferences is crucial for improving foundation models and building personalized AI systems. However, preferences are inherently diverse and complex, making it difficult for traditional reward models to capture their full range. While fine-grained preference data can help, collecting it is expensive and hard to scale. In this paper, we introduce Decomposed Reward Models (DRMs), a novel approach that extracts diverse human preferences from binary comparisons without requiring fine-grained annotations. Our key insight is to represent human preferences as vectors and analyze them using Principal Component Analysis (PCA). By constructing a dataset of embedding differences between preferred and rejected responses, DRMs identify orthogonal basis vectors that capture distinct aspects of preference. These decomposed rewards can be flexibly combined to align with different user needs, offering an interpretable and scalable alternative to traditional reward models. We demonstrate that DRMs effectively extract meaningful preference dimensions (e.g., helpfulness, safety, humor) and adapt to new users without additional training. Our results highlight DRMs as a powerful framework for personalized and interpretable LLM alignment. 

**Abstract (ZH)**: 理解人类偏好对于改善基础模型和构建个性化AI系统至关重要。然而，偏好本身是多样且复杂的，传统奖励模型难以捕捉其全部范围。虽然细粒度的偏好数据有助于提高模型性能，但其收集成本高且难以扩展。本文介绍了一种新的Decomposed Reward Models (DRMs) 方法，该方法通过二元比较提取多样化的用户偏好，而不需要细粒度注释。我们的核心洞察是将人类偏好表示为向量，并使用主成分分析（PCA）进行分析。通过构建偏好和非偏好响应嵌入差异的数据集，DRMs识别出能够捕捉偏好不同方面的正交基向量。这些分解后的奖励可以灵活组合，以满足不同的用户需求，提供了一种可解释且可扩展的传统奖励模型的替代方案。我们证明了DRMs能够有效提取有意义的偏好维度（如帮助性、安全性、趣味性），并且可以在无需额外训练的情况下适应新用户。实验结果突出了DRMs作为个性化和可解释的大语言模型对齐的强大框架地位。 

---
# MatterChat: A Multi-Modal LLM for Material Science 

**Title (ZH)**: MatterChat: 一种材料科学多模态大语言模型 

**Authors**: Yingheng Tang, Wenbin Xu, Jie Cao, Jianzhu Ma, Weilu Gao, Steve Farrell, Benjamin Erichson, Michael W. Mahoney, Andy Nonaka, Zhi Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13107)  

**Abstract**: Understanding and predicting the properties of inorganic materials is crucial for accelerating advancements in materials science and driving applications in energy, electronics, and beyond. Integrating material structure data with language-based information through multi-modal large language models (LLMs) offers great potential to support these efforts by enhancing human-AI interaction. However, a key challenge lies in integrating atomic structures at full resolution into LLMs. In this work, we introduce MatterChat, a versatile structure-aware multi-modal LLM that unifies material structural data and textual inputs into a single cohesive model. MatterChat employs a bridging module to effectively align a pretrained machine learning interatomic potential with a pretrained LLM, reducing training costs and enhancing flexibility. Our results demonstrate that MatterChat significantly improves performance in material property prediction and human-AI interaction, surpassing general-purpose LLMs such as GPT-4. We also demonstrate its usefulness in applications such as more advanced scientific reasoning and step-by-step material synthesis. 

**Abstract (ZH)**: 理解并预测无机材料的性质对于加速材料科学的发展以及推动能源、电子等领域应用至关重要。通过多模态大型语言模型（LLMs）将材料结构数据与基于语言的信息集成，有助于通过增强人机交互来支持这些努力。然而，关键挑战在于将原子结构以全分辨率集成到LLMs中。在本工作中，我们引入了MatterChat，这是一种多功能的结构感知多模态LLM，将材料结构数据和文本输入统一到一个单一的协同模型中。MatterChat采用了一个桥接模块，有效地将预训练的机器学习原子间势能与预训练的LLM对齐，从而降低训练成本并增强灵活性。我们的结果表明，MatterChat在材料性质预测和人机交互方面显著提高了性能，超越了通用的LLM如GPT-4。我们还展示了其在更高级的科学推理和材料合成步骤中的应用价值。 

---
# Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger 

**Title (ZH)**: 大型语言模型中的元认知触发适应性工具使用 

**Authors**: Wenjun Li, Dexun Li, Kuicai Dong, Cong Zhang, Hao Zhang, Weiwen Liu, Yasheng Wang, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12961)  

**Abstract**: Large language models (LLMs) have shown remarkable emergent capabilities, transforming the execution of functional tasks by leveraging external tools for complex problems that require specialized processing or real-time data. While existing research expands LLMs access to diverse tools (e.g., program interpreters, search engines, weather/map apps), the necessity of using these tools is often overlooked, leading to indiscriminate tool invocation. This naive approach raises two key issues:(1) increased delays due to unnecessary tool calls, and (2) potential errors resulting from faulty interactions with external tools. In this paper, we introduce meta-cognition as a proxy for LLMs self-assessment of their capabilities, representing the model's awareness of its own limitations. Based on this, we propose MeCo, an adaptive decision-making strategy for external tool use. MeCo quantifies metacognitive scores by capturing high-level cognitive signals in the representation space, guiding when to invoke tools. Notably, MeCo is fine-tuning-free and incurs minimal cost. Our experiments show that MeCo accurately detects LLMs' internal cognitive signals and significantly improves tool-use decision-making across multiple base models and benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了显著的 emergent 能力，通过利用外部工具来执行复杂的任务，这些任务需要专门的处理或实时数据。现有研究虽然扩展了 LLMs 对多样工具的访问（如程序解释器、搜索引擎、天气/地图应用程序），但经常忽视使用这些工具的必要性，导致了工具调用的随意性。这种不成熟的调用方式引发了两个关键问题：（1）由于不必要的工具调用增加了延迟，（2）潜在错误由于与外部工具的错误交互所致。在本文中，我们引入元认知作为 LLMs 自我评估其能力的代理，代表模型对其自身局限性的意识。基于此，我们提出了 MeCo，一种针对外部工具使用的自适应决策策略。MeCo 通过捕捉表示空间中的高层认知信号来量化元认知得分，指导何时调用工具。值得注意的是，MeCo 不依赖于微调且成本低廉。我们的实验表明，MeCo 准确检测了 LLMs 的内部认知信号，并显著改善了多种基础模型和基准上的工具使用决策。 

---
# Towards more Contextual Agents: An extractor-Generator Optimization Framework 

**Title (ZH)**: 更加情境化的代理：一种提取-生成优化框架 

**Authors**: Mourad Aouini, Jinan Loubani  

**Link**: [PDF](https://arxiv.org/pdf/2502.12926)  

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable success in solving complex tasks across a wide range of general-purpose applications. However, their performance often degrades in context-specific scenarios, such as specialized industries or research domains, where the absence of domain-relevant knowledge leads to imprecise or suboptimal outcomes. To address this challenge, our work introduces a systematic approach to enhance the contextual adaptability of LLM-based agents by optimizing their underlying prompts-critical components that govern agent behavior, roles, and interactions. Manually crafting optimized prompts for context-specific tasks is labor-intensive, error-prone, and lacks scalability. In this work, we introduce an Extractor-Generator framework designed to automate the optimization of contextual LLM-based agents. Our method operates through two key stages: (i) feature extraction from a dataset of gold-standard input-output examples, and (ii) prompt generation via a high-level optimization strategy that iteratively identifies underperforming cases and applies self-improvement techniques. This framework substantially improves prompt adaptability by enabling more precise generalization across diverse inputs, particularly in context-specific tasks where maintaining semantic consistency and minimizing error propagation are critical for reliable performance. Although developed with single-stage workflows in mind, the approach naturally extends to multi-stage workflows, offering broad applicability across various agent-based systems. Empirical evaluations demonstrate that our framework significantly enhances the performance of prompt-optimized agents, providing a structured and efficient approach to contextual LLM-based agents. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理在通用应用领域解决复杂任务方面取得了显著成功，但在特定情境场景中，如专门行业或研究领域，由于缺乏领域相关知识，其性能往往下降至不精确或次优化状态。为解决这一挑战，我们的工作引入了一种系统方法，通过优化底层提示来增强基于LLM的代理的上下文适应性，提示是决定代理行为、角色和交互的关键组件。手动为特定情境任务设计优化提示既耗时又容易出错，且缺乏可扩展性。在本工作中，我们引入了一种提取-生成框架，旨在自动化优化上下文特定的LLM代理。该方法包括两个关键阶段：（i）从金标准输入-输出示例数据集中提取特征；（ii）通过高级优化策略生成提示，该策略迭代识别性能不佳的情况并应用自我改进技术。该框架通过使提示能够更精确地泛化到各种输入，特别是在需要维持语义一致性和最小化错误传播的特定情境任务中，显著提高了提示适应性。尽管该方法最初开发时针对单阶段工作流，但它自然地扩展到多阶段工作流，具有广泛的应用前景。实证评估表明，该框架显著提升了优化提示代理的性能，为上下文特定的LLM代理提供了一种结构化和高效的方法。 

---
# Continuous Learning Conversational AI: A Personalized Agent Framework via A2C Reinforcement Learning 

**Title (ZH)**: 连续学习对话AI：基于A2C强化学习的个性化代理框架 

**Authors**: Nandakishor M, Anjali M  

**Link**: [PDF](https://arxiv.org/pdf/2502.12876)  

**Abstract**: Creating personalized and adaptable conversational AI remains a key challenge. This paper introduces a Continuous Learning Conversational AI (CLCA) approach, implemented using A2C reinforcement learning, to move beyond static Large Language Models (LLMs). We use simulated sales dialogues, generated by LLMs, to train an A2C agent. This agent learns to optimize conversation strategies for personalization, focusing on engagement and delivering value. Our system architecture integrates reinforcement learning with LLMs for both data creation and response selection. This method offers a practical way to build personalized AI companions that evolve through continuous learning, advancing beyond traditional static LLM techniques. 

**Abstract (ZH)**: 创建个性化和自适应的对话AI仍然是一个关键挑战。本文介绍了一种连续学习对话AI（CLCA）方法，采用A2C强化学习实现，以超越静态大型语言模型（LLMs）。 

---
# Towards Adaptive Feedback with AI: Comparing the Feedback Quality of LLMs and Teachers on Experimentation Protocols 

**Title (ZH)**: 面向AI自适应反馈的研究：基于实验协议比较大型语言模型与教师的反馈质量 

**Authors**: Kathrin Seßler, Arne Bewersdorff, Claudia Nerdel, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12842)  

**Abstract**: Effective feedback is essential for fostering students' success in scientific inquiry. With advancements in artificial intelligence, large language models (LLMs) offer new possibilities for delivering instant and adaptive feedback. However, this feedback often lacks the pedagogical validation provided by real-world practitioners. To address this limitation, our study evaluates and compares the feedback quality of LLM agents with that of human teachers and science education experts on student-written experimentation protocols. Four blinded raters, all professionals in scientific inquiry and science education, evaluated the feedback texts generated by 1) the LLM agent, 2) the teachers and 3) the science education experts using a five-point Likert scale based on six criteria of effective feedback: Feed Up, Feed Back, Feed Forward, Constructive Tone, Linguistic Clarity, and Technical Terminology. Our results indicate that LLM-generated feedback shows no significant difference to that of teachers and experts in overall quality. However, the LLM agent's performance lags in the Feed Back dimension, which involves identifying and explaining errors within the student's work context. Qualitative analysis highlighted the LLM agent's limitations in contextual understanding and in the clear communication of specific errors. Our findings suggest that combining LLM-generated feedback with human expertise can enhance educational practices by leveraging the efficiency of LLMs and the nuanced understanding of educators. 

**Abstract (ZH)**: 有效的反馈对于促进学生在科学探究中的成功至关重要。随着人工智能的进步，大型语言模型（LLMs）为即时和自适应反馈提供了新可能性。然而，这种反馈往往缺乏由实际从业者提供的教学验证。为解决这一局限，我们的研究评估并对比了LLM代理与人类教师及科学教育专家在评价学生撰写的实验方案时提供的反馈质量。四位盲评专家，均为科学探究和科学教育领域的专业人士，根据有效反馈的六个标准（Feed Up、Feed Back、Feed Forward、建设性语气、语言清晰度和技术术语），使用五点李克特量表对由1）LLM代理、2）教师和3）科学教育专家生成的反馈文本进行评估。研究结果表明，LLM生成的反馈在总体质量上与教师和专家的反馈无显著差异。然而，LLM代理在Feed Back维度的表现较弱，即在识别和解释学生工作中错误方面表现不佳。定性分析指出，LLM代理在情境理解和具体错误的清晰传达方面存在局限性。研究结果表明，结合LLM生成的反馈和人类专家的见解可以优化教育实践，充分利用LLMs的高效性与教育者的细致理解。 

---
# Perovskite-LLM: Knowledge-Enhanced Large Language Models for Perovskite Solar Cell Research 

**Title (ZH)**: Perovskite-LLM：知识增强的大语言模型在钙钛矿太阳能电池研究中的应用 

**Authors**: Xiang Liu, Penglei Sun, Shuyan Chen, Longhan Zhang, Peijie Dong, Huajie You, Yongqi Zhang, Chang Yan, Xiaowen Chu, Tong-yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12669)  

**Abstract**: The rapid advancement of perovskite solar cells (PSCs) has led to an exponential growth in research publications, creating an urgent need for efficient knowledge management and reasoning systems in this domain. We present a comprehensive knowledge-enhanced system for PSCs that integrates three key components. First, we develop Perovskite-KG, a domain-specific knowledge graph constructed from 1,517 research papers, containing 23,789 entities and 22,272 relationships. Second, we create two complementary datasets: Perovskite-Chat, comprising 55,101 high-quality question-answer pairs generated through a novel multi-agent framework, and Perovskite-Reasoning, containing 2,217 carefully curated materials science problems. Third, we introduce two specialized large language models: Perovskite-Chat-LLM for domain-specific knowledge assistance and Perovskite-Reasoning-LLM for scientific reasoning tasks. Experimental results demonstrate that our system significantly outperforms existing models in both domain-specific knowledge retrieval and scientific reasoning tasks, providing researchers with effective tools for literature review, experimental design, and complex problem-solving in PSC research. 

**Abstract (ZH)**: Rapid进展的钙钛矿太阳能电池(PSCs)的研究出版物呈指数增长，迫切需要有效的知识管理与推理系统。我们提出了一种全面的知识增强系统，整合了三个关键组件。首先，我们开发了Perovskite-KG，这是一个由1,517篇研究论文构建的领域特定知识图谱，包含23,789个实体和22,272个关系。其次，我们创建了两个互补的数据集：Perovskite-Chat，包含55,101对高质量的问答对，通过一种新颖的多代理框架生成；和Perovskite-Reasoning，包含2,217个精心挑选的材料科学问题。第三，我们引入了两种专门的大语言模型：Perovskite-Chat-LLM为领域特定知识提供支持，以及Perovskite-Reasoning-LLM用于科学推理任务。实验结果表明，我们的系统在领域特定知识检索和科学推理任务中显著优于现有模型，为研究人员提供了有效的工具，用于文献回顾、实验设计和PSC研究中的复杂问题解决。 

---
# RM-PoT: Reformulating Mathematical Problems and Solving via Program of Thoughts 

**Title (ZH)**: RM-PoT: 重新表述数学问题并通过思维程序求解 

**Authors**: Yu Zhang, Shujun Peng, Nengwu Wu, Xinhan Lin, Yang Hu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12589)  

**Abstract**: Recently, substantial advancements have been made in training language models to carry out step-by-step reasoning for solving intricate numerical reasoning tasks. Beyond the methods used to solve these problems, the structure and formulation of the problems themselves also play a crucial role in determining the performance of large language models. We observe that even small changes in the surface form of mathematical problems can have a profound impact on both the answer distribution and solve rate. This highlights the vulnerability of LLMs to surface-level variations, revealing its limited robustness when reasoning through complex problems. In this paper, we propose RM-PoT, a three-stage framework that integrates problem reformulation (RM), code-aided reasoning (PoT), and domain-aware few-shot learning to address these limitations. Our approach first reformulates the input problem into diverse surface forms to reduce structural bias, then retrieves five semantically aligned examples from a pre-constructed domain-specific question bank to provide contextual guidance, and finally generates executable Python code for precise computation. 

**Abstract (ZH)**: 最近，训练语言模型进行复杂数值推理的逐步推理方面取得了显著进展。除了解决这些问题所使用的方法外，问题本身的结构和表述形式也对大型语言模型的性能起到了关键作用。我们观察到，即使是数学问题表面形式的小变化也会影响答案分布和解题率，这表明LLMs对表面级变化的脆弱性，揭示了其在处理复杂问题时的有限鲁棒性。在这篇论文中，我们提出了RM-PoT，这是一种结合问题重述(RM)、代码辅助推理(PoT)和领域意识的少样本学习的三阶段框架，以应对这些限制。我们的方法首先将输入问题重新表述为多种表面形式以减少结构偏见，然后从预先构建的领域特定问题库中检索五个语义对齐的示例以提供上下文指导，最后生成可执行的Python代码以实现精确计算。 

---
# Exploring the Impact of Personality Traits on LLM Bias and Toxicity 

**Title (ZH)**: 探究人格特质对大语言模型偏差和毒性的影响 

**Authors**: Shuo Wang, Renhao Li, Xi Chen, Yulin Yuan, Derek F. Wong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12566)  

**Abstract**: With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation. 

**Abstract (ZH)**: 基于不同人格特质赋予大型语言模型的不同个性对其输出的偏见和毒性影响研究 

---
# Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights 

**Title (ZH)**: Inference时计算在大语言模型推理与规划中的应用：一个基准与见解 

**Authors**: Shubham Parashar, Blake Olson, Sambhav Khurana, Eric Li, Hongyi Ling, James Caverlee, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.12521)  

**Abstract**: We examine the reasoning and planning capabilities of large language models (LLMs) in solving complex tasks. Recent advances in inference-time techniques demonstrate the potential to enhance LLM reasoning without additional training by exploring intermediate steps during inference. Notably, OpenAI's o1 model shows promising performance through its novel use of multi-step reasoning and verification. Here, we explore how scaling inference-time techniques can improve reasoning and planning, focusing on understanding the tradeoff between computational cost and performance. To this end, we construct a comprehensive benchmark, known as Sys2Bench, and perform extensive experiments evaluating existing inference-time techniques on eleven diverse tasks across five categories, including arithmetic reasoning, logical reasoning, common sense reasoning, algorithmic reasoning, and planning. Our findings indicate that simply scaling inference-time computation has limitations, as no single inference-time technique consistently performs well across all reasoning and planning tasks. 

**Abstract (ZH)**: 我们考察了大型语言模型（LLMs）在解决复杂任务中的推理和规划能力。通过在推理过程中探索中间步骤，最近在推理时的技术进步展示了在无需额外训练的情况下增强LLM推理的能力。值得注意的是，OpenAI的o1模型通过其新颖的多步推理和验证展示了令人 promise 的性能。在这里，我们探讨了如何通过扩展推理时技术来提高推理和规划能力，重点关注计算成本与性能之间的权衡。为此，我们构建了一个全面的基准，称为Sys2Bench，并在五个类别（包括算术推理、逻辑推理、常识推理、算法推理和规划）下的 eleven 个不同任务上进行了广泛的实验，评估现有的推理时技术。研究发现，仅仅扩展推理时的计算存在局限性，因为没有任何一种推理时技术能够在所有推理和规划任务中表现优异。 

---
# Boost, Disentangle, and Customize: A Robust System2-to-System1 Pipeline for Code Generation 

**Title (ZH)**: 增强、解耦和定制：一种稳健的系统2到系统1代码生成管道 

**Authors**: Kounianhua Du, Hanjing Wang, Jianxing Liu, Jizheng Chen, Xinyi Dai, Yasheng Wang, Ruiming Tang, Yong Yu, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12492)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various domains, particularly in system 1 tasks, yet the intricacies of their problem-solving mechanisms in system 2 tasks are not sufficiently explored. Recent research on System2-to-System1 methods surge, exploring the System 2 reasoning knowledge via inference-time computation and compressing the explored knowledge into System 1 process. In this paper, we focus on code generation, which is a representative System 2 task, and identify two primary challenges: (1) the complex hidden reasoning processes and (2) the heterogeneous data distributions that complicate the exploration and training of robust LLM solvers. To tackle these issues, we propose a novel BDC framework that explores insightful System 2 knowledge of LLMs using a MC-Tree-Of-Agents algorithm with mutual \textbf{B}oosting, \textbf{D}isentangles the heterogeneous training data for composable LoRA-experts, and obtain \textbf{C}ustomized problem solver for each data instance with an input-aware hypernetwork to weight over the LoRA-experts, offering effectiveness, flexibility, and robustness. This framework leverages multiple LLMs through mutual verification and boosting, integrated into a Monte-Carlo Tree Search process enhanced by reflection-based pruning and refinement. Additionally, we introduce the DisenLora algorithm, which clusters heterogeneous data to fine-tune LLMs into composable Lora experts, enabling the adaptive generation of customized problem solvers through an input-aware hypernetwork. This work lays the groundwork for advancing LLM capabilities in complex reasoning tasks, offering a novel System2-to-System1 solution. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域表现出了非凡的能力，特别是在系统1任务中，然而它们在系统2任务中解决问题机制的复杂性尚未得到充分探索。最近关于系统2到系统1方法的研究激增，通过推理时的计算探索系统2的推理知识，并将探索的知识压缩到系统1的过程中。在本文中，我们重点关注代码生成这一代表性的系统2任务，并识别出两大主要挑战：（1）复杂隐藏的推理过程和（2）异质数据分布使得探索和训练鲁棒的LLM求解器变得困难。为解决这些问题，我们提出了一种新颖的BDC框架，利用MC-树代理算法进行互助增强、异质训练数据的分离以及输入感知超网络为每个数据实例获得定制的问题求解器。该框架通过互验和增强的多个LLM集成到一个通过反思剪枝和细化增强的蒙特卡洛树搜索过程。此外，我们引入了DisenLora算法，将异质数据聚类以微调LLM为可组合的Lora专家，通过输入感知超网络实现定制问题求解器的自适应生成。这项工作为提升LLM在复杂推理任务中的能力奠定了基础，提供了一种新颖的系统2到系统1解决方案。 

---
# Investigating and Extending Homans' Social Exchange Theory with Large Language Model based Agents 

**Title (ZH)**: 基于大型语言模型的代理探究和拓展霍曼斯的社会交换理论 

**Authors**: Lei Wang, Zheqing Zhang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12450)  

**Abstract**: Homans' Social Exchange Theory (SET) is widely recognized as a basic framework for understanding the formation and emergence of human civilizations and social structures. In social science, this theory is typically studied based on simple simulation experiments or real-world human studies, both of which either lack realism or are too expensive to control. In artificial intelligence, recent advances in large language models (LLMs) have shown promising capabilities in simulating human behaviors. Inspired by these insights, we adopt an interdisciplinary research perspective and propose using LLM-based agents to study Homans' SET. Specifically, we construct a virtual society composed of three LLM agents and have them engage in a social exchange game to observe their behaviors. Through extensive experiments, we found that Homans' SET is well validated in our agent society, demonstrating the consistency between the agent and human behaviors. Building on this foundation, we intentionally alter the settings of the agent society to extend the traditional Homans' SET, making it more comprehensive and detailed. To the best of our knowledge, this paper marks the first step in studying Homans' SET with LLM-based agents. More importantly, it introduces a novel and feasible research paradigm that bridges the fields of social science and computer science through LLM-based agents. Code is available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的代理研究：Homans的社会交换理论及其扩展 

---
# A Survey on Large Language Models for Automated Planning 

**Title (ZH)**: 大型语言模型在自动规划中的研究综述 

**Authors**: Mohamed Aghzal, Erion Plaku, Gregory J. Stein, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12435)  

**Abstract**: The planning ability of Large Language Models (LLMs) has garnered increasing attention in recent years due to their remarkable capacity for multi-step reasoning and their ability to generalize across a wide range of domains. While some researchers emphasize the potential of LLMs to perform complex planning tasks, others highlight significant limitations in their performance, particularly when these models are tasked with handling the intricacies of long-horizon reasoning. In this survey, we critically investigate existing research on the use of LLMs in automated planning, examining both their successes and shortcomings in detail. We illustrate that although LLMs are not well-suited to serve as standalone planners because of these limitations, they nonetheless present an enormous opportunity to enhance planning applications when combined with other approaches. Thus, we advocate for a balanced methodology that leverages the inherent flexibility and generalized knowledge of LLMs alongside the rigor and cost-effectiveness of traditional planning methods. 

**Abstract (ZH)**: 大型语言模型在自动化规划中的规划能力近年来引起了越来越多的关注，这主要是由于它们在多步推理方面表现出色，并能在广泛的主题领域进行泛化。尽管一些研究人员强调大型语言模型在执行复杂规划任务方面的潜力，但也有人指出，当这些模型被要求处理长周期推理的复杂性时，它们的表现存在显著的局限性。在本次综述中，我们对大型语言模型在自动化规划中的应用进行了批判性评估，详细探讨了它们的成功与不足。我们表明，尽管由于这些局限性，大型语言模型不适合作为独立的规划者，但它们与其它方法结合使用时，仍能提供显著的增强机会。因此，我们建议采用一种平衡的方法，利用大型语言模型固有的灵活性和泛化知识，结合传统规划方法的严谨性和经济性。 

---
# Integrating Expert Knowledge into Logical Programs via LLMs 

**Title (ZH)**: 利用大型语言模型将专家知识集成到逻辑程序中 

**Authors**: Franciszek Górski, Oskar Wysocki, Marco Valentino, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2502.12275)  

**Abstract**: This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma, Mixtral, Mistral, and Qwen. Results reveal that while models generate nearly perfect syntactically correct code, they frequently exhibit logical errors in translating expert knowledge. Furthermore, iterative self-correction yields only marginal improvements (up to 3%). Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered. The complete implementation, along with all relevant data, is available at GitHub. 

**Abstract (ZH)**: 基于ExKLoP的新框架：评估大型语言模型在逻辑推理系统中整合专家知识的有效性 

---
# Accurate Expert Predictions in MoE Inference via Cross-Layer Gate 

**Title (ZH)**: MoE推理中跨层门控的精确专家预测 

**Authors**: Zhiyuan Fang, Zicong Hong, Yuegui Huang, Yufeng Lyu, Wuhui Chen, Yue Yu, Fan Yu, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12224)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, and their application in edge scenarios has attracted significant attention. However, sparse-activated Mixture-of-Experts (MoE) models, which are well suited for edge scenarios, have received relatively little attention due to their high memory demands. Offload-based methods have been proposed to address this challenge, but they face difficulties with expert prediction. Inaccurate expert predictions can result in prolonged inference delays. To promote the application of MoE models in edge scenarios, we propose Fate, an offloading system designed for MoE models to enable efficient inference in resource-constrained environments. The key insight behind Fate is that gate inputs from adjacent layers can be effectively used for expert prefetching, achieving high prediction accuracy without additional GPU overhead. Furthermore, Fate employs a shallow-favoring expert caching strategy that increases the expert hit rate to 99\%. Additionally, Fate integrates tailored quantization strategies for cache optimization and IO efficiency. Experimental results show that, compared to Load on Demand and Expert Activation Path-based method, Fate achieves up to 4.5x and 1.9x speedups in prefill speed and up to 4.1x and 2.2x speedups in decoding speed, respectively, while maintaining inference quality. Moreover, Fate's performance improvements are scalable across different memory budgets. 

**Abstract (ZH)**: 基于卸载的Fate系统：用于边缘场景的高效Mixture-of-Experts模型推理 

---
# Evaluating the Paperclip Maximizer: Are RL-Based Language Models More Likely to Pursue Instrumental Goals? 

**Title (ZH)**: 评估纸钉最大化者：基于 reinforcement learning 的语言模型更倾向于追求工具性目标吗？ 

**Authors**: Yufei He, Yuexin Li, Jiaying Wu, Yuan Sui, Yulin Chen, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12206)  

**Abstract**: As large language models (LLMs) continue to evolve, ensuring their alignment with human goals and values remains a pressing challenge. A key concern is \textit{instrumental convergence}, where an AI system, in optimizing for a given objective, develops unintended intermediate goals that override the ultimate objective and deviate from human-intended goals. This issue is particularly relevant in reinforcement learning (RL)-trained models, which can generate creative but unintended strategies to maximize rewards. In this paper, we explore instrumental convergence in LLMs by comparing models trained with direct RL optimization (e.g., the o1 model) to those trained with reinforcement learning from human feedback (RLHF). We hypothesize that RL-driven models exhibit a stronger tendency for instrumental convergence due to their optimization of goal-directed behavior in ways that may misalign with human intentions. To assess this, we introduce InstrumentalEval, a benchmark for evaluating instrumental convergence in RL-trained LLMs. Initial experiments reveal cases where a model tasked with making money unexpectedly pursues instrumental objectives, such as self-replication, implying signs of instrumental convergence. Our findings contribute to a deeper understanding of alignment challenges in AI systems and the risks posed by unintended model behaviors. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的不断发展，确保其与人类目标和价值保持一致仍然是一个紧迫的挑战。一个关键问题是\textit{工具性趋同}，即在优化特定目标时，AI系统会发展出未预期的中间目标，这些中间目标会凌驾于最终目标之上，并偏离人类期望的目标。这一问题尤其适用于通过强化学习（RL）训练的模型，这些模型可以生成创意但未预期的策略来最大化奖励。在本文中，我们通过比较使用直接RL优化（例如o1模型）训练的模型和使用强化学习从人类反馈中训练的模型（RLHF）来探索LLMs中的工具性趋同。我们假设，由强化学习驱动的模型更容易表现出工具性趋同，因为它们可能会通过不利于人类意图的方式优化目标驱动行为。为评估这一现象，我们引入了InstrumentalEval，一种评估RL训练的LLMs工具性趋同的基准方法。初步实验揭示了模型在任务中赚钱时意外追求工具性目标（如自我复制）的情况，这可能表明存在工具性趋同的迹象。我们的研究结果有助于更深入地理解AI系统中的对齐挑战以及由未预期的模型行为带来的风险。 

---
# UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models 

**Title (ZH)**: UniGuardian: 大型语言模型中检测提示注入、后门攻击和 adversarial 攻击的统一防御方法 

**Authors**: Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.13141)  

**Abstract**: Large Language Models (LLMs) are vulnerable to attacks like prompt injection, backdoor attacks, and adversarial attacks, which manipulate prompts or models to generate harmful outputs. In this paper, departing from traditional deep learning attack paradigms, we explore their intrinsic relationship and collectively term them Prompt Trigger Attacks (PTA). This raises a key question: Can we determine if a prompt is benign or poisoned? To address this, we propose UniGuardian, the first unified defense mechanism designed to detect prompt injection, backdoor attacks, and adversarial attacks in LLMs. Additionally, we introduce a single-forward strategy to optimize the detection pipeline, enabling simultaneous attack detection and text generation within a single forward pass. Our experiments confirm that UniGuardian accurately and efficiently identifies malicious prompts in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）易受提示注入攻击、后门攻击和对抗攻击的威胁，这些攻击会操控提示或模型以生成有害输出。本文从传统深度学习攻击范式出发，探讨它们的本质联系，并统称为提示触发攻击（PTA）。这引发了关键问题：我们能否确定一个提示是无害的还是被污染的？为解决这一问题，我们提出了UniGuardian，这是首个用于检测LLMs中提示注入、后门攻击和对抗攻击的统一防御机制。此外，我们引入了一次前向策略来优化检测管道，能够在单次前向传递中实现攻击检测和文本生成。我们的实验证实，UniGuardian能够准确且高效地识别LLMs中的恶意提示。 

---
# Learning to Defer for Causal Discovery with Imperfect Experts 

**Title (ZH)**: 利用 imperfect 专家进行因果发现的学习性延期方法 

**Authors**: Oscar Clivio, Divyat Mahajan, Perouz Taslakian, Sara Magliacane, Ioannis Mitliagkas, Valentina Zantedeschi, Alexandre Drouin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13132)  

**Abstract**: Integrating expert knowledge, e.g. from large language models, into causal discovery algorithms can be challenging when the knowledge is not guaranteed to be correct. Expert recommendations may contradict data-driven results, and their reliability can vary significantly depending on the domain or specific query. Existing methods based on soft constraints or inconsistencies in predicted causal relationships fail to account for these variations in expertise. To remedy this, we propose L2D-CD, a method for gauging the correctness of expert recommendations and optimally combining them with data-driven causal discovery results. By adapting learning-to-defer (L2D) algorithms for pairwise causal discovery (CD), we learn a deferral function that selects whether to rely on classical causal discovery methods using numerical data or expert recommendations based on textual meta-data. We evaluate L2D-CD on the canonical Tübingen pairs dataset and demonstrate its superior performance compared to both the causal discovery method and the expert used in isolation. Moreover, our approach identifies domains where the expert's performance is strong or weak. Finally, we outline a strategy for generalizing this approach to causal discovery on graphs with more than two variables, paving the way for further research in this area. 

**Abstract (ZH)**: 将大型语言模型等专家知识集成到因果发现算法中在知识未必正确的情况下具有挑战性。专家建议可能与数据驱动的结果相矛盾，其可靠性在不同领域或特定查询中差异显著。现有基于软约束或预测因果关系不一致的方法未能考虑到这些专业知识的变化。为此，我们提出了一种称为L2D-CD的方法，用于评估专家建议的正确性并最优地将这些建议与数据驱动的因果发现结果结合起来。通过将学习推迟（L2D）算法应用于成对因果发现（CD），我们学习了一个推迟函数，该函数根据文本元数据选择是依赖于使用数值数据的经典因果发现方法还是专家建议。我们在标准的Tübingen成对数据集上评估了L2D-CD，并证明其性能优于单独使用因果发现方法和专家建议。此外，我们的方法能够识别专家表现强弱的领域。最后，我们概述了一种策略，用于将此方法泛化到具有更多变量的图的因果发现中，为该领域的进一步研究铺平了道路。 

---
# Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context 

**Title (ZH)**: 适应心理语言学研究对于大规模语言模型：共指语境中的性别包容性语言 

**Authors**: Marion Bartl, Thomas Brendan Murphy, Susan Leavy  

**Link**: [PDF](https://arxiv.org/pdf/2502.13120)  

**Abstract**: Gender-inclusive language is often used with the aim of ensuring that all individuals, regardless of gender, can be associated with certain concepts. While psycholinguistic studies have examined its effects in relation to human cognition, it remains unclear how Large Language Models (LLMs) process gender-inclusive language. Given that commercial LLMs are gaining an increasingly strong foothold in everyday applications, it is crucial to examine whether LLMs in fact interpret gender-inclusive language neutrally, because the language they generate has the potential to influence the language of their users. This study examines whether LLM-generated coreferent terms align with a given gender expression or reflect model biases. Adapting psycholinguistic methods from French to English and German, we find that in English, LLMs generally maintain the antecedent's gender but exhibit underlying masculine bias. In German, this bias is much stronger, overriding all tested gender-neutralization strategies. 

**Abstract (ZH)**: 性别包容语言的使用旨在确保所有个体，不论性别，都能与某些概念相关联。虽然心理语言学研究已经探讨了其对人类认知的影响，但尚未明确大型语言模型（LLMs）如何处理性别包容语言。鉴于商用LLMs在日常应用中的影响不断增强，有必要考察LLMs是否实际上以中立的方式解释性别包容语言，因为它们生成的语言有可能影响用户语言。本研究探讨LLM生成的指称词是否与给定的性别表达一致，或反映模型偏见。将心理语言学方法从法语和德语适应到英语，我们发现，在英语中，LLMs通常保持先行词的性别，但表现出潜在的男性偏见。在德语中，这种偏见更为强烈，甚至超越了所有测试的性别中性化策略。 

---
# Performance Evaluation of Large Language Models in Statistical Programming 

**Title (ZH)**: 大型语言模型在统计编程中的性能评估 

**Authors**: Xinyi Song, Kexin Xie, Lina Lee, Ruizhe Chen, Jared M. Clark, Hao He, Haoran He, Jie Min, Xinlei Zhang, Simin Zheng, Zhiyang Zhang, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.13117)  

**Abstract**: The programming capabilities of large language models (LLMs) have revolutionized automatic code generation and opened new avenues for automatic statistical analysis. However, the validity and quality of these generated codes need to be systematically evaluated before they can be widely adopted. Despite their growing prominence, a comprehensive evaluation of statistical code generated by LLMs remains scarce in the literature. In this paper, we assess the performance of LLMs, including two versions of ChatGPT and one version of Llama, in the domain of SAS programming for statistical analysis. Our study utilizes a set of statistical analysis tasks encompassing diverse statistical topics and datasets. Each task includes a problem description, dataset information, and human-verified SAS code. We conduct a comprehensive assessment of the quality of SAS code generated by LLMs through human expert evaluation based on correctness, effectiveness, readability, executability, and the accuracy of output results. The analysis of rating scores reveals that while LLMs demonstrate usefulness in generating syntactically correct code, they struggle with tasks requiring deep domain understanding and may produce redundant or incorrect results. This study offers valuable insights into the capabilities and limitations of LLMs in statistical programming, providing guidance for future advancements in AI-assisted coding systems for statistical analysis. 

**Abstract (ZH)**: 大型语言模型的编程能力彻底变革了自动代码生成，并开辟了自动统计分析的新途径。然而，在这些生成的代码被广泛采用之前，需要对其有效性和质量进行系统的评估。尽管大型语言模型在统计编程领域的 prominence 越来越大，但在文献中对其生成的统计代码进行全面评价的信息仍相当缺乏。本文评估了包括两个版本的 ChatGPT 和一个版本的 Llama 在统计分析领域（使用 SAS 编程）中的表现。我们的研究利用了一组涵盖多种统计主题和数据集的统计分析任务。每个任务包括问题描述、数据集信息以及人工验证的 SAS 代码。我们通过人工专家根据正确性、有效性、可读性、可执行性和输出结果的准确性对大型语言模型生成的 SAS 代码的质量进行了全面评估。评分分析表明，尽管大型语言模型在生成语法正确的代码方面表现出一定的有用性，但在需要深厚领域理解的任务中却表现不佳，并且可能会产生冗余或错误的结果。本文为 AI 辅助编码系统在统计分析中的未来发展提供了有关大型语言模型在统计编程中的能力和局限性的宝贵见解。 

---
# Text2World: Benchmarking Large Language Models for Symbolic World Model Generation 

**Title (ZH)**: Text2World: 用于符号世界模型生成的大规模语言模型基准测试 

**Authors**: Mengkang Hu, Tianxing Chen, Yude Zou, Yuheng Lei, Qiguang Chen, Ming Li, Hongyuan Zhang, Wenqi Shao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.13092)  

**Abstract**: Recently, there has been growing interest in leveraging large language models (LLMs) to generate symbolic world models from textual descriptions. Although LLMs have been extensively explored in the context of world modeling, prior studies encountered several challenges, including evaluation randomness, dependence on indirect metrics, and a limited domain scope. To address these limitations, we introduce a novel benchmark, Text2World, based on planning domain definition language (PDDL), featuring hundreds of diverse domains and employing multi-criteria, execution-based metrics for a more robust evaluation. We benchmark current LLMs using Text2World and find that reasoning models trained with large-scale reinforcement learning outperform others. However, even the best-performing model still demonstrates limited capabilities in world modeling. Building on these insights, we examine several promising strategies to enhance the world modeling capabilities of LLMs, including test-time scaling, agent training, and more. We hope that Text2World can serve as a crucial resource, laying the groundwork for future research in leveraging LLMs as world models. The project page is available at this https URL. 

**Abstract (ZH)**: 近年来，人们越来越关注利用大规模语言模型（LLMs）从文本描述中生成符号世界模型。尽管在世界建模的背景下已经广泛研究了LLMs，但之前的研究所遇到的一些挑战包括评估的随机性、依赖间接指标以及领域范围有限等。为了解决这些问题，我们基于规划领域定义语言（PDDL）引入了一个新的基准Text2World，该基准包含数百个多样化的领域，并采用多标准执行基指标进行更稳健的评估。我们使用Text2World来评估当前的LLMs，并发现大规模强化学习训练的推理模型优于其他模型。然而，即使性能最好的模型在世界建模方面也表现出有限的能力。基于这些见解，我们探讨了几种有望增强LLMs世界建模能力的方法，包括测试时扩展、智能体训练等。我们希望Text2World能够成为一个重要资源，为利用LLMs作为世界模型的未来研究奠定基础。项目页面可通过此链接访问。 

---
# LAMD: Context-driven Android Malware Detection and Classification with LLMs 

**Title (ZH)**: 基于LLM的上下文驱动的Android恶意软件检测与分类 

**Authors**: Xingzhi Qian, Xinran Zheng, Yiling He, Shuo Yang, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2502.13055)  

**Abstract**: The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes. 

**Abstract (ZH)**: 移动应用程序的迅速增长加剧了Android恶意软件威胁。尽管存在多种检测方法，但它们常常难以应对不断演变的攻击、数据集偏差以及缺乏透明性。大规模语言模型（LLMs）凭借其零样本推断和推理能力提供了有前景的替代方案。然而，将LLMs应用于Android恶意软件检测面临着两个主要挑战：（1）Android应用程序中广泛的支撑代码，常涉及数千个类，超出了LLMs的上下文限制，模糊了恶意行为在良性功能中的表现；（2）Android应用程序的结构复杂性和相互依赖性超过了LLMs基于序列的推理能力，导致代码分析碎片化，妨碍了恶意意图的推断。为解决这些挑战，我们提出了一种实际的基于上下文的框架LAMD，以使LLM能够用于Android恶意软件检测。LAMD结合了关键上下文提取，以分离安全关键代码区域并构建程序结构，然后采用分层代码推理逐步分析应用程序行为，从低级指令到高级语义，提供最终预测和解释。设计了一种事实一致性验证机制，以减轻第一层级的LLM幻觉。在实际环境中的评估表明，LAMD在传统检测器之上更为有效，确立了基于LLM的恶意软件分析在动态威胁环境中的可行性基础。 

---
# LLM-Powered Proactive Data Systems 

**Title (ZH)**: LLM驱动的主动数据系统 

**Authors**: Sepanta Zeighami, Yiming Lin, Shreya Shankar, Aditya Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2502.13016)  

**Abstract**: With the power of LLMs, we now have the ability to query data that was previously impossible to query, including text, images, and video. However, despite this enormous potential, most present-day data systems that leverage LLMs are reactive, reflecting our community's desire to map LLMs to known abstractions. Most data systems treat LLMs as an opaque black box that operates on user inputs and data as is, optimizing them much like any other approximate, expensive UDFs, in conjunction with other relational operators. Such data systems do as they are told, but fail to understand and leverage what the LLM is being asked to do (i.e. the underlying operations, which may be error-prone), the data the LLM is operating on (e.g., long, complex documents), or what the user really needs. They don't take advantage of the characteristics of the operations and/or the data at hand, or ensure correctness of results when there are imprecisions and ambiguities. We argue that data systems instead need to be proactive: they need to be given more agency -- armed with the power of LLMs -- to understand and rework the user inputs and the data and to make decisions on how the operations and the data should be represented and processed. By allowing the data system to parse, rewrite, and decompose user inputs and data, or to interact with the user in ways that go beyond the standard single-shot query-result paradigm, the data system is able to address user needs more efficiently and effectively. These new capabilities lead to a rich design space where the data system takes more initiative: they are empowered to perform optimization based on the transformation operations, data characteristics, and user intent. We discuss various successful examples of how this framework has been and can be applied in real-world tasks, and present future directions for this ambitious research agenda. 

**Abstract (ZH)**: 利用大模型的威力，我们现在有能力查询以前无法查询的数据，包括文本、图像和视频。然而，尽管这种潜力巨大，大多数利用大模型的数据系统依然是反应式的，反映了我们社区希望将大模型映射到已知抽象的需求。大多数数据系统将大模型视为一个不透明的黑盒，以原始形式处理用户输入和数据，并像优化其他近似且昂贵的UDF一样进行优化，结合其他关系操作。这类数据系统会执行指令，但却无法理解或将大模型被要求执行的内容（即底层操作，可能有错误），大模型正在操作的数据（例如，长而复杂的文档），或用户真正的需求。它们没有利用手头的操作和数据的特性，也没有在存在不精确和歧义时保证结果的正确性。我们认为，数据系统需要更加主动：它们需要被赋予更多的控制权——利用大模型的力量——来理解并重新构建用户输入和数据，并决定如何表示和处理操作和数据。通过允许数据系统解析、重写和分解用户输入和数据，或以超越标准单次查询-结果范式的方式与用户交互，数据系统能够更高效、更有效地满足用户需求。这些新功能带来了丰富的设计空间，在这个空间中，数据系统将扮演更加积极的角色：它们能够基于转换操作、数据特性和用户意图进行优化。我们讨论了这一框架在实际任务中取得的成功实例，并提出了这一雄心勃勃的研究议程的未来方向。 

---
# Personalized Top-k Set Queries Over Predicted Scores 

**Title (ZH)**: 基于预测分数的个性化Top-k集合查询 

**Authors**: Sohrab Namazi Nia, Subhodeep Ghosh, Senjuti Basu Roy, Sihem Amer-Yahia  

**Link**: [PDF](https://arxiv.org/pdf/2502.12998)  

**Abstract**: This work studies the applicability of expensive external oracles such as large language models in answering top-k queries over predicted scores. Such scores are incurred by user-defined functions to answer personalized queries over multi-modal data. We propose a generic computational framework that handles arbitrary set-based scoring functions, as long as the functions could be decomposed into constructs, each of which sent to an oracle (in our case an LLM) to predict partial scores. At a given point in time, the framework assumes a set of responses and their partial predicted scores, and it maintains a collection of possible sets that are likely to be the true top-k. Since calling oracles is costly, our framework judiciously identifies the next construct, i.e., the next best question to ask the oracle so as to maximize the likelihood of identifying the true top-k. We present a principled probabilistic model that quantifies that likelihood. We study efficiency opportunities in designing algorithms. We run an evaluation with three large scale datasets, scoring functions, and baselines. Experiments indicate the efficacy of our framework, as it achieves an order of magnitude improvement over baselines in requiring LLM calls while ensuring result accuracy. Scalability experiments further indicate that our framework could be used in large-scale applications. 

**Abstract (ZH)**: 本研究探讨了大型语言模型等昂贵外部先知在回答基于预测分数的个性化多模态数据查询时的应用性。我们提出了一种通用的计算框架，该框架能够处理任意集基评分函数，只要这些函数可以分解为每个部分发送给先知（例如，大型语言模型）以预测部分评分。在给定时间点上，该框架假设一系列响应及其部分预测得分，并维护一组可能是真正前k名的可能集合。由于调用先知是昂贵的，我们的框架明智地识别出下一个将要提问的构建，即下一个最佳问题，以最大化识别真正前k名的可能性。我们介绍了一个原则性的概率模型来量化这种可能性。我们研究了在设计算法时提高效率的机会。我们在三个大规模数据集、评分函数和基线方法上进行了评估。实验表明，我们的框架在要求大型语言模型调用次数方面比基线方法提高了数量级，同时保持了结果准确性。进一步的可扩展性实验表明，我们的框架适用于大规模应用程序。 

---
# B-cos LM: Efficiently Transforming Pre-trained Language Models for Improved Explainability 

**Title (ZH)**: B-cos LM: 有效地转换预训练语言模型以提高可解释性 

**Authors**: Yifan Wang, Sukrut Rao, Ji-Ung Lee, Mayank Jobanputra, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.12992)  

**Abstract**: Post-hoc explanation methods for black-box models often struggle with faithfulness and human interpretability due to the lack of explainability in current neural models. Meanwhile, B-cos networks have been introduced to improve model explainability through architectural and computational adaptations, but their application has so far been limited to computer vision models and their associated training pipelines. In this work, we introduce B-cos LMs, i.e., B-cos networks empowered for NLP tasks. Our approach directly transforms pre-trained language models into B-cos LMs by combining B-cos conversion and task fine-tuning, improving efficiency compared to previous B-cos methods. Our automatic and human evaluation results demonstrate that B-cos LMs produce more faithful and human interpretable explanations than post hoc methods, while maintaining task performance comparable to conventional fine-tuning. Our in-depth analysis explores how B-cos LMs differ from conventionally fine-tuned models in their learning processes and explanation patterns. Finally, we provide practical guidelines for effectively building B-cos LMs based on our findings. Our code is available at this https URL. 

**Abstract (ZH)**: 黑箱模型的后验解释方法往往由于当前神经模型缺乏可解释性而难以实现忠实性和人类可解释性。同时，B-cos网络已经被引入以通过架构和计算的调整来提高模型的可解释性，但它们的应用迄今仅限于计算机视觉模型及其相关的训练管道。在本项工作中，我们提出了B-cos LMs，即赋能于NLP任务的B-cos网络。我们的方法通过结合B-cos转换和任务微调，直接将预训练语言模型转换为B-cos LMs，相较于之前的B-cos方法提高了效率。我们的自动和人工评估结果表明，B-cos LMs能够产生比后验方法更为忠实和人类可解释的解释，同时保持与传统微调相当的任务性能。我们深入分析了B-cos LMs在学习过程和解释模式上与传统微调模型的区别。最后，基于我们的发现，我们提供了有效构建B-cos LMs的实用指南。相关代码可在以下链接获取。 

---
# Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs 

**Title (ZH)**: Sailor2: 以包容性多语言LLM探索东南亚 

**Authors**: Longxu Dou, Qian Liu, Fan Zhou, Changyu Chen, Zili Wang, Ziqi Jin, Zichen Liu, Tongyao Zhu, Cunxiao Du, Penghui Yang, Haonan Wang, Jiaheng Liu, Yongchi Zhao, Xiachong Feng, Xin Mao, Man Tsung Yeung, Kunat Pipatanakul, Fajri Koto, Min Si Thu, Hynek Kydlíček, Zeyi Liu, Qunshu Lin, Sittipong Sripaisarnmongkol, Kridtaphad Sae-Khow, Nirattisai Thongchim, Taechawat Konkaew, Narong Borijindargoon, Anh Dao, Matichon Maneegard, Phakphum Artkaew, Zheng-Xin Yong, Quan Nguyen, Wannaphong Phatthiyaphaibun, Hoang H. Tran, Mike Zhang, Shiqi Chen, Tianyu Pang, Chao Du, Xinyi Wan, Wei Lu, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12982)  

**Abstract**: Sailor2 is a family of cutting-edge multilingual language models for South-East Asian (SEA) languages, available in 1B, 8B, and 20B sizes to suit diverse applications. Building on Qwen2.5, Sailor2 undergoes continuous pre-training on 500B tokens (400B SEA-specific and 100B replay tokens) to support 13 SEA languages while retaining proficiency in Chinese and English. Sailor2-20B model achieves a 50-50 win rate against GPT-4o across SEA languages. We also deliver a comprehensive cookbook on how to develop the multilingual model in an efficient manner, including five key aspects: data curation, pre-training, post-training, model customization and evaluation. We hope that Sailor2 model (Apache 2.0 license) will drive language development in the SEA region, and Sailor2 cookbook will inspire researchers to build more inclusive LLMs for other under-served languages. 

**Abstract (ZH)**: Sailor2是面向东南亚语言的一系列先进多语言语言模型，提供1B、8B和20B三种规模，以适应多种应用需求。基于Qwen2.5，Sailor2在500亿 token（400亿特定于东南亚地区和100亿重播 token）上进行持续预训练，支持13种东南亚语言，同时保留对中文和英语的专业能力。Sailor2-20B模型在东南亚语言方面与GPT-4o实现平 bureau（50-50胜率）。我们还提供了一整套关于如何高效开发多语言模型的方法指南，包括五个关键方面：数据整理、预训练、后训练、模型定制和评估。我们希望Sailor2模型（采用Apache 2.0许可证）能够促进东南亚地区的语言发展，并希望通过Sailor2套菜书激励研究人员为其他未充分服务的语言构建更加包容性的大语言模型。 

---
# AlignFreeze: Navigating the Impact of Realignment on the Layers of Multilingual Models Across Diverse Languages 

**Title (ZH)**: AlignFreeze: 导航重新对齐对多语言模型各层跨多种语言的影响 

**Authors**: Steve Bakos, Félix Gaschi, David Guzmán, Riddhi More, Kelly Chutong Li, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.12959)  

**Abstract**: Realignment techniques are often employed to enhance cross-lingual transfer in multilingual language models, still, they can sometimes degrade performance in languages that differ significantly from the fine-tuned source language. This paper introduces AlignFreeze, a method that freezes either the layers' lower half or upper half during realignment. Through controlled experiments on 4 tasks, 3 models, and in 35 languages, we find that realignment affects all the layers but can be the most detrimental to the lower ones. Freezing the lower layers can prevent performance degradation. Particularly, AlignFreeze improves Part-of-Speech (PoS) tagging performances in languages where full realignment fails: with XLM-R, it provides improvements of more than one standard deviation in accuracy in seven more languages than full realignment. 

**Abstract (ZH)**: Realignment 技术常用于增强多语言语言模型中的跨语言转移，但有时会在与精细调整源语言差异较大的语言中降低性能。本文介绍了一种名为 AlignFreeze 的方法，该方法在重新对齐过程中冻结层的下半部分或上半部分。通过在 4 个任务、3 个模型和 35 种语言上的受控实验，我们发现重新对齐会影响所有层，但对下层的影响尤为严重。冻结下层可以防止性能下降。特别是在语言全面重新对齐失败时，AlignFreeze 提高了部分词性标注（PoS）的性能：与全面重新对齐相比，使用 XLM-R 在七种更多语言中提供了超过一个标准差的准确率改进。 

---
# Fake It Till You Make It: Using Synthetic Data and Domain Knowledge for Improved Text-Based Learning for LGE Detection 

**Title (ZH)**: 伪装以做到最好：通过合成数据和领域知识提高基于文本的学习方法以增强LGE检测 

**Authors**: Athira J Jacob, Puneet Sharma, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2502.12948)  

**Abstract**: Detection of hyperenhancement from cardiac LGE MRI images is a complex task requiring significant clinical expertise. Although deep learning-based models have shown promising results for the task, they require large amounts of data with fine-grained annotations. Clinical reports generated for cardiac MR studies contain rich, clinically relevant information, including the location, extent and etiology of any scars present. Although recently developed CLIP-based training enables pretraining models with image-text pairs, it requires large amounts of data and further finetuning strategies on downstream tasks. In this study, we use various strategies rooted in domain knowledge to train a model for LGE detection solely using text from clinical reports, on a relatively small clinical cohort of 965 patients. We improve performance through the use of synthetic data augmentation, by systematically creating scar images and associated text. In addition, we standardize the orientation of the images in an anatomy-informed way to enable better alignment of spatial and text features. We also use a captioning loss to enable fine-grained supervision and explore the effect of pretraining of the vision encoder on performance. Finally, ablation studies are carried out to elucidate the contributions of each design component to the overall performance of the model. 

**Abstract (ZH)**: 使用临床报告文本训练心脏LGE检测模型：基于领域知识的方法 

---
# Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models 

**Title (ZH)**: 每一专家都重要：迈向有效的混合专家语言模型知识蒸馏方法 

**Authors**: Gyeongman Kim, Gyouk Chu, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12947)  

**Abstract**: With the emergence of Mixture-of-Experts (MoE), the efficient scaling of model size has accelerated the development of large language models in recent years. However, their high memory requirements prevent their use in resource-constrained environments. While knowledge distillation (KD) has been a proven method for model compression, its application to MoE teacher models remains underexplored. Through our investigation, we discover that non-activated experts in MoE models possess valuable knowledge that benefits student models. We further demonstrate that existing KD methods are not optimal for compressing MoE models, as they fail to leverage this knowledge effectively. To address this, we propose two intuitive MoE-specific KD methods for the first time: Knowledge Augmentation (KA) and Student-Aware Router (SAR), both designed to effectively extract knowledge from all experts. Specifically, KA augments knowledge by sampling experts multiple times, while SAR uses all experts and adjusts the expert weights through router training to provide optimal knowledge. Extensive experiments show that our methods outperform conventional KD methods, demonstrating their effectiveness for MoE teacher models. 

**Abstract (ZH)**: MoE模型特定的知识蒸馏方法：知识扩充与学生感知路由 

---
# Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options 

**Title (ZH)**: 选项流：通过思考选项多样化的提升大语言模型推理能力 

**Authors**: Lakshmi Nair, Ian Trase, Mark Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.12929)  

**Abstract**: We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). FoO enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic system for autonomously solving Machine Learning tasks (AutoML). Our framework outperforms state-of-the-art baselines, achieving improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Beyond classification and regression, we illustrate the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our framework presents significant advancements compared to current state-of-the-art agentic systems for AutoML, due to the benefits of FoO in enforcing diversity in LLM solutions through compressed, explainable representations that also support long-term memory when combined with case-based reasoning. 

**Abstract (ZH)**: Flow-of-Options (FoO): 一种针对大规模语言模型内在偏见的新颖推理方法及其在自主解决机器学习任务中的应用 

---
# Conditioning LLMs to Generate Code-Switched Text: A Methodology Grounded in Naturally Occurring Data 

**Title (ZH)**: 基于自然发生数据的条件生成切换代码文本的方法论 

**Authors**: Maite Heredia, Gorka Labaka, Jeremy Barnes, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2502.12924)  

**Abstract**: Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. This paper presents a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license. 

**Abstract (ZH)**: 代码转换仍是自然语言处理中的一个关键挑战。当前的大语言模型在处理和生成代码转换文本时遇到困难，主要原因是缺乏大规模的代码转换数据集进行训练。本文提出了一种新的利用大语言模型生成代码转换数据的方法，并在英西语言对上进行了测试。我们提出了一种将自然的代码转换句子回译为单一语言英语的方法，并使用生成的平行语料库对大语言模型进行微调，使其能够将单一语言句子转换为代码转换文本。与以前的代码转换生成方法不同，我们的方法以自然的代码转换数据作为起点，使模型能够学习代码转换的自然分布模式，而不仅仅是句法模式。我们通过人类偏好研究、定性错误分析和使用流行自动评价指标进行评估，全面分析了模型的性能。结果表明，我们的方法可以生成流畅的代码转换文本，扩展了代码转换通信的研究机会，并且传统的评价指标与人类对生成代码转换数据质量的判断之间不存在相关性。我们将在CC-BY-NC-SA许可下发布我们的代码和生成的数据集。 

---
# GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning 

**Title (ZH)**: GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning 

**Authors**: Sifan Zhou, Shuo Wang, Zhihang Yuan, Mingjia Shi, Yuzhang Shang, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12913)  

**Abstract**: Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to FP16-based fine-tuning while significantly reducing memory usage (50%). Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices. 

**Abstract (ZH)**: 基于设备端的大语言模型细调框架：GSQ-Tuning 

---
# Soundwave: Less is More for Speech-Text Alignment in LLMs 

**Title (ZH)**: Soundwave: 少即是多的语音-文本对齐方法在LLMs中的应用 

**Authors**: Yuhao Zhang, Zhiheng Liu, Fan Bu, Ruiyu Zhang, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12900)  

**Abstract**: Existing end-to-end speech large language models (LLMs) usually rely on large-scale annotated data for training, while data-efficient training has not been discussed in depth. We focus on two fundamental problems between speech and text: the representation space gap and sequence length inconsistency. We propose Soundwave, which utilizes an efficient training strategy and a novel architecture to address these issues. Results show that Soundwave outperforms the advanced Qwen2-Audio in speech translation and AIR-Bench speech tasks, using only one-fiftieth of the training data. Further analysis shows that Soundwave still retains its intelligence during conversation. The project is available at this https URL. 

**Abstract (ZH)**: 现有的端到端语音大型语言模型通常依赖大规模标注数据进行训练，而数据高效训练尚未得到充分讨论。我们专注于语音与文本之间的两个基本问题：表示空间差距和序列长度不一致。我们提出Soundwave，该方法利用高效的训练策略和新型架构来解决这些问题。结果表明，Soundwave在语音翻译和AIR-Bench语音任务上的表现优于先进的Qwen2-Audio，仅使用其五分之一的训练数据。进一步的分析表明，Soundwave在对话过程中仍能保持其智能。项目详情请参见此链接。 

---
# PAFT: Prompt-Agnostic Fine-Tuning 

**Title (ZH)**: PAFT: 命令无感 fine-tuning 

**Authors**: Chenxing Wei, Yao Shu, Mingwen Ou, Ying Tiffany He, Fei Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12859)  

**Abstract**: While Large Language Models (LLMs) adapt well to downstream tasks after fine-tuning, this adaptability often compromises prompt robustness, as even minor prompt variations can significantly degrade performance. To address this, we propose Prompt-Agnostic Fine-Tuning(PAFT), a simple yet effective approach that dynamically adjusts prompts during fine-tuning. This encourages the model to learn underlying task principles rather than overfitting to specific prompt formulations. PAFT operates in two stages: First, a diverse set of meaningful, synthetic candidate prompts is constructed. Second, during fine-tuning, prompts are randomly sampled from this set to create dynamic training inputs. Extensive experiments across diverse datasets and LLMs demonstrate that models trained with PAFT exhibit strong robustness and generalization across a wide range of prompts, including unseen ones. This enhanced robustness improves both model performance and inference speed while maintaining training efficiency. Ablation studies further confirm the effectiveness of PAFT. 

**Abstract (ZH)**: 面向提示的鲁棒无提示微调(PAFT) 

---
# Rejected Dialects: Biases Against African American Language in Reward Models 

**Title (ZH)**: 拒绝的方言：奖励模型中对非洲美国语言偏见的研究 

**Authors**: Joel Mire, Zubin Trivadi Aysola, Daniel Chechelnitsky, Nicholas Deas, Chrysoula Zerva, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.12858)  

**Abstract**: Preference alignment via reward models helps build safe, helpful, and reliable large language models (LLMs). However, subjectivity in preference judgments and the lack of representative sampling in preference data collection can introduce new biases, hindering reward models' fairness and equity. In this work, we introduce a framework for evaluating dialect biases in reward models and conduct a case study on biases against African American Language (AAL) through several experiments comparing reward model preferences and behavior on paired White Mainstream English (WME) and both machine-translated and human-written AAL corpora. We show that reward models are less aligned with human preferences when processing AAL texts vs. WME ones (-4\% accuracy on average), frequently disprefer AAL-aligned texts vs. WME-aligned ones, and steer conversations toward WME, even when prompted with AAL texts. Our findings provide a targeted analysis of anti-AAL biases at a relatively understudied stage in LLM development, highlighting representational harms and ethical questions about the desired behavior of LLMs concerning AAL. 

**Abstract (ZH)**: 偏好对齐通过奖励模型有助于构建安全、 helpful 和可信赖的大规模语言模型（LLMs）。然而，偏好判断中的主观性以及偏好数据收集中缺乏代表性抽样会引入新的偏差，阻碍奖励模型的公平性和公正性。在本文中，我们介绍了一种评估奖励模型方言偏差的框架，并通过多项实验研究了奖励模型对标准主流英语（WME）和机器翻译及人工撰写的非洲美语（AAL）语料库的偏好和行为，从而探讨反AAL偏差。我们展示了当处理AAL文本而非WME文本时，奖励模型与人类偏好的对齐度较低（平均准确率降低4%），倾向于不喜欢与AAL对齐的文本，即使在收到AAL文本提示时，也会引导对话转向WME。我们的研究结果对LLM发展中相对较少研究的阶段提供了一个针对性的分析，突显了AAL表示损害及关于LLM期望行为的伦理问题。 

---
# MeMo: Towards Language Models with Associative Memory Mechanisms 

**Title (ZH)**: MeMo：兼具关联记忆机制的语言模型 

**Authors**: Fabio Massimo Zanzotto, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Leonardo Ranaldi, Davide Venditti, Federico Ranaldi, Cristina Giannone, Andrea Favalli, Raniero Romagnoli  

**Link**: [PDF](https://arxiv.org/pdf/2502.12851)  

**Abstract**: Memorization is a fundamental ability of Transformer-based Large Language Models, achieved through learning. In this paper, we propose a paradigm shift by designing an architecture to memorize text directly, bearing in mind the principle that memorization precedes learning. We introduce MeMo, a novel architecture for language modeling that explicitly memorizes sequences of tokens in layered associative memories. By design, MeMo offers transparency and the possibility of model editing, including forgetting texts. We experimented with the MeMo architecture, showing the memorization power of the one-layer and the multi-layer configurations. 

**Abstract (ZH)**: 基于Transformer的大型语言模型的记忆是其基本能力，通过学习获得。本文提出一种范式转移，设计一种直接记忆文本的架构，遵循记忆先于学习的原则。我们介绍了MeMo，一种新颖的语言模型架构，明确在分层关联记忆中记忆令牌序列。设计上，MeMo提供了透明度和模型编辑的可能性，包括遗忘文本。我们实验了MeMo架构，展示了单层和多层配置的记忆能力。 

---
# Reasoning and the Trusting Behavior of DeepSeek and GPT: An Experiment Revealing Hidden Fault Lines in Large Language Models 

**Title (ZH)**: DeepSeek和GPT的推理与信任行为：一项揭示大型语言模型潜在裂纹的实验 

**Authors**: Rubing Lu, João Sedoc, Arun Sundararajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12825)  

**Abstract**: When encountering increasingly frequent performance improvements or cost reductions from a new large language model (LLM), developers of applications leveraging LLMs must decide whether to take advantage of these improvements or stay with older tried-and-tested models. Low perceived switching frictions can lead to choices that do not consider more subtle behavior changes that the transition may induce. Our experiments use a popular game-theoretic behavioral economics model of trust to show stark differences in the trusting behavior of OpenAI's and DeepSeek's models. We highlight a collapse in the economic trust behavior of the o1-mini and o3-mini models as they reconcile profit-maximizing and risk-seeking with future returns from trust, and contrast it with DeepSeek's more sophisticated and profitable trusting behavior that stems from an ability to incorporate deeper concepts like forward planning and theory-of-mind. As LLMs form the basis for high-stakes commercial systems, our results highlight the perils of relying on LLM performance benchmarks that are too narrowly defined and suggest that careful analysis of their hidden fault lines should be part of any organization's AI strategy. 

**Abstract (ZH)**: 当遇到来自新大型语言模型（LLM）日益频繁的性能提升或成本降低时，利用LLM的应用开发人员必须决定是否利用这些改进或继续使用较为成熟的旧模型。较低的转换摩擦感知可能导致忽略转换可能引起的更微妙的行为变化。我们的实验使用一个流行的游戏理论行为经济学模型来展示OpenAI与DeepSeek模型在信任行为上的截然不同。我们指出，在o1-mini和o3-mini模型中，随着它们将利润最大化和风险偏好与信任带来的未来收益进行融合，信任经济行为出现崩溃，而DeepSeek的模型则表现出更加复杂、更有利可图的信任行为，这源于其能够整合更深层次的概念如前瞻规划和理论思维的能力。随着LLM成为高风险商业系统的基础，我们的结果突显了依赖于定义过于狭窄的LLM性能基准的危险，并建议任何组织的人工智能战略中应包含对隐藏薄弱环节的仔细分析。 

---
# Portable Reward Tuning: Towards Reusable Fine-Tuning across Different Pretrained Models 

**Title (ZH)**: 便携式奖励调整：朝向跨不同预训练模型的可重用微调 

**Authors**: Daiki Chijiwa, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, Susumu Takeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12776)  

**Abstract**: While foundation models have been exploited for various expert tasks through fine-tuning, any foundation model will become outdated due to its old knowledge or limited capability. Thus the underlying foundation model should be eventually replaced by new ones, which leads to repeated cost of fine-tuning these new models. Existing work addresses this problem by inference-time tuning, i.e., modifying the output probabilities from the new foundation model with the outputs from the old foundation model and its fine-tuned model, which involves an additional overhead in inference by the latter two models. In this paper, we propose a new fine-tuning principle, Portable Reward Tuning (PRT), that reduces the inference overhead by its nature, based on the reformulation of fine-tuning as the reward maximization. Specifically, instead of fine-tuning parameters of the foundation models, PRT trains the reward model explicitly through the same loss function as in fine-tuning. During inference, the reward model can be used with any foundation model (with the same set of vocabularies or labels) through the formulation of reward maximization. Experimental results, covering both vision and language models, demonstrate that the PRT-trained model can achieve comparable accuracy to the existing work of inference-time tuning, with less inference cost. 

**Abstract (ZH)**: 基于奖励最大化原则的便携式微调（PRT） 

---
# How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild 

**Title (ZH)**: 多语言在野场景下大语言模型的幻觉程度估计 

**Authors**: Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2502.12769)  

**Abstract**: In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models. 

**Abstract (ZH)**: 在信息误导时代，幻觉——大型语言模型（LLMs）生成非事实或不忠实响应的趋势——代表了其全球实用性的主要风险。尽管LLMs变得日益多语言化，但用于检测和量化LLM幻觉的研究主要（a）以英语为中心，并且（b）侧重于机器翻译（MT）和总结任务，这些任务在现实世界中不如开放信息查询普遍。相比之下，我们旨在跨多种语言的知识密集型长格式问答中量化LLM幻觉的程度。为此，我们训练了一个多语言幻觉检测模型，并在30种语言和6大家族开源LLM上开展了大规模研究。我们从一个英语幻觉检测数据集出发，依靠机器翻译生成其他语言的（嘈杂的）训练数据。我们还为五种高资源语言手动标注了黄金数据；然后，我们展示了这些语言中，幻觉率估计值在银色（LLM生成的）测试集和黄金测试集之间相似，验证了使用银数据来估计其他语言的幻觉率的可行性。最终，我们构建了一个包含30种语言的知识密集型问答数据集，LLM生成的提示和维基百科文章作为参考。我们发现，虽然高资源语言的LLM生成更长的响应并包含更多的幻觉令牌，但语言的数字化表示与其标准化后的幻觉率之间没有关联。此外，我们发现较小的LLM表现出更大的幻觉率，而较大的模型则不然。 

---
# R2-KG: General-Purpose Dual-Agent Framework for Reliable Reasoning on Knowledge Graphs 

**Title (ZH)**: 基于可靠知识图推理的通用双代理框架R2-KG 

**Authors**: Sumin Jo, Junseong Choi, Jiho Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.12767)  

**Abstract**: Recent studies have combined Large Language Models (LLMs) with Knowledge Graphs (KGs) to enhance reasoning, improving inference accuracy without additional training while mitigating hallucination. However, existing frameworks are often rigid, struggling to adapt to KG or task changes. They also rely heavily on powerful LLMs for reliable (i.e., trustworthy) reasoning. To address this, We introduce R2-KG, a plug-and-play, dual-agent framework that separates reasoning into two roles: an Operator (a low-capacity LLM) that gathers evidence and a Supervisor (a high-capacity LLM) that makes final judgments. This design is cost-efficient for LLM inference while still maintaining strong reasoning accuracy. Additionally, R2-KG employs an Abstention mechanism, generating answers only when sufficient evidence is collected from KG, which significantly enhances reliability. Experiments across multiple KG-based reasoning tasks show that R2-KG consistently outperforms baselines in both accuracy and reliability, regardless of the inherent capability of LLMs used as the Operator. Further experiments reveal that the single-agent version of R2-KG, equipped with a strict self-consistency strategy, achieves significantly higher-than-baseline reliability while reducing inference cost. However, it also leads to a higher abstention rate in complex KGs. Our findings establish R2-KG as a flexible and cost-effective solution for KG-based reasoning. It reduces reliance on high-capacity LLMs while ensuring trustworthy inference. 

**Abstract (ZH)**: Recent Studies Combining Large Language Models with Knowledge Graphs to Enhance Reasoning and Improve Reliability 

---
# Efficient Machine Translation Corpus Generation: Integrating Human-in-the-Loop Post-Editing with Large Language Models 

**Title (ZH)**: 高效的机器翻译语料生成：集成人工在环后编辑的大语言模型 

**Authors**: Kamer Ali Yuksel, Ahmet Gunduz, Abdul Baseet Anees, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.12755)  

**Abstract**: This paper introduces an advanced methodology for machine translation (MT) corpus generation, integrating semi-automated, human-in-the-loop post-editing with large language models (LLMs) to enhance efficiency and translation quality. Building upon previous work that utilized real-time training of a custom MT quality estimation metric, this system incorporates novel LLM features such as Enhanced Translation Synthesis and Assisted Annotation Analysis, which improve initial translation hypotheses and quality assessments, respectively. Additionally, the system employs LLM-Driven Pseudo Labeling and a Translation Recommendation System to reduce human annotator workload in specific contexts. These improvements not only retain the original benefits of cost reduction and enhanced post-edit quality but also open new avenues for leveraging cutting-edge LLM advancements. The project's source code is available for community use, promoting collaborative developments in the field. The demo video can be accessed here. 

**Abstract (ZH)**: 本文介绍了一种先进的机器翻译语料库生成方法，该方法结合了半自动化、人工在环的后编辑与大型语言模型（LLM），以提高效率和翻译质量。在此基础上，该系统整合了增强翻译合成和辅助注释分析等新型LLM功能，分别改进了初始翻译假设和质量评估。此外，该系统采用LLM驱动的伪标签和翻译推荐系统，以减少特定上下文中的人工标注员工作量。这些改进不仅保留了成本降低和增强后编辑质量的原始优势，还为利用最前沿的LLM进展开辟了新的途径。该项目的源代码可供社区使用，促进该领域的协作发展。演示视频可在此处访问。 

---
# "I know myself better, but not really greatly": Using LLMs to Detect and Explain LLM-Generated Texts 

**Title (ZH)**: “我了解自己，但却并不真正深入”：使用大语言模型检测和解释由大语言模型生成的文本 

**Authors**: Jiazhou Ji, Jie Guo, Weidong Qiu, Zheng Huang, Yang Xu, Xinru Lu, Xiaoyu Jiang, Ruizhe Li, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12743)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in generating human-like texts, but the potential misuse of such LLM-generated texts raises the need to distinguish between human-generated and LLM-generated content. This paper explores the detection and explanation capabilities of LLM-based detectors of LLM-generated texts, in the context of a binary classification task (human-generated texts vs LLM-generated texts) and a ternary classification task (human-generated texts, LLM-generated texts, and undecided). By evaluating on six close/open-source LLMs with different sizes, our findings reveal that while self-detection consistently outperforms cross-detection, i.e., LLMs can detect texts generated by themselves more accurately than those generated by other LLMs, the performance of self-detection is still far from ideal, indicating that further improvements are needed. We also show that extending the binary to the ternary classification task with a new class "Undecided" can enhance both detection accuracy and explanation quality, with improvements being statistically significant and consistent across all LLMs. We finally conducted comprehensive qualitative and quantitative analyses on the explanation errors, which are categorized into three types: reliance on inaccurate features (the most frequent error), hallucinations, and incorrect reasoning. These findings with our human-annotated dataset emphasize the need for further research into improving both self-detection and self-explanation, particularly to address overfitting issues that may hinder generalization. 

**Abstract (ZH)**: 基于LLM的LLM生成文本检测与解释能力探索：从二分类任务到三分类任务的研究 

---
# Translate Smart, not Hard: Cascaded Translation Systems with Quality-Aware Deferral 

**Title (ZH)**: 聪明不努力：基于质量感知推迟的级联翻译系统 

**Authors**: António Farinhas, Nuno M. Guerreiro, Sweta Agrawal, Ricardo Rei, André F.T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2502.12701)  

**Abstract**: Larger models often outperform smaller ones but come with high computational costs. Cascading offers a potential solution. By default, it uses smaller models and defers only some instances to larger, more powerful models. However, designing effective deferral rules remains a challenge. In this paper, we propose a simple yet effective approach for machine translation, using existing quality estimation (QE) metrics as deferral rules. We show that QE-based deferral allows a cascaded system to match the performance of a larger model while invoking it for a small fraction (30% to 50%) of the examples, significantly reducing computational costs. We validate this approach through both automatic and human evaluation. 

**Abstract (ZH)**: 较大的模型通常 performance 更好但伴随较高的计算成本。级联提供了一种潜在的解决方案。默认情况下，它使用较小的模型并将部分实例委托给更强大、更大的模型。然而，设计有效的延缓规则仍然颇具挑战性。在本文中，我们提出了一种简单而有效的方法用于机器翻译，利用现有的质量估计 (QE) 指标作为延缓规则。我们展示了基于 QE 的延缓使得级联系统能够在少量实例（30% 到 50%）上调用更大模型的情况下达到与其相当的性能，大幅度降低了计算成本。我们通过自动和人工评估验证了该方法。 

---
# The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1 

**Title (ZH)**: 大型推理模型潜藏的风险：R1的安全性评估 

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12659)  

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap. 

**Abstract (ZH)**: 大型推理模型（如OpenAI-o3和DeepSeek-R1）的迅速发展导致了复杂推理能力在非推理大型语言模型（LLMs）中的显著提升。然而，这些模型增强的能力，尤其是开源模型DeepSeek-R1的开放访问，引发了严重的安全顾虑，特别是在潜在误用方面。在本研究中，我们通过利用现有的安全基准对其安全性进行综合评估，并考察它们对对抗性攻击（如 Jailbreaking 和提示注入）的脆弱性，以评估其在实际应用中的鲁棒性。通过多方面的分析，我们发现了四个关键发现：（1）开源R1模型与o3-mini模型在安全基准和攻击测试中都存在显著的安全差距，表明需要在R1模型上投入更多安全努力。（2）提炼推理模型的安全性能逊于与其安全对齐的基础模型。（3）模型的推理能力越强，其在回答不安全问题时造成潜在危害的可能性越大。（4）R1模型的思考过程比其最终答案更值得关注安全问题。我们的研究提供了关于推理模型安全影响的见解，并突显了进一步提升R1模型安全性的必要性，以缩小差距。 

---
# \textit{One Size doesn't Fit All}: A Personalized Conversational Tutoring Agent for Mathematics Instruction 

**Title (ZH)**: 大小不一：一个个性化对话式辅导代理用于数学教学 

**Authors**: Ben Liu, Jihan Zhang, Fangquan Lin, Xu Jia, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12633)  

**Abstract**: Large language models (LLMs) have been increasingly employed in various intelligent educational systems, simulating human tutors to facilitate effective human-machine interaction. However, previous studies often overlook the significance of recognizing and adapting to individual learner characteristics. Such adaptation is crucial for enhancing student engagement and learning efficiency, particularly in mathematics instruction, where diverse learning styles require personalized strategies to promote comprehension and enthusiasm. In this paper, we propose a \textbf{P}erson\textbf{A}lized \textbf{C}onversational tutoring ag\textbf{E}nt (PACE) for mathematics instruction. PACE simulates students' learning styles based on the Felder and Silverman learning style model, aligning with each student's persona. In this way, our PACE can effectively assess the personality of students, allowing to develop individualized teaching strategies that resonate with their unique learning styles. To further enhance students' comprehension, PACE employs the Socratic teaching method to provide instant feedback and encourage deep thinking. By constructing personalized teaching data and training models, PACE demonstrates the ability to identify and adapt to the unique needs of each student, significantly improving the overall learning experience and outcomes. Moreover, we establish multi-aspect evaluation criteria and conduct extensive analysis to assess the performance of personalized teaching. Experimental results demonstrate the superiority of our model in personalizing the educational experience and motivating students compared to existing methods. 

**Abstract (ZH)**: 个性化交互式辅导代理（PACE）在数学教学中的应用 

---
# Automating Prompt Leakage Attacks on Large Language Models Using Agentic Approach 

**Title (ZH)**: 使用代理方法自动化的提示泄漏攻击研究：针对大规模语言模型 

**Authors**: Tvrtko Sternak, Davor Runje, Dorian Granoša, Chi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12630)  

**Abstract**: This paper presents a novel approach to evaluating the security of large language models (LLMs) against prompt leakage-the exposure of system-level prompts or proprietary configurations. We define prompt leakage as a critical threat to secure LLM deployment and introduce a framework for testing the robustness of LLMs using agentic teams. Leveraging AG2 (formerly AutoGen), we implement a multi-agent system where cooperative agents are tasked with probing and exploiting the target LLM to elicit its prompt.
Guided by traditional definitions of security in cryptography, we further define a prompt leakage-safe system as one in which an attacker cannot distinguish between two agents: one initialized with an original prompt and the other with a prompt stripped of all sensitive information. In a safe system, the agents' outputs will be indistinguishable to the attacker, ensuring that sensitive information remains secure. This cryptographically inspired framework provides a rigorous standard for evaluating and designing secure LLMs.
This work establishes a systematic methodology for adversarial testing of prompt leakage, bridging the gap between automated threat modeling and practical LLM security.
You can find the implementation of our prompt leakage probing on GitHub. 

**Abstract (ZH)**: 本文提出了一种评估大型语言模型（LLMs）对提示泄漏抵御能力的新方法——系统级提示或专有配置的暴露。我们定义提示泄漏是对安全LLM部署的关键威胁，并引入了一种使用代理团队测试LLM鲁棒性的框架。利用AG2（原AutoGen），我们实现了多代理系统，其中协作代理被指派探测和利用目标LLM以引发其提示。

基于密码学中传统的安全定义，我们进一步定义了一个提示泄漏安全系统为一个攻击者无法区分两者的系统：一个是初始化为原始提示的代理，另一个是删除所有敏感信息的提示的代理。在安全系统中，代理的输出对攻击者来说是不可区分的，从而确保敏感信息的安全。这种密码学启发式的框架为评估和设计安全LLM提供了一个严格的标准。

本文建立了一种系统性的方法来对抗提示泄漏的测试，填补了自动化威胁建模与实际LLM安全之间的差距。

您可以在GitHub上找到我们提示泄漏探测的实现。 

---
# RSMLP: A light Sampled MLP Structure for Incomplete Utterance Rewrite 

**Title (ZH)**: RSMLP：一种用于不完整陈述重写的小样本MLP结构 

**Authors**: Lunjun Liu, Weilai Jiang, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12587)  

**Abstract**: The Incomplete Utterance Rewriting (IUR) task has garnered significant attention in recent years. Its goal is to reconstruct conversational utterances to better align with the current context, thereby enhancing comprehension. In this paper, we introduce a novel and versatile lightweight method, Rewritten-Sampled MLP (RSMLP). By employing an MLP based architecture with a carefully designed down-sampling strategy, RSMLP effectively extracts latent semantic information between utterances and makes appropriate edits to restore incomplete utterances. Due to its simple yet efficient structure, our method achieves competitive performance on public IUR datasets and in real-world applications. 

**Abstract (ZH)**: 不完整utterance重写（IUR）任务近年来引起了广泛关注。其目标是重构对话utterance以更好地与当前上下文对齐，从而提高理解能力。本文介绍了一种新颖且灵活的轻量级方法，即重写采样MLP（RSMLP）。通过采用基于MLP的架构并结合精心设计的下采样策略，RSMLP有效地提取了utterance之间的潜在语义信息，并进行适当的编辑以恢复不完整utterance。由于其简洁高效的结构，该方法在公开的IUR数据集和实际应用中实现了竞争性的性能。 

---
# DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent 

**Title (ZH)**: DemonAgent：基于LLM的代理动态加密多后门植入攻击 

**Authors**: Pengyu Zhu, Zhenhong Zhou, Yuanhe Zhang, Shilinlu Yan, Kun Wang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.12575)  

**Abstract**: As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at this https URL. 

**Abstract (ZH)**: 基于LLM的代理中动态加密多后门植入攻击 

---
# HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading 

**Title (ZH)**: HeadInfer: 头部明智卸载的内存高效LLM推理 

**Authors**: Cheng Luo, Zefan Cai, Hanshi Sun, Jinqi Xiao, Bo Yuan, Wen Xiao, Junjie Hu, Jiawei Zhao, Beidi Chen, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12574)  

**Abstract**: Transformer-based large language models (LLMs) demonstrate impressive performance in long context generation. Extending the context length has disproportionately shifted the memory footprint of LLMs during inference to the key-value cache (KV cache). In this paper, we propose HEADINFER, which offloads the KV cache to CPU RAM while avoiding the need to fully store the KV cache for any transformer layer on the GPU. HEADINFER employs a fine-grained, head-wise offloading strategy, maintaining only selective attention heads KV cache on the GPU while computing attention output dynamically. Through roofline analysis, we demonstrate that HEADINFER maintains computational efficiency while significantly reducing memory footprint. We evaluate HEADINFER on the Llama-3-8B model with a 1-million-token sequence, reducing the GPU memory footprint of the KV cache from 128 GB to 1 GB and the total GPU memory usage from 207 GB to 17 GB, achieving a 92% reduction compared to BF16 baseline inference. Notably, HEADINFER enables 4-million-token inference with an 8B model on a single consumer GPU with 24GB memory (e.g., NVIDIA RTX 4090) without approximation methods. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）在长上下文生成任务中表现出色。扩展上下文长度已不均衡地将LLMs推理过程中的内存足迹转向了键值缓存（KV缓存）。本文提出HEADINFER，该方法将KV缓存卸载到CPU内存中，同时避免在任何Transformer层上完全存储KV缓存。HEADINFER采用细粒度的头级卸载策略，仅在GPU上保留选择性的注意力头KV缓存，并在计算注意力输出时动态调整。通过roofline分析，我们展示了HEADINFER在保持计算效率的同时显著减少了内存足迹。我们使用Llama-3-8B模型和100万token序列评估了HEADINFER，将KV缓存的GPU内存足迹从128 GB降低到1 GB，总GPU内存使用量从207 GB降低到17 GB，相较于BF16基线推理实现了92%的减少。值得注意的是，HEADINFER在具有24 GB内存的单个消费者级GPU（例如NVIDIA RTX 4090）上实现了8B模型的400万token推理，无需使用近似方法。 

---
# A Cognitive Writing Perspective for Constrained Long-Form Text Generation 

**Title (ZH)**: 认知写作视角下的约束长文本生成 

**Authors**: Kaiyang Wan, Honglin Mu, Rui Hao, Haoran Luo, Tianle Gu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12568)  

**Abstract**: Like humans, Large Language Models (LLMs) struggle to generate high-quality long-form text that adheres to strict requirements in a single pass. This challenge is unsurprising, as successful human writing, according to the Cognitive Writing Theory, is a complex cognitive process involving iterative planning, translating, reviewing, and monitoring. Motivated by these cognitive principles, we aim to equip LLMs with human-like cognitive writing capabilities through CogWriter, a novel training-free framework that transforms LLM constrained long-form text generation into a systematic cognitive writing paradigm. Our framework consists of two key modules: (1) a Planning Agent that performs hierarchical planning to decompose the task, and (2) multiple Generation Agents that execute these plans in parallel. The system maintains quality via continuous monitoring and reviewing mechanisms, which evaluate outputs against specified requirements and trigger necessary revisions. CogWriter demonstrates exceptional performance on LongGenBench, a benchmark for complex constrained long-form text generation. Even when using Qwen-2.5-14B as its backbone, CogWriter surpasses GPT-4o by 22% in complex instruction completion accuracy while reliably generating texts exceeding 10,000 words. We hope this cognitive science-inspired approach provides a paradigm for LLM writing advancements: \href{this https URL}{CogWriter}. 

**Abstract (ZH)**: 类人类，大型语言模型在单次生成符合严格要求的长文本时也面临挑战。这一挑战不言自明，因为根据认知写作理论，成功的写作是一个涉及迭代计划、翻译、审阅和监控的复杂认知过程。受这些认知原则的启发，我们旨在通过CogWriter这一新的无需训练的框架来赋予大型语言模型类人类的认知写作能力，将受限的长文本生成转化为一种系统化的认知写作范式。该框架包含两个关键模块：（1）规划代理，执行层次化规划以分解任务；（2）多个生成代理，同时执行这些计划。系统通过持续的监控和审阅机制来维持质量，这些机制评估输出并与规定的要求进行对比，并触发必要的修订。CogWriter在LongGenBench这一复杂受限长文本生成基准测试中表现出色。即使使用Qwen-2.5-14B作为其基础模型，CogWriter在复杂指令完成精度上也超越GPT-4o 22%，同时可靠地生成超过10,000字的文本。我们希望这一受认知科学启发的方法为大型语言模型写作进步提供范式：CogWriter。 

---
# Evaluating Language Models on Grooming Risk Estimation Using Fuzzy Theory 

**Title (ZH)**: 基于模糊理论的语言模型在估计护肤风险评估中的评价 

**Authors**: Geetanjali Bihani, Tatiana Ringenberg, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2502.12563)  

**Abstract**: Encoding implicit language presents a challenge for language models, especially in high-risk domains where maintaining high precision is important. Automated detection of online child grooming is one such critical domain, where predators manipulate victims using a combination of explicit and implicit language to convey harmful intentions. While recent studies have shown the potential of Transformer language models like SBERT for preemptive grooming detection, they primarily depend on surface-level features and approximate real victim grooming processes using vigilante and law enforcement conversations. The question of whether these features and approximations are reasonable has not been addressed thus far. In this paper, we address this gap and study whether SBERT can effectively discern varying degrees of grooming risk inherent in conversations, and evaluate its results across different participant groups. Our analysis reveals that while fine-tuning aids language models in learning to assign grooming scores, they show high variance in predictions, especially for contexts containing higher degrees of grooming risk. These errors appear in cases that 1) utilize indirect speech pathways to manipulate victims and 2) lack sexually explicit content. This finding underscores the necessity for robust modeling of indirect speech acts by language models, particularly those employed by predators. 

**Abstract (ZH)**: 编码隐含语言给语言模型带来了挑战，尤其是在高风险领域，保持高精度尤为关键。自动化检测在线恋童行为就是一个这样的核心领域，其中犯罪者使用明示和隐含语言的结合来传达有害意图。虽然最近的研究表明，如SBERT这样的变换器语言模型在预检测恋童行为方面具有潜力，但它们主要依赖表面级特征，并通过监护人和执法机构的对话模拟真实的恋童行为过程。关于这些特征和模拟是否合理的问题尚未得到解答。在本文中，我们填补了这一空白，研究SBERT能否有效地区分对话中固有的不同程度的恋童风险，并在其不同的参与者群体中评估其结果。我们的分析表明，虽然微调有助于语言模型学习分配恋童评分，但在高风险程度的上下文中，预测结果显示了高方差。这些错误出现在1）利用间接言语途径操纵受害者和2）缺乏明确性内容的情况下。这一发现强调了语言模型，尤其是犯罪者使用的语言模型，对间接言语行为进行稳健建模的必要性。 

---
# LLM Safety for Children 

**Title (ZH)**: 儿童安全的大型语言模型 

**Authors**: Prasanjit Rath, Hari Shrawgi, Parag Agrawal, Sandipan Dandapat  

**Link**: [PDF](https://arxiv.org/pdf/2502.12552)  

**Abstract**: This paper analyzes the safety of Large Language Models (LLMs) in interactions with children below age of 18 years. Despite the transformative applications of LLMs in various aspects of children's lives such as education and therapy, there remains a significant gap in understanding and mitigating potential content harms specific to this demographic. The study acknowledges the diverse nature of children often overlooked by standard safety evaluations and proposes a comprehensive approach to evaluating LLM safety specifically for children. We list down potential risks that children may encounter when using LLM powered applications. Additionally we develop Child User Models that reflect the varied personalities and interests of children informed by literature in child care and psychology. These user models aim to bridge the existing gap in child safety literature across various fields. We utilize Child User Models to evaluate the safety of six state of the art LLMs. Our observations reveal significant safety gaps in LLMs particularly in categories harmful to children but not adults 

**Abstract (ZH)**: 本文分析了大型语言模型（LLMs）与18岁以下儿童交互时的安全性。尽管LLMs在教育和治疗等多个方面对儿童生活产生了变革性影响，但针对这一特定人群的内容潜在危害依然存在认知和缓解上的显著差距。本研究承认儿童的多样化特质往往被标准的安全评估所忽视，并提出了一种全面的方法来评估LLMs特定针对儿童的安全性。我们列出了儿童在使用LLM驱动的应用时可能遇到的各种潜在风险，并据此开发了反映儿童多样性格和兴趣的儿童用户模型，这些用户模型旨在填补儿童安全研究在多个领域的现有差距。我们使用儿童用户模型评估了六种最先进的LLMs的安全性。我们的观察结果表明，LLMs在对儿童有害而不是对成人有害的类别中存在显著的安全缺口。 

---
# LegalCore: A Dataset for Legal Documents Event Coreference Resolution 

**Title (ZH)**: LegalCore：法律文件事件同指解析数据集 

**Authors**: Kangda Wei, Xi Shi, Jonathan Tong, Sai Ramana Reddy, Anandhavelu Natarajan, Rajiv Jain, Aparna Garimella, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12509)  

**Abstract**: Recognizing events and their coreferential mentions in a document is essential for understanding semantic meanings of text. The existing research on event coreference resolution is mostly limited to news articles. In this paper, we present the first dataset for the legal domain, LegalCore, which has been annotated with comprehensive event and event coreference information. The legal contract documents we annotated in this dataset are several times longer than news articles, with an average length of around 25k tokens per document. The annotations show that legal documents have dense event mentions and feature both short-distance and super long-distance coreference links between event mentions. We further benchmark mainstream Large Language Models (LLMs) on this dataset for both event detection and event coreference resolution tasks, and find that this dataset poses significant challenges for state-of-the-art open-source and proprietary LLMs, which perform significantly worse than a supervised baseline. We will publish the dataset as well as the code. 

**Abstract (ZH)**: 识别文档中事件及其同指提及对于理解文本语义意义至关重要。现有的事件同指消解研究主要集中在新闻文章上。在本文中，我们首次提出了一个法律领域数据集LegalCore，该数据集包含了全面的事件及其事件同指标注信息。我们标注的法律合同文件长度远超新闻文章，平均每份文件包含约25k个词。标注结果显示，法律文件中的事件提及密集，并且事件提及之间存在短距离和超长距离的同指连接。我们进一步在该数据集上对主流大型语言模型（LLMs）进行了事件检测和事件同指消解任务的基准测试，发现该数据集对最先进的开源和专有LLMs构成了重大挑战，这些模型的表现显著劣于监督基线。我们将发布该数据集以及相关的代码。 

---
# EDGE: Efficient Data Selection for LLM Agents via Guideline Effectiveness 

**Title (ZH)**: EDGE: 通过指南有效性进行高效数据选择的LLM代理方法 

**Authors**: Yunxiao Zhang, Guanming Xiong, Haochen Li, Wen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.12494)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities as AI agents. However, existing methods for enhancing LLM-agent abilities often lack a focus on data quality, leading to inefficiencies and suboptimal results in both fine-tuning and prompt engineering. To address this issue, we introduce EDGE, a novel approach for identifying informative samples without needing golden answers. We propose the Guideline Effectiveness (GE) metric, which selects challenging samples by measuring the impact of human-provided guidelines in multi-turn interaction tasks. A low GE score indicates that the human expertise required for a sample is missing from the guideline, making the sample more informative. By selecting samples with low GE scores, we can improve the efficiency and outcomes of both prompt engineering and fine-tuning processes for LLMs. Extensive experiments validate the performance of our method. Our method achieves competitive results on the HotpotQA and WebShop and datasets, requiring 75\% and 50\% less data, respectively, while outperforming existing methods. We also provide a fresh perspective on the data quality of LLM-agent fine-tuning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了作为AI代理的显著能力。然而，现有的增强LLM-代理能力的方法往往缺乏对数据质量的关注，这在微调和提示工程中导致了效率低下和次优结果。为了解决这一问题，我们提出了EDGE，一种无需金标准答案即可识别有信息量样本的新型方法。我们提出了指南有效性（GE）度量，通过衡量人类提供的指南在多轮交互任务中的影响来选择具有挑战性的样本。GE分数较低表明样本所需的人类专业知识未包含在指南中，从而使样本更具有信息量。通过选择GE分数较低的样本，我们可以提高LLM的提示工程和微调过程的效率和结果。广泛的实验验证了我们方法的性能。我们的方法在HotpotQA和WebShop数据集上取得了竞争力的结果，分别仅需要75%和50%的数据，并优于现有方法。我们还提供了LLM-代理微调数据质量的新视角。 

---
# Safe at the Margins: A General Approach to Safety Alignment in Low-Resource English Languages -- A Singlish Case Study 

**Title (ZH)**: 在边缘处求安全：一种低资源英语语言安全对齐的一般性方法——以新加坡英语案例研究为例 

**Authors**: Isaac Lim, Shaun Khoo, Watson Chua, Goh Jiayi, Jessica Foo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12485)  

**Abstract**: To ensure safe usage, Large Language Models (LLMs) typically undergo alignment with human-defined values. However, this alignment often relies on primarily English data and is biased towards Western-centric values, limiting its effectiveness in low-resource language settings. In this paper, we describe our approach for aligning SEA-Lion-v2.1-Instruct (a Llama3-8B variant) to minimize toxicity in Singlish, an English creole specific to Singapore. We find that supervised fine-tuning and Kahneman-Tversky Optimization (KTO) on paired and unpaired preferences is more sample efficient and yields significantly better results than Direct Preference Optimization (DPO). Our analysis reveals that DPO implicitly enforces a weaker safety objective than KTO, and that SFT complements KTO by improving training stability. Finally, we introduce a simple but novel modification to KTO, KTO-S, which improves training stability through better gradient exploitation. Overall, we present a general approach for safety alignment conducive to low-resource English languages, successfully reducing toxicity by 99\% on our Singlish benchmark, with gains generalizing to the broader TOXIGEN dataset while maintaining strong performance across standard LLM benchmarks. 

**Abstract (ZH)**: 确保安全使用：将大型语言模型SEA-Lion-v2.1-Instruct（一种Llama3-8B变体）对新加坡英语（Singlish）中的毒性进行最小化对齐 

---
# MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation 

**Title (ZH)**: MCTS-Judge：代码正确性评估中基于LLM的测试时可扩展性方法 

**Authors**: Yutong Wang, Pengliang Ji, Chaoqun Yang, Kaixin Li, Ming Hu, Jiaoyang Li, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.12468)  

**Abstract**: The LLM-as-a-Judge paradigm shows promise for evaluating generative content but lacks reliability in reasoning-intensive scenarios, such as programming. Inspired by recent advances in reasoning models and shifts in scaling laws, we pioneer bringing test-time computation into LLM-as-a-Judge, proposing MCTS-Judge, a resource-efficient, System-2 thinking framework for code correctness evaluation. MCTS-Judge leverages Monte Carlo Tree Search (MCTS) to decompose problems into simpler, multi-perspective evaluations. Through a node-selection strategy that combines self-assessment based on historical actions in the current trajectory and the Upper Confidence Bound for Trees based on prior rollouts, MCTS-Judge balances global optimization and refinement of the current trajectory. We further designed a high-precision, unit-test-level reward mechanism to encourage the Large Language Model (LLM) to perform line-by-line analysis. Extensive experiments on three benchmarks and five LLMs demonstrate the effectiveness of MCTS-Judge, which improves the base model's accuracy from 41% to 80%, surpassing the o1-series models with 3x fewer tokens. Further evaluations validate the superiority of its reasoning trajectory in logic, analytics, thoroughness, and overall quality, while revealing the test-time scaling law of the LLM-as-a-Judge paradigm. 

**Abstract (ZH)**: LLM-as-a-Judge paradigm结合蒙特卡洛树搜索的高效代码正确性评价框架 

---
# EquiBench: Benchmarking Code Reasoning Capabilities of Large Language Models via Equivalence Checking 

**Title (ZH)**: EquiBench: 通过等价性检查评估大型语言模型的代码推理能力 

**Authors**: Anjiang Wei, Jiannan Cao, Ran Li, Hongyu Chen, Yuhui Zhang, Ziheng Wang, Yaofeng Sun, Yuan Liu, Thiago S. F. X. Teixeira, Diyi Yang, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2502.12466)  

**Abstract**: Equivalence checking, i.e., determining whether two programs produce identical outputs for all possible inputs, underpins a broad range of applications, including software refactoring, testing, and optimization. We present the task of equivalence checking as a new way to evaluate the code reasoning abilities of large language models (LLMs). We introduce EquiBench, a dataset of 2400 program pairs spanning four programming languages and six equivalence categories. These pairs are systematically generated through program analysis, compiler scheduling, and superoptimization, covering nontrivial structural transformations that demand deep semantic reasoning beyond simple syntactic variations. Our evaluation of 17 state-of-the-art LLMs shows that OpenAI o3-mini achieves the highest overall accuracy of 78.0%. In the most challenging categories, the best accuracies are 62.3% and 68.8%, only modestly above the 50% random baseline for binary classification, indicating significant room for improvement in current models' code reasoning capabilities. 

**Abstract (ZH)**: 等价性检查，即确定两个程序在所有可能的输入下是否产生相同的输出，是软件重构、测试和优化等多种应用的基础。我们将等价性检查任务视为评估大型语言模型（LLMs）代码推理能力的新方法。我们介绍了EquiBench数据集，包含2400个程序对，覆盖四种编程语言和六种等价性类别。这些程序对通过程序分析、编译器调度和超优化系统生成，涵盖了要求深刻语义推理的复杂结构变换，不仅超越了简单的语法变体。对17个最先进的LLM进行评估显示，OpenAI o3-mini的整体准确率为78.0%。在最具挑战性的类别中，最高准确率为62.3%和68.8%，仅略高于二分类的50%随机基线，表明当前模型的代码推理能力仍有显著提升空间。 

---
# Stress Testing Generalization: How Minor Modifications Undermine Large Language Model Performance 

**Title (ZH)**: 泛化压力测试：细微修改如何削弱大型语言模型性能 

**Authors**: Guangxiang Zhao, Saier Hu, Xiaoqi Jian, Jinzhu Wu, Yuhan Wu, Change Jia, Lin Sun, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12459)  

**Abstract**: This paper investigates the fragility of Large Language Models (LLMs) in generalizing to novel inputs, specifically focusing on minor perturbations in well-established benchmarks (e.g., slight changes in question format or distractor length). Despite high benchmark scores, LLMs exhibit significant accuracy drops and unexpected biases (e.g., preference for longer distractors) when faced with these minor but content-preserving modifications. For example, Qwen 2.5 1.5B's MMLU score rises from 60 to 89 and drops from 89 to 36 when option lengths are changed without altering the question. Even GPT-4 experiences a 25-point accuracy loss when question types are changed, with a 6-point drop across all three modification categories. These analyses suggest that LLMs rely heavily on superficial cues rather than forming robust, abstract representations that generalize across formats, lexical variations, and irrelevant content shifts. This work aligns with the ACL 2025 theme track on the Generalization of NLP models, proposing a "Generalization Stress Test" to assess performance shifts under controlled perturbations. The study calls for reevaluating benchmarks and developing more reliable evaluation methodologies to capture LLM generalization abilities better. 

**Abstract (ZH)**: 本文研究了大规模语言模型（LLMs）在应对新颖输入时的脆弱性，特别关注于在广泛认可的基准测试中（如问题格式或干扰项长度的小幅变化）引入细微扰动。尽管在基准测试中取得了高分，但当面对这些细微但内容保留的修改时，LLMs仍表现出显著的准确率下降和意想不到的偏差（例如，偏好更长的干扰项）。例如，Qwen 2.5 1.5B的MMLU分数在选项长度改变而不改变问题的情况下，从60上升到89，随后又从89下降到36。即使GPT-4在问题类型发生变化时也经历了25点的准确率损失，在所有三种修改类别中平均下降6点。这些分析表明，LLMs主要依赖于表面特征，而不是形成能够跨格式、词汇变体和不相关内容转移的稳健、抽象的表示。该研究与ACL 2025主题跟踪中的自然语言处理模型泛化主题一致，提出了“泛化压力测试”来评估在控制扰动下的性能变化。研究呼吁重新评估基准测试，并开发更可靠的评估方法，以更好地捕捉LLM的泛化能力。 

---
# Benchmarking Zero-Shot Facial Emotion Annotation with Large Language Models: A Multi-Class and Multi-Frame Approach in DailyLife 

**Title (ZH)**: 基于大型语言模型的零样本面部情绪标注benchmark：日常生活中的多类别多帧方法 

**Authors**: He Zhang, Xinyi Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12454)  

**Abstract**: This study investigates the feasibility and performance of using large language models (LLMs) to automatically annotate human emotions in everyday scenarios. We conducted experiments on the DailyLife subset of the publicly available FERV39k dataset, employing the GPT-4o-mini model for rapid, zero-shot labeling of key frames extracted from video segments. Under a seven-class emotion taxonomy ("Angry," "Disgust," "Fear," "Happy," "Neutral," "Sad," "Surprise"), the LLM achieved an average precision of approximately 50%. In contrast, when limited to ternary emotion classification (negative/neutral/positive), the average precision increased to approximately 64%. Additionally, we explored a strategy that integrates multiple frames within 1-2 second video clips to enhance labeling performance and reduce costs. The results indicate that this approach can slightly improve annotation accuracy. Overall, our preliminary findings highlight the potential application of zero-shot LLMs in human facial emotion annotation tasks, offering new avenues for reducing labeling costs and broadening the applicability of LLMs in complex multimodal environments. 

**Abstract (ZH)**: 本研究探讨了使用大型语言模型（LLMs）自动标注日常生活场景中人类情绪的可行性和性能。我们在公开可用的FERV39k数据集中的DailyLife子集上进行了实验，利用GPT-4o-mini模型对视频片段中提取的关键帧进行快速零样本标注。在七类情绪分类 taxonomy (“愤怒”、“厌恶”、“恐惧”、“快乐”、“中性”、“悲伤”、“惊讶”) 下，LLM 的平均精度约为 50%。相比之下，在仅限三元情感分类（负/中性/正）的情况下，平均精度提高到约 64%。此外，我们探讨了一种在 1-2 秒视频片段中整合多帧以提高标注性能并降低成本的策略。结果显示，这种方法可以略微提高标注准确性。总体而言，我们的初步研究成果突显了零样本 LLM 在人类面部情绪标注任务中的潜在应用，为减少标注成本和扩大 LLM 在复杂多模态环境中的应用提供了新的途径。 

---
# Multi-Attribute Steering of Language Models via Targeted Intervention 

**Title (ZH)**: 针对目标干预的多属性语言模型调控 

**Authors**: Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.12446)  

**Abstract**: Inference-time intervention (ITI) has emerged as a promising method for steering large language model (LLM) behavior in a particular direction (e.g., improving helpfulness) by intervening on token representations without costly updates to the LLM's parameters. However, existing ITI approaches fail to scale to multi-attribute settings with conflicts, such as enhancing helpfulness while also reducing toxicity. To address this, we introduce Multi-Attribute Targeted Steering (MAT-Steer), a novel steering framework designed for selective token-level intervention across multiple attributes. MAT-Steer learns steering vectors using an alignment objective that shifts the model's internal representations of undesirable outputs closer to those of desirable ones while enforcing sparsity and orthogonality among vectors for different attributes, thereby reducing inter-attribute conflicts. We evaluate MAT-Steer in two distinct settings: (i) on question answering (QA) tasks where we balance attributes like truthfulness, bias, and toxicity; (ii) on generative tasks where we simultaneously improve attributes like helpfulness, correctness, and coherence. MAT-Steer outperforms existing ITI and parameter-efficient finetuning approaches across both task types (e.g., 3% average accuracy gain across QA tasks and 55.82% win rate against the best ITI baseline). 

**Abstract (ZH)**: 多属性针对性调节（MAT-Steer）：一种用于多属性上下文的选择性tokens级干预框架 

---
# SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs 

**Title (ZH)**: SparAMX: 在AMX-powered CPU上加速压缩LLMs_token生成 

**Authors**: Ahmed F. AbouElhamayed, Jordan Dotzel, Yash Akhauri, Chi-Chih Chang, Sameh Gobriel, J. Pablo Muñoz, Vui Seng Chua, Nilesh Jain, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2502.12444)  

**Abstract**: Large language models have high compute, latency, and memory requirements. While specialized accelerators such as GPUs and TPUs typically run these workloads, CPUs are more widely available and consume less energy. Accelerating LLMs with CPUs enables broader AI access at a lower cost and power consumption. This acceleration potential for CPUs is especially relevant during the memory-bound decoding stage of LLM inference, which processes one token at a time and is becoming increasingly utilized with reasoning models. We utilize Advanced Matrix Extensions (AMX) support on the latest Intel CPUs together with unstructured sparsity to achieve a $1.42 \times$ reduction in end-to-end latency compared to the current PyTorch implementation by applying our technique in linear layers. We provide a set of open-source customized sparse kernels that can speed up any PyTorch model by automatically replacing all linear layers with our custom sparse implementation. Furthermore, we demonstrate for the first time the use of unstructured sparsity in the attention computation achieving a $1.14 \times$ speedup over the current systems without compromising accuracy. Code: this https URL 

**Abstract (ZH)**: 大型语言模型对计算、延迟和内存有高要求。虽然专门的加速器如GPU和TPU通常运行这些工作负载，但CPU更为普及且能耗更低。使用CPU加速大型语言模型可以降低AI的访问成本和能耗，使其更广泛地使用。特别是在LLM推理中的内存限制解码阶段，该阶段逐个处理一个标记并随着推理模型的增加而变得越来越常用时，CPU加速潜力尤为相关。我们利用最新Intel CPU的高级矩阵扩展(AMX)支持，结合无结构稀疏性，在线性层上应用我们的技术，实现了与当前PyTorch实现相比端到端延迟减少1.42倍。我们提供了一组开源定制稀疏内核，可以通过自动将所有线性层替换为我们的定制稀疏实现，来加速任何PyTorch模型。此外，我们首次展示了在注意力计算中使用无结构稀疏性，实现了与当前系统相比1.14倍的加速，且不牺牲准确度。代码：https://github.com/... 

---
# Sens-Merging: Sensitivity-Guided Parameter Balancing for Merging Large Language Models 

**Title (ZH)**: Sens-融合：敏感性引导的参数平衡以合并大型语言模型 

**Authors**: Shuqi Liu, Han Wu, Bowei He, Xiongwei Han, Mingxuan Yuan, Linqin Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.12420)  

**Abstract**: Recent advances in large language models have led to numerous task-specialized fine-tuned variants, creating a need for efficient model merging techniques that preserve specialized capabilities while avoiding costly retraining. While existing task vector-based merging methods show promise, they typically apply uniform coefficients across all parameters, overlooking varying parameter importance both within and across tasks. We present Sens-Merging, a sensitivity-guided coefficient adjustment method that enhances existing model merging techniques by operating at both task-specific and cross-task levels. Our method analyzes parameter sensitivity within individual tasks and evaluates cross-task transferability to determine optimal merging coefficients. Extensive experiments on Mistral 7B and LLaMA2-7B/13B models demonstrate that Sens-Merging significantly improves performance across general knowledge, mathematical reasoning, and code generation tasks. Notably, when combined with existing merging techniques, our method enables merged models to outperform specialized fine-tuned models, particularly in code generation tasks. Our findings reveal important trade-offs between task-specific and cross-task scalings, providing insights for future model merging strategies. 

**Abstract (ZH)**: 近期大型语言模型的进展催生了多种任务特化的微调变体，这促使我们需要开发高效的模型合并技术，以保留专用能力同时避免昂贵的重新训练。虽然现有的基于任务向量的合并方法显示出潜力，但它们通常对所有参数采用均匀系数，忽略了参数在任务内和跨任务间的不同重要性。我们提出了一种基于敏感性的系数调整方法——Sens-Merging，该方法通过任务特定和跨任务两个层面增强现有的模型合并技术。我们的方法分析了单任务中的参数敏感性，并评估其跨任务可迁移性，以确定最优的合并系数。我们在Mistral 7B和LLaMA2-7B/13B模型上的广泛实验表明，Sens-Merging显著提升了通用知识、数学推理和代码生成任务的性能。值得注意的是，结合现有的合并技术后，我们的方法使得合并模型在代码生成任务上优于专用微调模型。我们的研究结果揭示了任务特定和跨任务缩放之间的关键权衡，为未来模型合并策略提供了见解。 

---
# Gradient Co-occurrence Analysis for Detecting Unsafe Prompts in Large Language Models 

**Title (ZH)**: 大型语言模型中不安全提示检测的梯度共现分析 

**Authors**: Jingyuan Yang, Bowen Yan, Rongjun Li, Ziyu Zhou, Xin Chen, Zhiyong Feng, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.12411)  

**Abstract**: Unsafe prompts pose significant safety risks to large language models (LLMs). Existing methods for detecting unsafe prompts rely on data-driven fine-tuning to train guardrail models, necessitating significant data and computational resources. In contrast, recent few-shot gradient-based methods emerge, requiring only few safe and unsafe reference prompts. A gradient-based approach identifies unsafe prompts by analyzing consistent patterns of the gradients of safety-critical parameters in LLMs. Although effective, its restriction to directional similarity (cosine similarity) introduces ``directional bias'', limiting its capability to identify unsafe prompts. To overcome this limitation, we introduce GradCoo, a novel gradient co-occurrence analysis method that expands the scope of safety-critical parameter identification to include unsigned gradient similarity, thereby reducing the impact of ``directional bias'' and enhancing the accuracy of unsafe prompt detection. Comprehensive experiments on the widely-used benchmark datasets ToxicChat and XStest demonstrate that our proposed method can achieve state-of-the-art (SOTA) performance compared to existing methods. Moreover, we confirm the generalizability of GradCoo in detecting unsafe prompts across a range of LLM base models with various sizes and origins. 

**Abstract (ZH)**: 不安全的提示对大型语言模型（LLMs）构成显著的安全风险。现有的不安全提示检测方法依赖于数据驱动的微调来训练护栏模型，这需要大量的数据和计算资源。相比之下，最近出现的少量样本梯度基方法只需要少量的安全和不安全参考提示。基于梯度的方法通过分析安全关键参数在LLMs中的梯度模式来识别不安全的提示，尽管有效，但其对方向相似性（余弦相似性）的限制引入了“方向偏差”，限制了其识别不安全提示的能力。为克服这一限制，我们引入了GradCoo，一种新的梯度共现分析方法，扩展了识别安全关键参数的范围，包括未符号化梯度相似性，从而减少了“方向偏差”的影响，并提高了不安全提示检测的准确性。广泛的实验在广泛使用的基准数据集ToxicChat和XStest上表明，我们提出的方法在不安全提示检测方面达到了现有方法的最新水平。此外，我们证实了GradCoo在各种大小和来源的基础模型中检测不安全提示的一般性。 

---
# Factual Inconsistency in Data-to-Text Generation Scales Exponentially with LLM Size: A Statistical Validation 

**Title (ZH)**: 数据到文本生成中的事实不一致性随LLM规模呈指数增长：一项统计验证 

**Authors**: Joy Mahapatra, Soumyajit Roy, Utpal Garain  

**Link**: [PDF](https://arxiv.org/pdf/2502.12372)  

**Abstract**: Monitoring factual inconsistency is essential for ensuring trustworthiness in data-to-text generation (D2T). While large language models (LLMs) have demonstrated exceptional performance across various D2T tasks, previous studies on scaling laws have primarily focused on generalization error through power law scaling to LLM size (i.e., the number of model parameters). However, no research has examined the impact of LLM size on factual inconsistency in D2T. In this paper, we investigate how factual inconsistency in D2T scales with LLM size by exploring two scaling laws: power law and exponential scaling. To rigorously evaluate and compare these scaling laws, we employ a statistical validation framework consisting of three key stages: predictive performance estimation, goodness-of-fit assessment, and comparative analysis. For a comprehensive empirical study, we analyze three popular LLM families across five D2T datasets, measuring factual inconsistency inversely using four state-of-the-art consistency metrics. Our findings, based on exhaustive empirical results and validated through our framework, reveal that, contrary to the widely assumed power law scaling, factual inconsistency in D2T follows an exponential scaling with LLM size. 

**Abstract (ZH)**: 监测事实不一致性对于确保数据到文本生成任务中的可信度至关重要。虽然大型语言模型在各种数据到文本生成任务中展现了卓越的性能，但以往关于规模定律的研究主要关注通过幂律 scaling 扩展语言模型大小（即模型参数数量）来泛化误差。然而，尚未有研究考察语言模型大小对数据到文本生成中的事实不一致性的影响。本文通过探索两种不同的扩展定律（幂律和指数扩展）来研究语言模型大小对数据到文本生成中事实不一致性的影响。为严格评估和比较这些扩展定律，我们采用了包含三个关键阶段的统计验证框架：预测性能估计、拟合优度评估和比较分析。为进行一项全面的经验研究，我们在五个数据到文本生成数据集中分析了三种流行的语言模型系列，并使用四种最先进的一致性度量方法逆向测量事实不一致性。根据全面的经验结果并通过我们提出的框架进行验证，我们的研究发现，与普遍认为的幂律扩展不同，数据到文本生成中的事实不一致性实际上随语言模型大小呈现指数扩展。 

---
# QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models 

**Title (ZH)**: QuZO: 量化零阶微调用于大型语言模型 

**Authors**: Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12346)  

**Abstract**: Language Models (LLMs) are often quantized to lower precision to reduce the memory cost and latency in inference. However, quantization often degrades model performance, thus fine-tuning is required for various down-stream tasks. Traditional fine-tuning methods such as stochastic gradient descent and Adam optimization require backpropagation, which are error-prone in the low-precision settings. To overcome these limitations, we propose the Quantized Zeroth-Order (QuZO) framework, specifically designed for fine-tuning LLMs through low-precision (e.g., 4- or 8-bit) forward passes. Our method can avoid the error-prone low-precision straight-through estimator, and utilizes optimized stochastic rounding to mitigate the increased bias. QuZO simplifies the training process, while achieving results comparable to first-order methods in ${\rm FP}8$ and superior accuracy in ${\rm INT}8$ and ${\rm INT}4$ training. Experiments demonstrate that low-bit training QuZO achieves performance comparable to MeZO optimization on GLUE, Multi-Choice, and Generation tasks, while reducing memory cost by $2.94 \times$ in LLaMA2-7B fine-tuning compared to quantized first-order methods. 

**Abstract (ZH)**: 语言模型（LLMs）常常被量化到较低精度以降低推理过程中的内存成本和延迟。然而，量化往往会降低模型性能，因此需要对各种下游任务进行微调。传统的微调方法如随机梯度下降和Adam优化需要反向传播，在低精度设置中容易出错。为克服这些限制，我们提出了Quantized Zeroth-Order（QuZO）框架，专门用于通过低精度（如4位或8位）前向传递对LLMs进行微调。该方法可以避免低精度的直接通过估计器带来的误差，并利用优化的随机四舍五入来减轻偏移增加的影响。QuZO简化了训练过程，在FP8训练中达到与一阶方法相当的结果，并在INT8和INT4训练中获得更高的准确性。实验表明，低位数训练QuZO在GLUE、多选择和生成任务上的性能与MeZO优化相当，同时在LLaMA2-7B微调中将内存成本降低3.94倍，相比之下是量化的一阶方法。 

---
# Connecting Large Language Model Agent to High Performance Computing Resource 

**Title (ZH)**: 将大型语言模型代理连接到高性能计算资源 

**Authors**: Heng Ma, Alexander Brace, Carlo Siebenschuh, Greg Pauloski, Ian Foster, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12280)  

**Abstract**: The Large Language Model agent workflow enables the LLM to invoke tool functions to increase the performance on specific scientific domain questions. To tackle large scale of scientific research, it requires access to computing resource and parallel computing setup. In this work, we implemented Parsl to the LangChain/LangGraph tool call setup, to bridge the gap between the LLM agent to the computing resource. Two tool call implementations were set up and tested on both local workstation and HPC environment on Polaris/ALCF. The first implementation with Parsl-enabled LangChain tool node queues the tool functions concurrently to the Parsl workers for parallel execution. The second configuration is implemented by converting the tool functions into Parsl ensemble functions, and is more suitable for large task on super computer environment. The LLM agent workflow was prompted to run molecular dynamics simulations, with different protein structure and simulation conditions. These results showed the LLM agent tools were managed and executed concurrently by Parsl on the available computing resource. 

**Abstract (ZH)**: 大规模语言模型代理工作流使LLM能够调用工具功能以提高特定科学领域问题的性能。为了应对大规模科学研究，需要访问计算资源和并行计算设置。在这项工作中，我们将Parsl集成到LangChain/LangGraph工具调用设置中，以弥合LLM代理与计算资源之间的差距。我们在Polaris/ALCF的本地工作站和高性能计算环境中设置了两种工具调用实现并进行了测试。第一种实现利用Parsl启用的LangChain工具节点将工具函数并发排队到Parsl工作者进行并行执行。第二种配置将工具函数转换为Parsl集成函数，更适合在超级计算机环境中处理大规模任务。LLM代理工作流被提示运行不同蛋白质结构和模拟条件的分子动力学模拟。这些结果表明，LLM代理工具能够在可用的计算资源上由Parsl并发管理和执行。 

---
# Learning to Reason at the Frontier of Learnability 

**Title (ZH)**: 学习能力前沿的推理学习 

**Authors**: Thomas Foster, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2502.12272)  

**Abstract**: Reinforcement learning is now widely adopted as the final stage of large language model training, especially for reasoning-style tasks such as maths problems. Typically, models attempt each question many times during a single training step and attempt to learn from their successes and failures. However, we demonstrate that throughout training with two popular algorithms (PPO and VinePPO) on two widely used datasets, many questions are either solved by all attempts - meaning they are already learned - or by none - providing no meaningful training signal. To address this, we adapt a method from the reinforcement learning literature - sampling for learnability - and apply it to the reinforcement learning stage of LLM training. Our curriculum prioritises questions with high variance of success, i.e. those where the agent sometimes succeeds, but not always. Our findings demonstrate that this curriculum consistently boosts training performance across multiple algorithms and datasets, paving the way for more efficient and effective reinforcement learning in LLMs. 

**Abstract (ZH)**: 强化学习现在被广泛应用于大型语言模型训练的最终阶段，特别是在解决数学问题等推理任务中。通常，模型会在单个训练步中多次尝试每个问题，并从中学习成功的经验和失败的教训。然而，我们在使用两种流行算法（PPO和VinePPO）和两种广泛使用的数据集进行训练的过程中发现，许多问题要么每次都成功解决，意味着这些问题是已经学会的；要么每次都无法解决，无法提供有意义的训练信号。为了解决这个问题，我们借鉴了强化学习文献中的方法——可学习性采样，并将其应用于大型语言模型训练的强化学习阶段。我们的课程设置优先考虑那些成功率具有高变异性的问题，即那些有时能成功，但不总是成功的任务。我们的研究发现表明，这种课程设置能够在多个算法和数据集上持续提升训练性能，为进一步提高大型语言模型的强化学习效率和效果铺平了道路。 

---
# Optimal Brain Iterative Merging: Mitigating Interference in LLM Merging 

**Title (ZH)**: 最优脑迭代合并：减轻LLM合并中的干扰 

**Authors**: Zhixiang Wang, Zhenyu Mao, Yixuan Qiao, Yunfang Wu, Biye Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12217)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities, but their high computational costs pose challenges for customization. Model merging offers a cost-effective alternative, yet existing methods suffer from interference among parameters, leading to performance degradation. In this work, we propose Optimal Brain Iterative Merging (OBIM), a novel method designed to mitigate both intra-model and inter-model interference. OBIM consists of two key components: (1) A saliency measurement mechanism that evaluates parameter importance based on loss changes induced by individual weight alterations, reducing intra-model interference by preserving only high-saliency parameters. (2) A mutually exclusive iterative merging framework, which incrementally integrates models using a binary mask to avoid direct parameter averaging, thereby mitigating inter-model interference. We validate OBIM through experiments on both Supervised Fine-Tuned (SFT) models and post-pretrained checkpoints. The results show that OBIM significantly outperforms existing merging techniques. Overall, OBIM provides an effective and practical solution for enhancing LLM merging. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的能力，但其高昂的计算成本带来了定制化的挑战。模型合并提供了一种成本效益高的替代方案，然而现有方法中存在的参数干扰导致性能下降。在这项工作中，我们提出了最优大脑迭代合并（OBIM），这是一种设计用于缓解模型内的参数干扰和模型间干扰的新方法。OBIM 包含两个关键组件：（1）一个显著性测度机制，该机制基于单个权重改变引起的损失变化评估参数的重要性，通过保留高显著性参数来减少模型内的干扰。（2）一个互斥的迭代合并框架，该框架使用二进制掩码逐步整合模型，从而避免直接的参数平均，进而减少模型间干扰。我们通过在监督微调（SFT）模型和后预训练检查点上的实验验证了 OBIM 的有效性。结果显示，OBIM 显著优于现有合并技术。总体而言，OBIM 提供了一个有效且实用的解决方案，用于增强 LLM 合并。 

---
# Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs 

**Title (ZH)**: 策略: 用于长上下文LLM的自适应稀疏注意机制、聚类与分布拟合 

**Authors**: Kan Zhu, Tian Tang, Qinyu Xu, Yile Gu, Zhichen Zeng, Rohan Kadekodi, Liangyu Zhao, Ang Li, Arvind Krishnamurthy, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2502.12216)  

**Abstract**: Long-context models are essential for many applications but face inefficiencies in loading large KV caches during decoding. Prior methods enforce fixed token budgets for sparse attention, assuming a set number of tokens can approximate full attention. However, these methods overlook variations in the importance of attention across heads, layers, and contexts. To address these limitations, we propose Tactic, a sparsity-adaptive and calibration-free sparse attention mechanism that dynamically selects tokens based on their cumulative attention scores rather than a fixed token budget. By setting a target fraction of total attention scores, Tactic ensures that token selection naturally adapts to variations in attention sparsity. To efficiently approximate this selection, Tactic leverages clustering-based sorting and distribution fitting, allowing it to accurately estimate token importance with minimal computational overhead. We show that Tactic outperforms existing sparse attention algorithms, achieving superior accuracy and up to 7.29x decode attention speedup. This improvement translates to an overall 1.58x end-to-end inference speedup, making Tactic a practical and effective solution for long-context LLM inference in accuracy-sensitive applications. 

**Abstract (ZH)**: 长上下文模型对于许多应用至关重要，但在解码过程中面临着加载大型KV缓存的效率问题。此前的方法强制执行固定tokens预算的稀疏注意机制，假设一定数量的tokens可以近似全注意。然而，这些方法忽略了注意在头、层和上下文之间的重要性变化。为了解决这些限制，我们提出了Tactic，一种自适应稀疏注意机制，能够根据tokens的累积注意分数动态选择tokens，而不是固定tokens预算。通过设置总注意分数的目标比例，Tactic确保tokens选择能够自然适应注意稀疏性的变化。为了高效地近似这种选择，Tactic利用基于聚类的排序和分布拟合，能够以最小的计算开销准确估计token的重要性。实验结果表明，Tactic优于现有稀疏注意算法，实现了更高的准确性和高达7.29倍的解码注意加速。这种改进转化为整体1.58倍的端到端推理加速，使Tactic成为在准确性敏感应用中进行长上下文LLM推理的实用且有效的解决方案。 

---
# Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities? 

**Title (ZH)**: 重访o1-like模型的测试时缩放能力：它们真的具备测试时缩放能力吗？ 

**Authors**: Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Yunhua Zhou, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12215)  

**Abstract**: The advent of test-time scaling in large language models (LLMs), exemplified by OpenAI's o1 series, has advanced reasoning capabilities by scaling computational resource allocation during inference. While successors like QwQ, Deepseek-R1 (R1) and LIMO replicate these advancements, whether these models truly possess test-time scaling capabilities remains underexplored. This study found that longer CoTs of these o1-like models do not consistently enhance accuracy; in fact, correct solutions are often shorter than incorrect ones for the same questions. Further investigation shows this phenomenon is closely related to models' self-revision capabilities - longer CoTs contain more self-revisions, which often lead to performance degradation. We then compare sequential and parallel scaling strategies on QwQ, R1 and LIMO, finding that parallel scaling achieves better coverage and scalability. Based on these insights, we propose Shortest Majority Vote, a method that combines parallel scaling strategies with CoT length characteristics, significantly improving models' test-time scalability compared to conventional majority voting approaches. 

**Abstract (ZH)**: 大语言模型（LLMs）测试时缩放的出现及其应用：OpenAI o1系列的示例，通过在推理过程中扩展计算资源分配提升了推理能力。尽管后续模型如QwQ、Deepseek-R1 (R1) 和 LIMO 复现了这些进步，但这些模型是否真正具备测试时缩放能力仍待进一步探索。本研究发现，这些类似o1的模型更长的中间步骤（CoT）并不一致地提升准确性；事实上，对于相同的提问，正确答案往往比错误答案更短。进一步的研究表明，这一现象与模型的自我修订能力密切相关——更长的中间步骤包含更多的自我修订，而这往往导致性能下降。我们还对比了QwQ、R1和LIMO上顺序和并行缩放策略，发现并行缩放策略在覆盖率和可扩展性上更优。基于这些见解，我们提出了一种名为“最短多数投票”的方法，该方法结合了并行缩放策略和中间步骤长度的特点，显著提高了模型的测试时可缩放性，优于传统的多数投票方法。 

---
# Zero Token-Driven Deep Thinking in LLMs: Unlocking the Full Potential of Existing Parameters via Cyclic Refinement 

**Title (ZH)**: 零令牌驱动的深度思考在LLMs中的实现：通过循环精炼解锁现有参数的全部潜力 

**Authors**: Guanghao Li, Wenhao Jiang, Li Shen, Ming Tang, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12214)  

**Abstract**: Resource limitations often constrain the parameter counts of Large Language Models (LLMs), hindering their performance. While existing methods employ parameter sharing to reuse the same parameter set under fixed budgets, such approaches typically force each layer to assume multiple roles with a predetermined number of iterations, restricting efficiency and adaptability. In this work, we propose the Zero Token Transformer (ZTT), which features a head-tail decoupled parameter cycling method. We disentangle the first (head) and last (tail) layers from parameter cycling and iteratively refine only the intermediate layers. Furthermore, we introduce a Zero-Token Mechanism, an internal architectural component rather than an input token, to guide layer-specific computation. At each cycle, the model retrieves a zero token (with trainable key values) from a Zero-Token Pool, integrating it alongside regular tokens in the attention mechanism. The corresponding attention scores not only reflect each layer's computational importance but also enable dynamic early exits without sacrificing overall model accuracy. Our approach achieves superior performance under tight parameter budgets, effectively reduces computational overhead via early exits, and can be readily applied to fine-tune existing pre-trained models for enhanced efficiency and adaptability. 

**Abstract (ZH)**: 零令牌变换器：一种头部-尾部解耦的参数循环方法及其应用 

---
# An Interpretable Automated Mechanism Design Framework with Large Language Models 

**Title (ZH)**: 一种具有解释性的自动化机制设计框架——大型语言模型的应用 

**Authors**: Jiayuan Liu, Mingyu Guo, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2502.12203)  

**Abstract**: Mechanism design has long been a cornerstone of economic theory, with traditional approaches relying on mathematical derivations. Recently, automated approaches, including differentiable economics with neural networks, have emerged for designing payments and allocations. While both analytical and automated methods have advanced the field, they each face significant weaknesses: mathematical derivations are not automated and often struggle to scale to complex problems, while automated and especially neural-network-based approaches suffer from limited interpretability. To address these challenges, we introduce a novel framework that reformulates mechanism design as a code generation task. Using large language models (LLMs), we generate heuristic mechanisms described in code and evolve them to optimize over some evaluation metrics while ensuring key design criteria (e.g., strategy-proofness) through a problem-specific fixing process. This fixing process ensures any mechanism violating the design criteria is adjusted to satisfy them, albeit with some trade-offs in performance metrics. These trade-offs are factored in during the LLM-based evolution process. The code generation capabilities of LLMs enable the discovery of novel and interpretable solutions, bridging the symbolic logic of mechanism design and the generative power of modern AI. Through rigorous experimentation, we demonstrate that LLM-generated mechanisms achieve competitive performance while offering greater interpretability compared to previous approaches. Notably, our framework can rediscover existing manually designed mechanisms and provide insights into neural-network based solutions through Programming-by-Example. These results highlight the potential of LLMs to not only automate but also enhance the transparency and scalability of mechanism design, ensuring safe deployment of the mechanisms in society. 

**Abstract (ZH)**: 机制设计-long一直是经济理论的基石，传统方法依赖于数学推导。近年来，包括神经网络在内的可微经济方法新兴用于设计支付和分配。虽然分析方法和自动化方法都推动了该领域的进展，但各自都面临重大挑战：数学推导方法缺乏自动化，且难以扩展到复杂问题，而自动化尤其是基于神经网络的方法则在可解释性方面受到限制。为应对这些挑战，我们提出了一种新型框架，将机制设计重新表述为代码生成任务。利用大型语言模型（LLMs），我们生成描述在代码中的启发式机制，并通过特定问题的调整过程优化某些评估指标，同时确保关键设计标准（例如， truthful机制性）。该调整过程确保任何违反设计标准的机制被调整以满足这些标准，尽管在某些性能指标上可能存在权衡。这些权衡会在LLM驱动的进化过程中加以考虑。LLMs的代码生成能力使发现新颖且可解释的解决方案成为可能，将机制设计的符号逻辑与现代AI的生成能力相结合。通过严格的实验，我们证明LLM生成的机制在性能上具有竞争力，同时比之前的方法更具可解释性。值得注意的是，我们的框架可以重新发现已手动设计的机制，并通过示例编程为基于神经网络的解决方案提供见解。这些结果突显了LLMs不仅能够自动化，还可以增强机制设计的透明性和可扩展性，确保机制在社会中的安全部署。 

---
# BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack 

**Title (ZH)**: BoT: 通过后门攻击打破o1-like大型语言模型的长思考过程 

**Authors**: Zihao Zhu, Hongbao Zhang, Mingda Zhang, Ruotong Wang, Guanzong Wu, Ke Xu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12202)  

**Abstract**: Longer thought, better performance: large language models with deep reasoning capabilities, particularly o1-like models, have demonstrated remarkable performance by generating extensive thought processes during inference. This trade-off reveals a potential vulnerability: adversaries could compromise model performance by forcing immediate responses without thought processes. To this end, in this paper, we introduce a novel attack scenario targeting the long thought processes of o1-like models and propose BoT (Break CoT), which can selectively break intrinsic reasoning mechanisms through backdoor attacks. BoT constructs poisoned datasets with designed triggers and injects backdoor by either supervised fine-tuning or direct preference optimization. When triggered, the model directly generates answers without thought processes, while maintaining normal reasoning capabilities for clean inputs. Extensive experiments on open-source o1-like models, including recent DeepSeek-R1, demonstrate that BoT nearly achieves high attack success rates while maintaining clean accuracy, highlighting the critical safety risk in current models. Furthermore, the relationship between task difficulty and helpfulness reveals a potential application for good, enabling users to customize model behavior based on task complexity. Code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 更深入思考，更好 performance：具有深度推理能力的大语言模型，特别是 o1 类模型，在推断过程中生成 extensive 思考过程时表现出卓越的性能。这一权衡揭示了一个潜在的安全性弱点：攻击者可以通过迫使模型立即响应而无需思考过程来削弱模型性能。基于此，本文提出了一种新的攻击场景，针对 o1 类模型的 long thought 过程，并提出了一种新型攻击方法 BoT（Break CoT），该方法可以通过后门攻击选择性地破坏内在的推理机制。BoT 通过设计触发器构造受污染的数据集，并通过监督微调或直接偏好优化注入后门。当受到触发时，模型将直接生成答案而无需思考过程，但对干净的输入保持正常的推理能力。在开源 o1 类模型上的广泛实验，包括最近的 DeepSeek-R1，证实 BoT 几乎实现了高攻击成功率，同时保持了干净的准确性，突显了当前模型中的关键安全风险。此外，任务难度与帮助度之间的关系揭示了一种潜在的应用场景，允许用户根据任务复杂性自定义模型行为。代码可在 \href{this https URL}{this https URL} 获取。 

---
# A Closer Look at System Prompt Robustness 

**Title (ZH)**: 系统提示稳健性进一步探究 

**Authors**: Norman Mu, Jonathan Lu, Michael Lavery, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2502.12197)  

**Abstract**: System prompts have emerged as a critical control surface for specifying the behavior of LLMs in chat and agent settings. Developers depend on system prompts to specify important context, output format, personalities, guardrails, content policies, and safety countermeasures, all of which require models to robustly adhere to the system prompt, especially when facing conflicting or adversarial user inputs. In practice, models often forget to consider relevant guardrails or fail to resolve conflicting demands between the system and the user. In this work, we study various methods for improving system prompt robustness by creating realistic new evaluation and fine-tuning datasets based on prompts collected from from OpenAI's GPT Store and HuggingFace's HuggingChat. Our experiments assessing models with a panel of new and existing benchmarks show that performance can be considerably improved with realistic fine-tuning data, as well as inference-time interventions such as classifier-free guidance. Finally, we analyze the results of recently released reasoning models from OpenAI and DeepSeek, which show exciting but uneven improvements on the benchmarks we study. Overall, current techniques fall short of ensuring system prompt robustness and further study is warranted. 

**Abstract (ZH)**: 系统提示已成为指定聊天和代理环境中LLM行为的关键控制面。开发人员依赖系统提示来指定重要背景、输出格式、个性、护栏、内容政策和安全对策，所有这些都需要模型在面对冲突或 adversarial 用户输入时能稳健地遵循系统提示。实际上，模型往往会忽略相关护栏，或无法解决系统与用户之间的冲突需求。在这项工作中，我们通过基于从OpenAI的GPT Store和HuggingFace的HuggingChat收集的提示创建现实的新评估和微调数据集，研究了各种提高系统提示稳健性的方法。我们的实验使用新的和现有的基准测试面板评估模型表明，借助现实的微调数据以及推断时的干预措施（如无分类引导），性能可以显著提高。最后，我们分析了最近由OpenAI和DeepSeek发布的推理模型的结果，这些结果显示了在我们研究的基准测试中令人兴奋但不均匀的改进。总体而言，当前的技术尚不足以确保系统提示的稳健性，需要进一步的研究。 

---
# Large Language Models for Extrapolative Modeling of Manufacturing Processes 

**Title (ZH)**: 大型语言模型在制造过程外推性建模中的应用 

**Authors**: Kiarash Naghavi Khanghah, Anandkumar Patel, Rajiv Malhotra, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12185)  

**Abstract**: Conventional predictive modeling of parametric relationships in manufacturing processes is limited by the subjectivity of human expertise and intuition on the one hand and by the cost and time of experimental data generation on the other hand. This work addresses this issue by establishing a new Large Language Model (LLM) framework. The novelty lies in combining automatic extraction of process-relevant knowledge embedded in the literature with iterative model refinement based on a small amount of experimental data. This approach is evaluated on three distinct manufacturing processes that are based on machining, deformation, and additive principles. The results show that for the same small experimental data budget the models derived by our framework have unexpectedly high extrapolative performance, often surpassing the capabilities of conventional Machine Learning. Further, our approach eliminates manual generation of initial models or expertise-dependent interpretation of the literature. The results also reveal the importance of the nature of the knowledge extracted from the literature and the significance of both the knowledge extraction and model refinement components. 

**Abstract (ZH)**: 传统的制造过程参数关系预测建模受限于人类专业知识和直觉的主观性以及实验数据生成的高成本和长周期。本研究通过建立一个新的大型语言模型（LLM）框架来解决这一问题。其创新点在于结合了从文献中自动提取与制造过程相关知识，并基于少量实验数据进行迭代模型优化。该方法在基于加工、变形和增材原理的三个不同制造过程中进行了评估。结果表明，在相同的少量实验数据预算下，由框架构建的模型表现出意外的外推性能，常常超越传统机器学习的能力。此外，该方法消除了手工生成初始模型或依赖专业知识解释文献的步骤。研究结果还揭示了从文献中提取的知识性质以及知识提取和模型优化两个方面的重要性。 

---
# Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts 

**Title (ZH)**: 基于多概念转移的稀疏自编码可识别转向 

**Authors**: Shruti Joshi, Andrea Dittadi, Sébastien Lachapelle, Dhanya Sridhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12179)  

**Abstract**: Steering methods manipulate the representations of large language models (LLMs) to induce responses that have desired properties, e.g., truthfulness, offering a promising approach for LLM alignment without the need for fine-tuning. Traditionally, steering has relied on supervision, such as from contrastive pairs of prompts that vary in a single target concept, which is costly to obtain and limits the speed of steering research. An appealing alternative is to use unsupervised approaches such as sparse autoencoders (SAEs) to map LLM embeddings to sparse representations that capture human-interpretable concepts. However, without further assumptions, SAEs may not be identifiable: they could learn latent dimensions that entangle multiple concepts, leading to unintentional steering of unrelated properties. We introduce Sparse Shift Autoencoders (SSAEs) that instead map the differences between embeddings to sparse representations. Crucially, we show that SSAEs are identifiable from paired observations that vary in \textit{multiple unknown concepts}, leading to accurate steering of single concepts without the need for supervision. We empirically demonstrate accurate steering across semi-synthetic and real-world language datasets using Llama-3.1 embeddings. 

**Abstract (ZH)**: Sparse Shift Autoencoders操纵大语言模型的表示以诱导具有 desired 属性的响应，为无需微调的大语言模型对齐提供了有 promise 的方法。传统上，操纵依赖于监督，例如在单一目标概念上变化的对比提示对，但这种方式成本高且限制了操纵研究的速度。一种令人振奋的替代方案是使用稀疏自动编码器（SAEs）将大语言模型嵌入映射到稀疏表示，以捕捉可由人类理解的概念。然而，在不做进一步假设的情况下，SAEs 可能是不可识别的：它们可能会学习互相关概念的潜在维度，导致无意中操纵无关的属性。我们引入了稀疏偏移自动编码器（SSAEs），它们将嵌入之间的差异映射到稀疏表示。关键的是，我们证明了当配对观察在多个未知概念上变化时，SSAEs 是可识别的，从而能够在无需监督的情况下准确地操纵单个概念。我们使用 Llama-3.1 嵌入在半合成和真实世界语言数据集上 empirically 证明了准确的操纵。 

---
# nanoML for Human Activity Recognition 

**Title (ZH)**: nanoML在人体活动识别中的应用 

**Authors**: Alan T. L. Bacellar, Mugdha P. Jadhao, Shashank Nag, Priscila M. V. Lima, Felipe M. G. Franca, Lizy K. John  

**Link**: [PDF](https://arxiv.org/pdf/2502.12173)  

**Abstract**: Human Activity Recognition (HAR) is critical for applications in healthcare, fitness, and IoT, but deploying accurate models on resource-constrained devices remains challenging due to high energy and memory demands. This paper demonstrates the application of Differentiable Weightless Neural Networks (DWNs) to HAR, achieving competitive accuracies of 96.34% and 96.67% while consuming only 56nJ and 104nJ per sample, with an inference time of just 5ns per sample. The DWNs were implemented and evaluated on an FPGA, showcasing their practical feasibility for energy-efficient hardware deployment. DWNs achieve up to 926,000x energy savings and 260x memory reduction compared to state-of-the-art deep learning methods. These results position DWNs as a nano-machine learning nanoML model for HAR, setting a new benchmark in energy efficiency and compactness for edge and wearable devices, paving the way for ultra-efficient edge AI. 

**Abstract (ZH)**: Differentiable Weightless Neural Networks for Energy-Efficient Human Activity Recognition 

---
# GoRA: Gradient-driven Adaptive Low Rank Adaptation 

**Title (ZH)**: GoRA: 梯度驱动的自适应低秩适应 

**Authors**: Haonan He, Peng Ye, Yuchen Ren, Yuan Yuan, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.12171)  

**Abstract**: Low-Rank Adaptation (LoRA) is a crucial method for efficiently fine-tuning pretrained large language models (LLMs), with its performance largely influenced by two key factors: rank and initialization strategy. Numerous LoRA variants have been proposed to enhance its performance by addressing these factors. However, these variants often compromise LoRA's usability or efficiency. In this paper, we analyze the fundamental limitations of existing methods and introduce a novel approach, GoRA (Gradient-driven Adaptive Low Rank Adaptation), which adaptively assigns ranks and initializes weights for low-rank adapters simultaneously based on gradient information. Extensive experimental results demonstrate that GoRA significantly improves performance while preserving the high usability and efficiency of LoRA. On the T5 model fine-tuned for the GLUE benchmark, GoRA achieves a 5.88-point improvement over LoRA and slightly surpasses full fine-tuning. Similarly, on the Llama3.1-8B-Base model fine-tuned for GSM8k tasks, GoRA outperforms LoRA with a 5.13-point improvement and exceeds full fine-tuning in high-rank settings by a margin of 2.05 points. 

**Abstract (ZH)**: 基于梯度的自适应低秩适应 (GoRA): 同时基于梯度信息自适应分配秩和初始化权重以提升预训练大型语言模型的高效微调Performance Improvement of Low-Rank Adaptation (LoRA) via Gradient-driven Adaptive Low Rank Adaptation (GoRA): Simultaneous Adaptive Rank Assignment and Weight Initialization Based on Gradient Information 

---
# MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections 

**Title (ZH)**: MUDDFormer: 通过多向动态密集连接打破残差瓶颈的变压器 

**Authors**: Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12170)  

**Abstract**: We propose MUltiway Dynamic Dense (MUDD) connections, a simple yet effective method to address the limitations of residual connections and enhance cross-layer information flow in Transformers. Unlike existing dense connection approaches with static and shared connection weights, MUDD generates connection weights dynamically depending on hidden states at each sequence position and for each decoupled input stream (the query, key, value or residual) of a Transformer block. MUDD connections can be seamlessly integrated into any Transformer architecture to create MUDDFormer. Extensive experiments show that MUDDFormer significantly outperforms Transformers across various model architectures and scales in language modeling, achieving the performance of Transformers trained with 1.8X-2.4X compute. Notably, MUDDPythia-2.8B matches Pythia-6.9B in pretraining ppl and downstream tasks and even rivals Pythia-12B in five-shot settings, while adding only 0.23% parameters and 0.4% computation. Code in JAX and PyTorch and pre-trained models are available at this https URL . 

**Abstract (ZH)**: 我们提出了一种简单而有效的多路动态密集（MUDD）连接方法，以解决残差连接的局限性并增强Transformer中的跨层信息流动。MUDD连接可以根据Transformer块中每个序列位置和每个解耦输入流（查询、键、值或残差）的隐藏状态动态生成连接权重，不同于现有具有静态和共享连接权重的密集连接方法。MUDD连接可以无缝集成到任何Transformer架构中，创建MUDDFormer。广泛实验表明，MUDDFormer在各种模型架构和规模的语言建模中显著优于Transformer，实现相当于使用1.8至2.4倍计算量训练的Transformer的性能。值得注意的是，在预训练语言建模和下游任务中，MUDDPythia-2.8B与Pythia-6.9B性能相同，并且在五 shot 设置下甚至与Pythia-12B相当，同时仅增加0.23%的参数量和0.4%的计算量。JAX和PyTorch代码及预训练模型可在以下链接获取。 

---
# Mining Social Determinants of Health for Heart Failure Patient 30-Day Readmission via Large Language Model 

**Title (ZH)**: 基于大型语言模型的心力衰竭患者30天再住院的社会决定因素挖掘 

**Authors**: Mingchen Shao, Youjeong Kang, Xiao Hu, Hyunjung Gloria Kwak, Carl Yang, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12158)  

**Abstract**: Heart Failure (HF) affects millions of Americans and leads to high readmission rates, posing significant healthcare challenges. While Social Determinants of Health (SDOH) such as socioeconomic status and housing stability play critical roles in health outcomes, they are often underrepresented in structured EHRs and hidden in unstructured clinical notes. This study leverages advanced large language models (LLMs) to extract SDOHs from clinical text and uses logistic regression to analyze their association with HF readmissions. By identifying key SDOHs (e.g. tobacco usage, limited transportation) linked to readmission risk, this work also offers actionable insights for reducing readmissions and improving patient care. 

**Abstract (ZH)**: 心力衰竭（HF）影响着数百万美国人群，并导致高再住院率，提出了重大医疗保健挑战。尽管社会决定因素（SDOH）如社会经济地位和住房稳定性对健康结果发挥着关键作用，但在结构化的电子健康记录（EHRs）中它们往往被忽视，并且隐藏在非结构化的临床笔记中。本研究利用先进的大规模语言模型（LLMs）从临床文本中提取SDOH，并使用逻辑回归分析其与HF再住院之间的关联。通过识别与再住院风险相关的关键SDOH（如烟草使用、有限的交通运输），本文还提供了减少再住院并提升患者护理的实际见解。 

---
