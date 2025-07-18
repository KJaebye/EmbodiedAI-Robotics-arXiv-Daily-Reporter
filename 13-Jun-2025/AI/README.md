# Spurious Rewards: Rethinking Training Signals in RLVR 

**Title (ZH)**: 虚假奖励：重新思考RLVR中的训练信号 

**Authors**: Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.10947)  

**Abstract**: We show that reinforcement learning with verifiable rewards (RLVR) can elicit strong mathematical reasoning in certain models even with spurious rewards that have little, no, or even negative correlation with the correct answer. For example, RLVR improves MATH-500 performance for Qwen2.5-Math-7B in absolute points by 21.4% (random reward), 13.8% (format reward), 24.1% (incorrect label), 26.0% (1-shot RL), and 27.1% (majority voting) -- nearly matching the 29.1% gained with ground truth rewards. However, the spurious rewards that work for Qwen often fail to yield gains with other model families like Llama3 or OLMo2. In particular, we find code reasoning -- thinking in code without actual code execution -- to be a distinctive Qwen2.5-Math behavior that becomes significantly more frequent after RLVR, from 65% to over 90%, even with spurious rewards. Overall, we hypothesize that, given the lack of useful reward signal, RLVR must somehow be surfacing useful reasoning representations learned during pretraining, although the exact mechanism remains a topic for future work. We suggest that future RLVR research should possibly be validated on diverse models rather than a single de facto choice, as we show that it is easy to get significant performance gains on Qwen models even with completely spurious reward signals. 

**Abstract (ZH)**: 我们展示了一种验证性奖励（RLVR）强化学习即使在与正确答案相关性很低、无相关或甚至负相关的虚假奖励下，仍能在某些模型中激发强烈的数学推理能力。强化学习与可验证奖励（RLVR）提高了Qwen2.5-Math-7B在MATH-500上的性能，分别在随机奖励、格式奖励、错误标签、单次RL和其他模型的多数投票中提高了21.4%、13.8%、24.1%、26.0%和27.1%——几乎与使用真实奖励信号获得的29.1%的改进持平。然而，适用于Qwen的虚假奖励往往不能为其他模型家族如Llama3或OLMo2带来提升。特别是，我们发现代码推理——在没有实际代码执行的情况下思考代码——是Qwen2.5-Math的独特行为，在RLVR后变得更为频繁，即使在虚假奖励下，这一比例也从65%上升到超过90%。总体而言，我们推测由于缺乏有用的奖励信号，RLVR必须以某种方式揭示预训练中学到的有用推理表示，但确切机制仍有待未来研究探讨。我们建议未来的RLVR研究应该在多种模型上进行验证，而不仅仅是在单一默认选择上，因为我们展示了即使在完全虚假的奖励信号下，也能在Qwen模型上获得显著性能提升。 

---
# Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification? 

**Title (ZH)**: 打破不良分子：MLLMs准备好进行结构层面的分子去毒处理了吗？ 

**Authors**: Fei Lin, Ziyang Gong, Cong Wang, Yonglin Tian, Tengchao Zhang, Xue Yang, Gen Luo, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10912)  

**Abstract**: Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair - generating structurally valid molecular alternatives with reduced toxicity - has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 560 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a high-throughput evaluation chain for repair success. We systematically assess nearly 30 mainstream general-purpose MLLMs and design multiple ablation studies to analyze key factors such as evaluation criteria, candidate diversity, and failure attribution. Experimental results show that although current MLLMs still face significant challenges on this task, they begin to demonstrate promising capabilities in toxicity understanding, semantic constraint adherence, and structure-aware molecule editing. 

**Abstract (ZH)**: 毒性仍然是早期药物开发失败的主要原因之一。尽管在分子设计和性质预测方面取得了进展，但分子毒性修复任务——生成结构有效且毒性降低的分子替代物——仍未被系统定义和基准测试。为填补这一空白，我们介绍了ToxiMol，这是第一个专注于分子毒性修复的一般用途多模态大型语言模型基准任务。我们构建了一个标准化数据集，涵盖了11个主要任务和560个具有多样化机制和粒度的代表有毒分子。我们设计了一种具备机制感知和任务适应能力的提示标注管道，该管道由专家毒理学知识指导。同时，我们提出了一个自动评估框架ToxiEval，将毒性终点预测、合成可及性、药效化学性和结构相似性整合到一个高通量评估链中，用于修复成功率评估。我们系统评估了近30种主流一般用途MLLMs，并设计了多种消融研究，以分析评估标准、候选多样性和失败归因等关键因素。实验结果表明，尽管目前的MLLMs在该任务上仍面临显著挑战，但它们开始在毒性理解、语义约束遵守和结构感知分子编辑方面显示出有希望的能力。 

---
# GenPlanX. Generation of Plans and Execution 

**Title (ZH)**: GenPlanX. 计划的生成与执行 

**Authors**: Daniel Borrajo, Giuseppe Canonaco, Tomás de la Rosa, Alfredo Garrachón, Sriram Gopalakrishnan, Simerjot Kaur, Marianela Morales, Sunandita Patra, Alberto Pozanco, Keshav Ramani, Charese Smiley, Pietro Totis, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2506.10897)  

**Abstract**: Classical AI Planning techniques generate sequences of actions for complex tasks. However, they lack the ability to understand planning tasks when provided using natural language. The advent of Large Language Models (LLMs) has introduced novel capabilities in human-computer interaction. In the context of planning tasks, LLMs have shown to be particularly good in interpreting human intents among other uses. This paper introduces GenPlanX that integrates LLMs for natural language-based description of planning tasks, with a classical AI planning engine, alongside an execution and monitoring framework. We demonstrate the efficacy of GenPlanX in assisting users with office-related tasks, highlighting its potential to streamline workflows and enhance productivity through seamless human-AI collaboration. 

**Abstract (ZH)**: 古典AI规划技术生成复杂任务的动作序列。然而，它们在仅通过自然语言提供规划任务时缺乏理解和处理的能力。大型语言模型（LLMs）的出现引入了人机交互的新能力。在规划任务的背景下，LLMs特别擅长于解释人类的意图等其他用途。本文介绍了一种名为GenPlanX的方法，该方法结合了LLMs用于基于自然语言描述规划任务，以及一个经典的AI规划引擎和执行与监控框架。我们展示了GenPlanX在帮助用户完成办公相关任务方面的有效性，突显了其通过无缝的人机协作简化工作流程和提高 productivity 的潜力。 

---
# A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models 

**Title (ZH)**: 基于MCP增强的链式思考大语言模型的个体时空活动生成方法研究 

**Authors**: Yu Zhang, Yang Hu, De Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10853)  

**Abstract**: Human spatiotemporal behavior simulation is critical for urban planning research, yet traditional rule-based and statistical approaches suffer from high computational costs, limited generalizability, and poor scalability. While large language models (LLMs) show promise as "world simulators," they face challenges in spatiotemporal reasoning including limited spatial cognition, lack of physical constraint understanding, and group homogenization tendencies. This paper introduces a framework integrating chain-of-thought (CoT) reasoning with Model Context Protocol (MCP) to enhance LLMs' capability in simulating spatiotemporal behaviors that correspond with validation data patterns. The methodology combines human-like progressive reasoning through a five-stage cognitive framework with comprehensive data processing via six specialized MCP tool categories: temporal management, spatial navigation, environmental perception, personal memory, social collaboration, and experience evaluation. Experiments in Shanghai's Lujiazui district validate the framework's effectiveness across 1,000 generated samples. Results demonstrate high similarity with real mobile signaling data, achieving generation quality scores of 7.86 to 8.36 across different base models. Parallel processing experiments show efficiency improvements, with generation times decreasing from 1.30 to 0.17 minutes per sample when scaling from 2 to 12 processes. This work contributes to integrating CoT reasoning with MCP for urban behavior modeling, advancing LLMs applications in urban computing and providing a practical approach for synthetic mobility data generation. The framework offers a foundation for smart city planning, transportation forecasting, and participatory urban design applications. 

**Abstract (ZH)**: 基于链式推理和Model Context Protocol的大型语言模型时空行为模拟框架 

---
# OPT-BENCH: Evaluating LLM Agent on Large-Scale Search Spaces Optimization Problems 

**Title (ZH)**: OPT-BENCH: 大规模搜索空间优化问题中评估LLM代理模型 

**Authors**: Xiaozhe Li, Jixuan Chen, Xinyu Fang, Shengyuan Ding, Haodong Duan, Qingwen Liu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10764)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in solving diverse tasks. However, their proficiency in iteratively optimizing complex solutions through learning from previous feedback remains insufficiently explored. To bridge this gap, we present OPT-BENCH, a comprehensive benchmark designed to evaluate LLM agents on large-scale search space optimization problems. OPT-BENCH includes 20 real-world machine learning tasks sourced from Kaggle and 10 classical NP problems, offering a diverse and challenging environment for assessing LLM agents on iterative reasoning and solution refinement. To enable rigorous evaluation, we introduce OPT-Agent, an end-to-end optimization framework that emulates human reasoning when tackling complex problems by generating, validating, and iteratively improving solutions through leveraging historical feedback. Through extensive experiments on 9 state-of-the-art LLMs from 6 model families, we analyze the effects of optimization iterations, temperature settings, and model architectures on solution quality and convergence. Our results demonstrate that incorporating historical context significantly enhances optimization performance across both ML and NP tasks. All datasets, code, and evaluation tools are open-sourced to promote further research in advancing LLM-driven optimization and iterative reasoning. Project page: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决多样化任务方面显示出了显著的能力。然而，它们在通过学习之前反馈逐步优化复杂解决方案方面的熟练程度尚未充分探索。为填补这一空白，我们提出了OPT-BENCH，一个综合基准，旨在评估LLM代理在大规模搜索空间优化问题上的性能。OPT-BENCH包含来自Kaggle的20个真实世界机器学习任务和10个经典NP问题，提供了一个多样且具有挑战性的环境，用于评估LLM代理的迭代推理和解精确化。为了实现严格的评估，我们引入了OPT-Agent，这是一个端到端的优化框架，通过利用历史反馈生成、验证和逐步改进解决方案，模拟人类解决复杂问题的推理过程。通过对6个模型家族中的9个最先进的LLM进行大量实验，我们分析了优化迭代次数、温度设置和模型架构对解质量和收敛性的影响。我们的结果表明，历史上下文的整合显著提升了在机器学习和NP任务上的优化性能。所有数据集、代码和评估工具均已开源，以促进LLM驱动的优化和迭代推理研究的进一步发展。项目页面：[这个链接](这个链接)。 

---
# Think before You Simulate: Symbolic Reasoning to Orchestrate Neural Computation for Counterfactual Question Answering 

**Title (ZH)**: 深思而后模拟：符号推理 orchestrating 神经计算以进行反事实问题回答 

**Authors**: Adam Ishay, Zhun Yang, Joohyung Lee, Ilgu Kang, Dongjae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10753)  

**Abstract**: Causal and temporal reasoning about video dynamics is a challenging problem. While neuro-symbolic models that combine symbolic reasoning with neural-based perception and prediction have shown promise, they exhibit limitations, especially in answering counterfactual questions. This paper introduces a method to enhance a neuro-symbolic model for counterfactual reasoning, leveraging symbolic reasoning about causal relations among events. We define the notion of a causal graph to represent such relations and use Answer Set Programming (ASP), a declarative logic programming method, to find how to coordinate perception and simulation modules. We validate the effectiveness of our approach on two benchmarks, CLEVRER and CRAFT. Our enhancement achieves state-of-the-art performance on the CLEVRER challenge, significantly outperforming existing models. In the case of the CRAFT benchmark, we leverage a large pre-trained language model, such as GPT-3.5 and GPT-4, as a proxy for a dynamics simulator. Our findings show that this method can further improve its performance on counterfactual questions by providing alternative prompts instructed by symbolic causal reasoning. 

**Abstract (ZH)**: 关于视频动态的因果和时间推理是一个具有挑战性的问题。虽然结合符号推理与基于神经网络的感知和预测的神经-符号模型显示出了潜力，但在回答反事实问题时表现出局限性。本文提出了一种增强神经-符号模型以进行反事实推理的方法，利用事件之间因果关系的符号推理。我们定义因果图来表示这些关系，并使用回答集编程（ASP），一种声明式逻辑编程方法，来找出感知和模拟模块的协调方式。我们在CLEVRER和CRAFT两个基准上验证了该方法的有效性。在CLEVRER挑战中，我们的增强方法达到了最先进的性能，显著优于现有模型。对于CRAFT基准，我们利用大型预训练语言模型，如GPT-3.5和GPT-4，作为动力学模拟器的代理。我们的研究结果表明，通过由符号因果推理提供的替代提示，这种方法可以在反事实问题上进一步提高其性能。 

---
# System ASPMT2SMT:Computing ASPMT Theories by SMT Solvers 

**Title (ZH)**: System ASPMT2SMT: 由SMT求解器计算ASPMT理论 

**Authors**: Michael Bartholomew, Joohyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.10708)  

**Abstract**: Answer Set Programming Modulo Theories (ASPMT) is an approach to combining answer set programming and satisfiability modulo theories based on the functional stable model semantics. It is shown that the tight fragment of ASPMT programs can be turned into SMT instances, thereby allowing SMT solvers to compute stable models of ASPMT programs. In this paper we present a compiler called {\sc aspsmt2smt}, which implements this translation. The system uses ASP grounder {\sc gringo} and SMT solver {\sc z3}. {\sc gringo} partially grounds input programs while leaving some variables to be processed by {\sc z3}. We demonstrate that the system can effectively handle real number computations for reasoning about continuous changes. 

**Abstract (ZH)**: Answer Set Programming Modulo Theories (基于理论的回答集编程)是一种结合回答集编程和理论饱和度的基础上的功能稳定模型语义的方法。证明了ASPMT的紧密片段可以转换为SMT实例，从而允许SMT求解器计算ASPMT程序的稳定模型。本文介绍了一个名为aspsmt2smt的编译器，实现了这一转换。该系统使用ASP填充器gringo和SMT求解器z3。gringo部分填充输入程序，保留一些变量供z3处理。我们展示了该系统能够有效处理实数计算，用于连续变化的推理。 

---
# Automated Validation of Textual Constraints Against AutomationML via LLMs and SHACL 

**Title (ZH)**: 基于LLM和SHACL的自动化验证文本约束 Against AutomationML 

**Authors**: Tom Westermann, Aljosha Köcher, Felix Gehlhoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.10678)  

**Abstract**: AutomationML (AML) enables standardized data exchange in engineering, yet existing recommendations for proper AML modeling are typically formulated as informal and textual constraints. These constraints cannot be validated automatically within AML itself. This work-in-progress paper introduces a pipeline to formalize and verify such constraints. First, AML models are mapped to OWL ontologies via RML and SPARQL. In addition, a Large Language Model translates textual rules into SHACL constraints, which are then validated against the previously generated AML ontology. Finally, SHACL validation results are automatically interpreted in natural language. The approach is demonstrated on a sample AML recommendation. Results show that even complex modeling rules can be semi-automatically checked -- without requiring users to understand formal methods or ontology technologies. 

**Abstract (ZH)**: 自动化ML (AML) 使工程领域内的数据交换标准化，然而现有的AML建模建议通常以非正式和文本形式表述，无法在AML本身内自动验证。本初步论文介绍了一种流水线，旨在形式化和验证此类约束。首先，通过RML和SPARQL将AML模型映射到OWL本体。此外，大型语言模型将文本规则翻译成SHACL约束，然后这些约束将针对之前生成的AML本体进行验证。最后，自动解释SHACL验证结果为自然语言。该方法在示例AML建议上进行了演示，结果表明，即使复杂的建模规则也可以半自动检查，而无需用户理解形式方法或本体技术。 

---
# TeleMath: A Benchmark for Large Language Models in Telecom Mathematical Problem Solving 

**Title (ZH)**: TeleMath: 电信数学问题解决中大型语言模型的基准 

**Authors**: Vincenzo Colle, Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Fadhel Ayed, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.10674)  

**Abstract**: The increasing adoption of artificial intelligence in telecommunications has raised interest in the capability of Large Language Models (LLMs) to address domain-specific, mathematically intensive tasks. Although recent advancements have improved the performance of LLMs in general mathematical reasoning, their effectiveness within specialized domains, such as signal processing, network optimization, and performance analysis, remains largely unexplored. To address this gap, we introduce TeleMath, the first benchmark dataset specifically designed to evaluate LLM performance in solving mathematical problems with numerical solutions in the telecommunications domain. Comprising 500 question-answer (QnA) pairs, TeleMath covers a wide spectrum of topics in the telecommunications field. This paper outlines the proposed QnAs generation pipeline, starting from a selected seed of problems crafted by Subject Matter Experts. The evaluation of a wide range of open-source LLMs reveals that best performance on TeleMath is achieved by recent models explicitly designed for mathematical or logical reasoning. In contrast, general-purpose models, even those with a large number of parameters, often struggle with these challenges. We have released the dataset and the evaluation code to ease result reproducibility and support future research. 

**Abstract (ZH)**: 电信领域中人工智能 Adoption of Artificial Intelligence in Telecommunications 提升对大型语言模型解决数学密集型特定领域问题能力的兴趣：TeleMath 数据集的设计与评估 

---
# Data Driven Diagnosis for Large Cyber-Physical-Systems with Minimal Prior Information 

**Title (ZH)**: 基于最少先验信息的数据驱动诊断在大型网络物理系统中的应用 

**Authors**: Henrik Sebastian Steude, Alexander Diedrich, Ingo Pill, Lukas Moddemann, Daniel Vranješ, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2506.10613)  

**Abstract**: Diagnostic processes for complex cyber-physical systems often require extensive prior knowledge in the form of detailed system models or comprehensive training data. However, obtaining such information poses a significant challenge. To address this issue, we present a new diagnostic approach that operates with minimal prior knowledge, requiring only a basic understanding of subsystem relationships and data from nominal operations. Our method combines a neural network-based symptom generator, which employs subsystem-level anomaly detection, with a new graph diagnosis algorithm that leverages minimal causal relationship information between subsystems-information that is typically available in practice. Our experiments with fully controllable simulated datasets show that our method includes the true causal component in its diagnosis set for 82 p.c. of all cases while effectively reducing the search space in 73 p.c. of the scenarios. Additional tests on the real-world Secure Water Treatment dataset showcase the approach's potential for practical scenarios. Our results thus highlight our approach's potential for practical applications with large and complex cyber-physical systems where limited prior knowledge is available. 

**Abstract (ZH)**: 复杂 cyber-物理系统故障诊断过程往往需要大量的先验知识，形式上为详细的系统模型或完备的训练数据。然而，获取这些信息是一项重大挑战。为了应对这一问题，我们提出了一种新的诊断方法，该方法在极少先验知识的情况下运作，仅需对子系统关系有基本理解以及正常操作数据。该方法结合了一种基于神经网络的症状生成器，该生成器采用子系统级别的异常检测，以及一种新的图诊断算法，该算法利用子系统之间最小的因果关系信息——这种信息在实践中通常是可以获得的。我们的实验结果显示，在82%的情况下，我们的方法在诊断集中包含了真正的因果组件，并且在73%的场景中有效地减少了搜索空间。此外，对实际的Secure Water Treatment数据集的测试还展示了该方法在实际场景中的潜在应用。因此，我们的结果突显了在大量且复杂的 cyber-物理系统中，当先验知识有限时，该方法的潜在应用价值。 

---
# Primender Sequence: A Novel Mathematical Construct for Testing Symbolic Inference and AI Reasoning 

**Title (ZH)**: 先行序列：一种新型的符号推理和AI推理测试的数学建构 

**Authors**: Mohd Anwar Jamal Faiz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10585)  

**Abstract**: This paper introduces the Primender sequence, a novel integer sequence defined by a hybrid rule that combines classical primality with modular digit-based conditions. Specifically, a number n is included in the sequence if it is prime or ends with a prime number of unit digit or any length. In other words, numbers which are primes or have at least one prime suffix. The resulting sequence exhibits a deterministic yet non-trivial structure, blending number-theoretic properties with symbolic patterning. We propose the Primender sequence as a benchmark for evaluating the symbolic reasoning capabilities of Large Language Models (LLMs). The study is motivated by the need for interpretable, rule-based testbeds that can assess an LLM's ability to infer hidden rules, validate mathematical hypotheses, and generalize symbolic logic at scale. A key hypothesis explored is: Whenever a number in the Primender sequence is exactly one more than the largest prime less than or equal to it, the difference between it and the previous number in the sequence is also 1. We design a structured prompt and evaluation framework to test this hypothesis across multiple state-of-the-art LLMs, including ChatGPT, Copilot, DeepSeek, Gemini, Grok, and LLaMA. The models are tasked with identifying the underlying rule, validating the hypothesis, and generating the next 100,000 terms of the sequence. Comparative metrics such as rule inference accuracy, hypothesis evaluation, sequence validity, and symbolic explanation quality are used to assess model performance. This work contributes a novel mathematical construct and a reproducible methodology for benchmarking LLMs in symbolic reasoning, hypothesis testing, and scalable pattern generalization - bridging the domains of number theory, artificial intelligence, and software engineering. 

**Abstract (ZH)**: 这种论文介绍了一种新的整数序列——Primender序列，该序列由结合经典质数条件与模数码条件的混合规则定义。具体而言，如果一个数是质数或其个位数是质数或具有任意长度，则该数包含在该序列中，换句话说，这些数是质数或至少有一个质数后缀。该序列表现出一种确定性但非平凡的结构，融合了数论性质与符号模式。我们提出Primender序列作为评估大型语言模型（LLMs）符号推理能力的基准。这项研究旨在满足对具有解释性和基于规则的测试平台的需求，这些平台可以评估LLM推理潜在规则、验证数学假设和大规模泛化符号逻辑的能力。一个核心假设是：当Primender序列中的一个数恰好比其最接近的不大于该数的质数大1时，该数与序列中前一个数之间的差也为1。我们设计了一个结构化的提示和评估框架，用以跨多个最先进LLM（包括ChatGPT、Copilot、DeepSeek、Gemini、Grok和LLaMA）测试这一假设。模型的任务是识别潜在规则、验证假设并生成序列的前100,000项。通过规则推断准确性、假设评估、序列有效性及符号解释质量等比较指标评估模型性能。这项工作贡献了一种新的数学构造和一种可重复的方法，用于在符号推理、假设测试和可扩展模式泛化方面对LLM进行基准测试——联结了数论、人工智能和软件工程领域。 

---
# LogiPlan: A Structured Benchmark for Logical Planning and Relational Reasoning in LLMs 

**Title (ZH)**: LogiPlan：面向大模型的逻辑规划与关系推理结构化基准 

**Authors**: Yanan Cai, Ahmed Salem, Besmira Nushi, Mark Russinovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.10527)  

**Abstract**: We introduce LogiPlan, a novel benchmark designed to evaluate the capabilities of large language models (LLMs) in logical planning and reasoning over complex relational structures. Logical relational reasoning is important for applications that may rely on LLMs to generate and query structured graphs of relations such as network infrastructure, knowledge bases, or business process schema. Our framework allows for dynamic variation of task complexity by controlling the number of objects, relations, and the minimum depth of relational chains, providing a fine-grained assessment of model performance across difficulty levels. LogiPlan encompasses three complementary tasks: (1) Plan Generation, where models must construct valid directed relational graphs meeting specified structural constraints; (2) Consistency Detection, testing models' ability to identify inconsistencies in relational structures; and (3) Comparison Question, evaluating models' capacity to determine the validity of queried relationships within a given graph. Additionally, we assess models' self-correction capabilities by prompting them to verify and refine their initial solutions. We evaluate state-of-the-art models including DeepSeek R1, Gemini 2.0 Pro, Gemini 2 Flash Thinking, GPT-4.5, GPT-4o, Llama 3.1 405B, O3-mini, O1, and Claude 3.7 Sonnet across these tasks, revealing significant performance gaps that correlate with model scale and architecture. Our analysis demonstrates that while recent reasoning-enhanced models show promising results on simpler instances, they struggle with more complex configurations requiring deeper logical planning. 

**Abstract (ZH)**: LogiPlan：一种用于评估大型语言模型在逻辑规划和复杂关系结构推理能力的新基准 

---
# Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning 

**Title (ZH)**: 科学家的首次考试：探究大语言模型的感知、理解与推理认知能力 

**Authors**: Yuhao Zhou, Yiheng Wang, Xuming He, Ruoyao Xiao, Zhiwei Li, Qiantai Feng, Zijie Guo, Yuejin Yang, Hao Wu, Wenxuan Huang, Jiaqi Wei, Dan Si, Xiuqi Yao, Jia Bu, Haiwen Huang, Tianfan Fu, Shixiang Tang, Ben Fei, Dongzhan Zhou, Fenghua Ling, Yan Lu, Siqi Sun, Chenhui Li, Guanjie Zheng, Jiancheng Lv, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.10521)  

**Abstract**: Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries. 

**Abstract (ZH)**: 科学发现 increasingly 依赖于基于密集科学数据和特定领域专业知识的复杂多模态推理。得益于专家级科学基准的支持，科学多模态大规模语言模型（MLLMs）有可能在实际工作流程中显著增强这一发现过程。然而，当前的科学基准主要集中在评估MLLMs的知识理解能力上，导致对其感知和推理能力评估不足。为解决这一问题，我们提出了科学家首考（SFE）基准，旨在通过三个相互关联的层次来评估MLLMs的科学认知能力：科学信号感知、科学属性理解、科学比较推理。具体而言，SFE 包含了 830 个专家验证的VQA对，涵盖了五个高价值学科的 66 个多模态任务，涉及三种问题类型。广泛的实验表明，当前最先进的 GPT-o3 和 InternVL-3 在 SFE 上的得分仅为 34.08% 和 26.52%，突显出MLLMs在科学领域仍有很大的改进空间。我们希望从SFE中获得的见解能够促进AI增强科学发现方面的进一步发展。 

---
# OIBench: Benchmarking Strong Reasoning Models with Olympiad in Informatics 

**Title (ZH)**: OIBench： Olympiad in Informatics 评估强大推理模型的基准测试 

**Authors**: Yaoming Zhu, Junxin Wang, Yiyang Li, Lin Qiu, ZongYu Wang, Jun Xu, Xuezhi Cao, Yuhuai Wei, Mingshi Wang, Xunliang Cai, Rong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.10481)  

**Abstract**: As models become increasingly sophisticated, conventional algorithm benchmarks are increasingly saturated, underscoring the need for more challenging benchmarks to guide future improvements in algorithmic reasoning. This paper introduces OIBench, a high-quality, private, and challenging olympiad-level informatics dataset comprising 250 carefully curated original problems. We detail the construction methodology of the benchmark, ensuring a comprehensive assessment across various programming paradigms and complexities, and we demonstrate its contamination-resistant properties via experiments. We propose Time/Space Completion Curves for finer-grained efficiency analysis and enable direct human-model comparisons through high-level participant evaluations. Our experiments reveal that while open-source models lag behind closed-source counterparts, current SOTA models already outperform most human participants in both correctness and efficiency, while still being suboptimal compared to the canonical solutions. By releasing OIBench as a fully open-source resource (this https URL), we hope this benchmark will contribute to advancing code reasoning capabilities for future LLMs. 

**Abstract (ZH)**: 随着模型日益复杂，传统算法基准逐渐饱和，强调了建立更具挑战性的基准以指导未来算法推理改进的必要性。本文介绍了OIBench，这是一个高质量、私有且具有奥林匹克级别信息学挑战性的数据集，包含250个精心筛选的原创问题。我们详细介绍了基准的构建方法，确保其在各类编程范式和复杂性方面的全面评估，并通过实验展示了其抵抗污染的特性。我们提出了时间/空间完成曲线用于更精细的效率分析，并通过高层次的人机比较使直接的人类模型对比成为可能。实验结果显示，开源模型落后于闭源模型，但当前的SOTA模型已经在正确性和效率上超越了大多数人类参与者，尽管仍然不及经典解决方案。通过将OIBench作为一个完全开源的资源发布（this https URL），我们希望此基准能促进未来LLM的代码推理能力的发展。 

---
# Multi-dimensional Autoscaling of Processing Services: A Comparison of Agent-based Methods 

**Title (ZH)**: 基于代理的方法多维度处理服务自动扩展比较 

**Authors**: Boris Sedlak, Alireza Furutanpey, Zihang Wang, Víctor Casamayor Pujol, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10420)  

**Abstract**: Edge computing breaks with traditional autoscaling due to strict resource constraints, thus, motivating more flexible scaling behaviors using multiple elasticity dimensions. This work introduces an agent-based autoscaling framework that dynamically adjusts both hardware resources and internal service configurations to maximize requirements fulfillment in constrained environments. We compare four types of scaling agents: Active Inference, Deep Q Network, Analysis of Structural Knowledge, and Deep Active Inference, using two real-world processing services running in parallel: YOLOv8 for visual recognition and OpenCV for QR code detection. Results show all agents achieve acceptable SLO performance with varying convergence patterns. While the Deep Q Network benefits from pre-training, the structural analysis converges quickly, and the deep active inference agent combines theoretical foundations with practical scalability advantages. Our findings provide evidence for the viability of multi-dimensional agent-based autoscaling for edge environments and encourage future work in this research direction. 

**Abstract (ZH)**: 边缘计算打破传统自动扩展模式，由于严格的资源约束，因此促进了多维度扩展行为的灵活性。本文介绍了一种基于代理的自动扩展框架，该框架能够动态调整硬件资源和内部服务配置，以在受限环境中最大化需求满足。我们使用两套并行运行的实际处理服务——YOLOv8用于视觉识别和OpenCV用于QR码检测——对比了四种类型的扩展代理：主动推理、深度Q网络、结构知识分析和深度主动推理。结果表明，所有代理都能达到可接受的服务水平目标（SLO）性能，但存在不同的收敛模式。深度Q网络从预训练中受益，结构分析迅速收敛，而深度主动推理代理结合了理论基础与实用扩展优势。本研究结果为多维度代理自动扩展在边缘环境中的可行性提供了证据，并鼓励未来在此研究方向上的工作。 

---
# Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges 

**Title (ZH)**: 基于系统1或系统2的RAG推理：面向工业挑战的推理代理检索增强生成综述 

**Authors**: Jintao Liang, Gang Su, Huifeng Lin, You Wu, Rui Zhao, Ziyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10408)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to overcome the knowledge limitations of Large Language Models (LLMs) by integrating external retrieval with language generation. While early RAG systems based on static pipelines have shown effectiveness in well-structured tasks, they struggle in real-world scenarios requiring complex reasoning, dynamic retrieval, and multi-modal integration. To address these challenges, the field has shifted toward Reasoning Agentic RAG, a paradigm that embeds decision-making and adaptive tool use directly into the retrieval process. In this paper, we present a comprehensive review of Reasoning Agentic RAG methods, categorizing them into two primary systems: predefined reasoning, which follows fixed modular pipelines to boost reasoning, and agentic reasoning, where the model autonomously orchestrates tool interaction during inference. We analyze representative techniques under both paradigms, covering architectural design, reasoning strategies, and tool coordination. Finally, we discuss key research challenges and propose future directions to advance the flexibility, robustness, and applicability of reasoning agentic RAG systems. Our collection of the relevant research has been organized into a this https URL. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)作为一种强大的框架，通过集成外部检索与语言生成，克服了大型语言模型的知识限制。虽然基于静态管道的早期RAG系统在结构良好的任务中表现出有效性，但在需要复杂推理、动态检索和多模态集成的现实场景中表现不佳。为应对这些挑战，该领域转向了Reasoning Agentic RAG范式，该范式直接将决策制定和自适应工具使用嵌入到检索过程中。在本文中，我们对Reasoning Agentic RAG方法进行了全面综述，将其分为两类主要系统：预定义推理，遵循固定模块化管道以增强推理；自主推理，其中模型在推理过程中自主协调工具交互。我们分析了这两种范式下的代表性技术，涵盖了架构设计、推理策略和工具协调。最后，我们讨论了关键研究挑战，并提出了推进Reasoning Agentic RAG系统灵活性、稳健性和适用性的未来方向。相关内容整理在 this https URL  。 

---
# Mirage-1: Augmenting and Updating GUI Agent with Hierarchical Multimodal Skills 

**Title (ZH)**: Mirage-1：增强和更新具有分层多模态技能的GUI代理 

**Authors**: Yuquan Xie, Zaijing Li, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Dongmei Jiang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10387)  

**Abstract**: Recent efforts to leverage the Multi-modal Large Language Model (MLLM) as GUI agents have yielded promising outcomes. However, these agents still struggle with long-horizon tasks in online environments, primarily due to insufficient knowledge and the inherent gap between offline and online domains. In this paper, inspired by how humans generalize knowledge in open-ended environments, we propose a Hierarchical Multimodal Skills (HMS) module to tackle the issue of insufficient knowledge. It progressively abstracts trajectories into execution skills, core skills, and ultimately meta-skills, providing a hierarchical knowledge structure for long-horizon task planning. To bridge the domain gap, we propose the Skill-Augmented Monte Carlo Tree Search (SA-MCTS) algorithm, which efficiently leverages skills acquired in offline environments to reduce the action search space during online tree exploration. Building on HMS, we propose Mirage-1, a multimodal, cross-platform, plug-and-play GUI agent. To validate the performance of Mirage-1 in real-world long-horizon scenarios, we constructed a new benchmark, AndroidLH. Experimental results show that Mirage-1 outperforms previous agents by 32\%, 19\%, 15\%, and 79\% on AndroidWorld, MobileMiniWob++, Mind2Web-Live, and AndroidLH, respectively. Project page: this https URL 

**Abstract (ZH)**: Recent efforts to leverage the Multi-modal Large Language Model (MLLM) as GUI agents have yielded promising outcomes. However, these agents still struggle with long-horizon tasks in online environments, primarily due to insufficient knowledge and the inherent gap between offline and online domains. Inspired by human knowledge generalization in open-ended environments, this paper proposes a Hierarchical Multimodal Skills (HMS) module to address insufficient knowledge. It progressively abstracts trajectories into execution skills, core skills, and ultimately meta-skills, providing a hierarchical knowledge structure for long-horizon task planning. To bridge the domain gap, the Skill-Augmented Monte Carlo Tree Search (SA-MCTS) algorithm is proposed, which efficiently leverages offline skills to reduce the action search space during online tree exploration. Building on HMS, Mirage-1, a multimodal, cross-platform, plug-and-play GUI agent, is proposed. To validate Mirage-1 in real-world long-horizon scenarios, a new benchmark, AndroidLH, is constructed. Experimental results show that Mirage-1 outperforms previous agents by 32%, 19%, 15%, and 79% on AndroidWorld, MobileMiniWob++, Mind2Web-Live, and AndroidLH, respectively. 

---
# NeuroPAL: Punctuated Anytime Learning with Neuroevolution for Macromanagement in Starcraft: Brood War 

**Title (ZH)**: NeuroPAL：《星际争霸：虫群之心》宏管理的间歇式持续学习神经进化的算法 

**Authors**: Jim O'Connor, Yeonghun Lee, Gary B Parker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10384)  

**Abstract**: StarCraft: Brood War remains a challenging benchmark for artificial intelligence research, particularly in the domain of macromanagement, where long-term strategic planning is required. Traditional approaches to StarCraft AI rely on rule-based systems or supervised deep learning, both of which face limitations in adaptability and computational efficiency. In this work, we introduce NeuroPAL, a neuroevolutionary framework that integrates Neuroevolution of Augmenting Topologies (NEAT) with Punctuated Anytime Learning (PAL) to improve the efficiency of evolutionary training. By alternating between frequent, low-fidelity training and periodic, high-fidelity evaluations, PAL enhances the sample efficiency of NEAT, enabling agents to discover effective strategies in fewer training iterations. We evaluate NeuroPAL in a fixed-map, single-race scenario in StarCraft: Brood War and compare its performance to standard NEAT-based training. Our results show that PAL significantly accelerates the learning process, allowing the agent to reach competitive levels of play in approximately half the training time required by NEAT alone. Additionally, the evolved agents exhibit emergent behaviors such as proxy barracks placement and defensive building optimization, strategies commonly used by expert human players. These findings suggest that structured evaluation mechanisms like PAL can enhance the scalability and effectiveness of neuroevolution in complex real-time strategy environments. 

**Abstract (ZH)**: StarCraft: Brood War依然是一个挑战性的人工智能基准，特别是在需要长期战略性规划的大局管理领域。传统的StarCraft AI方法依赖于基于规则的系统或监督深度学习，这两种方法在适应性和计算效率上都存在局限性。本工作中，我们引入了NeuroPAL，这是一种将Neuroevolution of Augmenting Topologies (NEAT)与Punctuated Anytime Learning (PAL)相结合的神经演化框架，以提高进化训练的效率。通过交替进行频繁的低保真度训练和定期的高保真度评估，PAL提升了NEAT的样本效率，使智能体能够在较少的训练迭代中发现有效的策略。我们在StarCraft: Brood War的固定地图、单种族场景中评估了NeuroPAL，并将其性能与基于标准NEAT的训练进行了比较。实验结果显示，PAL显著加速了学习过程，使代理能够在大约一半的训练时间内达到具有竞争力的水平。此外，进化出的代理还表现出诸如代理兵营布局和防御建筑优化等 emergent 行为，这些策略通常由专家人类玩家使用。这些发现表明，类似PAL的结构化评估机制可以增强在复杂实时战略环境中神经演化方法的扩展性和有效性。 

---
# Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts 

**Title (ZH)**: Optimus-3: 向Towards通用 multimodal Minecraft 代理的过渡，配备可扩展的任务专家。 

**Authors**: Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Weili Guan, Dongmei Jiang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10357)  

**Abstract**: Recently, agents based on multimodal large language models (MLLMs) have achieved remarkable progress across various domains. However, building a generalist agent with capabilities such as perception, planning, action, grounding, and reflection in open-world environments like Minecraft remains challenges: insufficient domain-specific data, interference among heterogeneous tasks, and visual diversity in open-world settings. In this paper, we address these challenges through three key contributions. 1) We propose a knowledge-enhanced data generation pipeline to provide scalable and high-quality training data for agent development. 2) To mitigate interference among heterogeneous tasks, we introduce a Mixture-of-Experts (MoE) architecture with task-level routing. 3) We develop a Multimodal Reasoning-Augmented Reinforcement Learning approach to enhance the agent's reasoning ability for visual diversity in Minecraft. Built upon these innovations, we present Optimus-3, a general-purpose agent for Minecraft. Extensive experimental results demonstrate that Optimus-3 surpasses both generalist multimodal large language models and existing state-of-the-art agents across a wide range of tasks in the Minecraft environment. Project page: this https URL 

**Abstract (ZH)**: 近日，基于多模态大型语言模型的代理已在多个领域取得了显著进展。然而，在像Minecraft这样的开放世界环境中构建具备感知、规划、行动、地面化和反思等能力的一般性代理依然面临挑战：领域特定数据不足、异构任务间的干扰以及开放世界环境中的视觉多样性。在本文中，我们通过三项关键贡献应对这些挑战。1) 我们提出了一种知识增强的数据生成管道，为代理开发提供可扩展且高质量的训练数据。2) 为减少异构任务间的干扰，我们引入了任务级路由的Mixture-of-Experts (MoE) 架构。3) 我们开发了一种多模态推理增强的强化学习方法，以增强代理在Minecraft中的视觉多样性推理能力。基于这些创新，我们介绍了一种通用型代理Optimus-3。广泛的经验结果表明，Optimus-3在Minecraft环境中的多种任务上均超越了现有的泛化多模态大型语言模型和先进代理。项目页面：this https URL 

---
# A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon 

**Title (ZH)**: 跨多样团队策略的通用性基准：基于竞争宝可梦的benchmark 

**Authors**: Cameron Angliss, Jiaxun Cui, Jiaheng Hu, Arrasy Rahman, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2506.10326)  

**Abstract**: Developing AI agents that can robustly adapt to dramatically different strategic landscapes without retraining is a central challenge for multi-agent learning. Pokémon Video Game Championships (VGC) is a domain with an extraordinarily large space of possible team configurations of approximately $10^{139}$ - far larger than those of Dota or Starcraft. The highly discrete, combinatorial nature of team building in Pokémon VGC causes optimal strategies to shift dramatically depending on both the team being piloted and the opponent's team, making generalization uniquely challenging. To advance research on this problem, we introduce VGC-Bench: a benchmark that provides critical infrastructure, standardizes evaluation protocols, and supplies human-play datasets and a range of baselines - from large-language-model agents and behavior cloning to reinforcement learning and empirical game-theoretic methods such as self-play, fictitious play, and double oracle. In the restricted setting where an agent is trained and evaluated on a single-team configuration, our methods are able to win against a professional VGC competitor. We extensively evaluated all baseline methods over progressively larger team sets and find that even the best-performing algorithm in the single-team setting struggles at scaling up as team size grows. Thus, policy generalization across diverse team strategies remains an open challenge for the community. Our code is open sourced at this https URL. 

**Abstract (ZH)**: 开发能够在没有重新训练的情况下 robust 地适应大幅不同的战略景观的 AI 代理是多代理学习中的一个核心挑战。Pokémon 视频游戏锦标赛 (VGC) 是一个具有极为庞大可能队伍配置空间的领域，大约为 \(10^{139}\)，远超 Dota 或 Starcraft。在 Pokémon VGC 中，队伍构建的高度离散和组合性质使得最优策略随着被操控队伍和对手队伍的不同而大幅变化，这使得泛化变得尤为具有挑战性。为了推进这一问题的研究，我们引入了 VGC-Bench：一个基准，提供了关键基础设施、标准化评估协议，并提供了人类比赛的数据集以及从大规模语言模型代理和行为克隆到强化学习和经验博弈论方法（如自博弈、虚构玩和双oracle）的各种 baselines。在仅在单一队伍配置下进行训练和评估的受限设置中，我们的方法能够战胜职业 VGC 竞赛选手。我们在逐渐增大的队伍集合上广泛评估了所有 baselines 方法，并发现即使在单一队伍设置中表现最佳的算法在队伍规模扩大时也难以扩展。因此，跨多样队伍策略的策略泛化仍然是社区中的一个开放挑战。我们的代码已开源：this https URL。 

---
# The Alignment Trap: Complexity Barriers 

**Title (ZH)**: 对齐陷阱：复杂性障碍 

**Authors**: Jasper Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10304)  

**Abstract**: We establish fundamental computational complexity barriers to verifying AI safety as system capabilities scale. Our main results show that for AI systems with expressiveness EXP$(m)$ above a critical threshold $\tau$, safety verification requires exponential time and is coNP-complete. We formalize the Capability-Risk Scaling (CRS) dynamic, which demonstrates how increasing AI capability drives societal safety requirements toward perfection, creating an inescapable tension with verification complexity. Through four core theorems, we prove that (1) verification complexity grows exponentially with system expressiveness, (2) safe policies comprise at most a $2^{-2^m}$ fraction of the policy space, (3) no finite set of alignment techniques can provide universal coverage, and (4) robust safety properties form measure-zero sets for neural networks. These results characterize an "intractability gap" where practical safety requirements fall within the region of computational intractability. We conclude by presenting a strategic trilemma: AI development must either constrain system complexity to maintain verifiable safety, accept unverifiable risks while scaling capabilities, or develop fundamentally new safety paradigms beyond verification. Our work provides the first systematic complexity-theoretic analysis of AI alignment and establishes rigorous bounds that any safety approach must confront. A formal verification of the core theorems in Lean4 is currently in progress. 

**Abstract (ZH)**: 我们建立了随着AI系统能力增强而验证AI安全性基本计算复杂性障碍。我们的主要结果表明，对于表示能力EXP$(m)$超过临界阈值$\tau$的AI系统，安全性验证需要指数时间且是coNP完全问题。我们正式化了能力-风险扩展（CRS）动态，显示了增强的AI能力如何推动社会安全性要求向完美发展，创造了验证复杂性不可避免的紧张关系。通过四条核心定理，我们证明了：（1）验证复杂性随系统表示能力指数增长；（2）安全策略最多占策略空间的$2^{-2^m}$部分；（3）没有任何有限的对齐技术能够提供普遍覆盖；（4）对于神经网络，稳健的安全性质形成测度零集。这些结果刻画了“不可处理性差距”，其中实用的安全要求处于计算不可处理性的区域。最后，我们提出了一个战略三难困境：AI开发必须要么限制系统复杂性以维持可验证的安全性，要么在增强能力的同时接受不可验证的风险，要么开发超越验证的基本新的安全范式。我们的工作提供了对AI对齐的第一个系统性的复杂性理论分析，并建立了任何安全方法都必须面对的严格界线。目前，核心定理在Lean4中正在进行形式验证。 

---
# Closer to Language than Steam: AI as the Cognitive Engine of a New Productivity Revolution 

**Title (ZH)**: 比蒸汽更接近语言：AI作为新生产力革命的认知引擎 

**Authors**: Xinmin Fang, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10281)  

**Abstract**: Artificial Intelligence (AI) is reframed as a cognitive engine driving a novel productivity revolution distinct from the Industrial Revolution's physical thrust. This paper develops a theoretical framing of AI as a cognitive revolution akin to written language - a transformative augmentation of human intellect rather than another mechanized tool. We compare AI's emergence to historical leaps in information technology to show how it amplifies knowledge work. Examples from various domains demonstrate AI's impact as a driver of productivity in cognitive tasks. We adopt a multidisciplinary perspective combining computer science advances with economic insights and sociological perspectives on how AI reshapes work and society. Through conceptual frameworks, we visualize the shift from manual to cognitive productivity. Our central argument is that AI functions as an engine of cognition - comparable to how human language revolutionized knowledge - heralding a new productivity paradigm. We discuss how this revolution demands rethinking of skills, organizations, and policies. This paper, balancing academic rigor with clarity, concludes that AI's promise lies in complementing human cognitive abilities, marking a new chapter in productivity evolution. 

**Abstract (ZH)**: 人工智能（AI）被重新定义为一种认知引擎，推动了一场不同于工业革命物理推动的新型生产力革命。本文从类比书写语言的角度，构建了AI作为一种认知革命的理论框架，将其视为人类智力的一种 transformative 增强，而非另一种机械化工具。我们将AI的出现与信息技术的历史飞跃进行对比，以展示其如何放大知识工作。从各个领域的例子可以看出，AI作为认知任务生产力驱动的影响。本文采用多学科视角，结合计算机科学进展、经济洞察以及社会学观点，探讨AI如何重塑工作和社会。通过概念框架，本文展现了从体力劳动到认知劳动生产力转变的图景。我们的主要论点是，AI作为一种认知引擎发挥着作用——类似于人类语言革新人类知识——预示着新的生产力范式的到来。本文讨论了这场革命对技能、组织和政策的重构需求。本文在保持学术严谨性的同时，清晰地表明了AI的潜力在于补充人类的认知能力，标志着生产力进化的新篇章。 

---
# WGSR-Bench: Wargame-based Game-theoretic Strategic Reasoning Benchmark for Large Language Models 

**Title (ZH)**: WGSR-Bench：基于战争博弈的博弈论战略推理基准测试（面向大型语言模型） 

**Authors**: Qiyue Yin, Pei Xu, Qiaozhe Li, Shengda Liu, Shengqi Shen, Tong Wang, Yihong Han, Xiaonan Zhao, Likun Yang, Shiyue Cao, Shiyu Qiu, Yuxuan Liu, Shizhao Yu, Lei Cui, Chengxin Yan, Jie Sun, Xiangquan Tang, Kaiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10264)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to a qualitative leap in artificial intelligence' s performance on reasoning tasks, particularly demonstrating remarkable capabilities in mathematical, symbolic, and commonsense reasoning. However, as a critical component of advanced human cognition, strategic reasoning, i.e., the ability to assess multi-agent behaviors in dynamic environments, formulate action plans, and adapt strategies, has yet to be systematically evaluated or modeled. To address this gap, this paper introduces WGSR-Bench, the first strategy reasoning benchmark for LLMs using wargame as its evaluation environment. Wargame, a quintessential high-complexity strategic scenario, integrates environmental uncertainty, adversarial dynamics, and non-unique strategic choices, making it an effective testbed for assessing LLMs' capabilities in multi-agent decision-making, intent inference, and counterfactual reasoning. WGSR-Bench designs test samples around three core tasks, i.e., Environmental situation awareness, Opponent risk modeling and Policy generation, which serve as the core S-POE architecture, to systematically assess main abilities of strategic reasoning. Finally, an LLM-based wargame agent is designed to integrate these parts for a comprehensive strategy reasoning assessment. With WGSR-Bench, we hope to assess the strengths and limitations of state-of-the-art LLMs in game-theoretic strategic reasoning and to advance research in large model-driven strategic intelligence. 

**Abstract (ZH)**: Recent突破在大型语言模型中的进展在人工智能在推理任务上的表现上带来了质的飞跃，特别是在数学、符号和常识推理方面展现了显著的能力。然而，作为高级人类认知的一个关键组成部分，战略推理，即评估多代理行为、制定行动方案和适应策略的能力，尚未被系统地评估或建模。为了解决这一差距，本文介绍了WGSR-Bench，这是首个使用战争游戏作为评估环境的战略推理基准。战争游戏是一个 quintessential 高复杂度战略场景，整合了环境不确定性、对抗动态和非唯一的战略选择，使其成为评估大型语言模型在多代理决策、意图推断和反事实推理方面能力的有效试验平台。WGSR-Bench 围繞三个核心任务即环境情况意识、对手风险建模和政策生成设计测试样本，以系统评估战略推理的主要能力。最后，设计了一个基于大型语言模型的战争游戏代理，将这些部分整合在一起进行全面的战略推理评估。通过WGSR-Bench，我们希望能够评估最先进的大型语言模型在博弈论战略推理方面的强点与局限性，并推动基于大型模型的战略智能研究。 

---
# Towards Responsible AI: Advances in Safety, Fairness, and Accountability of Autonomous Systems 

**Title (ZH)**: 负责任的人工智能：自主系统在安全性、公平性和问责制方面的进展 

**Authors**: Filip Cano  

**Link**: [PDF](https://arxiv.org/pdf/2506.10192)  

**Abstract**: Ensuring responsible use of artificial intelligence (AI) has become imperative as autonomous systems increasingly influence critical societal domains. However, the concept of trustworthy AI remains broad and multi-faceted. This thesis advances knowledge in the safety, fairness, transparency, and accountability of AI systems. In safety, we extend classical deterministic shielding techniques to become resilient against delayed observations, enabling practical deployment in real-world conditions. We also implement both deterministic and probabilistic safety shields into simulated autonomous vehicles to prevent collisions with road users, validating the use of these techniques in realistic driving simulators. We introduce fairness shields, a novel post-processing approach to enforce group fairness in sequential decision-making settings over finite and periodic time horizons. By optimizing intervention costs while strictly ensuring fairness constraints, this method efficiently balances fairness with minimal interference. For transparency and accountability, we propose a formal framework for assessing intentional behaviour in probabilistic decision-making agents, introducing quantitative metrics of agency and intention quotient. We use these metrics to propose a retrospective analysis of intention, useful for determining responsibility when autonomous systems cause unintended harm. Finally, we unify these contributions through the ``reactive decision-making'' framework, providing a general formalization that consolidates previous approaches. Collectively, the advancements presented contribute practically to the realization of safer, fairer, and more accountable AI systems, laying the foundations for future research in trustworthy AI. 

**Abstract (ZH)**: 确保人工智能的负责任使用已成为必要，随着自主系统在关键社会领域中的影响不断增加。然而，可信赖人工智能的概念依然宽泛且多维度。本论文推进了在安全、公平、透明和问责方面对人工智能系统的知识。在安全性方面，我们将传统的确定性防护技术扩展为能够在延迟观察下保持韧性，从而在实际条件下实现实用部署。我们还将确定性和概率性安全防护应用到模拟的自主车辆中，以防止与道路使用者发生碰撞，验证了这些技术在现实驾驶模拟器中的使用。我们提出了公平防护，这是一种新颖的后处理方法，用于在有限和周期性的时间框架内的顺序决策环境中强制实施群体公平性。通过在严格确保公平约束的同时优化干预成本，该方法能够高效地在公平性和最小化干扰之间取得平衡。在透明度和问责方面，我们提出了一种正式框架来评估概率决策代理的意图行为，并引入了代理性和意图商的定量指标。利用这些指标，我们提出了关于意图的回顾性分析，这在确定自主系统导致意外损害时的责任问题上有用。最后，我们通过“反应性决策”框架统一这些贡献，提供了一个综合的正式化方法，将先前的方法整合在一起。总体而言，所提出的发展实用地促进了实现更安全、更公平和更负责任的人工智能系统，并为未来可信赖人工智能的研究奠定了基础。 

---
# Correlation vs causation in Alzheimer's disease: an interpretability-driven study 

**Title (ZH)**: 阿尔茨海默病中相关性与因果性的区别：一种基于可解释性的研究 

**Authors**: Hamzah Dabool, Raghad Mustafa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10179)  

**Abstract**: Understanding the distinction between causation and correlation is critical in Alzheimer's disease (AD) research, as it impacts diagnosis, treatment, and the identification of true disease drivers. This experiment investigates the relationships among clinical, cognitive, genetic, and biomarker features using a combination of correlation analysis, machine learning classification, and model interpretability techniques. Employing the XGBoost algorithm, we identified key features influencing AD classification, including cognitive scores and genetic risk factors. Correlation matrices revealed clusters of interrelated variables, while SHAP (SHapley Additive exPlanations) values provided detailed insights into feature contributions across disease stages. Our results highlight that strong correlations do not necessarily imply causation, emphasizing the need for careful interpretation of associative data. By integrating feature importance and interpretability with classical statistical analysis, this work lays groundwork for future causal inference studies aimed at uncovering true pathological mechanisms. Ultimately, distinguishing causal factors from correlated markers can lead to improved early diagnosis and targeted interventions for Alzheimer's disease. 

**Abstract (ZH)**: 理解因果关系与相关性的区别在阿尔茨海默病（AD）研究中至关重要，这影响着诊断、治疗以及真正疾病驱动因素的识别。本研究结合相关性分析、机器学习分类和模型可解释性技术，探讨临床、认知、遗传和生物标志物特征之间的关系。采用XGBoost算法，我们识别出影响AD分类的关键特征，包括认知评分和遗传风险因素。相关矩阵揭示了相互关联变量的集群，而SHAP值提供了对疾病各阶段特征贡献的详细见解。研究结果强调，强烈的相关性并不必然意味着因果关系，突出了对关联性数据谨慎解释的必要性。通过将特征重要性与解释性与经典统计分析相结合，本研究为未来旨在揭示真正病理机制的因果推理研究奠定了基础。最终，区分因果因素和相关标志物可以提高阿尔茨海默病的早期诊断并实现针对性的干预。 

---
# One Patient, Many Contexts: Scaling Medical AI Through Contextual Intelligence 

**Title (ZH)**: 一个患者，多种情境：通过情境智能扩大医疗AI的应用规模 

**Authors**: Michelle M. Li, Ben Y. Reis, Adam Rodman, Tianxi Cai, Noa Dagan, Ran D. Balicer, Joseph Loscalzo, Isaac S. Kohane, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2506.10157)  

**Abstract**: Medical foundation models, including language models trained on clinical notes, vision-language models on medical images, and multimodal models on electronic health records, can summarize clinical notes, answer medical questions, and assist in decision-making. Adapting these models to new populations, specialties, or settings typically requires fine-tuning, careful prompting, or retrieval from knowledge bases. This can be impractical, and limits their ability to interpret unfamiliar inputs and adjust to clinical situations not represented during training. As a result, models are prone to contextual errors, where predictions appear reasonable but fail to account for critical patient-specific or contextual information. These errors stem from a fundamental limitation that current models struggle with: dynamically adjusting their behavior across evolving contexts of medical care. In this Perspective, we outline a vision for context-switching in medical AI: models that dynamically adapt their reasoning without retraining to new specialties, populations, workflows, and clinical roles. We envision context-switching AI to diagnose, manage, and treat a wide range of diseases across specialties and regions, and expand access to medical care. 

**Abstract (ZH)**: 医疗情境切换的人工智能：模型无需重新训练即可动态适应新的专科、人群、工作流程和临床角色，以诊断、管理和治疗各种疾病并扩大医疗访问范围。 

---
# A Conjecture on a Fundamental Trade-Off between Certainty and Scope in Symbolic and Generative AI 

**Title (ZH)**: 关于符号性和生成性AI中确定性和范围基本权衡的一种推测 

**Authors**: Luciano Floridi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10130)  

**Abstract**: This article introduces a conjecture that formalises a fundamental trade-off between provable correctness and broad data-mapping capacity in Artificial Intelligence (AI) systems. When an AI system is engineered for deductively watertight guarantees (demonstrable certainty about the error-free nature of its outputs) -- as in classical symbolic AI -- its operational domain must be narrowly circumscribed and pre-structured. Conversely, a system that can input high-dimensional data to produce rich information outputs -- as in contemporary generative models -- necessarily relinquishes the possibility of zero-error performance, incurring an irreducible risk of errors or misclassification. By making this previously implicit trade-off explicit and open to rigorous verification, the conjecture significantly reframes both engineering ambitions and philosophical expectations for AI. After reviewing the historical motivations for this tension, the article states the conjecture in information-theoretic form and contextualises it within broader debates in epistemology, formal verification, and the philosophy of technology. It then offers an analysis of its implications and consequences, drawing on notions of underdetermination, prudent epistemic risk, and moral responsibility. The discussion clarifies how, if correct, the conjecture would help reshape evaluation standards, governance frameworks, and hybrid system design. The conclusion underscores the importance of eventually proving or refuting the inequality for the future of trustworthy AI. 

**Abstract (ZH)**: 本文介绍了一个公设，该公设正式化了人工智能（AI）系统中可验证正确性和广泛数据映射能力之间的基本权衡。当一个AI系统被设计为具有演绎严密的保证（对其输出无錯誤性质的可证明确定性）——如同传统的符号AI——其操作领域必须被严格限定和预结构化。相反，能够输入高维数据以生成丰富信息输出的系统——如同当代的生成模型——必然放弃了零错误性能的可能性，不可避免地承担了错误或误分类的风险。通过使这种先前隐含的权衡变得明确并且可以通过严格的验证进行审视，该公设显著重塑了工程目标和对AI的哲学期望。文章回顾了这种紧张关系的历史动机，以信息论形式表述该公设，并将其置于更广泛的认识论、形式验证和技术哲学辩论的背景下。然后，文章提供了对该结论的分析，借鉴了确定性不足、审慎的认知风险和道德责任的概念。讨论阐明了，如果该公设正确，它将如何有助于重塑评估标准、治理框架和混合系统设计。结论强调了最终证明或反驳不等式对于可信AI未来发展的重要性。 

---
# Rethinking Losses for Diffusion Bridge Samplers 

**Title (ZH)**: 重新思考扩散桥梁采样中的损失函数 

**Authors**: Sebastian Sanokowski, Lukas Gruber, Christoph Bartmann, Sepp Hochreiter, Sebastian Lehner  

**Link**: [PDF](https://arxiv.org/pdf/2506.10982)  

**Abstract**: Diffusion bridges are a promising class of deep-learning methods for sampling from unnormalized distributions. Recent works show that the Log Variance (LV) loss consistently outperforms the reverse Kullback-Leibler (rKL) loss when using the reparametrization trick to compute rKL-gradients. While the on-policy LV loss yields identical gradients to the rKL loss when combined with the log-derivative trick for diffusion samplers with non-learnable forward processes, this equivalence does not hold for diffusion bridges or when diffusion coefficients are learned. Based on this insight we argue that for diffusion bridges the LV loss does not represent an optimization objective that can be motivated like the rKL loss via the data processing inequality. Our analysis shows that employing the rKL loss with the log-derivative trick (rKL-LD) does not only avoid these conceptual problems but also consistently outperforms the LV loss. Experimental results with different types of diffusion bridges on challenging benchmarks show that samplers trained with the rKL-LD loss achieve better performance. From a practical perspective we find that rKL-LD requires significantly less hyperparameter optimization and yields more stable training behavior. 

**Abstract (ZH)**: 扩散桥梁是一类从未正规化分布中采样的有前途的深度学习方法。最近的研究表明，在使用重参数化技巧计算反Kullback-Leibler (rKL)梯度时，Log Variance (LV)损失一直优于rKL损失。当与非可学习前向过程结合使用日志导数技巧时，针对扩散采样的on-policy LV损失会提供与rKL损失相同的梯度，但这种等价性并不适用于扩散桥梁，或当扩散系数被学习时。基于这一洞察，我们认为对于扩散桥梁，LV损失并不是可以通过数据处理不等式进行动机说明的优化目标。我们的分析表明，通过日志导数技巧使用rKL损失 (rKL-LD) 不仅避免了这些概念性问题，而且在一致性上优于LV损失。不同类型扩散桥梁在具有挑战性的基准上的实验结果表明，使用rKL-LD损失训练的采样器表现出更好的性能。从实用的角度来看，我们发现使用rKL-LD损失需要显著较少的超参数优化，并且具有更稳定的训练行为。 

---
# Fine-Grained Perturbation Guidance via Attention Head Selection 

**Title (ZH)**: 基于注意力头选择的细粒度扰动引导 

**Authors**: Donghoon Ahn, Jiwon Kang, Sanghyun Lee, Minjae Kim, Jaewon Min, Wooseok Jang, Saungwu Lee, Sayak Paul, Susung Hong, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10978)  

**Abstract**: Recent guidance methods in diffusion models steer reverse sampling by perturbing the model to construct an implicit weak model and guide generation away from it. Among these approaches, attention perturbation has demonstrated strong empirical performance in unconditional scenarios where classifier-free guidance is not applicable. However, existing attention perturbation methods lack principled approaches for determining where perturbations should be applied, particularly in Diffusion Transformer (DiT) architectures where quality-relevant computations are distributed across layers. In this paper, we investigate the granularity of attention perturbations, ranging from the layer level down to individual attention heads, and discover that specific heads govern distinct visual concepts such as structure, style, and texture quality. Building on this insight, we propose "HeadHunter", a systematic framework for iteratively selecting attention heads that align with user-centric objectives, enabling fine-grained control over generation quality and visual attributes. In addition, we introduce SoftPAG, which linearly interpolates each selected head's attention map toward an identity matrix, providing a continuous knob to tune perturbation strength and suppress artifacts. Our approach not only mitigates the oversmoothing issues of existing layer-level perturbation but also enables targeted manipulation of specific visual styles through compositional head selection. We validate our method on modern large-scale DiT-based text-to-image models including Stable Diffusion 3 and FLUX.1, demonstrating superior performance in both general quality enhancement and style-specific guidance. Our work provides the first head-level analysis of attention perturbation in diffusion models, uncovering interpretable specialization within attention layers and enabling practical design of effective perturbation strategies. 

**Abstract (ZH)**: Recent Guidance Methods in Diffusion Models通过扰动模型构造隐式弱模型并引导生成远离它，实现了逆向采样的调控。在这些方法中，注意力扰动在无条件场景中表现出强大的实证性能，尤其是在分类器自由指导不适用的情况下。然而，现有的注意力扰动方法缺乏确定扰动应应用于何处的原理性方法，特别是在质量相关的计算分布在各层中的扩散变换器(DiT)架构中。在本文中，我们考察了注意力扰动的粒度，从层级细化到单个注意力头，并发现特定的头控制着结构、样式和纹理质量等不同的视觉概念。基于这一洞察，我们提出了一种名为“HeadHunter”的系统框架，用于迭代选择与用户中心目标对齐的注意力头，从而实现对生成质量和视觉属性的细粒度控制。此外，我们引入了SoftPAG，它通过对每个选定头部的注意力图线性插值至单位矩阵，提供了一个连续的旋钮以调节扰动强度并抑制伪迹。我们的方法不仅缓解了现有层级扰动的过度平滑问题，还通过组成性头选择实现了对特定视觉样式的靶向操纵。我们在包括Stable Diffusion 3和FLUX.1的现代大规模基于DiT的文本到图像模型上验证了我们的方法，展示了在通用质量增强和样式特定指导方面的优越性能。我们的工作是首次对扩散模型中的注意力扰动进行头部级分析，揭示了注意力层内的可解释专业化，并为有效的扰动策略的设计提供了实用的设计指南。 

---
# AutoMind: Adaptive Knowledgeable Agent for Automated Data Science 

**Title (ZH)**: AutoMind：自适应知识型自动化数据科学代理 

**Authors**: Yixin Ou, Yujie Luo, Jingsheng Zheng, Lanning Wei, Shuofei Qiao, Jintian Zhang, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10974)  

**Abstract**: Large Language Model (LLM) agents have shown great potential in addressing real-world data science problems. LLM-driven data science agents promise to automate the entire machine learning pipeline, yet their real-world effectiveness remains limited. Existing frameworks depend on rigid, pre-defined workflows and inflexible coding strategies; consequently, they excel only on relatively simple, classical problems and fail to capture the empirical expertise that human practitioners bring to complex, innovative tasks. In this work, we introduce AutoMind, an adaptive, knowledgeable LLM-agent framework that overcomes these deficiencies through three key advances: (1) a curated expert knowledge base that grounds the agent in domain expert knowledge, (2) an agentic knowledgeable tree search algorithm that strategically explores possible solutions, and (3) a self-adaptive coding strategy that dynamically tailors code generation to task complexity. Evaluations on two automated data science benchmarks demonstrate that AutoMind delivers superior performance versus state-of-the-art baselines. Additional analyses confirm favorable effectiveness, efficiency, and qualitative solution quality, highlighting AutoMind as an efficient and robust step toward fully automated data science. 

**Abstract (ZH)**: 大语言模型（LLM）代理在解决实际数据科学问题方面表现出巨大潜力。虽然LLM驱动的数据科学代理有望自动化整个机器学习管道，但其实用效果仍受到限制。现有框架依赖于僵化的预定义工作流和僵硬的编程策略，仅在相对简单的经典问题上表现出色，无法捕捉到人类实践者在复杂、创新任务中带来的经验知识。本文介绍了一种名为AutoMind的自适应、知识丰富的LLM代理框架，通过三个方面的主要进步克服了这些缺陷：（1）精心策划的专家知识库，使代理扎根于领域专家知识；（2）一种智能的知识型树搜索算法，战略性地探索可能的解决方案；（3）一种自我适应的编程策略，动态调整代码生成以适应任务复杂性。对两种自动数据科学基准的评估显示，AutoMind在性能上优于现有先进基线。进一步的分析证实了其有利的有效性、效率和定性解决方案质量，突显了AutoMind是迈向完全自动化数据科学的高效且稳健的一步。 

---
# Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning 

**Title (ZH)**: 原理性的方法将神经架构拓展至函数空间进行算子学习 

**Authors**: Julius Berner, Miguel Liu-Schiaffini, Jean Kossaifi, Valentin Duruisseaux, Boris Bonev, Kamyar Azizzadenesheli, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10973)  

**Abstract**: A wide range of scientific problems, such as those described by continuous-time dynamical systems and partial differential equations (PDEs), are naturally formulated on function spaces. While function spaces are typically infinite-dimensional, deep learning has predominantly advanced through applications in computer vision and natural language processing that focus on mappings between finite-dimensional spaces. Such fundamental disparities in the nature of the data have limited neural networks from achieving a comparable level of success in scientific applications as seen in other fields. Neural operators are a principled way to generalize neural networks to mappings between function spaces, offering a pathway to replicate deep learning's transformative impact on scientific problems. For instance, neural operators can learn solution operators for entire classes of PDEs, e.g., physical systems with different boundary conditions, coefficient functions, and geometries. A key factor in deep learning's success has been the careful engineering of neural architectures through extensive empirical testing. Translating these neural architectures into neural operators allows operator learning to enjoy these same empirical optimizations. However, prior neural operator architectures have often been introduced as standalone models, not directly derived as extensions of existing neural network architectures. In this paper, we identify and distill the key principles for constructing practical implementations of mappings between infinite-dimensional function spaces. Using these principles, we propose a recipe for converting several popular neural architectures into neural operators with minimal modifications. This paper aims to guide practitioners through this process and details the steps to make neural operators work in practice. Our code can be found at this https URL 

**Abstract (ZH)**: 基于函数空间映射的神经运算：从原理到实践 

---
# Farseer: A Refined Scaling Law in Large Language Models 

**Title (ZH)**: Farseer: 一种精炼的大型语言模型扩展定律 

**Authors**: Houyi Li, Wenzhen Zheng, Qiufeng Wang, Zhenyu Ding, Haoying Wang, Zili Wang, Shijie Xuyang, Ning Ding, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10972)  

**Abstract**: Training Large Language Models (LLMs) is prohibitively expensive, creating a critical scaling gap where insights from small-scale experiments often fail to transfer to resource-intensive production systems, thereby hindering efficient innovation. To bridge this, we introduce Farseer, a novel and refined scaling law offering enhanced predictive accuracy across scales. By systematically constructing a model loss surface $L(N,D)$, Farseer achieves a significantly better fit to empirical data than prior laws (e.g., Chinchilla's law). Our methodology yields accurate, robust, and highly generalizable predictions, demonstrating excellent extrapolation capabilities, improving upon Chinchilla's law by reducing extrapolation error by 433\%. This allows for the reliable evaluation of competing training strategies across all $(N,D)$ settings, enabling conclusions from small-scale ablation studies to be confidently extrapolated to predict large-scale performance. Furthermore, Farseer provides new insights into optimal compute allocation, better reflecting the nuanced demands of modern LLM training. To validate our approach, we trained an extensive suite of approximately 1,000 LLMs across diverse scales and configurations, consuming roughly 3 million NVIDIA H100 GPU hours. We are comprehensively open-sourcing all models, data, results, and logs at this https URL to foster further research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的训练极其昂贵，造成了从小规模实验到资源密集型生产系统的可扩展性缺口，阻碍了高效的创新。为了解决这一问题，我们引入了Farseer，一种新颖且精确的可扩展性法则，能够在不同规模下提供更高的预测准确性。通过系统地构建模型损失表面$L(N,D)$，Farseer相较于之前的方法（如Chinchilla的法则）提供了显著更好的拟合效果，我们的方法能提供准确、稳健且高度泛化的预测，展示出卓越的外推能力，相比Chinchilla的法则，将外推误差降低了433%。这使得我们能够在所有$(N,D)$设置下可靠地评估竞争性的训练策略，使从小规模消融研究得出的结论能够自信地外推以预测大规模性能。此外，Farseer还为优化计算分配提供了新的见解，更好地反映了现代LLM训练的细微需求。为了验证我们的方法，我们在多种规模和配置下训练了大约1000个LLM，耗用了约300万NVIDIA H100 GPU小时。我们全面开源了所有模型、数据、结果和日志，以促进进一步研究。 

---
# Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs 

**Title (ZH)**: 超越注意力或相似性：最大化条件多样性以优化MLLMs中的 token 裁剪 

**Authors**: Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10967)  

**Abstract**: In multimodal large language models (MLLMs), the length of input visual tokens is often significantly greater than that of their textual counterparts, leading to a high inference cost. Many works aim to address this issue by removing redundant visual tokens. However, current approaches either rely on attention-based pruning, which retains numerous duplicate tokens, or use similarity-based pruning, overlooking the instruction relevance, consequently causing suboptimal performance. In this paper, we go beyond attention or similarity by proposing a novel visual token pruning method named CDPruner, which maximizes the conditional diversity of retained tokens. We first define the conditional similarity between visual tokens conditioned on the instruction, and then reformulate the token pruning problem with determinantal point process (DPP) to maximize the conditional diversity of the selected subset. The proposed CDPruner is training-free and model-agnostic, allowing easy application to various MLLMs. Extensive experiments across diverse MLLMs show that CDPruner establishes new state-of-the-art on various vision-language benchmarks. By maximizing conditional diversity through DPP, the selected subset better represents the input images while closely adhering to user instructions, thereby preserving strong performance even with high reduction ratios. When applied to LLaVA, CDPruner reduces FLOPs by 95\% and CUDA latency by 78\%, while maintaining 94\% of the original accuracy. Our code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型中基于条件多样性剪枝的视觉token裁剪方法 

---
# SpectralAR: Spectral Autoregressive Visual Generation 

**Title (ZH)**: SpectralAR：谱自回归视觉生成 

**Authors**: Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Yueqi Duan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10962)  

**Abstract**: Autoregressive visual generation has garnered increasing attention due to its scalability and compatibility with other modalities compared with diffusion models. Most existing methods construct visual sequences as spatial patches for autoregressive generation. However, image patches are inherently parallel, contradicting the causal nature of autoregressive modeling. To address this, we propose a Spectral AutoRegressive (SpectralAR) visual generation framework, which realizes causality for visual sequences from the spectral perspective. Specifically, we first transform an image into ordered spectral tokens with Nested Spectral Tokenization, representing lower to higher frequency components. We then perform autoregressive generation in a coarse-to-fine manner with the sequences of spectral tokens. By considering different levels of detail in images, our SpectralAR achieves both sequence causality and token efficiency without bells and whistles. We conduct extensive experiments on ImageNet-1K for image reconstruction and autoregressive generation, and SpectralAR achieves 3.02 gFID with only 64 tokens and 310M parameters. Project page: this https URL. 

**Abstract (ZH)**: 自回归视觉生成由于其可扩展性和与其他模态的兼容性，相比扩散模型越来越受到关注。大多数现有方法将视觉序列构建为空间 patches 进行自回归生成。然而，图像 patches 内在地是并行的，这与自回归建模的因果性质相矛盾。为解决这一问题，我们提出了一种从频谱视角实现视觉序列因果性的 Spectral AutoRegressive (SpectralAR) 视觉生成框架。具体而言，我们首先使用嵌套频谱分词将图像转换为有序的频谱 token，从低频到高频表示图像的不同频率分量。然后，我们以粗到细的方式对频谱 token 序列进行自回归生成。通过考虑图像的不同细节层级，我们的 SpectralAR 在实现序列因果性和 token 效率的同时，无需额外复杂性。我们在 ImageNet-1K 上进行了广泛的图像重建和自回归生成实验，SpectralAR 仅使用 64 个 token 和 310M 参数实现了 3.02 gFID。项目页面：this https URL。 

---
# ChineseHarm-Bench: A Chinese Harmful Content Detection Benchmark 

**Title (ZH)**: ChineseHarm-Bench: 中文有害内容检测基准 

**Authors**: Kangwei Liu, Siyuan Cheng, Bozhong Tian, Xiaozhuan Liang, Yuyang Yin, Meng Han, Ningyu Zhang, Bryan Hooi, Xi Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10960)  

**Abstract**: Large language models (LLMs) have been increasingly applied to automated harmful content detection tasks, assisting moderators in identifying policy violations and improving the overall efficiency and accuracy of content review. However, existing resources for harmful content detection are predominantly focused on English, with Chinese datasets remaining scarce and often limited in scope. We present a comprehensive, professionally annotated benchmark for Chinese content harm detection, which covers six representative categories and is constructed entirely from real-world data. Our annotation process further yields a knowledge rule base that provides explicit expert knowledge to assist LLMs in Chinese harmful content detection. In addition, we propose a knowledge-augmented baseline that integrates both human-annotated knowledge rules and implicit knowledge from large language models, enabling smaller models to achieve performance comparable to state-of-the-art LLMs. Code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动有害内容检测任务中的应用日益增多，助力モデरレーター政策违规行为的识别，并提高内容审核的整体效率和准确性。然而，现有的有害内容检测资源主要集中在英语上，中文数据集相对稀少且范围有限。我们提出了一种全面的专业注释基准数据集，涵盖六个代表性类别，并完全基于真实数据构建。我们的注释过程还产出了一种知识规则库，为大型语言模型提供显式的专家知识以辅助中文有害内容检测。此外，我们提出了一种知识增强的基础模型，结合了人工注释的知识规则和大型语言模型中的潜在知识，使较小的模型能够达到最先进的大型语言模型的性能。代码和数据可在此链接访问。 

---
# Understanding In-Context Learning on Structured Manifolds: Bridging Attention to Kernel Methods 

**Title (ZH)**: 结构流形上的上下文学习理解：注意力与核方法的桥梁 

**Authors**: Zhaiming Shen, Alexander Hsu, Rongjie Lai, Wenjing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10959)  

**Abstract**: While in-context learning (ICL) has achieved remarkable success in natural language and vision domains, its theoretical understanding--particularly in the context of structured geometric data--remains unexplored. In this work, we initiate a theoretical study of ICL for regression of Hölder functions on manifolds. By establishing a novel connection between the attention mechanism and classical kernel methods, we derive generalization error bounds in terms of the prompt length and the number of training tasks. When a sufficient number of training tasks are observed, transformers give rise to the minimax regression rate of Hölder functions on manifolds, which scales exponentially with the intrinsic dimension of the manifold, rather than the ambient space dimension. Our result also characterizes how the generalization error scales with the number of training tasks, shedding light on the complexity of transformers as in-context algorithm learners. Our findings provide foundational insights into the role of geometry in ICL and novels tools to study ICL of nonlinear models. 

**Abstract (ZH)**: 关于流形上Hölder函数回归的上下文学习的理论研究 

---
# ReGuidance: A Simple Diffusion Wrapper for Boosting Sample Quality on Hard Inverse Problems 

**Title (ZH)**: ReGuidance: 一种简单的扩散包裹方法以在棘手的逆问题中提升样本质量 

**Authors**: Aayush Karan, Kulin Shah, Sitan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10955)  

**Abstract**: There has been a flurry of activity around using pretrained diffusion models as informed data priors for solving inverse problems, and more generally around steering these models using reward models. Training-free methods like diffusion posterior sampling (DPS) and its many variants have offered flexible heuristic algorithms for these tasks, but when the reward is not informative enough, e.g., in hard inverse problems with low signal-to-noise ratio, these techniques veer off the data manifold, failing to produce realistic outputs. In this work, we devise a simple wrapper, ReGuidance, for boosting both the sample realism and reward achieved by these methods. Given a candidate solution $\hat{x}$ produced by an algorithm of the user's choice, we propose inverting the solution by running the unconditional probability flow ODE in reverse starting from $\hat{x}$, and then using the resulting latent as an initialization for DPS. We evaluate our wrapper on hard inverse problems like large box in-painting and super-resolution with high upscaling. Whereas state-of-the-art baselines visibly fail, we find that applying our wrapper on top of these baselines significantly boosts sample quality and measurement consistency. We complement these findings with theory proving that on certain multimodal data distributions, ReGuidance simultaneously boosts the reward and brings the candidate solution closer to the data manifold. To our knowledge, this constitutes the first rigorous algorithmic guarantee for DPS. 

**Abstract (ZH)**: 基于奖励模型引导的预训练扩散模型在逆问题求解中的增强方法 

---
# SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks 

**Title (ZH)**: SWE-Factory: 您的自动化故障排除训练数据和评估基准工厂 

**Authors**: Lianghong Guo, Yanlin Wang, Caihua Li, Pengyu Yang, Jiachi Chen, Wei Tao, Yingtian Zou, Duyu Tang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10954)  

**Abstract**: Constructing large-scale datasets for the GitHub issue resolution task is crucial for both training and evaluating the software engineering capabilities of Large Language Models (LLMs). However, the traditional process for creating such benchmarks is notoriously challenging and labor-intensive, particularly in the stages of setting up evaluation environments, grading test outcomes, and validating task instances. In this paper, we propose SWE-Factory, an automated pipeline designed to address these challenges. To tackle these issues, our pipeline integrates three core automated components. First, we introduce SWE-Builder, a multi-agent system that automates evaluation environment construction, which employs four specialized agents that work in a collaborative, iterative loop and leverages an environment memory pool to enhance efficiency. Second, we introduce a standardized, exit-code-based grading method that eliminates the need for manually writing custom parsers. Finally, we automate the fail2pass validation process using these reliable exit code signals. Experiments on 671 issues across four programming languages show that our pipeline can effectively construct valid task instances; for example, with GPT-4.1-mini, our SWE-Builder constructs 269 valid instances at $0.045 per instance, while with Gemini-2.5-flash, it achieves comparable performance at the lowest cost of $0.024 per instance. We also demonstrate that our exit-code-based grading achieves 100% accuracy compared to manual inspection, and our automated fail2pass validation reaches a precision of 0.92 and a recall of 1.00. We hope our automated pipeline will accelerate the collection of large-scale, high-quality GitHub issue resolution datasets for both training and evaluation. Our code and datasets are released at this https URL. 

**Abstract (ZH)**: 构建大型数据集以解决GitHub问题对于训练和评估大型语言模型的软件工程能力至关重要。然而，传统基准创建过程既复杂又劳动密集，特别是在设置评估环境、评分测试结果和验证任务实例阶段。本文提出SWE-Factory，一种自动流水线，旨在解决这些问题。为了应对这些挑战，我们的流水线整合了三个核心自动组件。首先，我们引入SWE-Builder，一个多功能系统，自动构建评估环境，采用四个专门的代理在协作迭代循环中工作，并利用环境记忆池提高效率。其次，我们引入标准化的基于退出代码的评分方法，消除手动编写自定义解析器的需要。最后，我们使用可靠的退出代码信号自动实现fail2pass验证流程。在涵盖四种编程语言的671个问题的实验中，我们的流水线能够有效构建有效的任务实例；例如，使用GPT-4.1-mini时，SWE-Builder每实例成本0.045美元构建269个有效实例，使用Gemini-2.5-flash时，成本最低，每实例0.024美元。我们还证明基于退出代码的评分与人工检查相匹配，精确率为100%，自动化的fail2pass验证精度为0.92，召回率为1.00。我们希望我们的自动流水线能够加速收集用于训练和评估的大规模高质量GitHub问题解决数据集。我们的代码和数据集在此处公开。 

---
# Domain2Vec: Vectorizing Datasets to Find the Optimal Data Mixture without Training 

**Title (ZH)**: 域2Vec：将数据集向量化以找到最优数据混合 without 训练 

**Authors**: Mozhi Zhang, Howe Tissue, Lu Wang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10952)  

**Abstract**: We introduce~\textsc{Domain2Vec}, a novel approach that decomposes any dataset into a linear combination of several \emph{meta-domains}, a new concept designed to capture the key underlying features of datasets. \textsc{Domain2Vec} maintains a vocabulary of meta-domains and uses a classifier to decompose any given dataset into a domain vector that corresponds to a distribution over this vocabulary. These domain vectors enable the identification of the optimal data mixture for language model (LM) pretraining in a training-free manner under the \emph{\textbf{D}istribution \textbf{A}lignment \textbf{A}ssumption} (DA$^{2}$), which suggests that when the data distributions of the training set and the validation set are better aligned, a lower validation loss is achieved. Moreover, \textsc{Domain2vec} can be seamlessly integrated into previous works to model the relationship between domain vectors and LM performance, greatly enhancing the efficiency and scalability of previous methods. Extensive experiments demonstrate that \textsc{Domain2Vec} helps find the data mixture that enhances downstream task performance with minimal computational overhead. Specifically, \textsc{Domain2Vec} achieves the same validation loss on Pile-CC using only $51.5\%$ of the computation required when training on the original mixture of The Pile dataset. Under equivalent compute budget, \textsc{Domain2Vec} improves downstream performance by an average of $2.83\%$. 

**Abstract (ZH)**: 我们介绍了Domain2Vec，这是一种新颖的方法，将任意数据集分解为多个元领域（meta-domains）的线性组合，元领域是一种新设计的概念，用于捕捉数据集的关键底层特征。Domain2Vec维护一个元领域的词汇表，并使用分类器将任意给定的数据集分解为对应于该词汇表分布的领域向量。这些领域向量使得在分布对齐假设（DA²）下以无训练方式找到用于语言模型预训练的最优数据混合变得更加可能，分布对齐假设认为，当训练集和验证集的数据分布更加对齐时，会实现更低的验证损失。此外，Domain2Vec可以无缝集成到先前的工作中，用于建模领域向量与语言模型性能之间的关系，极大地提高了先前方法的效率与扩展性。大量实验表明，Domain2Vec在最小计算开销的情况下帮助找到增强下游任务性能的数据混合。具体而言，与在Pile原始混合数据集上进行训练相比，Domain2Vec仅使用51.5%的计算量就能在Pile-CC上达到相同的验证损失。在等效计算预算下，Domain2Vec将下游性能平均提高2.83%。 

---
# Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors 

**Title (ZH)**: 使用轻量级序列监控检测LLMs中的分解攻击 

**Authors**: Chen Yueh-Han, Nitish Joshi, Yulin Chen, Maksym Andriushchenko, Rico Angell, He He  

**Link**: [PDF](https://arxiv.org/pdf/2506.10949)  

**Abstract**: Current LLM safety defenses fail under decomposition attacks, where a malicious goal is decomposed into benign subtasks that circumvent refusals. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent, leaving them blind to malicious intent that emerges over a sequence of seemingly benign instructions. We therefore propose adding an external monitor that observes the conversation at a higher granularity. To facilitate our study of monitoring decomposition attacks, we curate the largest and most diverse dataset to date, including question-answering, text-to-image, and agentic tasks. We verify our datasets by testing them on frontier LLMs and show an 87% attack success rate on average on GPT-4o. This confirms that decomposition attack is broadly effective. Additionally, we find that random tasks can be injected into the decomposed subtasks to further obfuscate malicious intents. To defend in real time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each subtask. We show that a carefully prompt engineered lightweight monitor achieves a 93% defense success rate, beating reasoning models like o3 mini as a monitor. Moreover, it remains robust against random task injection and cuts cost by 90% and latency by 50%. Our findings suggest that lightweight sequential monitors are highly effective in mitigating decomposition attacks and are viable in deployment. 

**Abstract (ZH)**: 当前的大型语言模型安全防护措施在分解攻击下失效，其中恶意目标被分解成规避拒绝的良性子任务。现有浅层次的安全对齐技术仅在即时提示中检测危害，而无法推理长期意图，因此使其对通过一系列貌似良性指令逐渐浮现的恶意意图视而不见。因此，我们提出增加一个外部监视器以从更高粒度观察对话。为了促进对分解攻击监控的研究，我们编纂了迄今为止规模最大、最多样化的数据集，包括问答、文本转图像和代理任务。我们通过在前沿大模型上测试数据集，并在平均87%的情况下攻破GPT-4o，验证了数据集的有效性，这证明了分解攻击的广泛有效性。此外，我们发现可以在分解的子任务中注入随机任务以进一步遮蔽恶意意图。为了实时防御，我们提出了一种轻量级顺序监控框架，逐项评估每个子任务。结果显示，一个精心设计的轻量级监控器在防御成功率上达到93%，优于如o3 mini等推理模型作为监控器。此外，它在面对随机任务注入时仍然保持鲁棒性，并将成本降低90%，延迟降低50%。我们的研究结果表明，轻量级顺序监控器在缓解分解攻击方面极为有效，并且可以在部署中实现。 

---
# GUARD: Guided Unlearning and Retention via Data Attribution for Large Language Models 

**Title (ZH)**: GUARD: 基于数据归属的引导遗忘与保留方法用于大型语言模型 

**Authors**: Evelyn Ma, Duo Zhou, Peizhi Niu, Huiting Zhou, Huan Zhang, Olgica Milenkovic, S. Rasoul Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2506.10946)  

**Abstract**: Unlearning in large language models (LLMs) is becoming increasingly important due to regulatory compliance, copyright protection, and privacy concerns. However, a key challenge in LLM unlearning is unintended forgetting, where the removal of specific data inadvertently impairs the utility of the model and its retention of valuable, desired information. While prior work has primarily focused on architectural innovations, the influence of data-level factors on unlearning performance remains underexplored. As a result, existing methods often suffer from degraded retention when forgetting high-impact data. To address this, we propose GUARD-a novel framework for Guided Unlearning And Retention via Data attribution. At its core, GUARD introduces a lightweight proxy data attribution metric tailored for LLM unlearning, which quantifies the "alignment" between the forget and retain sets while remaining computationally efficient. Building on this, we design a novel unlearning objective that assigns adaptive, nonuniform unlearning weights to samples, inversely proportional to their proxy attribution scores. Through such a reallocation of unlearning power, GUARD mitigates unintended losses in retention. We provide rigorous theoretical guarantees that GUARD significantly enhances retention while maintaining forgetting metrics comparable to prior methods. Extensive experiments on the TOFU benchmark across multiple LLM architectures demonstrate that GUARD substantially improves utility preservation while ensuring effective unlearning. Notably, GUARD reduces utility sacrifice on the Retain Set by up to 194.92% in terms of Truth Ratio when forgetting 10% of the training data. 

**Abstract (ZH)**: 大语言模型中的去学习：GUARD——基于数据归因的去学习与保留框架 

---
# VINCIE: Unlocking In-context Image Editing from Video 

**Title (ZH)**: VINCIE: 在上下文中解锁视频中的图像编辑 

**Authors**: Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10941)  

**Abstract**: In-context image editing aims to modify images based on a contextual sequence comprising text and previously generated images. Existing methods typically depend on task-specific pipelines and expert models (e.g., segmentation and inpainting) to curate training data. In this work, we explore whether an in-context image editing model can be learned directly from videos. We introduce a scalable approach to annotate videos as interleaved multimodal sequences. To effectively learn from this data, we design a block-causal diffusion transformer trained on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. Additionally, we propose a novel multi-turn image editing benchmark to advance research in this area. Extensive experiments demonstrate that our model exhibits strong in-context image editing capabilities and achieves state-of-the-art results on two multi-turn image editing benchmarks. Despite being trained exclusively on videos, our model also shows promising abilities in multi-concept composition, story generation, and chain-of-editing applications. 

**Abstract (ZH)**: 基于上下文的图像编辑旨在根据包含文本和先前生成图像的上下文字序列修改图像。现有方法通常依赖于特定任务的管道和专家模型（例如，分割和 inpainting）来整理训练数据。在本文中，我们探索是否可以直接从视频中学习基于上下文的图像编辑模型。我们介绍了一种可扩展的方法来标注视频为交错的多模态序列。为了有效地从这些数据中学习，我们设计了一个在三个代理任务（下一帧预测、当前分割预测和下一分割预测）上训练的块因果扩散变换器。此外，我们提出了一种新的多轮图像编辑基准来推进该领域的研究。广泛的实验表明，我们的模型表现出强大的基于上下文的图像编辑能力，并在两个多轮图像编辑基准上实现了最先进的结果。尽管仅在视频上进行训练，我们的模型在多概念组合、故事情节生成和连续编辑应用方面也表现出令人 Promise 的能力。 

---
# The Role of Generative AI in Facilitating Social Interactions: A Scoping Review 

**Title (ZH)**: 生成式人工智能在促进社会互动中的作用：一项范围性综述 

**Authors**: T. T. J. E. Arets, G. Perugia, M. Houben, W.A. IJsselsteijn  

**Link**: [PDF](https://arxiv.org/pdf/2506.10927)  

**Abstract**: Reduced social connectedness increasingly poses a threat to mental health, life expectancy, and general well-being. Generative AI (GAI) technologies, such as large language models (LLMs) and image generation tools, are increasingly integrated into applications aimed at enhancing human social experiences. Despite their growing presence, little is known about how these technologies influence social interactions. This scoping review investigates how GAI-based applications are currently designed to facilitate social interaction, what forms of social engagement they target, and which design and evaluation methodologies designers use to create and evaluate them. Through an analysis of 30 studies published since 2020, we identify key trends in application domains including storytelling, socio-emotional skills training, reminiscence, collaborative learning, music making, and general conversation. We highlight the role of participatory and co-design approaches in fostering both effective technology use and social engagement, while also examining socio-ethical concerns such as cultural bias and accessibility. This review underscores the potential of GAI to support dynamic and personalized interactions, but calls for greater attention to equitable design practices and inclusive evaluation strategies. 

**Abstract (ZH)**: reduced 社交联系的下降日益对心理健康、寿命和总体福祉构成威胁。生成型人工智能（GAI）技术，如大型语言模型（LLMs）和图像生成工具，正越来越多地被集成到旨在增强人类社交体验的应用中。尽管这些技术的影响力日益增强，但人们对它们如何影响社交互动知之甚少。本综述探讨了基于GAI的应用如何设计以促进社交互动、它们针对哪些形式的社交参与以及设计者采用哪些设计和评价方法来创建和评估这些应用。通过分析2020年以来发表的30项研究，我们识别了应用领域中的关键趋势，包括叙事、情感技能训练、回忆、协作学习、音乐创作和一般对话。 chúng我们强调参与式和共同设计方法在促进有效技术使用和社交参与中的作用，同时也探讨了文化偏见和可访问性等社会伦理问题。本综述强调了GAI在支持动态和个性化互动方面的潜力，但也呼吁对公平设计实践和包容性评估策略给予更多关注。 

---
# Agentic Semantic Control for Autonomous Wireless Space Networks: Extending Space-O-RAN with MCP-Driven Distributed Intelligence 

**Title (ZH)**: 自主无线太空网络中的代理语义控制：基于MCP驱动的分布式智能扩展Space-O-RAN 

**Authors**: Eduardo Baena, Paolo Testolina, Michele Polese, Sergi Aliaga, Andrew Benincasa, Dimitrios Koutsonikolas, Josep Jornet, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2506.10925)  

**Abstract**: Lunar surface operations impose stringent requirements on wireless communication systems, including autonomy, robustness to disruption, and the ability to adapt to environmental and mission-driven context. While Space-O-RAN provides a distributed orchestration model aligned with 3GPP standards, its decision logic is limited to static policies and lacks semantic integration. We propose a novel extension incorporating a semantic agentic layer enabled by the Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication protocols, allowing context-aware decision making across real-time, near-real-time, and non-real-time control layers. Distributed cognitive agents deployed in rovers, landers, and lunar base stations implement wireless-aware coordination strategies, including delay-adaptive reasoning and bandwidth-aware semantic compression, while interacting with multiple MCP servers to reason over telemetry, locomotion planning, and mission constraints. 

**Abstract (ZH)**: 月球表面操作对无线通信系统提出了严格要求，包括自主性、抗干扰能力和适应环境和任务驱动上下文的能力。尽管Space-O-RAN提供了与3GPP标准一致的分布式协同模型，但其决策逻辑仅限于静态策略且缺乏语义集成。我们提出了一种新的扩展，结合了由Model Context Protocol (MCP)和Agent-to-Agent (A2A)通信协议支持的语义代理层，允许跨实时、近实时和非实时控制层进行上下文感知决策。部署在月球车、着陆器和月球基站中的分布式认知代理实现了无线感知的协调策略，包括延迟自适应推理和带宽感知的语义压缩，同时与多个MCP服务器交互，以推理遥测、运动规划和任务约束。 

---
# Robustly Improving LLM Fairness in Realistic Settings via Interpretability 

**Title (ZH)**: 通过可解释性在实际场景中稳健提高大语言模型公平性 

**Authors**: Adam Karvonen, Samuel Marks  

**Link**: [PDF](https://arxiv.org/pdf/2506.10922)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios. Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning. To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance. Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）在高风险招聘应用中越来越广泛部署，对其直接决定人们的职业生涯和生计产生影响。虽然以往研究表明简单的反偏见提示可以在受控评估中消除人口统计学偏见，但我们发现当引入现实的上下文细节时，这些缓解措施会失效。我们通过内部偏见缓解措施解决了这些失败：通过识别并中和模型激活中的敏感属性方向，我们在所有测试场景中实现了稳健的偏见减少。在领先的商品化（GPT-4o、Claude 4 Sonnet、Gemini 2.5 Flash）和开源模型（Gemma-2 27B、Gemma-3、Mistral-24B）中，我们发现添加现实上下文，如公司名称、来自公共职业页面的文化描述以及选择性的招聘限制（例如，“仅接受前10%的候选人”），会显著引入种族和性别偏见（面试率最多相差12%）。当这些偏见出现时，它们在所有测试模型和场景中始终表现为更倾向于黑人而非白人候选人和女性而非男性候选人。此外，模型可以从微妙线索（如大学隶属关系）中推断出人口统计信息并产生偏见，即使检查模型的推理链也不易察觉这些偏见。为了应对这些局限性，我们的内部偏见缓解措施识别与种族和性别相关的方向，并在推理时应用仿射概念编辑。尽管使用来自简单合成数据集的方向，干预措施表现出稳健性，一致将偏见降低到非常低的水平（通常低于1%，始终低于2.5%），同时很大程度上保持了模型性能。我们的研究结果表明，部署LLMs进行招聘的实践者应采用更现实的评估方法，并考虑内部缓解策略以实现公平结果。 

---
# M4V: Multi-Modal Mamba for Text-to-Video Generation 

**Title (ZH)**: 多模态Mamba：面向文本到视频生成的多模态模型 

**Authors**: Jiancheng Huang, Gengwei Zhang, Zequn Jie, Siyu Jiao, Yinlong Qian, Ling Chen, Yunchao Wei, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.10915)  

**Abstract**: Text-to-video generation has significantly enriched content creation and holds the potential to evolve into powerful world simulators. However, modeling the vast spatiotemporal space remains computationally demanding, particularly when employing Transformers, which incur quadratic complexity in sequence processing and thus limit practical applications. Recent advancements in linear-time sequence modeling, particularly the Mamba architecture, offer a more efficient alternative. Nevertheless, its plain design limits its direct applicability to multi-modal and spatiotemporal video generation tasks. To address these challenges, we introduce M4V, a Multi-Modal Mamba framework for text-to-video generation. Specifically, we propose a multi-modal diffusion Mamba (MM-DiM) block that enables seamless integration of multi-modal information and spatiotemporal modeling through a multi-modal token re-composition design. As a result, the Mamba blocks in M4V reduce FLOPs by 45% compared to the attention-based alternative when generating videos at 768$\times$1280 resolution. Additionally, to mitigate the visual quality degradation in long-context autoregressive generation processes, we introduce a reward learning strategy that further enhances per-frame visual realism. Extensive experiments on text-to-video benchmarks demonstrate M4V's ability to produce high-quality videos while significantly lowering computational costs. Code and models will be publicly available at this https URL. 

**Abstract (ZH)**: 多模态Mamba框架：面向文本到视频生成的时空建模与高效计算 

---
# BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP 

**Title (ZH)**: BioClinical ModernBERT：一种生物医学和临床NLP领域的先进长上下文编码器 

**Authors**: Thomas Sounack, Joshua Davis, Brigitte Durieux, Antoine Chaffin, Tom J. Pollard, Eric Lehman, Alistair E. W. Johnson, Matthew McDermott, Tristan Naumann, Charlotta Lindvall  

**Link**: [PDF](https://arxiv.org/pdf/2506.10896)  

**Abstract**: Encoder-based transformer models are central to biomedical and clinical Natural Language Processing (NLP), as their bidirectional self-attention makes them well-suited for efficiently extracting structured information from unstructured text through discriminative tasks. However, encoders have seen slower development compared to decoder models, leading to limited domain adaptation in biomedical and clinical settings. We introduce BioClinical ModernBERT, a domain-adapted encoder that builds on the recent ModernBERT release, incorporating long-context processing and substantial improvements in speed and performance for biomedical and clinical NLP. BioClinical ModernBERT is developed through continued pretraining on the largest biomedical and clinical corpus to date, with over 53.5 billion tokens, and addresses a key limitation of prior clinical encoders by leveraging 20 datasets from diverse institutions, domains, and geographic regions, rather than relying on data from a single source. It outperforms existing biomedical and clinical encoders on four downstream tasks spanning a broad range of use cases. We release both base (150M parameters) and large (396M parameters) versions of BioClinical ModernBERT, along with training checkpoints to support further research. 

**Abstract (ZH)**: 基于编码器的变压器模型在生物医学和临床自然语言处理(NLP)中至关重要，因为它们的双向自注意力机制使其能够通过辨别任务高效地从无结构文本中提取结构化信息。然而，相较于解码器模型，编码器的发展较慢，导致生物医学和临床环境中的领域适应能力有限。我们介绍了BioClinical ModernBERT，这是一种基于最近发布的ModernBERT改进而来的领域适应编码器，它结合了长上下文处理，并在生物医学和临床NLP方面在速度和性能上取得了显著改进。BioClinical ModernBERT通过在迄今最大规模的生物医学和临床文本语料库上进行连续预训练（超过535亿个 token），并利用来自不同机构、领域和地理区域的20个数据集，解决了先前临床编码器的关键限制，而无需依赖单一数据源。它在四个下游任务上优于现有的生物医学和临床编码器，这些任务涵盖了广泛的应用场景。我们发布了BioClinical ModernBERT的基本版本（1.5亿参数）和大型版本（3.96亿参数），并提供了训练检查点以支持进一步研究。 

---
# AIR: Zero-shot Generative Model Adaptation with Iterative Refinement 

**Title (ZH)**: AIR：迭代精炼的零样本生成模型适应 

**Authors**: Guimeng Liu, Milad Abdollahzadeh, Ngai-Man Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.10895)  

**Abstract**: Zero-shot generative model adaptation (ZSGM) aims to adapt a pre-trained generator to a target domain using only text guidance and without any samples from the target domain. Central to recent ZSGM approaches are directional loss which use the text guidance in the form of aligning the image offset with text offset in the embedding space of a vision-language model like CLIP. This is similar to the analogical reasoning in NLP where the offset between one pair of words is used to identify a missing element in another pair by aligning the offset between these two pairs. However, a major limitation of existing ZSGM methods is that the learning objective assumes the complete alignment between image offset and text offset in the CLIP embedding space, resulting in quality degrade in generated images. Our work makes two main contributions. Inspired by the offset misalignment studies in NLP, as our first contribution, we perform an empirical study to analyze the misalignment between text offset and image offset in CLIP embedding space for various large publicly available datasets. Our important finding is that offset misalignment in CLIP embedding space is correlated with concept distance, i.e., close concepts have a less offset misalignment. To address the limitations of the current approaches, as our second contribution, we propose Adaptation with Iterative Refinement (AIR) which is the first ZSGM approach to focus on improving target domain image quality based on our new insight on offset this http URL, quantitative, and user study in 26 experiment setups consistently demonstrate the proposed AIR approach achieves SOTA performance. Additional experiments are in Supp. 

**Abstract (ZH)**: 零样本生成模型适应（ZSGM）旨在仅使用文本指导和支持目标领域无任何目标领域样本的情况下，适应预训练生成器到目标领域。近期ZSGM方法的核心在于方向性损失，它通过在类似于CLIP这类视觉-语言模型的嵌入空间中对齐图像偏移和文本偏移来利用文本指导。这类似于NLP中的类比推理，其中通过对齐一个词对之间的偏移来识别另一个词对中缺失的元素。然而，现有ZSGM方法的主要局限性在于学习目标假定了CLIP嵌入空间中图像偏移和文本偏移的完全对齐，导致生成的图像质量下降。我们的工作做出了两项主要贡献。受NLP中偏移不对齐研究的启发，作为我们第一项贡献，我们进行了实证研究，分析了各种大型公开数据集中CLIP嵌入空间中文本偏移和图像偏移之间的不对齐情况。我们的重要发现是，CLIP嵌入空间中的偏移不对齐与概念距离相关，即接近的概念具有较少的偏移不对齐。为了克服当前方法的局限性，作为我们第二项贡献，我们提出了一种迭代校准的适应方法（Adaptation with Iterative Refinement, AIR），这是首个基于我们对偏移的新洞察专注于提高目标领域图像质量的ZSGM方法。在26种实验设置下的定性、定量和用户研究结果一致表明，提出的AIR方法达到了当前最佳性能。附加实验详见附录。 

---
# The Diffusion Duality 

**Title (ZH)**: 扩散二重性 

**Authors**: Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2506.10892)  

**Abstract**: Uniform-state discrete diffusion models hold the promise of fast text generation due to their inherent ability to self-correct. However, they are typically outperformed by autoregressive models and masked diffusion models. In this work, we narrow this performance gap by leveraging a key insight: Uniform-state diffusion processes naturally emerge from an underlying Gaussian diffusion. Our method, Duo, transfers powerful techniques from Gaussian diffusion to improve both training and sampling. First, we introduce a curriculum learning strategy guided by the Gaussian process, doubling training speed by reducing variance. Models trained with curriculum learning surpass autoregressive models in zero-shot perplexity on 3 of 7 benchmarks. Second, we present Discrete Consistency Distillation, which adapts consistency distillation from the continuous to the discrete setting. This algorithm unlocks few-step generation in diffusion language models by accelerating sampling by two orders of magnitude. We provide the code and model checkpoints on the project page: this http URL 

**Abstract (ZH)**: 均匀状态离散扩散模型由于其固有的自校正能力，有望实现快速文本生成。然而，它们通常逊色于自回归模型和蒙混扩散模型。在这项工作中，我们通过利用一个关键洞察来缩小性能差距：均匀状态扩散过程自然地源自一个潜在的高斯扩散过程。我们的方法Duo将来自高斯扩散的强大技术转移到训练和采样中。首先，我们引入了一种由高斯过程指导的课程学习策略，通过降低方差将训练速度翻倍。使用课程学习训练的模型在3个基准上以零样本困惑度超越了自回归模型。其次，我们提出了离散一致性蒸馏，将一致性蒸馏从连续域扩展到离散域。该算法通过将采样速度提高两个数量级，解锁了扩散语言模型的多步生成。我们在项目页面提供了代码和模型检查点：this http URL。 

---
# Slimming Down LLMs Without Losing Their Minds 

**Title (ZH)**: 精简LLMs而不丧失其能力 

**Authors**: Qingda  

**Link**: [PDF](https://arxiv.org/pdf/2506.10885)  

**Abstract**: This paper investigates and validates the impact of fine-tuning on large language model performance, focusing on parameter-efficient methods (LoRA and QLoRA). We evaluate model capabilities across three key domains: (1) commonsense reasoning (HellaSwag), (2) mathematical reasoning (GSM8K), and (3) multi-domain knowledge (MMLU-CS).
Our findings demonstrate that: (1) LoRA-based methods effectively improve task-specific performance while maintaining computational efficiency, and (2) performance strongly depends on alignment between fine-tuning dataset and benchmark tasks. The study provides both theoretical insights into parameter-efficient mechanisms and practical guidance for developers implementing efficient LLM adaptation with limited resources. 

**Abstract (ZH)**: 本文 investigate 并验证了微调对大型语言模型性能的影响，并专注于参数高效方法（LoRA 和 QLoRA）。我们评估了模型能力在三个关键领域：(1) 常识推理（HellaSwag），(2) 数学推理（GSM8K），以及 (3) 多领域知识（MMLU-CS）。

我们的发现表明：(1) 基于 LoRA 的方法能有效提高任务特定性能同时保持计算效率，(2) 性能强烈依赖于微调数据集与基准任务之间的对齐。该研究提供了关于参数高效机制的理论见解，并为使用有限资源实现高效 LLM 调适提供了实用指导。 

---
# Data-Driven Prediction of Dynamic Interactions Between Robot Appendage and Granular Material 

**Title (ZH)**: 基于数据驱动的机器人肢体与颗粒材料动态相互作用预测 

**Authors**: Guanjin Wang, Xiangxue Zhao, Shapour Azarm, Balakumar Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2506.10875)  

**Abstract**: An alternative data-driven modeling approach has been proposed and employed to gain fundamental insights into robot motion interaction with granular terrain at certain length scales. The approach is based on an integration of dimension reduction (Sequentially Truncated Higher-Order Singular Value Decomposition), surrogate modeling (Gaussian Process), and data assimilation techniques (Reduced Order Particle Filter). This approach can be used online and is based on offline data, obtained from the offline collection of high-fidelity simulation data and a set of sparse experimental data. The results have shown that orders of magnitude reduction in computational time can be obtained from the proposed data-driven modeling approach compared with physics-based high-fidelity simulations. With only simulation data as input, the data-driven prediction technique can generate predictions that have comparable accuracy as simulations. With both simulation data and sparse physical experimental measurement as input, the data-driven approach with its embedded data assimilation techniques has the potential in outperforming only high-fidelity simulations for the long-horizon predictions. In addition, it is demonstrated that the data-driven modeling approach can also reproduce the scaling relationship recovered by physics-based simulations for maximum resistive forces, which may indicate its general predictability beyond a case-by-case basis. The results are expected to help robot navigation and exploration in unknown and complex terrains during both online and offline phases. 

**Abstract (ZH)**: 一种数据驱动建模方法被提出并应用于在特定长度尺度下获得机器人运动与颗粒地形相互作用的基本见解。该方法基于降维（顺序截断高阶singular值分解）、代理建模（高斯过程）和数据同化技术（降阶粒子滤波）的集成。该方法可以在线使用，并基于离线收集的高保真仿真数据和一组稀疏的实验数据。结果表明，与基于物理的高保真仿真相比，所提出的数据驱动建模方法可以大幅减少计算时间。仅使用仿真数据作为输入，数据驱动预测技术可以生成与仿真具有可比准确性的预测。同时，利用仿真数据和稀疏物理实验测量作为输入，数据驱动方法结合嵌入的数据同化技术有望在长时预测中优于仅基于高保真仿真的方法。此外，还展示了数据驱动建模方法可以重现基于物理仿真的最大阻力的标度关系，这可能表明其在个例之外的一般预测能力。预计这些结果将有助于机器人在未知和复杂地形中的导航与勘探，在离线和在线阶段均适用。 

---
# A multi-scale loss formulation for learning a probabilistic model with proper score optimisation 

**Title (ZH)**: 多尺度损失函数 formulation 用于具有适当评分优化的概率模型学习 

**Authors**: Simon Lang, Martin Leutbecher, Pedro Maciel  

**Link**: [PDF](https://arxiv.org/pdf/2506.10868)  

**Abstract**: We assess the impact of a multi-scale loss formulation for training probabilistic machine-learned weather forecasting models. The multi-scale loss is tested in AIFS-CRPS, a machine-learned weather forecasting model developed at the European Centre for Medium-Range Weather Forecasts (ECMWF). AIFS-CRPS is trained by directly optimising the almost fair continuous ranked probability score (afCRPS). The multi-scale loss better constrains small scale variability without negatively impacting forecast skill. This opens up promising directions for future work in scale-aware model training. 

**Abstract (ZH)**: 我们评估了多尺度损失函数对训练概率机器学习天气预报模型的影响。多尺度损失在欧洲中期天气预报中心（ECMWF）开发的AIFS-CRPS机器学习天气预报模型中进行了测试，通过直接优化近公允连续排名概率评分（afCRPS）进行训练。多尺度损失更好地约束了小尺度变异性，而不负面影响预报技巧。这为未来的尺度感知模型训练开辟了有 Promise 的方向。 

---
# Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information 

**Title (ZH)**: 通过后聚合全局上下文信息的精确零样本点wise排名lescoringewithLLMs 

**Authors**: Kehan Long, Shasha Li, Chen Xu, Jintao Tang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10859)  

**Abstract**: Recent advancements have successfully harnessed the power of Large Language Models (LLMs) for zero-shot document ranking, exploring a variety of prompting strategies. Comparative approaches like pairwise and listwise achieve high effectiveness but are computationally intensive and thus less practical for larger-scale applications. Scoring-based pointwise approaches exhibit superior efficiency by independently and simultaneously generating the relevance scores for each candidate document. However, this independence ignores critical comparative insights between documents, resulting in inconsistent scoring and suboptimal performance. In this paper, we aim to improve the effectiveness of pointwise methods while preserving their efficiency through two key innovations: (1) We propose a novel Global-Consistent Comparative Pointwise Ranking (GCCP) strategy that incorporates global reference comparisons between each candidate and an anchor document to generate contrastive relevance scores. We strategically design the anchor document as a query-focused summary of pseudo-relevant candidates, which serves as an effective reference point by capturing the global context for document comparison. (2) These contrastive relevance scores can be efficiently Post-Aggregated with existing pointwise methods, seamlessly integrating essential Global Context information in a training-free manner (PAGC). Extensive experiments on the TREC DL and BEIR benchmark demonstrate that our approach significantly outperforms previous pointwise methods while maintaining comparable efficiency. Our method also achieves competitive performance against comparative methods that require substantially more computational resources. More analyses further validate the efficacy of our anchor construction strategy. 

**Abstract (ZH)**: 最近的研究成功利用了大型语言模型（LLMs）进行零样本文档排名，并探索了多种提示策略。对比方法如对切和列表方法表现出较高的有效性，但计算成本高，因此在大规模应用中不太实用。基于评分的点wise方法通过独立同时生成每个候选文档的相关性评分，展现出优越的效率。然而，这种独立性忽略了文档之间重要的对比洞察，导致评分不一致且性能欠佳。本文旨在通过两项关键创新来提高点wise方法的有效性，同时保持其效率：(1) 我们提出了一种新颖的全局一致对比点wise排名（GCCP）策略，该策略在每个候选文档和锚文档之间引入全局引用对比，生成对比相关性评分。我们战略性地将锚文档设计为基于查询的伪相关候选摘要，这为文档对比提供了一个有效的参考点，捕获了文档比较的全局上下文。(2) 这些对比相关性评分可以与现有的点wise方法高效地后聚合（PAGC），以无训练方式无缝整合关键的全局上下文信息。在TREC DL和BEIR基准上的广泛实验表明，我们的方法在保持同等效率的同时显著优于之前的方法。此外，我们的方法在需要更大量计算资源的对比方法中也取得了竞争力的性能。更多分析进一步验证了我们锚构建策略的有效性。 

---
# VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos 

**Title (ZH)**: VRBench: 长叙事视频多步推理基准 

**Authors**: Jiashuo Yu, Yue Wu, Meng Chu, Zhifei Ren, Zizheng Huang, Pei Chu, Ruijie Zhang, Yinan He, Qirui Li, Songze Li, Zhenxiang Li, Zhongying Tu, Conghui He, Yu Qiao, Yali Wang, Yi Wang, Limin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10857)  

**Abstract**: We present VRBench, the first long narrative video benchmark crafted for evaluating large models' multi-step reasoning capabilities, addressing limitations in existing evaluations that overlook temporal reasoning and procedural validity. It comprises 1,010 long videos (with an average duration of 1.6 hours), along with 9,468 human-labeled multi-step question-answering pairs and 30,292 reasoning steps with timestamps. These videos are curated via a multi-stage filtering process including expert inter-rater reviewing to prioritize plot coherence. We develop a human-AI collaborative framework that generates coherent reasoning chains, each requiring multiple temporally grounded steps, spanning seven types (e.g., event attribution, implicit inference). VRBench designs a multi-phase evaluation pipeline that assesses models at both the outcome and process levels. Apart from the MCQs for the final results, we propose a progress-level LLM-guided scoring metric to evaluate the quality of the reasoning chain from multiple dimensions comprehensively. Through extensive evaluations of 12 LLMs and 16 VLMs on VRBench, we undertake a thorough analysis and provide valuable insights that advance the field of multi-step reasoning. 

**Abstract (ZH)**: VRBench: 针对大规模语言模型多步推理能力评估的第一个长叙事视频基准 

---
# Accelerating Diffusion Large Language Models with SlowFast: The Three Golden Principles 

**Title (ZH)**: 用SlowFast加速扩散大型语言模型：三大黄金原则 

**Authors**: Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10848)  

**Abstract**: Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across benchmarks and models show that SlowFast Sampling achieves up to 15.63$\times$ speedup on LLaDA with minimal accuracy drop, and up to 34.22$\times$ when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation. 

**Abstract (ZH)**: 基于扩散的语言模型（dLLMs）通过实现并行令牌生成和显著降低推断延迟，已成为传统自回归LLMs的有前途的替代方案。然而，现有的dLLMs抽样策略，如基于信心或半自回归解码，经常表现出静态行为，导致效率低下和灵活性受限。在本文中，我们提出了一种新颖的动态抽样策略SlowFast Sampling，该策略能够自适应地交替使用探索性和加速解码阶段。我们的方法遵循三个金科玉律：确定性原则、收敛性原则和位置性原则，以决定何时以及在哪里可以有效地解码令牌。我们还将我们的策略与dLLM-Cache结合使用，以减少冗余计算。在多个基准测试和模型上的广泛实验显示，SlowFast Sampling在LLaDA上的速度提升可达15.63倍，精度下降 minimal，在结合缓存时可达34.22倍。值得注意的是，我们的方法在吞吐量方面优于强自回归基线如LLaMA3 8B，证明了精心设计的抽样可以充分发挥dLLMs的潜力以实现快速和高质量的生成。 

---
# Post-Training Quantization for Video Matting 

**Title (ZH)**: 视频抠像的后训练量化 

**Authors**: Tianrui Zhu, Houyuan Chen, Ruihao Gong, Michele Magno, Haotong Qin, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10840)  

**Abstract**: Video matting is crucial for applications such as film production and virtual reality, yet deploying its computationally intensive models on resource-constrained devices presents challenges. Quantization is a key technique for model compression and acceleration. As an efficient approach, Post-Training Quantization (PTQ) is still in its nascent stages for video matting, facing significant hurdles in maintaining accuracy and temporal coherence. To address these challenges, this paper proposes a novel and general PTQ framework specifically designed for video matting models, marking, to the best of our knowledge, the first systematic attempt in this domain. Our contributions include: (1) A two-stage PTQ strategy that combines block-reconstruction-based optimization for fast, stable initial quantization and local dependency capture, followed by a global calibration of quantization parameters to minimize accuracy loss. (2) A Statistically-Driven Global Affine Calibration (GAC) method that enables the network to compensate for cumulative statistical distortions arising from factors such as neglected BN layer effects, even reducing the error of existing PTQ methods on video matting tasks up to 20%. (3) An Optical Flow Assistance (OFA) component that leverages temporal and semantic priors from frames to guide the PTQ process, enhancing the model's ability to distinguish moving foregrounds in complex scenes and ultimately achieving near full-precision performance even under ultra-low-bit quantization. Comprehensive quantitative and visual results show that our PTQ4VM achieves the state-of-the-art accuracy performance across different bit-widths compared to the existing quantization methods. We highlight that the 4-bit PTQ4VM even achieves performance close to the full-precision counterpart while enjoying 8x FLOP savings. 

**Abstract (ZH)**: 视频抠图在电影制作和虚拟现实等应用中至关重要，但将其计算密集型模型部署在资源受限设备上面临着挑战。量化是模型压缩和加速的关键技术。作为一种有效的手段，后训练量化（PTQ）仍处于初步阶段，特别是在视频抠图领域，保持准确性和时间一致性方面面临着重大挑战。为了解决这些挑战，本文提出了一种专为视频抠图模型设计的新型通用PTQ框架，据我们所知，这是在此领域中的首次系统性尝试。我们的贡献包括：（1）一种两阶段PTQ策略，结合块重建优化进行快速、稳定的初始量化和局部依赖捕捉，随后进行全局量化参数校准以最小化准确率损失。（2）一种统计驱动的全局仿射校准（GAC）方法，使网络能够补偿由未忽略BN层效应等因素引起的累积统计失真，甚至在视频抠图任务中将现有PTQ方法的误差降低多达20%。（3）一种光流辅助（OFA）组件，利用帧中的时间和语义先验引导PTQ过程，增强模型在复杂场景中区分移动前景的能力，最终即便在极低位宽量化下也能实现接近全精度的性能。综合定量和视觉结果表明，我们的PTQ4VM在不同位宽下相较于现有量化方法实现了最先进的准确率性能。我们强调，4-bit的PTQ4VM即使在享受8倍FLOP节省的情况下，性能也接近全精度的版本。 

---
# Efficiency Robustness of Dynamic Deep Learning Systems 

**Title (ZH)**: 动态深度学习系统的效率稳健性 

**Authors**: Ravishka Rathnasuriya, Tingxi Li, Zexin Xu, Zihe Song, Mirazul Haque, Simin Chen, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10831)  

**Abstract**: Deep Learning Systems (DLSs) are increasingly deployed in real-time applications, including those in resourceconstrained environments such as mobile and IoT devices. To address efficiency challenges, Dynamic Deep Learning Systems (DDLSs) adapt inference computation based on input complexity, reducing overhead. While this dynamic behavior improves efficiency, such behavior introduces new attack surfaces. In particular, efficiency adversarial attacks exploit these dynamic mechanisms to degrade system performance. This paper systematically explores efficiency robustness of DDLSs, presenting the first comprehensive taxonomy of efficiency attacks. We categorize these attacks based on three dynamic behaviors: (i) attacks on dynamic computations per inference, (ii) attacks on dynamic inference iterations, and (iii) attacks on dynamic output production for downstream tasks. Through an in-depth evaluation, we analyze adversarial strategies that target DDLSs efficiency and identify key challenges in securing these systems. In addition, we investigate existing defense mechanisms, demonstrating their limitations against increasingly popular efficiency attacks and the necessity for novel mitigation strategies to secure future adaptive DDLSs. 

**Abstract (ZH)**: 深度学习系统（DLSs）越来越多地应用于实时应用，包括移动设备和物联网设备等资源受限环境中。为了应对效率挑战，动态深度学习系统（DDLSs）根据输入的复杂性调整推理计算，从而减少开销。虽然这种动态行为能够提高效率，但也引入了新的攻击表面。特别是效率对抗攻击利用这些动态机制以降低系统性能。本文系统地探讨了DDLSs的效率鲁棒性，提出了首个全面的效率攻击分类。我们根据三种动态行为对这些攻击进行分类：(i) 每次推理中的动态计算攻击，(ii) 动态推理迭代攻击，以及(iii) 动态输出生成攻击以供下游任务使用。通过深入评估，我们分析了针对DDLSs效率的目标攻击策略，并确定了这些系统安全性的关键挑战。此外，我们还调查了现有的防御机制，证明它们在应对日益流行的效率攻击方面的局限性，并强调未来需要新的缓解策略来确保动态适应性DDLSs的安全性。 

---
# LLM-Driven Personalized Answer Generation and Evaluation 

**Title (ZH)**: LLM驱动的个性化答案生成与评估 

**Authors**: Mohammadreza Molavi, Mohammadreza Tavakoli, Mohammad Moein, Abdolali Faraji, Gábor Kismihók  

**Link**: [PDF](https://arxiv.org/pdf/2506.10829)  

**Abstract**: Online learning has experienced rapid growth due to its flexibility and accessibility. Personalization, adapted to the needs of individual learners, is crucial for enhancing the learning experience, particularly in online settings. A key aspect of personalization is providing learners with answers customized to their specific questions. This paper therefore explores the potential of Large Language Models (LLMs) to generate personalized answers to learners' questions, thereby enhancing engagement and reducing the workload on educators. To evaluate the effectiveness of LLMs in this context, we conducted a comprehensive study using the StackExchange platform in two distinct areas: language learning and programming. We developed a framework and a dataset for validating automatically generated personalized answers. Subsequently, we generated personalized answers using different strategies, including 0-shot, 1-shot, and few-shot scenarios. The generated answers were evaluated using three methods: 1. BERTScore, 2. LLM evaluation, and 3. human evaluation. Our findings indicated that providing LLMs with examples of desired answers (from the learner or similar learners) can significantly enhance the LLMs' ability to tailor responses to individual learners' needs. 

**Abstract (ZH)**: 在线学习由于其灵活性和易访问性经历了快速成长。个性化学习，适应个别学习者的需求，对于增强学习体验，特别是在在线环境中尤为重要。个性化的一个关键方面是为学习者提供针对其特定问题的定制答案。因此，本文探索了大规模语言模型（LLMs）生成学习者个性化答案的潜力，从而提高参与度并减轻教育者的负担。为了评估LLMs在这种情境下的有效性，我们在StackExchange平台的两个不同领域——语言学习和编程——进行了全面的研究。我们开发了一个框架和数据集来验证自动生成的个性化答案。之后，我们使用不同的策略生成了个性化答案，包括零样本、一样本和少样本场景。生成的答案通过三种方法进行了评估：1. BERTScore，2. LLM评估，3. 人工评估。我们的研究表明，为LLMs提供所需答案的示例（来自学习者或类似学习者）可以显著增强其根据个别学习者需求定制回应的能力。 

---
# Generalist Models in Medical Image Segmentation: A Survey and Performance Comparison with Task-Specific Approaches 

**Title (ZH)**: 医学图像分割中的通用模型：一项综述及与任务专用方法的性能比较 

**Authors**: Andrea Moglia, Matteo Leccardi, Matteo Cavicchioli, Alice Maccarini, Marco Marcon, Luca Mainardi, Pietro Cerveri  

**Link**: [PDF](https://arxiv.org/pdf/2506.10825)  

**Abstract**: Following the successful paradigm shift of large language models, leveraging pre-training on a massive corpus of data and fine-tuning on different downstream tasks, generalist models have made their foray into computer vision. The introduction of Segment Anything Model (SAM) set a milestone on segmentation of natural images, inspiring the design of a multitude of architectures for medical image segmentation. In this survey we offer a comprehensive and in-depth investigation on generalist models for medical image segmentation. We start with an introduction on the fundamentals concepts underpinning their development. Then, we provide a taxonomy on the different declinations of SAM in terms of zero-shot, few-shot, fine-tuning, adapters, on the recent SAM 2, on other innovative models trained on images alone, and others trained on both text and images. We thoroughly analyze their performances at the level of both primary research and best-in-literature, followed by a rigorous comparison with the state-of-the-art task-specific models. We emphasize the need to address challenges in terms of compliance with regulatory frameworks, privacy and security laws, budget, and trustworthy artificial intelligence (AI). Finally, we share our perspective on future directions concerning synthetic data, early fusion, lessons learnt from generalist models in natural language processing, agentic AI and physical AI, and clinical translation. 

**Abstract (ZH)**: 跟随大型语言模型的成功范式转变，借助大规模数据的预训练和不同下游任务的微调，通用模型已经进入计算机视觉领域。Segment Anything Model (SAM) 的引入在自然图像分割上树立了一个里程碑，激发了多种针对医学图像分割的架构设计。在本文综述中，我们提供了一种全面而深入的调查，探讨通用模型在医学图像分割中的应用。我们首先介绍支撑其发展的基本概念。然后，我们从零样本、少样本、微调、适配器、最新SAM 2、仅图像训练的其他创新模型以及同时在文本和图像上训练的其他模型等方面提供了分类。我们从原始研究和文献最佳实践的角度详细分析了它们的性能，并进行了与最先进的特定任务模型的严格比较。我们强调了合规性、隐私和安全法规、预算以及可信赖人工智能（AI）方面所面临挑战的重要性。最后，我们分享了关于合成数据、早期融合、自然语言处理中通用模型的经验教训、自主AI和物理AI以及临床转化的未来方向观点。 

---
# VideoDeepResearch: Long Video Understanding With Agentic Tool Using 

**Title (ZH)**: VideoDeepResearch: 使用代理工具理解长视频 

**Authors**: Huaying Yuan, Zheng Liu, Junjie Zhou, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.10821)  

**Abstract**: Long video understanding (LVU) presents a significant challenge for current multi-modal large language models (MLLMs) due to the task's inherent complexity and context window constraint. It is widely assumed that addressing LVU tasks requires foundation MLLMs with extended context windows, strong visual perception capabilities, and proficient domain expertise. In this work, we challenge this common belief by introducing VideoDeepResearch, a novel agentic framework for long video understanding. Our approach relies solely on a text-only large reasoning model (LRM) combined with a modular multi-modal toolkit, including multimodal retrievers and visual perceivers, all of which are readily available in practice. For each LVU task, the system formulates a problem-solving strategy through reasoning, while selectively accessing and utilizing essential video content via tool using. We conduct extensive experiments on popular LVU benchmarks, including MLVU, Video-MME, and LVBench. Our results demonstrate that VideoDeepResearch achieves substantial improvements over existing MLLM baselines, surpassing the previous state-of-the-art by 9.6%, 6.6%, and 3.9% on MLVU (test), LVBench, and LongVideoBench, respectively. These findings highlight the promise of agentic systems in overcoming key challenges in LVU problems. 

**Abstract (ZH)**: 长视频理解（LVU）对于当前的多模态大型语言模型（MLLMs）而言 poses 一项显著挑战，主要是由于该任务固有的复杂性和上下文窗口限制。人们普遍认为，解决LVU任务需要具有扩展上下文窗口、强大视觉感知能力和深厚领域专业知识的基础MLLMs。在本工作中，我们通过引入VideoDeepResearch，一种新颖的自主框架来挑战这一常见信念，VideoDeepResearch为长视频理解提供了解决方案。我们的方法仅依赖于一个纯文本大型推理模型（LRM），并结合了一个模块化的多模态工具包，包括多模态检索器和视觉感知器，它们在实践中都是现成可用的。对于每个LVU任务，系统通过推理来制定问题解决策略，并根据需要访问和利用关键视频内容。我们在流行的LVU基准测试中进行了广泛的实验，包括MLVU、Video-MME和LVBench。我们的结果表明，VideoDeepResearch在MLVU（测试）、LVBench和LongVideoBench上的表现分别优于现有MLLM基线9.6%、6.6%和3.9%，这些发现突显了自主系统在克服LVU问题核心挑战方面的潜力。 

---
# What Users Value and Critique: Large-Scale Analysis of User Feedback on AI-Powered Mobile Apps 

**Title (ZH)**: 用户的价值取向与批评：对AI驱动移动应用用户反馈的大规模分析 

**Authors**: Vinaik Chhetri, Krishna Upadhyay, A.B. Siddique, Umar Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2506.10785)  

**Abstract**: Artificial Intelligence (AI)-powered features have rapidly proliferated across mobile apps in various domains, including productivity, education, entertainment, and creativity. However, how users perceive, evaluate, and critique these AI features remains largely unexplored, primarily due to the overwhelming volume of user feedback. In this work, we present the first comprehensive, large-scale study of user feedback on AI-powered mobile apps, leveraging a curated dataset of 292 AI-driven apps across 14 categories with 894K AI-specific reviews from Google Play. We develop and validate a multi-stage analysis pipeline that begins with a human-labeled benchmark and systematically evaluates large language models (LLMs) and prompting strategies. Each stage, including review classification, aspect-sentiment extraction, and clustering, is validated for accuracy and consistency. Our pipeline enables scalable, high-precision analysis of user feedback, extracting over one million aspect-sentiment pairs clustered into 18 positive and 15 negative user topics. Our analysis reveals that users consistently focus on a narrow set of themes: positive comments emphasize productivity, reliability, and personalized assistance, while negative feedback highlights technical failures (e.g., scanning and recognition), pricing concerns, and limitations in language support. Our pipeline surfaces both satisfaction with one feature and frustration with another within the same review. These fine-grained, co-occurring sentiments are often missed by traditional approaches that treat positive and negative feedback in isolation or rely on coarse-grained analysis. To this end, our approach provides a more faithful reflection of the real-world user experiences with AI-powered apps. Category-aware analysis further uncovers both universal drivers of satisfaction and domain-specific frustrations. 

**Abstract (ZH)**: 基于人工智能功能的移动应用程序用户反馈的综合大规模研究：揭示用户体验的主题与矛盾 

---
# Improving Named Entity Transcription with Contextual LLM-based Revision 

**Title (ZH)**: 基于上下文大模型的修订改进命名实体转写 

**Authors**: Viet Anh Trinh, Xinlu He, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2506.10779)  

**Abstract**: With recent advances in modeling and the increasing amount of supervised training data, automatic speech recognition (ASR) systems have achieved remarkable performance on general speech. However, the word error rate (WER) of state-of-the-art ASR remains high for named entities. Since named entities are often the most critical keywords, misrecognizing them can affect all downstream applications, especially when the ASR system functions as the front end of a complex system. In this paper, we introduce a large language model (LLM) revision mechanism to revise incorrect named entities in ASR predictions by leveraging the LLM's reasoning ability as well as local context (e.g., lecture notes) containing a set of correct named entities. Finally, we introduce the NER-MIT-OpenCourseWare dataset, containing 45 hours of data from MIT courses for development and testing. On this dataset, our proposed technique achieves up to 30\% relative WER reduction for named entities. 

**Abstract (ZH)**: 近年来，随着建模技术的进步和监督训练数据的增多，自动语音识别（ASR）系统在通用语音上的性能取得了显著提升。然而，最先进的ASR系统在识别命名实体时的词错误率（WER）仍然较高。由于命名实体通常是最重要的关键词，识别错误会严重影响下游应用，特别是在ASR系统作为复杂系统前端时更为明显。在本文中，我们提出了一种大规模语言模型（LLM）修订机制，通过利用LLM的推理能力和包含一组正确命名实体的局部上下文（如讲座笔记），来修订ASR预测中的错误命名实体。此外，我们引入了NER-MIT-OpenCourseWare数据集，包含了来自麻省理工学院课程的45小时数据，用于开发和测试。在该数据集上，我们提出的技术实现了最高30%的命名实体相对WER减少。 

---
# SlotPi: Physics-informed Object-centric Reasoning Models 

**Title (ZH)**: SlotPi：物理信息导向的对象中心推理模型 

**Authors**: Jian Li, Wan Han, Ning Lin, Yu-Liang Zhan, Ruizhi Chengze, Haining Wang, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.10778)  

**Abstract**: Understanding and reasoning about dynamics governed by physical laws through visual observation, akin to human capabilities in the real world, poses significant challenges. Currently, object-centric dynamic simulation methods, which emulate human behavior, have achieved notable progress but overlook two critical aspects: 1) the integration of physical knowledge into models. Humans gain physical insights by observing the world and apply this knowledge to accurately reason about various dynamic scenarios; 2) the validation of model adaptability across diverse scenarios. Real-world dynamics, especially those involving fluids and objects, demand models that not only capture object interactions but also simulate fluid flow characteristics. To address these gaps, we introduce SlotPi, a slot-based physics-informed object-centric reasoning model. SlotPi integrates a physical module based on Hamiltonian principles with a spatio-temporal prediction module for dynamic forecasting. Our experiments highlight the model's strengths in tasks such as prediction and Visual Question Answering (VQA) on benchmark and fluid datasets. Furthermore, we have created a real-world dataset encompassing object interactions, fluid dynamics, and fluid-object interactions, on which we validated our model's capabilities. The model's robust performance across all datasets underscores its strong adaptability, laying a foundation for developing more advanced world models. 

**Abstract (ZH)**: 通过视觉观察理解由物理定律支配的动力学并进行推理，类似于人类在现实世界中的能力，提出了重大的挑战。当前，以对象为中心的动力学仿真方法尽管模仿了人类行为并在领域内取得了显著进展，但仍忽视了两个关键方面：1）将物理知识集成到模型中。人类通过观察世界获得物理直觉，并将这些知识应用于对各种动力学场景的准确推理；2）验证模型在不同场景下的适应性。尤其是涉及流体和物体的现实世界动力学要求模型不仅捕捉对象间的相互作用，还能模拟流体流动特性。为解决这些差距，我们提出了SlotPi，一个基于槽位的物理知情对象中心推理模型。SlotPi 结合了基于Hamilton原理的物理模块和时空预测模块进行动态预测。我们的实验展示了该模型在预测和视觉问答（VQA）任务中的优势，并在基准数据集和流体数据集上进行了验证。此外，我们创建了一个包含对象相互作用、流体动力学和流体-对象相互作用的现实世界数据集，并验证了该模型的能力。模型在所有数据集上的稳健表现突显了其强大的适应性，为开发更高级的世界模型奠定了基础。 

---
# ME: Trigger Element Combination Backdoor Attack on Copyright Infringement 

**Title (ZH)**: ME：版权侵权中的触发元素组合后门攻击 

**Authors**: Feiyu Yang, Siyuan Liang, Aishan Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10776)  

**Abstract**: The capability of generative diffusion models (DMs) like Stable Diffusion (SD) in replicating training data could be taken advantage of by attackers to launch the Copyright Infringement Attack, with duplicated poisoned image-text pairs. SilentBadDiffusion (SBD) is a method proposed recently, which shew outstanding performance in attacking SD in text-to-image tasks. However, the feasible data resources in this area are still limited, some of them are even constrained or prohibited due to the issues like copyright ownership or inappropriate contents; And not all of the images in current datasets are suitable for the proposed attacking methods; Besides, the state-of-the-art (SoTA) performance of SBD is far from ideal when few generated poisoning samples could be adopted for attacks. In this paper, we raised new datasets accessible for researching in attacks like SBD, and proposed Multi-Element (ME) attack method based on SBD by increasing the number of poisonous visual-text elements per poisoned sample to enhance the ability of attacking, while importing Discrete Cosine Transform (DCT) for the poisoned samples to maintain the stealthiness. The Copyright Infringement Rate (CIR) / First Attack Epoch (FAE) we got on the two new datasets were 16.78% / 39.50 and 51.20% / 23.60, respectively close to or even outperformed benchmark Pokemon and Mijourney datasets. In condition of low subsampling ratio (5%, 6 poisoned samples), MESI and DCT earned CIR / FAE of 0.23% / 84.00 and 12.73% / 65.50, both better than original SBD, which failed to attack at all. 

**Abstract (ZH)**: Generative扩散模型在版权侵权攻击中的应用与增强：SilentBadDiffusion及其Multi-Element攻击方法的研究 

---
# Stroke-based Cyclic Amplifier: Image Super-Resolution at Arbitrary Ultra-Large Scales 

**Title (ZH)**: 基于抽样周期的循环放大器：任意超大规模图像超分辨率 

**Authors**: Wenhao Guo, Peng Lu, Xujun Peng, Zhaoran Zhao, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10774)  

**Abstract**: Prior Arbitrary-Scale Image Super-Resolution (ASISR) methods often experience a significant performance decline when the upsampling factor exceeds the range covered by the training data, introducing substantial blurring. To address this issue, we propose a unified model, Stroke-based Cyclic Amplifier (SbCA), for ultra-large upsampling tasks. The key of SbCA is the stroke vector amplifier, which decomposes the image into a series of strokes represented as vector graphics for magnification. Then, the detail completion module also restores missing details, ensuring high-fidelity image reconstruction. Our cyclic strategy achieves ultra-large upsampling by iteratively refining details with this unified SbCA model, trained only once for all, while keeping sub-scales within the training range. Our approach effectively addresses the distribution drift issue and eliminates artifacts, noise and blurring, producing high-quality, high-resolution super-resolved images. Experimental validations on both synthetic and real-world datasets demonstrate that our approach significantly outperforms existing methods in ultra-large upsampling tasks (e.g. $\times100$), delivering visual quality far superior to state-of-the-art techniques. 

**Abstract (ZH)**: 基于笔画循环放大器的超大规模图像超分辨率方法 

---
# Learning Chaotic Dynamics with Neuromorphic Network Dynamics 

**Title (ZH)**: 学习混沌动态的神经形态网络动力学 

**Authors**: Yinhao Xu, Georg A. Gottwald, Zdenka Kuncic  

**Link**: [PDF](https://arxiv.org/pdf/2506.10773)  

**Abstract**: This study investigates how dynamical systems may be learned and modelled with a neuromorphic network which is itself a dynamical system. The neuromorphic network used in this study is based on a complex electrical circuit comprised of memristive elements that produce neuro-synaptic nonlinear responses to input electrical signals. To determine how computation may be performed using the physics of the underlying system, the neuromorphic network was simulated and evaluated on autonomous prediction of a multivariate chaotic time series, implemented with a reservoir computing framework. Through manipulating only input electrodes and voltages, optimal nonlinear dynamical responses were found when input voltages maximise the number of memristive components whose internal dynamics explore the entire dynamical range of the memristor model. Increasing the network coverage with the input electrodes was found to suppress other nonlinear responses that are less conducive to learning. These results provide valuable insights into how a practical neuromorphic network device can be optimised for learning complex dynamical systems using only external control parameters. 

**Abstract (ZH)**: 本研究探讨如何使用自身为动力学系统的神经形态网络来学习和建模动力学系统。该研究中使用的神经形态网络基于由忆阻元件组成的复杂数字电路，这些元件对输入电信号产生神经突触非线性响应。通过模拟和在动力学范围广的忆阻器模型中探究内部动力学的复杂数值时间序列的自主预测性能评估，研究了如何利用底层系统的物理特性来执行计算。通过仅调节输入电极和电压，研究发现在最大化探索忆阻器模型整个动力学范围的忆阻器组件数量时，可以实现最优的非线性动力学响应。增加输入电极覆盖范围被发现会抑制其他不利于学习的非线性响应。这些结果为仅通过外部控制参数优化实用神经形态网络设备以学习复杂动力学系统提供了有价值的见解。 

---
# Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding 

**Title (ZH)**: 基于开放词汇目标理解的无人机接地视觉语言导航 

**Authors**: Yuhang Zhang, Haosheng Yu, Jiaping Xiao, Mir Feroskhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10756)  

**Abstract**: Vision-and-language navigation (VLN) is a long-standing challenge in autonomous robotics, aiming to empower agents with the ability to follow human instructions while navigating complex environments. Two key bottlenecks remain in this field: generalization to out-of-distribution environments and reliance on fixed discrete action spaces. To address these challenges, we propose Vision-Language Fly (VLFly), a framework tailored for Unmanned Aerial Vehicles (UAVs) to execute language-guided flight. Without the requirement for localization or active ranging sensors, VLFly outputs continuous velocity commands purely from egocentric observations captured by an onboard monocular camera. The VLFly integrates three modules: an instruction encoder based on a large language model (LLM) that reformulates high-level language into structured prompts, a goal retriever powered by a vision-language model (VLM) that matches these prompts to goal images via vision-language similarity, and a waypoint planner that generates executable trajectories for real-time UAV control. VLFly is evaluated across diverse simulation environments without additional fine-tuning and consistently outperforms all baselines. Moreover, real-world VLN tasks in indoor and outdoor environments under direct and indirect instructions demonstrate that VLFly achieves robust open-vocabulary goal understanding and generalized navigation capabilities, even in the presence of abstract language input. 

**Abstract (ZH)**: 视觉-语言导航（VLN）是自主机器人领域的长期挑战，旨在赋予智能体在复杂环境中遵循人类指令的能力。该领域存在的两大瓶颈是跨分布环境泛化和依赖固定离散动作空间。为了解决这些挑战，我们提出了适用于无人驾驶飞行器（UAV）的视觉-语言飞行动作框架（VLFly），该框架能够执行语言引导的飞行。VLFly 不需要定位或主动测距传感器，仅通过机载单目相机拍摄的主观观测数据输出连续速度命令。VLFly 集成了三个模块：一个基于大型语言模型（LLM）的指令编码器，将高层次语言重新格式化为结构化提示；一个由视觉-语言模型（VLM）驱动的目标检索器，通过视觉-语言相似性将这些提示匹配到目标图像；以及一个航点规划器，生成用于实时UAV控制的可执行轨迹。VLFly 在多种模拟环境中进行了评估，无需额外微调，并且始终优于所有基线。此外，在室内和室外环境下的直接和间接指令下的真实世界VLN任务表明，VLFly 能够实现鲁棒性的开放词汇目标理解以及泛化的导航能力，即使在出现抽象语言输入的情况下也是如此。 

---
# BNMusic: Blending Environmental Noises into Personalized Music 

**Title (ZH)**: BNMusic: 将环境声音融合到个性化音乐中 

**Authors**: Chi Zuo, Martin B. Møller, Pablo Martínez-Nuevo, Huayang Huang, Yu Wu, Ye Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10754)  

**Abstract**: While being disturbed by environmental noises, the acoustic masking technique is a conventional way to reduce the annoyance in audio engineering that seeks to cover up the noises with other dominant yet less intrusive sounds. However, misalignment between the dominant sound and the noise-such as mismatched downbeats-often requires an excessive volume increase to achieve effective masking. Motivated by recent advances in cross-modal generation, in this work, we introduce an alternative method to acoustic masking, aiming to reduce the noticeability of environmental noises by blending them into personalized music generated based on user-provided text prompts. Following the paradigm of music generation using mel-spectrogram representations, we propose a Blending Noises into Personalized Music (BNMusic) framework with two key stages. The first stage synthesizes a complete piece of music in a mel-spectrogram representation that encapsulates the musical essence of the noise. In the second stage, we adaptively amplify the generated music segment to further reduce noise perception and enhance the blending effectiveness, while preserving auditory quality. Our experiments with comprehensive evaluations on MusicBench, EPIC-SOUNDS, and ESC-50 demonstrate the effectiveness of our framework, highlighting the ability to blend environmental noise with rhythmically aligned, adaptively amplified, and enjoyable music segments, minimizing the noticeability of the noise, thereby improving overall acoustic experiences. 

**Abstract (ZH)**: 利用个性化音乐融合环境噪音的掩蔽方法 

---
# TED-LaST: Towards Robust Backdoor Defense Against Adaptive Attacks 

**Title (ZH)**: TED-LaST: 面向适应性攻击的鲁棒后门防御方法 

**Authors**: Xiaoxing Mo, Yuxuan Cheng, Nan Sun, Leo Yu Zhang, Wei Luo, Shang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10722)  

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to backdoor attacks, where attackers implant hidden triggers during training to maliciously control model behavior. Topological Evolution Dynamics (TED) has recently emerged as a powerful tool for detecting backdoor attacks in DNNs. However, TED can be vulnerable to backdoor attacks that adaptively distort topological representation distributions across network layers. To address this limitation, we propose TED-LaST (Topological Evolution Dynamics against Laundry, Slow release, and Target mapping attack strategies), a novel defense strategy that enhances TED's robustness against adaptive attacks. TED-LaST introduces two key innovations: label-supervised dynamics tracking and adaptive layer emphasis. These enhancements enable the identification of stealthy threats that evade traditional TED-based defenses, even in cases of inseparability in topological space and subtle topological perturbations. We review and classify data poisoning tricks in state-of-the-art adaptive attacks and propose enhanced adaptive attack with target mapping, which can dynamically shift malicious tasks and fully leverage the stealthiness that adaptive attacks possess. Our comprehensive experiments on multiple datasets (CIFAR-10, GTSRB, and ImageNet100) and model architectures (ResNet20, ResNet101) show that TED-LaST effectively counteracts sophisticated backdoors like Adap-Blend, Adapt-Patch, and the proposed enhanced adaptive attack. TED-LaST sets a new benchmark for robust backdoor detection, substantially enhancing DNN security against evolving threats. 

**Abstract (ZH)**: 基于拓扑演化动力学的防洗牌缓释放标 targeted 防护策略：应对动态后门攻击 

---
# PREMISE: Scalable and Strategic Prompt Optimization for Efficient Mathematical Reasoning in Large Models 

**Title (ZH)**: 前提：面向大规模模型的可扩展与策略性提示优化以实现高效的数学推理 

**Authors**: Ye Yu, Yaoning Yu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10716)  

**Abstract**: Large reasoning models (LRMs) such as Claude 3.7 Sonnet and OpenAI o1 achieve strong performance on mathematical benchmarks using lengthy chain-of-thought (CoT) reasoning, but the resulting traces are often unnecessarily verbose. This inflates token usage and cost, limiting deployment in latency-sensitive or API-constrained settings. We introduce PREMISE (PRompt-based Efficient Mathematical Inference with Strategic Evaluation), a prompt-only framework that reduces reasoning overhead without modifying model weights. PREMISE combines trace-level diagnostics with gradient-inspired prompt optimization to minimize redundant computation while preserving answer accuracy. The approach jointly optimizes brevity and correctness through a multi-objective textual search that balances token length and answer validity. Unlike prior work, PREMISE runs in a single-pass black-box interface, so it can be applied directly to commercial LLMs. On GSM8K, SVAMP, and Math500 we match or exceed baseline accuracy ($96\%\rightarrow96\%$ with Claude, $91\%\rightarrow92\%$ with Gemini) while reducing reasoning tokens by up to $87.5\%$ and cutting dollar cost by $69$--$82\%$. These results show that prompt-level optimization is a practical and scalable path to efficient LRM inference without compromising reasoning quality. 

**Abstract (ZH)**: 基于提示的高效数学推理与战略评估（PREMISE） 

---
# Deep Learning-based Multi Project InP Wafer Simulation for Unsupervised Surface Defect Detection 

**Title (ZH)**: 基于深度学习的多项目InP晶圆仿真用于无监督表面缺陷检测 

**Authors**: Emílio Dolgener Cantú, Rolf Klemens Wittmann, Oliver Abdeen, Patrick Wagner, Wojciech Samek, Moritz Baier, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10713)  

**Abstract**: Quality management in semiconductor manufacturing often relies on template matching with known golden standards. For Indium-Phosphide (InP) multi-project wafer manufacturing, low production scale and high design variability lead to such golden standards being typically unavailable. Defect detection, in turn, is manual and labor-intensive. This work addresses this challenge by proposing a methodology to generate a synthetic golden standard using Deep Neural Networks, trained to simulate photo-realistic InP wafer images from CAD data. We evaluate various training objectives and assess the quality of the simulated images on both synthetic data and InP wafer photographs. Our deep-learning-based method outperforms a baseline decision-tree-based approach, enabling the use of a 'simulated golden die' from CAD plans in any user-defined region of a wafer for more efficient defect detection. We apply our method to a template matching procedure, to demonstrate its practical utility in surface defect detection. 

**Abstract (ZH)**: 半导体制造中的质量管理通常依赖于与已知黄金标准进行模板匹配。对于InP多项目晶圆制造，由于低生产规模和高设计变异性，通常缺乏黄金标准。因此，缺陷检测需要手工进行且劳动密集型。本工作通过提出一种使用深度神经网络生成合成黄金标准的方法来应对这一挑战，该网络训练以从CAD数据中模拟出光现实的InP晶圆图像。我们评估了各种训练目标，并在合成数据和InP晶圆照片上评估模拟图像的质量。基于深度学习的方法优于基准的决策树方法，使得在晶圆的任何用户定义区域内使用“从CAD计划生成的模拟黄金晶圆”进行缺陷检测更加高效。我们应用该方法到模板匹配过程，以展示其在表面缺陷检测中的实际应用价值。 

---
# Continual Hyperbolic Learning of Instances and Classes 

**Title (ZH)**: 持续双曲学习实例和类別 

**Authors**: Melika Ayoughi, Mina Ghadimi Atigh, Mohammad Mahdi Derakhshani, Cees G. M. Snoek, Pascal Mettes, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2506.10710)  

**Abstract**: Continual learning has traditionally focused on classifying either instances or classes, but real-world applications, such as robotics and self-driving cars, require models to handle both simultaneously. To mirror real-life scenarios, we introduce the task of continual learning of instances and classes, at the same time. This task challenges models to adapt to multiple levels of granularity over time, which requires balancing fine-grained instance recognition with coarse-grained class generalization. In this paper, we identify that classes and instances naturally form a hierarchical structure. To model these hierarchical relationships, we propose HyperCLIC, a continual learning algorithm that leverages hyperbolic space, which is uniquely suited for hierarchical data due to its ability to represent tree-like structures with low distortion and compact embeddings. Our framework incorporates hyperbolic classification and distillation objectives, enabling the continual embedding of hierarchical relations. To evaluate performance across multiple granularities, we introduce continual hierarchical metrics. We validate our approach on EgoObjects, the only dataset that captures the complexity of hierarchical object recognition in dynamic real-world environments. Empirical results show that HyperCLIC operates effectively at multiple granularities with improved hierarchical generalization. 

**Abstract (ZH)**: 持续学习传统上侧重于分类实例或类，但机器人技术和自动驾驶汽车等实际应用要求模型同时处理实例和类。为模拟现实场景，我们引入了实例和类的同时持续学习任务。此任务要求模型随时间适应不同的粒度层次，需要在细粒度实例识别与粗粒度类泛化之间取得平衡。在本文中，我们发现类和实例自然形成了一个层次结构。为了建模这些层次关系，我们提出了一种名为HyperCLIC的持续学习算法，该算法利用双曲空间，因其能够以低失真的方式表示树状结构并提供紧凑的嵌入而特别适合层次数据。我们的框架结合了双曲分类和蒸馏目标，能够持续嵌入层次关系。为了评估多粒度下的性能，我们引入了持续层次度量。我们通过EgoObjects数据集验证了该方法，这是唯一一个捕捉动态现实环境中超级对象识别复杂性的数据集。经验结果表明，HyperCLIC在多个粒度层次上有效运行，并具有改进的层次泛化能力。 

---
# ConTextTab: A Semantics-Aware Tabular In-Context Learner 

**Title (ZH)**: Semantics-Aware Tabular In-Context Learner: ConTextTab 

**Authors**: Marco Spinaci, Marek Polewczyk, Maximilian Schambach, Sam Thelin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10707)  

**Abstract**: Tabular in-context learning (ICL) has recently achieved state-of-the-art (SOTA) performance on several tabular prediction tasks. Previously restricted to classification problems on small tables, recent advances such as TabPFN and TabICL have extended its use to larger datasets. While being architecturally efficient and well-adapted to tabular data structures, current table-native ICL architectures, being trained exclusively on synthetic data, do not fully leverage the rich semantics and world knowledge contained in real-world tabular data. On another end of this spectrum, tabular ICL models based on pretrained large language models such as TabuLa-8B integrate deep semantic understanding and world knowledge but are only able to make use of a small amount of context due to inherent architectural limitations. With the aim to combine the best of both these worlds, we introduce ConTextTab, integrating semantic understanding and alignment into a table-native ICL framework. By employing specialized embeddings for different data modalities and by training on large-scale real-world tabular data, our model is competitive with SOTA across a broad set of benchmarks while setting a new standard on the semantically rich CARTE benchmark. 

**Abstract (ZH)**: 表格内模态学习（ICL）近年来已在多个表格预测任务中取得了最先进的性能。虽然当前的表本源ICL架构在合成数据上训练，能够在架构上高效且很好地适应表格数据结构，但未能充分利用真实世界表格数据中丰富的语义和世界知识。另一方面，基于预训练大规模语言模型（如TabuLa-8B）的表格ICL模型虽然能够整合深入的语义理解和世界知识，但由于固有的架构限制，只能利用少量的上下文信息。为了兼顾两者的优势，我们引入了ConTextTab，将其语义理解和对齐整合到一个表本源ICL框架中。通过使用不同数据模态的专业嵌入，并在大规模真实世界表格数据上进行训练，我们的模型在多个基准测试中与最先进的技术竞争，特别是在语义丰富的CARTE基准测试中树立了新的标准。 

---
# Formalising Software Requirements using Large Language Models 

**Title (ZH)**: 使用大型语言模型形式化软件需求 

**Authors**: Arshad Beg, Diarmuid O'Donoghue, Rosemary Monahan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10704)  

**Abstract**: This paper is a brief introduction to our recently initiated project named VERIFAI: Traceability and verification of natural language requirements. The project addresses the challenges in the traceability and verification of formal specifications through providing support for the automatic generation of the formal specifications and the traceability of the requirements from the initial software design stage through the systems implementation and verification. Approaches explored in this project include Natural Language Processing, use of ontologies to describe the software system domain, reuse of existing software artefacts from similar systems (i.e. through similarity based reuse) and large language models to identify and declare the specifications as well as use of artificial intelligence to guide the process. 

**Abstract (ZH)**: 本论文是对最近启动的VERIFAI项目的一个简要介绍：自然语言需求的可追溯性和验证。该项目通过提供自动生成形式化规范和支持从初始软件设计阶段到系统实现和验证的可追溯性来解决形式化规范的可追溯性和验证挑战。项目中探索的方法包括自然语言处理、使用Ontology描述软件系统领域、基于相似性重用现有软件 artefacts以及使用大规模语言模型标识和声明规范，并利用人工智能指导这一过程。 

---
# Large Language Models for Detection of Life-Threatening Texts 

**Title (ZH)**: 大型语言模型在危及生命文本检测中的应用 

**Authors**: Thanh Thi Nguyen, Campbell Wilson, Janis Dalins  

**Link**: [PDF](https://arxiv.org/pdf/2506.10687)  

**Abstract**: Detecting life-threatening language is essential for safeguarding individuals in distress, promoting mental health and well-being, and preventing potential harm and loss of life. This paper presents an effective approach to identifying life-threatening texts using large language models (LLMs) and compares them with traditional methods such as bag of words, word embedding, topic modeling, and Bidirectional Encoder Representations from Transformers. We fine-tune three open-source LLMs including Gemma, Mistral, and Llama-2 using their 7B parameter variants on different datasets, which are constructed with class balance, imbalance, and extreme imbalance scenarios. Experimental results demonstrate a strong performance of LLMs against traditional methods. More specifically, Mistral and Llama-2 models are top performers in both balanced and imbalanced data scenarios while Gemma is slightly behind. We employ the upsampling technique to deal with the imbalanced data scenarios and demonstrate that while this method benefits traditional approaches, it does not have as much impact on LLMs. This study demonstrates a great potential of LLMs for real-world life-threatening language detection problems. 

**Abstract (ZH)**: 检测危及生命的语言对于保护陷入困境的个体、促进心理健康和福祉以及预防潜在危害和生命损失至关重要。本文提出了一种有效的方法，使用大型语言模型（LLMs）来识别危及生命的文字，并将其与传统的词袋模型、词嵌入、主题建模和双向Transformer表示等方法进行了比较。我们使用不同数据集对三个开源LLM（Gemma、Mistral和Llama-2）的7B参数变体进行了微调，这些数据集构建了类别平衡、不平衡和极端不平衡的场景。实验结果表明，LLMs在传统方法面前表现出色。具体而言，在平衡和不平衡的数据场景中，Mistral和Llama-2模型表现最佳，而Gemma稍逊一筹。我们采用上采样技术来应对不平衡数据场景，并证明虽然这种方法对传统方法有益，但对LLMs的影响较小。本研究展示了LLMs在实际危及生命语言检测问题中的巨大潜力。 

---
# Saturation Self-Organizing Map 

**Title (ZH)**: 饱和自组织地图 

**Authors**: Igor Urbanik, Paweł Gajewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.10680)  

**Abstract**: Continual learning poses a fundamental challenge for neural systems, which often suffer from catastrophic forgetting when exposed to sequential tasks. Self-Organizing Maps (SOMs), despite their interpretability and efficiency, are not immune to this issue. In this paper, we introduce Saturation Self-Organizing Maps (SatSOM)-an extension of SOMs designed to improve knowledge retention in continual learning scenarios. SatSOM incorporates a novel saturation mechanism that gradually reduces the learning rate and neighborhood radius of neurons as they accumulate information. This effectively freezes well-trained neurons and redirects learning to underutilized areas of the map. 

**Abstract (ZH)**: 持续学习对神经系统构成了根本性的挑战，当神经网络暴露于顺序任务时常会出现灾难性遗忘。自组织映射(SOM)尽管具备可解释性和高效性，但也不免受到这一问题的影响。本文引入了饱和自组织映射(SatSOM)——一种为持续学习场景设计、旨在提高知识保留能力的SOM扩展模型。SatSOM集成了一种新颖的饱和机制，该机制逐渐降低累积信息的神经元的学习率和邻域半径，从而有效冻结训练良好的神经元，并将学习过程转向映射中利用率较低的区域。 

---
# PiPViT: Patch-based Visual Interpretable Prototypes for Retinal Image Analysis 

**Title (ZH)**: PiPViT：基于 patches 的视觉可解释原型.Retinal 图像分析 

**Authors**: Marzieh Oghbaie, Teresa Araújoa, Hrvoje Bogunović  

**Link**: [PDF](https://arxiv.org/pdf/2506.10669)  

**Abstract**: Background and Objective: Prototype-based methods improve interpretability by learning fine-grained part-prototypes; however, their visualization in the input pixel space is not always consistent with human-understandable biomarkers. In addition, well-known prototype-based approaches typically learn extremely granular prototypes that are less interpretable in medical imaging, where both the presence and extent of biomarkers and lesions are critical.
Methods: To address these challenges, we propose PiPViT (Patch-based Visual Interpretable Prototypes), an inherently interpretable prototypical model for image recognition. Leveraging a vision transformer (ViT), PiPViT captures long-range dependencies among patches to learn robust, human-interpretable prototypes that approximate lesion extent only using image-level labels. Additionally, PiPViT benefits from contrastive learning and multi-resolution input processing, which enables effective localization of biomarkers across scales.
Results: We evaluated PiPViT on retinal OCT image classification across four datasets, where it achieved competitive quantitative performance compared to state-of-the-art methods while delivering more meaningful explanations. Moreover, quantitative evaluation on a hold-out test set confirms that the learned prototypes are semantically and clinically relevant. We believe PiPViT can transparently explain its decisions and assist clinicians in understanding diagnostic outcomes. Github page: this https URL 

**Abstract (ZH)**: 背景与目标：基于原型的方法通过学习细粒度的部分原型提高了可解释性；然而，它们在输入像素空间中的可视化有时与人类可理解的生物标志物不一致。此外，著名的基于原型的方法通常学习极为细粒度的原型，在医学成像中可解释性较差，而医学成像中生物标志物和病变的存在及其范围至关重要。
方法：为了解决这些挑战，我们提出了一种名为PiPViT（patches-based visual interpretable prototypes）的模型，这是一种固有的可解释原型模型，用于图像识别。通过利用视觉变换器（ViT），PiPViT捕捉.patch之间的长程依赖关系，利用图像级标签学习鲁棒且人类可解释的原型，仅通过图像量级标签来近似病变范围。此外，PiPViT还受益于对比学习和多分辨率输入处理，这使其能够在不同尺度上有效地定位生物标志物。
结果：我们在四个数据集上对PiPViT进行了视网膜OCT图像分类的评估，其相较于最先进的方法在定量性能上具有竞争力，同时提供了更具有意义的解释。此外，对保留测试集的定量评估表明，学习到的原型具有语义和临床相关性。我们相信PiPViT能够透明地解释其决策，并辅助临床医生理解诊断结果。GitHub页面：this https URL 

---
# Contrastive Matrix Completion with Denoising and Augmented Graph Views for Robust Recommendation 

**Title (ZH)**: 降噪与增强图视图相结合的对比矩阵补全稳健推荐 

**Authors**: Narges Nemati, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2506.10658)  

**Abstract**: Matrix completion is a widely adopted framework in recommender systems, as predicting the missing entries in the user-item rating matrix enables a comprehensive understanding of user preferences. However, current graph neural network (GNN)-based approaches are highly sensitive to noisy or irrelevant edges--due to their inherent message-passing mechanisms--and are prone to overfitting, which limits their generalizability. To overcome these challenges, we propose a novel method called Matrix Completion using Contrastive Learning (MCCL). Our approach begins by extracting local neighborhood subgraphs for each interaction and subsequently generates two distinct graph representations. The first representation emphasizes denoising by integrating GNN layers with an attention mechanism, while the second is obtained via a graph variational autoencoder that aligns the feature distribution with a standard prior. A mutual learning loss function is employed during training to gradually harmonize these representations, enabling the model to capture common patterns and significantly enhance its generalizability. Extensive experiments on several real-world datasets demonstrate that our approach not only improves the numerical accuracy of the predicted scores--achieving up to a 0.8% improvement in RMSE--but also produces superior rankings with improvements of up to 36% in ranking metrics. 

**Abstract (ZH)**: 矩阵补全是一种广泛采用的推荐系统框架，通过预测用户-项评分矩阵中的缺失条目，可以全面理解用户偏好。然而，当前基于图神经网络（GNN）的方法对其固有的消息传递机制极为敏感，容易受到噪声或无关边的影响，并且容易过拟合，这限制了它们的泛化能力。为克服这些挑战，我们提出了一种新颖的方法，称为对比学习下的矩阵补全（MCCL）。该方法首先为每种交互提取局部邻域子图，然后生成两种不同的图表示。第一种表示通过结合GNN层和注意机制强调去噪，第二种表示通过图变分自编码器获得，该编码器将特征分布与标准先验对齐。在训练过程中采用互学习损失函数逐步协调这两种表示，使模型能够捕捉共同模式，显著增强其泛化能力。在多个真实世界数据集上的 extensive 实验表明，我们的方法不仅提高了预测分数的数值准确性——在均方根误差（RMSE）上提高了多达 0.8%——而且在排名指标上也表现出更优的效果，排名改进幅度高达 36%。 

---
# Data Shifts Hurt CoT: A Theoretical Study 

**Title (ZH)**: 数据偏移损害共注意力：一项理论研究 

**Authors**: Lang Yin, Debangshu Banerjee, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10647)  

**Abstract**: Chain of Thought (CoT) has been applied to various large language models (LLMs) and proven to be effective in improving the quality of outputs. In recent studies, transformers are proven to have absolute upper bounds in terms of expressive power, and consequently, they cannot solve many computationally difficult problems. However, empowered by CoT, transformers are proven to be able to solve some difficult problems effectively, such as the $k$-parity problem. Nevertheless, those works rely on two imperative assumptions: (1) identical training and testing distribution, and (2) corruption-free training data with correct reasoning steps. However, in the real world, these assumptions do not always hold. Although the risks of data shifts have caught attention, our work is the first to rigorously study the exact harm caused by such shifts to the best of our knowledge. Focusing on the $k$-parity problem, in this work we investigate the joint impact of two types of data shifts: the distribution shifts and data poisoning, on the quality of trained models obtained by a well-established CoT decomposition. In addition to revealing a surprising phenomenon that CoT leads to worse performance on learning parity than directly generating the prediction, our technical results also give a rigorous and comprehensive explanation of the mechanistic reasons of such impact. 

**Abstract (ZH)**: Chain of Thought (CoT)在大型语言模型中的应用及其数据偏移影响的研究 

---
# Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models 

**Title (ZH)**: 对称流匹配：基于评分生成模型的统一图像生成、分割和分类 

**Authors**: Francisco Caetano, Christiaan Viviers, Peter H.N. De With, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10634)  

**Abstract**: Flow Matching has emerged as a powerful framework for learning continuous transformations between distributions, enabling high-fidelity generative modeling. This work introduces Symmetrical Flow Matching (SymmFlow), a new formulation that unifies semantic segmentation, classification, and image generation within a single model. Using a symmetric learning objective, SymmFlow models forward and reverse transformations jointly, ensuring bi-directional consistency, while preserving sufficient entropy for generative diversity. A new training objective is introduced to explicitly retain semantic information across flows, featuring efficient sampling while preserving semantic structure, allowing for one-step segmentation and classification without iterative refinement. Unlike previous approaches that impose strict one-to-one mapping between masks and images, SymmFlow generalizes to flexible conditioning, supporting both pixel-level and image-level class labels. Experimental results on various benchmarks demonstrate that SymmFlow achieves state-of-the-art performance on semantic image synthesis, obtaining FID scores of 11.9 on CelebAMask-HQ and 7.0 on COCO-Stuff with only 25 inference steps. Additionally, it delivers competitive results on semantic segmentation and shows promising capabilities in classification tasks. The code will be publicly available. 

**Abstract (ZH)**: Symmetrical Flow Matching: Unifying Semantic Segmentation, Classification, and Image Generation with High-Fidelity Generative Modeling 

---
# Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs 

**Title (ZH)**: 时间序列预测作为一种推理：强化大型语言模型的慢思考方法 

**Authors**: Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, Jintao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10630)  

**Abstract**: To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations - including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow thinking capabilities and acquire strong time series reasoning skills. For this purpose, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets. 

**Abstract (ZH)**: 改进时间序列预测的LLM双阶段强化细调框架：Time-R1 

---
# Task Adaptation from Skills: Information Geometry, Disentanglement, and New Objectives for Unsupervised Reinforcement Learning 

**Title (ZH)**: 技能适配任务：信息几何、解缠绕及无监督强化学习的新目标 

**Authors**: Yucheng Yang, Tianyi Zhou, Qiang He, Lei Han, Mykola Pechenizkiy, Meng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10629)  

**Abstract**: Unsupervised reinforcement learning (URL) aims to learn general skills for unseen downstream tasks. Mutual Information Skill Learning (MISL) addresses URL by maximizing the mutual information between states and skills but lacks sufficient theoretical analysis, e.g., how well its learned skills can initialize a downstream task's policy. Our new theoretical analysis in this paper shows that the diversity and separability of learned skills are fundamentally critical to downstream task adaptation but MISL does not necessarily guarantee these properties. To complement MISL, we propose a novel disentanglement metric LSEPIN. Moreover, we build an information-geometric connection between LSEPIN and downstream task adaptation cost. For better geometric properties, we investigate a new strategy that replaces the KL divergence in information geometry with Wasserstein distance. We extend the geometric analysis to it, which leads to a novel skill-learning objective WSEP. It is theoretically justified to be helpful to downstream task adaptation and it is capable of discovering more initial policies for downstream tasks than MISL. We finally propose another Wasserstein distance-based algorithm PWSEP that can theoretically discover all optimal initial policies. 

**Abstract (ZH)**: 无监督强化学习中的互信息技能学习及其理论分析：一个新的几何视角 

---
# NeuralNexus at BEA 2025 Shared Task: Retrieval-Augmented Prompting for Mistake Identification in AI Tutors 

**Title (ZH)**: NeuralNexus在BEA 2025共享任务中的检索增强提示：AI辅导系统中的错误识别 

**Authors**: Numaan Naeem, Sarfraz Ahmad, Momina Ahsan, Hasan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10627)  

**Abstract**: This paper presents our system for Track 1: Mistake Identification in the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors. The task involves evaluating whether a tutor's response correctly identifies a mistake in a student's mathematical reasoning. We explore four approaches: (1) an ensemble of machine learning models over pooled token embeddings from multiple pretrained language models (LMs); (2) a frozen sentence-transformer using [CLS] embeddings with an MLP classifier; (3) a history-aware model with multi-head attention between token-level history and response embeddings; and (4) a retrieval-augmented few-shot prompting system with a large language model (LLM) i.e. GPT 4o. Our final system retrieves semantically similar examples, constructs structured prompts, and uses schema-guided output parsing to produce interpretable predictions. It outperforms all baselines, demonstrating the effectiveness of combining example-driven prompting with LLM reasoning for pedagogical feedback assessment. Our code is available at this https URL. 

**Abstract (ZH)**: 本文介绍了我们针对BEA 2025共享任务中的Track 1——AI辅导系统教学能力评估中的错误识别系统的方案。该任务涉及评估辅导系统响应是否正确识别了学生数学推理中的错误。我们探索了四种方法：（1）多种预训练语言模型（LMs）池化词嵌入的机器学习模型集成；（2）固定句向量变换器结合CLS嵌入及MLP分类器；（3）具有词汇和响应嵌入之间多头注意力的历史感知模型；以及（4）结合大型语言模型（LLM）如GPT 4的检索增强少量示例提示系统。最终系统检索语义相似的例子，构建结构化的提示，并使用模式导向的输出解析生成可解释的预测，其性能优于所有基线，展示了结合示例驱动的提示与LLM推理在教学反馈评估中的有效性。代码可在以下链接获取：this https URL。 

---
# SDialog: A Python Toolkit for Synthetic Dialogue Generation and Analysis 

**Title (ZH)**: SDialog：一个用于合成对话生成与分析的Python工具包 

**Authors**: Sergio Burdisso, Esaú Villatoro-Tello, Petr Motlicek  

**Link**: [PDF](https://arxiv.org/pdf/2506.10622)  

**Abstract**: The advancement of conversational AI systems relies on the availability of high-quality, flexible, and reproducible synthetic dialogues for training, evaluation, and benchmarking. SDialog is a modular, extensible Python toolkit designed to address the challenges of synthetic dialogue generation and analysis. By leveraging instruction-tuned Large Language Models (LLMs), SDialog provides abstractions for personas, orchestration, and scenario management, enabling the creation of realistic, diverse, and controllable conversational data for research and development. SDialog supports workflows such as multi-agent simulation and scenario-driven generation, and represents a step forward in the standardization of tools and frameworks for synthetic data generation, a crucial advancement for ensuring reproducibility in today's fast-evolving research landscape. 

**Abstract (ZH)**: 基于高质量、灵活且可重现的合成对话的先进性，对话AI系统的发展依赖于其在训练、评估和基准测试中的可用性。SDialog 是一个模块化且可扩展的 Python 工具包，旨在解决合成对话生成和分析的挑战。借助指令调优的大语言模型（LLMs），SDialog 提供了人设、编排和情景管理的抽象层，从而能够创建真实、多样且可控的对话数据，用于研究和开发。SDialog 支持多Agent模拟和情景驱动生成等流程，并代表了合成数据生成工具和框架标准化的重要进展，这对于确保当前快速发展的研究领域中的可重现性至关重要。 

---
# Deep Learning-Based Digitization of Overlapping ECG Images with Open-Source Python Code 

**Title (ZH)**: 基于深度学习的重叠心电图图像数字化开源Python代码实现 

**Authors**: Reza Karbasi, Masoud Rahimi, Abdol-Hossein Vahabie, Hadi Moradi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10617)  

**Abstract**: This paper addresses the persistent challenge of accurately digitizing paper-based electrocardiogram (ECG) recordings, with a particular focus on robustly handling single leads compromised by signal overlaps-a common yet under-addressed issue in existing methodologies. We propose a two-stage pipeline designed to overcome this limitation. The first stage employs a U-Net based segmentation network, trained on a dataset enriched with overlapping signals and fortified with custom data augmentations, to accurately isolate the primary ECG trace. The subsequent stage converts this refined binary mask into a time-series signal using established digitization techniques, enhanced by an adaptive grid detection module for improved versatility across different ECG formats and scales. Our experimental results demonstrate the efficacy of our approach. The U-Net architecture achieves an IoU of 0.87 for the fine-grained segmentation task. Crucially, our proposed digitization method yields superior performance compared to a well-established baseline technique across both non-overlapping and challenging overlapping ECG samples. For non-overlapping signals, our method achieved a Mean Squared Error (MSE) of 0.0010 and a Pearson Correlation Coefficient (rho) of 0.9644, compared to 0.0015 and 0.9366, respectively, for the baseline. On samples with signal overlap, our method achieved an MSE of 0.0029 and a rho of 0.9641, significantly improving upon the baseline's 0.0178 and 0.8676. This work demonstrates an effective strategy to significantly enhance digitization accuracy, especially in the presence of signal overlaps, thereby laying a strong foundation for the reliable conversion of analog ECG records into analyzable digital data for contemporary research and clinical applications. The implementation is publicly available at this GitHub repository: this https URL. 

**Abstract (ZH)**: 本文解决了一直存在的准确数字化基于纸张的心电图（ECG）记录的持续挑战，特别关注在现有方法中常被忽视但又十分常见的信号重叠导致单通道数据受损问题。我们提出了一种两阶段的流水线设计来克服这一限制。第一阶段采用一种基于U-Net的分割网络，该网络通过增强带有重叠信号的数据集并结合自定义数据增强进行训练，以准确地分离主要的ECG轨迹。第二阶段采用现有的数字化技术将这个精炼的二进制掩模转化为时间序列信号，并通过自适应网格检测模块提高不同ECG格式和规模下的灵活性。我们的实验结果证明了该方法的有效性。U-Net架构在细粒度分割任务中达到了0.87的IoU。至关重要的是，我们提出的数据数字化方法在非重叠信号和具有挑战性的重叠信号样本上均优于现有的基准技术。对于非重叠信号，该方法的均方误差（MSE）为0.0010，皮尔森相关系数（rho）为0.9644，而基准技术分别为0.0015和0.9366。在具有信号重叠的样本上，该方法的MSE为0.0029，rho为0.9641，显著优于基准技术的0.0178和0.8676。本文展示了在信号重叠存在的情况下显著提高数字化准确性的有效策略，为将模拟ECG记录可靠地转换为可分析的数字数据以供当前研究和临床应用奠定了坚实的基础。该实现已公开发布在 GitHub 仓库中：this https URL。 

---
# TexTailor: Customized Text-aligned Texturing via Effective Resampling 

**Title (ZH)**: TexTailor: 通过有效的重采样实现自定义文本对齐纹理化 

**Authors**: Suin Lee, Dae-Shik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10612)  

**Abstract**: We present TexTailor, a novel method for generating consistent object textures from textual descriptions. Existing text-to-texture synthesis approaches utilize depth-aware diffusion models to progressively generate images and synthesize textures across predefined multiple viewpoints. However, these approaches lead to a gradual shift in texture properties across viewpoints due to (1) insufficient integration of previously synthesized textures at each viewpoint during the diffusion process and (2) the autoregressive nature of the texture synthesis process. Moreover, the predefined selection of camera positions, which does not account for the object's geometry, limits the effective use of texture information synthesized from different viewpoints, ultimately degrading overall texture consistency. In TexTailor, we address these issues by (1) applying a resampling scheme that repeatedly integrates information from previously synthesized textures within the diffusion process, and (2) fine-tuning a depth-aware diffusion model on these resampled textures. During this process, we observed that using only a few training images restricts the model's original ability to generate high-fidelity images aligned with the conditioning, and therefore propose an performance preservation loss to mitigate this issue. Additionally, we improve the synthesis of view-consistent textures by adaptively adjusting camera positions based on the object's geometry. Experiments on a subset of the Objaverse dataset and the ShapeNet car dataset demonstrate that TexTailor outperforms state-of-the-art methods in synthesizing view-consistent textures. The source code for TexTailor is available at this https URL 

**Abstract (ZH)**: TexTailor：一种从文本描述生成一致对象纹理的新方法 

---
# SoK: Evaluating Jailbreak Guardrails for Large Language Models 

**Title (ZH)**: SoK: 评估大型语言模型的监禁门槛约束 

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10597)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）取得了显著进展，但其部署暴露了关键漏洞，特别是绕过安全机制的越界攻击。边界措施——外部防护机制，用于监控和控制LLM交互——已成为一种有前景的解决方案。然而，当前的LLM边界措施 landscape 是碎片化的，缺乏统一的分类体系和全面的评估框架。在本文综述性论文（SoK）中，我们首次对LLM越界边界措施进行了全面分析。我们提出了一种新颖的多维度分类法，按照六个关键维度对边界措施进行分类，并引入了安全-效率-实用性评估框架以评估它们的实际效果。通过广泛的分析和实验，我们识别了现有边界措施的优势和局限性，探讨了它们在不同类型攻击下的普适性，并提供了优化防御组合的见解。我们的工作为未来的研究和开发提供了一个结构化的基础，旨在指导稳健的LLM边界措施的原理发展和部署。相关代码可在以下链接找到：this https URL。 

---
# Size-adaptive Hypothesis Testing for Fairness 

**Title (ZH)**: 自适应样本大小公平性假设检验 

**Authors**: Antonio Ferrara, Francesco Cozzi, Alan Perotti, André Panisson, Francesco Bonchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10586)  

**Abstract**: Determining whether an algorithmic decision-making system discriminates against a specific demographic typically involves comparing a single point estimate of a fairness metric against a predefined threshold. This practice is statistically brittle: it ignores sampling error and treats small demographic subgroups the same as large ones. The problem intensifies in intersectional analyses, where multiple sensitive attributes are considered jointly, giving rise to a larger number of smaller groups. As these groups become more granular, the data representing them becomes too sparse for reliable estimation, and fairness metrics yield excessively wide confidence intervals, precluding meaningful conclusions about potential unfair treatments.
In this paper, we introduce a unified, size-adaptive, hypothesis-testing framework that turns fairness assessment into an evidence-based statistical decision. Our contribution is twofold. (i) For sufficiently large subgroups, we prove a Central-Limit result for the statistical parity difference, leading to analytic confidence intervals and a Wald test whose type-I (false positive) error is guaranteed at level $\alpha$. (ii) For the long tail of small intersectional groups, we derive a fully Bayesian Dirichlet-multinomial estimator; Monte-Carlo credible intervals are calibrated for any sample size and naturally converge to Wald intervals as more data becomes available. We validate our approach empirically on benchmark datasets, demonstrating how our tests provide interpretable, statistically rigorous decisions under varying degrees of data availability and intersectionality. 

**Abstract (ZH)**: 确定算法决策系统是否针对特定 demographic 进行歧视通常涉及将公平性指标的单点估计与预定义的阈值进行比较。这一做法在统计上是脆弱的：它忽视了抽样误差，并将小的 demographic 子群体与大的子群体同等对待。在考虑多个敏感属性的交叉分析中，这一问题更加严重，产生了更多的小群体。随着这些群体变得越来越细分，代表它们的数据变得过于稀疏，无法进行可靠的估计，公平性指标导出了过宽的置信区间，阻碍了对潜在不公平待遇的有意义结论。本文介绍了一个统一体积自适应假设检验框架，将其公平性评估转化为基于证据的统计决策。我们的贡献包括：（i）对于足够大的子群体，我们证明了统计均等差的中心极限定理，从而得到分析置信区间和类型-I错误（虚假阳性）保证在水平$\alpha$的沃尔德检验；（ii）对于小的交叉子群体的长尾部分，我们推导出一个完全贝叶斯狄利克雷-多项式估计器；蒙特卡洛可信区间根据样本大小进行了校准，并随着可用数据的增加自然收敛到沃尔德区间。我们在基准数据集上进行了实证验证，展示了我们的方法如何在不同数据可用性和交叉性程度下提供可解释且统计上严谨的决策。 

---
# DreamActor-H1: High-Fidelity Human-Product Demonstration Video Generation via Motion-designed Diffusion Transformers 

**Title (ZH)**: DreamActor-H1: 基于运动设计的扩散变换器高保真人-产品演示视频生成 

**Authors**: Lizhen Wang, Zhurong Xia, Tianshu Hu, Pengrui Wang, Pengfei Wang, Zerong Zheng, Ming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.10568)  

**Abstract**: In e-commerce and digital marketing, generating high-fidelity human-product demonstration videos is important for effective product presentation. However, most existing frameworks either fail to preserve the identities of both humans and products or lack an understanding of human-product spatial relationships, leading to unrealistic representations and unnatural interactions. To address these challenges, we propose a Diffusion Transformer (DiT)-based framework. Our method simultaneously preserves human identities and product-specific details, such as logos and textures, by injecting paired human-product reference information and utilizing an additional masked cross-attention mechanism. We employ a 3D body mesh template and product bounding boxes to provide precise motion guidance, enabling intuitive alignment of hand gestures with product placements. Additionally, structured text encoding is used to incorporate category-level semantics, enhancing 3D consistency during small rotational changes across frames. Trained on a hybrid dataset with extensive data augmentation strategies, our approach outperforms state-of-the-art techniques in maintaining the identity integrity of both humans and products and generating realistic demonstration motions. Project page: this https URL. 

**Abstract (ZH)**: 电子商务和数字营销中，生成高质量的人－产品示范视频对于有效的商品展示至关重要。然而，现有大多数框架要么无法同时保留人类和产品的身份，要么缺乏对人类－产品空间关系的理解，导致不真实的表示和不自然的交互。为了解决这些挑战，我们提出了一种基于扩散转换器（DiT）的框架。该方法通过注入成对的人－产品参考信息并利用附加的掩码交叉注意机制，同时保留人类身份和产品特定细节，如标志和纹理。我们采用3D身体网格模板和产品边界框提供精确的运动指导，使手部手势与产品放置直观对齐。此外，使用结构化文本编码增强类别级别的语义，从而在帧间小旋转变化中提高3D一致性。通过广泛数据增强策略训练的混合数据集，我们的方法在保持人类和产品身份完整性以及生成逼真示范动作方面优于现有技术。项目页面：this https URL。 

---
# Balancing Tails when Comparing Distributions: Comprehensive Equity Index (CEI) with Application to Bias Evaluation in Operational Face Biometrics 

**Title (ZH)**: 在比较分布时平衡尾部：综合公平指数（CEI）及其在操作面部生物识别偏差评估中的应用 

**Authors**: Imanol Solano, Julian Fierrez, Aythami Morales, Alejandro Peña, Ruben Tolosana, Francisco Zamora-Martinez, Javier San Agustin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10564)  

**Abstract**: Demographic bias in high-performance face recognition (FR) systems often eludes detection by existing metrics, especially with respect to subtle disparities in the tails of the score distribution. We introduce the Comprehensive Equity Index (CEI), a novel metric designed to address this limitation. CEI uniquely analyzes genuine and impostor score distributions separately, enabling a configurable focus on tail probabilities while also considering overall distribution shapes. Our extensive experiments (evaluating state-of-the-art FR systems, intentionally biased models, and diverse datasets) confirm CEI's superior ability to detect nuanced biases where previous methods fall short. Furthermore, we present CEI^A, an automated version of the metric that enhances objectivity and simplifies practical application. CEI provides a robust and sensitive tool for operational FR fairness assessment. The proposed methods have been developed particularly for bias evaluation in face biometrics but, in general, they are applicable for comparing statistical distributions in any problem where one is interested in analyzing the distribution tails. 

**Abstract (ZH)**: 全面公平指数（CEI）在高 PERFORMANCE 人脸鉴别系统中的性别偏差检测 

---
# From Images to Insights: Explainable Biodiversity Monitoring with Plain Language Habitat Explanations 

**Title (ZH)**: 从图像到洞察：以简洁语言解释的可解释生物多样性监测 

**Authors**: Yutong Zhou, Masahiro Ryo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10559)  

**Abstract**: Explaining why the species lives at a particular location is important for understanding ecological systems and conserving biodiversity. However, existing ecological workflows are fragmented and often inaccessible to non-specialists. We propose an end-to-end visual-to-causal framework that transforms a species image into interpretable causal insights about its habitat preference. The system integrates species recognition, global occurrence retrieval, pseudo-absence sampling, and climate data extraction. We then discover causal structures among environmental features and estimate their influence on species occurrence using modern causal inference methods. Finally, we generate statistically grounded, human-readable causal explanations from structured templates and large language models. We demonstrate the framework on a bee and a flower species and report early results as part of an ongoing project, showing the potential of the multimodal AI assistant backed up by a recommended ecological modeling practice for describing species habitat in human-understandable language. 

**Abstract (ZH)**: 解释物种为何生活在特定地点对于理解生态系统和保护生物多样性至关重要。然而，现有的生态工作流程往往是碎片化的，且往往难以为非专家所用。我们提出了一种端到端的视觉到因果关系框架，该框架能够将物种图像转换为关于其栖息地偏好的可解释因果洞察。该系统集成了物种识别、全球分布检索、伪缺失采样以及气候数据提取。然后，我们发现环境特征之间的因果结构，并使用现代因果推理方法估计这些特征对物种分布的影响。最后，我们通过结构化模板和大型语言模型生成统计上可靠的、易于理解的因果解释。我们在蜜蜂和一种花的物种上展示了该框架，并报告了作为正在进行项目一部分的初步结果，展示了支持生态建模实践的多模态AI助手在以人类可理解的语言描述物种栖息地方面的潜力。 

---
# StepProof: Step-by-step verification of natural language mathematical proofs 

**Title (ZH)**: StepProof: 自然语言数学证明的分步骤验证 

**Authors**: Xiaolin Hu, Qinghua Zhou, Bogdan Grechuk, Ivan Y. Tyukin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10558)  

**Abstract**: Interactive theorem provers (ITPs) are powerful tools for the formal verification of mathematical proofs down to the axiom level. However, their lack of a natural language interface remains a significant limitation. Recent advancements in large language models (LLMs) have enhanced the understanding of natural language inputs, paving the way for autoformalization - the process of translating natural language proofs into formal proofs that can be verified. Despite these advancements, existing autoformalization approaches are limited to verifying complete proofs and lack the capability for finer, sentence-level verification. To address this gap, we propose StepProof, a novel autoformalization method designed for granular, step-by-step verification. StepProof breaks down complete proofs into multiple verifiable subproofs, enabling sentence-level verification. Experimental results demonstrate that StepProof significantly improves proof success rates and efficiency compared to traditional methods. Additionally, we found that minor manual adjustments to the natural language proofs, tailoring them for step-level verification, further enhanced StepProof's performance in autoformalization. 

**Abstract (ZH)**: 交互定理证明器（ITPs）是用于数学证明形式验证的强大工具，直至公理层次。然而，它们缺乏自然语言界面仍然是一个重要限制。大型语言模型（LLMs）的最新进展增强了对自然语言输入的理解，为自动形式化铺平了道路——即将自然语言证明转换为可验证的形式证明。尽管如此，现有的自动形式化方法仅限于验证完整的证明，缺乏细粒度的、句子级别的验证能力。为解决这一问题，我们提出StepProof，这是一种专为细粒度、逐步验证设计的新型自动形式化方法。StepProof 将完整的证明分解为多个可验证的子证明，从而实现句子级别的验证。实验结果表明，与传统方法相比，StepProof 显著提高了证明的成功率和效率。此外，我们发现对自然语言证明进行适度的手动调整，以适应句子级别的验证，进一步提高了StepProof 在自动形式化中的性能。 

---
# CogStream: Context-guided Streaming Video Question Answering 

**Title (ZH)**: CogStream: 基于上下文的流式视频问答 

**Authors**: Zicheng Zhao, Kangyu Wang, Shijie Li, Rui Qian, Weiyao Lin, Huabin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10516)  

**Abstract**: Despite advancements in Video Large Language Models (Vid-LLMs) improving multimodal understanding, challenges persist in streaming video reasoning due to its reliance on contextual information. Existing paradigms feed all available historical contextual information into Vid-LLMs, resulting in a significant computational burden for visual data processing. Furthermore, the inclusion of irrelevant context distracts models from key details. This paper introduces a challenging task called Context-guided Streaming Video Reasoning (CogStream), which simulates real-world streaming video scenarios, requiring models to identify the most relevant historical contextual information to deduce answers for questions about the current stream. To support CogStream, we present a densely annotated dataset featuring extensive and hierarchical question-answer pairs, generated by a semi-automatic pipeline. Additionally, we present CogReasoner as a baseline model. It efficiently tackles this task by leveraging visual stream compression and historical dialogue retrieval. Extensive experiments prove the effectiveness of this method. Code will be released soon. 

**Abstract (ZH)**: 尽管视频大型语言模型(Vid-LLMs)在多模态理解方面取得了进展，但在流式视频推理中仍存在挑战，这归因于其对上下文信息的依赖。现有范式将所有可用的历史上下文信息输入到Vid-LLMs中，导致了视觉数据处理的巨大计算负担，并且不相关的上下文信息会分散模型对关键细节的注意力。本文引入了一个名为上下文引导的流式视频推理(CogStream)的具有挑战性的任务，该任务模拟了现实世界的流式视频场景，要求模型识别出与当前流相关的历史上下文信息，以推导出当前流相关问题的答案。为支持CogStream，我们提出了一种密集标注的数据集，该数据集包含了大量的层次化问答对，并由半自动管道生成。此外，我们还提出了CogReasoner作为一种基线模型，它通过利用视觉流压缩和历史对话检索有效地应对这一任务。广泛的实验证明了该方法的有效性。代码即将发布。 

---
# Reliable Reasoning Path: Distilling Effective Guidance for LLM Reasoning with Knowledge Graphs 

**Title (ZH)**: 可靠的推理路径：从知识图谱中提炼有效的指导用于LLM推理 

**Authors**: Yilin Xiao, Chuang Zhou, Qinggang Zhang, Bo Li, Qing Li, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10508)  

**Abstract**: Large language models (LLMs) often struggle with knowledge-intensive tasks due to a lack of background knowledge and a tendency to hallucinate. To address these limitations, integrating knowledge graphs (KGs) with LLMs has been intensively studied. Existing KG-enhanced LLMs focus on supplementary factual knowledge, but still struggle with solving complex questions. We argue that refining the relationships among facts and organizing them into a logically consistent reasoning path is equally important as factual knowledge itself. Despite their potential, extracting reliable reasoning paths from KGs poses the following challenges: the complexity of graph structures and the existence of multiple generated paths, making it difficult to distinguish between useful and redundant ones. To tackle these challenges, we propose the RRP framework to mine the knowledge graph, which combines the semantic strengths of LLMs with structural information obtained through relation embedding and bidirectional distribution learning. Additionally, we introduce a rethinking module that evaluates and refines reasoning paths according to their significance. Experimental results on two public datasets show that RRP achieves state-of-the-art performance compared to existing baseline methods. Moreover, RRP can be easily integrated into various LLMs to enhance their reasoning abilities in a plug-and-play manner. By generating high-quality reasoning paths tailored to specific questions, RRP distills effective guidance for LLM reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往在知识密集型任务上表现不佳，由于缺乏背景知识且倾向于虚构。为解决这些限制，将知识图谱（KGs）与LLMs集成的研究已经得到了广泛关注。现有的KG增强型LLMs重点在于补充事实知识，但仍难以解决复杂问题。我们 argue认为，精炼事实之间的关系并将它们组织成逻辑一致的推理路径与事实知识本身同样重要。尽管KG具有潜力，从KG中提取可靠推理路径仍面临以下挑战：图结构的复杂性以及存在多种生成路径，这使得区分有用和冗余路径变得困难。为应对这些挑战，我们提出了一种RRP框架，结合了LLMs的语义优势与通过关系嵌入和双向分布学习获得的结构信息。此外，我们引入了一个反思模块，根据路径的重要性对其进行评估和优化。在两个公开数据集上的实验结果表明，RRP在与现有基线方法相比时达到了最先进的性能。此外，RRP可以方便地集成到各种LLMs中，以插拔方式增强其推理能力。通过为特定问题生成高质量的推理路径，RRP提炼出有效的指导，以辅助LLMs的推理过程。 

---
# Beyond Single-User Dialogue: Assessing Multi-User Dialogue State Tracking Capabilities of Large Language Models 

**Title (ZH)**: 超越单一用户对话：评估大规模语言模型在多用户对话状态跟踪方面的能力 

**Authors**: Sangmin Song, Juhwan Choi, JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10504)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance in zero-shot dialogue state tracking (DST), reducing the need for task-specific training. However, conventional DST benchmarks primarily focus on structured user-agent conversations, failing to capture the complexities of real-world multi-user interactions. In this study, we assess the robustness of LLMs in multi-user DST while minimizing dataset construction costs. Inspired by recent advances in LLM-based data annotation, we extend an existing DST dataset by generating utterances of a second user based on speech act theory. Our methodology systematically incorporates a second user's utterances into conversations, enabling a controlled evaluation of LLMs in multi-user settings. Experimental results reveal a significant performance drop compared to single-user DST, highlighting the limitations of current LLMs in extracting and tracking dialogue states amidst multiple speakers. Our findings emphasize the need for future research to enhance LLMs for multi-user DST scenarios, paving the way for more realistic and robust DST models. 

**Abstract (ZH)**: 大型语言模型（LLMs）在零样本对话状态跟踪（DST）中表现出色，降低了特定任务训练的需求。然而，传统的DST基准主要关注结构化的用户-代理对话，未能捕捉到现实世界多用户交互的复杂性。在本研究中，我们评估了LLMs在多用户DST中的鲁棒性，同时尽量减少了数据集构建成本。受基于LLM的注释方法最新进展的启发，我们根据言语行为理论生成第二用户的言语，并扩展了现有的DST数据集。我们的方法系统地将第二用户的言语纳入对话中，从而在多用户环境中对LLMs进行受控评估。实验结果表明，与单用户DST相比，LLMs的性能显著下降，突显了当前LLMs在多说话人背景下提取和跟踪对话状态的局限性。我们的研究结果强调了未来研究需要增强LLMs以应对多用户DST场景的必要性，为更现实和鲁棒的DST模型铺平了道路。 

---
# Semantic Localization Guiding Segment Anything Model For Reference Remote Sensing Image Segmentation 

**Title (ZH)**: 基于语义定位的 Segment Anything 模型在参考遥感图像分割中的应用 

**Authors**: Shuyang Li, Shuang Wang, Zhuangzhuang Sun, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10503)  

**Abstract**: The Reference Remote Sensing Image Segmentation (RRSIS) task generates segmentation masks for specified objects in images based on textual descriptions, which has attracted widespread attention and research interest. Current RRSIS methods rely on multi-modal fusion backbones and semantic segmentation heads but face challenges like dense annotation requirements and complex scene interpretation. To address these issues, we propose a framework named \textit{prompt-generated semantic localization guiding Segment Anything Model}(PSLG-SAM), which decomposes the RRSIS task into two stages: coarse localization and fine segmentation. In coarse localization stage, a visual grounding network roughly locates the text-described object. In fine segmentation stage, the coordinates from the first stage guide the Segment Anything Model (SAM), enhanced by a clustering-based foreground point generator and a mask boundary iterative optimization strategy for precise segmentation. Notably, the second stage can be train-free, significantly reducing the annotation data burden for the RRSIS task. Additionally, decomposing the RRSIS task into two stages allows for focusing on specific region segmentation, avoiding interference from complex this http URL further contribute a high-quality, multi-category manually annotated dataset. Experimental validation on two datasets (RRSIS-D and RRSIS-M) demonstrates that PSLG-SAM achieves significant performance improvements and surpasses existing state-of-the-art this http URL code will be made publicly available. 

**Abstract (ZH)**: 基于提示生成的语义定位引导分割一切模型（PSLG-SAM）的任务框架 

---
# Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications 

**Title (ZH)**: 多-agent 大语言模型系统的设计与评估——原型及网络安全应用 

**Authors**: Felix Härer  

**Link**: [PDF](https://arxiv.org/pdf/2506.10467)  

**Abstract**: Recent advancements in LLMs indicate potential for novel applications, e.g., through reasoning capabilities in the latest OpenAI and DeepSeek models. For applying these models in specific domains beyond text generation, LLM-based multi-agent approaches can be utilized that solve complex tasks by combining reasoning techniques, code generation, and software execution. Applications might utilize these capabilities and the knowledge of specialized LLM agents. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application is not explored well. Defined specifications for multi-agent LLM systems are required to explore their potential and their suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research to specify and evaluate these aspects through a multi-agent system. The system architecture and prototype are extended from previous research and a specification is introduced for multi-agent systems. Test cases involving cybersecurity tasks indicate feasibility of the architecture and evaluation approach. In particular, the results show the evaluation of question answering, server security, and network security tasks that were completed correctly by agents with LLMs from OpenAI and DeepSeek. 

**Abstract (ZH)**: 近期大规模语言模型的进展表明了其在新型应用中的潜力，例如最新OpenAI和DeepSeek模型的推理能力。为了将这些模型应用于超越文本生成的具体领域，可以利用基于大规模语言模型的多智能体方法，通过结合推理技术、代码生成和软件执行来解决复杂任务。这些应用可能利用大规模语言模型智能体的能力和专业知识。然而，虽然对大规模语言模型、推理技术和应用进行了单独评估，但它们的联合规范及其组合应用尚未得到充分探索。为了探索多智能体大规模语言模型系统的潜力及其在特定应用中的适用性，需要定义明确的规范。本文报告了通过多智能体系统来规范和评估这些方面的一项探索性研究结果。该系统架构和原型基于先前的研究扩展，并引入了多智能体系统的规范。涉及网络安全任务的测试案例表明了该架构和评估方法的可行性。特别是，结果展示了利用来自OpenAI和DeepSeek的大规模语言模型的智能体正确完成的问题回答、服务器安全和网络安防任务的评估。 

---
# Starting Positions Matter: A Study on Better Weight Initialization for Neural Network Quantization 

**Title (ZH)**: 初始权重值的选择 matters：关于神经网络量化更优权重初始化方法的研究 

**Authors**: Stone Yun, Alexander Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.10463)  

**Abstract**: Deep neural network (DNN) quantization for fast, efficient inference has been an important tool in limiting the cost of machine learning (ML) model inference. Quantization-specific model development techniques such as regularization, quantization-aware training, and quantization-robustness penalties have served to greatly boost the accuracy and robustness of modern DNNs. However, very little exploration has been done on improving the initial conditions of DNN training for quantization. Just as random weight initialization has been shown to significantly impact test accuracy of floating point models, it would make sense that different weight initialization methods impact quantization robustness of trained models. We present an extensive study examining the effects of different weight initializations on a variety of CNN building blocks commonly used in efficient CNNs. This analysis reveals that even with varying CNN architectures, the choice of random weight initializer can significantly affect final quantization robustness. Next, we explore a new method for quantization-robust CNN initialization -- using Graph Hypernetworks (GHN) to predict parameters of quantized DNNs. Besides showing that GHN-predicted parameters are quantization-robust after regular float32 pretraining (of the GHN), we find that finetuning GHNs to predict parameters for quantized graphs (which we call GHN-QAT) can further improve quantized accuracy of CNNs. Notably, GHN-QAT shows significant accuracy improvements for even 4-bit quantization and better-than-random accuracy for 2-bits. To the best of our knowledge, this is the first in-depth study on quantization-aware DNN weight initialization. GHN-QAT offers a novel approach to quantized DNN model design. Future investigations, such as using GHN-QAT-initialized parameters for quantization-aware training, can further streamline the DNN quantization process. 

**Abstract (ZH)**: Deep神经网络（DNN）量化以实现快速、高效的推理一直是限制机器学习（ML）模型推理成本的重要工具。针对量化特异性的模型开发技术，如正则化、量化感知训练和量化鲁棒性惩罚，显著提升了现代DNN的准确性和鲁棒性。然而，在改进DNN训练的初始条件以适应量化方面，研究工作还很少。正如随机权重初始化已被证明对浮点模型的测试准确率有显著影响一样，不同权重初始化方法对训练模型的量化鲁棒性也理应有显著影响。我们进行了一项广泛的分析，研究了不同权重初始化方法对在高效CNN中常用的多种CNN构建块的影响。这一分析揭示了即使在不同的CNN架构下，随机权重初始化的选择也会显著影响最终的量化鲁棒性。随后，我们探索了一种新的量化鲁棒CNN初始化方法——使用Graph超网络（GHN）来预测量化DNN的参数。除了证明GHN预测的参数在常规float32预训练后具有量化鲁棒性之外，我们还发现，微调GHN以预测量化图的参数（我们称之为GHN-QAT）可以进一步提高CNN的量化准确性。值得注意的是，GHN-QAT甚至在4位量化时显示出显著的准确性提升，并在2位量化时优于随机准确性。据我们所知，这是关于量化感知DNN权重初始化的首个详细研究。GHN-QAT为量化DNN模型设计提供了一种新颖的方法。未来的进一步研究，如使用GHN-QAT初始化参数进行量化感知训练，可以进一步简化DNN的量化过程。 

---
# Equitable Mechanism Design for Facility Location 

**Title (ZH)**: 公平设施定位机制设计 

**Authors**: Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10460)  

**Abstract**: We consider strategy proof mechanisms for facility location which maximize equitability between agents. As is common in the literature, we measure equitability with the Gini index. We first prove a simple but fundamental impossibility result that no strategy proof mechanism can bound the approximation ratio of the optimal Gini index of utilities for one or more facilities. We propose instead computing approximation ratios of the complemented Gini index of utilities, and consider how well both deterministic and randomized mechanisms approximate this. In addition, as Nash welfare is often put forwards as an equitable compromise between egalitarian and utilitarian outcomes, we consider how well mechanisms approximate the Nash welfare. 

**Abstract (ZH)**: 我们考虑最大化代理人之间公平性的设施定位机制，并采用吉尼指数衡量公平性。我们首先证明了一个简单但基本的不可能性结果，即不存在能够限制单个或多个设施的最优吉尼指数近似比的策略证明机制。我们建议计算效用的补吉尼指数的近似比，并考虑确定性和随机机制对此的逼近程度。此外，由于纳什福利通常被认为是平等主义和功利主义结果的公平妥协，我们研究机制对此的逼近程度。 

---
# SOFT: Selective Data Obfuscation for Protecting LLM Fine-tuning against Membership Inference Attacks 

**Title (ZH)**: SOFT: 选择性数据模糊化以保护LLM微调免受成员推断攻击 

**Authors**: Kaiyuan Zhang, Siyuan Cheng, Hanxi Guo, Yuetian Chen, Zian Su, Shengwei An, Yuntao Du, Charles Fleming, Ashish Kundu, Xiangyu Zhang, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10424)  

**Abstract**: Large language models (LLMs) have achieved remarkable success and are widely adopted for diverse applications. However, fine-tuning these models often involves private or sensitive information, raising critical privacy concerns. In this work, we conduct the first comprehensive study evaluating the vulnerability of fine-tuned LLMs to membership inference attacks (MIAs). Our empirical analysis demonstrates that MIAs exploit the loss reduction during fine-tuning, making them highly effective in revealing membership information. These findings motivate the development of our defense. We propose SOFT (\textbf{S}elective data \textbf{O}bfuscation in LLM \textbf{F}ine-\textbf{T}uning), a novel defense technique that mitigates privacy leakage by leveraging influential data selection with an adjustable parameter to balance utility preservation and privacy protection. Our extensive experiments span six diverse domains and multiple LLM architectures and scales. Results show that SOFT effectively reduces privacy risks while maintaining competitive model performance, offering a practical and scalable solution to safeguard sensitive information in fine-tuned LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已取得显著成功，并广泛应用于多种应用中。然而，对这些模型进行微调往往涉及私人或敏感信息，引发了重要的隐私担忧。在本工作中，我们首次全面研究评估微调后的LLMs对抗成员推断攻击（MIAs）的脆弱性。我们的经验分析表明，MIAs利用微调过程中的损失减少，使其在揭示成员信息方面非常有效。这些发现促使我们开发了我们的防御方法。我们提出了SOFT（Selectively Obfuscating Data in LLM Fine-Tuning），这是一种新颖的防御技术，通过利用具有可调节参数的有影响的数据选择来平衡性能保留和隐私保护，从而减轻隐私泄露。我们广泛的研究覆盖了六个不同的领域和多种LLM架构与规模。结果表明，SOFT有效地减少了隐私风险，同时保持了竞争性的模型性能，提供了一种实用且可扩展的解决方案，以保护微调后的LLMs中的敏感信息。 

---
# PAL: Probing Audio Encoders via LLMs -- A Study of Information Transfer from Audio Encoders to LLMs 

**Title (ZH)**: PAL：通过语言模型探查音频编码器——音频编码器与语言模型之间信息传递的研究 

**Authors**: Tony Alex, Wish Suharitdamrong, Sara Atito, Armin Mustafa, Philip J. B. Jackson, Imran Razzak, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2506.10423)  

**Abstract**: The integration of audio perception capabilities into Large Language Models (LLMs) has enabled significant advances in Audio-LLMs. Although application-focused developments, particularly in curating training data for specific capabilities e.g., audio reasoning, have progressed rapidly, the underlying mechanisms that govern efficient transfer of rich semantic representations from audio encoders to LLMs remain under-explored. We conceptualize effective audio-LLM interaction as the LLM's ability to proficiently probe the audio encoder representations to satisfy textual queries. This paper presents a systematic investigation on how architectural design choices can affect that. Beginning with a standard Pengi/LLaVA-style audio-LLM architecture, we propose and evaluate several modifications guided by hypotheses derived from mechanistic interpretability studies and LLM operational principles. Our experiments demonstrate that: (1) delaying audio integration until the LLM's initial layers establish textual context that enhances its ability to probe the audio representations for relevant information; (2) the LLM can proficiently probe audio representations exclusively through LLM layer's attention submodule, without requiring propagation to its Feed-Forward Network (FFN) submodule; (3) an efficiently integrated ensemble of diverse audio encoders provides richer, complementary representations, thereby broadening the LLM's capacity to probe a wider spectrum of audio information. All hypotheses are evaluated using an identical three-stage training curriculum on a dataset of 5.6 million audio-text pairs, ensuring controlled comparisons. Our final architecture, which incorporates all proposed modifications, achieves relative improvements from 10\% to 60\% over the baseline, validating our approach to optimizing cross-modal information transfer in audio-LLMs. Project page: this https URL 

**Abstract (ZH)**: 将音频感知能力集成到大型语言模型（LLMs）中，推动了音频LLMs的显著进步。尽管在特定能力的数据集构建方面，尤其是针对音频推理的应用研发取得了快速进展，但指导高效迁移丰富语义表示从音频编码器到LLMs的基本机制仍需进一步探索。本文将有效音频LLM交互的概念化为LLM熟练地探查音频编码器表示以满足文本查询的能力，并系统地研究了架构设计选择如何影响这一过程。从标准的Pengi/LLaVA风格的音频LLM架构出发，我们基于机械可解释性研究和LLM操作原则提出并评估了几种修改。实验结果表明：（1）延迟音频集成直至LLM初始层建立文本语境，从而增强其探查音频表示的能力；（2）LLM仅通过馈送前注意力子模块独立有效地探查音频表示，无需传递到其馈送前网络子模块；（3）高效整合的多样化音频编码器集合提供了更丰富、互补的表示，从而拓宽了LLM探查更广泛音频信息的范围。所有假设均在包含560万个音频-文本对的数据集上采用相同的三阶段训练课程进行评估，确保了控制比较的环境。最终的架构整合了所有提出的修改，在基准模型上实现了10%至60%的相对改进，验证了我们优化音频LLMs跨模态信息传递的方法。项目页面：[此链接]。 

---
# Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series 

**Title (ZH)**: Time-IMM：不规则多模态多变量时间序列的数据集及基准 

**Authors**: Ching Chang, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wen-Chih Peng, Tien-Fu Chen, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10412)  

**Abstract**: Time series data in real-world applications such as healthcare, climate modeling, and finance are often irregular, multimodal, and messy, with varying sampling rates, asynchronous modalities, and pervasive missingness. However, existing benchmarks typically assume clean, regularly sampled, unimodal data, creating a significant gap between research and real-world deployment. We introduce Time-IMM, a dataset specifically designed to capture cause-driven irregularity in multimodal multivariate time series. Time-IMM represents nine distinct types of time series irregularity, categorized into trigger-based, constraint-based, and artifact-based mechanisms. Complementing the dataset, we introduce IMM-TSF, a benchmark library for forecasting on irregular multimodal time series, enabling asynchronous integration and realistic evaluation. IMM-TSF includes specialized fusion modules, including a timestamp-to-text fusion module and a multimodality fusion module, which support both recency-aware averaging and attention-based integration strategies. Empirical results demonstrate that explicitly modeling multimodality on irregular time series data leads to substantial gains in forecasting performance. Time-IMM and IMM-TSF provide a foundation for advancing time series analysis under real-world conditions. The dataset is publicly available at this https URL, and the benchmark library can be accessed at this https URL. 

**Abstract (ZH)**: 时间序列数据在医疗、气候建模和金融等实际应用中通常是不规则的、多模态的和混乱的，具有变化的采样率、异步的模态性和普遍的数据缺失问题。现有的基准数据集通常假定清洁的、定期采样的、单模态的数据，这在研究与实际部署之间造成了显著的差距。我们引入Time-IMM数据集，专门用于捕捉由因果驱动的多模态多变量时间序列的不规则性。Time-IMM代表九种不同类型的时间序列不规则性，分类为触发机制、约束机制和残余机制。为补充数据集，我们引入了IMM-TSF基准库，用于不规则多模态时间序列的预测，在异步集成和现实评价方面提供支持。IMM-TSF包含专门的融合模块，包括时间戳到文本融合模块和多模态融合模块，支持最近性感知平均策略和基于注意的集成策略。实证结果表明，在不规则时间序列数据中明确建模多模态性能够显著提高预测性能。Time-IMM和IMM-TSF为在实际条件下推进时间序列分析提供了基础。数据集可从以下网址获取：this https URL，基准库可从以下网址访问：this https URL。 

---
# Semi-Tensor-Product Based Convolutional Neural Networks 

**Title (ZH)**: 基于半张量积的卷积神经网络 

**Authors**: Daizhan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10407)  

**Abstract**: The semi-tensor product (STP) of vectors is a generalization of conventional inner product of vectors, which allows the factor vectors to of different dimensions. This paper proposes a domain-based convolutional product (CP). Combining domain-based CP with STP of vectors, a new CP is proposed. Since there is no zero or any other padding, it can avoid the junk information caused by padding. Using it, the STP-based convolutional neural network (CNN) is developed. Its application to image and third order signal identifications is considered. 

**Abstract (ZH)**: 基于半张量积的域域卷积产品及其在网络中的应用 

---
# PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier 

**Title (ZH)**: PAG：基于策略生成验证的多轮强化LLM自我修正 

**Authors**: Yuhua Jiang, Yuwen Xiong, Yufeng Yuan, Chao Xin, Wenyuan Xu, Yu Yue, Qianchuan Zhao, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10406)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in complex reasoning tasks, yet they still struggle to reliably verify the correctness of their own outputs. Existing solutions to this verification challenge often depend on separate verifier models or require multi-stage self-correction training pipelines, which limit scalability. In this paper, we propose Policy as Generative Verifier (PAG), a simple and effective framework that empowers LLMs to self-correct by alternating between policy and verifier roles within a unified multi-turn reinforcement learning (RL) paradigm. Distinct from prior approaches that always generate a second attempt regardless of model confidence, PAG introduces a selective revision mechanism: the model revises its answer only when its own generative verification step detects an error. This verify-then-revise workflow not only alleviates model collapse but also jointly enhances both reasoning and verification abilities. Extensive experiments across diverse reasoning benchmarks highlight PAG's dual advancements: as a policy, it enhances direct generation and self-correction accuracy; as a verifier, its self-verification outperforms self-consistency. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中展现了令人印象深刻的性能，但在可靠验证自身输出的正确性方面仍然面临挑战。现有的解决方案通常依赖于单独的验证模型或需要多阶段的自我纠正训练管道，这限制了其可扩展性。本文提出了一种简单有效的框架——Policy as Generative Verifier (PAG)，该框架在统一的多轮强化学习（RL）框架内交替扮演策略和验证者角色，以增强LLMs的自我纠正能力。PAG引入了一种选择性的修订机制：只有当模型的生成验证步骤检测到错误时，模型才会进行修订。这种先验证后修订的工作流程不仅减轻了模型崩溃的问题，还同时提升了推理和验证的能力。跨多种推理基准的广泛实验展示了PAG的双重进步：作为策略，它增强了直接生成和自我纠正的准确性；作为验证者，它的自我验证优于自我一致性。 

---
# Time To Impeach LLM-as-a-Judge: Programs are the Future of Evaluation 

**Title (ZH)**: 弹劾LLM-as-a-Judge的时代：程序将是评估的未来 

**Authors**: Tzu-Heng Huang, Harit Vishwakarma, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2506.10403)  

**Abstract**: Large language models (LLMs) are widely used to evaluate the quality of LLM generations and responses, but this leads to significant challenges: high API costs, uncertain reliability, inflexible pipelines, and inherent biases. To address these, we introduce PAJAMA (Program-As-a-Judge for Automated Model Assessment), a new alternative that uses LLMs to synthesize executable judging programs instead of directly scoring responses. These synthesized programs can be stored and run locally, costing orders of magnitude less while providing interpretable, and auditable judging logic that can be easily adapted. Program-based judges mitigate biases, improving judgment consistency by 15.83% and reducing biased responses by 23.7% on average compared to a Qwen2.5-14B-based LLM-as-a-judge. When program judgments are distilled into a model, PAJAMA outperforms LLM-as-a-judge on the challenging CHAT-HARD subset of RewardBench, outperforming metrics by 2.19% on Prometheus and 8.67% on the JudgeLM dataset, all at three orders of magnitude lower cost. 

**Abstract (ZH)**: PAJAMA：作为一种评判程序的自动化模型评估新方法 

---
# HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration 

**Title (ZH)**: HPCTransCompile: 一个用于高性能CUDA转换和LLM初步探索的AI编译器生成数据集 

**Authors**: Jiaqi Lv, Xufeng He, Yanchen Liu, Xu Dai, Yang Hu, Shouyi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10401)  

**Abstract**: The rapid growth of deep learning has driven exponential increases in model parameters and computational demands. NVIDIA GPUs and their CUDA-based software ecosystem provide robust support for parallel computing, significantly alleviating computational bottlenecks. Meanwhile, due to the cultivation of user programming habits and the high performance of GPUs, the CUDA ecosystem has established a dominant position in the field of parallel software. This dominance requires other hardware platforms to support CUDA-based software with performance portability. However, translating CUDA code to other platforms poses significant challenges due to differences in parallel programming paradigms and hardware architectures. Existing approaches rely on language extensions, domain-specific languages (DSLs), or compilers but face limitations in workload coverage and generalizability. Moreover, these methods often incur substantial development costs. Recently, LLMs have demonstrated extraordinary potential in various vertical domains, especially in code-related tasks. However, the performance of existing LLMs in CUDA transpilation, particularly for high-performance code, remains suboptimal. The main reason for this limitation lies in the lack of high-quality training datasets. To address these challenges, we propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The result demonstrates that our framework significantly improves CUDA transpilation, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem. 

**Abstract (ZH)**: 深度学习的迅速增长推动了模型参数和计算需求的指数级增加。NVIDIA GPU及其基于CUDA的软件生态系统提供了强大的并行计算支持，显著缓解了计算瓶颈问题。同时，由于用户的编程习惯培养和GPU的高性能，CUDA生态系统在并行软件领域建立了主导地位。这种主导地位要求其他硬件平台支持基于CUDA的软件，并具有性能可移植性。然而，将CUDA代码翻译到其他平台面临着显著挑战，因为并行编程范式和硬件架构存在差异。现有方法依赖于语言扩展、领域特定语言（DSL）或编译器，但在工作负载覆盖范围和普适性方面存在局限性。此外，这些方法往往需要大量的开发成本。近期，大规模语言模型（LLMs）在各类垂直领域展示了非凡的潜力，尤其是在代码相关任务方面。然而，现有LLMs在CUDA翻译，尤其是在高性能代码方面的表现仍然不尽如人意。这一局限性主要是由于缺乏高质量的训练数据集。为了解决这些挑战，我们提出了一种新的框架，用于生成高性能的CUDA及其对应的平台代码对，该框架结合了AI编译器和自动优化技术。我们进一步通过基于图的数据增强方法增强了该框架，并引入了HPCTransEval，这是一个用于评估LLMs在CUDA翻译性能的基准测试。我们使用CUDA到CPU翻译作为案例研究，在领先的大规模语言模型上进行了实验。实验结果表明，我们的框架显著提高了CUDA翻译性能，突显了LLMs在CUDA生态系统内解决兼容性挑战的潜力。 

---
# Pisces: An Auto-regressive Foundation Model for Image Understanding and Generation 

**Title (ZH)**: Pisces：一种用于图像理解与生成的自回归基础模型 

**Authors**: Zhiyang Xu, Jiuhai Chen, Zhaojiang Lin, Xichen Pan, Lifu Huang, Tianyi Zhou, Madian Khabsa, Qifan Wang, Di Jin, Michihiro Yasunaga, Lili Yu, Xi Victoria Lin, Shaoliang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10395)  

**Abstract**: Recent advances in large language models (LLMs) have enabled multimodal foundation models to tackle both image understanding and generation within a unified framework. Despite these gains, unified models often underperform compared to specialized models in either task. A key challenge in developing unified models lies in the inherent differences between the visual features needed for image understanding versus generation, as well as the distinct training processes required for each modality. In this work, we introduce Pisces, an auto-regressive multimodal foundation model that addresses this challenge through a novel decoupled visual encoding architecture and tailored training techniques optimized for multimodal generation. Combined with meticulous data curation, pretraining, and finetuning, Pisces achieves competitive performance in both image understanding and image generation. We evaluate Pisces on over 20 public benchmarks for image understanding, where it demonstrates strong performance across a wide range of tasks. Additionally, on GenEval, a widely adopted benchmark for image generation, Pisces exhibits robust generative capabilities. Our extensive analysis reveals the synergistic relationship between image understanding and generation, and the benefits of using separate visual encoders, advancing the field of unified multimodal models. 

**Abstract (ZH)**: 近期大规模语言模型（LLMs）的进展使多模态基础模型能够在统一框架内解决图像理解与生成问题。尽管取得了这些进展，统一模型在各项任务中的表现往往不如专门针对某一任务训练的模型。开发统一模型的关键挑战在于图像理解和生成所需视觉特征的内在差异，以及每种模态所需的独特训练过程。在此项工作中，我们引入了Pisces，一种通过新颖的解耦视觉编码架构和针对多模态生成优化的定制训练技术的自回归多模态基础模型。结合细致的数据整理、预训练和微调，Pisces在图像理解与图像生成任务中均表现出竞争性的性能。我们在超过20个公开图像理解基准上评估了Pisces，结果表明其在多种任务上表现强劲。此外，在广泛采用的图像生成基准GenEval上，Pisces展示了稳健的生成能力。我们深入的分析揭示了图像理解和生成之间的协同关系，并证实了使用独立视觉编码器的优势，推动了统一多模态模型的发展。 

---
# Discovering Hierarchical Latent Capabilities of Language Models via Causal Representation Learning 

**Title (ZH)**: 通过因果表示学习发现语言模型的分层潜在能力 

**Authors**: Jikai Jin, Vasilis Syrgkanis, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10378)  

**Abstract**: Faithful evaluation of language model capabilities is crucial for deriving actionable insights that can inform model development. However, rigorous causal evaluations in this domain face significant methodological challenges, including complex confounding effects and prohibitive computational costs associated with extensive retraining. To tackle these challenges, we propose a causal representation learning framework wherein observed benchmark performance is modeled as a linear transformation of a few latent capability factors. Crucially, these latent factors are identified as causally interrelated after appropriately controlling for the base model as a common confounder. Applying this approach to a comprehensive dataset encompassing over 1500 models evaluated across six benchmarks from the Open LLM Leaderboard, we identify a concise three-node linear causal structure that reliably explains the observed performance variations. Further interpretation of this causal structure provides substantial scientific insights beyond simple numerical rankings: specifically, we reveal a clear causal direction starting from general problem-solving capabilities, advancing through instruction-following proficiency, and culminating in mathematical reasoning ability. Our results underscore the essential role of carefully controlling base model variations during evaluation, a step critical to accurately uncovering the underlying causal relationships among latent model capabilities. 

**Abstract (ZH)**: 真实评估语言模型能力对于提取可操作的洞察以指导模型开发至关重要。然而，在此领域进行严格的因果评估面临着重大的方法论挑战，包括复杂的共变量效应以及与大量重新训练相关的高昂计算成本。为应对这些挑战，我们提出了一种因果表示学习框架，其中观察到的基准性能被建模为少量潜在能力因子的线性变换。关键的是，在适当控制基模型作为共同共变量之后，这些潜在因子被识别为彼此因果相关。将这种方法应用于包括来自Open LLM Leaderboard的六个基准测试中超过1500个模型的数据集，我们识别出一个可靠的三个节点线性因果结构，该结构能解释观察到的性能变化。进一步对这一因果结构的解释提供了超出简单数值排名的大量科学洞见：具体而言，我们揭示了一个明确的因果方向，从一般问题解决能力出发，经过指令跟随熟练程度，最终达到数学推理能力。我们的结果强调了在评估过程中仔细控制基模型变异性的关键作用，这是准确揭示潜在模型能力之间因果关系的必要步骤。 

---
# PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation 

**Title (ZH)**: 生 PhysioWave: 一种多尺度小波转换器的生理信号表示方法 

**Authors**: Yanlong Chen, Mattia Orlandi, Pierangelo Maria Rapa, Simone Benatti, Luca Benini, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10351)  

**Abstract**: Physiological signals are often corrupted by motion artifacts, baseline drift, and other low-SNR disturbances, which pose significant challenges for analysis. Additionally, these signals exhibit strong non-stationarity, with sharp peaks and abrupt changes that evolve continuously, making them difficult to represent using traditional time-domain or filtering methods. To address these issues, a novel wavelet-based approach for physiological signal analysis is presented, aiming to capture multi-scale time-frequency features in various physiological signals. Leveraging this technique, two large-scale pretrained models specific to EMG and ECG are introduced for the first time, achieving superior performance and setting new baselines in downstream tasks. Additionally, a unified multi-modal framework is constructed by integrating pretrained EEG model, where each modality is guided through its dedicated branch and fused via learnable weighted fusion. This design effectively addresses challenges such as low signal-to-noise ratio, high inter-subject variability, and device mismatch, outperforming existing methods on multi-modal tasks. The proposed wavelet-based architecture lays a solid foundation for analysis of diverse physiological signals, while the multi-modal design points to next-generation physiological signal processing with potential impact on wearable health monitoring, clinical diagnostics, and broader biomedical applications. 

**Abstract (ZH)**: 生理信号常受到运动伪影、基线漂移和其他低信噪比干扰的污染，这对分析构成了重大挑战。此外，这些信号表现出强烈的非 Stationarity，拥有不断变化的尖峰和突变，传统的时间域方法或滤波方法难以对其进行表示。为解决这些问题，本文提出了一种基于小波的新颖生理信号分析方法，旨在捕获各种生理信号的多尺度时频特征。利用该技术，首次引入了针对肌电图（EMG）和心电图（ECG）的两个大规模预训练模型，这些模型在下游任务中表现出色，并设立了新的基准。此外，通过将预训练的大脑电图（EEG）模型与多种模态整合构建了一个统一的多模态框架，每种模态通过其专门的分支并借助可学习加权融合进行融合。该设计有效解决了低信噪比、高个体间变异性及设备匹配不良等挑战，多模态任务上优于现有方法。提出的基于小波的架构为各种生理信号的分析奠定了坚实基础，而多模态设计预示着下一代生理信号处理的发展，有可能在穿戴式健康监测、临床诊断以及更广泛的生物医学应用中产生重要影响。 

---
# Code Execution as Grounded Supervision for LLM Reasoning 

**Title (ZH)**: 代码执行作为LLM推理的基于地面监督的方法 

**Authors**: Dongwon Jung, Wenxuan Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10343)  

**Abstract**: Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking. 

**Abstract (ZH)**: 利用程序执行的确定性生成高质量链式思维监督数据集以增强大型语言模型的推理能力 

---
# UrbanSense:AFramework for Quantitative Analysis of Urban Streetscapes leveraging Vision Large Language Models 

**Title (ZH)**: UrbanSense：一种基于视觉大规模语言模型的都市街道景观定量分析框架 

**Authors**: Jun Yin, Jing Zhong, Peilin Li, Pengyu Zeng, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10342)  

**Abstract**: Urban cultures and architectural styles vary significantly across cities due to geographical, chronological, historical, and socio-political factors. Understanding these differences is essential for anticipating how cities may evolve in the future. As representative cases of historical continuity and modern innovation in China, Beijing and Shenzhen offer valuable perspectives for exploring the transformation of urban streetscapes. However, conventional approaches to urban cultural studies often rely on expert interpretation and historical documentation, which are difficult to standardize across different contexts. To address this, we propose a multimodal research framework based on vision-language models, enabling automated and scalable analysis of urban streetscape style differences. This approach enhances the objectivity and data-driven nature of urban form research. The contributions of this study are as follows: First, we construct UrbanDiffBench, a curated dataset of urban streetscapes containing architectural images from different periods and regions. Second, we develop UrbanSense, the first vision-language-model-based framework for urban streetscape analysis, enabling the quantitative generation and comparison of urban style representations. Third, experimental results show that Over 80% of generated descriptions pass the t-test (p less than 0.05). High Phi scores (0.912 for cities, 0.833 for periods) from subjective evaluations confirm the method's ability to capture subtle stylistic differences. These results highlight the method's potential to quantify and interpret urban style evolution, offering a scientifically grounded lens for future design. 

**Abstract (ZH)**: 城市的文化与建筑风格因地理、历史、社会政治等因素在不同城市间存在显著差异。理解这些差异对于预测城市未来的演化至关重要。作为中国历史连续性和现代创新的代表案例，北京和深圳为探索城市街道景观的演变提供了宝贵视角。然而，传统城市文化研究方法往往依赖于专家解释和历史文献，难以在不同背景下标准化。为此，我们提出基于视觉-语言模型的多模态研究框架，实现对城市街道景观风格差异的自动化和可扩展分析。该方法增强了城市形态研究的客观性和数据驱动性质。本研究的贡献包括：首先，构建了包含不同历史时期和地区建筑图片的UrbanDiffBench数据集；其次，开发了基于视觉-语言模型的第一种城市街道景观分析框架UrbanSense，能够定量生成和比较城市风格表示；最后，实验结果表明，超过80%生成的描述通过了t检验（p<0.05），主观评价的高Phi分数（城市为0.912，时期为0.833）证实了该方法捕获微妙风格差异的能力。这些结果突显了该方法量化和解读城市风格演化的潜力，为未来设计提供了一种科学视角。 

---
# Using Vision Language Models to Detect Students' Academic Emotion through Facial Expressions 

**Title (ZH)**: 使用视觉语言模型检测学生通过面部表情表达的学术情绪 

**Authors**: Deliang Wang, Chao Yang, Gaowei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10334)  

**Abstract**: Students' academic emotions significantly influence their social behavior and learning performance. Traditional approaches to automatically and accurately analyze these emotions have predominantly relied on supervised machine learning algorithms. However, these models often struggle to generalize across different contexts, necessitating repeated cycles of data collection, annotation, and training. The emergence of Vision-Language Models (VLMs) offers a promising alternative, enabling generalization across visual recognition tasks through zero-shot prompting without requiring fine-tuning. This study investigates the potential of VLMs to analyze students' academic emotions via facial expressions in an online learning environment. We employed two VLMs, Llama-3.2-11B-Vision-Instruct and Qwen2.5-VL-7B-Instruct, to analyze 5,000 images depicting confused, distracted, happy, neutral, and tired expressions using zero-shot prompting. Preliminary results indicate that both models demonstrate moderate performance in academic facial expression recognition, with Qwen2.5-VL-7B-Instruct outperforming Llama-3.2-11B-Vision-Instruct. Notably, both models excel in identifying students' happy emotions but fail to detect distracted behavior. Additionally, Qwen2.5-VL-7B-Instruct exhibits relatively high performance in recognizing students' confused expressions, highlighting its potential for practical applications in identifying content that causes student confusion. 

**Abstract (ZH)**: 学生的情绪对其社交行为和学习表现有显著影响。传统的自动准确分析这些情绪的方法主要依赖于监督机器学习算法。然而，这些模型往往难以在不同情境下泛化，需要反复的数据收集、注释和训练。视觉-语言模型（VLMs）的出现为其提供了有前景的替代方案，通过零样本提示实现跨视觉识别任务的泛化而无需微调。本研究探讨了VLMs在在线学习环境中通过面部表情分析学生学术情绪的潜力。我们使用了两个VLMs，Llama-3.2-11B-Vision-Instruct和Qwen2.5-VL-7B-Instruct，对5,000张表情图像（包括困惑、分心、快乐、中性和平静）进行零样本提示分析。初步结果显示，两种模型在学术面部表情识别方面表现出中等性能，Qwen2.5-VL-7B-Instruct优于Llama-3.2-11B-Vision-Instruct。值得注意的是，两种模型都擅长识别学生的情绪喜悦，但在检测分心行为方面表现不佳。此外，Qwen2.5-VL-7B-Instruct在识别学生困惑的表情方面表现出相对较高的性能，突显了其在识别引起学生困惑的内容方面的潜在实际应用价值。 

---
# Augmenting Large Language Models with Static Code Analysis for Automated Code Quality Improvements 

**Title (ZH)**: 利用静态代码分析增强大型语言模型以实现自动化代码质量改进 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10330)  

**Abstract**: This study examined code issue detection and revision automation by integrating Large Language Models (LLMs) such as OpenAI's GPT-3.5 Turbo and GPT-4o into software development workflows. A static code analysis framework detects issues such as bugs, vulnerabilities, and code smells within a large-scale software project. Detailed information on each issue was extracted and organized to facilitate automated code revision using LLMs. An iterative prompt engineering process is applied to ensure that prompts are structured to produce accurate and organized outputs aligned with the project requirements. Retrieval-augmented generation (RAG) is implemented to enhance the relevance and precision of the revisions, enabling LLM to access and integrate real-time external knowledge. The issue of LLM hallucinations - where the model generates plausible but incorrect outputs - is addressed by a custom-built "Code Comparison App," which identifies and corrects erroneous changes before applying them to the codebase. Subsequent scans using the static code analysis framework revealed a significant reduction in code issues, demonstrating the effectiveness of combining LLMs, static analysis, and RAG to improve code quality, streamline the software development process, and reduce time and resource expenditure. 

**Abstract (ZH)**: 本研究通过将如OpenAI的GPT-3.5 Turbo和GPT-4这样的大规模语言模型（LLMs）整合到软件开发工作流程中，研究了代码问题检测与自动修订的自动化。静态代码分析框架在大型软件项目中检测错误、漏洞和代码气味等问题，并提取和组织详细信息以供使用LLMs进行自动代码修订。通过迭代的提示工程过程确保提示结构化，生成与项目需求一致的准确且组织良好的输出。通过检索增强生成（RAG）提升修订的相关性和精确度，使LLMs能够访问和整合实时外部知识。通过自定义构建的“代码比较应用”解决了LLMs幻觉问题，即模型生成看似合理但实际上错误的输出，该应用在将更改应用于代码库之前识别并纠正了错误变化。后续使用静态代码分析框架的扫描显示，通过结合LLMs、静态分析和RAG的方法，显著减少了代码问题，证明了提高代码质量、简化软件开发过程并减少时间和资源支出的有效性。 

---
# Towards Scalable SOAP Note Generation: A Weakly Supervised Multimodal Framework 

**Title (ZH)**: 面向可扩展的SOAP笔记生成：一种弱监督多模态框架 

**Authors**: Sadia Kamal, Tim Oates, Joy Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10328)  

**Abstract**: Skin carcinoma is the most prevalent form of cancer globally, accounting for over $8 billion in annual healthcare expenditures. In clinical settings, physicians document patient visits using detailed SOAP (Subjective, Objective, Assessment, and Plan) notes. However, manually generating these notes is labor-intensive and contributes to clinician burnout. In this work, we propose a weakly supervised multimodal framework to generate clinically structured SOAP notes from limited inputs, including lesion images and sparse clinical text. Our approach reduces reliance on manual annotations, enabling scalable, clinically grounded documentation while alleviating clinician burden and reducing the need for large annotated data. Our method achieves performance comparable to GPT-4o, Claude, and DeepSeek Janus Pro across key clinical relevance metrics. To evaluate clinical quality, we introduce two novel metrics MedConceptEval and Clinical Coherence Score (CCS) which assess semantic alignment with expert medical concepts and input features, respectively. 

**Abstract (ZH)**: 皮肤癌是全球最常见的癌症类型，每年在医疗保健支出中占比超过80亿美元。在临床环境中，医生利用详细的SOAP（主观、客观、评估、计划）笔记记录患者就诊情况。然而，手工生成这些笔记非常费时且增加了医务人员的职业倦怠。在本研究中，我们提出一种弱监督多模态框架，从有限输入（包括病损图像和稀疏临床文本）自动生成结构化的SOAP笔记。该方法减少了对人工标注的依赖，实现了可扩展的、基于临床的记录，减轻了医务人员的负担并减少了大量标注数据的需求。我们的方法在关键临床相关性指标上达到了与GPT-4o、Claude和DeepSeek Janus Pro相当的性能。为评估临床质量，我们引入了两个新指标：MedConceptEval和临床一致性分数（CCS），分别评估语义与专家医学概念的一致性和输入特征的一致性。 

---
# Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving 

**Title (ZH)**: 使用语言和道路手册告知自动驾驶中的地图重建 

**Authors**: Akshar Tumu, Henrik I. Christensen, Marcell Vazquez-Chanlatte, Chikao Tsuchiya, Dhaval Bhanderi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10317)  

**Abstract**: Lane-topology prediction is a critical component of safe and reliable autonomous navigation. An accurate understanding of the road environment aids this task. We observe that this information often follows conventions encoded in natural language, through design codes that reflect the road structure and road names that capture the road functionality. We augment this information in a lightweight manner to SMERF, a map-prior-based online lane-topology prediction model, by combining structured road metadata from OSM maps and lane-width priors from Road design manuals with the road centerline encodings. We evaluate our method on two geo-diverse complex intersection scenarios. Our method shows improvement in both lane and traffic element detection and their association. We report results using four topology-aware metrics to comprehensively assess the model performance. These results demonstrate the ability of our approach to generalize and scale to diverse topologies and conditions. 

**Abstract (ZH)**: 车道拓扑预测是实现安全可靠自主导航的关键组件。准确理解道路环境有助于这一任务。我们观察到这些信息通常遵循在自然语言中编码的惯例，通过反映道路结构的设计代码和捕捉道路功能的道路名称体现。我们以轻量级的方式将这些信息补充到基于地图先验的在线车道拓扑预测模型SMERF中，结合开放street地图（OSM）中的结构化道路元数据和道路设计手册中的车道宽度先验，以及道路中心线编码。我们在两个地理位置和复杂度各异的交叉口场景上评估了该方法。我们的方法在车道和交通元素检测及其关联性方面都表现出改进。我们使用四种拓扑感知指标全面评估了模型性能。这些结果展示了我们方法泛化和适应各种拓扑结构和条件的能力。 

---
# DUN-SRE: Deep Unrolling Network with Spatiotemporal Rotation Equivariance for Dynamic MRI Reconstruction 

**Title (ZH)**: DUN-SRE：具有时空旋转不变性的深层解卷网络用于动态MRI重建 

**Authors**: Yuliang Zhu, Jing Cheng, Qi Xie, Zhuo-Xu Cui, Qingyong Zhu, Yuanyuan Liu, Xin Liu, Jianfeng Ren, Chengbo Wang, Dong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10309)  

**Abstract**: Dynamic Magnetic Resonance Imaging (MRI) exhibits transformation symmetries, including spatial rotation symmetry within individual frames and temporal symmetry along the time dimension. Explicit incorporation of these symmetry priors in the reconstruction model can significantly improve image quality, especially under aggressive undersampling scenarios. Recently, Equivariant convolutional neural network (ECNN) has shown great promise in exploiting spatial symmetry priors. However, existing ECNNs critically fail to model temporal symmetry, arguably the most universal and informative structural prior in dynamic MRI reconstruction. To tackle this issue, we propose a novel Deep Unrolling Network with Spatiotemporal Rotation Equivariance (DUN-SRE) for Dynamic MRI Reconstruction. The DUN-SRE establishes spatiotemporal equivariance through a (2+1)D equivariant convolutional architecture. In particular, it integrates both the data consistency and proximal mapping module into a unified deep unrolling framework. This architecture ensures rigorous propagation of spatiotemporal rotation symmetry constraints throughout the reconstruction process, enabling more physically accurate modeling of cardiac motion dynamics in cine MRI. In addition, a high-fidelity group filter parameterization mechanism is developed to maintain representation precision while enforcing symmetry constraints. Comprehensive experiments on Cardiac CINE MRI datasets demonstrate that DUN-SRE achieves state-of-the-art performance, particularly in preserving rotation-symmetric structures, offering strong generalization capability to a broad range of dynamic MRI reconstruction tasks. 

**Abstract (ZH)**: 动态磁共振成像（MRI）表现出变换对称性，包括个体框架内的空间旋转对称性和时间维度上的时间对称性。在重建模型中明确 Incorporate 这些对称性先验可以显著提高图像质量，尤其是在激进下采样场景下。最近，空间对称性协变卷积神经网络（ECNN）在利用空间对称性先验方面展现出巨大的潜力。然而，现有的 ECNN 严重无法建模时间对称性，这被认为是动态 MRI 重建中最普遍且最具信息量的结构先验。为解决这一问题，我们提出了一种用于动态 MRI 重建的时空旋转协变 Deep Unrolling 网络（DUN-SRE）。DUN-SRE 通过（2+1）D 协变卷积架构建立了时空协变性。特别地，它将数据一致性模块和邻近映射模块结合进一个统一的 Deep Unrolling 框架。该架构确保在重建过程中严格传播时空旋转对称性约束，从而更准确地建模 cine MRI 中的心脏运动动态。此外，开发了一种高保真群滤波器参数化机制，以在施加对称性约束的同时保持表示精度。全面的实验结果表明，DUN-SRE 在保持旋转对称结构方面达到了最先进的性能，具有较强的泛化能力，能够应对广泛的动态 MRI 重建任务。 

---
# Uncertainty-Aware Deep Learning for Automated Skin Cancer Classification: A Comprehensive Evaluation 

**Title (ZH)**: 基于不确定性感知的深度学习在皮肤癌自动化分类中的全面评估 

**Authors**: Hamzeh Asgharnezhad, Pegah Tabarisaadi, Abbas Khosravi, Roohallah Alizadehsani, U. Rajendra Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2506.10302)  

**Abstract**: Accurate and reliable skin cancer diagnosis is critical for early treatment and improved patient outcomes. Deep learning (DL) models have shown promise in automating skin cancer classification, but their performance can be limited by data scarcity and a lack of uncertainty awareness. In this study, we present a comprehensive evaluation of DL-based skin lesion classification using transfer learning and uncertainty quantification (UQ) on the HAM10000 dataset. In the first phase, we benchmarked several pre-trained feature extractors-including Contrastive Language-Image Pretraining (CLIP) variants, Residual Network-50 (ResNet50), Densely Connected Convolutional Network (DenseNet121), Visual Geometry Group network (VGG16), and EfficientNet-V2-Large-combined with a range of traditional classifiers such as Support Vector Machine (SVM), eXtreme Gradient Boosting (XGBoost), and logistic regression. Our results show that CLIP-based vision transformers, particularly LAION CLIP ViT-H/14 with SVM, deliver the highest classification performance. In the second phase, we incorporated UQ using Monte Carlo Dropout (MCD), Ensemble, and Ensemble Monte Carlo Dropout (EMCD) to assess not only prediction accuracy but also the reliability of model outputs. We evaluated these models using uncertainty-aware metrics such as uncertainty accuracy(UAcc), uncertainty sensitivity(USen), uncertainty specificity(USpe), and uncertainty precision(UPre). The results demonstrate that ensemble methods offer a good trade-off between accuracy and uncertainty handling, while EMCD is more sensitive to uncertain predictions. This study highlights the importance of integrating UQ into DL-based medical diagnosis to enhance both performance and trustworthiness in real-world clinical applications. 

**Abstract (ZH)**: 准确可靠的皮肤癌诊断对于早期治疗和改善患者预后至关重要。深度学习模型在自动化皮肤癌分类方面显示出潜力，但其性能可能受限于数据稀缺性和不确定性意识不足。在本研究中，我们对基于迁移学习和不确定性量化（UQ）的深度学习皮肤病变分类进行了全面评估，使用了HAM10000数据集。在第一阶段，我们对标了几种预训练特征提取器，包括对比语言-图像预训练（CLIP）变体、残差网络-50（ResNet50）、密集连接卷积网络（DenseNet121）、视觉几何组网络（VGG16）和高效Net-V2-大型（EfficientNet-V2-Large），并结合了传统的分类器，如支持向量机（SVM）、极端梯度提升（XGBoost）和逻辑回归。结果表明，基于CLIP的视觉变换器，特别是LAION CLIP ViT-H/14与SVM结合，提供最高的分类性能。在第二阶段，我们引入了蒙特卡洛丢弃（MCD）、集成（Ensemble）和集成蒙特卡洛丢弃（EMCD），以评估预测准确性和模型输出的可靠性。我们使用不确定性意识度量标准，如不确定性准确度（UAcc）、不确定性敏感性（USen）、不确定性特异性（USpe）和不确定性精确度（UPre）评估这些模型。结果表明，集成方法在准确性和不确定性处理之间提供了良好的权衡，而EMCD对不确定的预测更为敏感。本研究强调了在基于深度学习的医疗诊断中集成不确定性量化的重要性，以提高实际临床应用中的性能和可信度。 

---
# Towards Understanding Bias in Synthetic Data for Evaluation 

**Title (ZH)**: 理解合成数据在评估中偏见的问题 

**Authors**: Hossein A. Rahmani, Varsha Ramineni, Nick Craswell, Bhaskar Mitra, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10301)  

**Abstract**: Test collections are crucial for evaluating Information Retrieval (IR) systems. Creating a diverse set of user queries for these collections can be challenging, and obtaining relevance judgments, which indicate how well retrieved documents match a query, is often costly and resource-intensive. Recently, generating synthetic datasets using Large Language Models (LLMs) has gained attention in various applications. While previous work has used LLMs to generate synthetic queries or documents to improve ranking models, using LLMs to create synthetic test collections is still relatively unexplored. Previous work~\cite{rahmani2024synthetic} showed that synthetic test collections have the potential to be used for system evaluation, however, more analysis is needed to validate this claim. In this paper, we thoroughly investigate the reliability of synthetic test collections constructed using LLMs, where LLMs are used to generate synthetic queries, labels, or both. In particular, we examine the potential biases that might occur when such test collections are used for evaluation. We first empirically show the presence of such bias in evaluation results and analyse the effects it might have on system evaluation. We further validate the presence of such bias using a linear mixed-effects model. Our analysis shows that while the effect of bias present in evaluation results obtained using synthetic test collections could be significant, for e.g.~computing absolute system performance, its effect may not be as significant in comparing relative system performance. Codes and data are available at: this https URL. 

**Abstract (ZH)**: 使用大规模语言模型生成的合成测试集合在信息检索系统评估中的可靠性探究 

---
# Flick: Few Labels Text Classification using K-Aware Intermediate Learning in Multi-Task Low-Resource Languages 

**Title (ZH)**: Flick：多任务低资源语言中的K- aware中间学习少标签文本分类 

**Authors**: Ali Almutairi, Abdullah Alsuhaibani, Shoaib Jameel, Usman Naseem, Gelareh Mohammadi, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2506.10292)  

**Abstract**: Training deep learning networks with minimal supervision has gained significant research attention due to its potential to reduce reliance on extensive labelled data. While self-training methods have proven effective in semi-supervised learning, they remain vulnerable to errors from noisy pseudo labels. Moreover, most recent approaches to the few-label classification problem are either designed for resource-rich languages such as English or involve complex cascading models that are prone to overfitting. To address the persistent challenge of few-label text classification in truly low-resource linguistic contexts, where existing methods often struggle with noisy pseudo-labels and domain adaptation, we propose Flick. Unlike prior methods that rely on generic multi-cluster pseudo-labelling or complex cascading architectures, Flick leverages the fundamental insight that distilling high-confidence pseudo-labels from a broader set of initial clusters can dramatically improve pseudo-label quality, particularly for linguistically diverse, low-resource settings. Flick introduces a novel pseudo-label refinement component, a departure from traditional pseudo-labelling strategies by identifying and leveraging top-performing pseudo-label clusters. This component specifically learns to distil highly reliable pseudo-labels from an initial broad set by focusing on single-cluster cohesion and leveraging an adaptive top-k selection mechanism. This targeted refinement process is crucial for mitigating the propagation of errors inherent in low-resource data, allowing for robust fine-tuning of pre-trained language models with only a handful of true labels. We demonstrate Flick's efficacy across 14 diverse datasets, encompassing challenging low-resource languages such as Arabic, Urdu, and Setswana, alongside English, showcasing its superior performance and adaptability. 

**Abstract (ZH)**: 使用最少监督训练深度学习网络由于其减少对大量标注数据依赖的潜力而受到广泛关注。不同于以往方法依赖通用多集群伪标签或复杂级联架构，Flick通过从更广泛的初始集群中提取高置信度伪标签，显著提高伪标签质量，特别是在语言多样且资源稀缺的环境中。Flick引入了一种新的伪标签精炼组件，通过识别和利用表现最佳的伪标签集群，该组件专注于单集群凝聚力，并利用自适应top-k选择机制来提取高度可靠的伪标签。这一有针对性的精炼过程对于减轻低资源数据中错误的传播至关重要，允许仅使用少量真实标签对预训练语言模型进行稳健调优。我们展示了Flick在14个多样化的数据集上的有效性，涵盖了如阿拉伯语、乌尔都语和塞茨瓦纳语等具有挑战性的低资源语言，以及英语，展示了其优越的性能和适应能力。 

---
# RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding 

**Title (ZH)**: RT-VC: 实时零样本语音转换与语音articulatory编码 

**Authors**: Yisi Liu, Chenyang Wang, Hanjo Kim, Raniya Khan, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10289)  

**Abstract**: Voice conversion has emerged as a pivotal technology in numerous applications ranging from assistive communication to entertainment. In this paper, we present RT-VC, a zero-shot real-time voice conversion system that delivers ultra-low latency and high-quality performance. Our approach leverages an articulatory feature space to naturally disentangle content and speaker characteristics, facilitating more robust and interpretable voice transformations. Additionally, the integration of differentiable digital signal processing (DDSP) enables efficient vocoding directly from articulatory features, significantly reducing conversion latency. Experimental evaluations demonstrate that, while maintaining synthesis quality comparable to the current state-of-the-art (SOTA) method, RT-VC achieves a CPU latency of 61.4 ms, representing a 13.3\% reduction in latency. 

**Abstract (ZH)**: 零样本实时语音转换系统RT-VC：超低延迟与高质量性能 

---
# ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs 

**Title (ZH)**: ClusterUCB: 效率导向的梯度基数据选择方法用于LLM的目标微调 

**Authors**: Zige Wang, Qi Zhu, Fei Mi, Minghui Xu, Ruochun Jin, Wenjing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10288)  

**Abstract**: Gradient-based data influence approximation has been leveraged to select useful data samples in the supervised fine-tuning of large language models. However, the computation of gradients throughout the fine-tuning process requires too many resources to be feasible in practice. In this paper, we propose an efficient gradient-based data selection framework with clustering and a modified Upper Confidence Bound (UCB) algorithm. Based on the intuition that data samples with similar gradient features will have similar influences, we first perform clustering on the training data pool. Then, we frame the inter-cluster data selection as a constrained computing budget allocation problem and consider it a multi-armed bandit problem. A modified UCB algorithm is leveraged to solve this problem. Specifically, during the iterative sampling process, historical data influence information is recorded to directly estimate the distributions of each cluster, and a cold start is adopted to balance exploration and exploitation. Experimental results on various benchmarks show that our proposed framework, ClusterUCB, can achieve comparable results to the original gradient-based data selection methods while greatly reducing computing consumption. 

**Abstract (ZH)**: 基于聚类和修改的Upper Confidence Bound算法的高效梯度引导数据选择框架 

---
# Discrete Audio Tokens: More Than a Survey! 

**Title (ZH)**: 离散音频令牌：远不止一项综述！ 

**Authors**: Pooneh Mousavi, Gallil Maimon, Adel Moumen, Darius Petermann, Jiatong Shi, Haibin Wu, Haici Yang, Anastasia Kuznetsova, Artem Ploujnikov, Ricard Marxer, Bhuvana Ramabhadran, Benjamin Elizalde, Loren Lugosch, Jinyu Li, Cem Subakan, Phil Woodland, Minje Kim, Hung-yi Lee, Shinji Watanabe, Yossi Adi, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2506.10274)  

**Abstract**: Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream this http URL provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: this https URL. 

**Abstract (ZH)**: 离散音频令牌是紧凑的表示，旨在保留感知质量、音素内容和说话人特征，同时实现高效的存储和推断，并在多种下游任务中获得竞争力的性能。它们为现代大型语言模型（LLMs）集成语音和音频提供了实用的替代方案。随着对基于令牌的音频处理兴趣的增长，各种令牌化方法相继出现，且已有若干综述对领域内最新进展进行了综述。然而，现有研究通常专注于特定领域或任务，缺乏跨不同基准的统一比较。本文对离散音频令牌化器进行了系统性综述和基准测试，涵盖了三个领域：语音、音乐和通用音频。我们基于编码器-解码器、量化技术、训练范式、可流式性和应用领域提出了令牌化方法的分类框架。我们在重建、下游性能和声学语言建模等多个基准上评估了令牌化器，并通过受控的消融研究分析了权衡取舍。我们的研究发现强调了关键局限性、实用考虑和公开挑战，为该快速发展的领域未来研究提供了见解和指导。欲了解更多详情，包括我们主要结果和令牌化数据库，请参阅我们的网站：this http URL。 

---
# Do Language Models Have Bayesian Brains? Distinguishing Stochastic and Deterministic Decision Patterns within Large Language Models 

**Title (ZH)**: 语言模型具备贝叶斯大脑吗？区分大型语言模型中的随机性和确定性决策模式 

**Authors**: Andrea Yaoyun Cui, Pengfei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10268)  

**Abstract**: Language models are essentially probability distributions over token sequences. Auto-regressive models generate sentences by iteratively computing and sampling from the distribution of the next token. This iterative sampling introduces stochasticity, leading to the assumption that language models make probabilistic decisions, similar to sampling from unknown distributions. Building on this assumption, prior research has used simulated Gibbs sampling, inspired by experiments designed to elicit human priors, to infer the priors of language models. In this paper, we revisit a critical question: Do language models possess Bayesian brains? Our findings show that under certain conditions, language models can exhibit near-deterministic decision-making, such as producing maximum likelihood estimations, even with a non-zero sampling temperature. This challenges the sampling assumption and undermines previous methods for eliciting human-like priors. Furthermore, we demonstrate that without proper scrutiny, a system with deterministic behavior undergoing simulated Gibbs sampling can converge to a "false prior." To address this, we propose a straightforward approach to distinguish between stochastic and deterministic decision patterns in Gibbs sampling, helping to prevent the inference of misleading language model priors. We experiment on a variety of large language models to identify their decision patterns under various circumstances. Our results provide key insights in understanding decision making of large language models. 

**Abstract (ZH)**: 语言模型本质上是词元序列的概率分布。自回归模型通过迭代计算和从下一个词元的概率分布中采样生成句子。这种迭代采样引入了随机性，使得人们假设语言模型做出概率决策，类似于从未知分布中采样。基于这种假设，先前的研究利用受激发人类先验实验启发的模拟吉布斯采样方法来推断语言模型的先验。在本文中，我们重新审视了一个核心问题：语言模型是否具有贝叶斯大脑？我们的研究发现，在某些条件下，语言模型可以表现出近乎确定性的决策模式，例如产生极大似然估计，即使在非零采样温度下也是如此。这挑战了采样假设，并削弱了之前的方法来推断人类似然先验的有效性。此外，我们证明了在未经适当审查的情况下，具有确定性行为的系统在模拟吉布斯采样过程中可能会收敛到“虚假先验”。为此，我们提出了一种简单的方法来区分吉布斯采样中的随机性和确定性决策模式，有助于防止错误地推断语言模型的先验。我们对多种大型语言模型进行了实验，以在不同情况下识别其决策模式。我们的结果为理解大型语言模型的决策模式提供了关键见解。 

---
# Extended Creativity: A Conceptual Framework for Understanding Human-AI Creative Relations 

**Title (ZH)**: 扩展创造力：理解人机创意关系的概念框架 

**Authors**: Andrea Gaggioli, Sabrina Bartolotta, Andrea Ubaldi, Katusha Gerardini, Eleonora Diletta Sarcinella, Alice Chirico  

**Link**: [PDF](https://arxiv.org/pdf/2506.10249)  

**Abstract**: Artificial Intelligence holds significant potential to enhance human creativity. However, achieving this vision requires a clearer understanding of how such enhancement can be effectively realized. Adopting the perspective of distributed creativity, we identify three primary modes through which AI can contribute to creative processes: Support, where AI acts as a tool; Synergy, where AI and humans collaborate in complementary ways; and Symbiosis, where human and AI cognition become so integrated that they form a unified creative system. These modes are defined along two key dimensions: the level of technical autonomy exhibited by the AI system and the degree of perceived agency attributed to it. We examine how each configuration influences different levels of creativity - from everyday problem-solving to paradigm-shifting innovation - and discuss the theoretical, ethical, and design implications. 

**Abstract (ZH)**: 人工智能在增强人类创造力方面具有重要的潜力。然而，实现这一愿景需要对如何有效地实现这种增强有更清晰的理解。从分布式创造力的视角出发，我们确定了人工智能可以通过三种主要方式贡献于创意过程：支持模式，其中人工智能作为工具发挥作用；协同模式，其中人工智能与人类以互补的方式合作；共生模式，其中人类与人工智能的认知深度融合，形成统一的创意系统。这些模式沿着两个关键维度定义：人工智能系统展现的技术自主性水平以及对其感知自主性的程度。我们探讨了每种配置如何影响不同层次的创造力——从日常问题解决到范式转变的创新——并讨论了相关的理论、伦理和设计意义。 

---
# ToxSyn-PT: A Large-Scale Synthetic Dataset for Hate Speech Detection in Portuguese 

**Title (ZH)**: ToxSyn-PT： Portuguese 垃圾言论检测的大规模合成数据集 

**Authors**: Iago Alves Brito, Julia Soares Dollis, Fernanda Bufon Färber, Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2506.10245)  

**Abstract**: We present ToxSyn-PT, the first large-scale Portuguese corpus that enables fine-grained hate-speech classification across nine legally protected minority groups. The dataset contains 53,274 synthetic sentences equally distributed between minorities groups and toxicity labels. ToxSyn-PT is created through a novel four-stage pipeline: (1) a compact, manually curated seed; (2) few-shot expansion with an instruction-tuned LLM; (3) paraphrase-based augmentation; and (4) enrichment, plus additional neutral texts to curb overfitting to group-specific cues. The resulting corpus is class-balanced, stylistically diverse, and free from the social-media domain that dominate existing Portuguese datasets. Despite domain differences with traditional benchmarks, experiments on both binary and multi-label classification on the corpus yields strong results across five public Portuguese hate-speech datasets, demonstrating robust generalization even across domain boundaries. The dataset is publicly released to advance research on synthetic data and hate-speech detection in low-resource settings. 

**Abstract (ZH)**: ToxSyn-PT：首个支持九个法律保护少数群体细粒度仇恨言论分类的大规模葡萄牙语语料库 

---
# Prompt Attacks Reveal Superficial Knowledge Removal in Unlearning Methods 

**Title (ZH)**: 提示攻击揭示去学习方法中浅层知识删除问题 

**Authors**: Yeonwoo Jang, Shariqah Hossain, Ashwin Sreevatsa, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10236)  

**Abstract**: In this work, we show that some machine unlearning methods may fail when subjected to straightforward prompt attacks. We systematically evaluate eight unlearning techniques across three model families, and employ output-based, logit-based, and probe analysis to determine to what extent supposedly unlearned knowledge can be retrieved. While methods like RMU and TAR demonstrate robust unlearning, ELM remains vulnerable to specific prompt attacks (e.g., Hindi filler text in original prompt recovering 57.3% accuracy). Our logit analysis also confirms that unlearned models are generally not hiding knowledge by modifying the way the answer is formatted, as the correlation between output and logit accuracy is strong. These results challenge prevailing assumptions about unlearning effectiveness and highlight the need for evaluation frameworks that can reliably distinguish between true knowledge removal and superficial output suppression. We also publicly make available our evaluation framework to easily evaluate prompting techniques to retrieve unlearning knowledge. 

**Abstract (ZH)**: 在本研究中，我们表明一些机器未学习方法在面对直接提示攻击时可能会失效。我们系统性地评估了八种未学习技术在三种模型家族中的效果，并通过输出分析、logit分析和探针分析来确定已声称未学习的知识能被恢复到何种程度。虽然像RMU和TAR这样的方法显示出较强的未学习能力，但ELM仍然对特定提示攻击（如原始提示中的印地语填充文本恢复57.3%的准确率）保持脆弱性。我们的logit分析也证实，未学习模型通常并不是通过改变答案格式的方式来隐藏知识，因为输出和logit准确率之间的相关性很强。这些结果挑战了现有的未学习效果假设，并强调了需要建立可靠的评估框架来区分真正的知识移除与表面上的输出抑制。我们还公开发布了我们的评估框架，以便于评估提示技术以检索未学习知识。 

---
# LaMAGIC2: Advanced Circuit Formulations for Language Model-Based Analog Topology Generation 

**Title (ZH)**: LaMAGIC2：基于语言模型的模拟拓扑生成高级电路公式 

**Authors**: Chen-Chia Chang, Wan-Hsuan Lin, Yikang Shen, Yiran Chen, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10235)  

**Abstract**: Automation of analog topology design is crucial due to customized requirements of modern applications with heavily manual engineering efforts. The state-of-the-art work applies a sequence-to-sequence approach and supervised finetuning on language models to generate topologies given user specifications. However, its circuit formulation is inefficient due to O(|V |2) token length and suffers from low precision sensitivity to numeric inputs. In this work, we introduce LaMAGIC2, a succinct float-input canonical formulation with identifier (SFCI) for language model-based analog topology generation. SFCI addresses these challenges by improving component-type recognition through identifier-based representations, reducing token length complexity to O(|V |), and enhancing numeric precision sensitivity for better performance under tight tolerances. Our experiments demonstrate that LaMAGIC2 achieves 34% higher success rates under a tight tolerance of 0.01 and 10X lower MSEs compared to a prior method. LaMAGIC2 also exhibits better transferability for circuits with more vertices with up to 58.5% improvement. These advancements establish LaMAGIC2 as a robust framework for analog topology generation. 

**Abstract (ZH)**: 基于语言模型的模拟拓扑生成的简洁浮点输入标准化表示（LaMAGIC2） 

---
# ScoreMix: Improving Face Recognition via Score Composition in Diffusion Generators 

**Title (ZH)**: ScoreMix: 通过扩散生成器中的评分合成提高面部识别性能 

**Authors**: Parsa Rahimi, Sebastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2506.10226)  

**Abstract**: In this paper, we propose ScoreMix, a novel yet simple data augmentation strategy leveraging the score compositional properties of diffusion models to enhance discriminator performance, particularly under scenarios with limited labeled data. By convexly mixing the scores from different class-conditioned trajectories during diffusion sampling, we generate challenging synthetic samples that significantly improve discriminative capabilities in all studied benchmarks. We systematically investigate class-selection strategies for mixing and discover that greater performance gains arise when combining classes distant in the discriminator's embedding space, rather than close in the generator's condition space. Moreover, we empirically show that, under standard metrics, the correlation between the generator's learned condition space and the discriminator's embedding space is minimal. Our approach achieves notable performance improvements without extensive parameter searches, demonstrating practical advantages for training discriminative models while effectively mitigating problems regarding collections of large datasets. Paper website: this https URL 

**Abstract (ZH)**: 在本文中，我们提出了一种新颖且简单的数据增强策略ScoreMix，该策略利用扩散模型的分数组成特性来增强判别器性能，尤其是在标记数据有限的情况下。通过在扩散采样过程中凸性混合不同类条件轨迹的分数，我们生成了具有挑战性的合成样本，这些样本在所有研究的基准测试中显著提高了辨别能力。我们系统地研究了混合的类选择策略，并发现当组合在判别器嵌入空间中距离较远的类时，可以获得更大的性能提升，而非在生成器条件空间中接近的类。此外，我们实证表明，在标准指标下，生成器学习的条件空间与判别器的嵌入空间之间的相关性最小。我们的方法在无需进行广泛的参数搜索的情况下取得了显著性能提升，显示了在训练辨别模型方面的实用优势，并有效缓解了大规模数据集收集的问题。论文网站：this https URL 

---
# Fine-Grained control over Music Generation with Activation Steering 

**Title (ZH)**: 基于激活导向的细粒度音乐生成控制 

**Authors**: Dipanshu Panda, Jayden Koshy Joe, Harshith M R, Swathi Narashiman, Pranay Mathur, Anish Veerakumar, Aniruddh Krishna, Keerthiharan A  

**Link**: [PDF](https://arxiv.org/pdf/2506.10225)  

**Abstract**: We present a method for fine-grained control over music generation through inference-time interventions on an autoregressive generative music transformer called MusicGen. Our approach enables timbre transfer, style transfer, and genre fusion by steering the residual stream using weights of linear probes trained on it, or by steering the attention layer activations in a similar manner. We observe that modelling this as a regression task provides improved performance, hypothesizing that the mean-squared-error better preserve meaningful directional information in the activation space. Combined with the global conditioning offered by text prompts in MusicGen, our method provides both global and local control over music generation. Audio samples illustrating our method are available at our demo page. 

**Abstract (ZH)**: 我们提出了一种通过在自回归生成音乐变压器MusicGen中进行推理时干预以实现细粒度音乐生成控制的方法。该方法通过使用在残差流上训练的线性探针的权重来引导音色转移、风格转移和流派融合，或将注意力层激活以类似方式引导。我们观察到将此建模为回归任务可以提高性能，假设均方误差更好地保留了激活空间中的有意义方向信息。结合MusicGen中文本提示提供的全局条件，该方法为音乐生成提供了全局和局部控制。我们在演示页面上有音频示例展示该方法。 

---
# Cross-Learning Between ECG and PCG: Exploring Common and Exclusive Characteristics of Bimodal Electromechanical Cardiac Waveforms 

**Title (ZH)**: ECG与PCG之间的交叉学习：探讨双模电磁心脏波形的共性和特异性特征 

**Authors**: Sajjad Karimi, Amit J. Shah, Gari D. Clifford, Reza Sameni  

**Link**: [PDF](https://arxiv.org/pdf/2506.10212)  

**Abstract**: Simultaneous electrocardiography (ECG) and phonocardiogram (PCG) provide a comprehensive, multimodal perspective on cardiac function by capturing the heart's electrical and mechanical activities, respectively. However, the distinct and overlapping information content of these signals, as well as their potential for mutual reconstruction and biomarker extraction, remains incompletely understood, especially under varying physiological conditions and across individuals.
In this study, we systematically investigate the common and exclusive characteristics of ECG and PCG using the EPHNOGRAM dataset of simultaneous ECG-PCG recordings during rest and exercise. We employ a suite of linear and nonlinear machine learning models, including non-causal LSTM networks, to reconstruct each modality from the other and analyze the influence of causality, physiological state, and cross-subject variability. Our results demonstrate that nonlinear models, particularly non-causal LSTM, provide superior reconstruction performance, with reconstructing ECG from PCG proving more tractable than the reverse. Exercise and cross-subject scenarios present significant challenges, but envelope-based modeling that utilizes instantaneous amplitude features substantially improves cross-subject generalizability for cross-modal learning. Furthermore, we demonstrate that clinically relevant ECG biomarkers, such as fiducial points and QT intervals, can be estimated from PCG in cross-subject settings.
These findings advance our understanding of the relationship between electromechanical cardiac modalities, in terms of both waveform characteristics and the timing of cardiac events, with potential applications in novel multimodal cardiac monitoring technologies. 

**Abstract (ZH)**: 同时记录心电图和心音图提供了一种综合的多模态视角，用于捕捉心脏的电活动和机械活动。然而，这些信号的独特和重叠信息内容，以及它们的相互重构和生物标志物提取的潜力，尤其是在不同生理条件下和不同个体之间的理解仍然不够充分。 

---
# TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games 

**Title (ZH)**: TTT-Bench：一种基于简单新颖的井字游戏评估推理能力的基准测试 

**Authors**: Prakamya Mishra, Jiang Liu, Jialian Wu, Xiaodong Yu, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.10209)  

**Abstract**: Large reasoning models (LRMs) have demonstrated impressive reasoning capabilities across a broad range of tasks including Olympiad-level mathematical problems, indicating evidence of their complex reasoning abilities. While many reasoning benchmarks focus on the STEM domain, the ability of LRMs to reason correctly in broader task domains remains underexplored. In this work, we introduce \textbf{TTT-Bench}, a new benchmark that is designed to evaluate basic strategic, spatial, and logical reasoning abilities in LRMs through a suite of four two-player Tic-Tac-Toe-style games that humans can effortlessly solve from a young age. We propose a simple yet scalable programmatic approach for generating verifiable two-player game problems for TTT-Bench. Although these games are trivial for humans, they require reasoning about the intentions of the opponent, as well as the game board's spatial configurations, to ensure a win. We evaluate a diverse set of state-of-the-art LRMs, and \textbf{discover that the models that excel at hard math problems frequently fail at these simple reasoning games}. Further testing reveals that our evaluated reasoning models score on average $\downarrow$ 41\% \& $\downarrow$ 5\% lower on TTT-Bench compared to MATH 500 \& AIME 2024 respectively, with larger models achieving higher performance using shorter reasoning traces, where most of the models struggle on long-term strategic reasoning situations on simple and new TTT-Bench tasks. 

**Abstract (ZH)**: 大型推理模型（LRMs）在包括奥林匹克级别数学问题在内的广泛任务中展示了令人印象深刻的推理能力，表明了它们复杂的推理能力。尽管许多推理基准关注STEM领域，但LRMs在更广泛任务域中的正确推理能力仍然未被充分探索。在本工作中，我们引入了TTT-Bench，这是一个新的基准，旨在通过一系列人类从小就能轻易解决的四款两人对弈的井字游戏来评估LRMs的基本战略、空间和逻辑推理能力。我们提出了一种简单而可扩展的编程方法，用于生成可验证的两人游戏问题以供TTT-Bench使用。虽然这些游戏对人类来说是简单的，但它们要求玩家不仅要考虑对手的意图，还要考虑游戏板的空间配置，以确保胜利。我们评估了一组多样化的最先进的LRMs，并发现擅长解决难题的模型在这些简单的推理游戏中经常表现不佳。进一步测试显示，我们评估的推理模型在TTT-Bench上的得分分别比MATH 500和AIME 2024低$\downarrow$ 41\% 和 $\downarrow$ 5%，其中较大的模型通过较短的推理过程获得更高的性能，而大多数模型在简单和全新的TTT-Bench任务中的长期战略推理情况上挣扎。 

---
# Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment 

**Title (ZH)**: 可扩展的非同构3D分子生成通过旋转对齐 

**Authors**: Yuhui Ding, Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.10186)  

**Abstract**: Equivariant diffusion models have achieved impressive performance in 3D molecule generation. These models incorporate Euclidean symmetries of 3D molecules by utilizing an SE(3)-equivariant denoising network. However, specialized equivariant architectures limit the scalability and efficiency of diffusion models. In this paper, we propose an approach that relaxes such equivariance constraints. Specifically, our approach learns a sample-dependent SO(3) transformation for each molecule to construct an aligned latent space. A non-equivariant diffusion model is then trained over the aligned representations. Experimental results demonstrate that our approach performs significantly better than previously reported non-equivariant models. It yields sample quality comparable to state-of-the-art equivariant diffusion models and offers improved training and sampling efficiency. Our code is available at this https URL 

**Abstract (ZH)**: 三维分子生成中不变性扩散模型已在各个方面取得了显著成果。这些模型通过利用一个SE(3)-不变性去噪网络，将3D分子的欧几里得对称性纳入其中。然而，专门设计的不变性架构限制了扩散模型的可扩展性和效率。在本文中，我们提出了一种放宽此类不变性约束的方法。具体来说，我们的方法为每个分子学习一个样本依赖的SO(3)变换，以构建对齐的潜在空间。然后在一个对齐的表征上训练一个非不变性扩散模型。实验结果显示，我们的方法在样本质量上明显优于之前报告的非不变性模型，并提供了更好的训练和采样效率。我们的代码可在以下链接获得：this https URL 

---
# Optimizing Genetic Algorithms with Multilayer Perceptron Networks for Enhancing TinyFace Recognition 

**Title (ZH)**: 使用多层感知器网络优化遗传算法以提高TinyFace识别性能 

**Authors**: Mohammad Subhi Al-Batah, Mowafaq Salem Alzboon, Muhyeeddin Alqaraleh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10184)  

**Abstract**: This study conducts an empirical examination of MLP networks investigated through a rigorous methodical experimentation process involving three diverse datasets: TinyFace, Heart Disease, and Iris. Study Overview: The study includes three key methods: a) a baseline training using the default settings for the Multi-Layer Perceptron (MLP), b) feature selection using Genetic Algorithm (GA) based refinement c) Principal Component Analysis (PCA) based dimension reduction. The results show important information on how such techniques affect performance. While PCA had showed benefits in low-dimensional and noise-free datasets GA consistently increased accuracy in complex datasets by accurately identifying critical features. Comparison reveals that feature selection and dimensionality reduction play interdependent roles in enhancing MLP performance. The study contributes to the literature on feature engineering and neural network parameter optimization, offering practical guidelines for a wide range of machine learning tasks 

**Abstract (ZH)**: 本研究通过严格的实验方法对三种不同数据集（TinyFace、Heart Disease和Iris）下的MLP网络进行了实证考察。研究概述：研究包括三种关键方法：a) 使用默认设置训练Multi-Layer Perceptron (MLP)基线模型，b) 使用遗传算法（GA）进行特征选择精炼，c) 使用主成分分析（PCA）进行维度缩减。结果表明这些技术对性能的影响信息。虽然PCA在低维度和无噪声数据集中显示出优势，但GA在复杂数据集中通过准确识别关键特征持续增加了准确性。比较表明，特征选择和维度缩减在提升MLP性能方面相互依存。本研究为特征工程和神经网络参数优化文献做出了贡献，为广泛领域的机器学习任务提供了实用指南。 

---
# A Comparative Study of Machine Learning Techniques for Early Prediction of Diabetes 

**Title (ZH)**: 机器学习技术在糖尿病早期预测中的比较研究 

**Authors**: Mowafaq Salem Alzboon, Mohammad Al-Batah, Muhyeeddin Alqaraleh, Ahmad Abuashour, Ahmad Fuad Bader  

**Link**: [PDF](https://arxiv.org/pdf/2506.10180)  

**Abstract**: In many nations, diabetes is becoming a significant health problem, and early identification and control are crucial. Using machine learning algorithms to predict diabetes has yielded encouraging results. Using the Pima Indians Diabetes dataset, this study attempts to evaluate the efficacy of several machine-learning methods for diabetes prediction. The collection includes information on 768 patients, such as their ages, BMIs, and glucose levels. The techniques assessed are Logistic Regression, Decision Tree, Random Forest, k-Nearest Neighbors, Naive Bayes, Support Vector Machine, Gradient Boosting, and Neural Network. The findings indicate that the Neural Network algorithm performed the best, with an accuracy of 78.57 percent, followed by the Random Forest method, with an accuracy of 76.30 percent. The study implies that machine learning algorithms can aid diabetes prediction and be an efficient early detection tool. 

**Abstract (ZH)**: 在许多国家，糖尿病已成为一个重要的健康问题，早期识别和控制至关重要。使用机器学习算法预测糖尿病取得了一些令人鼓舞的结果。本研究使用Pima Indians Diabetes数据集评估了几种机器学习方法在糖尿病预测中的有效性。该数据集包含了768名患者的信息，如年龄、BMI和血糖水平。评估的方法包括逻辑回归、决策树、随机森林、k-近邻、朴素贝叶斯、支持向量机、梯度提升和神经网络。研究结果表明，神经网络算法表现最佳，准确率为78.57%，紧随其后的是随机森林方法，准确率为76.30%。本研究表明，机器学习算法可以辅助糖尿病预测，并成为一种有效的早期检测工具。 

---
# SPARKE: Scalable Prompt-Aware Diversity Guidance in Diffusion Models via RKE Score 

**Title (ZH)**: SPARKE: 扩展的基于提示的多样性引导在扩散模型中通过RKE分数 

**Authors**: Mohammad Jalali, Haoyu Lei, Amin Gohari, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2506.10173)  

**Abstract**: Diffusion models have demonstrated remarkable success in high-fidelity image synthesis and prompt-guided generative modeling. However, ensuring adequate diversity in generated samples of prompt-guided diffusion models remains a challenge, particularly when the prompts span a broad semantic spectrum and the diversity of generated data needs to be evaluated in a prompt-aware fashion across semantically similar prompts. Recent methods have introduced guidance via diversity measures to encourage more varied generations. In this work, we extend the diversity measure-based approaches by proposing the Scalable Prompt-Aware Rény Kernel Entropy Diversity Guidance (SPARKE) method for prompt-aware diversity guidance. SPARKE utilizes conditional entropy for diversity guidance, which dynamically conditions diversity measurement on similar prompts and enables prompt-aware diversity control. While the entropy-based guidance approach enhances prompt-aware diversity, its reliance on the matrix-based entropy scores poses computational challenges in large-scale generation settings. To address this, we focus on the special case of Conditional latent RKE Score Guidance, reducing entropy computation and gradient-based optimization complexity from the $O(n^3)$ of general entropy measures to $O(n)$. The reduced computational complexity allows for diversity-guided sampling over potentially thousands of generation rounds on different prompts. We numerically test the SPARKE method on several text-to-image diffusion models, demonstrating that the proposed method improves the prompt-aware diversity of the generated data without incurring significant computational costs. We release our code on the project page: this https URL 

**Abstract (ZH)**: 基于启发式的可扩展提示感知 Rényi 核熵多样性引导方法（SPARKE） 

---
# A Navigation Framework Utilizing Vision-Language Models 

**Title (ZH)**: 利用视觉语言模型的导航框架 

**Authors**: Yicheng Duan, Kaiyu tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10172)  

**Abstract**: Vision-and-Language Navigation (VLN) presents a complex challenge in embodied AI, requiring agents to interpret natural language instructions and navigate through visually rich, unfamiliar environments. Recent advances in large vision-language models (LVLMs), such as CLIP and Flamingo, have significantly improved multimodal understanding but introduced new challenges related to computational cost and real-time deployment. In this project, we propose a modular, plug-and-play navigation framework that decouples vision-language understanding from action planning. By integrating a frozen vision-language model, Qwen2.5-VL-7B-Instruct, with lightweight planning logic, we aim to achieve flexible, fast, and adaptable navigation without extensive model fine-tuning. Our framework leverages prompt engineering, structured history management, and a two-frame visual input strategy to enhance decision-making continuity across navigation steps. We evaluate our system on the Room-to-Room benchmark within the VLN-CE setting using the Matterport3D dataset and Habitat-Lab simulation environment. Although our initial results reveal challenges in generalizing to unseen environments under strict evaluation settings, our modular approach lays a foundation for scalable and efficient navigation systems, highlighting promising directions for future improvement through enhanced environmental priors and expanded multimodal input integration. 

**Abstract (ZH)**: 基于视觉-语言的导航（VLN）在实体AI中提出了复杂挑战，要求代理解释自然语言指示并导航通过视觉丰富且不熟悉的环境。大型视觉-语言模型（LVLM）的最新进展，例如CLIP和Flamingo，显著提高了多模态理解能力，但也引入了计算成本和实时部署的新挑战。在本项目中，我们提出了一种模块化、即插即用的导航框架，将视觉-语言理解与动作规划解耦。通过将 frozen 视觉-语言模型 Qwen2.5-VL-7B-Instruct 与轻量级规划逻辑集成，我们旨在在无需大量模型微调的情况下实现灵活、快速且适应性强的导航。我们的框架利用提示工程、结构化历史管理以及双帧视觉输入策略来增强导航步骤间的决策连贯性。我们使用Matterport3D数据集和Habitat-Lab仿真环境，在VLN-CE设置下的Room-to-Room基准上评估了我们的系统。尽管初始结果在严格评估条件下显示了向未见环境泛化的挑战，但我们的模块化方法为可扩展和高效的导航系统奠定了基础，揭示了通过增强环境先验和扩展多模态输入集成的未来改进方向。 

---
# Disclosure Audits for LLM Agents 

**Title (ZH)**: LLM代理的披露审计 

**Authors**: Saswat Das, Jameson Sandler, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10171)  

**Abstract**: Large Language Model agents have begun to appear as personal assistants, customer service bots, and clinical aides. While these applications deliver substantial operational benefits, they also require continuous access to sensitive data, which increases the likelihood of unauthorized disclosures. This study proposes an auditing framework for conversational privacy that quantifies and audits these risks. The proposed Conversational Manipulation for Privacy Leakage (CMPL) framework, is an iterative probing strategy designed to stress-test agents that enforce strict privacy directives. Rather than focusing solely on a single disclosure event, CMPL simulates realistic multi-turn interactions to systematically uncover latent vulnerabilities. Our evaluation on diverse domains, data modalities, and safety configurations demonstrate the auditing framework's ability to reveal privacy risks that are not deterred by existing single-turn defenses. In addition to introducing CMPL as a diagnostic tool, the paper delivers (1) an auditing procedure grounded in quantifiable risk metrics and (2) an open benchmark for evaluation of conversational privacy across agent implementations. 

**Abstract (ZH)**: 大规模语言模型代理已经开始作为个人助手、客户服务机器人和临床辅助人员出现。虽然这些应用带来了显著的操作效益，但也需要持续访问敏感数据，增加了未经授权披露信息的可能性。本研究提出了一种对话隐私审计框架，用于量化和审查这些风险。所提出的对话操控以隐私泄露（CMPL）框架是一种迭代探测策略，旨在测试执行严格隐私指令的代理。CMPL不仅模拟了单一披露事件，还模拟了现实的多轮交互，系统地揭示潜在的漏洞。对不同领域、数据模态和安全配置的评估表明，该审计框架能够揭示现有单一交互防御无法阻止的隐私风险。除了介绍CMPL作为诊断工具外，该论文还提供了（1）基于可量化的风险指标的审计流程以及（2）跨代理实现的对话隐私评估的开源基准。 

---
# Can LLMs Generate Good Stories? Insights and Challenges from a Narrative Planning Perspective 

**Title (ZH)**: LLM能生成好的故事吗？从叙事规划视角的见解与挑战 

**Authors**: Yi Wang, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2506.10161)  

**Abstract**: Story generation has been a prominent application of Large Language Models (LLMs). However, understanding LLMs' ability to produce high-quality stories remains limited due to challenges in automatic evaluation methods and the high cost and subjectivity of manual evaluation. Computational narratology offers valuable insights into what constitutes a good story, which has been applied in the symbolic narrative planning approach to story generation. This work aims to deepen the understanding of LLMs' story generation capabilities by using them to solve narrative planning problems. We present a benchmark for evaluating LLMs on narrative planning based on literature examples, focusing on causal soundness, character intentionality, and dramatic conflict. Our experiments show that GPT-4 tier LLMs can generate causally sound stories at small scales, but planning with character intentionality and dramatic conflict remains challenging, requiring LLMs trained with reinforcement learning for complex reasoning. The results offer insights on the scale of stories that LLMs can generate while maintaining quality from different aspects. Our findings also highlight interesting problem solving behaviors and shed lights on challenges and considerations for applying LLM narrative planning in game environments. 

**Abstract (ZH)**: 大语言模型在故事生成方面的能力探讨：基于叙事规划的评估基准与启示 

---
# Measuring Corporate Human Capital Disclosures: Lexicon, Data, Code, and Research Opportunities 

**Title (ZH)**: 测量企业人力资本披露：词汇、数据、代码和研究机会 

**Authors**: Elizabeth Demers, Victor Xiaoqi Wang, Kean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10155)  

**Abstract**: Human capital (HC) is increasingly important to corporate value creation. Unlike other assets, however, HC is not currently subject to well-defined measurement or disclosure rules. We use a machine learning algorithm (word2vec) trained on a confirmed set of HC disclosures to develop a comprehensive list of HC-related keywords classified into five subcategories (DEI; health and safety; labor relations and culture; compensation and benefits; and demographics and other) that capture the multidimensional nature of HC management. We share our lexicon, corporate HC disclosures, and the Python code used to develop the lexicon, and we provide detailed examples of using our data and code, including for fine-tuning a BERT model. Researchers can use our HC lexicon (or modify the code to capture another construct of interest) with their samples of corporate communications to address pertinent HC questions. We close with a discussion of future research opportunities related to HC management and disclosure. 

**Abstract (ZH)**: 人力资本（HC）在企业价值创造中日益重要。然而，与其他资产不同，HC目前并没有明确的计量或披露规则。我们使用一种机器学习算法（word2vec），基于确认的HC披露数据集，开发了一个全面的HC相关关键词列表，按照五个子类别（DEI；健康与安全；劳动关系与文化；薪酬与福利；以及人口统计和其他）进行分类，以捕捉HC管理的多维性质。我们分享了我们的词汇表、公司HC披露数据以及开发词汇表所使用的Python代码，并提供了使用我们数据和代码的详细示例，包括对BERT模型进行微调。研究人员可以使用我们的HC词汇表（或修改代码以捕获另一个感兴趣的构造）与其公司的沟通样本相结合，以解决相关的人力资本问题。我们最后讨论了与HC管理和披露相关的未来研究机会。 

---
# Unsupervised Elicitation of Language Models 

**Title (ZH)**: 无监督语言模型 elicitation 方法 

**Authors**: Jiaxin Wen, Zachary Ankner, Arushi Somani, Peter Hase, Samuel Marks, Jacob Goldman-Wetzler, Linda Petrini, Henry Sleight, Collin Burns, He He, Shi Feng, Ethan Perez, Jan Leike  

**Link**: [PDF](https://arxiv.org/pdf/2506.10139)  

**Abstract**: To steer pretrained language models for downstream tasks, today's post-training paradigm relies on humans to specify desired behaviors. However, for models with superhuman capabilities, it is difficult or impossible to get high-quality human supervision. To address this challenge, we introduce a new unsupervised algorithm, Internal Coherence Maximization (ICM), to fine-tune pretrained language models on their own generated labels, \emph{without external supervision}. On GSM8k-verification, TruthfulQA, and Alpaca reward modeling tasks, our method matches the performance of training on golden supervision and outperforms training on crowdsourced human supervision. On tasks where LMs' capabilities are strongly superhuman, our method can elicit those capabilities significantly better than training on human labels. Finally, we show that our method can improve the training of frontier LMs: we use our method to train an unsupervised reward model and use reinforcement learning to train a Claude 3.5 Haiku-based assistant. Both the reward model and the assistant outperform their human-supervised counterparts. 

**Abstract (ZH)**: 为了Fine-tune预训练语言模型以适应下游任务，当前的后训练范式依赖人类指定所需行为。但对于具有超人类能力的模型，获得高质量的人类监督变得困难或不可能。为了解决这一挑战，我们引入了一种新的无监督算法——内部一致性最大化（ICM），用于在模型自身生成的标签上Fine-tune预训练语言模型，而无需外部监督。在GSM8k-verification、TruthfulQA和Alpaca奖励模型任务中，我们的方法匹配使用金标准监督的效果，并优于使用众包人类监督的效果。在语言模型能力强烈超越人类的任务中，我们的方法能显著更好地激发这些能力，优于使用人类标签的训练。最后，我们展示了我们的方法可以提升前沿语言模型的训练：我们使用我们的方法训练了一个无监督奖励模型，并使用强化学习训练了一个基于Haiku的Claude 3.5助手。这两种模型都优于其人类监督的对应版本。 

---
# Interpreting learned search: finding a transition model and value function in an RNN that plays Sokoban 

**Title (ZH)**: 解析学习到的搜索：在玩索班游戏的RNN中寻找状态转移模型和价值函数 

**Authors**: Mohammad Taufeeque, Aaron David Tucker, Adam Gleave, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2506.10138)  

**Abstract**: We partially reverse-engineer a convolutional recurrent neural network (RNN) trained to play the puzzle game Sokoban with model-free reinforcement learning. Prior work found that this network solves more levels with more test-time compute. Our analysis reveals several mechanisms analogous to components of classic bidirectional search. For each square, the RNN represents its plan in the activations of channels associated with specific directions. These state-action activations are analogous to a value function - their magnitudes determine when to backtrack and which plan branch survives pruning. Specialized kernels extend these activations (containing plan and value) forward and backward to create paths, forming a transition model. The algorithm is also unlike classical search in some ways. State representation is not unified; instead, the network considers each box separately. Each layer has its own plan representation and value function, increasing search depth. Far from being inscrutable, the mechanisms leveraging test-time compute learned in this network by model-free training can be understood in familiar terms. 

**Abstract (ZH)**: 我们部分反向-engineering一种用于使用模型无关强化学习训练的谜题游戏Sokoban玩法规则的卷积循环神经网络（RNN）。先前的研究发现，该网络在更多的测试时间计算资源下可以解决更多的关卡。我们的分析揭示了几种类似于经典双向搜索组件的机制。对于每个方格，RNN通过与特定方向相关的通道激活表示其计划。这些状态-动作激活类似于价值函数——其大小决定何时回溯以及哪些计划分支能够存活。专门的核向前和向后扩展这些包含计划和价值的激活来创建路径，形成转换模型。该算法在某些方面也不同于经典搜索。状态表示不是统一的；相反，网络分别考虑每个箱子。每层都有自己的计划表示和价值函数，从而增加了搜索深度。与普遍认为的难以理解不同，通过模型无关训练学习的利用测试时间计算资源的机制可以以熟悉的方式进行理解。 

---
# Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning 

**Title (ZH)**: 自预测表示在行为克隆中的组合泛化 

**Authors**: Daniel Lawson, Adriana Hugessen, Charlotte Cloutier, Glen Berseth, Khimya Khetarpal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10137)  

**Abstract**: Behavioral cloning (BC) methods trained with supervised learning (SL) are an effective way to learn policies from human demonstrations in domains like robotics. Goal-conditioning these policies enables a single generalist policy to capture diverse behaviors contained within an offline dataset. While goal-conditioned behavior cloning (GCBC) methods can perform well on in-distribution training tasks, they do not necessarily generalize zero-shot to tasks that require conditioning on novel state-goal pairs, i.e. combinatorial generalization. In part, this limitation can be attributed to a lack of temporal consistency in the state representation learned by BC; if temporally related states are encoded to similar latent representations, then the out-of-distribution gap for novel state-goal pairs would be reduced. Hence, encouraging this temporal consistency in the representation space should facilitate combinatorial generalization. Successor representations, which encode the distribution of future states visited from the current state, nicely encapsulate this property. However, previous methods for learning successor representations have relied on contrastive samples, temporal-difference (TD) learning, or both. In this work, we propose a simple yet effective representation learning objective, $\text{BYOL-}\gamma$ augmented GCBC, which is not only able to theoretically approximate the successor representation in the finite MDP case without contrastive samples or TD learning, but also, results in competitive empirical performance across a suite of challenging tasks requiring combinatorial generalization. 

**Abstract (ZH)**: 基于行为克隆的目标条件化方法通过监督学习训练，在 Robotics 等领域从人类演示中学习策略是一种有效的方式。目标条件化这些策略使得单一通用策略能够捕捉到离线数据集中包含的多样行为。虽然目标条件化行为克隆（GCBC）方法在同分布训练任务中表现良好，但在需要以新颖状态-目标对进行条件化的新任务上并不必然实现零样本泛化，即组合泛化。这一局限部分源于行为克隆学习的状态表示缺乏时间一致性；如果相关时间状态被编码到相似的潜在表示中，那么针对新颖状态-目标对的泛化差距将会减小。因此，在表示空间中鼓励这种时间一致性将有助于组合泛化。后继表示，它编码从当前状态访问的未来状态的分布，恰好体现了这一特性。然而，之前学习后继表示的方法依赖于对比样本、时间差分（TD）学习或者两者结合。在本文中，我们提出了一种简单而有效的表示学习目标——$\text{BYOL-}\gamma$增强GCBC，它不仅能够在不使用对比样本或TD学习的情况下理论上逼近在有限MDP情况下的后继表示，还在多种需要组合泛化的挑战性任务中实现了有竞争力的实验性能。 

---
# GRAIL: A Benchmark for GRaph ActIve Learning in Dynamic Sensing Environments 

**Title (ZH)**: GRAIL：动态传感环境中图活性学习的基准评测 

**Authors**: Maryam Khalid, Akane Sano  

**Link**: [PDF](https://arxiv.org/pdf/2506.10120)  

**Abstract**: Graph-based Active Learning (AL) leverages the structure of graphs to efficiently prioritize label queries, reducing labeling costs and user burden in applications like health monitoring, human behavior analysis, and sensor networks. By identifying strategically positioned nodes, graph AL minimizes data collection demands while maintaining model performance, making it a valuable tool for dynamic environments. Despite its potential, existing graph AL methods are often evaluated on static graph datasets and primarily focus on prediction accuracy, neglecting user-centric considerations such as sampling diversity, query fairness, and adaptability to dynamic settings. To bridge this gap, we introduce GRAIL, a novel benchmarking framework designed to evaluate graph AL strategies in dynamic, real-world environments. GRAIL introduces novel metrics to assess sustained effectiveness, diversity, and user burden, enabling a comprehensive evaluation of AL methods under varying conditions. Extensive experiments on datasets featuring dynamic, real-life human sensor data reveal trade-offs between prediction performance and user burden, highlighting limitations in existing AL strategies. GRAIL demonstrates the importance of balancing node importance, query diversity, and network topology, providing an evaluation mechanism for graph AL solutions in dynamic environments. 

**Abstract (ZH)**: 基于图的主动学习（AL）利用图的结构有效优先选择标签查询，减少健康监测、人类行为分析和传感器网络等应用中的标注成本和用户负担。通过识别战略节点，图AL在维持模型性能的同时降低数据收集需求，使其成为动态环境中的 valuable 工具。尽管具有潜力，现有的图AL方法大多在静态图数据集上进行评估，并主要关注预测准确性，忽视了以用户为中心的考虑，如采样多样性、查询公平性和动态环境适应性。为弥补这一差距，我们提出 GRAIL，一种新型基准测试框架，旨在评估图AL策略在动态的真实世界环境中的表现。GRAIL 引入新型指标评估持续有效性、多样性和用户负担，使AL方法在不同条件下进行全面评估。大规模实验结果显示，在动态现实生活中的人体传感器数据集上，预测性能与用户负担之间的权衡关系，突显了现有AL策略的局限性。GRAIL 指出了平衡节点重要性、查询多样性和网络拓扑的重要性，提供了在动态环境中评估图AL解决方案的机制。 

---
# Detecção da Psoríase Utilizando Visão Computacional: Uma Abordagem Comparativa Entre CNNs e Vision Transformers 

**Title (ZH)**: 利用计算机视觉检测银屑病：CNNs与Vision Transformers的比较研究 

**Authors**: Natanael Lucena, Fábio S. da Silva, Ricardo Rios  

**Link**: [PDF](https://arxiv.org/pdf/2506.10119)  

**Abstract**: This paper presents a comparison of the performance of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) in the task of multi-classifying images containing lesions of psoriasis and diseases similar to it. Models pre-trained on ImageNet were adapted to a specific data set. Both achieved high predictive metrics, but the ViTs stood out for their superior performance with smaller models. Dual Attention Vision Transformer-Base (DaViT-B) obtained the best results, with an f1-score of 96.4%, and is recommended as the most efficient architecture for automated psoriasis detection. This article reinforces the potential of ViTs for medical image classification tasks. 

**Abstract (ZH)**: 这篇论文比较了卷积神经网络（CNNs）和视觉变换器（ViTs）在鉴别银屑病及其类似疾病皮肤病图像多分类任务中的性能。预训练的ImageNet模型适应了特定数据集。两者都获得了较高的预测指标，但ViTs凭借更小模型的优越性能脱颖而出。Dual Attention Vision Transformer-Base (DaViT-B) 达到了96.4%的F1分数，并被推荐为自动银屑病检测的最高效架构。本文强化了ViTs在医学图像分类任务中的潜力。 

---
# One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture 

**Title (ZH)**: 一专多能：基于大语言模型的精准农业异构任务规划 

**Authors**: Marcos Abel Zuzuárregui, Mustafa Melih Toslak, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10106)  

**Abstract**: Artificial intelligence is transforming precision agriculture, offering farmers new tools to streamline their daily operations. While these technological advances promise increased efficiency, they often introduce additional complexity and steep learning curves that are particularly challenging for non-technical users who must balance tech adoption with existing workloads. In this paper, we present a natural language (NL) robotic mission planner that enables non-specialists to control heterogeneous robots through a common interface. By leveraging large language models (LLMs) and predefined primitives, our architecture seamlessly translates human language into intermediate descriptions that can be executed by different robotic platforms. With this system, users can formulate complex agricultural missions without writing any code. In the work presented in this paper, we extend our previous system tailored for wheeled robot mission planning through a new class of experiments involving robotic manipulation and computer vision tasks. Our results demonstrate that the architecture is both general enough to support a diverse set of robots and powerful enough to execute complex mission requests. This work represents a significant step toward making robotic automation in precision agriculture more accessible to non-technical users. 

**Abstract (ZH)**: 人工智能正在 transforming 精确农业，为农民提供新的工具以精简日常运营。尽管这些技术进步承诺提高效率，但它们通常会引入额外的复杂性和陡峭的学习曲线，这对必须在技术采用与现有工作量之间保持平衡的非技术用户尤其具有挑战性。在本文中，我们提出了一种自然语言（NL）机器人任务规划器，使非专家能够通过通用界面控制异构机器人。借助大规模语言模型（LLMs）和预定义的基本要素，我们的架构无缝地将自然语言转换为可以由不同机器人平台执行的中间描述。通过此系统，用户可以在不编写任何代码的情况下制定复杂的农业任务。在本文中，我们扩展了我们之前针对轮式机器人任务规划的系统，通过涉及机器人操作和计算机视觉任务的新类实验。我们的结果显示，该架构既通用到支持多种类型的机器人，又能执行复杂的任务请求。这项工作代表了使精确农业中的机器人自动化对非技术用户更具可访问性的重大进展。 

---
# Learning to Collaborate Over Graphs: A Selective Federated Multi-Task Learning Approach 

**Title (ZH)**: 基于图的协同学习：一种选择性联邦多任务学习方法 

**Authors**: Ahmed Elbakary, Chaouki Ben Issaid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.10102)  

**Abstract**: We present a novel federated multi-task learning method that leverages cross-client similarity to enable personalized learning for each client. To avoid transmitting the entire model to the parameter server, we propose a communication-efficient scheme that introduces a feature anchor, a compact vector representation that summarizes the features learned from the client's local classes. This feature anchor is shared with the server to account for local clients' distribution. In addition, the clients share the classification heads, a lightweight linear layer, and perform a graph-based regularization to enable collaboration among clients. By modeling collaboration between clients as a dynamic graph and continuously updating and refining this graph, we can account for any drift from the clients. To ensure beneficial knowledge transfer and prevent negative collaboration, we leverage a community detection-based approach that partitions this dynamic graph into homogeneous communities, maximizing the sum of task similarities, represented as the graph edges' weights, within each community. This mechanism restricts collaboration to highly similar clients within their formed communities, ensuring positive interaction and preserving personalization. Extensive experiments on two heterogeneous datasets demonstrate that our method significantly outperforms state-of-the-art baselines. Furthermore, we show that our method exhibits superior computation and communication efficiency and promotes fairness across clients. 

**Abstract (ZH)**: 我们提出了一种新颖的联邦多任务学习方法，通过利用客户端之间的相似性来实现个性化学习。为了避免传输整个模型到参数服务器，我们提出了一种通信高效的方案，引入了一个特征锚点，这是一种紧凑的向量表示，总结了客户端本地类别的特征学习结果。该特征锚点与服务器共享，以反映局部客户端的数据分布。此外，客户端共享分类头、一个轻量级线性层，并通过图谱正则化实现客户端之间的协作。通过将客户端的协作建模为动态图，并不断更新和完善这一图谱，可以考虑任何客户端可能发生的漂移。为了确保有益的知识迁移并防止负面协作，我们利用基于社区检测的方法将此动态图划分为同质社区，最大化每个社区内任务相似性的总和，用图边权重表示。这一机制限制了高度相似的客户端之间的协作，确保了正面互动并保持个性化。在两个异构数据集上的广泛实验表明，我们的方法显著优于现有最先进的基线方法。此外，我们展示了我们的方法在计算和通信效率方面表现出色，并促进了客户端之间的公平性。 

---
# Leveraging LLMs for Mission Planning in Precision Agriculture 

**Title (ZH)**: 利用大语言模型在精确农业中的任务规划应用 

**Authors**: Marcos Abel Zuzuárregui, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10093)  

**Abstract**: Robotics and artificial intelligence hold significant potential for advancing precision agriculture. While robotic systems have been successfully deployed for various tasks, adapting them to perform diverse missions remains challenging, particularly because end users often lack technical expertise. In this paper, we present an end-to-end system that leverages large language models (LLMs), specifically ChatGPT, to enable users to assign complex data collection tasks to autonomous robots using natural language instructions. To enhance reusability, mission plans are encoded using an existing IEEE task specification standard, and are executed on robots via ROS2 nodes that bridge high-level mission descriptions with existing ROS libraries. Through extensive experiments, we highlight the strengths and limitations of LLMs in this context, particularly regarding spatial reasoning and solving complex routing challenges, and show how our proposed implementation overcomes them. 

**Abstract (ZH)**: 机器人技术和人工智能在精准农业领域的应用具有重要潜力。尽管已成功部署了各种机器人系统，但将它们适应执行多种任务仍然具有挑战性，尤其是因为最终用户往往缺乏技术 expertise。在这种情况下，我们提出了一套端到端的系统，利用大规模语言模型（LLMs），特别是ChatGPT，使用户能够使用自然语言指令将复杂的数据收集任务分配给自主机器人。为了提高可重用性，任务计划使用现有的IEEE任务规范标准进行编码，并通过ROS2节点执行，该节点将高级任务描述与现有的ROS库连接起来。通过广泛实验，我们强调了LLMs在这一具体场景中的优势和局限性，特别是关于空间推理和解决复杂路径挑战的能力，并展示了我们提出的实现方法如何克服这些挑战。 

---
# Test-Time Adaptation for Generalizable Task Progress Estimation 

**Title (ZH)**: 运行时自适应以实现可泛化的任务进度估计 

**Authors**: Christos Ziakas, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10085)  

**Abstract**: We propose a test-time adaptation method that enables a progress estimation model to adapt online to the visual and temporal context of test trajectories by optimizing a learned self-supervised objective. To this end, we introduce a gradient-based meta-learning strategy to train the model on expert visual trajectories and their natural language task descriptions, such that test-time adaptation improves progress estimation relying on semantic content over temporal order. Our test-time adaptation method generalizes from a single training environment to diverse out-of-distribution tasks, environments, and embodiments, outperforming the state-of-the-art in-context learning approach using autoregressive vision-language models. 

**Abstract (ZH)**: 我们提出了一种测试时自适应方法，使进度估计模型能够通过优化一个学习到的自监督目标，在线适应测试轨迹的视觉和时间上下文。为此，我们引入了一种基于梯度的元学习策略，使模型能够在专家视觉轨迹及其自然语言任务描述上进行训练，从而测试时的自适应能够依靠语义内容而不是时间顺序来改善进度估计。我们的测试时自适应方法能够从单一训练环境泛化到多样化的未知分布任务、环境和实体，并且在使用自回归视觉语言模型的现有最佳上下文学习方法上表现出更高的性能。 

---
# A quantum semantic framework for natural language processing 

**Title (ZH)**: 量子语义框架在自然语言处理中的应用 

**Authors**: Christopher J. Agostino, Quan Le Thien, Molly Apsel, Denizhan Pak, Elina Lesyk, Ashabari Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10077)  

**Abstract**: Semantic degeneracy represents a fundamental property of natural language that extends beyond simple polysemy to encompass the combinatorial explosion of potential interpretations that emerges as semantic expressions increase in complexity. Large Language Models (LLMs) and other modern NLP systems face inherent limitations precisely because they operate within natural language itself, making them subject to the same interpretive constraints imposed by semantic degeneracy. In this work, we argue using Kolmogorov complexity that as an expression's complexity grows, the likelihood of any interpreting agent (human or LLM-powered AI) recovering the single intended meaning vanishes. This computational intractability suggests the classical view that linguistic forms possess meaning in and of themselves is flawed. We alternatively posit that meaning is instead actualized through an observer-dependent interpretive act. To test this, we conducted a semantic Bell inequality test using diverse LLM agents as ``computational cognitive systems'' to interpret ambiguous word pairs under varied contextual settings. Across several independent experiments, we found average CHSH expectation values ranging from 1.2 to 2.8, with several runs yielding values (e.g., 2.3-2.4) that significantly violate the classical boundary ($|S|\leq2$). This demonstrates that linguistic interpretation under ambiguity can exhibit non-classical contextuality, consistent with results from human cognition experiments. These results inherently imply that classical frequentist-based analytical approaches for natural language are necessarily lossy. Instead, we propose that Bayesian-style repeated sampling approaches can provide more practically useful and appropriate characterizations of linguistic meaning in context. 

**Abstract (ZH)**: 语义退化代表了自然语言的基本属性，超越了简单的多义性，涵盖了随着语义表达复杂性的增加而产生的潜在解释的组合爆炸。大型语言模型（LLMs）和其他现代NLP系统固有的局限性在于它们在自然语言本身的运作中，使其易受语义退化施加的解释约束的影响。在本文中，我们使用柯尔莫哥洛夫复杂性论证，随着表达式复杂性的增长，任何解释代理（人类或LLM驱动的AI）恢复单一意图意义的可能性消失。这种计算上的不可行性表明，语言形式本身具有意义的经典观点是不完整的。相反，我们认为意义是通过观察者依赖的解释行为实现的。为了验证这一观点，我们使用多种LLM代理作为“计算认知系统”，在不同的上下文设置下解释模棱两可的词对，进行了语义贝尔不等式测试。在数个独立实验中，我们发现平均CHSH期望值从1.2到2.8不等，有几次运行得到的值（例如2.3-2.4）显著违反了经典边界(|S|≤2)。这表明在模棱两可的语义解释中可以表现出非经典的上下文关联性，与人类认知实验的结果一致。这些结果本质上意味着基于经典频率分析的方法在自然语言中是必然有损失的。相反，我们建议贝叶斯式重采样方法可以为语境中文本意义提供更具实用性和适当的描述。 

---
# Textual Bayes: Quantifying Uncertainty in LLM-Based Systems 

**Title (ZH)**: 文本贝叶斯：量化基于LLM系统的不确定性 

**Authors**: Brendan Leigh Ross, Noël Vouitsis, Atiyeh Ashari Ghomi, Rasa Hosseinzadeh, Ji Xin, Zhaoyan Liu, Yi Sui, Shiyi Hou, Kin Kwan Leung, Gabriel Loaiza-Ganem, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2506.10060)  

**Abstract**: Although large language models (LLMs) are becoming increasingly capable of solving challenging real-world tasks, accurately quantifying their uncertainty remains a critical open problem, which limits their applicability in high-stakes domains. This challenge is further compounded by the closed-source, black-box nature of many state-of-the-art LLMs. Moreover, LLM-based systems can be highly sensitive to the prompts that bind them together, which often require significant manual tuning (i.e., prompt engineering). In this work, we address these challenges by viewing LLM-based systems through a Bayesian lens. We interpret prompts as textual parameters in a statistical model, allowing us to use a small training dataset to perform Bayesian inference over these prompts. This novel perspective enables principled uncertainty quantification over both the model's textual parameters and its downstream predictions, while also incorporating prior beliefs about these parameters expressed in free-form text. To perform Bayesian inference, a difficult problem even for well-studied data modalities, we introduce Metropolis-Hastings through LLM Proposals (MHLP), a novel Markov chain Monte Carlo (MCMC) algorithm that combines prompt optimization techniques with standard MCMC methods. MHLP is a turnkey modification to existing LLM pipelines, including those that rely exclusively on closed-source models. Empirically, we demonstrate that our method yields improvements in both predictive accuracy and uncertainty quantification (UQ) on a range of LLM benchmarks and UQ tasks. More broadly, our work demonstrates a viable path for incorporating methods from the rich Bayesian literature into the era of LLMs, paving the way for more reliable and calibrated LLM-based systems. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在解决复杂的现实任务方面越来越 capable，但准确量化其不确定性仍然是一个关键的开放问题，这限制了其在高风险领域中的应用。此外，许多最先进的LLMs具有封闭源代码和黑盒性质，进一步加剧了这一挑战。更糟糕的是，基于LLM的系统对将它们结合起来的提示非常敏感，这些提示通常需要大量的手动调整（即提示工程）。在本文中，我们通过贝叶斯视角来应对这些挑战。我们将提示视为统计模型中的文本参数，使我们能够使用有限的训练数据集对这些参数进行贝叶斯推断。这种新的视角使得我们能够在文本参数及其下游预测上进行严谨的不确定性量化，并且能够纳入以自由格式文本表达的先验信念。为了进行贝叶斯推断，即使对于研究充分的數據模態也是一项难题，我们提出了LLM提案的Metropolis-Hastings（MHLP），这是一种新的马尔可夫链蒙特卡洛（MCMC）算法，结合了提示优化技术与标准的MCMC方法。MHLP是一种现成的现有LLM流水线的修改，包括完全依赖封闭源代码模型的流水线。实证结果表明，我们的方法在多种LLM基准测试和不确定性量化任务中均提高了预测准确性和不确定性量化。更广泛地说，我们的工作展示了将丰富的贝叶斯文献中的方法纳入LLM时代的可行途径，为更可靠和校准的LLM基础系统铺平了道路。 

---
# Omni-DPO: A Dual-Perspective Paradigm for Dynamic Preference Learning of LLMs 

**Title (ZH)**: 全方位DPO：LLMs动态偏好学习的双向视角范式 

**Authors**: Shangpin Peng, Weinong Wang, Zhuotao Tian, Senqiao Yang, Xing Wu, Haotian Xu, Chengquan Zhang, Takashi Isobe, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10054)  

**Abstract**: Direct Preference Optimization (DPO) has become a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based approaches typically treat all preference pairs uniformly, ignoring critical variations in their inherent quality and learning utility, leading to suboptimal data utilization and performance. To address this challenge, we propose Omni-DPO, a dual-perspective optimization framework that jointly accounts for (1) the inherent quality of each preference pair and (2) the model's evolving performance on those pairs. By adaptively weighting samples according to both data quality and the model's learning dynamics during training, Omni-DPO enables more effective training data utilization and achieves better performance. Experimental results on various models and benchmarks demonstrate the superiority and generalization capabilities of Omni-DPO. On textual understanding tasks, Gemma-2-9b-it finetuned with Omni-DPO beats the leading LLM, Claude 3 Opus, by a significant margin of 6.7 points on the Arena-Hard benchmark. On mathematical reasoning tasks, Omni-DPO consistently outperforms the baseline methods across all benchmarks, providing strong empirical evidence for the effectiveness and robustness of our approach. Code and models will be available at this https URL. 

**Abstract (ZH)**: 直接偏好优化(Direct Preference Optimization, DPO)已成为基于人类反馈强化学习(Reinforcement Learning from Human Feedback, RLHF)的基石，这得益于其简洁性和高效性。然而，现有的DPO方法通常将所有偏好对均等处理，忽视了它们固有质量和学习效用的关键差异，导致数据利用不足和性能不佳。为解决这一挑战，我们提出了Omni-DPO，这是一种双视角优化框架，同时考虑了每对偏好自身的固有质量和模型在此基础上的渐进性能。通过在训练过程中根据数据质量和模型的学习动态自适应加权样本，Omni-DPO实现了更有效的训练数据利用，并取得了更好的性能。在各类模型和基准上的实验结果表明了Omni-DPO的优势和泛化能力。在文本理解任务中，使用Omni-DPO微调的Gemma-2-9b-it在Arena-Hard基准上的得分比领先的大规模语言模型Claude 3 Opus高6.7分。在数学推理任务中，Omni-DPO在所有基准上均优于基线方法，这为我们的方法的有效性和鲁棒性提供了强有力的实证证据。代码和模型将在以下链接处提供。 

---
# Ambient Diffusion Omni: Training Good Models with Bad Data 

**Title (ZH)**: Ambient Diffusion Omni: 使用不良数据训练优质模型 

**Authors**: Giannis Daras, Adrian Rodriguez-Munoz, Adam Klivans, Antonio Torralba, Constantinos Daskalakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.10038)  

**Abstract**: We show how to use low-quality, synthetic, and out-of-distribution images to improve the quality of a diffusion model. Typically, diffusion models are trained on curated datasets that emerge from highly filtered data pools from the Web and other sources. We show that there is immense value in the lower-quality images that are often discarded. We present Ambient Diffusion Omni, a simple, principled framework to train diffusion models that can extract signal from all available images during training. Our framework exploits two properties of natural images -- spectral power law decay and locality. We first validate our framework by successfully training diffusion models with images synthetically corrupted by Gaussian blur, JPEG compression, and motion blur. We then use our framework to achieve state-of-the-art ImageNet FID, and we show significant improvements in both image quality and diversity for text-to-image generative modeling. The core insight is that noise dampens the initial skew between the desired high-quality distribution and the mixed distribution we actually observe. We provide rigorous theoretical justification for our approach by analyzing the trade-off between learning from biased data versus limited unbiased data across diffusion times. 

**Abstract (ZH)**: 我们展示了如何利用低质量、合成和离分布域的图像来提高扩散模型的质量。我们证明了被常丢弃的低质量图像中蕴含的巨大价值。我们提出了Ambient Diffusion Omni这一简单且原理明确的框架，该框架能够利用所有可用图像在训练过程中提取信号。我们的框架利用了自然图像的两种特性——谱功率律衰减和局部性。我们首先通过成功训练被高斯模糊、JPEG压缩和运动模糊等合成噪声破坏的图像来验证该框架的有效性。然后，我们使用该框架实现了Imagenet FID的最新成果，并展示了在文本生成图像时图像质量和多样性的显著改进。核心洞见在于噪声减弱了所需的高质量分布与我们实际观察到的混合分布之间的初始偏差。我们通过对扩散时间段内从有偏数据学习与有限无偏数据学习之间的权衡进行严谨的理论分析，为我们的方法提供了理论依据。 

---
# FastFLUX: Pruning FLUX with Block-wise Replacement and Sandwich Training 

**Title (ZH)**: FastFLUX: 基于块级替换和三明治训练的FLUX剪枝 

**Authors**: Fuhan Cai, Yong Guo, Jie Li, Wenbo Li, Xiangzhong Fang, Jian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10035)  

**Abstract**: Recent advancements in text-to-image (T2I) generation have led to the emergence of highly expressive models such as diffusion transformers (DiTs), exemplified by FLUX. However, their massive parameter sizes lead to slow inference, high memory usage, and poor deployability. Existing acceleration methods (e.g., single-step distillation and attention pruning) often suffer from significant performance degradation and incur substantial training costs. To address these limitations, we propose FastFLUX, an architecture-level pruning framework designed to enhance the inference efficiency of FLUX. At its core is the Block-wise Replacement with Linear Layers (BRLL) method, which replaces structurally complex residual branches in ResBlocks with lightweight linear layers while preserving the original shortcut connections for stability. Furthermore, we introduce Sandwich Training (ST), a localized fine-tuning strategy that leverages LoRA to supervise neighboring blocks, mitigating performance drops caused by structural replacement. Experiments show that our FastFLUX maintains high image quality under both qualitative and quantitative evaluations, while significantly improving inference speed, even with 20\% of the hierarchy pruned. Our code will be available soon. 

**Abstract (ZH)**: Recent Advancements in Text-to-Image Generation Have Led to Highly Expressive Models Such as Diffusion Transformers (DiTs), Exemplified by FLUX. However, Their Massive Parameter Sizes Lead to Slow Inference, High Memory Usage, and Poor Deployability. To Address These Limitations, We Propose FastFLUX, an Architecture-Level Pruning Framework Designed to Enhance the Inference Efficiency of FLUX. 

---
# Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment 

**Title (ZH)**: multimodal知识版权在RAG-as-a-Service环境中的保障 

**Authors**: Tianyu Chen, Jian Lou, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10030)  

**Abstract**: As Retrieval-Augmented Generation (RAG) evolves into service-oriented platforms (Rag-as-a-Service) with shared knowledge bases, protecting the copyright of contributed data becomes essential. Existing watermarking methods in RAG focus solely on textual knowledge, leaving image knowledge unprotected. In this work, we propose AQUA, the first watermark framework for image knowledge protection in Multimodal RAG systems. AQUA embeds semantic signals into synthetic images using two complementary methods: acronym-based triggers and spatial relationship cues. These techniques ensure watermark signals survive indirect watermark propagation from image retriever to textual generator, being efficient, effective and imperceptible. Experiments across diverse models and datasets show that AQUA enables robust, stealthy, and reliable copyright tracing, filling a key gap in multimodal RAG protection. 

**Abstract (ZH)**: 基于多模态RAG系统的图像知识水印框架AQUA 

---
# Evaluation empirique de la sécurisation et de l'alignement de ChatGPT et Gemini: analyse comparative des vulnérabilités par expérimentations de jailbreaks 

**Title (ZH)**: ChatGPT和Gemini的安全性与对齐性实证评价： Jailbreak攻击下的 Vulnerabilities 比较分析 

**Authors**: Rafaël Nouailles  

**Link**: [PDF](https://arxiv.org/pdf/2506.10029)  

**Abstract**: Large Language models (LLMs) are transforming digital usage, particularly in text generation, image creation, information retrieval and code development. ChatGPT, launched by OpenAI in November 2022, quickly became a reference, prompting the emergence of competitors such as Google's Gemini. However, these technological advances raise new cybersecurity challenges, including prompt injection attacks, the circumvention of regulatory measures (jailbreaking), the spread of misinformation (hallucinations) and risks associated with deep fakes. This paper presents a comparative analysis of the security and alignment levels of ChatGPT and Gemini, as well as a taxonomy of jailbreak techniques associated with experiments. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在变革数字使用，特别是在文本生成、图像创作、信息检索和代码开发方面。2022年11月由OpenAI发布的ChatGPT迅速成为了一个参考，促使出现了竞争对手如Google的Gemini。然而，这些技术进步带来了新的网络安全挑战，包括提示注入攻击、规避监管措施（ Jailbreaking ）、传播虚假信息（幻觉）以及深度合成的相关风险。本文对ChatGPT和Gemini的安全性和对齐水平进行了比较分析，并提出了与实验相关的Jailbreak技术分类。 

---
# Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models 

**Title (ZH)**: 私人记忆编辑：将记忆转换为防护以增强大型语言模型的数据隐私 

**Authors**: Elena Sofia Ruzzetti, Giancarlo A. Xompero, Davide Venditti, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2506.10024)  

**Abstract**: Large Language Models (LLMs) memorize, and thus, among huge amounts of uncontrolled data, may memorize Personally Identifiable Information (PII), which should not be stored and, consequently, not leaked. In this paper, we introduce Private Memorization Editing (PME), an approach for preventing private data leakage that turns an apparent limitation, that is, the LLMs' memorization ability, into a powerful privacy defense strategy. While attacks against LLMs have been performed exploiting previous knowledge regarding their training data, our approach aims to exploit the same kind of knowledge in order to make a model more robust. We detect a memorized PII and then mitigate the memorization of PII by editing a model knowledge of its training data. We verify that our procedure does not affect the underlying language model while making it more robust against privacy Training Data Extraction attacks. We demonstrate that PME can effectively reduce the number of leaked PII in a number of configurations, in some cases even reducing the accuracy of the privacy attacks to zero. 

**Abstract (ZH)**: 大型语言模型（LLMs）存储数据，因此可能记住个人可识别信息（PII），这些信息不应该被存储，从而也不应该泄露。在本文中，我们介绍了私有记忆编辑（PME）方法，这是一种将LLMs的记忆能力这一看似局限转化为强大隐私防御策略的方法。虽然对LLMs的攻击利用了有关其训练数据的先前知识，但我们的方法旨在利用相同类型的知识以使模型更加健壮。我们检测已记住的PII，然后通过编辑模型的训练数据知识来减轻PII的记忆。我们验证了该过程不会影响底层语言模型，同时使其更健壮以抵抗隐私训练数据提取攻击。我们证明在多种配置下，PME可以有效减少泄露的PII数量，在某些情况下甚至将隐私攻击的准确率降低至零。 

---
# LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges 

**Title (ZH)**: LLMs处于困境：恶意软件请求与破解挑战 

**Authors**: Haoyang Li, Huan Gao, Zhiyuan Zhao, Zhiyu Lin, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10022)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has heightened concerns about their security, particularly their vulnerability to jailbreak attacks that leverage crafted prompts to generate malicious outputs. While prior research has been conducted on general security capabilities of LLMs, their specific susceptibility to jailbreak attacks in code generation remains largely unexplored. To fill this gap, we propose MalwareBench, a benchmark dataset containing 3,520 jailbreaking prompts for malicious code-generation, designed to evaluate LLM robustness against such threats. MalwareBench is based on 320 manually crafted malicious code generation requirements, covering 11 jailbreak methods and 29 code functionality categories. Experiments show that mainstream LLMs exhibit limited ability to reject malicious code-generation requirements, and the combination of multiple jailbreak methods further reduces the model's security capabilities: specifically, the average rejection rate for malicious content is 60.93%, dropping to 39.92% when combined with jailbreak attack algorithms. Our work highlights that the code security capabilities of LLMs still pose significant challenges. 

**Abstract (ZH)**: 大规模语言模型的广泛应用加剧了对其安全性的关注，尤其是它们容易受到利用精心构造的提示进行恶意输出生成的越狱攻击。尽管先前的研究已探讨了大型语言模型的一般安全能力，但它们在代码生成中对越狱攻击的特定易感性仍基本上未被探索。为了填补这一空白，我们提出了MalwareBench，这是一个基准数据集，包含3,520个用于恶意代码生成的越狱提示，旨在评估大型语言模型在面对此类威胁时的鲁棒性。MalwareBench基于320个手工构建的恶意代码生成需求，涵盖了11种越狱方法和29种代码功能类别。实验结果显示，主流的大型语言模型在拒绝恶意代码生成需求方面的能力有限，结合多种越狱方法进一步降低了模型的安全能力：具体来说，恶意内容的平均拒绝率为60.93%，而在结合越狱攻击算法后，该率下降至39.92%。我们的工作表明，大型语言模型的代码安全能力仍面临着重大挑战。 

---
# From Tool Calling to Symbolic Thinking: LLMs in a Persistent Lisp Metaprogramming Loop 

**Title (ZH)**: 从工具调用到符号思维：LLM在持久的Lisp元编程循环中的应用 

**Authors**: Jordi de la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2506.10021)  

**Abstract**: We propose a novel architecture for integrating large language models (LLMs) with a persistent, interactive Lisp environment. This setup enables LLMs to define, invoke, and evolve their own tools through programmatic interaction with a live REPL. By embedding Lisp expressions within generation and intercepting them via a middleware layer, the system allows for stateful external memory, reflective programming, and dynamic tool creation. We present a design framework and architectural principles to guide future implementations of interactive AI systems that integrate symbolic programming with neural language generation. 

**Abstract (ZH)**: 我们提出了一种将大型语言模型（LLMs）与持久的交互式Lisp环境集成的新架构。该设置使LLMs能够通过与实时REPL的程序化交互来定义、调用和演化自己的工具。通过在生成中嵌入Lisp表达式并在中间件层拦截它们，该系统允许状态化的外部内存、反射性编程和动态工具创建。我们提出了一种设计框架和架构原则，以指导将符号编程与神经语言生成集成的交互式AI系统的未来实现。 

---
# From Threat to Tool: Leveraging Refusal-Aware Injection Attacks for Safety Alignment 

**Title (ZH)**: 从威胁到工具：利用拒绝意识注入攻击实现安全对齐 

**Authors**: Kyubyung Chae, Hyunbin Jin, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10020)  

**Abstract**: Safely aligning large language models (LLMs) often demands extensive human-labeled preference data, a process that's both costly and time-consuming. While synthetic data offers a promising alternative, current methods frequently rely on complex iterative prompting or auxiliary models. To address this, we introduce Refusal-Aware Adaptive Injection (RAAI), a straightforward, training-free, and model-agnostic framework that repurposes LLM attack techniques. RAAI works by detecting internal refusal signals and adaptively injecting predefined phrases to elicit harmful, yet fluent, completions. Our experiments show RAAI effectively jailbreaks LLMs, increasing the harmful response rate from a baseline of 2.15% to up to 61.04% on average across four benchmarks. Crucially, fine-tuning LLMs with the synthetic data generated by RAAI improves model robustness against harmful prompts while preserving general capabilities on standard tasks like MMLU and ARC. This work highlights how LLM attack methodologies can be reframed as practical tools for scalable and controllable safety alignment. 

**Abstract (ZH)**: 安全对齐大型语言模型通常需要大量的人标注偏好数据，这一过程既耗时又昂贵。虽然合成数据提供了一种有前景的替代方案，但当前方法往往依赖于复杂的迭代提示或辅助模型。为了解决这一问题，我们提出了拒绝意识自适应注入（RAAI）框架，这是一种无需训练且模型无关的方法，重新利用了大型语言模型攻击技术。RAAI通过检测内部拒绝信号，并适应性地注入预定义短语以引发有害但流畅的完成。我们的实验表明，RAAI有效地突破了大型语言模型的限制，平均在四个基准测试中将有害响应率从基线的2.15%提高到61.04%。重要的是，使用RAAI生成的合成数据微调大型语言模型可以提高模型对有害提示的鲁棒性，同时在标准任务（如MMLU和ARC）上保留一般能力。这项工作突显了如何重新构想大型语言模型攻击方法，使之成为可扩展且可控的安全对齐的实用工具。 

---
# A Survey of Automatic Evaluation Methods on Text, Visual and Speech Generations 

**Title (ZH)**: 自动评价方法综述：文本、视觉和语音生成 

**Authors**: Tian Lan, Yang-Hao Zhou, Zi-Ao Ma, Fanshu Sun, Rui-Qing Sun, Junyu Luo, Rong-Cheng Tu, Heyan Huang, Chen Xu, Zhijing Wu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10019)  

**Abstract**: Recent advances in deep learning have significantly enhanced generative AI capabilities across text, images, and audio. However, automatically evaluating the quality of these generated outputs presents ongoing challenges. Although numerous automatic evaluation methods exist, current research lacks a systematic framework that comprehensively organizes these methods across text, visual, and audio modalities. To address this issue, we present a comprehensive review and a unified taxonomy of automatic evaluation methods for generated content across all three modalities; We identify five fundamental paradigms that characterize existing evaluation approaches across these domains. Our analysis begins by examining evaluation methods for text generation, where techniques are most mature. We then extend this framework to image and audio generation, demonstrating its broad applicability. Finally, we discuss promising directions for future research in cross-modal evaluation methodologies. 

**Abstract (ZH)**: 近期深度学习的进展极大地增强了生成型AI在文本、图像和音频方面的能力。然而，自动评估这些生成输出的质量仍然面临挑战。尽管存在大量的自动评估方法，但当前研究缺乏一个系统框架，能够全面组织这些方法，涵盖文本、视觉和音频模态。为解决这一问题，我们提出了一项全面的综述和一种统一的评估方法分类体系，涵盖了所有三个模态；我们确定了五个基本范式，以描述这些领域中现有评估方法的特征。我们的分析始于评估文本生成的方法，这些技术最为成熟。然后将这一框架扩展到图像和音频生成，展示了其广泛的适用性。最后，我们讨论了跨模态评估方法未来研究的有前途的方向。 

---
# Multimodal Large Language Models: A Survey 

**Title (ZH)**: 多模态大型语言模型：一个综述 

**Authors**: Longzhen Han, Awes Mubarak, Almas Baimagambetov, Nikolaos Polatidis, Thar Baker  

**Link**: [PDF](https://arxiv.org/pdf/2506.10016)  

**Abstract**: Multimodal Large Language Models (MLLMs) have rapidly evolved beyond text generation, now spanning diverse output modalities including images, music, video, human motion, and 3D objects, by integrating language with other sensory modalities under unified architectures. This survey categorises six primary generative modalities and examines how foundational techniques, namely Self-Supervised Learning (SSL), Mixture of Experts (MoE), Reinforcement Learning from Human Feedback (RLHF), and Chain-of-Thought (CoT) prompting, enable cross-modal capabilities. We analyze key models, architectural trends, and emergent cross-modal synergies, while highlighting transferable techniques and unresolved challenges. Architectural innovations like transformers and diffusion models underpin this convergence, enabling cross-modal transfer and modular specialization. We highlight emerging patterns of synergy, and identify open challenges in evaluation, modularity, and structured reasoning. This survey offers a unified perspective on MLLM development and identifies critical paths toward more general-purpose, adaptive, and interpretable multimodal systems. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）已经超越了文本生成，现在涵盖了包括图像、音乐、视频、人体运动和3D对象在内的多种输出模态，通过在统一架构中整合语言与其他感官模态。本文综述分类了六种主要生成模态，并探讨了自监督学习（SSL）、专家混合（MoE）、基于人类反馈的强化学习（RLHF）和思维链（CoT）提示等基础技术如何实现跨模态能力。我们分析了关键模型、架构趋势和新兴的跨模态协同效应，并强调了可转移技术以及未解决的挑战。架构创新如变换器和扩散模型支撑了这一融合过程，使得跨模态转移和模块化专业化成为可能。我们突显了正在形成的协同模式，并指出了评估、模块化和结构化推理等方面的开放挑战。本文综述提供了一种统一的视角，对于开发更通用、自适应和可解释的多模态系统指明了关键路径。 

---
# WDMIR: Wavelet-Driven Multimodal Intent Recognition 

**Title (ZH)**: 小波驱动的多模态意图识别 

**Authors**: Weiyin Gong, Kai Zhang, Yanghai Zhang, Qi Liu, Xinjie Sun, Junyu Lu, Linbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10011)  

**Abstract**: Multimodal intent recognition (MIR) seeks to accurately interpret user intentions by integrating verbal and non-verbal information across video, audio and text modalities. While existing approaches prioritize text analysis, they often overlook the rich semantic content embedded in non-verbal cues. This paper presents a novel Wavelet-Driven Multimodal Intent Recognition(WDMIR) framework that enhances intent understanding through frequency-domain analysis of non-verbal information. To be more specific, we propose: (1) a wavelet-driven fusion module that performs synchronized decomposition and integration of video-audio features in the frequency domain, enabling fine-grained analysis of temporal dynamics; (2) a cross-modal interaction mechanism that facilitates progressive feature enhancement from bimodal to trimodal integration, effectively bridging the semantic gap between verbal and non-verbal information. Extensive experiments on MIntRec demonstrate that our approach achieves state-of-the-art performance, surpassing previous methods by 1.13% on accuracy. Ablation studies further verify that the wavelet-driven fusion module significantly improves the extraction of semantic information from non-verbal sources, with a 0.41% increase in recognition accuracy when analyzing subtle emotional cues. 

**Abstract (ZH)**: 多模态意图识别中基于小波驱动的多模态意图识别框架（Wavelet-Driven Multimodal Intent Recognition Framework） 

---
# Structured Graph Representations for Visual Narrative Reasoning: A Hierarchical Framework for Comics 

**Title (ZH)**: 面向视觉叙事推理的结构化图表示：漫画的层次框架 

**Authors**: Yi-Chun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10008)  

**Abstract**: This paper presents a hierarchical knowledge graph framework for the structured understanding of visual narratives, focusing on multimodal media such as comics. The proposed method decomposes narrative content into multiple levels, from macro-level story arcs to fine-grained event segments. It represents them through integrated knowledge graphs that capture semantic, spatial, and temporal relationships. At the panel level, we construct multimodal graphs that link visual elements such as characters, objects, and actions with corresponding textual components, including dialogue and captions. These graphs are integrated across narrative levels to support reasoning over story structure, character continuity, and event progression.
We apply our approach to a manually annotated subset of the Manga109 dataset and demonstrate its ability to support symbolic reasoning across diverse narrative tasks, including action retrieval, dialogue tracing, character appearance mapping, and panel timeline reconstruction. Evaluation results show high precision and recall across tasks, validating the coherence and interpretability of the framework. This work contributes a scalable foundation for narrative-based content analysis, interactive storytelling, and multimodal reasoning in visual media. 

**Abstract (ZH)**: 本文提出了一种分层次的知识图谱框架，用于结构化理解视觉叙事，重点关注如漫画等多模态媒体。该提出的方珐将叙事内容分解为多个层次，从宏观的故事弧线到细微的事件片段。通过集成的知识图谱表示这些内容，捕捉语义、空间和时间关系。在分镜层面上，我们构建多模态图，将视觉元素如角色、物体和动作与相应的文本组件（包括对话和图注）相链接。这些图在叙事层面进行集成，以支持对故事结构、角色连续性和事件进程的推理。我们将该方法应用于Manga109数据集的手动注释子集，并展示了其在多种叙事任务中支持符号推理的能力，包括动作检索、对话追踪、角色出场映射和分镜时间线重构。评估结果表明，在各项任务中具有高的精确率和召回率，证明了该框架的一致性和可解释性。该工作为基于叙事的内容分析、交互式叙事以及视觉媒体中的多模态推理提供了可扩展的基础。 

---
# Controllable Expressive 3D Facial Animation via Diffusion in a Unified Multimodal Space 

**Title (ZH)**: 统一多模态空间中的可控表情3D面部动画扩散生成 

**Authors**: Kangwei Liu, Junwu Liu, Xiaowei Yi, Jinlin Guo, Yun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10007)  

**Abstract**: Audio-driven emotional 3D facial animation encounters two significant challenges: (1) reliance on single-modal control signals (videos, text, or emotion labels) without leveraging their complementary strengths for comprehensive emotion manipulation, and (2) deterministic regression-based mapping that constrains the stochastic nature of emotional expressions and non-verbal behaviors, limiting the expressiveness of synthesized animations. To address these challenges, we present a diffusion-based framework for controllable expressive 3D facial animation. Our approach introduces two key innovations: (1) a FLAME-centered multimodal emotion binding strategy that aligns diverse modalities (text, audio, and emotion labels) through contrastive learning, enabling flexible emotion control from multiple signal sources, and (2) an attention-based latent diffusion model with content-aware attention and emotion-guided layers, which enriches motion diversity while maintaining temporal coherence and natural facial dynamics. Extensive experiments demonstrate that our method outperforms existing approaches across most metrics, achieving a 21.6\% improvement in emotion similarity while preserving physiologically plausible facial dynamics. Project Page: this https URL. 

**Abstract (ZH)**: 音频驱动的情感3D面部动画面临两个重要挑战：（1）依赖单一模式的控制信号（视频、文本或情感标签），而未能充分利用这些信号的互补优势以实现全方位的情感操控；（2）确定性的回归映射方式限制了情感表达和非言语行为的随机性，从而限制了合成动画的表达性。为应对这些挑战，我们提出了一种基于扩散的可控情感表达3D面部动画框架。我们的方法提出了两个关键创新：（1）以FLAME为中心的多模态情感绑定策略，通过对比学习对齐不同的模态（文本、音频和情感标签），从而从多种信号源中实现灵活的情感控制；（2）基于注意力的潜在扩散模型，具备内容感知的注意力机制和情感导向层，在保持时空连贯性和自然面部动态的同时丰富了动作多样性。广泛的经验研究显示，我们的方法在大多数评估指标上优于现有方法，情感相似度提高了21.6%，同时保持了生理上合理的面部动态。项目页面：this https URL。 

---
# HER2 Expression Prediction with Flexible Multi-Modal Inputs via Dynamic Bidirectional Reconstruction 

**Title (ZH)**: 基于动态双相重构的灵活多模态输入HER2表达预测 

**Authors**: Jie Qin, Wei Yang, Yan Su, Yiran Zhu, Weizhen Li, Yunyue Pan, Chengchang Pan, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10006)  

**Abstract**: Current HER2 assessment models for breast cancer predominantly analyze H&E or IHC images in isolation,despite clinical reliance on their synergistic interpretation. However, concurrent acquisition of both modalities is often hindered by workflow complexity and cost constraints. We propose an adaptive bimodal framework enabling flexible single-/dual-modality HER2 prediction through three innovations: 1) A dynamic branch selector that activates either single-modality reconstruction or dual-modality joint inference based on input completeness; 2) A bidirectional cross-modal GAN performing context-aware feature-space reconstruction of missing modalities; 3) A hybrid training protocol integrating adversarial learning and multi-task optimization. This architecture elevates single-modality H&E prediction accuracy from 71.44% to 94.25% while achieving 95.09% dual-modality accuracy, maintaining 90.28% reliability with sole IHC inputs. The framework's "dual-preferred, single-compatible" design delivers near-bimodal performance without requiring synchronized acquisition, particularly benefiting resource-limited settings through IHC infrastructure cost reduction. Experimental validation confirms 22.81%/12.90% accuracy improvements over H&E/IHC baselines respectively, with cross-modal reconstruction enhancing F1-scores to 0.9609 (HE to IHC) and 0.9251 (IHC to HE). By dynamically routing inputs through reconstruction-enhanced or native fusion pathways, the system mitigates performance degradation from missing data while preserving computational efficiency (78.55% parameter reduction in lightweight variant). This elastic architecture demonstrates significant potential for democratizing precise HER2 assessment across diverse healthcare settings. 

**Abstract (ZH)**: 当前的HER2评估模型主要单独分析H&E或IHC图像，尽管临床依赖于两者协同解析。然而，同时获取两种模态的数据常受限于工作流程复杂性和成本约束。我们提出一种自适应双模态框架，通过三项创新实现灵活的单/双模态HER2预测：1) 动态分支选择器，根据输入完整性激活单模态重建或双模态联合推断；2) 双向跨模态GAN，进行上下文感知的特征空间重建缺失模态；3) 结合对抗学习和多任务优化的混合训练协议。该架构将单模态H&E预测准确性从71.44%提升到94.25%，同时实现95.09%的双模态准确性，并在仅使用IHC输入时保持90.28%的可靠性。该框架的“双模态优先、单模态兼容”设计在无需同步获取的情况下实现接近双模态性能，尤其通过减少IHC基础设施的成本为资源受限环境带来益处。实验验证表明，与H&E/IHC基线相比，分别获得22.81%/12.90%的准确性提升，跨模态重建提高F1分数至HE到IHC为0.9609，IHC到HE为0.9251。通过动态路由输入并通过重建增强或原生融合路径，系统减轻了因数据缺失导致的性能下降，同时保持计算效率（轻量级变体参数减少了78.55%）。该弹性架构在不同医疗保健环境中实现精确HER2评估的普及化展现出显著潜力。 

---
# Multimodal Cinematic Video Synthesis Using Text-to-Image and Audio Generation Models 

**Title (ZH)**: 基于文本到图像和音频生成模型的多模态cinematic视频合成 

**Authors**: Sridhar S, Nithin A, Shakeel Rifath, Vasantha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2506.10005)  

**Abstract**: Advances in generative artificial intelligence have altered multimedia creation, allowing for automatic cinematic video synthesis from text inputs. This work describes a method for creating 60-second cinematic movies incorporating Stable Diffusion for high-fidelity image synthesis, GPT-2 for narrative structuring, and a hybrid audio pipeline using gTTS and YouTube-sourced music. It uses a five-scene framework, which is augmented by linear frame interpolation, cinematic post-processing (e.g., sharpening), and audio-video synchronization to provide professional-quality results. It was created in a GPU-accelerated Google Colab environment using Python 3.11. It has a dual-mode Gradio interface (Simple and Advanced), which supports resolutions of up to 1024x768 and frame rates of 15-30 FPS. Optimizations such as CUDA memory management and error handling ensure reliability. The experiments demonstrate outstanding visual quality, narrative coherence, and efficiency, furthering text-to-video synthesis for creative, educational, and industrial applications. 

**Abstract (ZH)**: 生成式人工智能的进步 telah改变了多媒体创作，使其能够从文本输入自动合成电影级视频。本文描述了一种方法，该方法结合了 Stable Diffusion 用于高保真图像合成、GPT-2 用于叙事结构化以及使用 gTTS 和 YouTube 来源音乐的混合音频流水线，以创建 60 秒的电影。该方法采用五场景框架，并通过线性帧插值、电影后处理（如锐化）以及音频-视频同步来提供专业级别的结果。该方法在使用 Python 3.11 的 GPU 加速 Google Colab 环境中创建。它具有支持高达 1024x768 的分辨率和 15-30 FPS 帧率的双重模式 Gradio 接口（简易模式和高级模式）。CUDA 内存管理和错误处理等优化确保了可靠性。实验结果表明，该方法在视觉质量、叙事连贯性和效率方面表现出色，进一步推动了文本到视频合成在创意、教育和工业领域的应用。 

---
# Immersive Multimedia Communication: State-of-the-Art on eXtended Reality Streaming 

**Title (ZH)**: 沉浸式多媒体通信：扩展现实流媒体的现状 

**Authors**: Haopeng Wang, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2506.10004)  

**Abstract**: Extended reality (XR) is rapidly advancing, and poised to revolutionize content creation and consumption. In XR, users integrate various sensory inputs to form a cohesive perception of the virtual environment. This survey reviews the state-of-the-art in XR streaming, focusing on multiple paradigms. To begin, we define XR and introduce various XR headsets along with their multimodal interaction methods to provide a foundational understanding. We then analyze XR traffic characteristics to highlight the unique data transmission requirements. We also explore factors that influence the quality of experience in XR systems, aiming to identify key elements for enhancing user satisfaction. Following this, we present visual attention-based optimization methods for XR streaming to improve efficiency and performance. Finally, we examine current applications and highlight challenges to provide insights into ongoing and future developments of XR. 

**Abstract (ZH)**: 扩展现实（XR）正在rapidly advancing，并有望革命化内容创作与消费。本综述回顾了XR流传输的前沿技术，重点关注多种 paradigms。首先，我们定义XR并介绍各种XR头显及其多模态交互方法，以提供基础知识。接着，我们分析XR网络流量特性，突显其独特的数据传输需求。我们还探讨影响XR系统体验质量的因素，以识别提升用户满意度的关键要素。随后，我们呈现基于视觉注意力的优化方法以提高XR流传输的效率和性能。最后，我们考察当前应用并突出挑战，以提供有关XR持续和未来发展洞察。 

---
# EQ-TAA: Equivariant Traffic Accident Anticipation via Diffusion-Based Accident Video Synthesis 

**Title (ZH)**: EQ-TAA: 基于扩散模型的事故视频合成的同态交通事故预判 

**Authors**: Jianwu Fang, Lei-Lei Li, Zhedong Zheng, Hongkai Yu, Jianru Xue, Zhengguo Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.10002)  

**Abstract**: Traffic Accident Anticipation (TAA) in traffic scenes is a challenging problem for achieving zero fatalities in the future. Current approaches typically treat TAA as a supervised learning task needing the laborious annotation of accident occurrence duration. However, the inherent long-tailed, uncertain, and fast-evolving nature of traffic scenes has the problem that real causal parts of accidents are difficult to identify and are easily dominated by data bias, resulting in a background confounding issue. Thus, we propose an Attentive Video Diffusion (AVD) model that synthesizes additional accident video clips by generating the causal part in dashcam videos, i.e., from normal clips to accident clips. AVD aims to generate causal video frames based on accident or accident-free text prompts while preserving the style and content of frames for TAA after video generation. This approach can be trained using datasets collected from various driving scenes without any extra annotations. Additionally, AVD facilitates an Equivariant TAA (EQ-TAA) with an equivariant triple loss for an anchor accident-free video clip, along with the generated pair of contrastive pseudo-normal and pseudo-accident clips. Extensive experiments have been conducted to evaluate the performance of AVD and EQ-TAA, and competitive performance compared to state-of-the-art methods has been obtained. 

**Abstract (ZH)**: 交通事故预见（TAA）在交通场景中的研究是一个挑战性问题，旨在实现未来的零伤亡目标。当前方法通常将TAA视为需要 laborious 事故发生时间标注的监督学习任务。然而，交通场景固有的长尾分布、不确定性及快速演变特性导致事故的真实因果部分难以识别，且容易受到数据偏差的影响，产生背景混杂问题。因此，我们提出了一种注意视频扩散（AVD）模型，通过在行车记录仪视频中生成因果部分，即从正常片段到事故片段，来合成额外的事故视频片段。AVD 的目标是在事故或无事故文本提示的基础上生成因果视频帧，同时保留视频生成后的帧的风格和内容，以实现TAA。此方法可以通过收集各种驾驶场景下的数据集进行训练，无需额外标注。此外，AVD 还通过锚定无事故视频片段及生成对比伪正常和伪事故片段的不变三重损失，促进了一种不变性TAA（EQ-TAA）。进行了广泛的实验以评估AVD和EQ-TAA的性能，结果表明其在与现有最佳方法的性能上具有竞争力。 

---
# Semantic Communication-Enabled Cloud-Edge-End-collaborative Metaverse Services Architecure 

**Title (ZH)**: 基于语义通信的云-边-端协作元宇宙服务架构 

**Authors**: Yuxuan Li, Sheng Jinag, Bizhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10001)  

**Abstract**: With technology advancing and the pursuit of new audiovisual experiences strengthening, the metaverse has gained surging enthusiasm. However, it faces practical hurdles as substantial data like high-resolution virtual scenes must be transmitted between cloud platforms and VR devices. Specifically, the VR device's wireless transmission hampered by insufficient bandwidth, causes speed and delay problems. Meanwhile, poor channel quality leads to data errors and worsens user experience. To solve this, we've proposed the Semantic Communication-Enabled Cloud-Edge-End Collaborative Immersive Metaverse Service (SC-CEE-Meta) Architecture, which includes three modules: VR video semantic transmission, video synthesis, and 3D virtual scene reconstruction. By deploying semantic modules on VR devices and edge servers and sending key semantic info instead of focusing on bit-level reconstruction, it can cut latency, resolve the resource-bandwidth conflict, and better withstand channel interference. Also, the cloud deploys video synthesis and 3D scene reconstruction preprocessing, while edge devices host 3D reconstruction rendering modules, all for immersive services. Verified on Meta Quest Pro, the SC-CEE-Meta can reduce wireless transmission delay by 96.05\% and boost image quality by 43.99\% under poor channel condition. 

**Abstract (ZH)**: 随着技术的进步和对新视听体验的追求加强，元宇宙获得了高涨的热情。然而，它面临着实际挑战，因为大量高分辨率虚拟场景数据必须在云平台和VR设备之间传输。具体而言，由于无线传输受限于不足的带宽，导致速度和延迟问题。同时，差的信道质量引起数据错误并恶化用户体验。为了解决这些问题，我们提出了基于语义通信的云-边缘-端协作沉浸式元宇宙服务（SC-CEE-Meta）架构，该架构包括三个模块：VR视频语义传输、视频合成和三维虚拟场景重建。通过在VR设备和边缘服务器上部署语义模块，并发送关键语义信息而非注重位级重构，它可降低延迟、解决资源-带宽冲突，并更能承受信道干扰。此外，云部署视频合成和三维场景重建预处理，而边缘设备承载三维重建渲染模块，以提供沉浸式服务。在Meta Quest Pro上验证，SC-CEE-Meta在差的信道条件下可将无线传输延迟降低96.05%并提升图像质量43.99%。 

---
# Resa: Transparent Reasoning Models via SAEs 

**Title (ZH)**: Resa: 通过SAEs实现透明推理模型 

**Authors**: Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, Deqing Fu, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2506.09967)  

**Abstract**: How cost-effectively can we elicit strong reasoning in language models by leveraging their underlying representations? We answer this question with Resa, a family of 1.5B reasoning models trained via a novel and efficient sparse autoencoder tuning (SAE-Tuning) procedure. This method first trains an SAE to capture reasoning abilities from a source model, and then uses the trained SAE to guide a standard supervised fine-tuning process to elicit such abilities in a target model, all using verified question-answer data without any reasoning traces. Notably, when applied to certain base models before further RL post-training, SAE-Tuning retains >97% of its RL-trained counterpart's reasoning performance while reducing training costs by >2000x to roughly \$1 and training time by >450x to around 20 minutes. Furthermore, when applied to lightly RL-trained models (e.g., within 1 hour on 2 GPUs), it enables reasoning performance such as 43.33% Pass@1 on AIME24 and 90% Pass@1 on AMC23 for only around \$1 additional cost. Surprisingly, the reasoning abilities extracted via SAEs are potentially both generalizable and modular. Generality means abilities extracted from one dataset still elevate performance on a larger and overlapping corpus. Modularity means abilities extracted from Qwen or Qwen-Math can be attached to the R1-Distill model at test time, without any retraining, and yield comparable gains. Extensive ablations validate these findings and all artifacts are fully open-sourced. 

**Abstract (ZH)**: 通过利用语言模型的底层表示，我们如何最有效地激发强推理能力？Resa：一种新型高效的稀疏自编码器调优（SAE-Tuning）方法的研究 

---
# Tina: Tiny Reasoning Models via LoRA 

**Title (ZH)**: Tina: Tiny Reasoning Models via LoRA 

**Authors**: Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2504.15777)  

**Abstract**: How cost-effectively can strong reasoning abilities be achieved in language models? Driven by this fundamental question, we present Tina, a family of tiny reasoning models achieved with high cost-efficiency. Notably, Tina demonstrates that substantial reasoning performance can be developed using only minimal resources, by applying parameter-efficient updates during reinforcement learning (RL), using low-rank adaptation (LoRA), to an already tiny 1.5B parameter base model. This minimalist approach produces models that achieve reasoning performance which is competitive with, and sometimes surpasses, SOTA RL reasoning models built upon the same base model. Crucially, this is achieved at a tiny fraction of the computational post-training cost employed by existing SOTA models. In fact, the best Tina model achieves a >20\% reasoning performance increase and 43.33\% Pass@1 accuracy on AIME24, at only \$9 USD post-training and evaluation cost (i.e., an estimated 260x cost reduction). Our work reveals the surprising effectiveness of efficient RL reasoning via LoRA. We validate this across multiple open-source reasoning datasets and various ablation settings starting with a single, fixed set of hyperparameters. Furthermore, we hypothesize that this effectiveness and efficiency stem from LoRA rapidly adapting the model to the structural format of reasoning rewarded by RL, while largely preserving the base model's underlying knowledge. In service of accessibility and open research, we fully open-source all code, training logs, and model weights \& checkpoints. 

**Abstract (ZH)**: 如何以最经济的方式在语言模型中实现强大的推理能力？我们提出了Tina，这是一种通过高成本效率实现的小小推理模型 familia。Tina 显示出，仅通过在强化学习（RL）过程中使用参数高效更新和低秩适应（LoRA）对一个已有的超小型基础模型（1.5B参数）进行少量资源的训练，即可实现显著的推理性能提升。这种 minimalist 方法生成的模型在推理性能上与同基础模型构建的当前最佳强化学习推理模型相比，有时甚至更为出色。更重要的是，这种方法仅使用现有最佳模型所需计算后训练成本的一小部分即可实现。事实上，Tina 最优模型在 AIME24 上实现了超过 20% 的推理性能提升和 43.33% 的 Pass@1 准确率，仅需 9 美元的后训练和评估成本（估计成本降低了约 260 倍）。我们的工作揭示了通过 LoRA 实现高效 RL 推理的惊人效果。我们通过多种开源推理数据集和不同的 ablation 设置进行了验证，使用固定的超参数集进行了初始验证。此外，我们认为这种效果和效率来自于 LoRA 快速使模型适应 RL 奖励的推理结构，并且主要保留了基础模型的内在知识。为了促进可访问性和开放研究，我们完全开源了所有代码、训练日志、模型权重和检查点。 

---
