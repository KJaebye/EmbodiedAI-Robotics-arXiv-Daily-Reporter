# One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture 

**Title (ZH)**: 一专多能：基于大语言模型的精准农业异构任务规划 

**Authors**: Marcos Abel Zuzuárregui, Mustafa Melih Toslak, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10106)  

**Abstract**: Artificial intelligence is transforming precision agriculture, offering farmers new tools to streamline their daily operations. While these technological advances promise increased efficiency, they often introduce additional complexity and steep learning curves that are particularly challenging for non-technical users who must balance tech adoption with existing workloads. In this paper, we present a natural language (NL) robotic mission planner that enables non-specialists to control heterogeneous robots through a common interface. By leveraging large language models (LLMs) and predefined primitives, our architecture seamlessly translates human language into intermediate descriptions that can be executed by different robotic platforms. With this system, users can formulate complex agricultural missions without writing any code. In the work presented in this paper, we extend our previous system tailored for wheeled robot mission planning through a new class of experiments involving robotic manipulation and computer vision tasks. Our results demonstrate that the architecture is both general enough to support a diverse set of robots and powerful enough to execute complex mission requests. This work represents a significant step toward making robotic automation in precision agriculture more accessible to non-technical users. 

**Abstract (ZH)**: 人工智能正在 transforming 精确农业，为农民提供新的工具以精简日常运营。尽管这些技术进步承诺提高效率，但它们通常会引入额外的复杂性和陡峭的学习曲线，这对必须在技术采用与现有工作量之间保持平衡的非技术用户尤其具有挑战性。在本文中，我们提出了一种自然语言（NL）机器人任务规划器，使非专家能够通过通用界面控制异构机器人。借助大规模语言模型（LLMs）和预定义的基本要素，我们的架构无缝地将自然语言转换为可以由不同机器人平台执行的中间描述。通过此系统，用户可以在不编写任何代码的情况下制定复杂的农业任务。在本文中，我们扩展了我们之前针对轮式机器人任务规划的系统，通过涉及机器人操作和计算机视觉任务的新类实验。我们的结果显示，该架构既通用到支持多种类型的机器人，又能执行复杂的任务请求。这项工作代表了使精确农业中的机器人自动化对非技术用户更具可访问性的重大进展。 

---
# Leveraging LLMs for Mission Planning in Precision Agriculture 

**Title (ZH)**: 利用大语言模型在精确农业中的任务规划应用 

**Authors**: Marcos Abel Zuzuárregui, Stefano Carpin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10093)  

**Abstract**: Robotics and artificial intelligence hold significant potential for advancing precision agriculture. While robotic systems have been successfully deployed for various tasks, adapting them to perform diverse missions remains challenging, particularly because end users often lack technical expertise. In this paper, we present an end-to-end system that leverages large language models (LLMs), specifically ChatGPT, to enable users to assign complex data collection tasks to autonomous robots using natural language instructions. To enhance reusability, mission plans are encoded using an existing IEEE task specification standard, and are executed on robots via ROS2 nodes that bridge high-level mission descriptions with existing ROS libraries. Through extensive experiments, we highlight the strengths and limitations of LLMs in this context, particularly regarding spatial reasoning and solving complex routing challenges, and show how our proposed implementation overcomes them. 

**Abstract (ZH)**: 机器人技术和人工智能在精准农业领域的应用具有重要潜力。尽管已成功部署了各种机器人系统，但将它们适应执行多种任务仍然具有挑战性，尤其是因为最终用户往往缺乏技术 expertise。在这种情况下，我们提出了一套端到端的系统，利用大规模语言模型（LLMs），特别是ChatGPT，使用户能够使用自然语言指令将复杂的数据收集任务分配给自主机器人。为了提高可重用性，任务计划使用现有的IEEE任务规范标准进行编码，并通过ROS2节点执行，该节点将高级任务描述与现有的ROS库连接起来。通过广泛实验，我们强调了LLMs在这一具体场景中的优势和局限性，特别是关于空间推理和解决复杂路径挑战的能力，并展示了我们提出的实现方法如何克服这些挑战。 

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
# AutoMind: Adaptive Knowledgeable Agent for Automated Data Science 

**Title (ZH)**: AutoMind：自适应知识型自动化数据科学代理 

**Authors**: Yixin Ou, Yujie Luo, Jingsheng Zheng, Lanning Wei, Shuofei Qiao, Jintian Zhang, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10974)  

**Abstract**: Large Language Model (LLM) agents have shown great potential in addressing real-world data science problems. LLM-driven data science agents promise to automate the entire machine learning pipeline, yet their real-world effectiveness remains limited. Existing frameworks depend on rigid, pre-defined workflows and inflexible coding strategies; consequently, they excel only on relatively simple, classical problems and fail to capture the empirical expertise that human practitioners bring to complex, innovative tasks. In this work, we introduce AutoMind, an adaptive, knowledgeable LLM-agent framework that overcomes these deficiencies through three key advances: (1) a curated expert knowledge base that grounds the agent in domain expert knowledge, (2) an agentic knowledgeable tree search algorithm that strategically explores possible solutions, and (3) a self-adaptive coding strategy that dynamically tailors code generation to task complexity. Evaluations on two automated data science benchmarks demonstrate that AutoMind delivers superior performance versus state-of-the-art baselines. Additional analyses confirm favorable effectiveness, efficiency, and qualitative solution quality, highlighting AutoMind as an efficient and robust step toward fully automated data science. 

**Abstract (ZH)**: 大语言模型（LLM）代理在解决实际数据科学问题方面表现出巨大潜力。虽然LLM驱动的数据科学代理有望自动化整个机器学习管道，但其实用效果仍受到限制。现有框架依赖于僵化的预定义工作流和僵硬的编程策略，仅在相对简单的经典问题上表现出色，无法捕捉到人类实践者在复杂、创新任务中带来的经验知识。本文介绍了一种名为AutoMind的自适应、知识丰富的LLM代理框架，通过三个方面的主要进步克服了这些缺陷：（1）精心策划的专家知识库，使代理扎根于领域专家知识；（2）一种智能的知识型树搜索算法，战略性地探索可能的解决方案；（3）一种自我适应的编程策略，动态调整代码生成以适应任务复杂性。对两种自动数据科学基准的评估显示，AutoMind在性能上优于现有先进基线。进一步的分析证实了其有利的有效性、效率和定性解决方案质量，突显了AutoMind是迈向完全自动化数据科学的高效且稳健的一步。 

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
# ChineseHarm-Bench: A Chinese Harmful Content Detection Benchmark 

**Title (ZH)**: ChineseHarm-Bench: 中文有害内容检测基准 

**Authors**: Kangwei Liu, Siyuan Cheng, Bozhong Tian, Xiaozhuan Liang, Yuyang Yin, Meng Han, Ningyu Zhang, Bryan Hooi, Xi Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10960)  

**Abstract**: Large language models (LLMs) have been increasingly applied to automated harmful content detection tasks, assisting moderators in identifying policy violations and improving the overall efficiency and accuracy of content review. However, existing resources for harmful content detection are predominantly focused on English, with Chinese datasets remaining scarce and often limited in scope. We present a comprehensive, professionally annotated benchmark for Chinese content harm detection, which covers six representative categories and is constructed entirely from real-world data. Our annotation process further yields a knowledge rule base that provides explicit expert knowledge to assist LLMs in Chinese harmful content detection. In addition, we propose a knowledge-augmented baseline that integrates both human-annotated knowledge rules and implicit knowledge from large language models, enabling smaller models to achieve performance comparable to state-of-the-art LLMs. Code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动有害内容检测任务中的应用日益增多，助力モデरレーター政策违规行为的识别，并提高内容审核的整体效率和准确性。然而，现有的有害内容检测资源主要集中在英语上，中文数据集相对稀少且范围有限。我们提出了一种全面的专业注释基准数据集，涵盖六个代表性类别，并完全基于真实数据构建。我们的注释过程还产出了一种知识规则库，为大型语言模型提供显式的专家知识以辅助中文有害内容检测。此外，我们提出了一种知识增强的基础模型，结合了人工注释的知识规则和大型语言模型中的潜在知识，使较小的模型能够达到最先进的大型语言模型的性能。代码和数据可在此链接访问。 

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
# Robustly Improving LLM Fairness in Realistic Settings via Interpretability 

**Title (ZH)**: 通过可解释性在实际场景中稳健提高大语言模型公平性 

**Authors**: Adam Karvonen, Samuel Marks  

**Link**: [PDF](https://arxiv.org/pdf/2506.10922)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios. Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning. To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance. Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）在高风险招聘应用中越来越广泛部署，对其直接决定人们的职业生涯和生计产生影响。虽然以往研究表明简单的反偏见提示可以在受控评估中消除人口统计学偏见，但我们发现当引入现实的上下文细节时，这些缓解措施会失效。我们通过内部偏见缓解措施解决了这些失败：通过识别并中和模型激活中的敏感属性方向，我们在所有测试场景中实现了稳健的偏见减少。在领先的商品化（GPT-4o、Claude 4 Sonnet、Gemini 2.5 Flash）和开源模型（Gemma-2 27B、Gemma-3、Mistral-24B）中，我们发现添加现实上下文，如公司名称、来自公共职业页面的文化描述以及选择性的招聘限制（例如，“仅接受前10%的候选人”），会显著引入种族和性别偏见（面试率最多相差12%）。当这些偏见出现时，它们在所有测试模型和场景中始终表现为更倾向于黑人而非白人候选人和女性而非男性候选人。此外，模型可以从微妙线索（如大学隶属关系）中推断出人口统计信息并产生偏见，即使检查模型的推理链也不易察觉这些偏见。为了应对这些局限性，我们的内部偏见缓解措施识别与种族和性别相关的方向，并在推理时应用仿射概念编辑。尽管使用来自简单合成数据集的方向，干预措施表现出稳健性，一致将偏见降低到非常低的水平（通常低于1%，始终低于2.5%），同时很大程度上保持了模型性能。我们的研究结果表明，部署LLMs进行招聘的实践者应采用更现实的评估方法，并考虑内部缓解策略以实现公平结果。 

---
# BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP 

**Title (ZH)**: BioClinical ModernBERT：一种生物医学和临床NLP领域的先进长上下文编码器 

**Authors**: Thomas Sounack, Joshua Davis, Brigitte Durieux, Antoine Chaffin, Tom J. Pollard, Eric Lehman, Alistair E. W. Johnson, Matthew McDermott, Tristan Naumann, Charlotta Lindvall  

**Link**: [PDF](https://arxiv.org/pdf/2506.10896)  

**Abstract**: Encoder-based transformer models are central to biomedical and clinical Natural Language Processing (NLP), as their bidirectional self-attention makes them well-suited for efficiently extracting structured information from unstructured text through discriminative tasks. However, encoders have seen slower development compared to decoder models, leading to limited domain adaptation in biomedical and clinical settings. We introduce BioClinical ModernBERT, a domain-adapted encoder that builds on the recent ModernBERT release, incorporating long-context processing and substantial improvements in speed and performance for biomedical and clinical NLP. BioClinical ModernBERT is developed through continued pretraining on the largest biomedical and clinical corpus to date, with over 53.5 billion tokens, and addresses a key limitation of prior clinical encoders by leveraging 20 datasets from diverse institutions, domains, and geographic regions, rather than relying on data from a single source. It outperforms existing biomedical and clinical encoders on four downstream tasks spanning a broad range of use cases. We release both base (150M parameters) and large (396M parameters) versions of BioClinical ModernBERT, along with training checkpoints to support further research. 

**Abstract (ZH)**: 基于编码器的变压器模型在生物医学和临床自然语言处理(NLP)中至关重要，因为它们的双向自注意力机制使其能够通过辨别任务高效地从无结构文本中提取结构化信息。然而，相较于解码器模型，编码器的发展较慢，导致生物医学和临床环境中的领域适应能力有限。我们介绍了BioClinical ModernBERT，这是一种基于最近发布的ModernBERT改进而来的领域适应编码器，它结合了长上下文处理，并在生物医学和临床NLP方面在速度和性能上取得了显著改进。BioClinical ModernBERT通过在迄今最大规模的生物医学和临床文本语料库上进行连续预训练（超过535亿个 token），并利用来自不同机构、领域和地理区域的20个数据集，解决了先前临床编码器的关键限制，而无需依赖单一数据源。它在四个下游任务上优于现有的生物医学和临床编码器，这些任务涵盖了广泛的应用场景。我们发布了BioClinical ModernBERT的基本版本（1.5亿参数）和大型版本（3.96亿参数），并提供了训练检查点以支持进一步研究。 

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
# Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information 

**Title (ZH)**: 通过后聚合全局上下文信息的精确零样本点wise排名lescoringewithLLMs 

**Authors**: Kehan Long, Shasha Li, Chen Xu, Jintao Tang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10859)  

**Abstract**: Recent advancements have successfully harnessed the power of Large Language Models (LLMs) for zero-shot document ranking, exploring a variety of prompting strategies. Comparative approaches like pairwise and listwise achieve high effectiveness but are computationally intensive and thus less practical for larger-scale applications. Scoring-based pointwise approaches exhibit superior efficiency by independently and simultaneously generating the relevance scores for each candidate document. However, this independence ignores critical comparative insights between documents, resulting in inconsistent scoring and suboptimal performance. In this paper, we aim to improve the effectiveness of pointwise methods while preserving their efficiency through two key innovations: (1) We propose a novel Global-Consistent Comparative Pointwise Ranking (GCCP) strategy that incorporates global reference comparisons between each candidate and an anchor document to generate contrastive relevance scores. We strategically design the anchor document as a query-focused summary of pseudo-relevant candidates, which serves as an effective reference point by capturing the global context for document comparison. (2) These contrastive relevance scores can be efficiently Post-Aggregated with existing pointwise methods, seamlessly integrating essential Global Context information in a training-free manner (PAGC). Extensive experiments on the TREC DL and BEIR benchmark demonstrate that our approach significantly outperforms previous pointwise methods while maintaining comparable efficiency. Our method also achieves competitive performance against comparative methods that require substantially more computational resources. More analyses further validate the efficacy of our anchor construction strategy. 

**Abstract (ZH)**: 最近的研究成功利用了大型语言模型（LLMs）进行零样本文档排名，并探索了多种提示策略。对比方法如对切和列表方法表现出较高的有效性，但计算成本高，因此在大规模应用中不太实用。基于评分的点wise方法通过独立同时生成每个候选文档的相关性评分，展现出优越的效率。然而，这种独立性忽略了文档之间重要的对比洞察，导致评分不一致且性能欠佳。本文旨在通过两项关键创新来提高点wise方法的有效性，同时保持其效率：(1) 我们提出了一种新颖的全局一致对比点wise排名（GCCP）策略，该策略在每个候选文档和锚文档之间引入全局引用对比，生成对比相关性评分。我们战略性地将锚文档设计为基于查询的伪相关候选摘要，这为文档对比提供了一个有效的参考点，捕获了文档比较的全局上下文。(2) 这些对比相关性评分可以与现有的点wise方法高效地后聚合（PAGC），以无训练方式无缝整合关键的全局上下文信息。在TREC DL和BEIR基准上的广泛实验表明，我们的方法在保持同等效率的同时显著优于之前的方法。此外，我们的方法在需要更大量计算资源的对比方法中也取得了竞争力的性能。更多分析进一步验证了我们锚构建策略的有效性。 

---
# Accelerating Diffusion Large Language Models with SlowFast: The Three Golden Principles 

**Title (ZH)**: 用SlowFast加速扩散大型语言模型：三大黄金原则 

**Authors**: Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10848)  

**Abstract**: Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across benchmarks and models show that SlowFast Sampling achieves up to 15.63$\times$ speedup on LLaDA with minimal accuracy drop, and up to 34.22$\times$ when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation. 

**Abstract (ZH)**: 基于扩散的语言模型（dLLMs）通过实现并行令牌生成和显著降低推断延迟，已成为传统自回归LLMs的有前途的替代方案。然而，现有的dLLMs抽样策略，如基于信心或半自回归解码，经常表现出静态行为，导致效率低下和灵活性受限。在本文中，我们提出了一种新颖的动态抽样策略SlowFast Sampling，该策略能够自适应地交替使用探索性和加速解码阶段。我们的方法遵循三个金科玉律：确定性原则、收敛性原则和位置性原则，以决定何时以及在哪里可以有效地解码令牌。我们还将我们的策略与dLLM-Cache结合使用，以减少冗余计算。在多个基准测试和模型上的广泛实验显示，SlowFast Sampling在LLaDA上的速度提升可达15.63倍，精度下降 minimal，在结合缓存时可达34.22倍。值得注意的是，我们的方法在吞吐量方面优于强自回归基线如LLaMA3 8B，证明了精心设计的抽样可以充分发挥dLLMs的潜力以实现快速和高质量的生成。 

---
# LLM-Driven Personalized Answer Generation and Evaluation 

**Title (ZH)**: LLM驱动的个性化答案生成与评估 

**Authors**: Mohammadreza Molavi, Mohammadreza Tavakoli, Mohammad Moein, Abdolali Faraji, Gábor Kismihók  

**Link**: [PDF](https://arxiv.org/pdf/2506.10829)  

**Abstract**: Online learning has experienced rapid growth due to its flexibility and accessibility. Personalization, adapted to the needs of individual learners, is crucial for enhancing the learning experience, particularly in online settings. A key aspect of personalization is providing learners with answers customized to their specific questions. This paper therefore explores the potential of Large Language Models (LLMs) to generate personalized answers to learners' questions, thereby enhancing engagement and reducing the workload on educators. To evaluate the effectiveness of LLMs in this context, we conducted a comprehensive study using the StackExchange platform in two distinct areas: language learning and programming. We developed a framework and a dataset for validating automatically generated personalized answers. Subsequently, we generated personalized answers using different strategies, including 0-shot, 1-shot, and few-shot scenarios. The generated answers were evaluated using three methods: 1. BERTScore, 2. LLM evaluation, and 3. human evaluation. Our findings indicated that providing LLMs with examples of desired answers (from the learner or similar learners) can significantly enhance the LLMs' ability to tailor responses to individual learners' needs. 

**Abstract (ZH)**: 在线学习由于其灵活性和易访问性经历了快速成长。个性化学习，适应个别学习者的需求，对于增强学习体验，特别是在在线环境中尤为重要。个性化的一个关键方面是为学习者提供针对其特定问题的定制答案。因此，本文探索了大规模语言模型（LLMs）生成学习者个性化答案的潜力，从而提高参与度并减轻教育者的负担。为了评估LLMs在这种情境下的有效性，我们在StackExchange平台的两个不同领域——语言学习和编程——进行了全面的研究。我们开发了一个框架和数据集来验证自动生成的个性化答案。之后，我们使用不同的策略生成了个性化答案，包括零样本、一样本和少样本场景。生成的答案通过三种方法进行了评估：1. BERTScore，2. LLM评估，3. 人工评估。我们的研究表明，为LLMs提供所需答案的示例（来自学习者或类似学习者）可以显著增强其根据个别学习者需求定制回应的能力。 

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
# PREMISE: Scalable and Strategic Prompt Optimization for Efficient Mathematical Reasoning in Large Models 

**Title (ZH)**: 前提：面向大规模模型的可扩展与策略性提示优化以实现高效的数学推理 

**Authors**: Ye Yu, Yaoning Yu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10716)  

**Abstract**: Large reasoning models (LRMs) such as Claude 3.7 Sonnet and OpenAI o1 achieve strong performance on mathematical benchmarks using lengthy chain-of-thought (CoT) reasoning, but the resulting traces are often unnecessarily verbose. This inflates token usage and cost, limiting deployment in latency-sensitive or API-constrained settings. We introduce PREMISE (PRompt-based Efficient Mathematical Inference with Strategic Evaluation), a prompt-only framework that reduces reasoning overhead without modifying model weights. PREMISE combines trace-level diagnostics with gradient-inspired prompt optimization to minimize redundant computation while preserving answer accuracy. The approach jointly optimizes brevity and correctness through a multi-objective textual search that balances token length and answer validity. Unlike prior work, PREMISE runs in a single-pass black-box interface, so it can be applied directly to commercial LLMs. On GSM8K, SVAMP, and Math500 we match or exceed baseline accuracy ($96\%\rightarrow96\%$ with Claude, $91\%\rightarrow92\%$ with Gemini) while reducing reasoning tokens by up to $87.5\%$ and cutting dollar cost by $69$--$82\%$. These results show that prompt-level optimization is a practical and scalable path to efficient LRM inference without compromising reasoning quality. 

**Abstract (ZH)**: 基于提示的高效数学推理与战略评估（PREMISE） 

---
# ConTextTab: A Semantics-Aware Tabular In-Context Learner 

**Title (ZH)**: Semantics-Aware Tabular In-Context Learner: ConTextTab 

**Authors**: Marco Spinaci, Marek Polewczyk, Maximilian Schambach, Sam Thelin  

**Link**: [PDF](https://arxiv.org/pdf/2506.10707)  

**Abstract**: Tabular in-context learning (ICL) has recently achieved state-of-the-art (SOTA) performance on several tabular prediction tasks. Previously restricted to classification problems on small tables, recent advances such as TabPFN and TabICL have extended its use to larger datasets. While being architecturally efficient and well-adapted to tabular data structures, current table-native ICL architectures, being trained exclusively on synthetic data, do not fully leverage the rich semantics and world knowledge contained in real-world tabular data. On another end of this spectrum, tabular ICL models based on pretrained large language models such as TabuLa-8B integrate deep semantic understanding and world knowledge but are only able to make use of a small amount of context due to inherent architectural limitations. With the aim to combine the best of both these worlds, we introduce ConTextTab, integrating semantic understanding and alignment into a table-native ICL framework. By employing specialized embeddings for different data modalities and by training on large-scale real-world tabular data, our model is competitive with SOTA across a broad set of benchmarks while setting a new standard on the semantically rich CARTE benchmark. 

**Abstract (ZH)**: 表格内模态学习（ICL）近年来已在多个表格预测任务中取得了最先进的性能。虽然当前的表本源ICL架构在合成数据上训练，能够在架构上高效且很好地适应表格数据结构，但未能充分利用真实世界表格数据中丰富的语义和世界知识。另一方面，基于预训练大规模语言模型（如TabuLa-8B）的表格ICL模型虽然能够整合深入的语义理解和世界知识，但由于固有的架构限制，只能利用少量的上下文信息。为了兼顾两者的优势，我们引入了ConTextTab，将其语义理解和对齐整合到一个表本源ICL框架中。通过使用不同数据模态的专业嵌入，并在大规模真实世界表格数据上进行训练，我们的模型在多个基准测试中与最先进的技术竞争，特别是在语义丰富的CARTE基准测试中树立了新的标准。 

---
# Large Language Models for Detection of Life-Threatening Texts 

**Title (ZH)**: 大型语言模型在危及生命文本检测中的应用 

**Authors**: Thanh Thi Nguyen, Campbell Wilson, Janis Dalins  

**Link**: [PDF](https://arxiv.org/pdf/2506.10687)  

**Abstract**: Detecting life-threatening language is essential for safeguarding individuals in distress, promoting mental health and well-being, and preventing potential harm and loss of life. This paper presents an effective approach to identifying life-threatening texts using large language models (LLMs) and compares them with traditional methods such as bag of words, word embedding, topic modeling, and Bidirectional Encoder Representations from Transformers. We fine-tune three open-source LLMs including Gemma, Mistral, and Llama-2 using their 7B parameter variants on different datasets, which are constructed with class balance, imbalance, and extreme imbalance scenarios. Experimental results demonstrate a strong performance of LLMs against traditional methods. More specifically, Mistral and Llama-2 models are top performers in both balanced and imbalanced data scenarios while Gemma is slightly behind. We employ the upsampling technique to deal with the imbalanced data scenarios and demonstrate that while this method benefits traditional approaches, it does not have as much impact on LLMs. This study demonstrates a great potential of LLMs for real-world life-threatening language detection problems. 

**Abstract (ZH)**: 检测危及生命的语言对于保护陷入困境的个体、促进心理健康和福祉以及预防潜在危害和生命损失至关重要。本文提出了一种有效的方法，使用大型语言模型（LLMs）来识别危及生命的文字，并将其与传统的词袋模型、词嵌入、主题建模和双向Transformer表示等方法进行了比较。我们使用不同数据集对三个开源LLM（Gemma、Mistral和Llama-2）的7B参数变体进行了微调，这些数据集构建了类别平衡、不平衡和极端不平衡的场景。实验结果表明，LLMs在传统方法面前表现出色。具体而言，在平衡和不平衡的数据场景中，Mistral和Llama-2模型表现最佳，而Gemma稍逊一筹。我们采用上采样技术来应对不平衡数据场景，并证明虽然这种方法对传统方法有益，但对LLMs的影响较小。本研究展示了LLMs在实际危及生命语言检测问题中的巨大潜力。 

---
# Data Shifts Hurt CoT: A Theoretical Study 

**Title (ZH)**: 数据偏移损害共注意力：一项理论研究 

**Authors**: Lang Yin, Debangshu Banerjee, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.10647)  

**Abstract**: Chain of Thought (CoT) has been applied to various large language models (LLMs) and proven to be effective in improving the quality of outputs. In recent studies, transformers are proven to have absolute upper bounds in terms of expressive power, and consequently, they cannot solve many computationally difficult problems. However, empowered by CoT, transformers are proven to be able to solve some difficult problems effectively, such as the $k$-parity problem. Nevertheless, those works rely on two imperative assumptions: (1) identical training and testing distribution, and (2) corruption-free training data with correct reasoning steps. However, in the real world, these assumptions do not always hold. Although the risks of data shifts have caught attention, our work is the first to rigorously study the exact harm caused by such shifts to the best of our knowledge. Focusing on the $k$-parity problem, in this work we investigate the joint impact of two types of data shifts: the distribution shifts and data poisoning, on the quality of trained models obtained by a well-established CoT decomposition. In addition to revealing a surprising phenomenon that CoT leads to worse performance on learning parity than directly generating the prediction, our technical results also give a rigorous and comprehensive explanation of the mechanistic reasons of such impact. 

**Abstract (ZH)**: Chain of Thought (CoT)在大型语言模型中的应用及其数据偏移影响的研究 

---
# Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs 

**Title (ZH)**: 时间序列预测作为一种推理：强化大型语言模型的慢思考方法 

**Authors**: Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, Jintao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10630)  

**Abstract**: To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations - including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow thinking capabilities and acquire strong time series reasoning skills. For this purpose, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets. 

**Abstract (ZH)**: 改进时间序列预测的LLM双阶段强化细调框架：Time-R1 

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
# SoK: Evaluating Jailbreak Guardrails for Large Language Models 

**Title (ZH)**: SoK: 评估大型语言模型的监禁门槛约束 

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10597)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）取得了显著进展，但其部署暴露了关键漏洞，特别是绕过安全机制的越界攻击。边界措施——外部防护机制，用于监控和控制LLM交互——已成为一种有前景的解决方案。然而，当前的LLM边界措施 landscape 是碎片化的，缺乏统一的分类体系和全面的评估框架。在本文综述性论文（SoK）中，我们首次对LLM越界边界措施进行了全面分析。我们提出了一种新颖的多维度分类法，按照六个关键维度对边界措施进行分类，并引入了安全-效率-实用性评估框架以评估它们的实际效果。通过广泛的分析和实验，我们识别了现有边界措施的优势和局限性，探讨了它们在不同类型攻击下的普适性，并提供了优化防御组合的见解。我们的工作为未来的研究和开发提供了一个结构化的基础，旨在指导稳健的LLM边界措施的原理发展和部署。相关代码可在以下链接找到：this https URL。 

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
# Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications 

**Title (ZH)**: 多-agent 大语言模型系统的设计与评估——原型及网络安全应用 

**Authors**: Felix Härer  

**Link**: [PDF](https://arxiv.org/pdf/2506.10467)  

**Abstract**: Recent advancements in LLMs indicate potential for novel applications, e.g., through reasoning capabilities in the latest OpenAI and DeepSeek models. For applying these models in specific domains beyond text generation, LLM-based multi-agent approaches can be utilized that solve complex tasks by combining reasoning techniques, code generation, and software execution. Applications might utilize these capabilities and the knowledge of specialized LLM agents. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application is not explored well. Defined specifications for multi-agent LLM systems are required to explore their potential and their suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research to specify and evaluate these aspects through a multi-agent system. The system architecture and prototype are extended from previous research and a specification is introduced for multi-agent systems. Test cases involving cybersecurity tasks indicate feasibility of the architecture and evaluation approach. In particular, the results show the evaluation of question answering, server security, and network security tasks that were completed correctly by agents with LLMs from OpenAI and DeepSeek. 

**Abstract (ZH)**: 近期大规模语言模型的进展表明了其在新型应用中的潜力，例如最新OpenAI和DeepSeek模型的推理能力。为了将这些模型应用于超越文本生成的具体领域，可以利用基于大规模语言模型的多智能体方法，通过结合推理技术、代码生成和软件执行来解决复杂任务。这些应用可能利用大规模语言模型智能体的能力和专业知识。然而，虽然对大规模语言模型、推理技术和应用进行了单独评估，但它们的联合规范及其组合应用尚未得到充分探索。为了探索多智能体大规模语言模型系统的潜力及其在特定应用中的适用性，需要定义明确的规范。本文报告了通过多智能体系统来规范和评估这些方面的一项探索性研究结果。该系统架构和原型基于先前的研究扩展，并引入了多智能体系统的规范。涉及网络安全任务的测试案例表明了该架构和评估方法的可行性。特别是，结果展示了利用来自OpenAI和DeepSeek的大规模语言模型的智能体正确完成的问题回答、服务器安全和网络安防任务的评估。 

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
# Discovering Hierarchical Latent Capabilities of Language Models via Causal Representation Learning 

**Title (ZH)**: 通过因果表示学习发现语言模型的分层潜在能力 

**Authors**: Jikai Jin, Vasilis Syrgkanis, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10378)  

**Abstract**: Faithful evaluation of language model capabilities is crucial for deriving actionable insights that can inform model development. However, rigorous causal evaluations in this domain face significant methodological challenges, including complex confounding effects and prohibitive computational costs associated with extensive retraining. To tackle these challenges, we propose a causal representation learning framework wherein observed benchmark performance is modeled as a linear transformation of a few latent capability factors. Crucially, these latent factors are identified as causally interrelated after appropriately controlling for the base model as a common confounder. Applying this approach to a comprehensive dataset encompassing over 1500 models evaluated across six benchmarks from the Open LLM Leaderboard, we identify a concise three-node linear causal structure that reliably explains the observed performance variations. Further interpretation of this causal structure provides substantial scientific insights beyond simple numerical rankings: specifically, we reveal a clear causal direction starting from general problem-solving capabilities, advancing through instruction-following proficiency, and culminating in mathematical reasoning ability. Our results underscore the essential role of carefully controlling base model variations during evaluation, a step critical to accurately uncovering the underlying causal relationships among latent model capabilities. 

**Abstract (ZH)**: 真实评估语言模型能力对于提取可操作的洞察以指导模型开发至关重要。然而，在此领域进行严格的因果评估面临着重大的方法论挑战，包括复杂的共变量效应以及与大量重新训练相关的高昂计算成本。为应对这些挑战，我们提出了一种因果表示学习框架，其中观察到的基准性能被建模为少量潜在能力因子的线性变换。关键的是，在适当控制基模型作为共同共变量之后，这些潜在因子被识别为彼此因果相关。将这种方法应用于包括来自Open LLM Leaderboard的六个基准测试中超过1500个模型的数据集，我们识别出一个可靠的三个节点线性因果结构，该结构能解释观察到的性能变化。进一步对这一因果结构的解释提供了超出简单数值排名的大量科学洞见：具体而言，我们揭示了一个明确的因果方向，从一般问题解决能力出发，经过指令跟随熟练程度，最终达到数学推理能力。我们的结果强调了在评估过程中仔细控制基模型变异性的关键作用，这是准确揭示潜在模型能力之间因果关系的必要步骤。 

---
# Code Execution as Grounded Supervision for LLM Reasoning 

**Title (ZH)**: 代码执行作为LLM推理的基于地面监督的方法 

**Authors**: Dongwon Jung, Wenxuan Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10343)  

**Abstract**: Training large language models (LLMs) with chain-of-thought (CoT) supervision has proven effective for enhancing their reasoning abilities. However, obtaining reliable and accurate reasoning supervision remains a significant challenge. We propose a scalable method for generating a high-quality CoT supervision dataset by leveraging the determinism of program execution. Unlike existing reasoning dataset generation methods that rely on costly human annotations or error-prone LLM-generated CoT, our approach extracts verifiable, step-by-step reasoning traces from code execution and transforms them into a natural language CoT reasoning. Experiments on reasoning benchmarks across various domains show that our method effectively equips LLMs with transferable reasoning abilities across diverse tasks. Furthermore, the ablation studies validate that our method produces highly accurate reasoning data and reduces overall token length during inference by reducing meaningless repetition and overthinking. 

**Abstract (ZH)**: 利用程序执行的确定性生成高质量链式思维监督数据集以增强大型语言模型的推理能力 

---
# Augmenting Large Language Models with Static Code Analysis for Automated Code Quality Improvements 

**Title (ZH)**: 利用静态代码分析增强大型语言模型以实现自动化代码质量改进 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10330)  

**Abstract**: This study examined code issue detection and revision automation by integrating Large Language Models (LLMs) such as OpenAI's GPT-3.5 Turbo and GPT-4o into software development workflows. A static code analysis framework detects issues such as bugs, vulnerabilities, and code smells within a large-scale software project. Detailed information on each issue was extracted and organized to facilitate automated code revision using LLMs. An iterative prompt engineering process is applied to ensure that prompts are structured to produce accurate and organized outputs aligned with the project requirements. Retrieval-augmented generation (RAG) is implemented to enhance the relevance and precision of the revisions, enabling LLM to access and integrate real-time external knowledge. The issue of LLM hallucinations - where the model generates plausible but incorrect outputs - is addressed by a custom-built "Code Comparison App," which identifies and corrects erroneous changes before applying them to the codebase. Subsequent scans using the static code analysis framework revealed a significant reduction in code issues, demonstrating the effectiveness of combining LLMs, static analysis, and RAG to improve code quality, streamline the software development process, and reduce time and resource expenditure. 

**Abstract (ZH)**: 本研究通过将如OpenAI的GPT-3.5 Turbo和GPT-4这样的大规模语言模型（LLMs）整合到软件开发工作流程中，研究了代码问题检测与自动修订的自动化。静态代码分析框架在大型软件项目中检测错误、漏洞和代码气味等问题，并提取和组织详细信息以供使用LLMs进行自动代码修订。通过迭代的提示工程过程确保提示结构化，生成与项目需求一致的准确且组织良好的输出。通过检索增强生成（RAG）提升修订的相关性和精确度，使LLMs能够访问和整合实时外部知识。通过自定义构建的“代码比较应用”解决了LLMs幻觉问题，即模型生成看似合理但实际上错误的输出，该应用在将更改应用于代码库之前识别并纠正了错误变化。后续使用静态代码分析框架的扫描显示，通过结合LLMs、静态分析和RAG的方法，显著减少了代码问题，证明了提高代码质量、简化软件开发过程并减少时间和资源支出的有效性。 

---
# Towards Understanding Bias in Synthetic Data for Evaluation 

**Title (ZH)**: 理解合成数据在评估中偏见的问题 

**Authors**: Hossein A. Rahmani, Varsha Ramineni, Nick Craswell, Bhaskar Mitra, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10301)  

**Abstract**: Test collections are crucial for evaluating Information Retrieval (IR) systems. Creating a diverse set of user queries for these collections can be challenging, and obtaining relevance judgments, which indicate how well retrieved documents match a query, is often costly and resource-intensive. Recently, generating synthetic datasets using Large Language Models (LLMs) has gained attention in various applications. While previous work has used LLMs to generate synthetic queries or documents to improve ranking models, using LLMs to create synthetic test collections is still relatively unexplored. Previous work~\cite{rahmani2024synthetic} showed that synthetic test collections have the potential to be used for system evaluation, however, more analysis is needed to validate this claim. In this paper, we thoroughly investigate the reliability of synthetic test collections constructed using LLMs, where LLMs are used to generate synthetic queries, labels, or both. In particular, we examine the potential biases that might occur when such test collections are used for evaluation. We first empirically show the presence of such bias in evaluation results and analyse the effects it might have on system evaluation. We further validate the presence of such bias using a linear mixed-effects model. Our analysis shows that while the effect of bias present in evaluation results obtained using synthetic test collections could be significant, for e.g.~computing absolute system performance, its effect may not be as significant in comparing relative system performance. Codes and data are available at: this https URL. 

**Abstract (ZH)**: 使用大规模语言模型生成的合成测试集合在信息检索系统评估中的可靠性探究 

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
# LaMAGIC2: Advanced Circuit Formulations for Language Model-Based Analog Topology Generation 

**Title (ZH)**: LaMAGIC2：基于语言模型的模拟拓扑生成高级电路公式 

**Authors**: Chen-Chia Chang, Wan-Hsuan Lin, Yikang Shen, Yiran Chen, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10235)  

**Abstract**: Automation of analog topology design is crucial due to customized requirements of modern applications with heavily manual engineering efforts. The state-of-the-art work applies a sequence-to-sequence approach and supervised finetuning on language models to generate topologies given user specifications. However, its circuit formulation is inefficient due to O(|V |2) token length and suffers from low precision sensitivity to numeric inputs. In this work, we introduce LaMAGIC2, a succinct float-input canonical formulation with identifier (SFCI) for language model-based analog topology generation. SFCI addresses these challenges by improving component-type recognition through identifier-based representations, reducing token length complexity to O(|V |), and enhancing numeric precision sensitivity for better performance under tight tolerances. Our experiments demonstrate that LaMAGIC2 achieves 34% higher success rates under a tight tolerance of 0.01 and 10X lower MSEs compared to a prior method. LaMAGIC2 also exhibits better transferability for circuits with more vertices with up to 58.5% improvement. These advancements establish LaMAGIC2 as a robust framework for analog topology generation. 

**Abstract (ZH)**: 基于语言模型的模拟拓扑生成的简洁浮点输入标准化表示（LaMAGIC2） 

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
# Unsupervised Elicitation of Language Models 

**Title (ZH)**: 无监督语言模型 elicitation 方法 

**Authors**: Jiaxin Wen, Zachary Ankner, Arushi Somani, Peter Hase, Samuel Marks, Jacob Goldman-Wetzler, Linda Petrini, Henry Sleight, Collin Burns, He He, Shi Feng, Ethan Perez, Jan Leike  

**Link**: [PDF](https://arxiv.org/pdf/2506.10139)  

**Abstract**: To steer pretrained language models for downstream tasks, today's post-training paradigm relies on humans to specify desired behaviors. However, for models with superhuman capabilities, it is difficult or impossible to get high-quality human supervision. To address this challenge, we introduce a new unsupervised algorithm, Internal Coherence Maximization (ICM), to fine-tune pretrained language models on their own generated labels, \emph{without external supervision}. On GSM8k-verification, TruthfulQA, and Alpaca reward modeling tasks, our method matches the performance of training on golden supervision and outperforms training on crowdsourced human supervision. On tasks where LMs' capabilities are strongly superhuman, our method can elicit those capabilities significantly better than training on human labels. Finally, we show that our method can improve the training of frontier LMs: we use our method to train an unsupervised reward model and use reinforcement learning to train a Claude 3.5 Haiku-based assistant. Both the reward model and the assistant outperform their human-supervised counterparts. 

**Abstract (ZH)**: 为了Fine-tune预训练语言模型以适应下游任务，当前的后训练范式依赖人类指定所需行为。但对于具有超人类能力的模型，获得高质量的人类监督变得困难或不可能。为了解决这一挑战，我们引入了一种新的无监督算法——内部一致性最大化（ICM），用于在模型自身生成的标签上Fine-tune预训练语言模型，而无需外部监督。在GSM8k-verification、TruthfulQA和Alpaca奖励模型任务中，我们的方法匹配使用金标准监督的效果，并优于使用众包人类监督的效果。在语言模型能力强烈超越人类的任务中，我们的方法能显著更好地激发这些能力，优于使用人类标签的训练。最后，我们展示了我们的方法可以提升前沿语言模型的训练：我们使用我们的方法训练了一个无监督奖励模型，并使用强化学习训练了一个基于Haiku的Claude 3.5助手。这两种模型都优于其人类监督的对应版本。 

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
# Tina: Tiny Reasoning Models via LoRA 

**Title (ZH)**: Tina: Tiny Reasoning Models via LoRA 

**Authors**: Shangshang Wang, Julian Asilis, Ömer Faruk Akgül, Enes Burak Bilgin, Ollie Liu, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2504.15777)  

**Abstract**: How cost-effectively can strong reasoning abilities be achieved in language models? Driven by this fundamental question, we present Tina, a family of tiny reasoning models achieved with high cost-efficiency. Notably, Tina demonstrates that substantial reasoning performance can be developed using only minimal resources, by applying parameter-efficient updates during reinforcement learning (RL), using low-rank adaptation (LoRA), to an already tiny 1.5B parameter base model. This minimalist approach produces models that achieve reasoning performance which is competitive with, and sometimes surpasses, SOTA RL reasoning models built upon the same base model. Crucially, this is achieved at a tiny fraction of the computational post-training cost employed by existing SOTA models. In fact, the best Tina model achieves a >20\% reasoning performance increase and 43.33\% Pass@1 accuracy on AIME24, at only \$9 USD post-training and evaluation cost (i.e., an estimated 260x cost reduction). Our work reveals the surprising effectiveness of efficient RL reasoning via LoRA. We validate this across multiple open-source reasoning datasets and various ablation settings starting with a single, fixed set of hyperparameters. Furthermore, we hypothesize that this effectiveness and efficiency stem from LoRA rapidly adapting the model to the structural format of reasoning rewarded by RL, while largely preserving the base model's underlying knowledge. In service of accessibility and open research, we fully open-source all code, training logs, and model weights \& checkpoints. 

**Abstract (ZH)**: 如何以最经济的方式在语言模型中实现强大的推理能力？我们提出了Tina，这是一种通过高成本效率实现的小小推理模型 familia。Tina 显示出，仅通过在强化学习（RL）过程中使用参数高效更新和低秩适应（LoRA）对一个已有的超小型基础模型（1.5B参数）进行少量资源的训练，即可实现显著的推理性能提升。这种 minimalist 方法生成的模型在推理性能上与同基础模型构建的当前最佳强化学习推理模型相比，有时甚至更为出色。更重要的是，这种方法仅使用现有最佳模型所需计算后训练成本的一小部分即可实现。事实上，Tina 最优模型在 AIME24 上实现了超过 20% 的推理性能提升和 43.33% 的 Pass@1 准确率，仅需 9 美元的后训练和评估成本（估计成本降低了约 260 倍）。我们的工作揭示了通过 LoRA 实现高效 RL 推理的惊人效果。我们通过多种开源推理数据集和不同的 ablation 设置进行了验证，使用固定的超参数集进行了初始验证。此外，我们认为这种效果和效率来自于 LoRA 快速使模型适应 RL 奖励的推理结构，并且主要保留了基础模型的内在知识。为了促进可访问性和开放研究，我们完全开源了所有代码、训练日志、模型权重和检查点。 

---
