# Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance 

**Title (ZH)**: 使用LLM代理清理维护日志以改进预测性维护 

**Authors**: Valeriu Dimidov, Faisal Hawlader, Sasan Jafarnejad, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2511.05311)  

**Abstract**: Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities. 

**Abstract (ZH)**: 经济约束、可用数据集有限以及专用专家短缺长期被视为阻碍汽车领域预测性维护（PdM）采纳和发展的关键挑战。近年来，大型语言模型（LLMs）的进步为克服这些障碍并加速PdM从研究向工业实践的过渡提供了机遇。在这些条件下，我们探讨了基于LLM的代理在支持PdM清洗管道方面的潜力。具体而言，我们专注于维护日志，这是训练高性能机器学习（ML）模型的关键数据源，但这些日志常常受到拼写错误、缺失字段、近似重复条目和错误日期等错误的影响。我们评估了LLM代理在涉及六种不同类型的噪声的清洗任务中的表现。研究结果表明，LLM在处理通用清洗任务方面具有有效性，并为未来的工业应用奠定了有前途的基础。尽管领域特定的错误仍然具有挑战性，但这些结果突显了通过专门训练和增强代理能力以进一步改进的潜力。 

---
# ORCHID: Orchestrated Retrieval-Augmented Classification with Human-in-the-Loop Intelligent Decision-Making for High-Risk Property 

**Title (ZH)**: ORCHID: 组合检索增强分类与人类在环智能决策机制以应对高风险财产评估 

**Authors**: Maria Mahbub, Vanessa Lama, Sanjay Das, Brian Starks, Christopher Polchek, Saffell Silvers, Lauren Deck, Prasanna Balaprakash, Tirthankar Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2511.04956)  

**Abstract**: High-Risk Property (HRP) classification is critical at U.S. Department of Energy (DOE) sites, where inventories include sensitive and often dual-use equipment. Compliance must track evolving rules designated by various export control policies to make transparent and auditable decisions. Traditional expert-only workflows are time-consuming, backlog-prone, and struggle to keep pace with shifting regulatory boundaries. We demo ORCHID, a modular agentic system for HRP classification that pairs retrieval-augmented generation (RAG) with human oversight to produce policy-based outputs that can be audited. Small cooperating agents, retrieval, description refiner, classifier, validator, and feedback logger, coordinate via agent-to-agent messaging and invoke tools through the Model Context Protocol (MCP) for model-agnostic on-premise operation. The interface follows an Item to Evidence to Decision loop with step-by-step reasoning, on-policy citations, and append-only audit bundles (run-cards, prompts, evidence). In preliminary tests on real HRP cases, ORCHID improves accuracy and traceability over a non-agentic baseline while deferring uncertain items to Subject Matter Experts (SMEs). The demonstration shows single item submission, grounded citations, SME feedback capture, and exportable audit artifacts, illustrating a practical path to trustworthy LLM assistance in sensitive DOE compliance workflows. 

**Abstract (ZH)**: HRP分类对于美国能源部（DOE）站点至关重要，其中库存包括敏感且 often 双重用途 的设备。合规性必须遵循各种出口控制政策指定的 evolving 规则，以实现透明和可审计的决策。传统由专家独自操作的工作流程耗时、容易积压，并且难以跟上不断变化的监管界限。我们演示了ORCHID，这是一种模块化的代理系统，结合了检索增强生成（RAG）和人的监督，以生成可审核的基于政策的输出。小规模合作代理，检索、描述润色者、分类器、验证器和反馈记录器，通过代理间消息通信协调，并通过模型上下文协议（MCP）调用工具以实现模型无关的本地操作。界面遵循项目到证据到决策的循环，包括逐步推理、基于政策的引用和追加审计捆绑（运行卡、提示、证据）。初步测试结果显示，ORCHID在真实HRP案例中提高了准确性和可追溯性，并将不确定的项目提交给专家。演示展示了单个项目提交、基于参考的引文、专家反馈捕获以及可导出的审计产物，展示了在敏感DOE合规流程中实现可信的LLM辅助的实际路径。 

---
# SWE-Compass: Towards Unified Evaluation of Agentic Coding Abilities for Large Language Models 

**Title (ZH)**: SWE-Compass: 向统一评估大型语言模型自主编码能力的方向努力 

**Authors**: Jingxuan Xu, Ken Deng, Weihao Li, Songwei Yu, Huaixi Tang, Haoyang Huang, Zhiyi Lai, Zizheng Zhan, Yanan Wu, Chenchen Zhang, Kepeng Lei, Yifan Yao, Xinping Lei, Wenqiang Zhu, Zongxian Feng, Han Li, Junqi Xiong, Dailin Li, Zuchen Gao, Kun Wu, Wen Xiang, Ziqi Zhan, Yuanxing Zhang, Wuxuan Gong, Ziyuan Gao, Guanxiang Wang, Yirong Xue, Xiaojiang Zhang, Jinghui Wang, Huiming Wang, Wenhao Zhuang, Zhaoxiang Zhang, Yuqun Zhang, Haotian Zhang, Bin Chen, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05459)  

**Abstract**: Evaluating large language models (LLMs) for software engineering has been limited by narrow task coverage, language bias, and insufficient alignment with real-world developer workflows. Existing benchmarks often focus on algorithmic problems or Python-centric bug fixing, leaving critical dimensions of software engineering underexplored. To address these gaps, we introduce SWE-Compass1, a comprehensive benchmark that unifies heterogeneous code-related evaluations into a structured and production-aligned framework. SWE-Compass spans 8 task types, 8 programming scenarios, and 10 programming languages, with 2000 high-quality instances curated from authentic GitHub pull requests and refined through systematic filtering and validation. We benchmark ten state-of-the-art LLMs under two agentic frameworks, SWE-Agent and Claude Code, revealing a clear hierarchy of difficulty across task types, languages, and scenarios. Moreover, by aligning evaluation with real-world developer practices, SWE-Compass provides a rigorous and reproducible foundation for diagnosing and advancing agentic coding capabilities in large language models. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）在软件工程中的应用受限于任务覆盖范围狭窄、语言偏差以及与实际开发者工作流的不充分对齐。现有基准测试往往集中于算法问题或以Python为主的 bug 修复，导致软件工程的关键维度被忽略。为解决这些问题，我们引入了 SWE-Compass1，这是一种全面的基准测试，将异构代码相关评估统一到一个结构化且与生产环境对齐的框架中。SWE-Compass覆盖了8种任务类型、8种编程场景和10种编程语言，包括2000个高质量实例，这些实例来源于真实的GitHub拉取请求，并经过系统筛选和验证。我们使用SWE-Agent和Claude Code两种代理框架对10种最先进的LLM进行了基准测试，揭示了任务类型、语言和场景的清晰难度等级。此外，通过将评估与实际开发者实践对齐，SWE-Compass为诊断和提升大型语言模型的代理编码能力提供了严谨且可重复的基础。 

---
# TeaRAG: A Token-Efficient Agentic Retrieval-Augmented Generation Framework 

**Title (ZH)**: TeaRAG: 一种高效的代理检索增强生成框架 

**Authors**: Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen, Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05385)  

**Abstract**: Retrieval-Augmented Generation (RAG) utilizes external knowledge to augment Large Language Models' (LLMs) reliability. For flexibility, agentic RAG employs autonomous, multi-round retrieval and reasoning to resolve queries. Although recent agentic RAG has improved via reinforcement learning, they often incur substantial token overhead from search and reasoning processes. This trade-off prioritizes accuracy over efficiency. To address this issue, this work proposes TeaRAG, a token-efficient agentic RAG framework capable of compressing both retrieval content and reasoning steps. 1) First, the retrieved content is compressed by augmenting chunk-based semantic retrieval with a graph retrieval using concise triplets. A knowledge association graph is then built from semantic similarity and co-occurrence. Finally, Personalized PageRank is leveraged to highlight key knowledge within this graph, reducing the number of tokens per retrieval. 2) Besides, to reduce reasoning steps, Iterative Process-aware Direct Preference Optimization (IP-DPO) is proposed. Specifically, our reward function evaluates the knowledge sufficiency by a knowledge matching mechanism, while penalizing excessive reasoning steps. This design can produce high-quality preference-pair datasets, supporting iterative DPO to improve reasoning conciseness. Across six datasets, TeaRAG improves the average Exact Match by 4% and 2% while reducing output tokens by 61% and 59% on Llama3-8B-Instruct and Qwen2.5-14B-Instruct, respectively. Code is available at this https URL. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)利用外部知识增强大型语言模型的可靠性。为了提高灵活性，代理RAG采用自主多轮检索和推理来解决问题。尽管最近的代理RAG通过强化学习有所改进，但在检索和推理过程中通常会产生大量的token开销，这种权衡优先考虑准确性而牺牲了效率。为了解决这一问题，本工作提出了TeaRAG，这是一种token高效的代理RAG框架，能够同时压缩检索内容和推理步骤。1) 首先，通过将基于块的语义检索与简洁三元组的图检索相结合来压缩检索内容。然后，根据语义相似性和共现性构建知识关联图。最后，利用个性化PageRank突出显示图中的关键知识，从而减少每次检索的token数量。2) 此外，为了减少推理步骤，提出了迭代过程感知直接偏好优化（IP-DPO）。具体来说，我们的奖励函数通过知识匹配机制评估知识充分性，并对过多的推理步骤进行惩罚。这种设计能够生成高质量的偏好对数据集，支持迭代DPO提高推理的简洁性。在六个数据集上，TeaRAG分别将Llama3-8B-Instruct和Qwen2.5-14B-Instruct的精确匹配均值提高了4%和2%，同时分别减少了61%和59%的输出token。代码可通过此网站获得。 

---
# What Are the Facts? Automated Extraction of Court-Established Facts from Criminal-Court Opinions 

**Title (ZH)**: 法院确立的事实是什么？自动提取刑事法院判决中确立的事实 

**Authors**: Klára Bendová, Tomáš Knap, Jan Černý, Vojtěch Pour, Jaromir Savelka, Ivana Kvapilíková, Jakub Drápal  

**Link**: [PDF](https://arxiv.org/pdf/2511.05320)  

**Abstract**: Criminal justice administrative data contain only a limited amount of information about the committed offense. However, there is an unused source of extensive information in continental European courts' decisions: descriptions of criminal behaviors in verdicts by which offenders are found guilty. In this paper, we study the feasibility of extracting these descriptions from publicly available court decisions from Slovakia. We use two different approaches for retrieval: regular expressions and large language models (LLMs). Our baseline was a simple method employing regular expressions to identify typical words occurring before and after the description. The advanced regular expression approach further focused on "sparing" and its normalization (insertion of spaces between individual letters), typical for delineating the description. The LLM approach involved prompting the Gemini Flash 2.0 model to extract the descriptions using predefined instructions. Although the baseline identified descriptions in only 40.5% of verdicts, both methods significantly outperformed it, achieving 97% with advanced regular expressions and 98.75% with LLMs, and 99.5% when combined. Evaluation by law students showed that both advanced methods matched human annotations in about 90% of cases, compared to just 34.5% for the baseline. LLMs fully matched human-labeled descriptions in 91.75% of instances, and a combination of advanced regular expressions with LLMs reached 92%. 

**Abstract (ZH)**: 大陆欧洲法院判决中关于犯罪行为描述的提取：基于正则表达式和大型语言模型的方法研究 

---
# TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems 

**Title (ZH)**: TAMAS：多智能体大型语言模型系统的对抗风险benchmark研究 

**Authors**: Ishan Kavathekar, Hemang Jain, Ameya Rathod, Ponnurangam Kumaraguru, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05269)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems. 

**Abstract (ZH)**: 威胁和多agents系统中的攻击：多agents语言模型系统（TAMAS） 

---
# Generating Software Architecture Description from Source Code using Reverse Engineering and Large Language Model 

**Title (ZH)**: 使用逆向工程和大型语言模型从源代码生成软件架构描述 

**Authors**: Ahmad Hatahet, Christoph Knieke, Andreas Rausch  

**Link**: [PDF](https://arxiv.org/pdf/2511.05165)  

**Abstract**: Software Architecture Descriptions (SADs) are essential for managing the inherent complexity of modern software systems. They enable high-level architectural reasoning, guide design decisions, and facilitate effective communication among diverse stakeholders. However, in practice, SADs are often missing, outdated, or poorly aligned with the system's actual implementation. Consequently, developers are compelled to derive architectural insights directly from source code-a time-intensive process that increases cognitive load, slows new developer onboarding, and contributes to the gradual degradation of clarity over the system's lifetime. To address these issues, we propose a semi-automated generation of SADs from source code by integrating reverse engineering (RE) techniques with a Large Language Model (LLM). Our approach recovers both static and behavioral architectural views by extracting a comprehensive component diagram, filtering architecturally significant elements (core components) via prompt engineering, and generating state machine diagrams to model component behavior based on underlying code logic with few-shots prompting. This resulting views representation offer a scalable and maintainable alternative to traditional manual architectural documentation. This methodology, demonstrated using C++ examples, highlights the potent capability of LLMs to: 1) abstract the component diagram, thereby reducing the reliance on human expert involvement, and 2) accurately represent complex software behaviors, especially when enriched with domain-specific knowledge through few-shot prompting. These findings suggest a viable path toward significantly reducing manual effort while enhancing system understanding and long-term maintainability. 

**Abstract (ZH)**: 软件架构描述（SADs）对于管理现代软件系统的固有复杂性至关重要。它们 enabling 高级架构推理，指导设计决策，并促进多方利益相关者之间的有效沟通。然而，在实践中，SADs 往往缺失、过时或与系统的实际实现 poorly 对齐。因此，开发人员不得不直接从源代码中推导出架构见解——这一耗时的过程增加了认知负担，减慢了新开发者的入职速度，并导致系统在其生命周期中逐渐失去清晰度。为了解决这些问题，我们提出了一种半自动化的SADs 生成方法，通过将反向工程（RE）技术与大规模语言模型（LLM）集成来从源代码中生成SADs。该方法通过提取全面的组件图、利用提示工程筛选出架构上重要的元素（核心组件）以及基于底层代码逻辑生成状态机图来建模组件行为，从而恢复静态和行为架构视图。这些生成的视图表示为传统手动架构文档提供了可扩展且可维护的替代方案。该方法通过C++示例展示，突显了LLM的强大能力：1) 抽象组件图，从而减少对人工专家的依赖，2) 准确表示复杂软件行为，特别是在通过少样本提示增强领域特定知识时。这些发现表明了一条减轻手动努力、增强系统理解和长期可维护性的可行途径。 

---
# UA-Code-Bench: A Competitive Programming Benchmark for Evaluating LLM Code Generation in Ukrainian 

**Title (ZH)**: UA-Code-Bench: 用于评估乌克兰语编程代码生成的LLM基准 

**Authors**: Mykyta Syromiatnikov, Victoria Ruvinskaya  

**Link**: [PDF](https://arxiv.org/pdf/2511.05040)  

**Abstract**: Evaluating the real capabilities of large language models in low-resource languages still represents a challenge, as many existing benchmarks focus on widespread tasks translated from English or evaluate only simple language understanding. This paper introduces UA-Code-Bench, a new open-source benchmark established for a thorough evaluation of language models' code generation and competitive programming problem-solving abilities in Ukrainian. The benchmark comprises 500 problems from the Eolymp platform, evenly distributed across five complexity levels from very easy to very hard. A diverse set of 13 leading proprietary and open-source models, generating Python solutions based on a one-shot prompt, was evaluated via the dedicated Eolymp environment against hidden tests, ensuring code correctness. The obtained results reveal that even top-performing models, such as OpenAI o3 and GPT-5, solve only half of the problems, highlighting the challenge of code generation in low-resource natural language. Furthermore, this research presents a comprehensive analysis of performance across various difficulty levels, as well as an assessment of solution uniqueness and computational efficiency, measured by both elapsed time and memory consumption of the generated solutions. In conclusion, this work demonstrates the value of competitive programming benchmarks in evaluating large language models, especially in underrepresented languages. It also paves the way for future research on multilingual code generation and reasoning-enhanced models. The benchmark, data parsing, preparation, code generation, and evaluation scripts are available at this https URL. 

**Abstract (ZH)**: 评估大型语言模型在低资源语言中的实际能力仍然是一项挑战，因为许多现有基准聚焦于从英语翻译过来的广泛任务或仅评估简单的语言理解能力。本文介绍了UA-Code-Bench，一个新的开源基准，旨在全面评估语言模型在乌克兰语中的代码生成和竞赛编程问题解决能力。该基准包括500个来自Eolymp平台的问题，按复杂度均匀分布，从非常简单到非常困难。通过专用的Eolymp环境，在隐藏测试条件下评估了13种不同的领先专有和开源模型，确保代码正确性。获得的结果表明，即使是性能最佳的模型，如OpenAI o3和GPT-5，也只能解决一半的问题，突显了在低资源自然语言中的代码生成挑战。此外，本文还对不同难度级别的性能进行了全面分析，评估了解决方案的唯一性和计算效率，通过生成解决方案所花费的时间和内存消耗进行衡量。总之，这项工作证明了竞赛编程基准在评估大型语言模型方面的价值，特别是在代表性不足的语言中。此外，它也为多语言代码生成和增强推理模型的未来研究开辟了道路。基准、数据解析、准备、代码生成和评估脚本可在以下链接获取。 

---
# Pluralistic Behavior Suite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies 

**Title (ZH)**: 多元行为套件：多重轮次压力测试自定义行为政策的合规性 

**Authors**: Prasoon Varshney, Makesh Narsimhan Sreedhar, Liwei Jiang, Traian Rebedea, Christopher Parisien  

**Link**: [PDF](https://arxiv.org/pdf/2511.05018)  

**Abstract**: Large language models (LLMs) are typically aligned to a universal set of safety and usage principles intended for broad public acceptability. Yet, real-world applications of LLMs often take place within organizational ecosystems shaped by distinctive corporate policies, regulatory requirements, use cases, brand guidelines, and ethical commitments. This reality highlights the need for rigorous and comprehensive evaluation of LLMs with pluralistic alignment goals, an alignment paradigm that emphasizes adaptability to diverse user values and needs. In this work, we present PLURALISTIC BEHAVIOR SUITE (PBSUITE), a dynamic evaluation suite designed to systematically assess LLMs' capacity to adhere to pluralistic alignment specifications in multi-turn, interactive conversations. PBSUITE consists of (1) a diverse dataset of 300 realistic LLM behavioral policies, grounded in 30 industries; and (2) a dynamic evaluation framework for stress-testing model compliance with custom behavioral specifications under adversarial conditions. Using PBSUITE, We find that leading open- and closed-source LLMs maintain robust adherence to behavioral policies in single-turn settings (less than 4% failure rates), but their compliance weakens substantially in multi-turn adversarial interactions (up to 84% failure rates). These findings highlight that existing model alignment and safety moderation methods fall short in coherently enforcing pluralistic behavioral policies in real-world LLM interactions. Our work contributes both the dataset and analytical framework to support future research toward robust and context-aware pluralistic alignment techniques. 

**Abstract (ZH)**: PLURALISTIC BEHAVIOR SUITE: A DYNAMIC EVALUATION FRAMEWORK FOR SYSTEMATIC ASSESSMENT OF LLMs' ADHERENCE TO PLURALISTIC ALIGNMENT SPECIFICATIONS 

---
# Query Generation Pipeline with Enhanced Answerability Assessment for Financial Information Retrieval 

**Title (ZH)**: 带有增强答案可靠性评估的金融信息检索查询生成管道 

**Authors**: Hyunkyu Kim, Yeeun Yoo, Youngjun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2511.05000)  

**Abstract**: As financial applications of large language models (LLMs) gain attention, accurate Information Retrieval (IR) remains crucial for reliable AI services. However, existing benchmarks fail to capture the complex and domain-specific information needs of real-world banking scenarios. Building domain-specific IR benchmarks is costly and constrained by legal restrictions on using real customer data. To address these challenges, we propose a systematic methodology for constructing domain-specific IR benchmarks through LLM-based query generation. As a concrete implementation of this methodology, our pipeline combines single and multi-document query generation with an enhanced and reasoning-augmented answerability assessment method, achieving stronger alignment with human judgments than prior approaches. Using this methodology, we construct KoBankIR, comprising 815 queries derived from 204 official banking documents. Our experiments show that existing retrieval models struggle with the complex multi-document queries in KoBankIR, demonstrating the value of our systematic approach for domain-specific benchmark construction and underscoring the need for improved retrieval techniques in financial domains. 

**Abstract (ZH)**: 随着大规模语言模型在金融领域的应用引起关注，准确的信息检索（IR）对于可靠的AI服务仍然至关重要。然而，现有的基准测试未能捕捉到真实银行业场景中复杂和领域特定的信息需求。构建领域特定的IR基准测试成本高昂，并受限于使用真实客户数据的法律限制。为应对这些挑战，我们提出了一种通过基于大语言模型的查询生成来系统地构建领域特定IR基准测试的方法。作为该方法的具体实现，我们的流水线结合了单文档和多文档查询生成，并采用增强和推理增强的答案可评估性评估方法，比先前的方法更接近人类判断。通过这种方法，我们构建了KoBankIR，包含来自204份官方银行业文件的815个查询。我们的实验表明，现有的检索模型难以处理KoBankIR中的复杂多文档查询，这强调了在金融领域构建领域特定基准测试和改进检索技术的重要性。 

---
# Too Good to be Bad: On the Failure of LLMs to Role-Play Villains 

**Title (ZH)**: Too Good to be Bad: On the Failure of LLMs to Role-Play Villains 

**Authors**: Zihao Yi, Qingxuan Jiang, Ruotian Ma, Xingyu Chen, Qu Yang, Mengru Wang, Fanghua Ye, Ying Shen, Zhaopeng Tu, Xiaolong Li, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04962)  

**Abstract**: Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在创造性生成中的应用日益增多，包括模拟虚构角色。然而，它们表现非亲社会、敌对人格的能力仍鲜有研究。我们推测，现代LLMs的安全对齐在其诚实地扮演道德模糊或反派角色的任务上造成了根本矛盾。为了研究这一现象，我们引入了道德角色扮演基准（Moral RolePlay benchmark），该基准包括一个四级道德对齐尺度和一个平衡的测试集，以进行严格的评估。我们要求当前最先进的LLMs扮演从道德典范到纯粹反派的各种角色。大规模评估表明，随着角色道德性的降低，角色扮演的准确性呈一致且单调地下降。我们发现，模型在与安全原则直接对立的特质上最难处理，如“诡诈”和“ manipulative”，经常用表面的侵略性替代复杂的恶意。此外，我们证明了通用聊天机器人技能并不能很好地预测反派角色扮演能力，而高度安全对齐的模型在这方面表现尤其差。我们的工作提供了这一关键限制的首个系统性证据，高亮了模型安全与创造性准确之间的重要张力。我们的基准和发现为进一步发展更为细腻、情境感知的对齐方法铺平了道路。 

---
# BudgetMem: Learning Selective Memory Policies for Cost-Efficient Long-Context Processing in Language Models 

**Title (ZH)**: BudgetMem：学习成本效益的选择性内存策略以实现语言模型中高效长上下文处理 

**Authors**: Chandra Vamsi Krishna Alla, Harish Naidu Gaddam, Manohar Kommi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04919)  

**Abstract**: Large Language Models (LLMs) face significant computational and memory constraints when processing long contexts, despite growing demand for applications requiring reasoning over extensive documents, multi-session dialogues, and book length texts. While recent advances have extended context windows to 100K-1M tokens, such approaches incur prohibitive costs for resource constrained deployments. We propose BudgetMem, a novel memory augmented architecture that learns what to remember rather than remembering everything. Our system combines selective memory policies with feature based salience scoring (entity density, TF-IDF, discourse markers, position bias) to decide which information merits storage under strict budget constraints. Unlike existing retrieval augmented generation (RAG) systems that store all chunks, BudgetMem employs learned gating mechanisms coupled with BM25 sparse retrieval for efficient information access. Through comprehensive experiments on 700 question answer pairs across short (237 tokens) and long (5K-10K tokens) documents with Llama-3.2-3B-Instruct, we demonstrate that BudgetMem achieves remarkable results on long documents: only 1.0% F1 score degradation while saving 72.4% memory compared to baseline RAG. We validate our approach through budget sensitivity analysis (testing 7 budget ratios), naive baseline comparisons, and document length analysis, showing that BudgetMem's benefits increase with document length. Our work provides a practical pathway for deploying capable long context systems on modest hardware, democratizing access to advanced language understanding capabilities. 

**Abstract (ZH)**: BudgetMem：在严格内存预算下处理长上下文的新型记忆增强架构 

---
# Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs 

**Title (ZH)**: LLMs中行为自意识的最小且机制化条件 

**Authors**: Matthew Bozoukov, Matthew Nguyen, Shubkarman Singh, Bart Bussmann, Patrick Leask  

**Link**: [PDF](https://arxiv.org/pdf/2511.04875)  

**Abstract**: Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated. 

**Abstract (ZH)**: 近期的研究表明，大语言模型可以表现出行为自意识：即能够准确描述或预测其自身学习行为的能力，而无需明确的监督。这一能力引发了安全方面的担忧，因为这可能会使模型在评估中更好地隐藏其真正的能力。我们尝试 characterizing 使这种自意识出现的最小条件，以及其表现的机制性过程。通过在使用低秩适配器（LoRA）调整指令的大语言模型上进行受控的微调实验，我们发现：(1) 使用单一的秩1 LoRA 适配器可以可靠地诱导出自意识；(2) 学习到的自意识行为可以通过激活空间中的单一引导向量来很大程度上捕获，几乎恢复了所有微调行为效应；(3) 自意识不是普遍存在的，而是领域局部化的，具有跨任务的独立表示。这些发现共同表明，行为自意识作为一种领域特定的线性特征，可以容易地被诱导和调节。 

---
# Software Defined Vehicle Code Generation: A Few-Shot Prompting Approach 

**Title (ZH)**: 软件定义车辆代码生成：少量示例提示方法 

**Authors**: Quang-Dung Nguyen, Tri-Dung Tran, Thanh-Hieu Chu, Hoang-Loc Tran, Xiangwei Cheng, Dirk Slama  

**Link**: [PDF](https://arxiv.org/pdf/2511.04849)  

**Abstract**: The emergence of Software-Defined Vehicles (SDVs) marks a paradigm shift in the automotive industry, where software now plays a pivotal role in defining vehicle functionality, enabling rapid innovation of modern vehicles. Developing SDV-specific applications demands advanced tools to streamline code generation and improve development efficiency. In recent years, general-purpose large language models (LLMs) have demonstrated transformative potential across domains. Still, restricted access to proprietary model architectures hinders their adaption to specific tasks like SDV code generation. In this study, we propose using prompts, a common and basic strategy to interact with LLMs and redirect their responses. Using only system prompts with an appropriate and efficient prompt structure designed using advanced prompt engineering techniques, LLMs can be crafted without requiring a training session or access to their base design. This research investigates the extensive experiments on different models by applying various prompting techniques, including bare models, using a benchmark specifically created to evaluate LLMs' performance in generating SDV code. The results reveal that the model with a few-shot prompting strategy outperforms the others in adjusting the LLM answers to match the expected outcomes based on quantitative metrics. 

**Abstract (ZH)**: 软件定义车辆（SDVs）的出现标志着汽车行业的 paradigm shift，软件现在在定义车辆功能方面发挥着关键作用，推动了现代车辆的快速创新。开发针对SDV的应用程序需要先进的工具来简化代码生成并提高开发效率。近年来，通用的大语言模型（LLMs）在各个领域展现了变革潜力，但由于受限于专有模型架构的访问权限，它们难以适应如SDV代码生成这样的特定任务。本研究提出使用提示，这是一种常见且基本的策略，用于与LLMs交互并引导其响应。通过使用系统提示，并设计出一种高效的提示结构，利用高级提示工程技术，可以在不需进行训练或访问其基础设计的情况下，构建出LLMs。本研究通过应用各种提示技术进行广泛的实验，包括裸模型和使用专门为评估LLMs生成SDV代码性能而创建的基准，调查了不同模型的表现。结果显示，使用少量示例提示策略的模型在基于定量指标调整LLMs响应以匹配预期结果方面优于其他模型。 

---
# Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models 

**Title (ZH)**: 基于提示的安全指导对未学习的文本到图像扩散模型无效 

**Authors**: Jiwoo Shin, Byeonghu Na, Mina Kang, Wonhyeok Choi, Il-chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2511.04834)  

**Abstract**: Recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts. 

**Abstract (ZH)**: recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts. 

---
# PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference 

**Title (ZH)**: PuzzleMoE: 通过稀疏专家合并和位图推理高效压缩大型Mixture-of-Experts模型 

**Authors**: Yushu Zhao, Zheng Wang, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04805)  

**Abstract**: Mixture-of-Experts (MoE) models have shown strong potential in scaling language models efficiently by activating only a small subset of experts per input. However, their widespread deployment remains limited due to the high memory overhead associated with storing all expert parameters, particularly as the number of experts increases. To address this challenge, prior works have explored expert dropping and merging strategies, yet they often suffer from performance drop at high compression ratios. In this paper, we introduce PuzzleMoE, a training-free MoE compression method that achieves both high accuracy and efficient inference through two key innovations: First, PuzzleMoE performs sparse expert merging by identifying element-wise weight redundancy and specialization. It uses a dual-mask to capture both shared and expert-specific parameters. Second, to avoid the overhead of storing binary masks and signs, PuzzleMoE introduces a bit-packed encoding scheme that reuses underutilized exponent bits, enabling efficient MoE inference on GPUs. Extensive experiments demonstrate that PuzzleMoE can compress MoE models by up to 50% while maintaining accuracy across various tasks. Specifically, it outperforms prior MoE compression methods by up to 16.7% on MMLU at 50% compression ratio, and achieves up to 1.28\times inference speedup. 

**Abstract (ZH)**: PuzzleMoE：一种无需训练的MoE压缩方法及其高效推理 

---
# Trustworthiness Calibration Framework for Phishing Email Detection Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行钓鱼邮件检测的可信度校准框架 

**Authors**: Daniyal Ganiuly, Assel Smaiyl  

**Link**: [PDF](https://arxiv.org/pdf/2511.04728)  

**Abstract**: Phishing emails continue to pose a persistent challenge to online communication, exploiting human trust and evading automated filters through realistic language and adaptive tactics. While large language models (LLMs) such as GPT-4 and LLaMA-3-8B achieve strong accuracy in text classification, their deployment in security systems requires assessing reliability beyond benchmark performance. To address this, this study introduces the Trustworthiness Calibration Framework (TCF), a reproducible methodology for evaluating phishing detectors across three dimensions: calibration, consistency, and robustness. These components are integrated into a bounded index, the Trustworthiness Calibration Index (TCI), and complemented by the Cross-Dataset Stability (CDS) metric that quantifies stability of trustworthiness across datasets. Experiments conducted on five corpora, such as SecureMail 2025, Phishing Validation 2024, CSDMC2010, Enron-Spam, and Nazario, using DeBERTa-v3-base, LLaMA-3-8B, and GPT-4 demonstrate that GPT-4 achieves the strongest overall trust profile, followed by LLaMA-3-8B and DeBERTa-v3-base. Statistical analysis confirms that reliability varies independently of raw accuracy, underscoring the importance of trust-aware evaluation for real-world deployment. The proposed framework establishes a transparent and reproducible foundation for assessing model dependability in LLM-based phishing detection. 

**Abstract (ZH)**: 持续存在的钓鱼邮件挑战：通过现实语言和适应性策略利用人类信任并逃避自动过滤器，大型语言模型在文本分类中表现出色，但在安全系统中的应用需要超越基准性能评估其可靠性。为此，本研究引入了可信度校准框架（TCF），这是一种用于从三个维度评估钓鱼检测器可靠性的可重复方法：校准、一致性和稳健性。这些组件被整合到一个界标指数中，即可信度校准指数（TCI），并辅以跨数据集稳定性（CDS）指标，该指标量化了不同数据集中的可信度稳定性。使用DeBERTa-v3-base、LLaMA-3-8B和GPT-4在SecureMail 2025、Phishing Validation 2024、CSDMC2010、Enron-Spam和Nazario等五个数据集中进行的实验证明，GPT-4在整体可信度方面表现最佳，其次是LLaMA-3-8B和DeBERTa-v3-base。统计分析表明，可靠性与原始准确性无关，强调了可信度感知评估在实际部署中的重要性。所提出的框架为基于大型语言模型的钓鱼检测模型可靠性评估提供了透明和可重复的基础。 

---
# First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation 

**Title (ZH)**: 首个层次并不一定优于最终层次：语言模型数据影响评估中的层选择与聚合策略评价 

**Authors**: Dmytro Vitel, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2511.04715)  

**Abstract**: Identifying how training samples influence/impact Large Language Model (LLM) decision-making is essential for effectively interpreting model decisions and auditing large-scale datasets. Current training sample influence estimation methods (also known as influence functions) undertake this goal by utilizing information flow through the model via its first-order and higher-order gradient terms. However, owing to the large model sizes of today consisting of billions of parameters, these influence computations are often restricted to some subset of model layers to ensure computational feasibility. Prior seminal work by Yeh et al. (2022) in assessing which layers are best suited for computing language data influence concluded that the first (embedding) layers are the most informative for this purpose, using a hypothesis based on influence scores canceling out (i.e., the cancellation effect). In this work, we propose theoretical and empirical evidence demonstrating how the cancellation effect is unreliable, and that middle attention layers are better estimators for influence. Furthermore, we address the broader challenge of aggregating influence scores across layers, and showcase how alternatives to standard averaging (such as ranking and vote-based methods) can lead to significantly improved performance. Finally, we propose better methods for evaluating influence score efficacy in LLMs without undertaking model retraining, and propose a new metric known as the Noise Detection Rate (NDR) that exhibits strong predictive capability compared to the cancellation effect. Through extensive experiments across LLMs of varying types and scales, we concretely determine that the first (layers) are not necessarily better than the last (layers) for LLM influence estimation, contrasting with prior knowledge in the field. 

**Abstract (ZH)**: 识别训练样本如何影响大型语言模型（LLM）的决策对于有效解释模型决策和审计大规模数据集至关重要。当前的训练样本影响估计方法（也称为影响函数）通过利用模型中的梯度信息来实现这一目标，包括一阶和高阶梯度项。然而，由于今天大型模型包含数十亿个参数，这些影响计算通常仅限于模型的一些子层以确保计算可行性。Yeh等人的先前开创性工作（2022）评估了哪些层最适合计算语言数据影响得出结论认为嵌入层是这一目的最有信息量的层，基于影响分数相互抵消的假设（即抵消效应）。在本文中，我们提出了理论和实验证据，证明抵消效应不可靠，并且中间注意力层是更有效的影响力估计器。此外，我们解决了在不同层聚合影响力分数的更广泛挑战，并展示了标准平均之外的替代方法（如排名和投票方法）可以显著提高性能。最后，我们提出了在不重新训练模型的情况下评估影响力分数有效性的更好方法，并提出了一个名为噪声检测率（NDR）的新指标，该指标相对于抵消效应显示出更强的预测能力。通过在不同类型和规模的LLM上进行广泛实验，我们明确确定，对于LLM影响估计，第一层并不一定比最后一层更好，这与领域的先前知识相悖。 

---
# Jailbreaking in the Haystack 

**Title (ZH)**: haystack中的越狱 

**Authors**: Rishi Rajesh Shah, Chen Henry Wu, Shashwat Saxena, Ziqian Zhong, Alexander Robey, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04707)  

**Abstract**: Recent advances in long-context language models (LMs) have enabled million-token inputs, expanding their capabilities across complex tasks like computer-use agents. Yet, the safety implications of these extended contexts remain unclear. To bridge this gap, we introduce NINJA (short for Needle-in-haystack jailbreak attack), a method that jailbreaks aligned LMs by appending benign, model-generated content to harmful user goals. Critical to our method is the observation that the position of harmful goals play an important role in safety. Experiments on standard safety benchmark, HarmBench, show that NINJA significantly increases attack success rates across state-of-the-art open and proprietary models, including LLaMA, Qwen, Mistral, and Gemini. Unlike prior jailbreaking methods, our approach is low-resource, transferable, and less detectable. Moreover, we show that NINJA is compute-optimal -- under a fixed compute budget, increasing context length can outperform increasing the number of trials in best-of-N jailbreak. These findings reveal that even benign long contexts -- when crafted with careful goal positioning -- introduce fundamental vulnerabilities in modern LMs. 

**Abstract (ZH)**: 近期长上下文语言模型的进展使得输入可达百万token，扩展了其在复杂任务如计算机使用代理方面的能力。然而，这些扩展上下文的安全性影响尚不清晰。为弥补这一缺口，我们介绍了NINJA（Needle-in-haystack jailbreak攻击方法），该方法通过在有害用户目标后附加良性、模型生成的内容来劫持对齐的语言模型。我们方法的关键在于观察到有害目标的位置在安全性方面起着重要作用。在标准安全性基准HarmBench上的实验表明，NINJA显著提高了最先进的开源和专有模型（包括LLaMA、Qwen、Mistral和Gemini）的攻击成功率。与以往的劫持方法不同，我们的方法资源需求低、可迁移且难以被检测到。此外，我们证明了NINJA在固定计算预算下是计算最优的——即增加上下文长度可以比增加最佳-of-N劫持的试次数量更有效地提高攻击成功率。这些发现表明，即使是看似无害的长上下文，在精心设计目标位置的情况下，也会在现代语言模型中引入根本性的漏洞。 

---
# Prioritize Economy or Climate Action? Investigating ChatGPT Response Differences Based on Inferred Political Orientation 

**Title (ZH)**: 优先考虑经济效益还是气候变化行动？基于推理出的政治倾向探究ChatGPT的响应差异 

**Authors**: Pelin Karadal, Dilara Kekulluoglu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04706)  

**Abstract**: Large Language Models (LLMs) distinguish themselves by quickly delivering information and providing personalized responses through natural language prompts. However, they also infer user demographics, which can raise ethical concerns about bias and implicit personalization and create an echo chamber effect. This study aims to explore how inferred political views impact the responses of ChatGPT globally, regardless of the chat session. We also investigate how custom instruction and memory features alter responses in ChatGPT, considering the influence of political orientation. We developed three personas (two politically oriented and one neutral), each with four statements reflecting their viewpoints on DEI programs, abortion, gun rights, and vaccination. We convey the personas' remarks to ChatGPT using memory and custom instructions, allowing it to infer their political perspectives without directly stating them. We then ask eight questions to reveal differences in worldview among the personas and conduct a qualitative analysis of the responses. Our findings indicate that responses are aligned with the inferred political views of the personas, showing varied reasoning and vocabulary, even when discussing similar topics. We also find the inference happening with explicit custom instructions and the implicit memory feature in similar ways. Analyzing response similarities reveals that the closest matches occur between the democratic persona with custom instruction and the neutral persona, supporting the observation that ChatGPT's outputs lean left. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过自然语言提示迅速提供信息并生成个性化响应，但也会推断用户 demographics，这可能会引发关于偏见和隐含个性化行为的伦理问题，并形成回音室效应。本研究旨在探索推断的政治观点如何影响全球范围内ChatGPT的回答，而不考虑聊天会话。我们还研究定制指令和记忆功能如何改变ChatGPT的回答，考虑政治倾向的影响。我们开发了三个角色（两个具有政治倾向和一个中立），每个角色都有四个陈述反映了他们对DEI项目、堕胎、枪权和疫苗的看法。我们使用记忆和定制指令将角色的言论传达给ChatGPT，使其能够推断出他们的政治观点，而不直接陈述。然后，我们提出八个问题以揭示角色之间世界观的差异，并对回答进行定性分析。我们的研究发现，回答与角色推断的政治观点保持一致，即使讨论相似话题时也表现出不同的推理和词汇。我们还发现，明确的定制指令和隐性记忆功能在推断方面有类似的表现。通过分析回答的相似性发现，民主倾向角色与定制指令下的回答最接近中立角色，这支持了ChatGPT输出偏向左翼的观察。 

---
# POLIS-Bench: Towards Multi-Dimensional Evaluation of LLMs for Bilingual Policy Tasks in Governmental Scenarios 

**Title (ZH)**: POLIS-Bench:面向政府场景中多语言政策任务的LLM多维度评价平台 

**Authors**: Tingyue Yang, Junchi Yao, Yuhui Guo, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04705)  

**Abstract**: We introduce POLIS-Bench, the first rigorous, systematic evaluation suite designed for LLMs operating in governmental bilingual policy scenarios. Compared to existing benchmarks, POLIS-Bench introduces three major advancements. (i) Up-to-date Bilingual Corpus: We construct an extensive, up-to-date policy corpus that significantly scales the effective assessment sample size, ensuring relevance to current governance practice. (ii) Scenario-Grounded Task Design: We distill three specialized, scenario-grounded tasks -- Clause Retrieval & Interpretation, Solution Generation, and the Compliance Judgmen--to comprehensively probe model understanding and application. (iii) Dual-Metric Evaluation Framework: We establish a novel dual-metric evaluation framework combining semantic similarity with accuracy rate to precisely measure both content alignment and task requirement adherence. A large-scale evaluation of over 10 state-of-the-art LLMs on POLIS-Bench reveals a clear performance hierarchy where reasoning models maintain superior cross-task stability and accuracy, highlighting the difficulty of compliance tasks. Furthermore, leveraging our benchmark, we successfully fine-tune a lightweight open-source model. The resulting POLIS series models achieves parity with, or surpasses, strong proprietary baselines on multiple policy subtasks at a significantly reduced cost, providing a cost-effective and compliant path for robust real-world governmental deployment. 

**Abstract (ZH)**: POLIS-Bench：面向政府双语政策场景的首个严谨系统性评估套件 

---
# Measuring what Matters: Construct Validity in Large Language Model Benchmarks 

**Title (ZH)**: 关注核心：大型语言模型基准中的结构效度 

**Authors**: Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia Hafner, Harry Mayne, Jan Batzner, Negar Foroutan, Chris Schmitz, Karolina Korgul, Hunar Batra, Oishi Deb, Emma Beharry, Cornelius Emde, Thomas Foster, Anna Gausen, María Grandury, Simeng Han, Valentin Hofmann, Lujain Ibrahim, Hazel Kim, Hannah Rose Kirk, Fangru Lin, Gabrielle Kaili-May Liu, Lennart Luettgau, Jabez Magomere, Jonathan Rystrøm, Anna Sotnikova, Yushi Yang, Yilun Zhao, Adel Bibi, Antoine Bosselut, Ronald Clark, Arman Cohan, Jakob Foerster, Yarin Gal, Scott A. Hale, Inioluwa Deborah Raji, Christopher Summerfield, Philip H.S. Torr, Cozmin Ududec, Luc Rocher, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04703)  

**Abstract**: Evaluating large language models (LLMs) is crucial for both assessing their capabilities and identifying safety or robustness issues prior to deployment. Reliably measuring abstract and complex phenomena such as 'safety' and 'robustness' requires strong construct validity, that is, having measures that represent what matters to the phenomenon. With a team of 29 expert reviewers, we conduct a systematic review of 445 LLM benchmarks from leading conferences in natural language processing and machine learning. Across the reviewed articles, we find patterns related to the measured phenomena, tasks, and scoring metrics which undermine the validity of the resulting claims. To address these shortcomings, we provide eight key recommendations and detailed actionable guidance to researchers and practitioners in developing LLM benchmarks. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）对于评估其能力并在部署前识别安全或稳健性问题至关重要。可靠地测量诸如“安全性”和“稳健性”等抽象和复杂的现象需要强大的建构效度，即衡量指标能够代表现象的关键方面。依托29名专家评审，我们系统性地审查了来自自然语言处理和机器学习顶级会议的445个LLM基准。在这些被审查的文章中，我们发现了与所测量的现象、任务和评分指标相关的模式，这些模式削弱了最终声明的有效性。为了应对这些不足，我们提供了八条关键建议，并为研究人员和实践者开发LLM基准提供了详尽的操作指南。 

---
# Separate the Wheat from the Chaff: Winnowing Down Divergent Views in Retrieval Augmented Generation 

**Title (ZH)**: 辨真伪，去伪存真：检索增强生成中分歧观点的筛选 

**Authors**: Song Wang, Zihan Chen, Peng Wang, Zhepei Wei, Zhen Tan, Yu Meng, Cong Shen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.04700)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge sources to address their limitations in accessing up-to-date or specialized information. A natural strategy to increase the likelihood of retrieving relevant information is to expand the number of retrieved documents. However, involving more documents could introduce significant noise, as many documents may be irrelevant or misleading, thereby reducing the overall accuracy of the generated responses. To overcome the challenge associated with handling a larger number of documents, we propose WinnowRAG, a novel RAG framework designed to systematically filter out noisy documents while preserving valuable content -- a process we refer to as winnowing. WinnowRAG operates in two stages: In Stage I, we perform query-aware clustering to group similar documents and form distinct topic clusters. Each cluster is assigned to an LLM agent for generating a unique answer. In Stage II, we perform winnowing, wherein a critic LLM evaluates the outputs of multiple agents and iteratively separates useful documents from noisy ones. To retain useful documents when discarding agents, we propose two strategic merging techniques to ensure that only relevant knowledge is used for generating the final response. Crucially, WinnowRAG is model-agnostic and does not require any model fine-tuning, making it easily adaptable to various tasks. Extensive experiments on various realistic datasets demonstrate the effectiveness of WinnowRAG over state-of-the-art baselines. 

**Abstract (ZH)**: 检索增强生成（RAG）通过整合外部知识源来增强大规模语言模型（LLMs），以解决其在访问最新或专门信息方面的局限性。增加检索相关信息的可能性的自然策略是扩大检索文档的数量。然而，涉及更多文档可能会引入大量噪声，因为许多文档可能是无关或误导性的，从而降低生成响应的整体准确性。为了解决处理更多文档所面临的挑战，我们提出了WinnowRAG，这是一种新颖的RAG框架，旨在系统地过滤掉噪声文档同时保留有价值的内容——我们称之为筛选。WinnowRAG分为两个阶段：在阶段I，我们执行查询感知聚类来对相似文档进行分组并形成不同的主题集群，每个集群分配给LLM代理生成独特的答案。在阶段II，我们执行筛选，在此过程中，一个评论员LLM评估多个代理的输出，并迭代地将有用文档从噪声文档中分离出来。为了在丢弃代理时保留有用文档，我们提出了两种策略性的合并技术，以确保仅使用相关知识生成最终响应。最关键的是，WinnowRAG 是模型无关的，不需要任何模型微调，使其易于适应各种任务。在各种现实数据集上的广泛实验证明了WinnowRAG优于最先进的基线方法。 

---
# multiMentalRoBERTa: A Fine-tuned Multiclass Classifier for Mental Health Disorder 

**Title (ZH)**: 多类别精神健康障碍分类器：fine-tuned multiMentalRoBERTa 

**Authors**: K M Sajjadul Islam, John Fields, Praveen Madiraju  

**Link**: [PDF](https://arxiv.org/pdf/2511.04698)  

**Abstract**: The early detection of mental health disorders from social media text is critical for enabling timely support, risk assessment, and referral to appropriate resources. This work introduces multiMentalRoBERTa, a fine-tuned RoBERTa model designed for multiclass classification of common mental health conditions, including stress, anxiety, depression, post-traumatic stress disorder (PTSD), suicidal ideation, and neutral discourse. Drawing on multiple curated datasets, data exploration is conducted to analyze class overlaps, revealing strong correlations between depression and suicidal ideation as well as anxiety and PTSD, while stress emerges as a broad, overlapping category. Comparative experiments with traditional machine learning methods, domain-specific transformers, and prompting-based large language models demonstrate that multiMentalRoBERTa achieves superior performance, with macro F1-scores of 0.839 in the six-class setup and 0.870 in the five-class setup (excluding stress), outperforming both fine-tuned MentalBERT and baseline classifiers. Beyond predictive accuracy, explainability methods, including Layer Integrated Gradients and KeyBERT, are applied to identify lexical cues that drive classification, with a particular focus on distinguishing depression from suicidal ideation. The findings emphasize the effectiveness of fine-tuned transformers for reliable and interpretable detection in sensitive contexts, while also underscoring the importance of fairness, bias mitigation, and human-in-the-loop safety protocols. Overall, multiMentalRoBERTa is presented as a lightweight, robust, and deployable solution for enhancing support in mental health platforms. 

**Abstract (ZH)**: 从社交媒体文本中早期检测心理健康障碍对于及时支持、风险评估和转介至合适资源至关重要。本文介绍了multiMentalRoBERTa，一种针对常见心理健康状况进行多分类的微调RoBERTa模型，包括压力、焦虑、抑郁、创伤后应激障碍(PTSD)、自杀意念和中立话语。本文利用多个精心收集的数据集进行了数据分析，揭示了抑郁与自杀意念、焦虑与PTSD之间 strong 的相关性，同时将压力视为一个广泛且重叠的类别。与传统的机器学习方法、领域特定的变压器以及提示驱动的大语言模型进行对比实验表明，multiMentalRoBERTa 在六类设置下实现了 0.839 的宏 F1 得分，在五类设置下（不包括压力）实现了 0.870 的宏 F1 得分，超越了微调的MentalBERT和基线分类器。除了预测准确性之外，使用解释性方法，如层整合梯度和KeyBERT，来识别驱动分类的词汇线索，特别是区分抑郁与自杀意念。研究结果强调了微调变压器在敏感情境下进行可靠和可解释检测的有效性，同时也突显了公平性、偏见缓解和人工在环安全协议的重要性。总体而言，multiMentalRoBERTa 提出了一个轻量级、稳健且可部署的解决方案，以增强心理健康平台的支持。 

---
# Simulating Misinformation Vulnerabilities With Agent Personas 

**Title (ZH)**: 使用代理人格模拟误导信息漏洞 

**Authors**: David Farr, Lynnette Hui Xian Ng, Stephen Prochaska, Iain J. Cruickshank, Jevin West  

**Link**: [PDF](https://arxiv.org/pdf/2511.04697)  

**Abstract**: Disinformation campaigns can distort public perception and destabilize institutions. Understanding how different populations respond to information is crucial for designing effective interventions, yet real-world experimentation is impractical and ethically challenging. To address this, we develop an agent-based simulation using Large Language Models (LLMs) to model responses to misinformation. We construct agent personas spanning five professions and three mental schemas, and evaluate their reactions to news headlines. Our findings show that LLM-generated agents align closely with ground-truth labels and human predictions, supporting their use as proxies for studying information responses. We also find that mental schemas, more than professional background, influence how agents interpret misinformation. This work provides a validation of LLMs to be used as agents in an agent-based model of an information network for analyzing trust, polarization, and susceptibility to deceptive content in complex social systems. 

**Abstract (ZH)**: 大规模语言模型驱动的代理模拟：对信息响应的研究代理模型验证 

---
# EncouRAGe: Evaluating RAG Local, Fast, and Reliable 

**Title (ZH)**: EncouRAGe: 评估RAG本地、快速且可靠的方法 

**Authors**: Jan Strich, Adeline Scharfenberg, Chris Biemann, Martin Semmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.04696)  

**Abstract**: We introduce EncouRAGe, a comprehensive Python framework designed to streamline the development and evaluation of Retrieval-Augmented Generation (RAG) systems using Large Language Models (LLMs) and Embedding Models. EncouRAGe comprises five modular and extensible components: Type Manifest, RAG Factory, Inference, Vector Store, and Metrics, facilitating flexible experimentation and extensible development. The framework emphasizes scientific reproducibility, diverse evaluation metrics, and local deployment, enabling researchers to efficiently assess datasets within RAG workflows. This paper presents implementation details and an extensive evaluation across multiple benchmark datasets, including 25k QA pairs and over 51k documents. Our results show that RAG still underperforms compared to the Oracle Context, while Hybrid BM25 consistently achieves the best results across all four datasets. We further examine the effects of reranking, observing only marginal performance improvements accompanied by higher response latency. 

**Abstract (ZH)**: 我们介绍EncouRAGe，一个综合性的Python框架，旨在简化使用大规模语言模型（LLMs）和嵌入模型的检索增强生成（RAG）系统的开发和评估。EncouRAGe 包含五个模块化和扩展性强的组件：类型说明文件、RAG工厂、推理、向量存储和指标，促进灵活的实验和扩展的开发。该框架强调科学的可重复性、多样化的评估指标以及本地部署，使研究人员能够高效地评估RAG工作流中的数据集。本文介绍了该框架的实现细节，并在多个基准数据集上进行了广泛的评估，包括25,000个问答对和超过51,000份文档。我们的结果显示，RAG的表现仍然落后于Oracle Context，而混合BM25在所有四个数据集上始终取得最佳结果。我们进一步探讨了重排的效果，发现虽然性能有所提升，但响应延迟增加。 

---
# Reasoning Up the Instruction Ladder for Controllable Language Models 

**Title (ZH)**: 沿着指令梯阶进行可控语言模型的推理 

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.04694)  

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises both aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks. These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior. 

**Abstract (ZH)**: 基于大型语言模型的指令层级解决：实现在高风险决策中的可靠性和可控性 

---
# Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks 

**Title (ZH)**: 自适应测试用于大语言模型评估：心理测量学替代静态基准 

**Authors**: Peiyu Li, Xiuxiu Tang, Si Chen, Ying Cheng, Ronald Metoyer, Ting Hua, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2511.04689)  

**Abstract**: Large language model evaluation requires thousands of benchmark items, making evaluations expensive and slow. Existing methods compute average accuracy across fixed item sets, treating all items equally despite varying quality and informativeness. We present ATLAS an adaptive testing framework using Item Response Theory (IRT) to estimate model ability through Fisher information-guided item selection. Our analysis of five major benchmarks reveals that 3-6% of items exhibit negative discrimination, indicating annotation errors that corrupt static evaluation. ATLAS achieves 90% item reduction while maintaining measurement precision: on HellaSwag (5,608 items), we match full-benchmark estimates using only 42 items with 0.154 MAE. Our framework maintains item exposure rates below 10% and test overlap at 16-27%, compared to static benchmarks where every model sees all items (100% exposure). Among 4,000+ tested models, IRT ranks differ from accuracy ranks: models with the same accuracy get different IRT scores, and 23-31% of all models shift by more than 10 rank positions. Code and calibrated item banks are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型评估需要成千上万的基准项，使得评估既昂贵又缓慢。现有方法通过固定项集计算平均准确性，尽管各项的质量和信息量不同，仍同等对待。我们提出了一个基于项目反应理论（IRT）的自适应测试框架ATLAS，通过斐舍信息引导的项目选择来估计模型能力。对五大基准的分析显示，3-6%的项目表现出负区分度，表明存在标注错误，会污染静态评估。ATLAS实现了90%的项目减少，同时保持测量精度：在HellaSwag（5,608项）上，仅使用42项就达到了与完整基准相同的0.154 MAE估计值。我们的框架将项目曝光率保持在10%以下，并且测试重叠率为16-27%，而静态基准中，每个模型都会看到所有项目（曝光率为100%）。在4,000多个测试模型中，IRT排名与准确性排名不同：具有相同准确度的模型获得不同的IRT分数，且23-31%的模型排名位置发生了超过10位的变化。代码和校准后的项目银行可在以下链接获取。 

---
# Stateful KV Cache Management for LLMs: Balancing Space, Time, Accuracy, and Positional Fidelity 

**Title (ZH)**: 面向LLMs的状态型KV缓存管理：空间、时间、准确性和位置保真的平衡 

**Authors**: Pratik Poudel  

**Link**: [PDF](https://arxiv.org/pdf/2511.04686)  

**Abstract**: The Key-Value (KV) cache is integral to efficient autoregressive inference in large language models (LLMs), yet its unbounded growth in stateful multi-turn scenarios presents major challenges. This paper examines the interplay between KV cache management strategies, the architectural context limits of models like meta-llama/Meta-Llama-3-8b-instruct, and the often-overlooked integrity of positional encodings. Through empirical analysis using a stateful benchmarking framework, we show that LLM generation quality degrades sharply when the accumulated KV cache approaches or exceeds the model's trained context window (e.g., 8192 tokens for Llama 3), a failure mode distinct from GPU memory exhaustion. Common eviction strategies, even high-retention ones (e.g., 99% via AttentionTop), can worsen performance if they disrupt positional coherence. Because LLMs rely on consistent positional signals (e.g., RoPE), compacting a cache by removing non-contiguous tokens can scramble these signals and lead to degenerative outputs. We further show that simple strategies preserving contiguous context blocks (e.g., keeping an initial "gist") can yield more coherent generations than complex or positionally disruptive ones. We advocate for eviction techniques that respect architectural limits, preserve positional structure, and view "cache health" holistically beyond mere size. 

**Abstract (ZH)**: KV缓存是大规模语言模型（LLMs）高效自回归推理的关键组成部分，但在状态ful多轮场景中其无界增长带来了重大挑战。本文探讨了KV缓存管理策略、如Meta-Llama/Meta-Llama-3-8b-instruct等模型的架构限制与其经常被忽视的位置编码完整性的相互作用。通过使用状态ful基准测试框架进行实证分析，我们发现当累积的KV缓存接近或超过模型训练上下文窗口（例如，Llama 3的8192个标记）时，LLM生成质量会急剧下降，这是一种不同于GPU内存耗尽的失败模式。即使高保留率的常见淘汰策略（例如，通过AttentionTop实现99%保留）也可能因破坏位置一致性而恶化性能。因为LLMs依赖于一致的位置信号（例如，RoPE），通过移除非连续标记来压缩缓存可能会扰乱这些信号并导致退化输出。我们进一步表明，保留连续上下文块的简单策略（例如，保留初始的“概要”）比破坏位置结构的复杂策略能产生更一致的生成结果。我们提倡尊重架构限制、保持位置结构并在整体上关注“缓存健康”的淘汰技术。 

---
