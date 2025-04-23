# Research on Navigation Methods Based on LLMs 

**Title (ZH)**: 基于大语言模型的导航方法研究 

**Authors**: Anlong Zhang, Jianmin Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.15600)  

**Abstract**: In recent years, the field of indoor navigation has witnessed groundbreaking advancements through the integration of Large Language Models (LLMs). Traditional navigation approaches relying on pre-built maps or reinforcement learning exhibit limitations such as poor generalization and limited adaptability to dynamic environments. In contrast, LLMs offer a novel paradigm for complex indoor navigation tasks by leveraging their exceptional semantic comprehension, reasoning capabilities, and zero-shot generalization properties. We propose an LLM-based navigation framework that leverages function calling capabilities, positioning the LLM as the central controller. Our methodology involves modular decomposition of conventional navigation functions into reusable LLM tools with expandable configurations. This is complemented by a systematically designed, transferable system prompt template and interaction workflow that can be easily adapted across different implementations. Experimental validation in PyBullet simulation environments across diverse scenarios demonstrates the substantial potential and effectiveness of our approach, particularly in achieving context-aware navigation through dynamic tool composition. 

**Abstract (ZH)**: 近年来，通过将大型语言模型（LLMs）集成到室内导航领域，取得了突破性的进展。传统的依赖预构建地图或强化学习的导航方法存在局限性，如泛化能力差和对动态环境适应能力有限。相比之下，LLMs通过利用其出色的语义理解、推理能力和零样本泛化能力，为复杂的室内导航任务提供了新的范式。我们提出了一种基于LLM的导航框架，将LLM定位为中心控制器，并利用其实现功能调用能力。该方法包括将传统导航功能模块化分解为可重用的LLM工具，并具有可扩展的配置。此外，还设计了一种系统化、可转移的系统提示模板和交互工作流，可以在不同实施中轻松适应。在PyBullet仿真环境中对多样场景进行的实验验证表明，我们的方法具有巨大的潜力和效果，特别是在通过动态工具组合实现上下文感知导航方面。 

---
# Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations 

**Title (ZH)**: 噪声对模型温度考虑因素下抽象与推理语料任务（ARC）中大语言模型性能的影响 

**Authors**: Nikhil Khandalkar, Pavan Yadav, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15903)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility. 

**Abstract (ZH)**: 近期大规模语言模型的进展引发了对其结构化推理能力的兴趣，特别是在涉及抽象和模式识别的任务中。抽象与推理 corpus (ARC) 基准在评估这些能力方面发挥着关键作用，通过测试 AI 模型在解决新型问题时的泛化能力。尽管 GPT-4o 在零噪声条件下能够解决所有 ARC 任务，表现出强大的性能，但其他模型如 DeepSeek R1 和 LLaMA 3.2 未能解决任何任务，表明它们在超越简单模式匹配进行推理方面存在局限性。为了探索这一差距，我们系统地评估了这些模型在不同的噪声水平和温度设置下的表现。结果显示，无论架构如何，噪声的引入都会一致地损害模型性能。这一下降揭示了一个共同的脆弱性：尽管目前的大规模语言模型显示了抽象推理的迹象，但它们对输入干扰极其敏感。这种脆弱性引发了对其在噪声和不确定性普遍存在的真实世界应用场景中的适用性的担忧。通过比较不同模型架构对这些挑战的反应，我们提供了现代大规模语言模型在推理任务中结构弱点的见解。这项工作强调了开发更 robust 和适应性强的 AI 系统的必要性，这些系统能够处理真实世界场景中固有的模糊性和变异性。我们的发现旨在指导未来研究，朝着增强模型泛化能力、 robust 性和与人类认知灵活性相一致的方向发展。 

---
# Implementing Rational Choice Functions with LLMs and Measuring their Alignment with User Preferences 

**Title (ZH)**: 用LLMs实现理性选择函数并衡量其与用户偏好的一致程度 

**Authors**: Anna Karnysheva, Christian Drescher, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2504.15719)  

**Abstract**: As large language models (LLMs) become integral to intelligent user interfaces (IUIs), their role as decision-making agents raises critical concerns about alignment. Although extensive research has addressed issues such as factuality, bias, and toxicity, comparatively little attention has been paid to measuring alignment to preferences, i.e., the relative desirability of different alternatives, a concept used in decision making, economics, and social choice theory. However, a reliable decision-making agent makes choices that align well with user preferences.
In this paper, we generalize existing methods that exploit LLMs for ranking alternative outcomes by addressing alignment with the broader and more flexible concept of user preferences, which includes both strict preferences and indifference among alternatives. To this end, we put forward design principles for using LLMs to implement rational choice functions, and provide the necessary tools to measure preference satisfaction. We demonstrate the applicability of our approach through an empirical study in a practical application of an IUI in the automotive domain. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在智能用户界面（IUIs）中变得至关重要，它们作为决策代理的角色引发了对其与用户偏好对齐的关键关注。尽管已有大量研究解决了事实性、偏见和毒性等问题，但在测量与偏好对齐方面关注相对较少，即不同替代方案的相对理想性，这一概念在决策制定、经济学和社会选择理论中被使用。然而，可靠的决策代理能够做出与用户偏好高度一致的选择。

在本文中，我们通过将用户偏好更广泛和更具弹性的概念应用到现有方法中，来扩展现有的利用LLMs进行替代结果排名的方法，包括严格偏好和对替代方案的无差异。为此，我们提出了使用LLMs实现理性选择函数的设计原则，并提供了衡量偏好满足程度所需的工具。我们通过在汽车领域的智能用户界面实际应用中的实证研究，展示了我们方法的适用性。 

---
# DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models 

**Title (ZH)**: DianJin-R1: 评估与提升大型语言模型的金融推理能力 

**Authors**: Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15716)  

**Abstract**: Effective reasoning remains a core challenge for large language models (LLMs) in the financial domain, where tasks often require domain-specific knowledge, precise numerical calculations, and strict adherence to compliance rules. We propose DianJin-R1, a reasoning-enhanced framework designed to address these challenges through reasoning-augmented supervision and reinforcement learning. Central to our approach is DianJin-R1-Data, a high-quality dataset constructed from CFLUE, FinQA, and a proprietary compliance corpus (Chinese Compliance Check, CCC), combining diverse financial reasoning scenarios with verified annotations. Our models, DianJin-R1-7B and DianJin-R1-32B, are fine-tuned from Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct using a structured format that generates both reasoning steps and final answers. To further refine reasoning quality, we apply Group Relative Policy Optimization (GRPO), a reinforcement learning method that incorporates dual reward signals: one encouraging structured outputs and another rewarding answer correctness. We evaluate our models on five benchmarks: three financial datasets (CFLUE, FinQA, and CCC) and two general reasoning benchmarks (MATH-500 and GPQA-Diamond). Experimental results show that DianJin-R1 models consistently outperform their non-reasoning counterparts, especially on complex financial tasks. Moreover, on the real-world CCC dataset, our single-call reasoning models match or even surpass the performance of multi-agent systems that require significantly more computational cost. These findings demonstrate the effectiveness of DianJin-R1 in enhancing financial reasoning through structured supervision and reward-aligned learning, offering a scalable and practical solution for real-world applications. 

**Abstract (ZH)**: Effective Reasoning Remain a Core Challenge for Large Language Models (LLMs) in the Financial Domain, Where Tasks Often Require Domain-Specific Knowledge, Precise Numerical Calculations, and Strict Adherence to Compliance Rules: DianJin-R1——一种通过推理增强监督和强化学习应对这些挑战的框架 

---
# A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings 

**Title (ZH)**: 基于LoRA的方法：在资源受限环境中 fine-tuning 语言模型以提供教育指导 

**Authors**: Md Millat, Md Motiur  

**Link**: [PDF](https://arxiv.org/pdf/2504.15610)  

**Abstract**: The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy. 

**Abstract (ZH)**: 当前研究描述了一种经济有效的方法，将大型语言模型（LLM）适应于考虑海外学习背景下学术指导的应用，并适用于低资源环境下的文化适应。通过使用Mistral-7B-Instruct模型与低秩适应（LoRA）方法和4位量化方法，该模型在增强领域特定性的同时保持计算效率进行了分两个阶段的训练。在第一阶段，模型通过Gemini Pro API用合成数据集进行了预训练；在第二阶段，模型使用来自StudyAbroadGPT项目的手工整理数据集进行训练，以实现更具上下文针对性的响应。技术革新包括内存高效量化、参数高效适应以及通过Weights & Biases进行连续训练指标分析。经过训练，本研究显示训练损失减少了52.7%，在领域特定建议方面达到92%的准确率，实现了95%的Markdown格式支持，并在标准GPU设备上达到每秒100个样本的中位运行速率。这些发现支持对指令调整过的LLM在教育顾问中的有效应用，特别是在低资源机构场景中的应用。局限性包括通用性的降低以及使用合成生成的数据集，但此框架适用于扩展新的多语言增强和实时学术指导过程。未来方向可能包括引入检索增强生成、应用动态量化程序以及连接到实时学术数据库以提高适应能力和准确性。 

---
# Learning Adaptive Parallel Reasoning with Language Models 

**Title (ZH)**: 学习自适应并行推理的语言模型 

**Authors**: Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2504.15466)  

**Abstract**: Scaling inference-time computation has substantially improved the reasoning capabilities of language models. However, existing methods have significant limitations: serialized chain-of-thought approaches generate overly long outputs, leading to increased latency and exhausted context windows, while parallel methods such as self-consistency suffer from insufficient coordination, resulting in redundant computations and limited performance gains. To address these shortcomings, we propose Adaptive Parallel Reasoning (APR), a novel reasoning framework that enables language models to orchestrate both serialized and parallel computations end-to-end. APR generalizes existing reasoning methods by enabling adaptive multi-threaded inference using spawn() and join() operations. A key innovation is our end-to-end reinforcement learning strategy, optimizing both parent and child inference threads to enhance task success rate without requiring predefined reasoning structures. Experiments on the Countdown reasoning task demonstrate significant benefits of APR: (1) higher performance within the same context window (83.4% vs. 60.0% at 4k context); (2) superior scalability with increased computation (80.1% vs. 66.6% at 20k total tokens); (3) improved accuracy at equivalent latency (75.2% vs. 57.3% at approximately 5,000ms). APR represents a step towards enabling language models to autonomously optimize their reasoning processes through adaptive allocation of computation. 

**Abstract (ZH)**: 自适应并行推理：端到端调控的推理框架 

---
# KeDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments 

**Title (ZH)**: KeDiff: 基于键相似性的KV缓存淘汰策略以适应资源受限环境中的长上下文LLM推理 

**Authors**: Junyoung Park, Dalton Jones, Matt Morse, Raghavv Goel, Mingu Lee, Chris Lott  

**Link**: [PDF](https://arxiv.org/pdf/2504.15364)  

**Abstract**: In this work, we demonstrate that distinctive keys during LLM inference tend to have high attention scores. We explore this phenomenon and propose KeyDiff, a training-free KV cache eviction method based on key similarity. This method facilitates the deployment of LLM-based application requiring long input prompts in resource-constrained environments with limited memory and compute budgets. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We demonstrate that KeyDiff computes the optimal solution to a KV cache selection problem that maximizes key diversity, providing a theoretical understanding of KeyDiff. Notably,KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. We demonstrate the effectiveness of KeyDiff across diverse tasks and models, illustrating a performance gap of less than 0.04\% with 8K cache budget ($\sim$ 23\% KV cache reduction) from the non-evicting baseline on the LongBench benchmark for Llama 3.1-8B and Llama 3.2-3B. 

**Abstract (ZH)**: 在本文中，我们证明了在LLM推理过程中，具有高注意力分数的键通常具有显著性。我们探讨了这一现象，并提出了KeyDiff，一种基于键相似性的无训练KV缓存淘汰方法。该方法有助于在资源受限且内存和计算预算有限的环境中部署要求长输入提示的LLM应用程序。与其它KV缓存淘汰方法不同，KeyDiff可以在严格的资源限制内处理任意长度的提示，并高效生成响应。我们证明KeyDiff计算了最大化键多样性的一项KV缓存选择问题的最优解，提供了对KeyDiff的理论理解。值得注意的是，KeyDiff不依赖于注意力分数，允许使用优化的注意力机制如FlashAttention。我们展示了KeyDiff在多种任务和模型上的有效性，在LongBench基准上，与非淘汰基线相比，对于Llama 3.1-8B和Llama 3.2-3B，使用8K缓存预算（约减少23%的KV缓存）时性能差距小于0.04%。 

---
# LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities 

**Title (ZH)**: LLMs是贪婪代理：RL微调对决策能力的影响 

**Authors**: Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16078)  

**Abstract**: The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making. 

**Abstract (ZH)**: 大型语言模型在各类代理应用中的成功激发了广泛兴趣。一个关键假设是，大型语言模型通过利用常识和链式推理（CoT）能够有效探索并高效解决复杂领域。然而，大型语言模型代理存在次优探索和认知-行动差距的问题，即无法有效地将模型中存在的知识付诸行动。在这项工作中，我们系统研究了大型语言模型在决策场景中表现次优的原因。特别是，我们详细 examines 三种常见的失败模式：贪婪性、频率偏差和认知-行动差距。我们通过强化学习（RL）对自动生成的CoT推理进行微调，提出缓解这些缺点的方法。我们的实验表明，通过强化学习微调可以增强大型语言模型的决策能力，提高探索度并缩小认知-行动差距。最后，我们研究了经典探索机制（如$\epsilon$-贪婪）和特定于大型语言模型的方法（如自我纠正和自我一致性），以实现更有效的大型语言模型微调，从而更好地应用于决策场景。 

---
# LLMs meet Federated Learning for Scalable and Secure IoT Management 

**Title (ZH)**: LLMs与联邦学习结合实现可扩展和安全的物联网管理 

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.16032)  

**Abstract**: The rapid expansion of IoT ecosystems introduces severe challenges in scalability, security, and real-time decision-making. Traditional centralized architectures struggle with latency, privacy concerns, and excessive resource consumption, making them unsuitable for modern large-scale IoT deployments. This paper presents a novel Federated Learning-driven Large Language Model (FL-LLM) framework, designed to enhance IoT system intelligence while ensuring data privacy and computational efficiency. The framework integrates Generative IoT (GIoT) models with a Gradient Sensing Federated Strategy (GSFS), dynamically optimizing model updates based on real-time network conditions. By leveraging a hybrid edge-cloud processing architecture, our approach balances intelligence, scalability, and security in distributed IoT environments. Evaluations on the IoT-23 dataset demonstrate that our framework improves model accuracy, reduces response latency, and enhances energy efficiency, outperforming traditional FL techniques (i.e., FedAvg, FedOpt). These findings highlight the potential of integrating LLM-powered federated learning into large-scale IoT ecosystems, paving the way for more secure, scalable, and adaptive IoT management solutions. 

**Abstract (ZH)**: 物联网生态系统快速扩张引入了在扩展性、安全性和实时决策方面严峻的挑战。传统集中式架构难以应对延迟、隐私问题以及过度的资源消耗，使其不适合现代大规模物联网部署。本文提出了一个新颖的联邦学习驱动的大语言模型（FL-LLM）框架，旨在提高物联网系统的智能水平同时确保数据隐私和计算效率。该框架将生成式物联网（GIoT）模型与梯度感知联邦策略（GSFS）相结合，根据实时网络条件动态优化模型更新。通过利用混合边缘-云处理架构，我们的方法在分布式物联网环境中平衡了智能、扩展性和安全性。在IoT-23数据集上的评估表明，我们的框架提高了模型准确性、降低了响应延迟并增强了能量效率，优于传统联邦学习技术（如FedAvg、FedOpt）。这些结果突显了将LLM驱动的联邦学习整合到大规模物联网生态系统中的潜在价值，为更安全、更具扩展性和适应性的物联网管理解决方案铺平了道路。 

---
# Benchmarking LLM for Code Smells Detection: OpenAI GPT-4.0 vs DeepSeek-V3 

**Title (ZH)**: LLM在代码异味检测中的基准测试：OpenAI GPT-4.0 vs DeepSeek-V3 

**Authors**: Ahmed R. Sadik, Siddhata Govind  

**Link**: [PDF](https://arxiv.org/pdf/2504.16027)  

**Abstract**: Determining the most effective Large Language Model for code smell detection presents a complex challenge. This study introduces a structured methodology and evaluation matrix to tackle this issue, leveraging a curated dataset of code samples consistently annotated with known smells. The dataset spans four prominent programming languages Java, Python, JavaScript, and C++; allowing for cross language comparison. We benchmark two state of the art LLMs, OpenAI GPT 4.0 and DeepSeek-V3, using precision, recall, and F1 score as evaluation metrics. Our analysis covers three levels of detail: overall performance, category level performance, and individual code smell type performance. Additionally, we explore cost effectiveness by comparing the token based detection approach of GPT 4.0 with the pattern-matching techniques employed by DeepSeek V3. The study also includes a cost analysis relative to traditional static analysis tools such as SonarQube. The findings offer valuable guidance for practitioners in selecting an efficient, cost effective solution for automated code smell detection 

**Abstract (ZH)**: 确定最有效的大型语言模型进行代码异味检测是一项复杂挑战。本研究引入了一种结构化的方法和评估矩阵以应对这一问题，利用了一个经过一致标注已知异味的代码样本数据集。该数据集包含了四种主流编程语言（Java、Python、JavaScript 和 C++），便于进行跨语言比较。我们使用准确率、召回率和F1分数作为评估指标，对标了两种先进的大型语言模型——OpenAI GPT 4.0 和 DeepSeek-V3。我们的分析涵盖三个层次的详细情况：总体性能、类别性能和单个代码异味类型性能。此外，我们还探讨了成本效益，通过比较GPT 4.0基于令牌的检测方法与DeepSeek V3使用的模式匹配技术。该研究还包括与SonarQube等传统静态分析工具的成本分析。研究发现为实践者选择高效、成本效益高的自动化代码异味检测解决方案提供了宝贵的指导。 

---
# CAPO: Cost-Aware Prompt Optimization 

**Title (ZH)**: CAPO: 成本意识的提示优化 

**Authors**: Tom Zehle, Moritz Schlager, Timo Heiß, Matthias Feurer  

**Link**: [PDF](https://arxiv.org/pdf/2504.16005)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing by solving a wide range of tasks simply guided by a prompt. Yet their performance is highly sensitive to prompt formulation. While automated prompt optimization addresses this challenge by finding optimal prompts, current methods require a substantial number of LLM calls and input tokens, making prompt optimization expensive. We introduce CAPO (Cost-Aware Prompt Optimization), an algorithm that enhances prompt optimization efficiency by integrating AutoML techniques. CAPO is an evolutionary approach with LLMs as operators, incorporating racing to save evaluations and multi-objective optimization to balance performance with prompt length. It jointly optimizes instructions and few-shot examples while leveraging task descriptions for improved robustness. Our extensive experiments across diverse datasets and LLMs demonstrate that CAPO outperforms state-of-the-art discrete prompt optimization methods in 11/15 cases with improvements up to 21%p. Our algorithm achieves better performances already with smaller budgets, saves evaluations through racing, and decreases average prompt length via a length penalty, making it both cost-efficient and cost-aware. Even without few-shot examples, CAPO outperforms its competitors and generally remains robust to initial prompts. CAPO represents an important step toward making prompt optimization more powerful and accessible by improving cost-efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过简单的提示解决了一大片自然语言处理任务，但其表现高度依赖于提示的制定。虽然自动化提示优化通过寻找最优提示来应对这一挑战，但现有方法需要大量的LLM调用和输入 token，使得提示优化成本高昂。我们提出了一种成本感知提示优化（CAPO）算法，通过结合自动化机器学习技术来提高提示优化的效率。CAPO 是一种进化方法，以大型语言模型作为操作符，通过竞速节省评估次数，并通过多目标优化平衡性能和提示长度。它同时优化指令和少量示例，利用任务描述提高鲁棒性。我们的跨多种数据集和大型语言模型的广泛实验表明，CAPO 在 11/15 的情况下优于最先进的离散提示优化方法，性能改善高达 21%。我们的算法在较小的预算下就能获得更好的性能，通过竞速节省评估次数，并通过长度惩罚减少平均提示长度，使其在成本效率和成本意识方面都表现出色。即使没有少量示例，CAPO 也优于其竞争对手，并且通常对初始提示具有鲁棒性。CAPO 是朝着使提示优化更加强大和更加普及的重要一步，通过提高成本效率来实现这一目标。 

---
# W-PCA Based Gradient-Free Proxy for Efficient Search of Lightweight Language Models 

**Title (ZH)**: 基于W-PCA的无梯度代理模型以高效搜索轻量级语言模型 

**Authors**: Shang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15983)  

**Abstract**: The demand for efficient natural language processing (NLP) systems has led to the development of lightweight language models. Previous work in this area has primarily focused on manual design or training-based neural architecture search (NAS) methods. Recently, zero-shot NAS methods have been proposed for evaluating language models without the need for training. However, prevailing approaches to zero-shot NAS often face challenges such as biased evaluation metrics and computational inefficiencies. In this paper, we introduce weight-weighted PCA (W-PCA), a novel zero-shot NAS method specifically tailored for lightweight language models. Our approach utilizes two evaluation proxies: the parameter count and the number of principal components with cumulative contribution exceeding $\eta$ in the feed-forward neural (FFN) layer. Additionally, by eliminating the need for gradient computations, we optimize the evaluation time, thus enhancing the efficiency of designing and evaluating lightweight language models. We conduct a comparative analysis on the GLUE and SQuAD datasets to evaluate our approach. The results demonstrate that our method significantly reduces training time compared to one-shot NAS methods and achieves higher scores in the testing phase compared to previous state-of-the-art training-based methods. Furthermore, we perform ranking evaluations on a dataset sampled from the FlexiBERT search space. Our approach exhibits superior ranking correlation and further reduces solving time compared to other zero-shot NAS methods that require gradient computation. 

**Abstract (ZH)**: 基于稀疏编码的零样本神经架构搜索方法：针对轻量级语言模型的W-PCA 

---
# FairTranslate: An English-French Dataset for Gender Bias Evaluation in Machine Translation by Overcoming Gender Binarity 

**Title (ZH)**: FairTranslate: 一种克服性别二元性偏差的英法语数据集用于机器翻译中的性别偏差评估 

**Authors**: Fanny Jourdan, Yannick Chevalier, Cécile Favre  

**Link**: [PDF](https://arxiv.org/pdf/2504.15941)  

**Abstract**: Large Language Models (LLMs) are increasingly leveraged for translation tasks but often fall short when translating inclusive language -- such as texts containing the singular 'they' pronoun or otherwise reflecting fair linguistic protocols. Because these challenges span both computational and societal domains, it is imperative to critically evaluate how well LLMs handle inclusive translation with a well-founded framework.
This paper presents FairTranslate, a novel, fully human-annotated dataset designed to evaluate non-binary gender biases in machine translation systems from English to French. FairTranslate includes 2418 English-French sentence pairs related to occupations, annotated with rich metadata such as the stereotypical alignment of the occupation, grammatical gender indicator ambiguity, and the ground-truth gender label (male, female, or inclusive).
We evaluate four leading LLMs (Gemma2-2B, Mistral-7B, Llama3.1-8B, Llama3.3-70B) on this dataset under different prompting procedures. Our results reveal substantial biases in gender representation across LLMs, highlighting persistent challenges in achieving equitable outcomes in machine translation. These findings underscore the need for focused strategies and interventions aimed at ensuring fair and inclusive language usage in LLM-based translation systems.
We make the FairTranslate dataset publicly available on Hugging Face, and disclose the code for all experiments on GitHub. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在翻译任务中越来越广泛应用，但在处理包容性语言时往往表现不佳——例如包含单数“they”代词或体现公平语言规范的文本。由于这些挑战涉及计算和社会两个领域，建立一个坚实框架来批判性地评估LLMs在处理包容性翻译方面的表现至关重要。

本文介绍了FairTranslate，这是一个新型的完全由人类注释的数据集，旨在评估从英语到法语的机器翻译系统中的非二元性别偏见。FairTranslate包含2418组职业相关的英法语句子对，并标注了丰富的元数据，包括职业的刻板印象对齐、语法性别指标的模糊性以及真实的性别标签（男性、女性或包容性）。

我们使用四种领先的LLM（Gemma2-2B、Mistral-7B、Llama3.1-8B、Llama3.3-70B）在不同的提示程序下评估了这个数据集。我们的结果显示，这些LLM在性别表现方面存在显著偏差，突显了在机器翻译中实现公平 resultados 的持续挑战。这些发现强调了需要针对性的策略和干预措施，以确保在基于LLM的翻译系统中使用公平和包容性语言。

我们已将FairTranslate数据集在Hugging Face上公开，并在GitHub上披露所有实验的代码。 

---
# Dynamic Early Exit in Reasoning Models 

**Title (ZH)**: 推理模型中的动态早期退出 

**Authors**: Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15895)  

**Abstract**: Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points (e.g.,"Wait" tokens) and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on multiple reasoning benchmarks MATH-500, AMC 2023, GPQA Diamond and AIME 2024 show that the proposed method is consistently effective on deepseek-series reasoning LLMs, reducing the length of CoT sequences by an average of 31% to 43% while improving accuracy by 1.7% to 5.7%. 

**Abstract (ZH)**: Recent advances in大型推理语言模型（LRLMs）依赖于测试时缩放，这将长链推理（CoT）生成扩展以解决复杂任务。然而，长CoT中的过度推理不仅降低了问题解决的效率，还因推理步骤极度详细或冗余而增加了准确性损失的风险。我们提出了一种简单而有效的方法，允许LLMs在生成过程中通过早期退出自我截断CoT序列。该方法不依赖于固定的启发式规则，而是监控潜在推理过渡点（例如，“等待”标记）的模型行为，并在模型对试答表现出高信心时动态终止下一个推理链的生成。该方法不需要额外训练，并且可以无缝集成到现有的o1-like推理LLMs中。实验结果显示，该方法在MATH-500、AMC 2023、GPQA Diamond和AIME 2024多个推理基准测试中对深seek系列推理LLMs始终有效，平均减少CoT序列长度31%至43%，同时提高准确性1.7%至5.7%。 

---
# Insights from Verification: Training a Verilog Generation LLM with Reinforcement Learning with Testbench Feedback 

**Title (ZH)**: 从验证中获得的见解：使用测试台反馈的强化学习训练Verilog生成Large Language Model 

**Authors**: Ning Wang, Bingkun Yao, Jie Zhou, Yuchen Hu, Xi Wang, Nan Guan, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15804)  

**Abstract**: Large language models (LLMs) have shown strong performance in Verilog generation from natural language description. However, ensuring the functional correctness of the generated code remains a significant challenge. This paper introduces a method that integrates verification insights from testbench into the training of Verilog generation LLMs, aligning the training with the fundamental goal of hardware design: functional correctness. The main obstacle in using LLMs for Verilog code generation is the lack of sufficient functional verification data, particularly testbenches paired with design specifications and code. To address this problem, we introduce an automatic testbench generation pipeline that decomposes the process and uses feedback from the Verilog compiler simulator (VCS) to reduce hallucination and ensure correctness. We then use the testbench to evaluate the generated codes and collect them for further training, where verification insights are introduced. Our method applies reinforcement learning (RL), specifically direct preference optimization (DPO), to align Verilog code generation with functional correctness by training preference pairs based on testbench outcomes. In evaluations on VerilogEval-Machine, VerilogEval-Human, RTLLM v1.1, RTLLM v2, and VerilogEval v2, our approach consistently outperforms state-of-the-art baselines in generating functionally correct Verilog code. We open source all training code, data, and models at this https URL. 

**Abstract (ZH)**: 大规模语言模型在从自然语言描述生成Verilog代码方面表现出强大的性能，但在确保生成代码的功能正确性方面仍面临重大挑战。本文提出了一种方法，将测试平台的验证洞察力集成到Verilog生成的大规模语言模型训练中，使训练与硬件设计的基本目标——功能正确性——保持一致。使用大规模语言模型进行Verilog代码生成的主要障碍在于缺乏足够的功能性验证数据，特别是与设计规范和代码配对的测试平台。为了解决这一问题，我们引入了一个自动测试平台生成流水线，将过程分解，并利用Verilog编译器模拟器（VCS）的反馈来减少幻觉并确保正确性。然后使用测试平台评估生成的代码，并收集用于进一步训练的数据，其中引入了验证洞察力。我们的方法利用强化学习（RL），特别是直接偏好优化（DPO），通过基于测试平台结果训练偏好对来使Verilog代码生成与功能性正确性保持一致。在VerilogEval-Machine、VerilogEval-Human、RTLLM v1.1、RTLLM v2 和 VerilogEval v2 上的评估中，我们的方法在生成功能正确性Verilog代码方面始终优于最新的基线方法。我们已将所有训练代码、数据和模型开源发布在该网址。 

---
# A closer look at how large language models trust humans: patterns and biases 

**Title (ZH)**: 对大型语言模型信任人类的现象进行更深入的探讨：模式与偏见 

**Authors**: Valeria Lerman, Yaniv Dover  

**Link**: [PDF](https://arxiv.org/pdf/2504.15801)  

**Abstract**: As large language models (LLMs) and LLM-based agents increasingly interact with humans in decision-making contexts, understanding the trust dynamics between humans and AI agents becomes a central concern. While considerable literature studies how humans trust AI agents, it is much less understood how LLM-based agents develop effective trust in humans. LLM-based agents likely rely on some sort of implicit effective trust in trust-related contexts (e.g., evaluating individual loan applications) to assist and affect decision making. Using established behavioral theories, we develop an approach that studies whether LLMs trust depends on the three major trustworthiness dimensions: competence, benevolence and integrity of the human subject. We also study how demographic variables affect effective trust. Across 43,200 simulated experiments, for five popular language models, across five different scenarios we find that LLM trust development shows an overall similarity to human trust development. We find that in most, but not all cases, LLM trust is strongly predicted by trustworthiness, and in some cases also biased by age, religion and gender, especially in financial scenarios. This is particularly true for scenarios common in the literature and for newer models. While the overall patterns align with human-like mechanisms of effective trust formation, different models exhibit variation in how they estimate trust; in some cases, trustworthiness and demographic factors are weak predictors of effective trust. These findings call for a better understanding of AI-to-human trust dynamics and monitoring of biases and trust development patterns to prevent unintended and potentially harmful outcomes in trust-sensitive applications of AI. 

**Abstract (ZH)**: 基于大规模语言模型的代理在决策制定情境中与人类交互时，理解人类与AI代理之间的信任动态成为中央关注点。尽管大量文献研究了人类如何信任AI代理，但对于基于大规模语言模型的代理如何发展有效的信任机制却知之甚少。基于大规模语言模型的代理可能依赖于某种隐含的有效信任，在与人类交互的背景下（例如评估个人贷款申请）来协助和影响决策。通过运用已确立的行为理论，我们开发了一种方法来研究基于大规模语言模型的信任是否依赖于人类主体的三大信任维度：能力、善意和诚信。我们还研究了人口统计变量如何影响有效信任。在43,200次模拟实验中，针对五种流行的语言模型，在五种不同的情景下，我们发现基于大规模语言模型的信任发展总体上与人类信任发展相似。我们发现，在大多数情况下，但并非所有情况下，基于大规模语言模型的信任强烈取决于信任度，而在一些情况下，年龄、宗教和性别也会影响信任，尤其是在金融场景中。这一情形特别适用于文献中常见的场景和较新的模型。虽然总体模式与人类有效信任形成的机制一致，但不同的模型在估计信任时表现出差异；在某些情况下，信任度和人口统计因素不是有效信任的弱预测因子。这些发现呼吁对AI到人类的信任动态有更深入的理解，并监测偏见和信任发展模式，以防止在信任敏感的AI应用中产生意外且可能有害的结果。 

---
# Automated Creativity Evaluation for Large Language Models: A Reference-Based Approach 

**Title (ZH)**: 基于参考的大型语言模型自动化创造力评估方法 

**Authors**: Ruizhe Li, Chiwei Zhu, Benfeng Xu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15784)  

**Abstract**: Creative writing is a key capability of Large Language Models (LLMs), with potential applications in literature, storytelling, and various creative domains. However, evaluating the creativity of machine-generated texts remains a significant challenge, as existing methods either rely on costly manual annotations or fail to align closely with human assessments. In this paper, we propose an effective automated evaluation method based on the Torrance Test of Creative Writing (TTCW), which evaluates creativity as product. Our method employs a reference-based Likert-style approach, scoring generated creative texts relative to high-quality reference texts across various tests. Experimental results demonstrate that our method significantly improves the alignment between LLM evaluations and human assessments, achieving a pairwise accuracy of 0.75 (+15\%). 

**Abstract (ZH)**: 大型语言模型的创造性写作是一个关键能力，具有在文学、叙事以及各种创造性领域中的潜在应用。然而，评估机器生成文本的创造性仍然是一个重大挑战，因为现有方法要么依赖昂贵的手动标注，要么无法紧密契合人类评估。本文提出了一种基于Torrance创造性写作测试（TTCW）的有效自动化评价方法，该方法将创造性作为产品进行评估。该方法采用了参考文本为基础的李克特量表方法，在多种测试中根据高质量的参考文本对生成的创造性文本进行评分。实验结果表明，该方法显著改善了大型语言模型评价与人类评估之间的契合度，实现了成对准确性提高15%（达到0.75）。 

---
# VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation 

**Title (ZH)**: VeriCoder: 通过功能正确性验证提升基于LLM的RTL代码生成 

**Authors**: Anjiang Wei, Huanmi Tan, Tarun Suresh, Daniel Mendoza, Thiago S. F. X. Teixeira, Ke Wang, Caroline Trippel, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2504.15659)  

**Abstract**: Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of over 125,000 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4% respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. 

**Abstract (ZH)**: 近期大型语言模型的发展激发了将其应用于电子设计自动化任务，特别是寄存器传输级代码生成方面的兴趣。虽然已经引入了多个RTL数据集，但大多数侧重于语法有效性而非功能验证，导致训练示例能够编译但可能无法实现预期的行为。我们提出VERICODER，一种针对经过功能 correctness 验证的数据集进行微调的RTL代码生成模型。该微调数据集采用一种新颖的方法构建，该方法结合了单元测试生成和反馈导向的细化。给定一种自然语言规格和一个初始RTL设计，我们提示一个教师模型（GPT-4o-mini）生成单元测试并根据生成的测试及其仿真结果迭代修订RTL设计。必要时，教师模型还更新测试以确保其符合自然语言规格。通过这个过程，数据集中的每个示例都经过功能验证，包括自然语言描述、RTL实现和通过的测试。在包含超过125,000个示例的数据集上微调VERICODER后，其在VerilogEval和RTLLM上的功能正确性指标达到了最先进的水平，相对增益分别为71.7%和27.4%。进一步的消融研究显示，训练于我们功能验证数据集上的模型优于训练于功能未验证数据集上的模型，强调了高质量数据集在RTL代码生成中的重要性。 

---
# Cost-Effective Text Clustering with Large Language Models 

**Title (ZH)**: 成本效益高的文本聚类方法：利用大规模语言模型 

**Authors**: Hongtao Wang, Taiyan Zhang, Renchi Yang, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15640)  

**Abstract**: Text clustering aims to automatically partition a collection of text documents into distinct clusters based on linguistic features. In the literature, this task is usually framed as metric clustering based on text embeddings from pre-trained encoders or a graph clustering problem upon pairwise similarities from an oracle, e.g., a large ML model. Recently, large language models (LLMs) bring significant advancement in this field by offering contextualized text embeddings and highly accurate similarity scores, but meanwhile, present grand challenges to cope with substantial computational and/or financial overhead caused by numerous API-based queries or inference calls to the models.
In response, this paper proposes TECL, a cost-effective framework that taps into the feedback from LLMs for accurate text clustering within a limited budget of queries to LLMs. Under the hood, TECL adopts our EdgeLLM or TriangleLLM to construct must-link/cannot-link constraints for text pairs, and further leverages such constraints as supervision signals input to our weighted constrained clustering approach to generate clusters. Particularly, EdgeLLM (resp. TriangleLLM) enables the identification of informative text pairs (resp. triplets) for querying LLMs via well-thought-out greedy algorithms and accurate extraction of pairwise constraints through carefully-crafted prompting techniques. Our experiments on multiple benchmark datasets exhibit that TECL consistently and considerably outperforms existing solutions in unsupervised text clustering under the same query cost for LLMs. 

**Abstract (ZH)**: 基于LLM反馈的低成本文本聚类框架TECL 

---
# DR.FIX: Automatically Fixing Data Races at Industry Scale 

**Title (ZH)**: DR.FIX：大规模自动修复数据竞态条件 

**Authors**: Farnaz Behrang, Zhizhou Zhang, Georgian-Vlad Saioc, Peng Liu, Milind Chabbi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15637)  

**Abstract**: Data races are a prevalent class of concurrency bugs in shared-memory parallel programs, posing significant challenges to software reliability and reproducibility. While there is an extensive body of research on detecting data races and a wealth of practical detection tools across various programming languages, considerably less effort has been directed toward automatically fixing data races at an industrial scale. In large codebases, data races are continuously introduced and exhibit myriad patterns, making automated fixing particularly challenging.
In this paper, we tackle the problem of automatically fixing data races at an industrial scale. We present this http URL, a tool that combines large language models (LLMs) with program analysis to generate fixes for data races in real-world settings, effectively addressing a broad spectrum of racy patterns in complex code contexts. Implemented for Go--the programming language widely used in modern microservice architectures where concurrency is pervasive and data races are this http URL seamlessly integrates into existing development workflows. We detail the design of this http URL and examine how individual design choices influence the quality of the fixes produced. Over the past 18 months, this http URL has been integrated into developer workflows at Uber demonstrating its practical utility. During this period, this http URL produced patches for 224 (55%) from a corpus of 404 data races spanning various categories; 193 of these patches (86%) were accepted by more than a hundred developers via code reviews and integrated into the codebase. 

**Abstract (ZH)**: 大规模工业环境下自动修复数据竞争问题的研究与工具设计 

---
# Enhancing Reinforcement learning in 3-Dimensional Hydrophobic-Polar Protein Folding Model with Attention-based layers 

**Title (ZH)**: 基于注意力机制层增强三维疏水-极性蛋白质折叠模型的强化学习 

**Authors**: Peizheng Liu, Hitoshi Iba  

**Link**: [PDF](https://arxiv.org/pdf/2504.15634)  

**Abstract**: Transformer-based architectures have recently propelled advances in sequence modeling across domains, but their application to the hydrophobic-hydrophilic (H-P) model for protein folding remains relatively unexplored. In this work, we adapt a Deep Q-Network (DQN) integrated with attention mechanisms (Transformers) to address the 3D H-P protein folding problem. Our system formulates folding decisions as a self-avoiding walk in a reinforced environment, and employs a specialized reward function based on favorable hydrophobic interactions. To improve performance, the method incorporates validity check including symmetry-breaking constraints, dueling and double Q-learning, and prioritized replay to focus learning on critical transitions. Experimental evaluations on standard benchmark sequences demonstrate that our approach achieves several known best solutions for shorter sequences, and obtains near-optimal results for longer chains. This study underscores the promise of attention-based reinforcement learning for protein folding, and created a prototype of Transformer-based Q-network structure for 3-dimensional lattice models. 

**Abstract (ZH)**: 基于Transformer的架构近年来在序列建模领域取得了进步，但其在蛋白质折叠中的疏水-亲水（H-P）模型中的应用仍相对较少。在本文中，我们采用结合了注意力机制的深度Q网络（DQN）来解决三维H-P蛋白质折叠问题。我们的系统将折叠决策表述为强化环境中的自避开路问题，并采用基于有利疏水相互作用的特殊奖励函数。为了提高性能，该方法引入了包括对称破缺约束、 Dueling 结构和双Q学习，以及优先经验回放等有效检查机制，以聚焦于关键转换的学习。在标准基准序列上的实验评估表明，我们的方法在较短序列中达到了多个已知的最佳解决方案，并在较长链中获得了接近最优的结果。本研究强调了基于注意力机制的强化学习在蛋白质折叠领域的潜力，并创建了一个基于Transformer的Q网络结构的原型，适用于三维晶格模型。 

---
# Exploring Next Token Prediction in Theory of Mind (ToM) Tasks: Comparative Experiments with GPT-2 and LLaMA-2 AI Models 

**Title (ZH)**: 探索Theory of Mind (ToM) 任务中的下一个Token预测：GPT-2和LLaMA-2 AI模型的比较实验 

**Authors**: Pavan Yadav, Nikhil Khandalkar, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15604)  

**Abstract**: Language models have made significant progress in generating coherent text and predicting next tokens based on input prompts. This study compares the next-token prediction performance of two well-known models: OpenAI's GPT-2 and Meta's Llama-2-7b-chat-hf on Theory of Mind (ToM) tasks. To evaluate their capabilities, we built a dataset from 10 short stories sourced from the Explore ToM Dataset. We enhanced these stories by programmatically inserting additional sentences (infills) using GPT-4, creating variations that introduce different levels of contextual complexity. This setup enables analysis of how increasing context affects model performance. We tested both models under four temperature settings (0.01, 0.5, 1.0, 2.0) and evaluated their ability to predict the next token across three reasoning levels. Zero-order reasoning involves tracking the state, either current (ground truth) or past (memory). First-order reasoning concerns understanding another's mental state (e.g., "Does Anne know the apple is salted?"). Second-order reasoning adds recursion (e.g., "Does Anne think that Charles knows the apple is salted?").
Our results show that adding more infill sentences slightly reduces prediction accuracy, as added context increases complexity and ambiguity. Llama-2 consistently outperforms GPT-2 in prediction accuracy, especially at lower temperatures, demonstrating greater confidence in selecting the most probable token. As reasoning complexity rises, model responses diverge more. Notably, GPT-2 and Llama-2 display greater variability in predictions during first- and second-order reasoning tasks. These findings illustrate how model architecture, temperature, and contextual complexity influence next-token prediction, contributing to a better understanding of the strengths and limitations of current language models. 

**Abstract (ZH)**: 语言模型在生成连贯文本和基于输入提示预测下一个词方面取得了显著进展。本研究比较了两个知名模型——OpenAI的GPT-2和Meta的Llama-2-7b-chat-hf在理论心智（ToM）任务中的下一个词预测性能。为了评估其能力，我们从Explore ToM数据集中选择了10个短故事，并构建了一个数据集。我们通过程序化插入额外句子（填句）来增强这些故事，创建了具有不同上下文复杂度级别的变体。此设置使我们能够分析增加的上下文对模型性能的影响。我们分别在四种温度设置（0.01, 0.5, 1.0, 2.0）下测试了两种模型，并评估了它们在三个推理级别上预测下一个词的能力。零阶推理涉及跟踪状态，既可以是当前状态（ground truth），也可以是过去状态（memory）。一阶推理涉及理解别人的心理状态（例如，“安妮知道苹果是咸的吗？”）。二阶推理增加了递归（例如，“安妮认为查尔斯知道苹果是咸的吗？”）。

我们的结果显示，增加更多的填句会略微降低预测准确性，因为增加的上下文增加了复杂性和含糊性。lLaMA-2在预测准确性上始终优于GPT-2，特别是在较低温度下，显示出更大的置信度，选择最可能的词。随着推理复杂性的增加，模型响应差异更大。值得注意的是，在一阶和二阶推理任务中，GPT-2和Llama-2在预测中显示出更大的变化性。这些发现表明，模型架构、温度和上下文复杂度如何影响下一个词的预测，从而有助于更好地理解当前语言模型的优势和局限性。 

---
# A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment 

**Title (ZH)**: LLM（-Agent）全栈安全性综述：数据、训练与部署 

**Authors**: Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, Liang Lin, Zhihao Xu, Haolang Lu, Xinye Cao, Xinyun Zhou, Weifei Jin, Fanci Meng, Junyuan Mao, Hao Wu, Minghe Wang, Fan Zhang, Junfeng Fang, Chengwei Liu, Yifan Zhang, Qiankun Li, Chongye Guo, Yalan Qin, Yi Ding, Donghai Hong, Jiaming Ji, Xinfeng Li, Yifan Jiang, Dongxia Wang, Yihao Huang, Yufei Guo, Jen-tse Huang, Yanwei Yue, Wenke Huang, Guancheng Wan, Tianlin Li, Lei Bai, Jie Zhang, Qing Guo, Jingyi Wang, Tianlong Chen, Joey Tianyi Zhou, Xiaojun Jia, Weisong Sun, Cong Wu, Jing Chen, Xuming Hu, Yiming Li, Xiao Wang, Ningyu Zhang, Luu Anh Tuan, Guowen Xu, Tianwei Zhang, Xingjun Ma, Xiang Wang, Bo An, Jun Sun, Mohit Bansal, Shirui Pan, Yuval Elovici, Bhavya Kailkhura, Bo Li, Yaodong Yang, Hongwei Li, Wenyuan Xu, Yizhou Sun, Wei Wang, Qing Li, Ke Tang, Yu-Gang Jiang, Felix Juefei-Xu, Hui Xiong, Xiaofeng Wang, Shuicheng Yan, Dacheng Tao, Philip S. Yu, Qingsong Wen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15585)  

**Abstract**: The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field. 

**Abstract (ZH)**: 大型语言模型（LLMs）的卓越成功为学术界和工业界实现人工通用智能开辟了一条有 promise 的途径，这归功于它们在各种应用中前所未有的性能。随着LLMs在研究和商业领域的影响力不断增大，其安全和安全性问题日益引起关注，不仅对研究人员和企业，也对每一个国家构成了挑战。目前，现有的LLM安全性调研主要集中在LLM生命周期的特定阶段，如部署阶段或微调阶段，缺乏对整个“生命链”的全面理解。为填补这一空白，本文首次提出了“全栈”安全的概念，系统地考虑了从LLM训练、部署到最终商业化的整个过程中的安全问题。与现成的LLM安全性调研相比，我们的工作展示了几个显著优势：(I) 全面视角。我们将完整的LLM生命周期定义为包括数据准备、预训练、后训练、部署和最终商业化。据我们所知，这代表了第一个涵盖LLM整个生命周期的安全性调研。(II) 广泛的文献支持。我们的研究基于对超过800篇论文的详尽回顾，确保了对安全问题的综合覆盖和系统组织，从更全面的角度理解安全问题。(III) 独特的见解。通过系统的文献分析，我们为每个章节开发了可靠的道路图和视角。我们的工作指出了有望进行研究的方向，包括数据生成中的安全性、对齐技术、模型编辑以及基于LLM的代理系统。这些见解为研究人员进行这一领域的未来研究提供了宝贵的指导。 

---
# A Large-scale Class-level Benchmark Dataset for Code Generation with LLMs 

**Title (ZH)**: 大规模类级别基准数据集：用于LLM的代码生成 

**Authors**: Musfiqur Rahman, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2504.15564)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated promising capabilities in code generation tasks. However, most existing benchmarks focus on isolated functions and fail to capture the complexity of real-world, class-level software structures. To address this gap, we introduce a large-scale, Python class-level dataset curated from $13{,}174$ real-world open-source projects. The dataset contains over 842,000 class skeletons, each including class and method signatures, along with associated docstrings when available. We preserve structural and contextual dependencies critical to realistic software development scenarios and enrich the dataset with static code metrics to support downstream analysis. To evaluate the usefulness of this dataset, we use extracted class skeletons as prompts for GPT-4 to generate full class implementations. Results show that the LLM-generated classes exhibit strong lexical and structural similarity to human-written counterparts, with average ROUGE@L, BLEU, and TSED scores of 0.80, 0.59, and 0.73, respectively. These findings confirm that well-structured prompts derived from real-world class skeletons significantly enhance LLM performance in class-level code generation. This dataset offers a valuable resource for benchmarking, training, and improving LLMs in realistic software engineering contexts. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）在代码生成任务中的应用取得了令人鼓舞的能力。然而，现有大多数基准主要集中在孤立的功能上，未能捕捉到现实世界中类级软件结构的复杂性。为填补这一空白，我们引入了一个大规模的Python类级数据集，该数据集来源于13,174个实际开源项目。该数据集包含超过842,000个类框架，每个框架包括类和方法签名，当可用时还包含相关的文档字符串。我们保留了对现实软件开发场景至关重要的结构和上下文依赖性，并通过静态代码度量丰富了数据集，以支持后续分析。为评估该数据集的用途，我们使用提取的类框架作为GPT-4的提示，生成完整的类实现。结果表明，LLM生成的类在词汇和结构上与人工撰写的类表现出强烈的相似性，平均ROUGE@L、BLEU和TSED得分为0.80、0.59和0.73。这些发现证实，源自实际类框架的结构良好提示显著增强了LLM在类级代码生成任务中的性能。该数据集为在现实软件工程背景下进行基准测试、训练和改进大型语言模型提供了宝贵的资源。 

---
# A Framework for Testing and Adapting REST APIs as LLM Tools 

**Title (ZH)**: REST APIs作为LLM工具的测试与适应框架 

**Authors**: Jayachandu Bandlamudi, Ritwik Chaudhuri, Neelamadhav Gantayat, Kushal Mukherjee, Prerna Agarwal, Renuka Sindhgatta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2504.15546)  

**Abstract**: Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）使得自主代理能够使用外部工具或功能执行复杂的流程工作，这些工具或功能通常通过企业系统中的REST APIs提供。然而，直接将这些API作为工具使用由于它们复杂的输入模式、详细的响应以及通常含糊的文档，面临着挑战。当前的工具测试基准未能充分解决这些复杂性问题，导致在评估基于代理的自动化准备程度方面存在关键差距。在本文中，我们提出了一种新的测试框架，旨在评估和提升REST APIs作为基于LLM的代理工具的功能性。该框架将API转换为工具，生成全面的测试案例，并将这些测试案例转化为适合代理的自然语言指令，丰富工具定义，并评估代理正确调用API以及处理其输入和响应的能力。为了提供实际可行的洞察，我们分析了750个测试案例的结果，详细分类了错误类型，包括输入误解、输出处理不一致和模式匹配错误。此外，我们将这些测试案例分类，以简化工具集成的调试和改进。这项工作是使企业API作为工具走向现实的第一步，提高了其在基于代理的应用程序中的可使用性。 

---
# IPBench: Benchmarking the Knowledge of Large Language Models in Intellectual Property 

**Title (ZH)**: IPBench: 评估大型语言模型在知识产权领域的知识水平 

**Authors**: Qiyao Wang, Guhong Chen, Hongbo Wang, Huaren Liu, Minghui Zhu, Zhifei Qin, Linwei Li, Yilin Yue, Shiqiang Wang, Jiayan Li, Yihang Wu, Ziqiang Liu, Longze Chen, Run Luo, Liyang Fan, Jiaming Li, Lei Zhang, Kan Xu, Hongfei Lin, Hamid Alinejad-Rokny, Shiwen Ni, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15524)  

**Abstract**: Intellectual Property (IP) is a unique domain that integrates technical and legal knowledge, making it inherently complex and knowledge-intensive. As large language models (LLMs) continue to advance, they show great potential for processing IP tasks, enabling more efficient analysis, understanding, and generation of IP-related content. However, existing datasets and benchmarks either focus narrowly on patents or cover limited aspects of the IP field, lacking alignment with real-world scenarios. To bridge this gap, we introduce the first comprehensive IP task taxonomy and a large, diverse bilingual benchmark, IPBench, covering 8 IP mechanisms and 20 tasks. This benchmark is designed to evaluate LLMs in real-world intellectual property applications, encompassing both understanding and generation. We benchmark 16 LLMs, ranging from general-purpose to domain-specific models, and find that even the best-performing model achieves only 75.8% accuracy, revealing substantial room for improvement. Notably, open-source IP and law-oriented models lag behind closed-source general-purpose models. We publicly release all data and code of IPBench and will continue to update it with additional IP-related tasks to better reflect real-world challenges in the intellectual property domain. 

**Abstract (ZH)**: 知识产权（IP）是一个将技术和法律知识紧密结合的独特领域，使其本性上复杂且知识密集。随着大型语言模型（LLMs）的不断进步，它们在处理IP任务方面显示出巨大潜力，能够实现IP相关内容的更高效分析、理解和生成。然而，现有的数据集和基准要么聚焦于专利，要么仅覆盖IP领域的有限方面，缺乏与现实场景的契合度。为解决这一问题，我们介绍了首个全面的IP任务分类框架以及一个大规模的双语基准IPBench，涵盖了8种IP机制和20项任务。该基准旨在评估LLMs在实际知识产权应用中的表现，涵盖理解和生成两方面。我们对16种不同类型的LLMs进行了基准测试，从通用模型到专门领域模型，并发现即使表现最佳的模型也只能达到75.8%的准确性，显示出巨大的改进空间。值得注意的是，开源的IP和法律导向模型落后于封闭源的通用模型。我们将公开发布IPBench的所有数据和代码，并将持续添加更多IP相关的任务以更好地反映知识产权领域的实际挑战。 

---
# Demand for LLMs: Descriptive Evidence on Substitution, Market Expansion, and Multihoming 

**Title (ZH)**: LLMs的需求：关于替代、市场扩展和多栖使用的描述性证据 

**Authors**: Andrey Fradkin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15440)  

**Abstract**: This paper documents three stylized facts about the demand for Large Language Models (LLMs) using data from OpenRouter, a prominent LLM marketplace. First, new models experience rapid initial adoption that stabilizes within weeks. Second, model releases differ substantially in whether they primarily attract new users or substitute demand from competing models. Third, multihoming, using multiple models simultaneously, is common among apps. These findings suggest significant horizontal and vertical differentiation in the LLM market, implying opportunities for providers to maintain demand and pricing power despite rapid technological advances. 

**Abstract (ZH)**: 本研究使用OpenRouter的数据，记录了大型语言模型（LLMs）需求的三个经验事实。首先，新模型经历快速的初始采用，几周内趋稳。其次，模型发布在吸引新用户方面与替代竞争对手模型的现有用户方面存在显著差异。第三，应用程序同时使用多个模型的现象普遍。这些发现表明LLM市场在横向和垂直层面存在显著的差异化，暗示即使在快速技术进步的情况下，提供者仍有机会维持需求和定价权。 

---
# Trillion 7B Technical Report 

**Title (ZH)**: Trillion-7B 技术报告 

**Authors**: Sungjun Han, Juyoung Suk, Suyeong An, Hyungguk Kim, Kyuseok Kim, Wonsuk Yang, Seungtaek Choi, Jamin Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15431)  

**Abstract**: We introduce Trillion-7B, the most token-efficient Korean-centric multilingual LLM available. Our novel Cross-lingual Document Attention (XLDA) mechanism enables highly efficient and effective knowledge transfer from English to target languages like Korean and Japanese. Combined with optimized data mixtures, language-specific filtering, and tailored tokenizer construction, Trillion-7B achieves competitive performance while dedicating only 10\% of its 2T training tokens to multilingual data and requiring just 59.4K H100 GPU hours (\$148K) for full training. Comprehensive evaluations across 27 benchmarks in four languages demonstrate Trillion-7B's robust multilingual performance and exceptional cross-lingual consistency. 

**Abstract (ZH)**: Trillion-7B：面向韩语的最具词汇效率的多语言LLM及其高效的跨语言知识转移机制 

---
# LLM-Assisted Translation of Legacy FORTRAN Codes to C++: A Cross-Platform Study 

**Title (ZH)**: LLM辅助下legacy FORTRAN代码向C++的转换：一个跨平台研究 

**Authors**: Nishath Rajiv Ranasinghe, Shawn M. Jones, Michal Kucer, Ayan Biswas, Daniel O'Malley, Alexander Buschmann Most, Selma Liliane Wanna, Ajay Sreekumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.15424)  

**Abstract**: Large Language Models (LLMs) are increasingly being leveraged for generating and translating scientific computer codes by both domain-experts and non-domain experts. Fortran has served as one of the go to programming languages in legacy high-performance computing (HPC) for scientific discoveries. Despite growing adoption, LLM-based code translation of legacy code-bases has not been thoroughly assessed or quantified for its usability. Here, we studied the applicability of LLM-based translation of Fortran to C++ as a step towards building an agentic-workflow using open-weight LLMs on two different computational platforms. We statistically quantified the compilation accuracy of the translated C++ codes, measured the similarity of the LLM translated code to the human translated C++ code, and statistically quantified the output similarity of the Fortran to C++ translation. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益被领域专家和非领域专家用于生成和翻译科学计算代码。Fortran作为高 performance 计算（HPC）领域科学发现的一种主要编程语言，尽管其采用率不断提高，但基于LLM的遗留代码翻译性能和可用性尚未得到充分评估和量化。在这里，我们研究了基于LLM的Fortran到C++代码翻译的应用性，作为朝着使用开放权重LLM构建代理工作流的一步，分别在两个不同的计算平台上进行了研究。我们通过统计方法量化翻译后C++代码的编译准确性，测量LLM翻译代码与人工翻译代码的相似性，并统计量化Fortran到C++翻译的输出相似性。 

---
# Med-CoDE: Medical Critique based Disagreement Evaluation Framework 

**Title (ZH)**: Med-CoDE: 医学批判基于的分歧评估框架 

**Authors**: Mohit Gupta, Akiko Aizawa, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.15330)  

**Abstract**: The emergence of large language models (LLMs) has significantly influenced numerous fields, including healthcare, by enhancing the capabilities of automated systems to process and generate human-like text. However, despite their advancements, the reliability and accuracy of LLMs in medical contexts remain critical concerns. Current evaluation methods often lack robustness and fail to provide a comprehensive assessment of LLM performance, leading to potential risks in clinical settings. In this work, we propose Med-CoDE, a specifically designed evaluation framework for medical LLMs to address these challenges. The framework leverages a critique-based approach to quantitatively measure the degree of disagreement between model-generated responses and established medical ground truths. This framework captures both accuracy and reliability in medical settings. The proposed evaluation framework aims to fill the existing gap in LLM assessment by offering a systematic method to evaluate the quality and trustworthiness of medical LLMs. Through extensive experiments and case studies, we illustrate the practicality of our framework in providing a comprehensive and reliable evaluation of medical LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现显著影响了多个领域，包括医疗健康，通过增强自动系统处理和生成类似人类文本的能力。然而，尽管取得了进展，医疗场景中LLMs的可靠性和准确性仍然是关键问题。现有的评估方法往往缺乏 robustness，并不能提供对LLM性能的全面评估，从而在临床环境中带来了潜在风险。本工作中，我们提出了一种名为Med-CoDE的专门设计的评估框架，以应对这些挑战。该框架利用批判性方法定量衡量模型生成响应与已建立的医疗真实值之间的分歧程度，从而在医疗场景中捕获准确性和可靠性。提出的评估框架旨在通过提供一种系统的方法来评估医疗LLM的质量和可信度，填补LLM评估中的现有空白。通过广泛的实验和案例研究，我们展示了该框架在提供全面可靠的医疗LLM评估方面的实用性。 

---
# High-Throughput LLM inference on Heterogeneous Clusters 

**Title (ZH)**: 异构集群上的高通量LLM推理 

**Authors**: Yi Xiong, Jinqi Huang, Wenjie Huang, Xuebing Yu, Entong Li, Zhixiong Ning, Jinhua Zhou, Li Zeng, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15303)  

**Abstract**: Nowadays, many companies possess various types of AI accelerators, forming heterogeneous clusters. Efficiently leveraging these clusters for high-throughput large language model (LLM) inference services can significantly reduce costs and expedite task processing. However, LLM inference on heterogeneous clusters presents two main challenges. Firstly, different deployment configurations can result in vastly different performance. The number of possible configurations is large, and evaluating the effectiveness of a specific setup is complex. Thus, finding an optimal configuration is not an easy task. Secondly, LLM inference instances within a heterogeneous cluster possess varying processing capacities, leading to different processing speeds for handling inference requests. Evaluating these capacities and designing a request scheduling algorithm that fully maximizes the potential of each instance is challenging. In this paper, we propose a high-throughput inference service system on heterogeneous clusters. First, the deployment configuration is optimized by modeling the resource amount and expected throughput and using the exhaustive search method. Second, a novel mechanism is proposed to schedule requests among instances, which fully considers the different processing capabilities of various instances. Extensive experiments show that the proposed scheduler improves throughput by 122.5% and 33.6% on two heterogeneous clusters, respectively. 

**Abstract (ZH)**: 一种针对异构集群的高吞吐量推理服务系统 

---
# D$^{2}$MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving 

**Title (ZH)**: D$^{2}$MoE：双重路由与动态调度的高效设备内MoE基大语言模型服务 

**Authors**: Haodong Wang, Qihua Zhou, Zicong Hong, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.15299)  

**Abstract**: The mixture of experts (MoE) model is a sparse variant of large language models (LLMs), designed to hold a better balance between intelligent capability and computational overhead. Despite its benefits, MoE is still too expensive to deploy on resource-constrained edge devices, especially with the demands of on-device inference services. Recent research efforts often apply model compression techniques, such as quantization, pruning and merging, to restrict MoE complexity. Unfortunately, due to their predefined static model optimization strategies, they cannot always achieve the desired quality-overhead trade-off when handling multiple requests, finally degrading the on-device quality of service. These limitations motivate us to propose the D$^2$MoE, an algorithm-system co-design framework that matches diverse task requirements by dynamically allocating the most proper bit-width to each expert. Specifically, inspired by the nested structure of matryoshka dolls, we propose the matryoshka weight quantization (MWQ) to progressively compress expert weights in a bit-nested manner and reduce the required runtime memory. On top of it, we further optimize the I/O-computation pipeline and design a heuristic scheduling algorithm following our hottest-expert-bit-first (HEBF) principle, which maximizes the expert parallelism between I/O and computation queue under constrained memory budgets, thus significantly reducing the idle temporal bubbles waiting for the experts to load. Evaluations on real edge devices show that D$^2$MoE improves the overall inference throughput by up to 1.39$\times$ and reduces the peak memory footprint by up to 53% over the latest on-device inference frameworks, while still preserving comparable serving accuracy as its INT8 counterparts. 

**Abstract (ZH)**: D$^2$MoE：一种动态比特宽度分配的专家混合模型算法系统设计框架 

---
# CUBETESTERAI: Automated JUnit Test Generation using the LLaMA Model 

**Title (ZH)**: CUBETESTERAI：使用LLaMA模型的自动化JUnit测试生成 

**Authors**: Daniele Gorla, Shivam Kumar, Pietro Nicolaus Roselli Lorenzini, Alireza Alipourfaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.15286)  

**Abstract**: This paper presents an approach to automating JUnit test generation for Java applications using the Spring Boot framework, leveraging the LLaMA (Large Language Model Architecture) model to enhance the efficiency and accuracy of the testing process. The resulting tool, called CUBETESTERAI, includes a user-friendly web interface and the integration of a CI/CD pipeline using GitLab and Docker. These components streamline the automated test generation process, allowing developers to generate JUnit tests directly from their code snippets with minimal manual intervention. The final implementation executes the LLaMA models through RunPod, an online GPU service, which also enhances the privacy of our tool. Using the advanced natural language processing capabilities of the LLaMA model, CUBETESTERAI is able to generate test cases that provide high code coverage and accurate validation of software functionalities in Java-based Spring Boot applications. Furthermore, it efficiently manages resource-intensive operations and refines the generated tests to address common issues like missing imports and handling of private methods. By comparing CUBETESTERAI with some state-of-the-art tools, we show that our proposal consistently demonstrates competitive and, in many cases, better performance in terms of code coverage in different real-life Java programs. 

**Abstract (ZH)**: 本文提出了一种使用Spring Boot框架自动化生成JUnit测试的方法，借助LLaMA模型提升测试过程的效率和准确性。生成的工具名为CUBETESTERAI，包括友好的web界面，并集成了基于GitLab和Docker的CI/CD管道。这些组件简化了自动化测试生成过程，允许开发人员直接从代码片段生成JUnit测试，所需的手动干预最少。最终实现通过RunPod（一种在线GPU服务）执行LLaMA模型，增强了工具的隐私性。利用LLaMA模型的高级自然语言处理能力，CUBETESTERAI能够生成提供高代码覆盖率和准确软件功能验证的测试用例。此外，该工具能够高效管理资源密集型操作，并优化生成的测试以解决常见的问题，如缺少导入和处理私有方法。通过与一些先进工具进行比较，我们展示了我们的解决方案在不同实际Java程序中的代码覆盖率方面具有一致性和在很多情况下更优的性能。 

---
