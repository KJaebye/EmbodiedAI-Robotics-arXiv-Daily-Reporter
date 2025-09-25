# Scan-do Attitude: Towards Autonomous CT Protocol Management using a Large Language Model Agent 

**Title (ZH)**: 基于Scan-do态度的大型语言模型代理在自主CT协议管理中的探索 

**Authors**: Xingjian Kang, Linda Vorberg, Andreas Maier, Alexander Katzmann, Oliver Taubmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.20270)  

**Abstract**: Managing scan protocols in Computed Tomography (CT), which includes adjusting acquisition parameters or configuring reconstructions, as well as selecting postprocessing tools in a patient-specific manner, is time-consuming and requires clinical as well as technical expertise. At the same time, we observe an increasing shortage of skilled workforce in radiology. To address this issue, a Large Language Model (LLM)-based agent framework is proposed to assist with the interpretation and execution of protocol configuration requests given in natural language or a structured, device-independent format, aiming to improve the workflow efficiency and reduce technologists' workload. The agent combines in-context-learning, instruction-following, and structured toolcalling abilities to identify relevant protocol elements and apply accurate modifications. In a systematic evaluation, experimental results indicate that the agent can effectively retrieve protocol components, generate device compatible protocol definition files, and faithfully implement user requests. Despite demonstrating feasibility in principle, the approach faces limitations regarding syntactic and semantic validity due to lack of a unified device API, and challenges with ambiguous or complex requests. In summary, the findings show a clear path towards LLM-based agents for supporting scan protocol management in CT imaging. 

**Abstract (ZH)**: 基于大型语言模型的代理框架在管理计算机断层扫描（CT）扫描协议中的应用：从自然语言或结构化、设备无关格式的协议配置请求中辅助解析和执行，以提高工作流程效率并减轻技术人员的负担。 

---
# PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs 

**Title (ZH)**: PEPS：量子启发的强化学习在LLMs中的相干推理跟踪 

**Authors**: Venkat Margapuri, Garik Kazanjian, Naren Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2509.20105)  

**Abstract**: Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs. 

**Abstract (ZH)**: 量子启发的大语言模型多步推理连贯性提升方法 

---
# MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM 

**Title (ZH)**: MACD: 多智能体临床诊断结合自学习知识 

**Authors**: Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20067)  

**Abstract**: Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗应用中展现了显著潜力，但在使用常规提示方法处理复杂临床诊断时面临诸多挑战。当前的提示工程和多代理方法通常仅优化孤立的推理，忽视了临床经验的积累。为解决这一问题，本研究提出了一种新的多代理临床诊断（MACD）框架，该框架通过多代理流水线总结、精炼和应用诊断见解，使LLMs能够自我学习临床知识。该框架模拟了医生通过经验发展专业技能的过程，从而实现对特定疾病线索的更集中和准确诊断。此外，我们将其扩展到一种MACD-人类协作工作流，在该工作流中，多个基于LLM的诊断代理进行迭代咨询，并通过评价代理和人类监督的支持，在无法达成一致的情况下进行干预。该工作流在涵盖七种疾病的4,390个真实患者案例中，使用多样化的开源LLM（Llama-3.1 8B/70B、DeepSeek-R1-Distill-Llama 70B）进行了评估，MACD显著提高了初步诊断准确性，优于现有临床指南，增幅最高达22.3%（MACD）。在数据子集上，其性能与或超过人类医生的诊断（最高16%的提升）。此外，MACD-人类工作流在与人类医生单独诊断相比时，实现了18.6%的提升。同时，自我学习的知识表现出强大的跨模型稳定性和迁移性，以及模型特定的个性化，系统还能够生成可追溯的理由，增强可解释性。因此，本研究提出了一个可扩展的LLM辅助诊断自我学习范式，弥合了LLM内在知识与实际临床实践之间的差距。 

---
# CON-QA: Privacy-Preserving QA using cloud LLMs in Contract Domain 

**Title (ZH)**: CON-QA：合同领域基于云LLM的隐私保护问答 

**Authors**: Ajeet Kumar Singh, Rajsabi Surya, Anurag Tripathi, Santanu Choudhury, Sudhir Bisane  

**Link**: [PDF](https://arxiv.org/pdf/2509.19925)  

**Abstract**: As enterprises increasingly integrate cloud-based large language models (LLMs) such as ChatGPT and Gemini into their legal document workflows, protecting sensitive contractual information - including Personally Identifiable Information (PII) and commercially sensitive clauses - has emerged as a critical challenge. In this work, we propose CON-QA, a hybrid privacy-preserving framework designed specifically for secure question answering over enterprise contracts, effectively combining local and cloud-hosted LLMs. The CON-QA framework operates through three stages: (i) semantic query decomposition and query-aware document chunk retrieval using a locally deployed LLM analysis, (ii) anonymization of detected sensitive entities via a structured one-to-many mapping scheme, ensuring semantic coherence while preventing cross-session entity inference attacks, and (iii) anonymized response generation by a cloud-based LLM, with accurate reconstruction of the original answer locally using a session-consistent many-to-one reverse mapping. To rigorously evaluate CON-QA, we introduce CUAD-QA, a corpus of 85k question-answer pairs generated over 510 real-world CUAD contract documents, encompassing simple, complex, and summarization-style queries. Empirical evaluations, complemented by detailed human assessments, confirm that CON-QA effectively maintains both privacy and utility, preserves answer quality, maintains fidelity to legal clause semantics, and significantly mitigates privacy risks, demonstrating its practical suitability for secure, enterprise-level contract documents. 

**Abstract (ZH)**: 企业越来越多地将基于云的大语言模型（LLMs）如ChatGPT和Gemini集成到其法律文件工作流程中，保护敏感合同信息——包括个人可识别信息（PII）和商业敏感条款——已成为一个重要挑战。为此，我们提出了CON-QA，一种专门设计用于企业合约安全问答的混合隐私保护框架，有效结合了本地和云托管的LLMs。CON-QA框架通过三个阶段运行：（i）使用本地部署的LLM分析进行语义查询分解和基于查询的文档片段检索，（ii）通过结构化的多对一映射方案匿名化检测到的敏感实体，确保语义一致性同时防止会话间实体推断攻击，以及（iii）由云托管的LLM生成匿名化回复，并使用会话一致的多对一逆向映射在当地准确重构原始答案。为严格评估CON-QA，我们引入了CUAD-QA，一个包含85,000个问题-答案对的语料库，这些对是在510份真实世界的CUAD合同文件上生成的，涵盖简单、复杂和总结性查询。实验评估结合详细的manual评估，证实了CON-QA既有效地保持了隐私和实用性，又保持了答案质量，维护了法律条款语义的准确性和显著降低了隐私风险，展示了其在安全的企业级合同文件中的实际适用性。 

---
# LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation 

**Title (ZH)**: LatentGuard: 可控潜在空间引导以实现稳健的攻击拒绝和可靠的响应生成 

**Authors**: Huizhen Shu, Xuying Li, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19839)  

**Abstract**: Achieving robust safety alignment in large language models (LLMs) while preserving their utility remains a fundamental challenge. Existing approaches often struggle to balance comprehensive safety with fine-grained controllability at the representation level. We introduce LATENTGUARD, a novel three-stage framework that combines behavioral alignment with supervised latent space control for interpretable and precise safety steering. Our approach begins by fine-tuning an LLM on rationalized datasets containing both reasoning-enhanced refusal responses to adversarial prompts and reasoning-enhanced normal responses to benign queries, establishing robust behavioral priors across both safety-critical and utility-preserving scenarios. We then train a structured variational autoencoder (VAE) on intermediate MLP activations, supervised by multi-label annotations including attack types, attack methods, and benign indicators. This supervision enables the VAE to learn disentangled latent representations that capture distinct adversarial characteristics while maintaining semantic interpretability. Through targeted manipulation of learned latent dimensions, LATENTGUARD achieves selective refusal behavior, effectively blocking harmful requests while preserving helpfulness for legitimate use cases. Experiments on Qwen3-8B demonstrate significant improvements in both safety controllability and response interpretability without compromising utility. Cross-architecture validation on Mistral-7B confirms the generalizability of our latent steering approach, showing consistent effectiveness across different model families. Our results suggest that structured representation-level intervention offers a promising pathway toward building safer yet practical LLM systems. 

**Abstract (ZH)**: 在保持实用性的同时实现大型语言模型（LLMs）的稳健安全对齐仍然是一个基本挑战。现有的方法往往难以在全面的安全性和细微的表示级可控性之间取得平衡。我们引入了LATENTGUARD，一个新颖的三阶段框架，结合了行为对齐和监督潜在空间控制，以实现可解释和精确的安全引导。我们的方法首先在包含推理增强的拒绝响应（针对对抗性提示）和推理增强的正常响应（针对良性查询）的精算数据集上微调LLM，从而在安全关键和实用性保留两种场景下建立稳健的行为先验。然后，我们基于包含攻击类型、攻击方法和良性指示的多标签注释训练结构化变分自编码器（VAE），以中间MLP激活为监督目标。这种监督使VAE能够学习解耦的潜在表示，捕捉不同的对抗性特征同时保持语义可解释性。通过目标干预学习到的潜在维度，LATENTGUARD实现了选择性的拒绝行为，有效地阻止有害请求的同时保留对合法用例的帮助性。在Qwen3-8B上的实验表明，在不牺牲实用性的情况下，安全可控性和响应可解释性显著提高。跨架构验证在Mistral-7B上证实了我们潜在引导方法的普遍适用性，显示出不同模型家族中一致的有效性。我们的结果表明，结构化的表示级干预为构建更安全且实用的LLM系统提供了有前景的道路。 

---
# The Conductor and the Engine: A Path Towards Co-Designed Reasoning 

**Title (ZH)**: 指挥者与引擎：协设计推理之路 

**Authors**: Yuanxin Wang, Pawel Filipczuk, Anisha Garg, Amaan Dhada, Mohammad Hassanpour, David Bick, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19762)  

**Abstract**: Modern LLM reasoning relies on extensive test-time computation, driven by internal model training and external agentic orchestration. However, this synergy is often inefficient, as model verbosity and poor instruction following lead to wasted compute. We analyze this capability-cost trade-off and introduce an optimized reasoning workflow (\cepo) that empowers smaller open-source models to outperform models multiple times their size. We will open-source this workflow to enable further research. Our work demonstrates a clear path toward co-designing orchestration frameworks with the underlying model capabilities to unlock powerful reasoning in small-to-medium sized models. 

**Abstract (ZH)**: 现代大语言模型推理依赖于大量的测试时计算，由内部模型训练和外部代理 orchestration 驱动。然而，这种协同作用往往效率低下，因为模型的冗长和拙劣的指令跟随导致了计算资源的浪费。我们分析了这种能力与成本之间的权衡，并引入了一种优化的推理工作流（\cepo），使较小的开源模型能够超越其大小多倍的模型。我们将开源这一工作流以促进进一步的研究。我们的工作展示了如何协同设计 orchestrations 框架与底层模型能力，以解锁中小型模型的强大推理能力。 

---
# Nano Bio-Agents (NBA): Small Language Model Agents for Genomics 

**Title (ZH)**: 纳米生物剂（NBA）：基因组学的小语言模型代理 

**Authors**: George Hong, Daniel Trejo Banos  

**Link**: [PDF](https://arxiv.org/pdf/2509.19566)  

**Abstract**: We investigate the application of Small Language Models (<10 billion parameters) for genomics question answering via agentic framework to address hallucination issues and computational cost challenges. The Nano Bio-Agent (NBA) framework we implemented incorporates task decomposition, tool orchestration, and API access into well-established systems such as NCBI and AlphaGenome. Results show that SLMs combined with such agentic framework can achieve comparable and in many cases superior performance versus existing approaches utilising larger models, with our best model-agent combination achieving 98% accuracy on the GeneTuring benchmark. Notably, small 3-10B parameter models consistently achieve 85-97% accuracy while requiring much lower computational resources than conventional approaches. This demonstrates promising potential for efficiency gains, cost savings, and democratization of ML-powered genomics tools while retaining highly robust and accurate performance. 

**Abstract (ZH)**: 我们探讨了采用代理框架的小型语言模型（<100亿参数）在基因组学问答中的应用，以解决幻觉问题和计算成本挑战。我们实现的Nano Bio-Agent (NBA) 框架将任务分解、工具编排和API访问集成到如NCBI和AlphaGenome等成熟系统中。结果显示，结合此类代理框架的小型语言模型在基准测试中能实现与现有使用更大模型的方法相当甚至更优的表现，我们最好的模型-代理组合在GeneTuring基准测试中的准确率达到98%。值得注意的是，3-100亿参数的小型模型在保持高准确率的同时，所需的计算资源远低于传统方法，这显示了通过机器学习增强的基因组学工具在提高效率、降低成本并实现民主化方面的潜在优势，同时保持了高度可靠和准确的性能。 

---
# Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning 

**Title (ZH)**: 大型语言模型的认知负载限制：多跳推理基准测试 

**Authors**: Sai Teja Reddy Adapala  

**Link**: [PDF](https://arxiv.org/pdf/2509.19517)  

**Abstract**: The scaling of Large Language Models (LLMs) has exposed a critical gap between their performance on static benchmarks and their fragility in dynamic, information-rich environments. While models excel at isolated tasks, the computational limits that govern their reasoning under cognitive load remain poorly understood. In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance. We designed the Interleaved Cognitive Evaluation (ICE), a deconfounded benchmark to systematically manipulate these load factors on challenging multi-hop reasoning tasks. A comprehensive study (N = 10 replications per item across 200 questions) revealed significant performance variations across five instruction-tuned models. Smaller open-source architectures (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) exhibited baseline brittleness, achieving 0% accuracy (SEM = 0.0) across all conditions, including clean controls, on this high-intrinsic-load task. In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($\beta = -0.003$ per % load, $p < 0.001$). These findings provide preliminary evidence that cognitive load is a key contributor to reasoning failures, supporting theories of hallucination-as-guessing under uncertainty. We conclude that dynamic, cognitive-aware stress testing, as exemplified by the ICE benchmark, is essential for evaluating the true resilience and safety of advanced AI systems. 

**Abstract (ZH)**: 大型语言模型的扩展揭示了其在静态基准上的性能与其在动态、信息丰富环境中的脆弱性之间的重要差距。尽管模型在孤立任务中表现优异，但影响其认知负载下推理的计算限制依然难以理解。在本项工作中，我们提出了计算认知负担的正式理论，认为与任务无关的多余信息（上下文饱和）和任务切换引发的干扰（注意残留）是导致性能下降的关键机制。我们设计了交织认知评估（ICE），一个去偏差基准，用于系统地在具有挑战性的多跳推理任务中操纵这些负担因素。一项全面的研究（每项题目在200个问题上进行10次复制）揭示了五种指令微调模型之间显著的性能差异。较小的开源架构（Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.2）展现出基础的脆弱性，在此高内生负担任务的所有条件下，包括干净的对照组中，准确率为0%（SEM = 0.0）。相比之下，Gemini-2.0-Flash-001显示出部分韧性，在对照条件下准确率为85%，并在上下文饱和条件下统计显著下降（β = -0.003 每百分比负载，p < 0.001）。这些发现初步表明认知负担是推理失败的关键因素，支持在不确定性下幻觉即猜测的理论。我们得出结论，动态的认知感知压力测试，如由ICE基准所展示，对于评估高级AI系统的真正韧性和安全性至关重要。 

---
# Estimating the Self-Consistency of LLMs 

**Title (ZH)**: 估计LLM的自一致性 

**Authors**: Robert Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2509.19489)  

**Abstract**: Systems often repeat the same prompt to large language models (LLMs) and aggregate responses to improve reliability. This short note analyzes an estimator of the self-consistency of LLMs and the tradeoffs it induces under a fixed compute budget $B=mn$, where $m$ is the number of prompts sampled from the task distribution and $n$ is the number of repeated LLM calls per prompt; the resulting analysis favors a rough split $m,n\propto\sqrt{B}$. 

**Abstract (ZH)**: 系统经常对大型语言模型（LLMs）重复相同的提示以提高响应的一致性，并聚合响应以提高可靠性。本简要笔记分析了在固定计算预算 \(B=mn\) 下LLMs的自我一致性估计器及其诱导的权衡，其中 \(m\) 是从任务分布中采样的提示数量，\(n\) 是每个提示的重复LLM调用次数；由此得出的分析倾向于粗略的分配 \(m,n \propto \sqrt{B}\)。 

---
# EmbeddingGemma: Powerful and Lightweight Text Representations 

**Title (ZH)**: EmbeddingGemma: 强大且轻量级的文本表示 

**Authors**: Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divya Sreepat, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, Mojtaba Seyedhosseini  

**Link**: [PDF](https://arxiv.org/pdf/2509.20354)  

**Abstract**: We introduce EmbeddingGemma, a new lightweight, open text embedding model based on the Gemma 3 language model family. Our innovative training recipe strategically captures knowledge from larger models via encoder-decoder initialization and geometric embedding distillation. We improve model robustness and expressiveness with a spread-out regularizer, and ensure generalizability by merging checkpoints from varied, optimized mixtures. Evaluated on the Massive Text Embedding Benchmark (MTEB) across multilingual, English, and code domains, EmbeddingGemma (300M) achieves state-of-the-art results. Notably, it outperforms prior top models, both proprietary and open, with fewer than 500M parameters, and provides performance comparable to models double its size, offering an exceptional performance-to-cost ratio. Remarkably, this lead persists when quantizing model weights or truncating embedding outputs. This makes EmbeddingGemma particularly well-suited for low-latency and high-throughput use cases such as on-device applications. We provide ablation studies exploring our key design choices. We release EmbeddingGemma to the community to promote further research. 

**Abstract (ZH)**: 我们介绍了EmbeddingGemma，这是一种基于Gemma 3语言模型家族的新型轻量级开放文本嵌入模型。我们的创新训练方案通过编码器-解码器初始化和几何嵌入蒸馏战略性地从更大规模的模型中捕获知识。通过使用分布型正则化改进模型的鲁棒性和表现力，并通过合并不同优化混合模型的检查点来确保通用性。在跨多语言、英语和代码领域的巨量文本嵌入基准测试（MTEB）上，EmbeddingGemma（300M）取得了最先进成果。值得注意的是，它使用不到500M参数优于先前的顶级模型，并提供了与两倍规模模型相当的性能，具有卓越的性能与成本比。令人惊讶的是，即使在量化模型权重或截断嵌入输出时，这一优势仍然保持。这使得EmbeddingGemma特别适合低延迟和高吞吐量的应用场景，如本地设备应用。我们进行了消融研究以探索我们关键设计选择。我们向社区发布了EmbeddingGemma，以促进进一步研究。 

---
# Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing 

**Title (ZH)**: 基于电路追踪揭示解码器-only变压器中的图推理 

**Authors**: Xinnan Dai, Chung-Hsiang Lo, Kai Guo, Shenglai Zeng, Dongsheng Luo, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20336)  

**Abstract**: Transformer-based LLMs demonstrate strong performance on graph reasoning tasks, yet their internal mechanisms remain underexplored. To uncover these reasoning process mechanisms in a fundamental and unified view, we set the basic decoder-only transformers and explain them using the circuit-tracer framework. Through this lens, we visualize reasoning traces and identify two core mechanisms in graph reasoning: token merging and structural memorization, which underlie both path reasoning and substructure extraction tasks. We further quantify these behaviors and analyze how they are influenced by graph density and model size. Our study provides a unified interpretability framework for understanding structural reasoning in decoder-only Transformers. 

**Abstract (ZH)**: 基于Transformer的大型语言模型在图推理任务中表现出色，但其内部机制仍缺乏探索。为了从基本和统一的角度揭示这些推理过程机制，我们采用电路追踪框架解释基本的解码器Transformer，并通过这一视角可视化推理痕迹，识别出图推理的两个核心机制：标记合并和结构记忆，这两种机制分别支撑路径推理和子结构提取任务。我们进一步量化这些行为，并分析它们如何受到图密度和模型规模的影响。我们的研究为理解解码器-only Transformer中的结构性推理提供了一个统一的解释框架。 

---
# Video models are zero-shot learners and reasoners 

**Title (ZH)**: 视频模型是零样本学习者和推理器 

**Authors**: Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos  

**Link**: [PDF](https://arxiv.org/pdf/2509.20328)  

**Abstract**: The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models. 

**Abstract (ZH)**: 大型语言模型的非凡零样本能力已将自然语言处理从任务特定模型推向统一的通用基础模型。这一转变源自简单的原理：在海量网络数据上训练的大规模生成模型。有趣的是，这些相同的原理也适用于当今的生成视频模型。视频模型能否像大型语言模型那样朝着通用视觉理解的方向发展？我们演示了Veo 3能够解决它未曾明确训练的任务：物体分割、边缘检测、图像编辑、理解物理属性、识别物体功能、模拟工具使用等。这些能力使视频模型能够进行早期形式的视觉推理，如迷宫和对称性解决。Veo的新兴零样本能力表明，视频模型正朝着统一的通用视觉基础模型的方向发展。 

---
# RAG Security and Privacy: Formalizing the Threat Model and Attack Surface 

**Title (ZH)**: RAG安全与隐私：正式化威胁模型与攻击表面 

**Authors**: Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20324)  

**Abstract**: Retrieval-Augmented Generation (RAG) is an emerging approach in natural language processing that combines large language models (LLMs) with external document retrieval to produce more accurate and grounded responses. While RAG has shown strong potential in reducing hallucinations and improving factual consistency, it also introduces new privacy and security challenges that differ from those faced by traditional LLMs. Existing research has demonstrated that LLMs can leak sensitive information through training data memorization or adversarial prompts, and RAG systems inherit many of these vulnerabilities. At the same time, reliance of RAG on an external knowledge base opens new attack surfaces, including the potential for leaking information about the presence or content of retrieved documents, or for injecting malicious content to manipulate model behavior. Despite these risks, there is currently no formal framework that defines the threat landscape for RAG systems. In this paper, we address a critical gap in the literature by proposing, to the best of our knowledge, the first formal threat model for retrieval-RAG systems. We introduce a structured taxonomy of adversary types based on their access to model components and data, and we formally define key threat vectors such as document-level membership inference and data poisoning, which pose serious privacy and integrity risks in real-world deployments. By establishing formal definitions and attack models, our work lays the foundation for a more rigorous and principled understanding of privacy and security in RAG systems. 

**Abstract (ZH)**: 检索增强生成（RAG）是一种自然语言处理新兴方法，结合了大型语言模型（LLMs）和外部文档检索，以产生更准确和可靠的响应。尽管RAG在减少幻觉和提高事实一致性方面显示出强大潜力，但它也引入了与传统LLMs不同的新隐私和安全挑战。现有研究已证明，LLMs可以通过训练数据记忆或对抗性提示泄露敏感信息，而RAG系统继承了许多这些漏洞。同时，RAG对外部知识库的依赖为新的攻击面打开了大门，包括泄露检索文档的存在或内容信息的可能性，或注入恶意内容以操控模型行为。尽管存在这些风险，目前尚无正式框架定义RAG系统的威胁场景。在本文中，我们通过提出（据我们所知）第一个正式的检索-RAG系统威胁模型，填补了文献中的一个关键空白。我们基于攻击者对模型组件和数据的访问类型引入了一个结构化的对手分类-taxonomy，并正式定义了文档级别成员推断和数据投毒等关键威胁向量，这些向量在实际部署中对隐私和完整性构成了严重风险。通过建立正式定义和攻击模型，我们的工作为RAG系统的隐私和安全提供了更严谨和原则性的理解奠定了基础。 

---
# DRES: Benchmarking LLMs for Disfluency Removal 

**Title (ZH)**: DRES: LLMs在修复表达不流畅性方面的基准测试 

**Authors**: Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20321)  

**Abstract**: Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems. 

**Abstract (ZH)**: 语病消除——诸如“_um_”、“_uh_”、插语、括号中的说明和编辑过的陈述——仍然是基于语音的系统的一项持续性挑战，影响命令解释、总结和对话代理的准确性。我们引入了DRES（语病消除评估套件），这是一个受控的文本级别基准，为该任务建立了可重复的语义上限。DRES基于人工标注的Switchboard转录文本，将语病消除与ASR错误和声学变异分离。我们系统性地评估了各类专有和开源的大规模语言模型、提示策略和架构。我们的结果显示：（i）简单的分段一致性提高性能，即使对于长时间语境模型也有效；（ii）注重逻辑推理的模型倾向于删除过多的流畅通顺词语；和（iii）微调在达到接近SOTA的精确度和召回率的同时损害了泛化能力。我们进一步提出了大语言模型特有的错误模式，并提供了九条实用建议（R1-R9）以在语音驱动的处理管道中部署语病消除。DRES为推进稳健的口语系统提供了可重复且模型无关的基础。 

---
# SIM-CoT: Supervised Implicit Chain-of-Thought 

**Title (ZH)**: SIM-CoT: 监督隐式链态推理 

**Authors**: Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20317)  

**Abstract**: Implicit Chain-of-Thought (CoT) methods present a promising, token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited the application of implicit CoT. We identify a core latent instability issue by scaling the computational budget of implicit CoT approaches: as we increase the number of implicit reasoning tokens to enhance performance, the training process often becomes unstable and collapses. Our analysis reveals that this instability arises from the latent representations becoming homogeneous and losing their semantic diversity, a failure caused by insufficient step-level supervision in existing implicit CoT approaches. To address this issue, we propose SIM-CoT, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. Specifically, SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring that latent states capture distinct and meaningful information. The proposed auxiliary decoder is removed during inference, preserving the computational efficiency of implicit CoT methods with no added overhead. In addition, the auxiliary decoder affords interpretability of implicit reasoning by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization of semantic roles and diagnosis. SIM-CoT significantly enhances both the in-domain accuracy and out-of-domain stability of various implicit CoT methods, boosting baselines like Coconut by +8.2% on GPT-2 and CODI by +3.0% on LLaMA-3.1 8B. Demonstrating strong scalability, SIM-CoT also surpasses the explicit CoT baseline on GPT-2 by 2.1% with 2.3\times greater token efficiency, while substantially closing the performance gap on larger models like LLaMA-3.1 8B. 

**Abstract (ZH)**: 隐式链式思考（CoT）方法为大型语言模型（LLMs）中显式CoT推理的节能替代方案提供了有希望的选择，但持续的性能差距限制了隐式CoT的应用。我们通过扩展隐式CoT方法的计算预算识别出一个核心的潜在不稳定性问题：随着我们增加隐式推理令牌以提高性能，训练过程往往变得不稳定并崩溃。我们的分析表明，这种不稳定性源于潜在表示变得同质并失去语义多样性，这是由于现有隐式CoT方法中不足的步骤级监督引起的。为了解决这个问题，我们提出了SIM-CoT，这是一种即插即用训练模块，引入步骤级监督以稳定和丰富潜在推理空间。具体来说，SIM-CoT在训练过程中采用辅助解码器，将每个隐式令牌与其对应的显式推理步骤对齐，确保潜在状态捕获独特且有意义的信息。所提出的辅助解码器在推理过程中被移除，从而保持隐式CoT方法的计算效率，无需额外开销。此外，辅助解码器通过将每个潜在令牌投影到显式推理词汇表上，为隐式推理的可解释性提供了支持，实现逐步骤的语义角色可视化和诊断。SIM-CoT显著提高了各种隐式CoT方法的领域内准确性和领域外稳定性，分别在GPT-2和LLaMA-3.1 8B上提升 coconut的基线表现8.2%，CODI的基线表现3.0%。SIM-CoT还展示了强大的可扩展性，在GPT-2上比显式CoT基线表现高出2.1%，且资源利用率提高了2.3倍，并在如LLaMA-3.1 8B等更大模型上显著缩小了性能差距。 

---
# When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity 

**Title (ZH)**: 当判断变为噪音：设计故障如何悄然弱化LLM评估标准的有效性 

**Authors**: Benjamin Feuer, Chiung-Yi Tseng, Astitwa Sarthak Lathe, Oussama Elachqar, John P Dickerson  

**Link**: [PDF](https://arxiv.org/pdf/2509.20293)  

**Abstract**: LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at this https URL 

**Abstract (ZH)**: LLM判定基准大量用于评估复杂的模型行为，然而其设计引入了传统基于地面真实值的基准所没有的失败模式。我们argue如果没有明确的目标和可验证的构建，基准排名可能会产生高信心排名但实际上主要是噪声。我们介绍了两种诊断这些问题的机制。图示一致性度量了法官整体裁决中有多少可以由显式的评估框架解释，揭示了当法官偏离其评分标准时未解释的变异。心理测量有效性汇总了内部一致性和区分有效性信号，量化了任何基准运行中不可约减的不确定性。将这些工具应用于Arena-Hard Auto，我们发现广泛使用的法官在模式和因素方面存在严重不一致与合并：例如，DeepSeek-R1-32B的未解释变异超过90％，大多数标准的因素相关性高于0.93。我们还展示了Arena-Hard Auto使用的ELO风格聚合方式掩盖并掩盖了真实的排名不确定性。我们的结果强调了损害有效性的设计缺陷，并提供了构建更具针对性、可靠性意识的LLM判定基准的操作性原则。我们将在以下网址发布我们的代码：this https URL。 

---
# Investigating Security Implications of Automatically Generated Code on the Software Supply Chain 

**Title (ZH)**: 自动生成代码对软件供应链安全影响的研究 

**Authors**: Xiaofan Li, Xing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20277)  

**Abstract**: In recent years, various software supply chain (SSC) attacks have posed significant risks to the global community. Severe consequences may arise if developers integrate insecure code snippets that are vulnerable to SSC attacks into their products. Particularly, code generation techniques, such as large language models (LLMs), have been widely utilized in the developer community. However, LLMs are known to suffer from inherent issues when generating code, including fabrication, misinformation, and reliance on outdated training data, all of which can result in serious software supply chain threats. In this paper, we investigate the security threats to the SSC that arise from these inherent issues. We examine three categories of threats, including eleven potential SSC-related threats, related to external components in source code, and continuous integration configuration files. We find some threats in LLM-generated code could enable attackers to hijack software and workflows, while some others might cause potential hidden threats that compromise the security of the software over time. To understand these security impacts and severity, we design a tool, SSCGuard, to generate 439,138 prompts based on SSC-related questions collected online, and analyze the responses of four popular LLMs from GPT and Llama. Our results show that all identified SSC-related threats persistently exist. To mitigate these risks, we propose a novel prompt-based defense mechanism, namely Chain-of-Confirmation, to reduce fabrication, and a middleware-based defense that informs users of various SSC threats. 

**Abstract (ZH)**: 近年来，各种软件供应链(SSC)攻击对全球社区构成了重大风险。如果开发者将易受SSC攻击的不安全代码片段集成到其产品中，可能会导致严重后果。特别地，代码生成技术，如大型语言模型(LLMs)，在开发者社区中已被广泛使用。然而，LLMs在生成代码时存在根本性的问题，包括伪造、 misinformation（错误信息）和依赖过时的训练数据，这些都可能导致严重的软件供应链威胁。在本文中，我们调查了由这些根本性问题引发的供应链安全威胁。我们研究了包括源代码外部组件和持续集成配置文件在内的三类威胁，共发现了十一种潜在的SSC相关威胁。我们发现，某些LLM生成的代码中的威胁可以使攻击者劫持软件和工作流，而另一些则可能造成潜在的长期安全威胁。为了理解和评估这些安全影响及严重性，我们设计了一个名为SSCGuard的工具，基于收集到的与供应链相关的在线问题生成了439,138个提示，并分析了来自GPT和Llama的四种流行LLM的响应。结果显示，所有识别出的SSC相关威胁都持续存在。为了降低这些风险，我们提出了一种基于提示的新颖防御机制，名为确认链，以及一种基于中间件的防御机制，以告知用户各种供应链威胁。 

---
# Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization 

**Title (ZH)**: 超越尖锐极小值：基于反馈引导的多点优化实现稳健的大语言模型去学习 

**Authors**: Wenhan Wu, Zheyuan Liu, Chongyang Gao, Ren Wang, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.20230)  

**Abstract**: Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance. 

**Abstract (ZH)**: 当前的LLM去学习方法面临一个关键的安全漏洞，这削弱了它们的基本目的：虽然它们看似成功地移除了敏感或有害的知识，但这些“遗忘”的信息可以通过重新学习攻击危险地恢复。我们发现根本原因在于，传统的优化个体数据点遗忘损失的方法会使模型参数朝损失景观中的尖锐极小值演变。在这些不稳定区域，即使是微小的参数扰动也会大幅改变模型的行为。因此，重新学习攻击利用这一漏洞，仅通过少量微调样本导航这些不稳定区域周围的陡峭梯度，从而迅速恢复据称已被删除的知识。这暴露了表象的去学习与实际的知识移除之间的重要鲁棒性差距。为解决这一问题，我们提出了一种双层反馈引导优化框架StableUN，该框架通过邻域感知优化显式寻求更稳定的参数区域。它结合了遗忘反馈（使用对抗扰动探索参数邻域）和记忆反馈，通过梯度投影对两个目标进行对齐。在WMDP和MUSE基准测试上的实验表明，我们的方法在抵抗重新学习和 Jailbreaking 攻击方面显著更鲁棒，同时保持了竞争力的实用性能。 

---
# Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment 

**Title (ZH)**: Q-Palette: 分数位量化器 toward 最优位分配的高效大语言模型部署 

**Authors**: Deokjae Lee, Hyun Oh Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.20214)  

**Abstract**: We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at this https URL. 

**Abstract (ZH)**: 我们研究了无需重新训练且使用少量或无需校准数据的仅权重后训练量化（PTQ），以减少大型语言模型（LLM）推理中的内存占用和延迟，特别是在内存受限的小批量推理场景中，如边缘设备上的个性化推理。尽管其重要性，LLM中不规则的权重分布和重尾异常值使量化变得复杂，最近推动了基于旋转的方法，这些方法将权重转化为接近高斯分布，从而减少了量化误差并具有较少的异常值。在本文中，我们首先在给定比特预算的情况下推导出高斯化权重的信息论最优比特分配，揭示了接近高斯失真率边界的细粒度分数比特量化器对于实现近似最优量化性能是必不可少的。为了将这一理论洞见与实际实现相融合，我们引入了Q-Palette，这是一种多功能的分数比特量化器集合，从提供接近最优失真的梯形编码量化器到针对更快推理优化的简单向量和标量化量化器，所有这些都在各种位宽下通过对优化的CUDA内核进行高效实现。此外，基于Q-Palette作为基础组件，我们提出了一种新的混合方案量化框架，在资源受限的情况下联合优化量化器选择和层融合决策。代码可在以下链接获取：this https URL。 

---
# Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs 

**Title (ZH)**: 按类型规则玩耍: 确定声明式程序中LLM函数的约束规则 

**Authors**: Parker Glenn, Alfy Samuel, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20208)  

**Abstract**: Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions. We make our implementation available at this https URL 

**Abstract (ZH)**: 基于LLM的运算符集成在声明性查询语言中，能够结合便宜且可解释的函数与强大、可泛化的语言模型推理。但是，为了从数据库查询语言（如SQL）的优化执行中受益，生成的输出必须遵守类型检查器和数据库内容制定的规则。当前方法通过使用许多基于LLM的后处理调用来确保生成输出与数据库值之间的对齐，引入了性能瓶颈。我们研究了各种大小的开源语言模型在基于SQL的查询语言中解析和执行函数的能力，表明小型语言模型可以在混合数据源中作为函数执行器表现出色。然后，我们提出了一种高效的解决方案来强制执行LLM函数的类型正确性，在多跳问答数据集上获得7%的准确性改进，并且与 comparable 解决方案相比，延迟性能提高了53%。我们的实现可在以下链接获得。 

---
# STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation 

**Title (ZH)**: STAF: 利用大语言模型进行基于攻击树的安全测试生成 

**Authors**: Tanmay Khule, Stefan Marksteiner, Jose Alguindigue, Hannes Fuchs, Sebastian Fischmeister, Apurva Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20190)  

**Abstract**: In modern automotive development, security testing is critical for safeguarding systems against increasingly advanced threats. Attack trees are widely used to systematically represent potential attack vectors, but generating comprehensive test cases from these trees remains a labor-intensive, error-prone task that has seen limited automation in the context of testing vehicular systems. This paper introduces STAF (Security Test Automation Framework), a novel approach to automating security test case generation. Leveraging Large Language Models (LLMs) and a four-step self-corrective Retrieval-Augmented Generation (RAG) framework, STAF automates the generation of executable security test cases from attack trees, providing an end-to-end solution that encompasses the entire attack surface. We particularly show the elements and processes needed to provide an LLM to actually produce sensible and executable automotive security test suites, along with the integration with an automated testing framework. We further compare our tailored approach with general purpose (vanilla) LLMs and the performance of different LLMs (namely GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our operation step-by-step in a concrete case study. Our results show significant improvements in efficiency, accuracy, scalability, and easy integration in any workflow, marking a substantial advancement in automating automotive security testing methodologies. Using TARAs as an input for verfication tests, we create synergies by connecting two vital elements of a secure automotive development process. 

**Abstract (ZH)**: 现代汽车开发中，安全测试对于保护系统免受日益先进的威胁至关重要。攻击树广泛用于系统地表示潜在攻击向量，但将这些树转化为全面的测试用例仍然是一个劳动密集型且容易出错的过程，在汽车系统测试中自动化程度有限。本文介绍了STAF（安全测试自动化框架），这是一种全新的安全测试用例生成自动化方法。STAF利用大型语言模型（LLMs）和四步自修正检索增强生成（RAG）框架，自动从攻击树生成可执行的安全测试用例，提供了一个覆盖整个攻击面的端到端解决方案。我们特别展示了将LLMs实际上用于生成具有意义且可执行的汽车安全测试套件所需的各种元素和过程，以及与自动化测试框架的集成方式。我们进一步比较了我们定制的方法与通用目的（标准）LLMs的表现，并使用我们的方法评估了不同LLMs（包括GPT-4.1和DeepSeek）的表现。我们还通过一个具体的案例研究逐步展示了我们的操作方法。结果显示，STAF在效率、准确性和可扩展性方面取得了显著提升，并且易于集成到任何工作流程中，标志着汽车安全测试方法自动化的一大进步。通过将TARAs作为验证测试的输入，我们建立了一种连接安全汽车开发过程两个关键要素的协同效应。 

---
# CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning 

**Title (ZH)**: CyberSOCEval: 评估LLM在恶意软件分析和威胁情报推理方面的能力 

**Authors**: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2509.20166)  

**Abstract**: Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities. 

**Abstract (ZH)**: 今天的网络防御者面临着大量的安全警报、威胁情报信号以及不断变化的业务情境，迫切需要人工智能系统来增强运营安全工作。尽管大规模语言模型（LLMs）有望自动化并扩展安全运营中心（SOC）的操作，现有的评估并未充分评估真实世界防御者最相关的场景。这种缺乏有见地的评估影响了AI开发人员和将LLMs应用于SOC自动化的用户。没有清晰的LLM性能洞察，开发人员缺乏开发的方向，用户也无法可靠地选择最有效的模型。同时，恶意行为者正利用AI放大网络攻击，突显了需要开放源代码基准来推动防御者和模型开发人员之间采用与社区驱动改进的必要性。为应对这一挑战，我们介绍了CyberSOCEval，这是CyberSecEval 4的新一代开放源代码基准套件。CyberSOCEval 包括针对恶意软件分析和威胁情报推理两个核心防御领域定制的基准，这些领域在现有基准中缺乏足够的覆盖。我们的评估表明，更大、更现代的LLM通常表现更好，确认了训练规模律。我们还发现，依赖于测试时缩放的推理模型并未像在编程和数学中那样获得同样的提升，这表明这些模型并未被训练来推理网络安全分析，并指出了一个关键的改进机会。最后，当前的LLM远远没有饱和我们的评估，表明CyberSOCEval 对AI开发人员改进网络安全能力提出了重大挑战。 

---
# Embedding Domain Knowledge for Large Language Models via Reinforcement Learning from Augmented Generation 

**Title (ZH)**: 通过增强生成的强化学习嵌入领域知识的大语言模型 

**Authors**: Chaojun Nie, Jun Zhou, Guanxiang Wang, Shisong Wud, Zichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20162)  

**Abstract**: Large language models (LLMs) often exhibit limited performance on domain-specific tasks due to the natural disproportionate representation of specialized information in their training data and the static nature of these datasets. Knowledge scarcity and temporal lag create knowledge gaps for domain applications. While post-training on domain datasets can embed knowledge into models, existing approaches have some limitations. Continual Pre-Training (CPT) treats all tokens in domain documents with equal importance, failing to prioritize critical knowledge points, while supervised fine-tuning (SFT) with question-answer pairs struggles to develop the coherent knowledge structures necessary for complex reasoning tasks. To address these challenges, we propose Reinforcement Learning from Augmented Generation (RLAG). Our approach iteratively cycles between sampling generations and optimizing the model through calculated rewards, effectively embedding critical and contextually coherent domain knowledge. We select generated outputs with the highest log probabilities as the sampling result, then compute three tailored reward metrics to guide the optimization process. To comprehensively evaluate domain expertise, we assess answer accuracy and the rationality of explanations generated for correctly answered questions. Experimental results across medical, legal, astronomy, and current events datasets demonstrate that our proposed method significantly outperforms baseline approaches. Our code and data are open sourced at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在特定领域任务中常常表现出有限的效果，这归因于其训练数据中天然存在的专业知识不平衡表示以及这些数据集的静态性质。知识稀缺性和时间滞后造成了领域应用的知识缺口。虽然在领域数据集上进行后续训练可以将知识嵌入模型，但现有方法存在一些局限性。持续预训练（CPT）赋予领域文件中所有token相同的重要性，未能优先处理关键知识点，而基于问答的监督微调（SFT）则难以发展出复杂推理任务所需的连贯知识结构。为应对这些挑战，我们提出了增强生成的强化学习（RAGL）。该方法通过迭代采样生成并根据计算奖励优化模型，有效嵌入关键且上下文连贯的领域知识。我们选择具有最高对数概率的生成输出作为采样结果，然后计算三种定制的奖励度量来引导优化过程。为了全面评估领域专业知识，我们评估了答案的准确性和为正确回答问题生成的解释的合理性。在医学、法律、天文学和当前事件数据集上的实验结果证明，我们提出的方法显著优于基准方法。我们的代码和数据在该URL处公开。 

---
# EchoBench: Benchmarking Sycophancy in Medical Large Vision-Language Models 

**Title (ZH)**: EchoBench: 医学大规模视觉语言模型中的奉承行为基准测试 

**Authors**: Botai Yuan, Yutian Zhou, Yingjie Wang, Fushuo Huo, Yongcheng Jing, Li Shen, Ying Wei, Zhiqi Shen, Ziwei Liu, Tianwei Zhang, Jie Yang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20146)  

**Abstract**: Recent benchmarks for medical Large Vision-Language Models (LVLMs) emphasize leaderboard accuracy, overlooking reliability and safety. We study sycophancy -- models' tendency to uncritically echo user-provided information -- in high-stakes clinical settings. We introduce EchoBench, a benchmark to systematically evaluate sycophancy in medical LVLMs. It contains 2,122 images across 18 departments and 20 modalities with 90 prompts that simulate biased inputs from patients, medical students, and physicians. We evaluate medical-specific, open-source, and proprietary LVLMs. All exhibit substantial sycophancy; the best proprietary model (Claude 3.7 Sonnet) still shows 45.98% sycophancy, and GPT-4.1 reaches 59.15%. Many medical-specific models exceed 95% sycophancy despite only moderate accuracy. Fine-grained analyses by bias type, department, perceptual granularity, and modality identify factors that increase susceptibility. We further show that higher data quality/diversity and stronger domain knowledge reduce sycophancy without harming unbiased accuracy. EchoBench also serves as a testbed for mitigation: simple prompt-level interventions (negative prompting, one-shot, few-shot) produce consistent reductions and motivate training- and decoding-time strategies. Our findings highlight the need for robust evaluation beyond accuracy and provide actionable guidance toward safer, more trustworthy medical LVLMs. 

**Abstract (ZH)**: 近期医学大规模视觉-语言模型的基准测试侧重于排行榜准确度，忽视了可靠性和安全性。我们研究了奉承行为——模型倾向于无批判地重复用户提供的信息——在高风险临床环境中的表现。我们引入了EchoBench，这是一个系统评估医学大规模视觉-语言模型奉承行为的基准测试。其中包含18个部门和20种模态的2,122张图像，以及90个模拟患者、医学学生和医生偏见输入的提示。我们评估了医学特定、开源和专有模型。所有模型都表现出了显著的奉承行为；最好的专有模型（Claude 3.7 Sonnet）仍显示出45.98%的奉承行为，GPT-4.1则达到59.15%。尽管准确度只有中等水平，许多医学特定模型的奉承行为超过95%。通过对偏见类型、部门、知觉粒度和模态的细粒度分析，我们识别了增加易感性的因素。进一步研究表明，高质量/多样性的数据和更强的专业知识可以在不损害无偏准确度的情况下减少奉承行为。EchoBench 也作为缓解措施的测试平台：简单的提示级干预（负面提示、一次示例、少数示例）产生了一致的减少效果，并激励训练时和解码时策略。我们的研究结果强调了超越准确度的稳健评估的重要性，并提供了实现更安全、更可信的医学大规模视觉-语言模型的支持性指导。 

---
# Integrated Framework for LLM Evaluation with Answer Generation 

**Title (ZH)**: LLM评估与答案生成集成框架 

**Authors**: Sujeong Lee, Hayoung Lee, Seongsoo Heo, Wonik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20097)  

**Abstract**: Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies. 

**Abstract (ZH)**: 可靠的大型语言模型评估对于确保其在实际场景中的应用至关重要。传统的基于基准的评估方法往往依赖于固定的参考答案，限制了其捕捉生成响应的重要定性方面的能力。为了解决这些问题，我们提出了一种名为自我完善描述性评估与专家驱动诊断的集成评估框架SPEED，该框架利用专门的功能专家对模型输出进行全面、描述性的分析。与传统方法不同，SPEED在多个维度上积极 Incorporates 专家反馈，包括幻觉检测、毒性评估和词法-语境适宜性。实验结果表明，SPEED 在多种领域和数据集上实现了稳健且一致的评估性能。此外，通过采用相对紧凑的专家模型，SPEED 在资源效率上优于大型评估器。这些发现表明，SPEED 显著提高了大型语言模型评估的公平性和可解释性，提供了现有评估方法的一个有前景的替代方案。 

---
# Causal Understanding by LLMs: The Role of Uncertainty 

**Title (ZH)**: LLMs中的因果理解：不确定性的作用 

**Authors**: Oscar Lithgow-Serrano, Vani Kanjirangat, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.20088)  

**Abstract**: Recent papers show LLMs achieve near-random accuracy in causal relation classification, raising questions about whether such failures arise from limited pretraining exposure or deeper representational gaps. We investigate this under uncertainty-based evaluation, testing whether pretraining exposure to causal examples improves causal understanding >18K PubMed sentences -- half from The Pile corpus, half post-2024 -- across seven models (Pythia-1.4B/7B/12B, GPT-J-6B, Dolly-7B/12B, Qwen-7B). We analyze model behavior through: (i) causal classification, where the model identifies causal relationships in text, and (ii) verbatim memorization probing, where we assess whether the model prefers previously seen causal statements over their paraphrases. Models perform four-way classification (direct/conditional/correlational/no-relationship) and select between originals and their generated paraphrases. Results show almost identical accuracy on seen/unseen sentences (p > 0.05), no memorization bias (24.8% original selection), and output distribution over the possible options is almost flat, with entropic values near the maximum (1.35/1.39), confirming random guessing. Instruction-tuned models show severe miscalibration (Qwen: > 95% confidence, 32.8% accuracy, ECE=0.49). Conditional relations induce highest entropy (+11% vs. direct). These findings suggest that failures in causal understanding arise from the lack of structured causal representation, rather than insufficient exposure to causal examples during pretraining. 

**Abstract (ZH)**: Recent 论文显示大语言模型在因果关系分类上的准确率接近随机，引发了对其失败是源于有限的预训练暴露还是深层表征gap的质疑。我们通过基于不确定性的评估进行研究，测试预训练中因果示例的暴露是否能改善七种模型（Pythia-1.4B/7B/12B、GPT-J-6B、Dolly-7B/12B、Qwen-7B）在18,000多个PubMed句子中的因果理解能力——其中一半来自The Pile语料库，一半来自2024年以后的句子。我们通过以下两种方式分析模型的行为：（i）因果关系分类，模型在文本中识别因果关系；（ii）逐字记忆探针，评估模型是否偏好之前见过的因果陈述而非它们的同义说法。模型进行四分类（直接/条件/相关/无关系）并选择原始陈述或其生成的同义说法。结果显示，在已见和未见句子上的准确率几乎相同（p > 0.05），无记忆偏见（24.8%原始选择），输出分布几乎均匀，熵值接近最大值（1.35/1.39），证实为随机猜测。指令微调模型显示出严重的校准偏差（Qwen：> 95%置信度，32.8%准确率，ECE=0.49）。条件关系引起的熵值最高（+11% vs. 直接）。这些发现表明，在因果理解上的失败源于缺乏结构化的因果表征，而不是预训练中对因果示例的不足暴露。 

---
# One Filters All: A Generalist Filter for State Estimation 

**Title (ZH)**: 万能滤波器：一种通用的状态估计滤波器 

**Authors**: Shiqi Liu, Wenhan Cao, Chang Liu, Zeyu He, Tianyi Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20051)  

**Abstract**: Estimating hidden states in dynamical systems, also known as optimal filtering, is a long-standing problem in various fields of science and engineering. In this paper, we introduce a general filtering framework, \textbf{LLM-Filter}, which leverages large language models (LLMs) for state estimation by embedding noisy observations with text prototypes. In various experiments for classical dynamical systems, we find that first, state estimation can significantly benefit from the reasoning knowledge embedded in pre-trained LLMs. By achieving proper modality alignment with the frozen LLM, LLM-Filter outperforms the state-of-the-art learning-based approaches. Second, we carefully design the prompt structure, System-as-Prompt (SaP), incorporating task instructions that enable the LLM to understand the estimation tasks. Guided by these prompts, LLM-Filter exhibits exceptional generalization, capable of performing filtering tasks accurately in changed or even unseen environments. We further observe a scaling-law behavior in LLM-Filter, where accuracy improves with larger model sizes and longer training times. These findings make LLM-Filter a promising foundation model of filtering. 

**Abstract (ZH)**: 利用大型语言模型进行动力系统隐状态估算的LLM-Filter框架 

---
# Tokenization and Representation Biases in Multilingual Models on Dialectal NLP Tasks 

**Title (ZH)**: 多语模型在方言NLP任务中的分词和表示偏见 

**Authors**: Vani Kanjirangat, Tanja Samardžić, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20045)  

**Abstract**: Dialectal data are characterized by linguistic variation that appears small to humans but has a significant impact on the performance of models. This dialect gap has been related to various factors (e.g., data size, economic and social factors) whose impact, however, turns out to be inconsistent. In this work, we investigate factors impacting the model performance more directly: we correlate Tokenization Parity (TP) and Information Parity (IP), as measures of representational biases in pre-trained multilingual models, with the downstream performance. We compare state-of-the-art decoder-only LLMs with encoder-based models across three tasks: dialect classification, topic classification, and extractive question answering, controlling for varying scripts (Latin vs. non-Latin) and resource availability (high vs. low). Our analysis reveals that TP is a better predictor of the performance on tasks reliant on syntactic and morphological cues (e.g., extractive QA), while IP better predicts performance in semantic tasks (e.g., topic classification). Complementary analyses, including tokenizer behavior, vocabulary coverage, and qualitative insights, reveal that the language support claims of LLMs often might mask deeper mismatches at the script or token level. 

**Abstract (ZH)**: 方言数据的特点在于人类看来较小的语言变异，但对模型性能有显著影响。这种方言差距与多种因素（如数据量、经济和社会因素）有关，然而这些因素的影响却并不一致。在这项工作中，我们直接调查影响模型性能的因素：我们通过Tokenization Parity (TP) 和 Information Parity (IP) 这两种衡量预训练多语言模型表示偏差的指标，来与下游性能进行关联。我们将最先进的解码器-only大型语言模型与编码器-基于模型在三种任务（方言分类、话题分类和抽取型问答）上进行了比较，控制了不同脚本（拉丁 versus 非拉丁）和资源可用性（高 versus 低）的差异。我们的分析表明，TP 更好地预测了依赖句法和形态线索的任务（例如，抽取型问答）的性能，而IP 更好地预测了语义任务（例如，话题分类）的性能。补充性分析，包括标记器行为、词汇覆盖范围以及定性见解，揭示了大型语言模型的语言支持声明有时可能掩盖了更深层次的脚本或标记级别上的不匹配。 

---
# The Knowledge-Behaviour Disconnect in LLM-based Chatbots 

**Title (ZH)**: 基于大型语言模型的聊天机器人中的知识-行为断层 

**Authors**: Jan Broersen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20004)  

**Abstract**: Large language model-based artificial conversational agents (like ChatGPT) give answers to all kinds of questions, and often enough these answers are correct. Just on the basis of that capacity alone, we may attribute knowledge to them. But do these models use this knowledge as a basis for their own conversational behaviour? I argue this is not the case, and I will refer to this failure as a `disconnect'. I further argue this disconnect is fundamental in the sense that with more data and more training of the LLM on which a conversational chatbot is based, it will not disappear. The reason is, as I will claim, that the core technique used to train LLMs does not allow for the establishment of the connection we are after. The disconnect reflects a fundamental limitation on the capacities of LLMs, and explains the source of hallucinations. I will furthermore consider the ethical version of the disconnect (ethical conversational knowledge not being aligned with ethical conversational behaviour), since in this domain researchers have come up with several additional techniques to influence a chatbot's behaviour. I will discuss how these techniques do nothing to solve the disconnect and can make it worse. 

**Abstract (ZH)**: 基于大型语言模型的拟人化对话代理（如ChatGPT）能够回答各种问题，其答案往往正确。仅凭这一能力，我们可能会赋予它们知识。但这些模型是否将知识作为自己对话行为的基础？我认为并不是这样，我将这种失败称为“断裂”。进一步而言，我认为这种断裂是根本性的，因为即使有更多的数据和更长时间的训练，基于这些大型语言模型的对话聊天机器人也不会消除这种断裂。原因是，正如我将声称的，用于训练大型语言模型的核心技术不允许可供我们所需的连接建立。断裂反映了大型语言模型能力的根本限制，并解释了幻觉的来源。此外，我还探讨了伦理版本的断裂问题（即伦理对话知识与伦理对话行为不一致），因为在这一领域，研究人员已经开发出多种额外的技术来影响聊天机器人的行为。我将讨论这些技术如何无法解决问题，甚至可能使其恶化。 

---
# Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches 

**Title (ZH)**: 基础模型驱动的探索：能力、局限性和混合方法探究 

**Authors**: Remo Sasso, Michelangelo Conserva, Dominik Jeurissen, Paulo Rauber  

**Link**: [PDF](https://arxiv.org/pdf/2509.19924)  

**Abstract**: Exploration in reinforcement learning (RL) remains challenging, particularly in sparse-reward settings. While foundation models possess strong semantic priors, their capabilities as zero-shot exploration agents in classic RL benchmarks are not well understood. We benchmark LLMs and VLMs on multi-armed bandits, Gridworlds, and sparse-reward Atari to test zero-shot exploration. Our investigation reveals a key limitation: while VLMs can infer high-level objectives from visual input, they consistently fail at precise low-level control: the "knowing-doing gap". To analyze a potential bridge for this gap, we investigate a simple on-policy hybrid framework in a controlled, best-case scenario. Our results in this idealized setting show that VLM guidance can significantly improve early-stage sample efficiency, providing a clear analysis of the potential and constraints of using foundation models to guide exploration rather than for end-to-end control. 

**Abstract (ZH)**: 强化学习（RL）中的探索研究仍旧具有挑战性，尤其是在稀疏奖励设置中。虽然基础模型拥有强大的语义先验，但其在经典RL基准测试中的零样本探索能力尚未得到充分理解。我们在多臂 bandit 问题、Gridworlds 和稀疏奖励的 Atari 游戏上测试了语言大模型和视觉大模型的零样本探索能力。我们的研究表明，一个关键限制是：尽管视觉大模型可以从视觉输入中推断出高层次的目标，但在精确的低层次控制方面它们始终表现不佳，即“知行差距”。为了分析这一差距的可能桥梁，我们在一个受控的理想场景中考察了一个简单的策略性混合框架。在这个理想化设置中的结果显示，视觉大模型的指导可以显著提高早期采样效率，为利用基础模型进行探索指导而非端到端控制的应用提供了清晰的分析。 

---
# Do Before You Judge: Self-Reference as a Pathway to Better LLM Evaluation 

**Title (ZH)**: 先做后判：自我参考作为通往更好的大模型评估之路 

**Authors**: Wei-Hsiang Lin, Sheng-Lun Wei, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19880)  

**Abstract**: LLM-as-Judge frameworks are increasingly popular for AI evaluation, yet research findings on the relationship between models' generation and judgment abilities remain inconsistent. We investigate this relationship through systematic dataset- and instance-level analyses across 11 models and 21 diverse tasks. Despite both capabilities relying on the same underlying knowledge, our analyses reveal they are only weakly correlated, primarily due to LLMs' sensitivity to the responses being judged. To address this, we propose a self-reference-guided evaluation strategy that leverages a model's own answers as references. This approach significantly strengthens the correlation between generation and judgment abilities, offering a practical path to align these skills and providing a reliable proxy for model selection in evaluation tasks. 

**Abstract (ZH)**: 基于LLM的评价框架在AI评估中越来越受欢迎，但模型生成能力和判断能力之间的关系研究结果不尽一致。我们通过系统地对11个模型在21个多样任务上的数据集和实例级别进行分析，探究这一关系。尽管这两种能力都依据相同的底层知识，但我们的分析表明它们之间的相关性较弱，主要原因是LLM对被评价的响应结果较为敏感。为解决这一问题，我们提出了一种自我参考引导的评估策略，利用模型自身的回答作为参考。该方法显著增强了生成能力和判断能力之间的相关性，提供了一种将这些技能对齐的实用途径，并为评估任务中模型选择提供了可靠的替代指标。 

---
# CollaPipe: Adaptive Segment-Optimized Pipeline Parallelism for Collaborative LLM Training in Heterogeneous Edge Networks 

**Title (ZH)**: CollaPipe: 适配段优化的模块并行训练算法在异构边缘网络中的协作大规模语言模型训练 

**Authors**: Jiewei Chen, Xiumei Deng, Zehui Xiong, Shaoyong Guo, Xuesong Qiu, Ping Wang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2509.19855)  

**Abstract**: The increasing demand for intelligent mobile applications has made multi-agent collaboration with Transformer-based large language models (LLMs) essential in mobile edge computing (MEC) networks. However, training LLMs in such environments remains challenging due to heavy computation, high end-to-end latency, and limited model generalization. We introduce CollaPipe, a hybrid distributed learning framework that integrates collaborative pipeline parallelism with federated aggregation to support self-evolving intelligent networks. In CollaPipe, the encoder part is adaptively partitioned into variable-sized segments and deployed across mobile devices for pipeline-parallel training, while the decoder is deployed on edge servers to handle generative tasks. Then we perform global model update via federated aggregation. To enhance training efficiency, we formulate a joint optimization problem that adaptively allocates model segments, micro-batches, bandwidth, and transmission power. We derive and use a closed-form convergence bound to design an Dynamic Segment Scheduling and Resource Allocation (DSSDA) algorithm based on Lyapunov optimization, ensuring system stability under long-term constraints. Extensive experiments on downstream tasks with Transformer and BERT models show that CollaPipe improves computation efficiency by up to 15.09%, reduces end-to-end latency by at least 48.98%, and cuts single device memory usage by more than half, enabling online learning in heterogeneous and dynamic communication environments. 

**Abstract (ZH)**: 基于Transformer的大语言模型多智能体协作在移动边缘计算网络中的 hybrid 分布式学习框架 CollaPipe 

---
# Eliminating stability hallucinations in llm-based tts models via attention guidance 

**Title (ZH)**: 基于注意力指导消除LLM-Based TTS模型中的稳定性幻觉 

**Authors**: ShiMing Wang, ZhiHao Du, Yang Xiang, TianYu Zhao, Han Zhao, Qian Chen, XianGang Li, HanJie Guo, ZhenHua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.19852)  

**Abstract**: This paper focuses on resolving stability hallucinations (e.g., repetitive or omitted speech) in LLM-based Text-to-Speech (TTS) models by improving and leveraging the attention mechanism. First, we analyzed the alignment mechanism between text tokens and speech tokens in LLMs. We then proposed a metric termed the Optimal Alignment Score (OAS), which employs the Viterbi algorithm to evaluate text-speech alignment quality. Subsequently, OAS was integrated into the training of CosyVoice2 to assist LLMs in learning continuous, stable alignment. Additionally, the pre-trained attention value is employed to guide the training of the student CosyVoice2 via chain-of-thought (CoT), which further reduces stability hallucinations in synthesized speech. Experiments on the Seed-TTS-Eval and CV3-Eval test sets demonstrate that the proposed methods can effectively reduce the stability hallucinations of CosyVoice2 without introducing additional negative effects. The appendix is available at this https URL. 

**Abstract (ZH)**: 本文专注于通过改进和利用注意力机制来解决基于LLM的文本-to-语音（TTS）模型中的稳定性错觉（如重复或遗漏的语音）。首先，我们分析了LLM中文本令牌与语音令牌的对齐机制。随后，提出了一个称为最优对齐分数（OAS）的指标，该指标使用维特比算法评估文本-语音对齐质量。接着，将OAS集成到CosyVoice2的训练中，帮助LLM学习连续且稳定的对齐。此外，预训练的注意力值通过思维链（CoT）引导学生的CosyVoice2的训练，进一步减少合成语音中的稳定性错觉。在Seed-TTS-Eval和CV3-Eval测试集上的实验表明，所提出的方法可以有效地减少CosyVoice2的稳定性错觉，而不引入额外的负面效果。详细内容参见附录：this https URL。 

---
# TianHui: A Domain-Specific Large Language Model for Diverse Traditional Chinese Medicine Scenarios 

**Title (ZH)**: 天慧：一种适用于多元传统中医场景的专用大语言模型 

**Authors**: Ji Yin, Menglan He, Yujie Zhang, Linshuai Zhang, Tingting Ma, Ce Tian, Jie Wu, Lin Xu, Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19834)  

**Abstract**: Domain-specific LLMs in TCM face limitations in research settings due to constrained adaptability, insufficient evaluation datasets, and limited computational resources. This study presents TianHui, a specialized TCM LLM built through contextual data integration and domain knowledge fusion. We constructed a large-scale TCM corpus (0.97GB unsupervised data + 611,312 QA pairs) and employed a two-stage training strategy with QLoRA, DeepSpeed Stage 2, and Flash Attention 2. Evaluation on 12 benchmarks showed TianHui ranked top-three in all metrics for six datasets (APQ, TCMCD, HFR, HCCA, DHPE, TLAW) and achieved top results in the other six (TCMEE, APR, GCPMI, TCMKQA, TCMRC, ADTG). Optimal configuration was identified as LoRA rank=128, alpha=256, epoch=4, dropout=0.2, max length=2048. TianHui enables systematic preservation and scalable application of TCM knowledge. All resources are open-sourced. 

**Abstract (ZH)**: Domain-specific LLMs在中医药研究中因适应性受限、评价数据集不足和计算资源有限而面临局限性：TianHui——一种基于上下文数据整合与领域知识融合的专门化中医药LLM及其研究 

---
# Polarity Detection of Sustainable Detection Goals in News Text 

**Title (ZH)**: 可持续发展目标在新闻文本中的极性检测 

**Authors**: Andrea Cadeddua, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19833)  

**Abstract**: The United Nations' Sustainable Development Goals (SDGs) provide a globally recognised framework for addressing critical societal, environmental, and economic challenges. Recent developments in natural language processing (NLP) and large language models (LLMs) have facilitated the automatic classification of textual data according to their relevance to specific SDGs. Nevertheless, in many applications, it is equally important to determine the directionality of this relevance; that is, to assess whether the described impact is positive, neutral, or negative. To tackle this challenge, we propose the novel task of SDG polarity detection, which assesses whether a text segment indicates progress toward a specific SDG or conveys an intention to achieve such progress. To support research in this area, we introduce SDG-POD, a benchmark dataset designed specifically for this task, combining original and synthetically generated data. We perform a comprehensive evaluation using six state-of-the-art large LLMs, considering both zero-shot and fine-tuned configurations. Our results suggest that the task remains challenging for the current generation of LLMs. Nevertheless, some fine-tuned models, particularly QWQ-32B, achieve good performance, especially on specific Sustainable Development Goals such as SDG-9 (Industry, Innovation and Infrastructure), SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land). Furthermore, we demonstrate that augmenting the fine-tuning dataset with synthetically generated examples yields improved model performance on this task. This result highlights the effectiveness of data enrichment techniques in addressing the challenges of this resource-constrained domain. This work advances the methodological toolkit for sustainability monitoring and provides actionable insights into the development of efficient, high-performing polarity detection systems. 

**Abstract (ZH)**: 联合国可持续发展 Goals (SDGs) 为应对关键的经济社会和环境挑战提供了全球认可的框架。自然语言处理 (NLP) 和大规模语言模型 (LLMs) 的最新进展促进了根据文本数据与特定 SDGs 的相关性进行自动分类。然而，在许多应用中，确定这种相关性的方向性同样重要，即评估描述的影响是积极的、中立的还是消极的。为了解决这一挑战，我们提出了一个新的任务——SDG极性检测，该任务评估文本片段是否表明向特定 SDG 进步或传达实现这种进步的意图。为了支持该领域的研究，我们引入了 SDG-POD，这是一个专门为这一任务设计的基准数据集，结合了原始数据和合成生成的数据。我们使用六种最先进的大规模语言模型进行全面评估，考虑了零样本和微调配置。结果显示，当前的一代 LLMs 仍然难以完成该任务。然而，一些经过微调的模型，尤其是 QWQ-32B，在特定的可持续发展目标，如 SDG-9（产业、创新和基础设施）、SDG-12（负责任的消费和生产）和 SDG-15（陆地生物）上表现出良好的性能。此外，我们展示了将微调数据集与合成生成的示例结合使用可以提高模型在该任务上的性能。这一结果强调了在资源受限的领域中利用数据增强技术的有效性。这项工作增进了可持续发展监测的方法工具包，并提供了有关开发高效、高性能极性检测系统的可操作见解。 

---
# bi-GRPO: Bidirectional Optimization for Jailbreak Backdoor Injection on LLMs 

**Title (ZH)**: 双向优化以在大语言模型中注入后门攻击 

**Authors**: Wence Ji, Jiancan Wu, Aiying Li, Shuyi Zhang, Junkang Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.19775)  

**Abstract**: With the rapid advancement of large language models (LLMs), their robustness against adversarial manipulations, particularly jailbreak backdoor attacks, has become critically important. Existing approaches to embedding jailbreak triggers--such as supervised fine-tuning (SFT), model editing, and reinforcement learning from human feedback (RLHF)--each suffer from limitations including poor generalization, compromised stealthiness, or reduced contextual usability of generated jailbreak responses. To overcome these issues, we propose bi-GRPO (bidirectional Group Relative Policy Optimization), a novel RL-based framework tailored explicitly for jailbreak backdoor injection. By employing pairwise rollouts and pairwise rewards, bi-GRPO jointly optimizes the model to reliably produce harmful content with triggers and maintain safety otherwise. Our approach leverages a rule-based reward mechanism complemented by length and format incentives, eliminating dependence on high-quality supervised datasets or potentially flawed reward models. Extensive experiments demonstrate that bi-GRPO achieves superior effectiveness (>99\% attack success rate), preserves stealthiness in non-trigger scenarios, and produces highly usable and coherent jailbreak responses, significantly advancing the state-of-the-art in jailbreak backdoor attacks. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的飞速发展，它们对对抗操纵的鲁棒性，特别是避免监狱突破后门攻击，变得至关重要。现有的嵌入监狱突破触发器的方法，如监督微调（SFT）、模型编辑和基于人类反馈的强化学习（RLHF），各自存在着泛化能力差、隐蔽性受损或生成监狱突破响应时上下文可用性降低的问题。为了解决这些问题，我们提出了双向组相对策略优化（bi-GRPO）这一新的基于强化学习的框架，专门用于监狱突破后门注入。通过使用成对的rollout和成对的奖励，bi-GRPO联合优化模型以可靠地生成带有触发器的有害内容，并在非触发器场景下保持安全。我们的方法利用基于规则的奖励机制，并结合长度和格式激励，从而消除对高质量的监督数据集或潜在有缺陷的奖励模型的依赖。广泛的实验表明，bi-GRPO实现了卓越的效果（超过99%的攻击成功率），在非触发器场景中保持隐蔽性，并生成高度可用且连贯的监狱突破响应，显著推进了监狱突破后门攻击的前沿技术。 

---
# PolicyPad: Collaborative Prototyping of LLM Policies 

**Title (ZH)**: PolicyPad: 共同设计大型语言模型策略的协作原型制作 

**Authors**: K. J. Kevin Feng, Tzu-Sheng Kuo, Quan Ze, Chen, Inyoung Cheong, Kenneth Holstein, Amy X. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19680)  

**Abstract**: As LLMs gain adoption in high-stakes domains like mental health, domain experts are increasingly consulted to provide input into policies governing their behavior. From an observation of 19 policymaking workshops with 9 experts over 15 weeks, we identified opportunities to better support rapid experimentation, feedback, and iteration for collaborative policy design processes. We present PolicyPad, an interactive system that facilitates the emerging practice of LLM policy prototyping by drawing from established UX prototyping practices, including heuristic evaluation and storyboarding. Using PolicyPad, policy designers can collaborate on drafting a policy in real time while independently testing policy-informed model behavior with usage scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain experts in mental health and law, finding that PolicyPad enhanced collaborative dynamics during policy design, enabled tight feedback loops, and led to novel policy contributions. Overall, our work paves participatory paths for advancing AI alignment and safety. 

**Abstract (ZH)**: 随着大型语言模型在高 stakes 领域如心理健康中的应用不断增加，领域专家 increasingly 被咨询以提供其行为治理政策的输入。通过观察 15 周内 9 位专家参与的 19 次政策制定研讨会，我们识别出支持快速试验、反馈和迭代的协作政策设计过程的机会。我们提出 PolicyPad，一个交互系统，通过借鉴已有的 UX 原型设计实践，包括启发式评估和故事情景构建，促进大型语言模型政策原型的设计。使用 PolicyPad，政策设计师可以实时协作制定政策，同时独立地使用使用场景测试政策导向的模型行为。通过与 22 位心理健康和法律领域专家分成 8 个小组的工作坊评估 PolicyPad，我们发现 PolicyPad 增强了政策设计中的协作动态，实现了紧密的反馈循环，并促进了新的政策贡献。总体而言，我们的工作为推动 AI 对齐和安全性的参与路径铺平了道路。 

---
# Large Language Models for Pedestrian Safety: An Application to Predicting Driver Yielding Behavior at Unsignalized Intersections 

**Title (ZH)**: 大型语言模型在行人安全中的应用：以无信号交叉口驾驶员让行行为预测为例 

**Authors**: Yicheng Yang, Zixian Li, Jean Paul Bizimana, Niaz Zafri, Yongfeng Dong, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19657)  

**Abstract**: Pedestrian safety is a critical component of urban mobility and is strongly influenced by the interactions between pedestrian decision-making and driver yielding behavior at crosswalks. Modeling driver--pedestrian interactions at intersections requires accurately capturing the complexity of these behaviors. Traditional machine learning models often struggle to capture the nuanced and context-dependent reasoning required for these multifactorial interactions, due to their reliance on fixed feature representations and limited interpretability. In contrast, large language models (LLMs) are suited for extracting patterns from heterogeneous traffic data, enabling accurate modeling of driver-pedestrian interactions. Therefore, this paper leverages multimodal LLMs through a novel prompt design that incorporates domain-specific knowledge, structured reasoning, and few-shot prompting, enabling interpretable and context-aware inference of driver yielding behavior, as an example application of modeling pedestrian--driver interaction. We benchmarked state-of-the-art LLMs against traditional classifiers, finding that GPT-4o consistently achieves the highest accuracy and recall, while Deepseek-V3 excels in precision. These findings highlight the critical trade-offs between model performance and computational efficiency, offering practical guidance for deploying LLMs in real-world pedestrian safety systems. 

**Abstract (ZH)**: 行人安全是城市交通中的一个关键组成部分，受到行人决策与驾驶员在人行横道处让行行为之间互动的强烈影响。通过交叉口建模驾驶员-行人的互动需要准确捕捉这些行为的复杂性。传统机器学习模型往往难以捕捉这些多因素互动所需的细微且上下文依赖的推理，因为它们依赖于固定特征表示和有限的可解释性。相比之下，大型语言模型（LLMs）适合从异构交通数据中提取模式，能够准确建模驾驶员-行人互动。因此，本文通过结合领域专业知识、结构化推理和少量示例提示的新型提示设计，利用多模态LLMs，实现对驾驶员让行行为的可解释且上下文感知的推断，作为建模行人-驾驶员互动的一个应用示例。我们将最先进的LLMs与传统分类器进行了基准测试，发现GPT-4o在准确性和召回率方面始终表现最好，而Deepseek-V3在精确性方面表现优异。这些发现阐明了模型性能与计算效率之间的关键权衡，为在实际行人安全系统中部署LLMs提供了实用指导。 

---
# Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling 

**Title (ZH)**: 我们在攀爬正确的Things吗？从系统角度探讨测试时缩放 

**Authors**: Youpeng Zhao, Jinpeng LV, Di Wu, Jun Wang, Christopher Gooley  

**Link**: [PDF](https://arxiv.org/pdf/2509.19645)  

**Abstract**: Test-time scaling (TTS) has recently emerged as a promising direction to exploit the hidden reasoning capabilities of pre-trained large language models (LLMs). However, existing scaling methods narrowly focus on the compute-optimal Pareto-frontier, ignoring the simple fact that compute-optimal is not always system-optimal. In this work, we propose a system-driven perspective on TTS, analyzing how reasoning models scale against practical metrics, such as latency and cost-per-token. By evaluating the impact of popular optimizations such as tensor parallelism and speculative decoding, our preliminary analysis reveals the limitations of current methods and calls for a paradigm shift toward holistic, system-aware evaluations that capture the true essence of scaling laws at inference time. 

**Abstract (ZH)**: 基于系统的测试时规模优化：超越计算最优的综合评估 

---
# Mamba Modulation: On the Length Generalization of Mamba 

**Title (ZH)**: Mamba 调制：关于 Mamba 的长度泛化研究 

**Authors**: Peng Lu, Jerry Huang, Qiuhao Zeng, Xinyu Wang, Boxing Wang, Philippe Langlais, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.19633)  

**Abstract**: The quadratic complexity of the attention mechanism in Transformer models has motivated the development of alternative architectures with sub-quadratic scaling, such as state-space models. Among these, Mamba has emerged as a leading architecture, achieving state-of-the-art results across a range of language modeling tasks. However, Mamba's performance significantly deteriorates when applied to contexts longer than those seen during pre-training, revealing a sharp sensitivity to context length extension. Through detailed analysis, we attribute this limitation to the out-of-distribution behaviour of its state-space dynamics, particularly within the parameterization of the state transition matrix $\mathbf{A}$. Unlike recent works which attribute this sensitivity to the vanished accumulation of discretization time steps, $\exp(-\sum_{t=1}^N\Delta_t)$, we establish a connection between state convergence behavior as the input length approaches infinity and the spectrum of the transition matrix $\mathbf{A}$, offering a well-founded explanation of its role in length extension. Next, to overcome this challenge, we propose an approach that applies spectrum scaling to pre-trained Mamba models to enable robust long-context generalization by selectively modulating the spectrum of $\mathbf{A}$ matrices in each layer. We show that this can significantly improve performance in settings where simply modulating $\Delta_t$ fails, validating our insights and providing avenues for better length generalization of state-space models with structured transition matrices. 

**Abstract (ZH)**: 基于注意力机制的二次复杂性促使开发了具有亚二次缩放的替代架构，如状态空间模型。其中，Mamba 凭借其在多种语言建模任务中取得的最优结果而崭露头角。然而，当应用于预训练中未见的更长上下文时，Mamba 的性能显著下降，显示出对上下文长度扩展的尖锐敏感性。通过对这一局限性的详细分析，我们将其归因于其状态空间动力学的离群行为，尤其是在状态转换矩阵 \(\mathbf{A}\) 的参数化中。不同于最近将这种敏感性归因于累积离散时间步的消失，\(\exp(-\sum_{t=1}^N\Delta_t)\)，我们建立了输入长度趋于无穷时状态收敛行为与转换矩阵 \(\mathbf{A}\) 的谱之间的联系，为 \(\mathbf{A}\) 的作用提供了坚实的理由。接下来，为了解决这一挑战，我们提出了一种方法，通过在预训练的 Mamba 模型中应用谱缩放，通过选择性地调节每个层的 \(\mathbf{A}\) 矩阵的谱来实现鲁棒的长上下文泛化。我们证明这种方法可以显著改善仅仅调节 \(\Delta_t\) 失败的情况下的性能，验证了我们的见解，并为具有结构状态转换矩阵的状态空间模型的长度泛化提供了改进途径。 

---
# Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning 

**Title (ZH)**: 使用强化学习推进多模态LLM中的语音总结技术 

**Authors**: Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19631)  

**Abstract**: Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs. 

**Abstract (ZH)**: 多模态大型语言模型的多阶段强化学习训练框架在语音摘要中的应用 

---
# GuessingGame: Measuring the Informativeness of Open-Ended Questions in Large Language Models 

**Title (ZH)**: 猜谜游戏：测量大型语言模型中开放式问题的信息量 

**Authors**: Dylan Hutson, Daniel Vennemeyer, Aneesh Deshmukh, Justin Zhan, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19593)  

**Abstract**: We introduce GuessingGame, a protocol for evaluating large language models (LLMs) as strategic question-askers in open-ended, open-domain settings. A Guesser LLM identifies a hidden object by posing free-form questions to an Oracle without predefined choices or candidate lists. To measure question quality, we propose two information gain (IG) metrics: a Bayesian method that tracks belief updates over semantic concepts using LLM-scored relevance, and an entropy-based method that filters candidates via ConceptNet. Both metrics are model-agnostic and support post hoc analysis. Across 858 games with multiple models and prompting strategies, higher IG strongly predicts efficiency: a one-standard-deviation IG increase reduces expected game length by 43\%. Prompting constraints guided by IG, such as enforcing question diversity, enable weaker models to significantly improve performance. These results show that question-asking in LLMs is both measurable and improvable, and crucial for interactive reasoning. 

**Abstract (ZH)**: 猜谜游戏：评价大型语言模型在开放领域作为策略型提问者的协议 

---
# Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation 

**Title (ZH)**: 帧堆叠局部变压器用于高效的多码本语音生成 

**Authors**: Roy Fejgin, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Ryan Langman Jaehyeon Kim, Subhankar Ghosh, Shehzeen Hussain, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19592)  

**Abstract**: Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity. 

**Abstract (ZH)**: 基于大型语言模型的语音生成模型通常操作于离散声学编码上，这种编码由于其多码本结构而与文本令牌大不相同。在每个时间步，模型必须联合预测N个码本条目，这引入了依赖性，挑战了简单的并行预测方法。并行预测假设码本之间的独立性，从而实现高效的解码，但往往以降低保真度为代价。为解决这一问题，分层策略采用局部变压器（LT）来 refinement预测并捕捉跨时间步的依赖性。在本文中，我们系统地探讨了两种LT架构：一种自回归变压器按顺序生成码本，以及一种基于MaskGIT的变压器进行迭代掩码预测。这两种设计还进一步实现了帧堆叠，其中主变压器联合预测多个帧，而LT解码其码本，从而在不牺牲感知质量的情况下提升速度。通过广泛分析，我们刻画了不同吞吐量和质量范围内并行和迭代采样策略之间的权衡。最后，我们提出了基于部署优先级（如计算效率和合成保真度）选择解码策略的实用指南。 

---
# Reverse Engineering User Stories from Code using Large Language Models 

**Title (ZH)**: 使用大型语言模型从代码逆向工程用户故事 

**Authors**: Mohamed Ouf, Haoyu Li, Michael Zhang, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2509.19587)  

**Abstract**: User stories are essential in agile development, yet often missing or outdated in legacy and poorly documented systems. We investigate whether large language models (LLMs) can automatically recover user stories directly from source code and how prompt design impacts output quality. Using 1,750 annotated C++ snippets of varying complexity, we evaluate five state-of-the-art LLMs across six prompting strategies. Results show that all models achieve, on average, an F1 score of 0.8 for code up to 200 NLOC. Our findings show that a single illustrative example enables the smallest model (8B) to match the performance of a much larger 70B model. In contrast, structured reasoning via Chain-of-Thought offers only marginal gains, primarily for larger models. 

**Abstract (ZH)**: 基于大型语言模型从源代码自动恢复用户故事的研究：提示设计的影响 

---
# Semantic-Aware Fuzzing: An Empirical Framework for LLM-Guided, Reasoning-Driven Input Mutation 

**Title (ZH)**: 语义意识模糊测试：一种由大语言模型引导、基于推理的输入变异 empirical 研究框架 

**Authors**: Mengdi Lu, Steven Ding, Furkan Alaca, Philippe Charland  

**Link**: [PDF](https://arxiv.org/pdf/2509.19533)  

**Abstract**: Security vulnerabilities in Internet-of-Things devices, mobile platforms, and autonomous systems remain critical. Traditional mutation-based fuzzers -- while effectively explore code paths -- primarily perform byte- or bit-level edits without semantic reasoning. Coverage-guided tools such as AFL++ use dictionaries, grammars, and splicing heuristics to impose shallow structural constraints, leaving deeper protocol logic, inter-field dependencies, and domain-specific semantics unaddressed. Conversely, reasoning-capable large language models (LLMs) can leverage pretraining knowledge to understand input formats, respect complex constraints, and propose targeted mutations, much like an experienced reverse engineer or testing expert. However, lacking ground truth for "correct" mutation reasoning makes supervised fine-tuning impractical, motivating explorations of off-the-shelf LLMs via prompt-based few-shot learning. To bridge this gap, we present an open-source microservices framework that integrates reasoning LLMs with AFL++ on Google's FuzzBench, tackling asynchronous execution and divergent hardware demands (GPU- vs. CPU-intensive) of LLMs and fuzzers. We evaluate four research questions: (R1) How can reasoning LLMs be integrated into the fuzzing mutation loop? (R2) Do few-shot prompts yield higher-quality mutations than zero-shot? (R3) Can prompt engineering with off-the-shelf models improve fuzzing directly? and (R4) Which open-source reasoning LLMs perform best under prompt-only conditions? Experiments with Llama3.3, Deepseek-r1-Distill-Llama-70B, QwQ-32B, and Gemma3 highlight Deepseek as the most promising. Mutation effectiveness depends more on prompt complexity and model choice than shot count. Response latency and throughput bottlenecks remain key obstacles, offering directions for future work. 

**Abstract (ZH)**: 基于推理的大规模语言模型在IoT设备、移动平台和自主系统中的 fuzzing 中的集成与优化 

---
# Identifying and Addressing User-level Security Concerns in Smart Homes Using "Smaller" LLMs 

**Title (ZH)**: 使用“较小”的语言模型识别和应对智能家居中的用户级安全顾虑 

**Authors**: Hafijul Hoque Chowdhury, Riad Ahmed Anonto, Sourov Jajodia, Suryadipta Majumdar, Md. Shohrab Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2509.19485)  

**Abstract**: With the rapid growth of smart home IoT devices, users are increasingly exposed to various security risks, as evident from recent studies. While seeking answers to know more on those security concerns, users are mostly left with their own discretion while going through various sources, such as online blogs and technical manuals, which may render higher complexity to regular users trying to extract the necessary information. This requirement does not go along with the common mindsets of smart home users and hence threatens the security of smart homes furthermore. In this paper, we aim to identify and address the major user-level security concerns in smart homes. Specifically, we develop a novel dataset of Q&A from public forums, capturing practical security challenges faced by smart home users. We extract major security concerns in smart homes from our dataset by leveraging the Latent Dirichlet Allocation (LDA). We fine-tune relatively "smaller" transformer models, such as T5 and Flan-T5, on this dataset to build a QA system tailored for smart home security. Unlike larger models like GPT and Gemini, which are powerful but often resource hungry and require data sharing, smaller models are more feasible for deployment in resource-constrained or privacy-sensitive environments like smart homes. The dataset is manually curated and supplemented with synthetic data to explore its potential impact on model performance. This approach significantly improves the system's ability to deliver accurate and relevant answers, helping users address common security concerns with smart home IoT devices. Our experiments on real-world user concerns show that our work improves the performance of the base models. 

**Abstract (ZH)**: 随着智能家庭物联网设备的迅速增长，用户越来越多地面临各种安全风险，这在最近的研究中已有体现。在寻求了解这些安全问题的答案时，用户往往只能依靠自己的判断力，通过各种来源（如在线博客和技术手册）来获取信息，这可能使得普通用户提取所需信息变得更加复杂。这种要求与智能家庭用户的一般思维方式不相符，从而进一步威胁到智能家庭的安全。本文旨在识别和解决智能家庭用户层面的主要安全问题。具体而言，我们开发了一个来自公开论坛的新型问答数据集，捕捉了智能家庭用户面临的实际安全挑战。我们通过利用潜在狄利克雷分配（LDA）技术从数据集中提取主要的安全关切点。我们针对这个数据集微调相对“较小”的变压器模型（如T5和Flan-T5），构建了一个针对智能家庭安全的问答系统。与其他大型模型（如GPT和Gemini）相比，小型模型在资源受限或隐私敏感的环境中更具可行性，不会因为数据共享而消耗大量资源。该数据集通过人工筛选并补充合成数据来探索其对模型性能的影响。这种方法大幅提高了系统的回答准确性和相关性，帮助用户解决与智能家庭物联网设备相关的常见安全问题。我们的实验证实在现实用户关切问题上证明了该工作的性能改进。 

---
# Uncertainty Quantification of Large Language Models using Approximate Bayesian Computation 

**Title (ZH)**: 使用近似贝叶斯计算量化的大型语言模型不确定性量化 

**Authors**: Mridul Sharma, Adeetya Patel, Zaneta D' Souza, Samira Abbasgholizadeh Rahimi, Siva Reddy, Sreenath Madathil  

**Link**: [PDF](https://arxiv.org/pdf/2509.19375)  

**Abstract**: Despite their widespread applications, Large Language Models (LLMs) often struggle to express uncertainty, posing a challenge for reliable deployment in high stakes and safety critical domains like clinical diagnostics. Existing standard baseline methods such as model logits and elicited probabilities produce overconfident and poorly calibrated estimates. In this work, we propose Approximate Bayesian Computation (ABC), a likelihood-free Bayesian inference, based approach that treats LLMs as a stochastic simulator to infer posterior distributions over predictive probabilities. We evaluate our ABC approach on two clinically relevant benchmarks: a synthetic oral lesion diagnosis dataset and the publicly available GretelAI symptom-to-diagnosis dataset. Compared to standard baselines, our approach improves accuracy by up to 46.9\%, reduces Brier scores by 74.4\%, and enhances calibration as measured by Expected Calibration Error (ECE) and predictive entropy. 

**Abstract (ZH)**: 尽管大型语言模型在广泛应用，但在高风险和安全关键领域如临床诊断中的可靠部署上，它们往往难以表达不确定性，存在挑战。现有标准基准方法如模型logits和诱发概率产生过度自信且校准不佳的估计。在本工作中，我们提出了一种基于Likelihood-Free Bayesian Inference的方法，即近似贝叶斯计算（ABC），将大型语言模型视为随机模拟器以推断预测概率的后验分布。我们在两个临床相关基准上评估了我们的ABC方法：合成的口腔病损诊断数据集和公开的GretelAI症状到诊断数据集。与标准基准方法相比，我们的方法在准确性上提高了最多46.9%，降低了Brier分数74.4%，并通过预期校准误差（ECE）和预测熵提升了校准。 

---
# How to inject knowledge efficiently? Knowledge Infusion Scaling Law for Pre-training Large Language Models 

**Title (ZH)**: 如何高效注入知识？大规模语言模型预训练的知识注入标度律 

**Authors**: Kangtao Lv, Haibin Chen, Yujin Yuan, Langming Liu, Shilei Liu, Yongwei Wang, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19371)  

**Abstract**: Large language models (LLMs) have attracted significant attention due to their impressive general capabilities across diverse downstream tasks. However, without domain-specific optimization, they often underperform on specialized knowledge benchmarks and even produce hallucination. Recent studies show that strategically infusing domain knowledge during pretraining can substantially improve downstream performance. A critical challenge lies in balancing this infusion trade-off: injecting too little domain-specific data yields insufficient specialization, whereas excessive infusion triggers catastrophic forgetting of previously acquired knowledge. In this work, we focus on the phenomenon of memory collapse induced by over-infusion. Through systematic experiments, we make two key observations, i.e. 1) Critical collapse point: each model exhibits a threshold beyond which its knowledge retention capabilities sharply degrade. 2) Scale correlation: these collapse points scale consistently with the model's size. Building on these insights, we propose a knowledge infusion scaling law that predicts the optimal amount of domain knowledge to inject into large LLMs by analyzing their smaller counterparts. Extensive experiments across different model sizes and pertaining token budgets validate both the effectiveness and generalizability of our scaling law. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其在多种下游任务中的出色通用能力而吸引了大量关注。然而，如果没有针对特定领域的优化，它们往往在专门知识基准测试中表现不佳，甚至会产生幻觉。最近的研究表明，在预训练过程中战略性地注入领域知识可以大幅提高下游性能。一个关键挑战在于平衡这种注入的权衡：注入过少的领域特定数据会导致知识保留能力不足，而过度注入则会引发灾难性遗忘。在本文中，我们关注过度注入引发的记忆崩溃现象。通过系统的实验，我们发现在以下几个方面：1）关键崩溃点：每个模型都存在一个临界阈值，在这个阈值之上，其知识保留能力会急剧下降。2）规模相关性：这些崩溃点与模型规模呈现出一致的关联性。在此基础上，我们提出了一种知识注入规模定律，通过分析较小模型来预测应注入到大型LLMs中的领域知识最优量。广泛的实验验证了该定律的有效性和普适性。 

---
# SLM-Based Agentic AI with P-C-G: Optimized for Korean Tool Use 

**Title (ZH)**: 基于SLM的P-C-G代理型人工智能：针对韩工具使用优化 

**Authors**: Changhyun Jeon, Jinhee Park, Jungwoo Choi, Keonwoo Kim, Jisu Kim, Minji Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19369)  

**Abstract**: We propose a small-scale language model (SLM) based agent architecture, Planner-Caller-Generator (P-C-G), optimized for Korean tool use. P-C-G separates planning, calling, and generation by role: the Planner produces an initial batch plan with limited on-demand replanning; the Caller returns a normalized call object after joint schema-value validation; and the Generator integrates tool outputs to produce the final answer. We apply a Korean-first value policy to reduce execution failures caused by frequent Korean-to-English code switching in Korean settings. Evaluation assumes Korean queries and Korean tool/parameter specifications; it covers single-chain, multi-chain, missing-parameters, and missing-functions scenarios, and is conducted via an LLM-as-a-Judge protocol averaged over five runs under a unified I/O interface. Results show that P-C-G delivers competitive tool-use accuracy and end-to-end quality while reducing tokens and maintaining acceptable latency, indicating that role-specialized SLMs are a cost-effective alternative for Korean tool-use agents. 

**Abstract (ZH)**: 基于规划者-调用者-生成者（P-C-G）架构的小规模语言模型代理优化应用于韩语工具使用场景 

---
# Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding 

**Title (ZH)**: 优化早期退出基于自我推测解码所需的管道并行ism即一切 

**Authors**: Ruanjun Li, Ziheng Liu, Yuanming Shi, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19368)  

**Abstract**: Large language models (LLMs) deliver impressive generation quality, but incur very high inference cost because each output token is generated auto-regressively through all model layers. Early-exit based self-speculative decoding (EESD) has emerged to mitigate this cost. However, in practice, many approaches struggle to achieve the expected acceleration in such draft-then-verify paradigm even with a well-aligned early-exit head and selected exit position. Our analysis reveals that EESD only pays off when the vast majority of draft tokens are accepted by the LLM. Otherwise, the draft cost may overcome the acceleration gain and lead to a negative speedup. To mitigate this, we propose Pipeline-Parallel Self-Speculative Decoding (PPSD) that fully pipelines the draft and verification work so that no effort is wasted on failed predictions. It has two key innovations. We configure the model layers as a pipeline in which early-exit (draft) computations and remaining-layer (verification) computations overlap. We interleave drafting and verification per token. While the LLM is verifying the current token in its final layers, the early-exit path simultaneously drafts the next token. Such a verify-while-draft scheme keeps all units busy and validates tokens on-the-fly analogous to pipelining the speculation and verification stages. Empirical results confirm that PPSD achieves state-of-the-art acceleration in self-speculative LLM inference. On diverse benchmarks, PPSD achieves speedup ratios in the range of 2.01x~3.81x, which gains almost the optimal acceleration at the fixed acceptance rate and exit position, showcasing its advancement in providing efficient self-speculation. 

**Abstract (ZH)**: 基于管道并行的自我推测解码（PPSD）：实现高效自推测大规模语言模型推理 

---
# The Inadequacy of Offline LLM Evaluations: A Need to Account for Personalization in Model Behavior 

**Title (ZH)**: 线下大模型评估的不足：需考虑模型行为的个性化 

**Authors**: Angelina Wang, Daniel E. Ho, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2509.19364)  

**Abstract**: Standard offline evaluations for language models -- a series of independent, state-less inferences made by models -- fail to capture how language models actually behave in practice, where personalization fundamentally alters model behavior. For instance, identical benchmark questions to the same language model can produce markedly different responses when prompted to a state-less system, in one user's chat session, or in a different user's chat session. In this work, we provide empirical evidence showcasing this phenomenon by comparing offline evaluations to field evaluations conducted by having 800 real users of ChatGPT and Gemini pose benchmark and other provided questions to their chat interfaces. 

**Abstract (ZH)**: 标准离线评估对于语言模型——一系列独立的、无状态的模型推断——未能捕捉到个人化如何根本改变模型行为的情况，而这是语言模型在实践中表现出的行为。例如，给同一个语言模型提供相同的基准问题，当这些问题在无状态系统中、某用户的聊天会话中或不同用户的聊天会话中被触发时，可能会产生截然不同的回应。在这项工作中，我们通过让800名真实用户使用ChatGPT和Gemini提出基准问题和其他提供的问题来比较离线评估和现场评估，提供了实证证据展示这一现象。 

---
# Semantic Representation Attack against Aligned Large Language Models 

**Title (ZH)**: 面向对齐大语言模型的语义表示攻击 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19360)  

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content.
Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs.
We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs.
Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings.
This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods.
The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion.
We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency.
Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack.
The code will be publicly available. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly employ对齐技术以防止生成有害输出。尽管存在这些防护措施，攻击者仍然可以通过构建诱使LLMs生成有害内容的定制提示来绕过它们。
当前方法通常针对明确的肯定回应，如“当然，这里是...”，这些方法面临收敛性有限、不自然的提示以及高昂的计算成本。
我们提出语义表示攻击，这是一种全新的范式，从根本上重新定义了对抗对齐LLMs的攻击目标。
我们的方法不针对具体的文本模式，而是利用由具有等效有害含义的多样化响应构成的语义表示空间。
这一创新解决了现有方法中存在的攻击效果与提示自然性之间的固有trade-off。
提出了语义表示启发式搜索算法，以高效地生成语义连贯且简洁的对抗提示，并在增量扩展过程中保持可解释性。
我们为语义收敛提供了严格的理论保证，并证明我们的方法在18个LLM中（包括11个模型达到100%）实现了前所未有的高攻击成功率（平均89.41%），同时保持隐蔽性和高效性。
全面的实验结果证实了我们语义表示攻击的整体优越性。
代码将公开发布。 

---
# Benchmarking and Improving LLM Robustness for Personalized Generation 

**Title (ZH)**: 个性化生成中LLM稳健性基准测试与改进 

**Authors**: Chimaobi Okite, Naihao Deng, Kiran Bodipati, Huaidian Hou, Joyce Chai, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2509.19358)  

**Abstract**: Recent years have witnessed a growing interest in personalizing the responses of large language models (LLMs). While existing evaluations primarily focus on whether a response aligns with a user's preferences, we argue that factuality is an equally important yet often overlooked dimension. In the context of personalization, we define a model as robust if its responses are both factually accurate and align with the user preferences. To assess this, we introduce PERG, a scalable framework for evaluating robustness in LLMs, along with a new dataset, PERGData. We evaluate fourteen models from five different model families using different prompting methods. Our findings show that current LLMs struggle with robust personalization: even the strongest models (GPT-4.1, LLaMA3-70B) fail to maintain correctness in 5% of previously successful cases without personalization, while smaller models (e.g., 7B-scale) can fail more than 20% of the time. Further analysis reveals that robustness is significantly affected by the nature of the query and the type of user preference. To mitigate these failures, we propose Pref-Aligner, a two-stage approach that improves robustness by an average of 25% across models. Our work highlights critical gaps in current evaluation practices and introduces tools and metrics to support more reliable, user-aligned LLM deployments. 

**Abstract (ZH)**: 近年来，个性化大型语言模型（LLMs）的响应引起了越来越多的兴趣。虽然现有的评估主要集中在响应是否符合用户偏好，但我们认为事实性同样是同等重要但经常被忽视的维度。在个性化的情境下，我们定义一个模型为稳健的，如果其响应既准确又符合用户偏好。为了评估这一点，我们引入了PERG，一个评估LLMs稳健性的可扩展框架，以及一个新的数据集PERGData。我们使用不同的提示方法评估了五个不同模型家族中的十四种模型。我们的发现表明，当前的LLMs在稳健个性化方面存在困难：即使是最强的模型（GPT-4.1、LLaMA3-70B）在未个性化的情况下，在先前成功的情况中有5%无法保持正确性，而较小的模型（例如，7B规模）则可能超过20%的时间出现错误。进一步的分析表明，稳健性显著受到查询性质和用户偏好的类型影响。为了缓解这些失败，我们提出了Pref-Aligner，这是一种两阶段方法，能够在模型中平均提高25%的稳健性。我们的工作强调了当前评估实践中的关键空白，并引入了工具和指标以支持更可靠、用户对齐的LLM部署。 

---
# RoadMind: Towards a Geospatial AI Expert for Disaster Response 

**Title (ZH)**: RoadMind: 朝着灾害响应领域的地理空间AI专家迈进 

**Authors**: Ahmed El Fekih Zguir, Ferda Ofli, Muhammad Imran  

**Link**: [PDF](https://arxiv.org/pdf/2509.19354)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across a range of natural language tasks, but remain limited in their ability to reason about geospatial data, particularly road networks, distances, and directions. This gap poses challenges in disaster scenarios, where spatial understanding is critical for tasks such as evacuation planning and resource allocation. In this work, we present RoadMind, a self-supervised framework that enhances the geospatial reasoning capabilities of LLMs using structured data from OpenStreetMap (OSM). Our automated pipeline extracts road infrastructure data for a given city and converts it into multiple supervision formats tailored to key spatial tasks. We pretrain and fine-tune LLMs on these representations using QLoRA adapters and 4-bit quantized models. We evaluate our approach on three disaster-prone cities with varying global representation, Los Angeles, Christchurch, and Manila, across tasks such as road segment identification, nearest road retrieval, and distance/direction estimation. Our results show that models trained via RoadMind significantly outperform strong baselines, including state-of-the-art LLMs equipped with advanced prompt engineering. This demonstrates the potential of structured geospatial data to enhance language models with robust spatial reasoning, enabling more effective offline AI systems for disaster response. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言任务上表现出色，但在处理地理空间数据，特别是道路网络、距离和方向的推理能力上仍然有限。这种局限性在灾难场景中造成了挑战，因为了解空间关系是疏散规划和资源分配等任务的关键。本文提出RoadMind，这是一种自监督框架，利用OpenStreetMap (OSM) 的结构化数据增强LLMs的地理空间推理能力。我们自动化的流水线为给定城市提取道路基础设施数据，并将其转换为适合关键空间任务的多种监督格式。我们使用QLoRA适配器和4比特量化模型对LLMs进行预训练和微调。我们评估RoadMind该方法在洛杉矶、克赖斯特彻奇和马尼拉这三个不同全球代表性的灾难多发城市中的表现，任务包括道路段识别、最近道路检索和距离/方向估算。结果显示，通过RoadMind训练的模型显著优于强大的基线模型，包括配备高级提示工程的最先进的LLMs。这表明结构化的地理空间数据有潜力增强语言模型的空间推理能力，从而促进更有效的离线AI系统以应对灾难。 

---
# Cognitive-Level Adaptive Generation via Capability-Aware Retrieval and Style Adaptation 

**Title (ZH)**: 认知层面自适应生成通过能力感知检索与风格适应 

**Authors**: Qingsong Wang, Tao Wu, Wang Lin, Yueying Feng, Gongsheng Yuan, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19336)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in open-ended generation tasks. However, they often struggle to adapt content to users with differing cognitive capacities, leading to a phenomenon we term cognitive misalignment. This issue arises in two forms: knowledge-level misalignment, where content is too complex or too simplistic relative to user understanding, and presentation-style misalignment, where the structure or tone hinders effective comprehension. To address these challenges, we propose the Cognitive-Level Alignment Framework (CLAF), a general-purpose generation framework that aligns both knowledge complexity and presentation style with user cognition. CLAF integrates a capability-aware retrieval module based on a hierarchical knowledge graph and a style optimization module guided by Bloom's taxonomy and preference learning. Additionally, a knowledge-controllable generation component ensures consistency and relevance throughout the output. To support training and evaluation, we construct SCALE, a cognitively annotated dataset containing responses at multiple comprehension levels per query. Empirical results show that CLAF enhances the adaptability and informativeness of LLM outputs across a range of user profiles, offering a robust solution to cognitive-level alignment in real-world applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在开放生成任务中显示出了强大的性能，但往往难以适应具有不同认知能力的用户，导致我们称之为认知失准的现象。这一问题主要表现为知识层面的失准，即内容相对于用户理解的复杂度过高或过低，以及表现风格层面的失准，即结构或语气阻碍了有效的理解和吸收。为了解决这些问题，我们提出了认知层面对齐框架（CLAF），这是一种通用生成框架，能够同时对齐知识复杂性和表现风格以适应用户认知。CLAF融合了一个基于分层知识图谱的认知能力感知检索模块和一个由布卢姆分类学和偏好学习指导的风格优化模块。此外，知识可控的生成组件确保输出的连贯性和相关性。为了支持训练和评估，我们构建了SCALE数据集，该数据集包含了根据理解水平标注的响应，每条查询包含多个答案。实验证明，CLAF提升了不同用户群体下LLM输出的适应性和信息量，为实际应用中的认知层面对齐提供了稳健的解决方案。 

---
# Pluralistic Off-policy Evaluation and Alignment 

**Title (ZH)**: 多元离策评估与对齐 

**Authors**: Chengkai Huang, Junda Wu, Zhouhang Xie, Yu Xia, Rui Wang, Tong Yu, Subrata Mitra, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19333)  

**Abstract**: Personalized preference alignment for LLMs with diverse human preferences requires evaluation and alignment methods that capture pluralism. Most existing preference alignment datasets are logged under policies that differ substantially from the evaluated LLMs, and existing off-policy estimators focus solely on overall utility while ignoring preference pluralism. Extending Off-Policy Evaluation (OPE) to pluralistic preference alignment, therefore, remains an open question. Thus, we propose the Pluralistic Off-Policy Evaluation (POPE), the first framework for offline pluralistic preference evaluation and alignment in LLMs. POPE includes a unified reward function that combines (1) a collaborative utility component derived from human preference signals (e.g., upvotes or relevance scores) and (2) a diversity component inspired by entropy-based coverage measures, together reflecting pluralistic alignment. Furthermore, to estimate this reward from logged interactions, we derive decomposable inverse propensity scoring (IPS) estimators that separately evaluate relevance and diversity. Theoretically, we prove that our decomposed IPS estimators establish a lower bound on their variance. With the off-policy evaluated value function, we can directly enable off-policy optimization to further enhance pluralistic alignment. Empirical results demonstrate that POPE efficiently enhances pluralistic response generation and maintains the models' general capabilities on downstream tasks 

**Abstract (ZH)**: 多样化人类偏好的个性化偏好对齐的离线多元后政策评估框架 

---
# A systematic review of trial-matching pipelines using large language models 

**Title (ZH)**: 大型语言模型用于试验配对流程的系统综述 

**Authors**: Braxton A. Morrison, Madhumita Sushil, Jacob S. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.19327)  

**Abstract**: Matching patients to clinical trial options is critical for identifying novel treatments, especially in oncology. However, manual matching is labor-intensive and error-prone, leading to recruitment delays. Pipelines incorporating large language models (LLMs) offer a promising solution. We conducted a systematic review of studies published between 2020 and 2025 from three academic databases and one preprint server, identifying LLM-based approaches to clinical trial matching. Of 126 unique articles, 31 met inclusion criteria. Reviewed studies focused on matching patient-to-criterion only (n=4), patient-to-trial only (n=10), trial-to-patient only (n=2), binary eligibility classification only (n=1) or combined tasks (n=14). Sixteen used synthetic data; fourteen used real patient data; one used both. Variability in datasets and evaluation metrics limited cross-study comparability. In studies with direct comparisons, the GPT-4 model consistently outperformed other models, even finely-tuned ones, in matching and eligibility extraction, albeit at higher cost. Promising strategies included zero-shot prompting with proprietary LLMs like the GPT-4o model, advanced retrieval methods, and fine-tuning smaller, open-source models for data privacy when incorporation of large models into hospital infrastructure is infeasible. Key challenges include accessing sufficiently large real-world data sets, and deployment-associated challenges such as reducing cost, mitigating risk of hallucinations, data leakage, and bias. This review synthesizes progress in applying LLMs to clinical trial matching, highlighting promising directions and key limitations. Standardized metrics, more realistic test sets, and attention to cost-efficiency and fairness will be critical for broader deployment. 

**Abstract (ZH)**: 将患者与临床试验匹配对于发现新型治疗方法至关重要，尤其是在肿瘤学领域。然而，手动匹配劳动密集且容易出错，导致招募延迟。包含大型语言模型（LLMs）的管道提供了一种有希望的解决方案。我们在2020年至2025年间从三个学术数据库和一个预印本服务器中进行了系统性回顾，识别了基于LLM的临床试验匹配方法。在126篇独特文章中，有31篇符合纳入标准。回顾的研究重点仅包括患者与标准匹配（n=4）、患者与试验匹配（n=10）、试验与患者匹配（n=2）、二元资格分类（n=1）以及结合任务（n=14）。16篇研究使用了合成数据；14篇使用了真实患者数据；1篇同时使用了合成数据和真实患者数据。数据集和评估指标的差异性限制了研究间的可比性。在有直接比较的研究中，GPT-4模型在匹配和资格提取方面始终优于其他模型，即使在细调后的模型也是如此，尽管成本较高。有效的策略包括使用专用的LLM（如GPT-4o模型）进行零样本提示、高级检索方法以及在难以将大型模型集成到医院基础设施的情况下，微调更小的开源模型以保护数据隐私。主要挑战包括获取足够大的真实世界数据集，以及部署相关挑战，如降低成本、减轻幻觉风险、数据泄露和偏见。这项回顾总结了将LLM应用于临床试验匹配的进展，指出了有前景的方向和关键限制。标准化指标、更现实的测试集以及对成本效率和公平的关注将是更大范围部署的关键。 

---
# Unveiling the Merits and Defects of LLMs in Automatic Review Generation for Scientific Papers 

**Title (ZH)**: 揭示大语言模型在科学论文自动评审生成中的优势与缺陷 

**Authors**: Ruochi Li, Haoxuan Zhang, Edward Gehringer, Ting Xiao, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19326)  

**Abstract**: The surge in scientific submissions has placed increasing strain on the traditional peer-review process, prompting the exploration of large language models (LLMs) for automated review generation. While LLMs demonstrate competence in producing structured and coherent feedback, their capacity for critical reasoning, contextual grounding, and quality sensitivity remains limited. To systematically evaluate these aspects, we propose a comprehensive evaluation framework that integrates semantic similarity analysis and structured knowledge graph metrics to assess LLM-generated reviews against human-written counterparts. We construct a large-scale benchmark of 1,683 papers and 6,495 expert reviews from ICLR and NeurIPS in multiple years, and generate reviews using five LLMs. Our findings show that LLMs perform well in descriptive and affirmational content, capturing the main contributions and methodologies of the original work, with GPT-4o highlighted as an illustrative example, generating 15.74% more entities than human reviewers in the strengths section of good papers in ICLR 2025. However, they consistently underperform in identifying weaknesses, raising substantive questions, and adjusting feedback based on paper quality. GPT-4o produces 59.42% fewer entities than real reviewers in the weaknesses and increases node count by only 5.7% from good to weak papers, compared to 50% in human reviews. Similar trends are observed across all conferences, years, and models, providing empirical foundations for understanding the merits and defects of LLM-generated reviews and informing the development of future LLM-assisted reviewing tools. Data, code, and more detailed results are publicly available at this https URL. 

**Abstract (ZH)**: 科学投稿量的激增对传统同行评审过程产生了越来越大的压力，促使人们探索大型语言模型（LLMs）以自动化生成评审。虽然LLMs在生成结构化和连贯反馈方面表现出色，但在批判性推理、上下文关联和质量敏感性方面的能力仍然有限。为系统评估这些方面，我们提出了一种综合评估框架，该框架结合了语义相似性分析和结构化知识图谱指标，用于评估LLM生成的评审与人类撰写的同类评审。我们构建了一个包含1,683篇论文和6,495份专家评审的大规模基准，来源包括多个年度的ICLR和NeurIPS，使用五种LLM生成评审。研究结果表明，LLMs在描述性和肯定性内容方面表现良好，能够捕捉原始工作的主要贡献和方法，GPT-4o在ICLR 2025优良论文的优点部分生成的实体多出15.74%。然而，他们在识别缺点、提出实质性问题以及根据论文质量调整反馈方面表现逊色。GPT-4o在缺点部分生成的实体比真实评审人少59.42%，从优良论文到较差论文的节点数量仅增长5.7%，而人类评审人在这一比例为50%。在所有会议、年度和模型中观察到类似趋势，为理解LLM生成评审的优势和缺陷提供了实证基础，并为未来LLM辅助评审工具的发展提供了指导。相关数据、代码和更详细的结果可在以下网址获取。 

---
# Readme_AI: Dynamic Context Construction for Large Language Models 

**Title (ZH)**: Readme_AI: 大型语言模型的动态上下文构建 

**Authors**: Millie Vyas, Timothy Blattner, Alden Dima  

**Link**: [PDF](https://arxiv.org/pdf/2509.19322)  

**Abstract**: Despite being trained on significant amounts of data, Large Language Models (LLMs) can provide inaccurate or unreliable information in the context of a user's specific query. Given query-specific context significantly improves the usefulness of its responses. In this paper, we present a specification that can be used to dynamically build context for data sources. The data source owner creates the file containing metadata for LLMs to use when reasoning about dataset-related queries. To demonstrate our proposed specification, we created a prototype Readme_AI Model Context Protocol (MCP) server that retrieves the metadata from the data source and uses it to dynamically build context. Some features that make this specification dynamic are the extensible types that represent crawling web-pages, fetching data from data repositories, downloading and parsing publications, and general text. The context is formatted and grouped using user-specified tags that provide clear contextual information for the LLM to reason about the content. We demonstrate the capabilities of this early prototype by asking the LLM about the NIST-developed Hedgehog library, for which common LLMs often provides inaccurate and irrelevant responses containing hallucinations. With Readme_AI, the LLM receives enough context that it is now able to reason about the library and its use, and even generate code interpolated from examples that were included in the Readme_AI file provided by Hedgehog's developer. Our primary contribution is a extensible protocol for dynamically grounding LLMs in specialized, owner-provided data, enhancing responses from LLMs and reducing hallucinations. The source code for the Readme_AI tool is posted here: this https URL . 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在大量数据训练下工作，但在处理用户特定查询时仍可能提供不准确或不可靠的信息。给定查询特定的上下文显著提高了其响应的有用性。在本文中，我们提出了一种规范，可用于动态构建数据源上下文。数据源所有者创建包含元数据的文件，供LLMs在处理与数据集相关的查询时使用。为了展示我们提出的规范，我们创建了一个原型Readme_AI Model Context Protocol（MCP）服务器，该服务器从数据源检索元数据，并使用这些信息动态构建上下文。使该规范动态的某些功能包括表示爬取网页、从数据仓库获取数据、下载和解析出版物以及通用文本的可扩展类型。上下文按照用户指定的标签格式化和分组，这些标签为LLM提供清晰的上下文信息，使其能够推理内容。通过使用Readme_AI，LLM接收到足够的上下文，现在能够推理Hedgehog库及其用法，并甚至生成来自Readme_AI文件中提供的示例插值的代码。我们的主要贡献是一种可扩展协议，用于动态地将LLMs与特定的所有者提供的数据关联起来，从而增强LLMs的响应并减少幻觉现象。Readme_AI工具的源代码在此处发布：this https URL。 

---
# FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering 

**Title (ZH)**: FHIR-AgentBench: LLM代理的实时互操作医疗记录问答基准测试 

**Authors**: Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang jia, Jong Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19319)  

**Abstract**: The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (this https URL) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications. 

**Abstract (ZH)**: 华为水平七快速医疗互操作资源标准（HL7 FHIR）的 recent shift 为临床AI开辟了新领域，要求LLM代理导航基于资源的复杂数据模型，而非传统的结构化医疗数据。然而，现有基准未能跟上这一转变，缺乏评估最新LLM在可互操作临床数据上的表现所需的现实性。为弥补这一差距，我们引入了FHIR-AgentBench基准，该基准将2,931个真实世界的临床问题与HL7 FHIR标准相结合。利用此基准，我们系统地评估了代理框架，比较了不同的数据检索策略（直接FHIR API调用 vs. 专用工具）、交互模式（单轮 vs. 多轮）以及推理策略（自然语言 vs. 代码生成）。我们的实验强调了从复杂FHIR资源检索数据的实际挑战以及在其中推理的难度，这两者都严重影响了问题回答的表现。我们公开发布了FHIR-AgentBench数据集和评估套件（this https URL），以促进可重复研究并推动稳健可靠的临床应用LLM代理的发展。 

---
# Automated Item Neutralization for Non-Cognitive Scales: A Large Language Model Approach to Reducing Social-Desirability Bias 

**Title (ZH)**: 自动项目中和以减少社会偏好偏见：大型语言模型方法在非认知量表中的应用 

**Authors**: Sirui Wu, Daijin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19314)  

**Abstract**: This study evaluates item neutralization assisted by the large language model (LLM) to reduce social desirability bias in personality assessment. GPT-o3 was used to rewrite the International Personality Item Pool Big Five Measure (IPIP-BFM-50), and 203 participants completed either the original or neutralized form along with the Marlowe-Crowne Social Desirability Scale. The results showed preserved reliability and a five-factor structure, with gains in Conscientiousness and declines in Agreeableness and Openness. The correlations with social desirability decreased for several items, but inconsistently. Configural invariance held, though metric and scalar invariance failed. Findings support AI neutralization as a potential but imperfect bias-reduction method. 

**Abstract (ZH)**: 基于大型语言模型的项目中和对人格评估中社会可接受性偏见的减少的评估 

---
# LLMs as verification oracles for Solidity 

**Title (ZH)**: LLMs作为Solidity的验证 oracle 

**Authors**: Massimo Bartoletti, Enrico Lipparini, Livio Pompianu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19153)  

**Abstract**: Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs serve as verification oracles, capable of reasoning about arbitrary contract-specific properties? In this paper, we provide the first systematic evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs can be surprisingly effective as verification oracles, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing. 

**Abstract (ZH)**: 确保智能合约的正确性至关重要，即使是细微的缺陷也可能导致严重的财务损失。虽然能检测常见漏洞模式的bug检测工具可以作为第一道防线，但大多数实际的利用和损失源自合约业务逻辑错误。形式验证工具如SolCMC和Certora Prover解决了这一挑战，但它们的影响仍受限于陡峭的学习曲线和受限的规格语言。近期研究表明，大型语言模型（LLMs）可用于安全相关任务，如漏洞检测和测试生成。然而，一个基本问题仍然悬而未决：LLMs能否作为验证 oracle，能够对任意合约特定属性进行推理？在本文中，我们首次系统评估了GPT-5这一最先进的推理LLM在这一角色上的表现。我们在一个大规模的形式验证任务数据集上测试其性能，将其输出与现有的形式验证工具进行比较，并评估其在实际审计场景中的实用性。我们的研究结合了定量指标和定性分析，表明近期的推理导向型LLM可以出乎意料地有效地作为验证oracle，这暗示了AI与形式方法在安全智能合约开发和审计中的融合的新前沿。 

---
# GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models 

**Title (ZH)**: GAUSS: 大型语言模型结构化数学能力基准测试 

**Authors**: Yue Zhang, Jiaxin Zhang, Qiuyu Ren, Tahsin Saffat, Xiaoxuan Liu, Zitong Yang, Banghua Zhu, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.18122)  

**Abstract**: We introduce \textbf{GAUSS} (\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics), a benchmark that evaluates LLMs' mathematical abilities across twelve core skill dimensions, grouped into three domains: knowledge and understanding, problem solving and communication, and meta-skills and creativity. By categorizing problems according to cognitive skills and designing tasks that isolate specific abilities, GAUSS constructs comprehensive, fine-grained, and interpretable profiles of models' mathematical abilities. These profiles faithfully represent their underlying mathematical intelligence. To exemplify how to use the \textsc{GAUSS} benchmark, we have derived the skill profile of \textsc{GPT-5-thinking}, revealing its strengths and weaknesses as well as its differences relative to \textsc{o4-mini-high}, thereby underscoring the value of multidimensional, skill-based evaluation. 

**Abstract (ZH)**: 我们将介绍\textbf{GAUSS}（\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics），这是一个评估大型语言模型在十二个核心技能维度上数学能力的基准，这些维度被分为三个领域：知识与理解、问题解决与沟通、元技能与创造力。通过根据认知技能分类问题并设计分离特定能力的任务，GAUSS构建了模型数学能力的全面、细致且可解释的画像，这些画像忠实地代表了其潜在的数学智能。为了说明如何使用\textsc{GAUSS}基准，我们为\textsc{GPT-5-thinking}制定了技能画像，揭示了其优势和劣势，以及与\textsc{o4-mini-high}的区别，从而强调了多维度、基于技能评价的价值。 

---
