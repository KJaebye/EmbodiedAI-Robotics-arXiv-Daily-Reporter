# DSDrive: Distilling Large Language Model for Lightweight End-to-End Autonomous Driving with Unified Reasoning and Planning 

**Title (ZH)**: DSDrive：精炼大型语言模型以实现轻量级端到端自主驾驶，结合统一推理与规划 

**Authors**: Wenru Liu, Pei Liu, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.05360)  

**Abstract**: We present DSDrive, a streamlined end-to-end paradigm tailored for integrating the reasoning and planning of autonomous vehicles into a unified framework. DSDrive leverages a compact LLM that employs a distillation method to preserve the enhanced reasoning capabilities of a larger-sized vision language model (VLM). To effectively align the reasoning and planning tasks, a waypoint-driven dual-head coordination module is further developed, which synchronizes dataset structures, optimization objectives, and the learning process. By integrating these tasks into a unified framework, DSDrive anchors on the planning results while incorporating detailed reasoning insights, thereby enhancing the interpretability and reliability of the end-to-end pipeline. DSDrive has been thoroughly tested in closed-loop simulations, where it performs on par with benchmark models and even outperforms in many key metrics, all while being more compact in size. Additionally, the computational efficiency of DSDrive (as reflected in its time and memory requirements during inference) has been significantly enhanced. Evidently thus, this work brings promising aspects and underscores the potential of lightweight systems in delivering interpretable and efficient solutions for AD. 

**Abstract (ZH)**: DSDrive：一种集成自动驾驶车辆推理与规划的精简端到端 paradigm 

---
# PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes 

**Title (ZH)**: PlaceIt3D：面向真实3D场景的语言引导对象放置 

**Authors**: Ahmed Abdelreheem, Filippo Aleotti, Jamie Watson, Zawar Qureshi, Abdelrahman Eldesokey, Peter Wonka, Gabriel Brostow, Sara Vicente, Guillermo Garcia-Hernando  

**Link**: [PDF](https://arxiv.org/pdf/2505.05288)  

**Abstract**: We introduce the novel task of Language-Guided Object Placement in Real 3D Scenes. Our model is given a 3D scene's point cloud, a 3D asset, and a textual prompt broadly describing where the 3D asset should be placed. The task here is to find a valid placement for the 3D asset that respects the prompt. Compared with other language-guided localization tasks in 3D scenes such as grounding, this task has specific challenges: it is ambiguous because it has multiple valid solutions, and it requires reasoning about 3D geometric relationships and free space. We inaugurate this task by proposing a new benchmark and evaluation protocol. We also introduce a new dataset for training 3D LLMs on this task, as well as the first method to serve as a non-trivial baseline. We believe that this challenging task and our new benchmark could become part of the suite of benchmarks used to evaluate and compare generalist 3D LLM models. 

**Abstract (ZH)**: 语言引导的三维场景对象放置任务 

---
# Conversational Process Model Redesign 

**Title (ZH)**: 对话过程模型重设计 

**Authors**: Nataliia Klievtsova, Timotheus Kampik, Juergen Mangler, Stefanie Rinderle-Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.05453)  

**Abstract**: With the recent success of large language models (LLMs), the idea of AI-augmented Business Process Management systems is becoming more feasible. One of their essential characteristics is the ability to be conversationally actionable, allowing humans to interact with the LLM effectively to perform crucial process life cycle tasks such as process model design and redesign. However, most current research focuses on single-prompt execution and evaluation of results, rather than on continuous interaction between the user and the LLM. In this work, we aim to explore the feasibility of using LLMs to empower domain experts in the creation and redesign of process models in an iterative and effective way. The proposed conversational process model redesign (CPD) approach receives as input a process model and a redesign request by the user in natural language. Instead of just letting the LLM make changes, the LLM is employed to (a) identify process change patterns from literature, (b) re-phrase the change request to be aligned with an expected wording for the identified pattern (i.e., the meaning), and then to (c) apply the meaning of the change to the process model. This multi-step approach allows for explainable and reproducible changes. In order to ensure the feasibility of the CPD approach, and to find out how well the patterns from literature can be handled by the LLM, we performed an extensive evaluation. The results show that some patterns are hard to understand by LLMs and by users. Within the scope of the study, we demonstrated that users need support to describe the changes clearly. Overall the evaluation shows that the LLMs can handle most changes well according to a set of completeness and correctness criteria. 

**Abstract (ZH)**: 基于大型语言模型的迭代过程模型重设计方法探究 

---
# MARK: Memory Augmented Refinement of Knowledge 

**Title (ZH)**: MARK：基于记忆的知識精煉增強 

**Authors**: Anish Ganguli, Prabal Deb, Debleena Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.05177)  

**Abstract**: Large Language Models (LLMs) assist in specialized tasks but struggle to align with evolving domain knowledge without costly fine-tuning. Domain knowledge consists of: Knowledge: Immutable facts (e.g., 'A stone is solid') and generally accepted principles (e.g., ethical standards); Refined Memory: Evolving insights shaped by business needs and real-world changes. However, a significant gap often exists between a domain expert's deep, nuanced understanding and the system's domain knowledge, which can hinder accurate information retrieval and application. Our Memory-Augmented Refinement of Knowledge (MARK) framework enables LLMs to continuously learn without retraining by leveraging structured refined memory, inspired by the Society of Mind. MARK operates through specialized agents, each serving a distinct role: Residual Refined Memory Agent: Stores and retrieves domain-specific insights to maintain context over time; User Question Refined Memory Agent: Captures user-provided facts, abbreviations, and terminology for better comprehension; LLM Response Refined Memory Agent: Extracts key elements from responses for refinement and personalization. These agents analyse stored refined memory, detect patterns, resolve contradictions, and improve response accuracy. Temporal factors like recency and frequency prioritize relevant information while discarding outdated insights. MARK enhances LLMs in multiple ways: Ground Truth Strategy: Reduces hallucinations by establishing a structured reference; Domain-Specific Adaptation: Essential for fields like healthcare, law, and manufacturing, where proprietary insights are absent from public datasets; Personalized AI Assistants: Improves virtual assistants by remembering user preferences, ensuring coherent responses over time. 

**Abstract (ZH)**: Large Language Models的精炼知识增强框架：持续学习而不重新训练 

---
# Enigme: Generative Text Puzzles for Evaluating Reasoning in Language Models 

**Title (ZH)**: Enigme：生成文本谜题以评估语言模型的推理能力 

**Authors**: John Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2505.04914)  

**Abstract**: Transformer-decoder language models are a core innovation in text based generative artificial intelligence. These models are being deployed as general-purpose intelligence systems in many applications. Central to their utility is the capacity to understand natural language commands and exploit the reasoning embedded in human text corpora to apply some form of reasoning process to a wide variety of novel tasks. To understand the limitations of this approach to generating reasoning we argue that we need to consider the architectural constraints of these systems. Consideration of the latent variable structure of transformer-decoder models allows us to design reasoning tasks that should probe the boundary of their capacity to reason. We present enigme, an open-source library for generating text-based puzzles to be used in training and evaluating reasoning skills within transformer-decoder models and future AI architectures. 

**Abstract (ZH)**: 基于Transformer解码器的语言模型是文本生成式人工智能的核心创新。这些模型正在许多应用中被部署为通用智能系统。它们的功能核心在于理解自然语言命令，并利用人类文本语料库中嵌入的推理能力来处理各种新型任务。为了理解生成推理的这一方法的局限性，我们认为需要考虑这些系统架构上的限制。通过考虑Transformer解码器模型的潜在变量结构，我们可以设计出能够探测其推理能力边界的推理任务。我们介绍了enigme，一个开源库，用于生成基于文本的谜题，以用于训练和评估Transformer解码器模型及未来AI架构中的推理能力。 

---
# Large Language Models are Autonomous Cyber Defenders 

**Title (ZH)**: 大型语言模型是自主网络防御者 

**Authors**: Sebastián R. Castro, Roberto Campbell, Nancy Lau, Octavio Villalobos, Jiaqi Duan, Alvaro A. Cardenas  

**Link**: [PDF](https://arxiv.org/pdf/2505.04843)  

**Abstract**: Fast and effective incident response is essential to prevent adversarial cyberattacks. Autonomous Cyber Defense (ACD) aims to automate incident response through Artificial Intelligence (AI) agents that plan and execute actions. Most ACD approaches focus on single-agent scenarios and leverage Reinforcement Learning (RL). However, ACD RL-trained agents depend on costly training, and their reasoning is not always explainable or transferable. Large Language Models (LLMs) can address these concerns by providing explainable actions in general security contexts. Researchers have explored LLM agents for ACD but have not evaluated them on multi-agent scenarios or interacting with other ACD agents. In this paper, we show the first study on how LLMs perform in multi-agent ACD environments by proposing a new integration to the CybORG CAGE 4 environment. We examine how ACD teams of LLM and RL agents can interact by proposing a novel communication protocol. Our results highlight the strengths and weaknesses of LLMs and RL and help us identify promising research directions to create, train, and deploy future teams of ACD agents. 

**Abstract (ZH)**: Fast and Effective 多Agent 事件响应对于防止对抗性网络攻击至关重要。自主网络安全防护（ACD）旨在通过人工智能（AI）代理自动执行事件响应。大多数 ACD 方法侧重于单代理场景，并利用强化学习（RL）进行训练。然而，ACD RL 训练代理依赖于昂贵的训练过程，其推理过程往往缺乏可解释性和可迁移性。大规模语言模型（LLMs）可以通过在一般安全背景下提供可解释的行动来解决这些问题。研究人员已经探索了LLM代理在ACD中的应用，但尚未在多代理场景或与其他ACD代理交互的情况下进行评估。本文通过提出一种新的集成到CybORG CAGE 4环境中的方法，展示了LLMs在多代理ACD环境中的第一个研究成果，并提出了一种新的通信协议来探讨ACD团队中LLM和RL代理的交互方式。我们的研究结果突显了LLMs和RL的优缺点，并帮助我们确定未来ACD代理团队创建、训练和部署的研究方向。 

---
# The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems 

**Title (ZH)**: LLMs在智能辅导系统中构建逻辑问题证明和提示的潜力与局限性 

**Authors**: Sutapa Dey Tithi, Arun Kumar Ramesh, Clara DiMarco, Xiaoyi Tian, Nazia Alam, Kimia Fazeli, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2505.04736)  

**Abstract**: Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance with 84.4% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but requires additional modifications to ensure accuracy and pedagogical appropriateness. 

**Abstract (ZH)**: 智能辅导系统在教学形式命题逻辑证明方面表现出有效性，但其依赖于模板解释的限制影响了其提供个性化学生反馈的能力。尽管大规模语言模型（LLMs）提供了动态反馈生成的前景，但也存在产生幻觉或教学上不合适的解释的风险。我们评估了LLMs在构建多步骤符号逻辑证明时的逐步准确性，比较了六种提示技术在4个最先进的LLMs上解决358个命题逻辑问题的表现。结果表明，DeepSeek-V3实现了更高的性能，逐步证明构建准确率为84.4%，尤其在简单规则方面表现出色。我们进一步使用表现最佳的LLM生成了源自逻辑ITS的1,050个独特学生问题解决状态的解释性提示，并对这些建议进行了四项标准的评估，包括LLM评分和人工专家在20%样本上的评分。分析结果发现，LLM生成的提示准确率为75%，在一致性和清晰性方面得到了人类评估者的高度评价，但不太擅长解释提供提示的原因及其更大的上下文。我们的结果表明，LLMs可以用于增强逻辑辅导系统的提示，但需要进一步的修改以确保准确性和教学适宜性。 

---
# Towards Artificial Intelligence Research Assistant for Expert-Involved Learning 

**Title (ZH)**: 面向专家参与学习的人工智能研究助手 

**Authors**: Tianyu Liu, Simeng Han, Xiao Luo, Hanchen Wang, Pan Lu, Biqing Zhu, Yuge Wang, Keyi Li, Jiapeng Chen, Rihao Qu, Yufeng Liu, Xinyue Cui, Aviv Yaish, Yuhang Chen, Minsheng Hao, Chuhan Li, Kexing Li, Arman Cohan, Hua Xu, Mark Gerstein, James Zou, Hongyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04638)  

**Abstract**: Large Language Models (LLMs) and Large Multi-Modal Models (LMMs) have emerged as transformative tools in scientific research, yet their reliability and specific contributions to biomedical applications remain insufficiently characterized. In this study, we present \textbf{AR}tificial \textbf{I}ntelligence research assistant for \textbf{E}xpert-involved \textbf{L}earning (ARIEL), a multimodal dataset designed to benchmark and enhance two critical capabilities of LLMs and LMMs in biomedical research: summarizing extensive scientific texts and interpreting complex biomedical figures. To facilitate rigorous assessment, we create two open-source sets comprising biomedical articles and figures with designed questions. We systematically benchmark both open- and closed-source foundation models, incorporating expert-driven human evaluations conducted by doctoral-level experts. Furthermore, we improve model performance through targeted prompt engineering and fine-tuning strategies for summarizing research papers, and apply test-time computational scaling to enhance the reasoning capabilities of LMMs, achieving superior accuracy compared to human-expert corrections. We also explore the potential of using LMM Agents to generate scientific hypotheses from diverse multimodal inputs. Overall, our results delineate clear strengths and highlight significant limitations of current foundation models, providing actionable insights and guiding future advancements in deploying large-scale language and multi-modal models within biomedical research. 

**Abstract (ZH)**: 用于专家参与学习的人工智能研究助手（ARIEL）：大规模语言模型和大规模多模态模型在生物医学研究中的评估与提升 

---
# StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant 

**Title (ZH)**: StreamBridge: 将您的离线视频大型语言模型转化为 proactive 流式助手 

**Authors**: Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, Ping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05467)  

**Abstract**: We present StreamBridge, a simple yet effective framework that seamlessly transforms offline Video-LLMs into streaming-capable models. It addresses two fundamental challenges in adapting existing models into online scenarios: (1) limited capability for multi-turn real-time understanding, and (2) lack of proactive response mechanisms. Specifically, StreamBridge incorporates (1) a memory buffer combined with a round-decayed compression strategy, supporting long-context multi-turn interactions, and (2) a decoupled, lightweight activation model that can be effortlessly integrated into existing Video-LLMs, enabling continuous proactive responses. To further support StreamBridge, we construct Stream-IT, a large-scale dataset tailored for streaming video understanding, featuring interleaved video-text sequences and diverse instruction formats. Extensive experiments show that StreamBridge significantly improves the streaming understanding capabilities of offline Video-LLMs across various tasks, outperforming even proprietary models such as GPT-4o and Gemini 1.5 Pro. Simultaneously, it achieves competitive or superior performance on standard video understanding benchmarks. 

**Abstract (ZH)**: StreamBridge：一种简单有效的框架，无缝地将离线Video-LLMs转换为流式模型，解决现有模型适应在线场景中的两大根本挑战：（1）多轮实时理解能力有限，（2）缺乏主动响应机制。 

---
# ComPO: Preference Alignment via Comparison Oracles 

**Title (ZH)**: ComPO：通过比较或acles实现偏好对齐 

**Authors**: Peter Chen, Xi Chen, Wotao Yin, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05465)  

**Abstract**: Direct alignment methods are increasingly used for aligning large language models (LLMs) with human preferences. However, these methods suffer from the issues of verbosity and likelihood displacement, which can be driven by the noisy preference pairs that induce similar likelihood for preferred and dispreferred responses. The contributions of this paper are two-fold. First, we propose a new preference alignment method based on comparison oracles and provide the convergence guarantee for its basic scheme. Second, we improve our method using some heuristics and conduct the experiments to demonstrate the flexibility and compatibility of practical scheme in improving the performance of LLMs using noisy preference pairs. Evaluations are conducted across multiple base and instruction-tuned models (Mistral-7B, Llama-3-8B and Gemma-2-9B) with benchmarks (AlpacaEval 2, MT-Bench and Arena-Hard). Experimental results show the effectiveness of our method as an alternative to addressing the limitations of existing direct alignment methods. A highlight of our work is that we evidence the importance of designing specialized methods for preference pairs with distinct likelihood margin, which complements the recent findings in \citet{Razin-2025-Unintentional}. 

**Abstract (ZH)**: 直接对齐方法越来越多地被用于将大型语言模型（LLMs）与人类偏好对齐。然而，这些方法受到冗长性和似然性位移的问题困扰，这些问题可能是由引起偏好响应和未偏好响应类似似然性的嘈杂偏好对驱动的。本文的贡献主要有两点。首先，我们提出了一种基于比较或acles的新偏好对齐方法，并为其基本方案提供了收敛性保证。其次，我们通过一些启发式方法改进了该方法，并通过实验展示了如何使用嘈杂的偏好对提高LLM性能的灵活性和兼容性。我们在多种基模型和指令调整模型（Mistral-7B、Llama-3-8B 和 Gemma-2-9B）以及基准数据集（AlpacaEval 2、MT-Bench 和 Arena-Hard）上进行了评估。实验结果表明，我们的方法可以作为克服现有直接对齐方法局限性的替代方案的有效性。我们的工作亮点在于，我们证明了为具有不同似然性边际的偏好对设计专门方法的重要性，这补充了最近 \citet{Razin-2025-Unintentional} 的发现。 

---
# TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering 

**Title (ZH)**: TransProQA: 一种基于大型语言模型的专业问答文学翻译评估指标 

**Authors**: Ran Zhang, Wei Zhao, Lieve Macken, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05423)  

**Abstract**: The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce TransProQA, a novel, reference-free, LLM-based question-answering (QA) framework designed specifically for literary translation evaluation. TransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, TransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art (SOTA) metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, TransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations. 

**Abstract (ZH)**: 大型语言模型对文学领域的 Impact 建立专用于文学翻译评价的 TransProQA：一种参考无关的基于大型语言模型的问题回答框架 

---
# Crosslingual Reasoning through Test-Time Scaling 

**Title (ZH)**: 跨语言推理通过测试时缩放 

**Authors**: Zheng-Xin Yong, M. Farid Adilazuarda, Jonibek Mansurov, Ruochen Zhang, Niklas Muennighoff, Carsten Eickhoff, Genta Indra Winata, Julia Kreutzer, Stephen H. Bach, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2505.05408)  

**Abstract**: Reasoning capabilities of large language models are primarily studied for English, even when pretrained models are multilingual. In this work, we investigate to what extent English reasoning finetuning with long chain-of-thoughts (CoTs) can generalize across languages. First, we find that scaling up inference compute for English-centric reasoning language models (RLMs) improves multilingual mathematical reasoning across many languages including low-resource languages, to an extent where they outperform models twice their size. Second, we reveal that while English-centric RLM's CoTs are naturally predominantly English, they consistently follow a quote-and-think pattern to reason about quoted non-English inputs. Third, we discover an effective strategy to control the language of long CoT reasoning, and we observe that models reason better and more efficiently in high-resource languages. Finally, we observe poor out-of-domain reasoning generalization, in particular from STEM to cultural commonsense knowledge, even for English. Overall, we demonstrate the potentials, study the mechanisms and outline the limitations of crosslingual generalization of English reasoning test-time scaling. We conclude that practitioners should let English-centric RLMs reason in high-resource languages, while further work is needed to improve reasoning in low-resource languages and out-of-domain contexts. 

**Abstract (ZH)**: 大型语言模型的推理能力主要针对英语进行研究，即使预训练模型是多语言的。在本工作中，我们调查了以英语为中心的长链推理（CoTs）的推理微调在多大程度上能在不同语言之间泛化。首先，我们发现，增加以英语为中心的推理语言模型（RLMs）推理计算的规模，可以在包括低资源语言在内的多种语言中提高多语言数学推理能力，甚至使其优于两倍规模的模型。其次，我们揭示了虽然以英语为中心的RLMs的CoTs通常是英语为主的，但它们会一致地遵循引用并思考的模式来推理非英语输入。第三，我们发现了一种控制长CoTs推理语言的有效策略，并发现模型在高资源语言中推理更好且更高效。最后，我们观察到，在特定领域外的推理泛化表现较差，尤其是在STEM领域到文化常识知识方面，即使是英语也不例外。总体而言，我们展示了英语推理测试时缩放的跨语言泛化的潜力，研究了其机制并概述了其局限性。我们得出结论，实践者应该让以英语为中心的RLMs在高资源语言中推理，而进一步的工作需要提高低资源语言和特定领域外语境中的推理能力。 

---
# Software Development Life Cycle Perspective: A Survey of Benchmarks for CodeLLMs and Agents 

**Title (ZH)**: 软件开发生命周期视角：代码LLMs和代理的基准调研 

**Authors**: Kaixin Wang, Tianlin Li, Xiaoyu Zhang, Chong Wang, Weisong Sun, Yang Liu, Bin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05283)  

**Abstract**: Code large language models (CodeLLMs) and agents have shown great promise in tackling complex software engineering this http URL to traditional software engineering methods, CodeLLMs and agents offer stronger abilities, and can flexibly process inputs and outputs in both natural and code. Benchmarking plays a crucial role in evaluating the capabilities of CodeLLMs and agents, guiding their development and deployment. However, despite their growing significance, there remains a lack of comprehensive reviews of benchmarks for CodeLLMs and agents. To bridge this gap, this paper provides a comprehensive review of existing benchmarks for CodeLLMs and agents, studying and analyzing 181 benchmarks from 461 relevant papers, covering the different phases of the software development life cycle (SDLC). Our findings reveal a notable imbalance in the coverage of current benchmarks, with approximately 60% focused on the software development phase in SDLC, while requirements engineering and software design phases receive minimal attention at only 5% and 3%, respectively. Additionally, Python emerges as the dominant programming language across the reviewed benchmarks. Finally, this paper highlights the challenges of current research and proposes future directions, aiming to narrow the gap between the theoretical capabilities of CodeLLMs and agents and their application in real-world scenarios. 

**Abstract (ZH)**: Code大模型（CodeLLMs）和代理在复杂软件工程中的应用前景：面向传统软件工程方法，CodeLLMs和代理提供了更强的能力，并且能够在自然语言和代码之间灵活处理输入和输出。基准测试在评估CodeLLMs和代理的能力中扮演着至关重要的角色，指导其开发和部署。然而，尽管它们的重要性不断提高，仍缺乏对CodeLLMs和代理基准的全面综述。为了填补这一空白，本论文对现有的CodeLLMs和代理基准进行了全面综述，研究并分析了461篇相关论文中的181个基准，涵盖了软件开发生命周期（SDLC）的不同阶段。我们的研究发现，当前基准的覆盖范围存在明显不平衡，约60%的关注于开发阶段，而需求工程和设计阶段分别仅占5%和3%。此外，Python在所审阅的基准中占据主导地位。最后，本文指出现有研究中的挑战，并提出未来的研究方向，旨在缩小CodeLLMs和代理的理论能力与其在实际应用场景中的应用之间的差距。 

---
# Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks 

**Title (ZH)**: 通过自信息重写攻击揭示文本水印的弱点 

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2505.05190)  

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking. 

**Abstract (ZH)**: 文本水印旨在通过控制大型语言模型（LLM）的采样过程，微妙地将统计信号嵌入文本中，从而使水印检测器能够验证输出是由指定模型生成的。这些水印算法的鲁棒性已成为评估其有效性的重要因素。当前的文本水印算法将水印嵌入高熵令牌以确保文本质量。在本文中，我们揭示这一看似无害的设计可以被攻击者利用，对水印的鲁棒性构成重大威胁。我们提出了一种高效的通用改写攻击方法，即自我信息重写攻击（SIRA），该方法利用漏洞通过计算每个令牌的自我信息来识别潜在的模式令牌并执行有针对性的攻击。我们的工作揭示了当前水印算法中广泛存在的漏洞。实验结果表明，SIRA在仅需每百万令牌0.88美元成本的情况下，对七种最近的方法实现了近乎100%的攻击成功率。我们的方法不需要访问水印算法或水印的LLM，并且可以无缝转移到任何LLM，甚至是移动级别模型作为攻击模型。我们的研究结果突显了对更鲁棒水印方法的迫切需要。 

---
# Dukawalla: Voice Interfaces for Small Businesses in Africa 

**Title (ZH)**: Dukawalla: 非洲地区小型企业的声音界面 

**Authors**: Elizabeth Ankrah, Stephanie Nyairo, Mercy Muchai, Kagonya Awori, Millicent Ochieng, Mark Kariuki, Jacki O'Neill  

**Link**: [PDF](https://arxiv.org/pdf/2505.05170)  

**Abstract**: Small and medium sized businesses often struggle with data driven decision making do to a lack of advanced analytics tools, especially in African countries where they make up a majority of the workforce. Though many tools exist they are not designed to fit into the ways of working of SMB workers who are mobile first, have limited time to learn new workflows, and for whom social and business are tightly coupled. To address this, the Dukawalla prototype was created. This intelligent assistant bridges the gap between raw business data, and actionable insights by leveraging voice interaction and the power of generative AI. Dukawalla provides an intuitive way for business owners to interact with their data, aiding in informed decision making. This paper examines Dukawalla's deployment across SMBs in Nairobi, focusing on their experiences using this voice based assistant to streamline data collection and provide business insights 

**Abstract (ZH)**: 小型和中型企业常常因缺乏高级数据分析工具而难以进行数据驱动的决策，特别是在非洲国家，中小企业占据了大多数劳动力。虽然许多工具已经存在，但它们并未设计成适合移动优先的中小企业员工的工作方式，这些员工没有太多时间学习新的工作流程，而且他们的社交和商业活动紧密相连。为解决这个问题，我们创建了Dukawalla原型。这个智能助手通过利用语音交互和生成式AI的力量，在原始业务数据和可操作的洞察之间架起桥梁。Dukawalla为中小企业主提供了一种直观的数据交互方式，有助于做出明智的决策。本文探讨了Dukawalla在 Nairobi 的中小企业中的部署情况，重点关注这些企业如何使用基于语音的助手来简化数据收集并提供商业见解。 

---
# Understanding In-context Learning of Addition via Activation Subspaces 

**Title (ZH)**: 理解通过激活子空间进行的加法上下文学习 

**Authors**: Xinyan Hu, Kayo Yin, Michael I. Jordan, Jacob Steinhardt, Lijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05145)  

**Abstract**: To perform in-context learning, language models must extract signals from individual few-shot examples, aggregate these into a learned prediction rule, and then apply this rule to new examples. How is this implemented in the forward pass of modern transformer models? To study this, we consider a structured family of few-shot learning tasks for which the true prediction rule is to add an integer $k$ to the input. We find that Llama-3-8B attains high accuracy on this task for a range of $k$, and localize its few-shot ability to just three attention heads via a novel optimization approach. We further show the extracted signals lie in a six-dimensional subspace, where four of the dimensions track the unit digit and the other two dimensions track overall magnitude. We finally examine how these heads extract information from individual few-shot examples, identifying a self-correction mechanism in which mistakes from earlier examples are suppressed by later examples. Our results demonstrate how tracking low-dimensional subspaces across a forward pass can provide insight into fine-grained computational structures. 

**Abstract (ZH)**: 语言模型为了进行上下文学习，必须从个别少样本示例中提取信号，将这些信号整合成一个学习中的预测规则，然后将此规则应用于新示例。这种机制是如何在现代变压器模型的前向传播过程中实现的？为了研究这一点，我们考虑了一类结构化的少样本学习任务，其中真实的预测规则是将整数 \(k\) 加到输入中。我们发现 Llama-3-8B 在 \(k\) 的多种情况上都实现了高精度，并通过一种新型优化方法将其实现的少样本能力局部化到仅三个注意力头中。我们进一步展示了提取的信号位于一个六维子空间中，其中四个维度跟踪个位数，另外两个维度跟踪整体大小。最后，我们检查了这些头如何从个别少样本示例中提取信息，识别出一种自我修正机制，即早期示例中的错误被后来的示例抑制。我们的结果展示了在整个前向传播过程中跟踪低维子空间如何提供对精细计算结构的洞察。 

---
# Rethinking Invariance in In-context Learning 

**Title (ZH)**: 重思上下文学习中的不变性 

**Authors**: Lizhe Fang, Yifei Wang, Khashayar Gatmiry, Lei Fang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04994)  

**Abstract**: In-Context Learning (ICL) has emerged as a pivotal capability of auto-regressive large language models, yet it is hindered by a notable sensitivity to the ordering of context examples regardless of their mutual independence. To address this issue, recent studies have introduced several variant algorithms of ICL that achieve permutation invariance. However, many of these do not exhibit comparable performance with the standard auto-regressive ICL algorithm. In this work, we identify two crucial elements in the design of an invariant ICL algorithm: information non-leakage and context interdependence, which are not simultaneously achieved by any of the existing methods. These investigations lead us to the proposed Invariant ICL (InvICL), a methodology designed to achieve invariance in ICL while ensuring the two properties. Empirically, our findings reveal that InvICL surpasses previous models, both invariant and non-invariant, in most benchmark datasets, showcasing superior generalization capabilities across varying input lengths. Code is available at this https URL. 

**Abstract (ZH)**: 基于上下文学习（ICL）已成为自回归大型语言模型的关键能力，但其对上下文示例排序的敏感性限制了其发展，尤其是示例之间并非相互独立时。为解决这一问题，近期研究引入了几种ICI的变体算法以实现排列不变性，但许多变体算法的性能不及标准的自回归ICI算法。本文识别出设计一个不变的ICL算法的两个关键要素：信息不泄露和上下文互依赖性，这两个要素至今未由任何现有方法同时实现。这些研究推动我们提出了不变ICL（InvICL），一种旨在实现ICL不变性同时确保上述两个属性的方法。我们的实验发现表明，InvICL在大多数基准数据集中超越了之前的所有模型，展现出更强的泛化能力，覆盖不同输入长度。代码可在以下链接获取：this https URL。 

---
# Chain-of-Thought Tokens are Computer Program Variables 

**Title (ZH)**: Chain-of-Thought Tokens是计算机程序变量 

**Authors**: Fangwei Zhu, Peiyi Wang, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04955)  

**Abstract**: Chain-of-thoughts (CoT) requires large language models (LLMs) to generate intermediate steps before reaching the final answer, and has been proven effective to help LLMs solve complex reasoning tasks. However, the inner mechanism of CoT still remains largely unclear. In this paper, we empirically study the role of CoT tokens in LLMs on two compositional tasks: multi-digit multiplication and dynamic programming. While CoT is essential for solving these problems, we find that preserving only tokens that store intermediate results would achieve comparable performance. Furthermore, we observe that storing intermediate results in an alternative latent form will not affect model performance. We also randomly intervene some values in CoT, and notice that subsequent CoT tokens and the final answer would change correspondingly. These findings suggest that CoT tokens may function like variables in computer programs but with potential drawbacks like unintended shortcuts and computational complexity limits between tokens. The code and data are available at this https URL. 

**Abstract (ZH)**: 链思（CoT）要求大型语言模型（LLMs）在到达最终答案之前生成中间步骤，并已被证明有助于解决复杂的推理任务。然而，CoT的内在机制仍然 largely unclear。在本文中，我们实证研究了CoT令牌在LLMs上的作用，针对两个组合任务：多位数乘法和动态规划。虽然CoT对于解决这些问题是必要的，我们发现仅保留存储中间结果的令牌即可实现相当的性能。此外，我们观察到以另一种潜在的隐式形式存储中间结果不会影响模型性能。我们还随机干预了CoT中的某些值，注意到后续的CoT令牌和最终答案会相应地发生变化。这些发现表明，CoT令牌可能类似于计算机程序中的变量，但可能存在未预期的捷径和令牌之间计算复杂性的局限性。相关代码和数据可在以下链接获取：this https URL。 

---
# GroverGPT-2: Simulating Grover's Algorithm via Chain-of-Thought Reasoning and Quantum-Native Tokenization 

**Title (ZH)**: GroverGPT-2：通过链式推理和量子原生标记化模拟Grover算法 

**Authors**: Min Chen, Jinglei Cheng, Pingzhi Li, Haoran Wang, Tianlong Chen, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04880)  

**Abstract**: Quantum computing offers theoretical advantages over classical computing for specific tasks, yet the boundary of practical quantum advantage remains an open question. To investigate this boundary, it is crucial to understand whether, and how, classical machines can learn and simulate quantum algorithms. Recent progress in large language models (LLMs) has demonstrated strong reasoning abilities, prompting exploration into their potential for this challenge. In this work, we introduce GroverGPT-2, an LLM-based method for simulating Grover's algorithm using Chain-of-Thought (CoT) reasoning and quantum-native tokenization. Building on its predecessor, GroverGPT-2 performs simulation directly from quantum circuit representations while producing logically structured and interpretable outputs. Our results show that GroverGPT-2 can learn and internalize quantum circuit logic through efficient processing of quantum-native tokens, providing direct evidence that classical models like LLMs can capture the structure of quantum algorithms. Furthermore, GroverGPT-2 outputs interleave circuit data with natural language, embedding explicit reasoning into the simulation. This dual capability positions GroverGPT-2 as a prototype for advancing machine understanding of quantum algorithms and modeling quantum circuit logic. We also identify an empirical scaling law for GroverGPT-2 with increasing qubit numbers, suggesting a path toward scalable classical simulation. These findings open new directions for exploring the limits of classical simulatability, enhancing quantum education and research, and laying groundwork for future foundation models in quantum computing. 

**Abstract (ZH)**: 量子计算在特定任务上提供了理论上的优势，但实用的量子优势边界仍然是一个开放问题。为了探讨这一边界，了解经典机器能否学习和模拟量子算法至关重要。近年来，大型语言模型（LLMs）的进步展现出强大的推理能力，促使人们探索其在这项挑战中的潜力。在这项工作中，我们引入了GroverGPT-2，这是一种基于LLM的方法，通过链式思考（CoT）推理和量子本征标记来模拟Grover算法。GroverGPT-2在其前身的基础上，直接从量子电路表示中进行模拟，生成逻辑结构化和可解释的输出。我们的结果显示，GroverGPT-2可以通过高效处理量子本征标记来学习和内化量子电路逻辑，提供了经典模型如LLMs可以捕捉量子算法结构的直接证据。此外，GroverGPT-2输出中嵌入了量子电路数据与自然语言的交织，将显式推理嵌入到模拟中。这种双重能力使GroverGPT-2成为推进机器对量子算法的理解和建模量子电路逻辑的原型。我们还识别了GroverGPT-2随量子比特数量增加的实证缩放定律，这为可扩展的经典模拟指出了道路。这些发现为探索经典模拟极限提供了新方向，增强了量子教育与研究，并为未来量子计算的基础模型奠定了基础。 

---
# PR2: Peephole Raw Pointer Rewriting with LLMs for Translating C to Safer Rust 

**Title (ZH)**: PR2: 使用LLMs的崽孔原始指针重写以将C翻译为更安全的Rust 

**Authors**: Yifei Gao, Chengpeng Wang, Pengxiang Huang, Xuwei Liu, Mingwei Zheng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04852)  

**Abstract**: There has been a growing interest in translating C code to Rust due to Rust's robust memory and thread safety guarantees. Tools such as C2RUST enable syntax-guided transpilation from C to semantically equivalent Rust code. However, the resulting Rust programs often rely heavily on unsafe constructs--particularly raw pointers--which undermines Rust's safety guarantees. This paper aims to improve the memory safety of Rust programs generated by C2RUST by eliminating raw pointers. Specifically, we propose a peephole raw pointer rewriting technique that lifts raw pointers in individual functions to appropriate Rust data structures. Technically, PR2 employs decision-tree-based prompting to guide the pointer lifting process. Additionally, it leverages code change analysis to guide the repair of errors introduced during rewriting, effectively addressing errors encountered during compilation and test case execution. We implement PR2 as a prototype and evaluate it using gpt-4o-mini on 28 real-world C projects. The results show that PR2 successfully eliminates 13.22% of local raw pointers across these projects, significantly enhancing the safety of the translated Rust code. On average, PR2 completes the transformation of a project in 5.44 hours, at an average cost of $1.46. 

**Abstract (ZH)**: 基于决策树提示的无损指针重写技术以提升由C2RUST生成的Rust程序的内存安全性 

---
# Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards 

**Title (ZH)**: 基于 evolving leaderboards 的 LLM 忠实性在 RAG 中的基准测试 

**Authors**: Manveer Singh Tamber, Forrest Sheng Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04847)  

**Abstract**: Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce hallucinations by grounding responses in contexts. However, even when provided context, LLMs still frequently introduce unsupported information or contradictions. This paper presents our efforts to measure LLM hallucinations with a focus on summarization tasks, assessing how often various LLMs introduce hallucinations when summarizing documents. We discuss Vectara's existing LLM hallucination leaderboard, based on the Hughes Hallucination Evaluation Model (HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great research interest, we examine challenges faced by HHEM and current hallucination detection methods by analyzing the effectiveness of these methods on existing hallucination datasets. To address these limitations, we propose FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination annotations, which substantially improves automated LLM hallucination evaluation over current methods. We introduce an enhanced hallucination leaderboard centered on FaithJudge, alongside our current hallucination leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in RAG. 

**Abstract (ZH)**: LLMs中的幻觉仍然是一个持续的挑战。RAG通过将响应与上下文联系起来以减少幻觉。然而，即使提供了上下文，LLMs仍然经常引入未支持的信息或矛盾。本文旨在通过总结任务衡量LLMs的幻觉，评估各种LLMs在总结文档时引入幻觉的频率。我们讨论了Vectara现有的基于Hughes Hallucination Evaluation Model (HHEM)的LLM幻觉排行榜。虽然HHEM和Vectara的幻觉排行榜吸引了大量研究兴趣，但我们通过分析这些方法在现有幻觉数据集上的有效性，考察了HHEM和其他当前幻觉检测方法面临的一些挑战。为了解决这些限制，我们提出了FaithJudge，这是一种由少量人类幻觉注释引导的LLM作为裁判的方法，相较于当前的方法，FaithJudge显著提高了自动评估LLM幻觉的效果。我们引入了一个聚焦于FaithJudge的增强幻觉排行榜，并与当前的幻觉排行榜一起使用，为RAG中的LLM幻觉基准测试提供更可靠的指标。 

---
# Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers 

**Title (ZH)**: 将价值带回RL：通过统一LLM推理器与验证器提高测试时缩放效果 

**Authors**: Kusha Sareen, Morgane M Moss, Alessandro Sordoni, Rishabh Agarwal, Arian Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2505.04842)  

**Abstract**: Prevalent reinforcement learning~(RL) methods for fine-tuning LLM reasoners, such as GRPO or Leave-one-out PPO, abandon the learned value function in favor of empirically estimated returns. This hinders test-time compute scaling that relies on using the value-function for verification. In this work, we propose RL$^V$ that augments any ``value-free'' RL method by jointly training the LLM as both a reasoner and a generative verifier using RL-generated data, adding verification capabilities without significant overhead. Empirically, RL$^V$ boosts MATH accuracy by over 20\% with parallel sampling and enables $8-32\times$ efficient test-time compute scaling compared to the base RL method. RL$^V$ also exhibits strong generalization capabilities for both easy-to-hard and out-of-domain tasks. Furthermore, RL$^V$ achieves $1.2-1.6\times$ higher performance when jointly scaling parallel and sequential test-time compute with a long reasoning R1 model. 

**Abstract (ZH)**: 基于值的强化学习方法RL<sup>V</sup>用于增强LLM推理器的细调，同时显著减少计算开销 

---
# A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的聊天机器人操作风险评估Proposal 

**Authors**: Pedro Pinacho-Davidson, Fernando Gutierrez, Pablo Zapata, Rodolfo Vergara, Pablo Aqueveque  

**Link**: [PDF](https://arxiv.org/pdf/2505.04784)  

**Abstract**: The emergence of Generative AI (Gen AI) and Large Language Models (LLMs) has enabled more advanced chatbots capable of human-like interactions. However, these conversational agents introduce a broader set of operational risks that extend beyond traditional cybersecurity considerations. In this work, we propose a novel, instrumented risk-assessment metric that simultaneously evaluates potential threats to three key stakeholders: the service-providing organization, end users, and third parties. Our approach incorporates the technical complexity required to induce erroneous behaviors in the chatbot--ranging from non-induced failures to advanced prompt-injection attacks--as well as contextual factors such as the target industry, user age range, and vulnerability severity. To validate our metric, we leverage Garak, an open-source framework for LLM vulnerability testing. We further enhance Garak to capture a variety of threat vectors (e.g., misinformation, code hallucinations, social engineering, and malicious code generation). Our methodology is demonstrated in a scenario involving chatbots that employ retrieval-augmented generation (RAG), showing how the aggregated risk scores guide both short-term mitigation and longer-term improvements in model design and deployment. The results underscore the importance of multi-dimensional risk assessments in operationalizing secure, reliable AI-driven conversational systems. 

**Abstract (ZH)**: 生成式人工智能和大型语言模型的 emergence 使得更加先进的聊天机器人能够实现类人的交互。然而，这些对话代理引入了一套更广泛的操作风险，超出了传统的网络安全考量。在本研究中，我们提出了一种新颖的、可操作的风险评估指标，同时评估对服务提供组织、终端用户和第三方三个关键利益相关者的潜在威胁。我们的方法纳入了诱导聊天机器人错误行为所需的的技术复杂度，涵盖了从未诱导故障到高级提示注入攻击的各方面，同时考虑了目标行业、用户年龄范围和漏洞严重性等上下文因素。为了验证我们的指标，我们利用Garak，一个开源的大型语言模型漏洞测试框架。我们进一步增强了Garak以捕获各种威胁向量（如虚假信息、代码错觉、社会工程和恶意代码生成）。这种方法在涉及检索增强生成（RAG）的聊天机器人场景中得到了演示，展示了综合风险评分如何指导短期缓解措施和长时间内的模型设计与部署改进。研究结果强调了在实现安全可靠的AI驱动对话系统时进行多维度风险评估的重要性。 

---
# When Bad Data Leads to Good Models 

**Title (ZH)**: 当不良数据造就了良好模型 

**Authors**: Kenneth Li, Yida Chen, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.04741)  

**Abstract**: In large language model (LLM) pretraining, data quality is believed to determine model quality. In this paper, we re-examine the notion of "quality" from the perspective of pre- and post-training co-design. Specifically, we explore the possibility that pre-training on more toxic data can lead to better control in post-training, ultimately decreasing a model's output toxicity. First, we use a toy experiment to study how data composition affects the geometry of features in the representation space. Next, through controlled experiments with Olmo-1B models trained on varying ratios of clean and toxic data, we find that the concept of toxicity enjoys a less entangled linear representation as the proportion of toxic data increases. Furthermore, we show that although toxic data increases the generational toxicity of the base model, it also makes the toxicity easier to remove. Evaluations on Toxigen and Real Toxicity Prompts demonstrate that models trained on toxic data achieve a better trade-off between reducing generational toxicity and preserving general capabilities when detoxifying techniques such as inference-time intervention (ITI) are applied. Our findings suggest that, with post-training taken into account, bad data may lead to good models. 

**Abstract (ZH)**: 在大型语言模型（LLM）预训练中，数据质量被认为决定模型质量。本文从预训练和后训练协同设计的角度重新审视“质量”的概念。具体而言，我们探索了使用更具毒性的数据进行预训练是否能够在后训练中更好地控制模型输出的毒性，从而最终降低模型的输出毒性。首先，我们通过一个玩具实验研究数据组成如何影响表示空间中特征的几何结构。其次，通过使用Olmo-1B模型在不同比例的干净和有毒数据上进行训练的受控实验，我们发现随着有毒数据比例的增加，毒性概念在表示空间中享有更少纠缠的线性表示。此外，我们表明，尽管有毒数据增加了基模型生成的毒性，但它也使得去除毒性更加容易。在Toxigen和Real Toxicity Prompts上的评估表明，在应用推理时干预（ITI）等去毒技术时，使用有毒数据训练的模型在降低生成毒性与保留通用能力之间实现了更好的权衡。我们的研究结果表明，考虑到后训练因素，坏数据可能导致好模型。 

---
# QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort 

**Title (ZH)**: QBD-RankedDataGen: 生成定制排序数据集以利用LLM重排序改进查询-by-文档搜索，同时减少人工努力 

**Authors**: Sriram Gopalakrishnan, Sunandita Patra  

**Link**: [PDF](https://arxiv.org/pdf/2505.04732)  

**Abstract**: The Query-By-Document (QBD) problem is an information retrieval problem where the query is a document, and the retrieved candidates are documents that match the query document, often in a domain or query specific manner. This can be crucial for tasks such as patent matching, legal or compliance case retrieval, and academic literature review. Existing retrieval methods, including keyword search and document embeddings, can be optimized with domain-specific datasets to improve QBD search performance. However, creating these domain-specific datasets is often costly and time-consuming. Our work introduces a process to generate custom QBD-search datasets and compares a set of methods to use in this problem, which we refer to as QBD-RankedDatagen. We provide a comparative analysis of our proposed methods in terms of cost, speed, and the human interface with the domain experts. The methods we compare leverage Large Language Models (LLMs) which can incorporate domain expert input to produce document scores and rankings, as well as explanations for human review. The process and methods for it that we present can significantly reduce human effort in dataset creation for custom domains while still obtaining sufficient expert knowledge for tuning retrieval models. We evaluate our methods on QBD datasets from the Text Retrieval Conference (TREC) and finetune the parameters of the BM25 model -- which is used in many industrial-strength search engines like OpenSearch -- using the generated data. 

**Abstract (ZH)**: 基于文档的查询（QBD）问题是一种信息检索问题，其中查询是一个文档，检索的候选项是与查询文档匹配的文档，通常是在特定领域或查询的基础上进行匹配。这对于专利匹配、法律或合规案例检索以及学术文献回顾等任务至关重要。现有的检索方法，包括关键词搜索和文档嵌入，可以通过使用特定领域的数据集进行优化以提高QBD搜索性能。然而，创建这些特定领域的数据集通常成本高且耗时。我们的工作介绍了一种生成自定义QBD搜索数据集的过程，并比较了一系列在该问题中使用的方法，我们将其称为QBD-RankedDatagen。我们从成本、速度以及与领域专家的人机接口方面对提出的这些方法进行了比较分析。我们比较的方法利用了大型语言模型（LLMs），可以结合领域专家的输入来生成文档评分和排名，以及供人工审核的解释。我们提出的过程和方法可以显著减少在自定义领域创建数据集所需的人工努力，同时仍然能够获得足够的专家知识来调整检索模型。我们在Text Retrieval Conference (TREC)提供的QBD数据集上评估了我们的方法，并使用生成的数据对BM25模型进行了微调，该模型广泛应用于许多工业级搜索引擎，如OpenSearch。 

---
# REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM 

**Title (ZH)**: REVEAL: 图像输入危害的多轮评估 for 视觉LLM 

**Authors**: Madhur Jindal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2505.04673)  

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.
We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$). 

**Abstract (ZH)**: 视觉大型语言模型（VLLMs）：负责任的评估框架（REVEAL） 

---
# Personalized Risks and Regulatory Strategies of Large Language Models in Digital Advertising 

**Title (ZH)**: 大型语言模型在数字广告中的个性化风险与监管策略 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04665)  

**Abstract**: Although large language models have demonstrated the potential for personalized advertising recommendations in experimental environments, in actual operations, how advertising recommendation systems can be combined with measures such as user privacy protection and data security is still an area worthy of in-depth discussion. To this end, this paper studies the personalized risks and regulatory strategies of large language models in digital advertising. This study first outlines the principles of Large Language Model (LLM), especially the self-attention mechanism based on the Transformer architecture, and how to enable the model to understand and generate natural language text. Then, the BERT (Bidirectional Encoder Representations from Transformers) model and the attention mechanism are combined to construct an algorithmic model for personalized advertising recommendations and user factor risk protection. The specific steps include: data collection and preprocessing, feature selection and construction, using large language models such as BERT for advertising semantic embedding, and ad recommendations based on user portraits. Then, local model training and data encryption are used to ensure the security of user privacy and avoid the leakage of personal data. This paper designs an experiment for personalized advertising recommendation based on a large language model of BERT and verifies it with real user data. The experimental results show that BERT-based advertising push can effectively improve the click-through rate and conversion rate of advertisements. At the same time, through local model training and privacy protection mechanisms, the risk of user privacy leakage can be reduced to a certain extent. 

**Abstract (ZH)**: 大型语言模型在数字广告中的个性化风险与监管策略研究 

---
# Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs 

**Title (ZH)**: 自适应TOKEN边界：将人类段落划分机制集成到多模态LLMs中 

**Authors**: Dongxing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04637)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in processing diverse data types, yet significant disparities persist between human cognitive processes and computational approaches to multimodal information integration. This research presents a systematic investigation into the parallels between human cross-modal chunking mechanisms and token representation methodologies in MLLMs. Through empirical studies comparing human performance patterns with model behaviors across visual-linguistic tasks, we demonstrate that conventional static tokenization schemes fundamentally constrain current models' capacity to simulate the dynamic, context-sensitive nature of human information processing. We propose a novel framework for dynamic cross-modal tokenization that incorporates adaptive boundaries, hierarchical representations, and alignment mechanisms grounded in cognitive science principles. Quantitative evaluations demonstrate that our approach yields statistically significant improvements over state-of-the-art models on benchmark tasks (+7.8% on Visual Question Answering, +5.3% on Complex Scene Description) while exhibiting more human-aligned error patterns and attention distributions. These findings contribute to the theoretical understanding of the relationship between human cognition and artificial intelligence, while providing empirical evidence for developing more cognitively plausible AI systems. 

**Abstract (ZH)**: 最近在多模态大语言模型方面取得的进展展示了在处理多种数据类型方面的卓越能力，但人类认知过程与计算方法在多模态信息集成方面的差距依然显著。本研究系统探讨了人类跨模态片段化机制与多模态大语言模型中词元表示方法之间的相似性。通过比较人类和模型在视觉语言任务中的绩效模式，我们证明了传统的静态词元划分方案从根本上限制了当前模型模拟人类信息处理的动态性和上下文敏感性能力。我们提出了一种新的动态跨模态词元化框架，该框架结合了适应性边界、分层表示和基于认知科学原理的对齐机制。定量评估表明，我们的方法在基准任务中比最先进的模型表现出了统计意义上的显著提升（视觉问答任务上+7.8%，复杂场景描述任务上+5.3%），并且展现出更符合人类错误模式和注意力分布的特点。这些发现不仅丰富了人类认知与人工智能关系的理论理解，还为开发更加认知合理的AI系统提供了实证支持。 

---
# How Social is It? A Benchmark for LLMs' Capabilities in Multi-user Multi-turn Social Agent Tasks 

**Title (ZH)**: 多用户多轮社会代理任务能力基准：它有多社会？ 

**Authors**: Yusen Wu, Junwu Xiong, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04628)  

**Abstract**: Expanding the application of large language models (LLMs) to societal life, instead of primary function only as auxiliary assistants to communicate with only one person at a time, necessitates LLMs' capabilities to independently play roles in multi-user, multi-turn social agent tasks within complex social settings. However, currently the capability has not been systematically measured with available benchmarks. To address this gap, we first introduce an agent task leveling framework grounded in sociological principles. Concurrently, we propose a novel benchmark, How Social Is It (we call it HSII below), designed to assess LLM's social capabilities in comprehensive social agents tasks and benchmark representative models. HSII comprises four stages: format parsing, target selection, target switching conversation, and stable conversation, which collectively evaluate the communication and task completion capabilities of LLMs within realistic social interaction scenarios dataset, HSII-Dataset. The dataset is derived step by step from news dataset. We perform an ablation study by doing clustering to the dataset. Additionally, we investigate the impact of chain of thought (COT) method on enhancing LLMs' social performance. Since COT cost more computation, we further introduce a new statistical metric, COT-complexity, to quantify the efficiency of certain LLMs with COTs for specific social tasks and strike a better trade-off between measurement of correctness and efficiency. Various results of our experiments demonstrate that our benchmark is well-suited for evaluating social skills in LLMs. 

**Abstract (ZH)**: 扩展大型语言模型在社会生活中的应用，而非仅作为辅助助手与单一用户交流，要求LLMs具备在复杂社会环境中独立承担多用户、多轮社会智能代理任务的能力。然而，目前尚无系统化的基准来衡量这一能力。为填补这一空白，我们首先引入了一个基于社会学原理的任务层级框架。同时，我们提出了一种新的基准——How Social Is It（我们将其简称为HSII）——用于评估LLMs在综合社会智能代理任务中的社会能力，并以此作为代表性模型的基准。HSII包括四个阶段：格式解析、目标选择、目标切换对话和稳定对话，旨在评估LLMs在现实社会交互场景中的沟通和任务完成能力，数据集为HSII-Dataset。该数据集从新闻数据集逐步衍生而来。我们通过聚类分析对该数据集进行了消融研究，并探讨了思维链方法（COT）对提升LLMs社会表现的影响。由于COT消耗更多计算资源，我们进一步引入了一个新的统计指标——COT复杂性，以量化具有特定COT的LLMs在特定社会任务中的效率，并在正确性和效率之间找到更好的权衡。各种实验结果表明，我们的基准对评估LLMs的社会技能非常合适。 

---
