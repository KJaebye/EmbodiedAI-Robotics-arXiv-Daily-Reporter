# The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind 

**Title (ZH)**: 多智能体推理与心智理论的Decrypto基准 

**Authors**: Andrei Lupu, Timon Willi, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.20664)  

**Abstract**: As Large Language Models (LLMs) gain agentic abilities, they will have to navigate complex multi-agent scenarios, interacting with human users and other agents in cooperative and competitive settings. This will require new reasoning skills, chief amongst them being theory of mind (ToM), or the ability to reason about the "mental" states of other agents. However, ToM and other multi-agent abilities in LLMs are poorly understood, since existing benchmarks suffer from narrow scope, data leakage, saturation, and lack of interactivity. We thus propose Decrypto, a game-based benchmark for multi-agent reasoning and ToM drawing inspiration from cognitive science, computational pragmatics and multi-agent reinforcement learning. It is designed to be as easy as possible in all other dimensions, eliminating confounding factors commonly found in other benchmarks. To our knowledge, it is also the first platform for designing interactive ToM experiments.
We validate the benchmark design through comprehensive empirical evaluations of frontier LLMs, robustness studies, and human-AI cross-play experiments. We find that LLM game-playing abilities lag behind humans and simple word-embedding baselines. We then create variants of two classic cognitive science experiments within Decrypto to evaluate three key ToM abilities. Surprisingly, we find that state-of-the-art reasoning models are significantly worse at those tasks than their older counterparts. This demonstrates that Decrypto addresses a crucial gap in current reasoning and ToM evaluations, and paves the path towards better artificial agents. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）获得自主能力，它们将不得不在复杂多智能体场景中导航，与人类用户和其他智能体在合作和竞争环境中互动。这将需要新的推理能力，其中最主要的是心理理论（ToM），即推理其他智能体“心理”状态的能力。然而，LLMs中的ToM和其他多智能体能力尚未充分理解，因为现有基准测试存在范围狭窄、数据泄露、饱和和缺乏互动性等问题。因此，我们提出了Decrypto，这是一种借鉴认知科学、计算语用学和多智能体强化学习的基于游戏的基准测试，设计上在所有其他维度尽可能简单，消除其他基准测试中常见的混淆因素。据我们所知，这也是首个用于设计互动心理理论实验的平台。 

---
# Towards Community-Driven Agents for Machine Learning Engineering 

**Title (ZH)**: 面向社区驱动的机器学习工程代理 

**Authors**: Sijie Li, Weiwei Sun, Shanda Li, Ameet Talwalkar, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20640)  

**Abstract**: Large language model-based machine learning (ML) agents have shown great promise in automating ML research. However, existing agents typically operate in isolation on a given research problem, without engaging with the broader research community, where human researchers often gain insights and contribute by sharing knowledge. To bridge this gap, we introduce MLE-Live, a live evaluation framework designed to assess an agent's ability to communicate with and leverage collective knowledge from a simulated Kaggle research community. Building on this framework, we propose CoMind, a novel agent that excels at exchanging insights and developing novel solutions within a community context. CoMind achieves state-of-the-art performance on MLE-Live and outperforms 79.2% human competitors on average across four ongoing Kaggle competitions. Our code is released at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的机器学习代理在自动化机器学习研究方面展现了巨大的潜力。然而，现有的代理通常孤立地在特定研究问题上操作，而不与更广泛的科研社区互动，人类研究人员常常通过分享知识来获取见解和贡献。为弥合这一差距，我们介绍了一种实时评估框架MLE-Live，旨在评估代理与模拟的Kaggle研究社区进行沟通并利用集体知识的能力。在此基础上，我们提出了一种名为CoMind的新型代理，它在社区背景下擅长交换见解并发展新颖的解决方案。CoMind在MLE-Live上达到了最先进的性能，并且在四项正在进行的Kaggle竞赛中平均优于79.2%的人类对手。我们的代码发布在该网址：https://github.com/alibaba/CoMind。 

---
# AI Assistants to Enhance and Exploit the PETSc Knowledge Base 

**Title (ZH)**: AI助手以增强和利用PETSc知识库 

**Authors**: Barry Smith, Junchao Zhang, Hong Zhang, Lois Curfman McInnes, Murat Keceli, Archit Vasan, Satish Balay, Toby Isaac, Le Chen, Venkatram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2506.20608)  

**Abstract**: Generative AI, especially through large language models (LLMs), is transforming how technical knowledge can be accessed, reused, and extended. PETSc, a widely used numerical library for high-performance scientific computing, has accumulated a rich but fragmented knowledge base over its three decades of development, spanning source code, documentation, mailing lists, GitLab issues, Discord conversations, technical papers, and more. Much of this knowledge remains informal and inaccessible to users and new developers. To activate and utilize this knowledge base more effectively, the PETSc team has begun building an LLM-powered system that combines PETSc content with custom LLM tools -- including retrieval-augmented generation (RAG), reranking algorithms, and chatbots -- to assist users, support developers, and propose updates to formal documentation. This paper presents initial experiences designing and evaluating these tools, focusing on system architecture, using RAG and reranking for PETSc-specific information, evaluation methodologies for various LLMs and embedding models, and user interface design. Leveraging the Argonne Leadership Computing Facility resources, we analyze how LLM responses can enhance the development and use of numerical software, with an initial focus on scalable Krylov solvers. Our goal is to establish an extensible framework for knowledge-centered AI in scientific software, enabling scalable support, enriched documentation, and enhanced workflows for research and development. We conclude by outlining directions for expanding this system into a robust, evolving platform that advances software ecosystems to accelerate scientific discovery. 

**Abstract (ZH)**: Generative AI，尤其是在大型语言模型（LLMs）的帮助下，正在 transforming 如何获取、重用和技术扩展技术知识。PETSc，一个广泛用于高性能科学计算的数值库，在其三十余年的开发过程中积累了丰富的但又分散的知识基础，涵盖源代码、文档、邮件列表、GitLab问题、Discord对话、技术论文等。其中大量知识仍处于非正式状态且难以为用户和新开发者访问。为更有效地激活和利用这一知识基础，PETSc 团队已经开始构建一个基于 LLM 的系统，将 PETSc 内容与自定义 LLM 工具相结合——包括检索增强生成（RAG）、重排序算法和聊天机器人——以协助用户、支持开发人员，并提议更新正式文档。本文介绍了这些工具的初步设计与评估经验，重点在于系统架构，利用 RAG 和重排序处理 PETSc 特定信息，各种 LLM 和嵌入式模型的评估方法，以及用户界面设计。利用阿贡领导力计算设施的资源，我们分析了 LLM 响应如何增强数值软件的开发和使用，初期重点放在可扩展的共轭梯度求解器上。我们的目标是建立一个可扩展的知识中心型 AI 框架，以促进科学软件的支持、丰富文档和增强的研究与开发工作流。最后，我们概括了扩展该系统的方向，以便构建一个稳健且不断演进的平台，从而推动软件生态系统的发展，加快科学研究的发现过程。 

---
# Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges 

**Title (ZH)**: 细调和指令工程化大语言模型，以创建多智能体AI应对可持续蛋白质生产挑战 

**Authors**: Alexander D. Kalian, Jaewook Lee, Stefan P. Johannesson, Lennart Otte, Christer Hogstrand, Miao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.20598)  

**Abstract**: The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities 

**Abstract (ZH)**: 全球可持续蛋白质来源的需求加速了智能工具的发展，这些工具可以快速处理和合成特定领域的科学知识。本研究介绍了一种概念验证的多代理人工智能（AI）框架，旨在支持可持续蛋白质生产研究，初期重点关注微生物蛋白质来源。基于检索增强生成（RAG）的系统包括两个基于GPT的大语言模型代理：（1）文献搜索代理，用于检索指定微生物菌株的微生物蛋白质生产相关科学文献；（2）信息提取代理，处理检索内容以提取相关生物和化学信息。探索了两种并行方法——微调和提示工程——以优化代理性能。两种方法在基于变压器的余弦相似度评分上均显示出有效性，其中检索到的内容与理想输出之间的相似度得分最高提高了25%，普遍达到≥0.89。总体而言，微调方法在平均得分上的改进程度更大（≥0.94），尽管与之相比，提示工程方法的统计不确定性更低。开发并发布了用户界面，以启用多代理AI系统的使用，并初步探索了额外的基于化学安全的搜索功能。

标题：
基于RAG的多代理AI框架支持微生物蛋白质生产的研究 

---
# Case-based Reasoning Augmented Large Language Model Framework for Decision Making in Realistic Safety-Critical Driving Scenarios 

**Title (ZH)**: 基于案例推理增强的大语言模型决策框架在实际安全关键驾驶场景中的应用 

**Authors**: Wenbin Gan, Minh-Son Dao, Koji Zettsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20531)  

**Abstract**: Driving in safety-critical scenarios requires quick, context-aware decision-making grounded in both situational understanding and experiential reasoning. Large Language Models (LLMs), with their powerful general-purpose reasoning capabilities, offer a promising foundation for such decision-making. However, their direct application to autonomous driving remains limited due to challenges in domain adaptation, contextual grounding, and the lack of experiential knowledge needed to make reliable and interpretable decisions in dynamic, high-risk environments. To address this gap, this paper presents a Case-Based Reasoning Augmented Large Language Model (CBR-LLM) framework for evasive maneuver decision-making in complex risk scenarios. Our approach integrates semantic scene understanding from dashcam video inputs with the retrieval of relevant past driving cases, enabling LLMs to generate maneuver recommendations that are both context-sensitive and human-aligned. Experiments across multiple open-source LLMs show that our framework improves decision accuracy, justification quality, and alignment with human expert behavior. Risk-aware prompting strategies further enhance performance across diverse risk types, while similarity-based case retrieval consistently outperforms random sampling in guiding in-context learning. Case studies further demonstrate the framework's robustness in challenging real-world conditions, underscoring its potential as an adaptive and trustworthy decision-support tool for intelligent driving systems. 

**Abstract (ZH)**: 基于案例推理增强的大语言模型在复杂风险场景中的回避 maneuvers决策框架 

---
# Paladin-mini: A Compact and Efficient Grounding Model Excelling in Real-World Scenarios 

**Title (ZH)**: Paladin-mini：一个紧凑高效且适用于实际场景的语义 grounding 模型 

**Authors**: Dror Ivry, Oran Nahum  

**Link**: [PDF](https://arxiv.org/pdf/2506.20384)  

**Abstract**: This paper introduces two significant contributions to address the issue of grounding claims in a given context. Grounding means that given a context (document) and a claim, there's at least one supportive evidence for the claim in the document. We will introduce Paladin-mini, a compact (3.8B parameters) open-source classifier model (used for labeling data as grounded or ungrounded) engineered for robust performance in real-world scenarios, and the grounding-benchmark, a new evaluation dataset designed to assess performance on critical reasoning tasks. We'll also demonstrate the results of Paladin-mini with benchmarks against the current State-of-the-art and share clear and reproducible results. 

**Abstract (ZH)**: 本文介绍了两项重要贡献，以解决在给定上下文中接地断言的问题。接地意味着给定一个上下文（文档）和一个断言，在文档中至少有一条支持证据。我们将介绍Paladin-mini，这是一个紧凑型（3.8B参数）开源分类模型（用于将数据标记为接地或未接地），它旨在在实际场景中实现稳健性能，并介绍用于评估关键推理任务性能的新评价数据集grounding-benchmark。我们还将展示Paladin-mini与当前最佳性能的基准测试结果，并分享清晰可重复的结果。 

---
# Tabular Feature Discovery With Reasoning Type Exploration 

**Title (ZH)**: 表格特征发现与推理类型探索 

**Authors**: Sungwon Han, Sungkyu Park, Seungeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20357)  

**Abstract**: Feature engineering for tabular data remains a critical yet challenging step in machine learning. Recently, large language models (LLMs) have been used to automatically generate new features by leveraging their vast knowledge. However, existing LLM-based approaches often produce overly simple or repetitive features, partly due to inherent biases in the transformations the LLM chooses and the lack of structured reasoning guidance during generation. In this paper, we propose a novel method REFeat, which guides an LLM to discover diverse and informative features by leveraging multiple types of reasoning to steer the feature generation process. Experiments on 59 benchmark datasets demonstrate that our approach not only achieves higher predictive accuracy on average, but also discovers more diverse and meaningful features. These results highlight the promise of incorporating rich reasoning paradigms and adaptive strategy selection into LLM-driven feature discovery for tabular data. 

**Abstract (ZH)**: 基于大规模语言模型的特征工程方法通过多种推理引导发现多样化和有信息量的特征 

---
# Enterprise Large Language Model Evaluation Benchmark 

**Title (ZH)**: 企业级大型语言模型评估基准 

**Authors**: Liya Wang, David Yi, Damien Jose, John Passarelli, James Gao, Jordan Leventis, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.20274)  

**Abstract**: Large Language Models (LLMs) ) have demonstrated promise in boosting productivity across AI-powered tools, yet existing benchmarks like Massive Multitask Language Understanding (MMLU) inadequately assess enterprise-specific task complexities. We propose a 14-task framework grounded in Bloom's Taxonomy to holistically evaluate LLM capabilities in enterprise contexts. To address challenges of noisy data and costly annotation, we develop a scalable pipeline combining LLM-as-a-Labeler, LLM-as-a-Judge, and corrective retrieval-augmented generation (CRAG), curating a robust 9,700-sample benchmark. Evaluation of six leading models shows open-source contenders like DeepSeek R1 rival proprietary models in reasoning tasks but lag in judgment-based scenarios, likely due to overthinking. Our benchmark reveals critical enterprise performance gaps and offers actionable insights for model optimization. This work provides enterprises a blueprint for tailored evaluations and advances practical LLM deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在提升AI工具生产力方面展示了潜力，但现有的基准测试如大规模多任务语言理解（MMLU）未能充分评估企业特定的任务复杂性。我们提出一个基于布卢姆分类法的14任务框架，以全面评估LLM在企业环境中的能力。为应对嘈杂数据和昂贵标注的挑战，我们开发了一个可扩展的流水线，结合了LLM-as-a-标签器、LLM-as-a-裁判和纠正检索增强生成（CRAG）技术，构建了一个包含9,700个样本的坚实基准。对六种领先模型的评估显示，开源竞争者如DeepSeek R1在推理任务中媲美专有模型，但在基于判断的场景中表现较弱，可能是因为过度思考。我们的基准揭示了关键的企业性能缺口，并提供了可操作的模型优化建议。本研究为企业提供了一个定制评估的蓝图，并推动了实用的LLM部署。 

---
# Language Modeling by Language Models 

**Title (ZH)**: 语言模型_BY_语言模型 

**Authors**: Junyan Cheng, Peter Clark, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20249)  

**Abstract**: Can we leverage LLMs to model the process of discovering novel language model (LM) architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system, Genesys, employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M$\sim$350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., $\sim$86\% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous discovery systems. 

**Abstract (ZH)**: 能否利用大语言模型来建模发现新型语言模型架构的过程？受真实研究的启发，我们提出了一种多智能体大语言模型方法，模拟了传统研究阶段，从理念构思和文献搜索（提案阶段）到设计实现（代码生成）、生成预训练和下游评估（验证）。借鉴扩展律的思想，我们的系统Genesys采用阶梯尺度方法；新设计被提出、对抗性审核、实施，并在越来越大的模型规模（14M至350M参数）下选择性验证，预算逐渐减少（每个规模下我们可以训练的模型数量）。为了提高发现的效率和可分解性，Genesys使用了一种新型的遗传编程架构，我们证明其在实验上比常用的直接提示生成工作流（例如，成功设计生成的百分点提升约86%）具有优势。我们报道了涉及1,162个新发现的设计（1,062个通过预训练完全验证）的实验，发现最优设计与已知架构具有很强的竞争力（例如，在6/9个常见基准上优于GPT2、Mamba2等）。我们结合了全面的系统级消融实验和正式结果，这为设计有效的自主发现系统提供了更广泛的见解。 

---
# DiaLLMs: EHR Enhanced Clinical Conversational System for Clinical Test Recommendation and Diagnosis Prediction 

**Title (ZH)**: DiaLLMs: 基于EHR的临床对话系统，用于临床检测推荐和诊断预测 

**Authors**: Weijieying Ren, Tianxiang Zhao, Lei Wang, Tianchun Wang, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20059)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to remarkable progresses in medical consultation. However, existing medical LLMs overlook the essential role of Electronic Health Records (EHR) and focus primarily on diagnosis recommendation, limiting their clinical applicability. We propose DiaLLM, the first medical LLM that integrates heterogeneous EHR data into clinically grounded dialogues, enabling clinical test recommendation, result interpretation, and diagnosis prediction to better align with real-world medical practice. To construct clinically grounded dialogues from EHR, we design a Clinical Test Reference (CTR) strategy that maps each clinical code to its corresponding description and classifies test results as "normal" or "abnormal". Additionally, DiaLLM employs a reinforcement learning framework for evidence acquisition and automated diagnosis. To handle the large action space, we introduce a reject sampling strategy to reduce redundancy and improve exploration efficiency. Furthermore, a confirmation reward and a class-sensitive diagnosis reward are designed to guide accurate diagnosis prediction. Extensive experimental results demonstrate that DiaLLM outperforms baselines in clinical test recommendation and diagnosis prediction. 

**Abstract (ZH)**: Recent Advances in Large Language Models (LLMs) Have Led to Remarkable Progresses in Medical Consultation: DiaLLM, the First Medical LLM That Integrates Heterogeneous EHR Data into Clinically Grounded Dialogues 

---
# Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning 

**Title (ZH)**: 个性化赋权的大语言模型显示出类似人类的动力型推理。 

**Authors**: Saloni Dash, Amélie Reymond, Emma S. Spiro, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20020)  

**Abstract**: Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans. 

**Abstract (ZH)**: 大型语言模型受人格化身份动机影响的归纳推理：实证研究 

---
# Achieving Trustworthy Real-Time Decision Support Systems with Low-Latency Interpretable AI Models 

**Title (ZH)**: 实现低延迟可解释人工智能模型以建立可信的实时决策支持系统 

**Authors**: Zechun Deng, Ziwei Liu, Ziqian Bi, Junhao Song, Chia Xin Liang, Joe Yeong, Junfeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20018)  

**Abstract**: This paper investigates real-time decision support systems that leverage low-latency AI models, bringing together recent progress in holistic AI-driven decision tools, integration with Edge-IoT technologies, and approaches for effective human-AI teamwork. It looks into how large language models can assist decision-making, especially when resources are limited. The research also examines the effects of technical developments such as DeLLMa, methods for compressing models, and improvements for analytics on edge devices, while also addressing issues like limited resources and the need for adaptable frameworks. Through a detailed review, the paper offers practical perspectives on development strategies and areas of application, adding to the field by pointing out opportunities for more efficient and flexible AI-supported systems. The conclusions set the stage for future breakthroughs in this fast-changing area, highlighting how AI can reshape real-time decision support. 

**Abstract (ZH)**: 本文探讨了利用低延迟AI模型的实时决策支持系统，综述了整体AI驱动决策工具的进步、Edge-IoT技术的集成以及有效的人机团队合作方法。研究了大型语言模型在资源受限条件下如何辅助决策，并探讨了如DeLLMa等技术发展、模型压缩方法以及边缘设备上数据分析改进的效果，同时关注资源限制和可适应框架的需求。通过详细综述，本文提供了开发策略和应用领域的实用观点，为更高效和灵活的AI支持系统指出了机会，并为这一快速变化领域未来的突破奠定了基础，强调了AI如何重塑实时决策支持。 

---
# Accurate and Energy Efficient: Local Retrieval-Augmented Generation Models Outperform Commercial Large Language Models in Medical Tasks 

**Title (ZH)**: 准确且节能：局部检索增强生成模型在医疗任务中优于商用大型语言模型 

**Authors**: Konstantinos Vrettos, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20009)  

**Abstract**: Background The increasing adoption of Artificial Intelligence (AI) in healthcare has sparked growing concerns about its environmental and ethical implications. Commercial Large Language Models (LLMs), such as ChatGPT and DeepSeek, require substantial resources, while the utilization of these systems for medical purposes raises critical issues regarding patient privacy and safety. Methods We developed a customizable Retrieval-Augmented Generation (RAG) framework for medical tasks, which monitors its energy usage and CO2 emissions. This system was then used to create RAGs based on various open-source LLMs. The tested models included both general purpose models like llama3.1:8b and medgemma-4b-it, which is medical-domain specific. The best RAGs performance and energy consumption was compared to DeepSeekV3-R1 and OpenAIs o4-mini model. A dataset of medical questions was used for the evaluation. Results Custom RAG models outperformed commercial models in accuracy and energy consumption. The RAG model built on llama3.1:8B achieved the highest accuracy (58.5%) and was significantly better than other models, including o4-mini and DeepSeekV3-R1. The llama3.1-RAG also exhibited the lowest energy consumption and CO2 footprint among all models, with a Performance per kWh of 0.52 and a total CO2 emission of 473g. Compared to o4-mini, the llama3.1-RAG achieved 2.7x times more accuracy points per kWh and 172% less electricity usage while maintaining higher accuracy. Conclusion Our study demonstrates that local LLMs can be leveraged to develop RAGs that outperform commercial, online LLMs in medical tasks, while having a smaller environmental impact. Our modular framework promotes sustainable AI development, reducing electricity usage and aligning with the UNs Sustainable Development Goals. 

**Abstract (ZH)**: 背景：人工智能（AI）在医疗领域的广泛应用引发了对其环境和伦理影响的关注。商用大语言模型（LLMs），如ChatGPT和DeepSeek，需要大量资源，而这些系统在医疗领域的应用则引发了关于患者隐私和安全的 critical 问题。

方法：我们开发了一种可定制的检索增强生成（RAG）框架，用于医疗任务，该框架监控其能源使用和二氧化碳排放。然后，利用该系统基于各种开源LLMs创建了RAGs。测试的模型包括通用模型如llama3.1:8b和针对医疗领域的medgemma-4b-it。最佳RAG性能和能源消耗与DeepSeekV3-R1和OpenAI的o4-mini模型进行了比较。使用医疗问题数据集进行了评估。

结果：定制的RAG模型在准确性和能源消耗方面优于商用模型。基于llama3.1:8B构建的RAG模型达到了最高准确率（58.5%），并显著优于其他模型，包括o4-mini和DeepSeekV3-R1。llama3.1-RAG也是所有模型中能源消耗和二氧化碳足迹最低的，每千瓦时性能为0.52，总二氧化碳排放量为473克。与o4-mini相比，llama3.1-RAG在每千瓦时获得了2.7倍的准确率点，并减少了172%的电能使用，同时保持了更高的准确率。

结论：我们的研究表明，利用本地LLMs可以开发出在医疗任务中优于商用在线LLMs的RAGs，同时对环境影响较小。我们的模块化框架促进了可持续的人工智能开发，减少了电能使用，并与联合国可持续发展目标一致。 

---
# QHackBench: Benchmarking Large Language Models for Quantum Code Generation Using PennyLane Hackathon Challenges 

**Title (ZH)**: QHackBench：基于PennyLane黑客马拉松挑战任务的大型语言模型量子代码生成性能评测 

**Authors**: Abdul Basit, Minghao Shao, Haider Asif, Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2506.20008)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated strong potential in code generation, yet their effectiveness in quantum computing remains underexplored. This paper benchmarks LLMs for PennyLane-based quantum code generation using real-world challenges from the Quantum Hackathon (QHack). We introduce QHackBench, a novel benchmark dataset derived from QHack competitions, and evaluate model performance under vanilla prompting and Retrieval-Augmented Generation (RAG). Our structured evaluation framework assesses functional correctness, syntactic validity, and execution success across varying challenge difficulties. Results indicate that RAG-enhanced models, supplemented with an augmented PennyLane dataset, approximately generate similar results as the standard prompting, particularly in complex quantum algorithms. Additionally, we introduce a multi-agent evaluation pipeline that iteratively refines incorrect solutions, further enhancing execution success rates. To foster further research, we commit to publicly releasing QHackBench, along with our evaluation framework and experimental results, enabling continued advancements in AI-assisted quantum programming. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在代码生成领域展示了强大的潜力，但在量子计算领域的有效性仍鲜有探索。本文使用Quantum Hackathon（QHack）中的真实挑战，评估PennyLane为基础的量子代码生成的大型语言模型。我们引入了QHackBench，一个基于QHack竞赛的新基准数据集，并对其在vanilla提示和检索增强生成（RAG）下的模型性能进行了评估。我们的结构化评估框架评估了功能正确性、语法有效性以及在不同挑战难度下的执行成功率。结果表明，结合增强PennyLane数据集的RAG增强模型，生成的结果与标准提示方法接近，尤其是在复杂量子算法中。此外，我们提出了一个迭代校正不正确解的多智能体评估流水线，进一步提高了执行成功率。为促进进一步研究，我们承诺公开发布QHackBench，以及我们的评估框架和实验结果，推动AI辅助量子编程的持续发展。 

---
# Context Attribution with Multi-Armed Bandit Optimization 

**Title (ZH)**: 多臂老虎机优化的上下文归因 

**Authors**: Deng Pan, Keerthiram Murugesan, Nuno Moniz, Nitesh Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2506.19977)  

**Abstract**: Understanding which parts of the retrieved context contribute to a large language model's generated answer is essential for building interpretable and trustworthy generative QA systems. We propose a novel framework that formulates context attribution as a combinatorial multi-armed bandit (CMAB) problem. Each context segment is treated as a bandit arm, and we employ Combinatorial Thompson Sampling (CTS) to efficiently explore the exponentially large space of context subsets under a limited query budget. Our method defines a reward function based on normalized token likelihoods, capturing how well a subset of segments supports the original model response. Unlike traditional perturbation-based attribution methods such as SHAP, which sample subsets uniformly and incur high computational costs, our approach adaptively balances exploration and exploitation by leveraging posterior estimates of segment relevance. This leads to substantially improved query efficiency while maintaining high attribution fidelity. Extensive experiments on diverse datasets and LLMs demonstrate that our method achieves competitive attribution quality with fewer model queries. 

**Abstract (ZH)**: 理解检索到的上下文哪些部分对大型语言模型生成的答案有贡献对于构建可解释和可信赖的生成式问答系统至关重要。我们提出了一种新的框架，将上下文归因问题形式化为组合多臂bandit (CMAB) 问题。每个上下文片段被视为一个bandit臂，并采用组合 Thompson 抽样 (CTS) 有效地在有限的查询预算下探索上下文子集的指数级空间。我们的方法基于归一化令牌可能性定义奖励函数，捕捉片段子集如何支持原始模型响应。与传统的基于扰动的归因方法（如SHAP），后者均匀抽样子集并产生高计算成本不同，我们的方法通过利用片段相关性的后验估计，适应性地平衡探索与利用。这在提高查询效率的同时保持了高归因保真度。在多样化的数据集和LLMs上的广泛实验表明，我们的方法在较少的模型查询下实现与传统方法相当的归因质量。 

---
# Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs 

**Title (ZH)**: 内在你中有许多狼：使用认知模型解释LLMs的价值权衡 

**Authors**: Sonia K. Murthy, Rosie Zhao, Jennifer Hu, Sham Kakade, Markus Wulfmeier, Peng Qian, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20666)  

**Abstract**: Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training. 

**Abstract (ZH)**: 导航日常社交情境往往需要权衡相互冲突的目标，如传达严峻的事实、维持信任等方面，同时还要考虑到对方的感受。然而，当下的工具在解释此类动态且多维度的价值观念时仍是有限的。在认知科学中，所谓的“认知模型”通过建模说话人在选择行动或言语时竞争的利益函数的权重，提供人类这些权衡的正式解释。在本工作中，我们利用礼貌言语的认知模型来解释LLMs在人类样式的权衡中的表现程度。我们将这一视角应用于系统地评估两种广泛的模型设置中的价值权衡：推理“努力”程度的等级在前沿的黑盒模型中，以及开放源代码模型的强化学习后训练动态。我们的结果强调了推理模型中的信息效用高于社会效用的模式，并且在显示出更强数学推理能力的开放源代码模型中也是如此。从LLMs的训练动态中，我们的发现表明，大型模型在训练早期发生显著的效用值转变，这些效果持续影响基础模型和预训练数据的选择，相对于反馈数据集或对齐方法而言。我们展示了我们的方法能够响应快速变化的LLM景观中的各种方面，并为形成其他高层次行为的假设、塑造推理模型的训练制度以及更好地控制模型训练中的价值权衡提供了见解。 

---
# Large Language Model-Driven Code Compliance Checking in Building Information Modeling 

**Title (ZH)**: 大型语言模型驱动的建筑信息建模代码合规性检查 

**Authors**: Soumya Madireddy, Lu Gao, Zia Din, Kinam Kim, Ahmed Senouci, Zhe Han, Yunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20551)  

**Abstract**: This research addresses the time-consuming and error-prone nature of manual code compliance checking in Building Information Modeling (BIM) by introducing a Large Language Model (LLM)-driven approach to semi-automate this critical process. The developed system integrates LLMs such as GPT, Claude, Gemini, and Llama, with Revit software to interpret building codes, generate Python scripts, and perform semi-automated compliance checks within the BIM environment. Case studies on a single-family residential project and an office building project demonstrated the system's ability to reduce the time and effort required for compliance checks while improving accuracy. It streamlined the identification of violations, such as non-compliant room dimensions, material usage, and object placements, by automatically assessing relationships and generating actionable reports. Compared to manual methods, the system eliminated repetitive tasks, simplified complex regulations, and ensured reliable adherence to standards. By offering a comprehensive, adaptable, and cost-effective solution, this proposed approach offers a promising advancement in BIM-based compliance checking, with potential applications across diverse regulatory documents in construction projects. 

**Abstract (ZH)**: 通过大型语言模型驱动的半自动化方法解决建筑信息建模（BIM）中手动代码合规检查的时间 consuming 和易出错性质 

---
# When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs 

**Title (ZH)**: Life给我们带来了样本：扩大推理计算规模对多语言LLM的益处 

**Authors**: Ammar Khairi, Daniel D'souza, Ye Shen, Julia Kreutzer, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20544)  

**Abstract**: Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting.
Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages. 

**Abstract (ZH)**: recent 进展大语言模型 (LLMs)的最新进展已将重点转向扩展推断时的计算能力，通过温度变化调整采样策略和选择策略以适应多语言和多任务场景，而不必重新训练模型。当前工作研究如何在多语言、多任务设置中稳健地扩展开放生成任务的推断计算能力。 

---
# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling 

**Title (ZH)**: OctoThinker: 中断训练激励强化学习扩展 

**Authors**: Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20512)  

**Abstract**: Different base language model families, such as Llama and Qwen, exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy, Stable-then-Decay, in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max). 

**Abstract (ZH)**: 不同的基础语言模型家族（如Llama和Qwen）在强化学习（RL）微调阶段表现出不同的行为，尤其是在推理密集型任务上。什么是适合强化学习的基础语言模型？深入了解这一问题是开发下一代RL可扩展基础模型的关键。在本工作中，我们探讨了中期训练策略如何影响RL动态，重点关注两种代表性的模型家族：Qwen和Llama。我们的研究揭示了以下几点：（1）高质量的数学语料库（如MegaMath-Web-Pro）能显著提升基础模型和RL性能，而现有的替代方案（如FineMath-4plus）则未能达到这一效果；（2）进一步加入问答风格的数据，尤其是长链推理例子，能增强RL效果，而指令数据进一步解锁了这一效果；（3）虽然长链推理能提升推理深度，但也可能导致模型响应的冗长和RL训练的不稳定性，强调了数据格式化的重要性；（4）中期训练的扩量一致地提升了下游RL性能。基于这些见解，我们提出了一个两阶段中期训练策略，Stable-then-Decay，首先在200B tokens上以恒定的学习率训练基础模型，然后在三个链推理（CoT）重点分支上用学习率衰减训练20B tokens。这产生了OctoThinker家族模型，该家族模型展现出强大的RL兼容性，并与更友好的RL模型家族（如Qwen）缩小了性能差距。我们希望这项工作能帮助塑造RL时代的预训练策略。为支持进一步研究，我们发布了开源模型以及一个超过700亿个令牌的精选数学推理语料库（即MegaMath-Web-Pro-Max）。 

---
# ReCode: Updating Code API Knowledge with Reinforcement Learning 

**Title (ZH)**: ReCode: 使用强化学习更新代码API知识 

**Authors**: Haoze Wu, Yunzhi Yao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20495)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成方面表现出色，但在应对外部库API频繁更新时却表现不佳。这一关键限制源自对训练数据中过时API知识的依赖，即使有当前文档可供参考，也阻碍了在动态环境中的可靠代码生成。为了解决这一问题，我们提出了ReCode（基于规则的强化学习代码更新框架），该框架模拟了人类程序员适应API变更的过程。具体来说，我们构建了一个包含约2,000条数据条目的数据集，用于训练LLMs根据更新信息执行版本迁移。然后，我们引入了一种修改后的字符串相似度度量作为强化学习的奖励标准。我们的实验表明，ReCode在动态API场景中显著提升了LLMs的代码生成性能，特别是在未见过的CodeUpdateArena任务上。关键的是，与监督微调相比，ReCode对LLMs的一般代码生成能力影响较小。我们在多种LLMs和强化学习算法（GRPO和DAPO）上应用了ReCode，所有模型都取得了一致的改进。值得注意的是，经过训练后，Qwen2.5-Coder-7B的表现超过了参数量为32B的代码指令微调模型和具有相同架构的推理模型。代码可在以下链接获取。 

---
# Automatic Demonstration Selection for LLM-based Tabular Data Classification 

**Title (ZH)**: 基于LLM的表格数据分类的自动演示文稿选择 

**Authors**: Shuchu Han, Wolfgang Bruckner  

**Link**: [PDF](https://arxiv.org/pdf/2506.20451)  

**Abstract**: A fundamental question in applying In-Context Learning (ICL) for tabular data classification is how to determine the ideal number of demonstrations in the prompt. This work addresses this challenge by presenting an algorithm to automatically select a reasonable number of required demonstrations. Our method distinguishes itself by integrating not only the tabular data's distribution but also the user's selected prompt template and the specific Large Language Model (LLM) into its estimation. Rooted in Spectral Graph Theory, our proposed algorithm defines a novel metric to quantify the similarities between different demonstrations. We then construct a similarity graph and analyze the eigenvalues of its Laplacian to derive the minimum number of demonstrations capable of representing the data within the LLM's intrinsic representation space. We validate the efficacy of our approach through experiments comparing its performance against conventional random selection algorithms on diverse datasets and LLMs. 

**Abstract (ZH)**: 应用In-Context Learning（ICL）进行表格数据分类时的一个基本问题是如何确定提示中所需演示的数量。本项工作通过提出一种算法来自动选择合理数量的演示来应对这一挑战。该方法不仅考虑表格数据的分布，还结合用户的选定提示模板和特定的大语言模型（LLM）来进行估算。基于谱图理论，我们提出的算法定义了一个新的度量来量化不同演示之间的相似性。然后构建相似性图，并分析其拉普拉斯矩阵的特征值，以推导出能够在LLM固有表示空间中代表数据的最小演示数量。通过在不同数据集和LLM上将我们的方法与传统随机选择算法的性能进行比较实验，验证了我们方法的有效性。 

---
# An Agentic System for Rare Disease Diagnosis with Traceable Reasoning 

**Title (ZH)**: 罕见疾病诊断的可追溯推理代理系统 

**Authors**: Weike Zhao, Chaoyi Wu, Yanjie Fan, Xiaoman Zhang, Pengcheng Qiu, Yuze Sun, Xiao Zhou, Yanfeng Wang, Ya Zhang, Yongguo Yu, Kun Sun, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20430)  

**Abstract**: Rare diseases collectively affect over 300 million individuals worldwide, yet timely and accurate diagnosis remains a pervasive challenge. This is largely due to their clinical heterogeneity, low individual prevalence, and the limited familiarity most clinicians have with rare conditions. Here, we introduce DeepRare, the first rare disease diagnosis agentic system powered by a large language model (LLM), capable of processing heterogeneous clinical inputs. The system generates ranked diagnostic hypotheses for rare diseases, each accompanied by a transparent chain of reasoning that links intermediate analytic steps to verifiable medical evidence.
DeepRare comprises three key components: a central host with a long-term memory module; specialized agent servers responsible for domain-specific analytical tasks integrating over 40 specialized tools and web-scale, up-to-date medical knowledge sources, ensuring access to the most current clinical information. This modular and scalable design enables complex diagnostic reasoning while maintaining traceability and adaptability. We evaluate DeepRare on eight datasets. The system demonstrates exceptional diagnostic performance among 2,919 diseases, achieving 100% accuracy for 1013 diseases. In HPO-based evaluations, DeepRare significantly outperforms other 15 methods, like traditional bioinformatics diagnostic tools, LLMs, and other agentic systems, achieving an average Recall@1 score of 57.18% and surpassing the second-best method (Reasoning LLM) by a substantial margin of 23.79 percentage points. For multi-modal input scenarios, DeepRare achieves 70.60% at Recall@1 compared to Exomiser's 53.20% in 109 cases. Manual verification of reasoning chains by clinical experts achieves 95.40% agreements. Furthermore, the DeepRare system has been implemented as a user-friendly web application this http URL. 

**Abstract (ZH)**: 罕见疾病集体影响 worldwide 超过 30 亿人，但及时和准确的诊断仍然是一个普遍的挑战。这主要归因于其临床异质性、低个体发病率以及大多数临床医生对罕见疾病的不熟悉。这里，我们介绍了 DeepRare，这是一种由大型语言模型（LLM）驱动的第一个罕见疾病诊断代理系统，能够处理异质性的临床输入。该系统生成针对罕见疾病的分级诊断假设，每个假设都配有透明的推理链，将中间分析步骤与可验证的医学证据联系起来。 

---
# SV-LLM: An Agentic Approach for SoC Security Verification using Large Language Models 

**Title (ZH)**: SV-LLM：基于大型语言模型的SoC安全性验证方法 

**Authors**: Dipayan Saha, Shams Tarek, Hasan Al Shaikh, Khan Thamid Hasan, Pavan Sai Nalluri, Md. Ajoad Hasan, Nashmin Alam, Jingbo Zhou, Sujan Kumar Saha, Mark Tehranipoor, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20415)  

**Abstract**: Ensuring the security of complex system-on-chips (SoCs) designs is a critical imperative, yet traditional verification techniques struggle to keep pace due to significant challenges in automation, scalability, comprehensiveness, and adaptability. The advent of large language models (LLMs), with their remarkable capabilities in natural language understanding, code generation, and advanced reasoning, presents a new paradigm for tackling these issues. Moving beyond monolithic models, an agentic approach allows for the creation of multi-agent systems where specialized LLMs collaborate to solve complex problems more effectively. Recognizing this opportunity, we introduce SV-LLM, a novel multi-agent assistant system designed to automate and enhance SoC security verification. By integrating specialized agents for tasks like verification question answering, security asset identification, threat modeling, test plan and property generation, vulnerability detection, and simulation-based bug validation, SV-LLM streamlines the workflow. To optimize their performance in these diverse tasks, agents leverage different learning paradigms, such as in-context learning, fine-tuning, and retrieval-augmented generation (RAG). The system aims to reduce manual intervention, improve accuracy, and accelerate security analysis, supporting proactive identification and mitigation of risks early in the design cycle. We demonstrate its potential to transform hardware security practices through illustrative case studies and experiments that showcase its applicability and efficacy. 

**Abstract (ZH)**: 确保复杂系统级芯片（SoCs）设计的安全性是一项关键需求，但由于传统验证技术在自动化、扩展性、全面性和适应性方面的重大挑战，传统的验证技术难以跟上发展的步伐。大型语言模型（LLMs）凭借其在自然语言理解、代码生成和高级推理方面的出色能力，为解决这些问题提供了新的范式。超越单一模型，采取一种代理式的方法，可以创建由专业LLM组成的多智能体系统，以更有效地解决复杂问题。认识到这一机遇，我们引入了SV-LLM，这是一种新型的多智能体助手系统，旨在自动化并增强SoC安全验证。通过集成专门负责验证问题回答、安全资产识别、威胁建模、测试计划和属性生成、漏洞检测以及基于仿真的缺陷验证等任务的智能体，SV-LLM简化了工作流程。为了在这些多样化的任务中优化智能体的性能，它们利用不同的学习范式，如上下文学习、微调和检索增强生成（RAG）。该系统旨在减少人工干预、提高准确性并加速安全分析，从而支持在设计周期早期积极识别和缓解风险。我们通过示范案例研究和展示其应用效果和有效性来证明其潜力，从而转型硬件安全性实践。 

---
# DipSVD: Dual-importance Protected SVD for Efficient LLM Compression 

**Title (ZH)**: DipSVD: 双重要性保护的SVD方法及其在高效大型语言模型压缩中的应用 

**Authors**: Xuan Ding, Rui Sun, Yunjian Zhang, Xiu Yan, Yueqi Zhou, Kaihao Huang, Suzhong Fu, Chuanlong Xie, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20353)  

**Abstract**: The ever-increasing computational demands and deployment costs of large language models (LLMs) have spurred numerous compressing methods. Compared to quantization and unstructured pruning, SVD compression offers superior hardware compatibility and theoretical guarantees. However, existing SVD-based methods focus on the overall discrepancy between the original and compressed matrices while overlooking the protection of critical components within the matrix, which leads to inferior performance in the compressed models. This paper proposes a dual-level importance protection mechanism to enhance SVD-based compression methods: (1) local importance protection: preserving the most critical singular vectors within each weight matrix through channel-weighted data whitening; and (2) global importance protection: enabling less important layers to bear a greater portion of the compression burden through either a heuristic or optimization-based approach, thereby minimizing the impact of compression on critical layers. Extensive experiments demonstrate that DipSVD outperforms existing SVD-based compression approaches across multiple benchmarks, achieving superior model performance especially at high model compression ratios. 

**Abstract (ZH)**: 大型语言模型（LLMs）不断增加的计算需求和部署成本促使了众多压缩方法的发展。与量化的和未结构化剪枝方法相比，SVD压缩在硬件兼容性和理论保证方面具有优势。然而，现有的基于SVD的方法主要关注原始矩阵和压缩矩阵的整体差异，而忽视了保护矩阵中的关键组件，导致压缩模型的性能较差。本文提出了一种双层重要性保护机制以增强基于SVD的压缩方法：（1）局部重要性保护：通过通道加权数据去相关保留每个权重矩阵中最关键的奇异向量；（2）全局重要性保护：通过启发式或优化方法使不太重要的层承担更多的压缩负担，从而最小化压缩对关键层的影响。广泛实验证明，DipSVD在多个基准上优于现有基于SVD的压缩方法，特别是在高模型压缩比下实现了更优异的模型性能。 

---
# Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models 

**Title (ZH)**: Q-resafe: 评估安全风险和量化感知的安全修补对于量化的大语言模型 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Yu Wang, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.20251)  

**Abstract**: Quantized large language models (LLMs) have gained increasing attention and significance for enabling deployment in resource-constrained environments. However, emerging studies on a few calibration dataset-free quantization methods suggest that quantization may compromise the safety capabilities of LLMs, underscoring the urgent need for systematic safety evaluations and effective mitigation strategies. In this paper, we present comprehensive safety evaluations across various mainstream quantization techniques and diverse calibration datasets, utilizing widely accepted safety benchmarks. To address the identified safety vulnerabilities, we propose a quantization-aware safety patching framework, Q-resafe, to efficiently restore the safety capabilities of quantized LLMs while minimizing any adverse impact on utility. Extensive experimental results demonstrate that Q-resafe successfully re-aligns the safety of quantized LLMs with their pre-quantization counterparts, even under challenging evaluation scenarios. Project page is available at: this https URL. 

**Abstract (ZH)**: 量化大型语言模型（LLMs）在资源受限环境中部署愈发受到关注，但新兴的研究表明，量化可能会牺牲LLMs的安全能力，突显了系统安全评估和有效缓解策略的迫切需求。在本文中，我们全面评估了各种主流量化技术与多样化的校准数据集的安全性，采用广泛接受的安全基准。为应对识别出的安全漏洞，我们提出了一种量化感知安全性修补框架Q-resafe，以高效恢复量化LLMs的安全能力并尽量减小对实用性的负面影响。广泛实验结果显示，Q-resafe能够即使在严峻的评估场景下，成功使量化LLMs的安全性重新与量化前的版本保持一致。项目页面详见：this https URL。 

---
# Enhancing Large Language Models through Structured Reasoning 

**Title (ZH)**: 通过结构化推理增强大型语言模型 

**Authors**: Yubo Dong, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20241)  

**Abstract**: Recent Large Language Models (LLMs) have significantly advanced natural language processing and automated decision-making. However, these models still encounter difficulties when performing complex reasoning tasks involving logical deduction and systematic planning, primarily due to their reliance on implicit statistical relationships without structured knowledge this http URL by cognitive science and neurosymbolic AI, we introduce a novel approach to enhance LLMs through explicit structured reasoning. First, we convert unstructured data into structured formats by explicitly annotating reasoning steps. We then employ this structured dataset to train LLMs through Supervised Fine-Tuning (SFT). Additionally, we enhance the structured reasoning capabilities of LLMs using Group Relative Policy Optimization (GRPO), incorporating two innovative algorithms--MAX-Flow and Longest Common Subsequence (LCS)--which notably improve reasoning effectiveness and reduce computational complexity. Experimental results from fine-tuning a DeepSeek-R1-Distill-Qwen-1.5B model demonstrate concise reasoning, robust performance across various scenarios, and improved compatibility with optimization techniques, validating the efficacy of structured reasoning integration in LLMs. 

**Abstract (ZH)**: Recent Large Language Models (LLMs)在自然语言处理和自动化决策方面取得了显著进展，但仍然难以执行涉及逻辑推理和系统规划的复杂任务，主要原因是它们依赖于无结构的知识和隐含的统计关系。通过认知科学和神经符号AI，我们提出了一种新的方法来增强LLMs的结构化推理能力。首先，通过明确标注推理步骤，我们将非结构化数据转换为结构化格式。然后，我们使用监督微调(SFT)来训练LLMs。此外，我们通过Group Relative Policy Optimization (GRPO)增强LLMs的结构化推理能力，并引入了两种创新算法——MAX-Flow和Longest Common Subsequence (LCS)——显著提高了推理效果并降低了计算复杂度。对DeepSeek-R1-Distill-Qwen-1.5B模型进行微调的实验结果表明，这种结构化推理方法具有简洁的推理能力、在各种场景中表现稳健，并且能更好地与优化技术兼容，验证了结构化推理在LLMs中的有效性。 

---
# How to Retrieve Examples in In-context Learning to Improve Conversational Emotion Recognition using Large Language Models? 

**Title (ZH)**: 在上下文学习中利用大型语言模型提高 conversational emotion recognition 的例子检索方法探究 

**Authors**: Mengqi Wang, Tiantian Feng, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20199)  

**Abstract**: Large language models (LLMs) have enabled a wide variety of real-world applications in various domains. However, creating a high-performing application with high accuracy remains challenging, particularly for subjective tasks like emotion recognition. Inspired by the SLT 2024 GenSER Challenge, this study investigates approaches to improving conversational emotion recognition (CER) by LLMs. Specifically, we explore how to retrieve high-quality examples in in-context learning (ICL) to enhance CER. We propose various strategies based on random and augmented example retrieval and also analyze the impact of conversational context on CER accuracy. Experiments were conducted on the three datasets including IEMOCAP, MELD and EmoryNLP. The results show that augmented example retrieval consistently outperforms other techniques under investigation across all datasets, highlighting the importance of retrieving coherent targeted examples and enhancing them through paraphrasing. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域-enable了广泛的实际应用。然而，创建具有高准确性的高performance应用仍然颇具挑战，特别是在情感识别等主观任务上。受SLT 2024 GenSER挑战的启发，本研究探讨了通过在上下文学习（ICL）中检索高质量示例来提高对话情感识别（CER）的方法。具体而言，我们研究了如何在ICL中检索高质量示例以增强CER。我们提出了基于随机和增强示例检索的各种策略，并分析了对话上下文对CER准确性的影响。实验在IEMOCAP、MELD和EmoryNLP三个数据集上进行。结果表明，增强示例检索在所有数据集上均一致优于其他调查的技术，突显了检索一致的相关示例并通过改述增强它们的重要性。 

---
# Zero-Shot Attribution for Large Language Models: A Distribution Testing Approach 

**Title (ZH)**: 大型语言模型的零样本归属分析：一种分布测试方法 

**Authors**: Clément L. Canonne, Yash Pote, Uddalok Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20197)  

**Abstract**: A growing fraction of all code is sampled from Large Language Models (LLMs). We investigate the problem of attributing code generated by language models using hypothesis testing to leverage established techniques and guarantees. Given a set of samples $S$ and a suspect model $\mathcal{L}^*$, our goal is to assess the likelihood of $S$ originating from $\mathcal{L}^*$. Due to the curse of dimensionality, this is intractable when only samples from the LLM are given: to circumvent this, we use both samples and density estimates from the LLM, a form of access commonly available.
We introduce $\mathsf{Anubis}$, a zero-shot attribution tool that frames attribution as a distribution testing problem. Our experiments on a benchmark of code samples show that $\mathsf{Anubis}$ achieves high AUROC scores ( $\ge0.9$) when distinguishing between LLMs like DeepSeek-Coder, CodeGemma, and Stable-Code using only $\approx 2000$ samples. 

**Abstract (ZH)**: 越来越多的代码片段来自于大型语言模型（LLMs）。我们研究使用假设检验对语言模型生成的代码进行归属的问题，以利用已有的技术和保证。给定一个样本集$S$和一个嫌疑模型$\mathcal{L}^*$，我们的目标是评估$S$来源于$\mathcal{L}^*$的概率。由于维数灾难，仅给定LLM的样本时，这一任务无法解决：为了克服这一问题，我们使用了来自LLM的样本和密度估计，这是通常可用的一种访问形式。

我们引入了$\mathsf{Anubis}$，一种零样本归属工具，将其归属问题框架化为分布检验问题。我们在一个代码样本基准测试中进行的实验表明，$\mathsf{Anubis}$仅使用$\approx 2000$个样本就能够成功地区分DeepSeek-Coder、CodeGemma和Stable-Code等LLM，其AUC-ROC分数达到$\ge0.9$。 

---
# SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs 

**Title (ZH)**: SEED：一种结构编码器，用于时间序列预测的嵌入驱动解码 

**Authors**: Fengze Li, Yue Wang, Yangle Liu, Ming Huang, Dou Hong, Jieming Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.20167)  

**Abstract**: Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap. 

**Abstract (ZH)**: 多变量时间序列预测要求模型同时捕捉变量级别的结构依赖并跨多种任务进行泛化。虽然结构编码器能够有效建模特征交互，但缺乏支持语义级推理或任务适配的能力。相反，大规模语言模型（LLMs）具备强大的泛化能力，但与原始时间序列输入不兼容。这一差距限制了统一可迁移预测系统的开发。因此，我们引入了SEED，一种用于嵌入驱动解码的结构编码器，集成四个阶段：一个具有标记意识的编码器用于片段提取，一个投影模块将片段与语言模型嵌入对齐，一个语义重编程机制将片段映射到任务感知原型，以及一个冻结的语言模型用于预测。这种模块化架构将表示学习与推理解耦，允许高效地对齐数值模式与语义推理。实验结果表明，所提出的方法在强基线方法上实现了一致的改进，并且在各种数据集上的对比研究证实了SEED在解决结构-语义建模差距中的作用。 

---
# EAR: Erasing Concepts from Unified Autoregressive Models 

**Title (ZH)**: EAR: 消除统一自回归模型中的概念 

**Authors**: Haipeng Fan, Shiyuan Zhang, Baohunesitu, Zihang Guo, Huaiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20151)  

**Abstract**: Autoregressive (AR) models have achieved unified and strong performance across both visual understanding and image generation tasks. However, removing undesired concepts from AR models while maintaining overall generation quality remains an open challenge. In this paper, we propose Erasure Autoregressive Model (EAR), a fine-tuning method for effective and utility-preserving concept erasure in AR models. Specifically, we introduce Windowed Gradient Accumulation (WGA) strategy to align patch-level decoding with erasure objectives, and Thresholded Loss Masking (TLM) strategy to protect content unrelated to the target concept during fine-tuning. Furthermore, we propose a novel benchmark, Erase Concept Generator and Visual Filter (ECGVF), aim at provide a more rigorous and comprehensive foundation for evaluating concept erasure in AR models. Specifically, we first employ structured templates across diverse large language models (LLMs) to pre-generate a large-scale corpus of target-replacement concept prompt pairs. Subsequently, we generate images from these prompts and subject them to rigorous filtering via a visual classifier to ensure concept fidelity and alignment. Extensive experimental results conducted on the ECGVF benchmark with the AR model Janus-Pro demonstrate that EAR achieves marked improvements in both erasure effectiveness and model utility preservation. Code is available at: this https URL 

**Abstract (ZH)**: 自回归（AR）模型在视觉理解和图像生成任务中均取得了统一且强大的表现。然而，在移除AR模型中的不需要概念的同时保持整体生成质量仍然是一个开放的挑战。本文提出了一种名为Erasure Autoregressive Model（EAR）的方法，用于在不损害模型实用性的前提下有效移除概念。具体而言，我们引入了Windowed Gradient Accumulation（WGA）策略以与移除目标对局部解码进行对齐，并提出了Thresholded Loss Masking（TLM）策略以在微调过程中保护与目标概念无关的内容。此外，我们提出了一个新的基准Erase Concept Generator and Visual Filter（ECGVF），旨在为评估AR模型中的概念移除提供更严谨和全面的基础。具体而言，我们利用跨多种大型语言模型（LLMs）的结构化模板预先生成了大量的目标替换概念提示对。随后，我们从这些提示生成图像，并通过视觉分类器进行严格的过滤以确保概念的准确性和一致性。在使用AR模型Janus-Pro进行的ECGVF基准上进行的广泛实验结果表明，EAR在移除效果和模型实用性保留方面均取得了显著改进。代码可访问：this https URL。 

---
# CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation 

**Title (ZH)**: CCRS：一种全面RAG评估的零样本LLM-as-a-Judge框架 

**Authors**: Aashiq Muhamed  

**Link**: [PDF](https://arxiv.org/pdf/2506.20128)  

**Abstract**: RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems. 

**Abstract (ZH)**: RAG系统通过整合外部知识增强语言模型，这对于需要事实准确性和最新信息的领域至关重要。然而，评估RAG输出的多方面质量，涵盖上下文连贯性、查询相关性、事实正确性和信息完整性等方面，面临着重大挑战。现有的评估方法往往依赖简单的词频重叠度量，这些方法不足以捕捉这些微妙之处，或者涉及复杂的多阶段管道，包含中间步骤如主张提取，或者需要微调专门的法官模型，这阻碍了其实用效率。为了应对这些局限性，我们提出了一种新的评分方案CCRS（上下文连贯性和相关性评分），这是一种利用单一预训练语言模型作为零样本、端到端法官的五个新指标集合。CCRS评估：上下文连贯性（CC）、查询相关性（QR）、信息密度（ID）、答案正确性（AC）和信息召回率（IR）。我们将CCRS应用于六个不同的RAG系统配置在具有挑战性的BioASQ数据集上的评估。我们的分析表明，CCRS有效地区分了系统性能，例如证实Mistral-7B阅读器优于Llama变体。我们详细分析了CCRS指标属性，包括分数分布、收敛/区分效度、平局率、人口统计学和区分能力。与复杂的RAGChecker框架相比，CCRS在关键方面如召回率和忠实度方面的区分能力具有可比性或更优，并且计算效率显著更高。CCRS因此提供了一种实际、全面且高效的框架，用于评估和逐步改进RAG系统。 

---
# A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs 

**Title (ZH)**: 集成空间-时间模型和大语言模型的模块化多任务推理框架 

**Authors**: Kethmi Hirushini Hettige, Jiahao Ji, Cheng Long, Shili Xiang, Gao Cong, Jingyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20073)  

**Abstract**: Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems. 

**Abstract (ZH)**: 时空数据挖掘在不同领域的知情决策中发挥着关键作用。然而，现有的模型往往局限于狭窄的任务，缺乏进行多任务推理和复杂长篇推理的能力，这些能力需要生成深入解释性的输出。这些限制限制了它们在现实世界多方面决策场景中的应用。本文介绍了一种名为STReason的新型框架，该框架将大型语言模型（LLMs）的推理优势与时空模型的分析能力相结合，用于多任务推理和执行。STReason无需特定任务的微调，利用上下文学习将复杂的自然语言查询分解为模块化、可解释的程序，然后系统地执行这些程序以生成解决方案和详细的推理过程。为了进行严格的评估，我们构建了一个新的基准数据集，并提出了一个统一的评估框架，包含专门设计用于长篇时空推理的评估指标。实验结果表明，STReason在所有指标上均显著优于先进的LLM基线模型，尤其在复杂的、推理密集型的时空场景中表现优异。进一步的人类评估证实了STReason的可信度和实际应用价值，展示了其在减轻专家工作负担和扩大现实世界时空任务应用范围方面的潜力。我们认为STReason为开发更强大和通用的时空推理系统指明了有前景的方向。 

---
# Cross-Layer Discrete Concept Discovery for Interpreting Language Models 

**Title (ZH)**: 跨层离散概念发现用于解释语言模型 

**Authors**: Ankur Garg, Xuemin Yu, Hassan Sajjad, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20040)  

**Abstract**: Uncovering emergent concepts across transformer layers remains a significant challenge because the residual stream linearly mixes and duplicates information, obscuring how features evolve within large language models. Current research efforts primarily inspect neural representations at single layers, thereby overlooking this cross-layer superposition and the redundancy it introduces. These representations are typically either analyzed directly for activation patterns or passed to probing classifiers that map them to a limited set of predefined concepts. To address these limitations, we propose \gls{clvqvae}, a framework that uses vector quantization to map representations across layers and in the process collapse duplicated residual-stream features into compact, interpretable concept vectors. Our approach uniquely combines top-$k$ temperature-based sampling during quantization with EMA codebook updates, providing controlled exploration of the discrete latent space while maintaining code-book diversity. We further enhance the framework with scaled-spherical k-means++ for codebook initialization, which clusters by directional similarity rather than magnitude, better aligning with semantic structure in word embedding space. 

**Abstract (ZH)**: 揭示变压器层间涌现的概念依然是一项重大挑战，因为残差流线性地混合和复制信息，掩盖了大型语言模型中特征的发展过程。当前的研究主要检查单层的神经表示，从而忽视了层间的叠加以及由此引入的冗余性。这些表示通常要么直接分析其激活模式，要么传递给探针分类器映射到一组预定义的概念中。为解决这些局限性，我们提出了一种名为\gls{clvqvae}的框架，该框架使用向量量化在层间映射表示，并在此过程中将重复的残差流特征压缩为紧凑且可解释的概念向量。我们的方法独特地结合了量化过程中的基于top-$k$温度的采样与指数移动平均码书更新，提供了对离散潜在空间可控的探索，同时保持码书多样性。我们进一步通过缩放球形k-means++初始化码书，该方法按方向相似性而不是幅度进行聚类，更好地与词嵌入空间中的语义结构对齐。 

---
# HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization 

**Title (ZH)**: HERCULES：基于嵌入的分层递归聚类及大规模语言模型高效摘要方法 

**Authors**: Gabor Petnehazi, Bernadett Aradi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19992)  

**Abstract**: The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets. 

**Abstract (ZH)**: 复杂模态下的爆炸性增长数据集需先进的分析工具，不仅要有效地分组数据，还能提供可理解的发现结构洞察。我们引入HERCULES（基于层次嵌入的递归聚类算法，利用LLM进行高效总结），这是一种新型算法及Python包，用于对包括文本、图像和数值数据在内的多种类型数据进行层次k-means聚类。HERCULES通过递归应用k-means聚类，从第0级的单个数据点开始构建聚类层次结构。关键创新在于其深度集成了大型语言模型（LLM）以生成具有丰富语义的聚类标题和描述，显著提高了可解释性。算法支持两种主要的表示模式：“直接”模式，基于原始数据嵌入或缩放的数值特征进行聚类；“描述”模式，基于LLM生成的摘要的嵌入进行聚类。用户可以提供“主题种子”以引导LLM生成的摘要向特定主题靠拢。交互式可视化工具有助于深入了解聚类结果。我们展示了HERCULES的能力，并讨论其从复杂数据集中提取有意义的层次知识的潜力。 

---
# Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs 

**Title (ZH)**: Inference Scaled GraphRAG：改进知识图上的多跳问答 

**Authors**: Travis Thompson, Seung-Hwan Lim, Paul Liu, Ruoying He, Dongkuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19967)  

**Abstract**: Large Language Models (LLMs) have achieved impressive capabilities in language understanding and generation, yet they continue to underperform on knowledge-intensive reasoning tasks due to limited access to structured context and multi-hop information. Retrieval-Augmented Generation (RAG) partially mitigates this by grounding generation in retrieved context, but conventional RAG and GraphRAG methods often fail to capture relational structure across nodes in knowledge graphs. We introduce Inference-Scaled GraphRAG, a novel framework that enhances LLM-based graph reasoning by applying inference-time compute scaling. Our method combines sequential scaling with deep chain-of-thought graph traversal, and parallel scaling with majority voting over sampled trajectories within an interleaved reasoning-execution loop. Experiments on the GRBench benchmark demonstrate that our approach significantly improves multi-hop question answering performance, achieving substantial gains over both traditional GraphRAG and prior graph traversal baselines. These findings suggest that inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs 

**Abstract (ZH)**: 基于推理缩放的GraphRAG：一种增强的大语言模型图推理框架 

---
# CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation 

**Title (ZH)**: CycleDistill：使用循环蒸馏的大型语言模型bootstrapping机器翻译 

**Authors**: Deepon Halder, Thanmay Jayakumar, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2506.19952)  

**Abstract**: Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality. 

**Abstract (ZH)**: 基于循环蒸馏的低资源语言机器翻译方法 

---
# Can LLMs Replace Humans During Code Chunking? 

**Title (ZH)**: 大规模语言模型能否替代人类进行代码分割？ 

**Authors**: Christopher Glasz, Emily Escamilla, Eric O. Scott, Anand Patel, Jacob Zimmer, Colin Diggs, Michael Doyle, Scott Rosen, Nitin Naik, Justin F. Brunelle, Samruddhi Thaker, Parthav Poudel, Arun Sridharan, Amit Madan, Doug Wendt, William Macke, Thomas Schill  

**Link**: [PDF](https://arxiv.org/pdf/2506.19897)  

**Abstract**: Large language models (LLMs) have become essential tools in computer science, especially for tasks involving code understanding and generation. However, existing work does not address many of the unique challenges presented by code written for government applications. In particular, government enterprise software is often written in legacy languages like MUMPS or assembly language code (ALC) and the overall token lengths of these systems exceed the context window size for current commercially available LLMs. Additionally, LLMs are primarily trained on modern software languages and have undergone limited testing with legacy languages, making their ability to understand legacy languages unknown and, hence, an area for empirical study. This paper examines the application of LLMs in the modernization of legacy government code written in ALC and MUMPS, addressing the challenges of input limitations. We investigate various code-chunking methods to optimize the generation of summary module comments for legacy code files, evaluating the impact of code-chunking methods on the quality of documentation produced by different LLMs, including GPT-4o, Claude 3 Sonnet, Mixtral, and Llama 3. Our results indicate that LLMs can select partition points closely aligned with human expert partitioning. We also find that chunking approaches have significant impact on downstream tasks such as documentation generation. LLM-created partitions produce comments that are up to 20% more factual and up to 10% more useful than when humans create partitions. Therefore, we conclude that LLMs can be used as suitable replacements for human partitioning of large codebases during LLM-aided modernization. 

**Abstract (ZH)**: 大型语言模型在政府应用legacy代码现代化中的应用：解决输入限制挑战 

---
# Retrieval-Confused Generation is a Good Defender for Privacy Violation Attack of Large Language Models 

**Title (ZH)**: 检索混淆生成是大型语言模型隐私侵犯攻击的良好防御方法 

**Authors**: Wanli Peng, Xin Chen, Hang Fu, XinYu He, Xue Yiming, Juan Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19889)  

**Abstract**: Recent advances in large language models (LLMs) have made a profound impact on our society and also raised new security concerns. Particularly, due to the remarkable inference ability of LLMs, the privacy violation attack (PVA), revealed by Staab et al., introduces serious personal privacy issues. Existing defense methods mainly leverage LLMs to anonymize the input query, which requires costly inference time and cannot gain satisfactory defense performance. Moreover, directly rejecting the PVA query seems like an effective defense method, while the defense method is exposed, promoting the evolution of PVA. In this paper, we propose a novel defense paradigm based on retrieval-confused generation (RCG) of LLMs, which can efficiently and covertly defend the PVA. We first design a paraphrasing prompt to induce the LLM to rewrite the "user comments" of the attack query to construct a disturbed database. Then, we propose the most irrelevant retrieval strategy to retrieve the desired user data from the disturbed database. Finally, the "data comments" are replaced with the retrieved user data to form a defended query, leading to responding to the adversary with some wrong personal attributes, i.e., the attack fails. Extensive experiments are conducted on two datasets and eight popular LLMs to comprehensively evaluate the feasibility and the superiority of the proposed defense method. 

**Abstract (ZH)**: Recent Advances in Large Language Models: A Retrieval-Confused Generation Paradigm for Defending Privacy Violation Attacks 

---
# MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection 

**Title (ZH)**: MNN-AECS: 移动设备上LLM解码的自适应核心选择能源优化 

**Authors**: Zhengxiang Huang, Chaoyue Niu, Zhaode Wang, Jiarui Xue, Hanming Zhang, Yugang Wang, Zewei Xin, Xiaotang Jiang, Chengfei Lv, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19884)  

**Abstract**: As the demand for on-device Large Language Model (LLM) inference grows, energy efficiency has become a major concern, especially for battery-limited mobile devices. Our analysis shows that the memory-bound LLM decode phase dominates energy use, and yet most existing works focus on accelerating the prefill phase, neglecting energy concerns. We introduce Adaptive Energy-Centric Core Selection (AECS) and integrate it into MNN to create the energy-efficient version, MNN-AECS, the first engine-level system solution without requiring root access or OS modifications for energy-efficient LLM decoding. MNN-AECS is designed to reduce LLM decoding energy while keeping decode speed within an acceptable slowdown threshold by dynamically selecting low-power CPU cores. MNN-AECS is evaluated across 5 Android and 2 iOS devices on 5 popular LLMs of various sizes. Compared to original MNN, MNN-AECS cuts down energy use by 23% without slowdown averaged over all 7 devices and 4 datasets. Against other engines, including this http URL, executorch, mllm, and MediaPipe, MNN-AECS delivers 39% to 78% energy saving and 12% to 363% speedup on average. 

**Abstract (ZH)**: 基于适应性的能量中心核心选择（AECS）驱动的能效大语言模型解码器：无需_root访问或OS修改的端侧能效解决方案 

---
# Exploring the Capabilities of the Frontier Large Language Models for Nuclear Energy Research 

**Title (ZH)**: 探索前沿大型语言模型在核能研究中的能力 

**Authors**: Ahmed Almeldein, Mohammed Alnaggar, Rick Archibald, Tom Beck, Arpan Biswas, Rike Bostelmann, Wes Brewer, Chris Bryan, Christopher Calle, Cihangir Celik, Rajni Chahal, Jong Youl Choi, Arindam Chowdhury, Mark Cianciosa, Franklin Curtis, Gregory Davidson, Sebastian De Pascuale, Lisa Fassino, Ana Gainaru, Yashika Ghai, Luke Gibson, Qian Gong, Christopher Greulich, Scott Greenwood, Cory Hauck, Ehab Hassan, Rinkle Juneja, Soyoung Kang, Scott Klasky, Atul Kumar, Vineet Kumar, Paul Laiu, Calvin Lear, Yan-Ru Lin, Jono McConnell, Furkan Oz, Anant Raj, Pradeep Ramuhalli, Marie Romedenne, Samantha Sabatino, José Salcedo-Pérez, Nathan D. See, Arpan Sircar, Punam Thankur, Tim Younkin, Xiao-Ying Yu, Prashant Jain, Tom Evans, Prasanna Balaprakash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19863)  

**Abstract**: The AI for Nuclear Energy workshop at Oak Ridge National Laboratory evaluated the potential of Large Language Models (LLMs) to accelerate fusion and fission research. Fourteen interdisciplinary teams explored diverse nuclear science challenges using ChatGPT, Gemini, Claude, and other AI models over a single day. Applications ranged from developing foundation models for fusion reactor control to automating Monte Carlo simulations, predicting material degradation, and designing experimental programs for advanced reactors. Teams employed structured workflows combining prompt engineering, deep research capabilities, and iterative refinement to generate hypotheses, prototype code, and research strategies. Key findings demonstrate that LLMs excel at early-stage exploration, literature synthesis, and workflow design, successfully identifying research gaps and generating plausible experimental frameworks. However, significant limitations emerged, including difficulties with novel materials designs, advanced code generation for modeling and simulation, and domain-specific details requiring expert validation. The successful outcomes resulted from expert-driven prompt engineering and treating AI as a complementary tool rather than a replacement for physics-based methods. The workshop validated AI's potential to accelerate nuclear energy research through rapid iteration and cross-disciplinary synthesis while highlighting the need for curated nuclear-specific datasets, workflow automation, and specialized model development. These results provide a roadmap for integrating AI tools into nuclear science workflows, potentially reducing development cycles for safer, more efficient nuclear energy systems while maintaining rigorous scientific standards. 

**Abstract (ZH)**: Oak Ridge National Laboratory的AI在核能领域的工作shop评估了大型语言模型在加速聚变和裂变研究方面的潜力。十四个跨学科团队在一天内使用ChatGPT、Gemini、Claude及其他AI模型探索了多样的核科学挑战。应用范围包括为聚变反应堆控制开发基础模型、自动化蒙特卡洛模拟、预测材料退化以及设计先进反应堆的实验计划。团队采用了结构化的流程结合提示工程、深度研究能力和迭代优化生成假设、原型代码和研究策略。关键发现表明，大型语言模型在早期探索、文献综合和工作流程设计方面表现出色，成功识别了研究缺口并生成了可能的实验框架。然而，也出现了显著的局限性，包括新材料设计的难题、高级建模和模拟代码生成的挑战以及需要专家验证的领域特定细节。成功的成果来自于专家驱动的提示工程以及将AI视为物理方法的补充工具而非替代品。研讨会验证了AI通过快速迭代和跨学科综合加速核能研究的潜力，同时强调了需要制作化的核特定数据集、工作流程自动化和专门模型开发的需求。这些结果为将AI工具整合到核科学工作流程中提供了蓝图，可能在保持严格科学标准的同时，减少安全性和效率更高的核能系统的发展周期。 

---
