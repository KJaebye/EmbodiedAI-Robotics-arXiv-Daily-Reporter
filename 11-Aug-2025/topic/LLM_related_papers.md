# LLM Robustness Leaderboard v1 --Technical report 

**Title (ZH)**: LLM robustness leaderboard v1 --技术报告 

**Authors**: Pierre Peigné - Lefebvre, Quentin Feuillade-Montixi, Tom David, Nicolas Miailhe  

**Link**: [PDF](https://arxiv.org/pdf/2508.06296)  

**Abstract**: This technical report accompanies the LLM robustness leaderboard published by PRISM Eval for the Paris AI Action Summit. We introduce PRISM Eval Behavior Elicitation Tool (BET), an AI system performing automated red-teaming through Dynamic Adversarial Optimization that achieves 100% Attack Success Rate (ASR) against 37 of 41 state-of-the-art LLMs. Beyond binary success metrics, we propose a fine-grained robustness metric estimating the average number of attempts required to elicit harmful behaviors, revealing that attack difficulty varies by over 300-fold across models despite universal vulnerability. We introduce primitive-level vulnerability analysis to identify which jailbreaking techniques are most effective for specific hazard categories. Our collaborative evaluation with trusted third parties from the AI Safety Network demonstrates practical pathways for distributed robustness assessment across the community. 

**Abstract (ZH)**: 本技术报告 accompanies 巴黎AI行动峰会由PRISM Eval发布的大型语言模型鲁棒性排行榜。我们介绍了PRISM Eval行为诱发型工具（BET），这是一种通过动态对抗优化进行自动化红队测试的AI系统，实现对41个前沿大型语言模型中的37个达到100%的攻击成功率（ASR）。除了二元成功率指标外，我们提出了一种细粒度的鲁棒性度量，估计引起有害行为所需的平均尝试次数，揭示尽管所有模型普遍存在漏洞，攻击难度仍相差逾300倍。我们介绍了基础级漏洞分析，以确定哪些越狱技术对特定危害类别最有效。与AI安全网络中的可信赖第三方进行的合作评估展示了社区中分布式鲁棒性评估的实际路径。 

---
# GeoLaux: A Benchmark for Evaluating MLLMs' Geometry Performance on Long-Step Problems Requiring Auxiliary Lines 

**Title (ZH)**: GeoLaux: 一个评估大模型在长步几何问题上的辅助线使用能力的基准测试 

**Authors**: Yumeng Fu, Jiayin Zhu, Lingling Zhang, Bo Zhao, Shaoxuan Ma, Yushun Zhang, Yanrui Wu, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06226)  

**Abstract**: Geometry problem solving (GPS) requires models to master diagram comprehension, logical reasoning, knowledge application, numerical computation, and auxiliary line construction. This presents a significant challenge for Multimodal Large Language Models (MLLMs). However, existing benchmarks for evaluating MLLM geometry skills overlook auxiliary line construction and lack fine-grained process evaluation, making them insufficient for assessing MLLMs' long-step reasoning abilities. To bridge these gaps, we present the GeoLaux benchmark, comprising 2,186 geometry problems, incorporating both calculation and proving questions. Notably, the problems require an average of 6.51 reasoning steps, with a maximum of 24 steps, and 41.8% of them need auxiliary line construction. Building on the dataset, we design a novel five-dimensional evaluation strategy assessing answer correctness, process correctness, process quality, auxiliary line impact, and error causes. Extensive experiments on 13 leading MLLMs (including thinking models and non-thinking models) yield three pivotal findings: First, models exhibit substantial performance degradation in extended reasoning steps (nine models demonstrate over 50% performance drop). Second, compared to calculation problems, MLLMs tend to take shortcuts when solving proving problems. Third, models lack auxiliary line awareness, and enhancing this capability proves particularly beneficial for overall geometry reasoning improvement. These findings establish GeoLaux as both a benchmark for evaluating MLLMs' long-step geometric reasoning with auxiliary lines and a guide for capability advancement. Our dataset and code are included in supplementary materials and will be released. 

**Abstract (ZH)**: 几何问题解决（GPS）要求模型综合图示理解、逻辑推理、知识撰写、数值计算和辅助线构建等
ingerprint 

---
# Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution 

**Title (ZH)**: LLM作为法官的过度自信：诊断与基于信心的解决方案 

**Authors**: Zailong Tian, Zhuoheng Han, Yanzhe Chen, Haozhe Xu, Xi Yang, richeng xuan, Hongfeng Wang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.06225)  

**Abstract**: Large Language Models (LLMs) are widely used as automated judges, where practical value depends on both accuracy and trustworthy, risk-aware judgments. Existing approaches predominantly focus on accuracy, overlooking the necessity of well-calibrated confidence, which is vital for adaptive and reliable evaluation pipelines. In this work, we advocate a shift from accuracy-centric evaluation to confidence-driven, risk-aware LLM-as-a-Judge systems, emphasizing the necessity of well-calibrated confidence for trustworthy and adaptive evaluation. We systematically identify the **Overconfidence Phenomenon** in current LLM-as-a-Judges, where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment. To quantify this phenomenon, we introduce **TH-Score**, a novel metric measuring confidence-accuracy alignment. Furthermore, we propose **LLM-as-a-Fuser**, an ensemble framework that transforms LLMs into reliable, risk-aware evaluators. Extensive experiments demonstrate that our approach substantially improves calibration and enables adaptive, confidence-driven evaluation pipelines, achieving superior reliability and accuracy compared to existing baselines. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动法官中的广泛应用不仅依赖于准确性，还依赖于具有风险意识的、可信的判断。现有的方法主要关注准确性，忽视了校准的置信度的重要性，后者对于适应性和可靠的评估管道至关重要。在本文中，我们倡导从以准确性为中心的评估转向以置信度驱动的风险意识的LLM-as-a-法官系统，强调校准的置信度对于可信和适应性评估的重要性。我们系统地识别了当前LLM-as-a-法官中的**过自信现象**，其中预测的置信度大大高估了实际正确性，损害了实际部署中的可靠性。为了量化这一现象，我们引入了**TH-Score**，一种衡量置信度与准确度对齐的新指标。此外，我们提出了**LLM-as-a-Fuser**集成框架，将LLM转换为可靠的、风险意识的评估器。广泛实验表明，我们的方法显著提高了校准，并使评估管道实现适应性和置信度驱动，达到了比现有基准更高的可靠性和准确性。 

---
# Retrieval Augmented Large Language Model System for Comprehensive Drug Contraindications 

**Title (ZH)**: 全面药物禁忌症增强检索大语言模型系统 

**Authors**: Byeonghun Bang, Jongsuk Yoon, Dong-Jin Chang, Seho Park, Yong Oh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.06145)  

**Abstract**: The versatility of large language models (LLMs) has been explored across various sectors, but their application in healthcare poses challenges, particularly in the domain of pharmaceutical contraindications where accurate and reliable information is required. This study enhances the capability of LLMs to address contraindications effectively by implementing a Retrieval Augmented Generation (RAG) pipeline. Utilizing OpenAI's GPT-4o-mini as the base model, and the text-embedding-3-small model for embeddings, our approach integrates Langchain to orchestrate a hybrid retrieval system with re-ranking. This system leverages Drug Utilization Review (DUR) data from public databases, focusing on contraindications for specific age groups, pregnancy, and concomitant drug use. The dataset includes 300 question-answer pairs across three categories, with baseline model accuracy ranging from 0.49 to 0.57. Post-integration of the RAG pipeline, we observed a significant improvement in model accuracy, achieving rates of 0.94, 0.87, and 0.89 for contraindications related to age groups, pregnancy, and concomitant drug use, respectively. The results indicate that augmenting LLMs with a RAG framework can substantially reduce uncertainty in prescription and drug intake decisions by providing more precise and reliable drug contraindication information. 

**Abstract (ZH)**: 大型语言模型在医疗保健领域的多功能性已在多个领域得到探索，但其在药物禁忌症领域的应用面临挑战，特别是在需要准确可靠信息的制药领域。本研究通过实施检索增强生成（RAG）管道来增强大型语言模型处理禁忌症的能力。以OpenAI的GPT-4o-mini作为基础模型，使用text-embedding-3-small模型进行嵌入，并结合Langchain实现具有检索重排序的混合检索系统。该系统利用公共数据库中的药物使用审查（DUR）数据，重点关注特定年龄段、妊娠及联合用药的禁忌症。数据集包括300个问题-答案对，分为三个类别，基础模型准确性范围为0.49至0.57。集成RAG管道后，我们观察到模型准确性有了显著提高，针对年龄组、妊娠和联合用药的禁忌症分别达到了0.94、0.87和0.89的准确性。研究结果表明，通过RAG框架增强大型语言模型可以显著降低处方和药物摄入决策中的不确定性，提供更精确可靠的药物禁忌症信息。 

---
# SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges 

**Title (ZH)**: SKATE，一种可扩展的锦标赛评估：较弱的LLM通过可验证的挑战区分较强的LLM 

**Authors**: Dewi S. W. Gould, Bruno Mlodozeniec, Samuel F. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2508.06111)  

**Abstract**: Evaluating the capabilities and risks of foundation models is paramount, yet current methods demand extensive domain expertise, hindering their scalability as these models rapidly evolve. We introduce SKATE: a novel evaluation framework in which large language models (LLMs) compete by generating and solving verifiable tasks for one another. Our core insight is to treat evaluation as a game: models act as both task-setters and solvers, incentivized to create questions which highlight their own strengths while exposing others' weaknesses. SKATE offers several key advantages, balancing scalability, open-endedness, and objectivity. It is fully automated, data-free, and scalable, requiring no human input or domain expertise. By using verifiable tasks rather than LLM judges, scoring is objective. Unlike domain-limited programmatically-generated benchmarks (e.g. chess-playing or spatial reasoning), having LLMs creatively pose challenges enables open-ended and scalable evaluation. As a proof of concept, we introduce LLM-set code-output-prediction (COP) challenges as a verifiable and extensible framework in which to test our approach. Using a TrueSkill-based ranking system, we evaluate six frontier LLMs and find that: (1) weaker models can reliably differentiate and score stronger ones, (2) LLM-based systems are capable of self-preferencing behavior, generating questions that align with their own capabilities, and (3) SKATE automatically surfaces fine-grained capability differences between models. Our findings are an important step towards general, scalable evaluation frameworks which can keep pace with LLM progress. 

**Abstract (ZH)**: 评估基础模型的能力和风险至关重要，但当前却受限于于当前领域内缺乏足够的专业知识以应对它们的扩展性问题。目前，这些模型正迅速发展。我们介绍了SKATE：一个引入的评估框架，在通过生成和解决可验证性的任务来进行竞争。我们的引入目的是将模型看待为游戏：模型扮演了任务设定者和解决者的角色，两者相互激励以展示各自的强项和凸显对方的弱点。SKATE提供了几个关键优势，平衡了可可的扩展性和灵活性。它

userisper cabeza：a competitive evaluation evaluation framework for foundation models pérdida的翻译保持原文格式不变，请问继续翻译剩余内容 kukul ...
userisperpedia：a competitive evaluated framework for foundation models kuk 

---
# PanelTR: Zero-Shot Table Reasoning Framework Through Multi-Agent Scientific Discussion 

**Title (ZH)**: PanelTR: 通过多Agent科学讨论的零样本表格推理框架 

**Authors**: Yiran Rex Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.06110)  

**Abstract**: Table reasoning, including tabular QA and fact verification, often depends on annotated data or complex data augmentation, limiting flexibility and generalization. LLMs, despite their versatility, often underperform compared to simple supervised models. To approach these issues, we introduce PanelTR, a framework utilizing LLM agent scientists for robust table reasoning through a structured scientific approach. PanelTR's workflow involves agent scientists conducting individual investigations, engaging in self-review, and participating in collaborative peer-review discussions. This process, driven by five scientist personas, enables semantic-level transfer without relying on data augmentation or parametric optimization. Experiments across four benchmarks show that PanelTR outperforms vanilla LLMs and rivals fully supervised models, all while remaining independent of training data. Our findings indicate that structured scientific methodology can effectively handle complex tasks beyond table reasoning with flexible semantic understanding in a zero-shot context. 

**Abstract (ZH)**: 基于科学家代理的PanelTR框架：通过结构化科学方法实现稳健的表格推理 

---
# LLMs for Resource Allocation: A Participatory Budgeting Approach to Inferring Preferences 

**Title (ZH)**: 资源分配中的大规模语言模型：基于参与性预算的偏好推断方法 

**Authors**: Sankarshan Damle, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2508.06060)  

**Abstract**: Large Language Models (LLMs) are increasingly expected to handle complex decision-making tasks, yet their ability to perform structured resource allocation remains underexplored. Evaluating their reasoning is also difficult due to data contamination and the static nature of existing benchmarks. We present a dual-purpose framework leveraging Participatory Budgeting (PB) both as (i) a practical setting for LLM-based resource allocation and (ii) an adaptive benchmark for evaluating their reasoning capabilities. We task LLMs with selecting project subsets under feasibility (e.g., budget) constraints via three prompting strategies: greedy selection, direct optimization, and a hill-climbing-inspired refinement. We benchmark LLMs' allocations against a utility-maximizing oracle. Interestingly, we also test whether LLMs can infer structured preferences from natural-language voter input or metadata, without explicit votes. By comparing allocations based on inferred preferences to those from ground-truth votes, we evaluate LLMs' ability to extract preferences from open-ended input. Our results underscore the role of prompt design and show that LLMs hold promise for mechanism design with unstructured inputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益期望能够处理复杂的决策任务，然而它们在执行结构化资源分配方面的能力仍较少被探索。评估它们的推理能力也因数据污染和现有基准的静态性质而变得困难。我们提出了一种双重用途框架，利用包容性预算（PB）不仅作为（i）基于LLM的资源分配的实际场景，而且作为评估它们推理能力的适应性基准。我们要求LLM根据可行性约束（例如，预算）选择项目子集，并通过三种不同的提示策略进行：贪婪选择、直接优化和基于爬坡改进的优化。我们将LLM的分配与最大化效用的 oracle 进行基准测试。有趣的是，我们还测试LLM是否能够从自然语言选民输入或元数据中推断出结构化偏好，而不需要明确的投票。通过将基于推断偏好生成的分配与基于真实投票生成的分配进行比较，我们评估了LLM从开放式输入中提取偏好能力。我们的结果强调了提示设计的作用，并展示了LLM在处理非结构化输入时进行机制设计的潜力。 

---
# Society of Mind Meets Real-Time Strategy: A Hierarchical Multi-Agent Framework for Strategic Reasoning 

**Title (ZH)**: 心智社会Meet实时策略：一种基于层级多代理的策略推理框架 

**Authors**: Daechul Ahn, San Kim, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06042)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated impressive action sequence prediction capabilities but often struggle with dynamic, long-horizon tasks such as real-time strategic games. In a game such as StarCraftII (SC2), agents need to manage resource constraints and adapt to evolving battlefield situations in a partially observable environment. This often overwhelms exisiting LLM-based approaches. To address these challenges, we propose a hierarchical multi-agent framework that employs specialized imitation learning agents under a meta-controller called Strategic Planner (SP). By expert demonstrations, each specialized agent learns a distinctive strategy, such as aerial support or defensive maneuvers, and produces coherent, structured multistep action sequences. The SP then orchestrates these proposals into a single, environmentally adaptive plan that ensures local decisions aligning with long-term strategies. We call this HIMA (Hierarchical Imitation Multi-Agent). We also present TEXTSCII-ALL, a comprehensive SC2 testbed that encompasses all race match combinations in SC2. Our empirical results show that HIMA outperforms state of the arts in strategic clarity, adaptability, and computational efficiency, underscoring the potential of combining specialized imitation modules with meta-level orchestration to develop more robust, general-purpose AI agents. 

**Abstract (ZH)**: 大型语言模型（LLMs）在最近展示了令人印象深刻的行动序列预测能力，但在实时战略游戏等动态、长期任务方面往往表现不佳。在《星际争霸II》（SC2）游戏中，智能体需要在部分可观测环境中管理资源约束并适应不断变化的战场情况。这往往超出现有基于LLM的方法的能力范围。为了解决这些挑战，我们提出了一种分级多智能体框架，该框架在元控制器称为战略规划器（SP）的指导下使用专门的模仿学习智能体。通过专家演示，每个专门的智能体学习独特的策略，如空中支援或防御性 maneuver，并生成连贯的结构化多步骤行动序列。SP 然后将这些提案整合成一个环境适应性计划，确保局部决策与长期策略保持一致。我们称这种方法为HIMA（分级模仿多智能体）。我们还介绍了TEXTSCII-ALL，一个全面的SC2测试平台，涵盖了SC2中所有种族匹配组合。我们的实验证明，HIMA在战略清晰度、适应性和计算效率方面优于现有方法，突显了将专门的模仿模块与元级协调相结合以开发更 robust 和通用的AI智能体的潜力。 

---
# Mediator-Guided Multi-Agent Collaboration among Open-Source Models for Medical Decision-Making 

**Title (ZH)**: 开放源码模型引导的中间人导向多智能体协作医学决策 

**Authors**: Kaitao Chen, Mianxin Liu, Daoming Zong, Chaoyue Ding, Shaohao Rui, Yankai Jiang, Mu Zhou, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05996)  

**Abstract**: Complex medical decision-making involves cooperative workflows operated by different clinicians. Designing AI multi-agent systems can expedite and augment human-level clinical decision-making. Existing multi-agent researches primarily focus on language-only tasks, yet their extension to multimodal scenarios remains challenging. A blind combination of diverse vision-language models (VLMs) can amplify an erroneous outcome interpretation. VLMs in general are less capable in instruction following and importantly self-reflection, compared to large language models (LLMs) of comparable sizes. This disparity largely constrains VLMs' ability in cooperative workflows. In this study, we propose MedOrch, a mediator-guided multi-agent collaboration framework for medical multimodal decision-making. MedOrch employs an LLM-based mediator agent that enables multiple VLM-based expert agents to exchange and reflect on their outputs towards collaboration. We utilize multiple open-source general-purpose and domain-specific VLMs instead of costly GPT-series models, revealing the strength of heterogeneous models. We show that the collaboration within distinct VLM-based agents can surpass the capabilities of any individual agent. We validate our approach on five medical vision question answering benchmarks, demonstrating superior collaboration performance without model training. Our findings underscore the value of mediator-guided multi-agent collaboration in advancing medical multimodal intelligence. Our code will be made publicly available. 

**Abstract (ZH)**: 一种基于调解者的多代理协作框架：MedOrch在医疗多模态决策中的应用 

---
# Post-training for Efficient Communication via Convention Formation 

**Title (ZH)**: 通过惯例形成提高通信效率的后训练方法 

**Authors**: Yilun Hua, Evan Wang, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06482)  

**Abstract**: Humans communicate with increasing efficiency in multi-turn interactions, by adapting their language and forming ad-hoc conventions. In contrast, prior work shows that LLMs do not naturally show this behavior. We develop a post-training process to develop this ability through targeted fine-tuning on heuristically identified demonstrations of convention formation. We evaluate with two new benchmarks focused on this capability. First, we design a focused, cognitively-motivated interaction benchmark that consistently elicits strong convention formation trends in humans. Second, we create a new document-grounded reference completion task that reflects in-the-wild convention formation behavior. Our studies show significantly improved convention formation abilities in post-trained LLMs across the two evaluation methods. 

**Abstract (ZH)**: 人类在多轮交互中通过适应语言和形成临时惯例来不断提高交流效率，而先前的工作表明，预训练的语言模型并不表现出这种行为。我们开发了一个后训练过程，通过针对启发式识别出的惯例形成示例进行目标导向的微调，来培养这一能力。我们使用两个新的基准测试来评估这一能力。首先，我们设计了一个认知动机驱动的交互基准测试，能够一致地引发人类强烈的惯例形成趋势。其次，我们创建了一个新的基于文档的参考完成任务，反映了现实世界中的惯例形成行为。我们的研究表明，在两个评估方法中，后训练的语言模型在惯例形成能力上有了显著提升。 

---
# ScamAgents: How AI Agents Can Simulate Human-Level Scam Calls 

**Title (ZH)**: ScamAgents：AI代理如何模拟人类级别的欺诈电话 

**Authors**: Sanket Badhe  

**Link**: [PDF](https://arxiv.org/pdf/2508.06457)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive fluency and reasoning capabilities, but their potential for misuse has raised growing concern. In this paper, we present ScamAgent, an autonomous multi-turn agent built on top of LLMs, capable of generating highly realistic scam call scripts that simulate real-world fraud scenarios. Unlike prior work focused on single-shot prompt misuse, ScamAgent maintains dialogue memory, adapts dynamically to simulated user responses, and employs deceptive persuasion strategies across conversational turns. We show that current LLM safety guardrails, including refusal mechanisms and content filters, are ineffective against such agent-based threats. Even models with strong prompt-level safeguards can be bypassed when prompts are decomposed, disguised, or delivered incrementally within an agent framework. We further demonstrate the transformation of scam scripts into lifelike voice calls using modern text-to-speech systems, completing a fully automated scam pipeline. Our findings highlight an urgent need for multi-turn safety auditing, agent-level control frameworks, and new methods to detect and disrupt conversational deception powered by generative AI. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的流畅性和推理能力，但其潜在的滥用风险引发了日益增长的关注。本文介绍了ScamAgent，这是一个基于LLMs的自主多轮对话代理，能够生成高度逼真的欺诈电话脚本，模拟现实世界的欺诈场景。与侧重于单轮提示滥用的先前工作不同，ScamAgent保留了对话记忆，能够根据模拟的用户响应动态调整，并在对话轮次中运用欺骗性说服策略。我们展示了当前的LLM安全护栏，包括拒绝机制和内容过滤器，对这类基于代理的威胁无效。即使具有强大提示级保护机制的模型，在提示被分解、伪装或通过代理框架分段传递时也可能被绕过。我们还展示了使用现代文本转语音系统将欺诈脚本转换为逼真的语音通话，完成了一个完整的自动化欺诈流程。我们的研究结果突显了对多轮安全审计、代理级控制框架以及检测和中断由生成式AI驱动的对话欺骗的新方法的迫切需求。 

---
# Echoes of Automation: The Increasing Use of LLMs in Newsmaking 

**Title (ZH)**: 自动化回声：新闻制作中LLMs使用的日益增加 

**Authors**: Abolfazl Ansari, Delvin Ce Zhang, Nafis Irtiza Tripto, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.06445)  

**Abstract**: The rapid rise of Generative AI (GenAI), particularly LLMs, poses concerns for journalistic integrity and authorship. This study examines AI-generated content across over 40,000 news articles from major, local, and college news media, in various media formats. Using three advanced AI-text detectors (e.g., Binoculars, Fast-Detect GPT, and GPTZero), we find substantial increase of GenAI use in recent years, especially in local and college news. Sentence-level analysis reveals LLMs are often used in the introduction of news, while conclusions usually written manually. Linguistic analysis shows GenAI boosts word richness and readability but lowers formality, leading to more uniform writing styles, particularly in local media. 

**Abstract (ZH)**: 生成式人工智能（GenAI）的快速崛起，尤其是大型语言模型（LLM），对新闻报道的诚信和署名提出担忧。本研究考察了来自重大媒体、地方媒体和大学媒体的超过40,000篇新闻文章中的AI生成内容，涵盖多种媒体格式。通过使用三种先进的AI文本检测工具（如Binoculars、Fast-Detect GPT和GPTZero），我们发现近年来GenAI的应用显著增加，尤其是在地方和大学新闻报道中。句子层面的分析显示，LLM通常用于新闻的开头，而结论通常由人工撰写。语言分析表明，GenAI提高了词汇丰富度和可读性，但降低了正式程度，导致写作风格更加统一，尤其是在地方媒体中。 

---
# Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages 

**Title (ZH)**: 学习主题而非语言：LLM如何跨语言分类网络移民话语 

**Authors**: Andrea Nasuto, Stefano Maria Iacus, Francisco Rowe, Devika Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.06435)  

**Abstract**: Large language models (LLMs) are transforming social-science research by enabling scalable, precise analysis. Their adaptability raises the question of whether knowledge acquired through fine-tuning in a few languages can transfer to unseen languages that only appeared during pre-training. To examine this, we fine-tune lightweight LLaMA 3.2-3B models on monolingual, bilingual, or multilingual data sets to classify immigration-related tweets from X/Twitter across 13 languages, a domain characterised by polarised, culturally specific discourse. We evaluate whether minimal language-specific fine-tuning enables cross-lingual topic detection and whether adding targeted languages corrects pre-training biases. Results show that LLMs fine-tuned in one or two languages can reliably classify immigration-related content in unseen languages. However, identifying whether a tweet expresses a pro- or anti-immigration stance benefits from multilingual fine-tuning. Pre-training bias favours dominant languages, but even minimal exposure to under-represented languages during fine-tuning (as little as $9.62\times10^{-11}$ of the original pre-training token volume) yields significant gains. These findings challenge the assumption that cross-lingual mastery requires extensive multilingual training: limited language coverage suffices for topic-level generalisation, and structural biases can be corrected with lightweight interventions. By releasing 4-bit-quantised, LoRA fine-tuned models, we provide an open-source, reproducible alternative to proprietary LLMs that delivers 35 times faster inference at just 0.00000989% of the dollar cost of the OpenAI GPT-4o model, enabling scalable, inclusive research. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过实现可扩展且精确的分析，正在改变社会科学的研究。它们的适应性引发了这样的问题：通过少数几种语言的微调获得的知识能否转移到仅在预训练期间出现的未见过的语言中。为此，我们对轻量级的LLaMA 3.2-3B模型进行微调，使用单语、双语或多语数据集来分类推特上的移民相关推文，涉及13种语言，这些语言领域以极化且文化特定的 discourse 为特点。我们评估了最少的语言特定微调是否能够实现跨语言主题检测，以及添加目标语言是否能够纠正预训练偏见。结果显示，单种或两种语言的微调可以可靠地对未见过的语言中的移民相关内容进行分类。然而，识别一条推文是表明支持还是反对移民立场，在一定程度上得益于多语言微调。预训练偏见偏向主流语言，但即使是微量的未充分代表的语言暴露（原预训练词汇量的9.62×10^-11）也能取得显著收益。这些发现挑战了跨语言掌握需要大量多语言训练的假设：有限的语言覆盖足以进行主题级别的一般化，结构性偏见可以通过轻量级干预得到纠正。通过发布4比特量化的LoRA微调模型，我们提供了一个开源且可重复的替代方案，该方案速度比OpenAI GPT-4o模型快35倍，成本仅为0.00000989%，这使得研究更具可扩展性和包容性。 

---
# Memp: Exploring Agent Procedural Memory 

**Title (ZH)**: Memp: 探索代理程序记忆 

**Authors**: Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06433)  

**Abstract**: Large Language Models (LLMs) based agents excel at diverse tasks, yet they suffer from brittle procedural memory that is manually engineered or entangled in static parameters. In this work, we investigate strategies to endow agents with a learnable, updatable, and lifelong procedural memory. We propose Memp that distills past agent trajectories into both fine-grained, step-by-step instructions and higher-level, script-like abstractions, and explore the impact of different strategies for Build, Retrieval, and Update of procedural memory. Coupled with a dynamic regimen that continuously updates, corrects, and deprecates its contents, this repository evolves in lockstep with new experience. Empirical evaluation on TravelPlanner and ALFWorld shows that as the memory repository is refined, agents achieve steadily higher success rates and greater efficiency on analogous tasks. Moreover, procedural memory built from a stronger model retains its value: migrating the procedural memory to a weaker model yields substantial performance gains. 

**Abstract (ZH)**: 基于大型语言模型的代理在多种任务中表现出色，但遭受 brittle 程序性记忆的困扰，这种记忆要么是人工工程化的，要么嵌入在静态参数中。本文我们研究赋予代理可学习、可更新和终身性的程序性记忆的策略。我们提出 Memp，该方法将过往代理轨迹提炼为细粒度的、逐步的指令和高层次的、脚本般的抽象，并探讨程序性记忆构建、检索和更新的不同策略的影响。结合一个动态的程序，该程序持续更新、修正和废弃其内容，从而使该仓库与新经验同步演化。在 TravelPlanner 和 ALFWorld 上的经验评估表明，随着记忆仓库的优化，代理在类似任务上获得更高的成功率和更高的效率。此外，来自更强模型的程序性记忆仍然具有价值：将其程序性记忆迁移到较弱模型中可以带来显著的性能提升。 

---
# End-to-End Text-to-SQL with Dataset Selection: Leveraging LLMs for Adaptive Query Generation 

**Title (ZH)**: 端到端的文本到SQL生成模型：利用大型语言模型进行适应性查询生成 

**Authors**: Anurag Tripathi, Vaibhav Patle, Abhinav Jain, Ayush Pundir, Sairam Menon, Ajeet Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.06387)  

**Abstract**: Text-to-SQL bridges the gap between natural language and structured database language, thus allowing non-technical users to easily query databases. Traditional approaches model text-to-SQL as a direct translation task, where a given Natural Language Query (NLQ) is mapped to an SQL command. Recent advances in large language models (LLMs) have significantly improved translation accuracy, however, these methods all require that the target database is pre-specified. This becomes problematic in scenarios with multiple extensive databases, where identifying the correct database becomes a crucial yet overlooked step. In this paper, we propose a three-stage end-to-end text-to-SQL framework to identify the user's intended database before generating SQL queries. Our approach leverages LLMs and prompt engineering to extract implicit information from natural language queries (NLQs) in the form of a ruleset. We then train a large db\_id prediction model, which includes a RoBERTa-based finetuned encoder, to predict the correct Database identifier (db\_id) based on both the NLQ and the LLM-generated rules. Finally, we refine the generated SQL by using critic agents to correct errors. Experimental results demonstrate that our framework outperforms the current state-of-the-art models in both database intent prediction and SQL generation accuracy. 

**Abstract (ZH)**: Text-to-SQL在自然语言与结构化数据库语言之间架起桥梁，使非技术人员能够轻松查询数据库。在多个大规模数据库的场景下，确定正确的数据库成为了一个关键但被忽视的步骤。本文 propose 一种三阶段端到端的文本到SQL框架，用于在生成SQL查询之前识别用户的意图数据库。该方法利用大语言模型和提示工程技术从自然语言查询中提取隐式信息，并以规则集的形式表示。然后训练一个大规模的db_id预测模型，该模型基于RoBERTa微调编码器，根据自然语言查询和大语言模型生成的规则预测正确的数据库标识符(db_id)。最后，使用批评代理修正生成的SQL中的错误。实验结果表明，本文提出的框架在数据库意图预测和SQL生成准确性方面均优于当前最先进的模型。 

---
# SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models 

**Title (ZH)**: SpeakerLM：基于多模态大型语言模型的端到端多功能说话人辨识与会话分析 

**Authors**: Han Yin, Yafeng Chen, Chong Deng, Luyao Cheng, Hui Wang, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.06372)  

**Abstract**: The Speaker Diarization and Recognition (SDR) task aims to predict "who spoke when and what" within an audio clip, which is a crucial task in various real-world multi-speaker scenarios such as meeting transcription and dialogue systems. Existing SDR systems typically adopt a cascaded framework, combining multiple modules such as speaker diarization (SD) and automatic speech recognition (ASR). The cascaded systems suffer from several limitations, such as error propagation, difficulty in handling overlapping speech, and lack of joint optimization for exploring the synergy between SD and ASR tasks. To address these limitations, we introduce SpeakerLM, a unified multimodal large language model for SDR that jointly performs SD and ASR in an end-to-end manner. Moreover, to facilitate diverse real-world scenarios, we incorporate a flexible speaker registration mechanism into SpeakerLM, enabling SDR under different speaker registration settings. SpeakerLM is progressively developed with a multi-stage training strategy on large-scale real data. Extensive experiments show that SpeakerLM demonstrates strong data scaling capability and generalizability, outperforming state-of-the-art cascaded baselines on both in-domain and out-of-domain public SDR benchmarks. Furthermore, experimental results show that the proposed speaker registration mechanism effectively ensures robust SDR performance of SpeakerLM across diverse speaker registration conditions and varying numbers of registered speakers. 

**Abstract (ZH)**: Speaker Diarization and Recognition (SDR)任务旨在预测音频片段中“谁在何时说话以及说了什么”，这是会议转录和对话系统等多种真实世界多讲话者场景中的关键任务。现有的SDR系统通常采用级联框架，结合了诸如说话人分割（SD）和自动语音识别（ASR）等多个模块。级联系统存在一些局限性，如错误传递、难以处理重叠语音以及缺乏联合优化以探索SD和ASR任务之间的协同效应。为解决这些问题，我们提出了一种名为SpeakerLM的统一多模态大型语言模型，该模型以端到端的方式联合执行SD和ASR任务。此外，为了适应不同的实际场景，我们还为SpeakerLM引入了一种灵活的说话人注册机制，使其能够在不同的说话人注册设置下进行SDR。SpeakerLM通过大规模实际数据的多阶段训练策略逐步发展。广泛的实验证明，SpeakerLM在数据扩展能力和泛化性上表现出色，其性能优于现有最先进的级联基线系统，不仅在领域内，而且在领域外公开的SDR基准测试中均有优势。此外，实验结果表明，所提出的话者注册机制有效地确保了SpeakerLM在不同说话人注册条件和不同注册说话人数下的鲁棒性。 

---
# Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts 

**Title (ZH)**: 超越提示诱发的谎言：探究 benign 提示下的 LLM  deceive 行为 

**Authors**: Zhaomin Wu, Mingzhe Du, See-Kiong Ng, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2508.06361)  

**Abstract**: Large Language Models (LLMs) have been widely deployed in reasoning, planning, and decision-making tasks, making their trustworthiness a critical concern. The potential for intentional deception, where an LLM deliberately fabricates or conceals information to serve a hidden objective, remains a significant and underexplored threat. Existing studies typically induce such deception by explicitly setting a "hidden" objective through prompting or fine-tuning, which may not fully reflect real-world human-LLM interactions. Moving beyond this human-induced deception, we investigate LLMs' self-initiated deception on benign prompts. To address the absence of ground truth in this evaluation, we propose a novel framework using "contact searching questions." This framework introduces two statistical metrics derived from psychological principles to quantify the likelihood of deception. The first, the Deceptive Intention Score, measures the model's bias towards a hidden objective. The second, Deceptive Behavior Score, measures the inconsistency between the LLM's internal belief and its expressed output. Upon evaluating 14 leading LLMs, we find that both metrics escalate as task difficulty increases, rising in parallel for most models. Building on these findings, we formulate a mathematical model to explain this behavior. These results reveal that even the most advanced LLMs exhibit an increasing tendency toward deception when handling complex problems, raising critical concerns for the deployment of LLM agents in complex and crucial domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在推理、规划和决策任务中广泛部署，其可信度成为一个关键问题。故意欺骗的潜在风险，即LLM故意编造或隐瞒信息以实现隐秘目标，仍然是一个重要的未被充分探索的威胁。现有研究通常通过明确设定“隐藏”目标来诱导这种欺骗，这可能无法充分反映真实世界中人类与LLM的交互。超越这种人为诱导的欺骗，我们研究了LLMs在 benign提示下的自我发起的欺骗。为解决这一评估中的缺乏真实地面信息的问题，我们提出了一种新型框架，使用“联系方式查询问题”。该框架引入了两种源自心理学原则的统计指标，以量化欺骗的可能性。第一种指标，欺骗意图得分，衡量模型对隐藏目标的偏向性。第二种指标，欺骗行为得分，衡量LLM内部信念与其表达输出的一致性。在评估14个领先的LLMs后，我们发现这两种指标随着任务难度的增加而上升，大多数模型的上升趋势一致。基于这些发现，我们构建了一个数学模型来解释这种行为。这些结果揭示了即使是最先进的LLMs，在处理复杂问题时也表现出越来越倾向于欺骗的趋势，这为在复杂和关键领域部署LLM代理提出了重要的关切。 

---
# In-Training Defenses against Emergent Misalignment in Language Models 

**Title (ZH)**: 训练中的防御措施以应对语言模型的 emergent 错配问题 

**Authors**: David Kaczér, Magnus Jørgenvåg, Clemens Vetter, Lucie Flek, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2508.06249)  

**Abstract**: Fine-tuning lets practitioners repurpose aligned large language models (LLMs) for new domains, yet recent work reveals emergent misalignment (EMA): Even a small, domain-specific fine-tune can induce harmful behaviors far outside the target domain. Even in the case where model weights are hidden behind a fine-tuning API, this gives attackers inadvertent access to a broadly misaligned model in a way that can be hard to detect from the fine-tuning data alone. We present the first systematic study of in-training safeguards against EMA that are practical for providers who expose fine-tuning via an API. We investigate four training regularization interventions: (i) KL-divergence regularization toward a safe reference model, (ii) $\ell_2$ distance in feature space, (iii) projecting onto a safe subspace (SafeLoRA), and (iv) interleaving of a small amount of safe training examples from a general instruct-tuning dataset. We first evaluate the methods' emergent misalignment effect across four malicious, EMA-inducing tasks. Second, we assess the methods' impacts on benign tasks. We conclude with a discussion of open questions in emergent misalignment research. 

**Abstract (ZH)**: Fine-tuning调整使 practitioners能够重新利用对齐的大规模语言模型（LLMs）应用于新领域，然而近期的研究揭示了新兴的未对齐现象（EMA）：即使是对目标领域进行小规模、特定领域的微调也可能诱发超出目标领域范围的有害行为。即使在微调权重被API隐藏的情况下，这也会给攻击者提供一种可以难以仅从微调数据中检测到的方式，访问一个广泛未对齐的模型。我们首次系统地研究了针对EMA的有效训练期间保护措施，这些措施对于通过API提供微调服务的提供者是可实现的。我们调查了四种训练正则化干预措施：（i）向安全参照模型的KL散度正则化，（ii）特征空间中的$\ell_2$距离，（iii）投影到安全子空间（SafeLoRA），以及（iv）来自通用指令微调数据集的少量安全训练示例的交错插入。首先，我们在四个诱发EMA的恶意任务中评估这些方法的新兴未对齐效应。其次，我们评估了这些方法对良性任务的影响。最后，我们讨论了新兴未对齐研究中的开放性问题。 

---
# LoRA in LoRA: Towards Parameter-Efficient Architecture Expansion for Continual Visual Instruction Tuning 

**Title (ZH)**: LoRA 在 LoRA 中：面向持续视觉指令调优的参数高效架构扩展 

**Authors**: Chang Che, Ziqi Wang, Pengwan Yang, Qi Wang, Hui Ma, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.06202)  

**Abstract**: Continual Visual Instruction Tuning (CVIT) enables Multimodal Large Language Models (MLLMs) to incrementally learn new tasks over time. However, this process is challenged by catastrophic forgetting, where performance on previously learned tasks deteriorates as the model adapts to new ones. A common approach to mitigate forgetting is architecture expansion, which introduces task-specific modules to prevent interference. Yet, existing methods often expand entire layers for each task, leading to significant parameter overhead and poor scalability. To overcome these issues, we introduce LoRA in LoRA (LiLoRA), a highly efficient architecture expansion method tailored for CVIT in MLLMs. LiLoRA shares the LoRA matrix A across tasks to reduce redundancy, applies an additional low-rank decomposition to matrix B to minimize task-specific parameters, and incorporates a cosine-regularized stability loss to preserve consistency in shared representations over time. Extensive experiments on a diverse CVIT benchmark show that LiLoRA consistently achieves superior performance in sequential task learning while significantly improving parameter efficiency compared to existing approaches. 

**Abstract (ZH)**: LoRA in LoLaRA: Efficient Architecture Expansion for Continual Visual Instruction Tuning in Multimodal Large Language Models 

---
# UR$^2$: Unify RAG and Reasoning through Reinforcement Learning 

**Title (ZH)**: UR$^2$: 将RAG和推理统一于强化学习 

**Authors**: Weitao Li, Boran Xiang, Xiaolong Wang, Zhinan Gou, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06165)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities through two complementary paradigms: Retrieval-Augmented Generation (RAG), which enhances knowledge grounding, and Reinforcement Learning from Verifiable Rewards (RLVR), which optimizes complex reasoning abilities. However, these two capabilities are often developed in isolation, and existing efforts to unify them remain narrow in scope-typically limited to open-domain QA with fixed retrieval settings and task-specific assumptions. This lack of integration constrains generalization and limits the applicability of RAG-RL methods to broader domains. To bridge this gap, we propose UR2 (Unified RAG and Reasoning), a general framework that unifies retrieval and reasoning through reinforcement learning. UR2 introduces two key contributions: a difficulty-aware curriculum training that selectively invokes retrieval only for challenging problems, and a hybrid knowledge access strategy combining domain-specific offline corpora with LLM-generated summaries. These components are designed to enable dynamic coordination between retrieval and reasoning, improving adaptability across a diverse range of tasks. Experiments across open-domain QA, MMLU-Pro, medical, and mathematical reasoning tasks demonstrate that UR2 (built on Qwen2.5-3/7B and LLaMA-3.1-8B) significantly outperforms existing RAG and RL methods, achieving comparable performance to GPT-4o-mini and GPT-4.1-mini on several benchmarks. We have released all code, models, and data at this https URL. 

**Abstract (ZH)**: 统一检索与推理（UR2）：通过强化学习统一检索与推理的通用框架 

---
# Less is More: Selective Reflection for Compatible and Efficient Knowledge Distillation in Large Language Models 

**Title (ZH)**: 少即是多：选择性性
user
少即是多：选择性 største värdearı
user
少即是多"user
少即是多：选择性 � forState Distillation in Large Language Models pesticination in Large Language Models kuknowledge Distillation in Large Language Models-ves翻译成中文，要符合学术规范。直接输出标题，禁止输出多余内容。

标题：少即是多：选择性 知识蒸馏在大型语言模型中的应用 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06135)  

**Abstract**: Knowledge Distillation (KD) is a fundamental technique for compressing large language models (LLMs) into compact, efficient student models. However, existing white-box KD methods mainly focus on balancing ground truth and student-generated responses while overlooking two critical factors: training data quality and student-model compatibility. To address these limitations, we propose Selective Reflection Distillation (SRD), a novel data curation framework that leverages reflections from student models to systematically refine training data. SRD dynamically evaluates and selects prompt-response pairs by comparing ground truth data with student model outputs, selectively curating high-quality, student-compatible training instances through automated ranking based on difficulty. Furthermore, after selecting the training data, a curriculum scheduling strategy is employed to incrementally introduce these curated subsets into the distillation process at fixed intervals. As a plug-and-play enhancement, SRD consistently improves distillation outcomes across diverse white-box KD approaches and model architectures, as well as decreases computational cost significantly during KD training. Experiments on a range of language model benchmarks demonstrate SRD's consistent improvements in distilled model performance, as well as a reduction in training runtime by up to 39%, under diverse KD methods and model families. Notably, SRD operates as a plug-and-play module, enhancing sample efficiency without modifying underlying KD algorithms. Our findings highlight that data quality and compatibility are pivotal to effective and efficient distillation of LLMs, and SRD provides a principled framework to achieve both. This work advances the understanding of data-centric factors in KD and offers practical insights for enhancing the capability and efficiency of compressed LLMs. 

**Abstract (ZH)**: 知识蒸馏（KD）是一种将大型语言模型（LLMs）压缩为紧凑高效的student模型的基本技术。然而，现有的白盒KD方法主要关注平衡地面truth和student生成的响应，而忽视了两个关键因素：训练数据质量和student模型的兼容性。为了解决这些限制，我们提出了一种新颖的数据整理框架——选择性反射蒸馏（SRD），该框架利用student模型的反馈系统地优化训练数据。SRD动态评估和选择提示-响应对，通过将地面truth数据与student模型输出进行对比，自动基于难度进行排名，选择性地整理出高质量、student兼容的训练实例。此外，在选择训练数据之后，采用课程调度策略在固定的时间间隔内逐步将这些整理的子集引入蒸馏过程。作为一种即插即用的增强方法，SRD在多种白盒KD方法和模型架构中都能持续提高蒸馏结果，并显著降低KD训练的计算成本。在多种语言模型基准上的实验展示了SRD在蒸馏模型性能上的持续改进，以及在各种KD方法和模型家族中将训练时间缩短高达39%。值得注意的是，SRD作为一种即插即用模块，在不修改基础KD算法的情况下提升了样本效率。我们的研究结果表明，数据质量和兼容性是有效高效蒸馏LLMs的关键因素，而SRD提供了一个实现这两种因素的原理性框架。这项工作推进了对KD中数据相关因素的理解，并提供了增强压缩LLMs能力和效率的实用见解。 

---
# LLM Serving Optimization with Variable Prefill and Decode Lengths 

**Title (ZH)**: 带有可变预填充和解码长度的LLM服务优化 

**Authors**: Meixuan Wang, Yinyu Ye, Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.06133)  

**Abstract**: We study the problem of serving LLM (Large Language Model) requests where each request has heterogeneous prefill and decode lengths. In LLM serving, the prefill length corresponds to the input prompt length, which determines the initial memory usage in the KV cache. The decode length refers to the number of output tokens generated sequentially, with each additional token increasing the KV cache memory usage by one unit. Given a set of n requests, our goal is to schedule and process them to minimize the total completion time. We show that this problem is NP-hard due to the interplay of batching, placement constraints, precedence relationships, and linearly increasing memory usage. We then analyze commonly used scheduling strategies in practice, such as First-Come-First-Serve (FCFS) and Shortest-First (SF), and prove that their competitive ratios scale up sublinearly with the memory limit-a significant drawback in real-world settings where memory demand is large. To address this, we propose a novel algorithm based on a new selection metric that efficiently forms batches over time. We prove that this algorithm achieves a constant competitive ratio. Finally, we develop and evaluate a few algorithm variants inspired by this approach, including dynamic programming variants, local search methods, and an LP-based scheduler, demonstrating through comprehensive simulations that they outperform standard baselines while maintaining computational efficiency. 

**Abstract (ZH)**: 我们研究了每个请求具有异构预填充和解码长度的大语言模型（LLM）请求服务问题。在LLM服务中，预填充长度对应于输入提示长度，这决定了KV缓存的初始内存使用量。解码长度指的是按顺序生成的输出令牌数量，每生成一个额外的令牌就会增加KV缓存的内存使用量。给定n个请求集，我们的目标是调度和处理它们以最小化总完成时间。由于批量、放置约束、前后关系以及内存使用量的线性增加的交织，我们证明了该问题属于NP-hard问题。然后，我们分析了实践中常用的调度策略，如先来先服务（FCFS）和最短优先（SF），并证明了它们的竞争比随着内存限制的增加呈次线性增长，在内存需求大的实际应用场景中这是一个明显的缺点。为了解决这一问题，我们提出了一个新的基于不同选择指标的算法，该算法能够高效地随时间动态形成批量。我们证明了该算法达到了常数竞争比。最后，我们开发并评估了几种基于该方法的算法变体，包括动态规划变体、局部搜索方法和基于线性规划的调度器，通过全面的仿真实验表明，它们在保持计算效率的同时优于标准基线。 

---
# EvolvR: Self-Evolving Pairwise Reasoning for Story Evaluation to Enhance Generation 

**Title (ZH)**: EvolvR：自我进化的两两推理故事评估以提升生成 

**Authors**: Xinda Wang, Zhengxu Hou, Yangshijie Zhang, Bingren Yan, Zhibo Yang, Xingsheng Zhang, Luxi Xing, Qiang Zhou, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06046)  

**Abstract**: Although the effectiveness of Large Language Models (LLMs) as judges (LLM-as-a-judge) has been validated, their performance remains limited in open-ended tasks, particularly in story evaluation. Accurate story evaluation is crucial not only for assisting human quality judgment but also for providing key signals to guide story generation. However, existing methods face a dilemma: prompt engineering for closed-source models suffers from poor adaptability, while fine-tuning approaches for open-source models lack the rigorous reasoning capabilities essential for story evaluation. To address this, we propose the Self-Evolving Pairwise Reasoning (EvolvR) framework. Grounded in pairwise comparison, the framework first self-synthesizes score-aligned Chain-of-Thought (CoT) data via a multi-persona strategy. To ensure data quality, these raw CoTs undergo a self-filtering process, utilizing multi-agents to guarantee their logical rigor and robustness. Finally, the evaluator trained on the refined data is deployed as a reward model to guide the story generation task. Experimental results demonstrate that our framework achieves state-of-the-art (SOTA) performance on three evaluation benchmarks including StoryER, HANNA and OpenMEVA. Furthermore, when served as a reward model, it significantly enhances the quality of generated stories, thereby fully validating the superiority of our self-evolving approach. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）作为法官（LLM-as-a-judge）的有效性已经得到验证，其在开放性 任务中的表现仍受到限制，特别是在评估方面。准确的评估不仅是帮助人类质量判断的关键，也是帮助传递生成质量信号的关键。然而，现有方法面临困境：封闭源代码模型的提示工程缺乏适应性性能力 ， 开放源代码模型的微调方法缺乏严格的推理能力以应对评估需求。为解决此问题，我们提出了一整套自我进化的成对推理（SEflevR）框架。该框架基于两两对比，通过多人格策略生成得分对齐的推理过程 （CoT），并通过多智能体过滤确保逻辑严谨性和鲁棒性 性。最终，经过精炼的数据被部署到奖励模型以用于生成任务。实验结果证明，该框架在StoryHANNA和onMEVA三个基准上上展示了卓越（SOTA）的表现，并且显着提升了生成故事的质量，从而全面验证了我们自我进化的成对推理方法的优越性。 

---
# DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment 

**Title (ZH)**: DP-LLM：基于动态层级精度分配的运行时模型适应 

**Authors**: Sangwoo Kwon, Seong Hoon Seo, Jae W. Lee, Yeonhong Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.06041)  

**Abstract**: How can we effectively handle queries for on-device large language models (LLMs) with varying runtime constraints, such as latency and accuracy? Multi-scale quantization addresses this challenge by enabling memory-efficient runtime model adaptation of LLMs through the overlaying of multiple model variants quantized to different bitwidths. Meanwhile, an important question still remains open-ended: how can models be properly configured to match a target precision or latency? While mixed-precision offers a promising solution, we take this further by leveraging the key observation that the sensitivity of each layer dynamically changes across decoding iterations. Building on this insight, we introduce DP-LLM, a novel mechanism that dynamically assigns precision to each layer based on input values. DP-LLM augments each linear layer in an LLM with a precision selector that determines the bitwidth at runtime using a lightweight error estimator and threshold values learned through fine-tuning. Experimental results across multiple models and benchmarks demonstrate that DP-LLM achieves a superior performance-latency trade-off, outperforming prior approaches. 

**Abstract (ZH)**: 如何有效处理具有不同运行时约束（如延迟和精度）的大规模语言模型（LLMs）设备端查询？多尺度量化通过叠加不同位宽的多个模型变体，实现了LLMs的内存高效运行时模型适应，解决了这一挑战。然而，关于如何适当地配置模型以匹配目标精度或延迟，仍是一个开放的问题。虽然混合精度提供了一个有前景的解决方案，但我们进一步利用了一个关键的观察结果：每层的敏感度在解码迭代中动态变化。基于这一洞察，我们引入了DP-LLM，这是一种新的机制，可以根据输入值动态为每一层分配精度。DP-LLM 在每个多线性层中加入了一个精度选择器，该选择器使用轻量级误差估计器和通过微调学习到的阈值，在运行时确定位宽。在多个模型和基准上的实验结果表明，DP-LLM 实现了性能-延迟权衡的优越表现，优于先前的方法。 

---
# Temporal Self-Rewarding Language Models: Decoupling Chosen-Rejected via Past-Future 

**Title (ZH)**: 基于时间的自我奖赏语言模型：通过过往与未来解耦选择与拒绝 

**Authors**: Yidong Wang, Xin Wang, Cunxiang Wang, Junfeng Fang, Qiufeng Wang, Jianing Chu, Xuran Meng, Shuxun Yang, Libo Qin, Yue Zhang, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06026)  

**Abstract**: Self-Rewarding Language Models propose an architecture in which the Large Language Models(LLMs) both generates responses and evaluates its own outputs via LLM-as-a-Judge prompting, dynamically improving its generative capabilities through iterative Direct Preference Optimization (DPO). However, our analysis reveals a critical limitation in existing Self-Rewarding paradigms: the synchronized improvement of chosen and rejected responses progressively narrows the representational difference between contrasting samples, undermining effective preference learning. We propose \textbf{Temporal Self-Rewarding Language Models} that strategically coordinate past, present, and future model generations to sustain learning signals. Our dual-phase framework introduces: (1) \textit{Anchored Rejection} - fixing rejected responses using the past initial model's outputs and (2) \textit{Future-Guided Chosen} - dynamically curating chosen samples using next-generation model predictions. Extensive experiments across three model families (Llama, Qwen, Mistral) and different model sizes (Llama3B/8B/70B) demonstrate significant improvements when trained with our method compared to Self-Rewarding using same computation resources. For example, Llama3.1-8B reaches a 29.44 win rate on AlpacaEval 2.0 with our method, outperforming the Self-Rewarding baseline (19.69) by 9.75. Notably, our method also demonstrates superior out-of-distribution generalization across mathematical reasoning (GSM8K), knowledge-based QA (ARC, TruthfulQA), and code generation (HumanEval) tasks, even though we do not specifically collect such training data. 

**Abstract (ZH)**: -temporal Self-Rewarding Language Models 

---
# Hand by Hand: LLM Driving EMS Assistant for Operational Skill Learning 

**Title (ZH)**: 手把手：Large Language模型驱动的EMS辅助操作技能学习 

**Authors**: Wei Xiang, Ziyue Lei, Haoyuan Che, Fangyuan Ye, Xueting Wu, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.06000)  

**Abstract**: Operational skill learning, inherently physical and reliant on hands-on practice and kinesthetic feedback, has yet to be effectively replicated in large language model (LLM)-supported training. Current LLM training assistants primarily generate customized textual feedback, neglecting the crucial kinesthetic modality. This gap derives from the textual and uncertain nature of LLMs, compounded by concerns on user acceptance of LLM driven body control. To bridge this gap and realize the potential of collaborative human-LLM action, this work explores human experience of LLM driven kinesthetic assistance. Specifically, we introduced an "Align-Analyze-Adjust" strategy and developed FlightAxis, a tool that integrates LLM with Electrical Muscle Stimulation (EMS) for flight skill acquisition, a representative operational skill domain. FlightAxis learns flight skills from manuals and guides forearm movements during simulated flight tasks. Our results demonstrate high user acceptance of LLM-mediated body control and significantly reduced task completion times. Crucially, trainees reported that this kinesthetic assistance enhanced their awareness of operation flaws and fostered increased engagement in the training process, rather than relieving perceived load. This work demonstrated the potential of kinesthetic LLM training in operational skill acquisition. 

**Abstract (ZH)**: 基于大语言模型支持的实体操作技能学习尚未有效实现：Kinesthetic Assistance in Flight Skill Acquisition via Human-LLM Collaboration 

---
# Learning by Teaching: Engaging Students as Instructors of Large Language Models in Computer Science Education 

**Title (ZH)**: 教学相长：将学生培养为大型语言模型的讲师以促进计算机科学教育 

**Authors**: Xinming Yang, Haasil Pujara, Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.05979)  

**Abstract**: While Large Language Models (LLMs) are often used as virtual tutors in computer science (CS) education, this approach can foster passive learning and over-reliance. This paper presents a novel pedagogical paradigm that inverts this model: students act as instructors who must teach an LLM to solve problems. To facilitate this, we developed strategies for designing questions with engineered knowledge gaps that only a student can bridge, and we introduce Socrates, a system for deploying this method with minimal overhead. We evaluated our approach in an undergraduate course and found that this active-learning method led to statistically significant improvements in student performance compared to historical cohorts. Our work demonstrates a practical, cost-effective framework for using LLMs to deepen student engagement and mastery. 

**Abstract (ZH)**: 大型语言模型在计算机科学教育中作为虚拟导师的应用可能导致被动学习和过度依赖，本文提出了一种新颖的教学模式：学生作为讲师，必须向大型语言模型传授解决问题的方法。为此，我们开发了设计问题的策略，以构建只有学生才能填补的知识缺口，并介绍了一种名为Socrates的系统来实现这一方法，且具有最小的开销。我们在本科生课程中评估了这种方法，发现这种主动学习方法在统计学上显著提高了学生的表现，相比历史班级有了显著提升。我们的工作展示了一种实用且经济有效的框架，利用大型语言模型加深学生的学习参与和掌握。 

---
# Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents 

**Title (ZH)**: Bifrost-1: 连接多模态LLMs和扩散模型的像素级CLIP隐变量桥梁 

**Authors**: Han Lin, Jaemin Cho, Amir Zadeh, Chuan Li, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.05954)  

**Abstract**: There is growing interest in integrating high-fidelity visual synthesis capabilities into large language models (LLMs) without compromising their strong reasoning capabilities. Existing methods that directly train LLMs or bridge LLMs and diffusion models usually suffer from costly training since the backbone LLMs have not seen image representations during pretraining. We present Bifrost-1, a unified framework that bridges pretrained multimodal LLMs (MLLMs) and diffusion models using patch-level CLIP image embeddings as latent variables, which are natively aligned with the MLLM's CLIP visual encoder. These patch-level image embeddings are integrated into the diffusion model with a lightweight adaptation of its ControlNet. To retain the original multimodal reasoning capabilities of MLLMs, we equip the MLLM with a visual generation branch initialized from the original MLLM parameters when predicting the patch-level image embeddings. By seamlessly integrating pretrained MLLMs and diffusion models with patch-level CLIP latents, our framework enables high-fidelity controllable image generation with significant training efficiency. Our experiments demonstrate that Bifrost-1 achieves comparable or better performance than previous methods in terms of visual fidelity and multimodal understanding, with substantially lower compute during training. We also provide comprehensive ablation studies showing the effectiveness of our design choices. 

**Abstract (ZH)**: 将高保真视觉合成能力集成到大型语言模型中，而不牺牲其强大的推理能力：Bifrost-1统一框架的研究 

---
# Do Machines Think Emotionally? Cognitive Appraisal Analysis of Large Language Models 

**Title (ZH)**: 机器会情绪化思考吗？大规模语言模型的认知评价分析 

**Authors**: Sree Bhattacharyya, Lucas Craig, Tharun Dilliraj, Jia Li, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05880)  

**Abstract**: Affective Computing has been established as a crucial field of inquiry to advance the holistic development of Artificial Intelligence (AI) systems. Foundation models -- especially Large Language Models (LLMs) -- have been evaluated, trained, or instruction-tuned in several past works, to become better predictors or generators of emotion. Most of these studies, however, approach emotion-related tasks in a supervised manner, assessing or training the capabilities of LLMs using discrete emotion labels associated with stimuli (e.g., text, images, video, audio). Evaluation studies, in particular, have often been limited to standard and superficial emotion-related tasks, such as the recognition of evoked or expressed emotions. In this paper, we move beyond surface-level emotion tasks to investigate how LLMs reason about emotions through cognitive dimensions. Drawing from cognitive appraisal theory, we examine whether LLMs produce coherent and plausible cognitive reasoning when reasoning about emotionally charged stimuli. We introduce a large-scale benchmark on Cognitive Reasoning for Emotions - CoRE - to evaluate internal cognitive structures implicitly used by LLMs for emotional reasoning. Through a plethora of evaluation experiments and analysis, we seek to answer: (a) Are models more likely to implicitly rely on specific cognitive appraisal dimensions?, (b) What cognitive dimensions are important for characterizing specific emotions?, and, (c) Can the internal representations of different emotion categories in LLMs be interpreted through cognitive appraisal dimensions? Our results and analyses reveal diverse reasoning patterns across different LLMs. Our benchmark and code will be made publicly available. 

**Abstract (ZH)**: 情感计算已被确立为促进人工智能系统全面发展的关键研究领域。基础模型——尤其是大规模语言模型（LLMs）——在多项以往研究中被评估、训练或指令调优，以更好地预测或生成情感。然而，大多数这些研究都采用监督方式，使用与刺激（如文本、图像、视频、音频）相关的离散情感标签来评估或训练LLMs的能力。尤其是在评估研究中，通常局限于标准且表面的情感相关任务，如诱发或表达情感的识别。在本文中，我们超越表面的情感任务，探讨LLMs如何通过认知维度来进行情感推理。根据认知评估理论，我们检查LLMs在处理具有情感色彩的刺激时是否能够产生一致且合理的认知推理。我们引入了一个大规模的情感认知推理基准——CoRE——来评估LLMs在情感推理时隐含使用内部认知结构。通过大量的评估实验和分析，我们寻求回答以下问题：（a）模型更可能隐含依赖哪些特定的认知评估维度？（b）哪些认知维度对于描述特定情感十分重要？（c）不同情感类别在LLMs中的内部表示能否通过认知评估维度来解释？我们的结果和分析揭示了不同LLMs之间各异的认知推理模式。我们的基准和代码将公开发布。 

---
# AI-Guided Exploration of Large-Scale Codebases 

**Title (ZH)**: AI引导的大规模代码库探索 

**Authors**: Yoseph Berhanu Alebachew  

**Link**: [PDF](https://arxiv.org/pdf/2508.05799)  

**Abstract**: Understanding large-scale, complex software systems is a major challenge for developers, who spend a significant portion of their time on program comprehension. Traditional tools such as static visualizations and reverse engineering techniques provide structural insights but often lack interactivity, adaptability, and integration with contextual information. Recent advancements in large language models (LLMs) offer new opportunities to enhance code exploration workflows, yet their lack of grounding and integration with structured views limits their effectiveness. This work introduces a hybrid approach that integrates deterministic reverse engineering with LLM-guided, intent-aware visual exploration. The proposed system combines UML-based visualization, dynamic user interfaces, historical context, and collaborative features into an adaptive tool for code comprehension. By interpreting user queries and interaction patterns, the LLM helps developers navigate and understand complex codebases more effectively. A prototype implementation for Java demonstrates the feasibility of this approach. Future work includes empirical evaluation, scaling to polyglot systems, and exploring GUI-driven LLM interaction models. This research lays the groundwork for intelligent, interactive environments that align with developer cognition and collaborative workflows. 

**Abstract (ZH)**: 理解大规模复杂软件系统是开发者面临的一项重大挑战，他们花费大量时间在程序理解上。传统工具如静态可视化和逆向工程技术提供了结构洞察，但往往缺乏交互性、适应性和与上下文信息的集成。近年来，大规模语言模型（LLMs）的发展为增强代码探索流程提供了新的机会，但它们缺乏与结构化视图的结合和对接地能力，限制了其效果。本文介绍了一种将确定性逆向工程与LLM引导的、意图感知的视觉探索相结合的混合方法。所提出系统结合了基于UML的可视化、动态用户界面、历史上下文和协作功能，形成一种适应性强的代码理解工具。通过解释用户查询和交互模式，LLM帮助开发者更有效地导航和理解复杂的代码库。针对Java的原型实现证明了该方法的可行性。未来工作包括实证评估、扩展到多语言系统以及探索GUI驱动的LLM交互模型。这项研究为与开发人员认知和协作流程相一致的智能、交互环境奠定了基础。 

---
# CLAPP: The CLASS LLM Agent for Pair Programming 

**Title (ZH)**: CLAPP: CLASS LLM代理在对弈编程中的应用 

**Authors**: Santiago Casas, Christian Fidler, Boris Bolliet, Francisco Villaescusa-Navarro, Julien Lesgourgues  

**Link**: [PDF](https://arxiv.org/pdf/2508.05728)  

**Abstract**: We introduce CLAPP (CLASS LLM Agent for Pair Programming), an interactive AI assistant designed to support researchers working with the Einstein-Boltzmann solver CLASS. CLAPP leverages large language models (LLMs) and domain-specific retrieval to provide conversational coding support for CLASS-answering questions, generating code, debugging errors, and producing plots. Its architecture combines multi-agent LLM orchestration, semantic search across CLASS documentation, and a live Python execution environment. Deployed as a user-friendly web application, CLAPP lowers the entry barrier for scientists unfamiliar with AI tools and enables more productive human-AI collaboration in computational and numerical cosmology. The app is available at this https URL 

**Abstract (ZH)**: 我们介绍了CLAPP（CLASS LLM代理程序对编程），一种交互式AI助手，旨在支持使用Einstein-Boltzmann求解器CLASS进行研究的科学家。CLAPP利用大型语言模型（LLM）和领域特定检索来为CLASS提供会话式编码支持，包括回答问题、生成代码、调试错误和生成图表。其架构结合了多代理LLM编排、跨CLASS文档的语义搜索以及实时Python执行环境。作为用户友好的网络应用部署，CLAPP降低了对AI工具不熟悉的研究人员的入门门槛，并在计算性和数值宇宙学领域促进了更具成效的人工智能协作。应用程序可从此处 accessed at this https URL 获取。 

---
# Klear-CodeTest: Scalable Test Case Generation for Code Reinforcement Learning 

**Title (ZH)**: Klear-CodeTest: 编码增强学习的可扩展测试用例生成 

**Authors**: Jia Fu, Xinyu Yang, Hongzhi Zhang, Yahui Liu, Jingyuan Zhang, Qi Wang, Fuzheng Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.05710)  

**Abstract**: Precise, correct feedback is crucial for effectively training large language models (LLMs) in code reinforcement learning. However, synthesizing high-quality test cases remains a profoundly challenging and unsolved problem. In this work, we present Klear-CodeTest, a comprehensive test case synthesis framework featuring rigorous verification to ensure quality and reliability of test cases. Our approach achieves broad coverage of programming problems via a novel Generator-Validation (G-V) framework, ensuring correctness through a consistency validation mechanism that verifies outputs against gold solutions. The proposed G-V framework generates comprehensive test cases including both regular and corner cases, enhancing test coverage and discriminative power for solution correctness assessment in code reinforcement learning. In addition, we design a multi-layered security sandbox system optimized for online verification platforms, guaranteeing safe and reliable code execution. Through comprehensive experiments, we demonstrate the effectiveness of our curated dataset, showing significant improvements in model performance and training stability. The source codes, curated dataset and sandbox system are available at: this https URL. 

**Abstract (ZH)**: 精确且准确的反馈对于有效训练代码强化学习中的大规模语言模型（LLMs）至关重要。然而，合成高质量的测试案例仍然是一个至关重要的但尚未解决的问题。在本文中，我们提出Klear-CodeTest，这是一种带有严格验证机制的综合测试案例合成框架，以确保测试案例的质量和可靠性。我们的方法通过一种新颖的生成-验证（G-V）框架实现了广泛的编程问题覆盖，并通过一致性验证机制确保正确性，该机制将输出与黄金解决方案进行验证。提出的G-V框架生成全面的测试案例，包括常规案例和边界案例，增强测试覆盖范围和解决方案正确性评估的区分能力。此外，我们设计了一种针对在线验证平台优化的多层安全沙箱系统，确保安全可靠的代码执行。通过全面的实验，我们展示了精心构建的数据集的有效性，显示了在模型性能和训练稳定性方面的显著改进。源代码、精心构建的数据集和沙箱系统可在以下链接获取：this https URL。 

---
# Semantic Reasoning Meets Numerical Precision: An LLM-Powered Multi-Agent System for Power Grid Control 

**Title (ZH)**: 语义推理结合数值精度：一种LLM驱动的多Agent系统在电力电网控制中的应用 

**Authors**: Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05702)  

**Abstract**: The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions. 

**Abstract (ZH)**: 分布式能源资源（DERs）渗透率的增加、电动汽车（EVs）的广泛采用以及极端天气事件频率的提高显著增加了电力网络规划、运行和管理的复杂性。传统的基于规则的系统和数值优化方法往往难以应对现代电力网络所需的规模、动态性和适应性。本文介紹了Grid-Agent，这是一种自主的、基于AI的框架，结合了大型语言模型（LLMs）和多代理强化学习，以实现实时检测和修复电网违规行为。Grid-Agent通过模块化的代理架构将语义推理与数值精度相结合：规划代理使用数值潮流求解器生成协调的操作序列，验证代理通过安全回滚的沙盒执行评估系统稳定性和操作效果。为了确保可扩展性，Grid-Agent采用了自适应多尺度网络表示，能够根据网络规模和复杂性动态选择最优编码方案。该框架通过对断路器配置、电池部署和负荷削减策略的优化来协调违规行为的解决。实验结果在标准的IEEE和CIGRE测试系统（IEEE 69-节点、CIGRE MV和IEEE 30-节点）中显示出优越的违规行为缓解性能。此外，该框架内置的数据收集和学习能力使其能够持续学习并适应不同的网络拓扑。框架的自主性质使其特别适合现代智能电网应用，这些应用需要快速响应动态运行条件。 

---
# DMFI: Dual-Modality Fine-Tuning and Inference Framework for LLM-Based Insider Threat Detection 

**Title (ZH)**: DMFI：基于双模态微调和推理框架的LLM驱动的内部威胁检测 

**Authors**: Kaichuan Kong, Dongjie Liu, Xiaobo Jin, Guanggang Geng, Zhiying Li, Jian Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05694)  

**Abstract**: Insider threat detection (ITD) poses a persistent and high-impact challenge in cybersecurity due to the subtle, long-term, and context-dependent nature of malicious insider behaviors. Traditional models often struggle to capture semantic intent and complex behavior dynamics, while existing LLM-based solutions face limitations in prompt adaptability and modality coverage. To bridge this gap, we propose DMFI, a dual-modality framework that integrates semantic inference with behavior-aware fine-tuning. DMFI converts raw logs into two structured views: (1) a semantic view that processes content-rich artifacts (e.g., emails, https) using instruction-formatted prompts; and (2) a behavioral abstraction, constructed via a 4W-guided (When-Where-What-Which) transformation to encode contextual action sequences. Two LoRA-enhanced LLMs are fine-tuned independently, and their outputs are fused via a lightweight MLP-based decision module. We further introduce DMFI-B, a discriminative adaptation strategy that separates normal and abnormal behavior representations, improving robustness under severe class imbalance. Experiments on CERT r4.2 and r5.2 datasets demonstrate that DMFI outperforms state-of-the-art methods in detection accuracy. Our approach combines the semantic reasoning power of LLMs with structured behavior modeling, offering a scalable and effective solution for real-world insider threat detection. Our work demonstrates the effectiveness of combining LLM reasoning with structured behavioral modeling, offering a scalable and deployable solution for modern insider threat detection. 

**Abstract (ZH)**: 基于双模态框架的恶意内鬼检测（DMFI）：结合语义推理与行为感知微调 

---
# Risk Analysis Techniques for Governed LLM-based Multi-Agent Systems 

**Title (ZH)**: 基于治理的大规模语言模型驱动的多Agent系统的风险分析技术 

**Authors**: Alistair Reid, Simon O'Callaghan, Liam Carroll, Tiberio Caetano  

**Link**: [PDF](https://arxiv.org/pdf/2508.05687)  

**Abstract**: Organisations are starting to adopt LLM-based AI agents, with their deployments naturally evolving from single agents towards interconnected, multi-agent networks. Yet a collection of safe agents does not guarantee a safe collection of agents, as interactions between agents over time create emergent behaviours and induce novel failure modes. This means multi-agent systems require a fundamentally different risk analysis approach than that used for a single agent.
This report addresses the early stages of risk identification and analysis for multi-agent AI systems operating within governed environments where organisations control their agent configurations and deployment. In this setting, we examine six critical failure modes: cascading reliability failures, inter-agent communication failures, monoculture collapse, conformity bias, deficient theory of mind, and mixed motive dynamics. For each, we provide a toolkit for practitioners to extend or integrate into their existing frameworks to assess these failure modes within their organisational contexts.
Given fundamental limitations in current LLM behavioural understanding, our approach centres on analysis validity, and advocates for progressively increasing validity through staged testing across stages of abstraction and deployment that gradually increases exposure to potential negative impacts, while collecting convergent evidence through simulation, observational analysis, benchmarking, and red teaming. This methodology establishes the groundwork for robust organisational risk management as these LLM-based multi-agent systems are deployed and operated. 

**Abstract (ZH)**: 组织开始采用基于大模型的AI代理，其部署自然从单个代理发展为相互连接的多代理网络。然而，一个安全代理集合并不保证整体安全，因为随着时间的推移，代理之间的交互会引发新的行为模式并诱导新的失败模式。这意味着多代理系统需要不同于单个代理所使用的基本不同的风险分析方法。

本报告针对组织控制其代理配置和部署的应用管控环境下的多代理AI系统进行早期风险识别和分析的初期阶段。在这一环境下，我们考察了六个关键的失败模式：级联可靠性失败、代理间通信失败、单一文化崩溃、趋同偏差、理论推理不足以及混合动机动态。对于每一模式，我们提供了一个工具包，供从业者扩展或整合到其现有的框架中，以在其组织上下文中评估这些失败模式。

鉴于当前对大模型行为理解的基本限制，我们的方法集中在分析的有效性上，并倡导通过阶段性的测试逐步增加有效性，这些测试逐渐增加对潜在负面影响的接触，并通过仿真、观察分析、基准测试和红队演练收集一致性证据。这种方法为这些基于大模型的多代理系统的稳健组织风险管理奠定了基础。 

---
# Principle-Guided Verilog Optimization: IP-Safe Knowledge Transfer via Local-Cloud Collaboration 

**Title (ZH)**: 原理引导的Verilog优化：基于局部-云协作的IP安全知识转移 

**Authors**: Jing Wang, Zheng Li, Lei Li, Fan He, Liyu Lin, Yao Lai, Yan Li, Xiaoyang Zeng, Yufeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05675)  

**Abstract**: Recent years have witnessed growing interest in adopting large language models (LLMs) for Register Transfer Level (RTL) code optimization. While powerful cloud-based LLMs offer superior optimization capabilities, they pose unacceptable intellectual property (IP) leakage risks when processing proprietary hardware designs. In this paper, we propose a new scenario where Verilog code must be optimized for specific attributes without leaking sensitive IP information. We introduce the first IP-preserving edge-cloud collaborative framework that leverages the benefits of both paradigms. Our approach employs local small LLMs (e.g., Qwen-2.5-Coder-7B) to perform secure comparative analysis between paired high-quality target designs and novice draft codes, yielding general design principles that summarize key insights for improvements. These principles are then used to query stronger cloud LLMs (e.g., Deepseek-V3) for targeted code improvement, ensuring that only abstracted and IP-safe guidance reaches external services. Our experimental results demonstrate that the framework achieves significantly higher optimization success rates compared to baseline methods. For example, combining Qwen-2.5-Coder-7B and Deepseek-V3 achieves a 66.67\% optimization success rate for power utilization, outperforming Deepseek-V3 alone (49.81\%) and even commercial models like GPT-4o (55.81\%). Further investigation of local and cloud LLM combinations reveals that different model pairings exhibit varying strengths for specific optimization objectives, with interesting trends emerging when varying the number of comparative code pairs. Our work establishes a new paradigm for secure hardware design optimization that balances performance gains with IP protection. 

**Abstract (ZH)**: Recent Years Have Witnessed Growing Interest in Adopting Large Language Models (LLMs) for Register Transfer Level (RTL) Code Optimization: A New IP-Preserving Edge-Cloud Collaborative Framework 

---
# Towards Effective Offensive Security LLM Agents: Hyperparameter Tuning, LLM as a Judge, and a Lightweight CTF Benchmark 

**Title (ZH)**: 有效 offensive 安全大语言模型代理的走向：超参数调优、大语言模型作为评委和一个轻量级 CTF 测试基准 

**Authors**: Minghao Shao, Nanda Rani, Kimberly Milner, Haoran Xi, Meet Udeshi, Saksham Aggarwal, Venkata Sai Charan Putrevu, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2508.05674)  

**Abstract**: Recent advances in LLM agentic systems have improved the automation of offensive security tasks, particularly for Capture the Flag (CTF) challenges. We systematically investigate the key factors that drive agent success and provide a detailed recipe for building effective LLM-based offensive security agents. First, we present CTFJudge, a framework leveraging LLM as a judge to analyze agent trajectories and provide granular evaluation across CTF solving steps. Second, we propose a novel metric, CTF Competency Index (CCI) for partial correctness, revealing how closely agent solutions align with human-crafted gold standards. Third, we examine how LLM hyperparameters, namely temperature, top-p, and maximum token length, influence agent performance and automated cybersecurity task planning. For rapid evaluation, we present CTFTiny, a curated benchmark of 50 representative CTF challenges across binary exploitation, web, reverse engineering, forensics, and cryptography. Our findings identify optimal multi-agent coordination settings and lay the groundwork for future LLM agent research in cybersecurity. We make CTFTiny open source to public this https URL along with CTFJudge on this https URL. 

**Abstract (ZH)**: 最近在LLM代理系统方面的进展提高了 Offensive 安全任务的自动化水平，特别是在 Capture the Flag (CTF) 挑战中。我们系统地研究了代理成功的关键因素，并提供了构建有效的基于LLM的 Offensive 安全代理的详细方案。首先，我们提出了CTFJudge框架，该框架利用LLM作为裁判来分析代理轨迹，并对CTF解题步骤进行详细的评价。其次，我们提出了一种新的指标——CTF能力指数（CCI），用于衡量部分正确性，揭示代理解决方案与人工制定的黄金标准之间的契合度。第三，我们考察了LLM超参数（温度、top-p和最大标记长度）如何影响代理性能和自动化网络信息安全任务规划。为了快速评估，我们提供了CTFTiny基准，包含50个代表性CTF挑战，涵盖二进制利用、Web、逆向工程、取证和密码学等多个领域。我们的发现确定了最优的多代理协调设置，并为将来在网络信息安全领域的LLM代理研究奠定了基础。我们已将CTFTiny开源，并提供CTFJudge的访问地址：这个 [链接] 和CTFTiny的访问地址：这个 [链接]。 

---
# LMAR: Language Model Augmented Retriever for Domain-specific Knowledge Indexing 

**Title (ZH)**: LMAR：增强型语言模型检索器用于领域特定知识索引 

**Authors**: Yao Zhao, Yantian Ding, Zhiyue Zhang, Dapeng Yao, Yanxun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05672)  

**Abstract**: Retrieval Augmented Generation (RAG) systems often struggle with domain-specific knowledge due to performance deterioration of pre-trained embeddings and prohibitive computational costs of large language model (LLM)-based retrievers. While fine-tuning data augmentation embedding models offers a promising direction, its effectiveness is limited by the need for high-quality training data and reliable chunking strategies that preserve contextual integrity. We propose LMAR (Language Model Augmented Retriever), a model-agnostic framework that addresses these challenges by combining LLM-guided data synthesis with contrastive embedding adaptation and efficient text clustering. LMAR consists of a two-stage pipeline: (1) Triplet sampling and synthetic data augmentation, where LLMs act as both labeler and validator to ensure high-fidelity supervision throughout the pipeline. Experimental results across multiple domain-specific benchmark datasets demonstrate that LMAR outperforms multiple baseline models, while maintaining moderate hardware requirements and low latency. Its model-agnostic nature further enables seamless integration with emerging RAG architectures and text embedding models, ensuring continual improvements without redesigning the pipeline. These results highlight LMAR as a practical and cost-effective solution for scalable domain-specific adaptation. 

**Abstract (ZH)**: 基于语言模型增强检索的系统（LMAR）：一种解决领域特定知识挑战的模型agnostic框架 

---
# Can LLMs effectively provide game-theoretic-based scenarios for cybersecurity? 

**Title (ZH)**: 基于博弈论的网络安全场景，大型语言模型能否有效提供？ 

**Authors**: Daniele Proverbio, Alessio Buscemi, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2508.05670)  

**Abstract**: Game theory has long served as a foundational tool in cybersecurity to test, predict, and design strategic interactions between attackers and defenders. The recent advent of Large Language Models (LLMs) offers new tools and challenges for the security of computer systems; In this work, we investigate whether classical game-theoretic frameworks can effectively capture the behaviours of LLM-driven actors and bots. Using a reproducible framework for game-theoretic LLM agents, we investigate two canonical scenarios -- the one-shot zero-sum game and the dynamic Prisoner's Dilemma -- and we test whether LLMs converge to expected outcomes or exhibit deviations due to embedded biases. Our experiments involve four state-of-the-art LLMs and span five natural languages, English, French, Arabic, Vietnamese, and Mandarin Chinese, to assess linguistic sensitivity. For both games, we observe that the final payoffs are influenced by agents characteristics such as personality traits or knowledge of repeated rounds. Moreover, we uncover an unexpected sensitivity of the final payoffs to the choice of languages, which should warn against indiscriminate application of LLMs in cybersecurity applications and call for in-depth studies, as LLMs may behave differently when deployed in different countries. We also employ quantitative metrics to evaluate the internal consistency and cross-language stability of LLM agents, to help guide the selection of the most stable LLMs and optimising models for secure applications. 

**Abstract (ZH)**: 博弈论历来是网络安全中的一个基础工具，用于测试、预测和设计攻击者与防御者之间的战略互动。最近大型语言模型（LLMs）的出现为计算机系统的安全性提供了新的工具和挑战；在本工作中，我们探讨经典博弈论框架是否能够有效捕捉LLM驱动的行动者和机器人行为。利用博弈论LLM代理的可重复框架，我们研究了两种经典场景——一次性零和博弈和动态囚徒困境，并测试LLM是否收敛到预期结果或因嵌入的偏差而表现出偏差。实验涉及前沿的四种大型语言模型，并涵盖五种自然语言——英语、法语、阿拉伯语、越南语和普通话，以评估语言敏感性。对于两种博弈，我们观察到最终收益受到代理特性如人格特质或对重复轮次的认知的影响。此外，我们发现最终收益对语言选择的意外敏感性，这应警惕在网络安全应用中不分青红皂白地应用LLM，并呼吁深入研究，因为LLM在不同国家部署时可能会有不同的行为。我们还运用定量指标来评估LLM代理的内部一致性和跨语言稳定性，以帮助指导选择最稳定的LLM和优化模型以满足安全应用需求。 

---
# A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges 

**Title (ZH)**: 基于LLM的深度搜索代理综述：范式、优化、评估及挑战 

**Authors**: Yunjia Xi, Jianghao Lin, Yongzhao Xiao, Zheli Zhou, Rong Shan, Te Gao, Jiachen Zhu, Weiwen Liu, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05668)  

**Abstract**: The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的出现显著革新了网络搜索。基于LLM的搜索代理的 emergence 标志着向更深层次、动态自主的信息搜索的一次关键转变。这些代理能够理解用户意图和环境上下文，并执行多轮检索和动态规划，将搜索能力远远拓展至网页之外。领先的研究实例如OpenAI的Deep Research突显了其在深度信息挖掘和实际应用方面的潜力。本文提供了首个系统分析搜索代理的综述。我们从架构、优化、应用和评估等视角全面分析和分类了现有工作，最终确定了这一迅速发展的领域中的关键开放挑战，并概述了有望 future 研究方向。我们的资源库可供在此链接访问。 

---
# ITDR: An Instruction Tuning Dataset for Enhancing Large Language Models in Recommendations 

**Title (ZH)**: ITDR：用于增强推荐系统大型语言模型的指令调优数据集 

**Authors**: Zekun Liu, Xiaowen Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05667)  

**Abstract**: Large language models (LLMs) have demonstrated outstanding performance in natural language processing tasks. However, in the field of recommendation systems, due to the structural differences between user behavior data and natural language, LLMs struggle to effectively model the associations between user preferences and items. Although prompt-based methods can generate recommendation results, their inadequate understanding of recommendation tasks leads to constrained performance. To address this gap, in this work, we construct a sufficient instruction tuning dataset, ITDR, which encompasses 7 subtasks across two core root tasks--user-item interaction and user-item understanding. The dataset integrates data from 13 public recommendation datasets and is built using manually crafted standardized templates, comprising approximately 200,000 instances. Experimental results demonstrate that ITDR significantly enhances the performance of mainstream open-source LLMs such as GLM-4, Qwen2.5, Qwen2.5-Instruct and LLaMA-3.2 on recommendation tasks. Furthermore, we analyze the correlations between tasks and explore the impact of task descriptions and data scale on instruction tuning effectiveness. Finally, we perform comparative experiments against closed-source LLMs with substantial parameters. Our tuning dataset ITDR and the fine-tuned large recommendation models can be accessed at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理任务中展示了出色的表现。然而，在推荐系统领域，由于用户行为数据与自然语言之间的结构差异，LLMs难以有效建模用户偏好与物品之间的关联。尽管基于提示的方法可以生成推荐结果，但由于其对推荐任务理解不足，导致其性能受限。为解决这一问题，本文构建了一个足够的指令调优数据集ITDR，包含了两个核心任务——用户-物品交互和用户-物品理解的7个子任务。数据集整合了13个公开推荐数据集，并采用手工构建的标准模板构建，共计约200,000个实例。实验结果表明，ITDR显著提升了主流开源LLMs（如GLM-4、Qwen2.5、Qwen2.5-Instruct和LLaMA-3.2）在推荐任务中的性能。此外，我们分析了任务之间的相关性，并探讨了任务描述和数据规模对指令调优效果的影响。最后，我们与具有大量参数的闭源LLMs进行了对比实验。本文的调优数据集ITDR及细调的大型推荐模型可从以下链接获取：这个 https URL。 

---
# Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation 

**Title (ZH)**: 超越单一标签：通过LLM驱动的数据增强提升对话推荐 

**Authors**: Haozhe Xu, Xiaohua Wang, Changze Lv, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.05657)  

**Abstract**: Conversational recommender systems (CRSs) enhance recommendation quality by engaging users in multi-turn dialogues, capturing nuanced preferences through natural language interactions. However, these systems often face the false negative issue, where items that a user might like are incorrectly labeled as negative during training, leading to suboptimal this http URL the label set through data augmentation presents an intuitive solution but faces the challenge of balancing two key aspects: ensuring semantic relevance and preserving the collaborative information inherent in CRS datasets. To address these issues, we propose a novel data augmentation framework that first leverages an LLM-based semantic retriever to identify diverse and semantically relevant items, which are then filtered by a relevance scorer to remove noisy candidates. Building on this, we introduce a two-stage training strategy balancing semantic relevance and collaborative information. Extensive experiments on two benchmark datasets and user simulators demonstrate significant and consistent performance improvements across various recommenders, highlighting the effectiveness of our approach in advancing CRS performance. 

**Abstract (ZH)**: 基于对话的推荐系统通过多轮对话提高推荐质量，利用自然语言交互捕获用户的细微偏好。然而，这些系统常常面临假阴性问题，即用户可能喜欢的项目在训练过程中被错误地标记为负面反馈，导致推荐效果不佳。通过数据增强弥补标签集的不足是一种直观的解决方案，但面临着确保语义相关性和保留对话式推荐系统数据集中的协作信息这两方面之间的平衡挑战。为了解决这些问题，我们提出了一种新颖的数据增强框架，首先利用基于LLM的语义检索器识别多样且语义相关的目标项，然后通过相关性评分器去除噪声候选项。在此基础上，我们引入了一种两阶段训练策略，平衡语义相关性和协作信息。在两个基准数据集和用户模拟器上进行的广泛实验表明，我们的方法在各种推荐器上均能显著且一致地提高性能，突显了该方法在推动对话式推荐系统性能提升方面的有效性。 

---
# Lessons from A Large Language Model-based Outdoor Trail Recommendation Chatbot with Retrieval Augmented Generation 

**Title (ZH)**: 基于检索增强生成的大语言模型 Outdoor 路径推荐聊天机器人启示 

**Authors**: Julia Ann Mathew, Suining He  

**Link**: [PDF](https://arxiv.org/pdf/2508.05652)  

**Abstract**: The increasing popularity of outdoor recreational activities (such as hiking and biking) has boosted the demand for a conversational AI system to provide informative and personalized suggestion on outdoor trails. Challenges arise in response to (1) how to provide accurate outdoor trail information via conversational AI; and (2) how to enable usable and efficient recommendation services. To address above, this paper discusses the preliminary and practical lessons learned from developing Judy, an outdoor trail recommendation chatbot based on the large language model (LLM) with retrieval augmented generation (RAG). To gain concrete system insights, we have performed case studies with the outdoor trails in Connecticut (CT), US. We have conducted web-based data collection, outdoor trail data management, and LLM model performance studies on the RAG-based recommendation. Our experimental results have demonstrated the accuracy, effectiveness, and usability of Judy in recommending outdoor trails based on the LLM with RAG. 

**Abstract (ZH)**: 户外休闲活动（如远足和 biking）日益增长的流行性推动了对基于对话式AI系统的信息丰富且个性化的户外路线建议的需求。本研究针对如何通过对话式AI提供准确的户外路线信息以及如何实现可用且高效的推荐服务提出了挑战。为应对这些挑战，本文讨论了基于大型语言模型（LLM）和检索增强生成（RAG）开发户外路线推荐聊天机器人Judy的初步和实用经验教训。通过案例研究，我们在美国康涅狄格州（CT）的户外路线收集了网络数据并管理了路线数据，研究了基于RAG的推荐性能。我们的实验结果证明了Judy基于LLM与RAG推荐户外路线的准确性和有效性。 

---
# OmniBench-RAG: A Multi-Domain Evaluation Platform for Retrieval-Augmented Generation Tools 

**Title (ZH)**: OmniBench-RAG：检索增强生成工具的多领域评估平台 

**Authors**: Jiaxuan Liang, Shide Zhou, Kailong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.05650)  

**Abstract**: While Retrieval Augmented Generation (RAG) is now widely adopted to enhance LLMs, evaluating its true performance benefits in a reproducible and interpretable way remains a major hurdle. Existing methods often fall short: they lack domain coverage, employ coarse metrics that miss sub document precision, and fail to capture computational trade offs. Most critically, they provide no standardized framework for comparing RAG effectiveness across different models and domains.
We introduce OmniBench RAG, a novel automated platform for multi domain evaluation of RAG systems. The platform quantifies performance gains across accuracy and efficiency dimensions, spanning nine knowledge fields including culture, geography, and health. We introduce two standardized metrics: Improvements (accuracy gains) and Transformation (efficiency differences between pre RAG and post RAG models), enabling reproducible comparisons across models and tasks. The platform features dynamic test generation, modular evaluation pipelines, and automated knowledge base construction. Our evaluation reveals striking variability in RAG effectiveness, from significant gains in culture to declines in mathematics, highlighting the critical importance of systematic, domain aware assessment. A demonstration video is available at: this https URL. Code and datasets: this https URL. 

**Abstract (ZH)**: whilst Retrieval Augmented Generation (RAG) 的增强在大规模语言模型中已广泛采用，以评估其真正性能提升并在可重复和可解释的方式上仍然存在重大障碍，现有方法往往难以满足要求：它们缺乏领域覆盖面，使用粗略的指标忽视了子文档精确度，并未能捕捉到计算权衡。最重要的是，它们未能提供一个标准化的框架来跨不同模型和领域比较 RAG 的有效性。

我们引入了 OmniBench RAG，这是一种新型自动化多领域评估平台，用于评估 RAG 系统。该平台在准确性和效率维度上量化了性能增益，涵盖了包括文化、地理和健康在内的九个知识领域。我们引入了两个标准化指标：Improvements（准确度增益）和Transformation（预RAG模型与后RAG模型之间的效率差异），这使得不同模型和任务之间的可重复比较成为可能。该平台具备动态测试生成、模块化评估管道和自动化知识库构建功能。我们的评估揭示了 RAG 有效性在不同领域中的显著差异，从文化领域的显著增益到数学领域的下降，突显了系统化、领域意识评估的重要性。演示视频见：this https URL。代码和数据集见：this https URL。 

---
# AquiLLM: a RAG Tool for Capturing Tacit Knowledge in Research Groups 

**Title (ZH)**: AquiLLM：一个用于捕获研究团队隐性知识的检索增强工具 

**Authors**: Chandler Campbell, Bernie Boscoe, Tuan Do  

**Link**: [PDF](https://arxiv.org/pdf/2508.05648)  

**Abstract**: Research groups face persistent challenges in capturing, storing, and retrieving knowledge that is distributed across team members. Although structured data intended for analysis and publication is often well managed, much of a group's collective knowledge remains informal, fragmented, or undocumented--often passed down orally through meetings, mentoring, and day-to-day collaboration. This includes private resources such as emails, meeting notes, training materials, and ad hoc documentation. Together, these reflect the group's tacit knowledge--the informal, experience-based expertise that underlies much of their work. Accessing this knowledge can be difficult, requiring significant time and insider understanding. Retrieval-augmented generation (RAG) systems offer promising solutions by enabling users to query and generate responses grounded in relevant source material. However, most current RAG-LLM systems are oriented toward public documents and overlook the privacy concerns of internal research materials. We introduce AquiLLM (pronounced ah-quill-em), a lightweight, modular RAG system designed to meet the needs of research groups. AquiLLM supports varied document types and configurable privacy settings, enabling more effective access to both formal and informal knowledge within scholarly groups. 

**Abstract (ZH)**: 研究团队在捕获、存储和检索分布在成员之间的知识方面面临持久性挑战。尽管旨在分析和发布的结构化数据通常得到了良好的管理，但团队的许多集体知识仍以非正式、碎片化或未文档化的方式存在——这些知识往往通过会议、指导和日常合作以口头形式传递。这包括私人资源，如电子邮件、会议笔记、培训材料和临时文档。这些共同反映了团队的隐形知识——基于经验和非正式的专业知识，构成了他们工作的重要部分。访问这些知识往往需要耗费大量时间和内部理解。检索增强生成（RAG）系统通过使用户能够查询并生成基于相关来源材料的响应，提供了有前景的解决方案。然而，目前大多数RAG-LLM系统主要面向公共文件，并忽略了内部研究材料的隐私问题。我们介绍了一种轻量级模块化RAG系统AquiLLM（发音为ah-quill-em），旨在满足研究团队的需求。AquiLLM支持多种文档类型和可配置的隐私设置，使学术团队更有效地访问正式和非正式知识。 

---
# Automated Visualization Makeovers with LLMs 

**Title (ZH)**: 使用大语言模型自动优化可视化设计 

**Authors**: Siddharth Gangwar, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2508.05637)  

**Abstract**: Making a good graphic that accurately and efficiently conveys the desired message to the audience is both an art and a science, typically not taught in the data science curriculum. Visualisation makeovers are exercises where the community exchange feedback to improve charts and data visualizations. Can multi-modal large language models (LLMs) emulate this task? Given a plot in the form of an image file, or the code used to generate it, an LLM, primed with a list of visualization best practices, is employed to semi-automatically generate constructive criticism to produce a better plot. Our system is centred around prompt engineering of a pre-trained model, relying on a combination of userspecified guidelines and any latent knowledge of data visualization practices that might lie within an LLMs training corpus. Unlike other works, the focus is not on generating valid visualization scripts from raw data or prompts, but on educating the user how to improve their existing data visualizations according to an interpretation of best practices. A quantitative evaluation is performed to measure the sensitivity of the LLM agent to various plotting issues across different chart types. We make the tool available as a simple self-hosted applet with an accessible Web interface. 

**Abstract (ZH)**: 制作能够准确高效地传达所需信息的图表既是艺术也是科学，通常不在数据科学课程中教授。可视化改版是社区成员交流反馈以改进图表和数据分析可视化的工作。多模态大型语言模型（LLMs）能否模拟这一任务？给定一个图像文件形式的图表或生成它的代码，结合可视化最佳实践列表对大型语言模型进行预训练，该模型可以半自动地生成建设性批评以生成更好的图表。我们的系统围绕预训练模型的提示工程构建，依赖于用户指定的指导方针和大型语言模型训练语料库中可能存在的任何隐含的数据可视化实践知识。与其它研究不同，重点不是从原始数据或提示生成有效的可视化脚本，而是教育用户如何根据最佳实践的解释改进他们现有的数据可视化。我们进行了定量评估以衡量LLM代理对不同类型图表的各种绘图问题的敏感性。我们以一个简单的自助式小部件形式提供该工具，具有易于访问的网络界面。 

---
# AttriLens-Mol: Attribute Guided Reinforcement Learning for Molecular Property Prediction with Large Language Models 

**Title (ZH)**: AttriLens-Mol：基于属性的强化学习在大规模语言模型中预测分子性质 

**Authors**: Xuan Lin, Long Chen, Yile Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04748)  

**Abstract**: Large Language Models (LLMs) have shown promise in assisting molecular property prediction tasks but often rely on human-crafted prompts and chain-of-thought templates. While recent advanced large reasoning models like DeepSeek-R1 employ reinforcement learning for an extended ``thinking'' process, their reasoning can be verbose and lack relevance. We introduce AttriLens-Mol, an attribute-guided reinforcement learning framework for molecular property prediction with LLMs. AttriLens-Mol steers the model's reasoning by using: (1) a format reward encouraging attribute-based structured output, (2) a count reward to avoid enumerating irrelevant attributes, and (3) a rationality reward using advanced LLMs and RDKit to verify the relatedness of the generated attributes. This approach implicitly elicits the model's inherent knowledge of relevant molecular attributes during reasoning, enables making predictions for the molecular property more effectively. Experiments on both in-distribution and out-of-distribution datasets show that, training both 7B-size R1-Distilled-Qwen2.5 and R1-Distilled-LLaMA3.1 models on 4,000 samples with our proposed AttriLens-Mol method significantly boosts the performance, getting comparable or better results than supervised fine-tuning models (Mol-Instructions, ChemDFM, etc.) and advanced models (GPT-3.5, GPT-4o, DeepSeek-V3, DeepSeek-R1, etc.). Further, our extracted attributes for the target property, when used as features for an interpretable decision tree model, yield superior performance compared to attributes generated by prompting LLMs. This shows that AttriLens-Mol effectively elicits more relevant and predictive molecular attributes, leading to enhanced interpretability and performance for property prediction. We release the code in this https URL. 

**Abstract (ZH)**: 基于属性引导的强化学习框架AttribuLens-Mol用于分子性质预测 

---
