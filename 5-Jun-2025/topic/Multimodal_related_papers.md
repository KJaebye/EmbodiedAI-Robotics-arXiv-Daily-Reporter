# Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization 

**Title (ZH)**: 通过实体中心的多模态偏好优化减轻大型视觉-语言模型的幻觉问题 

**Authors**: Jiulong Wu, Zhengliang Shi, Shuaiqiang Wang, Jizhou Huang, Dawei Yin, Lingyong Yan, Min Cao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04039)  

**Abstract**: Large Visual Language Models (LVLMs) have demonstrated impressive capabilities across multiple tasks. However, their trustworthiness is often challenged by hallucinations, which can be attributed to the modality misalignment and the inherent hallucinations of their underlying Large Language Models (LLMs) backbone. Existing preference alignment methods focus on aligning model responses with human preferences while neglecting image-text modality alignment, resulting in over-reliance on LLMs and hallucinations. In this paper, we propose Entity-centric Multimodal Preference Optimization (EMPO), which achieves enhanced modality alignment than existing human preference alignment methods. Besides, to overcome the scarcity of high-quality multimodal preference data, we utilize open-source instruction datasets to automatically construct high-quality preference data across three aspects: image, instruction, and response. Experiments on two human preference datasets and five multimodal hallucination benchmarks demonstrate the effectiveness of EMPO, e.g., reducing hallucination rates by 85.9% on Object-HalBench and 49.8% on MM-HalBench. 

**Abstract (ZH)**: 大型多模态语言模型（LVLMs）在多个任务中展现了令人印象深刻的性能。然而，其可信度常常受到幻觉的挑战，这可以归因于模态错位以及其底层大型语言模型（LLMs）骨干的固有幻觉。现有偏好对齐方法侧重于将模型响应与人类偏好对齐，而忽视了图像-文本模态对齐，从而过度依赖LLMs和幻觉。本文提出了以实体为中心的多模态偏好优化（EMPO），实现了与现有基于人类偏好的对齐方法相比增强的模态对齐。此外，为了克服高质量多模态偏好数据的稀缺性，我们利用开源指令数据集自动构建涵盖图像、指令和响应三方面高质量偏好数据。在两个基于人类偏好的数据集和五个多模态幻觉基准上的实验表明，EMPO的有效性，例如，在Object-HalBench上将幻觉率降低85.9%、在MM-HalBench上降低49.8%。 

---
# Generating Pedagogically Meaningful Visuals for Math Word Problems: A New Benchmark and Analysis of Text-to-Image Models 

**Title (ZH)**: 为数学应用题生成具有教学意义的可视化内容：一个新的基准和文本到图像模型的分析 

**Authors**: Junling Wang, Anna Rutkiewicz, April Yi Wang, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03735)  

**Abstract**: Visuals are valuable tools for teaching math word problems (MWPs), helping young learners interpret textual descriptions into mathematical expressions before solving them. However, creating such visuals is labor-intensive and there is a lack of automated methods to support this process. In this paper, we present Math2Visual, an automatic framework for generating pedagogically meaningful visuals from MWP text descriptions. Math2Visual leverages a pre-defined visual language and a design space grounded in interviews with math teachers, to illustrate the core mathematical relationships in MWPs. Using Math2Visual, we construct an annotated dataset of 1,903 visuals and evaluate Text-to-Image (TTI) models for their ability to generate visuals that align with our design. We further fine-tune several TTI models with our dataset, demonstrating improvements in educational visual generation. Our work establishes a new benchmark for automated generation of pedagogically meaningful visuals and offers insights into key challenges in producing multimodal educational content, such as the misrepresentation of mathematical relationships and the omission of essential visual elements. 

**Abstract (ZH)**: 视觉工具是教授数学文字问题（MWPs）的有效工具，有助于年轻学生将文本描述转化为数学表达式以便解决。然而，创建这些视觉工具是劳动密集型的，并且缺乏自动化的支持方法。在这篇论文中，我们提出了Math2Visual，这是一种自动框架，用于从MWP文本描述生成具有教育意义的视觉工具。Math2Visual 利用预先定义的视觉语言和基于数学教师访谈的设计空间，以说明MWPs中的核心数学关系。使用Math2Visual，我们构建了一个包含1,903个注释视觉的数据库，并评估了文本到图像（TTI）模型生成符合我们设计的视觉的能力。我们进一步使用该数据库对几种TTI模型进行了微调，展示了在教育视觉生成方面的改进。我们的工作为自动化生成具有教育意义的视觉工具建立了新的基准，并提供了关于多模态教育内容生成的关键挑战的见解，如数学关系的误表征和关键视觉元素的缺失。 

---
# FLEX: A Large-Scale Multi-Modal Multi-Action Dataset for Fitness Action Quality Assessment 

**Title (ZH)**: FLEX：一个大规模多模态多动作数据集，用于健身动作质量评估 

**Authors**: Hao Yin, Lijun Gu, Paritosh Parmar, Lin Xu, Tianxiao Guo, Weiwei Fu, Yang Zhang, Tianyou Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03198)  

**Abstract**: With the increasing awareness of health and the growing desire for aesthetic physique, fitness has become a prevailing trend. However, the potential risks associated with fitness training, especially with weight-loaded fitness actions, cannot be overlooked. Action Quality Assessment (AQA), a technology that quantifies the quality of human action and provides feedback, holds the potential to assist fitness enthusiasts of varying skill levels in achieving better training outcomes. Nevertheless, current AQA methodologies and datasets are limited to single-view competitive sports scenarios and RGB modality and lack professional assessment and guidance of fitness actions. To address this gap, we propose the FLEX dataset, the first multi-modal, multi-action, large-scale dataset that incorporates surface electromyography (sEMG) signals into AQA. FLEX utilizes high-precision MoCap to collect 20 different weight-loaded actions performed by 38 subjects across 3 different skill levels for 10 repetitions each, containing 5 different views of the RGB video, 3D pose, sEMG, and physiological information. Additionally, FLEX incorporates knowledge graphs into AQA, constructing annotation rules in the form of penalty functions that map weight-loaded actions, action keysteps, error types, and feedback. We conducted various baseline methodologies on FLEX, demonstrating that multimodal data, multiview data, and fine-grained annotations significantly enhance model performance. FLEX not only advances AQA methodologies and datasets towards multi-modal and multi-action scenarios but also fosters the integration of artificial intelligence within the fitness domain. Dataset and code are available at this https URL. 

**Abstract (ZH)**: 随着健康意识的增强和对优美体态的日益追求，健身已成为一种流行趋势。然而，与健身训练相关联的潜在风险，尤其是与负重健身动作相关的风险，不容忽视。动作质量评估（AQA）技术通过量化人类动作的质量并提供反馈，有潜力辅助各技能等级的健身爱好者实现更好的训练效果。然而，当前的AQA方法和数据集局限于单视角竞技运动场景和RGB模态，并缺乏专业对健身动作的评估和指导。为填补这一空白，我们提出了FLEX数据集，这是首个将表面肌电图（sEMG）信号融入AQA的多模态、多动作大规模数据集。FLEX利用高精度动捕技术收集了38名受试者在三个不同技能等级下进行的20种不同负重动作，每种动作重复10次，包含RGB视频的5种视角、3D姿态、sEMG和生理信息。此外，FLEX还将知识图谱引入AQA中，构建了以惩罚函数形式表示的标注规则，将负重动作、动作关键步、错误类型和反馈映射起来。在FLEX上进行了多种基准方法测试，证明多模态数据、多视角数据和细粒度标注显著提升了模型性能。FLEX不仅推动了AQA方法和数据集向多模态、多动作场景的发展，还促进了人工智能在健身领域的应用。数据集和代码可在以下网址获取。 

---
# Multimodal Generative AI with Autoregressive LLMs for Human Motion Understanding and Generation: A Way Forward 

**Title (ZH)**: 基于自回归大语言模型的多模态生成人工智能：对人体运动理解与生成的一种新途径 

**Authors**: Muhammad Islam, Tao Huang, Euijoon Ahn, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.03191)  

**Abstract**: This paper presents an in-depth survey on the use of multimodal Generative Artificial Intelligence (GenAI) and autoregressive Large Language Models (LLMs) for human motion understanding and generation, offering insights into emerging methods, architectures, and their potential to advance realistic and versatile motion synthesis. Focusing exclusively on text and motion modalities, this research investigates how textual descriptions can guide the generation of complex, human-like motion sequences. The paper explores various generative approaches, including autoregressive models, diffusion models, Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and transformer-based models, by analyzing their strengths and limitations in terms of motion quality, computational efficiency, and adaptability. It highlights recent advances in text-conditioned motion generation, where textual inputs are used to control and refine motion outputs with greater precision. The integration of LLMs further enhances these models by enabling semantic alignment between instructions and motion, improving coherence and contextual relevance. This systematic survey underscores the transformative potential of text-to-motion GenAI and LLM architectures in applications such as healthcare, humanoids, gaming, animation, and assistive technologies, while addressing ongoing challenges in generating efficient and realistic human motion. 

**Abstract (ZH)**: 本文深入调研了多模态生成人工智能（GenAI）和自回归大型语言模型（LLMs）在人类动作理解和生成中的应用，提供了关于新兴方法、架构及其对逼真且多功能动作合成前景的见解。专注于文本和动作模态，本研究探讨了文本描述如何指导复杂人类样动作序列的生成。论文探讨了各种生成方法，包括自回归模型、扩散模型、生成对抗网络（GANs）、变分自编码器（VAEs）和基于转换器的模型，通过分析它们在动作质量、计算效率和适应性方面的优势和局限性。文章强调了文本条件动作生成的最新进展，其中文本输入被用于更精确地控制和细化动作输出。结合LLMs进一步增强了这些模型，使其能够实现指令与动作之间的语义对齐，提高一致性和上下文相关性。本系统调研突出了从文本到动作的GenAI和LLM架构在医疗保健、类人机器人、游戏、动画和辅助技术等领域中的变革潜力，同时应对生成高效且逼真人类动作的持续挑战。 

---
# Continual Learning in Vision-Language Models via Aligned Model Merging 

**Title (ZH)**: 视觉-语言模型中的持续学习通过对齐模型合并 

**Authors**: Ghada Sokar, Gintare Karolina Dziugaite, Anurag Arnab, Ahmet Iscen, Pablo Samuel Castro, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2506.03189)  

**Abstract**: Continual learning is conventionally tackled through sequential fine-tuning, a process that, while enabling adaptation, inherently favors plasticity over the stability needed to retain prior knowledge. While existing approaches attempt to mitigate catastrophic forgetting, a bias towards recent tasks persists as they build upon this sequential nature. In this work we present a new perspective based on model merging to maintain stability while still retaining plasticity. Rather than just sequentially updating the model weights, we propose merging newly trained task parameters with previously learned ones, promoting a better balance. To maximize the effectiveness of the merging process, we propose a simple mechanism that promotes learning aligned weights with previous ones, thereby avoiding interference when merging. We evaluate this approach on large Vision-Language Models (VLMs), and demonstrate its effectiveness in reducing forgetting, increasing robustness to various task orders and similarities, and improving generalization. 

**Abstract (ZH)**: 持续学习通常通过序列微调来处理，这一过程虽然能够实现适应，但本质上偏向于促进稳定性而牺牲保留先验知识所需的稳定性。现有的方法试图缓解灾难性遗忘，但它们建立在序列性的基础上，对最近任务的偏见仍然存在。在本文中，我们提出了一个新的基于模型合并的视角，旨在在保留可塑性的同时维持稳定性。我们不是仅仅顺序更新模型权重，而是提出将新训练的任务参数与之前学习的参数合并，以促进更好的平衡。为了最大化合并过程的有效性，我们提出了一种简单的机制来促进学习与之前参数对齐的权重，从而在合并时避免相互干扰。我们在大规模视觉-语言模型（VLMs）上评估了这种方法，并展示了其在减少遗忘、提高对各种任务顺序和相似性的鲁棒性以及提升泛化性能方面的有效性。 

---
# Multimodal Foundation Model for Cross-Modal Retrieval and Activity Recognition Tasks 

**Title (ZH)**: 多模态基础模型在跨模态检索与活动识别任务中的应用 

**Authors**: Koki Matsuishi, Kosuke Ukita, Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2506.03174)  

**Abstract**: In recent years, the widespread adoption of wearable devices has highlighted the growing importance of behavior analysis using IMU. While applications span diverse fields such as healthcare and robotics, recent studies have increasingly focused on multimodal analysis, in addition to unimodal analysis. Several studies have proposed multimodal foundation models that incorporate first-person video and text data; however, these models still fall short in providing a detailed analysis of full-body human activity. To address this limitation, we propose Activity Understanding and Representations Alignment - Multimodal Foundation Model (AURA-MFM), a foundational model integrating four modalities: third-person video, motion capture, IMU, and text. By incorporating third-person video and motion capture data, the model enables a detailed and multidimensional understanding of human activity, which first-person perspectives alone fail to capture. Additionally, a Transformer-based IMU encoder is employed to enhance the model's overall performance. Experimental evaluations on retrieval and activity recognition tasks demonstrate that our model surpasses existing methods. Notably, in the zero-shot classification for action recognition, our method achieved significantly higher performance, with an F1-score of 0.6226 and an accuracy of 0.7320, whereas the existing method recorded an F1-score of 0.0747 and an accuracy of 0.1961. 

**Abstract (ZH)**: 近年来，可穿戴设备的广泛应用凸显了使用IMU进行行为分析的重要性日益增强。尽管应用程序涵盖了医疗保健和机器人技术等多个领域，但最近的研究越来越多地专注于多模态分析，而不仅仅是单模态分析。已有研究提出了结合第一人称视频和文本数据的多模态基础模型，但这些模型仍然无法提供对全身人类活动的详细分析。为解决这一限制，我们提出一种整合四种模态的多模态基础模型——Activity Understanding and Representations Alignment - Multimodal Foundation Model (AURA-MFM)，该模型结合了第三人称视频、动作捕捉、IMU和文本数据。通过结合第三人称视频和动作捕捉数据，该模型能够提供对人类活动的详细和多维度理解，这是单纯的第一人称视角无法捕捉到的。此外，我们采用了基于Transformer的IMU编码器提升模型的整体性能。在检索和行为识别任务上的实验评估表明，我们的模型超越了现有方法。特别是在动作识别的零样本分类任务中，我们的方法取得了显著更高的性能，F1分数为0.6226，准确率为0.7320，而现有方法的F1分数为0.0747，准确率为0.1961。 

---
# Fusing Cross-Domain Knowledge from Multimodal Data to Solve Problems in the Physical World 

**Title (ZH)**: 融合多模态数据中的跨域知识以解决物理世界中的问题 

**Authors**: Yu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03155)  

**Abstract**: The proliferation of artificial intelligence has enabled a diversity of applications that bridge the gap between digital and physical worlds. As physical environments are too complex to model through a single information acquisition approach, it is crucial to fuse multimodal data generated by different sources, such as sensors, devices, systems, and people, to solve a problem in the real world. Unfortunately, it is neither applicable nor sustainable to deploy new resources to collect original data from scratch for every problem. Thus, when data is inadequate in the domain of problem, it is vital to fuse knowledge from multimodal data that is already available in other domains. We call this cross-domain knowledge fusion. Existing research focus on fusing multimodal data in a single domain, supposing the knowledge from different datasets is intrinsically aligned; however, this assumption may not hold in the scenarios of cross-domain knowledge fusion. In this paper, we formally define the cross-domain multimodal data fusion problem, discussing its unique challenges, differences and advantages beyond data fusion in a single domain. We propose a four-layer framework, consisting of Domains, Links, Models and Data layers, answering three key questions: "what to fuse", "why can be fused", and "how to fuse". The Domains Layer selects relevant data from different domains for a given problem. The Links Layer reveals the philosophy of knowledge alignment beyond specific model structures. The Models Layer provides two knowledge fusion paradigms based on the fundamental mechanisms for processing data. The Data Layer turns data of different structures, resolutions, scales and distributions into a consistent representation that can be fed into an AI model. With this framework, we can design end-to-end solutions that fuse cross-domain multimodal data effectively for solving real-world problems. 

**Abstract (ZH)**: 人工智能的普及使得数字世界与物理世界之间的多种应用成为可能。由于物理环境过于复杂，单靠一种信息获取方法无法进行建模，因此，将来自不同来源（如传感器、设备、系统和人员）的多模态数据进行融合以解决现实世界的问题至关重要。不幸的是，为每个问题从头开始收集原始数据并部署新资源是不可行且不可持续的。因此，在问题领域数据不足的情况下，融合其他领域已有的多模态数据中的知识变得至关重要。我们称之为跨域知识融合。现有研究专注于单一领域内多模态数据的融合，假设不同数据集的知识是固有对齐的；但在跨域知识融合场景中，这一假设可能不成立。在本文中，我们正式定义了跨域多模态数据融合问题，讨论其独特的挑战、差异及其在单一领域数据融合之外的优势。我们提出了一种四层框架，包括领域层、链接层、模型层和数据层，回答了三个关键问题：“融合什么”、“为何能够融合”和“如何融合”。领域层从不同领域筛选与给定问题相关联的数据。链接层揭示了超越特定模型结构的知识对齐理念。模型层基于处理数据的基本机制提供两种知识融合范式。数据层将不同结构、分辨率、比例和分布的数据转换为可用于AI模型的一致表示。通过该框架，我们可以设计端到端的解决方案，有效融合跨域多模态数据以解决实际问题。 

---
